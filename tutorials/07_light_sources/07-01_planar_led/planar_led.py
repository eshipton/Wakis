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
# # Tutorial 7.1 · Planar deep-UV LED: escape cone, total internal reflection, TE vs TM dipoles
#
# Most photons generated inside a planar, high-index semiconductor light-emitting
# diode (LED) never escape into free space. Two physical mechanisms conspire
# against light extraction in planar III-nitride ultraviolet (UV) LEDs:
#
# 1. **Total Internal Reflection (TIR) at the semiconductor/air interface**:
#    In a semiconductor of refractive index $n \approx 2.6\text{--}2.9$, Snell's
#    law $n \sin\theta_1 = n_{\text{air}} \sin\theta_{\text{air}}$ restricts escape
#    into air ($n_{\text{air}} = 1.0$) to rays within a narrow **escape cone** of
#    half-angle
#    $$\theta_c = \arcsin\!\left(\frac{1}{n}\right) \approx 22.6^\circ \quad (n = 2.60).$$
#    Rays emitted at $|\theta_1| > \theta_c$ undergo 100% TIR and become trapped
#    as guided slab modes or leak into the substrate.
# 2. **Anisotropic Quantum-Well Dipole Orientation (TE vs. TM) and Contact Absorption**:
#    In $\text{Al}_x\text{Ga}_{1-x}\text{N}$ multiple quantum wells (MQWs), crystal-field
#    splitting shifts the valence-band maximum from heavy-hole/light-hole ($\Gamma_9, \Gamma_7^+$)
#    toward the crystal-field split-off hole ($\Gamma_7^-$) as the Al mole fraction $x$
#    increases toward deep-UV wavelengths ($\lambda_0 \lesssim 280\;\text{nm}$):
#    - **TE-polarized emission ($\mathbf{E} \perp c$-axis, in-plane dipoles $J_x, J_y$)**:
#      radiates preferentially along the vertical growth axis ($\theta_1 = 0$, directly
#      into the escape cone) with angular pattern $p_{\text{TE}}(\theta_1) \propto \cos^2\theta_1$
#      in the p-polarized plane.
#    - **TM-polarized emission ($\mathbf{E} \parallel c$-axis, vertical dipole $J_z$)**:
#      has a radiation null along the dipole axis ($\theta_1 = 0$) and radiates with
#      $p_{\text{TM}}(\theta_1) \propto \sin^2\theta_1$, sending almost all power into
#      large angles $|\theta_1| > \theta_c$ where it is trapped by TIR.
#    - **p-GaN Contact Layer Absorption**: Because wide-bandgap $\text{p-Al}_x\text{Ga}_{1-x}\text{N}$
#      has poor hole conductivity, planar UV LEDs typically cap the p-AlGaN cladding
#      with a lower-bandgap **p-GaN** contact layer ($n_{\text{GaN}} = 2.90$), which is
#      strongly absorbing at $\lambda_0 = 280\;\text{nm}$ ($\alpha_{\text{GaN}} = 170{,}000\;\text{cm}^{-1}$,
#      penetration depth $\delta_p = 1/\alpha_{\text{GaN}} \approx 58.8\;\text{nm}$).
#
# Chapter 7 builds on [Tutorial 4.3 · Dipole near a mirror: image theory, Drexhage effect and Purcell factor](../../04_radiation_and_antennas/04-03_dipole_near_mirror/) as the basic emitter model. In this tutorial we build a complete 2D ($x$-$z$) Finite Integration Technique (FIT)
# model in **Wakis** following the planar deep-UV AlGaN LED benchmark of
# **H.-Y. Ryu, *Nanoscale Res. Lett.* 9, 58 (2014)** ([doi:10.1186/1556-276X-9-58](https://doi.org/10.1186/1556-276X-9-58)):
#
# 1. **Analytic 2D vs. 3D Escape-Cone Baseline**: We derive the exact 2D single-face
#    escape fraction $\eta_{\text{2D, cone}} = \theta_c / \pi = \arcsin(1/n)/\pi$ and
#    compare it with the 3D solid-angle formula $\eta_{\text{3D, cone}} = \frac{1}{2}(1 - \cos\theta_c) \approx \frac{1}{4n^2}$.
#    We include exact Fresnel power transmittance $T_s(\theta_1), T_p(\theta_1)$ and
#    dipole angular weights for all three dipole orientations ($J_y, J_x, J_z$) and
#    validate Wakis 2D simulations of a semi-infinite dielectric slab ($n \in \{1.5, 2.0, 2.6\}$)
#    directly against the closed-form integrals.
# 2. **Ryu (2014) Planar AlGaN UV LED Stack**: We model the full 6-region stratified stack
#    (Sapphire / $\text{n-Al}_{0.6}\text{GaN}$ / $\text{AlGaN}$ MQW / $\text{p-Al}_{0.6}\text{GaN}$ /
#    absorbing $\text{p-GaN}$ / Air) at $\lambda_0 = 280\;\text{nm}$, sweeping
#    $t_{\text{p-GaN}} \in [0, 200]\;\text{nm}$ and $n_{\text{AlGaN}} \in [2.50, 2.70]$
#    to reproduce the paper's Fig. 3 and Fig. 7 trends, complete with a 4-channel
#    Poynting flux budget (top air extraction, bottom sapphire leakage, lateral
#    guided-mode trapping, and active/contact absorption).
# 3. **Incoherent Spontaneous Emission & Far-Field Radiation (`Near2FarMonitor`)**:
#    We demonstrate why spontaneous emission from an extended MQW active layer must
#    be modelled by **averaging powers** across dipole positions and polarizations
#    rather than summing fields coherently, and we project the extracted near fields
#    to the far zone via `wakis.near2far.Near2FarMonitor` for comparison with a
#    Lambertian $\cos\theta$ emitter.
# 4. **Subpixel Smoothing (Kottke vs. Staircase) & Grid Convergence**: We compare
#    Kottke anisotropic subpixel interface smoothing against staircase snapping
#    across grid resolutions up to a fine reference grid.

# %% [markdown]
# ## 1. Physics & Analytic Derivation: 2D vs. 3D Escape Cones
#
# ### 1.1 Critical angle and the dimensionality gap ($\theta_c/\pi$ vs. $1/(4n^2)$)
#
# Consider a point dipole (in 3D) or line dipole (in 2D, invariant along $y$)
# embedded in a lossless semi-infinite dielectric slab of refractive index $n$
# occupying $z \le 0$, bounded by air ($n_{\text{air}} = 1$) in $z > 0$. A ray
# launched at polar angle $\theta_1 \in [0, \pi]$ from the $+z$ surface normal
# refracts into air at angle $\theta_{\text{air}}$ given by Snell's law:
#
# $$\sin\theta_{\text{air}} = n \sin\theta_1.$$
#
# Escape is possible only for upward rays with $\theta_1 < \theta_c = \arcsin(1/n)$.
#
# - **In 3D**, an isotropic radiator emits uniformly over the full sphere of solid
#   angle $\Omega_{\text{tot}} = 4\pi$. The differential solid angle of a polar ring
#   $(\theta_1, \theta_1 + d\theta_1)$ is $d\Omega = 2\pi \sin\theta_1\,d\theta_1$.
#   Neglecting Fresnel reflection, the fraction of total power escaping through the
#   top face is:
#   $$\eta_{\text{3D}}^{\text{no-Fresnel}}
#     = \frac{1}{4\pi} \int_0^{2\pi} d\phi \int_0^{\theta_c} \sin\theta_1\,d\theta_1
#     = \frac{1 - \cos\theta_c}{2}
#     = \frac{1 - \sqrt{1 - 1/n^2}}{2}
#     = \frac{1}{4n^2} + \frac{1}{16n^4} + \mathcal{O}(n^{-6}).$$
#   For AlGaN ($n = 2.60$, $\theta_c = 22.62^\circ$), $\eta_{\text{3D}}^{\text{no-Fresnel}} = 3.85\%$
#   (and the paraxial approximation $1/(4n^2)$ gives $3.70\%$). Why is it so small?
#   Because the Jacobian $\sin\theta_1$ vanishes at normal incidence ($\theta_1 = 0$),
#   putting very little solid angle near the surface normal!
#
# - **In 2D ($x$-$z$ cross-section)**, wavevectors lie in the $(x, z)$ plane and
#   directions are parameterized by a single polar angle $\theta_1 \in (-\pi, \pi]$
#   with uniform measure $d\theta_1$ out of $2\pi$ radians (no $\sin\theta_1$
#   azimuthal ring factor!). The top escape cone $\theta_1 \in [-\theta_c, +\theta_c]$
#   subtends plane angle $2\theta_c$, so an isotropic 2D radiator (out-of-plane
#   line current $J_y$) has single-face escape fraction:
#   $$\eta_{\text{2D}}^{\text{no-Fresnel}}
#     = \frac{1}{2\pi} \int_{-\theta_c}^{\theta_c} d\theta_1
#     = \frac{\theta_c}{\pi}
#     = \frac{\arcsin(1/n)}{\pi}
#     = \frac{1}{\pi n} + \frac{1}{6\pi n^3} + \mathcal{O}(n^{-5}).$$
#   For AlGaN ($n = 2.60$), $\eta_{\text{2D}}^{\text{no-Fresnel}} = 12.57\%$ — more than
#   **three times larger** than the 3D value ($3.85\%$), scaling as $\mathcal{O}(1/n)$
#   rather than $\mathcal{O}(1/n^2)$.
#
# ### 1.2 Exact 2D escape integrals with Fresnel reflection and dipole patterns
#
# In a 2D $(x, z)$ cross-section ($\partial/\partial y = 0$), Maxwell's equations
# decouple into two orthogonal polarizations with distinct dipole radiation patterns
# $p(\theta_1)$ (normalized so $\int_{-\pi}^{\pi} p(\theta_1)\,d\theta_1 = 1$) and
# Fresnel power transmittances $T_{s,p}(\theta_1) = 1 - |r_{s,p}(\theta_1)|^2$:
#
# 1. **Out-of-plane horizontal dipole $J_y$ (s-polarized / TE in $x$-$z$, fields $E_y, H_x, H_z$)**:
#    Radiates isotropically in the $x$-$z$ plane ($p_y(\theta_1) = \frac{1}{2\pi}$) with
#    s-polarized Fresnel transmittance:
#    $$\eta_{\text{2D}, y}^{\text{Fresnel}} = \frac{1}{2\pi} \int_{-\theta_c}^{\theta_c} T_s(\theta_1)\,d\theta_1,
#      \qquad T_s(\theta_1) = \frac{4 n \cos\theta_1 \cos\theta_{\text{air}}}{(n\cos\theta_1 + \cos\theta_{\text{air}})^2}.$$
# 2. **In-plane horizontal dipole $J_x$ (p-polarized in $x$-$z$, "TE mode" $\mathbf{E} \perp c$-axis, fields $E_x, E_z, H_y$)**:
#    Radiates with $p_x(\theta_1) = \frac{1}{\pi}\cos^2\theta_1$, peaking at $\theta_1 = 0$
#    right inside the escape cone, with p-polarized Fresnel transmittance:
#    $$\eta_{\text{2D}, x}^{\text{Fresnel}} = \frac{1}{\pi} \int_{-\theta_c}^{\theta_c} T_p(\theta_1) \cos^2\theta_1\,d\theta_1,
#      \qquad T_p(\theta_1) = \frac{4 n \cos\theta_1 \cos\theta_{\text{air}}}{(\cos\theta_1 + n\cos\theta_{\text{air}})^2}.$$
# 3. **Out-of-plane vertical dipole $J_z$ (p-polarized in $x$-$z$, "TM mode" $\mathbf{E} \parallel c$-axis, fields $E_x, E_z, H_y$)**:
#    Radiates with $p_z(\theta_1) = \frac{1}{\pi}\sin^2\theta_1$, which vanishes at
#    $\theta_1 = 0$ and is bounded by $\sin^2\theta_c = 1/n^2$ inside the escape cone:
#    $$\eta_{\text{2D}, z}^{\text{Fresnel}} = \frac{1}{\pi} \int_{-\theta_c}^{\theta_c} T_p(\theta_1) \sin^2\theta_1\,d\theta_1.$$
#
# Notice why the TE/TM extraction ratio is so large in high-index semiconductors:
# inside $|\theta_1| \le \theta_c$, $\cos^2\theta_1 \approx 1$ whereas
# $\sin^2\theta_1 \le 1/n^2$, giving $\eta_x / \eta_z \sim 3 n^2 \approx 18\text{--}20$
# at $n = 2.60$!

# %%
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from wakis import figstyle, schematic

TUTORIAL_DIR = Path(__file__).resolve().parent
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

import _helpers as h  # noqa: E402

FIG_DIR = TUTORIAL_DIR / "figures"
DATA_DIR = TUTORIAL_DIR / "data"

# Verified layer & spectral parameters from H.-Y. Ryu, Nanoscale Res. Lett. 9, 58 (2014)
WAVELENGTH_NM = 280.0
WAVELENGTH_M = WAVELENGTH_NM * 1e-9
N_MAX_REF = 2.90  # p-GaN has the highest refractive index in the stack
PGAN_THICKNESSES_NM = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0])
ALGAN_INDICES = np.array([2.50, 2.60, 2.70])
SEMI_INF_INDICES = np.array([1.50, 2.00, 2.60])
DIPOLE_X_POSITIONS_UM = np.array([0.0, 0.20, 0.40, 0.60, 0.80])
DIPOLE_X_WEIGHTS = np.array([1.0, 2.0, 2.0, 2.0, 2.0])  # x=0 + symmetric +/-x_i pairs


# %% [markdown]
# ## 2. Study Configuration, Graded Meshing, & Resolution Table
#
# Every non-demo simulation in both `--full` and `--quick` modes enforces three
# strict FDTD/FIT resolution rules:
#
# 1. **Wavelength resolution ($\ge 10\;\text{cpw}$ in the highest-index material)**:
#    $$\Delta x_{\text{bg}} \le \frac{\lambda_0}{10\,n_{\max}} = \frac{280\;\text{nm}}{10 \times 2.90} = 9.66\;\text{nm}, \qquad \lambda_{\min} = \frac{\lambda_0}{n_{\max}} = 96.55\;\text{nm}.$$
#    We use $20\;\text{cpw}$ ($\Delta x_{\text{bg}} = 4.83\;\text{nm}$, lateral box $L_x = 3.6\;\mu\text{m}$) for the `--full` sweeps
#    (and $12\;\text{cpw}$, $\Delta x_{\text{bg}} = 8.05\;\text{nm}$, $L_x = 2.8\;\mu\text{m}$ in `--quick` mode).
# 2. **Geometric feature resolution ($\ge 4$ cells across the thinnest layer)**:
#    The thinnest layer in the Ryu (2014) sweep is the $25\;\text{nm}$ p-GaN contact
#    layer (followed by the $50\;\text{nm}$ MQW and $50\;\text{nm}$ p-AlGaN layers).
#    Using `wakis.templates.meshing.graded_axis`, we place exact $z$-mesh nodes on
#    every layer interface (`z_min`, $-550\;\text{nm}$, $-100\;\text{nm}$, $-75\;\text{nm}$,
#    $-50\;\text{nm}$, $0\;\text{nm}$, $t_{\text{p-GaN}}$, `z_max`) with fine vertical
#    spacing $\Delta z_{\min} = 3.125\;\text{nm}$ ($8$ cells across $25\;\text{nm}$ in `--full`;
#    $\Delta z_{\min} = 6.25\;\text{nm}$, $4$ cells across $25\;\text{nm}$ in `--quick`) and smooth
#    grading ($\le 1.35$) into the thick n-AlGaN, sapphire, and air regions.
# 3. **Absorber skin-depth resolution ($\ge 4$ cells per skin depth, $< 1\%$ convergence)**:
#    At $\lambda_0 = 280\;\text{nm}$, p-GaN has $\alpha = 170{,}000\;\text{cm}^{-1}$
#    ($\kappa = \alpha\lambda_0 / (4\pi) = 0.3788$). Using the power skin-depth definition
#    $$\delta = \frac{\lambda_0}{4\pi\kappa} = \frac{1}{\alpha} = 58.82\;\text{nm}$$
#    (where the $1/e$ field-amplitude skin depth is $\delta_E = 2\delta = 117.65\;\text{nm}$),
#    our `--full` graded meshes resolve $\delta$ with $14.12\;\text{cells}/\delta$ ($\Delta z_{\min} = 4.17\;\text{nm}$,
#    $14\;\text{cpw}$), $18.82\;\text{cells}/\delta$ ($\Delta z_{\min} = 3.125\;\text{nm}$, $20\;\text{cpw}$ **accepted level**,
#    $0.03\%\text{--}0.32\%$ TE relative difference and $0.53\%$ TM relative difference vs. reference across
#    $t_{\text{p-GaN}} \in \{25, 50, 100\}\;\text{nm}$), and $23.53\;\text{cells}/\delta$ ($\Delta z_{\min} = 2.50\;\text{nm}$,
#    $28\;\text{cpw}$ reference).
#    Task 3 retains a single clearly labelled under-resolved demonstration pair
#    (`is_demo=True`, $\Delta z_{\text{p-GaN}} = 25\;\text{nm}$, $1$ cell across $25\;\text{nm}$,
#    $2.35\;\text{cells}/\delta$) to illustrate the $20.8\%\text{--}65.7\%$ extraction error caused by
#    under-resolving the absorber skin depth.
#
# ### Setup Schematics & Thin-Layer Graded Mesh (`figures/00a_setup_geometry.png`, `figures/00b_setup_mesh_and_excitation.png`, `figures/00_setup_table.md`)
#
# The setup figures generated by `wakis.schematic.save_setup_figures` show the stratified 6-layer Ryu (2014) planar AlGaN deep-UV LED domain, 8-cell CPML boundaries, the 18-dipole MQW ensemble (9 symmetric lateral sites $\times$ TE $J_x$ and TM $J_z$ polarizations at $z = -75\;\text{nm}$), the top `Near2FarMonitor` and substrate/lateral `FluxMonitor`s, the graded thin-layer mesh ($\Delta z_{\min} = 3.125\;\text{nm}$), and the $\lambda_0 = 280\;\text{nm}$ dipole excitation spectrum:
#
# ![Planar AlGaN deep-UV LED geometry, dipole ensemble, and monitors](figures/00a_setup_geometry.png)
# ![Planar AlGaN deep-UV LED graded thin-layer mesh and UV dipole excitation](figures/00b_setup_mesh_and_excitation.png)
#
# The table below summarizes the domain, discretization, boundary conditions, epitaxial layers, dipole ensemble, monitors, and task variations (`figures/00_setup_table.md`):
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 3.60 µm × 1 µm × 1.25 µm (z ∈ [-850 nm, 400 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 746 × 1 × 280 (208880 cells) |
# | Discretization | Δx & resolution | Δ_min = 3.12 nm, 30.9 cells/λ (in n_max = 2.90), 8.0 cells across feature (25 nm) |
# | Boundary | -x face | CPML (8 cells, 38.6 nm) |
# | Boundary | +x face | CPML (8 cells, 38.6 nm) |
# | Boundary | -y face | Periodic |
# | Boundary | +y face | Periodic |
# | Boundary | -z face | CPML (8 cells, 38.1 nm) |
# | Boundary | +z face | CPML (8 cells, 38.0 nm) |
# | Material | Background | Air superstrate, ε = 1.00 (isotropic) |
# | Material | Region 1: Sapphire (d_z = 300 nm, n = 1.80) | ε_r = 3.24, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.80 µm, 1.80 µm] (d = 3.60 µm) |
# | Material | Region 2: n-AlGaN (d_z = 450 nm, n = 2.60) | ε_r = 6.76, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.80 µm, 1.80 µm] (d = 3.60 µm) |
# | Material | Region 3: MQW active (d_z = 50 nm, n = 2.60, α = 10³ cm⁻¹) | ε_r = 6.76, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.80 µm, 1.80 µm] (d = 3.60 µm) |
# | Material | Region 4: p-AlGaN (d_z = 50 nm, n = 2.60) | ε_r = 6.76, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.80 µm, 1.80 µm] (d = 3.60 µm) |
# | Material | Region 5: p-GaN absorber (d_z = 50 nm, n = 2.90, α = 1.7×10⁵ cm⁻¹) | ε_r = 8.41, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.80 µm, 1.80 µm] (d = 3.60 µm) |
# | Source | 1 · CurrentSource | pos: z = -75 nm, 9 sites in x ∈ [-1.05, 1.05] µm (5 shown), pol: TE (J_x) & TM (J_z), pulse: f₀ = 1070.7 THz, Δf = 192.7 THz |
# | Monitor | 2 · Near2FarMonitor | Top Near2FarMonitor in air (z = +80 nm), pos: z = +80 nm, x ∈ [-1.75, 1.75] µm, freqs: f = 1070.7 THz |
# | Monitor | 3 · FluxMonitor | Bottom + lateral Poynting FluxMonitors (z_bot = -0.80 µm), pos: z = -802 nm, freqs: f = 1070.7 THz |
# | Variation | Task 0 — Semi-infinite dielectric slab validation | notes=n_slab ∈ {1.50, 2.00, 2.60} → Air (n = 1.00), z_dip = -180 nm, 4-face FluxMonitor box vs exact 2D Fresnel integral |
# | Variation | Task 1 — p-GaN thickness & AlGaN index sweeps (Ryu 2014 Figs. 3 & 7) | notes=t_p-GaN ∈ {0, 25, 50, 100, 150, 200} nm (α = 1.7×10⁵ cm⁻¹, δ = 58.82 nm), n_AlGaN ∈ {2.50, 2.60, 2.70} |
# | Variation | Task 2 — Incoherent dipole ensemble & Near2Far pattern | notes=N_pos = 5 lateral positions x_i ∈ {0.0, 0.25, 0.50, 0.75, 1.05} µm (9 symmetric sites) × TE (J_x) / TM (J_z) = 18 dipoles |
# | Variation | Tasks 3–6 — Kottke subpixel smoothing vs staircase convergence ladder | notes=cpw ∈ {10 (demo, Δz = 25 nm), 14 (Δz = 4.17 nm), 20 (accepted, Δz = 3.125 nm, 18.8 cells/δ), 28 (reference, Δz = 2.50 nm, 23.5 cells/δ)} |
#
# ### Task Resolution Table (`--full` & `--quick` Measured)
#
# | Mode | Task | Label | $\text{cpw}$ | $\Delta x_{\text{bg}}$ [nm] | $\Delta z_{\min}$ [nm] | Min Feature [nm] | Cells / Feature | Skin Depth $\delta$ [nm] | Cells / $\delta$ | $N_{\text{cells}}$ | $N_{\text{steps}}$ | Wall Time [s] |
# | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | `--full` | 0 | `semi_infinite_validation_cpw20` | $20$ | $5.37$ | $5.37$ | $180.0$ | $33$ | $\infty$ | $\infty$ | $100{,}500$ | $3{,}066$ | $128.8\;\text{s}$ |
# | `--full` | 1 | `ryu_stack_pgan_and_index_sweep_cpw20` | $20$ | $4.83$ | $3.12$ | $25.0$ | $8$ | $58.82$ | $18.82$ | $206{,}642$ | $5{,}501$ | $3278.9\;\text{s}$ |
# | `--full` | 2 | `incoherent_dipole_positions_cpw20` | $20$ | $4.83$ | $3.12$ | $50.0$ | $16$ | $58.82$ | $18.82$ | $208{,}880$ | $5{,}451$ | $1713.9\;\text{s}$ |
# | `--full` | 3 | `demo_underresolved_skin_depth_cpw10` *(demo)* | $10$ | $9.63$ | $9.63$ | $25.0$ | $1$ | $58.82$ | $2.35$ | $18{,}700$ | $2{,}286$ | $46.1\;\text{s}$ |
# | `--full` | 4 | `kottke_vs_staircase_cpw14` | $14$ | $6.90$ | $4.17$ | $25.0$ | $6$ | $58.82$ | $14.12$ | $102{,}834$ | $4{,}201$ | $390.8\;\text{s}$ |
# | `--full` | 5 | `kottke_vs_staircase_accepted_cpw20` | $20$ | $4.83$ | $3.12$ | $25.0$ | $8$ | $58.82$ | $18.82$ | $206{,}642$ | $5{,}501$ | $960.9\;\text{s}$ |
# | `--full` | 6 | `kottke_reference_cpw28` | $28$ | $3.45$ | $2.50$ | $25.0$ | $10$ | $58.82$ | $23.53$ | $396{,}720$ | $6{,}951$ | $1085.5\;\text{s}$ |
# | `--quick` | 0 | `semi_infinite_validation_cpw12` | $12$ | $8.97$ | $8.97$ | $180.0$ | $20$ | $\infty$ | $\infty$ | $28{,}080$ | $1{,}836$ | $29.8\;\text{s}$ |
# | `--quick` | 1 | `ryu_stack_pgan_and_index_sweep_cpw12` | $12$ | $8.05$ | $6.25$ | $25.0$ | $4$ | $58.82$ | $9.41$ | $56{,}724$ | $2{,}651$ | $1123.4\;\text{s}$ |
# | `--quick` | 2 | `incoherent_dipole_positions_cpw12` | $12$ | $8.05$ | $6.25$ | $50.0$ | $8$ | $58.82$ | $9.41$ | $57{,}072$ | $2{,}601$ | $741.0\;\text{s}$ |
# | `--quick` | 3 | `demo_underresolved_skin_depth_cpw10` *(demo)* | $10$ | $9.66$ | $9.66$ | $25.0$ | $1$ | $58.82$ | $2.35$ | $14{,}500$ | $2{,}251$ | $140.9\;\text{s}$ |
# | `--quick` | 4 | `kottke_vs_staircase_cpw10` | $10$ | $9.66$ | $6.25$ | $25.0$ | $4$ | $58.82$ | $9.41$ | $40{,}600$ | $2{,}451$ | $336.2\;\text{s}$ |
# | `--quick` | 5 | `kottke_vs_staircase_accepted_cpw14` | $14$ | $6.90$ | $4.17$ | $25.0$ | $6$ | $58.82$ | $14.12$ | $79{,}982$ | $3{,}501$ | $678.9\;\text{s}$ |
# | `--quick` | 6 | `kottke_reference_cpw20` | $20$ | $4.83$ | $3.12$ | $25.0$ | $8$ | $58.82$ | $18.82$ | $160{,}660$ | $4{,}551$ | $666.4\;\text{s}$ |


# %%
def _mode_config(mode: str) -> dict:
    if mode == "full":
        return {
            "cpw": 20.0,
            "dz_fine_nm": 3.125,
            "cpw_semi": 20.0,
            "Lx": 3.6e-6,
            "conv_levels": (
                {
                    "cpw": 10.0,
                    "dz_fine_nm": 25.0,
                    "min_cells": 1,
                    "is_demo": True,
                    "tag": "demo_cpw10",
                    "label": "demo_underresolved_skin_depth_cpw10",
                },
                {
                    "cpw": 14.0,
                    "dz_fine_nm": 25.0 / 6.0,
                    "min_cells": 4,
                    "is_demo": False,
                    "tag": "cpw14",
                    "label": "kottke_vs_staircase_cpw14",
                },
                {
                    "cpw": 20.0,
                    "dz_fine_nm": 3.125,
                    "min_cells": 4,
                    "is_demo": False,
                    "tag": "cpw20",
                    "label": "kottke_vs_staircase_accepted_cpw20",
                },
            ),
            "ref_level": {
                "cpw": 28.0,
                "dz_fine_nm": 2.50,
                "min_cells": 4,
                "is_demo": False,
                "tag": "cpw28",
                "label": "kottke_reference_cpw28",
            },
        }
    return {
        "cpw": 12.0,
        "dz_fine_nm": 6.25,
        "cpw_semi": 12.0,
        "Lx": 2.8e-6,
        "conv_levels": (
            {
                "cpw": 10.0,
                "dz_fine_nm": 25.0,
                "min_cells": 1,
                "is_demo": True,
                "tag": "demo_cpw10",
                "label": "demo_underresolved_skin_depth_cpw10",
            },
            {
                "cpw": 10.0,
                "dz_fine_nm": 6.25,
                "min_cells": 4,
                "is_demo": False,
                "tag": "cpw10",
                "label": "kottke_vs_staircase_cpw10",
            },
            {
                "cpw": 14.0,
                "dz_fine_nm": 25.0 / 6.0,
                "min_cells": 4,
                "is_demo": False,
                "tag": "cpw14",
                "label": "kottke_vs_staircase_accepted_cpw14",
            },
        ),
        "ref_level": {
            "cpw": 20.0,
            "dz_fine_nm": 3.125,
            "min_cells": 4,
            "is_demo": False,
            "tag": "cpw20",
            "label": "kottke_reference_cpw20",
        },
    }


def _task_table(mode: str) -> list[dict]:
    cfg = _mode_config(mode)
    tasks = [
        {
            "index": 0,
            "kind": "semi_infinite",
            "cpw": float(cfg["cpw_semi"]),
            "dz_fine_nm": None,
            "min_cells_per_layer": 4,
            "Lx": float(cfg["Lx"]),
            "is_demo": False,
            "file": "task_00_semi_infinite.npz",
            "label": f"semi_infinite_validation_cpw{int(cfg['cpw_semi'])}",
        },
        {
            "index": 1,
            "kind": "ryu_stack_sweep",
            "cpw": float(cfg["cpw"]),
            "dz_fine_nm": float(cfg["dz_fine_nm"]),
            "min_cells_per_layer": 4,
            "Lx": float(cfg["Lx"]),
            "is_demo": False,
            "file": "task_01_ryu_stack_sweep.npz",
            "label": f"ryu_stack_pgan_and_index_sweep_cpw{int(cfg['cpw'])}",
        },
        {
            "index": 2,
            "kind": "incoherent_positions",
            "cpw": float(cfg["cpw"]),
            "dz_fine_nm": float(cfg["dz_fine_nm"]),
            "min_cells_per_layer": 4,
            "Lx": float(cfg["Lx"]),
            "is_demo": False,
            "file": "task_02_incoherent_positions.npz",
            "label": f"incoherent_dipole_positions_cpw{int(cfg['cpw'])}",
        },
    ]
    for i, lv in enumerate(cfg["conv_levels"]):
        tasks.append(
            {
                "index": 3 + i,
                "kind": "convergence_pair",
                "cpw": float(lv["cpw"]),
                "dz_fine_nm": float(lv["dz_fine_nm"]),
                "min_cells_per_layer": int(lv["min_cells"]),
                "Lx": float(cfg["Lx"]),
                "is_demo": bool(lv["is_demo"]),
                "file": f"task_{3 + i:02d}_conv_{lv['tag']}.npz",
                "label": str(lv["label"]),
            }
        )
    ref_lv = cfg["ref_level"]
    tasks.append(
        {
            "index": 3 + len(cfg["conv_levels"]),
            "kind": "kottke_reference",
            "cpw": float(ref_lv["cpw"]),
            "dz_fine_nm": float(ref_lv["dz_fine_nm"]),
            "min_cells_per_layer": int(ref_lv["min_cells"]),
            "Lx": float(cfg["Lx"]),
            "is_demo": bool(ref_lv["is_demo"]),
            "file": f"task_{3 + len(cfg['conv_levels']):02d}_ref_{ref_lv['tag']}.npz",
            "label": str(ref_lv["label"]),
        }
    )
    return tasks


def build_task_mesh_info(task: dict) -> dict:
    """Build only the mesh for `task` (no solver execution) and return resolution metrics."""
    kind = task["kind"]
    cpw = float(task["cpw"])
    Lx = float(task.get("Lx", 2.8e-6))
    if kind == "semi_infinite":
        m = h.build_semi_infinite_mesh_2d(
            n_slab=float(np.max(SEMI_INF_INDICES)),
            cpw=cpw,
            lam0=WAVELENGTH_M,
            lx=Lx,
        )
        m["n_max_ref"] = float(np.max(SEMI_INF_INDICES))
        m["interface_z_errors_m"] = np.array([float(np.min(np.abs(m["grid"].z - 0.0)))])
        t_sim_s = 29.2e-15
    else:
        dz_fine_nm = task.get("dz_fine_nm", None)
        mc = int(task.get("min_cells_per_layer", 4))
        # For stack sweeps / convergence tasks, check the most demanding 25 nm p-GaN geometry
        t_check_nm = (
            25.0
            if kind in ("ryu_stack_sweep", "convergence_pair", "kottke_reference")
            else 50.0
        )
        layers, z_mqw = h.make_ryu_planar_layers(t_pgan_nm=t_check_nm)
        m = h.build_planar_led_mesh_2d(
            layers=layers,
            z_dipole=z_mqw,
            lam0=WAVELENGTH_M,
            n_max_ref=N_MAX_REF,
            cpw=cpw,
            dz_fine_nm=dz_fine_nm,
            min_cells_per_layer=mc,
            Lx=Lx,
        )
        m["n_max_ref"] = N_MAX_REF
        z_nodes = m["grid"].z
        iface_errs = [
            min(
                float(np.min(np.abs(z_nodes - lay.z_min))),
                float(np.min(np.abs(z_nodes - lay.z_max))),
            )
            for lay in layers
        ]
        m["interface_z_errors_m"] = np.asarray(iface_errs, dtype=float)
        t_sim_s = 39.2e-15

    dx_m = float(m["dx_background_nm"]) * 1e-9
    dz_m = float(m["dx_min_nm"]) * 1e-9
    dy_m = 1.0e-6
    dt_cfl = 1.0 / (
        h.C0 * np.sqrt(1.0 / dx_m**2 + 1.0 / dy_m**2 + 1.0 / dz_m**2)
    )
    m["n_steps"] = int(t_sim_s / dt_cfl)
    m["label"] = str(task.get("label", kind))
    m["cpw"] = cpw
    m["is_demo"] = bool(task.get("is_demo", False))
    m["n_max"] = float(m["n_max_ref"])
    return m


def _print_resolution_summary(tag: str, meta: dict, wall_s: float) -> None:
    skin_str = (
        f"skin={meta['skin_depth_nm']:.2f}nm, c/skin={meta['cells_per_skin_depth']:.2f}"
        if np.isfinite(meta["skin_depth_nm"])
        else "lossless (skin=inf)"
    )
    print(
        f"  [{tag}] dx_bg={meta['dx_background_nm']:.2f}nm, dx_min={meta['dx_min_nm']:.2f}nm, "
        f"feat={meta['min_feature_nm']:.1f}nm ({meta['cells_across_feature']} cells), "
        f"{skin_str}, n_cells={meta['n_cells']}, n_steps={meta['n_steps']}, time={wall_s:.1f}s"
    )


# %% [markdown]
# ## 3. Simulation Tasks: Semi-Infinite Slab, Ryu (2014) Stack, Incoherent Averaging, and Convergence


# %%
def run_task(task: dict, out_dir: Path) -> Path:
    """Execute one study task and save its results to an `.npz` archive."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / task["file"]
    kind = task["kind"]
    cpw = float(task["cpw"])
    dz_fine_nm = task.get("dz_fine_nm", None)
    mc = int(task.get("min_cells_per_layer", 4))
    Lx = float(task.get("Lx", 2.8e-6))
    t0 = time.perf_counter()

    if kind == "semi_infinite":
        pols = ("y", "x", "z")
        sim_lee = np.zeros((len(SEMI_INF_INDICES), len(pols)), dtype=float)
        ana_lee = np.zeros_like(sim_lee)
        cone_lee = np.zeros(len(SEMI_INF_INDICES), dtype=float)
        box_conservation = np.zeros_like(sim_lee)
        last_res = None

        for i_n, n_val in enumerate(SEMI_INF_INDICES):
            cone_lee[i_n] = float(h.escape_cone_2d_nofresnel(n_val, 1.0))
            for i_p, pol in enumerate(pols):
                res = h.run_semi_infinite_slab_2d(
                    n_slab=float(n_val),
                    dipole_pol=pol,
                    cpw=cpw,
                    lam0=WAVELENGTH_M,
                    lx=Lx,
                )
                sim_lee[i_n, i_p] = res["lee_top"]
                ana_lee[i_n, i_p] = res["lee_analytic"]
                box_conservation[i_n, i_p] = res["box_over_inj"]
                last_res = res

        wall_s = time.perf_counter() - t0
        assert last_res is not None
        _print_resolution_summary(task["label"], last_res, wall_s)
        np.savez(
            out_path,
            indices=SEMI_INF_INDICES,
            pols=np.array(pols),
            cpw=cpw,
            sim_lee=sim_lee,
            ana_lee=ana_lee,
            cone_lee=cone_lee,
            box_conservation=box_conservation,
            dx_background_nm=last_res["dx_background_nm"],
            dx_min_nm=last_res["dx_min_nm"],
            min_feature_nm=last_res["min_feature_nm"],
            cells_across_feature=last_res["cells_across_feature"],
            skin_depth_nm=last_res["skin_depth_nm"],
            cells_per_skin_depth=last_res["cells_per_skin_depth"],
            n_cells=last_res["n_cells"],
            n_steps=last_res["n_steps"],
            wall_time_s=wall_s,
        )
        return out_path

    if kind == "ryu_stack_sweep":
        n_t = len(PGAN_THICKNESSES_NM)
        lee_te_2d = np.zeros(n_t, dtype=float)
        lee_tm_2d = np.zeros(n_t, dtype=float)
        lee_te_3d = np.zeros(n_t, dtype=float)
        lee_tm_3d = np.zeros(n_t, dtype=float)
        bot_te = np.zeros(n_t, dtype=float)
        sides_te = np.zeros(n_t, dtype=float)
        abs_te = np.zeros(n_t, dtype=float)
        bot_tm = np.zeros(n_t, dtype=float)
        sides_tm = np.zeros(n_t, dtype=float)
        abs_tm = np.zeros(n_t, dtype=float)

        saved_fields: dict[str, np.ndarray] = {}
        ref_meta_25 = None

        for i, t_nm in enumerate(PGAN_THICKNESSES_NM):
            layers, z_mqw = h.make_ryu_planar_layers(t_pgan_nm=float(t_nm))
            want_extra = float(t_nm) in (0.0, 100.0)
            want_movie = float(t_nm) == 50.0

            r_te = h.run_2d_dipole_simulation(
                layers,
                z_mqw,
                dipole_pol="x",
                cpw=cpw,
                dz_fine_nm=dz_fine_nm,
                min_cells_per_layer=mc,
                Lx=Lx,
                record_dft_2d=want_extra,
                record_movie=want_movie,
                record_n2f=True,
            )
            r_tm = h.run_2d_dipole_simulation(
                layers,
                z_mqw,
                dipole_pol="z",
                cpw=cpw,
                dz_fine_nm=dz_fine_nm,
                min_cells_per_layer=mc,
                Lx=Lx,
                record_dft_2d=want_extra,
                record_movie=False,
                record_n2f=True,
            )
            if float(t_nm) == 25.0:
                ref_meta_25 = r_te

            lee_te_2d[i] = r_te["lee_top"]
            lee_tm_2d[i] = r_tm["lee_top"]
            lee_te_3d[i] = r_te["lee_3d_est"]
            lee_tm_3d[i] = r_tm["lee_3d_est"]
            bot_te[i] = r_te["frac_bot"]
            sides_te[i] = r_te["frac_sides"]
            abs_te[i] = r_te["frac_abs"]
            bot_tm[i] = r_tm["frac_bot"]
            sides_tm[i] = r_tm["frac_sides"]
            abs_tm[i] = r_tm["frac_abs"]

            if want_extra:
                tag = f"t{int(t_nm):03d}"
                saved_fields[f"{tag}_x"] = r_te["dft_2d"]["x"]
                saved_fields[f"{tag}_z"] = r_te["dft_2d"]["z"]
                saved_fields[f"{tag}_te_Ex"] = r_te["dft_2d"]["Ex"]
                saved_fields[f"{tag}_te_Ez"] = r_te["dft_2d"]["Ez"]
                saved_fields[f"{tag}_te_I"] = r_te["dft_2d"]["I"] / r_te["P_inj"]
                saved_fields[f"{tag}_tm_Ex"] = r_tm["dft_2d"]["Ex"]
                saved_fields[f"{tag}_tm_Ez"] = r_tm["dft_2d"]["Ez"]
                saved_fields[f"{tag}_tm_I"] = r_tm["dft_2d"]["I"] / r_tm["P_inj"]
                saved_fields[f"{tag}_theta_deg"] = r_te["n2f"]["theta_deg"]
                saved_fields[f"{tag}_te_n2f"] = r_te["n2f"]["d_eta_dtheta"]
                saved_fields[f"{tag}_tm_n2f"] = r_tm["n2f"]["d_eta_dtheta"]

            if want_movie:
                saved_fields["time_fs"] = r_te["time_fs"]
                saved_fields["src_trace"] = r_te["src_trace"]
                saved_fields["top_trace"] = r_te["top_trace"]
                saved_fields["energy_trace"] = r_te["energy_trace"]
                frames_arr = np.stack([f["field"] for f in r_te["movie_frames"]], axis=0)
                frames_t = np.array([f["t_fs"] for f in r_te["movie_frames"]], dtype=float)
                saved_fields["movie_frames"] = frames_arr
                saved_fields["movie_t_fs"] = frames_t
                saved_fields["mat_z_node"] = r_te["mat_profiles"]["z_node"]
                saved_fields["mat_z_centers"] = r_te["mat_profiles"]["z_centers"]
                saved_fields["mat_kottke_eps_x"] = r_te["mat_profiles"]["eps_x"]
                saved_fields["mat_kottke_eps_z"] = r_te["mat_profiles"]["eps_z"]
                saved_fields["mat_kottke_sigma"] = r_te["mat_profiles"]["sigma_x"]

                # Also extract 1D staircase profile at t_pGaN = 50 nm for Fig. 2
                r_stair_prof = h.run_2d_dipole_simulation(
                    layers,
                    z_mqw,
                    dipole_pol="x",
                    cpw=cpw,
                    dz_fine_nm=dz_fine_nm,
                    min_cells_per_layer=mc,
                    Lx=Lx,
                    averaging="staircase",
                    extra_time_fs=1.0,
                    record_n2f=False,
                )
                saved_fields["mat_stair_eps_x"] = r_stair_prof["mat_profiles"]["eps_x"]
                saved_fields["mat_stair_eps_z"] = r_stair_prof["mat_profiles"]["eps_z"]
                saved_fields["mat_stair_sigma"] = r_stair_prof["mat_profiles"]["sigma_x"]

        # AlGaN refractive index sensitivity sweep (Ryu Fig. 7 counterpart) at t_pGaN = 0 and 50 nm
        idx_te_t00 = np.zeros(len(ALGAN_INDICES), dtype=float)
        idx_tm_t00 = np.zeros(len(ALGAN_INDICES), dtype=float)
        idx_te_t50 = np.zeros(len(ALGAN_INDICES), dtype=float)
        idx_tm_t50 = np.zeros(len(ALGAN_INDICES), dtype=float)
        for j, n_alg in enumerate(ALGAN_INDICES):
            lay0, zm0 = h.make_ryu_planar_layers(t_pgan_nm=0.0, n_algan=float(n_alg))
            lay50, zm50 = h.make_ryu_planar_layers(t_pgan_nm=50.0, n_algan=float(n_alg))
            idx_te_t00[j] = h.run_2d_dipole_simulation(
                lay0,
                zm0,
                dipole_pol="x",
                cpw=cpw,
                dz_fine_nm=dz_fine_nm,
                min_cells_per_layer=mc,
                Lx=Lx,
                record_n2f=False,
            )["lee_top"]
            idx_tm_t00[j] = h.run_2d_dipole_simulation(
                lay0,
                zm0,
                dipole_pol="z",
                cpw=cpw,
                dz_fine_nm=dz_fine_nm,
                min_cells_per_layer=mc,
                Lx=Lx,
                record_n2f=False,
            )["lee_top"]
            idx_te_t50[j] = h.run_2d_dipole_simulation(
                lay50,
                zm50,
                dipole_pol="x",
                cpw=cpw,
                dz_fine_nm=dz_fine_nm,
                min_cells_per_layer=mc,
                Lx=Lx,
                record_n2f=False,
            )["lee_top"]
            idx_tm_t50[j] = h.run_2d_dipole_simulation(
                lay50,
                zm50,
                dipole_pol="z",
                cpw=cpw,
                dz_fine_nm=dz_fine_nm,
                min_cells_per_layer=mc,
                Lx=Lx,
                record_n2f=False,
            )["lee_top"]

        wall_s = time.perf_counter() - t0
        assert ref_meta_25 is not None
        _print_resolution_summary(task["label"], ref_meta_25, wall_s)
        np.savez(
            out_path,
            pgan_nm=PGAN_THICKNESSES_NM,
            lee_te_2d=lee_te_2d,
            lee_tm_2d=lee_tm_2d,
            lee_te_3d=lee_te_3d,
            lee_tm_3d=lee_tm_3d,
            bot_te=bot_te,
            sides_te=sides_te,
            abs_te=abs_te,
            bot_tm=bot_tm,
            sides_tm=sides_tm,
            abs_tm=abs_tm,
            algan_indices=ALGAN_INDICES,
            idx_te_t00=idx_te_t00,
            idx_tm_t00=idx_tm_t00,
            idx_te_t50=idx_te_t50,
            idx_tm_t50=idx_tm_t50,
            cpw=cpw,
            dx_background_nm=ref_meta_25["dx_background_nm"],
            dx_min_nm=ref_meta_25["dx_min_nm"],
            min_feature_nm=ref_meta_25["min_feature_nm"],
            cells_across_feature=ref_meta_25["cells_across_feature"],
            skin_depth_nm=ref_meta_25["skin_depth_nm"],
            cells_per_skin_depth=ref_meta_25["cells_per_skin_depth"],
            n_cells=ref_meta_25["n_cells"],
            n_steps=ref_meta_25["n_steps"],
            wall_time_s=wall_s,
            **saved_fields,
        )
        return out_path

    if kind == "incoherent_positions":
        layers, z_mqw = h.make_ryu_planar_layers(t_pgan_nm=50.0)
        runs_te = []
        runs_tm = []
        for x_um in DIPOLE_X_POSITIONS_UM:
            runs_te.append(
                h.run_2d_dipole_simulation(
                    layers,
                    z_mqw,
                    x_dipole=float(x_um * 1e-6),
                    dipole_pol="x",
                    cpw=cpw,
                    dz_fine_nm=dz_fine_nm,
                    min_cells_per_layer=mc,
                    Lx=Lx,
                    record_dft_2d=True,
                    record_n2f=True,
                )
            )
            runs_tm.append(
                h.run_2d_dipole_simulation(
                    layers,
                    z_mqw,
                    x_dipole=float(x_um * 1e-6),
                    dipole_pol="z",
                    cpw=cpw,
                    dz_fine_nm=dz_fine_nm,
                    min_cells_per_layer=mc,
                    Lx=Lx,
                    record_dft_2d=True,
                    record_n2f=True,
                )
            )

        comb_te = h.incoherent_vs_coherent_combine(runs_te, weights=DIPOLE_X_WEIGHTS)
        comb_tm = h.incoherent_vs_coherent_combine(runs_tm, weights=DIPOLE_X_WEIGHTS)

        # Also compute unpolarized (incoherent 1/2 TE + 1/2 TM in 2D)
        cum_lee_unpol = 0.5 * (comb_te["cum_lee_incoh"] + comb_tm["cum_lee_incoh"])
        d_eta_unpol = 0.5 * (comb_te["d_eta_incoh"] + comb_tm["d_eta_incoh"])
        wall_s = time.perf_counter() - t0
        r0 = runs_te[0]
        _print_resolution_summary(task["label"], r0, wall_s)

        np.savez(
            out_path,
            x_positions_um=DIPOLE_X_POSITIONS_UM,
            weights=DIPOLE_X_WEIGHTS,
            n_positions=comb_te["n_positions"],
            lee_te_each=np.array([r["lee_top"] for r in runs_te]),
            lee_tm_each=np.array([r["lee_top"] for r in runs_tm]),
            cum_lee_te=comb_te["cum_lee_incoh"],
            cum_lee_tm=comb_tm["cum_lee_incoh"],
            cum_lee_unpol=cum_lee_unpol,
            I_incoh_te=comb_te["I_incoh"],
            I_coh_te=comb_te["I_coh"],
            I_incoh_tm=comb_tm["I_incoh"],
            I_coh_tm=comb_tm["I_coh"],
            x_grid=runs_te[0]["dft_2d"]["x"],
            z_grid=runs_te[0]["dft_2d"]["z"],
            theta_deg=runs_te[0]["n2f"]["theta_deg"],
            d_eta_incoh_te=comb_te["d_eta_incoh"],
            d_eta_incoh_tm=comb_tm["d_eta_incoh"],
            d_eta_unpol=d_eta_unpol,
            d_eta_single_te=runs_te[0]["n2f"]["d_eta_dtheta"],
            cpw=cpw,
            dx_background_nm=r0["dx_background_nm"],
            dx_min_nm=r0["dx_min_nm"],
            min_feature_nm=r0["min_feature_nm"],
            cells_across_feature=r0["cells_across_feature"],
            skin_depth_nm=r0["skin_depth_nm"],
            cells_per_skin_depth=r0["cells_per_skin_depth"],
            n_cells=r0["n_cells"],
            n_steps=r0["n_steps"],
            wall_time_s=wall_s,
        )
        return out_path

    if kind == "convergence_pair":
        layers_50, z_mqw_50 = h.make_ryu_planar_layers(t_pgan_nm=50.0)
        r_kot_te = h.run_2d_dipole_simulation(
            layers_50,
            z_mqw_50,
            dipole_pol="x",
            cpw=cpw,
            dz_fine_nm=dz_fine_nm,
            min_cells_per_layer=mc,
            Lx=Lx,
            averaging="kottke",
            record_n2f=True,
        )
        r_kot_tm = h.run_2d_dipole_simulation(
            layers_50,
            z_mqw_50,
            dipole_pol="z",
            cpw=cpw,
            dz_fine_nm=dz_fine_nm,
            min_cells_per_layer=mc,
            Lx=Lx,
            averaging="kottke",
            record_n2f=True,
        )
        r_sta_te = h.run_2d_dipole_simulation(
            layers_50,
            z_mqw_50,
            dipole_pol="x",
            cpw=cpw,
            dz_fine_nm=dz_fine_nm,
            min_cells_per_layer=mc,
            Lx=Lx,
            averaging="staircase",
            record_n2f=False,
        )
        r_sta_tm = h.run_2d_dipole_simulation(
            layers_50,
            z_mqw_50,
            dipole_pol="z",
            cpw=cpw,
            dz_fine_nm=dz_fine_nm,
            min_cells_per_layer=mc,
            Lx=Lx,
            averaging="staircase",
            record_n2f=False,
        )

        # Resolution study on t_pGaN = 25 nm and 100 nm (Rule 2 & Rule 3 absorber study)
        layers_25, z_mqw_25 = h.make_ryu_planar_layers(t_pgan_nm=25.0)
        r_kot_te_25 = h.run_2d_dipole_simulation(
            layers_25,
            z_mqw_25,
            dipole_pol="x",
            cpw=cpw,
            dz_fine_nm=dz_fine_nm,
            min_cells_per_layer=mc,
            Lx=Lx,
            averaging="kottke",
            record_n2f=True,
        )
        r_sta_te_25 = h.run_2d_dipole_simulation(
            layers_25,
            z_mqw_25,
            dipole_pol="x",
            cpw=cpw,
            dz_fine_nm=dz_fine_nm,
            min_cells_per_layer=mc,
            Lx=Lx,
            averaging="staircase",
            record_n2f=False,
        )

        layers_100, z_mqw_100 = h.make_ryu_planar_layers(t_pgan_nm=100.0)
        r_kot_te_100 = h.run_2d_dipole_simulation(
            layers_100,
            z_mqw_100,
            dipole_pol="x",
            cpw=cpw,
            dz_fine_nm=dz_fine_nm,
            min_cells_per_layer=mc,
            Lx=Lx,
            averaging="kottke",
            record_n2f=True,
        )
        r_sta_te_100 = h.run_2d_dipole_simulation(
            layers_100,
            z_mqw_100,
            dipole_pol="x",
            cpw=cpw,
            dz_fine_nm=dz_fine_nm,
            min_cells_per_layer=mc,
            Lx=Lx,
            averaging="staircase",
            record_n2f=False,
        )

        wall_s = time.perf_counter() - t0
        _print_resolution_summary(task["label"], r_kot_te_25, wall_s)
        np.savez(
            out_path,
            cpw=cpw,
            is_demo=bool(task.get("is_demo", False)),
            dx_nm=r_kot_te_25["dx_min_nm"],
            dx_background_nm=r_kot_te_25["dx_background_nm"],
            dx_min_nm=r_kot_te_25["dx_min_nm"],
            min_feature_nm=r_kot_te_25["min_feature_nm"],
            cells_across_feature=r_kot_te_25["cells_across_feature"],
            skin_depth_nm=r_kot_te_25["skin_depth_nm"],
            cells_per_skin_depth=r_kot_te_25["cells_per_skin_depth"],
            n_cells=r_kot_te_25["n_cells"],
            n_steps=r_kot_te_25["n_steps"],
            kottke_te=r_kot_te["lee_top"],
            kottke_tm=r_kot_tm["lee_top"],
            stair_te=r_sta_te["lee_top"],
            stair_tm=r_sta_tm["lee_top"],
            kottke_te_t25=r_kot_te_25["lee_top"],
            kottke_te_3d_t25=r_kot_te_25["lee_3d_est"],
            abs_te_t25=r_kot_te_25["frac_abs"],
            stair_te_t25=r_sta_te_25["lee_top"],
            kottke_te_t100=r_kot_te_100["lee_top"],
            kottke_te_3d_t100=r_kot_te_100["lee_3d_est"],
            abs_te_t100=r_kot_te_100["frac_abs"],
            stair_te_t100=r_sta_te_100["lee_top"],
            wall_time_s=wall_s,
        )
        return out_path

    if kind == "kottke_reference":
        layers_50, z_mqw_50 = h.make_ryu_planar_layers(t_pgan_nm=50.0)
        r_ref_te = h.run_2d_dipole_simulation(
            layers_50,
            z_mqw_50,
            dipole_pol="x",
            cpw=cpw,
            dz_fine_nm=dz_fine_nm,
            min_cells_per_layer=mc,
            Lx=Lx,
            averaging="kottke",
            record_n2f=True,
        )
        r_ref_tm = h.run_2d_dipole_simulation(
            layers_50,
            z_mqw_50,
            dipole_pol="z",
            cpw=cpw,
            dz_fine_nm=dz_fine_nm,
            min_cells_per_layer=mc,
            Lx=Lx,
            averaging="kottke",
            record_n2f=True,
        )
        layers_25, z_mqw_25 = h.make_ryu_planar_layers(t_pgan_nm=25.0)
        r_ref_te_25 = h.run_2d_dipole_simulation(
            layers_25,
            z_mqw_25,
            dipole_pol="x",
            cpw=cpw,
            dz_fine_nm=dz_fine_nm,
            min_cells_per_layer=mc,
            Lx=Lx,
            averaging="kottke",
            record_n2f=True,
        )
        layers_100, z_mqw_100 = h.make_ryu_planar_layers(t_pgan_nm=100.0)
        r_ref_te_100 = h.run_2d_dipole_simulation(
            layers_100,
            z_mqw_100,
            dipole_pol="x",
            cpw=cpw,
            dz_fine_nm=dz_fine_nm,
            min_cells_per_layer=mc,
            Lx=Lx,
            averaging="kottke",
            record_n2f=True,
        )
        wall_s = time.perf_counter() - t0
        _print_resolution_summary(task["label"], r_ref_te_25, wall_s)
        np.savez(
            out_path,
            cpw=cpw,
            is_demo=False,
            dx_nm=r_ref_te_25["dx_min_nm"],
            dx_background_nm=r_ref_te_25["dx_background_nm"],
            dx_min_nm=r_ref_te_25["dx_min_nm"],
            min_feature_nm=r_ref_te_25["min_feature_nm"],
            cells_across_feature=r_ref_te_25["cells_across_feature"],
            skin_depth_nm=r_ref_te_25["skin_depth_nm"],
            cells_per_skin_depth=r_ref_te_25["cells_per_skin_depth"],
            n_cells=r_ref_te_25["n_cells"],
            n_steps=r_ref_te_25["n_steps"],
            ref_te=r_ref_te["lee_top"],
            ref_tm=r_ref_tm["lee_top"],
            ref_te_t25=r_ref_te_25["lee_top"],
            ref_te_3d_t25=r_ref_te_25["lee_3d_est"],
            abs_te_t25=r_ref_te_25["frac_abs"],
            ref_te_t100=r_ref_te_100["lee_top"],
            ref_te_3d_t100=r_ref_te_100["lee_3d_est"],
            abs_te_t100=r_ref_te_100["frac_abs"],
            wall_time_s=wall_s,
        )
        return out_path

    raise ValueError(f"Unknown task kind {kind!r}")


# %% [markdown]
# ## 4. Plotting & Visualization Pipeline (Setup Schematics + 15 Diagnostic Figures)
#
# Calling `plot_all(data_dir, fig_dir, mode)` renders the setup figures first
# followed by 15 publication-grade figures in `figures/`:
# 0. `00a_setup_geometry.png`, `00b_setup_mesh_and_excitation.png`, and `00_setup_table.md`:
#    Labelled setup geometry of the 6-layer Ryu (2014) planar AlGaN deep-UV LED stack
#    with the 18-dipole MQW ensemble (9 symmetric lateral sites $\times$ TE $J_x$, TM $J_z$)
#    and `Near2FarMonitor` / `FluxMonitor` planes, paired with the graded thin-layer mesh
#    ($\Delta z_{\min} = 3.125\;\text{nm}$) and UV dipole excitation spectrum.
# 1. `01_led_stack_geometry.png`: (a) Ryu (2014) stack schematic with escape cone $\theta_c = 22.6^\circ$,
#    (b) refractive index profile $n(z)$, and (c) absorption profile $\alpha(z)$ with skin depth $\delta = 58.8\;\text{nm}$.
# 2. `02_mesh_kottke_vs_staircase.png`: Subpixel Kottke harmonic/arithmetic permittivity
#    and ADE conductivity profiles vs. staircase discretization across the MQW and p-GaN interfaces.
# 3. `03_analytic_escape_cone_2d_vs_3d.png`: Analytic 2D ($\theta_c/\pi$) vs. 3D ($\frac{1}{2}(1-\cos\theta_c) \approx \frac{1}{4n^2}$)
#    escape-cone fractions and Fresnel-weighted polar patterns for $J_y, J_x, J_z$.
# 4. `04_semi_infinite_validation.png`: Validation of 2D Wakis FIT semi-infinite slab
#    extraction efficiency against exact 2D Fresnel integrals ($n \in \{1.5, 2.0, 2.6\}$).
# 5. `05_field_intensity_te_vs_tm.png` & `05b_field_intensity_pgan100.png`: 2D steady-state
#    intensity $|\mathbf{E}(x, z, \omega_0)|^2$ with full `overlay_geometry` (layers, CPML,
#    dipole, monitors) comparing TE ($J_x$) vs. TM ($J_z$) dipoles at $t_{\text{p-GaN}} = 0\;\text{nm}$ and $100\;\text{nm}$.
# 6. `06_field_wavefronts_te_vs_tm.png`: Real electric field wavefronts $\text{Re}(E_x)$ (TE)
#    and $\text{Re}(E_z)$ (TM) with `overlay_geometry` showing escape-cone refraction vs. guided slab trapping.
# 7. `07_dipole_emission_movie.gif`: Animated time-domain pulse propagation, TIR reflections,
#    and top-cone extraction into air with all 6 layers, CPML boundaries, dipole, and monitors labelled on every frame.
# 8. `08_lee_vs_pgan_ryu_fig3.png`: Light extraction efficiency vs. p-GaN contact thickness
#    ($0\text{--}200\;\text{nm}$) reproducing Ryu (2014) Fig. 3 (linear and semi-log Beer-Lambert scales).
# 9. `09_power_budget_4channel.png`: Complete 4-channel power budget (top extraction, bottom
#    sapphire leakage, lateral guided trapping, and active/p-GaN absorption) vs. $t_{\text{p-GaN}}$.
# 10. `10_lee_vs_algan_index_ryu_fig7.png`: Extraction efficiency sensitivity to the AlGaN
#     cladding refractive index $n_{\text{AlGaN}} \in \{2.50, 2.60, 2.70\}$ (Ryu Fig. 7 counterpart).
# 11. `11_farfield_n2f_vs_lambertian.png`: Far-field radiation pattern via `Near2FarMonitor`
#     (Cartesian and polar plots) for TE, TM, and incoherently averaged emission vs. Lambertian $\cos\theta$.
# 12. `12_incoherent_vs_coherent_convergence.png`: Incoherent power averaging vs. erroneous
#     coherent field summation across lateral dipole positions $x_i$ with `overlay_geometry`, plus convergence vs. $N_{\text{pos}}$.
# 13. `13_resolution_convergence_kottke_vs_staircase.png`: Grid convergence of TE and TM LEE
#     comparing Kottke subpixel smoothing against staircase snapping relative to the fine reference grid.
# 14. `14_time_trace_and_energy_decay.png`: Source/extraction time traces and electromagnetic
#     energy decay confirming steady-state DFT convergence.


# %%
def plot_all(data_dir: Path, fig_dir: Path, mode: str = "full") -> list[Path]:
    """Generate setup figures and all 15 figures from the saved `.npz` task archives."""
    figstyle.apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    tasks = _task_table(mode)
    cfg = _mode_config(mode)
    Lx_um = float(cfg["Lx"]) * 1e6

    d_semi = np.load(data_dir / tasks[0]["file"])
    d_ryu = np.load(data_dir / tasks[1]["file"])
    d_inc = np.load(data_dir / tasks[2]["file"])
    conv_tasks = [t for t in tasks if t["kind"] == "convergence_pair"]
    ref_task = [t for t in tasks if t["kind"] == "kottke_reference"][0]
    d_convs = [np.load(data_dir / t["file"]) for t in conv_tasks]
    d_ref = np.load(data_dir / ref_task["file"])

    written: list[Path] = []

    # ------------------------------------------------------------------
    # Fig 00a & 00b: Setup Geometry, Mesh & Excitation + Setup Table
    # ------------------------------------------------------------------
    setup_out = h.save_planar_led_setup_figures(fig_dir, mode=mode)
    written.extend([setup_out["geometry_figure"], setup_out["mesh_figure"]])
    tc_rad = h.escape_cone_angle(2.60, 1.0)

    # ------------------------------------------------------------------
    # Fig 01: LED Stack Geometry, Escape Cone, Index & Absorption Profiles (3 panels)
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(
        nrows=1, ncols=3, width=13.2, height=4.2, gridspec_kw={"width_ratios": [1.45, 0.8, 0.85]}
    )
    ryu_100 = h.build_ryu_schematic_objects(
        t_pgan_nm=100.0,
        cpw=float(cfg["cpw"]),
        dz_fine_nm=float(cfg["dz_fine_nm"]),
        Lx=float(cfg["Lx"]),
    )
    schematic.plot_schematic(
        ryu_100["solver"],
        plane="xz",
        ax=axes[0],
        unit="um",
        structures=ryu_100["structures"],
        sources=ryu_100["sources"],
        monitors=ryu_100["monitors"],
        show_mesh=False,
        show_wave_packet=False,
        title="(a) Ryu (2014) UV LED Stack ($\\lambda_0 = 280$ nm)",
    )
    h.annotate_thin_led_layers(
        axes[0], t_pgan_nm=100.0, x_dom_half_um=0.5 * Lx_um, fontsize=8.0, tidy_schematic=True
    )
    z_dip_um = -0.075
    z_top_surf = 0.10
    dx_cone = (z_top_surf - z_dip_um) * np.tan(tc_rad)
    cone_poly = mpatches.Polygon(
        [[0.0, z_dip_um], [-dx_cone * 2.3, 0.36], [dx_cone * 2.3, 0.36]],
        closed=True,
        facecolor="#fff7bc",
        edgecolor="#d95f02",
        linestyle="--",
        lw=1.3,
        alpha=0.50,
        label=f"Escape cone $\\theta_c = {np.degrees(tc_rad):.1f}^\\circ$",
        zorder=5,
    )
    axes[0].add_patch(cone_poly)
    axes[0].legend(loc="lower right", fontsize=7.0)

    z_fine = np.linspace(-0.85, 0.40, 1500)
    n_prof = np.ones_like(z_fine)
    alpha_prof = np.full_like(z_fine, 1e-1)
    lays_100, _ = h.make_ryu_planar_layers(t_pgan_nm=100.0)
    for lay in lays_100:
        m = (z_fine >= lay.z_min * 1e6) & (z_fine <= lay.z_max * 1e6)
        n_prof[m] = lay.n
        alpha_prof[m] = max(lay.alpha_cm, 1e-1)

    axes[1].plot(n_prof, z_fine, **figstyle.series_style(0), label="$n(z)$")
    for z_int in (-0.55, -0.10, -0.05, 0.0, 0.10):
        axes[1].axhline(z_int, color="#888888", ls=":", lw=0.7)
    axes[1].set_xlabel("Refractive index $n$")
    axes[1].set_ylabel("Vertical coordinate $z$ [$\\mu$m]")
    axes[1].set_xlim(0.8, 3.15)
    axes[1].set_ylim(-0.85, 0.40)
    axes[1].set_title("(b) Index Profile $n(z)$")

    axes[2].semilogx(alpha_prof, z_fine, **figstyle.series_style(1), label="$\\alpha(z)$")
    for z_int in (-0.55, -0.10, -0.05, 0.0, 0.10):
        axes[2].axhline(z_int, color="#888888", ls=":", lw=0.7)
    axes[2].annotate(
        "p-GaN absorber\n$\\delta = 58.8$ nm",
        xy=(1.7e5, 0.05),
        xytext=(1.5e2, 0.22),
        fontsize=7.0,
        bbox=bbox_z,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#333333"},
    )
    axes[2].set_xlabel("Absorption $\\alpha$ [cm$^{-1}$]")
    axes[2].set_xlim(5e-2, 5e5)
    axes[2].set_ylim(-0.85, 0.40)
    axes[2].set_title("(c) Absorption $\\alpha(z)$")
    figstyle.finish(fig)
    p1 = fig_dir / "01_led_stack_geometry.png"
    figstyle.save(fig, p1)
    written.append(p1)

    # ------------------------------------------------------------------
    # Fig 02: Mesh & Subpixel Kottke vs. Staircase Profiles
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=3.8)
    z_node_nm = d_ryu["mat_z_node"] * 1e9
    z_cent_nm = d_ryu["mat_z_centers"] * 1e9
    mask_zoom = (z_node_nm >= -140.0) & (z_node_nm <= 90.0)
    mask_cent = (z_cent_nm >= -140.0) & (z_cent_nm <= 90.0)

    ax = axes[0]
    ax.step(
        z_node_nm[mask_zoom],
        d_ryu["mat_stair_eps_x"][mask_zoom],
        where="mid",
        color="#636363",
        ls="--",
        lw=1.6,
        label="Staircase $\\varepsilon_{xx}, \\varepsilon_{zz}$",
    )
    ax.plot(
        z_node_nm[mask_zoom],
        d_ryu["mat_kottke_eps_x"][mask_zoom],
        "o-",
        color=figstyle.PALETTE[0],
        ms=4.0,
        lw=1.7,
        label="Kottke tangential $\\langle\\varepsilon_{xx}\\rangle$ (arithmetic)",
    )
    ax.plot(
        z_cent_nm[mask_cent],
        d_ryu["mat_kottke_eps_z"][mask_cent],
        "s-",
        color=figstyle.PALETTE[1],
        ms=4.0,
        lw=1.7,
        label="Kottke normal $\\langle\\varepsilon_{zz}^{-1}\\rangle^{-1}$ (harmonic)",
    )
    for zb, lbl in [
        (-100.0, "n-AlGaN/MQW"),
        (-50.0, "MQW/p-AlGaN"),
        (0.0, "p-AlGaN/p-GaN"),
        (50.0, "p-GaN/Air"),
    ]:
        ax.axvline(zb, color="#222222", ls=":", alpha=0.55)
        ax.text(zb + 1.5, 1.3, lbl, rotation=90, fontsize=7.0, va="bottom")
    ax.set_xlabel("Vertical coordinate $z$ [nm]")
    ax.set_ylabel("Effective relative permittivity $\\varepsilon_r$")
    ax.set_title("(a) Subpixel Permittivity Smoothing across Interfaces")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.step(
        z_node_nm[mask_zoom],
        d_ryu["mat_stair_sigma"][mask_zoom] / 1e3,
        where="mid",
        color="#636363",
        ls="--",
        lw=1.6,
        label="Staircase $\\sigma(z)$",
    )
    ax.plot(
        z_node_nm[mask_zoom],
        d_ryu["mat_kottke_sigma"][mask_zoom] / 1e3,
        "o-",
        color=figstyle.PALETTE[1],
        ms=4.0,
        lw=1.7,
        label="Kottke subcell-weighted $\\langle\\sigma(z)\\rangle$",
    )
    for zb in (-100.0, -50.0, 0.0, 50.0):
        ax.axvline(zb, color="#222222", ls=":", alpha=0.45)
    ax.set_xlabel("Vertical coordinate $z$ [nm]")
    ax.set_ylabel("Effective conductivity $\\sigma$ [$10^3$ S/m]")
    ax.set_title("(b) Subcell ADE Conductivity ($t_{\\mathrm{p-GaN}} = 50$ nm)")
    ax.legend(loc="upper left")
    figstyle.finish(fig)
    p2 = fig_dir / "02_mesh_kottke_vs_staircase.png"
    figstyle.save(fig, p2)
    written.append(p2)

    # ------------------------------------------------------------------
    # Fig 03: Analytic 2D vs 3D Escape Cone & Dipole Angular Integrands
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=3.9)
    n_grid = np.linspace(1.1, 3.5, 200)
    eta_2d_cone = h.escape_cone_2d_nofresnel(n_grid) * 100.0
    eta_3d_cone = h.escape_cone_3d_nofresnel(n_grid) * 100.0
    eta_3d_approx = (1.0 / (4.0 * n_grid**2)) * 100.0
    eta_2d_jx = np.array([h.escape_cone_2d_fresnel(float(nv), 1.0, "x") * 100.0 for nv in n_grid])
    eta_2d_jy = np.array([h.escape_cone_2d_fresnel(float(nv), 1.0, "y") * 100.0 for nv in n_grid])
    eta_2d_jz = np.array([h.escape_cone_2d_fresnel(float(nv), 1.0, "z") * 100.0 for nv in n_grid])

    ax = axes[0]
    ax.plot(n_grid, eta_2d_cone, "k-", lw=2.0, label="2D isotropic cone $\\theta_c / \\pi$")
    ax.plot(n_grid, eta_2d_jx, **figstyle.series_style(2), label="2D Fresnel TE ($J_x \\propto \\cos^2\\theta$)")
    ax.plot(n_grid, eta_2d_jy, **figstyle.series_style(4), label="2D Fresnel $J_y$ (s-pol)")
    ax.plot(n_grid, eta_2d_jz, **figstyle.series_style(1), label="2D Fresnel TM ($J_z \\propto \\sin^2\\theta$)")
    ax.plot(n_grid, eta_3d_cone, **figstyle.series_style(3), label="3D isotropic cone $(1 - \\cos\\theta_c)/2$")
    ax.plot(n_grid, eta_3d_approx, "--", color=figstyle.PALETTE[3], lw=1.2, alpha=0.75, label="3D paraxial $1/(4n^2)$")
    ax.axvline(2.60, color="gray", ls=":", lw=1.1, label="AlGaN ($n = 2.60$)")
    ax.set_xlabel("Slab refractive index $n$")
    ax.set_ylabel("Single-face extraction efficiency $\\eta$ [%]")
    ax.set_title("(a) 2D ($\\mathcal{O}(1/n)$) vs. 3D ($\\mathcal{O}(1/n^2)$) Escape-Cone Scaling")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 60.0)
    ax.legend(loc="upper right", fontsize=7.2)

    ax = axes[1]
    th_deg = np.linspace(-89.5, 89.5, 501)
    th_rad = np.radians(th_deg)
    tc_alg_deg = np.degrees(h.escape_cone_angle(2.60, 1.0))
    Tp_alg = h.fresnel_transmission(np.abs(th_rad), 2.60, 1.0, pol="p")
    Ts_alg = h.fresnel_transmission(np.abs(th_rad), 2.60, 1.0, pol="s")
    ax.plot(th_deg, np.cos(th_rad) ** 2, "--", color=figstyle.PALETTE[2], alpha=0.5, label="Internal TE $\\cos^2\\theta_1$")
    ax.plot(th_deg, np.sin(th_rad) ** 2, "--", color=figstyle.PALETTE[1], alpha=0.5, label="Internal TM $\\sin^2\\theta_1$")
    ax.plot(th_deg, Tp_alg * np.cos(th_rad) ** 2, "-", color=figstyle.PALETTE[2], lw=2.0, label="Extracted TE $T_p\\cos^2\\theta_1$")
    ax.plot(th_deg, Ts_alg * 0.5, "-.", color=figstyle.PALETTE[4], lw=1.7, label="Extracted $J_y$ $\\frac{1}{2}T_s$")
    ax.plot(th_deg, Tp_alg * np.sin(th_rad) ** 2, "-", color=figstyle.PALETTE[1], lw=2.0, label="Extracted TM $T_p\\sin^2\\theta_1$")
    ax.axvspan(-tc_alg_deg, tc_alg_deg, color="#fff7bc", alpha=0.45, label=f"Escape cone $|\\theta_1| \\leq {tc_alg_deg:.1f}^\\circ$")
    ax.set_xlabel("Internal emission angle $\\theta_1$ in AlGaN [deg]")
    ax.set_ylabel("Angular power density [arb. u.]")
    ax.set_title("(b) Why TM Dipoles Are Trapped ($n = 2.60$, $\\theta_c = 22.6^\\circ$)")
    ax.legend(loc="upper right", fontsize=7.2)
    figstyle.finish(fig)
    p3 = fig_dir / "03_analytic_escape_cone_2d_vs_3d.png"
    figstyle.save(fig, p3)
    written.append(p3)

    # ------------------------------------------------------------------
    # Fig 04: 2D Semi-Infinite Slab Wakis Validation vs Analytic Integrals
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=3.8)
    ind_vals = d_semi["indices"]
    sim_lee = d_semi["sim_lee"] * 100.0
    ana_lee = d_semi["ana_lee"] * 100.0

    ax = axes[0]
    ax.plot(n_grid, eta_2d_jx, "-", color=figstyle.PALETTE[2], lw=1.8, label="Analytic Fresnel $J_x$ (TE)")
    ax.plot(n_grid, eta_2d_jy, "-", color=figstyle.PALETTE[4], lw=1.8, label="Analytic Fresnel $J_y$ (s-pol)")
    ax.plot(n_grid, eta_2d_jz, "-", color=figstyle.PALETTE[1], lw=1.8, label="Analytic Fresnel $J_z$ (TM)")
    ax.plot(ind_vals, sim_lee[:, 1], "s", color=figstyle.PALETTE[2], ms=6.5, label="Wakis 2D FIT $J_x$")
    ax.plot(ind_vals, sim_lee[:, 0], "o", color=figstyle.PALETTE[4], ms=6.5, label="Wakis 2D FIT $J_y$")
    ax.plot(ind_vals, sim_lee[:, 2], "^", color=figstyle.PALETTE[1], ms=6.5, label="Wakis 2D FIT $J_z$")
    ax.set_xlabel("Semi-infinite slab refractive index $n$")
    ax.set_ylabel("Top single-face extraction efficiency [%]")
    ax.set_title("(a) 2D Wakis FIT vs. Exact 2D Fresnel Integrals")
    ax.legend(loc="upper right", ncol=2, fontsize=7.4)

    ax = axes[1]
    x_pos = np.arange(len(ind_vals))
    w_bar = 0.25
    rel_err = sim_lee - ana_lee
    ax.bar(x_pos - w_bar, rel_err[:, 0], width=w_bar, color=figstyle.PALETTE[4], label="$J_y$ (out-of-plane)")
    ax.bar(x_pos, rel_err[:, 1], width=w_bar, color=figstyle.PALETTE[2], label="$J_x$ (in-plane TE)")
    ax.bar(x_pos + w_bar, rel_err[:, 2], width=w_bar, color=figstyle.PALETTE[1], label="$J_z$ (vertical TM)")
    ax.axhline(0.0, color="k", lw=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"$n = {nv:.1f}$" for nv in ind_vals])
    ax.set_ylabel("Difference $\\eta_{\\mathrm{Wakis}} - \\eta_{\\mathrm{analytic}}$ [pp]")
    ax.set_title("(b) Half-Cavity Residual & Box Conservation")
    for i_n in range(len(ind_vals)):
        bc_val = float(np.mean(d_semi["box_conservation"][i_n])) * 100.0
        ax.text(
            i_n,
            max(0.2, float(np.max(rel_err[i_n])) + 0.15),
            f"Box flux:\n{bc_val:.1f}%",
            ha="center",
            fontsize=7.4,
        )
    ax.set_ylim(min(-3.5, float(np.min(rel_err)) - 0.8), max(3.5, float(np.max(rel_err)) + 1.3))
    ax.legend(loc="lower right", fontsize=7.6)
    figstyle.finish(fig)
    p4 = fig_dir / "04_semi_infinite_validation.png"
    figstyle.save(fig, p4)
    written.append(p4)

    # ------------------------------------------------------------------
    # Fig 05: 2D Steady-State Field Intensity Maps (3 panels, <= 3 panels rule)
    # and Fig 05b: TE vs TM at t_pGaN = 100 nm (2 panels)
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.2, height=4.1, sharey=True)
    panels_05 = [
        ("t000", "te", "x", "(a) TE ($J_x$), $t_{\\mathrm{p-GaN}} = 0$ nm", 0.0),
        ("t000", "tm", "z", "(b) TM ($J_z$), $t_{\\mathrm{p-GaN}} = 0$ nm (TIR)", 0.0),
        ("t100", "te", "x", "(c) TE ($J_x$), $t_{\\mathrm{p-GaN}} = 100$ nm", 100.0),
    ]
    I_ref_05 = float(np.percentile(d_ryu["t000_te_I"], 99.8))
    im_last = None
    for ax, (tag, pol_tag, dpol, title, t_nm_val) in zip(axes, panels_05):
        xg = d_ryu[f"{tag}_x"] * 1e6
        zg = d_ryu[f"{tag}_z"] * 1e6
        Lx_map_um = float(xg[-1] - xg[0])
        I_map = d_ryu[f"{tag}_{pol_tag}_I"].T
        I_db = np.clip(10.0 * np.log10(np.maximum(I_map / I_ref_05, 1e-5)), -32.0, 0.0)
        im_last = ax.pcolormesh(
            xg, zg, I_db, cmap=figstyle.CMAP_MAGNITUDE, vmin=-32.0, vmax=0.0, shading="auto"
        )
        h.overlay_ryu_stack(
            ax,
            t_pgan_nm=t_nm_val,
            Lx_um=Lx_map_um,
            dipole_pol=dpol,
            line_color="white",
            show_monitors=True,
            show_escape_cone=(pol_tag == "te" and t_nm_val == 0.0),
            fontsize=6.0,
        )
        ax.set_title(title)
        ax.set_xlabel("Lateral coordinate $x$ [$\\mu$m]")
    axes[0].set_ylabel("Vertical coordinate $z$ [$\\mu$m]")
    fig.colorbar(im_last, ax=axes[-1], label="Normalized $|\\mathbf{E}(\\omega_0)|^2$ [dB]")
    figstyle.finish(fig)
    p5 = fig_dir / "05_field_intensity_te_vs_tm.png"
    figstyle.save(fig, p5)
    written.append(p5)

    # Fig 05b: TE vs TM at t_pGaN = 100 nm (2 panels)
    fig_b, axes_b = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.0, sharey=True)
    for ax, (pol_tag, dpol, title) in zip(
        axes_b,
        [
            ("te", "x", "(a) TE Dipole ($J_x$), $t_{\\mathrm{p-GaN}} = 100$ nm (p-GaN Attenuated)"),
            ("tm", "z", "(b) TM Dipole ($J_z$), $t_{\\mathrm{p-GaN}} = 100$ nm (Trapped + Absorbed)"),
        ],
    ):
        xg = d_ryu["t100_x"] * 1e6
        zg = d_ryu["t100_z"] * 1e6
        Lx_map_um = float(xg[-1] - xg[0])
        I_map = d_ryu[f"t100_{pol_tag}_I"].T
        I_db = np.clip(10.0 * np.log10(np.maximum(I_map / I_ref_05, 1e-5)), -32.0, 0.0)
        im_b = ax.pcolormesh(
            xg, zg, I_db, cmap=figstyle.CMAP_MAGNITUDE, vmin=-32.0, vmax=0.0, shading="auto"
        )
        h.overlay_ryu_stack(
            ax,
            t_pgan_nm=100.0,
            Lx_um=Lx_map_um,
            dipole_pol=dpol,
            line_color="white",
            show_monitors=True,
            show_escape_cone=False,
            fontsize=6.3,
        )
        ax.set_title(title)
        ax.set_xlabel("Lateral coordinate $x$ [$\\mu$m]")
        fig_b.colorbar(im_b, ax=ax, label="Normalized $|\\mathbf{E}(\\omega_0)|^2$ [dB]")
    axes_b[0].set_ylabel("Vertical coordinate $z$ [$\\mu$m]")
    figstyle.finish(fig_b)
    p5b = fig_dir / "05b_field_intensity_pgan100.png"
    figstyle.save(fig_b, p5b)
    written.append(p5b)

    # ------------------------------------------------------------------
    # Fig 06: Real Electric Field Wavefronts Re(E_x) for TE vs Re(E_z) for TM
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.1, sharey=True)
    xg = d_ryu["t000_x"] * 1e6
    zg = d_ryu["t000_z"] * 1e6
    Lx_map_um = float(xg[-1] - xg[0])
    ex_re = np.real(d_ryu["t000_te_Ex"]).T
    ez_re = np.real(d_ryu["t000_tm_Ez"]).T
    vmax_te = float(np.percentile(np.abs(ex_re), 98.5))
    vmax_tm = float(np.percentile(np.abs(ez_re), 98.5))

    im0 = axes[0].pcolormesh(
        xg, zg, ex_re / vmax_te, cmap=figstyle.CMAP_DIVERGING, vmin=-1.0, vmax=1.0, shading="auto"
    )
    h.overlay_ryu_stack(
        axes[0],
        t_pgan_nm=0.0,
        Lx_um=Lx_map_um,
        dipole_pol="x",
        line_color="#222222",
        show_monitors=True,
        show_escape_cone=True,
        fontsize=6.3,
    )
    axes[0].set_title("(a) TE Wavefronts $\\mathrm{Re}[E_x(x, z, \\omega_0)]$ ($t_{\\mathrm{p-GaN}} = 0$)")
    axes[0].set_xlabel("Lateral coordinate $x$ [$\\mu$m]")
    axes[0].set_ylabel("Vertical coordinate $z$ [$\\mu$m]")
    fig.colorbar(im0, ax=axes[0], label="Normalized $\\mathrm{Re}(E_x)$")

    im1 = axes[1].pcolormesh(
        xg, zg, ez_re / vmax_tm, cmap=figstyle.CMAP_DIVERGING, vmin=-1.0, vmax=1.0, shading="auto"
    )
    h.overlay_ryu_stack(
        axes[1],
        t_pgan_nm=0.0,
        Lx_um=Lx_map_um,
        dipole_pol="z",
        line_color="#222222",
        show_monitors=True,
        show_escape_cone=False,
        fontsize=6.3,
    )
    axes[1].set_title("(b) TM Wavefronts $\\mathrm{Re}[E_z(x, z, \\omega_0)]$ (Guided Slab Modes)")
    axes[1].set_xlabel("Lateral coordinate $x$ [$\\mu$m]")
    fig.colorbar(im1, ax=axes[1], label="Normalized $\\mathrm{Re}(E_z)$")
    figstyle.finish(fig)
    p6 = fig_dir / "06_field_wavefronts_te_vs_tm.png"
    figstyle.save(fig, p6)
    written.append(p6)

    # ------------------------------------------------------------------
    # Fig 07: Animated Time-Domain Dipole Emission & TIR Trapping Movie (.gif)
    # ------------------------------------------------------------------
    # stored normalized (float16) with a movie_frames_scale factor to keep the committed data small
    frames = np.asarray(d_ryu["movie_frames"], dtype=float) * (
        float(d_ryu["movie_frames_scale"]) if "movie_frames_scale" in d_ryu else 1.0
    )
    frames_t = d_ryu["movie_t_fs"]
    xg_m = d_ryu["t000_x"] * 1e6
    zg_m = d_ryu["mat_z_node"][: frames.shape[2]] * 1e6
    Lx_m_um = float(xg_m[-1] - xg_m[0])
    f_scale = float(np.percentile(np.abs(frames), 99.2)) + 1e-30

    fig_m, ax_m = figstyle.new_figure(nrows=1, ncols=1, width=6.4, height=3.8)
    mesh_m = ax_m.pcolormesh(
        xg_m,
        zg_m,
        (frames[0].T) / f_scale,
        cmap=figstyle.CMAP_DIVERGING,
        vmin=-1.0,
        vmax=1.0,
        shading="nearest",
    )
    h.overlay_ryu_stack(
        ax_m,
        t_pgan_nm=50.0,
        Lx_um=Lx_m_um,
        dipole_pol="x",
        line_color="#222222",
        show_monitors=True,
        show_escape_cone=True,
        fontsize=6.2,
    )
    ax_m.set_xlabel("Lateral coordinate $x$ [$\\mu$m]")
    ax_m.set_ylabel("Vertical coordinate $z$ [$\\mu$m]")
    title_m = ax_m.set_title(
        f"TE Dipole Emission ($t_{{\\mathrm{{p-GaN}}}} = 50$ nm), $t = {frames_t[0]:.1f}$ fs"
    )
    fig_m.colorbar(mesh_m, ax=ax_m, label="Normalized $E_x(x, z, t)$")
    figstyle.finish(fig_m)

    def _update_frame(idx: int):
        mesh_m.set_array((frames[idx].T / f_scale).ravel())
        title_m.set_text(
            f"TE Dipole Emission ($t_{{\\mathrm{{p-GaN}}}} = 50$ nm), $t = {frames_t[idx]:.1f}$ fs"
        )
        return (mesh_m, title_m)

    anim = FuncAnimation(fig_m, _update_frame, frames=len(frames), interval=90, blit=False)
    p7 = fig_dir / "07_dipole_emission_movie.gif"
    anim.save(p7, writer=PillowWriter(fps=11), dpi=115)
    plt.close(fig_m)
    written.append(p7)

    # ------------------------------------------------------------------
    # Fig 08: LEE vs. p-GaN Thickness — Reproduction of Ryu (2014) Fig. 3
    # ------------------------------------------------------------------
    t_pgan = d_ryu["pgan_nm"]
    ana_ryu = h.ryu_beer_lambert_transfer(np.linspace(0.0, 200.0, 101))
    ref_pub = h.RYU_2014_REFERENCE

    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.0)
    ax = axes[0]
    ax.plot(t_pgan, d_ryu["lee_te_2d"] * 100.0, "o-", color=figstyle.PALETTE[2], lw=1.9, ms=5.5, label="Wakis 2D FIT: TE ($J_x$, direct 2D)")
    ax.plot(t_pgan, d_ryu["lee_tm_2d"] * 100.0, "s-", color=figstyle.PALETTE[1], lw=1.9, ms=5.5, label="Wakis 2D FIT: TM ($J_z$, direct 2D)")
    ax.plot(ana_ryu["t_pgan_nm"], ana_ryu["lee_2d_jx"] * 100.0, "--", color=figstyle.PALETTE[2], lw=1.5, alpha=0.8, label="Analytic 2D Beer-Lambert + Fresnel (TE)")
    ax.plot(t_pgan, d_ryu["lee_te_3d"] * 100.0, "^-", color=figstyle.PALETTE[0], lw=1.8, ms=5.5, label="Wakis 3D solid-angle weighted: TE")
    ax.plot(t_pgan, d_ryu["lee_tm_3d"] * 100.0, "v-", color=figstyle.PALETTE[3], lw=1.8, ms=5.5, label="Wakis 3D solid-angle weighted: TM")
    ax.plot(ref_pub["fig3_pgan_nm"], ref_pub["fig3_lee_te_3d_pct"], "D:", color=figstyle.PALETTE[0], ms=5.5, alpha=0.85, label="Ryu (2014) Fig. 3 (3D TE)")
    ax.plot(ref_pub["fig3_pgan_nm"], ref_pub["fig3_lee_tm_3d_pct"], "P:", color=figstyle.PALETTE[3], ms=5.5, alpha=0.85, label="Ryu (2014) Fig. 3 (3D TM)")
    ax.set_xlabel("p-GaN contact layer thickness $t_{\\mathrm{p-GaN}}$ [nm]")
    ax.set_ylabel("Top light extraction efficiency (LEE) [%]")
    ax.set_title("(a) Linear Scale: 2D vs. 3D Extraction & p-GaN Absorption")
    ax.legend(loc="upper right", fontsize=7.0)

    ax = axes[1]
    ax.semilogy(t_pgan, d_ryu["lee_te_2d"] * 100.0, "o-", color=figstyle.PALETTE[2], lw=1.9, label="Wakis 2D FIT: TE ($J_x$)")
    ax.semilogy(t_pgan, d_ryu["lee_tm_2d"] * 100.0, "s-", color=figstyle.PALETTE[1], lw=1.9, label="Wakis 2D FIT: TM ($J_z$)")
    ax.semilogy(t_pgan, d_ryu["lee_te_3d"] * 100.0, "^-", color=figstyle.PALETTE[0], lw=1.8, label="Wakis 3D-weighted: TE")
    ax.semilogy(t_pgan, d_ryu["lee_tm_3d"] * 100.0, "v-", color=figstyle.PALETTE[3], lw=1.8, label="Wakis 3D-weighted: TM")
    ax.semilogy(ref_pub["fig3_pgan_nm"], ref_pub["fig3_lee_te_3d_pct"], "D:", color=figstyle.PALETTE[0], ms=5.5, label="Ryu (2014) Fig. 3 (3D TE)")
    ax.semilogy(ref_pub["fig3_pgan_nm"], ref_pub["fig3_lee_tm_3d_pct"], "P:", color=figstyle.PALETTE[3], ms=5.5, label="Ryu (2014) Fig. 3 (3D TM)")
    ax.axhline(1.0, color="gray", ls="--", lw=1.1, label="1% 3D LEE threshold ($t_{\\mathrm{p-GaN}} \\approx 80$ nm)")
    ax.set_xlabel("p-GaN contact layer thickness $t_{\\mathrm{p-GaN}}$ [nm]")
    ax.set_ylabel("Top light extraction efficiency (LEE) [%, log scale]")
    ax.set_title("(b) Semi-Log Scale: Exponential Beer-Lambert Decay")
    ax.legend(loc="upper right", fontsize=7.0)
    figstyle.finish(fig)
    p8 = fig_dir / "08_lee_vs_pgan_ryu_fig3.png"
    figstyle.save(fig, p8)
    written.append(p8)

    # ------------------------------------------------------------------
    # Fig 09: 4-Channel Power Budget (TE vs TM across p-GaN Thickness)
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=3.9, sharey=True)
    for ax, pol_key, title in [
        (axes[0], "te", "(a) TE Dipole ($J_x$) 4-Channel Power Budget"),
        (axes[1], "tm", "(b) TM Dipole ($J_z$) 4-Channel Power Budget"),
    ]:
        top_c = d_ryu[f"lee_{pol_key}_2d"] * 100.0
        bot_c = d_ryu[f"bot_{pol_key}"] * 100.0
        sid_c = d_ryu[f"sides_{pol_key}"] * 100.0
        abs_c = d_ryu[f"abs_{pol_key}"] * 100.0
        ax.stackplot(
            t_pgan,
            top_c,
            bot_c,
            sid_c,
            abs_c,
            labels=[
                "Top air extraction ($\\Phi_{\\mathrm{top}} / P_{\\mathrm{inj}}$)",
                "Bottom sapphire leakage ($\\Phi_{\\mathrm{bot}} / P_{\\mathrm{inj}}$)",
                "Lateral guided slab modes ($\\Phi_{\\mathrm{sides}} / P_{\\mathrm{inj}}$)",
                "MQW + p-GaN absorption ($P_{\\mathrm{abs}} / P_{\\mathrm{inj}}$)",
            ],
            colors=[figstyle.PALETTE[2], figstyle.PALETTE[5], figstyle.PALETTE[6], figstyle.PALETTE[1]],
            alpha=0.86,
        )
        ax.set_xlabel("p-GaN contact layer thickness $t_{\\mathrm{p-GaN}}$ [nm]")
        ax.set_title(title)
        ax.set_xlim(0.0, 200.0)
        ax.set_ylim(0.0, 100.0)
    axes[0].set_ylabel("Share of total emitted dipole power [%]")
    axes[0].legend(loc="lower left", fontsize=7.4, framealpha=0.92)
    figstyle.finish(fig)
    p9 = fig_dir / "09_power_budget_4channel.png"
    figstyle.save(fig, p9)
    written.append(p9)

    # ------------------------------------------------------------------
    # Fig 10: Sensitivity to AlGaN Refractive Index (Ryu Fig. 7 Counterpart)
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=3.8)
    n_alg_arr = d_ryu["algan_indices"]
    ana_idx_te = np.array([h.escape_cone_2d_fresnel(float(nv), 1.0, "x") * 100.0 for nv in n_alg_arr])
    ana_idx_tm = np.array([h.escape_cone_2d_fresnel(float(nv), 1.0, "z") * 100.0 for nv in n_alg_arr])

    ax = axes[0]
    ax.plot(n_alg_arr, d_ryu["idx_te_t00"] * 100.0, "o-", color=figstyle.PALETTE[2], lw=1.9, ms=6.0, label="Wakis 2D TE ($t_{\\mathrm{p-GaN}} = 0$ nm)")
    ax.plot(n_alg_arr, d_ryu["idx_te_t50"] * 100.0, "s-", color=figstyle.PALETTE[0], lw=1.9, ms=6.0, label="Wakis 2D TE ($t_{\\mathrm{p-GaN}} = 50$ nm)")
    ax.plot(n_alg_arr, ana_idx_te, "--", color=figstyle.PALETTE[2], lw=1.5, label="Single-interface Fresnel $J_x$")
    ax.set_xlabel("AlGaN refractive index $n_{\\mathrm{AlGaN}}$")
    ax.set_ylabel("TE Extraction Efficiency [%]")
    ax.set_title("(a) TE Extraction vs. AlGaN Index $n_{\\mathrm{AlGaN}}$")
    ax.legend(loc="upper right", fontsize=7.6)

    ax = axes[1]
    ax.plot(n_alg_arr, d_ryu["idx_tm_t00"] * 100.0, "o-", color=figstyle.PALETTE[1], lw=1.9, ms=6.0, label="Wakis 2D TM ($t_{\\mathrm{p-GaN}} = 0$ nm)")
    ax.plot(n_alg_arr, d_ryu["idx_tm_t50"] * 100.0, "s-", color=figstyle.PALETTE[3], lw=1.9, ms=6.0, label="Wakis 2D TM ($t_{\\mathrm{p-GaN}} = 50$ nm)")
    ax.plot(n_alg_arr, ana_idx_tm, "--", color=figstyle.PALETTE[1], lw=1.5, label="Single-interface Fresnel $J_z$")
    ax.set_xlabel("AlGaN refractive index $n_{\\mathrm{AlGaN}}$")
    ax.set_ylabel("TM Extraction Efficiency [%]")
    ax.set_title("(b) TM Extraction vs. AlGaN Index (Ryu Fig. 7 Trend)")
    ax.legend(loc="upper right", fontsize=7.6)
    figstyle.finish(fig)
    p10 = fig_dir / "10_lee_vs_algan_index_ryu_fig7.png"
    figstyle.save(fig, p10)
    written.append(p10)

    # ------------------------------------------------------------------
    # Fig 11: Near2Far Far-Field Angular Pattern vs. Lambertian cos(theta)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(11.0, 4.0))
    ax_cart = fig.add_subplot(1, 2, 1)
    ax_pol = fig.add_subplot(1, 2, 2, projection="polar")

    th_deg_ff = d_inc["theta_deg"]
    th_rad_ff = np.radians(th_deg_ff)
    lamb = np.maximum(0.0, np.cos(th_rad_ff))

    te_ff = d_inc["d_eta_incoh_te"] / (np.max(d_inc["d_eta_incoh_te"]) + 1e-30)
    tm_ff = d_inc["d_eta_incoh_tm"] / (np.max(d_inc["d_eta_incoh_tm"]) + 1e-30)
    unp_ff = d_inc["d_eta_unpol"] / (np.max(d_inc["d_eta_unpol"]) + 1e-30)

    ax_cart.plot(th_deg_ff, lamb, "k--", lw=1.8, label="Ideal Lambertian $\\cos\\theta_{\\mathrm{air}}$")
    ax_cart.plot(th_deg_ff, te_ff, "-", color=figstyle.PALETTE[2], lw=2.0, label="Wakis Near2Far: Incoherent TE ($J_x$)")
    ax_cart.plot(th_deg_ff, tm_ff, "-", color=figstyle.PALETTE[1], lw=1.8, label="Wakis Near2Far: Incoherent TM ($J_z$, batwing)")
    ax_cart.plot(th_deg_ff, unp_ff, "-.", color=figstyle.PALETTE[0], lw=1.7, label="Wakis Near2Far: Unpolarized average")
    ax_cart.set_xlabel("Far-field polar angle in air $\\theta_{\\mathrm{air}}$ [deg]")
    ax_cart.set_ylabel("Normalized radiant intensity $I(\\theta_{\\mathrm{air}}) / I_{\\max}$")
    ax_cart.set_title("(a) Far-Field Pattern via `Near2FarMonitor`")
    ax_cart.set_xlim(-90.0, 90.0)
    ax_cart.set_ylim(0.0, 1.08)
    ax_cart.legend(loc="upper right", fontsize=7.4)

    from matplotlib.collections import PolyCollection

    ax_pol.add_collection(PolyCollection([[(0.0, 0.0), (0.0, 0.0)]], alpha=0.0))
    ax_pol.plot(th_rad_ff, lamb, "k--", lw=1.8, label="Lambertian")
    ax_pol.plot(th_rad_ff, te_ff, "-", color=figstyle.PALETTE[2], lw=2.0, label="TE ($J_x$)")
    ax_pol.plot(th_rad_ff, tm_ff, "-", color=figstyle.PALETTE[1], lw=1.8, label="TM ($J_z$)")
    ax_pol.set_theta_zero_location("N")
    ax_pol.set_theta_direction(-1)
    ax_pol.set_thetamin(-90)
    ax_pol.set_thetamax(90)
    ax_pol.set_title("(b) Polar Far-Field Radiation Diagram in Air", pad=12)
    ax_pol.legend(loc="lower right", fontsize=7.4)
    fig.tight_layout()
    p11 = fig_dir / "11_farfield_n2f_vs_lambertian.png"
    figstyle.save(fig, p11)
    written.append(p11)

    # ------------------------------------------------------------------
    # Fig 12: Incoherent Power Averaging vs. Coherent Field Summation (3 panels)
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.2, height=4.0)
    xg_i = d_inc["x_grid"] * 1e6
    zg_i = d_inc["z_grid"] * 1e6
    Lx_inc_um = float(xg_i[-1] - xg_i[0])

    I_ref_12 = float(np.percentile(d_inc["I_incoh_te"], 99.7))
    I_inc_db = np.clip(10.0 * np.log10(np.maximum(d_inc["I_incoh_te"].T / I_ref_12, 1e-4)), -28.0, 0.0)
    I_coh_db = np.clip(10.0 * np.log10(np.maximum(d_inc["I_coh_te"].T / I_ref_12, 1e-4)), -28.0, 0.0)

    im_a = axes[0].pcolormesh(
        xg_i, zg_i, I_inc_db, cmap=figstyle.CMAP_MAGNITUDE, vmin=-28.0, vmax=0.0, shading="auto"
    )
    h.overlay_ryu_stack(
        axes[0],
        t_pgan_nm=50.0,
        Lx_um=Lx_inc_um,
        dipole_pol="x",
        dipole_x_um=tuple(DIPOLE_X_POSITIONS_UM),
        line_color="white",
        show_monitors=True,
        show_escape_cone=False,
        fontsize=5.8,
    )
    axes[0].set_title("(a) Incoherent Sum $\\sum_i w_i |\\mathbf{E}_i|^2$ (Physical)")
    axes[0].set_xlabel("Lateral $x$ [$\\mu$m]")
    axes[0].set_ylabel("Vertical $z$ [$\\mu$m]")
    fig.colorbar(im_a, ax=axes[0], label="[dB]")

    im_b = axes[1].pcolormesh(
        xg_i, zg_i, I_coh_db, cmap=figstyle.CMAP_MAGNITUDE, vmin=-28.0, vmax=0.0, shading="auto"
    )
    h.overlay_ryu_stack(
        axes[1],
        t_pgan_nm=50.0,
        Lx_um=Lx_inc_um,
        dipole_pol="x",
        dipole_x_um=tuple(DIPOLE_X_POSITIONS_UM),
        line_color="white",
        show_monitors=True,
        show_escape_cone=False,
        fontsize=5.8,
    )
    axes[1].set_title("(b) Coherent Sum $|\\sum_i \\sqrt{w_i}\\mathbf{E}_i|^2$ (Fringes)")
    axes[1].set_xlabel("Lateral $x$ [$\\mu$m]")
    fig.colorbar(im_b, ax=axes[1], label="[dB]")

    ax_c = axes[2]
    n_eff = np.cumsum(d_inc["weights"]).astype(int)
    ax_c.plot(n_eff, d_inc["cum_lee_te"] * 100.0, "o-", color=figstyle.PALETTE[2], lw=1.9, ms=5.5, label="Cumulative incoherent TE")
    ax_c.plot(n_eff, d_inc["lee_te_each"] * 100.0, "s--", color=figstyle.PALETTE[5], lw=1.4, ms=4.5, label="Individual $x_i$ TE")
    ax_c.plot(n_eff, d_inc["cum_lee_unpol"] * 100.0, "^-", color=figstyle.PALETTE[0], lw=1.8, ms=5.5, label="Unpolarized ($\\frac{1}{2}$TE + $\\frac{1}{2}$TM)")
    ax_c.plot(n_eff, d_inc["cum_lee_tm"] * 100.0, "v-", color=figstyle.PALETTE[1], lw=1.8, ms=5.5, label="Cumulative incoherent TM")
    ax_c.set_xlabel("Cumulative dipole positions $N_{\\mathrm{pos}}$")
    ax_c.set_ylabel("Light extraction efficiency [%]")
    ax_c.set_title("(c) Convergence vs. $N_{\\mathrm{pos}}$")
    ax_c.legend(loc="center right", fontsize=7.0)
    figstyle.finish(fig)
    p12 = fig_dir / "12_incoherent_vs_coherent_convergence.png"
    figstyle.save(fig, p12)
    written.append(p12)

    # ------------------------------------------------------------------
    # Fig 13: Absorber Skin-Depth Resolution Study & Kottke vs. Staircase
    # ------------------------------------------------------------------
    cpsd_conv = np.array([float(d["cells_per_skin_depth"]) for d in d_convs])
    cpsd_ref = float(d_ref["cells_per_skin_depth"])
    cpsd_all = np.append(cpsd_conv, cpsd_ref)

    kot_te_25 = np.array([float(d["kottke_te_t25"]) for d in d_convs]) * 100.0
    sta_te_25 = np.array([float(d["stair_te_t25"]) for d in d_convs]) * 100.0
    kot_te_50 = np.array([float(d["kottke_te"]) for d in d_convs]) * 100.0
    sta_te_50 = np.array([float(d["stair_te"]) for d in d_convs]) * 100.0
    kot_te_100 = np.array([float(d["kottke_te_t100"]) for d in d_convs]) * 100.0
    sta_te_100 = np.array([float(d["stair_te_t100"]) for d in d_convs]) * 100.0

    ref_te_25 = float(d_ref["ref_te_t25"]) * 100.0
    ref_te_50 = float(d_ref["ref_te"]) * 100.0
    ref_te_100 = float(d_ref["ref_te_t100"]) * 100.0
    accepted_cpsd = float(cpsd_conv[-1])

    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.0)
    ax = axes[0]
    ax.plot(cpsd_all, np.append(kot_te_25, ref_te_25), "o-", color=figstyle.PALETTE[2], lw=1.9, ms=5.5, label="Kottke TE ($25$ nm)")
    ax.plot(cpsd_conv, sta_te_25, "s--", color=figstyle.PALETTE[2], lw=1.4, ms=5.0, alpha=0.75, label="Staircase TE ($25$ nm)")
    ax.plot(cpsd_all, np.append(kot_te_50, ref_te_50), "^-", color=figstyle.PALETTE[0], lw=1.9, ms=5.5, label="Kottke TE ($50$ nm)")
    ax.plot(cpsd_conv, sta_te_50, "v--", color=figstyle.PALETTE[0], lw=1.4, ms=5.0, alpha=0.75, label="Staircase TE ($50$ nm)")
    ax.plot(cpsd_all, np.append(kot_te_100, ref_te_100), "D-", color=figstyle.PALETTE[1], lw=1.9, ms=5.5, label="Kottke TE ($100$ nm)")
    ax.axvline(4.0, color=figstyle.PALETTE[3], ls=":", lw=1.5, label="Rule 3 min ($4$ cells/$\\delta$)")
    ax.axvline(accepted_cpsd, color=figstyle.PALETTE[2], ls="-.", lw=1.5, label=f"Accepted ({accepted_cpsd:.1f} cells/$\\delta$)")
    ax.set_xlabel("Cells per p-GaN power skin depth $\\delta / \\Delta z_{\\min}$ ($\\delta = 58.8$ nm)")
    ax.set_ylabel("2D TE Extraction Efficiency $\\eta_{\\mathrm{TE}}$ [%]")
    ax.set_title("(a) Target Metric vs. Cells per Skin Depth")
    ax.legend(loc="best", fontsize=7.0)

    ax = axes[1]
    err_kot_25 = np.maximum(np.abs(kot_te_25 - ref_te_25) / ref_te_25, 1e-4) * 100.0
    err_sta_25 = np.maximum(np.abs(sta_te_25 - ref_te_25) / ref_te_25, 1e-4) * 100.0
    err_kot_50 = np.maximum(np.abs(kot_te_50 - ref_te_50) / ref_te_50, 1e-4) * 100.0
    err_sta_50 = np.maximum(np.abs(sta_te_50 - ref_te_50) / ref_te_50, 1e-4) * 100.0
    err_kot_100 = np.maximum(np.abs(kot_te_100 - ref_te_100) / ref_te_100, 1e-4) * 100.0
    err_sta_100 = np.maximum(np.abs(sta_te_100 - ref_te_100) / ref_te_100, 1e-4) * 100.0

    ax.semilogy(cpsd_conv, err_kot_25, "o-", color=figstyle.PALETTE[2], lw=1.9, ms=5.5, label="Kottke TE ($25$ nm)")
    ax.semilogy(cpsd_conv, err_sta_25, "s--", color=figstyle.PALETTE[2], lw=1.4, ms=5.0, alpha=0.75, label="Staircase TE ($25$ nm)")
    ax.semilogy(cpsd_conv, err_kot_50, "^-", color=figstyle.PALETTE[0], lw=1.9, ms=5.5, label="Kottke TE ($50$ nm)")
    ax.semilogy(cpsd_conv, err_sta_50, "v--", color=figstyle.PALETTE[0], lw=1.4, ms=5.0, alpha=0.75, label="Staircase TE ($50$ nm)")
    ax.semilogy(cpsd_conv, err_kot_100, "D-", color=figstyle.PALETTE[1], lw=1.9, ms=5.5, label="Kottke TE ($100$ nm)")
    ax.semilogy(cpsd_conv, err_sta_100, "P--", color=figstyle.PALETTE[1], lw=1.4, ms=5.0, alpha=0.75, label="Staircase TE ($100$ nm)")
    ax.axhline(1.0, color="k", ls="--", lw=1.4, label="1% convergence threshold")
    ax.axvline(4.0, color=figstyle.PALETTE[3], ls=":", lw=1.5, label="Rule 3 min ($4$ cells/$\\delta$)")
    ax.axvline(accepted_cpsd, color=figstyle.PALETTE[2], ls="-.", lw=1.5, label=f"Accepted ({accepted_cpsd:.1f} cells/$\\delta$)")
    ax.set_xlabel("Cells per p-GaN power skin depth $\\delta / \\Delta z_{\\min}$")
    ax.set_ylabel(f"Relative error vs. reference ({cpsd_ref:.1f} cells/$\\delta$) [%]")
    ax.set_title("(b) Relative Error vs. Cells per Skin Depth")
    ax.legend(loc="best", fontsize=7.0, ncol=2)
    figstyle.finish(fig)
    p13 = fig_dir / "13_resolution_convergence_kottke_vs_staircase.png"
    figstyle.save(fig, p13)
    written.append(p13)

    # ------------------------------------------------------------------
    # Fig 14: Time-Domain Pulse Trace & Field Energy Decay
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=3.8)
    t_fs = d_ryu["time_fs"]
    src_tr = d_ryu["src_trace"]
    top_tr = d_ryu["top_trace"]
    en_tr = d_ryu["energy_trace"]

    ax = axes[0]
    ax.plot(t_fs, src_tr / (np.max(np.abs(src_tr)) + 1e-30), color=figstyle.PALETTE[2], lw=1.7, label="Source point $E_x(0, z_{\\mathrm{MQW}}, t)$")
    ax.plot(t_fs, top_tr / (np.max(np.abs(top_tr)) + 1e-30), color=figstyle.PALETTE[0], lw=1.7, alpha=0.85, label="Top air monitor $E_x(0, z_{\\mathrm{ext}}, t)$")
    ax.set_xlabel("Simulation time $t$ [fs]")
    ax.set_ylabel("Normalized electric field [arb. u.]")
    ax.set_title("(a) Dipole Excitation & Extracted Pulse Waveforms")
    ax.legend(loc="upper right", fontsize=7.6)

    ax = axes[1]
    en_norm = np.maximum(en_tr / (np.max(en_tr) + 1e-30), 1e-6)
    ax.semilogy(t_fs, en_norm, color=figstyle.PALETTE[1], lw=1.9, label="Normalized domain energy $U(t) / U_{\\max}$")
    ax.axhline(1e-3, color="k", ls="--", lw=1.1, label="Auto-shutoff threshold ($10^{-3}$ = $-30$ dB)")
    ax.set_xlabel("Simulation time $t$ [fs]")
    ax.set_ylabel("Normalized electromagnetic energy $U(t) / U_{\\max}$")
    ax.set_title("(b) Electromagnetic Energy Decay in Lossy Stack")
    ax.legend(loc="upper right", fontsize=7.6)
    figstyle.finish(fig)
    p14 = fig_dir / "14_time_trace_and_energy_decay.png"
    figstyle.save(fig, p14)
    written.append(p14)

    return written


# %% [markdown]
# ## 5. Quantitative Results (`--full` Study vs. `--quick`, Analytic, & Ryu 2014 Reference)
#
# ### 5.1 Semi-Infinite Lossless Slab Validation (Task 0: `--quick` $\text{cpw}=12$ vs. `--full` $\text{cpw}=20$ vs. Analytic)
#
# | Slab Index $n$ | Critical Angle $\theta_c$ | 2D Cone $\theta_c/\pi$ | 3D Cone $(1-\cos\theta_c)/2$ | $J_y$ (`--quick` / `--full` / Analytic) | $J_x$ TE (`--quick` / `--full` / Analytic) | $J_z$ TM (`--quick` / `--full` / Analytic) | Full Box Flux $\Phi_{\text{box}}/P_{\text{inj}}$ ($J_y / J_x / J_z$) |
# | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | $1.50$ | $41.81^\circ$ | $23.23\%$ | $12.73\%$ | $20.26\% / 20.71\% / 20.67\%$ | $35.45\% / 36.71\% / 37.76\%$ | $7.27\% / 7.23\% / 7.06\%$ | $100.37\% / 99.99\% / 99.97\%$ (mean $100.11\%$) |
# | $2.00$ | $30.00^\circ$ | $16.67\%$ | $6.70\%$ | $14.04\% / 14.17\% / 13.24\%$ | $32.41\% / 32.73\% / 28.11\%$ | $2.82\% / 2.79\% / 2.72\%$ | $100.21\% / 99.98\% / 99.99\%$ (mean $100.06\%$) |
# | $2.60$ | $22.62^\circ$ | $12.57\%$ | $3.85\%$ | $8.86\% / 8.89\% / 8.78\%$ | $21.51\% / 22.00\% / 20.61\%$ | $1.22\% / 1.20\% / 1.16\%$ | $88.90\% / 99.37\% / 76.65\%$ (mean $88.31\%$) |
#
# Refining from `--quick` ($12\;\text{cpw}$, $L_x = 2.8\;\mu\text{m}$) to `--full` ($20\;\text{cpw}$, $L_x = 3.6\;\mu\text{m}$)
# improves agreement with the exact 2D Fresnel integral at $n = 1.50$ (reducing the $J_y$ discrepancy from
# $-0.41\;\text{pp}$ to $+0.04\;\text{pp}$ and $J_x$ from $-2.31\;\text{pp}$ to $-1.05\;\text{pp}$) while keeping all
# $J_y$ and $J_z$ top-extraction values within $0.04\text{--}0.93\;\text{pp}$ of theory across $n \in \{1.5, 2.0, 2.6\}$.
# At $n = 2.60$ in `--full` ($L_x = 3.6\;\mu\text{m}$, lateral monitor half-width $w_{\text{ext}}/2 = 1.75\;\mu\text{m}$),
# the vertically beamed TE dipole ($J_x$) achieves $99.37\%$ closed-box flux conservation within the fixed $12\;\text{fs}$
# post-pulse window, whereas horizontally directed rays in the $n = 2.60$ slab ($v_g = c/2.60 = 0.115\;\mu\text{m/fs}$,
# requiring $15.2\;\text{fs}$ to reach $x = \pm 1.75\;\mu\text{m}$) are still partially in lateral transit when the monitor
# stops ($88.90\%$ for isotropic $J_y$ and $76.65\%$ for horizontally beamed $J_z \propto \sin^2\theta_1$). Because the upward
# escape cone ($\theta_c = 22.62^\circ$) crosses the top monitor at $z_{\text{ext}} = +10.7\;\text{nm}$ ($0.19\;\mu\text{m}$
# above the dipole, transit time $\sim 1.6\;\text{fs}$), the top extraction efficiency $\eta_{\text{top}}$ is already fully
# converged ($8.89\%$ vs. $8.78\%$ for $J_y$ and $1.20\%$ vs. $1.16\%$ for $J_z$).
#
# ### 5.2 Ryu (2014) Planar AlGaN UV LED Stack vs. p-GaN Thickness (Task 1: `--quick` vs. `--full` vs. Ryu 2014 Fig. 3)
#
# | $t_{\text{p-GaN}}$ [nm] | `--quick` 2D TE [%] | `--full` 2D TE [%] | `--quick` 3D TE [%] | `--full` 3D TE [%] | Ryu (2014) Fig. 3 TE [%] | Full 3D TE Residual ($\Delta\eta$ / Rel.) | `--quick` 2D TM [%] | `--full` 2D TM [%] | `--quick` 3D TM [%] | `--full` 3D TM [%] | Ryu (2014) Fig. 3 TM [%] | Full 3D TM Residual ($\Delta\eta$ / Rel.) |
# | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | $0$ | $22.50\%$ | $22.45\%$ | $4.79\%$ | $4.86\%$ | $4.00\%$ | $+0.86\;\text{pp}$ ($+21.6\%$) | $1.37\%$ | $1.41\%$ | $0.453\%$ | $0.466\%$ | $0.400\%$ | $+0.066\;\text{pp}$ ($+16.4\%$) |
# | $25$ | $11.07\%$ | $10.80\%$ | $2.32\%$ | $2.36\%$ | $2.60\%$ | $-0.24\;\text{pp}$ ($-9.3\%$) | $0.77\%$ | $0.81\%$ | $0.258\%$ | $0.269\%$ | $0.280\%$ | $-0.011\;\text{pp}$ ($-3.9\%$) |
# | $50$ | $8.15\%$ | $8.18\%$ | $1.72\%$ | $1.76\%$ | $1.70\%$ | $+0.06\;\text{pp}$ ($+3.8\%$) | $0.52\%$ | $0.55\%$ | $0.173\%$ | $0.182\%$ | $0.200\%$ | $-0.018\;\text{pp}$ ($-9.0\%$) |
# | $75$ | $4.77\%$ | $4.77\%$ | $1.00\%$ | $1.02\%$ | $1.10\%$ | $-0.08\;\text{pp}$ ($-7.2\%$) | $0.32\%$ | $0.34\%$ | $0.107\%$ | $0.113\%$ | $0.140\%$ | $-0.027\;\text{pp}$ ($-19.2\%$) |
# | $100$ | $3.25\%$ | $3.29\%$ | $0.68\%$ | $0.70\%$ | $0.72\%$ | $-0.02\;\text{pp}$ ($-2.3\%$) | $0.21\%$ | $0.22\%$ | $0.069\%$ | $0.074\%$ | $0.100\%$ | $-0.026\;\text{pp}$ ($-26.1\%$) |
# | $150$ | $1.32\%$ | $1.36\%$ | $0.27\%$ | $0.29\%$ | $0.31\%$ | $-0.02\;\text{pp}$ ($-7.5\%$) | $0.09\%$ | $0.09\%$ | $0.028\%$ | $0.030\%$ | $0.048\%$ | $-0.018\;\text{pp}$ ($-37.1\%$) |
# | $200$ | $0.54\%$ | $0.56\%$ | $0.11\%$ | $0.12\%$ | $0.13\%$ | $-0.01\;\text{pp}$ ($-9.1\%$) | $0.03\%$ | $0.04\%$ | $0.011\%$ | $0.012\%$ | $0.023\%$ | $-0.011\;\text{pp}$ ($-46.3\%$) |
#
# #### 4-Channel Power Budget in `--full` Mode ($\text{cpw} = 20$, $\Delta z_{\min} = 3.125\;\text{nm}$)
#
# | $t_{\text{p-GaN}}$ [nm] | 2D TE/TM Ratio | TE Power Budget: Top / Bot / Sides / Abs [%] | TM Power Budget: Top / Bot / Sides / Abs [%] |
# | :---: | :---: | :---: | :---: |
# | $0$ | $16.0$ | $22.5 / 45.4 / 30.5 / 1.7$ | $1.41 / 15.1 / 81.8 / 1.6$ |
# | $25$ | $13.3$ | $10.8 / 56.6 / 7.0 / 25.6$ | $0.81 / 11.4 / 45.0 / 42.9$ |
# | $50$ | $14.8$ | $8.2 / 42.9 / 3.9 / 45.1$ | $0.55 / 6.4 / 35.5 / 57.6$ |
# | $75$ | $14.0$ | $4.8 / 45.5 / 3.2 / 46.5$ | $0.34 / 8.5 / 37.2 / 53.9$ |
# | $100$ | $14.7$ | $3.3 / 42.4 / 3.1 / 51.2$ | $0.22 / 6.6 / 36.5 / 56.7$ |
# | $150$ | $14.9$ | $1.4 / 42.4 / 3.0 / 53.3$ | $0.09 / 7.0 / 36.1 / 56.8$ |
# | $200$ | $15.1$ | $0.6 / 42.6 / 3.0 / 53.8$ | $0.04 / 7.1 / 36.1 / 56.8$ |
#
# #### Residuals of 3D-Weighted Extraction vs. Ryu (2014) Fig. 3 and Their Physical Causes
#
# No model parameter was tuned to match Ryu (2014) Fig. 3; all layer indices, thicknesses, and
# absorption coefficients come directly from Section "Methods" of Ryu (2014) with a single dipole
# placed at the geometric center of the $50\;\text{nm}$ MQW ($z = -75\;\text{nm}$). The residuals
# at each p-GaN thickness arise from four specific physical mechanisms:
#
# 1. **$t_{\text{p-GaN}} = 0\;\text{nm}$ — Constructive Top-Surface Microcavity Phase & 2D p-Polarized Fresnel Boost
#    (TE: $+0.86\;\text{pp} / +21.6\%$; TM: $+0.066\;\text{pp} / +16.4\%$)**:
#    At $t_{\text{p-GaN}} = 0\;\text{nm}$, the $\text{p-AlGaN/Air}$ interface at $z = 0$ is lossless and sits
#    $d = 75\;\text{nm} = 0.696\,\lambda_{\text{AlGaN}}$ above the dipole ($\lambda_{\text{AlGaN}} = 280 / 2.60 = 107.7\;\text{nm}$),
#    placing a single dipole fixed at the MQW mid-plane near a constructive Fabry-Perot/Purcell interference
#    lobe in the upward escape cone. In Ryu's 3D multi-QW active model with $\Delta\lambda = 10\;\text{nm}$ spectral
#    width and a $2000\;\text{nm}$ n-AlGaN cladding, spatial averaging across the $50\;\text{nm}$ MQW thickness
#    ($\Delta d \approx 0.46\,\lambda_{\text{AlGaN}}$, nearly a full half-wave period) and spectral averaging damp this
#    standing-wave peak. In addition, an in-plane dipole $J_x$ in 2D ($x$-$z$) emits purely p-polarized waves with
#    Fresnel transmittance $T_p(\theta_1) \ge T_s(\theta_1)$, whereas a 3D horizontal dipole averages equally over
#    the p-polarized ($T_p\cos^2\theta_1$) and s-polarized ($T_s$) azimuths.
# 2. **$t_{\text{p-GaN}} = 25\;\text{nm}$ — Quarter-Wave Anti-Reflection Phase Flip
#    (TE: $-0.24\;\text{pp} / -9.3\%$; TM: $-0.011\;\text{pp} / -3.9\%$)**:
#    In p-GaN ($n = 2.90$) at $\lambda_0 = 280\;\text{nm}$, the quarter-wave thickness is
#    $\lambda_0 / (4 n_{\text{p-GaN}}) = 24.14\;\text{nm} \approx 25\;\text{nm}$. Adding a $25\;\text{nm}$ p-GaN cap
#    shifts the round-trip reflection phase from the top semiconductor/air boundary by
#    $\Delta\phi = 4\pi n_{\text{p-GaN}} t_{\text{p-GaN}} / \lambda_0 = 1.036\pi \approx 180^\circ$, flipping the single-plane
#    MQW dipole at $z = -75\;\text{nm}$ from constructive upward interference at $0\;\text{nm}$ (bottom sapphire leakage $45.4\%$)
#    to destructive upward interference at $25\;\text{nm}$ (redirecting power downward into the sapphire substrate, where
#    $\Phi_{\text{bot}}/P_{\text{inj}}$ jumps to $56.6\%$).
# 3. **$t_{\text{p-GaN}} = 50, 75, 100\;\text{nm}$ — Damped Cavity Regime with Weak Half-/Three-Quarter-Wave Modulation
#    (TE: $+0.06\;\text{pp} / +3.8\%$ at $50\;\text{nm}$, $-0.08\;\text{pp} / -7.2\%$ at $75\;\text{nm}$, $-0.02\;\text{pp} / -2.3\%$ at $100\;\text{nm}$;
#    TM: $-0.018\;\text{pp} / -9.0\%$ at $50\;\text{nm}$, $-0.027\;\text{pp} / -19.2\%$ at $75\;\text{nm}$, $-0.026\;\text{pp} / -26.1\%$ at $100\;\text{nm}$)**:
#    For $t_{\text{p-GaN}} \ge 50\;\text{nm}$, round-trip reflections through the absorbing p-GaN cap are attenuated by
#    $\exp(-2\alpha_{\text{p-GaN}} t_{\text{p-GaN}}) \le 18.3\%$ ($3.3\%$ at $100\;\text{nm}$), suppressing top-mirror cavity
#    oscillations. The small residual sign alternation in TE ($+3.8\%$ at the half-wave thickness $50\;\text{nm} \approx 2\lambda_{\text{p-GaN}}/4$,
#    $-7.2\%$ at the three-quarter-wave thickness $75\;\text{nm} \approx 3\lambda_{\text{p-GaN}}/4$, and $-2.3\%$ at the full-wave
#    thickness $100\;\text{nm} \approx \lambda_{\text{p-GaN}}$) reflects the decaying thin-film interference fringe.
# 4. **$t_{\text{p-GaN}} = 150, 200\;\text{nm}$ — Single-Pass Beer-Lambert Regime & Oblique TM Path-Length / Aperture Effects
#    (TE: $-0.02\;\text{pp} / -7.5\%$ at $150\;\text{nm}$, $-0.01\;\text{pp} / -9.1\%$ at $200\;\text{nm}$;
#    TM: $-0.018\;\text{pp} / -37.1\%$ at $150\;\text{nm}$, $-0.011\;\text{pp} / -46.3\%$ at $200\;\text{nm}$)**:
#    At $150\text{--}200\;\text{nm}$ ($\alpha t_{\text{p-GaN}} = 2.55\text{--}3.40$), TE matches Ryu (2014) Fig. 3 within
#    $0.01\text{--}0.02\;\text{pp}$. For TM ($J_z \propto \sin^2\theta_1$), escaping rays are concentrated near the critical
#    angle $\theta_c = 22.6^\circ$ ($\theta_{\text{p-GaN}} \approx 20.2^\circ$, $\cos\theta_{\text{p-GaN}} \approx 0.938$) rather
#    than normal incidence, increasing the oblique absorber path length by $1/\cos\theta_{\text{p-GaN}} \approx 1.066$
#    ($\alpha_{\text{eff, TM}} \approx 181{,}000\;\text{cm}^{-1}$; our analytic integral `ryu_beer_lambert_transfer` gives
#    $0.068\%, 0.027\%, 0.011\%$ at $100, 150, 200\;\text{nm}$, matching Wakis $0.074\%, 0.030\%, 0.012\%$). In addition,
#    because TM radiates into a wide-angle "batwing" lobe ($\theta_{\text{air}} \approx 50^\circ\text{--}85^\circ$, Figure 11),
#    raising the top `Near2FarMonitor` plane with $t_{\text{p-GaN}}$ ($h = z_{\text{ext}} - z_{\text{dip}}$ increasing from
#    $105\;\text{nm}$ to $305\;\text{nm}$ at fixed aperture width $w_{\text{ext}} = 3.50\;\mu\text{m}$) reduces the maximum
#    captured angle $\theta_{\max} = \arctan(w_{\text{ext}}/(2h))$ from $86.6^\circ$ to $80.1^\circ$, clipping the outermost
#    grazing tail on top of the $\sim 10\text{--}20\%$ digitization uncertainty of Ryu Fig. 3 at the $10^{-4}$ ($0.01\%\text{--}0.05\%$) level.
#
# ### 5.3 AlGaN Cladding Index Sensitivity ($n_{\text{AlGaN}} \in \{2.50, 2.60, 2.70\}$, Task 1)
#
# | $n_{\text{AlGaN}}$ | Critical Angle $\theta_c$ | Analytic $J_x$ / $J_z$ [%] | $t_{\text{p-GaN}}=0\;\text{nm}$ TE (`--quick` / `--full`) | $t_{\text{p-GaN}}=0\;\text{nm}$ TM (`--quick` / `--full`) | $t_{\text{p-GaN}}=50\;\text{nm}$ TE (`--quick` / `--full`) | $t_{\text{p-GaN}}=50\;\text{nm}$ TM (`--quick` / `--full`) |
# | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | $2.50$ | $23.58^\circ$ | $21.63\% / 1.31\%$ | $20.24\% / 20.68\%$ | $1.18\% / 1.18\%$ | $8.12\% / 8.43\%$ | $0.49\% / 0.48\%$ |
# | $2.60$ | $22.62^\circ$ | $20.61\% / 1.16\%$ | $22.50\% / 22.45\%$ | $1.37\% / 1.41\%$ | $8.15\% / 8.18\%$ | $0.52\% / 0.55\%$ |
# | $2.70$ | $21.74^\circ$ | $19.65\% / 1.02\%$ | $24.85\% / 25.40\%$ | $1.14\% / 1.14\%$ | $8.95\% / 9.35\%$ | $0.46\% / 0.45\%$ |
#
# Consistent with Ryu (2014) Fig. 7, increasing $n_{\text{AlGaN}}$ from $2.60$ to $2.70$ narrows the
# critical angle $\theta_c = \arcsin(1/n_{\text{AlGaN}})$ to $21.74^\circ$ and suppresses
# TM ($J_z$) top extraction from $1.41\%$ to $1.14\%$ at $t_{\text{p-GaN}} = 0\;\text{nm}$
# (and from $0.55\%$ to $0.45\%$ at $t_{\text{p-GaN}} = 50\;\text{nm}$). Because the
# $50\;\text{nm}$ AlGaN upper cladding forms a sub-wavelength ($\sim 0.46\,\lambda$)
# Fabry-Perot layer between the MQW/p-GaN interfaces and air, changing $n_{\text{AlGaN}}$ at
# fixed layer thickness also shifts the thin-film interference phase alongside the escape cone.
#
# ### 5.4 Incoherent Multi-Dipole Convergence (Task 2, $t_{\text{p-GaN}} = 50\;\text{nm}$)
#
# | Unique Lateral Positions $N_{\text{pos}}$ | Mirror-Symmetric Dipoles $N_{\text{dip}}$ | Individual Position TE / TM (`--full`) [%] | Cumulative Incoherent TE (`--quick` / `--full`) [%] | Cumulative Incoherent TM (`--quick` / `--full`) [%] | Unpolarized $\frac{1}{2}(\eta_{\text{TE}} + \eta_{\text{TM}})$ (`--quick` / `--full`) [%] |
# | :---: | :---: | :---: | :---: | :---: | :---: |
# | $1$ ($x_1 = 0.0\;\mu\text{m}$) | $1$ | $8.180\% / 0.554\%$ | $8.152\% / 8.180\%$ | $0.522\% / 0.554\%$ | $4.337\% / 4.367\%$ |
# | $2$ ($x \le 0.2\;\mu\text{m}$) | $3$ | $8.200\% / 0.546\%$ | $8.130\% / 8.193\%$ | $0.527\% / 0.549\%$ | $4.328\% / 4.371\%$ |
# | $3$ ($x \le 0.4\;\mu\text{m}$) | $5$ | $8.229\% / 0.547\%$ | $8.144\% / 8.207\%$ | $0.529\% / 0.548\%$ | $4.337\% / 4.378\%$ |
# | $4$ ($x \le 0.6\;\mu\text{m}$) | $7$ | $8.224\% / 0.550\%$ | $8.165\% / 8.212\%$ | $0.531\% / 0.549\%$ | $4.348\% / 4.380\%$ |
# | $5$ ($x \le 0.8\;\mu\text{m}$) | $9$ | $8.268\% / 0.547\%$ | $8.190\% / 8.225\%$ | $0.534\% / 0.548\%$ | $4.362\% / 4.386\%$ |
#
# Because the planar stack is horizontally invariant except near the lateral PML
# boundaries, the incoherently averaged extraction efficiency converges rapidly
# with the number of lateral dipole positions $N_{\text{pos}}$ across the active region:
# by $N_{\text{pos}} = 5$ unique positions ($9$ mirror-symmetric positions spanning
# $x \in [-0.8, +0.8]\;\mu\text{m}$), the cumulative incoherent TE LEE in `--full` mode stabilizes
# at $8.225\%$ (within $0.045$ percentage points of the single on-axis dipole value $8.180\%$, and
# $0.013\;\text{pp}$ between $N_{\text{pos}} = 4$ and $5$) while smoothing out finite-aperture ripples
# in the far-field pattern. By contrast, summing the complex electric fields coherently
# ($|\sum_i \sqrt{w_i}\mathbf{E}_i|^2$) creates unphysical multi-slit interference fringes in both the
# near-field intensity map and the far-field radiation pattern (Figure 12).
#
# ### 5.5 Absorber Skin-Depth Resolution Study & Kottke vs. Staircase (Tasks 3–6, $t_{\text{p-GaN}} \in \{25, 50, 100\}\;\text{nm}$)
#
# At $\lambda_0 = 280\;\text{nm}$, the p-GaN contact layer ($n = 2.90$, $\alpha = 170{,}000\;\text{cm}^{-1}$,
# $\kappa = \alpha\lambda_0 / (4\pi) = 0.3788$) has a power skin depth
# $$\delta = \frac{\lambda_0}{4\pi\kappa} = \frac{1}{\alpha} = 58.82\;\text{nm} \qquad (\text{field } 1/e \text{ amplitude skin depth } \delta_E = 2\delta = 117.65\;\text{nm}).$$
# Tasks 3–6 refine the vertical grid step $\Delta z_{\min}$ inside the active/p-GaN stack
# while snapping exact mesh nodes onto every layer interface via `graded_axis`:
#
# #### `--full` Convergence Study ($L_x = 3.6\;\mu\text{m}$, Reference at $\text{cpw} = 28$, $\Delta z_{\min} = 2.50\;\text{nm}$, $23.53\;\text{cells}/\delta$)
#
# | Task | $\text{cpw}$ | $\Delta x_{\text{bg}}$ [nm] | $\Delta z_{\min}$ [nm] | Cells / $25\;\text{nm}$ | Cells / $\delta$ | $25\;\text{nm}$ Kottke / Stair TE [%] (Rel. Err) | $50\;\text{nm}$ Kottke / Stair TE [%] (Rel. Err) | $100\;\text{nm}$ Kottke / Stair TE [%] (Rel. Err) | $50\;\text{nm}$ Kottke / Stair TM [%] (Rel. Err) |
# | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | 3 *(demo)* | $10$ | $9.63$ | $25.00^*$ | $1$ | $2.35$ | $8.551\% / 10.170\%$ ($20.83\% / 5.83\%$) | $4.720\% / 4.991\%$ ($42.48\% / 39.17\%$) | $1.134\% / 1.243\%$ ($65.67\% / 62.37\%$) | $0.230\% / 0.323\%$ ($58.61\%$) |
# | 4 | $14$ | $6.90$ | $4.17$ | $6$ | $14.12$ | $10.836\% / 11.516\%$ ($0.33\% / 6.63\%$) | $8.167\% / 8.790\%$ ($0.47\% / 7.13\%$) | $3.280\% / 3.510\%$ ($0.70\% / 6.27\%$) | $0.550\% / 0.592\%$ ($1.21\%$) |
# | **5 (Accepted)** | **$20$** | **$4.83$** | **$3.125$** | **$8$** | **$18.82$** | **$10.797\% / 10.931\%$ ($0.03\% / 1.21\%$)** | **$8.180\% / 8.179\%$ ($0.31\% / 0.32\%$)** | **$3.293\% / 3.297\%$ ($0.32\% / 0.18\%$)** | **$0.554\% / 0.552\%$ ($0.53\%$)** |
# | 6 (Ref.) | $28$ | $3.45$ | $2.50$ | $10$ | $23.53$ | $10.800\% / \text{—}$ (Ref.) | $8.205\% / \text{—}$ (Ref.) | $3.303\% / \text{—}$ (Ref.) | $0.557\% / \text{—}$ (Ref.) |
#
# *(Note: In Task 3 (`is_demo=True`), $\Delta z_{\text{p-GaN}} = 25.0\;\text{nm}$ across the absorber while $\min(\Delta x, \Delta z) = 9.63\;\text{nm}$; in `--quick` mode ($L_x = 2.8\;\mu\text{m}$, reference at $\text{cpw}=20$), the accepted $\text{cpw}=14$ level has relative differences of $0.50\%, 0.16\%, 0.30\%$ at $25, 50, 100\;\text{nm}$.)*
#
# Key findings from the `--full` absorber resolution study (Figure 13):
# - **Under-resolved demo (Task 3, $2.35\;\text{cells}/\delta$, $1$ cell across $25\;\text{nm}$)**:
#   When a $25\;\text{nm}$ p-GaN layer is spanned by only 1 cell ($\Delta z = 25\;\text{nm}$),
#   the exponential Beer-Lambert attenuation across the layer and the dual-cell interface
#   jump are severely under-resolved, causing a $20.8\%$ Kottke TE error at $t_{\text{p-GaN}} = 25\;\text{nm}$,
#   $42.5\%$ at $50\;\text{nm}$, $65.7\%$ at $100\;\text{nm}$, and $58.6\%$ for TM at $50\;\text{nm}$.
# - **Intermediate compliant grid (Task 4, $14\;\text{cpw}$, $14.12\;\text{cells}/\delta$, $6$ cells across $25\;\text{nm}$)**:
#   At $\Delta z_{\min} = 4.17\;\text{nm}$, Kottke subpixel smoothing is already within **$0.33\%$ ($25\;\text{nm}$),
#   $0.47\%$ ($50\;\text{nm}$), and $0.70\%$ ($100\;\text{nm}$)** of the $28\;\text{cpw}$ reference, whereas staircase
#   discretization (which rounds $w = 0.5 \to 1.0$ at the interface tangential node, shifting the effective electrical
#   thickness by half a cell) still exhibits **$6.63\%, 7.13\%, \text{ and } 6.27\%$** error!
# - **Accepted `--full` grid level (Task 5, $20\;\text{cpw}$, $18.82\;\text{cells}/\delta$, $8$ cells across $25\;\text{nm}$)**:
#   At $\Delta z_{\min} = 3.125\;\text{nm}$, the relative change between the last two levels ($\text{cpw}=20$ accepted vs.
#   $\text{cpw}=28$, $23.53\;\text{cells}/\delta$ reference) is **$0.03\%$ at $25\;\text{nm}$, $0.31\%$ at $50\;\text{nm}$,
#   $0.32\%$ at $100\;\text{nm}$ for 2D TE (and $0.72\%$ at $25\;\text{nm}$, $0.90\%$ at $100\;\text{nm}$ for 3D-weighted TE),
#   and $0.53\%$ for 2D TM at $50\;\text{nm}$** — strictly below the $< 1\%$ convergence criterion across all metrics.
#
# ### 5.6 Comparison Against an Un-Snapped Uniform 10 cpw Baseline
#
# On an un-snapped uniform baseline grid (`cpw = 10`, $\Delta z = 9.66\;\text{nm}$),
# the 3D-weighted TE extraction is $30\text{--}60\%$ below Ryu (2014) Fig. 3 for
# $t_{\text{p-GaN}} \ge 25\;\text{nm}$ ($1.69\%$ vs. $2.60\%$ at $25\;\text{nm}$ and
# $0.50\%$ vs. $0.72\%$ at $100\;\text{nm}$) while appearing close at $0\;\text{nm}$
# ($4.38\%$ vs. $4.00\%$). Four physical and numerical mechanisms govern the difference
# between the uniform baseline and the interface-snapped graded grid (`--full`, $20\;\text{cpw}$):
#
# 1. **2D-to-3D Solid-Angle Weighting Normalization**:
#    The `Near2FarMonitor` angular distribution $d\eta_{\text{2D}}/d\theta_{\text{air}}$ on
#    $\theta_{\text{air}} \in (-\pi/2, +\pi/2)$ is symmetric across $\pm\theta_{\text{air}}$
#    and normalized by $\frac{1}{\pi}\cos^2\theta_1$ (TE) or $\frac{1}{\pi}\sin^2\theta_1$ (TM).
#    Converting this folded 2D integral over $[-\theta_c, +\theta_c]$ to the 3D upper-hemisphere
#    solid-angle integral $\int_0^{\theta_c} p_{\text{3D}}(\theta_1)\sin\theta_1\,d\theta_1$
#    with $p_{\text{3D, TE}} = \frac{3}{8}(1 + \cos^2\theta_1)$ and
#    $p_{\text{3D, TM}} = \frac{3}{4}\sin^2\theta_1$ requires an exact prefactor of
#    $\frac{3}{8} = 0.375$ for both polarizations. On the interface-snapped graded grid
#    (`--full`, $20\;\text{cpw}$) with exact slice $(\varepsilon_{\text{eff}}, \sigma_{\text{eff}})$
#    and $26\;\text{fs}$ guided-mode transit time, $t_{\text{p-GaN}} = 0\;\text{nm}$ has $22.45\%$
#    2D TE LEE and $1.7\%$ MQW absorption, and the exact $\frac{3}{8}$-weighted 3D TE curve
#    matches Ryu (2014) Fig. 3 closely across $t_{\text{p-GaN}} = 50\text{--}200\;\text{nm}$
#    ($1.76\%$ vs. $1.70\%$ at $50\;\text{nm}$, $1.02\%$ vs. $1.10\%$ at $75\;\text{nm}$,
#    $0.70\%$ vs. $0.72\%$ at $100\;\text{nm}$, $0.29\%$ vs. $0.31\%$ at $150\;\text{nm}$,
#    $0.12\%$ vs. $0.13\%$ at $200\;\text{nm}$).
# 2. **Quarter-Wave Microcavity Standing-Wave Phase at $t_{\text{p-GaN}} = 25\;\text{nm}$ ($2.36\%$ vs. $2.60\%$)**:
#    At $\lambda_0 = 280\;\text{nm}$ in p-GaN ($n = 2.90$), the quarter-wave optical thickness is
#    $$\frac{\lambda_0}{4 n_{\text{p-GaN}}} = \frac{280\;\text{nm}}{4 \times 2.90} = 24.14\;\text{nm} \approx 25\;\text{nm}.$$
#    Adding a $25\;\text{nm}$ p-GaN cap shifts the round-trip phase from the top semiconductor/air
#    interface by $\Delta\phi = 4\pi n_{\text{p-GaN}} t_{\text{p-GaN}} / \lambda_0 = 1.036\pi \approx 180^\circ$
#    for a single dipole fixed at $z = -75\;\text{nm}$ (MQW center). This switches the top mirror
#    from constructive interference at $0\;\text{nm}$ (bottom sapphire flux $45.4\%$) to destructive
#    anti-reflection at $25\;\text{nm}$ (redirecting dipole emission downward so bottom sapphire
#    flux jumps to $56.6\%$), yielding $2.36\%$ 3D TE vs. Ryu's $2.60\%$ (where 3D multi-QW active
#    averaging damps the 1D Fabry-Perot fringe).
# 3. **Grid Resolution ($\le 0.9\%$ effect on 3D TE once Rules 1–3 are satisfied)**:
#    Refining $t_{\text{p-GaN}} = 25\;\text{nm}$ in `--full` from $14.12\;\text{cells}/\delta$ ($6$ cells/$25\;\text{nm}$,
#    $\text{cpw}=14$) to $18.82\;\text{cells}/\delta$ ($8$ cells, $\text{cpw}=20$) to $23.53\;\text{cells}/\delta$
#    ($10$ cells, $\text{cpw}=28$) moves 3D TE LEE by only $2.323\% \to 2.359\% \to 2.376\%$ (and 2D TE by
#    $10.836\% \to 10.797\% \to 10.800\%$), and at $t_{\text{p-GaN}} = 100\;\text{nm}$ moves 3D TE LEE by
#    $0.692\% \to 0.704\% \to 0.710\%$ (2D TE: $3.280\% \to 3.293\% \to 3.303\%$).
# 4. **Lateral Simulation Width $L_x$ ($< 2.5\%$ effect between $L_x = 2.8\;\mu\text{m}$ and $3.6\;\mu\text{m}$)**:
#    Widening the domain from $L_x = 2.8\;\mu\text{m}$ (`--quick`) to $L_x = 3.6\;\mu\text{m}$ (`--full`) at fixed
#    $20\;\text{cpw}$ resolution changes 2D TE extraction by $< 1.6\%$ relative ($10.975\% \to 10.797\%$ at $25\;\text{nm}$,
#    $8.257\% \to 8.180\%$ at $50\;\text{nm}$, $3.318\% \to 3.293\%$ at $100\;\text{nm}$), because the top escape cone
#    ($\theta_c = 22.6^\circ$) emerges within $|x| = h\tan\theta_c \lesssim 0.12\;\mu\text{m}$ of the dipole axis.
#
# %% [markdown]
# ## 6. Limits & Failure Modes
#
# 1. **2D ($\theta_c/\pi$) vs. 3D ($\frac{1}{2}(1-\cos\theta_c) \approx \frac{1}{4n^2}$) Dimensionality Caveat**:
#    A 2D $(x, z)$ simulation integrates uniformly over planar angle $d\theta_1 / \pi$,
#    missing the 3D azimuthal ring Jacobian $\frac{1}{2}\sin\theta_1\,d\theta_1$. Consequently,
#    unweighted 2D extraction efficiencies ($22.45\%$ for TE at $t_{\text{p-GaN}} = 0\;\text{nm}$) are
#    larger than 3D extraction efficiencies ($\sim 4.0\text{--}4.9\%$) by $\mathcal{O}(n)$. Applying
#    the exact 3D dipole solid-angle weight ($\text{pol\_3d\_factor} = \frac{3}{8}$) and
#    Snell-law internal Jacobian $\sin\theta_1 = \sin|\theta_{\text{air}}|/n_{\text{AlGaN}}$
#    to the 2D `Near2FarMonitor` angular distribution recovers the 3D scale ($4.86\%$ TE
#    and $0.466\%$ TM at $t_{\text{p-GaN}} = 0\;\text{nm}$, and $1.76\%$ TE vs. $1.70\%$ in
#    Ryu Fig. 3 at $t_{\text{p-GaN}} = 50\;\text{nm}$), while full 3D nanorod patterning
#    requires a 3D domain.
# 2. **Unreported Lateral Box Size in Ryu (2014)**:
#    Ryu (2014) Fig. 1a illustrates the planar LED FDTD domain and top detection line
#    (dotted red line in air), but the paper text does not specify the exact lateral width
#    $L_x$ of the planar simulation box. Any finite open detection aperture of half-width
#    $L_x/2$ at height $h$ above the dipole truncates grazing rays beyond
#    $\theta_{\max} = \arctan(L_x / (2h))$; here $L_x = 3.6\;\mu\text{m}$ (`--full`) and
#    $2.8\;\mu\text{m}$ (`--quick`) at $h \approx 0.18\;\mu\text{m}$ capture $\theta_{\max} > 82^\circ$
#    ($> 98\%$ of the upward hemisphere for TE, while clipping the outermost $>80^\circ\text{--}86^\circ$
#    grazing tail of the TM batwing lobe at thick $t_{\text{p-GaN}}$).
# 3. **Evanescent Near-Field Contamination of `Near2FarMonitor` if Placed Too Close**:
#    Rays at $|\theta_1| > \theta_c$ undergo total internal reflection and excite
#    evanescent tails in air decaying as $\exp(-2 k_0 \sqrt{n_{\text{AlGaN}}^2 \sin^2\theta_1 - 1}\,z)$.
#    If a `Near2FarMonitor` plane is placed within $<\lambda_0 / 10$ of the semiconductor
#    surface, the Chu-Stratton integral over the finite aperture diffracts those
#    evanescent surface currents into spurious grazing far-field lobes. Placing the
#    monitor $\ge 3\Delta z$ ($\sim 30\;\text{nm}$) above the top interface avoids
#    severe evanescent contamination while keeping the air buffer compact.
# 4. **Point-Dipole Near-Field Singularity in Absorbing Media**:
#    A point or line dipole placed inside a medium with non-zero conductivity $\sigma > 0$
#    experiences local non-radiative ohmic dissipation proportional to $\sigma / \Delta z^2$
#    from its quasi-static $1/r^2$ near field as $\Delta z \to 0$. Clearing a tiny lossless
#    neighborhood ($r_{\text{clear}} = 22\;\text{nm} < \lambda_{\min}/4$) around the point dipole
#    inside the weakly absorbing MQW ($z < -25\;\text{nm}$) while keeping all p-AlGaN and
#    p-GaN absorber layers untouched eliminates grid-divergent self-quenching so the mesh
#    converges cleanly under vertical refinement.
#
# %% [markdown]
# ## 7. Exercises
#
# 1. **Flip-Chip Substrate Extraction (Sapphire vs. Air)**:
#    In a flip-chip UV LED, the absorbing p-GaN layer is bonded to a mirror/submount and
#    light is extracted downward through the transparent sapphire substrate ($n_{\text{sap}} = 1.80$).
#    Use `h.escape_cone_2d_fresnel(2.60, 1.80, dipole="x")` and `h.escape_cone_3d_fresnel(2.60, 1.80, dipole="te")`
#    to compute the critical angle $\theta_c = \arcsin(1.80 / 2.60) = 43.8^\circ$ and compare
#    it with the bottom sapphire flux fraction $\Phi_{\text{bot}} / P_{\text{inj}} \approx 45.4\%$
#    measured in Task 1 at $t_{\text{p-GaN}} = 0\;\text{nm}$.
# 2. **Dipole Depth Standing-Wave (Purcell) Oscillations**:
#    Vary the vertical dipole position `z_dipole` across the $50\;\text{nm}$ MQW region
#    ($z \in [-95, -55]\;\text{nm}$) at $t_{\text{p-GaN}} = 0\;\text{nm}$ and $25\;\text{nm}$
#    and plot how constructive/destructive interference with the top reflection modulates
#    total injected power $P_{\text{inj}}$ and top extraction efficiency.
# 3. **Subwavelength Surface Roughness / Texturing**:
#    Add a periodic array of dielectric nanocones or boxes on top of the p-AlGaN layer
#    using `wakis.templates.Box` to scatter trapped TIR guided modes into the air escape
#    cone and measure the enhancement factor in TM extraction efficiency.
#
# %% [markdown]
# ## 8. References
#
# 1. H.-Y. Ryu, "Large enhancement of light extraction efficiency in AlGaN-based
#    nanorod ultraviolet light-emitting diode structures," *Nanoscale Research Letters*
#    **9**, 58 (2014), [doi:10.1186/1556-276X-9-58](https://doi.org/10.1186/1556-276X-9-58).
# 2. C. Kottke, A. Farjadpour, and S. G. Johnson, "Perturbation theory for anisotropic
#    dielectric interfaces, and application to subpixel smoothing of discretized
#    numerical methods," *Physical Review E* **77**, 036611 (2008),
#    [doi:10.1103/PhysRevE.77.036611](https://doi.org/10.1103/PhysRevE.77.036611).
# 3. E. F. Schubert, *Light-Emitting Diodes*, 2nd ed. (Cambridge University Press, 2006),
#    Ch. 5 & Ch. 9 (Escape cones, Snell's law, and planar extraction efficiency).
# 4. M. Kneissl, T.-Y. Seong, J. Han, and H. Amano, "The emergence and prospects of
#    deep-ultraviolet light-emitting diode technologies," *Nature Photonics* **13**,
#    233–244 (2019), [doi:10.1038/s41566-019-0359-9](https://doi.org/10.1038/s41566-019-0359-9).
#
#
# %%
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tutorial 7.1 · Planar deep-UV LED: escape cone, total internal reflection, TE vs TM dipoles"
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the parallelizable study task table and exit.",
    )
    parser.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single task index from --list-tasks.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR,
        help="Output directory for .npz task data.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick resolution (12 cpw graded, reference 20 cpw).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full resolution (20 cpw graded, reference 28 cpw).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render all figures in figures/ from existing data in --out.",
    )
    args = parser.parse_args(argv)

    if args.full:
        mode = "full"
    elif args.quick:
        mode = "quick"
    elif args.plot and all(
        (args.out / t["file"]).exists() for t in _task_table("full")
    ):
        mode = "full"
    else:
        mode = "quick"
    tasks = _task_table(mode)

    if args.list_tasks:
        for t in tasks:
            print(f"{t['index']}\t{t['label']}\t{t['file']}")
        return 0

    if args.plot and args.task is None:
        written = plot_all(args.out, FIG_DIR, mode=mode)
        print(f"Rendered {len(written)} figures in {FIG_DIR}")
        return 0

    if args.task is not None:
        selected = [t for t in tasks if t["index"] == args.task]
        if not selected:
            raise SystemExit(
                f"Unknown --task {args.task}; valid indices: 0..{len(tasks) - 1}"
            )
        p = run_task(selected[0], args.out)
        print(f"Finished task {args.task} -> {p}")
        return 0

    # Default when run with --quick or --full without --task: run all tasks + plot
    t_all = time.perf_counter()
    for t in tasks:
        p = run_task(t, args.out)
        print(f"[{t['index']}/{len(tasks) - 1}] wrote {p.name}")
    written = plot_all(args.out, FIG_DIR, mode=mode)
    print(
        f"Completed all {len(tasks)} tasks and rendered {len(written)} figures in {time.perf_counter() - t_all:.1f} s"
    )
    return 0


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        if all((DATA_DIR / t["file"]).exists() for t in _task_table("full")):
            plot_all(DATA_DIR, FIG_DIR, mode="full")
        elif all((DATA_DIR / t["file"]).exists() for t in _task_table("quick")):
            plot_all(DATA_DIR, FIG_DIR, mode="quick")
    else:
        raise SystemExit(main())
