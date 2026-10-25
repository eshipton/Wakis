# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

# %% [markdown]
# # Tutorial 7.3 · MicroLED with a graded-index micro-horn collimator
#
# **Abstract.** Sub-micrometre $\text{InGaN}/\text{GaN}$ micro-LEDs ($\mu\text{LEDs}$) are leading light-engine candidates for near-eye augmented-reality (AR) waveguide combiners, holographic displays, and visible-light communication, yet planar nitride pixels suffer from severe total internal reflection ($\theta_c = \arcsin(1/n_{\text{GaN}}) \approx 24.1^\circ$ at $\lambda_0 = 450\;\text{nm}$) and broad Lambertian emission that couples poorly into narrow-acceptance ($\Gamma = \pm 15^\circ$) diffractive waveguides. Recently, **Luce, Alaee, and Abass (*Scientific Reports* 16, 7391, 2026)** demonstrated that integrating a metallic micro-horn ($\mu\text{Horn}$) collimator filled with a 6-layer descending gradient-index ($\text{dGRIN}$) dielectric stack ($\text{Nb}_2\text{O}_5 / \text{Si}_3\text{N}_4 / \text{Al}_2\text{O}_3 / \text{SiO}_2 / \text{MgF}_2 / \text{Air}$) simultaneously breaks total internal reflection at the semiconductor aperture and bends oblique rays into the slanted silver ($\text{Ag}$) sidewalls, redirecting them into a collimated forward beam. In this tutorial we reproduce the native 2D finite-difference time-domain (FDTD) proof-of-principle study of Luce *et al.* (2026) using Wakis's 2D **Finite Integration Technique (FIT)** solver with **Drude dispersive silver**, **Kottke subpixel smoothing** on sloped metal–dielectric sidewalls, **2D cylindrical near-to-far-field transformations (`Near2FarMonitor2D`)**, and **incoherent dipole ensemble averaging** over quantum-well positions, polarizations, and emission spectra.
#
# ---
#
# ## Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. Model a dispersive metallic-clad semiconductor emitter in 2D FIT by coupling a calibrated 1-pole **Drude silver ($\text{Ag}$)** dispersion (`wakis.dispersive.DispersiveMaterial`) with **Kottke (2008) subpixel tensor averaging** (`wakis.tensor_dispersive`) across sloped `PolySlab` horn sidewalls.
# 2. Compute incoherent spontaneous emission from an active quantum-well (QW) plane by averaging **radiated powers**—never fields—over dipole lateral positions $x_d$, orientations $(J_x, J_y, J_z)$, and a Gaussian $\text{InGaN}$ electroluminescence spectrum ($\lambda_0 = 450\;\text{nm}$, $\text{FWHM} = 25\;\text{nm}$).
# 3. Extract 2D cylindrical far-field radiation patterns $\rho_{\text{LEE}}(\theta)$, forward cone extraction efficiencies $\text{LEE}_{15}$ ($\pm 15^\circ$), total extraction efficiencies $\text{LEE}_{90}$ ($\pm 90^\circ$), collimation ratios $\eta_{15} = \text{LEE}_{15}/\text{LEE}_{90}$, and cavity Purcell factors $F_P$ using `Near2FarMonitor2D`.
# 4. Compare our 2D FIT model against the 2D FDTD benchmark of **Luce *et al.* (2026)**—evaluating Bare, Air-filled ($\vartheta = 72^\circ$), $\text{SiO}_2$-filled ($\vartheta = 64^\circ$), and 6-layer $\text{dGRIN}$-filled ($\vartheta = 60^\circ$) micro-horns, sidewall angle and height sweeps (cf. Fig. 3), internal Poynting power-flow redirection (Fig. 3g–i), 2D quantum-well vertical offset sensitivity, dipole-ensemble convergence ($M \in \{1, 5, 9, 17\}$), and silver optical-constant models (Rakić 1998 vs Johnson & Christy 1972)—and quantify the convergence advantage of Kottke subpixel smoothing over staircasing on sloped metallic reflectors.
#
# ## Contents
#
# 1. [Physical background & paper specification](#1.-Physical-background-&-paper-specification)
# 2. [Analytical baselines: escape cone, TMM & 2D dipole power](#2.-Analytical-baselines:-escape-cone,-TMM-&-2D-dipole-power)
# 3. [2D FIT solver setup, dispersive Ag & Kottke sidewall smoothing](#3.-2D-FIT-solver-setup,-dispersive-Ag-&-Kottke-sidewall-smoothing)
# 4. [Study tasks & execution modes](#4.-Study-tasks-&-execution-modes)
# 5. [Generating the complete 15-figure suite](#5.-Generating-the-complete-15-figure-suite)
# 6. [Results & quantitative comparison with Luce *et al.* (2026)](#6.-Results-&-quantitative-comparison-with-Luce-et-al.-(2026))
# 7. [Resolution convergence: Kottke vs staircase on sloped metal sidewalls](#7.-Resolution-convergence:-Kottke-vs-staircase-on-sloped-metal-sidewalls)
# 8. [Limits, failure modes & investigation of the gaps vs Luce *et al.* (2026)](#8.-Limits,-failure-modes-&-investigation-of-the-gaps-vs-Luce-et-al.-(2026))
# 9. [Exercises](#9.-Exercises)
# 10. [References](#10.-References)

# %% [markdown]
# ## 1. Physical background & paper specification
#
# ### 1.1 The extraction and collimation bottleneck in sub-micron $\mu\text{LEDs}$
#
# The external quantum efficiency ($\text{EQE}$) of a micro-LED coupled into an optical system with acceptance half-angle $\Gamma$ (typically $\Gamma = 15^\circ$ for diffractive AR waveguide incouplers) depends on the internal quantum efficiency $\eta_{\text{IQE}}$, the total light extraction efficiency $\text{LEE}_{90}$ into the upper air hemisphere ($|\theta| \le 90^\circ$), and the directional collimation efficiency $\eta_\Gamma = \text{LEE}_\Gamma / \text{LEE}_{90}$ [1]:
#
# $$\text{LEE}_\Gamma = \int_{-\Gamma}^{+\Gamma} \rho_{\text{LEE}}(\theta)\,\mathrm{d}\theta, \qquad \text{LEE}_{90} = \int_{-90^\circ}^{+90^\circ} \rho_{\text{LEE}}(\theta)\,\mathrm{d}\theta, \qquad \eta_\Gamma = \frac{\text{LEE}_\Gamma}{\text{LEE}_{90}}.$$
#
# In a planar nitride emitter ($n_{\text{GaN}} = 2.45$ at $\lambda_0 = 450\;\text{nm}$) radiating into air ($n_{\text{air}} = 1.00$), Snell's law restricts direct transmission to internal angles below the critical angle:
#
# $$\theta_c = \arcsin\!\left(\frac{n_{\text{air}}}{n_{\text{GaN}}}\right) = \arcsin\!\left(\frac{1.00}{2.45}\right) \approx 24.09^\circ.$$
#
# Even with an ideal back reflector, rays launched at $|\theta_{\text{in}}| > \theta_c$ undergo total internal reflection (TIR) and remain trapped in guided slab modes until absorbed by the metallic contacts. Furthermore, the rays that do escape refract across the planar interface into a broad Lambertian-like angular distribution $I(\theta) \propto \cos\theta$, of which only $\sin(15^\circ) \approx 25.9\%$ in 2D ($\sin^2(15^\circ) \approx 6.7\%$ in 3D) falls within the $\pm 15^\circ$ acceptance cone.
#
# ### 1.2 Device geometry of Luce *et al.* (2026)
#
# To overcome both TIR trapping and angular divergence within a compact pixel footprint ($w_{\text{top}} \le 4.5\;\mu\text{m}$), **Luce, Alaee, and Abass (2026)** [1] placed a tapered silver ($\text{Ag}$) micro-horn ($\mu\text{Horn}$) collimator atop a $1.0\;\mu\text{m}$ $\text{GaN}$ pixel (Fig. 2 of [1]):
#
# - **Active $\text{GaN}$ mesa**: width $w_{\text{GaN}} = 0.60\;\mu\text{m}$ ($x \in [-0.30, +0.30]\;\mu\text{m}$), thickness $t_{\text{GaN}} = 0.30\;\mu\text{m}$ ($y \in [0, 0.30]\;\mu\text{m}$), refractive index $n_{\text{GaN}} = 2.45$.
# - **Quantum-well (QW) emission plane**: located at the vertical mid-plane $y_{\text{QW}} = 0.15\;\mu\text{m}$ ($150\;\text{nm}$ above the bottom $\text{Ag}$ mirror), modeled as an incoherent ensemble of dipoles distributed across $x \in (-0.30, +0.30)\;\mu\text{m}$ with a Gaussian electroluminescence spectrum centered at $\lambda_0 = 450\;\text{nm}$ ($\text{FWHM} = 25\;\text{nm}$, band $\Lambda \in [425, 475]\;\text{nm}$).
# - **Sidewall passivation**: lateral $\text{SiO}_2$ barriers ($n_{\text{SiO}_2} = 1.46$) of width $w_{\text{pass}} = 0.20\;\mu\text{m}$ on both sides of the $\text{GaN}$ core, yielding a total bottom pixel opening of $w_{\text{pixel}} = w_{\text{GaN}} + 2 w_{\text{pass}} = 1.00\;\mu\text{m}$.
# - **Silver ($\text{Ag}$) substrate and tapered $\mu\text{Horn}$ reflector**: surrounds the base pixel ($y \le 0$ and $|x| \ge 0.50\;\mu\text{m}$ for $y \in [0, 0.30]\;\mu\text{m}$) and tapers outward above $y = 0.30\;\mu\text{m}$ up to height $y_{\text{top}} = 0.30\;\mu\text{m} + h$ at sidewall elevation angle $\vartheta \in [50^\circ, 80^\circ]$ measured from the horizontal $x$-axis, giving top aperture width:
#
# $$w_{\text{top}}(h, \vartheta) = w_{\text{pixel}} + \frac{2 h}{\tan\vartheta}.$$
#
# - **Four horn filling configurations in the paper's 2D FDTD study** (Luce *et al.* 2026, Fig. 3 & App. A.2–A.3):
#   1. **Bare reference ($h = 0$)**: no horn above $y = 0.30\;\mu\text{m}$; published 2D $\text{LEE}_{15} = 14.9\%$ (Sec. A.3) and 2D $\text{LEE}_{90} = 55.9\%$ (Sec. A.2), giving 2D collimation fraction $\eta_{15} = \text{LEE}_{15}/\text{LEE}_{90} = 26.7\%$.
#   2. **Air-filled $\mu\text{Horn}$ ($n = 1.00$)**: optimal at $h = 3.0\;\mu\text{m}$, $\vartheta = 72^\circ$ ($w_{\text{top}} = 2.95\;\mu\text{m}$); published 2D $\text{LEE}_{15} = 27.5\%$ ($1.85\times$ bare, Sec. A.3 & Fig. 3a) and 2D $\text{LEE}_{90} \approx 46.7 \pm 1.0\%$ (read from Fig. 3d; $\eta_{15} \approx 58.9\%$).
#   3. **$\text{SiO}_2$-filled $\mu\text{Horn}$ ($n = 1.46$)**: optimal at $h = 3.0\;\mu\text{m}$, $\vartheta = 64^\circ$ ($w_{\text{top}} = 3.93\;\mu\text{m}$); published 2D $\text{LEE}_{15} = 30.4\%$ ($2.04\times$ bare, Sec. A.3 & Fig. 3b) and 2D $\text{LEE}_{90} \approx 65.1 \pm 1.0\%$ (read from Fig. 3e; $\eta_{15} \approx 46.7\%$).
#   4. **6-layer discrete GRIN ($\text{dGRIN}$) $\mu\text{Horn}$**: 6 equal-thickness ($h/6 = 500\;\text{nm}$) horizontal layers stepping down from $n_{\text{GaN}} = 2.45$ to air—$\text{Nb}_2\text{O}_5$ ($n = 2.37$), $\text{Si}_3\text{N}_4$ ($n = 2.05$), $\text{Al}_2\text{O}_3$ ($n = 1.77$), $\text{SiO}_2$ ($n = 1.46$), $\text{MgF}_2$ ($n = 1.38$), and $\text{Air}$ ($n = 1.00$)—optimal at $h = 3.0\;\mu\text{m}$, $\vartheta = 60^\circ$ ($w_{\text{top}} = 4.46\;\mu\text{m}$); published 2D $\text{LEE}_{15} = 37.7\%$ ($2.53\times$ bare, Sec. A.3 & Fig. 3c) and 2D $\text{LEE}_{90} \approx 75.3 \pm 1.0\%$ (read from Fig. 3f; $\eta_{15} \approx 50.1\%$).
#
# ![2D microLED + 6-layer dGRIN micro-horn setup geometry](figures/00a_setup_geometry.png)
# ![2D microLED + 6-layer dGRIN micro-horn mesh and excitation](figures/00b_setup_mesh_and_excitation.png)
#
# The 2D FIT domain models the $600\;\text{nm} \times 300\;\text{nm}$ $\text{GaN}$ active core ($n = 2.45$), $200\;\text{nm}$ lateral $\text{SiO}_2$ passivation barriers ($n = 1.46$), Drude $\text{Ag}$ bottom reflector and $60^\circ$ tapered $\mu\text{Horn}$ sidewalls ($\varepsilon_\infty = 4.0$), 6-layer descending-index $\text{dGRIN}$ stack ($\text{Nb}_2\text{O}_5 / \text{Si}_3\text{N}_4 / \text{Al}_2\text{O}_3 / \text{SiO}_2 / \text{MgF}_2 / \text{Air}$), 17-site incoherent quantum-well dipole ensemble at $y_{\text{QW}} = 150\;\text{nm}$ (5 representative sites shown), and top `Near2FarMonitor2D` aperture plane inside an 8-cell CPML box with a $3.0\;\text{nm}$ $\text{Ag}$ skin-depth collar.
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 5.19 µm × 4.29 µm × 500 nm (z ∈ [-250 nm, 250 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 746 × 608 × 1 (453568 cells) |
# | Discretization | Δx & resolution | Δ_min = 3 nm, 61.2 cells/λ (in n_max = 2.45), 5.1 cells across feature (15.4 nm) |
# | Boundary | -x face | CPML (8 cells, 59.9 nm) |
# | Boundary | +x face | CPML (8 cells, 59.9 nm) |
# | Boundary | -y face | CPML (8 cells, 59.7 nm) |
# | Boundary | +y face | CPML (8 cells, 60 nm) |
# | Boundary | -z face | Periodic |
# | Boundary | +z face | Periodic |
# | Material | Background | Air superstrate (n = 1.00), ε = 1.00 (isotropic) |
# | Material | Region 1: Ag reflector & mesa | ε_r = 4.00, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-2.59 µm, 2.59 µm] (d = 5.19 µm) |
# | Material | Region 2: SiO₂ passivation (n = 1.46) | ε_r = 2.13, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-500 nm, 500 nm] (d = 1 µm) |
# | Material | Region 3: GaN active core (n = 2.45) | ε_r = 6.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-300 nm, 300 nm] (d = 600 nm) |
# | Material | Region 4: dGRIN L1: Nb2O5 (n = 2.37) | ε_r = 5.62, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-789 nm, 789 nm] (d = 1.58 µm) |
# | Material | Region 5: dGRIN L2: Si3N4 (n = 2.05) | ε_r = 4.20, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.08 µm, 1.08 µm] (d = 2.15 µm) |
# | Material | Region 6: dGRIN L3: Al2O3 (n = 1.77) | ε_r = 3.13, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.37 µm, 1.37 µm] (d = 2.73 µm) |
# | Material | Region 7: dGRIN L4: SiO2 (n = 1.46) | ε_r = 2.13, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.65 µm, 1.65 µm] (d = 3.31 µm) |
# | Material | Region 8: dGRIN L5: MgF2 (n = 1.38) | ε_r = 1.90, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.94 µm, 1.94 µm] (d = 3.89 µm) |
# | Material | Region 9: Ag tapered µHorn (ϑ = 60°) | ε_r = 4.00, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-2.59 µm, -500 nm] (d = 2.09 µm) |
# | Material | Region 10: Ag tapered µHorn (ϑ = 60°) | ε_r = 4.00, μ_r = 1.00, model = DispersiveMaterial, x ∈ [500 nm, 2.59 µm] (d = 2.09 µm) |
# | Source | 1 · CurrentSource | pos: y = 150 nm, M = 17 sites in x ∈ [-240, +240] nm (5 shown), pol: TE (J_x, J_y) & TM (J_z), pulse: f₀ = 666.2 THz, Δf = 119.9 THz |
# | Monitor | 2 · Near2FarMonitor2D | Near2FarMonitor2D aperture plane (y = 3.46 µm, 21 freqs in 425–475 nm), pos: y = 3.46 µm, x ∈ [-2.52, +2.52] µm, freqs: f = 666.2 THz |
# | Variation | Task 0 — Bare reference pixel | notes=h = 0, w_top = 1.00 µm, M = 9 incoherent QW dipoles (y_QW = 150 nm) |
# | Variation | Task 1 — Air-filled µHorn | notes=h = 3.0 µm, ϑ = 72° (w_top = 2.95 µm), M = 17 incoherent QW dipoles |
# | Variation | Task 2 — SiO₂-filled µHorn | notes=h = 3.0 µm, ϑ = 64° (w_top = 3.93 µm, n = 1.46), M = 17 incoherent QW dipoles |
# | Variation | Task 3 — 6-layer dGRIN µHorn (primary) | notes=h = 3.0 µm, ϑ = 60° (w_top = 4.46 µm, Nb₂O₅/Si₃N₄/Al₂O₃/SiO₂/MgF₂/Air), M = 17 dipoles |
# | Variation | Tasks 4–6 — Angle/height, QW offset & skin-depth convergence | notes=ϑ ∈ [50°, 78°], h ∈ [0.6, 3.0] µm, Δy_QW ∈ [-40, +40] nm, Δx_min ∈ {15, 7.5, 5.0, 3.75, 2.5} nm |
#
# ![Three tapered micro-horn collimator configurations](figures/01_microled_horn_schematics.png)
#
# **Figures 00a–00b and 01** display the solver setup schematics generated via `wakis.schematic.save_setup_figures` and `wakis.schematic.plot_schematic`: Figures 00a–00b depict the 6-layer $\text{dGRIN}$ $\mu\text{Horn}$ ($h = 3.0\;\mu\text{m}$, $\vartheta = 60^\circ$, $M = 17$ incoherent QW dipoles across $x_d \in [-240, +240]\;\text{nm}$ at $y_{\text{QW}} = 150\;\text{nm}$, 5 representative sites shown) and the locally graded skin-depth collar ($\Delta x_{\min} = 3.0\;\text{nm}$, $5.13\;\text{cells}/\delta_I$, $10.25\;\text{cells}/\delta_E$; $\Delta x_{\text{bg}} = 7.5\;\text{nm}$, $24.5\;\text{cells}/\lambda$ in $\text{GaN}$), while Figure 01 compares the three tapered $\mu\text{Horn}$ geometries (Air $72^\circ$, $\text{SiO}_2$ $64^\circ$, and 6-layer $\text{dGRIN}$ $60^\circ$).

# %% [markdown]
# ## 2. Analytical baselines: escape cone, TMM & 2D dipole power
#
# Before running full-wave FDTD/FIT simulations, three analytical models establish rigorous physical bounds:
#
# 1. **Planar Snell escape cone & Fresnel transmission**: For an isotropic 2D emitter in medium $n_1$ beneath a planar interface to $n_2 < n_1$ with an ideal bottom reflector, the fraction of rays within the critical angle $\theta_c = \arcsin(n_2/n_1)$ weighted by the normal-incidence Fresnel transmittance $T_F = 4 n_1 n_2 / (n_1 + n_2)^2$ is:
#
# $$\eta_{\text{cone,2D}}(n_1 \to n_2) = 2 \left(\frac{\theta_c}{\pi}\right) \frac{4 n_1 n_2}{(n_1 + n_2)^2}.$$
#
#    For $\text{GaN} \to \text{Air}$ ($2.45 \to 1.00$), $\theta_c = 24.09^\circ$, $T_F = 82.37\%$, and $\eta_{\text{cone,2D}} = 22.05\%$ (in 3D, $\eta_{\text{cone,3D}} = (1 - \cos\theta_c) T_F = 7.18\%$). Replacing air at the $\text{GaN}$ exit face with $\text{SiO}_2$ ($n_2 = 1.46$) expands the critical angle to $\theta_c = 36.58^\circ$ ($\eta_{\text{cone,2D}} = 38.80\%$), and matching into $\text{Nb}_2\text{O}_5$ ($n_2 = 2.37$, the bottom $\text{dGRIN}$ layer) expands $\theta_c$ to $75.32^\circ$ ($\eta_{\text{cone,2D}} = 83.66\%$), eliminating TIR trapping at the $\text{GaN}$ aperture!
#
# 2. **Transfer-Matrix Method (TMM) normal reflectance**: The $2\times 2$ characteristic matrix of the 5 dielectric layers ($\text{Nb}_2\text{O}_5 / \text{Si}_3\text{N}_4 / \text{Al}_2\text{O}_3 / \text{SiO}_2 / \text{MgF}_2$, each $500\;\text{nm}$ thick) reduces the normal-incidence $\text{GaN} \to \text{Air}$ power reflectance from $R_{\text{bare}} = 17.63\%$ down to a spectrally averaged $\langle R_{\text{dGRIN}} \rangle \approx 3.6\%$ across $425\text{--}475\;\text{nm}$.
#
# 3. **2D line-dipole homogeneous reference power & cylindrical Near-to-Far-Field scaling**: On a 2D $(x, y)$ grid with 1 periodic cell of thickness $\Delta z$ along $z$ ($\partial_z = 0$), a point current source $I(t)$ on a Yee edge represents a 2D line current of linear density $I(t)$ $[\text{A}]$. In a homogeneous lossless dielectric of refractive index $n$, the analytical cycle-averaged power spectrum radiated per unit $z$-thickness is:
#
# $$P_{\text{hom},z}(f) = \frac{\omega \mu_0 |S(f)|^2}{4\,\Delta z} \quad (\text{TM}_z,\; E_z\text{ out-of-plane}), \qquad P_{\text{hom},x}(f) = P_{\text{hom},y}(f) = \frac{\omega \mu_0 |S(f)|^2}{8\,\Delta z} \quad (\text{TE}_{x,y},\; E_x, E_y\text{ in-plane}).$$
#
#    Because `wakis.near2far.Near2FarMonitor` evaluates the 3D Stratton–Chu spherical Green's function over the 1-cell periodic strip of height $\Delta z$, evaluating its far field in the $xy$-plane ($\theta_{\text{sph}} = \pi/2$, $\varphi_{\text{sph}} = \pi/2 - \alpha$ where $\alpha \in [-\pi/2, +\pi/2]$ is the polar angle from the $+y$ emission axis) and multiplying the 3D intensity by the exact 3D-to-2D cylindrical conversion factor $\frac{2\pi}{k\,\Delta z}$ yields the exact 2D angular radiation intensity $U_{2\text{D}}(\alpha, f)$ $[\text{W}\,\text{s}^2/\text{rad}]$ (`Near2FarMonitor2D` in `_helpers.py`).

# %% [markdown]
# ## 3. 2D FIT solver setup, dispersive Ag & Kottke sidewall smoothing
#
# Our 2D FIT model in `_helpers.py` combines four numerical techniques tailored to metallic-horn emitters:
#
# 1. **Calibrated 1-pole Drude Silver ($\text{Ag}$)**: Fitting a 1-pole Drude model $\varepsilon_{\text{Ag}}(\omega) = \varepsilon_\infty - \omega_p^2 / (\omega(\omega - \mathrm{i}\gamma))$ with $\varepsilon_\infty = 4.0$, $\omega_p = 1.286 \times 10^{16}\;\text{rad/s}$, and $\gamma = 2.822 \times 10^{14}\;\text{rad/s}$ matches the Rakic (1998) [4] complex permittivity of silver at $\lambda_0 = 450\;\text{nm}$ ($\varepsilon = -5.402 - 0.634\mathrm{i}$, $n_{\text{Ag}} = 0.136 - 2.328\mathrm{i}$), yielding an electric-field skin depth $\delta_E = c / (\omega \kappa) = 30.76\;\text{nm}$ and intensity skin depth $\delta_I = \delta_E / 2 = 15.38\;\text{nm}$ while remaining unconditionally stable at $\text{CFL} = 0.5$.
# 2. **Kottke subpixel smoothing on sloped metal–dielectric sidewalls (`assign_materials_2d_dispersive`)**: Sloped $\text{Ag}$ sidewalls at $\vartheta \in [50^\circ, 80^\circ]$ cut obliquely across Cartesian Yee cells. Using exact 2D polygon clipping (`PolySlab.fill`) and outward boundary normals (`PolySlab.normal`), each cut cell is decomposed into a parallel (tangential) branch $\varepsilon_\parallel = f_{\text{Ag}} \varepsilon_{\text{Ag}}(\omega) + (1 - f_{\text{Ag}}) \varepsilon_d$ and a series (normal) branch $\varepsilon_\perp^{-1} = f_{\text{Ag}} \varepsilon_{\text{Ag}}(\omega)^{-1} + (1 - f_{\text{Ag}}) \varepsilon_d^{-1}$, which are compiled into auxiliary differential equation (ADE) sub-capacitors via `wakis.tensor_dispersive`.
# 3. **Exact 2D TE/TM decoupling**: Because $\partial_z = 0$ on a 1-cell periodic-$z$ grid, in-plane $\text{TE}$ $(E_x, E_y, H_z)$ and out-of-plane $\text{TM}$ $(E_z, H_x, H_y)$ modes are decoupled to machine precision ($< 10^{-33}$) both in the Yee curl updates and in the Stratton–Chu far-field components ($E_\phi$ vs $E_\theta$ at $\theta_{\text{sph}} = \pi/2$). We therefore co-propagate $J_x$ and $J_z$ simultaneously in Pass 1 and $J_y$ in Pass 2, reducing the simulation count per dipole position from 3 to 2 with zero approximation.
# 4. **Bilinear Yee dipole interpolation (`InterpolatedCurrentSource2D`)**: Standard nearest-node source snapping places $E_x, E_z$ on primary $y$-nodes and $E_y$ on half-cell centers $y_{j+1/2}$, introducing $\pm \Delta y / 2$ mirror-distance jumps when sweeping $y_{\text{QW}}$ or grid resolution $\Delta x$. Bilinear cloud-in-cell weights keep all three dipole orientations centered at the exact continuous coordinate $(x_d, y_{\text{QW}})$.

# %%
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import resource
import shlex
import sys
import time

import matplotlib

matplotlib.use("Agg")
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light

TUTORIAL_DIR = Path(__file__).resolve().parent
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (
    DGRIN_INDICES,
    DGRIN_STACK,
    H_HORN_DEFAULT,
    N_AIR,
    N_GAN,
    N_SIO2,
    PAPER_2D_REFS,
    T_GAN,
    W_GAN,
    W_PASS,
    W_PIXEL,
    WL_0,
    WL_FWHM,
    WL_MAX,
    WL_MIN,
    Y_QW_DEFAULT,
    ag_skin_depth,
    analytic_escape_cone_lee,
    build_horn_schematic_objects,
    evaluate_horn_configuration,
    extract_2d_eps_map,
    fresnel_normal_transmission,
    lambertian_cone_fraction,
    led_gaussian_spectrum,
    make_ag_drude,
    overlay_horn_geometry,
    save_horn_setup_figures,
    snell_critical_angle_deg,
    tidy_horn_schematic_axes,
    tmm_normal_reflectance,
)
from wakis import figstyle, schematic
from wakis.dispersive import dispersive_lib

DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"


# %% [markdown]
# ## 4. Study tasks & execution modes
#
# Every simulation in this tutorial obeys three strict FDTD/FIT mesh resolution rules constructed via `wakis.templates.meshing.graded_axis` (`build_graded_horn_grid()` in `_helpers.py`, grading factor $\le 1.35$, with mesh nodes snapped to all horizontal dielectric/metal interfaces $y \in \{0, y_{\text{QW}}, t_{\text{GaN}}, y_k, y_{\text{top}}\}$ and vertical mesa/aperture walls $x \in \{0, \pm w_{\text{GaN}}/2, \pm w_{\text{pixel}}/2\}$):
#
# 1. **Wavelength sampling inside the highest-index dielectric ($\text{GaN}$, $n_{\max} = 2.45$)**: Never fewer than $10\;\text{cells}/\lambda$ at the shortest wavelength $\lambda_{\min} = 425\;\text{nm}$ ($\lambda_{\min}/n_{\text{GaN}} = 173.47\;\text{nm}$, $\lambda_0/n_{\text{GaN}} = 183.67\;\text{nm}$ at $\lambda_0 = 450\;\text{nm}$).
#    - `--quick`: $\Delta x_{\text{bg}} = 15.0\;\text{nm}$ ($11.56\;\text{cpw}$ at $425\;\text{nm}$, $12.24\;\text{cpw}$ at $450\;\text{nm}$).
#    - `--full`: $\Delta x_{\text{bg}} = 7.5\;\text{nm}$ ($23.13\;\text{cpw}$ at $425\;\text{nm}$, $24.49\;\text{cpw}$ at $450\;\text{nm}$).
# 2. **Geometric feature resolution ($\ge 4$ cells across the smallest feature)**: Across the $150\;\text{nm}$ lower/upper $\text{GaN}$ half-slabs ($110\;\text{nm}$ at $\Delta y_{\text{QW}} = -40\;\text{nm}$ in Task 5), the $200\;\text{nm}$ lateral $\text{SiO}_2$ passivation barriers, and the 6 $\text{dGRIN}$ layers ($500\;\text{nm}$ each at $h = 3.0\;\mu\text{m}$, $100\;\text{nm}$ each at $h = 0.6\;\mu\text{m}$ in Task 4), the graded mesh places $7\text{--}18$ cells across the thinnest feature in `--quick` and $14\text{--}30$ cells in `--full`.
# 3. **Metallic skin-depth resolution ($\ge 4$ cells per intensity skin depth $\delta_I = \lambda_0 / (4\pi\kappa)$)**: At $\lambda_0 = 450\;\text{nm}$, calibrated Drude $\text{Ag}$ ($\kappa = 2.328$) has power/intensity skin depth $\delta_I = \lambda_0 / (4\pi\kappa) = 15.38\;\text{nm}$ and electric-field amplitude skin depth $\delta_E = \lambda_0 / (2\pi\kappa) = 2\delta_I = 30.76\;\text{nm}$. Using local `graded_axis` collars around the bottom $\text{Ag}$ mirror ($y = 0$), the lateral $\text{Ag}$ mesa walls ($x = \pm 0.50\;\mu\text{m}$), and the horn throat ($y = 0.30\;\mu\text{m}$):
#    - `--quick`: $\Delta x_{\min} = 3.75\;\text{nm}$, giving **$4.10\;\text{cells}/\delta_I$** (**$8.20\;\text{cells}/\delta_E$**).
#    - `--full`: $\Delta x_{\min} = 3.00\;\text{nm}$ ($5.13\;\text{cells}/\delta_I$, $10.25\;\text{cells}/\delta_E$) for Tasks 0–5, and down to $\Delta x_{\min} = 2.50\;\text{nm}$ ($6.15\;\text{cells}/\delta_I$, $12.30\;\text{cells}/\delta_E$) at the accepted finest level of Task 6.
#
# ### Task resolution & runtime tables (`--quick` vs `--full`, both measured)
#
# | Task | Name | Mode | $\text{cpw}_{\text{GaN}}$ ($450 / 425\;\text{nm}$) | $\Delta x_{\text{bg}}$ [nm] | $\Delta x_{\min}$ [nm] | Min feature [nm] (cells) | Cells / skin depth ($\delta_I$ / $\delta_E$) | $N_{\text{cells}}$ | $N_{\text{steps}}$ | Solver runs | Time [s] |
# |---|---|---|---|---|---|---|---|---|---|---|---|
# | 0 | `bare_ref` | `--quick` | $12.24$ / $11.56$ | $15.00$ | $3.75$ | $150$ ($18$) | $4.10$ / $8.20$ | $31{,}395$ | $4{,}097$ | $6$ | $148.3$ (measured) |
# | 1 | `air_horn_opt` | `--quick` | $12.24$ / $11.56$ | $15.00$ | $3.75$ | $150$ ($18$) | $4.10$ / $8.20$ | $103{,}734$ | $6{,}622$ | $6$ | $549.7$ (measured) |
# | 2 | `sio2_horn_opt` | `--quick` | $12.24$ / $11.56$ | $15.00$ | $3.75$ | $150$ ($18$) | $4.10$ / $8.20$ | $115{,}974$ | $6{,}622$ | $6$ | $598.0$ (measured) |
# | 3 | `dgrin_horn_opt` | `--quick` | $12.24$ / $11.56$ | $15.00$ | $3.75$ | $150$ ($18$) | $4.10$ / $8.20$ | $125{,}550$ | $6{,}622$ | $10$ | $1{,}018.6$ (measured) |
# | 4 | `angle_and_height_sweep` | `--quick` | $12.24$ / $11.56$ | $15.00$ | $3.75$ | $100$ ($7$) | $4.10$ / $8.20$ | $125{,}550$ | $6{,}622$ | $42$ | $2{,}966.1$ (measured) |
# | 5 | `qw_position_sweep` | `--quick` | $12.24$ / $11.56$ | $15.00$ | $3.75$ | $110$ ($16$) | $4.10$ / $8.20$ | $125{,}550$ | $6{,}622$ | $20$ | $1{,}227.1$ (measured) |
# | 6 | `conv_kottke_vs_staircase` | `--quick` | $12.24$ / $11.56$ | $15.00$ | $3.75$ | $150$ ($18$) | $4.10$ / $8.20$ (ladder $1.03\text{--}4.10$) | $125{,}145$ | $6{,}622$ | $16$ | $813.3$ (measured) |
# | 0 | `bare_ref` | `--full` | $24.49$ / $23.13$ | $7.50$ | $3.00$ | $150$ ($27$) | $5.13$ / $10.25$ | $89{,}345$ | $5{,}121$ | $6$ | $447.6$ (measured) |
# | 1 | `air_horn_opt` | `--full` | $24.49$ / $23.13$ | $7.50$ | $3.00$ | $150$ ($27$) | $5.13$ / $10.25$ | $349{,}935$ | $8{,}277$ | $6$ | $1{,}928.2$ (measured) |
# | 2 | `sio2_horn_opt` | `--full` | $24.49$ / $23.13$ | $7.50$ | $3.00$ | $150$ ($27$) | $5.13$ / $10.25$ | $395{,}455$ | $8{,}277$ | $6$ | $2{,}148.8$ (measured) |
# | 3 | `dgrin_horn_opt` | `--full` | $24.49$ / $23.13$ | $7.50$ | $3.00$ | $150$ ($27$) | $5.13$ / $10.25$ | $425{,}790$ | $8{,}277$ | $12$ | $4{,}400.7$ (measured) |
# | 4 | `angle_and_height_sweep` | `--full` | $24.49$ / $23.13$ | $7.50$ | $3.00$ | $100$ ($14$) | $5.13$ / $10.25$ | $425{,}790$ | $8{,}277$ | $66$ | $18{,}486.0$ (measured) |
# | 5 | `qw_position_sweep` | `--full` | $24.49$ / $23.13$ | $7.50$ | $3.00$ | $110$ ($22$) | $5.13$ / $10.25$ | $425{,}790$ | $8{,}277$ | $36$ | $7{,}169.6$ (measured) |
# | 6 | `conv_kottke_vs_staircase` | `--full` | $24.49$ / $23.13$ | $7.50$ | $2.50$ | $150$ ($30$) | $6.15$ / $12.30$ (ladder $1.03\text{--}6.15$) | $444{,}066$ | $9{,}933$ | $20$ | $4{,}277.4$ (measured) |


# %%
def get_task_table(quick=False):
    """
    Return the 7-task study table for `--quick` or `--full` execution.

    Parameters
    ----------
    quick : bool, optional
        If True, use `--quick` resolution (`dx = 15 nm` background, `12.24 cpw` in GaN,
        `dx_skin = 3.75 nm` at Ag interfaces, `4.10 cells/delta_I`, `8.20 cells/delta_E`).
        If False, use `--full` resolution (`dx = 7.5 nm` background, `24.49 cpw` in GaN,
        `dx_skin = 3.0 nm` at Ag interfaces, `5.13 cells/delta_I`, `10.25 cells/delta_E`).

    Returns
    -------
    list of dict
        Task specification dictionaries (`index`, `name`, `kind`, `description`, `params`).
    """
    dx_main = 15e-9 if quick else 7.5e-9
    dx_skin_main = 3.75e-9 if quick else 3.0e-9
    dx_sweep = 15e-9 if quick else 7.5e-9
    dx_skin_sweep = 3.75e-9 if quick else 3.0e-9
    dip_xs_3 = (0.0, 0.12e-6, 0.24e-6)
    dip_ws_3 = (1.0, 2.0, 2.0)
    dip_xs_conv = (
        (0.0, 0.06e-6, 0.12e-6, 0.18e-6, 0.24e-6)
        if quick
        else (0.0, 0.05e-6, 0.10e-6, 0.15e-6, 0.20e-6, 0.25e-6)
    )
    dip_ws_conv = (1.0,) + (2.0,) * (len(dip_xs_conv) - 1)

    return [
        {
            "index": 0,
            "name": "bare_ref",
            "kind": "config",
            "description": "Bare InGaN/GaN reference pixel (h=0, w=1.0 um) + 2D DFT fields",
            "params": {
                "config": "bare",
                "h_horn": 0.0,
                "theta_deg": 90.0,
                "dx": dx_main,
                "dx_skin": dx_skin_main,
                "x_half_min": 1.80e-6,
                "dipole_xs": dip_xs_3,
                "dipole_weights": dip_ws_3,
                "collect_fields": True,
                "record_movie": False,
            },
        },
        {
            "index": 1,
            "name": "air_horn_opt",
            "kind": "config",
            "description": "Air-filled uHorn at published optimum (h=3.0 um, theta=72 deg) + 2D DFT fields",
            "params": {
                "config": "air",
                "h_horn": H_HORN_DEFAULT,
                "theta_deg": 72.0,
                "dx": dx_main,
                "dx_skin": dx_skin_main,
                "x_half_min": 2.10e-6,
                "dipole_xs": dip_xs_3,
                "dipole_weights": dip_ws_3,
                "collect_fields": True,
                "record_movie": False,
            },
        },
        {
            "index": 2,
            "name": "sio2_horn_opt",
            "kind": "config",
            "description": "SiO2-filled uHorn at published optimum (h=3.0 um, theta=64 deg) + 2D DFT fields",
            "params": {
                "config": "sio2",
                "h_horn": H_HORN_DEFAULT,
                "theta_deg": 64.0,
                "dx": dx_main,
                "dx_skin": dx_skin_main,
                "x_half_min": 2.40e-6,
                "dipole_xs": dip_xs_3,
                "dipole_weights": dip_ws_3,
                "collect_fields": True,
                "record_movie": False,
            },
        },
        {
            "index": 3,
            "name": "dgrin_horn_opt",
            "kind": "config",
            "description": "6-layer dGRIN uHorn (h=3.0 um, theta=60 deg), dipole convergence, DFT & movie",
            "params": {
                "config": "dgrin",
                "h_horn": H_HORN_DEFAULT,
                "theta_deg": 60.0,
                "dx": dx_main,
                "dx_skin": dx_skin_main,
                "x_half_min": 2.55e-6,
                "dipole_xs": dip_xs_conv,
                "dipole_weights": dip_ws_conv,
                "collect_fields": True,
                "record_movie": True,
                "movie_frames": 28,
            },
        },
        {
            "index": 4,
            "name": "angle_and_height_sweep",
            "kind": "angle_height_sweep",
            "description": "Sidewall angle (52..78 deg) and horn height (0.6..3.0 um) sweeps for Air, SiO2, dGRIN",
            "params": {
                "dx": dx_sweep,
                "dx_skin": dx_skin_sweep,
                "angles_deg": (52.0, 60.0, 64.0, 72.0, 78.0)
                if quick
                else (50.0, 56.0, 60.0, 64.0, 68.0, 72.0, 78.0),
                "heights_um": (0.6, 1.8, 3.0)
                if quick
                else (0.6, 1.2, 1.8, 2.4, 3.0),
                "dipole_xs": (0.0,),
                "dipole_weights": (1.0,),
                "run_extra_fs": None,
            },
        },
        {
            "index": 5,
            "name": "qw_position_sweep",
            "kind": "qw_sweep",
            "description": "Quantum-well vertical offset sweep dy_QW in [-40, +40] nm around y_QW=150 nm",
            "params": {
                "dx": dx_main,
                "dx_skin": dx_skin_main,
                "dy_nm": (-40.0, -20.0, 0.0, 20.0, 40.0)
                if quick
                else (-40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0),
                "dipole_xs": (0.0,),
                "dipole_weights": (1.0,),
            },
        },
        {
            "index": 6,
            "name": "conv_kottke_vs_staircase",
            "kind": "convergence",
            "description": "Ag skin-depth resolution ladder (1.03..4.10 cells/delta_I) & Kottke vs staircase on 60 deg sidewall",
            "params": {
                "dx": dx_main,
                "dx_bg_nm": (15.0, 15.0, 15.0, 15.0)
                if quick
                else (15.0, 7.5, 7.5, 7.5, 7.5),
                "dx_skins_nm": (15.0, 7.5, 5.0, 3.75)
                if quick
                else (15.0, 7.5, 5.0, 3.75, 2.5),
                "demo_underresolved": (True, False, False, False)
                if quick
                else (True, False, False, False, False),
                "dipole_xs": (0.0,),
                "dipole_weights": (1.0,),
                "run_extra_fs": None,
            },
        },
    ]


def write_task_json(out_json_path, task, payload):
    """Write scalar resolution and physics metrics to a companion `.json` file."""
    json_meta = {
        "index": int(task["index"]),
        "name": str(task["name"]),
        "kind": str(task["kind"]),
        "description": str(task["description"]),
        "dx_background_nm": round(float(payload["dx_background_nm"]), 4),
        "dx_min_nm": round(float(payload["dx_min_nm"]), 4),
        "cpw_gan": round(float(payload["cpw_gan"]), 4),
        "cpw_gan_min_wl": round(float(payload["cpw_gan_min_wl"]), 4),
        "min_feature_nm": round(float(payload["min_feature_nm"]), 4),
        "cells_across_feature": round(float(payload["cells_across_feature"]), 4),
        "skin_depth_nm": round(float(payload["skin_depth_nm"]), 4),
        "skin_depth_E_nm": round(float(payload["skin_depth_E_nm"]), 4),
        "skin_depth_I_nm": round(float(payload["skin_depth_I_nm"]), 4),
        "cells_per_skin_depth": round(float(payload["cells_per_skin_depth"]), 4),
        "cells_per_skin_depth_E": round(float(payload["cells_per_skin_depth_E"]), 4),
        "cells_per_skin_depth_I": round(float(payload["cells_per_skin_depth_I"]), 4),
        "n_cells": int(payload["n_cells"]),
        "n_steps": int(payload["n_steps"]),
        "elapsed_s": round(float(payload["elapsed_s"]), 3),
    }
    for k_opt in (
        "lee15_eq_pct",
        "lee90_eq_pct",
        "lee15_pct",
        "lee90_pct",
        "lee15_wrk_pct",
        "lee90_wrk_pct",
        "collimation_ratio_pct",
        "purcell_factor",
        "fwhm_deg",
        "directivity_2d",
        "ref_lee15",
        "ref_lee90",
    ):
        if k_opt in payload:
            json_meta[k_opt] = round(float(payload[k_opt]), 4)
    out_json_path.write_text(json.dumps(json_meta, indent=2) + "\n")


def run_single_task(task, out_dir=DATA_DIR):
    """
    Execute a single study task and save its output to `out_dir/task_{index:02d}_{name}.npz` and `.json`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = task["index"]
    name = task["name"]
    kind = task["kind"]
    p = task["params"]
    t0 = time.perf_counter()

    if kind == "config":
        payload = evaluate_horn_configuration(**p)
    elif kind == "angle_height_sweep":
        angles = np.asarray(p["angles_deg"], dtype=float)
        heights = np.asarray(p["heights_um"], dtype=float)
        dx = float(p["dx"])
        dx_skin = float(p.get("dx_skin", dx))
        d_xs = p["dipole_xs"]
        d_ws = p["dipole_weights"]
        run_extra_fs = p.get("run_extra_fs", None)
        configs = ("air", "sio2", "dgrin")
        opt_theta = {"air": 72.0, "sio2": 64.0, "dgrin": 60.0}
        x_half_by_cfg = {"air": 2.10e-6, "sio2": 2.40e-6, "dgrin": 2.55e-6}

        payload = {
            "dx": np.float64(dx),
            "dx_skin": np.float64(dx_skin),
            "angles_deg": angles,
            "heights_um": heights,
        }
        all_res = []
        ref_res = None
        for cfg in configs:
            l15_a = np.zeros(angles.size, dtype=float)
            l90_a = np.zeros(angles.size, dtype=float)
            cached_opt = None
            for i, th in enumerate(angles):
                res = evaluate_horn_configuration(
                    config=cfg,
                    h_horn=H_HORN_DEFAULT,
                    theta_deg=float(th),
                    dx=dx,
                    dx_skin=dx_skin,
                    x_half_min=x_half_by_cfg[cfg],
                    dipole_xs=d_xs,
                    dipole_weights=d_ws,
                    n_freqs=15,
                    run_extra_fs=run_extra_fs,
                )
                l15_a[i] = res["lee15_pct"]
                l90_a[i] = res["lee90_pct"]
                all_res.append(res)
                if abs(float(th) - opt_theta[cfg]) < 1e-6:
                    cached_opt = res
                    if cfg == "dgrin":
                        ref_res = res
            payload[f"{cfg}_angle_lee15"] = l15_a
            payload[f"{cfg}_angle_lee90"] = l90_a

            l15_h = np.zeros(heights.size, dtype=float)
            l90_h = np.zeros(heights.size, dtype=float)
            for j, h_um in enumerate(heights):
                if abs(float(h_um) - 3.0) < 1e-6 and cached_opt is not None:
                    res = cached_opt
                else:
                    res = evaluate_horn_configuration(
                        config=cfg,
                        h_horn=float(h_um) * 1e-6,
                        theta_deg=opt_theta[cfg],
                        dx=dx,
                        dx_skin=dx_skin,
                        x_half_min=x_half_by_cfg[cfg],
                        dipole_xs=d_xs,
                        dipole_weights=d_ws,
                        n_freqs=15,
                        run_extra_fs=run_extra_fs,
                    )
                    all_res.append(res)
                l15_h[j] = res["lee15_pct"]
                l90_h[j] = res["lee90_pct"]
            payload[f"{cfg}_height_lee15"] = l15_h
            payload[f"{cfg}_height_lee90"] = l90_h

        if ref_res is None:
            ref_res = all_res[-1]
        payload.update(
            {
                "dx_background_nm": np.float64(
                    max(float(r["dx_background_nm"]) for r in all_res)
                ),
                "dx_min_nm": np.float64(min(float(r["dx_min_nm"]) for r in all_res)),
                "cpw_gan": np.float64(min(float(r["cpw_gan"]) for r in all_res)),
                "cpw_gan_min_wl": np.float64(
                    min(float(r["cpw_gan_min_wl"]) for r in all_res)
                ),
                "min_feature_nm": np.float64(
                    min(float(r["min_feature_nm"]) for r in all_res)
                ),
                "cells_across_feature": np.float64(
                    min(float(r["cells_across_feature"]) for r in all_res)
                ),
                "skin_depth_nm": ref_res["skin_depth_nm"],
                "skin_depth_E_nm": ref_res["skin_depth_E_nm"],
                "skin_depth_I_nm": ref_res["skin_depth_I_nm"],
                "cells_per_skin_depth": ref_res["cells_per_skin_depth"],
                "cells_per_skin_depth_E": ref_res["cells_per_skin_depth_E"],
                "cells_per_skin_depth_I": ref_res["cells_per_skin_depth_I"],
                "n_cells": ref_res["n_cells"],
                "n_steps": ref_res["n_steps"],
                "elapsed_s": np.float64(time.perf_counter() - t0),
            }
        )

    elif kind == "qw_sweep":
        dy_nm = np.asarray(p["dy_nm"], dtype=float)
        y_qws = Y_QW_DEFAULT + dy_nm * 1e-9
        dx = float(p["dx"])
        dx_skin = float(p.get("dx_skin", dx))
        d_xs = p["dipole_xs"]
        d_ws = p["dipole_weights"]

        bare_15 = np.zeros(dy_nm.size, dtype=float)
        bare_90 = np.zeros(dy_nm.size, dtype=float)
        dgrin_15 = np.zeros(dy_nm.size, dtype=float)
        dgrin_90 = np.zeros(dy_nm.size, dtype=float)
        all_res = []
        ref_res = None

        for i, y_q in enumerate(y_qws):
            rb = evaluate_horn_configuration(
                config="bare",
                h_horn=0.0,
                theta_deg=90.0,
                y_qw=float(y_q),
                dx=dx,
                dx_skin=dx_skin,
                x_half_min=1.80e-6,
                dipole_xs=d_xs,
                dipole_weights=d_ws,
                n_freqs=15,
            )
            rd = evaluate_horn_configuration(
                config="dgrin",
                h_horn=H_HORN_DEFAULT,
                theta_deg=60.0,
                y_qw=float(y_q),
                dx=dx,
                dx_skin=dx_skin,
                x_half_min=2.55e-6,
                dipole_xs=d_xs,
                dipole_weights=d_ws,
                n_freqs=15,
            )
            bare_15[i] = rb["lee15_pct"]
            bare_90[i] = rb["lee90_pct"]
            dgrin_15[i] = rd["lee15_pct"]
            dgrin_90[i] = rd["lee90_pct"]
            all_res.extend([rb, rd])
            if abs(float(dy_nm[i])) < 1e-6:
                ref_res = rd

        if ref_res is None:
            ref_res = all_res[-1]
        payload = {
            "dx": np.float64(dx),
            "dx_skin": np.float64(dx_skin),
            "dy_nm": dy_nm,
            "y_qw_nm": y_qws * 1e9,
            "bare_lee15": bare_15,
            "bare_lee90": bare_90,
            "dgrin_lee15": dgrin_15,
            "dgrin_lee90": dgrin_90,
            "enhancement_15": dgrin_15 / np.maximum(bare_15, 1e-12),
            "dx_background_nm": np.float64(
                max(float(r["dx_background_nm"]) for r in all_res)
            ),
            "dx_min_nm": np.float64(min(float(r["dx_min_nm"]) for r in all_res)),
            "cpw_gan": np.float64(min(float(r["cpw_gan"]) for r in all_res)),
            "cpw_gan_min_wl": np.float64(
                min(float(r["cpw_gan_min_wl"]) for r in all_res)
            ),
            "min_feature_nm": np.float64(
                min(float(r["min_feature_nm"]) for r in all_res)
            ),
            "cells_across_feature": np.float64(
                min(float(r["cells_across_feature"]) for r in all_res)
            ),
            "skin_depth_nm": ref_res["skin_depth_nm"],
            "skin_depth_E_nm": ref_res["skin_depth_E_nm"],
            "skin_depth_I_nm": ref_res["skin_depth_I_nm"],
            "cells_per_skin_depth": ref_res["cells_per_skin_depth"],
            "cells_per_skin_depth_E": ref_res["cells_per_skin_depth_E"],
            "cells_per_skin_depth_I": ref_res["cells_per_skin_depth_I"],
            "n_cells": ref_res["n_cells"],
            "n_steps": ref_res["n_steps"],
            "elapsed_s": np.float64(time.perf_counter() - t0),
        }

    elif kind == "convergence":
        dx_bg_nm_arr = np.asarray(p["dx_bg_nm"], dtype=float)
        dx_skins_nm = np.asarray(p["dx_skins_nm"], dtype=float)
        demo_flags = np.asarray(p["demo_underresolved"], dtype=bool)
        d_xs = p["dipole_xs"]
        d_ws = p["dipole_weights"]
        run_extra_fs = p.get("run_extra_fs", None)

        n_pts = dx_skins_nm.size
        kottke_15 = np.zeros(n_pts, dtype=float)
        kottke_90 = np.zeros(n_pts, dtype=float)
        kottke_time = np.zeros(n_pts, dtype=float)
        stair_15 = np.zeros(n_pts, dtype=float)
        stair_90 = np.zeros(n_pts, dtype=float)
        stair_time = np.zeros(n_pts, dtype=float)

        dx_bg_meas = np.zeros(n_pts, dtype=float)
        dx_min_meas = np.zeros(n_pts, dtype=float)
        cpw_gan_arr = np.zeros(n_pts, dtype=float)
        cpw_gan_min_arr = np.zeros(n_pts, dtype=float)
        min_feat_arr = np.zeros(n_pts, dtype=float)
        cells_feat_arr = np.zeros(n_pts, dtype=float)
        cells_skin_i_arr = np.zeros(n_pts, dtype=float)
        cells_skin_e_arr = np.zeros(n_pts, dtype=float)
        n_cells_arr = np.zeros(n_pts, dtype=np.int32)
        n_steps_arr = np.zeros(n_pts, dtype=np.int32)
        rk_list = []

        for i, (bg_nm, sk_nm) in enumerate(zip(dx_bg_nm_arr, dx_skins_nm)):
            rk = evaluate_horn_configuration(
                config="dgrin",
                h_horn=H_HORN_DEFAULT,
                theta_deg=60.0,
                dx=float(bg_nm) * 1e-9,
                dx_skin=float(sk_nm) * 1e-9,
                averaging="kottke",
                dipole_xs=d_xs,
                dipole_weights=d_ws,
                n_freqs=15,
                x_half_min=2.55e-6,
                run_extra_fs=run_extra_fs,
                full_sidewall_skin=False,
            )
            rs = evaluate_horn_configuration(
                config="dgrin",
                h_horn=H_HORN_DEFAULT,
                theta_deg=60.0,
                dx=float(bg_nm) * 1e-9,
                dx_skin=float(sk_nm) * 1e-9,
                averaging="staircase",
                dipole_xs=d_xs,
                dipole_weights=d_ws,
                n_freqs=15,
                x_half_min=2.55e-6,
                run_extra_fs=run_extra_fs,
                full_sidewall_skin=False,
            )
            rk_list.append(rk)
            kottke_15[i] = rk["lee15_pct"]
            kottke_90[i] = rk["lee90_pct"]
            kottke_time[i] = rk["elapsed_s"]
            stair_15[i] = rs["lee15_pct"]
            stair_90[i] = rs["lee90_pct"]
            stair_time[i] = rs["elapsed_s"]

            dx_bg_meas[i] = rk["dx_background_nm"]
            dx_min_meas[i] = rk["dx_min_nm"]
            cpw_gan_arr[i] = rk["cpw_gan"]
            cpw_gan_min_arr[i] = rk["cpw_gan_min_wl"]
            min_feat_arr[i] = rk["min_feature_nm"]
            cells_feat_arr[i] = rk["cells_across_feature"]
            cells_skin_i_arr[i] = rk["cells_per_skin_depth"]
            cells_skin_e_arr[i] = rk["cells_per_skin_depth_E"]
            n_cells_arr[i] = rk["n_cells"]
            n_steps_arr[i] = rk["n_steps"]

        r_ref = rk_list[-1]
        rel_chg_15 = np.zeros(n_pts, dtype=float)
        rel_chg_90 = np.zeros(n_pts, dtype=float)
        for i in range(1, n_pts):
            rel_chg_15[i] = (
                100.0 * abs(kottke_15[i] - kottke_15[i - 1]) / max(kottke_15[i], 1e-12)
            )
            rel_chg_90[i] = (
                100.0 * abs(kottke_90[i] - kottke_90[i - 1]) / max(kottke_90[i], 1e-12)
            )

        payload = {
            "dxs_nm": dx_min_meas,
            "dx_skins_nm": dx_min_meas,
            "dx_background_nm_arr": dx_bg_meas,
            "dx_min_nm_arr": dx_min_meas,
            "cpw_gan_arr": cpw_gan_arr,
            "cpw_gan_min_wl_arr": cpw_gan_min_arr,
            "min_feature_nm_arr": min_feat_arr,
            "cells_across_feature_arr": cells_feat_arr,
            "cells_per_skin_depth_arr": cells_skin_i_arr,
            "cells_per_skin_depth_E_arr": cells_skin_e_arr,
            "cells_per_skin_depth_I_arr": cells_skin_i_arr,
            "demo_underresolved_arr": demo_flags,
            "n_cells_arr": n_cells_arr,
            "n_steps_arr": n_steps_arr,
            "kottke_lee15": kottke_15,
            "kottke_lee90": kottke_90,
            "kottke_time_s": kottke_time,
            "kottke_rel_change_lee15_pct": rel_chg_15,
            "kottke_rel_change_lee90_pct": rel_chg_90,
            "staircase_lee15": stair_15,
            "staircase_lee90": stair_90,
            "staircase_time_s": stair_time,
            "ref_dx_nm": np.float64(dx_min_meas[-1]),
            "ref_cpw_gan": np.float64(cpw_gan_arr[-1]),
            "ref_lee15": np.float64(r_ref["lee15_pct"]),
            "ref_lee90": np.float64(r_ref["lee90_pct"]),
            "ref_rho_lee_deg": r_ref["rho_lee_deg"],
            "ref_angles_deg": r_ref["angles_deg"],
            "ref_time_s": np.float64(r_ref["elapsed_s"]),
            # Required top-level resolution metrics at the accepted level
            "dx_background_nm": np.float64(r_ref["dx_background_nm"]),
            "dx_min_nm": np.float64(r_ref["dx_min_nm"]),
            "cpw_gan": np.float64(r_ref["cpw_gan"]),
            "cpw_gan_min_wl": np.float64(r_ref["cpw_gan_min_wl"]),
            "min_feature_nm": np.float64(r_ref["min_feature_nm"]),
            "cells_across_feature": np.float64(r_ref["cells_across_feature"]),
            "skin_depth_nm": np.float64(r_ref["skin_depth_nm"]),
            "skin_depth_E_nm": np.float64(r_ref["skin_depth_E_nm"]),
            "skin_depth_I_nm": np.float64(r_ref["skin_depth_I_nm"]),
            "cells_per_skin_depth": np.float64(r_ref["cells_per_skin_depth"]),
            "cells_per_skin_depth_E": np.float64(r_ref["cells_per_skin_depth_E"]),
            "cells_per_skin_depth_I": np.float64(r_ref["cells_per_skin_depth_I"]),
            "n_cells": np.int32(r_ref["n_cells"]),
            "n_steps": np.int32(r_ref["n_steps"]),
            "elapsed_s": np.float64(time.perf_counter() - t0),
        }
    else:
        raise ValueError(f"Unknown task kind {kind!r}")

    out_path = out_dir / f"task_{idx:02d}_{name}.npz"
    out_json = out_dir / f"task_{idx:02d}_{name}.json"
    np.savez_compressed(out_path, **payload)
    write_task_json(out_json, task, payload)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(
        f"[Task {idx:02d} {name}] saved to {out_path.name} "
        f"| dx_bg={float(payload['dx_background_nm']):.2f}nm, "
        f"dx_min={float(payload['dx_min_nm']):.2f}nm, "
        f"min_feat={float(payload['min_feature_nm']):.0f}nm ({float(payload['cells_across_feature']):.0f} cells), "
        f"delta_I={float(payload['skin_depth_nm']):.2f}nm ({float(payload['cells_per_skin_depth']):.2f} c/delta_I, "
        f"{float(payload['cells_per_skin_depth_E']):.2f} c/delta_E), "
        f"n_cells={int(payload['n_cells'])}, n_steps={int(payload['n_steps'])} "
        f"in {float(payload['elapsed_s']):.1f} s (peak RSS {peak_mb:.0f} MB)",
        flush=True,
    )
    return out_path


# %% [markdown]
# ## 5. Generating the complete figure suite
#
# `generate_all_figures()` reads the saved `.npz` files in `data/` and the diagnostic JSON archives in `data/diag/` to render the complete diagnostic figure suite (including the setup figures `00a_setup_geometry.png`, `00b_setup_mesh_and_excitation.png`, and `00_setup_table.md`, the time-domain `.gif` animation `06_pulse_propagation_dgrin.gif`, and the gap-diagnostic summary figure `15_gap_diagnostics.png`) in `figures/`.
#
# ### Complete figure list, captions & silver-model / dipole-ensemble inventory
#
# Rakić (1998) 1-pole Drude silver (`make_ag_drude()`) is kept as the tutorial's default metallic reflector material across Tasks 0–6 (`data/task_00..06.npz`), while Johnson & Christy (1972) 1-pole Drude silver (`make_ag_drude_jc()` in `gap_diagnostics.py`) and the dipole-count convergence ladders ($M \in \{1, 5, 9, 17\}$) are stored in `data/diag/*.json` and compared side by side in Section 6, Section 8.2, Figure 08b, and Figure 15:
#
# 0. `figures/00a_setup_geometry.png`, `figures/00b_setup_mesh_and_excitation.png`, and `figures/00_setup_table.md`: Two-figure setup layout and Markdown setup table built from live solver/grid/source/monitor objects via `wakis.schematic.save_setup_figures`: (00a) 6-layer $\text{dGRIN}$ $\mu\text{Horn}$ geometry ($h=3.0\;\mu\text{m}$, $\vartheta=60^\circ$, $M=17$ incoherent QW dipoles at $y_{\text{QW}}=150\;\text{nm}$, 5 representative sites shown) and `Near2FarMonitor2D` plane; (00b) graded mesh with 1× zoom around the $\text{Ag}$ mirror/throat skin-depth collar ($\Delta x_{\min}=3.0\;\text{nm}$, $5.13\;\text{cells}/\delta_I$) and Gaussian pulse excitation ($f_0 = 666.2\;\text{THz}$).
# 1. `figures/01_microled_horn_schematics.png`: 3-panel geometric comparison of the three tapered $\mu\text{Horn}$ collimators—Air ($\vartheta=72^\circ$), $\text{SiO}_2$ ($\vartheta=64^\circ$), and 6-layer $\text{dGRIN}$ ($\vartheta=60^\circ$)—showing the $\text{Ag}$ reflector, $600\;\text{nm}$ $\text{GaN}$ core, $200\;\text{nm}$ $\text{SiO}_2$ lateral passivation barriers, $\text{InGaN}$ QW dipole ensemble ($y_{\text{QW}} = 150\;\text{nm}$), and `Near2FarMonitor2D` plane. *(Geometry schematic; independent of Ag dispersion fit.)*
# 2. `figures/02_materials_and_tmm_reflectance.png`: (a) Calibrated 1-pole Drude $\varepsilon_{\text{Ag}}(\lambda)$ vs Rakić (1998) tabulated silver and field skin depth $\delta_E(\lambda)$; (b) vertical refractive-index step-down profile $n(y)$; (c) analytical $2\times 2$ TMM normal-incidence reflectance $R(\lambda)$ across $415\text{--}485\;\text{nm}$ overlaid with the $\text{InGaN}$ Gaussian spectrum $S(\lambda)$. *(Uses Rakić 1998 Ag in panel a.)*
# 3. `figures/03_sidewall_mesh_kottke_vs_staircase.png`: Full 2D FIT domain and zoomed $60^\circ$ $\text{Ag}$ sidewall mesh comparing voxel staircasing against Kottke subpixel tensor smoothing at $\Delta x_{\text{bg}} = 7.5\;\text{nm}$. *(Uses Rakić 1998 Ag high-frequency core $\varepsilon_\infty = 4.0$.)*
# 4. `figures/04_electric_field_wavefronts.png` and `figures/04b_electric_field_bare_vs_dgrin.png`: Steady-state $\operatorname{Re}(E_z)$ wavefronts at $\lambda_0 = 450\;\text{nm}$ for a single central $\text{TM}_z$ dipole ($x_d = 0$, $M = 1$, `--full` mesh) across the three $\mu\text{Horn}$ designs (`04`, 3 panels) and contrasting Bare vs 6-layer $\text{dGRIN}$ (`04b`, 2 panels) with `wakis.schematic.overlay_geometry`. *(Uses default Rakić 1998 Ag, $M = 1$ from `task_00..03.npz`.)*
# 5. `figures/05_poynting_power_flow_maps.png`: 3-panel time-averaged Poynting vector magnitude $|\mathbf{S}|$ and unit flow quivers reproducing Luce *et al.* (2026) Fig. 3g–i for Air ($72^\circ$), $\text{SiO}_2$ ($64^\circ$), and 6-layer $\text{dGRIN}$ ($60^\circ$). *(Uses default Rakić 1998 Ag, central dipole $M = 1$ from `task_01..03.npz`.)*
# 6. `figures/06_pulse_propagation_dgrin.gif`: Animated time-domain $E_z(x, y, t)$ pulse propagation through the 6-layer $\text{dGRIN}$ micro-horn ($t \in [2, 56]\;\text{fs}$). *(Uses default Rakić 1998 Ag, central dipole $M = 1$ from `task_03.npz`.)*
# 7. `figures/07_farfield_polar_patterns.png`: Absolute and peak-normalized 2D polar far-field extraction densities $\rho_{\text{LEE}}(\theta)$ ($[\%/\text{deg}]$) vs analytical $\cos\theta$ Lambertian emission and the $\pm 15^\circ$ target acceptance cone. *(Uses default Rakić 1998 Ag angular profiles from `task_00..03.npz`.)*
# 8. `figures/08_farfield_cartesian_and_paper_refs.png`: (a) Cartesian angular extraction density $\rho_{\text{LEE}}(\theta)$ (default Rakić 1998 Ag from `task_00..03.npz`); (b) converged `--full` ($M = 17$ for horns, $M = 9$ for Bare) bar comparison of Rakić (1998) Ag (`lee15_eq_pct`, `lee15_pct`, `lee15_wrk_pct`, `lee90_eq_pct`) and Johnson & Christy (1972) Ag (`jc_ag` $\text{LEE}_{15,\text{eq}}$ stars) against the published 2D FDTD benchmarks of Luce *et al.* (2026). *(Panel a uses Rakić 1998 Ag; Panel b shows both converged Rakić 1998 Ag and converged Johnson & Christy 1972 Ag vs Paper.)*
# 9. `figures/09_horn_angle_and_height_sweep.png`: On-axis ($M = 1$, $x_d = 0$) horn sidewall taper angle sweep $\vartheta \in [50^\circ, 78^\circ]$ at $h = 3.0\;\mu\text{m}$ and horn height sweep $h \in [0.6, 3.0]\;\mu\text{m}$ at paper $\vartheta_{\text{opt}}$ (cf. Fig. 3a–f of [1]). *(Uses default Rakić 1998 Ag, $M = 1$ from `task_04.npz`.)*
# 10. `figures/10_qw_position_sensitivity.png`: 2D on-axis ($M = 1$, $x_d = 0$) quantum-well vertical offset sensitivity across $\Delta y_{\text{QW}} \in [-40, +40]\;\text{nm}$ around $y_{\text{QW}} = 150\;\text{nm}$ comparing the 6-layer $\text{dGRIN}$ horn against the bare reference pixel. *(Uses default Rakić 1998 Ag, $M = 1$ from `task_05.npz`.)*
# 11. `figures/11_incoherent_dipole_convergence.png`: (a) Cumulative symmetrical dipole-pair partial averages ($N_{\text{dip}} \in \{1, 3, 5, 7, 9, 11\}$ across $[-250, +250]\;\text{nm}$) for the 6-layer $\text{dGRIN}$ horn; (b) physical incoherent power sum $\sum_i |E_i(\theta)|^2$ vs unphysical coherent field sum $|\sum_i E_i(\theta)|^2 / N_{\text{dip}}$. *(Uses default Rakić 1998 Ag from `task_03.npz`; see Fig. 15a for independent uniform-grid $M \in \{1, 5, 9, 17\}$ convergence across all four configurations.)*
# 12. `figures/12_polarization_and_spectral_breakdown.png`: (a) Converged `--full` ($M = 17$ for horns, $M = 9$ for Bare; Rakić 1998 Ag) per-polarization breakdown of $\text{LEE}_{15}$ into in-plane horizontal $J_x$ ($\text{TE}_x$), vertical $J_y$ ($\text{TE}_y$), and out-of-plane $J_z$ ($\text{TM}_z$); (b) broadband spectral extraction $\text{LEE}_{15}(\lambda)$ across $425\text{--}475\;\text{nm}$ from `task_00..03.npz`. *(Uses default Rakić 1998 Ag.)*
# 13. `figures/13_kottke_vs_staircase_convergence.png`: 5-level $\text{Ag}$ skin-depth resolution ladder ($\Delta x_{\min} \in \{15.0, 7.5, 5.0, 3.75, 2.50\}\;\text{nm}$, $1.03\text{--}6.15\;\text{cells}/\delta_I$) comparing Kottke subpixel smoothing against staircasing on the sloped $60^\circ$ $\text{dGRIN}$ sidewall. *(Uses default Rakić 1998 Ag, $M = 1$ from `task_06.npz`.)*
# 14. `figures/14_purcell_and_normalization_audit.png`: (a) Converged `--full` ($M = 17$ for horns, $M = 9$ for Bare; Rakić 1998 Ag) cavity Purcell factor $F_P = P_{\text{wrk}}/P_{\text{hom}}$ and 2D half-space directivity $D_{2\text{D}}$; (b) 2D collimation ratio $\eta_{15} = \text{LEE}_{15}/\text{LEE}_{90}$ vs Luce *et al.* (2026) and the 2D Lambertian $\sin(15^\circ) = 25.9\%$ limit. *(Uses default Rakić 1998 Ag.)*
# 15. `figures/15_gap_diagnostics.png`: **(a)** Equal-polarization forward extraction $\text{LEE}_{15,\text{eq}}(M)$ vs uniformly spaced QW dipole count $M \in \{1, 5, 9, 17\}$ across $[-240, +240]\;\text{nm}$ on the `--quick` mesh for all four configurations (Bare, Air, $\text{SiO}_2$, 6-layer $\text{dGRIN}$), with $\pm 0.5\;\text{pp}$ shaded convergence bands around $M = 17$ and `--full` Rakić Ag star markers at $M = 9$ and $M = 17$, demonstrating that $M = 5$ is not yet converged on $\text{SiO}_2$ ($28.17\% \to 24.00\% \to 24.13\%$) or $\text{dGRIN}$ ($39.04\% \to 31.37\% \to 30.19\%$); **(b)** converged `--full` ($M = 17$ for Air, $\text{SiO}_2$, $\text{dGRIN}$; $M = 9$ for Bare) bar comparison of Baseline Rakić (1998) Ag (`[8.03, 20.13, 25.45, 31.37]%`), Johnson & Christy (1972) Ag (`[9.89, 27.48, 31.11, 37.58]%`), and the Luce *et al.* (2026) 2D FDTD benchmarks (`[14.9, 27.5, 30.4, 37.7]%`), showing that with J&C (1972) silver and a converged dipole ensemble all three horns agree with the paper within $0.71\;\text{pp}$ while the bare reference still differs by $5.01\;\text{pp}$. *(Compares both Rakić 1998 Ag and Johnson & Christy 1972 Ag at converged $M$.)*


# %%
def _draw_horn_outline(ax, config, h_um=3.0, theta_deg=60.0, color="white", lw=1.1):
    """Overlay the microLED pixel and horn boundaries on a 2D (x [um], y [um]) axis."""
    ax.plot([-0.5, 0.5], [0.0, 0.0], color=color, lw=lw, alpha=0.85)
    ax.plot([-0.3, 0.3], [0.15, 0.15], color="#d946ef", ls="--", lw=1.0, alpha=0.9)
    ax.plot(
        [-0.3, -0.3, 0.3, 0.3], [0.0, 0.3, 0.3, 0.0], color=color, lw=0.9, alpha=0.75
    )
    if config != "bare" and h_um > 0:
        dx_h = h_um / np.tan(np.deg2rad(theta_deg))
        x_top = 0.5 + dx_h
        y_top = 0.3 + h_um
        ax.plot([-0.5, -x_top], [0.3, y_top], color=color, lw=lw + 0.3, alpha=0.9)
        ax.plot([0.5, x_top], [0.3, y_top], color=color, lw=lw + 0.3, alpha=0.9)
        if config == "dgrin":
            dh = h_um / 6.0
            for k in range(1, 6):
                yk = 0.3 + k * dh
                xk = 0.5 + (k * dh) / np.tan(np.deg2rad(theta_deg))
                ax.plot([-xk, xk], [yk, yk], color=color, ls=":", lw=0.75, alpha=0.65)


def _load_converged_diag_pair(diag_dir, cfg):
    """Load converged `--full` (`M=17` if available, else `M=9`) baseline and jc_ag dicts."""
    diag_dir = Path(diag_dir)
    for m_try in (17, 9):
        pb = diag_dir / f"baseline_{cfg}_M{m_try}_full.json"
        pj = diag_dir / f"jc_ag_{cfg}_M{m_try}_full.json"
        if pb.exists() and pj.exists():
            return json.loads(pb.read_text()), json.loads(pj.read_text())
    return None, None


def generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR):
    """
    Read `data/task_*.npz` and `data/diag/*.json` and generate all figures in `figures/`.
    """
    figstyle.apply_style()
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = data_dir / "diag"

    t0_d = np.load(data_dir / "task_00_bare_ref.npz")
    t1_d = np.load(data_dir / "task_01_air_horn_opt.npz")
    t2_d = np.load(data_dir / "task_02_sio2_horn_opt.npz")
    t3_d = np.load(data_dir / "task_03_dgrin_horn_opt.npz")
    t4_d = np.load(data_dir / "task_04_angle_and_height_sweep.npz")
    t5_d = np.load(data_dir / "task_05_qw_position_sweep.npz")
    t6_d = np.load(data_dir / "task_06_conv_kottke_vs_staircase.npz")

    cfg_map = {"bare": t0_d, "air": t1_d, "sio2": t2_d, "dgrin": t3_d}
    diag_base_map = {}
    diag_jc_map = {}
    for c_key in ("bare", "air", "sio2", "dgrin"):
        db, dj = _load_converged_diag_pair(diag_dir, c_key)
        if db is not None:
            diag_base_map[c_key] = db
            diag_jc_map[c_key] = dj
        else:
            diag_base_map[c_key] = cfg_map[c_key]
            diag_jc_map[c_key] = None
    colors = {
        "bare": figstyle.PALETTE[7],
        "air": figstyle.PALETTE[0],
        "sio2": figstyle.PALETTE[1],
        "dgrin": figstyle.PALETTE[2],
    }
    labels = {
        "bare": r"Bare reference ($h=0$)",
        "air": r"Air $\mu$Horn ($\vartheta=72^\circ$)",
        "sio2": r"SiO$_2$ $\mu$Horn ($\vartheta=64^\circ$)",
        "dgrin": r"6-layer dGRIN $\mu$Horn ($\vartheta=60^\circ$)",
    }

    # ------------------------------------------------------------------
    # Fig 00a & 00b: Standard setup figures & setup table
    # ------------------------------------------------------------------
    save_horn_setup_figures(fig_dir)

    # ------------------------------------------------------------------
    # Fig 01: 3-panel schematic comparison of the 3 tapered micro-horns
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.5, height=4.3, sharey=True)
    horn_specs_01 = [
        ("air", 3.0, 72.0, r"(a) Air $\mu$Horn ($\vartheta=72^\circ$, $M=17$)"),
        ("sio2", 3.0, 64.0, r"(b) SiO$_2$ $\mu$Horn ($\vartheta=64^\circ$, $M=17$)"),
        ("dgrin", 3.0, 60.0, r"(c) 6-layer dGRIN $\mu$Horn ($\vartheta=60^\circ$, $M=17$)"),
    ]
    for ax, (cfg, hh, th, title) in zip(axes, horn_specs_01):
        h_obj = build_horn_schematic_objects(config=cfg, h_horn=hh * 1e-6, theta_deg=th, dx=15e-9, dx_skin=7.5e-9, n_dip=17, n_pml=8)
        schematic.plot_schematic(
            h_obj["solver"],
            plane="xy",
            ax=ax,
            unit="um",
            structures=h_obj["structures"],
            sources=h_obj["sources"],
            monitors=h_obj["monitors"],
            show_mesh=False,
            show_wave_packet=False,
            title=title,
        )
        tidy_horn_schematic_axes(ax, config=cfg, h_um=hh, theta_deg=th, n_dip=17, n_pml=8, xlim=(-2.6, 2.6), ylim=(-0.30, 3.85))
    axes[0].set_ylabel(r"Vertical emission coordinate $y$ [$\mu$m]")
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "01_microled_horn_schematics.png", close=True)

    # ------------------------------------------------------------------
    # Fig 02: Material dispersion (Ag Drude vs Rakic), LED spectrum & TMM
    # ------------------------------------------------------------------
    lams_nm = np.linspace(400.0, 520.0, 241)
    lams_m = lams_nm * 1e-9
    freqs_plot = c_light / lams_m
    ag_drude = make_ag_drude()
    ag_rakic = dispersive_lib["silver"]
    eps_d = ag_drude.epsilon(freqs_plot)
    eps_r = ag_rakic.epsilon(freqs_plot)
    skin_e_nm = np.array([ag_skin_depth(wl * 1e-9, ag_drude)[0] * 1e9 for wl in lams_nm])

    R_bare, _ = tmm_normal_reflectance(lams_m, n_in=N_GAN, n_out=N_AIR)
    R_sio2, _ = tmm_normal_reflectance(
        lams_m, n_in=N_GAN, n_out=N_AIR, layer_indices=(N_SIO2,), layer_thicknesses=(3.0e-6,)
    )
    dgrin_mats = [n for _, n in DGRIN_STACK[:-1]]
    dgrin_thks = [0.50e-6] * len(dgrin_mats)
    R_dgrin, _ = tmm_normal_reflectance(
        lams_m, n_in=N_GAN, n_out=N_AIR, layer_indices=dgrin_mats, layer_thicknesses=dgrin_thks
    )
    S_led_plot = led_gaussian_spectrum(lams_m)

    fig, (ax0, ax1, ax2) = figstyle.new_figure(nrows=1, ncols=3, width=13.5, height=4.0)
    ax0.plot(lams_nm, eps_r.real, "k--", lw=1.4, label=r"Rakic Ag $\mathrm{Re}(\varepsilon)$")
    ax0.plot(lams_nm, eps_d.real, color=colors["air"], lw=1.8, label=r"1-pole Drude $\mathrm{Re}(\varepsilon)$")
    ax0.plot(lams_nm, eps_r.imag, "r:", lw=1.4, label=r"Rakic Ag $\mathrm{Im}(\varepsilon)$")
    ax0.plot(lams_nm, eps_d.imag, color=colors["sio2"], lw=1.8, label=r"1-pole Drude $\mathrm{Im}(\varepsilon)$")
    ax0.plot(lams_nm, 0.1 * skin_e_nm, color=colors["dgrin"], ls="-.", lw=1.6, label=r"Skin depth $\delta_E / 10$ [nm]")
    ax0.axvline(450.0, color="0.4", ls="--", lw=1.0)
    ax0.set(
        xlabel=r"Wavelength $\lambda$ [nm]",
        ylabel=r"$\varepsilon_{\mathrm{Ag}}(\lambda)$ and $\delta_E / 10$ [nm]",
        title=r"(a) Silver Drude fit vs Rakic (1998) & skin depth",
    )
    ax0.legend(loc="lower left", fontsize=7.0)

    steps_y = [0.0, 0.3, 0.8, 1.3, 1.8, 2.3, 2.8, 3.3]
    steps_n = [N_GAN, 2.37, 2.05, 1.77, 1.46, 1.38, 1.00, 1.00]
    ax1.step(steps_y, steps_n, where="post", color=colors["dgrin"], lw=2.2, label="6-layer dGRIN profile")
    ax1.step(
        [0.0, 0.3, 3.3], [N_GAN, N_SIO2, 1.0], where="post", color=colors["sio2"], ls="--", lw=1.6, label=r"Homogeneous SiO$_2$"
    )
    ax1.step(
        [0.0, 0.3, 3.3], [N_GAN, 1.0, 1.0], where="post", color=colors["air"], ls=":", lw=1.6, label="Unfilled Air"
    )
    ax1.set(
        xlabel=r"Vertical coordinate $y$ [$\mu$m]",
        ylabel=r"Refractive index $n(y)$",
        ylim=(0.85, 2.65),
        title=r"(b) Refractive-index step-down along $+y$",
    )
    ax1.legend(loc="upper right", fontsize=8.0)

    ax2.plot(lams_nm, 100.0 * R_bare, color=colors["bare"], lw=1.8, label=f"Bare GaN/Air ({100*R_bare[0]:.1f}%)")
    ax2.plot(lams_nm, 100.0 * R_sio2, color=colors["sio2"], lw=1.4, alpha=0.85, label=r"3 $\mu$m SiO$_2$ slab")
    ax2.plot(lams_nm, 100.0 * R_dgrin, color=colors["dgrin"], lw=1.8, label="6-layer dGRIN stack")
    ax2.fill_between(
        lams_nm,
        0,
        25.0 * S_led_plot,
        color="#a21caf",
        alpha=0.18,
        label=r"InGaN spectrum $S(\lambda)$ (scaled)",
    )
    ax2.set(
        xlabel=r"Wavelength $\lambda$ [nm]",
        ylabel=r"Normal-incidence TMM reflectance $R(\lambda)$ [%]",
        xlim=(415, 485),
        ylim=(0, 26),
        title=r"(c) Analytical 1D TMM normal reflectance",
    )
    ax2.legend(loc="upper right", fontsize=7.8)
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "02_materials_and_tmm_reflectance.png", close=True)

    # ------------------------------------------------------------------
    # Fig 03: Sidewall mesh zoom: Kottke vs Staircase on 60 deg sidewall
    # ------------------------------------------------------------------
    dx_bg_nm_fig = float(t3_d["dx_background_nm"])
    dx_bg_m_fig = dx_bg_nm_fig * 1e-9
    m_kot = extract_2d_eps_map(
        "dgrin", h_horn=3.0e-6, theta_deg=60.0, dx=dx_bg_m_fig, averaging="kottke"
    )
    m_st = extract_2d_eps_map(
        "dgrin", h_horn=3.0e-6, theta_deg=60.0, dx=dx_bg_m_fig, averaging="staircase"
    )

    fig, (ax0, ax1, ax2) = figstyle.new_figure(nrows=1, ncols=3, width=13.5, height=4.1)
    xc_um = m_kot["x_centers"] * 1e6
    yc_um = m_kot["y_centers"] * 1e6

    disp_full = np.where(m_kot["f_ag"] > 0.5, -2.0, m_kot["eps_x"])
    im0 = ax0.pcolormesh(
        xc_um, yc_um, disp_full.T, cmap=figstyle.CMAP_EPS, vmin=-2.0, vmax=6.2, shading="auto"
    )
    overlay_horn_geometry(ax0, config="dgrin", h_um=3.0, theta_deg=60.0, line_color="white", fontsize=5.8)
    ax0.add_patch(
        mpatches.Rectangle(
            (0.72, 0.65), 0.30, 0.30, fill=False, edgecolor="#f43f5e", lw=1.6, zorder=10
        )
    )
    ax0.set(
        xlabel=r"$x$ [$\mu$m]",
        ylabel=r"$y$ [$\mu$m]",
        xlim=(-2.5, 2.5),
        ylim=(-0.2, 3.65),
        title=f"(a) Full 2D FIT domain ($\\Delta x_{{\\mathrm{{bg}}}} = {dx_bg_nm_fig:.1f}$ nm)",
    )
    fig.colorbar(im0, ax=ax0, label=r"Effective $\varepsilon_x$ (Ag shown at $-2$)")

    bbox_sub = {"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "#475569", "lw": 0.45, "alpha": 0.86}
    for ax, m_dat, tag in [
        (ax1, m_st, r"(b) Staircase sidewall ($\vartheta=60^\circ$)"),
        (ax2, m_kot, r"(c) Kottke subpixel sidewall ($\vartheta=60^\circ$)"),
    ]:
        disp_z = m_dat["eps_x"]
        im = ax.pcolormesh(
            xc_um,
            yc_um,
            disp_z.T,
            cmap=figstyle.CMAP_MAGNITUDE,
            vmin=1.5,
            vmax=6.0,
            shading="auto",
            edgecolors="0.3",
            linewidth=0.25,
        )
        y_line = np.linspace(0.60, 1.00, 100)
        x_line = 0.50 + (y_line - 0.30) / np.tan(np.deg2rad(60.0))
        ax.plot(x_line, y_line, color="#38bdf8", lw=2.0, label=r"Exact $60^\circ$ Ag sidewall")
        ax.axhline(0.80, color="white", ls="--", lw=1.1, label=r"Nb$_2$O$_5$/Si$_3$N$_4$ interface")
        ax.text(0.76, 0.72, "Nb₂O₅\n(n=2.37)", ha="center", va="center", fontsize=6.6, bbox=bbox_sub, zorder=8)
        ax.text(0.78, 0.89, "Si₃N₄\n(n=2.05)", ha="center", va="center", fontsize=6.6, bbox=bbox_sub, zorder=8)
        ax.text(0.95, 0.72, "Ag horn\n(ε_∞=4.0)", ha="center", va="center", fontsize=6.6, bbox=bbox_sub, zorder=8)
        ax.set(
            xlabel=r"$x$ [$\mu$m]",
            ylabel=r"$y$ [$\mu$m]",
            xlim=(0.72, 1.02),
            ylim=(0.65, 0.95),
            title=tag,
        )
        ax.legend(loc="upper left", fontsize=7.0)
        fig.colorbar(im, ax=ax, label=r"High-freq cell $\varepsilon_{\infty,\mathrm{eff},x}$")
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "03_sidewall_mesh_kottke_vs_staircase.png", close=True)

    # ------------------------------------------------------------------
    # Fig 04: 3-panel steady-state Re(E_z) wavefronts across the 3 micro-horns
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.5, height=4.3, sharey=True)
    horn_wave_specs = [
        ("air", 3.0, 72.0, r"(a) Air $\mu$Horn ($\vartheta=72^\circ$)"),
        ("sio2", 3.0, 64.0, r"(b) SiO$_2$ $\mu$Horn ($\vartheta=64^\circ$)"),
        ("dgrin", 3.0, 60.0, r"(c) 6-layer dGRIN $\mu$Horn ($\vartheta=60^\circ$)"),
    ]
    for ax, (cfg, hh, th, title) in zip(axes, horn_wave_specs):
        d = cfg_map[cfg]
        x_u = d["dft_x"] * 1e6
        y_u = d["dft_y"] * 1e6
        ez = d["dft_Ez_real"]
        vmax = 0.35 * float(np.percentile(np.abs(ez), 99.2))
        im = ax.pcolormesh(
            x_u, y_u, ez.T, cmap=figstyle.CMAP_DIVERGING, vmin=-vmax, vmax=vmax, shading="auto"
        )
        overlay_horn_geometry(ax, config=cfg, h_um=hh, theta_deg=th, line_color="black", fontsize=5.8)
        ax.set(
            xlabel=r"$x$ [$\mu$m]",
            xlim=(-2.4, 2.4),
            ylim=(-0.15, 3.65),
            title=title,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=r"$\mathrm{Re}(E_z)$ [arb.]")
    axes[0].set_ylabel(r"Vertical coordinate $y$ [$\mu$m]")
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "04_electric_field_wavefronts.png", close=True)

    # ------------------------------------------------------------------
    # Fig 04b: 2-panel Re(E_z) wavefront comparison (Bare vs 6-layer dGRIN)
    # ------------------------------------------------------------------
    fig_b, axes_b = figstyle.new_figure(nrows=1, ncols=2, width=10.2, height=4.2, sharey=True)
    for ax, (cfg, hh, th, title) in zip(
        axes_b,
        [
            ("bare", 0.0, 90.0, r"(a) Bare reference pixel ($h=0$, TIR trapped)"),
            ("dgrin", 3.0, 60.0, r"(b) 6-layer dGRIN $\mu$Horn ($h=3.0\;\mu$m, $\vartheta=60^\circ$)"),
        ],
    ):
        d = cfg_map[cfg]
        x_u = d["dft_x"] * 1e6
        y_u = d["dft_y"] * 1e6
        ez = d["dft_Ez_real"]
        vmax = 0.35 * float(np.percentile(np.abs(ez), 99.2))
        im = ax.pcolormesh(
            x_u, y_u, ez.T, cmap=figstyle.CMAP_DIVERGING, vmin=-vmax, vmax=vmax, shading="auto"
        )
        overlay_horn_geometry(ax, config=cfg, h_um=hh, theta_deg=th, line_color="black", fontsize=6.0)
        ax.set(
            xlabel=r"$x$ [$\mu$m]",
            xlim=(-2.4, 2.4),
            ylim=(-0.15, 3.65),
            title=title,
        )
        fig_b.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=r"$\mathrm{Re}(E_z)$ [arb.]")
    axes_b[0].set_ylabel(r"Vertical coordinate $y$ [$\mu$m]")
    figstyle.finish(fig_b)
    figstyle.save(fig_b, fig_dir / "04b_electric_field_bare_vs_dgrin.png", close=True)

    # ------------------------------------------------------------------
    # Fig 05: 3-panel Poynting vector power-flow maps (reproducing Fig. 3g-i)
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.5, height=4.4, sharey=True)
    horn_specs = [
        ("air", 3.0, 72.0, r"(a) Air $\mu$Horn ($\vartheta=72^\circ$, Fig. 3g)"),
        ("sio2", 3.0, 64.0, r"(b) SiO$_2$ $\mu$Horn ($\vartheta=64^\circ$, Fig. 3h)"),
        ("dgrin", 3.0, 60.0, r"(c) 6-layer dGRIN $\mu$Horn ($\vartheta=60^\circ$, Fig. 3i)"),
    ]
    for ax, (cfg, hh, th, title) in zip(axes, horn_specs):
        d = cfg_map[cfg]
        x_u = d["dft_x"] * 1e6
        y_u = d["dft_y"] * 1e6
        sx = d["dft_Sx"]
        sy = d["dft_Sy"]
        smag = np.hypot(sx, sy)
        s_norm = smag / max(float(np.percentile(smag, 99.5)), 1e-30)
        log_s = np.log10(np.clip(s_norm, 1e-4, 1.0))

        im = ax.pcolormesh(
            x_u, y_u, log_s.T, cmap=figstyle.CMAP_MAGNITUDE, vmin=-3.5, vmax=0.0, shading="auto"
        )
        overlay_horn_geometry(ax, config=cfg, h_um=hh, theta_deg=th, line_color="cyan", fontsize=5.8)

        step_x = max(1, len(x_u) // 22)
        step_y = max(1, len(y_u) // 22)
        Xq, Yq = np.meshgrid(x_u[::step_x], y_u[::step_y], indexing="ij")
        sx_q = sx[::step_x, ::step_y]
        sy_q = sy[::step_x, ::step_y]
        mag_q = np.hypot(sx_q, sy_q)
        keep = (mag_q > 1e-3 * float(np.max(smag))) & (Yq > 0.05)
        ux = np.where(keep, sx_q / np.maximum(mag_q, 1e-30), 0.0)
        uy = np.where(keep, sy_q / np.maximum(mag_q, 1e-30), 0.0)
        ax.quiver(
            Xq[keep],
            Yq[keep],
            ux[keep],
            uy[keep],
            color="white",
            alpha=0.82,
            scale=24,
            width=0.004,
            zorder=8,
        )
        ax.set(
            xlabel=r"Lateral coordinate $x$ [$\mu$m]",
            xlim=(-2.4, 2.4),
            ylim=(-0.15, 3.65),
            title=title,
        )
        fig.colorbar(
            im, ax=ax, fraction=0.046, pad=0.03, label=r"$\log_{10}(|\mathbf{S}| / S_{\max})$"
        )
    axes[0].set_ylabel(r"Vertical coordinate $y$ [$\mu$m]")
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "05_poynting_power_flow_maps.png", close=True)

    # ------------------------------------------------------------------
    # Fig 06: Time-domain pulse propagation GIF through 6-layer dGRIN horn
    # ------------------------------------------------------------------
    if "movie_frames" in t3_d:
        mx = t3_d["movie_x"] * 1e6
        my = t3_d["movie_y"] * 1e6
        mtimes = t3_d["movie_times_fs"]
        mframes = np.asarray(t3_d["movie_frames"], dtype=float) * (float(t3_d["movie_frames_scale"]) if "movie_frames_scale" in t3_d else 1.0)
        vmax_m = 0.25 * float(np.max(np.abs(mframes)))

        fig_m, ax_m = figstyle.new_figure(nrows=1, ncols=1, width=6.2, height=5.0)
        im_m = ax_m.pcolormesh(
            mx,
            my,
            mframes[0].T,
            cmap=figstyle.CMAP_DIVERGING,
            vmin=-vmax_m,
            vmax=vmax_m,
            shading="auto",
        )
        overlay_horn_geometry(ax_m, config="dgrin", h_um=3.0, theta_deg=60.0, line_color="black", fontsize=6.4)
        ax_m.set(
            xlabel=r"Lateral coordinate $x$ [$\mu$m]",
            ylabel=r"Vertical coordinate $y$ [$\mu$m]",
            xlim=(-2.4, 2.4),
            ylim=(-0.15, 3.65),
        )
        title_m = ax_m.set_title(
            f"6-layer dGRIN $\\mu$Horn pulse propagation: $t = {mtimes[0]:.1f}$ fs"
        )
        fig_m.colorbar(im_m, ax=ax_m, label=r"$E_z(x, y, t)$ [arb.]")
        figstyle.finish(fig_m)

        def _update_frame(k):
            im_m.set_array(mframes[k].T.ravel())
            title_m.set_text(
                f"6-layer dGRIN $\\mu$Horn pulse propagation: $t = {mtimes[k]:.1f}$ fs"
            )
            return im_m, title_m

        anim = FuncAnimation(
            fig_m, _update_frame, frames=len(mtimes), interval=110, blit=False
        )
        anim.save(
            fig_dir / "06_pulse_propagation_dgrin.gif",
            writer=PillowWriter(fps=9),
            dpi=100,
        )
        plt.close(fig_m)

    # ------------------------------------------------------------------
    # Fig 07: Polar radiation patterns rho_LEE(theta) vs 2D Lambertian
    # ------------------------------------------------------------------
    from matplotlib.collections import PolyCollection

    fig = plt.figure(figsize=(11.5, 4.8))
    ax0 = fig.add_subplot(121, projection="polar")
    ax1 = fig.add_subplot(122, projection="polar")

    ang_rad = t3_d["angles_rad"]
    for cfg in ("bare", "air", "sio2", "dgrin"):
        d = cfg_map[cfg]
        rho = d["rho_lee_deg"]
        ax0.plot(ang_rad, rho, color=colors[cfg], lw=2.0, label=labels[cfg])
        ax1.plot(
            ang_rad,
            rho / max(float(np.max(rho)), 1e-30),
            color=colors[cfg],
            lw=2.0,
            label=labels[cfg],
        )
    ax1.plot(
        ang_rad,
        np.cos(ang_rad),
        "k--",
        lw=1.4,
        label=r"Analytical Lambertian $\cos\theta$",
    )
    for ax, ttl in [
        (ax0, r"(a) Absolute $\rho_{\mathrm{LEE}}(\theta)$ [%/deg]"),
        (ax1, r"(b) Peak-normalized radiation pattern"),
    ]:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        ax.fill_between(
            np.deg2rad(np.linspace(-15, 15, 61)),
            0,
            ax.get_rmax(),
            color="#fde047",
            alpha=0.22,
            label=r"$\pm 15^\circ$ target cone",
        )
        ax.add_collection(PolyCollection([[(0.0, 0.0), (0.0, 0.0)]], alpha=0.0))
        ax.set_title(ttl, pad=14, fontsize=10)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=8.0)
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "07_farfield_polar_patterns.png", close=True)

    # ------------------------------------------------------------------
    # Fig 08: Cartesian angular LEE density + comparison vs Luce et al. (2026)
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=12.4, height=4.3)
    ang_deg = t3_d["angles_deg"]
    ax0.axvspan(-15.0, 15.0, color="#fef08a", alpha=0.45, label=r"$\Gamma = \pm 15^\circ$ cone")
    for cfg in ("bare", "air", "sio2", "dgrin"):
        d = cfg_map[cfg]
        ax0.plot(
            ang_deg,
            d["rho_lee_deg"],
            color=colors[cfg],
            lw=2.0,
            label=f"{labels[cfg]} (FWHM={float(d['fwhm_deg']):.1f}$^\\circ$)",
        )
    ax0.set(
        xlabel=r"Far-field polar angle $\theta$ [deg]",
        ylabel=r"Angular extraction density $\rho_{\mathrm{LEE}}(\theta)$ [%/deg]",
        xlim=(-90, 90),
        title=r"(a) Incoherent far-field extraction density $\rho_{\mathrm{LEE}}(\theta)$",
    )
    ax0.legend(loc="upper right", fontsize=7.6)

    x_idx = np.arange(4)
    cfg_order = ("bare", "air", "sio2", "dgrin")
    our_15_eq = [float(diag_base_map[c]["lee15_eq_pct"]) for c in cfg_order]
    our_15 = [float(diag_base_map[c]["lee15_pct"]) for c in cfg_order]
    our_15_wrk = [float(diag_base_map[c]["lee15_wrk_pct"]) for c in cfg_order]
    our_90_eq = [float(diag_base_map[c]["lee90_eq_pct"]) for c in cfg_order]
    jc_15_eq = [
        float(diag_jc_map[c]["lee15_eq_pct"])
        if diag_jc_map[c] is not None
        else np.nan
        for c in cfg_order
    ]
    pub_2d_15 = [PAPER_2D_REFS[c]["lee15_pct"] for c in cfg_order]
    pub_2d_90 = [PAPER_2D_REFS[c]["lee90_pct"] for c in cfg_order]

    w_bar = 0.19
    ax1.bar(
        x_idx - 1.5 * w_bar,
        our_15_eq,
        width=w_bar,
        color=colors["dgrin"],
        label=r"Rakić Ag $\mathrm{LEE}_{15,\mathrm{eq}}$ ($M=17$; Bare $M=9$)",
    )
    ax1.bar(
        x_idx - 0.5 * w_bar,
        our_15,
        width=w_bar,
        color="#10b981",
        alpha=0.80,
        label=r"Rakić Ag $\mathrm{LEE}_{15}$ ($\mathrm{TM}_z\ 2\times$)",
    )
    ax1.bar(
        x_idx + 0.5 * w_bar,
        our_15_wrk,
        width=w_bar,
        color="#6ee7b7",
        alpha=0.85,
        label=r"Rakić Ag $\mathrm{LEE}_{15,\mathrm{wrk}}$ ($P_{\mathrm{wrk}}$)",
    )
    ax1.bar(
        x_idx + 1.5 * w_bar,
        our_90_eq,
        width=w_bar,
        color=colors["air"],
        alpha=0.80,
        label=r"Rakić Ag $\mathrm{LEE}_{90,\mathrm{eq}}$ ($w_{\mathrm{pol}}=1/3$)",
    )
    if np.all(np.isfinite(jc_15_eq)):
        ax1.scatter(
            x_idx - 1.5 * w_bar,
            jc_15_eq,
            marker="*",
            s=115,
            color="#f59e0b",
            edgecolor="black",
            zorder=7,
            label=r"J&C (1972) Ag $\mathrm{LEE}_{15,\mathrm{eq}}$ ($M=17$; Bare $M=9$)",
        )
    ax1.scatter(
        x_idx - 1.5 * w_bar,
        pub_2d_15,
        marker="D",
        s=65,
        color="#dc2626",
        zorder=6,
        label=r"Luce et al. (2026) 2D FDTD $\mathrm{LEE}_{15}$ (Fig. 3a–c)",
    )
    ax1.scatter(
        x_idx + 1.5 * w_bar,
        pub_2d_90,
        marker="^",
        s=65,
        color="#7c3aed",
        zorder=6,
        label=r"Luce et al. (2026) 2D FDTD $\mathrm{LEE}_{90}$ (Sec. A.2 / Fig. 3d–f)",
    )
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(
        [
            "Bare ($h=0$)\n($M=9$)",
            "Air ($72^\\circ$)\n($M=17$)",
            "SiO$_2$ ($64^\\circ$)\n($M=17$)",
            "6-layer dGRIN ($60^\\circ$)\n($M=17$)",
        ]
    )
    ax1.set(
        ylabel="Light Extraction Efficiency [%]",
        ylim=(0, 88),
        title=r"(b) Converged 2D FIT (Rakić & J&C Ag) vs Luce et al. (2026)",
    )
    ax1.legend(loc="upper left", fontsize=6.5)
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "08_farfield_cartesian_and_paper_refs.png", close=True)

    # ------------------------------------------------------------------
    # Fig 09: On-axis (N_dip=1) sidewall angle & horn height sweeps (cf. Fig. 3a-f)
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=12.0, height=4.2)
    sw_angles = t4_d["angles_deg"]
    sw_heights = t4_d["heights_um"]
    for cfg in ("air", "sio2", "dgrin"):
        ax0.plot(
            sw_angles,
            t4_d[f"{cfg}_angle_lee15"],
            "o-",
            color=colors[cfg],
            lw=2.0,
            label=labels[cfg],
        )
        ref = PAPER_2D_REFS[cfg]
        ax0.scatter(
            [ref["theta_deg"]],
            [ref["lee15_pct"]],
            marker="*",
            s=140,
            color=colors[cfg],
            edgecolor="black",
            zorder=7,
            label=f"Paper 2D opt. ({ref['theta_deg']:.0f}$^\\circ$, {ref['lee15_pct']:.1f}%)",
        )
        ax1.plot(
            sw_heights,
            t4_d[f"{cfg}_height_lee15"],
            "s-",
            color=colors[cfg],
            lw=2.0,
            label=f"{cfg.upper()} $\\mathrm{{LEE}}_{{15}}$",
        )
        ax1.plot(
            sw_heights,
            t4_d[f"{cfg}_height_lee90"],
            "^--",
            color=colors[cfg],
            lw=1.3,
            alpha=0.75,
            label=f"{cfg.upper()} $\\mathrm{{LEE}}_{{90}}$",
        )
    ax0.set(
        xlabel=r"Horn sidewall taper angle $\vartheta$ [deg]",
        ylabel=r"On-axis ($N_{\mathrm{dip}}=1$) $\mathrm{LEE}_{15}$ [%]",
        title=r"(a) On-axis ($N_{\mathrm{dip}}=1$) angle sweep at $h = 3.0\;\mu\mathrm{m}$ (cf. Fig. 3a–c)",
    )
    ax0.legend(loc="best", fontsize=7.3, ncol=2)

    ax1.set(
        xlabel=r"Micro-horn height $h$ [$\mu$m]",
        ylabel=r"On-axis ($N_{\mathrm{dip}}=1$) $\mathrm{LEE}_{15},\;\mathrm{LEE}_{90}$ [%]",
        title=r"(b) On-axis ($N_{\mathrm{dip}}=1$) height sweep at paper $\vartheta_{\mathrm{opt}}$ (cf. Fig. 3a–f)",
    )
    ax1.legend(loc="best", fontsize=7.5, ncol=2)
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "09_horn_angle_and_height_sweep.png", close=True)

    # ------------------------------------------------------------------
    # Fig 10: 2D on-axis (N_dip=1) QW vertical offset sensitivity (2D dGRIN vs 2D bare)
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.8, height=4.1)
    dy_nm = t5_d["dy_nm"]
    ax0.plot(
        dy_nm,
        t5_d["dgrin_lee15"],
        "o-",
        color=colors["dgrin"],
        lw=2.0,
        label=r"2D 6-layer dGRIN $\mu$Horn $\mathrm{LEE}_{15}$ ($N_{\mathrm{dip}}=1$)",
    )
    ax0.plot(
        dy_nm,
        t5_d["bare_lee15"],
        "s-",
        color=colors["bare"],
        lw=2.0,
        label=r"2D Bare reference $\mathrm{LEE}_{15}$ ($N_{\mathrm{dip}}=1$)",
    )
    ax0.plot(
        dy_nm,
        t5_d["dgrin_lee90"],
        "o--",
        color=colors["dgrin"],
        lw=1.3,
        alpha=0.75,
        label=r"2D 6-layer dGRIN $\mu$Horn $\mathrm{LEE}_{90}$ ($N_{\mathrm{dip}}=1$)",
    )
    ax0.plot(
        dy_nm,
        t5_d["bare_lee90"],
        "s--",
        color=colors["bare"],
        lw=1.3,
        alpha=0.75,
        label=r"2D Bare reference $\mathrm{LEE}_{90}$ ($N_{\mathrm{dip}}=1$)",
    )
    ax0.axvline(0.0, color="0.5", ls=":", lw=1.0)
    ax0.axhline(
        100.0,
        color="#dc2626",
        ls=":",
        lw=1.0,
        alpha=0.65,
        label=r"$P_{\mathrm{ext}}/P_{\mathrm{hom}}=100\%$ ($F_P>1$ at $\Delta y_{\mathrm{QW}}<0$)",
    )
    ax0.set(
        xlabel=r"Quantum-well vertical offset $\Delta y_{\mathrm{QW}}$ from 150 nm [nm]",
        ylabel=r"Hom.-normalized extraction $F_P \cdot \mathrm{LEE}_{\mathrm{wrk}}$ [%]",
        title=r"(a) 2D on-axis ($N_{\mathrm{dip}}=1$) $\mathrm{LEE}_{15}$ & $\mathrm{LEE}_{90}$ vs $\Delta y_{\mathrm{QW}}$",
    )
    ax0.legend(loc="best", fontsize=7.0)

    ax1.plot(
        dy_nm,
        t5_d["enhancement_15"],
        "D-",
        color="#7c3aed",
        lw=2.0,
        label=r"2D FIT on-axis $\mathrm{LEE}_{15}^{\mathrm{dGRIN}} / \mathrm{LEE}_{15}^{\mathrm{bare}}$ ($N_{\mathrm{dip}}=1$)",
    )
    ax1.axhline(
        PAPER_2D_REFS["dgrin"]["enh15"],
        color="#dc2626",
        ls="--",
        lw=1.4,
        label=f"Luce et al. (2026) 2D ensemble ratio ({PAPER_2D_REFS['dgrin']['enh15']:.2f}x at $\\Delta y_{{\\mathrm{{QW}}}}=0$)",
    )
    ax1.set(
        xlabel=r"Quantum-well vertical offset $\Delta y_{\mathrm{QW}}$ [nm]",
        ylabel=r"Forward enhancement factor $\mathrm{LEE}_{15}^{\mathrm{dGRIN}} / \mathrm{LEE}_{15}^{\mathrm{bare}}$",
        title=r"(b) 2D dGRIN / 2D bare $\mathrm{LEE}_{15}$ ratio across 80 nm QW window",
    )
    ax1.legend(loc="best", fontsize=7.6)
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "10_qw_position_sensitivity.png", close=True)

    # ------------------------------------------------------------------
    # Fig 11: Incoherent dipole count convergence & coherent vs incoherent
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.8, height=4.1)
    ndip = t3_d["conv_ndip_full"]
    c15 = t3_d["conv_lee15"]
    c90 = t3_d["conv_lee90"]
    ax0.plot(ndip, c15, "o-", color=colors["dgrin"], lw=2.0, label=r"dGRIN $\mathrm{LEE}_{15}(N_{\mathrm{dip}})$")
    ax0.plot(ndip, c90, "s--", color=colors["air"], lw=1.8, label=r"dGRIN $\mathrm{LEE}_{90}(N_{\mathrm{dip}})$")
    ax0.axhline(c15[-1], color=colors["dgrin"], ls=":", lw=1.1)
    ax0.set(
        xlabel=r"Total QW dipole positions across active width $N_{\mathrm{dip}}$",
        ylabel="Incoherently averaged LEE [%]",
        xticks=ndip,
        title=r"(a) Convergence of incoherent power average with $N_{\mathrm{dip}}$",
    )
    ax0.legend(loc="best", fontsize=8.5)

    ax1.plot(
        ang_deg,
        t3_d["rho_inc_z_deg"],
        color=colors["dgrin"],
        lw=2.2,
        label=r"Incoherent power sum $\sum_i |E_i(\theta)|^2$ (correct)",
    )
    ax1.plot(
        ang_deg,
        t3_d["rho_coh_z_deg"],
        color="#dc2626",
        ls="--",
        lw=1.6,
        label=r"Coherent field sum $|\sum_i E_i(\theta)|^2 / N_{\mathrm{dip}}$ (unphysical)",
    )
    ax1.axvspan(-15.0, 15.0, color="#fef08a", alpha=0.4)
    ax1.set(
        xlabel=r"Far-field polar angle $\theta$ [deg]",
        ylabel=r"TM$_z$ angular density $\rho_{\mathrm{LEE},z}(\theta)$ [%/deg]",
        xlim=(-90, 90),
        title="(b) Why spontaneous emission requires incoherent power averaging",
    )
    ax1.legend(loc="upper right", fontsize=8.0)
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "11_incoherent_dipole_convergence.png", close=True)

    # ------------------------------------------------------------------
    # Fig 12: Per-polarization breakdown (TE_x, TE_y, TM_z) & spectral LEE(lambda)
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=12.0, height=4.1)
    px = [float(diag_base_map[c]["lee15_pol_x"]) for c in cfg_order]
    py = [float(diag_base_map[c]["lee15_pol_y"]) for c in cfg_order]
    pz = [float(diag_base_map[c]["lee15_pol_z"]) for c in cfg_order]
    w_p = 0.24
    ax0.bar(
        x_idx - w_p,
        px,
        width=w_p,
        color=colors["air"],
        label=r"In-plane horizontal $J_x$ ($\mathrm{TE}_x$)",
    )
    ax0.bar(
        x_idx, py, width=w_p, color=colors["sio2"], label=r"Vertical $J_y$ ($\mathrm{TE}_y$)"
    )
    ax0.bar(
        x_idx + w_p,
        pz,
        width=w_p,
        color=colors["dgrin"],
        label=r"Out-of-plane $J_z$ ($\mathrm{TM}_z$)",
    )
    ax0.set_xticks(x_idx)
    ax0.set_xticklabels(
        [
            "Bare\n($M=9$)",
            "Air ($72^\\circ$)\n($M=17$)",
            "SiO$_2$ ($64^\\circ$)\n($M=17$)",
            "dGRIN ($60^\\circ$)\n($M=17$)",
        ]
    )
    ax0.set(
        ylabel=r"Forward cone extraction $\mathrm{LEE}_{15}$ [%]",
        title=r"(a) Polarization breakdown of $\mathrm{LEE}_{15}$ (Rakić Ag, converged $M$)",
    )
    ax0.legend(loc="upper left", fontsize=8.0)

    for cfg in cfg_order:
        d = cfg_map[cfg]
        wl_nm = d["wavelengths"] * 1e9
        order = np.argsort(wl_nm)
        ax1.plot(
            wl_nm[order],
            d["lee15_spectrum"][order],
            "o-",
            ms=3.5,
            color=colors[cfg],
            lw=1.8,
            label=labels[cfg],
        )
    ax1.fill_between(
        wl_nm[order],
        0,
        40.0 * t3_d["led_spectrum"][order],
        color="#a21caf",
        alpha=0.14,
        label=r"InGaN Gaussian spectrum $S(\lambda)$",
    )
    ax1.set(
        xlabel=r"Wavelength $\lambda$ [nm]",
        ylabel=r"Spectral forward extraction $\mathrm{LEE}_{15}(\lambda)$ [%]",
        xlim=(425, 475),
        title=r"(b) Broadband $\mathrm{LEE}_{15}(\lambda)$ across 425–475 nm (Rakić Ag)",
    )
    ax1.legend(loc="upper left", fontsize=7.6)
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "12_polarization_and_spectral_breakdown.png", close=True)

    # ------------------------------------------------------------------
    # Fig 13: Metal skin-depth resolution study & Kottke vs Staircase
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=12.6, height=4.3)
    c_skin_i = (
        t6_d["cells_per_skin_depth_arr"]
        if "cells_per_skin_depth_arr" in t6_d
        else t6_d["cells_per_skin_depth_I"]
    )
    c_skin_e = (
        t6_d["cells_per_skin_depth_E_arr"]
        if "cells_per_skin_depth_E_arr" in t6_d
        else t6_d["cells_per_skin_depth_E"]
    )
    dxs_min = t6_d["dxs_nm"]
    k15 = t6_d["kottke_lee15"]
    s15 = t6_d["staircase_lee15"]
    k90 = t6_d["kottke_lee90"]
    s90 = t6_d["staircase_lee90"]
    ref15 = float(t6_d["ref_lee15"])
    ref90 = float(t6_d["ref_lee90"])

    ax0.plot(
        c_skin_i,
        k15,
        "o-",
        color=colors["dgrin"],
        lw=2.1,
        label=r"Kottke $\mathrm{LEE}_{15}$",
    )
    ax0.plot(
        c_skin_i,
        s15,
        "s--",
        color="#dc2626",
        lw=1.9,
        label=r"Staircase $\mathrm{LEE}_{15}$",
    )
    ax0.plot(
        c_skin_i,
        k90,
        "^-",
        color=colors["air"],
        lw=2.0,
        alpha=0.9,
        label=r"Kottke $\mathrm{LEE}_{90}$",
    )
    ax0.plot(
        c_skin_i,
        s90,
        "v--",
        color=colors["sio2"],
        lw=1.8,
        alpha=0.9,
        label=r"Staircase $\mathrm{LEE}_{90}$",
    )
    ax0.axvline(
        float(c_skin_i[-1]),
        color=colors["dgrin"],
        ls="-.",
        lw=1.5,
        label=(
            f"Accepted grid ($\\Delta x_{{\\min}}={float(dxs_min[-1]):.2f}$ nm: "
            f"{float(c_skin_i[-1]):.2f} c/$\\delta_I$, {float(c_skin_e[-1]):.2f} c/$\\delta_E$)"
        ),
    )
    ax0.scatter(
        [c_skin_i[0], c_skin_i[-1]],
        [k15[0], k15[-1]],
        marker="*",
        s=130,
        color=["#f59e0b", colors["dgrin"]],
        edgecolor="black",
        zorder=7,
    )
    ax0.annotate(
        f"Demo under-resolved\n({dxs_min[0]:.0f} nm, {c_skin_i[0]:.2f} c/$\\delta_I$)",
        (c_skin_i[0], k15[0]),
        textcoords="offset points",
        xytext=(6, 10),
        fontsize=7.2,
        color="#b45309",
    )
    ax0.set(
        xlabel=r"Cells per Ag intensity skin depth $\delta_I = \lambda_0 / (4\pi\kappa) = 15.38$ nm",
        ylabel=r"6-layer dGRIN $\mathrm{LEE}_{15},\;\mathrm{LEE}_{90}$ [%]",
        title=r"(a) Extraction efficiency vs cells per Ag skin depth",
    )
    ax0_top = ax0.twiny()
    ax0_top.set_xlim(ax0.get_xlim()[0] * 2.0, ax0.get_xlim()[1] * 2.0)
    ax0_top.set_xlabel(
        r"Cells per Ag field skin depth $\delta_E = \lambda_0 / (2\pi\kappa) = 30.76$ nm",
        fontsize=8.5,
    )
    ax0.legend(loc="center right", fontsize=7.0)

    rel15 = (
        t6_d["kottke_rel_change_lee15_pct"]
        if "kottke_rel_change_lee15_pct" in t6_d
        else np.r_[0.0, 100.0 * np.abs(np.diff(k15)) / np.maximum(k15[1:], 1e-12)]
    )
    rel90 = (
        t6_d["kottke_rel_change_lee90_pct"]
        if "kottke_rel_change_lee90_pct" in t6_d
        else np.r_[0.0, 100.0 * np.abs(np.diff(k90)) / np.maximum(k90[1:], 1e-12)]
    )
    err_s = np.sqrt(0.5 * ((s15 - ref15) ** 2 + (s90 - ref90) ** 2))
    ax1.semilogy(
        c_skin_i[1:],
        rel15[1:],
        "o-",
        color=colors["dgrin"],
        lw=2.0,
        label=r"Kottke step-to-step $|\Delta\mathrm{LEE}_{15}|/\mathrm{LEE}_{15}$ [%]",
    )
    ax1.semilogy(
        c_skin_i[1:],
        rel90[1:],
        "^-",
        color=colors["air"],
        lw=1.8,
        label=r"Kottke step-to-step $|\Delta\mathrm{LEE}_{90}|/\mathrm{LEE}_{90}$ [%]",
    )
    ax1.semilogy(
        c_skin_i,
        err_s,
        "s--",
        color="#dc2626",
        lw=1.9,
        label=r"Staircase RMS $(\Delta\mathrm{LEE}_{15},\Delta\mathrm{LEE}_{90})$ error [pp]",
    )
    ax1.axhline(
        1.0,
        color=colors["dgrin"],
        ls=":",
        lw=1.4,
        label=r"1% relative convergence threshold",
    )
    ax1.axvline(float(c_skin_i[-1]), color=colors["dgrin"], ls="-.", lw=1.3)
    for ci, dx_n, ce, r15 in zip(c_skin_i[1:], dxs_min[1:], c_skin_e[1:], rel15[1:]):
        ax1.annotate(
            f"{dx_n:.2g} nm\n({r15:.2f}%)",
            (ci, r15),
            textcoords="offset points",
            xytext=(0, -18 if dx_n > 4.0 else 7),
            ha="center",
            fontsize=7.0,
        )
    ax1.set(
        xlabel=r"Cells per Ag intensity skin depth $\delta_I = \lambda_0 / (4\pi\kappa)$",
        ylabel="Relative step change [%] / Staircase RMS error [pp]",
        title=r"(b) Step-to-step convergence (~1% at 6.15 c/$\delta_I$) & staircase error",
    )
    ax1.legend(loc="best", fontsize=7.2)
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "13_kottke_vs_staircase_convergence.png", close=True)

    # ------------------------------------------------------------------
    # Fig 14: Purcell factor F_P and normalization audit across designs
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.8, height=4.0)
    fp_vals = [float(diag_base_map[c]["purcell_factor"]) for c in cfg_order]
    dir_vals = [float(diag_base_map[c]["directivity_2d"]) for c in cfg_order]
    ratio_eq_vals = [
        100.0
        * float(diag_base_map[c]["lee15_eq_pct"])
        / float(diag_base_map[c]["lee90_eq_pct"])
        for c in cfg_order
    ]
    ratio_vals = [float(diag_base_map[c]["collimation_ratio_pct"]) for c in cfg_order]

    ax0.bar(
        x_idx - 0.18,
        fp_vals,
        width=0.34,
        color="#7c3aed",
        label=r"Cavity Purcell factor $F_P = P_{\mathrm{wrk}}/P_{\mathrm{hom}}$",
    )
    ax0.bar(
        x_idx + 0.18,
        dir_vals,
        width=0.34,
        color=colors["air"],
        label=r"2D half-space directivity $D_{2\mathrm{D}}$",
    )
    ax0.axhline(1.0, color="0.4", ls="--", lw=1.0)
    ax0.set_xticks(x_idx)
    ax0.set_xticklabels(
        [
            "Bare\n($M=9$)",
            "Air ($72^\\circ$)\n($M=17$)",
            "SiO$_2$ ($64^\\circ$)\n($M=17$)",
            "dGRIN ($60^\\circ$)\n($M=17$)",
        ]
    )
    ax0.set(
        ylabel="Dimensionless factor",
        title=r"(a) Cavity Purcell factor $F_P$ & 2D directivity $D_{2\mathrm{D}}$ (Rakić Ag)",
    )
    ax0.legend(loc="upper left", fontsize=8.0)

    pub_ratio_2d = [PAPER_2D_REFS[c]["ratio_pct"] for c in cfg_order]
    w_r = 0.25
    ax1.bar(
        x_idx - w_r,
        ratio_eq_vals,
        width=w_r,
        color=colors["dgrin"],
        label=r"Rakić Ag $\eta_{15,\mathrm{eq}} = \mathrm{LEE}_{15,\mathrm{eq}}/\mathrm{LEE}_{90,\mathrm{eq}}$",
    )
    ax1.bar(
        x_idx,
        ratio_vals,
        width=w_r,
        color="#10b981",
        alpha=0.80,
        label=r"Rakić Ag 2D $\eta_{15}$ ($\mathrm{TM}_z\ 2\times$)",
    )
    ax1.bar(
        x_idx + w_r,
        pub_ratio_2d,
        width=w_r,
        color=colors["sio2"],
        alpha=0.85,
        label=r"Luce et al. (2026) 2D $\eta_{15}$ (Fig. 3 / App. A.2–A.3)",
    )
    ax1.axhline(
        100.0 * lambertian_cone_fraction(15.0, dim=2),
        color="black",
        ls="--",
        lw=1.2,
        label=f"2D Lambertian $\\sin(15^\\circ) = {100*lambertian_cone_fraction(15.0, dim=2):.1f}$%",
    )
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(
        [
            "Bare\n($M=9$)",
            "Air ($72^\\circ$)\n($M=17$)",
            "SiO$_2$ ($64^\\circ$)\n($M=17$)",
            "dGRIN ($60^\\circ$)\n($M=17$)",
        ]
    )
    ax1.set(
        ylabel=r"2D collimation fraction $\eta_{15} = \mathrm{LEE}_{15}/\mathrm{LEE}_{90}$ [%]",
        ylim=(0, 70),
        title=r"(b) 2D collimation fraction $\eta_{15}$ vs Luce et al. (2026) 2D",
    )
    ax1.legend(loc="upper left", fontsize=7.2)
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "14_purcell_and_normalization_audit.png", close=True)

    # ------------------------------------------------------------------
    # Fig 15: Dipole ensemble convergence & Rakic vs J&C Ag vs Paper
    # ------------------------------------------------------------------
    if diag_dir.exists():
        from gap_diagnostics import plot_gap_diagnostics

        plot_gap_diagnostics(
            diag_dir=diag_dir, fig_path=fig_dir / "15_gap_diagnostics.png"
        )

    print(f"Rendered 17 figures in {fig_dir}", flush=True)


# %% [markdown]
# ## 6. Results & quantitative comparison with Luce *et al.* (2026)
#
# In **Luce, Alaee, and Abass (2026)** [1], Equation (2) defines $P_0(\lambda)$ as *"the energy injected into the simulation"* (the analytical homogeneous-medium source power $P_{\text{hom},c}$ of each dipole orientation $c \in \{x, y, z\}$), and Equation (12) weights the three orthogonal dipole orientations equally ($w_{\text{pol}} = [1/3, 1/3, 1/3]$) and assigns uniform spatial weights ($w_{\text{pos}} = 1/M$) across the active quantum-well width in 2D. Our like-for-like **headline comparison metric** is therefore the equal-polarization, $P_{\text{hom}}$-normalized extraction efficiency **`lee15_eq_pct`** ($\text{LEE}_{15,\text{eq}} = \frac{1}{3}(\text{LEE}_{15,x} + \text{LEE}_{15,y} + \text{LEE}_{15,z})$) and **`lee90_eq_pct`** ($\text{LEE}_{90,\text{eq}}$).
#
# Table 1 reports the **converged dipole ensemble** on our **full-resolution (`--full`)** graded mesh ($\Delta x_{\text{bg}} = 7.50\;\text{nm}$, $24.49\;\text{cpw}$ at $450\;\text{nm}$ and $23.13\;\text{cpw}$ at $425\;\text{nm}$, $\Delta x_{\min} = 3.00\;\text{nm}$, $\Lambda \in [425, 475]\;\text{nm}$)—using **$M = 17$** uniformly spaced dipoles across $[-240, +240]\;\text{nm}$ for the three horns (with $M = 9$ also shown in brackets) and **$M = 9$** for the bare reference (where `baseline_bare` is already converged to $|M_9 - M_{17}| = 0.07\;\text{pp}$ on `--quick`: $7.59\%$ vs $7.52\%$)—for **both silver optical-constant models side by side with the paper's 2D FDTD benchmarks** (Sec. A.2–A.3 & Fig. 3 of [1]). Coarser $M = 5$ dipole-ensemble values (`task_00..02`, and $M = 11$ over $[-250, +250]\;\text{nm}$ in `task_03`) are also listed in the rightmost column for comparison with the $M$-convergence study in Section 8.2:
#
# | Configuration & geometry ($h$, $\vartheta$, $w_{\text{top}}$) | Converged $M$ (`--full`) | **Rakić (1998) Ag (default) `--full` $\text{LEE}_{15,\text{eq}}$** [$M=9$] (`--quick` $M=17$) & Gap vs Paper | **Johnson & Christy (1972) Ag `--full` $\text{LEE}_{15,\text{eq}}$** [$M=9$] (`--quick` $M=9$) & Gap vs Paper | **Luce *et al.* (2026) 2D FDTD $\text{LEE}_{15}$** (enh. vs bare) | **Rakić Ag / J&C Ag `--full` $\text{LEE}_{90,\text{eq}}$** vs **Paper 2D $\text{LEE}_{90}$** | **Rakić Ag / J&C Ag** per-pol $\text{LEE}_{15,(x,y,z)}$; secondary ($\text{TM}_z\ 2\times$ / wrk) | **2D $\eta_{15,\text{eq}}$** (Rakić / J&C / Paper), FWHM, $D_{2\text{D}}$, $F_P$ & *Coarse $M=5$* |
# |---|---|---|---|---|---|---|---|
# | **Analytical planar GaN/Air** ($h=0$, $\theta_c = 24.09^\circ$) | — | $5.71\%$ (2D Lambertian $\pm 15^\circ$) | — | — | $22.05\%$ (2D escape cone) | — | $\eta_{15} = 25.88\%$, $\text{FWHM} = 120.0^\circ$, $D_{2\text{D}} = 1.57$, $F_P = 1.000$ |
# | **Bare reference (`bare_ref`)** ($h=0\;\mu\text{m}$, $w_{\text{top}}=1.00\;\mu\text{m}$) | **$M = 9$** *(quick $M=17$: $7.52\%$)* | **$8.03\%$** ($7.52\%$) \| **$-6.87\;\text{pp}$** | **$9.89\%$** ($9.36\%$) \| **$-5.01\;\text{pp}$** | **$14.9\%$** ($1.00\times$) | **$37.81\%$** / **$46.21\%$** vs **$55.9\%$** ($-18.09$ / $-9.69\;\text{pp}$) | Rakić: $(11.89, 5.45, 6.75)\%$; $7.71\% / 6.99\%$<br>J&C: $(15.18, 5.98, 8.51)\%$; $9.54\% / 8.41\%$ | $\eta_{15,\text{eq}}: 21.24\% / 21.40\% / 26.65\%$; $\text{FWHM}: 126.7^\circ / 125.3^\circ$; $D_{2\text{D}}: 1.49 / 1.52$; $F_P: 1.104 / 1.135$<br>*Coarse $M=5$ (Rakić): $9.03\%$ (`--full`), $8.48\%$ (`--quick`); $\text{LEE}_{90,\text{eq}} = 42.09\%$* |
# | **Air-filled $\mu\text{Horn}$ (`air_horn_opt`)** ($h=3.0\;\mu\text{m}$, $\vartheta=72^\circ$, $w_{\text{top}}=2.95\;\mu\text{m}$) | **$M = 17$** *(and $M = 9$)* | **$20.13\%$** [$19.91\%$] ($19.04\%$; $2.51\times$) \| **$-7.37\;\text{pp}$** [$-7.59\;\text{pp}$] | **$27.48\%$** [$27.02\%$] ($27.36\%$; $2.78\times$) \| **$-0.02\;\text{pp}$** [$-0.48\;\text{pp}$] | **$27.5\%$** ($1.85\times$) | **$36.42\%$** [$36.17\%$] / **$54.59\%$** [$53.95\%$] vs **$46.7 \pm 1.0\%$** ($-10.28$ / $+7.89\;\text{pp}$) | Rakić: $(23.86, 14.86, 21.66)\%$; $20.51\% / 18.24\%$<br>J&C: $(34.83, 21.07, 26.55)\%$; $27.25\% / 23.45\%$ | $\eta_{15,\text{eq}}: 55.26\% / 50.35\% / 58.89\%$; $\text{FWHM}: 44.8^\circ / 50.8^\circ$; $D_{2\text{D}}: 3.95 / 3.71$; $F_P: 1.125 / 1.162$<br>*Coarse $M=5$ (Rakić): $20.82\%$ (`--full`), $19.91\%$ (`--quick`); $\text{LEE}_{90,\text{eq}} = 38.83\%$* |
# | **$\text{SiO}_2$-filled $\mu\text{Horn}$ (`sio2_horn_opt`)** ($h=3.0\;\mu\text{m}$, $\vartheta=64^\circ$, $w_{\text{top}}=3.93\;\mu\text{m}$) | **$M = 17$** *(and $M = 9$)* | **$25.45\%$** [$25.22\%$] ($24.13\%$; $3.17\times$) \| **$-4.95\;\text{pp}$** [$-5.18\;\text{pp}$] | **$31.11\%$** [$30.73\%$] ($29.27\%$; $3.15\times$) \| **$+0.71\;\text{pp}$** [$+0.33\;\text{pp}$] | **$30.4\%$** ($2.04\times$) | **$53.03\%$** [$52.73\%$] / **$65.66\%$** [$65.18\%$] vs **$65.1 \pm 1.0\%$** ($-12.07$ / $+0.56\;\text{pp}$) | Rakić: $(23.96, 25.25, 27.14)\%$; $25.87\% / 24.08\%$<br>J&C: $(29.72, 32.23, 31.39)\%$; $31.18\% / 27.68\%$ | $\eta_{15,\text{eq}}: 47.99\% / 47.39\% / 46.70\%$; $\text{FWHM}: 50.7^\circ / 50.5^\circ$; $D_{2\text{D}}: 3.65 / 3.69$; $F_P: 1.075 / 1.127$<br>*Coarse $M=5$ (Rakić): $29.75\%$ (`--full`), $28.17\%$ (`--quick`); $\text{LEE}_{90,\text{eq}} = 60.02\%$* |
# | **6-layer dGRIN $\mu\text{Horn}$ (`dgrin_horn_opt`)** ($h=3.0\;\mu\text{m}$, $\vartheta=60^\circ$, $w_{\text{top}}=4.46\;\mu\text{m}$) | **$M = 17$** *(and $M = 9$)* | **$31.37\%$** [$32.53\%$] ($30.19\%$; $3.91\times$) \| **$-6.33\;\text{pp}$** [$-5.17\;\text{pp}$] | **$37.58\%$** [$39.06\%$] ($38.03\%$; $3.80\times$) \| **$-0.12\;\text{pp}$** [$+1.36\;\text{pp}$] | **$37.7\%$** ($2.53\times$) | **$68.93\%$** [$69.28\%$] / **$81.81\%$** [$82.46\%$] vs **$75.3 \pm 1.0\%$** ($-6.37$ / $+6.51\;\text{pp}$) | Rakić: $(30.20, 23.00, 40.92)\%$; $33.76\% / 34.24\%$<br>J&C: $(37.81, 29.77, 45.14)\%$; $39.47\% / 38.45\%$ | $\eta_{15,\text{eq}}: 45.51\% / 45.93\% / 50.07\%$; $\text{FWHM}: 44.7^\circ / 45.3^\circ$; $D_{2\text{D}}: 4.11 / 4.01$; $F_P: 0.986 / 1.026$<br>*Coarse $M=5$ quick: $39.04\%$ (Rakić) / $47.58\%$ (J&C); `task_03` $M=11$ Rakić $32.71\%$* |
#
# ### Key physical observations
#
# 1. **Converged dipole ensemble ($M = 17$, Bare $M = 9$) & Rakić (1998) vs Johnson & Christy (1972) silver against Luce *et al.* (2026) (Fig. 08b & Fig. 15)**:
#    - Under the tutorial's default **Rakić (1998) Drude silver** (`baseline`, $\varepsilon = -5.402 - 0.634\mathrm{i}$ at $450\;\text{nm}$, normal-incidence $R_{\text{GaN/Ag}} = 88.24\%$), the converged `--full` ensemble reproduces the strict hierarchy $\text{LEE}_{15,\text{eq}}(\text{Bare}) < \text{LEE}_{15,\text{eq}}(\text{Air}) < \text{LEE}_{15,\text{eq}}(\text{SiO}_2) < \text{LEE}_{15,\text{eq}}(\text{dGRIN})$ (**$8.03\% < 20.13\% < 25.45\% < 31.37\%$** at $M = (9, 17, 17, 17)$, and **$8.03\% < 19.91\% < 25.22\% < 32.53\%$** at $M = 9$), but sits uniformly **$-4.95$ to $-7.37\;\text{pp}$ below the paper's 2D FDTD $\text{LEE}_{15}$ benchmarks across all four configurations** ($-6.87\;\text{pp}$ on Bare, $-7.37\;\text{pp}$ on Air, $-4.95\;\text{pp}$ on $\text{SiO}_2$, $-6.33\;\text{pp}$ on $\text{dGRIN}$). At a coarser $M = 5$ sampling ($120\;\text{nm}$ spacing), the $\text{SiO}_2$ value ($29.75\%$) sits artificially close to $30.4\%$ because $M = 5$ oversamples lateral standing-wave antinodes of the $600\;\text{nm}$ $\text{GaN}$ core (Fig. 15a; Section 8.2).
#    - Switching to **Johnson & Christy (1972) Drude silver** (`jc_ag`, $\varepsilon = -7.058 - 0.213\mathrm{i}$ at $450\;\text{nm}$, $R_{\text{GaN/Ag}} = 97.04\%$, $3\times$ lower $\operatorname{Im}\varepsilon$) at the converged `--full` ensemble ($M = 17$ for the horns, $M = 9$ for Bare) raises $\text{LEE}_{15,\text{eq}}$ by **$+5.66$ to $+7.36\;\text{pp}$** on the horns and brings **all three micro-horn configurations within $0.71\;\text{pp}$ of the paper's 2D FDTD benchmarks**: **$27.48\%$ vs $27.5\%$ on Air ($-0.02\;\text{pp}$)**, **$31.11\%$ vs $30.4\%$ on $\text{SiO}_2$ ($+0.71\;\text{pp}$)**, and **$37.58\%$ vs $37.7\%$ on 6-layer $\text{dGRIN}$ ($-0.12\;\text{pp}$)** (and on $\text{SiO}_2$ $\text{LEE}_{90,\text{eq}}$, **$65.66\%$ vs $65.1 \pm 1.0\%$, $+0.56\;\text{pp}$**). Meanwhile, the bare reference rises from $8.03\%$ to **$9.89\%$** and still differs from the paper's $14.9\%$ by **$-5.01\;\text{pp}$** (discussed in Section 8.2).
# 2. **Headline $\text{LEE}_{\text{eq}}$ ($w_{\text{pol}}=[1/3,1/3,1/3]$) vs secondary $\text{LEE}_{\text{hom}}$ ($\text{TM}_z\ 2\times$) and $\text{LEE}_{\text{wrk}}$ ($P_{\text{wrk}}$) definitions (Fig. 14a)**: Because a 2D out-of-plane line current $J_z$ ($\text{TM}_z$) radiates twice the homogeneous power of an in-plane line current $J_x$ or $J_y$ ($\text{TE}_{x,y}$) of equal current amplitude ($P_{\text{hom},z} = 2 P_{\text{hom},x} = 2 P_{\text{hom},y}$, Section 2), normalizing each polarization by its own $P_{\text{hom},c}$ with equal weights $w_{\text{pol}} = [1/3, 1/3, 1/3]$ gives our headline metric $\text{LEE}_{\text{eq}}$ (converged `--full` Rakić: **$[8.03, 20.13, 25.45, 31.37]\%$** for $\text{LEE}_{15,\text{eq}}$ and **$[37.81, 36.42, 53.03, 68.93]\%$** for $\text{LEE}_{90,\text{eq}}$; converged `--full` J&C: **$[9.89, 27.48, 31.11, 37.58]\%$** and **$[46.21, 54.59, 65.66, 81.81]\%$**). Two secondary metrics are also stored in `data/diag/*.json` and `data/task_*.json`:
#    - **Unweighted 2D current sum (`lee15_pct`, `lee90_pct`)**: Dividing $\sum_c P_{\text{ext},c}$ by $\sum_c P_{\text{hom},c} = 4 P_{\text{hom},x}$ weights $\text{TM}_z$ by $1/2$ and $\text{TE}_x, \text{TE}_y$ by $1/4$ each ($w_{\text{pol}} = [1/4, 1/4, 1/2]$), giving converged `--full` $\text{LEE}_{15} = [7.71, 20.51, 25.87, 33.76]\%$ ($\text{LEE}_{90} = [37.90, 35.76, 54.87, 73.42]\%$) for Rakić Ag and $[9.54, 27.25, 31.18, 39.47]\%$ ($\text{LEE}_{90} = [45.66, 51.20, 66.80, 85.44]\%$) for J&C Ag.
#    - **Cavity-work normalization (`lee15_wrk_pct`, `lee90_wrk_pct`)**: Dividing $\sum_c P_{\text{ext},c}$ by the actual power $P_{\text{wrk}} = -\sum_c \operatorname{Re}[E_{\text{dip},c}(f) S^*(f)]$ done by the dipoles inside each cavity relates to `lee15_pct` via the cavity Purcell factor $F_P = P_{\text{wrk}} / P_{\text{hom}}$ ($\text{LEE}_{\text{hom}} = F_P \cdot \text{LEE}_{\text{wrk}}$). On the converged `--full` ensemble, $F_P = [1.104, 1.125, 1.075, 0.986]$ for Rakić Ag ($\text{LEE}_{15,\text{wrk}} = [6.99, 18.24, 24.08, 34.24]\%$) and $F_P = [1.135, 1.162, 1.127, 1.026]$ for J&C Ag ($\text{LEE}_{15,\text{wrk}} = [8.41, 23.45, 27.68, 38.45]\%$).
# 3. **Why the air-filled $\mu\text{Horn}$ collimates emission ($\text{FWHM}: 126.7^\circ \to 44.8^\circ$) while plasmonic sidewall loss governs 2D $\text{LEE}_{90}$ (Fig. 08b & Fig. 14b)**: Adding an unfilled air horn ($\vartheta = 72^\circ$) narrows the far-field beam FWHM from $126.7^\circ$ to $44.8^\circ$ ($125.3^\circ \to 50.8^\circ$ with J&C Ag; $D_{2\text{D}}: 1.49 \to 3.95$) and boosts the 2D collimated fraction $\eta_{15,\text{eq}} = \text{LEE}_{15,\text{eq}}/\text{LEE}_{90,\text{eq}}$ from $21.24\%$ to $55.26\%$ ($21.40\% \to 50.35\%$ with J&C Ag, comparable to the paper's 2D $\eta_{15}$ rise from $26.65\%$ to $58.89\%$). Because the $\text{GaN} \to \text{Air}$ index step remains $2.45 \to 1.00$, oblique rays beyond $\theta_c = 24.1^\circ$ remain trapped inside the $\text{GaN}$ core while rays entering the $3\;\mu\text{m}$ tall horn reflect off the $\text{Ag}$ sidewalls; with lossier Rakić Ag, this lowers 2D $\text{LEE}_{90,\text{eq}}$ from $37.81\%$ (Bare) to $36.42\%$ (Air; $-1.39\;\text{pp}$), whereas with higher-reflectance J&C Ag, $\text{LEE}_{90,\text{eq}}$ is $46.21\%$ (Bare) and $54.59\%$ (Air).
# 4. **Refractive-index matching + continuous ray bending in the 6-layer dGRIN horn**: Filling the horn with the 6-layer descending stack ($\text{Nb}_2\text{O}_5 \to \dots \to \text{Air}$) expands the escape cone out of $\text{GaN}$ to $\theta_c(\text{GaN}/\text{Nb}_2\text{O}_5) = 75.3^\circ$, suppresses normal Fresnel reflection to $< 4.8\%$ at $450\;\text{nm}$ (Fig. 02c), and gradually refracts oblique rays toward the horizontal at successive layer interfaces until they are intercepted by the $60^\circ$ $\text{Ag}$ sidewalls and redirected into the $\pm 15^\circ$ forward cone (Fig. 05c and Fig. 06 movie), raising converged `--full` 2D $\text{LEE}_{15,\text{eq}}$ to **$31.37\%$** with Rakić Ag and **$37.58\%$** with J&C Ag (matching the paper's **$37.7\%$** within $-0.12\;\text{pp}$).
# 5. **What the $N_{\text{dip}}=1$ sidewall-angle & horn-height sweeps actually show (Fig. 09, Task 4)**: Unlike the paper's multi-dipole 2D maps in Fig. 3a–f, Task 4 runs a 7-angle ($\vartheta \in \{50^\circ, 56^\circ, 60^\circ, 64^\circ, 68^\circ, 72^\circ, 78^\circ\}$ at $h = 3.0\;\mu\text{m}$) and 5-height ($h \in \{0.6, 1.2, 1.8, 2.4, 3.0\}\;\mu\text{m}$ at paper $\vartheta_{\text{opt}}$) sweep using a **single on-axis central dipole ($x_d = 0$, $N_{\text{dip}} = 1$, default Rakić Ag, reporting `lee15_pct`)**. For **Air** and **$\text{SiO}_2$**, our $N_{\text{dip}}=1$ curves peak at the paper's optimal $(h, \vartheta)$: Air peaks at $\vartheta = 72^\circ$ ($\text{LEE}_{15} = 25.63\%$, vs $18.67\%$ at $60^\circ$ and $17.48\%$ at $78^\circ$) and rises monotonically with height ($9.82\% \to 16.62\% \to 24.67\% \to 25.27\% \to 25.63\%$), while $\text{SiO}_2$ peaks at $\vartheta = 64^\circ$ ($\text{LEE}_{15} = 29.56\%$, vs $26.85\%$ at $60^\circ$ and $22.40\%$ at $72^\circ$) and rises monotonically with height ($10.62\% \to 17.74\% \to 23.35\% \to 27.32\% \to 29.56\%$). For **6-layer $\text{dGRIN}$**, however, **our $N_{\text{dip}}=1$ sweep peaks elsewhere**: at $h = 3.0\;\mu\text{m}$, it peaks at **$\vartheta = 64^\circ$ ($50.16\%$)** rather than $60^\circ$ ($43.07\%$, with $56^\circ$ at $33.90\%$ and $68^\circ$ at $32.22\%$); and at $\vartheta = 60^\circ$, while total extraction $\text{LEE}_{90}$ rises monotonically with height ($88.16\% \to 92.79\% \to 97.22\% \to 98.38\% \to 99.46\%$), on-axis $\text{LEE}_{15}$ **peaks at $h = 1.8\;\mu\text{m}$ ($53.32\%$)** and $h = 2.4\;\mu\text{m}$ ($51.74\%$) before dropping to $43.07\%$ at $h = 3.0\;\mu\text{m}$ (the same non-monotonic $h=1.8\;\mu\text{m} > 3.0\;\mu\text{m}$ peak occurs in `--quick`: $52.54\%$ vs $41.80\%$). This occurs because a single central dipole at $x_d = 0$ forms a coherent on-axis interference pattern with its specular images in the $60^\circ$ sidewalls, whereas the paper's optimum $(h = 3.0\;\mu\text{m}, \vartheta = 60^\circ)$ maximizes the incoherent ensemble average across the full $600\;\text{nm}$ QW width.
# 6. **Incoherent dipole convergence & what the 2D QW-offset study shows (Figs. 10, 11 & 15a)**: Across independent uniform dipole grids $M \in \{1, 5, 9, 17\}$ spanning $[-240, +240]\;\text{nm}$ (`data/diag/baseline_*_M{1,5,9,17}_quick.json`, Fig. 15a), $M = 5$ ($120\;\text{nm}$ step) is not converged because it skips the lateral standing-wave nodes near $x_d = \pm 60, \pm 180\;\text{nm}$, whereas $M = 9$ ($60\;\text{nm}$ step) and $M = 17$ ($30\;\text{nm}$ step) agree within $0.07\text{--}0.23\;\text{pp}$ on Bare, Air, and $\text{SiO}_2$ and within $1.16\text{--}1.18\;\text{pp}$ on $\text{dGRIN}$. Coherently summing electric fields instead of powers produces an unphysical interference spike (Fig. 11b). **Note on Fig. 10 (Task 5)**: Fig. 10 is **not** a reproduction of Fig. 6 of Luce *et al.* (2026)—Fig. 6 of [1] is a **3D cylindrical FDTD** sweep of the **continuous GRIN ($\text{cGRIN}$)** horn vs the 3D bare $\mu\text{LED}$ ($\text{LEE}_{15} \approx 20.4\%\text{--}21.5\%$, $\text{LEE}_{90} \approx 76.2\%\text{--}77.8\%$ across $\Delta y \in [-40, +40]\;\text{nm}$). Instead, our Task 5 / Fig. 10 is a **2D single-on-axis-dipole ($N_{\text{dip}}=1$, Rakić Ag) sensitivity sweep of the 2D 6-layer discrete $\text{dGRIN}$ horn vs the 2D bare pixel** across $\Delta y_{\text{QW}} \in [-40, +40]\;\text{nm}$ in $10\;\text{nm}$ steps around $y_{\text{QW}} = 150\;\text{nm}$. Over this $80\;\text{nm}$ window, the 2D bare on-axis $\text{LEE}_{15}$ varies over $[5.77\%, 43.11\%]$ with a step sensitivity of **$-12.31$ to $+13.89\;\text{pp}$ per $10\;\text{nm}$** (mean $|\Delta\text{LEE}_{15}| = \mathbf{9.11\;\text{pp}}$ per $10\;\text{nm}$), whereas the 2D 6-layer $\text{dGRIN}$ horn maintains $\text{LEE}_{15} \in [37.18\%, 58.71\%]$ with a sensitivity of **$-6.65$ to $+2.42\;\text{pp}$ per $10\;\text{nm}$** (mean $|\Delta\text{LEE}_{15}| = \mathbf{3.48\;\text{pp}}$ per $10\;\text{nm}$, **$2.6\times$ less sensitive** than 2D bare).

# %% [markdown]
# ## 7. Resolution convergence: Kottke vs staircase on sloped metal sidewalls
#
# ### 7.1 Plasmonic Ag skin-depth resolution study & why Task 6 reports $\text{ref\_lee15} = 43.35\%$, $\text{ref\_lee90} = 99.92\%$
#
# Because silver at $\lambda_0 = 450\;\text{nm}$ is a lossy plasmonic metal ($\varepsilon_{\text{Ag}} = -5.402 - 0.634\mathrm{i}$, $\kappa = 2.328$), the evanescent field inside the reflector decays over the electric-field skin depth $\delta_E = \lambda_0 / (2\pi\kappa) = 30.76\;\text{nm}$ and intensity/power skin depth $\delta_I = \lambda_0 / (4\pi\kappa) = 15.38\;\text{nm}$. Task 6 (`conv_kottke_vs_staircase`, Fig. 13) performs a 5-level skin-depth resolution ladder on the 6-layer $\text{dGRIN}$ micro-horn ($h = 3.0\;\mu\text{m}$, $\vartheta = 60^\circ$, $y_{\text{QW}} = 150\;\text{nm}$) comparing `averaging="kottke"` against `averaging="staircase"` across $\Delta x_{\min} \in \{15.0, 7.5, 5.0, 3.75, 2.50\}\;\text{nm}$ ($1.03\text{--}6.15\;\text{cells}/\delta_I$, $2.05\text{--}12.30\;\text{cells}/\delta_E$):
#
# | Level | Mode / Flag | $\Delta x_{\text{bg}}$ [nm] | $\Delta x_{\min}$ [nm] | $\text{cpw}_{\text{GaN}}$ ($450 / 425\;\text{nm}$) | Min feat. [nm] (cells) | Cells / $\text{Ag}$ skin depth ($\delta_I$ / $\delta_E$) | $N_{\text{cells}}$ / $N_{\text{steps}}$ | Kottke $\text{LEE}_{15}$ / $\text{LEE}_{90}$ [%] | Kottke step $|\Delta\text{LEE}_{15}|/\text{LEE}_{15}$ / $|\Delta\text{LEE}_{90}|/\text{LEE}_{90}$ | Staircase $\text{LEE}_{15}$ / $\text{LEE}_{90}$ [%] | Staircase $\Delta\text{LEE}_{90}$ vs Kottke [pp] |
# |---|---|---|---|---|---|---|---|---|---|---|---|
# | 0 | `--full` & `--quick` (demo) | $15.0$ | $15.00$ | $12.24$ / $11.56$ | $150$ ($10$) | $1.03$ / $2.05$ | $92{,}649$ / $1{,}656$ | $38.20\%$ / $88.31\%$ | — / — | $39.70\%$ / $75.50\%$ | $-12.82$ pp |
# | 1 | `--full` ($\delta_I < 4$ cells*) | $7.5$ | $7.50$ | $24.49$ / $23.13$ | $150$ ($20$) | $2.05$ / $4.10$ | $369{,}369$ / $3{,}311$ | $42.45\%$ / $98.00\%$ | $10.00\%$ / $9.89\%$ | $43.14\%$ / $89.46\%$ | $-8.54$ pp |
# | 2 | `--full` ($\delta_I < 4$ cells*) | $7.5$ | $5.00$ | $24.49$ / $23.13$ | $150$ ($23$) | $3.08$ / $6.15$ | $389{,}628$ / $4{,}967$ | $42.32\%$ / $98.29\%$ | $0.29\%$ / $0.29\%$ | $43.67\%$ / $89.60\%$ | $-8.69$ pp |
# | 3 | `--full` ($\ge 4\;\text{c}/\delta_I$) | $7.5$ | $3.75$ | $24.49$ / $23.13$ | $150$ ($25$) | $4.10$ / $8.20$ | $407{,}511$ / $6{,}622$ | $42.85\%$ / $99.14\%$ | $1.22\%$ / $0.85\%$ | $44.37\%$ / $91.31\%$ | $-7.83$ pp |
# | **4** | **`--full` (accepted)** | **$7.5$** | **$2.50$** | **$24.49$ / $23.13$** | **$150$ ($30$)** | **$6.15$ / $12.30$** | **$444{,}066$ / $9{,}933$** | **$43.35\%$ / $99.92\%$** | **$1.16\%$ / $0.79\%$** | $44.96\%$ / $92.69\%$ | **$-7.23$ pp** |
# | *(3q)* | *`--quick` (accepted)* | *$15.0$* | *$3.75$* | *$12.24$ / $11.56$* | *$150$ ($18$)* | *$4.10$ / $8.20$* | *$125{,}145$ / $6{,}622$* | *$41.68\%$ / $96.78\%$* | *$1.11\%$ / $1.38\%$* | *$42.88\%$ / $82.45\%$* | *$-14.33$ pp* |
#
# *\*Resolution audit note*: Every top-level task archive (`task_00..06.json`) satisfies all three resolution rules ($\text{cpw}_{\text{GaN}} = 23.13\;\text{cpw}$ at $425\;\text{nm} \ge 10$; $14\text{--}30\;\text{cells}$ across the smallest feature $\ge 4$; $5.13\text{--}6.15\;\text{cells}/\delta_I \ge 4$). Inside `task_06`'s 5-level skin-depth ladder, only Level 0 ($\Delta x_{\min}=15\;\text{nm}$, $1.03\;\text{cells}/\delta_I$) is marked `demo_underresolved=True`; intermediate ladder steps Level 1 ($7.5\;\text{nm}$) and Level 2 ($5.0\;\text{nm}$) are marked `demo_underresolved=False` because they resolve the field skin depth $\delta_E = 30.76\;\text{nm}$ ($4.10$ and $6.15\;\text{cells}/\delta_E \ge 4$), **which violates the stricter intensity skin-depth threshold ($\ge 4\;\text{cells}/\delta_I$, since they have $2.05$ and $3.08\;\text{cells}/\delta_I < 4$)** unless counted as part of the skin-depth demonstration ladder. Levels 3 ($4.10\;\text{cells}/\delta_I$) and 4 ($6.15\;\text{cells}/\delta_I$), as well as all production runs in Tasks 0–5 ($5.13\;\text{cells}/\delta_I$), strictly satisfy $\ge 4\;\text{cells}/\delta_I$.
#
# **Why `task_06` reports $\text{ref\_lee15} = 43.35\%$ and $\text{ref\_lee90} = 99.92\%$ vs `task_03`'s $35.31\%$ and $74.34\%$**:
#
# 1. **Single on-axis central dipole ($x_d = 0$, $N_{\text{dip}} = 1$) vs 11-dipole lateral mesa average**: To run 10 high-resolution solver passes (5 skin-depth levels $\times$ Kottke & Staircase), `task_06` (like `task_04` and `task_05`) excites **only the single on-axis dipole at the mesa center $x_d = 0$** (`dipole_xs = (0.0,)`) inside the exact same 6-layer $\text{dGRIN}$ horn geometry ($h = 3.0\;\mu\text{m}$, $\vartheta = 60^\circ$, $y_{\text{QW}} = 150\;\text{nm}$), whereas `task_03` averages over 6 lateral positions across the $600\;\text{nm}$ $\text{GaN}$ core ($N_{\text{dip}} = 11$, $x_d \in \{0, \pm 50, \pm 100, \pm 150, \pm 200, \pm 250\}\;\text{nm}$). Indeed, the $N_{\text{dip}} = 1$ ($x_d = 0$) entry of `task_03` (`conv_lee15[0] = 43.06%`, `conv_lee90[0] = 99.44%` at $\Delta x_{\min} = 3.0\;\text{nm}$) and the $(\vartheta=60^\circ, h=3.0\;\mu\text{m}, \Delta y_{\text{QW}}=0)$ point of `task_04`/`task_05` ($43.07\%$ / $99.46\%$) sit right between Level 3 ($42.85\%$ / $99.14\%$ at $\Delta x_{\min} = 3.75\;\text{nm}$) and Level 4 ($43.35\%$ / $99.92\%$ at $\Delta x_{\min} = 2.50\;\text{nm}$) of `task_06`.
# 2. **Why on-axis $\text{LEE}_{90}$ is $\sim 99.92\%$ ($P_{\text{hom}}$ normalization at a lateral Purcell antinode)**: Because `lee90_pct` divides the extracted far-field power $P_{\text{ext}}(x_d=0)$ by the **unstructured homogeneous-medium dipole power** $P_{\text{hom}}$ rather than by the actual power $P_{\text{wrk}}(x_d=0)$ radiated by the central dipole, it measures $P_{\text{ext}}(0)/P_{\text{hom}} = F_P(0) \cdot \text{LEE}_{90,\text{wrk}}(0)$. At the symmetry axis $x_d = 0$ of the $600\;\text{nm}$ $\text{GaN}$ mesa ($w_{\text{GaN}} \approx 3.27\,\lambda_{\text{GaN}}$), the dipole sits on a constructive lateral standing-wave antinode where the local Purcell factor is $F_P(x_d=0) = P_{\text{wrk}}(0)/P_{\text{hom}} \approx 1.33 > 1$ (whereas off-axis positions sample lateral nodes with $F_P < 1$, bringing the 11-dipole mesa average in `task_03` to $F_P = 0.9988$ and $\text{LEE}_{90} = 74.34\%$). Thus $\text{ref\_lee90} = 99.92\%$ does **not** mean $99.92\%$ passive extraction (true passive extraction $P_{\text{ext}}(0)/P_{\text{wrk}}(0) = \text{ref\_lee90}/F_P(0)$ is $\sim 75\%$, consistent with `task_03`'s $74.43\%$); it is a direct consequence of normalizing a Purcell-enhanced central dipole ($F_P(0) > 1$) by $P_{\text{hom}}$ instead of $P_{\text{wrk}}(0)$.
#
# ### 7.2 Convergence verdict & Kottke vs staircase comparison
#
# 1. **Skin-depth ladder convergence ($1.03 \to 6.15\;\text{cells}/\delta_I$)**: When the $\text{Ag}$ skin depth is under-resolved at Level 0 ($\Delta x_{\text{bg}} = \Delta x_{\min} = 15.0\;\text{nm}$, $1.03\;\text{cells}/\delta_I$, $2.05\;\text{cells}/\delta_E$), finite-difference discretization of the exponential field decay inside the metal overestimates ohmic loss at the bottom $\text{Ag}$ mirror and horn throat, depressing Kottke $\text{LEE}_{15}$ by $5.15\;\text{pp}$ ($38.20\%$ vs $43.35\%$) and $\text{LEE}_{90}$ by $11.61\;\text{pp}$ ($88.31\%$ vs $99.92\%$). Stepping $\Delta x_{\text{bg}}$ to $7.5\;\text{nm}$ and refining $\Delta x_{\min}$ through $7.50 \to 5.00 \to 3.75 \to 2.50\;\text{nm}$ ($2.05 \to 3.08 \to 4.10 \to 6.15\;\text{cells}/\delta_I$) converges Kottke $\text{LEE}_{90}$ to within **$0.79\%$** relative change between the last two levels ($99.14\% \to 99.92\%$, $< 1\%$ threshold) and Kottke $\text{LEE}_{15}$ to within **$1.16\%$** ($42.85\% \to 43.35\%$, $+0.50\;\text{pp}$; in the $15\;\text{nm}$ `--quick` ladder $1.03 \to 4.10\;\text{cells}/\delta_I$, the final step change is $1.11\%$ in $\text{LEE}_{15}$ and $1.38\%$ in $\text{LEE}_{90}$). The residual $\sim 1.16\%$ step change in $\text{LEE}_{15}$ arises because `build_graded_horn_grid(..., full_sidewall_skin=False)` refines $\Delta x_{\min}$ inside the mirror and throat collars while the upper sloped $\text{Ag}$ sidewall remains at $\Delta x_{\text{bg}} = 7.5\;\text{nm}$.
# 2. **Persistent staircase absorption error on the sloped $60^\circ$ sidewall**: Even when the mirror and throat skin depths are resolved to $6.15\;\text{cells}/\delta_I$ ($\Delta x_{\min} = 2.50\;\text{nm}$), replacing Kottke subpixel tensor smoothing (`averaging="kottke"`) with voxel staircasing (`averaging="staircase"`) on the sloped $60^\circ$ $\text{Ag}$ horn sidewall turns the specular reflector into right-angle $\Delta x \times \Delta y$ steps. Grazing $\text{TE}$ waves excite localized lightning-rod surface-plasmon resonances on these staircase corners, artificially absorbing **$7.23\;\text{pp}$** of total power at $\Delta x_{\text{bg}} = 7.5\;\text{nm}$ ($\text{LEE}_{90} = 92.69\%$ vs $99.92\%$, and $-14.33\;\text{pp}$ at $\Delta x_{\text{bg}} = 15\;\text{nm}$) while biasing $\text{LEE}_{15}$ upward by $+1.62\;\text{pp}$ ($44.96\%$ vs $43.35\%$). Both local skin-depth refinement **and** Kottke subpixel smoothing are therefore essential for accurate metallic micro-horn simulations.

# %% [markdown]
# ## 8. Limits, failure modes & investigation of the gaps vs Luce *et al.* (2026)
#
# ### 8.1 Bare-reference cavity sensitivity ($5.77\%\text{--}43.11\%$ range) vs horn robustness
#
# As shown in Table 1, at a converged dipole ensemble ($M = 17$ for the horns, $M = 9$ for the bare reference), the tutorial's default Rakić (1998) $\text{Ag}$ model sits **$4.95\text{--}7.37\;\text{pp}$ below** Luce *et al.* (2026) across all four configurations (**Bare $8.03\%$ vs $14.9\%$ ($-6.87\;\text{pp}$)**, **Air $20.13\%$ vs $27.5\%$ ($-7.37\;\text{pp}$)**, **$\text{SiO}_2$ $25.45\%$ vs $30.4\%$ ($-4.95\;\text{pp}$)**, and **6-layer $\text{dGRIN}$ $31.37\%$ vs $37.7\%$ ($-6.33\;\text{pp}$)**), whereas switching to Johnson & Christy (1972) $\text{Ag}$ brings **all three horns within $0.71\;\text{pp}$ of the paper** (**Air $27.48\%$ vs $27.5\%$ ($-0.02\;\text{pp}$)**, **$\text{SiO}_2$ $31.11\%$ vs $30.4\%$ ($+0.71\;\text{pp}$)**, and **$\text{dGRIN}$ $37.58\%$ vs $37.7\%$ ($-0.12\;\text{pp}$)**) while the **Bare reference still differs by $-5.01\;\text{pp}$** ($9.89\%$ vs $14.9\%$). Before examining the diagnostic convergence and silver-model study (Section 8.2), four internal physical effects govern our 2D FIT numbers:
#
# 1. **Vertical Fabry–Pérot cavity interference in the bare $300\;\text{nm}$ $\text{GaN}$ slab & QW-offset sensitivity (`task_05`)**:
#    - In the bare pixel ($h = 0$), the top $\text{GaN}/\text{Air}$ interface has a large index contrast ($2.45 \to 1.00$, normal Fresnel reflectance $R = 17.6\%$) and forms a Fabry–Pérot resonator with the bottom Drude $\text{Ag}$ mirror ($\varepsilon_{\text{Ag}} = -5.402 - 0.634\mathrm{i}$ at $\lambda_0 = 450\;\text{nm}$ for Rakić Ag, internal wavelength $\lambda_{\text{GaN}} = \lambda_0 / n_{\text{GaN}} = 183.67\;\text{nm}$). Because real silver has a finite plasmonic penetration phase ($\arg r_{\text{GaN/Ag}} \approx 115^\circ$ rather than the $180^\circ$ of an ideal PEC), placing the QW at the geometric mid-plane $y_{\text{QW}} = 150\;\text{nm}$ ($0.817\,\lambda_{\text{GaN}}$ above the $\text{Ag}$ interface) sits near an on-axis ($\theta = 0^\circ$) Fabry–Pérot destructive dip for tangential dipoles ($M = 9$ Rakić $\text{TM}_z$ $\text{LEE}_{15} = 6.75\%$, $\text{TE}_y = 5.45\%$, $\text{TE}_x = 11.89\%$, equal-polarization headline $\text{LEE}_{15,\text{eq}} = 8.03\%$, secondary $\text{LEE}_{15} = 7.71\%$, cavity-work $\text{LEE}_{15,\text{wrk}} = 6.99\%$; with J&C Ag at $M = 9$, $\text{LEE}_{15,\text{eq}} = 9.89\%$). As a result, the bare radiation pattern $\rho_{\text{LEE}}(\theta)$ at $y_{\text{QW}} = 150\;\text{nm}$ is broad and off-axis-peaked ($\text{FWHM} = 126.7^\circ$, $\eta_{15,\text{eq}} = 21.24\%$).
#    - **Quantitative sensitivity per $10\;\text{nm}$ of QW offset from Task 5 (`qw_position_sweep`, Fig. 10a), and why moving the QW is not a validation**: Across the 9-point full-resolution 2D on-axis sweep $\Delta y_{\text{QW}} \in \{-40, -30, -20, -10, 0, +10, +20, +30, +40\}\;\text{nm}$ ($y_{\text{QW}} \in [110, 190]\;\text{nm}$ in $10\;\text{nm}$ increments), the 2D bare on-axis $\text{LEE}_{15}$ takes the values **$43.11\%, 37.89\%, 26.83\%, 14.52\%, 6.24\%, 5.77\%, 13.80\%, 27.45\%, 41.34\%$**—spanning a **$7.5\times$ range ($5.77\%\text{--}43.11\%$)** over $\pm 40\;\text{nm}$. Its step-to-step change per $+10\;\text{nm}$ of QW offset is **$-5.22, -11.06, -12.31, -8.28, -0.47, +8.03, +13.65, +13.89\;\text{pp}$ per $10\;\text{nm}$** (mean absolute sensitivity **$9.11\;\text{pp}$ per $10\;\text{nm}$**, max **$13.89\;\text{pp}$ per $10\;\text{nm}$**). Although the paper's $14.9\%$ lies inside this interval (our sweep passes through $14.52\%$ at $\Delta y_{\text{QW}} = -10\;\text{nm}$ and $13.80\%$ at $\Delta y_{\text{QW}} = +20\;\text{nm}$), matching a target value by shifting the emitter position is **not** a validation: a 2D slab cavity with a steep $\sim 9\text{--}14\;\text{pp}/10\;\text{nm}$ interference slope can cross any value between $5.8\%$ and $43.1\%$. We therefore never tune $y_{\text{QW}}$ to match [1]; instead, we keep $y_{\text{QW}} = 150\;\text{nm}$ fixed.
#    - **Why the 2D $\text{dGRIN}$ horn is $2.6\times$ less sensitive to QW position**: When the 6-layer $\text{dGRIN}$ stack is placed atop $\text{GaN}$ ($n_{\text{GaN}} = 2.45 \to n_{\text{Nb}_2\text{O}_5} = 2.37$), the top interface normal reflectance drops from $17.6\%$ to $0.03\%$, eliminating the top Fabry–Pérot mirror. Across the same $10\;\text{nm}$ 2D QW sweep ($y_{\text{QW}} \in [110, 190]\;\text{nm}$), the 6-layer $\text{dGRIN}$ on-axis $\text{LEE}_{15}$ takes the values **$56.35\%, 58.71\%, 55.87\%, 49.72\%, 43.07\%, 38.48\%, 37.18\%, 38.68\%, 41.09\%$** (`dgrin_lee90`: $120.27\%, 128.38\%, 125.50\%, 114.04\%, 99.46\%, 87.85\%, 83.28\%, 86.30\%, 93.80\%$). Its change per $+10\;\text{nm}$ of QW offset is **$+2.36, -2.84, -6.15, -6.65, -4.58, -1.30, +1.49, +2.42\;\text{pp}$ per $10\;\text{nm}$**—a mean absolute sensitivity of **$3.48\;\text{pp}$ per $10\;\text{nm}$** (max $6.65\;\text{pp}/10\;\text{nm}$), more than $2.6\times$ flatter than the 2D bare pixel. Note that at $\Delta y_{\text{QW}} \in [-40, -10]\;\text{nm}$ ($y_{\text{QW}} = 110\text{--}140\;\text{nm}$, constructive mirror antinode), the local Purcell factor exceeds unity ($F_P \approx 1.3\text{--}1.6$), so $P_{\text{hom}}$-normalized `bare_lee90` ($103.00\%$) and `dgrin_lee90` ($114.04\%\text{--}128.38\%$) exceed $100\%$ while true cavity-work extraction efficiency $\text{LEE}_{90,\text{wrk}} = \text{LEE}_{90}/F_P$ remains strictly passive ($< 100\%$).
# 2. **Lateral mesa standing waves & dipole ensemble sampling ($M = 1, 5, 9, 17$)**: Laterally, the $600\;\text{nm}$ wide $\text{GaN}$ core bounded by $\text{SiO}_2/\text{Ag}$ supports lateral standing-wave modes with half-period $\Lambda_x/2 \approx 100\text{--}120\;\text{nm}$. A coarse $M = 5$ grid on $[-240, +240]\;\text{nm}$ ($\Delta x_d = 120\;\text{nm}$, $x_d \in \{0, \pm 120, \pm 240\}\;\text{nm}$) aliases the lateral antinodes for both $\text{SiO}_2$ and $\text{dGRIN}$: on `--quick`, $\text{SiO}_2$ drops by $-4.17\;\text{pp}$ when refined from $M = 5$ ($28.17\%$) to $M = 9$ ($24.00\%$) and then changes by only $+0.13\;\text{pp}$ at $M = 17$ ($24.13\%$), while $\text{dGRIN}$ drops from $39.04\%$ ($M = 5$) to $31.37\%$ ($M = 9$) and $30.19\%$ ($M = 17$). On `--full`, refining from $M = 9$ ($\Delta x_d = 60\;\text{nm}$) to $M = 17$ ($\Delta x_d = 30\;\text{nm}$) changes Air and $\text{SiO}_2$ by less than $0.46\;\text{pp}$ under both silver models.
# 3. **2D slab vs 3D cylindrical solid-angle weighting**: In 2D ($\partial_z = 0$), the angular phase space is uniform ($\mathrm{d}\theta$, Lambertian $\pm 15^\circ$ fraction $\sin 15^\circ = 25.88\%$), whereas in the separate 3D cylindrical FDTD study of Luce *et al.* (2026) Section 3.2 (Table 1 of [1], where 3D $\text{LEE}_{15} = [2.3\%, 11.6\%, 11.7\%, 19.9\%]$ and 3D $\text{LEE}_{90} = [31.8\%, 27.7\%, 54.1\%, 75.6\%]$ for Bare, Air, $\text{SiO}_2$, and $\text{dGRIN}$), the solid-angle Jacobian $\sin\theta\,\mathrm{d}\theta\,\mathrm{d}\varphi$ weights oblique angles much more heavily (Lambertian $\pm 15^\circ$ fraction $\sin^2 15^\circ = 6.70\%$) and radial dipoles near the perimeter carry area weight $2\pi r_i\,\Delta r$. For this reason, 3D Table 1 numbers must never be mixed with 2D ones.
# 4. **Finite top-monitor aperture truncation**: A planar `Near2FarMonitor2D` line above the horn aperture ($y_{\text{mon}} = y_{\text{top}} + 160\;\text{nm}$) captures all upward radiation within $|\theta| \lesssim 82^\circ$ but misses grazing rays ($\theta \to \pm 90^\circ$) that enter the lateral CPMLs before crossing $y_{\text{mon}}$.
#
# ### 8.2 Dipole-ensemble convergence & silver optical-constant diagnostics (`gap_diagnostics.py`, `data/diag/*.json`)
#
# Systematic $M$-convergence runs in `data/diag/*.json` demonstrate why dense lateral dipole sampling is required across the $600\;\text{nm}$ $\text{GaN}$ core: a coarse $M = 5$ dipole ensemble ($x_d \in \{0, \pm 120, \pm 240\}\;\text{nm}$, $\Delta x_d = 120\;\text{nm}$ for Bare, Air, and $\text{SiO}_2$; $M = 11$ on $[-250, +250]\;\text{nm}$ in `task_03` for $\text{dGRIN}$) is not yet converged because its $\Delta x_d = 120\;\text{nm}$ spacing aliases lateral standing-wave antinodes, producing an apparent $\text{SiO}_2$ value of $29.75\%$ vs the paper's $30.4\%$ ($-0.65\;\text{pp}$) under Rakić (1998) silver. At converged dipole sampling ($M = 9$ and $M = 17$), `SiO2` settles from $29.75\%$ ($M = 5$) to $25.22\%$ ($M = 9$) and $25.45\%$ ($M = 17$), and Rakić silver sits $4.95\text{--}7.37\;\text{pp}$ below the paper across all four geometries.
#
# All diagnostic runs are archived in `tutorials/07_light_sources/07-03_microled_horn/data/diag/*.json` and summarized via `gap_diagnostics.py --summary` (plotted in **Fig. 15**, `figures/15_gap_diagnostics.png`). The paper states that its 2D Ansys Lumerical FDTD simulations weight dipole positions uniformly across the active layer, weight the three orthogonal dipole polarizations equally as $[0.33, 0.33, 0.33]$, and average over a Gaussian emission spectrum on $425\text{--}475\;\text{nm}$ with $\text{FWHM} = 25\;\text{nm}$ [1]. However, Luce *et al.* (2026) do not name their silver optical dataset in the main text, their Data Availability statement notes that specific models and explicit details were not disclosed due to competitive concerns, and the simulation setup appendix (Supplementary Information Section A.10) is not available to us.
#
# The table below reports every diagnostic series from the committed `data/diag/*.json` files (no uncommitted runs):
#
# | Diagnostic series (`data/diag/*.json`) | Mesh & silver model | Ensemble $M$ on $[-240, +240]\;\text{nm}$ | `[Bare, Air, SiO2, dGRIN]` $\text{LEE}_{15,\text{eq}}$ [%] | `[Bare, Air, SiO2, dGRIN]` $\text{LEE}_{90,\text{eq}}$ [%] | Gap vs paper 2D $\text{LEE}_{15}$ (`[14.9, 27.5, 30.4, 37.7]%`) [pp] |
# |---|---|---|---|---|---|
# | **1a. Quick $M$-convergence** (`baseline_*_M1_quick.json`) | `--quick` ($\Delta x_{\text{bg}}=15\;\text{nm}$), Rakić (1998) Ag | $M = 1$ ($x_d = 0$, central antinode) | `[7.30, 26.24, 29.47, 37.09]%` | `[46.59, 42.02, 71.92, 86.28]%` | `[-7.60, -1.26, -0.93, -0.61]` pp |
# | **1b. Quick $M$-convergence** (`baseline_*_M5_quick.json`) | `--quick` ($\Delta x_{\text{bg}}=15\;\text{nm}$), Rakić (1998) Ag | $M = 5$ ($\Delta x_d = 120\;\text{nm}$, **unconverged**) | `[8.48, 19.91, 28.17, 39.04]%` | `[40.60, 36.79, 57.47, 71.17]%` | `[-6.42, -7.59, -2.23, +1.34]` pp (`SiO2` $+4.04\;\text{pp}$ above $M = 17$) |
# | **1c. Quick $M$-convergence** (`baseline_*_M9_quick.json`) | `--quick` ($\Delta x_{\text{bg}}=15\;\text{nm}$), Rakić (1998) Ag | $M = 9$ ($\Delta x_d = 60\;\text{nm}$) | `[7.59, 18.90, 24.00, 31.37]%` | `[36.73, 34.37, 50.68, 66.41]%` | `[-7.31, -8.60, -6.40, -6.33]` pp |
# | **1d. Quick $M$-convergence** (`baseline_*_M17_quick.json`) | `--quick` ($\Delta x_{\text{bg}}=15\;\text{nm}$), Rakić (1998) Ag | $M = 17$ ($\Delta x_d = 30\;\text{nm}$) | `[7.52, 19.04, 24.13, 30.19]%` | `[36.62, 34.52, 50.82, 65.87]%` | `[-7.38, -8.46, -6.27, -7.51]` pp ($|M_9 - M_{17}| \le 0.14\;\text{pp}$ for Bare/Air/$\text{SiO}_2$) |
# | **2. Quick J&C (1972) Ag** (`jc_ag_*_M9_quick.json`, `dgrin_M5`) | `--quick` ($\Delta x_{\text{bg}}=15\;\text{nm}$), J&C (1972) Ag | $M = 9$ (`dGRIN` $M = 5$: `47.58%`) | `[9.36, 27.36, 29.27, 38.03]%` | `[44.84, 52.23, 62.33, 79.74]%` | `[-5.54, -0.14, -1.13, +0.33]` pp |
# | **3a. Full Rakić (1998) Ag ($M = 9$)** (`baseline_*_M9_full.json`) | `--full` ($\Delta x_{\text{bg}}=7.5\;\text{nm}$), Rakić (1998) Ag | $M = 9$ ($\Delta x_d = 60\;\text{nm}$) | `[8.03, 19.91, 25.22, 32.53]%` | `[37.81, 36.17, 52.73, 69.28]%` | `[-6.87, -7.59, -5.18, -5.17]` pp |
# | **3b. Full Rakić (1998) Ag ($M = 17$)** (`baseline_*_M17_full.json`) | `--full` ($\Delta x_{\text{bg}}=7.5\;\text{nm}$), Rakić (1998) Ag | $M = 17$ ($\Delta x_d = 30\;\text{nm}$; Bare at $M = 9$) | **`[8.03, 20.13, 25.45, 31.37]%`** | **`[37.81, 36.42, 53.03, 68.93]%`** | **`[-6.87, -7.37, -4.95, -6.33]` pp** ($|M_9 - M_{17}| = 0.22, 0.23\;\text{pp}$ on Air, $\text{SiO}_2$) |
# | **4a. Full J&C (1972) Ag ($M = 9$)** (`jc_ag_*_M9_full.json`) | `--full` ($\Delta x_{\text{bg}}=7.5\;\text{nm}$), J&C (1972) Ag | $M = 9$ ($\Delta x_d = 60\;\text{nm}$) | `[9.89, 27.02, 30.73, 39.06]%` | `[46.21, 53.95, 65.18, 82.46]%` | `[-5.01, -0.48, +0.33, +1.36]` pp |
# | **4b. Full J&C (1972) Ag ($M = 17$)** (`jc_ag_*_M17_full.json`) | `--full` ($\Delta x_{\text{bg}}=7.5\;\text{nm}$), J&C (1972) Ag | $M = 17$ ($\Delta x_d = 30\;\text{nm}$; Bare at $M = 9$) | **`[9.89, 27.48, 31.11, 37.58]%`** | **`[46.21, 54.59, 65.66, 81.81]%`** | **`[-5.01, -0.02, +0.71, -0.12]` pp** ($|M_9 - M_{17}| = 0.46, 0.38\;\text{pp}$ on Air, $\text{SiO}_2$) |
# | **5. Polarization re-weighting** (`*_M17_full.json`, Bare `M9`) | `--full` Rakić / J&C Ag archive re-weights | $w_{\text{pol}}=[1/2,0,1/2]$ (`xz`) / $\text{TM}_z\ 2\times$ (`hom`) | Rakić `xz`: `[9.32, 22.76, 25.55, 35.56]%`; `hom`: `[7.71, 20.51, 25.87, 33.76]%`<br>J&C `xz`: `[11.85, 30.69, 30.56, 41.48]%`; `hom`: `[9.54, 27.25, 31.18, 39.47]%` | Rakić `hom`: `[35.29, 37.24, 53.50, 72.18]%`<br>J&C `hom`: `[43.13, 55.14, 65.81, 84.99]%` | Confirms paper's $[1/3, 1/3, 1/3]$ weight matches J&C horns within $0.71\;\text{pp}$ |
#
# **Key conclusions from the committed diagnostics**:
#
# 1. **With Johnson & Christy (1972) silver and a converged dipole ensemble ($M = 17$), all three horns agree with Luce *et al.* (2026) within $0.71\;\text{pp}$**:
#    - At `--full` $M = 17$ under `jc_ag` ($\varepsilon_{\text{Ag}} = -7.06 - 0.21\mathrm{i}$ at $450\;\text{nm}$, $\text{GaN}/\text{Ag}$ normal reflectance $96.9\%$ vs Rakić's $\varepsilon_{\text{Ag}} = -5.40 - 0.63\mathrm{i}$ and $88.2\%$), we obtain **Air $= 27.48\%$** (paper $27.5\%$, **$-0.02\;\text{pp}$**), **$\text{SiO}_2 = 31.11\%$** (paper $30.4\%$, **$+0.71\;\text{pp}$**), and **$\text{dGRIN} = 37.58\%$** (paper $37.7\%$, **$-0.12\;\text{pp}$**).
#    - Switching from Rakić (1998) to Johnson & Christy (1972) silver at converged $M$ raises $\text{LEE}_{15,\text{eq}}$ by **$+7.35\;\text{pp}$ on Air**, **$+5.66\;\text{pp}$ on $\text{SiO}_2$**, and **$+6.21\;\text{pp}$ on $\text{dGRIN}$**, and raises $\text{LEE}_{90,\text{eq}}$ by **$+18.17\;\text{pp}$**, **$+12.63\;\text{pp}$**, and **$+12.88\;\text{pp}$** (`[54.59, 65.66, 81.81]%` vs paper `[46.7, 65.1, 75.3]%`).
# 2. **Epistemological status of the silver model (hypothesis, not confirmed fact)**:
#    - Because Luce *et al.* (2026) do not disclose their silver optical dataset in the main text or Data Availability statement and Supplementary Information Section A.10 is unavailable, **using Johnson & Christy (1972) silver is a hypothesis consistent with the data, not a confirmed fact**.
#    - Specifically, Johnson & Christy (1972) silver (a standard built-in Lumerical FDTD material entry) was selected for testing after observing the $-7.37\;\text{pp}$ Air horn deficit under Rakić (1998) silver; however, once fixed from published J&C (1972) optical constants with zero free parameters, **both $\text{SiO}_2$ ($31.11\%$ vs $30.4\%$) and $\text{dGRIN}$ ($37.58\%$ vs $37.7\%$) then agree with the paper without any further changes**. We retain Rakić (1998) as the tutorial's default material in `_helpers.py` and report both models side by side in Table 1 and Fig. 15.
# 3. **Why $M = 5$ is not converged, and why $M = 9 \to 17$ is converged**:
#    - On `--quick`, $\text{SiO}_2$ $\text{LEE}_{15,\text{eq}}$ shifts by **$4.04\;\text{pp}$** between $M = 5$ ($28.17\%$) and $M = 17$ ($24.13\%$), and $\text{dGRIN}$ shifts by **$8.85\;\text{pp}$** ($39.04\% \to 30.19\%$), because $M = 5$ ($\Delta x_d = 120\;\text{nm}$) samples near lateral cavity antinodes while missing the nodes near $x_d \approx \pm 60, \pm 180\;\text{nm}$.
#    - Once those nodes are included at $M = 9$ ($\Delta x_d = 60\;\text{nm}$) and $M = 17$ ($\Delta x_d = 30\;\text{nm}$), $|M_9 - M_{17}|$ on `--full` is **$0.22\;\text{pp}$ (Rakić) / $0.46\;\text{pp}$ (J&C) for Air** and **$0.23\;\text{pp}$ (Rakić) / $0.38\;\text{pp}$ (J&C) for $\text{SiO}_2$**—all well below $0.5\;\text{pp}$—and $1.16\;\text{pp}$ (Rakić) / $1.48\;\text{pp}$ (J&C) for the 6-layer $\text{dGRIN}$ horn.
# 4. **The bare reference still differs by $\sim 5\;\text{pp}$ ($9.89\%$ vs $14.9\%$, $-5.01\;\text{pp}$ with J&C Ag; $8.03\%$ vs $14.9\%$, $-6.87\;\text{pp}$ with Rakić Ag)**:
#    - Unlike the three horns, the bare reference gap remains $\sim 5\;\text{pp}$ even with J&C silver at $M = 9$ ($9.89\%$ vs $14.9\%$ in $\text{LEE}_{15,\text{eq}}$; $46.21\%$ vs $55.9\%$ in $\text{LEE}_{90,\text{eq}}$). This residual difference is **unexplained** by our committed diagnostics. Plausible candidates (listed without claiming which is responsible) include:
#      - *Vertical QW placement or active-region thickness averaging*: as shown in Section 8.1 (`task_05`), the bare $300\;\text{nm}$ $\text{GaN}$ slab forms a high-$Q$ Fabry–Pérot cavity whose on-axis $\text{LEE}_{15}$ varies at **$9.11\;\text{pp}$ per $10\;\text{nm}$** ($5.77\%\text{--}43.11\%$), so a $\lesssim 5\;\text{nm}$ difference in $y_{\text{QW}}$ or averaging over a finite multi-QW stack shifts the bare pixel by $\sim 5\;\text{pp}$ while barely perturbing the index-matched horns;
#      - *Dispersive multi-coefficient fits for $\text{Ag}$ and $\text{GaN}$ across $425\text{--}475\;\text{nm}$*: a slight difference in the $\text{GaN}/\text{Ag}$ reflection phase $\arg r_{\text{GaN/Ag}}(\lambda)$ shifts the bare slab's destructive interference dip at $y_{\text{QW}} = 150\;\text{nm}$;
#      - *Near-to-far monitor geometry and total-power normalization*: a 3-sided open box monitor vs our horizontal line monitor at $y_{\text{top}} + 160\;\text{nm}$, or whether the denominator of $\text{LEE}$ uses homogeneous-medium power $P_{\text{hom}}$ vs actual dipole work $P_{\text{wrk}}$ in the bare slab cavity;
#      - *Lateral dipole span in the bare reference*: whether the paper's bare pixel distributes dipoles across the same $[-240, +240]\;\text{nm}$ span or across the full $600\;\text{nm}$ $\text{GaN}$ width.
#
# ![Figure 15: Gap diagnostics — (a) dipole ensemble M-convergence and (b) Rakić vs J&C silver vs paper](figures/15_gap_diagnostics.png)

# %% [markdown]
# ## 9. Exercises
#
# 1. **Continuous GRIN (cGRIN) approximation**: Increase the number of discrete layers in `DGRIN_STACK` from 6 to 12 by linearly interpolating the refractive index from $n_1 = 2.45$ ($\text{GaN}$) down to $n_{12} = 1.05$ over $h = 3.0\;\mu\text{m}$ at $\vartheta = 60^\circ$. How much closer does $\text{LEE}_{15}$ and $\text{LEE}_{90}$ come to the ideal continuous-GRIN limit of Luce *et al.* (2026) Table 1?
# 2. **Parabolic / compound parabolic concentrator (CPC) horn profile**: Modify `build_microled_horn_structures()` in `_helpers.py` so the sidewall angle $\vartheta(y)$ varies from $\vartheta = 56^\circ$ in the bottom high-index layers ($\text{Nb}_2\text{O}_5, \text{Si}_3\text{N}_4$) to $\vartheta = 68^\circ$ in the top low-index layers ($\text{MgF}_2, \text{Air}$), as suggested in Section 3 of Luce *et al.* (2026). Does a two-slope horn beat the constant-$60^\circ$ $\text{LEE}_{15}$?
# 3. **Red $\text{AlGaInP}$ $\mu\text{LED}$ ($\lambda_0 = 630\;\text{nm}$, $n_{\text{core}} = 3.25$)**: Re-scale the wavelength band to $605\text{--}655\;\text{nm}$ and test whether the same 6-layer oxide/nitride stack (capped at $n_{\text{Nb}_2\text{O}_5} = 2.30$) still suppresses TIR when a high-index $\text{TiO}_2$ ($n = 2.58$) or $\text{GaP}$ ($n = 3.31$) bottom matching layer is inserted.

# %% [markdown]
# ## 10. References
#
# 1. A. Luce, R. Alaee, and A. Abass, "Ultra-directional and high-efficiency $\mu$LEDs via gradient index filled micro-horn collimators," *Sci. Rep.* **16**, 7391 (2026), DOI: [10.1038/s41598-026-39920-7](https://doi.org/10.1038/s41598-026-39920-7).
# 2. H.-Y. Ryu, "Evaluation of light extraction efficiency of GaN-based nanorod light-emitting diodes by averaging over source positions and polarizations," *Crystals* **8**(1), 27 (2018), DOI: [10.3390/cryst8010027](https://doi.org/10.3390/cryst8010027).
# 3. C. Kottke, A. Farjadpour, and S. G. Johnson, "Perturbation theory for anisotropic dielectric interfaces, and application to subpixel smoothing of discretized numerical methods," *Phys. Rev. E* **77**, 036611 (2008), DOI: [10.1103/PhysRevE.77.036611](https://doi.org/10.1103/PhysRevE.77.036611).
# 4. A. D. Rakić, A. B. Djurišić, J. M. Elazar, and M. L. Majewski, "Optical properties of metallic films for vertical-cavity optoelectronic devices," *Appl. Opt.* **37**(22), 5271–5283 (1998), DOI: [10.1364/AO.37.005271](https://doi.org/10.1364/AO.37.005271).
# 5. P. B. Johnson and R. W. Christy, "Optical constants of the noble metals," *Phys. Rev. B* **6**(12), 4370–4379 (1972), DOI: [10.1103/PhysRevB.6.4370](https://doi.org/10.1103/PhysRevB.6.4370).
# 6. E. F. Schubert, *Light-Emitting Diodes*, 2nd ed. (Cambridge University Press, 2006).


# %%
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="04. MicroLED with a Gradient-Index Micro-Horn Collimator (2D FIT)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run coarse 2D study locally (~10 cpw in GaN) and render all 15 figures",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run converged 2D study (~20 cpw in GaN) across all tasks and render all 15 figures",
    )
    ap.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single study task I (0..6); uses --full unless --quick is also passed",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR,
        help="Output directory for task .npz files (default: data/)",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Read data/task_*.npz and data/diag/*.json and generate all 15 figures in figures/",
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


def _run_task_entry(args_tuple):
    task, out_dir = args_tuple
    return str(run_single_task(task, out_dir))


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
    t_start = time.perf_counter()
    n_workers = min(4, len(tasks))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(_run_task_entry, [(t, args.out) for t in tasks]))
    generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR)
    total_s = time.perf_counter() - t_start
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(
        f"Completed all {len(tasks)} tasks + 14 figures in {total_s:.1f} s "
        f"(main process peak RSS {peak_mb:.0f} MB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
