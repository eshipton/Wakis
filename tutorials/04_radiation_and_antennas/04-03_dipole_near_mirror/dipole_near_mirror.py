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
# # Tutorial 4.3 · Dipole near a mirror: image theory, Drexhage effect and Purcell factor
#
# **Abstract.** Spontaneous emission is not an immutable property of an isolated atom or quantum
# well: it is a joint property of the emitter and the electromagnetic **local density of states
# (LDOS)** of its surroundings. When an emitter approaches a reflective boundary, the field
# reflected back onto the source point interferes constructively or destructively with the
# primary dipole oscillation, modulating both the total radiative decay rate (the **Drexhage
# effect** and **Purcell factor** $F_P = \Gamma / \Gamma_0 = P / P_0$) and the far-field angular
# radiation pattern. In this tutorial we use Wakis in a **2D finite-integration-technique (FIT)**
# cross-section (`GridFIT3D`, `SolverFIT3D`) to build a complete, quantitatively validated
# hierarchy of planar emitter benchmarks:
#
# 1. **Part A — Line dipole above a perfect electric conductor (PEC) mirror**: We extract total
#    radiated power via a closed/half-space Poynting `FluxMonitor` (`Near2FarMonitor` with
#    `ground_plane="-z"`) and local source work $-\operatorname{Re}[E_c(\mathbf{r}_0, f) S^*(f)]$
#    across normalized heights $kh \in [0.25, 9.5]$ for all three dipole orientations
#    ($\text{TE}_\parallel: J_y$, $\text{TM}_\parallel: J_x$, $\text{TM}_\perp: J_z$), verifying
#    the exact 2D Bessel image-dipole formulas to $< 1\%$ and reproducing the far-field lobe
#    splitting with `Near2FarMonitor`.
# 2. **Part B — Line dipole above a dispersive Silver (`Drude`) mirror**: Using Wakis's auxiliary
#    differential equation (ADE) `DispersiveMaterial` + `Drude` solver at $\lambda_0 = 600\;\text{nm}$
#    (resolving the $24.41\;\text{nm}$ electric-field skin depth), we benchmark $F_{\text{tot}}(kh)$
#    against the exact **2D Chance–Prock–Silbey (CPS) Sommerfeld integral**, separating upper-hemisphere
#    radiation ($F_{\text{rad}}$) from non-radiative surface-plasmon-polariton ($\text{SPP}$) and
#    near-field Ohmic quenching ($F_{\text{nonrad}}$).
# 3. **Part C — Planar microcavity Purcell factor $F_P(d)$ (2D vs. 3D DOS)**: Placing the line
#    dipole at the midplane of a planar cavity of optical thickness $d = n L / \lambda_0 \in [0.20, 2.35]$
#    ($n = 2.4$), we contrast the **3D step-function density of states** of **Abram *et al.* (IEEE
#    JQE 34, 71, 1998, Eq. 7)** against the **2D inverse-square-root Van Hove singularities**
#    $(d^2 - d_m^2)^{-1/2}$ at the waveguide cutoffs $d_m = m - 1/2$, and show how band-limited
#    spectral windowing eliminates zero-group-velocity ($v_g = 0$) cutoff ringing.
# 4. **Part D — Distributed-Bragg-Reflector (DBR) micro-RCLED & incoherent quantum-well emission**:
#    Using the verified blue $\text{InGaN}$ micro-RCLED DBR stack of **Lee *et al.* (Discover Nano
#    19, 75, 2024)** and the Fabry–Pérot extraction theory of **Benisty, De Neve & Weisbuch (IEEE
#    JQE 34, 1612, 1998)**, we model cavity beaming, rabbit-ear detuning lobes, and **incoherent
#    spontaneous emission** by averaging **powers** (never fields) over emitter positions,
#    polarizations, and random-phase Monte Carlo realizations.
#
# ---
#
# ## Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. Relate the classical time-averaged power $P(\mathbf{r}_0, \omega)$ radiated by a line current
#    source `CurrentSource` in 2D FIT to the **projected local density of states (LDOS)** and
#    Purcell factor $F_P(\mathbf{r}_0, \omega) = P(\mathbf{r}_0, \omega) / P_0(\omega)$.
# 2. Configure a 2D $(x, z)$ periodic cross-section in Wakis and extract consistent total power
#    and angular far-field patterns from `FluxMonitor`, `DFTMonitor`, and `Near2FarMonitor(..., ground_plane="-z")`.
# 3. Model a lossy plasmonic mirror with `DispersiveMaterial` and `Drude`, audit the mesh
#    resolution in **cells per skin depth** ($\delta_E / \Delta z$), and interpret the CPS
#    Sommerfeld $k_\parallel$-spectrum across the light line ($u = 1$) and SPP pole ($u_{\text{SPP}}$).
# 4. Explain why a planar microcavity exhibits **step discontinuities** in 3D (Abram *et al.* 1998)
#    but **$1/\sqrt{d - d_m}$ Van Hove divergences** in 2D, and avoid time-domain DFT ringing at
#    waveguide cutoffs.
# 5. Simulate an extended **incoherent** quantum-well emitter layer by averaging power over
#    positions/orientations or random-phase ensembles, and quantify the $1/\sqrt{N_{\text{real}}}$
#    Monte Carlo convergence rate and **Kottke subpixel smoothing** vs. staircasing on off-grid DBR layers.
#
# ## Contents
#
# 1. [Physics of modified spontaneous emission: LDOS, Drexhage & CPS theory](#1.-Physics-of-modified-spontaneous-emission:-LDOS,-Drexhage-&-CPS-theory)
# 2. [2D vs. 3D planar microcavity density of states & RCLED extraction](#2.-2D-vs.-3D-planar-microcavity-density-of-states-&-RCLED-extraction)
# 3. [Incoherent spontaneous emission & subpixel DBR discretization](#3.-Incoherent-spontaneous-emission-&-subpixel-DBR-discretization)
# 4. [Simulation building blocks & study task table](#4.-Simulation-building-blocks-&-study-task-table)
# 5. [Generating the setup schematics and 14-figure diagnostic suite](#5.-Generating-the-setup-schematics-and-14-figure-diagnostic-suite)
# 6. [Quantitative results & benchmark comparison tables](#6.-Quantitative-results-&-benchmark-comparison-tables)
# 7. [Resolution convergence: Kottke vs. staircase & skin-depth audit](#7.-Resolution-convergence:-Kottke-vs.-staircase-&-skin-depth-audit)
# 8. [Limits & failure modes](#8.-Limits-&-failure-modes)
# 9. [Exercises](#9.-Exercises)
# 10. [References](#10.-References)
#
# ---
#
# ## Simulation setup schematics & parameters
#
# The four 2D FIT configurations studied in this tutorial are rendered directly from the live `SolverFIT3D` objects via `wakis.schematic.save_setup_figures`:
#
# ### Part A setup · Line dipole above a PEC ground mirror (`00a`, `00b`)
#
# In Part A, a line dipole (`CurrentSource`, $\lambda_0 = 1.0\;\mu\text{m}$) at height $h$ above a PEC ground plane at $z = 0$ radiates into an 8-cell CPML upper half-space enclosed by a half-space `Near2FarMonitor` box (`ground_plane="-z"`).
#
# ![Part A setup geometry: line dipole above PEC mirror](figures/00a_setup_geometry.png)
# ![Part A mesh and broadband excitation](figures/00b_setup_mesh_and_excitation.png)
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 2.40 µm × 120 nm × 2.08 µm (z ∈ [0 m, 2.08 µm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 60 × 3 × 52 (9360 cells) |
# | Discretization | Δx & resolution | Δ_min = 40 nm, 25.0 cells/λ (in n_max = 1.00), 6.0 cells across feature (240 nm) |
# | Boundary | -x face | CPML (8 cells, 320 nm) |
# | Boundary | +x face | CPML (8 cells, 320 nm) |
# | Boundary | -y face | Periodic |
# | Boundary | +y face | Periodic |
# | Boundary | -z face | PEC |
# | Boundary | +z face | CPML (8 cells, 320 nm) |
# | Material | Background | Vacuum, ε = 1.00 (isotropic) |
# | Source | 1 · CurrentSource | pos: x = 0 m, z = 240 nm, pol: J_y (E_y), pulse: f₀ = 299.8 THz, Δf = 89.94 THz |
# | Monitor | 2 · Near2FarMonitor | Near2FarMonitor box (ground_plane='-z'), pos: x ∈ [-0.28, 0.28] µm, z ∈ [0, 0.40] µm, freqs: f = 299.8 THz |
# | Variation | Polarizations | param=component ∈ {'y' (TE_∥), 'x' (TM_∥), 'z' (TM_⊥)}, note=Exact 2D Bessel Purcell factors 1 - J₀(2kh), 1 - J₀(2kh) + J₂(2kh), 1 + J₀(2kh) + J₂(2kh) |
# | Variation | Broadband & monochromatic kh sweep | param=kh ∈ [0.25, 9.5], h/λ₀ ∈ {0.24, 0.48, 0.76}, note=Local source work DFTMonitor at (0, y_mid, h) and Near2FarMonitor(ground_plane='-z') angular patterns |
#
# ### Part B setup · Line dipole above a dispersive Drude silver mirror (`00a_silver`, `00b_silver`)
#
# In Part B, a line dipole ($\lambda_0 = 600\;\text{nm}$) sits above a $150\;\text{nm}$ dispersive `Drude` silver slab ($\varepsilon_\infty = 3.7$, $\hbar\omega_p = 9.01\;\text{eV}$, $\hbar\gamma = 0.048\;\text{eV}$) on a locally graded mesh resolving the electric-field skin depth $\delta_E = 24.41\;\text{nm}$ with $\Delta z_{\text{Ag}} = 2.50\;\text{nm}$ ($9.8\;\text{cells}/\delta_E$).
#
# ![Part B setup geometry: line dipole above Drude silver mirror](figures/00a_setup_geometry_silver.png)
# ![Part B skin-depth graded mesh and excitation](figures/00b_setup_mesh_and_excitation_silver.png)
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 960 nm × 15 nm × 610 nm (z ∈ [-150 nm, 460 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 56 × 3 × 145 (24360 cells) |
# | Discretization | Δx & resolution | Δ_min = 2.50 nm, 124.8 cells/λ (in n_max = 1.92), 9.8 cells across feature (24.4 nm) |
# | Boundary | -x face | CPML (8 cells, 191 nm) |
# | Boundary | +x face | CPML (8 cells, 191 nm) |
# | Boundary | -y face | Periodic |
# | Boundary | +y face | Periodic |
# | Boundary | -z face | PEC |
# | Boundary | +z face | CPML (8 cells, 195.6 nm) |
# | Material | Background | Vacuum, ε = 1.00 (isotropic) |
# | Material | Region 1: Drude Ag mirror (δE = 24.4 nm) | ε_r = 3.70, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-480 nm, 480 nm] (d = 960 nm) |
# | Source | 1 · CurrentSource | pos: x = 0 m, z = 47.5 nm, pol: J_y (E_y), pulse: f₀ = 499.7 THz, Δf = 209.9 THz |
# | Monitor | 2 · FluxMonitor | Dipole FluxMonitor box (P_tot), pos: x ∈ [-10.0, 10.0] nm, z ∈ [42.5, 52.5] nm, freqs: f = 499.7 THz |
# | Monitor | 3 · FluxMonitor | FluxMonitor (z = 2.50 nm), pos: z = 2.50 nm, freqs: f = 499.7 THz |
# | Variation | Height & polarization sweep | param=h ∈ [20, 220] nm (kh ∈ [0.21, 2.30]), pol ∈ {'y', 'x', 'z'}, note=Benchmarked against 2D Chance–Prock–Silbey (CPS) Sommerfeld integral (F_tot, F_rad, F_nonrad, u_SPP = 1.0338) |
# | Variation | Skin-depth resolution study | param=Δz_Ag ∈ [2.0, 24.0] nm (1.02 to 12.21 cells / δE), note=Electric-field skin depth δE = 24.41 nm, power skin depth δI = 12.21 nm at λ₀ = 600 nm |
#
# ### Part C setup · Planar PEC microcavity (`00a_cavity`, `00b_cavity`)
#
# In Part C, the line dipole is placed at the midplane $z = L_{\text{cav}} / 2 = 500\;\text{nm}$ of a planar dielectric microcavity ($n = 2.4$, $\varepsilon_r = 5.76$) bounded vertically by PEC mirrors at $z = 0$ and $z = 1.0\;\mu\text{m}$ and laterally by 8-cell CPML absorbers.
#
# ![Part C setup geometry: planar PEC microcavity](figures/00a_setup_geometry_cavity.png)
# ![Part C mesh and band-limited cavity excitation](figures/00b_setup_mesh_and_excitation_cavity.png)
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 3 µm × 62.5 nm × 1 µm (z ∈ [0 m, 1 µm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 144 × 3 × 48 (20736 cells) |
# | Discretization | Δx & resolution | Δ_min = 20.8 nm, 50.5 cells/λ (in n_max = 2.40), 24.0 cells across feature (500 nm) |
# | Boundary | -x face | CPML (8 cells, 166.7 nm) |
# | Boundary | +x face | CPML (8 cells, 166.7 nm) |
# | Boundary | -y face | Periodic |
# | Boundary | +y face | Periodic |
# | Boundary | -z face | PEC |
# | Boundary | +z face | PEC |
# | Material | Background | Vacuum, ε = 1.00 (isotropic) |
# | Material | Region 1: Dielectric | ε_r = 5.76, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.50 µm, 1.50 µm] (d = 3 µm) |
# | Source | 1 · CurrentSource | pos: x = 0 m, z = 500 nm, pol: J_y (E_y), pulse: f₀ = 118.7 THz, Δf = 23.73 THz |
# | Monitor | 2 · DFTMonitor | DFTMonitor (x = 0 m), pos: x = 0 m, freqs: 15 freqs in [74.9, 169.9] THz |
# | Variation | Band-limited optical thickness sweep | param=d = nL/λ₀ ∈ [0.22, 2.32] across 3 spectral windows, note=Avoids v_g = 0 Van Hove waveguide cutoffs (d = 0.5, 1.5 for J_y, J_x; d = 1.0, 2.0 for J_z) |
#
# ### Part D setup · Blue InGaN DBR micro-RCLED (`00a_rcled`, `00b_rcled`)
#
# In Part D, a quantum-well line emitter ($\lambda_0 = 450\;\text{nm}$) inside a $1\lambda$ $\text{GaN}$ spacer ($n = 2.39$) is sandwiched between a bottom $\text{NP-GaN/GaN}$ DBR ($n = 1.70/2.39$) and a top $\text{SiO}_2/\text{Ta}_2\text{O}_5$ DBR ($n = 1.46/2.15$) on an interface-snapped `graded_axis` mesh with a top `Near2FarMonitor` extraction aperture.
#
# ![Part D setup geometry: blue InGaN DBR micro-RCLED](figures/00a_setup_geometry_rcled.png)
# ![Part D interface-snapped graded mesh and excitation](figures/00b_setup_mesh_and_excitation_rcled.png)
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 2.26 µm × 28.2 nm × 1.20 µm (z ∈ [-547 nm, 657 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 240 × 3 × 109 (78480 cells) |
# | Discretization | Δx & resolution | Δ_min = 8.69 nm, 21.7 cells/λ (in n_max = 2.39), 5.4 cells across feature (47.1 nm) |
# | Boundary | -x face | CPML (8 cells, 75.2 nm) |
# | Boundary | +x face | CPML (8 cells, 75.2 nm) |
# | Boundary | -y face | Periodic |
# | Boundary | +y face | Periodic |
# | Boundary | -z face | CPML (8 cells, 75.3 nm) |
# | Boundary | +z face | CPML (8 cells, 168.6 nm) |
# | Material | Background | Vacuum, ε = 1.00 (isotropic) |
# | Material | Region 1: Bottom NP-GaN/GaN DBR | ε_r = 2.89, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.13 µm, 1.13 µm] (d = 2.26 µm) |
# | Material | Region 2: 1λ GaN cavity | ε_r = 5.71, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.13 µm, 1.13 µm] (d = 2.26 µm) |
# | Material | Region 3: Top Ta₂O₅/SiO₂ DBR | ε_r = 4.62, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.13 µm, 1.13 µm] (d = 2.26 µm) |
# | Source | 1 · CurrentSource | pos: x = 0 m, z = 94.1 nm, pol: J_y (E_y), pulse: f₀ = 666.2 THz, Δf = 186.5 THz |
# | Monitor | 2 · FluxMonitor | Dipole FluxMonitor box, pos: x ∈ [-18.8, 18.8] nm, z ∈ [56.5, 131.8] nm, freqs: f = 666.2 THz |
# | Monitor | 3 · FluxMonitor | FluxMonitor (z = 477 nm), pos: z = 477 nm, freqs: f = 666.2 THz |
# | Variation | Cavity detuning & angular beaming | param=d_opt ∈ {1.00 (tuned), 1.055 (red-detuned rabbit ears)}, note=Bottom NP-GaN/GaN DBR (n = 1.70/2.39) + GaN cavity + top SiO₂/Ta₂O₅ DBR (n = 1.46/2.15) |
# | Variation | Incoherent quantum-well array | param=N_dip = 7 emitters across w_QW = 0.72 µm, N_real ∈ [1, 128], note=Deterministic power sum vs random-phase Monte Carlo O(N_real^-1/2) convergence |

# %% [markdown]
# ## 1. Physics of modified spontaneous emission: LDOS, Drexhage & CPS theory
#
# ### 1.1 Poynting work identity and the 2D Purcell factor
#
# By Fermi's golden rule and Poynting's theorem for a harmonic current density
# $\mathbf{J}(\mathbf{r}, t) = \operatorname{Re}[\mathbf{J}(\mathbf{r}, \omega) e^{i\omega t}]$,
# the normalized spontaneous emission rate $\Gamma / \Gamma_0$ of a quantum emitter equals the
# classical ratio of time-averaged power $P$ dissipated by an equivalent point (3D) or line (2D)
# dipole to its homogeneous-medium radiation power $P_0$:
#
# $$F_P(\mathbf{r}_0, \omega) \equiv \frac{\Gamma(\mathbf{r}_0, \omega)}{\Gamma_0(\omega)}
# = \frac{P(\mathbf{r}_0, \omega)}{P_0(\omega)}
# = \frac{-\frac{1}{2}\operatorname{Re}\!\left[\mathbf{p}^* \cdot \mathbf{E}(\mathbf{r}_0, \omega)\right]}
#        {-\frac{1}{2}\operatorname{Re}\!\left[\mathbf{p}^* \cdot \mathbf{E}_0(\mathbf{r}_0, \omega)\right]}
# = 1 + \frac{\operatorname{Im}\!\left[\hat{\mathbf{u}}_p \cdot \bar{\bar{\mathbf{G}}}_{\text{scat}}(\mathbf{r}_0, \mathbf{r}_0; \omega)\,\hat{\mathbf{u}}_p\right]}
#            {\operatorname{Im}\!\left[\hat{\mathbf{u}}_p \cdot \bar{\bar{\mathbf{G}}}_0(\mathbf{r}_0, \mathbf{r}_0; \omega)\,\hat{\mathbf{u}}_p\right]}.$$
#
# In a 2D $(x, z)$ cross-section invariant along $y$ (modeled in Wakis with `Ny = 3` periodic cells
# of total thickness $L_y = N_y \Delta y$), a line current $\mathbf{J}(\mathbf{r}, \omega) = I_0 S(\omega)\,\delta(x - x_0)\delta(z - z_0)\hat{\mathbf{e}}_c$
# implemented via `CurrentSource(..., size=(0.0, L_y, 0.0))` radiates cylindrical Hankel waves
# $H_0^{(1)}(k\rho)$. On the normalization scale of `FluxMonitor.get_flux() / L_y` (which integrates
# $\operatorname{Re}[\mathbf{E}(f)\times\mathbf{H}^*(f)\cdot\mathbf{n}]$), the exact 2D homogeneous
# radiated power per unit $y$-length is:
#
# $$P_{0,y}'(f) = \frac{1}{4} k \eta |S(f)|^2 \quad (\text{TE}_\parallel,\; J_y),
# \qquad
# P_{0,x}'(f) = P_{0,z}'(f) = \frac{1}{8} k \eta |S(f)|^2 \quad (\text{TM}_\parallel,\; J_x \;\text{and}\; \text{TM}_\perp,\; J_z),$$
#
# where $k = 2\pi n f / c_0$ and $\eta = \eta_0 / n$. The factor of $1/2$ between TM ($J_x, J_z$)
# and TE ($J_y$) arises because an in-plane dipole radiates with a $\cos^2\theta$ or $\sin^2\theta$
# dipolar pattern in the $(x, z)$ plane ($\frac{1}{2\pi}\int_0^{2\pi}\cos^2\theta\,\mathrm{d}\theta = \frac{1}{2}$),
# whereas an out-of-plane line current $J_y$ radiates isotropically in the $(x, z)$ plane.
#
# ### 1.2 Exact 2D Drexhage formulas above a PEC mirror
#
# Placing the line dipole at height $z = h$ above a perfect electric conductor (PEC) mirror at
# $z = 0$ replaces the lower half-space by an image line dipole at $z = -h$. Under PEC reflection
# ($E_x, E_y \to 0$ at $z = 0$), tangential image currents reverse sign ($J_x^{\text{img}} = -J_x$,
# $J_y^{\text{img}} = -J_y$) while normal image currents keep their sign ($J_z^{\text{img}} = +J_z$).
# Evaluating the cylindrical Green's function $G_{\text{2D}}(\rho) = \frac{i}{4}H_0^{(1)}(k\rho)$ at
# separation $\rho = 2h$ yields the exact 2D Bessel Purcell factors:
#
# $$\begin{aligned}
# F_{P,\text{2D}}^{\text{TE},\parallel}(kh) &= 1 - J_0(2kh) \xrightarrow{kh \to 0} 0 \qquad (J_y\;\text{out-of-plane}), \\[4pt]
# F_{P,\text{2D}}^{\text{TM},\parallel}(kh) &= 1 - 2\left[J_0(2kh) - \frac{J_1(2kh)}{2kh}\right] = 1 - J_0(2kh) + J_2(2kh) \xrightarrow{kh \to 0} 0 \qquad (J_x\;\text{in-plane}), \\[4pt]
# F_{P,\text{2D}}^{\text{TM},\perp}(kh) &= 1 + 2\,\frac{J_1(2kh)}{2kh} = 1 + J_0(2kh) + J_2(2kh) \xrightarrow{kh \to 0} 2 \qquad (J_z\;\text{vertical}).
# \end{aligned}$$
#
# Notice the dimensional contrast with **Drexhage's 3D experiment** [1]: in 2D, cylindrical
# reflections decay slowly as $J_0(2kh) \sim (\pi kh)^{-1/2}\cos(2kh - \pi/4)$, whereas in 3D spherical
# reflections decay as $\sin(2kh)/(2kh) = \mathcal{O}((kh)^{-1})$.
#
# The corresponding 2D angular power distributions $\frac{1}{P_0'}\frac{\mathrm{d}P'}{\mathrm{d}\theta}$
# in the upper half-space $\theta \in [-\pi/2, \pi/2]$ (measured from the $+z$ normal) are:
#
# $$I_y(\theta) = \frac{2}{\pi}\sin^2(kh\cos\theta), \qquad
# I_x(\theta) = \frac{4}{\pi}\cos^2\theta\sin^2(kh\cos\theta), \qquad
# I_z(\theta) = \frac{4}{\pi}\sin^2\theta\cos^2(kh\cos\theta),$$
#
# whose integrals over $\theta \in [-\pi/2, \pi/2]$ equal $F_{P,\text{2D}}^\alpha(kh)$ identically.
#
# ### 1.3 Lossy Silver (`Drude`) mirror & 2D Chance–Prock–Silbey (CPS) Sommerfeld theory
#
# When the ideal PEC mirror is replaced by a dispersive silver mirror with Drude permittivity
# ($\varepsilon_\infty = 3.7$, $\hbar\omega_p = 9.01\;\text{eV}$, $\hbar\gamma = 0.048\;\text{eV}$;
# at $\lambda_0 = 600\;\text{nm}$, $\varepsilon_m = -15.301 + 0.441i$ in the physics $e^{-i\omega t}$
# convention and electric-field skin depth $\delta_E = [k_0 \operatorname{Im}\sqrt{\varepsilon_m}]^{-1} = 24.41\;\text{nm}$),
# the **2D Chance–Prock–Silbey (CPS)** plane-wave expansion [2] expresses the total normalized
# decay rate as a Sommerfeld integral over the normalized in-plane wavevector $u = k_x / k_1$:
#
# $$\begin{aligned}
# F_{\text{tot}}^{\text{TE},\parallel}(kh) &= \frac{2}{\pi}\operatorname{Re}\!\int_0^\infty \frac{1 + r_s(u)e^{2i k_{z1} h}}{\sqrt{1 - u^2}}\,\mathrm{d}u, \\[4pt]
# F_{\text{tot}}^{\text{TM},\parallel}(kh) &= \frac{4}{\pi}\operatorname{Re}\!\int_0^\infty \sqrt{1 - u^2}\left[1 - r_p(u)e^{2i k_{z1} h}\right]\mathrm{d}u, \\[4pt]
# F_{\text{tot}}^{\text{TM},\perp}(kh) &= \frac{4}{\pi}\operatorname{Re}\!\int_0^\infty \frac{u^2}{\sqrt{1 - u^2}}\left[1 + r_p(u)e^{2i k_{z1} h}\right]\mathrm{d}u,
# \end{aligned}$$
#
# where $k_{z1} = k_1\sqrt{1 - u^2}$ ($\operatorname{Im} k_{z1} \ge 0$), $r_s(u)$ and $r_p(u)$ are
# the Fresnel reflection coefficients of the metal interface. This integral splits cleanly at the
# **light line $u = 1$**:
#
# * **Radiative cone ($0 \le u < 1$)**: Propagating waves escaping to the upper hemisphere yield
#   $F_{\text{rad}}(kh)$ (weighted by $\frac{1}{2}[1 + |r|^2 \pm 2\operatorname{Re}(r e^{2i k_{z1} h})]$).
# * **Evanescent region ($u > 1$)**: Non-radiative near-field transfer $F_{\text{nonrad}}(kh) = F_{\text{tot}} - F_{\text{rad}}$
#   proportional to $\operatorname{Im}[r_{s,p}(u)] e^{-2\sqrt{u^2 - 1}\,kh}$. For TM polarizations
#   ($J_x, J_z$), the denominator $\varepsilon_m k_{z1} + \varepsilon_1 k_{z2} = 0$ of $r_p(u)$ creates
#   a giant **Surface Plasmon Polariton (SPP)** pole at
#   $u_{\text{SPP}} = \operatorname{Re}\sqrt{\frac{\varepsilon_m \varepsilon_1}{\varepsilon_m + \varepsilon_1}} = 1.0338$,
#   launching guided plasmonic waves along the interface and causing strong near-field quenching as $kh \to 0$.

# %% [markdown]
# ## 2. 2D vs. 3D planar microcavity density of states & RCLED extraction
#
# ### 2.1 Planar PEC microcavity: 3D step DOS (Abram *et al.* 1998) vs. 2D Van Hove singularities
#
# Consider a dipole at the midplane ($z_0 = L / 2$) of a planar microcavity of thickness $L$ and
# refractive index $n = 2.4$ between two ideal mirrors, parameterized by the normalized optical
# thickness $d = n L / \lambda_0 = k L / (2\pi)$. Quantization of the vertical wavenumber
# $k_{z,m} = m\pi / L$ restricts emission to discrete guided mode branches with cutoffs at
# $d_m = m / 2$. At the midplane $z_0 = L/2$, tangential electric fields ($\propto \sin(m\pi z_0 / L)$)
# couple only to **odd** modes $m = 2p - 1$ (cutoffs at half-integers $d = 0.5, 1.5, 2.5, \dots$),
# whereas the normal electric field $E_z \propto \cos(m\pi z_0 / L)$ couples to the cutoff-free
# **TEM mode $m = 0$** plus **even** modes $m = 2p$ (cutoffs at integers $d = 1.0, 2.0, \dots$).
#
# In **3D**, integrating over the 2D in-plane wavevector $\mathbf{k}_\parallel$ ($\mathrm{d}^2 k_\parallel = 2\pi k_\parallel\,\mathrm{d}k_\parallel = -2\pi k_z\,\mathrm{d}k_z$)
# yields a **constant 2D density of states per branch**, producing the famous **piecewise-smooth
# step discontinuities** of **Abram, Robert & Kuszelewicz, IEEE JQE 34, 71 (1998), Eq. (7)** [3]:
#
# $$F_{P,\text{3D}}^{\parallel}(d) = \frac{3}{4d}\sum_{m=1}^{\lfloor d + 1/2 \rfloor}\left[1 + \left(\frac{m - 1/2}{d}\right)^{\!2}\right],
# \qquad
# F_{P,\text{3D}}^{\perp}(d) = \frac{1}{2d} + \frac{3}{2d}\sum_{m=1}^{\lfloor d \rfloor}\left[1 - \left(\frac{m}{d}\right)^{\!2}\right].$$
#
# In **2D** (one transverse wavenumber $k_x = \pm\sqrt{k^2 - k_{z,m}^2}$), the 1D waveguide Jacobian
# $\mathrm{d}k_x / \mathrm{d}\omega = \omega / (c^2 k_x) = 1 / (c \sqrt{1 - (m / 2d)^2})$ diverges
# as an **inverse-square-root Van Hove singularity** at every mode cutoff! Poisson summation of
# the 2D image-dipole series yields the exact 2D planar-cavity Purcell factors:
#
# $$\begin{aligned}
# F_{P,\text{2D}}^{\text{TE},\parallel}(d) &= \frac{2}{\pi d}\sum_{m=1}^{\lfloor d + 1/2 \rfloor}\frac{1}{\sqrt{1 - \left(\frac{m - 1/2}{d}\right)^{\!2}}}, \\[4pt]
# F_{P,\text{2D}}^{\text{TM},\parallel}(d) &= \frac{4}{\pi d}\sum_{m=1}^{\lfloor d + 1/2 \rfloor}\frac{\left(\frac{m - 1/2}{d}\right)^{\!2}}{\sqrt{1 - \left(\frac{m - 1/2}{d}\right)^{\!2}}}, \\[4pt]
# F_{P,\text{2D}}^{\text{TM},\perp}(d) &= \frac{2}{\pi d} + \frac{4}{\pi d}\sum_{m=1}^{\lfloor d \rfloor}\sqrt{1 - \left(\frac{m}{d}\right)^{\!2}}.
# \end{aligned}$$
#
# **Key physical & numerical insight**: For $d < 0.5$ (sub-half-wavelength cavity), both
# $F_{P}^{\text{TE},\parallel}(d) = 0$ and $F_{P}^{\text{TM},\parallel}(d) = 0$—all parallel-dipole
# spontaneous emission is **completely inhibited** because every TE and odd-TM mode is below cutoff!
# Meanwhile, $F_{P,\text{2D}}^{\text{TM},\perp}(d) = \frac{2}{\pi d}$ diverges as $d \to 0$ via the
# cutoff-free $\text{TEM}_0$ parallel-plate mode. Numerically, at $d = 0.5$ or $1.5$, the newly
# opened mode has zero horizontal group velocity ($v_g = c k_x / k = 0$), so energy injected at the
# exact cutoff frequency bounces vertically between the mirrors forever ($\propto t^{-1/2}$ in 2D)
# and never reaches the lateral CPMLs. Driving the cavity with band-limited pulses whose spectra stay
# strictly inside each smooth inter-cutoff band eliminates this $\operatorname{sinc}$ spectral ringing.
#
# ### 2.2 DBR micro-RCLED extraction & angular beaming (Benisty *et al.* 1998; Lee *et al.* 2024)
#
# Ideal PEC mirrors trap all emitted photons inside guided modes. A practical **Resonant-Cavity
# LED (RCLED)** replaces the mirrors with finite-reflectivity **Distributed Bragg Reflectors (DBRs)**
# so that the fundamental cavity resonance lies inside the air escape cone ($\theta_{\text{cav}} < \arcsin(1/n_{\text{cav}})$).
# Following the open-access blue $\text{InGaN}$ micro-RCLED ($\lambda_0 = 450\;\text{nm}$) of
# **T.-Y. Lee *et al.*, Discover Nano 19, 75 (2024), Fig. 1a,b** [5] and the Fabry–Pérot cavity
# theory of **Benisty, De Neve & Weisbuch, IEEE JQE 34, 1612 (1998)** [4]:
#
# * **Active GaN cavity**: refractive index $n_{\text{GaN}} = 2.39$ at $\lambda_0 = 450\;\text{nm}$
#   (critical angle $\theta_c = \arcsin(1/2.39) = 24.73^\circ$), optical thickness $d = n_{\text{GaN}} L_{\text{cav}} / \lambda_0$.
# * **Bottom high-reflectivity DBR**: Nanoporous-GaN / GaN ($\text{NP-GaN/GaN}$) quarter-wave pairs
#   with $n_{\text{NP-GaN}} = 1.70$ ($d_L = \lambda_0 / (4 n_{\text{NP-GaN}}) = 66.18\;\text{nm}$)
#   and $n_{\text{GaN}} = 2.39$ ($d_H = \lambda_0 / (4 n_{\text{GaN}}) = 47.07\;\text{nm}$),
#   verified from Lee *et al.* (2024) Fig. 1b.
# * **Top output-coupling DBR**: $\text{Ta}_2\text{O}_5 / \text{SiO}_2$ quarter-wave pairs with
#   $n_{\text{Ta}_2\text{O}_5} = 2.15$ ($d_H = 52.33\;\text{nm}$) and $n_{\text{SiO}_2} = 1.46$
#   ($d_L = 77.05\;\text{nm}$) facing air ($n_{\text{ext}} = 1.0$).
#
# By **Benisty *et al.* (1998)**, the external angular intensity $I_{\text{ext}}(\theta_{\text{ext}})$
# transmitted through the top DBR is governed by the product of the source-to-bottom-mirror
# interference factor and the Fabry–Pérot Airy resonance denominator:
#
# $$I_{\text{ext}}(\theta_{\text{ext}}) \propto T_{\text{top}}(\theta_{\text{cav}})
# \frac{\left|1 + r_{\text{bot}}(\theta_{\text{cav}}) e^{2 i k_{z,\text{cav}} z_{\text{dip}}}\right|^2}
#      {\left|1 - r_{\text{top}}(\theta_{\text{cav}}) r_{\text{bot}}(\theta_{\text{cav}}) e^{2 i k_{z,\text{cav}} L_{\text{cav}}}\right|^2}
# \frac{n_{\text{ext}}\cos\theta_{\text{ext}}}{n_{\text{cav}}\cos\theta_{\text{cav}}}.$$
#
# When $d = 1.0$ (tuned cavity), the Airy peak sits at normal incidence ($\theta_{\text{ext}} = 0^\circ$),
# producing a narrow forward-directed beam. When $d > 1.0$ (positively red-detuned cavity, e.g. $d = 1.055$ in our
# 2D Wakis RCLED benchmark or $d = 1.12$ at larger detuning), the resonance condition
# $k_{z,\text{cav}} L_{\text{cav}} = 2\pi d \cos\theta_{\text{cav}} = 2\pi$ shifts to an oblique angle
# $\theta_{\text{cav}} = \arccos(1/d)$, splitting the far-field pattern into symmetric **"rabbit-ear" sidelobes**
# ($\theta_{\text{ext}} \approx \pm 26^\circ\text{--}29^\circ$ at $d = 1.055$, or $\pm 40^\circ$ at $d = 1.12$).
#
# > **Provenance note on literature targets**: The $\text{NP-GaN/GaN}$ indices ($1.70 / 2.39$),
# > $\text{Ta}_2\text{O}_5/\text{SiO}_2$ top DBR architecture, $\lambda_0 = 450\;\text{nm}$ operating
# > wavelength, and $5\times 5\;\mu\text{m}^2$ mesa dimensions are verified directly from open-access
# > **Lee, Huang, Hung, Chen *et al.* (2024), "InGaN blue resonant cavity micro-LED with RGY quantum dot layer for broad gamut, efficient displays," Fig. 1a,b & text** (`doi:10.1186/s11671-024-04018-4`). The 1998 IEEE JQE
# > companions by **Abram, Robert & Kuszelewicz (1998), "Spontaneous emission control in semiconductor microcavities with metallic or Bragg mirrors"** (`doi:10.1109/3.655009`, Eq. 7, $n = 2.4$) and **Benisty, De Neve & Weisbuch (1998), "Impact of planar microcavity effects on light extraction—Part I: Basic concepts and analytical trends"**
# > (`doi:10.1109/3.709578`, Airy extraction formula) are paywalled on IEEE Xplore; their analytical
# > equations are self-contained and verified here from first-principles Green's function / Poisson-summation
# > derivations.

# %% [markdown]
# ## 3. Incoherent spontaneous emission & subpixel DBR discretization
#
# ### 3.1 Why spontaneous emission requires averaging POWERS, never fields
#
# In an LED quantum well of lateral width $w_{\text{QW}}$, electron–hole pairs recombine
# spontaneously at different positions $\mathbf{r}_m$, with uncorrelated random phases $\phi_m \in [0, 2\pi)$
# and unpolarized dipole orientations:
#
# $$\langle c_m^* c_n \rangle = \delta_{mn}\,|c_m|^2.$$
#
# If we drive $N_{\text{dip}}$ dipoles simultaneously in a **single coherent simulation** with equal
# phases ($\phi_m = 0$), they act as a phased antenna array: their electric fields add coherently
# ($\mathbf{E}_{\text{coh}} = \sum_{m=1}^{N_{\text{dip}}} \mathbf{E}_m$), creating artificial
# multi-slit interference fringes in the far field! Instead, the physically correct **incoherent
# emission** is the ensemble average of the **power**:
#
# $$\left\langle |\mathbf{E}(\mathbf{r}, \omega)|^2 \right\rangle
# = \left\langle \left|\sum_{m=1}^{N_{\text{dip}}} e^{i\phi_m}\mathbf{E}_m(\mathbf{r}, \omega)\right|^2 \right\rangle
# = \sum_{m=1}^{N_{\text{dip}}} |\mathbf{E}_m(\mathbf{r}, \omega)|^2
# + \sum_{m \ne n} \underbrace{\langle e^{i(\phi_m - \phi_n)} \rangle}_{=\,0} \mathbf{E}_m \cdot \mathbf{E}_n^*
# = \sum_{m=1}^{N_{\text{dip}}} |\mathbf{E}_m(\mathbf{r}, \omega)|^2.$$
#
# Likewise, the incoherent far-field angular intensity is the sum of the individual dipole intensities
# $I_{\text{incoh}}(\theta) = \frac{1}{N_{\text{dip}}}\sum_{m=1}^{N_{\text{dip}}} I_m(\theta)$
# averaged over TE ($J_y$) and TM ($J_x$) quantum-well polarizations. If one instead uses a
# **random-phase Monte Carlo** method (running $N_{\text{real}}$ realizations in which all dipoles
# fire simultaneously with independent random phases $\phi_m^{(r)} \sim \mathcal{U}(0, 2\pi)$),
# the cross-terms decay slowly as $\mathcal{O}(N_{\text{real}}^{-1/2})$. We demonstrate both the
# exact deterministic individual-dipole power sum ( exploiting mirror symmetry $x_m \leftrightarrow -x_m$
# so $N_{\text{dip}} = 7$ emitters require only $4$ runs per polarization) and the $1/\sqrt{N_{\text{real}}}$
# Monte Carlo convergence curve.
#
# ### 3.2 Subpixel Kottke smoothing vs. staircasing for quarter-wave DBR stacks
#
# Quarter-wave DBR layers have thicknesses $d_H = \lambda_0 / (4 n_H)$ and $d_L = \lambda_0 / (4 n_L)$
# whose ratio $d_L / d_H = n_H / n_L$ is generally irrational relative to a uniform FDTD/FIT grid
# step $\Delta z$. Under **staircase** assignment (`averaging="staircase"`), each interface snaps
# to the nearest grid cell boundary, shifting the optical thickness of individual Bragg layers by up
# to $\pm \frac{1}{2} n \Delta z$ and detuning the cavity resonance. Using `SlabXZ` with
# `assign_materials(..., averaging="kottke")`, Wakis applies **Kottke (2008) harmonic/arithmetic
# subpixel permittivity smoothing** [6]: tangential electric components ($E_x, E_y$) receive the
# volume-weighted arithmetic mean $\langle \varepsilon \rangle = f \varepsilon_1 + (1-f)\varepsilon_2$,
# while the normal component ($E_z$) receives the harmonic mean
# $\langle \varepsilon^{-1} \rangle^{-1} = [f/\varepsilon_1 + (1-f)/\varepsilon_2]^{-1}$, restoring
# smooth second-order convergence across off-grid DBR interfaces.

# %%
from __future__ import annotations

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
from PIL import Image
from scipy.constants import c as c0
from scipy.constants import epsilon_0 as eps0
from scipy.constants import mu_0 as mu0

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _helpers as hlp
from wakis import GridFIT3D, SolverFIT3D, figstyle, schematic
from wakis.dispersive import DispersiveMaterial, Drude
from wakis.field_monitors import DFTMonitor, FluxMonitor
from wakis.near2far import Near2FarMonitor
from wakis.sources import CurrentSource, GaussianSource
from wakis.templates.meshing import graded_axis
from wakis.templates.simulation import assign_materials

DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
ETA0 = float(np.sqrt(mu0 / eps0))

# Reference physical parameters verified from literature:
# 1. PEC mirror & cavity: lambda_0 = 1.0 um (normalized units kh and d = n L / lambda_0), n_cav = 2.4 (Abram et al. 1998)
LAM_PEC = 1.0e-6
N_CAV_ABRAM = 2.4

# 2. Silver (Drude) mirror at lambda_0 = 600 nm (delta_E = 24.41 nm)
LAM_AG = 600.0e-9
AG_EPS_INF = 3.7
AG_WP_EV = 9.01
AG_GAMMA_EV = 0.048

# 3. Blue InGaN micro-RCLED at lambda_0 = 450 nm (Lee et al., Discover Nano 19, 75, 2024, Fig. 1a,b)
LAM_RCLED = 450.0e-9
N_GAN = 2.39
N_NPGAN = 1.70
N_TA2O5 = 2.15
N_SIO2 = 1.46


# %% [markdown]
# ## 4. Simulation building blocks & study task table
#
# Each study task runs a self-contained 2D FIT experiment in the $(x, z)$ plane with `Ny = 3`
# periodic cells along $y$ (`bc_low=["cpml", "periodic", ...]`, `bc_high=["cpml", "periodic", ...]`)
# and writes its compressed results to `data/task_I.npz` and mesh/timing/memory metadata to
# `data/task_I.json`:
#
# * **Task 0 (`pec_drexhage`)**: Line dipole above a PEC mirror (`bc_low[2] = "pec"` at $z = 0$)
#   for $\text{TE}_\parallel$ ($J_y$), $\text{TM}_\parallel$ ($J_x$), and $\text{TM}_\perp$ ($J_z$)
#   at $\text{cpw} = 20$ (`--quick`, $\Delta x = 50\;\text{nm}$, $16\;\text{cpw}$ at $\lambda_{\min} = 800\;\text{nm}$,
#   $4.0$ cells across $h_{\min} = 200\;\text{nm}$) and $\text{cpw} = 25$ (`--full`, $\Delta x = 40\;\text{nm}$,
#   $20\;\text{cpw}$ at $\lambda_{\min}$, $5.0$ cells across $h_{\min}$). Combines multi-frequency broadband
#   extraction across $kh \in [0.25, 9.5]$ with monochromatic runs at $h/\lambda_0 \in \{0.25, 0.50, 0.75\}$
#   recording `Near2FarMonitor(..., ground_plane="-z")` far-field angular patterns, 2D $(x, z)$ DFT
#   wave-front maps, and time-domain movie frames.
# * **Task 1 (`silver_drude`)**: Line dipole at $\lambda_0 = 600\;\text{nm}$ above a $150\;\text{nm}$
#   ($> 6\,\delta_E$) dispersive `Drude` silver mirror across 7 heights $h \in [20, 220]\;\text{nm}$
#   for $J_y$, $J_x$, and $J_z$ on a locally refined `graded_axis` grid ($\Delta z_{\text{Ag}} = 4.0\;\text{nm}$
#   in `--quick`, giving $6.10$ cells per E-field skin depth $\delta_E = 24.41\;\text{nm}$ and $5.0$ cells
#   across $h_{\min} = 20\;\text{nm}$; $\Delta z_{\text{Ag}} = 2.5\;\text{nm}$ in `--full`, giving $9.76$
#   cells/$\delta_E$ and $8.0$ cells across $h_{\min}$). Records total flux-box power $F_{\text{tot}}(kh)$,
#   absorbed power across the metal surface, and vertical electric-field skin-depth decay inside the silver.
# * **Task 2 (`planar_cavity_pec`)**: Line dipole at the midplane of a planar PEC microcavity
#   ($n = 2.4$, Abram *et al.* 1998) across $d = n L / \lambda_0 \in [0.22, 2.32]$ for $J_y$, $J_x$,
#   and $J_z$ using 3 band-limited spectral windows ($N_z = 28$, $12.1\;\text{cpw}$ at $d_{\max}$ in `--quick`;
#   $N_z = 48$, $20.7\;\text{cpw}$ at $d_{\max}$ in `--full`), plus a broadband cutoff-crossing diagnostic
#   run demonstrating the $v_g = 0$ Van Hove cutoff ringing at $d = 0.5$.
# * **Task 3 (`rcled_and_incoherent`)**: Blue $\text{InGaN}$ DBR micro-RCLED ($\lambda_0 = 450\;\text{nm}$,
#   Lee *et al.* 2024 $\text{NP-GaN/GaN}$ bottom DBR + $\text{Ta}_2\text{O}_5/\text{SiO}_2$ top DBR)
#   discretized on an interface-snapped `graded_axis` with $\ge 4.0$ cells (`--quick`, $\text{cpw} = 16$) or
#   $\ge 5.0$ cells (`--full`, $\text{cpw} = 20$) across the thinnest $47.07\;\text{nm}$ GaN quarter-wave
#   layer, comparing tuned ($d = 1.00$) and red-detuned ($d = 1.055$) cavity angular beaming, plus an
#   extended $N_{\text{dip}} = 7$ quantum-well dipole ensemble comparing coherent phased-array fringes
#   against deterministic incoherent power summation and random-phase Monte Carlo realizations.
# * **Tasks 4–10 (`kottke_*`, `staircase_*`, `kottke_reference`)**: Multi-resolution convergence
#   study across `coarse` ($16\;\text{cpw}$, $4.0$ cells/GaN layer), `medium` ($20\;\text{cpw}$ quick /
#   $24\;\text{cpw}$ full), `fine` ($28\;\text{cpw}$ quick / $32\;\text{cpw}$ full), and `reference`
#   ($40\;\text{cpw}$ quick / $48\;\text{cpw}$ full), auditing PEC mirror Purcell error, off-grid DBR
#   resonance convergence under **Kottke subpixel smoothing** vs. **staircasing**, and the Drude silver
#   plasmonic skin-depth resolution study (from $1.02$ and $1.53$ under-resolved demo cells/$\delta_E$
#   up to $10.17\text{--}12.21$ cells/$\delta_E$).


# %%
def get_task_table(quick: bool = False) -> list[dict]:
    """Return the ordered list of study tasks for ``--quick`` or ``--full`` execution.

    Every non-demo task in both ``--quick`` and ``--full`` satisfies the project
    resolution rules:
    1. >= 10 cells per wavelength in the highest-index material at lambda_min in ``--quick``
       (>= 20 cpw for the primary ``--full`` physics tasks).
    2. >= 4 cells across the smallest geometric feature (closest dipole-to-mirror gap h_min
       or thinnest quarter-wave DBR layer d_GaN = lambda_0 / (4 n_GaN) = 47.07 nm).
    3. >= 4 cells per electric-field skin depth delta_E = lambda_0 / (2 pi kappa) = 24.41 nm
       in dispersive Drude silver at the accepted level, validated by a step-by-step
       plasmonic resolution study from 1.02 to 12.21 cells per skin depth.
    """
    if quick:
        cpw_pec = 20  # 20 cpw at lambda_0=1.0 um (16 cpw at lambda_min=800 nm); 4 cells across h_min=200 nm
        cpw_ag = 20
        dz_ag_nm = 4.0  # 6.10 cells per delta_E=24.41 nm (3.05 per delta_I); 5.0 cells across h_min=20 nm
        cpw_cav = 12  # Nz=28 -> 12.07 cpw at d_max=2.32 (28 cpw at d=1.0); 14 cells across L_cav/2
        cpw_rcled = 16  # graded_axis with nodes on every DBR interface: 4 cells across 47.07 nm GaN layer
        cpw_levels = {"coarse": 16, "medium": 20, "fine": 28, "reference": 40}
    else:
        cpw_pec = 25  # 25 cpw at lambda_0=1.0 um (20 cpw at lambda_min=800 nm); 5 cells across h_min=200 nm
        cpw_ag = 24
        dz_ag_nm = 2.5  # 9.77 cells per delta_E=24.41 nm (4.88 per delta_I); 8.0 cells across h_min=20 nm
        cpw_cav = 20  # Nz=48 -> 20.69 cpw at d_max=2.32 (48 cpw at d=1.0); 24 cells across L_cav/2
        cpw_rcled = 20  # graded_axis with nodes on every DBR interface: 5 cells across 47.07 nm GaN layer
        cpw_levels = {"coarse": 16, "medium": 24, "fine": 32, "reference": 48}

    tasks = [
        {
            "index": 0,
            "name": "pec_drexhage",
            "kind": "pec_mirror",
            "cpw": cpw_pec,
            "quick": quick,
            "is_demo": False,
            "averaging": "kottke",
            "description": f"2D PEC mirror Drexhage sweep (Jy, Jx, Jz), Near2Far patterns & GIF (cpw={cpw_pec})",
        },
        {
            "index": 1,
            "name": "silver_drude",
            "kind": "silver_mirror",
            "cpw": cpw_ag,
            "dz_ag_nm": dz_ag_nm,
            "quick": quick,
            "is_demo": False,
            "averaging": "kottke",
            "description": f"2D Drude Ag mirror height sweep vs CPS Sommerfeld integral (dz={dz_ag_nm} nm)",
        },
        {
            "index": 2,
            "name": "planar_cavity_pec",
            "kind": "pec_cavity",
            "cpw": cpw_cav,
            "quick": quick,
            "is_demo": False,
            "averaging": "kottke",
            "description": f"2D planar PEC microcavity Purcell factor F_P(d) vs 2D/3D DOS (cpw={cpw_cav})",
        },
        {
            "index": 3,
            "name": "rcled_and_incoherent",
            "kind": "rcled",
            "cpw": cpw_rcled,
            "quick": quick,
            "is_demo": False,
            "averaging": "kottke",
            "description": f"Blue InGaN DBR micro-RCLED beaming & N=7 incoherent dipole ensemble (cpw={cpw_rcled})",
        },
    ]

    idx = 4
    for lvl in ("coarse", "medium", "fine"):
        for avg in ("kottke", "staircase"):
            tasks.append(
                {
                    "index": idx,
                    "name": f"{avg}_{lvl}",
                    "kind": "convergence",
                    "level": lvl,
                    "cpw": cpw_levels[lvl],
                    "quick": quick,
                    "is_demo": False,
                    "averaging": avg,
                    "description": f"Convergence {avg} {lvl} (cpw={cpw_levels[lvl]})",
                }
            )
            idx += 1

    tasks.append(
        {
            "index": idx,
            "name": "kottke_reference",
            "kind": "convergence",
            "level": "reference",
            "cpw": cpw_levels["reference"],
            "quick": quick,
            "is_demo": False,
            "averaging": "kottke",
            "description": f"High-resolution Kottke reference (cpw={cpw_levels['reference']})",
        }
    )
    return tasks


def _p0_2d(freqs: np.ndarray | float, s_f: np.ndarray | complex, pol: str, n_bg: float = 1.0) -> np.ndarray:
    """Analytical 2D homogeneous-medium power per unit length on FluxMonitor.get_flux()/ly scale."""
    f_arr = np.asarray(freqs, dtype=float)
    k = 2.0 * np.pi * n_bg * f_arr / c0
    eta = ETA0 / n_bg
    pref = 0.25 if pol.lower() in ("y", "te_par", "te") else 0.125
    return pref * k * eta * (np.abs(s_f) ** 2)


def _cfl_dt(x: np.ndarray, y: np.ndarray, z: np.ndarray, cfln: float = 0.5) -> float:
    """Compute the 3D FIT Courant-Friedrichs-Lewy time step dt [s] from node arrays."""
    dx_min = float(np.min(np.diff(x)))
    dy_min = float(np.min(np.diff(y)))
    dz_min = float(np.min(np.diff(z)))
    inv_l = np.sqrt(1.0 / dx_min**2 + 1.0 / dy_min**2 + 1.0 / dz_min**2)
    return float(cfln / (c0 * inv_l))


def _build_rcled_z_axis(
    st: dict, cpw: int, z_min: float, z_mon_top: float, z_max: float
) -> tuple[np.ndarray, float]:
    """Construct a graded z-axis snapped to every RCLED DBR interface and the dipole midplane."""
    min_c = max(4, cpw // 4)
    dz_gan = LAM_RCLED / (cpw * N_GAN)
    z_dip = 0.5 * st["L_cav"]

    pts = [z_min, st["z_bot_end"]]
    sizes = [dz_gan]
    min_cells_list = [max(8, int(round((st["z_bot_end"] - z_min) / dz_gan)))]
    for z0, z1, n_val in st["slabs"]:
        dz_t = LAM_RCLED / (cpw * n_val)
        if abs(z0 - 0.0) < 1e-12 and abs(z1 - st["L_cav"]) < 1e-12:
            pts.extend([z_dip, z1])
            sizes.extend([dz_t, dz_t])
            min_cells_list.extend([2 * min_c, 2 * min_c])
        else:
            pts.append(z1)
            sizes.append(dz_t)
            min_cells_list.append(min_c)
    pts.extend([z_mon_top, z_max])
    sizes.extend([10.0e-9, LAM_RCLED / cpw])
    min_cells_list.extend([3, 8])

    z = graded_axis(pts, sizes, min_cells=min_cells_list, grading=1.3)
    return z, z_dip


def _build_silver_graded_axes(
    dz_ag: float, cpw: int, h_max_fine: float = 240.0e-9, x_max: float = 480.0e-9
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Construct graded (x, y, z) node arrays for the Drude silver mirror with uniform fine core."""
    dx_core = min(2.0 * dz_ag, 16.0e-9)
    dx_bg = LAM_AG / cpw
    n_core = max(3, int(round(48.0e-9 / dx_core)))
    x_core = np.arange(-n_core, n_core + 1) * dx_core
    x_right = graded_axis([x_core[-1], x_max], [dx_bg], min_cells=10, grading=1.25)[1:]
    x = np.concatenate([-x_right[::-1], x_core, x_right])

    Ny = 3
    y = np.arange(Ny + 1) * dx_core

    n_ag_fine = max(4, int(round(84.0e-9 / dz_ag)))
    n_air_fine = max(6, int(round(h_max_fine / dz_ag)))
    z_fine = np.arange(-n_ag_fine, n_air_fine + 1) * dz_ag
    z_top = graded_axis(
        [z_fine[-1], max(h_max_fine + 220.0e-9, 440.0e-9)],
        [dx_bg],
        min_cells=8,
        grading=1.25,
    )[1:]
    z_bot_start = -150.0e-9
    if z_fine[0] > z_bot_start + 1e-12:
        z_bot = -graded_axis(
            [abs(z_fine[0]), abs(z_bot_start)],
            [max(2.0 * dz_ag, 12.0e-9)],
            min_cells=3,
            grading=1.25,
        )[1:][::-1]
        z = np.concatenate([z_bot, z_fine, z_top])
    else:
        z = np.concatenate([z_fine, z_top])

    return x, y, z, dx_core


def build_task_mesh(task_cfg: dict) -> dict:
    """Build the primary ``GridFIT3D`` mesh for ``task_cfg`` and measure resolution metrics.

    Constructs the mesh without running ``SolverFIT3D.emsolve``, measuring
    ``dx_background_nm``, ``dx_min_nm``, ``min_feature_nm``, ``cells_across_feature``,
    ``cpw_worst``, and (for metal tasks) ``skin_depth_nm`` and ``cells_per_skin_depth``
    directly from the ``GridFIT3D`` coordinate arrays.
    """
    kind = str(task_cfg["kind"])
    cpw = int(task_cfg["cpw"])
    quick = bool(task_cfg.get("quick", cpw < 20))
    Ny = 3

    if kind == "pec_mirror":
        dx = LAM_PEC / cpw
        n_x = int(round(1.20e-6 / dx))
        n_z = int(round(2.10e-6 / dx))
        x = np.arange(-n_x, n_x + 1) * dx
        y = np.arange(Ny + 1) * dx
        z = np.arange(0, n_z + 1) * dx
        grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
        h_min = 200.0e-9
        lam_min = LAM_PEC / 1.25  # broadband sweep reaches f_max = 1.25 * f_0
        cells_feat = hlp.count_cells_in_interval(grid.z, 0.0, h_min)
        dx_bg_nm = float(np.max(grid.dz) * 1e9)
        dx_min_nm = float(min(np.min(grid.dx), np.min(grid.dz)) * 1e9)
        cpw_worst = float(lam_min / max(np.max(grid.dx), np.max(grid.dz)))
        f0 = c0 / LAM_PEC
        pulse = GaussianSource(f0, 0.30 * f0, cutoff=3.8)
        dt = _cfl_dt(x, y, z, cfln=0.5)
        n_steps = int(np.ceil((pulse.end_time + 2.5 * LAM_PEC / c0) / dt))
        return {
            "grid": grid,
            "x": x,
            "y": y,
            "z": z,
            "dx_background_nm": round(dx_bg_nm, 3),
            "dx_min_nm": round(dx_min_nm, 3),
            "min_feature_nm": round(h_min * 1e9, 3),
            "cells_across_feature": round(cells_feat, 3),
            "cpw_worst": round(cpw_worst, 3),
            "skin_depth_nm": None,
            "cells_per_skin_depth": None,
            "n_cells": int(grid.Nx * grid.Ny * grid.Nz),
            "n_steps": n_steps,
            "is_demo": bool(task_cfg.get("is_demo", False)),
        }

    if kind == "silver_mirror":
        dz_ag = float(task_cfg.get("dz_ag_nm", 4.0 if quick else 2.5)) * 1e-9
        _, _, delta_e = hlp.drude_silver_permittivity(LAM_AG, AG_EPS_INF, AG_WP_EV, AG_GAMMA_EV)
        x, y, z, dx_core = _build_silver_graded_axes(dz_ag, cpw, h_max_fine=240.0e-9, x_max=480.0e-9)
        grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
        h_min = 20.0e-9
        cells_feat = hlp.count_cells_in_interval(grid.z, 0.0, h_min)
        cells_skin = hlp.count_cells_in_interval(grid.z, -float(delta_e), 0.0)
        dx_bg_nm = float(max(np.max(grid.dx), np.max(grid.dz)) * 1e9)
        dx_min_nm = float(min(np.min(grid.dx), np.min(grid.dz)) * 1e9)
        cpw_worst = float(LAM_AG / max(np.max(grid.dx), np.max(grid.dz)))
        f0 = c0 / LAM_AG
        pulse = GaussianSource(f0, 0.42 * f0, cutoff=3.5)
        dt = _cfl_dt(x, y, z, cfln=0.48)
        n_steps = int(np.ceil((pulse.end_time + 1.8 * LAM_AG / c0) / dt))
        return {
            "grid": grid,
            "x": x,
            "y": y,
            "z": z,
            "dx_core": dx_core,
            "dx_background_nm": round(dx_bg_nm, 3),
            "dx_min_nm": round(dx_min_nm, 3),
            "min_feature_nm": round(h_min * 1e9, 3),
            "cells_across_feature": round(cells_feat, 3),
            "cpw_worst": round(cpw_worst, 3),
            "skin_depth_nm": round(float(delta_e * 1e9), 3),
            "cells_per_skin_depth": round(cells_skin, 3),
            "n_cells": int(grid.Nx * grid.Ny * grid.Nz),
            "n_steps": n_steps,
            "is_demo": bool(task_cfg.get("is_demo", False)),
        }

    if kind == "pec_cavity":
        L_cav = 1.0e-6
        d_max = 2.32
        Nz = 28 if cpw <= 14 else 48
        dz = L_cav / Nz
        dx = dz
        n_x = int(round(1.50e-6 / dx))
        x = np.arange(-n_x, n_x + 1) * dx
        y = np.arange(Ny + 1) * dx
        z = np.linspace(0.0, L_cav, Nz + 1)
        grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
        h_min = 0.5 * L_cav
        cells_feat = hlp.count_cells_in_interval(grid.z, 0.0, h_min)
        dx_bg_nm = float( np.max(grid.dz) * 1e9)
        dx_min_nm = float(min(np.min(grid.dx), np.min(grid.dz)) * 1e9)
        lam_min_mat = L_cav / d_max
        cpw_worst = float(lam_min_mat / max(np.max(grid.dx), np.max(grid.dz)))
        f0 = 0.95 * c0 / (N_CAV_ABRAM * L_cav)
        pulse = GaussianSource(f0, 0.20 * f0, cutoff=4.2)
        dt = _cfl_dt(x, y, z, cfln=0.5) * N_CAV_ABRAM
        n_steps = int(np.ceil((pulse.end_time + 14.0 * N_CAV_ABRAM * L_cav / c0) / dt))
        return {
            "grid": grid,
            "x": x,
            "y": y,
            "z": z,
            "Nz": Nz,
            "dx_background_nm": round(dx_bg_nm, 3),
            "dx_min_nm": round(dx_min_nm, 3),
            "min_feature_nm": round(h_min * 1e9, 3),
            "cells_across_feature": round(cells_feat, 3),
            "cpw_worst": round(cpw_worst, 3),
            "skin_depth_nm": None,
            "cells_per_skin_depth": None,
            "n_cells": int(grid.Nx * grid.Ny * grid.Nz),
            "n_steps": n_steps,
            "is_demo": bool(task_cfg.get("is_demo", False)),
        }

    if kind == "rcled":
        dx = 15.0e-9 if cpw <= 16 else 9.40e-9
        n_x = int(round(1.125e-6 / dx))
        x = np.arange(-n_x, n_x + 1) * dx
        y = np.arange(Ny + 1) * dx
        st = _build_rcled_stack(d_opt=1.00, n_bot_pairs=4, n_top_pairs=2)
        dz_gan = LAM_RCLED / (cpw * N_GAN)
        z_min = st["z_bot_end"] - 10 * dz_gan
        z_mon_top = st["z_top_end"] + 30.0e-9
        z_max = z_mon_top + 8 * (LAM_RCLED / cpw)
        z, z_dip = _build_rcled_z_axis(st, cpw, z_min, z_mon_top, z_max)
        grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
        d_gan_q = LAM_RCLED / (4.0 * N_GAN)
        # Measure cells across the thinnest quarter-wave GaN layer directly from grid.z
        gan_layers = [
            (z0, z1) for (z0, z1, nv) in st["slabs"] if abs((z1 - z0) - d_gan_q) < 1e-11
        ]
        cells_feat = min(
            hlp.count_cells_in_interval(grid.z, z0, z1) for (z0, z1) in gan_layers
        )
        dx_bg_nm = float(max(np.max(grid.dx), np.max(grid.dz)) * 1e9)
        dx_min_nm = float(min(np.min(grid.dx), np.min(grid.dz)) * 1e9)
        # Worst-case cells per wavelength across all dielectric slabs and lateral GaN extent
        cpw_z_min = min(
            (LAM_RCLED / nv)
            / max((z1 - z0) / hlp.count_cells_in_interval(grid.z, z0, z1), 1e-30)
            for (z0, z1, nv) in st["slabs"]
        )
        cpw_x_gan = (LAM_RCLED / N_GAN) / float(np.max(grid.dx))
        cpw_worst = float(min(cpw_z_min, cpw_x_gan))
        f0 = c0 / LAM_RCLED
        pulse = GaussianSource(f0, 0.28 * f0, cutoff=3.8)
        dt = _cfl_dt(x, y, z, cfln=0.5)
        n_steps = int(np.ceil((pulse.end_time + 14.0 * LAM_RCLED / c0) / dt))
        return {
            "grid": grid,
            "x": x,
            "y": y,
            "z": z,
            "z_dip": z_dip,
            "z_mon_top": z_mon_top,
            "dx_background_nm": round(dx_bg_nm, 3),
            "dx_min_nm": round(dx_min_nm, 3),
            "min_feature_nm": round(d_gan_q * 1e9, 3),
            "cells_across_feature": round(cells_feat, 3),
            "cpw_worst": round(cpw_worst, 3),
            "skin_depth_nm": None,
            "cells_per_skin_depth": None,
            "n_cells": int(grid.Nx * grid.Ny * grid.Nz),
            "n_steps": n_steps,
            "is_demo": bool(task_cfg.get("is_demo", False)),
        }

    if kind == "convergence":
        dz_dbr = LAM_RCLED / (cpw * N_GAN)
        dx_dbr = LAM_RCLED / (22.0 * N_GAN)
        x_dbr = np.arange(-26, 27) * dx_dbr
        y_dbr = np.arange(Ny + 1) * dx_dbr
        z_surf = -0.172 * LAM_RCLED
        d_hi = LAM_RCLED / (4.0 * N_GAN)
        d_lo = LAM_RCLED / (4.0 * N_SIO2)
        intervals_dbr = []
        zc = z_surf
        for _ in range(2):
            intervals_dbr.append((zc - d_hi, zc, N_GAN**2))
            zc -= d_hi
            intervals_dbr.append((zc - d_lo, zc, N_SIO2**2))
            zc -= d_lo
        intervals_dbr = list(reversed(intervals_dbr))
        z_bot_dbr = intervals_dbr[0][0] - 120.0e-9
        z_top_dbr = 260.0e-9
        z_dbr = (
            np.arange(
                int(np.floor(z_bot_dbr / dz_dbr)), int(np.ceil(z_top_dbr / dz_dbr)) + 1
            )
            * dz_dbr
        )
        grid_d = GridFIT3D(x=x_dbr, y=y_dbr, z=z_dbr, verbose=0)
        # Measure cells across the thinnest off-grid GaN quarter-wave layer (47.071 nm)
        top_gan_lo, top_gan_hi, _ = intervals_dbr[-1]
        cells_feat = hlp.count_cells_in_interval(grid_d.z, top_gan_lo, top_gan_hi)
        dx_bg_nm = float(max(np.max(grid_d.dx), np.max(grid_d.dz)) * 1e9)
        dx_min_nm = float(min(np.min(grid_d.dx), np.min(grid_d.dz)) * 1e9)
        lam_min_dbr = (LAM_RCLED / 1.08) / N_GAN
        cpw_worst = float(lam_min_dbr / max(np.max(grid_d.dx), np.max(grid_d.dz)))

        # Measure Drude silver skin-depth resolution on the companion silver mesh
        _, _, delta_e = hlp.drude_silver_permittivity(LAM_AG, AG_EPS_INF, AG_WP_EV, AG_GAMMA_EV)
        dz_ag = 96.0e-9 / cpw
        _, _, z_ag, _ = _build_silver_graded_axes(dz_ag, max(cpw, 20), h_max_fine=72.0e-9, x_max=420.0e-9)
        cells_skin = hlp.count_cells_in_interval(z_ag, -float(delta_e), 0.0)
        has_metal = str(task_cfg.get("averaging", "kottke")) == "kottke"

        f0_dbr = c0 / LAM_RCLED
        pulse_d = GaussianSource(f0_dbr, 0.35 * f0_dbr, cutoff=3.8)
        dt_d = _cfl_dt(x_dbr, y_dbr, z_dbr, cfln=0.5)
        n_steps = int(np.ceil((pulse_d.end_time + 3.2 * LAM_RCLED / c0) / dt_d))
        return {
            "grid": grid_d,
            "x": x_dbr,
            "y": y_dbr,
            "z": z_dbr,
            "intervals_dbr": intervals_dbr,
            "dx_background_nm": round(dx_bg_nm, 3),
            "dx_min_nm": round(dx_min_nm, 3),
            "min_feature_nm": round(d_hi * 1e9, 3),
            "cells_across_feature": round(cells_feat, 3),
            "cpw_worst": round(cpw_worst, 3),
            "skin_depth_nm": round(float(delta_e * 1e9), 3) if has_metal else None,
            "cells_per_skin_depth": round(cells_skin, 3) if has_metal else None,
            "n_cells": int(grid_d.Nx * grid_d.Ny * grid_d.Nz),
            "n_steps": n_steps,
            "is_demo": bool(task_cfg.get("is_demo", False)),
        }

    raise ValueError(f"Unknown task kind {kind!r}")


def _attach_mesh_meta(payload: dict, mesh_info: dict) -> dict:
    """Attach standard mesh resolution audit fields to ``payload`` for NPZ storage."""
    payload["dx_background_nm"] = float(mesh_info["dx_background_nm"])
    payload["dx_min_nm"] = float(mesh_info["dx_min_nm"])
    payload["min_feature_nm"] = float(mesh_info["min_feature_nm"])
    payload["cells_across_feature"] = float(mesh_info["cells_across_feature"])
    payload["skin_depth_nm"] = (
        float(mesh_info["skin_depth_nm"])
        if mesh_info["skin_depth_nm"] is not None
        else np.nan
    )
    payload["cells_per_skin_depth"] = (
        float(mesh_info["cells_per_skin_depth"])
        if mesh_info["cells_per_skin_depth"] is not None
        else np.nan
    )
    payload["n_cells"] = int(mesh_info["n_cells"])
    payload["n_steps"] = int(mesh_info["n_steps"])
    return payload


def _run_pec_drexhage_task(cfg: dict) -> dict:
    """Run Task 0: PEC mirror Drexhage sweep, angular patterns, field snapshots, and movie frames."""
    mesh_info = build_task_mesh(cfg)
    cpw = int(cfg["cpw"])
    dx = LAM_PEC / cpw
    f0 = c0 / LAM_PEC
    k0 = 2.0 * np.pi / LAM_PEC
    Ny = 3
    ly_box = dx
    x = mesh_info["x"]
    y = mesh_info["y"]
    z = mesh_info["z"]
    grid = mesh_info["grid"]
    y_mid = 0.5 * Ny * dx

    # 1. Broadband multi-frequency sweep across 3 anchor heights (h >= 200 nm >= 4 cells)
    #    covering kh in [0.25, 9.5] with f_max <= 1.25 * f0 (lambda_min = 800 nm >= 16 cpw)
    anchor_heights_m = [0.20 * LAM_PEC, 0.50 * LAM_PEC, 1.20 * LAM_PEC]
    anchor_bands = [
        np.linspace(0.20 * f0, 1.15 * f0, 12),
        np.linspace(0.48 * f0, 1.25 * f0, 12),
        np.linspace(0.54 * f0, 1.25 * f0, 14),
    ]
    kh_all = {"y": [], "x": [], "z": []}
    f_work_all = {"y": [], "x": [], "z": []}

    for pol in ("y", "x", "z"):
        for h_target, f_eval in zip(anchor_heights_m, anchor_bands):
            ic = max(4, int(round(h_target / dx)))
            h_val = (ic + 0.5) * dx if pol == "z" else ic * dx
            solver = SolverFIT3D(
                grid,
                bc_low=["cpml", "periodic", "pec"],
                bc_high=["cpml", "periodic", "cpml"],
                n_pml=8,
                use_stl=False,
                verbose=0,
            )
            pulse = GaussianSource(f0, 0.45 * f0, cutoff=3.8)
            Nt = int(np.ceil((pulse.end_time + 2.8 * (2.0 * h_val + LAM_PEC) / c0) / solver.dt))
            src = CurrentSource(
                pulse,
                component=pol,
                center=(0.0, y_mid, h_val),
                size=(0.0, 1.0, 0.0),
                amplitude=1.0,
            )
            dft_pt = DFTMonitor(
                f_eval, ["E" + pol], center=(0.0, y_mid, h_val), size=(0.0, 0.0, 0.0)
            )
            solver.emsolve(Nt, source=[src], monitors=[dft_pt])
            s_f = pulse.fourier_transform(f_eval)
            e_pt = dft_pt.get_field("E" + pol)[0].reshape(-1)
            p0 = _p0_2d(f_eval, s_f, pol)
            f_w = -np.real(e_pt * np.conj(s_f)) / p0
            kh_pts = (2.0 * np.pi * f_eval / c0) * h_val
            kh_all[pol].append(kh_pts)
            f_work_all[pol].append(f_w)

    for pol in ("y", "x", "z"):
        kh_cat = np.concatenate(kh_all[pol])
        fw_cat = np.concatenate(f_work_all[pol])
        order = np.argsort(kh_cat)
        kh_all[pol] = kh_cat[order]
        f_work_all[pol] = fw_cat[order]

    # 2. Monochromatic heights h / lambda_0 in {0.25, 0.50, 0.75} (5, 10, 15 cells in quick):
    #    Near2FarMonitor(ground_plane="-z") box power + angular radiation patterns + 2D field maps + movie
    theta_deg = np.linspace(-86.0, 86.0, 173)
    theta_rad = np.deg2rad(theta_deg)

    h_ratios = [0.25, 0.50, 0.75]
    mono_kh = {"y": [], "x": [], "z": []}
    mono_flux_f = {"y": [], "x": [], "z": []}
    mono_work_f = {"y": [], "x": [], "z": []}
    patterns_num = {}
    patterns_ana = {}
    snapshots = {}
    movie_frames = []
    movie_times = []
    nz_snap = min(len(z) - 8, int(round(1.70e-6 / dx)))

    for pol in ("y", "x", "z"):
        for h_rat in h_ratios:
            ic = max(4, int(round(h_rat * LAM_PEC / dx)))
            h_val = (ic + 0.5) * dx if pol == "z" else ic * dx
            kh_val = k0 * h_val
            solver = SolverFIT3D(
                grid,
                bc_low=["cpml", "periodic", "pec"],
                bc_high=["cpml", "periodic", "cpml"],
                n_pml=8,
                use_stl=False,
                verbose=0,
            )
            pulse = GaussianSource(f0, 0.30 * f0, cutoff=3.8)
            Nt = int(np.ceil((pulse.end_time + 2.5 * LAM_PEC / c0) / solver.dt))
            src = CurrentSource(
                pulse,
                component=pol,
                center=(0.0, y_mid, h_val),
                size=(0.0, 1.0, 0.0),
                amplitude=1.0,
            )
            z_top_box = max((ic + 4) * dx, 10 * dx)
            n2f = Near2FarMonitor(
                frequencies=[f0],
                center=(0.0, y_mid, 0.5 * z_top_box),
                size=(14 * dx, ly_box, z_top_box),
                ground_plane="-z",
            )
            dft_pt = DFTMonitor(
                [f0], ["E" + pol], center=(0.0, y_mid, h_val), size=(0.0, 0.0, 0.0)
            )
            mons = [n2f, dft_pt]

            want_snap = (pol == "y") and (h_rat in (0.25, 0.50))
            if want_snap:
                z_snap_span = nz_snap * dx
                dft_2d = DFTMonitor(
                    [f0],
                    ["Ey"],
                    center=(0.0, y_mid, 0.5 * z_snap_span),
                    size=(2.0 * (x[-9] - 0.0), 0.0, z_snap_span),
                )
                mons.append(dft_2d)

            want_movie = (pol == "y") and (h_rat == 0.50)
            if want_movie:
                stride = max(1, Nt // 32)

                def _cb(s, t, _st=stride):
                    if s.nstep % _st == 0 and len(movie_frames) < 30:
                        ey_slice = np.asarray(
                            s.E[8:-8, 1, :nz_snap, "y"], dtype=float
                        ).copy()
                        movie_frames.append(ey_slice)
                        movie_times.append(float(t))

                solver.emsolve(Nt, source=[src], monitors=mons, callback=_cb)
            else:
                solver.emsolve(Nt, source=[src], monitors=mons)

            s_f = pulse.fourier_transform(f0)
            p0 = float(_p0_2d(f0, s_f, pol))
            f_flux = float((n2f.get_flux()[0] / ly_box) / p0)
            e_pt = complex(dft_pt.get_field("E" + pol)[0].squeeze())
            f_w = float(-np.real(e_pt * np.conj(s_f)) / p0)

            mono_kh[pol].append(kh_val)
            mono_flux_f[pol].append(f_flux)
            mono_work_f[pol].append(f_w)

            pat_n = hlp.farfield_2d_from_monitor(n2f, theta_rad, ly=ly_box, freq_idx=0) / p0
            pat_a = hlp.pec_mirror_pattern_2d(theta_rad, kh_val, pol)
            key = f"{pol}_{int(round(h_rat * 100)):03d}"
            patterns_num[key] = pat_n
            patterns_ana[key] = pat_a

            if want_snap:
                ey_map, coords = dft_2d.get_field("Ey")
                snapshots[f"ey_{int(round(h_rat * 100)):03d}"] = ey_map[0, :, 0, :]
                snapshots["snap_x"] = coords[0]
                snapshots["snap_z"] = coords[2]

    payload = {
        "name": cfg["name"],
        "cpw": cpw,
        "theta_deg": theta_deg,
        "theta_rad": theta_rad,
        "kh_y": kh_all["y"],
        "kh_x": kh_all["x"],
        "kh_z": kh_all["z"],
        "F_work_y": f_work_all["y"],
        "F_work_x": f_work_all["x"],
        "F_work_z": f_work_all["z"],
        "F_ana_y": hlp.pec_mirror_purcell_2d(kh_all["y"], "y"),
        "F_ana_x": hlp.pec_mirror_purcell_2d(kh_all["x"], "x"),
        "F_ana_z": hlp.pec_mirror_purcell_2d(kh_all["z"], "z"),
        "mono_kh_y": np.asarray(mono_kh["y"]),
        "mono_kh_x": np.asarray(mono_kh["x"]),
        "mono_kh_z": np.asarray(mono_kh["z"]),
        "mono_flux_y": np.asarray(mono_flux_f["y"]),
        "mono_flux_x": np.asarray(mono_flux_f["x"]),
        "mono_flux_z": np.asarray(mono_flux_f["z"]),
        "mono_work_y": np.asarray(mono_work_f["y"]),
        "mono_work_x": np.asarray(mono_work_f["x"]),
        "mono_work_z": np.asarray(mono_work_f["z"]),
        "movie_frames": np.asarray(movie_frames),
        "movie_times": np.asarray(movie_times),
        "movie_x": x[8:-8],
        "movie_z": z[:nz_snap],
    }
    for k, v in patterns_num.items():
        payload[f"pat_num_{k}"] = v
        payload[f"pat_ana_{k}"] = patterns_ana[k]
    payload.update(snapshots)
    return _attach_mesh_meta(payload, mesh_info)


def _run_silver_drude_task(cfg: dict) -> dict:
    """Run Task 1: 2D line dipole above a dispersive Drude silver mirror vs CPS Sommerfeld."""
    mesh_info = build_task_mesh(cfg)
    dz = float(cfg.get("dz_ag_nm", 4.0)) * 1e-9
    dx_core = float(mesh_info["dx_core"])
    f0 = c0 / LAM_AG
    k0 = 2.0 * np.pi / LAM_AG
    eps_m, _, delta_e = hlp.drude_silver_permittivity(
        LAM_AG, AG_EPS_INF, AG_WP_EV, AG_GAMMA_EV
    )
    cells_per_skin = float(mesh_info["cells_per_skin_depth"])

    Ny = 3
    ly = dx_core
    x = mesh_info["x"]
    y = mesh_info["y"]
    z = mesh_info["z"]
    grid = mesh_info["grid"]
    y_mid = 0.5 * Ny * dx_core
    z_ag = float(z[0])

    # Every dipole height h >= 20.0 nm spans >= 5.0 cells in --quick (dz=4.0 nm)
    # and >= 8.0 cells in --full (dz=2.5 nm)
    h_target_nm = np.array([20.0, 32.0, 48.0, 72.0, 112.0, 160.0, 220.0])
    kh_num = {"y": [], "x": [], "z": []}
    f_tot_box = {"y": [], "x": [], "z": []}
    f_abs_metal = {"y": [], "x": [], "z": []}
    skin_z_nm = None
    skin_ey_norm = None
    skin_ez_norm = None
    x_plane_span = 2.0 * float(abs(x[8]))

    for pol in ("y", "x", "z"):
        for h_nm in h_target_nm:
            ih = max(5, int(round(h_nm * 1e-9 / dz)))
            h_val = (ih + 0.5) * dz if pol == "z" else ih * dz
            kh = k0 * h_val

            solver = SolverFIT3D(
                grid,
                bc_low=["cpml", "periodic", "pec"],
                bc_high=["cpml", "periodic", "cpml"],
                n_pml=8,
                cfln=0.48,
                use_stl=False,
                verbose=0,
            )
            ag = DispersiveMaterial(
                AG_EPS_INF,
                [Drude(AG_WP_EV * hlp.EV_TO_RAD_S, AG_GAMMA_EV * hlp.EV_TO_RAD_S)],
            )
            solver.add_dispersive_material(
                ag, center=(0.0, 0.0, 0.5 * z_ag), size=(2.0, 2.0, abs(z_ag))
            )
            pulse = GaussianSource(f0, 0.42 * f0, cutoff=3.5)
            Nt = int(np.ceil((pulse.end_time + 1.8 * LAM_AG / c0) / solver.dt))
            src = CurrentSource(
                pulse,
                component=pol,
                center=(0.0, y_mid, h_val),
                size=(0.0, 1.0, 0.0),
                amplitude=1.0,
            )
            box = FluxMonitor(
                [f0], center=(0.0, y_mid, h_val), size=(4 * dx_core, ly, 4 * dz)
            )
            plane_ag = FluxMonitor(
                [f0], center=(0.0, y_mid, dz), size=(x_plane_span, ly, 0.0)
            )
            mons = [box, plane_ag]

            want_skin = (abs(h_nm - 48.0) < 1e-3) and (pol in ("y", "z"))
            if want_skin:
                comp_prof = "Ey" if pol == "y" else "Ex"
                dft_z = DFTMonitor(
                    [f0],
                    [comp_prof],
                    center=(3 * dx_core if pol == "z" else 0.0, y_mid, -20.0e-9),
                    size=(0.0, 0.0, 220.0e-9),
                )
                mons.append(dft_z)

            solver.emsolve(Nt, source=[src], monitors=mons)
            s_f = pulse.fourier_transform(f0)
            p0 = float(_p0_2d(f0, s_f, pol))
            ft_val = float((box.get_flux()[0] / ly) / p0)
            fa_val = float(-(plane_ag.get_flux()[0] / ly) / p0)

            kh_num[pol].append(kh)
            f_tot_box[pol].append(ft_val)
            f_abs_metal[pol].append(max(0.0, fa_val))

            if want_skin:
                f_arr, coords = dft_z.get_field(comp_prof)
                z_prof = coords[2]
                amp = np.abs(f_arr.reshape(-1))
                iz_surf = int(np.argmin(np.abs(z_prof - 0.0)))
                amp_norm = amp / max(float(amp[iz_surf]), 1e-30)
                skin_z_nm = z_prof * 1e9
                if pol == "y":
                    skin_ey_norm = amp_norm
                else:
                    skin_ez_norm = amp_norm

    # Dense CPS analytical reference curves
    kh_dense = np.linspace(0.20, 4.2, 85)
    cps_y = hlp.cps_sommerfeld_2d(kh_dense, eps_m, pol="y")
    cps_x = hlp.cps_sommerfeld_2d(kh_dense, eps_m, pol="x")
    cps_z = hlp.cps_sommerfeld_2d(kh_dense, eps_m, pol="z")

    # CPS exact values at the simulated kh points
    cps_num_y = hlp.cps_sommerfeld_2d(kh_num["y"], eps_m, pol="y")
    cps_num_x = hlp.cps_sommerfeld_2d(kh_num["x"], eps_m, pol="x")
    cps_num_z = hlp.cps_sommerfeld_2d(kh_num["z"], eps_m, pol="z")

    # Wavenumber-resolved CPS integrand dF/du across the light line (u=1) and SPP pole
    u_grid = np.linspace(0.0, 2.6, 500)
    kh_spec = 0.42
    df_du_y = hlp.cps_sommerfeld_integrand_2d(u_grid, kh_spec, eps_m, pol="y")
    df_du_x = hlp.cps_sommerfeld_integrand_2d(u_grid, kh_spec, eps_m, pol="x")
    df_du_z = hlp.cps_sommerfeld_integrand_2d(u_grid, kh_spec, eps_m, pol="z")
    u_spp = float(np.real(np.sqrt(eps_m / (eps_m + 1.0))))

    payload = {
        "name": cfg["name"],
        "dz_ag_nm": float(dz * 1e9),
        "skin_depth_nm": float(delta_e * 1e9),
        "skin_depth_power_nm": float(0.5 * delta_e * 1e9),
        "cells_per_skin_depth": cells_per_skin,
        "eps_m_real": float(np.real(eps_m)),
        "eps_m_imag": float(np.imag(eps_m)),
        "u_spp": u_spp,
        "kh_num_y": np.asarray(kh_num["y"]),
        "kh_num_x": np.asarray(kh_num["x"]),
        "kh_num_z": np.asarray(kh_num["z"]),
        "F_tot_num_y": np.asarray(f_tot_box["y"]),
        "F_tot_num_x": np.asarray(f_tot_box["x"]),
        "F_tot_num_z": np.asarray(f_tot_box["z"]),
        "F_abs_num_y": np.asarray(f_abs_metal["y"]),
        "F_abs_num_x": np.asarray(f_abs_metal["x"]),
        "F_abs_num_z": np.asarray(f_abs_metal["z"]),
        "F_tot_cps_pts_y": cps_num_y["F_tot"],
        "F_tot_cps_pts_x": cps_num_x["F_tot"],
        "F_tot_cps_pts_z": cps_num_z["F_tot"],
        "kh_dense": kh_dense,
        "cps_y_tot": cps_y["F_tot"],
        "cps_y_rad": cps_y["F_rad"],
        "cps_y_nonrad": cps_y["F_nonrad"],
        "cps_x_tot": cps_x["F_tot"],
        "cps_x_rad": cps_x["F_rad"],
        "cps_x_nonrad": cps_x["F_nonrad"],
        "cps_z_tot": cps_z["F_tot"],
        "cps_z_rad": cps_z["F_rad"],
        "cps_z_nonrad": cps_z["F_nonrad"],
        "cps_z_eta": cps_z["eta_rad"],
        "cps_y_eta": cps_y["eta_rad"],
        "u_grid": u_grid,
        "df_du_y": df_du_y,
        "df_du_x": df_du_x,
        "df_du_z": df_du_z,
        "skin_z_nm": skin_z_nm if skin_z_nm is not None else np.zeros(1),
        "skin_ey_norm": skin_ey_norm if skin_ey_norm is not None else np.zeros(1),
        "skin_ez_norm": skin_ez_norm if skin_ez_norm is not None else np.zeros(1),
    }
    return _attach_mesh_meta(payload, mesh_info)



def _run_planar_cavity_pec_task(cfg: dict) -> dict:
    """Run Task 2: 2D planar PEC microcavity Purcell factor F_P(d) vs 2D and 3D analytical DOS."""
    mesh_info = build_task_mesh(cfg)
    n_cav = N_CAV_ABRAM
    eps_cav = n_cav**2
    L_cav = 1.0e-6
    Nz = int(mesh_info["Nz"])
    dz = L_cav / Nz
    Ny = 3
    x = mesh_info["x"]
    y = mesh_info["y"]
    z = mesh_info["z"]
    grid = mesh_info["grid"]
    y_mid = 0.5 * Ny * dz
    z_mid_node = 0.5 * L_cav
    z_mid_edge = (Nz // 2 + 0.5) * dz

    # Band-limited spectral windows strictly avoiding the v_g = 0 cutoffs (d = 0.5, 1.5 for y,x; d = 1.0, 2.0 for z)
    bands_yx = [
        (0.32, 0.25, np.linspace(0.22, 0.45, 8)),
        (0.95, 0.20, np.linspace(0.60, 1.36, 15)),
        (1.92, 0.14, np.linspace(1.62, 2.30, 12)),
    ]
    bands_z = [
        (0.55, 0.25, np.linspace(0.25, 0.88, 12)),
        (1.48, 0.16, np.linspace(1.12, 1.86, 12)),
        (2.20, 0.10, np.linspace(2.08, 2.32, 6)),
    ]

    d_sim = {"y": [], "x": [], "z": []}
    f_sim = {"y": [], "x": [], "z": []}

    for pol in ("y", "x", "z"):
        z_src = z_mid_edge if pol == "z" else z_mid_node
        bands = bands_z if pol == "z" else bands_yx
        for d0, bw_rel, d_pts in bands:
            f0 = d0 * c0 / (n_cav * L_cav)
            f_pts = d_pts * c0 / (n_cav * L_cav)
            solver = SolverFIT3D(
                grid,
                bc_low=["cpml", "periodic", "pec"],
                bc_high=["cpml", "periodic", "pec"],
                bg=[eps_cav, 1.0],
                n_pml=8,
                use_stl=False,
                verbose=0,
            )
            pulse = GaussianSource(f0, bw_rel * f0, cutoff=4.2)
            Nt = int(np.ceil((pulse.end_time + 14.0 * n_cav * L_cav / c0) / solver.dt))
            src = CurrentSource(
                pulse,
                component=pol,
                center=(0.0, y_mid, z_src),
                size=(0.0, 1.0, 0.0),
                amplitude=1.0,
            )
            dft_pt = DFTMonitor(
                f_pts, ["E" + pol], center=(0.0, y_mid, z_src), size=(0.0, 0.0, 0.0)
            )
            solver.emsolve(Nt, source=[src], monitors=[dft_pt])
            s_f = pulse.fourier_transform(f_pts)
            e_pt = dft_pt.get_field("E" + pol)[0].reshape(-1)
            p0 = _p0_2d(f_pts, s_f, pol, n_bg=n_cav)
            f_val = -np.real(e_pt * np.conj(s_f)) / p0
            # Account for half-cell offset of Ez edge from exact midplane z/L = 0.5
            if pol == "z":
                corr = hlp.planar_cavity_purcell_2d(d_pts, "z", z_rel=0.5) / np.maximum(
                    hlp.planar_cavity_purcell_2d(d_pts, "z", z_rel=z_src / L_cav), 1e-12
                )
                f_val = f_val * corr
            d_sim[pol].append(d_pts)
            f_sim[pol].append(np.maximum(0.0, f_val))

    for pol in ("y", "x", "z"):
        d_sim[pol] = np.concatenate(d_sim[pol])
        f_sim[pol] = np.concatenate(f_sim[pol])

    # Diagnostic run: broadband pulse crossing the d = 0.50 TE1 cutoff (demonstrating v_g = 0 ringing)
    d_ring_pts = np.linspace(0.25, 1.35, 70)
    f_ring_pts = d_ring_pts * c0 / (n_cav * L_cav)
    f0_ring = 0.65 * c0 / (n_cav * L_cav)
    solver_r = SolverFIT3D(
        grid,
        bc_low=["cpml", "periodic", "pec"],
        bc_high=["cpml", "periodic", "pec"],
        bg=[eps_cav, 1.0],
        n_pml=8,
        use_stl=False,
        verbose=0,
    )
    pulse_r = GaussianSource(f0_ring, 0.42 * f0_ring, cutoff=3.8)
    Nt_r = int(np.ceil((pulse_r.end_time + 10.0 * n_cav * L_cav / c0) / solver_r.dt))
    src_r = CurrentSource(
        pulse_r,
        component="y",
        center=(0.0, y_mid, z_mid_node),
        size=(0.0, 1.0, 0.0),
        amplitude=1.0,
    )
    dft_r = DFTMonitor(
        f_ring_pts, ["Ey"], center=(0.0, y_mid, z_mid_node), size=(0.0, 0.0, 0.0)
    )
    trace_t = []
    trace_ey = []
    ix_mid = grid.Nx // 2
    iz_mid = grid.Nz // 2

    def _cb_ring(s, t):
        trace_t.append(float(t))
        trace_ey.append(float(s.E[ix_mid, 1, iz_mid, "y"]))

    solver_r.emsolve(Nt_r, source=[src_r], monitors=[dft_r], callback=_cb_ring)
    s_f_r = pulse_r.fourier_transform(f_ring_pts)
    e_pt_r = dft_r.get_field("Ey")[0].reshape(-1)
    p0_r = _p0_2d(f_ring_pts, s_f_r, "y", n_bg=n_cav)
    f_ring_spec = -np.real(e_pt_r * np.conj(s_f_r)) / p0_r

    d_dense = np.linspace(0.18, 2.38, 400)
    payload = {
        "name": cfg["name"],
        "n_cav": n_cav,
        "d_sim_y": d_sim["y"],
        "d_sim_x": d_sim["x"],
        "d_sim_z": d_sim["z"],
        "F_sim_y": f_sim["y"],
        "F_sim_x": f_sim["x"],
        "F_sim_z": f_sim["z"],
        "F_ana_pts_y": hlp.planar_cavity_purcell_2d(d_sim["y"], "y"),
        "F_ana_pts_x": hlp.planar_cavity_purcell_2d(d_sim["x"], "x"),
        "F_ana_pts_z": hlp.planar_cavity_purcell_2d(d_sim["z"], "z"),
        "d_dense": d_dense,
        "F_2d_dense_y": hlp.planar_cavity_purcell_2d(d_dense, "y"),
        "F_2d_dense_x": hlp.planar_cavity_purcell_2d(d_dense, "x"),
        "F_2d_dense_z": hlp.planar_cavity_purcell_2d(d_dense, "z"),
        "F_3d_dense_par": hlp.planar_cavity_purcell_3d(d_dense, "par"),
        "F_3d_dense_perp": hlp.planar_cavity_purcell_3d(d_dense, "perp"),
        "d_ring_pts": d_ring_pts,
        "F_ring_spec": f_ring_spec,
        "trace_t": np.asarray(trace_t),
        "trace_ey": np.asarray(trace_ey),
    }
    return _attach_mesh_meta(payload, mesh_info)


def _build_rcled_stack(d_opt: float = 1.0, n_bot_pairs: int = 8, n_top_pairs: int = 4):
    """Construct the Lee et al. (2024) blue InGaN DBR micro-RCLED layer stack along z."""
    d_npgan = LAM_RCLED / (4.0 * N_NPGAN)
    d_gan_q = LAM_RCLED / (4.0 * N_GAN)
    d_ta2o5 = LAM_RCLED / (4.0 * N_TA2O5)
    d_sio2 = LAM_RCLED / (4.0 * N_SIO2)
    L_cav = d_opt * LAM_RCLED / N_GAN

    bot_layers = []
    for _ in range(n_bot_pairs):
        bot_layers.append((d_npgan, N_NPGAN))
        bot_layers.append((d_gan_q, N_GAN))

    # Facing the high-index GaN cavity (n = 2.39), the low-index SiO2 quarter-wave layer
    # comes first so both mirrors have zero reflection phase (r_top > 0, r_bot > 0).
    top_layers = []
    for _ in range(n_top_pairs):
        top_layers.append((d_sio2, N_SIO2))
        top_layers.append((d_ta2o5, N_TA2O5))

    # Build z-coordinate intervals ordered from bottom (z_bot_end) to top (z_top_end)
    bot_slabs = []
    z_cur = 0.0
    for thick, n_val in bot_layers:
        bot_slabs.append((z_cur - thick, z_cur, n_val))
        z_cur -= thick
    z_bot_end = z_cur
    slabs = list(reversed(bot_slabs))

    # Active GaN cavity [0, L_cav]
    slabs.append((0.0, L_cav, N_GAN))
    z_cur = L_cav
    for thick, n_val in top_layers:
        slabs.append((z_cur, z_cur + thick, n_val))
        z_cur += thick
    z_top_end = z_cur

    return {
        "L_cav": L_cav,
        "z_bot_end": z_bot_end,
        "z_top_end": z_top_end,
        "bot_layers": bot_layers,
        "top_layers": top_layers,
        "slabs": slabs,
    }


def _run_rcled_and_incoherent_task(cfg: dict) -> dict:
    """Run Task 3: Blue InGaN DBR micro-RCLED angular beaming + N=7 incoherent dipole ensemble."""
    mesh_info = build_task_mesh(cfg)
    cpw = int(cfg["cpw"])
    f0 = c0 / LAM_RCLED
    x = mesh_info["x"]
    y = mesh_info["y"]
    dx = float(x[1] - x[0])
    Ny = 3
    ly = dx
    y_mid = 0.5 * Ny * dx
    dz_gan = LAM_RCLED / (cpw * N_GAN)

    theta_deg = np.linspace(-80.0, 80.0, 161)
    theta_rad = np.deg2rad(theta_deg)

    # 1. TMM reflectance spectra of bottom NP-GaN/GaN DBR, top SiO2/Ta2O5 DBR, and full RCLED
    wl_tmm = np.linspace(380.0e-9, 530.0e-9, 301)
    st_tuned = _build_rcled_stack(d_opt=1.0, n_bot_pairs=10, n_top_pairs=5)
    r_bot_spec = hlp.tmm_stack_reflectance(
        wl_tmm, st_tuned["bot_layers"], n_inc=N_GAN, n_sub=N_GAN
    )["R"]
    r_top_spec = hlp.tmm_stack_reflectance(
        wl_tmm, st_tuned["top_layers"], n_inc=N_GAN, n_sub=1.0
    )["R"]
    full_layers = (
        list(reversed(st_tuned["bot_layers"]))
        + [(st_tuned["L_cav"], N_GAN)]
        + st_tuned["top_layers"]
    )
    r_full_spec = hlp.tmm_stack_reflectance(
        wl_tmm, full_layers, n_inc=N_GAN, n_sub=1.0
    )["R"]

    # 2. Wakis 2D RCLED simulations for tuned (d = 1.00) and red-detuned (d = 1.055) cavities
    #    using graded_axis with nodes on every DBR interface and >= 4 cells per quarter-wave layer
    rcled_results = {}
    for label, d_opt in (("tuned", 1.00), ("detuned", 1.055)):
        st = _build_rcled_stack(d_opt=d_opt, n_bot_pairs=4, n_top_pairs=2)
        z_min = st["z_bot_end"] - 10 * dz_gan
        z_mon_top = st["z_top_end"] + 30.0e-9
        z_max = z_mon_top + 8 * (LAM_RCLED / cpw)
        z, z_dip = _build_rcled_z_axis(st, cpw, z_min, z_mon_top, z_max)
        grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
        solver = SolverFIT3D(
            grid,
            bc_low=["cpml", "periodic", "cpml"],
            bc_high=["cpml", "periodic", "cpml"],
            n_pml=8,
            use_stl=False,
            verbose=0,
        )
        dz_pad = float(np.max(grid.dz))
        n_slabs = len(st["slabs"])
        shapes = [
            hlp.SlabXZ(
                -1.5,
                1.5,
                z_min - dz_pad,
                st["z_bot_end"] + 0.6 * dz_pad,
                material=N_GAN**2,
                snap=False,
            )
        ] + [
            hlp.SlabXZ(
                -1.5,
                1.5,
                z0,
                z1 + (0.6 * dz_pad if idx_s < n_slabs - 1 else 0.0),
                material=n_val**2,
                snap=False,
            )
            for idx_s, (z0, z1, n_val) in enumerate(st["slabs"])
        ]
        assign_materials(solver, shapes, f0, averaging="kottke")

        pulse = GaussianSource(f0, 0.28 * f0, cutoff=3.8)
        Nt = int(np.ceil((pulse.end_time + 14.0 * LAM_RCLED / c0) / solver.dt))
        src = CurrentSource(
            pulse,
            component="y",
            center=(0.0, y_mid, z_dip),
            size=(0.0, 1.0, 0.0),
            amplitude=1.0,
        )
        x_ap_span = 2.0 * float(abs(x[9]))
        n2f_top = Near2FarMonitor(
            frequencies=[f0],
            center=(0.0, y_mid, z_mon_top),
            size=(x_ap_span, ly, 0.0),
            eps_r=1.0,
        )
        dz_cav = float(st["L_cav"] / (2 * max(4, cpw // 4)))
        box_src = FluxMonitor(
            [f0], center=(0.0, y_mid, z_dip), size=(4 * dx, ly, 4 * dz_cav)
        )
        mons = [n2f_top, box_src]
        if label == "tuned":
            dft_xz = DFTMonitor(
                [f0],
                ["Ey"],
                center=(0.0, y_mid, 0.5 * (st["z_bot_end"] + st["z_top_end"])),
                size=(
                    2.0 * float(abs(x[14])),
                    0.0,
                    (st["z_top_end"] - st["z_bot_end"]) + 80.0e-9,
                ),
            )
            mons.append(dft_xz)

        solver.emsolve(Nt, source=[src], monitors=mons)
        pat_2d = hlp.farfield_2d_from_monitor(n2f_top, theta_rad, ly=ly, freq_idx=0)
        pat_norm = pat_2d / max(float(np.max(pat_2d)), 1e-30)
        pat_tmm = hlp.rcled_fabry_perot_pattern_2d(
            theta_rad,
            LAM_RCLED,
            st["L_cav"],
            z_dip,
            N_GAN,
            st["top_layers"],
            st["bot_layers"],
            n_ext=1.0,
            n_sub=N_GAN,
            pol="te",
        )
        pat_tmm_norm = pat_tmm / max(float(np.max(pat_tmm)), 1e-30)
        eta_top = float(n2f_top.get_flux()[0] / max(box_src.get_flux()[0], 1e-30))
        rcled_results[f"pat_wakis_{label}"] = pat_norm
        rcled_results[f"pat_tmm_{label}"] = pat_tmm_norm
        rcled_results[f"eta_top_{label}"] = eta_top

        if label == "tuned":
            ey_f, coords = dft_xz.get_field("Ey")
            rcled_results["rcled_x"] = coords[0]
            rcled_results["rcled_z"] = coords[2]
            rcled_results["rcled_ey"] = ey_f[0, :, 0, :]
            rcled_results["rcled_z_bot"] = float(st["z_bot_end"])
            rcled_results["rcled_z_cav_top"] = float(st["L_cav"])
            rcled_results["rcled_z_top"] = float(st["z_top_end"])

    # 3. Extraction efficiency into NA = 0.50 (|theta_ext| <= 30 deg) vs cavity detuning d in [0.88, 1.16]
    d_detune_grid = np.linspace(0.88, 1.16, 45)
    eta_na05_tmm = np.zeros_like(d_detune_grid)
    eta_tot_tmm = np.zeros_like(d_detune_grid)
    na_mask = np.abs(theta_deg) <= 30.0
    for idx_d, d_val in enumerate(d_detune_grid):
        st_d = _build_rcled_stack(d_opt=float(d_val), n_bot_pairs=4, n_top_pairs=2)
        p_d = hlp.rcled_fabry_perot_pattern_2d(
            theta_rad,
            LAM_RCLED,
            st_d["L_cav"],
            0.5 * st_d["L_cav"],
            N_GAN,
            st_d["top_layers"],
            st_d["bot_layers"],
            n_ext=1.0,
            n_sub=N_GAN,
            pol="te",
        )
        tot_p = float(np.trapezoid(p_d, theta_rad))
        na_p = float(np.trapezoid(p_d[na_mask], theta_rad[na_mask]))
        eta_tot_tmm[idx_d] = tot_p
        eta_na05_tmm[idx_d] = na_p / max(tot_p, 1e-30)

    # 4. Incoherent Quantum-Well Emitter Array (N_dip = 7 across w_QW = 0.72 um):
    #    Use mirror symmetry x_m <-> -x_m so 4 runs give all 7 dipole fields and far-field amplitudes!
    st_qw = _build_rcled_stack(d_opt=1.00, n_bot_pairs=4, n_top_pairs=2)
    grid_q = mesh_info["grid"]
    z_dip_q = float(mesh_info["z_dip"])
    z_mon_q = float(mesh_info["z_mon_top"])
    dz_pad_q = float(np.max(grid_q.dz))

    x_offsets_m = [0.0, 120.0e-9, 240.0e-9, 360.0e-9]
    ey_maps_7 = []
    ff_ephi_7 = []
    abs_th = np.abs(theta_rad)
    phi_th = np.where(theta_rad >= 0.0, 0.0, np.pi)
    x_ap_span = 2.0 * float(abs(x[9]))

    for x_target in x_offsets_m:
        ic_x = int(round(x_target / dx))
        x0 = ic_x * dx
        solver_q = SolverFIT3D(
            grid_q,
            bc_low=["cpml", "periodic", "cpml"],
            bc_high=["cpml", "periodic", "cpml"],
            n_pml=8,
            use_stl=False,
            verbose=0,
        )
        n_slabs_q = len(st_qw["slabs"])
        shapes_q = [
            hlp.SlabXZ(
                -1.5,
                1.5,
                float(grid_q.z[0]) - dz_pad_q,
                st_qw["z_bot_end"] + 0.6 * dz_pad_q,
                material=N_GAN**2,
                snap=False,
            )
        ] + [
            hlp.SlabXZ(
                -1.5,
                1.5,
                z0,
                z1 + (0.6 * dz_pad_q if idx_s < n_slabs_q - 1 else 0.0),
                material=n_val**2,
                snap=False,
            )
            for idx_s, (z0, z1, n_val) in enumerate(st_qw["slabs"])
        ]
        assign_materials(solver_q, shapes_q, f0, averaging="kottke")

        pulse_q = GaussianSource(f0, 0.28 * f0, cutoff=3.6)
        Nt_q = int(np.ceil((pulse_q.end_time + 10.0 * LAM_RCLED / c0) / solver_q.dt))
        src_q = CurrentSource(
            pulse_q,
            component="y",
            center=(x0, y_mid, z_dip_q),
            size=(0.0, 1.0, 0.0),
            amplitude=1.0,
        )
        n2f_q = Near2FarMonitor(
            frequencies=[f0],
            center=(0.0, y_mid, z_mon_q),
            size=(x_ap_span, ly, 0.0),
            eps_r=1.0,
        )
        dft_q = DFTMonitor(
            [f0],
            ["Ey"],
            center=(0.0, y_mid, 0.5 * (st_qw["z_bot_end"] + z_mon_q)),
            size=(
                2.0 * float(abs(x[14])),
                0.0,
                (z_mon_q - st_qw["z_bot_end"]) + 60.0e-9,
            ),
        )
        solver_q.emsolve(Nt_q, source=[src_q], monitors=[n2f_q, dft_q])
        ey_m, coords_q = dft_q.get_field("Ey")
        ey_2d = ey_m[0, :, 0, :]
        ff_m = n2f_q.get_farfield(abs_th, phi_th)["Ephi"][0]

        ey_maps_7.append(ey_2d)
        ff_ephi_7.append(ff_m)
        if ic_x > 0:
            # Mirror partner at -x0: flip x axis!
            ey_maps_7.append(ey_2d[::-1, :])
            ff_ephi_7.append(ff_m[::-1])

    ey_maps_7 = np.asarray(ey_maps_7)  # shape (7, Nx_sub, Nz_sub)
    ff_ephi_7 = np.asarray(ff_ephi_7)  # shape (7, N_theta)

    # Coherent phased-array sum vs deterministic incoherent power sum
    i_map_coh = np.abs(np.sum(ey_maps_7, axis=0)) ** 2 / (len(ey_maps_7) ** 2)
    i_map_incoh = np.mean(np.abs(ey_maps_7) ** 2, axis=0)

    pat_coh = np.abs(np.sum(ff_ephi_7, axis=0)) ** 2
    pat_coh_norm = pat_coh / max(float(np.max(pat_coh)), 1e-30)
    pat_incoh = np.mean(np.abs(ff_ephi_7) ** 2, axis=0)
    pat_incoh_norm = pat_incoh / max(float(np.max(pat_incoh)), 1e-30)

    # Random-phase Monte Carlo convergence study over N_real in [1, 2, 4, 8, 16, 32, 64, 128]
    rng = np.random.default_rng(42)
    n_real_list = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=int)
    n_trials = 40
    mc_rms_err = np.zeros(len(n_real_list))
    mc_sample_curves = {}
    for idx_r, n_r in enumerate(n_real_list):
        errs = []
        for tr in range(n_trials):
            phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_r, len(ff_ephi_7), 1))
            e_tot_r = np.sum(ff_ephi_7[None, :, :] * np.exp(1j * phases), axis=1)
            p_mc = np.mean(np.abs(e_tot_r) ** 2, axis=0)
            p_mc_norm = p_mc / max(float(np.max(p_mc)), 1e-30)
            rms = float(
                np.sqrt(np.mean((p_mc_norm - pat_incoh_norm) ** 2))
                / np.sqrt(np.mean(pat_incoh_norm**2))
            )
            errs.append(rms)
            if tr == 0 and n_r in (1, 4, 16, 64):
                mc_sample_curves[f"mc_pat_{n_r}"] = p_mc_norm
        mc_rms_err[idx_r] = float(np.mean(errs))

    # Convergence with number of spatial dipoles N_dip in {1, 3, 5, 7}
    ndip_list = np.array([1, 3, 5, 7], dtype=int)
    ndip_err = np.zeros(len(ndip_list))
    for i_nd, nd in enumerate(ndip_list):
        sub_p = np.mean(np.abs(ff_ephi_7[:nd]) ** 2, axis=0)
        sub_p_norm = sub_p / max(float(np.max(sub_p)), 1e-30)
        ndip_err[i_nd] = float(
            np.sqrt(np.mean((sub_p_norm - pat_incoh_norm) ** 2))
            / np.sqrt(np.mean(pat_incoh_norm**2))
        )

    out = {
        "name": cfg["name"],
        "wl_tmm_nm": wl_tmm * 1e9,
        "r_bot_spec": r_bot_spec,
        "r_top_spec": r_top_spec,
        "r_full_spec": r_full_spec,
        "theta_deg": theta_deg,
        "theta_rad": theta_rad,
        "d_detune_grid": d_detune_grid,
        "eta_na05_tmm": eta_na05_tmm,
        "eta_tot_tmm": eta_tot_tmm,
        "qw_x": coords_q[0],
        "qw_z": coords_q[2],
        "i_map_coh": i_map_coh,
        "i_map_incoh": i_map_incoh,
        "pat_coh_norm": pat_coh_norm,
        "pat_incoh_norm": pat_incoh_norm,
        "n_real_list": n_real_list,
        "mc_rms_err": mc_rms_err,
        "ndip_list": ndip_list,
        "ndip_err": ndip_err,
    }
    out.update(rcled_results)
    out.update(mc_sample_curves)
    return _attach_mesh_meta(out, mesh_info)


def _run_silver_single_height(
    dz_ag: float, cpw_bg: int, h_ag_val: float
) -> tuple[float, float, float]:
    """Run a single Drude silver Purcell extraction at height ``h_ag_val`` and step ``dz_ag``."""
    eps_m, _, delta_e = hlp.drude_silver_permittivity(
        LAM_AG, AG_EPS_INF, AG_WP_EV, AG_GAMMA_EV
    )
    x_ag, y_ag, z_ag_arr, dx_ag = _build_silver_graded_axes(
        dz_ag, cpw_bg, h_max_fine=72.0e-9, x_max=420.0e-9
    )
    cells_skin = hlp.count_cells_in_interval(z_ag_arr, -float(delta_e), 0.0)
    grid_ag = GridFIT3D(x=x_ag, y=y_ag, z=z_ag_arr, verbose=0)
    solver_ag = SolverFIT3D(
        grid_ag,
        bc_low=["cpml", "periodic", "pec"],
        bc_high=["cpml", "periodic", "cpml"],
        n_pml=8,
        cfln=0.48,
        use_stl=False,
        verbose=0,
    )
    ag_mat = DispersiveMaterial(
        AG_EPS_INF,
        [Drude(AG_WP_EV * hlp.EV_TO_RAD_S, AG_GAMMA_EV * hlp.EV_TO_RAD_S)],
    )
    z_ag_bot = float(z_ag_arr[0])
    solver_ag.add_dispersive_material(
        ag_mat, center=(0.0, 0.0, 0.5 * z_ag_bot), size=(2.0, 2.0, abs(z_ag_bot))
    )
    f0_ag = c0 / LAM_AG
    pulse_ag = GaussianSource(f0_ag, 0.42 * f0_ag, cutoff=3.5)
    Nt_ag = int(np.ceil((pulse_ag.end_time + 1.8 * LAM_AG / c0) / solver_ag.dt))
    Ny = 3
    src_ag = CurrentSource(
        pulse_ag,
        component="y",
        center=(0.0, 0.5 * Ny * dx_ag, h_ag_val),
        size=(0.0, 1.0, 0.0),
        amplitude=1.0,
    )
    box_ag = FluxMonitor(
        [f0_ag],
        center=(0.0, 0.5 * Ny * dx_ag, h_ag_val),
        size=(4 * dx_ag, dx_ag, 4 * dz_ag),
    )
    solver_ag.emsolve(Nt_ag, source=[src_ag], monitors=[box_ag])
    s_fag = pulse_ag.fourier_transform(f0_ag)
    f_ag_num = float((box_ag.get_flux()[0] / dx_ag) / _p0_2d(f0_ag, s_fag, "y"))
    kh_ag_val = (2.0 * np.pi / LAM_AG) * h_ag_val
    f_ag_cps = float(hlp.cps_sommerfeld_2d(kh_ag_val, eps_m, pol="y")["F_tot"][0])
    return f_ag_num, f_ag_cps, cells_skin


def _run_convergence_task(cfg: dict) -> dict:
    """Run Tasks 4-10: Resolution and Kottke-vs-staircase subpixel convergence study."""
    mesh_info = build_task_mesh(cfg)
    cpw = int(cfg["cpw"])
    avg = str(cfg["averaging"])
    Ny = 3

    # 1. PEC mirror Purcell factor at h = 0.375 * lambda_0 (kh = 2.35619, >= 6 cells across h)
    #    Fixed physical domain across all cpw levels so PML reflections are identical
    dx_pec = LAM_PEC / cpw
    y_pec = np.arange(Ny + 1) * dx_pec
    y_mid_pec = 0.5 * Ny * dx_pec
    n_x_pec = int(round(1.00e-6 / dx_pec))
    n_z_pec = int(round(1.75e-6 / dx_pec))
    x_pec = np.arange(-n_x_pec, n_x_pec + 1) * dx_pec
    z_pec = np.arange(0, n_z_pec + 1) * dx_pec
    f0_pec = c0 / LAM_PEC
    ic_h = max(4, int(round(0.375 * cpw)))
    h_pec = ic_h * dx_pec
    kh_pec = (2.0 * np.pi / LAM_PEC) * h_pec

    grid_p = GridFIT3D(x=x_pec, y=y_pec, z=z_pec, verbose=0)
    solver_p = SolverFIT3D(
        grid_p,
        bc_low=["cpml", "periodic", "pec"],
        bc_high=["cpml", "periodic", "cpml"],
        n_pml=8,
        use_stl=False,
        verbose=0,
    )
    pulse_p = GaussianSource(f0_pec, 0.35 * f0_pec, cutoff=3.8)
    Nt_p = int(np.ceil((pulse_p.end_time + 2.4 * LAM_PEC / c0) / solver_p.dt))
    src_p = CurrentSource(
        pulse_p,
        component="y",
        center=(0.0, y_mid_pec, h_pec),
        size=(0.0, 1.0, 0.0),
        amplitude=1.0,
    )
    dft_p = DFTMonitor(
        [f0_pec], ["Ey"], center=(0.0, y_mid_pec, h_pec), size=(0.0, 0.0, 0.0)
    )
    solver_p.emsolve(Nt_p, source=[src_p], monitors=[dft_p])
    s_fp = pulse_p.fourier_transform(f0_pec)
    e_p = complex(dft_p.get_field("Ey")[0].squeeze())
    f_pec_num = float(-np.real(e_p * np.conj(s_fp)) / _p0_2d(f0_pec, s_fp, "y"))
    f_pec_ana = float(hlp.pec_mirror_purcell_2d(kh_pec, "y"))

    # 2. Off-grid dielectric DBR mirror (exercising Kottke vs staircase on SlabXZ interfaces)
    #    Dipole at exact grid node z = 0 above a 2-pair GaN/SiO2 Bragg mirror at z_surf = -0.172 * LAM_RCLED
    #    All levels have cpw >= 16 (>= 4.0 cells across the 47.07 nm GaN quarter-wave layer)
    grid_d = mesh_info["grid"]
    x_dbr = mesh_info["x"]
    y_dbr = mesh_info["y"]
    intervals_dbr = mesh_info["intervals_dbr"]
    dz_dbr = LAM_RCLED / (cpw * N_GAN)
    dx_dbr = float(x_dbr[1] - x_dbr[0])
    y_mid_dbr = 0.5 * Ny * dx_dbr

    solver_d = SolverFIT3D(
        grid_d,
        bc_low=["cpml", "periodic", "cpml"],
        bc_high=["cpml", "periodic", "cpml"],
        n_pml=8,
        use_stl=False,
        verbose=0,
    )
    f0_dbr = c0 / LAM_RCLED
    shapes_dbr = [
        hlp.SlabXZ(
            -1.0,
            1.0,
            z0,
            z1 + (0.8 * dz_dbr if idx_d < len(intervals_dbr) - 1 else 0.0),
            material=eps_val,
            snap=False,
        )
        for idx_d, (z0, z1, eps_val) in enumerate(intervals_dbr)
    ]
    assign_materials(solver_d, shapes_dbr, f0_dbr, averaging=avg)

    # Save 1D z-permittivity profiles (excluding boundary padding node -1) for Fig. 1c
    ieps_x = np.asarray(solver_d.ieps[grid_d.Nx // 2, 1, :-1, "x"], dtype=float)
    ieps_z = np.asarray(solver_d.ieps[grid_d.Nx // 2, 1, :-1, "z"], dtype=float)
    eps_x_prof = 1.0 / (eps0 * np.maximum(ieps_x, 1e-30))
    eps_z_prof = 1.0 / (eps0 * np.maximum(ieps_z, 1e-30))
    z_dbr_prof_nm = solver_d.z[: len(eps_x_prof)] * 1e9

    f_scan_dbr = np.linspace(0.92 * f0_dbr, 1.08 * f0_dbr, 25)
    pulse_d = GaussianSource(f0_dbr, 0.35 * f0_dbr, cutoff=3.8)
    Nt_d = int(np.ceil((pulse_d.end_time + 3.2 * LAM_RCLED / c0) / solver_d.dt))
    src_d = CurrentSource(
        pulse_d,
        component="y",
        center=(0.0, y_mid_dbr, 0.0),
        size=(0.0, 1.0, 0.0),
        amplitude=1.0,
    )
    dft_d = DFTMonitor(
        f_scan_dbr, ["Ey"], center=(0.0, y_mid_dbr, 0.0), size=(0.0, 0.0, 0.0)
    )
    solver_d.emsolve(Nt_d, source=[src_d], monitors=[dft_d])
    s_fd = pulse_d.fourier_transform(f_scan_dbr)
    e_d = dft_d.get_field("Ey")[0].reshape(-1)
    f_dbr_spec = -np.real(e_d * np.conj(s_fd)) / _p0_2d(f_scan_dbr, s_fd, "y", n_bg=1.0)
    f_dbr_at_f0 = float(f_dbr_spec[len(f_scan_dbr) // 2])

    # 3. Drude silver skin-depth resolution study at fixed h = 48.0 nm (ih = cpw // 2 >= 8 cells)
    #    dz_ag = 96.0 nm / cpw gives 4.07, 5.09, 7.12, 10.17 cells per delta_E in --quick
    dz_ag = 96.0e-9 / cpw
    h_ag_val = 48.0e-9
    cells_skin = (
        float(mesh_info["cells_per_skin_depth"])
        if mesh_info["cells_per_skin_depth"] is not None
        else np.nan
    )
    f_ag_num = np.nan
    f_ag_cps = np.nan
    demo_dz_nm = np.zeros(0)
    demo_cells_skin = np.zeros(0)
    demo_f_num = np.zeros(0)
    demo_rel_err = np.zeros(0)

    if avg == "kottke":
        f_ag_num, f_ag_cps, cells_skin = _run_silver_single_height(
            dz_ag, max(cpw, 20), h_ag_val
        )
        # On the reference task, also run the clearly labelled under-resolved demonstration pair
        # (dz_ag = 24.0 nm and 16.0 nm -> 1.02 and 1.53 cells per skin depth delta_E)
        if str(cfg.get("level", "")) == "reference":
            d_dz = [24.0e-9, 16.0e-9]
            d_cs, d_fn, d_err = [], [], []
            for dz_d in d_dz:
                fn_d, fc_d, cs_d = _run_silver_single_height(dz_d, 20, h_ag_val)
                d_cs.append(cs_d)
                d_fn.append(fn_d)
                d_err.append(abs(fn_d - fc_d) / fc_d)
            demo_dz_nm = np.asarray([z_v * 1e9 for z_v in d_dz], dtype=float)
            demo_cells_skin = np.asarray(d_cs, dtype=float)
            demo_f_num = np.asarray(d_fn, dtype=float)
            demo_rel_err = np.asarray(d_err, dtype=float)

    payload = {
        "name": cfg["name"],
        "level": cfg["level"],
        "cpw": cpw,
        "averaging": avg,
        "kh_pec": kh_pec,
        "f_pec_num": f_pec_num,
        "f_pec_ana": f_pec_ana,
        "pec_rel_err": abs(f_pec_num - f_pec_ana) / f_pec_ana,
        "f_scan_dbr": f_scan_dbr,
        "f_dbr_spec": f_dbr_spec,
        "f_dbr_at_f0": f_dbr_at_f0,
        "z_dbr_nm": z_dbr_prof_nm,
        "eps_x_prof": eps_x_prof,
        "eps_z_prof": eps_z_prof,
        "f_ag_num": f_ag_num,
        "f_ag_cps": f_ag_cps,
        "ag_rel_err": abs(f_ag_num - f_ag_cps) / f_ag_cps if not np.isnan(f_ag_num) else np.nan,
        "demo_ag_dz_nm": demo_dz_nm,
        "demo_ag_cells_skin": demo_cells_skin,
        "demo_ag_f_num": demo_f_num,
        "demo_ag_rel_err": demo_rel_err,
        "grid_shape": np.array([grid_d.Nx, grid_d.Ny, grid_d.Nz], dtype=int),
    }
    return _attach_mesh_meta(payload, mesh_info)


def run_single_task(task_cfg: dict, out_dir: Path = DATA_DIR) -> dict:
    """Execute one study task and write ``task_I.npz`` and ``task_I.json`` to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = int(task_cfg["index"])
    kind = task_cfg["kind"]
    t0 = time.perf_counter()

    if kind == "pec_mirror":
        payload = _run_pec_drexhage_task(task_cfg)
    elif kind == "silver_mirror":
        payload = _run_silver_drude_task(task_cfg)
    elif kind == "pec_cavity":
        payload = _run_planar_cavity_pec_task(task_cfg)
    elif kind == "rcled":
        payload = _run_rcled_and_incoherent_task(task_cfg)
    elif kind == "convergence":
        payload = _run_convergence_task(task_cfg)
    else:
        raise ValueError(f"Unknown task kind {kind!r}")

    npz_path = out_dir / f"task_{idx}.npz"
    np.savez_compressed(npz_path, **payload)

    elapsed = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    skin_nm_val = (
        None
        if np.isnan(payload["skin_depth_nm"])
        else round(float(payload["skin_depth_nm"]), 3)
    )
    cells_skin_val = (
        None
        if np.isnan(payload["cells_per_skin_depth"])
        else round(float(payload["cells_per_skin_depth"]), 3)
    )
    meta = {
        "task_index": idx,
        "name": task_cfg["name"],
        "kind": kind,
        "averaging": task_cfg["averaging"],
        "cpw": int(task_cfg["cpw"]),
        "dx_background_nm": round(float(payload["dx_background_nm"]), 3),
        "dx_min_nm": round(float(payload["dx_min_nm"]), 3),
        "min_feature_nm": round(float(payload["min_feature_nm"]), 3),
        "cells_across_feature": round(float(payload["cells_across_feature"]), 3),
        "skin_depth_nm": skin_nm_val,
        "cells_per_skin_depth": cells_skin_val,
        "n_cells": int(payload["n_cells"]),
        "n_steps": int(payload["n_steps"]),
        "elapsed_s": round(elapsed, 3),
        "peak_rss_mb": round(rss_mb, 2),
        "npz_bytes": int(npz_path.stat().st_size),
    }
    (out_dir / f"task_{idx}.json").write_text(json.dumps(meta, indent=2))
    skin_str = (
        f" skin={meta['cells_per_skin_depth']:.2f}c/{meta['skin_depth_nm']:.1f}nm"
        if meta["cells_per_skin_depth"] is not None
        else ""
    )
    print(
        f"[task_{idx} {task_cfg['name']}] cpw={meta['cpw']} "
        f"dx_bg={meta['dx_background_nm']:.2f}nm dx_min={meta['dx_min_nm']:.2f}nm "
        f"feat={meta['cells_across_feature']:.2f}c/{meta['min_feature_nm']:.1f}nm"
        f"{skin_str} cells={meta['n_cells']} steps={meta['n_steps']} "
        f"time={elapsed:.2f}s rss={rss_mb:.1f}MB"
    )
    return meta


# %% [markdown]
# ## 5. Generating the setup schematics and 14-figure diagnostic suite
#
# When invoked with `--plot` (or at the end of `--quick` / `--full`), `generate_all_figures`
# constructs the solver setups for schematic inspection, reads `data/task_*.npz`, and renders
# the two-figure setup schematics (`00a_setup_geometry*.png`, `00b_setup_mesh_and_excitation*.png`,
# and `00_setup_table*.md`) followed by the 14 diagnostic figures into `figures/`, all styled
# with `wakis.figstyle` and `wakis.schematic`:
#
# 0. `00a_setup_geometry*.png`, `00b_setup_mesh_and_excitation*.png`, `00_setup_table*.md` — Two-figure setup layouts and setup tables generated directly from the `SolverFIT3D`, `CurrentSource`, `FluxMonitor`, `DFTMonitor`, and `Near2FarMonitor` objects via `wakis.schematic.save_setup_figures` for Part A (`00a`/`00b`, PEC mirror), Part B (`_silver`, dispersive `Drude` silver mirror with $\delta_E = 24.41\;\text{nm}$), Part C (`_cavity`, planar PEC microcavity), and Part D (`_rcled`, blue $\text{InGaN}$ DBR micro-RCLED).
# 1. `01_geometry_and_mesh_overview.png` — Three-panel mesh & discretization view: (a) `wakis.schematic.draw_mesh` view of the locally graded Drude silver mesh ($\Delta z_{\text{Ag}} = 2.50\;\text{nm}$, $9.76\;\text{cells}/\delta_E$, $\Delta x_{\text{core}} = 5.0\;\text{nm}$) with a zoom inset on the $\text{Ag/air}$ interface and geometry overlays; (b) vertical grid-step profile $\Delta z(z)$ across the Drude silver skin-depth refinement zone and the interface-snapped RCLED DBR stack; (c) off-grid DBR permittivity profile $\varepsilon_r(z)$ comparing Kottke tensorial subpixel smoothing vs. staircasing.
# 2. `02_pec_mirror_drexhage_purcell.png` — (a) Wakis 2D FIT Purcell factor $F_P(kh)$ (`FluxMonitor` box & local dipole work) vs. exact 2D Bessel analytical curves for $\text{TE}_\parallel$ ($J_y$), $\text{TM}_\parallel$ ($J_x$), and $\text{TM}_\perp$ ($J_z$); (b) 2D cylindrical Bessel decay $\mathcal{O}((kh)^{-1/2})$ vs. 3D Drexhage spherical decay $\mathcal{O}((kh)^{-1})$.
# 3. `03_pec_mirror_angular_patterns.png` — Far-field angular radiation patterns $\frac{1}{P_0'}\frac{\mathrm{d}P'}{\mathrm{d}\theta}$ from `Near2FarMonitor(..., ground_plane="-z")` overlaid with exact 2D analytical curves at $h/\lambda_0 \in \{0.25, 0.50, 0.75\}$ for $J_y$, $J_x$, and $J_z$.
# 4. `04_pec_mirror_field_snapshots.png` — 2D $(x, z)$ steady-state wave-front maps $\operatorname{Re}(E_y)$ comparing constructive ($h = 0.24\lambda_0$) and destructive ($h = 0.48\lambda_0$) image-dipole interference, with `wakis.schematic.overlay_geometry` labelling the PEC mirror, CPML boundaries, `Near2FarMonitor` Huygens box, and dipole source.
# 5. `05_dipole_mirror_interference.gif` — Animated time-domain movie GIF of a line dipole pulse reflecting off the PEC mirror and forming far-field interference lobes, with labelled geometry and boundary overlays on every frame.
# 6. `06_drude_silver_drexhage_cps.png` — Line dipole above a dispersive Drude silver mirror at $\lambda_0 = 600\;\text{nm}$ ($\delta_E = 24.41\;\text{nm}$): Wakis total power $F_{\text{tot}}(kh)$ vs. exact 2D CPS Sommerfeld integrals ($F_{\text{tot}}, F_{\text{rad}}, F_{\text{nonrad}}$) and radiative quantum efficiency $\eta_{\text{rad}}(kh)$.
# 7. `07_sommerfeld_spp_kspace_spectrum.png` — (a) Wavenumber-resolved CPS power dissipation density $\mathrm{d}F/\mathrm{d}u$ vs. $u = k_x / k_0$, highlighting the light line ($u = 1$) and the TM surface-plasmon-polariton pole at $u_{\text{SPP}} = 1.0338$; (b) Wakis vertical electric-field decay $|E(z)|$ inside the silver mirror vs. the analytical skin depth $\delta_E = 24.41\;\text{nm}$.
# 8. `08_planar_cavity_purcell_2d_vs_3d.png` — Planar PEC microcavity Purcell factor $F_P(d)$ ($n = 2.4$, Abram *et al.* 1998): (a) Wakis 2D FIT multi-band extraction vs. exact 2D Van Hove curves for $J_y, J_x, J_z$; (b) direct comparison of 2D $1/\sqrt{d^2 - d_m^2}$ Van Hove singularities against the 3D step-function DOS of Abram *et al.* (IEEE JQE 1998, Eq. 7, Fig. 2).
# 9. `09_cavity_cutoff_ringing_diagnostic.png` — Numerical pathology & cure at the $d = 0.50$ waveguide cutoff: (a) trapped $v_g = 0$ time-domain oscillation $E_y(\mathbf{r}_0, t)$; (b) broadband cutoff-crossing DFT ringing vs. band-limited spectral windowing.
# 10. `10_rcled_dbr_reflectance_and_cavity_field.png` — Blue InGaN micro-RCLED ($\lambda_0 = 450\;\text{nm}$, Lee *et al.* 2024 Fig. 1): (a) TMM reflectance spectra of the bottom $\text{NP-GaN/GaN}$ DBR ($n = 1.70/2.39$), top $\text{Ta}_2\text{O}_5/\text{SiO}_2$ DBR ($n = 2.15/1.46$), and full microcavity; (b) 2D Wakis intracavity standing-wave field $|E_y(x, z)|^2$ with `wakis.schematic.overlay_geometry` labelling the DBR mirrors, GaN cavity, QW dipole, and top extraction aperture.
# 11. `11_rcled_angular_extraction_beaming.png` — RCLED angular extraction & detuning (Benisty *et al.* 1998): (a) Wakis 2D far-field patterns into air vs. TMM Fabry–Pérot Airy curves for tuned ($d = 1.00$, normal beam) and detuned ($d = 1.055$, rabbit-ear sidelobes) cavities; (b) fraction of extracted power captured within $\text{NA} = 0.50$ ($|\theta_{\text{ext}}| \le 30^\circ$) vs. cavity optical thickness $d$.
# 12. `12_incoherent_quantum_well_averaging.png` — Extended $N_{\text{dip}} = 7$ quantum-well emitter layer: (a) artificial multi-slit interference fringes of a coherent in-phase dipole array vs. the smooth deterministic incoherent power sum; (b) lateral near-field intensity profiles in the superstrate comparing coherent phased-array beaming vs. incoherent power superposition.
# 13. `13_incoherent_monte_carlo_convergence.png` — Statistical convergence of random-phase Monte Carlo realizations ($N_{\text{real}} \in [1, 128]$) and spatial emitter sampling ($N_{\text{dip}} \in [1, 7]$) toward the deterministic incoherent benchmark, verifying the $1/\sqrt{N_{\text{real}}}$ law.
# 14. `14_mesh_and_subpixel_convergence.png` — Multi-resolution convergence across `coarse`, `medium`, `fine`, and `reference`: (a) PEC mirror & off-grid DBR resonator error comparing **Kottke subpixel smoothing** vs. **staircasing**; (b) Drude silver relative error vs. **cells per skin depth** ($\delta_E / \Delta z$).


# %%
def _build_schematic_setups(cpw_pec: int = 25, cpw_ag: int = 24, dz_ag_nm: float = 2.5, cpw_rcled: int = 20) -> dict:
    """Build lightweight SolverFIT3D + source/monitor/structure bundles for schematics and overlays."""
    # 1. Part A: PEC mirror setup (lambda_0 = 1.0 um)
    m_pec = build_task_mesh({"kind": "pec_mirror", "cpw": cpw_pec, "quick": False})
    grid_pec = m_pec["grid"]
    dx_pec = float(grid_pec.dx[0])
    y_mid_pec = 1.5 * dx_pec
    solver_pec = SolverFIT3D(
        grid_pec,
        bc_low=["cpml", "periodic", "pec"],
        bc_high=["cpml", "periodic", "cpml"],
        n_pml=8,
        use_stl=False,
        verbose=0,
    )
    f0_pec = c0 / LAM_PEC
    pulse_pec = GaussianSource(f0_pec, 0.30 * f0_pec, cutoff=3.8)
    h_pec_025 = max(4, int(round(0.25 * LAM_PEC / dx_pec))) * dx_pec
    h_pec_050 = max(4, int(round(0.50 * LAM_PEC / dx_pec))) * dx_pec
    src_pec_025 = CurrentSource(
        pulse_pec, component="y", center=(0.0, y_mid_pec, h_pec_025), size=(0.0, 1.0, 0.0), amplitude=1.0
    )
    src_pec_050 = CurrentSource(
        pulse_pec, component="y", center=(0.0, y_mid_pec, h_pec_050), size=(0.0, 1.0, 0.0), amplitude=1.0
    )
    z_top_box = max(h_pec_025 + 4 * dx_pec, 10 * dx_pec)
    n2f_pec = hlp.set_monitor_bounds(
        Near2FarMonitor(
            frequencies=[f0_pec],
            center=(0.0, y_mid_pec, 0.5 * z_top_box),
            size=(14 * dx_pec, dx_pec, z_top_box),
            ground_plane="-z",
        ),
        label="Near2Far box (ground_plane='-z')",
    )

    # 2. Part B: Drude silver mirror setup (lambda_0 = 600 nm, skin depth delta_E = 24.41 nm)
    m_ag = build_task_mesh({"kind": "silver_mirror", "cpw": cpw_ag, "dz_ag_nm": dz_ag_nm, "quick": False})
    grid_ag = m_ag["grid"]
    dx_core_ag = float(m_ag["dx_core"])
    dz_ag = dz_ag_nm * 1e-9
    y_mid_ag = 1.5 * dx_core_ag
    z_ag_bot = float(grid_ag.z[0])
    solver_ag = SolverFIT3D(
        grid_ag,
        bc_low=["cpml", "periodic", "pec"],
        bc_high=["cpml", "periodic", "cpml"],
        n_pml=8,
        cfln=0.48,
        use_stl=False,
        verbose=0,
    )
    ag_mat = DispersiveMaterial(
        AG_EPS_INF,
        [Drude(AG_WP_EV * hlp.EV_TO_RAD_S, AG_GAMMA_EV * hlp.EV_TO_RAD_S)],
    )
    solver_ag.add_dispersive_material(
        ag_mat, center=(0.0, 0.0, 0.5 * z_ag_bot), size=(2.0, 2.0, abs(z_ag_bot))
    )
    ag_struct = hlp.SlabXZ(
        float(grid_ag.x[0]),
        float(grid_ag.x[-1]),
        z_ag_bot,
        0.0,
        material=ag_mat,
        name="Drude Ag mirror (δE = 24.4 nm)",
    )
    f0_ag = c0 / LAM_AG
    pulse_ag = GaussianSource(f0_ag, 0.42 * f0_ag, cutoff=3.5)
    h_ag = max(5, int(round(48.0e-9 / dz_ag))) * dz_ag
    src_ag = CurrentSource(
        pulse_ag, component="y", center=(0.0, y_mid_ag, h_ag), size=(0.0, 1.0, 0.0), amplitude=1.0
    )
    box_ag = hlp.set_monitor_bounds(
        FluxMonitor([f0_ag], center=(0.0, y_mid_ag, h_ag), size=(4 * dx_core_ag, dx_core_ag, 4 * dz_ag)),
        label="Dipole FluxMonitor box (P_tot)",
    )
    x_plane_span = 2.0 * float(abs(grid_ag.x[8]))
    plane_ag = hlp.set_monitor_bounds(
        FluxMonitor([f0_ag], center=(0.0, y_mid_ag, dz_ag), size=(x_plane_span, dx_core_ag, 0.0)),
        label="Surface FluxMonitor (P_metal)",
    )

    # 3. Part C: Planar PEC microcavity setup (L_cav = 1.0 um, n_cav = 2.4)
    m_cav = build_task_mesh({"kind": "pec_cavity", "cpw": cpw_rcled, "quick": False})
    grid_cav = m_cav["grid"]
    dz_cav_pec = float(grid_cav.dz[0])
    y_mid_cav = 1.5 * dz_cav_pec
    z_mid_cav = 0.5e-6
    solver_cav = SolverFIT3D(
        grid_cav,
        bc_low=["cpml", "periodic", "pec"],
        bc_high=["cpml", "periodic", "pec"],
        bg=[N_CAV_ABRAM**2, 1.0],
        n_pml=8,
        use_stl=False,
        verbose=0,
    )
    f0_cav = 0.95 * c0 / (N_CAV_ABRAM * 1.0e-6)
    pulse_cav = GaussianSource(f0_cav, 0.20 * f0_cav, cutoff=4.2)
    src_cav = CurrentSource(
        pulse_cav,
        component="y",
        center=(0.0, y_mid_cav, z_mid_cav),
        size=(0.0, 1.0, 0.0),
        amplitude=1.0,
    )
    f_pts_cav = np.linspace(0.60, 1.36, 15) * c0 / (N_CAV_ABRAM * 1.0e-6)
    dft_cav = DFTMonitor(
        f_pts_cav,
        ["Ey"],
        center=(0.0, y_mid_cav, z_mid_cav),
        size=(0.0, 0.0, 0.0),
    )

    # 4. Part D: Blue InGaN DBR micro-RCLED setup (lambda_0 = 450 nm)
    m_rc = build_task_mesh({"kind": "rcled", "cpw": cpw_rcled, "quick": False})
    grid_rc = m_rc["grid"]
    dx_rc = float(grid_rc.dx[0])
    y_mid_rc = 1.5 * dx_rc
    z_dip_rc = float(m_rc["z_dip"])
    z_mon_top_rc = float(m_rc["z_mon_top"])
    st_rc = _build_rcled_stack(d_opt=1.00, n_bot_pairs=4, n_top_pairs=2)
    solver_rc = SolverFIT3D(
        grid_rc,
        bc_low=["cpml", "periodic", "cpml"],
        bc_high=["cpml", "periodic", "cpml"],
        n_pml=8,
        use_stl=False,
        verbose=0,
    )
    f0_rc = c0 / LAM_RCLED
    dz_pad_rc = float(np.max(grid_rc.dz))
    n_slabs_rc = len(st_rc["slabs"])
    shapes_rc = [
        hlp.SlabXZ(
            -1.5, 1.5, float(grid_rc.z[0]) - dz_pad_rc, st_rc["z_bot_end"] + 0.6 * dz_pad_rc, material=N_GAN**2, snap=False
        )
    ] + [
        hlp.SlabXZ(
            -1.5, 1.5, z0, z1 + (0.6 * dz_pad_rc if idx_s < n_slabs_rc - 1 else 0.0), material=n_val**2, snap=False
        )
        for idx_s, (z0, z1, n_val) in enumerate(st_rc["slabs"])
    ]
    assign_materials(solver_rc, shapes_rc, f0_rc, averaging="kottke")

    x_rc_lo, x_rc_hi = float(grid_rc.x[0]), float(grid_rc.x[-1])
    rc_macro_structs = [
        hlp.SlabXZ(
            x_rc_lo,
            x_rc_hi,
            st_rc["z_bot_end"],
            0.0,
            material=N_NPGAN**2,
            name="Bottom NP-GaN/GaN DBR",
        ),
        hlp.SlabXZ(
            x_rc_lo,
            x_rc_hi,
            0.0,
            st_rc["L_cav"],
            material=N_GAN**2,
            name="1λ GaN cavity",
        ),
        hlp.SlabXZ(
            x_rc_lo,
            x_rc_hi,
            st_rc["L_cav"],
            st_rc["z_top_end"],
            material=N_TA2O5**2,
            name="Top Ta₂O₅/SiO₂ DBR",
        ),
    ]
    rc_macro_names = {
        "region_0": "Bottom NP-GaN/GaN DBR",
        "region_1": "1λ GaN cavity",
        "region_2": "Top Ta₂O₅/SiO₂ DBR",
    }
    pulse_rc = GaussianSource(f0_rc, 0.28 * f0_rc, cutoff=3.8)
    src_rc = CurrentSource(
        pulse_rc, component="y", center=(0.0, y_mid_rc, z_dip_rc), size=(0.0, 1.0, 0.0), amplitude=1.0
    )
    x_ap_span = 2.0 * float(abs(grid_rc.x[9]))
    n2f_rc = hlp.set_monitor_bounds(
        FluxMonitor([f0_rc], center=(0.0, y_mid_rc, z_mon_top_rc), size=(x_ap_span, dx_rc, 0.0)),
        label="Top Near2Far aperture",
    )
    dz_cav = float(st_rc["L_cav"] / (2 * max(4, cpw_rcled // 4)))
    box_rc = hlp.set_monitor_bounds(
        FluxMonitor([f0_rc], center=(0.0, y_mid_rc, z_dip_rc), size=(4 * dx_rc, dx_rc, 4 * dz_cav)),
        label="Dipole FluxMonitor box",
    )

    return {
        "pec": {
            "solver": solver_pec,
            "src_025": src_pec_025,
            "src_050": src_pec_050,
            "n2f": n2f_pec,
            "h_025": h_pec_025,
            "h_050": h_pec_050,
            "dx": dx_pec,
            "y_mid": y_mid_pec,
            "z_top_box": z_top_box,
        },
        "ag": {
            "solver": solver_ag,
            "structures": [ag_struct],
            "material_names": {"region_0": "Drude Ag mirror (δE = 24.4 nm)"},
            "src": src_ag,
            "box": box_ag,
            "plane": plane_ag,
            "mesh_info": m_ag,
            "dx_core": dx_core_ag,
            "dz_ag": dz_ag,
            "y_mid": y_mid_ag,
            "h_ag": h_ag,
        },
        "cavity": {
            "solver": solver_cav,
            "src": src_cav,
            "dft": dft_cav,
            "mesh_info": m_cav,
            "y_mid": y_mid_cav,
            "z_mid": z_mid_cav,
            "f0": f0_cav,
        },
        "rcled": {
            "solver": solver_rc,
            "macro_structs": rc_macro_structs,
            "macro_names": rc_macro_names,
            "src": src_rc,
            "n2f": n2f_rc,
            "box": box_rc,
            "stack": st_rc,
            "mesh_info": m_rc,
            "dx": dx_rc,
            "dz_cav": dz_cav,
            "y_mid": y_mid_rc,
            "z_dip": z_dip_rc,
        },
    }


def plot_00_setup_schematics(fig_dir: Path = FIG_DIR, setups: dict | None = None) -> None:
    """Render two-figure setup schematics and setup tables for all four FIT configurations."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if setups is None:
        setups = _build_schematic_setups()

    pec_s = setups["pec"]
    ag_s = setups["ag"]
    cav_s = setups["cavity"]
    rc_s = setups["rcled"]

    # 1. Part A: Line dipole above PEC mirror (suffix="")
    dx_pec = float(pec_s["dx"])
    z_top_box = float(pec_s["z_top_box"])
    f0_pec = c0 / LAM_PEC
    n2f_pec_dict = {
        "kind": "box",
        "type_name": "Near2FarMonitor",
        "short_word": "N2F",
        "desc": f"Near2FarMonitor box (ground_plane='-z', x ∈ [-{7 * dx_pec * 1e6:.2f}, {7 * dx_pec * 1e6:.2f}] µm, z ∈ [0, {z_top_box * 1e6:.2f}] µm)",
        "label": "Near2FarMonitor box (ground_plane='-z')",
        "pos_str": f"x ∈ [-{7 * dx_pec * 1e6:.2f}, {7 * dx_pec * 1e6:.2f}] µm, z ∈ [0, {z_top_box * 1e6:.2f}] µm",
        "freq_str": f"f = {f0_pec * 1e-12:.1f} THz",
        "frequencies": [f0_pec],
        "h_bounds": (-7.0 * dx_pec, 7.0 * dx_pec),
        "v_bounds": (0.0, z_top_box),
    }
    schematic.save_setup_figures(
        pec_s["solver"],
        fig_dir,
        plane="xz",
        cut_coord=float(pec_s["y_mid"]),
        unit="um",
        sources=[pec_s["src_025"]],
        monitors=[n2f_pec_dict],
        wavelength=LAM_PEC,
        freq=f0_pec,
        min_feature=float(pec_s["h_025"]),
        title_geometry="Part A · Line Dipole Above PEC Mirror (Drexhage Effect & Near2Far Box)",
        variations=[
            {
                "name": "Polarizations",
                "param": "component ∈ {'y' (TE_∥), 'x' (TM_∥), 'z' (TM_⊥)}",
                "note": "Exact 2D Bessel Purcell factors 1 - J₀(2kh), 1 - J₀(2kh) + J₂(2kh), 1 + J₀(2kh) + J₂(2kh)",
            },
            {
                "name": "Broadband & monochromatic kh sweep",
                "param": "kh ∈ [0.25, 9.5], h/λ₀ ∈ {0.24, 0.48, 0.76}",
                "note": "Local source work DFTMonitor at (0, y_mid, h) and Near2FarMonitor(ground_plane='-z') angular patterns",
            },
        ],
    )

    # 2. Part B: Line dipole above dispersive Drude silver mirror (suffix="_silver")
    dx_core_ag = float(ag_s["dx_core"])
    dz_ag = float(ag_s["dz_ag"])
    h_ag = float(ag_s["h_ag"])
    f0_ag = c0 / LAM_AG
    box_ag_dict = {
        "kind": "box",
        "type_name": "FluxMonitor",
        "short_word": "Flux",
        "desc": f"Dipole FluxMonitor box (P_tot, 4Δx × 4Δz around z = {h_ag * 1e9:.1f} nm)",
        "label": "Dipole FluxMonitor box (P_tot)",
        "pos_str": f"x ∈ [-{2 * dx_core_ag * 1e9:.1f}, {2 * dx_core_ag * 1e9:.1f}] nm, z ∈ [{(h_ag - 2 * dz_ag) * 1e9:.1f}, {(h_ag + 2 * dz_ag) * 1e9:.1f}] nm",
        "freq_str": f"f = {f0_ag * 1e-12:.1f} THz",
        "frequencies": [f0_ag],
        "h_bounds": (-2.0 * dx_core_ag, 2.0 * dx_core_ag),
        "v_bounds": (h_ag - 2.0 * dz_ag, h_ag + 2.0 * dz_ag),
    }
    schematic.save_setup_figures(
        ag_s["solver"],
        fig_dir,
        plane="xz",
        cut_coord=float(ag_s["y_mid"]),
        unit="nm",
        structures=ag_s["structures"],
        material_names=ag_s["material_names"],
        sources=[ag_s["src"]],
        monitors=[box_ag_dict, ag_s["plane"]],
        wavelength=LAM_AG,
        freq=f0_ag,
        min_feature=float(ag_s["mesh_info"]["skin_depth_nm"]) * 1e-9,
        suffix="_silver",
        title_geometry="Part B · Line Dipole Above Dispersive Drude Silver Mirror (δE = 24.41 nm)",
        variations=[
            {
                "name": "Height & polarization sweep",
                "param": "h ∈ [20, 220] nm (kh ∈ [0.21, 2.30]), pol ∈ {'y', 'x', 'z'}",
                "note": "Benchmarked against 2D Chance–Prock–Silbey (CPS) Sommerfeld integral (F_tot, F_rad, F_nonrad, u_SPP = 1.0338)",
            },
            {
                "name": "Skin-depth resolution study",
                "param": "Δz_Ag ∈ [2.0, 24.0] nm (1.02 to 12.21 cells / δE)",
                "note": "Electric-field skin depth δE = 24.41 nm, power skin depth δI = 12.21 nm at λ₀ = 600 nm",
            },
        ],
    )

    # 3. Part C: Planar PEC microcavity (suffix="_cavity")
    f0_cav = float(cav_s["f0"])
    lam_cav_0 = c0 / f0_cav
    schematic.save_setup_figures(
        cav_s["solver"],
        fig_dir,
        plane="xz",
        cut_coord=float(cav_s["y_mid"]),
        unit="um",
        sources=[cav_s["src"]],
        monitors=[cav_s["dft"]],
        wavelength=lam_cav_0,
        freq=f0_cav,
        min_feature=0.50e-6,
        suffix="_cavity",
        title_geometry="Part C · Line Dipole in Planar PEC Microcavity (n = 2.4, L = 1.0 µm)",
        variations=[
            {
                "name": "Band-limited optical thickness sweep",
                "param": "d = nL/λ₀ ∈ [0.22, 2.32] across 3 spectral windows",
                "note": "Avoids v_g = 0 Van Hove waveguide cutoffs (d = 0.5, 1.5 for J_y, J_x; d = 1.0, 2.0 for J_z)",
            },
        ],
    )

    # 4. Part D: Blue InGaN DBR micro-RCLED (suffix="_rcled")
    dx_rc = float(rc_s["dx"])
    dz_cav = float(rc_s["dz_cav"])
    z_dip_rc = float(rc_s["z_dip"])
    f0_rc = c0 / LAM_RCLED
    box_rc_dict = {
        "kind": "box",
        "type_name": "FluxMonitor",
        "short_word": "Flux",
        "desc": f"Dipole FluxMonitor box (4Δx × 4Δz_cav around z_dip = {z_dip_rc * 1e9:.1f} nm)",
        "label": "Dipole FluxMonitor box",
        "pos_str": f"x ∈ [-{2 * dx_rc * 1e9:.1f}, {2 * dx_rc * 1e9:.1f}] nm, z ∈ [{(z_dip_rc - 2 * dz_cav) * 1e9:.1f}, {(z_dip_rc + 2 * dz_cav) * 1e9:.1f}] nm",
        "freq_str": f"f = {f0_rc * 1e-12:.1f} THz",
        "frequencies": [f0_rc],
        "h_bounds": (-2.0 * dx_rc, 2.0 * dx_rc),
        "v_bounds": (z_dip_rc - 2.0 * dz_cav, z_dip_rc + 2.0 * dz_cav),
    }
    schematic.save_setup_figures(
        rc_s["solver"],
        fig_dir,
        plane="xz",
        cut_coord=float(rc_s["y_mid"]),
        unit="nm",
        structures=rc_s["macro_structs"],
        material_names=rc_s["macro_names"],
        sources=[rc_s["src"]],
        monitors=[box_rc_dict, rc_s["n2f"]],
        wavelength=LAM_RCLED,
        freq=f0_rc,
        min_feature=LAM_RCLED / (4.0 * N_GAN),
        suffix="_rcled",
        title_geometry="Part D · Blue InGaN DBR Micro-RCLED (λ₀ = 450 nm, Interface-Snapped Graded Mesh)",
        variations=[
            {
                "name": "Cavity detuning & angular beaming",
                "param": "d_opt ∈ {1.00 (tuned), 1.055 (red-detuned rabbit ears)}",
                "note": "Bottom NP-GaN/GaN DBR (n = 1.70/2.39) + GaN cavity + top SiO₂/Ta₂O₅ DBR (n = 1.46/2.15)",
            },
            {
                "name": "Incoherent quantum-well array",
                "param": "N_dip = 7 emitters across w_QW = 0.72 µm, N_real ∈ [1, 128]",
                "note": "Deterministic power sum vs random-phase Monte Carlo O(N_real^-1/2) convergence",
            },
        ],
    )


def _save_movie_gif(
    x_arr: np.ndarray,
    z_arr: np.ndarray,
    frames: np.ndarray,
    times: np.ndarray,
    out_path: Path,
    pec_setup: dict | None = None,
    fps: int = 10,
) -> None:
    """Render an animated GIF of 2D dipole-mirror wave interference with geometry overlays."""
    if frames.size == 0:
        return
    figstyle.apply_style()
    vmin, vmax = figstyle.symmetric_limits(frames, percentile=99.2)
    vmax = max(float(vmax) * 0.65, 1e-30)
    vmin = -vmax
    pil_frames = []
    x_um = x_arr * 1e6
    z_um = z_arr * 1e6
    for idx in range(len(frames)):
        fig, ax = figstyle.new_figure(nrows=1, ncols=1, width=6.2, height=4.6, dpi=105)
        im = ax.pcolormesh(
            x_um,
            z_um,
            frames[idx].T,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ax.set(
            xlim=(float(x_um[0]), float(x_um[-1])),
            ylim=(float(z_um[0]), float(z_um[-1])),
            xlabel="Horizontal position x [µm]",
            ylabel="Height above PEC mirror z [µm]",
            title=f"Line dipole pulse near PEC mirror (t = {times[idx] * 1e15:.1f} fs)",
        )
        if pec_setup is not None:
            schematic.overlay_geometry(
                ax,
                pec_setup["solver"],
                plane="xz",
                unit="um",
                sources=[pec_setup["src_050"]],
                monitors=[pec_setup["n2f"]],
                show_boundaries=True,
                show_materials=False,
                show_sources=True,
                show_monitors=True,
            )
            hlp.overlay_xz_boundaries_and_boxes(
                ax,
                pec_setup["solver"],
                unit="um",
                pec_z_label="z = 0: PEC mirror",
                show_z_pml=True,
            )
        else:
            ax.axhline(0.0, color="k", lw=2.2, label="PEC Mirror (z = 0)")
            ax.plot(0.0, 0.50, "ko", ms=5, label="Dipole (h = 0.50 µm)")
        fig.colorbar(im, ax=ax, label=r"Electric field $E_y(x, z, t)$ [V/m]")
        figstyle.finish(fig, auto_legend_below=False)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        pil_frames.append(Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE))
        plt.close(fig)

    if pil_frames:
        pil_frames[0].save(
            out_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )


def generate_all_figures(data_dir: Path = DATA_DIR, fig_dir: Path = FIG_DIR) -> None:
    """Read ``data_dir/task_*.npz`` and render ``00a``/``00b`` setup schematics plus all 14 figures into ``fig_dir``."""
    figstyle.apply_style()
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tasks_data = {}
    for p in sorted(data_dir.glob("task_*.npz")):
        d = dict(np.load(p, allow_pickle=False))
        tasks_data[str(d["name"])] = d

    required = ("pec_drexhage", "silver_drude", "planar_cavity_pec", "rcled_and_incoherent")
    for req in required:
        if req not in tasks_data:
            raise FileNotFoundError(
                f"Missing task {req!r} in {data_dir}. Run --quick or --full first."
            )

    t_pec = tasks_data["pec_drexhage"]
    t_ag = tasks_data["silver_drude"]
    t_cav = tasks_data["planar_cavity_pec"]
    t_rc = tasks_data["rcled_and_incoherent"]

    setups = _build_schematic_setups(
        cpw_pec=int(t_pec.get("cpw", 25)),
        cpw_ag=24 if float(t_ag.get("dz_ag_nm", 2.5)) <= 3.0 else 20,
        dz_ag_nm=float(t_ag.get("dz_ag_nm", 2.5)),
        cpw_rcled=20 if float(t_rc.get("dx_min_nm", 8.69)) < 9.0 else 16,
    )
    pec_s = setups["pec"]
    ag_s = setups["ag"]
    rc_s = setups["rcled"]

    # ------------------------------------------------------------------
    # Fig 00a/00b: Setup Schematics & Setup Tables for All 4 Configurations
    # ------------------------------------------------------------------
    plot_00_setup_schematics(fig_dir=fig_dir, setups=setups)

    # ------------------------------------------------------------------
    # Fig 01: Skin-Depth Refined Mesh View, Graded Step Profile & Kottke DBR
    # ------------------------------------------------------------------
    fig, (ax0, ax1, ax2) = figstyle.new_figure(nrows=1, ncols=3, width=14.4, height=4.5)

    # Panel (a): Skin-depth graded 2D FIT mesh on Drude Silver with zoom inset
    schematic.draw_mesh(
        ax0,
        ag_s["solver"],
        plane="xz",
        unit="nm",
        every=2,
        wavelength=LAM_AG,
        freq=c0 / LAM_AG,
        zoom_inset=True,
        zoom_bounds=(-40.0e-9, 40.0e-9, -50.0e-9, 30.0e-9),
    )
    for _ax_cand in fig.axes:
        if _ax_cand not in (ax0, ax1, ax2):
            _ax_cand._is_inset = True
    schematic.overlay_geometry(
        ax0,
        ag_s["solver"],
        plane="xz",
        unit="nm",
        structures=ag_s["structures"],
        material_names=ag_s["material_names"],
        sources=[ag_s["src"]],
        monitors=[ag_s["plane"]],
        show_boundaries=True,
        show_materials=True,
        show_sources=True,
        show_monitors=True,
    )
    hlp.overlay_xz_boundaries_and_boxes(
        ax0,
        ag_s["solver"],
        unit="nm",
        box_monitors=[(ag_s["box"], "P_tot FluxBox")],
        pec_z_label=None,
        show_z_pml=False,
    )
    for t_obj in list(ax0.texts):
        s_txt = t_obj.get_text()
        if "Drude Ag" in s_txt:
            t_obj.set_text("Drude Ag mirror (δE = 24.4 nm)")
            t_obj.set_position((95.0, -68.0))
    ax0.text(
        0.50,
        0.96,
        f"Δz_Ag = {float(t_ag['dz_ag_nm']):.2f} nm ({float(t_ag['cells_per_skin_depth']):.2f} cells / δE)\n"
        f"δE = {float(t_ag['skin_depth_nm']):.2f} nm, h_min = 20 nm ({float(t_ag['cells_across_feature']):.1f} cells)",
        transform=ax0.transAxes,
        va="top",
        ha="center",
        fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.90, edgecolor="#cbd5e1"),
        zorder=9,
    )
    ax0.set_title("(a) Drude Ag skin-depth graded mesh & zoom")

    # Panel (b): Vertical grid-step profile dz(z) across Silver skin-depth & RCLED DBR stack
    z_ag_nodes_nm = np.asarray(ag_s["solver"].grid.z, dtype=float) * 1e9
    dz_ag_cells_nm = np.diff(z_ag_nodes_nm)
    z_ag_mid_nm = 0.5 * (z_ag_nodes_nm[:-1] + z_ag_nodes_nm[1:])

    z_rc_nodes_nm = np.asarray(rc_s["solver"].grid.z, dtype=float) * 1e9
    dz_rc_cells_nm = np.diff(z_rc_nodes_nm)
    z_rc_mid_nm = 0.5 * (z_rc_nodes_nm[:-1] + z_rc_nodes_nm[1:])

    ax1.step(
        z_ag_mid_nm,
        dz_ag_cells_nm,
        where="mid",
        **figstyle.series_style(0),
        label=f"Part B Drude Ag Δz(z) (min {np.min(dz_ag_cells_nm):.2f} nm)",
    )
    ax1.step(
        z_rc_mid_nm,
        dz_rc_cells_nm,
        where="mid",
        **figstyle.series_style(1),
        label=f"Part D RCLED DBR Δz(z) (min {np.min(dz_rc_cells_nm):.2f} nm)",
    )
    ax1.axhline(
        float(t_ag["skin_depth_nm"]) / 4.0,
        color=figstyle.PALETTE[2],
        ls="--",
        lw=1.3,
        label=f"Skin-depth rule δE / 4 = {float(t_ag['skin_depth_nm']) / 4.0:.2f} nm",
    )
    ax1.axhline(
        (LAM_RCLED / (4.0 * N_GAN)) * 1e9 / 4.0,
        color=figstyle.PALETTE[3],
        ls=":",
        lw=1.3,
        label=f"DBR quarter-wave rule d_GaN / 4 = {(LAM_RCLED / (4.0 * N_GAN)) * 1e9 / 4.0:.2f} nm",
    )
    ax1.axvspan(-150.0, 0.0, color="#94a3b8", alpha=0.22, label="Drude Ag slab (z ∈ [−150, 0] nm)")
    ax1.set(
        xlim=(-480.0, 480.0),
        ylim=(0.0, 28.0),
        xlabel="Vertical coordinate z [nm]",
        ylabel="Local vertical cell size Δz(z) [nm]",
        title="(b) Graded vertical mesh step Δz(z) audit",
    )
    ax1.legend(loc="upper left", fontsize=7.0)

    # Panel (c): Kottke vs Staircase permittivity profile on off-grid DBR layers
    if "kottke_medium" in tasks_data and "staircase_medium" in tasks_data:
        dk = tasks_data["kottke_medium"]
        ds = tasks_data["staircase_medium"]
        n_s = min(len(ds["z_dbr_nm"]), len(ds["eps_x_prof"]))
        n_k = min(len(dk["z_dbr_nm"]), len(dk["eps_x_prof"]))
        ax2.step(
            ds["z_dbr_nm"][:n_s],
            ds["eps_x_prof"][:n_s],
            where="mid",
            color=figstyle.PALETTE[1],
            lw=1.8,
            label=r"Staircase $\varepsilon_{xx}(z)$",
        )
        ax2.plot(
            dk["z_dbr_nm"][:n_k],
            dk["eps_x_prof"][:n_k],
            "o-",
            color=figstyle.PALETTE[2],
            lw=1.6,
            ms=3.5,
            label=r"Kottke arithmetic $\langle\varepsilon\rangle_x$",
        )
        ax2.plot(
            dk["z_dbr_nm"][:n_k],
            dk["eps_z_prof"][:n_k],
            "s--",
            color=figstyle.PALETTE[0],
            lw=1.3,
            ms=3.0,
            label=r"Kottke harmonic $\langle\varepsilon^{-1}\rangle_z^{-1}$",
        )
        ax2.set_xlim(-350.0, 20.0)
        ax2.set_ylim(0.5, 6.5)
    ax2.set(
        xlabel="Vertical coordinate z [nm]",
        ylabel=r"Relative permittivity $\varepsilon_r(z)$ [dimensionless]",
        title="(c) Subpixel DBR smoothing: Kottke vs. staircase",
    )
    ax2.legend(loc="upper right", fontsize=7.5)
    figstyle.finish(fig, auto_legend_below=False)
    figstyle.save(fig, fig_dir / "01_geometry_and_mesh_overview.png", auto_legend_below=False)

    # ------------------------------------------------------------------
    # Fig 02: PEC Mirror Drexhage Purcell Factor (2D FIT vs Exact & 2D vs 3D)
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.2)
    kh_fine = np.linspace(0.01, 9.6, 400)
    cols = {"y": "#1b9e77", "x": "#2b5c8f", "z": "#d95f02"}
    labels_pol = {
        "y": r"$\mathrm{TE}_\parallel$ ($J_y$, $1 - J_0(2kh)$)",
        "x": r"$\mathrm{TM}_\parallel$ ($J_x$, $1 - J_0 + J_2$)",
        "z": r"$\mathrm{TM}_\perp$ ($J_z$, $1 + J_0 + J_2$)",
    }
    for pol in ("y", "x", "z"):
        ax0.plot(
            kh_fine,
            hlp.pec_mirror_purcell_2d(kh_fine, pol),
            color=cols[pol],
            lw=2.0,
            label=f"Exact 2D {labels_pol[pol]}",
        )
        ax0.plot(
            t_pec[f"kh_{pol}"],
            t_pec[f"F_work_{pol}"],
            "o",
            color=cols[pol],
            ms=4.2,
            alpha=0.85,
            label=f"Wakis 2D FIT ({pol.upper()} work)",
        )
        ax0.plot(
            t_pec[f"mono_kh_{pol}"],
            t_pec[f"mono_flux_{pol}"],
            "s",
            color=cols[pol],
            ms=6.2,
            mfc="none",
            mew=1.6,
        )
    ax0.axhline(1.0, color="0.4", ls=":", lw=1.2)
    ax0.set(
        xlabel="Normalized dipole height kh = 2πh / λ₀ [dimensionless]",
        ylabel=r"Purcell factor $F_P(kh) = P'(h) / P'_0$ [dimensionless]",
        xlim=(0.0, 9.6),
        ylim=(0.0, 2.15),
        title="(a) 2D PEC mirror Drexhage oscillations: Wakis FIT vs. exact Bessel",
    )
    ax0.legend(loc="upper right", fontsize=7.2, ncol=2)

    # Panel (b): 2D cylindrical Bessel decay O((kh)^-1/2) vs 3D spherical decay O((kh)^-1)
    kh_comp = np.linspace(0.01, 14.0, 500)
    ax1.plot(
        kh_comp,
        hlp.pec_mirror_purcell_2d(kh_comp, "y"),
        color="#1b9e77",
        lw=2.0,
        label=r"2D $\mathrm{TE}_\parallel$: $1 - J_0(2kh) \sim \mathcal{O}((kh)^{-1/2})$",
    )
    ax1.plot(
        kh_comp,
        hlp.pec_mirror_purcell_3d(kh_comp, "par"),
        "--",
        color="#2b5c8f",
        lw=1.9,
        label=r"3D parallel (Drexhage): $\sim \mathcal{O}((kh)^{-1})$",
    )
    ax1.plot(
        kh_comp,
        hlp.pec_mirror_purcell_2d(kh_comp, "z"),
        color="#d95f02",
        lw=2.0,
        label=r"2D $\mathrm{TM}_\perp$: $1 + 2J_1(2kh)/(2kh)$",
    )
    ax1.plot(
        kh_comp,
        hlp.pec_mirror_purcell_3d(kh_comp, "perp"),
        "--",
        color="#7570b3",
        lw=1.9,
        label=r"3D perpendicular (Drexhage)",
    )
    ax1.axhline(1.0, color="0.4", ls=":", lw=1.2)
    ax1.set(
        xlabel="Normalized dipole height kh = 2πh / λ₀ [dimensionless]",
        ylabel=r"Purcell factor $F_P(kh)$ [dimensionless]",
        xlim=(0.0, 14.0),
        ylim=(0.0, 2.15),
        title="(b) Dimensional comparison: 2D cylindrical vs. 3D spherical decay",
    )
    ax1.legend(loc="upper right", fontsize=7.8)
    figstyle.finish(fig, auto_legend_below=False)
    figstyle.save(fig, fig_dir / "02_pec_mirror_drexhage_purcell.png", auto_legend_below=False)

    # ------------------------------------------------------------------
    # Fig 03: PEC Mirror Far-Field Angular Radiation Patterns (Near2FarMonitor)
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.2, height=4.0, sharey=False)
    th_deg = t_pec["theta_deg"]
    h_tags = [("025", 0, "#1b9e77"), ("050", 1, "#d95f02"), ("075", 2, "#2b5c8f")]
    titles_p3 = {
        "y": r"(a) $\mathrm{TE}_\parallel$ ($J_y$): $\frac{2}{\pi}\sin^2(kh\cos\theta)$",
        "x": r"(b) $\mathrm{TM}_\parallel$ ($J_x$): $\frac{4}{\pi}\cos^2\theta\sin^2(kh\cos\theta)$",
        "z": r"(c) $\mathrm{TM}_\perp$ ($J_z$): $\frac{4}{\pi}\sin^2\theta\cos^2(kh\cos\theta)$",
    }
    for ax, pol in zip(axes, ("y", "x", "z")):
        kh_pol = np.asarray(t_pec[f"mono_kh_{pol}"], dtype=float)
        for tag, idx_h, col in h_tags:
            h_lam_val = float(kh_pol[idx_h] / (2.0 * np.pi))
            h_lbl = rf"$h = {h_lam_val:.2f}\lambda_0$"
            p_ana = t_pec[f"pat_ana_{pol}_{tag}"]
            p_num = t_pec[f"pat_num_{pol}_{tag}"]
            ax.plot(th_deg, p_ana, color=col, lw=2.0, label=f"Exact {h_lbl}")
            ax.plot(
                th_deg[::4],
                p_num[::4],
                "o",
                color=col,
                ms=4.0,
                mfc="none",
                mew=1.3,
                label=f"Near2Far {h_lbl}",
            )
        ax.set(
            xlabel="Polar angle θ from surface normal [deg]",
            ylabel=r"Angular intensity $\frac{1}{P'_0}\frac{\mathrm{d}P'}{\mathrm{d}\theta}$ [1/rad]",
            xlim=(-90.0, 90.0),
            title=titles_p3[pol],
        )
        ax.legend(loc="upper right", fontsize=7.0, ncol=2)
    figstyle.finish(fig, auto_legend_below=False)
    figstyle.save(fig, fig_dir / "03_pec_mirror_angular_patterns.png", auto_legend_below=False)

    # ------------------------------------------------------------------
    # Fig 04: 2D Steady-State Field Snapshots (h ~ 0.25 lambda vs h ~ 0.50 lambda)
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.4)
    sx_um = t_pec["snap_x"] * 1e6
    sz_um = t_pec["snap_z"] * 1e6
    h_025_um = float(t_pec["mono_kh_y"][0] / (2.0 * np.pi))
    h_050_um = float(t_pec["mono_kh_y"][1] / (2.0 * np.pi))
    for ax, tag, h_um, src_obj, desc in (
        (axes[0], "025", h_025_um, pec_s["src_025"], f"(a) Constructive beaming (h = {h_025_um:.2f} λ₀)"),
        (axes[1], "050", h_050_um, pec_s["src_050"], f"(b) Destructive node / split (h = {h_050_um:.2f} λ₀)"),
    ):
        ey_c = t_pec[f"ey_{tag}"]
        re_ey = np.real(ey_c) * 1e6
        vmin, vmax = figstyle.symmetric_limits(re_ey, percentile=98.5)
        im = ax.pcolormesh(
            sx_um, sz_um, re_ey.T, cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto"
        )
        ax.set(
            xlim=(float(sx_um[0]), float(sx_um[-1])),
            ylim=(float(sz_um[0]), float(sz_um[-1])),
            xlabel="Horizontal position x [µm]",
            ylabel="Height above PEC mirror z [µm]",
            title=desc,
        )
        schematic.overlay_geometry(
            ax,
            pec_s["solver"],
            plane="xz",
            unit="um",
            sources=[src_obj],
            monitors=[pec_s["n2f"]],
            show_boundaries=True,
            show_materials=False,
            show_sources=True,
            show_monitors=True,
        )
        hlp.overlay_xz_boundaries_and_boxes(
            ax,
            pec_s["solver"],
            unit="um",
            pec_z_label="z = 0: PEC mirror",
            show_z_pml=True,
        )
        fig.colorbar(im, ax=ax, label=r"Steady-state $\mathrm{Re}[E_y(x, z)]$ [µV s/m]")
    figstyle.finish(fig, auto_legend_below=False)
    figstyle.save(fig, fig_dir / "04_pec_mirror_field_snapshots.png", auto_legend_below=False)

    # ------------------------------------------------------------------
    # Fig 05: Animated Time-Domain Movie GIF
    # ------------------------------------------------------------------
    _save_movie_gif(
        t_pec["movie_x"],
        t_pec["movie_z"],
        t_pec["movie_frames"],
        t_pec["movie_times"],
        fig_dir / "05_dipole_mirror_interference.gif",
        pec_setup=pec_s,
    )

    # ------------------------------------------------------------------
    # Fig 06: Drude Silver Mirror Drexhage & CPS Radiative/SPP Partition
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.2)
    kh_d = t_ag["kh_dense"]
    ax0.plot(kh_d, t_ag["cps_z_tot"], color="#d95f02", lw=2.2, label=r"CPS $\mathrm{TM}_\perp$ $F_{\mathrm{tot}}$ (rad + SPP + Ohmic)")
    ax0.plot(kh_d, t_ag["cps_z_rad"], "--", color="#d95f02", lw=1.7, label=r"CPS $\mathrm{TM}_\perp$ $F_{\mathrm{rad}}$ (upper hemisphere)")
    ax0.plot(kh_d, t_ag["cps_z_nonrad"], ":", color="#d95f02", lw=1.7, label=r"CPS $\mathrm{TM}_\perp$ $F_{\mathrm{nonrad}}$ (SPP + Ohmic)")
    ax0.plot(t_ag["kh_num_z"], t_ag["F_tot_num_z"], "o", color="#d95f02", ms=6.0, label=r"Wakis 2D FIT $\mathrm{TM}_\perp$ $F_{\mathrm{tot}}$ (FluxBox)")

    ax0.plot(kh_d, t_ag["cps_y_tot"], color="#1b9e77", lw=2.2, label=r"CPS $\mathrm{TE}_\parallel$ $F_{\mathrm{tot}}$")
    ax0.plot(kh_d, t_ag["cps_y_rad"], "--", color="#1b9e77", lw=1.6, label=r"CPS $\mathrm{TE}_\parallel$ $F_{\mathrm{rad}}$")
    ax0.plot(t_ag["kh_num_y"], t_ag["F_tot_num_y"], "s", color="#1b9e77", ms=5.8, label=r"Wakis 2D FIT $\mathrm{TE}_\parallel$ $F_{\mathrm{tot}}$ (FluxBox)")

    ax0.plot(kh_d, t_ag["cps_x_tot"], color="#2b5c8f", lw=1.8, alpha=0.85, label=r"CPS $\mathrm{TM}_\parallel$ $F_{\mathrm{tot}}$")
    ax0.plot(t_ag["kh_num_x"], t_ag["F_tot_num_x"], "^", color="#2b5c8f", ms=5.5, label=r"Wakis 2D FIT $\mathrm{TM}_\parallel$ $F_{\mathrm{tot}}$")

    ax0.set(
        xlabel="Normalized height kh = 2πh / λ₀ (λ₀ = 600 nm) [dimensionless]",
        ylabel="Normalized emission rate F(kh) [dimensionless]",
        xlim=(0.2, 4.0),
        ylim=(0.0, 3.3),
        title="(a) Silver (Drude) mirror: Wakis 2D FIT vs. CPS Sommerfeld",
    )
    ax0.legend(loc="upper right", fontsize=7.0, ncol=2)

    ax1.plot(kh_d, t_ag["cps_y_eta"] * 100.0, color="#1b9e77", lw=2.2, label=r"$\mathrm{TE}_\parallel$ ($J_y$) radiative efficiency $\eta_{\mathrm{rad}}$")
    ax1.plot(kh_d, t_ag["cps_z_eta"] * 100.0, color="#d95f02", lw=2.2, label=r"$\mathrm{TM}_\perp$ ($J_z$) radiative efficiency $\eta_{\mathrm{rad}}$")
    eta_x = np.where(t_ag["cps_x_tot"] > 0, t_ag["cps_x_rad"] / t_ag["cps_x_tot"], 0.0)
    ax1.plot(kh_d, eta_x * 100.0, "--", color="#2b5c8f", lw=1.8, label=r"$\mathrm{TM}_\parallel$ ($J_x$) radiative efficiency $\eta_{\mathrm{rad}}$")
    ax1.plot(
        t_ag["kh_num_y"],
        (1.0 - t_ag["F_abs_num_y"] / np.maximum(t_ag["F_tot_num_y"], 1e-12)) * 100.0,
        "s",
        color="#1b9e77",
        ms=5.5,
        label=r"Wakis $\mathrm{TE}_\parallel$ $(1 - P_{\mathrm{metal}}/P_{\mathrm{tot}})$",
    )
    ax1.set(
        xlabel="Normalized height kh = 2πh / λ₀ [dimensionless]",
        ylabel=r"Radiative quantum efficiency $\eta_{\mathrm{rad}} = F_{\mathrm{rad}}/F_{\mathrm{tot}}$ [%]",
        xlim=(0.2, 4.0),
        ylim=(0.0, 105.0),
        title="(b) Near-field SPP & Ohmic quenching on Silver",
    )
    ax1.legend(loc="lower right", fontsize=7.8)
    figstyle.finish(fig, auto_legend_below=False)
    figstyle.save(fig, fig_dir / "06_drude_silver_drexhage_cps.png", auto_legend_below=False)

    # ------------------------------------------------------------------
    # Fig 07: CPS Wavevector Spectrum (SPP Pole) & Skin-Depth Field Profile
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.2)
    u_g = t_ag["u_grid"]
    u_spp = float(t_ag["u_spp"])
    ax0.semilogy(u_g, np.maximum(t_ag["df_du_z"], 1e-5), color="#d95f02", lw=2.1, label=r"$\mathrm{TM}_\perp$ ($J_z$) power density $\mathrm{d}F/\mathrm{d}u$")
    ax0.semilogy(u_g, np.maximum(t_ag["df_du_x"], 1e-5), color="#2b5c8f", lw=1.9, label=r"$\mathrm{TM}_\parallel$ ($J_x$) power density $\mathrm{d}F/\mathrm{d}u$")
    ax0.semilogy(u_g, np.maximum(t_ag["df_du_y"], 1e-5), color="#1b9e77", lw=2.0, label=r"$\mathrm{TE}_\parallel$ ($J_y$, no SPP pole)")
    ax0.axvline(1.0, color="0.3", ls="--", lw=1.3, label="Light line u = kx / k₀ = 1")
    ax0.axvline(u_spp, color="#d95f02", ls=":", lw=1.6, label=f"SPP pole u_SPP = {u_spp:.4f}")
    ax0.axvspan(0.0, 1.0, color="#e0f2fe", alpha=0.35, label="Radiative cone (u < 1)")
    ax0.set(
        xlabel=r"Normalized in-plane wavevector $u = k_x / k_0$ [dimensionless]",
        ylabel=r"Spectral dissipation density $\mathrm{d}F/\mathrm{d}u$ (log scale)",
        xlim=(0.0, 2.5),
        ylim=(1e-4, 3e2),
        title="(a) CPS k-space spectrum at kh = 0.42: light cone vs. SPP pole",
    )
    ax0.grid(True, which="both", ls=":", alpha=0.4)
    ax0.legend(loc="upper right", fontsize=7.5)

    z_nm = t_ag["skin_z_nm"]
    m_ag = z_nm <= 0.0
    delta_nm = float(t_ag["skin_depth_nm"])
    z_ana_nm = np.linspace(-120.0, 0.0, 200)
    ax1.semilogy(
        z_nm[m_ag],
        np.maximum(t_ag["skin_ey_norm"][m_ag], 1e-4),
        "o-",
        color="#1b9e77",
        lw=1.8,
        ms=4.5,
        label=f"Wakis |Ey(z)| in Ag ({t_ag['cells_per_skin_depth']:.2f} cells / δE)",
    )
    ax1.semilogy(
        z_ana_nm,
        np.exp(z_ana_nm / delta_nm),
        "--",
        color="k",
        lw=2.0,
        label=f"Analytical exp(z / δE), δE = {delta_nm:.2f} nm",
    )
    ax1.axvline(-delta_nm, color="#d95f02", ls=":", lw=1.5, label=f"1/e skin depth z = −{delta_nm:.1f} nm")
    ax1.set(
        xlabel="Depth inside Silver mirror z [nm] (interface at z = 0)",
        ylabel="Normalized electric field |E(z)| / |E(0)| (log scale)",
        xlim=(-120.0, 5.0),
        ylim=(2e-3, 1.5),
        title="(b) Electric-field skin-depth decay inside Drude Silver",
    )
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax1.legend(loc="lower left", fontsize=7.8)
    figstyle.finish(fig, auto_legend_below=False)
    figstyle.save(fig, fig_dir / "07_sommerfeld_spp_kspace_spectrum.png", auto_legend_below=False)

    # ------------------------------------------------------------------
    # Fig 08: Planar PEC Microcavity Purcell Factor F_P(d) (2D vs 3D Abram 1998)
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.3)
    d_d = t_cav["d_dense"]
    ax0.plot(d_d, t_cav["F_2d_dense_y"], color="#1b9e77", lw=2.0, label=r"Exact 2D $\mathrm{TE}_\parallel$ ($J_y$)")
    ax0.plot(t_cav["d_sim_y"], t_cav["F_sim_y"], "o", color="#1b9e77", ms=4.5, label=r"Wakis 2D FIT $\mathrm{TE}_\parallel$")

    ax0.plot(d_d, t_cav["F_2d_dense_x"], color="#2b5c8f", lw=1.9, label=r"Exact 2D $\mathrm{TM}_\parallel$ ($J_x$)")
    ax0.plot(t_cav["d_sim_x"], t_cav["F_sim_x"], "s", color="#2b5c8f", ms=4.2, mfc="none", mew=1.4, label=r"Wakis 2D FIT $\mathrm{TM}_\parallel$")

    ax0.plot(d_d, t_cav["F_2d_dense_z"], color="#d95f02", lw=2.0, label=r"Exact 2D $\mathrm{TM}_\perp$ ($J_z$, TEM₀ + even)")
    ax0.plot(t_cav["d_sim_z"], t_cav["F_sim_z"], "^", color="#d95f02", ms=4.5, label=r"Wakis 2D FIT $\mathrm{TM}_\perp$")
    ax0.axhline(1.0, color="0.4", ls=":", lw=1.2)
    ax0.set(
        xlabel=r"Normalized cavity thickness $d = nL / \lambda_0$ ($n = 2.4$) [dimensionless]",
        ylabel=r"2D Cavity Purcell factor $F_P(d)$ [dimensionless]",
        xlim=(0.18, 2.38),
        ylim=(0.0, 3.2),
        title="(a) 2D Planar PEC microcavity: Wakis FIT vs. Poisson-sum DOS",
    )
    ax0.legend(loc="upper right", fontsize=7.2, ncol=2)

    # Panel (b): 2D 1/sqrt Van Hove singularities vs 3D Abram et al. (1998) Eq. 7 step DOS
    ax1.plot(d_d, t_cav["F_2d_dense_y"], color="#1b9e77", lw=2.0, label=r"2D $\mathrm{TE}_\parallel$: $1/\sqrt{d^2 - (m-\frac{1}{2})^2}$ Van Hove DOS")
    ax1.plot(d_d, t_cav["F_3d_dense_par"], "--", color="#1b9e77", lw=2.2, label=r"3D parallel (Abram 1998 Eq. 7): step DOS")
    ax1.plot(d_d, t_cav["F_2d_dense_z"], color="#d95f02", lw=2.0, label=r"2D $\mathrm{TM}_\perp$: $\frac{2}{\pi d} + \frac{4}{\pi d}\sum\sqrt{1-(m/d)^2}$")
    ax1.plot(d_d, t_cav["F_3d_dense_perp"], "--", color="#d95f02", lw=2.2, label=r"3D perp (Abram 1998 Eq. 7): $\frac{1}{2d} + \frac{3}{2d}\sum(1-(m/d)^2)$")
    for dc in (0.5, 1.0, 1.5, 2.0):
        ax1.axvline(dc, color="0.6", ls=":", lw=1.0)
    ax1.set(
        xlabel=r"Normalized cavity thickness $d = nL / \lambda_0$ [dimensionless]",
        ylabel=r"Purcell factor $F_P(d)$ [dimensionless]",
        xlim=(0.18, 2.38),
        ylim=(0.0, 3.2),
        title="(b) 2D Van Hove singularities vs. 3D step DOS (Abram et al. 1998)",
    )
    ax1.legend(loc="upper right", fontsize=7.4)
    figstyle.finish(fig, auto_legend_below=False)
    figstyle.save(fig, fig_dir / "08_planar_cavity_purcell_2d_vs_3d.png", auto_legend_below=False)

    # ------------------------------------------------------------------
    # Fig 09: Waveguide Cutoff Ringing Diagnostic & Cure (v_g = 0 at d = 0.5)
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.2, height=4.0)
    t_fs = t_cav["trace_t"] * 1e15
    ey_tr = t_cav["trace_ey"]
    ey_norm = ey_tr / max(float(np.max(np.abs(ey_tr))), 1e-30)
    ax0.plot(t_fs, ey_norm, color="#2b5c8f", lw=1.3, label=r"Broadband pulse crossing $d = 0.50$ cutoff")
    ax0.set(
        xlabel="Simulation time t [fs]",
        ylabel=r"Normalized source field $E_y(\mathbf{r}_0, t) / \max|E_y|$",
        ylim=(-0.45, 0.45),
        title=r"(a) Trapped $v_g = 0$ Van Hove cutoff oscillation ($t^{-1/2}$ tail)",
    )
    ax0.legend(loc="upper right", fontsize=8.0)

    ax1.plot(t_cav["d_dense"], t_cav["F_2d_dense_y"], color="k", lw=2.0, label="Exact 2D analytical DOS")
    ax1.plot(
        t_cav["d_ring_pts"],
        t_cav["F_ring_spec"],
        "--",
        color="#d95f02",
        lw=1.6,
        label="Single cutoff-crossing pulse (sinc ringing)",
    )
    m_b = t_cav["d_sim_y"] <= 1.40
    ax1.plot(
        t_cav["d_sim_y"][m_b],
        t_cav["F_sim_y"][m_b],
        "o",
        color="#1b9e77",
        ms=5.0,
        label="Band-limited windowing (ringing eliminated)",
    )
    ax1.set(
        xlabel=r"Normalized cavity thickness $d = nL / \lambda_0$ [dimensionless]",
        ylabel=r"Extracted Purcell factor $F_P^{\mathrm{TE},\parallel}(d)$",
        xlim=(0.22, 1.38),
        ylim=(-0.15, 3.2),
        title="(b) Spectral sinc ringing vs. band-limited windowing",
    )
    ax1.legend(loc="upper right", fontsize=7.8)
    figstyle.finish(fig, auto_legend_below=False)
    figstyle.save(fig, fig_dir / "09_cavity_cutoff_ringing_diagnostic.png", auto_legend_below=False)

    # ------------------------------------------------------------------
    # Fig 10: Blue InGaN DBR Micro-RCLED Reflectance Spectra & Cavity Field
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.8, height=4.4)
    wl_nm = t_rc["wl_tmm_nm"]
    ax0.plot(wl_nm, t_rc["r_bot_spec"] * 100.0, color="#2b5c8f", lw=2.0, label="Bottom NP-GaN/GaN DBR (n = 1.70/2.39, Lee 2024)")
    ax0.plot(wl_nm, t_rc["r_top_spec"] * 100.0, "--", color="#1b9e77", lw=2.0, label="Top SiO₂/Ta₂O₅ DBR (n = 1.46/2.15, Lee 2024)")
    ax0.plot(wl_nm, t_rc["r_full_spec"] * 100.0, color="#d95f02", lw=1.8, label="Full RCLED Fabry–Pérot cavity (λ₀ = 450 nm dip)")
    ax0.axvline(450.0, color="0.3", ls=":", lw=1.3)
    ax0.set(
        xlabel="Free-space wavelength λ [nm]",
        ylabel="Normal-incidence reflectance R(λ) [%]",
        xlim=(380.0, 530.0),
        ylim=(0.0, 103.0),
        title="(a) Lee et al. (2024) blue InGaN RCLED DBR stopband",
    )
    figstyle.spectrum_grid(ax0)
    ax0.legend(loc="lower right", fontsize=7.4)

    rx_nm = t_rc["rcled_x"] * 1e9
    rz_nm = t_rc["rcled_z"] * 1e9
    i_rc = np.abs(t_rc["rcled_ey"]) ** 2
    i_rc_db = 10.0 * np.log10(np.maximum(i_rc / max(float(np.max(i_rc)), 1e-30), 1e-4))
    im = ax1.pcolormesh(rx_nm, rz_nm, i_rc_db.T, cmap="inferno", vmin=-32.0, vmax=0.0, shading="auto")
    ax1.set(
        xlim=(float(rx_nm[0]), float(rx_nm[-1])),
        ylim=(float(rz_nm[0]), float(rz_nm[-1])),
        xlabel="Lateral position x [nm]",
        ylabel="Vertical position z [nm]",
        title="(b) Tuned RCLED standing-wave intensity [dB]",
    )
    schematic.overlay_geometry(
        ax1,
        rc_s["solver"],
        plane="xz",
        unit="nm",
        structures=rc_s["macro_structs"],
        material_names=rc_s["macro_names"],
        sources=[rc_s["src"]],
        monitors=[rc_s["n2f"]],
        show_boundaries=True,
        show_materials=True,
        show_sources=True,
        show_monitors=True,
    )
    fig.colorbar(im, ax=ax1, label="Normalized intensity [dB]")
    figstyle.finish(fig, auto_legend_below=False)
    figstyle.save(fig, fig_dir / "10_rcled_dbr_reflectance_and_cavity_field.png", auto_legend_below=False)

    # ------------------------------------------------------------------
    # Fig 11: RCLED Angular Extraction & Detuning Beaming (Benisty et al. 1998)
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.2)
    th_rc = t_rc["theta_deg"]
    ax0.plot(th_rc, t_rc["pat_tmm_tuned"], color="#1b9e77", lw=2.2, label="TMM Airy formula (tuned d = 1.00)")
    ax0.plot(th_rc[::3], t_rc["pat_wakis_tuned"][::3], "o", color="#1b9e77", ms=4.5, label="Wakis 2D FIT Near2Far (tuned d = 1.00)")
    ax0.plot(th_rc, t_rc["pat_tmm_detuned"], color="#d95f02", lw=2.2, label="TMM Airy formula (detuned d = 1.055, rabbit ears)")
    ax0.plot(th_rc[::3], t_rc["pat_wakis_detuned"][::3], "s", color="#d95f02", ms=4.5, mfc="none", mew=1.4, label="Wakis 2D FIT Near2Far (detuned d = 1.055)")
    ax0.axvspan(-30.0, 30.0, color="#e0f2fe", alpha=0.35, label="NA = 0.50 collection cone (±30°)")
    ax0.set(
        xlabel="External air emission angle θ_ext [deg]",
        ylabel="Normalized far-field intensity I(θ_ext) / I_max",
        xlim=(-80.0, 80.0),
        ylim=(0.0, 1.12),
        title="(a) RCLED angular beaming: tuned normal lobe vs. detuned rabbit ears",
    )
    ax0.legend(loc="upper right", fontsize=7.2)

    d_det = t_rc["d_detune_grid"]
    ax1.plot(d_det, t_rc["eta_na05_tmm"] * 100.0, color="#2b5c8f", lw=2.2, label="Fraction of top emission inside NA = 0.50 (±30°)")
    ax1.plot(
        d_det,
        t_rc["eta_tot_tmm"] / max(float(np.max(t_rc["eta_tot_tmm"])), 1e-30) * 100.0,
        "--",
        color="#d95f02",
        lw=2.0,
        label="Normalized total top-extracted power (arb. %)",
    )
    ax1.axvline(1.00, color="#1b9e77", ls=":", lw=1.5, label="Tuned resonance d = 1.00")
    ax1.axvline(1.055, color="#d95f02", ls=":", lw=1.5, label="Detuned rabbit-ear d = 1.055")
    ax1.set(
        xlabel=r"Normalized cavity optical thickness $d = n_{\mathrm{GaN}} L_{\mathrm{cav}} / \lambda_0$",
        ylabel="Collection efficiency / relative power [%]",
        xlim=(0.88, 1.16),
        ylim=(0.0, 105.0),
        title="(b) NA = 0.50 directionality vs. cavity detuning (Benisty 1998)",
    )
    ax1.legend(loc="lower left", fontsize=7.6)
    figstyle.finish(fig, auto_legend_below=False)
    figstyle.save(fig, fig_dir / "11_rcled_angular_extraction_beaming.png", auto_legend_below=False)

    # ------------------------------------------------------------------
    # Fig 12: Incoherent Quantum-Well Emission (Power Sum vs Coherent Field Sum)
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.2)
    ax0.plot(
        th_rc,
        t_rc["pat_coh_norm"],
        "--",
        color="#d95f02",
        lw=1.8,
        label=r"Coherent field sum $|\sum_{m=1}^7 \mathbf{E}_m(\theta)|^2$ (artificial fringes!)",
    )
    ax0.plot(
        th_rc,
        t_rc["pat_incoh_norm"],
        color="#1b9e77",
        lw=2.4,
        label=r"Incoherent power sum $\sum_{m=1}^7 |\mathbf{E}_m(\theta)|^2$ (physical QW)",
    )
    if "mc_pat_1" in t_rc and "mc_pat_16" in t_rc:
        ax0.plot(
            th_rc,
            t_rc["mc_pat_1"],
            ":",
            color="#7570b3",
            lw=1.3,
            alpha=0.8,
            label="Single random-phase realization (N_real = 1)",
        )
        ax0.plot(
            th_rc,
            t_rc["mc_pat_16"],
            "-.",
            color="#2b5c8f",
            lw=1.6,
            label="Random-phase Monte Carlo average (N_real = 16)",
        )
    ax0.set(
        xlabel="External air emission angle θ_ext [deg]",
        ylabel="Normalized angular power I(θ_ext) / I_max",
        xlim=(-82.0, 82.0),
        ylim=(0.0, 1.12),
        title="(a) Far-field: coherent fringes vs. incoherent power average",
    )
    ax0.legend(loc="upper right", fontsize=7.3)

    # Panel (b): Lateral intensity cross-section in the air superstrate comparing coherent vs incoherent
    qx_um = t_rc["qw_x"] * 1e6
    iz_air = -3
    prof_coh = t_rc["i_map_coh"][:, iz_air]
    prof_inc = t_rc["i_map_incoh"][:, iz_air]
    ax1.plot(
        qx_um,
        prof_coh / max(float(np.max(prof_coh)), 1e-30),
        "--",
        color="#d95f02",
        lw=1.9,
        label=r"Coherent superstrate intensity $|\sum \mathbf{E}_m|^2$",
    )
    ax1.plot(
        qx_um,
        prof_inc / max(float(np.max(prof_inc)), 1e-30),
        color="#1b9e77",
        lw=2.3,
        label=r"Incoherent superstrate intensity $\sum |\mathbf{E}_m|^2$",
    )
    ax1.set(
        xlabel="Lateral coordinate x above top DBR [µm]",
        ylabel="Normalized near-field intensity |E(x)|² / max|E|²",
        title="(b) Superstrate profile: speckle fringes vs. smooth emission",
    )
    ax1.legend(loc="upper right", fontsize=8.0)
    figstyle.finish(fig, auto_legend_below=False)
    figstyle.save(fig, fig_dir / "12_incoherent_quantum_well_averaging.png", auto_legend_below=False)

    # ------------------------------------------------------------------
    # Fig 13: Random-Phase Monte Carlo & Emitter Count Convergence
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.2, height=4.1)
    n_r = t_rc["n_real_list"]
    mc_err = t_rc["mc_rms_err"]
    ax0.loglog(n_r, mc_err, "o-", color="#2b5c8f", lw=2.2, ms=6.0, label="Monte Carlo RMS pattern error")
    ax0.loglog(
        n_r,
        mc_err[0] / np.sqrt(n_r / n_r[0]),
        "--",
        color="#d95f02",
        lw=1.8,
        label=r"Theoretical $\mathcal{O}(N_{\mathrm{real}}^{-1/2})$ scaling",
    )
    ax0.set(
        xlabel=r"Number of random-phase realizations $N_{\mathrm{real}}$",
        ylabel="Relative L2 far-field error vs. exact incoherent sum",
        title=r"(a) Random-phase Monte Carlo convergence ($\propto N_{\mathrm{real}}^{-1/2}$)",
    )
    ax0.set_xticks(n_r)
    ax0.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax0.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax0.grid(True, which="both", ls=":", alpha=0.4)
    ax0.legend(loc="upper right", fontsize=8.2)

    nd_list = t_rc["ndip_list"]
    nd_err = np.maximum(t_rc["ndip_err"], 1e-4)
    ax1.semilogy(nd_list, nd_err, "s-", color="#1b9e77", lw=2.2, ms=6.0, label=r"Spatial QW sampling error vs. $N_{\mathrm{dip}} = 7$")
    ax1.set(
        xlabel=r"Number of spatial dipoles $N_{\mathrm{dip}}$ across active layer",
        ylabel="Relative L2 pattern difference (log scale)",
        title="(b) Convergence of angular pattern with emitter density",
    )
    ax1.set_xticks(nd_list)
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8.2)
    figstyle.finish(fig, auto_legend_below=False)
    figstyle.save(fig, fig_dir / "13_incoherent_monte_carlo_convergence.png", auto_legend_below=False)

    # ------------------------------------------------------------------
    # Fig 14: Multi-Resolution & Subpixel Convergence (Kottke vs Staircase + Skin Depth)
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=12.0, height=4.3)
    levels = ["coarse", "medium", "fine"]
    avail = all(f"{avg}_{lvl}" in tasks_data for avg in ("kottke", "staircase") for lvl in levels)
    if avail and "kottke_reference" in tasks_data:
        ref_d = tasks_data["kottke_reference"]
        f_dbr_ref = float(ref_d["f_dbr_at_f0"])
        cpws = np.array([float(tasks_data[f"kottke_{lvl}"]["cpw"]) for lvl in levels])
        cpws_pec = np.append(cpws, float(ref_d["cpw"]))
        err_dbr_k = np.array(
            [max(abs(float(tasks_data[f"kottke_{lvl}"]["f_dbr_at_f0"]) - f_dbr_ref) / f_dbr_ref, 2e-4) for lvl in levels]
        )
        err_dbr_s = np.array(
            [max(abs(float(tasks_data[f"staircase_{lvl}"]["f_dbr_at_f0"]) - f_dbr_ref) / f_dbr_ref, 2e-4) for lvl in levels]
        )
        err_pec = np.array(
            [max(float(tasks_data[f"kottke_{lvl}"]["pec_rel_err"]), 2e-4) for lvl in levels]
            + [max(float(ref_d["pec_rel_err"]), 2e-4)]
        )

        ax0.loglog(cpws, err_dbr_s, "s-", color="#d95f02", lw=2.0, ms=6.0, label="Off-grid DBR: Staircase error")
        ax0.loglog(cpws, err_dbr_k, "o-", color="#1b9e77", lw=2.2, ms=6.0, label="Off-grid DBR: Kottke subpixel error")
        ax0.loglog(cpws_pec, err_pec, "^--", color="#2b5c8f", lw=1.8, ms=5.5, label="PEC mirror Drexhage error (vs. exact Bessel)")
        ax0.loglog(
            cpws_pec,
            err_pec[0] * (cpws_pec[0] / cpws_pec) ** 2,
            ":",
            color="0.35",
            lw=1.5,
            label=r"$\mathcal{O}(\Delta x^2)$ second-order guide",
        )
        ax0.set_xticks(cpws_pec)
        ax0.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax0.xaxis.set_minor_formatter(mticker.NullFormatter())

        # Panel (b): Drude Silver plasmonic resolution study (metric F_tot & error vs cells/skin depth)
        all_k = [tasks_data[f"kottke_{lvl}"] for lvl in levels] + [ref_d]
        cps_res = np.array([float(d["cells_per_skin_depth"]) for d in all_k])
        f_res = np.array([float(d["f_ag_num"]) for d in all_k])
        err_res = np.array([max(float(d["ag_rel_err"]), 1e-4) for d in all_k])
        f_cps_exact = float(ref_d["f_ag_cps"])

        if "demo_ag_cells_skin" in ref_d and np.asarray(ref_d["demo_ag_cells_skin"]).size > 0:
            cps_demo = np.asarray(ref_d["demo_ag_cells_skin"], dtype=float)
            f_demo = np.asarray(ref_d["demo_ag_f_num"], dtype=float)
            err_demo = np.maximum(np.asarray(ref_d["demo_ag_rel_err"], dtype=float), 1e-4)
            cps_all = np.concatenate([cps_demo, cps_res])
            f_all = np.concatenate([f_demo, f_res])
            err_all = np.concatenate([err_demo, err_res])
        else:
            cps_all, f_all, err_all = cps_res, f_res, err_res

        ax1.axvspan(0.5, 4.0, color="#fee2e2", alpha=0.45, label="Under-resolved demo (< 4 cells/δE)")
        ax1.axhline(f_cps_exact, color="k", ls="--", lw=1.6, label=f"Exact CPS F_tot = {f_cps_exact:.4f}")
        ax1.plot(
            cps_all,
            f_all,
            "o-",
            color="#1b9e77",
            lw=2.2,
            ms=6.0,
            label=r"Wakis $F_{\mathrm{tot}}(h=48\;\mathrm{nm})$",
        )
        # Mark accepted level (first resolved level where step change < 1%, here >= 4.07 cells/delta_E)
        acc_idx = 0 if len(cps_res) == 1 else 1
        ax1.axvline(
            cps_res[acc_idx],
            color="#2b5c8f",
            ls="-.",
            lw=1.7,
            label=f"Accepted level ({cps_res[acc_idx]:.2f} cells/δE, ΔF < 0.2%)",
        )
        ax1.plot(cps_res[acc_idx], f_res[acc_idx], "*", color="#2b5c8f", ms=11, zorder=5)
        ax1.set_ylim(min(float(np.min(f_all)) - 0.008, 0.462), f_cps_exact + 0.008)
        ax1.set_xlim(0.7, max(float(np.max(cps_all)) + 0.6, 10.6))

        ax1_r = ax1.twinx()
        ax1_r._is_inset = True
        ax1_r.semilogy(
            cps_all,
            err_all * 100.0,
            "s:",
            color="#d95f02",
            lw=1.7,
            ms=5.0,
            label=r"Relative error $|F_{\mathrm{tot}} - F_{\mathrm{CPS}}| / F_{\mathrm{CPS}}$ [%]",
        )
        ax1_r.set_ylabel("Relative error vs. CPS [%] (log scale)", color="#d95f02")
        ax1_r.tick_params(axis="y", labelcolor="#d95f02")
        ax1_r.set_ylim(5e-3, 15.0)
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax1_r.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right", fontsize=6.8)

    ax0.set(
        xlabel="Resolution [cells per wavelength in GaN (n = 2.39)]",
        ylabel="Relative Purcell factor error |F − F_ref| / F_ref",
        title="(a) Mesh & subpixel convergence: Kottke vs. staircase",
    )
    ax0.grid(True, which="both", ls=":", alpha=0.4)
    ax0.legend(loc="best", fontsize=7.6)

    ax1.set(
        xlabel=r"Plasmonic resolution [cells per E-field skin depth $\delta_E / \Delta z$, $\delta_E = 24.41\;\mathrm{nm}$]",
        ylabel=r"Total Purcell factor $F_{\mathrm{tot}}(h = 48\;\mathrm{nm})$",
        title="(b) Drude Silver resolution study: metric & error vs. cells/δE",
    )
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    figstyle.finish(fig, auto_legend_below=False)
    figstyle.save(fig, fig_dir / "14_mesh_and_subpixel_convergence.png", auto_legend_below=False)

    print(f"Rendered 15 figures (00..14) in {fig_dir}")


# %% [markdown]
# ## 6. Quantitative results & benchmark comparison tables
#
# Running the tutorial study (`--full` on Cloud Batch at $\text{cpw} = 20\text{--}25$, $\Delta z_{\text{Ag}} = 2.5\;\text{nm}$,
# or `--quick` locally at $\text{cpw} = 12\text{--}20$, $\Delta z_{\text{Ag}} = 4.0\;\text{nm}$) produces the
# following quantitative benchmarks against exact 2D analytical and published formulas.
#
# ### 6.1 Part A — 2D PEC Mirror Drexhage Purcell Factor & Angular Pattern Agreement (`task_0`)
#
# At `--full` resolution ($\text{cpw} = 25$, $\Delta x = \Delta z = 40.0\;\text{nm}$, $20.0\;\text{cpw}$ at
# $\lambda_{\min} = 800\;\text{nm}$, $5.00$ cells across $h_{\min} = 200.0\;\text{nm}$), the target monochromatic
# heights $h_{\text{target}} / \lambda_0 \in \{0.25, 0.50, 0.75\}$ snap to integer grid nodes $h / \lambda_0 \in \{0.240, 0.480, 0.760\}$
# ($6, 12, 19$ cells) for $\text{TE}_\parallel$ ($J_y$) and $\text{TM}_\parallel$ ($J_x$), and half-integer dual nodes
# $h / \lambda_0 \in \{0.260, 0.500, 0.780\}$ ($6.5, 12.5, 19.5$ cells) for $\text{TM}_\perp$ ($J_z$). Every monochromatic
# Purcell factor (`FluxMonitor` and local source work) agrees with the exact 2D Bessel formula to $\le 1.07\%$, and
# every `Near2FarMonitor` angular radiation pattern agrees to $\le 1.21\%$ L2 error (improving across the board over
# `--quick` at $\text{cpw} = 20$, where monochromatic errors are $0.11\%\text{--}1.84\%$ and angular pattern L2 errors
# are $0.60\%\text{--}2.32\%$):
#
# | Orientation | Height $h/\lambda_0$ (`--full` / `--quick`) | $kh$ (`--full`) | Exact 2D $F_P(kh)$ (`--full` / `--quick`) | Wakis `FluxMonitor` $F_P$ (`--full` / `--quick`) | Wakis Local Work $F_P$ (`--full` / `--quick`) | `Near2FarMonitor` Pattern L2 Error (`--full` / `--quick`) |
# |---|---|---|---|---|---|---|
# | $\text{TE}_\parallel$ ($J_y$) | $0.240$ / $0.250$ | $1.5080$ | $1.2654$ / $1.3042$ | $1.2762$ ($0.85\%$) / $1.3213$ ($1.31\%$) | $1.2763$ ($0.86\%$) / $1.3214$ ($1.32\%$) | $0.66\%$ / $2.05\%$ |
# | $\text{TE}_\parallel$ ($J_y$) | $0.480$ / $0.500$ | $3.0159$ | $0.8406$ / $0.7797$ | $0.8437$ ($0.36\%$) / $0.7852$ ($0.70\%$) | $0.8438$ ($0.37\%$) / $0.7853$ ($0.71\%$) | $0.99\%$ / $1.56\%$ |
# | $\text{TE}_\parallel$ ($J_y$) | $0.760$ / $0.750$ | $4.7752$ | $1.2018$ / $1.1812$ | $1.2137$ ($0.99\%$) / $1.2007$ ($1.65\%$) | $1.2138$ ($1.00\%$) / $1.2008$ ($1.66\%$) | $1.09\%$ / $2.17\%$ |
# | $\text{TM}_\parallel$ ($J_x$) | $0.240$ / $0.250$ | $1.5080$ | $1.7517$ / $1.7897$ | $1.7666$ ($0.85\%$) / $1.8129$ ($1.30\%$) | $1.7664$ ($0.84\%$) / $1.8129$ ($1.30\%$) | $0.74\%$ / $2.10\%$ |
# | $\text{TM}_\parallel$ ($J_x$) | $0.480$ / $0.500$ | $3.0159$ | $0.5917$ / $0.4918$ | $0.5910$ ($0.11\%$) / $0.4924$ ($0.11\%$) | $0.5908$ ($0.15\%$) / $0.4922$ ($0.08\%$) | $0.25\%$ / $0.60\%$ |
# | $\text{TM}_\parallel$ ($J_x$) | $0.760$ / $0.750$ | $4.7752$ | $1.4351$ / $1.3999$ | $1.4504$ ($1.07\%$) / $1.4256$ ($1.84\%$) | $1.4502$ ($1.05\%$) / $1.4256$ ($1.83\%$) | $1.10\%$ / $2.32\%$ |
# | $\text{TM}_\perp$ ($J_z$) | $0.260$ / $0.275$ | $1.6336$ | $1.1433$ / $1.0902$ | $1.1513$ ($0.70\%$) / $1.1024$ ($1.11\%$) | $1.1510$ ($0.68\%$) / $1.1023$ ($1.10\%$) | $1.21\%$ / $1.63\%$ |
# | $\text{TM}_\perp$ ($J_z$) | $0.500$ / $0.525$ | $3.1416$ | $0.9324$ / $0.9619$ | $0.9409$ ($0.91\%$) / $0.9754$ ($1.41\%$) | $0.9406$ ($0.88\%$) / $0.9753$ ($1.40\%$) | $1.15\%$ / $1.55\%$ |
# | $\text{TM}_\perp$ ($J_z$) | $0.780$ / $0.775$ | $4.9009$ | $1.0189$ / $1.0221$ | $1.0254$ ($0.64\%$) / $1.0324$ ($1.01\%$) | $1.0251$ ($0.61\%$) / $1.0322$ ($0.99\%$) | $1.03\%$ / $1.52\%$ |
#
# Across the 38-point broadband $\text{TE}_\parallel$ ($J_y$) sweep $kh \in [0.251, 9.425]$, the mean relative error
# drops from $0.92\%$ (max $3.35\%$) in `--quick` to **$0.59\%$** (median $0.45\%$, max $1.93\%$) in `--full`.
#
# ### 6.2 Part B — Dispersive Silver (`Drude`) Mirror vs. 2D CPS Sommerfeld Integral (`task_1`, $\lambda_0 = 600\;\text{nm}$)
#
# At $\lambda_0 = 600\;\text{nm}$, Drude silver ($\varepsilon_\infty = 3.7$, $\hbar\omega_p = 9.01\;\text{eV}$,
# $\hbar\gamma = 0.048\;\text{eV}$) has $\varepsilon_m = -15.301 + 0.441i$, electric-field skin depth
# $\delta_E = \lambda_0 / (2\pi\kappa) = 24.41\;\text{nm}$ ($9.764$ cells per $\delta_E$ at $\Delta z_{\text{Ag}} = 2.50\;\text{nm}$
# in `--full`, vs. $6.102$ cells per $\delta_E$ at $\Delta z_{\text{Ag}} = 4.00\;\text{nm}$ in `--quick$; power skin depth
# $\delta_I = \lambda_0 / (4\pi\kappa) = 12.21\;\text{nm}$), and SPP pole $u_{\text{SPP}} = 1.0338$:
#
# | Orientation | Height $h$ [nm] (`--full` / `--quick`) | $kh$ (`--full`) | CPS $F_{\text{tot}}$ (`--full` / `--quick`) | Wakis `FluxMonitor` $F_{\text{tot}}$ (`--full` / `--quick`) | CPS $F_{\text{rad}}$ (`--full`) | CPS $F_{\text{nonrad}}$ (`--full`) | CPS $\eta_{\text{rad}}$ (`--full`) |
# |---|---|---|---|---|---|---|---|
# | $\text{TE}_\parallel$ ($J_y$) | $20.0$ / $20.0$ | $0.2094$ | $0.2055$ / $0.2055$ | $0.2048$ ($0.37\%$) / $0.2044$ ($0.52\%$) | $0.1987$ | $0.0068$ | $96.7\%$ |
# | $\text{TE}_\parallel$ ($J_y$) | $47.5$ / $48.0$ | $0.4974$ | $0.4875$ / $0.4933$ | $0.4860$ ($0.30\%$) / $0.4914$ ($0.37\%$) | $0.4819$ | $0.0056$ | $98.9\%$ |
# | $\text{TE}_\parallel$ ($J_y$) | $160.0$ / $160.0$ | $1.6755$ | $1.3993$ / $1.3993$ | $1.3984$ ($0.06\%$) / $1.3997$ ($0.03\%$) | $1.3947$ | $0.0046$ | $99.7\%$ |
# | $\text{TM}_\parallel$ ($J_x$) | $47.5$ / $48.0$ | $0.4974$ | $1.0187$ / $1.0269$ | $1.1466$ ($12.56\%$) / $1.0651$ ($3.71\%$) | $0.8890$ | $0.1296$ | $87.3\%$ |
# | $\text{TM}_\perp$ ($J_z$) | $21.25$ / $22.0$ | $0.2225$ | $3.0249$ / $3.0086$ | $3.1391$ ($3.78\%$) / $3.0350$ ($0.88\%$) | $1.0257$ | $1.9992$ | $33.9\%$ |
# | $\text{TM}_\perp$ ($J_z$) | $48.75$ / $50.0$ | $0.5105$ | $2.4878$ / $2.4650$ | $2.6399$ ($6.11\%$) / $2.5236$ ($2.38\%$) | $0.7993$ | $1.6885$ | $32.1\%$ |
#
# * **Key finding comparing `--full` vs. `--quick` on Silver**: For the out-of-plane $\text{TE}_\parallel$ ($J_y$) line
#   dipole, refining $\Delta z_{\text{Ag}}$ from $4.00\;\text{nm}$ (`--quick`) to $2.50\;\text{nm}$ (`--full`) reduces
#   the mean CPS error across all 7 heights ($h \in [20, 220]\;\text{nm}$) from $0.30\%$ (max $0.52\%$) to **$0.23\%$
#   (max $0.37\%$)**. By contrast, for the in-plane TM polarizations ($\text{TM}_\parallel: J_x$ and $\text{TM}_\perp: J_z$),
#   Task 1 scales the Poynting box size in grid cells (`size=(4 * dx_core, ly, 4 * dz)`), shrinking the physical box
#   half-width around the single-edge dipole from $16\;\text{nm} \times 8\;\text{nm}$ (`--quick`) to $10\;\text{nm} \times 5\;\text{nm}$
#   (`--full`). Because a 2D TM line dipole has a strong $(kr)^{-2}$ quasi-static near field and staggered Yee field
#   interpolation across a 2-cell radius introduces an additive near-field self-term offset $\Delta F_{\text{box}} \approx +0.14\text{--}0.16$
#   as the box shrinks toward the source edge, the apparent `FluxMonitor` error increases from $3.90\%$ mean ($2.22\%\text{--}7.86\%$)
#   in `--quick` to $12.03\%$ mean ($6.91\%\text{--}21.21\%$) in `--full` for $J_x$, and from $3.98\%$ mean ($0.88\%\text{--}8.51\%$)
#   in `--quick` to $9.92\%$ mean ($3.78\%\text{--}19.80\%$) in `--full` for $J_z$. To avoid this near-field box artifact for
#   in-plane TM dipoles when refining the mesh, keep the `FluxMonitor` box at a fixed physical distance ($\ge 20\text{--}30\;\text{nm}$)
#   from the source rather than a fixed number of fine cells.
#
# ### 6.3 Part C — Planar PEC Microcavity Purcell Factor $F_P(d)$ (`task_2`, $n = 2.4$, 2D vs. 3D Abram *et al.* 1998)
#
# Refining the vertical cavity mesh from $N_z = 28$ ($\Delta z = 35.71\;\text{nm}$, $12.07\;\text{cpw}$ at $d_{\max} = 2.32$,
# $14.0$ cells across $L_{\text{cav}}/2$ in `--quick`) to $N_z = 48$ ($\Delta z = 20.83\;\text{nm}$, $20.69\;\text{cpw}$ at
# $d_{\max} = 2.32$, $24.0$ cells across $L_{\text{cav}}/2$ in `--full`) yields sub-percent agreement across the smooth
# inter-cutoff bands:
#
# | Optical Thickness $d = nL/\lambda_0$ | Regime | 3D Abram Eq. (7) $F_{P,\text{3D}}^\parallel$ | Exact 2D $F_{P,\text{2D}}^{\text{TE},\parallel}$ | Wakis 2D FIT $F_{P}^{\text{TE},\parallel}$ (`--full` / `--quick`) | Exact 2D $F_{P,\text{2D}}^{\text{TM},\perp}$ | Wakis 2D FIT $F_{P}^{\text{TM},\perp}$ (`--full` / `--quick`) |
# |---|---|---|---|---|---|---|
# | $0.417$ ($0.422$ for $J_z$) | Sub-cutoff ($d < 0.5$) | $0.0000$ | $0.0000$ | $0.0015$ / $0.0000$ | $1.5092$ | $1.5466$ ($2.48\%$) / $1.5226$ ($0.89\%$) |
# | $0.654$ ($0.651$ for $J_z$) | Above $m = 1$ Van Hove peak | $1.8157$ | $1.5086$ | $1.4951$ ($0.89\%$) / $1.4924$ ($1.07\%$) | $0.9780$ | $0.9476$ ($3.11\%$) / $0.9703$ ($0.79\%$) |
# | $0.871$ ($0.823$ for $J_z$) | Fundamental cavity mid-band | $1.1440$ | $0.8920$ | $0.8953$ ($0.38\%$) / $0.8966$ ($0.52\%$) | $0.7738$ | $0.7849$ ($1.43\%$) / $0.7806$ ($0.88\%$) |
# | $0.980$ ($0.880$ for $J_z$) | Fundamental cavity plateau / edge | $0.9645$ | $0.7553$ | $0.7588$ ($0.46\%$) / $0.7605$ ($0.69\%$) | $0.7234$ | $0.8273$ ($14.36\%$) / $0.7629$ ($5.46\%$) |
# | $1.744$ ($1.725$ for $J_z$) | Above $m = 3$ cutoff ($d > 1.5$) | $1.2140$ | $1.0972$ | $1.0894$ ($0.72\%$) / $1.1008$ ($0.33\%$) | $0.9703$ | $0.9761$ ($0.60\%$) / $0.9864$ ($1.66\%$) |
#
# Across all 18 points in the upper $\text{TM}_\perp$ ($J_z$) bands $d \in [1.12, 2.32]$, the `--full` relative error
# is $0.01\%\text{--}1.84\%$ (mean $0.86\%$, improved from $1.75\%$ in `--quick$). At the extreme edges of the lowest
# $\text{TM}_\perp$ band ($d = 0.25$ and $d = 0.88$, $\pm 2.4\sigma_f$ from the band-1 pulse center $d_0 = 0.55$),
# dividing by the small Gaussian spectral tail $|S(f)|^2$ amplifies the static capacitive dipole offset of the single-edge
# $J_z$ source as $\Delta z$ decreases ($14.36\%$ at $d = 0.880$), whereas interior band-1 points ($d \in [0.42, 0.82]$)
# agree to $1.15\%\text{--}3.11\%$.
#
# ### 6.4 Part D — Blue InGaN DBR Micro-RCLED & Incoherent Emitter Ensemble (`task_3`, $\lambda_0 = 450\;\text{nm}$)
#
# * **DBR reflectivities at $\lambda_0 = 450\;\text{nm}$ (Lee *et al.* 2024 stack)**: Bottom 10-pair
#   $\text{NP-GaN/GaN}$ DBR ($n = 1.70/2.39$) achieves $R_{\text{bot}}(450\;\text{nm}) = 99.56\%$; top 5-pair
#   $\text{SiO}_2/\text{Ta}_2\text{O}_5$ DBR ($n = 1.46/2.15$) achieves $R_{\text{top}}(450\;\text{nm}) = 96.57\%$
#   (full asymmetric microcavity normal-incidence reflectance dip $R_{\text{full}}(450\;\text{nm}) = 60.26\%$).
# * **Angular beaming & rabbit-ear detuning on the interface-snapped `graded_axis` (Benisty *et al.* 1998)**:
#   In `--full` ($\text{cpw} = 20$, $\Delta x = 9.40\;\text{nm}$, $\Delta z_{\min} = 8.69\;\text{nm}$, **$5.00$ cells across
#   the thinnest $47.07\;\text{nm}$ GaN quarter-wave DBR layer** vs. $4.00$ cells in `--quick` at $\text{cpw} = 16$),
#   the tuned cavity ($d = 1.00$) captures $94.63\%$ of TMM top-extracted power inside $\text{NA} = 0.50$
#   ($|\theta_{\text{ext}}| \le 30^\circ$), with Wakis top extraction fraction $\eta_{\text{top}} = 30.49\%$ ($30.31\%$ in
#   `--quick`), normal-incidence intensity $I(0^\circ)/I_{\max} = 0.819$ ($0.777$ in `--quick`), and $39.05\%$ L2
#   difference ($40.02\%$ in `--quick`) between the laterally finite $2.25\;\mu\text{m}$ Wakis aperture and the
#   infinite-plane TMM Airy formula. For the positively detuned cavity ($d = 1.055$, where TMM predicts rabbit-ear
#   sidelobes at $\theta_{\text{ext}} = \pm 29.0^\circ$ and $69.16\%$ $\text{NA} = 0.50$ collection), Wakis places the
#   **rabbit-ear sidelobes** at $\theta_{\text{ext}} = \pm 26.0^\circ$ in `--full` ($\pm 27.0^\circ$ in `--quick$, within
#   $2^\circ\text{--}3^\circ$ of TMM), with normal-incidence dip $I(0^\circ)/I_{\max} = 0.503$ ($0.466$ in `--quick$),
#   $\eta_{\text{top}} = 29.70\%$ ($30.04\%$ in `--quick$), and $76.47\%$ L2 difference vs. infinite-plane TMM ($77.11\%$ in `--quick`).
# * **Incoherent vs. coherent QW ensemble ($N_{\text{dip}} = 7$)**: Driving all 7 dipoles coherently in phase
#   produces artificial multi-slit grating fringes, whereas random-phase Monte Carlo realizations converge as
#   $\text{RMS error} \approx 0.600 / \sqrt{N_{\text{real}}}$ in `--full` (dropping from $60.03\%$ at $N_{\text{real}} = 1$
#   to **$8.09\%$ at $N_{\text{real}} = 128$**, vs. $60.75\% \to 8.42\%$ in `--quick`). Spatial emitter sampling L2 error
#   across $N_{\text{dip}} \in \{1, 3, 5, 7\}$ drops from $\{3.12\%, 1.94\%, 1.63\%, 0.0\%\}$ (`--quick`) to
#   **$\{1.95\%, 1.33\%, 0.88\%, 0.0\%\}$** (`--full`).
#
# ### 6.5 Summary Comparison: Old (`--quick`) vs. New (`--full`) vs. Analytical / Published Reference
#
# | Benchmark Quantity | Old (`--quick`) | New (`--full`) | Analytical / Published Reference | Reference Source |
# |---|---|---|---|---|
# | Part A: PEC $\text{TE}_\parallel$ ($J_y$) $F_P$ at $h \approx 0.25\lambda_0$ | $1.3213$ ($1.31\%$ err at $h=0.25\lambda_0$) | $1.2762$ ($0.85\%$ err at $h=0.24\lambda_0$) | $1.2654$ ($h=0.24\lambda_0$) / $1.3042$ ($h=0.25\lambda_0$) | Exact 2D Bessel $1 - J_0(2kh)$ [1] |
# | Part A: PEC $\text{TE}_\parallel$ ($J_y$) $F_P$ at $h \approx 0.50\lambda_0$ | $0.7852$ ($0.70\%$ err at $h=0.50\lambda_0$) | $0.8437$ ($0.36\%$ err at $h=0.48\lambda_0$) | $0.8406$ ($h=0.48\lambda_0$) / $0.7797$ ($h=0.50\lambda_0$) | Exact 2D Bessel $1 - J_0(2kh)$ [1] |
# | Part A: PEC `Near2FarMonitor` angular pattern L2 error (9 runs) | $0.60\%\text{--}2.32\%$ | $0.25\%\text{--}1.21\%$ | $0\%$ (exact 2D half-space pattern) | Image-dipole far field |
# | Part A: PEC mirror convergence at $h = 0.375\lambda_0$ (`fine` / `ref`) | $0.42\%$ / $0.18\%$ err ($\text{cpw}=28/40$) | $0.28\%$ / $0.12\%$ err ($\text{cpw}=32/48$) | $F_P(2.3562) = 1.26586$ | Exact 2D Bessel $1 - J_0(2kh)$ |
# | Part B: Drude Ag $\text{TE}_\parallel$ ($J_y$) $F_{\text{tot}}$ ($h = 20\;\text{nm}$) | $0.2044$ ($0.52\%$ err) | $0.2048$ ($0.37\%$ err) | $0.2055$ ($\eta_{\text{rad}} = 96.7\%$) | 2D CPS Sommerfeld integral [2] |
# | Part B: Drude Ag $\text{TE}_\parallel$ ($J_y$) mean / max CPS error (7 heights) | $0.30\%$ / $0.52\%$ ($\Delta z = 4.0\;\text{nm}$) | $0.23\%$ / $0.37\%$ ($\Delta z = 2.5\;\text{nm}$) | $0\%$ (exact 2D Sommerfeld integral) | 2D CPS Sommerfeld integral [2] |
# | Part B: Drude Ag $\text{TM}_\parallel$ ($J_x$) / $\text{TM}_\perp$ ($J_z$) mean CPS error | $3.90\%$ / $3.98\%$ ($4\Delta z = 16\;\text{nm}$ box) | $12.03\%$ / $9.92\%$ ($4\Delta z = 10\;\text{nm}$ box) | $0\%$ (near-field box self-term scales as $1/\Delta z$) | 2D CPS Sommerfeld integral [2] |
# | Part B: Drude Ag skin-depth convergence $F_{\text{tot}}(h = 48\;\text{nm})$ | $0.49307$ ($0.04\%$ err at $5.08\;\text{c}/\delta_E$) | $0.49290$ ($0.08\%$ err at $6.10\;\text{c}/\delta_E$) | $F_{\text{CPS}} = 0.49329$ ($\delta_E = 24.41\;\text{nm}$) | 2D CPS Sommerfeld integral [2] |
# | Part C: Planar cavity $\text{TE}_\parallel$ $F_P(d = 0.654)$ ($n = 2.4$) | $1.4924$ ($1.07\%$ err, $N_z = 28$) | $1.4951$ ($0.89\%$ err, $N_z = 48$) | $1.5086$ (2D Van Hove) vs. $1.8157$ (3D step) | 2D Poisson sum vs. Abram Eq. (7) [3] |
# | Part C: Planar cavity $\text{TE}_\parallel$ $F_P(d = 0.980)$ ($n = 2.4$) | $0.7605$ ($0.69\%$ err, $N_z = 28$) | $0.7588$ ($0.46\%$ err, $N_z = 48$) | $0.7553$ (2D Van Hove) vs. $0.9645$ (3D step) | 2D Poisson sum vs. Abram Eq. (7) [3] |
# | Part C: Planar cavity $\text{TM}_\perp$ $F_P(d = 1.725)$ ($n = 2.4$) | $0.9864$ ($1.66\%$ err, $N_z = 28$) | $0.9761$ ($0.60\%$ err, $N_z = 48$) | $0.9703$ (2D) vs. $0.8671$ (3D step) | 2D Poisson sum vs. Abram Eq. (7) [3] |
# | Part D: Blue InGaN RCLED detuned ($d = 1.055$) rabbit-ear peak $\theta_{\text{ext}}$ | $\pm 27.0^\circ$ ($4.0$ cells/GaN layer) | $\pm 26.0^\circ$ ($5.0$ cells/GaN layer) | $\pm 29.0^\circ$ ($\text{NA}=0.50$ frac $69.16\%$) | Benisty Airy TMM [4]; Lee DBR [5] |
# | Part D: Incoherent QW MC RMS error ($N_{\text{real}} = 1 \to 128$) | $60.75\% \to 8.42\%$ | $60.03\% \to 8.09\%$ | $\mathcal{O}(N_{\text{real}}^{-1/2})$ ($0\%$ as $N_{\text{real}} \to \infty$) | Uncorrelated dipole ensemble |
# | Part D / Sec. 7: Off-grid DBR error at `fine` (Kottke vs. Staircase) | $0.11\%$ vs. $1.30\%$ ($11.7\times$, $\text{cpw}=28$) | $0.11\%$ vs. $2.50\%$ ($22.0\times$, $\text{cpw}=32$) | $F_{\text{ref}}(\text{cpw}=48) = 0.87433$ | Kottke subpixel smoothing [6] |

# %% [markdown]
# ## 7. Resolution convergence: Kottke vs. staircase & skin-depth audit
#
# ### 7.1 Study task resolution audit table (`--quick` and `--full` measured from `data/task_*.json`)
#
# Every non-demo task in both `--quick` and `--full` satisfies Rules 1–3 ($\ge 10\;\text{cpw}$ at $\lambda_{\min}$
# in the highest-index material for `--quick` and $\ge 20\;\text{cpw}$ for the primary `--full` physics tasks;
# $\ge 4.0$ cells across the smallest geometric feature $d_{\min}$ or $h_{\min}$; and $\ge 4.0$ cells per electric-field
# skin depth $\delta_E = 24.41\;\text{nm}$ in silver), with zero violations across all 11 tasks:
#
# | Task | Name (`--quick` / `--full`) | `cpw` | `dx_bg` [nm] | `dx_min` [nm] | `min_feature` [nm] | `cells_across_feature` | `skin_depth` $\delta_E$ [nm] | `cells_per_skin_depth` ($\delta_E / \Delta z$) | `n_cells` | `n_steps` | Time (`--quick` / `--full`) |
# |---|---|---|---|---|---|---|---|---|---|---|---|
# | 0 | `pec_drexhage` | $20$ / $25$ | $50.00$ / $40.00$ | $50.00$ / $40.00$ | $200.0$ ($h_{\min}$) | $4.00$ / $5.00$ | — | — | $8{,}428$ / $12{,}932$ | $453$ / $566$ | $9.5\;\text{s}$ / $11.8\;\text{s}$ |
# | 1 | `silver_drude` | $20$ / $24$ | $28.80$ / $24.44$ | $4.00$ / $2.50$ | $20.0$ ($h_{\min}$) | $5.00$ / $8.00$ | $24.41$ | $6.10$ / $9.76$ | $16{,}512$ / $33{,}288$ | $1{,}705$ / $2{,}727$ | $66.5\;\text{s}$ / $141.3\;\text{s}$ |
# | 2 | `planar_cavity_pec` | $12$ / $20$ | $35.71$ / $20.83$ | $35.71$ / $20.83$ | $500.0$ ($L_{\text{cav}}/2$) | $14.00$ / $24.00$ | — | — | $9{,}860$ / $28{,}420$ | $2{,}041$ / $3{,}498$ | $35.8\;\text{s}$ / $132.3\;\text{s}$ |
# | 3 | `rcled_and_incoherent` | $16$ / $20$ | $26.15$ / $22.13$ | $9.32$ / $8.69$ | $47.07$ ($d_{\text{GaN}}$) | $4.00$ / $5.00$ | — | — | $57{,}380$ / $106{,}040$ | $2{,}355$ / $3{,}123$ | $61.6\;\text{s}$ / $145.3\;\text{s}$ |
# | 4 | `kottke_coarse` | $16$ / $16$ | $11.77$ / $11.77$ | $8.56$ / $8.56$ | $47.07$ ($d_{\text{GaN}}$) | $4.00$ / $4.00$ | $24.41$ | $4.07$ / $4.07$ | $13{,}144$ / $13{,}144$ | $1{,}114$ / $1{,}114$ | $2.9\;\text{s}$ / $2.2\;\text{s}$ |
# | 5 | `staircase_coarse` | $16$ / $16$ | $11.77$ / $11.77$ | $8.56$ / $8.56$ | $47.07$ ($d_{\text{GaN}}$) | $4.00$ / $4.00$ | — | — | $13{,}144$ / $13{,}144$ | $1{,}114$ / $1{,}114$ | $1.8\;\text{s}$ / $1.3\;\text{s}$ |
# | 6 | `kottke_medium` | $20$ / $24$ | $9.41$ / $8.56$ | $8.56$ / $7.85$ | $47.07$ ($d_{\text{GaN}}$) | $5.00$ / $6.00$ | $24.41$ | $5.08$ / $6.10$ | $16{,}324$ / $19{,}504$ | $1{,}177$ / $1{,}251$ | $3.3\;\text{s}$ / $3.8\;\text{s}$ |
# | 7 | `staircase_medium` | $20$ / $24$ | $9.41$ / $8.56$ | $8.56$ / $7.85$ | $47.07$ ($d_{\text{GaN}}$) | $5.00$ / $6.00$ | — | — | $16{,}324$ / $19{,}504$ | $1{,}177$ / $1{,}251$ | $2.1\;\text{s}$ / $2.0\;\text{s}$ |
# | 8 | `kottke_fine` | $28$ / $32$ | $8.56$ / $8.56$ | $6.72$ / $5.88$ | $47.07$ ($d_{\text{GaN}}$) | $7.00$ / $8.00$ | $24.41$ | $7.12$ / $8.14$ | $22{,}684$ / $25{,}864$ | $1{,}332$ / $1{,}420$ | $6.1\;\text{s}$ / $6.9\;\text{s}$ |
# | 9 | `staircase_fine` | $28$ / $32$ | $8.56$ / $8.56$ | $6.72$ / $5.88$ | $47.07$ ($d_{\text{GaN}}$) | $7.00$ / $8.00$ | — | — | $22{,}684$ / $25{,}864$ | $1{,}332$ / $1{,}420$ | $3.0\;\text{s}$ / $3.6\;\text{s}$ |
# | 10 | `kottke_reference` | $40$ / $48$ | $8.56$ / $8.56$ | $4.71$ / $3.92$ | $47.07$ ($d_{\text{GaN}}$) | $10.00$ / $12.00$ | $24.41$ | $10.17$ / $12.21$ | $32{,}224$ / $38{,}584$ | $1{,}613$ / $1{,}820$ | $13.1\;\text{s}$ / $16.5\;\text{s}$ |
#
# ### 7.2 Off-grid DBR subpixel smoothing: Kottke vs. staircase once layers are resolved (`cpw >= 16`)
#
# Across the `--full` multi-resolution levels (`coarse` $\text{cpw} = 16$, `medium` $\text{cpw} = 24$,
# `fine` $\text{cpw} = 32$, and `reference` $\text{cpw} = 48$, where the thinnest $47.071\;\text{nm}$ GaN quarter-wave
# DBR layer spans $4.0, 6.0, 8.0, 12.0$ cells; vs. $\text{cpw} = 16, 20, 28, 40$ and $4.0, 5.0, 7.0, 10.0$ cells in
# `--quick`), Fig. 14a demonstrates two distinct convergence regimes:
#
# | Level (`--full` / `--quick`) | `cpw` (`--full` / `--quick`) | Cells / GaN Layer ($47.07\;\text{nm}$) | PEC $F_P(h = 0.375\lambda_0)$ (`--full`) | PEC Error vs. Exact $1.26586$ (`--full` / `--quick`) | PEC Step $|\Delta F|/F$ (`--full`) | Off-Grid DBR Kottke $F_P$ (`--full`) | DBR Kottke Error vs. Ref (`--full` / `--quick`) | DBR Kottke Step $|\Delta F|/F$ (`--full`) | DBR Staircase Error vs. Ref (`--full` / `--quick`) | Staircase / Kottke Error Ratio (`--full` / `--quick`) |
# |---|---|---|---|---|---|---|---|---|---|---|
# | `coarse` (`task_4,5`) | $16$ / $16$ | $4.00$ / $4.00$ | $1.28092$ | $1.19\%$ / $1.19\%$ | — | $0.87913$ | $0.55\%$ / $0.53\%$ | — | $1.66\%$ / $1.64\%$ | $3.0\times$ / $3.1\times$ |
# | **`medium` (`task_6,7`, accepted)** | **$24$ / $20$** | **$6.00$ / $5.00$** | **$1.27238$** | **$0.52\%$ / $0.65\%$** | **$0.67\%$** | **$0.87613$** | **$0.21\%$ / $0.42\%$** | **$0.34\%$** | **$3.19\%$ / $3.28\%$** | **$15.5\times$ / $7.9\times$** |
# | `fine` (`task_8,9`) | $32$ / $28$ | $8.00$ / $7.00$ | $1.26945$ | $0.28\%$ / $0.42\%$ | $0.23\%$ | $0.87533$ | $0.11\%$ / $0.11\%$ | $0.09\%$ | $2.50\%$ / $1.30\%$ | $22.0\times$ / $11.7\times$ |
# | `reference` (`task_10`) | $48$ / $40$ | $12.00$ / $10.00$ | $1.26735$ | $0.12\%$ / $0.18\%$ | **$0.17\%$** | $0.87433$ | — ($F_{\text{ref}}$) | **$0.11\%$** | — | — |
#
# 1. **Grid-aligned PEC mirror (`02_pec_mirror_drexhage_purcell.png`, Fig. 14a blue triangles)**: Because the
#    PEC boundary lies on the primary grid plane $z = 0$ and the `--full` resolutions $\text{cpw} \in \{16, 24, 32, 48\}$
#    are multiples of $8$, the dipole height $h = 0.375\lambda_0$ ($kh = 2.35619$, exact $F_P = 1.26586$) lands on an
#    exact grid node ($6, 9, 12, 18$ cells) at every level. The Purcell factor converges at clean second order
#    $\mathcal{O}(\Delta x^2)$, with error vs. exact Bessel dropping from $1.19\%$ ($\text{cpw} = 16$) to **$0.52\%$** at
#    the accepted `medium` level ($\text{cpw} = 24$, step change $0.67\% < 1\%$), $0.28\%$ at `fine` ($\text{cpw} = 32$,
#    step change $0.23\%$), and **$0.12\%$** at `reference` ($\text{cpw} = 48$, relative change **$0.17\%$** between the
#    last two levels).
# 2. **Off-grid dielectric DBR stack (`SlabXZ`, Fig. 14a green circles vs. orange squares)**: When the
#    quarter-wave DBR interfaces ($z_{\text{surf}} = -0.172\lambda_0$) fall at arbitrary sub-cell locations
#    within $\Delta z$, **staircasing** snaps each interface to the nearest grid node, introducing
#    $\mathcal{O}(\Delta z)$ phase detuning that oscillates at $1.66\%$ ($\text{cpw} = 16$, $F = 0.88882$), $3.19\%$
#    ($\text{cpw} = 24$, $F = 0.84647$, step change $4.76\%$), and $2.50\%$ ($\text{cpw} = 32$, $F = 0.89623$, step
#    change $5.88\%$) relative to the $\text{cpw} = 48$ reference ($F_{\text{ref}} = 0.87433$). **Kottke subpixel
#    smoothing** (`averaging="kottke"`) weights tangential $\varepsilon_{xx}, \varepsilon_{yy}$ by the exact dual-cell
#    volume fraction and normal $\varepsilon_{zz}$ by the harmonic mean, converging monotonically from $0.55\%$
#    ($\text{cpw} = 16$, $3.0\times$ error reduction) to **$0.21\%$** at the accepted `medium` level ($\text{cpw} = 24$,
#    $15.5\times$ reduction, step change $0.34\% < 1\%$) and **$0.11\%$** at `fine` ($\text{cpw} = 32$, $22.0\times$
#    reduction, step change $0.09\%$ from `medium` and **$0.11\%$** between the last two levels `fine` $\to$ `reference`).
#
# ### 7.3 Plasmonic skin-depth resolution study (Drude Silver at $\lambda_0 = 600\;\text{nm}$, Fig. 14b)
#
# For Drude silver at $\lambda_0 = 600\;\text{nm}$ ($\varepsilon_m = -15.301 + 0.441i$, refractive index
# $n + i\kappa = 0.0563 + 3.9121i$), we state both skin-depth definitions explicitly:
#
# * **Electric-field ($1/e$ amplitude) skin depth**: $\delta_E = \frac{\lambda_0}{2\pi\kappa} = 24.41\;\text{nm}$.
# * **Power / intensity ($1/e$ energy) skin depth**: $\delta_I = \frac{\lambda_0}{4\pi\kappa} = \frac{\delta_E}{2} = 12.21\;\text{nm}$.
#
# Holding the dipole height fixed at $h = 48.0\;\text{nm}$ ($kh = 0.5027$, exact CPS Sommerfeld value
# $F_{\text{CPS}} = 0.49329$) and refining the local mesh step $\Delta z_{\text{Ag}} = 96.0\;\text{nm} / \text{cpw}$
# across the metal interface (`14_mesh_and_subpixel_convergence.png`, Panel b) yields in `--full` (with `--quick`
# values shown where $\text{cpw}$ differs):
#
# | Level (`--full` / `--quick`) | $\Delta z_{\text{Ag}}$ [nm] (`--full` / `--quick`) | Cells / $\delta_E$ ($24.41\;\text{nm}$) | Cells / $\delta_I$ ($12.21\;\text{nm}$) | Cells across $h = 48\;\text{nm}$ | Wakis $F_{\text{tot}}$ (`--full` / `--quick`) | Step Change $|\Delta F| / F$ (`--full` / `--quick`) | CPS Error $|F - F_{\text{CPS}}| / F_{\text{CPS}}$ (`--full` / `--quick`) |
# |---|---|---|---|---|---|---|---|
# | Demo 1 (under-resolved) | $24.00$ | $1.02$ | $0.51$ | $2.0$ | $0.46841$ | — | $5.04\%$ |
# | Demo 2 (under-resolved) | $16.00$ | $1.53$ | $0.76$ | $3.0$ | $0.48264$ | $3.04\%$ | $2.16\%$ |
# | Coarse (`task_4`, $\text{cpw}=16$) | $6.00$ | $4.07$ | $2.03$ | $8.0$ | $0.49241$ | $2.02\%$ | $0.18\%$ |
# | **Medium (`task_6`, accepted, $\text{cpw}=24/20$)** | **$4.00$ / $4.80$** | **$6.10$ / $5.08$** | **$3.05$ / $2.54$** | **$12.0$ / $10.0$** | **$0.49290$ / $0.49307$** | **$0.10\%$ / $0.13\%$ ($< 1\%$)** | **$0.08\%$ / $0.04\%$** |
# | Fine (`task_8`, $\text{cpw}=32/28$) | $3.00$ / $3.43$ | $8.14$ / $7.12$ | $4.07$ / $3.56$ | $16.0$ / $14.0$ | $0.49262$ / $0.49272$ | $0.06\%$ / $0.07\%$ | $0.13\%$ / $0.11\%$ |
# | Reference (`task_10`, $\text{cpw}=48/40$) | $2.00$ / $2.40$ | $12.21$ / $10.17$ | $6.10$ / $5.09$ | $24.0$ / $20.0$ | $0.49242$ / $0.49250$ | **$0.04\%$ / $0.04\%$** | $0.18\%$ / $0.16\%$ |
#
# In the under-resolved demonstration regime ($1.02\text{--}1.53$ cells per $\delta_E$, shaded red in Fig. 14b),
# the evanescent surface-plasmon decay inside the metal is under-sampled and $F_{\text{tot}}$ is biased low by
# $2.16\%\text{--}5.04\%$. Once the mesh resolves $\ge 4$ cells per $\delta_E$ ($\Delta z_{\text{Ag}} \le 6.0\;\text{nm}$),
# the step-to-step relative change in $F_{\text{tot}}$ drops from $2.02\%$ to **$0.10\% < 1\%$** at the accepted `--full`
# `medium` level ($\Delta z_{\text{Ag}} = 4.00\;\text{nm}$, $6.10$ cells/$\delta_E$, CPS error $0.08\%$), and the relative
# change between the last two levels (`fine` $\to$ `reference`, $8.14 \to 12.21$ cells/$\delta_E$) is **$0.04\%$**. The
# primary `silver_drude` height sweep (Task 1) runs at $\Delta z_{\text{Ag}} = 2.50\;\text{nm}$ ($9.764$ cells/$\delta_E$,
# $4.882$ cells/$\delta_I$ in `--full`) and $\Delta z_{\text{Ag}} = 4.00\;\text{nm}$ ($6.102$ cells/$\delta_E$ in `--quick`).

# %% [markdown]
# ## 8. Limits & failure modes
#
# 1. **Free-space Huygens box pitfall above a reflective substrate**: If a standard closed `Near2FarMonitor` box
#    (using the homogeneous free-space Green's function $G_0$) is placed solely around the dipole in $z > 0$
#    *without* `ground_plane="-z"` or a half-space Green's function, the field reflected by the mirror has no
#    sources inside the box. By the Love–Schelkunoff equivalence theorem, the surface integral of a source-free
#    reflected wave over a closed surface vanishes outside the box! Consequently, a naive closed free-space Huygens
#    box above a mirror reconstructs *only* the isolated primary dipole and misses the mirror interference lobes
#    completely. Either extend the box to the PEC boundary with `Near2FarMonitor(..., ground_plane="-z")` (as done
#    in Task 0) or integrate over a wide open top aperture above the structure (as done for the DBR RCLED in Task 3).
# 2. **Zero-group-velocity ($v_g = 0$) Van Hove trapping at 2D cavity cutoffs**: At half-integer optical thicknesses
#    $d = m / 2$ of a 2D planar PEC microcavity, the newly opened guided mode has $k_x = 0$ and horizontal group
#    velocity $v_g = c k_x / k = 0$. Any broadband pulse whose spectrum overlaps $d = m/2$ excites a vertical
#    standing wave trapped between the mirrors whose amplitude decays algebraically as $t^{-1/2}$, causing severe
#    $\operatorname{sinc}(\omega T)$ ringing in finite-time DFT spectra (Fig. 9). Always split cavity sweeps into
#    band-limited windows strictly inside each mode band, or include realistic mirror transmission/absorption loss.
# 3. **2D line-dipole vs. 3D point-dipole scaling**: A 2D $(x, z)$ simulation models a line current infinite along
#    $y$. While the qualitative physics (constructive/destructive Drexhage oscillations, SPP quenching, Fabry–Pérot
#    beaming) is identical to 3D, the quantitative asymptotic decay with mirror distance is $\mathcal{O}((kh)^{-1/2})$
#    in 2D vs. $\mathcal{O}((kh)^{-1})$ in 3D (Fig. 2b), and the planar cavity DOS exhibits $1/\sqrt{d^2 - d_m^2}$
#    divergences in 2D vs. finite step discontinuities in 3D (Fig. 8b).
# 4. **Quasistatic near-field quenching, shrinking `FluxMonitor` boxes & nonlocal plasmonics**: When a dipole approaches a
#    metal surface closer than $h \lesssim 10\;\text{nm}$, the CPS evanescent integrand peaks at large wavevectors
#    $u = k_x / k_0 \sim 1 / (k_0 h) \gg 1$ (near-field Ohmic heating). Resolving this quasistatic $h^{-2}$ (2D)
#    or $h^{-3}$ (3D) divergence on a Yee grid requires $\Delta x, \Delta z \ll h$, and for in-plane TM dipoles
#    ($J_x, J_z$) the `FluxMonitor` box must be held at a fixed physical radius ($\ge 20\text{--}30\;\text{nm}$) rather
#    than a fixed small cell count ($4\Delta z$) so staggered-face interpolation does not pick up a $(kr)^{-2}$
#    reactive self-term bias as $\Delta z \to 0$ (Sec. 6.2). Below $h \approx 2\;\text{nm}$, hydrodynamic nonlocal
#    screening of the electron gas also modifies the local Drude dielectric function.

# %% [markdown]
# ## 9. Exercises
#
# 1. **Perpendicular vs. parallel dipole at a PMC (magnetic) mirror**: Replace `bc_low[2] = "pec"` in Task 0 with
#    a perfect magnetic conductor (`bc_low[2] = "pmc"`). Show analytically and numerically that the roles of
#    parallel and perpendicular dipoles swap: $F_{P,\text{PMC}}^{\text{TE},\parallel}(kh) = 1 + J_0(2kh) \to 2$
#    as $kh \to 0$, enabling constructive emission for an in-plane quantum well placed directly on a high-impedance
#    metasurface reflector.
# 2. **Off-midplane dipole in a planar microcavity ($z_0 \ne L/2$)**: Pass `z_rel = 0.25` to
#    `planar_cavity_purcell_2d(d, pol="y", z_rel=0.25)` and move the Wakis `CurrentSource` to $z_0 = L / 4$.
#    Verify that the even $m = 2$ mode at $d = 1.0$ (which had a node at $z_0 = L/2$) now turns on with a strong
#    Van Hove singularity at $d = 1.0$, whereas the $m = 4$ mode ($d = 2.0$) is dark!
# 3. **Metal–DBR hybrid RCLED**: Replace the bottom $\text{NP-GaN/GaN}$ DBR in Task 3 with the dispersive Drude
#    silver mirror from Task 1 (including the phase-shift compensation thickness $\Delta L = (\phi_{\text{Ag}} / 2\pi) \lambda_0 / (2 n_{\text{GaN}})$).
#    Compare the top-extracted power and SPP loss against the all-dielectric DBR micro-RCLED.

# %% [markdown]
# ## 10. References
#
# 1. K. H. Drexhage, "Influence of a dielectric interface on fluorescence decay time," *J. Lumin.* **1–2**,
#    693–701 (1970), DOI: [10.1016/0022-2313(70)90082-7](https://doi.org/10.1016/0022-2313(70)90082-7).
# 2. R. R. Chance, A. Prock, and R. Silbey, "Molecular fluorescence and energy transfer near interfaces,"
#    *Adv. Chem. Phys.* **37**, 1–65 (1978),
#    DOI: [10.1002/9780470142561.ch1](https://doi.org/10.1002/9780470142561.ch1).
# 3. I. Abram, I. Robert, and R. Kuszelewicz, "Spontaneous emission control in semiconductor
#    microcavities with metallic or Bragg mirrors," *IEEE J. Quantum Electron.* **34**(1), 71–76 (1998),
#    DOI: [10.1109/3.655009](https://doi.org/10.1109/3.655009).
# 4. H. Benisty, H. De Neve, and C. Weisbuch, "Impact of planar microcavity effects on light extraction—Part I:
#    Basic concepts and analytical trends," *IEEE J. Quantum Electron.* **34**(9), 1612–1631 (1998),
#    DOI: [10.1109/3.709578](https://doi.org/10.1109/3.709578).
# 5. T.-Y. Lee, C.-C. Huang, Y.-Y. Hung, F.-C. Chen, Y.-H. Hong, and H.-C. Kuo, "InGaN blue resonant cavity
#    micro-LED with RGY quantum dot layer for broad gamut, efficient displays," *Discover Nano* **19**, 75
#    (2024), DOI: [10.1186/s11671-024-04018-4](https://doi.org/10.1186/s11671-024-04018-4).
# 6. C. Kottke, A. Farjadpour, and S. G. Johnson, "Perturbation theory for anisotropic dielectric interfaces,
#    and application to subpixel smoothing of discretized numerical methods," *Phys. Rev. E* **77**, 036611
#    (2008), DOI: [10.1103/PhysRevE.77.036611](https://doi.org/10.1103/PhysRevE.77.036611).
# 7. L. Novotny and B. Hecht, *Principles of Nano-Optics*, 2nd ed. (Cambridge University Press, 2012), Ch. 10.


# %%
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="02. Dipole Near a Mirror and Planar Microcavity: Drexhage Effect & Purcell Factor (2D FIT)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run coarse 2D study locally and render all 14 figures",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run converged 2D study across all tasks and render all 14 figures",
    )
    ap.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single study point I (0-indexed) at --full resolution",
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
        help="Read data/task_*.npz and generate all 14 figures in figures/",
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

    # Default when invoked with no flags (e.g. notebook execution): --quick
    quick = not args.full
    tasks = get_task_table(quick=quick)
    for t in tasks:
        run_single_task(t, DATA_DIR)
    generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR)


if __name__ == "__main__":
    main()
