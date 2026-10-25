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
# # Tutorial 7.5 · Partially coherent light through a double slit
#
# **Abstract.** Classical electromagnetic time-domain solvers evolve a single
# deterministic realization of Maxwell's equations, so any collection of sources
# driven simultaneously in one run is inherently 100% phase-coherent. Modelling
# **partially coherent light**—such as thermal emission, LEDs, multimode vertical-cavity
# surface-emitting lasers (VCSELs), or synchrotron undulator radiation—requires
# mapping second-order statistical optics onto an **incoherent power superposition**
# of uncorrelated field realizations. In this tutorial we simulate Young's
# double-slit experiment in 2D TM (`Ez`, `Hx`, `Hy`) using Wakis's Finite
# Integration Technique (FIT) solver and demonstrate three complementary
# routes to optical coherence:
# (1) **spatial coherence from an extended primary incoherent strip source** of
# width $D$ at distance $L_1$, reproducing the **van Cittert–Zernike theorem**
# $\mu_{12}(D) = \operatorname{sinc}(\pi D d / (\lambda_0 L_1))$ including
# complete fringe washout at $D_0 = \lambda_0 L_1 / d$ and **fringe phase
# reversal** ($\mu_{12} < 0$, dark central minimum at $x=0$) in the secondary
# lobe;
# (2) **Gaussian Schell-model (GSM) beams** synthesized via **Wolf's coherent-mode
# (Mercer) decomposition** into orthonormal Hermite–Gaussian eigenmodes
# (Starikov & Wolf, *J. Opt. Soc. Am.* **72**, 923, 1982), reproducing the
# Gaussian visibility law $\mathcal{V}(\sigma_\mu) = \exp(-d^2 / (2\sigma_\mu^2))$; and
# (3) **broadband temporal coherence** via pulse-normalized `DFTMonitor` spectral
# quadrature, reproducing the **Wiener–Khinchin** Gaussian fringe envelope
# $\exp(-\Delta L^2 / (2 L_c^2))$ as a function of optical path difference
# $\Delta L(x) = r_1(x) - r_2(x)$. Finally, we compare an ideal PEC screen
# against a **dispersive Drude silver ($\text{Ag}$) screen** at $\lambda_0 = 500\;\text{nm}$
# (skin depth $\delta_{\text{skin}} = 28.8\;\text{nm}$) and audit convergence
# with respect to ensemble size ($N_s$, $M$) and spatial mesh resolution ($\text{cpw}$).
#
# ---
#
# ## Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. Explain why summing complex electric fields $\big|\sum_n E_z^{(n)}(\mathbf{r})\big|^2$
#    in a single deterministic FDTD/FIT simulation produces unphysical cross-term
#    interference, and construct the physically correct **incoherent power sum**
#    $\sum_n w_n \big|E_z^{(n)}(\mathbf{r})\big|^2$.
# 2. Formulate and verify the **2D van Cittert–Zernike theorem** for an extended
#    incoherent strip source of width $D$, predicting both the first visibility
#    null $D_0 = \lambda_0 L_1 / d$ and the negative-coherence phase-inverted
#    secondary regime $D \in (D_0, 2 D_0)$.
# 3. Synthesize a **Gaussian Schell-Model (GSM)** partially coherent beam using
#    the exact **Starikov & Wolf (1982) Mercer coherent-mode expansion** and
#    Mehler's Hermite polynomial kernel without random Monte Carlo phase noise.
# 4. Extract broadband **temporal coherence** envelopes from a single pair of
#    pulse-excited `DFTMonitor` runs using the **Wiener–Khinchin theorem**, and
#    quantify finite-skin-depth effects in a dispersive silver screen.
#
# ## Contents
#
# 1. [Physical background & second-order coherence theory](#1.-Physical-background-&-second-order-coherence-theory)
# 2. [Geometry, parameters & 2D FIT domain setup](#2.-Geometry,-parameters-&-2D-FIT-domain-setup)
# 3. [Method A: Extended incoherent strip source (van Cittert–Zernike)](#3.-Method-A:-Extended-incoherent-strip-source-(van-Cittert–Zernike))
# 4. [Method B: Gaussian Schell-Model beam via Mercer coherent modes](#4.-Method-B:-Gaussian-Schell-Model-beam-via-Mercer-coherent-modes)
# 5. [Method C: Temporal coherence & dispersive silver screen](#5.-Method-C:-Temporal-coherence-&-dispersive-silver-screen)
# 6. [Study tasks & CLI execution modes](#6.-Study-tasks-&-CLI-execution-modes)
# 7. [Generating the 15-figure suite](#7.-Generating-the-15-figure-suite)
# 8. [Quantitative results & benchmark comparison tables](#8.-Quantitative-results-&-benchmark-comparison-tables)
# 9. [Limits & failure modes](#9.-Limits-&-failure-modes)
# 10. [Exercises](#10.-Exercises)
# 11. [References](#11.-References)

# %% [markdown]
# ## 1. Physical background & second-order coherence theory
#
# ### 1.1 Cross-spectral density and Young's interference law
#
# Consider a 2D TM-polarized optical field ($\mathbf{E} = E_z(x, y, t)\,\hat{\mathbf{z}}$,
# $\mathbf{H} = H_x\,\hat{\mathbf{x}} + H_y\,\hat{\mathbf{y}}$) incident from
# $y < 0$ onto an opaque screen at $y = 0$ pierced by two slits of width $a$
# centered at $\mathbf{r}_1 = (-d/2, 0)$ and $\mathbf{r}_2 = (+d/2, 0)$. For a
# stationary statistical ensemble of monochromatic realizations $\{E_z^{(n)}(\mathbf{r}, \omega)\}$
# with non-negative weights $\{w_n\}$, spatial coherence is governed by the
# **cross-spectral density (CSD)** (Wolf 1982; Mandel & Wolf 1995, §4.3.2):
#
# $$W(\mathbf{r}_1, \mathbf{r}_2, \omega) = \bigl\langle E_z^*(\mathbf{r}_1, \omega)\,E_z(\mathbf{r}_2, \omega) \bigr\rangle = \sum_{n} w_n\,\bigl[E_z^{(n)}(\mathbf{r}_1, \omega)\bigr]^*\,E_z^{(n)}(\mathbf{r}_2, \omega).$$
#
# The spectral intensity (optical power density) is the diagonal
# $I(\mathbf{r}, \omega) = W(\mathbf{r}, \mathbf{r}, \omega) = \sum_n w_n |E_z^{(n)}(\mathbf{r}, \omega)|^2$,
# and the normalized **complex degree of spatial coherence** between the two slits is:
#
# $$\mu_{12}(\omega) \equiv \frac{W(\mathbf{r}_1, \mathbf{r}_2, \omega)}{\sqrt{I(\mathbf{r}_1, \omega)\,I(\mathbf{r}_2, \omega)}}, \qquad 0 \le |\mu_{12}(\omega)| \le 1.$$
#
# On a downstream observation screen at $y = L_2$, each slit radiates a 2D
# cylindrical diffraction wave with transfer function $G_j(x)$ ($j=1,2$). For
# symmetric illumination ($I(\mathbf{r}_1) = I(\mathbf{r}_2)$), the downstream
# intensity obeys the **spectral interference law** (Mandel & Wolf 1995, Eq. 4.3-25):
#
# $$I(x, L_2) = I_1(x) + I_2(x) + 2\sqrt{I_1(x)\,I_2(x)}\,|\mu_{12}|\,\cos\!\bigl[k_0\,\Delta L(x) + \arg\mu_{12}\bigr], \qquad \Delta L(x) = r_1(x) - r_2(x),$$
#
# where $r_{1,2}(x) = \sqrt{(x \pm d/2)^2 + L_2^2}$ and $k_0 = 2\pi / \lambda_0$.
# Near the optical axis ($x \approx 0$, where $I_1(x) = I_2(x)$), the **Michelson
# fringe visibility** equals the magnitude of the complex degree of coherence
# between the slits:
#
# $$\mathcal{V} \equiv \frac{I_{\max} - I_{\min}}{I_{\max} + I_{\min}} = |\mu_{12}(\omega)|.$$
#
# > [!IMPORTANT]
# > **Why incoherent sources must be summed in power, never in field amplitude:**
# > If $N_s$ point emitters are excited simultaneously in a single deterministic
# > FDTD/FIT simulation, the solver evolves the linear sum $E_{\text{tot}}(\mathbf{r}) = \sum_{n=1}^{N_s} E_z^{(n)}(\mathbf{r})$.
# > Squaring that field yields
# > $|E_{\text{tot}}|^2 = \sum_n |E_z^{(n)}|^2 + \sum_{n \neq m} [E_z^{(n)}]^*\,E_z^{(m)}$,
# > where the $N_s(N_s - 1)$ deterministic cross terms turn the extended source
# > into a **phased antenna array** with $|\mu_{12}| = 1$ regardless of source size $D$.
# > True statistical incoherence $\langle [E_z^{(n)}]^* E_z^{(m)} \rangle = \delta_{nm}$
# > requires independent forward solves (or orthogonal eigenmodes) accumulated strictly in **power**.
#
# ---
#
# ### 1.2 Method A: Van Cittert–Zernike theorem for an extended incoherent strip
#
# Consider a spatially incoherent, uniform primary strip source of width $D$ at
# upstream distance $y = -L_1$, composed of uncorrelated line emitters distributed
# over $x_s \in [-D/2, +D/2]$. In 2D TM electrodynamics, a line current at
# $(x_s, -L_1)$ radiates the exact cylindrical Green's function proportional to
# the Hankel function $H_0^{(2)}(k_0 R) \sim \sqrt{2/(\pi k_0 R)}\,e^{-i(k_0 R - \pi/4)}$.
# Integrating the mutual intensity across the strip $x_s \in [-D/2, +D/2]$ in the
# paraxial regime ($R_{1,2}(x_s) \approx L_1 + (x_s \pm d/2)^2 / (2 L_1)$) yields
# the **1D van Cittert–Zernike theorem** (van Cittert 1934; Zernike 1938;
# Born & Wolf 2019, §10.4.2; Goodman 2015, §5.6):
#
# $$\mu_{12}(D) = \frac{\sin(\pi D d / (\lambda_0 L_1))}{\pi D d / (\lambda_0 L_1)} \equiv \operatorname{sinc}\!\left(\frac{D}{D_0}\right), \qquad D_0 \equiv \frac{\lambda_0 L_1}{d},$$
#
# where $\operatorname{sinc}(u) \equiv \sin(\pi u)/(\pi u)$ is the normalized sinc function:
#
# * **Coherent limit ($D \ll D_0$)**: $\mu_{12} \to +1$, giving bright central
#   maximum at $x = 0$ and visibility $\mathcal{V} \to 1$.
# * **First coherence null ($D = D_0 = \lambda_0 L_1 / d$)**: $\mu_{12}(D_0) = 0$;
#   the fringes completely vanish ($\mathcal{V} = 0$) and the screen intensity reduces
#   to the smooth sum of two single-slit envelopes $I_1(x) + I_2(x)$.
# * **Phase-reversed secondary lobe ($D_0 < D < 2 D_0$, peak at $D \approx 1.4303\,D_0$)**:
#   $\mu_{12}(D) < 0$ ($\arg\mu_{12} = \pi$, minimum $\mu_{12} \approx -0.2172$).
#   The fringes **reappear with inverted contrast**, placing a **dark minimum on
#   the optical axis $x = 0$** where a bright peak normally sits (Born & Wolf §10.4.2, Fig. 10.11).
#
# ---
#
# ### 1.3 Method B: Gaussian Schell-Model (GSM) beam via Wolf's Mercer decomposition
#
# A **Gaussian Schell-Model (GSM)** beam has a Gaussian intensity profile of RMS
# width $\sigma_s$ and a shift-invariant Gaussian degree of spatial coherence of
# RMS width $\sigma_\mu$:
#
# $$W_{\text{GSM}}(x_1, x_2) = I_0\,\exp\!\left(-\frac{x_1^2 + x_2^2}{4\sigma_s^2}\right)\,\exp\!\left(-\frac{(x_1 - x_2)^2}{2\sigma_\mu^2}\right) \implies \mathcal{V}_{\text{GSM}} = |\mu_{12}(d)| = \exp\!\left(-\frac{d^2}{2\sigma_\mu^2}\right).$$
#
# By **Wolf's coherent-mode theorem** (Wolf 1982), $W_{\text{GSM}}(x_1, x_2)$
# admits an exact Fredholm–Mercer expansion into mutually orthogonal, 100%
# spatially coherent eigenmodes $\{\Phi_m(x)\}_{m=0}^\infty$ with non-negative
# eigenvalues $\{\lambda_m\}_{m=0}^\infty$:
#
# $$W_{\text{GSM}}(x_1, x_2) = \sum_{m=0}^{\infty} \lambda_m\,\Phi_m^*(x_1)\,\Phi_m(x_2).$$
#
# **Starikov & Wolf** (*J. Opt. Soc. Am.* **72**, 923–928, 1982, Eqs. 12–21) proved
# via Mehler's Hermite polynomial identity that the eigenmodes are **orthonormal
# Hermite–Gaussian functions** with geometrically decaying weights:
#
# $$\Phi_m(x) = \frac{1}{(\pi w_m^2)^{1/4}\sqrt{2^m m!}}\,H_m\!\left(\frac{x}{w_m}\right)\,\exp\!\left(-\frac{x^2}{2 w_m^2}\right), \qquad \frac{\lambda_m}{\sum_{k=0}^\infty \lambda_k} = (1 - q)\,q^m,$$
#
# where $a = (4\sigma_s^2)^{-1}$, $b = (2\sigma_\mu^2)^{-1}$, $c = \sqrt{a^2 + 2ab}$,
# modal scale $w_m = 1/\sqrt{2c}$, and geometric ratio
#
# $$q = \frac{b}{a + b + c} = \frac{\sqrt{1 + \gamma^2} - 1}{\gamma} \in [0, 1), \qquad \gamma \equiv \left(\frac{w_m}{\sigma_\mu}\right)^2.$$
#
# Because $\Phi_m(x)$ depends *only* on the modal scale $w_m$, a **single bank
# of $M$ simulated Hermite–Gaussian FDTD/FIT runs** at fixed $w_m$ synthesizes the
# exact GSM diffraction pattern for *any* coherence width $\sigma_\mu \in (0, \infty)$
# purely in post-processing by re-weighting the mode powers with $w_m(q) = (1 - q)q^m$!
# Moreover, since $\Phi_m(-x) = (-1)^m \Phi_m(x)$, even modes ($m=0,2,4,\dots$)
# drive the two slits in phase ($\mu_{12}^{(m)} = +1$) while odd modes ($m=1,3,5,\dots$)
# drive the slits out of phase ($\mu_{12}^{(m)} = -1$).
#
# ---
#
# ### 1.4 Method C: Broadband temporal coherence (Wiener–Khinchin theorem)
#
# When two slits are illuminated with spatial coherence $\mu_{12} = 1$ by a
# polychromatic source with normalized Gaussian power spectral density
# $S(f) = \frac{1}{\sqrt{2\pi}\sigma_f}\exp[-(f - f_0)^2 / (2\sigma_f^2)]$, the
# time-integrated screen intensity is the spectral integral
# $I_{\text{poly}}(x) = \int_0^\infty S(f)\,|H(x, f)|^2\,df$, where
# $H(x, f) = E_z(x, f) / S_{\text{pulse}}(f)$ is the linear transfer function
# from the source to detector coordinate $x$. By the **Wiener–Khinchin theorem**
# (Mandel & Wolf 1995, §4.3.1; Goodman 2015, §5.2), the fringe modulation vs
# optical path difference $\Delta L(x) = r_1(x) - r_2(x)$ is enveloped by the
# Fourier transform of $S(f)$:
#
# $$|\gamma_{11}(\Delta L)| = \exp\!\left(-\frac{\Delta L^2}{2 L_c^2}\right), \qquad L_c \equiv \frac{c}{2\pi\sigma_f} = \frac{\lambda_0}{2\pi(\sigma_f / f_0)}.$$
#
# On the optical axis ($\Delta L(0) = 0$), the zero-order white-light fringe remains
# bright for all bandwidths, while higher-order fringes ($|\Delta L| = m\lambda_0$)
# decay exponentially once $|\Delta L| \gtrsim L_c$.

# %% [markdown]
# ## 2. Geometry, parameters & 2D FIT domain setup
#
# We work in 2D TM polarization (`Ez`, `Hx`, `Hy`) using a 1-cell periodic slab
# along $z$ ($\text{BC}_z = \text{periodic}$). To prevent the 1-cell $z$-thickness
# from restricting the 3D Courant step $\Delta t \le (\Delta x^{-2} + \Delta y^{-2} + \Delta z^{-2})^{-1/2}/c$,
# we set $\Delta z = 10\,\Delta x_{\min}$, allowing a 2D Courant number $\text{CFL} = 0.65$
# across all PEC and dispersive silver runs.
#
# | Parameter | Symbol | Value | Physical role |
# | :--- | :---: | :---: | :--- |
# | Center vacuum wavelength | $\lambda_0$ | $0.50\;\mu\text{m}$ ($500\;\text{nm}$) | Center frequency $f_0 = 599.585\;\text{THz}$ |
# | Individual slit width | $a$ | $0.40\;\mu\text{m}$ ($0.8\,\lambda_0$) | Single fundamental $\text{TM}_0$ slit mode ($\ge 8$ cells) |
# | Center-to-center slit separation | $d$ | $2.00\;\mu\text{m}$ ($4.0\,\lambda_0$) | Slits centered at $x = \pm 1.00\;\mu\text{m}$ |
# | Screen thickness | $t_{\text{screen}}$ | $0.20\;\mu\text{m}$ ($0.4\,\lambda_0$) | $\ge 4$ cells; $6.95\,\delta_E = 13.90\,\delta_I$ for $\text{Ag}$ at $500\;\text{nm}$ |
# | Slit-to-detector distance | $L_2$ | $6.00\;\mu\text{m}$ ($12\,\lambda_0$) | Fringe period $\Lambda_f = \lambda_0 L_2 / d = 1.50\;\mu\text{m}$ |
# | Primary strip-to-slit distance | $L_1$ | $8.00\;\mu\text{m}$ ($16\,\lambda_0$) | First van Cittert–Zernike null $D_0 = \lambda_0 L_1 / d = 2.00\;\mu\text{m}$ |
# | GSM Hermite–Gaussian scale | $w_m$ | $1.40\;\mu\text{m}$ ($2.8\,\lambda_0$) | Mercer mode waist injected at $y_{\text{GSM}} = -1.00\;\mu\text{m}$ |
# | Silver permittivity at $500\;\text{nm}$ | $\varepsilon_{\text{Ag}}(\omega_0)$ | $-7.632 - 0.731i$ | Rakic *et al.* (1998); $\delta_E = 28.77\;\text{nm}$, $\delta_I = 14.39\;\text{nm}$ |
# | Background resolution (`--quick` / `--full` / ref) | $\text{cpw}_{\text{bg}}$ | $10\text{--}13$ / $20\text{--}26$ / $40$ | Graded to $\Delta x_{\min} = 2.86\;\text{nm}$ ($10.07\;\text{cells}/\delta_E$ in `--full`; $3.57\;\text{nm}$, $8.06\;\text{cells}/\delta_E$ in `--quick`) at Ag slit jaws |
#
# ### 2.1 Simulation setup schematic & graded skin-depth mesh
#
# The setup figures (`figures/00a_setup_geometry.png`, `figures/00b_setup_mesh_and_excitation.png`,
# `figures/00a_setup_geometry_gsm_silver.png`, and `figures/00b_setup_mesh_and_excitation_gsm_silver.png`)
# are built directly from the live Wakis `GridFIT3D`, `SolverFIT3D`, `Box` screen jaws, `CurrentSource` arrays,
# and `DFTMonitor` planes via `wakis.schematic.save_setup_figures`:
#
# - **Method A (Extended Incoherent Strip Source):** An upstream strip of width $D \in [0, 3.60\;\mu\text{m}]$
#   at $y = -L_1 = -8.00\;\mu\text{m}$ sampled by $N_s = 37$ uncorrelated $J_z$ line dipoles (`--full`, with 5 representative
#   emitter positions shown), illuminating the $200\;\text{nm}$-thick $\text{PEC}$ screen at $y = 0$ pierced by two slits of
#   width $a = 0.40\;\mu\text{m}$ separated by $d = 2.00\;\mu\text{m}$, with downstream `DFTMonitor` at $y = +L_2 = +6.00\;\mu\text{m}$
#   and 8-cell CPML boundaries on $\pm x, \pm y$.
# - **Methods B & C (GSM Mercer Eigenmodes, Broadband Pulse & Drude Ag Screen):** Orthonormal Hermite–Gaussian Mercer modes
#   $\Phi_m(x)$ ($w_m = 1.40\;\mu\text{m}$) and broadband pulses injected at $y_{\text{GSM}} = -1.00\;\mu\text{m}$
#   upstream of the $\text{PEC}$ / Drude $\text{Ag}$ double-slit screen, with an interface-snapped graded collar
#   (`--full`: $\Delta x_{\text{bg}} = 24.92\;\text{nm}$, $\Delta x_{\min} = 2.86\;\text{nm}$, $70$ cells across
#   the $200\;\text{nm}$ screen thickness, $67$ cells across the $400\;\text{nm}$ slit opening, and
#   $10.07\;\text{cells}/\delta_E = 5.04\;\text{cells}/\delta_I$ across the $28.77\;\text{nm}$ $\text{Ag}$ skin depth).
#
# ![Setup Geometry (Method A — VCZ Strip)](figures/00a_setup_geometry.png)
# ![Setup Mesh and Excitation (Method A — VCZ Strip)](figures/00b_setup_mesh_and_excitation.png)
#
# The table below summarizes the Method A extended incoherent strip-source double-slit setup, boundary conditions, materials, monitors, and task variations.
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 10.4 µm × 15.3 µm × 250 nm (z ∈ [-125 nm, 125 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 416 × 612 × 1 (254592 cells) |
# | Discretization | Δx & resolution | Δ_min = 25 nm, 20.0 cells/λ (in n_max = 1.00), 8.0 cells across feature (200 nm) |
# | Boundary | -x face | CPML (8 cells, 200 nm) |
# | Boundary | +x face | CPML (8 cells, 200 nm) |
# | Boundary | -y face | CPML (8 cells, 200 nm) |
# | Boundary | +y face | CPML (8 cells, 200 nm) |
# | Boundary | -z face | Periodic |
# | Boundary | +z face | Periodic |
# | Material | Background | Vacuum / Air (n = 1.00), ε = 1.00 (isotropic) |
# | Material | Region 1: PEC double-slit screen (t = 200 nm) | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-5.20 µm, -1.20 µm] (d = 4 µm) |
# | Material | Region 2: PEC double-slit screen (t = 200 nm) | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-800 nm, 800 nm] (d = 1.60 µm) |
# | Material | Region 3: PEC double-slit screen (t = 200 nm) | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [1.20 µm, 5.20 µm] (d = 4 µm) |
# | Source | 1 · CurrentSource | pos: y = -8.00 µm, N_s = 37 dipoles in x ∈ [-1.80, +1.80] µm (5 shown), pol: J_z (TM), pulse: f₀ = 599.6 THz, Δf = 119.9 THz |
# | Monitor | 2 · DFTMonitor | Slit-plane DFTMonitor (y = 0.00 µm; records E_z at slits x = ±1.00 µm for μ_12), pos: y = 0.00 µm, x ∈ [-4.60, +4.60] µm (slits at x = ±1.00 µm, a = 0.40 µm), freqs: f = 599.6 THz |
# | Monitor | 3 · DFTMonitor | Downstream detector DFTMonitor (y = +6.00 µm; records interference profile I(x)), pos: y = +6.00 µm, x ∈ [-4.60, +4.60] µm (L_2 = 6.00 µm, Λ_f = 1.50 µm), freqs: f = 599.6 THz |
# | Variation | Tasks 0–1 — Method A VCZ strip (`vcz_batch_a`, `vcz_batch_b`) | notes=N_s = 37 uncorrelated J_z line dipoles (19 symmetric runs at cpw = 20) spanning D ∈ [0, 3.60] µm (D_0 = 2.00 µm) |
# | Variation | Tasks 6–7 — Spatial mesh convergence (`convergence_coarse`, `convergence_ref`) | notes=PEC double-slit mesh sweep cpw ∈ {10, 14, 20} vs. cpw = 40 reference |
#
# ![Setup Geometry (Methods B & C — GSM & Drude Ag Screen)](figures/00a_setup_geometry_gsm_silver.png)
# ![Setup Mesh and Excitation (Methods B & C — GSM & Drude Ag Screen)](figures/00b_setup_mesh_and_excitation_gsm_silver.png)
#
# The table below summarizes the Methods B and C setup using Hermite–Gaussian Mercer eigenmodes, broadband pulse excitation, and the graded Drude silver screen.
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 9.40 µm × 8.20 µm × 250 nm (z ∈ [-125 nm, 125 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 581 × 443 × 1 (257383 cells) |
# | Discretization | Δx & resolution | Δ_min = 2.86 nm, 91.0 cells/λ (in n_max = 1.92), 10.1 cells across feature (28.8 nm) |
# | Boundary | -x face | CPML (8 cells, 199.3 nm) |
# | Boundary | +x face | CPML (8 cells, 199.3 nm) |
# | Boundary | -y face | CPML (8 cells, 197.8 nm) |
# | Boundary | +y face | CPML (8 cells, 199.3 nm) |
# | Boundary | -z face | Periodic |
# | Boundary | +z face | Periodic |
# | Material | Background | Vacuum / Air (n = 1.00), ε = 1.00 (isotropic) |
# | Material | Region 1: Drude Ag screen (δ_E = 28.8 nm) | ε_r = 3.70, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-4.70 µm, -1.20 µm] (d = 3.50 µm) |
# | Material | Region 2: Drude Ag screen (δ_E = 28.8 nm) | ε_r = 3.70, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-800 nm, 800 nm] (d = 1.60 µm) |
# | Material | Region 3: Drude Ag screen (δ_E = 28.8 nm) | ε_r = 3.70, μ_r = 1.00, model = DispersiveMaterial, x ∈ [1.20 µm, 4.70 µm] (d = 3.50 µm) |
# | Source | 1 · CurrentSource | pos: y = -1.00 µm, x ∈ [-4.00, +4.00] µm (Hermite-Gaussian w_m = 1.40 µm), pol: J_z (TM), pulse: f₀ = 599.6 THz, Δf = 131.9 THz |
# | Monitor | 2 · DFTMonitor | Slit-plane DFTMonitor (y = 0.00 µm; modal amplitude & broadband transfer normalization), pos: y = 0.00 µm, x ∈ [-4.20, +4.20] µm (81 broadband freqs in 0.72–1.28 f₀), freqs: f = 599.6 THz |
# | Monitor | 3 · DFTMonitor | Downstream detector DFTMonitor (y = +6.00 µm; GSM modal & Wiener-Khinchin polychromatic fringes), pos: y = +6.00 µm, x ∈ [-4.20, +4.20] µm (81 broadband freqs in 0.72–1.28 f₀), freqs: f = 599.6 THz |
# | Variation | Tasks 2–3 — Method B & C GSM Mercer modes (`gsm_modes_even`, `gsm_modes_odd`) | notes=M = 12 Hermite-Gaussian modes (m = 0..11, w_m = 1.40 µm) + 81-frequency broadband DFT at cpw = 26 |
# | Variation | Task 4 — Dispersive Drude Ag screen (`silver_dispersive`) | notes=Rakic Ag at 500 nm (ε = -7.632 - 0.731i, δ_E = 28.77 nm) with Δ_min = 2.86 nm (10.07 cells/δ_E) |
# | Variation | Task 5 — Ag skin-depth resolution ladder (`silver_skin_depth_demo`, demo=True) | notes=Under-resolved uniform (0.58, 1.15 cells/δ_E) vs. graded collar (2, 4, 6, 8, 10 cells/δ_E) |

# %%
import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light

from wakis import figstyle, schematic

TUTORIAL_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

import _helpers as h  # noqa: E402
from wakis.field_monitors import DFTMonitor  # noqa: E402
from wakis.sources import GaussianSource  # noqa: E402

DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"

# Physical & geometric constants [m, Hz]
WAVELENGTH_0 = 0.50e-6
F0 = c_light / WAVELENGTH_0
SLIT_WIDTH = 0.40e-6
SLIT_SEP = 2.00e-6
SCREEN_THICK = 0.20e-6
L1_VCZ = 8.00e-6
L2_DET = 6.00e-6
D0_VCZ = WAVELENGTH_0 * L1_VCZ / SLIT_SEP  # 2.00 um
Y_GSM_SRC = -1.00e-6
W_M_GSM = 1.40e-6
N_PML = 8


# %% [markdown]
# ## 3. Method A: Extended incoherent strip source (van Cittert–Zernike)
#
# To model an extended incoherent strip source of variable width $D \in [0, 3.60\;\mu\text{m}]$
# at $y = -L_1 = -8.00\;\mu\text{m}$, we place point-dipole line sources at a
# uniform grid of transverse positions $x_s \in [-1.80, +1.80]\;\mu\text{m}$.
# Because the double-slit geometry is mirror-symmetric under $x \mapsto -x$, the
# complex electric field excited by a dipole at $-x_s$ is the exact mirror image
# of the field excited at $+x_s$:
#
# $$E_z(-x, y; -x_s) = E_z(+x, y; +x_s).$$
#
# Simulating the non-negative half-space $x_s \ge 0$ (10 positions in `--quick`
# with $\Delta x_s = 0.20\;\mu\text{m}$, 19 positions in `--full` with
# $\Delta x_s = 0.10\;\mu\text{m}$) therefore yields a full symmetric bank of
# $N_s = 19$ (`--quick`) or $N_s = 37$ (`--full`) independent dipole realizations.
# For any strip width $D$, we integrate the incoherent intensity over $[-D/2, +D/2]$
# using composite trapezoidal/cell-overlap weights $w_n(D)$.


# %%
def run_vcz_dipole_batch(xs_positive_list, cpw=10, record_2d_indices=()):
    """
    Simulate a batch of upstream line-dipole sources at ``(x_s, -L1_VCZ)``
    for the van Cittert-Zernike extended strip study.

    Parameters
    ----------
    xs_positive_list : sequence of float
        Non-negative source x-coordinates [m].
    cpw : int
        Cells per wavelength.
    record_2d_indices : sequence of int
        Indices within ``xs_positive_list`` for which full 2D DFT fields
        ``Ez(x, y, f0)`` and time snapshots are stored.

    Returns
    -------
    result : dict of ndarray
        Monitored detector fields, slit fields, coordinates, and optional 2D maps.
    """
    pulse = GaussianSource(F0, 0.20 * F0)
    ez_det_list = []
    ez_slit_list = []
    ez_2d_maps = {}
    td_snaps = {}
    x_det = None
    x_slit = None
    x_2d = None
    y_2d = None

    for idx, xs_val in enumerate(xs_positive_list):
        solver, grid, _ = h.build_double_slit_solver(
            x_bounds=(-5.0e-6, 5.0e-6),
            y_bounds=(-8.5e-6, 6.4e-6),
            wavelength=WAVELENGTH_0,
            cpw=cpw,
            n_pml=N_PML,
            slit_sep=SLIT_SEP,
            slit_width=SLIT_WIDTH,
            screen_thickness=SCREEN_THICK,
            screen_material="pec",
        )
        ix_s = int(np.argmin(np.abs(grid.x - float(xs_val))))
        iy_s = int(np.argmin(np.abs(grid.y - (-L1_VCZ))))
        src = h.CustomLineSource(ix_s, iy_s, 1.0, pulse)

        mon_det = DFTMonitor(
            [F0],
            components=("Ez",),
            center=(0.0, L2_DET, 0.0),
            size=(9.6e-6, 0.0, 0.0),
        )
        mon_slit = DFTMonitor(
            [F0],
            components=("Ez",),
            center=(0.0, 0.0, 0.0),
            size=(9.6e-6, 0.0, 0.0),
        )
        monitors = [mon_det, mon_slit]

        want_2d = idx in record_2d_indices
        if want_2d:
            mon_2d = DFTMonitor(
                [F0],
                components=("Ez",),
                center=(0.0, -1.0e-6, 0.0),
                size=(9.6e-6, 14.6e-6, 0.0),
            )
            monitors.append(mon_2d)

        t_end = pulse.t0 + 4.5 / (0.20 * F0) + (15.5e-6 / c_light) * 1.38
        n_steps = int(np.ceil(t_end / solver.dt))
        solver.emsolve(n_steps, source=src, monitors=monitors)

        f_det, c_det = mon_det.get_field("Ez")
        f_slit, c_slit = mon_slit.get_field("Ez")
        ez_det_list.append(f_det[0, :, 0, 0])
        ez_slit_list.append(f_slit[0, :, 0, 0])
        if x_det is None:
            x_det = c_det[0]
            x_slit = c_slit[0]

        if want_2d:
            f_2d, c_2d = mon_2d.get_field("Ez")
            ez_2d_maps[f"ez2d_idx{idx}"] = f_2d[0, :, :, 0]
            if x_2d is None:
                x_2d = c_2d[0]
                y_2d = c_2d[1]

    res_info = h.measure_mesh_resolution(
        grid,
        wavelength=WAVELENGTH_0,
        slit_sep=SLIT_SEP,
        slit_width=SLIT_WIDTH,
        screen_thickness=SCREEN_THICK,
        screen_material="pec",
        n_steps=n_steps,
    )
    out = {
        "xs_positive": np.asarray(xs_positive_list, dtype=float),
        "x_det": x_det,
        "x_slit": x_slit,
        "ez_det": np.asarray(ez_det_list),
        "ez_slit": np.asarray(ez_slit_list),
    }
    for k, v in res_info.items():
        out[k] = np.array(v)
    if x_2d is not None:
        out["x_2d"] = x_2d
        out["y_2d"] = y_2d
        out.update(ez_2d_maps)
        out.update(td_snaps)
    return out


# %% [markdown]
# ## 4. Method B: Gaussian Schell-Model beam via Mercer coherent modes
#
# For the Gaussian Schell-Model study, we inject the orthonormal Hermite–Gaussian
# eigenmodes $\Phi_m(x)$ ($m = 0, 1, \dots, M_{\max}-1$) with modal scale
# $w_m = 1.40\;\mu\text{m}$ along the source line $y_{\text{GSM}} = -1.00\;\mu\text{m}$.
# On the fundamental $m=0$ and first antisymmetric $m=1$ modes we also attach a
# broadband `DFTMonitor` spanning $f \in [0.72\,f_0, 1.28\,f_0]$ (57 frequency
# channels in `--quick`, 81 in `--full`) and capture time-domain snapshots
# $E_z(x, y, t_k)$ for the wavepacket propagation animation. Because the broadband
# monitor reaches $f_{\max} = 1.28\,f_0$ ($\lambda_{\min} = 390.6\;\text{nm}$), we
# run the GSM mode tasks at $\text{cpw} = 13$ ($\Delta x = 38.5\;\text{nm}$, giving
# $10.16\;\text{cells}/\lambda_{\min} \ge 10$) in `--quick` and $\text{cpw} = 26$
# ($\Delta x = 19.2\;\text{nm}$, $20.31\;\text{cells}/\lambda_{\min} \ge 20$) in `--full`.


# %%
def run_gsm_mode_batch(mode_indices, cpw=13, n_freqs_broadband=57, record_snapshots=False):
    """
    Simulate a batch of Hermite-Gaussian Mercer modes ``Phi_m(x)`` illuminating
    the double-slit screen.

    Parameters
    ----------
    mode_indices : sequence of int
        Mode orders ``m`` to simulate (e.g. ``[0, 2, 4, 6, 8]``).
    cpw : int
        Cells per center wavelength ``WAVELENGTH_0``.
    n_freqs_broadband : int
        Number of frequencies across ``[0.72*F0, 1.28*F0]`` for ``m in (0, 1)``.
    record_snapshots : bool
        If True and ``0 in mode_indices``, record time-domain ``Ez(x, y, t)``
        frames for the wavepacket animation and snapshot plots.

    Returns
    -------
    result : dict of ndarray
        Monitored single-frequency and broadband DFT spectra and 2D maps.
    """
    freqs_bb = np.linspace(0.72 * F0, 1.28 * F0, int(n_freqs_broadband))
    pulse = GaussianSource(F0, 0.22 * F0)

    ez_det_f0 = []
    ez_slit_f0 = []
    broadband_data = {}
    maps_2d = {}
    x_det = None
    x_slit = None

    for m in mode_indices:
        solver, grid, _ = h.build_double_slit_solver(
            x_bounds=(-5.0e-6, 5.0e-6),
            y_bounds=(-1.6e-6, 6.4e-6),
            wavelength=WAVELENGTH_0,
            cpw=cpw,
            n_pml=N_PML,
            slit_sep=SLIT_SEP,
            slit_width=SLIT_WIDTH,
            screen_thickness=SCREEN_THICK,
            screen_material="pec",
        )
        ix_src = np.arange(N_PML, grid.Nx - N_PML)
        x_src = grid.x[ix_src]
        iy_src = int(np.argmin(np.abs(grid.y - Y_GSM_SRC)))
        phi_m = h.gsm_hermite_mode(m, x_src, w_m=W_M_GSM)
        src = h.CustomLineSource(ix_src, iy_src, phi_m, pulse)

        mon_freqs = freqs_bb if m in (0, 1) else np.array([F0])
        mon_det = DFTMonitor(
            mon_freqs,
            components=("Ez",),
            center=(0.0, L2_DET, 0.0),
            size=(9.6e-6, 0.0, 0.0),
        )
        mon_slit = DFTMonitor(
            mon_freqs,
            components=("Ez",),
            center=(0.0, 0.0, 0.0),
            size=(9.6e-6, 0.0, 0.0),
        )
        monitors = [mon_det, mon_slit]

        want_2d = m in (0, 1)
        if want_2d:
            mon_2d = DFTMonitor(
                [F0],
                components=("Ex", "Ey", "Ez", "Hx", "Hy"),
                center=(0.0, 2.4e-6, 0.0),
                size=(9.6e-6, 7.8e-6, 0.0),
            )
            monitors.append(mon_2d)

        t_end = pulse.t0 + 4.5 / (0.22 * F0) + (8.6e-6 / c_light) * 1.4
        n_steps = int(np.ceil(t_end / solver.dt))

        if record_snapshots and m == 0:
            snap_steps = np.linspace(
                int(0.14 * n_steps), int(0.76 * n_steps), 24, dtype=int
            )
            snap_set = set(snap_steps.tolist())
            frames_ez = []
            frames_t_fs = []
            ix_s = slice(N_PML, grid.Nx - N_PML)
            iy_s = slice(N_PML, grid.Ny - N_PML)
            for step in range(n_steps):
                t_now = step * solver.dt
                src.update(solver, t_now)
                solver.one_step()
                for mon in monitors:
                    mon.update(solver, t_now)
                if step in snap_set:
                    frames_ez.append(
                        np.array(solver.E[ix_s, iy_s, 0, "z"], dtype=np.float32)
                    )
                    frames_t_fs.append(float((t_now + solver.dt) * 1e15))
            maps_2d["td_frames_ez"] = np.asarray(frames_ez)
            maps_2d["td_frames_t_fs"] = np.asarray(frames_t_fs)
            maps_2d["td_x"] = grid.x[ix_s]
            maps_2d["td_y"] = grid.y[iy_s]
        else:
            solver.emsolve(n_steps, source=src, monitors=monitors)

        f_det, c_det = mon_det.get_field("Ez")
        f_slit, c_slit = mon_slit.get_field("Ez")
        if x_det is None:
            x_det = c_det[0]
            x_slit = c_slit[0]

        if m in (0, 1):
            ic = int(np.argmin(np.abs(freqs_bb - F0)))
            ez_det_f0.append(f_det[ic, :, 0, 0])
            ez_slit_f0.append(f_slit[ic, :, 0, 0])
            broadband_data[f"dft_det_m{m}"] = f_det[:, :, 0, 0]
            broadband_data[f"dft_slit_m{m}"] = f_slit[:, :, 0, 0]
            broadband_data["freqs_bb"] = freqs_bb
        else:
            ez_det_f0.append(f_det[0, :, 0, 0])
            ez_slit_f0.append(f_slit[0, :, 0, 0])

        if want_2d:
            ez2d, c2d = mon_2d.get_field("Ez")
            hx2d, _ = mon_2d.get_field("Hx")
            hy2d, _ = mon_2d.get_field("Hy")
            maps_2d[f"ez_2d_m{m}"] = ez2d[0, :, :, 0]
            maps_2d[f"hx_2d_m{m}"] = hx2d[0, :, :, 0]
            maps_2d[f"hy_2d_m{m}"] = hy2d[0, :, :, 0]
            maps_2d["x_2d"] = c2d[0]
            maps_2d["y_2d"] = c2d[1]

    res_info = h.measure_mesh_resolution(
        grid,
        wavelength=WAVELENGTH_0,
        slit_sep=SLIT_SEP,
        slit_width=SLIT_WIDTH,
        screen_thickness=SCREEN_THICK,
        screen_material="pec",
        n_steps=n_steps,
    )
    out = {
        "mode_indices": np.asarray(mode_indices, dtype=int),
        "x_det": x_det,
        "x_slit": x_slit,
        "ez_det_f0": np.asarray(ez_det_f0),
        "ez_slit_f0": np.asarray(ez_slit_f0),
    }
    for k, v in res_info.items():
        out[k] = np.array(v)
    out.update(broadband_data)
    out.update(maps_2d)
    return out


# %% [markdown]
# ## 5. Method C: Temporal coherence & dispersive silver screen
#
# Real metallic screens at visible wavelengths ($\lambda_0 = 500\;\text{nm}$) are
# not ideal perfect electric conductors ($\text{PEC}$): electromagnetic fields
# penetrate into the metal jaws over the electric-field $1/e$ amplitude skin depth
# $\delta_E = \lambda_0 / (2\pi\kappa) = 28.77\;\text{nm}$ (corresponding to the
# optical intensity/power $1/e$ skin depth $\delta_I = \lambda_0 / (4\pi\kappa) = \delta_E / 2 = 14.39\;\text{nm}$,
# where $\kappa = \operatorname{Im}\sqrt{\varepsilon_{\text{Ag}}^*} = 2.766$),
# slightly widening the effective optical aperture while attenuating direct
# transmission through a screen of thickness
# $t_{\text{screen}} = 200\;\text{nm} = 6.95\,\delta_E = 13.90\,\delta_I$.
#
# In `silver_drude_material`, we match a 1-pole Drude auxiliary differential
# equation (ADE) model to the Rakic *et al.* (1998) experimental permittivity of
# silver at $\lambda_0 = 500\;\text{nm}$ ($\varepsilon = -7.632 - 0.731i$), avoiding
# the high-frequency numerical stiffness ($\omega_p \Delta t > 2$) of the 20.29 eV
# ultraviolet Lorentz pole in the 6-pole Rakic parameterization.
#
# ### 5.1 Skin-depth mesh resolution study for the dispersive silver screen
#
# On a uniform background mesh at $\text{cpw} = 10$ ($\Delta x = 50.0\;\text{nm}$)
# or $\text{cpw} = 20$ ($\Delta x = 25.0\;\text{nm}$), the silver skin depth spans
# only $\delta_E / \Delta x = 0.58$ or $1.15$ cells ($\delta_I / \Delta x = 0.29$
# or $0.58$ cells). Because the exponential field decay inside the slit jaws is
# under-resolved by a single staircase cell at each slit wall, an unrefined
# $10\;\text{cpw}$ mesh artificially inflates the on-axis Ag-to-PEC transmission
# ratio $I_{\text{Ag}}(0) / I_{\text{PEC}}(0)$ to $+10.2\%$ ($+10.5\%$ when PEC
# and Ag are stepped at different Courant numbers).
#
# To resolve the metallic boundary layer without the prohibitive cost of a
# globally uniform $3.5\;\text{nm}$ grid, `build_double_slit_solver` uses
# `wakis.templates.meshing.graded_axis` (`grading <= 1.4`) to place a symmetric
# fine collar of width $\pm 2.5\,\delta_E \approx \pm 72\;\text{nm}$ around all
# four slit walls ($x = \pm d/2 \pm a/2$) and across the screen faces ($y = \pm t_{\text{screen}}/2$),
# while keeping $\ge 16$ cells across the $400\;\text{nm}$ slit openings and
# grading smoothly out to the background cell size $\Delta x_{\text{bg}}$. Both
# the PEC and Drude Ag solves in `run_silver_vs_pec_study` and
# `run_silver_skin_depth_ladder` are evaluated on the **identical graded mesh**
# at matched Courant number $\text{CFL} = 0.65$. We refine step by step across
# $\delta_E / \Delta x_{\min} \in \{0.58, 1.15, 2.01, 4.03, 6.04, 8.06, 10.07\}$
# ($\delta_I / \Delta x_{\min} \in \{0.29, 0.58, 1.01, 2.01, 3.02, 4.03, 5.04\}$) and
# accept the $\Delta x_{\min} = 2.86\;\text{nm}$ level ($10.07\;\text{cells}/\delta_E$,
# $5.04\;\text{cells}/\delta_I$ at $\text{cpw}_{\text{bg}} = 20$ in `--full`), where
# $I_{\text{Ag}}(0)/I_{\text{PEC}}(0)$ changes by only $0.05\%$ ($0.0461\%$) relative
# to the $8.06\;\text{cells}/\delta_E$ level (well below the $< 1\%$ convergence
# threshold) and converges to $1.0575$ ($+5.75\%$; in `--quick` at
# $\text{cpw}_{\text{bg}} = 12$ and $8.06\;\text{cells}/\delta_E$, it converges to
# $1.0501$ or $+5.01\%$ with a $0.11\%$ step change).


# %%
def run_silver_vs_pec_study(cpw=12, cells_per_skin_depth=8.0):
    """
    Compare double-slit diffraction for a dispersive Drude silver screen vs PEC
    on a skin-depth-resolved graded mesh (`cells_per_skin_depth >= 8.0` for
    field skin depth `delta_E`, `>= 4.0` for power skin depth `delta_I`).
    """
    pulse = GaussianSource(F0, 0.22 * F0)
    results = {}

    for mat_name in ("pec", "silver"):
        solver, grid, _ = h.build_double_slit_solver(
            x_bounds=(-4.5e-6, 4.5e-6),
            y_bounds=(-1.5e-6, 6.3e-6),
            wavelength=WAVELENGTH_0,
            cpw=cpw,
            n_pml=N_PML,
            slit_sep=SLIT_SEP,
            slit_width=SLIT_WIDTH,
            screen_thickness=SCREEN_THICK,
            screen_material=mat_name,
            cfln=0.65,
            cells_per_skin_depth=cells_per_skin_depth,
        )
        ix_src = np.arange(N_PML, grid.Nx - N_PML)
        x_src = grid.x[ix_src]
        iy_src = int(np.argmin(np.abs(grid.y - Y_GSM_SRC)))
        phi_0 = h.gsm_hermite_mode(0, x_src, w_m=W_M_GSM)
        src = h.CustomLineSource(ix_src, iy_src, phi_0, pulse)

        mon_det = DFTMonitor(
            [F0],
            components=("Ez",),
            center=(0.0, L2_DET, 0.0),
            size=(8.6e-6, 0.0, 0.0),
        )
        mon_slit = DFTMonitor(
            [F0],
            components=("Ez",),
            center=(0.0, 0.0, 0.0),
            size=(8.6e-6, 0.0, 0.0),
        )
        t_end = 2.2 * pulse.t0 + (8.5e-6 / c_light) * 1.4
        n_steps = int(np.ceil(t_end / solver.dt))
        solver.emsolve(n_steps, source=src, monitors=[mon_det, mon_slit])

        f_det, c_det = mon_det.get_field("Ez")
        f_slit, c_slit = mon_slit.get_field("Ez")
        results[f"ez_det_{mat_name}_m0"] = f_det[0, :, 0, 0]
        results[f"ez_slit_{mat_name}_m0"] = f_slit[0, :, 0, 0]
        results["x_det"] = c_det[0]
        results["x_slit"] = c_slit[0]

    _, skin_depth = h.silver_drude_material(WAVELENGTH_0)
    res_info = h.measure_mesh_resolution(
        grid,
        wavelength=WAVELENGTH_0,
        slit_sep=SLIT_SEP,
        slit_width=SLIT_WIDTH,
        screen_thickness=SCREEN_THICK,
        screen_material="silver",
        n_steps=n_steps,
    )
    results["skin_depth_m"] = np.array(skin_depth)
    results["cpw"] = np.array(cpw)
    for k, v in res_info.items():
        results[k] = np.array(v)
    return results


def run_silver_skin_depth_ladder(
    cpw_bg=12, cpsd_levels=("uniform_10", "uniform_20", 2.0, 4.0, 6.0)
):
    """
    Run the Drude silver vs PEC skin-depth resolution ladder across under-resolved
    uniform grids (`0.58` and `1.15` cells per `delta_E`) and locally refined
    graded meshes (`2.0, 4.0, 6.0, ...` cells per `delta_E`).
    """
    pulse = GaussianSource(F0, 0.22 * F0)
    dx_min_list = []
    dx_bg_list = []
    cpsd_field_list = []
    cpsd_power_list = []
    cells_feat_list = []
    ncells_list = []
    nsteps_list = []
    runtimes_list = []
    ratio_0_list = []
    vis_pec_list = []
    vis_ag_list = []
    slit_ag_coarse = None
    slit_pec_coarse = None
    x_slit_coarse = None

    for lvl in cpsd_levels:
        t0 = time.perf_counter()
        if lvl == "uniform_10":
            cpw_use, cpsd_use = 10, "uniform"
        elif lvl == "uniform_20":
            cpw_use, cpsd_use = 20, "uniform"
        else:
            cpw_use, cpsd_use = int(cpw_bg), float(lvl)

        vals_i0 = {}
        vals_vis = {}
        for mat_name in ("pec", "silver"):
            solver, grid, _ = h.build_double_slit_solver(
                x_bounds=(-4.5e-6, 4.5e-6),
                y_bounds=(-1.5e-6, 6.3e-6),
                wavelength=WAVELENGTH_0,
                cpw=cpw_use,
                n_pml=N_PML,
                slit_sep=SLIT_SEP,
                slit_width=SLIT_WIDTH,
                screen_thickness=SCREEN_THICK,
                screen_material=mat_name,
                cfln=0.65,
                cells_per_skin_depth=cpsd_use,
            )
            ix_src = np.arange(N_PML, grid.Nx - N_PML)
            x_src = grid.x[ix_src]
            iy_src = int(np.argmin(np.abs(grid.y - Y_GSM_SRC)))
            phi_0 = h.gsm_hermite_mode(0, x_src, w_m=W_M_GSM)
            src = h.CustomLineSource(ix_src, iy_src, phi_0, pulse)

            mon_det = DFTMonitor(
                [F0],
                components=("Ez",),
                center=(0.0, L2_DET, 0.0),
                size=(8.6e-6, 0.0, 0.0),
            )
            monitors = [mon_det]
            if lvl == "uniform_10":
                mon_slit = DFTMonitor(
                    [F0],
                    components=("Ez",),
                    center=(0.0, 0.0, 0.0),
                    size=(8.6e-6, 0.0, 0.0),
                )
                monitors.append(mon_slit)

            t_end = 2.2 * pulse.t0 + (8.5e-6 / c_light) * 1.4
            n_steps = int(np.ceil(t_end / solver.dt))
            solver.emsolve(n_steps, source=src, monitors=monitors)

            f_det, c_det = mon_det.get_field("Ez")
            i_det = np.abs(f_det[0, :, 0, 0]) ** 2
            ix0 = int(np.argmin(np.abs(c_det[0] - 0.0)))
            vals_i0[mat_name] = float(i_det[ix0])
            v_val, _ = h.extract_fringe_visibility(
                c_det[0], i_det, d=SLIT_SEP, L2=L2_DET, wavelength=WAVELENGTH_0
            )
            vals_vis[mat_name] = float(v_val)
            if lvl == "uniform_10":
                f_slit, c_slit = mon_slit.get_field("Ez")
                x_slit_coarse = c_slit[0]
                if mat_name == "pec":
                    slit_pec_coarse = f_slit[0, :, 0, 0]
                else:
                    slit_ag_coarse = f_slit[0, :, 0, 0]

        elapsed = time.perf_counter() - t0
        res_info = h.measure_mesh_resolution(
            grid,
            wavelength=WAVELENGTH_0,
            slit_sep=SLIT_SEP,
            slit_width=SLIT_WIDTH,
            screen_thickness=SCREEN_THICK,
            screen_material="silver",
            n_steps=n_steps,
        )
        dx_min_list.append(res_info["dx_min_nm"])
        dx_bg_list.append(res_info["dx_background_nm"])
        cpsd_field_list.append(res_info["cells_per_skin_depth"])
        cpsd_power_list.append(res_info["cells_per_skin_depth_power"])
        cells_feat_list.append(res_info["cells_across_feature"])
        ncells_list.append(res_info["n_cells"])
        nsteps_list.append(res_info["n_steps"])
        runtimes_list.append(elapsed)
        ratio_0_list.append(vals_i0["silver"] / vals_i0["pec"])
        vis_pec_list.append(vals_vis["pec"])
        vis_ag_list.append(vals_vis["silver"])

    out = {
        "dx_min_nm_ladder": np.asarray(dx_min_list, dtype=float),
        "dx_bg_nm_ladder": np.asarray(dx_bg_list, dtype=float),
        "cpsd_field_ladder": np.asarray(cpsd_field_list, dtype=float),
        "cpsd_power_ladder": np.asarray(cpsd_power_list, dtype=float),
        "cells_across_feature_ladder": np.asarray(cells_feat_list, dtype=int),
        "n_cells_ladder": np.asarray(ncells_list, dtype=int),
        "n_steps_ladder": np.asarray(nsteps_list, dtype=int),
        "runtimes_s": np.asarray(runtimes_list, dtype=float),
        "ratio_ag_to_pec": np.asarray(ratio_0_list, dtype=float),
        "vis_pec": np.asarray(vis_pec_list, dtype=float),
        "vis_silver": np.asarray(vis_ag_list, dtype=float),
        "dx_background_nm": np.array(dx_bg_list[-1]),
        "dx_min_nm": np.array(dx_min_list[0]),
        "min_feature_nm": np.array(200.0),
        "cells_across_feature": np.array(min(cells_feat_list)),
        "skin_depth_nm": np.array(res_info["skin_depth_nm"]),
        "cells_per_skin_depth": np.array(cpsd_field_list[0]),
        "skin_depth_power_nm": np.array(res_info["skin_depth_power_nm"]),
        "cells_per_skin_depth_power": np.array(cpsd_power_list[0]),
        "n_cells": np.array(max(ncells_list)),
        "n_steps": np.array(max(nsteps_list)),
    }
    if x_slit_coarse is not None:
        out["x_slit_coarse"] = x_slit_coarse
        out["ez_slit_pec_coarse"] = slit_pec_coarse
        out["ez_slit_silver_coarse"] = slit_ag_coarse
    return out


def run_resolution_convergence_sweep(cpw_list):
    """Run m=0 and m=1 double-slit diffraction across a list of grid resolutions."""
    pulse = GaussianSource(F0, 0.22 * F0)
    x_common = np.linspace(-4.2e-6, 4.2e-6, 337)
    out = {
        "cpw_list": np.asarray(cpw_list, dtype=int),
        "x_common": x_common,
    }
    i_coh_list = []
    i_pc_list = []
    vis_coh_list = []
    vis_pc_list = []
    runtimes = []
    dx_bg_list = []
    dx_min_list = []
    cells_feat_list = []
    ncells_list = []
    nsteps_list = []

    for cpw in cpw_list:
        t0 = time.perf_counter()
        det_modes = []
        slit_modes = []
        x_det = None
        for m in (0, 1):
            solver, grid, _ = h.build_double_slit_solver(
                x_bounds=(-5.0e-6, 5.0e-6),
                y_bounds=(-1.6e-6, 6.4e-6),
                wavelength=WAVELENGTH_0,
                cpw=int(cpw),
                n_pml=N_PML,
                slit_sep=SLIT_SEP,
                slit_width=SLIT_WIDTH,
                screen_thickness=SCREEN_THICK,
                screen_material="pec",
            )
            ix_src = np.arange(N_PML, grid.Nx - N_PML)
            x_src = grid.x[ix_src]
            iy_src = int(np.argmin(np.abs(grid.y - Y_GSM_SRC)))
            phi_m = h.gsm_hermite_mode(m, x_src, w_m=W_M_GSM)
            src = h.CustomLineSource(ix_src, iy_src, phi_m, pulse)

            mon_det = DFTMonitor(
                [F0],
                components=("Ez",),
                center=(0.0, L2_DET, 0.0),
                size=(9.0e-6, 0.0, 0.0),
            )
            mon_slit = DFTMonitor(
                [F0],
                components=("Ez",),
                center=(0.0, 0.0, 0.0),
                size=(9.0e-6, 0.0, 0.0),
            )
            t_end = pulse.t0 + 4.5 / (0.22 * F0) + (8.6e-6 / c_light) * 1.4
            n_steps = int(np.ceil(t_end / solver.dt))
            solver.emsolve(n_steps, source=src, monitors=[mon_det, mon_slit])

            f_det, c_det = mon_det.get_field("Ez")
            f_slit, c_slit = mon_slit.get_field("Ez")
            det_modes.append(f_det[0, :, 0, 0])
            x_slit = c_slit[0]
            ix_s1 = int(np.argmin(np.abs(x_slit - (-0.5 * SLIT_SEP))))
            slit_modes.append(f_slit[0, ix_s1, 0, 0])
            if x_det is None:
                x_det = c_det[0]

        elapsed = time.perf_counter() - t0
        res_info = h.measure_mesh_resolution(
            grid,
            wavelength=WAVELENGTH_0,
            slit_sep=SLIT_SEP,
            slit_width=SLIT_WIDTH,
            screen_thickness=SCREEN_THICK,
            screen_material="pec",
            n_steps=n_steps,
        )
        dx_bg_list.append(res_info["dx_background_nm"])
        dx_min_list.append(res_info["dx_min_nm"])
        cells_feat_list.append(res_info["cells_across_feature"])
        ncells_list.append(res_info["n_cells"])
        nsteps_list.append(res_info["n_steps"])

        e0 = det_modes[0] / np.abs(slit_modes[0])
        e1 = det_modes[1] / np.abs(slit_modes[1])
        i_coh = np.abs(e0) ** 2
        i_pc = 0.75 * (np.abs(e0) ** 2) + 0.25 * (np.abs(e1) ** 2)  # target mu_12 = 0.50

        v_coh, _ = h.extract_fringe_visibility(
            x_det, i_coh, d=SLIT_SEP, L2=L2_DET, wavelength=WAVELENGTH_0
        )
        v_pc, _ = h.extract_fringe_visibility(
            x_det, i_pc, d=SLIT_SEP, L2=L2_DET, wavelength=WAVELENGTH_0
        )
        i_coh_interp = np.interp(x_common, x_det, i_coh)
        i_coh_interp /= np.mean(i_coh_interp)
        i_pc_interp = np.interp(x_common, x_det, i_pc)
        i_pc_interp /= np.mean(i_pc_interp)

        i_coh_list.append(i_coh_interp)
        i_pc_list.append(i_pc_interp)
        vis_coh_list.append(v_coh)
        vis_pc_list.append(v_pc)
        runtimes.append(elapsed)

    out["i_coh_norm"] = np.asarray(i_coh_list)
    out["i_pc_norm"] = np.asarray(i_pc_list)
    out["vis_coh"] = np.asarray(vis_coh_list)
    out["vis_pc"] = np.asarray(vis_pc_list)
    out["runtimes_s"] = np.asarray(runtimes)
    out["dx_bg_nm_list"] = np.asarray(dx_bg_list)
    out["dx_min_nm_list"] = np.asarray(dx_min_list)
    out["cells_across_feature_list"] = np.asarray(cells_feat_list)
    out["n_cells_list"] = np.asarray(ncells_list)
    out["n_steps_list"] = np.asarray(nsteps_list)
    out["dx_background_nm"] = np.array(max(dx_bg_list))
    out["dx_min_nm"] = np.array(min(dx_min_list))
    out["min_feature_nm"] = np.array(200.0)
    out["cells_across_feature"] = np.array(min(cells_feat_list))
    out["n_cells"] = np.array(max(ncells_list))
    out["n_steps"] = np.array(max(nsteps_list))
    return out


# %% [markdown]
# ## 6. Study tasks & CLI execution modes
#
# The study is partitioned into **8 parallelizable tasks** (`--list-tasks`,
# `--task I --out DIR`). Every non-demo task satisfies all three project mesh
# resolution rules: (1) $\ge 10\;\text{cpw}$ at the shortest monitored wavelength
# $\lambda_{\min}$ in `--quick` ($\ge 20\;\text{cpw}$ in `--full`), (2) $\ge 4$
# cells across the smallest geometric feature ($t_{\text{screen}} = 200\;\text{nm}$,
# $\ge 8$ cells across each $a = 400\;\text{nm}$ slit), and (3) for the dispersive
# silver screen (Task `4`), local skin-depth refinement to $\Delta x_{\min} = 3.57\;\text{nm}$
# ($8.06\;\text{cells}/\delta_E = 4.03\;\text{cells}/\delta_I$ in `--quick`, and
# $\Delta x_{\min} = 2.86\;\text{nm}$, $10.07\;\text{cells}/\delta_E = 5.04\;\text{cells}/\delta_I$
# in `--full`). Task `5` (`silver_skin_depth_demo`, marked `demo=True`) is the
# clearly labelled under-resolved demonstration and convergence ladder comparing
# unrefined $0.58$ and $1.15\;\text{cells}/\delta_E$ grids against locally refined
# graded meshes up to the accepted level:
#
# | Task ID | Name | Demo? | $\text{cpw}_{\text{bg}}$ (`--quick` / `--full`) | $\Delta x_{\text{bg}}$ [$\text{nm}$] | $\Delta x_{\min}$ [$\text{nm}$] | Cells across $t_{\text{screen}}$ ($200\;\text{nm}$) | Cells / $\delta_E$ ($\delta_I$) | $N_{\text{cells}}$ (`--quick` / `--full`) | $N_{\text{steps}}$ (`--quick` / `--full`) |
# | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | `0` | `vcz_batch_a` | No | $10$ / $20$ | $50.00$ / $25.00$ | $50.00$ / $25.00$ | $4$ / $8$ | — | $68\,355$ / $255\,621$ | $1\,511$ / $3\,022$ |
# | `1` | `vcz_batch_b` | No | $10$ / $20$ | $50.00$ / $25.00$ | $50.00$ / $25.00$ | $4$ / $8$ | — | $68\,355$ / $255\,621$ | $1\,511$ / $3\,022$ |
# | `2` | `gsm_modes_even` | No | $13$ / $26$ | $38.46$ / $19.23$ | $33.33$ / $18.18$ | $6$ / $11$ | — | $63\,054$ / $233\,926$ | $1\,511$ / $2\,823$ |
# | `3` | `gsm_modes_odd` | No | $13$ / $26$ | $38.46$ / $19.23$ | $33.33$ / $18.18$ | $6$ / $11$ | — | $63\,054$ / $233\,926$ | $1\,511$ / $2\,823$ |
# | `4` | `silver_dispersive` | No | $12$ / $20$ | $41.51$ / $24.92$ | $3.57$ / $2.86$ | $56$ / $70$ ($67$ min feat) | $8.06\;(4.03)$ / $10.07\;(5.04)$ | $125\,216$ / $258\,408$ | $9\,640$ / $12\,051$ |
# | `5` | `silver_skin_depth_demo` | **Yes** | $10\text{--}12$ / $10\text{--}20$ | $50.00\to 25.00$ | $50.00\to 3.57$ | $4\to 56$ | $0.58\to 8.06$ ($0.29\to 4.03$) | $34\,081\to 224\,466$ | $693\to 9\,641$ |
# | `6` | `convergence_coarse` | No | $[10,14,18]$ / $[10,14,20]$ | $50.00\to 25.00$ | $50.00\to 25.00$ | $4\to 8$ | — | $38\,409\to 140\,529$ | $1\,051\to 2\,101$ |
# | `7` | `convergence_ref` | No | $24$ / $40$ | $20.83$ / $12.50$ | $20.00$ / $12.50$ | $10$ / $16$ | — | $201\,000$ / $536\,769$ | $2\,626$ / $4\,202$ |


# %%
def get_task_table(quick=True):
    cpw_vcz = 10 if quick else 20
    cpw_gsm = 13 if quick else 26
    cpw_ag_bg = 12 if quick else 20
    cpsd_ag_accepted = 8.0 if quick else 10.0
    cpsd_ag_ladder = (
        ["uniform_10", "uniform_20", 2.0, 4.0, 6.0]
        if quick
        else ["uniform_10", "uniform_20", 2.0, 4.0, 6.0, 8.0]
    )
    xs_a = [0.0, 0.2e-6, 0.4e-6, 0.6e-6, 0.8e-6] if quick else [0.1e-6 * k for k in range(10)]
    xs_b = [1.0e-6, 1.2e-6, 1.4e-6, 1.6e-6, 1.8e-6] if quick else [0.1e-6 * k for k in range(10, 19)]
    even_modes = [0, 2, 4, 6, 8] if quick else [0, 2, 4, 6, 8, 10]
    odd_modes = [1, 3, 5, 7, 9] if quick else [1, 3, 5, 7, 9, 11]
    n_bb = 57 if quick else 81
    cpw_sweep = [10, 14, 18] if quick else [10, 14, 20]
    cpw_ref = 24 if quick else 40

    return [
        {
            "index": 0,
            "name": "vcz_batch_a",
            "kind": "vcz",
            "demo": False,
            "cpw": cpw_vcz,
            "f_max": F0,
            "xs_positive": xs_a,
            "record_2d_indices": [0, len(xs_a) - 1],
            "description": f"VCZ inner dipole batch ({len(xs_a)} runs at cpw={cpw_vcz})",
        },
        {
            "index": 1,
            "name": "vcz_batch_b",
            "kind": "vcz",
            "demo": False,
            "cpw": cpw_vcz,
            "f_max": F0,
            "xs_positive": xs_b,
            "record_2d_indices": [],
            "description": f"VCZ outer dipole batch ({len(xs_b)} runs at cpw={cpw_vcz})",
        },
        {
            "index": 2,
            "name": "gsm_modes_even",
            "kind": "gsm",
            "demo": False,
            "cpw": cpw_gsm,
            "f_max": 1.28 * F0,
            "modes": even_modes,
            "n_freqs_bb": n_bb,
            "record_snapshots": True,
            "description": f"GSM Mercer even modes {even_modes} + broadband DFT (cpw={cpw_gsm})",
        },
        {
            "index": 3,
            "name": "gsm_modes_odd",
            "kind": "gsm",
            "demo": False,
            "cpw": cpw_gsm,
            "f_max": 1.28 * F0,
            "modes": odd_modes,
            "n_freqs_bb": n_bb,
            "record_snapshots": False,
            "description": f"GSM Mercer odd modes {odd_modes} + broadband DFT (cpw={cpw_gsm})",
        },
        {
            "index": 4,
            "name": "silver_dispersive",
            "kind": "silver",
            "demo": False,
            "cpw": cpw_ag_bg,
            "f_max": F0,
            "cells_per_skin_depth": cpsd_ag_accepted,
            "description": (
                f"Dispersive Drude silver vs PEC screen resolved comparison "
                f"(cpw_bg={cpw_ag_bg}, cpsd_E={cpsd_ag_accepted})"
            ),
        },
        {
            "index": 5,
            "name": "silver_skin_depth_demo",
            "kind": "silver_ladder",
            "demo": True,
            "cpw": cpw_ag_bg,
            "f_max": F0,
            "cpsd_levels": cpsd_ag_ladder,
            "description": (
                f"DEMO: under-resolved (0.58, 1.15 cpsd_E) vs graded skin-depth "
                f"resolution ladder {cpsd_ag_ladder}"
            ),
        },
        {
            "index": 6,
            "name": "convergence_coarse",
            "kind": "conv",
            "demo": False,
            "cpw_list": cpw_sweep,
            "f_max": F0,
            "description": f"Grid convergence sweep cpw={cpw_sweep}",
        },
        {
            "index": 7,
            "name": "convergence_ref",
            "kind": "conv",
            "demo": False,
            "cpw_list": [cpw_ref],
            "f_max": F0,
            "description": f"Fine-grid convergence reference cpw={cpw_ref}",
        },
    ]


def build_task_meshes(task):
    """
    Build only the ``GridFIT3D`` mesh(es) for ``task`` (without running
    ``SolverFIT3D``) and return a list of resolution audit dictionaries from
    ``h.measure_mesh_resolution``.
    """
    kind = task["kind"]
    f_max = float(task.get("f_max", F0))
    lam_min_nm = (c_light / f_max) * 1e9
    audits = []

    if kind == "vcz":
        _, grid, _ = h.build_double_slit_solver(
            x_bounds=(-5.0e-6, 5.0e-6),
            y_bounds=(-8.5e-6, 6.4e-6),
            wavelength=WAVELENGTH_0,
            cpw=task["cpw"],
            n_pml=N_PML,
            slit_sep=SLIT_SEP,
            slit_width=SLIT_WIDTH,
            screen_thickness=SCREEN_THICK,
            screen_material="pec",
            grid_only=True,
        )
        info = h.measure_mesh_resolution(
            grid, WAVELENGTH_0, SLIT_SEP, SLIT_WIDTH, SCREEN_THICK, "pec"
        )
        info["cpw_shortest_wavelength"] = round(lam_min_nm / info["dx_background_nm"], 2)
        audits.append(info)
    elif kind == "gsm":
        _, grid, _ = h.build_double_slit_solver(
            x_bounds=(-5.0e-6, 5.0e-6),
            y_bounds=(-1.6e-6, 6.4e-6),
            wavelength=WAVELENGTH_0,
            cpw=task["cpw"],
            n_pml=N_PML,
            slit_sep=SLIT_SEP,
            slit_width=SLIT_WIDTH,
            screen_thickness=SCREEN_THICK,
            screen_material="pec",
            grid_only=True,
        )
        info = h.measure_mesh_resolution(
            grid, WAVELENGTH_0, SLIT_SEP, SLIT_WIDTH, SCREEN_THICK, "pec"
        )
        info["cpw_shortest_wavelength"] = round(lam_min_nm / info["dx_background_nm"], 2)
        audits.append(info)
    elif kind == "silver":
        _, grid, _ = h.build_double_slit_solver(
            x_bounds=(-4.5e-6, 4.5e-6),
            y_bounds=(-1.5e-6, 6.3e-6),
            wavelength=WAVELENGTH_0,
            cpw=task["cpw"],
            n_pml=N_PML,
            slit_sep=SLIT_SEP,
            slit_width=SLIT_WIDTH,
            screen_thickness=SCREEN_THICK,
            screen_material="silver",
            cells_per_skin_depth=task["cells_per_skin_depth"],
            grid_only=True,
        )
        info = h.measure_mesh_resolution(
            grid, WAVELENGTH_0, SLIT_SEP, SLIT_WIDTH, SCREEN_THICK, "silver"
        )
        info["cpw_shortest_wavelength"] = round(lam_min_nm / info["dx_background_nm"], 2)
        audits.append(info)
    elif kind == "silver_ladder":
        for lvl in task["cpsd_levels"]:
            if lvl == "uniform_10":
                cpw_use, cpsd_use = 10, "uniform"
            elif lvl == "uniform_20":
                cpw_use, cpsd_use = 20, "uniform"
            else:
                cpw_use, cpsd_use = int(task["cpw"]), float(lvl)
            _, grid, _ = h.build_double_slit_solver(
                x_bounds=(-4.5e-6, 4.5e-6),
                y_bounds=(-1.5e-6, 6.3e-6),
                wavelength=WAVELENGTH_0,
                cpw=cpw_use,
                n_pml=N_PML,
                slit_sep=SLIT_SEP,
                slit_width=SLIT_WIDTH,
                screen_thickness=SCREEN_THICK,
                screen_material="silver",
                cells_per_skin_depth=cpsd_use,
                grid_only=True,
            )
            info = h.measure_mesh_resolution(
                grid, WAVELENGTH_0, SLIT_SEP, SLIT_WIDTH, SCREEN_THICK, "silver"
            )
            info["cpw_shortest_wavelength"] = round(lam_min_nm / info["dx_background_nm"], 2)
            audits.append(info)
    elif kind == "conv":
        for cpw_val in task["cpw_list"]:
            _, grid, _ = h.build_double_slit_solver(
                x_bounds=(-5.0e-6, 5.0e-6),
                y_bounds=(-1.6e-6, 6.4e-6),
                wavelength=WAVELENGTH_0,
                cpw=int(cpw_val),
                n_pml=N_PML,
                slit_sep=SLIT_SEP,
                slit_width=SLIT_WIDTH,
                screen_thickness=SCREEN_THICK,
                screen_material="pec",
                grid_only=True,
            )
            info = h.measure_mesh_resolution(
                grid, WAVELENGTH_0, SLIT_SEP, SLIT_WIDTH, SCREEN_THICK, "pec"
            )
            info["cpw_shortest_wavelength"] = round(lam_min_nm / info["dx_background_nm"], 2)
            audits.append(info)
    else:
        raise ValueError(f"Unknown task kind {kind!r}")
    return audits


def run_single_task(task, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = task["index"]
    name = task["name"]
    t0 = time.perf_counter()

    if task["kind"] == "vcz":
        payload = run_vcz_dipole_batch(
            task["xs_positive"],
            cpw=task["cpw"],
            record_2d_indices=task["record_2d_indices"],
        )
    elif task["kind"] == "gsm":
        payload = run_gsm_mode_batch(
            task["modes"],
            cpw=task["cpw"],
            n_freqs_broadband=task["n_freqs_bb"],
            record_snapshots=task["record_snapshots"],
        )
    elif task["kind"] == "silver":
        payload = run_silver_vs_pec_study(
            cpw=task["cpw"],
            cells_per_skin_depth=task["cells_per_skin_depth"],
        )
    elif task["kind"] == "silver_ladder":
        payload = run_silver_skin_depth_ladder(
            cpw_bg=task["cpw"],
            cpsd_levels=task["cpsd_levels"],
        )
    elif task["kind"] == "conv":
        payload = run_resolution_convergence_sweep(task["cpw_list"])
    else:
        raise ValueError(f"Unknown task kind {task['kind']!r}")

    elapsed = time.perf_counter() - t0
    npz_path = out_dir / f"task_{idx:02d}_{name}.npz"
    json_path = out_dir / f"task_{idx:02d}_{name}.json"
    np.savez_compressed(npz_path, **payload)
    meta = {
        "index": idx,
        "name": name,
        "demo": bool(task.get("demo", False)),
        "description": task["description"],
        "elapsed_s": round(elapsed, 3),
        "dx_background_nm": float(payload["dx_background_nm"]),
        "dx_min_nm": float(payload["dx_min_nm"]),
        "min_feature_nm": float(payload["min_feature_nm"]),
        "cells_across_feature": int(payload["cells_across_feature"]),
        "n_cells": int(payload["n_cells"]),
        "n_steps": int(payload["n_steps"]),
    }
    if "skin_depth_nm" in payload:
        meta["skin_depth_nm"] = float(payload["skin_depth_nm"])
        meta["cells_per_skin_depth"] = float(payload["cells_per_skin_depth"])
        meta["skin_depth_power_nm"] = float(payload["skin_depth_power_nm"])
        meta["cells_per_skin_depth_power"] = float(payload["cells_per_skin_depth_power"])
    json_path.write_text(json.dumps(meta, indent=2) + "\n")
    skin_str = (
        f", skin_depth={meta['skin_depth_nm']:.2f}nm, "
        f"cpsd_E={meta['cells_per_skin_depth']:.2f} (cpsd_I={meta['cells_per_skin_depth_power']:.2f})"
        if "skin_depth_nm" in meta
        else ""
    )
    print(
        f"[task {idx:02d} {name}] done in {elapsed:.2f}s | "
        f"dx_bg={meta['dx_background_nm']:.2f}nm, dx_min={meta['dx_min_nm']:.2f}nm, "
        f"min_feat={meta['min_feature_nm']:.1f}nm ({meta['cells_across_feature']} cells)"
        f"{skin_str}, n_cells={meta['n_cells']}, n_steps={meta['n_steps']} -> {npz_path.name}"
    )


# %% [markdown]
# ## 7. Generating the 15-figure suite
#
# `generate_all_figures` reads `data/task_*.npz`, reconstructs the full van
# Cittert–Zernike dipole ensemble and Gaussian Schell-Model Mercer mode bank,
# evaluates all benchmark metrics against analytical formulas, writes
# `data/summary_metrics.json`, and renders 15 figures into `figures/`.


# %%
def _reconstruct_vcz_bank(d0, d1):
    """Combine Task 0 and Task 1 positive-x dipole runs with x -> -x mirror symmetry."""
    xs_pos = np.concatenate([d0["xs_positive"], d1["xs_positive"]])
    ez_det_pos = np.concatenate([d0["ez_det"], d1["ez_det"]], axis=0)
    ez_slit_pos = np.concatenate([d0["ez_slit"], d1["ez_slit"]], axis=0)
    x_det = d0["x_det"]
    x_slit = d0["x_slit"]

    # Mirror positive sources (xs > 0) to negative sources (-xs) via Ez(-x; -xs) = Ez(+x; +xs)
    pos_mask = xs_pos > 1e-12
    xs_neg = -xs_pos[pos_mask][::-1]
    ez_det_neg = ez_det_pos[pos_mask][::-1, ::-1]
    ez_slit_neg = ez_slit_pos[pos_mask][::-1, ::-1]

    xs_full = np.concatenate([xs_neg, xs_pos])
    ez_det_full = np.concatenate([ez_det_neg, ez_det_pos], axis=0)
    ez_slit_full = np.concatenate([ez_slit_neg, ez_slit_pos], axis=0)
    return xs_full, x_det, x_slit, ez_det_full, ez_slit_full


def _strip_overlap_weights(xs_full, width_D):
    """Quadrature weights for integrating a strip [-D/2, +D/2] over dipole nodes xs_full."""
    if width_D <= 1e-12:
        w = np.zeros_like(xs_full)
        w[int(np.argmin(np.abs(xs_full)))] = 1.0
        return w
    dxs = float(np.median(np.diff(xs_full)))
    half_D = 0.5 * float(width_D)
    cell_lo = xs_full - 0.5 * dxs
    cell_hi = xs_full + 0.5 * dxs
    overlap = np.clip(np.minimum(cell_hi, half_D) - np.maximum(cell_lo, -half_D), 0.0, dxs)
    total = float(np.sum(overlap))
    if total <= 1e-30:
        w = np.zeros_like(xs_full)
        w[int(np.argmin(np.abs(xs_full)))] = 1.0
        return w
    return overlap / total


def _reconstruct_gsm_bank(d_even, d_odd):
    """Merge even and odd Mercer mode runs sorted by mode index m = 0, 1, ..., M-1."""
    modes_all = np.concatenate([d_even["mode_indices"], d_odd["mode_indices"]])
    det_all = np.concatenate([d_even["ez_det_f0"], d_odd["ez_det_f0"]], axis=0)
    slit_all = np.concatenate([d_even["ez_slit_f0"], d_odd["ez_slit_f0"]], axis=0)
    order = np.argsort(modes_all)
    modes_sorted = modes_all[order]
    det_sorted = det_all[order]
    slit_sorted = slit_all[order]

    x_det = d_even["x_det"]
    x_slit = d_even["x_slit"]
    ix_s1 = int(np.argmin(np.abs(x_slit - (-0.5 * SLIT_SEP))))

    # Normalize each numerical mode so its slit-center amplitude matches the exact
    # analytical Hermite-Gaussian mode value Phi_m(-d/2; w_m)
    phi_s1_exact = np.array(
        [h.gsm_hermite_mode(int(m), -0.5 * SLIT_SEP, w_m=W_M_GSM) for m in modes_sorted]
    )
    scales = np.abs(phi_s1_exact) / np.maximum(np.abs(slit_sorted[:, ix_s1]), 1e-30)
    det_calibrated = det_sorted * scales[:, None]
    slit_calibrated = slit_sorted * scales[:, None]
    return modes_sorted, x_det, x_slit, det_calibrated, slit_calibrated


def _plot_fig00_setup_schematic(fig_dir, d4):
    """00a/00b: Setup geometry, mesh & excitation figures and tables via wakis.schematic.save_setup_figures."""
    _ = d4
    h.save_double_slit_setup_figures(fig_dir)


def _plot_fig01_schematic(fig_dir):
    """01: Annotated 2D geometry schematic of Young's double-slit coherence experiment."""
    fig, ax = figstyle.new_figure(nrows=1, ncols=1, width=10.0, height=4.4)
    ax.set_facecolor("#f8fafc")

    # Upstream incoherent strip source at y = -8 um
    ax.plot([-8.0, -8.0], [-1.5, 1.5], color="#d97706", lw=5.0, solid_capstyle="round")
    ys_dots = np.linspace(-1.4, 1.4, 9)
    ax.scatter(
        np.full_like(ys_dots, -8.0),
        ys_dots,
        s=42,
        color="#f59e0b",
        edgecolors="#78350f",
        zorder=5,
        label=r"Method A: Incoherent strip $D$ at $y=-L_1$",
    )
    ax.annotate(
        r"$D \in [0, 3.6]\;\mu\mathrm{m}$",
        xy=(-8.0, -1.55),
        xytext=(-8.9, -2.85),
        fontsize=8.5,
        fontweight="bold",
        color="#92400e",
        arrowprops=dict(arrowstyle="->", color="#92400e", lw=1.2),
    )

    # GSM Mercer mode injection plane at y = -1.0 um
    ax.plot(
        [-1.0, -1.0],
        [-3.8, 3.8],
        color="#2563eb",
        ls="--",
        lw=2.0,
        label=r"Method B: GSM Mercer modes $\Phi_m(x)$ at $y=-1\;\mu\mathrm{m}$",
    )
    x_prof = np.linspace(-3.5, 3.5, 200)
    phi0 = np.exp(-0.5 * (x_prof / 1.4) ** 2) * 0.7
    ax.fill_betweenx(x_prof, -1.0, -1.0 + phi0, color="#3b82f6", alpha=0.22)

    # Double-slit PEC/Ag screen at y = 0 (slits at x = +-1.0 um, width a = 0.4 um)
    for x_lo, x_hi in [(-4.8, -1.2), (-0.8, 0.8), (1.2, 4.8)]:
        rect = mpatches.Rectangle(
            (-0.15, x_lo),
            0.30,
            x_hi - x_lo,
            facecolor="#334155",
            edgecolor="#0f172a",
            lw=1.2,
            zorder=4,
        )
        ax.add_patch(rect)
    ax.text(
        0.0,
        3.15,
        "PEC / Drude Ag screen\n" + r"($t = 0.20\;\mu\mathrm{m},\;a = 0.40\;\mu\mathrm{m},\;d = 2.00\;\mu\mathrm{m}$)",
        fontsize=8.0,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#64748b", alpha=0.92),
    )

    # Cylindrical wave arcs from the two slits
    theta = np.linspace(-0.44 * np.pi, 0.44 * np.pi, 160)
    for r_arc in np.linspace(1.0, 5.8, 7):
        ax.plot(r_arc * np.cos(theta), -1.0 + r_arc * np.sin(theta), color="#0ea5e9", lw=0.8, alpha=0.45)
        ax.plot(r_arc * np.cos(theta), +1.0 + r_arc * np.sin(theta), color="#ec4899", lw=0.8, alpha=0.45)

    # Observation screen at y = +6.0 um with schematic fringe pattern
    ax.plot(
        [6.0, 6.0],
        [-4.5, 4.5],
        color="#059669",
        lw=2.5,
        label=r"Detector plane $y = L_2 = 6.0\;\mu\mathrm{m}$",
    )
    x_det_s = np.linspace(-4.2, 4.2, 300)
    env_s = np.exp(-0.08 * x_det_s**2)
    i_coh_s = env_s * (1.0 + np.cos(2.0 * np.pi * x_det_s / 1.5)) * 0.55
    i_inc_s = env_s * 0.55
    ax.plot(6.0 + i_coh_s, x_det_s, color="#047857", lw=1.8, label=r"Coherent fringes ($\mathcal{V}=1$)")
    ax.plot(6.0 + i_inc_s, x_det_s, color="#dc2626", ls=":", lw=1.8, label=r"Incoherent washout ($\mathcal{V}=0$)")

    # Dimension arrows
    ax.annotate("", xy=(0.0, -4.2), xytext=(-8.0, -4.2), arrowprops=dict(arrowstyle="<->", color="#475569", lw=1.3))
    ax.text(-4.0, -4.55, r"$L_1 = 8.0\;\mu\mathrm{m}$ ($16\lambda_0$)", ha="center", fontsize=8.5, color="#334155")
    ax.annotate("", xy=(6.0, -4.2), xytext=(0.0, -4.2), arrowprops=dict(arrowstyle="<->", color="#475569", lw=1.3))
    ax.text(3.0, -4.55, r"$L_2 = 6.0\;\mu\mathrm{m}$ ($12\lambda_0$)", ha="center", fontsize=8.5, color="#334155")

    ax.set_xlim(-9.4, 7.6)
    ax.set_ylim(-5.0, 4.8)
    ax.set_xlabel(r"Propagation coordinate $y$ [$\mu\mathrm{m}$]")
    ax.set_ylabel(r"Transverse coordinate $x$ [$\mu\mathrm{m}$]")
    ax.set_title(
        r"Young's Double-Slit Partial Coherence Geometry ($\lambda_0 = 0.50\;\mu\mathrm{m}$, 2D TM)"
    )
    ax.legend(loc="upper left", fontsize=7.8, framealpha=0.94)
    figstyle.save(fig, fig_dir / "01_double_slit_schematic.png")


def _plot_fig02_grid_and_slit_zoom(fig_dir, d4):
    """02: 2D FIT grid discretization and zoomed graded skin-depth slit aperture."""
    cpw_ag = int(d4["cpw"]) if "cpw" in d4 else 20
    cpsd_target = round(float(d4["cells_per_skin_depth"]))
    dx_min_nm = float(d4["dx_min_nm"])
    cpsd_e = float(d4["cells_per_skin_depth"])

    solver_pec, grid_pec, _ = h.build_double_slit_solver(
        x_bounds=(-3.0e-6, 3.0e-6),
        y_bounds=(-1.2e-6, 1.2e-6),
        wavelength=WAVELENGTH_0,
        cpw=20,
        screen_material="pec",
    )
    _, grid_ag, _ = h.build_double_slit_solver(
        x_bounds=(-3.0e-6, 3.0e-6),
        y_bounds=(-1.2e-6, 1.2e-6),
        wavelength=WAVELENGTH_0,
        cpw=cpw_ag,
        screen_material="silver",
        cells_per_skin_depth=cpsd_target,
        grid_only=True,
    )
    ieps_z = solver_pec.ieps.to_matrix("z")[:, :, 0]
    pec_mask = (ieps_z == 0.0).astype(float)

    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=10.8, height=4.0)
    xc = 0.5 * (grid_pec.x[:-1] + grid_pec.x[1:]) * 1e6
    yc = 0.5 * (grid_pec.y[:-1] + grid_pec.y[1:]) * 1e6

    im1 = ax1.pcolormesh(
        xc, yc, pec_mask[:-1, :-1].T, cmap="Greys", vmin=0.0, vmax=1.2, shading="auto"
    )
    h.overlay_double_slit_geometry(
        ax1,
        line_color="black",
        show_source=True,
        show_detector=False,
        show_labels=True,
        compact=False,
    )
    ax1.set_xlim(-2.6, 2.6)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_xlabel(r"Transverse coordinate $x$ [$\mu\mathrm{m}$]")
    ax1.set_ylabel(r"Propagation coordinate $y$ [$\mu\mathrm{m}$]")
    ax1.set_title(r"(a) Double-Slit Screen PEC Mask ($\mathrm{cpw}=20,\;\Delta x=25\;\mathrm{nm}$)")
    plt.colorbar(im1, ax=ax1, label=r"Screen conductor mask ($1-\varepsilon_r^{-1}$)")

    # Panel (b): Zoom around the upper slit x in [0.62, 1.38] um showing graded FIT grid lines and Ag skin depth
    ax2.pcolormesh(xc, yc, pec_mask[:-1, :-1].T, cmap="Blues", vmin=0.0, vmax=1.3, shading="auto")
    schematic.draw_mesh(
        ax2,
        grid_ag,
        plane="xy",
        unit="um",
        color="#334155",
        alpha=0.5,
        linewidth=0.4,
    )
    _, skin_m = h.silver_drude_material(WAVELENGTH_0)
    skin_um = skin_m * 1e6
    for x_edge, sign in [(0.80, -1.0), (1.20, +1.0)]:
        ax2.axvline(x_edge, color="#0f172a", lw=2.0)
        ax2.axvline(
            x_edge + sign * skin_um,
            color="#e11d48",
            ls="--",
            lw=1.8,
            label=rf"Ag skin depth $\delta_E = 28.8\;\mathrm{{nm}}$ (${cpsd_e:.1f}\,\Delta x_{{\min}}$)" if x_edge == 0.80 else None,
        )
    ax2.set_xlim(0.62, 1.38)
    ax2.set_ylim(-0.26, 0.26)
    ax2.set_xlabel(r"Transverse coordinate $x$ [$\mu\mathrm{m}$]")
    ax2.set_ylabel(r"Propagation coordinate $y$ [$\mu\mathrm{m}$]")
    ax2.set_title(
        rf"(b) Graded Slit & Skin-Depth Mesh ($\Delta x_{{\min}}={dx_min_nm:.2f}\;\mathrm{{nm}},\;{cpsd_e:.2f}\;\mathrm{{cells}}/\delta_E$)"
    )
    ax2.legend(loc="upper right", fontsize=7.5)

    figstyle.save(fig, fig_dir / "02_grid_and_slit_zoom.png")


def _plot_fig03_gsm_mercer_modes(fig_dir):
    """03: Hermite-Gaussian Mercer eigenmodes, eigenvalue spectra, and CSD W(x1, x2)."""
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=11.6, height=3.8)
    x = np.linspace(-3.8e-6, 3.8e-6, 401)

    for m in range(6):
        phi_m = h.gsm_hermite_mode(m, x, w_m=W_M_GSM) / np.sqrt(1e6)
        st = figstyle.series_style(m)
        axes[0].plot(x * 1e6, phi_m, color=st["color"], ls=st["linestyle"], lw=1.6, label=rf"$m={m}$")
    axes[0].axvline(-1.0, color="k", ls=":", lw=1.1)
    axes[0].axvline(+1.0, color="k", ls=":", lw=1.1, label=r"Slits $x=\pm 1\;\mu\mathrm{m}$")
    axes[0].set_xlabel(r"Transverse coordinate $x$ [$\mu\mathrm{m}$]")
    axes[0].set_ylabel(r"Eigenmode $\Phi_m(x)$ [$\mu\mathrm{m}^{-1/2}$]")
    axes[0].set_title(r"(a) Mercer Eigenmodes $\Phi_m(x)$ ($w_m = 1.4\;\mu\mathrm{m}$)")
    axes[0].legend(loc="upper right", fontsize=7.2, ncol=2)

    m_arr = np.arange(10)
    for idx_r, ratio in enumerate([0.5, 0.75, 1.0, 2.0]):
        info = h.gsm_parameters_from_modal_scale(ratio * SLIT_SEP, w_m=W_M_GSM, n_modes=10)
        axes[1].semilogy(
            m_arr,
            info["weights"],
            **figstyle.series_style(idx_r),
            label=rf"$\sigma_\mu / d = {ratio:.2f}\;(q={info['q']:.2f})$",
        )
    axes[1].set_xlabel(r"Mercer mode index $m$")
    axes[1].set_ylabel(r"Normalized modal weight $w_m = (1-q)q^m$")
    axes[1].set_title("(b) Starikov & Wolf (1982) Eigenvalue Spectrum")
    axes[1].legend(loc="upper right", fontsize=7.4)

    # Panel (c): Cross-spectral density W(x1, x2) reconstructed at sigma_mu / d = 0.75
    info_c = h.gsm_parameters_from_modal_scale(0.75 * SLIT_SEP, w_m=W_M_GSM, n_modes=12)
    x_c = np.linspace(-3.2e-6, 3.2e-6, 121)
    phis = np.array([h.gsm_hermite_mode(m, x_c, w_m=W_M_GSM) for m in range(12)])
    w_csd = (info_c["weights"][:, None, None] * phis[:, :, None] * phis[:, None, :]).sum(axis=0)
    w_csd /= np.max(w_csd)
    im = axes[2].pcolormesh(x_c * 1e6, x_c * 1e6, w_csd, cmap=figstyle.CMAP_MAGNITUDE, shading="auto")
    axes[2].plot([-1.0, 1.0], [1.0, -1.0], "wo", ms=6, mec="k", label=r"Slits $(\pm d/2, \mp d/2)$")
    axes[2].set_xlabel(r"$x_1$ [$\mu\mathrm{m}$]")
    axes[2].set_ylabel(r"$x_2$ [$\mu\mathrm{m}$]")
    axes[2].set_title(r"(c) Reconstructed CSD $W(x_1, x_2)/W(0,0)$ ($\sigma_\mu/d=0.75$)")
    axes[2].legend(loc="upper left", fontsize=7.2)
    plt.colorbar(im, ax=axes[2], label=r"Normalized $W(x_1, x_2)$")

    figstyle.save(fig, fig_dir / "03_gsm_mercer_eigenmodes.png")


def _plot_fig04_coherent_vs_incoherent_2d(fig_dir, d0, d2, d3):
    """04 & 04b: 3-panel physical Mercer mode synthesis + 2-panel unphysical coherent sum pitfall."""
    x_2d = d2["x_2d"] * 1e6
    y_2d = d2["y_2d"] * 1e6
    ez_m0 = d2["ez_2d_m0"]
    ez_m1 = d3["ez_2d_m1"]

    # Normalize m=0 and m=1 to equal slit intensity
    iy_slit = int(np.argmin(np.abs(d2["y_2d"] - 0.0)))
    ix_s1 = int(np.argmin(np.abs(d2["x_2d"] - (-0.5 * SLIT_SEP))))
    e0 = ez_m0 / np.abs(ez_m0[ix_s1, iy_slit])
    e1 = ez_m1 / np.abs(ez_m1[ix_s1, iy_slit])

    i_m0 = np.abs(e0) ** 2
    i_m1 = np.abs(e1) ** 2
    # Incoherent power sum (50% m=0 + 50% m=1 -> mu_12 = 0, complete fringe washout)
    i_inc = 0.5 * i_m0 + 0.5 * i_m1
    # Erroneous coherent field sum |0.707*e0 + 0.707*e1|^2 -> illuminates only 1 slit / skewed!
    i_coh_err = np.abs((e0 + e1) / np.sqrt(2.0)) ** 2

    panels_main = [
        (i_m0, "(a) Symmetric Mode $m=0$\n($\\mu_{12}=+1,\\;\\mathcal{V}=1.00$)"),
        (i_m1, "(b) Antisymmetric Mode $m=1$\n($\\mu_{12}=-1$, dark center)"),
        (i_inc, "(c) Incoherent Power Sum\n$\\sum_m w_m |E_z^{(m)}|^2$ ($\\mathcal{V}=0$)"),
    ]

    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=12.6, height=4.4)
    for ax, (field_i, title) in zip(axes.ravel(), panels_main):
        mask_down = d2["y_2d"] >= 0.25e-6
        vmax = float(np.percentile(field_i[:, mask_down], 99.2))
        im = ax.pcolormesh(
            x_2d, y_2d, np.clip(field_i.T / vmax, 0.0, 1.0), cmap=figstyle.CMAP_MAGNITUDE, shading="auto"
        )
        h.overlay_double_slit_geometry(
            ax,
            line_color="white",
            show_source=True,
            show_detector=True,
            show_labels=True,
            compact=True,
        )
        ax.set_xlabel(r"$x$ [$\mu\mathrm{m}$]")
        ax.set_ylabel(r"$y$ [$\mu\mathrm{m}$]")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label=r"Normalized $|E_z(x,y)|^2$")

    figstyle.save(fig, fig_dir / "04_coherent_vs_incoherent_2d_intensity.png")

    # Companion 2-panel figure contrasting physical incoherent power sum vs unphysical coherent field sum
    fig_b, axes_b = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.1)
    for ax, (field_i, title) in zip(
        axes_b.ravel(),
        [
            (i_inc, r"(a) Physical Incoherent Sum $\sum_m w_m |E_z^{(m)}|^2$ ($\mathcal{V}=0$)"),
            (i_coh_err, r"(b) Erroneous Coherent Sum $|\sum_m \sqrt{w_m} E_z^{(m)}|^2$ (Skewed)"),
        ],
    ):
        mask_down = d2["y_2d"] >= 0.25e-6
        vmax = float(np.percentile(field_i[:, mask_down], 99.2))
        im = ax.pcolormesh(
            x_2d, y_2d, np.clip(field_i.T / vmax, 0.0, 1.0), cmap=figstyle.CMAP_MAGNITUDE, shading="auto"
        )
        h.overlay_double_slit_geometry(
            ax,
            line_color="white",
            show_source=True,
            show_detector=True,
            show_labels=True,
            compact=True,
        )
        ax.set_xlabel(r"$x$ [$\mu\mathrm{m}$]")
        ax.set_ylabel(r"$y$ [$\mu\mathrm{m}$]")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label=r"Normalized $|E_z(x,y)|^2$")

    figstyle.save(fig_b, fig_dir / "04b_coherent_vs_incoherent_error_2d.png")


def _plot_fig05_field_snapshots(fig_dir, d2):
    """05: Three time-domain wavepacket snapshots Ez(x, y, t) during double-slit diffraction."""
    frames = np.asarray(d2["td_frames_ez"], dtype=float) * (
        float(d2["td_frames_ez_scale"]) if "td_frames_ez_scale" in d2 else 1.0
    )
    times_fs = d2["td_frames_t_fs"]
    tx = d2["td_x"] * 1e6
    ty = d2["td_y"] * 1e6
    pick = [3, 9, 18]

    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=11.8, height=4.1)
    vmax = float(np.max(np.abs(frames))) * 0.45
    vmin, vmax = figstyle.symmetric_limits(np.array([-vmax, vmax]))
    for idx_p, (ax, idx) in enumerate(zip(axes.ravel(), pick)):
        snap = frames[idx]
        im = ax.pcolormesh(
            tx, ty, snap.T, cmap=figstyle.CMAP_DIVERGING, vmin=vmin, vmax=vmax, shading="auto"
        )
        h.overlay_double_slit_geometry(
            ax,
            line_color="black",
            show_source=True,
            show_detector=True,
            show_labels=True,
            compact=True,
        )
        ax.set_xlabel(r"$x$ [$\mu\mathrm{m}$]")
        ax.set_ylabel(r"$y$ [$\mu\mathrm{m}$]")
        ax.set_title(rf"({chr(ord('a') + idx_p)}) $E_z(x,y,t)$ at $t = {times_fs[idx]:.1f}\;\mathrm{{fs}}$")
        plt.colorbar(im, ax=ax, label=r"$E_z$ [V/m, arb.]")

    figstyle.save(fig, fig_dir / "05_field_snapshots_ez.png")


def _plot_fig06_poynting_and_slit_profile(fig_dir, d2):
    """06: Time-averaged Poynting vector map <S(x,y)> and slit aperture mode profile."""
    ez_raw = d2["ez_2d_m0"]
    hx_raw = d2["hx_2d_m0"]
    hy_raw = d2["hy_2d_m0"]
    nx_c = min(ez_raw.shape[0], hx_raw.shape[0], hy_raw.shape[0])
    ny_c = min(ez_raw.shape[1], hx_raw.shape[1], hy_raw.shape[1])
    ez = ez_raw[:nx_c, :ny_c]
    hx = hx_raw[:nx_c, :ny_c]
    hy = hy_raw[:nx_c, :ny_c]
    x_2d = d2["x_2d"][:nx_c] * 1e6
    y_2d = d2["y_2d"][:ny_c] * 1e6

    sx = -0.5 * np.real(ez * np.conj(hy))
    sy = +0.5 * np.real(ez * np.conj(hx))
    s_mag = np.hypot(sx, sy)

    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.1)
    mask_down = y_2d >= 0.2
    vmax = float(np.percentile(s_mag[:, mask_down], 99.0))
    im = ax1.pcolormesh(
        x_2d, y_2d, np.clip(s_mag.T / vmax, 0.0, 1.0), cmap=figstyle.CMAP_MAGNITUDE, shading="auto"
    )
    step_q = max(1, len(x_2d) // 24)
    Xq, Yq = np.meshgrid(x_2d[::step_q], y_2d[::step_q], indexing="ij")
    sx_q = sx[::step_q, ::step_q]
    sy_q = sy[::step_q, ::step_q]
    norm_q = np.maximum(np.hypot(sx_q, sy_q), vmax * 0.02)
    ax1.quiver(
        Xq,
        Yq,
        sx_q / norm_q,
        sy_q / norm_q,
        color="white",
        alpha=0.72,
        scale=28,
        width=0.003,
    )
    h.overlay_double_slit_geometry(
        ax1,
        line_color="white",
        show_source=True,
        show_detector=True,
        show_labels=True,
        compact=False,
    )
    ax1.set_xlabel(r"$x$ [$\mu\mathrm{m}$]")
    ax1.set_ylabel(r"$y$ [$\mu\mathrm{m}$]")
    ax1.set_title(r"(a) Time-Averaged Poynting Vector $\langle\mathbf{S}(x,y)\rangle$")
    plt.colorbar(im, ax=ax1, label=r"Normalized $|\langle\mathbf{S}\rangle|$")

    # Panel (b): Slit plane field profile vs analytical fundamental TM_0 waveguide mode cos(pi*u/a)
    x_slit = d2["x_slit"] * 1e6
    ez_slit = np.abs(d2["ez_slit_f0"][0])
    ez_slit /= np.max(ez_slit)
    st0 = figstyle.series_style(0)
    st1 = figstyle.series_style(1)
    ax2.plot(
        x_slit,
        ez_slit,
        color=st0["color"],
        ls="-",
        marker="o",
        ms=3.2,
        lw=1.6,
        label=r"Wakis 2D FIT $|E_z(x, y=0)|$",
    )

    u_fine = np.linspace(-0.20, 0.20, 101)
    cos_mode = np.cos(np.pi * u_fine / 0.40)
    ax2.plot(-1.0 + u_fine, cos_mode, ls="--", color=st1["color"], lw=2.0, label=r"Analytical $\mathrm{TM}_0$ slit mode $\cos(\pi u / a)$")
    ax2.plot(+1.0 + u_fine, cos_mode, ls="--", color=st1["color"], lw=2.0)
    ax2.set_xlim(-2.2, 2.2)
    ax2.set_ylim(-0.05, 1.12)
    ax2.set_xlabel(r"Slit plane coordinate $x$ [$\mu\mathrm{m}$]")
    ax2.set_ylabel(r"Normalized slit aperture amplitude $|E_z(x,0)|$")
    ax2.set_title(r"(b) Double-Slit Aperture Field ($a = 0.40\;\mu\mathrm{m},\;d = 2.00\;\mu\mathrm{m}$)")
    ax2.legend(loc="upper right", fontsize=8.0)

    figstyle.save(fig, fig_dir / "06_poynting_and_slit_profile.png")


def _plot_fig07_propagation_gif(fig_dir, d2):
    """07: Animated GIF movie of the TM wavepacket propagating through the double slit."""
    frames = np.asarray(d2["td_frames_ez"], dtype=float) * (
        float(d2["td_frames_ez_scale"]) if "td_frames_ez_scale" in d2 else 1.0
    )
    times_fs = d2["td_frames_t_fs"]
    tx = d2["td_x"] * 1e6
    ty = d2["td_y"] * 1e6
    vmax = float(np.max(np.abs(frames))) * 0.42
    vmin, vmax = figstyle.symmetric_limits(np.array([-vmax, vmax]))

    fig, ax = figstyle.new_figure(nrows=1, ncols=1, width=5.6, height=4.6)
    fig.set_dpi(105)
    quad = ax.pcolormesh(
        tx, ty, frames[0].T, cmap=figstyle.CMAP_DIVERGING, vmin=vmin, vmax=vmax, shading="auto"
    )
    h.overlay_double_slit_geometry(
        ax,
        line_color="black",
        show_source=True,
        show_detector=True,
        show_labels=True,
        compact=False,
    )
    ax.set_xlabel(r"Transverse coordinate $x$ [$\mu\mathrm{m}$]")
    ax.set_ylabel(r"Propagation coordinate $y$ [$\mu\mathrm{m}$]")
    title = ax.set_title(rf"2D FIT Double-Slit Wavepacket: $t = {times_fs[0]:.1f}\;\mathrm{{fs}}$")
    plt.colorbar(quad, ax=ax, label=r"$E_z(x,y,t)$ [arb.]")
    figstyle.finish(fig)

    def _update(frame_idx):
        quad.set_array(frames[frame_idx].T.ravel())
        title.set_text(rf"2D FIT Double-Slit Wavepacket: $t = {times_fs[frame_idx]:.1f}\;\mathrm{{fs}}$")
        return quad, title

    ani = animation.FuncAnimation(fig, _update, frames=len(frames), interval=90, blit=False)
    ani.save(fig_dir / "07_double_slit_propagation.gif", writer="pillow", fps=11)
    plt.close(fig)


def _plot_fig08_and_fig09_vcz(fig_dir, xs_full, x_det, x_slit, ez_det_full, ez_slit_full):
    """08 & 09: Van Cittert-Zernike fringe profiles (with phase reversal) and visibility vs D."""
    ix_s1 = int(np.argmin(np.abs(x_slit - (-0.5 * SLIT_SEP))))
    ix_s2 = int(np.argmin(np.abs(x_slit - (+0.5 * SLIT_SEP))))

    # Normalize each dipole field so total slit power is uniform across xs
    slit_pwr = 0.5 * (np.abs(ez_slit_full[:, ix_s1]) ** 2 + np.abs(ez_slit_full[:, ix_s2]) ** 2)
    ez_det_norm = ez_det_full / np.sqrt(np.maximum(slit_pwr, 1e-30))[:, None]
    ez_s1_norm = ez_slit_full[:, ix_s1] / np.sqrt(np.maximum(slit_pwr, 1e-30))
    ez_s2_norm = ez_slit_full[:, ix_s2] / np.sqrt(np.maximum(slit_pwr, 1e-30))

    # Figure 08: Screen fringe profiles at selected strip widths D / D0
    d_ratios_prof = [0.0, 0.4, 0.7, 1.0, 1.4]
    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.1)

    i0_ref = None
    for idx_r, ratio in enumerate(d_ratios_prof):
        width_D = ratio * D0_VCZ
        w = _strip_overlap_weights(xs_full, width_D)
        i_prof = h.incoherent_power_sum(ez_det_norm, weights=w)
        if i0_ref is None:
            i0_ref = float(np.max(i_prof))
        i_prof_n = i_prof / i0_ref
        vis, mu_s = h.extract_fringe_visibility(
            x_det, i_prof_n, d=SLIT_SEP, L2=L2_DET, wavelength=WAVELENGTH_0
        )
        st = figstyle.series_style(idx_r)
        ax1.plot(
            x_det * 1e6,
            i_prof_n,
            lw=1.7,
            color=st["color"],
            ls=st["linestyle"],
            label=rf"$D/D_0={ratio:.1f}\;(\mu_{{12}}={mu_s:+.2f},\;\mathcal{{V}}={vis:.2f})$",
        )

    ax1.axvline(0.0, color="k", ls=":", lw=1.1)
    ax1.set_xlim(-4.2, 4.2)
    ax1.set_xlabel(r"Detector coordinate $x$ [$\mu\mathrm{m}$]")
    ax1.set_ylabel(r"Normalized screen intensity $I(x)/I_0$")
    ax1.set_title(r"(a) Wakis 2D FIT Fringe Profiles vs Strip Width $D$")
    ax1.legend(loc="upper right", fontsize=7.5)

    # Panel (b): Direct zoom on phase reversal at D/D0 = 1.4 (mu_12 < 0) vs D/D0 = 0.0 and 1.0
    for idx_r, (ratio, ls_style) in enumerate([(0.0, "-"), (1.0, "--"), (1.4, "-")]):
        width_D = ratio * D0_VCZ
        w = _strip_overlap_weights(xs_full, width_D)
        i_prof_n = h.incoherent_power_sum(ez_det_norm, weights=w) / i0_ref
        mu_th = h.van_cittert_zernike_2d_hankel(width_D, SLIT_SEP, WAVELENGTH_0, L1_VCZ)
        i_th, _ = h.double_slit_2d_hankel_intensity(
            x_det, mu_th, d=SLIT_SEP, a=SLIT_WIDTH, L2=L2_DET, wavelength=WAVELENGTH_0
        )
        col = figstyle.series_style(idx_r)["color"]
        ax2.plot(
            x_det * 1e6,
            i_prof_n,
            ls=ls_style,
            lw=2.0,
            color=col,
            label=rf"Wakis FIT $D/D_0={ratio:.1f}$",
        )
        ax2.plot(
            x_det * 1e6,
            i_th * (np.mean(i_prof_n) / np.mean(i_th)),
            ls=":",
            lw=1.4,
            color=col,
            alpha=0.85,
        )
    ax2.axvline(0.0, color="k", ls=":", lw=1.2, label=r"Optical axis $x=0$ (phase reversal!)")
    ax2.set_xlim(-2.4, 2.4)
    ax2.set_xlabel(r"Detector coordinate $x$ [$\mu\mathrm{m}$]")
    ax2.set_ylabel(r"Normalized intensity $I(x)/I_0$")
    ax2.set_title(r"(b) Secondary-Lobe Phase Reversal ($\mu_{12} < 0$ at $D = 1.4\,D_0$)")
    ax2.legend(loc="upper right", fontsize=7.5)

    figstyle.save(fig, fig_dir / "08_vcz_fringe_profiles_vs_D.png")

    # Figure 09: Visibility V(D) and signed coherence mu_12(D) vs D / D0
    d_sweep = np.linspace(0.0, 3.6e-6, 19)
    vis_sim = []
    mu_sim = []
    mu_slit_sim = []
    for width_D in d_sweep:
        w = _strip_overlap_weights(xs_full, width_D)
        i_prof = h.incoherent_power_sum(ez_det_norm, weights=w)
        v_val, m_val = h.extract_fringe_visibility(
            x_det, i_prof, d=SLIT_SEP, L2=L2_DET, wavelength=WAVELENGTH_0
        )
        vis_sim.append(v_val)
        mu_sim.append(m_val)
        mu_slit = h.complex_degree_of_coherence(ez_s1_norm, ez_s2_norm, weights=w)
        mu_slit_sim.append(float(np.real(mu_slit)))

    vis_sim = np.asarray(vis_sim)
    mu_sim = np.asarray(mu_sim)
    mu_slit_sim = np.asarray(mu_slit_sim)

    d_fine = np.linspace(0.0, 3.6e-6, 201)
    u_fine = d_fine / D0_VCZ
    u_sweep = d_sweep / D0_VCZ
    mu_sinc_fine = h.van_cittert_zernike_sinc(d_fine, SLIT_SEP, WAVELENGTH_0, L1_VCZ)
    mu_hankel_fine = h.van_cittert_zernike_2d_hankel(d_fine, SLIT_SEP, WAVELENGTH_0, L1_VCZ)

    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.1)
    st0 = figstyle.series_style(0)
    st1 = figstyle.series_style(1)
    st2 = figstyle.series_style(2)
    ax1.plot(u_fine, np.abs(mu_sinc_fine), "--", color="#64748b", lw=1.7, label=r"Paraxial VCZ $|\mathrm{sinc}(D/D_0)|$ (Zernike 1938)")
    ax1.plot(u_fine, np.abs(mu_hankel_fine), "-", color=st1["color"], lw=2.0, label=r"Exact 2D Hankel $|W_{12}|/\sqrt{I_1 I_2}$")
    ax1.plot(u_sweep, vis_sim, "o", color=st0["color"], ms=5.8, label=r"Wakis 2D FIT screen visibility $\mathcal{V}$")
    ax1.axvline(1.0, color="#d97706", ls=":", lw=1.4, label=r"First null $D_0 = \lambda_0 L_1 / d = 2.0\;\mu\mathrm{m}$")
    ax1.set_xlabel(r"Normalized strip width $D / D_0 = D d / (\lambda_0 L_1)$")
    ax1.set_ylabel(r"Fringe visibility $\mathcal{V} = |\mu_{12}|$")
    ax1.set_title(r"(a) Fringe Visibility $\mathcal{V}(D)$ vs Van Cittert–Zernike Theory")
    ax1.legend(loc="upper right", fontsize=7.6)

    ax2.plot(u_fine, mu_sinc_fine, "--", color="#64748b", lw=1.7, label=r"Paraxial $\mathrm{sinc}(D/D_0)$ (Born & Wolf Fig. 10.11)")
    ax2.plot(u_fine, mu_hankel_fine, "-", color=st1["color"], lw=2.0, label=r"Exact 2D Hankel $\mathrm{Re}\,\mu_{12}(D)$")
    ax2.plot(u_sweep, mu_sim, "o", color=st0["color"], ms=5.8, label=r"Wakis 2D FIT signed fringe contrast $\mu_{\mathrm{fit}}$")
    ax2.plot(u_sweep, mu_slit_sim, "s", color=st2["color"], ms=4.6, alpha=0.85, label=r"Wakis slit-plane CSD $\mathrm{Re}\,\mu_{12}$")
    ax2.axhline(0.0, color="k", lw=1.0)
    ax2.fill_between(u_fine, mu_hankel_fine, 0.0, where=(mu_hankel_fine < 0), color="#fecaca", alpha=0.45, label=r"Phase-reversed lobe ($\mu_{12} < 0$)")
    ax2.set_xlabel(r"Normalized strip width $D / D_0 = D d / (\lambda_0 L_1)$")
    ax2.set_ylabel(r"Signed degree of spatial coherence $\mathrm{Re}\,\mu_{12}$")
    ax2.set_title(r"(b) Signed Coherence & Secondary-Lobe Phase Reversal")
    ax2.legend(loc="upper right", fontsize=7.4)

    figstyle.save(fig, fig_dir / "09_vcz_visibility_vs_source_width.png")

    return {
        "d_sweep_um": (d_sweep * 1e6).tolist(),
        "u_sweep": u_sweep.tolist(),
        "vis_sim": vis_sim.tolist(),
        "mu_sim": mu_sim.tolist(),
        "mu_sinc": h.van_cittert_zernike_sinc(d_sweep, SLIT_SEP, WAVELENGTH_0, L1_VCZ).tolist(),
        "mu_hankel": h.van_cittert_zernike_2d_hankel(d_sweep, SLIT_SEP, WAVELENGTH_0, L1_VCZ).tolist(),
    }


def _plot_fig10_gsm(fig_dir, modes_sorted, x_det, det_calibrated):
    """10: Gaussian Schell-Model fringe profiles and visibility vs sigma_mu / d."""
    n_modes = len(modes_sorted)
    ratios_prof = [0.4, 0.6, 0.8, 1.2, np.inf]

    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.1)

    for idx_r, ratio in enumerate(ratios_prof):
        smu = ratio * SLIT_SEP if np.isfinite(ratio) else np.inf
        info = h.gsm_parameters_from_modal_scale(smu, w_m=W_M_GSM, n_modes=n_modes)
        i_prof = h.incoherent_power_sum(det_calibrated, weights=info["weights"])
        i_prof_n = i_prof / np.mean(i_prof[np.abs(x_det) <= 3.0e-6])
        v_sim, _ = h.extract_fringe_visibility(
            x_det, i_prof_n, d=SLIT_SEP, L2=L2_DET, wavelength=WAVELENGTH_0
        )
        lbl = (
            rf"$\sigma_\mu / d = \infty\;(\mathcal{{V}}={v_sim:.3f})$"
            if np.isinf(ratio)
            else rf"$\sigma_\mu / d = {ratio:.1f}\;(\mathcal{{V}}={v_sim:.3f})$"
        )
        st = figstyle.series_style(idx_r)
        ax1.plot(x_det * 1e6, i_prof_n, lw=1.7, color=st["color"], ls=st["linestyle"], label=lbl)

    ax1.set_xlim(-4.2, 4.2)
    ax1.set_xlabel(r"Detector coordinate $x$ [$\mu\mathrm{m}$]")
    ax1.set_ylabel(r"Normalized screen intensity $I(x)/\langle I\rangle$")
    ax1.set_title(rf"(a) GSM Fringe Profiles via $M={n_modes}$ Mercer Modes")
    ax1.legend(loc="upper right", fontsize=7.6)

    # Panel (b): Visibility vs sigma_mu / d compared against exp(-d^2 / (2*sigma_mu^2))
    ratio_sweep = np.array([0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0])
    vis_gsm_sim = []
    vis_gsm_modal = []
    for ratio in ratio_sweep:
        smu = ratio * SLIT_SEP
        info = h.gsm_parameters_from_modal_scale(smu, w_m=W_M_GSM, n_modes=n_modes)
        i_prof = h.incoherent_power_sum(det_calibrated, weights=info["weights"])
        v_s, _ = h.extract_fringe_visibility(
            x_det, i_prof, d=SLIT_SEP, L2=L2_DET, wavelength=WAVELENGTH_0
        )
        vis_gsm_sim.append(v_s)
        vis_gsm_modal.append(
            abs(h.gsm_modal_slit_coherence(smu, d=SLIT_SEP, w_m=W_M_GSM, n_modes=n_modes))
        )

    vis_gsm_sim = np.asarray(vis_gsm_sim)
    vis_gsm_modal = np.asarray(vis_gsm_modal)
    r_fine = np.linspace(0.25, 5.2, 200)
    vis_gsm_ana_fine = h.gsm_visibility_analytic(SLIT_SEP, r_fine * SLIT_SEP)
    vis_gsm_ana = h.gsm_visibility_analytic(SLIT_SEP, ratio_sweep * SLIT_SEP)

    st0 = figstyle.series_style(0)
    st1 = figstyle.series_style(1)
    st2 = figstyle.series_style(2)
    ax2.plot(
        r_fine,
        vis_gsm_ana_fine,
        "-",
        color=st1["color"],
        lw=2.0,
        label=r"Analytical GSM $\exp[-d^2/(2\sigma_\mu^2)]$ (Mandel & Wolf §5.6.4)",
    )
    ax2.plot(
        ratio_sweep,
        vis_gsm_modal,
        "s--",
        color=st2["color"],
        ms=5.0,
        lw=1.3,
        label=rf"Exact $M={n_modes}$ Mercer truncation $|\mu_{{12}}^{{(M)}}|$",
    )
    ax2.plot(
        ratio_sweep,
        vis_gsm_sim,
        "o",
        color=st0["color"],
        ms=6.0,
        label=rf"Wakis 2D FIT screen visibility ($M={n_modes}$)",
    )
    ax2.set_xlabel(r"Normalized spatial coherence width $\sigma_\mu / d$")
    ax2.set_ylabel(r"Fringe visibility $\mathcal{V} = |\mu_{12}|$")
    ax2.set_title(r"(b) GSM Fringe Visibility vs Coherence Width $\sigma_\mu / d$")
    ax2.legend(loc="lower right", fontsize=7.6)

    figstyle.save(fig, fig_dir / "10_gsm_fringe_profiles_and_visibility.png")

    return {
        "ratio_sweep": ratio_sweep.tolist(),
        "vis_gsm_sim": vis_gsm_sim.tolist(),
        "vis_gsm_modal": vis_gsm_modal.tolist(),
        "vis_gsm_ana": vis_gsm_ana.tolist(),
    }


def _plot_fig11_temporal_coherence(fig_dir, d2, d3):
    """11: Broadband temporal coherence profiles and Wiener-Khinchin contrast envelope."""
    freqs = d2["freqs_bb"]
    x_det = d2["x_det"]
    x_slit = d2["x_slit"]
    ix_s1 = int(np.argmin(np.abs(x_slit - (-0.5 * SLIT_SEP))))

    scale_0 = 1.0 / np.maximum(np.abs(d2["dft_slit_m0"][:, ix_s1]), 1e-30)[:, None]
    scale_1 = 1.0 / np.maximum(np.abs(d3["dft_slit_m1"][:, ix_s1]), 1e-30)[:, None]
    dft_even = d2["dft_det_m0"] * scale_0
    dft_odd = d3["dft_det_m1"] * scale_1
    pulse = GaussianSource(F0, 0.22 * F0)

    bw_list = [0.0, 0.04, 0.08, 0.12, 0.16]
    delta_L = np.hypot(x_det + 0.5 * SLIT_SEP, L2_DET) - np.hypot(x_det - 0.5 * SLIT_SEP, L2_DET)

    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.1)
    temporal_summary = {}

    for idx_b, bw in enumerate(bw_list):
        i_poly, i_inc, contrast = h.polychromatic_screen_intensity(
            freqs, dft_even, dft_odd, pulse, F0, rel_bw_sigma=bw, x_det=x_det
        )
        gamma_wk, Lc = h.temporal_coherence_envelope(
            delta_L, wavelength_0=WAVELENGTH_0, rel_bw_sigma=bw
        )
        lbl = (
            r"Monochromatic $\sigma_f/f_0 = 0\%\;(L_c = \infty)$"
            if bw == 0.0
            else rf"$\sigma_f/f_0 = {100*bw:.0f}\%\;(L_c = {Lc*1e6:.2f}\;\mu\mathrm{{m}})$"
        )
        col = figstyle.series_style(idx_b)["color"]
        ax1.plot(x_det * 1e6, i_poly, lw=1.7, color=col, label=lbl)
        if bw in (0.04, 0.08, 0.16):
            ax2.plot(
                delta_L / WAVELENGTH_0,
                contrast,
                lw=1.7,
                color=col,
                label=rf"Wakis FIT $\sigma_f/f_0={100*bw:.0f}\%$",
            )
            ax2.plot(
                delta_L / WAVELENGTH_0,
                gamma_wk,
                "--",
                lw=1.5,
                color=col,
                label=rf"Wiener–Khinchin $\pm\exp(-\Delta L^2/2L_c^2)$ ({100*bw:.0f}%)",
            )
            ax2.plot(delta_L / WAVELENGTH_0, -gamma_wk, "--", lw=1.5, color=col)

        m1_mask = np.abs(delta_L / WAVELENGTH_0 - 1.0) < 0.22
        m2_mask = np.abs(delta_L / WAVELENGTH_0 - 2.0) < 0.25
        c_m1_sim = float(np.max(contrast[m1_mask]))
        c_m2_sim = float(np.max(contrast[m2_mask]))
        c_m1_ana, _ = h.temporal_coherence_envelope(WAVELENGTH_0, WAVELENGTH_0, bw)
        c_m2_ana, _ = h.temporal_coherence_envelope(2.0 * WAVELENGTH_0, WAVELENGTH_0, bw)
        temporal_summary[f"bw_{int(round(100*bw)):02d}pct"] = {
            "Lc_um": None if np.isinf(Lc) else round(Lc * 1e6, 4),
            "m1_sim": round(c_m1_sim, 4),
            "m1_ana": round(float(c_m1_ana), 4),
            "m2_sim": round(c_m2_sim, 4),
            "m2_ana": round(float(c_m2_ana), 4),
        }

    ax1.set_xlim(-4.2, 4.2)
    ax1.set_xlabel(r"Detector coordinate $x$ [$\mu\mathrm{m}$]")
    ax1.set_ylabel(r"Polychromatic intensity $I_{\mathrm{poly}}(x)$ [arb.]")
    ax1.set_title(r"(a) Broadband Polychromatic Fringe Profiles ($y = L_2$)")
    ax1.legend(loc="upper right", fontsize=7.4)

    ax2.set_xlim(-2.6, 2.6)
    ax2.set_ylim(-1.08, 1.08)
    ax2.set_xlabel(r"Optical path difference $\Delta L(x) / \lambda_0 = [r_1(x)-r_2(x)]/\lambda_0$")
    ax2.set_ylabel(r"Normalized fringe modulation $(I_{\mathrm{poly}}-I_{\mathrm{inc}})/I_{\mathrm{inc}}$")
    ax2.set_title(r"(b) Wiener–Khinchin Temporal Coherence Envelope $|\gamma_{11}(\Delta L)|$")
    ax2.legend(loc="upper right", fontsize=7.0, ncol=2)

    figstyle.save(fig, fig_dir / "11_temporal_coherence_broadband.png")
    return temporal_summary


def _plot_fig12_ensemble_convergence(
    fig_dir, xs_full, x_det, x_slit, ez_det_full, ez_slit_full, modes_sorted, det_calibrated
):
    """12: Convergence of fringe visibility vs number of incoherent sources N_s and Mercer modes M."""
    ix_s1 = int(np.argmin(np.abs(x_slit - (-0.5 * SLIT_SEP))))
    ix_s2 = int(np.argmin(np.abs(x_slit - (+0.5 * SLIT_SEP))))
    slit_pwr = 0.5 * (np.abs(ez_slit_full[:, ix_s1]) ** 2 + np.abs(ez_slit_full[:, ix_s2]) ** 2)
    ez_det_norm = ez_det_full / np.sqrt(np.maximum(slit_pwr, 1e-30))[:, None]

    # Panel (a): Method A convergence with number of symmetric point dipoles N_s across D
    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.1)
    ns_base = [1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 29, 33, len(xs_full)]
    ns_list = sorted(list({n for n in ns_base if n <= len(xs_full)}))

    for idx_r, ratio_D in enumerate([0.5, 1.0, 1.4]):
        width_D = ratio_D * D0_VCZ
        vis_vs_ns = []
        for ns in ns_list:
            if ns == 1:
                idx_pick = [len(xs_full) // 2]
                i_sub = np.abs(ez_det_norm[idx_pick[0]]) ** 2
                v_sub, _ = h.extract_fringe_visibility(
                    x_det, i_sub, d=SLIT_SEP, L2=L2_DET, wavelength=WAVELENGTH_0
                )
                vis_vs_ns.append(v_sub)
                continue
            pts = np.linspace(-0.5 * width_D, 0.5 * width_D, ns)
            re_interp = np.array([np.interp(pts, xs_full, ez_det_norm[:, ix].real) for ix in range(len(x_det))]).T
            im_interp = np.array([np.interp(pts, xs_full, ez_det_norm[:, ix].imag) for ix in range(len(x_det))]).T
            f_sub = re_interp + 1j * im_interp
            w_sub = np.ones(ns) / ns
            w_sub[0] *= 0.5
            w_sub[-1] *= 0.5
            w_sub /= np.sum(w_sub)
            i_sub = h.incoherent_power_sum(f_sub, weights=w_sub)
            v_sub, _ = h.extract_fringe_visibility(
                x_det, i_sub, d=SLIT_SEP, L2=L2_DET, wavelength=WAVELENGTH_0
            )
            vis_vs_ns.append(v_sub)

        vis_vs_ns = np.asarray(vis_vs_ns)
        err_ns = np.abs(vis_vs_ns - vis_vs_ns[-1]) + 1e-4
        ax1.semilogy(
            ns_list[:-1],
            err_ns[:-1],
            **figstyle.series_style(idx_r),
            label=rf"$D/D_0 = {ratio_D:.1f}\;(\mathcal{{V}}_\infty={vis_vs_ns[-1]:.3f})$",
        )

    ax1.set_xlabel(r"Number of incoherent point emitters $N_s$")
    ax1.set_ylabel(r"Visibility quadrature error $|\mathcal{V}(N_s) - \mathcal{V}(N_{\max})|$")
    ax1.set_title(r"(a) Method A Convergence vs Number of Point Sources $N_s$")
    ax1.legend(loc="upper right", fontsize=7.6)

    # Panel (b): Method B convergence vs number of Mercer modes M = 1..M_max
    m_counts = np.arange(1, len(modes_sorted) + 1)
    for idx_r, ratio_smu in enumerate([0.5, 0.75, 1.25]):
        smu = ratio_smu * SLIT_SEP
        info = h.gsm_parameters_from_modal_scale(smu, w_m=W_M_GSM, n_modes=len(modes_sorted))
        v_inf = abs(h.gsm_modal_slit_coherence(smu, d=SLIT_SEP, w_m=W_M_GSM, n_modes=20))
        err_m = []
        for m_cut in m_counts:
            v_m = abs(h.gsm_modal_slit_coherence(smu, d=SLIT_SEP, w_m=W_M_GSM, n_modes=int(m_cut)))
            err_m.append(max(abs(v_m - v_inf), 1e-6))
        st = figstyle.series_style(idx_r)
        ax2.semilogy(
            m_counts,
            err_m,
            **st,
            label=rf"Wakis Mercer $\sigma_\mu/d={ratio_smu:.2f}\;(q={info['q']:.2f})$",
        )
        ax2.semilogy(
            m_counts,
            info["q"] ** m_counts,
            "--",
            lw=1.2,
            color=st["color"],
            alpha=0.75,
            label=rf"Asymptotic bound $q^M$ ($\sigma_\mu/d={ratio_smu:.2f}$)",
        )

    ax2.set_xlabel(r"Number of coherent Mercer modes $M$")
    ax2.set_ylabel(r"Modal truncation error $|\mathcal{V}^{(M)} - \mathcal{V}^{(\infty)}|$")
    ax2.set_title(r"(b) Method B Exponential Convergence vs Mode Count $M$")
    ax2.legend(loc="upper right", fontsize=7.2, ncol=1)

    figstyle.save(fig, fig_dir / "12_ensemble_and_mode_convergence.png")


def _plot_fig13_silver_vs_pec(fig_dir, d4, d5_ag):
    """13: Dispersive Drude silver screen vs PEC screen comparison on resolved graded mesh."""
    x_slit = d4["x_slit"] * 1e6
    x_det = d4["x_det"] * 1e6
    skin_nm = float(d4["skin_depth_m"]) * 1e9
    cpsd_acc = float(d4["cells_per_skin_depth"])
    dx_min_nm = float(d4["dx_min_nm"])

    slit_pec = np.abs(d4["ez_slit_pec_m0"])
    slit_ag = np.abs(d4["ez_slit_silver_m0"])
    norm_s = float(np.max(slit_pec))
    slit_pec_n = slit_pec / norm_s
    slit_ag_n = slit_ag / norm_s

    det_pec = np.abs(d4["ez_det_pec_m0"]) ** 2
    det_ag = np.abs(d4["ez_det_silver_m0"]) ** 2
    ix0 = int(np.argmin(np.abs(d4["x_det"] - 0.0)))
    norm_d = float(det_pec[ix0])
    det_pec_n = det_pec / norm_d
    det_ag_n = det_ag / norm_d
    ratio_0 = float(det_ag_n[ix0])

    v_pec, _ = h.extract_fringe_visibility(
        d4["x_det"], det_pec_n, d=SLIT_SEP, L2=L2_DET, wavelength=WAVELENGTH_0
    )
    v_ag, _ = h.extract_fringe_visibility(
        d4["x_det"], det_ag_n, d=SLIT_SEP, L2=L2_DET, wavelength=WAVELENGTH_0
    )

    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.1)
    st0 = figstyle.series_style(0)
    st1 = figstyle.series_style(1)
    st2 = figstyle.series_style(3)
    ax1.plot(x_slit, slit_pec_n, "-", color=st0["color"], lw=1.8, label="Ideal PEC screen (resolved)")
    ax1.plot(
        x_slit,
        slit_ag_n,
        "--",
        color=st1["color"],
        lw=1.8,
        label=rf"Drude Ag resolved ($\Delta x_{{\min}}={dx_min_nm:.2f}\;\mathrm{{nm}},\;{cpsd_acc:.1f}\;\mathrm{{cells}}/\delta_E$)",
    )
    if "x_slit_coarse" in d5_ag:
        xs_c = d5_ag["x_slit_coarse"] * 1e6
        ag_c = np.abs(d5_ag["ez_slit_silver_coarse"]) / float(np.max(np.abs(d5_ag["ez_slit_pec_coarse"])))
        ax1.plot(
            xs_c,
            ag_c,
            "o:",
            color=st2["color"],
            ms=3.6,
            lw=1.4,
            alpha=0.85,
            label=r"DEMO: Under-resolved Ag ($\mathrm{cpw}=10,\;0.58\;\mathrm{cells}/\delta_E$)",
        )
    ax1.axvline(0.80, color="k", ls=":", lw=1.1, label=r"Geometric slit walls $x \in [0.8, 1.2]\;\mu\mathrm{m}$")
    ax1.axvline(1.20, color="k", ls=":", lw=1.1)
    ax1.set_xlim(0.45, 1.55)
    ax1.set_xlabel(r"Slit-plane coordinate $x$ [$\mu\mathrm{m}$]")
    ax1.set_ylabel(r"Normalized slit electric field $|E_z(x, 0)|$")
    ax1.set_title(r"(a) Upper-Slit Aperture Field & Skin-Depth Penetration")
    ax1.legend(loc="upper right", fontsize=7.4)

    ax2.plot(
        x_det,
        det_pec_n,
        "-",
        color=st0["color"],
        lw=1.8,
        label=rf"PEC screen ($\mathcal{{V}} = {v_pec:.4f}$)",
    )
    ax2.plot(
        x_det,
        det_ag_n,
        "--",
        color=st1["color"],
        lw=1.8,
        label=rf"Drude Ag screen ($\mathcal{{V}} = {v_ag:.4f},\;I_{{\mathrm{{Ag}}}}(0)/I_{{\mathrm{{PEC}}}}(0)={ratio_0:.4f}$)",
    )
    ax2.set_xlim(-4.2, 4.2)
    ax2.set_xlabel(r"Detector coordinate $x$ [$\mu\mathrm{m}$]")
    ax2.set_ylabel(r"Normalized detector intensity $I(x)/I_{\mathrm{PEC}}(0)$")
    ax2.set_title(r"(b) Downstream Interference Fringes: PEC vs Dispersive Ag")
    ax2.legend(loc="upper right", fontsize=7.6)

    figstyle.save(fig, fig_dir / "13_pec_vs_silver_screen.png")
    return {
        "skin_depth_nm": round(skin_nm, 2),
        "skin_depth_power_nm": round(0.5 * skin_nm, 2),
        "dx_min_nm": round(dx_min_nm, 2),
        "cells_per_skin_depth": round(cpsd_acc, 2),
        "cells_per_skin_depth_power": round(float(d4["cells_per_skin_depth_power"]), 2),
        "vis_pec": round(v_pec, 4),
        "vis_silver": round(v_ag, 4),
        "peak_ratio_silver_to_pec": round(ratio_0, 4),
    }


def _plot_fig14_mesh_convergence(fig_dir, d6, d7):
    """14: Spatial mesh resolution convergence against fine reference run."""
    cpw_coarse = d6["cpw_list"]
    cpw_ref = int(d7["cpw_list"][0])
    i_pc_ref = d7["i_pc_norm"][0]
    i_coh_ref = d7["i_coh_norm"][0]
    vis_pc_ref = float(d7["vis_pc"][0])

    l2_err_pc = [
        float(np.linalg.norm(d6["i_pc_norm"][k] - i_pc_ref) / np.linalg.norm(i_pc_ref))
        for k in range(len(cpw_coarse))
    ]
    l2_err_coh = [
        float(np.linalg.norm(d6["i_coh_norm"][k] - i_coh_ref) / np.linalg.norm(i_coh_ref))
        for k in range(len(cpw_coarse))
    ]
    vis_err_pc = [
        float(abs(d6["vis_pc"][k] - vis_pc_ref)) for k in range(len(cpw_coarse))
    ]

    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.1)
    ax1.loglog(
        cpw_coarse,
        l2_err_pc,
        **figstyle.series_style(0),
        label=r"Partially coherent $L_2$ fringe error ($\mu_{12}=0.5$)",
    )
    ax1.loglog(
        cpw_coarse,
        l2_err_coh,
        **figstyle.series_style(2),
        label=r"Coherent $L_2$ fringe error ($\mu_{12}=1.0$)",
    )
    ref_slope = l2_err_pc[0] * (cpw_coarse[0] / cpw_coarse.astype(float)) ** 2
    ax1.loglog(
        cpw_coarse,
        ref_slope,
        ":",
        color=figstyle.series_style(1)["color"],
        lw=1.8,
        label=r"Second-order FIT reference $\mathcal{O}(\Delta x^2)$",
    )
    ax1.set_xlabel(r"Grid resolution $\mathrm{cpw} = \lambda_0 / \Delta x$ [cells/wavelength]")
    ax1.set_ylabel(rf"Relative $L_2$ screen intensity error vs $\mathrm{{cpw}}={cpw_ref}$")
    ax1.set_title(rf"(a) Spatial Mesh Convergence (Reference $\mathrm{{cpw}}={cpw_ref}$)")
    ax1.legend(loc="upper right", fontsize=7.6)

    all_cpw = np.append(cpw_coarse, cpw_ref)
    all_times = np.append(d6["runtimes_s"], d7["runtimes_s"])
    ax2.plot(
        all_cpw,
        all_times,
        **figstyle.series_style(4),
        label="2-mode solve wall time [s]",
    )
    ax2.set_xlabel(r"Grid resolution $\mathrm{cpw} = \lambda_0 / \Delta x$ [cells/wavelength]")
    ax2.set_ylabel("Wall-clock runtime for 2 modes [s]")
    ax2.set_title(r"(b) Computational Cost Scaling $\mathcal{O}(\mathrm{cpw}^3)$ in 2D Time-Domain")
    ax2.legend(loc="upper left", fontsize=7.8)

    figstyle.save(fig, fig_dir / "14_mesh_resolution_convergence.png")

    return {
        "cpw_coarse": cpw_coarse.tolist(),
        "cpw_ref": cpw_ref,
        "dx_bg_nm": [round(float(x), 3) for x in np.append(d6["dx_bg_nm_list"], d7["dx_bg_nm_list"])],
        "dx_min_nm": [round(float(x), 3) for x in np.append(d6["dx_min_nm_list"], d7["dx_min_nm_list"])],
        "cells_across_feature": [int(x) for x in np.append(d6["cells_across_feature_list"], d7["cells_across_feature_list"])],
        "n_cells": [int(x) for x in np.append(d6["n_cells_list"], d7["n_cells_list"])],
        "n_steps": [int(x) for x in np.append(d6["n_steps_list"], d7["n_steps_list"])],
        "vis_pc": [round(float(x), 5) for x in np.append(d6["vis_pc"], d7["vis_pc"])],
        "vis_coh": [round(float(x), 5) for x in np.append(d6["vis_coh"], d7["vis_coh"])],
        "l2_err_pc": [round(x, 5) for x in l2_err_pc],
        "l2_err_coh": [round(x, 5) for x in l2_err_coh],
        "vis_err_pc": [round(x, 5) for x in vis_err_pc],
        "runtimes_s": [round(float(x), 2) for x in all_times],
    }


def _plot_fig15_silver_skin_depth_resolution(fig_dir, d4, d5_ag):
    """15: Skin-depth resolution study on I_Ag(0)/I_PEC(0) vs cells per skin depth."""
    det_pec_acc = np.abs(d4["ez_det_pec_m0"]) ** 2
    det_ag_acc = np.abs(d4["ez_det_silver_m0"]) ** 2
    ix0 = int(np.argmin(np.abs(d4["x_det"] - 0.0)))
    r_acc = float(det_ag_acc[ix0] / det_pec_acc[ix0])
    cpsd_e_acc = float(d4["cells_per_skin_depth"])
    cpsd_i_acc = float(d4["cells_per_skin_depth_power"])
    dx_min_acc = float(d4["dx_min_nm"])

    cpsd_e_all = np.append(d5_ag["cpsd_field_ladder"], cpsd_e_acc)
    cpsd_i_all = np.append(d5_ag["cpsd_power_ladder"], cpsd_i_acc)
    dx_min_all = np.append(d5_ag["dx_min_nm_ladder"], dx_min_acc)
    r_all = np.append(d5_ag["ratio_ag_to_pec"], r_acc)
    ncells_all = np.append(d5_ag["n_cells_ladder"], int(d4["n_cells"]))
    nsteps_all = np.append(d5_ag["n_steps_ladder"], int(d4["n_steps"]))

    rel_step_pct = np.abs(np.diff(r_all)) / r_all[:-1] * 100.0
    rel_err_vs_acc_pct = np.abs(r_all - r_acc) / r_acc * 100.0

    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.1)
    st0 = figstyle.series_style(0)
    st1 = figstyle.series_style(1)
    st2 = figstyle.series_style(2)
    ax1.plot(
        cpsd_e_all,
        r_all,
        "o-",
        color=st0["color"],
        ms=6.0,
        lw=1.8,
        label=r"Graded mesh $I_{\mathrm{Ag}}(0) / I_{\mathrm{PEC}}(0)$",
    )
    ax1.plot(
        cpsd_e_all[:2],
        r_all[:2],
        "s",
        color="#d97706",
        ms=8.0,
        mec="#78350f",
        zorder=5,
        label=r"DEMO: Under-resolved uniform ($\mathrm{cpw}=10, 20$)",
    )
    ax1.plot(
        [cpsd_e_acc],
        [r_acc],
        "*",
        color=st1["color"],
        ms=13.0,
        mec="#881337",
        zorder=6,
        label=rf"Accepted level: {cpsd_e_acc:.2f} cells/$\delta_E$ ({cpsd_i_acc:.2f} cells/$\delta_I$, $R={r_acc:.4f}$)",
    )
    ax1.axhline(r_acc, color=st1["color"], ls="--", lw=1.2, alpha=0.8)
    ax1.axhspan(r_acc * 0.995, r_acc * 1.005, color="#fecdd3", alpha=0.38, label=r"$\pm 0.5\%$ convergence band")
    ax1.axvline(8.0, color=st2["color"], ls=":", lw=1.4, label=r"Rule 3 threshold: $8\;\mathrm{cells}/\delta_E = 4\;\mathrm{cells}/\delta_I$")
    ax1.set_xlabel(r"Cells per electric-field skin depth $\delta_E / \Delta x_{\min}$ ($\delta_E = 28.8\;\mathrm{nm}$)")
    ax1.set_ylabel(r"On-axis transmission ratio $R = I_{\mathrm{Ag}}(0) / I_{\mathrm{PEC}}(0)$")
    ax1.set_title(r"(a) Silver Screen Transmission Ratio vs Skin-Depth Resolution")
    ax1.legend(loc="upper right", fontsize=7.2)

    ax2.semilogy(
        cpsd_e_all[1:],
        rel_step_pct,
        "o-",
        color=st0["color"],
        ms=6.0,
        lw=1.8,
        label=r"Step-to-step relative change $|R_k - R_{k-1}| / R_{k-1}$ [%]",
    )
    ax2.semilogy(
        cpsd_e_all[:-1],
        rel_err_vs_acc_pct[:-1],
        "s--",
        color=st2["color"],
        ms=5.5,
        lw=1.7,
        label=rf"Relative error vs accepted ($R={r_acc:.4f}$) [%]",
    )
    ax2.plot(
        [cpsd_e_acc],
        [rel_step_pct[-1]],
        "*",
        color=st1["color"],
        ms=12.5,
        mec="#881337",
        zorder=6,
        label=rf"Accepted step change: {rel_step_pct[-1]:.2f}% (< 1% threshold)",
    )
    ax2.axhline(1.0, color=st1["color"], ls=":", lw=1.5, label="1.0% relative acceptance threshold")
    ax2.set_xlabel(r"Cells per electric-field skin depth $\delta_E / \Delta x_{\min}$")
    ax2.set_ylabel("Relative change / error [%]")
    ax2.set_title(r"(b) Skin-Depth Convergence Rate & Acceptance Criterion")
    ax2.legend(loc="upper right", fontsize=7.4)

    figstyle.save(fig, fig_dir / "15_silver_skin_depth_resolution.png")

    return {
        "cpsd_field": [round(float(x), 2) for x in cpsd_e_all],
        "cpsd_power": [round(float(x), 2) for x in cpsd_i_all],
        "dx_min_nm": [round(float(x), 2) for x in dx_min_all],
        "n_cells": [int(x) for x in ncells_all],
        "n_steps": [int(x) for x in nsteps_all],
        "ratio_ag_to_pec": [round(float(x), 5) for x in r_all],
        "rel_step_change_pct": [round(float(x), 4) for x in rel_step_pct],
        "accepted_cpsd_field": round(cpsd_e_acc, 2),
        "accepted_cpsd_power": round(cpsd_i_acc, 2),
        "accepted_ratio_ag_to_pec": round(r_acc, 4),
        "last_step_rel_change_pct": round(float(rel_step_pct[-1]), 4),
    }


def _extract_npz_res(d_npz, name, demo=False, lam_min_nm=500.0):
    cells_feat = int(d_npz["cells_across_feature"])
    cells_slit = int(d_npz["cells_across_slit"]) if "cells_across_slit" in d_npz else 2 * cells_feat
    info = {
        "name": name,
        "demo": demo,
        "dx_background_nm": float(d_npz["dx_background_nm"]),
        "dx_min_nm": float(d_npz["dx_min_nm"]),
        "min_feature_nm": float(d_npz["min_feature_nm"]),
        "cells_across_feature": cells_feat,
        "cells_across_slit": cells_slit,
        "cpw_shortest_wavelength": round(lam_min_nm / float(d_npz["dx_background_nm"]), 2),
        "n_cells": int(d_npz["n_cells"]),
        "n_steps": int(d_npz["n_steps"]),
    }
    if "skin_depth_nm" in d_npz:
        info["skin_depth_nm"] = float(d_npz["skin_depth_nm"])
        info["cells_per_skin_depth"] = float(d_npz["cells_per_skin_depth"])
        info["skin_depth_power_nm"] = float(d_npz["skin_depth_power_nm"])
        info["cells_per_skin_depth_power"] = float(d_npz["cells_per_skin_depth_power"])
    return info


def generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR):
    """Load data/task_*.npz, render all figures (including 00_setup_schematic.png), and save summary_metrics.json."""
    figstyle.apply_style()
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    d0 = np.load(data_dir / "task_00_vcz_batch_a.npz")
    d1 = np.load(data_dir / "task_01_vcz_batch_b.npz")
    d2 = np.load(data_dir / "task_02_gsm_modes_even.npz")
    d3 = np.load(data_dir / "task_03_gsm_modes_odd.npz")
    d4 = np.load(data_dir / "task_04_silver_dispersive.npz")
    d5_ag = np.load(data_dir / "task_05_silver_skin_depth_demo.npz")
    d6 = np.load(data_dir / "task_06_convergence_coarse.npz")
    d7 = np.load(data_dir / "task_07_convergence_ref.npz")

    xs_full, x_det_vcz, x_slit_vcz, ez_det_vcz, ez_slit_vcz = _reconstruct_vcz_bank(d0, d1)
    modes_sorted, x_det_gsm, _, det_cal_gsm, _ = _reconstruct_gsm_bank(d2, d3)

    _plot_fig00_setup_schematic(fig_dir, d4)
    _plot_fig01_schematic(fig_dir)
    _plot_fig02_grid_and_slit_zoom(fig_dir, d4)
    _plot_fig03_gsm_mercer_modes(fig_dir)
    _plot_fig04_coherent_vs_incoherent_2d(fig_dir, d0, d2, d3)
    _plot_fig05_field_snapshots(fig_dir, d2)
    _plot_fig06_poynting_and_slit_profile(fig_dir, d2)
    _plot_fig07_propagation_gif(fig_dir, d2)
    vcz_metrics = _plot_fig08_and_fig09_vcz(
        fig_dir, xs_full, x_det_vcz, x_slit_vcz, ez_det_vcz, ez_slit_vcz
    )
    gsm_metrics = _plot_fig10_gsm(fig_dir, modes_sorted, x_det_gsm, det_cal_gsm)
    temp_metrics = _plot_fig11_temporal_coherence(fig_dir, d2, d3)
    _plot_fig12_ensemble_convergence(
        fig_dir,
        xs_full,
        x_det_vcz,
        x_slit_vcz,
        ez_det_vcz,
        ez_slit_vcz,
        modes_sorted,
        det_cal_gsm,
    )
    ag_metrics = _plot_fig13_silver_vs_pec(fig_dir, d4, d5_ag)
    conv_metrics = _plot_fig14_mesh_convergence(fig_dir, d6, d7)
    ag_ladder_metrics = _plot_fig15_silver_skin_depth_resolution(fig_dir, d4, d5_ag)

    lam_min_gsm_nm = 500.0 / 1.28
    task_res = [
        _extract_npz_res(d0, "vcz_batch_a", demo=False, lam_min_nm=500.0),
        _extract_npz_res(d1, "vcz_batch_b", demo=False, lam_min_nm=500.0),
        _extract_npz_res(d2, "gsm_modes_even", demo=False, lam_min_nm=lam_min_gsm_nm),
        _extract_npz_res(d3, "gsm_modes_odd", demo=False, lam_min_nm=lam_min_gsm_nm),
        _extract_npz_res(d4, "silver_dispersive", demo=False, lam_min_nm=500.0),
        _extract_npz_res(d5_ag, "silver_skin_depth_demo", demo=True, lam_min_nm=500.0),
        _extract_npz_res(d6, "convergence_coarse", demo=False, lam_min_nm=500.0),
        _extract_npz_res(d7, "convergence_ref", demo=False, lam_min_nm=500.0),
    ]

    summary = {
        "vcz": vcz_metrics,
        "gsm": gsm_metrics,
        "temporal": temp_metrics,
        "silver_vs_pec": ag_metrics,
        "silver_skin_depth_study": ag_ladder_metrics,
        "convergence": conv_metrics,
        "task_resolutions": task_res,
    }
    (data_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Rendered 17 figures in {fig_dir} and wrote {data_dir / 'summary_metrics.json'}")


# %% [markdown]
# ## 8. Quantitative results & benchmark comparison tables
#
# ### 8.1 Method A: Extended incoherent strip source (van Cittert–Zernike theorem)
#
# Comparing the Wakis 2D FIT extended strip simulation (`08_vcz_fringe_profiles_vs_D.png`
# and `09_vcz_visibility_vs_source_width.png`, simulated in `--full` at $\text{cpw}_{\text{bg}} = 20$,
# $\Delta x = 25.00\;\text{nm}$, $N_s = 37$ dipoles with $\Delta x_s = 0.10\;\mu\text{m}$, and in
# `--quick` at $\text{cpw}_{\text{bg}} = 10$, $\Delta x = 50.00\;\text{nm}$, $N_s = 19$ dipoles with
# $\Delta x_s = 0.20\;\mu\text{m}$) against both the paraxial van Cittert–Zernike formula
# $\mu_{12}(D) = \operatorname{sinc}(D/D_0)$ (van Cittert 1934; Zernike 1938;
# Born & Wolf §10.4.2, Fig. 10.11) and the exact 2D Hankel Green's function integral
# (`van_cittert_zernike_2d_hankel`). Doubling the dipole density ($N_s = 19 \to 37$)
# and spatial grid resolution ($\text{cpw} = 10 \to 20$) reduces the RMS error
# against the exact 2D Hankel curve across all 19 widths from $0.0329$ (`--quick`)
# to $0.0124$ (`--full`), and cuts the residual visibility at the first coherence null
# ($D = D_0 = 2.00\;\mu\text{m}$) from $0.0593$ (`--quick`) to $0.0269$ (`--full`):
#
# | Strip width $D$ [$\mu\text{m}$] | $D / D_0$ | Paraxial $\operatorname{sinc}(D/D_0)$ | 2D Hankel $\operatorname{Re}\mu_{12}$ | `--quick` signed $\mu_{\text{fit}}$ | `--full` signed $\mu_{\text{fit}}$ | `--full` visibility $\mathcal{V}$ | Regime |
# | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
# | $0.00$ | $0.00$ | $+1.0000$ | $+1.0000$ | $+1.0000$ | $+0.9960$ | $0.9960$ | Spatially coherent point emitter |
# | $0.40$ | $0.20$ | $+0.9355$ | $+0.9365$ | $+0.9168$ | $+0.9266$ | $0.9266$ | Near-coherent strip |
# | $0.80$ | $0.40$ | $+0.7568$ | $+0.7607$ | $+0.7559$ | $+0.7559$ | $0.7559$ | Primary lobe (high contrast) |
# | $1.00$ | $0.50$ | $+0.6366$ | $+0.6424$ | $+0.6774$ | $+0.6409$ | $0.6409$ | Half-null width ($2/\pi$ paraxial) |
# | $1.20$ | $0.60$ | $+0.5046$ | $+0.5124$ | $+0.5278$ | $+0.5145$ | $0.5145$ | Half-visibility transition |
# | $2.00$ | $1.00$ | $+0.0000$ | $+0.0123$ | $+0.0593$ | $+0.0269$ | $0.0269$ | First van Cittert–Zernike null ($D = D_0$) |
# | $2.80$ | $1.40$ | $-0.2162$ | $-0.2179$ | $-0.1839$ | $-0.1994$ | $0.1994$ | Phase-inverted secondary lobe ($\mu_{12} < 0$) |
# | $3.00$ | $1.50$ | $-0.2122$ | $-0.2206$ | $-0.2055$ | $-0.2029$ | $0.2029$ | Secondary lobe peak contrast ($\mu_{12} < 0$) |
# | $3.60$ | $1.80$ | $-0.1039$ | $-0.1330$ | $-0.1443$ | $-0.1225$ | $0.1225$ | Approach to second null ($2 D_0$) |
#
# ### 8.2 Method B: Gaussian Schell-Model beam via Starikov & Wolf Mercer modes
#
# Comparing the $M = 12$ Hermite–Gaussian Mercer mode expansion (`10_gsm_fringe_profiles_and_visibility.png`,
# simulated in `--full` at $\text{cpw}_{\text{bg}} = 26$, $\Delta x_{\text{bg}} = 19.23\;\text{nm}$,
# $\Delta x_{\min} = 18.18\;\text{nm}$ across the slits, vs $M = 10$ modes at
# $\text{cpw}_{\text{bg}} = 13$ in `--quick`) against the analytical Gaussian Schell-Model
# visibility $\mathcal{V}_{\text{GSM}}(\sigma_\mu / d) = \exp[-d^2 / (2\sigma_\mu^2)]$
# (Starikov & Wolf 1982; Mandel & Wolf §5.6.4; Goodman §5.4). Across all 10
# coherence ratios $\sigma_\mu / d \in [0.40, 5.00]$, the `--full` maximum absolute
# deviation from the analytical GSM law is $|\Delta\mathcal{V}|_{\max} = 0.0103$
# (RMS error $0.0067$, vs $0.0112$ max and $0.0073$ RMS in `--quick`):
#
# | Coherence ratio $\sigma_\mu / d$ | Geometric ratio $q$ | Analytical $\exp[-d^2/(2\sigma_\mu^2)]$ | $M=12$ Mercer $|\mu_{12}^{(12)}|$ | `--quick` ($M=10$) FIT $\mathcal{V}$ | `--full` ($M=12$) FIT $\mathcal{V}$ | `--full` error $|\Delta\mathcal{V}|$ |
# | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | $0.40$ | $0.7440$ | $0.0439$ | $0.0422$ | $0.0451$ | $0.0516$ | $0.0077$ |
# | $0.50$ | $0.6629$ | $0.1353$ | $0.1350$ | $0.1448$ | $0.1454$ | $0.0100$ |
# | $0.60$ | $0.5797$ | $0.2494$ | $0.2493$ | $0.2605$ | $0.2596$ | $0.0103$ |
# | $0.75$ | $0.4648$ | $0.4111$ | $0.4111$ | $0.4217$ | $0.4199$ | $0.0088$ |
# | $1.00$ | $0.3139$ | $0.6065$ | $0.6065$ | $0.6152$ | $0.6117$ | $0.0052$ |
# | $1.25$ | $0.2192$ | $0.7261$ | $0.7261$ | $0.7333$ | $0.7282$ | $0.0021$ |
# | $1.50$ | $0.1595$ | $0.8007$ | $0.8007$ | $0.8069$ | $0.8006$ | $0.0001$ |
# | $2.00$ | $0.0936$ | $0.8825$ | $0.8825$ | $0.8875$ | $0.8797$ | $0.0028$ |
# | $3.00$ | $0.0426$ | $0.9460$ | $0.9460$ | $0.9501$ | $0.9409$ | $0.0051$ |
# | $5.00$ | $0.0156$ | $0.9802$ | $0.9802$ | $0.9838$ | $0.9738$ | $0.0064$ |
#
# ### 8.3 Method C: Broadband temporal coherence & dispersive silver screen
#
# Comparing the polychromatic `DFTMonitor` fringe modulation at the first
# ($|\Delta L| = \lambda_0$) and second ($|\Delta L| = 2\lambda_0$) off-axis
# interference peaks (`11_temporal_coherence_broadband.png`, resolved in `--full` with
# $81$ frequency channels and $20.31\;\text{cells}/\lambda_{\min}$ at $f_{\max} = 1.28\,f_0$,
# vs $57$ channels and $10.16\;\text{cells}/\lambda_{\min}$ in `--quick`)
# against the Wiener–Khinchin formula $|\gamma_{11}(\Delta L)| = \exp(-\Delta L^2 / (2 L_c^2))$:
#
# | Fractional bandwidth $\sigma_f / f_0$ | Coherence length $L_c$ [$\mu\text{m}$] | $|\gamma_{11}(\lambda_0)|$ theory | `--quick` $m=1$ peak | `--full` $m=1$ peak | $|\gamma_{11}(2\lambda_0)|$ theory | `--quick` $m=2$ peak | `--full` $m=2$ peak |
# | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | $0\%$ (monochromatic) | $\infty$ | $1.0000$ | $0.9822$ | $0.9793$ | $1.0000$ | $0.9494$ | $0.9447$ |
# | $4\%$ | $1.9894$ | $0.9689$ | $0.9554$ | $0.9497$ | $0.8813$ | $0.8426$ | $0.8345$ |
# | $8\%$ | $0.9947$ | $0.8813$ | $0.8800$ | $0.8684$ | $0.6033$ | $0.5908$ | $0.5767$ |
# | $12\%$ | $0.6631$ | $0.7526$ | $0.7865$ | $0.7688$ | $0.3208$ | $0.3462$ | $0.3289$ |
# | $16\%$ | $0.4974$ | $0.6033$ | $0.7182$ | $0.6979$ | $0.1325$ | $0.1924$ | $0.1746$ |
#
# For the **dispersive Drude silver screen** at $\lambda_0 = 500\;\text{nm}$
# (`13_pec_vs_silver_screen.png` and `15_silver_skin_depth_resolution.png`,
# $\varepsilon_{\text{Ag}} = -7.632 - 0.731i$, electric-field skin depth
# $\delta_E = \lambda_0 / (2\pi\kappa) = 28.77\;\text{nm}$, power skin depth
# $\delta_I = \lambda_0 / (4\pi\kappa) = 14.39\;\text{nm}$), the screen thickness
# $t_{\text{screen}} = 200\;\text{nm} = 6.95\,\delta_E = 13.90\,\delta_I$ provides
# $> 50\;\text{dB}$ direct attenuation through the central barrier while skin-depth
# penetration into the slit sidewalls increases peak downstream transmission by
# **$+5.75\%$ ($I_{\text{Ag}}(0) / I_{\text{PEC}}(0) = 1.0575$)** at the converged
# `--full` graded mesh level ($\Delta x_{\min} = 2.86\;\text{nm}$,
# $10.07\;\text{cells}/\delta_E$, $5.04\;\text{cells}/\delta_I$, background
# $\text{cpw}_{\text{bg}} = 20$, $\Delta x_{\text{bg}} = 24.92\;\text{nm}$) while
# preserving coherent fringe visibility ($\mathcal{V}_{\text{Ag}} = 0.9939$ vs
# $\mathcal{V}_{\text{PEC}} = 0.9935$). In `--quick` ($\text{cpw}_{\text{bg}} = 12$,
# $\Delta x_{\min} = 3.57\;\text{nm}$, $8.06\;\text{cells}/\delta_E$), the accepted
# ratio was $+5.01\%$ ($1.0501$); the $+0.74\%$ difference comes from refining the
# background propagation grid from $\text{cpw}_{\text{bg}} = 12$ to $20$.
#
# > [!NOTE]
# > **Why skin-depth resolution changed the Ag/PEC transmission ratio from $+10.2\%$ to $+5.75\%$:**
# > On an unrefined uniform $10\;\text{cpw}$ mesh ($\Delta x = 50.0\;\text{nm}$, only
# > $0.58\;\text{cells}/\delta_E = 0.29\;\text{cells}/\delta_I$), sub-cell staircase
# > smearing of the $28.77\;\text{nm}$ skin depth at each slit jaw inflated the
# > apparent transmission boost to $+10.17\%$ ($R = 1.10167$ at matched $\text{CFL} = 0.65$,
# > or $1.1053$ with mismatched $\text{CFL}$). Refining the graded collar around the
# > slit walls and screen faces at $\text{cpw}_{\text{bg}} = 20$ (`15_silver_skin_depth_resolution.png`)
# > reduces $R = I_{\text{Ag}}(0)/I_{\text{PEC}}(0)$ monotonically across all 7 levels
# > until the step-to-step change between $8.06$ and $10.07\;\text{cells}/\delta_E$
# > drops to **$0.0461\%$ ($0.05\% \ll 1\%$)** at $R = 1.05753$:
#
# | Mesh level (`--full`, $\text{cpw}_{\text{bg}}=20$) | $\Delta x_{\min}$ [$\text{nm}$] | Cells / $\delta_E$ ($28.77\;\text{nm}$) | Cells / $\delta_I$ ($14.39\;\text{nm}$) | Cells across $t_{\text{screen}}$ | $N_{\text{cells}}$ | $N_{\text{steps}}$ | $R = I_{\text{Ag}}(0)/I_{\text{PEC}}(0)$ | Step change $|\Delta R|/R$ |
# | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | Uniform $10\;\text{cpw}$ (`demo`) | $50.00$ | $0.58$ | $0.29$ | $4$ | $34\,081$ | $693$ | $1.10167$ ($+10.17\%$) | — |
# | Uniform $20\;\text{cpw}$ (`demo`) | $25.00$ | $1.15$ | $0.58$ | $8$ | $124\,033$ | $1\,386$ | $1.07618$ ($+7.62\%$) | $2.3136\%$ |
# | Graded collar ($2\;\text{cpsd}_E$, `demo`) | $14.29$ | $2.01$ | $1.01$ | $14$ | $135\,718$ | $2\,412$ | $1.06734$ ($+6.73\%$) | $0.8219\%$ |
# | Graded collar ($4\;\text{cpsd}_E$, `demo`) | $7.14$ | $4.03$ | $2.01$ | $28$ | $163\,682$ | $4\,821$ | $1.06091$ ($+6.09\%$) | $0.6024\%$ |
# | Graded collar ($6\;\text{cpsd}_E$, `demo`) | $4.76$ | $6.04$ | $3.02$ | $42$ | $194\,142$ | $7\,231$ | $1.05896$ ($+5.90\%$) | $0.1837\%$ |
# | Graded collar ($8\;\text{cpsd}_E$) | $3.57$ | $8.06$ | $4.03$ | $56$ | $224\,466$ | $9\,641$ | $1.05802$ ($+5.80\%$) | $0.0884\%$ |
# | **Accepted graded ($10\;\text{cpsd}_E$)** | **$2.86$** | **$10.07$** | **$5.04$** | **$70$ ($67$ slit)** | **$258\,408$** | **$12\,051$** | **$1.05753$ ($+5.75\%$)** | **$0.0461\% < 1\%$** |
#
# ### 8.4 Spatial mesh resolution convergence (`14_mesh_resolution_convergence.png`)
#
# Refining the background grid across $\text{cpw}_{\text{bg}} \in \{10, 14, 20\}$ against
# the fine reference $\text{cpw}_{\text{ref}} = 40$ ($\Delta x = 12.50\;\text{nm}$,
# $16$ cells across $t_{\text{screen}}$, $536\,769$ cells, $4\,202$ steps) confirms
# second-order $\mathcal{O}(\Delta x^2)$ convergence (empirical order $p = 2.38$ for
# $\mu_{12} = 0.5$ and $p = 2.39$ for $\mu_{12} = 1.0$). At the accepted `--full`
# resolution ($\text{cpw}_{\text{bg}} = 20$), the $L_2$ screen intensity profile error is
# $1.42\%$ for partially coherent illumination ($\mu_{12} = 0.5$) and $2.22\%$ for
# coherent illumination ($\mu_{12} = 1.0$), while the extracted fringe visibility
# $\mathcal{V}_{\text{pc}}$ changes by only $|\Delta\mathcal{V}_{\text{pc}}| = 0.00024$
# ($0.048\%$ relative to $\mathcal{V}_{\text{pc}}(40) = 0.50396$):
#
# | Resolution $\text{cpw}_{\text{bg}}$ | $\Delta x_{\text{bg}}$ [$\text{nm}$] | $\Delta x_{\min}$ [$\text{nm}$] | Cells across $t_{\text{screen}}$ | $N_{\text{cells}}$ | $N_{\text{steps}}$ | Partially coherent $L_2$ error ($\mu_{12}=0.5$) | Coherent $L_2$ error ($\mu_{12}=1.0$) | $\mathcal{V}_{\text{pc}}$ ($|\Delta\mathcal{V}_{\text{pc}}|$ vs ref) | 2-mode runtime [$\text{s}$] |
# | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | $10$ (`--quick`) | $50.00$ | $50.00$ | $4$ | $38\,409$ | $1\,051$ | $7.38\%$ | $11.64\%$ | $0.50015$ ($0.00381$) | $6.25$ |
# | $14$ | $35.71$ | $33.33$ | $6$ | $72\,600$ | $1\,575$ | $3.58\%$ | $5.74\%$ | $0.50405$ ($0.00009$) | $19.90$ |
# | $20$ (`--full` accepted) | $25.00$ | $25.00$ | $8$ | $140\,529$ | $2\,101$ | $1.42\%$ | $2.22\%$ | $0.50420$ ($0.00024$) | $51.35$ |
# | $40$ (`--full` fine ref) | $12.50$ | $12.50$ | $16$ | $536\,769$ | $4\,202$ | $0.00\%$ (ref) | $0.00\%$ (ref) | $0.50396$ (ref) | $386.94$ |
#
# ### 8.5 Summary comparison: `--quick` vs `--full` vs analytical/published reference
#
# All comparisons below use fixed physical geometry ($a = 0.40\;\mu\text{m}$,
# $d = 2.00\;\mu\text{m}$, $t_{\text{screen}} = 0.20\;\mu\text{m}$, $L_1 = 8.00\;\mu\text{m}$,
# $L_2 = 6.00\;\mu\text{m}$, $\lambda_0 = 0.50\;\mu\text{m}$) without parameter tuning:
#
# | Benchmark metric | `--quick` (10–13 cpw) | `--full` (20–26 cpw) | Analytic / published reference | Reference source |
# | :--- | :---: | :---: | :---: | :--- |
# | VCZ $\mu_{12}$ at $D = 0.5\,D_0 = 1.0\;\mu\text{m}$ | $+0.6774$ | $+0.6409$ | $+0.6424$ (Hankel) / $+0.6366$ ($\operatorname{sinc}$) | Zernike (1938); Born & Wolf §10.4.2 |
# | VCZ $\mu_{12}$ at $D = 0.6\,D_0 = 1.2\;\mu\text{m}$ | $+0.5278$ | $+0.5145$ | $+0.5124$ (Hankel) / $+0.5046$ ($\operatorname{sinc}$) | Zernike (1938); Born & Wolf §10.4.2 |
# | VCZ null visibility $\mathcal{V}(D_0 = 2.0\;\mu\text{m})$ | $0.0593$ | $0.0269$ | $0.0123$ (Hankel) / $0.0000$ ($\operatorname{sinc}$) | van Cittert (1934); Zernike (1938) |
# | VCZ secondary lobe $\mu_{12}(1.4\,D_0 = 2.8\;\mu\text{m})$ | $-0.1839$ | $-0.1994$ | $-0.2179$ (Hankel) / $-0.2162$ ($\operatorname{sinc}$) | Born & Wolf §10.4.2, Fig. 10.11 |
# | VCZ secondary lobe $\mu_{12}(1.5\,D_0 = 3.0\;\mu\text{m})$ | $-0.2055$ | $-0.2029$ | $-0.2206$ (Hankel) / $-0.2122$ ($\operatorname{sinc}$) | Born & Wolf §10.4.2, Fig. 10.11 |
# | GSM visibility $\mathcal{V}(\sigma_\mu / d = 0.50)$ | $0.1448$ | $0.1454$ | $0.1353$ ($\exp(-2)$) / $0.1350$ ($M=12$) | Starikov & Wolf (1982); Mandel & Wolf §5.6.4 |
# | GSM visibility $\mathcal{V}(\sigma_\mu / d = 0.75)$ | $0.4217$ | $0.4199$ | $0.4111$ ($\exp(-8/9)$) | Starikov & Wolf (1982); Mandel & Wolf §5.6.4 |
# | GSM visibility $\mathcal{V}(\sigma_\mu / d = 1.00)$ | $0.6152$ | $0.6117$ | $0.6065$ ($\exp(-1/2)$) | Starikov & Wolf (1982); Mandel & Wolf §5.6.4 |
# | GSM visibility $\mathcal{V}(\sigma_\mu / d = 1.50)$ | $0.8069$ | $0.8006$ | $0.8007$ ($\exp(-2/9)$) | Starikov & Wolf (1982); Mandel & Wolf §5.6.4 |
# | GSM visibility $\mathcal{V}(\sigma_\mu / d = 2.00)$ | $0.8875$ | $0.8797$ | $0.8825$ ($\exp(-1/8)$) | Starikov & Wolf (1982); Mandel & Wolf §5.6.4 |
# | Temporal contrast at $\sigma_f/f_0 = 8\%$, $m=1$ / $m=2$ | $0.8800$ / $0.5908$ | $0.8684$ / $0.5767$ | $0.8813$ / $0.6033$ | Wiener–Khinchin; Goodman (2015) §5.2 |
# | Temporal contrast at $\sigma_f/f_0 = 12\%$, $m=1$ / $m=2$ | $0.7865$ / $0.3462$ | $0.7688$ / $0.3289$ | $0.7526$ / $0.3208$ | Wiener–Khinchin; Goodman (2015) §5.2 |
# | Drude Ag vs PEC on-axis ratio $I_{\text{Ag}}(0)/I_{\text{PEC}}(0)$ | $1.0501$ ($+5.01\%$) | $1.0575$ ($+5.75\%$) | $\delta_E = 28.77\;\text{nm}$ ($0.046\%$ last step change) | Rakić *et al.* (1998) at $\lambda_0 = 500\;\text{nm}$ |
# | Partially coherent $L_2$ mesh error ($\mu_{12}=0.5$) | $1.23\%$ ($\text{cpw}=18$ vs $24$) | $1.42\%$ ($\text{cpw}=20$ vs $40$) | $\mathcal{O}(\Delta x^2)$ ($p = 2.38$, $\|\Delta\mathcal{V}_{\text{pc}}\| = 2.4\times 10^{-4}$) | 2nd-order FIT spatial convergence |

# %% [markdown]
# ## 9. Limits & failure modes
#
# 1. **Coherent amplitude addition vs incoherent power summation**:
#    Driving all $N_s$ emitters simultaneously in a single FDTD/FIT run computes
#    $\big|\sum_n E_z^{(n)}\big|^2$ instead of $\sum_n \big|E_z^{(n)}\big|^2$. As shown
#    in `04_coherent_vs_incoherent_2d_intensity.png` and verified in our mutation test
#    `test_mutation_coherent_field_sum_and_corrupted_weights_fail`, coherent addition
#    at $D = D_0$ yields $\mathcal{V} \approx 1.0$ instead of $\mathcal{V} = 0.0$.
# 2. **Non-paraxial breakdown of the $\operatorname{sinc}$ formula**:
#    The classical $\operatorname{sinc}(\pi D d / (\lambda_0 L_1))$ law relies on the
#    Fresnel paraxial expansion $R_{1,2}(x_s) \approx L_1 + (x_s \pm d/2)^2 / (2 L_1)$.
#    When $D / L_1 \gtrsim 0.4$ or $d / L_1 \gtrsim 0.4$, higher-order phase terms and
#    $R^{-1/2}$ cylindrical amplitude obliquity shift the coherence zero slightly;
#    in this regime the exact 2D Hankel integral `van_cittert_zernike_2d_hankel` must
#    be used as the reference benchmark.
# 3. **Mercer mode truncation in the deeply incoherent limit ($\sigma_\mu \ll w_m$)**:
#    The Starikov & Wolf eigenvalue ratio $q = (\sqrt{1 + \gamma^2} - 1)/\gamma$
#    approaches $1 - 1/\gamma$ as $\gamma = (w_m/\sigma_\mu)^2 \to \infty$. Achieving
#    truncation error $q^M < \epsilon$ requires $M \gtrsim (\sigma_s / \sigma_\mu)\ln(1/\epsilon)$
#    modes. When $\sigma_\mu \ll d$, direct point-dipole quadrature (Method A) becomes
#    more economical than Mercer mode summation.
# 4. **Multi-pole UV Lorentz instability in coarse-grid dispersive metals**:
#    Full 6-pole Rakic Lorentz–Drude metal tables include ultraviolet interband
#    resonances ($\hbar\omega_5 = 20.29\;\text{eV}$, $\omega_5 \approx 3.08\times 10^{16}\;\text{rad/s}$)
#    that violate the explicit leapfrog stability limit $\omega_p \Delta t < 2$ at
#    coarse optical grids ($\text{cpw} = 10 \implies \Delta t \approx 1.08\times 10^{-16}\;\text{s}$,
#    $\omega_5 \Delta t \approx 3.3 > 2$). Fitting a band-matched single-pole Drude
#    model (`silver_drude_material`) or refining the mesh ($\text{cpw} \ge 30$) avoids
#    UV pole blow-up while reproducing the exact optical skin depth $\delta_{\text{skin}} = 28.77\;\text{nm}$.

# %% [markdown]
# ## 10. Exercises
#
# 1. **Asymmetric slit widths and unequal intensities ($I_1 \neq I_2$)**:
#    Modify `build_double_slit_solver` so slit 1 has width $a_1 = 0.30\;\mu\text{m}$
#    and slit 2 has width $a_2 = 0.50\;\mu\text{m}$. Verify that even for
#    spatially coherent illumination ($|\mu_{12}| = 1$), the on-axis fringe visibility
#    is bounded by $\mathcal{V}_{\max} = 2\sqrt{I_1 I_2}/(I_1 + I_2) < 1$
#    (Mandel & Wolf 1995, Eq. 4.3-29).
# 2. **Second van Cittert–Zernike null ($D = 2 D_0 = 4.00\;\mu\text{m}$)**:
#    Extend `xs_positive` in `get_task_table` out to $x_s = 2.20\;\mu\text{m}$
#    ($D_{\max} = 4.40\;\mu\text{m}$) and verify that $\mu_{12}(D)$ crosses zero a
#    second time at $D = 2 D_0 = 4.00\;\mu\text{m}$ before returning to positive
#    phase ($\mu_{12} > 0$, bright central peak at $x=0$) in the third lobe.
# 3. **Sub-skin-depth silver film leakage ($t_{\text{screen}} = 40\;\text{nm}$)**:
#    Reduce `SCREEN_THICK` in `run_silver_vs_pec_study` from $200\;\text{nm}$
#    ($6.95\,\delta_{\text{skin}}$) to $40\;\text{nm}$ ($1.39\,\delta_{\text{skin}}$).
#    Quantify how direct evanescent tunneling through the central silver barrier
#    introduces a third coherent background wave that alters the downstream fringe contrast.

# %% [markdown]
# ## 11. References
#
# 1. P. H. van Cittert, "Die wahrscheinliche Schwingungsverteilung in einer von
#    einer Lichtquelle direkt oder mittels einer Linse beleuchteten Ebene,"
#    *Physica* **1**(1–6), 201–210 (1934),
#    DOI: [10.1016/S0031-8914(34)90026-4](https://doi.org/10.1016/S0031-8914(34)90026-4).
# 2. F. Zernike, "The concept of degree of coherence and its application to
#    optical problems," *Physica* **5**(8), 785–795 (1938),
#    DOI: [10.1016/S0031-8914(38)80203-2](https://doi.org/10.1016/S0031-8914(38)80203-2).
# 3. E. Wolf, "New theory of partial coherence in the space–frequency domain.
#    Part I: spectra and cross-spectra of steady-state sources,"
#    *J. Opt. Soc. Am.* **72**(3), 343–351 (1982),
#    DOI: [10.1364/JOSA.72.000343](https://doi.org/10.1364/JOSA.72.000343).
# 4. A. Starikov and E. Wolf, "Coherent-mode representation of Gaussian
#    Schell-model sources and of their radiation fields,"
#    *J. Opt. Soc. Am.* **72**(7), 923–928 (1982),
#    DOI: [10.1364/JOSA.72.000923](https://doi.org/10.1364/JOSA.72.000923).
# 5. L. Mandel and E. Wolf, *Optical Coherence and Quantum Optics* (Cambridge
#    University Press, 1995), Chapters 4 and 5,
#    DOI: [10.1017/CBO9781139644105](https://doi.org/10.1017/CBO9781139644105).
# 6. J. W. Goodman, *Statistical Optics*, 2nd ed. (Wiley, 2015), Chapter 5.
# 7. M. Born and E. Wolf, *Principles of Optics*, 7th (expanded) ed. (Cambridge
#    University Press, 2019), §10.4,
#    DOI: [10.1017/9781108769914](https://doi.org/10.1017/9781108769914).
# 8. A. D. Rakić, A. B. Djurišić, J. M. Elazar, and M. L. Majewski, "Optical
#    properties of metallic films for vertical-cavity optoelectronic devices,"
#    *Appl. Opt.* **37**(22), 5271–5283 (1998),
#    DOI: [10.1364/AO.37.005271](https://doi.org/10.1364/AO.37.005271).


# %%
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="06. Young's Double Slit with Partially Coherent Light (2D Wakis FIT)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run resolved 2D study locally (cpw>=10, cpsd_E=8) and render all 15 figures",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run converged 2D study across all tasks (cpw>=20, ref=40) and render all 15 figures",
    )
    ap.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single study point I (0-indexed); uses --quick if passed, else --full",
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
    import resource

    t_start = time.perf_counter()
    quick = not args.full
    tasks = get_task_table(quick=quick)
    for t in tasks:
        run_single_task(t, DATA_DIR)
    generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR)
    total_s = time.perf_counter() - t_start
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[summary] mode={'quick' if quick else 'full'} total_time={total_s:.2f}s peak_rss={peak_mb:.1f}MB")


if __name__ == "__main__":
    main()
