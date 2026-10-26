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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eshipton/Wakis/blob/main/tutorials/04_radiation_and_antennas/04-08_magnetic_antenna_ground/magnetic_antenna_ground.ipynb)
#
# # Tutorial 4.8 · Magnetic antennas on and near a ground plane
# <h3 align="center">Magnetic Lorentz/Debye ADE dispersion, hesitivity invariance, four-orientation PEC image theory, volume-convention resolution, PEC/PMC duality, and conformal efficiency–bandwidth product in 3D FIT</h3>
#
# ---
#
# ### Abstract
#
# **Magnetic antennas—radiators whose primary far field originates from an impressed or induced magnetic current density $\mathbf{M} = j\omega\mu_0(\mu_r - 1)\mathbf{H}$ inside a permeable core or from a small current loop—behave in ways that often contradict standard electric-antenna intuition when placed near a metallic ground plane or evaluated with lossy dispersive ferrites and ferromagnetic laminates. In this tutorial we build and validate a complete 3D Finite Integration Technique (`SolverFIT3D`) workflow for magnetic antennas on and near a Perfect Electric Conductor (PEC) ground plane. Starting from the time-domain magnetic Lorentz and Debye Auxiliary Differential Equations (`wakis.dispersive`), we verify 1D normal-incidence plane-wave reflection, transmission, and volumetric magnetic absorption to $< 0.11\%$, derive the Heaviside magnetic conductivity $\sigma_m(\omega) = \omega\mu_0\mu''(\omega)$ and its peak/asymptotic invariant—the *hesitivity* $h_m\;\text{[}\Omega/\text{m]}$ [1, 2]—and prove the exact algebraic identity $f(\mu) \equiv \mu''/|\mu_r - 1|^2 = \omega\mu_0/h_m$ across all frequencies without low-loss approximations. Using a unified permeable rod geometry ($\ell = 72\;\text{mm}, w = 24\;\text{mm}, t = 24\;\text{mm}$ on PEC, $V_{\text{phys}} = 4.15\times 10^{-5}\;\text{m}^3$) resolved on a graded mesh ($\Delta_{\text{fine}} = 4.0\;\text{mm}$, $6$ cells across the rod cross-section, $42.3\;\text{cells}/\lambda_{\text{mat}}$, $3.65\%\;\text{below the Richardson extrapolate}$), we verify discrete leapfrog Poynting energy conservation ($W_{\text{src}} = P_{\text{rad,flux}} + P_{\text{loss,mag}}$ to $0.00059\%$) and demonstrate in 3D FIT that two distinct Lorentz poles with equal $h_m$ yield the same radiation efficiency within $0.11\;\text{dB}$ while halving $h_m$ doubles the loss-to-radiation ratio $(1/e - 1)$ by a factor of $1.9993$. Next, we derive and benchmark the exact spherical-Bessel half-space radiation resistance ratios $R_{\text{rad}}(h)/R_{\text{free}}$ for all four elementary dipole orientations over PEC (HMD, VMD, VED, HED) [3, 4], resolve the literature ambiguity in the closed-form efficiency $e_{\text{MD}} = [1 + 6\pi\eta_0/(h_m\,\text{Vol}\,k_0^2)]^{-1}$ between physical half-rod volume $V_{\text{phys}}$ and full image-mirrored volume $V_{\text{full}} = 2 V_{\text{phys}}$, collapse 4 Lorentz and 3 Debye pole families onto universal equal-$h_m$ efficiency curves (showing why the legacy Rumsey–Weeks formula [5, 6] fails at high loss and how Cauchy–Schwarz volume filling $\eta_{\text{fill}} = 0.8129 \le 1$ and the driving loop's vacuum moment satisfy $e_{\text{pol}}^{\text{DFT}} < e_{\text{tot}}^{\text{FIT}} < e_{\text{MD}}^{\text{bound}}$ across the $200\text{–}320\;\text{MHz}$ passband), verify continuous Maxwell PEC$\leftrightarrow$PMC duality between a horizontal permeable rod on PEC and an identical horizontal dielectric rod on a dual-shifted PMC grid (discrete dual up to outer-grid grading $\mathcal{O}((g-1)^2\Delta)$, shrinking from $0.171\;\text{dB}$ at $g = 1.35$ to $0.038\;\text{dB}$ at $g = 1.18$ across $\sim 1.2$ decades of susceptibility) alongside Sebastian's orientational PEC pair [2], and compare a conformal permeable bar against a conformal metal strip in the same volume [1, 7–10, 12].**
#
# ### Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. **Derive and verify** the causal **Lorentz** and **Debye** magnetic susceptibility models in `wakis.dispersive`, the **hesitivity** invariant $h_m = \omega_0\mu_0\chi_{\text{DC}}/\alpha$ (Lorentz) and $h_m = \omega_0\mu_0\chi_{\text{DC}}$ (Debye) [1, 2], the exact frequency-domain loss identity $f(\mu) \equiv \mu''/|\mu_r - 1|^2 = \omega\mu_0/h_m$, and exact leapfrog time-centered energy balance $W_{\text{src}} = P_{\text{rad,flux}} + P_{\text{loss,mag}}$ in 3D FIT.
# 2. **Formulate** PEC image theory from the boundary conditions $\hat{\mathbf{n}}\times\mathbf{E} = \mathbf{0}$ and $\hat{\mathbf{n}}\cdot\mathbf{B} = 0$, explain why only a **horizontal magnetic dipole (HMD, $\mathbf{M} \parallel \text{PEC}$)** or **vertical electric dipole (VED, $\mathbf{J} \perp \text{PEC}$)** experiences constructive image doubling ($R_{\text{rad}}(h)/R_{\text{free}} \to 2$ as $h/\lambda_0 \to 0$) whereas a **vertical magnetic dipole (VMD, $\mathbf{M} \perp \text{PEC}$)** or **horizontal electric dipole (HED, $\mathbf{J} \parallel \text{PEC}$)** is shorted out ($R_{\text{rad}}(h)/R_{\text{free}} \to 0$) [3, 4], and distinguish fixed-moment radiation intensity quadrupling ($4\times = +6.02\;\text{dB}$) and radiated power / resistance doubling ($2\times = +3.01\;\text{dB}$) from directivity doubling ($D = 2 D_{\text{free}} = 3.0$, $+3.01\;\text{dB}$ over the free-space dipole, or $4.77\;\text{dBi}$).
# 3. **Resolve** the free-space versus half-space volume convention in the Yousefi–Diaz efficiency law $e_{\text{MD}} = [1 + 6\pi\eta_0/(h_m\,\text{Vol}\,k_0^2)]^{-1}$ (where $6\pi\eta_0 = 7101.20\;\Omega$ is the exact SI impedance constant and $720\pi^2 = 7106.12\;\Omega \approx 7106\;\Omega$ is the $\eta_0 \approx 120\pi\;\Omega$ form of [1]), proving by symmetry and 3D FIT that a half-rod of physical volume $V_{\text{phys}}$ flush on PEC is governed by $\text{Vol} = 2 V_{\text{phys}} = V_{\text{full}}$ and transitions smoothly to $\text{Vol} = V_{\text{phys}}$ as it is lifted to height $h$ via $\text{Vol}_{\text{eff}}(h) = F_{\text{HMD}}(k_0 h)\,V_{\text{phys}}$.
# 4. **Demonstrate** in 3D FIT why the conventional magnetic loss tangent $\tan\delta_m = \mu''/\mu'$ does not govern radiation efficiency in electrically small flux-channeled permeable antennas at equal hesitivity $h_m$, why the classical Rumsey–Weeks / DeVore–Bohley approximation $f_{\text{RW}}(\mu) \approx \mu''/(\mu')^2$ [5, 6] breaks down near ferromagnetic resonance, and how the Cauchy–Schwarz volume-filling factor $\eta_{\text{fill}} \le 1$ bounds the core polarization efficiency $e_{\text{pol}}^{\text{DFT}}$ (an algebraic identity check with `closed_form_md_efficiency(Vol·η_fill)`) while the total rod+loop run ($59.6\times$ larger radiated power than the bare vacuum loop, so the rod-only moment ratio is $\sqrt{59.6} - 1 = 6.72$) sits $+0.31\;\text{dB}$ above $e_{\text{pol}}^{\text{DFT}}$ at $280\;\text{MHz}$, satisfying $e_{\text{pol}}^{\text{DFT}} < e_{\text{tot}}^{\text{FIT}} < e_{\text{MD}}^{\text{bound}}$ across $200\text{–}320\;\text{MHz}$.
# 5. **Verify** continuous Maxwell PEC$\leftrightarrow$PMC duality ($\mathbf{E} \leftrightarrow \eta_0\mathbf{H}$, $\varepsilon_r \leftrightarrow \mu_r$, $\text{PEC} \leftrightarrow \text{PMC}$) between a horizontal permeable rod on PEC and an identical horizontal dielectric rod on a dual-shifted PMC grid (showing that the discrete difference is governed by outer-grid grading $\mathcal{O}((g-1)^2\Delta)$ and shrinks to $\le 0.038\;\text{dB}$ at $g = 1.18$), explain how it differs from Sebastian's orientational PEC pair (a vertical dielectric monopole on PEC sharing the $+1$ image sign) [2], and quantify the $> 10^2\times$ radiation resistance (3D FIT) and $19\text{–}43\;\text{dB}$ Efficiency–Bandwidth Product ($\text{EBWP}$, analytical lumped/transmission-line model) advantage of a conformal permeable bar over a conformal metal strip in the same volume [1, 7–9, 12], contrasted with the resonant-cavity patch limit of Ikonen et al. [10].
#
# ### Contents
#
# 1. [Theory & First-Principles Derivations](#1.-Theory-&-First-Principles-Derivations)
# 2. [Simulation Setup, Mesh Criteria & Preflight](#2.-Simulation-Setup,-Mesh-Criteria-&-Preflight)
# 3. [Results: Parts 0 through 6 in 3D FIT](#3.-Results:-Parts-0-through-6-in-3D-FIT)
# 4. [Quantitative Summary](#4.-Quantitative-Summary)
# 5. [Limits & Failure Modes](#5.-Limits-&-Failure-Modes)
# 6. [References](#6.-References)
#
# ---
#
# ## 1. Theory & First-Principles Derivations
#
# ### 1.1 Causal magnetic dispersion, Heaviside magnetic conductivity, and hesitivity $h_m$
#
# In the $\exp(+j\omega t)$ harmonic convention used throughout `wakis.dispersive`, a passive isotropic magnetic medium has complex relative permeability
#
# $$\mu_r(\omega) = 1 + \chi(\omega) = \mu'(\omega) - j\mu''(\omega), \qquad \mu''(\omega) = -\operatorname{Im}[\mu_r(\omega)] \ge 0.$$
#
# Two canonical causal susceptibility laws model practical permeable antenna cores across VHF/UHF:
#
# 1. **Single-pole Debye relaxation** (domain-wall relaxation in a commercially available NiZn ferrite [1]):
#    $$\chi_{\text{Debye}}(\omega) = \frac{\chi_{\text{DC}}}{1 + j(\omega/\omega_0)}, \qquad \omega_0 = \frac{1}{\tau}, \qquad \chi_{\text{DC}} = \mu_{\text{DC}} - 1.$$
# 2. **Single-pole Lorentz / Landau–Lifshitz–Gilbert resonance** (gyromagnetic spin resonance in a ferromagnetic CoZrNb laminate with static susceptibility $\chi_{\text{DC}}$, resonance frequency $\omega_0 = 2\pi f_0$, and dimensionless damping factor $\alpha = \gamma/\omega_0$ [1, 2]):
#    $$\chi_{\text{Lor}}(\omega) = \frac{\chi_{\text{DC}}}{1 + j\alpha(\omega/\omega_0) - (\omega/\omega_0)^2}.$$
#
# Following Heaviside's symmetrization of Maxwell's curl equations, the volumetric time-averaged power dissipated per unit volume by a harmonic magnetic field $\mathbf{H}(\mathbf{r}, \omega)$ is
#
# $$p_{\text{loss,mag}}(\mathbf{r}, \omega) = \frac{1}{2}\,\omega\mu_0\mu''(\omega)\,|\mathbf{H}(\mathbf{r}, \omega)|^2 = \frac{1}{2}\,\sigma_m(\omega)\,|\mathbf{H}(\mathbf{r}, \omega)|^2, \qquad \sigma_m(\omega) \equiv \omega\mu_0\mu''(\omega)\;\;[\Omega/\text{m}].$$
#
# **Derivation of the hesitivity invariant $h_m$ (derived here following [1, 2]):**
#
# * For a **Lorentz pole**, substituting $\mu''(\omega) = \chi_{\text{DC}}\,\alpha(\omega/\omega_0) \big/ \big[(1 - (\omega/\omega_0)^2)^2 + \alpha^2(\omega/\omega_0)^2\big]$ into $\sigma_m(\omega) = \omega\mu_0\mu''(\omega)$ gives
#   $$\sigma_m^{\text{Lor}}(\omega) = \frac{\omega_0\mu_0\chi_{\text{DC}}}{\alpha}\,\frac{\alpha^2(\omega/\omega_0)^2}{\left[1 - (\omega/\omega_0)^2\right]^2 + \alpha^2(\omega/\omega_0)^2}.$$
#   Because $\big[1 - (\omega/\omega_0)^2\big]^2 \ge 0$ with equality if and only if $\omega = \omega_0$, $\sigma_m^{\text{Lor}}(\omega)$ attains its unique global maximum at $\omega = \omega_0$. Defining this peak magnetic conductivity as the **hesitivity** $h_m$:
#   $$h_m^{\text{Lor}} \equiv \max_{\omega} \sigma_m^{\text{Lor}}(\omega) = \sigma_m^{\text{Lor}}(\omega_0) = \frac{\omega_0 \mu_0 \chi_{\text{DC}}}{\alpha} = \frac{2\pi f_0 \mu_0 (\mu_{\text{DC}} - 1)}{\alpha}\;\;[\Omega/\text{m}].$$
#   For the reference CoZrNb (CZN) ferromagnetic laminate ($\mu_{\text{DC}} = 40 \Rightarrow \chi_{\text{DC}} = 39$, $f_0 = 1.5\;\text{GHz}$, $1/\alpha = 0.75 \Rightarrow \alpha = 4/3$ [1, 2]), this yields
#   $$h_m^{\text{CZN}} = 2\pi (1.5\times 10^9\;\text{Hz})(4\pi\times 10^{-7}\;\text{H/m})(39)(0.75) = 3.464231\times 10^5\;\Omega/\text{m}.$$
# * For a **Debye pole**, $\sigma_m^{\text{Debye}}(\omega) = \omega_0\mu_0\chi_{\text{DC}}\,(\omega/\omega_0)^2 / [1 + (\omega/\omega_0)^2]$ increases monotonically to its high-frequency supremum $\lim_{\omega\to\infty}\sigma_m^{\text{Debye}}(\omega) = 2\,\sigma_m^{\text{Debye}}(\omega_0)$, which defines the Debye hesitivity (Snoek's product $\chi_{\text{DC}} f_0$ scaled by $2\pi\mu_0$ [1, 11]):
#   $$h_m^{\text{Debye}} \equiv \sup_{\omega} \sigma_m^{\text{Debye}}(\omega) = 2\,\sigma_m^{\text{Debye}}(\omega_0) = \omega_0 \mu_0 \chi_{\text{DC}} = \frac{\mu_0 \chi_{\text{DC}}}{\tau}\;\;[\Omega/\text{m}].$$
#
# **Exact algebraic identity $f(\mu) = \omega\mu_0 / h_m$ at all frequencies (derived here):**
#
# By the volume equivalence principle, a permeable body of volume $\text{Vol}$ immersed in an internal magnetic field $\mathbf{H}$ carries an equivalent polarization magnetic current density $\mathbf{M}_p = j\omega\mu_0(\mu_r - 1)\mathbf{H} = j\omega\mu_0\chi(\omega)\mathbf{H}$. Expressing $\mathbf{H} = \mathbf{M}_p / [j\omega\mu_0(\mu_r - 1)]$, the volumetric material dissipation for a given induced magnetic moment is governed by the dimensionless **magnetic loss function** $f(\mu)$:
#
# $$p_{\text{loss,mag}} = \frac{1}{2}\omega\mu_0\mu''|\mathbf{H}|^2 = \frac{|\mathbf{M}_p|^2}{2\omega\mu_0}\,\frac{\mu''(\omega)}{|\mu_r(\omega) - 1|^2} \equiv \frac{|\mathbf{M}_p|^2}{2\omega\mu_0}\,f(\mu), \qquad f(\mu) \equiv \frac{\mu''(\omega)}{|\mu_r(\omega) - 1|^2} = \frac{-\operatorname{Im}[\chi(\omega)]}{|\chi(\omega)|^2} = -\operatorname{Im}\!\left[\frac{1}{\chi(\omega)}\right].$$
#
# Inverting $\chi_{\text{Lor}}(\omega)$ and $\chi_{\text{Debye}}(\omega)$ directly reveals why $1/\chi(\omega)$ is linear in $\operatorname{Im}[1/\chi]$:
#
# $$\frac{1}{\chi_{\text{Lor}}(\omega)} = \frac{1 - (\omega/\omega_0)^2}{\chi_{\text{DC}}} + j\,\frac{\alpha\omega}{\omega_0\chi_{\text{DC}}} \quad\Longrightarrow\quad f(\mu_{\text{Lor}}) = -\operatorname{Im}\!\left[\frac{1}{\chi_{\text{Lor}}(\omega)}\right] = \frac{\alpha\omega}{\omega_0\chi_{\text{DC}}} = \frac{\omega\mu_0}{h_m^{\text{Lor}}},$$
#
# $$\frac{1}{\chi_{\text{Debye}}(\omega)} = \frac{1}{\chi_{\text{DC}}} + j\,\frac{\omega}{\omega_0\chi_{\text{DC}}} \quad\Longrightarrow\quad f(\mu_{\text{Debye}}) = -\operatorname{Im}\!\left[\frac{1}{\chi_{\text{Debye}}(\omega)}\right] = \frac{\omega}{\omega_0\chi_{\text{DC}}} = \frac{\omega\mu_0}{h_m^{\text{Debye}}}.$$
#
# This identity is **exact at every frequency $\omega$** (requiring no $\omega \ll \omega_0$ or $\mu'' \ll \mu'$ approximation): in both single-pole families, the real dispersive terms $1 - (\omega/\omega_0)^2$ and $1$ reside purely in $\operatorname{Re}[1/\chi(\omega)]$ and drop out of $-\operatorname{Im}[1/\chi(\omega)]$ identically. By contrast, the legacy Rumsey–Weeks / DeVore–Bohley formula [5, 6] approximates $|\mu_r - 1|^2 = (\mu' - 1)^2 + (\mu'')^2 \approx (\mu')^2$, replacing $f(\mu)$ with $f_{\text{RW}}(\mu) = \mu''/(\mu')^2 = \tan\delta_m / \mu'$, which diverges sharply from the exact loss whenever $\mu''$ is comparable to $\mu' - 1$ or near resonance where $\mu' \to 1$.
#
# **Discrete leapfrog Poynting theorem and ADE loss bookkeeping (derived here):**
#
# On the staggered FIT Yee grid with time step $\Delta t$, electric fields $\mathbf{E}^n$ live at integer steps $t_n = n\Delta t$, while magnetic fields $\mathbf{H}^{n+1/2}$ and impressed electric currents $\mathbf{J}^{n+1/2}$ live at half-integer steps $t_{n+1/2} = (n + 1/2)\Delta t$. In `SolverFIT3D`, the magnetic ADE update advances $\mathbf{H}$ from $n - 1/2$ to $n + 1/2$ across center time $t_n$ so that the discrete Faraday equation at frequency $\omega$ satisfies
#
# $$\nabla_h \times \widehat{\mathbf{E}}(\omega) = -j\,\tilde{\omega}\,\mu_0\,\mu_d(\omega, \Delta t)\,\widehat{\mathbf{H}}_{\text{mid}}(\omega), \qquad \tilde{\omega} \equiv \frac{2}{\Delta t}\sin\!\left(\frac{\omega\Delta t}{2}\right),$$
#
# where $\mu_d(\omega, \Delta t)$ is the exact discrete permeability returned by `material.mu_discrete(f, dt)`, $\widehat{\mathbf{H}}_{\text{mid}}(\omega) = \sum_n \frac{1}{2}(\mathbf{H}^{n-1/2} + \mathbf{H}^{n+1/2}) e^{-j\omega n\Delta t}\Delta t$ is time-centered at $t_n$, and the source input work spectral density $W_{\text{src}}(\omega) = -\operatorname{Re}\sum_{\text{edges}} \widehat{\mathbf{E}}_{\text{mid}}(\omega)\,\widehat{\mathbf{J}}^*(\omega)\,\Delta V$ uses $\widehat{\mathbf{E}}_{\text{mid}}(\omega)$ time-centered at $t_{n+1/2}$ via $\frac{1}{2}(\mathbf{E}^n + \mathbf{E}^{n+1})$. Taking the inner product of the discrete curl equations yields the exact discrete energy conservation law
#
# $$W_{\text{src}}(\omega) = P_{\text{rad,flux}}(\omega) + P_{\text{loss,mag}}(\omega), \qquad P_{\text{loss,mag}}(\omega) = \tilde{\omega}\,\mu_0\,(-\operatorname{Im}[\mu_d(\omega, \Delta t)]) \sum_{e \in \text{rod}} w_e\,|\widehat{H}_{\text{mid},e}(\omega)|^2\,\Delta V_e.$$
#
# Omitting the half-step time centering (sampling $\mathbf{E}^{n+1}$ at $t_n$ or $\mathbf{H}^{n+1/2}$ at $t_{n+1/2}$) introduces a spurious phase rotation $\exp(\pm j\omega\Delta t/2)$ that mixes reactive stored energy into real power and breaks energy conservation by $> 10\%$.
#
# ---
#
# ### 1.2 Image theory over a PEC ground plane: four dipole orientations (derived here)
#
# A Perfect Electric Conductor (PEC) occupying the plane $z = 0$ enforces $\hat{\mathbf{z}} \times \mathbf{E} = \mathbf{0}$ and $\hat{\mathbf{z}} \cdot \mathbf{B} = 0$ at $z = 0^+$. By the method of images, an elementary source at height $\mathbf{r}_0 = (0, 0, h)$ in the upper half-space $z > 0$ produces the exact same field in $z > 0$ as the source together with its image at $\mathbf{r}_{\text{img}} = (0, 0, -h)$ in unbounded free space:
#
# * **Electric current element $\mathbf{J} = (J_x, J_y, J_z)$:** Reversing the tangential electric field at $z = 0$ while doubling the normal electric field requires
#   $$\mathbf{J}_{\text{img}} = (-J_x, -J_y, +J_z) \quad\Longrightarrow\quad s_{\text{HED}} = -1\;\;(\text{antiphase}), \qquad s_{\text{VED}} = +1\;\;(\text{in phase}).$$
# * **Magnetic current element $\mathbf{M} = (M_x, M_y, M_z)$ (or small current loop of moment $\mathbf{m} = \mathbf{M}/(j\omega\mu_0)$):** Because $\mathbf{E} \propto -\nabla \times \mathbf{F}$ for a magnetic vector potential $\mathbf{F} \propto \mathbf{M}$, making $\hat{\mathbf{z}} \times \mathbf{E} = \mathbf{0}$ and $H_z = 0$ on $z = 0$ requires the **opposite** reflection rule:
#   $$\mathbf{M}_{\text{img}} = (+M_x, +M_y, -M_z) \quad\Longrightarrow\quad s_{\text{HMD}} = +1\;\;(\text{in phase}), \qquad s_{\text{VMD}} = -1\;\;(\text{antiphase}).$$
#
# **Derivation of the half-space radiation resistance ratios $R_{\text{rad}}(h)/R_{\text{free}}$ (derived here):**
#
# For a source of fixed current amplitude at height $h$ with image sign $s \in \{+1, -1\}$, interference between the source at $+h$ and its image at $-h$ multiplies the free-space far-field amplitude in direction $(\theta, \phi)$ by $e^{+jk_0 h\cos\theta} + s\,e^{-jk_0 h\cos\theta}$, whose squared magnitude is $2\big[1 + s\cos(2k_0 h\cos\theta)\big]$. Integrating the radiation intensity over the physical upper hemisphere $\theta \in [0, \pi/2], \phi \in [0, 2\pi)$ and normalizing by the full-sphere free-space radiated power $P_{\text{free}}$ yields, with $\xi \equiv 2 k_0 h = 4\pi(h/\lambda_0)$, $j_0(\xi) = \frac{\sin\xi}{\xi}$, and $\frac{j_1(\xi)}{\xi} = \frac{\sin\xi - \xi\cos\xi}{\xi^3}$:
#
# 1. **Vertical dipoles ($\mathbf{M} \perp \text{PEC}$ or $\mathbf{J} \perp \text{PEC}$, pattern $\propto \sin^2\theta = 1 - u^2$ with $u = \cos\theta \in [0, 1]$):**
#    $$\frac{R_{\text{rad}}^{\perp}(h)}{R_{\text{free}}} = \frac{3}{2}\int_0^1 (1 - u^2)\big[1 + s\cos(\xi u)\big]\,du = 1 + 3s\,\frac{\sin\xi - \xi\cos\xi}{\xi^3} = 1 + 3s\,\frac{j_1(\xi)}{\xi}.$$
#    Using the Taylor series $\frac{j_1(\xi)}{\xi} = \frac{1}{3} - \frac{\xi^2}{30} + \mathcal{O}(\xi^4)$ as $\xi = 2k_0 h \to 0$:
#    * **VED ($J_z \perp \text{PEC}$, $s = +1$):**
#      $$\frac{R_{\text{VED}}(h)}{R_{\text{free}}} = 1 + 3\,\frac{j_1(2k_0 h)}{2k_0 h} = 1 - 3\left[\frac{\cos(2k_0 h)}{(2k_0 h)^2} - \frac{\sin(2k_0 h)}{(2k_0 h)^3}\right] \;\xrightarrow{h\to 0}\; 2 - \frac{2}{5}(k_0 h)^2 \to 2.$$
#    * **VMD ($M_z \perp \text{PEC}$, $s = -1$):**
#      $$\frac{R_{\text{VMD}}(h)}{R_{\text{free}}} = 1 - 3\,\frac{j_1(2k_0 h)}{2k_0 h} = 1 + 3\left[\frac{\cos(2k_0 h)}{(2k_0 h)^2} - \frac{\sin(2k_0 h)}{(2k_0 h)^3}\right] \;\xrightarrow{h\to 0}\; \frac{2}{5}(k_0 h)^2 \to 0.$$
#
# 2. **Horizontal dipoles ($\mathbf{M} \parallel \text{PEC}$ or $\mathbf{J} \parallel \text{PEC}$, azimuthally averaged pattern $\propto \frac{1}{2}(1 + \cos^2\theta) = \frac{1}{2}(1 + u^2)$):**
#    $$\frac{R_{\text{rad}}^{\parallel}(h)}{R_{\text{free}}} = \frac{3}{4}\int_0^1 (1 + u^2)\big[1 + s\cos(\xi u)\big]\,du = 1 + \frac{3s}{2}\left[\frac{\sin\xi}{\xi} + \frac{\cos\xi}{\xi^2} - \frac{\sin\xi}{\xi^3}\right] = 1 + \frac{3s}{2}\left[j_0(\xi) - \frac{j_1(\xi)}{\xi}\right].$$
#    Using $j_0(\xi) - \frac{j_1(\xi)}{\xi} = \frac{2}{3} - \frac{2\xi^2}{15} + \mathcal{O}(\xi^4)$ as $\xi = 2k_0 h \to 0$:
#    * **HMD ($M_x \parallel \text{PEC}$, $s = +1$):**
#      $$\frac{R_{\text{HMD}}(h)}{R_{\text{free}}} = 1 + \frac{3}{2}\left[j_0(2k_0 h) - \frac{j_1(2k_0 h)}{2k_0 h}\right] \;\xrightarrow{h\to 0}\; 2 - \frac{4}{5}(k_0 h)^2 \to 2.$$
#    * **HED ($J_x \parallel \text{PEC}$, $s = -1$):**
#      $$\frac{R_{\text{HED}}(h)}{R_{\text{free}}} = 1 - \frac{3}{2}\left[j_0(2k_0 h) - \frac{j_1(2k_0 h)}{2k_0 h}\right] \;\xrightarrow{h\to 0}\; \frac{4}{5}(k_0 h)^2 \to 0.$$
#
# > [!IMPORTANT]
# > **Two physical conditions must always be stated alongside PEC ground-plane claims:**
# > 1. **Orientation condition:** A magnetic antenna is enhanced by a nearby PEC ground plane **only when its magnetic dipole axis is tangential to the conductor** ($\mathbf{M} \parallel \text{PEC}$, HMD). A vertical magnetic dipole ($\mathbf{M} \perp \text{PEC}$, VMD, such as a coplanar loop lying flat on a metal sheet) induces an opposing image magnetic dipole and is shorted out as $(k_0 h)^2 \to 0$.
# > 2. **Fixed-moment intensity ($4\times = +6.02\;\text{dB}$), power ($2\times = +3.01\;\text{dB}$), and directivity ($2\times = +3.01\;\text{dB}$) distinction:** For a fixed magnetic current moment $I_m \ell = \int |\mathbf{M}|\,dV$ as $h \to 0^+$, the far-field amplitude in the upper hemisphere doubles ($|\mathbf{E}| \to 2|\mathbf{E}_{\text{free}}|$), the radiation intensity $U(\theta,\phi)$ quadruples ($4\times = +6.02\;\text{dB}$), and integrating over the upper hemisphere ($2\pi\;\text{sr}$ vs $4\pi\;\text{sr}$) shows that total radiated power and radiation resistance **double** ($P_{\text{rad}} \to 2 P_{\text{free}}$, $R_{\text{rad}} \to 2 R_{\text{free}}$, $+3.01\;\text{dB}$). Because directivity is $D = 4\pi U_{\max}/P_{\text{rad}} = 4\pi(4 U_{\text{free},\max})/(2 P_{\text{free}}) = 2 D_{\text{free}} = 3.0$, the directivity **doubles** ($+3.01\;\text{dB}$ over the free-space dipole $D_{\text{free}} = 1.5$, reaching $10\log_{10} 3 = 4.77\;\text{dBi}$).
#
# ---
#
# ### 1.3 Closed-form permeable rod efficiency, the $6\pi\eta_0$ impedance constant, and the volume convention (derived here)
#
# Consider an electrically small permeable rod of length $\ell$ and cross-sectional area $A_c$ carrying a uniform axial magnetic field $H_x$ in free space, with physical volume $V_{\text{free}} = \ell A_c$. Its equivalent magnetic current moment is $I_m \ell = \omega\mu_0|\mu_r - 1|\,H_x\,V_{\text{free}}$. In free space, an elementary magnetic dipole of moment $I_m \ell$ radiates total power
#
# $$P_{\text{rad,free}} = \frac{k_0^2 (I_m \ell)^2}{12\pi\eta_0}, \qquad \eta_0 = \sqrt{\frac{\mu_0}{\varepsilon_0}} = 376.730313\;\Omega,$$
#
# while the power dissipated in the permeable core is $P_{\text{loss,free}} = \frac{1}{2}\omega\mu_0\mu'' H_x^2 V_{\text{free}}$. Dividing $P_{\text{loss,free}}$ by $P_{\text{rad,free}}$ and substituting $f(\mu) = \mu''/|\mu_r - 1|^2 = \omega\mu_0 / h_m$ yields (**derived here** following [1]):
#
# $$\frac{P_{\text{loss,free}}}{P_{\text{rad,free}}} = \frac{\frac{1}{2}\omega\mu_0\mu'' H_x^2 V_{\text{free}}}{\dfrac{k_0^2\,\omega^2\mu_0^2|\mu_r - 1|^2 H_x^2 V_{\text{free}}^2}{12\pi\eta_0}} = \frac{6\pi\eta_0\,f(\mu)}{\omega\mu_0\,V_{\text{free}}\,k_0^2} = \frac{6\pi\eta_0}{h_m\,V_{\text{free}}\,k_0^2},$$
#
# $$\boxed{e_{\text{MD}}(\omega) = \frac{P_{\text{rad}}}{P_{\text{rad}} + P_{\text{loss}}} = \frac{1}{1 + \dfrac{6\pi\eta_0}{h_m\,\text{Vol}\,k_0^2}}, \qquad 6\pi\eta_0 = 6\pi\sqrt{\frac{\mu_0}{\varepsilon_0}} = 7101.20\;\Omega.}$$
#
# Note how the constant is written in Yousefi et al. [1]: substituting the rational approximation $\eta_0 \approx 120\pi\;\Omega$ into $1/(12\pi\eta_0) = 120\pi/(12\pi\eta_0^2) = 10/\eta_0^2$ expresses $P_{\text{rad,free}} \approx (10\,\Omega/\eta_0^2)(k_0 I_m \ell)^2$ and replaces the exact constant $6\pi\eta_0 = 7101.20\;\Omega$ with $6\pi(120\pi) = 720\pi^2 = 7106.12\;\Omega \approx 7106\;\Omega$ (which agrees with $6\pi\eta_0$ to $0.069\%$; note that evaluating $\eta_0^2/20 = 7096.29\;\Omega$ with exact $\eta_0$ mixes the two conventions because $1/(12\pi\eta_0) = 10/\eta_0^2$ holds only when $\eta_0 = 120\pi\;\Omega$).
#
# **Cauchy–Schwarz volume-filling bound vs lossless loop moment (derived here):**
#
# For a non-uniform internal magnetic field $\mathbf{H}(\mathbf{r})$ inside a core of volume $\text{Vol}$, the core polarization magnetic moment is $I_{m,\text{pol}}\ell = \omega\mu_0|\mu_r - 1|\left|\int_{\text{Vol}} H_x\,dV\right|$ while the core dissipation is $P_{\text{loss,mag}} = \frac{1}{2}\omega\mu_0\mu''\int_{\text{Vol}} |\mathbf{H}|^2\,dV$. By the Cauchy–Schwarz inequality,
#
# $$\eta_{\text{fill}} \equiv \frac{\left|\int_{\text{Vol}} H_x\,dV\right|^2}{\text{Vol}\int_{\text{Vol}} |\mathbf{H}|^2\,dV} \in (0, 1], \qquad e_{\text{pol}}(\omega) = \frac{1}{1 + \dfrac{6\pi\eta_0}{\eta_{\text{fill}}\,h_m\,\text{Vol}\,k_0^2}} \le e_{\text{MD}}(\omega).$$
#
# Thus $e_{\text{MD}}(\omega)$ is a strict upper bound on the efficiency $e_{\text{pol}}(\omega)$ produced by the **core polarization current $\mathbf{M}_p$ alone**. In our 3D FIT rod ($\ell = 72\;\text{mm}, w = 24\;\text{mm}, t = 24\;\text{mm}$ on PEC, $\text{Vol} = 2 V_{\text{phys}} = 8.2944\times 10^{-5}\;\text{m}^3$), open-end demagnetization and finite material wavelength make $\mathbf{H}(\mathbf{r})$ non-uniform across the core ($\eta_{\text{fill}} = 0.8129$ at $280\;\text{MHz}$), lowering the DFT core-polarization efficiency to $e_{\text{pol}}^{\text{DFT}} = -9.92\;\text{dB}$ versus the uniform-field upper bound $e_{\text{MD}} = -9.13\;\text{dB}$ (note that $e_{\text{pol}}^{\text{DFT}}$ equals `closed_form_md_efficiency(Vol·η_fill)` algebraically up to discrete leapfrog $\tilde{\omega}, \mu_d$ vs continuous $\omega, \mu$, so the $0.05\;\text{dB}$ agreement and $e_{\text{pol}}^{\text{DFT}} < e_{\text{MD}}$ for $\eta_{\text{fill}} < 1$ are algebraic identity/consistency checks, whereas comparing the flux-based total efficiency $e_{\text{tot}}^{\text{FIT}}$ against $e_{\text{MD}}$ is the physical check). Meanwhile, the external inductive wire loop of area $A_{\text{loop}}$ carrying current $I$ contributes a lossless vacuum magnetic dipole moment $\mathbf{m}_{\text{loop}} = \mu_0 I A_{\text{loop}}\hat{\mathbf{x}}$. Simulating the identical air-core vacuum loop (`material=None`) directly in 3D FIT shows that the rod+loop run radiates $P_{\text{rad,rod}}/P_{\text{rad,loop}} = 59.6\times$ more power than the bare loop (total moment ratio $\sqrt{59.6} = 7.72$; because the rod run already contains the loop, the rod-only to loop moment ratio is $\sqrt{59.6} - 1 = 6.72$). However, the in-phase estimate $20\log_{10}(1 + 1/\sqrt{59.6}) = 1.06\;\text{dB}$ ($\times(1 - e) = 0.94\;\text{dB}$) exceeds the measured $e_{\text{tot}}^{\text{FIT}} - e_{\text{pol}}^{\text{DFT}} = +0.31\;\text{dB}$; the loop's contribution is therefore not a coherent in-phase addition, and the residual is unexplained. Across the main passband $200\text{–}320\;\text{MHz}$ (test window `[:, 1:5]`), $e_{\text{pol}}^{\text{DFT}} (-9.92\;\text{dB}) < e_{\text{tot}}^{\text{FIT}} (-9.62\;\text{dB}) < e_{\text{MD}}^{\text{bound}} (-9.13\;\text{dB})$, whereas at the band edges $160, 360,$ and $400\;\text{MHz}$ (excluded by `[:, 1:5]`), the total FIT efficiency $e_{\text{tot}}^{\text{FIT}}$ exceeds $e_{\text{MD}}$ by up to $+0.25\;\text{dB}$ (Lorentz, Fig. 4b) and $+0.28\;\text{dB}$ (Debye, Fig. 4c)—physically allowed because $e_{\text{MD}}$ bounds core polarization $e_{\text{pol}}$ only.
#
# **Resolving the half-space vs free-space volume convention (derived here):**
#
# Suppose a permeable half-rod of physical volume $V_{\text{phys}}$ rests flush on an infinite PEC ground plane ($z \in [0, t]$), driven by a half-loop arch whose feet are shorted to the PEC plane. By exact image theory, the PEC boundary plane $z = 0$ is the symmetry plane of a full rod of volume $V_{\text{full}} = 2\,V_{\text{phys}}$ ($z \in [-t, +t]$) in free space carrying the same internal magnetic field $H_x$:
#
# 1. Because the upper half-space field is identical to that of the full rod in free space, the outward Poynting flux integrated over the upper hemisphere $z > 0$ is **half** of the full free-space radiated power: $P_{\text{rad}}^{\text{half,PEC}} = \frac{1}{2}\,P_{\text{rad}}^{\text{full,free}}$.
# 2. Because the permeable material in the half-space $z > 0$ has volume $V_{\text{phys}} = \frac{1}{2}\,V_{\text{full}}$ and identical internal $|\mathbf{H}|^2$, the material dissipation in the half-rod is **also half** of the full free-space material dissipation: $P_{\text{loss}}^{\text{half,PEC}} = \frac{1}{2}\,P_{\text{loss}}^{\text{full,free}}$.
# 3. Consequently, the radiation efficiency of the half-rod of physical volume $V_{\text{phys}}$ on PEC equals the radiation efficiency of the **full mirrored rod** of volume $V_{\text{full}} = 2\,V_{\text{phys}}$ in free space:
#    $$e_{\text{half,PEC}}(V_{\text{phys}}) = \frac{\frac{1}{2}P_{\text{rad}}^{\text{full,free}}}{\frac{1}{2}P_{\text{rad}}^{\text{full,free}} + \frac{1}{2}P_{\text{loss}}^{\text{full,free}}} = e_{\text{full,free}}(2\,V_{\text{phys}}) \quad\Longrightarrow\quad \text{Vol}_{\text{eff}}(h \to 0) = 2\,V_{\text{phys}} = V_{\text{full}}.$$
# 4. Conversely, if the same physical rod of volume $V_{\text{phys}}$ is lifted to height $h$ above the PEC ground plane, its internal loss for a given induced magnetic moment $I_m\ell = \omega\mu_0|\mu_r - 1|H_x V_{\text{phys}}$ remains proportional to $1/V_{\text{phys}}$, whereas its radiated power into the upper hemisphere is multiplied by the HMD image factor $F_{\text{HMD}}(k_0 h) = R_{\text{HMD}}(h)/R_{\text{free}} \in [0.66, 2]$. Thus the effective volume entering the closed-form efficiency law at arbitrary height $h$ is
#    $$\text{Vol}_{\text{eff}}(h) = F_{\text{HMD}}(k_0 h)\,V_{\text{phys}} \;\xrightarrow{h \to 0}\; 2\,V_{\text{phys}}, \qquad \text{Vol}_{\text{eff}}(h) \;\xrightarrow{h \to \infty}\; V_{\text{phys}}.$$
#
# **Dimensionless similarity scaling law for explicit 3D FIT:**
# Notice that $e_{\text{MD}}$ depends only on the dimensionless product $h_m\,\text{Vol}\,k_0^2 = (h_m \ell / \eta_0)\,(k_0 \ell)\,(k_0 A_c/\ell)\,\eta_0$. Scaling frequency up by $s$ and linear dimensions down by $1/s$ at fixed $(k_0 \ell, k_0 \rho)$ and fixed $h_m \ell$ preserves the electrical size and radiation physics while reducing the number of explicit Courant time steps by $s$. In Parts 0, 2, 3, and 6 we use a unified UHF rod geometry ($\ell = 72\;\text{mm}, w = 24\;\text{mm}, t = 24\;\text{mm}$ on PEC, $f \in [160, 400]\;\text{MHz}$, resolved on the converged $\Delta_{\text{fine}} = 4.0\;\text{mm}$ grid with $6$ cells across the rod thickness/width and $42.3\;\text{cells}/\lambda_{\text{mat}}$ inside the core), and in Part 5 we evaluate the full VHF/UHF conformal scale ($10\text{–}300\;\text{MHz}$).
#
# ---
#
# ### 1.4 PEC/PMC electromagnetic duality vs Sebastian's orientational PEC pair (derived here)
#
# Maxwell's equations in a source-free region are invariant under the duality transformation
#
# $$\mathbf{E} \to \eta_0\mathbf{H}, \qquad \eta_0\mathbf{H} \to -\mathbf{E}, \qquad \varepsilon_r(\omega) \leftrightarrow \mu_r(\omega), \qquad \mathbf{J} \leftrightarrow \frac{\mathbf{M}}{\eta_0}, \qquad \text{PEC}\;(\hat{\mathbf{n}}\times\mathbf{E}=\mathbf{0}) \leftrightarrow \text{PMC}\;(\hat{\mathbf{n}}\times\mathbf{H}=\mathbf{0}).$$
#
# Applying this transformation to a **horizontal permeable rod on a PEC ground plane** ($\mu_r = \mu' - j\mu''$, $\varepsilon_r = 1$, $\mathbf{M}_p \parallel \text{PEC}$) distinguishes two physical configurations (**derived here** following [2]):
#
# 1. **Continuous Maxwell boundary-value dual (horizontal dielectric rod on a dual-shifted PMC ground plane, discrete dual up to outer-grid grading $\mathcal{O}((g-1)^2\Delta)$):** Keeping the rod geometry and horizontal orientation identical ($\mathbf{J}_p \parallel \text{ground}$), swapping $\mu_r(\omega) \leftrightarrow \varepsilon_r(\omega)$, replacing the inductive loop exciting $H_x$ with its dual electric source exciting $E_x$, and swapping the PEC ground plane ($\hat{\mathbf{z}}\times\mathbf{E} = \mathbf{0}$) to a **PMC ground plane** ($\hat{\mathbf{z}}\times\mathbf{H} = \mathbf{0}$, `bc_low=['cpml','cpml','pmc']`) on a half-cell dual-shifted FIT grid (so primary electric nodes $\mathbf{E}$ of the PMC problem sit at the exact coordinates of dual magnetic nodes $\mathbf{H}$ of the PEC problem) yields an exact continuous electromagnetic dual. On a **uniform** grid, discrete primal/dual symmetry is exact; on a **non-uniformly graded** outer buffer grid with geometric stretch ratio $g > 1$, shifting the grid by half a cell shifts the geometric transition points in the outer air region and introduces an $\mathcal{O}((g - 1)^2\Delta)$ truncation difference. Refining the outer grading ratio from $g = 1.35$ (`dual_pmc_coarse_max_diff_f0_db = 0.171 dB`) to $g = 1.18$ (`dual_pmc_max_diff_f0_db = 0.038 dB`) reduces the maximum PEC$\leftrightarrow$PMC efficiency difference by $4.5\times$ to $0.038\;\text{dB}$ across $\sim 1.2$ decades of susceptibility.
# 2. **Sebastian's orientational PEC pair (vertical dielectric monopole on a PEC ground plane, Sebastian Figs. 3-16/3-17 [2]):** In practical applications where a physical ground plane is metallic (PEC) rather than PMC, a horizontal dielectric rod is shorted out by its $-1$ antiphase image ($s_{\text{HED}} = -1$). Rotating the dielectric rod by $90^\circ$ to be **vertical** ($\mathbf{J}_p \perp \text{PEC}$) restores the $+1$ constructive image sign ($s_{\text{VED}} = +1 = s_{\text{HMD}}$) so that its radiation efficiency is governed by the same hesitivity law $h_e = \omega\varepsilon_0 |\varepsilon_r - 1|^2 / \varepsilon''$ when $\varepsilon_r(\omega) = \mu_r(\omega)$. However, this orientational PEC pair is **not** a boundary-value dual because the rod axis is perpendicular rather than parallel to the ground plane, resulting in a geometric/depolarization offset of $0.57\text{–}1.13\;\text{dB}$.
#
# ---
#
# ### 1.5 Conformal metal strip vs permeable bar in the same volume, EBWP, resonant-patch limit, and finite-ground diffraction
#
# For a conformal antenna restricted to a low-profile bounding box of length $\ell = 39'' = 0.9906\;\text{m}$, width $w = 3.15'' = 0.080\;\text{m}$, and thickness $t = 1'' = 0.0254\;\text{m} \ll \lambda_0$ flush on a PEC ground plane [1, 7–9, 12]:
#
# * **Conformal metal strip at height $h = t$:** Image cancellation between the strip current $J_x$ at $z = t$ and its opposing PEC image $-J_x$ at $z = -t$ suppresses the far-field array factor by $2\sin(k_0 t\cos\theta) \approx 2k_0 t\cos\theta$, reducing the radiation resistance by $\propto (k_0 t)^2$ (Yousefi et al. [1] Eq. (21); Balanis [3] Ch. 4):
#   $$R_{\text{rad}}^{\text{strip}}(\omega) \approx 200\,\Omega\,\left(\frac{\ell}{\lambda_0}\right)^2 \sin^2(k_0 t).$$
#   Simultaneously, by Wheeler's wide-strip microstrip formula $Z_{0,\text{ms}} = (\eta_0 / 2\pi)\big/\!\big[\frac{w}{2\pi t} + 0.2217 + 0.1062\ln(\frac{w}{t} + 1.444)\big] \approx 68\;\Omega$ for $w/t = 3.15$ in air [1, 12], the close PEC image creates a low characteristic transmission-line impedance $Z_{0,\text{ms}} \approx 68\;\Omega$ and large reactive stored energy, yielding a small Best–Yaghjian fractional matched bandwidth [9]
#   $$\text{FBW}_{s}(\omega) = \frac{4\sqrt{\beta}\,R_{\text{in}}(\omega)}{\omega\,|Z_{\text{in}}'(\omega)|}, \qquad \sqrt{\beta} = \frac{s - 1}{2\sqrt{s}}\;\;\left(\text{for target VSWR } s\right), \qquad \text{EBWP}(\omega) \equiv e_{\text{rad}}(\omega)\,\text{FBW}_s(\omega).$$
# * **Conformal permeable bar of thickness $t$ on PEC:** The horizontal magnetic polarization current $M_x$ inside the permeable bar adds constructively with its PEC image ($s_{\text{HMD}} = +1$), avoiding the $\sin^2(k_0 t)$ image cancellation and raising the radiation resistance by **more than two orders of magnitude** ($> 10^2\times$, verified directly in 3D FIT below). Evaluating the analytical lumped/transmission-line impedance model of [1] with a single-turn feed ($N = 1$, no empirical turn-count fitting) across $10\text{–}300\;\text{MHz}$ shows a $19\text{–}43\;\text{dB}$ $\text{EBWP}$ advantage over the conformal metal strip in the same volume.
# * **Contrast with resonant standing-wave cavity patches (Ikonen et al. 2006 [10]):** In a half-wave microstrip patch resonator filled with a dispersive magnetic substrate operating at a fixed cavity eigenmode (with benchmark unloaded radiation quality factor $Q_{\text{rad}} = 35$ following Ikonen et al. [10] Sec. III), the total loaded quality factor is $Q_{\text{tot}}^{-1} = Q_{\text{rad}}^{-1} + \tan\delta_m$. Because radiation efficiency is $e_{\text{patch}} = Q_{\text{tot}}/Q_{\text{rad}} = 1/(1 + Q_{\text{rad}}\tan\delta_m)$ while fractional bandwidth is $\text{FBW}_{\text{patch}} = 2\sqrt{\beta}/Q_{\text{tot}}$, their product $\text{EBWP}_{\text{patch}} = e_{\text{patch}}\,\text{FBW}_{\text{patch}} = 2\sqrt{\beta}/Q_{\text{rad}}$ is **independent of $\tan\delta_m$**: magnetic loss widens the impedance bandwidth of a resonant cavity only by dissipating power as heat. Electrically small flux-channeled permeable rods operate far below cavity self-resonance in the dipole flux-channeling regime, where $e_{\text{MD}}$ is governed by $h_m\,\text{Vol}\,k_0^2$ rather than cavity $Q_{\text{rad}}\tan\delta_m$.
# * **Finite PEC ground plane edge diffraction (derived here following [3]):** For an HMD ($M_x \parallel \text{PEC}$) at height $h = 0.5\Delta_g = 0.025\lambda_0$ above a finite square PEC plate of side $L_{\text{gnd}}$ ($x, y \in [-L_{\text{gnd}}/2, +L_{\text{gnd}}/2]$), the primary magnetic dipole wave grazes the E-plane plate edges at $y = \pm L_{\text{gnd}}/2$, where the truncation of the tangential surface current $J_y$ excites Geometrical Theory of Diffraction (GTD) edge-diffracted waves [3]. While the integrated radiation resistance ratio $R_{\text{rad}}(L_{\text{gnd}})/R_{\text{free}}$ settles toward the infinite-PEC value $R_\infty/R_{\text{free}} = 1.980$ as $L_{\text{gnd}} \to 1.10\lambda_0$ ($R_{\text{rad}}/R_{\text{free}} = 1.998$), the two symmetric edge waves from $y = \pm L_{\text{gnd}}/2$ always add in phase with each other at broadside ($\theta = 0^\circ$, where path lengths from the two edges are equal); the net broadside enhancement depends on their phase relative to the direct wave ($\approx k_0 L_{\text{gnd}}/2$ plus the diffraction-coefficient phase), which predicts a turn-down at larger $L_{\text{gnd}}$ that is untested here ($D(\theta = 0^\circ)$ rises $3.200 \to 4.738$, or $6.76\;\text{dBi}$, $+2.05\;\text{dB}$ above $D_\infty(0^\circ) = 2.956$, over $L_{\text{gnd}} = 0.30\text{–}1.10\lambda_0$).
#
# ---
#
# ## 2. Simulation Setup, Mesh Criteria & Preflight
#
# ![Setup geometry (XZ plane, y = 0)](figures/00a_setup_geometry.png)
# ![Setup mesh and broadband excitation](figures/00b_setup_mesh_and_excitation.png)
#
# The baseline 3D FIT setup (`xz` symmetry plane at $y = 0\;\text{mm}$) places a conformal permeable rod (`DispersiveMaterial` with Lorentz or Debye $\mu(\omega)$, static relative permeability $\mu_r(0) = 40.00$, refractive index $n_{\max} = \sqrt{40} = 6.32$, physical dimensions $\ell = 72\;\text{mm} \times w = 24\;\text{mm} \times t = 24\;\text{mm}$, $V_{\text{phys}} = 4.1472\times 10^{-5}\;\text{m}^3$) flush on the PEC ground plane at $z = 0$ (`bc_low=['cpml', 'cpml', 'pec']`), excited by a divergence-free inductive loop arch (`CurrentSource`) and enclosed by a 5-face half-space `Near2FarMonitor` (`ground_plane='-z'`). All headline rod simulations in Parts 0, 2, and 3 use the **Medium grid ($3.65\%$ below the Richardson extrapolate)** ($\Delta_{\text{fine}} = 4.0\;\text{mm}$, **$6.0$ cells across both thickness and width**, **$42.3$ cells per material wavelength $\lambda_{\text{mat}} = \lambda_0 / \operatorname{Re}\sqrt{\varepsilon_r\mu_r}$** inside the high-index core, and **$54.8$ cells per magnetic skin depth $\delta_m$**). Across all three convergence grids ($\Delta_{\text{fine}} \in \{6.0, 4.0, 3.0\}\;\text{mm}$), the core is resolved with $4, 6, 8$ cells across its $24\;\text{mm}$ cross-section, $28.2$ to $56.4\;\text{cells}/\lambda_{\text{mat}}$, and $36.5$ to $73.1\;\text{cells}/\delta_m$.
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 467 mm × 419 mm × 235.4 mm (z ∈ [0 m, 235.4 mm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 43 × 31 × 19 (25327 cells) |
# | Discretization | Δx & resolution | Δ_min = 4 mm, 42.3 cells/λ (in n_max = 6.32), 6.0 cells across feature (24 mm) |
# | Boundary | -x face | CPML (4 cells, 109 mm) |
# | Boundary | +x face | CPML (4 cells, 112 mm) |
# | Boundary | -y face | CPML (4 cells, 109 mm) |
# | Boundary | +y face | CPML (4 cells, 112 mm) |
# | Boundary | -z face | PEC |
# | Boundary | +z face | CPML (4 cells, 112 mm) |
# | Material | Background | Vacuum, ε = 1.00 (isotropic) |
# | Material | Region 1: Permeable rod (CZN Lorentz mu) | ε_r = 1.00, μ_r = 40.00, model = DispersiveMaterial, x ∈ [-36 mm, 36 mm] (d = 72 mm) |
# | Source | 1 · CurrentSource | pos: z = 24 mm, pol: J_y (E_y), pulse: f₀ = 280.0 MHz, Δf = 126.0 MHz |
# | Monitor | 2 · Near2FarMonitor | Near2FarMonitor, pos: x ∈ [-91 mm, 91 mm], freqs: f = 280.0 MHz |
# | Variation | 1D Magnetic ADE slab | TEM waveguide ($2\times 2\times N_z$), Lorentz & Debye $\mu(\omega)$ slabs |
# | Variation | 4-orientation PEC image sweep | HMD, VMD, VED, HED over $h/\lambda_0 \in [0.04, 0.50]$ |
# | Variation | Volume convention & lift | Half-rod on PEC vs full free-space rod ($2 V_{\mathrm{phys}}$) vs lifted rod |
# | Variation | Equal-$h_m$ Lorentz & Debye families | 4 Lorentz poles ($h_m=3.46\times 10^5\,\Omega$/m) & 3 Debye poles ($h_m=2.36\times 10^5\,\Omega$/m) |
# | Variation | PEC/PMC duality & finite ground | Horiz permeable rod vs vert dielectric monopole; finite PEC $L_{\mathrm{gnd}}/\lambda_0$ |

# %%
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="IProgress not found")

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")
import _helpers as h
import numpy as np

DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tutorial 4.8: Magnetic antennas on and near a ground plane")
    parser.add_argument("--quick", action="store_true", help="Run reduced-resolution fast verification")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    raw = shlex.split(os.environ.get("WAKIS_TUTORIAL_ARGS", ""))
    return parser.parse_args(raw if raw else [])


ARGS = _parse_args()

# Execute all 6 tutorial parts sequentially (or load precomputed archives when matching)
summary = h.run_all_tasks(DATA_DIR, FIG_DIR, quick=ARGS.quick, overwrite=False)
d0 = np.load(DATA_DIR / "part0_magnetic_ade.npz")
d1 = np.load(DATA_DIR / "part1_image_theory.npz")
d2 = np.load(DATA_DIR / "part2_volume_convention.npz")
d3 = np.load(DATA_DIR / "part3_equal_hm.npz")
d4 = np.load(DATA_DIR / "part4_dual_pair.npz")
d5 = np.load(DATA_DIR / "part5_ebwp_and_convergence.npz")

print(f"Exact wave-impedance constant 6*pi*eta0 = {summary['c_yousefi_exact_ohm']:.2f} Ohm (120*pi form: {summary['c_yousefi_120pi_ohm']:.2f} Ohm)")
print(f"CZN Lorentz hesitivity h_m = {summary['czn_hm_ohm_m']:.1f} Ohm/m")
print(f"Max relative error of f(mu) = omega*mu0/h_m (Lorentz / Debye): {summary['rel_err_fmu_lor']:.2e} / {summary['rel_err_fmu_deb']:.2e}")
print(f"1D magnetic ADE slab max |R_FIT - R_ana| (Lorentz / Debye): {summary['slab_lor_max_err']:.5f} / {summary['slab_deb_max_err']:.5f}")
print(f"3D rod discrete energy balance error (centered vs uncentered): {summary['rod_energy_balance_rel_err']*100:.5f}% vs {summary['rod_mutated_balance_rel_err']*100:.2f}%")

# %% [markdown]
# ---
#
# ## 3. Results: Parts 0 through 6 in 3D FIT
#
# ### 3.1 Part 0 · Magnetic ADE verification, hesitivity invariance, and discrete energy balance
#
# ![Magnetic ADE dispersion, hesitivity, and 1D slab verification](figures/01_magnetic_ade_and_hesitivity.png)
#
# Figure 1 validates the magnetic Lorentz and Debye Auxiliary Differential Equations (`wakis.dispersive`) from three independent perspectives:
#
# 1. **Causal susceptibility and hesitivity peak (panels a, b):** For the CZN ferromagnetic laminate ($\chi_{\text{DC}} = 39$, $f_0 = 1.5\;\text{GHz}$, $\alpha = 4/3$), the magnetic conductivity $\sigma_m(\omega) = \omega\mu_0\mu''(\omega)$ peaks at $f_0 = 1.5\;\text{GHz}$ with $h_m = 3.4642\times 10^5\;\Omega/\text{m}$, while for a commercially available NiZn ferrite modeled by a Debye pole ($\chi_{\text{DC}} = 75$, $f_0 = 1.6\;\text{GHz}$), $\sigma_m(\omega)$ approaches $h_m = 2\sigma_m(\omega_0) = 9.4748\times 10^5\;\Omega/\text{m}$. Across $0.05\text{–}3.0\;\text{GHz}$, the exact algebraic identity $f(\mu) = \mu''/|\mu_r - 1|^2 = \omega\mu_0/h_m$ holds to machine precision ($6.13\times 10^{-16}$ for Lorentz and $6.54\times 10^{-16}$ for Debye).
# 2. **1D FIT plane-wave slab benchmark (panel c):** Propagating a broadband TEM plane wave through a magnetic ADE slab of thickness $d = 8\Delta z = 16\;\text{mm}$ in `SolverFIT3D` reproduces the exact transfer-matrix reflection $|R(\omega)|$ and transmission $|T(\omega)|$ to a maximum error of $0.00094$ ($0.094\%$) for Lorentz and $0.00102$ ($0.102\%$) for Debye, while the volumetric ADE dissipation integral $\tilde{\omega}\mu_0(-\operatorname{Im}\mu_d)\sum |H_{y,\text{mid}}|^2\Delta z$ matches the wave-power absorption $A(\omega) = 1 - |R(\omega)|^2 - |T(\omega)|^2$ across the entire passband.
# 3. **3D FIT rod hesitivity invariance and mutation test:** Simulating the unified permeable rod ($\ell = 72\;\text{mm}, w = 24\;\text{mm}, t = 24\;\text{mm}$ on PEC, $\Delta_{\text{fine}} = 4.0\;\text{mm}$, $6$ cells across the rod cross-section) at $f_0 = 280\;\text{MHz}$ with Pole A ($\chi_{\text{DC}} = 39, f_{\text{res}} = 1.5\;\text{GHz}, \alpha = 4/3, h_m = 3.464\times 10^5\;\Omega/\text{m}$) and Pole B ($\chi_{\text{DC}} = 26, f_{\text{res}} = 2.0\;\text{GHz}, \alpha = 32/27$, identical $h_m = 3.464\times 10^5\;\Omega/\text{m}$) gives radiation efficiencies of $-9.62\;\text{dB}$ and $-9.50\;\text{dB}$ (a difference of only $0.11\;\text{dB}$ despite distinct $\chi_{\text{DC}}$ and $f_{\text{res}}$). Mutating Pole A by doubling its damping ($\alpha \to 2\alpha$, halving $h_m$ to $1.732\times 10^5\;\Omega/\text{m}$) increases the loss-to-radiation ratio $(1/e - 1) = P_{\text{loss}}/P_{\text{rad}}$ by a factor of $1.9993$ (versus the exact analytical factor $2.0000$), while time-centered discrete power balance satisfies $|W_{\text{src}} - (P_{\text{rad,flux}} + P_{\text{loss,mag}})|/|W_{\text{src}}| = 5.90\times 10^{-6}$ ($0.00059\%$) compared to $7.49\%$ when half-step time centering is omitted.

# %%
print("=== Part 1: 4-Orientation PEC Image Theory Benchmark ===")
print(
    f"HMD (M_x || PEC) at h/lam0 = {summary['hmd_low_h_lam']:.4f}: "
    f"FIT R/R_free = {summary['hmd_low_h_fit']:.4f} vs exact = {summary['hmd_low_h_ana']:.4f} "
    f"(rel err = {summary['hmd_low_h_rel_err_pct']:.2f}%)"
)
print(
    f"VED (J_z _|_ PEC) at h/lam0 = {summary['hmd_low_h_lam']:.4f}: "
    f"FIT R/R_free = {summary['ved_low_h_fit']:.4f} vs exact = {summary['ved_low_h_ana']:.4f}"
)
print(
    f"VMD (M_z _|_ PEC) at h/lam0 = {summary['vmd_low_h_lam']:.4f}: "
    f"FIT R/R_free = {summary['vmd_low_h_fit']:.4f} vs exact = {summary['vmd_low_h_ana']:.4f}"
)
print(
    f"HED (J_x || PEC) at h/lam0 = {summary['vmd_low_h_lam']:.4f}: "
    f"FIT R/R_free = {summary['hed_low_h_fit']:.4f} vs exact = {summary['hed_low_h_ana']:.4f}"
)

# %% [markdown]
# ### 3.2 Part 1 · Four-orientation PEC image theory and animated wave comparison
#
# ![PEC image theory for four dipole orientations and permeable rod Hx map](figures/02_pec_image_theory_four_dipoles.png)
#
# ![Animated time-domain comparison of HMD vs HED over PEC](figures/07_hmd_vs_hed_pec_ground.gif)
#
# Figure 2 and the time-domain animation in Figure 7 verify the exact opposite PEC image rules of magnetic and electric dipoles:
#
# * **Horizontal Magnetic Dipole ($\text{HMD}, M_x \parallel \text{PEC}$) and Vertical Electric Dipole ($\text{VED}, J_z \perp \text{PEC}$):** Both have an in-phase ($s = +1$) image. At $h/\lambda_0 = 0.0625$, 3D FIT gives $R_{\text{HMD}}/R_{\text{free}} = 1.8899$ versus the exact spherical-Bessel value $1.8805$ ($0.50\%$ relative error) and $R_{\text{VED}}/R_{\text{free}} = 1.9630$ versus $1.9396$ ($1.21\%$ error), both approaching $2.0$ as $h/\lambda_0 \to 0$ and oscillating toward $1.0$ as $h/\lambda_0$ increases.
# * **Vertical Magnetic Dipole ($\text{VMD}, M_z \perp \text{PEC}$) and Horizontal Electric Dipole ($\text{HED}, J_x \parallel \text{PEC}$):** Both have an antiphase ($s = -1$) image and are shorted out near the conductor. At $h/\lambda_0 = 0.0417$, 3D FIT gives $R_{\text{VMD}}/R_{\text{free}} = 0.0273$ versus $0.0272$ ($0.54\%$ error) and $R_{\text{HED}}/R_{\text{free}} = 0.0546$ versus $0.0541$ ($0.91\%$ error).
# * **2D field map and time-domain animation:** Panel 2(c) overlays the permeable rod geometry, inductive loop source, and `Near2FarMonitor` box (`schematic.overlay_geometry`) on normalized $\operatorname{Re}[H_x]/\max|H_x|$ at $f_0 = 280\;\text{MHz}$, showing strong axial magnetic flux channeling right at the PEC surface ($z = 0$) where tangential $H_x$ is doubled. Figure 7 animates the steady-state wave radiation in the $yz$ plane ($x = 0$) for an HMD ($M_x$ at $h = 0.04\lambda_0$, panel a) versus an HED ($J_x$ at $h = 0.04\lambda_0$, panel b) on the same color scale, visually demonstrating constructive hemispherical wave launching for $M_x \parallel \text{PEC}$ versus destructive image cancellation for $J_x \parallel \text{PEC}$.

# %%
print("=== Part 2: Free-Space vs Half-Space Volume Convention (f0 = 280 MHz) ===")
print(f"3D FIT Half-rod (V_phys) flush on PEC:         e = {summary['vol_half_pec_eff']*100:.2f}%")
print(f"3D FIT Full mirrored rod (2*V_phys) free space: e = {summary['vol_full_free_eff']*100:.2f}%")
print(f"Closed-form Eq. (19b) with Vol = 2*V_phys:     e = {summary['vol_ana_full_eff']*100:.2f}%")
print(f"3D FIT Physical rod (V_phys) in free space:    e = {summary['vol_phys_free_eff']*100:.2f}%")
print(f"Closed-form Eq. (19b) with Vol = V_phys:       e = {summary['vol_ana_phys_eff']*100:.2f}%")
print(
    f"Full-to-half ratios: P_rad_full/P_rad_half = {summary['ratio_prad_full_to_half']:.4f}, "
    f"P_loss_full/P_loss_half = {summary['ratio_ploss_full_to_half']:.4f}"
)

# %% [markdown]
# ### 3.3 Part 2 · Free-space $\to$ half-space mapping and the volume convention
#
# ![Volume convention resolution, 2x symmetry ratios, and height lift transition](figures/03_volume_convention_and_lift.png)
#
# Figure 3 resolves the volume ambiguity in $e_{\text{MD}} = [1 + 6\pi\eta_0/(h_m\,\text{Vol}\,k_0^2)]^{-1}$ on the unified converged mesh ($\Delta_{\text{fine}} = 4.0\;\text{mm}$, $6$ cells across the $24\;\text{mm}$ half-rod thickness and width, $648$ fine cells inside $V_{\text{phys}} = 4.1472\times 10^{-5}\;\text{m}^3$, with $\pm 3.65\%$ discretization error bar from Part 6):
#
# * **Flush on PEC vs mirrored free space (panels a, b):** At $f_0 = 280\;\text{MHz}$, a half-rod of physical volume $V_{\text{phys}}$ resting flush on PEC achieves $e_{\text{half,PEC}} = 10.92\%$ ($-9.62\;\text{dB}$), matching the full mirrored rod of volume $V_{\text{full}} = 2\,V_{\text{phys}} = 8.2944\times 10^{-5}\;\text{m}^3$ in free space ($e_{\text{full,free}} = 10.88\%$, relative difference $0.38\%$) and lying slightly below the uniform-field upper bound with $\text{Vol} = 2\,V_{\text{phys}}$ ($e_{\text{ana,full}} = 12.23\%$, $-9.13\;\text{dB}$) due to the core volume-filling factor $\eta_{\text{fill}} = 0.8129$. Panel 3(b) confirms the underlying symmetry identity directly from the 3D FIT monitors: both the radiated power and the material dissipation of the full free-space rod are twice those of the half-rod on PEC ($P_{\text{rad}}^{\text{full,free}}/P_{\text{rad}}^{\text{half,PEC}} = 1.9912$ and $P_{\text{loss}}^{\text{full,free}}/P_{\text{loss}}^{\text{half,PEC}} = 1.9998$).
# * **Same physical rod in free space and lifted to height $h$ (panels a, c):** When the same physical rod of volume $V_{\text{phys}}$ is placed in unbounded free space ($h \to \infty$), its efficiency drops by a factor of two to $e_{\text{phys,free}} = 5.53\%$ ($-12.57\;\text{dB}$), tracking the closed-form formula evaluated with $\text{Vol} = V_{\text{phys}}$ ($e_{\text{ana,phys}} = 6.51\%$). Using $V_{\text{phys}}$ instead of $2\,V_{\text{phys}}$ for the flush PEC half-rod underpredicts its efficiency by nearly a factor of two ($40.3\%$ error). As the rod is lifted through heights $h/\lambda_0 \in [0.0037, 0.0224]$ above the PEC plane (panel c), its 3D FIT efficiency follows the image-weighted trajectory $\text{Vol}_{\text{eff}}(h) = F_{\text{HMD}}(k_0 h)\,V_{\text{phys}}$ from the $2\,V_{\text{phys}}$ half-space curve toward the $V_{\text{phys}}$ free-space asymptote.

# %%
print("=== Part 3: Equal-h_m Lorentz & Debye Families, Cauchy-Schwarz Bound & Rumsey-Weeks ===")
print(f"4-pole Lorentz family max spread across 160-400 MHz: {summary['lor_family_spread_db']:.3f} dB")
print(f"3-pole Debye family max spread across 160-400 MHz:   {summary['deb_family_spread_db']:.3f} dB")
print(
    f"Cauchy-Schwarz fill factor eta_fill(f0) = {summary['lor_eta_fill_f0']:.4f} <= 1: "
    f"e_pol(DFT) = {summary['lor_eff_pol_f0_db']:.2f} dB < e_tot(FIT) = {summary['lor_eff_tot_f0_db']:.2f} dB "
    f"< e_MD(bound) = {summary['lor_eff_ana_f0_db']:.2f} dB "
    f"(rod/vacuum-loop P_rad ratio = {summary['rod_to_loop_prad_ratio_f0']:.1f}x, "
    f"sqrt(ratio) = {summary['rod_to_loop_moment_ratio_f0']:.2f}, "
    f"rod-only moment ratio = {summary['rod_only_to_loop_moment_ratio_f0']:.2f}, "
    f"in-phase estimate = {summary['loop_in_phase_prad_boost_db']:.2f} dB [*(1-e) = {summary['loop_in_phase_eff_boost_db']:.2f} dB] "
    f"vs measured e_tot - e_pol = +{summary['lor_eff_tot_minus_pol_f0_db']:.2f} dB; "
    f"band-edge max e_tot - e_MD = +{summary['lor_max_excess_over_bound_db']:.2f} dB [Lorentz] / +{summary['deb_max_excess_over_bound_db']:.2f} dB [Debye])"
)

print("\n=== Part 4: Continuous PEC<->PMC Dual vs Sebastian Orientational PEC Pair (f0 = 250 MHz) ===")
print(
    f"mu_r / eps_r = 300 - j15: Horiz Mag PEC = {summary['dual_300_mag_db']:.2f} dB | "
    f"Horiz Diel PMC = {summary['dual_300_pmc_db']:.2f} dB | "
    f"Vert Diel PEC = {summary['dual_300_diel_db']:.2f} dB"
)
print(
    f"mu_r / eps_r =  80 - j4 : Horiz Mag PEC = {summary['dual_80_mag_db']:.2f} dB | "
    f"Horiz Diel PMC = {summary['dual_80_pmc_db']:.2f} dB | "
    f"Vert Diel PEC = {summary['dual_80_diel_db']:.2f} dB"
)
print(
    f"mu_r / eps_r =  20 - j1 : Horiz Mag PEC = {summary['dual_20_mag_db']:.2f} dB | "
    f"Horiz Diel PMC = {summary['dual_20_pmc_db']:.2f} dB | "
    f"Vert Diel PEC = {summary['dual_20_diel_db']:.2f} dB"
)
print(
    f"PEC<->PMC max diff: refined outer grading (g=1.18) = {summary['dual_pmc_max_diff_f0_db']:.3f} dB "
    f"vs coarse outer grading (g=1.35) = {summary['dual_pmc_coarse_max_diff_f0_db']:.3f} dB"
)

print("\n=== Part 5 & 6: Metal Strip vs Permeable Bar EBWP, Finite Ground & 3-Resolution Convergence ===")
print(f"3D FIT radiated power ratio (conformal magnetic bar / electric strip): {summary['fit_mbar_to_strip_ratio']:.1f}x")
print(f"Analytical circuit model EBWP advantage (N=1, 10-300 MHz): {summary['ebwp_min_advantage_db']:.2f} to {summary['ebwp_max_advantage_db']:.2f} dB")
print(
    f"Finite PEC ground (h = 0.025 lam0, L_gnd = 0.30 -> 1.10 lam0): R_rad/R_free = {summary['finite_gnd_r_ratio_030']:.3f} -> {summary['finite_gnd_r_ratio_110']:.3f} "
    f"(inf-PEC = {summary['inf_gnd_r_ratio']:.3f}), zenith directivity D(0) = {summary['finite_gnd_zenith_dir_030']:.3f} -> {summary['finite_gnd_zenith_dir_110']:.3f} "
    f"(inf-PEC = {summary['inf_gnd_zenith_dir']:.3f})"
)
print(
    f"3-resolution convergence (Coarse 6.0 mm [4 cells] / Med 4.0 mm [6 cells] / Fine 3.0 mm [8 cells] / Rich): "
    f"{summary['conv_eff_coarse']:.5f} / {summary['conv_eff_medium']:.5f} / "
    f"{summary['conv_eff_fine']:.5f} / {summary['conv_eff_rich']:.5f} "
    f"(p_obs = {summary['conv_p_obs']:.2f}, rel err = {summary['conv_coarse_rel_err_pct']:.2f}% / {summary['conv_medium_rel_err_pct']:.2f}% / {summary['conv_fine_rel_err_pct']:.2f}%)"
)

# %% [markdown]
# ### 3.4 Part 3 · Why loss tangent $\tan\delta_m$ does not govern efficiency at equal $h_m$, and Cauchy–Schwarz reconciliation
#
# ![Equal-hm Lorentz and Debye efficiency collapse vs Rumsey-Weeks](figures/04_equal_hm_collapse_vs_rumsey.png)
#
# Figure 4 tests four distinct Lorentz poles sharing $h_m = 3.464\times 10^5\;\Omega/\text{m}$—`Lor-1` ($f_{\text{res}} = 0.80\;\text{GHz}, \chi_{\text{DC}} = 54$), `Lor-2` ($1.15\;\text{GHz}, \chi_{\text{DC}} = 45$), `Lor-3` ($1.50\;\text{GHz}, \chi_{\text{DC}} = 39$), and `Lor-4` ($2.10\;\text{GHz}, \chi_{\text{DC}} = 32$)—and three Debye poles sharing $h_m = 2.357\times 10^5\;\Omega/\text{m}$ ($\chi_{\text{DC}} \in \{56, 42, 32\}$) on the unified converged mesh ($\Delta_{\text{fine}} = 4.0\;\text{mm}$, $6$ cells across $24\;\text{mm}$):
#
# * **Universal equal-$h_m$ collapse vs Rumsey–Weeks:** Although the magnetic loss tangent $\tan\delta_m = \mu''/\mu'$ varies by more than a factor of $2.1\times$ across the Lorentz family (panel a), all four 3D FIT efficiency curves collapse onto a single universal trajectory across $160\text{–}400\;\text{MHz}$ with a maximum peak-to-peak spread of **$0.156\;\text{dB}$** (panel b), and all three Debye poles collapse with a maximum spread of **$0.130\;\text{dB}$** (panel c). By contrast, the legacy Rumsey–Weeks formula (dotted red curve in panels b and c), which replaces $|\mu_r - 1|^2$ with $(\mu')^2$, underpredicts the high-frequency Lorentz efficiency as $\omega$ approaches $\omega_0$ and $\mu''$ becomes comparable to $\mu' - 1$.
# * **Reconciling total 3D FIT efficiency $e_{\text{tot}}$, DFT core-polarization efficiency $e_{\text{pol}}^{\text{DFT}}$, and the uniform-field upper bound $e_{\text{MD}}$:** As derived in Section 1.3, the closed-form Yousefi–Diaz curve $e_{\text{MD}} = -9.13\;\text{dB}$ at $f_0 = 280\;\text{MHz}$ (black dashed curve in Figure 4b) assumes a spatially uniform axial magnetic field ($\eta_{\text{fill}} = 1$). Integrating the 3D FIT discrete Fourier transform (DFT) magnetic field across the permeable core at $280\;\text{MHz}$ yields a Cauchy–Schwarz volume-filling factor $\eta_{\text{fill}} = \left|\int H_x\,dV\right|^2 / \left(\text{Vol}\int |\mathbf{H}|^2\,dV\right) = 0.8129 \le 1$, and evaluating the core-polarization efficiency from the DFT magnetic moment $I_{m,\text{pol}}\ell = \tilde{\omega}\mu_0|\mu_d - 1|\left|\int H_x\,dV\right|$ and DFT core loss $P_{\text{loss,mag}}$ gives $e_{\text{pol}}^{\text{DFT}} = -9.92\;\text{dB}$ ($0.80\;\text{dB}$ below $e_{\text{MD}}$). Because $e_{\text{pol}}^{\text{DFT}}$ equals `closed_form_md_efficiency(Vol·η_fill)` algebraically (up to discrete $\tilde{\omega}, \mu_d$ vs continuous $\omega, \mu$), the $0.05\;\text{dB}$ agreement and $e_{\text{pol}}^{\text{DFT}} < e_{\text{MD}}$ for $\eta_{\text{fill}} < 1$ are algebraic identity/consistency checks, while the physical check is that the flux-based total 3D FIT efficiency $e_{\text{tot}}^{\text{FIT}}$ lies below $e_{\text{MD}}$ across the $200\text{–}320\;\text{MHz}$ passband (`[:, 1:5]`). At $160, 360,$ and $400\;\text{MHz}$, the total FIT efficiency exceeds $e_{\text{MD}}$ by up to $+0.25\;\text{dB}$ (Lorentz, Fig. 4b) / $+0.28\;\text{dB}$ (Debye, Fig. 4c)—physically allowed because $e_{\text{MD}}$ bounds core polarization only—and the test window `[:, 1:5]` excludes those points. Meanwhile, running an otherwise identical 3D FIT simulation of the air-core vacuum loop (`material=None`) shows that the rod+loop run radiates **$59.6\times$** more power than the bare loop (total moment ratio $\sqrt{59.6} = 7.72$; since the rod run already contains the loop, the rod-only moment ratio is $\sqrt{59.6} - 1 = 6.72$). However, the in-phase estimate $20\log_{10}(1 + 1/\sqrt{59.6}) = 1.06\;\text{dB}$ ($\times(1 - e) = 0.94\;\text{dB}$) exceeds the measured $e_{\text{tot}}^{\text{FIT}} - e_{\text{pol}}^{\text{DFT}} = +0.31\;\text{dB}$ ($e_{\text{tot}}^{\text{FIT}} = -9.62\;\text{dB} \pm 0.16\;\text{dB}$ vs $e_{\text{pol}}^{\text{DFT}} = -9.92\;\text{dB}$); the loop's contribution is therefore not a coherent in-phase addition, and the residual is unexplained.
#
# ---
#
# ### 3.5 Part 4 · Continuous PEC$\leftrightarrow$PMC duality versus Sebastian's orientational PEC pair
#
# Figure 5(a) compares three constitutive parameter pairs—$300 - j15$, $80 - j4$, and $20 - j1$ (spanning $\sim 1.2$ decades of susceptibility with identical loss tangent $0.05$ at $f_0 = 250\;\text{MHz}$ and hesitivity proportional to $|\chi|^2/\chi''$)—across three 3D FIT configurations on a graded mesh with $4$ uniform cells across the rod cross-section:
#
# 1. **Horizontal permeable rod on PEC ($\mu_r$, solid lines):** Achieves $-3.93\;\text{dB}$ ($300-j15$), $-8.11\;\text{dB}$ ($80-j4$), and $-13.53\;\text{dB}$ ($20-j1$) at $f_0 = 250\;\text{MHz}$.
# 2. **Continuous Maxwell boundary-value dual—horizontal dielectric rod on a dual-shifted PMC grid ($\varepsilon_r = \mu_r$, `bc_low=['cpml','cpml','pmc']`, crosses $\times$):** Built with the exact same horizontal rod dimensions ($12\Delta \times 4\Delta \times 4\Delta$), the exact dual source geometry ($8\Delta \times 4\Delta \times 5\Delta$), and a half-cell shifted grid inside the uniform core/monitor box so primary electric nodes of the PMC problem coincide with dual magnetic nodes of the PEC problem. With refined outer air-buffer grading ($g = 1.18$), it achieves $-3.96\;\text{dB}$, $-8.14\;\text{dB}$, and $-13.57\;\text{dB}$ at $250\;\text{MHz}$—agreeing with the horizontal permeable rod on PEC to within **$0.024\text{–}0.038\;\text{dB}$** across the entire $15\times$ range of susceptibility (shrinking by $4.5\times$ from the $0.171\;\text{dB}$ discrepancy on a coarser $g = 1.35$ outer grading, confirming that the residual gap is purely the $\mathcal{O}((g-1)^2\Delta)$ outer-grid grading error).
# 3. **Sebastian's orientational PEC pair—vertical dielectric monopole on PEC ($\varepsilon_r = \mu_r$, `bc_low=['cpml','cpml','pec']`, filled circles $\bullet$):** Because the ground plane remains PEC and the dielectric rod is rotated vertical ($\mathbf{J}_p \perp \text{PEC}$) to preserve the $+1$ image sign ($s_{\text{VED}} = s_{\text{HMD}} = +1$) [2], it is an orientational PEC pair rather than a boundary-value dual. It achieves $-3.36\;\text{dB}$, $-7.25\;\text{dB}$, and $-12.40\;\text{dB}$ at $250\;\text{MHz}$, tracking the same hesitivity scaling across susceptibility with a $0.57\text{–}1.13\;\text{dB}$ geometric/depolarization offset.
#
# ---
#
# ### 3.6 Parts 5 & 6 · Conformal metal strip vs permeable bar EBWP, finite ground diffraction, and convergence
#
# ![PEC/PMC duality, radiation resistance degradation, and EBWP](figures/05_dual_pair_and_ebwp.png)
#
# ![Finite PEC ground pattern tilt, Ikonen cavity patch limit, and 3-resolution FIT convergence](figures/06_finite_ground_and_convergence.png)
#
# * **Radiation resistance (3D FIT) and EBWP (analytical circuit model) in a conformal $39'' \times 3.15'' \times 1''$ volume (Figures 5b, 5c):** Over a PEC ground plane, a conformal copper strip at height $t = 25.4\;\text{mm}$ suffers severe image cancellation ($R_{\text{rad}} \propto (k_0 t)^2$), whereas a conformal CZN permeable bar of the same thickness channels magnetic flux parallel to the PEC surface with constructive image doubling. In 3D FIT at $t/\lambda_0 = 1/60$, the permeable bar radiates **$176.3\times$** more power than the conformal electric strip for matched free-space moment ($> 10^2\times$). Evaluating the analytical lumped/transmission-line impedance model of [1, 12] ($Z_{0,\text{ms}} \approx 68\;\Omega$, $R_{\text{rad}}^{\text{strip}} \approx 200(\ell/\lambda_0)^2\sin^2(k_0 t)$) with a single-turn feed ($N = 1$) across $10\text{–}300\;\text{MHz}$ yields an $\text{EBWP}$ advantage of **$18.97\;\text{dB}$ to $43.18\;\text{dB}$** over the open metal strip (Figure 5c).
# * **Finite PEC ground plane diffraction vs infinite PEC plane (Figure 6a):** Sweeping a finite PEC sheet of width $L_{\text{gnd}}/\lambda_0 \in \{0.30, 0.50, 0.80, 1.10\}$ strictly inside the `Near2FarMonitor` Huygens box (with $\ge 0.30\lambda_0 > \lambda_0/4$ clearance to CPML) beneath a horizontal magnetic dipole at height $h = 0.5\Delta_g = 0.025\lambda_0$ in 3D FIT separates the integrated resistance ratio from broadside edge diffraction: at $L_{\text{gnd}} = 0.30\lambda_0$, edge diffraction yields a resistance ratio $R_{\text{rad}}/R_{\text{free}} = 3.168$ while spilling power into the shadow hemisphere ($\theta > 90^\circ$) so that broadside directivity is $D(\theta = 0^\circ) = 3.200$. As the plate widens to $L_{\text{gnd}} = 1.10\lambda_0$, the integrated resistance ratio converges to $R_{\text{rad}}/R_{\text{free}} = 1.998$—within $0.9\%$ of the infinite-PEC value $R_\infty/R_{\text{free}} = 1.980$. Meanwhile, the two symmetric edge-diffracted waves from $y = \pm L_{\text{gnd}}/2$ always add at broadside ($\theta = 0^\circ$), and their net enhancement depends on their phase relative to the direct wave ($\approx k_0 L_{\text{gnd}}/2$ plus the diffraction-coefficient phase), which predicts a turn-down at larger $L_{\text{gnd}}$ that is untested here ($D(\theta = 0^\circ)$ rises $3.200 \to 4.738$ over $0.30\text{–}1.10\lambda_0$, reaching $+2.05\;\text{dB}$ above the infinite-PEC value $D_\infty(0^\circ) = 2.956$ at $1.10\lambda_0$) [3].
# * **3-resolution FIT spatial convergence on the headline Part 2/3 rod (Figure 6c):** Refining the graded mesh on the exact Part 2/3 permeable rod ($\ell = 72\;\text{mm}, w = 24\;\text{mm}, t = 24\;\text{mm}$ on PEC, $f_0 = 280\;\text{MHz}$) across $\Delta_{\text{fine}} \in \{6.0, 4.0, 3.0\}\;\text{mm}$—corresponding to **$4, 6, 8$ cells across the $24\;\text{mm}$ rod thickness and width** ($\ge 4$ cells across at the coarsest level), $28.2 \to 56.4$ cells per material wavelength $\lambda_{\text{mat}} = \lambda_0/\operatorname{Re}\sqrt{\mu_r}$, and $36.5 \to 73.1$ cells per magnetic skin depth $\delta_m$—produces strictly monotone convergence of the radiation efficiency from $0.10511$ (Coarse, $7.26\%$ error) to $0.10919$ (Medium headline grid, $3.65\%$ error) and $0.11079$ (Fine, $2.24\%$ error). Computing the observed convergence order directly from the three grid levels yields **$p_{\text{obs}} = 1.69$**, with a Richardson extrapolate of $e_{\text{Rich}} = 0.11333$, an $\mathcal{O}(\Delta^2)$ reference slope anchored at the Coarse point only in Figure 6(c), and discrete power balance error below $10^{-3}\%$ on every grid.
#
# ---
#
# ## 4. Quantitative Summary
#
# Every numerical value in the table below is generated directly by `_helpers.py` and verified against `data/summary.json` by the automated test suite.
#
# | Part & Physical Quantity | 3D FIT / Numerical Value | Analytical / Reference Value | Relative Error / Spread |
# | :--- | :--- | :--- | :--- |
# | **Part 0:** Exact impedance constant $6\pi\eta_0$ vs $\eta_0 \approx 120\pi$ form $720\pi^2$ | $7101.20\;\Omega$ | $7106.12\;\Omega$ ($120\pi$ form [1]) | $0.069\%$ |
# | **Part 0:** CZN Lorentz hesitivity $h_m = \omega_0\mu_0\chi_{\text{DC}}/\alpha$ | $346423.1\;\Omega/\text{m}$ | $346423.1\;\Omega/\text{m}$ | $0.00\%$ |
# | **Part 0:** Algebraic identity $f(\mu) = \omega\mu_0/h_m$ (Lorentz / Debye) | $\max\text{rel. err.} = 6.13\times 10^{-16}$ | Exact $0$ | $6.54\times 10^{-16}$ (Debye) |
# | **Part 0:** 1D magnetic ADE slab $\max\big||R_{\text{FIT}}| - |R_{\text{ana}}|\big|$ (Lorentz / Debye) | $0.00094$ / $0.00102$ | $0.00000$ | $0.094\%$ / $0.102\%$ |
# | **Part 0:** 3D rod discrete power balance $|W_{\text{src}} - P_{\text{tot}}|/|W_{\text{src}}|$ | $5.90\times 10^{-6}$ (centered) | $0$ (uncentered: $0.0749$) | $0.00059\%$ vs $7.49\%$ |
# | **Part 0:** 3D rod equal-$h_m$ Pole A vs Pole B efficiency ($280\;\text{MHz}$) | $-9.62\;\text{dB}$ vs $-9.50\;\text{dB}$ | Equal $h_m = 3.464\times 10^5\;\Omega/\text{m}$ | $0.11\;\text{dB}$ |
# | **Part 0:** Mutated damping $\alpha \to 2\alpha$ loss-ratio factor $(1/e_C - 1)/(1/e_A - 1)$ | $1.9993$ | $2.0000$ | $0.03\%$ |
# | **Part 1:** HMD ($M_x \parallel \text{PEC}$) $R_{\text{rad}}/R_{\text{free}}$ at $h/\lambda_0 = 0.0625$ | $1.8899$ | $1.8805$ | $0.50\%$ |
# | **Part 1:** VED ($J_z \perp \text{PEC}$) $R_{\text{rad}}/R_{\text{free}}$ at $h/\lambda_0 = 0.0625$ | $1.9630$ | $1.9396$ | $1.21\%$ |
# | **Part 1:** VMD ($M_z \perp \text{PEC}$) $R_{\text{rad}}/R_{\text{free}}$ at $h/\lambda_0 = 0.0417$ | $0.0273$ | $0.0272$ | $0.54\%$ |
# | **Part 1:** HED ($J_x \parallel \text{PEC}$) $R_{\text{rad}}/R_{\text{free}}$ at $h/\lambda_0 = 0.0417$ | $0.0546$ | $0.0541$ | $0.91\%$ |
# | **Part 2:** Half-rod ($V_{\text{phys}}$) on PEC vs full rod ($2 V_{\text{phys}}$) free-space $e$ | $10.92\%$ vs $10.88\%$ | $12.23\%$ ($\text{Vol} = 2 V_{\text{phys}}$ bound) | $0.38\%$ (PEC vs full) |
# | **Part 2:** Physical rod ($V_{\text{phys}}$) in free space ($h \to \infty$) $e$ | $5.53\%$ | $6.51\%$ ($\text{Vol} = V_{\text{phys}}$ bound) | $3.65\%$ grid err |
# | **Part 2:** Full-space to half-space ratios $P_{\text{rad,full}}/P_{\text{rad,half}}$ & $P_{\text{loss,full}}/P_{\text{loss,half}}$ | $1.9912$ & $1.9998$ | $2.0000$ & $2.0000$ | $0.44\%$ & $0.01\%$ |
# | **Part 3:** 4-pole Lorentz / 3-pole Debye equal-$h_m$ efficiency spread | $0.156\;\text{dB}$ / $0.130\;\text{dB}$ | $0.000\;\text{dB}$ (ideal dipole) | $\le 0.16\;\text{dB}$ |
# | **Part 3:** Cauchy–Schwarz fill $\eta_{\text{fill}} = 0.8129$ & rod/loop ratio $59.6\times$ ($\sqrt{59.6}=7.72$, rod-only $6.72$) | $-9.92\;\text{dB}$ ($e_{\text{pol}}^{\text{DFT}}$) / $-9.62\;\text{dB}$ ($e_{\text{tot}}^{\text{FIT}}$, $\Delta = +0.31\;\text{dB}$) | $-9.13\;\text{dB}$ ($e_{\text{MD}}$ bound; in-phase loop $1.06\;\text{dB}$, $\times(1-e) = 0.94\;\text{dB}$) | $200\text{–}320\;\text{MHz}$: $e_{\text{pol}}^{\text{DFT}} < e_{\text{tot}}^{\text{FIT}} < e_{\text{MD}}$ (edges up to $+0.25/+0.28\;\text{dB}$) |
# | **Part 4:** Continuous PEC$\leftrightarrow$PMC dual ($300-j15$ / $80-j4$ / $20-j1$ at $250\;\text{MHz}$) | $-3.93 / -8.11 / -13.53\;\text{dB}$ (PEC) | $-3.96 / -8.14 / -13.57\;\text{dB}$ (PMC) | $0.038\;\text{dB}$ ($g=1.18$) vs $0.171\;\text{dB}$ ($g=1.35$) |
# | **Part 4:** Sebastian orientational PEC pair ($\mu_r$ horiz vs $\varepsilon_r$ vert on PEC) | $-3.93 / -8.11 / -13.53\;\text{dB}$ | $-3.36 / -7.25 / -12.40\;\text{dB}$ | $0.57\text{–}1.13\;\text{dB}$ |
# | **Part 5:** Conformal magnetic bar vs electric strip radiation ratio ($t/\lambda_0 = 1/60$) | $176.3\times$ (3D FIT) | $> 100\times$ ($> 10^2$) | $18.97\text{–}43.18\;\text{dB}$ EBWP ($N=1$) |
# | **Part 5:** Finite PEC ground ($h=0.025\lambda_0, L_{\text{gnd}} = 0.30\lambda_0 \to 1.10\lambda_0$) vs $\infty$-PEC | $R/R_{\text{free}} = 3.168 \to 1.998$ ($\infty$: $1.980$) | $D(0^\circ) = 3.200 \to 4.738$ ($\infty$: $2.956$) | Edge diffraction $+2.05\;\text{dB}$ over $D_\infty(0^\circ)$ |
# | **Part 6:** 3-resolution FIT convergence ($\Delta_{\text{fine}} = 6.0 / 4.0 / 3.0\;\text{mm}$ vs Rich) | $0.10511 / 0.10919 / 0.11079$ ($p_{\text{obs}} = 1.69$) | $0.11333$ ($e_{\text{Rich}}$, $56.4\;\text{cells}/\lambda_{\text{mat}}$) | $7.26\% / 3.65\% / 2.24\%$ |
#
# ---
#
# ## 5. Limits & Failure Modes
#
# Every closed-form model validated in this tutorial rests on specific physical assumptions that can fail outside their regime of validity:
#
# 1. **Electrically thin cross-section and uniform internal $\mathbf{H}$ assumption ($2\rho \ll \lambda_{\text{mat}}/2$ and $2\rho \ll 2\delta_m$):**
#    The Yousefi–Diaz efficiency formula $e_{\text{MD}} = [1 + 6\pi\eta_0/(h_m\,\text{Vol}\,k_0^2)]^{-1}$ assumes that the driving magnetic field $\mathbf{H}$ penetrates the entire permeable cross-section uniformly ($\eta_{\text{fill}} \to 1$) so that $I_m\ell \propto \text{Vol}\,H_x$ and $P_{\text{loss}} \propto \text{Vol}\,H_x^2$.
#    * *Counter-case:* Even at $24\;\text{mm}$ cross-section ($f_0 = 280\;\text{MHz}$), open-end demagnetization and finite material wavelength reduce the Cauchy–Schwarz filling factor to $\eta_{\text{fill}} = 0.8129$, lowering the core polarization efficiency $e_{\text{pol}}^{\text{DFT}} = -9.92\;\text{dB}$ by $0.80\;\text{dB}$ below the uniform-field bound $e_{\text{MD}} = -9.13\;\text{dB}$. If the rod diameter $2\rho$ exceeds half a wavelength inside the permeable medium ($\lambda_{\text{mat}} = \lambda_0 / \operatorname{Re}\sqrt{\varepsilon_r\mu_r}$) or twice the magnetic/conductive skin depth $\delta_m = 1/[k_0\operatorname{Im}\sqrt{\varepsilon_r\mu_r}]$, internal standing-wave cavity resonances ($\text{TE}_{01\delta}$/$\text{HE}_{11\delta}$ magneto-dielectric resonator modes) or skin-effect flux expulsion confine $\mathbf{H}$ to a surface shell ($\eta_{\text{fill}} \ll 1$). Increasing the core radius beyond $\delta_m$ then adds dead interior volume without increasing $I_m\ell$, causing the core efficiency to saturate well below $e_{\text{MD}}$. Similarly, in metallic ferromagnetic alloys such as CoZrNb ($\sigma_e \sim 10^6\;\text{S/m}$), the material **must** be laminated into micron-scale insulated films thinner than the eddy-current skin depth $\delta_{\text{eddy}} = 1/\sqrt{\pi f \mu_0\mu' \sigma_e}$; a monolithic conductive metal bar of the same permeability is shorted by bulk eddy currents.
# 2. **Flux-channeled dipole regime versus resonant cavity patch regime ($k_0 \sqrt{\varepsilon_r\mu_r}\,\ell \ll \pi$ vs $\approx \pi$):**
#    Our hesitivity invariance result—that efficiency depends on $h_m$ rather than $\tan\delta_m$—applies specifically to electrically small **flux-channeled magnetic dipole antennas**, where the permeable core acts as an open magnetic flux conduit driven by an external current loop and the radiation resistance scales as $\propto |\mu_r - 1|^2 \text{Vol}^2 k_0^4$.
#    * *Counter-case:* In a half-wave resonant microstrip patch antenna filled with a permeable substrate (the geometry analyzed by Ikonen et al. [10], Figure 6b), the electric field is trapped in a standing-wave cavity mode with unloaded radiation quality factor $Q_{\text{rad}} = 35$, and the radiation efficiency is $e_{\text{patch}} = 1/(1 + Q_{\text{rad}}\tan\delta_m)$. In that resonant-cavity regime, increasing $\tan\delta_m$ directly degrades efficiency and widens fractional bandwidth by the exact same factor $Q_{\text{tot}}$, leaving $\text{EBWP}_{\text{patch}} = 2\sqrt{\beta}/Q_{\text{rad}}$ unchanged.
# 3. **Infinite PEC ground plane versus finite ground plane edge diffraction ($L_{\text{gnd}} \to \infty$ vs $L_{\text{gnd}} \sim \lambda_0$):**
#    Exact image doubling ($R_{\text{HMD}}/R_{\text{free}} \to 2$, $\text{Vol}_{\text{eff}} = 2\,V_{\text{phys}}$, and hemisphere directivity doubling $D = 2 D_{\text{free}} = 3.0$, or $+3.01\;\text{dB}$ over the free-space dipole) assumes an infinite, edgeless PEC plane.
#    * *Counter-case:* As shown in Figure 6(a), on a finite square PEC ground plane of width $L_{\text{gnd}} = 0.30\lambda_0$, diffraction at the plate edges spills power into the shadow hemisphere ($\theta > 90^\circ$) and raises the resistance ratio $R_{\text{rad}}/R_{\text{free}}$ to $3.168$. Conversely, at $L_{\text{gnd}} = 1.10\lambda_0$, while the integrated resistance ratio $R_{\text{rad}}/R_{\text{free}} = 1.998$ has already settled to within $0.9\%$ of the infinite-plane value ($1.980$ at $h = 0.025\lambda_0$), the two symmetric E-plane edge-diffracted waves always add at zenith ($\theta = 0^\circ$) with a phase relative to the direct wave ($\approx k_0 L_{\text{gnd}}/2$ plus the diffraction-coefficient phase) that predicts a turn-down at larger $L_{\text{gnd}}$ (untested here; $D(\theta = 0^\circ)$ rises $3.200 \to 4.738$ over $0.30\text{–}1.10\lambda_0$, $+2.05\;\text{dB}$ above $D_\infty(0^\circ) = 2.956$) [3].
#
# ---
#
# ## 6. References
#
# 1. T. Yousefi, T. Sebastian, and R. E. Diaz, "Why the Magnetic Loss Tangent Is Not a Relevant Constraint for Permeable Conformal Antennas," *IEEE Transactions on Antennas and Propagation*, vol. 64, no. 7, pp. 2679–2690, Jul. 2016, doi: 10.1109/TAP.2016.2554198 .
# 2. T. Sebastian, *An Investigation of Magnetic Antennas for Compact Low Frequency Communication Systems*, Ph.D. dissertation, Arizona State University, Tempe, AZ, USA, 2013.
# 3. C. A. Balanis, *Antenna Theory: Analysis and Design*, 4th ed. Hoboken, NJ, USA: Wiley, 2016, ch. 4, sec. 4.7–4.8, and ch. 12, sec. 12.5–12.6.
# 4. R. F. Harrington, *Time-Harmonic Electromagnetic Fields*. New York, NY, USA: McGraw-Hill, 1961 (reissued Wiley-IEEE Press, 2001), ch. 3, sec. 3-5.
# 5. V. H. Rumsey and W. L. Weeks, "Electrically small, ferrite-loaded loop antennas," *IRE Convention Record*, vol. 4, pt. 1, pp. 165–170, 1956.
# 6. R. DeVore and P. Bohley, "The electrically small, magnetically loaded, multiturn loop antenna," *IEEE Transactions on Antennas and Propagation*, vol. 25, no. 4, pp. 496–505, Jul. 1977.
# 7. H. A. Wheeler, "Fundamental Limitations of Small Antennas," *Proceedings of the IRE*, vol. 35, no. 12, pp. 1479–1484, Dec. 1947, doi: 10.1109/JRPROC.1947.226199 .
# 8. H. A. Wheeler, "Small Antennas," *IEEE Transactions on Antennas and Propagation*, vol. 23, no. 4, pp. 462–469, Jul. 1975, doi: 10.1109/TAP.1975.1141115 .
# 9. A. D. Yaghjian and S. R. Best, "Impedance, Bandwidth, and $Q$ of Antennas," *IEEE Transactions on Antennas and Propagation*, vol. 53, no. 4, pp. 1298–1324, Apr. 2005, doi: 10.1109/TAP.2005.844443 .
# 10. P. M. T. Ikonen, S. I. Maslovski, C. R. Simovski, and S. A. Tretyakov, "On Artificial Magnetodielectric Loading for Improving the Impedance Bandwidth Properties of Microstrip Patch Antennas," *IEEE Transactions on Antennas and Propagation*, vol. 54, no. 6, pp. 1654–1662, Jun. 2006, doi: 10.1109/TAP.2006.875912 .
# 11. J. L. Snoek, "Dispersion and absorption in magnetic ferrites at frequencies above one Mc/s," *Physica*, vol. 14, no. 4, pp. 207–217, May 1948, doi: 10.1016/0031-8914(48)90038-X .
# 12. H. A. Wheeler, "Transmission-Line Properties of Parallel Strips Separated by a Dielectric Sheet," *IEEE Transactions on Microwave Theory and Techniques*, vol. 13, no. 2, pp. 172–185, Mar. 1965, doi: 10.1109/TMTT.1965.1125962 .
