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
# Copyright (c) CERN, 2026.                   #
# ########################################### #

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eshipton/Wakis/blob/main/tutorials/05_guided_waves/05-11_kerr_microcomb/kerr_microcomb.ipynb)
#
# # Tutorial 5.11 · Kerr ($\chi^{(3)}$) Nonlinear Media in FIT and FIT-Informed Soliton Microcombs
# <h3 align="center">Sub-cycle Kerr time stepping in <code>SolverFIT3D</code>, cold-cavity dispersion &amp; mode-volume extraction, and Lugiato–Lefever dissipative Kerr solitons</h3>
#
# ---
#
# ### Abstract
#
# **Dissipative Kerr solitons (DKS) in high-$Q$ dielectric microresonators convert a continuous-wave (CW) laser pump into an ultrashort pulse train circulating around the ring and a coherent, phase-locked optical frequency comb with a $\mathrm{sech}^2$ spectral envelope [3–5]. A direct full-wave Finite Integration Technique (FIT) time-domain simulation of comb formation would require $10^5\text{–}10^6$ cavity round trips ($\sim 10^8\text{–}10^9$ Courant timesteps), which is orders of magnitude beyond explicit sub-cycle Maxwell solvers. In this tutorial we build an honest hybrid multiscale workflow using `wakis.nonlinear`: (1) full-wave 2D TM FIT simulations (`SolverFIT3D` + `harminv`) of a microring resonator ($R = 2.0\;\mu\mathrm{m}, w = 0.60\;\mu\mathrm{m}$) extract the azimuthal resonance family $\omega_\mu$, free spectral range $D_1/(2\pi)$, second-order integrated dispersion $D_{\mathrm{int}}(\mu) = \omega_\mu - \omega_0 - D_1\mu \approx \frac{1}{2}D_2\mu^2$, bus-coupled decay rates $(\kappa_0, \kappa_{\mathrm{ex}})$ across a $g = 160\;\mathrm{nm}$ gap (cf. [Tutorial 5.9](../05-09_ring_coupling/ring_coupling.ipynb)), and the effective nonlinear mode volume $V_{\mathrm{eff,NL}}$; (2) a short full-wave 2D FIT Kerr ring simulation (`KerrMaterial` with $\chi^{(3)} = 2.4\times 10^{-19}\;\mathrm{m^2/V^2}$) directly verifies the intracavity nonlinear resonance shift $\Delta f_{\mathrm{NL}}(U)$ against first-order perturbation theory; and (3) the FIT-measured parameters drive a split-step Fourier Lugiato–Lefever Equation (LLE) solver (`wakis.solve_lle`) across a pump-detuning scan from modulation instability (MI) and chaotic breathers onto the single-soliton step, validating the soliton existence window and $\mathrm{sech}^2$ comb bandwidth scaling $\Delta\mu_{\mathrm{FWHM}} \propto D_2^{-1/2}$.**
#
# ### Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. **Derive and apply** the sub-cycle third-order nonlinear constitutive relation $\mathbf{D} = \varepsilon_0 \varepsilon_\infty \mathbf{E} + \mathbf{P}_{\mathrm{L}} + \varepsilon_0 \chi^{(3)} |\mathbf{E}|^2 \mathbf{E}$ in `wakis.KerrMaterial` (per-edge Newton, fixed-point, and explicit ADE updates [1, 2]) and prove why a monochromatic real field $E(t) = \operatorname{Re}[\tilde{E}_0 e^{i\omega_0 t}]$ acquires the factor **$3/4$** in the intensity-dependent refractive index $n_2 = \frac{3\chi^{(3)}}{4 c \varepsilon_0 n_0^2}$ [6].
# 2. **Extract** the azimuthal whispering-gallery resonance frequencies $f_m$ ($m \in \{14,\dots,20\}$, pump mode $m_0 = 17$ at $f_0 \approx 214.5\;\mathrm{THz}$) and integrated dispersion $D_{\mathrm{int}}(\mu) = \omega_\mu - \omega_0 - D_1\mu$ from a 2D FIT cold-cavity microring run via azimuthal Fourier projection and `wakis.harminv` [7], and benchmark against the exact cylindrical Bessel/Neumann eigenvalue determinant.
# 3. **Contrast** normal geometric dispersion ($D_2 / (2\pi) = -23.01\;\mathrm{GHz} < 0$) in an uncompensated tight microring against anomalous dispersion ($D_2 / (2\pi) = +30.85\;\mathrm{GHz} > 0$) engineered via an artificial infrared Lorentz pole ($f_{\mathrm{IR}} = 80\;\mathrm{THz}$), and extract the intrinsic ($Q_0$) and external bus-coupling ($Q_{\mathrm{ex}}$) quality factors from a bus-coupled 2D FIT simulation.
# 4. **Verify** the Kerr-induced resonance frequency shift $\Delta f_{\mathrm{NL}}(U) = -f_0 \frac{c n_2 U}{n_0^2 V_{\mathrm{eff,NL}}}$ in a short 2D FIT nonlinear ring simulation against `wakis.effective_nonlinear_volume` and `wakis.kerr_cavity_frequency_shift`.
# 5. **Simulate** the FIT-informed Lugiato–Lefever Equation (`wakis.solve_lle`, `wakis.lle_cw_bistability`, `wakis.lle_soliton_analytic`) [3–5] across a laser pump-detuning scan ($\zeta_0 \in [-1.2, +5.2]$) to observe Turing rolls (primary MI), chaotic breathers, the soliton step, and a stationary single dissipative Kerr soliton whose $\mathrm{sech}^2$ comb spectrum scales as $\Delta\mu_{\mathrm{FWHM}} \propto \sqrt{\zeta_0 / D_2}$.
#
# ### Contents
#
# 1. [Theory & Derivations: Sub-Cycle Kerr FIT and the Lugiato–Lefever Equation](#1.-Theory-&-Derivations:-Sub-Cycle-Kerr-FIT-and-the-Lugiato–Lefever-Equation)
# 2. [Simulation Setup: 2D TM Kerr Microring & Bus Coupler](#2.-Simulation-Setup:-2D-TM-Kerr-Microring-&-Bus-Coupler)
# 3. [Part 1 · Cold-Cavity Whispering-Gallery Dispersion & Bus-Coupled $Q$ in 2D FIT](#3.-Part-1-·-Cold-Cavity-Whispering-Gallery-Dispersion-&-Bus-Coupled-$Q$-in-2D-FIT)
# 4. [Part 2 · Direct 2D FIT Verification of the Kerr Cavity Resonance Shift](#4.-Part-2-·-Direct-2D-FIT-Verification-of-the-Kerr-Cavity-Resonance-Shift)
# 5. [Part 3 · Resolution Convergence Study & Locally Refined Graded Mesh](#5.-Part-3-·-Resolution-Convergence-Study-&-Locally-Refined-Graded-Mesh)
# 6. [Part 4 · FIT-Informed Lugiato–Lefever Soliton Microcomb Dynamics](#6.-Part-4-·-FIT-Informed-Lugiato–Lefever-Soliton-Microcomb-Dynamics)
# 7. [Quantitative Summary](#7.-Quantitative-Summary)
# 8. [Limits & Failure Modes](#8.-Limits-&-Failure-Modes)
# 9. [Verified References](#9.-Verified-References)

# %% [markdown]
# ---
#
# ## 1. Theory & Derivations: Sub-Cycle Kerr FIT and the Lugiato–Lefever Equation
#
# ### 1.1 Sub-cycle constitutive update for $\chi^{(3)}$ Kerr + Raman + Lorentz media in FIT *(derived here)*
#
# On the staggered Finite Integration Technique (FIT) grid, Faraday's law advances the magnetic field $\mathbf{H}^{n+1/2}$ from $\mathbf{E}^n$, and Ampère's law advances the electric flux density $\mathbf{D}$ from $t = n\Delta t$ to $t = (n+1)\Delta t$:
#
# $$
# \frac{\mathbf{D}^{n+1} - \mathbf{D}^n}{\Delta t} = \nabla \times \mathbf{H}^{n+1/2} - \sigma \mathbf{E}^{n+1/2} - \mathbf{J}_{\mathrm{src}}^{n+1/2}.
# $$
#
# In a centrosymmetric dielectric with background high-frequency permittivity $\varepsilon_\infty$, linear dispersive Drude/Lorentz/Debye poles $\mathbf{P}_{\mathrm{L}} = \sum_p \mathbf{P}_p$, and third-order susceptibility $\chi^{(3)}$ [$\mathrm{m^2/V^2}$] split between an instantaneous electronic fraction $(1 - f_R)$ and a delayed vibrational Raman fraction $f_R \in [0, 1]$ [1, 2, 6]:
#
# $$
# \mathbf{D}(t) = \varepsilon_0 \varepsilon_\infty \mathbf{E}(t) + \mathbf{P}_{\mathrm{L}}(t) + \varepsilon_0 \chi^{(3)}\left[(1 - f_R)|\mathbf{E}(t)|^2 + f_R S(t)\right]\mathbf{E}(t),
# $$
#
# where the scalar Raman convolution state $S(t) = \int_0^\infty h_R(\tau)|\mathbf{E}(t-\tau)|^2\,d\tau$ obeys the auxiliary differential equation (ADE) [1, 2]
#
# $$
# \frac{d^2 S}{dt^2} + 2\Gamma_R \frac{dS}{dt} + \Omega_R^2 S = \Omega_R^2 |\mathbf{E}(t)|^2, \qquad \Omega_R = \sqrt{\tau_1^{-2} + \tau_2^{-2}}, \quad \Gamma_R = \tau_2^{-1}.
# $$
#
# Discretizing the Raman ADE with second-order central differences centered at $t = n\Delta t$ gives an explicit two-step recurrence for $S^{n+1}$ before the $\mathbf{E}^{n+1}$ inversion:
#
# $$
# S^{n+1} = \frac{\left(2 - \Omega_R^2 \Delta t^2\right) S^n - \left(1 - \Gamma_R \Delta t\right) S^{n-1} + \Omega_R^2 \Delta t^2 |\mathbf{E}^n|^2}{1 + \Gamma_R \Delta t}.
# $$
#
# After advancing the linear auxiliary polarizations $\mathbf{P}_p^{n+1}$ via the bilinear transform (`wakis.dispersive`), the remaining algebraic constitutive relation at $t = (n+1)\Delta t$ on each nonlinear edge with fill weight $w \in (0, 1]$ is
#
# $$
# \mathbf{D}_*^{n+1} \equiv \mathbf{D}^{n+1} - \mathbf{P}_{\mathrm{L}}^{n+1} = \varepsilon_0\left[\varepsilon_{\mathrm{eff}} + w f_R \chi^{(3)} S^{n+1} + w(1 - f_R)\chi^{(3)}|\mathbf{E}^{n+1}|^2\right]\mathbf{E}^{n+1}.
# $$
#
# Taking the norm of both sides reduces the vector update to a scalar cubic equation for the electric field magnitude $x = |\mathbf{E}^{n+1}| \ge 0$ in terms of the known trial displacement magnitude $y = |\mathbf{D}_*^{n+1}| / \varepsilon_0 \ge 0$:
#
# $$
# \boxed{f(x) \equiv \alpha x^3 + \beta x - y = 0, \qquad \alpha = w(1 - f_R)\chi^{(3)}, \quad \beta = \varepsilon_{\mathrm{eff}} + w f_R \chi^{(3)} S^{n+1}.}
# $$
#
# Whenever $\beta > 0$ and $\alpha \ge 0$ (focusing Kerr nonlinearity), $f'(x) = 3\alpha x^2 + \beta > 0$ for all $x \ge 0$, guaranteeing a **unique real non-negative root** that converges quadratically in 3–5 Newton–Raphson iterations (`method='newton'`):
#
# $$
# x^{(k+1)} = x^{(k)} - \frac{\alpha\left(x^{(k)}\right)^3 + \beta x^{(k)} - y}{3\alpha\left(x^{(k)}\right)^2 + \beta}, \qquad \mathbf{E}^{n+1} = \frac{\mathbf{D}_*^{n+1}}{\varepsilon_0\left(\beta + \alpha x^2\right)}.
# $$
#
# Notice that when $\chi^{(3)} = 0$, $\alpha = 0$ and the update reduces **bit-for-bit** to the linear division $\mathbf{E}^{n+1} = \mathbf{D}_*^{n+1}/(\varepsilon_0 \varepsilon_{\mathrm{eff}})$.
#
# ---
#
# ### 1.2 The real-field factor $3/4$ between $\chi^{(3)}$ and $n_2$, and third-harmonic generation *(derived here)*
#
# Under the Wakis harmonic phasor convention $E(z, t) = \operatorname{Re}[\tilde{E}_0 e^{i(\omega_0 t - k z)}] = \frac{1}{2}(\tilde{E}_0 e^{i\theta} + \tilde{E}_0^* e^{-i\theta})$ with $\theta = \omega_0 t - k z$, cubing the real instantaneous electric field yields both a fundamental Kerr response at $\omega_0$ and a third-harmonic response at $3\omega_0$:
#
# $$
# E^3(z, t) = |\tilde{E}_0|^3 \cos^3\theta = |\tilde{E}_0|^3\left(\frac{3}{4}\cos\theta + \frac{1}{4}\cos 3\theta\right) = \operatorname{Re}\!\left[\frac{3}{4}|\tilde{E}_0|^2 \tilde{E}_0 e^{i\theta} + \frac{1}{4}\tilde{E}_0^3 e^{i 3\theta}\right].
# $$
#
# Comparing the fundamental Fourier component $D(\omega_0) = \varepsilon_0\left(\varepsilon_r + \frac{3}{4}\chi^{(3)}|\tilde{E}_0|^2\right)\tilde{E}_0 \equiv \varepsilon_0 n_{\mathrm{eff}}^2 \tilde{E}_0$ with the first-order refractive index perturbation $n_{\mathrm{eff}}^2 = (n_0 + \Delta n_{\mathrm{NL}})^2 \approx n_0^2 + 2 n_0 \Delta n_{\mathrm{NL}}$ and the time-averaged optical intensity $I = \frac{1}{2} c \varepsilon_0 n_0 |\tilde{E}_0|^2$ gives the fundamental conversion identity (`wakis.chi3_to_n2`, `wakis.spm_phase_shift`) [6]:
#
# $$
# \boxed{\Delta n_{\mathrm{NL}} = \frac{3\chi^{(3)}}{8 n_0}|\tilde{E}_0|^2 = n_2 I, \qquad n_2 = \frac{3\chi^{(3)}}{4 c \varepsilon_0 n_0^2}\quad [\mathrm{m^2/W}], \qquad \Delta\Phi_{\mathrm{NL}} = n_2 k_0 I L.}
# $$
#
# Simultaneously, the $\frac{1}{4}\tilde{E}_0^3 e^{i 3\theta}$ term drives **third-harmonic generation (THG)** at $3\omega_0$. Integrating the undepleted-pump wave equation across a slab of thickness $L$ with phase mismatch $\Delta k = k(3\omega_0) - 3 k(\omega_0) = \frac{3\omega_0}{c}[n(3\omega_0) - n(\omega_0)]$ yields the Maker-fringe conversion efficiency (`wakis.thg_conversion_efficiency`) [6]:
#
# $$
# \eta_{\mathrm{THG}}(L) = \frac{I_{3\omega}(L)}{I_\omega(0)} = \frac{9 \omega_0^2 \left(\chi^{(3)}\right)^2 I_\omega^2 L^2}{16 \varepsilon_0^2 c^4 n_\omega^3 n_{3\omega}}\,\mathrm{sinc}^2\!\left(\frac{\Delta k L}{2}\right), \qquad \mathrm{sinc}(u) \equiv \frac{\sin u}{u}.
# $$
#
# ---
#
# ### 1.3 Whispering-gallery dispersion $D_{\mathrm{int}}(\mu)$, bus coupling, and Kerr cavity perturbation *(derived here)*
#
# Consider a 2D TM dielectric microring resonator of centerline radius $R$ and radial waveguide width $w$ ($R_{\mathrm{in}} = R - w/2$, $R_{\mathrm{out}} = R + w/2$) with out-of-plane electric field $E_z(r, \phi, t) = u_m(r)\cos(m\phi)\operatorname{Re}[e^{i\omega_m t}]$. Inside the ring ($R_{\mathrm{in}} \le r \le R_{\mathrm{out}}$) and cladding, the exact radial Helmholtz equation in cylindrical coordinates is Bessel's equation of azimuthal order $m$:
#
# $$
# \frac{d^2 u_m}{dr^2} + \frac{1}{r}\frac{du_m}{dr} + \left(n^2(\omega_m) k_0^2 - \frac{m^2}{r^2}\right) u_m(r) = 0, \qquad k_0 = \frac{\omega_m}{c}.
# $$
#
# Matching continuity of $E_z \propto u_m(r)$ and $H_\phi \propto du_m/dr$ at $r = R_{\mathrm{in}}$ and $r = R_{\mathrm{out}}$ between $J_m(k_1 r)$ in the inner cladding ($r < R_{\mathrm{in}}$), $B J_m(k_2 r) + C Y_m(k_2 r)$ in the core, and $D Y_m(k_1 r)$ in the radiation-trapped outer evanescent zone gives a closed-form $2\times 2$ determinant (`bessel_ring_mode`) for the cold-cavity eigenfrequencies $\omega_m = 2\pi f_m$.
#
# Expanding $\omega_\mu \equiv \omega_{m_0 + \mu}$ in a Taylor series around the pumped azimuthal order $m_0$ ($\mu = m - m_0 \in \mathbb{Z}$) defines the **integrated dispersion** $D_{\mathrm{int}}(\mu)$ [3, 5]:
#
# $$
# \boxed{\omega_\mu = \omega_0 + D_1 \mu + \frac{1}{2} D_2 \mu^2 + \frac{1}{6} D_3 \mu^3 + \cdots, \qquad D_{\mathrm{int}}(\mu) \equiv \omega_\mu - \omega_0 - D_1 \mu \approx \frac{1}{2} D_2 \mu^2,}
# $$
#
# where $D_1 / (2\pi) = \mathrm{FSR}$ is the free spectral range at $m_0$ and $D_2 = -\frac{c}{n_g} D_1^2 \beta_2$ is the second-order dispersion parameter:
# - **Uncompensated dispersionless core ($\varepsilon_r = \varepsilon_\infty = 4.00$):** Tight whispering-gallery geometric curvature pushes shorter wavelengths toward the outer rim ($r \to R_{\mathrm{out}}$), producing **normal dispersion** ($D_2 < 0$, $\beta_2 > 0$), which prevents bright soliton formation.
# - **Dispersion-engineered core ($\varepsilon_\infty = 4.35$ + infrared vibrational Lorentz pole at $f_{\mathrm{IR}} = 80\;\mathrm{THz}, \Delta\varepsilon = 1.50$):** The normal-dispersion tail of the reststrahlen/vibrational resonance contributes anomalous material group-velocity dispersion ($\beta_{2,\mathrm{mat}} < 0$) that overcompensates the geometric waveguide curvature around $m_0 = 17$ ($f_0 \approx 214.5\;\mathrm{THz}$), yielding **anomalous dispersion** ($D_2 > 0$).
#
# When a straight bus waveguide is placed at gap $g$ from the outer rim of the ring (cf. [Tutorial 5.9](../05-09_ring_coupling/ring_coupling.ipynb)), the total cavity linewidth $\kappa = \omega_0 / Q_L$ is the sum of intrinsic loss $\kappa_0 = \omega_0 / Q_0$ and external bus coupling $\kappa_{\mathrm{ex}} = \omega_0 / Q_{\mathrm{ex}}$:
#
# $$
# \frac{1}{Q_L} = \frac{1}{Q_0} + \frac{1}{Q_{\mathrm{ex}}}, \qquad \kappa = \kappa_0 + \kappa_{\mathrm{ex}}.
# $$
#
# For a standing or travelling cavity mode $\mathbf{E}(\mathbf{r}, t) = \operatorname{Re}[\tilde{\mathbf{E}}(\mathbf{r}) e^{i\omega_0 t}]$ in a Kerr medium, first-order cavity perturbation theory (`wakis.effective_nonlinear_volume`, `wakis.kerr_cavity_frequency_shift`) gives the nonlinear resonance shift in terms of the stored electromagnetic energy $U = \frac{1}{2}\varepsilon_0 \int \varepsilon_r(\mathbf{r})|\tilde{\mathbf{E}}(\mathbf{r})|^2\,dV$ and the **effective nonlinear mode volume** $V_{\mathrm{eff,NL}}$:
#
# $$
# \boxed{\frac{\Delta\omega_{\mathrm{NL}}}{\omega_0} = \frac{\Delta f_{\mathrm{NL}}}{f_0} = -\frac{3\chi^{(3)}\int_{\mathrm{NL}} |\tilde{\mathbf{E}}(\mathbf{r})|^4\,dV}{8\int \varepsilon_r(\mathbf{r})|\tilde{\mathbf{E}}(\mathbf{r})|^2\,dV} = -\frac{c n_2 U}{n_0^2 V_{\mathrm{eff,NL}}}, \qquad V_{\mathrm{eff,NL}} \equiv \frac{\left(\int \varepsilon_r(\mathbf{r})|\tilde{\mathbf{E}}(\mathbf{r})|^2\,dV\right)^2}{n_0^2 \int_{\mathrm{NL}} \varepsilon_r(\mathbf{r})|\tilde{\mathbf{E}}(\mathbf{r})|^4\,dV}.}
# $$
#
# ---
#
# ### 1.4 Honest Hybrid Link: FIT Parameters $\to$ Normalized Lugiato–Lefever Equation *(derived here)*
#
# Why use a hybrid FIT $\to$ Lugiato–Lefever workflow instead of stepping Maxwell's equations directly across a laser pump-detuning scan? In our compact 2D FIT benchmark ring ($R = 2.0\;\mu\mathrm{m}$) strongly overcoupled to the bus waveguide ($g = 160\;\mathrm{nm}$), the intrinsic radiation-limited quality factor is $Q_0 = 5.02\times 10^4$ and the external bus coupling is $Q_{\mathrm{ex}} = 540.0$, giving a loaded quality factor $Q_L = 534.2$ ($\kappa/(2\pi) = 401.06\;\mathrm{GHz}$, coupling efficiency $\eta = \kappa_{\mathrm{ex}}/\kappa = 0.989$). By contrast, larger experimental $\mathrm{Si}_3\mathrm{N}_4$ or silica microresonators ($R \sim 20\text{–}1000\;\mu\mathrm{m}$) have negligible bending loss and operate at $Q_L \sim 10^5\text{–}10^7$; because the parametric oscillation threshold scales as $P_{\mathrm{th}} \propto V_{\mathrm{eff,NL}} / Q_L^2$, a $Q_L = 534.2$ ring requires watt-scale pump power ($P_{\mathrm{th}} = 0.638\;\mathrm{W}$; choosing the normalized pump $f = 2.350$ inside the soliton-step regime corresponds to a bus input power $P_{\mathrm{in}} = f^2 P_{\mathrm{th}} = 3.521\;\mathrm{W}$) whereas a $Q_L \sim 10^6$ ring operates at milliwatt power, and simulating thousands of slow photon lifetimes $\tau_{\mathrm{ph}} = 1/\kappa$ in full-wave FIT would require $10^6\text{–}10^9$ Courant timesteps. Crucially, every coefficient of the mean-field **Lugiato–Lefever Equation (LLE)** [3–5] is a cold-cavity or single-mode Kerr property that 2D/3D FIT measures directly in a few thousand timesteps:
#
# 1. **Slow time & detuning:** $\tau = \frac{\kappa t}{2}$ (in units of twice the photon lifetime) and normalized pump detuning $\zeta_0 = \frac{2(\omega_0 - \omega_p)}{\kappa}$ ($\zeta_0 > 0$ on the red-detuned soliton side).
# 2. **Normalized dispersion:** $d_2 = \frac{2 D_2}{\kappa} = 0.1391$, derived directly from the FIT cold-cavity dispersion $D_2/(2\pi) = +27.89\;\mathrm{GHz}$ and the FIT bus-coupled linewidth $\kappa/(2\pi) = 401.06\;\mathrm{GHz}$.
# 3. **Normalized external pump & corresponding bus input power:** $f = \sqrt{\frac{8 g_0 \eta P_{\mathrm{in}}}{\hbar\omega_0 \kappa^2}} = \sqrt{\frac{P_{\mathrm{in}}}{P_{\mathrm{th}}}}$, where the single-photon Kerr shift $g_0/(2\pi) = \frac{\hbar\omega_0^2 c n_2}{2\pi n_0^2 V_{\mathrm{eff,NL}}} = 28.44\;\mathrm{kHz}$ (using the FIT travelling-wave mode volume $V_{\mathrm{eff,NL}} = 1.290\;\mu\mathrm{m}^3$) and FIT coupling efficiency $\eta = 0.9894$ set the parametric threshold power $P_{\mathrm{th}} = \frac{\hbar\omega_0 \kappa^2}{8 g_0 \eta} = 0.638\;\mathrm{W}$. We choose $f = 2.350$ to operate squarely inside the single-soliton step regime ($f > 1.241$, $\zeta_{\max} = (\pi^2/8)f^2 = 6.81$), from which the required bus input power $P_{\mathrm{in}} = f^2 P_{\mathrm{th}} = 3.521\;\mathrm{W}$ follows directly.
#
# In these dimensionless variables, the slowly varying intracavity envelope $\Psi(\phi, \tau)$ in the co-rotating azimuthal frame $\phi \in [-\pi, \pi)$ obeys [3, 4]
#
# $$
# \boxed{\frac{\partial \Psi(\phi, \tau)}{\partial \tau} = -\left(1 + i\zeta_0\right)\Psi + i\frac{d_2}{2}\frac{\partial^2 \Psi}{\partial \phi^2} + i|\Psi|^2\Psi + f.}
# $$
#
# `wakis.solve_lle` integrates this equation on a periodic $N_\mu$-point azimuthal grid using a symmetric Strang split-step Fourier method (SSFM) with exact analytical linear propagator in $\mu$-space and fourth-order Runge–Kutta integration of the Kerr + pump step in $\phi$-space.
#
# - **CW bistability (`wakis.lle_cw_bistability`):** Homogeneous solutions $\partial_\phi \Psi = 0$ satisfy the cubic equation $Y^3 - 2\zeta_0 Y^2 + (1 + \zeta_0^2)Y = f^2$ for intracavity power $Y = |\Psi_{\mathrm{CW}}|^2$, which is bistable (three real roots) for $f > (8/3\sqrt{3})^{1/2} \approx 1.2407$ and $\zeta_0 > \sqrt{3}$.
# - **Single dissipative Kerr soliton (`wakis.lle_soliton_analytic`):** On the red-detuned bistable step $\zeta_{\min} < \zeta_0 \le \zeta_{\max} = \frac{\pi^2}{8} f^2$ [3], the localized pulse sitting on the low-power CW background $\Psi_{\mathrm{bg}} \approx -i f / \zeta_0$ is approximated by
#
# $$
# \Psi(\phi) \approx \Psi_{\mathrm{bg}} + \sqrt{2\zeta_0}\,e^{i\phi_0}\,\mathrm{sech}\!\left(\sqrt{\frac{2\zeta_0}{d_2}}\,\phi\right), \qquad \Delta\phi_{\mathrm{FWHM}} = 2\operatorname{arccosh}(\sqrt{2})\sqrt{\frac{d_2}{2\zeta_0}}.
# $$
#
# Taking the Fourier transform $\Psi_\mu = \frac{1}{2\pi}\int_{-\pi}^{\pi} \Psi(\phi) e^{-i\mu\phi}\,d\phi$ yields the **$\mathrm{sech}^2$ microcomb power spectrum** ($\mu \ne 0$) and its 3 dB spectral FWHM bandwidth in comb lines:
#
# $$
# \boxed{|\Psi_\mu|^2 = \frac{d_2}{4}\,\mathrm{sech}^2\!\left(\frac{\pi\mu}{2}\sqrt{\frac{d_2}{2\zeta_0}}\right), \qquad \Delta\mu_{\mathrm{FWHM}} = \frac{4\operatorname{arccosh}(\sqrt{2})}{\pi}\sqrt{\frac{2\zeta_0}{d_2}} \propto \sqrt{\frac{\zeta_0}{D_2}}.}
# $$

# %%
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="IProgress not found")

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    import subprocess

    subprocess.run("apt-get -qq install -y xvfb > /dev/null", shell=True, check=False)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "wakis @ git+https://github.com/eshipton/Wakis.git@main",
        ],
        check=True,
    )

import matplotlib
import numpy as np

if not hasattr(sys, "ps1") and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from IPython.display import Image, Markdown, display
except ImportError:

    def display(*_args, **_kwargs):
        """No-op outside a notebook."""

    def Image(*_args, **_kwargs):
        """Placeholder outside a notebook."""

    def Markdown(*_args, **_kwargs):
        """Placeholder outside a notebook."""


HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

from _helpers import (  # noqa: E402
    DEFAULT_BUS_WIDTH_M,
    DEFAULT_CHI3_RING,
    DEFAULT_DOMAIN_HALF_M,
    DEFAULT_EPS_CLAD,
    DEFAULT_EPS_INF_ANOM,
    DEFAULT_EPS_INF_NORM,
    DEFAULT_GAP_M,
    DEFAULT_IR_DELTA_EPS,
    DEFAULT_IR_FREQ_HZ,
    DEFAULT_M0,
    DEFAULT_M_LIST,
    DEFAULT_RING_RADIUS_M,
    DEFAULT_RING_WIDTH_M,
    DEFAULT_SLAB_DZ_M,
    augment_convergence_dict,
    bessel_ring_dispersion,
    bessel_ring_mode,
    build_microring_graded_axes,
    generate_all_figures,
    make_ring_material,
    run_convergence_study,
    run_fit_bus_coupled_ring,
    run_fit_cold_ring,
    run_fit_kerr_ring_shift,
    run_lle_detuning_scan,
    save_microcomb_setup_figures,
)

TASKS = (
    (
        0,
        "cold_dispersion",
        "2D FIT cold-cavity microring WGM resonances (m=14..20) & integrated dispersion D_int(mu) (anomalous vs normal)",
    ),
    (
        1,
        "bus_coupling",
        "2D FIT bus-coupled microring quality factors (Q_0, Q_ex, Q_L) across the 160 nm coupling gap",
    ),
    (
        2,
        "kerr_shift",
        "Short 2D FIT Kerr microring simulation (chi^(3) > 0) verifying Delta f_NL(U) vs V_eff,NL perturbation theory",
    ),
    (
        3,
        "convergence",
        "3-level uniform resolution convergence study (30, 25, 20 nm, >= 20 cells/lambda) + locally refined graded mesh + Richardson",
    ),
    (
        4,
        "lle_microcomb",
        "FIT-informed 256-mode Lugiato-Lefever split-step Fourier detuning scan (MI -> breather -> single DKS) & D2^(-1/2) scaling",
    ),
)


def _strip_arrays_for_json(d: dict) -> dict:
    """Return a JSON-serializable copy of `d`, omitting large 2D NumPy arrays."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            if v.ndim == 1 and v.size <= 512:
                out[k] = v.tolist()
            continue
        out[k] = v
    return out


def compute_all_study_data(
    quick: bool = True,
    out_dir: Path = DATA_DIR,
    selected_task: int | None = None,
) -> dict:
    """
    Execute all (or a single selected) tutorial tasks sequentially, preflighting
    every 2D FIT run and saving structured `.json` and `.npz` artifacts into `out_dir`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    dx_main = 30.0e-9 if quick else 20.0e-9
    dx_bus = 30.0e-9 if quick else 25.0e-9
    dx_kerr = 35.0e-9 if quick else 30.0e-9
    dx_levels = (35.0e-9, 30.0e-9, 25.0e-9) if quick else (30.0e-9, 25.0e-9, 20.0e-9)
    n_steps_cold = 3200 if quick else None
    n_steps_bus = 3200 if quick else 3600
    n_steps_kerr = 2400 if quick else 2800
    n_steps_conv = 2800 if quick else None

    cold_anom = None
    cold_norm = None
    bus_res = None
    kerr_res = None
    conv_res = None
    lle_res = None

    # Task 0: Cold-cavity dispersion (anomalous D2 > 0 and normal D2 < 0)
    if selected_task in (None, 0):
        cold_anom = run_fit_cold_ring(
            dx_m=dx_main, anomalous=True, graded=False, n_steps=n_steps_cold
        )
        cold_norm = run_fit_cold_ring(
            dx_m=dx_main, anomalous=False, graded=False, n_steps=n_steps_cold
        )
        t0_json = {
            "anomalous": _strip_arrays_for_json(cold_anom),
            "normal": _strip_arrays_for_json(cold_norm),
        }
        (out_dir / "task_0_cold_dispersion.json").write_text(
            json.dumps(t0_json, indent=2)
        )
        np.savez_compressed(
            out_dir / "task_0_cold_dispersion.npz",
            x_e=cold_anom["x_e"],
            y_e=cold_anom["y_e"],
            fill_ring=cold_anom["fill_ring"],
            snap_ez=cold_anom["snap_ez"],
            mode_m0_profile=cold_anom["mode_m0_profile"],
            radial_m0_profile=cold_anom["radial_m0_profile"],
            f_fit_anom_hz=np.array(cold_anom["f_fit_hz"]),
            f_bessel_anom_hz=np.array(cold_anom["f_bessel_hz"]),
            D_int_fit_anom_hz=np.array(cold_anom["D_int_fit_hz"]),
            D_int_bessel_anom_hz=np.array(cold_anom["D_int_bessel_hz"]),
            D_int_fit_norm_hz=np.array(cold_norm["D_int_fit_hz"]),
            D_int_bessel_norm_hz=np.array(cold_norm["D_int_bessel_hz"]),
        )

    # Task 1: Bus-coupled quality factors Q_0, Q_ex, Q_L
    if selected_task in (None, 1):
        bus_res = run_fit_bus_coupled_ring(
            dx_m=dx_bus, gap_m=DEFAULT_GAP_M, n_steps=n_steps_bus
        )
        (out_dir / "task_1_bus_coupling.json").write_text(
            json.dumps(_strip_arrays_for_json(bus_res), indent=2)
        )
        np.savez_compressed(
            out_dir / "task_1_bus_coupling.npz",
            x_e=bus_res["x_e"],
            y_e=bus_res["y_e"],
            fill_bus_tot=bus_res["fill_bus_tot"],
            snap_bus_ez=bus_res["snap_bus_ez"],
        )

    # Task 2: Short 2D FIT Kerr ring resonance shift vs V_eff,NL perturbation theory
    if selected_task in (None, 2):
        kerr_res = run_fit_kerr_ring_shift(
            dx_m=dx_kerr,
            chi3=DEFAULT_CHI3_RING,
            n_steps=n_steps_kerr,
        )
        (out_dir / "task_2_kerr_shift.json").write_text(
            json.dumps(_strip_arrays_for_json(kerr_res), indent=2)
        )
        np.savez_compressed(
            out_dir / "task_2_kerr_shift.npz",
            E_peaks_v_m=np.array(kerr_res["E_peaks_v_m"]),
            stored_energy_j=np.array(kerr_res["stored_energy_j"]),
            df_fit_hz=np.array(kerr_res["df_fit_hz"]),
            df_analytic_hz=np.array(kerr_res["df_analytic_hz"]),
        )

    # Task 3: 3-level resolution convergence study + locally refined graded mesh + Richardson
    if selected_task in (None, 3):
        conv_res = run_convergence_study(
            dx_levels_m=dx_levels,
            n_steps=n_steps_conv,
        )
        (out_dir / "task_3_convergence.json").write_text(
            json.dumps(_strip_arrays_for_json(conv_res), indent=2)
        )
        np.savez_compressed(
            out_dir / "task_3_convergence.npz",
            dx_nm=np.array(conv_res["dx_nm"]),
            f0_hz=np.array(conv_res["f0_hz"]),
            D1_hz=np.array(conv_res["D1_hz"]),
            D2_hz=np.array(conv_res["D2_hz"]),
            f0_rel_err_vs_bessel=np.array(conv_res["f0_rel_err_vs_bessel"]),
            D1_rel_err_vs_bessel=np.array(conv_res["D1_rel_err_vs_bessel"]),
            D2_rel_err_vs_bessel=np.array(conv_res["D2_rel_err_vs_bessel"]),
        )

    # Task 4: FIT-informed Lugiato-Lefever split-step Fourier detuning scan & soliton spectrum
    if selected_task in (None, 4):
        if cold_anom is None and (out_dir / "task_0_cold_dispersion.json").exists():
            t0_loaded = json.loads((out_dir / "task_0_cold_dispersion.json").read_text())
            cold_anom = dict(t0_loaded["anomalous"])
        if bus_res is None and (out_dir / "task_1_bus_coupling.json").exists():
            bus_res = json.loads((out_dir / "task_1_bus_coupling.json").read_text())

        d2_use = float(cold_anom["D2_fit_hz"]) if cold_anom is not None else 27.89e9
        f0_use = float(cold_anom["f0_fit_hz"]) if cold_anom is not None else 214.34e12
        n0_use = float(cold_anom["n_core_f0"]) if cold_anom is not None else 2.0265
        veff_use = float(cold_anom["V_eff_travelling_m3"]) if cold_anom is not None else 1.2899e-18
        kappa_use = (
            float(bus_res["kappa_tot_rad_s"] / (2.0 * np.pi))
            if bus_res is not None
            else 401.06e9
        )
        eta_use = float(bus_res["eta_coupling"]) if bus_res is not None else 0.9894

        # Normalized pump f = 2.350 is chosen inside the single-soliton step regime
        # (f > 1.241); bus input power P_in = f^2 * P_th = 3.521 W follows from FIT P_th.
        lle_res = run_lle_detuning_scan(
            d2_fit_hz=d2_use,
            kappa_tot_hz=kappa_use,
            f_pump=2.350,
            n_modes=256,
            f0_fit_hz=f0_use,
            n0_core=n0_use,
            v_eff_trav_m3=veff_use,
            chi3_m2_v2=DEFAULT_CHI3_RING,
            eta_coupling=eta_use,
        )
        lle_json = {
            k: v
            for k, v in lle_res.items()
            if k not in ("scan_snapshots_abs2", "scan_zeta_snaps")
        }
        (out_dir / "task_4_lle_microcomb.json").write_text(
            json.dumps(lle_json, indent=2)
        )
        np.savez_compressed(
            out_dir / "task_4_lle_microcomb.npz",
            phi=np.array(lle_res["phi"]),
            mu=np.array(lle_res["mu"]),
            psi_mi_abs2=np.array(lle_res["psi_mi_abs2"]),
            psi_breather_abs2=np.array(lle_res["psi_breather_abs2"]),
            psi_sol_abs2=np.array(lle_res["psi_sol_abs2"]),
            psi_sol_analytic_abs2=np.array(lle_res["psi_sol_analytic_abs2"]),
            spectrum_sol_sim=np.array(lle_res["spectrum_sol_sim"]),
            spectrum_sol_an=np.array(lle_res["spectrum_sol_an"]),
            scan_snapshots_abs2=np.array(lle_res["scan_snapshots_abs2"]),
            scan_zeta_snaps=np.array(lle_res["scan_zeta_snaps"]),
        )

    all_tasks_exist = all(
        (out_dir / fname).exists()
        for fname in (
            "task_0_cold_dispersion.json",
            "task_1_bus_coupling.json",
            "task_2_kerr_shift.json",
            "task_3_convergence.json",
            "task_4_lle_microcomb.json",
        )
    )
    if selected_task is None or all_tasks_exist:
        if cold_anom is None or cold_norm is None:
            t0_loaded = json.loads((out_dir / "task_0_cold_dispersion.json").read_text())
            cold_anom = dict(t0_loaded["anomalous"])
            cold_norm = dict(t0_loaded["normal"])
        if bus_res is None:
            bus_res = json.loads((out_dir / "task_1_bus_coupling.json").read_text())
        if kerr_res is None:
            kerr_res = json.loads((out_dir / "task_2_kerr_shift.json").read_text())
        if conv_res is None:
            conv_res = augment_convergence_dict(
                json.loads((out_dir / "task_3_convergence.json").read_text())
            )
            (out_dir / "task_3_convergence.json").write_text(
                json.dumps(_strip_arrays_for_json(conv_res), indent=2)
            )
        else:
            conv_res = augment_convergence_dict(conv_res)
        if lle_res is None:
            lle_res = json.loads((out_dir / "task_4_lle_microcomb.json").read_text())
        wall_sum = float(
            cold_anom.get("wall_time_s", 0.0)
            + cold_norm.get("wall_time_s", 0.0)
            + bus_res.get("wall_time_s", 0.0)
            + kerr_res.get("wall_time_s", 0.0)
            + conv_res.get("total_wall_time_s", conv_res.get("wall_time_s", 0.0))
            + lle_res.get("wall_time_s", 0.0)
        )
        summary = {
            "quick": bool(quick),
            "ring_radius_um": DEFAULT_RING_RADIUS_M * 1e6,
            "ring_width_um": DEFAULT_RING_WIDTH_M * 1e6,
            "bus_gap_nm": DEFAULT_GAP_M * 1e9,
            "chi3_m2_v2": DEFAULT_CHI3_RING,
            "m0": DEFAULT_M0,
            "f0_fit_thz": cold_anom["f0_fit_hz"] * 1e-12,
            "f0_bessel_thz": cold_anom["f0_bessel_hz"] * 1e-12,
            "f0_rel_error": abs(cold_anom["f0_fit_hz"] - cold_anom["f0_bessel_hz"])
            / cold_anom["f0_bessel_hz"],
            "D1_fit_thz": cold_anom["D1_fit_hz"] * 1e-12,
            "D1_bessel_thz": cold_anom["D1_bessel_hz"] * 1e-12,
            "D1_rel_error": cold_anom["D1_rel_error"],
            "D2_anom_fit_ghz": cold_anom["D2_fit_hz"] * 1e-9,
            "D2_anom_bessel_ghz": cold_anom["D2_bessel_hz"] * 1e-9,
            "D2_anom_rel_error": cold_anom["D2_rel_error"],
            "D2_norm_fit_ghz": cold_norm["D2_fit_hz"] * 1e-9,
            "D2_norm_bessel_ghz": cold_norm["D2_bessel_hz"] * 1e-9,
            "D2_norm_rel_error": cold_norm["D2_rel_error"],
            "Delta_D2_pole_fit_ghz": (cold_anom["D2_fit_hz"] - cold_norm["D2_fit_hz"]) * 1e-9,
            "Delta_D2_pole_bessel_ghz": (
                cold_anom["D2_bessel_hz"] - cold_norm["D2_bessel_hz"]
            )
            * 1e-9,
            "Delta_D2_pole_rel_error": abs(
                (cold_anom["D2_fit_hz"] - cold_norm["D2_fit_hz"])
                - (cold_anom["D2_bessel_hz"] - cold_norm["D2_bessel_hz"])
            )
            / abs(cold_anom["D2_bessel_hz"] - cold_norm["D2_bessel_hz"]),
            "Q_0": bus_res["Q_0"],
            "Q_ex": bus_res["Q_ex"],
            "Q_loaded": bus_res["Q_loaded"],
            "kappa_tot_ghz": bus_res["kappa_tot_ghz"],
            "eta_coupling": bus_res["eta_coupling"],
            "V_eff_standing_um3": kerr_res["V_eff_nl_um3"],
            "V_eff_travelling_um3": cold_anom["V_eff_travelling_m3"] * 1e18,
            "kerr_shift_max_fit_ghz": kerr_res["df_fit_hz"][-1] * 1e-9,
            "kerr_shift_max_analytic_ghz": kerr_res["df_analytic_hz"][-1] * 1e-9,
            "kerr_shift_max_rel_error": kerr_res["max_rel_error"],
            "conv_fine_ppw": conv_res["ppw_core"][-1],
            "conv_fine_gap_cells": conv_res["cells_across_gap"][-1],
            "conv_f0_observed_order": conv_res["f0_observed_order"],
            "conv_f0_richardson_thz": conv_res["f0_richardson_hz"] * 1e-12,
            "conv_f0_richardson_rel_error": abs(
                conv_res["f0_richardson_hz"] - conv_res["f0_bessel_hz"]
            )
            / conv_res["f0_bessel_hz"],
            "conv_D1_observed_order": conv_res["D1_observed_order"],
            "conv_D1_richardson_thz": conv_res["D1_richardson_hz"] * 1e-12,
            "conv_D1_richardson_rel_error": abs(
                conv_res["D1_richardson_hz"] - conv_res["D1_bessel_hz"]
            )
            / conv_res["D1_bessel_hz"],
            "conv_D2_observed_order": conv_res["D2_observed_order"],
            "conv_D2_richardson_ghz": conv_res["D2_richardson_hz"] * 1e-9,
            "conv_D2_richardson_rel_error": conv_res["D2_richardson_rel_err"],
            "conv_D2_richardson_obs_ghz": conv_res["D2_richardson_obs_hz"] * 1e-9,
            "conv_D2_richardson_obs_rel_error": conv_res["D2_richardson_obs_rel_err"],
            "conv_D2_richardson_band_ghz": [
                x * 1e-9 for x in conv_res["D2_richardson_band_hz"]
            ],
            "conv_D2_richardson_band_rel_error": conv_res["D2_richardson_band_rel_err"],
            "conv_graded_D2_ghz": conv_res["graded_D2_hz"] * 1e-9,
            "conv_graded_D2_rel_error": conv_res["graded_D2_rel_err"],
            "conv_graded_cell_reduction_pct": conv_res["graded_cell_reduction_pct"],
            "lle_d2_norm": lle_res["d2_norm"],
            "lle_f_pump": lle_res["f_pump"],
            "lle_p_in_w": lle_res["p_in_w"],
            "lle_p_thresh_w": lle_res["p_thresh_w"],
            "lle_g0_single_photon_hz": lle_res["g0_single_photon_hz"],
            "lle_zeta_max_analytic": lle_res["zeta_max_analytic"],
            "lle_fwhm_mu_sim": lle_res["fwhm_mu_sim"],
            "lle_fwhm_mu_analytic": lle_res["fwhm_mu_an"],
            "lle_fwhm_mu_rel_error": lle_res["fwhm_mu_rel_error"],
            "lle_fwhm_phi_sim_rad": lle_res["fwhm_phi_sim"],
            "lle_fwhm_phi_analytic_rad": lle_res["fwhm_phi_an"],
            "lle_fwhm_phi_rel_error": lle_res["fwhm_phi_rel_error"],
            "lle_dispersion_scaling_exponent": lle_res["dispersion_scaling_exponent"],
            "lle_dispersion_exponent_rel_error": lle_res["dispersion_exponent_rel_error"],
            "total_wall_time_s": wall_sum,
        }
        (out_dir / "microcomb_summary.json").write_text(json.dumps(summary, indent=2))
        return {
            "cold_anom": cold_anom,
            "cold_norm": cold_norm,
            "bus_res": bus_res,
            "kerr_res": kerr_res,
            "conv_res": conv_res,
            "lle_res": lle_res,
            "summary": summary,
        }
    return {}


def render_figures_from_disk(data_dir: Path = DATA_DIR, fig_dir: Path = FIG_DIR) -> list[Path]:
    """Load saved `.json` and `.npz` task outputs from `data_dir` and render all figures."""
    setup_info = save_microcomb_setup_figures(fig_dir)
    t0_j = json.loads((data_dir / "task_0_cold_dispersion.json").read_text())
    t1_j = json.loads((data_dir / "task_1_bus_coupling.json").read_text())
    t2_j = json.loads((data_dir / "task_2_kerr_shift.json").read_text())
    t3_j = json.loads((data_dir / "task_3_convergence.json").read_text())
    t4_j = json.loads((data_dir / "task_4_lle_microcomb.json").read_text())

    with np.load(data_dir / "task_0_cold_dispersion.npz") as z0:
        cold_anom = dict(t0_j["anomalous"])
        cold_anom["x_e"] = z0["x_e"]
        cold_anom["y_e"] = z0["y_e"]
        cold_anom["fill_ring"] = z0["fill_ring"]
        cold_anom["snap_ez"] = z0["snap_ez"]
        cold_anom["mode_m0_profile"] = z0["mode_m0_profile"]
        cold_anom["radial_m0_profile"] = z0["radial_m0_profile"]
        cold_norm = dict(t0_j["normal"])

    with np.load(data_dir / "task_1_bus_coupling.npz") as z1:
        bus_res = dict(t1_j)
        bus_res["x_e"] = z1["x_e"]
        bus_res["y_e"] = z1["y_e"]
        bus_res["fill_bus_tot"] = z1["fill_bus_tot"]
        bus_res["snap_bus_ez"] = z1["snap_bus_ez"]

    with np.load(data_dir / "task_4_lle_microcomb.npz") as z4:
        lle_res = dict(t4_j)
        lle_res["scan_snapshots_abs2"] = z4["scan_snapshots_abs2"]
        lle_res["scan_zeta_snaps"] = z4["scan_zeta_snaps"]

    paths = [setup_info["geometry_figure"], setup_info["mesh_figure"]]
    paths.extend(
        generate_all_figures(
            cold_anom=cold_anom,
            cold_norm=cold_norm,
            bus_res=bus_res,
            kerr_res=t2_j,
            conv_res=t3_j,
            lle_res=lle_res,
            fig_dir=fig_dir,
        )
    )
    return paths


# %% [markdown]
# ---
#
# ## 2. Simulation Setup: 2D TM Kerr Microring & Bus Coupler
#
# Figures [`00a_setup_geometry.png`](figures/00a_setup_geometry.png) and [`00b_setup_mesh_and_excitation.png`](figures/00b_setup_mesh_and_excitation.png) (generated via `wakis.schematic.save_setup_figures`) show the 2D TM microring resonator ($R = 2.00\;\mu\mathrm{m}$, width $w = 0.60\;\mu\mathrm{m}$, inner radius $R_{\mathrm{in}} = 1.70\;\mu\mathrm{m}$, outer radius $R_{\mathrm{out}} = 2.30\;\mu\mathrm{m}$) coupled across a $g = 160\;\mathrm{nm}$ gap to a top bus waveguide of width $w_{\mathrm{bus}} = 0.28\;\mu\mathrm{m}$. The domain is terminated by 10-cell CPML absorbers on $\pm x$ and $\pm y$ and periodic boundary conditions along $z$, and discretized with a locally refined non-uniform Cartesian mesh (`build_microring_graded_axes`) that places $\Delta_{\min} = 20.0\;\mathrm{nm}$ ($34.5\;\mathrm{cells}/\lambda_{\mathrm{core}}$ in the bus coupler, $\ge 23\;\mathrm{cells}/\lambda_{\mathrm{core}}$ around the entire $360^\circ$ ring core, and $8.0\;\mathrm{cells}$ across the $160\;\mathrm{nm}$ bus–ring gap) while grading smoothly to $35\;\mathrm{nm}$ in the cladding.
#
# ![2D TM Kerr Microring & Bus Coupler Setup](figures/00a_setup_geometry.png)
# ![Locally Refined Graded Mesh & Pump Spectrum](figures/00b_setup_mesh_and_excitation.png)
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 5.60 µm × 5.60 µm × 200 nm (z ∈ [-100 nm, 100 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 199 × 204 × 2 (81192 cells) |
# | Discretization | Δx & resolution | Δ_min = 22 nm, 30.4 cells/λ (in n_max = 2.09), 7.3 cells across feature (160 nm) |
# | Boundary | -x face | CPML (10 cells, 341.6 nm) |
# | Boundary | +x face | CPML (10 cells, 341.6 nm) |
# | Boundary | -y face | CPML (10 cells, 341.6 nm) |
# | Boundary | +y face | CPML (10 cells, 243 nm) |
# | Boundary | -z face | Periodic |
# | Boundary | +z face | Periodic |
# | Material | Background | Vacuum, ε = 1.00 (isotropic) |
# | Material | Region 1: Kerr Bus Waveguide | ε_r = 4.35, μ_r = 1.00, model = KerrMaterial, x ∈ [-2.80 µm, 2.80 µm] (d = 5.60 µm) |
# | Material | Region 2: Kerr Microring Core | ε_r = 4.35, μ_r = 1.00, model = KerrMaterial, x ∈ [-2.30 µm, 2.30 µm] (d = 4.60 µm) |
# | Material | Region 3: Inner Air Cladding | ε_r = 1.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.70 µm, 1.70 µm] (d = 3.40 µm) |
# | Source | 1 · CurrentSource | pos: x = -1.80 µm, pol: J_z (E_z), pulse: f₀ = 214.5 THz, Δf = 35.00 THz |
# | Monitor | 2 · DFTMonitor | Through Port (DFTMonitor), pos: x = 1.80 µm, freqs: 41 freqs in [170.0, 250.0] THz |
# | Monitor | 3 · ResonanceMonitor | Ring WGM Probe (ResonanceMonitor), pos: x = 0 m, freqs: time-domain |
# | Variation | Cold-cavity dispersion extraction (task_0) | Azimuthal orders m ∈ {14..20} (m₀ = 17, f₀ = 214.5 THz), anomalous (D₂ > 0) vs normal (D₂ < 0) |
# | Variation | Bus-coupled Q extraction & Kerr shift (task_1, task_2) | 160 nm bus–ring gap (Q₀, Q_ex, κ) and short 2D FIT Kerr ring resonance shift vs V_eff,NL |
# | Variation | 3-level convergence + graded mesh (task_3) | Uniform Δx ∈ {30, 25, 20} nm (≥ 20 cells/λ_core) + locally refined graded mesh & Richardson extrapolation |
# | Variation | FIT-informed Lugiato–Lefever microcomb (task_4) | 256-mode split-step Fourier scan (MI → chaotic breather → single DKS soliton step) + sech² spectrum |

# %% [markdown]
# ---
#
# ## 3. Part 1 · Cold-Cavity Whispering-Gallery Dispersion & Bus-Coupled $Q$ in 2D FIT
#
# In Task 0 (`task_0_cold_dispersion`) and Task 1 (`task_1_bus_coupling`), we discretize the ring at $\Delta x = 20\;\mathrm{nm}$ ($34.5\;\mathrm{cells}/\lambda_{\mathrm{core}}$, $8.0\;\mathrm{cells}$ across the $160\;\mathrm{nm}$ gap) using $16\times 16$ dual-cell subpixel permittivity averaging along the curved inner and outer ring walls, excite the fundamental radial $\mathrm{TM}_{m,0}$ whispering-gallery mode family $m \in \{14, 15, 16, 17, 18, 19, 20\}$ centered on $m_0 = 17$, project the 2D core field onto the analytical radial-azimuthal mode profiles $u_m(r)\cos(m\phi)$, and extract the complex eigenfrequencies over a $150\;\mathrm{fs}$ ring-down via `wakis.harminv`:
#
# - **Whispering-gallery resonance family (Figure 1):** Across all seven azimuthal orders $\mu = m - m_0 \in [-3, +3]$, the 2D FIT resonance frequencies match the cylindrical Bessel eigenvalue solution within **$0.081\%$** at $\Delta x = 20\;\mathrm{nm}$ ($f_0 = 214.3395\;\mathrm{THz}$ vs $214.5126\;\mathrm{THz}$ analytical; Richardson-extrapolated $f_0 = 214.5107\;\mathrm{THz}$, **$0.0009\%$** error) and $\mathrm{FSR} = D_1/(2\pi) = 10.3187\;\mathrm{THz}$ vs $10.3488\;\mathrm{THz}$ analytical ($0.291\%$ error at $20\;\mathrm{nm}$; Richardson-extrapolated $10.3483\;\mathrm{THz}$, **$0.0053\%$** error).
# - **Anomalous vs normal integrated dispersion $D_{\mathrm{int}}(\mu)$ and $D_2$ error band (Figure 2a):** Because $D_2$ is a second difference of $\omega_m$ with magnitude $\sim 1.3\times 10^{-4}\omega_0$, its finite-grid error is amplified relative to $f_0$ and $D_1$. In the uncompensated dispersionless ring ($\varepsilon_\infty = 4.00$), the 2D FIT integrated dispersion parabola curves downward with **normal dispersion** $D_2/(2\pi) = -26.04\;\mathrm{GHz}$ at $\Delta x = 20\;\mathrm{nm}$ vs $-23.01\;\mathrm{GHz}$ continuum Bessel theory (**$13.17\%$** relative error). Adding the artificial infrared Lorentz pole ($f_{\mathrm{IR}} = 80\;\mathrm{THz}, \Delta\varepsilon = 1.50$) induces a positive GVD shift $\Delta D_2/(2\pi) = +53.93\;\mathrm{GHz}$ (vs $+53.86\;\mathrm{GHz}$ Bessel theory, **$0.13\%$** relative error), flipping the curvature upward into **anomalous dispersion** $D_2/(2\pi) = +27.89\;\mathrm{GHz}$ at $\Delta x = 20\;\mathrm{nm}$ vs $+30.85\;\mathrm{GHz}$ Bessel theory (**$9.59\%$** relative error; across $\Delta x \in \{30, 25, 20\}\;\mathrm{nm}$ the anomalous $D_2$ error band is $22.44\% \to 14.32\% \to 9.59\%$, $14.22\%$ on the locally refined graded mesh, and **$1.19\%\text{–}5.92\%$** across the Richardson extrapolation band $[+29.03, +30.49]\;\mathrm{GHz}$ spanned by the observed 3-grid order $p_{\mathrm{obs}} = 3.71$ and the nominal second-order $p = 2$).
# - **Bus-coupled quality factors (Figure 1b):** Introducing the top bus waveguide across the $g = 160\;\mathrm{nm}$ gap ($6.4\;\mathrm{cells}$ across the gap at $25\;\mathrm{nm}$) lowers the quality factor of mode $m_0 = 17$ from the intrinsic radiation-limited value $Q_0 = 50239.4$ to $Q_L = 534.2$, yielding the external coupling quality factor $Q_{\mathrm{ex}} = (Q_L^{-1} - Q_0^{-1})^{-1} = 540.0$, total cavity decay rate $\kappa/(2\pi) = f_0 / Q_L = 401.06\;\mathrm{GHz}$, and overcoupled bus efficiency $\eta = \kappa_{\mathrm{ex}}/\kappa = 0.989$.

# %%
if ("ipykernel" in sys.modules or hasattr(sys, "ps1")) and not (
    DATA_DIR / "microcomb_summary.json"
).exists():
    compute_all_study_data(quick=True, out_dir=DATA_DIR)
    render_figures_from_disk(data_dir=DATA_DIR, fig_dir=FIG_DIR)

if (DATA_DIR / "microcomb_summary.json").exists():
    summary_data = json.loads((DATA_DIR / "microcomb_summary.json").read_text())
    t0_data = json.loads((DATA_DIR / "task_0_cold_dispersion.json").read_text())
    t1_data = json.loads((DATA_DIR / "task_1_bus_coupling.json").read_text())

    if (FIG_DIR / "01_microring_wgm_fields_and_resonances.png").exists():
        display(Image(filename=str(FIG_DIR / "01_microring_wgm_fields_and_resonances.png"), width=980))

    print(
        f"Part 1 Cold-Cavity & Bus-Coupled Microring Results (m0 = {summary_data['m0']}):\n"
        f"  Pump resonance f0         : FIT = {summary_data['f0_fit_thz']:.4f} THz vs Bessel = {summary_data['f0_bessel_thz']:.4f} THz "
        f"(rel. error = {summary_data['f0_rel_error']*100:.3f}%)\n"
        f"  Free spectral range D1/2pi: FIT = {summary_data['D1_fit_thz']:.4f} THz vs Bessel = {summary_data['D1_bessel_thz']:.4f} THz "
        f"(rel. error = {summary_data['D1_rel_error']*100:.3f}%)\n"
        f"  Anomalous D2/2pi (IR pole): FIT = {summary_data['D2_anom_fit_ghz']:+.2f} GHz vs Bessel = {summary_data['D2_anom_bessel_ghz']:+.2f} GHz "
        f"(rel. error = {summary_data['D2_anom_rel_error']*100:.2f}%)\n"
        f"  Normal D2/2pi (no pole)   : FIT = {summary_data['D2_norm_fit_ghz']:+.2f} GHz vs Bessel = {summary_data['D2_norm_bessel_ghz']:+.2f} GHz "
        f"(rel. error = {summary_data['D2_norm_rel_error']*100:.2f}%)\n"
        f"  Bus coupling (g = 160 nm) : Q_0 = {summary_data['Q_0']:.1f}, Q_ex = {summary_data['Q_ex']:.1f}, "
        f"Q_L = {summary_data['Q_loaded']:.1f}, eta = {summary_data['eta_coupling']:.3f}"
    )

# %% [markdown]
# ---
#
# ## 4. Part 2 · Direct 2D FIT Verification of the Kerr Cavity Resonance Shift
#
# In Task 2 (`task_2_kerr_shift`), we attach `KerrMaterial(eps_inf=4.0, chi3=2.4e-19, method='newton')` to the 2D microring core and run short time-domain ring-down simulations at three peak modal electric field amplitudes $E_{\mathrm{peak}} \in \{3.0, 6.0, 9.0\}\times 10^7\;\mathrm{V/m}$:
#
# - As shown in **Figure 2b**, the red-shifted resonance frequency $\Delta f_{\mathrm{NL}}(U) = f(\chi^{(3)}, U) - f(0) < 0$ extracted by `wakis.harminv` scales linearly with the stored intracavity mode energy $U$ ($[7.740, 30.961, 69.661]\;\mathrm{fJ}$) and matches the first-order cavity perturbation formula (`wakis.kerr_cavity_frequency_shift`) evaluated with the mode's effective nonlinear volume $V_{\mathrm{eff,NL}} = 0.863\;\mu\mathrm{m}^3$ (standing wave; $1.290\;\mu\mathrm{m}^3$ travelling wave) within **$2.01\%$** maximum relative error ($-21.91\;\mathrm{GHz}$ vs $-22.36\;\mathrm{GHz}$ at $U_{\max} = 69.66\;\mathrm{fJ}$).
# - Replacing $V_{\mathrm{eff,NL}}$ by a $2\times$ wrong mode volume (dashed black line in Figure 2b) underpredicts $|\Delta f_{\mathrm{NL}}|$ by $50\%$, which is caught by our automated mutation test in `tests/test_tut_05_11_kerr_microcomb.py`.

# %%
if (DATA_DIR / "task_2_kerr_shift.json").exists() and (
    DATA_DIR / "microcomb_summary.json"
).exists():
    t2_data = json.loads((DATA_DIR / "task_2_kerr_shift.json").read_text())

    if (FIG_DIR / "02_dispersion_and_kerr_ring_shift.png").exists():
        display(Image(filename=str(FIG_DIR / "02_dispersion_and_kerr_ring_shift.png"), width=980))

    print(
        f"Part 2 2D FIT Kerr Ring Resonance Shift Verification:\n"
        f"  Standing-wave V_eff,NL    : {t2_data['V_eff_nl_um3']:.3f} um^3 "
        f"(travelling-wave V_eff,NL = {summary_data['V_eff_travelling_um3']:.3f} um^3)\n"
        f"  Stored energies U [fJ]    : {[round(u * 1e15, 3) for u in t2_data['stored_energy_j']]}\n"
        f"  FIT Kerr shifts [GHz]     : {[round(df * 1e-9, 3) for df in t2_data['df_fit_hz']]}\n"
        f"  Perturbation theory [GHz] : {[round(df * 1e-9, 3) for df in t2_data['df_analytic_hz']]}\n"
        f"  Max relative error        : {t2_data['max_rel_error'] * 100:.2f}%"
    )

# %% [markdown]
# ---
#
# ## 5. Part 3 · Resolution Convergence Study & Locally Refined Graded Mesh
#
# Task 3 (`task_3_convergence`) verifies convergence of the 2D FIT microring dispersion parameters across three uniform grid resolutions—**Coarse** ($\Delta x = 30\;\mathrm{nm}$, $23.2\;\mathrm{cells}/\lambda_{\mathrm{core}}$, $5.4\;\mathrm{cells}$ across the $160\;\mathrm{nm}$ gap), **Medium** ($\Delta x = 25\;\mathrm{nm}$, $27.6\;\mathrm{cells}/\lambda_{\mathrm{core}}$, $6.4\;\mathrm{cells}$ across the gap), and **Fine** ($\Delta x = 20\;\mathrm{nm}$, $34.5\;\mathrm{cells}/\lambda_{\mathrm{core}}$, $8.0\;\mathrm{cells}$ across the gap)—alongside the **locally refined graded mesh** (`build_microring_graded_axes`, $\Delta x \in [20.0, 35.0]\;\mathrm{nm}$, $-20.9\%$ 2D cell count) and **Richardson extrapolation** ($\Delta x \to 0$). Across $\Delta x \in \{30, 25, 20\}\;\mathrm{nm}$, the observed 3-grid convergence orders are $p_{\mathrm{obs}}(f_0) = 1.75$, $p_{\mathrm{obs}}(D_1) = 2.04$, and $p_{\mathrm{obs}}(D_2) = 3.71$ (elevated above 2 by non-monotonic staircasing cancellation across the three azimuthal modes $m \in \{16, 17, 18\}$). We therefore report both the nominal $p = 2$ Richardson extrapolation ($D_2^{\mathrm{Rich}}(p=2)/(2\pi) = +30.49\;\mathrm{GHz}$, **$1.19\%$** error vs Bessel) and the empirical $p_{\mathrm{obs}} = 3.71$ extrapolation ($D_2^{\mathrm{Rich}}(p_{\mathrm{obs}})/(2\pi) = +29.03\;\mathrm{GHz}$, **$5.92\%$** error vs Bessel), spanning the extrapolation band $[+29.03, +30.49]\;\mathrm{GHz}$ ($1.19\%\text{–}5.92\%$) (Figure 3):

# %%
if (DATA_DIR / "task_3_convergence.json").exists():
    t3_data = augment_convergence_dict(
        json.loads((DATA_DIR / "task_3_convergence.json").read_text())
    )

    if (FIG_DIR / "03_resolution_convergence.png").exists():
        display(Image(filename=str(FIG_DIR / "03_resolution_convergence.png"), width=980))

    conv_rows = [
        "| Grid Level | $\\Delta x$ [nm] | 2D Cells ($N_x \\times N_y$) | $\\mathrm{cells}/\\lambda_{\\mathrm{core}}$ | Gap Cells ($160\\;\\mathrm{nm}$) | $f_0$ [THz] | $D_1/(2\\pi)$ [THz] | $D_2/(2\\pi)$ [GHz] | $D_2$ Rel. Error vs Bessel [\\%] |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for i, lvl in enumerate(t3_data["levels"]):
        conv_rows.append(
            f"| **Uniform {lvl}** | {t3_data['dx_nm'][i]:.1f} | {t3_data['cells_2d'][i]:,} | "
            f"{t3_data['ppw_core'][i]:.1f} | {t3_data['cells_across_gap'][i]:.1f} | "
            f"{t3_data['f0_hz'][i]*1e-12:.4f} | {t3_data['D1_hz'][i]*1e-12:.4f} | "
            f"{t3_data['D2_hz'][i]*1e-9:+.2f} | {t3_data['D2_rel_err_vs_bessel'][i]*100:.2f}% |"
        )
    conv_rows.append(
        f"| **Locally Refined Graded** | 20.0–35.0 | **{t3_data['graded_cells_2d']:,}** ($-{t3_data['graded_cell_reduction_pct']:.1f}\\%$) | "
        f"29.3–34.5 | 8.0 | {t3_data['graded_f0_hz']*1e-12:.4f} | {t3_data['graded_D1_hz']*1e-12:.4f} | "
        f"**{t3_data['graded_D2_hz']*1e-9:+.2f}** | **{t3_data['graded_D2_rel_err']*100:.2f}%** |"
    )
    conv_rows.append(
        f"| **Richardson ($p = 2$ nominal)** | $\\to 0$ | — | $\\to\\infty$ | $\\to\\infty$ | "
        f"**{t3_data['f0_richardson_hz']*1e-12:.4f}** | **{t3_data['D1_richardson_hz']*1e-12:.4f}** | "
        f"**{t3_data['D2_richardson_hz']*1e-9:+.2f}** | "
        f"**{abs(t3_data['D2_richardson_hz'] - t3_data['D2_bessel_hz'])/t3_data['D2_bessel_hz']*100:.2f}%** |"
    )
    conv_rows.append(
        f"| **Richardson ($p_{{\\mathrm{{obs}}}} = {t3_data['D2_observed_order']:.2f}$)** | $\\to 0$ | — | $\\to\\infty$ | $\\to\\infty$ | "
        f"**{t3_data['f0_richardson_obs_hz']*1e-12:.4f}** | **{t3_data['D1_richardson_obs_hz']*1e-12:.4f}** | "
        f"**{t3_data['D2_richardson_obs_hz']*1e-9:+.2f}** | "
        f"**{t3_data['D2_richardson_obs_rel_err']*100:.2f}%** |"
    )
    conv_rows.append(
        f"| **Cylindrical Bessel Theory** | Exact | — | — | — | "
        f"**{t3_data['f0_bessel_hz']*1e-12:.4f}** | **{t3_data['D1_bessel_hz']*1e-12:.4f}** | "
        f"**{t3_data['D2_bessel_hz']*1e-9:+.2f}** | **0.00%** |"
    )
    display(Markdown("\n".join(conv_rows)))

# %% [markdown]
# ---
#
# ## 6. Part 4 · FIT-Informed Lugiato–Lefever Soliton Microcomb Dynamics
#
# Passing the FIT-measured cold-cavity dispersion ($D_2/(2\pi) = +27.89\;\mathrm{GHz}$), bus-coupled linewidth ($Q_L = 534.2, \kappa/(2\pi) = 401.06\;\mathrm{GHz}, \eta = 0.9894$), and single-photon Kerr shift ($g_0/(2\pi) = 28.44\;\mathrm{kHz}$ from $V_{\mathrm{eff,NL}} = 1.290\;\mu\mathrm{m}^3$ and $\chi^{(3)} = 2.4\times 10^{-19}\;\mathrm{m^2/V^2}$) into `wakis.solve_lle` ($N_\mu = 256$ azimuthal modes) yields the FIT-derived dimensionless dispersion $d_2 = 2 D_2 / \kappa = 0.1391$ and parametric threshold $P_{\mathrm{th}} = 0.638\;\mathrm{W}$. Choosing the dimensionless external pump at $f = 2.350$ (inside the single-soliton regime $f > 1.241$, with required bus input power $P_{\mathrm{in}} = f^2 P_{\mathrm{th}} = 3.521\;\mathrm{W}$ following from $P_{\mathrm{th}}$), Task 4 (`task_4_lle_microcomb`) sweeps the laser pump detuning from blue detuning ($\zeta_0 = -1.2$) across cavity resonance into the red-detuned bistable regime ($\zeta_0 = +5.2$):
#
# - **Bistability, MI $\to$ Breather $\to$ Single Soliton Step (Figure 4):** As the laser scans across $\zeta_0 \approx 0$, the intracavity power rises along the upper CW branch and breaks into **Turing rolls (primary modulation instability)** at $\zeta_0 \approx 0.55$, transitions through **chaotic breathers** around $\zeta_0 \approx 1.75$, and drops onto the **soliton step** for $\zeta_0 > 2.6$, stabilizing into a single dissipative Kerr soliton (DKS) at $\zeta_0 = 4.2 < \zeta_{\max} = (\pi^2/8)f^2 = 6.81$.
# - **$\mathrm{sech}^2$ Microcomb Spectrum & $D_2^{-1/2}$ Bandwidth Scaling (Figure 5):** At $\zeta_0 = 4.2$ and $d_2 = 0.1391$, the simulated azimuthal pulse width $\Delta\phi_{\mathrm{FWHM}} = 0.2115\;\mathrm{rad}$ matches the analytical $\mathrm{sech}$ profile ($0.2268\;\mathrm{rad}$) within **$6.74\%$**, and the 3 dB comb spectral bandwidth $\Delta\mu_{\mathrm{FWHM}} = 8.90\;\mathrm{modes}$ matches the analytical $\mathrm{sech}^2$ law ($8.72\;\mathrm{modes}$) within **$2.08\%$**. Sweeping $d_2 \in \{0.02, 0.04, 0.08, 0.16\}$ confirms the power-law scaling $\Delta\mu_{\mathrm{FWHM}} \propto d_2^{-1/2}$ with fitted log-log exponent **$-0.4977$** ($0.45\%$ error vs $-0.5000$).

# %%
if (DATA_DIR / "task_4_lle_microcomb.json").exists():
    t4_data = json.loads((DATA_DIR / "task_4_lle_microcomb.json").read_text())

    if (FIG_DIR / "04_lle_detuning_scan_and_soliton_regimes.png").exists():
        display(Image(filename=str(FIG_DIR / "04_lle_detuning_scan_and_soliton_regimes.png"), width=980))
    if (FIG_DIR / "05_soliton_comb_spectrum_and_d2_scaling.png").exists():
        display(Image(filename=str(FIG_DIR / "05_soliton_comb_spectrum_and_d2_scaling.png"), width=980))

    print(
        f"Part 4 FIT-Informed Lugiato-Lefever Soliton Microcomb Results:\n"
        f"  Max soliton detuning zeta_max = (pi^2/8)f^2 : {t4_data['zeta_max_analytic']:.3f} (operating at zeta_0 = {t4_data['zeta_sol']:.2f})\n"
        f"  Azimuthal pulse FWHM [rad]  : SSFM = {t4_data['fwhm_phi_sim']:.4f} vs Analytic = {t4_data['fwhm_phi_an']:.4f} "
        f"(rel. error = {t4_data['fwhm_phi_rel_error']*100:.2f}%)\n"
        f"  Spectral comb FWHM [modes]  : SSFM = {t4_data['fwhm_mu_sim']:.2f} vs Analytic = {t4_data['fwhm_mu_an']:.2f} "
        f"(rel. error = {t4_data['fwhm_mu_rel_error']*100:.2f}%)\n"
        f"  Bandwidth vs d2 log-log slope: fitted = {t4_data['dispersion_scaling_exponent']:.4f} vs analytic = -0.5000 "
        f"(rel. error = {t4_data['dispersion_exponent_rel_error']*100:.2f}%)"
    )

# %% [markdown]
# The animated GIF below (`figures/kerr_microcomb_dynamics.gif`) reconstructs the 2D microring intracavity intensity $|\Psi(\phi, \tau) u_{m_0}(r)|^2$ alongside the azimuthal waveform $|\Psi(\phi, \tau)|^2$ across the pump-detuning scan from primary MI rolls through chaotic breathers onto the stationary single dissipative Kerr soliton:
#
# ![Kerr microcomb soliton formation dynamics](figures/kerr_microcomb_dynamics.gif)

# %% [markdown]
# ---
#
# ## 7. Quantitative Summary
#
# Every value in the table below is generated by `compute_all_study_data` (`--full`, `"quick": false`) and stored in `data/microcomb_summary.json` (verified automatically by `tests/test_tut_05_11_kerr_microcomb.py`):
#
# | Benchmark / Quantity | Theory / Analytical Reference | Wakis 2D FIT / SSFM Result | Relative Error |
# | :--- | :---: | :---: | :---: |
# | **Cold-cavity pump frequency** $f_0$ ($m_0 = 17, \Delta x = 20\;\mathrm{nm}$) | $214.513\;\mathrm{THz}$ | $214.339\;\mathrm{THz}$ | **$0.08\%$** |
# | **Cold-cavity pump frequency** $f_0$ (Richardson $\Delta x \to 0$) | $214.513\;\mathrm{THz}$ | $214.511\;\mathrm{THz}$ | **$0.0009\%$** |
# | **Free spectral range** $D_1 / (2\pi)$ ($\mathrm{FSR}, \Delta x = 20\;\mathrm{nm}$) | $10.349\;\mathrm{THz}$ | $10.319\;\mathrm{THz}$ | **$0.29\%$** |
# | **Free spectral range** $D_1 / (2\pi)$ (Richardson $\Delta x \to 0$) | $10.349\;\mathrm{THz}$ | $10.348\;\mathrm{THz}$ | **$0.0053\%$** |
# | **IR-pole differential dispersion shift** $\Delta D_2 / (2\pi)$ | $+53.86\;\mathrm{GHz}$ | $+53.93\;\mathrm{GHz}$ | **$0.13\%$** |
# | **Anomalous dispersion** $D_2 / (2\pi)$ (Fine $\Delta x = 20\;\mathrm{nm}$) | $+30.85\;\mathrm{GHz}$ | $+27.89\;\mathrm{GHz}$ | **$9.59\%$** |
# | **Anomalous dispersion** $D_2 / (2\pi)$ (Richardson $p = 2$, $\Delta x \to 0$) | $+30.85\;\mathrm{GHz}$ | $+30.49\;\mathrm{GHz}$ | **$1.19\%$** |
# | **Anomalous dispersion** $D_2 / (2\pi)$ (Richardson $p_{\mathrm{obs}} = 3.71$, $\Delta x \to 0$) | $+30.85\;\mathrm{GHz}$ | $+29.03\;\mathrm{GHz}$ | **$5.92\%$** |
# | **Normal dispersion** $D_2 / (2\pi)$ (no IR pole, $\Delta x = 20\;\mathrm{nm}$) | $-23.01\;\mathrm{GHz}$ | $-26.04\;\mathrm{GHz}$ | **$13.17\%$** |
# | **2D FIT Kerr ring resonance shift** $\Delta f_{\mathrm{NL}}(U_{\max})$ | $-22.36\;\mathrm{GHz}$ ($V_{\mathrm{eff,NL}} = 0.863\;\mu\mathrm{m}^3$) | $-21.91\;\mathrm{GHz}$ | **$2.01\%$** |
# | **LLE single-DKS azimuthal pulse FWHM** $\Delta\phi_{\mathrm{FWHM}}$ ($\zeta_0 = 4.2$) | $0.2268\;\mathrm{rad}$ | $0.2115\;\mathrm{rad}$ | **$6.74\%$** |
# | **LLE $\mathrm{sech}^2$ comb spectral FWHM** $\Delta\mu_{\mathrm{FWHM}}$ ($d_2 = 0.1391$) | $8.72\;\mathrm{modes}$ | $8.90\;\mathrm{modes}$ | **$2.08\%$** |
# | **Soliton bandwidth dispersion exponent** $\Delta\mu_{\mathrm{FWHM}} \propto d_2^\alpha$ | $-0.5000$ | $-0.4977$ | **$0.45\%$** |

# %% [markdown]
# ---
#
# ## 8. Limits & Failure Modes
#
# 1. **Normal group-velocity dispersion ($D_2 < 0$) prevents bright Kerr soliton formation:**
#    In a focusing Kerr medium ($n_2 > 0$), bright dissipative Kerr solitons require anomalous dispersion ($D_2 > 0, \beta_2 < 0$) so that anomalous group-velocity dispersion balances self-phase modulation. As demonstrated in Figure 2a (`cold_norm`, where omitting the infrared pole yields $D_2/(2\pi) = -26.04\;\mathrm{GHz} < 0$) and verified in our mutation test, flipping $D_2 \to -|D_2|$ eliminates bright soliton steps in the red-detuned LLE scan (supporting only dark/platicon states on the upper branch).
# 2. **Finite-grid dispersion error band on Cartesian meshes ($D_2$ sensitivity):**
#    Because $D_2 = \omega_{m_0+1} - 2\omega_{m_0} + \omega_{m_0-1}$ is a second finite difference whose magnitude ($\sim 30\;\mathrm{GHz}$) is $7\times 10^3$ times smaller than the optical carrier ($214.5\;\mathrm{THz}$), small Cartesian staircasing errors along the curved ring walls are strongly amplified in $D_2$. Even with $16\times 16$ dual-cell subpixel permittivity averaging and 2D core modal projection, the direct 2D FIT error vs the continuum Bessel reference spans $22.44\%$ at $\Delta x = 30\;\mathrm{nm}$, $14.32\%$ at $\Delta x = 25\;\mathrm{nm}$, $14.22\%$ on the locally refined graded mesh, and $9.59\%$ at $\Delta x = 20\;\mathrm{nm}$ ($13.17\%$ for normal $D_2$). Across $\Delta x \in \{30, 25, 20\}\;\mathrm{nm}$, the observed 3-grid convergence order is $p_{\mathrm{obs}} = 3.71$, yielding an empirical Richardson extrapolation of $+29.03\;\mathrm{GHz}$ (**$5.92\%$** error vs $+30.85\;\mathrm{GHz}$ Bessel), whereas nominal second-order ($p = 2$) Richardson extrapolation yields $+30.49\;\mathrm{GHz}$ (**$1.19\%$** error), defining an extrapolated continuum band of $[+29.03, +30.49]\;\mathrm{GHz}$ ($1.19\%\text{–}5.92\%$ error).
# 3. **Artificial 80 THz Lorentz pole for 2D dispersion engineering & compact-$R$ quality factor:**
#    In experimental 3D $\mathrm{Si}_3\mathrm{N}_4$ microresonators ($R \sim 20\text{–}1000\;\mu\mathrm{m}$), anomalous dispersion is achieved geometrically by tuning the vertical slab thickness and waveguide width, and loaded quality factors reach $Q_L \sim 10^5\text{–}10^7$. In a 2D TM model of an ultra-compact $R = 2.0\;\mu\mathrm{m}$ ring, geometric bending dispersion is strongly normal ($D_2/(2\pi) = -23.01\;\mathrm{GHz}$) and the overcoupled bus gap ($g = 160\;\mathrm{nm}$) yields $Q_L = 534.2$. The lossless $f_{\mathrm{IR}} = 80\;\mathrm{THz}$ Lorentz pole ($\Delta\varepsilon = 1.50$) is an artificial dispersion-engineering device introduced solely to flip $D_2 > 0$ in 2D without a 3D cross-sectional mode sweep.
# 4. **Pump detuning exceeding the existence bound $\zeta_0 > \zeta_{\max} = \frac{\pi^2}{8} f^2$:**
#    Because cavity loss must be compensated by parametric four-wave mixing from the CW background, single dissipative Kerr solitons exist only up to the maximum red detuning $\zeta_{\max} = (\pi^2/8)f^2$ (`lle_soliton_analytic` returns `in_existence_range=False` above this threshold) [3]. Scanning $\zeta_0$ beyond $\zeta_{\max}$ causes the soliton to collapse abruptly to the homogeneous lower CW branch.
# 5. **Sub-cycle modulation instability, defocusing blowup, and complex-field restriction in `wakis.KerrMaterial`:**
#    In full-wave FIT time stepping (`wakis.KerrMaterial`), if a strong negative (defocusing) susceptibility $\chi^{(3)} < 0$ or an unphysically large field drives the instantaneous differential permittivity $\frac{dD}{dE} = \varepsilon_0(\varepsilon_{\mathrm{eff}} + 3\alpha |\mathbf{E}|^2) \le 0$, the local wave speed $v = (\mu_0\,dD/dE)^{-1/2}$ becomes imaginary or exceeds the vacuum Courant limit, triggering immediate grid-scale blowup. Furthermore, because the sub-cycle cubic update $\mathbf{D}_{\mathrm{NL}} = \varepsilon_0 \chi^{(3)}|\mathbf{E}|^2\mathbf{E}$ relies on real instantaneous fields so that $\cos^3\theta = \frac{3}{4}\cos\theta + \frac{1}{4}\cos 3\theta$ produces the physical $3/4$ self-phase-modulation factor and third-harmonic generation, `KerrMaterial` raises `NotImplementedError` if attached to a solver with complex-dtype fields (where $|\mathbf{E}|^2\mathbf{E}$ would overpredict the Kerr shift by $4/3$ and omit THG).
# 6. **Unphase-matched third-harmonic generation & carrier-shock steepening in full-wave FIT vs slowly varying LLE:**
#    The mean-field LLE retains only the fundamental resonant comb family around $\omega_0$ (with the $3/4$ real-field Kerr factor) and neglects third-harmonic generation at $3\omega_0$ and self-steepening $(1/\omega_0)\partial_t(|\Psi|^2\Psi)$. In a full-wave sub-cycle FIT simulation at very high index contrast $\Delta n_{\mathrm{NL}} / n_0 \gtrsim 0.05$, optical carrier shock formation (Ch. 9 of Taflove & Hagness [1]) and Cherenkov dispersive-wave radiation via higher-order dispersion $D_3 \mu^3 / 6$ generate spectral sidebands beyond the second-order $\mathrm{sech}^2$ envelope.
#
# ---
#
# ## 9. Verified References
#
# 1. **A. Taflove and S. C. Hagness**, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed., Artech House, Norwood, MA (2005), Chapter 9 ("Dispersive, Nonlinear, and Gain Materials"). — Formulates the sub-cycle FDTD/FIT update for instantaneous Kerr and delayed Lorentz/Raman nonlinear polarizations.
# 2. **R. M. Joseph and A. Taflove**, "FDTD Maxwell's equations models for nonlinear electrodynamics and optics," *IEEE Transactions on Antennas and Propagation* **45**(3), 364–374 (1997), DOI: [`10.1109/8.558652`](https://doi.org/10.1109/8.558652). — Derives the coupled auxiliary differential equation (ADE) and Newton/fixed-point constitutive inversions for $\chi^{(3)}$ Kerr + Raman media in time-domain Maxwell solvers.
# 3. **T. Herr, V. Brasch, J. D. Jost, C. Y. Wang, N. M. Kondratiev, M. L. Gorodetsky, and T. J. Kippenberg**, "Temporal solitons in optical microresonators," *Nature Photonics* **8**(2), 145–152 (2014), DOI: [`10.1038/nphoton.2013.343`](https://doi.org/10.1038/nphoton.2013.343). — Demonstrates dissipative Kerr soliton formation, soliton steps on a laser detuning scan, the analytical single-soliton $\mathrm{sech}$ approximation, and the maximum detuning bound $\zeta_{\max} = (\pi^2/8)f^2$.
# 4. **Y. K. Chembo and C. R. Menyuk**, "Spatiotemporal Lugiato-Lefever formalism for Kerr-comb generation in whispering-gallery-mode resonators," *Physical Review A* **87**(5), 053852 (2013), DOI: [`10.1103/PhysRevA.87.053852`](https://doi.org/10.1103/PhysRevA.87.053852). — Derives the spatiotemporal Lugiato–Lefever equation from the 3D Maxwell equations and whispering-gallery mode expansion with integrated dispersion $D_{\mathrm{int}}(\mu)$ and effective nonlinear mode volume.
# 5. **T. J. Kippenberg, A. L. Gaeta, M. Lipson, and M. L. Gorodetsky**, "Dissipative Kerr solitons in optical microresonators," *Science* **361**(6402), eaan8083 (2018), DOI: [`10.1126/science.aan8083`](https://doi.org/10.1126/science.aan8083). — Comprehensive review of microresonator dispersion engineering, modulation instability, breather solitons, and $\mathrm{sech}^2$ microcomb bandwidth scaling.
# 6. **R. W. Boyd**, *Nonlinear Optics*, 4th ed., Academic Press, San Diego (2020), Chapter 4 ("The Intensity-Dependent Refractive Index") & Chapter 2 ("Wave-Equation Description of Nonlinear Optical Interactions"). — Standard reference for the real-field factor $3/4$ relating $\chi^{(3)}$ to $n_2$ and undepleted-pump Maker-fringe third-harmonic generation.
# 7. **V. A. Mandelshtam and H. S. Taylor**, "Harmonic inversion of time signals and its applications," *Journal of Chemical Physics* **107**(17), 6756–6769 (1997), DOI: [`10.1063/1.475324`](https://doi.org/10.1063/1.475324). — Filter-diagonalization method (`wakis.harminv`) used to extract sub-Fourier-limit cavity resonance frequencies and quality factors from short FIT ring-down traces.


# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments (`--quick`, `--full`, `--task`, `--list-tasks`, `--plot`, `--out`)."""
    ap = argparse.ArgumentParser(
        description="Tutorial 5.11: Kerr (chi^(3)) Nonlinear Media in FIT and FIT-Informed Soliton Microcombs"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run fast CPU study tasks and render all figures and GIF.",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run full-resolution study tasks and render all figures and GIF.",
    )
    ap.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single study task index (0..4).",
    )
    ap.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the study task table and exit.",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Regenerate all figures and GIF in figures/ from saved data/.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR,
        help="Output data directory (default: data/).",
    )
    if argv is None:
        env_args = os.environ.get("WAKIS_TUTORIAL_ARGS")
        argv = shlex.split(env_args) if env_args else sys.argv[1:]
    args, _ = ap.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for Tutorial 5.11."""
    args = parse_args(argv)
    quick = not args.full

    if args.list_tasks:
        print(f"# Total tasks: {len(TASKS)}")
        for idx, name, desc in TASKS:
            print(f"{idx}\t{name}\t{desc}")
        return

    if args.task is not None:
        if not (0 <= args.task < len(TASKS)):
            raise SystemExit(f"--task must be in 0..{len(TASKS) - 1}")
        compute_all_study_data(quick=quick, out_dir=args.out, selected_task=int(args.task))
        return

    if args.plot:
        render_figures_from_disk(data_dir=args.out, fig_dir=FIG_DIR)
        return

    compute_all_study_data(quick=quick, out_dir=args.out, selected_task=None)
    render_figures_from_disk(data_dir=args.out, fig_dir=FIG_DIR)


if __name__ == "__main__":
    main()
