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
# # Tutorial 2.1 · Where dispersion comes from
#
# **Abstract.** No material except vacuum responds instantaneously to an
# electromagnetic field at all frequencies: bound electrons have inertia and
# restoring forces, crystal ions vibrate at infrared optical-phonon frequencies,
# conduction electrons scatter with a finite momentum-relaxation rate, and
# permanent molecular dipoles reorient by thermal rotational diffusion. In the
# time domain, these microscopic mechanisms make the electric flux density depend
# on the history of the electric field through a causal susceptibility kernel
# $\chi(t)$; in the frequency domain, they produce the canonical **Lorentz**,
# **Drude**, and **Debye** dispersion relations and explain why noble metals such
# as gold depart from a pure free-electron Drude response below
# $\lambda \approx 550\;\text{nm}$ due to $5d \to 6sp$ **interband transitions**.
# In this tutorial we derive each dispersion model from its microscopic equation
# of motion in Wakis's $\exp(+i\omega t)$ time-harmonic convention (where
# **passivity requires $\operatorname{Im}\varepsilon(\omega) \le 0$** for
# $\omega > 0$), verify **causality**, the **Kramers–Kronig (KK) relations**, and
# the **Thomas–Reiche–Kuhn $f$-sum rule** both analytically and numerically,
# show the unified **Auxiliary Differential Equation (ADE)** that `SolverFIT3D`
# integrates at each leapfrog step, and benchmark quasi-1D FIT slab
# reflectance/transmittance spectra $R(f), T(f)$ and $O(\Delta t^2)$ temporal
# convergence against the exact transfer matrix evaluated with
# `DispersiveMaterial.epsilon_discrete(f, dt)` and `epsilon(f)`.
#
# ---
#
# ## Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. **Derive** the Lorentz (bound electrons and polar-crystal IR optical phonons
#    with the Lyddane–Sachs–Teller relation), Drude (free carriers, plasma
#    reflection edge, and epsilon-near-zero $\operatorname{Re}\varepsilon = 0$
#    point), and Debye (orientational dipole relaxation) permittivity models
#    $\varepsilon(\omega)$ together with their causal time-domain susceptibility
#    kernels $\chi(t)$ in the $\exp(+i\omega t)$ convention
#    ($\operatorname{Im}\varepsilon(\omega) \le 0$ for passive media).
# 2. **Explain** why gold (`Au`) departs from a single free-carrier Drude model
#    below $\lambda \approx 550\;\text{nm}$ ($\hbar\omega \gtrsim 2.25\;\text{eV}$)
#    using the experimental **Johnson & Christy (1972)** [3] dataset and the
#    **Rakić *et al.* (1998)** [2] Drude + 5-Lorentz critical-point fit in
#    `wakis.material_library`.
# 3. **Verify** causality, passivity ($\omega\,\varepsilon''(\omega) =
#    -\omega\operatorname{Im}\varepsilon(\omega) \ge 0$), the **Kramers–Kronig**
#    Hilbert-transform pair, and the oscillator-strength **$f$-sum rule**
#    (**Jackson** Ch. 7 [1]) on both analytical pole models and tabulated optical
#    data.
# 4. **Formulate** the second-order Auxiliary Differential Equation (ADE)
#    discretization in `wakis.dispersive` (**Taflove & Hagness** Ch. 9 [6]),
#    derive its exact discrete frequency response `epsilon_discrete(f, dt)`, and
#    determine the leapfrog stability ceiling (`is_stable(dt)`,
#    $\omega_{0,p}\Delta t < 2$).
# 5. **Simulate** broadband plane-wave reflection $R(f)$, transmission $T(f)$,
#    and absorption $A(f) = 1 - R(f) - T(f)$ through Lorentz (GaN Reststrahlen
#    band, **Barker & Ilegems 1973** [4]), Drude (ENZ/plasma edge), and Debye
#    (liquid water) slabs in `SolverFIT3D`, and quantify the $O(\Delta t^2)$
#    discretization error between `epsilon_discrete(f, dt)` and `epsilon(f)`.
#
# ## Contents
#
# 1. [Time-harmonic convention, passivity & causal susceptibility kernels](#1.-Time-harmonic-convention,-passivity-&-causal-susceptibility-kernels)
# 2. [Microscopic dispersion models: Lorentz, Drude, Debye & interband transitions](#2.-Microscopic-dispersion-models:-Lorentz,-Drude,-Debye-&-interband-transitions)
# 3. [Causality, Kramers–Kronig relations & the $f$-sum rule](#3.-Causality,-Kramers–Kronig-relations-&-the-f-sum-rule)
# 4. [The Auxiliary Differential Equation (ADE) in Wakis](#4.-The-Auxiliary-Differential-Equation-(ADE)-in-Wakis)
# 5. [Study tasks, resolution audit & preflight setup](#5.-Study-tasks,-resolution-audit-&-preflight-setup)
# 6. [Slab reflection/transmission & ADE convergence results](#6.-Slab-reflection/transmission-&-ADE-convergence-results)
# 7. [Limits & failure modes](#7.-Limits-&-failure-modes)
# 8. [References](#8.-References)
#
# ### Simulation setups & geometry overview
#
# Two distinct `SolverFIT3D` geometries are exercised across the five benchmark
# tasks (`figures/00_setup_schematic.png`):
#
# - **Panel (a) — Bulk 1-cell periodic ADE testbench (`Task 0`):** A single 3D
#   Yee/FIT cell ($N_x = N_y = N_z = 1$, $20\times 20\times 20\;\text{mm}^3$)
#   with `periodic` boundary conditions on all six faces (`k = 0` spatially
#   uniform mode), filled with a homogeneous `DispersiveMaterial` (`Lorentz`,
#   `Drude`, or `Debye`) and driven by a broadband $E_x$-polarized volumetric
#   current density $J_x(t)$ (`f0 = 2.0 GHz`). Co-located $E_x(t)$ and $P_x(t)$
#   probes verify the ADE recursion directly against `epsilon_discrete(f, dt)`.
# - **Panel (b) — Quasi-1D dispersive slab R/T testbench (`Tasks 1–4`):** A
#   $2\times 2\times N_z$ grid with `periodic` transverse faces ($\pm x, \pm y$)
#   and 15-cell `cpml` absorbers on $\pm z$. An $E_x$-polarized `CurrentSource`
#   sheet at $z_{\text{src}}$ launches a plane-wave pulse toward a dispersive
#   slab of thickness $d$ occupying $z \in [0, d]$ (wurtzite $\text{GaN}$
#   Reststrahlen `Lorentz` slab with $d = 2.5\;\mu\text{m}$ in `Task 1`,
#   free-carrier `Drude` ENZ slab with $d = 40\;\mu\text{m}$ in `Task 2`, and
#   liquid-water `Debye` slab with $d = 0.5\;\text{mm}$ in `Task 3`).
#   Reflection and transmission `FluxMonitor` planes at $z_r < 0$ and $z_t > d$
#   measure $R(f)$, $T(f)$, and $A(f) = 1 - R(f) - T(f)$.
#
# ![Simulation setups for Tutorial 2.1: bulk 1-cell periodic ADE testbench and quasi-1D dispersive slab R/T setup](figures/00_setup_schematic.png)
#
# ---
#
# ## 1. Time-harmonic convention, passivity & causal susceptibility kernels
#
# In a linear, isotropic, time-translation-invariant medium, the electric flux
# density $\mathbf{D}(t)$ responds to the electric field $\mathbf{E}(t)$ through
# an instantaneous high-frequency background permittivity $\varepsilon_\infty \ge 1$
# (arising from core shells far above the frequency band of interest), an optional
# DC ohmic conductivity $\sigma \ge 0$, and a causal susceptibility kernel
# $\chi(t)$ that vanishes identically for $t < 0$ (**Jackson** Ch. 7 [1]):
#
# $$\mathbf{D}(t) = \varepsilon_0 \varepsilon_\infty \mathbf{E}(t)
#   + \varepsilon_0 \int_0^\infty \chi(\tau)\,\mathbf{E}(t - \tau)\,d\tau,
#   \qquad \chi(t) = 0 \;\;\text{for}\;\; t < 0.$$
#
# ### The $\exp(+i\omega t)$ sign convention and passivity
#
# Throughout Wakis (`wakis.dispersive` and `wakis.material_library`), monochromatic
# fields vary in time as $\mathbf{E}(t) = \operatorname{Re}\!\left[\hat{\mathbf{E}}(\omega)\,e^{+i\omega t}\right]$.
# Substituting $e^{+i\omega t}$ into the convolution integral defines the
# one-sided Fourier transform
#
# $$\chi(\omega) = \int_0^\infty \chi(t)\,e^{-i\omega t}\,dt,
#   \qquad
#   \frac{\varepsilon(\omega)}{\varepsilon_0}
#   = \varepsilon_\infty + \sum_{p=1}^{N_p} \chi_p(\omega)
#     + \frac{\sigma}{i\omega\varepsilon_0}
#   \equiv \varepsilon'(\omega) - i\,\varepsilon''(\omega).$$
#
# Under the $\exp(+i\omega t)$ convention, the time-averaged electromagnetic
# power dissipated per unit volume in a non-magnetic medium at frequency
# $\omega > 0$ is
#
# $$\langle p_{\text{diss}} \rangle
#   = \frac{1}{2}\operatorname{Re}\!\left[\hat{\mathbf{E}}^* \cdot \left(i\omega\varepsilon_0\varepsilon(\omega)\hat{\mathbf{E}}\right)\right]
#   = -\frac{1}{2}\varepsilon_0\,\omega\,\operatorname{Im}\varepsilon(\omega)\,|\hat{\mathbf{E}}|^2
#   = \frac{1}{2}\varepsilon_0\,\omega\,\varepsilon''(\omega)\,|\hat{\mathbf{E}}|^2.$$
#
# > **Passivity rule ($\exp(+i\omega t)$ convention):** A passive medium can only
# > absorb—never generate—net electromagnetic energy from a monochromatic field.
# > Therefore **every passive material satisfies $\operatorname{Im}\varepsilon(\omega) \le 0$**
# > (equivalently $\varepsilon''(\omega) = -\operatorname{Im}\varepsilon(\omega) \ge 0$)
# > for all $\omega > 0$. Similarly, a forward-propagating wave along $+z$,
# > $\exp(+i(\omega t - n\omega z / c_0))$, decays exponentially with distance
# > when the complex refractive index $n(\omega) = \sqrt{\varepsilon(\omega)/\varepsilon_0}
# > = n'(\omega) - i\,\kappa(\omega)$ has **$\operatorname{Im} n(\omega) \le 0$**
# > ($\kappa \ge 0$), corresponding to $\varepsilon = (n' - i\kappa)^2 = (n'^2 - \kappa^2) - 2i n'\kappa$.
#
# ---
#
# ## 2. Microscopic dispersion models: Lorentz, Drude, Debye & interband transitions
#
# Across the electromagnetic spectrum from radio frequencies ($\text{MHz}$) to
# the extreme ultraviolet ($\text{PHz}$), the macroscopic permittivity
# $\varepsilon(\omega) = \varepsilon'(\omega) - i\,\varepsilon''(\omega)$ exhibits
# a hierarchy of step drops in $\varepsilon'(\omega)$ and absorption peaks in
# $\varepsilon''(\omega)$ (**Kittel** Ch. 16 [7]; **Jackson** Ch. 7 [1]). As the
# driving frequency increases, heavier charge carriers with longer inertial or
# viscous response times cease to follow the oscillating field ("freeze out"),
# mapping each physical regime onto a specific pole class in `wakis.dispersive`:
#
# - **Microwave / RF orientational (dipolar) relaxation ($\sim 10^8\text{–}10^{11}\;\text{Hz}$)** $\longrightarrow$
#   `Debye(delta_eps, tau)`: permanent molecular dipoles in polar liquids and
#   biological tissues reorient against thermal rotational friction without
#   inertial overshoot, producing a broad symmetric loss peak at
#   $f_{\text{peak}} = 1/(2\pi\tau)$ and a step drop $\Delta\varepsilon_{\text{dip}}$
#   in $\varepsilon'(f)$.
# - **Infrared / THz ionic lattice vibrations ($\sim 10^{12}\text{–}3\times 10^{13}\;\text{Hz}$)** $\longrightarrow$
#   `Lorentz(delta_eps, omega_0, gamma)`: counter-oscillating cation and anion
#   sublattices undergo resonant transverse optical ($\omega_{\text{TO}}$) and
#   longitudinal optical ($\omega_{\text{LO}}$) phonon vibrations, opening a
#   negative-permittivity Reststrahlen reflection band between $\omega_{\text{TO}}$
#   and $\omega_{\text{LO}}$.
# - **Visible / UV electronic shell transitions ($\sim 10^{15}\text{–}3\times 10^{16}\;\text{Hz}$)** $\longrightarrow$
#   `Lorentz(delta_eps, omega_0, gamma)`: bound valence electrons oscillate
#   against the atomic core at exciton and interband transition frequencies before
#   $\varepsilon(\omega) \to \varepsilon_0$ in the X-ray limit.
# - **Intraband free-carrier conduction (DC to plasma edge $\omega_{\text{ENZ}}$)** $\longrightarrow$
#   `Drude(omega_p, gamma)`: unbound conduction electrons in metals and doped
#   semiconductors produce an $f^{-1}$ conductive loss tail at low frequencies and
#   $\varepsilon'(\omega) < 0$ below the screened plasma frequency.
#
# ![Broadband dielectric dispersion mechanisms and loss-tangent phasor](figures/wikimedia/dielectric_responses_wakis.png)
#
# *Figure: (a) Broadband dielectric dispersion spectrum $\varepsilon'(f)$ and
# $\varepsilon''(f) = -\operatorname{Im}\varepsilon(f)$ from $1\;\text{MHz}$ to
# $30\;\text{PHz}$ showing orientational (`Debye`), ionic/phonon (`Lorentz`), and
# electronic (`Lorentz`) polarization regimes alongside free-carrier (`Drude`)
# conduction, computed directly from `wakis.dispersive` models (layout after
# [`File:Dielectric_responses.svg`](https://commons.wikimedia.org/wiki/File:Dielectric_responses.svg)
# on Wikimedia Commons, vectorized by Pieter Kuiper after User:Archimerged; physics
# after Kittel [7], ch. 16, and Jackson [1], §7.5). (b) Complex current-density
# and loss-tangent phasor diagram ($\tan\delta = \varepsilon''/\varepsilon'$) in
# Wakis's $\exp(+i\omega t)$ convention (after
# [`File:Loss_tangent_phasors_1.svg`](https://commons.wikimedia.org/wiki/File:Loss_tangent_phasors_1.svg)
# by User:Sakurambo / User:LSTech, Public Domain, Wikimedia Commons; see
# `figures/wikimedia/ATTRIBUTION.md`, CC BY 4.0).*
#
# Each pole class in `wakis.dispersive` (`Lorentz`, `Drude`, `Debye`)
# originates from a specific microscopic equation of motion for the polarization
# $\mathbf{P}_p(t) = \varepsilon_0 \int_0^\infty \chi_p(\tau)\mathbf{E}(t-\tau)\,d\tau$:
#
# ### 2.1 Lorentz oscillator: bound electrons (UV/visible) and polar-crystal lattice phonons (IR Reststrahlen)
#
# Consider $N$ bound charges $q$ of effective mass $m^*$ per unit volume held near
# equilibrium by a harmonic restoring force $-m^*\omega_0^2 \mathbf{x}$ and subject
# to viscous/radiative damping $-m^*\gamma\dot{\mathbf{x}}$ ($\omega_0 > 0$,
# $\gamma \ge 0$). Multiplying $m^*(\ddot{\mathbf{x}} + \gamma\dot{\mathbf{x}} + \omega_0^2 \mathbf{x}) = q\mathbf{E}(t)$
# by $N q$ gives the macroscopic polarization ODE integrated by `Lorentz(delta_eps, omega_0, gamma)`:
#
# $$\ddot{\mathbf{P}}_p + \gamma_p \dot{\mathbf{P}}_p + \omega_{0,p}^2 \mathbf{P}_p
#   = \varepsilon_0\,\Delta\varepsilon_p\,\omega_{0,p}^2\,\mathbf{E}(t),
#   \qquad
#   \Delta\varepsilon_p = \frac{N q^2}{\varepsilon_0 m^* \omega_{0,p}^2} > 0.$$
#
# Taking the $\exp(+i\omega t)$ transform ($\partial_t \to +i\omega$) and the
# inverse Fourier transform yields the frequency-domain susceptibility and the
# causal time-domain impulse response:
#
# $$\chi_{\text{Lor}}(\omega)
#   = \frac{\Delta\varepsilon_p\,\omega_{0,p}^2}{\omega_{0,p}^2 - \omega^2 + i\gamma_p\omega},
#   \qquad
#   \chi_{\text{Lor}}(t)
#   = \Theta(t)\,\frac{\Delta\varepsilon_p\,\omega_{0,p}^2}{\Omega_{d,p}}\,
#     e^{-\gamma_p t / 2}\sin(\Omega_{d,p} t),
#   \qquad
#   \Omega_{d,p} = \sqrt{\omega_{0,p}^2 - (\gamma_p / 2)^2}.$$
#
# Notice that $\operatorname{Im}\chi_{\text{Lor}}(\omega) =
# -\Delta\varepsilon_p \omega_{0,p}^2 \gamma_p \omega / [(\omega_{0,p}^2 - \omega^2)^2 + \gamma_p^2\omega^2] \le 0$
# for all $\omega > 0$. Two distinct physical regimes are governed by the Lorentz
# oscillator:
#
# 1. **Electronic UV/visible resonances and the Sellmeier glass formula:**
#    In transparent dielectrics such as fused silica ($\text{SiO}_2$, **Malitson 1965** [5]),
#    valence electrons have ultraviolet resonances at $\lambda_1 = 68.404\;\text{nm}$
#    ($18.1\;\text{eV}$) and $\lambda_2 = 116.241\;\text{nm}$ ($10.7\;\text{eV}$),
#    plus an infrared $\text{Si–O}$ stretching vibration at $\lambda_3 = 9.896\;\mu\text{m}$.
#    In the transparent optical window far from resonance ($|\omega^2 - \omega_{0,p}^2| \gg \gamma_p\omega$),
#    setting $\gamma_p \to 0^+$ recovers the classic **Sellmeier formula**
#    $$\varepsilon(\lambda) = 1 + \sum_{p=1}^3 \frac{B_p \lambda^2}{\lambda^2 - \lambda_p^2}
#      = 1 + \sum_{p=1}^3 \frac{B_p \omega_{0,p}^2}{\omega_{0,p}^2 - \omega^2},$$
#    which is stored in `wakis.material_library` as `dispersive_lib["sio2"]` with
#    three lossless `Lorentz` poles.
# 2. **Infrared optical phonons and the Reststrahlen band of polar crystals:**
#    In an ionic or polar semiconductor crystal such as **wurtzite GaN**
#    (**Barker & Ilegems 1973** [4], $\mathbf{E} \perp c$, registered in
#    `dispersive_lib["gan o"]`), a transverse electric field drives relative
#    displacement of the $\text{Ga}^+$ and $\text{N}^-$ sublattices at the
#    transverse optical (TO) phonon frequency $\tilde{\nu}_{\text{TO}} = 560\;\text{cm}^{-1}$
#    ($f_{\text{TO}} = 16.788\;\text{THz}$, $\lambda_{\text{TO}} = 17.86\;\mu\text{m}$):
#    $$\varepsilon(\omega) = \varepsilon_\infty
#      + \frac{(\varepsilon_{\text{st}} - \varepsilon_\infty)\,\omega_{\text{TO}}^2}
#             {\omega_{\text{TO}}^2 - \omega^2 + i\gamma_{\text{ph}}\omega}
#      = \varepsilon_\infty\,\frac{\omega_{\text{LO}}^2 - \omega^2 + i\gamma_{\text{ph}}\omega}
#                                 {\omega_{\text{TO}}^2 - \omega^2 + i\gamma_{\text{ph}}\omega},$$
#    where $\varepsilon_\infty = 5.35$, $\varepsilon_{\text{st}} = 9.492$
#    ($\Delta\varepsilon = 4.142$), $\tilde{\nu}_{\text{LO}} = 746\;\text{cm}^{-1}$
#    ($f_{\text{LO}} = 22.365\;\text{THz}$) is the longitudinal optical (LO)
#    phonon frequency where $\operatorname{Re}\varepsilon(\omega_{\text{LO}}) \approx 0$,
#    and $\omega_{\text{LO}} / \omega_{\text{TO}} = \sqrt{\varepsilon_{\text{st}} / \varepsilon_\infty} = 1.332$
#    is the **Lyddane–Sachs–Teller (LST) relation**. Inside the **Reststrahlen
#    ("residual-ray") band** $\omega_{\text{TO}} < \omega < \omega_{\text{LO}}$,
#    $\operatorname{Re}\varepsilon(\omega) < 0$: electromagnetic waves are
#    evanescent inside the crystal and the surface becomes a near-ideal mirror
#    ($R \to 1$) without any free electrons!
#
# ### 2.2 Drude model: free carriers, ENZ point & plasma reflection edge
#
# In metals and doped semiconductors, conduction electrons ($n_e$ per unit volume,
# effective mass $m^*$) experience no restoring force ($\omega_0 \to 0$) and scatter
# with momentum-relaxation rate $\gamma_D = 1/\tau_{\text{coll}}$ (`Drude(omega_p, gamma)`):
#
# $$\ddot{\mathbf{P}}_D + \gamma_D \dot{\mathbf{P}}_D
#   = \varepsilon_0 \omega_p^2 \mathbf{E}(t),
#   \qquad
#   \omega_p = \sqrt{\frac{n_e e^2}{\varepsilon_0 m^*}},$$
#
# $$\chi_{\text{Drude}}(\omega)
#   = -\frac{\omega_p^2}{\omega^2 - i\gamma_D\omega}
#   = -\frac{\omega_p^2}{\omega^2 + \gamma_D^2}
#     - i\,\frac{\omega_p^2 \gamma_D}{\omega(\omega^2 + \gamma_D^2)},
#   \qquad
#   \chi_{\text{Drude}}(t)
#   = \Theta(t)\,\frac{\omega_p^2}{\gamma_D}\left(1 - e^{-\gamma_D t}\right).$$
#
# Note that the conduction current $\mathbf{J}_D(t) = \dot{\mathbf{P}}_D(t)$ has
# the purely exponential memory kernel $\sigma_D(t) = \varepsilon_0\dot{\chi}_{\text{Drude}}(t)
# = \Theta(t)\,\varepsilon_0\omega_p^2 e^{-\gamma_D t}$, which reduces in the DC
# limit ($\omega \ll \gamma_D$) to Ohm's law with conductivity
# $\sigma_0 = \varepsilon_0 \omega_p^2 / \gamma_D = n_e e^2 \tau_{\text{coll}} / m^*$.
# With bound-core background $\varepsilon_\infty$, the real permittivity
# $\operatorname{Re}\varepsilon(\omega) = \varepsilon_\infty - \omega_p^2 / (\omega^2 + \gamma_D^2)$
# crosses zero at the **epsilon-near-zero (ENZ) / screened plasma frequency**
#
# $$\omega_{\text{ENZ}} = \sqrt{\frac{\omega_p^2}{\varepsilon_\infty} - \gamma_D^2}.$$
#
# - For $\omega < \omega_{\text{ENZ}}$, $\operatorname{Re}\varepsilon(\omega) < 0$:
#   waves inside the conductor are evanescent and bulk interfaces are strongly
#   reflecting ($R \to 1$ as $\gamma_D/\omega \to 0$).
# - At $\omega = \omega_{\text{ENZ}}$, $\operatorname{Re}\varepsilon(\omega_{\text{ENZ}}) = 0$
#   and $\varepsilon(\omega_{\text{ENZ}}) = -i\varepsilon_\infty \gamma_D / \omega_{\text{ENZ}}$:
#   the phase velocity diverges while the normal displacement component continuity
#   enhances longitudinal fields.
# - For $\omega > \omega_{\text{ENZ}}$, $\operatorname{Re}\varepsilon(\omega) > 0$:
#   the medium crosses its **plasma reflection edge** and becomes an
#   underdense dielectric that transmits propagating waves.
#
# ### 2.3 Debye model: orientational relaxation of polar molecules
#
# In polar liquids such as water ($\text{H}_2\text{O}$) and biological tissues at
# RF/microwave frequencies, permanent molecular dipoles reorient in a viscous
# thermal bath where inertia is negligible compared to friction ($m^*\ddot{\mathbf{x}} \to 0$).
# The polarization relaxes toward its equilibrium value $\varepsilon_0\Delta\varepsilon_p\mathbf{E}$
# with rotational correlation time $\tau_p > 0$ (`Debye(delta_eps, tau)`):
#
# $$\tau_p \dot{\mathbf{P}}_p + \mathbf{P}_p
#   = \varepsilon_0 \Delta\varepsilon_p \mathbf{E}(t),$$
#
# $$\chi_{\text{Debye}}(\omega)
#   = \frac{\Delta\varepsilon_p}{1 + i\omega\tau_p}
#   = \frac{\Delta\varepsilon_p}{1 + (\omega\tau_p)^2}
#     - i\,\frac{\Delta\varepsilon_p\,\omega\tau_p}{1 + (\omega\tau_p)^2},
#   \qquad
#   \chi_{\text{Debye}}(t)
#   = \Theta(t)\,\frac{\Delta\varepsilon_p}{\tau_p}\,e^{-t/\tau_p}.$$
#
# At the **Debye relaxation frequency** $f_{\text{peak}} = 1 / (2\pi\tau_p)$
# ($\omega\tau_p = 1$), $\operatorname{Re}\varepsilon(\omega)$ drops halfway from
# its static value $\varepsilon_s = \varepsilon_\infty + \Delta\varepsilon$ to
# $\varepsilon_\infty$, and the dielectric loss reaches its characteristic
# symmetric log-frequency peak
# $\varepsilon''(\omega_{\text{peak}}) = -\operatorname{Im}\varepsilon(\omega_{\text{peak}}) = \Delta\varepsilon / 2$.
# For liquid water at $20\,^\circ\text{C}$ ($\varepsilon_\infty = 4.9$,
# $\varepsilon_s = 80.1$, $\Delta\varepsilon = 75.2$, $\tau = 9.3\;\text{ps}$),
# this loss peak sits at $f_{\text{peak}} = 17.11\;\text{GHz}$, which is why K-band
# and 2.45 GHz microwave ovens efficiently heat water!
#
# ### 2.4 Interband transitions: why gold (`Au`) departs from Drude below $\sim 550\;\text{nm}$
#
# In a noble metal (`Cu`, `Ag`, `Au`), conduction occurs in the half-filled $s$-$p$
# band while a completely filled $d$ band lies just below the Fermi level $E_F$.
# In silver (`Ag`), the $4d \to 5sp$ interband threshold lies in the ultraviolet
# at $\hbar\omega_{\text{ib}} \approx 3.9\;\text{eV}$ ($\lambda \approx 318\;\text{nm}$),
# so Ag behaves as a clean Drude metal across the entire visible spectrum and
# reflects all visible colors equally (silvery-white appearance).
#
# In **gold (`Au`)**, relativistic contraction of the $6s$ orbital lowers the gap
# between the top of the $5d$ band and $E_F$ near the $L$ and $X$ critical points
# of the face-centered-cubic Brillouin zone to $\hbar\omega_{\text{ib}} \approx 2.25\text{–}2.4\;\text{eV}$
# ($\lambda \approx 520\text{–}550\;\text{nm}$). Above this photon energy
# ($\lambda \lesssim 550\;\text{nm}$, green/blue/violet light), photons promote
# bound $5d$ electrons into unoccupied $6sp$ states above $E_F$. In the
# experimental **Johnson & Christy (1972)** [3] dataset (`tests/data/refractiveindex_info/Au_Johnson.yml`),
# this interband absorption causes $-\operatorname{Im}\varepsilon(\omega)$ to surge
# upward below $550\;\text{nm}$ instead of falling as $\propto \omega^{-3}$ like a
# pure Drude metal, and pulls normal-incidence reflectance down from $R > 0.95$ in
# the red/IR to $R \approx 0.38$ in the blue/green—giving gold its warm yellow color.
# Following **Rakić *et al.* (1998)** [2], `dispersive_lib["au"]` represents gold
# across $0.2\text{–}2.0\;\mu\text{m}$ as one intraband `Drude` pole plus five
# `Lorentz` oscillators capturing the $5d \to 6sp$ interband critical points.
#
# ---
#
# ## 3. Causality, Kramers–Kronig relations & the $f$-sum rule
#
# ### 3.1 Causality and the Kramers–Kronig Hilbert transform
#
# Because $\chi(t) = 0$ for $t < 0$, the complex susceptibility
# $\chi(\omega) = \varepsilon(\omega)/\varepsilon_0 - \varepsilon_\infty - \sigma/(i\omega\varepsilon_0)
# = \chi'(\omega) - i\,\chi''(\omega)$ (with $\chi'' = -\operatorname{Im}\chi \ge 0$
# in the $\exp(+i\omega t)$ convention) is analytic in the entire lower half of the
# complex $\omega$-plane ($\operatorname{Im}\omega < 0$; all poles of passive
# Lorentz, Drude, and Debye susceptibilities lie in the upper half-plane
# $\operatorname{Im}\omega > 0$). Applying Cauchy's theorem along the real axis
# together with conjugate symmetry $\chi(-\omega) = \chi^*(\omega)$ (since $\chi(t)$
# is real) yields the **Kramers–Kronig (KK) dispersion relations** (**Jackson** §7.10 [1]):
#
# $$\chi'(\omega)
#   = \operatorname{Re}\chi(\omega)
#   = \frac{2}{\pi}\,\mathcal{P}\!\int_0^\infty
#     \frac{\omega'\,\chi''(\omega')}{\omega'^2 - \omega^2}\,d\omega'
#   = -\frac{2}{\pi}\,\mathcal{P}\!\int_0^\infty
#     \frac{\omega'\,\operatorname{Im}\chi(\omega')}{\omega'^2 - \omega^2}\,d\omega',$$
#
# $$\chi''(\omega)
#   = -\operatorname{Im}\chi(\omega)
#   = -\frac{2\omega}{\pi}\,\mathcal{P}\!\int_0^\infty
#     \frac{\chi'(\omega') - \chi'(\omega)}{\omega'^2 - \omega^2}\,d\omega',$$
#
# where $\mathcal{P}$ denotes the Cauchy principal value. An immediate physical
# corollary is that **no passive medium can be both dispersive and lossless at all
# frequencies**: setting $\chi'(\omega) = \text{const}$ forces $\chi''(\omega) = 0$,
# and setting $\omega = 0$ in the first relation yields the static permittivity sum rule
#
# $$\varepsilon_{\text{st}} - \varepsilon_\infty
#   = \chi'(0)
#   = \frac{2}{\pi}\int_0^\infty \frac{\chi''(\omega')}{\omega'}\,d\omega' > 0
#   \qquad (\text{whenever }\chi''(\omega') \ge 0\text{ is not identically zero}).$$
#
# ### 3.2 High-frequency asymptotics and the oscillator-strength ($f$-sum) rule
#
# At frequencies far above every bound resonance and collision rate
# ($\omega \gg \omega_{0,p}, \gamma_p$), every bound and free charge responds as a
# free inertial particle ($\ddot{\mathbf{x}} \approx q\mathbf{E}/m^*$), so
# $\chi(\omega) \sim -\omega_{p,\text{tot}}^2 / \omega^2$ where
#
# $$\omega_{p,\text{tot}}^2
#   = \omega_{p,\text{Drude}}^2 + \sum_{p \in \text{Lorentz}} \Delta\varepsilon_p\,\omega_{0,p}^2.$$
#
# Taking the $\omega \to \infty$ limit of $\omega^2 \chi'(\omega)$ in the first
# Kramers–Kronig integral yields the **Thomas–Reiche–Kuhn (TRK) / oscillator-strength
# $f$-sum rule** (**Jackson** §7.10 [1]):
#
# $$\int_0^\infty \omega\,\chi''(\omega)\,d\omega
#   = -\int_0^\infty \omega\,\operatorname{Im}\chi(\omega)\,d\omega
#   = \frac{\pi}{2}\,\omega_{p,\text{tot}}^2
#   = \frac{\pi}{2}\left(\omega_{p,\text{Drude}}^2 + \sum_{p \in \text{Lorentz}} \Delta\varepsilon_p\,\omega_{0,p}^2\right).$$
#
# Notice that each Lorentz pole contributes $\frac{\pi}{2}\Delta\varepsilon_p\omega_{0,p}^2$
# **independently of its damping $\gamma_p$**: increasing $\gamma_p$ broadens and
# lowers the absorption peak $\Delta\varepsilon_p \omega_{0,p}/\gamma_p$ while
# conserving the integrated area $\int_0^\infty \omega\,\chi''_{\text{Lor}}(\omega)\,d\omega$!
#
# ---
#
# ## 4. The Auxiliary Differential Equation (ADE) in Wakis
#
# Directly evaluating the history integral $\int_0^\infty \chi(\tau)\mathbf{E}(t-\tau)\,d\tau$
# at every timestep would require $O(N_t)$ storage and work per grid cell. Instead,
# `wakis.dispersive` writes every pole in terms of normalized polarization
# $\mathbf{p}_p = \mathbf{P}_p / \varepsilon_0$ (**Taflove & Hagness** Ch. 9 [6];
# see `docs/physicsguide.md` §"Dispersive materials"):
#
# - **Lorentz / Drude (`Lorentz(delta_eps, omega_0, gamma)`, `Drude(omega_p, gamma)`):**
#   Centered differences at $t^n$ for $\ddot{\mathbf{p}}_p + \gamma_p \dot{\mathbf{p}}_p + \omega_{0,p}^2 \mathbf{p}_p = s_p \mathbf{E}$
#   (with strength $s_p = \Delta\varepsilon_p \omega_{0,p}^2$ for Lorentz and $s_p = \omega_p^2$ for Drude, $\omega_{0,p}=0$) give
#   $$\mathbf{p}_p^{n+1} = \alpha_p \mathbf{p}_p^n + \xi_p \mathbf{p}_p^{n-1} + \zeta_p \mathbf{E}^n,
#     \qquad
#     \alpha_p = \frac{2 - (\omega_{0,p}\Delta t)^2}{1 + \gamma_p\Delta t / 2}, \quad
#     \xi_p = \frac{\gamma_p\Delta t / 2 - 1}{1 + \gamma_p\Delta t / 2}, \quad
#     \zeta_p = \frac{s_p \Delta t^2}{1 + \gamma_p\Delta t / 2}.$$
# - **Debye (`Debye(delta_eps, tau)`):**
#   Crank–Nicolson averaging at $t^{n+1/2}$ for $\tau_p \dot{\mathbf{p}}_p + \mathbf{p}_p = \Delta\varepsilon_p \mathbf{E}$
#   with $r_p = \Delta t / (2\tau_p)$ gives
#   $$\mathbf{p}_p^{n+1} = \kappa_p \mathbf{p}_p^n + \beta_p \left(\mathbf{E}^{n+1} + \mathbf{E}^n\right),
#     \qquad
#     \kappa_p = \frac{1 - r_p}{1 + r_p}, \quad
#     \beta_p = \frac{\Delta\varepsilon_p r_p}{1 + r_p}.$$
#
# Substituting a discrete harmonic mode $\mathbf{E}^n = \hat{\mathbf{E}} e^{i\omega n\Delta t}$
# into the coupled Ampère–Maxwell and ADE difference equations yields the **exact
# discrete permittivity** implemented in `DispersiveMaterial.epsilon_discrete(f, dt)`:
#
# $$\varepsilon_{\text{disc}}(\omega, \Delta t)
#   = \varepsilon_\infty
#   + \frac{\sigma \cos(\omega\Delta t / 2)}{i\varepsilon_0\,\tilde{\omega}}
#   + \sum_{p \in \text{Lor,Dru}}
#     \frac{s_p}{\omega_{0,p}^2 - \tilde{\omega}^2 + i\gamma_p \sin(\omega\Delta t)/\Delta t}
#   + \sum_{p \in \text{Debye}}
#     \frac{\Delta\varepsilon_p}{1 + i(2\tau_p/\Delta t)\tan(\omega\Delta t / 2)},
#   \qquad
#   \tilde{\omega} = \frac{2}{\Delta t}\sin\!\left(\frac{\omega\Delta t}{2}\right).$$
#
# Expanding in powers of $\omega\Delta t \ll 1$ shows that
# $\varepsilon_{\text{disc}}(\omega, \Delta t) = \varepsilon(\omega) + O(\Delta t^2)$:
# both the central-difference second-order update and the Crank–Nicolson first-order
# update are **second-order accurate in time**. Moreover, for a lossless Lorentz
# oscillator coupled to the electromagnetic grid at Courant number $S$,
# `DispersiveMaterial.is_stable(dt, courant=S)` verifies that all eigenvalues of
# the one-step amplification matrix `_amplification_matrix(mat, dt, q)` remain on or
# inside the unit circle ($|\lambda_i| \le 1$), which requires
# $(\omega_0\Delta t / 2)^2 \le (\varepsilon_\infty - S^2) / (\varepsilon_s - S^2)$
# (and $\omega_0\Delta t \le 2$ for the uncoupled pole recursion).

# %%
import argparse
import json
import os
from pathlib import Path
import resource
import shlex
import sys
import time
from types import SimpleNamespace
import warnings

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c0
from scipy.constants import e as q_e
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import hbar

from wakis import FluxMonitor, GridFIT3D, SolverFIT3D
from wakis.dispersive import (
    Debye,
    DispersiveMaterial,
    Drude,
    Lorentz,
    _ade_coefficients,
    _amplification_matrix,
    dispersive_lib,
    eps_from_nk,
)
import wakis.figstyle as figstyle
import wakis.material_library as _matlib  # noqa: F401
import wakis.schematic as schematic
from wakis.sources import CurrentSource, GaussianSource

warnings.filterwarnings("ignore", message="IProgress not found")
figstyle.apply_style()

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_ROOT = HERE.parents[2] if (HERE.name == "02-01_dispersion_physics") else Path.cwd()
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
JC_AU_PATH = REPO_ROOT / "tests" / "data" / "refractiveindex_info" / "Au_Johnson.yml"


# %%
def load_refractiveindex_yml(yml_path):
    """
    Parse a ``refractiveindex.info`` YAML file (``tabulated nk`` format)
    without requiring PyYAML, returning ``(wl_m, n, k, eps_complex)`` in the
    ``exp(+i*omega*t)`` convention (``Im(eps) <= 0``).

    Parameters
    ----------
    yml_path : str or Path
        Path to the ``refractiveindex.info`` YAML file.

    Returns
    -------
    wl_m : ndarray
        Vacuum wavelengths in meters [m], sorted in ascending order.
    n_tab : ndarray
        Real refractive index ``n`` (dimensionless).
    k_tab : ndarray
        Extinction coefficient ``k >= 0`` (dimensionless).
    eps_tab : ndarray
        Complex relative permittivity ``(n - 1j*k)**2`` (dimensionless).
    """
    text = Path(yml_path).read_text(encoding="utf-8")
    in_data = False
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("data:"):
            in_data = True
            continue
        if in_data:
            if not s or (not s[0].isdigit() and not s[0] == "."):
                if rows:
                    break
                continue
            parts = s.split()
            if len(parts) == 3:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    arr = np.asarray(rows, dtype=np.float64)
    order = np.argsort(arr[:, 0])
    wl_m = arr[order, 0] * 1e-6
    n_tab = arr[order, 1]
    k_tab = arr[order, 2]
    eps_tab = eps_from_nk(n_tab, k_tab)
    return wl_m, n_tab, k_tab, eps_tab


def transfer_matrix_slab(freqs, d, eps_r, mu_r=1.0):
    """
    Exact normal-incidence complex amplitude ``(r, t)`` and power ``(R, T, A)``
    of a homogeneous slab of thickness ``d`` [m] in vacuum under the
    ``exp(+i*omega*t)`` convention (passive: ``Im(eps_r) <= 0``, ``Im(mu_r) <= 0``).

    Parameters
    ----------
    freqs : array_like
        Frequencies in Hz [Hz].
    d : float
        Slab thickness in meters [m].
    eps_r, mu_r : array_like or float
        Complex relative permittivity and permeability at ``freqs``.

    Returns
    -------
    dict
        Dictionary with keys ``r``, ``t``, ``R``, ``T``, ``A``.
    """
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64)
    eps_r = np.asarray(eps_r, dtype=np.complex128)
    mu_r = np.broadcast_to(np.asarray(mu_r, dtype=np.complex128), eps_r.shape)
    n = np.sqrt(eps_r * mu_r)
    n = np.where(np.imag(n) > 0.0, -n, n)
    eta = np.sqrt(mu_r / eps_r)
    eta = np.where(np.real(eta) < 0.0, -eta, eta)
    r01 = (eta - 1.0) / (eta + 1.0)
    p1 = np.exp(-1j * omega * n * d / c0)
    p2 = p1 * p1
    denom = 1.0 - (r01 * r01) * p2
    r = r01 * (1.0 - p2) / denom
    t = (1.0 - r01 * r01) * p1 / denom
    R = np.abs(r) ** 2
    T = np.abs(t) ** 2
    A = 1.0 - R - T
    return {"r": r, "t": t, "R": R, "T": T, "A": A}


def susceptibility_kernel_analytic(pole, t):
    """
    Exact causal time-domain susceptibility kernel ``chi(t)`` [1/s] for a
    single ``Lorentz``, ``Drude``, or ``Debye`` pole at times ``t`` [s].

    Parameters
    ----------
    pole : Lorentz, Drude, or Debye
        Pole specification.
    t : array_like
        Time coordinates in seconds [s].

    Returns
    -------
    ndarray
        Susceptibility kernel ``chi(t)`` [1/s], vanishing for ``t < 0``.
    """
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros_like(t)
    pos = t >= 0.0
    tp = t[pos]
    if isinstance(pole, Drude):
        wp = float(pole.omega_p)
        g = float(pole.gamma)
        if g > 0.0:
            out[pos] = (wp * wp / g) * (1.0 - np.exp(-g * tp))
        else:
            out[pos] = (wp * wp) * tp
    elif isinstance(pole, Lorentz):
        w0 = float(pole.omega_0)
        g = float(pole.gamma)
        de = float(pole.delta_eps)
        disc = w0 * w0 - 0.25 * g * g
        if disc > 0.0:
            wd = np.sqrt(disc)
            out[pos] = (de * w0 * w0 / wd) * np.exp(-0.5 * g * tp) * np.sin(wd * tp)
        elif disc == 0.0:
            out[pos] = (de * w0 * w0) * tp * np.exp(-0.5 * g * tp)
        else:
            kappa = np.sqrt(-disc)
            out[pos] = (de * w0 * w0 / kappa) * np.exp(-0.5 * g * tp) * np.sinh(kappa * tp)
    elif isinstance(pole, Debye):
        tau = float(pole.tau)
        de = float(pole.delta_eps)
        out[pos] = (de / tau) * np.exp(-tp / tau)
    else:
        raise ValueError(f"Unsupported pole type: {type(pole).__name__}")
    return out


def kramers_kronig_re_from_im(omega_grid, im_chi_grid, omega_eval=None):
    """
    Compute ``Re[chi(omega)]`` from ``Im[chi(omega)]`` in the ``exp(+i*omega*t)``
    convention via singularity-subtracted Cauchy principal-value quadrature:

        Re[chi(w)] = -(2/pi) * P int_0^inf [w' * Im[chi(w')] / (w'^2 - w^2)] dw'

    Parameters
    ----------
    omega_grid : ndarray
        Positive angular-frequency grid ``[w_min, ..., w_max]`` [rad/s].
    im_chi_grid : ndarray
        Imaginary part ``Im[chi(w')] <= 0`` sampled on ``omega_grid``.
    omega_eval : ndarray, optional
        Evaluation points inside ``(w_min, w_max)`` [rad/s].

    Returns
    -------
    ndarray
        Reconstructed ``Re[chi(omega_eval)]``.
    """
    w = np.asarray(omega_grid, dtype=np.float64)
    im_vals = np.asarray(im_chi_grid, dtype=np.float64)
    if omega_eval is None:
        omega_eval = w
    omega_eval = np.asarray(omega_eval, dtype=np.float64)
    w_max = float(w[-1])
    w_min = float(w[0])
    im_at_eval = np.interp(omega_eval, w, im_vals)

    out = np.empty_like(omega_eval)
    for idx, w0 in enumerate(omega_eval):
        im0 = im_at_eval[idx]
        diff = w * im_vals - w0 * im0
        denom = w * w - w0 * w0
        safe = np.abs(denom) > 1e-12 * (w0 * w0 + 1.0)
        integrand = np.zeros_like(w)
        integrand[safe] = diff[safe] / denom[safe]
        if w_min < w0 < w_max:
            pv_tail = 0.5 * im0 * np.log(
                ((w_max - w0) * (w0 + w_min)) / ((w_max + w0) * (w0 - w_min))
            )
        else:
            pv_tail = 0.0
        integral = np.trapezoid(integrand, w) + pv_tail
        out[idx] = -(2.0 / np.pi) * integral
    return out


def kramers_kronig_im_from_re(omega_grid, re_chi_grid, omega_eval=None):
    """
    Compute ``Im[chi(omega)]`` from ``Re[chi(omega)]`` in the ``exp(+i*omega*t)``
    convention via singularity-subtracted Cauchy principal-value quadrature:

        Im[chi(w)] = +(2*w/pi) * P int_0^inf [(Re[chi(w')] - Re[chi(w)]) / (w'^2 - w^2)] dw'

    Parameters
    ----------
    omega_grid : ndarray
        Angular-frequency grid ``[w_min, ..., w_max]`` [rad/s].
    re_chi_grid : ndarray
        Real part ``Re[chi(w')]`` sampled on ``omega_grid``.
    omega_eval : ndarray, optional
        Evaluation points inside ``(w_min, w_max)`` [rad/s].

    Returns
    -------
    ndarray
        Reconstructed ``Im[chi(omega_eval)] <= 0``.
    """
    w = np.asarray(omega_grid, dtype=np.float64)
    re_vals = np.asarray(re_chi_grid, dtype=np.float64)
    if omega_eval is None:
        omega_eval = w
    omega_eval = np.asarray(omega_eval, dtype=np.float64)
    w_max = float(w[-1])
    w_min = float(w[0])
    re_at_eval = np.interp(omega_eval, w, re_vals)

    out = np.empty_like(omega_eval)
    for idx, w0 in enumerate(omega_eval):
        re0 = re_at_eval[idx]
        diff = re_vals - re0
        denom = w * w - w0 * w0
        safe = np.abs(denom) > 1e-12 * (w0 * w0 + 1.0)
        integrand = np.zeros_like(w)
        integrand[safe] = diff[safe] / denom[safe]
        if w_min < w0 < w_max:
            pv_tail = (re0 / (2.0 * w0)) * np.log(
                ((w_max - w0) * (w0 + w_min)) / ((w_max + w0) * (w0 - w_min))
            )
        else:
            pv_tail = 0.0
        integral = np.trapezoid(integrand, w) + pv_tail
        out[idx] = (2.0 * w0 / np.pi) * integral
    return out


def ade_amplification_spectral_radius(pole, eps_inf, dt, courant=0.5):
    """
    Compute both:
    1. ``rho_pole``: the uncoupled single-pole recursion root magnitude
       (``p^{n+1} = alpha p^n + xi p^{n-1}`` for Lorentz/Drude, which leaves the
       unit circle at ``omega_0 * dt = 2``), and
    2. ``rho_coupled``: the maximum eigenvalue magnitude of the full coupled
       electromagnetic + ADE amplification matrix ``_amplification_matrix(mat, dt, q)``
       over ``q in [0, 2*courant]``, which is what ``mat.is_stable(dt, courant)``
       certifies.

    Parameters
    ----------
    pole : Lorentz, Drude, or Debye
        Dispersive pole.
    eps_inf : float
        High-frequency relative permittivity ``eps_inf >= 1``.
    dt : float
        Timestep in seconds [s].
    courant : float, optional
        3D Courant number (default 0.5).

    Returns
    -------
    rho_pole : float
        Max root magnitude of the uncoupled pole recursion.
    rho_coupled : float
        Max eigenvalue magnitude of the coupled EM + ADE amplification matrix.
    """
    mat = DispersiveMaterial(eps_inf=eps_inf, poles=[pole])
    lorentz_e, debye_e = _ade_coefficients(mat, dt)
    if lorentz_e:
        alpha, xi, _ = lorentz_e[0]
        roots_pole = np.roots([1.0, -alpha, -xi])
        rho_pole = float(np.max(np.abs(roots_pole)))
    else:
        kappa, _ = debye_e[0]
        rho_pole = float(abs(kappa))

    rho_coupled = 0.0
    for q in np.linspace(0.0, 2.0 * courant, 41):
        A = _amplification_matrix(mat, dt, q)
        rho_coupled = max(rho_coupled, float(np.max(np.abs(np.linalg.eigvals(A)))))
    return rho_pole, rho_coupled


def run_single_cell_bulk_ade(material, dt, nt, f0, fwidth, freqs):
    """
    Drive a 1-cell periodic ``SolverFIT3D`` (``k = 0`` uniform bulk mode) with a
    broadband Gaussian current source ``J_x(t)``, record ``E_x(t)`` and the
    impressed displacement ``D_x(t) = -int J_x(t) dt``, and extract both the
    measured permittivity ``eps_FIT(f) = D_x(f) / (eps_0 * E_x(f))`` and the
    time-domain polarization ``P_x(t) = D_x(t) - eps_0 * eps_inf * E_x(t)``.
    """
    grid = GridFIT3D(-0.01, 0.01, -0.01, 0.01, -0.01, 0.01, 1, 1, 1, verbose=0)
    solver = SolverFIT3D(
        grid,
        dt=dt,
        bc_low=["periodic", "periodic", "periodic"],
        bc_high=["periodic", "periodic", "periodic"],
        verbose=0,
    )
    solver.add_dispersive_material(
        material, center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0)
    )

    t_in = (np.arange(nt) + 0.5) * dt
    t0 = 4.5 / (2.0 * np.pi * fwidth)
    sig_t = 1.0 / (2.0 * np.pi * fwidth)
    j_trace = (
        -((t_in - t0) / sig_t)
        * np.exp(-0.5 * ((t_in - t0) / sig_t) ** 2)
        * np.cos(2.0 * np.pi * f0 * (t_in - t0))
    )

    e_trace = np.zeros(nt, dtype=np.float64)
    d_trace = np.zeros(nt, dtype=np.float64)
    d_running = 0.0
    for n in range(nt):
        solver.J[0, 0, 0, "x"] = j_trace[n]
        solver.one_step()
        d_running -= dt * j_trace[n]
        e_trace[n] = float(solver.E[0, 0, 0, "x"])
        d_trace[n] = d_running

    t_e = (np.arange(nt) + 1.0) * dt
    p_trace = d_trace - eps_0 * material.eps_inf * e_trace
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64)
    phase = np.exp(-1j * omega[:, None] * t_e[None, :])
    e_hat = np.sum(e_trace[None, :] * phase, axis=1) * dt
    d_hat = np.sum(d_trace[None, :] * phase, axis=1) * dt
    eps_fit = d_hat / (eps_0 * e_hat)
    eps_disc = material.epsilon_discrete(freqs, dt)
    eps_cont = material.epsilon(freqs)

    rel_err_disc = float(
        np.linalg.norm(eps_fit - eps_disc) / np.linalg.norm(eps_disc)
    )
    rel_err_cont = float(
        np.linalg.norm(eps_fit - eps_cont) / np.linalg.norm(eps_cont)
    )
    return {
        "t": t_e,
        "e_trace": e_trace,
        "d_trace": d_trace,
        "p_trace": p_trace,
        "eps_fit": eps_fit,
        "eps_disc": eps_disc,
        "eps_cont": eps_cont,
        "rel_err_disc": rel_err_disc,
        "rel_err_cont": rel_err_cont,
    }


def run_dispersive_slab_pair(
    material,
    freqs,
    thickness,
    lz,
    dz,
    f0,
    fwidth,
    t_after_src,
    z_src,
    z_r,
    z_t,
    cfln=0.5,
):
    """
    Run paired quasi-1D ``SolverFIT3D`` simulations (dispersive slab + vacuum
    reference) with periodic transverse boundaries and 15-cell CPML absorbers.
    """
    nz = int(round(lz / dz))
    zmin = -0.45 * lz
    zmax = zmin + nz * dz
    dxy = 4.0 * dz
    freqs = np.asarray(freqs, dtype=np.float64)

    def _run_one(mat):
        grid = GridFIT3D(-dxy, dxy, -dxy, dxy, zmin, zmax, 2, 2, nz, verbose=0)
        solver = SolverFIT3D(
            grid,
            cfln=cfln,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            n_pml=15,
            verbose=0,
        )
        if mat is not None:
            solver.add_dispersive_material(
                mat,
                center=(0.0, 0.0, 0.5 * thickness),
                size=(10.0 * dxy, 10.0 * dxy, thickness),
            )
        src_time = GaussianSource(frequency=f0, fwidth=fwidth)
        sheet = CurrentSource(
            src_time,
            component="x",
            center=(0.0, 0.0, z_src),
            size=(2.0 * dxy, 2.0 * dxy, 0.0),
        )
        planes = [
            FluxMonitor(
                freqs, center=(0.0, 0.0, z0), size=(2.0 * dxy, 2.0 * dxy, 0.0)
            )
            for z0 in (z_r, z_t)
        ]
        z_centers = 0.5 * (grid.z[:-1] + grid.z[1:])
        kr = int(np.argmin(np.abs(z_centers - z_r)))
        kt = int(np.argmin(np.abs(z_centers - z_t)))
        nt = int(round((src_time.end_time + t_after_src) / solver.dt))

        ex_r = np.zeros(nt, dtype=np.float64)
        ex_t = np.zeros(nt, dtype=np.float64)

        for n in range(nt):
            t_now = n * solver.dt
            solver.J[:, :, :, "x"] = 0.0
            sheet.update(solver, t_now)
            solver.one_step()
            for mon in planes:
                mon.update(solver, t_now)
            ex_r[n] = float(solver.E[0, 0, kr, "x"])
            ex_t[n] = float(solver.E[0, 0, kt, "x"])

        return (
            [p.get_flux() for p in planes],
            ex_r,
            ex_t,
            float(solver.dt),
            int(nt),
            float(z_centers[kr]),
            float(z_centers[kt]),
        )

    (flux_r_ref, flux_t_ref), ex_r_ref, ex_t_ref, dt, nt, zr_act, zt_act = _run_one(
        None
    )
    (flux_r, flux_t), ex_r_slab, ex_t_slab, _, _, _, _ = _run_one(material)

    R_flux = 1.0 - flux_r / flux_r_ref
    T_flux = flux_t / flux_t_ref
    A_flux = 1.0 - R_flux - T_flux

    t_arr = (np.arange(nt) + 1.0) * dt
    omega = 2.0 * np.pi * freqs
    phase = np.exp(-1j * omega[:, None] * t_arr[None, :])
    e_inc_r = np.sum(ex_r_ref[None, :] * phase, axis=1)
    e_ref_r = np.sum((ex_r_slab - ex_r_ref)[None, :] * phase, axis=1)
    e_inc_t = np.sum(ex_t_ref[None, :] * phase, axis=1)
    e_trn_t = np.sum(ex_t_slab[None, :] * phase, axis=1)

    R_field = np.abs(e_ref_r / e_inc_r) ** 2
    T_field = np.abs(e_trn_t / e_inc_t) ** 2

    eps_disc = material.epsilon_discrete(freqs, dt)
    eps_cont = material.epsilon(freqs)
    tm_disc = transfer_matrix_slab(freqs, thickness, eps_disc, 1.0)
    tm_cont = transfer_matrix_slab(freqs, thickness, eps_cont, 1.0)

    rms_R_disc = float(np.sqrt(np.mean((R_flux - tm_disc["R"]) ** 2)))
    rms_T_disc = float(np.sqrt(np.mean((T_flux - tm_disc["T"]) ** 2)))
    rms_R_cont = float(np.sqrt(np.mean((R_flux - tm_cont["R"]) ** 2)))
    rms_T_cont = float(np.sqrt(np.mean((T_flux - tm_cont["T"]) ** 2)))

    n_complex = np.sqrt(eps_cont)
    n_complex = np.where(np.imag(n_complex) > 0.0, -n_complex, n_complex)
    n_max = float(np.max(np.maximum(np.real(n_complex), 1.0)))
    lam_min_vac = float(c0 / np.max(freqs))
    lam_min_mat = lam_min_vac / n_max
    cpw_vac = lam_min_vac / dz
    cpw_mat = lam_min_mat / dz
    cells_across = thickness / dz

    return {
        "freqs": freqs,
        "dt": dt,
        "nt": nt,
        "nz": nz,
        "dz": dz,
        "thickness": thickness,
        "t_arr": t_arr,
        "ex_r_ref": ex_r_ref,
        "ex_r_slab": ex_r_slab,
        "ex_t_ref": ex_t_ref,
        "ex_t_slab": ex_t_slab,
        "R_fit": R_flux,
        "T_fit": T_flux,
        "A_fit": A_flux,
        "R_field": R_field,
        "T_field": T_field,
        "R_disc": tm_disc["R"],
        "T_disc": tm_disc["T"],
        "A_disc": tm_disc["A"],
        "R_cont": tm_cont["R"],
        "T_cont": tm_cont["T"],
        "A_cont": tm_cont["A"],
        "eps_disc": eps_disc,
        "eps_cont": eps_cont,
        "rms_R_disc": rms_R_disc,
        "rms_T_disc": rms_T_disc,
        "rms_R_cont": rms_R_cont,
        "rms_T_cont": rms_T_cont,
        "dx_background": float(dz),
        "dx_min": float(dz),
        "cells_across_feature": float(cells_across),
        "cpw": float(cpw_mat),
        "cpw_vacuum": float(cpw_vac),
        "n_cells": int(4 * nz),
        "n_steps": int(2 * nt),
    }


# %% [markdown]
# ## 5. Study tasks, resolution audit & preflight setup
#
# The tutorial executes 5 self-contained tasks (`task_0` through `task_4`) in both
# `--quick` and `--full` modes:
#
# | Task ID | Name | Physical system & validation target |
# |---|---|---|
# | `0` | `bulk_ade_and_sum_rules` | Single-cell periodic (`k = 0`) ADE impulse/Gaussian response for Lorentz, Drude, and Debye poles vs. $\chi(t)$ and `epsilon_discrete(f, dt)`; numerical Kramers–Kronig Hilbert transform on analytical poles and Johnson & Christy Au data; TRK $f$-sum rule verification. |
# | `1` | `lorentz_reststrahlen_slab` | Polar-crystal **wurtzite GaN** IR optical-phonon Reststrahlen slab ($\tilde{\nu}_{\text{TO}} = 560\;\text{cm}^{-1}$, $\tilde{\nu}_{\text{LO}} = 746\;\text{cm}^{-1}$, $d = 2.5\;\mu\text{m}$, **Barker & Ilegems 1973** [4]) and fused-silica (`sio2`) UV/IR Sellmeier dispersion: FIT $R(f), T(f), A(f)$ vs. transfer matrix across $10\text{–}28\;\text{THz}$. |
# | `2` | `drude_enz_slab` | Free-carrier **Drude** slab ($\varepsilon_\infty = 4.0$, screened ENZ frequency $f_{\text{ENZ}} = 2.0\;\text{THz}$, $d = 40\;\mu\text{m}$) across the plasma reflection edge ($0.8\text{–}3.5\;\text{THz}$), plus **gold (`Au`)** pure Drude vs. 5-Lorentz interband model (`dispersive_lib["au"]`) vs. **Johnson & Christy (1972)** [3] below $\lambda \approx 550\;\text{nm}$. |
# | `3` | `debye_water_slab` | Orientational **Debye** relaxation slab of liquid water at $20\,^\circ\text{C}$ ($\varepsilon_\infty = 4.9$, $\varepsilon_s = 80.1$, $\tau = 9.3\;\text{ps}$, loss peak $f_{\text{peak}} = 17.11\;\text{GHz}$, $d = 0.5\;\text{mm}$) across $2\text{–}40\;\text{GHz}$: FIT $R(f), T(f), A(f)$ vs. transfer matrix. |
# | `4` | `dt_convergence_and_stability` | Second-order $O(\Delta t^2)$ temporal convergence of `epsilon_discrete(f, dt)` and FIT slab $R(f)$ as $\Delta t$ is refined across 5 grid levels, plus the ADE characteristic root locus $|\lambda(\omega_0\Delta t)|$ across the leapfrog stability ceiling (`is_stable(dt)`). |
#
# Every spatial slab task enforces `cpw >= 12` cells per minimum wavelength inside
# the dispersive medium (`cpw_vacuum >= 50`) and `cells_across_feature >= 20`
# across the slab thickness $d$.


# %%
def get_task_table(quick=False):
    """
    Return the 5-task study table for Tutorial 2.1 in ``--quick`` or ``--full``
    configuration.
    """
    return [
        {
            "index": 0,
            "name": "bulk_ade_and_sum_rules",
            "kind": "bulk_and_kk",
            "nt_bulk": 2000 if quick else 3200,
            "n_kk": 1500 if quick else 3000,
            "dx_background": 0.002,
            "dx_min": 0.002,
            "cells_across_feature": 10.0,
            "cpw": 50.0,
            "description": (
                "Single-cell ADE bulk response (Lorentz/Drude/Debye), "
                "Kramers-Kronig Hilbert transform & TRK f-sum rule"
            ),
        },
        {
            "index": 1,
            "name": "lorentz_reststrahlen_slab",
            "kind": "gan_slab",
            "thickness": 2.5e-6,
            "lz": 16.0e-6,
            "dz": 0.05e-6 if quick else 0.025e-6,
            "fmin": 10.0e12,
            "fmax": 28.0e12,
            "n_freqs": 81 if quick else 141,
            "description": (
                "Wurtzite GaN IR Reststrahlen phonon Lorentz slab (d=2.5 um, "
                "10-28 THz) & SiO2 Sellmeier dispersion"
            ),
        },
        {
            "index": 2,
            "name": "drude_enz_slab",
            "kind": "drude_slab",
            "thickness": 40.0e-6,
            "lz": 320.0e-6,
            "dz": 1.0e-6 if quick else 0.5e-6,
            "fmin": 0.8e12,
            "fmax": 3.5e12,
            "n_freqs": 81 if quick else 141,
            "description": (
                "Drude free-carrier slab across ENZ & plasma edge (d=40 um, "
                "f_ENZ=2.0 THz) + Au Drude vs interband J&C"
            ),
        },
        {
            "index": 3,
            "name": "debye_water_slab",
            "kind": "debye_slab",
            "thickness": 0.5e-3,
            "lz": 6.0e-3,
            "dz": 20.0e-6 if quick else 10.0e-6,
            "fmin": 2.0e9,
            "fmax": 40.0e9,
            "n_freqs": 77 if quick else 153,
            "description": (
                "Liquid water Debye relaxation slab (d=0.5 mm, tau=9.3 ps, "
                "f_peak=17.1 GHz) across 2-40 GHz"
            ),
        },
        {
            "index": 4,
            "name": "dt_convergence_and_stability",
            "kind": "dt_study",
            "dz_list": (
                [3.2e-3, 2.4e-3, 1.8e-3, 1.35e-3]
                if quick
                else [3.2e-3, 2.4e-3, 1.8e-3, 1.35e-3, 1.0e-3]
            ),
            "thickness": 0.04,
            "lz": 0.48,
            "description": (
                "Second-order O(dt^2) discrete vs continuous ADE convergence "
                "& pole stability boundary (omega_0 * dt = 2)"
            ),
        },
    ]


def run_single_task(task_cfg, out_dir=DATA_DIR):
    """Execute one study task and write ``task_<I>.npz`` and ``task_<I>.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = int(task_cfg["index"])
    name = str(task_cfg["name"])
    kind = str(task_cfg["kind"])
    t0 = time.perf_counter()

    if kind == "bulk_and_kk":
        dt_bulk = 10.0e-12  # 10 ps step for GHz canonical poles
        nt_bulk = int(task_cfg["nt_bulk"])
        freqs_ghz = np.linspace(0.3e9, 4.0e9, 120)

        lor_pole = Lorentz(3.0, 2.0 * np.pi * 2.0e9, 2.0 * np.pi * 0.25e9)
        dru_pole = Drude(2.0 * np.pi * 2.5e9, 2.0 * np.pi * 0.30e9)
        deb_pole = Debye(4.0, 1.0 / (2.0 * np.pi * 1.5e9))

        mat_lor = DispersiveMaterial(eps_inf=1.5, poles=[lor_pole])
        mat_dru = DispersiveMaterial(eps_inf=1.5, poles=[dru_pole])
        mat_deb = DispersiveMaterial(eps_inf=1.5, poles=[deb_pole])

        res_lor = run_single_cell_bulk_ade(
            mat_lor, dt_bulk, nt_bulk, f0=2.0e9, fwidth=1.2e9, freqs=freqs_ghz
        )
        res_dru = run_single_cell_bulk_ade(
            mat_dru, dt_bulk, nt_bulk, f0=2.0e9, fwidth=1.2e9, freqs=freqs_ghz
        )
        res_deb = run_single_cell_bulk_ade(
            mat_deb, dt_bulk, nt_bulk, f0=2.0e9, fwidth=1.2e9, freqs=freqs_ghz
        )

        # Analytical susceptibility kernels chi(t)
        t_kern = np.linspace(0.0, 4.0e-9, 500)
        chi_t_lor = susceptibility_kernel_analytic(lor_pole, t_kern)
        chi_t_dru = susceptibility_kernel_analytic(dru_pole, t_kern)
        chi_t_deb = susceptibility_kernel_analytic(deb_pole, t_kern)

        # Discrete impulse response via 1-step delta E^0 = 1/dt kick
        def _discrete_impulse_kernel(pole, dt_val, n_steps):
            mat_tmp = DispersiveMaterial(eps_inf=1.0, poles=[pole])
            lorentz_e, debye_e = _ade_coefficients(mat_tmp, dt_val)
            p_hist = np.zeros(n_steps, dtype=np.float64)
            p_nm1 = 0.0
            p_n = 0.0
            for step in range(n_steps):
                if lorentz_e:
                    alpha, xi, zeta = lorentz_e[0]
                    e_n = (1.0 / dt_val) if step == 0 else 0.0
                    p_np1 = alpha * p_n + xi * p_nm1 + zeta * e_n
                else:
                    kappa, beta = debye_e[0]
                    e_sum = (1.0 / dt_val) if step in (0, 1) else 0.0
                    p_np1 = kappa * p_n + beta * e_sum
                p_nm1 = p_n
                p_n = p_np1
                p_hist[step] = p_n
            return p_hist

        n_kern_steps = int(round(4.0e-9 / dt_bulk))
        t_kern_disc = (np.arange(n_kern_steps) + 1.0) * dt_bulk
        chi_disc_lor = _discrete_impulse_kernel(lor_pole, dt_bulk, n_kern_steps)
        chi_disc_dru = _discrete_impulse_kernel(dru_pole, dt_bulk, n_kern_steps)
        chi_disc_deb = _discrete_impulse_kernel(deb_pole, dt_bulk, n_kern_steps)

        # Kramers-Kronig check on a two-pole Lorentz + Debye model
        p1 = Lorentz(2.4, 2.0 * np.pi * 1.8e9, 2.0 * np.pi * 0.35e9)
        p2 = Lorentz(1.6, 2.0 * np.pi * 3.4e9, 2.0 * np.pi * 0.45e9)
        p3 = Debye(1.5, 1.0 / (2.0 * np.pi * 0.8e9))
        mat_kk = DispersiveMaterial(eps_inf=2.0, poles=[p1, p2, p3])

        n_kk = int(task_cfg["n_kk"])
        f_kk_grid = np.linspace(1e6, 40.0e9, n_kk)
        w_kk_grid = 2.0 * np.pi * f_kk_grid
        chi_kk_grid = mat_kk.epsilon(f_kk_grid) - mat_kk.eps_inf

        f_kk_eval = np.linspace(0.15e9, 6.0e9, 120)
        w_kk_eval = 2.0 * np.pi * f_kk_eval
        chi_kk_exact = mat_kk.epsilon(f_kk_eval) - mat_kk.eps_inf
        re_kk_num = kramers_kronig_re_from_im(
            w_kk_grid, np.imag(chi_kk_grid), w_kk_eval
        )
        im_kk_num = kramers_kronig_im_from_re(
            w_kk_grid, np.real(chi_kk_grid), w_kk_eval
        )

        kk_re_rel_err = float(
            np.linalg.norm(re_kk_num - np.real(chi_kk_exact))
            / np.linalg.norm(np.real(chi_kk_exact))
        )
        kk_im_rel_err = float(
            np.linalg.norm(im_kk_num - np.imag(chi_kk_exact))
            / np.linalg.norm(np.imag(chi_kk_exact))
        )

        # Kramers-Kronig on tabulated Johnson & Christy Au data (finite window)
        wl_au, n_au, k_au, eps_au = load_refractiveindex_yml(JC_AU_PATH)
        f_au = c0 / wl_au
        order_f = np.argsort(f_au)
        f_au = f_au[order_f]
        w_au = 2.0 * np.pi * f_au
        eps_au = eps_au[order_f]
        w_au_uni = np.linspace(w_au[0], w_au[-1], 800)
        im_au_uni = np.interp(w_au_uni, w_au, np.imag(eps_au))
        w_au_eval = w_au[4:-4]
        re_au_kk_window = 1.0 + kramers_kronig_re_from_im(
            w_au_uni, im_au_uni, w_au_eval
        )

        mat_au_lib = dispersive_lib["au"]
        f_au_wide = np.linspace(1e12, 3.0e16, 4000)
        w_au_wide = 2.0 * np.pi * f_au_wide
        chi_au_wide = mat_au_lib.epsilon(f_au_wide) - mat_au_lib.eps_inf
        re_au_kk_full = mat_au_lib.eps_inf + kramers_kronig_re_from_im(
            w_au_wide, np.imag(chi_au_wide), w_au_eval
        )
        re_au_lib_eval = np.real(mat_au_lib.epsilon(w_au_eval / (2.0 * np.pi)))

        # TRK f-sum rule check on multi-Lorentz + Drude model (indep. of damping)
        mat_fsum_low_g = DispersiveMaterial(
            eps_inf=1.0,
            poles=[
                Drude(2.0 * np.pi * 1.2e9, 2.0 * np.pi * 0.15e9),
                Lorentz(2.5, 2.0 * np.pi * 2.2e9, 2.0 * np.pi * 0.20e9),
                Lorentz(1.8, 2.0 * np.pi * 3.6e9, 2.0 * np.pi * 0.30e9),
            ],
        )
        mat_fsum_high_g = DispersiveMaterial(
            eps_inf=1.0,
            poles=[
                Drude(2.0 * np.pi * 1.2e9, 2.0 * np.pi * 0.55e9),
                Lorentz(2.5, 2.0 * np.pi * 2.2e9, 2.0 * np.pi * 0.80e9),
                Lorentz(1.8, 2.0 * np.pi * 3.6e9, 2.0 * np.pi * 1.10e9),
            ],
        )
        wp_tot_sq_exact = (2.0 * np.pi) ** 2 * (
            1.2e9**2 + 2.5 * (2.2e9**2) + 1.8 * (3.6e9**2)
        )
        f_sum_grid = np.geomspace(1e5, 5.0e12, 6000)
        w_sum_grid = 2.0 * np.pi * f_sum_grid
        im_low = -np.imag(mat_fsum_low_g.epsilon(f_sum_grid))
        im_high = -np.imag(mat_fsum_high_g.epsilon(f_sum_grid))
        dw = np.diff(w_sum_grid)
        integrand_low = w_sum_grid * im_low
        integrand_high = w_sum_grid * im_high
        cum_low = np.concatenate(
            [
                [0.0],
                (2.0 / np.pi)
                * np.cumsum(0.5 * (integrand_low[:-1] + integrand_low[1:]) * dw),
            ]
        )
        cum_high = np.concatenate(
            [
                [0.0],
                (2.0 / np.pi)
                * np.cumsum(0.5 * (integrand_high[:-1] + integrand_high[1:]) * dw),
            ]
        )
        fsum_rel_err_low = float(
            abs(cum_low[-1] - wp_tot_sq_exact) / wp_tot_sq_exact
        )
        fsum_rel_err_high = float(
            abs(cum_high[-1] - wp_tot_sq_exact) / wp_tot_sq_exact
        )

        npz_data = {
            "freqs_ghz": freqs_ghz,
            "t_bulk": res_lor["t"],
            "e_lor": res_lor["e_trace"],
            "p_lor": res_lor["p_trace"],
            "eps_fit_lor": res_lor["eps_fit"],
            "eps_disc_lor": res_lor["eps_disc"],
            "eps_cont_lor": res_lor["eps_cont"],
            "eps_fit_dru": res_dru["eps_fit"],
            "eps_disc_dru": res_dru["eps_disc"],
            "eps_cont_dru": res_dru["eps_cont"],
            "eps_fit_deb": res_deb["eps_fit"],
            "eps_disc_deb": res_deb["eps_disc"],
            "eps_cont_deb": res_deb["eps_cont"],
            "t_kern": t_kern,
            "chi_t_lor": chi_t_lor,
            "chi_t_dru": chi_t_dru,
            "chi_t_deb": chi_t_deb,
            "t_kern_disc": t_kern_disc,
            "chi_disc_lor": chi_disc_lor,
            "chi_disc_dru": chi_disc_dru,
            "chi_disc_deb": chi_disc_deb,
            "f_kk_eval": f_kk_eval,
            "chi_kk_exact": chi_kk_exact,
            "re_kk_num": re_kk_num,
            "im_kk_num": im_kk_num,
            "w_au": w_au,
            "eps_au": eps_au,
            "w_au_eval": w_au_eval,
            "re_au_kk_window": re_au_kk_window,
            "re_au_kk_full": re_au_kk_full,
            "re_au_lib_eval": re_au_lib_eval,
            "f_sum_grid": f_sum_grid,
            "cum_fsum_low": cum_low / wp_tot_sq_exact,
            "cum_fsum_high": cum_high / wp_tot_sq_exact,
            "im_fsum_low": im_low,
            "im_fsum_high": im_high,
        }
        meta = {
            "task_index": idx,
            "name": name,
            "dx_background": float(task_cfg["dx_background"]),
            "dx_min": float(task_cfg["dx_min"]),
            "cells_across_feature": float(task_cfg["cells_across_feature"]),
            "cpw": float(task_cfg["cpw"]),
            "n_cells": 3,
            "n_steps": int(3 * nt_bulk),
            "lor_bulk_rel_err_disc": res_lor["rel_err_disc"],
            "dru_bulk_rel_err_disc": res_dru["rel_err_disc"],
            "deb_bulk_rel_err_disc": res_deb["rel_err_disc"],
            "kk_re_rel_err": kk_re_rel_err,
            "kk_im_rel_err": kk_im_rel_err,
            "fsum_rel_err_low_gamma": fsum_rel_err_low,
            "fsum_rel_err_high_gamma": fsum_rel_err_high,
        }

    elif kind == "gan_slab":
        f_to = 560.0 * 100.0 * c0
        f_lo = 746.0 * 100.0 * c0
        f_gam = 17.0 * 100.0 * c0
        eps_inf_gan = 5.35
        delta_eps_gan = eps_inf_gan * ((f_lo / f_to) ** 2 - 1.0)
        mat_gan = DispersiveMaterial(
            eps_inf=eps_inf_gan,
            poles=[
                Lorentz(
                    delta_eps_gan,
                    2.0 * np.pi * f_to,
                    2.0 * np.pi * f_gam,
                )
            ],
        )

        freqs = np.linspace(task_cfg["fmin"], task_cfg["fmax"], task_cfg["n_freqs"])
        thickness = float(task_cfg["thickness"])
        lz = float(task_cfg["lz"])
        dz = float(task_cfg["dz"])
        slab_res = run_dispersive_slab_pair(
            mat_gan,
            freqs=freqs,
            thickness=thickness,
            lz=lz,
            dz=dz,
            f0=19.0e12,
            fwidth=6.5e12,
            t_after_src=2.2e-12,
            z_src=-4.5e-6,
            z_r=-2.2e-6,
            z_t=thickness + 2.2e-6,
            cfln=0.5,
        )

        mat_sio2 = dispersive_lib["sio2"]
        wl_sio2 = np.geomspace(0.18e-6, 15.0e-6, 500)
        f_sio2 = c0 / wl_sio2
        eps_sio2 = mat_sio2.epsilon(f_sio2)

        in_reststrahlen = (freqs >= f_to) & (freqs <= f_lo)
        peak_R_reststrahlen = float(np.max(slab_res["R_fit"][in_reststrahlen]))

        npz_data = {
            **{
                k: v
                for k, v in slab_res.items()
                if isinstance(v, np.ndarray)
            },
            "f_to": np.array(f_to),
            "f_lo": np.array(f_lo),
            "wl_sio2": wl_sio2,
            "eps_sio2": eps_sio2,
        }
        meta = {
            "task_index": idx,
            "name": name,
            "f_TO_THz": f_to * 1e-12,
            "f_LO_THz": f_lo * 1e-12,
            "eps_inf": eps_inf_gan,
            "eps_st": eps_inf_gan + delta_eps_gan,
            "lst_ratio_sq": (f_lo / f_to) ** 2,
            "peak_R_reststrahlen": peak_R_reststrahlen,
            "rms_R_disc": slab_res["rms_R_disc"],
            "rms_T_disc": slab_res["rms_T_disc"],
            "rms_R_cont": slab_res["rms_R_cont"],
            "rms_T_cont": slab_res["rms_T_cont"],
            "dx_background": slab_res["dx_background"],
            "dx_min": slab_res["dx_min"],
            "cells_across_feature": slab_res["cells_across_feature"],
            "cpw": slab_res["cpw"],
            "cpw_vacuum": slab_res["cpw_vacuum"],
            "n_cells": slab_res["n_cells"],
            "n_steps": slab_res["n_steps"],
        }

    elif kind == "drude_slab":
        eps_inf_d = 4.0
        f_enz = 2.0e12
        f_gam = 0.20e12
        f_p = np.sqrt(eps_inf_d * (f_enz**2 + f_gam**2))
        mat_drude = DispersiveMaterial(
            eps_inf=eps_inf_d,
            poles=[Drude(2.0 * np.pi * f_p, 2.0 * np.pi * f_gam)],
        )

        freqs = np.linspace(task_cfg["fmin"], task_cfg["fmax"], task_cfg["n_freqs"])
        thickness = float(task_cfg["thickness"])
        lz = float(task_cfg["lz"])
        dz = float(task_cfg["dz"])
        slab_res = run_dispersive_slab_pair(
            mat_drude,
            freqs=freqs,
            thickness=thickness,
            lz=lz,
            dz=dz,
            f0=2.15e12,
            fwidth=0.95e12,
            t_after_src=12.0e-12,
            z_src=-90.0e-6,
            z_r=-45.0e-6,
            z_t=thickness + 45.0e-6,
            cfln=0.5,
        )

        wl_jc, n_jc, k_jc, eps_jc = load_refractiveindex_yml(JC_AU_PATH)
        mat_au_full = dispersive_lib["au"]
        drude_pole_au = [p for p in mat_au_full.poles if isinstance(p, Drude)][0]
        mat_au_drude_only = DispersiveMaterial(
            eps_inf=mat_au_full.eps_inf,
            poles=[drude_pole_au],
        )
        wl_opt = np.linspace(0.20e-6, 1.05e-6, 300)
        f_opt = c0 / wl_opt
        eps_au_full = mat_au_full.epsilon(f_opt)
        eps_au_drude = mat_au_drude_only.epsilon(f_opt)

        def _bulk_reflectance(eps_arr):
            n_c = np.sqrt(eps_arr)
            n_c = np.where(np.imag(n_c) > 0.0, -n_c, n_c)
            return np.abs((n_c - 1.0) / (n_c + 1.0)) ** 2

        R_au_full = _bulk_reflectance(eps_au_full)
        R_au_drude = _bulk_reflectance(eps_au_drude)
        R_au_jc = _bulk_reflectance(eps_jc)

        npz_data = {
            **{
                k: v
                for k, v in slab_res.items()
                if isinstance(v, np.ndarray)
            },
            "f_enz": np.array(f_enz),
            "f_p": np.array(f_p),
            "wl_jc": wl_jc,
            "eps_jc": eps_jc,
            "R_au_jc": R_au_jc,
            "wl_opt": wl_opt,
            "eps_au_full": eps_au_full,
            "eps_au_drude": eps_au_drude,
            "R_au_full": R_au_full,
            "R_au_drude": R_au_drude,
        }
        meta = {
            "task_index": idx,
            "name": name,
            "f_ENZ_THz": f_enz * 1e-12,
            "f_p_THz": float(f_p * 1e-12),
            "rms_R_disc": slab_res["rms_R_disc"],
            "rms_T_disc": slab_res["rms_T_disc"],
            "rms_R_cont": slab_res["rms_R_cont"],
            "rms_T_cont": slab_res["rms_T_cont"],
            "dx_background": slab_res["dx_background"],
            "dx_min": slab_res["dx_min"],
            "cells_across_feature": slab_res["cells_across_feature"],
            "cpw": slab_res["cpw"],
            "cpw_vacuum": slab_res["cpw_vacuum"],
            "n_cells": slab_res["n_cells"],
            "n_steps": slab_res["n_steps"],
        }

    elif kind == "debye_slab":
        eps_inf_w = 4.9
        delta_eps_w = 75.2
        tau_w = 9.3e-12
        f_peak = 1.0 / (2.0 * np.pi * tau_w)
        mat_water = DispersiveMaterial(
            eps_inf=eps_inf_w,
            poles=[Debye(delta_eps_w, tau_w)],
        )

        freqs = np.linspace(task_cfg["fmin"], task_cfg["fmax"], task_cfg["n_freqs"])
        thickness = float(task_cfg["thickness"])
        lz = float(task_cfg["lz"])
        dz = float(task_cfg["dz"])
        slab_res = run_dispersive_slab_pair(
            mat_water,
            freqs=freqs,
            thickness=thickness,
            lz=lz,
            dz=dz,
            f0=18.0e9,
            fwidth=12.0e9,
            t_after_src=0.35e-9,
            z_src=-1.4e-3,
            z_r=-0.7e-3,
            z_t=thickness + 0.7e-3,
            cfln=0.5,
        )

        npz_data = {
            **{
                k: v
                for k, v in slab_res.items()
                if isinstance(v, np.ndarray)
            },
            "f_peak": np.array(f_peak),
            "tau_w": np.array(tau_w),
        }
        meta = {
            "task_index": idx,
            "name": name,
            "f_peak_GHz": f_peak * 1e-9,
            "tau_ps": tau_w * 1e12,
            "rms_R_disc": slab_res["rms_R_disc"],
            "rms_T_disc": slab_res["rms_T_disc"],
            "rms_R_cont": slab_res["rms_R_cont"],
            "rms_T_cont": slab_res["rms_T_cont"],
            "dx_background": slab_res["dx_background"],
            "dx_min": slab_res["dx_min"],
            "cells_across_feature": slab_res["cells_across_feature"],
            "cpw": slab_res["cpw"],
            "cpw_vacuum": slab_res["cpw_vacuum"],
            "n_cells": slab_res["n_cells"],
            "n_steps": slab_res["n_steps"],
        }

    elif kind == "dt_study":
        pole_lor = Lorentz(2.5, 2.0 * np.pi * 1.5e9, 2.0 * np.pi * 0.25e9)
        pole_dru = Drude(2.0 * np.pi * 1.8e9, 2.0 * np.pi * 0.20e9)
        pole_deb = Debye(3.0, 1.0 / (2.0 * np.pi * 1.2e9))
        mat_lor = DispersiveMaterial(eps_inf=2.0, poles=[pole_lor])
        mat_dru = DispersiveMaterial(eps_inf=2.0, poles=[pole_dru])
        mat_deb = DispersiveMaterial(eps_inf=2.0, poles=[pole_deb])

        f_eval = np.linspace(0.5e9, 2.5e9, 81)
        dt_sweep = np.geomspace(2.0e-12, 80.0e-12, 25)
        err_eps_lor = np.zeros_like(dt_sweep)
        err_eps_dru = np.zeros_like(dt_sweep)
        err_eps_deb = np.zeros_like(dt_sweep)

        eps_lor_cont = mat_lor.epsilon(f_eval)
        eps_dru_cont = mat_dru.epsilon(f_eval)
        eps_deb_cont = mat_deb.epsilon(f_eval)
        for i, dti in enumerate(dt_sweep):
            err_eps_lor[i] = np.linalg.norm(
                mat_lor.epsilon_discrete(f_eval, dti) - eps_lor_cont
            ) / np.linalg.norm(eps_lor_cont)
            err_eps_dru[i] = np.linalg.norm(
                mat_dru.epsilon_discrete(f_eval, dti) - eps_dru_cont
            ) / np.linalg.norm(eps_dru_cont)
            err_eps_deb[i] = np.linalg.norm(
                mat_deb.epsilon_discrete(f_eval, dti) - eps_deb_cont
            ) / np.linalg.norm(eps_deb_cont)

        slope_eps_lor = float(
            np.polyfit(np.log(dt_sweep[:12]), np.log(err_eps_lor[:12]), 1)[0]
        )
        slope_eps_dru = float(
            np.polyfit(np.log(dt_sweep[:12]), np.log(err_eps_dru[:12]), 1)[0]
        )
        slope_eps_deb = float(
            np.polyfit(np.log(dt_sweep[:12]), np.log(err_eps_deb[:12]), 1)[0]
        )

        dz_list = [float(x) for x in task_cfg["dz_list"]]
        dt_slab_list = []
        rms_R_cont_list = []
        rms_R_disc_list = []
        n_cells_tot = 0
        n_steps_tot = 0
        cpw_min = 1e9
        cells_across_min = 1e9

        for dz_val in dz_list:
            sr = run_dispersive_slab_pair(
                mat_lor,
                freqs=f_eval,
                thickness=float(task_cfg["thickness"]),
                lz=float(task_cfg["lz"]),
                dz=dz_val,
                f0=1.5e9,
                fwidth=0.65e9,
                t_after_src=8.0e-9,
                z_src=-0.12,
                z_r=-0.06,
                z_t=float(task_cfg["thickness"]) + 0.06,
                cfln=0.5,
            )
            dt_slab_list.append(sr["dt"])
            rms_R_cont_list.append(sr["rms_R_cont"])
            rms_R_disc_list.append(sr["rms_R_disc"])
            n_cells_tot += sr["n_cells"]
            n_steps_tot += sr["n_steps"]
            cpw_min = min(cpw_min, sr["cpw"])
            cells_across_min = min(cells_across_min, sr["cells_across_feature"])

        slope_slab_R = float(
            np.polyfit(np.log(dt_slab_list), np.log(rms_R_cont_list), 1)[0]
        )

        w0_dt_grid = np.linspace(0.05, 2.35, 120)
        w0_ref = 2.0 * np.pi * 1.5e9
        pole_undamped = Lorentz(2.0, w0_ref, 2.0 * np.pi * 0.02e9)
        mat_stab = DispersiveMaterial(eps_inf=2.0, poles=[pole_undamped])

        rho_pole_arr = np.zeros_like(w0_dt_grid)
        rho_coupled_arr = np.zeros_like(w0_dt_grid)
        is_stable_arr = np.zeros(len(w0_dt_grid), dtype=bool)
        for i, wdt in enumerate(w0_dt_grid):
            dti = wdt / w0_ref
            rp, rc = ade_amplification_spectral_radius(
                pole_undamped, 2.0, dti, courant=0.5
            )
            rho_pole_arr[i] = rp
            rho_coupled_arr[i] = rc
            is_stable_arr[i] = bool(mat_stab.is_stable(dti, courant=0.5))

        npz_data = {
            "dt_sweep": dt_sweep,
            "w0_dt_sweep": w0_ref * dt_sweep,
            "err_eps_lor": err_eps_lor,
            "err_eps_dru": err_eps_dru,
            "err_eps_deb": err_eps_deb,
            "dz_list": np.asarray(dz_list),
            "dt_slab_list": np.asarray(dt_slab_list),
            "rms_R_cont_list": np.asarray(rms_R_cont_list),
            "rms_R_disc_list": np.asarray(rms_R_disc_list),
            "w0_dt_grid": w0_dt_grid,
            "rho_pole_arr": rho_pole_arr,
            "rho_coupled_arr": rho_coupled_arr,
            "is_stable_arr": is_stable_arr.astype(np.int8),
        }
        meta = {
            "task_index": idx,
            "name": name,
            "slope_eps_lor": slope_eps_lor,
            "slope_eps_dru": slope_eps_dru,
            "slope_eps_deb": slope_eps_deb,
            "slope_slab_R": slope_slab_R,
            "rms_R_cont_coarse": float(rms_R_cont_list[0]),
            "rms_R_cont_fine": float(rms_R_cont_list[-1]),
            "dx_background": float(max(dz_list)),
            "dx_min": float(min(dz_list)),
            "cells_across_feature": float(cells_across_min),
            "cpw": float(cpw_min),
            "n_cells": int(n_cells_tot),
            "n_steps": int(n_steps_tot),
        }
    else:
        raise ValueError(f"Unknown task kind: {kind}")

    elapsed = time.perf_counter() - t0
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    meta["elapsed_s"] = round(elapsed, 3)
    meta["peak_rss_mb"] = round(peak_rss_mb, 2)

    np.savez_compressed(out_dir / f"task_{idx}.npz", **npz_data)
    (out_dir / f"task_{idx}.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[task_{idx}] {name}: elapsed={elapsed:.2f}s, "
        f"cpw={meta['cpw']:.1f}, cells_across={meta['cells_across_feature']:.1f}"
    )
    return meta


# %%
def _fix_schematic_units(ax):
    """Ensure a thin space separates numbers and units inside math-mode schematic labels."""
    import re

    for txt in ax.texts:
        s = txt.get_text()
        s_new = re.sub(
            r"(\d)(mm|cm|nm|m|GHz|MHz|THz|kHz|ps|ns|fs)\b",
            r"\1\\;\\mathrm{\2}",
            s,
        )
        s_new = re.sub(r"(\d)\\mu m", r"\1\\;\\mu\\mathrm{m}", s_new)
        if s_new != s:
            txt.set_text(s_new)


def _plot_setup_schematic(fig_dir):
    """Render ``00_setup_schematic.png`` showing both Tutorial 2.1 setups."""
    fig, (ax_bulk, ax_slab) = figstyle.new_figure(
        nrows=1, ncols=2, width=12.8, height=4.8
    )

    # Panel (a): Bulk 1-cell periodic ADE testbench (Task 0)
    grid_bulk = GridFIT3D(
        -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, 1, 1, 1, verbose=0
    )
    solver_bulk = SolverFIT3D(
        grid_bulk,
        dt=10.0e-12,
        bc_low=["periodic", "periodic", "periodic"],
        bc_high=["periodic", "periodic", "periodic"],
        verbose=0,
    )
    bulk_struct = SimpleNamespace(
        bounds=((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)),
        name="Homogeneous ADE cell",
        material="DispersiveMaterial",
    )
    src_bulk = CurrentSource(
        GaussianSource(frequency=2.0e9, fwidth=1.2e9),
        component="x",
        center=(0.0, 0.0, 0.0),
        size=(0.02, 0.02, 0.0),
    )
    schematic.plot_schematic(
        solver_bulk,
        plane="zx",
        ax=ax_bulk,
        structures=[bulk_struct],
        sources=[src_bulk],
        probes=[{"pos": (0.0, 0.0, 0.0045), "label": "E_x, P_x(t) probe"}],
        show_mesh=False,
        show_wave_packet=False,
        freq=2.0e9,
        title="(a) Bulk 1-cell periodic ADE setup (Task 0)",
    )
    ax_bulk.set_yticks([])
    ax_bulk.set_ylabel("Transverse x (periodic, 1 cell)")
    _fix_schematic_units(ax_bulk)

    # Panel (b): Quasi-1D dispersive slab R/T testbench (Tasks 1-4)
    dz = 0.025e-6
    lz = 16.0e-6
    thickness = 2.5e-6
    nz = int(round(lz / dz))
    zmin = -0.45 * lz
    zmax = zmin + nz * dz
    dxy = 0.8e-6
    grid_slab = GridFIT3D(
        -dxy, dxy, -dxy, dxy, zmin, zmax, 2, 2, nz, verbose=0
    )
    solver_slab = SolverFIT3D(
        grid_slab,
        cfln=0.5,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=15,
        verbose=0,
    )
    slab_struct = SimpleNamespace(
        bounds=((-dxy, -dxy, 0.0), (dxy, dxy, thickness)),
        name="GaN Lorentz slab",
        material=DispersiveMaterial(
            eps_inf=5.35,
            poles=[Lorentz(4.142, 2.0 * np.pi * 16.79e12, 2.0 * np.pi * 0.51e12)],
        ),
    )
    src_slab = CurrentSource(
        GaussianSource(frequency=19.0e12, fwidth=6.5e12),
        component="x",
        center=(0.0, 0.0, -4.5e-6),
        size=(2.0 * dxy, 2.0 * dxy, 0.0),
    )
    mons_slab = [
        FluxMonitor(
            np.linspace(10.0e12, 28.0e12, 5),
            center=(0.0, 0.0, -2.2e-6),
            size=(2.0 * dxy, 2.0 * dxy, 0.0),
        ),
        FluxMonitor(
            np.linspace(10.0e12, 28.0e12, 5),
            center=(0.0, 0.0, thickness + 2.2e-6),
            size=(2.0 * dxy, 2.0 * dxy, 0.0),
        ),
    ]
    schematic.plot_schematic(
        solver_slab,
        plane="zx",
        ax=ax_slab,
        structures=[slab_struct],
        sources=[src_slab],
        monitors=mons_slab,
        show_mesh=True,
        mesh_every=8,
        mesh_zoom=True,
        freq=28.0e12,
        title="(b) Quasi-1D dispersive slab R/T setup (Tasks 1–4)",
    )
    ax_slab.set_yticks([])
    ax_slab.set_ylabel("Transverse x (periodic, 2 cells)")
    _fix_schematic_units(ax_slab)
    figstyle.save(fig, Path(fig_dir) / "00_setup_schematic.png")


def _plot_wikimedia_dielectric_mechanisms(fig_dir):
    """Render ``figures/wikimedia/dielectric_responses_wakis.png``."""
    wiki_dir = Path(fig_dir) / "wikimedia"
    wiki_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_spec, ax_phasor) = figstyle.new_figure(
        nrows=1, ncols=2, width=12.6, height=4.8
    )

    # Broadband frequency grid from 1 MHz to 30 PHz
    freqs = np.geomspace(1.0e6, 3.0e16, 1600)
    pole_dip = Debye(delta_eps=32.0, tau=1.0 / (2.0 * np.pi * 1.5e9))
    pole_ion = Lorentz(
        delta_eps=6.0,
        omega_0=2.0 * np.pi * 8.0e12,
        gamma=2.0 * np.pi * 1.1e12,
    )
    pole_el = Lorentz(
        delta_eps=2.5,
        omega_0=2.0 * np.pi * 3.0e15,
        gamma=2.0 * np.pi * 0.45e15,
    )
    mat_broad = DispersiveMaterial(
        eps_inf=1.0, poles=[pole_dip, pole_ion, pole_el]
    )
    eps_broad = mat_broad.epsilon(freqs)

    # Free-carrier Drude overlay at low frequencies
    mat_drude_low = DispersiveMaterial(
        eps_inf=1.0,
        poles=[Drude(omega_p=2.0 * np.pi * 0.5e11, gamma=2.0 * np.pi * 1.2e10)],
    )
    eps_drude_low = mat_drude_low.epsilon(freqs)

    ax_spec.plot(
        freqs,
        np.real(eps_broad),
        **figstyle.series_style(0, with_marker=False),
        label=r"$\varepsilon'(f)$ (storage permittivity)",
    )
    ax_spec.plot(
        freqs,
        -np.imag(eps_broad),
        **figstyle.series_style(1, with_marker=False),
        label=r"$\varepsilon''(f) = -\mathrm{Im}\,\varepsilon(f)$ (loss factor)",
    )
    ax_spec.plot(
        freqs[:600],
        np.clip(-np.imag(eps_drude_low[:600]), 0.0, 50.0),
        **figstyle.series_style(2, with_marker=False),
        label=r"Free-carrier `Drude` tail ($\varepsilon'' \propto f^{-1}$)",
    )
    ax_spec.axhline(1.0, color="#64748B", ls=":", lw=1.0)
    ax_spec.axhline(0.0, color="#94A3B8", ls=":", lw=0.8)

    # Shaded mechanism bands
    ax_spec.axvspan(1.0e8, 3.0e10, color="#0072B2", alpha=0.10)
    ax_spec.axvspan(1.5e12, 3.0e13, color="#009E73", alpha=0.11)
    ax_spec.axvspan(6.0e14, 1.2e16, color="#CC79A7", alpha=0.11)

    bbox_kw = {
        "boxstyle": "round,pad=0.22",
        "facecolor": "white",
        "edgecolor": "#CBD5E1",
        "alpha": 0.92,
    }
    ax_spec.text(
        1.5e9,
        41.5,
        "Orientational / dipolar\n(`Debye`: $\\tau \\dot{\\mathbf{P}} + \\mathbf{P} = \\varepsilon_0\\Delta\\varepsilon\\mathbf{E}$)\nMicrowave / GHz",
        fontsize=7.3,
        ha="center",
        va="top",
        bbox=bbox_kw,
    )
    ax_spec.text(
        7.5e12,
        29.0,
        "Ionic / IR phonon\n(`Lorentz`: TO/LO lattice)\nTHz Reststrahlen",
        fontsize=7.3,
        ha="center",
        va="top",
        bbox=bbox_kw,
    )
    ax_spec.text(
        2.5e15,
        21.0,
        "Electronic shell\n(`Lorentz`: UV exciton)\nVisible / UV",
        fontsize=7.3,
        ha="center",
        va="top",
        bbox=bbox_kw,
    )
    ax_spec.set_xscale("log")
    ax_spec.set_xlim(1.0e6, 3.0e16)
    ax_spec.set_ylim(-6.0, 46.0)
    ax_spec.set_xlabel("Frequency $f$ [Hz] (RF $\\rightarrow$ Microwave $\\rightarrow$ IR $\\rightarrow$ UV)")
    ax_spec.set_ylabel(r"Relative permittivity $\varepsilon'(f),\;\varepsilon''(f)$")
    ax_spec.set_title(
        "(a) Dielectric dispersion hierarchy mapped to Wakis poles"
    )
    ax_spec.legend(fontsize=7.6, loc="lower left")

    # Panel (b): Loss-tangent phasor diagram (after File:Loss_tangent_phasors_1.svg)
    ax_phasor.set_xlim(-0.3, 4.2)
    ax_phasor.set_ylim(-0.4, 3.8)
    ax_phasor.axhline(0.0, color="#94A3B8", lw=1.0)
    ax_phasor.axvline(0.0, color="#94A3B8", lw=1.0)

    j_loss = 1.25
    j_disp = 2.85
    # Vector arrows: J_loss (in-phase with E along +x), J_disp (quadrature +i along +y), J_tot
    ax_phasor.annotate(
        "",
        xy=(j_loss, 0.0),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "-|>", "lw": 2.3, "color": "#D55E00"},
    )
    ax_phasor.annotate(
        "",
        xy=(0.0, j_disp),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "-|>", "lw": 2.3, "color": "#0072B2"},
    )
    ax_phasor.annotate(
        "",
        xy=(j_loss, j_disp),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "-|>", "lw": 2.5, "color": "#009E73"},
    )
    ax_phasor.vlines(
        j_loss, 0.0, j_disp, colors="#64748B", linestyles="--", lw=1.1
    )
    ax_phasor.hlines(
        j_disp, 0.0, j_loss, colors="#64748B", linestyles="--", lw=1.1
    )

    # Loss angle delta arc from +y axis (reactive) toward J_tot
    from matplotlib.patches import Arc

    angle_deg = float(np.degrees(np.arctan2(j_disp, j_loss)))
    arc = Arc(
        (0.0, 0.0),
        2.3,
        2.3,
        angle=0.0,
        theta1=angle_deg,
        theta2=90.0,
        color="#1E293B",
        lw=1.5,
    )
    ax_phasor.add_patch(arc)
    ax_phasor.text(
        0.26,
        1.25,
        r"$\delta$",
        fontsize=10.0,
        fontweight="bold",
        color="#1E293B",
    )
    ax_phasor.text(
        j_loss * 0.52,
        -0.28,
        r"$J_{\mathrm{loss}} = \omega\varepsilon_0\varepsilon'' E_0$ (in phase with $\hat{\mathbf{E}}$)",
        color="#D55E00",
        fontsize=8.0,
        ha="center",
    )
    ax_phasor.text(
        0.10,
        j_disp + 0.14,
        r"$i J_{\mathrm{disp}} = i\omega\varepsilon_0\varepsilon' E_0$ (reactive $90^\circ$)",
        color="#0072B2",
        fontsize=8.0,
        ha="left",
    )
    ax_phasor.text(
        j_loss + 0.10,
        j_disp - 0.05,
        r"$\hat{\mathbf{J}}_{\mathrm{tot}} = i\omega\varepsilon_0(\varepsilon' - i\varepsilon'')\hat{\mathbf{E}}$",
        color="#009E73",
        fontsize=8.3,
        fontweight="bold",
        ha="left",
    )
    ax_phasor.text(
        1.55,
        1.05,
        "Loss tangent ($\\mathrm{e}^{+i\\omega t}$ convention):\n"
        "$\\tan\\delta = \\frac{\\varepsilon''(\\omega) + \\sigma/(\\omega\\varepsilon_0)}{\\varepsilon'(\\omega)} \\geq 0$\n\n"
        "Wakis ADE pole classes:\n"
        "• `Debye(delta_eps, tau)` — 1st-order CN\n"
        "• `Lorentz(delta_eps, w0, g)` — 2nd-order leapfrog\n"
        "• `Drude(wp, g)` — 2nd-order ($\\omega_0 = 0$)",
        fontsize=7.6,
        va="center",
        ha="left",
        bbox=bbox_kw,
    )
    ax_phasor.set_xlabel(r"In-phase dissipative axis ($\parallel \hat{\mathbf{E}}$, real power loss)")
    ax_phasor.set_ylabel(r"Quadrature reactive axis ($\parallel +i\hat{\mathbf{E}}$, stored energy)")
    ax_phasor.set_title(
        r"(b) Loss-tangent phasor diagram ($\tan\delta = \varepsilon''/\varepsilon'$)"
    )

    figstyle.save(fig, wiki_dir / "dielectric_responses_wakis.png")


def generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR):
    """Read ``data/task_*.npz`` and render setup, Wikimedia, and study figures."""
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figstyle.apply_style()

    _plot_setup_schematic(fig_dir)
    _plot_wikimedia_dielectric_mechanisms(fig_dir)

    d0 = np.load(data_dir / "task_0.npz")
    d1 = np.load(data_dir / "task_1.npz")
    d2 = np.load(data_dir / "task_2.npz")
    d3 = np.load(data_dir / "task_3.npz")
    d4 = np.load(data_dir / "task_4.npz")

    # Figure 1: Canonical Lorentz, Drude, Debye in frequency & time domain (3 panels + chi(t) insets)
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.6, height=4.3)
    f_ghz = d0["freqs_ghz"] * 1e-9
    t_ns = d0["t_kern"] * 1e9
    t_disc_ns = d0["t_kern_disc"] * 1e9

    models_f1 = [
        (
            r"(a) Lorentz oscillator ($f_0 = 2.0$ GHz)",
            d0["eps_cont_lor"],
            d0["eps_disc_lor"],
            d0["eps_fit_lor"],
            d0["chi_t_lor"] * 1e-9,
            d0["chi_disc_lor"] * 1e-9,
            2.0,
        ),
        (
            r"(b) Drude carriers ($f_p = 2.5$ GHz)",
            d0["eps_cont_dru"],
            d0["eps_disc_dru"],
            d0["eps_fit_dru"],
            d0["chi_t_dru"] * 1e-9,
            d0["chi_disc_dru"] * 1e-9,
            np.sqrt(2.5**2 / 1.5 - 0.30**2),
        ),
        (
            r"(c) Debye relaxation ($f_{\mathrm{peak}} = 1.5$ GHz)",
            d0["eps_cont_deb"],
            d0["eps_disc_deb"],
            d0["eps_fit_deb"],
            d0["chi_t_deb"] * 1e-9,
            d0["chi_disc_deb"] * 1e-9,
            1.5,
        ),
    ]
    for col, (title, eps_c, _eps_d, eps_f, chi_c, chi_d, f_mark) in enumerate(
        models_f1
    ):
        ax_f = axes[col]
        ax_f.plot(
            f_ghz,
            np.real(eps_c),
            **figstyle.series_style(0, with_marker=False),
            label=r"$\varepsilon'(f)$ exact",
        )
        ax_f.plot(
            f_ghz,
            -np.imag(eps_c),
            **figstyle.series_style(1, with_marker=False),
            label=r"$\varepsilon''(f) = -\mathrm{Im}\,\varepsilon \geq 0$",
        )
        ax_f.plot(
            f_ghz[::6],
            np.real(eps_f[::6]),
            "o",
            color=figstyle.PALETTE[0],
            ms=3.8,
            label=r"$\mathrm{Re}\,\varepsilon_{\mathrm{FIT}}(f)$ (1-cell ADE)",
        )
        ax_f.plot(
            f_ghz[::6],
            -np.imag(eps_f[::6]),
            "s",
            color=figstyle.PALETTE[1],
            ms=3.8,
            label=r"$-\mathrm{Im}\,\varepsilon_{\mathrm{FIT}}(f)$ (1-cell ADE)",
        )
        ax_f.axhline(0.0, color="#64748B", ls=":", lw=0.9)
        ax_f.axvline(f_mark, color="#475569", ls="--", lw=1.0)
        ax_f.set_title(title)
        ax_f.set_xlabel("Frequency $f$ [GHz]")
        if col == 0:
            ax_f.set_ylabel(r"Relative permittivity $\varepsilon(f)$")
            ax_f.legend(fontsize=7.3, loc="lower left")

        # Causal time-domain impulse kernel inset chi(t)
        ax_ins = ax_f.inset_axes([0.50, 0.54, 0.46, 0.40])
        ax_ins._is_inset = True
        ax_ins.plot(
            t_ns,
            chi_c,
            **figstyle.series_style(2, markevery=60, ms=2.8, lw=1.4),
            label=r"Analytic $\chi(t)$",
        )
        ax_ins.plot(
            t_disc_ns[::12],
            chi_d[::12],
            color="#1E293B",
            ls=":",
            marker="o",
            ms=2.8,
            lw=0.8,
            label="ADE impulse",
        )
        ax_ins.axhline(0.0, color="#94A3B8", ls=":", lw=0.7)
        ax_ins.set_title(r"Causal kernel $\chi(t)$ [$\mathrm{ns}^{-1}$]", fontsize=7.0)
        ax_ins.set_xlabel("$t$ [ns]", fontsize=6.8)
        ax_ins.tick_params(labelsize=6.5)

    figstyle.save(fig, fig_dir / "01_microscopic_models_freq_and_time.png")

    # Figure 2: Bound electrons (SiO2 Sellmeier) & IR phonons (GaN Reststrahlen)
    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.3)
    wl_sio2_um = d1["wl_sio2"] * 1e6
    eps_sio2 = d1["eps_sio2"]
    n_sio2 = np.sqrt(eps_sio2)
    n_sio2 = np.where(np.imag(n_sio2) > 0, -n_sio2, n_sio2)

    ax1.plot(
        wl_sio2_um,
        np.real(n_sio2),
        **figstyle.series_style(0, with_marker=False),
        label=r"$n'(\lambda)$",
    )
    ax1.plot(
        wl_sio2_um,
        -np.imag(n_sio2),
        **figstyle.series_style(1, with_marker=False),
        label=r"$\kappa(\lambda) = -\mathrm{Im}\,n$",
    )
    ax1.axvspan(
        0.4, 0.7, color="#E69F00", alpha=0.18, label=r"Visible window ($n \approx 1.46$)"
    )
    ax1.axvline(
        9.896, color="#D55E00", ls=":", lw=1.2, label=r"Si–O IR phonon ($9.90\;\mu\mathrm{m}$)"
    )
    ax1.set_xscale("log")
    ax1.set_ylim(-0.1, 3.2)
    ax1.set_xlabel(r"Vacuum wavelength $\lambda$ [$\mu$m]")
    ax1.set_ylabel(r"Refractive index $n', \kappa$")
    ax1.set_title(r"Fused silica ($\mathrm{SiO_2}$, Malitson 1965): UV electrons + IR mode")
    ax1.legend(fontsize=8.0)

    f_gan_thz = d1["freqs"] * 1e-12
    eps_gan = d1["eps_cont"]
    f_to_thz = float(d1["f_to"]) * 1e-12
    f_lo_thz = float(d1["f_lo"]) * 1e-12
    ax2.plot(
        f_gan_thz,
        np.real(eps_gan),
        **figstyle.series_style(0, with_marker=False),
        label=r"$\varepsilon'(f)$",
    )
    ax2.plot(
        f_gan_thz,
        -np.imag(eps_gan),
        **figstyle.series_style(1, with_marker=False),
        label=r"$\varepsilon''(f) = -\mathrm{Im}\,\varepsilon$",
    )
    ax2.axvspan(
        f_to_thz,
        f_lo_thz,
        color="#D55E00",
        alpha=0.14,
        label=rf"Reststrahlen band ($\varepsilon' < 0$: {f_to_thz:.1f}–{f_lo_thz:.1f} THz)",
    )
    ax2.axhline(0.0, color="#64748B", ls=":", lw=1.0)
    ax2.axvline(
        f_to_thz, color="#D55E00", ls="--", lw=1.1, label=r"$\omega_{\mathrm{TO}}$ ($560\;\mathrm{cm}^{-1}$)"
    )
    ax2.axvline(
        f_lo_thz, color="#009E73", ls="--", lw=1.1, label=r"$\omega_{\mathrm{LO}}$ ($746\;\mathrm{cm}^{-1}$)"
    )
    ax2.set_ylim(-25, 35)
    ax2.set_xlabel("Frequency $f$ [THz]")
    ax2.set_ylabel(r"Relative permittivity $\varepsilon(f)$")
    ax2.set_title("Wurtzite GaN optical phonon (Barker & Ilegems 1973)")
    ax2.legend(fontsize=7.8, loc="upper right")

    figstyle.save(fig, fig_dir / "02_bound_electrons_and_reststrahlen.png")

    # Figure 3: Gold (Au) Drude vs Interband transitions (Johnson & Christy)
    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.3)
    wl_opt_nm = d2["wl_opt"] * 1e9
    wl_jc_nm = d2["wl_jc"] * 1e9
    mask_jc = (wl_jc_nm >= 200.0) & (wl_jc_nm <= 1050.0)

    ax1.plot(
        wl_opt_nm,
        -np.imag(d2["eps_au_drude"]),
        **figstyle.series_style(0, with_marker=False),
        label=r"Intraband Drude only ($6sp$ free electrons)",
    )
    ax1.plot(
        wl_opt_nm,
        -np.imag(d2["eps_au_full"]),
        **figstyle.series_style(1, with_marker=False),
        label="Drude + 5 Lorentz poles (`dispersive_lib['au']`)",
    )
    ax1.plot(
        wl_jc_nm[mask_jc],
        -np.imag(d2["eps_jc"][mask_jc]),
        "ko",
        ms=4.2,
        label="Johnson & Christy (1972) measured Au",
    )
    ax1.axvspan(
        200,
        550,
        color="#CC79A7",
        alpha=0.14,
        label=r"$5d \to 6sp$ interband ($\lambda \leq 550$ nm)",
    )
    ax1.set_xlim(200, 1000)
    ax1.set_ylim(0, 12.0)
    ax1.set_xlabel(r"Vacuum wavelength $\lambda$ [nm]")
    ax1.set_ylabel(r"Loss factor $\varepsilon''(\lambda) = -\mathrm{Im}\,\varepsilon$")
    ax1.set_title(r"Gold ($\mathrm{Au}$) absorption: why pure Drude fails below $550$ nm")
    ax1.legend(fontsize=7.8, loc="upper right")

    ax2.plot(
        wl_opt_nm,
        d2["R_au_drude"],
        **figstyle.series_style(0, with_marker=False),
        label=r"Intraband Drude reflectance $R(\lambda)$",
    )
    ax2.plot(
        wl_opt_nm,
        d2["R_au_full"],
        **figstyle.series_style(1, with_marker=False),
        label=r"Drude + 5 Lorentz reflectance $R(\lambda)$",
    )
    ax2.plot(
        wl_jc_nm[mask_jc],
        d2["R_au_jc"][mask_jc],
        "ko",
        ms=4.2,
        label=r"Johnson & Christy (1972) measured $R(\lambda)$",
    )
    ax2.axvline(
        550.0,
        color="#D55E00",
        ls=":",
        lw=1.3,
        label=r"Interband edge $\lambda \approx 550$ nm ($2.25$ eV)",
    )
    ax2.set_xlim(200, 1000)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlabel(r"Vacuum wavelength $\lambda$ [nm]")
    ax2.set_ylabel("Normal-incidence bulk reflectance $R$")
    ax2.set_title("Origin of gold's yellow reflection edge")
    ax2.legend(fontsize=7.8, loc="lower right")

    figstyle.save(fig, fig_dir / "03_gold_drude_vs_interband_jc.png")

    # Figure 4: Kramers-Kronig Hilbert transform & TRK f-sum rule
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.6, height=4.3)
    f_kk = d0["f_kk_eval"] * 1e-9
    chi_kk = d0["chi_kk_exact"]
    axes[0].plot(
        f_kk,
        np.real(chi_kk),
        **figstyle.series_style(0, with_marker=False),
        label=r"Exact $\mathrm{Re}\,\chi(\omega)$",
    )
    axes[0].plot(
        f_kk[::4],
        d0["re_kk_num"][::4],
        "o",
        color=figstyle.PALETTE[0],
        ms=3.8,
        label=r"KK from $-\mathrm{Im}\,\chi(\omega')$",
    )
    axes[0].plot(
        f_kk,
        -np.imag(chi_kk),
        **figstyle.series_style(1, with_marker=False),
        label=r"Exact $-\mathrm{Im}\,\chi(\omega)$",
    )
    axes[0].plot(
        f_kk[::4],
        -d0["im_kk_num"][::4],
        "s",
        color=figstyle.PALETTE[1],
        ms=3.8,
        label=r"KK from $\mathrm{Re}\,\chi(\omega')$",
    )
    axes[0].axhline(0.0, color="#64748B", ls=":", lw=0.9)
    axes[0].set_xlabel("Frequency $f$ [GHz]")
    axes[0].set_ylabel(r"Susceptibility $\chi(\omega)$")
    axes[0].set_title("Kramers–Kronig on 2-Lorentz + Debye model")
    axes[0].legend(fontsize=7.5)

    ev_au = (hbar * d0["w_au"]) / q_e
    ev_eval = (hbar * d0["w_au_eval"]) / q_e
    axes[1].plot(
        ev_au,
        np.real(d0["eps_au"]),
        "ko",
        ms=3.8,
        label=r"Measured $\mathrm{Re}\,\varepsilon$ (J&C Au)",
    )
    axes[1].plot(
        ev_eval,
        d0["re_au_kk_full"],
        **figstyle.series_style(0, with_marker=False),
        label="Full-band KK (Drude+5L extrapolated)",
    )
    axes[1].plot(
        ev_eval,
        d0["re_au_kk_window"],
        **figstyle.series_style(1, with_marker=False),
        label="Truncated KK (J&C 0.64–6.6 eV only)",
    )
    axes[1].set_xlabel(r"Photon energy $\hbar\omega$ [eV]")
    axes[1].set_ylabel(r"$\mathrm{Re}\,\varepsilon(\omega)$")
    axes[1].set_title("KK on Au data & finite-window truncation")
    axes[1].legend(fontsize=7.5)

    f_sum_ghz = d0["f_sum_grid"] * 1e-9
    axes[2].plot(
        f_sum_ghz,
        d0["cum_fsum_low"],
        **figstyle.series_style(0, with_marker=False),
        label=r"Low damping ($\gamma_p \times 1$)",
    )
    axes[2].plot(
        f_sum_ghz,
        d0["cum_fsum_high"],
        **figstyle.series_style(1, with_marker=False),
        label=r"High damping ($\gamma_p \times 3.7$)",
    )
    axes[2].axhline(
        1.0,
        color="#1E293B",
        ls=":",
        lw=1.2,
        label=r"Exact TRK limit $\omega_{p,\mathrm{tot}}^2$",
    )
    axes[2].set_xscale("log")
    axes[2].set_xlim(0.1, 500.0)
    axes[2].set_ylim(0.0, 1.12)
    axes[2].set_xlabel(r"Upper cutoff frequency $\Omega / (2\pi)$ [GHz]")
    axes[2].set_ylabel(
        r"Normalized $f$-sum $\frac{2}{\pi\omega_{p,\mathrm{tot}}^2}\int_0^\Omega \omega\varepsilon''(\omega)d\omega$"
    )
    axes[2].set_title(r"Thomas–Reiche–Kuhn $f$-sum rule conservation")
    axes[2].legend(fontsize=7.5, loc="lower right")

    figstyle.save(fig, fig_dir / "04_kramers_kronig_and_f_sum_rule.png")

    # Figure 5: Lorentz Reststrahlen slab (GaN) R(f), T(f), A(f)
    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.3)
    f_gan = d1["freqs"] * 1e-12
    f_to_thz = float(d1["f_to"]) * 1e-12
    f_lo_thz = float(d1["f_lo"]) * 1e-12

    ax1.plot(
        f_gan,
        d1["R_disc"],
        **figstyle.series_style(0, with_marker=False),
        label=r"$R_{\mathrm{TM}}(\varepsilon_{\mathrm{disc}})$",
    )
    ax1.plot(
        f_gan,
        d1["T_disc"],
        **figstyle.series_style(2, with_marker=False),
        label=r"$T_{\mathrm{TM}}(\varepsilon_{\mathrm{disc}})$",
    )
    ax1.plot(
        f_gan,
        d1["A_disc"],
        **figstyle.series_style(1, with_marker=False),
        label=r"$A_{\mathrm{TM}} = 1 - R - T$",
    )
    ax1.plot(
        f_gan[::4],
        d1["R_fit"][::4],
        "o",
        color=figstyle.PALETTE[0],
        ms=3.8,
        label="FIT $R(f)$",
    )
    ax1.plot(
        f_gan[::4],
        d1["T_fit"][::4],
        "s",
        color=figstyle.PALETTE[2],
        ms=3.8,
        label="FIT $T(f)$",
    )
    ax1.plot(
        f_gan[::4],
        d1["A_fit"][::4],
        "^",
        color=figstyle.PALETTE[1],
        ms=3.8,
        label="FIT $A(f)$",
    )
    ax1.axvspan(
        f_to_thz, f_lo_thz, color="#D55E00", alpha=0.14, label="Reststrahlen band"
    )
    ax1.set_ylim(-0.03, 1.05)
    ax1.set_xlabel("Frequency $f$ [THz]")
    ax1.set_ylabel("Power reflectance / transmittance / absorptance")
    ax1.set_title(r"GaN Reststrahlen slab ($d = 2.5\;\mu\mathrm{m}$): FIT vs. analytic")
    ax1.legend(fontsize=7.6, ncol=2)

    t_ps = d1["t_arr"] * 1e12
    ax2.plot(
        t_ps,
        d1["ex_r_ref"],
        color="#64748B",
        lw=1.4,
        label=r"Incident pulse $E_{\mathrm{inc}}(z_r, t)$",
    )
    ax2.plot(
        t_ps,
        d1["ex_r_slab"] - d1["ex_r_ref"],
        **figstyle.series_style(0, with_marker=False),
        label=r"Reflected pulse $E_{\mathrm{refl}}(z_r, t)$",
    )
    ax2.plot(
        t_ps,
        d1["ex_t_slab"],
        **figstyle.series_style(2, with_marker=False),
        label=r"Transmitted pulse $E_{\mathrm{trans}}(z_t, t)$",
    )
    ax2.set_xlim(0.0, min(2.2, float(t_ps[-1])))
    ax2.set_xlabel("Time $t$ [ps]")
    ax2.set_ylabel("Electric field $E_x$ [arb. units]")
    ax2.set_title(r"Time-domain phonon ringing at $\omega_{\mathrm{TO}}$ and $\omega_{\mathrm{LO}}$")
    ax2.legend(fontsize=8.0)

    figstyle.save(fig, fig_dir / "05_lorentz_reststrahlen_slab_rt.png")

    # Figure 6: Drude ENZ & plasma reflection edge slab
    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.3)
    f_dr = d2["freqs"] * 1e-12
    f_enz_thz = float(d2["f_enz"]) * 1e-12

    ax1.plot(
        f_dr,
        np.real(d2["eps_cont"]),
        **figstyle.series_style(0, with_marker=False),
        label=r"$\mathrm{Re}\,\varepsilon(f)$",
    )
    ax1.plot(
        f_dr,
        -np.imag(d2["eps_cont"]),
        **figstyle.series_style(1, with_marker=False),
        label=r"$-\mathrm{Im}\,\varepsilon(f)$",
    )
    ax1.axhline(0.0, color="#1E293B", ls=":", lw=1.0)
    ax1.axvline(
        f_enz_thz,
        color="#D55E00",
        ls="--",
        lw=1.3,
        label=rf"ENZ frequency $f_{{\mathrm{{ENZ}}}} = {f_enz_thz:.2f}$ THz",
    )
    ax1.axvspan(
        f_dr[0],
        f_enz_thz,
        color="#CBD5E1",
        alpha=0.40,
        label=r"Overdense ($\mathrm{Re}\,\varepsilon < 0$)",
    )
    ax1.set_xlabel("Frequency $f$ [THz]")
    ax1.set_ylabel(r"Relative permittivity $\varepsilon(f)$")
    ax1.set_title("Drude carrier permittivity across ENZ point")
    ax1.legend(fontsize=7.8)

    ax2.plot(
        f_dr,
        d2["R_disc"],
        **figstyle.series_style(0, with_marker=False),
        label=r"$R_{\mathrm{TM}}(\varepsilon_{\mathrm{disc}})$",
    )
    ax2.plot(
        f_dr,
        d2["T_disc"],
        **figstyle.series_style(2, with_marker=False),
        label=r"$T_{\mathrm{TM}}(\varepsilon_{\mathrm{disc}})$",
    )
    ax2.plot(
        f_dr,
        d2["A_disc"],
        **figstyle.series_style(1, with_marker=False),
        label=r"$A_{\mathrm{TM}}(\varepsilon_{\mathrm{disc}})$",
    )
    ax2.plot(
        f_dr[::4],
        d2["R_fit"][::4],
        "o",
        color=figstyle.PALETTE[0],
        ms=3.8,
        label="FIT $R(f)$",
    )
    ax2.plot(
        f_dr[::4],
        d2["T_fit"][::4],
        "s",
        color=figstyle.PALETTE[2],
        ms=3.8,
        label="FIT $T(f)$",
    )
    ax2.plot(
        f_dr[::4],
        d2["A_fit"][::4],
        "^",
        color=figstyle.PALETTE[1],
        ms=3.8,
        label="FIT $A(f)$",
    )
    ax2.axvline(
        f_enz_thz, color="#D55E00", ls="--", lw=1.3, label=r"$f_{\mathrm{ENZ}}$ plasma edge"
    )
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_xlabel("Frequency $f$ [THz]")
    ax2.set_ylabel("Power reflectance / transmittance / absorptance")
    ax2.set_title(r"Drude slab ($d = 40\;\mu\mathrm{m}$): metallic reflection to transparency")
    ax2.legend(fontsize=7.6, ncol=2)

    figstyle.save(fig, fig_dir / "06_drude_enz_slab_rt.png")

    # Figure 7: Debye polar-molecule relaxation slab (liquid water)
    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.3)
    f_deb = d3["freqs"] * 1e-9
    f_pk_ghz = float(d3["f_peak"]) * 1e-9

    ax1.plot(
        f_deb,
        np.real(d3["eps_cont"]),
        **figstyle.series_style(0, with_marker=False),
        label=r"$\varepsilon'(f)$ continuous",
    )
    ax1.plot(
        f_deb,
        -np.imag(d3["eps_cont"]),
        **figstyle.series_style(1, with_marker=False),
        label=r"$\varepsilon''(f) = -\mathrm{Im}\,\varepsilon$",
    )
    ax1.plot(
        f_deb[::5],
        np.real(d3["eps_disc"][::5]),
        "o",
        color=figstyle.PALETTE[0],
        ms=3.8,
        label=r"$\mathrm{Re}\,\varepsilon_{\mathrm{disc}}(f, \Delta t)$",
    )
    ax1.plot(
        f_deb[::5],
        -np.imag(d3["eps_disc"][::5]),
        "s",
        color=figstyle.PALETTE[1],
        ms=3.8,
        label=r"$-\mathrm{Im}\,\varepsilon_{\mathrm{disc}}(f, \Delta t)$",
    )
    ax1.axvline(
        f_pk_ghz,
        color="#D55E00",
        ls="--",
        lw=1.2,
        label=rf"Loss peak $1/(2\pi\tau) = {f_pk_ghz:.1f}$ GHz",
    )
    ax1.set_xlabel("Frequency $f$ [GHz]")
    ax1.set_ylabel(r"Relative permittivity $\varepsilon(f)$")
    ax1.set_title(r"Liquid water at $20\,^\circ\mathrm{C}$ ($\tau = 9.3$ ps, $\Delta\varepsilon = 75.2$)")
    ax1.legend(fontsize=7.8)

    ax2.plot(
        f_deb,
        d3["R_disc"],
        **figstyle.series_style(0, with_marker=False),
        label=r"$R_{\mathrm{TM}}(\varepsilon_{\mathrm{disc}})$",
    )
    ax2.plot(
        f_deb,
        d3["T_disc"],
        **figstyle.series_style(2, with_marker=False),
        label=r"$T_{\mathrm{TM}}(\varepsilon_{\mathrm{disc}})$",
    )
    ax2.plot(
        f_deb,
        d3["A_disc"],
        **figstyle.series_style(1, with_marker=False),
        label=r"$A_{\mathrm{TM}}(\varepsilon_{\mathrm{disc}})$",
    )
    ax2.plot(
        f_deb[::5],
        d3["R_fit"][::5],
        "o",
        color=figstyle.PALETTE[0],
        ms=3.8,
        label="FIT $R(f)$",
    )
    ax2.plot(
        f_deb[::5],
        d3["T_fit"][::5],
        "s",
        color=figstyle.PALETTE[2],
        ms=3.8,
        label="FIT $T(f)$",
    )
    ax2.plot(
        f_deb[::5],
        d3["A_fit"][::5],
        "^",
        color=figstyle.PALETTE[1],
        ms=3.8,
        label="FIT $A(f)$",
    )
    ax2.axvline(f_pk_ghz, color="#D55E00", ls="--", lw=1.2)
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_xlabel("Frequency $f$ [GHz]")
    ax2.set_ylabel("Power reflectance / transmittance / absorptance")
    ax2.set_title(r"Debye water slab ($d = 0.5$ mm): microwave reflection & absorption")
    ax2.legend(fontsize=7.6, ncol=2)

    figstyle.save(fig, fig_dir / "07_debye_water_slab_rt.png")

    # Figure 8: ADE O(dt^2) convergence & pole stability limit (is_stable)
    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.3)
    dt_ps = d4["dt_sweep"] * 1e12
    dt_slab_ps = d4["dt_slab_list"] * 1e12

    ax1.loglog(
        dt_ps,
        d4["err_eps_lor"],
        **figstyle.series_style(0, with_marker=False),
        label=r"$|\varepsilon_{\mathrm{disc}} - \varepsilon_{\mathrm{cont}}|_2$ (Lorentz)",
    )
    ax1.loglog(
        dt_ps,
        d4["err_eps_dru"],
        **figstyle.series_style(2, with_marker=False),
        label=r"$|\varepsilon_{\mathrm{disc}} - \varepsilon_{\mathrm{cont}}|_2$ (Drude)",
    )
    ax1.loglog(
        dt_ps,
        d4["err_eps_deb"],
        **figstyle.series_style(1, with_marker=False),
        label=r"$|\varepsilon_{\mathrm{disc}} - \varepsilon_{\mathrm{cont}}|_2$ (Debye)",
    )
    ax1.loglog(
        dt_slab_ps,
        d4["rms_R_cont_list"],
        "ks-",
        ms=5.0,
        lw=1.8,
        label=r"FIT slab $\mathrm{RMS}(R_{\mathrm{FIT}} - R_{\mathrm{cont}})$",
    )
    ref_line = d4["err_eps_lor"][0] * (dt_ps / dt_ps[0]) ** 2
    ax1.loglog(
        dt_ps, ref_line, color="#64748B", ls=":", lw=1.3, label=r"Reference slope $O(\Delta t^2)$"
    )
    ax1.set_xlabel(r"Timestep $\Delta t$ [ps]")
    ax1.set_ylabel("Relative $L_2$ / RMS error")
    ax1.set_title(r"Second-order $O(\Delta t^2)$ accuracy of ADE discretization")
    ax1.legend(fontsize=7.8)

    w0_dt = d4["w0_dt_grid"]
    ax2.plot(
        w0_dt,
        d4["rho_pole_arr"],
        **figstyle.series_style(0, with_marker=False),
        label=r"Uncoupled pole recursion $\max|z_i|$",
    )
    ax2.plot(
        w0_dt,
        d4["rho_coupled_arr"],
        **figstyle.series_style(1, with_marker=False),
        label=r"Coupled EM + ADE $\max|\lambda_i|$ (`is_stable`)",
    )
    ax2.axhline(
        1.0, color="#1E293B", ls=":", lw=1.1, label=r"Unit circle $|\lambda| = 1$ (stability limit)"
    )
    ax2.axvline(
        2.0,
        color="#D55E00",
        ls="--",
        lw=1.3,
        label=r"Pole ceiling $\omega_0\Delta t = 2$",
    )
    unstable_mask = d4["is_stable_arr"] == 0
    if np.any(unstable_mask):
        w_crit = float(w0_dt[np.argmax(unstable_mask)])
        ax2.axvspan(
            w_crit,
            float(w0_dt[-1]),
            color="#D55E00",
            alpha=0.14,
            label="`is_stable(dt) == False`",
        )
    ax2.set_ylim(0.85, 2.1)
    ax2.set_xlabel(r"Normalized timestep $\omega_0 \Delta t$ [rad]")
    ax2.set_ylabel(r"Spectral radius $\rho = \max|\lambda_i|$")
    ax2.set_title("Leapfrog ADE stability boundary per Lorentz pole")
    ax2.legend(fontsize=7.6, loc="upper left")

    figstyle.save(fig, fig_dir / "08_ade_discretization_and_stability.png")
    print(f"Rendered setup schematic, Wikimedia diagram, and 8 figures in {fig_dir}")


# %% [markdown]
# ## 6. Slab reflection/transmission & ADE convergence results
#
# Running the complete study produces the following quantitative benchmarks
# (`data/task_*.json` and `figures/01_...`–`08_...`):
#
# | Benchmark (`task_I`) | Metric | Measured / simulated value (`--full`) | Theoretical / discrete reference |
# |---|---|---|---|
# | `task_0` (`bulk_ade_and_sum_rules`) | 1-cell FIT $\varepsilon_{\text{FIT}}(f)$ vs. `epsilon_discrete(f, dt)` | Relative $L_2$ error: $8.65 \times 10^{-5}$ (Lorentz), $8.77 \times 10^{-7}$ (Drude), $3.04 \times 10^{-6}$ (Debye) | Exact discrete ADE impedance identity |
# | `task_0` (`bulk_ade_and_sum_rules`) | Kramers–Kronig $\operatorname{Re}\chi$ from $\operatorname{Im}\chi$ & $\operatorname{Im}\chi$ from $\operatorname{Re}\chi$ | Relative $L_2$ error: $4.53 \times 10^{-3}$ ($\operatorname{Re}\chi$) & $1.08 \times 10^{-3}$ ($\operatorname{Im}\chi$) | Exact Cauchy principal-value pair |
# | `task_0` (`bulk_ade_and_sum_rules`) | TRK $f$-sum rule $\frac{2}{\pi}\int_0^\Omega \omega\varepsilon''(\omega)d\omega / \omega_{p,\text{tot}}^2$ | Relative error $4.84 \times 10^{-5}$ (low $\gamma_p$) & $1.28 \times 10^{-4}$ ($3.7\times$ high $\gamma_p$) | $1.0000$ (independent of damping $\gamma_p$) |
# | `task_1` (`lorentz_reststrahlen_slab`) | GaN Reststrahlen slab ($d = 2.5\;\mu\text{m}$, $10\text{–}28\;\text{THz}$) | Peak $R_{\text{FIT}} = 0.8984$ in $[\omega_{\text{TO}}, \omega_{\text{LO}}]$; $\text{RMS}(R_{\text{FIT}} - R_{\text{disc}}) = 9.95 \times 10^{-4}$, $\text{RMS}(T_{\text{FIT}} - T_{\text{disc}}) = 3.32 \times 10^{-4}$ | Transfer matrix with `epsilon_discrete` |
# | `task_2` (`drude_enz_slab`) | Drude ENZ slab ($d = 40\;\mu\text{m}$, $f_{\text{ENZ}} = 2.0\;\text{THz}$) | $\text{RMS}(R_{\text{FIT}} - R_{\text{disc}}) = 5.07 \times 10^{-5}$; $\text{RMS}(T_{\text{FIT}} - T_{\text{disc}}) = 3.61 \times 10^{-5}$ | Transfer matrix with `epsilon_discrete` |
# | `task_3` (`debye_water_slab`) | Liquid-water Debye slab ($d = 0.5\;\text{mm}$, $f_{\text{peak}} = 17.11\;\text{GHz}$) | $\text{RMS}(R_{\text{FIT}} - R_{\text{disc}}) = 3.23 \times 10^{-4}$; $\text{RMS}(T_{\text{FIT}} - T_{\text{disc}}) = 3.79 \times 10^{-4}$ | Transfer matrix with `epsilon_discrete` |
# | `task_4` (`dt_convergence_and_stability`) | Empirical log-log slope of $\|\varepsilon_{\text{disc}} - \varepsilon_{\text{cont}}\|_2$ & slab $R$ vs. $\Delta t$ | $p = 2.001$ (Lorentz), $2.000$ (Drude), $2.000$ (Debye); slab $R$ error drops $2.84 \times 10^{-3} \to 2.87 \times 10^{-4}$ ($p = 1.88$) | Second-order $O(\Delta t^2)$ centering |
#
# ![Microscopic dispersion models in frequency and time domain](figures/01_microscopic_models_freq_and_time.png)
#
# ![Bound electrons in SiO2 and Reststrahlen phonon band in GaN](figures/02_bound_electrons_and_reststrahlen.png)
#
# ![Gold Drude vs interband transitions and Johnson & Christy data](figures/03_gold_drude_vs_interband_jc.png)
#
# ![Kramers-Kronig Hilbert transform and TRK f-sum rule](figures/04_kramers_kronig_and_f_sum_rule.png)
#
# ![GaN Reststrahlen Lorentz slab reflection and transmission](figures/05_lorentz_reststrahlen_slab_rt.png)
#
# ![Drude ENZ slab reflection and transmission](figures/06_drude_enz_slab_rt.png)
#
# ![Debye liquid-water slab reflection and transmission](figures/07_debye_water_slab_rt.png)
#
# ![ADE second-order convergence and stability boundary](figures/08_ade_discretization_and_stability.png)
#
# ---
#
# ## 7. Limits & failure modes
#
# 1. **High-frequency pole instability ($\omega_{0,p}\Delta t \ge 2$):**
#    Even when the vacuum Courant condition $c_0\Delta t \sqrt{\Delta x^{-2}+\Delta y^{-2}+\Delta z^{-2}} \le 1$
#    is satisfied on a coarse spatial mesh, an ultraviolet Lorentz pole with
#    $(\omega_{0,p}\Delta t / 2)^2 > (\varepsilon_\infty - S^2) / (\varepsilon_s - S^2)$
#    drives the coupled ADE amplification matrix outside the unit circle
#    (`08_ade_discretization_and_stability.png`). For example, the
#    shortest-wavelength Sellmeier UV resonance of fused silica ($\lambda_1 = 68.4\;\text{nm}$,
#    $\omega_{0,1} = 2.75 \times 10^{16}\;\text{rad/s}$) requires $\Delta t \lesssim 68\;\text{as}$.
#    When simulating at microwave or mid-IR frequencies far below a UV resonance,
#    fold off-resonant UV poles into $\varepsilon_\infty \leftarrow \varepsilon_\infty + \Delta\varepsilon_{\text{UV}}$
#    rather than keeping stiff poles with $\omega_{0,p}\Delta t \sim O(1)$
#    (see Tutorial 2.2).
# 2. **Spatial skin-depth and Reststrahlen decay resolution ($\delta_s / \Delta z \gtrsim 4$):**
#    Inside a Reststrahlen band ($\operatorname{Re}\varepsilon < 0$) or a Drude
#    metal below $\omega_{\text{ENZ}}$, the field decays exponentially over the
#    penetration depth $\delta_s = c_0 / (\omega \kappa)$. If $\Delta z \gtrsim \delta_s$,
#    staircase interface averaging misestimates the surface impedance and produces
#    spurious transmission leakage.
# 3. **Finite-bandwidth truncation in experimental Kramers–Kronig inversions:**
#    As shown in the middle panel of `04_kramers_kronig_and_f_sum_rule.png`,
#    applying the KK integral directly to tabulated $(n, k)$ data over a bounded
#    spectral window $[0.64, 6.6]\;\text{eV}$ omits both the infrared Drude tail
#    ($\omega < \omega_{\min}$) and deep-UV core transitions ($\omega > \omega_{\max}$),
#    shifting the reconstructed $\operatorname{Re}\varepsilon(\omega)$ by a large
#    offset unless analytic low- and high-frequency asymptotes are appended.
# 4. **Local isotropic linear response only:**
#    `Lorentz`, `Drude`, and `Debye` describe local ($\mathbf{k}$-independent),
#    linear, isotropic susceptibility kernels. Nonlocal hydrodynamic pressure in
#    sub-5 nm metallic gaps ($\omega_p^2 / (\omega^2 - \beta_F^2 k^2 + i\gamma\omega)$),
#    gyrotropic magnetoplasma/ferrite tensors, and nonlinear Kerr/saturable
#    polarizations require spatial-derivative or nonlinear auxiliary equations
#    (see Tutorial 2.3 and Chapter 8).
#
# > **Forward pointers:**
# > - **Tutorial 2.2 (`02-02_fitting_measured_data`)** shows how to go from
# >   experimentally measured $(n, k)$ tables and fractional Cole–Cole tissue
# >   spectra to passive, stable multi-pole `DispersiveMaterial` models.
# > - **Tutorial 2.3 (`02-03_magnetic_materials`)** extends the Lorentz and Debye
# >   ADE formalism to magnetic permeability $\mu(\omega)$ derived from the
# >   Landau–Lifshitz–Gilbert equation and Snoek's law.
# > - **Tutorial 3.6 (`03-06_plasmonic_cylinder`)** benchmarks pole count versus
# >   localized surface-plasmon accuracy and runtime cost in 2D/3D scattering.
#
# ---
#
# ## 8. References
#
# 1. J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley, New York (1999),
#    Ch. 7 ("Magnetohydrodynamics and Plasma Physics" / "Models for Index of
#    Refraction, Kramer–Kronig Relations"), ISBN 978-0-471-30932-1.
# 2. A. D. Rakić, A. B. Djurišić, J. M. Elazar, and M. L. Majewski, "Optical
#    properties of metallic films for vertical-cavity optoelectronic devices,"
#    *Appl. Opt.* **37**(22), 5271–5283 (1998),
#    DOI: [10.1364/AO.37.005271](https://doi.org/10.1364/AO.37.005271).
# 3. P. B. Johnson and R. W. Christy, "Optical constants of the noble metals,"
#    *Phys. Rev. B* **6**(12), 4370–4379 (1972),
#    DOI: [10.1103/PhysRevB.6.4370](https://doi.org/10.1103/PhysRevB.6.4370).
# 4. A. S. Barker Jr. and M. Ilegems, "Infrared lattice vibrations and
#    free-electron dispersion in GaN," *Phys. Rev. B* **7**(2), 743–750 (1973),
#    DOI: [10.1103/PhysRevB.7.743](https://doi.org/10.1103/PhysRevB.7.743).
# 5. I. H. Malitson, "Interspecimen comparison of the refractive index of fused
#    silica," *J. Opt. Soc. Am.* **55**(10), 1205–1209 (1965),
#    DOI: [10.1364/JOSA.55.001205](https://doi.org/10.1364/JOSA.55.001205).
# 6. A. Taflove and S. C. Hagness, *Computational Electrodynamics: The
#    Finite-Difference Time-Domain Method*, 3rd ed., Artech House, Norwood, MA
#    (2005), Ch. 9 ("Dispersive Materials").
# 7. C. Kittel, *Introduction to Solid State Physics*, 8th ed., Wiley, New York
#    (2005), Ch. 16 ("Dielectrics and Ferroelectrics"), ISBN 978-0-471-41526-8.


# %%
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Tutorial 2.1 · Where dispersion comes from (Lorentz, Drude, Debye, ADE)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run fast study (< 30 s) and render all 8 figures",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run converged high-resolution study across all 5 tasks and render all 8 figures",
    )
    ap.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single study task I (0..4)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR,
        help="Output directory for --task / --plot (default: data/)",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Read data/task_*.npz and generate all 8 figures in figures/",
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

    quick = not args.full
    tasks = get_task_table(quick=quick)
    for t in tasks:
        run_single_task(t, args.out)
    generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR)


if __name__ == "__main__":
    main()
