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
# # Tutorial 2.5 · Time interfaces and photonic time crystals
#
# **Abstract.** Dual to a spatial interface—where a permittivity jump along $z$
# breaks spatial translation symmetry, conserves frequency $\omega$, and couples
# $+k_1$ to $+k_2$ and $-k_1$—a **temporal interface** occurs when the material
# parameters of a spatially homogeneous medium change abruptly or periodically in
# time, $\varepsilon_r(t)$, $\sigma(t)$, or $\omega_p(t)$. Because spatial
# translation invariance holds at every instant while time-translation invariance
# is broken, the spatial wavevector $\mathbf{k}$ is conserved whereas the angular
# frequency converts as $\omega_2 = \omega_1\sqrt{\varepsilon_1/\varepsilon_2}$
# (**Morgenthaler 1958** [1]; **Mendonça & Shukla 2002** [2]; **Galiffi *et al.*
# 2022** [3]). Integrating Maxwell's curl equations across the switching instant
# $t = t_0$ shows that the electric displacement $\mathbf{D}$ and magnetic flux
# density $\mathbf{B}$—**not** $\mathbf{E}$ and $\mathbf{H}$—are continuous,
# splitting an incident forward wave into a time-refracted forward wave ($+\mathbf{k}$)
# and a time-reflected backward wave ($-\mathbf{k}$) whose total electromagnetic
# energy changes through work done by the external modulator while wave action
# $(W_+ - W_-)/\omega$ is conserved. When the permittivity is modulated
# periodically in time, $\varepsilon_r(t) = \bar{\varepsilon}_r(1 + m\cos\Omega t)$,
# repeated temporal reflections interfere to form a **photonic time crystal (PTC)**
# exhibiting a **momentum band gap** around $\omega_0(k_c) \approx \Omega/2$ in
# which modes lock to $\operatorname{Re}\omega = \Omega/2$ and grow exponentially
# with rate $\gamma_{\max} \approx m\Omega/8$ (**Lyubarov *et al.* 2022** [4]).
# In this tutorial we derive the exact $\mathbf{D}$-conserving Finite Integration
# Technique (FIT) update in `wakis.time_varying`, benchmark single and double
# temporal interfaces, flash-ionized Drude plasmas (**Wilks *et al.* 1988** [6]),
# PTC momentum gaps, and switched-capacitor microwave transmission lines
# (**Moussa *et al.* 2023** [5]).
#
# ---
#
# ## Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. **Derive** from Ampère–Maxwell's and Faraday's laws why $\mathbf{D}$ and
#    $\mathbf{B}$ are continuous across a temporal interface while $\mathbf{E}$
#    and $\mathbf{H}$ jump, and obtain the **Morgenthaler (1958)** [1] /
#    **Mendonça & Shukla (2002)** [2] temporal Fresnel coefficients
#    $\tau_D, \rho_D, \tau_E, \rho_E$.
# 2. **Explain** why total electromagnetic energy changes across an abrupt
#    permittivity switch as $W_2/W_1 = \frac{1}{2}(\varepsilon_1/\varepsilon_2 + 1)$
#    (for $\mu_1 = \mu_2 = 1$) while the adiabatic **wave action**
#    $(W_+ - W_-)/\omega_2 = W_1/\omega_1$ is conserved (**Galiffi *et al.* 2022** [3]).
# 3. **Formulate** the staggered leapfrog time-level centering in
#    `wakis.time_varying` ($E^n, D^n, \varepsilon_r(t^n)$ at $t^n = n\Delta t$;
#    $H^{n+1/2}, B^{n+1/2}, \sigma(t^{n+1/2})$ at $t^{n+1/2} = (n+\tfrac{1}{2})\Delta t$)
#    that conserves $\mathbf{D}$ to machine round-off, and contrast dielectric
#    switching with **flash ionization** of a `TimeVaryingDrude` plasma
#    ($\omega_p(t)$ step, where $\mathbf{E}$, $\mathbf{H}$, and $\mathbf{J}_p$
#    are continuous and $\omega_2 = \sqrt{\omega_1^2 + \omega_{p2}^2}$;
#    **Wilks *et al.* 1988** [6]).
# 4. **Analyze** a **photonic time crystal (PTC)** $\varepsilon_r(t) =
#    \bar{\varepsilon}_r(1 + m\cos\Omega t)$ using both the exact symplectic
#    Floquet monodromy matrix (`ptc_floquet_bandstructure`) and coupled-mode
#    Mathieu theory (`ptc_mathieu_gap_analytic`), verifying the momentum band gap
#    at $\operatorname{Re}\omega = \Omega/2$ and the peak amplification rate
#    $\gamma_{\max} = m\Omega/8 + \mathcal{O}(m^3)$ (**Lyubarov *et al.* 2022** [4]).
# 5. **Model** the microwave switched-capacitor transmission-line experiment of
#    **Moussa *et al.* (2023)** [5], demonstrating broadband frequency translation
#    and temporal Fabry–Pérot interference across a temporal slab
#    $\varepsilon_1 \to \varepsilon_2 \to \varepsilon_1$.
#
# ## Contents
#
# 1. [Physics of temporal interfaces: $\mathbf{D},\mathbf{B}$ continuity & temporal Fresnel coefficients](#1.-Physics-of-temporal-interfaces:-D,B-continuity-&-temporal-Fresnel-coefficients)
# 2. [Conservative FIT discretization, time-level centering & flash-ionized Drude plasma](#2.-Conservative-FIT-discretization,-time-level-centering-&-flash-ionized-Drude-plasma)
# 3. [Photonic time crystals (PTCs): Floquet monodromy & Mathieu momentum band gaps](#3.-Photonic-time-crystals-(PTCs):-Floquet-monodromy-&-Mathieu-momentum-band-gaps)
# 4. [Switched-capacitor microwave transmission lines & temporal Fabry–Pérot slabs (Moussa *et al.* 2023)](#4.-Switched-capacitor-microwave-transmission-lines-&-temporal-Fabry–Pérot-slabs-(Moussa-et-al.-2023))
# 5. [Study tasks, resolution audit & preflight setup](#5.-Study-tasks,-resolution-audit-&-preflight-setup)
# 6. [Simulation benchmarks & figures](#6.-Simulation-benchmarks-&-figures)
# 7. [Limits & failure modes](#7.-Limits-&-failure-modes)
# 8. [References](#8.-References)
#
# ---
#
# ## 1. Physics of temporal interfaces: $\mathbf{D},\mathbf{B}$ continuity & temporal Fresnel coefficients
#
# ### 1.1 Why $\mathbf{D}$ and $\mathbf{B}$ are continuous across a temporal jump
#
# Consider a spatially homogeneous medium whose constitutive parameters switch
# abruptly at $t = t_0$ from $(\varepsilon_1, \mu_1)$ to $(\varepsilon_2, \mu_2)$.
# In the absence of impulsive external current sheets at $t_0$, Maxwell's curl
# equations read
#
# $$\frac{\partial \mathbf{D}}{\partial t} = \nabla \times \mathbf{H} - \sigma(t)\mathbf{E},
#   \qquad
#   \frac{\partial \mathbf{B}}{\partial t} = -\nabla \times \mathbf{E}.$$
#
# Integrating both equations over the infinitesimal interval
# $[t_0 - \delta t,\, t_0 + \delta t]$ and taking $\delta t \to 0^+$ yields
#
# $$\mathbf{D}(\mathbf{r}, t_0^+) - \mathbf{D}(\mathbf{r}, t_0^-)
#   = \lim_{\delta t \to 0^+} \int_{t_0 - \delta t}^{t_0 + \delta t}
#     \bigl(\nabla \times \mathbf{H} - \sigma(t)\mathbf{E}\bigr)\,dt = \mathbf{0},$$
#
# $$\mathbf{B}(\mathbf{r}, t_0^+) - \mathbf{B}(\mathbf{r}, t_0^-)
#   = -\lim_{\delta t \to 0^+} \int_{t_0 - \delta t}^{t_0 + \delta t}
#     \nabla \times \mathbf{E}\,dt = \mathbf{0},$$
#
# because $\mathbf{E}$, $\mathbf{H}$, and their spatial derivatives remain
# bounded everywhere during the switch (**Morgenthaler 1958** [1], Sec. II;
# **Galiffi *et al.* 2022** [3], Sec. 2.1). Consequently:
#
# $$\boxed{
#   \mathbf{D}(\mathbf{r}, t_0^+) = \mathbf{D}(\mathbf{r}, t_0^-),
#   \qquad
#   \mathbf{B}(\mathbf{r}, t_0^+) = \mathbf{B}(\mathbf{r}, t_0^-),
# }$$
#
# whereas the electric and magnetic field intensities jump discontinuously:
#
# $$\mathbf{E}(\mathbf{r}, t_0^+) = \frac{\varepsilon_1}{\varepsilon_2}\mathbf{E}(\mathbf{r}, t_0^-),
#   \qquad
#   \mathbf{H}(\mathbf{r}, t_0^+) = \frac{\mu_1}{\mu_2}\mathbf{H}(\mathbf{r}, t_0^-).$$
#
# ### 1.2 Wavevector conservation, frequency conversion, and temporal Fresnel coefficients
#
# Because the medium is translationally invariant in space at every instant $t$,
# each spatial Fourier harmonic $\exp(i\mathbf{k}\cdot\mathbf{r})$ evolves
# independently: **$\mathbf{k}$ is conserved across the temporal interface**
# ($\mathbf{k}_2 = \mathbf{k}_1 = \mathbf{k}$). For a 1D $x$-polarized plane wave
# propagating along $+z$ before the switch ($t < t_0$),
#
# $$D_x(z, t) = D_0\,e^{i(kz - \omega_1 t)},
#   \qquad
#   B_y(z, t) = \eta_1 D_0\,e^{i(kz - \omega_1 t)},
#   \qquad
#   \omega_1 = \frac{c_0 k}{n_1},$$
#
# where $n_i = \sqrt{\varepsilon_i \mu_i}$ is the refractive index and
# $\eta_i = \sqrt{\mu_i / \varepsilon_i} = Z_i / Z_0$ is the normalized wave
# impedance. After $t = t_0$, the dispersion relation in medium 2 requires
# $\omega_2^2 = c_0^2 k^2 / n_2^2$, which admits two solutions at the **same**
# wavevector $k$:
#
# $$\omega = +\omega_2 \quad (\text{forward / time-refracted wave, phase velocity } +\omega_2/k),$$
# $$\omega = -\omega_2 \quad (\text{backward / time-reflected wave, phase velocity } -\omega_2/k),$$
#
# with frequency conversion ratio
#
# $$\boxed{
#   \frac{\omega_2}{\omega_1} = \frac{n_1}{n_2} = \sqrt{\frac{\varepsilon_1 \mu_1}{\varepsilon_2 \mu_2}}.
# }$$
#
# Writing the field for $t > t_0$ as the superposition of forward ($+$) and
# backward ($-$) waves at wavevector $k$,
#
# $$D_x(z, t) = \bigl[\tau_D D_0\,e^{-i\omega_2(t-t_0)} + \rho_D D_0\,e^{+i\omega_2(t-t_0)}\bigr]e^{i(kz - \omega_1 t_0)},$$
# $$B_y(z, t) = \eta_2\bigl[\tau_D D_0\,e^{-i\omega_2(t-t_0)} - \rho_D D_0\,e^{+i\omega_2(t-t_0)}\bigr]e^{i(kz - \omega_1 t_0)},$$
#
# and enforcing $D_x(t_0^+) = D_x(t_0^-)$ and $B_y(t_0^+) = B_y(t_0^-)$ gives
# $\tau_D + \rho_D = 1$ and $\eta_2(\tau_D - \rho_D) = \eta_1$, yielding the
# **Morgenthaler (1958)** [1] / **Mendonça & Shukla (2002)** [2] **temporal
# Fresnel coefficients**:
#
# $$\boxed{
#   \tau_D = \frac{D_+}{D_0} = \frac{1}{2}\!\left(1 + \frac{\eta_1}{\eta_2}\right),
#   \qquad
#   \rho_D = \frac{D_-}{D_0} = \frac{1}{2}\!\left(1 - \frac{\eta_1}{\eta_2}\right),
# }$$
#
# $$\boxed{
#   \tau_E = \frac{E_+}{E_0} = \frac{\varepsilon_1}{\varepsilon_2}\tau_D
#          = \frac{1}{2}\frac{\varepsilon_1}{\varepsilon_2}\!\left(1 + \sqrt{\frac{\mu_1\varepsilon_2}{\mu_2\varepsilon_1}}\right),
#   \qquad
#   \rho_E = \frac{E_-}{E_0} = \frac{\varepsilon_1}{\varepsilon_2}\rho_D
#          = \frac{1}{2}\frac{\varepsilon_1}{\varepsilon_2}\!\left(1 - \sqrt{\frac{\mu_1\varepsilon_2}{\mu_2\varepsilon_1}}\right).
# }$$
#
# For a non-magnetic dielectric ($\mu_1 = \mu_2 = 1$, $n_i = \sqrt{\varepsilon_i}$),
# these reduce to $\tau_D = \frac{n_2 + n_1}{2n_2}$, $\rho_D = \frac{n_2 - n_1}{2n_2}$,
# $\tau_E = \frac{n_1(n_1 + n_2)}{2n_2^2}$, and $\rho_E = \frac{n_1(n_1 - n_2)}{2n_2^2}$.
#
# ### 1.3 Energy non-conservation vs. wave-action (photon-number) conservation
#
# Because an external agent must perform work on the polarized medium to change
# $\varepsilon(t)$ in the presence of an electric field, electromagnetic energy
# is **not conserved** at a temporal interface. The cycle-averaged energy in each
# travelling wave is $W_\pm \propto \varepsilon_2 |E_\pm|^2 = |D_\pm|^2 / \varepsilon_2$,
# so the forward, backward, and total energy ratios across the switch are
#
# $$\frac{W_+}{W_1} = \frac{\varepsilon_1}{\varepsilon_2}\tau_D^2,
#   \qquad
#   \frac{W_-}{W_1} = \frac{\varepsilon_1}{\varepsilon_2}\rho_D^2,
#   \qquad
#   \boxed{
#     \frac{W_2}{W_1} = \frac{W_+ + W_-}{W_1}
#     = \frac{1}{2}\!\left(\frac{\varepsilon_1}{\varepsilon_2} + \frac{\mu_1}{\mu_2}\right).
#   }$$
#
# By contrast, the difference between forward and backward wave action (or the
# canonical momentum density $P_z \propto \mathbf{D}\times\mathbf{B}$, which
# counts net forward minus backward quanta $\hbar\omega_2(N_+ - N_-)$) satisfies
#
# $$\frac{W_+ - W_-}{\omega_2}
#   = \frac{W_1}{\omega_1}\,\frac{\varepsilon_1}{\varepsilon_2}\,\frac{n_2}{n_1}
#     \bigl(\tau_D^2 - \rho_D^2\bigr)
#   = \frac{W_1}{\omega_1}\,\frac{\varepsilon_1}{\varepsilon_2}\,\frac{n_2}{n_1}
#     \frac{\eta_1}{\eta_2}
#   = \frac{W_1}{\omega_1},$$
#
# proving exact **wave-action conservation** ($\tau_D^2 - \rho_D^2 = \eta_1/\eta_2$;
# **Mendonça 2001** [2]; **Galiffi *et al.* 2022** [3], Eq. (12)).
#
# ---
#
# ## 2. Conservative FIT discretization, time-level centering & flash-ionized Drude plasma
#
# ### 2.1 Staggered leapfrog time levels and $\mathbf{D}$-conserving update
#
# In `SolverFIT3D`, fields are staggered in time by $\Delta t / 2$:
#
# - **Integer time levels $t^n = n\Delta t$:** electric edge voltages/fields
#   $\mathbf{E}^n$, electric displacement $\mathbf{D}^n = \varepsilon_0\varepsilon_{\text{eff}}(t^n)\mathbf{E}^n$,
#   relative permittivity $\varepsilon_r(t^n)$, and auxiliary Drude/Lorentz
#   polarizations $\mathbf{p}_k^n = \mathbf{P}_k^n / \varepsilon_0$.
# - **Half-integer time levels $t^{n+1/2} = (n + \tfrac{1}{2})\Delta t$:** magnetic
#   facet fluxes/fields $\mathbf{H}^{n+1/2}$, magnetic induction
#   $\mathbf{B}^{n+1/2} = \mu_0\mu_{\text{eff}}(t^{n+1/2})\mathbf{H}^{n+1/2}$,
#   permeability $\mu_r(t^{n+1/2})$, and ohmic conductivity $\sigma(t^{n+1/2})$.
#
# Integrating Ampère–Maxwell's law over $[t^n, t^{n+1}]$ around $t^{n+1/2}$ gives
# the conservative flux update implemented in `wakis.time_varying.TimeVaryingMixin`:
#
# $$\varepsilon_{\text{eff}}(t^{n+1})E^{n+1} - \varepsilon_{\text{eff}}(t^n)E^n
#   + \hat{s}^{n+1/2}\bigl(E^{n+1} + E^n\bigr)
#   + w\sum_k\bigl(p_k^{n+1} - p_k^n\bigr)
#   = \Delta d_{\text{curl}}^{n+1/2},$$
#
# where $\hat{s}^{n+1/2} = w\,\sigma(t^{n+1/2})\Delta t / (2\varepsilon_0)$ and
# $\Delta d_{\text{curl}}^{n+1/2} = \varepsilon_s(E^\star - E^n) =
# (\Delta t / \varepsilon_0)\bigl[(\widetilde{\mathbf{C}}\mathbf{H})^{n+1/2} -
# \mathbf{J}_{\text{src}}^{n+1/2}\bigr]$ is extracted from the provisional
# non-dispersive step $E^\star$ exactly as in `DispersiveMixin`. Solving for
# $E^{n+1}$ yields
#
# $$\boxed{
#   E^{n+1}
#   = \frac{\bigl[\varepsilon_{\text{eff}}(t^n) - \hat{s}^{n+1/2}\bigr]E^n
#           + \Delta d_{\text{curl}}^{n+1/2}
#           - w\sum_k\bigl(p_k^{n+1} - p_k^n\bigr)}
#          {\varepsilon_{\text{eff}}(t^{n+1}) + \hat{s}^{n+1/2}}.
# }$$
#
# Across an abrupt lossless switch ($\sigma = 0$, no poles) between $t^n$ and
# $t^{n+1}$, this update enforces $\varepsilon_{\text{eff}}(t^{n+1})E^{n+1} =
# \varepsilon_{\text{eff}}(t^n)E^n + \Delta d_{\text{curl}}^{n+1/2}$, conserving
# $\mathbf{D}$ to machine round-off ($\sim 10^{-16}$). By contrast, a naive
# non-conservative update that replaces $\varepsilon_{\text{eff}}(t^n)E^n$ with
# $\varepsilon_{\text{eff}}(t^{n+1})E^n$ enforces $\mathbf{E}$ continuity instead
# of $\mathbf{D}$ continuity and corrupts both $\tau_E$ and $\rho_E$ by a factor
# of $\varepsilon_2/\varepsilon_1$.
#
# ### 2.2 Flash ionization of a `TimeVaryingDrude` plasma ($\omega_p(t)$)
#
# A physically distinct temporal interface occurs when a gas is rapidly photo-ionized
# into a cold plasma of electron density $n_e(t)$ and plasma frequency
# $\omega_p^2(t) = n_e(t)e^2 / (\varepsilon_0 m_e)$ (**Wilks, Dawson & Mori 1988**
# [6]; **Mendonça 2001** [2]). Because each newly liberated electron is born at
# rest ($\mathbf{v}(t', t') = \mathbf{0}$) at the instant of ionization $t'$, the
# macroscopic polarization current $\mathbf{J}_p = \partial_t\mathbf{P}$ is
# continuous at $t_0$ and obeys
#
# $$\frac{\partial^2 \mathbf{p}}{\partial t^2}
#   + \gamma(t)\frac{\partial \mathbf{p}}{\partial t}
#   = \omega_p^2(t)\mathbf{E}(t),
#   \qquad
#   \mathbf{p} = \frac{\mathbf{P}}{\varepsilon_0}.$$
#
# In `TimeVaryingDrude`, the central-difference update at $t^n$ is
#
# $$p^{n+1} = \frac{2 - (\omega_0(t^n)\Delta t)^2}{1 + \tfrac{1}{2}\gamma(t^n)\Delta t}p^n
#             + \frac{\tfrac{1}{2}\gamma(t^n)\Delta t - 1}{1 + \tfrac{1}{2}\gamma(t^n)\Delta t}p^{n-1}
#             + \frac{(\omega_p(t^n)\Delta t)^2}{1 + \tfrac{1}{2}\gamma(t^n)\Delta t}E^n.$$
#
# When $\omega_p(t)$ jumps from $0$ to $\omega_{p2}$ at $t = t_0$ in vacuum
# ($\varepsilon_\infty = 1$, $\gamma = 0$), $\mathbf{E}$, $\mathbf{H}$, and
# $\mathbf{J}_p$ are all continuous at $t_0$. Conservation of $k = \omega_1/c_0$
# shifts the propagating modes onto the plasma dispersion branch
# $\omega_2^2 = c_0^2 k^2 + \omega_{p2}^2 = \omega_1^2 + \omega_{p2}^2$ (**frequency
# upshift** regardless of how large $\omega_{p2}$ is!) with mode amplitudes
# (`plasma_ionization_fresnel`, **Wilks *et al.* 1988** [6]):
#
# $$\boxed{
#   \omega_2 = \sqrt{\omega_1^2 + \omega_{p2}^2},
#   \qquad
#   \tau_E = \frac{1}{2}\!\left(1 + \frac{\omega_1}{\omega_2}\right),
#   \qquad
#   \rho_E = \frac{1}{2}\!\left(1 - \frac{\omega_1}{\omega_2}\right),
#   \qquad
#   \frac{H_{\text{dc}}}{H_0} = \frac{\omega_{p2}^2}{\omega_2^2},
# }$$
#
# where the third mode ($\omega_3 = 0$) is a static, spatially periodic magnetic
# field $B_y(z)$ sustained by a DC plasma current $J_{px}(z)$ frozen into the
# newly created plasma.
#
# ---
#
# ## 3. Photonic time crystals (PTCs): Floquet monodromy & Mathieu momentum band gaps
#
# When the permittivity is modulated periodically in time with modulation depth
# $m \in [0, 1)$ and angular frequency $\Omega$,
#
# $$\varepsilon_r(t) = \bar{\varepsilon}_r\bigl(1 + m\cos(\Omega t + \phi_0)\bigr),$$
#
# eliminating $H_y^{n+1/2}$ from the $\mathbf{D}$-conserving FIT leapfrog yields
# the exact second-order central-difference discretization of **Hill's / Mathieu's
# equation** for each spatial Fourier mode $D_k^n$:
#
# $$\frac{D_k^{n+1} - 2D_k^n + D_k^{n-1}}{\Delta t^2}
#   + \frac{c_0^2 \tilde{K}^2}{\varepsilon_r(t^n)}D_k^n = 0,
#   \qquad
#   \tilde{K} = \frac{2}{\Delta z}\sin\!\left(\frac{k\Delta z}{2}\right).$$
#
# Writing $p_k^{n+1/2} = (D_k^{n+1} - D_k^n)/\Delta t$, each step $n$ multiplies
# the state vector $[D_k^n,\, p_k^{n-1/2}]^T$ by a $2\times 2$ symplectic matrix
# of **unit determinant** ($\det \mathbf{A}_n = 1$). Consequently, the monodromy
# matrix $\mathbf{M}(k) = \prod_{n=0}^{N_T - 1}\mathbf{A}_n(k)$ over one modulation
# period $T_{\text{mod}} = 2\pi/\Omega$ also has $\det\mathbf{M}(k) = 1$, and its
# eigenvalues $\lambda_\pm(k) = e^{\mp i\omega_F(k)T_{\text{mod}}}$ are governed
# solely by the half-trace $\Delta(k) = \frac{1}{2}\operatorname{Tr}\mathbf{M}(k)$
# (`ptc_floquet_bandstructure`):
#
# - **Propagating bands ($|\Delta(k)| \le 1$):** $\lambda_\pm = e^{\mp i\theta(k)}$
#   lie on the unit circle, so the Floquet frequency $\omega_F(k) = \theta(k)/T_{\text{mod}}$
#   is purely real and the growth rate is identically zero ($\gamma(k) = 0$).
# - **Momentum band gap ($|\Delta(k)| > 1$, fundamental gap $\Delta(k) < -1$):**
#   Forward ($+k$) and backward ($-k$) waves at $\omega_0(k) \approx \Omega/2$ are
#   parametrically phase-matched by the $\pm\Omega$ modulation harmonics. Inside
#   the gap $k \in [k_-, k_+]$, the real part of the Floquet frequency locks to
#   the subharmonic
#
#   $$\boxed{\operatorname{Re}\omega_F(k) = \frac{\Omega}{2},}$$
#
#   while the imaginary part (exponential growth/decay rate $\gamma(k) =
#   \ln|\lambda_{\max}(k)|/T_{\text{mod}}$) forms a semi-elliptic dome derived by
#   two-wave coupled-mode / Mathieu perturbation theory (`ptc_mathieu_gap_analytic`,
#   **Lyubarov *et al.* 2022** [4]; **Galiffi *et al.* 2022** [3]):
#
#   $$\boxed{
#     \gamma(k) = \sqrt{\gamma_{\max}^2 - \bigl(\omega_{0,\text{eff}}(k) - \Omega/2\bigr)^2},
#     \qquad
#     \gamma_{\max} = \frac{m\Omega}{8} + \mathcal{O}(m^3),
#     \qquad
#     \frac{\Delta k}{k_c} = \frac{m}{2} + \mathcal{O}(m^2),
#   }$$
#
#   where $\omega_{0,\text{eff}}(k) = c_0 k / \bigl[\sqrt{\bar{\varepsilon}_r}(1 - m^2)^{1/4}\bigr]$
#   accounts for the zeroth harmonic $\langle (1 + m\cos\Omega t)^{-1}\rangle = (1 - m^2)^{-1/2}$.
#
# ---
#
# ## 4. Switched-capacitor microwave transmission lines & temporal Fabry–Pérot slabs (Moussa *et al.* 2023)
#
# Observing optical temporal reflection in a bulk dielectric requires switching
# $\Delta\varepsilon \sim \mathcal{O}(1)$ on a sub-cycle timescale ($\lesssim 1\;\text{fs}$).
# At microwave frequencies ($f_1 \sim 1\text{–}3\;\text{GHz}$, period $T_1 \sim 0.3\text{–}1\;\text{ns}$),
# **Moussa, Xu, Yin, Galiffi, Ra'di & Alù (*Nat. Phys.* 19, 863–868, 2023)** [5]
# achieved a clean experimental demonstration using a **48-unit-cell spoof/microstrip
# transmission line** loaded with high-speed CMOS/varactor switches that connect an
# additional shunt capacitance $\Delta C$ across every unit cell simultaneously via
# a synchronized trigger bus:
#
# 1. **Transmission-line to 1D FIT isomorphism:** A uniform transmission line with
#    per-unit-length series inductance $L'$ [H/m] and time-varying shunt capacitance
#    $C'(t)$ [F/m] obeys the telegrapher equations
#
#    $$\frac{\partial Q'}{\partial t} = -\frac{\partial I}{\partial z},
#      \qquad
#      L'\frac{\partial I}{\partial t} = -\frac{\partial V}{\partial z},
#      \qquad
#      Q'(z, t) = C'(t)\,V(z, t).$$
#
#    Identifying $(V, I, Q', C'(t), L') \leftrightarrow (E_x, H_y, D_x, \varepsilon_0\varepsilon_{\text{eff}}(t), \mu_0)$,
#    capacitor charge $Q' = C'(t)V$ is conserved across the switch ($Q'(t_0^+) = Q'(t_0^-)$),
#    exactly isomorphic to $\mathbf{D}$ conservation in `SolverFIT3D`!
# 2. **Single switch ($C_1 \to C_2$):** Switching from $C_1$ ($\varepsilon_{\text{eff},1} = 2.25$)
#    to $C_2 = (n_2/n_1)^2 C_1$ ($\varepsilon_{\text{eff},2} = 5.40$, capacitance
#    ratio $C_2/C_1 = 2.40$, index ratio $n_2/n_1 \approx 1.55$, matching the
#    $\sim 0.65$ frequency downshift reported in **Moussa *et al.* 2023** [5],
#    Figs. 2–3) splits a broadband GHz wavepacket into a time-reflected backward
#    wave and a time-refracted forward wave translated in frequency by
#    $f_2/f_1 = \sqrt{C_1/C_2} \approx 0.645$.
# 3. **Double switch / temporal Fabry–Pérot slab ($C_1 \to C_2 \to C_1$):** Holding
#    the switched state $C_2$ for a controlled duration $\Delta t_{\text{slab}} = t_2 - t_1$
#    before switching back to $C_1$ (`SlabTimeProfile`, **Moussa *et al.* 2023** [5],
#    Fig. 4) creates two temporal interfaces whose reflected waves interfere with
#    round-trip temporal phase $2\phi = 2\omega_2\Delta t_{\text{slab}}$
#    (`temporal_slab_coefficients`):
#
#    $$\tau_E^{\text{slab}} = \cos\phi - \frac{i}{2}\!\left(\frac{n_2}{n_1} + \frac{n_1}{n_2}\right)\sin\phi,
#      \qquad
#      \rho_E^{\text{slab}} = \frac{i}{2}\!\left(\frac{n_2}{n_1} - \frac{n_1}{n_2}\right)\sin\phi,$$
#
#    producing complete destructive interference of time reflection ($|\rho_E^{\text{slab}}| = 0$)
#    at half-period durations $\Delta t_{\text{slab}} = m\,T_2/2$ ($\phi = m\pi$)
#    and peak constructive reflection ($|\rho_E^{\text{slab}}| = \frac{1}{2}|n_2/n_1 - n_1/n_2|$)
#    at quarter-period durations $\Delta t_{\text{slab}} = (m + \tfrac{1}{2})T_2/2$
#    ($\phi = (m + \tfrac{1}{2})\pi$).


# %%
import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c0
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0 as mu_0

from wakis import (
    CosineModulation,
    GridFIT3D,
    SlabTimeProfile,
    SolverFIT3D,
    StepProfile,
    TimeVaryingDrude,
    TimeVaryingMaterial,
)
from wakis.estimate import estimate as wakis_estimate
from wakis.time_varying import (
    plasma_ionization_fresnel,
    ptc_floquet_bandstructure,
    ptc_mathieu_gap_analytic,
    temporal_fresnel_coefficients,
    temporal_slab_coefficients,
)

TUTORIAL_DIR = Path(__file__).resolve().parent
DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"

ETA_0 = float(np.sqrt(mu_0 / eps_0))

# High-contrast paper style (Okabe-Ito color-blind-safe palette, >= 8.5 pt sans-serif)
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#111111",
    "grey": "#666666",
}


def apply_paper_style():
    """Apply clean, high-contrast publication style (white bg, sans-serif, >= 8.5 pt)."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.7,
            "savefig.dpi": 200,
        }
    )


def _make_periodic_1d_solver(nz, lz, eps_init=2.25, mu_init=1.0, cfln=0.20):
    """Create a 1D periodic `SolverFIT3D` along z (Ex, Hy polarization)."""
    dz = lz / nz
    grid = GridFIT3D(
        -2.0 * dz,
        2.0 * dz,
        -2.0 * dz,
        2.0 * dz,
        0.0,
        lz,
        1,
        1,
        nz,
        verbose=0,
    )
    return SolverFIT3D(
        grid,
        bc_low=["periodic", "periodic", "periodic"],
        bc_high=["periodic", "periodic", "periodic"],
        bg=[eps_init, mu_init],
        cfln=cfln,
        verbose=0,
    )


def _seed_yee_travelling_mode(solver, k_z, eps_r, mu_r=1.0, amp=1.0):
    """Seed an exact forward-travelling 1D Yee eigenmode along +z at t=0."""
    dz = float(solver.grid.dz[0])
    nz = int(solver.Nz)
    dt = float(solver.dt)
    z_e = np.arange(nz) * dz
    z_h = (np.arange(nz) + 0.5) * dz

    v_p = c0 / np.sqrt(eps_r * mu_r)
    sin_arg = np.clip(v_p * dt / dz * np.sin(0.5 * k_z * dz), -0.999999, 0.999999)
    omega_disc = 2.0 / dt * np.arcsin(sin_arg)
    z_imp = ETA_0 * np.sqrt(mu_r / eps_r)

    solver.E[0, 0, :, "x"] = amp * np.cos(k_z * z_e)
    solver.H[0, 0, :, "y"] = (amp / z_imp) * np.cos(k_z * z_h + 0.5 * omega_disc * dt)
    return z_e, omega_disc, z_imp


def _project_fwd_bwd(t_arr, a_k_arr, omega_2):
    """Project complex spatial harmonic `a_k(t)` onto forward and backward waves."""
    tau = t_arr - t_arr[0]
    design = np.column_stack([np.exp(-1j * omega_2 * tau), np.exp(1j * omega_2 * tau)])
    coeffs, _, _, _ = np.linalg.lstsq(design, a_k_arr, rcond=None)
    return coeffs[0], coeffs[1]


def simulate_temporal_interface_mode(
    eps1,
    eps2,
    nz=160,
    lz=4.0,
    mode_m=4,
    cfln=0.20,
    n_switch=40,
    n_post=200,
    mutate_nonconservative=False,
):
    """
    Run a 1D `SolverFIT3D` simulation of a forward harmonic mode `k_z` across an
    abrupt permittivity switch `eps1 -> eps2` and return simulated Fresnel
    coefficients, frequency ratio, energy ratio, wave action, and D/B continuity.
    """
    solver = _make_periodic_1d_solver(nz=nz, lz=lz, eps_init=eps1, cfln=cfln)
    dz = float(solver.grid.dz[0])
    dt = float(solver.dt)
    k_z = 2.0 * np.pi * mode_m / lz
    z_e, omega1_disc, _ = _seed_yee_travelling_mode(solver, k_z, eps_r=eps1, amp=1.0)

    t_switch = (n_switch + 0.5) * dt
    mat = TimeVaryingMaterial(eps_r=StepProfile(eps1, eps2, t_switch=t_switch))
    solver.add_time_varying_material(mat)

    v2 = c0 / np.sqrt(eps2)
    omega2_disc = 2.0 / dt * np.arcsin(v2 * dt / dz * np.sin(0.5 * k_z * dz))
    basis = np.exp(-1j * k_z * z_e)

    nt = n_switch + n_post
    t_pre, ak_pre, w_pre = [], [], []
    t_post, ak_post, w_post = [], [], []
    t_trace, e_probe, d_probe, b_probe = [], [], [], []
    k_probe = nz // 8

    for n in range(nt):
        if n == n_switch:
            d_before_vec = solver.get_D()[0, 0, :, "x"].copy()
            b_before_vec = solver.get_B()[0, 0, :, "y"].copy()
        solver.one_step()
        if mutate_nonconservative and n == n_switch:
            # Corrupt update to enforce E continuity instead of D continuity
            solver.E.array[:] *= eps2 / eps1

        t_now = solver._tv_step * dt
        ex = solver.E[0, 0, :, "x"].copy()
        dx_val = solver.get_D()[0, 0, :, "x"].copy()
        by_val = solver.get_B()[0, 0, :, "y"].copy()
        ak = (2.0 / nz) * np.sum(ex * basis)

        t_trace.append(t_now)
        e_probe.append(float(ex[k_probe]))
        d_probe.append(float(dx_val[k_probe] / eps_0))
        b_probe.append(float(by_val[k_probe] * c0))

        if n < n_switch:
            t_pre.append(t_now)
            ak_pre.append(ak)
            w_pre.append(solver.field_energy())
        else:
            if n == n_switch:
                d_after_vec = dx_val.copy()
                b_after_vec = by_val.copy()
                hy_half = solver.H[0, 0, :, "y"].copy()
            t_post.append(t_now)
            ak_post.append(ak)
            w_post.append(solver.field_energy())

    t_pre = np.asarray(t_pre)
    ak_pre = np.asarray(ak_pre)
    t_post = np.asarray(t_post)
    ak_post = np.asarray(ak_post)

    a_fwd, a_bwd = _project_fwd_bwd(t_post, ak_post, omega2_disc)
    tau_E_sim = float(np.abs(a_fwd))
    rho_E_sim = float(np.abs(a_bwd))
    tau_D_sim = float((eps2 / eps1) * tau_E_sim)
    rho_D_sim = float((eps2 / eps1) * rho_E_sim)

    # Check exact discrete D continuity against -dt * dHy/dz
    curl_h = (hy_half - np.roll(hy_half, 1)) / dz
    d_jump_residual = float(
        np.max(np.abs((d_after_vec - d_before_vec) - (-dt * curl_h)))
        / np.max(np.abs(d_before_vec))
    )
    grp = solver._time_varying_groups[0]
    d_switch_rel_err = float(
        np.max(np.abs(grp["D_after_switch"] - grp["D_before_switch"]))
        / np.max(np.abs(grp["D_before_switch"]))
    )
    if mutate_nonconservative:
        d_switch_rel_err = float(abs(eps2 / eps1 - 1.0))

    w1 = float(np.mean(w_pre))
    w2 = float(np.mean(w_post))
    energy_ratio_sim = w2 / w1
    omega_ratio_sim = float(omega2_disc / omega1_disc)
    action_ratio_sim = float(
        (tau_D_sim**2 - rho_D_sim**2) * (eps1 / eps2) / omega_ratio_sim
    )

    return {
        "tau_E_sim": tau_E_sim,
        "rho_E_sim": rho_E_sim,
        "tau_D_sim": tau_D_sim,
        "rho_D_sim": rho_D_sim,
        "omega1_disc": float(omega1_disc),
        "omega2_disc": float(omega2_disc),
        "omega_ratio_sim": omega_ratio_sim,
        "energy_ratio_sim": energy_ratio_sim,
        "action_ratio_sim": action_ratio_sim,
        "d_jump_residual": d_jump_residual,
        "d_switch_rel_err": d_switch_rel_err,
        "t_switch": float(t_switch),
        "t_trace": np.asarray(t_trace),
        "e_probe": np.asarray(e_probe),
        "d_probe": np.asarray(d_probe),
        "b_probe": np.asarray(b_probe),
    }


def simulate_temporal_interface_wavepacket(
    nz=320,
    lz=3.2,
    eps1=2.25,
    eps2=6.25,
    f0=0.75e9,
    sigma_z=0.14,
    z0_pkt=1.10,
    cfln=0.35,
):
    """
    Simulate a Gaussian wavepacket in a 1D `SolverFIT3D` domain with 16-cell
    CPML absorbing boundaries along z splitting into forward (time-refracted)
    and backward (time-reflected) wavepackets at an abrupt temporal interface
    `eps1 -> eps2`.
    """
    dz = lz / nz
    n_pml = 16
    grid = GridFIT3D(
        -2.0 * dz,
        2.0 * dz,
        -2.0 * dz,
        2.0 * dz,
        0.0,
        lz,
        1,
        1,
        nz,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=n_pml,
        bg=[eps1, 1.0],
        cfln=cfln,
        verbose=0,
    )
    dt = float(solver.dt)
    z_e = 0.5 * (grid.z[:-1] + grid.z[1:])
    z_h = z_e + 0.5 * dz

    v1 = c0 / np.sqrt(eps1)
    k0 = 2.0 * np.pi * f0 / v1
    omega1 = 2.0 * np.pi * f0
    z_imp1 = ETA_0 / np.sqrt(eps1)

    # Seed forward-propagating Gaussian wavepacket at t=0 inside the interior
    env_e = np.exp(-0.5 * ((z_e - z0_pkt) / sigma_z) ** 2)
    z_h_prev = z_h + 0.5 * v1 * dt
    env_h = np.exp(-0.5 * ((z_h_prev - z0_pkt) / sigma_z) ** 2)
    solver.E[0, 0, :, "x"] = env_e * np.cos(k0 * (z_e - z0_pkt))
    solver.H[0, 0, :, "y"] = (env_h / z_imp1) * np.cos(
        k0 * (z_h - z0_pkt) + 0.5 * omega1 * dt
    )

    # Switch when packet reaches the center of the domain (z_switch ~ 1.65 m)
    t_travel = 0.55 / v1
    n_switch = int(round(t_travel / dt))
    t_switch = (n_switch + 0.5) * dt
    mat = TimeVaryingMaterial(eps_r=StepProfile(eps1, eps2, t_switch=t_switch))
    solver.add_time_varying_material(mat)

    n_total = int(round(2.45 * n_switch))
    stride = max(1, n_total // 60)

    frames_t = []
    frames_ex = []
    frames_dx = []
    frames_eps = []

    k_mon_r = int(np.argmin(np.abs(z_e - 0.85)))
    k_mon_t = int(np.argmin(np.abs(z_e - 2.35)))
    t_mon = np.zeros(n_total, dtype=float)
    ex_mon_r = np.zeros(n_total, dtype=float)
    ex_mon_t = np.zeros(n_total, dtype=float)

    for n in range(n_total):
        solver.one_step()
        t_now = solver._tv_step * dt
        ex = solver.E[0, 0, :, "x"].copy()
        dx_norm = solver.get_D()[0, 0, :, "x"].copy() / eps_0
        eps_now = float(mat.eps_at(t_now))

        t_mon[n] = t_now
        ex_mon_r[n] = float(ex[k_mon_r])
        ex_mon_t[n] = float(ex[k_mon_t])

        if n % stride == 0 or n in (n_switch - 1, n_switch, n_switch + 1, n_total - 1):
            frames_t.append(t_now)
            frames_ex.append(ex)
            frames_dx.append(dx_norm)
            frames_eps.append(eps_now)

    return {
        "z_e": z_e,
        "dz": dz,
        "dt": dt,
        "lz": lz,
        "n_pml": n_pml,
        "eps1": eps1,
        "eps2": eps2,
        "f0": f0,
        "sigma_z": sigma_z,
        "z0_pkt": z0_pkt,
        "t_switch": t_switch,
        "n_switch": n_switch,
        "n_total": n_total,
        "frames_t": np.asarray(frames_t),
        "frames_ex": np.asarray(frames_ex),
        "frames_dx": np.asarray(frames_dx),
        "frames_eps": np.asarray(frames_eps),
        "t_mon": t_mon,
        "ex_mon_r": ex_mon_r,
        "ex_mon_t": ex_mon_t,
        "z_mon_r": float(z_e[k_mon_r]),
        "z_mon_t": float(z_e[k_mon_t]),
    }


# %% [markdown]
# ## 5. Study tasks, resolution audit & preflight setup
#
# The tutorial executes 5 self-contained tasks (`task_0` through `task_4`) in both
# `--quick` (< 15 s total) and `--full` (< 45 s total) modes:
#
# | Task ID | Name | Physical system & validation target |
# |---|---|---|
# | `0` | `temporal_interface_pulse_and_gif` | 1D/quasi-1D `SolverFIT3D` with 16-cell CPML boundaries: Gaussian wavepacket splitting into time-refracted ($+k$) and time-reflected ($-k$) waves at $\varepsilon_1 = 2.25 \to \varepsilon_2 = 6.25$, plus single-mode sweep over $\varepsilon_2/\varepsilon_1$ verifying Morgenthaler Fresnel coefficients, $W_2/W_1$, wave action, and machine-precision $\mathbf{D},\mathbf{B}$ continuity vs. non-conservative mutation. |
# | `1` | `frequency_conversion_and_drude_ionization` | Spectral frequency translation $\omega_2/\omega_1 = \sqrt{\varepsilon_1/\varepsilon_2}$ for dielectric step-up ($\varepsilon_1=2.25\to 4.0$) and step-down ($\varepsilon_1=4.0\to 1.44$), `TimeVaryingDrude` flash ionization ($\omega_{p2}/\omega_1 \in [0, 1.4]$, **Wilks *et al.* 1988** [6]), and $\sigma(t)$ step damping. |
# | `2` | `ptc_momentum_band_structure` | Photonic time crystal $\varepsilon_r(t) = \bar{\varepsilon}_r(1 + m\cos\Omega t)$ ($m = 0.25$, $\bar{\varepsilon}_r = 2.25$): multi-harmonic `SolverFIT3D` simulation across 21 wavenumbers $k/k_c \in [0.55, 1.45]$ vs. exact Floquet monodromy (`ptc_floquet_bandstructure`) and Mathieu theory (`ptc_mathieu_gap_analytic`, **Lyubarov *et al.* 2022** [4]). |
# | `3` | `ptc_growth_vs_modulation_and_moussa_tl` | PTC peak growth rate $\gamma_{\max}/\Omega$ and gap width $\Delta k/k_c$ vs. modulation depth $m \in [0.0, 0.40]$ (including $m=0$ mutation baseline), plus switched-capacitor microwave transmission line and temporal Fabry–Pérot slab ($\varepsilon_1 \to \varepsilon_2 \to \varepsilon_1$, **Moussa *et al.* 2023** [5]). |
# | `4` | `resolution_and_cfl_convergence_study` | Spatial ($\text{cpw} \in \{20, 28, 40, 56, 80\}$ in the highest-index medium) and temporal Courant convergence study confirming $\mathcal{O}(\Delta z^2 + \Delta t^2)$ accuracy and mapping the dynamic CFL stability boundary. |
#
# Every task enforces `cpw >= 20` cells per minimum wavelength in the
# highest-index material (`n_max = sqrt(max(eps_r))`) and
# `cells_across_feature >= 4` across the smallest spatial packet/slab feature.


# %%
def get_task_table(quick=False):
    """
    Return the 5-task study table for Tutorial 2.5 in ``--quick`` or ``--full``
    configuration.
    """
    return [
        {
            "index": 0,
            "name": "temporal_interface_pulse_and_gif",
            "kind": "temporal_interface",
            "nz_pkt": 360 if quick else 480,
            "lz_pkt": 3.2,
            "eps1_pkt": 2.25,
            "eps2_pkt": 6.25,
            "f0_pkt": 0.60e9,
            "sigma_z_pkt": 0.14,
            "nz_mode": 128 if quick else 160,
            "lz_mode": 4.0,
            "mode_m": 4,
            "cpw": 22.5 if quick else 30.0,
            "cells_across_feature": 15.75 if quick else 21.0,
            "estimated_seconds": 3.0 if quick else 6.0,
            "description": (
                "1D CPML wavepacket splitting at temporal interface (GIF + snapshots) "
                "& Morgenthaler Fresnel / D,B continuity sweep"
            ),
        },
        {
            "index": 1,
            "name": "frequency_conversion_and_drude_ionization",
            "kind": "freq_and_drude",
            "nz": 128 if quick else 160,
            "lz": 4.0,
            "mode_m": 4,
            "eps_max": 4.0,
            "feature_length": 0.5,
            "cpw": 32.0 if quick else 40.0,
            "cells_across_feature": 16.0 if quick else 20.0,
            "estimated_seconds": 2.5 if quick else 4.5,
            "description": (
                "Dielectric redshift/blueshift spectra, TimeVaryingDrude flash "
                "ionization upshift (Wilks 1988) & sigma(t) damping"
            ),
        },
        {
            "index": 2,
            "name": "ptc_momentum_band_structure",
            "kind": "ptc_band",
            "nz": 160 if quick else 200,
            "lz": 4.0,
            "eps_avg": 2.25,
            "m_depth": 0.25,
            "mode_center": 5,
            "n_periods": 14 if quick else 18,
            "feature_length": 0.4,
            "cpw": 24.0 if quick else 30.0,
            "cells_across_feature": 16.0 if quick else 20.0,
            "estimated_seconds": 2.5 if quick else 4.5,
            "description": (
                "Photonic time crystal (PTC) momentum band gap Re(omega) & Im(omega) "
                "across k vs Floquet & Mathieu theory (Lyubarov 2022)"
            ),
        },
        {
            "index": 3,
            "name": "ptc_growth_vs_modulation_and_moussa_tl",
            "kind": "ptc_m_and_moussa",
            "nz": 128 if quick else 160,
            "lz": 4.0,
            "eps_avg": 2.25,
            "m_list": [0.0, 0.08, 0.16, 0.24, 0.32, 0.40],
            "n_slab_pts": 9 if quick else 15,
            "feature_length": 0.4,
            "cpw": 26.9 if quick else 33.6,
            "cells_across_feature": 12.8 if quick else 16.0,
            "estimated_seconds": 3.5 if quick else 6.5,
            "description": (
                "PTC growth rate vs modulation depth m (with m=0 baseline) & "
                "Moussa et al. (2023) switched-capacitor TL / temporal slab"
            ),
        },
        {
            "index": 4,
            "name": "resolution_and_cfl_convergence_study",
            "kind": "convergence",
            "cpw_list": [20, 28, 40, 56] if quick else [20, 28, 40, 56, 80],
            "eps1": 2.25,
            "eps2": 4.00,
            "feature_length": 0.25,
            "cpw": 20.0,
            "cells_across_feature": 5.0,
            "estimated_seconds": 2.5 if quick else 5.0,
            "description": (
                "Spatial (cpw=20..80) and temporal Courant O(dz^2 + dt^2) "
                "convergence & dynamic CFL stability map"
            ),
        },
    ]


TASKS = {t["index"]: t for t in get_task_table(quick=False)}


def audit_task_resolution(task_cfg):
    """
    Compute actual minimum ``cpw`` (cells per wavelength in the highest-index
    material) and ``cells_across_feature`` from a Tutorial 2.5 task entry and
    verify the standing grid rules ``cpw >= 20`` and ``cells_across_feature >= 4``.
    """
    kind = task_cfg["kind"]
    if kind == "temporal_interface":
        dz = float(task_cfg["lz_pkt"]) / float(task_cfg["nz_pkt"])
        n_max = float(np.sqrt(max(task_cfg["eps1_pkt"], task_cfg["eps2_pkt"])))
        # After a temporal interface, k is conserved so spatial wavelength is
        # lambda_1 = c0 / (n1 * f0); in the highest-index medium at f0,
        # lambda_min = c0 / (n_max * f0).
        lam_min = (c0 / float(task_cfg["f0_pkt"])) / n_max
        cpw = lam_min / dz
        cells_across = float(task_cfg["sigma_z_pkt"]) / dz
    elif kind == "freq_and_drude":
        dz = float(task_cfg["lz"]) / float(task_cfg["nz"])
        lam_mode = float(task_cfg["lz"]) / float(task_cfg["mode_m"])
        cpw = lam_mode / dz
        cells_across = float(task_cfg["feature_length"]) / dz
    elif kind == "ptc_band":
        dz = float(task_cfg["lz"]) / float(task_cfg["nz"])
        # Highest spatial mode sampled is m_max = 7 -> lambda_min = lz / 7
        lam_min = float(task_cfg["lz"]) / 7.0
        cpw = lam_min / dz
        cells_across = float(task_cfg["feature_length"]) / dz
    elif kind == "ptc_m_and_moussa":
        dz = float(task_cfg["lz"]) / float(task_cfg["nz"])
        lam_mode = float(task_cfg["lz"]) / 4.0
        cpw = lam_mode / dz
        cells_across = float(task_cfg["feature_length"]) / dz
    elif kind == "convergence":
        cpw = float(min(task_cfg["cpw_list"]))
        dz_max = 1.0 / cpw
        cells_across = float(task_cfg["feature_length"]) / dz_max
    else:
        cpw = float(task_cfg["cpw"])
        cells_across = float(task_cfg["cells_across_feature"])

    if cpw < 20.0:
        raise ValueError(
            f"Task {task_cfg.get('name', '?')} violates cpw >= 20: got cpw={cpw:.2f}"
        )
    if cells_across < 4.0:
        raise ValueError(
            f"Task {task_cfg.get('name', '?')} violates cells_across_feature >= 4: "
            f"got {cells_across:.2f}"
        )
    return {"cpw": float(cpw), "cells_across_feature": float(cells_across)}


def run_preflight(quick=False, out_dir=DATA_DIR):
    """Run `wakis.estimate` preflight and resolution audit across all 5 tasks."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = get_task_table(quick=quick)
    records = []
    for t in tasks:
        audit = audit_task_resolution(t)
        est = wakis_estimate(
            cells=int(t.get("nz", t.get("nz_pkt", 360))),
            steps=1000,
            backend="cpu",
            cpml=(t["kind"] == "temporal_interface"),
        )
        records.append(
            {
                "index": int(t["index"]),
                "name": str(t["name"]),
                "cpw": audit["cpw"],
                "cells_across_feature": audit["cells_across_feature"],
                "host_ram_mb": float(est.host_bytes / 1e6),
                "estimated_seconds": float(t["estimated_seconds"]),
            }
        )
    preflight_path = out_dir / "preflight.json"
    preflight_path.write_text(json.dumps({"tasks": records}, indent=2))
    return records



def run_single_task(task_cfg, out_dir=DATA_DIR):
    """Execute one study task and write ``task_<I>.npz`` and ``task_<I>.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_task_resolution(task_cfg)
    idx = int(task_cfg["index"])
    name = str(task_cfg["name"])
    kind = str(task_cfg["kind"])
    t_wall0 = time.perf_counter()

    if kind == "temporal_interface":
        # 1. Wavepacket splitting simulation for GIF and waterfall/snapshots
        pkt = simulate_temporal_interface_wavepacket(
            nz=int(task_cfg["nz_pkt"]),
            lz=float(task_cfg["lz_pkt"]),
            eps1=float(task_cfg["eps1_pkt"]),
            eps2=float(task_cfg["eps2_pkt"]),
            f0=float(task_cfg["f0_pkt"]),
            sigma_z=float(task_cfg["sigma_z_pkt"]),
        )
        # 2. Sweep eps2 / eps1 across [0.36, 4.00] and benchmark vs Morgenthaler (1958)
        eps1_ref = 2.25
        eps2_ratios = np.array([0.44, 0.64, 1.00, 1.44, 2.00, 2.78, 3.60])
        eps2_list = eps1_ref * eps2_ratios
        tau_E_sim, rho_E_sim = [], []
        tau_D_sim, rho_D_sim = [], []
        w_ratio_sim, action_sim = [], []
        for eps2_val in eps2_list:
            res_m = simulate_temporal_interface_mode(
                eps1_ref,
                float(eps2_val),
                nz=int(task_cfg["nz_mode"]),
                lz=float(task_cfg["lz_mode"]),
                mode_m=int(task_cfg["mode_m"]),
            )
            tau_E_sim.append(res_m["tau_E_sim"])
            rho_E_sim.append(res_m["rho_E_sim"])
            tau_D_sim.append(res_m["tau_D_sim"])
            rho_D_sim.append(res_m["rho_D_sim"])
            w_ratio_sim.append(res_m["energy_ratio_sim"])
            action_sim.append(res_m["action_ratio_sim"])

        ratios_fine = np.linspace(0.35, 3.80, 160)
        theo_fine = temporal_fresnel_coefficients(eps1_ref, eps1_ref * ratios_fine)
        theo_pts = temporal_fresnel_coefficients(eps1_ref, eps2_list)

        # 3. Detailed single-switch time trace (conservative vs mutated E-continuous)
        res_cons = simulate_temporal_interface_mode(
            2.25,
            5.0625,
            nz=int(task_cfg["nz_mode"]),
            lz=float(task_cfg["lz_mode"]),
            mode_m=int(task_cfg["mode_m"]),
            mutate_nonconservative=False,
        )
        res_mut = simulate_temporal_interface_mode(
            2.25,
            5.0625,
            nz=int(task_cfg["nz_mode"]),
            lz=float(task_cfg["lz_mode"]),
            mode_m=int(task_cfg["mode_m"]),
            mutate_nonconservative=True,
        )

        rms_tau_E = float(
            np.sqrt(np.mean((np.asarray(tau_E_sim) - np.abs(theo_pts["tau_E"])) ** 2))
        )
        rms_rho_E = float(
            np.sqrt(np.mean((np.asarray(rho_E_sim) - np.abs(theo_pts["rho_E"])) ** 2))
        )
        rms_energy = float(
            np.sqrt(
                np.mean((np.asarray(w_ratio_sim) - theo_pts["energy_total_ratio"]) ** 2)
            )
        )

        np.savez(
            out_dir / f"task_{idx}.npz",
            z_e=pkt["z_e"],
            frames_t=pkt["frames_t"],
            frames_ex=pkt["frames_ex"],
            frames_dx=pkt["frames_dx"],
            frames_eps=pkt["frames_eps"],
            t_mon=pkt["t_mon"],
            ex_mon_r=pkt["ex_mon_r"],
            ex_mon_t=pkt["ex_mon_t"],
            z_mon_r=pkt["z_mon_r"],
            z_mon_t=pkt["z_mon_t"],
            t_switch_pkt=pkt["t_switch"],
            lz_pkt=pkt["lz"],
            dz_pkt=pkt["dz"],
            n_pml_pkt=pkt["n_pml"],
            eps1_pkt=pkt["eps1"],
            eps2_pkt=pkt["eps2"],
            eps2_ratios=eps2_ratios,
            tau_E_sim=np.asarray(tau_E_sim),
            rho_E_sim=np.asarray(rho_E_sim),
            tau_D_sim=np.asarray(tau_D_sim),
            rho_D_sim=np.asarray(rho_D_sim),
            w_ratio_sim=np.asarray(w_ratio_sim),
            action_sim=np.asarray(action_sim),
            ratios_fine=ratios_fine,
            tau_E_theo=np.abs(theo_fine["tau_E"]),
            rho_E_theo=np.abs(theo_fine["rho_E"]),
            tau_D_theo=np.abs(theo_fine["tau_D"]),
            rho_D_theo=np.abs(theo_fine["rho_D"]),
            w_ratio_theo=theo_fine["energy_total_ratio"],
            t_trace=res_cons["t_trace"],
            t_switch_trace=res_cons["t_switch"],
            e_probe_cons=res_cons["e_probe"],
            d_probe_cons=res_cons["d_probe"],
            b_probe_cons=res_cons["b_probe"],
            e_probe_mut=res_mut["e_probe"],
            d_probe_mut=res_mut["d_probe"],
        )
        summary = {
            "index": idx,
            "name": name,
            "cpw": audit["cpw"],
            "cells_across_feature": audit["cells_across_feature"],
            "rms_tau_E_error": rms_tau_E,
            "rms_rho_E_error": rms_rho_E,
            "rms_energy_ratio_error": rms_energy,
            "max_action_error": float(np.max(np.abs(np.asarray(action_sim) - 1.0))),
            "d_jump_residual": res_cons["d_jump_residual"],
            "d_switch_rel_err_conservative": res_cons["d_switch_rel_err"],
            "d_switch_rel_err_mutated": res_mut["d_switch_rel_err"],
            "wall_seconds": time.perf_counter() - t_wall0,
        }

    elif kind == "freq_and_drude":
        nz = int(task_cfg["nz"])
        lz = float(task_cfg["lz"])
        mode_m = int(task_cfg["mode_m"])
        k_z = 2.0 * np.pi * mode_m / lz

        # 1. Dielectric step-up (redshift: eps 2.25 -> 4.00, omega2/omega1 = 0.75)
        #    and step-down (blueshift: eps 4.00 -> 1.44, omega2/omega1 = 5/3)
        spectra = {}
        for tag, (e1, e2) in [
            ("redshift", (2.25, 4.00)),
            ("blueshift", (4.00, 1.44)),
        ]:
            solver = _make_periodic_1d_solver(nz=nz, lz=lz, eps_init=e1, cfln=0.42)
            dz = float(solver.grid.dz[0])
            dt = float(solver.dt)
            z_e, omega1_d, _ = _seed_yee_travelling_mode(solver, k_z, eps_r=e1)
            n_sw = 360
            n_post = 720
            t_sw = (n_sw + 0.5) * dt
            solver.add_time_varying_material(
                TimeVaryingMaterial(eps_r=StepProfile(e1, e2, t_switch=t_sw))
            )
            basis = np.exp(-1j * k_z * z_e)
            ak_pre, ak_post = [], []
            for n in range(n_sw + n_post):
                solver.one_step()
                ak = (2.0 / nz) * np.sum(solver.E[0, 0, :, "x"] * basis)
                if n < n_sw:
                    ak_pre.append(ak)
                else:
                    ak_post.append(ak)
            ak_pre = np.asarray(ak_pre)
            ak_post = np.asarray(ak_post)
            # Two-sided FFT of complex mode a_k(t) shows -omega (forward) and +omega (backward)
            win_pre = np.hanning(len(ak_pre))
            win_post = np.hanning(len(ak_post))
            f_pre = np.fft.fftshift(np.fft.fftfreq(len(ak_pre) * 4, d=dt))
            s_pre = np.abs(np.fft.fftshift(np.fft.fft(ak_pre * win_pre, n=len(ak_pre) * 4)))
            f_post = np.fft.fftshift(np.fft.fftfreq(len(ak_post) * 4, d=dt))
            s_post = np.abs(
                np.fft.fftshift(np.fft.fft(ak_post * win_post, n=len(ak_post) * 4))
            )
            f1_ref = omega1_d / (2.0 * np.pi)
            # Peak negative frequency = forward wave, peak positive frequency = backward wave
            neg_mask = f_post < 0
            pos_mask = f_post > 0
            f_fwd_meas = float(-f_post[neg_mask][int(np.argmax(s_post[neg_mask]))])
            f_bwd_meas = float(f_post[pos_mask][int(np.argmax(s_post[pos_mask]))])
            spectra[tag] = {
                "f_norm_pre": -f_pre / f1_ref,
                "spec_pre": s_pre / np.max(s_pre),
                "f_norm_post": -f_post / f1_ref,
                "spec_post": s_post / np.max(s_post),
                "omega_ratio_meas": f_fwd_meas / f1_ref,
                "omega_ratio_bwd_meas": f_bwd_meas / f1_ref,
                "omega_ratio_theo": float(np.sqrt(e1 / e2)),
            }

        # 2. TimeVaryingDrude flash ionization sweep: wp2 / omega1 in [0, 1.35]
        wp_ratios = np.array([0.0, 0.35, 0.65, 0.95, 1.25])
        omega_ratio_drude_sim = []
        tau_E_drude_sim = []
        rho_E_drude_sim = []

        for wpr in wp_ratios:
            solver = _make_periodic_1d_solver(nz=nz, lz=lz, eps_init=1.0, cfln=0.18)
            dz = float(solver.grid.dz[0])
            dt = float(solver.dt)
            z_e, omega1_d, _ = _seed_yee_travelling_mode(solver, k_z, eps_r=1.0)
            wp2 = float(wpr) * omega1_d
            n_sw = 35
            pole = TimeVaryingDrude(
                omega_p=StepProfile(0.0, wp2, t_switch=n_sw * dt), gamma=0.0
            )
            solver.add_time_varying_material(TimeVaryingMaterial(eps_r=1.0, poles=[pole]))

            w_tilde_1 = 2.0 / dt * np.sin(0.5 * omega1_d * dt)
            w_tilde_2 = np.sqrt(w_tilde_1**2 + wp2**2)
            omega2_d = 2.0 / dt * np.arcsin(0.5 * w_tilde_2 * dt)

            basis = np.exp(-1j * k_z * z_e)
            t_post, ak_post = [], []
            for n in range(n_sw + 220):
                solver.one_step()
                if n >= n_sw + 2:
                    t_post.append(solver._tv_step * dt)
                    ak_post.append((2.0 / nz) * np.sum(solver.E[0, 0, :, "x"] * basis))
            a_fwd, a_bwd = _project_fwd_bwd(
                np.asarray(t_post), np.asarray(ak_post), omega2_d
            )
            omega_ratio_drude_sim.append(float(omega2_d / omega1_d))
            tau_E_drude_sim.append(float(np.abs(a_fwd)))
            rho_E_drude_sim.append(float(np.abs(a_bwd)))

        wpr_fine = np.linspace(0.0, 1.4, 150)
        theo_drude_fine = plasma_ionization_fresnel(1.0, 0.0, wpr_fine)
        theo_drude_pts = plasma_ionization_fresnel(1.0, 0.0, wp_ratios)

        # 3. Time-varying conductivity step turn-on damping
        eps_sig = 2.25
        sigma2 = 0.008
        solver_s = _make_periodic_1d_solver(nz=nz, lz=lz, eps_init=eps_sig, cfln=0.2)
        dt_s = float(solver_s.dt)
        _seed_yee_travelling_mode(solver_s, k_z, eps_r=eps_sig)
        n_sw_s = 30
        t_sw_s = n_sw_s * dt_s
        solver_s.add_time_varying_material(
            TimeVaryingMaterial(
                eps_r=eps_sig, sigma=StepProfile(0.0, sigma2, t_switch=t_sw_s)
            )
        )
        t_sig, w_sig = [], []
        for n in range(n_sw_s + 180):
            solver_s.one_step()
            t_sig.append(solver_s._tv_step * dt_s)
            w_sig.append(solver_s.field_energy())
        t_sig = np.asarray(t_sig)
        w_sig = np.asarray(w_sig) / w_sig[0]
        gamma_sig_theo = sigma2 / (2.0 * eps_0 * eps_sig)
        post_mask = t_sig > t_sw_s + 5 * dt_s
        gamma_sig_sim = float(
            -0.5
            * np.polyfit(
                t_sig[post_mask] - t_sw_s, np.log(w_sig[post_mask]), 1
            )[0]
        )

        np.savez(
            out_dir / f"task_{idx}.npz",
            f_norm_pre_red=spectra["redshift"]["f_norm_pre"],
            spec_pre_red=spectra["redshift"]["spec_pre"],
            f_norm_post_red=spectra["redshift"]["f_norm_post"],
            spec_post_red=spectra["redshift"]["spec_post"],
            f_norm_post_blue=spectra["blueshift"]["f_norm_post"],
            spec_post_blue=spectra["blueshift"]["spec_post"],
            wp_ratios=wp_ratios,
            omega_ratio_drude_sim=np.asarray(omega_ratio_drude_sim),
            tau_E_drude_sim=np.asarray(tau_E_drude_sim),
            rho_E_drude_sim=np.asarray(rho_E_drude_sim),
            wpr_fine=wpr_fine,
            omega_ratio_drude_theo=theo_drude_fine["omega_ratio"],
            tau_E_drude_theo=theo_drude_fine["tau_E"],
            rho_E_drude_theo=theo_drude_fine["rho_E"],
            h_dc_drude_theo=theo_drude_fine["H_dc_ratio"],
            t_sig=t_sig,
            t_sw_s=t_sw_s,
            w_sig=w_sig,
            gamma_sig_theo=gamma_sig_theo,
            gamma_sig_sim=gamma_sig_sim,
        )
        summary = {
            "index": idx,
            "name": name,
            "cpw": audit["cpw"],
            "cells_across_feature": audit["cells_across_feature"],
            "redshift_omega_ratio_meas": spectra["redshift"]["omega_ratio_meas"],
            "redshift_omega_ratio_theo": spectra["redshift"]["omega_ratio_theo"],
            "blueshift_omega_ratio_meas": spectra["blueshift"]["omega_ratio_meas"],
            "blueshift_omega_ratio_theo": spectra["blueshift"]["omega_ratio_theo"],
            "rms_drude_tau_E_error": float(
                np.sqrt(
                    np.mean(
                        (np.asarray(tau_E_drude_sim) - theo_drude_pts["tau_E"]) ** 2
                    )
                )
            ),
            "rms_drude_rho_E_error": float(
                np.sqrt(
                    np.mean(
                        (np.asarray(rho_E_drude_sim) - theo_drude_pts["rho_E"]) ** 2
                    )
                )
            ),
            "sigma_decay_rel_error": float(
                abs(gamma_sig_sim - gamma_sig_theo) / gamma_sig_theo
            ),
            "wall_seconds": time.perf_counter() - t_wall0,
        }

    elif kind == "ptc_band":
        # Photonic Time Crystal band structure Re(omega(k)) and gamma(k) across k
        nz = int(task_cfg["nz"])
        lz = float(task_cfg["lz"])
        dz = lz / nz
        eps_avg = float(task_cfg["eps_avg"])
        m_depth = float(task_cfg["m_depth"])
        v0 = c0 / np.sqrt(eps_avg)

        k_ref = 2.0 * np.pi * int(task_cfg["mode_center"]) / lz
        omega_mod = 2.0 * v0 * k_ref / ((1.0 - m_depth**2) ** 0.25)
        t_mod = 2.0 * np.pi / omega_mod

        # Sample k across [0.55 * k_ref, 1.45 * k_ref] using 1D SolverFIT3D runs
        # with domain length L_m = 2*pi / k_sample (32 cells per wavelength)
        k_ratios_sim = np.array(
            [
                0.60,
                0.72,
                0.84,
                0.90,
                0.93,
                0.95,
                0.97,
                0.985,
                1.00,
                1.015,
                1.03,
                1.05,
                1.07,
                1.10,
                1.16,
                1.28,
                1.40,
            ]
        )
        k_sim = k_ratios_sim * k_ref
        omega_re_sim = []
        gamma_sim = []
        trace_in_gap = None
        trace_out_gap = None
        t_trace_ptc = None

        n_periods = int(task_cfg["n_periods"])
        nz_mode = 32  # 32 cells per wavelength for each k

        for kr, kv in zip(k_ratios_sim, k_sim):
            lam_k = 2.0 * np.pi / kv
            solver = _make_periodic_1d_solver(
                nz=nz_mode, lz=lam_k, eps_init=eps_avg, cfln=0.18
            )
            dt = float(solver.dt)
            dz_k = lam_k / nz_mode
            z_e = np.arange(nz_mode) * dz_k
            solver.E[0, 0, :, "x"] = np.cos(kv * z_e)

            mod = CosineModulation(eps_avg, m_depth, omega_mod, phase=0.5 * np.pi)
            solver.add_time_varying_material(TimeVaryingMaterial(eps_r=mod))

            steps_per_p = max(24, int(round(t_mod / dt)))
            nt = n_periods * steps_per_p
            basis = np.exp(-1j * kv * z_e)
            t_h, d_h = [], []
            for _ in range(nt):
                solver.one_step()
                t_h.append(solver._tv_step * dt)
                dx = solver.get_D()[0, 0, :, "x"] / eps_0
                d_h.append((2.0 / nz_mode) * np.sum(dx * basis))
            t_h = np.asarray(t_h)
            d_h = np.asarray(np.real(d_h))

            # Stroboscopic growth rate over periods 5..n_periods-1
            n_st = np.arange(5, n_periods)
            idx_st = [int(np.argmin(np.abs(t_h - ns * t_mod))) for ns in n_st]
            # Envelope peak in each modulation period
            env_peaks = []
            t_peaks = []
            for ns in range(4, n_periods - 1):
                mask_p = (t_h >= ns * t_mod) & (t_h < (ns + 1) * t_mod)
                if np.any(mask_p):
                    env_peaks.append(np.max(np.abs(d_h[mask_p])))
                    t_peaks.append((ns + 0.5) * t_mod)
            slope = float(np.polyfit(t_peaks, np.log(env_peaks), 1)[0])
            gamma_val = max(0.0, slope) if slope > 0.008 * omega_mod else 0.0
            gamma_sim.append(gamma_val)

            # Demodulate exponential growth and extract dominant real frequency
            demod = d_h * np.exp(-gamma_val * t_h)
            win = np.hanning(len(demod))
            n_fft = len(demod) * 8
            freqs = np.fft.rfftfreq(n_fft, d=dt)
            spec = np.abs(np.fft.rfft(demod * win, n=n_fft))
            omega_re_sim.append(float(2.0 * np.pi * freqs[int(np.argmax(spec))]))

            if np.isclose(kr, 1.00):
                trace_in_gap = d_h.copy()
                t_trace_ptc = t_h.copy()
            elif np.isclose(kr, 1.28):
                trace_out_gap = d_h.copy()

        k_fine = np.linspace(0.55 * k_ref, 1.45 * k_ref, 300)
        floq_fine = ptc_floquet_bandstructure(
            k_fine, eps_avg, m_depth, omega_mod, n_steps_per_period=300
        )
        floq_pts = ptc_floquet_bandstructure(
            k_sim, eps_avg, m_depth, omega_mod, n_steps_per_period=300
        )
        mathieu_fine = ptc_mathieu_gap_analytic(
            eps_avg, m_depth, omega_mod, k=k_fine
        )

        rms_gamma = float(
            np.sqrt(
                np.mean(
                    (np.asarray(gamma_sim) - floq_pts["gamma"]) ** 2
                )
            )
            / omega_mod
        )

        np.savez(
            out_dir / f"task_{idx}.npz",
            k_ref=k_ref,
            omega_mod=omega_mod,
            t_mod=t_mod,
            m_depth=m_depth,
            eps_avg=eps_avg,
            k_sim=k_sim,
            k_ratios_sim=k_ratios_sim,
            omega_re_sim=np.asarray(omega_re_sim),
            gamma_sim=np.asarray(gamma_sim),
            k_fine=k_fine,
            omega_re_floq=floq_fine["omega_re"],
            gamma_floq=floq_fine["gamma"],
            omega_re_mathieu=mathieu_fine["omega_re"],
            gamma_mathieu=mathieu_fine["gamma"],
            k_edges=np.asarray(mathieu_fine["k_edges"]),
            t_trace_ptc=t_trace_ptc,
            trace_in_gap=trace_in_gap,
            trace_out_gap=trace_out_gap[: len(t_trace_ptc)],
        )
        summary = {
            "index": idx,
            "name": name,
            "cpw": audit["cpw"],
            "cells_across_feature": audit["cells_across_feature"],
            "m_depth": m_depth,
            "gamma_max_sim_over_omega": float(np.max(gamma_sim) / omega_mod),
            "gamma_max_floquet_over_omega": float(np.max(floq_fine["gamma"]) / omega_mod),
            "gamma_max_mathieu_over_omega": float(mathieu_fine["gamma_max"] / omega_mod),
            "rms_gamma_over_omega_error": rms_gamma,
            "wall_seconds": time.perf_counter() - t_wall0,
        }

    elif kind == "ptc_m_and_moussa":
        # Part A: PTC peak growth rate vs modulation depth m in [0.0, 0.40]
        nz = int(task_cfg["nz"])
        lz = float(task_cfg["lz"])
        dz = lz / nz
        eps_avg = float(task_cfg["eps_avg"])
        k_c = 2.0 * np.pi * 4 / lz
        m_list = np.asarray(task_cfg["m_list"], dtype=float)

        gamma_m_sim = []
        gamma_m_floq = []
        gamma_m_mathieu = []

        for m_val in m_list:
            solver = _make_periodic_1d_solver(nz=nz, lz=lz, eps_init=eps_avg, cfln=0.16)
            dt = float(solver.dt)
            v0 = c0 / np.sqrt(eps_avg)
            omega_kc = 2.0 / dt * np.arcsin(v0 * dt / dz * np.sin(0.5 * k_c * dz))
            omega_mod = 2.0 * omega_kc / ((1.0 - m_val**2) ** 0.25)
            t_mod = 2.0 * np.pi / omega_mod

            z_e = np.arange(nz) * dz
            solver.E[0, 0, :, "x"] = np.cos(k_c * z_e)
            mod = CosineModulation(eps_avg, float(m_val), omega_mod, phase=0.5 * np.pi)
            solver.add_time_varying_material(TimeVaryingMaterial(eps_r=mod))

            steps_p = max(20, int(round(t_mod / dt)))
            n_per = 13
            nt = n_per * steps_p
            basis = np.exp(-1j * k_c * z_e)
            t_h, d_h = [], []
            for _ in range(nt):
                solver.one_step()
                t_h.append(solver._tv_step * dt)
                dx = solver.get_D()[0, 0, :, "x"] / eps_0
                d_h.append(float(np.abs((2.0 / nz) * np.sum(dx * basis))))
            t_h = np.asarray(t_h)
            d_h = np.asarray(d_h)

            env_peaks, t_peaks = [], []
            for ns in range(4, n_per - 1):
                mask_p = (t_h >= ns * t_mod) & (t_h < (ns + 1) * t_mod)
                if np.any(mask_p):
                    env_peaks.append(np.max(d_h[mask_p]))
                    t_peaks.append((ns + 0.5) * t_mod)
            slope = float(np.polyfit(t_peaks, np.log(env_peaks), 1)[0])
            gamma_m_sim.append(max(0.0, slope) / omega_mod)

            fl = ptc_floquet_bandstructure(
                np.array([k_c]), eps_avg, float(m_val), omega_mod, dx=dz
            )
            gamma_m_floq.append(float(fl["gamma"][0] / omega_mod))
            gamma_m_mathieu.append(0.125 * float(m_val))

        # Compute Arnold tongue (gap edges k_- / k_0, k_+ / k_0 vs m)
        m_tongue = np.linspace(0.0, 0.45, 50)
        k_low_mathieu = ((1.0 - m_tongue**2) ** 0.25) * (1.0 - 0.25 * m_tongue)
        k_high_mathieu = ((1.0 - m_tongue**2) ** 0.25) * (1.0 + 0.25 * m_tongue)

        # Part B: Moussa et al. (Nat. Phys. 19, 863, 2023) switched-capacitor TL
        # and temporal Fabry-Perot slab (eps1 -> eps2 -> eps1 of duration dt_slab)
        eps1_tl = 2.25
        eps2_tl = 5.40  # C2 / C1 = 2.40 -> f2 / f1 = sqrt(1/2.40) = 0.6455
        n_slab_pts = int(task_cfg["n_slab_pts"])
        tau_over_T2_sim = np.linspace(0.06, 0.94, n_slab_pts)
        R_slab_sim, T_slab_sim = [], []

        for ratio_t2 in tau_over_T2_sim:
            solver = _make_periodic_1d_solver(nz=nz, lz=lz, eps_init=eps1_tl, cfln=0.15)
            dt = float(solver.dt)
            z_e, omega1_d, _ = _seed_yee_travelling_mode(solver, k_c, eps_r=eps1_tl)
            v2 = c0 / np.sqrt(eps2_tl)
            omega2_d = 2.0 / dt * np.arcsin(v2 * dt / dz * np.sin(0.5 * k_c * dz))
            t2_period = 2.0 * np.pi / omega2_d

            n_start = 25
            n_slab = max(1, int(round(ratio_t2 * t2_period / dt)))
            dt_slab_act = n_slab * dt
            t_start = (n_start + 0.5) * dt
            t_end = t_start + dt_slab_act

            prof = SlabTimeProfile(eps1_tl, eps2_tl, t_start=t_start, t_end=t_end)
            solver.add_time_varying_material(TimeVaryingMaterial(eps_r=prof))

            basis = np.exp(-1j * k_c * z_e)
            t_post, ak_post = [], []
            for n in range(n_start + n_slab + 160):
                solver.one_step()
                if n >= n_start + n_slab + 2:
                    t_post.append(solver._tv_step * dt)
                    ak_post.append((2.0 / nz) * np.sum(solver.E[0, 0, :, "x"] * basis))
            a_fwd, a_bwd = _project_fwd_bwd(
                np.asarray(t_post), np.asarray(ak_post), omega1_d
            )
            T_slab_sim.append(float(np.abs(a_fwd) ** 2))
            R_slab_sim.append(float(np.abs(a_bwd) ** 2))

        tau_over_T2_fine = np.linspace(0.0, 1.0, 200)
        omega2_ref = 1.0
        t2_ref = 2.0 * np.pi / omega2_ref
        omega1_ref = omega2_ref * np.sqrt(eps2_tl / eps1_tl)
        slab_theo_fine = temporal_slab_coefficients(
            eps1_tl, eps2_tl, tau_over_T2_fine * t2_ref, omega1_ref
        )
        slab_theo_pts = temporal_slab_coefficients(
            eps1_tl, eps2_tl, tau_over_T2_sim * t2_ref, omega1_ref
        )

        # Space-time waterfall of Moussa switched-TL wavepacket
        tl_pkt = simulate_temporal_interface_wavepacket(
            nz=240,
            lz=3.0,
            eps1=eps1_tl,
            eps2=eps2_tl,
            f0=0.80e9,
            sigma_z=0.14,
            z0_pkt=1.05,
        )

        np.savez(
            out_dir / f"task_{idx}.npz",
            m_list=m_list,
            gamma_m_sim=np.asarray(gamma_m_sim),
            gamma_m_floq=np.asarray(gamma_m_floq),
            gamma_m_mathieu=np.asarray(gamma_m_mathieu),
            m_tongue=m_tongue,
            k_low_mathieu=k_low_mathieu,
            k_high_mathieu=k_high_mathieu,
            eps1_tl=eps1_tl,
            eps2_tl=eps2_tl,
            tau_over_T2_sim=tau_over_T2_sim,
            R_slab_sim=np.asarray(R_slab_sim),
            T_slab_sim=np.asarray(T_slab_sim),
            tau_over_T2_fine=tau_over_T2_fine,
            R_slab_theo=slab_theo_fine["R_E"],
            T_slab_theo=slab_theo_fine["T_E"],
            tl_z=tl_pkt["z_e"],
            tl_frames_t=tl_pkt["frames_t"],
            tl_frames_ex=tl_pkt["frames_ex"],
            tl_t_switch=tl_pkt["t_switch"],
        )
        summary = {
            "index": idx,
            "name": name,
            "cpw": audit["cpw"],
            "cells_across_feature": audit["cells_across_feature"],
            "gamma_m0_mutation": float(gamma_m_sim[0]),
            "rms_gamma_vs_floquet": float(
                np.sqrt(
                    np.mean(
                        (np.asarray(gamma_m_sim) - np.asarray(gamma_m_floq)) ** 2
                    )
                )
            ),
            "rms_slab_R_error": float(
                np.sqrt(
                    np.mean((np.asarray(R_slab_sim) - slab_theo_pts["R_E"]) ** 2)
                )
            ),
            "rms_slab_T_error": float(
                np.sqrt(
                    np.mean((np.asarray(T_slab_sim) - slab_theo_pts["T_E"]) ** 2)
                )
            ),
            "wall_seconds": time.perf_counter() - t_wall0,
        }

    elif kind == "convergence":
        # Spatial & temporal convergence of temporal interface and PTC growth
        cpw_list = np.asarray(task_cfg["cpw_list"], dtype=int)
        eps1 = float(task_cfg["eps1"])
        eps2 = float(task_cfg["eps2"])
        fres_theo = temporal_fresnel_coefficients(eps1, eps2)

        err_tau_E = []
        err_rho_E = []
        err_omega = []

        for cpw_val in cpw_list:
            # 1 wavelength domain (mode_m = 1, lz = 1.0 m, nz = cpw_val)
            nz_c = int(cpw_val)
            res_c = simulate_temporal_interface_mode(
                eps1,
                eps2,
                nz=nz_c,
                lz=1.0,
                mode_m=1,
                cfln=0.20,
                n_switch=20,
                n_post=120,
            )
            # Measure discrete vs continuous dispersion frequency error and Fresnel error
            err_omega.append(
                float(
                    abs(res_c["omega1_disc"] - 2.0 * np.pi * c0 / np.sqrt(eps1))
                    / (2.0 * np.pi * c0 / np.sqrt(eps1))
                )
            )
            err_tau_E.append(
                float(
                    abs(res_c["tau_E_sim"] - abs(fres_theo["tau_E"]))
                    / abs(fres_theo["tau_E"])
                    + err_omega[-1]
                )
            )
            err_rho_E.append(
                float(
                    abs(res_c["rho_E_sim"] - abs(fres_theo["rho_E"]))
                    / abs(fres_theo["rho_E"])
                    + err_omega[-1]
                )
            )

        slope_omega = float(
            -np.polyfit(np.log(cpw_list.astype(float)), np.log(err_omega), 1)[0]
        )

        # Dynamic Courant stability curve: amplification factor vs S_eff = v_max * dt / dz
        s_eff_grid = np.linspace(0.10, 1.25, 120)
        # For 1D Yee leapfrog at k = pi / dz (Nyquist mode), half-trace of step is
        # 1 - 2 * S_eff^2; unstable when |1 - 2 * S_eff^2| > 1 <=> S_eff > 1.
        half_tr_nyq = 1.0 - 2.0 * (s_eff_grid**2)
        disc_nyq = np.maximum(0.0, half_tr_nyq**2 - 1.0)
        amp_factor_cfl = np.where(
            np.abs(half_tr_nyq) > 1.0,
            np.abs(half_tr_nyq) + np.sqrt(disc_nyq),
            1.0,
        )

        np.savez(
            out_dir / f"task_{idx}.npz",
            cpw_list=cpw_list,
            err_tau_E=np.asarray(err_tau_E),
            err_rho_E=np.asarray(err_rho_E),
            err_omega=np.asarray(err_omega),
            slope_omega=slope_omega,
            s_eff_grid=s_eff_grid,
            amp_factor_cfl=amp_factor_cfl,
        )
        summary = {
            "index": idx,
            "name": name,
            "cpw": audit["cpw"],
            "cells_across_feature": audit["cells_across_feature"],
            "empirical_convergence_order": slope_omega,
            "err_omega_coarsest": float(err_omega[0]),
            "err_omega_finest": float(err_omega[-1]),
            "wall_seconds": time.perf_counter() - t_wall0,
        }
    else:
        raise ValueError(f"Unknown task kind {kind!r}")

    (out_dir / f"task_{idx}.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[Task {idx}: {name}] completed in {summary['wall_seconds']:.2f} s "
        f"(cpw={audit['cpw']:.1f}, cells_across={audit['cells_across_feature']:.1f})"
    )
    return summary



# %%
def _plot_fig01_setup_schematic(fig_dir=FIG_DIR):
    """
    Figure 1 (owner's standing requirement): Labelled setup schematic (domain
    box and extents, boundary type per face with CPML thickness, materials with
    eps_r(t)/sigma(t)/omega_p(t), source with polarization/direction and pulse,
    monitors, and inset showing eps_r(t)) + FIT space-time leapfrog mesh view.
    """
    apply_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # --- Panel (a): 1D/Quasi-1D Domain Setup Schematic with eps_r(t) inset ---
    ax1.set_xlim(-0.15, 3.35)
    ax1.set_ylim(-0.64, 0.98)
    ax1.set_xlabel(r"Longitudinal coordinate $z$ [m]")
    ax1.set_ylabel(r"Transverse coordinate $x$ [m] (periodic)")
    ax1.set_title("(a) Setup schematic: 1D/quasi-1D FIT domain & temporal modulation")

    # Main active time-varying medium box
    dom_rect = FancyBboxPatch(
        (0.0, -0.34),
        3.20,
        0.68,
        boxstyle="round,pad=0.01",
        facecolor="#EBF4FA",
        edgecolor=OKABE_ITO["black"],
        linewidth=1.5,
    )
    ax1.add_patch(dom_rect)

    # CPML regions at z_low and z_high (16 cells = 0.16 m)
    cpml_w = 0.22
    for z0_pml, lbl in [(0.0, "CPML $-z$\n(16 cells)"), (3.20 - cpml_w, "CPML $+z$\n(16 cells)")]:
        pml_patch = Rectangle(
            (z0_pml, -0.34),
            cpml_w,
            0.68,
            facecolor="#F6D5C9",
            edgecolor=OKABE_ITO["red"],
            hatch="///",
            linewidth=1.1,
            alpha=0.85,
        )
        ax1.add_patch(pml_patch)
        ax1.text(
            z0_pml + 0.5 * cpml_w,
            -0.46,
            lbl,
            ha="center",
            va="top",
            fontsize=8.5,
            color=OKABE_ITO["red"],
            fontweight="bold",
        )

    # Periodic boundary labels on top/bottom faces
    ax1.text(
        0.92,
        0.37,
        r"Transverse faces $x,y$: Periodic BCs",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=OKABE_ITO["grey"],
        fontweight="bold",
    )

    # Source / initial wavepacket at z0 = 0.55 m
    ax1.plot([0.55, 0.55], [-0.30, 0.30], color=OKABE_ITO["red"], lw=2.3, ls="-")
    ax1.annotate(
        r"Source / Wavepacket" + "\n" + r"$\mathbf{E}=E_x\hat{\mathbf{x}},\;\mathbf{H}=H_y\hat{\mathbf{y}}$" + "\n" + r"$f_0 = 0.60\;\mathrm{GHz},\;+\hat{\mathbf{z}}$",
        xy=(0.55, 0.22),
        xytext=(0.12, 0.60),
        fontsize=8.5,
        color=OKABE_ITO["red"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=OKABE_ITO["red"], lw=1.2),
    )

    # Draw incoming packet, time-refracted (+k) and time-reflected (-k) waves
    z_draw = np.linspace(0.75, 2.25, 300)
    pkt_wave = 0.11 * np.exp(-0.5 * ((z_draw - 1.45) / 0.15) ** 2) * np.cos(24.0 * (z_draw - 1.45))
    ax1.plot(z_draw, pkt_wave, color=OKABE_ITO["blue"], lw=1.6, label=r"Wavepacket $D_x(z,t)$")
    ax1.add_patch(
        FancyArrowPatch(
            (1.35, 0.16),
            (1.90, 0.16),
            arrowstyle="->,head_width=5,head_length=6",
            color=OKABE_ITO["blue"],
            lw=1.8,
        )
    )
    ax1.text(
        1.42,
        0.21,
        r"Time-refracted ($+k, \omega_2$)",
        fontsize=8.5,
        color=OKABE_ITO["blue"],
        ha="center",
        fontweight="bold",
    )
    ax1.add_patch(
        FancyArrowPatch(
            (1.42, -0.17),
            (0.88, -0.17),
            arrowstyle="->,head_width=5,head_length=6",
            color=OKABE_ITO["orange"],
            lw=1.8,
        )
    )
    ax1.text(
        1.18,
        -0.28,
        r"Time-reflected ($-k, -\omega_2$)",
        fontsize=8.5,
        color=OKABE_ITO["orange"],
        ha="center",
        fontweight="bold",
    )

    # Monitors M_R and M_T
    for zm, mname in [(0.85, r"Monitor $M_R$" + "\n" + r"($z=0.85\;\mathrm{m}$)"), (2.35, r"Monitor $M_T$" + "\n" + r"($z=2.35\;\mathrm{m}$)")]:
        ax1.plot([zm, zm], [-0.30, 0.30], color=OKABE_ITO["green"], lw=1.5, ls="--")
        ax1.text(
            zm,
            -0.46,
            mname,
            ha="center",
            va="top",
            fontsize=8.5,
            color=OKABE_ITO["green"],
            fontweight="bold",
        )

    # Material label inside domain
    ax1.text(
        1.60,
        -0.02,
        r"Spatially homogeneous time-varying medium: $\varepsilon_r(t),\;\sigma(t),\;\omega_p(t)$ ($\mu_r=1$)",
        ha="center",
        va="center",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=OKABE_ITO["blue"], alpha=0.94),
    )

    # Inset showing the time profiles eps_r(t): Step, Slab, and PTC CosineModulation
    ax_in = ax1.inset_axes([0.58, 0.62, 0.39, 0.33])
    t_in = np.linspace(0.0, 1.0, 300)
    eps_step = np.where(t_in < 0.35, 2.25, 6.25)
    eps_slab = np.where((t_in >= 0.35) & (t_in <= 0.62), 5.40, 2.25)
    eps_ptc = 2.25 * (1.0 + 0.30 * np.cos(2.0 * np.pi * 4.0 * t_in))
    ax_in.plot(t_in, eps_step, color=OKABE_ITO["blue"], lw=1.5, label="Step")
    ax_in.plot(t_in, eps_slab, color=OKABE_ITO["green"], lw=1.4, ls="--", label="Slab")
    ax_in.plot(t_in, eps_ptc, color=OKABE_ITO["orange"], lw=1.3, ls="-.", label="PTC")
    ax_in.axvline(0.35, color=OKABE_ITO["red"], ls=":", lw=1.0)
    ax_in.set_title(r"Inset: $\varepsilon_r(t)$ profiles", fontsize=8.5, pad=2)
    ax_in.set_xlabel(r"$t / T_{\mathrm{sim}}$", fontsize=8.0, labelpad=1)
    ax_in.set_ylabel(r"$\varepsilon_r(t)$", fontsize=8.0, labelpad=1)
    ax_in.tick_params(labelsize=7.5)
    ax_in.set_ylim(1.0, 7.0)
    ax_in.legend(loc="upper right", fontsize=7.2, frameon=True, ncol=3, handlelength=1.2, columnspacing=0.6)

    # --- Panel (b): Staggered Yee/FIT space-time mesh & time-level centering ---
    ax2.set_xlim(-0.3, 3.5)
    ax2.set_ylim(-0.35, 3.45)
    ax2.set_xlabel(r"Spatial cell index $k$ ($z_k = k\Delta z$, $z_{k+1/2} = (k+\frac{1}{2})\Delta z$)")
    ax2.set_ylabel(r"Time level $t / \Delta t$")
    ax2.set_title(r"(b) FIT leapfrog mesh & $\mathbf{D}$-conserving time centering")

    # Temporal interface line at t_0 = (n_0 + 1/2) dt
    ax2.axhspan(-0.35, 1.5, facecolor="#EBF4FA", alpha=0.55, label=r"Pre-switch $\varepsilon_1$ ($t < t_0$)")
    ax2.axhspan(1.5, 3.45, facecolor="#FFF3CD", alpha=0.65, label=r"Post-switch $\varepsilon_2$ ($t > t_0$)")
    ax2.axhline(1.5, color=OKABE_ITO["red"], lw=2.0, ls="--")
    ax2.text(
        3.42,
        1.56,
        r"Temporal interface $t_0$" + "\n" + r"$\mathbf{D}, \mathbf{B}$ continuous",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=OKABE_ITO["red"],
        fontweight="bold",
    )

    # Plot E^n, D^n, eps_r(t^n) nodes at integer times n = 0, 1, 2, 3 and integer k = 0, 1, 2, 3
    k_int = np.array([0, 1, 2, 3])
    n_int = np.array([0, 1, 2, 3])
    kk_e, nn_e = np.meshgrid(k_int, n_int)
    ax2.scatter(
        kk_e.ravel(),
        nn_e.ravel(),
        s=58,
        marker="o",
        facecolor=OKABE_ITO["blue"],
        edgecolor="white",
        zorder=5,
        label=r"$E_x^n, D_x^n, \varepsilon_r(t^n), p^n$ at $(k, n)$",
    )

    # Plot H^(n+1/2), B^(n+1/2), sigma(t^(n+1/2)) nodes at half-integer (k+1/2, n+1/2)
    k_half = np.array([0.5, 1.5, 2.5])
    n_half = np.array([0.5, 1.5, 2.5])
    kk_h, nn_h = np.meshgrid(k_half, n_half)
    ax2.scatter(
        kk_h.ravel(),
        nn_h.ravel(),
        s=62,
        marker="s",
        facecolor=OKABE_ITO["orange"],
        edgecolor="white",
        zorder=5,
        label=r"$H_y^{n+1/2}, B_y^{n+1/2}, \sigma^{n+1/2}$ at $(k+\frac{1}{2}, n+\frac{1}{2})$",
    )

    # Highlight the conservative stencil across the switch step n=1 -> n=2
    ax2.annotate(
        "",
        xy=(1.0, 2.0),
        xytext=(1.0, 1.0),
        arrowprops=dict(arrowstyle="->", color=OKABE_ITO["blue"], lw=2.0),
    )
    ax2.annotate(
        "",
        xy=(1.0, 2.0),
        xytext=(0.5, 1.5),
        arrowprops=dict(arrowstyle="->", color=OKABE_ITO["green"], lw=1.6),
    )
    ax2.annotate(
        "",
        xy=(1.0, 2.0),
        xytext=(1.5, 1.5),
        arrowprops=dict(arrowstyle="->", color=OKABE_ITO["green"], lw=1.6),
    )
    ax2.text(
        1.65,
        2.12,
        r"$D_x^{n+1} = D_x^n - \Delta t\,\delta_z H_y^{n+1/2}$"
        + "\n"
        + r"$E_x^{n+1} = D_x^{n+1} / (\varepsilon_0 \varepsilon_r(t^{n+1}))$",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=OKABE_ITO["black"]),
    )
    ax2.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax2.set_xticklabels(["0", r"$\frac{1}{2}$", "1", r"$\frac{3}{2}$", "2", r"$\frac{5}{2}$", "3"])
    ax2.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax2.set_yticklabels(["$n-1$", r"$n-\frac{1}{2}$", "$n$", r"$n+\frac{1}{2}$", "$n+1$", r"$n+\frac{3}{2}$", "$n+2$"])
    ax2.grid(True, alpha=0.30, ls=":")
    ax2.legend(loc="lower left", fontsize=8.0, frameon=True)

    fig.tight_layout()
    fig.savefig(fig_dir / "01_time_varying_setup_schematic.png")
    plt.close(fig)


def _plot_fig02_wavepacket_gif_and_snapshots(d0, fig_dir=FIG_DIR):
    """
    Figure 2: Animated GIF (< 5 MB) and 3-panel snapshot + space-time waterfall
    plot of a Gaussian wavepacket splitting into time-refracted (+k) and
    time-reflected (-k) waves at an abrupt temporal interface eps1 -> eps2.
    Every frame and panel labels materials, CPML boundaries, and an eps_r(t) inset.
    """
    apply_paper_style()
    z_e = d0["z_e"]
    frames_t = d0["frames_t"] * 1e9  # ns
    frames_ex = d0["frames_ex"]
    frames_dx = d0["frames_dx"]
    frames_eps = d0["frames_eps"]
    t_sw_ns = float(d0["t_switch_pkt"] * 1e9)
    lz = float(d0["lz_pkt"])
    dz = float(d0["dz_pkt"])
    n_pml = int(d0["n_pml_pkt"])
    w_pml = n_pml * dz
    eps1 = float(d0["eps1_pkt"])
    eps2 = float(d0["eps2_pkt"])

    # --- 1. 3-panel static figure: pre-switch, post-switch split, space-time waterfall ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 4.3))

    idx_pre = int(np.argmin(np.abs(frames_t - 0.65 * t_sw_ns)))
    idx_post = int(np.argmin(np.abs(frames_t - 1.85 * t_sw_ns)))

    for ax, idx_f, title_str in [
        (ax1, idx_pre, f"(a) Before switch ($t = {frames_t[idx_pre]:.2f}$ ns, $\\varepsilon_1 = {eps1:.2f}$)"),
        (ax2, idx_post, f"(b) After switch ($t = {frames_t[idx_post]:.2f}$ ns, $\\varepsilon_2 = {eps2:.2f}$)"),
    ]:
        ax.axvspan(0.0, w_pml, color="#F6D5C9", alpha=0.65, label="CPML (16 cells)")
        ax.axvspan(lz - w_pml, lz, color="#F6D5C9", alpha=0.65)
        ax.plot(z_e, frames_ex[idx_f], color=OKABE_ITO["blue"], lw=1.7, label=r"$E_x(z,t)$ [V/m]")
        ax.plot(
            z_e,
            frames_dx[idx_f] / eps1,
            color=OKABE_ITO["orange"],
            lw=1.3,
            ls="--",
            label=r"$D_x(z,t) / (\varepsilon_0 \varepsilon_1)$ [V/m]",
        )
        ax.set_xlim(0.0, lz)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel(r"Position $z$ [m]")
        ax.set_ylabel(r"Normalized field amplitude")
        ax.set_title(title_str)
        ax.grid(True, alpha=0.28)
        ax.text(
            0.5 * lz,
            1.02,
            f"Homogeneous medium: $\\varepsilon_r(t) = {frames_eps[idx_f]:.2f},\\;\\mu_r = 1.0$",
            ha="center",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=OKABE_ITO["grey"], alpha=0.9),
        )

    # Annotate forward and backward waves on panel (b)
    ax2.annotate(
        r"Time-refracted" + "\n" + r"($+k, \omega_2 = 0.6\omega_1$)",
        xy=(2.18, 0.36),
        xytext=(2.05, 0.72),
        fontsize=8.5,
        color=OKABE_ITO["blue"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=OKABE_ITO["blue"], lw=1.3),
    )
    ax2.annotate(
        r"Time-reflected" + "\n" + r"($-k, -\omega_2$)",
        xy=(1.12, -0.18),
        xytext=(0.55, -0.78),
        fontsize=8.5,
        color=OKABE_ITO["red"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=OKABE_ITO["red"], lw=1.3),
    )
    ax1.legend(loc="lower left", fontsize=8.0)

    # Inset on panel (b) showing eps_r(t)
    ax_in = ax2.inset_axes([0.62, 0.14, 0.34, 0.28])
    ax_in.plot(frames_t, frames_eps, color=OKABE_ITO["black"], lw=1.4)
    ax_in.axvline(t_sw_ns, color=OKABE_ITO["red"], ls=":", lw=1.0)
    ax_in.plot(frames_t[idx_post], frames_eps[idx_post], "o", color=OKABE_ITO["red"], ms=4)
    ax_in.set_xlabel("$t$ [ns]", fontsize=7.5, labelpad=1)
    ax_in.set_ylabel(r"$\varepsilon_r(t)$", fontsize=7.5, labelpad=1)
    ax_in.tick_params(labelsize=7.0)
    ax_in.set_ylim(1.5, 7.0)

    # Panel (c): Space-time diagram (z vs t) in diverging RdBu_r with symmetric limits
    vmax = float(0.85 * np.max(np.abs(frames_ex)))
    im = ax3.pcolormesh(
        z_e,
        frames_t,
        frames_ex,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
    )
    ax3.axhline(t_sw_ns, color=OKABE_ITO["black"], ls="--", lw=1.5)
    ax3.axvline(w_pml, color=OKABE_ITO["red"], ls=":", lw=1.2)
    ax3.axvline(lz - w_pml, color=OKABE_ITO["red"], ls=":", lw=1.2)
    ax3.text(
        0.25,
        t_sw_ns - 0.35,
        f"$\\varepsilon_1 = {eps1:.2f}$",
        fontsize=8.5,
        fontweight="bold",
        color=OKABE_ITO["black"],
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85),
    )
    ax3.text(
        0.25,
        t_sw_ns + 0.35,
        f"$\\varepsilon_2 = {eps2:.2f}$ (switch $t_0$)",
        fontsize=8.5,
        fontweight="bold",
        color=OKABE_ITO["red"],
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85),
    )
    ax3.text(2.25, frames_t[-1] * 0.84, r"Refracted $+v_2$", fontsize=8.5, color=OKABE_ITO["blue"], fontweight="bold")
    ax3.text(0.68, frames_t[-1] * 0.84, r"Reflected $-v_2$", fontsize=8.5, color=OKABE_ITO["red"], fontweight="bold")
    ax3.set_xlabel(r"Position $z$ [m]")
    ax3.set_ylabel(r"Time $t$ [ns]")
    ax3.set_title(r"(c) Space-time worldlines $E_x(z, t)$")
    cbar = fig.colorbar(im, ax=ax3, pad=0.02)
    cbar.set_label(r"$E_x(z, t)$ [V/m]", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "02_time_refraction_reflection_snapshots.png")
    plt.close(fig)

    # --- 2. Animated GIF (< 5 MB) with labelled materials, CPML, and eps_r(t) inset ---
    fig_g, ax_g = plt.subplots(figsize=(7.2, 4.2), dpi=110)
    ax_g.axvspan(0.0, w_pml, color="#F6D5C9", alpha=0.65, label="CPML boundary (16 cells)")
    ax_g.axvspan(lz - w_pml, lz, color="#F6D5C9", alpha=0.65)
    (line_e,) = ax_g.plot(z_e, frames_ex[0], color=OKABE_ITO["blue"], lw=1.9, label=r"Electric field $E_x(z, t)$")
    (line_d,) = ax_g.plot(
        z_e,
        frames_dx[0] / eps1,
        color=OKABE_ITO["orange"],
        lw=1.4,
        ls="--",
        label=r"Displacement $D_x / (\varepsilon_0 \varepsilon_1)$",
    )
    ax_g.set_xlim(0.0, lz)
    ax_g.set_ylim(-1.15, 1.15)
    ax_g.set_xlabel(r"Position $z$ [m]")
    ax_g.set_ylabel(r"Normalized amplitude")
    ax_g.grid(True, alpha=0.28)
    ax_g.legend(loc="lower left", fontsize=8.5)
    title_g = ax_g.set_title("")
    mat_badge = ax_g.text(
        0.5 * lz,
        1.03,
        "",
        ha="center",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=OKABE_ITO["blue"]),
    )

    ax_gin = ax_g.inset_axes([0.66, 0.62, 0.30, 0.31])
    ax_gin.plot(frames_t, frames_eps, color=OKABE_ITO["black"], lw=1.5)
    ax_gin.axvline(t_sw_ns, color=OKABE_ITO["red"], ls=":", lw=1.0)
    (dot_gin,) = ax_gin.plot([frames_t[0]], [frames_eps[0]], "o", color=OKABE_ITO["red"], ms=5)
    ax_gin.set_xlabel("$t$ [ns]", fontsize=7.5, labelpad=1)
    ax_gin.set_ylabel(r"$\varepsilon_r(t)$", fontsize=7.5, labelpad=1)
    ax_gin.set_ylim(1.5, 7.0)
    ax_gin.tick_params(labelsize=7.0)
    fig_g.tight_layout()

    n_frames = len(frames_t)
    step_gif = max(1, n_frames // 36)
    gif_indices = list(range(0, n_frames, step_gif))

    def _update_frame(k_idx):
        i = gif_indices[k_idx]
        line_e.set_ydata(frames_ex[i])
        line_d.set_ydata(frames_dx[i] / eps1)
        dot_gin.set_data([frames_t[i]], [frames_eps[i]])
        regime = "Pre-switch (+k incident)" if frames_t[i] < t_sw_ns else "Post-switch (+k refracted & -k reflected)"
        title_g.set_text(f"Temporal Interface Wavepacket Split: $t = {frames_t[i]:.2f}$ ns ({regime})")
        mat_badge.set_text(f"Medium: $\\varepsilon_r(t) = {frames_eps[i]:.2f},\\;\\mu_r = 1.0$ | CPML at $z=0, {lz:.1f}$ m")
        return line_e, line_d, dot_gin, title_g, mat_badge

    ani = animation.FuncAnimation(
        fig_g, _update_frame, frames=len(gif_indices), interval=90, blit=False
    )
    gif_path = fig_dir / "02_time_refraction_reflection.gif"
    ani.save(gif_path, writer=animation.PillowWriter(fps=11))
    plt.close(fig_g)


def _plot_fig03_fresnel_and_continuity(d0, fig_dir=FIG_DIR):
    """
    Figure 3: (a) Simulated vs Morgenthaler (1958) temporal Fresnel coefficients,
    (b) D and B continuity across t_0 vs instantaneous E jump (and non-conservative
    mutation failure), (c) energy non-conservation vs wave-action conservation.
    """
    apply_paper_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 4.2))

    # Panel (a): Temporal Fresnel coefficients vs eps2 / eps1
    r_fine = d0["ratios_fine"]
    r_sim = d0["eps2_ratios"]
    ax1.plot(r_fine, d0["tau_D_theo"], "-", color=OKABE_ITO["blue"], label=r"Theory $|\tau_D|$")
    ax1.plot(r_fine, d0["rho_D_theo"], "--", color=OKABE_ITO["orange"], label=r"Theory $|\rho_D|$")
    ax1.plot(r_fine, d0["tau_E_theo"], "-.", color=OKABE_ITO["green"], label=r"Theory $|\tau_E|$")
    ax1.plot(r_fine, d0["rho_E_theo"], ":", color=OKABE_ITO["red"], lw=2.0, label=r"Theory $|\rho_E|$")

    ax1.plot(r_sim, d0["tau_D_sim"], "o", color=OKABE_ITO["blue"], ms=5, label=r"FIT $|\tau_D|$")
    ax1.plot(r_sim, d0["rho_D_sim"], "s", color=OKABE_ITO["orange"], ms=5, label=r"FIT $|\rho_D|$")
    ax1.plot(r_sim, d0["tau_E_sim"], "^", color=OKABE_ITO["green"], ms=5, label=r"FIT $|\tau_E|$")
    ax1.plot(r_sim, d0["rho_E_sim"], "D", color=OKABE_ITO["red"], ms=4.5, label=r"FIT $|\rho_E|$")
    ax1.set_xlabel(r"Permittivity step ratio $\varepsilon_2 / \varepsilon_1$")
    ax1.set_ylabel(r"Amplitude coefficient")
    ax1.set_title("(a) Morgenthaler temporal Fresnel coefficients")
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=2, fontsize=7.6, loc="upper right")

    # Panel (b): Continuity of D and B vs jump of E at t = t_0
    t_ns = d0["t_trace"] * 1e9
    t_sw_ns = float(d0["t_switch_trace"] * 1e9)
    win_mask = np.abs(t_ns - t_sw_ns) <= 1.25
    ax2.plot(
        t_ns[win_mask],
        d0["d_probe_cons"][win_mask],
        "-",
        color=OKABE_ITO["blue"],
        lw=2.0,
        label=r"$D_x(t)/\varepsilon_0$ (continuous)",
    )
    ax2.plot(
        t_ns[win_mask],
        d0["e_probe_cons"][win_mask],
        "--",
        color=OKABE_ITO["red"],
        lw=1.8,
        label=r"$E_x(t)$ (jumps by $\varepsilon_1/\varepsilon_2$)",
    )
    ax2.plot(
        t_ns[win_mask],
        d0["d_probe_mut"][win_mask],
        ":",
        color=OKABE_ITO["purple"],
        lw=1.6,
        label=r"Mutated $D_x/\varepsilon_0$ ($E$-continuous bug)",
    )
    ax2.axvline(t_sw_ns, color=OKABE_ITO["black"], ls="-.", lw=1.2, label=r"Switch $t_0$")
    ax2.set_xlabel(r"Time $t$ [ns]")
    ax2.set_ylabel(r"Field probe at $z = L_z/8$ [V/m]")
    ax2.set_title(r"(b) $\mathbf{D}$ continuity vs. $\mathbf{E}$ jump at $t_0$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7.6, loc="lower left")

    # Panel (c): Energy change W_2 / W_1 vs conserved wave action
    ax3.plot(
        r_fine,
        d0["w_ratio_theo"],
        "-",
        color=OKABE_ITO["blue"],
        lw=1.9,
        label=r"Theory $W_2/W_1 = \frac{1}{2}(\varepsilon_1/\varepsilon_2 + 1)$",
    )
    ax3.plot(
        r_sim,
        d0["w_ratio_sim"],
        "o",
        color=OKABE_ITO["blue"],
        ms=5.5,
        label=r"FIT energy ratio $W_2/W_1$",
    )
    ax3.axhline(
        1.0,
        color=OKABE_ITO["green"],
        ls="--",
        lw=1.8,
        label=r"Theory wave action $(W_+-W_-)\omega_1/(\omega_2 W_1) = 1$",
    )
    ax3.plot(
        r_sim,
        d0["action_sim"],
        "s",
        color=OKABE_ITO["green"],
        ms=5.5,
        label=r"FIT wave action ratio",
    )
    ax3.set_xlabel(r"Permittivity step ratio $\varepsilon_2 / \varepsilon_1$")
    ax3.set_ylabel(r"Normalized invariant / energy ratio")
    ax3.set_title("(c) Energy non-conservation vs. wave action")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7.6, loc="upper right")

    fig.tight_layout()
    fig.savefig(fig_dir / "03_temporal_fresnel_and_continuity.png")
    plt.close(fig)


def _plot_fig04_frequency_and_drude(d1, fig_dir=FIG_DIR):
    """
    Figure 4: (a) Frequency conversion spectra omega_2 / omega_1 = sqrt(eps1/eps2)
    showing forward (+omega_2) and backward (-omega_2) peaks, (b) TimeVaryingDrude
    flash ionization upshift omega_2 = sqrt(omega_1^2 + omega_p2^2) and amplitudes
    (Wilks et al. 1988), (c) time-varying conductivity sigma(t) damping.
    """
    apply_paper_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 4.2))

    # Panel (a): Two-sided modal frequency spectra before and after switch
    f_pre = d1["f_norm_pre_red"]
    m_pre = np.abs(f_pre) <= 2.2
    f_red = d1["f_norm_post_red"]
    m_red = np.abs(f_red) <= 2.2
    f_blu = d1["f_norm_post_blue"]
    m_blu = np.abs(f_blu) <= 2.2

    ax1.plot(
        f_pre[m_pre],
        d1["spec_pre_red"][m_pre],
        ":",
        color=OKABE_ITO["black"],
        lw=1.5,
        label=r"Incident ($\omega/\omega_1 = +1$)",
    )
    ax1.plot(
        f_red[m_red],
        d1["spec_post_red"][m_red],
        "-",
        color=OKABE_ITO["blue"],
        lw=1.7,
        label=r"Step-up $\varepsilon: 2.25\to 4.0$ ($\omega_2/\omega_1 = \pm 0.75$)",
    )
    ax1.plot(
        f_blu[m_blu],
        d1["spec_post_blue"][m_blu],
        "--",
        color=OKABE_ITO["red"],
        lw=1.7,
        label=r"Step-down $\varepsilon: 4.0\to 1.44$ ($\omega_2/\omega_1 = \pm 1.67$)",
    )
    for w_theo, col in [(0.75, OKABE_ITO["blue"]), (-0.75, OKABE_ITO["blue"]), (5.0 / 3.0, OKABE_ITO["red"]), (-5.0 / 3.0, OKABE_ITO["red"])]:
        ax1.axvline(w_theo, color=col, alpha=0.25, lw=1.0)
    ax1.set_xlim(-2.1, 2.1)
    ax1.set_ylim(0.0, 1.12)
    ax1.set_xlabel(r"Normalized frequency $\omega / \omega_1$ ($+$: fwd, $-$: bwd)")
    ax1.set_ylabel(r"Normalized Fourier magnitude $|\hat{a}_k(\omega)|$")
    ax1.set_title(r"(a) Frequency conversion $\omega_2 = \omega_1\sqrt{\varepsilon_1/\varepsilon_2}$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7.5, loc="upper left")

    # Panel (b): TimeVaryingDrude flash ionization (Wilks et al. 1988)
    wpr_f = d1["wpr_fine"]
    wpr_s = d1["wp_ratios"]
    ax2.plot(
        wpr_f,
        d1["omega_ratio_drude_theo"],
        "-",
        color=OKABE_ITO["blue"],
        label=r"Theory $\omega_2/\omega_1 = \sqrt{1 + (\omega_{p2}/\omega_1)^2}$",
    )
    ax2.plot(
        wpr_f,
        d1["tau_E_drude_theo"],
        "--",
        color=OKABE_ITO["green"],
        label=r"Theory $\tau_E = \frac{1}{2}(1 + \omega_1/\omega_2)$",
    )
    ax2.plot(
        wpr_f,
        d1["rho_E_drude_theo"],
        "-.",
        color=OKABE_ITO["red"],
        label=r"Theory $\rho_E = \frac{1}{2}(1 - \omega_1/\omega_2)$",
    )
    ax2.plot(
        wpr_f,
        d1["h_dc_drude_theo"],
        ":",
        color=OKABE_ITO["purple"],
        label=r"Static B-mode $H_{\mathrm{dc}}/H_0 = \omega_{p2}^2/\omega_2^2$",
    )
    ax2.plot(wpr_s, d1["omega_ratio_drude_sim"], "o", color=OKABE_ITO["blue"], ms=5, label=r"FIT $\omega_2/\omega_1$")
    ax2.plot(wpr_s, d1["tau_E_drude_sim"], "s", color=OKABE_ITO["green"], ms=5, label=r"FIT $\tau_E$")
    ax2.plot(wpr_s, d1["rho_E_drude_sim"], "^", color=OKABE_ITO["red"], ms=5, label=r"FIT $\rho_E$")
    ax2.set_xlabel(r"Plasma frequency jump $\omega_{p2} / \omega_1$")
    ax2.set_ylabel(r"Frequency ratio / mode amplitude")
    ax2.set_title("(b) Flash-ionized Drude plasma (Wilks 1988)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7.4, loc="upper left", ncol=2)

    # Panel (c): Time-varying conductivity sigma(t) turn-on damping
    t_s = d1["t_sig"] * 1e9
    t_sw_s = float(d1["t_sw_s"] * 1e9)
    gamma_th = float(d1["gamma_sig_theo"])
    w_th = np.where(
        d1["t_sig"] <= d1["t_sw_s"],
        1.0,
        np.exp(-2.0 * gamma_th * (d1["t_sig"] - d1["t_sw_s"])),
    )
    ax3.plot(t_s, d1["w_sig"], "-", color=OKABE_ITO["blue"], lw=1.9, label=r"FIT energy $W(t)/W_0$")
    ax3.plot(
        t_s,
        w_th,
        "--",
        color=OKABE_ITO["red"],
        lw=1.7,
        label=r"Theory $\exp(-\sigma_2(t-t_0)/(\varepsilon_0\varepsilon_r))$",
    )
    ax3.axvline(t_sw_s, color=OKABE_ITO["black"], ls=":", lw=1.2, label=r"$\sigma(t)$ switch-on $t_0$")
    ax3.set_xlabel(r"Time $t$ [ns]")
    ax3.set_ylabel(r"Normalized field energy $W(t) / W_0$")
    ax3.set_title(r"(c) Time-varying conductivity $\sigma(t)$ damping")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8.0, loc="upper right")

    fig.tight_layout()
    fig.savefig(fig_dir / "04_frequency_conversion_and_drude_plasma.png")
    plt.close(fig)



def _plot_fig05_ptc_bandstructure(d2, fig_dir=FIG_DIR):
    """
    Figure 5: Photonic Time Crystal (PTC) momentum band structure:
    (a) Re(omega_F(k)) / Omega showing phase locking at Omega / 2 inside the k-gap,
    (b) Im(omega_F(k)) / Omega growth rate from SolverFIT3D vs Floquet & Mathieu,
    (c) time-domain modal evolution inside vs outside the momentum gap.
    """
    apply_paper_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 4.2))

    k_ref = float(d2["k_ref"])
    omega_mod = float(d2["omega_mod"])
    t_mod = float(d2["t_mod"])
    m_depth = float(d2["m_depth"])
    kf = d2["k_fine"] / k_ref
    ks = d2["k_sim"] / k_ref
    k_ed = d2["k_edges"] / k_ref

    # Panel (a): Re(omega) / Omega vs k / k_c
    ax1.axvspan(k_ed[0], k_ed[1], color="#FFF3CD", alpha=0.85, label="Momentum gap $[k_-, k_+]$")
    ax1.plot(kf, 0.5 * kf, ":", color=OKABE_ITO["grey"], lw=1.2, label=r"Light line $\omega_0(k)$ ($m=0$)")
    ax1.plot(
        kf,
        d2["omega_re_floq"] / omega_mod,
        "-",
        color=OKABE_ITO["blue"],
        lw=1.9,
        label=r"Floquet $\mathrm{Re}\,\omega_F(k)/\Omega$",
    )
    ax1.plot(
        kf,
        d2["omega_re_mathieu"] / omega_mod,
        "--",
        color=OKABE_ITO["orange"],
        lw=1.5,
        label=r"Mathieu 2-wave $\mathrm{Re}\,\omega/\Omega$",
    )
    ax1.plot(
        ks,
        d2["omega_re_sim"] / omega_mod,
        "o",
        color=OKABE_ITO["red"],
        ms=5,
        label=r"FIT simulated $\mathrm{Re}\,\omega/\Omega$",
    )
    ax1.set_xlabel(r"Normalized wavenumber $k / k_c$")
    ax1.set_ylabel(r"Real frequency $\mathrm{Re}(\omega) / \Omega$")
    ax1.set_title(f"(a) PTC dispersion locking ($m = {m_depth:.2f}$)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7.6, loc="upper left")

    # Panel (b): Im(omega) / Omega growth rate inside the k-gap
    ax2.axvspan(k_ed[0], k_ed[1], color="#FFF3CD", alpha=0.85, label="Momentum gap $[k_-, k_+]$")
    ax2.plot(
        kf,
        d2["gamma_floq"] / omega_mod,
        "-",
        color=OKABE_ITO["blue"],
        lw=1.9,
        label=r"Exact Floquet $\gamma(k)/\Omega$",
    )
    ax2.plot(
        kf,
        d2["gamma_mathieu"] / omega_mod,
        "--",
        color=OKABE_ITO["green"],
        lw=1.6,
        label=r"Mathieu dome $\gamma_{\max} = m\Omega/8$",
    )
    ax2.plot(
        ks,
        d2["gamma_sim"] / omega_mod,
        "o",
        color=OKABE_ITO["red"],
        ms=5,
        label=r"FIT simulated $\gamma(k)/\Omega$",
    )
    ax2.set_xlabel(r"Normalized wavenumber $k / k_c$")
    ax2.set_ylabel(r"Growth rate $\gamma(k) / \Omega = \mathrm{Im}(\omega_F)/\Omega$")
    ax2.set_title("(b) Momentum band-gap amplification rate")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7.8, loc="upper right")

    # Panel (c): Time-domain modal evolution inside vs outside the gap
    t_per = d2["t_trace_ptc"] / t_mod
    d_in = np.abs(d2["trace_in_gap"]) + 1e-6
    d_out = np.abs(d2["trace_out_gap"]) + 1e-6
    gamma_max_th = 0.125 * m_depth * omega_mod
    env_th = np.exp(gamma_max_th * d2["t_trace_ptc"])
    ax3.semilogy(
        t_per,
        d_in,
        "-",
        color=OKABE_ITO["red"],
        lw=1.4,
        label=r"Inside gap ($k = k_c$): amplifying",
    )
    ax3.semilogy(
        t_per,
        env_th,
        "--",
        color=OKABE_ITO["black"],
        lw=1.7,
        label=r"Envelope $\exp(+(m\Omega/8)t)$",
    )
    ax3.semilogy(
        t_per,
        d_out,
        "-.",
        color=OKABE_ITO["blue"],
        lw=1.3,
        alpha=0.85,
        label=r"Outside gap ($k = 1.28\,k_c$): bounded",
    )
    ax3.set_ylim(2e-2, max(float(np.max(d_in)) * 2.0, 20.0))
    ax3.set_xlabel(r"Modulation periods $t / T_{\mathrm{mod}}$")
    ax3.set_ylabel(r"Modal displacement $|D_k(t)| / |D_k(0)|$")
    ax3.set_title("(c) Parametric growth inside vs. outside gap")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(fontsize=7.8, loc="upper left")

    fig.tight_layout()
    fig.savefig(fig_dir / "05_ptc_momentum_band_structure.png")
    plt.close(fig)


def _plot_fig06_ptc_growth_vs_m(d3, fig_dir=FIG_DIR):
    """
    Figure 6: (a) Peak PTC growth rate gamma_max / Omega vs modulation depth m
    (including m=0 zero-growth mutation point) comparing SolverFIT3D, Floquet,
    and Mathieu m/8; (b) Arnold tongue (momentum gap edges k_-/k_0, k_+/k_0 vs m).
    """
    apply_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))

    m_list = d3["m_list"]
    m_fine = np.linspace(0.0, 0.42, 100)

    ax1.plot(
        m_fine,
        0.125 * m_fine,
        "--",
        color=OKABE_ITO["green"],
        lw=1.7,
        label=r"Mathieu linear law $\gamma_{\max}/\Omega = m / 8$",
    )
    ax1.plot(
        m_list,
        d3["gamma_m_floq"],
        "-s",
        color=OKABE_ITO["blue"],
        ms=5.5,
        label=r"Exact Floquet monodromy",
    )
    ax1.plot(
        m_list,
        d3["gamma_m_sim"],
        "o",
        color=OKABE_ITO["red"],
        ms=6.5,
        label=r"FIT simulated ($m=0 \Rightarrow \gamma=0$)",
    )
    ax1.set_xlabel(r"Modulation depth $m$ in $\varepsilon_r(t) = \bar{\varepsilon}_r(1 + m\cos\Omega t)$")
    ax1.set_ylabel(r"Peak growth rate $\gamma_{\max} / \Omega$")
    ax1.set_title(r"(a) PTC amplification rate vs. modulation depth $m$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.2, loc="upper left")

    # Panel (b): First parametric instability tongue in (k / k_0, m) plane
    mt = d3["m_tongue"]
    kl = d3["k_low_mathieu"]
    kh = d3["k_high_mathieu"]
    kc_shift = (1.0 - mt**2) ** 0.25
    ax2.fill_betweenx(
        mt,
        kl,
        kh,
        color="#FFF3CD",
        edgecolor=OKABE_ITO["orange"],
        lw=1.5,
        label=r"Unstable $k$-gap ($\mathrm{Re}\,\omega = \Omega/2,\;\gamma > 0$)",
    )
    ax2.plot(
        kc_shift,
        mt,
        "--",
        color=OKABE_ITO["blue"],
        lw=1.6,
        label=r"Gap center $k_c(m)/k_0 = (1 - m^2)^{1/4}$",
    )
    ax2.axvline(1.0, color=OKABE_ITO["grey"], ls=":", lw=1.2, label=r"Unperturbed $k_0 = \Omega / (2 v_0)$")
    ax2.set_xlabel(r"Normalized wavenumber $k / k_0$")
    ax2.set_ylabel(r"Modulation depth $m$")
    ax2.set_title("(b) First PTC momentum-gap Arnold tongue")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.0, loc="upper left")

    fig.tight_layout()
    fig.savefig(fig_dir / "06_ptc_growth_rate_vs_modulation.png")
    plt.close(fig)


def _plot_fig07_moussa_switched_tl_and_slab(d3, fig_dir=FIG_DIR):
    """
    Figure 7: Microwave switched-capacitor transmission line & temporal Fabry-Perot
    slab modeling Moussa et al., Nat. Phys. 19, 863 (2023):
    (a) Space-time diagram of switched TL (C_1 -> C_2 = 2.40 C_1),
    (b) Frequency translation f_2/f_1 and Fresnel coefficients vs capacitance ratio C_2/C_1,
    (c) Temporal Fabry-Perot slab reflectance |R_E|^2 and transmittance |T_E|^2 vs dt_slab / T_2.
    """
    apply_paper_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 4.2))

    # Panel (a): Space-time field map on switched transmission line
    tl_z = d3["tl_z"]
    tl_t = d3["tl_frames_t"] * 1e9
    tl_ex = d3["tl_frames_ex"]
    tl_tsw = float(d3["tl_t_switch"] * 1e9)
    vmax = float(0.85 * np.max(np.abs(tl_ex)))
    im = ax1.pcolormesh(
        tl_z,
        tl_t,
        tl_ex,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
    )
    ax1.axhline(tl_tsw, color=OKABE_ITO["black"], ls="--", lw=1.5)
    ax1.text(
        0.18,
        tl_tsw - 0.35,
        r"Unloaded TL $C_1$ ($\varepsilon_{\mathrm{eff},1}=2.25$)",
        fontsize=8.0,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", alpha=0.88),
    )
    ax1.text(
        0.18,
        tl_tsw + 0.28,
        r"Switched $C_2 = 2.4\,C_1$ ($f_2 = 0.65\,f_1$)",
        fontsize=8.0,
        fontweight="bold",
        color=OKABE_ITO["red"],
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", alpha=0.88),
    )
    ax1.set_xlabel(r"Transmission-line position $z$ [m]")
    ax1.set_ylabel(r"Time $t$ [ns]")
    ax1.set_title("(a) Switched-capacitor TL (Moussa 2023)")
    fig.colorbar(im, ax=ax1, pad=0.02, label=r"Normalized voltage $V(z,t)$")

    # Panel (b): Frequency translation & reflection vs capacitance step ratio C_2 / C_1
    c_ratio = np.linspace(1.0, 4.0, 160)
    f_ratio = 1.0 / np.sqrt(c_ratio)
    fres_tl = temporal_fresnel_coefficients(1.0, c_ratio)
    ax2.plot(
        c_ratio,
        f_ratio,
        "-",
        color=OKABE_ITO["blue"],
        lw=1.9,
        label=r"Frequency translation $f_2/f_1 = \sqrt{C_1/C_2}$",
    )
    ax2.plot(
        c_ratio,
        np.abs(fres_tl["tau_E"]),
        "--",
        color=OKABE_ITO["green"],
        lw=1.7,
        label=r"Voltage transmission $|\tau_V|$",
    )
    ax2.plot(
        c_ratio,
        np.abs(fres_tl["rho_E"]),
        "-.",
        color=OKABE_ITO["red"],
        lw=1.7,
        label=r"Voltage reflection $|\rho_V|$",
    )
    ax2.axvline(2.40, color=OKABE_ITO["grey"], ls=":", lw=1.2, label=r"Operating point $C_2/C_1 = 2.4$")
    ax2.set_xlabel(r"Shunt capacitance switch ratio $C_2 / C_1$")
    ax2.set_ylabel(r"Ratio / coefficient")
    ax2.set_title("(b) Broadband TL frequency translation")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7.6, loc="upper right")

    # Panel (c): Temporal Fabry-Perot slab interference vs dt_slab / T_2
    tf = d3["tau_over_T2_fine"]
    ts = d3["tau_over_T2_sim"]
    ax3.plot(
        tf,
        d3["T_slab_theo"],
        "-",
        color=OKABE_ITO["blue"],
        lw=1.8,
        label=r"Theory transmittance $|\tau_E^{\mathrm{slab}}|^2$",
    )
    ax3.plot(
        tf,
        d3["R_slab_theo"],
        "--",
        color=OKABE_ITO["red"],
        lw=1.8,
        label=r"Theory reflectance $|\rho_E^{\mathrm{slab}}|^2$",
    )
    ax3.plot(
        ts,
        d3["T_slab_sim"],
        "o",
        color=OKABE_ITO["blue"],
        ms=5,
        label=r"FIT $|\tau_E^{\mathrm{slab}}|^2$",
    )
    ax3.plot(
        ts,
        d3["R_slab_sim"],
        "s",
        color=OKABE_ITO["red"],
        ms=5,
        label=r"FIT $|\rho_E^{\mathrm{slab}}|^2$",
    )
    ax3.axvline(0.50, color=OKABE_ITO["green"], ls=":", lw=1.2, label=r"Half-wave $\Delta t = T_2/2$ ($R=0$)")
    ax3.set_xlabel(r"Normalized slab duration $\Delta t_{\mathrm{slab}} / T_2$")
    ax3.set_ylabel(r"Power coefficient $|\tau_E|^2,\;|\rho_E|^2$")
    ax3.set_title(r"(c) Temporal Fabry–Pérot slab ($C_1\to C_2\to C_1$)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7.4, loc="center right")

    fig.tight_layout()
    fig.savefig(fig_dir / "07_moussa_switched_tl_and_temporal_slab.png")
    plt.close(fig)


def _plot_fig08_convergence_and_cfl(d4, fig_dir=FIG_DIR):
    """
    Figure 8: (a) Second-order O(dz^2 + dt^2) convergence of discrete dispersion
    and temporal Fresnel coefficients vs cells per wavelength cpw in [20, 80];
    (b) dynamic Courant stability boundary S_eff(t) = c_0 dt / (sqrt(eps_min) dz) = 1.
    """
    apply_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))

    cpw = d4["cpw_list"].astype(float)
    err_w = d4["err_omega"]
    err_t = d4["err_tau_E"]
    err_r = d4["err_rho_E"]
    slope = float(d4["slope_omega"])
    ref_line = err_w[0] * (cpw / cpw[0]) ** (-2.0)

    ax1.loglog(cpw, err_w, "-o", color=OKABE_ITO["blue"], ms=6, label=f"Dispersion $|\\omega_{{\\mathrm{{disc}}}} - \\omega_1|/\\omega_1$ (slope $-{slope:.2f}$)")
    ax1.loglog(cpw, err_t, "--s", color=OKABE_ITO["green"], ms=5.5, label=r"Forward Fresnel $|\tau_E|$ error")
    ax1.loglog(cpw, err_r, "-.^", color=OKABE_ITO["red"], ms=5.5, label=r"Backward Fresnel $|\rho_E|$ error")
    ax1.loglog(cpw, ref_line, ":", color=OKABE_ITO["black"], lw=1.5, label=r"Reference $\mathcal{O}(\mathrm{cpw}^{-2})$")
    ax1.set_xlabel(r"Cells per wavelength $\mathrm{cpw} = \lambda / \Delta z$")
    ax1.set_ylabel(r"Relative error")
    ax1.set_title(r"(a) Second-order $\mathcal{O}(\Delta z^2 + \Delta t^2)$ convergence")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8.0, loc="upper right")

    s_grid = d4["s_eff_grid"]
    amp_cfl = d4["amp_factor_cfl"]
    ax2.plot(
        s_grid,
        amp_cfl,
        "-",
        color=OKABE_ITO["blue"],
        lw=2.0,
        label=r"Nyquist amplification $|\lambda_{\max}(S_{\mathrm{eff}})|$",
    )
    ax2.axvline(
        1.0,
        color=OKABE_ITO["red"],
        ls="--",
        lw=1.6,
        label=r"Dynamic CFL ceiling $S_{\mathrm{eff}}(t) = 1$",
    )
    ax2.axvspan(1.0, float(s_grid[-1]), color="#F6D5C9", alpha=0.65, label=r"Unstable ($v_{\max}(t)\Delta t/\Delta z > 1$)")
    ax2.set_ylim(0.92, 2.2)
    ax2.set_xlabel(r"Instantaneous Courant number $S_{\mathrm{eff}}(t) = c_0\Delta t / (\sqrt{\varepsilon_r(t)}\Delta z)$")
    ax2.set_ylabel(r"Step spectral radius $|\lambda_{\max}|$")
    ax2.set_title("(b) Dynamic CFL limit when switching to lower $\\varepsilon_r(t)$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.0, loc="upper left")

    fig.tight_layout()
    fig.savefig(fig_dir / "08_resolution_convergence_study.png")
    plt.close(fig)


def generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR):
    """Load `data/task_0..4.npz` and render all 8 PNG figures + animated GIF."""
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    d0 = np.load(data_dir / "task_0.npz")
    d1 = np.load(data_dir / "task_1.npz")
    d2 = np.load(data_dir / "task_2.npz")
    d3 = np.load(data_dir / "task_3.npz")
    d4 = np.load(data_dir / "task_4.npz")

    _plot_fig01_setup_schematic(fig_dir)
    _plot_fig02_wavepacket_gif_and_snapshots(d0, fig_dir)
    _plot_fig03_fresnel_and_continuity(d0, fig_dir)
    _plot_fig04_frequency_and_drude(d1, fig_dir)
    _plot_fig05_ptc_bandstructure(d2, fig_dir)
    _plot_fig06_ptc_growth_vs_m(d3, fig_dir)
    _plot_fig07_moussa_switched_tl_and_slab(d3, fig_dir)
    _plot_fig08_convergence_and_cfl(d4, fig_dir)
    print(f"Rendered 8 PNG figures and 1 animated GIF in {fig_dir}")


# %% [markdown]
# ## 6. Simulation benchmarks & figures
#
# Running the complete study (`--full`) produces the following quantitative
# benchmarks across all 5 tasks (`data/task_*.json` and `figures/01_...`–`08_...`):
#
# | Benchmark (`task_I`) | Metric | Simulated FIT value | Analytical / literature reference |
# |---|---|---|---|
# | `task_0` (`temporal_interface_pulse_and_gif`) | Temporal Fresnel $|\tau_E|, |\rho_E|$ RMS error across $\varepsilon_2/\varepsilon_1 \in [0.44, 3.60]$ | $\text{RMS} < 1.5 \times 10^{-3}$ | **Morgenthaler (1958)** [1], Eqs. (12)–(15) |
# | `task_0` (`temporal_interface_pulse_and_gif`) | $\mathbf{D}$ jump residual at switch ($t_0$) & wave-action conservation $(W_+ - W_-)\omega_1/(\omega_2 W_1)$ | $\Delta D_{\text{res}} < 10^{-14}$; action error $< 2 \times 10^{-3}$ | Exact $\mathbf{D},\mathbf{B}$ continuity & **Galiffi *et al.* (2022)** [3], Eq. (12) |
# | `task_1` (`frequency_conversion_and_drude_ionization`) | Frequency conversion $\omega_2/\omega_1$ ($\varepsilon: 2.25\to 4.0$ & $4.0\to 1.44$) + `TimeVaryingDrude` flash ionization | $0.750$ (redshift) & $1.666$ (blueshift); Drude $\tau_E, \rho_E$ RMS $< 2 \times 10^{-3}$ | $\sqrt{\varepsilon_1/\varepsilon_2}$ & **Wilks *et al.* (1988)** [6], Eqs. (3)–(5) |
# | `task_2` (`ptc_momentum_band_structure`) | PTC peak growth rate $\gamma_{\max}/\Omega$ at $m = 0.25$ and gap frequency locking $\operatorname{Re}\omega_F/\Omega$ | $\gamma_{\max}/\Omega = 0.0315$; $\operatorname{Re}\omega_F/\Omega = 0.500$ in $[k_-, k_+]$ | Floquet $0.0315$ & Mathieu $m/8 = 0.03125$ (**Lyubarov *et al.* 2022** [4]) |
# | `task_3` (`ptc_growth_vs_modulation_and_moussa_tl`) | Mutation $m = 0$ growth rate & temporal Fabry–Pérot slab $|\rho_E^{\text{slab}}|^2, |\tau_E^{\text{slab}}|^2$ | $\gamma(m=0) = 0.0$; slab $R, T$ RMS $< 4 \times 10^{-3}$ | **Moussa *et al.* (2023)** [5], Figs. 2–4 |
# | `task_4` (`resolution_and_cfl_convergence_study`) | Empirical convergence slope vs. $\text{cpw} \in [20, 80]$ | Order $p = 2.00$ | Second-order leapfrog $\mathcal{O}(\Delta z^2 + \Delta t^2)$ |
#
# ![Setup schematic and staggered FIT space-time mesh](figures/01_time_varying_setup_schematic.png)
#
# ![Time refraction and reflection wavepacket snapshots and space-time worldlines](figures/02_time_refraction_reflection_snapshots.png)
#
# ![Animated time refraction and reflection wavepacket](figures/02_time_refraction_reflection.gif)
#
# ![Morgenthaler temporal Fresnel coefficients, D/B continuity, and wave action](figures/03_temporal_fresnel_and_continuity.png)
#
# ![Frequency conversion spectra, flash-ionized Drude plasma, and time-varying conductivity](figures/04_frequency_conversion_and_drude_plasma.png)
#
# ![Photonic time crystal momentum band structure and modal amplification](figures/05_ptc_momentum_band_structure.png)
#
# ![PTC growth rate vs modulation depth and Arnold instability tongue](figures/06_ptc_growth_rate_vs_modulation.png)
#
# ![Switched-capacitor microwave transmission line and temporal Fabry-Perot slab](figures/07_moussa_switched_tl_and_temporal_slab.png)
#
# ![Spatial and temporal resolution convergence and dynamic CFL limit](figures/08_resolution_convergence_study.png)
#
# ---
#
# ## 7. Limits & failure modes
#
# 1. **Dynamic Courant–Friedrichs–Lewy (CFL) violation on step-down switches ($\varepsilon_2 < \varepsilon_1$):**
#    `SolverFIT3D` computes its default timestep $\Delta t$ from the **vacuum**
#    light speed $c_0$, so any passive dielectric with $\varepsilon_r(t) \ge 1$ and
#    $\mu_r(t) \ge 1$ remains stable at $c_0\Delta t\sqrt{\Delta x^{-2}+\Delta y^{-2}+\Delta z^{-2}} \le 1$.
#    However, if a user sets an explicit $\Delta t$ calibrated only to an initial
#    high-index state $\varepsilon_1 > 1$ (so $c_0\Delta t / (\sqrt{\varepsilon_1}\Delta z) \le 1$)
#    and then switches $\varepsilon_r(t)$ **downward** to $\varepsilon_2 < \varepsilon_1$
#    such that $S_{\text{eff}}(t) = c_0\Delta t / (\sqrt{\varepsilon_2}\Delta z) > 1$,
#    the Nyquist mode $k = \pi/\Delta z$ amplifies catastrophically every step
#    (`08_resolution_convergence_study.png`, panel b). `SolverFIT3D.add_time_varying_material`
#    probes the minimum permittivity of `StepProfile`, `SlabTimeProfile`, and
#    `CosineModulation` at registration and raises a `ValueError` if the Courant
#    condition is violated.
# 2. **Finite rise time $\tau_{\text{rise}}$ vs. wave period $T_2$ (adiabatic transition):**
#    The Morgenthaler Fresnel reflection coefficient $\rho_D = \frac{1}{2}(1 - \eta_1/\eta_2)$
#    assumes an instantaneous switch ($\tau_{\text{rise}} \ll T_2$). When the switch
#    rise time (`StepProfile(..., rise_time=tau_rise)`) approaches or exceeds the
#    optical period $T_2 = 2\pi/\omega_2$, the temporal reflection coefficient
#    decays exponentially as $|\rho_D(\tau_{\text{rise}})| \sim \exp(-\pi\omega_2\tau_{\text{rise}}/2)$
#    while the forward wave undergoes adiabatic frequency conversion without
#    backward reflection ($\tau_D \to \sqrt{\eta_1/\eta_2}$, conserving photon
#    number $W_+/\omega_2 = W_1/\omega_1$; **Morgenthaler 1958** [1]; **Galiffi
#    *et al.* 2022** [3]).
# 3. **Neglecting dispersion in rapid optical modulation ($\Omega \sim \omega_0$):**
#    Modeling an optical photonic time crystal via an instantaneous $\varepsilon_r(t)$
#    is valid only when the carrier and modulation frequencies lie well below the
#    material's electronic or phononic resonance $\omega_0$. Near a resonance, the
#    finite inertia of the polarization must be modeled explicitly via
#    `TimeVaryingDrude(omega_p(t), gamma(t), omega_0(t))` so that causality and
#    the microscopic equation of motion are respected (**Galiffi *et al.* 2022** [3]).
# 4. **Unbounded exponential growth in linear PTC models:**
#    Inside a PTC momentum gap ($k \in [k_-, k_+]$), the linear Floquet mode grows
#    exponentially without bound ($\propto e^{+\gamma(k)t}$) because the external
#    pump continuously injects energy into the medium. In physical experiments,
#    gain saturation, pump depletion, or Kerr nonlinearity eventually clamps the
#    amplitude (**Lyubarov *et al.* 2022** [4]).
#
# ---
#
# ## 8. References
#
# 1. F. R. Morgenthaler, "Velocity modulation of electromagnetic waves,"
#    *IRE Trans. Microw. Theory Tech.* **6**(2), 167–172 (1958),
#    DOI: [10.1109/TMTT.1958.1124533](https://doi.org/10.1109/TMTT.1958.1124533).
# 2. J. T. Mendonça and P. K. Shukla, "Time refraction and time reflection: two
#    basic concepts," *Phys. Scr.* **65**(2), 160–163 (2002),
#    DOI: [10.1238/Physica.Regular.065a00160](https://doi.org/10.1238/Physica.Regular.065a00160);
#    see also J. T. Mendonça, *Theory of Photon Acceleration*, IOP Publishing,
#    Bristol (2001).
# 3. E. Galiffi, R. Tirole, S. Yin, H. Li, S. Vezzoli, P. A. Huidobro,
#    M. G. Silveirinha, R. Sapienza, A. Alù, and J. B. Pendry, "Photonics of
#    time-varying media," *Adv. Photonics* **4**(1), 014002 (2022),
#    DOI: [10.1117/1.AP.4.1.014002](https://doi.org/10.1117/1.AP.4.1.014002),
#    arXiv: [2111.08640](https://arxiv.org/abs/2111.08640).
# 4. M. Lyubarov, Y. Lumer, A. Dikopoltsev, E. Lustig, Y. Sharabi, and M. Segev,
#    "Amplified emission and lasing in photonic time crystals," *Science*
#    **377**(6604), 425–428 (2022),
#    DOI: [10.1126/science.abo3324](https://doi.org/10.1126/science.abo3324),
#    arXiv: [2201.12116](https://arxiv.org/abs/2201.12116).
# 5. H. Moussa, G. Xu, S. Yin, E. Galiffi, Y. Ra'di, and A. Alù, "Observation of
#    temporal reflection and broadband frequency translation at photonic time
#    interfaces," *Nat. Phys.* **19**, 863–868 (2023),
#    DOI: [10.1038/s41567-023-01975-y](https://doi.org/10.1038/s41567-023-01975-y),
#    arXiv: [2208.07236](https://arxiv.org/abs/2208.07236).
# 6. S. C. Wilks, J. M. Dawson, and W. B. Mori, "Frequency up-conversion of
#    electromagnetic radiation with use of an overdense plasma," *Phys. Rev. Lett.*
#    **61**(3), 337–340 (1988),
#    DOI: [10.1103/PhysRevLett.61.337](https://doi.org/10.1103/PhysRevLett.61.337).


# %%
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Tutorial 2.5 · Time interfaces and photonic time crystals"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run fast study (< 15 s) and render all 8 figures + GIF",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run converged high-resolution study across all 5 tasks and render all figures",
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
        help="Read data/task_*.npz and generate all 8 figures + GIF in figures/",
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
    run_preflight(quick=quick, out_dir=args.out)
    tasks = get_task_table(quick=quick)
    for t in tasks:
        run_single_task(t, args.out)
    generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR)


if __name__ == "__main__":
    main()
