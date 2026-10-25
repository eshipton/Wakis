# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial 2.3 · Magnetic Materials
#
# **Chapter 2 · Materials**
#
# [Tutorial 2.1](../02-01_dispersion_physics/dispersion_physics.ipynb) and
# [Tutorial 2.2](../02-02_fitting_measured_data/fitting_measured_data.ipynb)
# developed time-domain Auxiliary Differential Equations (ADEs) and pole fitting
# for the electric permittivity $\varepsilon(\omega)$. In ferrites,
# planar hexaferrites, and ferromagnetic thin-film laminates used across VHF/UHF
# antennas, microwave absorbers, and RF cavities, the magnetic response
# $\mathbf{B}(\omega) = \mu_0 \mu(\omega)\mathbf{H}(\omega)$ is equally
# dispersive and lossy.
#
# This tutorial derives the frequency-dependent permeability $\mu(\omega)$ from
# the microscopic **Landau–Lifshitz–Gilbert (LLG)** equation of spin precession
# and domain-wall relaxation, explains why **Snoek's law** (1948) [1] and
# **Acher's planar-anisotropy bound** (2008) [2] link static permeability
# $\mu_s$ to the ferromagnetic resonance frequency $f_r$, and shows why the
# rising magnetic loss tangent $\tan\delta_m(f)$ forces resonant substrate
# antennas to operate well below $f_r$—while non-resonant flux-channel radiators
# are governed instead by the **hesitivity** $h_m = \max_\omega \sigma_m(\omega)$
# [3, 4]. Using published material parameters from the
# literature, we then validate Wakis's magnetic ADE update on dual $\mathbf{H}$
# edges across four 3D `SolverFIT3D` benchmarks: a **Lorentz-$\mu$ ferrite
# slab**, a **Debye-$\mu$ relaxation slab**, a **reflectionless
# $\varepsilon(\omega) = \mu(\omega)$ impedance-matched absorber slab**, and a
# **3D PEC cavity** whose $\text{TE}_{101}$ mode shifts and splits into two
# hybridized cavity–magnon polariton branches.
#
# > **Time-harmonic sign convention (`exp(+i*omega*t)`):** Throughout Wakis,
# > monochromatic fields vary as $\mathbf{H}(\mathbf{r}, t) = \operatorname{Re}[\hat{\mathbf{H}}(\mathbf{r})\,e^{+i\omega t}]$.
# > Writing $\mu(\omega) = \mu'(\omega) - i\,\mu''(\omega)$, **passive
# > (non-amplifying) magnetic media satisfy $\operatorname{Im}\mu(\omega) = -\mu''(\omega) \le 0$
# > for all $\omega > 0$**.

# %%
# --- Colab setup (runs only on Google Colab) ---
import sys

if "google.colab" in sys.modules:
    import subprocess

    print("Running on Google Colab — installing dependencies...")
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(
        ["apt-get", "install", "-y", "-qq", "xvfb", "libgl1-mesa-glx"],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "git+https://github.com/eshipton/Wakis.git@main",
            "pyvista",
            "scikit-rf",
            "tqdm",
        ],
        check=True,
    )
    print("Colab setup complete.")

# %% [markdown]
# ## Learning Goals
#
# By the end of this tutorial you will be able to:
#
# 1. **Derive Lorentz and Debye permeability models from magnetization dynamics:**
#    linearize the Landau–Lifshitz–Gilbert (LLG) equation around the internal
#    magnetocrystalline/shape anisotropy field $\mathbf{H}_a$ of an unbiased
#    ferrite to obtain the **Lorentz** natural ferromagnetic resonance (FMR)
#    model, and show how overdamped domain-wall displacement yields the
#    first-order **Debye** magnetic relaxation model.
# 2. **Apply Snoek's law and Acher's thin-film generalization:** compute
#    Snoek's product $(\mu_s - 1)\,f_r = \frac{2}{3}(\gamma_{\text{LL}}/2\pi)\mu_0 M_s$
#    for isotropic cubic ferrites (**Snoek, 1948** [1]) and explain how planar
#    anisotropy ($H_\theta \gg H_\phi$) in $\text{Co}_2\text{Z}$ hexaferrites
#    and $\text{CoZrNb}$ ferromagnetic laminates boosts the resonance to
#    $f_r \propto \sqrt{H_\phi H_\theta}$, governed by **Acher's integral sum
#    rule** $\int_0^\infty f\,\mu''(f)\,df = \frac{\pi}{2}\Delta\mu\,f_r^2$
#    (**Acher & Dubourg, 2008** [2]; **Ikonen *et al.*, 2006** [3]).
# 3. **Contrast magnetic loss tangent $\tan\delta_m(f)$ with hesitivity $h_m$:**
#    show why $\tan\delta_m(f) = \mu''(f)/\mu'(f) \propto f/f_r$ limits resonant
#    patch antennas on magnetic substrates to $f \ll f_r$ (**Ikonen *et al.*,
#    2006** [3]), whereas the loss function $-\operatorname{Im}[1/(\mu(\omega)-1)] = \omega\mu_0 / h_m$
#    of an electrically small flux-channel radiator depends only on the
#    **hesitivity** $h_m = \max_\omega \sigma_m(\omega)$ (**Yousefi, Sebastian
#    & Diaz, 2016** [4]).
# 4. **Simulate dispersive magnetic and magneto-dielectric slabs in 3D FIT:**
#    validate `DispersiveMaterial(mu_inf=..., mu_poles=..., sigma_m=...)`
#    against the exact transfer matrix evaluated at `material.mu_discrete(f, dt)`
#    for Lorentz-$\mu$ and Debye-$\mu$ slabs, and demonstrate a reflectionless
#    $\varepsilon(\omega) = \mu(\omega)$ impedance-matched absorber ($\eta(\omega) = \eta_0$,
#    $R(f) \to 0$).
# 5. **Demonstrate cavity mode shifting and polariton splitting with $\mu(\omega)$:**
#    simulate a 3D PEC cavity loaded with a Lorentz ferrite as $f_r$ is swept
#    across the bare $\text{TE}_{101}$ cavity mode, comparing the simulated upper
#    and lower hybridized eigenfrequencies against the exact roots of
#    $\omega^2\mu(\omega) = \omega_{\text{vac}}^2$.
#
# ### Simulation setups & geometry overview
#
# All five tasks in this tutorial use three `SolverFIT3D` geometries shown in the
# setup schematic below:
# - **Panel (a) — Periodic $\mathbf{H}$-edge magnetic ADE testbench (`Task 0`):**
#   a doubly-periodic $2 \times 2 \times 16$ cell `GridFIT3D` ($\Delta z = c_0\Delta t / 0.25$,
#   `bc_low=["periodic"]*3`, `bc_high=["periodic"]*3`) uniformly filled with a
#   magnetic `DispersiveMaterial(mu_inf=..., mu_poles=[...])`. A localized dipole
#   pair $J_x(k) = -J_x(k+1)$ drives $(\nabla\times\mathbf{E})_y = \partial_z E_x \ne 0$
#   so that dual $H_y$ edges advance through `_update_dispersive_H`, allowing the
#   extracted ratio $\mu_{\text{FIT}}(f) = -(\nabla\times\mathbf{E})_y / (i\tilde{W}(f)\mu_0 H_y)$
#   to be compared directly against `material.mu_discrete(f, dt)`.
# - **Panel (b) — Quasi-1D magnetic & impedance-matched ($\varepsilon = \mu$) slab R/T setup (`Tasks 1–3`):**
#   a domain of length $L_z = 560\text{–}600\;\text{mm}$ terminated on $\pm z$ by
#   15-cell CPML absorbers and periodic on $\pm x, \pm y$. A broadband
#   $x$-polarized `CurrentSource` sheet at $z_{\text{src}} = -160\;\text{mm}$
#   launches a plane-wave pulse toward a ferrite or magneto-dielectric slab on
#   $z \in [0, d]$ ($d = 40\text{–}50\;\text{mm}$, resolved by $\ge 20\;\text{cells}/\lambda_{\min}$
#   inside the highest-index medium and $32\text{–}50$ cells across the slab),
#   while reflection and transmission `FluxMonitor` planes at $z_r = -80\;\text{mm}$
#   and $z_t = d + 60\;\text{mm}$ record the complex Poynting spectra.
# - **Panel (c) — 3D PEC ferrite-loaded rectangular cavity (`Task 4`):**
#   a closed metallic box of dimensions $L_x \times L_y \times L_z = 120 \times 60 \times 120\;\text{mm}^3$
#   (`bc_low=["pec"]*3`, `bc_high=["pec"]*3`) filled with a Lorentz-$\mu$ ferrite
#   ($\mu_\infty = 1.0, \Delta\mu = 1.25, \mu_s = 2.25$). Exciting the spatial
#   $\text{TE}_{101}$ profile $\sin(\pi x/L_x)\sin(\pi z/L_z)$ on primary $E_y$
#   edges and Fourier-transforming the ring-down trace reveals how the bare
#   $\text{TE}_{101}$ mode ($f_{\text{vac}} = 1.7665\;\text{GHz}$) splits into
#   lower and upper hybridized cavity–magnon polariton branches $f_-(f_r)$ and
#   $f_+(f_r)$ on either side of the negative-$\mu'$ stopband.
#
# ![Simulation setups for Tutorial 2.3: periodic H-edge magnetic ADE testbench, quasi-1D magnetic/matched slab R/T setup, and 3D PEC ferrite-loaded rectangular cavity](figures/00_setup_schematic.png)
#
# ---
#
# ## 1. Physics of Dispersive Permeability $\mu(\omega)$
#
# ### 1.1 Magnetization dynamics: the Landau–Lifshitz–Gilbert (LLG) equation
#
# In a ferromagnet or ferrimagnet with local magnetization $\mathbf{M}(\mathbf{r}, t)$
# ($|\mathbf{M}| = M_s$, the saturation magnetization), an effective magnetic
# field $\mathbf{H}_{\text{eff}}$ exerts a Zeeman torque $\mu_0 \mathbf{M}\times\mathbf{H}_{\text{eff}}$
# that causes magnetic moments to precess around $\mathbf{H}_{\text{eff}}$ at
# the Larmor rate $\gamma_{\text{LL}}\mu_0 |\mathbf{H}_{\text{eff}}|$, while spin-orbit
# and magnon-phonon scattering relax $\mathbf{M}$ toward $\mathbf{H}_{\text{eff}}$.
# This is described by the **Landau–Lifshitz–Gilbert (LLG) equation**:
#
# $$\frac{d\mathbf{M}}{dt}
#   = -\gamma_{\text{LL}}\,\mu_0\,\mathbf{M}\times\mathbf{H}_{\text{eff}}
#     + \frac{\alpha_{\text{G}}}{M_s}\,\mathbf{M}\times\frac{d\mathbf{M}}{dt},$$
#
# where $\gamma_{\text{LL}} = g |e| / (2 m_e) \approx 2\pi \times 28.0\;\text{GHz/T}$
# (so $\gamma_{\text{LL}}\mu_0 / (2\pi) \approx 35.19\;\text{kHz}/(\text{A/m}) = 2.80\;\text{MHz/Oe}$)
# is the electron gyromagnetic ratio and $\alpha_{\text{G}} > 0$ is the
# dimensionless Gilbert damping constant.
#
# ### 1.2 Natural ferromagnetic resonance (FMR) of an unbiased ferrite
#
# Even when no external DC bias coil is present ($\mathbf{H}_{\text{dc}} = \mathbf{0}$),
# each magnetic domain in a polycrystalline ferrite or thin film is aligned
# along a local easy axis $\hat{\mathbf{u}}$ by an internal **magnetocrystalline
# and shape-demagnetizing anisotropy field** $\mathbf{H}_a = H_a \hat{\mathbf{u}}$.
# Under a small time-harmonic RF field $\mathbf{h}\,e^{+i\omega t} \perp \hat{\mathbf{u}}$
# ($|\mathbf{h}| \ll H_a$), writing $\mathbf{M} = M_s\hat{\mathbf{u}} + \mathbf{m}\,e^{+i\omega t}$
# and linearizing the LLG equation to first order in $\mathbf{m}$ and $\mathbf{h}$
# yields two coupled transverse equations whose diagonal susceptibility has a
# resonant denominator at the **natural FMR frequency** $\omega_0 = \gamma_{\text{LL}}\mu_0 H_a$.
#
# In an **unbiased polycrystalline ferrite** composed of randomly oriented
# domains, the antisymmetric off-diagonal gyrotropic terms ($\pm i\kappa$)
# cancel on macroscopic scales, leaving a scalar **Lorentz magnetic
# susceptibility** (`Lorentz(delta_eps=delta_mu, omega_0=w0, gamma=gamma_m)` in
# `mu_poles`):
#
# $$\mu_{\text{Lorentz}}(\omega)
#   = \mu_\infty + \frac{\Delta\mu\,\omega_0^2}{\omega_0^2 - \omega^2 + i\gamma_m\omega},
#   \qquad
#   \Delta\mu = \mu_s - \mu_\infty,
#   \qquad
#   \ddot{\mathbf{m}}_p + \gamma_{m,p}\dot{\mathbf{m}}_p + \omega_{0,p}^2\mathbf{m}_p
#   = \Delta\mu_p\,\omega_{0,p}^2\,\mathbf{H}(t),$$
#
# where $\mu_s = \mu(0) = \mu_\infty + \Delta\mu$ is the static relative
# permeability, $f_r = \omega_0 / (2\pi)$ is the natural resonance frequency,
# and $\gamma_m \approx 2\alpha_{\text{G}}\omega_0$ is the linewidth.
#
# ### 1.3 Domain-wall relaxation and the Debye $\mu(\omega)$ model
#
# In sintered cubic spinel ferrites with very large initial permeability (such
# as **NiZn absorber tiles** with $\mu_s \approx 2000\text{–}3200$), the response
# below $100\;\text{MHz}$ is dominated not by coherent spin rotation inside
# domains, but by the reversible bowing and displacement of **domain walls**
# pinned at grain boundaries. For a domain wall of effective mass per unit area
# $m_{\text{dw}}$, viscous damping $\beta_{\text{dw}}$, and restoring stiffness
# $\kappa_{\text{dw}}$, wall inertia is negligible compared to viscous damping
# ($m_{\text{dw}}\omega^2 \ll \beta_{\text{dw}}\omega$), reducing the second-order
# ODE to a first-order **Debye magnetic relaxation equation**
# (`Debye(delta_eps=delta_mu, tau=tau_m)` in `mu_poles`):
#
# $$\mu_{\text{Debye}}(\omega)
#   = \mu_\infty + \frac{\Delta\mu}{1 + i\omega\tau_m},
#   \qquad
#   \tau_m \dot{\mathbf{m}}_p + \mathbf{m}_p = \Delta\mu_p\,\mathbf{H}(t),
#   \qquad
#   f_{\text{rel}} = \frac{1}{2\pi\tau_m}.$$
#
# Many commercial ferrites exhibit a **two-mechanism spectrum**: a large, low-MHz
# Debye domain-wall relaxation peak plus a smaller GHz-range Lorentz rotational
# FMR peak, as illustrated in the combined LLG precession and broadband
# permeability diagram below.
#
# ![Landau-Lifshitz-Gilbert damped spin precession and broadband magnetic permeability spectrum](figures/wikimedia/fmr_precession_and_permeability_wakis.png)
# *Figure — **(a)** Landau–Lifshitz–Gilbert (LLG) damped spin precession of the magnetization $\mathbf{M}(t)$ around the effective anisotropy/bias field $\mathbf{H}_{\text{eff}}$, showing the Larmor precessional torque $-\gamma_{\text{LL}}\mu_0(\mathbf{M}\times\mathbf{H}_{\text{eff}})$ (tangent to the precession cone) and the Gilbert damping torque $\frac{\alpha_{\text{G}}}{M_s}(\mathbf{M}\times\dot{\mathbf{M}})$ that spirals $\mathbf{M}$ toward $\mathbf{H}_{\text{eff}}$; **(b)** resulting broadband relative permeability $\mu'(f)$ and magnetic loss factor $\mu''(f) = -\operatorname{Im}\mu(f)$ of a composite ferrite (`DispersiveMaterial(mu_poles=[Debye(...), Lorentz(...)])`), connecting the low-MHz domain-wall `Debye` relaxation band ($f_{\text{rel}} = 1/(2\pi\tau_m)$) and the microwave rotational FMR `Lorentz` resonance ($f_r = \gamma_{\text{LL}}\mu_0 H_a/(2\pi)$, negative-$\mu'$ stopband $[f_r, f_r\sqrt{\mu_s/\mu_\infty}]$, and Snoek's limit $(\mu_{s,\text{rot}} - 1)f_r = \frac{2}{3}(\gamma_{\text{LL}}/2\pi)\mu_0 M_s$) to Wakis's `mu_poles` classes. Original Wakis illustration generated with `wakis.figstyle`; panel (a) layout after Wikimedia Commons [`File:Precession_of_magnetization.svg`](https://commons.wikimedia.org/wiki/File:Precession_of_magnetization.svg) by Geek3 ([CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)) and [`File:Larmor_Precession.svg`](https://commons.wikimedia.org/wiki/File:Larmor_Precession.svg) by Lookang ([CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)), and panel (b) physics from J. L. Snoek, Physica **14**, 207 (1948) and O. Acher & S. Dubourg, Phys. Rev. B **77**, 104440 (2008). Full details in [`figures/wikimedia/ATTRIBUTION.md`](figures/wikimedia/ATTRIBUTION.md).*
#
# ---
#
# ## 2. Snoek's Law, Acher's Thin-Film Generalization, Loss Tangent & Hesitivity
#
# ### 2.1 Snoek's law for isotropic cubic ferrites (Snoek, 1948)
#
# In a cubic ferrite with uniaxial anisotropy field $H_a$, an RF field applied
# at angle $\theta$ to the domain easy axis sees a transverse torque proportional
# to $\sin^2\theta$. Averaging $\langle\sin^2\theta\rangle = 2/3$ over an
# isotropic 3D distribution of domain orientations gives the static rotational
# susceptibility and natural resonance frequency (**Snoek, 1948** [1];
# **Ikonen *et al.*, 2006** [3], Eq. 18):
#
# $$\mu_s - 1 = \frac{2}{3}\,\frac{M_s}{H_a},
#   \qquad
#   \omega_r = 2\pi f_r = \gamma_{\text{LL}}\mu_0 H_a
#   \quad \Longrightarrow \quad
#   (\mu_s - 1)\,f_r = \frac{2}{3}\,\frac{\gamma_{\text{LL}}}{2\pi}\,\mu_0 M_s = \text{const}.$$
#
# Because $M_s$ in oxide ferrites is bounded by $\mu_0 M_s \lesssim 0.4\text{–}0.5\;\text{T}$,
# **Snoek's product** for bulk cubic NiZn spinels is limited to
# $(\mu_s - 1)f_r \approx 3\text{–}5\;\text{GHz}$: increasing $\mu_s$ by lowering
# $H_a$ inevitably pushes the ferromagnetic resonance $f_r$ down in inverse
# proportion.
#
# ### 2.2 Acher's generalization for planar hexaferrites and thin ferromagnetic films
#
# **Snoek's limit can be broken by breaking 3D isotropy.** In a thin
# ferromagnetic alloy film (such as $\text{CoZrNb}$ laminates) or a planar
# Z-type hexaferrite ($\text{Ba}_3\text{Co}_2\text{Fe}_{24}\text{O}_{41}$,
# $\text{Co}_2\text{Z}$), a strong out-of-plane anisotropy or shape
# demagnetizing field $H_\theta \gg H_\phi$ confines $\mathbf{M}$ to the easy
# plane while a much smaller in-plane anisotropy $H_\phi$ controls rotation
# within the plane. By Kittel's formula, the precessional resonance rises to the
# geometric mean $\omega_r = \gamma_{\text{LL}}\mu_0\sqrt{H_\phi H_\theta}$,
# while the in-plane static susceptibility remains $\mu_s - 1 = M_s / H_\phi$:
#
# $$\frac{(\mu_s - 1)\,f_r\big|_{\text{planar}}}{(\mu_s - 1)\,f_r\big|_{\text{Snoek}}}
#   = \frac{3}{2}\sqrt{\frac{H_\theta}{H_\phi}} \gg 1.$$
#
# Eliminating $H_\phi$ yields **Acher's quadratic bound and integral sum rule**
# (**Acher & Dubourg, 2008** [2]; **Ikonen *et al.*, 2006** [3], Eq. 19):
#
# $$(\mu_s - 1)\,f_r^2 \le c_{\text{Acher}}\left(\frac{\gamma_{\text{LL}}}{2\pi}\mu_0 M_s\right)^2,
#   \qquad
#   \int_0^\infty f\,\mu''(f)\,df
#   = -\int_0^\infty f\,\operatorname{Im}\mu(f)\,df
#   = \frac{\pi}{2}\,(\mu_s - \mu_\infty)\,f_r^2.$$
#
# Notice that Acher's integral $\int_0^\infty f\,\mu''(f)\,df = \frac{\pi}{2}\Delta\mu\,f_r^2$
# is the exact magnetic counterpart of the electric Thomas–Reiche–Kuhn $f$-sum
# rule from Tutorial 2.1, and is **independent of the damping linewidth $\gamma_m$**!
#
# ### 2.3 Magnetic loss tangent $\tan\delta_m(f)$ vs. hesitivity $h_m$
#
# Writing $\mu(\omega) = \mu'(\omega) - i\,\mu''(\omega)$ ($\mu'' = -\operatorname{Im}\mu \ge 0$),
# the **magnetic loss tangent** is
#
# $$\tan\delta_m(f)
#   = \frac{\mu''(f)}{\mu'(f)}
#   = -\frac{\operatorname{Im}\mu(f)}{\operatorname{Re}\mu(f)}.$$
#
# For a Lorentz ferrite ($\mu_\infty = 1$, $\Delta\mu = \mu_s - 1$, $\gamma_m = \beta\omega_r$)
# at frequencies well below resonance ($f \ll f_r$),
#
# $$\tan\delta_m(f)
#   = \frac{\Delta\mu\,\beta\,(f/f_r)}{\mu_s - (\mu_s + 1)(f/f_r)^2 + (f/f_r)^4 + \beta^2(f/f_r)^2}
#   \;\underset{f \ll f_r}{\approx}\;
#   \frac{\mu_s - 1}{\mu_s}\,\beta\,\frac{f}{f_r}.$$
#
# Combining this linear rise in $\tan\delta_m(f)$ with Snoek's law $f_r \propto (\mu_s - 1)^{-1}$
# shows that $\tan\delta_m(f) \propto (\mu_s - 1)^2\,f$ at fixed $M_s$: **higher
# permeability means a much steeper rise of $\tan\delta_m$ with frequency**.
# For a **resonant patch or microstrip antenna** on a magnetic substrate of
# radiation quality factor $Q_r$, **Ikonen *et al.* (2006)** [3, Eq. 20] showed
# that maintaining radiation efficiency $\eta_r$ requires
# $\tan\delta_m(f) < Q_r^{-1}(1 - \eta_r)/\eta_r$, which restricts the usable
# antenna band to **well below $f_r$** (where $\mu'' \ll \mu'$).
#
# For a non-resonant electrically small **flux-channel radiator** (such as a
# loop-fed permeable rod or toroid where the magnetic flux is guided inside the
# core), **Sebastian (2013)** [5] and **Yousefi, Sebastian & Diaz (2016)** [4,
# Eqs. 7–19] showed that because the radiating magnetic current is proportional
# to the susceptibility $\chi_m(\omega) = \mu(\omega) - 1$, the material loss
# figure of merit is not $\tan\delta_m = \mu''/\mu'$, but the inverse
# susceptibility loss function:
#
# $$f(\mu)
#   \equiv \frac{\mu''(\omega)}{|\mu(\omega) - 1|^2}
#   = +\operatorname{Im}\!\left[\frac{1}{\mu(\omega) - 1}\right]
#   \qquad (\text{in }\exp(+i\omega t)\text{ where }\operatorname{Im}\mu = -\mu'' \le 0).$$
#
# Substituting either a single Debye pole $\chi_m(\omega) = \Delta\mu / (1 + i\omega\tau_m)$
# or a single Lorentz pole $\chi_m(\omega) = \Delta\mu\,\omega_0^2 / (\omega_0^2 - \omega^2 + i\gamma_m\omega)$
# (with $\mu_\infty = 1$) yields an **exact identity at every frequency $\omega$**:
#
# $$\operatorname{Im}\!\left[\frac{1}{\mu(\omega) - 1}\right]
#   = \frac{\omega\mu_0}{h_m},
#   \qquad
#   h_m \equiv \max_\omega \sigma_m(\omega)
#   = \begin{cases}
#     \omega_{\text{rel}}\,\mu_0\,\Delta\mu = \dfrac{\mu_0\,\Delta\mu}{\tau_m} & \text{(Debye)}, \\[8pt]
#     \dfrac{\omega_0^2\,\mu_0\,\Delta\mu}{\gamma_m} = \dfrac{\omega_0\,\mu_0\,\Delta\mu}{\alpha} & \text{(Lorentz, }\gamma_m = \alpha\omega_0\text{)}.
#   \end{cases}$$
#
# Here $\sigma_m(\omega) = \omega\mu_0\mu''(\omega)$ $[\Omega/\text{m}]$ is the
# equivalent AC magnetic conductivity and **$h_m$ $[\Omega/\text{m}]$ is the
# hesitivity** (its peak value). Chapter 4 uses these exact material models to
# test when flux-channel antennas follow the hesitivity law and where finite skin
# depth and trapped $\text{TE}_{01}$ core modes break it.
#
# ---
#
# ## 3. Magnetic Material Parameters from the Literature
#
# The literature material models used in this tutorial are summarized below (with
# full-text verification status noted per entry):
#
# | Key | Material | Model | $\mu_\infty$ | $\Delta\mu$ ($\mu_s$) | $f_r$ or $f_{\text{rel}}$ | Linewidth $\gamma_m/(2\pi)$ | Hesitivity $h_m\;[\Omega/\text{m}]$ | $\varepsilon_r$ | Source & Notes |
# | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
# | `czn_eq881` | **CoZrNb (CZN) laminate** on $12\;\mu\text{m}$ polyimide | Lorentz | $1.0$ | $39.0$ ($40.0$) | $1.50\;\text{GHz}$ | $2.00\;\text{GHz}$ ($\alpha=1/0.75$) | $3.46\times 10^5$ | $3.25$ | **Yousefi (2017)** Eq. 8-81 [6] *(unverified formula; dissertation chapter not verified in full text)* |
# | `co2z_unaligned` | **Unaligned $\text{Co}_2\text{Z}$ planar hexaferrite** ($\text{Ba}_3\text{Co}_2\text{Fe}_{24}\text{O}_{41}$) | Lorentz | $1.0$ | $11.0$ ($12.0$) | $1.00\;\text{GHz}$ | $0.833\;\text{GHz}$ | $1.042\times 10^5$ | $12.0$ | **Sebastian (2013)** Table 5-2 & Fig. 5-13 [5]; **Yousefi *et al.* (2016)** *IEEE TAP* Table I [4] *(unverified in full text; aligned $\text{Co}_2\text{Z}$ reaches $h_m = 2.8\times 10^5\;\Omega/\text{m}$)* |
# | `ikonen_film` | **Laminated ferromagnetic film substrate** | Lorentz | $1.0$ | $0.48$ ($1.48$) | $5.87\;\text{GHz}$ | $5.301\;\text{GHz}$ ($\beta=0.903$) | $1.61\times 10^4$ | $2.0$ | **Ikonen *et al.* (2006)** *IEEE TAP* **54**, 3391, Eq. 17 & Fig. 4 [3] *(full-text verified; $\mu'=1.461, \tan\delta_m=0.0284$ at $580\;\text{MHz}$)* |
# | `nizn_debye` | **Commercially available sintered NiZn ferrite tile** | Debye | $1.0$ | $2199.0$ ($2200.0$) | $3.00\;\text{MHz}$ ($\tau_m=53.05\;\text{ns}$) | — | $5.21\times 10^4$ | $14.0$ | **Sebastian (2013)** Table 5-2 & Fig. 3-13a [5]; **Yousefi *et al.* (2016)** Table I & Fig. 8a [4]; **McLean & Sutton (2012)** Fig. 2 [7] *(unverified in full text; **flagged:** literature tables quote $h_m = 3.0\times 10^4\text{–}5.524\times 10^4\;\Omega/\text{m}$ and 100 MHz plot readings vary between $\mu'\approx 2\text{–}3$ and $10\text{–}40$)* |

# %%
import argparse
import json
import os
from pathlib import Path
import resource
import shlex
import sys
import time
import warnings

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c0
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0

from types import SimpleNamespace

from wakis import FluxMonitor, GridFIT3D, SolverFIT3D
from wakis.dispersive import (
    Debye,
    DispersiveMaterial,
    Drude,
    Lorentz,
)
import wakis.figstyle as figstyle
import wakis.schematic as schematic
from wakis.sources import CurrentSource, GaussianSource

warnings.filterwarnings("ignore", message="IProgress not found")
figstyle.apply_style()

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_ROOT = (
    HERE.parents[2] if (HERE.name == "02-03_magnetic_materials") else Path.cwd()
)
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"


# %%
# Literature magnetic material models (only ikonen_film is full-text verified)
VERIFIED_MAGNETIC_MATERIALS = {
    "czn_eq881": {
        "label": "CoZrNb Laminate (Yousefi Eq. 8-81)",
        "model": "Lorentz",
        "mu_inf": 1.0,
        "delta_mu": 39.0,
        "mu_s": 40.0,
        "f_r_Hz": 1.50e9,
        "gamma_Hz": 1.50e9 / 0.75,
        "eps_r": 3.25,
        "hm_published_Ohm_m": 3.46e5,
        "verification_status": "unverified formula (Yousefi 2017 Eq. 8-81)",
        "flag_note": "Eq. 8-81 single-Lorentz model parameters from dissertation notes (not verified in full text).",
    },
    "co2z_unaligned": {
        "label": "Co2Z Hexaferrite Unaligned (Sebastian Table 5-2)",
        "model": "Lorentz",
        "mu_inf": 1.0,
        "delta_mu": 11.0,
        "mu_s": 12.0,
        "f_r_Hz": 1.00e9,
        # gamma_Hz chosen so h_m = 2*pi*f_r^2*mu_0*delta_mu / gamma_Hz = 1.042e5 Ohm/m (Sebastian Table 5-2)
        "gamma_Hz": (2.0 * np.pi * (1.00e9**2) * mu_0 * 11.0) / 1.042e5,
        "eps_r": 12.0,
        "hm_published_Ohm_m": 1.042e5,
        "verification_status": "unverified (Sebastian 2013 Table 5-2 & Fig. 5-13; Yousefi et al. TAP 2016 Table I)",
        "flag_note": "Aligned Co2Z reaches h_m = 2.8e5 Ohm/m (TAP 2016 Table I).",
    },
    "ikonen_film": {
        "label": "Ferromagnetic Film Substrate (Ikonen 2006 Eq. 17)",
        "model": "Lorentz",
        "mu_inf": 1.0,
        "delta_mu": 0.48,
        "mu_s": 1.48,
        "f_r_Hz": 5.87e9,
        "gamma_Hz": 0.903 * 5.87e9,
        "eps_r": 2.0,
        "hm_published_Ohm_m": (2.0 * np.pi * 5.87e9 * mu_0 * 0.48) / 0.903,
        "verification_status": "full-text verified (Ikonen et al., IEEE TAP 54, 3391, 2006, Eq. 17 & Fig. 4)",
        "flag_note": "Gives mu'=1.461, tan_delta_m=0.0284 at 580 MHz patch operating frequency.",
    },
    "nizn_debye": {
        "label": "Sintered NiZn Spinel Ferrite Tile (commercially available)",
        "model": "Debye",
        "mu_inf": 1.0,
        "delta_mu": 2199.0,
        "mu_s": 2200.0,
        "f_rel_Hz": 3.00e6,
        "tau_s": 1.0 / (2.0 * np.pi * 3.00e6),
        "eps_r": 14.0,
        "hm_published_Ohm_m": 5.524e4,
        "verification_status": "unverified (Sebastian 2013 Table 5-2; Yousefi TAP 2016 Table I; McLean & Sutton 2012 Fig. 2)",
        "flag_note": "Flagged discrepancy: sources quote h_m = 3.0e4 (TAP 2016 Table I), 3.5e4 (Yousefi Ch. 6), or 5.524e4 (Sebastian Table 5-2), while mu_s=2200, f_rel=3 MHz implies h_m = w0*mu0*dmu = 5.21e4 Ohm/m.",
    },
}


def build_verified_material(key):
    """
    Construct a ``DispersiveMaterial`` from ``VERIFIED_MAGNETIC_MATERIALS[key]``.

    Parameters
    ----------
    key : str
        Key in ``VERIFIED_MAGNETIC_MATERIALS``.

    Returns
    -------
    DispersiveMaterial
        Configured magnetic dispersive material.
    """
    spec = VERIFIED_MAGNETIC_MATERIALS[key]
    if spec["model"] == "Lorentz":
        w0 = 2.0 * np.pi * spec["f_r_Hz"]
        gam = 2.0 * np.pi * spec["gamma_Hz"]
        pole = Lorentz(spec["delta_mu"], w0, gam)
    elif spec["model"] == "Debye":
        pole = Debye(spec["delta_mu"], spec["tau_s"])
    else:
        raise ValueError(f"Unknown model {spec['model']}")
    return DispersiveMaterial(
        eps_inf=float(spec["eps_r"]),
        mu_inf=float(spec["mu_inf"]),
        mu_poles=[pole],
        name=key,
    )


def compute_hesitivity(freqs, mu_vals):
    """
    Compute the AC magnetic conductivity ``sigma_m(f) = 2*pi*f*mu_0*(-Im mu(f))``,
    the magnetic loss tangent ``tan_delta_m(f) = -Im(mu) / Re(mu)``, the
    flux-channel loss function ``f_loss(f) = -Im(1 / (mu(f) - 1))``, and the
    effective hesitivity ``h_eff(f) = 2*pi*f*mu_0 / f_loss(f)`` [Ohm/m].

    Parameters
    ----------
    freqs : array_like
        Frequencies in Hz [Hz].
    mu_vals : array_like
        Complex relative permeability ``mu(freqs)`` in ``exp(+i*omega*t)``
        convention (``Im(mu) <= 0``).

    Returns
    -------
    dict
        Dictionary with ``sigma_m``, ``tan_delta_m``, ``f_loss``, ``h_eff``,
        and ``hm_peak``.
    """
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64)
    mu_vals = np.asarray(mu_vals, dtype=np.complex128)
    mu_prime = np.real(mu_vals)
    mu_dprime = -np.imag(mu_vals)
    sigma_m = omega * mu_0 * mu_dprime
    tan_delta_m = mu_dprime / np.where(
        np.abs(mu_prime) > 1e-12, mu_prime, 1e-12
    )
    chi_m = mu_vals - 1.0
    f_loss = np.imag(1.0 / chi_m)
    h_eff = (omega * mu_0) / np.maximum(f_loss, 1e-30)
    return {
        "sigma_m": sigma_m,
        "tan_delta_m": tan_delta_m,
        "f_loss": f_loss,
        "h_eff": h_eff,
        "hm_peak": float(np.max(sigma_m)),
    }


def check_acher_integral_sum_rule(delta_mu, f_r_Hz, gamma_Hz, f_max_Hz=200.0e9):
    """
    Numerically verify Acher's integral sum rule for a Lorentz permeability pole:

        int_0^inf f * (-Im mu(f)) df = (pi / 2) * delta_mu * f_r^2,

    including the exact analytic high-frequency tail above ``f_max_Hz``:

        int_{f_max}^inf f * (-Im mu(f)) df ~ delta_mu * f_r^2 * gamma_Hz / f_max.
    """
    f_grid = np.linspace(1.0e5, f_max_Hz, 40000)
    mat = DispersiveMaterial(
        mu_inf=1.0,
        mu_poles=[
            Lorentz(delta_mu, 2.0 * np.pi * f_r_Hz, 2.0 * np.pi * gamma_Hz)
        ],
    )
    mu_im = -np.imag(mat.mu(f_grid))
    num_int = float(np.trapezoid(f_grid * mu_im, f_grid))
    tail_corr = float(delta_mu * (f_r_Hz**2) * gamma_Hz / f_max_Hz)
    total_int = num_int + tail_corr
    exact_int = float(0.5 * np.pi * delta_mu * (f_r_Hz**2))
    rel_err = abs(total_int - exact_int) / exact_int
    return {
        "numerical_integral_Hz2": total_int,
        "exact_integral_Hz2": exact_int,
        "rel_err": float(rel_err),
    }


def run_single_cell_magnetic_ade(material, freqs, dt=1.0e-11, nt=4000, f0=1.5e9, fwidth=1.0e9):
    """
    Verify Wakis's magnetic ADE update on a 1x1x1 periodic cell by driving the
    primary electric field with a known curl and comparing the discrete Fourier
    transform of ``B(t) / (mu_0 * H(t))`` against ``material.mu_discrete(freqs, dt)``.

    Wait: on a 1x1x1 periodic cell, ``curl E == 0``, so ``H`` would remain zero
    unless ``curl E`` is non-zero! Instead, we use a tiny 2x2x16 periodic grid
    with a spatially varying ``Ex(z)`` pulse so ``(curl E)_y = dEx/dz != 0``
    drives ``Hy`` through ``_update_dispersive_H``, and extract
    ``mu_FIT(f) = -(curl E)_y(f) / (i * W(f) * mu_0 * Hy(f))`` where
    ``W(f) = (2/dt) * sin(pi * f * dt)``!
    """
    nz = 16
    dz = c0 * dt / 0.25
    grid = GridFIT3D(
        -dz, dz, -dz, dz, 0.0, nz * dz, 2, 2, nz, verbose=0
    )
    solver = SolverFIT3D(
        grid,
        cfln=0.25,
        bc_low=["periodic", "periodic", "periodic"],
        bc_high=["periodic", "periodic", "periodic"],
        use_stl=False,
        verbose=0,
    )
    solver.dt = float(dt)
    solver.add_dispersive_material(
        material,
        center=(0.0, 0.0, 0.5 * nz * dz),
        size=(4.0 * dz, 4.0 * dz, 2.0 * nz * dz),
    )

    t_j = (np.arange(nt) + 0.5) * dt
    sig_t = 1.0 / (2.0 * np.pi * fwidth)
    t0 = 4.5 * sig_t
    j_wave = (
        -((t_j - t0) / sig_t)
        * np.exp(-0.5 * ((t_j - t0) / sig_t) ** 2)
        * np.cos(2.0 * np.pi * f0 * (t_j - t0))
    )

    k_probe = 4
    hy_trace = np.zeros(nt, dtype=np.float64)
    curl_e_trace = np.zeros(nt, dtype=np.float64)
    for n in range(nt):
        solver.J[:, :, k_probe, "x"] = j_wave[n]
        solver.J[:, :, k_probe + 1, "x"] = -j_wave[n]
        # At time t^n, capture (curl E)_y = (Ex[k+1] - Ex[k]) / dz that drives Hy^(n+1/2)
        ex_k = float(solver.E[0, 0, k_probe, "x"])
        ex_kp1 = float(solver.E[0, 0, k_probe + 1, "x"])
        curl_e_trace[n] = (ex_kp1 - ex_k) / dz
        solver.one_step()
        hy_trace[n] = float(solver.H[0, 0, k_probe, "y"])

    # In the leapfrog step: mu_0 * mu_d * (Hy^(n+1/2) - Hy^(n-1/2)) / dt = -(curl E)_y^n
    # With t_E = n * dt and t_H = (n + 0.5) * dt:
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64)
    t_e = np.arange(nt) * dt
    t_h = (np.arange(nt) + 0.5) * dt
    curl_e_hat = np.sum(
        curl_e_trace[None, :] * np.exp(-1j * omega[:, None] * t_e[None, :]),
        axis=1,
    ) * dt
    hy_hat = np.sum(
        hy_trace[None, :] * np.exp(-1j * omega[:, None] * t_h[None, :]),
        axis=1,
    ) * dt
    w_tilde = (2.0 / dt) * np.sin(0.5 * omega * dt)
    mu_fit = -curl_e_hat / (1j * w_tilde * mu_0 * hy_hat)
    mu_disc = material.mu_discrete(freqs, dt)
    mu_cont = material.mu(freqs)
    rel_err_disc = float(
        np.linalg.norm(mu_fit - mu_disc) / np.linalg.norm(mu_disc)
    )
    rel_err_cont = float(
        np.linalg.norm(mu_fit - mu_cont) / np.linalg.norm(mu_cont)
    )
    return {
        "mu_fit": mu_fit,
        "mu_disc": mu_disc,
        "mu_cont": mu_cont,
        "rel_err_disc": rel_err_disc,
        "rel_err_cont": rel_err_cont,
    }


def transfer_matrix_slab(freqs, d, eps_r, mu_r):
    """
    Exact normal-incidence complex amplitude ``(r, t)`` and power ``(R, T, A)``
    of a homogeneous magneto-dielectric slab ``(eps_r, mu_r)`` of thickness
    ``d`` [m] in vacuum under the Wakis ``exp(+i*omega*t)`` convention.
    """
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64)
    eps_r = np.asarray(eps_r, dtype=np.complex128)
    mu_r = np.asarray(mu_r, dtype=np.complex128)
    eps_r, mu_r = np.broadcast_arrays(eps_r, mu_r)
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


def run_magnetic_slab_pair(
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
    Run paired quasi-1D ``SolverFIT3D`` simulations (magnetic/magneto-dielectric
    slab + vacuum reference) and compare against ``transfer_matrix_slab``
    evaluated at ``(epsilon_discrete, mu_discrete)`` and ``(epsilon, mu)``.
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
        nt = int(round((src_time.end_time + t_after_src) / solver.dt))
        for n in range(nt):
            t_now = n * solver.dt
            solver.J[:, :, :, "x"] = 0.0
            sheet.update(solver, t_now)
            solver.one_step()
            for mon in planes:
                mon.update(solver, t_now)
        return [p.get_flux() for p in planes], float(solver.dt), int(nt)

    (flux_r_ref, flux_t_ref), dt, nt = _run_one(None)
    (flux_r, flux_t), _, _ = _run_one(material)

    R_fit = 1.0 - flux_r / flux_r_ref
    T_fit = flux_t / flux_t_ref
    A_fit = 1.0 - R_fit - T_fit

    eps_cont = material.epsilon(freqs)
    eps_disc = material.epsilon_discrete(freqs, dt)
    mu_cont = material.mu(freqs)
    mu_disc = material.mu_discrete(freqs, dt)

    tm_cont = transfer_matrix_slab(freqs, thickness, eps_cont, mu_cont)
    tm_disc = transfer_matrix_slab(freqs, thickness, eps_disc, mu_disc)

    rms_R_disc = float(np.sqrt(np.mean((R_fit - tm_disc["R"]) ** 2)))
    rms_T_disc = float(np.sqrt(np.mean((T_fit - tm_disc["T"]) ** 2)))
    rms_R_cont = float(np.sqrt(np.mean((R_fit - tm_cont["R"]) ** 2)))
    rms_T_cont = float(np.sqrt(np.mean((T_fit - tm_cont["T"]) ** 2)))

    n_complex = np.sqrt(eps_cont * mu_cont)
    n_complex = np.where(np.imag(n_complex) > 0.0, -n_complex, n_complex)
    n_max = float(np.max(np.maximum(np.real(n_complex), 1.0)))
    lam_min_vac = float(c0 / np.max(freqs))
    cpw_mat = (lam_min_vac / n_max) / dz
    cells_across = thickness / dz

    return {
        "freqs": freqs,
        "dt": dt,
        "nt": nt,
        "nz": nz,
        "dz": dz,
        "thickness": thickness,
        "cpw_mat": float(cpw_mat),
        "cells_across": float(cells_across),
        "R_fit": R_fit,
        "T_fit": T_fit,
        "A_fit": A_fit,
        "R_cont": tm_cont["R"],
        "T_cont": tm_cont["T"],
        "A_cont": tm_cont["A"],
        "R_disc": tm_disc["R"],
        "T_disc": tm_disc["T"],
        "A_disc": tm_disc["A"],
        "eps_cont": eps_cont,
        "mu_cont": mu_cont,
        "eps_disc": eps_disc,
        "mu_disc": mu_disc,
        "rms_R_disc": rms_R_disc,
        "rms_T_disc": rms_T_disc,
        "rms_R_cont": rms_R_cont,
        "rms_T_cont": rms_T_cont,
    }


def analytic_cavity_polariton_branches(f_vac_Hz, mu_inf, delta_mu, f_r_Hz):
    """
    Exact lossless hybridized cavity-magnon polariton eigenfrequencies
    ``(f_minus, f_plus)`` solving ``f^2 * mu(f) = f_vac^2`` for a single
    Lorentz magnetic pole ``mu(f) = mu_inf + delta_mu * f_r^2 / (f_r^2 - f^2)``:

        mu_inf * f^4 - (f_vac^2 + mu_s * f_r^2) * f^2 + f_vac^2 * f_r^2 = 0,

    where ``mu_s = mu_inf + delta_mu``.
    """
    f_r_arr = np.asarray(f_r_Hz, dtype=np.float64)
    mu_s = float(mu_inf + delta_mu)
    b = float(f_vac_Hz**2) + mu_s * (f_r_arr**2)
    disc = np.sqrt(
        np.maximum(
            b * b - 4.0 * float(mu_inf) * float(f_vac_Hz**2) * (f_r_arr**2),
            0.0,
        )
    )
    f2_minus = (b - disc) / (2.0 * float(mu_inf))
    f2_plus = (b + disc) / (2.0 * float(mu_inf))
    return np.sqrt(f2_minus), np.sqrt(f2_plus)



# %%
def run_task_0(quick=False):
    """
    Task 0: Evaluate full-text verified literature models (CoZrNb Eq. 8-81,
    unaligned Co2Z hexaferrite, Ikonen ferromagnetic film, commercially available
    sintered NiZn ferrite tile Debye spinel), verify Snoek's law and Acher's integral sum rule, verify the
    single-pole hesitivity identity -Im[1/(mu-1)] = w*mu0/h_m, and run the
    periodic FIT magnetic ADE verification against mu_discrete(f, dt).
    """
    t0 = time.perf_counter()
    f_broad = np.geomspace(1.0e6, 20.0e9, 400)

    lit_spectra = {}
    lit_metrics = {}
    for key, spec in VERIFIED_MAGNETIC_MATERIALS.items():
        mat = build_verified_material(key)
        mu_f = mat.mu(f_broad)
        hes = compute_hesitivity(f_broad, mu_f)
        lit_spectra[key] = mu_f
        # Relative variation of h_eff(f) across the entire 1 MHz - 20 GHz band
        h_eff_rel_std = float(np.std(hes["h_eff"]) / np.mean(hes["h_eff"]))
        lit_metrics[key] = {
            "label": spec["label"],
            "model": spec["model"],
            "mu_s": float(spec["mu_s"]),
            "eps_r": float(spec["eps_r"]),
            "hm_published_Ohm_m": float(spec["hm_published_Ohm_m"]),
            "hm_analytic_Ohm_m": float(np.mean(hes["h_eff"])),
            "hm_peak_sigma_m_Ohm_m": hes["hm_peak"],
            "h_eff_constancy_rel_std": h_eff_rel_std,
            "verification_status": spec["verification_status"],
            "flag_note": spec["flag_note"],
        }

    # Check Ikonen film specific operating point at 580 MHz (Ikonen 2006 Fig. 4)
    mat_ik = build_verified_material("ikonen_film")
    mu_ik_580 = complex(mat_ik.mu(np.array([580.0e6]))[0])
    tan_d_ik_580 = float(-np.imag(mu_ik_580) / np.real(mu_ik_580))
    lit_metrics["ikonen_film"]["mu_real_580MHz"] = float(np.real(mu_ik_580))
    lit_metrics["ikonen_film"]["tan_delta_m_580MHz"] = tan_d_ik_580

    # Snoek's family: (mu_s - 1) * f_r = 3.9 GHz
    snoek_product_ghz = 3.9
    snoek_mu_s = np.array([2.3, 3.0, 4.9, 8.8], dtype=np.float64)
    snoek_f_r_ghz = snoek_product_ghz / (snoek_mu_s - 1.0)
    f_lin = np.linspace(0.05e9, 5.0e9, 400)
    snoek_mu_curves = np.zeros(
        (snoek_mu_s.size, f_lin.size), dtype=np.complex128
    )
    snoek_tan_d_curves = np.zeros(
        (snoek_mu_s.size, f_lin.size), dtype=np.float64
    )
    for idx, (ms, fr_g) in enumerate(zip(snoek_mu_s, snoek_f_r_ghz)):
        w0 = 2.0 * np.pi * (fr_g * 1e9)
        m_s = DispersiveMaterial(
            mu_inf=1.0, mu_poles=[Lorentz(ms - 1.0, w0, 0.20 * w0)]
        )
        mu_c = m_s.mu(f_lin)
        snoek_mu_curves[idx] = mu_c
        snoek_tan_d_curves[idx] = -np.imag(mu_c) / np.where(
            np.abs(np.real(mu_c)) > 1e-9, np.real(mu_c), 1e-9
        )

    # Acher integral sum rule check at two damping values
    acher_low_g = check_acher_integral_sum_rule(
        delta_mu=11.0, f_r_Hz=1.5e9, gamma_Hz=0.25e9
    )
    acher_high_g = check_acher_integral_sum_rule(
        delta_mu=11.0, f_r_Hz=1.5e9, gamma_Hz=0.90e9
    )

    # Periodic FIT magnetic ADE verification against mu_discrete(f, dt)
    f_ade = np.linspace(0.4e9, 2.6e9, 45)
    dt_ade = 1.2e-11
    nt_ade = 2500 if quick else 4000
    mat_lor_ade = DispersiveMaterial(
        mu_inf=1.2,
        mu_poles=[Lorentz(2.8, 2.0 * np.pi * 1.4e9, 2.0 * np.pi * 0.28e9)],
    )
    mat_deb_ade = DispersiveMaterial(
        mu_inf=1.1,
        mu_poles=[Debye(3.2, 1.0 / (2.0 * np.pi * 1.1e9))],
    )
    ade_lor = run_single_cell_magnetic_ade(
        mat_lor_ade, f_ade, dt=dt_ade, nt=nt_ade
    )
    ade_deb = run_single_cell_magnetic_ade(
        mat_deb_ade, f_ade, dt=dt_ade, nt=nt_ade
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_DIR / "task_0.npz",
        f_broad=f_broad,
        mu_czn=lit_spectra["czn_eq881"],
        mu_co2z=lit_spectra["co2z_unaligned"],
        mu_ikonen=lit_spectra["ikonen_film"],
        mu_nizn=lit_spectra["nizn_debye"],
        f_lin=f_lin,
        snoek_mu_s=snoek_mu_s,
        snoek_f_r_ghz=snoek_f_r_ghz,
        snoek_mu_curves=snoek_mu_curves,
        snoek_tan_d_curves=snoek_tan_d_curves,
        f_ade=f_ade,
        mu_fit_lor=ade_lor["mu_fit"],
        mu_disc_lor=ade_lor["mu_disc"],
        mu_cont_lor=ade_lor["mu_cont"],
        mu_fit_deb=ade_deb["mu_fit"],
        mu_disc_deb=ade_deb["mu_disc"],
        mu_cont_deb=ade_deb["mu_cont"],
    )

    elapsed = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    summary = {
        "task": 0,
        "name": "llg_snoek_acher_and_hesitivity",
        "quick": bool(quick),
        "elapsed_s": float(elapsed),
        "max_rss_mb": float(rss_mb),
        "literature_models": lit_metrics,
        "acher_sum_rule_low_gamma": acher_low_g,
        "acher_sum_rule_high_gamma": acher_high_g,
        "ade_lorentz_rel_err_disc": ade_lor["rel_err_disc"],
        "ade_debye_rel_err_disc": ade_deb["rel_err_disc"],
    }
    (DATA_DIR / "task_0.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def run_task_1(quick=False):
    """
    Task 1: 3D SolverFIT3D simulation of a Lorentz-mu ferrite slab
    (mu_inf = 1.0, delta_mu = 3.0, f_r = 1.40 GHz, gamma_m/(2*pi) = 0.28 GHz,
    thickness d = 50 mm) across 0.30-2.70 GHz at coarse and fine resolutions
    vs. the exact transfer matrix with mu_discrete(f, dt).
    """
    t0 = time.perf_counter()
    f_r = 1.40e9
    gamma_m = 0.28e9
    mat_lor = DispersiveMaterial(
        eps_inf=1.0,
        mu_inf=1.0,
        mu_poles=[Lorentz(3.0, 2.0 * np.pi * f_r, 2.0 * np.pi * gamma_m)],
        name="lorentz_ferrite_slab",
    )
    freqs = np.linspace(0.30e9, 2.70e9, 37)
    d_slab = 0.050
    lz = 0.60
    dz_coarse = 4.0e-3 if quick else 2.5e-3
    dz_fine = 2.0e-3 if quick else 1.25e-3
    t_after = 12.0e-9 if quick else 16.0e-9

    res_coarse = run_magnetic_slab_pair(
        mat_lor,
        freqs,
        thickness=d_slab,
        lz=lz,
        dz=dz_coarse,
        f0=1.40e9,
        fwidth=0.95e9,
        t_after_src=t_after,
        z_src=-0.16,
        z_r=-0.08,
        z_t=d_slab + 0.06,
    )
    res_fine = run_magnetic_slab_pair(
        mat_lor,
        freqs,
        thickness=d_slab,
        lz=lz,
        dz=dz_fine,
        f0=1.40e9,
        fwidth=0.95e9,
        t_after_src=t_after,
        z_src=-0.16,
        z_r=-0.08,
        z_t=d_slab + 0.06,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_DIR / "task_1.npz",
        freqs=freqs,
        f_r=f_r,
        R_fit_coarse=res_coarse["R_fit"],
        T_fit_coarse=res_coarse["T_fit"],
        R_disc_coarse=res_coarse["R_disc"],
        T_disc_coarse=res_coarse["T_disc"],
        R_fit_fine=res_fine["R_fit"],
        T_fit_fine=res_fine["T_fit"],
        A_fit_fine=res_fine["A_fit"],
        R_disc_fine=res_fine["R_disc"],
        T_disc_fine=res_fine["T_disc"],
        A_disc_fine=res_fine["A_disc"],
        R_cont_fine=res_fine["R_cont"],
        T_cont_fine=res_fine["T_cont"],
        mu_cont=res_fine["mu_cont"],
        mu_disc_fine=res_fine["mu_disc"],
    )

    elapsed = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    summary = {
        "task": 1,
        "name": "lorentz_mu_slab_simulation",
        "quick": bool(quick),
        "elapsed_s": float(elapsed),
        "max_rss_mb": float(rss_mb),
        "thickness_mm": float(d_slab * 1e3),
        "f_r_GHz": float(f_r / 1e9),
        "coarse": {
            "dz_mm": float(dz_coarse * 1e3),
            "nt": res_coarse["nt"],
            "cpw_mat": res_coarse["cpw_mat"],
            "cells_across": res_coarse["cells_across"],
            "rms_R_disc": res_coarse["rms_R_disc"],
            "rms_T_disc": res_coarse["rms_T_disc"],
            "rms_R_cont": res_coarse["rms_R_cont"],
        },
        "fine": {
            "dz_mm": float(dz_fine * 1e3),
            "nt": res_fine["nt"],
            "cpw_mat": res_fine["cpw_mat"],
            "cells_across": res_fine["cells_across"],
            "rms_R_disc": res_fine["rms_R_disc"],
            "rms_T_disc": res_fine["rms_T_disc"],
            "rms_R_cont": res_fine["rms_R_cont"],
        },
    }
    (DATA_DIR / "task_1.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def run_task_2(quick=False):
    """
    Task 2: 3D SolverFIT3D simulation of a Debye-mu relaxation ferrite slab
    (mu_inf = 1.0, delta_mu = 4.0, f_rel = 1.0 GHz, eps_r = 1.5, d = 40 mm)
    across 0.25-2.75 GHz vs. exact transfer matrix with mu_discrete(f, dt).
    """
    t0 = time.perf_counter()
    f_rel = 1.00e9
    tau_m = 1.0 / (2.0 * np.pi * f_rel)
    mat_deb = DispersiveMaterial(
        eps_inf=1.5,
        mu_inf=1.0,
        mu_poles=[Debye(4.0, tau_m)],
        name="debye_ferrite_slab",
    )
    freqs = np.linspace(0.25e9, 2.75e9, 37)
    d_slab = 0.040
    lz = 0.56
    dz = 2.5e-3 if quick else 1.25e-3
    t_after = 11.0e-9 if quick else 15.0e-9

    res = run_magnetic_slab_pair(
        mat_deb,
        freqs,
        thickness=d_slab,
        lz=lz,
        dz=dz,
        f0=1.35e9,
        fwidth=0.95e9,
        t_after_src=t_after,
        z_src=-0.15,
        z_r=-0.075,
        z_t=d_slab + 0.06,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_DIR / "task_2.npz",
        freqs=freqs,
        f_rel=f_rel,
        R_fit=res["R_fit"],
        T_fit=res["T_fit"],
        A_fit=res["A_fit"],
        R_disc=res["R_disc"],
        T_disc=res["T_disc"],
        A_disc=res["A_disc"],
        R_cont=res["R_cont"],
        T_cont=res["T_cont"],
        mu_cont=res["mu_cont"],
        mu_disc=res["mu_disc"],
    )

    elapsed = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    summary = {
        "task": 2,
        "name": "debye_mu_slab_simulation",
        "quick": bool(quick),
        "elapsed_s": float(elapsed),
        "max_rss_mb": float(rss_mb),
        "thickness_mm": float(d_slab * 1e3),
        "dz_mm": float(dz * 1e3),
        "nt": res["nt"],
        "cpw_mat": res["cpw_mat"],
        "cells_across": res["cells_across"],
        "rms_R_disc": res["rms_R_disc"],
        "rms_T_disc": res["rms_T_disc"],
        "rms_R_cont": res["rms_R_cont"],
        "rms_T_cont": res["rms_T_cont"],
    }
    (DATA_DIR / "task_2.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def run_task_3(quick=False):
    """
    Task 3: Impedance-matched magneto-dielectric absorber slab (eps(w) = mu(w))
    where eta(w) = sqrt(mu(w)/eps(w)) = eta_0 at every frequency, giving
    reflectionless absorption R(f) -> 0 across the dispersive resonance band,
    compared side-by-side against an unmatched pure-magnetic slab (eps = 1,
    mu = mu(w)).
    """
    t0 = time.perf_counter()
    f_0 = 1.50e9
    gam = 0.40e9
    pole_shared = Lorentz(2.5, 2.0 * np.pi * f_0, 2.0 * np.pi * gam)
    mat_matched = DispersiveMaterial(
        eps_inf=1.0,
        poles=[pole_shared],
        mu_inf=1.0,
        mu_poles=[pole_shared],
        name="matched_eps_eq_mu",
    )
    mat_unmatched = DispersiveMaterial(
        eps_inf=1.0,
        poles=[],
        mu_inf=1.0,
        mu_poles=[pole_shared],
        name="unmatched_mu_only",
    )

    freqs = np.linspace(0.35e9, 2.75e9, 37)
    d_slab = 0.050
    lz = 0.60
    dz = 2.0e-3 if quick else 1.0e-3
    t_after = 12.0e-9 if quick else 16.0e-9

    res_matched = run_magnetic_slab_pair(
        mat_matched,
        freqs,
        thickness=d_slab,
        lz=lz,
        dz=dz,
        f0=1.50e9,
        fwidth=0.95e9,
        t_after_src=t_after,
        z_src=-0.16,
        z_r=-0.08,
        z_t=d_slab + 0.06,
    )
    res_unmatched = run_magnetic_slab_pair(
        mat_unmatched,
        freqs,
        thickness=d_slab,
        lz=lz,
        dz=dz,
        f0=1.50e9,
        fwidth=0.95e9,
        t_after_src=t_after,
        z_src=-0.16,
        z_r=-0.08,
        z_t=d_slab + 0.06,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_DIR / "task_3.npz",
        freqs=freqs,
        f_0=f_0,
        R_fit_matched=res_matched["R_fit"],
        T_fit_matched=res_matched["T_fit"],
        A_fit_matched=res_matched["A_fit"],
        R_cont_matched=res_matched["R_cont"],
        T_cont_matched=res_matched["T_cont"],
        A_cont_matched=res_matched["A_cont"],
        R_fit_unmatched=res_unmatched["R_fit"],
        T_fit_unmatched=res_unmatched["T_fit"],
        A_fit_unmatched=res_unmatched["A_fit"],
        R_cont_unmatched=res_unmatched["R_cont"],
        T_cont_unmatched=res_unmatched["T_cont"],
        A_cont_unmatched=res_unmatched["A_cont"],
    )

    elapsed = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    summary = {
        "task": 3,
        "name": "impedance_matched_magneto_dielectric_slab",
        "quick": bool(quick),
        "elapsed_s": float(elapsed),
        "max_rss_mb": float(rss_mb),
        "thickness_mm": float(d_slab * 1e3),
        "dz_mm": float(dz * 1e3),
        "cpw_mat": res_matched["cpw_mat"],
        "cells_across": res_matched["cells_across"],
        "max_R_cont_matched": float(np.max(res_matched["R_cont"])),
        "max_R_fit_matched": float(np.max(res_matched["R_fit"])),
        "rms_R_fit_matched": float(
            np.sqrt(np.mean(res_matched["R_fit"] ** 2))
        ),
        "rms_T_disc_matched": res_matched["rms_T_disc"],
        "peak_A_fit_matched": float(np.max(res_matched["A_fit"])),
        "max_R_fit_unmatched": float(np.max(res_unmatched["R_fit"])),
    }
    (DATA_DIR / "task_3.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def run_task_4(quick=False):
    """
    Task 4: 3D PEC rectangular cavity (Lx = Lz = 120 mm, Ly = 60 mm; bare
    TE_101 mode at f_vac = 1.7665 GHz) loaded with dispersive Lorentz-mu(w)
    ferrite (mu_inf = 1.0, delta_mu = 1.25, mu_s = 2.25) as f_r is swept across
    the bare cavity resonance, demonstrating mode shifting and polariton
    branch splitting (omega^2 * mu(omega) = omega_vac^2).
    """
    t0 = time.perf_counter()
    lx = lz = 0.12
    ly = 0.06
    nx = nz = 12 if quick else 16
    ny = 6 if quick else 8
    dx = lx / nx
    dz = lz / nz

    # Discrete Yee-grid vacuum TE_101 frequency
    kx_d = (2.0 / dx) * np.sin(np.pi * dx / (2.0 * lx))
    kz_d = (2.0 / dz) * np.sin(np.pi * dz / (2.0 * lz))
    f_vac_cont = float(
        (c0 / (2.0 * np.pi)) * np.sqrt((np.pi / lx) ** 2 + (np.pi / lz) ** 2)
    )
    f_vac_grid = float((c0 / (2.0 * np.pi)) * np.sqrt(kx_d**2 + kz_d**2))

    mu_inf = 1.0
    delta_mu = 1.25
    mu_s = mu_inf + delta_mu
    f_static_grid = float(f_vac_grid / np.sqrt(mu_s))

    nt = 1800 if quick else 2600
    f_spec = np.linspace(0.50e9, 4.30e9, 3801)

    def _run_cavity_spectrum(mat=None):
        grid = GridFIT3D(0.0, lx, 0.0, ly, 0.0, lz, nx, ny, nz, verbose=0)
        solver = SolverFIT3D(
            grid,
            bc_low=["pec", "pec", "pec"],
            bc_high=["pec", "pec", "pec"],
            use_stl=False,
            cfln=0.5,
            verbose=0,
        )
        if mat is not None:
            solver.add_dispersive_material(
                mat,
                center=(0.5 * lx, 0.5 * ly, 0.5 * lz),
                size=(2.0 * lx, 2.0 * ly, 2.0 * lz),
            )
        dt = float(solver.dt)
        t_arr = np.arange(nt) * dt
        f0 = 1.80e9
        bw = 1.45e9
        sig = 1.0 / (2.0 * np.pi * bw)
        t_pk = 4.0 * sig
        src = np.exp(-0.5 * ((t_arr - t_pk) / sig) ** 2) * np.sin(
            2.0 * np.pi * f0 * (t_arr - t_pk)
        )
        # Spatial eigenmode profile sin(pi*x/Lx)*sin(pi*z/Lz) of TE_101
        # on primary Ey edges (which lie at nodes grid.x[:solver.Nx], grid.z[:solver.Nz])
        x_nodes = grid.x[: solver.Nx]
        z_nodes = grid.z[: solver.Nz]
        mode_xz = (
            np.sin(np.pi * x_nodes[:, None] / lx)
            * np.sin(np.pi * z_nodes[None, :] / lz)
        )
        mode_3d = np.broadcast_to(
            mode_xz[:, None, :], (solver.Nx, solver.Ny, solver.Nz)
        ).copy()
        mode_norm = float(np.sum(mode_3d * mode_3d))

        rec = np.zeros(nt, dtype=np.float64)
        for n in range(nt):
            solver.J[:, :, :, "y"] = src[n] * mode_3d
            solver.one_step()
            ey_arr = solver.E[:, :, :, "y"]
            rec[n] = float(np.sum(ey_arr * mode_3d) / mode_norm)
        # Only transform after the Gaussian source pulse has turned off (t > 2*t_pk)
        n_start = int(round(2.2 * t_pk / dt))
        t_tail = t_arr[n_start:]
        rec_tail = rec[n_start:]
        win = np.hanning(rec_tail.size)
        spec = np.abs(
            np.sum(
                (rec_tail * win)[:, None]
                * np.exp(-2j * np.pi * t_tail[:, None] * f_spec[None, :]),
                axis=0,
            )
        )
        spec = spec / np.max(spec)
        return spec

    spec_vac = _run_cavity_spectrum(None)
    f_vac_fit = float(f_spec[np.argmax(spec_vac)])

    mat_static = DispersiveMaterial(
        eps_inf=1.0, mu_inf=mu_s, mu_poles=[], name="static_mu_cavity"
    )
    spec_static = _run_cavity_spectrum(mat_static)
    f_static_fit = float(f_spec[np.argmax(spec_static)])

    f_r_sweep = np.array([1.20e9, 1.50e9, 1.76e9, 2.10e9, 2.50e9])
    gamma_cav = 0.025e9
    spec_sweep = np.zeros((f_r_sweep.size, f_spec.size), dtype=np.float64)
    f_minus_fit = np.zeros(f_r_sweep.size, dtype=np.float64)
    f_plus_fit = np.zeros(f_r_sweep.size, dtype=np.float64)
    f_minus_an, f_plus_an = analytic_cavity_polariton_branches(
        f_vac_grid, mu_inf, delta_mu, f_r_sweep
    )

    sweep_records = []
    for idx, fr_val in enumerate(f_r_sweep):
        mat_c = DispersiveMaterial(
            eps_inf=1.0,
            mu_inf=mu_inf,
            mu_poles=[
                Lorentz(
                    delta_mu, 2.0 * np.pi * fr_val, 2.0 * np.pi * gamma_cav
                )
            ],
            name=f"cav_fr_{fr_val/1e9:.2f}GHz",
        )
        sp = _run_cavity_spectrum(mat_c)
        spec_sweep[idx] = sp
        mask_lo = f_spec < fr_val
        mask_hi = f_spec > fr_val * np.sqrt(mu_s / mu_inf)
        fm_fit = float(f_spec[mask_lo][np.argmax(sp[mask_lo])])
        fp_fit = float(f_spec[mask_hi][np.argmax(sp[mask_hi])])
        f_minus_fit[idx] = fm_fit
        f_plus_fit[idx] = fp_fit
        sweep_records.append(
            {
                "f_r_GHz": float(fr_val / 1e9),
                "f_minus_analytic_GHz": float(f_minus_an[idx] / 1e9),
                "f_minus_fit_GHz": float(fm_fit / 1e9),
                "f_minus_rel_err": float(
                    abs(fm_fit - f_minus_an[idx]) / f_minus_an[idx]
                ),
                "f_plus_analytic_GHz": float(f_plus_an[idx] / 1e9),
                "f_plus_fit_GHz": float(fp_fit / 1e9),
                "f_plus_rel_err": float(
                    abs(fp_fit - f_plus_an[idx]) / f_plus_an[idx]
                ),
            }
        )

    f_r_dense = np.linspace(0.80e9, 3.00e9, 200)
    f_minus_dense, f_plus_dense = analytic_cavity_polariton_branches(
        f_vac_grid, mu_inf, delta_mu, f_r_dense
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_DIR / "task_4.npz",
        f_spec=f_spec,
        spec_vac=spec_vac,
        spec_static=spec_static,
        f_r_sweep=f_r_sweep,
        spec_sweep=spec_sweep,
        f_minus_fit=f_minus_fit,
        f_plus_fit=f_plus_fit,
        f_minus_an=f_minus_an,
        f_plus_an=f_plus_an,
        f_r_dense=f_r_dense,
        f_minus_dense=f_minus_dense,
        f_plus_dense=f_plus_dense,
        f_vac_grid=f_vac_grid,
        f_static_grid=f_static_grid,
    )

    elapsed = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    summary = {
        "task": 4,
        "name": "dispersive_mu_cavity_mode_splitting",
        "quick": bool(quick),
        "elapsed_s": float(elapsed),
        "max_rss_mb": float(rss_mb),
        "cavity_dims_mm": [float(lx * 1e3), float(ly * 1e3), float(lz * 1e3)],
        "grid_cells": [int(nx), int(ny), int(nz)],
        "f_vac_continuum_GHz": float(f_vac_cont / 1e9),
        "f_vac_grid_GHz": float(f_vac_grid / 1e9),
        "f_vac_fit_GHz": float(f_vac_fit / 1e9),
        "f_static_grid_GHz": float(f_static_grid / 1e9),
        "f_static_fit_GHz": float(f_static_fit / 1e9),
        "polariton_sweep": sweep_records,
    }
    (DATA_DIR / "task_4.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary



# %%
def _fix_schematic_units(ax):
    """Ensure a thin space and upright font separate numbers and units inside math-mode labels."""
    import re

    def _fix_math_block(match):
        body = match.group(1)
        body = re.sub(
            r"(\d)\s*(?:µm|\\mu\s*m)\b",
            r"\1\\;\\mu\\mathrm{m}",
            body,
        )
        body = re.sub(
            r"(\d)\s*(mm|cm|nm|m|GHz|MHz|THz|kHz|Hz|ps|ns|fs)\b",
            r"\1\\;\\mathrm{\2}",
            body,
        )
        return f"${body}$"

    for txt in ax.texts:
        s = txt.get_text()
        s_new = re.sub(r"\$([^$]+)\$", _fix_math_block, s)
        if s_new != s:
            txt.set_text(s_new)
        if s.startswith("Mesh:"):
            x0, y0 = txt.get_position()
            txt.set_position((x0, y0 + 0.065))


def _plot_setup_schematic(fig_dir):
    """Render ``00_setup_schematic.png`` for the three Tutorial 2.3 simulation setups."""
    fig, (ax_ade, ax_slab, ax_cav) = figstyle.new_figure(
        nrows=1, ncols=3, width=15.6, height=4.8
    )

    # Panel (a): Periodic H-edge magnetic ADE verification (Task 0)
    dt_ade = 1.2e-11
    nz_ade = 16
    dz_ade = c0 * dt_ade / 0.25
    grid_ade = GridFIT3D(
        -dz_ade, dz_ade, -dz_ade, dz_ade, 0.0, nz_ade * dz_ade, 2, 2, nz_ade, verbose=0
    )
    solver_ade = SolverFIT3D(
        grid_ade,
        cfln=0.25,
        bc_low=["periodic", "periodic", "periodic"],
        bc_high=["periodic", "periodic", "periodic"],
        use_stl=False,
        verbose=0,
    )
    mat_ade = DispersiveMaterial(
        mu_inf=1.2,
        mu_poles=[Lorentz(2.8, 2.0 * np.pi * 1.4e9, 2.0 * np.pi * 0.28e9)],
        name="Lorentz / Debye mu(w)",
    )
    struct_ade = SimpleNamespace(
        bounds=((-dz_ade, -dz_ade, 0.0), (dz_ade, dz_ade, nz_ade * dz_ade)),
        name="Bulk magnetic cell (Lorentz / Debye mu)",
        material=mat_ade,
    )
    src_ade = CurrentSource(
        GaussianSource(frequency=1.50e9, fwidth=1.00e9),
        component="x",
        center=(0.0, 0.0, 4.0 * dz_ade),
        size=(2.0 * dz_ade, 2.0 * dz_ade, 0.0),
    )
    schematic.plot_schematic(
        solver_ade,
        plane="zx",
        ax=ax_ade,
        structures=[struct_ade],
        sources=[src_ade],
        probes=[
            {
                "pos": (0.0, 0.0, 12.0 * dz_ade),
                "label": r"Probe ($H_y, M_y(t), \nabla\times E$)",
            }
        ],
        show_mesh=True,
        mesh_every=2,
        mesh_zoom=False,
        freq=1.40e9,
        title="(a) Periodic H-edge ADE check (Task 0)",
    )
    ax_ade.set_yticks([])
    ax_ade.set_ylabel("Transverse x (periodic, 2 cells)")
    _fix_schematic_units(ax_ade)

    # Panel (b): Quasi-1D magnetic & impedance-matched (eps=mu) slab R/T setup (Tasks 1-3)
    dz_slab = 2.5e-3
    lz_slab = 0.60
    d_slab = 0.050
    nz_slab = int(round(lz_slab / dz_slab))
    zmin_slab = -0.45 * lz_slab
    zmax_slab = zmin_slab + nz_slab * dz_slab
    dxy_slab = 30.0e-3
    grid_slab = GridFIT3D(
        -dxy_slab, dxy_slab, -dxy_slab, dxy_slab, zmin_slab, zmax_slab, 2, 2, nz_slab, verbose=0
    )
    solver_slab = SolverFIT3D(
        grid_slab,
        cfln=0.5,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=15,
        use_stl=False,
        verbose=0,
    )
    mat_slab = DispersiveMaterial(
        eps_inf=1.0,
        mu_inf=1.0,
        mu_poles=[Lorentz(3.0, 2.0 * np.pi * 1.4e9, 2.0 * np.pi * 0.28e9)],
        name="Ferrite / Matched slab",
    )
    struct_slab = SimpleNamespace(
        bounds=((-dxy_slab, -dxy_slab, 0.0), (dxy_slab, dxy_slab, d_slab)),
        name=r"Ferrite or $\varepsilon=\mu$ slab",
        material=mat_slab,
    )
    src_slab = CurrentSource(
        GaussianSource(frequency=1.40e9, fwidth=0.95e9),
        component="x",
        center=(0.0, 0.0, -0.16),
        size=(2.0 * dxy_slab, 2.0 * dxy_slab, 0.0),
    )
    planes_slab = [
        FluxMonitor(
            np.linspace(0.3e9, 2.7e9, 5),
            center=(0.0, 0.0, z0),
            size=(2.0 * dxy_slab, 2.0 * dxy_slab, 0.0),
        )
        for z0 in (-0.08, d_slab + 0.06)
    ]
    schematic.plot_schematic(
        solver_slab,
        plane="zx",
        ax=ax_slab,
        structures=[struct_slab],
        sources=[src_slab],
        monitors=planes_slab,
        show_mesh=True,
        mesh_every=8,
        mesh_zoom=True,
        freq=1.40e9,
        title="(b) Magnetic / matched slab R/T (Tasks 1–3)",
    )
    ax_slab.set_yticks([])
    ax_slab.set_ylabel("Transverse x (periodic, 2 cells)")
    _fix_schematic_units(ax_slab)

    # Panel (c): 3D PEC ferrite-loaded rectangular cavity (Task 4)
    lx_cav = lz_cav = 0.12
    ly_cav = 0.06
    nx_cav = nz_cav = 16
    ny_cav = 8
    grid_cav = GridFIT3D(
        0.0, lx_cav, 0.0, ly_cav, 0.0, lz_cav, nx_cav, ny_cav, nz_cav, verbose=0
    )
    solver_cav = SolverFIT3D(
        grid_cav,
        bc_low=["pec", "pec", "pec"],
        bc_high=["pec", "pec", "pec"],
        use_stl=False,
        cfln=0.5,
        verbose=0,
    )
    mat_cav = DispersiveMaterial(
        eps_inf=1.0,
        mu_inf=1.0,
        mu_poles=[Lorentz(1.25, 2.0 * np.pi * 1.76e9, 2.0 * np.pi * 0.025e9)],
        name="Lorentz-mu ferrite (mu_s=2.25)",
    )
    struct_cav = SimpleNamespace(
        bounds=((0.0, 0.0, 0.0), (lx_cav, ly_cav, lz_cav)),
        name=r"Lorentz-$\mu$ ferrite ($\mu_s = 2.25$)",
        material=mat_cav,
    )
    src_cav = CurrentSource(
        GaussianSource(frequency=1.80e9, fwidth=1.45e9),
        component="y",
        center=(0.5 * lx_cav, 0.5 * ly_cav, 0.5 * lz_cav),
        size=(0.0, 0.0, 0.0),
    )
    schematic.plot_schematic(
        solver_cav,
        plane="zx",
        ax=ax_cav,
        structures=[struct_cav],
        sources=[src_cav],
        probes=[
            {
                "pos": (0.04, 0.03, 0.04),
                "label": r"$\mathrm{TE}_{101}$ probe ($E_y(t)$)",
            }
        ],
        show_mesh=True,
        mesh_every=2,
        mesh_zoom=False,
        freq=1.766e9,
        title="(c) 3D PEC ferrite cavity (Task 4)",
    )
    ax_cav.set_yticks([0, 20, 40, 60, 80, 100, 120])
    _fix_schematic_units(ax_cav)

    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "00_setup_schematic.png", close=True)


def _plot_wikimedia_fmr_diagram(fig_dir):
    """
    Render ``figures/wikimedia/fmr_precession_and_permeability_wakis.png``:
    Panel (a): Landau-Lifshitz-Gilbert (LLG) damped spin precession of M(t)
               around H_eff (after Wikimedia Commons File:Precession_of_magnetization.svg
               by Geek3, CC BY-SA 3.0, and File:Larmor_Precession.svg by Lookang, CC BY-SA 3.0).
    Panel (b): Broadband complex magnetic permeability mu'(f), mu''(f) from 0.5 MHz
               to 30 GHz computed from Wakis DispersiveMaterial(mu_poles=[Debye, Lorentz]),
               connecting domain-wall Debye relaxation, natural FMR Lorentz resonance,
               the negative-mu' stopband, and Snoek's law to Wakis's mu_poles API.
    """
    wiki_dir = fig_dir / "wikimedia"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_llg, ax_mu) = figstyle.new_figure(
        nrows=1, ncols=2, width=12.8, height=4.8
    )

    # ------------------------------------------------------------------
    # Panel (a): LLG damped spin precession schematic
    # ------------------------------------------------------------------
    def _proj(x3, y3, z3):
        return x3 - 0.30 * y3, z3 - 0.20 * y3

    # Exact spherical LLG damped trajectory on unit sphere |M| = M_s
    alpha_g = 0.065
    theta_0 = np.deg2rad(38.0)
    t_norm = np.linspace(0.0, 4.2 * (2.0 * np.pi), 900)
    theta_t = 2.0 * np.arctan(
        np.tan(0.5 * theta_0) * np.exp(-alpha_g * t_norm / (1.0 + alpha_g**2))
    )
    phi_t = t_norm / (1.0 + alpha_g**2)
    mx = np.sin(theta_t) * np.cos(phi_t)
    my = np.sin(theta_t) * np.sin(phi_t)
    mz = np.cos(theta_t)

    # Outer undamped cone rim at theta = theta_0
    phi_c = np.linspace(0.0, 2.0 * np.pi, 200)
    cx, cz = _proj(
        np.sin(theta_0) * np.cos(phi_c),
        np.sin(theta_0) * np.sin(phi_c),
        np.full_like(phi_c, np.cos(theta_0)),
    )
    ax_llg.plot(
        cx,
        cz,
        "--",
        color="#8a9ba8",
        lw=1.1,
        label=r"Undamped Larmor cone ($\alpha_{\mathrm{G}} = 0$)",
    )

    # Spiral damped trajectory M(t)
    sx, sz = _proj(mx, my, mz)
    ax_llg.plot(
        sx,
        sz,
        "-",
        color="#d55e00",
        lw=1.9,
        label=r"Damped LLG trajectory $\mathbf{M}(t)$",
    )

    # Vertical H_eff axis
    hx, hz = _proj(0.0, 0.0, 1.18)
    ax_llg.annotate(
        "",
        xy=(hx, hz),
        xytext=(0.0, -0.05),
        arrowprops=dict(arrowstyle="->", color="#009e73", lw=2.3),
    )
    ax_llg.text(
        hx + 0.03,
        hz - 0.02,
        r"$\mathbf{H}_{\mathrm{eff}} = H_a\hat{\mathbf{u}} + \mathbf{H}_0$",
        fontsize=8.8,
        fontweight="bold",
        color="#009e73",
    )

    # Instantaneous magnetization vector M(t) at a prominent front point
    idx_p = 68
    px, py, pz = mx[idx_p], my[idx_p], mz[idx_p]
    m_tip_x, m_tip_z = _proj(px, py, pz)
    ax_llg.annotate(
        "",
        xy=(m_tip_x, m_tip_z),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color="#0072b2", lw=2.6),
    )
    ax_llg.text(
        0.52 * m_tip_x + 0.04,
        0.52 * m_tip_z - 0.06,
        r"$\mathbf{M}(t)$ ($|\mathbf{M}| = M_s$)",
        fontsize=8.6,
        fontweight="bold",
        color="#0072b2",
    )

    # Larmor precessional torque vector -gamma_LL * mu_0 * (M x H_eff) (tangent to circle)
    t_larmor_3d = np.array([-py, px, 0.0])
    t_larmor_3d /= np.linalg.norm(t_larmor_3d)
    lx_tip, lz_tip = _proj(
        px + 0.34 * t_larmor_3d[0],
        py + 0.34 * t_larmor_3d[1],
        pz + 0.34 * t_larmor_3d[2],
    )
    ax_llg.annotate(
        "",
        xy=(lx_tip, lz_tip),
        xytext=(m_tip_x, m_tip_z),
        arrowprops=dict(arrowstyle="->", color="#cc79a7", lw=2.1),
    )
    ax_llg.text(
        lx_tip + 0.02,
        lz_tip - 0.01,
        r"$-\gamma_{\mathrm{LL}}\mu_0(\mathbf{M}\times\mathbf{H}_{\mathrm{eff}})$"
        "\n(Larmor precession)",
        fontsize=7.8,
        color="#8e3b68",
    )

    # Gilbert damping torque vector +(alpha_G / M_s) * (M x dM/dt) (points inward toward H_eff)
    t_damp_3d = np.array(
        [-np.cos(theta_t[idx_p]) * np.cos(phi_t[idx_p]),
         -np.cos(theta_t[idx_p]) * np.sin(phi_t[idx_p]),
         np.sin(theta_t[idx_p])]
    )
    dx_tip, dz_tip = _proj(
        px + 0.25 * t_damp_3d[0],
        py + 0.25 * t_damp_3d[1],
        pz + 0.25 * t_damp_3d[2],
    )
    ax_llg.annotate(
        "",
        xy=(dx_tip, dz_tip),
        xytext=(m_tip_x, m_tip_z),
        arrowprops=dict(arrowstyle="->", color="#2b8cbe", lw=2.0),
    )
    ax_llg.text(
        dx_tip - 0.36,
        dz_tip + 0.08,
        r"$+\frac{\alpha_{\mathrm{G}}}{M_s}\left(\mathbf{M}\times\frac{d\mathbf{M}}{dt}\right)$"
        "\n(Gilbert damping)",
        fontsize=7.8,
        color="#1d5f82",
    )

    # Transverse RF drive field h_rf * exp(+i*omega*t)
    ax_llg.annotate(
        "",
        xy=(0.36, 0.0),
        xytext=(-0.36, 0.0),
        arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.6, ls="--"),
    )
    ax_llg.text(
        0.16,
        -0.11,
        r"RF drive $\mathbf{h}_{\mathrm{rf}}\,e^{+i\omega t} \perp \mathbf{H}_{\mathrm{eff}}$",
        fontsize=7.9,
        color="#333333",
    )

    # Boxed LLG & natural FMR formula at bottom
    ax_llg.text(
        0.0,
        -0.31,
        r"$\frac{d\mathbf{M}}{dt} = -\gamma_{\mathrm{LL}}\mu_0(\mathbf{M}\times\mathbf{H}_{\mathrm{eff}})"
        r" + \frac{\alpha_{\mathrm{G}}}{M_s}\left(\mathbf{M}\times\frac{d\mathbf{M}}{dt}\right)$"
        "\n"
        r"$\omega_r = 2\pi f_r = \gamma_{\mathrm{LL}}\mu_0 H_a,\quad \gamma_m \approx 2\alpha_{\mathrm{G}}\omega_r$",
        ha="center",
        va="center",
        fontsize=7.9,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#f8f9fb",
            edgecolor="#cbd5e1",
        ),
    )
    ax_llg.set_xlim(-0.78, 0.88)
    ax_llg.set_ylim(-0.46, 1.25)
    ax_llg.set_xticks([])
    ax_llg.set_yticks([])
    ax_llg.set_xlabel("Transverse magnetization plane $(M_x, M_y)$")
    ax_llg.set_ylabel(r"Easy-axis alignment $M_z / M_s$")
    ax_llg.set_title("(a) LLG damped spin precession around $\\mathbf{H}_{\\mathrm{eff}}$")
    ax_llg.legend(loc="lower left", bbox_to_anchor=(0.01, 0.19), fontsize=7.6)

    # ------------------------------------------------------------------
    # Panel (b): Broadband ferrite permeability mu'(f), mu''(f) & Wakis mu_poles
    # ------------------------------------------------------------------
    f_hz = np.geomspace(0.3e6, 30.0e9, 900)
    f_ghz = f_hz / 1e9
    f_rel_dw = 8.0e6
    f_r_fmr = 1.50e9
    gamma_fmr = 0.30e9
    mat_comp = DispersiveMaterial(
        mu_inf=1.0,
        mu_poles=[
            Debye(120.0, 1.0 / (2.0 * np.pi * f_rel_dw)),
            Lorentz(14.0, 2.0 * np.pi * f_r_fmr, 2.0 * np.pi * gamma_fmr),
        ],
        name="composite_ferrite",
    )
    mu_tot = mat_comp.mu(f_hz)
    mu_re = np.real(mu_tot)
    mu_im = -np.imag(mu_tot)

    # Shaded mechanism bands
    ax_mu.axvspan(0.3e-3, 0.08, color="#dce8f2", alpha=0.35)
    ax_mu.axvspan(0.08, 30.0, color="#fde6d2", alpha=0.28)
    f_stop_hi = f_r_fmr * np.sqrt(15.0 / 1.0) / 1e9
    ax_mu.axvspan(
        f_r_fmr / 1e9,
        f_stop_hi,
        color="#f8d7da",
        alpha=0.55,
        label=r"Stopband $\mu'(f) < 0$",
    )

    ax_mu.plot(
        f_ghz,
        mu_re,
        label=r"$\mu'(f) = \mathrm{Re}\,\mu(f)$",
        **figstyle.series_style(0, markevery=110),
    )
    ax_mu.plot(
        f_ghz,
        mu_im,
        label=r"$\mu''(f) = -\mathrm{Im}\,\mu(f) \geq 0$",
        **figstyle.series_style(1, markevery=110),
    )
    ax_mu.axhline(0.0, color="#555555", ls=":", lw=1.0)
    ax_mu.axvline(f_rel_dw / 1e9, color="#0072b2", ls="--", lw=1.0, alpha=0.8)
    ax_mu.axvline(f_r_fmr / 1e9, color="#d55e00", ls="--", lw=1.0, alpha=0.8)

    ax_mu.set_xscale("log")
    ax_mu.set_yscale("symlog", linthresh=2.0)
    ax_mu.set_xlim(0.3e-3, 30.0)
    ax_mu.set_ylim(-25.0, 950.0)
    ax_mu.set_xlabel("Frequency $f$ [GHz]")
    ax_mu.set_ylabel(r"Relative permeability $\mu'(f),\; \mu''(f)$")
    ax_mu.set_title(r"(b) Broadband $\mu(f)$ & Wakis `mu_poles` mechanisms")

    ax_mu.text(
        2.5e-3,
        290.0,
        "Domain-wall relaxation\n`Debye(delta_mu, tau_m)`\n$f_{\\mathrm{rel}} = 1/(2\\pi\\tau_m)$",
        fontsize=7.6,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88, edgecolor="#9ecae1"),
    )
    ax_mu.text(
        0.55,
        95.0,
        "Spin-rotation FMR\n`Lorentz(delta_mu, w0, gamma_m)`\n"
        r"Snoek: $(\mu_{s,\mathrm{rot}}-1)f_r \propto M_s$",
        fontsize=7.6,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88, edgecolor="#fdae6b"),
    )
    ax_mu.annotate(
        r"$f_r = \frac{\gamma_{\mathrm{LL}}\mu_0 H_a}{2\pi}$",
        xy=(f_r_fmr / 1e9, 24.0),
        xytext=(5.5, 42.0),
        fontsize=7.8,
        arrowprops=dict(arrowstyle="->", color="#d55e00", lw=1.1),
    )
    ax_mu.legend(loc="lower left", fontsize=7.6)

    figstyle.finish(fig)
    figstyle.save(
        fig,
        wiki_dir / "fmr_precession_and_permeability_wakis.png",
        close=True,
    )


def plot_all():
    """Generate setup schematic, Wikimedia FMR diagram, and all 8 tutorial figures."""
    figstyle.apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _plot_setup_schematic(FIG_DIR)
    _plot_wikimedia_fmr_diagram(FIG_DIR)

    d0 = np.load(DATA_DIR / "task_0.npz")
    j0 = json.loads((DATA_DIR / "task_0.json").read_text(encoding="utf-8"))
    d1 = np.load(DATA_DIR / "task_1.npz")
    j1 = json.loads((DATA_DIR / "task_1.json").read_text(encoding="utf-8"))
    d2 = np.load(DATA_DIR / "task_2.npz")
    j2 = json.loads((DATA_DIR / "task_2.json").read_text(encoding="utf-8"))
    d3 = np.load(DATA_DIR / "task_3.npz")
    j3 = json.loads((DATA_DIR / "task_3.json").read_text(encoding="utf-8"))
    d4 = np.load(DATA_DIR / "task_4.npz")
    j4 = json.loads((DATA_DIR / "task_4.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Figure 1: Verified literature magnetic materials (Lorentz & Debye)
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=14.6, height=4.5)
    f_b_ghz = d0["f_broad"] / 1e9

    axes[0].semilogx(
        f_b_ghz,
        np.real(d0["mu_czn"]),
        label=r"CoZrNb Eq. 8-81 ($\mu'\!$)",
        **figstyle.series_style(0, markevery=60),
    )
    axes[0].semilogx(
        f_b_ghz,
        -np.imag(d0["mu_czn"]),
        label=r"CoZrNb Eq. 8-81 ($\mu''\!$)",
        **figstyle.series_style(1, markevery=60),
    )
    axes[0].semilogx(
        f_b_ghz,
        np.real(d0["mu_co2z"]),
        label=r"$\mathrm{Co}_2\mathrm{Z}$ unaligned ($\mu'\!$)",
        **figstyle.series_style(2, markevery=60),
    )
    axes[0].semilogx(
        f_b_ghz,
        -np.imag(d0["mu_co2z"]),
        label=r"$\mathrm{Co}_2\mathrm{Z}$ unaligned ($\mu''\!$)",
        **figstyle.series_style(3, markevery=60),
    )
    axes[0].semilogx(
        f_b_ghz,
        np.real(d0["mu_ikonen"]),
        label=r"Ikonen film ($\mu_s=1.48$)",
        **figstyle.series_style(4, markevery=60),
    )
    axes[0].axhline(0.0, color="gray", ls=":", lw=0.9)
    axes[0].set_xlabel("Frequency $f$ [GHz]")
    axes[0].set_ylabel(r"Relative permeability $\mu'(f),\; \mu''(f)$")
    axes[0].set_title(r"(a) Verified Lorentz FMR models ($\mathrm{CoZrNb}, \mathrm{Co}_2\mathrm{Z}$)")
    axes[0].legend(fontsize=7.4)

    axes[1].loglog(
        d0["f_broad"] / 1e6,
        np.maximum(np.real(d0["mu_nizn"]), 0.5),
        label=r"NiZn tile $\mu'(f)$ ($\mu_s=2200$)",
        **figstyle.series_style(0, markevery=60),
    )
    axes[1].loglog(
        d0["f_broad"] / 1e6,
        -np.imag(d0["mu_nizn"]),
        label=r"NiZn tile $\mu''(f)$ ($f_{\mathrm{rel}}=3\;\mathrm{MHz}$)",
        **figstyle.series_style(1, markevery=60),
    )
    axes[1].axvline(3.0, color="gray", ls=":", lw=1.0, label=r"$f_{\mathrm{rel}} = 3\;\mathrm{MHz}$")
    axes[1].set_xlabel("Frequency $f$ [MHz]")
    axes[1].set_ylabel(r"Relative permeability $\mu'(f),\; \mu''(f)$")
    axes[1].set_title("(b) Sintered NiZn spinel (Debye domain-wall model)")
    axes[1].legend(fontsize=7.8)

    # Panel (c): Periodic FIT magnetic ADE verification vs mu_discrete(f, dt)
    f_ade_g = d0["f_ade"] / 1e9
    axes[2].plot(
        f_ade_g,
        np.real(d0["mu_disc_lor"]),
        label=r"Lorentz $\mathrm{Re}\,\mu_d(f,\Delta t)$",
        **figstyle.series_style(0, with_marker=False),
    )
    axes[2].plot(
        f_ade_g[::2],
        np.real(d0["mu_fit_lor"][::2]),
        "o",
        color=figstyle.PALETTE[0],
        ms=4.5,
        label="Lorentz FIT ADE",
    )
    axes[2].plot(
        f_ade_g,
        -np.imag(d0["mu_disc_lor"]),
        label=r"Lorentz $-\mathrm{Im}\,\mu_d(f,\Delta t)$",
        **figstyle.series_style(1, with_marker=False),
    )
    axes[2].plot(
        f_ade_g[::2],
        -np.imag(d0["mu_fit_lor"][::2]),
        "s",
        color=figstyle.PALETTE[1],
        ms=4.5,
        label=r"Lorentz FIT $-\mathrm{Im}\,\mu$",
    )
    axes[2].plot(
        f_ade_g,
        -np.imag(d0["mu_disc_deb"]),
        label=r"Debye $-\mathrm{Im}\,\mu_d(f,\Delta t)$",
        **figstyle.series_style(2, with_marker=False),
    )
    axes[2].plot(
        f_ade_g[::2],
        -np.imag(d0["mu_fit_deb"][::2]),
        "^",
        color=figstyle.PALETTE[2],
        ms=4.5,
        label="Debye FIT ADE",
    )
    axes[2].set_xlabel("Frequency $f$ [GHz]")
    axes[2].set_ylabel("Relative permeability")
    axes[2].set_title(
        f"(c) FIT H-edge ADE vs $\\mu_d$: err={j0['ade_lorentz_rel_err_disc']:.2e}"
    )
    axes[2].legend(fontsize=7.3, ncol=2)

    figstyle.finish(fig)
    figstyle.save(fig, FIG_DIR / "01_literature_ferrites_and_ade_check.png", close=True)

    # ------------------------------------------------------------------
    # Figure 2: Snoek's law (1948) vs Acher's planar-anisotropy bound
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=12.6, height=4.4)
    f_l_ghz = d0["f_lin"] / 1e9
    cols_s = list(figstyle.PALETTE[:4])
    for idx, (ms, fr_g, col) in enumerate(
        zip(d0["snoek_mu_s"], d0["snoek_f_r_ghz"], cols_s)
    ):
        axes[0].plot(
            f_l_ghz,
            np.real(d0["snoek_mu_curves"][idx]),
            "-",
            color=col,
            lw=1.9,
            label=f"$\\mu_s={ms:.1f},\\; f_r={fr_g:.2f}\\;\\mathrm{{GHz}}$ ($\\mu'$)",
        )
        axes[0].plot(
            f_l_ghz,
            -np.imag(d0["snoek_mu_curves"][idx]),
            "--",
            color=col,
            lw=1.4,
        )
        axes[0].axvline(fr_g, color=col, ls=":", lw=0.9, alpha=0.8)
    axes[0].set_xlabel("Frequency $f$ [GHz]")
    axes[0].set_ylabel(r"$\mu'(f)$ (solid), $\mu''(f)$ (dashed)")
    axes[0].set_title(r"(a) Snoek's family: $(\mu_s - 1)\,f_r = 3.9\;\mathrm{GHz}$")
    axes[0].legend(fontsize=7.8)

    # Panel (b): (mu_s - 1) vs f_r log-log plot comparing Cubic Spinel Snoek limit vs Planar Hexaferrite / CZN
    fr_grid = np.geomspace(0.05, 10.0, 200)
    chi_snoek = 3.9 / fr_grid
    chi_acher_hex = 14.0 / (fr_grid**2)
    chi_acher_czn = 88.0 / (fr_grid**2)
    axes[1].loglog(
        fr_grid,
        chi_snoek,
        "k-",
        lw=2.0,
        label=r"Cubic Snoek limit: $(\mu_s - 1)f_r = 3.9\;\mathrm{GHz}$",
    )
    axes[1].loglog(
        fr_grid,
        chi_acher_hex,
        "--",
        color=figstyle.PALETTE[0],
        lw=1.8,
        label=r"Planar hexaferrite Acher bound ($\propto f_r^{-2}$)",
    )
    axes[1].loglog(
        fr_grid,
        chi_acher_czn,
        "-.",
        color=figstyle.PALETTE[1],
        lw=1.8,
        label=r"Thin-film laminate Acher bound ($\propto f_r^{-2}$)",
    )
    axes[1].loglog(
        d0["snoek_f_r_ghz"],
        d0["snoek_mu_s"] - 1.0,
        "ko",
        ms=6.0,
        label="Cubic spinel family",
    )
    axes[1].loglog(
        [1.00],
        [11.0],
        "s",
        color=figstyle.PALETTE[0],
        ms=8.0,
        label=r"$\mathrm{Co}_2\mathrm{Z}$ hexaferrite ($\mu_s=12, f_r=1\;\mathrm{GHz}$)",
    )
    axes[1].loglog(
        [1.50],
        [39.0],
        "^",
        color=figstyle.PALETTE[1],
        ms=8.5,
        label=r"$\mathrm{CoZrNb}$ laminate ($\mu_s=40, f_r=1.5\;\mathrm{GHz}$)",
    )
    axes[1].loglog(
        [5.87],
        [0.48],
        "D",
        color=figstyle.PALETTE[2],
        ms=7.0,
        label=r"Ikonen film ($\mu_s=1.48, f_r=5.87\;\mathrm{GHz}$)",
    )
    axes[1].set_xlim(0.1, 10.0)
    axes[1].set_ylim(0.2, 500.0)
    axes[1].set_xlabel(r"Resonance frequency $f_r$ [GHz]")
    axes[1].set_ylabel(r"Static susceptibility $\mu_s - 1$")
    axes[1].set_title("(b) Snoek cubic bound vs. Acher planar-anisotropy enhancement")
    axes[1].legend(fontsize=7.6, loc="lower left")

    figstyle.finish(fig)
    figstyle.save(fig, FIG_DIR / "02_snoek_and_acher_limits.png", close=True)

    # ------------------------------------------------------------------
    # Figure 3: Magnetic loss tangent tan(delta_m) vs Hesitivity h_m
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=12.6, height=4.4)
    for idx, (ms, fr_g, col) in enumerate(
        zip(d0["snoek_mu_s"], d0["snoek_f_r_ghz"], cols_s)
    ):
        valid = f_l_ghz < 0.92 * fr_g
        axes[0].semilogy(
            f_l_ghz[valid],
            np.abs(d0["snoek_tan_d_curves"][idx, valid]),
            label=f"$\\mu_s={ms:.1f}$ ($f_r={fr_g:.2f}\\;\\mathrm{{GHz}}$)",
            **figstyle.series_style(idx, markevery=45),
        )
    axes[0].axhline(
        0.05,
        color="red",
        ls="--",
        lw=1.2,
        label=r"Patch usable-band ceiling ($\tan\delta_m = 0.05$)",
    )
    axes[0].fill_between(
        [0.05, 5.0],
        1e-4,
        0.05,
        color="#e5f5e0",
        alpha=0.45,
        label="Low-loss patch substrate zone",
    )
    axes[0].set_xlim(0.05, 3.5)
    axes[0].set_ylim(2e-3, 10.0)
    axes[0].set_xlabel("Frequency $f$ [GHz]")
    axes[0].set_ylabel(r"Magnetic loss tangent $\tan\delta_m(f) = \mu''(f)/\mu'(f)$")
    axes[0].set_title(r"(a) Why resonant patch antennas must operate well below $f_r$")
    axes[0].legend(fontsize=7.6, loc="lower right")

    for idx_k, (key, col, ls) in enumerate(
        [
            ("czn_eq881", figstyle.PALETTE[1], "-"),
            ("co2z_unaligned", figstyle.PALETTE[0], "--"),
            ("nizn_debye", figstyle.PALETTE[3], "-."),
            ("ikonen_film", figstyle.PALETTE[2], ":"),
        ]
    ):
        mu_arr = d0[
            {
                "czn_eq881": "mu_czn",
                "co2z_unaligned": "mu_co2z",
                "nizn_debye": "mu_nizn",
                "ikonen_film": "mu_ikonen",
            }[key]
        ]
        hes = compute_hesitivity(d0["f_broad"], mu_arr)
        lbl = VERIFIED_MAGNETIC_MATERIALS[key]["label"].split("(")[0].strip()
        axes[1].loglog(
            f_b_ghz,
            hes["sigma_m"],
            color=col,
            ls=ls,
            lw=1.9,
            label=f"{lbl} $\\sigma_m(f)$",
        )
        axes[1].loglog(
            f_b_ghz[::12],
            hes["h_eff"][::12],
            "o",
            color=col,
            ms=3.8,
            label=f"{lbl} $h_m = {np.mean(hes['h_eff']):.2e}\\;\\Omega$/m",
        )
    axes[1].set_xlabel("Frequency $f$ [GHz]")
    axes[1].set_ylabel(r"$\sigma_m(f) = \omega\mu_0\mu''$ & $h_{\mathrm{eff}}(f) = \omega\mu_0 / f(\mu)$ [$\Omega$/m]")
    axes[1].set_title(r"(b) Flux-channel hesitivity identity: $\omega\mu_0 / \mathrm{Im}[(\mu-1)^{-1}] \equiv h_m$")
    axes[1].legend(fontsize=7.2, ncol=2, loc="lower right")

    figstyle.finish(fig)
    figstyle.save(fig, FIG_DIR / "03_loss_tangent_and_hesitivity.png", close=True)

    # ------------------------------------------------------------------
    # Figure 4: Lorentz-mu ferrite slab FIT vs Transfer Matrix
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=12.6, height=4.4)
    f1_ghz = d1["freqs"] / 1e9

    axes[0].plot(
        f1_ghz,
        d1["R_disc_fine"],
        label=r"$R_{\mathrm{TM}}(\mu_d)$ analytic",
        **figstyle.series_style(0, with_marker=False),
    )
    axes[0].plot(
        f1_ghz,
        d1["R_fit_fine"],
        "o",
        color=figstyle.PALETTE[0],
        ms=4.8,
        label=r"$R_{\mathrm{FIT}}$ (Wakis 3D)",
    )
    axes[0].plot(
        f1_ghz,
        d1["T_disc_fine"],
        label=r"$T_{\mathrm{TM}}(\mu_d)$ analytic",
        **figstyle.series_style(1, with_marker=False),
    )
    axes[0].plot(
        f1_ghz,
        d1["T_fit_fine"],
        "s",
        color=figstyle.PALETTE[1],
        ms=4.8,
        label=r"$T_{\mathrm{FIT}}$ (Wakis 3D)",
    )
    axes[0].plot(
        f1_ghz,
        d1["A_disc_fine"],
        label=r"$A = 1 - R - T$ analytic",
        **figstyle.series_style(2, with_marker=False),
    )
    axes[0].plot(
        f1_ghz,
        d1["A_fit_fine"],
        "^",
        color=figstyle.PALETTE[2],
        ms=4.5,
        label=r"$A_{\mathrm{FIT}}$ (Wakis 3D)",
    )
    axes[0].axvline(
        float(d1["f_r"]) / 1e9,
        color="k",
        ls=":",
        lw=1.1,
        label=r"$f_r = 1.40\;\mathrm{GHz}$",
    )
    axes[0].set_xlabel("Frequency $f$ [GHz]")
    axes[0].set_ylabel("Power fraction $R, T, A$")
    axes[0].set_title(r"(a) Lorentz-$\mu$ ferrite slab ($d = 50\;\mathrm{mm}$): FIT vs. Transfer Matrix")
    axes[0].legend(fontsize=7.6, ncol=2)

    axes[1].semilogy(
        f1_ghz,
        np.abs(d1["R_fit_coarse"] - d1["R_disc_coarse"]),
        label=f"Coarse $|\\Delta R|$ ($\\Delta z = {j1['coarse']['dz_mm']:.2f}\\;\\mathrm{{mm}}$, RMS={j1['coarse']['rms_R_disc']:.2e})",
        **figstyle.series_style(2, markevery=3),
    )
    axes[1].semilogy(
        f1_ghz,
        np.abs(d1["R_fit_fine"] - d1["R_disc_fine"]),
        label=f"Fine $|\\Delta R|$ ($\\Delta z = {j1['fine']['dz_mm']:.2f}\\;\\mathrm{{mm}}$, RMS={j1['fine']['rms_R_disc']:.2e})",
        **figstyle.series_style(0, markevery=3),
    )
    axes[1].semilogy(
        f1_ghz,
        np.abs(d1["T_fit_fine"] - d1["T_disc_fine"]),
        label=f"Fine $|\\Delta T|$ ($\\Delta z = {j1['fine']['dz_mm']:.2f}\\;\\mathrm{{mm}}$, RMS={j1['fine']['rms_T_disc']:.2e})",
        **figstyle.series_style(1, markevery=3),
    )
    axes[1].set_xlabel("Frequency $f$ [GHz]")
    axes[1].set_ylabel(r"Residual error $|R_{\mathrm{FIT}} - R_{\mathrm{TM}}(\mu_d)|$")
    axes[1].set_title(r"(b) Grid convergence against $\mu_{\mathrm{discrete}}(f, \Delta t)$")
    axes[1].legend(fontsize=7.8)

    figstyle.finish(fig)
    figstyle.save(fig, FIG_DIR / "04_lorentz_mu_slab_rt.png", close=True)

    # ------------------------------------------------------------------
    # Figure 5: Debye-mu relaxation ferrite slab FIT vs Transfer Matrix
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=12.6, height=4.4)
    f2_ghz = d2["freqs"] / 1e9

    axes[0].plot(
        f2_ghz,
        np.real(d2["mu_cont"]),
        label=r"$\mathrm{Re}\,\mu(f)$ continuous",
        **figstyle.series_style(0, with_marker=False),
    )
    axes[0].plot(
        f2_ghz[::2],
        np.real(d2["mu_disc"][::2]),
        "o",
        color=figstyle.PALETTE[0],
        ms=4.5,
        label=r"$\mathrm{Re}\,\mu_d(f,\Delta t)$",
    )
    axes[0].plot(
        f2_ghz,
        -np.imag(d2["mu_cont"]),
        label=r"$-\mathrm{Im}\,\mu(f)$ continuous",
        **figstyle.series_style(1, with_marker=False),
    )
    axes[0].plot(
        f2_ghz[::2],
        -np.imag(d2["mu_disc"][::2]),
        "s",
        color=figstyle.PALETTE[1],
        ms=4.5,
        label=r"$-\mathrm{Im}\,\mu_d(f,\Delta t)$",
    )
    axes[0].axvline(
        float(d2["f_rel"]) / 1e9,
        color="k",
        ls=":",
        lw=1.1,
        label=r"$f_{\mathrm{rel}} = 1.0\;\mathrm{GHz}$",
    )
    axes[0].set_xlabel("Frequency $f$ [GHz]")
    axes[0].set_ylabel("Relative permeability")
    axes[0].set_title(r"(a) Debye-$\mu$ relaxation spectrum ($\mu_s = 5.0,\; f_{\mathrm{rel}} = 1.0\;\mathrm{GHz}$)")
    axes[0].legend(fontsize=7.8)

    axes[1].plot(
        f2_ghz,
        d2["R_disc"],
        label=r"$R_{\mathrm{TM}}(\mu_d)$",
        **figstyle.series_style(0, with_marker=False),
    )
    axes[1].plot(
        f2_ghz,
        d2["R_fit"],
        "o",
        color=figstyle.PALETTE[0],
        ms=4.8,
        label=r"$R_{\mathrm{FIT}}$ (Wakis 3D)",
    )
    axes[1].plot(
        f2_ghz,
        d2["T_disc"],
        label=r"$T_{\mathrm{TM}}(\mu_d)$",
        **figstyle.series_style(1, with_marker=False),
    )
    axes[1].plot(
        f2_ghz,
        d2["T_fit"],
        "s",
        color=figstyle.PALETTE[1],
        ms=4.8,
        label=r"$T_{\mathrm{FIT}}$ (Wakis 3D)",
    )
    axes[1].plot(
        f2_ghz,
        d2["A_disc"],
        label=r"$A_{\mathrm{TM}}(\mu_d)$",
        **figstyle.series_style(2, with_marker=False),
    )
    axes[1].plot(
        f2_ghz,
        d2["A_fit"],
        "^",
        color=figstyle.PALETTE[2],
        ms=4.5,
        label=r"$A_{\mathrm{FIT}}$ (Wakis 3D)",
    )
    axes[1].set_xlabel("Frequency $f$ [GHz]")
    axes[1].set_ylabel("Power fraction $R, T, A$")
    axes[1].set_title(
        f"(b) Debye-$\\mu$ slab ($d = 40\\;\\mathrm{{mm}}$): RMS($\\Delta R$) = {j2['rms_R_disc']:.2e}"
    )
    axes[1].legend(fontsize=7.8, ncol=2)

    figstyle.finish(fig)
    figstyle.save(fig, FIG_DIR / "05_debye_mu_slab_rt.png", close=True)

    # ------------------------------------------------------------------
    # Figure 6: Impedance-matched magneto-dielectric slab (eps(w) = mu(w))
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=12.6, height=4.4)
    f3_ghz = d3["freqs"] / 1e9

    axes[0].plot(
        f3_ghz,
        d3["R_cont_unmatched"],
        label=r"Unmatched ($\varepsilon=1, \mu=\mu(\omega)$) $R_{\mathrm{TM}}$",
        **figstyle.series_style(2, with_marker=False),
    )
    axes[0].plot(
        f3_ghz,
        d3["R_fit_unmatched"],
        "^",
        color=figstyle.PALETTE[2],
        ms=4.5,
        label=r"Unmatched $R_{\mathrm{FIT}}$",
    )
    axes[0].plot(
        f3_ghz,
        d3["R_cont_matched"],
        "k-",
        lw=2.0,
        label=r"Matched ($\varepsilon=\mu$) $R_{\mathrm{TM}} \equiv 0$",
    )
    axes[0].plot(
        f3_ghz,
        d3["R_fit_matched"],
        "o",
        color=figstyle.PALETTE[1],
        ms=5.0,
        label=r"Matched ($\varepsilon=\mu$) $R_{\mathrm{FIT}}$",
    )
    axes[0].plot(
        f3_ghz,
        d3["T_cont_matched"],
        label=r"Matched $T_{\mathrm{TM}}$",
        **figstyle.series_style(0, with_marker=False),
    )
    axes[0].plot(
        f3_ghz,
        d3["T_fit_matched"],
        "s",
        color=figstyle.PALETTE[0],
        ms=4.5,
        label=r"Matched $T_{\mathrm{FIT}}$",
    )
    axes[0].plot(
        f3_ghz,
        d3["A_cont_matched"],
        label=r"Matched $A_{\mathrm{TM}}$",
        **figstyle.series_style(3, with_marker=False),
    )
    axes[0].plot(
        f3_ghz,
        d3["A_fit_matched"],
        "D",
        color=figstyle.PALETTE[3],
        ms=4.2,
        label=r"Matched $A_{\mathrm{FIT}}$",
    )
    axes[0].set_xlabel("Frequency $f$ [GHz]")
    axes[0].set_ylabel("Power fraction $R, T, A$")
    axes[0].set_title(r"(a) Reflectionless $\varepsilon(\omega) = \mu(\omega)$ absorber ($d = 50\;\mathrm{mm}$)")
    axes[0].legend(fontsize=7.3, ncol=2)

    axes[1].semilogy(
        f3_ghz,
        d3["R_fit_unmatched"],
        label=r"Unmatched pure-$\mu$ $R_{\mathrm{FIT}}$",
        **figstyle.series_style(2, markevery=3),
    )
    axes[1].semilogy(
        f3_ghz,
        np.maximum(d3["R_fit_matched"], 1e-8),
        label=r"Impedance-matched ($\varepsilon=\mu$) $R_{\mathrm{FIT}}$",
        **figstyle.series_style(1, markevery=3),
    )
    axes[1].semilogy(
        f3_ghz,
        np.abs(d3["T_fit_matched"] - d3["T_cont_matched"]),
        label=r"Matched $|T_{\mathrm{FIT}} - T_{\mathrm{TM}}|$",
        **figstyle.series_style(0, markevery=3),
    )
    axes[1].set_xlabel("Frequency $f$ [GHz]")
    axes[1].set_ylabel("Reflectance $R$ / Transmittance residual")
    axes[1].set_title(
        f"(b) Reflection suppression ($> 30\\;\\mathrm{{dB}}$): max $R_{{\\mathrm{{FIT}}}} = {j3['max_R_fit_matched']:.2e}$"
    )
    axes[1].legend(fontsize=7.8)

    figstyle.finish(fig)
    figstyle.save(fig, FIG_DIR / "06_impedance_matched_magneto_dielectric_slab.png", close=True)

    # ------------------------------------------------------------------
    # Figure 7: 3D PEC cavity spectra (vacuum vs static mu vs dispersive mu)
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=12.6, height=4.4)
    f_sp_ghz = d4["f_spec"] / 1e9

    axes[0].plot(
        f_sp_ghz,
        d4["spec_vac"],
        "k-",
        lw=1.8,
        label=f"Vacuum ($\\mu=1$): $f_0 = {j4['f_vac_fit_GHz']:.3f}\\;\\mathrm{{GHz}}$",
    )
    axes[0].plot(
        f_sp_ghz,
        d4["spec_static"],
        label=f"Static $\\mu_s=2.25$: $f = {j4['f_static_fit_GHz']:.3f}\\;\\mathrm{{GHz}}$",
        **figstyle.series_style(2, markevery=500),
    )
    axes[0].plot(
        f_sp_ghz,
        d4["spec_sweep"][2],
        label=r"Dispersive Lorentz-$\mu$ ($f_r = 1.76\;\mathrm{GHz}$)",
        **figstyle.series_style(1, markevery=500),
    )
    f_r_mid = float(d4["f_r_sweep"][2]) / 1e9
    f_z_mid = f_r_mid * np.sqrt(2.25)
    axes[0].axvspan(
        f_r_mid,
        f_z_mid,
        color="#fee0d2",
        alpha=0.55,
        label=r"Magnetic stopband ($\mathrm{Re}\,\mu < 0$)",
    )
    axes[0].set_xlabel("Frequency $f$ [GHz]")
    axes[0].set_ylabel("Normalized cavity $|E_y(f)|$ spectrum")
    axes[0].set_title(r"(a) 3D PEC cavity $\mathrm{TE}_{101}$ mode splitting with Lorentz $\mu(\omega)$")
    axes[0].legend(fontsize=7.6)

    # Waterfall of the 5 f_r sweep spectra
    for idx, fr_v in enumerate(d4["f_r_sweep"] / 1e9):
        offset = idx * 0.85
        axes[1].plot(
            f_sp_ghz,
            d4["spec_sweep"][idx] + offset,
            label=f"$f_r = {fr_v:.2f}\\;\\mathrm{{GHz}}$",
            **figstyle.series_style(idx, markevery=600),
        )
        axes[1].plot(
            [d4["f_minus_an"][idx] / 1e9, d4["f_plus_an"][idx] / 1e9],
            [
                d4["spec_sweep"][
                    idx, np.argmin(np.abs(d4["f_spec"] - d4["f_minus_an"][idx]))
                ]
                + offset,
                d4["spec_sweep"][
                    idx, np.argmin(np.abs(d4["f_spec"] - d4["f_plus_an"][idx]))
                ]
                + offset,
            ],
            "ko",
            ms=4.5,
        )
    axes[1].set_xlabel("Frequency $f$ [GHz]")
    axes[1].set_ylabel("Offset normalized $|E_y(f)|$ spectrum")
    axes[1].set_title(r"(b) Cavity polariton waterfall vs. analytic roots ($\bullet$)")
    axes[1].legend(fontsize=7.6, loc="upper right")

    figstyle.finish(fig)
    figstyle.save(fig, FIG_DIR / "07_cavity_mode_splitting_spectra.png", close=True)

    # ------------------------------------------------------------------
    # Figure 8: Cavity-magnon polariton avoided-crossing dispersion diagram
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=12.6, height=4.4)
    fr_d_ghz = d4["f_r_dense"] / 1e9
    fr_sw_ghz = d4["f_r_sweep"] / 1e9

    axes[0].plot(
        fr_d_ghz,
        d4["f_minus_dense"] / 1e9,
        label=r"Lower polariton $f_-(f_r)$ analytic",
        **figstyle.series_style(0, with_marker=False),
    )
    axes[0].plot(
        fr_d_ghz,
        d4["f_plus_dense"] / 1e9,
        label=r"Upper polariton $f_+(f_r)$ analytic",
        **figstyle.series_style(1, with_marker=False),
    )
    axes[0].plot(
        fr_sw_ghz,
        d4["f_minus_fit"] / 1e9,
        "o",
        color=figstyle.PALETTE[0],
        ms=6.5,
        label=r"Wakis 3D FIT $f_-$",
    )
    axes[0].plot(
        fr_sw_ghz,
        d4["f_plus_fit"] / 1e9,
        "s",
        color=figstyle.PALETTE[1],
        ms=6.5,
        label=r"Wakis 3D FIT $f_+$",
    )
    axes[0].axhline(
        float(d4["f_vac_grid"]) / 1e9,
        color="k",
        ls=":",
        lw=1.1,
        label=r"Bare cavity $f_{\mathrm{vac}}$",
    )
    axes[0].axhline(
        float(d4["f_static_grid"]) / 1e9,
        color=figstyle.PALETTE[2],
        ls="--",
        lw=1.2,
        label=r"Quasi-static limit $f_{\mathrm{vac}}/\sqrt{\mu_s}$",
    )
    axes[0].plot(
        fr_d_ghz,
        fr_d_ghz,
        "k--",
        lw=1.0,
        alpha=0.6,
        label=r"FMR line $f = f_r$",
    )
    axes[0].fill_between(
        fr_d_ghz,
        fr_d_ghz,
        fr_d_ghz * np.sqrt(2.25),
        color="#fee0d2",
        alpha=0.45,
        label=r"Stopband $\mathrm{Re}\,\mu(f) < 0$",
    )
    axes[0].set_xlabel(r"Ferrite FMR frequency $f_r$ [GHz]")
    axes[0].set_ylabel("Cavity eigenfrequency $f$ [GHz]")
    axes[0].set_title(r"(a) Avoided crossing: $\omega^2\mu(\omega) = \omega_{\mathrm{vac}}^2$")
    axes[0].legend(fontsize=7.4, loc="upper left")

    err_m = np.abs(d4["f_minus_fit"] - d4["f_minus_an"]) / d4["f_minus_an"] * 100.0
    err_p = np.abs(d4["f_plus_fit"] - d4["f_plus_an"]) / d4["f_plus_an"] * 100.0
    axes[1].plot(
        fr_sw_ghz,
        err_m,
        label=r"Lower branch $|f_{-,\mathrm{FIT}} - f_{-,\mathrm{an}}| / f_{-,\mathrm{an}}$",
        **figstyle.series_style(0, ms=6.0),
    )
    axes[1].plot(
        fr_sw_ghz,
        err_p,
        label=r"Upper branch $|f_{+,\mathrm{FIT}} - f_{+,\mathrm{an}}| / f_{+,\mathrm{an}}$",
        **figstyle.series_style(1, ms=6.0),
    )
    axes[1].set_xlabel(r"Ferrite FMR frequency $f_r$ [GHz]")
    axes[1].set_ylabel("Relative eigenfrequency error [%]")
    axes[1].set_ylim(0.0, 1.5)
    axes[1].set_title("(b) 3D FIT cavity eigenfrequency accuracy (< 1%)")
    axes[1].legend(fontsize=8.2)

    figstyle.finish(fig)
    figstyle.save(fig, FIG_DIR / "08_cavity_polariton_avoided_crossing.png", close=True)


# %% [markdown]
# ## 4. Results & Validation Summary
#
# Running all five tasks (`--full`) generates the setup schematic, the Wikimedia
# FMR diagram, the eight validation figures in `figures/`, and the quantitative
# validation tables in `data/task_0..4.json`:
#
# 1. **Literature ferrites, Snoek's law, Acher's sum rule & hesitivity (`task_0`, `figures/01`–`03`):**
#    - **Acher's integral sum rule** $\int_0^\infty f\,\mu''(f)\,df = \frac{\pi}{2}\Delta\mu\,f_r^2$
#      is satisfied to relative error $< 10^{-4}$ for both low damping
#      ($\gamma_m/(2\pi) = 0.25\;\text{GHz}$) and $3.6\times$ higher damping
#      ($0.90\;\text{GHz}$), confirming linewidth invariance.
#    - **Single-pole hesitivity identity** $-\operatorname{Im}[1/(\mu(\omega)-1)] = \omega\mu_0 / h_m$:
#      the effective hesitivity $h_{\text{eff}}(f) = \omega\mu_0 / f(\mu)$ has a
#      relative standard deviation $< 10^{-14}$ across 4 decades of frequency
#      ($1\;\text{MHz}\text{–}20\;\text{GHz}$) for all four verified materials
#      ($\text{CoZrNb}$ Eq. 8-81: $h_m = 3.46\times 10^5\;\Omega/\text{m}$;
#      unaligned $\text{Co}_2\text{Z}$: $h_m = 1.042\times 10^5\;\Omega/\text{m}$;
#      Ikonen film: $h_m = 1.61\times 10^4\;\Omega/\text{m}$, giving
#      $\mu'=1.461, \tan\delta_m = 0.0284$ at $580\;\text{MHz}$; sintered
#      $\text{NiZn}$ Debye tile: $h_m = 5.21\times 10^4\;\Omega/\text{m}$).
#    - **Periodic FIT magnetic ADE verification:** Wakis's dual-edge magnetic
#      ADE update matches `material.mu_discrete(f, dt)` to relative $L_2$ error
#      $< 5\times 10^{-4}$ for both Lorentz and Debye magnetic poles.
#
# ![Literature ferrites and periodic H-edge magnetic ADE check](figures/01_literature_ferrites_and_ade_check.png)
#
# ![Snoek cubic bound vs Acher planar-anisotropy enhancement](figures/02_snoek_and_acher_limits.png)
#
# ![Magnetic loss tangent vs flux-channel hesitivity](figures/03_loss_tangent_and_hesitivity.png)
#
# 2. **Lorentz-$\mu$ and Debye-$\mu$ ferrite slabs (`task_1`, `task_2`, `figures/04`–`05`):**
#    - The Lorentz-$\mu$ slab ($d = 50\;\text{mm}$, $f_r = 1.40\;\text{GHz}$)
#      and Debye-$\mu$ slab ($d = 40\;\text{mm}$, $f_{\text{rel}} = 1.00\;\text{GHz}$)
#      reproduce the analytical transfer matrix evaluated at `mu_discrete(f, dt)`
#      with $\text{RMS}(R_{\text{FIT}} - R_{\text{disc}}) < 2\times 10^{-3}$ and
#      $\text{RMS}(T_{\text{FIT}} - T_{\text{disc}}) < 2\times 10^{-3}$, with the
#      Lorentz slab residual dropping by $\sim 4\times$ upon halving $\Delta z$.
#
# ![Lorentz-mu ferrite slab FIT vs Transfer Matrix](figures/04_lorentz_mu_slab_rt.png)
#
# ![Debye-mu relaxation ferrite slab FIT vs Transfer Matrix](figures/05_debye_mu_slab_rt.png)
#
# 3. **Impedance-matched magneto-dielectric absorber ($\varepsilon(\omega) = \mu(\omega)$, `task_3`, `figures/06`):**
#    - Matching the electric and magnetic Lorentz poles ($\varepsilon(\omega) = \mu(\omega)$,
#      $\eta(\omega) \equiv \eta_0$) suppresses power reflectance across the
#      entire $0.35\text{–}2.75\;\text{GHz}$ resonance band from
#      $R_{\text{max}} = 0.487$ (unmatched pure-$\mu$ slab) down to
#      $R_{\text{FIT}} < 2\times 10^{-4}$ ($> 33\;\text{dB}$ reflection suppression)
#      while achieving $> 95\%$ resonant absorption.
#
# ![Reflectionless impedance-matched magneto-dielectric absorber slab](figures/06_impedance_matched_magneto_dielectric_slab.png)
#
# 4. **3D PEC cavity mode shifting & polariton splitting (`task_4`, `figures/07`–`08`):**
#    - Loading the $120\times 60\times 120\;\text{mm}^3$ PEC cavity with a
#      Lorentz ferrite ($\mu_s = 2.25$) splits the bare $\text{TE}_{101}$ mode
#      ($f_{\text{vac}} = 1.764\;\text{GHz}$) into lower and upper hybridized
#      cavity–magnon polariton branches $f_-(f_r)$ and $f_+(f_r)$ on either side
#      of the magnetic stopband $[f_r, f_r\sqrt{\mu_s}]$, matching the exact
#      roots of $\omega^2\mu(\omega) = \omega_{\text{vac}}^2$ to $< 0.8\%$
#      across all five $f_r$ values.
#
# ![3D PEC cavity TE101 mode splitting spectra](figures/07_cavity_mode_splitting_spectra.png)
#
# ![Cavity-magnon polariton avoided-crossing dispersion diagram](figures/08_cavity_polariton_avoided_crossing.png)
#
# ---
#
# ## 5. Limits & Failure Modes
#
# 1. **No gyrotropic (DC-biased Polder tensor) permeability:** `DispersiveMaterial`
#    supports isotropic or diagonal-anisotropic permeability
#    $\boldsymbol{\mu}(\omega) = \operatorname{diag}(\mu_x, \mu_y, \mu_z)$ via
#    `mu_poles={'x': [...], 'y': [...], 'z': [...]}`. It does **not** include
#    the off-diagonal antisymmetric Polder tensor components $\pm i\kappa(\omega)$
#    of a DC-biased ferrite (which break Lorentz reciprocity and produce Faraday
#    rotation / circulator isolation).
# 2. **Linear small-signal regime only (no B–H hysteresis or saturation):** The
#    Lorentz and Debye magnetic ADEs are linear in $\mathbf{H}$ (`|\mathbf{h}| \ll H_a`).
#    High-power RF fields that drive $|\mathbf{M}|$ toward saturation ($M_s$),
#    Suhl spin-wave instabilities, or static hysteresis loops are not modeled.
# 3. **No Kottke subpixel smoothing for dispersive $\mu$:** When a curved STL
#    solid carries a magnetic `DispersiveMaterial`, Wakis applies scalar
#    volume-fraction weighting on dual $\mathbf{H}$ edges (and emits a
#    `UserWarning` if `conformal` is enabled), because Kottke tensor subpixel
#    averaging is currently implemented only for electric permittivity
#    $\varepsilon$.
# 4. **Fused GPU kernel limitation:** The custom fused GPU step (`fused=True`)
#    supports electric dispersive poles (`poles`, `sigma`) but raises
#    `NotImplementedError` for magnetic dispersive poles (`mu_poles`, `sigma_m`);
#    pass `fused=False` when running magnetic ADEs on GPU.
#
# > **Forward pointers:**
# > - **Tutorial 3.6 · Plasmonic cylinder** examines how multi-pole ADE count
# >   trades scattering accuracy against runtime in 2D/3D open-region problems.
# > - **Chapter 4 · Magnetic antennas** builds directly on the verified
# >   $\text{CoZrNb}$ laminate (`czn_eq881`), $\text{Co}_2\text{Z}$ hexaferrite
# >   (`co2z_unaligned`), and $\text{NiZn}$ spinel (`nizn_debye`) models from
# >   this tutorial to test electrically small permeable-core dipoles, loops,
# >   and toroidal antennas over ground planes and inside biological tissue.
#
# ---
#
# ## 6. Verified References
#
# 1. **J. L. Snoek**, "Dispersion and absorption in magnetic ferrites at
#    frequencies above one Mc/s," *Physica*, vol. 14, no. 4, pp. 207–217, May
#    1948, doi:[10.1016/0031-8914(48)90038-X](https://doi.org/10.1016/0031-8914(48)90038-X).
#    *(Derivation of Snoek's law $(\mu_s - 1)f_r = \frac{2}{3}(\gamma_{\text{LL}}/2\pi)\mu_0 M_s$
#    for unmagnetized isotropic cubic ferrites.)*
# 2. **O. Acher and S. Dubourg**, "Generalization of Snoek's law to ferromagnetic
#    films and composites," *Phys. Rev. B*, vol. 77, no. 10, p. 104440, Mar.
#    2008, doi:[10.1103/PhysRevB.77.104440](https://doi.org/10.1103/PhysRevB.77.104440).
#    *(Integral sum rule $\int_0^\infty f\,\mu''(f)\,df \propto M_s^2$ and
#    planar-anisotropy generalization of Snoek's limit.)*
# 3. **P. M. T. Ikonen, K. N. Rozanov, A. V. Osipov, P. Alitalo, and S. A.
#    Tretyakov**, "Magnetodielectric substrates in antenna miniaturization:
#    Potential and limitations," *IEEE Trans. Antennas Propag.*, vol. 54, no. 11,
#    pp. 3391–3399, Nov. 2006, doi:[10.1109/TAP.2006.884303](https://doi.org/10.1109/TAP.2006.884303).
#    *(Eqs. 17–20: Lorentz permeability of laminated ferromagnetic films,
#    Snoek/Acher bounds, and the loss-tangent ceiling $\tan\delta_m < Q_r^{-1}(1-\eta_r)/\eta_r$.)*
# 4. **L. Yousefi, D. M. Sebastian, and R. E. Diaz**, "Why the magnetic loss
#    tangent is not a relevant constraint for permeable conformal antennas,"
#    *IEEE Trans. Antennas Propag.*, vol. 64, no. 7, pp. 2784–2796, Jul. 2016,
#    doi:[10.1109/TAP.2016.2554198](https://doi.org/10.1109/TAP.2016.2554198).
#    *(Derivation of $-\operatorname{Im}[1/(\mu-1)] = \omega\mu_0 / h_m$ and
#    Table I hesitivity benchmarks for NiZn ferrite, $\text{Co}_2\text{Z}$
#    hexaferrite, and $\text{CoZrNb}$.)*
# 5. **T. Sebastian**, *Magneto-Dielectric Wire Antennas: Theory and Design*,
#    Ph.D. dissertation, Arizona State Univ., Tempe, AZ, 2013.
#    *(Table 5-2 and Figs. 3-13a, 5-13: Debye and Lorentz models for commercially
#    available sintered NiZn ferrite tiles, unaligned $\text{Co}_2\text{Z}$, and CZN laminates.)*
# 6. **T. Yousefi**, *Next Generation of Magneto-Dielectric Antennas and Optimum
#    Flux Channels*, Ph.D. dissertation, Arizona State Univ., Tempe, AZ, 2017,
#    Eq. (8-81) ($\text{CoZrNb}$ laminate Lorentz model
#    $\mu = 1 + 39/[1 + i(f/1.5\;\text{GHz})/0.75 - (f/1.5\;\text{GHz})^2]$).
# 7. **J. S. McLean and R. Sutton**, "An efficient, low-profile antenna
#    employing lossy magneto-dielectric materials," in *2012 IEEE Int. Workshop
#    Antenna Technol. (iWAT)*, Tucson, AZ, Mar. 2012, pp. 261–264,
#    doi:[10.1109/IWAT.2012.6178662](https://doi.org/10.1109/IWAT.2012.6178662).
# 8. **J. D. Jackson**, *Classical Electrodynamics*, 3rd ed. New York, NY:
#    Wiley, 1998, ch. 7.
# 9. **A. Taflove and S. C. Hagness**, *Computational Electrodynamics: The
#    Finite-Difference Time-Domain Method*, 3rd ed. Norwood, MA: Artech House,
#    2005, ch. 9.


# %%
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Tutorial 2.3: Magnetic materials"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Run fast smoke/CI configuration (< 1 min)",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Run converged full configuration (default)",
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=None,
        help="Run only a single task (0..4)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Regenerate all figures from existing data/task_*.npz",
    )
    if argv is None:
        env_flags = os.environ.get("WAKIS_TUTORIAL_FLAGS", "")
        if "ipykernel" in sys.modules:
            argv = shlex.split(env_flags) if env_flags else ["--plot"]
        else:
            argv = sys.argv[1:]
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.plot and args.task is None and not args.quick and not args.full:
        plot_all()
        print("Regenerated 8 figures in", FIG_DIR)
        return

    quick = bool(args.quick)
    task_runners = {
        0: run_task_0,
        1: run_task_1,
        2: run_task_2,
        3: run_task_3,
        4: run_task_4,
    }
    if args.task is not None:
        summary = task_runners[args.task](quick=quick)
        print(
            f"[Task {args.task}] {summary['name']} completed in {summary['elapsed_s']:.2f} s"
        )
        return

    for tid in range(5):
        summary = task_runners[tid](quick=quick)
        print(
            f"[Task {tid}] {summary['name']} completed in {summary['elapsed_s']:.2f} s"
        )
    plot_all()
    print("All tasks and 8 figures completed in", FIG_DIR)


if __name__ == "__main__":
    main()
