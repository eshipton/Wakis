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
# # Tutorial 2.2 · Fitting Measured Material Data
#
# **Chapter 2 · Materials**
#
# [Tutorial 2.1](../02-01_dispersion_physics/dispersion_physics.ipynb) derived
# the Lorentz, Drude, and Debye dispersion relations from microscopic equations
# of motion and showed how Wakis advances each rational susceptibility pole in
# the time domain via an Auxiliary Differential Equation (ADE) with $O(1)$
# state variables per grid edge.
#
# Real experimental materials, however, rarely arrive as neat pole lists.
# Optical constants of metals and dielectrics are published either as
# **tabulated $(n, k)$ spectra** measured by spectroscopic ellipsometry and
# reflectometry (such as **Johnson & Christy, 1972** [1] for $\text{Au}$,
# $\text{Ag}$, and $\text{Cu}$) or as **empirical dispersion formulas** over a
# bounded wavelength window (such as **Malitson, 1965** [2] for fused silica
# $\text{SiO}_2$). At RF and microwave frequencies, biological tissues are
# characterized by the **Gabriel, Lau & Gabriel (1996)** [3] **four-term
# Cole–Cole model**, whose fractional power-law relaxation terms
# $(i\omega\tau_m)^{1-\alpha_m}$ ($0 < \alpha_m < 1$) correspond to branch cuts
# in the frequency domain and fractional derivatives in time that **cannot be
# integrated directly as rational ADE poles**.
#
# This tutorial shows how to go systematically from experimentally measured
# optical or RF material data to a **causal, passive, and discrete-time-stable**
# multi-pole `DispersiveMaterial` ready for 3D time-domain FIT simulation, and
# how to separate **pole-fitting error** from **ADE/spatial discretization
# error** by comparing time-domain slab reflectance/transmittance directly
# against the exact transfer matrix evaluated at the **measured data**.
#
# > **Time-harmonic sign convention (`exp(+i*omega*t)`):** Throughout Wakis,
# > monochromatic fields vary as $\mathbf{E}(\mathbf{r}, t) = \operatorname{Re}[\hat{\mathbf{E}}(\mathbf{r})\,e^{+i\omega t}]$.
# > Under this convention, **passive (dissipative, non-amplifying) media
# > satisfy $\operatorname{Im}\varepsilon(\omega) \le 0$ for all $\omega > 0$**.
# > Optical tables that report complex refractive index as $\tilde{n} = n + ik$
# > with $k \ge 0$ (the $\exp(-i\omega t)$ physics convention) are converted via
# > `eps_from_nk(n, k)` $= (n - ik)^2 = (n^2 - k^2) - i\,2nk$.

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
# 1. **Convert tabulated $(n, k)$ optical data to complex permittivity** in the
#    Wakis $\exp(+i\omega t)$ convention using `wakis.dispersive.eps_from_nk`
#    and verify passivity ($\operatorname{Im}\varepsilon = -2nk \le 0$).
# 2. **Fit $N = 1\dots 6$ Lorentz–Drude poles** to experimental noble-metal data
#    (**Johnson & Christy, 1972** [1] $\text{Au}$ and $\text{Ag}$, CC0
#    `refractiveindex.info`) with `wakis.dispersive.fit_poles`, tracking the
#    relative RMS fit error $\varepsilon_{\text{rms}}(N)$ as additional
#    interband Lorentz oscillators are added.
# 3. **Distinguish continuous passivity from discrete ADE stability:** explain
#    how log-parameterization inside `fit_poles` guarantees causality and
#    passivity ($\operatorname{Im}\varepsilon(\omega) \le 0$ for all $\omega \ge 0$)
#    by construction, why an unconstrained fit (`eps_inf=None`) on a narrow
#    dielectric window can still violate the explicit leapfrog stability limit
#    (`material.is_stable(dt, courant)`), and how fixing `eps_inf` or filtering
#    pole frequencies restores stability.
# 4. **Choose the frequency/wavelength fit band wisely** and diagnose what
#    happens **outside the fitted window** (out-of-band extrapolation error vs.
#    in-band accuracy and maximum stable timestep $\Delta t_{\max}$).
# 5. **Query the Wakis dispersive material library** (`wakis.material_library`:
#    `list_materials`, `info`) to inspect peer-reviewed sources, validity ranges,
#    fit errors, and $50\;\text{as}$ optical stability metadata across metals,
#    dielectrics, semiconductors, polymers, uniaxial crystals, and RF substrates.
# 6. **Approximate non-rational biological dispersion (Gabriel *et al.*, 1996
#    four-term Cole–Cole model)** [3] for **muscle**, **dry skin**,
#    **non-infiltrated fat**, and **cortical bone** over $10\;\text{MHz}\text{–}10\;\text{GHz}$
#    using a $1\dots 5$-pole **Debye cascade** plus static ionic conductivity
#    $\sigma_i$—the exact tissue models reused by the Chapter 4 biomedical
#    implant antenna tutorial.
# 7. **Decompose total time-domain simulation error into pole-fit error plus
#    ADE/grid discretization error** across four 3D `SolverFIT3D` slab
#    benchmarks ($\text{Au}$, $\text{Ag}$, fused silica $\text{SiO}_2$, and
#    human muscle tissue) by comparing FIT $R(f)/T(f)$ against the exact
#    transfer matrix evaluated directly at the **measured data**.
#
# ### Simulation setups & geometry overview
#
# Four 3D `SolverFIT3D` slab benchmarks are executed across the optical and
# biomedical RF regimes (`figures/00a_setup_geometry.png` and
# `figures/00b_setup_mesh_and_excitation.png`):
#
# - **Optical thin-film & dielectric etalon R/T setup (`Tasks 1 & 2`):**
#   A quasi-1D $2\times 2\times N_z$ `GridFIT3D` with `periodic` transverse
#   boundaries ($\pm x, \pm y$) and 15-cell `cpml` absorbers on $\pm z$. An
#   $E_x$-polarized `CurrentSource` sheet at $z_{\text{src}} < 0$ illuminates a
#   fitted dispersive slab on $z \in [0, d]$: Johnson & Christy noble-metal films
#   ($\text{Au}$ $d = 40\;\text{nm}$ and $\text{Ag}$ $d = 36\;\text{nm}$ with
#   $\Delta z = 1.0\;\text{nm}$ in `Task 1`) and a Malitson fused-silica
#   ($\text{SiO}_2$) Fabry–Pérot etalon ($d = 600\;\text{nm}$, $\Delta z = 5.0\;\text{nm}$,
#   $120\;\text{cells}$ across $d$ in `Task 2`). Reflection and transmission
#   `FluxMonitor` planes at $z_r$ and $z_t$ record $R(\lambda)$ and $T(\lambda)$.
# - **Biomedical RF human muscle tissue R/T setup (`Task 4`):**
#   A quasi-1D $2\times 2\times N_z$ `GridFIT3D` ($z \in [-12, 20]\;\text{mm}$,
#   $\Delta z = 0.10\;\text{mm}$, 15-cell `cpml` on $\pm z$) containing an
#   $8.0\;\text{mm}$ human muscle tissue layer ($z \in [0, 8.0]\;\text{mm}$)
#   parameterized by a 4-pole `Debye` cascade plus static ionic conductivity
#   $\sigma_i = 0.20\;\text{S/m}$ fit to the Gabriel *et al.* (1996) four-term
#   Cole–Cole model [3].
#
# ![Setup geometry and boundary conditions for the Tutorial 2.2 quasi-1D fitted-material slab benchmark](figures/00a_setup_geometry.png)
# ![Grid resolution, smallest-feature zoom, and broadband excitation for Tutorial 2.2](figures/00b_setup_mesh_and_excitation.png)
#
# The table below summarizes the baseline Johnson & Christy gold thin-film
# reflection/transmission setup and the material/frequency variations across
# the silver thin-film, fused-silica etalon, and human muscle tissue benchmarks.
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 8 nm × 8 nm × 240 nm (z ∈ [-108 nm, 132 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 2 × 2 × 240 (960 cells) |
# | Discretization | Δz & resolution | Δ_min = 1 nm, 188.3 cells/λ (in n_max = 2.84), 40.0 cells across feature (40 nm) |
# | Boundary | -x face | Periodic |
# | Boundary | +x face | Periodic |
# | Boundary | -y face | Periodic |
# | Boundary | +y face | Periodic |
# | Boundary | -z face | CPML (15 cells, 15 nm) |
# | Boundary | +z face | CPML (15 cells, 15 nm) |
# | Material | Background | Vacuum, ε = 1.00 (isotropic) |
# | Material | Region 1: Au thin film (J&C fit) | ε_r = 8.08, μ_r = 1.00, model = DispersiveMaterial, z ∈ [0 m, 40 nm] (d = 40 nm) |
# | Source | 1 · CurrentSource | pos: z = -50 nm, pol: J_x (E_x), pulse: f₀ = 560.0 THz, Δf = 360.0 THz |
# | Monitor | 2 · FluxMonitor | FluxMonitor (z = -25 nm), pos: z = -25 nm, freqs: 17 freqs in [336.1, 814.9] THz |
# | Monitor | 3 · FluxMonitor | FluxMonitor (z = 65 nm), pos: z = 65 nm, freqs: 17 freqs in [336.1, 814.9] THz |
# | Variation | Task 0 & Task 3 · Pole fitting & library audit | Non-FIT nonlinear least-squares pole fitting (fit_poles N = 1..6 on J&C Au/Ag; N = 1..5 Debye on Gabriel 1996 tissues) |
# | Variation | Task 1b · Silver thin film (Ag, J&C 1972) | Ag slab (1 Drude + 2 Lorentz fit, d = 36 nm, Lz = 240 nm, Δz = 1.0 nm, λ ∈ [310, 900] nm) |
# | Variation | Task 2 · Fused-silica etalon (SiO₂, Malitson 1965) | SiO₂ Fabry–Pérot slab (2-pole folded Lorentz fit, ε_∞ = 1.696, d = 600 nm, Lz = 1.60 µm, Δz = 5.0 nm, λ ∈ [350, 950] nm) |
# | Variation | Task 4 · Human muscle tissue slab (Gabriel 1996) | Muscle tissue slab (4-pole Debye + σ_i = 0.20 S/m fit, d = 8.0 mm, Lz = 32.0 mm, Δz = 0.10 mm, f ∈ [0.6, 6.0] GHz) |
#
# ---
#
# ## 1. From Tabulated $(n, k)$ to Rational ADE Poles
#
# ### 1.1 Sign conventions and `eps_from_nk`
#
# Optical handbooks and the public-domain `refractiveindex.info` database
# tabulate the complex refractive index as $\tilde{n}_{\text{opt}}(\omega) = n(\omega) + i k(\omega)$
# with real refractive index $n > 0$ and extinction coefficient $k \ge 0$, using
# the $\exp(-i\omega t)$ convention where a plane wave propagating in $+z$ has
# phase $\exp[i(\omega \tilde{n}_{\text{opt}} z / c_0 - \omega t)] = \exp(-\omega k z / c_0)\exp[i\omega(n z / c_0 - t)]$.
#
# Under Wakis's $\exp(+i\omega t)$ time dependence, a $+z$-propagating wave with
# decaying amplitude $\exp(-\omega k z / c_0)$ has complex phase
# $\exp[i(\omega t - \omega \tilde{n}_{\text{Wakis}} z / c_0)]$, which requires
# $\tilde{n}_{\text{Wakis}}(\omega) = n(\omega) - i k(\omega) = \tilde{n}_{\text{opt}}^*(\omega)$.
# Squaring $\tilde{n}_{\text{Wakis}}$ yields the complex relative permittivity
# (`wakis.dispersive.eps_from_nk(n, k)`):
#
# $$\varepsilon(\omega)
#   = \big(n(\omega) - i\,k(\omega)\big)^2
#   = \underbrace{\big(n^2(\omega) - k^2(\omega)\big)}_{\varepsilon'(\omega) = \operatorname{Re}\varepsilon(\omega)}
#     - i\,\underbrace{2\,n(\omega)\,k(\omega)}_{\varepsilon''(\omega) = -\operatorname{Im}\varepsilon(\omega) \ge 0}.$$
#
# For any passive medium ($n \ge 0, k \ge 0$), $\operatorname{Im}\varepsilon(\omega) = -2nk \le 0$.
#
# ### 1.2 Why FDTD/FIT needs rational poles (`fit_poles`)
#
# In the frequency domain, the constitutive relation is multiplicative:
# $\hat{\mathbf{D}}(\omega) = \varepsilon_0 \varepsilon(\omega)\hat{\mathbf{E}}(\omega)$.
# In the time domain, multiplication becomes a causal convolution:
#
# $$\mathbf{D}(t)
#   = \varepsilon_0 \varepsilon_\infty \mathbf{E}(t)
#   + \varepsilon_0 \int_0^t \chi(\tau)\,\mathbf{E}(t - \tau)\,d\tau.$$
#
# If $\chi(\tau)$ were obtained purely by numerical inverse Fourier transform of
# tabulated data, advancing $N_t$ timesteps would require $O(N_t)$ history
# arrays per electric-field edge and $O(N_t^2)$ arithmetic operations. Instead,
# `wakis.dispersive.fit_poles` approximates the tabulated spectrum by a rational
# sum of $N_{\text{Lor}}$ Lorentz, $N_{\text{Dru}}$ Drude, and $N_{\text{Deb}}$
# Debye poles (plus optional static conductivity $\sigma$ in `DispersiveMaterial`):
#
# $$\varepsilon_{\text{model}}(\omega)
#   = \varepsilon_\infty
#   + \sum_{p=1}^{N_{\text{Lor}}}
#     \frac{\Delta\varepsilon_p\,\omega_{0,p}^2}{\omega_{0,p}^2 - \omega^2 + i\gamma_p\omega}
#   - \sum_{p=1}^{N_{\text{Dru}}}
#     \frac{\omega_{p,p}^2}{\omega(\omega - i\gamma_{D,p})}
#   + \sum_{p=1}^{N_{\text{Deb}}}
#     \frac{\Delta\varepsilon_{D,p}}{1 + i\omega\tau_p}
#   + \frac{\sigma}{i\omega\varepsilon_0}.$$
#
# Given $M$ tabulated frequency samples $\{f_j, \varepsilon_{\text{data}}(f_j)\}_{j=1}^M$,
# `fit_poles` minimizes the relative complex residual via Levenberg–Marquardt
# (`scipy.optimize.least_squares(..., method='lm')`) across `n_starts`
# logarithmically spaced initial damping guesses:
#
# $$r_j(\mathbf{x})
#   = \frac{\varepsilon_{\text{model}}(f_j;\,\mathbf{x}) - \varepsilon_{\text{data}}(f_j)}
#          {\max\!\big(|\varepsilon_{\text{data}}(f_j)|,\,10^{-12}\big)},
#   \qquad
#   \varepsilon_{\text{rms}}
#   = \sqrt{\frac{1}{M}\sum_{j=1}^M |r_j(\mathbf{x}^*)|^2}
#   \equiv \texttt{material.fit\_error}.$$
#
# ### 1.3 Enforcing passivity vs. discrete ADE stability
#
# Inside `fit_poles`, every physical parameter ($\varepsilon_\infty$,
# $\Delta\varepsilon_p$, $\omega_{0,p}$, $\gamma_p$, $\omega_{p,p}$, $\tau_p$)
# is parameterized by its logarithm, $\theta_k = \exp(x_k) > 0$. This has two
# immediate consequences:
#
# 1. **Passivity & causality are guaranteed by construction:** Because every
#    oscillator strength, resonance frequency, damping rate, and relaxation time
#    is strictly positive, every pole of $\chi(\omega)$ lies in the upper half of
#    the complex $\omega$-plane ($\operatorname{Im}\omega > 0$), and
#
#    $$\operatorname{Im}\varepsilon_{\text{model}}(\omega)
#      = -\sum_{p=1}^{N_{\text{Lor}}}
#        \frac{\Delta\varepsilon_p\,\omega_{0,p}^2\,\gamma_p\,\omega}
#             {(\omega_{0,p}^2 - \omega^2)^2 + \gamma_p^2\omega^2}
#        -\sum_{p=1}^{N_{\text{Dru}}}
#        \frac{\omega_{p,p}^2\,\gamma_{D,p}}{\omega(\omega^2 + \gamma_{D,p}^2)}
#        -\sum_{p=1}^{N_{\text{Deb}}}
#        \frac{\Delta\varepsilon_{D,p}\,\omega\tau_p}{1 + (\omega\tau_p)^2}
#        -\frac{\sigma}{\omega\varepsilon_0}
#      \le 0
#      \qquad \forall\,\omega > 0,$$
#
#    **even outside the wavelength window used for fitting!**
# 2. **Continuous passivity does NOT automatically imply discrete-time ADE
#    stability (`is_stable`):** As shown in Tutorial 2.1, the explicit second-order
#    leapfrog update coupled to a Lorentz pole at $\omega_{0,p}$ is stable at
#    timestep $\Delta t$ and 3D Courant number $S = c_0\Delta t\sqrt{\Delta x^{-2}+\Delta y^{-2}+\Delta z^{-2}}$
#    only when all eigenvalues of the amplification matrix
#    ` M = _amplification_matrix(material, dt, q)` satisfy $|\lambda_i| \le 1$
#    for $q \in [0, 2S]$. In particular:
#    - Every Lorentz pole must satisfy $\omega_{0,p}\Delta t < 2$ (and tightened
#      by the static-to-optical permittivity ratio when $\gamma_p \to 0$);
#    - The high-frequency limit must satisfy $\varepsilon_\infty > S^2$.
#      When fitting a **lossless normal-dispersion dielectric** (like fused
#      silica $\text{SiO}_2$) over a narrow visible window that excludes the UV
#      resonances, an unconstrained fit (`eps_inf=None`) can trade $\varepsilon_\infty \to 0$
#      against a far-UV Lorentz pole ($\omega_{0,p} \gg \omega_{\text{max}}$).
#      The resulting fit is continuous-time passive ($\operatorname{Im}\varepsilon \le 0$),
#      yet `material.is_stable(dt, courant=0.5)` returns `False`! Fixing
#      `eps_inf >= 1.0` (e.g. `eps_inf=1.0` or folding out-of-band UV poles into
#      `eps_inf`) eliminates this degeneracy and yields a large stable timestep
#      $\Delta t_{\max}$.
#
# ---
#
# ## 2. Biological Tissues: Gabriel *et al.* (1996) Four-Term Cole–Cole Model
#
# ### 2.1 Why a Cole–Cole term is not an ADE pole
#
# In their landmark parametric survey across $10\;\text{Hz}\text{–}100\;\text{GHz}$,
# **Gabriel, Lau & Gabriel (1996)** [3] modeled the complex relative permittivity
# of human tissues as the sum of four **Cole–Cole dispersion regions** ($\alpha$,
# $\beta$, $\gamma$, and $\delta$ dispersions arising from counterion diffusion,
# cellular membrane charging / Maxwell–Wagner interfacial polarization,
# protein-bound water, and free water dipolar relaxation) plus a static ionic
# conductivity $\sigma_i$. In the $\exp(+i\omega t)$ convention:
#
# $$\varepsilon_{\text{CC}}(\omega)
#   = \varepsilon_\infty
#   + \sum_{m=1}^4 \frac{\Delta\varepsilon_m}{1 + (i\omega\tau_m)^{1 - \alpha_m}}
#   + \frac{\sigma_i}{i\omega\varepsilon_0},
#   \qquad 0 \le \alpha_m < 1.$$
#
# When the broadening exponent $\alpha_m = 0$, term $m$ reduces to a standard
# first-order `Debye(delta_eps, tau)` pole. When $0 < \alpha_m < 1$, however,
# $(i\omega\tau_m)^{1-\alpha_m} = (\omega\tau_m)^{1-\alpha_m}\exp\!\big[i(1-\alpha_m)\pi/2\big]$
# is a **fractional power of $i\omega$**:
#
# - In the complex $\omega$-plane, $1/[1 + (i\omega\tau_m)^{1-\alpha_m}]$ has a
#   **branch point at $\omega = 0$** (and a branch cut along the positive
#   imaginary axis) rather than a finite set of isolated simple poles; on a
#   Cole–Cole Argand diagram ($\varepsilon''$ vs. $\varepsilon'$), the Debye
#   semicircle is depressed below the real axis by angle $\alpha_m \pi / 2$.
# - In the time domain, $(i\omega)^{1-\alpha_m}$ corresponds to a **Caputo
#   fractional derivative** ${}_0 D_t^{1-\alpha_m}$ whose impulse response is a
#   two-parameter Mittag-Leffler function decaying as an algebraic power law
#   $t^{-(2-\alpha_m)}$ rather than an exponential $\exp(-t/\tau_p)$—requiring the
#   entire time history if integrated directly.
#
# ### 2.2 Multi-pole Debye cascade approximation over $10\;\text{MHz}\text{–}10\;\text{GHz}$
#
# Physically, a fractional Cole–Cole term represents a continuous log-normal-like
# distribution of molecular and cellular relaxation times $\tau$. Over any
# finite RF/microwave bandwidth—such as $10\;\text{MHz}\text{–}10\;\text{GHz}$
# (3 decades covering the $402\text{–}405\;\text{MHz}$ MICS band, $915\;\text{MHz}$
# and $2.45\;\text{GHz}$ ISM bands, and $3.1\text{–}10.6\;\text{GHz}$ UWB band)—this
# continuous relaxation spectrum can be approximated to sub-percent accuracy by
# $N = 3\text{–}5$ discrete **Debye poles** plus the exact static ionic
# conductivity $\sigma_i$:
#
# $$\varepsilon_{\text{pol}}(\omega)
#   \equiv \varepsilon_{\text{CC}}(\omega) - \frac{\sigma_i}{i\omega\varepsilon_0}
#   \approx \varepsilon_\infty + \sum_{p=1}^N \frac{\Delta\varepsilon_p}{1 + i\omega\tau_p}
#   \quad \Longrightarrow \quad
#   \texttt{DispersiveMaterial}\big(\varepsilon_\infty,\,\{\texttt{Debye}(\Delta\varepsilon_p, \tau_p)\}_{p=1}^N,\,\texttt{sigma}=\sigma_i\big).$$
#
# Because Wakis integrates both `Debye` poles and `sigma` via Crank–Nicolson
# time-averaging at $t^{n+1/2}$, the resulting multi-pole Debye tissue model is
# **unconditionally stable for any Courant number $S \le 1$** regardless of how
# small or large $\tau_p$ is!
#
# ### 2.3 Published Gabriel, Lau & Gabriel (1996) Table 1 parameters
#
# Table 1 of **Gabriel, Lau & Gabriel (1996)** (*Phys. Med. Biol.* **41**(11),
# 2271–2293, DOI: [`10.1088/0031-9155/41/11/003`](https://doi.org/10.1088/0031-9155/41/11/003))
# gives the following four-term Cole–Cole parameters at body temperature ($37\,^\circ\text{C}$):
#
# | Tissue | $\varepsilon_\infty$ | $\Delta\varepsilon_1$ | $\tau_1\;[\text{ps}]$ | $\alpha_1$ | $\Delta\varepsilon_2$ | $\tau_2\;[\text{ns}]$ | $\alpha_2$ | $\Delta\varepsilon_3$ | $\tau_3\;[\mu\text{s}]$ | $\alpha_3$ | $\Delta\varepsilon_4$ | $\tau_4\;[\text{ms}]$ | $\alpha_4$ | $\sigma_i\;[\text{S/m}]$ |
# | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | **Muscle** | $4.0$ | $50.0$ | $7.23$ | $0.10$ | $7000$ | $353.68$ | $0.10$ | $1.2\times 10^6$ | $318.31$ | $0.10$ | $2.5\times 10^7$ | $2.274$ | $0.00$ | $0.200$ |
# | **Skin (Dry)** | $4.0$ | $32.0$ | $7.23$ | $0.00$ | $1100$ | $32.48$ | $0.20$ | $0.0$ | — | — | $0.0$ | — | — | $0.0002$ |
# | **Fat (Not Infiltrated)** | $2.5$ | $3.0$ | $7.96$ | $0.20$ | $15.0$ | $15.92$ | $0.10$ | $3.3\times 10^4$ | $159.15$ | $0.05$ | $1.0\times 10^7$ | $7.958$ | $0.01$ | $0.010$ |
# | **Bone (Cortical)** | $2.5$ | $10.0$ | $13.13$ | $0.20$ | $180.0$ | $79.58$ | $0.20$ | $5.0\times 10^3$ | $159.15$ | $0.20$ | $1.0\times 10^5$ | $15.915$ | $0.00$ | $0.020$ |
#
# ---
#
# ## 3. Error Decomposition in a Time-Domain Slab Simulation
#
# When we simulate a dispersive slab of thickness $d$ in `SolverFIT3D` using a
# pole model $\varepsilon_{\text{fit}}(\omega)$ fitted to measured data
# $\varepsilon_{\text{meas}}(\omega)$, the total discrepancy between the
# simulated reflectance $R_{\text{FIT}}(f)$ and the ground-truth reflectance
# $R_{\text{meas}}(f) = R_{\text{TM}}\big(f, d, \varepsilon_{\text{meas}}(f)\big)$
# computed directly from the **measured data** splits into two distinct physical
# and numerical contributions:
#
# $$\underbrace{R_{\text{FIT}}(f) - R_{\text{meas}}(f)}_{\text{Total error }\Delta R_{\text{tot}}(f)}
#   = \underbrace{\Big[R_{\text{TM}}\big(f, d, \varepsilon_{\text{fit}}(f)\big) - R_{\text{TM}}\big(f, d, \varepsilon_{\text{meas}}(f)\big)\Big]}_{\text{Pole-fit error }\Delta R_{\text{fit}}(f)}
#   + \underbrace{\Big[R_{\text{FIT}}(f) - R_{\text{TM}}\big(f, d, \varepsilon_{\text{fit}}(f)\big)\Big]}_{\text{ADE + spatial grid discretization error }\Delta R_{\text{disc}}(f)}.$$
#
# Moreover, evaluating the analytical transfer matrix at Wakis's discrete-time
# ADE permittivity `material.epsilon_discrete(f, dt)` isolates the temporal ADE
# truncation error $O(\Delta t^2)$ from the spatial Yee-grid dispersion and
# interface error $O(\Delta z^2)$. Whenever $\|\Delta R_{\text{fit}}\| \gg \|\Delta R_{\text{disc}}\|$
# (e.g. an $N=1$ Drude fit to gold across its $550\;\text{nm}$ interband edge,
# or an $N=1$ Debye fit to muscle across $0.5\text{–}6\;\text{GHz}$), **refining
# the FDTD grid cannot reduce the total error**—you must add poles or narrow the
# fit band. Conversely, once $\|\Delta R_{\text{fit}}\| \ll \|\Delta R_{\text{disc}}\|$
# (e.g. $N \ge 3$ poles on gold in the visible/NIR or $N = 4$ Debye poles on
# muscle), adding more poles only wastes memory and ADE operations per step
# (a trade-off explored for 2D/3D plasmonic scattering in **Tutorial 3.6 · Plasmonic cylinder**).

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
    dispersive_lib,
    eps_from_nk,
    fit_poles,
)
import wakis.figstyle as figstyle
from wakis.material_library import info as material_info
from wakis.material_library import list_materials
import wakis.schematic as schematic
from wakis.sources import CurrentSource, GaussianSource

warnings.filterwarnings("ignore", message="IProgress not found")
figstyle.apply_style()

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_ROOT = (
    HERE.parents[2] if (HERE.name == "02-02_fitting_measured_data") else Path.cwd()
)
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
RI_DIR = REPO_ROOT / "tests" / "data" / "refractiveindex_info"
JC_AU_PATH = RI_DIR / "Au_Johnson.yml"
JC_AG_PATH = RI_DIR / "Ag_Johnson.yml"
SIO2_PATH = RI_DIR / "SiO2_Malitson.yml"


# %%
# Gabriel, Lau & Gabriel, Phys. Med. Biol. 41(11), 2271-2293 (1996), Table 1
# DOI: 10.1088/0031-9155/41/11/003
GABRIEL_1996_TISSUES = {
    "muscle": {
        "label": "Muscle",
        "eps_inf": 4.0,
        "sigma_i": 0.200,
        "terms": [
            (50.0, 7.23e-12, 0.10),
            (7000.0, 353.68e-9, 0.10),
            (1.2e6, 318.31e-6, 0.10),
            (2.5e7, 2.274e-3, 0.00),
        ],
    },
    "skin_dry": {
        "label": "Skin (Dry)",
        "eps_inf": 4.0,
        "sigma_i": 0.0002,
        "terms": [
            (32.0, 7.23e-12, 0.00),
            (1100.0, 32.48e-9, 0.20),
            (0.0, 1.0e-6, 0.00),
            (0.0, 1.0e-3, 0.00),
        ],
    },
    "fat": {
        "label": "Fat (Not Infiltrated)",
        "eps_inf": 2.5,
        "sigma_i": 0.010,
        "terms": [
            (3.0, 7.96e-12, 0.20),
            (15.0, 15.92e-9, 0.10),
            (3.3e4, 159.15e-6, 0.05),
            (1.0e7, 7.958e-3, 0.01),
        ],
    },
    "bone_cortical": {
        "label": "Bone (Cortical)",
        "eps_inf": 2.5,
        "sigma_i": 0.020,
        "terms": [
            (10.0, 13.13e-12, 0.20),
            (180.0, 79.58e-9, 0.20),
            (5.0e3, 159.15e-6, 0.20),
            (1.0e5, 15.915e-3, 0.00),
        ],
    },
}


def load_refractiveindex_nk(yml_path):
    """
    Parse a ``refractiveindex.info`` YAML file in ``tabulated nk`` format
    without requiring PyYAML, returning arrays sorted by ascending frequency
    (descending wavelength).

    Parameters
    ----------
    yml_path : str or Path
        Path to the ``refractiveindex.info`` YAML file.

    Returns
    -------
    lam_um : ndarray
        Vacuum wavelengths in micrometres [um], descending.
    freqs : ndarray
        Frequencies in Hz [Hz], ascending.
    n_tab : ndarray
        Real refractive index ``n`` (dimensionless).
    k_tab : ndarray
        Extinction coefficient ``k >= 0`` (dimensionless).
    eps_tab : ndarray
        Complex relative permittivity ``eps_from_nk(n, k)`` in the Wakis
        ``exp(+i*omega*t)`` convention (``Im(eps) <= 0``).
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
    # Sort by descending wavelength so frequency is strictly ascending
    order = np.argsort(arr[:, 0])[::-1]
    lam_um = arr[order, 0]
    n_tab = arr[order, 1]
    k_tab = arr[order, 2]
    freqs = c0 / (lam_um * 1e-6)
    eps_tab = eps_from_nk(n_tab, k_tab)
    return lam_um, freqs, n_tab, k_tab, eps_tab


def load_refractiveindex_sellmeier(yml_path):
    """
    Parse a ``refractiveindex.info`` YAML file in ``formula 1`` (Sellmeier)
    format, returning ``(wavelength_range_um, eps_inf, B_list, C_um_list)``.

    Parameters
    ----------
    yml_path : str or Path
        Path to the Sellmeier YAML file (e.g. ``SiO2_Malitson.yml``).

    Returns
    -------
    wl_range_um : tuple of float
        Validity range ``(lam_min_um, lam_max_um)`` [um].
    eps_inf : float
        High-frequency offset ``1 + c_0`` (dimensionless).
    B_vals : ndarray
        Oscillator strengths ``B_p`` (dimensionless).
    C_um_vals : ndarray
        Resonance wavelengths ``C_p`` in micrometres [um].
    """
    text = Path(yml_path).read_text(encoding="utf-8")
    wl_range = (0.21, 6.7)
    coeffs = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("wavelength_range:"):
            parts = s.split(":", 1)[1].split()
            wl_range = (float(parts[0]), float(parts[1]))
        elif s.startswith("coefficients:"):
            coeffs = [float(x) for x in s.split(":", 1)[1].split()]
    eps_inf = 1.0 + coeffs[0]
    B_vals = np.asarray(coeffs[1::2], dtype=np.float64)
    C_um_vals = np.asarray(coeffs[2::2], dtype=np.float64)
    return wl_range, eps_inf, B_vals, C_um_vals


def sellmeier_eps_exact(lam_um, eps_inf, B_vals, C_um_vals):
    """
    Evaluate the lossless Sellmeier permittivity at vacuum wavelengths
    ``lam_um`` [um]:

        eps(lam) = eps_inf + sum_p B_p * lam^2 / (lam^2 - C_p^2).

    Parameters
    ----------
    lam_um : array_like
        Vacuum wavelength in micrometres [um].
    eps_inf : float
        High-frequency permittivity offset.
    B_vals, C_um_vals : array_like
        Sellmeier oscillator strengths and resonance wavelengths [um].

    Returns
    -------
    ndarray
        Complex relative permittivity ``eps(lam_um) + 0j``.
    """
    lam2 = np.asarray(lam_um, dtype=np.float64) ** 2
    eps = np.full_like(lam2, float(eps_inf), dtype=np.complex128)
    for b_p, c_p in zip(B_vals, C_um_vals):
        eps = eps + (b_p * lam2) / (lam2 - c_p * c_p)
    return eps


def cole_cole_permittivity(freqs, tissue_key, include_sigma=True):
    """
    Evaluate the Gabriel, Lau & Gabriel (1996) four-term Cole-Cole complex
    relative permittivity in the Wakis ``exp(+i*omega*t)`` convention
    (``Im(eps) <= 0``).

    Parameters
    ----------
    freqs : array_like
        Frequencies in Hz [Hz], strictly positive.
    tissue_key : str
        Key in ``GABRIEL_1996_TISSUES`` (``'muscle'``, ``'skin_dry'``,
        ``'fat'``, or ``'bone_cortical'``).
    include_sigma : bool, optional
        If ``True`` (default), include the ionic conductivity term
        ``sigma_i / (1j * omega * eps_0)``.

    Returns
    -------
    ndarray
        Complex relative permittivity ``eps_CC(freqs)``.
    """
    spec = GABRIEL_1996_TISSUES[tissue_key]
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64)
    eps = np.full(omega.shape, float(spec["eps_inf"]), dtype=np.complex128)
    for d_eps, tau, alpha in spec["terms"]:
        if d_eps > 0.0 and tau > 0.0:
            eps = eps + d_eps / (1.0 + (1j * omega * tau) ** (1.0 - alpha))
    if include_sigma and spec["sigma_i"] > 0.0:
        eps = eps + spec["sigma_i"] / (1j * omega * eps_0)
    return eps


def fit_cole_cole_debye(freqs, tissue_key, n_debye=3, n_starts=4):
    """
    Fit a passive ``n_debye``-pole Debye cascade plus exact static ionic
    conductivity ``sigma_i`` to the Gabriel et al. (1996) four-term Cole-Cole
    model over ``freqs`` [Hz].

    Parameters
    ----------
    freqs : array_like
        Fit frequencies in Hz [Hz].
    tissue_key : str
        Key in ``GABRIEL_1996_TISSUES``.
    n_debye : int, optional
        Number of Debye relaxation poles (default 3).
    n_starts : int, optional
        Number of deterministic multi-start guesses passed to ``fit_poles``.

    Returns
    -------
    mat : DispersiveMaterial
        Fitted ``DispersiveMaterial`` with ``n_debye`` ``Debye`` poles and
        ``sigma = sigma_i``; ``mat.fit_error`` stores the relative RMS error
        against the full Cole-Cole permittivity (including ``sigma_i``).
    """
    spec = GABRIEL_1996_TISSUES[tissue_key]
    freqs = np.asarray(freqs, dtype=np.float64)
    eps_cc_tot = cole_cole_permittivity(freqs, tissue_key, include_sigma=True)
    eps_cc_pol = cole_cole_permittivity(freqs, tissue_key, include_sigma=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pol_fit = fit_poles(
            freqs,
            eps_cc_pol,
            n_lorentz=0,
            n_drude=0,
            n_debye=n_debye,
            eps_inf=None,
            n_starts=n_starts,
            name=f"{tissue_key}_debye{n_debye}",
        )
    # Sort Debye poles by descending relaxation time tau for clean reporting
    sorted_poles = sorted(pol_fit.poles, key=lambda p: -float(p.tau))
    mat = DispersiveMaterial(
        float(pol_fit.eps_inf),
        sorted_poles,
        sigma=float(spec["sigma_i"]),
        name=f"{tissue_key}_debye{n_debye}",
    )
    rel_res = (mat.epsilon(freqs) - eps_cc_tot) / np.abs(eps_cc_tot)
    mat.fit_error = float(np.sqrt(np.mean(np.abs(rel_res) ** 2)))
    mat.pol_fit_error = float(pol_fit.fit_error)
    return mat


def find_max_stable_dt(material, courant=0.5, dt_min=1e-18, dt_max=1e-8):
    """
    Approximate the maximum stable ADE timestep ``dt_max_stable`` [s] at
    3D Courant number ``courant`` via logarithmic bracket + bisection on
    ``material.is_stable(dt, courant)``.

    Parameters
    ----------
    material : DispersiveMaterial
        Material model to test.
    courant : float, optional
        3D Courant number (default 0.5).
    dt_min, dt_max : float, optional
        Search bounds in seconds [s].

    Returns
    -------
    float
        Maximum stable timestep [s] (``0.0`` if unstable even at ``dt_min``).
    """
    if not material.is_stable(dt_min, courant):
        return 0.0
    grid = np.geomspace(dt_min, dt_max, 120)
    lo = dt_min
    hi = None
    for dt_try in grid[1:]:
        if material.is_stable(float(dt_try), courant):
            lo = float(dt_try)
        else:
            hi = float(dt_try)
            break
    if hi is None:
        return float(dt_max)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if material.is_stable(mid, courant):
            lo = mid
        else:
            hi = mid
    return float(lo)


def transfer_matrix_slab(freqs, d, eps_r, mu_r=1.0):
    """
    Exact normal-incidence complex amplitude ``(r, t)`` and power ``(R, T, A)``
    of a homogeneous slab of thickness ``d`` [m] in vacuum under the Wakis
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


def run_dispersive_slab_pair(
    material,
    freqs,
    eps_measured,
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
    reference) with periodic transverse boundaries and 15-cell CPML absorbers,
    and compare against three analytical transfer-matrix references:
    1. ``tm_meas``: evaluated directly at the MEASURED permittivity ``eps_measured``,
    2. ``tm_cont``: evaluated at the continuous fitted model ``material.epsilon(freqs)``,
    3. ``tm_disc``: evaluated at the discrete ADE model ``material.epsilon_discrete(freqs, dt)``.
    """
    nz = int(round(lz / dz))
    zmin = -0.45 * lz
    zmax = zmin + nz * dz
    dxy = 4.0 * dz
    freqs = np.asarray(freqs, dtype=np.float64)
    eps_measured = np.asarray(eps_measured, dtype=np.complex128)

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
    tm_meas = transfer_matrix_slab(freqs, thickness, eps_measured, 1.0)
    tm_cont = transfer_matrix_slab(freqs, thickness, eps_cont, 1.0)
    tm_disc = transfer_matrix_slab(freqs, thickness, eps_disc, 1.0)

    # Error decomposition:
    # Total error (FIT vs measured) = Pole-fit error (cont vs measured) + Discretization error (FIT vs cont)
    rms_R_tot = float(np.sqrt(np.mean((R_fit - tm_meas["R"]) ** 2)))
    rms_T_tot = float(np.sqrt(np.mean((T_fit - tm_meas["T"]) ** 2)))
    rms_R_fit_err = float(np.sqrt(np.mean((tm_cont["R"] - tm_meas["R"]) ** 2)))
    rms_T_fit_err = float(np.sqrt(np.mean((tm_cont["T"] - tm_meas["T"]) ** 2)))
    rms_R_disc_err = float(np.sqrt(np.mean((R_fit - tm_disc["R"]) ** 2)))
    rms_T_disc_err = float(np.sqrt(np.mean((T_fit - tm_disc["T"]) ** 2)))

    n_complex = np.sqrt(eps_cont)
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
        "eps_measured": eps_measured,
        "eps_cont": eps_cont,
        "eps_disc": eps_disc,
        "R_meas": tm_meas["R"],
        "T_meas": tm_meas["T"],
        "A_meas": tm_meas["A"],
        "R_cont": tm_cont["R"],
        "T_cont": tm_cont["T"],
        "A_cont": tm_cont["A"],
        "R_disc": tm_disc["R"],
        "T_disc": tm_disc["T"],
        "A_disc": tm_disc["A"],
        "rms_R_tot": rms_R_tot,
        "rms_T_tot": rms_T_tot,
        "rms_R_fit_err": rms_R_fit_err,
        "rms_T_fit_err": rms_T_fit_err,
        "rms_R_disc_err": rms_R_disc_err,
        "rms_T_disc_err": rms_T_disc_err,
    }


# %%
def serialize_poles(material):
    """Return a JSON-serializable list of pole dicts for ``material``."""
    out = []
    for p in material.poles:
        if isinstance(p, Drude):
            out.append(
                {
                    "type": "Drude",
                    "omega_p_rad_s": float(p.omega_p),
                    "gamma_rad_s": float(p.gamma),
                }
            )
        elif isinstance(p, Lorentz):
            out.append(
                {
                    "type": "Lorentz",
                    "delta_eps": float(p.delta_eps),
                    "omega_0_rad_s": float(p.omega_0),
                    "gamma_rad_s": float(p.gamma),
                }
            )
        elif isinstance(p, Debye):
            out.append(
                {
                    "type": "Debye",
                    "delta_eps": float(p.delta_eps),
                    "tau_s": float(p.tau),
                }
            )
    return out


def deserialize_material(spec):
    """Reconstruct a ``DispersiveMaterial`` from a JSON dictionary."""
    poles = []
    for p in spec["poles"]:
        if p["type"] == "Drude":
            poles.append(Drude(p["omega_p_rad_s"], p["gamma_rad_s"]))
        elif p["type"] == "Lorentz":
            poles.append(
                Lorentz(p["delta_eps"], p["omega_0_rad_s"], p["gamma_rad_s"])
            )
        elif p["type"] == "Debye":
            poles.append(Debye(p["delta_eps"], p["tau_s"]))
    return DispersiveMaterial(
        float(spec["eps_inf"]),
        poles,
        sigma=float(spec.get("sigma", 0.0)),
        name=spec.get("name", "fitted"),
    )


def run_task_0(quick=False):
    """
    Task 0: Fit N = 1..6 Lorentz-Drude poles to Johnson & Christy (1972) Au and
    Ag data over 0.30-1.00 um, verify passivity (Im eps <= 0) and discrete ADE
    stability (is_stable), perform the fit-band extrapolation study (narrow
    0.50-0.90 um vs visible-NIR 0.35-0.90 um vs full UV-NIR 0.19-1.94 um), and
    audit wakis.material_library.
    """
    t0 = time.perf_counter()
    lam_au, f_au, n_au, k_au, eps_au = load_refractiveindex_nk(JC_AU_PATH)
    lam_ag, f_ag, n_ag, k_ag, eps_ag = load_refractiveindex_nk(JC_AG_PATH)

    n_starts = 3 if quick else 5
    pole_counts = np.arange(1, 7, dtype=int)

    # Fit Au and Ag over the optical/NIR window 0.30-1.00 um with 1 Drude + (N-1) Lorentz poles
    mask_au_opt = (lam_au >= 0.30) & (lam_au <= 1.00)
    mask_ag_opt = (lam_ag >= 0.30) & (lam_ag <= 1.00)
    f_dense = c0 / (np.linspace(0.188, 1.937, 240)[::-1] * 1e-6)
    f_passivity_check = np.geomspace(1e12, 1e17, 500)

    au_fits = {}
    ag_fits = {}
    au_errors = np.zeros(pole_counts.size)
    ag_errors = np.zeros(pole_counts.size)
    au_max_dt = np.zeros(pole_counts.size)
    ag_max_dt = np.zeros(pole_counts.size)
    au_eps_dense = np.zeros((pole_counts.size, f_dense.size), dtype=np.complex128)
    ag_eps_dense = np.zeros((pole_counts.size, f_dense.size), dtype=np.complex128)

    for idx, n_tot in enumerate(pole_counts):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m_au = fit_poles(
                f_au[mask_au_opt],
                eps_au[mask_au_opt],
                n_lorentz=int(n_tot - 1),
                n_drude=1,
                n_starts=n_starts,
                name=f"au_jc_N{n_tot}",
            )
            m_ag = fit_poles(
                f_ag[mask_ag_opt],
                eps_ag[mask_ag_opt],
                n_lorentz=int(n_tot - 1),
                n_drude=1,
                n_starts=n_starts,
                name=f"ag_jc_N{n_tot}",
            )
        au_errors[idx] = float(m_au.fit_error)
        ag_errors[idx] = float(m_ag.fit_error)
        au_max_dt[idx] = find_max_stable_dt(
            m_au, courant=0.5, dt_min=1e-18, dt_max=2e-15
        )
        ag_max_dt[idx] = find_max_stable_dt(
            m_ag, courant=0.5, dt_min=1e-18, dt_max=2e-15
        )
        au_eps_dense[idx] = m_au.epsilon(f_dense)
        ag_eps_dense[idx] = m_ag.epsilon(f_dense)
        max_im_au = float(np.max(np.imag(m_au.epsilon(f_passivity_check))))
        max_im_ag = float(np.max(np.imag(m_ag.epsilon(f_passivity_check))))
        au_fits[f"N{n_tot}"] = {
            "name": m_au.name,
            "n_poles": int(n_tot),
            "eps_inf": float(m_au.eps_inf),
            "fit_error": float(m_au.fit_error),
            "stable_50as": bool(m_au.is_stable(5.0e-17, 0.5)),
            "max_stable_dt_fs": float(au_max_dt[idx] * 1e15),
            "max_im_eps": max_im_au,
            "poles": serialize_poles(m_au),
        }
        ag_fits[f"N{n_tot}"] = {
            "name": m_ag.name,
            "n_poles": int(n_tot),
            "eps_inf": float(m_ag.eps_inf),
            "fit_error": float(m_ag.fit_error),
            "stable_50as": bool(m_ag.is_stable(5.0e-17, 0.5)),
            "max_stable_dt_fs": float(ag_max_dt[idx] * 1e15),
            "max_im_eps": max_im_ag,
            "poles": serialize_poles(m_ag),
        }

    # Fit-band choice study on Gold (Au) with 2 poles (1 Drude + 1 Lorentz):
    # Band A: Narrow red/NIR plasmonic band [0.55, 0.95] um
    # Band B: Visible-NIR band [0.38, 0.90] um
    # Band C: Full UV-to-NIR band [0.188, 1.937] um
    band_defs = {
        "narrow_red_nir": (0.55, 0.95),
        "visible_nir": (0.38, 0.90),
        "full_uv_nir": (0.187, 1.94),
    }
    band_results = {}
    band_eps_dense = np.zeros((3, f_dense.size), dtype=np.complex128)
    for b_idx, (b_key, (lmin, lmax)) in enumerate(band_defs.items()):
        b_mask = (lam_au >= lmin) & (lam_au <= lmax)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m_b = fit_poles(
                f_au[b_mask],
                eps_au[b_mask],
                n_lorentz=1,
                n_drude=1,
                n_starts=n_starts,
                name=f"au_band_{b_key}",
            )
        band_eps_dense[b_idx] = m_b.epsilon(f_dense)
        eps_all = m_b.epsilon(f_au)
        rel_all = np.abs(eps_all - eps_au) / np.abs(eps_au)
        in_err = float(np.sqrt(np.mean(rel_all[b_mask] ** 2)))
        out_mask = ~b_mask
        out_err = (
            float(np.sqrt(np.mean(rel_all[out_mask] ** 2)))
            if np.any(out_mask)
            else in_err
        )
        full_err = float(np.sqrt(np.mean(rel_all**2)))
        band_results[b_key] = {
            "lambda_range_um": [float(lmin), float(lmax)],
            "n_points": int(np.sum(b_mask)),
            "in_band_rel_rms": in_err,
            "out_of_band_rel_rms": out_err,
            "full_table_rel_rms": full_err,
            "max_im_eps_global": float(
                np.max(np.imag(m_b.epsilon(f_passivity_check)))
            ),
            "stable_50as": bool(m_b.is_stable(5.0e-17, 0.5)),
            "max_stable_dt_fs": float(
                find_max_stable_dt(m_b, courant=0.5, dt_min=1e-18, dt_max=2e-15)
                * 1e15
            ),
            "eps_inf": float(m_b.eps_inf),
            "poles": serialize_poles(m_b),
        }

    # Audit wakis.material_library
    categories = (
        "metal",
        "dielectric",
        "semiconductor",
        "polymer",
        "uniaxial",
        "rf_substrate",
    )
    cat_counts = {cat: len(list_materials(category=cat)) for cat in categories}
    featured_names = (
        "gold",
        "gold jc",
        "gold jc wide",
        "silver",
        "silver jc",
        "silver jc wide",
        "fused silica",
        "silicon",
        "gan o",
        "sapphire o",
        "fr4",
        "ro4350b",
    )
    library_audit = {}
    for mname in featured_names:
        meta = material_info(mname)
        library_audit[mname] = {
            "category": meta["category"],
            "model": meta["model"],
            "wavelength_range_um": list(meta["wavelength_range_um"]),
            "fit_error": meta["fit_error"],
            "stable_50as": meta["stable_50as"],
            "max_stable_dt_s": meta["max_stable_dt_s"],
            "reference": meta["citation"]["reference"],
            "doi": meta["citation"]["doi"],
        }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_DIR / "task_0.npz",
        lam_au=lam_au,
        f_au=f_au,
        n_au=n_au,
        k_au=k_au,
        eps_au=eps_au,
        lam_ag=lam_ag,
        f_ag=f_ag,
        n_ag=n_ag,
        k_ag=k_ag,
        eps_ag=eps_ag,
        pole_counts=pole_counts,
        au_errors=au_errors,
        ag_errors=ag_errors,
        au_max_dt=au_max_dt,
        ag_max_dt=ag_max_dt,
        f_dense=f_dense,
        lam_dense=c0 / f_dense * 1e6,
        au_eps_dense=au_eps_dense,
        ag_eps_dense=ag_eps_dense,
        band_eps_dense=band_eps_dense,
        rakic_au_eps_dense=dispersive_lib["gold"].epsilon(f_dense),
        rakic_ag_eps_dense=dispersive_lib["silver"].epsilon(f_dense),
    )

    elapsed = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    summary = {
        "task": 0,
        "name": "noble_metal_pole_convergence_and_bands",
        "quick": bool(quick),
        "elapsed_s": float(elapsed),
        "max_rss_mb": float(rss_mb),
        "au_pole_fits": au_fits,
        "ag_pole_fits": ag_fits,
        "au_band_study": band_results,
        "material_library_category_counts": cat_counts,
        "material_library_total_canonical": len(list_materials()),
        "material_library_total_with_aliases": len(
            list_materials(include_aliases=True)
        ),
        "material_library_featured": library_audit,
    }
    (DATA_DIR / "task_0.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def run_task_1(quick=False):
    """
    Task 1: Run 3D SolverFIT3D thin-film slab transmission/reflection
    simulations for Johnson & Christy Gold (Au, d = 40 nm) and Silver
    (Ag, d = 35 nm) across 0.36-0.89 um, comparing FIT R/T against the exact
    transfer matrix computed directly from the MEASURED (n, k) data as well as
    from the fitted N=1 (Drude) and N=3 (1 Drude + 2 Lorentz) pole models.
    """
    t0 = time.perf_counter()
    lam_au, f_au, _, _, eps_au = load_refractiveindex_nk(JC_AU_PATH)
    lam_ag, f_ag, _, _, eps_ag = load_refractiveindex_nk(JC_AG_PATH)

    n_starts = 3 if quick else 5
    mask_fit_au = (lam_au >= 0.34) & (lam_au <= 0.92)
    mask_fit_ag = (lam_ag >= 0.30) & (lam_ag <= 0.92)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mat_au_n1 = fit_poles(
            f_au[mask_fit_au],
            eps_au[mask_fit_au],
            n_lorentz=0,
            n_drude=1,
            n_starts=n_starts,
            name="au_fit_N1",
        )
        mat_au_n3 = fit_poles(
            f_au[mask_fit_au],
            eps_au[mask_fit_au],
            n_lorentz=2,
            n_drude=1,
            n_starts=n_starts,
            name="au_fit_N3",
        )
        mat_ag_n3 = fit_poles(
            f_ag[mask_fit_ag],
            eps_ag[mask_fit_ag],
            n_lorentz=2,
            n_drude=1,
            n_starts=n_starts,
            name="ag_fit_N3",
        )

    # Evaluate at the exact experimental Johnson & Christy points in [0.36, 0.90] um
    mask_eval_au = (lam_au >= 0.36) & (lam_au <= 0.90)
    f_eval_au = f_au[mask_eval_au]
    lam_eval_au = lam_au[mask_eval_au]
    eps_meas_au = eps_au[mask_eval_au]

    mask_eval_ag = (lam_ag >= 0.31) & (lam_ag <= 0.90)
    f_eval_ag = f_ag[mask_eval_ag]
    lam_eval_ag = lam_ag[mask_eval_ag]
    eps_meas_ag = eps_ag[mask_eval_ag]

    dz = 2.0e-9 if quick else 1.0e-9
    d_au = 40.0e-9
    d_ag = 36.0e-9
    lz = 240.0e-9
    t_after = 10.0e-15 if quick else 14.0e-15

    res_au_n1 = run_dispersive_slab_pair(
        mat_au_n1,
        f_eval_au,
        eps_meas_au,
        thickness=d_au,
        lz=lz,
        dz=dz,
        f0=560.0e12,
        fwidth=360.0e12,
        t_after_src=t_after,
        z_src=-50.0e-9,
        z_r=-25.0e-9,
        z_t=d_au + 25.0e-9,
    )
    res_au_n3 = run_dispersive_slab_pair(
        mat_au_n3,
        f_eval_au,
        eps_meas_au,
        thickness=d_au,
        lz=lz,
        dz=dz,
        f0=560.0e12,
        fwidth=360.0e12,
        t_after_src=t_after,
        z_src=-50.0e-9,
        z_r=-25.0e-9,
        z_t=d_au + 25.0e-9,
    )
    res_ag_n3 = run_dispersive_slab_pair(
        mat_ag_n3,
        f_eval_ag,
        eps_meas_ag,
        thickness=d_ag,
        lz=lz,
        dz=dz,
        f0=600.0e12,
        fwidth=400.0e12,
        t_after_src=t_after,
        z_src=-50.0e-9,
        z_r=-25.0e-9,
        z_t=d_ag + 25.0e-9,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_DIR / "task_1.npz",
        lam_eval_au=lam_eval_au,
        f_eval_au=f_eval_au,
        eps_meas_au=eps_meas_au,
        R_meas_au=res_au_n3["R_meas"],
        T_meas_au=res_au_n3["T_meas"],
        A_meas_au=res_au_n3["A_meas"],
        R_fit_au_n1=res_au_n1["R_fit"],
        T_fit_au_n1=res_au_n1["T_fit"],
        R_cont_au_n1=res_au_n1["R_cont"],
        T_cont_au_n1=res_au_n1["T_cont"],
        R_fit_au_n3=res_au_n3["R_fit"],
        T_fit_au_n3=res_au_n3["T_fit"],
        A_fit_au_n3=res_au_n3["A_fit"],
        R_cont_au_n3=res_au_n3["R_cont"],
        T_cont_au_n3=res_au_n3["T_cont"],
        R_disc_au_n3=res_au_n3["R_disc"],
        T_disc_au_n3=res_au_n3["T_disc"],
        lam_eval_ag=lam_eval_ag,
        f_eval_ag=f_eval_ag,
        eps_meas_ag=eps_meas_ag,
        R_meas_ag=res_ag_n3["R_meas"],
        T_meas_ag=res_ag_n3["T_meas"],
        A_meas_ag=res_ag_n3["A_meas"],
        R_fit_ag_n3=res_ag_n3["R_fit"],
        T_fit_ag_n3=res_ag_n3["T_fit"],
        A_fit_ag_n3=res_ag_n3["A_fit"],
        R_cont_ag_n3=res_ag_n3["R_cont"],
        T_cont_ag_n3=res_ag_n3["T_cont"],
        R_disc_ag_n3=res_ag_n3["R_disc"],
        T_disc_ag_n3=res_ag_n3["T_disc"],
    )

    elapsed = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    summary = {
        "task": 1,
        "name": "au_and_ag_slab_simulations",
        "quick": bool(quick),
        "elapsed_s": float(elapsed),
        "max_rss_mb": float(rss_mb),
        "dz_nm": float(dz * 1e9),
        "au_slab": {
            "thickness_nm": float(d_au * 1e9),
            "nt": res_au_n3["nt"],
            "dt_as": float(res_au_n3["dt"] * 1e18),
            "cells_across": res_au_n3["cells_across"],
            "n1_drude_only": {
                "eps_fit_error": float(mat_au_n1.fit_error),
                "rms_R_tot_vs_measured": res_au_n1["rms_R_tot"],
                "rms_T_tot_vs_measured": res_au_n1["rms_T_tot"],
                "rms_R_pole_fit_err": res_au_n1["rms_R_fit_err"],
                "rms_R_disc_err": res_au_n1["rms_R_disc_err"],
            },
            "n3_drude_2lorentz": {
                "eps_fit_error": float(mat_au_n3.fit_error),
                "rms_R_tot_vs_measured": res_au_n3["rms_R_tot"],
                "rms_T_tot_vs_measured": res_au_n3["rms_T_tot"],
                "rms_R_pole_fit_err": res_au_n3["rms_R_fit_err"],
                "rms_T_pole_fit_err": res_au_n3["rms_T_fit_err"],
                "rms_R_disc_err": res_au_n3["rms_R_disc_err"],
                "rms_T_disc_err": res_au_n3["rms_T_disc_err"],
            },
        },
        "ag_slab": {
            "thickness_nm": float(d_ag * 1e9),
            "nt": res_ag_n3["nt"],
            "dt_as": float(res_ag_n3["dt"] * 1e18),
            "cells_across": res_ag_n3["cells_across"],
            "eps_fit_error": float(mat_ag_n3.fit_error),
            "rms_R_tot_vs_measured": res_ag_n3["rms_R_tot"],
            "rms_T_tot_vs_measured": res_ag_n3["rms_T_tot"],
            "rms_R_pole_fit_err": res_ag_n3["rms_R_fit_err"],
            "rms_T_pole_fit_err": res_ag_n3["rms_T_fit_err"],
            "rms_R_disc_err": res_ag_n3["rms_R_disc_err"],
            "rms_T_disc_err": res_ag_n3["rms_T_disc_err"],
        },
    }
    (DATA_DIR / "task_1.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def run_task_2(quick=False):
    """
    Task 2: Fused silica (SiO2, Malitson 1965) Sellmeier dielectric formula vs.
    reduced 1-pole and 2-pole Lorentz fits (demonstrating the unconstrained
    eps_inf -> 0 ADE stability trap vs constrained eps_inf >= 1), plus a 3D
    SolverFIT3D Fabry-Perot etalon slab simulation (d = 600 nm) compared
    against the exact transfer matrix from Malitson's formula.
    """
    t0 = time.perf_counter()
    wl_range, eps_inf_mal, B_vals, C_um_vals = load_refractiveindex_sellmeier(
        SIO2_PATH
    )
    mat_sio2_lib = dispersive_lib["sio2"]

    lam_fit = np.linspace(0.25, 1.50, 50)[::-1]
    f_fit = c0 / (lam_fit * 1e-6)
    eps_mal_fit = sellmeier_eps_exact(lam_fit, eps_inf_mal, B_vals, C_um_vals)
    eps_lib_fit = mat_sio2_lib.epsilon(f_fit)
    lib_vs_yml_rel_diff = float(
        np.linalg.norm(eps_lib_fit - eps_mal_fit) / np.linalg.norm(eps_mal_fit)
    )

    # 1. Show unconstrained 2-pole fit on narrow visible window [0.40, 0.90] um
    # where eps_inf=None can drift below S^2 = 0.25 or place a pole at high UV frequency
    lam_narrow = np.linspace(0.35, 0.95, 25)[::-1]
    f_narrow = c0 / (lam_narrow * 1e-6)
    eps_narrow = sellmeier_eps_exact(lam_narrow, eps_inf_mal, B_vals, C_um_vals)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mat_uncon = fit_poles(
            f_narrow,
            eps_narrow,
            n_lorentz=2,
            n_drude=0,
            eps_inf=None,
            n_starts=4,
            name="sio2_unconstrained_narrow",
        )
        # 2. Constrained 1-pole fit with eps_inf = 1.0
        mat_fit1 = fit_poles(
            f_fit,
            eps_mal_fit,
            n_lorentz=1,
            n_drude=0,
            eps_inf=1.0,
            n_starts=4,
            name="sio2_1pole_eps1",
        )
        # 3. Constrained 2-pole fit folding the deep-UV 68.4 nm pole into eps_inf = 1 + B_1
        mat_fit2 = fit_poles(
            f_fit,
            eps_mal_fit,
            n_lorentz=2,
            n_drude=0,
            eps_inf=float(1.0 + B_vals[0]),
            n_starts=4,
            name="sio2_2pole_folded",
        )

    dt_max_lib = find_max_stable_dt(
        mat_sio2_lib, courant=0.5, dt_min=1e-18, dt_max=2e-15
    )
    dt_max_uncon = find_max_stable_dt(
        mat_uncon, courant=0.5, dt_min=1e-18, dt_max=2e-15
    )
    dt_max_fit1 = find_max_stable_dt(
        mat_fit1, courant=0.5, dt_min=1e-18, dt_max=2e-15
    )
    dt_max_fit2 = find_max_stable_dt(
        mat_fit2, courant=0.5, dt_min=1e-18, dt_max=2e-15
    )

    # Evaluate across full Malitson range 0.21-3.71 um
    lam_broad = np.linspace(0.21, 3.71, 200)[::-1]
    f_broad = c0 / (lam_broad * 1e-6)
    eps_mal_broad = sellmeier_eps_exact(
        lam_broad, eps_inf_mal, B_vals, C_um_vals
    )
    eps_fit1_broad = mat_fit1.epsilon(f_broad)
    eps_fit2_broad = mat_fit2.epsilon(f_broad)

    # Run 3D SolverFIT3D Fabry-Perot slab simulation (d = 600 nm)
    lam_eval = np.linspace(0.35, 0.95, 31)[::-1]
    f_eval = c0 / (lam_eval * 1e-6)
    eps_meas_eval = sellmeier_eps_exact(
        lam_eval, eps_inf_mal, B_vals, C_um_vals
    )
    d_sio2 = 600.0e-9
    dz = 10.0e-9 if quick else 5.0e-9
    t_after = 24.0e-15 if quick else 32.0e-15

    res_sio2 = run_dispersive_slab_pair(
        mat_fit2,
        f_eval,
        eps_meas_eval,
        thickness=d_sio2,
        lz=1600.0e-9,
        dz=dz,
        f0=550.0e12,
        fwidth=350.0e12,
        t_after_src=t_after,
        z_src=-200.0e-9,
        z_r=-100.0e-9,
        z_t=d_sio2 + 100.0e-9,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_DIR / "task_2.npz",
        lam_broad=lam_broad,
        f_broad=f_broad,
        eps_mal_broad=eps_mal_broad,
        eps_fit1_broad=eps_fit1_broad,
        eps_fit2_broad=eps_fit2_broad,
        lam_eval=lam_eval,
        f_eval=f_eval,
        eps_meas_eval=eps_meas_eval,
        R_meas=res_sio2["R_meas"],
        T_meas=res_sio2["T_meas"],
        R_fit=res_sio2["R_fit"],
        T_fit=res_sio2["T_fit"],
        R_cont=res_sio2["R_cont"],
        T_cont=res_sio2["T_cont"],
        R_disc=res_sio2["R_disc"],
        T_disc=res_sio2["T_disc"],
    )

    elapsed = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    summary = {
        "task": 2,
        "name": "sellmeier_sio2_fit_and_slab",
        "quick": bool(quick),
        "elapsed_s": float(elapsed),
        "max_rss_mb": float(rss_mb),
        "malitson_wavelength_range_um": list(wl_range),
        "lib_vs_yml_rel_diff": lib_vs_yml_rel_diff,
        "models": {
            "malitson_3pole_exact": {
                "eps_inf": float(mat_sio2_lib.eps_inf),
                "stable_50as": bool(mat_sio2_lib.is_stable(5.0e-17, 0.5)),
                "max_stable_dt_fs": float(dt_max_lib * 1e15),
            },
            "unconstrained_narrow_2pole": {
                "eps_inf": float(mat_uncon.eps_inf),
                "fit_error": float(mat_uncon.fit_error),
                "stable_50as": bool(mat_uncon.is_stable(5.0e-17, 0.5)),
                "max_stable_dt_fs": float(dt_max_uncon * 1e15),
            },
            "constrained_1pole_eps1": {
                "eps_inf": float(mat_fit1.eps_inf),
                "fit_error": float(mat_fit1.fit_error),
                "stable_50as": bool(mat_fit1.is_stable(5.0e-17, 0.5)),
                "max_stable_dt_fs": float(dt_max_fit1 * 1e15),
            },
            "constrained_2pole_folded": {
                "eps_inf": float(mat_fit2.eps_inf),
                "fit_error": float(mat_fit2.fit_error),
                "stable_50as": bool(mat_fit2.is_stable(5.0e-17, 0.5)),
                "max_stable_dt_fs": float(dt_max_fit2 * 1e15),
            },
        },
        "sio2_slab": {
            "thickness_nm": float(d_sio2 * 1e9),
            "dz_nm": float(dz * 1e9),
            "nt": res_sio2["nt"],
            "dt_as": float(res_sio2["dt"] * 1e18),
            "cells_across": res_sio2["cells_across"],
            "cpw_mat": res_sio2["cpw_mat"],
            "rms_R_tot_vs_measured": res_sio2["rms_R_tot"],
            "rms_T_tot_vs_measured": res_sio2["rms_T_tot"],
            "rms_R_pole_fit_err": res_sio2["rms_R_fit_err"],
            "rms_T_pole_fit_err": res_sio2["rms_T_fit_err"],
            "rms_R_disc_err": res_sio2["rms_R_disc_err"],
            "rms_T_disc_err": res_sio2["rms_T_disc_err"],
        },
    }
    (DATA_DIR / "task_2.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary



def run_task_3(quick=False):
    """
    Task 3: Fit N = 1..5 Debye poles + static ionic conductivity sigma_i to the
    Gabriel, Lau & Gabriel (1996) four-term Cole-Cole model over 10 MHz - 10 GHz
    for Muscle, Skin (Dry), Fat (Not Infiltrated), and Bone (Cortical).
    """
    t0 = time.perf_counter()
    f_rf = np.geomspace(10.0e6, 10.0e9, 60)
    n_starts = 3 if quick else 5
    debye_counts = np.arange(1, 6, dtype=int)
    tissue_keys = ("muscle", "skin_dry", "fat", "bone_cortical")

    err_matrix = np.zeros(
        (len(tissue_keys), debye_counts.size), dtype=np.float64
    )
    eps_cc_matrix = np.zeros(
        (len(tissue_keys), f_rf.size), dtype=np.complex128
    )
    eps_pol_cc_matrix = np.zeros(
        (len(tissue_keys), f_rf.size), dtype=np.complex128
    )
    eps_fit4_matrix = np.zeros(
        (len(tissue_keys), f_rf.size), dtype=np.complex128
    )
    eps_pol_fit4_matrix = np.zeros(
        (len(tissue_keys), f_rf.size), dtype=np.complex128
    )
    muscle_fits_vs_n = np.zeros(
        (debye_counts.size, f_rf.size), dtype=np.complex128
    )

    tissue_summary = {}
    for t_idx, tkey in enumerate(tissue_keys):
        eps_cc = cole_cole_permittivity(f_rf, tkey, include_sigma=True)
        eps_pol_cc = cole_cole_permittivity(f_rf, tkey, include_sigma=False)
        eps_cc_matrix[t_idx] = eps_cc
        eps_pol_cc_matrix[t_idx] = eps_pol_cc

        per_n = {}
        best_err_so_far = np.inf
        best_mat_so_far = None
        for n_idx, n_deb in enumerate(debye_counts):
            mat = fit_cole_cole_debye(
                f_rf, tkey, n_debye=int(n_deb), n_starts=n_starts
            )
            # Keep monotone non-increasing best fit across N
            if mat.fit_error > best_err_so_far and best_mat_so_far is not None:
                mat = best_mat_so_far
            else:
                best_err_so_far = mat.fit_error
                best_mat_so_far = mat
            err_matrix[t_idx, n_idx] = float(mat.fit_error)
            if tkey == "muscle":
                muscle_fits_vs_n[n_idx] = mat.epsilon(f_rf)
            if n_deb == 4:
                eps_fit4_matrix[t_idx] = mat.epsilon(f_rf)
                mat_pol_only = DispersiveMaterial(
                    mat.eps_inf, mat.poles, sigma=0.0
                )
                eps_pol_fit4_matrix[t_idx] = mat_pol_only.epsilon(f_rf)
            per_n[f"N{n_deb}"] = {
                "n_debye": int(n_deb),
                "eps_inf": float(mat.eps_inf),
                "sigma_S_m": float(mat.sigma),
                "rel_rms_error": float(mat.fit_error),
                "pol_rel_rms_error": float(mat.pol_fit_error),
                "stable_10ps": bool(mat.is_stable(1.0e-11, 0.5)),
                "max_im_eps": float(np.max(np.imag(mat.epsilon(f_rf)))),
                "poles": serialize_poles(mat),
            }
        tissue_summary[tkey] = {
            "label": GABRIEL_1996_TISSUES[tkey]["label"],
            "gabriel_eps_inf": GABRIEL_1996_TISSUES[tkey]["eps_inf"],
            "gabriel_sigma_i_S_m": GABRIEL_1996_TISSUES[tkey]["sigma_i"],
            "fits_10MHz_to_10GHz": per_n,
        }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_DIR / "task_3.npz",
        f_rf=f_rf,
        debye_counts=debye_counts,
        tissue_keys=np.asarray(tissue_keys),
        err_matrix=err_matrix,
        eps_cc_matrix=eps_cc_matrix,
        eps_pol_cc_matrix=eps_pol_cc_matrix,
        eps_fit4_matrix=eps_fit4_matrix,
        eps_pol_fit4_matrix=eps_pol_fit4_matrix,
        muscle_fits_vs_n=muscle_fits_vs_n,
    )

    elapsed = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    summary = {
        "task": 3,
        "name": "gabriel_cole_cole_tissue_fits",
        "quick": bool(quick),
        "elapsed_s": float(elapsed),
        "max_rss_mb": float(rss_mb),
        "fit_band_Hz": [1.0e7, 1.0e10],
        "citation": (
            "S. Gabriel, R. W. Lau, and C. Gabriel, 'The dielectric properties "
            "of biological tissues: III. Parametric models for the dielectric "
            "spectrum of tissues,' Phys. Med. Biol. 41(11), 2271-2293 (1996), "
            "doi:10.1088/0031-9155/41/11/003"
        ),
        "tissues": tissue_summary,
    }
    (DATA_DIR / "task_3.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def run_task_4(quick=False):
    """
    Task 4: Run a 3D SolverFIT3D microwave slab simulation of human muscle
    tissue (d = 8.0 mm) across 0.6-6.0 GHz using both the N=1 and N=4 Debye +
    sigma_i fits, comparing FIT R/T/A against the exact transfer matrix from
    Gabriel's 4-term Cole-Cole formula.
    """
    t0 = time.perf_counter()
    f_rf = np.geomspace(10.0e6, 10.0e9, 60)
    n_starts = 3 if quick else 5
    mat_mus_n1 = fit_cole_cole_debye(
        f_rf, "muscle", n_debye=1, n_starts=n_starts
    )
    mat_mus_n4 = fit_cole_cole_debye(
        f_rf, "muscle", n_debye=4, n_starts=n_starts
    )

    f_eval = np.linspace(0.6e9, 6.0e9, 25)
    eps_meas_cc = cole_cole_permittivity(f_eval, "muscle", include_sigma=True)
    d_mus = 8.0e-3
    dz = 0.2e-3 if quick else 0.1e-3
    lz = 32.0e-3
    t_after = 1.1e-9 if quick else 1.5e-9

    res_n1 = run_dispersive_slab_pair(
        mat_mus_n1,
        f_eval,
        eps_meas_cc,
        thickness=d_mus,
        lz=lz,
        dz=dz,
        f0=3.2e9,
        fwidth=2.5e9,
        t_after_src=t_after,
        z_src=-6.0e-3,
        z_r=-3.0e-3,
        z_t=d_mus + 3.0e-3,
    )
    res_n4 = run_dispersive_slab_pair(
        mat_mus_n4,
        f_eval,
        eps_meas_cc,
        thickness=d_mus,
        lz=lz,
        dz=dz,
        f0=3.2e9,
        fwidth=2.5e9,
        t_after_src=t_after,
        z_src=-6.0e-3,
        z_r=-3.0e-3,
        z_t=d_mus + 3.0e-3,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_DIR / "task_4.npz",
        f_eval=f_eval,
        eps_meas_cc=eps_meas_cc,
        R_meas_cc=res_n4["R_meas"],
        T_meas_cc=res_n4["T_meas"],
        A_meas_cc=res_n4["A_meas"],
        R_fit_n1=res_n1["R_fit"],
        T_fit_n1=res_n1["T_fit"],
        A_fit_n1=res_n1["A_fit"],
        R_cont_n1=res_n1["R_cont"],
        T_cont_n1=res_n1["T_cont"],
        R_fit_n4=res_n4["R_fit"],
        T_fit_n4=res_n4["T_fit"],
        A_fit_n4=res_n4["A_fit"],
        R_cont_n4=res_n4["R_cont"],
        T_cont_n4=res_n4["T_cont"],
        R_disc_n4=res_n4["R_disc"],
        T_disc_n4=res_n4["T_disc"],
    )

    elapsed = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    summary = {
        "task": 4,
        "name": "muscle_tissue_slab_simulation",
        "quick": bool(quick),
        "elapsed_s": float(elapsed),
        "max_rss_mb": float(rss_mb),
        "thickness_mm": float(d_mus * 1e3),
        "dz_mm": float(dz * 1e3),
        "nt": res_n4["nt"],
        "dt_ps": float(res_n4["dt"] * 1e12),
        "cells_across": res_n4["cells_across"],
        "cpw_mat": res_n4["cpw_mat"],
        "muscle_n1_debye": {
            "eps_fit_error": float(mat_mus_n1.fit_error),
            "rms_R_tot_vs_cole_cole": res_n1["rms_R_tot"],
            "rms_T_tot_vs_cole_cole": res_n1["rms_T_tot"],
            "rms_R_pole_fit_err": res_n1["rms_R_fit_err"],
            "rms_T_pole_fit_err": res_n1["rms_T_fit_err"],
            "rms_R_disc_err": res_n1["rms_R_disc_err"],
            "rms_T_disc_err": res_n1["rms_T_disc_err"],
        },
        "muscle_n4_debye": {
            "eps_fit_error": float(mat_mus_n4.fit_error),
            "rms_R_tot_vs_cole_cole": res_n4["rms_R_tot"],
            "rms_T_tot_vs_cole_cole": res_n4["rms_T_tot"],
            "rms_R_pole_fit_err": res_n4["rms_R_fit_err"],
            "rms_T_pole_fit_err": res_n4["rms_T_fit_err"],
            "rms_R_disc_err": res_n4["rms_R_disc_err"],
            "rms_T_disc_err": res_n4["rms_T_disc_err"],
        },
    }
    (DATA_DIR / "task_4.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary



# %%
def _plot_setup_schematic(fig_dir):
    """Render ``00a_setup_geometry.png``, ``00b_setup_mesh_and_excitation.png``, and ``00_setup_table.md``."""
    lam_au, f_au, _, _, _ = load_refractiveindex_nk(JC_AU_PATH)
    f_eval_au = f_au[(lam_au >= 0.36) & (lam_au <= 0.90)]

    dz_opt = 1.0e-9
    lz_opt = 240.0e-9
    d_au = 40.0e-9
    nz_opt = int(round(lz_opt / dz_opt))
    zmin_opt = -0.45 * lz_opt
    zmax_opt = zmin_opt + nz_opt * dz_opt
    dxy_opt = 4.0 * dz_opt
    grid_opt = GridFIT3D(
        -dxy_opt,
        dxy_opt,
        -dxy_opt,
        dxy_opt,
        zmin_opt,
        zmax_opt,
        2,
        2,
        nz_opt,
        verbose=0,
    )
    solver_opt = SolverFIT3D(
        grid_opt,
        cfln=0.5,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=15,
        verbose=0,
    )
    mat_au = dispersive_lib["gold jc"]
    solver_opt.add_dispersive_material(
        mat_au,
        center=(0.0, 0.0, 0.5 * d_au),
        size=(10.0 * dxy_opt, 10.0 * dxy_opt, d_au),
    )
    opt_struct = SimpleNamespace(
        bounds=((-dxy_opt, -dxy_opt, 0.0), (dxy_opt, dxy_opt, d_au)),
        name="Au thin film (J&C fit)",
        material=mat_au,
    )
    src_opt = CurrentSource(
        GaussianSource(frequency=560.0e12, fwidth=360.0e12),
        component="x",
        center=(0.0, 0.0, -50.0e-9),
        size=(2.0 * dxy_opt, 2.0 * dxy_opt, 0.0),
    )
    mons_opt = [
        FluxMonitor(
            f_eval_au,
            center=(0.0, 0.0, -25.0e-9),
            size=(2.0 * dxy_opt, 2.0 * dxy_opt, 0.0),
        ),
        FluxMonitor(
            f_eval_au,
            center=(0.0, 0.0, d_au + 25.0e-9),
            size=(2.0 * dxy_opt, 2.0 * dxy_opt, 0.0),
        ),
    ]
    return schematic.save_setup_figures(
        solver_opt,
        fig_dir,
        plane="zx",
        structures=[opt_struct],
        sources=[src_opt],
        monitors=mons_opt,
        min_feature=d_au,
        title_geometry="Quasi-1D fitted-material slab R/T setup: geometry, source, monitors & boundaries (ZX plane)",
        title_mesh="Quasi-1D fitted-material slab setup: mesh resolution, wave packet & broadband pulse",
        variations=[
            {
                "name": "Task 0 & Task 3 · Pole fitting & library audit",
                "details": "Non-FIT nonlinear least-squares pole fitting (fit_poles N = 1..6 on J&C Au/Ag; N = 1..5 Debye on Gabriel 1996 tissues)",
            },
            {
                "name": "Task 1b · Silver thin film (Ag, J&C 1972)",
                "details": "Ag slab (1 Drude + 2 Lorentz fit, d = 36 nm, Lz = 240 nm, Δz = 1.0 nm, λ ∈ [310, 900] nm)",
            },
            {
                "name": "Task 2 · Fused-silica etalon (SiO₂, Malitson 1965)",
                "details": "SiO₂ Fabry–Pérot slab (2-pole folded Lorentz fit, ε_∞ = 1.696, d = 600 nm, Lz = 1.60 µm, Δz = 5.0 nm, λ ∈ [350, 950] nm)",
            },
            {
                "name": "Task 4 · Human muscle tissue slab (Gabriel 1996)",
                "details": "Muscle tissue slab (4-pole Debye + σ_i = 0.20 S/m fit, d = 8.0 mm, Lz = 32.0 mm, Δz = 0.10 mm, f ∈ [0.6, 6.0] GHz)",
            },
        ],
    )


def plot_all():
    """Generate setup schematic and all 8 figures from saved ``data/task_*.npz`` and ``.json``."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figstyle.apply_style()
    _plot_setup_schematic(FIG_DIR)

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
    # Figure 1: Noble-metal pole convergence (N = 1..6) on J&C Au and Ag
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.8, height=4.4)
    lam_d = d0["lam_dense"]
    mask_p = (lam_d >= 0.25) & (lam_d <= 1.20)
    lam_au = d0["lam_au"]
    m_au_p = (lam_au >= 0.25) & (lam_au <= 1.20)

    axes[0].plot(
        lam_au[m_au_p] * 1e3,
        np.real(d0["eps_au"][m_au_p]),
        "ko",
        ms=4.2,
        label="J&C (1972) measured",
    )
    for s_idx, (idx, n_tot) in enumerate([(0, 1), (1, 2), (2, 3), (5, 6)]):
        axes[0].plot(
            lam_d[mask_p] * 1e3,
            np.real(d0["au_eps_dense"][idx, mask_p]),
            **figstyle.series_style(s_idx, markevery=85, ms=3.4, lw=1.8),
            label=f"$N={n_tot}$ (err={d0['au_errors'][idx]*100:.1f}%)",
        )
    axes[0].axvspan(300, 1000, color="#E2E8F0", alpha=0.5, zorder=0, label="Fit band")
    axes[0].set_xlabel(r"Vacuum wavelength $\lambda_0$ [nm]")
    axes[0].set_ylabel(r"$\mathrm{Re}\,\varepsilon(\lambda)$")
    axes[0].set_title(r"(a) Gold ($\mathrm{Au}$): $\mathrm{Re}\,\varepsilon(\lambda)$")
    axes[0].set_ylim(-45, 5)
    axes[0].legend(fontsize=7.5, loc="lower left")

    axes[1].plot(
        lam_au[m_au_p] * 1e3,
        -np.imag(d0["eps_au"][m_au_p]),
        "ko",
        ms=4.2,
        label="J&C (1972) measured",
    )
    for s_idx, (idx, n_tot) in enumerate([(0, 1), (1, 2), (2, 3), (5, 6)]):
        axes[1].plot(
            lam_d[mask_p] * 1e3,
            -np.imag(d0["au_eps_dense"][idx, mask_p]),
            **figstyle.series_style(s_idx, markevery=85, ms=3.4, lw=1.8),
            label=f"$N={n_tot}$ (1D+{n_tot-1}L)",
        )
    axes[1].axvline(550, color="#64748B", ls=":", lw=1.0)
    axes[1].text(
        560, 5.2, "5d $\\to$ 6sp\ninterband edge", fontsize=7.8, color="#334155"
    )
    axes[1].set_xlabel(r"Vacuum wavelength $\lambda_0$ [nm]")
    axes[1].set_ylabel(r"$-\mathrm{Im}\,\varepsilon(\lambda) \geq 0$")
    axes[1].set_title(
        r"(b) Gold ($\mathrm{Au}$): $-\mathrm{Im}\,\varepsilon(\lambda)$"
    )
    axes[1].set_ylim(0, 7.0)
    axes[1].legend(fontsize=7.5, loc="upper right")

    n_arr = d0["pole_counts"]
    axes[2].semilogy(
        n_arr,
        d0["au_errors"] * 100.0,
        **figstyle.series_style(0),
        label="Au (0.30–1.00 $\\mu$m)",
    )
    axes[2].semilogy(
        n_arr,
        d0["ag_errors"] * 100.0,
        **figstyle.series_style(2),
        label="Ag (0.30–1.00 $\\mu$m)",
    )
    ax2_t = axes[2].twinx()
    ax2_t._is_inset = True
    ax2_t.plot(
        n_arr,
        d0["au_max_dt"] * 1e18,
        "^--",
        color=figstyle.PALETTE[4],
        alpha=0.85,
        label=r"Au $\Delta t_{\max}$ [as]",
    )
    ax2_t.axhline(50.0, color="#D55E00", ls=":", lw=1.0, label="Ref. 50 as")
    ax2_t.set_ylabel(r"Max stable ADE step $\Delta t_{\max}$ [as]", color=figstyle.PALETTE[4])
    axes[2].set_xlabel("Total poles $N$ (1 Drude + $(N-1)$ Lorentz)")
    axes[2].set_ylabel("Relative RMS fit error [%]")
    axes[2].set_title("(c) Error vs. pole count $N$ & stability")
    axes[2].set_xticks(n_arr)
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2_t.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc="center right")

    figstyle.save(fig, FIG_DIR / "01_noble_metal_pole_convergence.png")

    # ------------------------------------------------------------------
    # Figure 2: Choosing the fit band & out-of-band extrapolation (Au)
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.8, height=4.3)
    axes[0].plot(
        lam_au * 1e3,
        -np.imag(d0["eps_au"]),
        "ko",
        ms=3.8,
        label="J&C (1972) measured",
    )
    b_styles = [
        (0, "narrow_red_nir", "Narrow [550, 950] nm"),
        (1, "visible_nir", "Vis-NIR [380, 900] nm"),
        (2, "full_uv_nir", "Full [188, 1937] nm"),
    ]
    for b_idx, _b_key, b_label in b_styles:
        axes[0].plot(
            lam_d * 1e3,
            -np.imag(d0["band_eps_dense"][b_idx]),
            **figstyle.series_style(b_idx, markevery=90, ms=3.4, lw=1.8),
            label=b_label,
        )
    axes[0].axvspan(550, 950, color="#CC79A7", alpha=0.14, zorder=0)
    axes[0].set_xlim(180, 1400)
    axes[0].set_ylim(0, 8.0)
    axes[0].set_xlabel(r"Vacuum wavelength $\lambda_0$ [nm]")
    axes[0].set_ylabel(r"$-\mathrm{Im}\,\varepsilon(\lambda) \geq 0$")
    axes[0].set_title(r"(a) Out-of-band extrapolation in $-\mathrm{Im}\,\varepsilon$")
    axes[0].legend(fontsize=7.6)

    axes[1].plot(
        lam_au * 1e3,
        np.real(d0["eps_au"]),
        "ko",
        ms=3.8,
        label="J&C (1972) measured",
    )
    for b_idx, _b_key, b_label in b_styles:
        axes[1].plot(
            lam_d * 1e3,
            np.real(d0["band_eps_dense"][b_idx]),
            **figstyle.series_style(b_idx, markevery=90, ms=3.4, lw=1.8),
            label=b_label,
        )
    axes[1].set_xlim(180, 1400)
    axes[1].set_ylim(-95, 10)
    axes[1].set_xlabel(r"Vacuum wavelength $\lambda_0$ [nm]")
    axes[1].set_ylabel(r"$\mathrm{Re}\,\varepsilon(\lambda)$")
    axes[1].set_title(r"(b) Out-of-band extrapolation in $\mathrm{Re}\,\varepsilon$")
    axes[1].legend(fontsize=7.6)

    b_keys = ["narrow_red_nir", "visible_nir", "full_uv_nir"]
    b_names = ["Narrow\n[0.55, 0.95] $\\mu$m", "Vis-NIR\n[0.38, 0.90] $\\mu$m", "Full UV-NIR\n[0.19, 1.94] $\\mu$m"]
    in_errs = [j0["au_band_study"][k]["in_band_rel_rms"] * 100 for k in b_keys]
    out_errs = [j0["au_band_study"][k]["out_of_band_rel_rms"] * 100 for k in b_keys]
    full_errs = [j0["au_band_study"][k]["full_table_rel_rms"] * 100 for k in b_keys]
    x_pos = np.arange(len(b_keys))
    w_bar = 0.25
    axes[2].bar(x_pos - w_bar, in_errs, width=w_bar, color=figstyle.PALETTE[2], label="In-band RMS [%]")
    axes[2].bar(x_pos, out_errs, width=w_bar, color=figstyle.PALETTE[1], label="Out-of-band RMS [%]")
    axes[2].bar(x_pos + w_bar, full_errs, width=w_bar, color=figstyle.PALETTE[0], label="Full-table RMS [%]")
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(b_names, fontsize=8.0)
    axes[2].set_ylabel("Relative RMS error [%]")
    axes[2].set_title("(c) In-band accuracy vs. out-of-band divergence")
    axes[2].legend(fontsize=7.6)

    figstyle.save(fig, FIG_DIR / "02_fit_band_extrapolation_and_library.png")

    # ------------------------------------------------------------------
    # Figure 3: Gold (Au) 40 nm thin-film FIT simulation vs Measured J&C
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=12.4, height=4.4)
    lam_au_ev = d1["lam_eval_au"] * 1e3
    ord_au = np.argsort(lam_au_ev)
    L_au = lam_au_ev[ord_au]

    axes[0].plot(L_au, d1["R_meas_au"][ord_au], "ko-", ms=4.2, lw=1.6, label=r"$R_{\mathrm{meas}}$ (TM from J&C data)")
    axes[0].plot(L_au, d1["R_cont_au_n3"][ord_au], **figstyle.series_style(2, markevery=6, ms=3.6), label=r"$R_{\mathrm{cont}}$ ($N=3$ pole fit)")
    axes[0].plot(L_au, d1["R_fit_au_n3"][ord_au], "s", color=figstyle.PALETTE[1], ms=4.8, label=r"$R_{\mathrm{FIT}}$ ($N=3$ Wakis 3D)")
    axes[0].plot(L_au, d1["R_fit_au_n1"][ord_au], "^:", color=figstyle.PALETTE[4], ms=4.2, lw=1.4, label=r"$R_{\mathrm{FIT}}$ ($N=1$ Drude only)")

    axes[0].plot(L_au, d1["T_meas_au"][ord_au], "kD--", ms=3.6, lw=1.2, alpha=0.75, label=r"$T_{\mathrm{meas}}$ (TM from J&C data)")
    axes[0].plot(L_au, d1["T_fit_au_n3"][ord_au], **figstyle.series_style(0, markevery=4, ms=4.0), label=r"$T_{\mathrm{FIT}}$ ($N=3$ Wakis 3D)")
    axes[0].set_xlabel(r"Vacuum wavelength $\lambda_0$ [nm]")
    axes[0].set_ylabel("Reflectance $R$, Transmittance $T$")
    axes[0].set_title(r"(a) Gold film ($d = 40$ nm): FIT vs. Measured J&C data")
    axes[0].legend(fontsize=7.5, ncol=2)

    err_tot_n1 = np.abs(d1["R_fit_au_n1"][ord_au] - d1["R_meas_au"][ord_au])
    err_tot_n3 = np.abs(d1["R_fit_au_n3"][ord_au] - d1["R_meas_au"][ord_au])
    err_fit_n3 = np.abs(d1["R_cont_au_n3"][ord_au] - d1["R_meas_au"][ord_au])
    err_disc_n3 = np.abs(d1["R_fit_au_n3"][ord_au] - d1["R_disc_au_n3"][ord_au])

    axes[1].semilogy(L_au, err_tot_n1, **figstyle.series_style(3, markevery=3), label=r"Total $|R_{\mathrm{FIT}} - R_{\mathrm{meas}}|$ ($N=1$ Drude)")
    axes[1].semilogy(L_au, err_tot_n3, **figstyle.series_style(1, markevery=3), label=r"Total $|R_{\mathrm{FIT}} - R_{\mathrm{meas}}|$ ($N=3$ fit)")
    axes[1].semilogy(L_au, err_fit_n3, **figstyle.series_style(2, markevery=3), label=r"Pole-fit error $|R_{\mathrm{cont}} - R_{\mathrm{meas}}|$ ($N=3$)")
    axes[1].semilogy(L_au, err_disc_n3, **figstyle.series_style(0, markevery=3), label=r"Discretization error $|R_{\mathrm{FIT}} - R_{\mathrm{disc}}|$ ($N=3$)")
    axes[1].set_xlabel(r"Vacuum wavelength $\lambda_0$ [nm]")
    axes[1].set_ylabel(r"Absolute reflectance error $|\Delta R|$")
    axes[1].set_title(r"(b) Error decomposition: $\Delta R_{\mathrm{tot}} = \Delta R_{\mathrm{fit}} + \Delta R_{\mathrm{disc}}$")
    axes[1].legend(fontsize=7.6)

    figstyle.save(fig, FIG_DIR / "03_au_slab_fit_vs_measured.png")

    # ------------------------------------------------------------------
    # Figure 4: Silver (Ag) 36 nm film FIT simulation vs Measured J&C
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=12.4, height=4.3)
    lam_ag_ev = d1["lam_eval_ag"] * 1e3
    ord_ag = np.argsort(lam_ag_ev)
    L_ag = lam_ag_ev[ord_ag]

    axes[0].plot(L_ag, d1["R_meas_ag"][ord_ag], "ko-", ms=4.2, lw=1.5, label=r"$R_{\mathrm{meas}}$ (J&C Ag data)")
    axes[0].plot(L_ag, d1["R_cont_ag_n3"][ord_ag], **figstyle.series_style(2, markevery=5, ms=3.6), label=r"$R_{\mathrm{cont}}$ ($N=3$ fit)")
    axes[0].plot(L_ag, d1["R_fit_ag_n3"][ord_ag], "s", color=figstyle.PALETTE[1], ms=4.8, label=r"$R_{\mathrm{FIT}}$ (Wakis 3D)")
    axes[0].plot(L_ag, d1["T_meas_ag"][ord_ag], "kD--", ms=3.8, lw=1.3, alpha=0.75, label=r"$T_{\mathrm{meas}}$ (J&C Ag data)")
    axes[0].plot(L_ag, d1["T_fit_ag_n3"][ord_ag], **figstyle.series_style(0, markevery=4, ms=4.0), label=r"$T_{\mathrm{FIT}}$ (Wakis 3D)")
    axes[0].axvline(326, color="#64748B", ls=":", lw=1.1)
    axes[0].text(335, 0.58, "Bulk plasmon\ntransparency peak\n($\\mathrm{Re}\\,\\varepsilon \\approx 0$)", fontsize=7.8)
    axes[0].set_xlabel(r"Vacuum wavelength $\lambda_0$ [nm]")
    axes[0].set_ylabel("Reflectance $R$, Transmittance $T$")
    axes[0].set_title(r"(a) Silver film ($d = 36$ nm): UV transparency peak & mirror edge")
    axes[0].legend(fontsize=7.6)

    axes[1].semilogy(
        L_ag,
        np.abs(d1["R_fit_ag_n3"][ord_ag] - d1["R_meas_ag"][ord_ag]),
        **figstyle.series_style(1, markevery=3),
        label=r"Total $|R_{\mathrm{FIT}} - R_{\mathrm{meas}}|$",
    )
    axes[1].semilogy(
        L_ag,
        np.abs(d1["R_cont_ag_n3"][ord_ag] - d1["R_meas_ag"][ord_ag]),
        **figstyle.series_style(2, markevery=3),
        label=r"Pole-fit $|R_{\mathrm{cont}} - R_{\mathrm{meas}}|$",
    )
    axes[1].semilogy(
        L_ag,
        np.abs(d1["R_fit_ag_n3"][ord_ag] - d1["R_disc_ag_n3"][ord_ag]),
        **figstyle.series_style(0, markevery=3),
        label=r"Discretization $|R_{\mathrm{FIT}} - R_{\mathrm{disc}}|$",
    )
    axes[1].set_xlabel(r"Vacuum wavelength $\lambda_0$ [nm]")
    axes[1].set_ylabel(r"Absolute reflectance error $|\Delta R|$")
    axes[1].set_title(r"(b) Silver slab error decomposition ($N=3$ fit)")
    axes[1].legend(fontsize=7.8)

    figstyle.save(fig, FIG_DIR / "04_ag_slab_fit_vs_measured.png")

    # ------------------------------------------------------------------
    # Figure 5: Fused silica (SiO2, Malitson 1965) Sellmeier & etalon slab
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=12.4, height=4.3)
    lam_b = d2["lam_broad"]
    ord_b = np.argsort(lam_b)
    axes[0].plot(
        lam_b[ord_b],
        np.sqrt(np.real(d2["eps_mal_broad"][ord_b])),
        **figstyle.series_style(0, markevery=60, ms=3.4),
        label="Malitson (1965) 3-term Sellmeier",
    )
    axes[0].plot(
        lam_b[ord_b],
        np.sqrt(np.real(d2["eps_fit2_broad"][ord_b])),
        **figstyle.series_style(2, markevery=60, ms=3.4),
        label=r"2-pole fit ($\varepsilon_\infty = 1 + B_1$, $\Delta t_{\max} = $"
        + f"{j2['models']['constrained_2pole_folded']['max_stable_dt_fs']*1e3:.0f} as)",
    )
    axes[0].plot(
        lam_b[ord_b],
        np.sqrt(np.real(d2["eps_fit1_broad"][ord_b])),
        **figstyle.series_style(1, markevery=60, ms=3.4),
        label=r"1-pole fit ($\varepsilon_\infty = 1$, UV only)",
    )
    axes[0].axvspan(0.25, 1.50, color="#E2E8F0", alpha=0.5, zorder=0, label="Fit band [0.25, 1.50] $\\mu$m")
    axes[0].set_xlabel(r"Vacuum wavelength $\lambda_0$ [$\mu$m]")
    axes[0].set_ylabel(r"Refractive index $n(\lambda) = \sqrt{\varepsilon(\lambda)}$")
    axes[0].set_title(r"(a) Fused silica ($\mathrm{SiO}_2$): Sellmeier vs. reduced pole fits")
    axes[0].legend(fontsize=7.6)

    lam_s = d2["lam_eval"] * 1e3
    ord_s = np.argsort(lam_s)
    axes[1].plot(
        lam_s[ord_s],
        d2["R_meas"][ord_s],
        **figstyle.series_style(0, markevery=8, ms=3.6),
        label=r"$R_{\mathrm{meas}}$ (Malitson Sellmeier TM)",
    )
    axes[1].plot(
        lam_s[ord_s],
        d2["R_fit"][ord_s],
        "o",
        color=figstyle.PALETTE[1],
        ms=4.5,
        label=r"$R_{\mathrm{FIT}}$ (Wakis 3D, $d=600$ nm)",
    )
    ax1_r = axes[1].twinx()
    ax1_r._is_inset = True
    ax1_r.plot(
        lam_s[ord_s],
        d2["T_meas"][ord_s],
        "--",
        color=figstyle.PALETTE[2],
        lw=1.6,
        label=r"$T_{\mathrm{meas}}$ (Malitson TM)",
    )
    ax1_r.plot(
        lam_s[ord_s],
        d2["T_fit"][ord_s],
        "s",
        color=figstyle.PALETTE[4],
        ms=4.0,
        label=r"$T_{\mathrm{FIT}}$ (Wakis 3D)",
    )
    ax1_r.set_ylabel("Transmittance $T$", color=figstyle.PALETTE[2])
    axes[1].set_xlabel(r"Vacuum wavelength $\lambda_0$ [nm]")
    axes[1].set_ylabel("Reflectance $R$")
    axes[1].set_title(
        f"(b) $\\mathrm{{SiO}}_2$ etalon ($d=600$ nm): RMS($\\Delta R$) = {j2['sio2_slab']['rms_R_tot_vs_measured']:.2e}"
    )
    l1, lb1 = axes[1].get_legend_handles_labels()
    l2, lb2 = ax1_r.get_legend_handles_labels()
    axes[1].legend(l1 + l2, lb1 + lb2, fontsize=7.6, loc="center right")

    figstyle.save(fig, FIG_DIR / "05_sio2_sellmeier_and_slab.png")

    # ------------------------------------------------------------------
    # Figure 6: Gabriel et al. (1996) 4-term Cole-Cole vs multi-pole Debye
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.8, height=4.4)
    f_rf_ghz = d3["f_rf"] / 1e9
    t_labels = ["Muscle", "Skin (Dry)", "Fat", "Bone (Cortical)"]

    for i, lbl in enumerate(t_labels):
        col = figstyle.PALETTE[i]
        axes[0].semilogx(
            f_rf_ghz,
            np.real(d3["eps_cc_matrix"][i]),
            **figstyle.series_style(i, markevery=30, ms=3.2),
            label=f"{lbl} (Cole–Cole)",
        )
        axes[0].semilogx(
            f_rf_ghz[::4],
            np.real(d3["eps_fit4_matrix"][i, ::4]),
            "o",
            color=col,
            ms=3.6,
        )
        axes[1].loglog(
            f_rf_ghz,
            -np.imag(d3["eps_cc_matrix"][i]),
            **figstyle.series_style(i, markevery=30, ms=3.2),
            label=f"{lbl} (Cole–Cole)",
        )
        axes[1].loglog(
            f_rf_ghz[::4],
            -np.imag(d3["eps_fit4_matrix"][i, ::4]),
            "o",
            color=col,
            ms=3.6,
        )

    axes[0].set_xlabel("Frequency $f$ [GHz]")
    axes[0].set_ylabel(r"$\mathrm{Re}\,\varepsilon(f)$")
    axes[0].set_title(r"(a) Gabriel (1996) $\mathrm{Re}\,\varepsilon$: exact vs $N=4$ Debye")
    axes[0].legend(fontsize=7.6)

    axes[1].set_xlabel("Frequency $f$ [GHz]")
    axes[1].set_ylabel(r"$-\mathrm{Im}\,\varepsilon(f) = \varepsilon''(f) + \sigma_i / (\omega\varepsilon_0)$")
    axes[1].set_title(r"(b) Total loss factor $-\mathrm{Im}\,\varepsilon(f)$ (10 MHz – 10 GHz)")
    axes[1].legend(fontsize=7.6)

    re_pol_cc = np.real(d3["eps_pol_cc_matrix"][0])
    im_pol_cc = -np.imag(d3["eps_pol_cc_matrix"][0])
    re_pol_d4 = np.real(d3["eps_pol_fit4_matrix"][0])
    im_pol_d4 = -np.imag(d3["eps_pol_fit4_matrix"][0])
    axes[2].plot(re_pol_cc, im_pol_cc, **figstyle.series_style(0, markevery=20, ms=3.4), label=r"Exact Cole–Cole ($\alpha_1=\alpha_2=0.10$)")
    axes[2].plot(re_pol_d4, im_pol_d4, **figstyle.series_style(1, markevery=10, ms=3.8), label="$N=4$ Debye cascade")
    axes[2].set_xlim(4.0, 62.0)
    axes[2].set_ylim(0.0, 28.0)
    axes[2].set_xlabel(r"$\mathrm{Re}\,\varepsilon_{\mathrm{pol}}(\omega)$")
    axes[2].set_ylabel(r"$-\mathrm{Im}\,\varepsilon_{\mathrm{pol}}(\omega)$")
    axes[2].set_title("(c) Muscle Cole–Cole Argand arc ($\\gamma$-dispersion)")
    axes[2].legend(fontsize=7.6)

    figstyle.save(fig, FIG_DIR / "06_gabriel_cole_cole_tissues.png")

    # ------------------------------------------------------------------
    # Figure 7: Tissue Debye fit error vs N = 1..5 across all 4 tissues
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=12.4, height=4.2)
    deb_n = d3["debye_counts"]
    for i, lbl in enumerate(t_labels):
        axes[0].semilogy(
            deb_n,
            d3["err_matrix"][i] * 100.0,
            **figstyle.series_style(i),
            label=lbl,
        )
    axes[0].axhline(1.0, color="#64748B", ls="--", lw=1.0, label="1% RMS threshold")
    axes[0].set_xticks(deb_n)
    axes[0].set_xlabel("Number of Debye poles $N$ (+ static $\\sigma_i$)")
    axes[0].set_ylabel("Relative RMS permittivity error [%]")
    axes[0].set_title("(a) Cole–Cole to Debye fit error vs. $N$ (10 MHz – 10 GHz)")
    axes[0].legend(fontsize=8.0)

    eps_cc_mus = d3["eps_cc_matrix"][0]
    for n_idx, n_deb in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        rel_f = (
            np.abs(d3["muscle_fits_vs_n"][n_idx] - eps_cc_mus)
            / np.abs(eps_cc_mus)
            * 100.0
        )
        axes[1].loglog(
            f_rf_ghz,
            rel_f,
            **figstyle.series_style(n_idx, markevery=25, ms=3.4),
            label=f"Muscle $N={n_deb}$ (RMS={d3['err_matrix'][0, n_idx]*100:.2f}%)",
        )
    for f_band, b_name in [(0.403, "MICS"), (0.915, "ISM"), (2.45, "ISM"), (5.8, "ISM")]:
        axes[1].axvline(f_band, color="#64748B", ls=":", lw=0.9)
        axes[1].text(f_band * 1.04, 0.08, b_name, fontsize=7.5, rotation=90, color="#475569")
    axes[1].set_xlabel("Frequency $f$ [GHz]")
    axes[1].set_ylabel(r"Pointwise relative error $|\varepsilon_{\mathrm{Debye}} - \varepsilon_{\mathrm{CC}}| / |\varepsilon_{\mathrm{CC}}|$ [%]")
    axes[1].set_title("(b) Muscle pointwise error across biomedical RF/microwave bands")
    axes[1].legend(fontsize=7.6)

    figstyle.save(fig, FIG_DIR / "07_tissue_debye_convergence.png")

    # ------------------------------------------------------------------
    # Figure 8: Muscle slab (d = 8 mm) FIT simulation vs Cole-Cole TM
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=12.4, height=4.3)
    f_mus_ghz = d4["f_eval"] / 1e9

    axes[0].plot(f_mus_ghz, d4["R_meas_cc"], "ko-", ms=4.2, lw=1.6, label=r"$R_{\mathrm{meas}}$ (Exact 4-Cole–Cole TM)")
    axes[0].plot(f_mus_ghz, d4["R_cont_n4"], **figstyle.series_style(2, markevery=5, ms=3.6), label=r"$R_{\mathrm{cont}}$ ($N=4$ Debye + $\sigma_i$)")
    axes[0].plot(f_mus_ghz, d4["R_fit_n4"], "s", color=figstyle.PALETTE[1], ms=4.8, label=r"$R_{\mathrm{FIT}}$ ($N=4$ Wakis 3D)")
    axes[0].plot(f_mus_ghz, d4["R_fit_n1"], "^:", color=figstyle.PALETTE[4], ms=4.2, lw=1.4, label=r"$R_{\mathrm{FIT}}$ ($N=1$ Debye + $\sigma_i$)")

    axes[0].plot(f_mus_ghz, d4["T_meas_cc"], "kD--", ms=3.6, lw=1.2, alpha=0.75, label=r"$T_{\mathrm{meas}}$ (Exact 4-Cole–Cole TM)")
    axes[0].plot(f_mus_ghz, d4["T_fit_n4"], **figstyle.series_style(0, markevery=4, ms=4.0), label=r"$T_{\mathrm{FIT}}$ ($N=4$ Wakis 3D)")
    axes[0].set_xlabel("Frequency $f$ [GHz]")
    axes[0].set_ylabel("Reflectance $R$, Transmittance $T$")
    axes[0].set_title("(a) Muscle layer ($d = 8$ mm): FIT vs. Gabriel (1996) Cole–Cole")
    axes[0].legend(fontsize=7.5, ncol=2)

    axes[1].semilogy(
        f_mus_ghz,
        np.abs(d4["R_fit_n1"] - d4["R_meas_cc"]),
        **figstyle.series_style(3, markevery=3),
        label=r"Total $|R_{\mathrm{FIT}} - R_{\mathrm{CC}}|$ ($N=1$ Debye)",
    )
    axes[1].semilogy(
        f_mus_ghz,
        np.abs(d4["R_fit_n4"] - d4["R_meas_cc"]),
        **figstyle.series_style(1, markevery=3),
        label=r"Total $|R_{\mathrm{FIT}} - R_{\mathrm{CC}}|$ ($N=4$ Debye)",
    )
    axes[1].semilogy(
        f_mus_ghz,
        np.abs(d4["R_cont_n4"] - d4["R_meas_cc"]),
        **figstyle.series_style(2, markevery=3),
        label=r"Pole-fit $|R_{\mathrm{cont}} - R_{\mathrm{CC}}|$ ($N=4$ Debye)",
    )
    axes[1].semilogy(
        f_mus_ghz,
        np.abs(d4["R_fit_n4"] - d4["R_disc_n4"]),
        **figstyle.series_style(0, markevery=3),
        label=r"Discretization $|R_{\mathrm{FIT}} - R_{\mathrm{disc}}|$ ($N=4$ Debye)",
    )
    axes[1].set_xlabel("Frequency $f$ [GHz]")
    axes[1].set_ylabel(r"Absolute reflectance error $|\Delta R|$")
    axes[1].set_title("(b) Muscle slab error decomposition ($N=1$ vs. $N=4$ Debye)")
    axes[1].legend(fontsize=7.6)

    figstyle.save(fig, FIG_DIR / "08_muscle_slab_fit_vs_cole_cole.png")


# %% [markdown]
# ## 4. Results & Error Decomposition Summary
#
# Running all five tasks (`--full`) produces the eight figures in `figures/` and
# the quantitative validation tables in `data/task_0..4.json`:
#
# 1. **Pole convergence on Johnson & Christy (1972) noble metals (`figures/01_noble_metal_pole_convergence.png`):**
#    - For **Gold (`Au`)** over $\lambda \in [0.30, 1.00]\;\mu\text{m}$, a
#      single Drude pole ($N=1$) has a $64.2\%$ relative RMS error because it
#      completely misses the $5d \to 6sp$ interband absorption edge below
#      $550\;\text{nm}$. Adding one Lorentz oscillator ($N=2$) drops the error
#      to $13.6\%$, and adding a second Lorentz oscillator ($N=3$, `1 Drude + 2 Lorentz`)
#      lowers it to **$5.89\%$** while remaining stable up to
#      $\Delta t_{\max} \approx 213\;\text{as}$ ($> 4\times$ the $50\;\text{as}$
#      reference optical step).
#    - For **Silver (`Ag`)** over $\lambda \in [0.30, 1.00]\;\mu\text{m}$, adding
#      one UV Lorentz oscillator ($N=2$) drops the error from $31.9\%$ to
#      $11.0\%$ (and $0.95\%$ inside the visible $0.40\text{–}0.80\;\mu\text{m}$
#      window of `dispersive_lib["silver jc"]`).
#    - Every fitted model satisfies $\max_\omega \operatorname{Im}\varepsilon(\omega) \le 0$
#      across $10^{12}\text{–}10^{17}\;\text{Hz}$ by construction.
# 2. **Choosing the fit band & what happens outside it (`figures/02_fit_band_extrapolation_and_library.png`):**
#    - Fitting a 2-pole model to **Gold** on the narrow red/NIR plasmonic band
#      $[0.55, 0.95]\;\mu\text{m}$ achieves **$1.14\%$ in-band RMS error**, but
#      extrapolating that fit into the UV/blue ($< 0.50\;\mu\text{m}$) produces a
#      **$66.5\%$ out-of-band RMS error** because the unconstrained UV interband
#      transitions were excluded from the fit window.
# 3. **Dielectric Sellmeier vs. reduced pole fits (`figures/05_sio2_sellmeier_and_slab.png`):**
#    - Malitson's 3-term Sellmeier formula for fused silica (`dispersive_lib["sio2"]`)
#      matches `tests/data/refractiveindex_info/SiO2_Malitson.yml` to machine
#      precision ($\sim 10^{-16}$) and has $\Delta t_{\max} = 51.2\;\text{as}$
#      governed by its deep-UV $68.4\;\text{nm}$ pole. Folding that deep-UV pole
#      into $\varepsilon_\infty = 1 + B_1 = 1.6962$ (`sio2_2pole_folded`) gives
#      a **$5.3\times 10^{-4}$** relative RMS error across $0.25\text{–}1.50\;\mu\text{m}$
#      while more than doubling $\Delta t_{\max}$ to **$136\;\text{as}$**.
# 4. **Gabriel *et al.* (1996) Cole–Cole biological tissues (`figures/06_gabriel_cole_cole_tissues.png`, `figures/07_tissue_debye_convergence.png`):**
#    - Fitting the fractional Cole–Cole spectrum over $10\;\text{MHz}\text{–}10\;\text{GHz}$
#      with $N = 1 \to 5$ Debye poles plus exact static conductivity $\sigma_i$
#      drives the relative RMS permittivity error smoothly below $1\%$ at $N=4$
#      for all four tissues (**Muscle**: $15.0\% \to 0.60\% \to 0.22\%$;
#      **Dry Skin**: $20.6\% \to 1.05\%$; **Fat**: $10.3\% \to 0.61\% \to 0.20\%$;
#      **Cortical Bone**: $21.2\% \to 0.99\% \to 0.35\%$).
# 5. **Four 3D FIT slab simulations vs. transfer matrix from measured data (`figures/03`, `04`, `05`, `08`):**
#    - Across all four slabs ($\text{Au}$ $40\;\text{nm}$, $\text{Ag}$ $36\;\text{nm}$,
#      $\text{SiO}_2$ $600\;\text{nm}$, and $\text{Muscle}$ $8\;\text{mm}$), the
#      error decomposition confirms that upgrading from $N=1$ to $N=3\text{–}4$
#      poles slashes the pole-fit error $\Delta R_{\text{fit}}$ by an order of
#      magnitude and brings the total time-domain simulation error
#      $\|R_{\text{FIT}} - R_{\text{meas}}\|_2$ down to the underlying FDTD/FIT
#      discretization floor.
#
# ---
#
# ## 5. Limits & Failure Modes
#
# 1. **Out-of-band extrapolation divergence:** As shown in Figure 2, log-parameterization
#    guarantees that a fitted pole model stays **passive** ($\operatorname{Im}\varepsilon(\omega) \le 0$)
#    at all frequencies, but it does **not** constrain $\operatorname{Re}\varepsilon(\omega)$
#    or $\operatorname{Im}\varepsilon(\omega)$ to remain accurate outside the
#    fitted frequency interval $[f_{\min}, f_{\max}]$. Always ensure your time-domain
#    source spectrum (`GaussianSource(frequency=f0, fwidth=df)`) is contained
#    within the fitted band.
# 2. **Unconstrained $\varepsilon_\infty \to 0$ stability trap on narrow dielectric windows:**
#    When fitting a low-loss normal-dispersion dielectric over a narrow band far
#    below its UV resonance, calling `fit_poles(..., eps_inf=None)` can trade
#    $\varepsilon_\infty \to 0$ against a far-UV pole $\omega_0 \gg \omega_{\max}$,
#    violating explicit leapfrog stability ($\varepsilon_\infty > S^2$ and
#    $\omega_0\Delta t < 2$). Fix `eps_inf=1.0` (or fold known deep-UV terms into
#    `eps_inf`) and always verify `material.is_stable(dt, courant)` before
#    launching a 3D run.
# 3. **Overfitting noisy experimental tables with too many poles:** Adding more
#    Lorentz poles than the number of physical absorption features can place
#    narrow, weakly damped parasitic poles between experimental sample points;
#    these create spurious ringing in the time-domain impulse response $\chi(t)$
#    and require longer FDTD ring-down times.
# 4. **Cole–Cole approximation below $10\;\text{MHz}$ ($\alpha$ and $\beta$ dispersions):**
#    Our $N=4$ Debye tissue fits target $10\;\text{MHz}\text{–}10\;\text{GHz}$.
#    Extending a single explicit FDTD simulation down to kHz/Hz frequencies to
#    cover the millisecond $\alpha$-dispersion ($\tau_4 \sim \text{ms}$) is
#    impractical both because of the $10^9$ ratio between RF timestep and
#    $\tau_4$ and because $6\text{–}8$ Debye poles are needed across 9 decades of
#    frequency.
#
# > **Forward pointers:**
# > - [**Tutorial 2.3 · Magnetic materials**](../02-03_magnetic_materials/magnetic_materials.ipynb)
# >   extends the same rational Lorentz and Debye ADE machinery to dispersive
# >   magnetic permeability $\mu(\omega)$ (`mu_poles`, `mu_discrete`), LLG
# >   ferromagnetic resonance, and Snoek's law.
# > - **Tutorial 3.6 · Plasmonic cylinder** benchmarks how the pole count $N$ in
# >   these Johnson & Christy $\text{Au}$/$\text{Ag}$ fits trades 2D/3D Mie
# >   scattering accuracy near the localized surface plasmon resonance against
# >   per-step memory and runtime.
# > - **Chapter 4 · Biomedical implant antenna** reuses the 4-pole Debye +
# >   $\sigma_i$ Gabriel (1996) tissue models exported in `data/task_3.json`.
#
# ---
#
# ## 6. Verified References
#
# 1. **P. B. Johnson and R. W. Christy**, "Optical constants of the noble metals,"
#    *Phys. Rev. B*, vol. 6, no. 12, pp. 4370–4379, Dec. 1972,
#    doi:[10.1103/PhysRevB.6.4370](https://doi.org/10.1103/PhysRevB.6.4370).
#    *(Tabulated room-temperature $(n, k)$ data for $\text{Cu}$, $\text{Ag}$, and
#    $\text{Au}$ across $0.188\text{–}1.937\;\mu\text{m}$; CC0 tables in
#    `tests/data/refractiveindex_info/`.)*
# 2. **I. H. Malitson**, "Interspecimen comparison of the refractive index of
#    fused silica," *J. Opt. Soc. Am.*, vol. 55, no. 10, pp. 1205–1209, Oct. 1965,
#    doi:[10.1364/JOSA.55.001205](https://doi.org/10.1364/JOSA.55.001205).
#    *(Three-term Sellmeier dispersion formula for fused silica $\text{SiO}_2$
#    at $20\,^\circ\text{C}$, `tests/data/refractiveindex_info/SiO2_Malitson.yml`.)*
# 3. **S. Gabriel, R. W. Lau, and C. Gabriel**, "The dielectric properties of
#    biological tissues: III. Parametric models for the dielectric spectrum of
#    tissues," *Phys. Med. Biol.*, vol. 41, no. 11, pp. 2271–2293, Nov. 1996,
#    doi:[10.1088/0031-9155/41/11/003](https://doi.org/10.1088/0031-9155/41/11/003).
#    *(Table 1 four-term Cole–Cole parametric coefficients for muscle, dry skin,
#    non-infiltrated fat, and cortical bone.)*
# 4. **A. D. Rakić, A. B. Djurišić, J. M. Elazar, and M. L. Majewski**, "Optical
#    properties of metallic films for vertical-cavity optoelectronic devices,"
#    *Appl. Opt.*, vol. 37, no. 22, pp. 5271–5283, Aug. 1998,
#    doi:[10.1364/AO.37.005271](https://doi.org/10.1364/AO.37.005271).
#    *(Lorentz–Drude multi-pole parameterization of 11 metals in `wakis.material_library`.)*
# 5. **J. D. Jackson**, *Classical Electrodynamics*, 3rd ed. New York, NY: Wiley,
#    1998, ch. 7 ("Plane Electromagnetic Waves and Wave Propagation," §7.5 & §7.10).
# 6. **A. Taflove and S. C. Hagness**, *Computational Electrodynamics: The
#    Finite-Difference Time-Domain Method*, 3rd ed. Norwood, MA: Artech House,
#    2005, ch. 9 ("Dispersive, Nonlinear, and Gain Materials," ADE formulation).


# %%
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Tutorial 2.2: Fitting measured material data"
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
