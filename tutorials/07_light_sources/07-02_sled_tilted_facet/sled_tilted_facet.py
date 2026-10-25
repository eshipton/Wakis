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
# # Tutorial 7.2 · Superluminescent diode: residual reflectivity of a tilted facet
#
# **2D Finite Integration Technique (Wakis FIT) vs. Marcuse (1989) Coupled-Mode Overlap & Zah et al. (1987) Angled-Facet SLED/TWA Experiments**
#
# ## Table of Contents
# 1. [Physical Motivation: Why SLEDs and Travelling-Wave Amplifiers Use Tilted Facets](#1-physical-motivation)
# 2. [Analytical Theory: Marcuse Overlap, Step-Index Slab Interference & Mode Conversion](#2-analytical-theory)
# 3. [Waveguide Parameterization & 2D Effective-Index Mapping](#3-waveguide-parameterization)
# 4. [Incoherent Spontaneous Emission vs. Coherent Mode Excitation](#4-incoherent-emission)
# 5. [Simulation Setup, Task Table & Execution](#5-simulation-setup)
# 6. [Visualizing Geometry, Kottke Subpixel Mesh, Eigenmodes & Pulse Propagation](#6-visualizing-fields)
# 7. [High-Dynamic-Range Reflectivity $R_{00}(\theta_t)$: Bare vs. AR-Coated Facets](#7-reflectivity-vs-tilt)
# 8. [Higher-Order Mode Conversion $R_{01}(\theta_t)$ & Far-Field Beam Deflection](#8-mode-conversion-farfield)
# 9. [SLED Cavity Spectral Ripple (Hakki–Paoli) & PML Dynamic-Range Floor](#9-sled-ripple-and-pml)
# 10. [Incoherent Dipole Ensemble Convergence & Subpixel Grid Convergence](#10-convergence-studies)
# 11. [Summary of Quantitative Results](#11-summary-table)
# 12. [Limits & Failure Modes](#12-limits-and-failure-modes)
# 13. [Exercises](#13-exercises)
# 14. [References](#14-references)

# %% [markdown]
# <a id="1-physical-motivation"></a>
# ## 1. Physical Motivation: Why SLEDs and Travelling-Wave Amplifiers Use Tilted Facets
#
# A **superluminescent diode (SLED)** occupies the regime between a semiconductor laser diode (high spatial coherence, narrow spectral line) and a surface-emitting LED (broad spectrum, low spatial coherence). In an SLED—or a **travelling-wave semiconductor optical amplifier (TWA/SOA)**—spontaneous emission generated inside an electrically pumped ridge waveguide undergoes large single-pass optical gain:
#
# $$G_s(\lambda) = \exp\!\bigl[\bigl(\Gamma g(\lambda) - \alpha_i\bigr) L\bigr] \sim 10^2 \text{–} 10^3 \quad (20\text{–}30\;\text{dB}),$$
#
# while remaining strictly in a **single-pass travelling-wave regime** without reaching the Fabry–Pérot lasing threshold.
#
# At a normal cleaved semiconductor-to-air facet ($n_{\text{eff}} \approx 3.33$), Fresnel reflection returns
#
# $$R_f(0^\circ) = \left(\frac{n_{\text{eff}} - 1}{n_{\text{eff}} + 1}\right)^2 \approx 0.289 \quad (28.9\%)$$
#
# of the guided mode power back into the cavity. Even a moderate single-pass gain of $G_s = 1 / R_f \approx 3.5$ ($5.4\;\text{dB}$) triggers round-trip lasing oscillation ($G_s \sqrt{R_1 R_2} \ge 1$), collapsing the $50\text{–}60\;\text{nm}$ spontaneous-emission spectrum into discrete longitudinal laser modes. Even below threshold, residual round-trip feedback $\gamma_{\text{rt}} = G_s \sqrt{R_1 R_2}$ imprints a periodic **Fabry–Pérot spectral ripple** with peak-to-valley modulation depth given by the Hakki–Paoli formula:
#
# $$\Delta G_{\text{dB}} = 20 \log_{10}\!\left(\frac{1 + G_s \sqrt{R_1 R_2}}{1 - G_s \sqrt{R_1 R_2}}\right).$$
#
# For optical coherence tomography (OCT) and fiber-optic gyroscopes, maintaining $\Delta G_{\text{dB}} < 0.5\text{–}1.0\;\text{dB}$ at $G_s = 23\;\text{dB}$ ($G_s = 200$) requires an effective modal facet reflectivity $R_{\text{eff}} = \sqrt{R_1 R_2} < 3 \times 10^{-4}$.
#
# Achieving $R < 10^{-4}$ across a $55\;\text{nm}$ bandwidth using only a dielectric anti-reflection (AR) coating is notoriously sensitive to refractive-index tolerances ($n_{\text{AR}} = \sqrt{n_{\text{eff}}} \approx 1.825$) and sub-nanometer film thickness variations. **Zah et al., *Electron. Lett.* 23, 990 (1987)** demonstrated that tilting the waveguide stripe by $\theta_t = 7^\circ$ relative to the cleaved crystal facet normal deflects the specular reflection by $2\theta_t = 14^\circ$ in the junction plane, decoupling the reflected wave from the backward guided mode into unguided cladding radiation modes and reducing the bare modal reflectivity by two orders of magnitude ($R_{00}(7^\circ) \approx 2\text{–}3 \times 10^{-3}$). Combining the $7^\circ$ angled facet with a relaxed single-layer AR coating suppresses residual modal reflectivity to the $10^{-5}$ regime.

# %% [markdown]
# <a id="2-analytical-theory"></a>
# ## 2. Analytical Theory: Marcuse Overlap, Step-Index Slab Interference & Mode Conversion
#
# Consider a 2D planar waveguide in the junction plane $(x, y)$ whose axis runs along $+x$ and meets a flat dielectric facet along the line $x_{\text{facet}}(y) = y \tan\theta_t$, where $\theta_t$ is the tilt angle between the waveguide axis and the facet normal.
#
# ### 2.1 Coupled-Mode Overlap Integral at a Tilted Facet
# An incident forward-propagating guided mode $\psi_0(y) e^{-i \beta_0 x}$ arriving at $x = y \tan\theta_t$ acquires a transverse phase gradient across the facet. After specular reflection with Fresnel amplitude coefficient $r_f(\theta_t) = \sqrt{R_f(\theta_t)}$, the backward wave at $x = 0$ carries a round-trip path-length phase shift $2 \beta_0 x_{\text{facet}}(y) = 2 n_{\text{eff},0} k_0 y \tan\theta_t$. Projecting this reflected field onto the backward guided mode $\psi_m(y)$ (normalized so that $\int_{-\infty}^{\infty} |\psi_m(y)|^2\,\mathrm{d}y = 1$) yields the modal power reflectivity (**Marcuse, *J. Lightwave Technol.* 7, 336, 1989**, Eqs. 10–13):
#
# $$R_{0m}^{\text{slab}}(\theta_t) = R_f(\theta_t) \left| \int_{-\infty}^{\infty} \psi_0(y)\,\psi_m(y) \exp\!\Bigl[i \bigl(\beta_0 + \beta_m\bigr) y \tan\theta_t\Bigr] \mathrm{d}y \right|^2.$$
#
# ### 2.2 Marcuse Gaussian-Mode Approximation ($R_{00}$ and $R_{01}$)
# Approximating the fundamental mode by a normalized Gaussian with $1/e$ field half-width (waist radius) $w_0$,
#
# $$\psi_0^{\text{Gauss}}(y) = \left(\frac{2}{\pi w_0^2}\right)^{1/4} \exp\!\left(-\frac{y^2}{w_0^2}\right), \qquad \int_{-\infty}^{\infty} \bigl|\psi_0^{\text{Gauss}}(y)\bigr|^2\,\mathrm{d}y = 1,$$
#
# the Fourier integral evaluates in closed form:
#
# $$R_{00}^{\text{Gauss}}(\theta_t) = R_f(\theta_t) \exp\!\left[-\left(\frac{2\pi n_{\text{eff}} w_0 \tan\theta_t}{\lambda_0}\right)^2\right] = R_f(\theta_t)\,e^{-u^2}, \qquad u(\theta_t) \equiv \frac{2\pi n_{\text{eff}} w_0 \tan\theta_t}{\lambda_0}.$$
#
# For small tilt angles ($\tan\theta_t \approx \theta_t$), this reproduces the classic Gaussian roll-off $R_{00}(\theta_t) \approx R_f \exp[-(2\pi n_{\text{eff}} w_0 \theta_t / \lambda_0)^2]$.
#
# When the waveguide is wide enough to support the first odd transverse mode, approximated by the first Hermite–Gaussian function $\psi_1^{\text{Gauss}}(y) = \frac{2 y}{w_0} \psi_0^{\text{Gauss}}(y)$, the $\text{TE}_0 \to \text{TE}_1$ mode-conversion power reflectivity is:
#
# $$R_{01}^{\text{Gauss}}(\theta_t) = R_f(\theta_t)\,u^2 e^{-u^2}.$$
#
# Notice three key physical features of $R_{01}(\theta_t)$:
# 1. At normal incidence ($\theta_t = 0^\circ \Rightarrow u = 0$), $R_{01}(0^\circ) = 0$ by even–odd parity orthogonality.
# 2. As $\theta_t$ increases, the linear phase ramp across the tilted facet breaks transverse parity and couples strongly into the odd $\text{TE}_1$ mode, peaking at $u = 1$ with $R_{01,\max} = R_f / e \approx 0.368\,R_f \approx 10.6\%$.
# 3. For $u > 1$ ($\theta_t \gtrsim 3^\circ$), $R_{01}(\theta_t) = u^2 R_{00}(\theta_t) > R_{00}(\theta_t)$: in a bimodal angled-facet guide, **parasitic reflection is dominated by $\text{TE}_1$ mode conversion rather than fundamental $\text{TE}_0$ reflection**.
#
# ### 2.3 Exact Step-Index Slab Interference Minimum (Marcuse 1989)
# While the Gaussian approximation decreases monotonically with $\theta_t$, the exact step-index slab mode $\psi_0^{\text{slab}}(y)$ has a discontinuous second derivative at the core boundaries $y = \pm w/2$. As Marcuse (1989) showed, at large tilt angles ($\theta_t \sim 9^\circ\text{–}12^\circ$) the Fourier transform of $|\psi_0^{\text{slab}}(y)|^2$ is governed by interference between the two core sidewalls at $y = \pm w/2$, producing a sharp **destructive-interference null** near $\theta_t \approx 9.7^\circ\text{–}10^\circ$ ($R_{00}^{\text{slab}}(9.70^\circ) = 9.0 \times 10^{-12}$, $R_{00}^{\text{slab}}(10^\circ) = 5.82 \times 10^{-6}$, and $R_{00}^{\text{FIT}}(10^\circ) = 1.39 \times 10^{-8}$ at `cpw = 20`) followed by a weak diffraction sidelobe near $\theta_t = 12^\circ$ ($R_{00}^{\text{slab}}(12^\circ) = 5.75 \times 10^{-5}$, $R_{00}^{\text{FIT}}(12^\circ) = 3.49 \times 10^{-5}$). Our 2D Wakis FIT simulations resolve both the Gaussian regime ($0^\circ\text{–}8^\circ$) and the Marcuse sidewall-interference null at $10^\circ$.

# %% [markdown]
# <a id="3-waveguide-parameterization"></a>
# ## 3. Waveguide Parameterization & 2D Effective-Index Mapping
#
# We anchor our model to the $1.49\;\mu\text{m}$ InGaAsP/InP angled-facet travelling-wave amplifier / SLED experiments of **Zah et al. (1987)**:
# - **Wavelength**: $\lambda_0 = 1.49\;\mu\text{m}$ ($f_0 = 201.203\;\text{THz}$).
# - **Core & Cladding Effective Indices**: Zah et al. report that a $\theta_t = 7^\circ$ tilted stripe radiates into air at $\theta_{\text{out}} = 24^\circ$ from the facet normal. By Snell's law,
#   $$n_{\text{core}} = \frac{\sin 24^\circ}{\sin 7^\circ} = 3.337.$$
#   With lateral effective-index contrast $\Delta n = 0.026$ ($n_{\text{clad}} = 3.311$), the numerical aperture is $\text{NA} = \sqrt{n_{\text{core}}^2 - n_{\text{clad}}^2} = 0.4158$.
# - **Single-Mode vs. Two-Mode Core Widths ($w_1$ and $w_2$)**:
#   In a 2D symmetric step-index slab of full width $w$, the half-width normalized frequency is $V_{\text{half}} = \frac{\pi w}{\lambda_0}\text{NA}$, with $\text{TE}_1$ cutoff at $V_{\text{half}} = \pi/2 \approx 1.5708$ ($w_{\text{cut},1} = 1.792\;\mu\text{m}$) and $\text{TE}_2$ cutoff at $V_{\text{half}} = \pi$ ($w_{\text{cut},2} = 3.583\;\mu\text{m}$).
#   - **Single-mode guide**: $w_1 = 1.75\;\mu\text{m}$ gives $V_{\text{half}} = 1.534 < \pi/2$ (strictly single-mode), $n_{\text{eff},0} = 3.3276$, and an optimal Gaussian $1/e$ field waist $w_0 = 1.293\;\mu\text{m}$ (power overlap $99.65\%$), representing the lateral spot size of Zah et al.'s narrow weakly etched 3D ridge ($3.8\;\mu\text{m}$ nominal trench mask). At $\theta_t = 7^\circ$, this parameterization predicts $R_{00}^{\text{Gauss}}(7^\circ) = 2.027 \times 10^{-3}$ in the Marcuse Gaussian approximation, $R_{00}^{\text{slab}}(7^\circ) = 3.473 \times 10^{-3}$ in the exact 1D step-index slab overlap integral, and $R_{00}^{\text{FIT}}(7^\circ) = 3.546 \times 10^{-3}$ at `cpw = 20` ($3.648 \times 10^{-3}$ at `cpw = 40`) in Wakis 2D FIT. Note that Zah et al. (1987) quote an experimental range of $R \approx 2\text{–}3 \times 10^{-3}$ for their 3D etched ridge without tabulating the lateral effective-index step $\Delta n_{\text{eff}}$; because $R_{00}(\theta_t) \propto \exp[-(2\pi n_{\text{eff}} w_0 \tan\theta_t / \lambda_0)^2]$ depends exponentially on $w_0^2$, the exact reflectivity value at $7^\circ$ is sensitive to the chosen $(w_1, \Delta n)$ parameters (which we fix a priori rather than tuning to match a published number). By contrast, comparing Wakis 2D FIT against the exact 1D step-index slab overlap integral for the *same* $(w_1, n_{\text{core}}, n_{\text{clad}})$ has zero free parameters and agrees to within $2.1\%$ at `cpw = 20`.
#   - **Two-mode guide**: $w_2 = 2.80\;\mu\text{m}$ gives $V_{\text{half}} = 2.454 \in (\pi/2, \pi)$ (supporting $\text{TE}_0$ with $n_{\text{eff},0} = 3.3318$ and $\text{TE}_1$ with $n_{\text{eff},1} = 3.3178$), mapping Zah et al.'s wider $4.8\;\mu\text{m}$ bimodal ridge that exhibited double-lobed $\text{TE}_1$ far-field radiation (Zah et al. Fig. 2b).
# - **Single-Layer AR Coating**: A quarter-wave dielectric film ($\text{Al}_2\text{O}_3/\text{SiO}_x\text{N}_y$, $n_{\text{AR}} = 1.72$, thickness $t_{\text{AR}} = \lambda_0 / (4 n_{\text{AR}}) = 216.57\;\text{nm}$) reduces the normal-incidence facet reflectance from $R_f(0^\circ) = 0.2893$ to $R_f^{\text{AR}}(0^\circ) = 3.449 \times 10^{-3}$ (1D TMM; $3.608 \times 10^{-3}$ in 2D FIT at `cpw = 20`, $3.949 \times 10^{-3}$ at `cpw = 40`), and reduces the $7^\circ$ tilted-facet modal reflectivity to $R_{00}^{\text{AR}}(7^\circ) = 4.355 \times 10^{-5}$ in 2D FIT ($5.083 \times 10^{-5}$ in 1D slab overlap, $2.966 \times 10^{-5}$ in Marcuse Gaussian).
#
# > [!NOTE]
# > **Single-Cut Painter's-Order Geometry in Wakis FIT:** Rather than abutting two polygons along the tilted line $x = y \tan\theta_t$ (which would cause subpixel smoothing to blend cells twice along the seam), we construct the geometry in painter's order: (1) background cladding `Box` ($n_{\text{clad}}$), (2) continuous straight waveguide core `PolySlab` ($n_{\text{core}}$), (3) optional AR-coating half-space `PolySlab` ($x > y\tan\theta_t$), and (4) air half-space `PolySlab` ($x > y\tan\theta_t + t_{\text{AR}}/\cos\theta_t$). Each interface is cut cleanly once by Kottke subpixel averaging.

# %% [markdown]
# <a id="4-incoherent-emission"></a>
# ## 4. Incoherent Spontaneous Emission vs. Coherent Mode Excitation
#
# In an SLED, optical power originates from **spontaneous emission** by electron–hole recombination across the active core $y \in [-w/2, w/2]$. Because independent emitters at positions $y_j$ have uncorrelated random phases $\phi_j$ ($\langle e^{i(\phi_j - \phi_k)}\rangle = \delta_{jk}$), **spontaneous emission is spatially and temporally incoherent: one must average modal and radiated POWERS over dipole positions, orientations, and frequencies, never complex fields**:
#
# $$P_m^{\text{incoh}} = \frac{1}{N_{\text{dip}}} \sum_{j=1}^{N_{\text{dip}}} \bigl|a_{m,j}^+\bigr|^2 \;\xrightarrow{N_{\text{dip}} \to \infty}\; \frac{1}{w} \int_{-w/2}^{w/2} \bigl|a_m^+(y_0)\bigr|^2\,\mathrm{d}y_0 \propto \Gamma_m.$$
#
# If one erroneously sums dipole fields coherently in a single simulation ($\phi_j = 0$), the emitters act as a broadside phased array: even-mode ($\text{TE}_0$) excitation is artificially enhanced while odd-mode ($\text{TE}_1$) excitation cancels to zero by symmetry ($\sum_j a_{1,j}^+ = 0$). In **Task 3**, we simulate $N_{\text{dip}} = 9$ point dipoles (`CurrentSource` $J_y$) across the active core of the bimodal guide in Wakis 2D FIT (`cpw = 20`), obtaining an incoherent odd-mode power fraction $\eta_{\text{TE}_1}^{\text{incoh}} = 0.4074$ ($40.74\%$) versus $\eta_{\text{TE}_1}^{\text{coh}} = 4.10 \times 10^{-6}$ for the coherent field sum. We also examine both deterministic spatial quadrature convergence (where an unweighted $N_{\text{dip}}$-point sample mean across $[-0.42 w_2, +0.42 w_2]$ reaches $6.35\%$ and $7.69\%$ error for individual $\text{TE}_0$ and $\text{TE}_1$ powers at $N_{\text{dip}} = 9$ and $2.13\%\text{–}3.00\%$ at $N_{\text{dip}} = 25$, while the relative mode-fraction ratio $\eta_{\text{TE}_1}$ cancels endpoint weight errors to $< 1\%$ by $N_{\text{dip}} = 9$) and stochastic random-phase Monte Carlo convergence ($\sigma_{\text{rel}} \sim 1/\sqrt{N_{\text{real}}}$, requiring $N_{\text{real}} \gtrsim 64\text{–}128$ realizations for $\le 10\%$ RMS statistical error: $7.77\%\text{–}9.99\%$ at $N_{\text{real}} = 128$ and $4.74\%\text{–}5.53\%$ at $N_{\text{real}} = 256$).

# %% [markdown]
# <a id="5-simulation-setup"></a>
# ## 5. Simulation Setup, Task Table & Execution
#
# ### 5.1 Locally Refined Non-Uniform Mesh (`mesh_overrides`)
# Every simulation in both `--quick` and `--full` modes satisfies strict FDTD/FIT resolution requirements across the entire broadband window $\lambda \in [1.43, 1.55]\;\mu\text{m}$ inside the highest-index material ($n_{\max} = n_{\text{core}} = 3.337$, so $\lambda_{\min} / (10\,n_{\max}) = 42.85\;\text{nm}$; no metallic media are present):
# 1. **Rule 1 ($\ge 10$ cells per wavelength everywhere)**: The background grid uses `cells_per_wavelength = max(10.5, 0.75 * cpw)` at $\lambda_{\min} = 1.43\;\mu\text{m}$, giving $\Delta x_{\text{background}} = 28.57\;\text{nm}$ ($15.0\;\text{cpw}$ at $\lambda_{\min}$, $15.6\;\text{cpw}$ at $\lambda_0$) for the `--full` sweeps (`cpw = 20`) and $\Delta x_{\text{background}} \le 40.81\;\text{nm} < 42.85\;\text{nm}$ ($\ge 10.5\;\text{cpw}$ at $\lambda_{\min}$) across all tasks (`cpw` $\in \{14, 20, 28, 40\}$ in `--full`, and `cpw` $\in \{10, 14, 20, 28\}$ in `--quick`).
# 2. **Rule 2 ($\ge 4$ cells across the quarter-wave AR coating & node-snapped interfaces)**: Using `Simulation(..., mesh_overrides=...)`, we place three contiguous refinement zones `[xmin, 0.0]`, `[0.0, t_AR]`, and `[t_AR, xmax_ref]` with longitudinal step $h_x = t_{\text{AR}} / N_{\text{AR}}$, giving $N_{\text{AR}} \in \{7, 10, 14, 20\}$ cells across $t_{\text{AR}} = 216.57\;\text{nm}$ in `--full` ($N_{\text{AR}} = 6$ at `cpw = 10` in `--quick`). At normal incidence ($\theta_t = 0^\circ$), both AR coating interfaces $x = 0$ and $x = t_{\text{AR}}$ lie on exact grid nodes; at tilted angles ($\theta_t > 0^\circ$), Kottke subpixel smoothing resolves the oblique dielectric interfaces without staircase artifacts. Keeping $\Delta x = h_x$ uniform from `xmin` through `ModeSource`, `ModeMonitor`, and the facet eliminates parasitic grid-step reflections upstream of the facet so the $\sim 10^{-8}$ Marcuse destructive-interference null at $\theta_t = 10^\circ$ is cleanly resolved.
# 3. **Exact Core Sidewall Nodes in $y$**: Across the guided mode waist $[-y_{\text{waist}}, +y_{\text{waist}}]$, the transverse step $h_y = w / N_{\text{core},y}$ ($N_{\text{core},y}$ even) places exact mesh nodes at $y = 0$ and the waveguide sidewalls $y = \pm w/2$, while grading smoothly out to the cladding boundaries.
#
# The table below lists the `--full` production study stored in `data/` (with `--quick` using `cpw = 14`, $\Delta x_{\text{bg}} = 40.81\;\text{nm}$, $\Delta x_{\min} = 30.94\;\text{nm}$, $7$ cells across $t_{\text{AR}}$, $71{,}687$ cells, and $4{,}047$ steps for Tasks `00–03`, and `cpw` $\in \{10, 14, 20, 28\}$ for Tasks `04–10`):
#
# | Task Index & Name (`--full`) | `cpw` | $\Delta x_{\text{bg}}$ [nm] | $\Delta x_{\min}$ [nm] | Min Feature $t_{\text{AR}}$ [nm] | Cells Across $t_{\text{AR}}$ | $N_{\text{cells}}$ ($N_x \times N_y$) | $N_{\text{steps}}$ | Runtime [s] |
# | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | `00 bare_tilt_sweep` (12 angles + fields/movie) | $20$ | $28.57$ | $21.66$ | $216.57$ | $10$ | $138{,}431$ ($289 \times 479$) | $5{,}781$ | $861.2$ |
# | `01 ar_tilt_sweep` (12 angles + AR spectrum) | $20$ | $28.57$ | $21.66$ | $216.57$ | $10$ | $138{,}431$ ($289 \times 479$) | $5{,}781$ | $774.9$ |
# | `02 bimodal_tilt_sweep` ($w_2=2.80\;\mu\text{m}$, 8 angles) | $20$ | $28.57$ | $21.66$ | $216.57$ | $10$ | $140{,}743$ ($289 \times 487$) | $5{,}747$ | $571.0$ |
# | `03 pml_and_incoherent_study` (CPML + 9 dipoles) | $20$ | $28.57$ | $21.66$ | $216.57$ | $10$ | $137{,}061$ ($291 \times 471$) | $5{,}781$ | $450.9$ |
# | `04 kottke_cpw14` ($7^\circ, 10^\circ, 0^\circ$ AR) | $14$ | $40.81$ | $30.94$ | $216.57$ | $7$ | $71{,}687$ ($209 \times 343$) | $4{,}047$ | $67.1$ |
# | `05 kottke_cpw20` ($7^\circ, 10^\circ, 0^\circ$ AR) | $20$ | $28.57$ | $21.66$ | $216.57$ | $10$ | $138{,}431$ ($289 \times 479$) | $5{,}781$ | $183.1$ |
# | `06 kottke_cpw28` ($7^\circ, 10^\circ, 0^\circ$ AR) | $28$ | $20.41$ | $15.47$ | $216.57$ | $14$ | $261{,}623$ ($397 \times 659$) | $8{,}039$ | $498.0$ |
# | `07 staircase_cpw14` ($7^\circ, 10^\circ, 0^\circ$ AR) | $14$ | $40.81$ | $30.94$ | $216.57$ | $7$ | $71{,}687$ ($209 \times 343$) | $4{,}047$ | $68.4$ |
# | `08 staircase_cpw20` ($7^\circ, 10^\circ, 0^\circ$ AR) | $20$ | $28.57$ | $21.66$ | $216.57$ | $10$ | $138{,}431$ ($289 \times 479$) | $5{,}781$ | $195.1$ |
# | `09 staircase_cpw28` ($7^\circ, 10^\circ, 0^\circ$ AR) | $28$ | $20.41$ | $15.47$ | $216.57$ | $14$ | $261{,}623$ ($397 \times 659$) | $8{,}039$ | $507.2$ |
# | `10 kottke_reference` ($7^\circ, 10^\circ, 0^\circ$ AR) | $40$ | $14.28$ | $10.83$ | $216.57$ | $20$ | $519{,}681$ ($557 \times 933$) | $11{,}507$ | $1{,}477.0$ |

# %%
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.constants import c as C_LIGHT

from wakis import figstyle, schematic

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _helpers as h  # noqa: E402

FIG_DIR = SCRIPT_DIR / "figures"
DATA_DIR = SCRIPT_DIR / "data"

THETA_SWEEP_SM = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0)
THETA_SWEEP_SM_QUICK = (0.0, 2.0, 4.0, 6.0, 7.0, 8.0, 10.0, 12.0)
THETA_SWEEP_MM = (0.0, 2.0, 4.0, 6.0, 7.0, 8.0, 10.0, 12.0)
THETA_SWEEP_MM_QUICK = (0.0, 2.0, 4.0, 7.0, 10.0, 12.0)


def get_task_table(quick: bool = True) -> list[dict]:
    """Return the task table for the Tutorial 7.2 tilted-facet SLED study.

    Parameters
    ----------
    quick : bool, optional
        If True, uses ``cpw = 14`` for sweeps (with 8 SM and 6 MM angles) and
        ``cpw in (10, 14, 20, 28)`` for the grid convergence study. If False
        (``--full``), uses ``cpw = 20`` for sweeps (all 12 SM and 8 MM angles)
        and ``cpw in (14, 20, 28, 40)`` for convergence. Every task in both
        modes satisfies ``>= 10`` cells per wavelength in ``n_core`` across the
        full ``1.43-1.55 um`` band and ``>= 6`` cells across the ``216.6 nm``
        quarter-wave AR coating.

    Returns
    -------
    tasks : list of dict
        Task descriptors indexed by ``index``.
    """
    cpw_main = 14 if quick else 20
    cpw_conv = [10, 14, 20] if quick else [14, 20, 28]
    cpw_ref = 28 if quick else 40
    thetas_sm = list(THETA_SWEEP_SM_QUICK if quick else THETA_SWEEP_SM)
    thetas_mm = list(THETA_SWEEP_MM_QUICK if quick else THETA_SWEEP_MM)

    tasks = [
        {
            "index": 0,
            "name": "bare_tilt_sweep",
            "kind": "bare_sweep",
            "cpw": cpw_main,
            "thetas": thetas_sm,
            "description": f"Single-mode (w=1.75 um) bare facet sweep 0-12 deg, DFT fields, movie & far-field (cpw={cpw_main})",
        },
        {
            "index": 1,
            "name": "ar_tilt_sweep",
            "kind": "ar_sweep",
            "cpw": cpw_main,
            "thetas": thetas_sm,
            "description": f"Single-mode (w=1.75 um) single-layer AR coated facet sweep 0-12 deg & spectrum (cpw={cpw_main})",
        },
        {
            "index": 2,
            "name": "bimodal_tilt_sweep",
            "kind": "bimodal_sweep",
            "cpw": cpw_main,
            "thetas": thetas_mm,
            "description": f"Two-mode (w=2.80 um) bare facet R00 & R01 sweep + TE0/TE1 far-field at 7 deg (cpw={cpw_main})",
        },
        {
            "index": 3,
            "name": "pml_and_incoherent_study",
            "kind": "pml_incoherent",
            "cpw": cpw_main,
            "description": f"CPML parasitic floor vs n_pml (6..14) + incoherent 9-dipole ensemble in 2D FIT (cpw={cpw_main})",
        },
    ]

    idx = 4
    for avg in ("kottke", "staircase"):
        for c_val in cpw_conv:
            tasks.append(
                {
                    "index": idx,
                    "name": f"{avg}_cpw{c_val}",
                    "kind": "conv_point",
                    "averaging": avg,
                    "cpw": c_val,
                    "theta_deg": 7.0,
                    "description": f"Convergence point at theta_t=7 & 10 deg + AR 0 deg, averaging={avg}, cpw={c_val}",
                }
            )
            idx += 1

    tasks.append(
        {
            "index": idx,
            "name": "kottke_reference",
            "kind": "conv_point",
            "averaging": "kottke",
            "cpw": cpw_ref,
            "theta_deg": 7.0,
            "description": f"Fine reference run at theta_t=7 & 10 deg + AR 0 deg, averaging=kottke, cpw={cpw_ref}",
        }
    )
    return tasks


def _extract_mesh_meta(res_dict: dict) -> dict:
    """Extract standard mesh resolution metadata for NPZ and JSON output."""
    return {
        "dx_background_nm": float(res_dict["dx_background_nm"]),
        "dx_min_nm": float(res_dict["dx_min_nm"]),
        "min_feature_nm": float(res_dict["min_feature_nm"]),
        "cells_across_feature": int(res_dict["cells_across_feature"]),
        "n_cells": int(res_dict["n_cells"]),
        "n_steps": int(res_dict["n_steps"]),
    }


def run_task(task_index: int, out_dir: Path, quick: bool = True) -> dict:
    """Execute a single study task and save ``task_XX.npz`` and ``task_XX.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = get_task_table(quick=quick)
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"Task index {task_index} out of range (0..{len(tasks) - 1})")
    task = tasks[task_index]
    kind = task["kind"]
    cpw = int(task["cpw"])
    t_start = time.perf_counter()

    freqs_broad = C_LIGHT / np.linspace(1.43e-6, 1.55e-6, 13)

    if kind == "bare_sweep":
        thetas = np.asarray(task["thetas"], dtype=float)
        r00_arr = []
        neff0_val = 0.0
        spec_0, spec_7 = None, None
        res_0_fields, res_7_fields = None, None
        for th in thetas:
            is_0 = abs(th - 0.0) < 1e-6
            is_7 = abs(th - 7.0) < 1e-6
            res = h.run_tilted_facet_fit(
                theta_deg=float(th),
                width=h.WIDTH_SM,
                cpw=cpw,
                ar_coated=False,
                record_fields=(is_0 or is_7),
                record_farfield=(is_0 or is_7),
                record_movie=is_7,
                freqs=freqs_broad if (is_0 or is_7) else None,
            )
            r00_arr.append(res["r00"])
            neff0_val = res["neff_0"]
            if is_0:
                spec_0 = res["r00_spectrum"]
                res_0_fields = res
            if is_7:
                spec_7 = res["r00_spectrum"]
                res_7_fields = res

        # Also extract staircase permittivity at 7 deg for mesh comparison plot
        res_7_stair = h.run_tilted_facet_fit(
            theta_deg=7.0,
            width=h.WIDTH_SM,
            cpw=cpw,
            ar_coated=False,
            averaging="staircase",
            record_fields=False,
        )
        mesh_meta = _extract_mesh_meta(res_7_fields)

        payload = {
            "thetas": thetas,
            "r00": np.asarray(r00_arr, dtype=float),
            "neff_0": np.array(neff0_val),
            "freqs_broad": freqs_broad,
            "r00_spec_0deg": spec_0,
            "r00_spec_7deg": spec_7,
            "x_coords": res_7_fields["x_coords"],
            "y_coords": res_7_fields["y_coords"],
            "ey0_mode": res_7_fields["ey0_mode"],
            "eps_y_kottke_7deg": res_7_fields["eps_y"],
            "eps_y_stair_7deg": res_7_stair["eps_y"],
            "dft_x": res_7_fields["dft_x"],
            "dft_y": res_7_fields["dft_y"],
            "ey_2d_0deg": res_0_fields["ey_2d"],
            "ey_2d_7deg": res_7_fields["ey_2d"],
            "phi_x_deg": res_7_fields["phi_x_deg"],
            "theta_norm_0deg": res_0_fields["theta_normal_deg"],
            "ff_0deg": res_0_fields["ff_intensity"],
            "theta_norm_7deg": res_7_fields["theta_normal_deg"],
            "ff_7deg": res_7_fields["ff_intensity"],
            "movie_frames": res_7_fields["movie_frames"],
            "movie_times_fs": res_7_fields["movie_times_fs"],
            **{k: np.array(v) for k, v in mesh_meta.items()},
        }
        summary = {
            "task": task,
            **mesh_meta,
            "neff_0": float(neff0_val),
            "r00_0deg": float(r00_arr[0]),
            "r00_4deg": float(r00_arr[list(thetas).index(4.0)]),
            "r00_7deg": float(r00_arr[list(thetas).index(7.0)]),
            "r00_10deg": float(r00_arr[list(thetas).index(10.0)]),
            "r00_12deg": float(r00_arr[list(thetas).index(12.0)]),
            "ff_peak_0deg": float(
                res_0_fields["theta_normal_deg"][
                    np.argmax(res_0_fields["ff_intensity"])
                ]
            ),
            "ff_peak_7deg": float(
                res_7_fields["theta_normal_deg"][
                    np.argmax(res_7_fields["ff_intensity"])
                ]
            ),
            "runtime_s": float(time.perf_counter() - t_start),
        }

    elif kind == "ar_sweep":
        thetas = np.asarray(task["thetas"], dtype=float)
        r00_ar = []
        spec_ar_0, spec_ar_7 = None, None
        res_ar_7 = None
        for th in thetas:
            is_0 = abs(th - 0.0) < 1e-6
            is_7 = abs(th - 7.0) < 1e-6
            res = h.run_tilted_facet_fit(
                theta_deg=float(th),
                width=h.WIDTH_SM,
                cpw=cpw,
                ar_coated=True,
                record_fields=is_7,
                freqs=freqs_broad if (is_0 or is_7) else None,
            )
            r00_ar.append(res["r00"])
            if is_0:
                spec_ar_0 = res["r00_spectrum"]
            if is_7:
                spec_ar_7 = res["r00_spectrum"]
                res_ar_7 = res

        mesh_meta = _extract_mesh_meta(res_ar_7)
        payload = {
            "thetas": thetas,
            "r00_ar": np.asarray(r00_ar, dtype=float),
            "freqs_broad": freqs_broad,
            "r00_ar_spec_0deg": spec_ar_0,
            "r00_ar_spec_7deg": spec_ar_7,
            "eps_y_ar_7deg": res_ar_7["eps_y"],
            "ey_2d_ar_7deg": res_ar_7["ey_2d"],
            **{k: np.array(v) for k, v in mesh_meta.items()},
        }
        summary = {
            "task": task,
            **mesh_meta,
            "r00_ar_0deg": float(r00_ar[0]),
            "r00_ar_7deg": float(r00_ar[list(thetas).index(7.0)]),
            "r00_ar_10deg": float(r00_ar[list(thetas).index(10.0)]),
            "runtime_s": float(time.perf_counter() - t_start),
        }

    elif kind == "bimodal_sweep":
        thetas = np.asarray(task["thetas"], dtype=float)
        r00_mm, r01_mm = [], []
        res_7_te0 = None
        for th in thetas:
            is_7 = abs(th - 7.0) < 1e-6
            res = h.run_tilted_facet_fit(
                theta_deg=float(th),
                width=h.WIDTH_MM,
                cpw=cpw,
                ar_coated=False,
                compute_bimodal=True,
                record_farfield=is_7,
            )
            r00_mm.append(res["r00"])
            r01_mm.append(res["r01"])
            if is_7:
                res_7_te0 = res

        # Also launch TE1 at theta_t = 7 deg to get the double-lobed TE1 far-field (Zah Fig. 2b)
        res_7_te1 = h.run_tilted_facet_fit(
            theta_deg=7.0,
            width=h.WIDTH_MM,
            cpw=cpw,
            ar_coated=False,
            launch_mode_index=1,
            compute_bimodal=True,
            record_farfield=True,
        )
        mesh_meta = _extract_mesh_meta(res_7_te0)

        payload = {
            "thetas": thetas,
            "r00_mm": np.asarray(r00_mm, dtype=float),
            "r01_mm": np.asarray(r01_mm, dtype=float),
            "neff_0_mm": np.array(res_7_te0["neff_0"]),
            "neff_1_mm": np.array(res_7_te0["neff_1"]),
            "y_coords_mm": res_7_te0["y_coords"],
            "ey0_mode_mm": res_7_te0["ey0_mode"],
            "ey1_mode_mm": res_7_te1["ey1_mode"],
            "theta_norm_7deg": res_7_te0["theta_normal_deg"],
            "ff_mm_te0_7deg": res_7_te0["ff_intensity"],
            "ff_mm_te1_7deg": res_7_te1["ff_intensity"],
            **{k: np.array(v) for k, v in mesh_meta.items()},
        }
        th_norm = res_7_te1["theta_normal_deg"]
        ff_te1 = res_7_te1["ff_intensity"]
        lobe1 = float(th_norm[th_norm < 24.0][np.argmax(ff_te1[th_norm < 24.0])])
        lobe2 = float(th_norm[th_norm >= 24.0][np.argmax(ff_te1[th_norm >= 24.0])])
        summary = {
            "task": task,
            **mesh_meta,
            "neff_0_mm": float(res_7_te0["neff_0"]),
            "neff_1_mm": float(res_7_te0["neff_1"]),
            "r01_mm_2deg": float(r01_mm[list(thetas).index(2.0)]),
            "r00_mm_7deg": float(r00_mm[list(thetas).index(7.0)]),
            "r01_mm_7deg": float(r01_mm[list(thetas).index(7.0)]),
            "r01_mm_max": float(np.max(r01_mm)),
            "te1_ff_lobes_deg": [lobe1, lobe2],
            "runtime_s": float(time.perf_counter() - t_start),
        }

    elif kind == "pml_incoherent":
        pml_res = h.run_pml_floor_study(
            n_pml_list=(6, 8, 10, 12, 14),
            cpw=cpw,
            width=h.WIDTH_SM,
        )
        dip_res = h.run_incoherent_dipole_ensemble_fit(
            cpw=cpw,
            width=h.WIDTH_MM,
            n_unique_half=5,
        )
        mesh_meta = _extract_mesh_meta(pml_res)
        payload = {
            "n_pml": pml_res["n_pml"],
            "r_floor": pml_res["r_floor"],
            **dip_res,
            **{k: np.array(v) for k, v in mesh_meta.items()},
        }
        summary = {
            "task": task,
            **mesh_meta,
            "r_floor_npml6": float(pml_res["r_floor"][0]),
            "r_floor_npml10": float(pml_res["r_floor"][2]),
            "r_floor_npml14": float(pml_res["r_floor"][4]),
            "eta_te1_incoh": float(dip_res["eta_te1_incoh"]),
            "eta_te1_coh": float(dip_res["eta_te1_coh"]),
            "runtime_s": float(time.perf_counter() - t_start),
        }

    elif kind == "conv_point":
        avg_mode = str(task["averaging"])
        res_7 = h.run_tilted_facet_fit(
            theta_deg=7.0,
            width=h.WIDTH_SM,
            cpw=cpw,
            ar_coated=False,
            averaging=avg_mode,
        )
        res_10 = h.run_tilted_facet_fit(
            theta_deg=10.0,
            width=h.WIDTH_SM,
            cpw=cpw,
            ar_coated=False,
            averaging=avg_mode,
        )
        res_ar0 = h.run_tilted_facet_fit(
            theta_deg=0.0,
            width=h.WIDTH_SM,
            cpw=cpw,
            ar_coated=True,
            averaging=avg_mode,
        )
        mesh_meta = _extract_mesh_meta(res_7)
        payload = {
            "cpw": np.array(cpw),
            "h_cell": np.array(res_7["h_cell"]),
            "r00": np.array(res_7["r00"]),
            "r00_7deg": np.array(res_7["r00"]),
            "r00_10deg": np.array(res_10["r00"]),
            "r_ar_0deg": np.array(res_ar0["r00"]),
            "neff_0": np.array(res_7["neff_0"]),
            "averaging": np.array(avg_mode),
            "runtime_s": np.array(time.perf_counter() - t_start),
            **{k: np.array(v) for k, v in mesh_meta.items()},
        }
        summary = {
            "task": task,
            **mesh_meta,
            "cpw": cpw,
            "h_cell_nm": float(res_7["h_cell"] * 1e9),
            "r00_7deg": float(res_7["r00"]),
            "r00_10deg": float(res_10["r00"]),
            "r_ar_0deg": float(res_ar0["r00"]),
            "neff_0": float(res_7["neff_0"]),
            "runtime_s": float(time.perf_counter() - t_start),
        }
    else:
        raise ValueError(f"Unknown task kind: {kind}")

    npz_path = out_dir / f"task_{task_index:02d}.npz"
    json_path = out_dir / f"task_{task_index:02d}.json"
    np.savez_compressed(npz_path, **payload)
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[Task {task_index:02d} | {task['name']}] cpw={cpw}, "
        f"dx_bg={summary['dx_background_nm']:.2f} nm, "
        f"dx_min={summary['dx_min_nm']:.2f} nm, "
        f"min_feat={summary['min_feature_nm']:.2f} nm, "
        f"cells_feat={summary['cells_across_feature']}, "
        f"n_cells={summary['n_cells']}, n_steps={summary['n_steps']}, "
        f"time={summary['runtime_s']:.1f} s -> {npz_path.name}"
    )
    return summary


def run_all_tasks(out_dir: Path, quick: bool = True) -> list[dict]:
    """Run all tasks sequentially if not already present."""
    tasks = get_task_table(quick=quick)
    summaries = []
    for t in tasks:
        idx = t["index"]
        summaries.append(run_task(idx, out_dir=out_dir, quick=quick))
    return summaries


def load_all_results(data_dir: Path, quick: bool | None = None) -> dict:
    """Load saved ``task_XX.npz`` and ``task_XX.json`` outputs from ``data_dir``."""
    t0_json = data_dir / "task_00.json"
    if t0_json.exists():
        saved_cpw = int(json.loads(t0_json.read_text()).get("task", {}).get("cpw", 14))
        quick = saved_cpw < 20
    elif quick is None:
        quick = True
    tasks = get_task_table(quick=quick)
    npz_data = {}
    json_data = {}
    for t in tasks:
        idx = t["index"]
        npz_path = data_dir / f"task_{idx:02d}.npz"
        json_path = data_dir / f"task_{idx:02d}.json"
        if npz_path.exists():
            npz_data[t["name"]] = dict(np.load(npz_path, allow_pickle=True))
        if json_path.exists():
            json_data[t["name"]] = json.loads(json_path.read_text())
    return {"npz": npz_data, "json": json_data, "tasks": tasks}


# %% [markdown]
# <a id="6-visualizing-fields"></a>
# ## 6. Visualizing Geometry, Kottke Subpixel Mesh, Eigenmodes & Pulse Propagation
#
# Figure `00_setup_schematic.png` shows the complete simulation setup before examining any fields:
# - **Panel (a)** depicts the 2D $(x, y)$ tilted-facet waveguide domain ($x \in [-4.2, 1.8]\;\mu\text{m}$, $y \in [-6.0, 6.0]\;\mu\text{m}$) with 10-cell CPML boundaries on $\pm x, \pm y$ and periodic BCs in $z$, the ridge core ($w_1 = 1.75\;\mu\text{m}$, $n_{\text{core}} = 3.337$) in cladding ($n_{\text{clad}} = 3.311$), the $\theta_t = 7^\circ$ angled facet coated with a quarter-wave $\text{Al}_2\text{O}_3/\text{SiO}_x\text{N}_y$ film ($t_{\text{AR}} = 216.6\;\text{nm}$, $n_{\text{AR}} = 1.72$), the `ModeSource` at $x = -3.3\;\mu\text{m}$, the upstream `ModeMonitor` at $x = -2.6\;\mu\text{m}$, and the `Near2FarMonitor` at $x = +0.85\;\mu\text{m}$ in air.
# - **Panel (b)** shows the 9-dipole incoherent spontaneous-emission ensemble (`CurrentSource` $J_y$ at $x = -1.0\;\mu\text{m}$, $y_j \in [-0.42 w_2, +0.42 w_2]$) inside the bimodal waveguide ($w_2 = 2.80\;\mu\text{m}$) with the downstream two-mode `ModeMonitor` ($\text{TE}_0, \text{TE}_1$) at $x = +1.0\;\mu\text{m}$.
# - **Panel (c)** zooms into the locally refined non-uniform mesh across the quarter-wave AR coating ($x \in [0, 216.6]\;\text{nm}$, resolved with $10$ cells at $\Delta x_{\min} = 21.66\;\text{nm}$, $20\;\text{cells}/\lambda$) and the core sidewalls $y = \pm w_1/2$.
#
# ![Setup schematic and thin-layer mesh](figures/00_setup_schematic.png)
#
# We next compare staircase vs. Kottke subpixel permittivity averaging along the $\theta_t = 7^\circ$ inclined facet (`sled_schematic_and_mesh.png`), examine the exact vs. Wakis 2D FIT eigenmode profiles ($\text{TE}_0$ and $\text{TE}_1$), and visualize both steady-state frequency-domain fields and time-domain wavepacket reflection/refraction with full geometry overlays.

# %%
def plot_geometry_and_eigenmodes(results: dict, fig_dir: Path) -> None:
    """Generate Figures 0, 1, and 2: setup schematic, subpixel mesh, and eigenmodes."""
    figstyle.apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    bare = results["npz"]["bare_tilt_sweep"]
    ar_d = results["npz"]["ar_tilt_sweep"]
    bimod = results["npz"]["bimodal_tilt_sweep"]
    cpw_val = int(round(1.49e-6 / (h.N_CORE * float(bare["dx_min_nm"]) * 1e-9)))
    cpw_val = max(14, cpw_val)

    # --- Figure 0: 00_setup_schematic.png (Tilted facet + 9-dipole ensemble + AR mesh zoom) ---
    fig0, axes0 = figstyle.new_figure(
        nrows=1, ncols=3, width=13.5, height=4.4, gridspec_kw={"width_ratios": [1.1, 1.05, 0.95]}
    )
    sled_obj = h.build_sled_schematic_objects(theta_deg=7.0, width=h.WIDTH_SM, cpw=cpw_val, ar_coated=True)
    schematic.plot_schematic(
        sled_obj["solver"],
        plane="xy",
        ax=axes0[0],
        unit="um",
        structures=sled_obj["structures"],
        sources=sled_obj["sources"],
        monitors=sled_obj["monitors"],
        show_mesh=False,
        show_wave_packet=False,
        title="(a) Tasks 0–2: Tilted Facet & AR Setup",
    )
    h.tidy_sled_schematic_axes(axes0[0], xlim=(-4.2, 1.8), ylim=(-6.0, 7.6), is_dipole=False, n_pml=10)

    dip_obj = h.build_incoherent_dipole_schematic_objects(width=h.WIDTH_MM, cpw=cpw_val, n_unique_half=5)
    schematic.plot_schematic(
        dip_obj["solver"],
        plane="xy",
        ax=axes0[1],
        unit="um",
        structures=dip_obj["structures"],
        sources=dip_obj["sources"],
        monitors=dip_obj["monitors"],
        show_mesh=False,
        show_wave_packet=False,
        title="(b) Task 3: 9-Dipole Incoherent Ensemble",
    )
    h.tidy_sled_schematic_axes(axes0[1], xlim=(-2.4, 2.4), ylim=(-5.0, 6.4), is_dipole=True, n_pml=8)

    ax_m = axes0[2]
    t_ar_um = h.T_AR_COAT * 1e6
    w_half_um = 0.5 * h.WIDTH_SM * 1e6
    ax_m.axvspan(-0.18, 0.0, facecolor="#9ecae1", alpha=0.55, zorder=1)
    ax_m.axvspan(0.0, t_ar_um, facecolor="#fdae6b", alpha=0.65, zorder=1)
    ax_m.axvspan(t_ar_um, 0.36, facecolor="#f7fbff", alpha=0.65, zorder=1)
    schematic.draw_mesh(
        ax_m,
        sled_obj["grid"],
        plane="xy",
        unit="um",
        every=1,
        color="#222222",
        alpha=0.55,
        linewidth=0.45,
        show_cpw=False,
    )
    ax_m.axvline(0.0, color="#111111", lw=1.2, zorder=5)
    ax_m.axvline(t_ar_um, color="#111111", lw=1.2, zorder=5)
    ax_m.axhline(w_half_um, color="#b30000", ls="--", lw=1.1, zorder=5)
    ax_m.axhline(-w_half_um, color="#b30000", ls="--", lw=1.1, zorder=5)
    ax_m.set_xlim(-0.14, 0.34)
    ax_m.set_ylim(-1.05, 1.05)
    n_ar_cells = int(sled_obj["mesh_info"]["cells_across_feature"])
    dx_min_nm = float(sled_obj["mesh_info"]["dx_min_nm"])
    bbox_z = {"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "#444444", "lw": 0.5, "alpha": 0.90}
    ax_m.text(0.5 * t_ar_um, 0.42, f"λ/4 AR film\n216.6 nm: {n_ar_cells} cells", ha="center", va="center", fontsize=8.0, bbox=bbox_z, zorder=7)
    ax_m.text(-0.07, -0.42, "Core\n$n=3.337$", ha="center", va="center", fontsize=7.8, bbox=bbox_z, zorder=7)
    ax_m.text(0.28, -0.42, "Air\n$n=1.00$", ha="center", va="center", fontsize=7.8, bbox=bbox_z, zorder=7)
    ax_m.text(0.10, 0.90, f"$\\Delta x_{{\\min}} = {dx_min_nm:.1f}$ nm, {cpw_val} cells/$\\lambda$", ha="center", va="center", fontsize=8.0, bbox=bbox_z, zorder=7)
    ax_m.set_xlabel("$x$ [$\\mu$m]")
    ax_m.set_ylabel("$y$ [$\\mu$m]")
    ax_m.set_title("(c) Local Mesh at AR Coating & Core")
    figstyle.finish(fig0)
    figstyle.save(fig0, fig_dir / "00_setup_schematic.png", close=True)

    x_um = bare["x_coords"] * 1e6
    y_um = bare["y_coords"] * 1e6
    eps_kottke = bare["eps_y_kottke_7deg"] / np.min(bare["eps_y_kottke_7deg"])
    eps_stair = bare["eps_y_stair_7deg"] / np.min(bare["eps_y_stair_7deg"])
    eps_ar = ar_d["eps_y_ar_7deg"] / np.min(ar_d["eps_y_ar_7deg"])

    # --- Figure 1: Full 2D layout + Staircase vs Kottke zoom on 7 deg facet ---
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.5, height=4.2)

    ax = axes[0]
    im0 = ax.pcolormesh(
        x_um, y_um, np.sqrt(eps_ar).T, shading="auto", cmap=figstyle.CMAP_EPS, vmin=1.0, vmax=3.45
    )
    h.overlay_sled_geometry(
        ax,
        theta_deg=7.0,
        width=h.WIDTH_SM,
        ar_coated=True,
        line_color="white",
        show_monitors=True,
        show_snell_beam=True,
        fontsize=6.2,
    )
    ax.set_xlim(-4.2, 1.8)
    ax.set_ylim(-5.5, 5.5)
    ax.set_xlabel("$x$ [$\\mu$m]")
    ax.set_ylabel("$y$ [$\\mu$m]")
    ax.set_title("(a) 2D SLED Tilted-Facet Layout ($n(x,y)$, AR)")
    fig.colorbar(im0, ax=ax, label="Refractive index $n(x,y)$ [-]")

    y_line = np.linspace(-6.0, 6.0, 200)
    for ax_z, eps_map, title_str in [
        (axes[1], eps_stair, "(b) Staircase $\\varepsilon_y(x,y)$ ($\\theta_t=7^\\circ$)"),
        (axes[2], eps_kottke, "(c) Kottke Subpixel $\\varepsilon_y(x,y)$ ($\\theta_t=7^\\circ$)"),
    ]:
        im_z = ax_z.pcolormesh(
            x_um, y_um, eps_map.T, shading="auto", cmap=figstyle.CMAP_MAGNITUDE, vmin=1.0, vmax=11.3
        )
        ax_z.plot(y_line * np.tan(np.deg2rad(7.0)), y_line, "c--", lw=1.4, label="Exact facet ($7^\\circ$)")
        ax_z.axhline(0.5 * h.WIDTH_SM * 1e6, color="w", ls=":", lw=1.1)
        ax_z.axhline(-0.5 * h.WIDTH_SM * 1e6, color="w", ls=":", lw=1.1, label="Core $\\pm w_1/2$")
        ax_z.set_xlim(-0.55, 0.55)
        ax_z.set_ylim(-1.35, 1.35)
        ax_z.set_xlabel("$x$ [$\\mu$m]")
        ax_z.set_ylabel("$y$ [$\\mu$m]")
        ax_z.set_title(title_str)
        ax_z.legend(loc="upper left", fontsize=7.5)
        fig.colorbar(im_z, ax=ax_z, label="Permittivity $\\varepsilon_y$ [-]")

    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "sled_schematic_and_mesh.png", close=True)

    # --- Figure 2: Single-mode and Two-mode eigenmode profiles ---
    slab_sm = h.solve_1d_slab_modes(width=h.WIDTH_SM)
    slab_mm = h.solve_1d_slab_modes(width=h.WIDTH_MM)

    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.5, height=4.1)

    ax = axes[0]
    y_s = slab_sm["y"] * 1e6
    psi0_s = slab_sm["psi_0"] / np.max(np.abs(slab_sm["psi_0"]))
    w0_fit = slab_sm["w0_overlap"]
    g_fit = np.exp(-((slab_sm["y"] / w0_fit) ** 2))
    ey0_w = bare["ey0_mode"]
    if ey0_w[len(ey0_w) // 2] < 0:
        ey0_w = -ey0_w
    ey0_w = ey0_w / np.max(np.abs(ey0_w))

    ax.axvspan(-0.5 * h.WIDTH_SM * 1e6, 0.5 * h.WIDTH_SM * 1e6, color="#e0e8f5", alpha=0.7, label="Core ($w_1=1.75\\;\\mu$m)")
    ax.plot(y_s, psi0_s, "k-", lw=2.0, label=f"Exact 1D slab $\\mathrm{{TE}}_0$ ($n_{{\\mathrm{{eff}}}}={slab_sm['neff_0']:.4f}$)")
    ax.plot(
        y_s,
        g_fit,
        "--",
        color=figstyle.PALETTE[1],
        lw=1.8,
        label=f"Gaussian fit ($w_0={w0_fit*1e6:.3f}\\;\\mu$m, $99.65\\%$ overlap)",
    )
    ax.plot(
        bare["y_coords"] * 1e6,
        ey0_w,
        "o",
        color=figstyle.PALETTE[0],
        ms=3.5,
        alpha=0.8,
        label="Wakis 2D FIT ModeMonitor $\\mathrm{TE}_0$",
    )
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-0.1, 1.12)
    ax.set_xlabel("Transverse coordinate $y$ [$\\mu$m]")
    ax.set_ylabel("Normalized field $E_y(y) / \\max|E_y|$ [-]")
    ax.set_title("(a) Single-Mode Waveguide ($w_1=1.75\\;\\mu$m, $V_{\\mathrm{half}}=1.534 < \\pi/2$)")
    ax.legend(loc="upper right", fontsize=7.6)

    ax = axes[1]
    y_m = slab_mm["y"] * 1e6
    psi0_m = slab_mm["psi_0"] / np.max(np.abs(slab_mm["psi_0"]))
    psi1_m = slab_mm["psi_1"] / np.max(np.abs(slab_mm["psi_1"]))
    ey0_mm = bimod["ey0_mode_mm"]
    if ey0_mm[len(ey0_mm) // 2] < 0:
        ey0_mm = -ey0_mm
    ey0_mm = ey0_mm / np.max(np.abs(ey0_mm))
    ey1_mm = bimod["ey1_mode_mm"]
    if np.sum(ey1_mm[len(ey1_mm) // 2 :] * np.arange(len(ey1_mm) - len(ey1_mm) // 2)) < 0:
        ey1_mm = -ey1_mm
    ey1_mm = ey1_mm / np.max(np.abs(ey1_mm))

    ax.axvspan(-0.5 * h.WIDTH_MM * 1e6, 0.5 * h.WIDTH_MM * 1e6, color="#f5e8e0", alpha=0.7, label="Core ($w_2=2.80\\;\\mu$m)")
    ax.plot(y_m, psi0_m, "k-", lw=2.0, label=f"Exact $\\mathrm{{TE}}_0$ ($n_{{\\mathrm{{eff}},0}}={slab_mm['neff_0']:.4f}$)")
    ax.plot(y_m, psi1_m, "-", color=figstyle.PALETTE[2], lw=2.0, label=f"Exact $\\mathrm{{TE}}_1$ ($n_{{\\mathrm{{eff}},1}}={slab_mm['neff_1']:.4f}$)")
    ax.plot(bimod["y_coords_mm"] * 1e6, ey0_mm, "o", color=figstyle.PALETTE[0], ms=3.2, alpha=0.75, label="Wakis FIT $\\mathrm{TE}_0$")
    ax.plot(bimod["y_coords_mm"] * 1e6, ey1_mm, "^", color=figstyle.PALETTE[4], ms=3.2, alpha=0.75, label="Wakis FIT $\\mathrm{TE}_1$")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-1.12, 1.12)
    ax.set_xlabel("Transverse coordinate $y$ [$\\mu$m]")
    ax.set_ylabel("Normalized field $E_y(y) / \\max|E_y|$ [-]")
    ax.set_title("(b) Bimodal Waveguide ($w_2=2.80\\;\\mu$m, $V_{\\mathrm{half}}=2.454 \\in (\\pi/2, \\pi)$)")
    ax.legend(loc="lower right", fontsize=7.6)

    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "eigenmode_profiles.png", close=True)


def plot_fields_and_movie(results: dict, fig_dir: Path) -> None:
    """Generate steady-state DFT field maps (<=3 panels) and time-domain pulse movie/snapshots."""
    figstyle.apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    bare = results["npz"]["bare_tilt_sweep"]

    ey_0 = bare["ey_2d_0deg"]
    ey_7 = bare["ey_2d_7deg"]
    dx_um = bare["dft_x"] * 1e6
    dy_um = bare["dft_y"] * 1e6

    # --- Figure 3: 3-panel steady-state DFT field comparison (0 deg vs 7 deg) ---
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.5, height=4.2, sharey=True)
    re_0 = np.real(ey_0) / np.max(np.abs(ey_0))
    re_7 = np.real(ey_7) / np.max(np.abs(ey_7))
    i_db_0 = 10.0 * np.log10(np.maximum(np.abs(ey_0) ** 2 / np.max(np.abs(ey_0) ** 2), 1e-5))
    i_db_7 = 10.0 * np.log10(np.maximum(np.abs(ey_7) ** 2 / np.max(np.abs(ey_7) ** 2), 1e-5))

    im0 = axes[0].pcolormesh(
        dx_um, dy_um, re_0.T, shading="auto", cmap=figstyle.CMAP_DIVERGING, vmin=-0.85, vmax=0.85
    )
    h.overlay_sled_geometry(
        axes[0], theta_deg=0.0, width=h.WIDTH_SM, line_color="#222222", show_monitors=True, fontsize=6.0
    )
    axes[0].set_xlim(-3.8, 1.6)
    axes[0].set_ylim(-5.2, 5.2)
    axes[0].set_xlabel("$x$ [$\\mu$m]")
    axes[0].set_ylabel("$y$ [$\\mu$m]")
    axes[0].set_title("(a) $\\mathrm{Re}\\{\\tilde{E}_y\\}$ at $\\theta_t = 0^\\circ$ (Standing Wave)")
    fig.colorbar(im0, ax=axes[0], label="Normalized $\\mathrm{Re}(E_y)$")

    im1 = axes[1].pcolormesh(
        dx_um, dy_um, re_7.T, shading="auto", cmap=figstyle.CMAP_DIVERGING, vmin=-0.85, vmax=0.85
    )
    h.overlay_sled_geometry(
        axes[1], theta_deg=7.0, width=h.WIDTH_SM, line_color="#222222", show_monitors=True, show_snell_beam=True, fontsize=6.0
    )
    axes[1].set_xlim(-3.8, 1.6)
    axes[1].set_ylim(-5.2, 5.2)
    axes[1].set_xlabel("$x$ [$\\mu$m]")
    axes[1].set_title("(b) $\\mathrm{Re}\\{\\tilde{E}_y\\}$ at $\\theta_t = 7^\\circ$ (Deflected)")
    fig.colorbar(im1, ax=axes[1], label="Normalized $\\mathrm{Re}(E_y)$")

    im2 = axes[2].pcolormesh(
        dx_um, dy_um, i_db_7.T, shading="auto", cmap=figstyle.CMAP_MAGNITUDE, vmin=-35.0, vmax=0.0
    )
    h.overlay_sled_geometry(
        axes[2], theta_deg=7.0, width=h.WIDTH_SM, line_color="white", show_monitors=True, show_snell_beam=True, fontsize=6.0
    )
    axes[2].set_xlim(-3.8, 1.6)
    axes[2].set_ylim(-5.2, 5.2)
    axes[2].set_xlabel("$x$ [$\\mu$m]")
    axes[2].set_title("(c) $|\\tilde{E}_y|^2$ [dB] at $\\theta_t = 7^\\circ$")
    fig.colorbar(im2, ax=axes[2], label="Relative intensity [dB]")

    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "steady_state_fields_0_vs_7deg.png", close=True)

    # --- Companion 2-panel intensity comparison (0 deg vs 7 deg) ---
    fig_i, axes_i = figstyle.new_figure(nrows=1, ncols=2, width=11.2, height=4.1, sharey=True)
    for ax_db, th_val, idb_map, lbl in [
        (axes_i[0], 0.0, i_db_0, "(a) Intensity $|\\tilde{E}_y|^2$ [dB] at $\\theta_t = 0^\\circ$"),
        (axes_i[1], 7.0, i_db_7, "(b) Intensity $|\\tilde{E}_y|^2$ [dB] at $\\theta_t = 7^\\circ$"),
    ]:
        im_db = ax_db.pcolormesh(
            dx_um, dy_um, idb_map.T, shading="auto", cmap=figstyle.CMAP_MAGNITUDE, vmin=-35.0, vmax=0.0
        )
        h.overlay_sled_geometry(
            ax_db, theta_deg=th_val, width=h.WIDTH_SM, line_color="white", show_monitors=True, show_snell_beam=(th_val > 0), fontsize=6.2
        )
        ax_db.set_xlim(-3.8, 1.6)
        ax_db.set_ylim(-5.2, 5.2)
        ax_db.set_xlabel("$x$ [$\\mu$m]")
        ax_db.set_title(lbl)
        fig_i.colorbar(im_db, ax=ax_db, label="Relative intensity [dB]")
    axes_i[0].set_ylabel("$y$ [$\\mu$m]")
    figstyle.finish(fig_i)
    figstyle.save(fig_i, fig_dir / "steady_state_intensity_0_vs_7deg.png", close=True)

    # --- Figure 4 (.gif) and Figure 5 (.png): Time-domain pulse propagation at 7 deg ---
    # stored normalized (float16) with a movie_frames_scale factor to keep the committed data small
    frames = np.asarray(bare["movie_frames"], dtype=float) * (
        float(bare["movie_frames_scale"]) if "movie_frames_scale" in bare else 1.0
    )
    times_fs = bare["movie_times_fs"]
    x_um = bare["x_coords"] * 1e6
    y_um = bare["y_coords"] * 1e6
    vmax = 0.65 * float(np.max(np.abs(frames)))

    pil_images = []
    for k_f in range(len(frames)):
        fig_m, ax_m = figstyle.new_figure(nrows=1, ncols=1, width=6.4, height=4.4)
        im_m = ax_m.pcolormesh(
            x_um,
            y_um,
            frames[k_f].T / max(vmax, 1e-30),
            shading="auto",
            cmap=figstyle.CMAP_DIVERGING,
            vmin=-1.0,
            vmax=1.0,
        )
        h.overlay_sled_geometry(
            ax_m,
            theta_deg=7.0,
            width=h.WIDTH_SM,
            line_color="#222222",
            show_monitors=True,
            show_snell_beam=True,
            fontsize=6.2,
        )
        ax_m.set_xlim(-3.8, 1.6)
        ax_m.set_ylim(-5.2, 5.2)
        ax_m.set_xlabel("$x$ [$\\mu$m]")
        ax_m.set_ylabel("$y$ [$\\mu$m]")
        ax_m.set_title(
            f"Wakis 2D FIT Pulse at Tilted Facet ($\\theta_t=7^\\circ$, $t={times_fs[k_f]:.1f}$ fs)"
        )
        fig_m.colorbar(im_m, ax=ax_m, label="Normalized $E_y(x,y,t)$ [-]")
        figstyle.finish(fig_m)
        fig_m.canvas.draw()
        rgba = np.asarray(fig_m.canvas.buffer_rgba())
        pil_images.append(Image.fromarray(rgba[:, :, :3]))
        plt.close(fig_m)

    if pil_images:
        pil_images[0].save(
            fig_dir / "pulse_propagation_7deg.gif",
            save_all=True,
            append_images=pil_images[1:],
            duration=130,
            loop=0,
        )

    # Static 3-snapshot strip (Figure 5, <= 3 panels rule)
    sel_indices = np.linspace(2, len(frames) - 3, 3, dtype=int)
    fig_s, axes_s = figstyle.new_figure(nrows=1, ncols=3, width=13.5, height=4.1, sharey=True)
    im_k = None
    for idx_p, (ax_k, idx_f) in enumerate(zip(axes_s, sel_indices)):
        im_k = ax_k.pcolormesh(
            x_um,
            y_um,
            frames[idx_f].T / max(vmax, 1e-30),
            shading="auto",
            cmap=figstyle.CMAP_DIVERGING,
            vmin=-1.0,
            vmax=1.0,
        )
        h.overlay_sled_geometry(
            ax_k,
            theta_deg=7.0,
            width=h.WIDTH_SM,
            line_color="#222222",
            show_monitors=True,
            show_snell_beam=True,
            fontsize=5.8,
        )
        ax_k.set_xlim(-3.8, 1.6)
        ax_k.set_ylim(-5.0, 5.0)
        ax_k.set_xlabel("$x$ [$\\mu$m]")
        ax_k.set_title(f"({chr(ord('a') + idx_p)}) $t = {times_fs[idx_f]:.1f}$ fs")
    axes_s[0].set_ylabel("$y$ [$\\mu$m]")
    fig_s.colorbar(im_k, ax=axes_s[-1], label="Normalized $E_y(x,y,t)$ [-]")
    figstyle.finish(fig_s)
    figstyle.save(fig_s, fig_dir / "pulse_snapshots_7deg.png", close=True)


# %% [markdown]
# <a id="7-reflectivity-vs-tilt"></a>
# ## 7. High-Dynamic-Range Reflectivity $R_{00}(\theta_t)$: Bare vs. AR-Coated Facets
#
# Next we compare the Wakis 2D FIT modal reflectivity $R_{00}(\theta_t)$ (`cpw = 20`, 12 tilt angles $\theta_t \in \{0^\circ, 1^\circ, 2^\circ, 3^\circ, 4^\circ, 5^\circ, 6^\circ, 7^\circ, 8^\circ, 9^\circ, 10^\circ, 12^\circ\}$) across nearly eight orders of magnitude ($10^0$ down to $10^{-8}$) against:
# 1. **Marcuse (1989) Gaussian-mode formula** $R_{00}^{\text{Gauss}}(\theta_t) = R_f(\theta_t) \exp[-(2\pi n_{\text{eff}} w_0 \tan\theta_t / \lambda_0)^2]$,
# 2. **Exact 1D step-index slab overlap integral** $R_{00}^{\text{slab}}(\theta_t)$ (which predicts the sidewall-diffraction destructive interference null near $\theta_t = 9.7^\circ\text{–}10^\circ$ and the sidelobe at $12^\circ$), and
# 3. **Zah et al. (1987)** experimental residual reflectivity at $\theta_t = 7^\circ$ ($R \approx 2\text{–}3 \times 10^{-3}$ for a bare angled facet).
#
# Key findings from the `--full` (`cpw = 20`) bare and AR-coated sweeps (`Tasks 00` and `01`):
# - **Bare facet (`Task 00`)**: Wakis 2D FIT tracks the exact 1D step-index slab overlap integral closely across the entire angular range: $R_{00}(0^\circ) = 0.2793$ ($27.93\%$, vs. Fresnel $0.2893$), $R_{00}(2^\circ) = 0.1879$ (vs. slab $0.1932$), $R_{00}(4^\circ) = 6.146 \times 10^{-2}$ (vs. slab $6.335 \times 10^{-2}$, $3.0\%$ difference), and $R_{00}(7^\circ) = 3.546 \times 10^{-3}$ (vs. slab $3.473 \times 10^{-3}$, $2.1\%$ difference, and Marcuse Gaussian $2.027 \times 10^{-3}$). At $\theta_t = 10^\circ$, the 2D FIT reflectivity plunges to the Marcuse sidewall destructive-interference null at $R_{00}(10^\circ) = 1.386 \times 10^{-8}$ before rebounding to the diffraction sidelobe $R_{00}(12^\circ) = 3.487 \times 10^{-5}$ (slab: $5.749 \times 10^{-5}$; Gaussian: $1.011 \times 10^{-7}$).
# - **Single-layer $\lambda_0/4$ AR-coated facet (`Task 01`, $n_{\text{AR}} = 1.72$, $10$ grid cells across $t_{\text{AR}} = 216.57\;\text{nm}$)**: The coating suppresses normal-incidence reflection by $77.4\times$ to $R_{00}^{\text{AR}}(0^\circ) = 3.608 \times 10^{-3}$ (vs. 1D TMM $3.449 \times 10^{-3}$, $4.6\%$ difference). At $\theta_t = 7^\circ$, combining the tilted facet with the $\lambda_0/4$ coating yields $R_{00}^{\text{AR}}(7^\circ) = 4.355 \times 10^{-5}$ (vs. 1D slab + TMM $5.083 \times 10^{-5}$ and Gaussian + TMM $2.966 \times 10^{-5}$)—an $81.4\times$ reduction below the bare $7^\circ$ facet and $6{,}413\times$ below the bare cleaved facet. The AR-coated curve reaches a deep minimum at $\theta_t = 9^\circ$ ($R_{00}^{\text{AR}}(9^\circ) = 1.689 \times 10^{-7}$) followed by a weak sidewall sidelobe at $10^\circ\text{–}12^\circ$ ($3.953 \times 10^{-6}$ at $10^\circ$ and $6.652 \times 10^{-6}$ at $12^\circ$).

# %%
def plot_reflectivity_and_ar(results: dict, fig_dir: Path) -> None:
    """Generate Figures 6 and 7: R_00(theta_t) log-scale sweep and broadband AR spectrum."""
    figstyle.apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    bare = results["npz"]["bare_tilt_sweep"]
    ar_d = results["npz"]["ar_tilt_sweep"]
    pml_d = results["npz"]["pml_and_incoherent_study"]

    slab_sm = h.solve_1d_slab_modes(width=h.WIDTH_SM)
    neff0 = slab_sm["neff_0"]
    w0 = slab_sm["w0_overlap"]

    th_fine = np.linspace(0.0, 12.5, 251)
    r_gauss_bare = h.marcuse_gaussian_r00(th_fine, neff=neff0, w0=w0)
    r_slab_bare = h.slab_overlap_reflectivity(th_fine, slab_sm, mode_out=0)

    r_gauss_ar = h.marcuse_gaussian_r00(
        th_fine, neff=neff0, w0=w0, n_ar=h.N_AR_COAT, t_ar=h.T_AR_COAT
    )
    r_slab_ar = h.slab_overlap_reflectivity(
        th_fine, slab_sm, mode_out=0, n_ar=h.N_AR_COAT, t_ar=h.T_AR_COAT
    )

    # --- Figure 6: High-dynamic-range R00(theta_t) bare & AR-coated ---
    fig, ax = figstyle.new_figure(nrows=1, ncols=1, width=7.4, height=4.8)

    ax.semilogy(
        th_fine,
        r_gauss_bare,
        "k--",
        lw=1.8,
        label=f"Marcuse (1989) Gaussian ($w_0={w0*1e6:.3f}\\;\\mu$m, bare)",
    )
    ax.semilogy(
        th_fine,
        r_slab_bare,
        "-",
        color=figstyle.PALETTE[0],
        lw=2.1,
        label="Exact 1D step-index slab overlap (bare)",
    )
    ax.semilogy(
        bare["thetas"],
        bare["r00"],
        "o",
        color=figstyle.PALETTE[0],
        ms=6.5,
        mfc="white",
        mew=2.0,
        label="Wakis 2D FIT (bare facet, Kottke)",
    )

    ax.semilogy(
        th_fine,
        r_gauss_ar,
        ":",
        color=figstyle.PALETTE[1],
        lw=1.8,
        label=f"Marcuse Gaussian + $\\lambda/4$ AR ($n_{{\\mathrm{{AR}}}}={h.N_AR_COAT}$)",
    )
    ax.semilogy(
        th_fine,
        r_slab_ar,
        "-",
        color=figstyle.PALETTE[1],
        lw=1.9,
        alpha=0.85,
        label="Exact 1D slab overlap + $\\lambda/4$ AR coating",
    )
    ax.semilogy(
        ar_d["thetas"],
        ar_d["r00_ar"],
        "s",
        color=figstyle.PALETTE[1],
        ms=6.2,
        mfc="white",
        mew=1.8,
        label="Wakis 2D FIT (single-layer AR facet)",
    )

    ax.errorbar(
        [7.0],
        [2.5e-3],
        yerr=[[0.5e-3], [0.6e-3]],
        fmt="D",
        color=figstyle.PALETTE[2],
        ms=7.0,
        capsize=5,
        lw=2.0,
        label="Zah et al. (1987) exp. range at $7^\\circ$ ($2\\text{--}3\\times 10^{-3}$)",
    )

    r_fl_10 = float(pml_d["r_floor"][list(pml_d["n_pml"]).index(10)])
    ax.axhline(
        r_fl_10,
        color="gray",
        ls="-.",
        lw=1.4,
        label=f"Straight-WG CPML floor ($n_{{\\mathrm{{pml}}}}=10$: $R={r_fl_10:.2e}$)",
    )

    ax.set_xlim(-0.3, 12.4)
    ax.set_ylim(5e-9, 1.0)
    ax.set_xlabel("Waveguide facet tilt angle $\\theta_t$ [deg]")
    ax.set_ylabel("Fundamental modal reflectivity $R_{00}(\\theta_t)$ [-]")
    ax.set_title("Tilted-Facet Modal Reflectivity $R_{00}(\\theta_t)$: Wakis 2D FIT vs. Marcuse (1989)")
    ax.legend(loc="lower left", fontsize=7.6)

    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "reflectivity_vs_tilt_marcuse.png", close=True)

    # --- Figure 7: Broadband wavelength dependence R00(lambda) ---
    wls_um = (C_LIGHT / bare["freqs_broad"]) * 1e6
    sort_idx = np.argsort(wls_um)
    wls_um = wls_um[sort_idx]

    wls_fine = np.linspace(1.425e-6, 1.555e-6, 200)
    q_7 = 2.0 * neff0 * (2.0 * np.pi / wls_fine) * np.tan(np.deg2rad(7.0))
    ov_7_sq = (
        np.abs(
            np.trapezoid(
                np.exp(1j * np.outer(q_7, slab_sm["y"]))
                * (slab_sm["psi_0"] ** 2)[None, :],
                slab_sm["y"],
                axis=1,
            )
        )
        ** 2
    )
    r_tmm_0_bare = h.transfer_matrix_ar_reflectance(
        0.0, n_inc=neff0, wavelength=wls_fine
    )
    r_tmm_7_bare = (
        h.transfer_matrix_ar_reflectance(7.0, n_inc=neff0, wavelength=wls_fine)
        * ov_7_sq
    )
    r_tmm_0_ar = h.transfer_matrix_ar_reflectance(
        0.0, n_inc=neff0, wavelength=wls_fine, n_ar=h.N_AR_COAT, t_ar=h.T_AR_COAT
    )
    r_tmm_7_ar = (
        h.transfer_matrix_ar_reflectance(
            7.0, n_inc=neff0, wavelength=wls_fine, n_ar=h.N_AR_COAT, t_ar=h.T_AR_COAT
        )
        * ov_7_sq
    )
    n_ar_ideal = np.sqrt(neff0)
    t_ar_ideal = h.WAVELENGTH_0 / (4.0 * n_ar_ideal)
    r_tmm_7_ideal = (
        h.transfer_matrix_ar_reflectance(
            7.0, n_inc=neff0, wavelength=wls_fine, n_ar=n_ar_ideal, t_ar=t_ar_ideal
        )
        * ov_7_sq
    )

    fig, ax = figstyle.new_figure(nrows=1, ncols=1, width=7.6, height=4.6)
    ax.semilogy(wls_fine * 1e6, r_tmm_0_bare, "k--", lw=1.6, label="1D TMM $\\theta_t=0^\\circ$ (bare)")
    ax.semilogy(
        wls_um,
        bare["r00_spec_0deg"][sort_idx],
        "ko",
        ms=5.0,
        label="Wakis FIT $\\theta_t=0^\\circ$ (bare)",
    )
    ax.semilogy(wls_fine * 1e6, r_tmm_0_ar, "--", color=figstyle.PALETTE[0], lw=1.6, label="1D TMM $\\theta_t=0^\\circ$ ($\\lambda_0/4$ AR, $n=1.72$)")
    ax.semilogy(
        wls_um,
        ar_d["r00_ar_spec_0deg"][sort_idx],
        "s",
        color=figstyle.PALETTE[0],
        ms=5.0,
        label="Wakis FIT $\\theta_t=0^\\circ$ ($\\lambda_0/4$ AR)",
    )
    ax.semilogy(wls_fine * 1e6, r_tmm_7_bare, "-", color=figstyle.PALETTE[2], lw=1.8, label="1D slab + TMM $\\theta_t=7^\\circ$ (bare)")
    ax.semilogy(
        wls_um,
        bare["r00_spec_7deg"][sort_idx],
        "^",
        color=figstyle.PALETTE[2],
        ms=5.5,
        label="Wakis FIT $\\theta_t=7^\\circ$ (bare)",
    )
    ax.semilogy(wls_fine * 1e6, r_tmm_7_ar, "-", color=figstyle.PALETTE[1], lw=1.8, label="1D slab + TMM $\\theta_t=7^\\circ$ + AR ($n=1.72$)")
    ax.semilogy(
        wls_um,
        ar_d["r00_ar_spec_7deg"][sort_idx],
        "d",
        color=figstyle.PALETTE[1],
        ms=5.5,
        label="Wakis FIT $\\theta_t=7^\\circ$ + AR ($n=1.72$)",
    )
    ax.semilogy(
        wls_fine * 1e6,
        r_tmm_7_ideal,
        ":",
        color=figstyle.PALETTE[4],
        lw=1.8,
        label=f"1D slab + ideal $\\sqrt{{n_{{\\mathrm{{eff}}}}}}$ AR ($n={n_ar_ideal:.3f}$)",
    )

    ax.set_xlim(1.425, 1.555)
    ax.set_ylim(1e-7, 1.0)
    ax.set_xlabel("Free-space wavelength $\\lambda$ [$\\mu$m]")
    ax.set_ylabel("Modal power reflectivity $R_{00}(\\lambda)$ [-]")
    ax.set_title("Broadband Spectral Reflectivity $R_{00}(\\lambda)$: Bare vs. Single-Layer AR Coating")
    ax.legend(loc="lower right", fontsize=7.4, ncol=2)

    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "broadband_ar_spectrum.png", close=True)


# %% [markdown]
# <a id="8-mode-conversion-farfield"></a>
# ## 8. Higher-Order Mode Conversion $R_{01}(\theta_t)$ & Far-Field Beam Deflection
#
# In a wider waveguide ($w_2 = 2.80\;\mu\text{m}$, $V_{\text{half}} = 2.454 \in (\pi/2, \pi)$), Wakis 2D FIT (`Task 02`, `cpw = 20`) resolves the two guided modes with effective indices $n_{\text{eff},0} = 3.3444$ (exact 1D slab: $3.3318$, $0.38\%$ error) and $n_{\text{eff},1} = 3.3303$ (exact: $3.3178$, $0.38\%$ error). The tilted facet breaks transverse mirror symmetry and couples the incident fundamental $\text{TE}_0$ mode into the backward odd $\text{TE}_1$ mode:
# - At $\theta_t = 0^\circ$, $R_{01}(0^\circ) = 6.87 \times 10^{-21} \approx 0$ by even–odd parity orthogonality, while $R_{00}(0^\circ) = 0.2816$ (exact slab: $0.2898$).
# - At $\theta_t = 2^\circ$, $\text{TE}_0 \to \text{TE}_1$ conversion peaks at $R_{01}(2^\circ) = 9.168 \times 10^{-2}$ ($9.17\%$, agreeing with the exact 1D slab overlap $9.304 \times 10^{-2}$ to within $1.5\%$ and Hermite–Gaussian $9.347 \times 10^{-2}$).
# - For $\theta_t \ge 4^\circ$, odd-mode conversion dominates over fundamental self-reflection: at $\theta_t = 4^\circ$, $R_{01}(4^\circ) = 7.264 \times 10^{-2} > R_{00}(4^\circ) = 2.624 \times 10^{-2}$; at $\theta_t = 6^\circ$, $R_{01}(6^\circ) = 1.174 \times 10^{-2} \gg R_{00}(6^\circ) = 5.027 \times 10^{-4}$; and at the $\theta_t = 7^\circ$ SLED tilt angle, $R_{01}(7^\circ) = 2.268 \times 10^{-3}$ (exact slab: $2.220 \times 10^{-3}$, $2.2\%$ difference) exceeds $R_{00}(7^\circ) = 1.029 \times 10^{-5}$ (exact slab: $1.855 \times 10^{-5}$) by a factor of **$220\times$**.
#
# Projecting the transmitted air fields onto the far-field arc with `Near2FarMonitor` (`cpw = 20`) reproduces the experimental far-field patterns of **Zah et al. (1987), Fig. 2(a) and Fig. 2(b)**:
# - At $\theta_t = 7^\circ$, the fundamental $\text{TE}_0$ mode refracts into air as a single lobe peaking at $\theta_\parallel = +23.75^\circ$ (vs. Snell's law $\theta_{\text{out}} = \arcsin(n_{\text{core}}\sin 7^\circ) = +24.0^\circ$ relative to the facet normal).
# - The odd $\text{TE}_1$ mode radiates a **double-lobed far-field pattern** with a central phase-reversal null at $+24^\circ$ and two symmetric lobes at $+12.75^\circ$ and $+35.50^\circ$ ($+13.0^\circ$ and $+35.8^\circ$ in `--quick` at `cpw = 14`).

# %%
def plot_bimodal_and_farfield(results: dict, fig_dir: Path) -> None:
    """Generate Figures 8 and 9: bimodal mode conversion R01 and far-field patterns."""
    figstyle.apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    bare = results["npz"]["bare_tilt_sweep"]
    bimod = results["npz"]["bimodal_tilt_sweep"]

    slab_mm = h.solve_1d_slab_modes(width=h.WIDTH_MM)
    neff_avg = 0.5 * (slab_mm["neff_0"] + slab_mm["neff_1"])
    w0_mm = slab_mm["w0_overlap"]

    th_fine = np.linspace(0.0, 12.2, 245)
    r00_slab_mm = h.slab_overlap_reflectivity(th_fine, slab_mm, mode_out=0)
    r01_slab_mm = h.slab_overlap_reflectivity(th_fine, slab_mm, mode_out=1)
    r00_gauss_mm = h.marcuse_gaussian_r00(th_fine, neff=slab_mm["neff_0"], w0=w0_mm)
    r01_gauss_mm = h.marcuse_gaussian_r01(th_fine, neff=neff_avg, w0=w0_mm)

    # --- Figure 8: Bimodal R00 vs R01 mode conversion ---
    fig, ax = figstyle.new_figure(nrows=1, ncols=1, width=7.4, height=4.6)
    ax.semilogy(th_fine, r00_slab_mm, "-", color=figstyle.PALETTE[0], lw=2.0, label="Exact 1D slab $R_{00}(\\theta_t)$ ($\\mathrm{TE}_0 \\to \\mathrm{TE}_0$)")
    ax.semilogy(th_fine, r00_gauss_mm, "--", color=figstyle.PALETTE[0], lw=1.4, alpha=0.75, label=f"Marcuse Gaussian $R_{{00}}$ ($w_0={w0_mm*1e6:.2f}\\;\\mu$m)")
    ax.semilogy(
        bimod["thetas"],
        bimod["r00_mm"],
        "o",
        color=figstyle.PALETTE[0],
        ms=6.5,
        mfc="white",
        mew=2.0,
        label="Wakis 2D FIT $R_{00}(\\theta_t)$ ($w_2=2.80\\;\\mu$m)",
    )

    ax.semilogy(th_fine, np.maximum(r01_slab_mm, 1e-8), "-", color=figstyle.PALETTE[1], lw=2.0, label="Exact 1D slab $R_{01}(\\theta_t)$ ($\\mathrm{TE}_0 \\to \\mathrm{TE}_1$)")
    ax.semilogy(
        th_fine,
        np.maximum(r01_gauss_mm, 1e-8),
        "--",
        color=figstyle.PALETTE[1],
        lw=1.4,
        alpha=0.75,
        label="Hermite–Gaussian $R_{01} = R_f u^2 e^{-u^2}$",
    )
    ax.semilogy(
        bimod["thetas"],
        np.maximum(bimod["r01_mm"], 1e-7),
        "s",
        color=figstyle.PALETTE[1],
        ms=6.5,
        mfc="white",
        mew=2.0,
        label="Wakis 2D FIT $R_{01}(\\theta_t)$ ($\\mathrm{TE}_0 \\to \\mathrm{TE}_1$)",
    )

    ax.set_xlim(-0.2, 12.2)
    ax.set_ylim(1e-7, 1.0)
    ax.set_xlabel("Waveguide facet tilt angle $\\theta_t$ [deg]")
    ax.set_ylabel("Modal power reflectivity $R_{0m}(\\theta_t)$ [-]")
    ax.set_title("Mode Conversion in a Bimodal Guide ($w_2=2.80\\;\\mu$m): $R_{00}(\\theta_t)$ vs. $R_{01}(\\theta_t)$")
    ax.legend(loc="lower left", fontsize=7.6)

    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "bimodal_mode_conversion.png", close=True)

    # --- Figure 9: Far-field radiation patterns reproducing Zah et al. (1987) Fig. 2 ---
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.5, height=4.1)

    th_0 = bare["theta_norm_0deg"]
    ff_0 = bare["ff_0deg"] / np.max(bare["ff_0deg"])
    th_7 = bare["theta_norm_7deg"]
    ff_7 = bare["ff_7deg"] / np.max(bare["ff_7deg"])
    snell_7 = float(h.snell_farfield_angle_deg(7.0, neff=h.N_CORE))

    ax = axes[0]
    ax.plot(th_0, ff_0, "k--", lw=1.8, label="$\\theta_t=0^\\circ$ ($\\mathrm{TE}_0$, peak $0.0^\\circ$)")
    ax.plot(
        th_7,
        ff_7,
        "-",
        color=figstyle.PALETTE[0],
        lw=2.2,
        label=f"$\\theta_t=7^\\circ$ ($\\mathrm{{TE}}_0$, Wakis peak $+{th_7[np.argmax(ff_7)]:.1f}^\\circ$)",
    )
    ax.axvline(
        snell_7,
        color=figstyle.PALETTE[1],
        ls=":",
        lw=1.8,
        label=f"Snell law / Zah Fig. 2(a) ($+{snell_7:.1f}^\\circ$)",
    )
    ax.set_xlim(-45.0, 60.0)
    ax.set_ylim(0.0, 1.12)
    ax.set_xlabel("Angle from facet normal $\\theta_\\parallel$ [deg]")
    ax.set_ylabel("Normalized far-field intensity $U(\\theta_\\parallel)/U_{\\max}$ [-]")
    ax.set_title("(a) Single-Mode Guide ($w_1=1.75\\;\\mu$m): Snell Beam Deflection")
    ax.legend(loc="upper left", fontsize=7.6)

    ax = axes[1]
    th_mm = bimod["theta_norm_7deg"]
    ff_mm0 = bimod["ff_mm_te0_7deg"] / np.max(bimod["ff_mm_te0_7deg"])
    ff_mm1 = bimod["ff_mm_te1_7deg"] / np.max(bimod["ff_mm_te1_7deg"])
    ax.plot(th_mm, ff_mm0, "--", color=figstyle.PALETTE[0], lw=1.7, alpha=0.8, label="$\\mathrm{TE}_0$ launched ($w_2=2.80\\;\\mu$m)")
    ax.plot(th_mm, ff_mm1, "-", color=figstyle.PALETTE[4], lw=2.2, label="$\\mathrm{TE}_1$ launched (double-lobed, Zah Fig. 2b)")
    ax.axvline(snell_7, color="k", ls=":", lw=1.5, label=f"Refracted axis $+{snell_7:.1f}^\\circ$ (TE$_1$ null)")
    ax.set_xlim(-45.0, 60.0)
    ax.set_ylim(0.0, 1.12)
    ax.set_xlabel("Angle from facet normal $\\theta_\\parallel$ [deg]")
    ax.set_ylabel("Normalized far-field intensity $U(\\theta_\\parallel)/U_{\\max}$ [-]")
    ax.set_title("(b) Bimodal Guide ($w_2=2.80\\;\\mu$m, $\\theta_t=7^\\circ$): $\\mathrm{TE}_0$ vs. $\\mathrm{TE}_1$ Far Field")
    ax.legend(loc="upper left", fontsize=7.6)

    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "farfield_radiation_zah1987.png", close=True)


# %% [markdown]
# <a id="9-sled-ripple-and-pml"></a>
# ## 9. SLED Cavity Spectral Ripple (Hakki–Paoli) & PML Dynamic-Range Floor
#
# Using the `--full` simulated modal reflectivities from Wakis 2D FIT (`cpw = 20`: $R_{00}(0^\circ) = 0.2793$, $R_{00}(7^\circ,\text{bare}) = 3.546 \times 10^{-3}$, and $R_{00}(7^\circ,\text{AR}) = 4.355 \times 10^{-5}$; in `--quick` at `cpw = 14` these were $0.2749$, $3.403 \times 10^{-3}$, and $3.829 \times 10^{-5}$), we evaluate the Fabry–Pérot amplified spontaneous emission (ASE) spectrum of an $L = 380\;\mu\text{m}$ InGaAsP SLED cavity at single-pass gain $G_s = 120$ ($20.8\;\text{dB}$, with rear facet $R_1 = 3.546 \times 10^{-3}$) and map the Hakki–Paoli ripple contrast $\Delta G_{\text{dB}}$ as a function of output-facet reflectivity $R_2$:
# - With two bare $7^\circ$ facets ($R_1 = R_2 = 3.546 \times 10^{-3}$), the round-trip feedback $\gamma_{\text{rt}} = G_s R_1 = 0.4256$ creates a strong $\Delta G_{\text{dB}} = 7.89\;\text{dB}$ ($7.895\;\text{dB}$) peak-to-valley spectral modulation ($7.53\;\text{dB}$ in `--quick`).
# - Applying the single-layer $\lambda_0/4$ AR coating to the $7^\circ$ output facet ($R_2 = 4.355 \times 10^{-5}$, effective cavity reflectivity $\sqrt{R_1 R_2} = 3.930 \times 10^{-4}$) suppresses the ripple to $\Delta G_{\text{dB}} = 0.82\;\text{dB} < 1\;\text{dB}$ ($0.75\;\text{dB}$ in `--quick`), and AR-coating both $7^\circ$ facets ($R_1 = R_2 = 4.355 \times 10^{-5}$) reduces the ripple to $\Delta G_{\text{dB}} = 0.091\;\text{dB}$.
# - In **Task 03** (`cpw = 20`), sweeping the straight-waveguide CPML absorber thickness across $n_{\text{pml}} \in \{6, 8, 10, 12, 14\}$ reduces the parasitic reflection floor from $R_{\text{floor}}(6) = 1.91 \times 10^{-5}$ and $R_{\text{floor}}(8) = 2.53 \times 10^{-7}$ down to $R_{\text{floor}}(10) = 1.86 \times 10^{-12}$, $R_{\text{floor}}(12) = 2.91 \times 10^{-13}$, and $R_{\text{floor}}(14) = 6.25 \times 10^{-14}$—more than three orders of magnitude below the $10^\circ$ Marcuse null ($1.39 \times 10^{-8}$).

# %%
def plot_sled_ripple_and_pml(results: dict, fig_dir: Path) -> None:
    """Generate Figures 10 and 11: SLED ASE spectral ripple and CPML floor."""
    figstyle.apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    bare = results["npz"]["bare_tilt_sweep"]
    ar_d = results["npz"]["ar_tilt_sweep"]
    pml_d = results["npz"]["pml_and_incoherent_study"]

    idx_7 = list(bare["thetas"]).index(7.0)
    r_bare_7 = float(bare["r00"][idx_7])
    r_ar_7 = float(ar_d["r00_ar"][idx_7])

    # --- Figure 10: SLED Fabry-Perot ASE spectrum and Hakki-Paoli ripple ---
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.2)

    wls_zoom = np.linspace(1.4795e-6, 1.4845e-6, 2500)
    g_s_val = 120.0
    spec_bare, env_bare, rip_bare = h.sled_ase_spectrum(
        wls_zoom, g_peak_linear=g_s_val, r1=r_bare_7, r2=r_bare_7
    )
    spec_ar, _, rip_ar = h.sled_ase_spectrum(
        wls_zoom, g_peak_linear=g_s_val, r1=r_bare_7, r2=r_ar_7
    )

    ax = axes[0]
    ax.plot(
        wls_zoom * 1e9,
        10.0 * np.log10(np.maximum(spec_bare, 1e-6)),
        "-",
        color=figstyle.PALETTE[0],
        lw=1.3,
        alpha=0.85,
        label=f"Bare $7^\\circ$ facets ($R_1=R_2={r_bare_7:.2e}$, $\\Delta G={rip_bare:.2f}$ dB)",
    )
    ax.plot(
        wls_zoom * 1e9,
        10.0 * np.log10(np.maximum(spec_ar, 1e-6)),
        "-",
        color=figstyle.PALETTE[1],
        lw=1.8,
        label=f"$7^\\circ$ + AR output ($R_2={r_ar_7:.2e}$, $\\Delta G={rip_ar:.2f}$ dB)",
    )
    ax.plot(
        wls_zoom * 1e9,
        10.0 * np.log10(np.maximum(env_bare, 1e-6)),
        "k--",
        lw=1.4,
        label="Ideal travelling-wave ASE envelope ($R=0$)",
    )
    ax.set_xlim(1479.5, 1484.5)
    ax.set_ylim(-5.0, 5.2)
    ax.set_xlabel("Free-space wavelength $\\lambda$ [nm]")
    ax.set_ylabel("Relative ASE spectral density [dB]")
    ax.set_title("(a) SLED Longitudinal Mode Ripple ($L=380\\;\\mu$m, $G_s=20.8$ dB)")
    ax.legend(loc="lower center", fontsize=7.5)

    ax = axes[1]
    r_eff_grid = np.logspace(-6, -1, 400)
    for g_idx, g_db in enumerate((15.0, 20.8, 25.0, 30.0)):
        g_lin = 10.0 ** (0.1 * g_db)
        gamma = g_lin * r_eff_grid
        valid = gamma < 0.995
        rip_curve = h.hakki_paoli_ripple_db(gamma[valid])
        ax.loglog(
            r_eff_grid[valid],
            rip_curve,
            color=figstyle.PALETTE[g_idx],
            lw=1.9,
            label=f"Single-pass gain $G_s = {g_db:.1f}$ dB ($\\times{g_lin:.0f}$)",
        )

    ax.axvline(
        r_bare_7,
        color=figstyle.PALETTE[0],
        ls="--",
        lw=1.5,
        label=f"Bare $7^\\circ$ facet ($R_{{00}}={r_bare_7:.2e}$)",
    )
    ax.axvline(
        np.sqrt(r_bare_7 * r_ar_7),
        color=figstyle.PALETTE[1],
        ls="-.",
        lw=1.5,
        label=f"Hybrid $7^\\circ$ bare/AR ($\\sqrt{{R_1 R_2}}={np.sqrt(r_bare_7*r_ar_7):.2e}$)",
    )
    ax.axvline(
        r_ar_7,
        color=figstyle.PALETTE[2],
        ls=":",
        lw=1.6,
        label=f"Dual $7^\\circ$ + AR ($R_{{00}}={r_ar_7:.2e}$)",
    )
    ax.axhline(1.0, color="gray", ls="--", lw=1.1, label="OCT 1 dB ripple specification")

    ax.set_xlim(1e-6, 1e-1)
    ax.set_ylim(1e-2, 40.0)
    ax.set_xlabel("Effective modal facet reflectivity $R_{\\mathrm{eff}} = \\sqrt{R_1 R_2}$ [-]")
    ax.set_ylabel("Hakki–Paoli peak-to-valley ripple $\\Delta G_{\\mathrm{dB}}$ [dB]")
    ax.set_title("(b) Fabry–Pérot Ripple Contrast vs. Residual Facet Reflectivity")
    ax.legend(loc="upper left", fontsize=7.2)

    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "sled_ase_spectral_ripple.png", close=True)

    # --- Figure 11: Straight-waveguide CPML parasitic reflectivity floor ---
    fig, ax = figstyle.new_figure(nrows=1, ncols=1, width=7.0, height=4.4)
    n_pmls = pml_d["n_pml"]
    r_fls = pml_d["r_floor"]
    ax.semilogy(
        n_pmls,
        r_fls,
        "ko-",
        ms=6.5,
        lw=2.0,
        mfc=figstyle.PALETTE[5],
        label="Straight-WG into CPML parasitic floor $R_{\\mathrm{floor}}(n_{\\mathrm{pml}})$",
    )
    ax.axhline(
        r_bare_7,
        color=figstyle.PALETTE[0],
        ls="--",
        lw=1.6,
        label=f"Bare $7^\\circ$ facet $R_{{00}}(7^\\circ) = {r_bare_7:.2e}$",
    )
    ax.axhline(
        r_ar_7,
        color=figstyle.PALETTE[1],
        ls="-.",
        lw=1.6,
        label=f"AR-coated $7^\\circ$ facet $R_{{00}}^{{\\mathrm{{AR}}}}(7^\\circ) = {r_ar_7:.2e}$",
    )
    idx_10 = list(bare["thetas"]).index(10.0)
    r_bare_10 = float(bare["r00"][idx_10])
    ax.axhline(
        r_bare_10,
        color=figstyle.PALETTE[2],
        ls=":",
        lw=1.6,
        label=f"Marcuse slab null $R_{{00}}(10^\\circ) = {r_bare_10:.2e}$",
    )
    ax.set_xlabel("Number of CPML absorber cells $n_{\\mathrm{pml}}$ [-]")
    ax.set_ylabel("Measured modal power reflectivity $R_{00}$ [-]")
    ax.set_title("High-Dynamic-Range Measurement Floor vs. CPML Thickness $n_{\\mathrm{pml}}$")
    ax.set_xticks(n_pmls)
    ax.set_ylim(1e-14, 1e-2)
    ax.legend(loc="upper right", fontsize=7.6)

    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "pml_reflectivity_floor.png", close=True)


# %% [markdown]
# <a id="10-convergence-studies"></a>
# ## 10. Incoherent Dipole Ensemble Convergence & Subpixel Grid Convergence
#
# Finally, we examine two essential numerical verification studies:
# 1. **Spontaneous-Emission Incoherent Dipole Averaging**: Using the 9-dipole ensemble in the bimodal waveguide (`Task 03`, `cpw = 20`), we compare the 2D intensity distribution of the physical **incoherent power sum** $\frac{1}{N_{\text{dip}}}\sum_j |E_{y,j}(x,y)|^2$ ($\eta_{\text{TE}_1}^{\text{incoh}} = 0.4074$, vs. $0.4072$ at `cpw = 14`, a $0.05\%$ difference) against the unphysical **coherent field sum** $|\frac{1}{N_{\text{dip}}}\sum_j E_{y,j}(x,y)|^2$ ($\eta_{\text{TE}_1}^{\text{coh}} = 4.10 \times 10^{-6}$, vs. $8.38 \times 10^{-6}$ at `cpw = 14`), and plot the convergence of the modal coupling error with respect to $N_{\text{dip}}$ and $N_{\text{real}}$.
# 2. **Grid Resolution & Kottke Subpixel Convergence**: Using `Tasks 04–10` across the `--full` resolution ladder `cpw` $\in \{14, 20, 28\}$ with the fine Kottke reference at `cpw = 40` ($\Delta x_{\min} = 10.83\;\text{nm}$, $\Delta x_{\text{bg}} = 14.28\;\text{nm}$, $20$ cells across $t_{\text{AR}}$)—plus the coarse `--quick` level at `cpw = 10` ($\Delta x_{\min} = 36.09\;\text{nm}$, $6$ cells across $t_{\text{AR}}$)—we evaluate $n_{\text{eff},0}$, $R_{00}(7^\circ)$, $R_{00}(10^\circ)$, and $R_{\text{AR}}(0^\circ)$ for both Kottke subpixel smoothing and staircase discretization:
#
# | Level (`cpw`) | $\Delta x_{\text{bg}}$ [nm] | $\Delta x_{\min}$ [nm] | Cells Across $t_{\text{AR}}$ | $n_{\text{eff},0}$ (exact $3.3276$) | $R_{00}(7^\circ)$ Kottke / Stair | $R_{00}(10^\circ)$ Kottke / Stair | $R_{\text{AR}}(0^\circ)$ Kottke / Stair |
# | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | $10$ (`--quick` min) | $40.81$ | $36.09$ | $6$ | $3.3634$ ($1.08\%$) | $3.323 \times 10^{-3}$ / $1.928 \times 10^{-3}$ | $2.069 \times 10^{-7}$ / $4.551 \times 10^{-6}$ | $2.910 \times 10^{-3}$ / $5.289 \times 10^{-3}$ |
# | $14$ (`--quick` main) | $40.81$ | $30.94$ | $7$ | $3.3537$ ($0.78\%$) | $3.403 \times 10^{-3}$ / $3.275 \times 10^{-3}$ | $1.410 \times 10^{-7}$ / $8.281 \times 10^{-6}$ | $3.151 \times 10^{-3}$ / $4.868 \times 10^{-3}$ |
# | $20$ (`--full` main) | $28.57$ | $21.66$ | $10$ | $3.3402$ ($0.38\%$) | $3.546 \times 10^{-3}$ / $3.284 \times 10^{-3}$ | $1.386 \times 10^{-8}$ / $1.821 \times 10^{-6}$ | $3.608 \times 10^{-3}$ / $4.439 \times 10^{-3}$ |
# | $28$ (`--full` conv) | $20.41$ | $15.47$ | $14$ | $3.3339$ ($0.19\%$) | $3.612 \times 10^{-3}$ / $4.080 \times 10^{-3}$ | $1.648 \times 10^{-8}$ / $8.165 \times 10^{-6}$ | $3.831 \times 10^{-3}$ / $4.253 \times 10^{-3}$ |
# | $40$ (`--full` ref) | $14.28$ | $10.83$ | $20$ | $3.3306$ ($0.09\%$) | $3.648 \times 10^{-3}$ / — | $3.131 \times 10^{-8}$ / — | $3.949 \times 10^{-3}$ / — |
#
# **Convergence Verdict from the Full Ladder (`cpw = 14, 20, 28, 40`)**:
# - **Effective index $n_{\text{eff},0}$**: Converges monotonically with second-order $O(\Delta x_{\min}^2)$ accuracy; the relative error against the exact 1D slab dispersion ($3.327586$) drops from $0.784\%$ (`cpw = 14`) $\to 0.379\%$ (`cpw = 20`) $\to 0.191\%$ (`cpw = 28`) $\to 0.092\%$ (`cpw = 40`), with a relative change of **$0.099\%$** between the last two levels (`cpw = 28` and `40`).
# - **Tilted-facet reflectivity $R_{00}(7^\circ)$**: With Kottke subpixel smoothing, $R_{00}(7^\circ)$ converges monotonically ($3.403 \times 10^{-3} \to 3.546 \times 10^{-3} \to 3.612 \times 10^{-3} \to 3.648 \times 10^{-3}$), with a **$0.98\%$** relative change between the last two levels (`cpw = 28` and `40`) and a $2.78\%$ difference between the accepted `--full` sweep resolution (`cpw = 20`) and the `cpw = 40` reference. By contrast, **staircase discretization** fails to converge monotonically: while `cpw = 14` ($3.275 \times 10^{-3}$) and `cpw = 20` ($3.284 \times 10^{-3}$) appeared artificially close in `--quick`, the `--full` run at `cpw = 28` jumps by $+24.2\%$ to $4.080 \times 10^{-3}$ ($11.8\%$ above the Kottke `cpw = 40` reference) as discrete stair steps migrate across the $7^\circ$ inclined facet.
# - **AR-coated normal reflectivity $R_{\text{AR}}(0^\circ)$**: With exact node snapping across the $216.57\;\text{nm}$ quarter-wave layer ($7, 10, 14, 20$ cells), Kottke converges smoothly ($3.151 \times 10^{-3} \to 3.608 \times 10^{-3} \to 3.831 \times 10^{-3} \to 3.949 \times 10^{-3}$), with a **$3.00\%$** relative change between `cpw = 28` and `40` ($5.80\%$ between `cpw = 20` and `28`), while staircase approaches from above ($4.868 \times 10^{-3} \to 4.439 \times 10^{-3} \to 4.253 \times 10^{-3}$).
# - **Marcuse $10^\circ$ interference null $R_{00}(10^\circ)$**: Kottke resolves the deep destructive-interference zero in the $1.39\text{–}3.13 \times 10^{-8}$ range across `cpw = 20, 28, 40` ($\sim 10^{-7}$ relative to the normal-incidence reflection $R_{00}(0^\circ) = 0.2793$), whereas staircase remains trapped two orders of magnitude higher at $1.82\text{–}8.28 \times 10^{-6}$ due to facet staircase scattering.

# %%
def plot_convergence_studies(results: dict, fig_dir: Path) -> None:
    """Generate Figures 12, 13, and 14: incoherent dipole maps/convergence and grid convergence."""
    figstyle.apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    pml_d = results["npz"]["pml_and_incoherent_study"]

    # --- Figure 12: Incoherent power average vs Coherent field sum 2D maps ---
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.5, height=4.2)
    dx_um = pml_d["x_coords"] * 1e6
    dy_um = pml_d["y_coords"] * 1e6
    mask_wg = dx_um > -0.35
    i_incoh = np.clip(
        pml_d["intensity_incoh"] / np.max(pml_d["intensity_incoh"][mask_wg, :]), 0.0, 1.0
    )
    i_coh = np.clip(
        pml_d["intensity_coh"] / np.max(pml_d["intensity_coh"][mask_wg, :]), 0.0, 1.0
    )
    y_dips = pml_d["y_dip_all"] * 1e6

    for ax_i, imap, t_str in [
        (
            axes[0],
            i_incoh,
            "(a) Incoherent Power Average $\\langle |E_y|^2 \\rangle$ ($N_{\\mathrm{dip}}=9$)",
        ),
        (
            axes[1],
            i_coh,
            "(b) Coherent Field Sum $|\\langle E_y \\rangle|^2$ (Phased-Array Artifact)",
        ),
    ]:
        im = ax_i.pcolormesh(
            dx_um, dy_um, imap.T, shading="auto", cmap=figstyle.CMAP_MAGNITUDE, vmin=0.0, vmax=1.0
        )
        h.overlay_sled_geometry(
            ax_i,
            width=h.WIDTH_MM,
            is_dipole_ensemble=True,
            dipole_y_um=y_dips,
            line_color="white",
            show_monitors=True,
            fontsize=5.8,
        )
        ax_i.set_xlabel("$x$ [$\\mu$m]")
        ax_i.set_ylabel("$y$ [$\\mu$m]")
        ax_i.set_title(t_str)
        fig.colorbar(im, ax=ax_i, label="Normalized guided intensity [-]")

    # Panel (c): Per-dipole modal power coupling |a_0,j|^2 and |a_1,j|^2 vs y_j
    ax = axes[2]
    p0_norm = pml_d["a0_dipoles"] / np.max(pml_d["a0_dipoles"])
    p1_norm = pml_d["a1_dipoles"] / np.max(pml_d["a0_dipoles"])
    slab_mm = h.solve_1d_slab_modes(width=h.WIDTH_MM)
    y_s = slab_mm["y"] * 1e6
    psi0_sq = (slab_mm["psi_0"] ** 2) / np.max(slab_mm["psi_0"] ** 2)
    psi1_sq = (
        (slab_mm["psi_1"] ** 2)
        / np.max(slab_mm["psi_0"] ** 2)
        * (
            np.max(p1_norm)
            / np.max((slab_mm["psi_1"] ** 2) / np.max(slab_mm["psi_0"] ** 2))
        )
    )

    ax.plot(y_s, psi0_sq, "-", color=figstyle.PALETTE[0], lw=1.8, label="Exact slab $|\\psi_0(y)|^2$ ($\\mathrm{TE}_0$)")
    ax.plot(y_s, psi1_sq, "--", color=figstyle.PALETTE[1], lw=1.8, label="Exact slab $|\\psi_1(y)|^2$ ($\\mathrm{TE}_1$)")
    ax.plot(y_dips, p0_norm, "o", color=figstyle.PALETTE[0], ms=6.0, label="Wakis FIT dipole $|a_{0,j}^+|^2$")
    ax.plot(y_dips, p1_norm, "s", color=figstyle.PALETTE[1], ms=6.0, label="Wakis FIT dipole $|a_{1,j}^+|^2$")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("Dipole lateral position $y_j$ [$\\mu$m]")
    ax.set_ylabel("Normalized modal power coupling [-]")
    ax.set_title(
        f"(c) Per-Dipole Coupling ($\\eta_{{\\mathrm{{TE1}}}}^{{\\mathrm{{incoh}}}}={pml_d['eta_te1_incoh']:.2f}$, $\\eta_{{\\mathrm{{TE1}}}}^{{\\mathrm{{coh}}}}={pml_d['eta_te1_coh']:.1e}$)"
    )
    ax.legend(loc="upper center", fontsize=7.4)

    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "incoherent_dipole_maps.png", close=True)

    # --- Figure 13: Incoherent dipole convergence vs N_dip and N_real ---
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.1)

    ax = axes[0]
    nd_grid = pml_d["n_dip_grid"]
    ax.semilogy(
        nd_grid,
        np.maximum(pml_d["quad_err_te0"], 1e-5),
        "o-",
        color=figstyle.PALETTE[0],
        ms=6.0,
        lw=1.9,
        label="Spatial quadrature error ($\\mathrm{TE}_0$ power)",
    )
    ax.semilogy(
        nd_grid,
        np.maximum(pml_d["quad_err_te1"], 1e-5),
        "s--",
        color=figstyle.PALETTE[1],
        ms=6.0,
        lw=1.9,
        label="Spatial quadrature error ($\\mathrm{TE}_1$ power)",
    )
    ax.semilogy(
        nd_grid,
        0.55 / nd_grid,
        "k:",
        lw=1.5,
        label="$O(N_{\\mathrm{dip}}^{-1})$ uniform-weight reference",
    )
    ax.set_xlabel("Number of dipole positions across core $N_{\\mathrm{dip}}$ [-]")
    ax.set_ylabel("Relative quadrature error $|\\bar{P}_{N} - P_{\\infty}| / P_{\\infty}$ [-]")
    ax.set_title("(a) Deterministic Spatial Quadrature Convergence vs. $N_{\\mathrm{dip}}$")
    ax.legend(loc="upper right", fontsize=7.6)

    ax = axes[1]
    nr_grid = pml_d["n_real_grid"]
    ax.loglog(
        nr_grid,
        pml_d["rel_err_mc_te0"],
        "o-",
        color=figstyle.PALETTE[0],
        ms=6.0,
        lw=1.9,
        label="Random-phase RMS error ($\\mathrm{TE}_0$)",
    )
    ax.loglog(
        nr_grid,
        pml_d["rel_err_mc_te1"],
        "s--",
        color=figstyle.PALETTE[1],
        ms=6.0,
        lw=1.9,
        label="Random-phase RMS error ($\\mathrm{TE}_1$)",
    )
    ax.loglog(
        nr_grid,
        0.85 / np.sqrt(nr_grid),
        "k:",
        lw=1.6,
        label="$O(N_{\\mathrm{real}}^{-1/2})$ Monte Carlo scaling",
    )
    ax.set_xlabel("Number of random-phase realizations $N_{\\mathrm{real}}$ [-]")
    ax.set_ylabel("RMS relative error vs. exact incoherent sum [-]")
    ax.set_title("(b) Stochastic Phase-Realization Convergence vs. $N_{\\mathrm{real}}$")
    ax.legend(loc="upper right", fontsize=7.6)

    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "incoherent_dipole_convergence.png", close=True)

    # --- Figure 14: Grid resolution convergence (R00(7 deg), R00(10 deg), R_AR(0 deg)) ---
    ref_npz = results["npz"]["kottke_reference"]
    r7_ref = float(ref_npz["r00_7deg"])
    r10_ref = float(ref_npz["r00_10deg"])
    rar0_ref = float(ref_npz["r_ar_0deg"])
    neff_ref = float(ref_npz["neff_0"])
    cpw_ref = int(ref_npz["cpw"])
    slab_sm = h.solve_1d_slab_modes(width=h.WIDTH_SM)
    neff_exact = float(slab_sm["neff_0"])
    r_ar0_tmm = float(
        h.transfer_matrix_ar_reflectance(
            0.0, n_inc=neff_exact, n_ar=h.N_AR_COAT, t_ar=h.T_AR_COAT
        )
    )

    kottke_cpw, kottke_h, kottke_r7, kottke_r10, kottke_rar0, kottke_neff = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    stair_cpw, stair_h, stair_r7, stair_r10, stair_rar0 = [], [], [], [], []
    for t in results["tasks"]:
        if t["kind"] == "conv_point" and t["name"] != "kottke_reference":
            d = results["npz"][t["name"]]
            if str(d["averaging"]) == "kottke":
                kottke_cpw.append(int(d["cpw"]))
                kottke_h.append(float(d["dx_min_nm"]))
                kottke_r7.append(float(d["r00_7deg"]))
                kottke_r10.append(float(d["r00_10deg"]))
                kottke_rar0.append(float(d["r_ar_0deg"]))
                kottke_neff.append(float(d["neff_0"]))
            else:
                stair_cpw.append(int(d["cpw"]))
                stair_h.append(float(d["dx_min_nm"]))
                stair_r7.append(float(d["r00_7deg"]))
                stair_r10.append(float(d["r00_10deg"]))
                stair_rar0.append(float(d["r_ar_0deg"]))

    kottke_err = np.abs(np.asarray(kottke_r7) - r7_ref) / r7_ref
    stair_err = np.abs(np.asarray(stair_r7) - r7_ref) / r7_ref
    neff_err = np.abs(np.asarray(kottke_neff) - neff_exact) / neff_exact

    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.5, height=4.2)

    ax = axes[0]
    ax.plot(
        kottke_cpw + [cpw_ref],
        np.asarray(kottke_r7 + [r7_ref]) * 1e3,
        "o-",
        color=figstyle.PALETTE[0],
        ms=6.0,
        lw=2.0,
        label="Kottke $R_{00}(7^\\circ)$ (tilted facet)",
    )
    ax.plot(
        stair_cpw,
        np.asarray(stair_r7) * 1e3,
        "s--",
        color=figstyle.PALETTE[1],
        ms=6.0,
        lw=2.0,
        label="Staircase $R_{00}(7^\\circ)$ (tilted facet)",
    )
    ax.plot(
        kottke_cpw + [cpw_ref],
        np.asarray(kottke_rar0 + [rar0_ref]) * 1e3,
        "^-.",
        color=figstyle.PALETTE[2],
        ms=6.0,
        lw=2.0,
        label="Node-snapped $R_{\\mathrm{AR}}(0^\\circ)$ ($\\geq 6$ cells in AR)",
    )
    ax.axhline(
        r_ar0_tmm * 1e3,
        color=figstyle.PALETTE[2],
        ls=":",
        lw=1.5,
        label=f"1D TMM $R_{{\\mathrm{{AR}}}}(0^\\circ) = {r_ar0_tmm*1e3:.2f}\\times 10^{{-3}}$",
    )
    ax.set_xlabel("Resolution [cells per wavelength in core, cpw]")
    ax.set_ylabel("Modal reflectivity $R_{00}$ [$\\times 10^{-3}$]")
    ax.set_title("(a) $R_{00}(7^\\circ)$ & $R_{\\mathrm{AR}}(0^\\circ)$ vs. Resolution")
    ax.legend(loc="best", fontsize=7.2)

    ax = axes[1]
    ax.semilogy(
        kottke_cpw + [cpw_ref],
        kottke_r10 + [r10_ref],
        "o-",
        color=figstyle.PALETTE[0],
        ms=6.0,
        lw=2.0,
        label="Kottke $R_{00}(10^\\circ)$ (Marcuse null)",
    )
    ax.semilogy(
        stair_cpw,
        stair_r10,
        "s--",
        color=figstyle.PALETTE[1],
        ms=6.0,
        lw=2.0,
        label="Staircase $R_{00}(10^\\circ)$ (facet step noise)",
    )
    r10_slab = float(h.slab_overlap_reflectivity(10.0, slab_sm, mode_out=0))
    ax.axhline(
        r10_slab,
        color="k",
        ls=":",
        lw=1.6,
        label=f"1D slab overlap at $10^\\circ$ ({r10_slab:.2e})",
    )
    ax.set_xlabel("Resolution [cells per wavelength in core, cpw]")
    ax.set_ylabel("Modal reflectivity $R_{00}(10^\\circ)$ [-]")
    ax.set_title("(b) Marcuse $10^\\circ$ Interference Null vs. Resolution")
    ax.set_ylim(5e-9, 1e-3)
    ax.legend(loc="best", fontsize=7.2)

    ax = axes[2]
    ax.loglog(
        kottke_h,
        kottke_err,
        "o-",
        color=figstyle.PALETTE[0],
        ms=6.0,
        lw=2.0,
        label=f"Kottke $|R_{{00}}(7^\\circ) - R_{{00}}^{{\\mathrm{{ref}}}}| / R_{{00}}^{{\\mathrm{{ref}}}}$ (cpw={cpw_ref})",
    )
    ax.loglog(
        stair_h,
        stair_err,
        "s--",
        color=figstyle.PALETTE[1],
        ms=6.0,
        lw=2.0,
        label="Staircase $|R_{00}(7^\\circ) - R_{00}^{\\mathrm{ref}}| / R_{00}^{\\mathrm{ref}}$",
    )
    ax.loglog(
        kottke_h + [float(ref_npz["dx_min_nm"])],
        list(neff_err) + [abs(neff_ref - neff_exact) / neff_exact],
        "^-.",
        color=figstyle.PALETTE[4],
        ms=6.0,
        lw=1.8,
        label="$|n_{\\mathrm{eff},0} - n_{\\mathrm{eff},0}^{\\mathrm{exact}}| / n_{\\mathrm{eff},0}^{\\mathrm{exact}}$",
    )
    h_arr = np.asarray(kottke_h, dtype=float)
    ax.loglog(
        h_arr,
        kottke_err[0] * (h_arr / h_arr[0]) ** 2,
        "k:",
        lw=1.5,
        label="$O(\\Delta x_{\\min}^2)$ second-order reference",
    )
    ax.set_xlabel("Minimum cell size $\\Delta x_{\\min}$ [nm]")
    ax.set_ylabel("Relative error [-]")
    ax.set_title("(c) Convergence Rate with $\\Delta x_{\\min}$")
    ax.legend(loc="best", fontsize=7.2)

    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "grid_convergence_kottke_vs_staircase.png", close=True)


def plot_all(data_dir: Path, fig_dir: Path, quick: bool | None = None) -> None:
    """Load all task outputs and generate all 16 tutorial figures."""
    results = load_all_results(data_dir=data_dir, quick=quick)
    plot_geometry_and_eigenmodes(results, fig_dir=fig_dir)
    plot_fields_and_movie(results, fig_dir=fig_dir)
    plot_reflectivity_and_ar(results, fig_dir=fig_dir)
    plot_bimodal_and_farfield(results, fig_dir=fig_dir)
    plot_sled_ripple_and_pml(results, fig_dir=fig_dir)
    plot_convergence_studies(results, fig_dir=fig_dir)
    print(f"Generated 16 figures in {fig_dir}")


# %% [markdown]
# <a id="11-summary-table"></a>
# ## 11. Summary of Quantitative Results
#
# The table below compares the coarse Wakis 2D FIT `--quick` grid (`cpw = 14`, $\Delta x_{\min} = 30.94\;\text{nm}$, $\Delta x_{\text{bg}} = 40.81\;\text{nm}$, $7$ cells across $t_{\text{AR}}$) and the `--full` production study (`cpw = 20`, $\Delta x_{\min} = 21.66\;\text{nm}$, $\Delta x_{\text{bg}} = 28.57\;\text{nm}$, $10$ cells across $t_{\text{AR}}$, plus the `cpw = 40` fine reference with $\Delta x_{\min} = 10.83\;\text{nm}$ and $20$ cells across $t_{\text{AR}}$) against the analytical models of **Marcuse (1989)** and the experimental benchmarks of **Zah et al. (1987)**:
#
# | Quantity | Waveguide / Condition | Coarse Grid (`--quick`, `cpw=14`) | Production Grid (`--full`, `cpw=20` / ref `cpw=40`) | Analytic / Published Reference | Source & Parameter Sensitivity Notes |
# | :--- | :--- | :---: | :---: | :---: | :--- |
# | Fundamental $n_{\text{eff},0}$ | Single-mode ($w_1 = 1.75\;\mu\text{m}$) | $3.3537$ ($0.78\%$ err) | $3.3402$ ($0.38\%$) / `cpw=40`: $3.3306$ ($0.09\%$) | $3.3276$ | Exact 1D slab dispersion ($V_{\text{half}} = 1.534$) |
# | Bimodal $n_{\text{eff},0},\;n_{\text{eff},1}$ | Two-mode ($w_2 = 2.80\;\mu\text{m}$) | $3.3578,\;3.3435$ | $3.3444,\;3.3303$ ($0.38\%$ err) | $3.3318,\;3.3178$ | Exact 1D slab dispersion ($V_{\text{half}} = 2.454$) |
# | Gaussian waist $w_0$ | Single-mode ($w_1 = 1.75\;\mu\text{m}$) | $1.293\;\mu\text{m}$ | $1.293\;\mu\text{m}$ | $1.293\;\mu\text{m}$ | $99.65\%$ power overlap with exact $\text{TE}_0$ |
# | $R_{00}(0^\circ)$ (bare) | Single-mode, normal cleaved facet | $2.749 \times 10^{-1}$ | $2.793 \times 10^{-1}$ | $2.893 \times 10^{-1}$ | Fresnel $((n_{\text{eff}} - 1)/(n_{\text{eff}} + 1))^2$ |
# | $R_{00}(4^\circ)$ (bare) | Single-mode, $\theta_t = 4^\circ$ | $6.046 \times 10^{-2}$ | $6.146 \times 10^{-2}$ | $6.335 \times 10^{-2}$ (slab) / $5.788 \times 10^{-2}$ (Gauss) | Marcuse (1989); FIT within $3.0\%$ of 1D slab |
# | $R_{00}(7^\circ)$ (bare) | Single-mode, $\theta_t = 7^\circ$ | $3.403 \times 10^{-3}$ | $3.546 \times 10^{-3}$ (`cpw=40`: $3.648 \times 10^{-3}$) | $3.473 \times 10^{-3}$ (slab) / $2.027 \times 10^{-3}$ (Gauss) / $2\text{–}3 \times 10^{-3}$ (exp.) | FIT within $2.1\%$ of parameter-free 1D slab; absolute level depends on $(w_1, \Delta n)$ |
# | $R_{00}(10^\circ)$ (bare) | Single-mode, Marcuse interference null | $1.410 \times 10^{-7}$ | $1.386 \times 10^{-8}$ (`cpw=40`: $3.131 \times 10^{-8}$) | $5.815 \times 10^{-6}$ at $10^\circ$ ($9.0 \times 10^{-12}$ null at $9.70^\circ$) | Sidewall destructive interference zero |
# | $R_{00}(12^\circ)$ (bare) | Single-mode, diffraction sidelobe | $3.476 \times 10^{-5}$ | $3.487 \times 10^{-5}$ | $5.749 \times 10^{-5}$ (slab) / $1.011 \times 10^{-7}$ (Gauss) | Step-index slab sidelobe beyond $10^\circ$ null |
# | $R_{00}^{\text{AR}}(0^\circ)$ | Single-mode + $\lambda_0/4$ AR ($n_{\text{AR}}=1.72$) | $3.151 \times 10^{-3}$ | $3.608 \times 10^{-3}$ (`cpw=40`: $3.949 \times 10^{-3}$) | $3.449 \times 10^{-3}$ (1D TMM) | Node-snapped $\lambda_0/4$ film ($10\text{–}20$ cells) |
# | $R_{00}^{\text{AR}}(7^\circ)$ | Single-mode + $\lambda_0/4$ AR, $\theta_t = 7^\circ$ | $3.829 \times 10^{-5}$ | $4.355 \times 10^{-5}$ | $5.083 \times 10^{-5}$ (slab) / $2.966 \times 10^{-5}$ (Gauss) | $81.4\times$ suppression over bare $7^\circ$ facet |
# | Bimodal $R_{01}(2^\circ)$ | Two-mode ($w_2 = 2.80\;\mu\text{m}$), $\text{TE}_0 \to \text{TE}_1$ | $9.154 \times 10^{-2}$ | $9.168 \times 10^{-2}$ | $9.304 \times 10^{-2}$ (slab) / $9.347 \times 10^{-2}$ (Gauss) | Peaks near $u = 1$; FIT within $1.5\%$ of 1D slab |
# | Bimodal $R_{01}(7^\circ)$ vs. $R_{00}(7^\circ)$ | Two-mode ($w_2 = 2.80\;\mu\text{m}$), $\theta_t = 7^\circ$ | $2.125 \times 10^{-3}$ vs. $1.324 \times 10^{-5}$ | $2.268 \times 10^{-3}$ vs. $1.029 \times 10^{-5}$ ($220\times$) | $2.220 \times 10^{-3}$ vs. $1.855 \times 10^{-5}$ (slab) | Odd-mode conversion dominates feedback ($2.2\%$ err) |
# | Far-field peak $\theta_\parallel(7^\circ, \text{TE}_0)$ | Single-mode into air (`Near2FarMonitor`) | $+23.75^\circ$ | $+23.75^\circ$ | $+24.0^\circ$ | Snell's law $\arcsin(3.337\sin 7^\circ)$, Zah Fig. 2(a) |
# | Far-field lobes $\theta_\parallel(7^\circ, \text{TE}_1)$ | Bimodal $\text{TE}_1$ into air (`Near2FarMonitor`) | $+13.0^\circ,\;+35.8^\circ$ | $+12.75^\circ,\;+35.50^\circ$ (null at $+24^\circ$) | Twin lobes around $+24^\circ$ | Reproduces Zah et al. (1987) Fig. 2(b) |
# | SLED ripple $\Delta G_{\text{dB}}$ ($G_s = 20.8\;\text{dB}$) | Bare $7^\circ$ vs. AR-coated $7^\circ$ output | $7.53\;\text{dB} \to 0.75\;\text{dB}$ | $7.89\;\text{dB}$ (bare) $\to 0.82\;\text{dB}$ (AR) | $< 1\;\text{dB}$ with AR + angled facet | Hakki–Paoli formula ($0.091\;\text{dB}$ dual AR) |
# | CPML parasitic floor $R_{\text{floor}}$ | Straight WG into CPML ($n_{\text{pml}} = 6, 10, 14$) | $6.01 \times 10^{-5},\;3.08 \times 10^{-8},\;3.34 \times 10^{-13}$ | $1.91 \times 10^{-5},\;1.86 \times 10^{-12},\;6.25 \times 10^{-14}$ | $< 10^{-8}$ required for $10^\circ$ Marcuse null | Sets dynamic-range floor of mode monitor |
# | Incoherent vs. coherent $\eta_{\text{TE}_1}$ | 9 dipoles across bimodal core ($w_2 = 2.80\;\mu\text{m}$) | $0.4072$ vs. $8.38 \times 10^{-6}$ | $0.4074$ (incoherent) vs. $4.10 \times 10^{-6}$ (coherent) | Coherent sum falsely zeroes odd modes | Demonstrates necessity of power averaging |

# %% [markdown]
# <a id="12-limits-and-failure-modes"></a>
# ## 12. Limits & Failure Modes
#
# 1. **CPML Reflectivity Floor vs. Physical Facet Reflectivity**:
#    Measuring $R_{00}(\theta_t)$ down to $10^{-5}\text{–}10^{-8}$ is a **high-dynamic-range numerical measurement** (60–80 dB below the incident wave). As shown in Figure 11 (`cpw = 20`), a thin $n_{\text{pml}} = 6$ absorber has a parasitic reflection floor of $R_{\text{floor}} = 1.91 \times 10^{-5}$ ($6.01 \times 10^{-5}$ at `cpw = 14`), which is comparable to the AR-coated $7^\circ$ reflectivity ($4.36 \times 10^{-5}$) and three orders of magnitude above the $10^\circ$ Marcuse null ($1.39 \times 10^{-8}$). Using $n_{\text{pml}} = 10$ reduces $R_{\text{floor}}$ to $1.86 \times 10^{-12}$ at `cpw = 20` ($3.08 \times 10^{-8}$ at `cpw = 14`).
# 2. **Double-Cut Subpixel Seam & Grid-Step Artifacts**:
#    If a tilted waveguide is constructed by abutting two separate polygons along the facet line $x = y \tan\theta_t$, Kottke subpixel averaging evaluates each polygon sequentially and blends the boundary cells twice, creating a thin artificial dielectric sheet along the facet that distorts $R_{00}(\theta_t)$ below $10^{-3}$. Similarly, changing the longitudinal grid step $\Delta x$ between the `ModeMonitor` and the facet creates a numerical wave-impedance reflection ($\sim 10^{-5}\text{–}10^{-4}$) that obscures the $10^\circ$ Marcuse null. Overlapping the background cladding, continuous core slab, and half-space air polygon in painter's order on a uniform longitudinal core mesh (`hx = t_AR / n_AR`) ensures each interface is cut cleanly once.
# 3. **Coherent Dipole Summation Fallacy**:
#    Summing spontaneous-emission dipoles coherently in a single FDTD/FIT run forces all emitters to oscillate in phase ($\phi_j = 0$). In any symmetric waveguide, this phased-array artifact suppresses odd-parity modes ($\eta_{\text{TE}_1}^{\text{coh}} = 4.10 \times 10^{-6}$ at `cpw = 20` instead of the physical $40.74\%$) and narrows the radiated angular distribution.
# 4. **2D Effective-Index Approximation vs. Full 3D Etched Ridge**:
#    Our 2D model captures lateral diffraction and mode conversion in the junction plane $(x, y)$ but treats vertical confinement in $z$ through the effective index $n_{\text{core}} = 3.337$. In a full 3D ridge with a large vertical index step ($\Delta n_\perp \sim 0.3$), the tilted facet also couples slightly into vertical radiation modes, and the Fresnel reflection coefficient differs between TE and TM polarizations at oblique angles.

# %% [markdown]
# <a id="13-exercises"></a>
# ## 13. Exercises
#
# 1. **Optimal Quarter-Wave AR Index ($n_{\text{AR}} = \sqrt{n_{\text{eff}}}$)**:
#    Modify `N_AR_COAT` from $1.72$ to the ideal normal-incidence matching value $\sqrt{n_{\text{eff},0}} \approx 1.824$ ($t_{\text{AR}} = \lambda_0 / (4 n_{\text{AR}}) = 204.2\;\text{nm}$) and compute the minimum reflectivity $R_{00}^{\text{AR}}(7^\circ, \lambda)$ across $1.43\text{–}1.55\;\mu\text{m}$. At what CPML thickness $n_{\text{pml}}$ does the Wakis FIT measurement become limited by the absorber floor?
# 2. **Curved Waveguide (J-Shape) vs. Straight Tilted Waveguide**:
#    In monolithic SLEDs where one facet must remain normal ($0^\circ$, high reflectivity) and the output facet is tilted at $7^\circ$, the waveguide bends smoothly from $\theta = 0^\circ$ to $\theta_t = 7^\circ$. Use `wakis.templates.photonics.s_bend` to simulate a bent ridge meeting the facet at $7^\circ$ and measure the residual $\text{TE}_0 \to \text{TE}_1$ bend-induced mode conversion.
# 3. **Two-Layer Broadband AR Coating**:
#    Extend `transfer_matrix_ar_reflectance` in `_helpers.py` to a two-layer $\text{TiO}_2 / \text{SiO}_2$ coating and determine the layer thicknesses that keep $R_{00}(\theta_t = 7^\circ, \lambda) < 10^{-5}$ over a full $100\;\text{nm}$ octave-spanning SLED gain bandwidth.

# %% [markdown]
# <a id="14-references"></a>
# ## 14. References
#
# 1. D. Marcuse, "Reflection loss of improperly joined single-mode fibers," and "Reflection of a Gaussian beam from a tilted waveguide facet," *J. Lightwave Technol.*, vol. 7, no. 2, pp. 336–339, Feb. 1989, doi:[10.1109/50.17776](https://doi.org/10.1109/50.17776).
# 2. C. E. Zah, J. S. Osinski, C. Caneau, S. G. Menocal, L. A. Reith, J. Salzman, F. K. Shokoohi, and T. P. Lee, "1.5 $\mu$m compressive-strained multiquantum-well 20-channel wavelength-division-multiplexed laser array / 1.5 $\mu$m GaInAsP angled-facet flared-waveguide travelling-wave laser amplifiers," *Electron. Lett.*, vol. 23, no. 19, pp. 990–992, Sep. 1987, doi:[10.1049/el:19870695](https://doi.org/10.1049/el:19870695).
# 3. B. W. Hakki and T. L. Paoli, "Gain spectra in GaAs double-heterostructure injection lasers," *J. Appl. Phys.*, vol. 46, no. 3, pp. 1299–1306, Mar. 1975, doi:[10.1063/1.321696](https://doi.org/10.1063/1.321696).
# 4. G. A. Alphonse, D. B. Gilbert, M. G. Harvey, and M. Ettenberg, "High-power superluminescent diodes," *IEEE J. Quantum Electron.*, vol. 24, no. 12, pp. 2454–2457, Dec. 1988, doi:[10.1109/3.14376](https://doi.org/10.1109/3.14376).

# %%
def main(argv: list[str] | None = None) -> int:
    """CLI entry point supporting --list-tasks, --task, --quick, --full, --plot."""
    parser = argparse.ArgumentParser(
        description="Tutorial 7.2 · Superluminescent diode: residual reflectivity of a tilted facet (2D Wakis FIT)."
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the study task table and exit.",
    )
    parser.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single task index from the task table.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR,
        help="Output directory for task_XX.npz and task_XX.json files.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=FIG_DIR,
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Use quick resolution (cpw=14).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full production resolution (cpw=20).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate all figures from existing data in --out.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Accepted for compatibility with tutorial runner.",
    )

    if argv is None:
        env_args = os.environ.get("WAKIS_TUTORIAL_ARGS")
        argv = shlex.split(env_args) if env_args else sys.argv[1:]
    # Strip ipykernel arguments when executed inside a notebook
    clean_argv = [a for a in argv if not a.startswith("-f") and not a.endswith(".json")]
    args, _ = parser.parse_known_args(clean_argv)
    if args.full:
        quick = False
    elif args.quick:
        quick = True
    else:
        t0_json = args.out / "task_00.json"
        if t0_json.exists():
            saved_cpw = int(
                json.loads(t0_json.read_text()).get("task", {}).get("cpw", 14)
            )
            quick = saved_cpw < 20
        else:
            quick = True

    if args.list_tasks:
        tasks = get_task_table(quick=quick)
        print(f"Total tasks ({'quick' if quick else 'full'} mode): {len(tasks)}")
        for t in tasks:
            print(f"  [{t['index']:02d}] {t['name']:<26s} : {t['description']}")
        return 0

    if args.task is not None:
        run_task(args.task, out_dir=args.out, quick=quick)
        return 0

    if args.plot:
        plot_all(data_dir=args.out, fig_dir=args.fig_dir, quick=quick)
        return 0

    # Default when run without --task or --plot: run all tasks if missing, then plot
    tasks = get_task_table(quick=quick)
    for t in tasks:
        npz_p = args.out / f"task_{t['index']:02d}.npz"
        if not npz_p.exists():
            run_task(t["index"], out_dir=args.out, quick=quick)
    plot_all(data_dir=args.out, fig_dir=args.fig_dir, quick=quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
