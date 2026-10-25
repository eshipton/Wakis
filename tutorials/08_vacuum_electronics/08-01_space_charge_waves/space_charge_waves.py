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
# # Tutorial 8.1 · Space-charge waves on a drifting electron beam
#
# **Abstract.** In microwave and millimeter-wave vacuum electronic devices—such
# as klystrons, traveling-wave tubes (TWTs), and backward-wave oscillators
# (BWOs)—amplification and bunching arise from the longitudinal interaction
# between electromagnetic circuit fields and **fast and slow space-charge waves**
# supported by a drifting electron beam. In this tutorial we model a
# magnetically confined, relativistic cold-fluid electron beam in the
# small-signal limit using `wakis.beam_fluid.LinearizedBeam` coupled directly to
# `SolverFIT3D` in both 2D planar sheet-beam and 3D tunnel configurations. We
# benchmark the simulated $\omega$-$\beta$ space-charge wave dispersion,
# plasma-frequency reduction factor $R(\omega, b, a)$, and klystron-like
# drift-space current beat against the analytical theories of **Ramo (1939)**
# [1], **Branch & Mihran (1955)** [2], and **Mealy, Marosi & Capolino (2022)**
# [3], verify **Chu's kinetic power theorem** [4] (demonstrating the negative
# kinetic energy of the slow space-charge wave and coupled Chu–Poynting energy
# conservation), and delineate the small-signal validity boundaries beyond which
# Lagrangian Particle-In-Cell (PIC) modeling is required.
#
# ---
#
# ## Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. Formulate the linearized relativistic cold-fluid equations for a 1D-confined
#    drifting electron beam ($\mathbf{u}_0 = u_0 \hat{\mathbf{s}}$) with
#    longitudinal effective mass $\gamma_0^3 m_e$, and couple them to Wakis's
#    Finite Integration Technique (FIT) leapfrog update via `LinearizedBeam`.
# 2. Extract the **fast** ($\omega_+ = \beta u_0 + \omega_q$) and **slow**
#    ($\omega_- = \beta u_0 - \omega_q$) space-charge wave branches across the
#    Brillouin zone using Bloch boundary conditions, and explain why transverse
#    electric-field fringing to the surrounding tunnel walls reduces the plasma
#    frequency by the geometric reduction factor $R(\beta) \in (0, 1)$.
# 3. Reproduce the universal **Branch & Mihran (1955)** [2] and **Mealy, Marosi
#    & Capolino (2022)** [3] curves of $R$ versus beam filling ratio $b / a$ and
#    modulation frequency $f \in [1, 10]\;\text{GHz}$, and contrast 2D planar
#    sheet-beam fringing with 3D pencil-beam fringing.
# 4. Simulate **ballistic-to-space-charge bunching** downstream of a
#    velocity-modulating RF gap in an open drift tube (with smooth DC density
#    tapering before the CPML absorber), measuring the quarter-plasma-wavelength
#    klystron bunching distance $\lambda_q / 4 = \pi / (2 \beta_q)$.
# 5. Audit **Chu's kinetic power theorem** ($P_k > 0$ for the fast wave,
#    $P_k < 0$ for the slow wave) and identify when trajectory overtaking
#    ($X \ge 1$), harmonic generation, and particle trapping require PIC.
#
# ## Contents
#
# 1. [Physical background & linearized fluid equations](#1.-Physical-background-&-linearized-fluid-equations)
# 2. [Transverse resonance & plasma-frequency reduction factor $R$](#2.-Transverse-resonance-&-plasma-frequency-reduction-factor-R)
# 3. [Chu's kinetic power theorem & negative-energy slow waves](#3.-Chu's-kinetic-power-theorem-&-negative-energy-slow-waves)
# 4. [Staggered Crank–Nicolson FIT discretization](#4.-Staggered-Crank–Nicolson-FIT-discretization)
# 5. [Study tasks, resolution rules & preflight estimates](#5.-Study-tasks,-resolution-rules-&-preflight-estimates)
# 6. [Generating the complete 14-figure suite](#6.-Generating-the-complete-14-figure-suite)
# 7. [Dispersion, reduction factor & klystron bunching results](#7.-Dispersion,-reduction-factor-&-klystron-bunching-results)
# 8. [Mesh resolution convergence study](#8.-Mesh-resolution-convergence-study)
# 9. [Limits & failure modes: when PIC is required](#9.-Limits-&-failure-modes:-when-PIC-is-required)
# 10. [References](#10.-References)

# %% [markdown]
# ## 1. Physical background & linearized fluid equations
#
# ### 1.0 Simulation setup schematic
#
# Figure `00a_setup_geometry.png` and the setup specification table below
# summarize the benchmark 2D Bloch space-charge wave cell (`task_0`–`task_4`:
# $L_x = \lambda_e = 9.59\;\text{mm}$, PEC drift-tube walls at $y = \pm 2.0\;\text{mm}$,
# drifting sheet electron beam of half-thickness $b = 0.5\;\text{mm}$ at
# $u_0 = 0.16\,c$, $V_0 = 6.70\;\text{kV}$, $f_p = 1.35\;\text{GHz}$) together
# with the study variations (`task_5` 3D square PEC tunnel, `task_6` open
# klystron drift tube with RF velocity-modulating gap and CPML + 10-cell
# collector density taper, and `task_7` Chu kinetic-power and coupled energy
# conservation audit). Figure `00b_setup_mesh_and_excitation.png` displays the
# $64 \times 128$ FIT mesh ($16$ cells across $b$, $64$ cells per electronic
# wavelength $\lambda_e$) and the synchronous space-charge wave excitation
# spectrum centered at $f_0 = 5.0\;\text{GHz}$.
#
# ![Simulation setup schematic: geometry, PEC drift-tube walls, Bloch boundaries, sheet electron beam, and monitors](figures/00a_setup_geometry.png)
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 9.59 mm × 4 mm × 150 µm (z ∈ [-75 µm, 75 µm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 64 × 128 × 1 (8192 cells) |
# | Discretization | Δx & resolution | Δ_min = 31.2 µm, 307.0 cells/λ (in n_max = 1.00), 16.0 cells across feature (500 µm) |
# | Boundary | -x face | Bloch (k_x = -655 rad/m) |
# | Boundary | +x face | Bloch (k_x = -655 rad/m) |
# | Boundary | -y face | PEC |
# | Boundary | +y face | PEC |
# | Boundary | -z face | Periodic |
# | Boundary | +z face | Periodic |
# | Material | Background | Vacuum, ε = 1.00 (isotropic) |
# | Material | Region 1: Drifting e⁻ sheet beam (b = 0.5 mm, u₀ = 0.16c, V₀ = 6.70 kV, f_p = 1.35 GHz) | ε_r = 1.00, μ_r = 1.00, model = Custom, x ∈ [0 m, 9.59 mm] (d = 9.59 mm) |
# | Source | 1 · LinearizedBeam | pos: y ∈ [−0.50 mm, +0.50 mm], x ∈ [0 mm, 9.59 mm], pol: J_{1x}, E_x, pulse: f₀ = 5 GHz, Δf = 2.7 GHz |
# | Monitor | 2 · EigenmodeSlice | TM₁₀ eigenmode slice (x = 2.40 mm), pos: x = 2.40 mm, y ∈ [−2.0 mm, +2.0 mm], freqs: 6 freqs in [1.5, 10.0] GHz |
# | Monitor | 3 · Probe | Doppler doublet FFT probe E_x(x_p, 0, t) (x_p = 2.40 mm), pos: x = 2.40 mm, freqs: time-domain |
# | Variation | Tasks 0–2: 2D Bloch dispersion & convergence (§5, §7–8) | 2D Bloch cell L_x = λ_e = 9.59 mm, 2a = 4.0 mm, PEC ±y walls, sheet e⁻ beam b = 0.5 mm (u₀ = 0.16c, V₀ = 6.70 kV, f_p = 1.35 GHz); grids 16×32, 24×48, 64×128 |
# | Variation | Tasks 3–4: Frequency & filling-ratio sweeps (§7) | 2D Bloch cell (24×40); f₀ ∈ {1.5, 3.0, 5.0, 7.0, 8.5, 10.0} GHz and b/a ∈ {0.25, 0.375, 0.50, 0.75, 1.00} |
# | Variation | Task 5: 3D square PEC drift tunnel (§7) | 3D domain 4.0 mm × 4.0 mm × 9.59 mm (40×40×24), PEC ±x, ±y walls, Bloch ±z, pencil beam b = 0.5 mm (f_p = 1.65 GHz) |
# | Variation | Task 6: 2D klystron velocity-modulated drift tube (§7) | Open tube L_z = 115 mm × 4.0 mm (360×32), CPML ±z (N_pml = 6, 1.92 mm) + 10-cell ρ₀→0 taper, PEC ±y walls, RF gap V_m = 100 V at z_g = 3.51 mm, beam b = 0.8 mm |
# | Variation | Task 7: Chu kinetic power & energy audit (§3, §7) | Periodic ±x, PEC ±y (48×48, 2400 steps); fast wave (P_k > 0) vs slow wave (P_k < 0) and coupled Chu–Poynting conservation |
#
# ![Discretization mesh and space-charge wave excitation spectrum](figures/00b_setup_mesh_and_excitation.png)
#
# Consider a continuous electron beam drifting at DC velocity
# $\mathbf{u}_0 = u_0 \hat{\mathbf{s}}$ ($0 < |u_0| < c$, Lorentz factor
# $\gamma_0 = (1 - u_0^2/c^2)^{-1/2}$) along a drift tunnel axis
# $s \in \{x, y, z\}$, focused by a strong axial magnetic field
# $\mathbf{B}_0 = B_0 \hat{\mathbf{s}}$ ($\omega_c = e B_0 / (\gamma_0 m_e) \gg \omega_p$)
# so that transverse velocity perturbations are suppressed ($\mathbf{v}_{1,\perp} = 0$).
# Let $\rho_0(\mathbf{r}_\perp) = -e n_0(\mathbf{r}_\perp) \le 0$ denote the
# unperturbed DC space-charge density inside the beam cross-section ( neutralized
# in the DC state by the focusing equilibrium) and $J_{0,s} = \rho_0 u_0$ the DC
# convection current density.
#
# In the small-signal regime ($|v_1| \ll u_0$, $|\rho_1| \ll |\rho_0|$),
# linearizing the relativistic axial momentum equation
# $(\partial_t + v_s \partial_s)(\gamma m_e v_s) = -e E_s$ and the charge
# continuity equation $\partial_t \rho + \partial_s (\rho v_s) = 0$ yields the
# closed first-order hyperbolic system:
#
# $$\left(\frac{\partial}{\partial t} + u_0 \frac{\partial}{\partial s} + \nu_b\right) v_1
#   = -\frac{e}{\gamma_0^3 m_e} E_s,$$
#
# $$\left(\frac{\partial}{\partial t} + u_0 \frac{\partial}{\partial s}\right) \rho_1
#   = -\rho_0 \frac{\partial v_1}{\partial s},$$
#
# $$J_{1,s} = \rho_0 v_1 + u_0 \rho_1,$$
#
# where $\gamma_0^3 m_e$ is the relativistic longitudinal mass (arising from
# $d(\gamma v_s)/dv_s = \gamma_0^3$), $\nu_b \ge 0$ is an optional collision/damping
# rate, and $J_{1,s}$ is injected into Ampère's law on the primary $E_s$ edges.
# For harmonic perturbations $\propto \exp(i(\omega t - \beta s))$ with $\nu_b = 0$,
# eliminating $v_1$ and $\rho_1$ gives the spatially dispersive axial
# permittivity inside the beam:
#
# $$\frac{\varepsilon_{ss}(\omega, \beta)}{\varepsilon_0}
#   = 1 - \frac{\omega_{p,\parallel}^2}{(\omega - \beta u_0)^2}, \qquad
#   \omega_p = \sqrt{\frac{n_0 e^2}{\varepsilon_0 m_e}} = \sqrt{\frac{-\rho_0 e}{\varepsilon_0 m_e}}, \qquad
#   \omega_{p,\parallel} = \frac{\omega_p}{\gamma_0^{3/2}}.$$
#
# In an unbounded 1D slab ($\nabla_\perp = 0$), $\varepsilon_{ss}(\omega, \beta) = 0$
# yields the two Doppler-shifted plasma oscillations $\omega_\pm = \beta u_0 \pm \omega_{p,\parallel}$.

# %% [markdown]
# ## 2. Transverse resonance & plasma-frequency reduction factor $R$
#
# In a beam of finite transverse radius or half-thickness $b$ enclosed inside a
# grounded PEC drift tube of half-width or radius $a \ge b$, part of the axial
# displacement flux $\varepsilon_0 \mathbf{E}_1$ fringes transversely onto the
# conducting walls rather than terminating on adjacent space-charge bunches
# (**Ramo 1939** [1]; **Branch & Mihran 1955** [2]). This weakens the longitudinal
# restoring force and reduces the space-charge oscillation frequency to
#
# $$\omega_q(\beta) = R(\beta)\,\frac{\omega_p}{\gamma_0^{3/2}}, \qquad
#   \omega_\pm(\beta) = \beta u_0 \pm \omega_q(\beta), \qquad
#   \beta_q = \frac{\omega_q}{u_0}, \qquad 0 < R(\beta) < 1.$$
#
# With transverse slow-wave decay parameter $\tau = \sqrt{\beta^2 - (\omega/c)^2} \approx \beta / \gamma_0$
# in the vacuum clearance ($b < r_\perp < a$) and internal transverse wavenumber
# $T = \tau \sqrt{1/R^2 - 1}$ inside the beam ($r_\perp \le b$), matching $E_s$
# and $E_\perp$ across the beam boundary yields the exact transverse resonance
# equations implemented in `wakis.beam_fluid`:
#
# - **2D Planar Sheet Beam** (`reduction_factor_planar`, Branch & Mihran 1955 [2]):
#   $$T b \tan(T b) = \tau b \coth\!\big[\tau (a - b)\big], \qquad
#     R_{\text{2D}} = \frac{1}{\sqrt{1 + (T / \tau)^2}}.$$
# - **3D Cylindrical Pencil Beam** (`reduction_factor_cylindrical`, Branch & Mihran 1955 [2]; Mealy *et al.* 2022, Eqs. (10)–(11) [3]):
#   $$T R_b \frac{J_1(T R_b)}{J_0(T R_b)}
#     = \tau R_b \frac{I_0(\tau R_t) K_1(\tau R_b) + I_1(\tau R_b) K_0(\tau R_t)}
#                     {I_0(\tau R_t) K_0(\tau R_b) - I_0(\tau R_b) K_0(\tau R_t)}, \qquad
#     R_{\text{3D,cyl}} = \frac{1}{\sqrt{1 + (T / \tau)^2}}.$$
#
# ### Benchmark parameters from Mealy, Marosi & Capolino (2022), Section III [3]
#
# | Parameter | Symbol | Value |
# |---|---|---|
# | DC beam kinetic voltage | $V_0$ | $6.70\;\text{kV}$ |
# | DC drift velocity / Lorentz factor | $u_0,\;\gamma_0,\;\gamma_0^3$ | $0.16\,c = 4.797 \times 10^7\;\text{m/s}$, $1.0131$, $1.0398$ |
# | Nominal modulation frequency | $f_0$ | $5.0\;\text{GHz}$ ($\lambda_e = u_0/f_0 = 9.593\;\text{mm}$, $\beta_0 = 654.95\;\text{rad/m}$) |
# | Drift tunnel half-height / radius | $a = R_t$ | $2.0\;\text{mm}$ |
# | Nominal beam half-thickness / radius | $b = R_b$ | $0.5\;\text{mm}$ ($b/a = 0.25$, $\tau b = 0.3233$) |
# | Analytic reduction factor at $5\;\text{GHz}$ ($b/a = 0.25$) | $R_{\text{2D}},\;R_{\text{3D,rect}},\;R_{\text{3D,cyl}}$ | $0.46622$ (2D sheet), $0.29160$ (3D square), $0.26730$ (3D round; $\approx 0.26 \pm 0.01$ read from Fig. 3a/3b of [3]) |
# | Velocity-modulating RF gap voltage | $V_m$ | $100\;\text{V}$ ($V_m / V_0 = 0.0149 \ll 1$, $d_{\text{grid}} = 0.2\;\text{mm}$ in [3]) |
#
# > **Note on 2D vs. 3D geometry:** Mealy, Marosi & Capolino (2022) [3] (Section III,
# > Fig. 3a and Fig. 3b) simulate a **3D cylindrical pencil beam** ($R_b = 0.5\;\text{mm}$
# > in a circular tube $R_t = 2.0\;\text{mm}$, where $R_{\text{3D,cyl}} = 0.26730$
# > from Eqs. (10)–(11) and $\approx 0.26 \pm 0.01$ read from Fig. 3a/3b). In this
# > tutorial, Tasks `0`–`4` and `6`–`7` adopt the same beam kinematics ($V_0 = 6.70\;\text{kV}$,
# > $u_0 = 0.16\,c$, $f_0 = 5.0\;\text{GHz}$) and transverse dimensions ($b = 0.5\;\text{mm}$,
# > $a = 2.0\;\text{mm}$) in a **2D planar sheet-beam analog** ($R_{\text{2D}} = 0.46622$),
# > while Task `5` models a **3D square beam in a square PEC tunnel** ($R_{\text{3D,rect}} = 0.29160$)
# > and compares all three geometries against the 3D cylindrical Bessel theory.

# %% [markdown]
# ## 3. Chu's kinetic power theorem & negative-energy slow waves
#
# Introducing the small-signal **Chu kinetic voltage** (**Chu 1951**; **Haus &
# Bobroff 1957** [4])
#
# $$V_1 = -\frac{\gamma_0^3 m_e}{e}\,u_0 v_1,$$
#
# the linearized momentum and continuity equations combine with Poynting's
# theorem into the exact small-signal conservation law
#
# $$\frac{\partial}{\partial t}\left(w_{\text{EM}} + w_k\right)
#   + \nabla \cdot \left(\mathbf{E}_1 \times \mathbf{H}_1 + V_1 J_{1,s} \hat{\mathbf{s}}\right)
#   = -\sigma |\mathbf{E}_1|^2 - \nu_b \gamma_0^3 m_e n_0 v_1^2 \le 0,$$
#
# where the cross-section-integrated **Chu kinetic power** for a single
# space-charge wave $\omega_\pm = \beta u_0 \pm \omega_q$ is
#
# $$P_k = \frac{1}{2} \operatorname{Re} \iint V_1 J_{1,s}^*\,dA_\perp
#       = \pm \frac{\omega_\pm}{\omega_q}\,\frac{\gamma_0^3 m_e n_0 u_0}{2} \iint |v_1|^2\,dA_\perp.$$
#
# Because $\rho_0 < 0$ and $V_1 \propto -v_1$:
# - The **fast space-charge wave** ($\omega_+ = \beta u_0 + \omega_q$, $v_p > u_0$)
#   carries **positive kinetic power** ($P_k > 0$): $V_1$ and $J_{1,s}$ are in phase.
# - The **slow space-charge wave** ($\omega_- = \beta u_0 - \omega_q$, $v_p < u_0$)
#   carries **negative kinetic power** ($P_k < 0$): $V_1$ and $J_{1,s}$ are $180^\circ$
#   out of phase, meaning that exciting the slow wave *extracts* net kinetic
#   energy from the drifting beam and transfers it to a synchronous circuit wave
#   ($\Delta P_{\text{EM}} = -\Delta P_k > 0$).

# %% [markdown]
# ## 4. Staggered Crank–Nicolson FIT discretization
#
# `LinearizedBeam` co-locates $(v_1, \rho_1)$ on the active primary $E_s$ edges
# inside the beam mask and staggers them at half-integer time levels
# $t^{n-1/2} \to t^{n+1/2}$ alongside $\mathbf{H}^{n+1/2}$ and $\mathbf{J}^{n+1/2}$.
# Along each 1D beam line of $M$ edges, the convective derivative
# $(\partial_t + u_0 D_s)$ is integrated by the unconditionally unitary,
# non-dissipative **Crank–Nicolson (trapezoidal)** scheme:
#
# $$\left[\left(1 + \frac{\Delta t}{2}\nu_b\right) I + \frac{\Delta t}{2} u_0 D_s\right] v_1^{n+1/2}
#   = \left[\left(1 - \frac{\Delta t}{2}\nu_b\right) I - \frac{\Delta t}{2} u_0 D_s\right] v_1^{n-1/2}
#   - \Delta t\,\frac{e}{\gamma_0^3 m_e} E_s^n,$$
#
# $$\left(I + \frac{\Delta t}{2} u_0 D_s\right) \rho_1^{n+1/2}
#   = \left(I - \frac{\Delta t}{2} u_0 D_s\right) \rho_1^{n-1/2}
#   - \frac{\Delta t}{2} \rho_0 D_s \left(v_1^{n+1/2} + v_1^{n-1/2}\right),$$
#
# $$J_{1,s}^{n+1/2} = \rho_0 v_1^{n+1/2} + u_0 \rho_1^{n+1/2}.$$
#
# Each timestep solves two decoupled tridiagonal systems in $O(M)$ operations
# across all transverse beam lines simultaneously, admits an **exact Hermitian
# transpose** (`BeamTranspose`, verified to $\sim 10^{-16}$ dot-product accuracy),
# and tapers $\rho_0(s) \to 0$ smoothly before any outflow PML so that no active
# convective current enters the absorber.

# %%
import argparse
import json
import os
from pathlib import Path
import resource
import shlex
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _helpers import (  # noqa: E402
    A_TUNNEL_REF,
    B_BEAM_REF,
    F0_REF,
    GAMMA0_REF,
    U0_REF,
    V0_REF,
    VM_REF,
    WP_REF,
    render_all_figures,
    run_2d_bloch_dispersion_point,
    run_3d_bloch_dispersion_point,
    run_chu_energy_and_slow_wave_audit,
    run_modulated_drift_tube,
)

DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

# %% [markdown]
# ## 5. Study tasks, resolution rules & preflight estimates
#
# Every task in both `--quick` and `--full` modes enforces the project resolution
# rules:
# - **Wavelength sampling**: $\ge 16$ cells per electronic wavelength
#   $\lambda_e = u_0 / f_{\max}$ in `--quick` ($\ge 16\text{--}64$ in `--full`),
#   which corresponds to $\ge 100\text{--}400$ cells per free-space electromagnetic
#   wavelength $\lambda_0 = c / f_{\max}$.
# - **Feature sampling**: $\ge 4$ cells across the smallest geometric feature
#   (`cells_across_feature >= 4.0` across the beam half-thickness $b$ or the
#   velocity-modulating RF gap $d_{\text{gap}} = 4\Delta x$).
# - **Skin-depth rule**: All tunnel walls are ideal PEC boundary conditions
#   (no penetrable lossy metal interiors), so the finite-conductivity skin-depth
#   constraint is automatically satisfied with zero violations.
# - **Task JSON audit**: Every `data/task_<I>.json` records `dx_background`,
#   `dx_min`, `cells_across_feature`, `cpw`, `n_cells`, and `n_steps`.
#
# ### Recorded `--full` study metrics (8 vCPU Cloud Batch VM, `data/task_*.json`)
#
# | Task ID | Name | Grid $(N_x, N_y, N_z)$ | `n_cells` | `n_steps` | `cpw` ($\lambda_e/\Delta s$) | `cells_across_feature` | Peak RSS | `--full` wall time |
# |---|---|---|---|---|---|---|---|---|
# | `0` | `disp_res_coarse` | $(16, 32, 1)$ | $512$ | $9\,992$ | $16.0$ | $4.0$ | $221.3\;\text{MB}$ | $0.89\;\text{s}$ |
# | `1` | `disp_res_medium` | $(24, 48, 1)$ | $1\,152$ | $14\,988$ | $24.0$ | $6.0$ | $224.8\;\text{MB}$ | $2.01\;\text{s}$ |
# | `2` | `disp_res_fine` | $(64, 128, 1)$ | $8\,192$ | $56\,171$ | $64.0$ | $16.0$ | $265.4\;\text{MB}$ | $35.84\;\text{s}$ |
# | `3` | `kw_and_freq_sweep` | $6 \times (24, 40, 1)$ | $5\,760$ | $123\,381$ | $24.0$ | $5.0$ | $265.3\;\text{MB}$ | $14.52\;\text{s}$ |
# | `4` | `thickness_sweep` | $5 \times (24, 40, 1)$ | $4\,800$ | $68\,301$ | $24.0$ | $5.0$ | $240.7\;\text{MB}$ | $9.56\;\text{s}$ |
# | `5` | `beam_3d_tunnel` | $(40, 40, 24)$ | $38\,400$ | $25\,736$ | $24.0$ | $5.0$ | $284.3\;\text{MB}$ | $111.71\;\text{s}$ |
# | `6` | `drift_tube_bunching` | $(360, 32, 1)$ | $11\,520$ | $21\,251$ | $30.031$ | $4.0$ ($4.0/d_{\text{gap}}$, $6.4/b$) | $229.8\;\text{MB}$ | $23.41\;\text{s}$ |
# | `7` | `chu_energy_and_slow_wave` | $(48, 48, 1)$ | $2\,304$ | $2\,400$ | $48.0$ | $6.0$ | $216.6\;\text{MB}$ | $0.60\;\text{s}$ |


# %%
def get_task_table(quick=False):
    """
    Return the 8-task study table for Tutorial 8.1 in ``--quick`` or ``--full``
    configuration. Every task satisfies ``cpw >= 16`` (cells per electronic
    wavelength ``lambda_e = u0 / f``) and ``cells_across_feature >= 4.0``.
    """
    fine_nx, fine_ny = (32, 64) if quick else (64, 128)
    fine_beats = 1.85 if quick else 2.6

    sweep_nx, sweep_ny = (20, 32) if quick else (24, 40)
    sweep_beats = 1.75 if quick else 2.4

    b3d_b = 0.8e-3 if quick else 0.5e-3
    b3d_nxy, b3d_nz = (20, 16) if quick else (40, 24)

    bunch_nx, bunch_ny = (240, 20) if quick else (360, 32)
    chu_nx, chu_ny, chu_nt = (32, 32, 1600) if quick else (48, 48, 2400)

    return [
        {
            "index": 0,
            "name": "disp_res_coarse",
            "kind": "bloch_2d",
            "nx": 16,
            "ny": 32,
            "b_half": B_BEAM_REF,
            "a_half": A_TUNNEL_REF,
            "f0": F0_REF,
            "n_beat_periods": 1.85,
            "record_profiles": True,
            "description": "2D sheet-beam Bloch dispersion (coarse: 16x32, 4 cells/b, 16 cells/lambda_e)",
        },
        {
            "index": 1,
            "name": "disp_res_medium",
            "kind": "bloch_2d",
            "nx": 24,
            "ny": 48,
            "b_half": B_BEAM_REF,
            "a_half": A_TUNNEL_REF,
            "f0": F0_REF,
            "n_beat_periods": 1.85,
            "record_profiles": True,
            "description": "2D sheet-beam Bloch dispersion (medium: 24x48, 6 cells/b, 24 cells/lambda_e)",
        },
        {
            "index": 2,
            "name": "disp_res_fine",
            "kind": "bloch_2d",
            "nx": fine_nx,
            "ny": fine_ny,
            "b_half": B_BEAM_REF,
            "a_half": A_TUNNEL_REF,
            "f0": F0_REF,
            "n_beat_periods": fine_beats,
            "record_profiles": True,
            "description": (
                f"2D sheet-beam Bloch dispersion & TM mode profiles "
                f"(fine: {fine_nx}x{fine_ny}, {fine_ny // 8} cells/b, {fine_nx} cells/lambda_e)"
            ),
        },
        {
            "index": 3,
            "name": "kw_and_freq_sweep",
            "kind": "freq_sweep_2d",
            "nx": sweep_nx,
            "ny": sweep_ny,
            "b_half": B_BEAM_REF,
            "a_half": A_TUNNEL_REF,
            "freqs_ghz": [1.5, 3.0, 5.0, 7.0, 8.5, 10.0],
            "n_beat_periods": sweep_beats,
            "description": "2D k-omega diagram & frequency sweep R(f) across 1.5-10 GHz",
        },
        {
            "index": 4,
            "name": "thickness_sweep",
            "kind": "thickness_sweep_2d",
            "nx": sweep_nx,
            "ny": sweep_ny,
            "fill_ratios": [0.25, 0.375, 0.50, 0.75, 1.00],
            "a_half": A_TUNNEL_REF,
            "f0": F0_REF,
            "n_beat_periods": sweep_beats,
            "description": "2D sheet-beam reduction factor R vs filling ratio b/a in [0.25, 1.0]",
        },
        {
            "index": 5,
            "name": "beam_3d_tunnel",
            "kind": "bloch_3d",
            "nx": b3d_nxy,
            "ny": b3d_nxy,
            "nz": b3d_nz,
            "b_half": b3d_b,
            "a_half": A_TUNNEL_REF,
            "f0": F0_REF,
            "n_beat_periods": 1.65 if quick else 2.2,
            "description": f"3D beam in square PEC tunnel ({b3d_nxy}x{b3d_nxy}x{b3d_nz})",
        },
        {
            "index": 6,
            "name": "drift_tube_bunching",
            "kind": "bunching_2d",
            "nx": bunch_nx,
            "ny": bunch_ny,
            "b_half": 0.8e-3,
            "a_half": A_TUNNEL_REF,
            "L_tube": 0.115,
            "f_mod": F0_REF,
            "record_movie": True,
            "n_frames": 26 if quick else 36,
            "description": f"2D velocity-modulated drift tube klystron bunching & movie ({bunch_nx}x{bunch_ny})",
        },
        {
            "index": 7,
            "name": "chu_energy_and_slow_wave",
            "kind": "chu_audit",
            "nx": chu_nx,
            "ny": chu_ny,
            "nt": chu_nt,
            "b_half": B_BEAM_REF,
            "a_half": A_TUNNEL_REF,
            "f0": F0_REF,
            "description": f"Chu kinetic power sign (fast vs slow wave) & energy conservation ({chu_nx}x{chu_ny})",
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

    if kind == "bloch_2d":
        res = run_2d_bloch_dispersion_point(
            b_half=task_cfg["b_half"],
            a_half=task_cfg["a_half"],
            f0=task_cfg["f0"],
            nx=task_cfg["nx"],
            ny=task_cfg["ny"],
            n_beat_periods=task_cfg["n_beat_periods"],
            record_profiles=task_cfg.get("record_profiles", False),
        )
        grid_shape = [int(task_cfg["nx"]), int(task_cfg["ny"]), 1]
    elif kind == "freq_sweep_2d":
        freqs = np.asarray(task_cfg["freqs_ghz"], dtype=np.float64) * 1e9
        betas = 2.0 * np.pi * freqs / U0_REF
        w_fast, w_slow, w_q, r_fit, r_th = [], [], [], [], []
        dx_bg_max, dx_min_val, cells_across_min, cpw_min = 0.0, 1e9, 1e9, 1e9
        n_cells_tot, n_steps_tot = 0, 0
        for f_val in freqs:
            pt = run_2d_bloch_dispersion_point(
                b_half=task_cfg["b_half"],
                a_half=task_cfg["a_half"],
                f0=f_val,
                nx=task_cfg["nx"],
                ny=task_cfg["ny"],
                n_beat_periods=task_cfg["n_beat_periods"],
                record_profiles=False,
            )
            w_fast.append(pt["omega_fast"])
            w_slow.append(pt["omega_slow"])
            w_q.append(pt["omega_q"])
            r_fit.append(pt["R_fit"])
            r_th.append(pt["R_th"])
            dx_bg_max = max(dx_bg_max, pt["dx_background"])
            dx_min_val = min(dx_min_val, pt["dx_min"])
            cells_across_min = min(cells_across_min, pt["cells_across_feature"])
            cpw_min = min(cpw_min, pt["cpw"])
            n_cells_tot += pt["n_cells"]
            n_steps_tot += pt["n_steps"]
        res = {
            "freqs": freqs,
            "betas": betas,
            "omega_fast": np.asarray(w_fast, dtype=np.float64),
            "omega_slow": np.asarray(w_slow, dtype=np.float64),
            "omega_q": np.asarray(w_q, dtype=np.float64),
            "R_fit": np.asarray(r_fit, dtype=np.float64),
            "R_th": np.asarray(r_th, dtype=np.float64),
            "dx_background": dx_bg_max,
            "dx_min": dx_min_val,
            "min_feature": float(task_cfg["b_half"]),
            "cells_across_feature": cells_across_min,
            "cpw": cpw_min,
            "n_cells": n_cells_tot,
            "n_steps": n_steps_tot,
        }
        grid_shape = [int(task_cfg["nx"]), int(task_cfg["ny"]), 1]
    elif kind == "thickness_sweep_2d":
        fills = np.asarray(task_cfg["fill_ratios"], dtype=np.float64)
        a_h = float(task_cfg["a_half"])
        r_fit, r_th, w_q = [], [], []
        dx_bg_max, dx_min_val, cells_across_min, cpw_min = 0.0, 1e9, 1e9, 1e9
        n_cells_tot, n_steps_tot = 0, 0
        for fr in fills:
            pt = run_2d_bloch_dispersion_point(
                b_half=fr * a_h,
                a_half=a_h,
                f0=task_cfg["f0"],
                nx=task_cfg["nx"],
                ny=task_cfg["ny"],
                n_beat_periods=task_cfg["n_beat_periods"],
                record_profiles=False,
            )
            r_fit.append(pt["R_fit"])
            r_th.append(pt["R_th"])
            w_q.append(pt["omega_q"])
            dx_bg_max = max(dx_bg_max, pt["dx_background"])
            dx_min_val = min(dx_min_val, pt["dx_min"])
            cells_across_min = min(cells_across_min, pt["cells_across_feature"])
            cpw_min = min(cpw_min, pt["cpw"])
            n_cells_tot += pt["n_cells"]
            n_steps_tot += pt["n_steps"]
        res = {
            "fill_ratios": fills,
            "b_values": fills * a_h,
            "R_fit": np.asarray(r_fit, dtype=np.float64),
            "R_th": np.asarray(r_th, dtype=np.float64),
            "omega_q": np.asarray(w_q, dtype=np.float64),
            "dx_background": dx_bg_max,
            "dx_min": dx_min_val,
            "min_feature": float(fills[0] * a_h),
            "cells_across_feature": cells_across_min,
            "cpw": cpw_min,
            "n_cells": n_cells_tot,
            "n_steps": n_steps_tot,
        }
        grid_shape = [int(task_cfg["nx"]), int(task_cfg["ny"]), 1]
    elif kind == "bloch_3d":
        res = run_3d_bloch_dispersion_point(
            b_half=task_cfg["b_half"],
            a_half=task_cfg["a_half"],
            f0=task_cfg["f0"],
            nx=task_cfg["nx"],
            ny=task_cfg["ny"],
            nz=task_cfg["nz"],
            n_beat_periods=task_cfg["n_beat_periods"],
        )
        grid_shape = [int(task_cfg["nx"]), int(task_cfg["ny"]), int(task_cfg["nz"])]
    elif kind == "bunching_2d":
        res = run_modulated_drift_tube(
            b_half=task_cfg["b_half"],
            a_half=task_cfg["a_half"],
            L_tube=task_cfg["L_tube"],
            f_mod=task_cfg["f_mod"],
            nx=task_cfg["nx"],
            ny=task_cfg["ny"],
            record_movie=task_cfg.get("record_movie", True),
            n_frames=task_cfg.get("n_frames", 28),
        )
        grid_shape = [int(task_cfg["nx"]), int(task_cfg["ny"]), 1]
    elif kind == "chu_audit":
        res = run_chu_energy_and_slow_wave_audit(
            b_half=task_cfg["b_half"],
            a_half=task_cfg["a_half"],
            f0=task_cfg["f0"],
            nx=task_cfg["nx"],
            ny=task_cfg["ny"],
            nt=task_cfg["nt"],
        )
        grid_shape = [int(task_cfg["nx"]), int(task_cfg["ny"]), 1]
    else:
        raise ValueError(f"Unknown task kind: {kind}")

    payload = dict(res)
    payload["name"] = np.array(name)
    payload["task_index"] = np.int64(idx)
    npz_path = out_dir / f"task_{idx}.npz"
    np.savez_compressed(npz_path, **payload)

    elapsed = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    meta = {
        "task_index": idx,
        "name": name,
        "kind": kind,
        "grid_shape": grid_shape,
        "dx_background": float(res["dx_background"]),
        "dx_min": float(res["dx_min"]),
        "dx_background_mm": round(float(res["dx_background"]) * 1e3, 4),
        "dx_min_mm": round(float(res["dx_min"]) * 1e3, 4),
        "min_feature_mm": round(float(res["min_feature"]) * 1e3, 4),
        "cells_across_feature": round(float(res["cells_across_feature"]), 3),
        "cpw": round(float(res["cpw"]), 3),
        "n_cells": int(res["n_cells"]),
        "n_steps": int(res["n_steps"]),
        "elapsed_s": round(elapsed, 3),
        "peak_rss_mb": round(rss_mb, 2),
        "npz_bytes": int(npz_path.stat().st_size),
    }
    if "R_fit" in res and np.ndim(res["R_fit"]) == 0:
        meta["R_fit"] = round(float(res["R_fit"]), 5)
        meta["R_th"] = round(float(res["R_th"]), 5)
    elif "R_3d_fit" in res:
        meta["R_3d_fit"] = round(float(res["R_3d_fit"]), 5)
        meta["R_3d_rect_th"] = round(float(res["R_3d_rect_th"]), 5)
    elif "R_beat_fit" in res:
        meta["R_beat_fit"] = round(float(res["R_beat_fit"]), 5)
        meta["L_beat_fit_mm"] = round(float(res["L_beat_fit"]) * 1e3, 3)
        meta["L_beat_th_mm"] = round(float(res["L_beat_th"]) * 1e3, 3)

    (out_dir / f"task_{idx}.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[task_{idx} {name}] grid={grid_shape} dx_bg={meta['dx_background_mm']:.3f}mm "
        f"dx_min={meta['dx_min_mm']:.3f}mm cells_across={meta['cells_across_feature']:.1f} "
        f"cpw={meta['cpw']:.1f} n_cells={meta['n_cells']} n_steps={meta['n_steps']} "
        f"time={elapsed:.2f}s rss={rss_mb:.1f}MB"
    )
    return meta


# %% [markdown]
# ## 6. Generating the complete 14-figure suite
#
# Running `--plot` (or `--plot --full`, or the end of `--quick` / `--full`)
# loads `data/task_*.npz` and generates 14 figures into `figures/`:
#
# 1. `01_drift_tube_geometry.png` — Schematic layout of the velocity-modulated drift tube and 2D/3D cross-sections (Mealy *et al.* 2022 [3]).
# 2. `02_mesh_and_profile_zoom.png` — Subcell transverse beam weight $w_\perp(y)$ across $|y|=b$ and smooth axial density ramp $w_s(z) \to 0$ before the outflow CPML.
# 3. `03_transverse_tm_eigenmodes.png` — Transverse space-charge wave profiles $E_z(y)$, $E_y(y)$, and $J_{1z}(y)$ vs. Branch & Mihran (1955) [2] theory.
# 4. `04_kw_dispersion_diagram.png` — Space-charge wave $(\beta, \omega)$ dispersion diagram and co-moving Doppler shift $\pm \omega_q(\beta)$.
# 5. `05_doppler_splitting_spectra.png` — Time-domain beating waveform $\operatorname{Re}[E_z(t)]$ and Doppler-split FFT spectrum at $5.0\;\text{GHz}$.
# 6. `06_reduction_factor_vs_thickness.png` — 2D sheet-beam reduction factor $R$ vs. filling ratio $b/a$, universal planar Branch & Mihran (1955) [2] curves, and comparison against 3D square-tunnel and 3D cylindrical (Mealy *et al.* 2022 [3], Fig. 3b) theories.
# 7. `07_reduction_factor_vs_frequency.png` — 2D sheet-beam reduction factor $R(f)$ and reduced plasma wavenumber $\beta_q(f)$ across $f \in [1, 10]\;\text{GHz}$ vs. planar Branch & Mihran (1955) [2] theory, alongside the 3D cylindrical Bessel curve of Mealy *et al.* (2022, Fig. 3a) [3].
# 8. `08_2d_vs_3d_fringing_comparison.png` — 3D square-tunnel $E_z(x,y)$ and transverse $\mathbf{E}_\perp$ quiver map vs. 1D/2D/3D reduction factors.
# 9. `09_modulated_beam_field_map.png` — 2D $(z, y)$ snapshots of $\rho_1$, $J_{1z}$, and $E_z$ in the velocity-modulated drift tube.
# 10. `10_klystron_bunching_beat.png` — Standing-wave envelopes $|I_1(z)|$ and $|V_1(z)|$ vs. ballistic theory (Ramo 1939 [1]; Mealy *et al.* 2022, Eqs. (1), (5) & Fig. 1b [3]), showing the $\lambda_q/4$ klystron bunching peak and $\pi/2$ phase quadrature.
# 11. `11_space_charge_wave_propagation.gif` — Animated time-domain movie of velocity modulation converting into space-charge bunches along the drift tube.
# 12. `12_chu_kinetic_power_and_energy.png` — Chu kinetic power $P_k(z)$ sign ($P_k > 0$ fast wave vs. $P_k < 0$ slow wave) and coupled Chu–Poynting energy conservation $U_{\text{EM}}(t) + U_{\text{kin}}(t) = U_{\text{tot}}$.
# 13. `13_resolution_convergence.png` — Mesh-refinement convergence study of $|R_{\text{FIT}} - R_\infty| / R_\infty$ and $|R_{\text{FIT}} - R_{\text{BM}}| / R_{\text{BM}}$ across coarse, medium, and fine grids.
# 14. `14_linear_vs_nonlinear_bunching_limits.png` — Linear cold-fluid vs. nonlinear Bessel harmonic currents $I_n(X) = 2 I_0 J_n(nX)$ and phase-space $(z, v_z)$ overtaking at $X \ge 1$.


# %%
def generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR):
    """Load ``data_dir/task_*.npz`` and render all 14 figures in ``fig_dir``."""
    data_dir = Path(data_dir)
    tasks_data = {}
    for p in sorted(data_dir.glob("task_*.npz")):
        d = dict(np.load(p, allow_pickle=False))
        tasks_data[str(d["name"])] = d
    required = {
        "disp_res_coarse",
        "disp_res_medium",
        "disp_res_fine",
        "kw_and_freq_sweep",
        "thickness_sweep",
        "beam_3d_tunnel",
        "drift_tube_bunching",
        "chu_energy_and_slow_wave",
    }
    missing = sorted(required - set(tasks_data))
    if missing:
        raise FileNotFoundError(
            f"Missing task outputs {missing} in {data_dir}. Run --quick or --full first."
        )
    render_all_figures(tasks_data, fig_dir)


# %% [markdown]
# ## 7. Dispersion, reduction factor & klystron bunching results
#
# ### Quick (`--quick`) vs. Full (`--full`) vs. Analytic/Published Reference
#
# | Study / Metric | Task | Quick (`--quick`) | Full (`--full`) | Analytic / Published Reference | `--full` Rel. Error |
# |---|---|---|---|---|---|
# | 2D Bloch $R$ ($5\;\text{GHz}$, $b/a=0.25$, coarse $16\times 32$) | `task_0` | $0.46576$ | $0.46576$ | $0.46622$ (2D Branch–Mihran [2]) | $-0.098\%$ |
# | 2D Bloch $R$ ($5\;\text{GHz}$, $b/a=0.25$, medium $24\times 48$) | `task_1` | $0.46709$ | $0.46709$ | $0.46622$ (2D Branch–Mihran [2]) | $+0.186\%$ |
# | 2D Bloch $R$ ($5\;\text{GHz}$, $b/a=0.25$, fine $32\times 64$ / $64\times 128$) | `task_2` | $0.46755$ ($32\times 64$) | $0.46660$ ($64\times 128$) | $0.46622$ (2D Branch–Mihran [2]) | $+0.082\%$ |
# | 2D $R(f)$ at $f = 1.5\;\text{GHz}$ ($\tau b = 0.097$) | `task_3` | $0.17163$ ($20\times 32$) | $0.17226$ ($24\times 40$) | $0.17243$ (2D [2]; 3D cyl.: $0.0680$, $\approx 0.07 \pm 0.01$ read from Fig. 3a [3]) | $-0.100\%$ |
# | 2D $R(f)$ at $f = 5.0\;\text{GHz}$ ($\tau b = 0.323$) | `task_3` | $0.46456$ ($20\times 32$) | $0.46615$ ($24\times 40$) | $0.46622$ (2D [2]; 3D cyl.: $0.2673$, $\approx 0.26 \pm 0.01$ read from Fig. 3a [3]) | $-0.016\%$ |
# | 2D $R(f)$ at $f = 10.0\;\text{GHz}$ ($\tau b = 0.647$) | `task_3` | $0.65645$ ($20\times 32$) | $0.65861$ ($24\times 40$) | $0.65835$ (2D [2]; 3D cyl.: $0.4789$, $\approx 0.48 \pm 0.01$ read from Fig. 3a [3]) | $+0.039\%$ |
# | 2D $R(b/a)$ across $b/a \in [0.25, 1.00]$ ($5\;\text{GHz}$) | `task_4` | $0.46456 \to 0.63405$ | $0.46615 \to 0.63592$ | $0.46622 \to 0.63554$ (2D [2]; 3D cyl.: $0.2673 \to 0.4744$, $\approx 0.26\text{--}0.47 \pm 0.01$ read from Fig. 3b [3]) | $\le 0.11\%$ |
# | 3D square-tunnel $R_{\text{3D}}$ ($5\;\text{GHz}$) | `task_5` | $0.38128$ ($b/a=0.40$, $20^2\times 16$) | $0.29030$ ($b/a=0.25$, $40^2\times 24$) | $0.38553$ ($b/a=0.40$) / $0.29160$ ($b/a=0.25$, 3D Galerkin) | $-0.443\%$ |
# | 2D drift-tube beat half-wavelength $L_{\text{beat}}$ ($b/a=0.40$) | `task_6` | $39.845\;\text{mm}$ ($240\times 20$) | $41.218\;\text{mm}$ ($360\times 32$) | $42.491\;\text{mm}$ (continuum ideal gap) / $41.357\;\text{mm}$ (discrete FD grid) | $-3.00\%$ ($-0.33\%$ vs. FD) |
# | 2D drift-tube effective $R_{\text{beat}}$ ($b/a=0.40$) | `task_6` | $0.58451$ ($240\times 20$) | $0.56505$ ($360\times 32$) | $0.54811$ (continuum planar [2]) | $+3.09\%$ |
# | Coupled Chu–Poynting energy drift $\Delta U_{\text{tot}} / \langle U_{\text{tot}} \rangle$ | `task_7` | $9.48 \times 10^{-6}$ ($32\times 32$, $1600$ steps) | $4.21 \times 10^{-6}$ ($48\times 48$, $2400$ steps) | $0$ (exact continuum conservation [4]; $O(\Delta t^2)$ discrete) | $4.21 \times 10^{-6}$ |
#
# - **Space-charge wave dispersion (`04_kw_dispersion_diagram.png`, `05_doppler_splitting_spectra.png`)**:
#   At every wavenumber $\beta$, the 2D Bloch FIT solver resolves the symmetric
#   Doppler doublet $\omega_\pm(\beta) = \beta u_0 \pm \omega_q(\beta)$ straddling
#   the synchronous beam line $\omega = \beta u_0$. At the nominal $f_0 = 5.0\;\text{GHz}$
#   benchmark point ($b = 0.5\;\text{mm}$, $a = 2.0\;\text{mm}$, $u_0 = 0.16\,c$),
#   the analytical Branch–Mihran planar reduction factor is $R_{\text{2D,BM}} = 0.46622$
#   ($f_+ = 5.6173\;\text{GHz}$, $f_- = 4.3827\;\text{GHz}$, $f_q = 617.27\;\text{MHz}$),
#   and the `--full` fine Wakis FIT mesh ($64 \times 128$, $16$ cells/$b$, $64\;\text{cpw}$)
#   reproduces $R_{\text{2D,FIT}} = 0.46660$ ($+0.082\%$ error; vs. $0.46755$, $+0.285\%$
#   in `--quick` on $32 \times 64$).
# - **Thickness & frequency sweeps (`06_reduction_factor_vs_thickness.png`, `07_reduction_factor_vs_frequency.png`)**:
#   As the beam filling ratio increases from $b/a = 0.25$ to $b/a = 1.0$ (`task_4`,
#   $24 \times 40$ in `--full`), the 2D reduction factor rises from $R_{\text{FIT}} = 0.46615$
#   (analytic $0.46622$) to $R_{\text{FIT}} = 0.63592$ (analytic $0.63554$, max error
#   $0.11\%$). Sweeping the modulation frequency across $f \in \{1.5, 3.0, 5.0, 7.0, 8.5, 10.0\}\;\text{GHz}$
#   (`task_3`, $\tau b \in [0.097, 0.647]$) increases the 2D planar $R_{\text{FIT}}(f)$
#   from $0.17226$ (analytic $0.17243$) to $0.65861$ (analytic $0.65835$, max error
#   $0.10\%$) because shorter axial wavelengths confine the space-charge electric
#   field more tightly within the beam core (compared with the unbounded $a \to \infty$
#   limit $0.289 \to 0.694$). In both figures,
#   the 2D sheet-beam results are contrasted with the 3D cylindrical pencil-beam
#   curves of Mealy *et al.* (2022) [3] ($R_{\text{3D,cyl}} = 0.0680 \to 0.4789$
#   across $1.5\text{--}10\;\text{GHz}$, $\approx 0.07 \to 0.48 \pm 0.01$ read from
#   Fig. 3a; and $R_{\text{3D,cyl}} = 0.2673 \to 0.4744$ across $R_b/R_t \in [0.25, 1.0]$,
#   $\approx 0.26 \to 0.47 \pm 0.01$ read from Fig. 3b).
# - **2D sheet vs. 3D tunnel fringing (`08_2d_vs_3d_fringing_comparison.png`)**:
#   In 3D, electric flux fringes in both transverse directions $(x, y)$ toward
#   the four tunnel walls, reducing $R$ more strongly than in a 2D slab. In `--full`
#   (`task_5`, $40 \times 40 \times 24$, $5.0$ cells/$b$, $24\;\text{cpw}$), the 3D
#   simulation runs at the nominal benchmark filling ratio $b/a = 0.25$
#   ($b = 0.5\;\text{mm}$, $a = 2.0\;\text{mm}$, $f_0 = 5.0\;\text{GHz}$) and yields
#   $R_{\text{3D,FIT}} = 0.29030$ against the 3D rectangular Galerkin theory
#   $R_{\text{3D,rect}} = 0.29160$ ($0.44\%$ error; cf. $R_{\text{2D}} = 0.46622$
#   for a 2D slab and $R_{\text{3D,cyl}} = 0.26730$ for a circular tunnel). In
#   `--quick`, `task_5` uses a wider beam $b/a = 0.40$ ($b = 0.8\;\text{mm}$) on a
#   $20 \times 20 \times 16$ mesh so that $b/\Delta x = 4.0$ still satisfies the
#   $\ge 4$-cell feature rule, yielding $R_{\text{3D,FIT}} = 0.38128$ vs.
#   $R_{\text{3D,rect}} = 0.38553$ ($1.10\%$ error).
# - **Klystron drift-space bunching & explanation of the $3.00\%$ beat-length gap (`09_modulated_beam_field_map.png`, `10_klystron_bunching_beat.png`, `11_space_charge_wave_propagation.gif`)**:
#   Downstream of the velocity-modulating RF gap (`task_6`, $b = 0.8\;\text{mm}$,
#   $a = 2.0\;\text{mm}$, $f_{\text{mod}} = 5.0\;\text{GHz}$, $f_p = 1.05\;\text{GHz}$;
#   in `--full`, $360 \times 32$, $\Delta x = 0.3194\;\text{mm}$, $d_{\text{gap}} = 4\Delta x = 1.278\;\text{mm}$,
#   $\theta_g = 0.837\;\text{rad}$, $M_{\text{gap}} = 0.971$), the velocity modulation
#   beats between the fast and slow space-charge waves as $V_1(z) \propto \cos(\beta_q z)$
#   and $I_1(z) \propto -i \sin(\beta_q z)$ (Ramo 1939 [1]; Mealy *et al.* 2022,
#   Eqs. (1), (5) & Fig. 1b [3]). In `--full`, the measured spatial beat
#   half-wavelength is $L_{\text{beat,FIT}} = 41.218\;\text{mm}$ (`R_beat_fit = 0.56505`,
#   first current peak at $z = 20.92\;\text{mm}$, first current node at $z = 41.69\;\text{mm}$)
#   versus the continuum analytical value $L_{\text{beat,th}} = 42.491\;\text{mm}$
#   (`R_th = 0.54811`)—a **$-3.00\%$ gap** (halved from the $-6.23\%$ gap,
#   $L_{\text{beat,FIT}} = 39.845\;\text{mm}$, in `--quick` at $240 \times 20$).
#   This $-3.00\%$ shift arises from three quantitative effects in the spatial
#   boundary-value problem at fixed driving frequency $\omega_{\text{mod}}$:
#   1. *Centered-difference spatial group-velocity dispersion ($-2.21\%$ in $L_{\text{beat}}$)*:
#      Unlike the temporal Bloch runs (`task_0`–`task_2`) which fix $\beta = \beta_0$
#      and measure $\omega_\pm(\beta_0)$, the drift tube drives the beam at fixed
#      $\omega_{\text{mod}}$ and measures the spatial wavenumber splitting
#      $\beta_{\text{slow}}(\omega_{\text{mod}}) - \beta_{\text{fast}}(\omega_{\text{mod}}) = 2\beta_q$.
#      Because `LinearizedBeam` discretizes $\partial_x$ with second-order centered
#      differences $\tilde{\beta}(\beta) = \sin(\beta \Delta x)/\Delta x$, inverting
#      $\tilde{\beta}_\pm = \beta_0 \pm \beta_q$ around the synchronous carrier
#      $\sin(\beta_{\text{sync}} \Delta x) = \beta_0 \Delta x$ ($\beta_0 \Delta x = 0.2092$
#      at $30.0\;\text{cpw}$ in `--full`; $0.3138$ at $20.0\;\text{cpw}$ in `--quick`)
#      multiplies $\beta_q$ by $d\beta / d\tilde{\beta} = 1 / \sqrt{1 - (\beta_0 \Delta x)^2} = 1.02263$
#      ($+2.26\%$ in $\beta_q$ in `--full`; $+5.32\%$ in `--quick`).
#   2. *Discrete carrier wavenumber shift in $R(\beta_{\text{sync}})$ ($-0.45\%$ in $L_{\text{beat}}$)*:
#      Because $\beta_{\text{sync}} = \arcsin(\beta_0 \Delta x)/\Delta x = 1.0074\,\beta_0 > \beta_0$,
#      the transverse decay parameter $\tau_{\text{sync}} = \beta_{\text{sync}}/\gamma_0$
#      is slightly larger on the grid, raising the fundamental planar reduction
#      factor by $R(\beta_{\text{sync}}) / R(\beta_0) = 1.00456$ ($+0.46\%$ in `--full`;
#      $+1.05\%$ in `--quick`). Combining (1) and (2) gives an exact discrete-grid
#      analytical beat length $L_{\text{beat,grid}} = 41.357\;\text{mm}$ in `--full`
#      ($39.914\;\text{mm}$ in `--quick`), accounting for $2.67\%$ of the $3.00\%$ shift.
#   3. *Finite Gaussian RF gap width and uniform transverse gap excitation ($-0.33\%$ in $L_{\text{beat}}$)*:
#      The remaining $0.14\;\text{mm}$ ($0.33\%$) difference between $L_{\text{beat,FIT}} = 41.218\;\text{mm}$
#      and $L_{\text{beat,grid}} = 41.357\;\text{mm}$ comes from the finite Gaussian
#      gap width ($\sigma_{\text{gap}} = 2\Delta x = 0.64\;\text{mm}$) and the uniform
#      transverse kick across $|y| \le b$, which weakly excites higher-order even
#      $\text{TM}_{1n}$ space-charge modes near the gap. The downstream 10-cell
#      collector density taper is excluded from the fit domain (`valid` ends $12$
#      cells upstream of the CPML) and does not affect $L_{\text{beat,FIT}}$.
# - **Chu kinetic power & discrete energy conservation (`12_chu_kinetic_power_and_energy.png`)**:
#   Evaluating `beam.kinetic_power()` (`task_7`) confirms $\langle P_{k,+} \rangle = +0.0934\;\text{mW}$
#   ($> 0$) for the fast wave and $\langle P_{k,-} \rangle = -0.0729\;\text{mW}$ ($< 0$)
#   for the slow wave, matching the analytical ratio $P_{k,+} / |P_{k,-}| = \omega_+ / \omega_- = 1.2817$
#   to machine precision. In the coupled 2D FIT + `LinearizedBeam` time-domain
#   audit (`run_chu_energy_and_slow_wave_audit`), the monitored discrete energy is
#   the half-step positive-definite functional
#   $$U_{\text{tot}}^{n+1/2}
#     = \underbrace{\frac{1}{4}\sum \varepsilon \left(|E^{n+1}|^2 + |E^n|^2\right)\Delta V_e
#     + \frac{1}{2}\sum \mu |H^{n+1/2}|^2 \Delta V_h}_{U_{\text{EM}}^{n+1/2}}
#     + \underbrace{\sum \Delta V \left[\frac{1}{2} n_0 \gamma_0^3 m_e |v_1^{n+1/2}|^2
#     + \operatorname{Re}\!\left((\rho_1^{n+1/2})^* V_1^{n+1/2}\right)\right]}_{U_{\text{kin}}^{n+1/2}}.$$
#   Over $2400$ steps in `--full` ($48 \times 48$), the relative peak-to-peak
#   variation $\big(\max U_{\text{tot}} - \min U_{\text{tot}}\big) / \langle U_{\text{tot}} \rangle$
#   is **$4.21 \times 10^{-6}$ ($0.00042\%$)**, down from **$9.48 \times 10^{-6}$
#   ($0.00095\%$)** over $1600$ steps in `--quick` ($32 \times 32$; by contrast,
#   an un-centered leapfrog evaluation where $|E^{n+1}|^2$ is not averaged
#   with $|E^n|^2$ exhibits a $\sim 7.9 \times 10^{-4}$ ripple). Why is $U_{\text{tot}}^{n+1/2}$ conserved to $O(\Delta t^2)$
#   rather than machine precision ($\sim 10^{-15}$)?
#   1. The arithmetic electric-energy average $\frac{1}{4}(|E^{n+1}|^2 + |E^n|^2)$
#      differs from the exact staggered leapfrog bilinear invariant
#      $\frac{1}{2}\operatorname{Re}(E^n (E^{n+1})^*)$ by
#      $\frac{1}{4}|E^{n+1} - E^n|^2 = O(\omega^2 \Delta t^2)$.
#   2. In `_pre_step`, the sequential Crank–Nicolson update advances $v_1^{n+1/2}$
#      from $E_s^n$ and $\rho_1^{n+1/2}$ from $\bar{v}_1^n = \frac{1}{2}(v_1^{n+1/2} + v_1^{n-1/2})$,
#      and then injects the endpoint current $J_{1,s}^{n+1/2}$ (rather than the
#      midpoint average $\bar{J}_{1,s}^n$) into Ampère's law, introducing an
#      $O(\Delta t^2)$ operator-splitting phase offset between $U_{\text{EM}}^{n+1/2}$
#      and $U_{\text{kin}}^{n+1/2}$.
#   Because all eigenvalues of the coupled amplification matrix lie strictly on
#   the unit circle ($|\lambda_i| = 1$), there is **zero secular energy drift**,
#   and the bounded ripple scales as $(\Delta t_{\text{full}} / \Delta t_{\text{quick}})^2 = (32/48)^2 = 0.4444$
#   (measured ratio: $4.214 \times 10^{-6} / 9.479 \times 10^{-6} = 0.4446$).
#
# ## 8. Mesh resolution convergence study
#
# Tasks `0`, `1`, and `2` (`disp_res_coarse`, `disp_res_medium`, `disp_res_fine`)
# refine both the transverse and axial grid spacing at the $5.0\;\text{GHz}$,
# $b = 0.5\;\text{mm}$, $a = 2.0\;\text{mm}$ benchmark point against the
# analytical Branch–Mihran planar root $R_{\text{BM}} = 0.46622$ and the
# Richardson-extrapolated continuum limit $R_\infty = 0.46652$ (`--full`):
#
# | Level | Grid $(N_x, N_y)$ | Cells across $b$ | `cpw` ($\lambda_e/\Delta x$) | Beat periods $N_{\text{beat}}$ | $R_{\text{FIT}}$ | $|R_{\text{FIT}} - R_{\text{BM}}|/R_{\text{BM}}$ | $|R_{\text{FIT}} - R_\infty|/R_\infty$ |
# |---|---|---|---|---|---|---|---|
# | Coarse (`task_0`) | $16 \times 32$ | $4.0$ | $16.0$ | $1.85$ | $0.46576$ | $9.79 \times 10^{-4}$ ($0.098\%$) | $1.63 \times 10^{-3}$ |
# | Medium (`task_1`) | $24 \times 48$ | $6.0$ | $24.0$ | $1.85$ | $0.46709$ | $1.86 \times 10^{-3}$ ($0.186\%$) | $1.20 \times 10^{-3}$ |
# | Fine (`task_2`, `--quick`) | $32 \times 64$ | $8.0$ | $32.0$ | $1.85$ | $0.46755$ | $2.85 \times 10^{-3}$ ($0.285\%$) | — |
# | Fine (`task_2`, `--full`) | $64 \times 128$ | $16.0$ | $64.0$ | $2.60$ | $0.46660$ | $8.22 \times 10^{-4}$ ($0.082\%$) | $1.69 \times 10^{-4}$ |
#
# **Convergence verdict (`--full`)**: The metric $R_{\text{FIT}}$ is **converged**
# at the accepted fine level (`task_2`, $64 \times 128$): the relative change
# between the last two `--full` resolution levels (`medium` $\to$ `fine`) is
# $|R_{\text{FIT,fine}} - R_{\text{FIT,medium}}| / R_{\text{FIT,fine}} = |0.46660 - 0.46709| / 0.46660 = 1.03 \times 10^{-3}$
# ($0.103\%$), and the agreement with the analytical Branch–Mihran root
# $R_{\text{BM}} = 0.46622$ is $8.22 \times 10^{-4}$ ($0.082\%$).
#
# **Spectral window comparison between `--quick` and `--full`**: In `--quick`, all three levels
# used the same short FFT window ($N_{\text{beat}} = 1.85$), where Hann-window
# mainlobe overlap between the fast and slow spectral peaks biased the
# log-parabolic peak interpolation upward by $\approx +0.2\%$, making $R_{\text{FIT}}$
# appear to converge monotonically toward $0.46814$ ($0.46576 \to 0.46709 \to 0.46755$).
# In `--full`, `task_2` simultaneously refines the grid to $64 \times 128$ ($16$
# cells across $b$, $64\;\text{cpw}$) and extends the integration window to
# $N_{\text{beat}} = 2.60$ beat periods ($56\,171$ steps). The narrower FFT
# mainlobe eliminates the spectral-overlap bias, so $R_{\text{FIT}}$ drops back
# from $0.46709$ (`medium`) to $0.46660$ (`fine`), matching $R_{\text{BM}} = 0.46622$
# to within $0.082\%$ (`13_resolution_convergence.png`).

# %% [markdown]
# ## 9. Limits & failure modes: when PIC is required
#
# Because `LinearizedBeam` solves the **Eulerian small-signal cold-fluid**
# equations, it is fast, noise-free, and exactly differentiable (`BeamTranspose`),
# making it ideal for small-signal dispersion, cold/hot S-parameters, Pierce gain
# rate, and adjoint optimization. However, as shown in
# `14_linear_vs_nonlinear_bunching_limits.png`, the linearized fluid approximation
# breaks down when any of the following physical thresholds is crossed:
#
# 1. **Trajectory overtaking / wave breaking ($X \ge 1$)**: In a ballistic or
#    space-charge-limited drift region, the arrival phase of an electron leaving
#    the modulating gap at phase $\phi_0 = \omega t_0$ is
#    $\psi(\phi_0) = \phi_0 - X \sin\phi_0$, where
#    $$X(z) = \frac{\beta_0 V_m M}{2 V_0}\,\frac{\sin(\beta_q z)}{\beta_q}$$
#    is the Webster–Hahn bunching parameter. At $X = 1$, $d\psi / d\phi_0 = 0$:
#    fast electrons overtake slow electrons, the phase-space curve $(z, v_z)$
#    folds into a multi-stream distribution, and the Eulerian charge density
#    diverges ($\rho_1 \to \infty$). Only a Lagrangian particle/PIC code can
#    represent multi-valued velocities at a single spatial point.
# 2. **Harmonic generation & gain compression ($X \gtrsim 0.5$)**: Expand the
#    bunched convection current in a Fourier–Bessel series $I_n(z) = 2 I_0 J_n(n X)$.
#    The linearized fluid model retains only the first-order term
#    $I_1^{\text{lin}} / I_0 = X$, which overpredicts the fundamental current by
#    $> 12\%$ at $X = 1$, misses the saturation maximum $\max_X [2 J_1(X)] = 1.164 I_0$
#    at $X = 1.841$, and identically zeroes all higher harmonics ($2\omega, 3\omega$)
#    because the nonlinear convective terms $v_1 \partial_s v_1$ and
#    $\partial_s(\rho_1 v_1)$ are dropped.
# 3. **Electron trapping & large-signal saturation**: In high-gain TWTs or BWOs
#    approaching saturation, electrons lose enough kinetic energy to drop out of
#    synchronism and become trapped in the potential troughs of the slow circuit
#    wave, executing synchrotron phase-space oscillations that require PIC.
# 4. **Thermal velocity spread & finite-$B_0$ transverse dynamics**: Warm-beam
#    Landau damping ($\sigma_{v_z} \gtrsim \omega_q / \beta$) and finite
#    magnetic focusing effects (Brillouin scalloping, cyclotron-space-charge
#    hybrid modes, and sheet-beam $E \times B$ diocotron instabilities) are
#    excluded by the cold, 1D-confined ($B_0 \to \infty$) fluid ansatz.
#
# ## 10. References
#
# 1. S. Ramo, "Space Charge and Field Waves in an Electron Beam," *Phys. Rev.*
#    **56**(3), 276–283 (1939),
#    DOI: [10.1103/PhysRev.56.276](https://doi.org/10.1103/PhysRev.56.276).
# 2. G. M. Branch and T. G. Mihran, "Plasma Frequency Reduction Factors in
#    Electron Beams," *IRE Trans. Electron Devices* **2**(2), 3–11 (1955),
#    DOI: [10.1109/T-ED.1955.14065](https://doi.org/10.1109/T-ED.1955.14065).
# 3. T. Mealy, R. Marosi, and F. Capolino, "Reduced Plasma Frequency Calculation
#    Based on Particle-in-Cell Simulations," *IEEE Trans. Plasma Sci.* **50**(10),
#    3570–3577 (2022),
#    DOI: [10.1109/TPS.2022.3201036](https://doi.org/10.1109/TPS.2022.3201036)
#    (`arXiv:2205.05806`).
# 4. H. A. Haus and D. L. Bobroff, "Small Signal Power Theorem for Electron
#    Beams," *J. Appl. Phys.* **28**(6), 694–704 (1957),
#    DOI: [10.1063/1.1722832](https://doi.org/10.1063/1.1722832).
# 5. T. Mealy and F. Capolino, "Traveling Wave Tube Eigenmode Solution for
#    Beam-Loaded Slow Wave Structure Based on Particle-in-Cell Simulations,"
#    *IEEE Trans. Plasma Sci.* **50**(3), 635–648 (2022),
#    DOI: [10.1109/TPS.2022.3147171](https://doi.org/10.1109/TPS.2022.3147171).


# %%
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Tutorial 8.1 · Space-charge waves on a drifting electron beam (LinearizedBeam)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run coarse/medium-fast study locally (< 60 s) and render all 14 figures",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run high-resolution study across all tasks and render all 14 figures",
    )
    ap.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single study task I (0-indexed)",
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
    for t in tasks:
        run_single_task(t, args.out)
    generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR)


if __name__ == "__main__":
    main()
