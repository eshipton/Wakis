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
# # Tutorial 1.2 · Reflection and transmission at a slab
#
# **Abstract.** Building directly on Tutorial 1.1, this second foundational
# tutorial introduces **dielectric materials**, **absorbing boundary conditions
# (`cpml` vs. `abc` vs. `pec`)**, **Poynting flux monitors (`FluxMonitor`)**,
# the **two-run reference normalization workflow** (Oskooi *et al.* 2010 [4];
# Oskooi & Johnson 2013 [5]), **oblique-incidence Bloch-periodic boundaries**,
# and **Kottke (2008) subpixel permittivity smoothing** [8] on interfaces that do
# not align with grid lines. We validate Wakis time-domain reflection and
# transmission spectra against the exact **Fresnel / Airy transfer matrix**
# across multiple Fabry–Pérot fringes and incidence angles (including
# **Brewster's angle** $\theta_B = \arctan(n_{\text{slab}}/n_0)$), benchmark
# spurious CPML reflections down to $-91\;\text{dB}$ across layer count and
# polynomial grading order, and demonstrate why subpixel smoothing restores
# second-order $\mathcal{O}(\Delta z^2)$ accuracy over first-order
# $\mathcal{O}(\Delta z^1)$ staircasing.
#
# ---
#
# ## Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. Derive the **Fresnel reflection and transmission coefficients** ($r_s, t_s$
#    and $r_p, t_p$), the $2\times 2$ **characteristic transfer matrix** of a
#    dielectric slab, and the **Airy Fabry–Pérot formulas**
#    (`wakis.templates.formulas.slab_reflection_transmission`).
# 2. Configure **convolutional perfectly matched layers (`cpml`)** (Berenger
#    1994 [6]; Roden & Gedney 2000 [7]), measure their spurious numerical
#    reflection $|r_{\text{PML}}(f)|^2$ (in $\text{dB}$) as a function of layer
#    count `n_pml` and polynomial grading order `pml_exp`, and compare against a
#    `'pec'` wall ($0\;\text{dB}$) and a first-order Mur `'abc'` boundary.
# 3. Execute the **two-run normalization workflow** with `FluxMonitor` (empty
#    reference run for incident flux $\Phi_{\text{inc}}(f)$; slab run for
#    transmitted flux $T = \Phi_t/\Phi_{t,\text{ref}}$ and reflected flux
#    $R = 1 - \Phi_r/\Phi_{r,\text{ref}}$) and verify discrete energy
#    conservation $R(f) + T(f) = 1$ to $< 10^{-6}$.
# 4. Launch oblique-incidence s/TE and p/TM plane waves using **Bloch-periodic
#    boundaries** (`bc_low`/`bc_high = 'bloch'`, `k_bloch`, `dtype=np.complex128`)
#    and reproduce **Brewster's angle** total transmission ($R_p(\theta_B) = 0$).
# 5. Quantify the discretization error when dielectric interfaces lie **off the
#    grid lines**, and show that **Kottke (2008) subpixel smoothing**
#    (`assign_materials(..., averaging='kottke')`) achieves second-order
#    $\mathcal{O}(\Delta z^2)$ convergence whereas `'staircase'` rounding stalls
#    at first order $\mathcal{O}(\Delta z^1)$.
#
# ## Where each feature is used next in the book
#
# | Solver feature introduced here | Physical role in Tutorial 1.2 | Where it is used later |
# |---|---|---|
# | `FluxMonitor` + two-run reference normalization | Incident, reflected ($1 - \Phi_r/\Phi_{r,\text{ref}}$), and transmitted ($\Phi_t/\Phi_{t,\text{ref}}$) power spectra | Ch. 2.1 & 2.2 (plasmonic slabs & metamaterials), Ch. 3.4 & 3.5 (Mie cross sections), Ch. 5 (waveguide S-parameters) |
# | `bc_low`/`bc_high = 'cpml'` (`n_pml`, `pml_exp`) | Stretched-coordinate absorbing boundary with $<-70\;\text{dB}$ reflection | All open-region radiation, scattering, and waveguide tutorials (Ch. 2–9) |
# | `'bloch'` boundaries + `k_bloch` + `complex128` | Oblique incidence at transverse wavevector $k_x = k_0\sin\theta$ | Ch. 6.1 & 6.2 (Bragg gratings & slow-wave Brillouin zones) |
# | `assign_materials(..., averaging='kottke' \| 'staircase')` | Anisotropic subpixel permittivity smoothing on off-grid interfaces | Ch. 4.1 (DBR micro-RCLED), Ch. 5.2 (Euler bends), Ch. 6.1 (50 nm Bragg grating teeth), Ch. 7.2 (tilted SLEDs) |
#
# ---
#
# ## 1. Physics & numerics: Fresnel equations, Fabry–Pérot etalons, CPML, and Kottke smoothing
#
# ### 1.0 Simulation setup schematic
#
# ![Tutorial 1.2 Figure A — geometry, dielectric slab, source, monitors, and boundaries](figures/00a_setup_geometry.png)
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 17.9 nm × 17.9 nm × 2.84 µm (z ∈ [0 m, 2.84 µm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 1 × 1 × 159 (159 cells) |
# | Discretization | Δz & resolution | Δ_min = 17.9 nm, 28.0 cells/λ (in n_max = 2.00), 35.0 cells across feature (625 nm) |
# | Boundary | -x face | Periodic |
# | Boundary | +x face | Periodic |
# | Boundary | -y face | Periodic |
# | Boundary | +y face | Periodic |
# | Boundary | -z face | CPML (10 cells, 178.6 nm) |
# | Boundary | +z face | CPML (10 cells, 178.6 nm) |
# | Material | Background | Vacuum, ε = 1.00 (isotropic) |
# | Material | Region 1: Dielectric slab (n = 2.0) | ε_r = 4.00, μ_r = 1.00, model = Isotropic dielectric, z ∈ [1 µm, 1.62 µm] (d = 625 nm) |
# | Source | 1 · CurrentSource | pos: z = 536 nm, pol: J_x (E_x), pulse: f₀ = 299.8 THz, Δf = 113.9 THz |
# | Monitor | 2 · FluxMonitor | FluxMonitor (z = 857 nm), pos: z = 857 nm, freqs: 41 freqs in [179.9, 419.7] THz |
# | Monitor | 3 · FluxMonitor | FluxMonitor (z = 1.97 µm), pos: z = 1.97 µm, freqs: 41 freqs in [179.9, 419.7] THz |
# | Variation | Run 1a: Reference run (§1.1) | Vacuum everywhere (ε_r = 1.00); records incident flux Φ_ref(f) and E_ref(z_r, t) |
# | Variation | Run 1b: Slab run (§1.1–1.2) | Dielectric slab ε_r = 4.00 (n = 2.00, d = 0.625 µm); R = 1 − Φ_r/Φ_ref, T = Φ_t/Φ_ref |
# | Variation | Run 2: Oblique / Brewster scan (§1.3) | 2D (x, z) domain; Bloch ±x boundary (k_x = k₀ sin θ_i), θ_i ∈ [0°, 75°], glass n = 1.50 (θ_B = 56.31°) |
# | Variation | Run 3: Boundary benchmark (§1.4) | Vacuum domain; +z test boundary ∈ {CPML (N_pml = 4, 8, 12, 16), Mur ABC, PEC} |
# | Variation | Run 4: Subpixel convergence (§1.5) | Off-grid slab d = 0.655 µm; Kottke tensor vs staircase at cpw ∈ {10, 15, 20, 30, 40} |
#
# ![Tutorial 1.2 Figure B — discretization and excitation](figures/00b_setup_mesh_and_excitation.png)
#
# ![Tutorial 1.2 Figure C — setups for oblique Bloch-periodic and boundary benchmark runs](figures/00c_setup_oblique_and_bc_runs.png)
#
# The setup figures and table above (generated directly from the live
# `SolverFIT3D`, `PlanarSlabZ`, `CurrentSource`, and `FluxMonitor` objects via
# `wakis.schematic.save_setup_figures`) summarize the reference Fabry–Pérot slab
# simulation (`task_0` at $\text{cpw}_{\text{slab}} = 28$ cells per media
# wavelength) alongside the oblique-incidence Bloch setup (§1.3) and boundary
# reflection benchmark (§1.4):
#
# * **Where the computational box is**: $z \in [0, 2.84]\;\mu\text{m}$
#   ($L_z = 2.84\;\mu\text{m}$, $N_z = 159$ cells at $\Delta z = 17.86\;\text{nm}$,
#   corresponding to $56\;\text{cells}/\lambda_0$ in vacuum and
#   $28\;\text{cells}/(\lambda_0/n_{\text{slab}})$ inside the slab) with a
#   single transverse cell $x, y \in [0, 17.86]\;\text{nm}$.
# * **Which boundary conditions terminate each face**: transverse faces
#   $\pm x$ and $\pm y$ are **periodic** (`bc_low=['periodic', 'periodic', 'cpml']`)
#   for normal incidence, or **Bloch-periodic** (`'bloch'` with transverse
#   wavevector $k_x = k_0\sin\theta$) for oblique incidence; longitudinal faces
#   $\pm z$ are terminated with $N_{\text{pml}} = 10$ cells ($178.6\;\text{nm}$,
#   hatched) of **stretched-coordinate CPML** (compared in §4 against a
#   first-order Mur `'abc'` [11] and a `'pec'` wall).
# * **Where the dielectric slab is**: a lossless dielectric slab of refractive
#   index $n_{\text{slab}} = 2.0$ ($\varepsilon_r = 4.0$) and thickness
#   $d = 1.25\,\lambda_0/n_{\text{slab}} = 625\;\text{nm}$ occupies
#   $z \in [1.00, 1.625]\;\mu\text{m}$, assigned to the FIT grid via
#   `assign_materials(..., averaging='kottke')`.
# * **How the incident wave is launched and where it extends**: a transparent
#   soft `CurrentSource` sheet at $z_{\text{src}} = 536\;\text{nm}$ injects a
#   broadband `GaussianSource` pulse ($f_0 = 299.79\;\text{THz}$,
#   $\lambda_0 = 1.0\;\mu\text{m}$, $\Delta f = 113.92\;\text{THz}$) with
#   spatial packet length $\Delta z_{\text{pulse}} \approx 1.68\;\mu\text{m}$.
# * **Where the monitors are**: an upstream reflection `FluxMonitor` plane at
#   $z_r = 857\;\text{nm}$ (between the source and the front slab face) and a
#   downstream transmission `FluxMonitor` plane at $z_t = 1.97\;\mu\text{m}$
#   (between the back slab face and the $+z$ CPML).
#
# ### 1.1 Fresnel coefficients and Brewster's angle
#
# When a monochromatic plane wave in medium $n_1$ impinges at angle $\theta_1$ on
# a planar interface with medium $n_2$, continuity of tangential $\mathbf{E}_\parallel$
# and $\mathbf{H}_\parallel$ yields **Snell's law** $n_1\sin\theta_1 = n_2\sin\theta_2$
# and the **Fresnel amplitude coefficients** (Born & Wolf 1999 [9]):
#
# $$r_s = \frac{n_1\cos\theta_1 - n_2\cos\theta_2}{n_1\cos\theta_1 + n_2\cos\theta_2},
# \qquad
# t_s = \frac{2 n_1\cos\theta_1}{n_1\cos\theta_1 + n_2\cos\theta_2}
# \quad (\text{s / TE polarization}),$$
#
# $$r_p = \frac{n_2\cos\theta_1 - n_1\cos\theta_2}{n_2\cos\theta_1 + n_1\cos\theta_2},
# \qquad
# t_p = \frac{2 n_1\cos\theta_1}{n_2\cos\theta_1 + n_1\cos\theta_2}
# \quad (\text{p / TM polarization}).$$
#
# For p/TM polarization ($\mathbf{E}$ in the plane of incidence), $r_p$ vanishes
# when $\theta_1 + \theta_2 = \pi/2$, defining **Brewster's angle**:
#
# $$\theta_B = \arctan\!\left(\frac{n_2}{n_1}\right)
# \quad \left(\theta_B = \arctan(2) \approx 63.435^\circ \text{ for } n_1 = 1,\; n_2 = 2\right).$$
#
# ### 1.2 Dielectric slab transfer matrix and Airy Fabry–Pérot fringes
#
# Across a homogeneous slab of refractive index $n_2$, thickness $d$, and
# single-pass phase thickness $\delta = (2\pi f / c_0)\,n_2\,d\cos\theta_2$, the
# tangential field amplitudes at $z = z_{\text{lo}}$ and $z = z_{\text{hi}} = z_{\text{lo}} + d$
# are linked by the $2\times 2$ **characteristic transfer matrix** (Born & Wolf
# 1999 [9]; Yeh, Yariv & Hong 1977 [10]):
#
# $$\begin{pmatrix} U(z_{\text{lo}}) \\ V(z_{\text{lo}}) \end{pmatrix}
# = \begin{pmatrix} \cos\delta & \frac{i}{q_2}\sin\delta \\ i q_2\sin\delta & \cos\delta \end{pmatrix}
# \begin{pmatrix} U(z_{\text{hi}}) \\ V(z_{\text{hi}}) \end{pmatrix},
# \qquad
# q_m = \begin{cases} n_m\cos\theta_m & (\text{s / TE}), \\ n_m / \cos\theta_m & (\text{p / TM}). \end{cases}$$
#
# For a symmetric surrounding medium ($q_1 = q_3$) with single-interface
# reflection $r_{12} = (q_1 - q_2)/(q_1 + q_2) = -r_{21}$ and $R_0 = |r_{12}|^2$,
# coherent summation of all multiple internal bounces yields the **Airy formulas**
# (implemented for normal incidence in
# `wakis.templates.formulas.slab_reflection_transmission`):
#
# $$r(f, \theta) = \frac{r_{12}\left(1 - e^{-2i\delta}\right)}{1 - r_{12}^2 e^{-2i\delta}},
# \qquad
# t(f, \theta) = \frac{(1 - r_{12}^2)\,e^{-i\delta}}{1 - r_{12}^2 e^{-2i\delta}},$$
#
# $$R(f, \theta) = |r|^2 = \frac{4 R_0 \sin^2\delta}{(1 - R_0)^2 + 4 R_0 \sin^2\delta},
# \qquad
# T(f, \theta) = |t|^2 = \frac{(1 - R_0)^2}{(1 - R_0)^2 + 4 R_0 \sin^2\delta} = 1 - R(f, \theta).$$
#
# * **Half-wave etalon resonances** ($\delta = m\pi \iff d\cos\theta_2 = m\frac{\lambda_0}{2 n_2}$):
#   $\sin\delta = 0 \implies R = 0,\; T = 1$ (destructive interference of front-
#   and back-face reflections). Notice that at **Brewster's angle**
#   $\theta_1 = \theta_B$, $r_{12,p} = 0 \implies R_0 = 0$, so a p-polarized wave
#   transmits with $R_p(f, \theta_B) = 0$ and $T_p(f, \theta_B) = 1$ at **every
#   frequency and slab thickness**!
# * **Quarter-wave reflectance maxima** ($\delta = (m + \frac{1}{2})\pi \iff d\cos\theta_2 = (2m+1)\frac{\lambda_0}{4 n_2}$):
#   $\sin^2\delta = 1 \implies R_{\max} = \left(\frac{q_1^2 - q_2^2}{q_1^2 + q_2^2}\right)^2$
#   (at normal incidence with $n_2 = 2, n_1 = 1$, $R_{\max} = (3/5)^2 = 0.36$ and
#   $T_{\min} = 0.64$).
#
# ### 1.3 Why two runs are needed for `FluxMonitor` reflection normalization
#
# A `FluxMonitor` plane at $z = z_r$ integrates the normal Poynting flux
# $\Phi(f) = \operatorname{Re}\iint (\mathbf{E}(f)\times\mathbf{H}^*(f))\cdot\hat{\mathbf{z}}\,\mathrm{d}x\,\mathrm{d}y$.
# When $z_r$ is placed between the source ($z_{\text{src}}$) and the slab
# ($z_{\text{lo}}$), the total field is the sum of the $+z$-going incident wave
# and the $-z$-going reflected wave:
#
# $$E_x(z_r, f) = E_{\text{inc}}(f) + E_{\text{refl}}(f),
# \qquad
# H_y(z_r, f) = Y_1\left(E_{\text{inc}}(f) - E_{\text{refl}}(f)\right).$$
#
# Because the cross-term $E_{\text{refl}}E_{\text{inc}}^* - E_{\text{inc}}E_{\text{refl}}^* = 2i\operatorname{Im}(E_{\text{refl}}E_{\text{inc}}^*)$
# is purely imaginary, its real part vanishes identically! Thus the net $+z$ flux
# recorded in the slab run is $\Phi_r(f) = \Phi_{\text{inc}}(f) - \Phi_{\text{refl}}(f)$.
# Running a fast **empty-domain reference simulation** (Run 1) provides
# $\Phi_{r,\text{ref}}(f) = \Phi_{\text{inc}}(f)$ and $\Phi_{t,\text{ref}}(f) = \Phi_{\text{inc}}(f)$,
# from which the reflected and transmitted power fractions follow directly
# (Oskooi *et al.* 2010 [4]; Oskooi & Johnson 2013 [5]):
#
# $$R(f) = \frac{\Phi_{\text{refl}}(f)}{\Phi_{r,\text{ref}}(f)} = 1 - \frac{\Phi_r(f)}{\Phi_{r,\text{ref}}(f)},
# \qquad
# T(f) = \frac{\Phi_t(f)}{\Phi_{t,\text{ref}}(f)}.$$
#
# ### 1.4 Stretched-coordinate CPML and Kottke subpixel smoothing
#
# * **Convolutional PML (`cpml`)**: Berenger's PML [6] and Roden & Gedney's
#   complex-frequency-shifted CPML [7] stretch the normal derivative inside the
#   boundary layer of thickness $L_{\text{PML}} = N_{\text{pml}}\Delta z$ via
#
#   $$\frac{\partial}{\partial z} \;\longrightarrow\; \frac{1}{s_z(\omega)}\frac{\partial}{\partial z},
#   \qquad
#   s_z(\omega) = \kappa_z(z) + \frac{\sigma_z(z)}{\alpha_z(z) + i\omega\varepsilon_0},$$
#
#   with polynomial grading $\sigma_z(\rho) = \sigma_{\max}(\rho / L_{\text{PML}})^m$
#   (`pml_exp` $= m$). In the continuum limit $s_z$ is impedance-matched at all
#   frequencies and angles; on the discrete FIT grid, the step-to-step change in
#   $\sigma_z$ produces a small residual reflection that decreases exponentially
#   with $N_{\text{pml}}$.
# * **Kottke (2008) subpixel permittivity smoothing**: When a planar interface
#   with unit normal $\hat{\mathbf{n}}$ cuts a grid cell with volume fill fraction
#   $w \in (0, 1)$ of permittivity $\varepsilon_s$ in background $\varepsilon_b$,
#   rounding $w \to \{0, 1\}$ (`averaging='staircase'`) shifts the interface by up
#   to $\pm\Delta z/2$, introducing an $\mathcal{O}(\Delta z^1)$ optical-thickness
#   error. Perturbation theory (Kottke, Farjadpour & Johnson 2008 [8]) shows that
#   continuous tangential $\mathbf{E}_\parallel$ requires the **arithmetic mean**
#   $\langle\varepsilon\rangle = w\varepsilon_s + (1-w)\varepsilon_b$ while
#   continuous normal $D_\perp = \varepsilon E_\perp$ requires the **harmonic
#   mean** $\langle\varepsilon^{-1}\rangle^{-1} = (w/\varepsilon_s + (1-w)/\varepsilon_b)^{-1}$:
#
#   $$\varepsilon_{\text{eff},c}^{-1} = n_c^2\left(\frac{w}{\varepsilon_s} + \frac{1-w}{\varepsilon_b}\right)
#   + (1 - n_c^2)\frac{1}{w\varepsilon_s + (1-w)\varepsilon_b},$$
#
#   which places the effective optical boundary at its exact subcell coordinate
#   and restores second-order $\mathcal{O}(\Delta z^2)$ accuracy.


# %%
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
from scipy.constants import c as c0

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

TUTORIAL_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    EPS_SLAB,
    F0,
    LAM0,
    N_SLAB,
    PlanarSlabZ,
    fresnel_single_interface,
    render_all_figures,
    run_cpml_reflection_task,
    run_fabry_perot_and_oblique_task,
    run_subpixel_convergence_task,
    slab_rt_oblique,
)

from wakis import GridFIT3D, SolverFIT3D  # noqa: E402
from wakis.estimate import estimate, preflight  # noqa: E402
from wakis.field_monitors import DFTMonitor, FluxMonitor  # noqa: E402
from wakis.sources import CurrentSource, GaussianSource  # noqa: E402
from wakis.templates.formulas import slab_reflection_transmission  # noqa: E402
from wakis.templates.shapes import assign_materials  # noqa: E402

DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"

# Run the interactive walkthrough cells when invoked inside Jupyter / ipython
# or directly as a script with no CLI flags.
_RUN_WALKTHROUGH = "ipykernel" in sys.modules or (
    __name__ == "__main__" and len(sys.argv) == 1
)

# %% [markdown]
# ## 2. Step-by-step Wakis feature walkthrough
#
# Before launching the multi-task benchmark and resolution sweep, we walk
# through each new solver concept introduced in this tutorial with short,
# self-contained runnable cells.
#
# ### Step 1 — Analytical Fresnel, Brewster's angle, and Fabry–Pérot checks
#
# Let us verify the transfer-matrix identities for our reference dielectric slab
# ($n_{\text{slab}} = 2.0$, $\varepsilon_r = 4.0$ in vacuum $n_1 = n_3 = 1.0$):
#
# 1. At a single planar interface and normal incidence ($\theta_1 = 0$),
#    $r = (1 - 2)/(1 + 2) = -1/3$, so $R_0 = |r|^2 = 1/9 \approx 0.1111$.
# 2. For $p$-polarization (TM), single-interface reflection vanishes at
#    **Brewster's angle** $\theta_B = \arctan(n_{\text{slab}}/n_1) = \arctan(2) \approx 63.435^\circ$,
#    so a slab of any thickness is $100\%$ transparent ($R_p(\theta_B) = 0$,
#    $T_p(\theta_B) = 1$).
# 3. At normal incidence, every half-wave multiple $d = m\,\lambda_0/(2 n_{\text{slab}})$
#    gives $\delta = m\pi$ and $R = 0, T = 1$, whereas every quarter-wave odd
#    multiple $d = (2m+1)\,\lambda_0/(4 n_{\text{slab}})$ maximizes reflection at
#    $R_{\max} = ((n_{\text{slab}}^2 - 1)/(n_{\text{slab}}^2 + 1))^2 = (3/5)^2 = 0.36$.

# %%
if _RUN_WALKTHROUGH:
    # 1. Single-interface Fresnel check at normal incidence and Brewster's angle
    _, _, r0_s, t0_s = fresnel_single_interface(1.0, N_SLAB, 0.0, "s")
    theta_brewster = np.arctan(N_SLAB / 1.0)
    _, _, rb_s, tb_s = fresnel_single_interface(1.0, N_SLAB, theta_brewster, "s")
    _, _, rb_p, tb_p = fresnel_single_interface(1.0, N_SLAB, theta_brewster, "p")

    # 2. Quarter-wave and half-wave Fabry-Perot slab check vs Wakis template formula
    d_qw = 0.25 * (LAM0 / N_SLAB)
    d_hw = 0.50 * (LAM0 / N_SLAB)
    R_qw, T_qw = slab_reflection_transmission(EPS_SLAB, d_qw, np.array([F0]))
    R_hw, T_hw = slab_reflection_transmission(EPS_SLAB, d_hw, np.array([F0]))
    ob_brewster = slab_rt_oblique(EPS_SLAB, 0.375 * (LAM0 / N_SLAB), F0, theta_brewster, "p")

    print(
        f"Single interface (theta = 0 deg): R = {float(r0_s[0]):.6f} (exact 1/9 = {1/9:.6f}), "
        f"R + T = {float(r0_s[0] + t0_s[0]):.6f}"
    )
    print(
        f"Brewster angle theta_B = {np.rad2deg(theta_brewster):.3f} deg: "
        f"single-interface R_p = {float(rb_p[0]):.2e}, slab R_p = {float(ob_brewster['R'][0]):.2e}, "
        f"slab T_p = {float(ob_brewster['T'][0]):.6f}"
    )
    print(
        f"Quarter-wave slab (d = lam0 / 4n): R_max = {float(R_qw[0]):.6f} (exact (3/5)^2 = 0.360000), "
        f"Half-wave slab (d = lam0 / 2n): R_min = {float(R_hw[0]):.2e}"
    )

# %% [markdown]
# ### Step 2 — Absorbing boundaries: `cpml` vs `abc` vs `pec`
#
# In an open-region scattering or transmission calculation, waves leaving the
# domain at $z_{\min}$ and $z_{\max}$ must not reflect back into the monitors.
# Wakis supports three termination families on `SolverFIT3D`:
#
# * `bc_high=[..., 'pec']`: a perfect electric conductor ($E_t = 0$), which
#   reflects $100\%$ of the incident power ($0\;\text{dB}$, $r = -1$).
# * `bc_high=[..., 'abc']`: a first-order Mur absorbing boundary condition,
#   which cancels waves at a single phase velocity $c_0$ to roughly $-48\;\text{dB}$
#   at normal incidence on a 20–24 cpw grid, but degrades rapidly with angle and
#   grid dispersion.
# * `bc_high=[..., 'cpml']`: a stretched-coordinate Convolutional PML of
#   thickness `n_pml` cells and polynomial grading order `pml_exp`.
#
# To measure the **spurious boundary reflection** $|r_{\text{PML}}(f)|^2$ cleanly,
# we compare a domain terminated at $z_{\text{wall}}$ against a longer reference
# domain whose far boundary is unreachable within the time window:
#
# $$r_{\text{bc}}(f) = \frac{E_{x,\text{tot}}(z_{\text{mon}}, f) - E_{x,\text{ref}}(z_{\text{mon}}, f)}{E_{x,\text{ref}}(z_{\text{mon}}, f)}.$$

# %%
if _RUN_WALKTHROUGH:
    dz_demo = LAM0 / 20.0
    x_d = np.array([0.0, dz_demo])
    y_d = np.array([0.0, dz_demo])
    pulse_d = GaussianSource(F0, fwidth=0.35 * F0, cutoff=4.5)

    # Reference domain long enough that the far wall cannot reflect back in nt_d steps
    g_long = GridFIT3D(x=x_d, y=y_d, z=np.arange(161) * dz_demo, verbose=0)
    s_long = SolverFIT3D(
        g_long,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=10,
        cfln=0.5,
        use_stl=False,
        verbose=0,
    )
    nt_d = int(np.ceil((pulse_d.end_time + 2.8 * (20 * dz_demo) / c0) / s_long.dt))
    src_l = CurrentSource(
        pulse_d,
        component="x",
        center=(0.5 * dz_demo, 0.5 * dz_demo, 62 * dz_demo),
        size=(dz_demo, dz_demo, 0.0),
    )
    mon_l = DFTMonitor([F0], ["Ex"], center=(0.5 * dz_demo, 0.5 * dz_demo, 68 * dz_demo), size=(0.0, 0.0, 0.0))
    s_long.emsolve(nt_d, source=[src_l], monitors=[mon_l])
    e_inc_f0 = complex(mon_l.get_field("Ex")[0][0, 0, 0, 0])

    for bc_label, bc_name, n_p in [("PEC wall", "pec", 8), ("CPML (N=4)", "cpml", 4), ("CPML (N=10)", "cpml", 10)]:
        nz_b = 32 + (n_p if bc_name == "cpml" else 0)
        g_b = GridFIT3D(x=x_d, y=y_d, z=np.arange(nz_b + 1) * dz_demo, verbose=0)
        s_b = SolverFIT3D(
            g_b,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", bc_name],
            n_pml=n_p,
            pml_exp=3,
            cfln=0.5,
            use_stl=False,
            verbose=0,
        )
        src_b = CurrentSource(
            pulse_d,
            component="x",
            center=(0.5 * dz_demo, 0.5 * dz_demo, 12 * dz_demo),
            size=(dz_demo, dz_demo, 0.0),
        )
        mon_b = DFTMonitor([F0], ["Ex"], center=(0.5 * dz_demo, 0.5 * dz_demo, 18 * dz_demo), size=(0.0, 0.0, 0.0))
        s_b.emsolve(nt_d, source=[src_b], monitors=[mon_b])
        e_tot_f0 = complex(mon_b.get_field("Ex")[0][0, 0, 0, 0])
        r_db_val = 10.0 * np.log10(max(abs((e_tot_f0 - e_inc_f0) / e_inc_f0) ** 2, 1e-16))
        print(f"{bc_label:14s}: spurious power reflection at f0 = {r_db_val:+6.2f} dB")

# %% [markdown]
# ### Step 3 — `FluxMonitor` and the two-run reference normalization
#
# Next we place a dielectric slab of refractive index $n_{\text{slab}} = 2.0$ and
# thickness $d = 1.25\,\lambda_0/n_{\text{slab}} = 625\;\text{nm}$ in vacuum and
# measure its broadband reflectance $R(f)$ and transmittance $T(f)$ using two
# `FluxMonitor` planes (`z_refl` upstream of the slab, `z_tran` downstream):
#
# * **Project grid rules**: We size $\Delta z$ in the **highest-index material**
#   ($n_{\text{slab}} = 2.0$) at $\text{cpw}_{\text{slab}} = 20$ cells per media
#   wavelength ($\Delta z = \lambda_0/(n_{\text{slab}}\cdot 20) = 25\;\text{nm}$,
#   which corresponds to $40\;\text{cpw}$ in the vacuum background) and puts
#   $d/\Delta z = 25 \ge 4$ cells across the slab thickness. We verify this
#    programmatically with `preflight(...)` and `estimate(...)`.
# * **Two-run normalization**: Run 1 (empty vacuum) measures $\Phi_{r,\text{ref}}(f)$
#   and $\Phi_{t,\text{ref}}(f)$; Run 2 (with slab) measures the net $+z$ flux
#   $\Phi_r(f) = \Phi_{\text{inc}}(f) - \Phi_{\text{refl}}(f)$ and $\Phi_t(f)$,
#   giving $R(f) = 1 - \Phi_r/\Phi_{r,\text{ref}}$ and $T(f) = \Phi_t/\Phi_{t,\text{ref}}$.

# %%
if _RUN_WALKTHROUGH:
    cpw_w = 20
    dz_w = (LAM0 / N_SLAB) / cpw_w  # 25 nm (20 cpw in n=2.0 slab, 40 cpw in vacuum)
    d_slab_w = 1.25 * (LAM0 / N_SLAB)  # 625 nm (exact 25 grid cells)
    n_pml_w = 10
    z_lo_w = (n_pml_w + 20) * dz_w
    z_hi_w = z_lo_w + d_slab_w
    nz_w = n_pml_w + 20 + int(round(d_slab_w / dz_w)) + 20 + n_pml_w

    freqs_w = np.linspace(0.75 * F0, 1.25 * F0, 11)
    pulse_w = GaussianSource(F0, fwidth=0.40 * F0, cutoff=4.5)
    z_src_w = (n_pml_w + 6) * dz_w
    z_r_w = (n_pml_w + 13) * dz_w
    z_t_w = z_hi_w + 7 * dz_w

    # Run 1: Empty reference domain
    g_ref_w = GridFIT3D(
        x=np.array([0.0, dz_w]),
        y=np.array([0.0, dz_w]),
        z=np.arange(nz_w + 1) * dz_w,
        verbose=0,
    )
    s_ref_w = SolverFIT3D(
        g_ref_w,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=n_pml_w,
        cfln=0.5,
        use_stl=False,
        verbose=0,
    )
    nt_w = int(np.ceil((pulse_w.end_time + 18.0 * LAM0 / c0) / s_ref_w.dt))
    est_w = estimate(
        (g_ref_w.Nx, g_ref_w.Ny, g_ref_w.Nz),
        steps=nt_w,
        cpml=True,
        monitors=[(len(freqs_w), 1, 2), (len(freqs_w), 1, 2)],
    )
    pf_problems_w = preflight(est_w, {"ram": 2e9, "gpu": 0}, raise_error=False)
    mr_ref = FluxMonitor(freqs_w, center=(0.5 * dz_w, 0.5 * dz_w, z_r_w), size=(dz_w, dz_w, 0.0))
    mt_ref = FluxMonitor(freqs_w, center=(0.5 * dz_w, 0.5 * dz_w, z_t_w), size=(dz_w, dz_w, 0.0))
    s_ref_w.emsolve(
        nt_w,
        source=[CurrentSource(pulse_w, component="x", center=(0.5 * dz_w, 0.5 * dz_w, z_src_w), size=(dz_w, dz_w, 0.0))],
        monitors=[mr_ref, mt_ref],
    )

    # Run 2: Dielectric slab domain
    g_slab_w = GridFIT3D(
        x=np.array([0.0, dz_w]),
        y=np.array([0.0, dz_w]),
        z=np.arange(nz_w + 1) * dz_w,
        verbose=0,
    )
    s_slab_w = SolverFIT3D(
        g_slab_w,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=n_pml_w,
        cfln=0.5,
        use_stl=False,
        verbose=0,
    )
    assign_materials(s_slab_w, [PlanarSlabZ(z_lo_w, z_hi_w, EPS_SLAB)], F0, averaging="kottke")
    mr_slab = FluxMonitor(freqs_w, center=(0.5 * dz_w, 0.5 * dz_w, z_r_w), size=(dz_w, dz_w, 0.0))
    mt_slab = FluxMonitor(freqs_w, center=(0.5 * dz_w, 0.5 * dz_w, z_t_w), size=(dz_w, dz_w, 0.0))
    s_slab_w.emsolve(
        nt_w,
        source=[CurrentSource(pulse_w, component="x", center=(0.5 * dz_w, 0.5 * dz_w, z_src_w), size=(dz_w, dz_w, 0.0))],
        monitors=[mr_slab, mt_slab],
    )

    R_w = 1.0 - mr_slab.get_flux() / mr_ref.get_flux()
    T_w = mt_slab.get_flux() / mt_ref.get_flux()
    R_ex_w, T_ex_w = slab_reflection_transmission(EPS_SLAB, d_slab_w, freqs_w)
    print(
        f"Preflight ok={len(pf_problems_w) == 0} (cpw_in_slab={(LAM0 / N_SLAB) / dz_w:.1f}, "
        f"cells_across_slab={d_slab_w / dz_w:.1f}, est_ram={est_w.host_bytes / 1e6:.1f} MB)"
    )
    print(
        f"Two-run FluxMonitor: RMS(R - R_exact) = {np.sqrt(np.mean((R_w - R_ex_w)**2)):.3e}, "
        f"max |R + T - 1| = {np.max(np.abs(R_w + T_w - 1.0)):.2e}"
    )

# %% [markdown]
# ### Step 4 — Interfaces off the grid lines: `averaging='kottke'` vs `'staircase'`
#
# When a dielectric boundary does not coincide with a grid plane, staircase
# assignment (`averaging='staircase'`) snaps each boundary cell to either vacuum
# or dielectric according to whether its fill fraction $w$ exceeds $0.5$. This
# changes the physical slab thickness by up to $\pm\Delta z$, shifting the
# Fabry–Pérot fringes by $\mathcal{O}(\Delta z^1)$. Passing `averaging='kottke'`
# to `assign_materials` applies Kottke's subpixel smoothing (arithmetic mean
# $\langle\varepsilon\rangle = w\varepsilon_{\text{slab}} + (1-w)\varepsilon_0$
# for tangential $E_x, E_y$ and harmonic mean for normal $E_z$), preserving the
# exact optical thickness of the cut cells.

# %%
if _RUN_WALKTHROUGH:
    dz_sub = (LAM0 / N_SLAB) / 14  # ~35.71 nm cell size
    d_off = 0.618 * (LAM0 / N_SLAB)  # 309.0 nm (8.652 cells -> cuts both boundary cells)
    z_lo_off = 1.137 * LAM0
    z_hi_off = z_lo_off + d_off
    g_sub = GridFIT3D(
        x=np.array([0.0, dz_sub]),
        y=np.array([0.0, dz_sub]),
        z=np.arange(80) * dz_sub,
        verbose=0,
    )
    for avg_mode in ("staircase", "kottke"):
        s_sub = SolverFIT3D(g_sub, use_stl=False, verbose=0)
        assign_materials(s_sub, [PlanarSlabZ(z_lo_off, z_hi_off, EPS_SLAB)], F0, averaging=avg_mode)
        eps_x_arr = 1.0 / (8.8541878128e-12 * np.asarray(s_sub.ieps[0, 0, :-1, "x"], dtype=float))
        # Optical excess integral \int (\eps_x(z) - 1) dz compared with exact (\eps_slab - 1) * d_off
        opt_int_sim = float(np.sum(eps_x_arr - 1.0) * dz_sub)
        opt_int_exact = (EPS_SLAB - 1.0) * d_off
        rel_thick_err = abs(opt_int_sim - opt_int_exact) / opt_int_exact
        print(
            f"averaging={avg_mode:9s}: integrated (eps_xx - 1)*dz = {opt_int_sim*1e9:7.2f} nm "
            f"(exact {opt_int_exact*1e9:7.2f} nm, rel. error = {rel_thick_err:.3e})"
        )

# %% [markdown]
# ## 3. Multi-task benchmark and resolution study runner
#
# The script defines a deterministic task table covering:
#
# * `task_0` (`fabry_perot_and_oblique`): broadband Fabry–Pérot etalon
#   reflectance and transmittance ($d = 1.25\,\lambda_0/n_{\text{slab}}$) via
#   two-run `FluxMonitor` and `DFTMonitor` normalization, space-time waterfall,
#   and an oblique Bloch-periodic incidence sweep ($\theta \in [0^\circ, 72^\circ]$)
#   for both $s$/TE and $p$/TM polarizations including Brewster's angle
#   $\theta_B = 63.435^\circ$.
# * `task_1` (`cpml_reflection`): quantitative benchmark of spurious boundary
#   reflection $10\log_{10}|r(f)|^2\;\text{[dB]}$ for `cpml` across
#   $N_{\text{pml}} \in \{4, 6, 8, 10, 12, 16\}$ and grading exponents
#   `pml_exp` $\in \{2, 3, 4\}$, compared directly against a `pec` wall and Mur
#   first-order `abc`.
# * `task_2..13` (`kottke_cpw*` and `staircase_cpw*`): off-grid dielectric slab
#   ($d = 0.618\,\lambda_0/n_{\text{slab}} = 309\;\text{nm}$, $z_{\text{lo}} = 1.137\,\lambda_0$)
#   comparing Kottke subpixel smoothing against staircase rounding across
#   $\text{cpw}_{\text{slab}} \in \{10, 14, 20\}$ (`--quick`) and
#   $\{10, 14, 20, 28, 40, 60\}$ (`--full`).

# %%
def get_task_table(quick: bool = False) -> list[dict]:
    """Return the task table for ``--quick`` (8 tasks) or ``--full`` (14 tasks)."""
    fp_cpw = 20 if quick else 28
    pml_cpw = 20 if quick else 24
    cpw_levels = [
        ("coarse", 10),
        ("medium_coarse", 14),
        ("medium", 20),
    ]
    if not quick:
        cpw_levels.extend(
            [
                ("fine", 28),
                ("very_fine", 40),
                ("reference", 60),
            ]
        )

    tasks = [
        {
            "index": 0,
            "name": "fabry_perot_and_oblique",
            "kind": "fabry_perot",
            "cpw": fp_cpw,
            "quick": quick,
        },
        {
            "index": 1,
            "name": "cpml_reflection",
            "kind": "cpml",
            "cpw": pml_cpw,
            "quick": quick,
        },
    ]
    idx = 2
    for level_name, cpw_val in cpw_levels:
        for avg_mode in ("kottke", "staircase"):
            tasks.append(
                {
                    "index": idx,
                    "name": f"{avg_mode}_cpw{cpw_val}",
                    "kind": "subpixel",
                    "level": level_name,
                    "cpw": cpw_val,
                    "averaging": avg_mode,
                    "quick": quick,
                }
            )
            idx += 1
    return tasks


def run_single_task(cfg: dict, data_dir: Path = DATA_DIR) -> dict:
    """Execute one task from ``get_task_table`` and write ``task_<idx>.{json,npz}``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if cfg["kind"] == "fabry_perot":
        res = run_fabry_perot_and_oblique_task(cfg)
    elif cfg["kind"] == "cpml":
        res = run_cpml_reflection_task(cfg)
    elif cfg["kind"] == "subpixel":
        res = run_subpixel_convergence_task(cfg)
    else:
        raise ValueError(f"Unknown task kind {cfg['kind']!r}.")
    wall = time.perf_counter() - t0

    idx = int(cfg["index"])
    npz_payload = {
        k: (np.asarray(v) if not isinstance(v, np.ndarray) else v)
        for k, v in res.items()
    }
    npz_payload["wall_time_s"] = np.asarray(wall, dtype=float)
    np.savez_compressed(data_dir / f"task_{idx}.npz", **npz_payload)

    meta = {
        "index": idx,
        "name": str(res["name"]),
        "kind": str(cfg["kind"]),
        "cpw": int(res["cpw"]),
        "averaging": str(res["averaging"]),
        "dx_background_nm": float(res["dx_background_nm"]),
        "dx_min_nm": float(res["dx_min_nm"]),
        "min_feature_nm": float(res["min_feature_nm"]),
        "cells_across_feature": float(res["cells_across_feature"]),
        "n_cells": int(res["n_cells"]),
        "n_steps": int(res["n_steps"]),
        "wall_time_s": float(wall),
    }
    if cfg["kind"] == "fabry_perot":
        meta["fp_rms_err"] = float(res["fp_rms_err"])
        meta["fp_energy_err"] = float(res["fp_energy_err"])
        meta["brewster_Rp_sim"] = float(res["brewster_Rp_sim"])
        meta["brewster_Tp_sim"] = float(res["brewster_Tp_sim"])
        meta["oblique_max_energy_err"] = float(res["oblique_max_energy_err"])
    elif cfg["kind"] == "cpml":
        meta["r_db_pec_f0"] = float(res["r_db_pec"][len(res["freqs"]) // 2])
        meta["r_db_abc_f0"] = float(res["r_db_abc"][len(res["freqs"]) // 2])
        meta["r_db_cpml10_f0"] = float(res["r_db_vs_npml_f0"][list(res["n_pml_list"]).index(10)])
        meta["r_db_cpml16_f0"] = float(res["r_db_vs_npml_f0"][list(res["n_pml_list"]).index(16)])
    elif cfg["kind"] == "subpixel":
        meta["level"] = str(res["level"])
        meta["rms_err"] = float(res["rms_err"])
        meta["max_err"] = float(res["max_err"])
        meta["energy_err"] = float(res["energy_err"])

    with open(data_dir / f"task_{idx}.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def generate_all_figures(data_dir: Path = DATA_DIR, fig_dir: Path = FIG_DIR) -> list[Path]:
    """Render all 6 tutorial figures from saved ``data_dir/task_*.npz`` files."""
    return render_all_figures(data_dir=data_dir, fig_dir=fig_dir)


# %% [markdown]
# ## 4. Results & validation against exact Fresnel and Airy theory
#
# Running `slab_rt.py --full` executes all 14 tasks and renders the six figures
# in `figures/`:
#
# ![Two-run normalization setup and reflected echo subtraction](figures/01_two_run_normalization_and_setup.png)
#
# ![Fabry-Perot Airy fringes and oblique Bloch-periodic Fresnel sweep](figures/02_fabry_perot_and_oblique_fresnel.png)
#
# ![CPML reflection benchmark vs PEC wall and Mur ABC](figures/03_cpml_reflection_benchmark.png)
#
# ![Off-grid permittivity profiles and Fabry-Perot spectra](figures/04_subpixel_permittivity_and_spectra.png)
#
# | Observable / benchmark | Wakis simulation (`--full`) | Exact analytical result | Error / residual |
# | :--- | :---: | :---: | :---: |
# | Fabry–Pérot broadband $R(f)$ ($d = 1.25\,\lambda_0/n_{\text{slab}}$, $28\;\text{cpw}$) | `FluxMonitor` two-run | Airy formula $R(f)$ | $\text{RMS}(R_{\text{sim}} - R_{\text{exact}}) = 5.39 \times 10^{-3}$ |
# | Normal-incidence energy balance $\max_f |R(f) + T(f) - 1|$ | $1.000000$ | $1.000000$ | $2.52 \times 10^{-7}$ |
# | Brewster's angle ($\theta_B = 63.435^\circ$) $p$/TM reflectance $R_p(\theta_B)$ | $6.36 \times 10^{-4}$ | $0.0000$ | $6.36 \times 10^{-4}$ |
# | Brewster's angle ($\theta_B = 63.435^\circ$) $p$/TM transmittance $T_p(\theta_B)$ | $0.99963$ | $1.0000$ | $3.73 \times 10^{-4}$ |
# | Brewster's angle ($\theta_B = 63.435^\circ$) $s$/TE reflectance $R_s(\theta_B)$ | $0.7209$ | $0.7220$ | $1.1 \times 10^{-3}$ |
# | Boundary reflection at $f_0$: `pec` wall | $0.00\;\text{dB}$ | $0.00\;\text{dB}$ ($|r|^2 = 1$) | $< 10^{-15}$ |
# | Boundary reflection at $f_0$: Mur 1st-order `abc` ($24\;\text{cpw}$) | $-47.91\;\text{dB}$ | $\mathcal{O}((v_\phi - c_0)/c_0)$ | Dispersive mismatch floor |
# | Boundary reflection at $f_0$: `cpml` ($N_{\text{pml}} = 4, 6, 8, 10, 12, 16$, $m=3$) | $\{-39.95, -51.65, -62.25, -70.89, -78.08, -87.24\}\;\text{dB}$ | $0\;\text{ as }N_{\text{pml}}\to\infty$ | Monotonic exponential drop |
# | Boundary reflection at $f_0$: `cpml` ($N_{\text{pml}} = 10$, `pml_exp` $\in \{2, 3, 4\}$) | $\{-48.07, -70.89, -91.59\}\;\text{dB}$ | Smooth polynomial grading | $43.5\;\text{dB}$ gain from $m=2 \to 4$ |
#
# ## 5. Resolution study — Kottke subpixel smoothing vs. staircasing
#
# ![Subpixel convergence study: Kottke O(dz^2) vs Staircase O(dz^1)](figures/05_subpixel_convergence_study.png)
#
# ![Animated Fabry-Perot slab reflections](figures/06_slab_fabry_perot_etalon.gif)
#
# For the off-grid slab ($d = 0.737\,\lambda_0/n_{\text{slab}} = 368.5\;\text{nm}$,
# $z_{\text{lo}} = 1.137\,\lambda_0$), all 6 resolution levels satisfy the
# project grid rules ($\text{cpw}_{\text{slab}} \ge 10$ in $n_{\text{slab}} = 2.0$
# and $d/\Delta z \ge 7.37 \ge 4$ cells across the slab):
#
# | Level | $\text{cpw}_{\text{slab}}$ ($n=2$) | $\Delta z_{\text{bg}}$ [nm] | $\Delta z_{\min}$ [nm] | Cells across slab $d/\Delta z$ | Staircase RMS error | Kottke RMS error | Kottke improvement factor | Max $|R+T-1|$ (Kottke) |
# | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | Coarse (`task_2,3`) | $10$ | $100.00$ | $50.00$ | $7.37$ | $1.25 \times 10^{-1}$ | $4.01 \times 10^{-2}$ | $3.1\times$ | $7.3 \times 10^{-8}$ |
# | Medium-coarse (`task_4,5`) | $14$ | $71.43$ | $35.71$ | $10.32$ | $9.39 \times 10^{-2}$ | $2.08 \times 10^{-2}$ | $4.5\times$ | $6.8 \times 10^{-8}$ |
# | **Medium (`task_6,7`, 20 cpw rule)** | **$20$** | **$50.00$** | **$25.00$** | **$14.74$** | **$2.75 \times 10^{-2}$** | **$6.17 \times 10^{-3}$** | **$4.5\times$** | **$6.8 \times 10^{-8}$** |
# | Fine (`task_8,9`) | $28$ | $35.71$ | $17.86$ | $20.64$ | $2.48 \times 10^{-2}$ | $3.67 \times 10^{-3}$ | $6.8\times$ | $3.1 \times 10^{-8}$ |
# | Very fine (`task_10,11`) | $40$ | $25.00$ | $12.50$ | $29.48$ | $2.36 \times 10^{-2}$ | $1.74 \times 10^{-3}$ | $13.5\times$ | $1.5 \times 10^{-9}$ |
# | Reference (`task_12,13`) | $60$ | $16.67$ | $8.33$ | $44.22$ | $5.83 \times 10^{-3}$ | $6.11 \times 10^{-4}$ | $9.5\times$ | $9.2 \times 10^{-9}$ |
#
# Across $\text{cpw}_{\text{slab}} \in [10, 60]$, `averaging='kottke'` converges
# monotonically with empirical log-log slope **$-2.32 \approx -2$**
# ($\mathcal{O}(\Delta z^2)$), whereas `averaging='staircase'` converges only at
# first order (**slope $-1.58$**, $\mathcal{O}(\Delta z^1)$) with plateauing jumps
# as the slab faces cross grid cell midpoints. At $40\text{–}60\;\text{cpw}$,
# Kottke smoothing reduces the broadband reflectance error by $9.5\times\text{–}13.5\times$
# over staircasing, while discrete energy conservation $|R(f) + T(f) - 1|$ holds
# to $< 10^{-7}$ across every resolution.
#
# ## 6. Limits & failure modes
#
# 1. **Too-thin CPML or abrupt conductivity grading (`n_pml < 6` or `pml_exp = 1`)**:
#    At $N_{\text{pml}} = 4$, discrete impedance mismatch inside the PML leaves
#    a $-40\;\text{dB}$ ($1\%$ amplitude) reflection that pollutes weak
#    backscattering measurements and high-finesse Fabry–Pérot fringes. Using
#    $N_{\text{pml}} \in [8, 12]$ with `pml_exp = 3` suppresses normal-incidence
#    reflection below $-62\text{ to }-78\;\text{dB}$.
# 2. **Grazing-incidence and evanescent waves in PML**:
#    As the incidence angle $\theta_1 \to 90^\circ$, the normal wavevector
#    $k_z = k_0\cos\theta_1 \to 0$ and the effective optical depth of the PML
#    shrinks as $\cos\theta_1$. Furthermore, in Bloch-periodic oblique runs at
#    fixed $k_x = k_0\sin\theta_1$, frequencies below the cutoff
#    $f_{\text{cut}} = c_0 k_x / (2\pi) = f_0\sin\theta_1$ become evanescent in
#    vacuum and have vanishing group velocity $v_{g,z} \to 0$ at $f = f_{\text{cut}}$;
#    always choose the pulse bandwidth `fwidth` so the excited spectrum stays
#    strictly above $f_{\text{cut}}$.
# 3. **Truncating the time integration before Fabry–Pérot ring-down completes**:
#    A high-index or multi-layer resonator traps energy across many internal
#    round trips $\tau_{\text{rt}} = 2 n_{\text{slab}} d / c_0$. Stopping the
#    time loop before the trapped field has decayed causes artificial ripples in
#    the DFT spectrum and violates $R(f) + T(f) = 1$. Always monitor
#    $\max_f |R(f) + T(f) - 1|$ (for lossless media) as a self-consistency check.
# 4. **When Kottke subpixel smoothing does not achieve $\mathcal{O}(\Delta z^2)$**:
#    Kottke's formula assumes a locally smooth, non-dispersive dielectric
#    interface with well-defined normal $\hat{\mathbf{n}}$. Near sharp dielectric
#    corners/wedges (where the field is singular, $\mathbf{E} \sim r^{\nu-1}$)
#    or on dispersive metallic/Drude boundaries (Tutorial 1.3), standard
#    dielectric harmonic/arithmetic averaging does not restore full second-order
#    convergence.
#
# ## 7. Exercises
#
# 1. **Single-layer anti-reflection (AR) coating**: Design a quarter-wave coating
#    of refractive index $n_{\text{ar}} = \sqrt{n_1 n_3}$ and thickness
#    $d_{\text{ar}} = \lambda_0 / (4 n_{\text{ar}})$ on top of a semi-infinite
#    substrate $n_3 = 2.25$ by stacking two `PlanarSlabZ` objects, and verify
#    that $R(f_0) < 10^{-4}$ (previewing Tutorial 2.1).
# 2. **Lossy dielectric slab ($R + T + A = 1$)**: Add a small conductivity or
#    complex permittivity $\varepsilon_r = 4.0 - 0.2i$ in `slab_rt_oblique` and
#    compute the absorbed power fraction $A(f) = 1 - R(f) - T(f)$ (previewing
#    Tutorial 1.3).
# 3. **Total internal reflection (TIR)**: Using `fresnel_single_interface(2.0, 1.0, theta, 's')`,
#    sweep $\theta$ past the critical angle $\theta_c = \arcsin(1/2) = 30^\circ$
#    and verify that $|r| = 1$ ($R = 1, T = 0$) while the phase $\arg(r(\theta))$
#    rotates continuously (the Goos–Hänchen phase shift, Tutorial 2.3).
#
# ## Verified references
#
# 1. M. Born and E. Wolf, *Principles of Optics*, 7th ed., Cambridge University
#    Press, Cambridge (1999), §1.5 (Fresnel equations) and §7.6 (Fabry–Pérot
#    interferometer and Airy formulas).
# 2. K. S. Yee, "Numerical solution of initial boundary value problems involving
#    Maxwell's equations in isotropic media," *IEEE Trans. Antennas Propag.*
#    **14**(3), 302–307 (1966), doi:[10.1109/TAP.1966.1138693](https://doi.org/10.1109/TAP.1966.1138693).
# 3. T. Weiland, "A discretization method for the solution of Maxwell's equations
#    for six-component fields," *AEÜ — Int. J. Electron. Commun.* **31**(3),
#    116–120 (1977); see also T. Weiland, "Time domain electromagnetic field
#    computation with finite difference methods," *Int. J. Numer. Model.:
#    Electron. Netw. Devices Fields* **9**(4), 295–319 (1996),
#    doi:[10.1002/(SICI)1099-1204(199607)9:4<295::AID-JNM240>3.0.CO;2-8](https://doi.org/10.1002/(SICI)1099-1204(199607)9:4%3C295::AID-JNM240%3E3.0.CO;2-8),
#    and M. Clemens and T. Weiland, "Discrete electromagnetism with the Finite
#    Integration Technique," *Prog. Electromagn. Res.* **32**, 65–87 (2001),
#    doi:[10.2528/PIER00080103](https://doi.org/10.2528/PIER00080103).
# 4. A. F. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J. D. Joannopoulos, and
#    S. G. Johnson, "Meep: A flexible free-software package for electromagnetic
#    simulations by the FDTD method," *Comput. Phys. Commun.* **181**(3),
#    687–702 (2010), doi:[10.1016/j.cpc.2009.11.008](https://doi.org/10.1016/j.cpc.2009.11.008).
# 5. A. Oskooi and S. G. Johnson, "Electromagnetic wave source conditions," in
#    *Advances in FDTD Computational Electrodynamics: Photonics and
#    Nanotechnology*, A. Taflove, A. Oskooi, and S. G. Johnson (eds.), ch. 4,
#    pp. 65–100, Artech House, Boston (2013); preprint
#    [arXiv:1301.5366](https://arxiv.org/abs/1301.5366).
# 6. J.-P. Berenger, "A perfectly matched layer for the absorption of
#    electromagnetic waves," *J. Comput. Phys.* **114**(2), 185–200 (1994),
#    doi:[10.1006/jcph.1994.1159](https://doi.org/10.1006/jcph.1994.1159).
# 7. J. A. Roden and S. D. Gedney, "Convolution PML (CPML): An efficient FDTD
#    implementation of the CFS–PML for arbitrary media," *Microw. Opt. Technol.
#    Lett.* **27**(5), 334–339 (2000), doi:[10.1002/1098-2760(20001205)27:5<334::AID-MOP14>3.0.CO;2-A](https://doi.org/10.1002/1098-2760(20001205)27:5%3C334::AID-MOP14%3E3.0.CO;2-A).
# 8. A. Kottke, A. Farjadpour, and S. G. Johnson, "Perturbation theory for
#    anisotropic dielectric interfaces, and application to subpixel smoothing of
#    discretized numerical methods," *Phys. Rev. E* **77**(3), 036611 (2008),
#    doi:[10.1103/PhysRevE.77.036611](https://doi.org/10.1103/PhysRevE.77.036611).
# 9. A. Taflove and S. C. Hagness, *Computational Electrodynamics: The
#    Finite-Difference Time-Domain Method*, 3rd ed., Artech House, Norwood, MA
#    (2005), ch. 7 (PML absorbing boundaries).
# 10. P. Yeh, A. Yariv, and C.-S. Hong, "Electromagnetic propagation in periodic
#     stratified media. I. General theory," *J. Opt. Soc. Am.* **67**(4),
#     423–438 (1977), doi:[10.1364/JOSA.67.000423](https://doi.org/10.1364/JOSA.67.000423).
# 11. G. Mur, "Absorbing boundary conditions for the finite-difference
#     approximation of the time-domain electromagnetic-field equations," *IEEE
#     Trans. Electromagn. Compat.* **EMC-23**(4), 377–382 (1981),
#     doi:[10.1109/TEMC.1981.303970](https://doi.org/10.1109/TEMC.1981.303970).

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tutorial 1.2: Reflection and transmission at a dielectric slab (Fresnel, Fabry-Perot, CPML, Kottke subpixel)."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Run the fast 8-task configuration (coarse-to-medium resolutions) and render figures.",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Run the full 14-task configuration up to 60 cpw and render all figures.",
    )
    parser.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run only a single task index (0..N-1) and write data/task_<idx>.{json,npz}.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render figures/ from existing data/task_*.npz without re-running simulations.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    quick = bool(args.quick)
    if not args.quick and not args.full and not args.plot and args.task is None:
        quick = True

    if args.plot and args.task is None and not args.quick and not args.full:
        written = generate_all_figures(DATA_DIR, FIG_DIR)
        print(f"Rendered {len(written)} figures in {FIG_DIR}")
        return 0

    tasks = get_task_table(quick=quick)
    if args.task is not None:
        if args.task < 0 or args.task >= len(tasks):
            raise ValueError(f"--task must be in [0, {len(tasks) - 1}], got {args.task}.")
        meta = run_single_task(tasks[args.task], data_dir=DATA_DIR)
        print(json.dumps(meta, indent=2))
        return 0

    for cfg in tasks:
        meta = run_single_task(cfg, data_dir=DATA_DIR)
        print(
            f"[Task {meta['index']:02d}/{len(tasks)-1:02d}] {meta['name']:24s} "
            f"cpw={meta['cpw']:2d} wall={meta['wall_time_s']:.2f}s"
        )

    written = generate_all_figures(DATA_DIR, FIG_DIR)
    print(f"Rendered {len(written)} figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
