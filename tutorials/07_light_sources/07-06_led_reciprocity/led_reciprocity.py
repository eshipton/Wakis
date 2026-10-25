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
# # Tutorial 7.6 · Thin-film LED: emission computed as absorption (Helmholtz reciprocity)
#
# **Learning goals**
# 1. **Turn an emission problem into an absorption problem**: Understand how **Helmholtz reciprocity** lets us treat a spontaneous-emission light-emitting diode (LED) as an **absorber** illuminated by incoming plane waves from the observation directions, replacing $3 N_{\text{sites}}$ direct dipole solves with one Bloch unit-cell plane-wave solve per angle and polarization (Janssen, Wachters & Urbach, *Opt. Express* **18**, 24522, 2010).
# 2. **Validate the exact 2D reciprocity identity**: Verify point-by-point ($0.46\text{--}1.68\%$ matrix L2) and ensemble ($0.066\%$ L2 at $\mathrm{cpw}=20$; $0.083\%$ at $\mathrm{cpw}=12$) agreement between direct incoherent dipole summation (`CurrentSource` + `Near2FarMonitor2D`) and reciprocal plane-wave absorption (`make_huygens_planewave_2d` + `DFTMonitor`), as well as analytical Transfer-Matrix Method (TMM) benchmarks on PEC and silver (Ag) mirrors.
# 3. **Optimize the absorption in the emitting region**: Reproduce the 2D blue GaN LED ($\lambda_0 = 450\;\text{nm}$, $n_{\text{GaN}} = 2.45$, silver contact from `wakis.materials.dispersive_lib["silver jc"]` and PEC check) optimization of Janssen *et al.* (2010):
#    - **(a) Shallow-grating guided-mode resonances** ($\Lambda = 250\;\text{nm}, d = 5\;\text{nm}$, their Fig. 2): sharp Fano peaks rising far above the flat slab background with $Q \sim 10^5$ on PEC vs $Q \sim 10^3$ on Ag.
#    - **(b) $(t, d)$ thickness–depth landscape and steepest-ascent optimization** ($\Lambda = 450\;\text{nm}$ block grating, their Fig. 4): multiple local maxima and gradient-ascent trajectories, with gradients verified **both ways** via the Wakis discrete adjoint (`wakis.inverse.AdjointSimulation`) and central finite differences.
#    - **(c) Optimal active-layer position** (their Fig. 5): standing-wave antinode peaking near $y_0 \approx 120\;\text{nm}$ above the Ag mirror.
#    - **(d) Grating pitch scan and $\approx 2\times$ extraction enhancement** (their Fig. 6): dispersive Bragg diffraction branches peaking near $\Lambda/\lambda_0 \approx 0.8$ ($\Lambda \approx 360\;\text{nm}$), where the optimized grating achieves $\approx 2\times$ the extraction of the optimized flat LED.
#
# ### Simulation Setup & Reciprocity Schematic
#
# The setup figures (`figures/00a_setup_geometry.png`, `figures/00b_setup_mesh_and_excitation.png`,
# `figures/00a_setup_geometry_reciprocal.png`, and `figures/00b_setup_mesh_and_excitation_reciprocal.png`,
# alongside `figures/00_emitter_absorber_schematic.png`), rendered directly from the live Wakis `SolverFIT3D`,
# `GridFIT3D`, `PolySlab`, `CurrentSource`, `make_huygens_planewave_2d`, `DFTMonitor`, and `Near2FarMonitor2D`
# objects via `wakis.schematic.save_setup_figures`, show the complete forward and reciprocal simulation setups
# before any solver runs:
# - **Forward Setup (Direct Problem — LED as an Incoherent Emitter, $3 N_{\text{sites}} = 75$ Runs):** Uncorrelated point
#   dipoles $(J_x, J_y, J_z)$ in the active quantum-well plane $y_0 = 49\;\text{nm}$ (5 representative sites shown) inside
#   the high-index GaN slab ($n = 2.45$, $d_{\text{slab}} = 225\;\text{nm}$) etched with a 7-period trapezoidal surface
#   grating ($\Lambda = 270\;\text{nm}$, $h_g = 72\;\text{nm}$) above a Drude Ag (`silver jc`, $\delta_s = 26.9\;\text{nm}$)
#   or PEC back-reflector ($y \le 0$), monitored in the air cladding by a `Near2FarMonitor2D` line at $y_{\text{mon}} = 378\;\text{nm}$.
# - **Reciprocal Setup (Reciprocal Problem — LED as an Absorber, 1 Run / Angle):** A downward Huygens plane wave
#   (`make_huygens_planewave_2d`) arriving from observation angle $-\theta$ (or a 1-period Bloch unit cell $x \in [0, \Lambda]$)
#   excites the internal standing wave recorded by an active-layer `DFTMonitor` at $y = y_0 = 49\;\text{nm}$, with vertical
#   mesh grading (`graded_axis`) providing $\Delta y_{\min} = 2.58\;\text{nm}$ ($10.0\;\text{cells}/\delta_s = 5.0\;\text{cells}/\delta_p$
#   in silver), $8.0$ cells across $h_g$, and $20\;\text{cells}/\lambda_{\text{GaN}}$ in the bulk slab.
#
# ![Setup Geometry (Forward — Direct QW Dipole Ensemble)](figures/00a_setup_geometry.png)
# ![Setup Mesh and Excitation (Forward — Direct QW Dipole Ensemble)](figures/00b_setup_mesh_and_excitation.png)
#
# The table below summarizes the forward (direct incoherent dipole ensemble) thin-film GaN LED simulation setup, boundary conditions, materials, far-field monitor, and task variations.
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 2.70 µm × 742 nm × 9.18 nm (z ∈ [0 m, 9.18 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 294 × 111 × 1 (32634 cells) |
# | Discretization | Δx & resolution | Δ_min = 2.58 nm, 71.3 cells/λ (in n_max = 2.45), 10.4 cells across feature (26.9 nm) |
# | Boundary | -x face | CPML (10 cells, 91.8 nm) |
# | Boundary | +x face | CPML (10 cells, 91.8 nm) |
# | Boundary | -y face | PEC |
# | Boundary | +y face | CPML (10 cells, 90 nm) |
# | Boundary | -z face | Periodic |
# | Boundary | +z face | Periodic |
# | Material | Background | Air cladding (n = 1.00), ε = 1.00 (isotropic) |
# | Material | Region 1: Drude Ag back-reflector (δ_s = 26.9 nm) | ε_r = 3.70, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-945 nm, 945 nm] (d = 1.89 µm) |
# | Material | Region 2: GaN unetched base slab (n = 2.45, y_base = 153 nm) | ε_r = 6.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-945 nm, 945 nm] (d = 1.89 µm) |
# | Material | Region 3: GaN 7-period trapezoidal grating (n = 2.45, Λ = 270 nm, h_g = 72 nm) | ε_r = 6.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-945 nm, 945 nm] (d = 1.89 µm) |
# | Source | 1 · CurrentSource | pos: y_0 = 49.0 nm, N_sites = 25 × 3 orientations = 75 dipoles in x ∈ [-0.81, +0.81] µm (5 shown), pol: J_x, J_y, J_z, pulse: f₀ = 666.2 THz, Δf = 133.2 THz |
# | Monitor | 2 · Near2FarMonitor2D | Cylindrical far-field line Near2FarMonitor2D (y_mon = 378.0 nm; projects U_2D(θ) into air), pos: y_mon = 378.0 nm, x ∈ [-1.08, +1.08] µm, freqs: f = 666.2 THz |
# | Variation | Task 1 — Direct vs. reciprocal ensemble (`direct_vs_reciprocity_ensemble`) | notes=N_pos = 25 dipole sites × 3 orientations (J_x, J_y, J_z) = 75 forward solves vs. 29 reciprocal angles (cpw = 20, L2 diff = 0.066%) |
# | Variation | Tasks 9–10 — Drude Ag skin-depth resolution ladder (`ag_grating_resolution_study`) | notes=Graded Ag collar 4, 6, 8, 10, 12 cells/δ_s (accepted Δ_min = 2.69 nm, 10.0 cells/δ_s) vs. under-resolved demo (1.6, 2.6 cells/δ_s) |
#
# ![Setup Geometry (Reciprocal — Downward Huygens Plane Wave)](figures/00a_setup_geometry_reciprocal.png)
# ![Setup Mesh and Excitation (Reciprocal — Downward Huygens Plane Wave)](figures/00b_setup_mesh_and_excitation_reciprocal.png)
#
# The table below summarizes the reciprocal (downward Huygens plane-wave absorber) simulation setup, active-layer DFT monitor, and unit-cell/grating variations.
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 2.70 µm × 742 nm × 9.18 nm (z ∈ [0 m, 9.18 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 294 × 111 × 1 (32634 cells) |
# | Discretization | Δx & resolution | Δ_min = 2.58 nm, 71.3 cells/λ (in n_max = 2.45), 10.4 cells across feature (26.9 nm) |
# | Boundary | -x face | CPML (10 cells, 91.8 nm) |
# | Boundary | +x face | CPML (10 cells, 91.8 nm) |
# | Boundary | -y face | PEC |
# | Boundary | +y face | CPML (10 cells, 90 nm) |
# | Boundary | -z face | Periodic |
# | Boundary | +z face | Periodic |
# | Material | Background | Air cladding (n = 1.00), ε = 1.00 (isotropic) |
# | Material | Region 1: Drude Ag back-reflector (δ_s = 26.9 nm) | ε_r = 3.70, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-945 nm, 945 nm] (d = 1.89 µm) |
# | Material | Region 2: GaN unetched base slab (n = 2.45, y_base = 153 nm) | ε_r = 6.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-945 nm, 945 nm] (d = 1.89 µm) |
# | Material | Region 3: GaN 7-period trapezoidal grating (n = 2.45, Λ = 270 nm, h_g = 72 nm) | ε_r = 6.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-945 nm, 945 nm] (d = 1.89 µm) |
# | Source | 1 · CurrentSource | pos: y = 378.0 nm, x ∈ [-1.08, +1.08] µm (θ ∈ [-70°, +70°]), pol: s (TM) + p (TE), pulse: f₀ = 666.2 THz, Δf = 133.2 THz |
# | Monitor | 2 · DFTMonitor | Active-QW DFTMonitor (y_0 = 49.0 nm; integrates |E_rec(x, y_0; θ) / E_inc|² across 7 grating periods), pos: y_0 = 49.0 nm, x ∈ [-0.94, +0.94] µm (records E_x, E_y, E_z), freqs: f = 666.2 THz |
# | Variation | Task 0 — 1D Bloch flat-slab reciprocity vs. TMM (`flat_slab_bloch_tmm`) | notes=Bloch unit cell on PEC & Drude Ag mirrors (cpw ∈ {10, 14, 20, 28, 40}, cpsd ∈ [1.6, 10.0]) |
# | Variation | Tasks 2–8 — Kottke subpixel vs. snapped staircase convergence (`kottke_*`, `staircase_*`) | notes=7-period GaN grating reciprocal angular sweep cpw ∈ {14, 20, 28} vs. cpw = 40 reference |
# | Variation | Section 5 — Janssen et al. (2010) unit-cell optimization & discrete adjoint | notes=2D Bloch FIT + `ResonanceMonitor.harminv` + `AdjointSimulation` shape gradients on Ag (`silver jc`, ~2× extraction) |
#
# ---
#
# ## 1. Physics: Treating the LED as an Absorber via Helmholtz Reciprocity
#
# ### 1.1 The Emitter $\to$ Absorber Picture (Janssen et al. 2010, Eqs. 1, 7–10, 14)
#
# Light-emitting diodes generate light by **spontaneous emission** from electron–hole recombination across an extended quantum-well (QW) active region $\mathcal{O}$ (in a blue InGaN/GaN LED at $\lambda_0 = 450\;\text{nm}$, refractive index $n_{\text{GaN}} = 2.45$, located $y_0 \approx 100\text{--}120\;\text{nm}$ above a metallic silver back-reflector). Because dipoles at different positions $\mathbf{r}_0 \in \mathcal{O}$ and orientations $\hat{\mathbf{p}}$ emit with **uncorrelated random phases**, a single coherent sheet source cannot be used: one must average the emitted **power** over all dipole positions and orientations.
#
# Rather than running thousands of forward simulations for individual point dipoles inside $\mathcal{O}$, the method of Janssen, Wachters & Urbach (*Opt. Express* **18**, 24522, 2010) **treats the LED as an absorber instead of an emitter**:
# - A plane wave is launched from the far-field observation direction $(\theta, \varphi)$ toward the LED and is **absorbed** (builds up local energy density $|\mathbf{E}|^2$) inside the emitting quantum-well region $\mathcal{O}$.
# - By **Helmholtz reciprocity** (the optical form of the Lorentz reciprocity theorem $\int (\mathbf{E}_1 \cdot \mathbf{J}_2 - \mathbf{H}_1 \cdot \mathbf{M}_2)\,dV = \int (\mathbf{E}_2 \cdot \mathbf{J}_1 - \mathbf{H}_2 \cdot \mathbf{M}_1)\,dV$, from which Janssen *et al.* derive it in their Eq. 1), the electric field $\mathbf{E}_d(\mathbf{r}_2; \mathbf{r}_1, \mathbf{p})$ radiated to a far-field point $\mathbf{r}_2$ by a point dipole $\mathbf{p}$ at $\mathbf{r}_1 \in \mathcal{O}$ equals the internal electric field $\mathbf{E}_{\text{pw}}(\mathbf{r}_1; \mathbf{k}_{\text{inc}}, \mathbf{e}_{\text{pol}})$ excited at $\mathbf{r}_1$ by a reciprocal plane wave arriving from $\mathbf{r}_2$:
#
# $$\mathbf{p} \cdot \mathbf{E}_{\text{pw}}(\mathbf{r}_1) = \mathbf{p}_{\text{far}} \cdot \mathbf{E}_d(\mathbf{r}_2) \qquad \text{(Janssen et al., Eq. 7)}.$$
#
# Averaging over uncorrelated dipole orientations and integrating over the emitting region $\mathcal{O}$ within one grating period $\Lambda$ converts the incoherent emission integral into an **absorbed field-intensity integral** over $\mathcal{O}$:
#
# - **Isotropic 3D dipole ensemble** ($\langle |\mathbf{p} \cdot \mathbf{E}_{\text{pw}}|^2 \rangle = \tfrac{1}{3}|\mathbf{p}|^2 |\mathbf{E}_{\text{pw}}|^2$, Janssen *et al.* Eq. 8):
#   $$I_{\text{iso}}(\theta, \text{pol}) \propto \frac{1}{3 V_{\mathcal{O}}} \int_{\mathcal{O}} \left(|E_{\text{pw},x}(\mathbf{r}; \theta, \text{pol})|^2 + |E_{\text{pw},y}(\mathbf{r}; \theta, \text{pol})|^2 + |E_{\text{pw},z}(\mathbf{r}; \theta, \text{pol})|^2\right) d\mathbf{r}.$$
# - **In-plane quantum-well dipole ensemble** (dipoles confined to the horizontal $(x, z)$ QW plane parallel to the mirror, $\langle |\mathbf{p}_\parallel \cdot \mathbf{E}_{\text{pw}}|^2 \rangle = \tfrac{1}{2}|\mathbf{p}|^2 (|E_x|^2 + |E_z|^2)$, Janssen *et al.* Eq. 9):
#   $$I_{\text{QW},\parallel}(\theta, \text{pol}) \propto \frac{1}{2 V_{\mathcal{O}}} \int_{\mathcal{O}} \left(|E_{\text{pw},x}(\mathbf{r}; \theta, \text{pol})|^2 + |E_{\text{pw},z}(\mathbf{r}; \theta, \text{pol})|^2\right) d\mathbf{r}.$$
# - **Both polarizations ($s$ and $p$) and extraction-cone figure of merit (Janssen *et al.* Eqs. 10 & 14)**:
#   In 2D ($\partial/\partial z = 0$, plane of incidence $(x, y)$, normalized transverse wavevector $\hat{k}_x = \sin\theta$), the two orthogonal incoming plane-wave polarizations decouple completely:
#   1. **$s$-polarization (TM, out-of-plane electric field $\mathbf{E}_{\text{inc}} = E_{\text{inc}}\hat{\mathbf{z}}$)**: excites only $E_z(x, y; \theta)$, measuring the emission of out-of-plane dipoles $J_z$.
#   2. **$p$-polarization (TE, in-plane electric field $\mathbf{H}_{\text{inc}} = (E_{\text{inc}}/\eta_0)\hat{\mathbf{z}}$)**: excites both $E_x(x, y; \theta)$ and $E_y(x, y; \theta)$ simultaneously, measuring the emission of in-plane horizontal ($J_x$) and vertical ($J_y$) dipoles.
#
#   Summing over both polarizations $\text{pol} \in \{s, p\}$ and integrating over an extraction cone $|\theta| \le \theta_{\max}$ (such as the $60^\circ$ full cone $|\hat{k}_x| \le \sin 30^\circ = 0.5$ or the $90^\circ$ full cone $|\hat{k}_x| \le \sin 45^\circ = 1/\sqrt{2}$) yields the total extracted power (Janssen *et al.* Eqs. 10 & 14):
#
#   $$P_{\text{cone}}(\theta_{\max}) \propto \int_{-\theta_{\max}}^{+\theta_{\max}} \cos^2\theta\,d\theta \sum_{\text{pol} \in \{s, p\}} \frac{1}{\Lambda} \int_{\mathcal{O}} \left|\frac{\mathbf{E}_{\text{pw}}(\mathbf{r}; \theta, \text{pol})}{E_{\text{inc}}}\right|^2 d\mathbf{r}.$$
#
# ### 1.2 Exact 2D Cylindrical Normalization Identity
#
# In a 2D FIT domain (`Nz = 1`, periodic along $z$ with thickness $\Delta z$), a line current source $I_{\text{2D}}(\omega) = S(\omega)/\Delta z$ $[\text{A}\cdot\text{s}]$ at $\mathbf{r}_0 = (x_0, y_0)$ in unbounded free space radiates a cylindrical wave with isotropic TM radiation intensity
#
# $$U_{\text{free,2D}} = \frac{k_0 \eta_0 |I_{\text{2D}}(\omega)|^2}{8\pi} \quad \left[\frac{\text{W}\cdot\text{s}^2}{\text{m}\cdot\text{rad}}\right],$$
#
# where $k_0 = \omega / c = 2\pi / \lambda_0$ and $\eta_0 = \sqrt{\mu_0/\varepsilon_0} \approx 376.73\;\Omega$. Dividing the 2D far-field radiation intensity $U_{\text{2D}}^{(\mathbf{r}_0, \alpha)}(\theta)$ of an $\alpha$-oriented dipole ($\alpha \in \{x, y, z\}$) by $U_{\text{free,2D}}$ gives a dimensionless directional enhancement factor whose proportionality constant with the reciprocal internal field intensity is **exact unity ($1$)**:
#
# $$\tilde{U}_\alpha(\theta; \mathbf{r}_0) \equiv \frac{U_{\text{2D}}^{(\mathbf{r}_0, \alpha)}(\theta)}{U_{\text{free,2D}}} = \left|\frac{E_{\text{rec},\alpha}(\mathbf{r}_0; \theta)}{E_{\text{inc}}}\right|^2, \qquad \alpha \in \{x, y, z\}.$$
#
# Because 2D TM ($E_z, H_x, H_y$) and 2D TE ($E_x, E_y, H_z$) are decoupled when $\partial/\partial z = 0$, **both $s$ (TM) and $p$ (TE) reciprocal plane waves can be launched simultaneously in a single time-domain solver run per angle $\theta$**. Integrating over $\mathcal{O}$ also recovers the effective absorption volume $V_{\text{eff}}^{(i)}(\omega, \Omega) = \int_{\mathcal{O}} d\mathbf{r}\,|\mathbf{E}^{(i)}(\mathbf{r}, \omega, \Omega)|^2 / |\mathbf{E}_{\text{inc}}^{(i)}(\omega, \Omega)|^2$ (Roques-Carmes *et al.*, *Science* **375**, eabm9293, 2022, Eqs. 1–2) and the trace formulation of Yao *et al.* (*Struct. Multidiscip. Optim.* **65**, 336, 2022).
#
# ### 1.3 Analytical Transfer-Matrix Method (TMM) for the Flat Slab
#
# For an untextured high-index dielectric slab ($0 \le y \le d_{\text{slab}}$, permittivity $\varepsilon_f = n_{\text{slab}}^2$) backed at $y = 0$ by a mirror with Fresnel reflection coefficients $r_{fm}^{(s)}, r_{fm}^{(p)}$ (where $r_{fm}^{(s)} = -1, r_{fm}^{(p)} = +1$ for an ideal PEC mirror, or $r_{fm}^{(s)} = (k_{yf} - k_{ym})/(k_{yf} + k_{ym})$ and $r_{fm}^{(p)} = (\varepsilon_m k_{yf} - \varepsilon_f k_{ym})/(\varepsilon_m k_{yf} + \varepsilon_f k_{ym})$ for a metal of complex permittivity $\varepsilon_m$), the reciprocal internal standing wave at height $y_0 \in [0, d_{\text{slab}}]$ has the exact closed form:
#
# $$\tilde{U}_{\text{TM}}(\theta, y_0) = \left|\frac{2 k_{y0}\left(e^{+j k_{yf} y_0} + r_{fm}^{(s)} e^{-j k_{yf} y_0}\right)}{(k_{y0} + k_{yf})e^{+j k_{yf} d_{\text{slab}}} + (k_{y0} - k_{yf})r_{fm}^{(s)}e^{-j k_{yf} d_{\text{slab}}}}\right|^2,$$
#
# $$\tilde{U}_{\text{TE},x}(\theta, y_0) = \left|\frac{2 k_{y0} k_{yf}\left(e^{+j k_{yf} y_0} - r_{fm}^{(p)} e^{-j k_{yf} y_0}\right)}{k_0\left[(\varepsilon_f k_{y0} + k_{yf})e^{+j k_{yf} d_{\text{slab}}} + (\varepsilon_f k_{y0} - k_{yf})r_{fm}^{(p)}e^{-j k_{yf} d_{\text{slab}}}\right]}\right|^2,$$
#
# $$\tilde{U}_{\text{TE},y}(\theta, y_0) = \left|\frac{-2 k_{y0} k_x\left(e^{+j k_{yf} y_0} + r_{fm}^{(p)} e^{-j k_{yf} y_0}\right)}{k_0\left[(\varepsilon_f k_{y0} + k_{yf})e^{+j k_{yf} d_{\text{slab}}} + (\varepsilon_f k_{y0} - k_{yf})r_{fm}^{(p)}e^{-j k_{yf} d_{\text{slab}}}\right]}\right|^2,$$
#
# where $k_x = k_0\sin\theta$, $k_{y0} = k_0\cos\theta$, and $k_{yf} = k_0\sqrt{\varepsilon_f - \sin^2\theta}$.

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon as MplPolygon
from scipy.constants import c as c_light

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from wakis import figstyle, schematic

from _helpers import (  # noqa: E402
    Z0,
    build_grating_led_polyslab,
    build_reciprocity_schematic_objects,
    compute_grating_gradients_both_ways,
    cone_integrated_reciprocity,
    extract_shallow_grating_resonances,
    get_silver_material,
    make_drude_silver,
    make_finite_led_solver,
    overlay_reciprocity_led_geometry,
    rcwa_grating_batch,
    reconstruct_unit_cell_field_map,
    run_finite_led_direct_dipoles,
    run_finite_led_reciprocity,
    run_janssen_optimization_study,
    save_reciprocity_setup_figures,
    solve_bloch_flat_slab,
    solve_bloch_grating_unit_cell,
    tidy_reciprocity_schematic_axes,
    tmm_angle_integrated_power,
    tmm_slab_emission,
)

# %% [markdown]
# ## 2. Physical Parameters & Study Configuration
#
# We model a blue thin-film GaN LED operating at vacuum wavelength $\lambda_0 = 450\;\text{nm}$
# with parameters grounded in two primary references:
# - **Janssen, Wachters & Urbach (2010)**, *Opt. Express* **18**(24), 24522–24535,
#   DOI: `10.1364/OE.18.024522`: blue GaN LED at $\lambda_0 = 450\;\text{nm}$
#   ($n_{\text{GaN}} = 2.45$), backed by a silver (Ag) reflector (`wakis.materials.dispersive_lib["silver jc"]`,
#   $\varepsilon_{\text{Ag}} = -7.08 - 0.22j$, amplitude skin depth $\delta_s = 26.9\;\text{nm}$,
#   power skin depth $\delta_p = 13.5\;\text{nm}$) and an ideal PEC reflector check,
#   GaN thickness $t \approx 700\;\text{nm}$ (swept over $450\text{--}800\;\text{nm}$),
#   active layer $y_0 \approx 100\text{--}120\;\text{nm}$ above the metal, and a 1D surface grating
#   ($\Lambda \in [250, 600]\;\text{nm}$, depth $d \in [0, 350]\;\text{nm}$) yielding an optimized
#   $\approx 2\times$ light-extraction enhancement over the optimized flat structure.
# - **Yao, Verdugo, Christiansen & Johnson (2022)**, *Struct. Multidiscip. Optim.* **65**,
#   336, DOI: `10.1007/s00158-022-03389-5` (verified directly against the open-access
#   full text, Sec. 2.3, Sec. 4.2 & Appendix 8.2): periodic emitting surface LED with
#   period $\Lambda = 0.60\lambda_0 = 270\;\text{nm}$, slab thickness $H_d = 0.50\lambda_0 = 225\;\text{nm}$,
#   and active layer embedded in the lower unetched portion of the high-index film.

# %%
LAM0 = 0.45e-6  # Vacuum wavelength [m] (450 nm blue GaN LED)
F0 = c_light / LAM0  # Optical frequency [Hz] (~666.2 THz)
N_SLAB = 2.45  # GaN refractive index at 450 nm
D_SLAB = 0.50 * LAM0  # Total slab thickness [m] (225 nm = 0.50 * lam0, Yao et al. 2022)
D_FLAT_BENCH = 0.60 * LAM0  # Flat-slab Bloch benchmark thickness [m] (270 nm)
PITCH = 0.60 * LAM0  # Grating period Lambda [m] (270 nm = 0.60 * lam0, Yao et al. 2022)
H_GRAT = 0.16 * LAM0  # Grating etch depth [m] (72 nm)
N_PERIODS = 7  # Number of grating periods in finite 2D LED model
DUTY_BOT = 0.65  # Trapezoidal tooth bottom duty cycle
DUTY_TOP = 0.35  # Trapezoidal tooth top duty cycle (sloped sidewalls for Kottke test)
Y_EMIT = 0.32 * (D_SLAB - H_GRAT)  # Nominal active QW height above mirror [m] (~49 nm)
Y_EMIT_BENCH = 0.16 * LAM0  # Flat-slab Bloch benchmark emitter height [m] (72 nm)


def build_tasks(quick=True):
    """
    Build the list of independent study tasks (`0 .. 10`) for CLI execution.

    Every non-demo task (`demo=False`) satisfies Rules 1-3:
    1. `>= 10` cells per wavelength inside GaN (`n_max = 2.45`, `dx <= 18.37 nm`),
       and `>= 20` cpw for the reference tasks (`task_00` and `task_08`).
    2. `>= 4` cells across the smallest geometric feature (`h_grat = 72.0 nm` and
       tooth top width `w_top = 94.5 nm`, so `dx_slab <= 18.00 nm`).
    3. For the Drude Ag back-reflector (`delta_s = lam0 / (2 pi kappa) = 26.91 nm`,
       power skin depth `delta_p = lam0 / (4 pi kappa) = 13.45 nm`), `graded_axis`
       refines the mesh near `y = 0` to `>= 8.0` cells per `delta_s` (`>= 4.0` cells
       per `delta_p`, `dy_min <= 3.36 nm`), where both the standing-wave error and
       the 2D grating extraction enhancement change by `< 1%`.
    Task 9 (`ag_grating_underresolved_demo`, `demo=True`) is the single labelled
    under-resolved demonstration task (`1.6` and `2.6` cells per `delta_s`) paired
    with the resolved Task 10 (`ag_grating_resolution_study`, `demo=False`).

    Parameters
    ----------
    quick : bool, optional
        If True, use `--quick` resolution (`cpw = 10..18`, reference `cpw = 22`);
        if False (`--full`), use `cpw = 14..28` and reference `cpw = 40`.

    Returns
    -------
    tasks : list of dict
        Task descriptors with `'index'`, `'name'`, `'demo'`, and solver parameters.
    """
    cpw_levels = (10, 14, 18) if quick else (14, 20, 28)
    cpw_ref = 22 if quick else 40
    cpw_ensemble = 12 if quick else 20
    n_angles_half = 16 if quick else 31

    tasks = [
        {
            "index": 0,
            "name": "flat_slab_bloch_tmm",
            "demo": False,
            "mirror": "ag",
            "cpw": 20 if quick else 28,
            "cpw_bloch": 20 if quick else 28,
            "cpsd_target": 8.0 if quick else 10.0,
            "cpw_conv": [10, 14, 20, 28] if quick else [10, 14, 20, 28, 40],
            "cpsd_ag_ladder": [1.6, 2.6, 4.0, 6.0, 8.0, 10.0],
            "n_angles": n_angles_half,
        },
        {
            "index": 1,
            "name": "direct_vs_reciprocity_ensemble",
            "demo": False,
            "mirror": "pec",
            "cpw": cpw_ensemble,
            "n_pos": 11 if quick else 25,
            "n_angles_sym": 15 if quick else 29,
        },
    ]
    idx = 2
    for avg in ("kottke", "staircase"):
        for level_name, cpw in zip(("coarse", "medium", "fine"), cpw_levels):
            tasks.append(
                {
                    "index": idx,
                    "name": f"{avg}_{level_name}",
                    "demo": False,
                    "mirror": "pec",
                    "averaging": avg,
                    "level": level_name,
                    "cpw": cpw,
                    "n_angles_half": n_angles_half,
                    "capture_fields": bool(avg == "kottke" and level_name == "fine"),
                }
            )
            idx += 1

    tasks.append(
        {
            "index": idx,
            "name": "kottke_reference",
            "demo": False,
            "mirror": "pec",
            "averaging": "kottke",
            "level": "reference",
            "cpw": cpw_ref,
            "n_angles_half": n_angles_half,
            "capture_fields": False,
        }
    )
    idx += 1

    # Single clearly-labelled under-resolved demonstration task (1.6 & 2.6 cells per delta_s)
    tasks.append(
        {
            "index": idx,
            "name": "ag_grating_underresolved_demo",
            "demo": True,
            "mirror": "ag",
            "averaging": "kottke",
            "level": "demo_underresolved",
            "cpw": 10,
            "cpsd_target": 1.6,
            "cpsd_levels": [1.6, 2.6],
            "n_periods": 5 if quick else 7,
            "n_angles_half": 8 if quick else 16,
        }
    )
    idx += 1

    # Resolved graded-mesh Drude Ag 2D grating LED study (4, 6, 8 [accepted], 10 cells per delta_s)
    tasks.append(
        {
            "index": idx,
            "name": "ag_grating_resolution_study",
            "demo": False,
            "mirror": "ag",
            "averaging": "kottke",
            "level": "metal_resolved",
            "cpw": 12 if quick else 20,
            "cpsd_target": 8.0 if quick else 10.0,
            "cpsd_levels": [4.0, 6.0, 8.0, 10.0] if quick else [4.0, 6.0, 8.0, 10.0, 12.0],
            "n_periods": 5 if quick else 7,
            "n_angles_half": 8 if quick else 16,
        }
    )
    return tasks


def build_task_mesh_metadata(task):
    """
    Build the mesh (without running `solver.emsolve`) for a task descriptor and
    return its measured mesh resolution metadata.

    Parameters
    ----------
    task : dict
        Task dictionary from `build_tasks(quick=...)`.

    Returns
    -------
    info : dict
        Measured mesh resolution metrics (`dx_background_nm`, `dx_min_nm`,
        `min_feature_nm`, `cells_across_feature`, `skin_depth_nm`,
        `power_skin_depth_nm`, `cells_per_skin_depth`,
        `cells_per_power_skin_depth`, `n_cells`, `n_steps`, `demo`, `mirror`, `cpw`).
    """
    idx = int(task["index"])
    is_demo = bool(task.get("demo", False))
    if idx == 0:
        cpw_min = min([int(task["cpw_bloch"])] + [int(c) for c in task["cpw_conv"]])
        meta_pec = solve_bloch_flat_slab(
            25.0,
            D_FLAT_BENCH,
            N_SLAB,
            LAM0,
            cpw=cpw_min,
            mirror="pec",
            n_cycles=28,
            allow_underresolved=is_demo,
            build_only=True,
        )
        meta_ag = solve_bloch_flat_slab(
            0.0,
            D_FLAT_BENCH,
            N_SLAB,
            LAM0,
            cpw=int(task["cpw_bloch"]),
            mirror="ag",
            n_cycles=28,
            cpsd_target=float(task.get("cpsd_target", 8.0)),
            allow_underresolved=is_demo,
            build_only=True,
        )
        return {
            "task": idx,
            "name": task["name"],
            "demo": is_demo,
            "mirror": "ag",
            "cpw": int(task["cpw_bloch"]),
            "cpw_conv_min": cpw_min,
            "dx_background_nm": meta_ag["dx_background_nm"],
            "dx_conv_max_nm": meta_pec["dx_background_nm"],
            "dx_min_nm": meta_ag["dx_min_nm"],
            "min_feature_nm": min(meta_pec["min_feature_nm"], meta_ag["min_feature_nm"]),
            "cells_across_feature": meta_ag["cells_across_feature"],
            "cells_across_feature_conv_min": meta_pec["cells_across_feature"],
            "skin_depth_nm": meta_ag["skin_depth_nm"],
            "power_skin_depth_nm": meta_ag["power_skin_depth_nm"],
            "cells_per_skin_depth": meta_ag["cells_per_skin_depth"],
            "cells_per_power_skin_depth": meta_ag["cells_per_power_skin_depth"],
            "n_cells": meta_ag["n_cells"],
            "n_steps": meta_ag["n_steps"],
        }

    mirror = task.get("mirror", "pec")
    cpsd_target = task.get("cpsd_target", None)
    n_periods = int(task.get("n_periods", N_PERIODS))
    cpw = int(task["cpw"])
    _, meta = make_finite_led_solver(
        lam0=LAM0,
        n_slab=N_SLAB,
        d_slab=D_SLAB,
        pitch=PITCH,
        h_grat=H_GRAT,
        n_periods=n_periods,
        cpw=cpw,
        averaging=task.get("averaging", "kottke"),
        mirror=mirror,
        cpsd_target=cpsd_target,
        allow_underresolved=is_demo,
        n_cycles=22,
    )
    return {
        "task": idx,
        "name": task["name"],
        "demo": is_demo,
        "mirror": mirror,
        "cpw": cpw,
        "snap_grid": bool(meta.get("snap_grid", False)),
        "dx_background_nm": meta["dx_background_nm"],
        "dx_min_nm": meta["dx_min_nm"],
        "min_feature_nm": meta["min_feature_nm"],
        "cells_across_feature": meta["cells_across_feature"],
        "skin_depth_nm": meta["skin_depth_nm"],
        "power_skin_depth_nm": meta["power_skin_depth_nm"],
        "cells_per_skin_depth": meta["cells_per_skin_depth"],
        "cells_per_power_skin_depth": meta["cells_per_power_skin_depth"],
        "n_cells": meta["n_cells"],
        "n_steps": meta["n_steps"],
        "staircase_geometry": meta.get("staircase_geometry"),
    }


# %% [markdown]
# ## 3. Task Runners: Bloch Unit Cell, Direct Ensemble & Grating Reciprocity
#
# Each task saves a self-contained `.npz` archive and `.json` summary in `data/`
# so that studies can run in parallel across cloud workers and figures can be
# regenerated at any time with `--plot`.


# %%
def run_task_0_flat_bloch(task, out_dir):
    """Run Task 0: flat-slab Bloch reciprocity vs analytical TMM (PEC and graded Drude Ag)."""
    t0 = time.perf_counter()
    cpw = int(task["cpw_bloch"])
    cpsd_ag = float(task.get("cpsd_target", 8.0))
    thetas_deg = np.linspace(0.0, 75.0, int(task["n_angles"]))

    # Standing-wave profile at oblique angle theta = 25 deg (PEC) and theta = 0, 25 deg (graded Ag)
    res_pec_25 = solve_bloch_flat_slab(
        25.0, D_FLAT_BENCH, N_SLAB, LAM0, cpw=cpw, mirror="pec", n_cycles=28
    )
    res_pec_0 = solve_bloch_flat_slab(
        0.0, D_FLAT_BENCH, N_SLAB, LAM0, cpw=cpw, mirror="pec", n_cycles=28
    )
    res_ag_0 = solve_bloch_flat_slab(
        0.0,
        D_FLAT_BENCH,
        N_SLAB,
        LAM0,
        cpw=cpw,
        mirror="ag",
        n_cycles=28,
        cpsd_target=cpsd_ag,
    )
    res_ag_25 = solve_bloch_flat_slab(
        25.0,
        D_FLAT_BENCH,
        N_SLAB,
        LAM0,
        cpw=cpw,
        mirror="ag",
        n_cycles=28,
        cpsd_target=cpsd_ag,
    )

    # Angular sweep across theta in [0, 75] deg on PEC and Ag mirrors
    j_pick = int(np.argmin(np.abs(res_pec_25["y_ez"] - 0.16 * LAM0)))
    y_emit_ez = float(res_pec_25["y_ez"][j_pick])
    j_pick_y = int(np.argmin(np.abs(res_pec_25["y_ey"] - 0.16 * LAM0)))
    y_emit_ey = float(res_pec_25["y_ey"][j_pick_y])

    I_tm_ang_pec = np.zeros_like(thetas_deg)
    I_tex_ang_pec = np.zeros_like(thetas_deg)
    I_tey_ang_pec = np.zeros_like(thetas_deg)
    I_tm_ang_ag = np.zeros_like(thetas_deg)

    for i, th in enumerate(thetas_deg):
        r_p = solve_bloch_flat_slab(
            float(th), D_FLAT_BENCH, N_SLAB, LAM0, cpw=cpw, mirror="pec", n_cycles=26
        )
        jp = int(np.argmin(np.abs(r_p["y_ez"] - y_emit_ez)))
        jpy = int(np.argmin(np.abs(r_p["y_ey"] - y_emit_ey)))
        I_tm_ang_pec[i] = r_p["I_tm"][jp]
        I_tex_ang_pec[i] = r_p["I_te_x"][jp]
        I_tey_ang_pec[i] = r_p["I_te_y"][jpy]

        r_a = solve_bloch_flat_slab(
            float(th),
            D_FLAT_BENCH,
            N_SLAB,
            LAM0,
            cpw=cpw,
            mirror="ag",
            n_cycles=26,
            cpsd_target=cpsd_ag,
        )
        jpa = int(np.argmin(np.abs(r_a["y_ez"] - y_emit_ez)))
        I_tm_ang_ag[i] = r_a["I_tm"][jpa]

    _, eps_ag, skin_depth = make_drude_silver(LAM0)
    thetas_rad = np.deg2rad(thetas_deg)
    I_tm_ana_pec = tmm_slab_emission(
        thetas_rad, y_emit_ez, D_FLAT_BENCH, N_SLAB, LAM0, "TM", eps_mirror=np.inf
    )
    I_tex_ana_pec = tmm_slab_emission(
        thetas_rad, y_emit_ez, D_FLAT_BENCH, N_SLAB, LAM0, "TE_x", eps_mirror=np.inf
    )
    I_tey_ana_pec = tmm_slab_emission(
        thetas_rad, y_emit_ey, D_FLAT_BENCH, N_SLAB, LAM0, "TE_y", eps_mirror=np.inf
    )
    I_tm_ana_ag = tmm_slab_emission(
        thetas_rad, y_emit_ez, D_FLAT_BENCH, N_SLAB, LAM0, "TM", eps_mirror=eps_ag
    )

    # 1D Bloch PEC resolution convergence sweep at theta = 25 deg
    cpw_conv = np.asarray(task["cpw_conv"], dtype=int)
    err_tm_conv = np.zeros(len(cpw_conv), dtype=float)
    err_tex_conv = np.zeros(len(cpw_conv), dtype=float)
    err_tey_conv = np.zeros(len(cpw_conv), dtype=float)
    dy_conv = np.zeros(len(cpw_conv), dtype=float)
    for ic, c_val in enumerate(cpw_conv):
        rc = solve_bloch_flat_slab(
            25.0, D_FLAT_BENCH, N_SLAB, LAM0, cpw=int(c_val), mirror="pec", n_cycles=28
        )
        dy_conv[ic] = rc["dy"]
        err_tm_conv[ic] = np.linalg.norm(rc["I_tm"] - rc["I_tm_ana"]) / np.linalg.norm(
            rc["I_tm_ana"]
        )
        err_tex_conv[ic] = np.linalg.norm(rc["I_te_x"] - rc["I_te_x_ana"]) / np.linalg.norm(
            rc["I_te_x_ana"]
        )
        err_tey_conv[ic] = np.linalg.norm(rc["I_te_y"] - rc["I_te_y_ana"]) / np.linalg.norm(
            rc["I_te_y_ana"]
        )

    # 1D Bloch Drude Ag graded skin-depth resolution ladder (theta = 0 and 25 deg)
    cpsd_ag_targets = np.asarray(
        task.get("cpsd_ag_ladder", [1.6, 2.6, 4.0, 6.0, 8.0, 10.0]), dtype=float
    )
    cpsd_ag_meas = np.zeros_like(cpsd_ag_targets)
    cpsd_pow_ag_meas = np.zeros_like(cpsd_ag_targets)
    dy_min_ag_nm = np.zeros_like(cpsd_ag_targets)
    err_ag0_ladder = np.zeros_like(cpsd_ag_targets)
    err_ag25_ladder = np.zeros_like(cpsd_ag_targets)
    I_qw_ag0_ladder = np.zeros_like(cpsd_ag_targets)
    for ia, cpsd_t in enumerate(cpsd_ag_targets):
        is_under = bool(cpsd_t < 3.5)
        ra0 = solve_bloch_flat_slab(
            0.0,
            D_FLAT_BENCH,
            N_SLAB,
            LAM0,
            cpw=cpw,
            mirror="ag",
            n_cycles=28,
            cpsd_target=float(cpsd_t),
            allow_underresolved=is_under,
        )
        ra25 = solve_bloch_flat_slab(
            25.0,
            D_FLAT_BENCH,
            N_SLAB,
            LAM0,
            cpw=cpw,
            mirror="ag",
            n_cycles=28,
            cpsd_target=float(cpsd_t),
            allow_underresolved=is_under,
        )
        cpsd_ag_meas[ia] = ra0["cells_per_skin_depth"]
        cpsd_pow_ag_meas[ia] = ra0["cells_per_power_skin_depth"]
        dy_min_ag_nm[ia] = ra0["dx_min_nm"]
        err_ag0_ladder[ia] = np.linalg.norm(ra0["I_tm"] - ra0["I_tm_ana"]) / np.linalg.norm(
            ra0["I_tm_ana"]
        )
        err_ag25_ladder[ia] = np.linalg.norm(ra25["I_tm"] - ra25["I_tm_ana"]) / np.linalg.norm(
            ra25["I_tm_ana"]
        )
        jqw = int(np.argmin(np.abs(ra0["y_ez"] - y_emit_ez)))
        I_qw_ag0_ladder[ia] = float(ra0["I_tm"][jqw])

    elapsed = time.perf_counter() - t0
    idx = task["index"]
    np.savez_compressed(
        out_dir / f"task_{idx:02d}.npz",
        thetas_deg=thetas_deg,
        y_emit_ez=y_emit_ez,
        y_emit_ey=y_emit_ey,
        y_ez_pec=res_pec_25["y_ez"],
        y_ey_pec=res_pec_25["y_ey"],
        I_tm_prof_pec25=res_pec_25["I_tm"],
        I_tex_prof_pec25=res_pec_25["I_te_x"],
        I_tey_prof_pec25=res_pec_25["I_te_y"],
        I_tm_prof_ana25=res_pec_25["I_tm_ana"],
        I_tex_prof_ana25=res_pec_25["I_te_x_ana"],
        I_tey_prof_ana25=res_pec_25["I_te_y_ana"],
        I_tm_prof_pec0=res_pec_0["I_tm"],
        I_tm_prof_pec0_ana=res_pec_0["I_tm_ana"],
        y_ez_ag=res_ag_0["y_ez"],
        I_tm_prof_ag0=res_ag_0["I_tm"],
        I_tm_prof_ag0_ana=res_ag_0["I_tm_ana"],
        I_tm_prof_ag25=res_ag_25["I_tm"],
        I_tm_prof_ag25_ana=res_ag_25["I_tm_ana"],
        I_tm_ang_pec=I_tm_ang_pec,
        I_tex_ang_pec=I_tex_ang_pec,
        I_tey_ang_pec=I_tey_ang_pec,
        I_tm_ang_ag=I_tm_ang_ag,
        I_tm_ana_pec=I_tm_ana_pec,
        I_tex_ana_pec=I_tex_ana_pec,
        I_tey_ana_pec=I_tey_ana_pec,
        I_tm_ana_ag=I_tm_ana_ag,
        cpw_conv=cpw_conv,
        dy_conv=dy_conv,
        err_tm_conv=err_tm_conv,
        err_tex_conv=err_tex_conv,
        err_tey_conv=err_tey_conv,
        cpsd_ag_targets=cpsd_ag_targets,
        cpsd_ag_meas=cpsd_ag_meas,
        cpsd_pow_ag_meas=cpsd_pow_ag_meas,
        dy_min_ag_nm=dy_min_ag_nm,
        err_ag0_ladder=err_ag0_ladder,
        err_ag25_ladder=err_ag25_ladder,
        I_qw_ag0_ladder=I_qw_ag0_ladder,
        eps_ag_real=float(np.real(eps_ag)),
        eps_ag_imag=float(np.imag(eps_ag)),
        dx_background_nm=res_ag_0["dx_background_nm"],
        dx_min_nm=res_ag_0["dx_min_nm"],
        min_feature_nm=res_ag_0["min_feature_nm"],
        cells_across_feature=res_ag_0["cells_across_feature"],
        skin_depth_nm=res_ag_0["skin_depth_nm"],
        power_skin_depth_nm=res_ag_0["power_skin_depth_nm"],
        cells_per_skin_depth=res_ag_0["cells_per_skin_depth"],
        cells_per_power_skin_depth=res_ag_0["cells_per_power_skin_depth"],
        skin_depth_cells=res_ag_0["cells_per_skin_depth"],
        n_cells=res_ag_0["n_cells"],
        n_steps=res_ag_0["n_steps"],
    )
    summary = {
        "task": idx,
        "name": task["name"],
        "demo": False,
        "cpw": cpw,
        "dx_background_nm": res_ag_0["dx_background_nm"],
        "dx_min_nm": res_ag_0["dx_min_nm"],
        "min_feature_nm": res_ag_0["min_feature_nm"],
        "cells_across_feature": res_ag_0["cells_across_feature"],
        "skin_depth_nm": res_ag_0["skin_depth_nm"],
        "power_skin_depth_nm": res_ag_0["power_skin_depth_nm"],
        "cells_per_skin_depth": res_ag_0["cells_per_skin_depth"],
        "cells_per_power_skin_depth": res_ag_0["cells_per_power_skin_depth"],
        "skin_depth_cells": res_ag_0["cells_per_skin_depth"],
        "n_cells": res_ag_0["n_cells"],
        "n_steps": res_ag_0["n_steps"],
        "elapsed_s": round(elapsed, 2),
        "l2_err_tm_pec25": float(
            np.linalg.norm(res_pec_25["I_tm"] - res_pec_25["I_tm_ana"])
            / np.linalg.norm(res_pec_25["I_tm_ana"])
        ),
        "l2_err_tex_pec25": float(
            np.linalg.norm(res_pec_25["I_te_x"] - res_pec_25["I_te_x_ana"])
            / np.linalg.norm(res_pec_25["I_te_x_ana"])
        ),
        "l2_err_tey_pec25": float(
            np.linalg.norm(res_pec_25["I_te_y"] - res_pec_25["I_te_y_ana"])
            / np.linalg.norm(res_pec_25["I_te_y_ana"])
        ),
        "l2_err_tm_ag0": float(
            np.linalg.norm(res_ag_0["I_tm"] - res_ag_0["I_tm_ana"])
            / np.linalg.norm(res_ag_0["I_tm_ana"])
        ),
        "l2_err_tm_ag25": float(
            np.linalg.norm(res_ag_25["I_tm"] - res_ag_25["I_tm_ana"])
            / np.linalg.norm(res_ag_25["I_tm_ana"])
        ),
        "ag_ladder_cpsd": [round(float(x), 2) for x in cpsd_ag_meas],
        "ag_ladder_err_tm0": [round(float(x), 5) for x in err_ag0_ladder],
    }
    (out_dir / f"task_{idx:02d}.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_task_1_ensemble(task, out_dir):
    """Run Task 1: Method (A) Direct Dipole Sum vs Method (B) Reciprocity on grating & flat LEDs."""
    t0 = time.perf_counter()
    cpw = int(task["cpw"])
    n_pos = int(task["n_pos"])
    thetas_deg = np.linspace(-70.0, 70.0, int(task["n_angles_sym"]))

    # Build solver once to inspect the central 3 grating periods for dipole placement
    _, meta_probe = make_finite_led_solver(
        lam0=LAM0,
        n_slab=N_SLAB,
        d_slab=D_SLAB,
        pitch=PITCH,
        h_grat=H_GRAT,
        n_periods=N_PERIODS,
        cpw=cpw,
    )
    dx = meta_probe["dx"]
    max_offset = int(round(1.5 * PITCH / dx))
    di_offsets = np.unique(np.round(np.linspace(-max_offset, max_offset, n_pos)).astype(int))

    # 1. Reciprocal Method (B) on grating LED
    t_rec0 = time.perf_counter()
    rec_grat = run_finite_led_reciprocity(
        thetas_deg=thetas_deg,
        lam0=LAM0,
        n_slab=N_SLAB,
        d_slab=D_SLAB,
        pitch=PITCH,
        h_grat=H_GRAT,
        n_periods=N_PERIODS,
        cpw=cpw,
        averaging="kottke",
        n_cycles=22,
    )
    time_rec_total = time.perf_counter() - t_rec0
    time_per_angle_rec = time_rec_total / len(thetas_deg)

    # 2. Direct Method (A) on grating LED (3 orientations x len(di_offsets) positions)
    t_dir0 = time.perf_counter()
    dir_grat = run_finite_led_direct_dipoles(
        x_indices_offset=di_offsets,
        thetas_deg=thetas_deg,
        lam0=LAM0,
        n_slab=N_SLAB,
        d_slab=D_SLAB,
        pitch=PITCH,
        h_grat=H_GRAT,
        n_periods=N_PERIODS,
        cpw=cpw,
        averaging="kottke",
        n_cycles=22,
        components=("z", "x", "y"),
        capture_snapshot_first=True,
    )
    time_dir_total = time.perf_counter() - t_dir0
    n_dir_runs = 3 * len(di_offsets)
    time_per_dipole_run = time_dir_total / n_dir_runs

    # 3. Flat untextured LED (h_grat = 0) via Reciprocity + Direct TM dipoles for comparison
    rec_flat = run_finite_led_reciprocity(
        thetas_deg=thetas_deg,
        lam0=LAM0,
        n_slab=N_SLAB,
        d_slab=D_SLAB,
        pitch=PITCH,
        h_grat=0.0,
        n_periods=N_PERIODS,
        cpw=cpw,
        averaging="kottke",
        n_cycles=22,
    )
    dir_flat = run_finite_led_direct_dipoles(
        x_indices_offset=di_offsets,
        thetas_deg=thetas_deg,
        lam0=LAM0,
        n_slab=N_SLAB,
        d_slab=D_SLAB,
        pitch=PITCH,
        h_grat=0.0,
        n_periods=N_PERIODS,
        cpw=cpw,
        averaging="kottke",
        n_cycles=22,
        components=("z",),
        capture_snapshot_first=False,
    )

    # Extract reciprocal intensities at the exact dipole coordinates for point-by-point comparison
    comp_map = {"z": ("x_ez", "I_tm"), "x": ("x_ex", "I_te_x"), "y": ("x_ey", "I_te_y")}
    rec_at_dip = {}
    max_pt_rel_err = {}
    for c, (xk, Ik) in comp_map.items():
        arr = np.zeros_like(dir_grat["I_dip"][c])
        for ip, xd in enumerate(dir_grat["x_pos"][c]):
            ix = int(np.argmin(np.abs(rec_grat[xk] - xd)))
            arr[ip, :] = rec_grat[Ik][:, ix]
        rec_at_dip[c] = arr
        max_pt_rel_err[c] = float(
            np.max(np.abs(dir_grat["I_dip"][c] - arr) / np.maximum(arr, 0.1))
        )

    # Ensemble averages at the sampled dipole positions vs full continuous active-layer integral
    rec_sample_mean = {c: np.mean(rec_at_dip[c], axis=0) for c in ("z", "x", "y")}
    rec_cont_mean = {
        "z": np.mean(rec_grat["I_tm"], axis=1),
        "x": np.mean(rec_grat["I_te_x"], axis=1),
        "y": np.mean(rec_grat["I_te_y"], axis=1),
    }
    rec_cont_unpol = (rec_cont_mean["z"] + rec_cont_mean["x"] + rec_cont_mean["y"]) / 3.0

    rec_flat_mean = {
        "z": np.mean(rec_flat["I_tm"], axis=1),
        "x": np.mean(rec_flat["I_te_x"], axis=1),
        "y": np.mean(rec_flat["I_te_y"], axis=1),
    }
    rec_flat_unpol = (rec_flat_mean["z"] + rec_flat_mean["x"] + rec_flat_mean["y"]) / 3.0

    # Full fine-x reciprocal profiles at theta = 0 deg and theta ~ 25 deg for Fig 6
    i_th0 = int(np.argmin(np.abs(thetas_deg - 0.0)))
    i_th25 = int(np.argmin(np.abs(thetas_deg - 25.0)))

    elapsed = time.perf_counter() - t0
    idx = task["index"]
    dip_snap = dir_grat["dipole_dft_2d"]
    m_grat = rec_grat["meta"]
    np.savez_compressed(
        out_dir / f"task_{idx:02d}.npz",
        thetas_deg=thetas_deg,
        di_offsets=di_offsets,
        x_ez=rec_grat["x_ez"],
        x_ex=rec_grat["x_ex"],
        x_ey=rec_grat["x_ey"],
        I_tm_rec_full=rec_grat["I_tm"],
        I_tex_rec_full=rec_grat["I_te_x"],
        I_tey_rec_full=rec_grat["I_te_y"],
        x_dip_z=dir_grat["x_pos"]["z"],
        x_dip_x=dir_grat["x_pos"]["x"],
        x_dip_y=dir_grat["x_pos"]["y"],
        I_dip_z=dir_grat["I_dip"]["z"],
        I_dip_x=dir_grat["I_dip"]["x"],
        I_dip_y=dir_grat["I_dip"]["y"],
        rec_at_dip_z=rec_at_dip["z"],
        rec_at_dip_x=rec_at_dip["x"],
        rec_at_dip_y=rec_at_dip["y"],
        dir_incoh_z=dir_grat["I_incoh_by_comp"]["z"],
        dir_incoh_x=dir_grat["I_incoh_by_comp"]["x"],
        dir_incoh_y=dir_grat["I_incoh_by_comp"]["y"],
        dir_incoh_unpol=dir_grat["I_incoh_mean"],
        dir_coh_tm=dir_grat["I_coh_tm"],
        dir_coh_te=dir_grat["I_coh_te"],
        dir_flat_incoh_z=dir_flat["I_incoh_by_comp"]["z"],
        dir_flat_coh_tm=dir_flat["I_coh_tm"],
        rec_sample_z=rec_sample_mean["z"],
        rec_sample_x=rec_sample_mean["x"],
        rec_sample_y=rec_sample_mean["y"],
        rec_cont_z=rec_cont_mean["z"],
        rec_cont_x=rec_cont_mean["x"],
        rec_cont_y=rec_cont_mean["y"],
        rec_cont_unpol=rec_cont_unpol,
        rec_flat_z=rec_flat_mean["z"],
        rec_flat_x=rec_flat_mean["x"],
        rec_flat_y=rec_flat_mean["y"],
        rec_flat_unpol=rec_flat_unpol,
        y_emit_ez=m_grat["y_emit_ez"],
        y_emit_ey=m_grat["y_emit_ey"],
        i_th0=i_th0,
        i_th25=i_th25,
        time_per_angle_rec=time_per_angle_rec,
        time_per_dipole_run=time_per_dipole_run,
        n_dir_runs=n_dir_runs,
        dip_snap_x=dip_snap["x_ez"],
        dip_snap_y=dip_snap["y_ez"],
        dip_snap_Ez=dip_snap["Ez"],
        dip_snap_xdip=dip_snap["x_dip"],
        dip_snap_ydip=dip_snap["y_dip"],
        dx_background_nm=m_grat["dx_background_nm"],
        dx_min_nm=m_grat["dx_min_nm"],
        min_feature_nm=m_grat["min_feature_nm"],
        cells_across_feature=m_grat["cells_across_feature"],
        n_cells=m_grat["n_cells"],
        n_steps=m_grat["n_steps"],
    )

    # Compute angle-integrated extraction enhancement (grating vs flat)
    th_rad = np.deg2rad(thetas_deg)
    P_grat_unpol = float(np.trapezoid(rec_cont_unpol, th_rad))
    P_flat_unpol = float(np.trapezoid(rec_flat_unpol, th_rad))
    P_grat_tm = float(np.trapezoid(rec_cont_mean["z"], th_rad))
    P_flat_tm = float(np.trapezoid(rec_flat_mean["z"], th_rad))

    summary = {
        "task": idx,
        "name": task["name"],
        "demo": False,
        "cpw": cpw,
        "dx_background_nm": m_grat["dx_background_nm"],
        "dx_min_nm": m_grat["dx_min_nm"],
        "min_feature_nm": m_grat["min_feature_nm"],
        "cells_across_feature": m_grat["cells_across_feature"],
        "skin_depth_nm": None,
        "cells_per_skin_depth": None,
        "n_cells": m_grat["n_cells"],
        "n_steps": m_grat["n_steps"],
        "elapsed_s": round(elapsed, 2),
        "n_pos": int(len(di_offsets)),
        "n_dir_runs": int(n_dir_runs),
        "n_angles": int(len(thetas_deg)),
        "time_per_angle_rec_s": round(time_per_angle_rec, 4),
        "time_per_dipole_run_s": round(time_per_dipole_run, 4),
        "max_pt_rel_err_z": round(max_pt_rel_err["z"], 6),
        "max_pt_rel_err_x": round(max_pt_rel_err["x"], 6),
        "max_pt_rel_err_y": round(max_pt_rel_err["y"], 6),
        "ensemble_l2_err_unpol": float(
            np.linalg.norm(
                dir_grat["I_incoh_mean"]
                - np.mean(
                    [rec_sample_mean["z"], rec_sample_mean["x"], rec_sample_mean["y"]], axis=0
                )
            )
            / np.linalg.norm(dir_grat["I_incoh_mean"])
        ),
        "grating_enhancement_unpol": round(P_grat_unpol / P_flat_unpol, 4),
        "grating_enhancement_tm": round(P_grat_tm / P_flat_tm, 4),
    }
    (out_dir / f"task_{idx:02d}.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_task_grating_resolution(task, out_dir):
    """Run Tasks 2..8: Kottke vs Staircase (snapped & un-snapped) resolution sweep on the 2D grating LED."""
    t0 = time.perf_counter()
    cpw = int(task["cpw"])
    avg = task["averaging"]
    thetas_half = np.linspace(0.0, 75.0, int(task["n_angles_half"]))
    capture = bool(task.get("capture_fields", False))

    res = run_finite_led_reciprocity(
        thetas_deg=thetas_half,
        lam0=LAM0,
        n_slab=N_SLAB,
        d_slab=D_SLAB,
        pitch=PITCH,
        h_grat=H_GRAT,
        n_periods=N_PERIODS,
        cpw=cpw,
        averaging=avg,
        n_cycles=22,
        capture_2d_theta_deg=25.0 if capture else None,
        capture_frames=capture,
        snap_grid=(True if avg == "staircase" else False),
    )

    # For staircase tasks, also evaluate the un-snapped uniform grid (snap_grid=False)
    # to quantify the non-monotonic period and slab-thickness rounding error.
    # At cpw == 10, the snapped and un-snapped grids are identical (dx = dy = 18.0 nm).
    res_unsnapped = None
    if avg == "staircase":
        if cpw == 10:
            res_unsnapped = res
        else:
            res_unsnapped = run_finite_led_reciprocity(
                thetas_deg=thetas_half,
                lam0=LAM0,
                n_slab=N_SLAB,
                d_slab=D_SLAB,
                pitch=PITCH,
                h_grat=H_GRAT,
                n_periods=N_PERIODS,
                cpw=cpw,
                averaging="staircase",
                n_cycles=22,
                snap_grid=False,
            )

    # Also capture 2D DFT at normal incidence theta = 0 deg when capture_fields is True
    dft_0 = None
    if capture:
        res0 = run_finite_led_reciprocity(
            thetas_deg=[0.0],
            lam0=LAM0,
            n_slab=N_SLAB,
            d_slab=D_SLAB,
            pitch=PITCH,
            h_grat=H_GRAT,
            n_periods=N_PERIODS,
            cpw=cpw,
            averaging=avg,
            n_cycles=22,
            capture_2d_theta_deg=0.0,
            capture_frames=False,
            snap_grid=False,
        )
        dft_0 = res0["dft_2d"]

    I_tm_mean = np.mean(res["I_tm"], axis=1)
    I_tex_mean = np.mean(res["I_te_x"], axis=1)
    I_tey_mean = np.mean(res["I_te_y"], axis=1)
    I_unpol_mean = (I_tm_mean + I_tex_mean + I_tey_mean) / 3.0

    # Also store staircase permittivity tensor on the same grid when capture is True
    eps_stair_x = None
    eps_stair_y = None
    eps_stair_z = None
    if capture:
        solver_st, _ = make_finite_led_solver(
            lam0=LAM0,
            n_slab=N_SLAB,
            d_slab=D_SLAB,
            pitch=PITCH,
            h_grat=H_GRAT,
            n_periods=N_PERIODS,
            cpw=cpw,
            averaging="staircase",
            snap_grid=False,
        )
        from scipy.constants import epsilon_0 as _eps0

        eps_stair_x = np.where(
            np.asarray(solver_st.ieps[:, :, 0, "x"]) > 0,
            1.0 / (_eps0 * np.maximum(np.asarray(solver_st.ieps[:, :, 0, "x"]), 1e-30)),
            np.inf,
        )
        eps_stair_y = np.where(
            np.asarray(solver_st.ieps[:, :, 0, "y"]) > 0,
            1.0 / (_eps0 * np.maximum(np.asarray(solver_st.ieps[:, :, 0, "y"]), 1e-30)),
            np.inf,
        )
        eps_stair_z = np.where(
            np.asarray(solver_st.ieps[:, :, 0, "z"]) > 0,
            1.0 / (_eps0 * np.maximum(np.asarray(solver_st.ieps[:, :, 0, "z"]), 1e-30)),
            np.inf,
        )

    m = res["meta"]
    save_dict = {
        "thetas_half": thetas_half,
        "cpw": cpw,
        "averaging": avg,
        "snap_grid": bool(m.get("snap_grid", False)),
        "dx": m["dx"],
        "dx_background_nm": m["dx_background_nm"],
        "dx_min_nm": m["dx_min_nm"],
        "min_feature_nm": m["min_feature_nm"],
        "cells_across_feature": m["cells_across_feature"],
        "n_cells": m["n_cells"],
        "n_steps": m["n_steps"],
        "y_emit_ez": m["y_emit_ez"],
        "y_emit_ey": m["y_emit_ey"],
        "y_mon": m["y_mon"],
        "w_mon": m["w_mon"],
        "w_slab": m["w_slab"],
        "I_tm_mean": I_tm_mean,
        "I_tex_mean": I_tex_mean,
        "I_tey_mean": I_tey_mean,
        "I_unpol_mean": I_unpol_mean,
    }
    if res_unsnapped is not None:
        m_u = res_unsnapped["meta"]
        I_tm_u = np.mean(res_unsnapped["I_tm"], axis=1)
        I_tex_u = np.mean(res_unsnapped["I_te_x"], axis=1)
        I_tey_u = np.mean(res_unsnapped["I_te_y"], axis=1)
        I_unpol_u = (I_tm_u + I_tex_u + I_tey_u) / 3.0
        save_dict.update(
            {
                "dx_unsnapped": m_u["dx"],
                "y_emit_ez_unsnapped": m_u["y_emit_ez"],
                "y_emit_ey_unsnapped": m_u["y_emit_ey"],
                "I_tm_unsnapped_mean": I_tm_u,
                "I_tex_unsnapped_mean": I_tex_u,
                "I_tey_unsnapped_mean": I_tey_u,
                "I_unpol_unsnapped_mean": I_unpol_u,
            }
        )
    if capture and res["dft_2d"] is not None:
        d25 = res["dft_2d"]
        save_dict.update(
            {
                "dft25_x_ez": d25["x_ez"],
                "dft25_y_ez": d25["y_ez"],
                "dft25_Ez": d25["Ez_norm"],
                "dft25_x_ex": d25["x_ex"],
                "dft25_y_ex": d25["y_ex"],
                "dft25_Ex": d25["Ex_norm"],
                "dft25_x_ey": d25["x_ey"],
                "dft25_y_ey": d25["y_ey"],
                "dft25_Ey": d25["Ey_norm"],
                "dft0_Ez": dft_0["Ez_norm"],
                "dft0_Ex": dft_0["Ex_norm"],
                "dft0_Ey": dft_0["Ey_norm"],
                "frames_Ez": res["frames"],
                "grid_x": res["grid_x"],
                "grid_y": res["grid_y"],
                "eps_kottke_x": res["eps_x"],
                "eps_kottke_y": res["eps_y"],
                "eps_kottke_z": res["eps_z"],
                "eps_stair_x": eps_stair_x,
                "eps_stair_y": eps_stair_y,
                "eps_stair_z": eps_stair_z,
            }
        )

    elapsed = time.perf_counter() - t0
    idx = task["index"]
    np.savez_compressed(out_dir / f"task_{idx:02d}.npz", **save_dict)

    th_rad = np.deg2rad(thetas_half)
    P_int_unpol = 2.0 * float(np.trapezoid(I_unpol_mean, th_rad))
    P_int_tm = 2.0 * float(np.trapezoid(I_tm_mean, th_rad))
    summary = {
        "task": idx,
        "name": task["name"],
        "demo": False,
        "averaging": avg,
        "snap_grid": bool(m.get("snap_grid", False)),
        "level": task["level"],
        "cpw": cpw,
        "dx_nm": round(m["dx"] * 1e9, 2),
        "dx_background_nm": m["dx_background_nm"],
        "dx_min_nm": m["dx_min_nm"],
        "min_feature_nm": m["min_feature_nm"],
        "cells_across_feature": m["cells_across_feature"],
        "skin_depth_nm": None,
        "cells_per_skin_depth": None,
        "n_cells": m["n_cells"],
        "n_steps": m["n_steps"],
        "elapsed_s": round(elapsed, 2),
        "P_int_unpol_rad": round(P_int_unpol, 4),
        "P_int_tm_rad": round(P_int_tm, 4),
        "I_unpol_normal": round(float(I_unpol_mean[0]), 4),
        "I_tm_normal": round(float(I_tm_mean[0]), 4),
        "staircase_geometry": m.get("staircase_geometry"),
    }
    if res_unsnapped is not None:
        m_u = res_unsnapped["meta"]
        P_int_unpol_u = 2.0 * float(np.trapezoid(save_dict["I_unpol_unsnapped_mean"], th_rad))
        P_int_tm_u = 2.0 * float(np.trapezoid(save_dict["I_tm_unsnapped_mean"], th_rad))
        summary.update(
            {
                "dx_unsnapped_nm": round(m_u["dx"] * 1e9, 2),
                "P_int_unpol_unsnapped_rad": round(P_int_unpol_u, 4),
                "P_int_tm_unsnapped_rad": round(P_int_tm_u, 4),
                "I_unpol_normal_unsnapped": round(float(save_dict["I_unpol_unsnapped_mean"][0]), 4),
                "I_tm_normal_unsnapped": round(float(save_dict["I_tm_unsnapped_mean"][0]), 4),
                "staircase_geometry_unsnapped": m_u.get("staircase_geometry"),
            }
        )
    (out_dir / f"task_{idx:02d}.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_task_ag_metal_study(task, out_dir):
    """
    Run Tasks 9 (under-resolved demo) and 10 (resolved graded-mesh study) on the
    2D grating LED backed by the dispersive Drude Ag mirror (`y <= 0`).
    """
    t0 = time.perf_counter()
    idx = int(task["index"])
    is_demo = bool(task.get("demo", False))
    cpw = int(task["cpw"])
    n_periods = int(task.get("n_periods", 5))
    thetas_half = np.linspace(0.0, 70.0, int(task.get("n_angles_half", 8)))
    th_rad = np.deg2rad(thetas_half)
    cpsd_levels = [float(x) for x in task["cpsd_levels"]]

    _, eps_ag, skin_depth = make_drude_silver(LAM0)
    # Analytic TMM flat-slab reference on Drude Ag for the same slab thickness D_SLAB and emitter Y_EMIT
    I_flat_ag_tmm_unpol = tmm_slab_emission(
        th_rad, Y_EMIT, D_SLAB, N_SLAB, LAM0, pol="unpolarized", eps_mirror=eps_ag
    )
    I_flat_ag_tmm_tm = tmm_slab_emission(
        th_rad, Y_EMIT, D_SLAB, N_SLAB, LAM0, pol="TM", eps_mirror=eps_ag
    )
    P_flat_ag_tmm_unpol = 2.0 * float(np.trapezoid(I_flat_ag_tmm_unpol, th_rad))
    P_flat_ag_tmm_tm = 2.0 * float(np.trapezoid(I_flat_ag_tmm_tm, th_rad))

    cpsd_meas = []
    cpsd_pow_meas = []
    dx_bg_list = []
    dx_min_list = []
    cells_feat_list = []
    n_cells_list = []
    n_steps_list = []
    P_grat_unpol_list = []
    P_grat_tm_list = []
    enh_unpol_list = []
    enh_tm_list = []
    I_tm_normal_list = []
    I_unpol_curves = []
    I_tm_curves = []
    sw_l2_err_list = []

    meta_primary = None
    for cpsd_t in cpsd_levels:
        res_g = run_finite_led_reciprocity(
            thetas_deg=thetas_half,
            lam0=LAM0,
            n_slab=N_SLAB,
            d_slab=D_SLAB,
            pitch=PITCH,
            h_grat=H_GRAT,
            n_periods=n_periods,
            cpw=cpw,
            averaging="kottke",
            n_cycles=20,
            mirror="ag",
            cpsd_target=cpsd_t,
            allow_underresolved=is_demo,
        )
        mg = res_g["meta"]
        if meta_primary is None or abs(cpsd_t - float(task["cpsd_target"])) < 1e-6:
            meta_primary = mg

        I_tm_m = np.mean(res_g["I_tm"], axis=1)
        I_tex_m = np.mean(res_g["I_te_x"], axis=1)
        I_tey_m = np.mean(res_g["I_te_y"], axis=1)
        I_unpol_m = (I_tm_m + I_tex_m + I_tey_m) / 3.0

        P_u = 2.0 * float(np.trapezoid(I_unpol_m, th_rad))
        P_t = 2.0 * float(np.trapezoid(I_tm_m, th_rad))

        # Standing-wave L2 error on the corresponding 1D Bloch flat slab at the same cpsd_t
        rb = solve_bloch_flat_slab(
            0.0,
            D_FLAT_BENCH,
            N_SLAB,
            LAM0,
            cpw=max(cpw, 20),
            mirror="ag",
            n_cycles=28,
            cpsd_target=cpsd_t,
            allow_underresolved=is_demo,
        )
        sw_err = float(
            np.linalg.norm(rb["I_tm"] - rb["I_tm_ana"]) / np.linalg.norm(rb["I_tm_ana"])
        )

        cpsd_meas.append(float(mg["cells_per_skin_depth"]))
        cpsd_pow_meas.append(float(mg["cells_per_power_skin_depth"]))
        dx_bg_list.append(float(mg["dx_background_nm"]))
        dx_min_list.append(float(mg["dx_min_nm"]))
        cells_feat_list.append(float(mg["cells_across_feature"]))
        n_cells_list.append(int(mg["n_cells"]))
        n_steps_list.append(int(mg["n_steps"]))
        P_grat_unpol_list.append(P_u)
        P_grat_tm_list.append(P_t)
        enh_unpol_list.append(P_u / P_flat_ag_tmm_unpol)
        enh_tm_list.append(P_t / P_flat_ag_tmm_tm)
        I_tm_normal_list.append(float(I_tm_m[0]))
        I_unpol_curves.append(I_unpol_m)
        I_tm_curves.append(I_tm_m)
        sw_l2_err_list.append(sw_err)

    elapsed = time.perf_counter() - t0
    np.savez_compressed(
        out_dir / f"task_{idx:02d}.npz",
        thetas_half=thetas_half,
        cpsd_targets=np.asarray(cpsd_levels, dtype=float),
        cpsd_meas=np.asarray(cpsd_meas, dtype=float),
        cpsd_pow_meas=np.asarray(cpsd_pow_meas, dtype=float),
        dx_bg_list=np.asarray(dx_bg_list, dtype=float),
        dx_min_list=np.asarray(dx_min_list, dtype=float),
        cells_feat_list=np.asarray(cells_feat_list, dtype=float),
        n_cells_list=np.asarray(n_cells_list, dtype=int),
        n_steps_list=np.asarray(n_steps_list, dtype=int),
        P_grat_unpol_list=np.asarray(P_grat_unpol_list, dtype=float),
        P_grat_tm_list=np.asarray(P_grat_tm_list, dtype=float),
        P_flat_ag_tmm_unpol=P_flat_ag_tmm_unpol,
        P_flat_ag_tmm_tm=P_flat_ag_tmm_tm,
        enh_unpol_list=np.asarray(enh_unpol_list, dtype=float),
        enh_tm_list=np.asarray(enh_tm_list, dtype=float),
        I_tm_normal_list=np.asarray(I_tm_normal_list, dtype=float),
        I_unpol_curves=np.asarray(I_unpol_curves, dtype=float),
        I_tm_curves=np.asarray(I_tm_curves, dtype=float),
        sw_l2_err_list=np.asarray(sw_l2_err_list, dtype=float),
        dx_background_nm=meta_primary["dx_background_nm"],
        dx_min_nm=meta_primary["dx_min_nm"],
        min_feature_nm=meta_primary["min_feature_nm"],
        cells_across_feature=meta_primary["cells_across_feature"],
        skin_depth_nm=meta_primary["skin_depth_nm"],
        power_skin_depth_nm=meta_primary["power_skin_depth_nm"],
        cells_per_skin_depth=meta_primary["cells_per_skin_depth"],
        cells_per_power_skin_depth=meta_primary["cells_per_power_skin_depth"],
        n_cells=meta_primary["n_cells"],
        n_steps=meta_primary["n_steps"],
    )

    rel_change_last = (
        abs(P_grat_unpol_list[-1] - P_grat_unpol_list[-2]) / P_grat_unpol_list[-1]
        if len(P_grat_unpol_list) >= 2
        else 0.0
    )
    summary = {
        "task": idx,
        "name": task["name"],
        "demo": is_demo,
        "mirror": "ag",
        "cpw": cpw,
        "dx_background_nm": meta_primary["dx_background_nm"],
        "dx_min_nm": meta_primary["dx_min_nm"],
        "min_feature_nm": meta_primary["min_feature_nm"],
        "cells_across_feature": meta_primary["cells_across_feature"],
        "skin_depth_nm": meta_primary["skin_depth_nm"],
        "power_skin_depth_nm": meta_primary["power_skin_depth_nm"],
        "cells_per_skin_depth": meta_primary["cells_per_skin_depth"],
        "cells_per_power_skin_depth": meta_primary["cells_per_power_skin_depth"],
        "n_cells": meta_primary["n_cells"],
        "n_steps": meta_primary["n_steps"],
        "elapsed_s": round(elapsed, 2),
        "cpsd_meas": [round(x, 2) for x in cpsd_meas],
        "cpsd_pow_meas": [round(x, 2) for x in cpsd_pow_meas],
        "P_grat_unpol_rad": [round(x, 4) for x in P_grat_unpol_list],
        "P_grat_tm_rad": [round(x, 4) for x in P_grat_tm_list],
        "grating_enhancement_ag_unpol": [round(x, 4) for x in enh_unpol_list],
        "grating_enhancement_ag_tm": [round(x, 4) for x in enh_tm_list],
        "sw_l2_err": [round(x, 5) for x in sw_l2_err_list],
        "rel_change_last_two_levels": round(float(rel_change_last), 5),
    }
    (out_dir / f"task_{idx:02d}.json").write_text(json.dumps(summary, indent=2))
    return summary


def execute_task(task, out_dir):
    """Dispatch a single task dictionary, print mesh resolution metrics, and write outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = task["index"]
    if idx == 0:
        summary = run_task_0_flat_bloch(task, out_dir)
    elif idx == 1:
        summary = run_task_1_ensemble(task, out_dir)
    elif idx in (9, 10):
        summary = run_task_ag_metal_study(task, out_dir)
    else:
        summary = run_task_grating_resolution(task, out_dir)

    print(
        f"[7.6 task_{idx:02d} {summary['name']}] "
        f"demo={summary.get('demo', False)} "
        f"dx_bg={summary['dx_background_nm']:.2f}nm "
        f"dx_min={summary['dx_min_nm']:.2f}nm "
        f"min_feat={summary['min_feature_nm']:.1f}nm "
        f"cells_feat={summary['cells_across_feature']:.2f} "
        f"skin_depth={summary.get('skin_depth_nm')}nm "
        f"cpsd={summary.get('cells_per_skin_depth')} "
        f"n_cells={summary['n_cells']} "
        f"n_steps={summary['n_steps']} "
        f"elapsed={summary['elapsed_s']:.2f}s",
        flush=True,
    )
    return summary


# %% [markdown]
# ## 4. Plotting & Visualization (20 Figures: Schematic, Reciprocity Validation & Grating Optimization)
#
# All figures are generated from the saved `data/task_*.npz`, `data/task_*.json`, and
# `data/janssen_optimization.npz` outputs so `--plot` can be re-run independently of
# the electromagnetic simulations.

# %%
def _plot_emitter_absorber_schematic(fig_dir):
    """Render `00_setup_schematic.png` and `00_emitter_absorber_schematic.png` using `wakis.schematic`."""
    import matplotlib.patches as mpatches

    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.8, height=4.5)

    # Panel (a): Direct Problem — LED as an Incoherent Emitter (3 * N_sites runs)
    obj_a = build_reciprocity_schematic_objects(mode="direct", cpw=12, mirror="ag", cpsd_target=10.0)
    schematic.plot_schematic(
        obj_a["solver"],
        plane="xy",
        ax=axes[0],
        unit="um",
        structures=obj_a["structures"],
        sources=obj_a["sources"],
        monitors=obj_a["monitors"],
        show_mesh=False,
        show_wave_packet=False,
        title="(a) Direct Problem: Incoherent Emitter\n($3 N_{\\mathrm{sites}}$ Dipole Solves)",
    )
    tidy_reciprocity_schematic_axes(axes[0], obj_a["meta"], mode="direct")

    # Panel (b): Reciprocal Problem — LED as an Absorber (Helmholtz Reciprocity, 1 run / angle)
    obj_b = build_reciprocity_schematic_objects(mode="reciprocal", cpw=12, mirror="ag", cpsd_target=10.0)
    schematic.plot_schematic(
        obj_b["solver"],
        plane="xy",
        ax=axes[1],
        unit="um",
        structures=obj_b["structures"],
        sources=obj_b["sources"],
        monitors=obj_b["monitors"],
        show_mesh=False,
        show_wave_packet=False,
        title="(b) Reciprocal Problem: Absorber\n(Huygens Plane Wave, 1 Run / Angle)",
    )
    tidy_reciprocity_schematic_axes(axes[1], obj_b["meta"], mode="reciprocal")

    # Panel (c): Graded Mesh Zoom around a grating tooth, active QW, and Drude Ag skin-depth collar
    _, meta_c = make_finite_led_solver(cpw=20, averaging="kottke", mirror="ag", cpsd_target=10.0)
    solver_c, _ = make_finite_led_solver(cpw=20, averaging="kottke", mirror="ag", cpsd_target=10.0)
    poly_c_um = np.asarray(meta_c["slab_poly"].vertices, dtype=float) * 1e6
    y_emit_c_um = float(meta_c["y_emit"]) * 1e6

    ax_c = axes[2]
    ax_c.add_patch(
        mpatches.Rectangle(
            (-0.5, -0.10),
            1.0,
            0.10,
            facecolor="#94a3b8",
            edgecolor="#334155",
            lw=1.0,
            alpha=0.80,
            zorder=2,
        )
    )
    ax_c.add_patch(
        mpatches.Polygon(
            poly_c_um,
            closed=True,
            facecolor="#bae6fd",
            edgecolor="#0284c7",
            lw=1.2,
            alpha=0.65,
            zorder=3,
        )
    )
    schematic.draw_mesh(
        ax_c,
        solver_c.grid,
        plane="xy",
        unit="um",
        color="#334155",
        alpha=0.45,
        linewidth=0.45,
    )
    ax_c.axhline(0.0, color="#0f172a", lw=1.2, zorder=5)
    ax_c.axhline(-0.0269, color="#e11d48", ls="--", lw=1.1, zorder=5)
    ax_c.axhline(y_emit_c_um, color="#dc2626", ls=":", lw=1.2, zorder=5)
    ax_c.set_xlim(-0.27, 0.27)
    ax_c.set_ylim(-0.06, 0.25)
    ax_c.set_aspect("auto")
    ax_c.set_xlabel(r"$x$ [$\mu\mathrm{m}$]")
    ax_c.set_ylabel(r"$y$ [$\mu\mathrm{m}$]")
    ax_c.set_title("(c) Graded Skin-Depth & Grating\nMesh Zoom (`draw_mesh`)")

    bbox_c = {
        "boxstyle": "round,pad=0.14",
        "facecolor": "white",
        "edgecolor": "#334155",
        "linewidth": 0.5,
        "alpha": 0.92,
    }
    ax_c.text(
        0.0,
        0.19,
        f"Grating tooth $h_g = 72\\;\\mathrm{{nm}}$: {meta_c['cells_across_feature']:.1f} cells\n"
        f"(GaN $n=2.45$: $20\\;\\mathrm{{cells}}/\\lambda_{{\\mathrm{{GaN}}}}$)",
        ha="center",
        va="center",
        fontsize=6.2,
        color="#0c4a6e",
        bbox=bbox_c,
        zorder=8,
    )
    ax_c.text(
        0.0,
        0.075,
        f"Active QW plane $y_0 = {y_emit_c_um * 1e3:.1f}\\;\\mathrm{{nm}}$",
        ha="center",
        va="center",
        fontsize=6.2,
        color="#991b1b",
        bbox=bbox_c,
        zorder=8,
    )
    ax_c.text(
        0.0,
        -0.032,
        f"Ag $\\delta_s = 26.9\\;\\mathrm{{nm}}$: {meta_c['cells_per_skin_depth']:.1f} cells/$\\delta_s$\n"
        f"($\\Delta y_{{\\min}} = {meta_c['dx_min_nm']:.2f}\\;\\mathrm{{nm}}$, {meta_c['cells_per_power_skin_depth']:.1f} cells/$\\delta_p$)",
        ha="center",
        va="center",
        fontsize=6.1,
        color="#9f1239",
        bbox=bbox_c,
        zorder=8,
    )

    figstyle.finish(fig)
    fig.savefig(fig_dir / "00_emitter_absorber_schematic.png")
    plt.close(fig)
    save_reciprocity_setup_figures(fig_dir)


def plot_all(out_dir, fig_dir):
    """Generate all tutorial figures from `out_dir/*.npz` into `fig_dir`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Polygon as MplPolygon

    figstyle.apply_style()

    out_dir = Path(out_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 0: Setup & Emitter -> Absorber schematic (Helmholtz reciprocity)
    _plot_emitter_absorber_schematic(fig_dir)

    d0 = np.load(out_dir / "task_00.npz")
    d1 = np.load(out_dir / "task_01.npz")
    # Task 4 (kottke_fine) captures 2D fields, frames, and permittivity tensors
    d2 = np.load(out_dir / "task_04.npz")

    # Load all available resolution tasks (2..8)
    res_tasks = {}
    for idx in range(2, 9):
        p_npz = out_dir / f"task_{idx:02d}.npz"
        p_json = out_dir / f"task_{idx:02d}.json"
        if p_npz.exists() and p_json.exists():
            res_tasks[idx] = (np.load(p_npz), json.loads(p_json.read_text()))

    # ------------------------------------------------------------------
    # Figure 1: Geometry, Mesh, and Kottke vs Staircase Permittivity
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(
        nrows=1, ncols=3, width=13.8, height=4.3, gridspec_kw={"width_ratios": [1.45, 1.0, 1.0]}
    )

    ax = axes[0]
    gx = d2["grid_x"] * 1e6
    gy = d2["grid_y"] * 1e6
    eps_kz = np.where(np.isinf(d2["eps_kottke_z"]), np.nan, d2["eps_kottke_z"])
    im0 = ax.pcolormesh(gx, gy, eps_kz.T, cmap="Blues", vmin=1.0, vmax=N_SLAB**2 + 0.5, shading="auto")
    fig.colorbar(im0, ax=ax, label=r"Relative permittivity $\varepsilon_z(x, y)$")

    # Overlay exact PolySlab polygon & wakis.schematic geometry overlay
    slab_poly = build_grating_led_polyslab(
        width_x=float(d2["w_slab"]),
        d_slab=D_SLAB,
        pitch=PITCH,
        h_grat=H_GRAT,
        duty_bot=DUTY_BOT,
        duty_top=DUTY_TOP,
        n_slab=N_SLAB,
    )
    poly_um = np.asarray(slab_poly.vertices) * 1e6
    y_emit_um = float(d2["y_emit_ez"]) * 1e6
    w_mon_um = float(d2["w_mon"]) * 1e6
    y_mon_um = float(d2["y_mon"]) * 1e6
    overlay_reciprocity_led_geometry(
        ax,
        poly_um,
        y_emit_um=y_emit_um,
        y_mon_um=y_mon_um,
        w_mon_um=w_mon_um,
        line_color="black",
        show_labels=True,
        compact=False,
    )
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-0.02, y_mon_um + 0.10)
    ax.set_xlabel(r"Lateral coordinate $x$ [$\mu\mathrm{m}$]")
    ax.set_ylabel(r"Vertical coordinate $y$ [$\mu\mathrm{m}$]")
    ax.set_title("(a) 2D Grating LED Geometry\n& Monitors")

    # Zoomed comparison of Kottke vs Staircase on one grating tooth
    x_zoom = (-0.28, 0.28)
    y_zoom = ((D_SLAB - H_GRAT - 0.03e-6) * 1e6, (D_SLAB + 0.03e-6) * 1e6)
    for ax_z, eps_arr, title_str in [
        (axes[1], d2["eps_kottke_x"], "(b) Kottke Anisotropic $\\varepsilon_{xx}$\n(Tooth Zoom)"),
        (axes[2], d2["eps_stair_x"], "(c) Staircase $\\varepsilon_{xx}$\n(Tooth Zoom)"),
    ]:
        eps_clean = np.where(np.isinf(eps_arr), np.nan, eps_arr)
        im_z = ax_z.pcolormesh(
            gx,
            gy,
            eps_clean.T,
            cmap="viridis",
            vmin=1.0,
            vmax=N_SLAB**2,
            shading="auto",
            edgecolors="0.85",
            linewidth=0.25,
        )
        ax_z.add_patch(MplPolygon(poly_um, closed=True, fill=False, edgecolor="white", lw=1.6, ls="--"))
        ax_z.set_xlim(*x_zoom)
        ax_z.set_ylim(*y_zoom)
        ax_z.set_xlabel(r"$x$ [$\mu\mathrm{m}$]")
        ax_z.set_ylabel(r"$y$ [$\mu\mathrm{m}$]")
        ax_z.set_title(title_str)
        fig.colorbar(im_z, ax=ax_z, label=r"$\varepsilon_{xx}$")

    figstyle.save(fig, fig_dir / "01_geometry_mesh.png")

    # ------------------------------------------------------------------
    # Figure 2: Reciprocal Time-Domain Wave Snapshots (1x3) & Animated GIF
    # ------------------------------------------------------------------
    frames = np.asarray(d2["frames_Ez"], dtype=float) * (float(d2["frames_Ez_scale"]) if "frames_Ez_scale" in d2 else 1.0)
    n_f = len(frames)
    pick_idx = [max(0, int(round(q * (n_f - 1)))) for q in (0.20, 0.50, 0.85)]
    vmin_f, vmax_f = figstyle.symmetric_limits(frames, percentile=99.2)

    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.2, height=4.2)
    for k, (ax, fi) in enumerate(zip(axes.flat, pick_idx)):
        im = ax.pcolormesh(
            gx,
            gy,
            frames[fi].T,
            cmap=figstyle.CMAP_DIVERGING,
            vmin=vmin_f,
            vmax=vmax_f,
            shading="auto",
        )
        overlay_reciprocity_led_geometry(
            ax,
            poly_um,
            y_emit_um=y_emit_um,
            y_mon_um=y_mon_um,
            w_mon_um=w_mon_um,
            line_color="black",
            show_labels=True,
            compact=True,
        )
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(0.0, y_mon_um + 0.08)
        ax.set_title(f"({chr(97 + k)}) Snapshot {k + 1}/3\n(frame {fi + 1}/{n_f}, $\\theta = 25^\\circ$)")
        ax.set_xlabel(r"$x$ [$\mu\mathrm{m}$]")
        ax.set_ylabel(r"$y$ [$\mu\mathrm{m}$]")
        fig.colorbar(im, ax=ax, label=r"$E_z(x, y, t)$ [a.u.]")

    figstyle.save(fig, fig_dir / "02_field_snapshots_and_movie.png")

    # Animated GIF of the reciprocal wave
    fig_gif, ax_gif = figstyle.new_figure(nrows=1, ncols=1, width=6.8, height=3.8)
    im_gif = ax_gif.pcolormesh(
        gx,
        gy,
        frames[0].T,
        cmap=figstyle.CMAP_DIVERGING,
        vmin=vmin_f,
        vmax=vmax_f,
        shading="auto",
    )
    overlay_reciprocity_led_geometry(
        ax_gif,
        poly_um,
        y_emit_um=y_emit_um,
        y_mon_um=y_mon_um,
        w_mon_um=w_mon_um,
        line_color="black",
        show_labels=True,
        compact=False,
    )
    ax_gif.set_xlim(-1.35, 1.35)
    ax_gif.set_ylim(0.0, y_mon_um + 0.08)
    ax_gif.set_xlabel(r"$x$ [$\mu\mathrm{m}$]")
    ax_gif.set_ylabel(r"$y$ [$\mu\mathrm{m}$]")
    title_gif = ax_gif.set_title("Reciprocal Wave Propagation ($\\theta = 25^\\circ$, frame 1)")
    fig_gif.colorbar(im_gif, ax=ax_gif, label=r"$E_z(x, y, t)$")
    figstyle.finish(fig_gif)

    def _update_gif(frame_i):
        im_gif.set_array(frames[frame_i].T.ravel())
        title_gif.set_text(f"Reciprocal Wave Propagation ($\\theta = 25^\\circ$, frame {frame_i + 1}/{n_f})")
        return (im_gif, title_gif)

    ani = animation.FuncAnimation(fig_gif, _update_gif, frames=n_f, interval=90, blit=False)
    ani.save(fig_dir / "02_reciprocal_wave_movie.gif", writer="pillow", fps=11, dpi=110)
    plt.close(fig_gif)

    # ------------------------------------------------------------------
    # Figure 3: Flat Slab Analytic TMM vs Wakis FIT vs Angle
    # ------------------------------------------------------------------
    th_bloch = d0["thetas_deg"]
    th_dense = np.linspace(0.0, 75.0, 301)
    th_dense_rad = np.deg2rad(th_dense)
    y_ez_b = float(d0["y_emit_ez"])
    y_ey_b = float(d0["y_emit_ey"])

    tmm_dense_tm = tmm_slab_emission(th_dense_rad, y_ez_b, D_FLAT_BENCH, N_SLAB, LAM0, pol="TM")
    tmm_dense_tex = tmm_slab_emission(th_dense_rad, y_ez_b, D_FLAT_BENCH, N_SLAB, LAM0, pol="TE_x")
    tmm_dense_tey = tmm_slab_emission(th_dense_rad, y_ey_b, D_FLAT_BENCH, N_SLAB, LAM0, pol="TE_y")
    tmm_dense_unpol = (tmm_dense_tm + tmm_dense_tex + tmm_dense_tey) / 3.0

    wakis_pec_unpol = (d0["I_tm_ang_pec"] + d0["I_tex_ang_pec"] + d0["I_tey_ang_pec"]) / 3.0

    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.3, gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    ax.plot(th_dense, tmm_dense_tm, "-", color="tab:blue", lw=2.0, label="Analytic TMM: TM ($J_z$, $s$-pol)")
    ax.plot(th_bloch, d0["I_tm_ang_pec"], "o", color="tab:blue", ms=6, label="Wakis FIT Reciprocity: TM ($J_z$)")

    ax.plot(th_dense, tmm_dense_tex, "-", color="tab:orange", lw=2.0, label="Analytic TMM: $\\mathrm{TE}_x$ ($J_x$, $p$-pol)")
    ax.plot(th_bloch, d0["I_tex_ang_pec"], "s", color="tab:orange", ms=5.5, label="Wakis FIT Reciprocity: $\\mathrm{TE}_x$ ($J_x$)")

    ax.plot(th_dense, tmm_dense_tey, "-", color="tab:green", lw=2.0, label="Analytic TMM: $\\mathrm{TE}_y$ ($J_y$, $p$-pol)")
    ax.plot(th_bloch, d0["I_tey_ang_pec"], "^", color="tab:green", ms=6, label="Wakis FIT Reciprocity: $\\mathrm{TE}_y$ ($J_y$)")

    ax.plot(th_dense, tmm_dense_unpol, "k--", lw=2.0, label="Analytic TMM: Unpolarized $(J_x+J_y+J_z)/3$")
    ax.plot(th_bloch, wakis_pec_unpol, "kd", ms=5.5, label="Wakis FIT: Unpolarized")

    ax.set_xlabel(r"Observation Angle $\theta$ in Air [deg]")
    ax.set_ylabel(r"Normalized Intensity $I_\alpha(\theta)/U_{\mathrm{free,2D}} = |E_{\mathrm{rec},\alpha}/E_{\mathrm{inc}}|^2$")
    ax.set_title(f"(a) Flat GaN Slab ($d={D_FLAT_BENCH*1e9:.0f}\\,\\mathrm{{nm}}, y_0={y_ez_b*1e9:.0f}\\,\\mathrm{{nm}}$) vs Exact TMM")
    ax.legend(loc="upper right", ncol=1)

    ax2 = axes[1]
    ax2.plot(th_bloch, np.abs(d0["I_tm_ang_pec"] - d0["I_tm_ana_pec"]), "o-", color="tab:blue", label="|$\\Delta I_{\\mathrm{TM}}|$ ($J_z$)")
    ax2.plot(th_bloch, np.abs(d0["I_tex_ang_pec"] - d0["I_tex_ana_pec"]), "s-", color="tab:orange", label="|$\\Delta I_{\\mathrm{TE},x}|$ ($J_x$)")
    ax2.plot(th_bloch, np.abs(d0["I_tey_ang_pec"] - d0["I_tey_ana_pec"]), "^-", color="tab:green", label="|$\\Delta I_{\\mathrm{TE},y}|$ ($J_y$)")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Observation Angle $\theta$ [deg]")
    ax2.set_ylabel(r"Absolute Pointwise Error $|I_{\mathrm{Wakis}} - I_{\mathrm{TMM}}|$")
    ax2.set_title("(b) Pointwise Error vs Analytic Transfer Matrix")
    ax2.legend(loc="best")

    figstyle.save(fig, fig_dir / "03_flat_slab_tmm_vs_wakis_angle.png")

    # ------------------------------------------------------------------
    # Figure 4: Flat Slab Depth Standing-Wave Profile vs Emitter Height y0
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.3, sharey=True)
    y_nm_ez = d0["y_ez_pec"] * 1e9
    y_nm_ey = d0["y_ey_pec"] * 1e9

    ax = axes[0]
    ax.plot(y_nm_ez, d0["I_tm_prof_pec0_ana"], "-", color="tab:blue", lw=2.2, label="TMM Exact ($\\theta=0^\\circ$, TM & $\\mathrm{TE}_x$)")
    ax.plot(y_nm_ez, d0["I_tm_prof_pec0"], "o", color="tab:blue", ms=5, label="Wakis FIT $|E_z(y_0)/E_{\\mathrm{inc}}|^2$")
    ax.axvline(y_ez_b * 1e9, color="crimson", ls="--", lw=1.6, label=f"Design QW $y_0 = {y_ez_b*1e9:.0f}\\,\\mathrm{{nm}}$")
    ax.axvline(0.5 * (LAM0 / N_SLAB) * 1e9, color="gray", ls=":", lw=1.6, label=f"Standing-Wave Node ($\\lambda_{{GaN}}/2 = {0.5*(LAM0/N_SLAB)*1e9:.1f}\\,\\mathrm{{nm}}$)")
    ax.set_xlabel(r"Emitter Height Above PEC Mirror $y_0$ [nm]")
    ax.set_ylabel(r"Normalized Intensity $|E_{\mathrm{rec}}(y_0)/E_{\mathrm{inc}}|^2$")
    ax.set_title(r"(a) Normal Incidence ($\theta = 0^\circ$) Standing Wave")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(y_nm_ez, d0["I_tm_prof_ana25"], "-", color="tab:blue", lw=2.0, label="TMM Exact: TM ($|E_z|^2$)")
    ax.plot(y_nm_ez, d0["I_tm_prof_pec25"], "o", color="tab:blue", ms=5, label="Wakis FIT: TM ($|E_z|^2$)")
    ax.plot(y_nm_ez, d0["I_tex_prof_ana25"], "-", color="tab:orange", lw=2.0, label="TMM Exact: $\\mathrm{TE}_x$ ($|E_x|^2$)")
    ax.plot(y_nm_ez, d0["I_tex_prof_pec25"], "s", color="tab:orange", ms=5, label="Wakis FIT: $\\mathrm{TE}_x$ ($|E_x|^2$)")
    ax.plot(y_nm_ey, d0["I_tey_prof_ana25"], "-", color="tab:green", lw=2.0, label="TMM Exact: $\\mathrm{TE}_y$ ($|E_y|^2$)")
    ax.plot(y_nm_ey, d0["I_tey_prof_pec25"], "^", color="tab:green", ms=5, label="Wakis FIT: $\\mathrm{TE}_y$ ($|E_y|^2$)")
    ax.axvline(y_ez_b * 1e9, color="crimson", ls="--", lw=1.5, label=f"QW $y_0 = {y_ez_b*1e9:.0f}\\,\\mathrm{{nm}}$")
    ax.set_xlabel(r"Emitter Height Above PEC Mirror $y_0$ [nm]")
    ax.set_title(r"(b) Oblique Incidence ($\theta = 25^\circ$) Depth Profiles")
    ax.legend(loc="upper right")

    figstyle.save(fig, fig_dir / "04_flat_slab_depth_standing_wave.png")

    # ------------------------------------------------------------------
    # Figure 5: Realistic Drude Silver (Ag) Mirror vs Ideal PEC Mirror
    # ------------------------------------------------------------------
    _, eps_ag_val, skin_depth_val = make_drude_silver(LAM0)
    tmm_dense_ag_tm = tmm_slab_emission(th_dense_rad, y_ez_b, D_FLAT_BENCH, N_SLAB, LAM0, pol="TM", eps_mirror=eps_ag_val)
    tmm_dense_ag_unpol = tmm_slab_emission(th_dense_rad, y_ez_b, D_FLAT_BENCH, N_SLAB, LAM0, pol="unpolarized", eps_mirror=eps_ag_val)

    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.3)
    ax = axes[0]
    y_ag_nm = d0["y_ez_ag"] * 1e9
    ax.plot(y_nm_ez, d0["I_tm_prof_pec0_ana"], "k--", lw=1.8, label="PEC Mirror TMM ($r_s = -1$, $\\theta=0^\\circ$)")
    ax.plot(y_ag_nm, d0["I_tm_prof_ag0_ana"], "-", color="tab:purple", lw=2.2, label="Drude Ag TMM ($\\varepsilon_{\\mathrm{Ag}} = -7.08 - 0.29j$, $\\theta=0^\\circ$)")
    ax.plot(y_ag_nm, d0["I_tm_prof_ag0"], "o", color="tab:purple", ms=5.0, label="Wakis FIT Drude Ag ($\\theta=0^\\circ$)")
    ax.plot(y_ag_nm, d0["I_tm_prof_ag25_ana"], "-", color="teal", lw=2.0, label="Drude Ag TMM ($\\theta=25^\\circ$)")
    ax.plot(y_ag_nm, d0["I_tm_prof_ag25"], "s", color="teal", ms=4.8, label="Wakis FIT Drude Ag ($\\theta=25^\\circ$)")
    ax.axvline(y_ez_b * 1e9, color="crimson", ls=":", lw=1.5, label=f"QW plane $y_0 = {y_ez_b*1e9:.0f}\\,\\mathrm{{nm}}$ ($\\delta_s = {float(d0['skin_depth_nm']):.1f}\\,\\mathrm{{nm}}$)")
    ax.set_xlabel(r"Emitter Height Above Mirror $y_0$ [nm]")
    ax.set_ylabel(r"Normalized Standing-Wave Intensity $|E_z(y_0)/E_{\mathrm{inc}}|^2$")
    ax.set_title(r"(a) Phase Shift & Attenuation on Drude Silver vs PEC Mirror")
    ax.legend(loc="upper right", fontsize=8)

    ax2 = axes[1]
    ax2.plot(th_dense, tmm_dense_tm, "k--", lw=1.8, label="PEC Mirror TMM: TM ($J_z$)")
    ax2.plot(th_dense, tmm_dense_ag_tm, "-", color="tab:purple", lw=2.0, label="Drude Ag TMM: TM ($J_z$)")
    ax2.plot(th_bloch, d0["I_tm_ang_ag"], "o", color="tab:purple", ms=5.5, label="Wakis FIT Drude Ag: TM ($J_z$)")

    ax2.plot(th_dense, tmm_dense_unpol, "k:", lw=1.8, label="PEC Mirror TMM: Unpolarized")
    ax2.plot(th_dense, tmm_dense_ag_unpol, "-", color="teal", lw=2.0, label="Drude Ag TMM: Unpolarized")

    ax2.set_xlabel(r"Observation Angle $\theta$ [deg]")
    ax2.set_ylabel(r"Normalized Angular Intensity $I(\theta)/U_{\mathrm{free,2D}}$")
    ax2.set_title(f"(b) Angular Emission at $y_0 = {y_ez_b*1e9:.0f}\\,\\mathrm{{nm}}$ (Ag vs PEC)")
    ax2.legend(loc="upper right", fontsize=8)

    figstyle.save(fig, fig_dir / "05_silver_vs_pec_mirror.png")

    # ------------------------------------------------------------------
    # Figure 6: Point-by-Point Reciprocity across the Grating Active Layer
    # ------------------------------------------------------------------
    i_th0 = int(d1["i_th0"])
    i_th25 = int(d1["i_th25"])
    th0_val = float(d1["thetas_deg"][i_th0])
    th25_val = float(d1["thetas_deg"][i_th25])

    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.8, height=4.4, sharey=False)
    for ax, ith, th_val, panel in [
        (axes[0], i_th0, th0_val, "(a)"),
        (axes[1], i_th25, th25_val, "(b)"),
    ]:
        ax.plot(
            d1["x_ez"] * 1e6,
            d1["I_tm_rec_full"][ith],
            "-",
            color="tab:blue",
            lw=2.0,
            label="Method (B) Reciprocal $|E_z(x_0;\\theta)/E_{\\mathrm{inc}}|^2$",
        )
        ax.plot(
            d1["x_dip_z"] * 1e6,
            d1["I_dip_z"][:, ith],
            "o",
            color="tab:blue",
            ms=6.5,
            mfc="none",
            mew=1.8,
            label="Method (A) Direct $J_z$ Dipoles $U_z(\\theta;x_0)/U_{\\mathrm{free}}$",
        )

        ax.plot(
            d1["x_ex"] * 1e6,
            d1["I_tex_rec_full"][ith],
            "-",
            color="tab:orange",
            lw=2.0,
            label="Method (B) Reciprocal $|E_x(x_0;\\theta)/E_{\\mathrm{inc}}|^2$",
        )
        ax.plot(
            d1["x_dip_x"] * 1e6,
            d1["I_dip_x"][:, ith],
            "s",
            color="tab:orange",
            ms=6.0,
            mfc="none",
            mew=1.8,
            label="Method (A) Direct $J_x$ Dipoles $U_x(\\theta;x_0)/U_{\\mathrm{free}}$",
        )

        ax.plot(
            d1["x_ey"] * 1e6,
            d1["I_tey_rec_full"][ith],
            "-",
            color="tab:green",
            lw=2.0,
            label="Method (B) Reciprocal $|E_y(x_0;\\theta)/E_{\\mathrm{inc}}|^2$",
        )
        ax.plot(
            d1["x_dip_y"] * 1e6,
            d1["I_dip_y"][:, ith],
            "^",
            color="tab:green",
            ms=6.5,
            mfc="none",
            mew=1.8,
            label="Method (A) Direct $J_y$ Dipoles $U_y(\\theta;x_0)/U_{\\mathrm{free}}$",
        )

        for m_p in (-1, 0, 1):
            xc = m_p * PITCH * 1e6
            w_tooth = 0.5 * (DUTY_BOT + DUTY_TOP) * PITCH * 1e6
            ax.axvspan(xc - 0.5 * w_tooth, xc + 0.5 * w_tooth, color="gray", alpha=0.14)

        ax.set_xlabel(r"Dipole Lateral Position $x_0$ across 3 Grating Periods [$\mu\mathrm{m}$]")
        ax.set_ylabel(r"Normalized Emission Intensity $I_\alpha(\theta; x_0) / U_{\mathrm{free,2D}}$")
        ax.set_title(f"{panel} Point-by-Point Helmholtz Reciprocity at $\\theta = {th_val:+.1f}^\\circ$")
        if ith == i_th0:
            ax.legend(loc="upper right", fontsize=7.8)

    figstyle.save(fig, fig_dir / "06_reciprocity_point_by_point.png")

    # ------------------------------------------------------------------
    # Figure 7: Ensemble Angular Pattern: Method (A) Direct vs Method (B) Reciprocity
    # ------------------------------------------------------------------
    th_sym = d1["thetas_deg"]
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.4)

    ax = axes[0]
    ax.plot(th_sym, d1["rec_sample_z"], "-", color="tab:blue", lw=2.0, label="Method (B) Reciprocity: TM ($J_z$)")
    ax.plot(th_sym, d1["dir_incoh_z"], "o", color="tab:blue", ms=5.5, mfc="none", mew=1.6, label="Method (A) Direct Incoherent Sum: $J_z$")

    ax.plot(th_sym, d1["rec_sample_x"], "-", color="tab:orange", lw=2.0, label="Method (B) Reciprocity: $\\mathrm{TE}_x$ ($J_x$)")
    ax.plot(th_sym, d1["dir_incoh_x"], "s", color="tab:orange", ms=5.5, mfc="none", mew=1.6, label="Method (A) Direct Incoherent Sum: $J_x$")

    ax.plot(th_sym, d1["rec_sample_y"], "-", color="tab:green", lw=2.0, label="Method (B) Reciprocity: $\\mathrm{TE}_y$ ($J_y$)")
    ax.plot(th_sym, d1["dir_incoh_y"], "^", color="tab:green", ms=5.5, mfc="none", mew=1.6, label="Method (A) Direct Incoherent Sum: $J_y$")

    ax.set_xlabel(r"Observation Angle $\theta$ [deg]")
    ax.set_ylabel(r"Position-Averaged Intensity $\langle I_\alpha(\theta)\rangle_x / U_{\mathrm{free,2D}}$")
    ax.set_title("(a) Polarization-Resolved Grating LED Emission: (A) vs (B)")
    ax.legend(loc="upper right", fontsize=8)

    ax2 = axes[1]
    rec_sample_unpol = (d1["rec_sample_z"] + d1["rec_sample_x"] + d1["rec_sample_y"]) / 3.0
    ax2.plot(th_sym, d1["rec_cont_unpol"], "k-", lw=2.2, label="Method (B) Continuous Active-Layer Integral")
    ax2.plot(th_sym, rec_sample_unpol, "--", color="crimson", lw=1.8, label=f"Method (B) Sampled at $N_{{\\mathrm{{pos}}}}={len(d1['x_dip_z'])}$ Sites")
    ax2.plot(th_sym, d1["dir_incoh_unpol"], "ko", ms=5.5, mfc="none", mew=1.6, label=f"Method (A) Direct Sum ($3 \\times {len(d1['x_dip_z'])} = {int(d1['n_dir_runs'])}$ runs)")
    ax2.set_xlabel(r"Observation Angle $\theta$ [deg]")
    ax2.set_ylabel(r"Unpolarized Intensity $\langle I_{\mathrm{unpol}}(\theta)\rangle_x / U_{\mathrm{free,2D}}$")
    ax2.set_title("(b) Isotropic 3D Unpolarized Ensemble Average")
    ax2.legend(loc="upper right")

    figstyle.save(fig, fig_dir / "07_ensemble_direct_vs_reciprocity.png")

    # ------------------------------------------------------------------
    # Figure 8: Incoherent Power Average vs Coherent Field Sum (Failure Mode)
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.4)
    ax = axes[0]
    n_dips = d1["I_dip_z"].shape[0]
    for ip in range(0, n_dips, max(1, n_dips // 4)):
        x_um = float(d1["x_dip_z"][ip]) * 1e6
        ax.plot(th_sym, d1["I_dip_z"][ip], lw=1.1, alpha=0.55, label=f"Single $J_z$ dipole at $x_0 = {x_um:+.2f}\\,\\mu\\mathrm{{m}}$")
    ax.plot(th_sym, d1["dir_incoh_z"], "k-", lw=2.5, label="Incoherent Power Mean $\\frac{1}{N}\\sum_m |E_{\\mathrm{ff},m}(\\theta)|^2$")
    ax.set_xlabel(r"Observation Angle $\theta$ [deg]")
    ax.set_ylabel(r"Normalized TM Intensity $I_z(\theta) / U_{\mathrm{free,2D}}$")
    ax.set_title("(a) Individual Dipoles vs Smooth Incoherent Ensemble Mean")
    ax.legend(loc="upper right", fontsize=8)

    ax2 = axes[1]
    ax2.plot(th_sym, d1["dir_incoh_z"], "k-", lw=2.3, label="Physical Incoherent Sum (Powers Averaged)")
    ax2.plot(th_sym, d1["dir_coh_tm"], "r--", lw=2.0, label="Unphysical Coherent Sum $|\\frac{1}{N}\\sum_m E_{\\mathrm{ff},m}(\\theta)|^2$ (Grating)")
    ax2.plot(th_sym, d1["dir_flat_coh_tm"], "b:", lw=1.8, label="Unphysical Coherent Sum (Flat Slab Phased Array)")
    ax2.set_xlabel(r"Observation Angle $\theta$ [deg]")
    ax2.set_ylabel(r"Normalized Intensity $I_z(\theta) / U_{\mathrm{free,2D}}$")
    ax2.set_title("(b) Pitfall: Coherent Field Sum Creates False Phased-Array Lobe")
    ax2.legend(loc="upper right", fontsize=8)

    figstyle.save(fig, fig_dir / "08_incoherent_vs_coherent_sum.png")

    # ------------------------------------------------------------------
    # Figure 9: Dipole-Count Convergence (Spatial Sampling vs Continuous Integral)
    # ------------------------------------------------------------------
    I_full_z = d1["I_tm_rec_full"]
    I_full_x = d1["I_tex_rec_full"]
    I_full_y = d1["I_tey_rec_full"]
    n_fine = min(I_full_z.shape[1], I_full_x.shape[1], I_full_y.shape[1])
    I_full_z = I_full_z[:, :n_fine]
    I_full_x = I_full_x[:, :n_fine]
    I_full_y = I_full_y[:, :n_fine]
    ref_curve = (np.mean(I_full_z, axis=1) + np.mean(I_full_x, axis=1) + np.mean(I_full_y, axis=1)) / 3.0

    n_samples_list = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48])
    n_samples_list = n_samples_list[n_samples_list <= n_fine]
    err_reg = []
    err_mc = []
    rng = np.random.default_rng(42)
    for ns in n_samples_list:
        idx_reg = np.round(np.linspace(0, n_fine - 1, ns)).astype(int)
        c_reg = (
            np.mean(I_full_z[:, idx_reg], axis=1)
            + np.mean(I_full_x[:, idx_reg], axis=1)
            + np.mean(I_full_y[:, idx_reg], axis=1)
        ) / 3.0
        err_reg.append(float(np.linalg.norm(c_reg - ref_curve) / np.linalg.norm(ref_curve)))

        mc_trials = []
        for _ in range(40):
            idx_mc = rng.integers(0, n_fine, size=ns)
            c_mc = (
                np.mean(I_full_z[:, idx_mc], axis=1)
                + np.mean(I_full_x[:, idx_mc], axis=1)
                + np.mean(I_full_y[:, idx_mc], axis=1)
            ) / 3.0
            mc_trials.append(np.linalg.norm(c_mc - ref_curve) / np.linalg.norm(ref_curve))
        err_mc.append(float(np.mean(mc_trials)))

    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.3)
    ax = axes[0]
    ax.loglog(n_samples_list, err_mc, "o-", color="tab:red", lw=2.0, label="Random Dipole Realizations (Monte Carlo)")
    ax.loglog(n_samples_list, err_reg, "s-", color="tab:blue", lw=2.0, label="Uniform Grid Sampling across 3 Periods")
    ax.loglog(
        n_samples_list,
        err_mc[0] / np.sqrt(n_samples_list),
        "r--",
        lw=1.3,
        label=r"Monte Carlo $O(N_{\mathrm{pos}}^{-1/2})$ reference",
    )
    ax.loglog(
        n_samples_list,
        err_reg[0] / (n_samples_list**1.0),
        "b:",
        lw=1.3,
        label=r"Quadrature $O(N_{\mathrm{pos}}^{-1})$ reference",
    )
    ax.axhline(0.02, color="gray", ls="--", lw=1.0, label="2% Relative L2 Threshold")
    ax.set_xlabel(r"Number of Dipole Positions Sampled $N_{\mathrm{pos}}$")
    ax.set_ylabel(r"Relative Angular L2 Error vs Continuous Integral")
    ax.set_title("(a) Convergence of Ensemble Average with Dipole Count $N_{\\mathrm{pos}}$")
    ax.legend(loc="upper right", fontsize=8)

    ax2 = axes[1]
    for ns, col, ls in [(1, "tab:gray", ":"), (3, "tab:orange", "--"), (9, "tab:blue", "-.")]:
        idx_sub = np.round(np.linspace(0, n_fine - 1, min(ns, n_fine))).astype(int)
        curve_sub = (
            np.mean(I_full_z[:, idx_sub], axis=1)
            + np.mean(I_full_x[:, idx_sub], axis=1)
            + np.mean(I_full_y[:, idx_sub], axis=1)
        ) / 3.0
        ax2.plot(th_sym, curve_sub, color=col, ls=ls, lw=1.8, label=f"$N_{{\\mathrm{{pos}}}} = {ns}$ ($3 N_{{\\mathrm{{pos}}}} = {3*ns}$ runs)")
    ax2.plot(th_sym, ref_curve, "k-", lw=2.3, label=f"Continuous Reciprocal Integral ($N_x = {n_fine}$ cells)")
    ax2.set_xlabel(r"Observation Angle $\theta$ [deg]")
    ax2.set_ylabel(r"Unpolarized Intensity $\langle I(\theta)\rangle_x / U_{\mathrm{free,2D}}$")
    ax2.set_title("(b) Angular Pattern Stabilization with Increasing $N_{\\mathrm{pos}}$")
    ax2.legend(loc="upper right")

    figstyle.save(fig, fig_dir / "09_dipole_count_convergence.png")

    # ------------------------------------------------------------------
    # Figure 10: Grating LED vs Flat Slab Extraction Enhancement (Cartesian + Polar)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(11.8, 4.6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")

    th_rad_sym = np.deg2rad(th_sym)
    tmm_sym_unpol = tmm_slab_emission(th_rad_sym, float(d1["y_emit_ez"]), D_SLAB, N_SLAB, LAM0, pol="unpolarized")
    tmm_sym_tm = tmm_slab_emission(th_rad_sym, float(d1["y_emit_ez"]), D_SLAB, N_SLAB, LAM0, pol="TM")

    P_grat_u = float(np.trapezoid(d1["rec_cont_unpol"], th_rad_sym))
    P_flat_u = float(np.trapezoid(d1["rec_flat_unpol"], th_rad_sym))
    enh_u = P_grat_u / P_flat_u

    P_grat_tm = float(np.trapezoid(d1["rec_cont_z"], th_rad_sym))
    P_flat_tm = float(np.trapezoid(d1["rec_flat_z"], th_rad_sym))
    enh_tm = P_grat_tm / P_flat_tm

    ax1.plot(th_sym, d1["rec_cont_z"], "-", color="tab:blue", lw=2.2, label=f"Grating LED: TM ($J_z$, $\\times{enh_tm:.2f}$ integrated)")
    ax1.plot(th_sym, d1["rec_flat_z"], "--", color="tab:blue", lw=1.7, label="Flat LED (Wakis FIT): TM ($J_z$)")
    ax1.plot(th_sym, tmm_sym_tm, ":", color="navy", lw=1.5, label="Flat LED (Analytic TMM): TM")

    ax1.plot(th_sym, d1["rec_cont_unpol"], "-", color="crimson", lw=2.2, label=f"Grating LED: Unpolarized ($\\times{enh_u:.2f}$ integrated)")
    ax1.plot(th_sym, d1["rec_flat_unpol"], "--", color="crimson", lw=1.7, label="Flat LED (Wakis FIT): Unpolarized")
    ax1.plot(th_sym, tmm_sym_unpol, "k:", lw=1.5, label="Flat LED (Analytic TMM): Unpolarized")

    ax1.set_xlabel(r"Observation Angle $\theta$ [deg]")
    ax1.set_ylabel(r"Normalized Far-Field Intensity $I(\theta)/U_{\mathrm{free,2D}}$")
    ax1.set_title("(a) Bragg Extraction of Guided Modes: Grating vs Flat LED")
    ax1.legend(loc="upper right", fontsize=7.8)

    # Polar radiation plot (upper hemisphere -90 to +90 deg)
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_thetamin(-80)
    ax2.set_thetamax(80)
    ax2.plot(th_rad_sym, d1["rec_cont_z"], "-", color="tab:blue", lw=2.2, label="Grating TM ($J_z$)")
    ax2.plot(th_rad_sym, d1["rec_flat_z"], "--", color="tab:blue", lw=1.6, label="Flat TM ($J_z$)")
    ax2.plot(th_rad_sym, d1["rec_cont_unpol"], "-", color="crimson", lw=2.2, label="Grating Unpolarized")
    ax2.plot(th_rad_sym, d1["rec_flat_unpol"], "--", color="crimson", lw=1.6, label="Flat Unpolarized")
    ax2.set_title("(b) Upper-Hemisphere Polar Radiation Pattern", pad=14)
    ax2.legend(loc="lower left", bbox_to_anchor=(-0.12, -0.15), fontsize=8)
    ax2.add_collection(PolyCollection([[(0.0, 0.0), (0.0, 0.0)]], alpha=0.0))

    figstyle.save(fig, fig_dir / "10_grating_vs_flat_extraction.png")

    # ------------------------------------------------------------------
    # Figure 11 (1x3) & Figure 11b (1x2): 2D Spatial Maps of Reciprocal vs Direct Fields
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.2, height=4.2)
    x_ez_um = d2["dft25_x_ez"] * 1e6
    y_ez_um = d2["dft25_y_ez"] * 1e6
    x_ey_um = d2["dft25_x_ey"] * 1e6
    y_ey_um = d2["dft25_y_ey"] * 1e6

    panels_11 = [
        (axes[0], x_ez_um, y_ez_um, np.abs(d2["dft0_Ez"]) ** 2, "(a) Reciprocal $|E_z(x,y)/E_{\\mathrm{inc}}|^2$\nat $\\theta = 0^\\circ$ (TM)", figstyle.CMAP_MAGNITUDE, 3.2),
        (axes[1], x_ez_um, y_ez_um, np.abs(d2["dft25_Ez"]) ** 2, "(b) Reciprocal $|E_z(x,y)/E_{\\mathrm{inc}}|^2$\nat $\\theta = 25^\\circ$ (TM)", figstyle.CMAP_MAGNITUDE, 3.2),
        (axes[2], x_ey_um, y_ey_um, np.abs(d2["dft25_Ey"]) ** 2, "(c) Reciprocal $|E_y(x,y)/E_{\\mathrm{inc}}|^2$\nat $\\theta = 25^\\circ$ ($\\mathrm{TE}_y$)", figstyle.CMAP_MAGNITUDE, 1.2),
    ]
    for ax, xx, yy, field2d, t_str, cmap_name, vmax_val in panels_11:
        im = ax.pcolormesh(xx, yy, field2d.T, cmap=cmap_name, vmin=0.0, vmax=vmax_val, shading="auto")
        overlay_reciprocity_led_geometry(
            ax,
            poly_um,
            y_emit_um=y_emit_um,
            y_mon_um=y_mon_um,
            w_mon_um=w_mon_um,
            line_color="white",
            show_labels=True,
            compact=True,
        )
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(0.0, y_mon_um + 0.05)
        ax.set_xlabel(r"$x$ [$\mu\mathrm{m}$]")
        ax.set_ylabel(r"$y$ [$\mu\mathrm{m}$]")
        ax.set_title(t_str)
        fig.colorbar(im, ax=ax, label=r"$|E_{\mathrm{rec}}/E_{\mathrm{inc}}|^2$")

    figstyle.save(fig, fig_dir / "11_reciprocal_2d_field_maps.png")

    # Companion 2-panel Figure 11b: Forward Single-Dipole Field vs Reciprocal Plane-Wave Field
    fig_11b, axes_11b = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.2)
    dip_Ez = np.abs(d1["dip_snap_Ez"])
    dip_Ez_norm = dip_Ez / (np.percentile(dip_Ez, 99.0) or 1.0)
    im_d = axes_11b[0].pcolormesh(
        d1["dip_snap_x"] * 1e6,
        d1["dip_snap_y"] * 1e6,
        dip_Ez_norm.T,
        cmap=figstyle.CMAP_MAGNITUDE,
        vmin=0.0,
        vmax=1.0,
        shading="auto",
    )
    overlay_reciprocity_led_geometry(
        axes_11b[0],
        poly_um,
        y_emit_um=y_emit_um,
        y_mon_um=y_mon_um,
        w_mon_um=w_mon_um,
        line_color="white",
        show_dipole=(float(d1["dip_snap_xdip"]) * 1e6, float(d1["dip_snap_ydip"]) * 1e6),
        show_labels=True,
        compact=False,
    )
    axes_11b[0].set_xlim(-1.25, 1.25)
    axes_11b[0].set_ylim(0.0, y_mon_um + 0.05)
    axes_11b[0].set_xlabel(r"$x$ [$\mu\mathrm{m}$]")
    axes_11b[0].set_ylabel(r"$y$ [$\mu\mathrm{m}$]")
    axes_11b[0].set_title("(a) Direct Single-Dipole Field $|E_z(x,y)|$\n(1 of $3 N_{\\mathrm{sites}}$ Solves)")
    fig_11b.colorbar(im_d, ax=axes_11b[0], label=r"Normalized $|E_z(x,y)|$")

    im_r = axes_11b[1].pcolormesh(
        x_ez_um,
        y_ez_um,
        (np.abs(d2["dft0_Ez"]) ** 2).T,
        cmap=figstyle.CMAP_MAGNITUDE,
        vmin=0.0,
        vmax=3.2,
        shading="auto",
    )
    overlay_reciprocity_led_geometry(
        axes_11b[1],
        poly_um,
        y_emit_um=y_emit_um,
        y_mon_um=y_mon_um,
        w_mon_um=w_mon_um,
        line_color="white",
        show_labels=True,
        compact=False,
    )
    axes_11b[1].set_xlim(-1.25, 1.25)
    axes_11b[1].set_ylim(0.0, y_mon_um + 0.05)
    axes_11b[1].set_xlabel(r"$x$ [$\mu\mathrm{m}$]")
    axes_11b[1].set_ylabel(r"$y$ [$\mu\mathrm{m}$]")
    axes_11b[1].set_title("(b) Reciprocal Plane-Wave Intensity $|E_z(x,y)/E_{\\mathrm{inc}}|^2$\n(All $x_0$ in 1 Solve at $\\theta = 0^\\circ$)")
    fig_11b.colorbar(im_r, ax=axes_11b[1], label=r"$|E_{\mathrm{rec}}/E_{\mathrm{inc}}|^2$")

    figstyle.save(fig_11b, fig_dir / "11b_forward_vs_reciprocal_field_map.png")

    # ------------------------------------------------------------------
    # Figure 12: Computational Cost Crossover (Method A vs Method B)
    # ------------------------------------------------------------------
    t_rec_1 = float(d1["time_per_angle_rec"])
    t_dip_1 = float(d1["time_per_dipole_run"])

    n_theta_grid = np.arange(1, 65)
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.3)
    ax = axes[0]
    ax.plot(
        n_theta_grid,
        n_theta_grid * t_rec_1,
        "-",
        color="tab:blue",
        lw=2.4,
        label="Method (B) Reciprocity: $N_{\\mathrm{runs}} = N_\\theta$ (all $(x_0, y_0)$ free!)",
    )
    for n_pos_case, n_y_planes, col, ls in [
        (7, 1, "tab:orange", "--"),
        (25, 1, "tab:red", "-."),
        (25, 3, "purple", ":"),
    ]:
        n_runs_a = 3 * n_pos_case * n_y_planes
        cost_a = np.full_like(n_theta_grid, n_runs_a * t_dip_1, dtype=float)
        n_cross = n_runs_a * (t_dip_1 / t_rec_1)
        ax.plot(
            n_theta_grid,
            cost_a,
            color=col,
            ls=ls,
            lw=2.0,
            label=f"Method (A) Direct: $3 \\times {n_pos_case} \\times {n_y_planes} = {n_runs_a}$ runs ($N_\\theta^* \\approx {n_cross:.0f}$)",
        )
        if n_cross <= 64:
            ax.plot(n_cross, n_runs_a * t_dip_1, "ko", ms=6)

    ax.set_xlabel(r"Number of Far-Field Observation Angles $N_\theta$ (Symmetry-Reduced)")
    ax.set_ylabel("Total Wall-Clock Time on 2D Grid [s]")
    ax.set_title("(a) Measured 2D Cost Crossover: Direct Sum vs Reciprocity")
    ax.legend(loc="upper left", fontsize=8)

    ax2 = axes[1]
    n_sites_arr = np.logspace(1, 4, 50)
    ax2.loglog(n_sites_arr, 3.0 * n_sites_arr, "r-", lw=2.3, label="Method (A) Direct Dipole Runs: $N_{\\mathrm{runs}} = 3 N_{\\mathrm{sites}}$")
    ax2.axhline(11, color="tab:blue", ls="--", lw=2.0, label="Method (B) 2D Reciprocity ($N_\\theta = 11$ angles, 1 pol/solve)")
    ax2.axhline(90, color="navy", ls="-.", lw=2.0, label="Method (B) 3D Reciprocity ($45$ $(\\theta,\\phi)$ dirs $\\times 2$ pols $= 90$ runs)")
    ax2.axvline(30, color="gray", ls=":", lw=1.4, label="3D Crossover ($N_{\\mathrm{sites}} = 30$ emitters)")
    ax2.set_xlabel(r"Number of Emitter Grid Sites $N_{\mathrm{sites}} = N_x N_y (N_z)$ in Active Layer")
    ax2.set_ylabel(r"Required Number of Time-Domain Solves $N_{\mathrm{runs}}$")
    ax2.set_title("(b) Scaling with Active-Volume Sites (2D & 3D LEDs)")
    ax2.legend(loc="upper left", fontsize=8)

    figstyle.save(fig, fig_dir / "12_cost_crossover_scaling.png")

    # ------------------------------------------------------------------
    # Figure 13: Resolution Convergence (Angular Patterns, Integrated Power, & L2 Errors)
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.8, height=4.4)

    # Extract Kottke, Snapped Staircase, and Un-snapped Staircase series from tasks 2..8
    kottke_pts = []
    stair_snap_pts = []
    stair_unsnap_pts = []
    for idx, (npz_i, js_i) in sorted(res_tasks.items()):
        c_val = int(js_i["cpw"])
        dx_val = float(js_i["dx_nm"])
        th_val = np.asarray(npz_i["thetas_half"])
        I_val = np.asarray(npz_i["I_unpol_mean"])
        P_val = float(js_i["P_int_unpol_rad"])
        if js_i["averaging"] == "kottke":
            kottke_pts.append((c_val, dx_val, th_val, I_val, P_val))
        else:
            stair_snap_pts.append((c_val, dx_val, th_val, I_val, P_val))
            if "I_unpol_unsnapped_mean" in npz_i:
                I_u = np.asarray(npz_i["I_unpol_unsnapped_mean"])
                P_u = float(
                    js_i.get(
                        "P_int_unpol_unsnapped_rad",
                        2.0 * float(np.trapezoid(I_u, np.deg2rad(th_val))),
                    )
                )
                dx_u = float(js_i.get("dx_unsnapped_nm", dx_val))
                stair_unsnap_pts.append((c_val, dx_u, th_val, I_u, P_u))
            else:
                stair_unsnap_pts.append((c_val, dx_val, th_val, I_val, P_val))

    kottke_pts.sort(key=lambda x: x[0])
    stair_snap_pts.sort(key=lambda x: x[0])
    stair_unsnap_pts.sort(key=lambda x: x[0])

    # Finest Kottke run (cpw=22 in quick / cpw=40 in full) is our reference for the textured grating LED
    ref_cpw, ref_dx, ref_th, ref_I, ref_P = kottke_pts[-1]

    def _compute_errs(pts):
        cpws = []
        rel_P_errs = []
        rel_L2_errs = []
        P_vals = []
        for c_i, dx_i, th_i, I_i, P_i in pts:
            cpws.append(c_i)
            P_vals.append(P_i)
            I_interp = np.interp(ref_th, th_i, I_i)
            rel_P_errs.append(abs(P_i - ref_P) / ref_P)
            rel_L2_errs.append(float(np.linalg.norm(I_interp - ref_I) / np.linalg.norm(ref_I)))
        return np.array(cpws), np.array(P_vals), np.array(rel_P_errs), np.array(rel_L2_errs)

    cpw_k, P_k, errP_k, errL2_k = _compute_errs(kottke_pts)
    cpw_ss, P_ss, errP_ss, errL2_ss = _compute_errs(stair_snap_pts)
    cpw_su, P_su, errP_su, errL2_su = _compute_errs(stair_unsnap_pts)

    # Panel (a): Angular patterns I_unpol(theta) at all cpw on one figure
    ax0 = axes[0]
    k_styles = [("#7aa6c2", ":"), ("#3b7ea1", "--"), ("#004c6d", "-.")]
    for (c_i, _, th_i, I_i, _), (col, ls) in zip(kottke_pts[:-1], k_styles):
        ax0.plot(th_i, I_i, ls=ls, color=col, lw=1.6, label=f"Kottke $\\mathrm{{cpw}}={c_i}$")
    for (c_i, _, th_i, I_i, _), mk in zip(stair_snap_pts, ("v", "d", "^")):
        if c_i == 10:
            continue
        ax0.plot(
            th_i,
            I_i,
            ls="-",
            marker=mk,
            ms=4.5,
            color="seagreen",
            alpha=0.85,
            lw=1.5,
            label=f"Staircase Snapped $\\mathrm{{cpw}}={c_i}$",
        )
    u_styles = [("#f4a582", ":", "s"), ("#d6604d", "--", "s"), ("#b2182b", "-.", "s")]
    for (c_i, _, th_i, I_i, _), (col, ls, mk) in zip(stair_unsnap_pts, u_styles):
        lbl = (
            f"Staircase $\\mathrm{{cpw}}={c_i}$ (commensurate)"
            if c_i == 10
            else f"Staircase Un-snapped $\\mathrm{{cpw}}={c_i}$"
        )
        ax0.plot(th_i, I_i, ls=ls, marker=mk, ms=4.2, color=col, lw=1.6, label=lbl)
    ax0.plot(
        ref_th,
        ref_I,
        "k-",
        lw=2.4,
        label=f"Kottke Ref ($\\mathrm{{cpw}}={ref_cpw}$)",
    )
    ax0.set_xlabel(r"Observation Angle $\theta$ [deg]")
    ax0.set_ylabel(r"Unpolarized Intensity $\langle I_{\mathrm{unpol}}(\theta)\rangle_x / U_{\mathrm{free,2D}}$")
    ax0.set_title(r"(a) Angular Patterns & $\theta=0^\circ$ GMR Shift Across $\mathrm{cpw}$")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", fontsize=7.0, ncol=1)

    # Panel (b): Angle-integrated extraction power vs cpw
    ax1 = axes[1]
    ax1.plot(cpw_k, P_k, "o-", color="tab:blue", lw=2.2, ms=6.5, label="Kottke Subpixel Smoothing")
    ax1.plot(
        cpw_ss,
        P_ss,
        "d-",
        color="seagreen",
        lw=2.1,
        ms=6.2,
        label="Staircase (Snapped Grid: $\\Lambda, y_{\\mathrm{base}}, d_{\\mathrm{slab}}$)",
    )
    ax1.plot(
        cpw_su,
        P_su,
        "s--",
        color="tab:red",
        lw=2.0,
        ms=5.8,
        label="Staircase (Un-snapped Grid: Geometry Drift)",
    )
    ax1.axhline(
        ref_P,
        color="k",
        ls=":",
        lw=1.4,
        label=f"Fine Reference ($\\mathrm{{cpw}}={ref_cpw}$, $P={ref_P:.4f}\\,\\mathrm{{rad}}$)",
    )
    ax1.set_xlabel(r"Resolution $\mathrm{cpw} = \lambda_0 / (n_{\mathrm{GaN}}\,\Delta x)$ [cells / wavelength]")
    ax1.set_ylabel(r"Angle-Integrated Extraction Power $\int I_{\mathrm{unpol}}(\theta)\,\mathrm{d}\theta$ [rad]")
    ax1.set_title("(b) Angle-Integrated Power Convergence")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=7.5)

    # Panel (c): Log-log Angular L2 Error + Integrated-Power Error vs cpw
    ax2 = axes[2]
    mask_k = cpw_k < ref_cpw
    ax2.loglog(
        cpw_k[mask_k],
        np.maximum(errL2_k[mask_k], 1e-4),
        "o-",
        color="tab:blue",
        lw=2.2,
        label=f"Kottke: Angular L2 Err vs $\\mathrm{{cpw}}={ref_cpw}$",
    )
    ax2.loglog(
        cpw_ss,
        np.maximum(errL2_ss, 1e-4),
        "d-",
        color="seagreen",
        lw=2.1,
        label=f"Staircase (Snapped): Angular L2 Err vs $\\mathrm{{cpw}}={ref_cpw}$",
    )
    ax2.loglog(
        cpw_su,
        np.maximum(errL2_su, 1e-4),
        "s--",
        color="tab:red",
        lw=2.0,
        label="Staircase (Un-snapped): Angular L2 Err (GMR shift)",
    )
    ax2.loglog(
        cpw_k[mask_k],
        np.maximum(errP_k[mask_k], 1e-4),
        "o:",
        color="tab:blue",
        lw=1.5,
        alpha=0.8,
        label="Kottke: Integrated-Power Error $|\\Delta P_{\\mathrm{int}}|/P_{\\mathrm{ref}}$",
    )
    ax2.loglog(
        cpw_ss,
        np.maximum(errP_ss, 1e-4),
        "d:",
        color="seagreen",
        lw=1.5,
        alpha=0.8,
        label="Staircase (Snapped): Integrated-Power Error",
    )
    ax2.loglog(
        cpw_su,
        np.maximum(errP_su, 1e-4),
        "s:",
        color="tab:red",
        lw=1.5,
        alpha=0.8,
        label="Staircase (Un-snapped): Integrated-Power Error",
    )
    ax2.loglog(
        d0["cpw_conv"],
        d0["err_tm_conv"],
        "^-.",
        color="darkgoldenrod",
        lw=1.7,
        label="Flat Slab (Bloch FIT vs Exact Analytic TMM)",
    )
    cpw_ref_line = np.array([8.0, float(ref_cpw)])
    ax2.loglog(
        cpw_ref_line,
        0.05 * (8.0 / cpw_ref_line) ** 2,
        "k:",
        lw=1.3,
        label=r"Second-order $O(\Delta x^2)$ slope",
    )
    ax2.set_xlabel(r"Resolution $\mathrm{cpw}$ [cells / $\lambda_{\mathrm{GaN}}$]")
    ax2.set_ylabel("Relative Error (Angular L2 & Integrated Power)")
    ax2.set_title("(c) Angular L2 & Integrated-Power Error vs $\\mathrm{cpw}$")
    ax2.legend(loc="lower left", fontsize=6.8, ncol=1)

    figstyle.save(fig, fig_dir / "13_resolution_convergence_kottke_vs_staircase.png")

    # ------------------------------------------------------------------
    # Figure 14: Broadband Incoherent Spectral Averaging (Finite Linewidth)
    # ------------------------------------------------------------------
    lams_nm = np.linspace(425.0, 475.0, 41)
    sigma_nm = 10.0  # FWHM ~ 23.5 nm typical blue InGaN QW electroluminescence
    weights_lam = np.exp(-0.5 * ((lams_nm - 450.0) / sigma_nm) ** 2)
    weights_lam /= np.trapezoid(weights_lam, lams_nm)

    th_deg_14 = np.linspace(-85.0, 85.0, 341)
    th_rad_14 = np.deg2rad(th_deg_14)

    I_mono_435 = tmm_slab_emission(th_rad_14, Y_EMIT_BENCH, D_FLAT_BENCH, N_SLAB, 435e-9, pol="unpolarized")
    I_mono_450 = tmm_slab_emission(th_rad_14, Y_EMIT_BENCH, D_FLAT_BENCH, N_SLAB, 450e-9, pol="unpolarized")
    I_mono_465 = tmm_slab_emission(th_rad_14, Y_EMIT_BENCH, D_FLAT_BENCH, N_SLAB, 465e-9, pol="unpolarized")

    I_spec_stack = np.array(
        [tmm_slab_emission(th_rad_14, Y_EMIT_BENCH, D_FLAT_BENCH, N_SLAB, lam * 1e-9, pol="unpolarized") for lam in lams_nm]
    )
    I_broadband = np.trapezoid(I_spec_stack * weights_lam[:, None], lams_nm, axis=0)

    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.3)
    ax = axes[0]
    ax.plot(th_deg_14, I_mono_435, ":", color="royalblue", lw=1.6, label=r"Monochromatic $\lambda = 435\,\mathrm{nm}$")
    ax.plot(th_deg_14, I_mono_450, "--", color="tab:green", lw=1.8, label=r"Monochromatic $\lambda_0 = 450\,\mathrm{nm}$")
    ax.plot(th_deg_14, I_mono_465, "-.", color="darkorange", lw=1.6, label=r"Monochromatic $\lambda = 465\,\mathrm{nm}$")
    ax.plot(
        th_deg_14,
        I_broadband,
        "k-",
        lw=2.4,
        label=r"Spectrally Averaged $\int S(\lambda) I(\theta,\lambda)\,\mathrm{d}\lambda$ ($\mathrm{FWHM}=23.5\,\mathrm{nm}$)",
    )
    ax.set_xlabel(r"Observation Angle $\theta$ [deg]")
    ax.set_ylabel(r"Unpolarized Intensity $I(\theta)/U_{\mathrm{free,2D}}$")
    ax.set_title("(a) Spectral Incoherent Averaging Over InGaN QW Linewidth")
    ax.legend(loc="upper right", fontsize=8)

    ax2 = axes[1]
    im14 = ax2.pcolormesh(
        th_deg_14,
        lams_nm,
        I_spec_stack,
        cmap=figstyle.CMAP_MAGNITUDE,
        shading="auto",
    )
    ax2.axhline(450.0, color="white", ls="--", lw=1.5, label=r"Center $\lambda_0 = 450\,\mathrm{nm}$")
    ax2.set_xlabel(r"Observation Angle $\theta$ [deg]")
    ax2.set_ylabel(r"Emission Wavelength $\lambda$ [nm]")
    ax2.set_title(r"(b) Angle-Wavelength Map $I_{\mathrm{unpol}}(\theta, \lambda)$")
    fig.colorbar(im14, ax=ax2, label=r"$I_{\mathrm{unpol}}(\theta, \lambda)/U_{\mathrm{free,2D}}$")

    figstyle.save(fig, fig_dir / "14_wavelength_broadband_incoherent.png")

    # ------------------------------------------------------------------
    # Figure 15: Drude Ag Metal Skin-Depth Resolution Study (Graded Mesh)
    # ------------------------------------------------------------------
    p9 = out_dir / "task_09.npz"
    p10 = out_dir / "task_10.npz"
    if p9.exists() and p10.exists():
        d9 = np.load(p9)
        d10 = np.load(p10)
        cpsd_all = np.concatenate([d9["cpsd_meas"], d10["cpsd_meas"]])
        cpsd_pow_all = np.concatenate([d9["cpsd_pow_meas"], d10["cpsd_pow_meas"]])
        sw_err_all = np.concatenate([d9["sw_l2_err_list"], d10["sw_l2_err_list"]]) * 100.0
        P_unpol_all = np.concatenate([d9["P_grat_unpol_list"], d10["P_grat_unpol_list"]])
        P_tm_all = np.concatenate([d9["P_grat_tm_list"], d10["P_grat_tm_list"]])
        enh_unpol_all = np.concatenate([d9["enh_unpol_list"], d10["enh_unpol_list"]])
        enh_tm_all = np.concatenate([d9["enh_tm_list"], d10["enh_tm_list"]])
        n_demo = len(d9["cpsd_meas"])

        idx_acc = int(np.argmin(np.abs(cpsd_all - float(d10["cells_per_skin_depth"]))))
        cpsd_acc = float(cpsd_all[idx_acc])

        fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.8, height=4.4)

        ax = axes[0]
        if "cpsd_ag_meas" in d0:
            ax.plot(
                d0["cpsd_ag_meas"],
                d0["err_ag0_ladder"] * 100.0,
                "o-",
                color="tab:purple",
                lw=2.2,
                ms=6.5,
                label=r"1D Bloch Flat Slab ($\theta=0^\circ$, TM)",
            )
            ax.plot(
                d0["cpsd_ag_meas"],
                d0["err_ag25_ladder"] * 100.0,
                "s--",
                color="teal",
                lw=1.9,
                ms=5.8,
                label=r"1D Bloch Flat Slab ($\theta=25^\circ$, TM)",
            )
        ax.plot(
            cpsd_all[:n_demo],
            sw_err_all[:n_demo],
            "rx",
            ms=9,
            mew=2.2,
            label="Under-resolved Demo ($1.6, 2.6\\,\\mathrm{cells}/\\delta_s$)",
        )
        ax.axvline(
            cpsd_acc,
            color="darkgreen",
            ls="--",
            lw=1.8,
            label=f"Accepted Level ($\\mathrm{{cpsd}}_s={cpsd_acc:.1f}, \\mathrm{{cpsd}}_p={0.5*cpsd_acc:.1f}$)",
        )
        ax.set_xlabel(
            r"Cells / Amplitude Skin Depth $\delta_s = \lambda_0/(2\pi\kappa) = 26.9\,\mathrm{nm}$"
            "\n"
            r"(Cells / Power Skin Depth $\delta_p = \lambda_0/(4\pi\kappa) = 13.5\,\mathrm{nm}$ is $0.5\times$)"
        )
        ax.set_ylabel("Standing-Wave Relative L2 Error vs TMM [%]")
        ax.set_title("(a) Flat-Slab Standing-Wave Error on Drude Ag")
        ax.legend(loc="upper right", fontsize=7.8)

        ax2 = axes[1]
        P_ref_u = float(P_unpol_all[-1])
        ax2.axhspan(
            0.99 * P_ref_u,
            1.01 * P_ref_u,
            color="limegreen",
            alpha=0.18,
            label=r"$\pm 1\%$ Convergence Band",
        )
        ax2.plot(
            cpsd_all,
            P_unpol_all,
            "o-",
            color="crimson",
            lw=2.2,
            ms=6.5,
            label=r"2D Grating on Ag: Unpolarized $P_{\mathrm{int}}$",
        )
        ax2.plot(
            cpsd_all[:n_demo],
            P_unpol_all[:n_demo],
            "rx",
            ms=9,
            mew=2.2,
            label="Demo Under-resolved (`task_09`)",
        )
        ax2.plot(
            cpsd_acc,
            P_unpol_all[idx_acc],
            "g*",
            ms=13,
            label=f"Accepted (`task_10`, $P={P_unpol_all[idx_acc]:.4f}\\,\\mathrm{{rad}}$)",
        )
        ax2.axvline(cpsd_acc, color="darkgreen", ls="--", lw=1.6)
        ax2.set_xlabel(r"Cells / Amplitude Skin Depth $\delta_s$ ($\Delta y_{\min}$ at Ag Interface)")
        ax2.set_ylabel(r"Angle-Integrated Power $\int I_{\mathrm{unpol}}(\theta)\,\mathrm{d}\theta$ [rad]")
        ax2.set_title("(b) 2D Grating LED Extraction Power vs Skin-Depth Grid")
        ax2.legend(loc="upper right", fontsize=7.8)

        ax3 = axes[2]
        enh_ref_u = float(enh_unpol_all[-1])
        ax3.axhspan(
            0.99 * enh_ref_u,
            1.01 * enh_ref_u,
            color="limegreen",
            alpha=0.18,
            label=r"$\pm 1\%$ Convergence Band",
        )
        ax3.plot(
            cpsd_all,
            enh_unpol_all,
            "o-",
            color="crimson",
            lw=2.2,
            ms=6.5,
            label=r"Unpolarized Enhancement $\eta_{\mathrm{grat}}/\eta_{\mathrm{flat}}$",
        )
        ax3.plot(
            cpsd_all,
            enh_tm_all,
            "s--",
            color="tab:blue",
            lw=2.0,
            ms=6.0,
            label=r"TM ($J_z$) Enhancement $\eta_{\mathrm{grat,TM}}/\eta_{\mathrm{flat,TM}}$",
        )
        ax3.plot(
            cpsd_acc,
            enh_unpol_all[idx_acc],
            "g*",
            ms=13,
            label=f"Accepted ($\\times{enh_unpol_all[idx_acc]:.3f}$ unpol, $\\times{enh_tm_all[idx_acc]:.3f}$ TM)",
        )
        ax3.axvline(cpsd_acc, color="darkgreen", ls="--", lw=1.6)
        ax3.set_xlabel(r"Cells / Amplitude Skin Depth $\delta_s$ ($2\times$ Cells / $\delta_p$)")
        ax3.set_ylabel(r"Grating Extraction Enhancement $\eta_{\mathrm{grat,Ag}} / \eta_{\mathrm{flat,Ag}}$")
        ax3.set_title("(c) Grating Extraction Enhancement on Drude Ag Mirror")
        ax3.legend(loc="upper right", fontsize=7.8)

        figstyle.save(fig, fig_dir / "15_metal_skin_depth_resolution_study.png")

    # ------------------------------------------------------------------
    # Figures 16..19: Janssen et al. (2010) Optimization Study (Figs. 2, 4, 5, 6)
    # ------------------------------------------------------------------
    p_janssen_npz = out_dir / "janssen_optimization.npz"
    p_janssen_json = out_dir / "janssen_optimization.json"
    if not p_janssen_npz.exists() or not p_janssen_json.exists():
        run_janssen_optimization_study(quick=True, out_dir=out_dir)

    j_npz = np.load(p_janssen_npz, allow_pickle=True)
    j_sum = json.loads(p_janssen_json.read_text())

    # Figure 16: Shallow-grating guided-mode resonances (Janssen Fig. 2, pitch=250 nm, d=5 nm)
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.8, height=4.4)
    kx_f2 = np.asarray(j_npz["kx_grid_fig2"], dtype=float)
    kx_dense_pec = np.asarray(j_npz["kx_dense_pec_fig2"], dtype=float)

    ax = axes[0]
    ax.plot(
        kx_dense_pec,
        np.minimum(j_npz["pec_grat_dense_s"], 28.0),
        color="tab:blue",
        lw=1.1,
        label=r"Grating $s$-pol ($d=5\,\mathrm{nm}$)",
    )
    ax.plot(
        kx_dense_pec,
        np.minimum(j_npz["pec_grat_dense_p"], 28.0),
        color="tab:red",
        lw=1.0,
        alpha=0.85,
        label=r"Grating $p$-pol ($d=5\,\mathrm{nm}$)",
    )
    ax.plot(
        kx_f2,
        j_npz["pec_flat_s"],
        "k--",
        lw=1.8,
        label=r"Flat $s$-pol ($t=698\,\mathrm{nm}$)",
    )
    ax.plot(
        kx_f2,
        j_npz["pec_flat_p"],
        color="0.35",
        ls=":",
        lw=1.8,
        label=r"Flat $p$-pol ($t=698\,\mathrm{nm}$)",
    )
    ax.axvspan(0.0, 0.5, color="gold", alpha=0.10, label=r"$60^\circ$ cone ($|\hat{k}_x|\leq 0.5$)")
    ax.set_xlim(0.0, 0.98)
    ax.set_ylim(0.0, 22.0)
    ax.set_xlabel(r"Normalized Wavevector $\hat{k}_x = k_x / k_0 = \sin\theta$")
    ax.set_ylabel(r"Absorbed / Emitted Intensity $\int_{\mathcal{O}} |\mathbf{E}|^2\,\mathrm{d}x / (\Lambda |E_{\mathrm{inc}}|^2)$")
    ax.set_title(
        f"(a) PEC Mirror ($\\Lambda=250\\,\\mathrm{{nm}}, d=5\\,\\mathrm{{nm}}$)\n"
        f"Ultra-narrow GMR spikes ($Q \\sim 10^5$)"
    )
    ax.legend(loc="upper right", fontsize=7.2)

    ax = axes[1]
    ax.plot(
        kx_f2,
        np.minimum(j_npz["ag_grat_s"], 22.0),
        color="tab:blue",
        lw=1.5,
        label=r"Grating $s$-pol (`silver jc`, $d=5\,\mathrm{nm}$)",
    )
    ax.plot(
        kx_f2,
        np.minimum(j_npz["ag_grat_p"], 22.0),
        color="tab:red",
        lw=1.4,
        alpha=0.9,
        label=r"Grating $p$-pol (`silver jc`, $d=5\,\mathrm{nm}$)",
    )
    ax.plot(
        kx_f2,
        j_npz["ag_flat_s"],
        "k--",
        lw=1.8,
        label=r"Flat $s$-pol ($t=698\,\mathrm{nm}$)",
    )
    ax.plot(
        kx_f2,
        j_npz["ag_flat_p"],
        color="0.35",
        ls=":",
        lw=1.8,
        label=r"Flat $p$-pol ($t=698\,\mathrm{nm}$)",
    )
    ax.axvspan(0.0, 0.5, color="gold", alpha=0.10, label=r"$60^\circ$ cone ($|\hat{k}_x|\leq 0.5$)")
    ax.set_xlim(0.0, 0.98)
    ax.set_ylim(0.0, 18.0)
    ax.set_xlabel(r"Normalized Wavevector $\hat{k}_x = k_x / k_0 = \sin\theta$")
    ax.set_ylabel(r"Absorbed / Emitted Intensity $\int_{\mathcal{O}} |\mathbf{E}|^2\,\mathrm{d}x / (\Lambda |E_{\mathrm{inc}}|^2)$")
    ax.set_title(
        f"(b) Silver Mirror (`silver jc`, $\\varepsilon_{{\\mathrm{{Ag}}}}={j_sum['eps_ag_450nm_real']:.2f}{j_sum['eps_ag_450nm_imag']:+.2f}j$)\n"
        f"Ohmic damping broadens resonances to $Q \\sim 10^3$"
    )
    ax.legend(loc="upper right", fontsize=7.2)

    ax = axes[2]
    kx_zp = np.asarray(j_npz["kx_zoom_pec"], dtype=float)
    I_zp = np.asarray(j_npz["I_zoom_pec_s"], dtype=float)
    kx_za = np.asarray(j_npz["kx_zoom_ag"], dtype=float)
    I_za = np.asarray(j_npz["I_zoom_ag_s"], dtype=float)
    k0_zp = float(kx_zp[int(np.argmax(I_zp))])
    k0_za = float(kx_za[int(np.argmax(I_za))])
    ax.plot(
        (kx_zp - k0_zp) * 1e3,
        I_zp / max(float(np.max(I_zp)), 1e-12),
        color="navy",
        lw=2.1,
        label=(
            f"PEC Resonance ($Q_{{\\mathrm{{mean}}}}={j_sum['fig2_Q_pec_median_s']:.1e}$, "
            f"harminv $Q={j_sum['fig2_harminv_pec_Q']:.2e}$)"
        ),
    )
    ax.plot(
        (kx_za - k0_za) * 1e3,
        I_za / max(float(np.max(I_za)), 1e-12),
        color="crimson",
        lw=2.1,
        ls="--",
        label=(
            f"Silver (`silver jc`) ($Q_{{\\mathrm{{med}}}}={j_sum['fig2_Q_ag_median_s']:.1e}$, "
            f"harminv $Q={j_sum['fig2_harminv_ag_Q']:.2e}$)"
        ),
    )
    ax.axhline(0.5, color="0.4", ls=":", lw=1.1, label="Half-Maximum (FWHM)")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel(r"Detuning from Peak $\Delta \hat{k}_x \times 10^3$")
    ax.set_ylabel("Peak-Normalized Resonance LineShape")
    ax.set_title(
        "(c) Resonance Linewidth & Wakis `harminv` Check\n"
        r"PEC $Q \sim 10^5$ vs Silver $Q \sim 10^3$ (Janssen Fig. 2)"
    )
    ax.legend(loc="upper right", fontsize=7.2)

    figstyle.save(fig, fig_dir / "16_janssen_fig2_shallow_grating_resonances.png")

    # Figure 17: (t, d) landscape + steepest-ascent paths + both-ways gradient verification (Janssen Fig. 4)
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.8, height=4.4)
    t_grid_nm = np.asarray(j_npz["t_grid_nm"], dtype=float)
    d_grid_nm = np.asarray(j_npz["d_grid_nm"], dtype=float)
    F60_iso = np.asarray(j_npz["F60_iso"], dtype=float)
    F90_iso = np.asarray(j_npz["F90_iso"], dtype=float)

    ax = axes[0]
    im0 = ax.pcolormesh(
        t_grid_nm,
        d_grid_nm,
        F60_iso.T,
        shading="auto",
        cmap=figstyle.CMAP_MAGNITUDE,
        vmin=0.5,
        vmax=float(np.nanpercentile(F60_iso, 98)),
    )
    fig.colorbar(im0, ax=ax, label=r"$60^\circ$-Cone Extracted Intensity $F_{60^\circ}(t, d)$")
    path_colors = ["cyan", "lime", "deepskyblue", "springgreen"]
    for ip_idx in range(4):
        traj = np.asarray(j_npz[f"ascent_path_{ip_idx}"], dtype=float)
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            "-",
            color=path_colors[ip_idx],
            lw=2.0,
            label="Steepest-Ascent Path" if ip_idx == 0 else None,
        )
        ax.plot(traj[0, 0], traj[0, 1], "o", color="white", mec="black", ms=5.5)
        ax.plot(
            traj[-1, 0],
            traj[-1, 1],
            "*",
            color="yellow",
            mec="black",
            ms=11.0,
            label="Local Optimum" if ip_idx == 0 else None,
        )
    ax.set_xlabel(r"Total GaN Thickness $t$ [nm]")
    ax.set_ylabel(r"Grating Etch Depth $d$ [nm]")
    ax.set_title(
        r"(a) $(t, d)$ Landscape ($60^\circ$ Cone, $\Lambda=450\,\mathrm{nm}$)"
        "\nMultiple Local Maxima & Steepest-Ascent Paths"
    )
    ax.legend(loc="upper left", fontsize=7.4, framealpha=0.88)

    ax = axes[1]
    im1 = ax.pcolormesh(
        t_grid_nm,
        d_grid_nm,
        F90_iso.T,
        shading="auto",
        cmap="viridis",
        vmin=0.5,
        vmax=float(np.nanpercentile(F90_iso, 98)),
    )
    fig.colorbar(im1, ax=ax, label=r"$90^\circ$-Cone Extracted Intensity $F_{90^\circ}(t, d)$")
    for lm in j_sum["fig4_local_maxima"][:10]:
        ax.plot(lm["t_nm"], lm["d_nm"], "w+", ms=7, mew=1.4)
    ax.set_xlabel(r"Total GaN Thickness $t$ [nm]")
    ax.set_ylabel(r"Grating Etch Depth $d$ [nm]")
    ax.set_title(
        r"(b) $(t, d)$ Landscape ($90^\circ$ Cone, $|\hat{k}_x| \leq 1/\sqrt{2}$)"
        f"\nPeak $F_{{90^\\circ}}={j_sum['fig4_best_90_iso']:.2f}$ ({j_sum['fig4_enh_vs_flat_90']:.2f}$\\times$ Flat)"
    )

    ax = axes[2]
    gc = j_sum["gradient_check"]
    labels = [r"$\partial \ln F / \partial t$", r"$\partial \ln F / \partial d$"]
    x_pos = np.arange(len(labels))
    w_bar = 0.25
    adj_vals = np.array([gc["dlnF_dt_adj_per_m"], gc["dlnF_dd_adj_per_m"]]) * 1e-7
    fd1_vals = np.array([gc["dlnF_dt_fd1_per_m"], gc["dlnF_dd_fd1_per_m"]]) * 1e-7
    fd2_vals = np.array([gc["dlnF_dt_fd2_per_m"], gc["dlnF_dd_fd2_per_m"]]) * 1e-7
    ax.bar(x_pos - w_bar, adj_vals, width=w_bar, color="tab:blue", label="Wakis `AdjointSimulation` (VJP)")
    ax.bar(x_pos, fd1_vals, width=w_bar, color="tab:orange", label=r"Central FD ($\delta_1 = 0.10\,\mathrm{nm}$)")
    ax.bar(x_pos + w_bar, fd2_vals, width=w_bar, color="tab:green", label=r"Central FD ($\delta_2 = 0.05\,\mathrm{nm}$)")
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"Normalized Shape Gradient [$10^7\,\mathrm{m}^{-1}$]")
    ax.set_title(
        "(c) Both-Ways Shape Gradient Verification\n"
        f"Quadratic $O(\\delta^2)$ ratio: {gc['order_ratio_dt']:.2f}$\\times$ ($t$), {gc['order_ratio_dd']:.2f}$\\times$ ($d$)"
    )
    ax.legend(loc="upper right", fontsize=7.2)

    figstyle.save(fig, fig_dir / "17_janssen_fig4_thickness_depth_landscape.png")

    # Figure 18: Optimal active-layer position near y_0 ~ 120 nm (Janssen Fig. 5) + 2D FIT check
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.8, height=4.4)
    x_rel_5 = np.asarray(j_npz["x_rel_fig5"], dtype=float)
    y_nm_5 = np.asarray(j_npz["y_arr_fig5_nm"], dtype=float)
    map_5_iso = np.asarray(j_npz["map_fig5_iso"], dtype=float)

    ax = axes[0]
    im5 = ax.pcolormesh(
        x_rel_5,
        y_nm_5,
        map_5_iso.T,
        shading="auto",
        cmap=figstyle.CMAP_MAGNITUDE,
    )
    fig.colorbar(im5, ax=ax, label=r"Local Reciprocal Intensity $I_{\mathrm{iso}}(x_0, y_0)$")
    ax.axhline(
        j_sum["fig5_y_opt_grat_450_nm"],
        color="cyan",
        ls="--",
        lw=2.0,
        label=f"Optimal QW Height $y_0 = {j_sum['fig5_y_opt_grat_450_nm']:.0f}\\,\\mathrm{{nm}}$",
    )
    ax.axvline(0.25, color="white", ls=":", lw=1.1, alpha=0.85)
    ax.axvline(0.75, color="white", ls=":", lw=1.1, alpha=0.85)
    ax.text(
        0.50,
        365.0,
        "Unetched GaN ($n=2.45$) under tooth ($x_0/\\Lambda \\in [0.25, 0.75]$)",
        ha="center",
        va="top",
        fontsize=6.2,
        color="#0f172a",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "#475569", "alpha": 0.85},
    )
    ax.text(
        0.50,
        18.0,
        "Silver (`silver jc`) mirror ($y_0 = 0$) | Bloch unit cell $\\pm x$",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#0f172a",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "#475569", "alpha": 0.85},
    )
    ax.set_xlabel(r"Unit-Cell Coordinate $x_0 / \Lambda$")
    ax.set_ylabel(r"Height Above Silver Mirror $y_0$ [nm]")
    ax.set_title(
        r"(a) Local Extraction Map $I(x_0, y_0)$ in Unetched GaN"
        "\n($t=700\\,\\mathrm{nm},\\,d=300\\,\\mathrm{nm},\\,\\Lambda=450\\,\\mathrm{nm}$)"
    )
    ax.legend(loc="upper right", fontsize=7.4)

    ax = axes[1]
    ax.plot(
        y_nm_5,
        j_npz["prof_fig5_iso"],
        color="tab:red",
        lw=2.2,
        label=f"Grating $\\Lambda=450\\,\\mathrm{{nm}}$ (peak $y_0={j_sum['fig5_y_opt_grat_450_nm']:.0f}\\,\\mathrm{{nm}}$)",
    )
    ax.plot(
        y_nm_5,
        j_npz["prof_fig5_360_iso"],
        color="tab:blue",
        lw=2.0,
        ls="-.",
        label=f"Grating $\\Lambda=360\\,\\mathrm{{nm}}$ (peak $y_0={j_sum['fig5_y_opt_grat_360_nm']:.0f}\\,\\mathrm{{nm}}$)",
    )
    ax.plot(
        y_nm_5,
        j_npz["prof_flat_iso"],
        color="black",
        lw=1.9,
        ls="--",
        label=f"Flat LED $t=672\\,\\mathrm{{nm}}$ (peak $y_0={j_sum['fig5_y_opt_flat_nm']:.0f}\\,\\mathrm{{nm}}$)",
    )
    ax.axvline(120.0, color="darkgreen", ls=":", lw=1.7, label=r"First Antinode $y_0 \approx 120\,\mathrm{nm}$")
    ax.set_xlabel(r"Active-Layer Position Above Silver Mirror $y_0$ [nm]")
    ax.set_ylabel(r"$60^\circ$-Cone Extracted Intensity $\langle I_{\mathrm{iso}}(y_0)\rangle_x$")
    ax.set_title(
        "(b) Active-Layer Position Scan (Janssen Fig. 5b)\n"
        r"First antinode peak near $y_0 \approx 120\,\mathrm{nm}$ above Ag"
    )
    ax.legend(loc="upper right", fontsize=7.2)

    ax = axes[2]
    y_fit_nm = np.asarray(j_npz["fit_unit_y_nm"], dtype=float)
    ax.plot(
        y_fit_nm,
        j_npz["rcwa_unit_0_s"],
        "k-",
        lw=2.0,
        label=r"RCWA ($\theta = 0^\circ$, $s$-pol)",
    )
    ax.plot(
        y_fit_nm,
        j_npz["fit_unit_0_s"],
        "o",
        color="tab:red",
        ms=4.5,
        label=r"Wakis 2D Bloch FIT ($\theta = 0^\circ$, `silver jc`)",
    )
    ax.plot(
        y_fit_nm,
        j_npz["rcwa_unit_15_s"],
        "--",
        color="navy",
        lw=1.8,
        label=r"RCWA ($\theta = 25^\circ$, $s$-pol)",
    )
    ax.plot(
        y_fit_nm,
        j_npz["fit_unit_15_s"],
        "s",
        color="tab:blue",
        ms=4.2,
        label=r"Wakis 2D Bloch FIT ($\theta = 25^\circ$, `silver jc`)",
    )
    ax.axvline(120.0, color="darkgreen", ls=":", lw=1.5)
    ax.set_xlabel(r"Vertical Coordinate $y_0$ [nm]")
    ax.set_ylabel(r"Unit-Cell-Averaged $\langle |E_z(x, y_0)|^2 \rangle_x / |E_{\mathrm{inc}}|^2$")
    ax.set_title(
        "(c) Wakis 2D Bloch FIT vs RCWA Unit-Cell Standing Wave\n"
        r"(`dispersive_lib['silver jc']`, $\Lambda=450\,\mathrm{nm},\,t=700\,\mathrm{nm}$)"
    )
    ax.legend(loc="upper right", fontsize=7.2)

    figstyle.save(fig, fig_dir / "18_janssen_fig5_active_layer_position.png")

    # Figure 19: Pitch scan Lambda / lam0 and ~2x enhancement headline (Janssen Fig. 6)
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.8, height=4.4)
    rat_6 = np.asarray(j_npz["ratio_grid_fig6"], dtype=float)
    kx_6 = np.asarray(j_npz["kx_fig6"], dtype=float)
    map6_s = np.asarray(j_npz["map_fig6_s"], dtype=float)
    map6_p = np.asarray(j_npz["map_fig6_p"], dtype=float)

    ax = axes[0]
    im6s = ax.pcolormesh(
        rat_6,
        kx_6,
        np.minimum(map6_s.T, 12.0),
        shading="auto",
        cmap=figstyle.CMAP_MAGNITUDE,
        vmin=0.0,
        vmax=10.0,
    )
    fig.colorbar(im6s, ax=ax, label=r"$s$-Pol Intensity $I_s(\Lambda/\lambda_0, \hat{k}_x)$")
    ax.axhline(0.5, color="cyan", ls="--", lw=1.5, label=r"$60^\circ$ cone ($|\hat{k}_x|=0.5$)")
    ax.axvline(0.8, color="white", ls=":", lw=1.5, label=r"$\Lambda/\lambda_0 = 0.8$")
    ax.set_xlabel(r"Normalized Grating Pitch $\Lambda / \lambda_0$")
    ax.set_ylabel(r"Normalized Wavevector $\hat{k}_x = \sin\theta$")
    ax.set_title(
        r"(a) $s$-Polarization Pitch Dispersion Map"
        "\n($t=700\\,\\mathrm{nm},\\,d=300\\,\\mathrm{nm}$, Janssen Fig. 6a)"
    )
    ax.legend(loc="upper right", fontsize=7.4)

    ax = axes[1]
    im6p = ax.pcolormesh(
        rat_6,
        kx_6,
        np.minimum(map6_p.T, 12.0),
        shading="auto",
        cmap=figstyle.CMAP_MAGNITUDE,
        vmin=0.0,
        vmax=10.0,
    )
    fig.colorbar(im6p, ax=ax, label=r"$p$-Pol Intensity $I_p(\Lambda/\lambda_0, \hat{k}_x)$")
    ax.axhline(0.5, color="cyan", ls="--", lw=1.5, label=r"$60^\circ$ cone ($|\hat{k}_x|=0.5$)")
    ax.axvline(0.8, color="white", ls=":", lw=1.5, label=r"$\Lambda/\lambda_0 = 0.8$")
    ax.set_xlabel(r"Normalized Grating Pitch $\Lambda / \lambda_0$")
    ax.set_ylabel(r"Normalized Wavevector $\hat{k}_x = \sin\theta$")
    ax.set_title(
        r"(b) $p$-Polarization Pitch Dispersion Map"
        "\n($t=700\\,\\mathrm{nm},\\,d=300\\,\\mathrm{nm}$, Janssen Fig. 6b)"
    )
    ax.legend(loc="upper right", fontsize=7.4)

    ax = axes[2]
    c60_120 = np.asarray(j_npz["cone60_iso_120"], dtype=float)
    c90_120 = np.asarray(j_npz["cone90_iso_120"], dtype=float)
    flat_60_best = float(j_sum["headline_best_flat_60_iso"])
    flat_90_best = float(j_sum["headline_best_flat_90_iso"])
    ax.plot(
        rat_6,
        c60_120,
        color="tab:red",
        lw=2.1,
        label=f"Grating $60^\\circ$ Cone ($y_0=120\\,\\mathrm{{nm}}$, max {j_sum['headline_enhancement_global_60']:.2f}$\\times$ flat)",
    )
    ax.plot(
        rat_6,
        c90_120,
        color="tab:blue",
        lw=2.0,
        ls="-.",
        label=f"Grating $90^\\circ$ Cone ($y_0=120\\,\\mathrm{{nm}}$, max {j_sum['headline_enhancement_global_90']:.2f}$\\times$ flat)",
    )
    ax.axhline(
        flat_60_best,
        color="tab:red",
        ls="--",
        lw=1.6,
        label=f"Optimized Flat LED $60^\\circ$ Cone ($F={flat_60_best:.2f}$)",
    )
    ax.axhline(
        flat_90_best,
        color="tab:blue",
        ls=":",
        lw=1.6,
        label=f"Optimized Flat LED $90^\\circ$ Cone ($F={flat_90_best:.2f}$)",
    )
    ax.axvspan(0.70, 0.85, color="gold", alpha=0.18, label=r"Peak Band $\Lambda/\lambda_0 \approx 0.8$")
    ax.set_xlabel(r"Normalized Grating Pitch $\Lambda / \lambda_0$")
    ax.set_ylabel(r"Cone-Integrated Reciprocal Emission $F_{\mathrm{cone}}$")
    ax.set_title(
        "(c) Cone-Integrated Pitch Scan vs Optimized Flat LED\n"
        f"Optimized grating $\\approx {j_sum['headline_enhancement_global_60']:.2f}\\times$ ($60^\\circ$) / "
        f"{j_sum['headline_enhancement_global_90']:.2f}$\\times$ ($90^\\circ$) flat"
    )
    ax.legend(loc="upper right", fontsize=7.2)

    figstyle.save(fig, fig_dir / "19_janssen_fig6_pitch_scan_enhancement.png")


# %% [markdown]
# ## 5. Optimization of the Absorption in the Emitting Region (Janssen et al. 2010)
#
# Having validated the emitter-to-absorber equivalence (to $0.066\%$ ensemble L2 error at
# `--full` and $0.083\%$ at `--quick`), we now reproduce the main optimization study of
# **Janssen, Wachters & Urbach**, *Opt. Express* **18**(24), 24522–24535 (2010),
# `doi:10.1364/OE.18.024522`.
#
# ### 5.1 2D Periodic Configuration & Reciprocal Figure of Merit
#
# Following Janssen *et al.* (2010), Section 3, we consider a 2D periodic thin-film blue LED
# at vacuum wavelength $\lambda_0 = 450\;\text{nm}$:
# - **Metallic contact ($y \le 0$):** Silver from the Wakis materials library
#   (`wakis.materials.dispersive_lib["silver jc"]`, Johnson & Christy Lorentz-Drude fit with
#   $\varepsilon_{\text{Ag}}(450\;\text{nm}) = -7.0756 - 0.2201j$, amplitude skin depth
#   $\delta_s = 26.92\;\text{nm}$, power skin depth $\delta_p = 13.46\;\text{nm}$), plus an
#   ideal **PEC** mirror check ($\varepsilon_m \to -\infty$).
# - **GaN semiconductor slab ($0 \le y \le t$):** Refractive index $n_{\text{GaN}} = 2.45$
#   ($\varepsilon_{\text{GaN}} = 6.0025$), total thickness $t \sim 700\;\text{nm}$
#   ($t \in [450, 800]\;\text{nm}$), etched at the top interface ($t - d \le y \le t$) with
#   a 1D binary block surface grating of period (pitch) $\Lambda$, etch depth $d \in [0, 350]\;\text{nm}$,
#   and duty cycle $f = 0.50$, radiating into air ($n_{\text{air}} = 1.0$) above $y > t$.
# - **Emitting active layer $\mathcal{O}$ ($y = y_0$):** An InGaN/GaN quantum-well plane
#   located at height $y_0 \approx 100\;\text{nm}$ (and optimized over $y_0 \in [10, 390]\;\text{nm}$)
#   above the silver mirror.
# - **Figure of merit (Eqs. 7–10, 14 of Janssen *et al.*):** For an observation direction
#   $\theta$ ($\hat{k}_x = k_x / k_0 = \sin\theta$) and polarization $\nu \in \{s, p\}$, a
#   single Bloch-periodic unit-cell plane-wave solve yields the reciprocal internal electric
#   field $\mathbf{E}_{\text{rec}}^{(\nu)}(x_0, y_0; \theta)$. Integrating over the active layer
#   gives the directional emitted intensity:
#   $$I_s(\theta, y_0) = \frac{1}{\Lambda}\int_0^\Lambda \frac{|E_z(x_0, y_0; \theta)|^2}{|E_{\text{inc}}|^2}\,\mathrm{d}x_0, \qquad I_p(\theta, y_0) = \frac{1}{\Lambda}\int_0^\Lambda \frac{|E_x(x_0, y_0; \theta)|^2 + |E_y(x_0, y_0; \theta)|^2}{|E_{\text{inc}}|^2}\,\mathrm{d}x_0,$$
#   with isotropic 3D dipole average $I_{\text{iso}} = (I_s + I_p)/3$ (Janssen Eq. 8) and
#   in-plane quantum-well dipole average $I_{\text{QW,}\parallel} = (I_s + I_{p,x})/2$
#   (Janssen Eq. 9). Integrating over an extraction cone $|\theta| \le \Theta_{\text{cone}}/2$
#   of full opening angle $\Theta_{\text{cone}} \in \{60^\circ, 90^\circ\}$ ($|\hat{k}_x| \le 0.5$
#   for the $60^\circ$ cone, $|\hat{k}_x| \le 1/\sqrt{2} \approx 0.7071$ for the $90^\circ$ cone)
#   with Janssen's Eq. 10 flux weighting $\cos^2\theta\,\mathrm{d}\theta = \cos\theta\,\mathrm{d}\hat{k}_x$
#   yields the cone-averaged extraction objective $F_{\Theta_{\text{cone}}}(t, d, \Lambda, y_0)$.
#
# ---
#
# ### 5.2 Shallow-Grating Guided-Mode Resonances: PEC ($Q \sim 10^5$) vs Silver ($Q \sim 10^3$, Janssen Fig. 2)
#
# In Section 3.1 (Fig. 2) of Janssen *et al.* (2010), a very shallow grating ($d = 5\;\text{nm}$,
# $\Lambda = 250\;\text{nm}$, $t = 700\;\text{nm}$, $y_0 = 100\;\text{nm}$) is compared against a
# flat LED of the same volume-averaged thickness ($t_{\text{flat}} = 698\;\text{nm}$):
# - Without the grating ($d = 0$), the guided waveguide modes of the high-index GaN slab lie
#   outside the air light cone ($|k_x| > k_0$) and cannot couple to radiation.
# - The $d = 5\;\text{nm}$ periodic perturbation folds the guided slab modes into the air cone at
#   the Bragg condition $\hat{k}_{x,\text{peak}} = \pm n_{\text{eff}} - m \lambda_0 / \Lambda$,
#   creating sharp **guided-mode resonances (GMRs)** far above the flat Fabry–Pérot background.
# - **PEC back-reflector check:** Because PEC is lossless and the $d = 5\;\text{nm} = \lambda_0/90$
#   grating couples only weakly to air, the resonances are ultra-narrow with quality factors
#   **$Q_{\text{PEC}} \sim 10^5$**: across the $s$-polarized guided modes we measure
#   $Q_{\text{PEC,mean}} = 7.57 \times 10^4$ and $Q_{\text{PEC,max}} = 4.00 \times 10^5$ at
#   $\hat{k}_x = 0.630512$ ($\mathrm{FWHM}(\hat{k}_x) = 6.1 \times 10^{-6}$), while direct time-domain
#   ring-down analysis on the Wakis 2D Bloch FIT grid using `wakis.resonances.ResonanceMonitor.harminv`
#   yields **$Q_{\text{PEC,harminv}} = 1.78 \times 10^5$**.
# - **Silver (`silver jc`) back-reflector:** Ohmic absorption in the metallic skin layer
#   ($\delta_s = 26.92\;\text{nm}$) damps the guided modes as they bounce between the grating and
#   the mirror, broadening the resonances by nearly two orders of magnitude to **$Q_{\text{Ag}} \sim 10^3$**:
#   inside the $60^\circ$ extraction cone we measure $Q_{\text{Ag,median}} = 4.21 \times 10^3$
#   ($Q \in [1.16 \times 10^3, 6.96 \times 10^3]$ across the primary $s$- and $p$-polarized GMRs)
#   and Wakis time-domain `harminv` yields **$Q_{\text{Ag,harminv}} = 6.80 \times 10^3$**.
#
# ![Shallow-Grating Guided-Mode Resonances (Janssen Fig. 2)](figures/16_janssen_fig2_shallow_grating_resonances.png)
#
# ---
#
# ### 5.3 $(t, d)$ Optimization Landscape, Steepest-Ascent Paths & Both-Ways Gradient Check (Janssen Fig. 4)
#
# Because the shallow-grating resonances are too narrow in angle to substantially boost the
# cone-integrated extraction efficiency, Janssen *et al.* (Section 3.2, Fig. 4) sweep the
# total GaN thickness $t \in [450, 800]\;\text{nm}$ and grating etch depth $d \in [0, 350]\;\text{nm}$
# for a deep block grating of pitch $\Lambda = 450\;\text{nm} = \lambda_0$ and active-layer height
# $y_0 = 100\;\text{nm}$ on silver:
# - **Multi-modal $(t, d)$ landscape:** As shown in Figure 17(a,b), the cone-integrated objective
#   $F_{60^\circ}(t, d)$ (and $F_{90^\circ}(t, d)$) exhibits a diagonal ridge structure governed by
#   Fabry–Pérot interference in the unetched base thickness $t_{\text{base}} = t - d$ (period
#   $\Delta t_{\text{base}} = \lambda_0 / (2 n_{\text{GaN}}) = 91.8\;\text{nm}$) intersected by
#   guided-mode diffraction peaks at deep etch depths $d \in [100, 310]\;\text{nm}$. Consequently,
#   local gradient-based optimization (steepest-ascent paths in Figure 17a) converges to different
#   local maxima depending on the starting guess $(t_0, d_0)$ (e.g., local optima near
#   $(t, d) = (500, 190)\;\text{nm}$, $(590, 190)\;\text{nm}$, $(620, 120)\;\text{nm}$, and
#   $(700, 100)\;\text{nm}$, with $F_{60^\circ,\text{iso}}$ reaching $4.06$, or $2.43\times$ the
#   flat LED at $y_0 = 100\;\text{nm}$).
# - **How Wakis Computes Gradients (`wakis.inverse.AdjointSimulation` vs Central Finite Differences):**
#   The Wakis discrete adjoint module (`wakis.inverse.AdjointSimulation` coupled with
#   `wakis.inverse.MaterialGrid`) differentiates frequency-domain monitor objectives with respect
#   to the per-voxel material density array $\boldsymbol{\rho} \in [0, 1]^{N_x \times N_y \times N_z}$
#   on a fixed FIT grid via the exact time-reversed vector-Jacobian product (`sim.vjp`), rather
#   than moving mesh vertices directly. We therefore verify the shape gradient
#   $\nabla_{(t, d)} F = (\partial F / \partial t,\,\partial F / \partial d)$ at
#   $(t_0, d_0) = (650\;\text{nm}, 180\;\text{nm})$, $\theta = 15^\circ$ **both ways**
#   (`compute_grating_gradients_both_ways` in `_helpers.py`):
#   1. **Way 1 — Wakis Discrete Adjoint (`AdjointSimulation.vjp` + smooth boundary projection chain rule) vs Central FD:**
#      Parameterizing the upper and lower grating interfaces $y_{\text{top}} = t$ and
#      $y_{\text{base}} = t - d$ on a `MaterialGrid` via a smooth hyperbolic-tangent boundary
#      projection $\rho_k(t, d)$ and applying the chain rule
#      $\partial F / \partial t = \sum_k (\partial F / \partial \rho_k)(\partial \rho_k / \partial t)$,
#      $\partial F / \partial d = \sum_k (\partial F / \partial \rho_k)(\partial \rho_k / \partial d)$
#      yields in **1 forward + 1 adjoint FIT run**:
#      - $\partial \ln F / \partial t_{\text{adj}} = +8.716191 \times 10^7\;\text{m}^{-1}$ vs central FD
#        $+8.716139 \times 10^7\;\text{m}^{-1}$ at $\delta_1 = 0.10\;\text{nm}$ (rel. error
#        **$5.89 \times 10^{-6}$**) and $+8.716178 \times 10^7\;\text{m}^{-1}$ at
#        $\delta_2 = 0.05\;\text{nm}$ (rel. error **$1.47 \times 10^{-6}$**, error ratio
#        **$4.0001\times$**, confirming exact $O(\delta^2)$ second-order convergence).
#      - $\partial \ln F / \partial d_{\text{adj}} = -7.580340 \times 10^6\;\text{m}^{-1}$ vs central FD
#        $-7.580294 \times 10^6\;\text{m}^{-1}$ at $\delta_1 = 0.10\;\text{nm}$ (rel. error
#        **$6.06 \times 10^{-6}$**) and $-7.580328 \times 10^6\;\text{m}^{-1}$ at
#        $\delta_2 = 0.05\;\text{nm}$ (rel. error **$1.51 \times 10^{-6}$**, error ratio
#        **$4.0000\times$**).
#   2. **Way 2 — Direct Geometric Central Finite Differences on the Cone-Integrated Objective:**
#      Evaluating $F_{60^\circ,\text{iso}}(t, d)$ directly on the sharp geometric grating interface
#      at two central-difference steps $\delta_1 = 10\;\text{pm}$ and $\delta_2 = 5\;\text{pm}$ gives
#      $\partial F_{60^\circ} / \partial t = -1.2840 \times 10^8\;\text{m}^{-1}$ vs
#      $-1.2838 \times 10^8\;\text{m}^{-1}$ (relative difference **$1.41 \times 10^{-4}$**) and
#      $\partial F_{60^\circ} / \partial d = +1.1325 \times 10^8\;\text{m}^{-1}$ vs
#      $+1.1324 \times 10^8\;\text{m}^{-1}$ (relative difference **$1.22 \times 10^{-4}$**).
#   > **Forward pointer to Chapter 9:** Full free-form topology optimization and multi-parameter
#   > adjoint inverse design using `wakis.inverse.AdjointSimulation` and `MaterialGrid` are
#   > developed in detail in **Chapter 9 (`tutorials/09_inverse_design/`)**.
#
# ![Thickness-Depth Landscape & Gradient Verification (Janssen Fig. 4)](figures/17_janssen_fig4_thickness_depth_landscape.png)
#
# ---
#
# ### 5.4 Optimal Active-Layer Position Near $y_0 \approx 120\;\text{nm}$ (Janssen Fig. 5)
#
# Because the reciprocal electric field $\mathbf{E}_{\text{rec}}(x_0, y_0; \theta)$ is recorded
# throughout the entire unetched GaN volume in the **same** plane-wave simulations without any
# additional solves, Janssen *et al.* (Section 3.3, Fig. 5) evaluate the local extraction intensity
# $I_{\text{iso}}(x_0, y_0)$ as a function of the quantum-well position $(x_0, y_0)$ for a deep
# grating ($t = 700\;\text{nm}, d = 300\;\text{nm}$):
# - **Standing-wave antinode structure:** Reflections from the silver mirror at $y = 0$ create
#   pronounced horizontal standing-wave fringes with vertical period
#   $\Delta y \approx \lambda_0 / (2 n_{\text{GaN}}) = 91.8\;\text{nm}$ (Figure 18a).
# - **Optimal active-layer height $y_0 \approx 120\;\text{nm}$:** Because the Ag mirror reflection
#   phase is shifted from $\pi$ by the $\delta_s = 26.9\;\text{nm}$ skin-depth penetration, the
#   first constructive-interference antinode above the metal occurs at **$y_0 = 120.0\;\text{nm}$**
#   for both $\Lambda = 450\;\text{nm}$ and $\Lambda = 360\;\text{nm}$ gratings (and at
#   $y_0 = 115.0\;\text{nm}$ for the flat LED), in exact quantitative agreement with
#   Janssen *et al.* Fig. 5 ($\approx 120\;\text{nm}$).
# - **Wakis 2D Bloch FIT Cross-Check:** Figure 18(c) overlays the vertical standing-wave profile
#   $\langle |E_z(x, y_0)|^2 \rangle_x / |E_{\text{inc}}|^2$ computed by Wakis 2D Bloch FIT
#   (`solve_bloch_grating_unit_cell` using `wakis.materials.dispersive_lib["silver jc"]`) against
#   RCWA at $\theta = 0^\circ$ and $\theta = 25^\circ$, confirming that the 2D time-domain FIT
#   solver reproduces the standing-wave node/antinode heights and the $y_0 \approx 120\;\text{nm}$
#   peak directly on the Yee/FIT grid.
#
# ![Active-Layer Position Optimization (Janssen Fig. 5)](figures/18_janssen_fig5_active_layer_position.png)
#
# ---
#
# ### 5.5 Grating Pitch Scan ($\Lambda/\lambda_0 \approx 0.8$) & Headline $\approx 2\times$ Enhancement (Janssen Fig. 6)
#
# Finally, Janssen *et al.* (Section 3.4, Fig. 6) scan the grating pitch $\Lambda$ across
# $\Lambda / \lambda_0 \in [0.15, 1.35]$ for $t = 700\;\text{nm}, d = 300\;\text{nm}$ on silver:
# - **Sub-diffractive regime ($\Lambda / \lambda_0 < 1 / n_{\text{GaN}} \approx 0.41$):**
#   Even the first-order grating wavevector $2\pi / \Lambda > n_{\text{GaN}} k_0$ is too large to
#   fold guided slab modes into the air cone; the grating acts as an effective-medium anti-reflection
#   layer with no guided-mode resonance bands (left side of Figure 19a,b).
# - **Strong Bragg extraction band near $\Lambda / \lambda_0 \approx 0.8$:**
#   For $\Lambda / \lambda_0 \in [0.70, 0.85]$ ($\Lambda \in [315, 383]\;\text{nm}$), multiple
#   $m = \pm 1$ and $m = \pm 2$ guided-mode dispersion branches cross through the $\pm 30^\circ$
#   ($60^\circ$ full angle) and $\pm 45^\circ$ ($90^\circ$ full angle) cones for both $s$- and
#   $p$-polarizations (Figure 19a,b), producing a strong extraction maximum in the
#   $\Lambda / \lambda_0 \approx 0.70\text{--}0.80$ band (Figure 19c).
# - **Headline comparison (Optimized Grating vs Optimized Flat LED):**
#   Optimizing the flat GaN-on-Ag LED over both slab thickness $t$ and active-layer height $y_0$
#   yields $F_{\text{flat,opt},60^\circ} = 2.3284$ ($F_{\text{flat,opt},90^\circ} = 2.0460$).
#   The optimized grating achieves $F_{\text{grat,opt},60^\circ} = 5.0248$
#   ($F_{\text{grat,opt},90^\circ} = 4.1586$), delivering an overall extraction enhancement of
#   **$2.158\times$** in the $60^\circ$ cone and **$2.033\times$** in the $90^\circ$ cone
#   (and **$1.940\times$** / **$1.947\times$** at fixed $y_0 = 100\;\text{nm}$)—reproducing
#   Janssen *et al.*'s headline conclusion that **the optimized grating extracts $\approx 2\times$
#   the light of the optimized flat structure**.
#
# ![Grating Pitch Scan & 2x Enhancement Headline (Janssen Fig. 6)](figures/19_janssen_fig6_pitch_scan_enhancement.png)
#
# ---
#
# ### 5.6 Quantitative Comparison: Tutorial 7.6 vs Janssen, Wachters & Urbach (2010) & 2D FIT vs 2D FEM
#
# | Paper Figure & Physical Quantity | Janssen *et al.* (2010) 2D FEM Result | Tutorial 7.6 Result (Wakis 2D FIT / RCWA) | Where 2D FIT vs 2D FEM Differ & Physical Notes |
# | :--- | :--- | :--- | :--- |
# | **Fig. 2:** Shallow-grating resonances ($\Lambda=250\;\text{nm}, d=5\;\text{nm}, t=700\;\text{nm}, y_0=100\;\text{nm}$) | Sharp GMR peaks far above $t=698\;\text{nm}$ flat background; **$Q \sim 10^5$ (PEC)** vs **$Q \sim 10^3$ (Ag)** | PEC: $Q_{\text{mean}} = 7.57 \times 10^4$, $Q_{\max} = 4.00 \times 10^5$, Wakis `harminv` **$Q = 1.78 \times 10^5$**; Ag (`silver jc`): $Q_{\text{med}} = 4.21 \times 10^3$, Wakis `harminv` **$Q = 6.80 \times 10^3$** | `wakis.materials.dispersive_lib["silver jc"]` has $\varepsilon_{\text{Ag}}(450\;\text{nm}) = -7.08 - 0.22j$ (lower ohmic damping than Palik Ag $\mathrm{Im}\,\varepsilon \approx 0.74$), giving slightly higher $Q_{\text{Ag}} \approx 4\text{--}7 \times 10^3$ while preserving the $\sim 10^2\times$ PEC-to-Ag $Q$ drop |
# | **Fig. 4:** $(t, d)$ landscape for $\Lambda=450\;\text{nm}$ block grating ($t \in [450, 800]\;\text{nm}, d \in [0, 350]\;\text{nm}$) | Multiple local maxima along Fabry–Pérot / GMR ridges; steepest-ascent paths trapped in local optima | 23 local maxima in $(t, d)$; 4 steepest-ascent trajectories converge to distinct local optima ($F_{60^\circ} \in [2.20, 4.06]$, up to **$2.44\times$** flat at $y_0=100\;\text{nm}$) | Janssen's 2D FEM uses body-fitted unstructured triangles that move continuously with $(t, d)$; Wakis 2D FIT uses a structured Cartesian grid with Kottke subpixel smoothing or `MaterialGrid` density projection |
# | **Fig. 4 Gradient Check:** Shape derivative $\nabla_{(t, d)} F$ | Finite-difference / steepest-ascent gradients on $(t, d)$ | **Way 1 (Wakis `AdjointSimulation` + `MaterialGrid` vs central FD):** rel. err **$1.47 \times 10^{-6}$** ($t$) & **$1.51 \times 10^{-6}$** ($d$), $O(\delta^2)$ ratio **$4.00\times$**; **Way 2 (Geometric central FD):** rel. diff **$1.41 \times 10^{-4}$** ($t$) & **$1.22 \times 10^{-4}$** ($d$) | `wakis.inverse.AdjointSimulation` differentiates w.r.t. voxel density $\rho$ via exact time-domain VJP; chain-ruling through a smooth interface projection gives exact shape gradients in 2 solves |
# | **Fig. 5:** Optimal active-layer position $y_0$ ($t=700\;\text{nm}, d=300\;\text{nm}$) | First standing-wave antinode optimum near **$y_0 \approx 120\;\text{nm}$** above Ag mirror | **$y_{0,\text{opt}} = 120.0\;\text{nm}$** ($\Lambda=450\;\text{nm}$ and $\Lambda=360\;\text{nm}$ gratings) and $115.0\;\text{nm}$ (flat LED); verified directly by 2D Bloch FIT standing-wave profile | Both 2D FIT (with graded skin-depth mesh $\Delta y_{\min} \le \delta_s/6$) and 2D FEM resolve the $\delta_s = 26.9\;\text{nm}$ Goos–Hänchen phase penetration into silver |
# | **Fig. 6:** Pitch scan $\Lambda/\lambda_0 \in [0.15, 1.35]$ ($t=700\;\text{nm}, d=300\;\text{nm}$) | Cutoff for $\Lambda/\lambda_0 \lesssim 0.4$; strong extraction peak near **$\Lambda/\lambda_0 \approx 0.8$** | Cutoff below $\Lambda/\lambda_0 = 1/n_{\text{GaN}} = 0.41$; broad Bragg extraction peak across **$\Lambda/\lambda_0 \in [0.70, 0.85]$** ($p$-pol peak at $\Lambda/\lambda_0 = 0.78$) | Time-domain FIT launches a broadband Gaussian pulse (capable of extracting multi-wavelength spectra in 1 run), whereas frequency-domain FEM/RCWA solves one $\lambda_0$ per linear system |
# | **Headline:** Optimized grating vs optimized flat LED | **$\approx 2\times$** enhancement over optimized flat structure | **$2.158\times$** ($60^\circ$ cone) and **$2.033\times$** ($90^\circ$ cone) global; **$1.940\times$** ($60^\circ$) and **$1.947\times$** ($90^\circ$) at $y_0=100\;\text{nm}$; finite 7-period 2D FIT LED (`task_10`): **$2.016\times$** unpol (**$2.137\times$** TM) | Consistent $\approx 2.0\times$ extraction enhancement across both periodic unit-cell and finite-aperture 2D FIT models |
#
#
# ## 6. Finite-Device Validation Table, Resolution Study, Limits & Failure Modes
#
# ### 6.1 Quantitative Summary: `--quick` vs. `--full` vs. Reference / Analytic Theory
#
# The table below compares the Wakis 2D FIT simulation results at both `--quick` and `--full`
# resolutions against the exact analytical Transfer-Matrix Method (TMM), published benchmarks
# (Janssen et al. 2010; Benisty et al. 1998; Yao et al. 2022), and between Method (A) Direct
# Dipole Summation and Method (B) Reciprocal Plane-Wave Absorption:
#
# | Benchmark / Metric | Analytic / Reference Target | `--quick` Result | `--full` Result | Convergence / Agreement & Parameter Notes | Source / Figure |
# | :--- | :--- | :--- | :--- | :--- | :--- |
# | Flat GaN slab on PEC ($\theta=25^\circ$, TM $J_z$ standing wave) | Exact TMM standing wave (Eqs. 5–7, Benisty et al. 1998) | `task_00`, Bloch $\mathrm{cpw}=20$: L2 error **$2.26\%$** (sweep $[10, 14, 20, 28]$: $9.56\% \to 4.68\% \to 2.26\% \to 1.15\%$) | `task_00`, Bloch $\mathrm{cpw}=28$: L2 error **$1.15\%$** (sweep $[10, 14, 20, 28, 40]$: $9.56\% \to 4.68\% \to 2.26\% \to 1.15\% \to 0.58\%$) | Second-order $O(\Delta y^2)$ convergence across all 5 levels down to $0.58\%$ at $\mathrm{cpw}=40$ | Fig. 3, Fig. 4, Fig. 13 |
# | Flat GaN slab on PEC ($\theta=25^\circ$, $\mathrm{TE}_x, \mathrm{TE}_y$ standing wave) | Exact TMM standing wave (Benisty et al. 1998) | `task_00` ($\mathrm{cpw}=20$): $\mathrm{TE}_x: 2.29\%$, $\mathrm{TE}_y: 2.54\%$ | `task_00` ($\mathrm{cpw}=28$): $\mathrm{TE}_x:$ **$1.16\%$**, $\mathrm{TE}_y:$ **$1.28\%$** ($\mathrm{cpw}=40$: $\mathrm{TE}_x: 0.59\%$, $\mathrm{TE}_y: 0.65\%$) | Simultaneous TM+TE Huygens launch validates all 3 dipole components in 1 run/angle | Fig. 3, Fig. 4 |
# | Flat GaN slab on Drude Ag ($\varepsilon_{\mathrm{Ag}} = -7.08 - 0.29j$, $\delta_s = 26.91\,\mathrm{nm}$, $\delta_p = 13.45\,\mathrm{nm}$) | Complex-$\varepsilon_m$ TMM (Janssen et al. 2010) | `task_00` ($\mathrm{cpw}=20$, $8.0$ cells/$\delta_s$, $4.0$ cells/$\delta_p$, $\Delta y_{\min}=3.36\,\mathrm{nm}$): TM L2 error **$2.08\%$** ($\theta=0^\circ$), **$2.04\%$** ($\theta=25^\circ$) | `task_00` ($\mathrm{cpw}=28$, $10.0$ cells/$\delta_s$, $5.0$ cells/$\delta_p$, $\Delta y_{\min}=2.69\,\mathrm{nm}$): TM L2 error **$1.62\%$** ($\theta=0^\circ$), **$1.61\%$** ($\theta=25^\circ$) | Monotonically improves from $12.16\%$ ($1.59$ cells/$\delta_s$) and $6.37\%$ ($2.59$ cells/$\delta_s$) down to $1.62\%$ ($10.0$ cells/$\delta_s$) | Fig. 5, Fig. 15 |
# | Point-by-point reciprocity on 7-period grating LED ($\theta \in [-70^\circ, 70^\circ]$) | $U_\alpha(\theta; x_0)/U_{\mathrm{free,2D}} = |E_{\mathrm{rec},\alpha}(x_0, y_0; \theta)/E_{\mathrm{inc}}|^2$ | `task_01` ($\mathrm{cpw}=12$, $11$ sites $\times 3$ pols vs $15$ angles): max pt diff $J_x: 1.87\%$, $J_z: 2.70\%$, $J_y: 3.11\%$ | `task_01` ($\mathrm{cpw}=20$, $25$ sites $\times 3$ pols vs $29$ angles): matrix L2 error **$J_z: 0.46\%$, $J_x: 0.54\%$, $J_y: 1.68\%$**; max pt diff (at deep nodes $I \approx 0.06$, norm. by $\max(I, 0.1)$) $J_x: 4.07\%$, $J_y: 7.45\%$, $J_z: 11.76\%$ | Denser $25\times 29$ grid samples deep interference nodes ($I_z \approx 0.058$ vs $0.070$, $|\Delta I|=0.012$); global $(N_{\mathrm{pos}}\times N_\theta)$ matrix L2 error is $< 0.55\%$ for $J_z, J_x$ and $1.68\%$ for $J_y$ | Fig. 6 |
# | Ensemble angular pattern $\langle I_{\mathrm{unpol}}(\theta)\rangle_x$ (Method A vs Method B) | Exact agreement at sampled dipole sites | `task_01` ($\mathrm{cpw}=12$): $33$ direct runs ($2.34\,\mathrm{s}$/run) vs $15$ angle runs ($4.37\,\mathrm{s}$/angle); rel. L2 error **$0.083\%$** ($8.33 \times 10^{-4}$) | `task_01` ($\mathrm{cpw}=20$): $75$ direct runs ($6.58\,\mathrm{s}$/run) vs $29$ angle runs ($15.32\,\mathrm{s}$/angle); rel. L2 error **$0.066\%$** ($6.63 \times 10^{-4}$; by pol: $J_z: 0.078\%$, $J_x: 0.115\%$, $J_y: 0.374\%$) | Ensemble equivalence tightens from $0.083\%$ to $0.066\%$ at $\mathrm{cpw}=20$; 2D cost crossover $N_\theta^* = 3 N_{\mathrm{pos}}(t_{\mathrm{dip}}/t_{\mathrm{rec}}) \approx 32$ angles for $25$ sites | Fig. 7, Fig. 12 |
# | Grating vs Flat LED extraction enhancement on PEC ($\theta \in [-70^\circ, 70^\circ]$) | Janssen et al. (2010): $\sim 1.7\text{--}2.0\times$ enhancement | `task_01` ($\mathrm{cpw}=12$): Unpolarized **$\times 2.0694$**, TM ($J_z$) **$\times 1.8992$** | `task_01` ($\mathrm{cpw}=20$): Unpolarized **$\times 1.6753$** ($P_{\mathrm{grat}}=3.1683\,\mathrm{rad}$ vs $P_{\mathrm{flat}}=1.8912\,\mathrm{rad}$), TM ($J_z$) **$\times 1.4976$** ($P_{\mathrm{grat,TM}}=4.4572\,\mathrm{rad}$ vs $P_{\mathrm{flat,TM}}=2.9763\,\mathrm{rad}$) | Parameter note: when `h_grat=0` in `make_finite_led_solver`, $y_{\mathrm{emit,target}} = 0.32 d_{\mathrm{slab}} = 72.0\,\mathrm{nm}$ snaps to grid node $y_0 = 76.53\,\mathrm{nm}$ at $\mathrm{cpw}=12$ (closer to the $\lambda_{\mathrm{GaN}}/2 = 91.8\,\mathrm{nm}$ node, lowering $P_{\mathrm{flat}}$) vs $y_0 = 73.47\,\mathrm{nm}$ at $\mathrm{cpw}=20$ | Fig. 10 |
# | Grating vs Flat LED extraction enhancement on Drude Ag ($\theta \in [-70^\circ, 70^\circ]$) | Janssen et al. (2010) Ag-reflector grating LED ($\sim 1.7\text{--}2.0\times$); TMM flat-slab baseline at fixed $y_0 = 48.96\,\mathrm{nm}$ ($P_{\mathrm{flat,Ag,unpol}}=1.6039\,\mathrm{rad}$, $P_{\mathrm{flat,Ag,TM}}=3.0877\,\mathrm{rad}$) | `task_10` ($\mathrm{cpw}=12$, $N_{\mathrm{periods}}=5$, $8.0$ cells/$\delta_s$): Unpol **$\times 1.8822$** ($P_{\mathrm{int}}=3.0001\,\mathrm{rad}$), TM **$\times 2.0215$** ($6.1922\,\mathrm{rad}$); rel. change $8\to 10$ cells/$\delta_s$: $0.31\%$ | `task_10` ($\mathrm{cpw}=20$, $N_{\mathrm{periods}}=7$, $10.0$ cells/$\delta_s$ [$12.0$ max]): Unpol **$\times 2.0158$** ($P_{\mathrm{int}}=3.2333\,\mathrm{rad}$; $\times 2.0134, 3.2294\,\mathrm{rad}$ at $12$ cells/$\delta_s$), TM **$\times 2.1374$** ($6.5997\,\mathrm{rad}$; $\times 2.1366, 6.5971\,\mathrm{rad}$ at $12$ cells/$\delta_s$) | Converged to **$0.12\%$** rel. change across last two levels ($10\to 12$ cells/$\delta_s$, and $0.35\%$ from $8\to 10$ cells/$\delta_s$), vs **$4.76\%$** jump in under-resolved demo `task_09` ($1.6\to 2.6$ cells/$\delta_s$) | Fig. 15 |
# | Kottke vs Snapped & Un-snapped Staircase on 2D Grating LED (`task_02..08`) | `task_08` Kottke ref: `--full` $\mathrm{cpw}=40$ ($P_{\mathrm{int,unpol}} = 3.1937\,\mathrm{rad}$, $P_{\mathrm{int,TM}} = 4.3780\,\mathrm{rad}$); `--quick` $\mathrm{cpw}=22$ ($P_{\mathrm{int,unpol}} = 3.2135\,\mathrm{rad}$, $P_{\mathrm{int,TM}} = 4.4513\,\mathrm{rad}$) | `--quick` ($\mathrm{cpw} \in \{10, 14, 18\}$ vs $22$): Kottke L2 $16.37\% \to 5.27\% \to 2.89\%$ ($P_{\mathrm{int}}$ err $6.00\% \to 1.14\% \to 2.01\%$); Snapped Staircase L2 $12.02\% \to 7.20\% \to 3.96\%$ ($P_{\mathrm{int}}$ err $1.50\% \to 1.29\% \to 1.83\%$); Un-snapped Staircase L2 $12.02\% \to 15.12\% \to 32.31\%$ | `--full` ($\mathrm{cpw} \in \{14, 20, 28\}$ vs $40$): Kottke L2 **$6.87\% \to 4.71\% \to 2.23\%$** ($P_{\mathrm{int}}$ err **$1.77\% \to 1.11\% \to 0.62\%$**); Snapped Staircase L2 **$8.75\% \to 4.40\% \to 1.95\%$** ($P_{\mathrm{int}}$ err **$1.99\% \to 3.18\% \to 0.99\%$**); Un-snapped Staircase L2 **$16.49\% \to 3.88\% \to 8.21\%$** | Both Kottke ($2.23\%$ L2, $0.62\%$ $P_{\mathrm{int}}$ at $\mathrm{cpw}=28$) and Snapped Staircase ($1.95\%$ L2, $0.99\%$ $P_{\mathrm{int}}$ at $\mathrm{cpw}=28$) converge monotonically; Un-snapped Staircase rebounds from $3.88\%$ to $8.21\%$ L2 ($4.08\% \to 14.35\%$ TM) at $\mathrm{cpw}=28$ due to $+1.16\,\mathrm{nm}$ / $+1.31\,\mathrm{nm}$ upward rounding of $y_{\mathrm{base}}, d_{\mathrm{slab}}$ shifting the $\theta=0^\circ$ GMR peak | Fig. 13 |
#
# ### 6.2 Mesh Resolution Table (`--quick` & `--full` Measured)
#
# Every non-demo task (`demo=False`) strictly enforces all three resolution rules:
# 1. **Rule 1 ($\ge 10$ cells per wavelength in GaN, $n_{\mathrm{GaN}} = 2.45$, $\lambda_{\mathrm{GaN}} = 183.67\,\mathrm{nm}$):**
#    $\Delta x_{\mathrm{bg}} \le 18.37\,\mathrm{nm}$ in every non-demo task ($14 \le \mathrm{cpw} \le 40$ in `--full`, $\Delta x_{\mathrm{bg}} \in [4.59, 13.12]\,\mathrm{nm}$), with $\mathrm{cpw} \ge 20$ for reference tasks (`task_00` at $\mathrm{cpw}=28$ and `task_08` at $\mathrm{cpw}=40$).
# 2. **Rule 2 ($\ge 4$ cells across the smallest geometric feature):**
#    The smallest feature is the grating etch depth $h_{\mathrm{grat}} = 72.0\,\mathrm{nm}$ (and trapezoidal tooth top width $w_{\mathrm{top}} = 94.5\,\mathrm{nm}$); clamping $\Delta x_{\mathrm{slab}} \le \min(\lambda_0/(n_{\mathrm{GaN}}\,\mathrm{cpw}),\,h_{\mathrm{grat}}/4) = 18.00\,\mathrm{nm}$ guarantees $\ge 4.00$ cells across $h_{\mathrm{grat}}$ even at $\mathrm{cpw}=10$, and $5.49\text{--}15.68$ cells across $h_{\mathrm{grat}}$ across all non-demo `--full` tasks.
# 3. **Rule 3 ($\ge 4$ cells per skin depth near Drude Ag via `graded_axis`):**
#    At $\lambda_0 = 450\,\mathrm{nm}$, Drude silver has $\varepsilon_{\mathrm{Ag}} = -7.08 - 0.29j$ ($\sqrt{\varepsilon_{\mathrm{Ag}}} = 0.0543 - 2.6614j$, extinction coefficient $\kappa = 2.6614$). We report both standard skin-depth definitions:
#    - **Field amplitude skin depth:** $\delta_s = \lambda_0 / (2\pi\kappa) = 26.908\,\mathrm{nm}$,
#    - **Power (intensity) skin depth:** $\delta_p = \lambda_0 / (4\pi\kappa) = \delta_s / 2 = 13.454\,\mathrm{nm}$.
#    At the primary accepted `--full` level ($\mathrm{cpsd}_s = 10.0$, $\mathrm{cpsd}_p = 5.0$, $\Delta y_{\min} = 2.58\text{--}2.69\,\mathrm{nm}$; and $\mathrm{cpsd}_s = 8.0$, $\mathrm{cpsd}_p = 4.0$, $\Delta y_{\min} = 3.26\text{--}3.36\,\mathrm{nm}$ in `--quick`), every non-demo Ag task satisfies $\ge 4$ cells per skin depth under **both** definitions (`task_09` is the single explicitly labelled `demo=True` under-resolved comparison run at $\mathrm{cpsd}_s \in \{1.6, 2.6\}$).
#
# | Task | Name | Demo? | Mode | $\mathrm{cpw}$ | $\Delta x_{\mathrm{bg}}$ [nm] | $\Delta x_{\min}$ [nm] | Min Feat [nm] | Cells / Feat | $\delta_s$ / $\delta_p$ [nm] | Cells / $\delta_s$ ($\delta_p$) | $N_{\mathrm{cells}}$ | $N_{\mathrm{steps}}$ | Elapsed Time (`--quick` / `--full` actual) |
# | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | 0 | `flat_slab_bloch_tmm` | False | quick / full | 20 / 28 | 5.60 / 5.51 | 3.36 / 2.69 | 72.0 | 12.86 / 13.08 | 26.91 / 13.45 | 8.0 (4.0) / 10.0 (5.0) | 179 / 189 | 7958 / 9171 | 61.8 s / 106.4 s (1.8 m) |
# | 1 | `direct_vs_reciprocity_ensemble` | False | quick / full | 12 / 20 | 15.31 / 9.18 | 15.31 / 9.18 | 72.0 | 4.70 / 7.84 | — | — | 7434 / 20650 | 3051 / 5085 | 239.5 s / 1562.3 s (26.0 m) |
# | 2 | `kottke_coarse` | False | quick / full | 10 / 14 | 18.00 / 13.12 | 18.00 / 13.12 | 72.0 | 4.00 / 5.49 | — | — | 5436 / 10143 | 2594 / 3559 | 45.9 s / 180.0 s (3.0 m) |
# | 3 | `kottke_medium` | False | quick / full | 14 / 20 | 13.12 / 9.18 | 13.12 / 9.18 | 72.0 | 5.49 / 7.84 | — | — | 10143 / 20650 | 3559 / 5085 | 115.2 s / 548.6 s (9.1 m) |
# | 4 | `kottke_fine` | False | quick / full | 18 / 28 | 10.20 / 6.56 | 10.20 / 6.56 | 72.0 | 7.06 / 10.98 | — | — | 16695 / 40061 | 4576 / 7119 | 244.0 s / 1676.7 s (27.9 m) |
# | 5 | `staircase_coarse` | False | quick / full | 10 / 14 | 18.00 / 12.86 | 18.00 / 12.00 | 72.0 | 4.00 / 6.00 | — | — | 5436 / 11183 | 2594 / 3720 | 50.8 s / 417.7 s (7.0 m) |
# | 6 | `staircase_medium` | False | quick / full | 14 / 20 | 12.86 / 9.00 | 12.00 / 8.74 | 72.0 | 6.00 / 8.00 | — | — | 11183 / 21672 | 3720 / 5241 | 243.6 s / 1094.2 s (18.2 m) |
# | 7 | `staircase_fine` | False | quick / full | 18 / 28 | 10.00 / 6.55 | 9.00 / 6.43 | 72.0 | 8.00 / 11.00 | — | — | 18970 / 40837 | 4849 / 7234 | 494.9 s / 3314.1 s (55.2 m) |
# | 8 | `kottke_reference` | False | quick / full | 22 / 40 | 8.35 / 4.59 | 8.35 / 4.59 | 72.0 | 8.62 / 15.68 | — | — | 24700 / 81282 | 5593 / 10170 | 433.9 s / 4769.6 s (79.5 m) |
# | 9 | `ag_grating_underresolved_demo` | **True** | quick / full | 10 / 10 | 18.37 / 18.37 | 15.59 / 15.59 | 72.0 | 4.00 / 4.00 | 26.91 / 13.45 | 1.6 (0.8), 2.6 (1.3) | 5236 / 6556 | 2521 / 2521 | 58.6 s / 123.2 s (2.1 m) |
# | 10 | `ag_grating_resolution_study` | False | quick / full | 12 / 20 | 15.31 / 9.18 | 3.26 / 2.58 | 72.0 | 5.00 / 8.00 | 26.91 / 13.45 | 8.0 (4.0) / 10.0 (5.0) | 11154 / 33040 | 8051 / 10505 | 547.2 s / 4191.4 s (69.9 m) |
#
# ### 6.3 Kottke Subpixel Smoothing vs. Staircase Discretization (Geometry Rounding & Guided-Mode Resonance Shift)
#
# In the `--full` resolution study ($\mathrm{cpw} \in \{14, 20, 28\}$ compared against the $\mathrm{cpw}=40$
# Kottke reference `task_08`, where $P_{\mathrm{int,unpol}} = 3.1937\,\mathrm{rad}$ and
# $P_{\mathrm{int,TM}} = 4.3780\,\mathrm{rad}$), **both Kottke subpixel smoothing and the snapped-grid
# staircase (`snap_grid=True`) converge monotonically**, whereas the **un-snapped staircase (`snap_grid=False`)
# fails to converge monotonically**:
# - **Kottke subpixel smoothing (`task_02..04` vs `task_08` at `--full`):**
#   The unpolarized angular L2 error decreases monotonically across $\mathrm{cpw} \in \{14, 20, 28\}$ as
#   **$6.87\% \to 4.71\% \to 2.23\%$** (TM: **$12.96\% \to 8.61\% \to 3.57\%$**; $\mathrm{TE}_x$:
#   **$2.96\% \to 1.62\% \to 1.54\%$**), and the angle-integrated extraction power $P_{\mathrm{int,unpol}}$
#   converges smoothly from $3.2503 \to 3.2293 \to 3.2134 \to 3.1937\,\mathrm{rad}$ (relative error
#   **$1.77\% \to 1.11\% \to 0.62\%$** vs $\mathrm{cpw}=40$; relative change between the last two levels
#   $\mathrm{cpw}=28 \to 40$ is **$0.62\%$** unpolarized and **$1.01\%$** TM).
#   (In `--quick`, across $\mathrm{cpw} \in \{10, 14, 18\}$ vs $\mathrm{cpw}=22$, Kottke angular L2 error
#   similarly falls monotonically: **$16.37\% \to 5.27\% \to 2.89\%$**.)
# - **Snapped-grid staircase (`snap_grid=True`, `task_05..07` vs `task_08` at `--full`):**
#   When `snap_grid=True` (default for `averaging="staircase"`), `make_finite_led_solver` snaps
#   $\Delta x = \Lambda / \lceil \Lambda / \Delta x_{\max}\rceil$ ($\Lambda/\Delta x \in \{21, 30, 42\}$ at
#   $\mathrm{cpw} \in \{14, 20, 28\}$, and $\{15, 21, 27\}$ in `--quick`) and aligns dual-cell $y$ boundaries
#   with $y_{\mathrm{base}} = 153.00\,\mathrm{nm}$ and $d_{\mathrm{slab}} = 225.00\,\mathrm{nm}$
#   ($h_{\mathrm{grat}} = 72.00\,\mathrm{nm}$ at every $\mathrm{cpw}$). With the period and slab heights held
#   invariant, the snapped-staircase angular L2 error vs the $\mathrm{cpw}=40$ reference decreases
#   **strictly monotonically** across $\mathrm{cpw} \in \{14, 20, 28\}$:
#   **$8.75\% \to 4.40\% \to 1.95\%$** unpolarized (**$13.30\% \to 5.68\% \to 2.12\%$** TM;
#   **$9.82\% \to 6.03\% \to 3.73\%$** $\mathrm{TE}_x$), reaching $P_{\mathrm{int,unpol}} = 3.2254\,\mathrm{rad}$
#   (**$0.99\%$** vs $\mathrm{cpw}=40$ Kottke) and $P_{\mathrm{int,TM}} = 4.3880\,\mathrm{rad}$ (**$0.23\%$** vs
#   $\mathrm{cpw}=40$ Kottke) at $\mathrm{cpw}=28$. (In `--quick` vs $\mathrm{cpw}=22$, snapped staircase
#   L2 error also falls monotonically: **$12.02\% \to 7.20\% \to 3.96\%$**.)
# - **Un-snapped staircase (`snap_grid=False` in `task_05..07`):**
#   On an un-snapped uniform grid ($\Delta x = \Delta y = \min(\lambda_0/(n_{\mathrm{GaN}}\,\mathrm{cpw}),\,h_{\mathrm{grat}}/4)$),
#   the un-snapped staircase angular L2 error vs $\mathrm{cpw}=40$ oscillates non-monotonically across
#   $\mathrm{cpw} \in \{14, 20, 28\}$ (**$16.49\% \to 3.88\% \to 8.21\%$** unpolarized;
#   **$25.86\% \to 4.08\% \to 14.35\%$** TM), rebounding upward by more than $2\times$ (and $>3.5\times$ in TM)
#   when refining from $\mathrm{cpw}=20$ to $\mathrm{cpw}=28$! In `--quick` ($\mathrm{cpw} \in \{10, 14, 18\}$
#   vs $\mathrm{cpw}=22$), the un-snapped staircase error grows even more dramatically with resolution
#   (**$12.02\% \to 15.12\% \to 32.31\%$** unpolarized; **$20.28\% \to 22.38\% \to 58.05\%$** TM).
#
# Systematic inspection of the discretized permittivity tensors (`measure_staircase_grating_geometry` in
# `_helpers.py`) establishes the exact physical mechanism behind this non-monotonic un-snapped staircase behavior:
#
# 1. **Identical Solver Configuration & Exact Lateral Symmetry:**
#    At every $\mathrm{cpw}$, the un-snapped staircase and Kottke runs share identical time steps $\Delta t$,
#    grid coordinates $(x_i, y_j)$, active-layer indices $(y_{\mathrm{emit},Ez}, y_{\mathrm{emit},Ey})$,
#    Huygens monitor geometry $(W_{\mathrm{mon}}, y_{\mathrm{mon}})$, PML thickness, and source normalization;
#    moreover, the staircase $\boldsymbol{\varepsilon}^{-1}$ tensor is strictly $x$-symmetric (`symmetry_diff = 0`).
# 2. **Non-Monotonic Staircase Geometry Rounding Across $\mathrm{cpw}$:**
#    In `assign_materials(..., averaging="staircase")`, tangential $E_z$ and $E_x$ nodes sit on primary grid
#    lines $y_j = j\Delta y$ with dual cells $[(j-0.5)\Delta y, (j+0.5)\Delta y]$. Thresholding the $16\times 16$
#    subcell midpoint fill fraction at $w \ge 0.5$ snaps horizontal dielectric interfaces to half-integer
#    dual-cell boundaries $y_{\mathrm{stair}} = (\lfloor y_{\mathrm{int}}/\Delta y + \tfrac{1}{32}\rfloor + \tfrac{1}{2})\Delta y$,
#    while in $x$ any non-integer $\Lambda/\Delta x$ forces adjacent tooth-to-tooth periods to alternate by
#    $0.5\text{--}1.0\,\mathrm{cell}$:
#    - **$\mathrm{cpw}=10$ (`--quick`, $\Delta x = \Delta y = 18.000\,\mathrm{nm}$, clamped by $h_{\mathrm{grat}}/4$):**
#      Because $\gcd(\Lambda=270\,\mathrm{nm}, y_{\mathrm{base}}=153\,\mathrm{nm}, d_{\mathrm{slab}}=225\,\mathrm{nm}, h_{\mathrm{grat}}=72\,\mathrm{nm}) = 9\,\mathrm{nm}$
#      and $\Delta x = 2 \times 9\,\mathrm{nm}$, the ratios $\Lambda/\Delta x = 15.000$ (integer),
#      $y_{\mathrm{base}}/\Delta y = 8.500$ (half-integer), $d_{\mathrm{slab}}/\Delta y = 12.500$ (half-integer),
#      and $h_{\mathrm{grat}}/\Delta y = 4.000$ (integer) align **exactly** with the staircase grid!
#      Every tooth has identical period $\Lambda_{\mathrm{stair}} = 270.00\,\mathrm{nm}$ (span $0.00\,\mathrm{nm}$),
#      $y_{\mathrm{base,stair}} = 153.00\,\mathrm{nm}$, $d_{\mathrm{slab,stair}} = 225.00\,\mathrm{nm}$,
#      $h_{\mathrm{grat,stair}} = 72.00\,\mathrm{nm}$, and mean duty cycle $\bar{f}_{\mathrm{stair}} = 0.5000$.
#    - **$\mathrm{cpw}=14$ (`--full` coarse / `--quick` medium, $\Delta x = \Delta y = 13.120\,\mathrm{nm}$):**
#      The grid is incommensurate ($\Lambda/\Delta x = 20.580$, $y_{\mathrm{base}}/\Delta y = 11.662$,
#      $d_{\mathrm{slab}}/\Delta y = 17.150$). Local tooth periods alternate between $262.39\,\mathrm{nm}$ and
#      $275.51\,\mathrm{nm}$ (span $13.12\,\mathrm{nm} = 1.00\Delta x$), $y_{\mathrm{base,stair}} = 150.875\,\mathrm{nm}$
#      ($-2.125\,\mathrm{nm}$), $d_{\mathrm{slab,stair}} = 229.592\,\mathrm{nm}$ ($+4.592\,\mathrm{nm}$), and the
#      tooth height swells to $h_{\mathrm{grat,stair}} = 78.717\,\mathrm{nm}$ ($+6.717\,\mathrm{nm}$, $+9.3\%$).
#    - **$\mathrm{cpw}=18$ (`--quick` fine, $\Delta x = \Delta y = 10.204\,\mathrm{nm}$):**
#      Here $y_{\mathrm{base}}/\Delta y = 14.994 \in [14.969, 15.0]$ and $d_{\mathrm{slab}}/\Delta y = 22.050$
#      **both round upward by $+0.5\Delta y$**, shifting the unetched slab base to
#      $y_{\mathrm{base,stair}} = 158.163\,\mathrm{nm}$ (**$+5.163\,\mathrm{nm}$**) and total slab thickness to
#      $d_{\mathrm{slab,stair}} = 229.592\,\mathrm{nm}$ (**$+4.592\,\mathrm{nm}$**), while $\Lambda/\Delta x = 26.460$
#      causes adjacent teeth to alternate in width by $10.204\,\mathrm{nm}$ ($\bar{f}_{\mathrm{stair}} = 0.4743$).
#    - **$\mathrm{cpw}=20$ (`--full` medium, $\Delta x = \Delta y = 9.184\,\mathrm{nm}$):**
#      Because $d_{\mathrm{slab}}/\Delta y = 225.0 / 9.1837 = 24.500$ is an exact half-integer and
#      $y_{\mathrm{base}}/\Delta y = 16.660$ rounds to $16.5\Delta y = 151.531\,\mathrm{nm}$ ($-1.469\,\mathrm{nm}$),
#      the un-snapped total slab height $d_{\mathrm{slab,stair}} = 225.000\,\mathrm{nm}$ lands **exactly** on the
#      physical slab thickness ($h_{\mathrm{grat,stair}} = 73.469\,\mathrm{nm}$, local periods $266.33\text{--}275.51\,\mathrm{nm}$,
#      $\bar{f}_{\mathrm{stair}} = 0.5053$). Consequently, the un-snapped staircase error happens to dip to
#      $3.88\%$ L2 ($4.08\%$ TM) at $\mathrm{cpw}=20$.
#    - **$\mathrm{cpw}=28$ (`--full` fine, $\Delta x = \Delta y = 6.560\,\mathrm{nm}$):**
#      At $\mathrm{cpw}=28$, $y_{\mathrm{base}}/\Delta y = 23.324$ and $d_{\mathrm{slab}}/\Delta y = 34.300$ both
#      round **upward** to $23.5\Delta y$ and $34.5\Delta y$, shifting the un-snapped staircase interfaces to
#      $y_{\mathrm{base,stair}} = 154.155\,\mathrm{nm}$ (**$+1.155\,\mathrm{nm}$**) and
#      $d_{\mathrm{slab,stair}} = 226.312\,\mathrm{nm}$ (**$+1.312\,\mathrm{nm}$**, $h_{\mathrm{grat,stair}} = 72.157\,\mathrm{nm}$),
#      while adjacent teeth alternate in width by $6.560\,\mathrm{nm}$ (period span $3.280\,\mathrm{nm} = 0.50\Delta x$,
#      $\bar{f}_{\mathrm{stair}} = 0.4938$).
# 3. **Guided-Mode Resonance (GMR) Shift at Normal Incidence ($\theta = 0^\circ$):**
#    First-order Bragg diffraction ($2\pi/\Lambda = 1.667 k_0$) folds the fundamental GaN guided slab mode
#    into the radiative air cone near normal incidence $\theta = 0^\circ$. Whenever the un-snapped staircase
#    rounds $y_{\mathrm{base}}$ and $d_{\mathrm{slab}}$ upward (by $+1.16\,\mathrm{nm}$ / $+1.31\,\mathrm{nm}$ at
#    $\mathrm{cpw}=28$, or $+5.16\,\mathrm{nm}$ / $+4.59\,\mathrm{nm}$ at $\mathrm{cpw}=18$), the higher effective
#    index of the thicker slab shifts the Bragg GMR peak onto $\theta = 0^\circ$: in Figure 13(a),
#    $I_{\mathrm{TM}}(0^\circ)$ on the un-snapped grid rebounds from $4.0829$ ($\mathrm{cpw}=20$) to
#    $4.8608$ ($\mathrm{cpw}=28$, vs $3.8318$ at $\mathrm{cpw}=40$ Kottke and $3.9873$ at $\mathrm{cpw}=28$
#    snapped staircase), and $I_{\mathrm{unpol}}(0^\circ)$ rebounds from $1.8165$ to $2.0930$ (vs $1.7204$
#    reference). Meanwhile, the non-resonant $\mathrm{TE}_x$ channel converges monotonically even on the
#    un-snapped staircase grid ($8.79\% \to 4.54\% \to 2.20\%$ across $\mathrm{cpw} \in \{14, 20, 28\}$).
#
# | Series & $\mathrm{cpw}$ (`--full` vs $\mathrm{cpw}=40$ ref; `[--quick vs cpw=22]`) | $\Delta x$ / $\Delta y_{\mathrm{slab}}$ [nm] | $\Lambda_{\mathrm{stair}}$ (min–max, span) [nm] | $y_{\mathrm{base,stair}}$ / $d_{\mathrm{slab,stair}}$ / $h_{\mathrm{grat,stair}}$ [nm] | Mean Duty $\bar{f}_{\mathrm{stair}}$ | $I_{\mathrm{unpol}}(0^\circ)$ / $I_{\mathrm{TM}}(0^\circ)$ | $P_{\mathrm{int,unpol}}$ [rad] (Rel. Err vs $\mathrm{cpw}=40$ `[vs 22]`) | Angular L2 Err vs $\mathrm{cpw}=40$ (Unpol / TM) `[vs 22]` |
# | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | **Physical Target / `--full` Ref (`cpw=40` Kottke, `task_08`)** | $4.59$ / $4.59$ | $270.00$ (exact polygon) | $153.00$ / $225.00$ / $72.00$ | $0.5000$ | $1.7204$ / $3.8318$ | $3.1937$ (ref) | — |
# | *(Quick Ref: `cpw=22` Kottke, `--quick task_08`)* | $8.35$ / $8.35$ | $270.00$ (exact polygon) | $153.00$ / $225.00$ / $72.00$ | $0.5000$ | $1.8151$ / $4.0977$ | $3.2135$ (`[ref]`) | — |
# | *(Quick Coarse: `cpw=10` Kottke, `--quick task_02`)* | $18.00$ / $18.00$ | $270.00$ (subpixel) | $153.00$ / $225.00$ / $72.00$ | $0.5000$ | $2.4992$ / $6.1053$ | $3.4062$ (`[6.00%]`) | `[16.37% / 29.97%]` |
# | **Kottke `cpw=14`** (`task_02` full / `task_03` quick) | $13.12$ / $13.12$ | $270.00$ (subpixel) | $153.00$ / $225.00$ / $72.00$ | $0.5000$ | $2.0281$ / $4.7462$ | $3.2503$ ($1.77\%$) `[1.14%]` | **$6.87\%$ / $12.96\%$** `[5.27% / 9.64%]` |
# | *(Quick Fine: `cpw=18` Kottke, `--quick task_04`)* | $10.20$ / $10.20$ | $270.00$ (subpixel) | $153.00$ / $225.00$ / $72.00$ | $0.5000$ | $1.9170$ / $4.4198$ | $3.2781$ (`[2.01%]`) | `[2.89% / 4.79%]` |
# | **Kottke `cpw=20`** (`task_03` full) | $9.18$ / $9.18$ | $270.00$ (subpixel) | $153.00$ / $225.00$ / $72.00$ | $0.5000$ | $1.9419$ / $4.4671$ | $3.2293$ ($1.11\%$) | **$4.71\%$ / $8.61\%$** |
# | **Kottke `cpw=28`** (`task_04` full) | $6.56$ / $6.56$ | $270.00$ (subpixel) | $153.00$ / $225.00$ / $72.00$ | $0.5000$ | $1.8145$ / $4.0850$ | $3.2134$ ($0.62\%$) | **$2.23\%$ / $3.57\%$** |
# | *(Quick Coarse: `cpw=10` Staircase, `--quick task_05`)* | $18.00$ / $18.00$ | $270.00\text{--}270.00$ ($0.00$) | $153.00$ / $225.00$ / $72.00$ | $0.5000$ | $2.2242$ / $5.5410$ | $3.2617$ (`[1.50%]`) | `[12.02% / 20.28%]` |
# | **Staircase Snapped `cpw=14`** (`task_05` full / `task_06` quick) | $12.86$ / $12.00$ | $270.00\text{--}270.00$ ($0.00$) | $153.00$ / $225.00$ / $72.00$ | $0.4921$ | $1.9902$ / $4.7450$ | $3.2574$ ($1.99\%$) `[1.29%]` | **$8.75\%$ / $13.30\%$** `[7.20% / 10.05%]` |
# | *(Quick Fine: `cpw=18` Staircase Snapped, `--quick task_07`)* | $10.00$ / $9.00$ | $270.00\text{--}270.00$ ($0.00$) | $153.00$ / $225.00$ / $72.00$ | $0.5093$ | $1.9309$ / $4.5268$ | $3.2723$ (`[1.83%]`) | `[3.96% / 5.84%]` |
# | **Staircase Snapped `cpw=20`** (`task_06` full) | $9.00$ / $9.00$ | $270.00\text{--}270.00$ ($0.00$) | $153.00$ / $225.00$ / $72.00$ | $0.5000$ | $1.8633$ / $4.2552$ | $3.2951$ ($3.18\%$) | **$4.40\%$ / $5.68\%$** |
# | **Staircase Snapped `cpw=28`** (`task_07` full) | $6.43$ / $6.55$ | $270.00\text{--}270.00$ ($0.00$) | $153.00$ / $225.00$ / $72.00$ | $0.5000$ | $1.7733$ / $3.9873$ | $3.2254$ ($0.99\%$) | **$1.95\%$ / $2.12\%$** |
# | **Staircase Un-snapped `cpw=14`** (`task_05` full / `task_06` quick) | $13.12$ / $13.12$ | $262.39\text{--}275.51$ ($13.12$) | $150.88$ / $229.59$ / $78.72$ | $0.4905$ | $2.2870$ / $5.2936$ | $3.5671$ ($11.69\%$) `[10.94%]` | **$16.49\%$ / $25.86\%$** `[15.12% / 22.38%]` |
# | *(Quick Fine: `cpw=18` Staircase Un-snapped, `--quick task_07`)* | $10.20$ / $10.20$ | $265.31\text{--}275.51$ ($10.20$) | $158.16$ / $229.59$ / $71.43$ | $0.4743$ | $3.0693$ / $7.8175$ | $3.5205$ (`[9.55%]`) | `[32.31% / 58.05%]` |
# | **Staircase Un-snapped `cpw=20`** (`task_06` full) | $9.18$ / $9.18$ | $266.33\text{--}275.51$ ($9.18$) | $151.53$ / $225.00$ / $73.47$ | $0.5053$ | $1.8165$ / $4.0829$ | $3.2791$ ($2.67\%$) | **$3.88\%$ / $4.08\%$** |
# | **Staircase Un-snapped `cpw=28`** (`task_07` full) | $6.56$ / $6.56$ | $268.95\text{--}272.23$ ($3.28$) | $154.16$ / $226.31$ / $72.16$ | $0.4938$ | $2.0930$ / $4.8608$ | $3.2820$ ($2.76\%$) | **$8.21\%$ / $14.35\%$** (rebound!) |
#
# ![Kottke vs Staircase Convergence](figures/13_resolution_convergence_kottke_vs_staircase.png)
#
# ### 6.4 Drude Silver (Ag) Metal Skin-Depth Resolution Study (Graded Mesh)
#
# In metallic back-reflector LEDs (Janssen et al. 2010), the optical field penetrates
# into the Drude silver mirror ($y \le 0$) with an evanescent decay length governed by
# $\kappa = |\mathrm{Im}\sqrt{\varepsilon_{\mathrm{Ag}}}| = 2.6614$ at $\lambda_0 = 450\,\mathrm{nm}$:
# - **Amplitude skin depth:** $\delta_s = \lambda_0 / (2\pi\kappa) = 26.908\,\mathrm{nm}$ (where $|\mathbf{E}|$ decays by $e^{-1}$),
# - **Power skin depth:** $\delta_p = \lambda_0 / (4\pi\kappa) = 13.454\,\mathrm{nm}$ (where $|\mathbf{E}|^2$ decays by $e^{-1}$).
#
# On a uniform grid at $\mathrm{cpw}=18$ ($\Delta y = 10.20\,\mathrm{nm}$), the skin layer is
# sampled by only $2.59$ cells per $\delta_s$ ($1.30$ cells per $\delta_p$), leaving a $6.37\%\text{--}6.42\%$
# standing-wave phase/amplitude L2 error inside the GaN slab. Refining the entire 2D grid
# uniformly to $\Delta x = \Delta y = 2.58\,\mathrm{nm}$ would increase the 2D cell count by $> 25\times$.
# Instead, we use `wakis.templates.meshing.graded_axis` (`grading = 1.3 <= 1.4`) along $y$
# to place a fine skin-depth band ($\Delta y_{\min} = \delta_s / \mathrm{cpsd}_s$) across the
# Ag/GaN interface $y \in [-4\delta_s, +2.5\delta_s]$, grading smoothly to $\Delta y_{\mathrm{slab}}$
# across the grating teeth and $\Delta y_{\mathrm{air}}$ in the air cladding.
#
# Figure 15 (`figures/15_metal_skin_depth_resolution_study.png`) compares the intentionally
# under-resolved demonstration task (`task_09`, $\mathrm{cpsd}_s \in \{1.6, 2.6\}$, $\mathrm{cpsd}_p \in \{0.8, 1.3\}$)
# against the resolved graded-mesh ladder (`task_10`, $\mathrm{cpsd}_s \in \{4.0, 6.0, 8.0, 10.0, 12.0\}$,
# $\mathrm{cpsd}_p \in \{2.0, 3.0, 4.0, 5.0, 6.0\}$ at $\mathrm{cpw}=20, N_{\mathrm{periods}}=7$ in `--full`;
# $\mathrm{cpsd}_s \in \{4.0, 6.0, 8.0, 10.0\}$ at $\mathrm{cpw}=12, N_{\mathrm{periods}}=5$ in `--quick`):
# - **1D Bloch Flat-Slab Standing-Wave Error vs Exact TMM (`task_00`, Fig. 15a):**
#   Refining $\mathrm{cpsd}_s$ across $[1.59, 2.59, 4.47, 6.0, 8.0, 10.0]$ (with $12.0$ in `task_10`) monotonically reduces
#   the TM standing-wave L2 error at $\theta = 0^\circ$ from **$12.16\% \to 6.37\% \to 4.46\% \to 3.01\% \to 2.08\% \to 1.62\% \to 1.37\%$**
#   (and at $\theta = 25^\circ$ from **$12.06\% \to 6.42\% \to 4.20\% \to 2.89\% \to 2.04\% \to 1.61\%$**).
# - **2D Grating LED Angle-Integrated Extraction Power & Enhancement (`task_09` & `task_10`, Fig. 15b–c):**
#   In the `--full` under-resolved demo (`task_09`, $N_{\mathrm{periods}}=7$), stepping from $1.6$ to $2.6$ cells/$\delta_s$
#   causes a **$4.76\%$ jump** in unpolarized extraction power ($3.5262 \to 3.3658\,\mathrm{rad}$, enhancement
#   $2.1985\times \to 2.0985\times$; in `--quick` at $N_{\mathrm{periods}}=5$, a $5.81\%$ jump from $3.1303 \to 2.9586\,\mathrm{rad}$).
#   Across the `--full` resolved graded ladder (`task_10`, $4.0 \to 6.0 \to 8.0 \to 10.0 \to 12.0$ cells/$\delta_s$),
#   $P_{\mathrm{grat,unpol}}$ converges smoothly (**$3.2858 \to 3.2562 \to 3.2447 \to 3.2333 \to 3.2294\,\mathrm{rad}$**)
#   and the unpolarized extraction enhancement over the TMM flat-slab baseline ($P_{\mathrm{flat,Ag,unpol}} = 1.6039\,\mathrm{rad}$,
#   $P_{\mathrm{flat,Ag,TM}} = 3.0877\,\mathrm{rad}$) stabilizes at
#   **$2.0486\times \to 2.0302\times \to 2.0230\times \to 2.0158\times \to 2.0134\times$**
#   (TM enhancement **$2.1471\times \to 2.1421\times \to 2.1403\times \to 2.1374\times \to 2.1366\times$**;
#   in `--quick`, $1.9181\times \to 1.8959\times \to 1.8822\times \to 1.8765\times$ unpolarized).
# - **Accepted Resolution Level ($\mathrm{cpsd}_s = 10.0$, $\mathrm{cpsd}_p = 5.0$, $\Delta y_{\min} = 2.577\,\mathrm{nm}$ in `--full`; $\mathrm{cpsd}_s = 8.0$, $\mathrm{cpsd}_p = 4.0$, $\Delta y_{\min} = 3.264\,\mathrm{nm}$ in `--quick`):**
#   Using a $< 1\%$ relative-change acceptance threshold on the target extraction metrics, the step-to-step
#   change in $P_{\mathrm{grat,unpol}}$ in `--full` is **$0.91\%$** ($4\to 6$ cells/$\delta_s$), **$0.36\%$**
#   ($6\to 8$ cells/$\delta_s$), **$0.35\%$** ($8\to 10$ cells/$\delta_s$), and **$0.12\%$** between the last
#   two levels ($10\to 12$ cells/$\delta_s$, `rel_change_last_two_levels = 0.0012 < 0.01`, and $0.04\%$ for TM),
#   while providing $5.0$ cells per power skin depth $\delta_p$ and $10.0$ cells per amplitude skin depth $\delta_s$
#   at the primary accepted `--full` level.
#
# | Task & Level (`--full`, $N_{\mathrm{periods}}=7$) | Demo? | $\mathrm{cpw}_{\mathrm{bulk}}$ | $\mathrm{cpsd}_s$ ($\delta_s = 26.91\,\mathrm{nm}$) | $\mathrm{cpsd}_p$ ($\delta_p = 13.45\,\mathrm{nm}$) | $\Delta y_{\min}$ [nm] | $N_{\mathrm{cells}}$ / $N_{\mathrm{steps}}$ | 1D Standing-Wave L2 Err ($\theta=0^\circ$) | $P_{\mathrm{grat,unpol}}$ / $P_{\mathrm{grat,TM}}$ [rad] | Enhancement $\eta_{\mathrm{grat}}/\eta_{\mathrm{flat}}$ (Unpol / TM) | Step-to-Step $|\Delta P_{\mathrm{unpol}}|/P$ |
# | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | `task_09` (`cpsd_s = 1.6`) | **True** | $10$ | $1.6$ | $0.8$ | $15.588$ | $6556$ / $2521$ | $12.16\%$ | $3.5262$ / $6.9744$ | $2.1985\times$ / $2.2588\times$ | — |
# | `task_09` (`cpsd_s = 2.6`) | **True** | $10$ | $2.6$ | $1.3$ | $9.792$ | $7301$ / $3218$ | $6.37\%$ | $3.3658$ / $6.7878$ | $2.0985\times$ / $2.1984\times$ | $4.76\%$ (under-resolved jump) |
# | `task_10` (`cpsd_s = 4.0`) | False | $20$ | $4.0$ | $2.0$ | $6.120$ | $26255$ / $5649$ | $4.46\%$ | $3.2858$ / $6.6295$ | $2.0486\times$ / $2.1471\times$ | $2.43\%$ (vs `cpsd_s=2.6`) |
# | `task_10` (`cpsd_s = 6.0`) | False | $20$ | $6.0$ | $3.0$ | $4.451$ | $28320$ / $6853$ | $3.01\%$ | $3.2562$ / $6.6140$ | $2.0302\times$ / $2.1421\times$ | $0.91\%$ |
# | `task_10` (`cpsd_s = 8.0`) | False | $20$ | $8.0$ | $4.0$ | $3.264$ | $30680$ / $8627$ | $2.08\%$ | $3.2447$ / $6.6084$ | $2.0230\times$ / $2.1403\times$ | $0.36\%$ |
# | **`task_10` (`cpsd_s = 10.0`, accepted)** | **False** | **$20$** | **$10.0$** | **$5.0$** | **$2.577$** | **$33040$ / $10505$** | **$1.62\%$** | **$3.2333$ / $6.5997$** | **$2.0158\times$ / $2.1374\times$** | **$0.35\%$** |
# | `task_10` (`cpsd_s = 12.0`, finest ref) | False | $20$ | $12.0$ | $6.0$ | $2.225$ | $35105$ / $11951$ | $1.37\%$ | $3.2294$ / $6.5971$ | $2.0134\times$ / $2.1366\times$ | **$0.12\%$** (`rel_change = 0.0012`) |
#
# ![Drude Ag Metal Skin-Depth Resolution Study](figures/15_metal_skin_depth_resolution_study.png)
#
# ### 6.5 Limits & Failure Modes
#
# 1. **When Reciprocity Loses its Cost Advantage ($N_\theta > 3 N_{\mathrm{sites}}$):**
#    If a device has only a single localized emitter (e.g. a single quantum dot in a
#    nanocavity or a single nitrogen-vacancy center, $N_{\mathrm{sites}} = 1$), direct
#    simulation requires at most $3$ runs ($J_x, J_y, J_z$) and yields the entire continuous
#    far-field hemisphere via `Near2FarMonitor` in one post-processing step. Reciprocity
#    is advantageous specifically for **spatially extended incoherent emitters** ($N_{\mathrm{sites}} \gg N_\theta / 3$).
# 2. **Total Internal Reflection & Non-Radiative / Guided Power ($\eta_{\mathrm{EQE}}$ Normalization):**
#    Reciprocal plane waves launched from the air cladding ($|\sin\theta| < 1$) probe only
#    the **radiative air cone** ($|k_x| < k_0$). They do *not* directly measure the power
#    lost to trapped waveguide modes ($k_0 < |k_x| < n_{\mathrm{slab}} k_0$) or surface-plasmon
#    polaritons on the metal mirror, which determines the Purcell factor and total spontaneous
#    emission rate $\Gamma_{\mathrm{tot}}(x_0, y_0)$. If the local density of optical states (LDOS)
#    varies strongly across the active layer, the true internal quantum weight of each dipole
#    depends on $\Gamma_{\mathrm{tot}}(x_0, y_0)$, which still requires either a few representative
#    direct dipole Purcell runs or an evanescent-wave / Green's function integration.
# 3. **Nonlinear, Lasing, or Stimulated-Emission Regimes:**
#    Helmholtz reciprocity ($\boldsymbol{\varepsilon}^T = \boldsymbol{\varepsilon}, \boldsymbol{\mu}^T = \boldsymbol{\mu}$)
#    holds for linear, time-invariant media (including lossy dispersive metals like silver),
#    and the incoherent power-addition theorem holds only for uncorrelated spontaneous emission
#    below the lasing threshold. Superluminescent diodes, amplified spontaneous emission (ASE),
#    and lasers require coupled rate-equation or Maxwell-Bloch solvers.
# 4. **Finite-Aperture Diffraction at Grazing Angles ($|\theta| \to 90^\circ$):**
#    In a finite non-periodic FDTD/FIT domain of width $W_{\mathrm{mon}}$, a reciprocal
#    plane wave launched at grazing incidence ($\cos\theta \to 0$) has a projected beam width
#    $W_{\mathrm{mon}}\cos\theta$ that shrinks to zero. Consequently, angles beyond
#    $|\theta| \gtrsim 75^\circ$ suffer from aperture-edge diffraction unless either the lateral
#    domain $W_{\mathrm{mon}}$ is widened or periodic Bloch boundary conditions are used.
#
# ### 6.6 Exercises
#
# 1. **Quantum-Well Depth Optimization (Standing-Wave Node vs Antinode):**
#    Using `tmm_slab_emission` and `solve_bloch_flat_slab`, sweep the quantum-well height
#    $y_0 \in [10\,\mathrm{nm}, 260\,\mathrm{nm}]$ and find the first and second extraction
#    optima for TM and TE dipoles on both PEC and silver mirrors. Verify that placing
#    the QW at the standing-wave node $y_0 = \lambda_0 / (2 n_{\mathrm{GaN}}) \approx 91.8\,\mathrm{nm}$
#    suppresses normal-incidence emission by more than two orders of magnitude.
# 2. **Grating Pitch & First-Order Bragg Angle Tuning:**
#    Change `PITCH` from $270\,\mathrm{nm}$ to $350\,\mathrm{nm}$ and $450\,\mathrm{nm}$ in
#    `run_finite_led_reciprocity`. Measure how the angular extraction peaks shift according
#    to the 1D Bragg relation $k_0 \sin\theta_{\mathrm{peak}} = \pm (\beta_{\mathrm{guided}} - 2\pi m / \Lambda)$.
# 3. **Anisotropic Quantum-Well Dipole Orientation (In-Plane TE Dominance):**
#    Compressively strained c-plane InGaN/GaN quantum wells emit predominantly via in-plane
#    ($J_x, J_y$) dipoles with an in-plane/out-of-plane ratio of roughly $90\% : 10\%$ rather
#    than isotropic $1:1:1$. Modify the polarization weights in `run_task_1_ensemble` to
#    $w_x = w_y = 0.45, w_z = 0.10$ and compute the resulting extraction efficiency enhancement.
#
# ### 6.7 References (Verified Open-Access & Archival Sources)
#
# 1. **O. T. A. Janssen, A. J. H. Wachters, and H. P. Urbach**, *"Efficient optimization method for the
#    light extraction from periodically modulated LEDs using reciprocity"*,
#    Opt. Express **18**(24), 24522–24535 (2010), DOI: `10.1364/OE.18.024522`
#    (primary reference for Tutorial 7.6: Helmholtz reciprocity formulation of incoherent LED emission as absorption,
#    blue $\lambda_0 = 450\,\mathrm{nm}$ GaN LED with metallic Ag back-reflector, equivalence between
#    direct dipole ensemble sum and reciprocal plane-wave absorption, and optimization of Figs. 2, 4, 5, 6).
# 2. **C. Roques-Carmes et al.**, *"A framework for scintillation in nanophotonics"*,
#    Science **375**(6583), eabm9293 (2022), DOI: `10.1126/science.abm9293`
#    (verified open-access full text, Eqs. 1–2: reciprocal effective absorption volume
#    $V_{\mathrm{eff}}^{(i)}(\omega, \Omega) = \int_{V_S} \mathrm{d}\mathbf{r}\,|\mathbf{E}^{(i)}(\mathbf{r},\omega,\Omega)|^2 / |\mathbf{E}_{\mathrm{inc}}^{(i)}(\omega,\Omega)|^2$
#    for spatially extended incoherent emitters).
# 3. **J. Yao, F. Verdugo, R. E. Christiansen, and S. G. Johnson**, *"Trace formulation for photonic
#    inverse design with incoherent sources"*,
#    Struct. Multidiscip. Optim. **65**, 336 (2022), DOI: `10.1007/s00158-022-03389-5`
#    (verified open-access full text, Sec. 2.3, Sec. 4.2 & Appendix 8.2: reciprocal emission into a
#    small number of far-field channels vs direct dipole runs; periodic LED geometry $\Lambda = 0.60\lambda_0$,
#    $H_d = 0.50\lambda_0$).
# 4. **L. Zschiedrich, H. J. Greiner, S. Burger, and F. Schmidt**, *"Numerical analysis of light extraction
#    efficiency of LEDs using the finite element method and reciprocity"*,
#    Proc. SPIE **8641**, Light-Emitting Diodes: Materials, Devices, and Applications for Solid State Lighting XVII,
#    86410B (2013), DOI: `10.1117/12.2001132`.
# 5. **W. Lukin**, *"Wide-angle interference of light and light emission from thin dielectric layers"*,
#    J. Opt. Soc. Am. **69**(4), 545–555 (1979); and **H. Benisty, H. De Neve, and C. Weisbuch**,
#    *"Impact of planar microcavity effects on light extraction — Part I: Basic concepts and analytical trends"*,
#    IEEE J. Quantum Electron. **34**(9), 1612–1631 (1998), DOI: `10.1109/3.709578`
#    (analytical transfer-matrix formulas for horizontal and vertical dipoles in a planar slab on a mirror, Eqs. 5–7 & Figs. 3–4).
# 6. **A. David, H. Benisty, and C. Weisbuch**, *"Fast factorization of the extraction of light in
#    photonic crystal light-emitting diodes"*,
#    Rep. Prog. Phys. **75**(12), 126501 (2012), DOI: `10.1088/0034-4885/75/12/126501`
#    (comprehensive review of GaN thin-film and grating/photonic-crystal LED light extraction physics, Fig. 10).
# 7. **C. Kottke, A. Farjadpour, and S. G. Johnson**, *"Perturbation theory for anisotropic dielectric
#    interfaces, and application to subpixel smoothing of discretized numerical methods"*,
#    Phys. Rev. E **77**(3), 036611 (2008), DOI: `10.1103/PhysRevE.77.036611`
#    (anisotropic dielectric tensor subpixel smoothing used in `assign_materials` for sloped grating teeth, Fig. 1 & Fig. 13).

# %% [markdown]
# ## 7. Command-Line Interface (`--list-tasks`, `--task`, `--quick`, `--full`, `--plot`)

# %%
def main(argv=None):
    """CLI entrypoint for Tutorial 7.6 (`led_reciprocity.py`)."""
    parser = argparse.ArgumentParser(
        description="Tutorial 7.6 · Thin-film LED: emission computed as absorption (Helmholtz reciprocity)."
    )
    parser.add_argument("--list-tasks", action="store_true", help="Print JSON task manifest and exit.")
    parser.add_argument("--task", type=int, default=None, help="Run a single task index I.")
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "data",
        help="Output directory for task_*.npz and task_*.json (default: ./data).",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=HERE / "figures",
        help="Output directory for generated figures (default: ./figures).",
    )
    parser.add_argument("--quick", action="store_true", help="Run optimization study at --quick sizing and plot.")
    parser.add_argument("--full", action="store_true", help="Run all tasks at --full sizing and plot.")
    parser.add_argument("--run-all-tasks", action="store_true", help="Force re-running all 11 finite-LED tasks.")
    parser.add_argument("--plot", action="store_true", help="Generate all 20 figures from existing data/.")

    args = parser.parse_args(argv)
    mode = "full" if args.full else "quick"
    tasks = build_tasks(quick=(not args.full))

    if args.list_tasks:
        print(json.dumps({"mode": mode, "n_tasks": len(tasks), "tasks": tasks}, indent=2))
        return 0

    if args.task is not None:
        matched = [t for t in tasks if t["index"] == args.task]
        if not matched:
            raise ValueError(f"Unknown --task {args.task}; valid indices are 0..{len(tasks) - 1}.")
        summary = execute_task(matched[0], args.out)
        print(json.dumps(summary, indent=2))
        return 0

    if args.plot:
        import resource

        plot_all(args.out, args.fig_dir)
        summaries = []
        for idx_t in range(len(tasks)):
            p_js = args.out / f"task_{idx_t:02d}.json"
            if p_js.exists():
                summaries.append(json.loads(p_js.read_text()))
        if summaries:
            if not args.quick and not args.full and int(summaries[0].get("cpw", 20)) >= 28:
                mode = "full"
            total_s = sum(float(s.get("elapsed_s", 0.0)) for s in summaries)
            peak_rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
            manifest = {
                "mode": mode,
                "total_elapsed_s": round(total_s, 2),
                "peak_rss_mb": peak_rss_mb,
                "tasks": summaries,
            }
            if (args.out / "janssen_optimization.json").exists():
                manifest["janssen_optimization"] = json.loads(
                    (args.out / "janssen_optimization.json").read_text()
                )
            (args.out / "summary.json").write_text(json.dumps(manifest, indent=2))
        print(f"[7.6:{mode}] Generated 20 figures in {args.fig_dir}.")
        return 0

    if args.quick or args.full:
        import resource

        t_all = time.perf_counter()
        summaries = []
        for t in tasks:
            p_js = args.out / f"task_{t['index']:02d}.json"
            p_npz = args.out / f"task_{t['index']:02d}.npz"
            if args.quick and not args.run_all_tasks and p_js.exists() and p_npz.exists():
                summaries.append(json.loads(p_js.read_text()))
                continue
            print(f"[7.6:{mode}] Running task {t['index']}/{len(tasks) - 1}: {t['name']} ...", flush=True)
            s = execute_task(t, args.out)
            summaries.append(s)
            print(f"  -> done in {s['elapsed_s']:.2f} s", flush=True)
        print(f"[7.6:{mode}] Running Janssen et al. (2010) optimization study ...", flush=True)
        janssen_summary = run_janssen_optimization_study(quick=(not args.full), out_dir=args.out)
        print(f"  -> Janssen study done in {janssen_summary['elapsed_s']:.2f} s", flush=True)
        plot_all(args.out, args.fig_dir)
        total_s = time.perf_counter() - t_all
        peak_rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
        manifest_mode = "full" if (summaries and int(summaries[0].get("cpw", 20)) >= 28) else mode
        manifest = {
            "mode": manifest_mode,
            "total_elapsed_s": round(
                sum(float(s.get("elapsed_s", 0.0)) for s in summaries)
                if manifest_mode == "full" and args.quick
                else total_s,
                2,
            ),
            "peak_rss_mb": peak_rss_mb,
            "tasks": summaries,
            "janssen_optimization": janssen_summary,
        }
        (args.out / "summary.json").write_text(json.dumps(manifest, indent=2))
        print(
            f"[7.6:{mode}] Completed in {total_s:.2f} s "
            f"(peak RSS: {peak_rss_mb:.1f} MB); 20 figures saved in {args.fig_dir}."
        )
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
