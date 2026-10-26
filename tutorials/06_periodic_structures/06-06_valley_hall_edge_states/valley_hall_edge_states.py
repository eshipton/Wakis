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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eshipton/Wakis/blob/main/tutorials/06_periodic_structures/06-06_valley_hall_edge_states/valley_hall_edge_states.ipynb)
#
# # Tutorial 6.6 · Valley-Hall Photonic Crystal and Domain-Wall Kink Edge States
# <h3 align="center">Hexagonal lattices on a Cartesian FIT grid, supercell band unfolding, Fukui–Hatsugai–Suzuki Berry curvature, and valley-Hall 120° Z-bend waveguiding</h3>
#
# ---
#
# ### Abstract
#
# **Two-dimensional triangular and honeycomb photonic crystals host massless Dirac cones at the $K$ and $K'$ corners of the hexagonal Brillouin zone when spatial inversion symmetry is preserved. Breaking inversion symmetry—by making the two sublattice rod radii unequal, $r_A = r_0(1+\delta)$ and $r_B = r_0(1-\delta)$—lifts the Dirac degeneracy and opens a valley-Hall photonic bandgap endowed with opposite Berry curvature $\Omega_1(\mathbf{k})$ at the $K$ and $K'$ valleys (Ma & Shvets, 2016 [3]; Gao et al., 2018 [4]). In this tutorial we represent hexagonal lattices on a Cartesian Finite Integration Technique (`SolverFIT3D`) mesh via their centred-rectangular supercell ($L_x = a$, $L_y = \sqrt{3}a$, containing two primitive Bravais points separated by $\boldsymbol{\tau} = (a/2, \sqrt{3}a/2)$), unfold supercell eigenmodes back onto the primitive hexagonal Brillouin zone, extract gauge-invariant Berry curvature and valley Chern numbers directly from 2D PWE (`wakis.lattices.PWE2D`) and 2D FIT ring-down fields using the Fukui–Hatsugai–Suzuki plaquette formula [5], compute the projected dispersion $\omega(k_x)$ of Jackiw–Rebbi domain-wall kink states along a $\mathrm{VPC}_+/\mathrm{VPC}_-$ interface, and compare straight-reference-normalized transmission across a sharp $120^\circ$ Z-bend waveguide against a tuned line-defect photonic-crystal waveguide across the full bulk valley bandgap.**
#
# ### Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. **Represent** a 2D triangular or honeycomb lattice on an orthogonal Cartesian FIT mesh using its **centred-rectangular supercell** ($a \times \sqrt{3}a$), apply Bloch boundary phases $(\phi_x, \phi_y) = (e^{i k_x a}, e^{i k_y \sqrt{3}a})$ at any wavevector $\mathbf{k}$ in the hexagonal Brillouin zone, and **unfold** supercell bands back onto the primitive zone using the internal translation operator $\hat{T}_{\boldsymbol{\tau}} \psi_{\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\boldsymbol{\tau}}\psi_{\mathbf{k}}(\mathbf{r})$ [8, 9].
# 2. **Validate** 2D FIT time-domain band extraction (`SolverFIT3D` + `harminv`) against an independent 2D plane-wave-expansion solver (`PWE2D`, formulated from Ho, Chan & Soukoulis [1], Plihal & Maradudin [2], and Joannopoulos et al. [10]) for both symmetric Dirac-cone ($\delta = 0$) and broken-inversion valley-Hall ($\delta \neq 0$) honeycomb crystals.
# 3. **Derive and compute** the **Berry curvature** $\Omega_1(\mathbf{k})$ and **valley Chern patch numbers** around $K$ and $K'$ from both PWE eigenvectors and 2D FIT Bloch envelopes via the gauge-invariant **Fukui–Hatsugai–Suzuki** link-variable method [5], verifying the exact sign reversal $\Omega_1(\mathbf{k}, -\delta) = -\Omega_1(\mathbf{k}, +\delta)$ and $\Omega_1(-\mathbf{k}, \delta) = -\Omega_1(\mathbf{k}, \delta)$.
# 4. **Analyze** the **Jackiw–Rebbi domain-wall kink states** between opposite valley phases ($\delta > 0$ vs $\delta < 0$) alongside a tuned trivial line-defect waveguide ($r_{\mathrm{def}} = 0.65\,r_0 = 0.117\,a$, centered on the $A_1\text{–}B_2$ defect channel at $y_w + 0.25\,L_y$) in a ribbon supercell ($a \times N_y\sqrt{3}a$), programmatically derive the single-mode and mini-gap sub-windows from the ribbon bands, and compare $120^\circ$ Z-bend transmission normalized to a matched straight reference waveguide across the full bulk valley bandgap in 2D FIT using `FluxMonitor` and `DFTMonitor`.
#
# ---
#
# ## 1. Theory and Derivations
#
# ### 1.1 Centred-rectangular supercell and primitive-zone band unfolding (**derived here**)
#
# A 2D triangular Bravais lattice of pitch $a$ is spanned by the non-orthogonal primitive vectors
#
# $$\mathbf{a}_1 = a(1, 0), \qquad \mathbf{a}_2 = a\left(\frac{1}{2}, \frac{\sqrt{3}}{2}\right), \qquad A_{\mathrm{prim}} = |\mathbf{a}_1 \times \mathbf{a}_2| = \frac{\sqrt{3}}{2}a^2, \tag{1}$$
#
# with reciprocal-lattice basis vectors $\mathbf{b}_1 = \frac{2\pi}{a}(1, -1/\sqrt{3})$ and $\mathbf{b}_2 = \frac{2\pi}{a}(0, 2/\sqrt{3})$ satisfying $\mathbf{a}_i \cdot \mathbf{b}_j = 2\pi\delta_{ij}$. The first Brillouin zone is a regular hexagon with corners at the inequivalent valleys $\mathbf{K} = (\frac{4\pi}{3a}, 0)$ and $\mathbf{K}' = (-\frac{4\pi}{3a}, 0)$ and edge midpoints at $\mathbf{M} = (\frac{\pi}{a}, \frac{\pi}{\sqrt{3}a})$.
#
# To discretize this lattice on an orthogonal Cartesian FIT grid without staircasing the periodic boundaries, we form the **centred-rectangular supercell**
#
# $$\mathbf{A}_1 = \mathbf{a}_1 = (a, 0), \qquad \mathbf{A}_2 = 2\mathbf{a}_2 - \mathbf{a}_1 = (0, \sqrt{3}\,a), \qquad A_{\mathrm{sup}} = \sqrt{3}\,a^2 = 2 A_{\mathrm{prim}}. \tag{2}$$
#
# Under Wakis's $\exp(+i\omega t)$ time-harmonic and $\psi_{\mathbf{k}}(\mathbf{r}+\mathbf{L}) = \exp(+i\mathbf{k}\cdot\mathbf{L})\psi_{\mathbf{k}}(\mathbf{r})$ spatial Bloch convention, any wavevector $\mathbf{k} = (k_x, k_y)$ in the hexagonal Brillouin zone is enforced on the rectangular supercell faces via
#
# $$\phi_x = \exp(+i k_x a), \qquad \phi_y = \exp(+i k_y \sqrt{3}\,a). \tag{3}$$
#
# Because $A_{\mathrm{sup}} = 2 A_{\mathrm{prim}}$, the supercell Brillouin zone has half the area of the primitive hexagonal Brillouin zone and folds wavevectors separated by $\mathbf{G}_{\mathrm{fold}} = (0, \frac{2\pi}{\sqrt{3}a})$ onto the same supercell Bloch phase (since $e^{i\mathbf{G}_{\mathrm{fold}}\cdot\mathbf{A}_1} = 1$ and $e^{i\mathbf{G}_{\mathrm{fold}}\cdot\mathbf{A}_2} = e^{i 2\pi} = 1$). However, the crystal permittivity $\varepsilon(\mathbf{r})$ is invariant under the **internal primitive translation** $\boldsymbol{\tau} = \mathbf{a}_2 = (a/2, \sqrt{3}a/2)$, which satisfies
#
# $$\exp(i\mathbf{G}_{\mathrm{fold}}\cdot\boldsymbol{\tau}) = \exp(i\pi) = -1. \tag{4}$$
#
# Consequently, the translation operator $\hat{T}_{\boldsymbol{\tau}} f(\mathbf{r}) = f(\mathbf{r} + \boldsymbol{\tau})$ commutes with the Maxwell operator and partitions the supercell eigenstates at $\mathbf{k}$ into two orthogonal subspaces (**derived here** following the spectral unfolding formalism of Popescu & Zunger [8] and Ku, Berlijn & Lee [9]):
#
# * **Unfolded primitive branch** (originating at $\mathbf{k}$): $\hat{T}_{\boldsymbol{\tau}}\psi_{\mathbf{k}}(\mathbf{r}) = +e^{+i\mathbf{k}\cdot\boldsymbol{\tau}}\psi_{\mathbf{k}}(\mathbf{r})$, with spectral weight
#
# $$W_n(\mathbf{k}) = \frac{\left\|\psi_{n,\mathbf{k}}(\mathbf{r}) + e^{-i\mathbf{k}\cdot\boldsymbol{\tau}}\psi_{n,\mathbf{k}}(\mathbf{r}+\boldsymbol{\tau})\right\|_\varepsilon^2}{2\left(\|\psi_{n,\mathbf{k}}(\mathbf{r})\|_\varepsilon^2 + \|\psi_{n,\mathbf{k}}(\mathbf{r}+\boldsymbol{\tau})\|_\varepsilon^2\right)} = 1. \tag{5}$$
#
# * **Folded supercell branch** (originating at $\mathbf{k} + \mathbf{G}_{\mathrm{fold}}$): $\hat{T}_{\boldsymbol{\tau}}\psi_{\mathbf{k}}(\mathbf{r}) = -e^{+i\mathbf{k}\cdot\boldsymbol{\tau}}\psi_{\mathbf{k}}(\mathbf{r})$, giving $W_n(\mathbf{k}) = 0$.
#
# In time-domain 2D FIT simulations (`fit_supercell_bands_at_k`), recording a pair of probes $s_1(t) = E_z(\mathbf{r}_1, t)$ and $s_2(t) = E_z(\mathbf{r}_2, t)$ at $\boldsymbol{\tau}$-paired grid nodes ($\Delta\mathbf{r} = \mathbf{r}_2 - \mathbf{r}_1 \equiv \boldsymbol{\tau} \pmod{\mathbf{A}_{1,2}}$) projects the time series directly onto the unfolded and folded branches prior to harmonic inversion (`harminv`):
#
# $$s_{\pm}(t) = \frac{1}{2}\Bigl(s_1(t) \pm e^{-i\mathbf{k}\cdot\Delta\mathbf{r}}\,s_2(t)\Bigr). \tag{6}$$
#
# ---
#
# ### 1.2 Honeycomb Dirac cone, inversion-breaking mass, and valley Berry curvature (**derived here**)
#
# A honeycomb crystal contains two dielectric rods per primitive unit cell at sublattice positions $\mathbf{d}_A = \frac{1}{3}(\mathbf{a}_1 + \mathbf{a}_2)$ and $\mathbf{d}_B = \frac{2}{3}(\mathbf{a}_1 + \mathbf{a}_2)$ (four rods $A_1, A_2, B_1, B_2$ in the $a \times \sqrt{3}a$ supercell) with radii
#
# $$r_A = r_0(1 + \delta), \qquad r_B = r_0(1 - \delta), \tag{7}$$
#
# in air ($\varepsilon_{\mathrm{bg}} = 1$, $\varepsilon_{\mathrm{rod}} = 12$). For TM polarization ($\mathbf{E} = E_z\hat{\mathbf{z}}$), the frequency-domain Maxwell equation is $-\nabla_\perp^2 E_z = (\omega/c)^2 \varepsilon(\mathbf{r}) E_z$.
#
# When $\delta = 0$ ($r_A = r_B = r_0$), the crystal possesses $C_{6v}$ point-group symmetry. At the valley corners $\mathbf{K}_{\tau_v} = (\tau_v \frac{4\pi}{3a}, 0)$ ($\tau_v = +1$ for $K$, $\tau_v = -1$ for $K'$), the little group $C_{3v}$ has a two-dimensional irreducible representation $E$, enforcing a **massless Dirac degeneracy** $\omega_1(\mathbf{K}_{\tau_v}) = \omega_2(\mathbf{K}_{\tau_v}) = \omega_D$.
#
# When $\delta \neq 0$, spatial inversion $\mathcal{P}$ (which swaps sublattices $A \leftrightarrow B$) is broken, reducing the point group to $C_{3v}$ at $\Gamma$ and $C_3$ at $K, K'$. By first-order electromagnetic perturbation theory (**derived here**, valid for $|\delta| \ll 1$ at fixed $r_0$ and $\varepsilon_{\mathrm{rod}}$), the dielectric perturbation $\delta\varepsilon(\mathbf{r})$ shifts the sublattice-$A$-polarized state ($|A\rangle$) and sublattice-$B$-polarized state ($|B\rangle$) in opposite directions by
#
# $$\Delta\omega_{\mathrm{signed}} \equiv \omega_B(\mathbf{K}) - \omega_A(\mathbf{K}) \approx \alpha\,\delta\,\frac{2\pi c}{a} \equiv 2 m, \tag{8}$$
#
# opening a complete valley-Hall bandgap of width $\Delta\omega_K = |\omega_2(\mathbf{K}) - \omega_1(\mathbf{K})| = 2|m| \propto |\delta|$. Expanding the $\mathbf{k}\cdot\mathbf{p}$ Maxwell operator to first order in $\mathbf{q} = \mathbf{k} - \mathbf{K}_{\tau_v}$ yields the two-band **massive Dirac Hamiltonian** (**derived here** following Haldane & Raghu [6] and Ma & Shvets [3]):
#
# $$H_{\mathrm{eff}}(\mathbf{q}) = \omega_D\,\sigma_0 + v_D\left(\tau_v q_x \sigma_x + q_y \sigma_y\right) + m\,\sigma_z, \qquad \omega_{\pm}(\mathbf{q}) = \omega_D \pm \sqrt{v_D^2 |\mathbf{q}|^2 + m^2}, \tag{9}$$
#
# where $v_D$ is the isotropic Dirac cone group velocity. Writing $\mathbf{d}(\mathbf{q}) = (\tau_v v_D q_x,\, v_D q_y,\, m)$, the lower-band ($-$) eigenstate carries the analytical **Berry curvature** (**derived here**):
#
# $$\Omega_1(\mathbf{q}) = i\left(\langle\partial_{q_x} u_1 | \partial_{q_y} u_1\rangle - \langle\partial_{q_y} u_1 | \partial_{q_x} u_1\rangle\right) = \frac{1}{2}\frac{\mathbf{d}\cdot(\partial_{q_x}\mathbf{d} \times \partial_{q_y}\mathbf{d})}{|\mathbf{d}|^3} = -\frac{\tau_v v_D^2 m}{2\left(v_D^2(q_x^2 + q_y^2) + m^2\right)^{3/2}}. \tag{10}$$
#
# Integrating Eq. (10) over the 2D $\mathbf{q}$-plane around valley $\tau_v$ gives the quantized half-integer **valley Chern number**
#
# $$C_{\tau_v} = \frac{1}{2\pi}\int_{\mathrm{valley }\,\tau_v} \Omega_1(\mathbf{q})\,\mathrm{d}^2q = -\frac{1}{2}\,\tau_v\,\mathrm{sgn}(m) = -\frac{1}{2}\,\tau_v\,\mathrm{sgn}(\delta). \tag{11}$$
#
# On a discretized wavevector mesh $(k_{x,i}, k_{y,j})$, we evaluate $\Omega_1$ without gauge-fixing ambiguities using the **Fukui–Hatsugai–Suzuki (FHS)** plaquette phase [5]:
#
# $$\Omega_{i,j} = \frac{1}{\Delta k_x \Delta k_y}\,\mathrm{Im}\,\ln\Bigl(\langle u_{i,j}|u_{i+1,j}\rangle_\varepsilon \langle u_{i+1,j}|u_{i+1,j+1}\rangle_\varepsilon \langle u_{i+1,j+1}|u_{i,j+1}\rangle_\varepsilon \langle u_{i,j+1}|u_{i,j}\rangle_\varepsilon\Bigr), \tag{12}$$
#
# where $\langle u | v \rangle_\varepsilon = \int_{\mathrm{cell}} u^*(\mathbf{r})\,\varepsilon(\mathbf{r})\,v(\mathbf{r})\,\mathrm{d}^2r$ is the Maxwell inner product.
#
# ---
#
# ### 1.3 Jackiw–Rebbi domain-wall kink states and $120^\circ$ bend protection (**derived here**)
#
# Across a domain wall at $y = y_0$ separating two inverted valley-Hall domains—$\mathrm{VPC}_+$ ($\delta > 0, m > 0$ for $y < y_0$) and $\mathrm{VPC}_-$ ($\delta < 0, m < 0$ for $y > y_0$)—the valley Chern number jumps by
#
# $$\Delta C_{\tau_v} = C_{\tau_v}(y > y_0) - C_{\tau_v}(y < y_0) = +\tau_v\,\mathrm{sgn}(\delta) = \pm 1. \tag{13}$$
#
# Replacing $q_y \to -i\partial_y$ in Eq. (9) with a spatially varying mass $m(y)$ that changes sign at $y = y_0$ yields the 1D Dirac equation
#
# $$\left[v_D\left(\tau_v q_x \sigma_x - i\sigma_y\partial_y\right) + m(y)\sigma_z\right]\boldsymbol{\psi}(y) = (\omega - \omega_D)\boldsymbol{\psi}(y). \tag{14}$$
#
# Multiplying by $\sigma_y$ shows that the spinor eigenstate $\sigma_x \boldsymbol{\chi}_\eta = \eta\boldsymbol{\chi}_\eta$ ($\eta = -\mathrm{sgn}(m(+\infty) - m(-\infty)) = +\mathrm{sgn}(\delta)$) satisfies $\partial_y \psi = -(|m(y)|/v_D)\psi$, giving the exponentially localized **Jackiw–Rebbi topological kink state** (**derived here**):
#
# $$\boldsymbol{\psi}_{q_x}(x, y) \propto \exp(i q_x x)\,\exp\!\left(-\frac{1}{v_D}\left|\int_{y_0}^y m(y')\,\mathrm{d}y'\right|\right)\boldsymbol{\chi}_\eta, \qquad \omega_{\mathrm{edge}}(q_x) = \omega_D + \eta\,\tau_v v_D q_x. \tag{15}$$
#
# Equation (15) exhibits **valley-momentum locking**: the forward-propagating mode ($v_g = \partial\omega_{\mathrm{edge}}/\partial q_x = +\eta v_D > 0$) lives exclusively at the $\tau_v = +1$ ($K$) valley, whereas the backward-propagating mode ($v_g = -\eta v_D < 0$) lives at the $\tau_v = -1$ ($K'$) valley. A sharp $120^\circ$ bend aligned with the primitive lattice vectors $\mathbf{a}_1$ and $\mathbf{a}_2$ maps the $K$-valley onto an equivalent $K$-corner of the hexagonal Brillouin zone without coupling $K \to K'$, strongly suppressing backscattering across the topological bandgap [3, 4, 7].

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path

from IPython.display import Image, Markdown, display

from _helpers import (
    DATA_DIR,
    FIG_DIR,
    HERE,
    audit_honeycomb_resolution,
    build_quantitative_summary_table,
    run_all_simulations,
)

# Run or load cached simulation data
RESULTS_JSON = DATA_DIR / "valley_hall_results.json"
if not RESULTS_JSON.exists() or "--regenerate" in sys.argv:
    results = run_all_simulations(quick=False)
else:
    results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))

b_met = results["bulk_and_berry"]
r_met = results["ribbon"]
w_met = results["waveguide_bend"]
c_met = results["convergence"]

# %% [markdown]
# ---
#
# ## 2. Simulation Setup (`00a` / `00b` Schematics and Setup Tables)
#
# Every 2D FIT simulation is preflighted with `wakis.estimate.estimate` and `wakis.estimate.preflight` and audited with `audit_honeycomb_resolution` to ensure $\ge 20$ cells per wavelength inside the $\varepsilon_{\mathrm{rod}} = 12$ cylinders and $\ge 4$ cells across the smallest rod diameter $2 r_B = 2 r_0(1 - |\delta|) = 0.3168\,a$.
#
# ### 2.1 Centred-rectangular supercell setup ($a \times \sqrt{3}a$)
#
# The unit-cell band and Berry-curvature simulations use the $a \times \sqrt{3}a$ centred-rectangular supercell with Bloch boundary conditions (`bc_low=["bloch", "bloch", "periodic"]`) and subpixel permittivity averaging (`fill_circular_inclusions`).
#
# ![Supercell geometry schematic](figures/00a_setup_geometry_supercell.png)
#
# ![Supercell mesh and excitation](figures/00b_setup_mesh_and_excitation_supercell.png)
#
# ### 2.2 Valley-Hall $120^\circ$ Z-bend waveguide setup
#
# The waveguide transmission simulations model a $9a \times 7\sqrt{3}a$ domain containing a sharp $120^\circ$ Z-bend interface between $\mathrm{VPC}_+$ ($\delta = +0.12$) and $\mathrm{VPC}_-$ ($\delta = -0.12$) domains (with the $-60^\circ$ diagonal segment shifted to half-integer column $x_1 = 3.5\,a$ so all three arms share the same small-rod $B\text{–}A$ interface), terminated by CPML absorbers on all four in-plane boundaries (`bc_low=["cpml", "cpml", "periodic"]`) and monitored by input and output `FluxMonitor` planes and a 2D midgap `DFTMonitor`.
#
# ![Waveguide geometry schematic](figures/00a_setup_geometry.png)
#
# ![Waveguide mesh and excitation](figures/00b_setup_mesh_and_excitation.png)

# %%
audit_info = audit_honeycomb_resolution(
    nx_per_a=int(c_met["grids_nx"][-1]),
    ny_per_supercell=int(c_met["grids_ny"][-1]),
)
print(
    f"Finest supercell grid ({c_met['grids_nx'][-1]}x{c_met['grids_ny'][-1]}): "
    f"{audit_info['cells_per_wavelength']:.1f} cells/lambda in eps=12, "
    f"{audit_info['cells_across_min_diameter']:.1f} cells across smallest rod diameter."
)
display(Markdown((HERE / "00_setup_table.md").read_text(encoding="utf-8")))

# %% [markdown]
# ---
#
# ## 3. Results
#
# ### 3.1 Supercell band unfolding, massless Dirac cone, and valley-Hall bandgap scaling
#
# Unfolding the 8 lowest supercell bands of the $a \times \sqrt{3}a$ cell onto the primitive hexagonal Brillouin zone via Eq. (5) reproduces the 4 primitive honeycomb bands along $\Gamma\text{–}M\text{–}K\text{–}\Gamma$. For the inversion-symmetric honeycomb crystal ($\delta = 0$), bands 1 and 2 touch linearly at $K$ at normalized frequency $\omega_D a / (2\pi c)$ with Dirac group velocity $v_D / c$. Breaking inversion symmetry with $\delta = 0.12$ opens a complete TM valley-Hall bandgap at $K$, in quantitative agreement between `PWE2D` and 2D FIT (`SolverFIT3D` + `harminv`).
#
# ![Supercell unfolding and valley-Hall gap](figures/01_supercell_unfolding_and_dirac_gap.png)

# %%
print(
    f"Dirac frequency at K (delta=0): omega_D a/(2*pi*c) = {b_met['dirac_freq_norm']:.4f}, "
    f"degeneracy split = {b_met['dirac_degeneracy_split']:.2e}, v_D/c = {b_met['dirac_velocity_over_c']:.4f}"
)
print(
    f"Supercell unfolding max error vs primitive PWE: {b_met['unfolding_max_error']:.2e}"
)
print(
    f"Valley-Hall gap at K (delta=0.12): PWE [{b_met['gap_bottom_pwe']:.4f}, {b_met['gap_top_pwe']:.4f}] "
    f"(width = {b_met['gap_width_pwe']:.4f}, midgap ratio = {100.0 * b_met['gap_midgap_ratio_pwe']:.2f}%)"
)
print(
    f"2D FIT + harminv at K (delta=0.12): band 1 = {b_met['fit_band1_K']:.4f} "
    f"(rel. err {100.0 * b_met['fit_band1_rel_err']:.2f}%), "
    f"band 2 = {b_met['fit_band2_K']:.4f} (rel. err {100.0 * b_met['fit_band2_rel_err']:.2f}%)"
)

# %% [markdown]
# ### 3.2 Berry curvature $\Omega_1(\mathbf{k})$ and valley Chern sign reversal (PWE, 2D FIT, and Dirac theory)
#
# Evaluating the gauge-invariant Fukui–Hatsugai–Suzuki plaquette phase [Eq. (12)] across the hexagonal Brillouin zone shows that the lower-band Berry curvature $\Omega_1(\mathbf{k})$ is sharply concentrated at the $K$ and $K'$ corners with opposite signs: $\Omega_1(\mathbf{K}) < 0$ and $\Omega_1(\mathbf{K}') = -\Omega_1(\mathbf{K}) > 0$ for $\delta = +0.12$. Because the Hanning-windowed Fourier projection in `fit_berry_curvature_patch` must resolve the lower valley band ($\omega_1(K) a/(2\pi c) = 0.2715$) from the upper band ($\omega_2(K) a/(2\pi c) = 0.3059$, separation $\Delta f = 0.0344\,c/a$), a short ring-down of $N_{\mathrm{steps}} = 1600$ ($\sim 33\,a/c$) leaves upper-band admixtures (`C_K = -0.2382`), whereas increasing the ring-down duration to $N_{\mathrm{steps}} = 6000$ ($\sim 120\,a/c$, the default) converges to $C_K^{\mathrm{FIT}} = -0.1175$ ($N_{\mathrm{steps}} = 10000 \to -0.1174$), matching the grid-independent PWE value $C_K^{\mathrm{PWE}} = -0.1188$ to $1.1\%$. For the inversion-symmetric $\delta = 0$ control, offsetting the $\mathbf{k}$-patch away from the degenerate Dirac point and off the $k_y = 0$ mirror line yields $C_K^{\mathrm{FIT}}(\delta=0) = -3.18\times 10^{-6}$ without relying on mirror cancellation.
#
# ![Berry curvature and valley Chern sign flip](figures/02_berry_curvature_and_valley_chern.png)

# %%
print(
    f"Patch valley Chern number at K (delta=+0.12): "
    f"Dirac = {b_met['c_patch_dirac_K_pos']:+.4f}, "
    f"PWE = {b_met['c_patch_pwe_K_pos']:+.4f}, "
    f"2D FIT (N=6000) = {b_met['c_patch_fit_K_pos']:+.4f} "
    f"(rel. err vs PWE = {100.0 * b_met['c_patch_fit_rel_err_vs_pwe']:.2f}%)"
)
print(
    "2D FIT patch Chern ring-down convergence (delta=+0.12): "
    + ", ".join(
        f"N={ns} -> {cp:+.4f}"
        for ns, cp in zip(b_met["berry_nsteps_sweep"], b_met["c_patch_fit_vs_nsteps"])
    )
)
print(
    f"Patch valley Chern number at K (delta=-0.12): "
    f"PWE = {b_met['c_patch_pwe_K_neg']:+.4f}, "
    f"2D FIT = {b_met['c_patch_fit_K_neg']:+.4f} "
    f"(delta=0 offset-patch control: {b_met['c_patch_fit_K_zero']:+.2e})"
)
print(
    f"Patch valley Chern number at K' (delta=+0.12): "
    f"PWE = {b_met['c_patch_pwe_Kp_pos']:+.4f}"
)

# %% [markdown]
# ### 3.3 Ribbon dispersion: valley-Hall domain wall vs tuned trivial defect guide
#
# Joining a $\mathrm{VPC}_+$ ($\delta = +0.12$) half-space and a $\mathrm{VPC}_-$ ($\delta = -0.12$) half-space in a periodic ribbon supercell ($a \times 6\sqrt{3}a$) creates **two inequivalent domain walls** with valley Chern jump $|\Delta C_v| = 1$: the interior small-rod $B\text{–}A$ interface at $y = L_{\mathrm{rib}}/2$ (ribbon mode 2, crossing $K_x = \frac{2\pi}{3a}$ at $\omega_{\mathrm{int}}(K_x) a/(2\pi c) = 0.2929$ with $92.2\%$ wall confinement) and the periodic wrap-around large-rod $A\text{–}B$ interface at $y = 0$ (ribbon mode 1, crossing $K_x$ at $\omega_{\mathrm{bnd}}(K_x) a/(2\pi c) = 0.2820$; two-wall midpoint $0.2874$). Because the finite ribbon couples the two interfaces and breaks $C_{3v}$ valley symmetry along the wall normal, modes 1 and 2 anticross near $k_x a/(2\pi) \approx 0.375$: mode 1 peaks at $0.2857\,c/a$ (producing two crossings per valley below the peak) and mode 2 dips to $0.2867\,c/a$, opening a narrow periodic-ribbon mini-gap $[0.2857, 0.2867]\,c/a$ and a flat folded minimum of mode 2 at $0.2915\,c/a$ around $k_x a/(2\pi) = 0.5$.
#
# For comparison with a conventional photonic-crystal waveguide, we tune the defect rod radius of a trivial line-defect guide (in a uniform $\delta = +0.12$ cladding with the $B_1$ and $A_1$ rods of row $j_{\mathrm{wall}}$ reduced to $r_{\mathrm{def}} = 0.65\,r_0 = 0.117\,a$, centering the donor defect channel on the $A_1\text{–}B_2$ chain at $y_w + 0.25\,L_y = y_w + 0.433\,a$) so its donor band traverses the bulk valley bandgap with $\omega_{\mathrm{triv}}(K_x) a/(2\pi c) = 0.2890$ and $87.1\%$ core confinement. Programmatically inspecting the computed ribbon bands partitions the bulk valley bandgap $[0.2715, 0.3059]\,c/a$ into: (i) the **lower window** $[0.2715, 0.2857]\,c/a$ below the mode-1 peak, (ii) the **periodic-ribbon mini-gap** $[0.2857, 0.2867]\,c/a$, and (iii) the **upper single-mode window** $[0.2915, 0.3033]\,c/a$ above the mode-2 folded minimum, where both the interior small-rod $B\text{–}A$ domain wall (used in the 2D FIT waveguide below) and the tuned trivial defect guide have a single monotonic branch per valley. (An earlier $[0.290, 0.302]\,c/a$ window approximated this upper interior-wall branch by hand, whereas centering a $[0.281, 0.289]\,c/a$ window on the two-wall midpoint $0.2874\,c/a$ straddled the $[0.2857, 0.2867]\,c/a$ ribbon mini-gap and the double crossings of mode 1.)
#
# ![Domain-wall ribbon dispersion and edge mode profile](figures/03_ribbon_domain_wall_dispersion.png)

# %%
print(
    f"Ribbon domain-wall modes at projected K (kx = 2*pi/(3a)): "
    f"interior wall = {r_met['interior_wall_freq_at_K_norm']:.4f}, "
    f"boundary wall = {r_met['boundary_wall_freq_at_K_norm']:.4f}, "
    f"midpoint = {r_met['edge_freq_at_K_norm']:.4f} "
    f"(wall confinement = {100.0 * r_met['edge_confinement_fraction']:.1f}%)"
)
print(
    f"Derived ribbon windows: bulk gap = [{r_met['bulk_gap_window'][0]:.4f}, {r_met['bulk_gap_window'][1]:.4f}], "
    f"lower = [{r_met['lower_window'][0]:.4f}, {r_met['lower_window'][1]:.4f}], "
    f"mini-gap = [{r_met['ribbon_minigap_window'][0]:.4f}, {r_met['ribbon_minigap_window'][1]:.4f}], "
    f"upper single-mode = [{r_met['upper_single_mode_window'][0]:.4f}, {r_met['upper_single_mode_window'][1]:.4f}]"
)
print(
    f"Tuned trivial guide (r_def = {r_met['trivial_defect_radius_over_a']:.3f} a, "
    f"channel offset +{r_met['trivial_core_offset_over_a']:.3f} a) at projected K: "
    f"omega_triv a/(2*pi*c) = {r_met['trivial_freq_at_K_norm']:.4f} "
    f"(core confinement = {100.0 * r_met['trivial_confinement_fraction']:.1f}%)"
)

# %% [markdown]
# ### 3.4 Straight and $120^\circ$ Z-bend transmission comparison and grid convergence
#
# We simulate four waveguides sequentially in 2D FIT—straight and $120^\circ$ Z-bend configurations for both the valley-Hall small-rod $B\text{–}A$ domain wall and the tuned trivial defect guide ($r_{\mathrm{def}} = 0.117\,a$). To prevent unguided dipole near-field radiation and end-facet Bloch reflections from biasing the normalization, each waveguide is terminated at $x < 1.1\,a$ and $x > 7.5\,a$ by a matched $(E, H)$ cubic sponge layer ahead of the CPML, and the output flux $\Phi_{\mathrm{out}}^{\mathrm{bend}}(\omega)$ at $x_{\mathrm{out}} = 7.2\,a$ is divided by the transmitted flux $\Phi_{\mathrm{out}}^{\mathrm{straight}}(\omega)$ of the matched straight guide (a bend-to-straight ratio, not an absolute transmission; $T_{\mathrm{straight}}\equiv 1$ by construction) measured at the identical plane $x_{\mathrm{out}} = 7.2\,a$ of the matched straight reference waveguide with the same source ($T_{\mathrm{straight}}(\omega) \equiv 1.000$).
#
# Across the derived **upper single-mode window** $[0.2915, 0.3033]\,c/a$, the valley-Hall $120^\circ$ Z-bend achieves $\langle T_{\mathrm{bend}}\rangle_{\mathrm{up}} = 1.010$ (peak $1.015$, the ratio exceeds 1 by up to $1.5\%$, which is within the grid-to-grid shift $|\Delta T| = 0.0175$; $\langle T_{\mathrm{bend}}\rangle_{\mathrm{up}} = 0.993$ with peak $0.9998 \le 1$ on the $16 \times 28$ grid) compared with $\langle T_{\mathrm{bend}}\rangle_{\mathrm{up}} = 0.072$ for the tuned trivial bend ($14.1\times$, or $+11.48\;\mathrm{dB}$ bend-to-straight ratio). In the **lower window** $[0.2715, 0.2857]\,c/a$, the valley-Hall bend transmits $\langle T_{\mathrm{bend}}\rangle_{\mathrm{low}} = 0.891$ versus $0.235$ for the trivial bend ($+5.79\;\mathrm{dB}$), and across the **entire bulk valley bandgap** $[0.2715, 0.3059]\,c/a$ it averages $\langle T_{\mathrm{bend}}\rangle_{\mathrm{gap}} = 0.956$ versus $0.148$ (peak $0.274$) for the trivial bend ($6.45\times$, or $+8.10\;\mathrm{dB}$).
#
# Figure 5 compares the steady-state $|E_z(x, y)|$ DFT maps at the interior-wall $K$-crossing frequency $\omega a/(2\pi c) = 0.2929$: in the valley-Hall Z-bend [Fig. 5(a)], the guided mode rounds both $120^\circ$ corners along the input ($y = 6.93\,a$), diagonal, and output ($y = 5.20\,a$) arms, while showing finite transverse evanescent penetration into the adjacent cladding rows ($y \approx 4.1\,a\text{–}8.5\,a$, consistent with the ribbon profile in Fig. 3b), local source radiation near $x = 1.5\,a$, and partial cladding scattering near the corners and output sponge ($x \approx 7.5\text{–}8.0\,a$). In the tuned trivial guide [Fig. 5(b)], aligning the source, monitors, and dashed guide path with the $A_1\text{–}B_2$ defect channel axis at $y_w(x) + 0.433\,a$ ($y = 7.36\,a$ in the input arm, $5.63\,a$ in the output arm) shows the guided mode well centered on the input defect channel before reflecting strongly at the first $120^\circ$ corner ($x \approx 3.8\,a$), leaving only weak field amplitude in the output arm.
#
# ![Bend transmission comparison and grid convergence](figures/04_bend_transmission_and_convergence.png)
#
# ![Steady-state bend field maps](figures/05_bend_field_maps.png)
#
# ![Animated wavefront rounding the 120-deg Z-bend](figures/06_valley_hall_bend_wavefront.gif)

# %%
print(
    f"Straight reference waveguide transmission: "
    f"Valley-Hall = {w_met['mean_transmission_valley_straight']:.3f}, "
    f"Tuned Trivial = {w_met['mean_transmission_trivial_straight']:.3f}"
)
print(
    f"120-deg Z-bend transmission in upper single-mode window [{w_met['upper_single_mode_window'][0]:.4f}, {w_met['upper_single_mode_window'][1]:.4f}]: "
    f"Valley-Hall = {w_met['mean_transmission_valley_bend']:.3f} (peak {w_met['peak_transmission_valley_bend']:.3f}) "
    f"vs Tuned Trivial = {w_met['mean_transmission_trivial_bend']:.3f} "
    f"-> +{w_met['bend_transmission_advantage_db']:.2f} dB."
)
print(
    f"120-deg Z-bend transmission across full bulk gap [{w_met['bulk_gap_window'][0]:.4f}, {w_met['bulk_gap_window'][1]:.4f}]: "
    f"Valley-Hall = {w_met['mean_transmission_valley_bend_full_gap']:.3f} "
    f"vs Tuned Trivial = {w_met['mean_transmission_trivial_bend_full_gap']:.3f} (peak {w_met['peak_transmission_trivial_bend']:.3f}) "
    f"-> +{w_met['bend_transmission_advantage_full_gap_db']:.2f} dB "
    f"(lower window [{w_met['lower_window'][0]:.4f}, {w_met['lower_window'][1]:.4f}]: "
    f"{w_met['mean_transmission_valley_bend_lower']:.3f} vs {w_met['mean_transmission_trivial_bend_lower']:.3f}, "
    f"+{w_met['bend_transmission_advantage_lower_db']:.2f} dB)."
)
print(
    f"3-resolution 2D FIT convergence (Nx = {c_met['grids_nx']}): "
    f"omega_1(K) = {[round(x, 5) for x in c_met['f1_K_values']]}, "
    f"Richardson = {c_met['f1_K_richardson']:.5f} vs PWE = {c_met['f1_K_pwe_ref']:.5f} "
    f"(rel. err = {100.0 * c_met['richardson_vs_pwe_rel_err']:.2f}%), "
    f"observed order p = {c_met['observed_order_p']:.2f}."
)

# %% [markdown]
# ---
#
# ## 4. Quantitative Summary
#
# Every numerical value below is loaded directly from `data/valley_hall_results.json` and cross-checked against analytical theory or converged `PWE2D` reference values.

# %%
summary_md = build_quantitative_summary_table(results)
display(Markdown(summary_md))

# %% [markdown]
# ---
#
# ## 5. Limits & Failure Modes
#
# 1. **Large inversion asymmetry ($|\delta| \gtrsim 0.25$) and Berry-curvature delocalization:**
#    The quantized half-integer valley Chern number $C_{\tau_v} = -\frac{1}{2}\tau_v\mathrm{sgn}(\delta)$ in Eq. (11) relies on the massive Dirac approximation $|m| \ll v_D |\mathbf{K} - \mathbf{M}|$, under which the Berry curvature $\Omega_1(\mathbf{k})$ is tightly localized around the $K$ and $K'$ corners with radius $\Delta q \sim |m|/v_D$. *Counter-case:* When $|\delta| \gtrsim 0.25$, the Berry curvature spreads across the hexagonal Brillouin zone boundary and overlaps between $K$ and $K'$; the single-valley integral drops well below $1/2$ and intervalley coupling across sharp corners increases.
# 2. **Out-of-plane radiative leakage above the cladding light line in 3D slabs:**
#    Our 2D TM model assumes infinitely long dielectric rods ($\partial_z = 0, k_z = 0$), so every mode at any in-plane wavevector $\mathbf{k}_\parallel$ is strictly bound. *Counter-case:* In a finite-thickness 3D silicon-on-insulator photonic-crystal slab [7, 11], modes with $|\mathbf{k}_\parallel| < \omega n_{\mathrm{clad}} / c$ couple to radiating continuum modes in the cladding; if the domain wall folds the $K/K'$ valleys inside the light cone (or at sharp bends that scatter into radiative $\mathbf{k}_\parallel$ components), the edge state acquires a finite radiation $Q_{\mathrm{rad}}$.
# 3. **Short-range disorder and non-$120^\circ$ bend angles (intervalley backscattering):**
#    Unlike Chern photonic topological insulators with broken time-reversal symmetry [6], valley-Hall topological protection is **symmetry-protected** by conservation of the valley pseudospin $\tau_v = \pm 1$. *Counter-case:* A $90^\circ$ bend (which is incompatible with the $60^\circ / 120^\circ$ hexagonal lattice axes) or a point defect (such as a missing rod right on the domain wall) contains spatial Fourier components at $|\Delta\mathbf{k}| = |\mathbf{K} - \mathbf{K}'| = \frac{4\pi}{3a}$, coupling the forward $K$-valley mode into the backward $K'$-valley mode and producing Fabry–Pérot standing waves.
#
# ---
#
# ## 6. References
#
# 1. K. M. Ho, C. T. Chan, and C. M. Soukoulis, "Existence of a photonic gap in periodic dielectric structures," *Phys. Rev. Lett.* **65**, 3152–3155 (1990), [doi:10.1103/PhysRevLett.65.3152](<https://doi.org/10.1103/PhysRevLett.65.3152>).
# 2. M. Plihal and A. A. Maradudin, "Photonic band structure of two-dimensional systems: The triangular lattice," *Phys. Rev. B* **44**, 8565–8571 (1991), [doi:10.1103/PhysRevB.44.8565](<https://doi.org/10.1103/PhysRevB.44.8565>).
# 3. T. Ma and G. Shvets, "All-Si valley-Hall photonic topological insulator," *New J. Phys.* **18**, 025012 (2016), [doi:10.1088/1367-2630/18/2/025012](<https://doi.org/10.1088/1367-2630/18/2/025012>).
# 4. F. Gao, H. Xue, Z. Yang, K. Lai, Y. Yu, X. Lin, Y. Chong, G. Shvets, and B. Zhang, "Topologically protected refraction of robust kink states in valley photonic crystals," *Nat. Phys.* **14**, 140–144 (2018), [doi:10.1038/nphys4304](<https://doi.org/10.1038/nphys4304>).
# 5. T. Fukui, Y. Hatsugai, and H. Suzuki, "Chern Numbers in Discretized Brillouin Zone: Efficient Method of Computing (Spin) Hall Conductances," *J. Phys. Soc. Jpn.* **74**, 1674–1677 (2005), [doi:10.1143/JPSJ.74.1674](<https://doi.org/10.1143/JPSJ.74.1674>).
# 6. F. D. M. Haldane and S. Raghu, "Possible Realization of Directional Optical Waveguides in Photonic Crystals with Broken Time-Reversal Symmetry," *Phys. Rev. Lett.* **100**, 013904 (2008), [doi:10.1103/PhysRevLett.100.013904](<https://doi.org/10.1103/PhysRevLett.100.013904>).
# 7. X.-T. He, E.-T. Liang, J.-J. Yuan, H.-Y. Qiu, X.-D. Chen, F.-L. Zhao, and J.-W. Dong, "A silicon-on-insulator slab for topological valley transport," *Nat. Commun.* **10**, 872 (2019), [doi:10.1038/s41467-019-08881-z](<https://doi.org/10.1038/s41467-019-08881-z>).
# 8. V. Popescu and A. Zunger, "Effective Band Structure of Random Alloys," *Phys. Rev. Lett.* **104**, 236403 (2010), [doi:10.1103/PhysRevLett.104.236403](<https://doi.org/10.1103/PhysRevLett.104.236403>).
# 9. W. Ku, T. Berlijn, and C.-C. Lee, "Unfolding First-Principles Band Structures," *Phys. Rev. Lett.* **104**, 216401 (2010), [doi:10.1103/PhysRevLett.104.216401](<https://doi.org/10.1103/PhysRevLett.104.216401>).
# 10. J. D. Joannopoulos, S. G. Johnson, J. N. Winn, and R. D. Meade, *Photonic Crystals: Molding the Flow of Light*, 2nd ed. (Princeton University Press, Princeton, NJ, 2008).
# 11. M. I. Shalaev, W. Walasik, A. Tsukernik, Y. Xu, and N. M. Litchinitser, "Robust topologically protected transport in photonic crystals at telecommunication wavelengths," *Nat. Nanotechnol.* **14**, 31–34 (2019), [doi:10.1038/s41565-018-0297-6](<https://doi.org/10.1038/s41565-018-0297-6>).
