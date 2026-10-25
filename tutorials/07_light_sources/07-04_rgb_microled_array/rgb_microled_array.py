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
# # Tutorial 7.4 · RGB microLED array: incoherent averaging, collimation and crosstalk
#
# **Reproducing the baseline 2D FDTD physics of:**
# L. Chen, Z. Li, P. Wang, S. Wu, W. Li, J. Wang, Y. Cao, M. Mortazavi, L. Peng, and P. Wu,
# *"Dual-Task Optimization Method for Inverse Design of RGB Micro-LED Light Collimator"*,
# **Nanomaterials** 15(3), 190 (2025), [DOI: 10.3390/nano15030190](https://doi.org/10.3390/nano15030190) (`PMC11820347`).
#
# ---
#
# ## Abstract & Learning Goals
#
# Near-eye augmented-reality (AR) and mixed-reality (MR) microdisplays pack full-color
# red/green/blue (RGB) micro-LED pixels into pitches of only a few micrometers ($P_{\text{unit}} = 4.5\;\mu\text{m}$,
# subpixel pitch $P_{\text{sub}} = 1.5\;\mu\text{m}$). At these sub-10-$\mu\text{m}$ dimensions, three coupled
# electromagnetic challenges govern device performance:
#
# 1. **Spatially incoherent spontaneous emission**: Electron–hole recombination inside an active
#    multiple-quantum-well (MQW) slab of width $W_{\text{ch}} = 1.05\;\mu\text{m}$ and thickness
#    $t_{\text{MQW}} = 200\;\text{nm}$ excites an ensemble of mutually uncorrelated point emitters.
#    Because relative phases fluctuate on femtosecond carrier-scattering timescales, **optical powers
#    (or field intensities $|\mathbf{E}|^2$), never complex field phasors $\mathbf{E}$, must be averaged**
#    over emitter positions.
# 2. **Large angular divergence vs. étendue limits**: Uncollimated GaN micro-LEDs radiate with a broad,
#    quasi-Lambertian angular profile ($\bar{P}_r(\theta) \propto \cos\theta$), coupling only
#    $\eta_{\pm 20^\circ} \approx 30\%\text{--}36\%$ of their extracted power into the $\pm 20^\circ$
#    acceptance cone of pancake lenses or diffractive AR waveguides.
# 3. **Inter-pixel optical crosstalk ($C_{ij}$) and reciprocal color routing**: Wide-angle emission from
#    subpixel $j \in \{\text{B}, \text{G}, \text{R}\}$ spills laterally into neighboring subpixel exit windows
#    $i \neq j$ and couples into adjacent MQW regions across the $0.45\;\mu\text{m}$ polyimide isolation trenches.
#    By Lorentz reciprocity, collimating spatially incoherent RGB emission into $\pm 20^\circ$ is the dual task
#    of routing incident plane waves within $|\theta_{\text{inc}}| \le 15^\circ$ into their respective B, G, R
#    MQW monitors $M_{\text{B}}, M_{\text{G}}, M_{\text{R}}$.
#
# In this tutorial you will:
# - Build the exact 2D RGB micro-LED pixel cross-section and dispersive material model from **Table 1** and
#   **Section 2.2** of Chen et al. (2025) in Wakis FIT (`GridFIT3D` / `SolverFIT3D` with $N_z = 1$,
#   Kottke subpixel smoothing, and conductive/dispersive Silicon substrate).
# - Compare **Method A** ($N = 20$ independent single-dipole FDTD runs summed in intensity) against
#   **Method B** ($K$ Monte Carlo realizations of $N = 20$ simultaneous random-phase dipoles), validating
#   the exact analytical variance law $\text{RSE}(K, N) = \sqrt{(1 - 1/N)/K}$ and cost crossover.
# - Demonstrate why coherent field summation ($|\sum_m \mathbf{E}_m|^2$) fails catastrophically for LEDs.
# - Reproduce the paper's uncollimated angular power ratio density $\bar{P}_r(\theta)$ (**Eq. 5**,
#   **Fig. 4c,e,g**), $\pm 20^\circ$ coupling efficiency $\eta_{\pm 20^\circ}$, total extraction efficiency
#   $\text{LEE}_{\text{tot}}$, forward inter-pixel crosstalk matrix $C_{ij}^{\text{emit}}$, and reciprocal
#   dual-task routing matrix $C_{ij}^{\text{dual}}$ (**Eq. 4**, **Fig. 3d**).
#
# ---
#
# ## Physics & Mathematical Framework
#
# ### 1. Escape Cone, 2D vs. 3D Lambertian Baselines, and Étendue Bounds
#
# Inside the high-index GaN/MQW mesa ($n_{\text{MQW}} \in [2.37, 2.51]$), rays impinging on the top planar
# GaN/air interface ($y = 0$) undergo total internal reflection (TIR) above the critical angle
# $$\theta_c(\lambda) = \arcsin\!\left(\frac{n_{\text{air}}}{n_{\text{MQW}}(\lambda)}\right) \approx 23.5^\circ\text{--}25.0^\circ.$$
# In two dimensions (line emitters in the $xy$-plane, isotropic over $2\pi\;\text{rad}$), a single planar
# escape cone subtends a fraction $\theta_c / \pi \approx 13.0\%\text{--}13.9\%$ of the full circle
# (compared with $\frac{1}{2}(1 - \cos\theta_c) \approx 4.1\%\text{--}4.7\%$ in 3D). Including partial reflection
# from the bottom GaN/Silicon interface ($R_{\text{Si}} = |(n_{\text{GaN}} - \tilde{n}_{\text{Si}})/(n_{\text{GaN}} + \tilde{n}_{\text{Si}})|^2 \approx 6.0\%\text{--}8.8\%$)
# and lateral escape through the $1.10\;\mu\text{m}$-tall mesa sidewalls into the lower-index polyimide trenches
# ($n_{\text{PI}} = 1.50$), the total upward extraction efficiency $\text{LEE}_{\text{tot}} = P_{\text{top}} / P_{\text{MQW}}$
# reaches $\sim 24\%\text{--}31\%$.
#
# Once refracted into air, an ideal 2D surface emitter follows the cosine (Lambertian) angular distribution
# $P_{\text{Lamb}}(\theta) = P_0 \cos\theta$ on $\theta \in [-90^\circ, +90^\circ]$. Chen et al. (2025) define
# the **normalized angular power ratio density** (**Equation 5**) over $N_\theta$ discrete far-field angles
# with resolution $\Delta\theta = 180^\circ / N_\theta$:
# $$\bar{P}_r(\theta) = \frac{P(\theta)}{\sum_{\theta=-90^\circ}^{90^\circ} P(\theta)\,\Delta\theta} \quad [\text{deg}^{-1}], \qquad \sum_{\theta=-90^\circ}^{90^\circ} \bar{P}_r(\theta)\,\Delta\theta = 1.$$
# For an ideal 2D Lambertian source, the fraction of far-field power inside a divergence half-angle $\theta_0$ is:
# $$\eta_{\text{Lamb,2D}}(\pm\theta_0) = \frac{\int_{-\theta_0}^{\theta_0} \cos\theta\,d\theta}{\int_{-\pi/2}^{\pi/2} \cos\theta\,d\theta} = \sin\theta_0 \quad \Longrightarrow \quad \eta_{\text{Lamb,2D}}(\pm 20^\circ) = \sin 20^\circ = 34.20\%,$$
# whereas in 3D axisymmetric geometry $\eta_{\text{Lamb,3D}}(\pm\theta_0) = \sin^2\theta_0 = 11.70\%$ at $\theta_0 = 20^\circ$.
#
# **Why single-subpixel microlenses fail (2D Étendue Theorem):**
# A spatially incoherent 2D Lambertian emitter of width $W_{\text{ch}} = 1.05\;\mu\text{m}$ radiating into air
# ($n = 1$) carries two-dimensional phase-space étendue $\mathcal{E}_{\text{src}} = 2 W_{\text{ch}} \sin 90^\circ = 2.10\;\mu\text{m}\cdot\text{rad}$.
# Any passive optical collimator of exit aperture $W_{\text{ap}}$ cannot compress phase-space density, imposing the
# thermodynamic upper bound:
# $$\eta_{\max}(\pm\theta_0) \le \min\!\left(1,\, \frac{W_{\text{ap}}}{W_{\text{ch}}}\sin\theta_0\right).$$
# A conventional microlens restricted to a single subpixel pitch ($W_{\text{ap}} = P_{\text{sub}} = 1.50\;\mu\text{m}$)
# is fundamentally capped at $\eta_{\max,\text{sub}}(\pm 20^\circ) = (1.50/1.05)\sin 20^\circ = 48.86\%$ (and achieves
# only $\sim 31\%$ in practice due to off-axis curvature and Fresnel losses). Only by multiplexing all three
# wavelengths across the **entire $P_{\text{unit}} = 4.50\;\mu\text{m}$ RGB unit cell** ($W_{\text{ap}}/W_{\text{ch}} = 4.29 \implies \eta_{\max,\text{unit}}(\pm 20^\circ) = 100\%$)
# can an inverse-designed dichromatic/trichromatic collimator boost $\eta_{\pm 20^\circ}$ to $\sim 56\%\text{--}66\%$.
#
# ---
#
# ### 2. Spatially Incoherent Averaging: Method A vs. Method B
#
# Consider $N$ mutually incoherent dipole emitters at positions $\mathbf{r}_m$ ($m = 1, \dots, N$) within the MQW,
# each generating a deterministic single-dipole complex frequency-domain field $\tilde{\mathbf{E}}_m(\mathbf{r}, \omega_0)$
# and far-field angular amplitude $\tilde{E}_m(\theta)$.
#
# - **Method A (Deterministic Power Superposition — exact in $N$ runs):**
#   Run $N$ separate FDTD simulations, exciting one dipole at a time, and sum their intensities/powers:
#   $$I_A(\mathbf{r}) = \sum_{m=1}^{N} \left|\tilde{\mathbf{E}}_m(\mathbf{r}, \omega_0)\right|^2, \qquad P_A(\theta) = \sum_{m=1}^{N} \left|\tilde{E}_m(\theta)\right|^2.$$
#   Cost: exactly $N$ FDTD simulations per color channel ($3 \times 20 = 60$ runs for an RGB pixel).
#
# - **Method B (Simultaneous Random-Phase Ensemble — Monte Carlo in $K$ runs):**
#   In each realization $k \in \{1, \dots, K\}$, excite all $N$ dipoles simultaneously with independent
#   uniform random carrier phases $\phi_m^{(k)} \stackrel{\text{i.i.d.}}{\sim} \mathcal{U}(0, 2\pi)$:
#   $$\tilde{E}^{(k)}(\theta) = \sum_{m=1}^{N} \tilde{E}_m(\theta)\,e^{i\phi_m^{(k)}}, \qquad \hat{P}_B^{(K)}(\theta) = \frac{1}{K}\sum_{k=1}^{K} \left|\tilde{E}^{(k)}(\theta)\right|^2.$$
#   Expanding the modulus squared separates the diagonal incoherent target from the off-diagonal cross-interference terms:
#   $$\hat{P}_B^{(K)}(\theta) = P_A(\theta) + \frac{1}{K}\sum_{k=1}^{K} \sum_{m \neq n} \tilde{E}_m(\theta)\tilde{E}_n^*(\theta)\,e^{i(\phi_m^{(k)} - \phi_n^{(k)})}.$$
#   Since $\mathbb{E}[e^{i(\phi_m - \phi_n)}] = 0$ for $m \neq n$ and $\mathbb{E}[e^{i(\phi_m - \phi_n)} e^{-i(\phi_p - \phi_q)}] = \delta_{mp}\delta_{nq}$:
#   1. **Zero Bias:** $\mathbb{E}[\hat{P}_B^{(K)}(\theta)] = P_A(\theta)$ for every $K \ge 1$.
#   2. **Exact Analytical Variance:** For $N$ equal-power dipoles, the variance of a single realization is
#      $\text{Var}[\hat{P}_B^{(1)}(\theta)] = \sum_{m \neq n} |\tilde{E}_m|^2 |\tilde{E}_n|^2 = P_A(\theta)^2 (1 - 1/N_{\text{eff}})$,
#      yielding the exact relative standard error (RSE) after $K$ realizations:
#      $$\text{RSE}(K, N) = \frac{\sigma[\hat{P}_B^{(K)}(\theta)]}{P_A(\theta)} = \sqrt{\frac{1 - 1/N}{K}}.$$
#   3. **Ratio Estimator Subtlety (Jensen's Inequality):** For normalized metrics such as
#      $\eta_{\pm 20^\circ} = P_{\text{cone}} / P_{\text{tot}}$, one must average the numerator and denominator
#      across the $K$ realizations *before* dividing ($\hat{\eta}_B^{(K)} = \overline{P_{\text{cone}}}^{(K)} / \overline{P_{\text{tot}}}^{(K)}$);
#      averaging single-realization ratios $\frac{1}{K}\sum_k (P_{\text{cone}}^{(k)} / P_{\text{tot}}^{(k)})$ introduces
#      an $O(1/K)$ ratio bias!
#
# ---
#
# ### 3. Inter-Pixel Optical Crosstalk & Dual-Task Reciprocity ($C_{ij}$)
#
# We quantify crosstalk using two complementary $3 \times 3$ matrices over $(i, j) \in \{\text{B}, \text{G}, \text{R}\}^2$:
#
# 1. **Forward Emission Crosstalk Matrix $C_{ij}^{\text{emit}}$**: When channel $j$ emits incoherently (20 dipoles),
#    $C_{ij}^{\text{win}} = P_{\text{win},i}^{(j)} / \sum_{k} P_{\text{win},k}^{(j)}$ measures the fraction of upward
#    Poynting flux crossing subpixel exit window $W_i$ ($x \in [x_i - 0.75\;\mu\text{m}, x_i + 0.75\;\mu\text{m}]$ at $y = 0.25\;\mu\text{m}$),
#    while $C_{ij}^{\text{mqw}}$ measures lateral field-intensity coupling into neighboring MQW monitors $M_i$.
# 2. **Dual-Task Reciprocal Color-Routing Matrix $C_{ij}^{\text{dual}}$ (Equations 1–4 of Chen et al. 2025)**:
#    Under plane-wave illumination from above at wavelength $\lambda_j$ and incident angles
#    $\theta_{\text{inc}} \in \{-15^\circ, -7.5^\circ, 0^\circ, 7.5^\circ, 15^\circ\}$, planar monitors $M_i$ at
#    the MQW center ($y = -0.30\;\mu\text{m}$) record
#    $$c_{i,j}(\theta_{\text{inc}}) = \int_{M_i} \left|E_x^j(x, y_{\text{MQW}}; \theta_{\text{inc}})\right|^2 dx, \qquad \tilde{C}_{ij} = \frac{c_{i,j}}{\sum_{k \in \{\text{B,G,R}\}} c_{k,j}},$$
#    where diagonal entries $\text{OE}(i) = \tilde{C}_{ii}$ represent color-routing optical efficiency and
#    off-diagonal entries $\text{OX}(j, i) = \tilde{C}_{ij}$ ($i \neq j$) represent optical crosstalk.
#
# ### 4. Simulation Setup Schematic & Graded Mesh Architecture
#
# The setup figures and tables below are generated directly from the live Wakis `GridFIT3D`,
# `SolverFIT3D`, `Box` structures, `CurrentSource` arrays, and `Near2FarMonitor`/`FluxMonitor` objects via
# `wakis.schematic.save_setup_figures`:
#
# ![Forward Setup Geometry](figures/00a_setup_geometry.png)
# ![Forward Setup Mesh and Excitation](figures/00b_setup_mesh_and_excitation.png)
#
# The forward incoherent emission setup places a $5 \times 4 = 20$ horizontal $J_x$ dipole ensemble inside the active subpixel's $200\;\text{nm}$ MQW layer ($y \in [-0.40, -0.20]\;\mu\text{m}$, with 5 representative dipole positions shown) across the 3-pixel B/G/R micro-LED mesa array on an absorbing Silicon substrate, recording upward extraction and far-field radiation on a top `Near2FarMonitor` and exit windows $W_{\text{B}}, W_{\text{G}}, W_{\text{R}}$ at $y = +0.25\;\mu\text{m}$.
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 5.70 µm × 1.96 µm × 1 µm (z ∈ [-500 nm, 500 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 630 × 282 × 1 (177660 cells) |
# | Discretization | Δx & resolution | Δ_min = 4.88 nm, 25.6 cells/λ (in n_max = 4.21), 41.0 cells across feature (200 nm) |
# | Boundary | -x face | CPML (8 cells, 72.5 nm) |
# | Boundary | +x face | CPML (8 cells, 72.5 nm) |
# | Boundary | -y face | CPML (8 cells, 39.3 nm) |
# | Boundary | +y face | CPML (8 cells, 71.9 nm) |
# | Boundary | -z face | Periodic |
# | Boundary | +z face | Periodic |
# | Material | Background | Air superstrate (n = 1.00), ε = 1.00 (isotropic) |
# | Material | Region 1: Si substrate (4.22 + 0.048i) | ε_r = 17.76, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-2.85 µm, 2.85 µm] (d = 5.70 µm) |
# | Material | Region 2: Polyimide trench (n = 1.50) | ε_r = 2.25, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-2.85 µm, 2.85 µm] (d = 5.70 µm) |
# | Material | Region 3: B n-GaN (n = 2.49) | ε_r = 6.20, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-2.02 µm, -975 nm] (d = 1.05 µm) |
# | Material | Region 4: B MQW (n = 2.51, 457 nm) | ε_r = 6.30, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-2.02 µm, -975 nm] (d = 1.05 µm) |
# | Material | Region 5: B p-GaN (n = 2.49) | ε_r = 6.20, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-2.02 µm, -975 nm] (d = 1.05 µm) |
# | Material | Region 6: G n-GaN (n = 2.38) | ε_r = 5.66, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-525 nm, 525 nm] (d = 1.05 µm) |
# | Material | Region 7: G MQW (n = 2.41, 527 nm) | ε_r = 5.81, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-525 nm, 525 nm] (d = 1.05 µm) |
# | Material | Region 8: G p-GaN (n = 2.38) | ε_r = 5.66, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-525 nm, 525 nm] (d = 1.05 µm) |
# | Material | Region 9: R n-GaN (n = 2.35) | ε_r = 5.52, μ_r = 1.00, model = Isotropic dielectric, x ∈ [975 nm, 2.02 µm] (d = 1.05 µm) |
# | Material | Region 10: R MQW (n = 2.37, 611 nm) | ε_r = 5.62, μ_r = 1.00, model = Isotropic dielectric, x ∈ [975 nm, 2.02 µm] (d = 1.05 µm) |
# | Material | Region 11: R p-GaN (n = 2.35) | ε_r = 5.52, μ_r = 1.00, model = Isotropic dielectric, x ∈ [975 nm, 2.02 µm] (d = 1.05 µm) |
# | Source | 1 · CurrentSource | pos: G MQW: x ∈ [-0.42, +0.42] µm, y ∈ [-0.38, -0.22] µm (5×4 = 20 dipoles, 5 shown), pol: J_x (horizontal), pulse: f₀ = 568.9 THz, Δf = 199.1 THz |
# | Monitor | 2 · Near2FarMonitor | Near2FarMonitor + subpixel exit windows W_B, W_G, W_R (y = +0.25 µm; normalized by MQW dipole box), pos: y = +0.25 µm, x ∈ [-2.60, +2.60] µm, freqs: f = 568.9 THz |
# | Variation | Task 0 — Blue Method A (`method_a_B`) | notes=Blue mesa active (λ_B = 457 nm, x_c = -1.50 µm); Method A exact 20-run incoherent dipole sum |
# | Variation | Task 1 — Green Method A (`method_a_G`) | notes=Green mesa active (λ_G = 527 nm, x_c = 0.00 µm); Method A exact 20-run incoherent dipole sum |
# | Variation | Task 2 — Red Method A (`method_a_R`) | notes=Red mesa active (λ_R = 611 nm, x_c = +1.50 µm); Method A exact 20-run incoherent dipole sum |
# | Variation | Task 3 — Green Method B (`method_b_G`) | notes=Simultaneous 20-dipole random-phase realizations (K = 1, 2, 4, 8, 16) vs. exact Method A |
# | Variation | Tasks 5–9 — Grid & Si skin-depth convergence (`conv_cpw10..28`) | notes=cpw ∈ {10, 12, 16, 20, 28} (56.9–449.9 cells/δ_P in absorbing Si substrate) |
#
# ![Reciprocal Setup Geometry](figures/00a_setup_geometry_dual_routing.png)
# ![Reciprocal Setup Mesh and Excitation](figures/00b_setup_mesh_and_excitation_dual_routing.png)
#
# The reciprocal dual-task color-routing setup (Task 4) illuminates the 3-pixel array from above with an oblique plane-wave line source at $y = +0.35\;\mu\text{m}$ ($\theta_{\text{inc}} \in [-15^\circ, +15^\circ]$ across $\lambda_{\text{B}} = 457\;\text{nm}$, $\lambda_{\text{G}} = 527\;\text{nm}$, and $\lambda_{\text{R}} = 611\;\text{nm}$) while planar monitors $M_{\text{B}}, M_{\text{G}}, M_{\text{R}}$ at the MQW center ($y = -0.30\;\mu\text{m}$) record the routed intensity matrix $C_{ij}^{\text{dual}}$ (Eq. 4).
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 5.70 µm × 1.96 µm × 1 µm (z ∈ [-500 nm, 500 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 630 × 282 × 1 (177660 cells) |
# | Discretization | Δx & resolution | Δ_min = 4.88 nm, 25.6 cells/λ (in n_max = 4.21), 41.0 cells across feature (200 nm) |
# | Boundary | -x face | CPML (8 cells, 72.5 nm) |
# | Boundary | +x face | CPML (8 cells, 72.5 nm) |
# | Boundary | -y face | CPML (8 cells, 39.3 nm) |
# | Boundary | +y face | CPML (8 cells, 71.9 nm) |
# | Boundary | -z face | Periodic |
# | Boundary | +z face | Periodic |
# | Material | Background | Air superstrate (n = 1.00), ε = 1.00 (isotropic) |
# | Material | Region 1: Si substrate (4.22 + 0.048i) | ε_r = 17.76, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-2.85 µm, 2.85 µm] (d = 5.70 µm) |
# | Material | Region 2: Polyimide trench (n = 1.50) | ε_r = 2.25, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-2.85 µm, 2.85 µm] (d = 5.70 µm) |
# | Material | Region 3: B n-GaN (n = 2.49) | ε_r = 6.20, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-2.02 µm, -975 nm] (d = 1.05 µm) |
# | Material | Region 4: B MQW (n = 2.51, 457 nm) | ε_r = 6.30, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-2.02 µm, -975 nm] (d = 1.05 µm) |
# | Material | Region 5: B p-GaN (n = 2.49) | ε_r = 6.20, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-2.02 µm, -975 nm] (d = 1.05 µm) |
# | Material | Region 6: G n-GaN (n = 2.38) | ε_r = 5.66, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-525 nm, 525 nm] (d = 1.05 µm) |
# | Material | Region 7: G MQW (n = 2.41, 527 nm) | ε_r = 5.81, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-525 nm, 525 nm] (d = 1.05 µm) |
# | Material | Region 8: G p-GaN (n = 2.38) | ε_r = 5.66, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-525 nm, 525 nm] (d = 1.05 µm) |
# | Material | Region 9: R n-GaN (n = 2.35) | ε_r = 5.52, μ_r = 1.00, model = Isotropic dielectric, x ∈ [975 nm, 2.02 µm] (d = 1.05 µm) |
# | Material | Region 10: R MQW (n = 2.37, 611 nm) | ε_r = 5.62, μ_r = 1.00, model = Isotropic dielectric, x ∈ [975 nm, 2.02 µm] (d = 1.05 µm) |
# | Material | Region 11: R p-GaN (n = 2.35) | ε_r = 5.52, μ_r = 1.00, model = Isotropic dielectric, x ∈ [975 nm, 2.02 µm] (d = 1.05 µm) |
# | Source | 1 · CurrentSource | pos: y = +0.35 µm, x ∈ [-2.45, +2.45] µm (θ_inc ∈ [-15°, +15°]), pol: J_x (TE_x), pulse: f₀ = 568.9 THz, Δf = 199.1 THz |
# | Monitor | 2 · FluxMonitor | Planar MQW routing monitor M_B (y = -0.30 µm, x_c = -1.50 µm; Eq. 4), pos: y = -0.30 µm, x_c = -1.50 µm (W_ch = 1.05 µm), freqs: f = 568.9 THz |
# | Monitor | 3 · FluxMonitor | Planar MQW routing monitor M_G (y = -0.30 µm, x_c = +0.00 µm; Eq. 4), pos: y = -0.30 µm, x_c = +0.00 µm (W_ch = 1.05 µm), freqs: f = 568.9 THz |
# | Monitor | 4 · FluxMonitor | Planar MQW routing monitor M_R (y = -0.30 µm, x_c = +1.50 µm; Eq. 4), pos: y = -0.30 µm, x_c = +1.50 µm (W_ch = 1.05 µm), freqs: f = 568.9 THz |
# | Variation | Task 4 — Reciprocal color routing (`dual_routing`) | notes=Plane waves at λ_B = 457 nm, λ_G = 527 nm, λ_R = 611 nm across 5 angles θ_inc ∈ [-15°, +15°] |

# %%
from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
from matplotlib.collections import PolyCollection
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from wakis import figstyle, schematic

TUTORIAL_DIR = Path(__file__).resolve().parent
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    CHANNELS,
    PAPER_TARGETS,
    PIXEL_GEOMETRY,
    RGB_MATERIALS,
    build_rgb_array_solver,
    build_rgb_schematic_objects,
    escape_cone_2d_analytical,
    etendue_collimation_bound_2d,
    evaluate_method_b_statistics,
    grid_spacing_from_cpw,
    lambertian_2d_cone_fraction,
    lambertian_2d_density,
    lambertian_3d_cone_fraction,
    mqw_dipole_grid,
    overlay_rgb_geometry,
    random_phase_rse_theory,
    record_incoherent_Vs_coherent_movie_frames,
    run_dual_task_color_routing_baseline,
    run_method_b_fdtd_realizations,
    run_resolution_convergence_point,
    run_single_channel_method_a,
    save_rgb_setup_figures,
    silicon_skin_depth,
    summarize_angular_distribution,
    tidy_rgb_schematic_axes,
)

# %% [markdown]
# ## Study Tasks & Resolution Configuration
#
# Following the project resolution rules, the 2D mesh is built with
# `wakis.templates.meshing.graded_axis` snapped to all horizontal layer interfaces
# ($y = -1.10, -0.40, -0.20, 0.00\;\mu\text{m}$) and vertical mesa/trench boundaries
# ($x_c \pm W_{\text{ch}}/2$):
# $$\Delta x_{\text{bg}} = \frac{\lambda_{\min}}{\text{cpw} \cdot n_{\max,\text{MQW}}}, \qquad \Delta y_{\min} = \frac{\lambda_{\min}}{\text{cpw} \cdot n_{\max,\text{Si}}}, \qquad \lambda_{\min} = 457\;\text{nm}, \quad n_{\max,\text{MQW}} = 2.51, \quad n_{\max,\text{Si}} = 4.583.$$
# - `--full` (production study in `data/`): $\text{cpw} = 20$ ($\Delta x_{\text{bg}} = 9.08\;\text{nm}$, $\Delta y_{\min} = 4.88\;\text{nm}$, $41$ cells across the $200\;\text{nm}$ MQW/p-GaN layers, $50$ cells across the $450\;\text{nm}$ trenches, and $56.9\text{--}449.9$ cells per Silicon power skin depth $\delta_P = \lambda / (4\pi\kappa)$ [$113.9\text{--}899.8$ per field skin depth $\delta_E$], matching the $10\;\text{nm}$ design pixelation of Chen et al. 2025).
# - `--quick`: $\text{cpw} = 10$ ($\Delta x_{\text{bg}} = 18.10\;\text{nm}$, $\Delta y_{\min} = 9.52\;\text{nm}$, giving $21$ cells across the $200\;\text{nm}$ MQW and p-GaN layers, $25$ cells across the $450\;\text{nm}$ polyimide trenches, and $29.0\text{--}228.9$ cells per Silicon power skin depth $\delta_P$).
# - Convergence sweep: $\text{cpw} \in \{10, 12, 16, 20, 28\}$ ($\Delta y_{\min}$ down to $3.51\;\text{nm}$ at the 28-cpw reference run; every level satisfies $\text{cpw} \ge 10$).

# %%
STUDY_TASKS: list[dict] = [
    {"id": 0, "name": "method_a_B", "kind": "method_a", "channel": "B"},
    {"id": 1, "name": "method_a_G", "kind": "method_a", "channel": "G"},
    {"id": 2, "name": "method_a_R", "kind": "method_a", "channel": "R"},
    {"id": 3, "name": "method_b_G", "kind": "method_b", "channel": "G"},
    {"id": 4, "name": "dual_routing", "kind": "dual_routing"},
    {"id": 5, "name": "conv_cpw10", "kind": "convergence", "cpw": 10.0},
    {"id": 6, "name": "conv_cpw12", "kind": "convergence", "cpw": 12.0},
    {"id": 7, "name": "conv_cpw16", "kind": "convergence", "cpw": 16.0},
    {"id": 8, "name": "conv_cpw20", "kind": "convergence", "cpw": 20.0},
    {"id": 9, "name": "conv_cpw28", "kind": "convergence", "cpw": 28.0},
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments or ``WAKIS_TUTORIAL_ARGS`` when executed in a notebook."""
    parser = argparse.ArgumentParser(
        description="Tutorial 7.4 · RGB microLED array: incoherent averaging, collimation and crosstalk"
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the table of study tasks and exit.",
    )
    parser.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single study task by integer ID (0..9).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=TUTORIAL_DIR / "data",
        help="Output directory for .npz task data files.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=TUTORIAL_DIR / "figures",
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick resolution (10 cells/wavelength in active MQW and Silicon).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full resolution (20 cells/wavelength in active MQW and Silicon).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate all figures from existing data/ without re-running tasks.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Compatibility flag for CPU execution.",
    )
    if argv is None:
        env_args = os.environ.get("WAKIS_TUTORIAL_ARGS", "")
        if env_args and "ipykernel" in sys.modules:
            argv = shlex.split(env_args)
        elif "ipykernel" in sys.modules:
            argv = ["--quick"]
        else:
            argv = sys.argv[1:]
    return parser.parse_args(argv)


def run_task(task_id: int, out_dir: Path, full: bool = False) -> Path:
    """Execute a single study task and write its output ``.npz`` file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = STUDY_TASKS[task_id]
    name = spec["name"]
    kind = spec["kind"]
    cpw_main = 20.0 if full else 10.0
    target_path = out_dir / f"task_{task_id:02d}_{name}.npz"

    t0 = time.perf_counter()
    if kind == "method_a":
        ch = spec["channel"]
        res = run_single_channel_method_a(channel=ch, cpw=cpw_main, record_fields=True)
        meta = res["meta"]
        n_steps_val = int(res["n_steps"])
        np.savez_compressed(
            target_path,
            channel=ch,
            cpw=meta["cpw"],
            dx_m=meta["dx_m"],
            dy_m=meta["dy_m"],
            dz_m=meta["dz_m"],
            dx_background_nm=meta["dx_background_nm"],
            dx_min_nm=meta["dx_min_nm"],
            min_feature_nm=meta["min_feature_nm"],
            cells_across_feature=meta["cells_across_feature"],
            cells_across_trench=meta["cells_across_trench"],
            skin_depth_nm=meta["skin_depth_nm"],
            field_skin_depth_nm=meta["field_skin_depth_nm"],
            cells_per_skin_depth=meta["cells_per_skin_depth"],
            cells_per_field_skin_depth=meta["cells_per_field_skin_depth"],
            cells_per_power_skin_depth=meta["cells_per_power_skin_depth"],
            nx=meta["nx"],
            ny=meta["ny"],
            n_cells=meta["n_cells"],
            nt=res["nt"],
            n_steps=n_steps_val,
            runtime_s=res["runtime_s"],
            dipole_coords_m=res["dipole_coords_m"],
            x_cells_m=res["x_cells_m"],
            y_cells_m=res["y_cells_m"],
            theta_deg=res["theta_deg"],
            u_2d_per_dip=res["u_2d_per_dip"],
            e_ff_per_dip=res["e_ff_per_dip"],
            u_2d_inc=res["u_2d_inc"],
            p_ratio_density_inc=res["p_ratio_density_inc"],
            eta_20_inc_pct=res["eta_20_inc_pct"],
            u_2d_coh=res["u_2d_coh"],
            p_ratio_density_coh=res["p_ratio_density_coh"],
            eta_20_coh_pct=res["eta_20_coh_pct"],
            lee_tot_pct=res["lee_tot_pct"],
            flux_mqw_per_dip=res["flux_mqw_per_dip"],
            flux_top_per_dip=res["flux_top_per_dip"],
            eta_20_cumulative_pct=res["eta_20_cumulative_pct"],
            lee_tot_cumulative_pct=res["lee_tot_cumulative_pct"],
            win_flux_sum=res["win_flux_sum"],
            win_crosstalk_row=res["win_crosstalk_row"],
            mqw_int_sum=res["mqw_int_sum"],
            mqw_crosstalk_row=res["mqw_crosstalk_row"],
            intensity_inc_2d=res["intensity_inc_2d"],
            intensity_coh_2d=res["intensity_coh_2d"],
        )
    elif kind == "method_b":
        ch = spec["channel"]
        n_real = 16 if full else 8
        res_b = run_method_b_fdtd_realizations(
            channel=ch, cpw=cpw_main, n_realizations=n_real
        )
        meta = res_b["meta"]
        n_steps_val = int(res_b["n_steps"])
        np.savez_compressed(
            target_path,
            channel=ch,
            cpw=cpw_main,
            dx_background_nm=meta["dx_background_nm"],
            dx_min_nm=meta["dx_min_nm"],
            min_feature_nm=meta["min_feature_nm"],
            cells_across_feature=meta["cells_across_feature"],
            cells_across_trench=meta["cells_across_trench"],
            skin_depth_nm=meta["skin_depth_nm"],
            field_skin_depth_nm=meta["field_skin_depth_nm"],
            cells_per_skin_depth=meta["cells_per_skin_depth"],
            cells_per_field_skin_depth=meta["cells_per_field_skin_depth"],
            cells_per_power_skin_depth=meta["cells_per_power_skin_depth"],
            nx=meta["nx"],
            ny=meta["ny"],
            n_cells=meta["n_cells"],
            nt=n_steps_val,
            n_steps=n_steps_val,
            n_realizations=res_b["n_realizations"],
            n_dipoles=res_b["n_dipoles"],
            phases_rad=res_b["phases_rad"],
            theta_deg=res_b["theta_deg"],
            u_2d_realizations=res_b["u_2d_realizations"],
            runtime_s=res_b["runtime_s"],
        )
    elif kind == "dual_routing":
        res_dual = run_dual_task_color_routing_baseline(cpw=cpw_main)
        res_mov = record_incoherent_Vs_coherent_movie_frames(
            channel="G", cpw=cpw_main, n_frames=28
        )
        meta = res_dual["meta"]
        n_steps_val = int(res_dual["n_steps"])
        np.savez_compressed(
            target_path,
            cpw=cpw_main,
            dx_background_nm=meta["dx_background_nm"],
            dx_min_nm=meta["dx_min_nm"],
            min_feature_nm=meta["min_feature_nm"],
            cells_across_feature=meta["cells_across_feature"],
            cells_across_trench=meta["cells_across_trench"],
            skin_depth_nm=meta["skin_depth_nm"],
            field_skin_depth_nm=meta["field_skin_depth_nm"],
            cells_per_skin_depth=meta["cells_per_skin_depth"],
            cells_per_field_skin_depth=meta["cells_per_field_skin_depth"],
            cells_per_power_skin_depth=meta["cells_per_power_skin_depth"],
            nx=meta["nx"],
            ny=meta["ny"],
            n_cells=meta["n_cells"],
            nt=n_steps_val,
            n_steps=n_steps_val,
            angles_deg=res_dual["angles_deg"],
            c_raw_per_angle=res_dual["c_raw_per_angle"],
            c_norm_per_angle=res_dual["c_norm_per_angle"],
            c_norm_mean=res_dual["c_norm_mean"],
            oe_uncol_pct=res_dual["oe_uncol_pct"],
            ox_uncol_pct=res_dual["ox_uncol_pct"],
            frames_ex=res_mov["frames_ex"],
            times_fs=res_mov["times_fs"],
            mov_x_um=res_mov["x_um"],
            mov_y_um=res_mov["y_um"],
            runtime_s=time.perf_counter() - t0,
        )
    elif kind == "convergence":
        cpw_val = float(spec["cpw"])
        res_conv = run_resolution_convergence_point(cpw=cpw_val, channel="G")
        meta = res_conv
        n_steps_val = int(res_conv["n_steps"])
        np.savez_compressed(target_path, **res_conv)
    else:
        raise ValueError(f"Unknown task kind: {kind}")

    elapsed = time.perf_counter() - t0
    print(
        f"[Task {task_id:02d} | {name}] saved to {target_path.name} ({elapsed:.2f} s) | "
        f"cpw={float(meta['cpw']):.1f}, dx_bg={float(meta['dx_background_nm']):.2f} nm, "
        f"dx_min={float(meta['dx_min_nm']):.2f} nm, feat={float(meta['min_feature_nm']):.0f} nm "
        f"({int(meta['cells_across_feature'])} cells), "
        f"skin={float(meta['skin_depth_nm']):.1f} nm ({float(meta['cells_per_skin_depth']):.1f} cells/delta), "
        f"n_cells={int(meta['n_cells'])}, n_steps={n_steps_val}"
    )
    return target_path


# %% [markdown]
# ## Figure Generation & Analysis Pipeline
#
# When all study tasks have completed, `generate_all_figures_and_summary` loads the `.npz`
# archives from `data/` and produces the setup figure pairs and 15 analysis figures (including an animated GIF) in `figures/`:
#
# 0. `00a_setup_geometry.png`, `00b_setup_mesh_and_excitation.png`, `00_setup_table.md` (forward 20-dipole emission setup) and `00a_setup_geometry_dual_routing.png`, `00b_setup_mesh_and_excitation_dual_routing.png`, `00_setup_table_dual_routing.md` (reciprocal plane-wave color-routing setup) — built from live solver/grid/source/monitor objects via `wakis.schematic.save_setup_figures`.
# 1. `fig01_geometry_permittivity.png` — 2D RGB microLED cross-section, Kottke permittivity map, and $5\times 4$ MQW dipole grid.
# 2. `fig02_material_dispersion_skin_depth.png` — Table 1 optical constants and Silicon penetration/skin depth resolution.
# 3. `fig03_incoherent_rgb_intensity_maps.png` — 20-dipole incoherent near-field intensity maps $I_{\text{inc}}(x, y)$ for B, G, R channels (cf. Fig. 4b,d,f of Chen et al. 2025).
# 4. `fig04_coherent_vs_incoherent_comparison.png` — Physical incoherent power sum ($\sum |\mathbf{E}_m|^2$) vs. wrong coherent field sum ($|\sum \mathbf{E}_m|^2$).
# 5. `fig05_angular_power_density_rgb.png` — Far-field normalized angular power ratio density $\bar{P}_r(\theta)$ (Eq. 5, Fig. 4c,e,g of Chen et al. 2025) vs. 2D Lambertian cosine baseline.
# 6. `fig06_polar_radiation_patterns.png` — Polar radiation patterns for B, G, R channels and the $\pm 20^\circ$ target cone.
# 7. `fig07_collimation_efficiency_comparison.png` — $\eta_{\pm 20^\circ}$ and $\text{LEE}_{\text{tot}}$ comparison against Chen et al. (2025), 2D/3D Lambertian, and 2D étendue limits.
# 8. `fig08_dipole_count_convergence.png` — Convergence of $\eta_{\pm 20^\circ}(N)$ and $\text{LEE}_{\text{tot}}(N)$ vs. dipole count $N = 1\dots 20$.
# 9. `fig09_method_a_vs_method_b_profiles.png` — Wakis FDTD Method B ($K = 1, 2, 4, 8, 16$ random-phase realizations) vs. exact Method A angular profile.
# 10. `fig10_method_b_bias_variance_cost.png` — Method B Monte Carlo relative RMSE vs. exact $\sqrt{(1-1/N)/K}$, cost trade-off, and Jensen ratio bias.
# 11. `fig11_forward_crosstalk_matrices.png` — Forward inter-pixel exit-window and MQW crosstalk matrices $C_{ij}^{\text{emit}}$.
# 12. `fig12_dual_task_color_routing.png` — Reciprocal dual-task plane-wave color-routing matrix $C_{ij}^{\text{dual}}$ (Eq. 4, Fig. 3d of Chen et al. 2025).
# 13. `fig13_resolution_convergence.png` — Spatial grid and skin-depth convergence from $\text{cpw} = 10$ to the $\text{cpw} = 28$ reference run.
# 14. `fig14_time_domain_snapshots.png` — 3-panel time-domain snapshots of $E_x(x, y, t)$ during an incoherent multi-dipole burst.
# 15. `fig15_rgb_emission_movie.gif` — Animated time-domain wave propagation GIF with labelled geometry overlay.


# %%
def generate_all_figures_and_summary(data_dir: Path, fig_dir: Path) -> dict:
    """Load task outputs from ``data_dir`` and create all 16 figures in ``fig_dir``."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    figstyle.apply_style()

    data_a = {
        "B": np.load(data_dir / "task_00_method_a_B.npz"),
        "G": np.load(data_dir / "task_01_method_a_G.npz"),
        "R": np.load(data_dir / "task_02_method_a_R.npz"),
    }
    data_b = np.load(data_dir / "task_03_method_b_G.npz")
    data_dual = np.load(data_dir / "task_04_dual_routing.npz")
    conv_files = [
        data_dir / "task_05_conv_cpw10.npz",
        data_dir / "task_06_conv_cpw12.npz",
        data_dir / "task_07_conv_cpw16.npz",
        data_dir / "task_08_conv_cpw20.npz",
        data_dir / "task_09_conv_cpw28.npz",
    ]
    conv_list = [np.load(f) for f in conv_files]

    # -------------------------------------------------------------------------
    # Figure 00: Setup Schematics & Tables (Forward 20-Dipole + Reciprocal PW)
    # -------------------------------------------------------------------------
    cpw_fig1 = float(data_a["G"]["cpw"])
    fwd_obj = build_rgb_schematic_objects(mode="forward", channel="G", cpw=cpw_fig1)
    save_rgb_setup_figures(fig_dir, cpw=cpw_fig1)

    # -------------------------------------------------------------------------
    # Figure 1: 2D Geometry, Kottke Permittivity & 20-Dipole MQW Sampling Grid
    # -------------------------------------------------------------------------
    from scipy.constants import epsilon_0 as _eps0

    grid_g, solver_g = fwd_obj["grid"], fwd_obj["solver"]
    eps_x = np.real(
        1.0 / np.maximum(1e-30, _eps0 * np.asarray(solver_g.ieps[:, :, 0, "x"]))
    )
    n_map = np.sqrt(np.clip(eps_x, 1.0, 25.0))
    x_um = (grid_g.x[:-1] + 0.5 * grid_g.dx) * 1e6
    y_um = (grid_g.y[:-1] + 0.5 * grid_g.dy) * 1e6

    fig, ax = figstyle.new_figure(nrows=1, ncols=1, width=9.2, height=4.4)
    im = ax.pcolormesh(
        x_um, y_um, n_map.T, cmap="viridis", shading="auto", vmin=1.0, vmax=4.3
    )
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Refractive index $n(x, y)$ at $\\lambda_G = 527$ nm [-]")
    for ch in CHANNELS:
        dips = mqw_dipole_grid(ch) * 1e6
        ax.scatter(
            dips[:, 0],
            dips[:, 1],
            s=18,
            c=RGB_MATERIALS[ch]["color"],
            edgecolors="white",
            linewidths=0.6,
            zorder=8,
            label=f"{ch} MQW dipoles (5×4=20)" if ch == "B" else f"{ch} MQW dipoles",
        )
    overlay_rgb_geometry(
        ax,
        active_channel=None,
        line_color="white",
        show_dipoles=False,
        show_monitors=True,
        show_labels=True,
    )
    ax.set_xlabel("Lateral coordinate $x$ [$\\mu$m]")
    ax.set_ylabel("Vertical coordinate $y$ [$\\mu$m]")
    ax.set_title("3-Pixel RGB Micro-LED Cross-Section, Kottke Refractive Index Map & 20-Dipole MQW Grids")
    ax.legend(loc="upper left", ncol=3, framealpha=0.88, fontsize=7.5)
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "fig01_geometry_permittivity.png")

    # -------------------------------------------------------------------------
    # Figure 2: Table 1 Material Dispersion & Silicon Skin Depth Resolution
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=10.2, height=4.2)
    wls_nm = [RGB_MATERIALS[ch]["wavelength_m"] * 1e9 for ch in CHANNELS]
    x_idx = np.arange(3)
    width_b = 0.14
    mat_series = [
        ("GaN", [RGB_MATERIALS[ch]["n_GaN"] for ch in CHANNELS], figstyle.PALETTE[0]),
        ("MQW", [RGB_MATERIALS[ch]["n_MQW"] for ch in CHANNELS], figstyle.PALETTE[1]),
        ("SiN$_x$", [RGB_MATERIALS[ch]["n_SiNx"] for ch in CHANNELS], figstyle.PALETTE[2]),
        ("Polyimide", [RGB_MATERIALS[ch]["n_PI"] for ch in CHANNELS], figstyle.PALETTE[4]),
        ("SiO$_2$", [RGB_MATERIALS[ch]["n_SiO2"] for ch in CHANNELS], figstyle.PALETTE[7]),
        ("Si ($n$)", [RGB_MATERIALS[ch]["n_Si_real"] for ch in CHANNELS], figstyle.PALETTE[5]),
    ]
    for i_m, (m_label, vals, col) in enumerate(mat_series):
        ax1.bar(
            x_idx + (i_m - 2.5) * width_b,
            vals,
            width=width_b,
            label=m_label,
            color=col,
            edgecolor="black",
            linewidth=0.5,
        )
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels([f"{ch}\n({w:.0f} nm)" for ch, w in zip(CHANNELS, wls_nm)])
    ax1.set_ylabel("Real refractive index $n(\\lambda)$ [-]")
    ax1.set_title("(a) Table 1 Refractive Indices (Chen et al., 2025)")
    ax1.legend(ncol=2, loc="upper right", fontsize=7.5)

    delta_e_nm = [
        silicon_skin_depth(
            RGB_MATERIALS[ch]["wavelength_m"], RGB_MATERIALS[ch]["n_Si_imag"]
        )["delta_field_m"]
        * 1e9
        for ch in CHANNELS
    ]
    delta_p_nm = [0.5 * d for d in delta_e_nm]
    dx_full_bg_nm = float(data_a["B"]["dx_background_nm"])
    dx_full_si_nm = float(data_a["B"]["dx_min_nm"])
    dx_quick_si_nm = float(conv_list[0]["dx_min_nm"])

    ax2.plot(wls_nm, delta_e_nm, **figstyle.series_style(0), label="Field skin depth $\\delta_E = \\lambda / (2\\pi \\kappa_{\\mathrm{Si}})$")
    ax2.plot(wls_nm, delta_p_nm, **figstyle.series_style(1), label="Power skin depth $\\delta_P = \\lambda / (4\\pi \\kappa_{\\mathrm{Si}})$")
    ax2.axhline(dx_full_bg_nm, color=figstyle.PALETTE[7], linestyle=":", lw=1.4, label=f"--full $\\Delta x_{{\\mathrm{{bg}}}} = {dx_full_bg_nm:.2f}$ nm (GaN/PI, cpw=20)")
    ax2.axhline(dx_quick_si_nm, color=figstyle.PALETTE[3], linestyle="--", lw=1.5, label=f"--quick $\\Delta y_{{\\min}} = {dx_quick_si_nm:.2f}$ nm (MQW/Si, cpw=10)")
    ax2.axhline(dx_full_si_nm, color=figstyle.PALETTE[2], linestyle="-.", lw=1.5, label=f"--full $\\Delta y_{{\\min}} = {dx_full_si_nm:.2f}$ nm (MQW/Si, cpw=20)")
    ax2.set_yscale("log")
    ax2.set_xlabel("Wavelength $\\lambda_0$ [nm]")
    ax2.set_ylabel("Length scale [nm]")
    ax2.set_title("(b) Silicon Skin Depth vs. Graded Wakis FIT Cell Sizes")
    ax2.legend(loc="upper left", fontsize=7.4)
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "fig02_material_dispersion_skin_depth.png")

    # -------------------------------------------------------------------------
    # Figure 3: 3-Panel Spatially Incoherent Near-Field Intensity Maps (B, G, R)
    # -------------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.0, height=4.1, sharey=True)
    for ax, ch in zip(axes, CHANNELS):
        d = data_a[ch]
        x_c = d["x_cells_m"] * 1e6
        y_c = d["y_cells_m"] * 1e6
        inten = d["intensity_inc_2d"]
        ref_i = max(float(np.percentile(inten, 99.5)), 1e-30)
        inten_db = 10.0 * np.log10(np.clip(inten / ref_i, 1e-3, 1.0))
        im = ax.pcolormesh(
            x_c, y_c, inten_db.T, cmap=figstyle.CMAP_MAGNITUDE, shading="auto", vmin=-25.0, vmax=0.0
        )
        overlay_rgb_geometry(
            ax,
            active_channel=ch,
            line_color="white",
            show_dipoles=True,
            show_monitors=True,
            show_labels=True,
            compact=True,
        )
        ax.set_xlabel("Lateral position $x$ [$\\mu$m]")
        ax.set_title(f"{RGB_MATERIALS[ch]['label']}\n20-Dipole Incoherent $\\sum_m |\\mathbf{{E}}_m|^2$")
    axes[0].set_ylabel("Vertical position $y$ [$\\mu$m]")
    cb = fig.colorbar(im, ax=axes[-1], pad=0.02)
    cb.set_label("Normalized intensity $10\\log_{10}(I_{\\mathrm{inc}} / I_{99.5\\%})$ [dB]")
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "fig03_incoherent_rgb_intensity_maps.png")

    # -------------------------------------------------------------------------
    # Figure 4: Incoherent Power Sum vs. Wrong Coherent Field Sum (G Channel)
    # -------------------------------------------------------------------------
    dg = data_a["G"]
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.0, height=4.1)
    x_c = dg["x_cells_m"] * 1e6
    y_c = dg["y_cells_m"] * 1e6
    ref_inc = max(float(np.percentile(dg["intensity_inc_2d"], 99.5)), 1e-30)
    ref_coh = max(float(np.percentile(dg["intensity_coh_2d"], 99.5)), 1e-30)
    i_inc_db = 10.0 * np.log10(np.clip(dg["intensity_inc_2d"] / ref_inc, 1e-3, 1.0))
    i_coh_db = 10.0 * np.log10(np.clip(dg["intensity_coh_2d"] / ref_coh, 1e-3, 1.0))

    im0 = axes[0].pcolormesh(x_c, y_c, i_inc_db.T, cmap=figstyle.CMAP_MAGNITUDE, shading="auto", vmin=-25, vmax=0)
    overlay_rgb_geometry(axes[0], active_channel="G", line_color="white", show_dipoles=True, show_monitors=True, show_labels=True, compact=True)
    axes[0].set_title("(a) Incoherent Sum: $\\sum_{m=1}^{20} |\\mathbf{E}_m(x,y)|^2$")
    axes[0].set_xlabel("$x$ [$\\mu$m]")
    axes[0].set_ylabel("$y$ [$\\mu$m]")
    fig.colorbar(im0, ax=axes[0], label="[dB]")

    im1 = axes[1].pcolormesh(x_c, y_c, i_coh_db.T, cmap=figstyle.CMAP_MAGNITUDE, shading="auto", vmin=-25, vmax=0)
    overlay_rgb_geometry(axes[1], active_channel="G", line_color="white", show_dipoles=True, show_monitors=True, show_labels=True, compact=True)
    axes[1].set_title("(b) Wrong Coherent Sum: $|\\sum_{m=1}^{20} \\mathbf{E}_m(x,y)|^2$")
    axes[1].set_xlabel("$x$ [$\\mu$m]")
    axes[1].set_ylabel("$y$ [$\\mu$m]")
    fig.colorbar(im1, ax=axes[1], label="[dB]")

    th = dg["theta_deg"]
    axes[2].plot(
        th,
        dg["p_ratio_density_inc"],
        **figstyle.series_style(2),
        label=f"Incoherent $\\sum |E_m|^2$ ($\\eta_{{\\pm 20^\\circ}}={float(dg['eta_20_inc_pct']):.1f}\\%$)",
    )
    axes[2].plot(
        th,
        dg["p_ratio_density_coh"],
        **figstyle.series_style(5),
        label=f"Wrong coherent $|\\sum E_m|^2$ ($\\eta_{{\\pm 20^\\circ}}={float(dg['eta_20_coh_pct']):.1f}\\%$)",
    )
    axes[2].plot(
        th,
        lambertian_2d_density(th),
        color="black",
        linestyle=":",
        lw=1.4,
        label="2D Lambertian ($\\sin 20^\\circ = 34.2\\%$)",
    )
    axes[2].axvspan(-20, 20, color="#fefcbf", alpha=0.5, label="$\\pm 20^\\circ$ target cone")
    axes[2].set_xlabel("Far-field angle $\\theta$ [deg]")
    axes[2].set_ylabel("Power ratio density $\\bar{P}_r(\\theta)$ [deg$^{-1}$]")
    axes[2].set_title("(c) Far-Field Phasing Error")
    axes[2].legend(loc="upper right", fontsize=7.2)
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "fig04_coherent_vs_incoherent_comparison.png")

    # -------------------------------------------------------------------------
    # Figure 5: Reproduced Angular Power Ratio Density (Fig. 4c,e,g of Paper)
    # -------------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=12.6, height=4.1, sharey=True)
    th = data_a["B"]["theta_deg"]
    p_lamb = lambertian_2d_density(th)
    for ax, ch in zip(axes, CHANNELS):
        d = data_a[ch]
        col = RGB_MATERIALS[ch]["color"]
        eta_w = float(d["eta_20_inc_pct"])
        eta_pub = PAPER_TARGETS[ch]["eta_20_uncol_pct"]
        ax.axvspan(-20, 20, color=col, alpha=0.12, label="$\\pm 20^\\circ$ coupling cone")
        ax.plot(
            th,
            d["p_ratio_density_inc"],
            color=col,
            lw=2.2,
            label=f"Wakis 20-dip ($\\eta_{{\\pm 20^\\circ}} = {eta_w:.2f}\\%$)",
        )
        ax.plot(
            th,
            p_lamb,
            "k--",
            lw=1.4,
            label=f"2D Lambertian ($34.20\\%$);\nChen et al. baseline: ${eta_pub:.2f}\\%$",
        )
        u_dips = d["u_2d_per_dip"]
        dtheta = 180.0 / len(th)
        p_dips = u_dips / (np.sum(u_dips, axis=1, keepdims=True) * dtheta)
        ax.fill_between(
            th,
            np.percentile(p_dips, 15, axis=0),
            np.percentile(p_dips, 85, axis=0),
            color=col,
            alpha=0.18,
            label="Single-dipole 15–85% band",
        )
        ax.set_xlim(-90, 90)
        ax.set_xlabel("Far-field emission angle $\\theta$ [deg]")
        ax.set_title(f"{RGB_MATERIALS[ch]['label']} (cf. Chen et al. Fig. 4)")
        ax.legend(loc="upper right", fontsize=7.5)
    axes[0].set_ylabel("Normalized angular power density $\\bar{P}_r(\\theta)$ [deg$^{-1}$]")
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "fig05_angular_power_density_rgb.png")

    # -------------------------------------------------------------------------
    # Figure 6: Polar Radiation Patterns of B, G, R vs. 2D Lambertian
    # -------------------------------------------------------------------------
    fig, ax = figstyle.new_figure(
        nrows=1, ncols=1, width=6.6, height=5.4, subplot_kw={"projection": "polar"}
    )
    ax.add_collection(PolyCollection([[(0.0, 0.0), (0.0, 0.0)]], alpha=0.0))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    th_rad = np.deg2rad(th)
    for ch in CHANNELS:
        d = data_a[ch]
        ax.plot(
            th_rad,
            d["p_ratio_density_inc"],
            color=RGB_MATERIALS[ch]["color"],
            lw=2.0,
            label=f"{ch} ({RGB_MATERIALS[ch]['wavelength_m']*1e9:.0f} nm): $\\eta_{{\\pm 20^\\circ}}={float(d['eta_20_inc_pct']):.1f}\\%$",
        )
    ax.plot(th_rad, p_lamb, "k--", lw=1.5, label="2D Lambertian ($\\cos\\theta$, $34.2\\%$)")
    ax.axvline(np.deg2rad(-20.0), color="#d69e2e", linestyle=":", lw=1.5)
    ax.axvline(np.deg2rad(+20.0), color="#d69e2e", linestyle=":", lw=1.5, label="$\\pm 20^\\circ$ acceptance cone")
    ax.set_title("Polar Far-Field Radiation Patterns $\\bar{P}_r(\\theta)$ [deg$^{-1}$]", pad=16)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.8)
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "fig06_polar_radiation_patterns.png")

    # -------------------------------------------------------------------------
    # Figure 7: Collimation Efficiency & Extraction Efficiency vs. Published/Theory
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=11.2, height=4.3)
    x_pos = np.arange(3)
    w_bar = 0.24
    eta_wakis = [float(data_a[ch]["eta_20_inc_pct"]) for ch in CHANNELS]
    eta_paper_uncol = [PAPER_TARGETS[ch]["eta_20_uncol_pct"] for ch in CHANNELS]
    eta_paper_col = [PAPER_TARGETS[ch]["eta_20_col_pct"] for ch in CHANNELS]

    ax1.bar(x_pos - w_bar, eta_wakis, width=w_bar, color=figstyle.PALETTE[0], edgecolor="black", label="Wakis 2D FIT (uncollimated)")
    ax1.bar(x_pos, eta_paper_uncol, width=w_bar, color=figstyle.PALETTE[1], edgecolor="black", label="Chen et al. 2025 (uncollimated)")
    ax1.bar(x_pos + w_bar, eta_paper_col, width=w_bar, color=figstyle.PALETTE[2], edgecolor="black", label="Chen et al. 2025 (with collimator)")
    ax1.axhline(lambertian_2d_cone_fraction(20.0) * 100.0, color="black", linestyle="--", lw=1.4, label="2D Lambertian $\\sin 20^\\circ = 34.20\\%$")
    ax1.axhline(etendue_collimation_bound_2d(1.5e-6, 1.05e-6, 20.0) * 100.0, color="#d69e2e", linestyle="-.", lw=1.4, label="1-subpixel étendue cap ($48.86\\%$)")
    ax1.axhline(lambertian_3d_cone_fraction(20.0) * 100.0, color="#718096", linestyle=":", lw=1.3, label="3D Lambertian $\\sin^2 20^\\circ = 11.70\\%$")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([RGB_MATERIALS[ch]["label"] for ch in CHANNELS])
    ax1.set_ylabel("Power fraction within $\\pm 20^\\circ$, $\\eta_{\\pm 20^\\circ}$ [%]")
    ax1.set_ylim(0, 78)
    ax1.set_title("(a) Far-Field Collimation Efficiency $\\eta_{\\pm 20^\\circ}$")
    ax1.legend(loc="upper left", fontsize=7.4)

    lee_wakis = [float(data_a[ch]["lee_tot_pct"]) for ch in CHANNELS]
    lee_paper_uncol = [PAPER_TARGETS[ch]["lee_tot_uncol_pct"] for ch in CHANNELS]
    lee_analytic = [
        escape_cone_2d_analytical(
            RGB_MATERIALS[ch]["n_MQW"],
            1.0,
            RGB_MATERIALS[ch]["n_Si_real"],
            RGB_MATERIALS[ch]["n_Si_imag"],
        )["lee_2d_planar_est"]
        * 100.0
        for ch in CHANNELS
    ]
    ax2.bar(x_pos - w_bar, lee_wakis, width=w_bar, color=figstyle.PALETTE[1], edgecolor="black", label="Wakis 2D FIT ($y_{\\mathrm{mon}}=0.25\\;\\mu$m)")
    ax2.bar(x_pos, lee_paper_uncol, width=w_bar, color=figstyle.PALETTE[3], edgecolor="black", label="Chen et al. 2025 (top of $2\\;\\mu$m region)")
    ax2.bar(x_pos + w_bar, lee_analytic, width=w_bar, color=figstyle.PALETTE[4], edgecolor="black", label="Planar 2D escape-cone estimate")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([RGB_MATERIALS[ch]["label"] for ch in CHANNELS])
    ax2.set_ylabel("Total extraction efficiency $\\mathrm{LEE}_{\\mathrm{tot}}$ [%]")
    ax2.set_ylim(0, 42)
    ax2.set_title("(b) Total Extraction $\\mathrm{LEE}_{\\mathrm{tot}} = P_{\\mathrm{top}} / P_{\\mathrm{MQW}}$")
    ax2.legend(loc="upper right", fontsize=7.6)
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "fig07_collimation_efficiency_comparison.png")

    # -------------------------------------------------------------------------
    # Figure 8: Convergence of Incoherent Metrics vs. Number of Dipoles N
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=10.8, height=4.1)
    n_axis = np.arange(1, 21)
    for ch in CHANNELS:
        d = data_a[ch]
        col = RGB_MATERIALS[ch]["color"]
        ax1.plot(n_axis, d["eta_20_cumulative_pct"], "o-", color=col, lw=1.8, markersize=4, label=f"{ch} ($N=20$: {float(d['eta_20_inc_pct']):.2f}%)")
        ax1.axhline(PAPER_TARGETS[ch]["eta_20_uncol_pct"], color=col, linestyle="--", alpha=0.6)
        ax2.plot(n_axis, d["lee_tot_cumulative_pct"], "s-", color=col, lw=1.8, markersize=4, label=f"{ch} ($N=20$: {float(d['lee_tot_pct']):.2f}%)")
        ax2.axhline(PAPER_TARGETS[ch]["lee_tot_uncol_pct"], color=col, linestyle="--", alpha=0.6)

    ax1.set_xlabel("Number of MQW dipoles $N$ [-]")
    ax1.set_ylabel("Cumulative $\\eta_{\\pm 20^\\circ}(N)$ [%]")
    ax1.set_title("(a) Convergence of $\\eta_{\\pm 20^\\circ}$ vs. Dipole Count $N$")
    ax1.legend()

    ax2.set_xlabel("Number of MQW dipoles $N$ [-]")
    ax2.set_ylabel("Cumulative $\\mathrm{LEE}_{\\mathrm{tot}}(N)$ [%]")
    ax2.set_title("(b) Convergence of $\\mathrm{LEE}_{\\mathrm{tot}}$ vs. Dipole Count $N$")
    ax2.legend()
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "fig08_dipole_count_convergence.png")

    # -------------------------------------------------------------------------
    # Figure 9: Method A vs. Method B FDTD Realizations (Angular Profiles)
    # -------------------------------------------------------------------------
    fig, ax = figstyle.new_figure(nrows=1, ncols=1, width=8.6, height=4.4)
    u_reals = data_b["u_2d_realizations"]
    n_reals_fdtd = len(u_reals)
    k_show = [k for k in (1, 2, 4, 8, 16) if k <= n_reals_fdtd]
    colors_k = [figstyle.PALETTE[7], figstyle.PALETTE[3], figstyle.PALETTE[6], figstyle.PALETTE[0], figstyle.PALETTE[4]][: len(k_show)]
    for k_val, c_k in zip(k_show, colors_k):
        u_mean_k = np.mean(u_reals[:k_val], axis=0)
        sum_k = summarize_angular_distribution(th, u_mean_k)
        ax.plot(
            th,
            sum_k["p_ratio_density"],
            color=c_k,
            lw=1.5 if k_val < n_reals_fdtd else 2.2,
            linestyle="-." if k_val == 1 else "-",
            label=f"Method B FDTD ($K={k_val}$ runs): $\\eta_{{\\pm 20^\\circ}}={sum_k['eta_20_pct']:.2f}\\%$",
        )
    ax.plot(
        th,
        dg["p_ratio_density_inc"],
        color=figstyle.PALETTE[5],
        linestyle="--",
        lw=2.4,
        label=f"Method A Exact ($N=20$ runs): $\\eta_{{\\pm 20^\\circ}}={float(dg['eta_20_inc_pct']):.2f}\\%$",
    )
    ax.set_xlim(-90, 90)
    ax.set_xlabel("Far-field emission angle $\\theta$ [deg]")
    ax.set_ylabel("Normalized angular power density $\\bar{P}_r(\\theta)$ [deg$^{-1}$]")
    ax.set_title("Method A (20 Single-Dipole Runs) vs. Method B ($K$ Random-Phase 20-Dipole FDTD Runs)")
    ax.legend(loc="upper right")
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "fig09_method_a_vs_method_b_profiles.png")

    # -------------------------------------------------------------------------
    # Figure 10: Method B Bias, Variance ($\propto K^{-1/2}$) & Cost Crossover
    # -------------------------------------------------------------------------
    stats_b = evaluate_method_b_statistics(
        dg["e_ff_per_dip"],
        dg["u_2d_inc"],
        th,
        k_values=(1, 2, 4, 8, 16, 32, 64),
        n_trials=400,
    )
    k_fdtd_axis = np.arange(1, n_reals_fdtd + 1)
    rse_fdtd = np.zeros(n_reals_fdtd, dtype=float)
    p_exact_g = dg["p_ratio_density_inc"]
    mask_g = p_exact_g > 0.15 * np.max(p_exact_g)
    for k_i in k_fdtd_axis:
        p_k = summarize_angular_distribution(th, np.mean(u_reals[:k_i], axis=0))[
            "p_ratio_density"
        ]
        rse_fdtd[k_i - 1] = float(
            np.sqrt(np.mean(((p_k[mask_g] - p_exact_g[mask_g]) / p_exact_g[mask_g]) ** 2))
        )

    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=11.0, height=4.3)
    k_vals = stats_b["k_values"]
    ax1.loglog(
        k_vals,
        stats_b["rse_theory"] * 100.0,
        "k--",
        lw=2.0,
        label="Exact theory $\\sqrt{(1 - 1/N)/K}$ ($N=20$)",
    )
    ax1.loglog(
        k_vals,
        stats_b["rse_empirical"] * 100.0,
        **figstyle.series_style(0),
        label="Method B Monte Carlo ensemble (400 trials)",
    )
    ax1.loglog(
        k_fdtd_axis,
        rse_fdtd * 100.0,
        "s",
        color=figstyle.PALETTE[5],
        markersize=6,
        label="Single Wakis FDTD trajectory ($k=1\\dots K$)",
    )
    ax1.axvline(20, color=figstyle.PALETTE[2], linestyle=":", lw=1.6, label="Cost break-even $K = N_{\\mathrm{dip}} = 20$")
    ax1.set_xlabel("Number of random-phase realizations $K$ [-]")
    ax1.set_ylabel("Relative angular root-mean-square error [%]")
    ax1.set_title("(a) Method B Variance Scaling ($K^{-1/2}$) & Cost Crossover")
    ax1.legend(loc="upper right", fontsize=7.6)

    ax2.errorbar(
        k_vals,
        stats_b["eta_bias_vs_k_pct"],
        yerr=stats_b["eta_std_vs_k_pct"],
        fmt="o-",
        color=figstyle.PALETTE[2],
        capsize=3,
        lw=1.8,
        label="Proper ratio of $K$-means: $\\overline{P}_{\\mathrm{cone}}^{(K)} / \\overline{P}_{\\mathrm{tot}}^{(K)}$ ($\\pm 1\\sigma$)",
    )
    ax2.plot(
        k_vals,
        stats_b["eta_naive_ratio_bias_pct"],
        "d--",
        color=figstyle.PALETTE[5],
        lw=1.8,
        label="Naive mean of single-run ratios (Jensen $O(1/K)$ bias)",
    )
    ax2.axhline(0.0, color="black", linestyle=":", lw=1.2)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Number of random-phase realizations $K$ [-]")
    ax2.set_ylabel("Bias in $\\eta_{\\pm 20^\\circ}^{(K)} - \\eta_{\\pm 20^\\circ}^{\\mathrm{exact}}$ [% pts]")
    ax2.set_title("(b) Method B Estimator Unbiasedness vs. Naive Ratio Bias")
    ax2.legend(loc="upper right", fontsize=7.6)
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "fig10_method_b_bias_variance_cost.png")

    # -------------------------------------------------------------------------
    # Figure 11: Forward Emission Inter-Pixel Crosstalk Matrices C_ij
    # -------------------------------------------------------------------------
    c_win = np.zeros((3, 3), dtype=float)
    c_mqw = np.zeros((3, 3), dtype=float)
    for j_idx, ch in enumerate(CHANNELS):
        c_win[:, j_idx] = data_a[ch]["win_crosstalk_row"] * 100.0
        c_mqw[:, j_idx] = data_a[ch]["mqw_crosstalk_row"] * 100.0

    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=10.6, height=4.3)
    for ax, mat_c, title_str in [
        (ax1, c_win, "(a) Exit-Window Flux Routing $C_{ij}^{\\mathrm{win}}$ [%]"),
        (ax2, c_mqw, "(b) MQW Lateral Intensity Crosstalk $C_{ij}^{\\mathrm{mqw}}$ [%]"),
    ]:
        im = ax.imshow(mat_c, cmap="Blues", vmin=0.0, vmax=100.0)
        for r_i in range(3):
            for c_j in range(3):
                val = mat_c[r_i, c_j]
                ax.text(
                    c_j,
                    r_i,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if val > 50.0 else "black",
                    fontweight="bold",
                )
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([f"Emit {ch}" for ch in CHANNELS])
        ax.set_yticklabels([f"Monitor {ch}" for ch in CHANNELS])
        ax.set_xlabel("Active emitting subpixel $j$")
        ax.set_ylabel("Receiving subpixel monitor $i$")
        ax.set_title(title_str)
        fig.colorbar(im, ax=ax, shrink=0.84, label="[%]")
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "fig11_forward_crosstalk_matrices.png")

    # -------------------------------------------------------------------------
    # Figure 12: Dual-Task Plane-Wave Color-Routing Matrix C_ij (Eq. 4, Fig. 3d)
    # -------------------------------------------------------------------------
    c_dual_mean = data_dual["c_norm_mean"] * 100.0
    c_dual_ang = data_dual["c_norm_per_angle"] * 100.0
    angles_dual = data_dual["angles_deg"]

    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=10.8, height=4.3)
    im = ax1.imshow(c_dual_mean, cmap="Greens", vmin=0.0, vmax=60.0)
    for r_i in range(3):
        for c_j in range(3):
            val = c_dual_mean[r_i, c_j]
            ax1.text(
                c_j,
                r_i,
                f"{val:.1f}%",
                ha="center",
                va="center",
                color="white" if val > 38.0 else "black",
                fontweight="bold",
            )
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels([f"$\\lambda_{{{ch}}}$" for ch in CHANNELS])
    ax1.set_yticklabels([f"Monitor $M_{{{ch}}}$" for ch in CHANNELS])
    ax1.set_xlabel("Incident plane-wave wavelength $\\lambda_j$")
    ax1.set_ylabel("MQW monitor $M_i$ ($y = -0.30\\;\\mu$m)")
    ax1.set_title("(a) Uncollimated Dual-Task Matrix $\\tilde{C}_{ij}^{\\mathrm{dual}}$ (Eq. 4)")
    fig.colorbar(im, ax=ax1, shrink=0.84, label="Routed fraction [%]")

    x_ang = np.arange(len(angles_dual))
    w_a = 0.24
    for idx_c, ch in enumerate(CHANNELS):
        oe_vs_ang = c_dual_ang[:, idx_c, idx_c]
        ax2.bar(
            x_ang + (idx_c - 1) * w_a,
            oe_vs_ang,
            width=w_a,
            color=RGB_MATERIALS[ch]["color"],
            edgecolor="black",
            label=f"Baseline $\\mathrm{{OE}}({ch})$ (~{np.mean(oe_vs_ang):.1f}%)",
        )
    ax2.axhline(33.33, color="black", linestyle="--", lw=1.3, label="Uniform unrouted baseline (33.3%)")
    ax2.axhline(80.0, color=figstyle.PALETTE[2], linestyle="-.", lw=1.6, label="Chen et al. Fig. 3d router (~80%)")
    ax2.set_xticks(x_ang)
    ax2.set_xticklabels([f"{a:+.1f}°" for a in angles_dual])
    ax2.set_xlabel("Incident plane-wave angle $\\theta_{\\mathrm{inc}}$ [deg]")
    ax2.set_ylabel("Diagonal optical routing efficiency $\\mathrm{OE}(i)$ [%]")
    ax2.set_ylim(0, 92)
    ax2.set_title("(b) Dual-Task Color Routing vs. Incident Angle")
    ax2.legend(loc="upper right", fontsize=7.6)
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "fig12_dual_task_color_routing.png")

    # -------------------------------------------------------------------------
    # Figure 13: Spatial Resolution & Skin-Depth Convergence (cpw = 10, 12, 16, 20, 28)
    # -------------------------------------------------------------------------
    cpws = np.array([float(c["cpw"]) for c in conv_list])
    dxs_nm = np.array([float(c["dx_min_nm"]) for c in conv_list])
    cps_skin = np.array([float(c["cells_per_skin_depth"]) for c in conv_list])
    lees = np.array([float(c["lee_tot_pct"]) for c in conv_list])
    etas = np.array([float(c["eta_20_pct"]) for c in conv_list])
    runtimes = np.array([float(c["runtime_s"]) for c in conv_list])

    rel_err_lee = np.abs(lees[:-1] - lees[-1]) / max(1e-12, abs(lees[-1])) * 100.0
    rel_err_eta = np.abs(etas[:-1] - etas[-1]) / max(1e-12, abs(etas[-1])) * 100.0

    fig, (ax1, ax2) = figstyle.new_figure(nrows=1, ncols=2, width=11.2, height=4.4)
    ax1.plot(cpws, etas, **figstyle.series_style(0), label="$\\eta_{\\pm 20^\\circ}$ [%]")
    ax1.plot(cpws, lees, **figstyle.series_style(1), label="$\\mathrm{LEE}_{\\mathrm{tot}}$ [%]")
    ax1.axvline(10, color=figstyle.PALETTE[7], linestyle=":", lw=1.5, label="--quick (10 cpw, 91 c/$\\delta_P$)")
    ax1.axvline(20, color=figstyle.PALETTE[2], linestyle="-.", lw=1.7, label="Accepted --full (20 cpw, 179 c/$\\delta_P$, $<1\\%$ $\\Delta$)")
    ax1.set_xticks(cpws)
    ax1.set_xticklabels([f"{c:.0f}\n({s:.0f} c/$\\delta_P$)" for c, s in zip(cpws, cps_skin)])
    ax1.set_xlabel("Cells per minimum wavelength $\\mathrm{cpw}$ (and cells per Si $\\delta_P$ at 527 nm)")
    ax1.set_ylabel("Efficiency metric [%]")
    ax1.set_title("(a) Grid & Skin-Depth Convergence (up to 28-cpw Reference)")
    ax1.legend(loc="best", fontsize=7.6)

    ax2.loglog(dxs_nm[:-1], np.maximum(rel_err_eta, 1e-2), **figstyle.series_style(0), label="$|\\Delta \\eta_{\\pm 20^\\circ}| / \\eta_{\\mathrm{ref}}$ [%]")
    ax2.loglog(dxs_nm[:-1], np.maximum(rel_err_lee, 1e-2), **figstyle.series_style(1), label="$|\\Delta \\mathrm{LEE}_{\\mathrm{tot}}| / \\mathrm{LEE}_{\\mathrm{ref}}$ [%]")
    dx_guide = np.linspace(dxs_nm[-2], dxs_nm[0], 50)
    err_guide = max(rel_err_eta[0], 1.0) * (dx_guide / dxs_nm[0]) ** 2
    ax2.loglog(dx_guide, err_guide, "k--", lw=1.4, label="$O(\\Delta y_{\\min}^2)$ second-order guide")
    ax2.axhline(1.0, color=figstyle.PALETTE[2], linestyle="-.", lw=1.3, label="1% convergence threshold (met at 20 cpw)")
    ax2.set_xlabel("Finest vertical cell size $\\Delta y_{\\min}$ (MQW & Si) [nm]")
    ax2.set_ylabel("Relative error vs. 28-cpw reference [%]")
    ax2.set_title("(b) Relative Discretization Error vs. Cell Size $\\Delta y_{\\min}$")
    ax2.legend(loc="best", fontsize=7.6)
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "fig13_resolution_convergence.png")

    # -------------------------------------------------------------------------
    # Figure 14: 3-Panel Time-Domain Wavefront Snapshots & Figure 15: Animated GIF
    # -------------------------------------------------------------------------
    frames_ex = np.asarray(data_dual["frames_ex"], dtype=float) * (float(data_dual["frames_ex_scale"]) if "frames_ex_scale" in data_dual else 1.0)
    times_fs = data_dual["times_fs"]
    mx_um = data_dual["mov_x_um"]
    my_um = data_dual["mov_y_um"]
    vmin_ex, vmax_ex = figstyle.symmetric_limits(frames_ex)

    snap_indices = [
        int(round(f * (len(frames_ex) - 1))) for f in (0.20, 0.50, 0.85)
    ]
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=12.8, height=4.2, sharey=True)
    for idx_p, (ax, idx_s) in enumerate(zip(axes, snap_indices)):
        im = ax.pcolormesh(
            mx_um,
            my_um,
            frames_ex[idx_s].T,
            cmap=figstyle.CMAP_DIVERGING,
            shading="auto",
            vmin=vmin_ex,
            vmax=vmax_ex,
        )
        overlay_rgb_geometry(
            ax,
            active_channel="G",
            line_color="black",
            show_dipoles=True,
            show_monitors=True,
            show_labels=True,
            compact=True,
        )
        ax.set_xlabel("Lateral coordinate $x$ [$\\mu$m]")
        ax.set_title(f"({'abc'[idx_p]}) 20-Dipole Burst at $t = {times_fs[idx_s]:.1f}$ fs")
    axes[0].set_ylabel("Vertical coordinate $y$ [$\\mu$m]")
    fig.colorbar(im, ax=axes[-1], pad=0.02, label="$E_x(x, y, t)$ [V/m]")
    figstyle.finish(fig)
    figstyle.save(fig, fig_dir / "fig14_time_domain_snapshots.png")

    # Animated GIF (Figure 15)
    fig_anim, ax_anim = figstyle.new_figure(nrows=1, ncols=1, width=7.2, height=4.1)
    quad = ax_anim.pcolormesh(
        mx_um,
        my_um,
        frames_ex[0].T,
        cmap=figstyle.CMAP_DIVERGING,
        shading="auto",
        vmin=vmin_ex,
        vmax=vmax_ex,
    )
    overlay_rgb_geometry(
        ax_anim,
        active_channel="G",
        line_color="black",
        show_dipoles=True,
        show_monitors=True,
        show_labels=True,
        compact=False,
    )
    ax_anim.set_xlabel("Lateral coordinate $x$ [$\\mu$m]")
    ax_anim.set_ylabel("Vertical coordinate $y$ [$\\mu$m]")
    title_anim = ax_anim.set_title(f"20-Dipole Random-Phase Burst ($t = {times_fs[0]:.1f}$ fs)")
    fig_anim.colorbar(quad, ax=ax_anim, label="$E_x(x, y, t)$ [V/m]")
    figstyle.finish(fig_anim)

    def _update_frame(i_f: int):
        quad.set_array(frames_ex[i_f].T.ravel())
        title_anim.set_text(f"20-Dipole Random-Phase Burst ($t = {times_fs[i_f]:.1f}$ fs)")
        return quad, title_anim

    ani = animation.FuncAnimation(
        fig_anim, _update_frame, frames=len(frames_ex), interval=90, blit=False
    )
    ani.save(fig_dir / "fig15_rgb_emission_movie.gif", writer="pillow", fps=11)
    plt.close(fig_anim)

    summary = {
        "eta_20_inc_pct": {ch: float(data_a[ch]["eta_20_inc_pct"]) for ch in CHANNELS},
        "eta_20_coh_pct": {ch: float(data_a[ch]["eta_20_coh_pct"]) for ch in CHANNELS},
        "lee_tot_pct": {ch: float(data_a[ch]["lee_tot_pct"]) for ch in CHANNELS},
        "c_win_pct": c_win,
        "c_mqw_pct": c_mqw,
        "c_dual_mean_pct": c_dual_mean,
        "conv_cpw": cpws,
        "conv_dx_nm": dxs_nm,
        "conv_eta_20_pct": etas,
        "conv_lee_tot_pct": lees,
        "conv_runtime_s": runtimes,
        "method_b_rse_empirical": stats_b["rse_empirical"],
        "method_b_rse_theory": stats_b["rse_theory"],
    }
    return summary


# %%
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_tasks:
        print("ID | Name          | Kind         | Description")
        print("---+---------------+--------------+------------------------------------------")
        for t in STUDY_TASKS:
            print(f"{t['id']:2d} | {t['name']:13s} | {t['kind']:12s} | {t}")
        return 0

    full = bool(args.full)
    if args.task is not None:
        run_task(args.task, args.out, full=full)
        return 0

    if not args.plot:
        for t in STUDY_TASKS:
            run_task(t["id"], args.out, full=full)

    summary = generate_all_figures_and_summary(args.out, args.fig_dir)
    print("\n=== Tutorial 7.4 Summary vs. Chen et al., Nanomaterials 15, 190 (2025) ===")
    for ch in CHANNELS:
        eta_w = summary["eta_20_inc_pct"][ch]
        eta_p = PAPER_TARGETS[ch]["eta_20_uncol_pct"]
        lee_w = summary["lee_tot_pct"][ch]
        lee_p = PAPER_TARGETS[ch]["lee_tot_uncol_pct"]
        print(
            f"  {ch} ({RGB_MATERIALS[ch]['wavelength_m']*1e9:.0f} nm): "
            f"eta_20 = {eta_w:.2f}% (paper: {eta_p:.2f}%, 2D Lamb: 34.20%), "
            f"LEE_tot = {lee_w:.2f}% (paper: {lee_p:.2f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# %% [markdown]
# ## Results & Comparison with Published Reference Values
#
# ### 1. Uncollimated 20-Dipole Incoherent Far-Field & Extraction Metrics (`--full` vs. `--quick` vs. Reference)
#
# | Channel ($\lambda_0$) | Wakis `--full` (20 cpw) $\eta_{\pm 20^\circ}$ (`--quick` 10 cpw) | Chen et al. (2025) $\eta_{\pm 20^\circ}$ (Uncollimated) | 2D Lambertian $\sin 20^\circ$ | Wakis Wrong Coherent $\eta_{\pm 20^\circ}^{\text{coh}}$ (`--full` / `--quick`) | Wakis `--full` $\text{LEE}_{\text{tot}}$ at $y=0.25\;\mu\text{m}$ (`--quick`) | Chen et al. (2025) $\text{LEE}_{\text{tot}}$ ($y=2.0\;\mu\text{m}$) | 2D Analytical Escape Cone Est. |
# | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | **Blue ($457\;\text{nm}$)** | **32.29%** (32.02%) | 30.29% (Fig. 4c) | 34.20% | 31.36% (15.53%) | **30.76%** (30.13%) | 26.63% (Sec. 3.2) | 18.80% |
# | **Green ($527\;\text{nm}$)** | **30.50%** (29.97%) | 34.84% (Fig. 4e) | 34.20% | 27.31% (44.65%) | **29.07%** (29.06%) | 26.53% (Sec. 3.2) | 19.80% |
# | **Red ($611\;\text{nm}$)** | **37.71%** (36.68%) | 35.83% (Fig. 4g) | 34.20% | 90.14% (90.73%) | **31.33%** (30.75%) | 23.98% (Sec. 3.2) | 20.11% |
#
# **Key observations:**
# 1. **Converged `--full` $\eta_{\pm 20^\circ}$ vs. Chen et al. (2025) and why 10 cpw was not converged for a single dipole:**
#    In the single-dipole convergence study (`task_05`..`task_09`), the `--quick` 10-cpw value of $\eta_{\pm 20^\circ} = 41.20\%$
#    was **not converged** ($15.20\%$ relative error compared to $35.77\%$ at the 28-cpw reference), whereas the `--full`
#    20-cpw grid is converged to within $0.99\%$ ($36.12\%$ vs. $35.77\%$ at 28 cpw). In the 20-dipole Method A ensemble
#    spanning the $5 \times 4 = 20$ MQW grid (`task_00`..`task_02`), spatial averaging across the 4 vertical rows and 5
#    lateral columns partially cancels single-dipole Fabry–Pérot standing-wave phase errors, so the 20-dipole incoherent
#    $\eta_{\pm 20^\circ}$ shifts by only $+0.27$ to $+1.03$ percentage points from `--quick` (10 cpw: $32.02\%$ B,
#    $29.97\%$ G, $36.68\%$ R) to the converged `--full` run (20 cpw: **$32.29\%$ B, $30.50\%$ G, $37.71\%$ R**).
#    These converged `--full` values agree with the uncollimated baselines of Chen et al. (2025) ($30.29\%$ B, $34.84\%$ G,
#    $35.83\%$ R) to within $1.88\text{--}4.34$ percentage points ($+2.00$ pp for B, $-4.34$ pp for G, $+1.88$ pp for R)
#    and bracket the 2D Lambertian analytical benchmark $\sin 20^\circ = 34.20\%$.
# 2. **Parameter dependence of the exact $\eta_{\pm 20^\circ}$ value (no parameter tuning):** Chen et al. (2025) state that
#    20 dipoles are placed inside the MQW of each LED channel (Section 2.4) but do not specify the exact $(x_m, y_m)$
#    coordinates, dipole orientations ($J_x$ vs. $J_y$), or near-to-far-field monitor width. Because the $200\;\text{nm}$
#    MQW in GaN ($n \approx 2.4$) spans an optical thickness $n t_{\text{MQW}} \approx 480\;\text{nm} \approx 1\,\lambda$
#    inside the vertical GaN/Si Fabry–Pérot cavity, individual dipole heights $y_m \in [-0.40, -0.20]\;\mu\text{m}$ yield
#    single-dipole $\eta_{\pm 20^\circ}$ values ranging from $\sim 20\%$ to $\sim 45\%$ (see the 15–85% single-dipole
#    bands in Figure 5 and cumulative curves in Figure 8). Our uniform $5 \times 4$ horizontal-$J_x$ dipole grid is held
#    fixed without tuning; sampling across the $200\;\text{nm}$ vertical thickness with $41$ cells ($\Delta y_{\min} = 4.88\;\text{nm}$
#    in `--full`, $21$ cells in `--quick`) averages smoothly over the vertical interference fringe.
# 3. **Non-convergence and distortion of the wrong coherent field sum ($|\sum_m \mathbf{E}_m|^2$):** Summing complex field
#    phasors coherently produces an unphysical phased-array pattern (Figure 4) with a sharp main lobe at $\theta = 0^\circ$
#    nearly $2\times\text{--}4\times$ the physical incoherent peak density ($0.0161$ vs. $0.0087\;\text{deg}^{-1}$ for B,
#    $0.0161$ vs. $0.0086\;\text{deg}^{-1}$ for G, and $0.0372$ vs. $0.0095\;\text{deg}^{-1}$ for R in `--full`), giving a
#    $63.1\%\text{--}139.9\%$ relative $L_2$ profile error across B, G, and R. Moreover, whereas the physical incoherent
#    sum changes by $\le 1.03$ percentage points between `--quick` (10 cpw) and `--full` (20 cpw), the wrong coherent
#    cone fraction $\eta_{\pm 20^\circ}^{\text{coh}}$ swings wildly from $15.53\% \to 31.36\%$ (B) and $44.65\% \to 27.31\%$
#    (G) while staying at $90.73\% \to 90.14\%$ (R), because coherent cancellation across the 4 vertical dipole rows is
#    hypersensitive to sub-cell Yee-grid dipole snapping.
# 4. **Near-field vs. top-of-collimator monitor height for $\text{LEE}_{\text{tot}}$:** Immediately above the
#    pixel surface ($y = 0.25\;\mu\text{m}$, $W_{\text{N2F}} = 5.20\;\mu\text{m}$), the `--full` upward extraction efficiency
#    is **$30.76\%$ (B), $29.07\%$ (G), and $31.33\%$ (R)** (exceeding the planar 2D escape-cone estimate of $18.80\%\text{--}20.11\%$
#    due to lateral extraction through the $1.10\;\mu\text{m}$ mesa sidewalls into the $n_{\text{PI}} = 1.50$ polyimide trenches).
#    When evaluated over a narrower single-subpixel aperture or at the top of the $2.0\;\mu\text{m}$ collimator volume
#    (as in Section 3.2 of Chen et al. 2025 and our single-subpixel convergence tasks), high-angle grazing rays exit
#    laterally before reaching the top monitor, giving $\text{LEE}_{\text{tot}} = 26.00\%$ at `--full` 20 cpw and $25.86\%$
#    at the 28-cpw reference (vs. $26.53\%$ in Chen et al. 2025).
#
# ### 2. Method A vs. Method B Incoherent Averaging Trade-off
#
# | Realizations $K$ | Theoretical $\text{RSE} = \sqrt{(1 - 1/20)/K}$ | Empirical MC $\text{RSE}$ (`--full`) | Wakis `--full` FDTD $\eta_{\pm 20^\circ}^{(K)}$ (G) (`--quick`) | Proper Ensemble Ratio Bias [% pts] (`--full` / `--quick`) | Naive Ratio Bias [% pts] (`--full`) | FDTD Cost vs. Method A ($N=20$) |
# | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | **$K = 1$** | 97.47% | 100.6% | **36.29%** (32.47%) | $+2.04\%$ ($+2.24\%$) | $+2.04\%$ | $0.05\times$ ($20\times$ faster) |
# | **$K = 2$** | 68.92% | 68.2% | **22.99%** (21.19%) | $+0.97\%$ ($+1.09\%$) | $+1.30\%$ | $0.10\times$ ($10\times$ faster) |
# | **$K = 4$** | 48.73% | 48.1% | **25.01%** (24.24%) | $+0.86\%$ ($+0.91\%$) | $+1.28\%$ | $0.20\times$ ($5\times$ faster) |
# | **$K = 8$** | 34.46% | 34.1% | **27.83%** (27.78%) | $+0.04\%$ ($-0.04\%$) | $+0.55\%$ | $0.40\times$ ($2.5\times$ faster) |
# | **$K = 16$** | 24.37% | 24.4% | **26.85%** (—) | $-0.17\%$ ($-0.18\%$) | $+0.31\%$ | $0.80\times$ ($1.25\times$ faster; $548\;\text{s}$ vs. $866\;\text{s}$) |
# | **$K = 32$** | 17.23% | 17.4% | — | $-0.23\%$ ($-0.20\%$) | $+0.47\%$ | $1.60\times$ (slower than Method A) |
# | **$K = 64$** | 12.18% | 11.9% | — | $+0.24\%$ ($+0.25\%$) | $+0.91\%$ | $3.20\times$ (slower than Method A) |
#
# - **Cost crossover rule:** Method B costs $K$ FDTD runs independent of the number of dipoles $N$, whereas
#   Method A costs $N$ runs with **zero Monte Carlo speckle noise**. For $N = 20$ dipoles in 2D, Method A requires
#   only 20 runs and is exact; for 3D micro-LED volumes with $N \sim 10^3\text{--}10^4$ dipoles across position,
#   orientation, and spectrum, Method B with $K \approx 16\text{--}32$ realizations achieves $\sim 2\text{--}3.8\%$
#   uncertainty on integrated cone metrics at $< 1\%$ of Method A's cost.
#
# ### 3. Inter-Pixel Crosstalk ($C_{ij}^{\text{emit}}$) & Dual-Task Routing ($C_{ij}^{\text{dual}}$)
#
# - **Forward exit-window crosstalk ($C_{ij}^{\text{win}}$ at $y = 0.25\;\mu\text{m}$):** Under 20-dipole incoherent
#   excitation in `--full` (20 cpw), **$95.05\%$ (B), $90.83\%$ (G), and $95.26\%$ (R)** of the upward flux crosses the
#   active subpixel's own $1.50\;\mu\text{m}$ exit window (vs. $95.27\%$, $90.79\%$, and $95.31\%$ in `--quick`), while
#   **$4.58\%\text{--}4.91\%$ spills into each adjacent subpixel window** ($4.91\%$ B$\to$G, $4.59\%$ G$\to$B, $4.58\%$
#   G$\to$R, $4.73\%$ R$\to$G; $9.17\%$ total lateral spillover for the central G channel already at $y = 250\;\text{nm}$,
#   and $0.01\%\text{--}0.04\%$ into the second-nearest subpixel).
# - **MQW lateral guided crosstalk ($C_{ij}^{\text{mqw}}$):** The $0.45\;\mu\text{m}$ polyimide isolation trenches
#   ($n_{\text{PI}} = 1.50$ vs. $n_{\text{GaN}} \approx 2.4$, resolved with $50$ horizontal cells in `--full` and $25$
#   cells in `--quick`) reflect most guided slab modes, keeping self-MQW intensity fractions at **$98.27\%$ (B), $95.95\%$ (G),
#   and $98.07\%$ (R)** in `--full` (vs. $98.34\%$, $96.92\%$, $98.52\%$ in `--quick`) and nearest-neighbor MQW-to-MQW
#   intensity crosstalk down to **$1.63\%\text{--}2.04\%$** per adjacent channel ($1.63\%$ B$\to$G, $2.04\%$ G$\to$B,
#   $2.01\%$ G$\to$R, $1.77\%$ R$\to$G; $4.05\%$ total for G in `--full` vs. $1.37\%\text{--}1.57\%$ / $3.08\%$ total in `--quick`).
# - **Reciprocal dual-task routing ($C_{ij}^{\text{dual}}$, Eq. 4):** Without the inverse-designed $\text{SiN}_x/\text{SiO}_2$
#   color router, plane waves incident at $\theta_{\text{inc}} \in [-15^\circ, +15^\circ]$ illuminate all three
#   subpixels nearly equally: in `--full` (20 cpw), **$\text{OE}(\text{B}) = 33.43\%$, $\text{OE}(\text{G}) = 33.57\%$,
#   $\text{OE}(\text{R}) = 33.27\%$** with off-diagonal crosstalk **$\text{OX}(\text{B}) = 66.57\%$, $\text{OX}(\text{G}) = 66.43\%$,
#   $\text{OX}(\text{R}) = 66.73\%$** (vs. $\text{OE} = 33.49\%, 33.57\%, 33.35\%$ in `--quick`), matching the unoptimized
#   Iteration-0 baseline ($\approx 33.3\%$) of Chen et al. (2025) Figure 3b,d (which inverse design boosts to $79.5\%$,
#   $81.0\%$, and $79.8\%$).
#
# ### 4. Interface-Snapped Grid & Silicon Skin-Depth Convergence Study ($\text{cpw} \in \{10, 12, 16, 20, 28\}$)
#
# Lossy Silicon is dispersive across the visible band ($\tilde{n}_{\text{Si}} = 4.583 + 0.130i$ at $457\;\text{nm}$,
# $4.215 + 0.048i$ at $527\;\text{nm}$, $3.906 + 0.022i$ at $611\;\text{nm}$). We track both the **power skin depth**
# $\delta_P = \lambda / (4\pi\kappa)$ and the **$1/e$ field skin depth** $\delta_E = \lambda / (2\pi\kappa) = 2\delta_P$.
# Because the $1.10\;\mu\text{m}$ GaN/MQW mesa forms a vertical Fabry–Pérot cavity between the top GaN/air interface
# ($y = 0$) and the bottom GaN/Si interface ($y = -1.10\;\mu\text{m}$), an unsnapped uniform grid shifts the cavity
# interfaces by up to $\pm 1$ cell as $\text{cpw}$ varies, inducing spurious $\sim 7\%$ standing-wave oscillations in
# $\eta_{\pm 20^\circ}$. Using `wakis.templates.meshing.graded_axis` to snap exact mesh nodes to all horizontal layer
# interfaces ($y = -1.10, -0.40, -0.20, 0.00\;\mu\text{m}$) and vertical mesa edges ($x_c \pm 0.525\;\mu\text{m}$),
# while refining vertically to $\Delta y_{\min} = \lambda_{\min} / (\text{cpw} \cdot n_{\max,\text{Si}})$ inside the
# $200\;\text{nm}$ MQW, $200\;\text{nm}$ p-GaN, and Silicon substrate, eliminates interface staircasing and yields
# monotonic convergence:
#
# | $\text{cpw}$ | $\Delta x_{\text{bg}}$ [nm] | $\Delta y_{\min}$ [nm] | Grid $(N_x \times N_y)$ | Cells / $200\;\text{nm}$ Layer (Trench) | Cells / Si $\delta_P$ ($\delta_E$) at $527\;\text{nm}$ | $\eta_{\pm 20^\circ}$ [%] | $\text{LEE}_{\text{tot}}$ [%] | Rel. Error vs. $\text{cpw}=28$ | Runtime [s] |
# | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | **10 (`--quick`)** | 18.10 | 9.52 | $173 \times 133$ ($23{,}009$) | 21 (25) | 91.3 (182.7) | 41.20% | 25.58% | $15.20\%$ / $1.06\%$ | 1.77 |
# | **12** | 15.17 | 8.00 | $207 \times 158$ ($32{,}706$) | 25 (30) | 107.2 (214.5) | 40.00% | 25.56% | $11.83\%$ / $1.14\%$ | 3.40 |
# | **16** | 11.29 | 6.06 | $276 \times 209$ ($57{,}684$) | 33 (40) | 143.0 (285.9) | 36.66% | 26.06% | $2.49\%$ / $0.79\%$ | 7.84 |
# | **20 (`--full`, accepted)** | 9.08 | 4.88 | $345 \times 260$ ($89{,}700$) | 41 (50) | 178.7 (357.4) | **36.12%** | **26.00%** | **$0.99\%$ / $0.55\%$** | 14.09 |
# | **28 (ref)** | 6.48 | 3.51 | $481 \times 361$ ($173{,}641$) | 57 (70) | 246.2 (492.4) | 35.77% | 25.86% | $0.00\%$ / $0.00\%$ | 44.17 |
#
# - **Convergence verdict at the accepted `--full` level ($\text{cpw} = 20$ vs. $\text{cpw} = 28$):** At `--quick`
#   ($\text{cpw} = 10$), the single-dipole $\eta_{\pm 20^\circ} = 41.20\%$ was not converged ($15.20\%$ relative error vs.
#   the 28-cpw reference). Between the last two levels — the accepted `--full` level ($\text{cpw} = 20$, $\Delta y_{\min} = 4.88\;\text{nm}$,
#   $178.7$ cells per power skin depth $\delta_P$) and the $\text{cpw} = 28$ reference ($\Delta y_{\min} = 3.51\;\text{nm}$,
#   $246.2$ cells per $\delta_P$) — $\eta_{\pm 20^\circ}$ changes by only $|36.1227\% - 35.7683\%| / 35.7683\% = \mathbf{0.99\% < 1\%}$
#   ($0.35$ percentage points) and $\text{LEE}_{\text{tot}}$ changes by only $|25.9980\% - 25.8560\%| / 25.8560\% = \mathbf{0.55\% < 1\%}$
#   ($0.14$ percentage points).
#
# ### 5. Complete Task Resolution & Preflight Summary (`--full` Measured vs. `--quick`)
#
# Every run in `data/` satisfies all project resolution requirements with zero violations:
# 1. **Cells per wavelength in highest-index materials ($\ge 10\;\text{cpw}$):** `--full` tasks (`task_00`..`task_04`) run at
#    $\text{cpw} = 20.0$ ($20.06\;\text{cpw}$ in Blue MQW $n = 2.51$ at $\Delta x_{\text{bg}} = 9.08\;\text{nm}$, and
#    $20.44\;\text{cpw}$ in Blue Silicon $n = 4.583$ at $\Delta y_{\min} = 4.88\;\text{nm}$; $24.10\text{--}28.41\;\text{cpw}$
#    in G/R MQW and $25.63\text{--}32.07\;\text{cpw}$ in G/R Silicon). Even the coarsest convergence task (`task_05`,
#    $\text{cpw} = 10.0$) achieves $10.06\;\text{cpw}$ in Blue MQW and $10.47\;\text{cpw}$ in Blue Silicon.
# 2. **Cells across smallest geometric features ($\ge 4$ cells):** `--full` tasks resolve the $200\;\text{nm}$ MQW and p-GaN
#    layers with **$41$ vertical cells** and the $450\;\text{nm}$ polyimide trenches with **$50$ horizontal cells** ($21$ and
#    $25$ cells in `--quick`).
# 3. **Cells per Silicon skin depth ($\ge 4$ cells):** `--full` tasks resolve the power skin depth $\delta_P = \lambda/(4\pi\kappa)$
#    with **$56.9$ cells (B, $\delta_P = 279.7\;\text{nm}$), $177.9$ cells (G, $\delta_P = 873.7\;\text{nm}$), and $449.9$ cells
#    (R, $\delta_P = 2210.1\;\text{nm}$)** ($113.9$, $355.7$, and $899.8$ cells per $1/e$ field skin depth $\delta_E = 2\delta_P$).
#
# | Task ID & Name | `--full` $\text{cpw}$ (`--quick`) | `--full` $\Delta x_{\text{bg}}$ (`--quick`) [nm] | `--full` $\Delta y_{\min}$ (`--quick`) [nm] | Min Feature $200\;\text{nm}$ Cells (`--full` / `--quick`) | Si $\delta_P$ ($\delta_E$) [nm] | `--full` Cells / $\delta_P$ ($\delta_E$) (`--quick` $\delta_P$) | `--full` Grid $N_{\text{cells}}$ ($N_x \times N_y$) (`--quick`) | `--full` Steps $\times$ Runs (`--quick`) | `--full` Time [s] (`--quick`) |
# | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | `0: method_a_B` ($457\;\text{nm}$) | $20$ ($10$) | $9.08$ ($18.10$) | $4.88$ ($9.52$) | $41$ ($21$) | $279.7$ ($559.5$) | $56.9$ ($113.9$) [$29.0$] | $178{,}573$ ($631\times 283$) [$45{,}965$] | $1936\times 20$ ($987\times 20$) | **788.57 s** ($118.05\;\text{s}$) |
# | `1: method_a_G` ($527\;\text{nm}$) | $20$ ($10$) | $9.08$ ($18.10$) | $4.88$ ($9.52$) | $41$ ($21$) | $873.7$ ($1747.4$) | $177.9$ ($355.7$) [$90.5$] | $178{,}573$ ($631\times 283$) [$45{,}965$] | $2006\times 20$ ($1022\times 20$) | **865.99 s** ($120.49\;\text{s}$) |
# | `2: method_a_R` ($611\;\text{nm}$) | $20$ ($10$) | $9.08$ ($18.10$) | $4.88$ ($9.52$) | $41$ ($21$) | $2210.1$ ($4420.2$) | $449.9$ ($899.8$) [$228.9$] | $178{,}573$ ($631\times 283$) [$45{,}965$] | $2090\times 20$ ($1065\times 20$) | **883.97 s** ($127.99\;\text{s}$) |
# | `3: method_b_G` ($527\;\text{nm}$) | $20$ ($10$) | $9.08$ ($18.10$) | $4.88$ ($9.52$) | $41$ ($21$) | $873.7$ ($1747.4$) | $177.9$ ($355.7$) [$90.5$] | $178{,}573$ ($631\times 283$) [$45{,}965$] | $2006\times 16$ ($1022\times 8$) | **548.00 s** ($41.32\;\text{s}$) |
# | `4: dual_routing` (RGB) | $20$ ($10$) | $9.08$ ($18.10$) | $4.88$ ($9.52$) | $41$ ($21$) | $279.7\text{--}2210.1$ | $56.9\text{--}449.9$ [$29.0\text{--}228.9$] | $178{,}573$ ($631\times 283$) [$45{,}965$] | $1936\text{--}2090\times 16$ ($987\text{--}1065\times 16$) | **792.67 s** ($115.21\;\text{s}$) |
# | `5..9: conv_cpw10..28` ($527\;\text{nm}$) | $10..28$ | $18.10..6.48$ | $9.52..3.51$ | $21..57$ | $873.7$ ($1747.4$) | $91.3..246.2$ ($182.7..492.4$) | $23{,}009..173{,}641$ | $938..2564\times 1$ | **71.27 s** total |
#
# ---
#
# ## Limits & Failure Modes
#
# 1. **Coherent Phasing Fallacy:** Driving all 20 dipoles in phase in a single simulation ($K = 1, \phi_m = 0$)
#    turns the micro-LED into a phased-array antenna, creating artificial grating lobes with $63.1\%\text{--}139.9\%$
#    relative $L_2$ angular profile error and distorting $\eta_{\pm 20^\circ}$ by $-0.93$ to $+52.44$ percentage points in
#    `--full` ($-16.49$ to $+54.05$ percentage points in `--quick`) across B, G, and R (Figure 4).
# 2. **2D vs. 3D Escape Cone & Étendue Scaling:** Native 2D FDTD models a line-extruded pixel array along $z$.
#    In 2D, the Lambertian $\pm 20^\circ$ cone fraction is $\sin 20^\circ = 34.20\%$ and the single-interface escape
#    cone is $\theta_c / \pi \approx 13.5\%$; in 3D, solid-angle integration squares the sine ($\sin^2 20^\circ = 11.70\%$)
#    and reduces the planar escape cone to $\frac{1}{2}(1 - \cos\theta_c) \approx 4.4\%$. Never compare 2D angular
#    power ratios directly against 3D solid-angle efficiency formulas without the $\sin\theta$ Jacobian.
# 3. **Single-Subpixel Aperture Bottleneck:** Passive collimation is bounded by étendue conservation
#    $\eta_{\max}(\pm\theta_0) \le (W_{\text{ap}}/W_{\text{ch}})\sin\theta_0$. A $1.05\;\mu\text{m}$ emitter cannot be
#    collimated above $48.9\%$ within $\pm 20^\circ$ by any structure confined to a single $1.50\;\mu\text{m}$
#    subpixel pitch; cross-pixel wavelength multiplexing across the $4.50\;\mu\text{m}$ unit cell is mandatory.
# 4. **Monochromatic vs. Finite LED Linewidth:** Chen et al. (2025) evaluate each channel at its center wavelength
#    ($457\;\text{nm}$, $527\;\text{nm}$, $611\;\text{nm}$). Real InGaN/AlGaInP micro-LEDs have spectral FWHMs of
#    $\Delta\lambda \approx 20\text{--}40\;\text{nm}$, which smooths high-$Q$ resonant features in nanostructured
#    collimators.
#
# ---
#
# ## Exercises
#
# 1. **In-Plane Polarization Mixing ($E_x$ vs. $E_y$):** Modify `run_single_channel_method_a` in `_helpers.py`
#    to include both horizontal ($J_x$, heavy-hole TE-like) and vertical ($J_y$, TM-like) dipole orientations
#    with a $3:1$ power weight. How much does $J_y$ increase lateral waveguide coupling $C_{ij}^{\text{mqw}}$ across
#    the polyimide trenches?
# 2. **Polyimide Trench Index & Width Sweep:** Replace the polyimide filler ($n_{\text{PI}} = 1.50$) with air trenches
#    ($n = 1.0$) or $\text{SiO}_2$ ($n \approx 1.46$) and vary $W_{\text{gap}}$ from $0.15\;\mu\text{m}$ to $0.75\;\mu\text{m}$.
#    Plot the trade-off between MQW lateral crosstalk $C_{ij}^{\text{mqw}}$ and the étendue limit $(P_{\text{sub}}/W_{\text{ch}})\sin 20^\circ$.
# 3. **Dual-Task Reciprocity Verification:** Using `build_rgb_array_solver(..., collimator_domes=True)`, compare
#    the change in forward $\eta_{\pm 20^\circ}$ against the change in backward plane-wave focusing $c_{i,i}(\theta_{\text{inc}})$
#    for $|\theta_{\text{inc}}| \le 15^\circ$.
#
# ---
#
# ## References
#
# 1. L. Chen, Z. Li, P. Wang, S. Wu, W. Li, J. Wang, Y. Cao, M. Mortazavi, L. Peng, and P. Wu,
#    "Dual-Task Optimization Method for Inverse Design of RGB Micro-LED Light Collimator",
#    *Nanomaterials* **15**(3), 190 (2025), [doi:10.3390/nano15030190](https://doi.org/10.3390/nano15030190).
# 2. E. Chen et al., "Broadband Beam Collimation Metasurface for Full-Color Micro-LED Displays",
#    *Optics Express* **32**(6), 10252–10264 (2024), [doi:10.1364/OE.518535](https://doi.org/10.1364/OE.518535).
# 3. M. Catrysse, N. Zhao, W. Jin, and S. Fan, "Subwavelength Bayes-type color router for CMOS image sensors",
#    *Optics Letters* **47**(13), 3307–3310 (2022).
# 4. E. D. Palik (ed.), *Handbook of Optical Constants of Solids*, Academic Press, Boston (1985)
#    (source of Table 1 refractive indices in Chen et al. 2025).
