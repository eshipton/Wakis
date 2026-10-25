| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 9.40 µm × 8.20 µm × 250 nm (z ∈ [-125 nm, 125 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 581 × 443 × 1 (257383 cells) |
| Discretization | Δx & resolution | Δ_min = 2.86 nm, 91.0 cells/λ (in n_max = 1.92), 10.1 cells across feature (28.8 nm) |
| Boundary | -x face | CPML (8 cells, 199.3 nm) |
| Boundary | +x face | CPML (8 cells, 199.3 nm) |
| Boundary | -y face | CPML (8 cells, 197.8 nm) |
| Boundary | +y face | CPML (8 cells, 199.3 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Vacuum / Air (n = 1.00), ε = 1.00 (isotropic) |
| Material | Region 1: Drude Ag screen (δ_E = 28.8 nm) | ε_r = 3.70, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-4.70 µm, -1.20 µm] (d = 3.50 µm) |
| Material | Region 2: Drude Ag screen (δ_E = 28.8 nm) | ε_r = 3.70, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-800 nm, 800 nm] (d = 1.60 µm) |
| Material | Region 3: Drude Ag screen (δ_E = 28.8 nm) | ε_r = 3.70, μ_r = 1.00, model = DispersiveMaterial, x ∈ [1.20 µm, 4.70 µm] (d = 3.50 µm) |
| Source | 1 · CurrentSource | pos: y = -1.00 µm, x ∈ [-4.00, +4.00] µm (Hermite-Gaussian w_m = 1.40 µm), pol: J_z (TM), pulse: f₀ = 599.6 THz, Δf = 131.9 THz |
| Monitor | 2 · DFTMonitor | Slit-plane DFTMonitor (y = 0.00 µm; modal amplitude & broadband transfer normalization), pos: y = 0.00 µm, x ∈ [-4.20, +4.20] µm (81 broadband freqs in 0.72–1.28 f₀), freqs: f = 599.6 THz |
| Monitor | 3 · DFTMonitor | Downstream detector DFTMonitor (y = +6.00 µm; GSM modal & Wiener-Khinchin polychromatic fringes), pos: y = +6.00 µm, x ∈ [-4.20, +4.20] µm (81 broadband freqs in 0.72–1.28 f₀), freqs: f = 599.6 THz |
| Variation | Tasks 2–3 — Method B & C GSM Mercer modes (`gsm_modes_even`, `gsm_modes_odd`) | notes=M = 12 Hermite-Gaussian modes (m = 0..11, w_m = 1.40 µm) + 81-frequency broadband DFT at cpw = 26 |
| Variation | Task 4 — Dispersive Drude Ag screen (`silver_dispersive`) | notes=Rakic Ag at 500 nm (ε = -7.632 - 0.731i, δ_E = 28.77 nm) with Δ_min = 2.86 nm (10.07 cells/δ_E) |
| Variation | Task 5 — Ag skin-depth resolution ladder (`silver_skin_depth_demo`, demo=True) | notes=Under-resolved uniform (0.58, 1.15 cells/δ_E) vs. graded collar (2, 4, 6, 8, 10 cells/δ_E) |
