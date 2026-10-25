| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 2.26 µm × 28.2 nm × 1.20 µm (z ∈ [-547 nm, 657 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 240 × 3 × 109 (78480 cells) |
| Discretization | Δx & resolution | Δ_min = 8.69 nm, 21.7 cells/λ (in n_max = 2.39), 5.4 cells across feature (47.1 nm) |
| Boundary | -x face | CPML (8 cells, 75.2 nm) |
| Boundary | +x face | CPML (8 cells, 75.2 nm) |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (8 cells, 75.3 nm) |
| Boundary | +z face | CPML (8 cells, 168.6 nm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Bottom NP-GaN/GaN DBR | ε_r = 2.89, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.13 µm, 1.13 µm] (d = 2.26 µm) |
| Material | Region 2: 1λ GaN cavity | ε_r = 5.71, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.13 µm, 1.13 µm] (d = 2.26 µm) |
| Material | Region 3: Top Ta₂O₅/SiO₂ DBR | ε_r = 4.62, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.13 µm, 1.13 µm] (d = 2.26 µm) |
| Source | 1 · CurrentSource | pos: x = 0 m, z = 94.1 nm, pol: J_y (E_y), pulse: f₀ = 666.2 THz, Δf = 186.5 THz |
| Monitor | 2 · FluxMonitor | Dipole FluxMonitor box, pos: x ∈ [-18.8, 18.8] nm, z ∈ [56.5, 131.8] nm, freqs: f = 666.2 THz |
| Monitor | 3 · FluxMonitor | FluxMonitor (z = 477 nm), pos: z = 477 nm, freqs: f = 666.2 THz |
| Variation | Cavity detuning & angular beaming | param=d_opt ∈ {1.00 (tuned), 1.055 (red-detuned rabbit ears)}, note=Bottom NP-GaN/GaN DBR (n = 1.70/2.39) + GaN cavity + top SiO₂/Ta₂O₅ DBR (n = 1.46/2.15) |
| Variation | Incoherent quantum-well array | param=N_dip = 7 emitters across w_QW = 0.72 µm, N_real ∈ [1, 128], note=Deterministic power sum vs random-phase Monte Carlo O(N_real^-1/2) convergence |
