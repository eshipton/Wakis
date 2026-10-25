| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 3.44 µm × 2.80 µm × 10 nm (z ∈ [-5 nm, 5 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 344 × 280 × 1 (96320 cells) |
| Discretization | Δx & resolution | Δ_min = 10 nm, 36.1 cells/λ (in n_max = 3.50), 8.0 cells across feature (80 nm) |
| Boundary | -x face | CPML (20 cells, 200 nm) |
| Boundary | +x face | CPML (20 cells, 200 nm) |
| Boundary | -y face | CPML (20 cells, 200 nm) |
| Boundary | +y face | CPML (20 cells, 200 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | SiO₂ cladding (ε = 2.25, n = 1.50) (isotropic) |
| Material | Region 1: Input Si waveguide (ε = 12.25, n = 3.50, w = 400 nm) | ε_r = 12.25, μ_r = 1.00, model = Isotropic dielectric, x ∈ [5 nm, 920 nm] (d = 915 nm) |
| Material | Region 2: Upper output Si waveguide (ε = 12.25, n = 3.50, w = 400 nm) | ε_r = 12.25, μ_r = 1.00, model = Isotropic dielectric, x ∈ [2.52 µm, 3.44 µm] (d = 915 nm) |
| Material | Region 3: Lower output Si waveguide (ε = 12.25, n = 3.50, w = 400 nm) | ε_r = 12.25, μ_r = 1.00, model = Isotropic dielectric, x ∈ [2.52 µm, 3.44 µm] (d = 915 nm) |
| Material | Region 4: Design region Ω_d (160×160 px, 1.60 µm × 1.60 µm, ε ∈ {2.25, 12.25}) | ε_r = 12.25, μ_r = 1.00, model = Isotropic dielectric, x ∈ [920 nm, 2.52 µm] (d = 1.60 µm) |
| Source | 1 · PolarizedModeSource | pos: x = 0.30 µm, y ∈ [0.68, 2.12] µm, pol: TM₀ (E_z), pulse: f₀ = 234.2 THz, Δf = 14.05 THz |
| Monitor | 2 · PolarizedModeMonitor | Port 1 PolarizedModeMonitor (S₁₁ reflection, TM₀ E_z), pos: x = 0.25 µm, y ∈ [0.68, 2.12] µm, freqs: 6 target channels λ ∈ {1265..1295} nm (67 dense λ ∈ [1245, 1315] nm) |
| Monitor | 3 · PolarizedModeMonitor | Port 2 Lower-Arm Objective PolarizedModeMonitor (S₂₁ transmission, TM₀ E_z), pos: x = 3.21 µm, y ∈ [0.20, 1.60] µm, freqs: 6 target channels λ ∈ {1265..1295} nm (67 dense λ ∈ [1245, 1315] nm) |
| Monitor | 4 · PolarizedModeMonitor | Port 3 Upper-Arm Objective PolarizedModeMonitor (S₃₁ transmission, TM₀ E_z), pos: x = 3.21 µm, y ∈ [1.20, 2.60] µm, freqs: 6 target channels λ ∈ {1265..1295} nm (67 dense λ ∈ [1245, 1315] nm) |
| Variation | Splitter initial (_initial) | Stage-1 iter 0 initial density (mean ρ ≈ 0.49) in 160 × 160 design box Ω_d (1.60 µm × 1.60 µm) |
| Variation | Splitter final (_final) | Stage-2 STE binary design (ste_beta64:11, 8/8 px = 80 nm, official eval_metric = +0.0682) from round2_splitter/design.npy |
| Variation | Mode converter (_mode_converter_*) | TM₀ → TM₁ spatial mode converter (official eval_metric = +0.3122, 8/8 px = 80 nm) from round2_mode_converter/design.npy |
