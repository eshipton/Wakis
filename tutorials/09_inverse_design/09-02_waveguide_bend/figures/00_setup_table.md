| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 3.12 µm × 3.12 µm × 10 nm (z ∈ [-5 nm, 5 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 312 × 312 × 1 (97344 cells) |
| Discretization | Δx & resolution | Δ_min = 10 nm, 36.1 cells/λ (in n_max = 3.50), 8.0 cells across feature (80 nm) |
| Boundary | -x face | CPML (20 cells, 200 nm) |
| Boundary | +x face | CPML (20 cells, 200 nm) |
| Boundary | -y face | CPML (20 cells, 200 nm) |
| Boundary | +y face | CPML (20 cells, 200 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | SiO₂ cladding (ε = 2.25, n = 1.50) (isotropic) |
| Material | Region 1: Input Si waveguide (ε = 12.25, n = 3.50, w = 400 nm) | ε_r = 12.25, μ_r = 1.00, model = Isotropic dielectric, x ∈ [5 nm, 920 nm] (d = 915 nm) |
| Material | Region 2: Output Si waveguide (ε = 12.25, n = 3.50, w = 400 nm) | ε_r = 12.25, μ_r = 1.00, model = Isotropic dielectric, x ∈ [1.52 µm, 1.92 µm] (d = 400 nm) |
| Material | Region 3: Design region Ω_d (160×160 px, 1.60 µm × 1.60 µm, ε ∈ {2.25, 12.25}) | ε_r = 12.25, μ_r = 1.00, model = Isotropic dielectric, x ∈ [920 nm, 2.52 µm] (d = 1.60 µm) |
| Source | 1 · PolarizedModeSource | pos: x = 0.25 µm, y ∈ [1.00, 2.44] µm, pol: TM₀ (E_z), pulse: f₀ = 234.4 THz, Δf = 28.13 THz |
| Monitor | 2 · PolarizedModeMonitor | Port 1 PolarizedModeMonitor (S₁₁ reflection, TM₀ E_z), pos: x = 0.30 µm, y ∈ [1.00, 2.44] µm, freqs: 6 target channels λ ∈ {1265..1295} nm (67 dense λ ∈ [1245, 1315] nm) |
| Monitor | 3 · PolarizedModeMonitor | Port 2 Objective PolarizedModeMonitor (S₂₁ transmission, TM₀ E_z), pos: x ∈ [1.00, 2.44] µm, y = 0.30 µm, freqs: 6 target channels λ ∈ {1265..1295} nm (67 dense λ ∈ [1245, 1315] nm) |
| Variation | Initial design (_initial) | Stage-1 iter 0 initial greyscale density (mean ρ ≈ 0.49, β = 4) in 160 × 160 design box Ω_d (1.60 µm × 1.60 µm) |
| Variation | Final Round-2 design (_final) | Stage-2 STE binary design (ste_beta64:34, 8/8 px = 80 nm, official eval_metric = +0.6427) from data/round2_bend/design.npy |
| Variation | Quick re-simulation (task_1) | 2× coarsened 20 nm grid (156 × 156 × 1, 18.1 cells/λ_min) vs stored 10 nm reference |
