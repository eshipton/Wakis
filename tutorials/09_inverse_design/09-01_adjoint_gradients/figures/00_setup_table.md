| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 157 mm × 157 mm × 35.7 mm (z ∈ [-17.8 mm, 17.8 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 44 × 44 × 1 (1936 cells) |
| Discretization | Δx & resolution | Δ_min = 3.57 mm, 11.4 cells/λ (in n_max = 2.45), 6.0 cells across feature (21.4 mm) |
| Boundary | -x face | CPML (8 cells, 28.6 mm) |
| Boundary | +x face | CPML (8 cells, 28.6 mm) |
| Boundary | -y face | CPML (8 cells, 28.6 mm) |
| Boundary | +y face | CPML (8 cells, 28.6 mm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Vacuum cladding (ε = 1.00, n = 1.00) (isotropic) |
| Material | Region 1: Input waveguide (ε = 6.00, n = 2.45) | ε_r = 6.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-78.5 mm, -28.6 mm] (d = 50 mm) |
| Material | Region 2: Output waveguide (ε = 6.00, n = 2.45) | ε_r = 6.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-10.7 mm, 10.7 mm] (d = 21.4 mm) |
| Material | Region 3: Design region Ω_d (16×16 cells, ε ∈ [1.00, 6.00]) | ε_r = 6.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-28.6 mm, 28.6 mm] (d = 57.1 mm) |
| Source | 1 · CurrentSource | pos: x = -3.93 cm, y ∈ [-1.07, 1.07] cm, pol: E_z (TM_z, J_z), pulse: f₀ = 3 GHz, Δf = 2 GHz |
| Monitor | 2 · DFTMonitor | Objective DFTMonitor (guided output E_z), pos: x ∈ [-1.07, 1.07] cm, y = 3.93 cm, freqs: 2.55–3.45 GHz (11 freqs, f₀ = 3.00 GHz) |
| Variation | Initial design (_initial) | Uniform greyscale ρ = 0.50 (ε_r = 3.50) across 16 × 16 design box Ω_d (5.71 cm × 5.71 cm) |
| Variation | Final design (_final) | β-continuation optimized 90° bend (β = 2 → 4 → 8, ε_r ∈ [1.00, 6.00]) stored in data/task_4_optimization.npz |
| Variation | Memory ladder (task_3) | store (float64/float32), checkpoint (revolve DP), reverse (CPML strip recording), nyquist (decimation d ∈ {4, 8, 16}) |
