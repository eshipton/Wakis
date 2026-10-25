| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 90 mm × 90 mm × 96 mm (z ∈ [-48 mm, 48 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 30 × 30 × 32 (28800 cells) |
| Discretization | Δz & resolution | Δ_min = 3 mm, 38.8 cells/λ (in n_max = 1.00), 4.0 cells across feature (12 mm) |
| Boundary | -x face | PEC |
| Boundary | +x face | PEC |
| Boundary | -y face | PEC |
| Boundary | +y face | PEC |
| Boundary | -z face | CPML (6 cells, 18 mm) |
| Boundary | +z face | CPML (6 cells, 18 mm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: PEC iris & cavity wall | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [-48 mm, -12 mm] (d = 36 mm) |
| Material | Region 2: PEC | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [12 mm, 48 mm] (d = 36 mm) |
| Source | 1 · Beam | pos: x = 0 m, y = 0 m, pol: J_z, pulse: continuous / custom |
| Source | 2 · TM010EigenMode | pos: z ∈ [-12, 12] mm, pol: E_z (TM₀₁₀), pulse: f₀ = 2.6 GHz, Δf = 901.4 MHz |
| Monitor | 3 · FluxMonitor | Pipe flux (-z), pos: z = -30 mm, freqs: time-domain |
| Monitor | 4 · FluxMonitor | Pipe flux (+z), pos: z = 30 mm, freqs: time-domain |
| Variation | Run (i) | Drive only: TM₀₁₀ mode (E₀ = 100 kV/m, q = 0) |
| Variation | Run (ii) | Beam only: relativistic bunch (q = 1 nC, σ_z = 4 mm, E₀ = 0) |
| Variation | Run (iii) | Drive + beam: phased across φ ∈ [0, 2π] and E₀ ∈ [25, 200] kV/m |
