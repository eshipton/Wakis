| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 300 mm × 300 mm × 250 mm (z ∈ [-125 mm, 125 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 48 × 48 × 34 (78336 cells) |
| Discretization | Δx & resolution | Δ_min = 2.50 mm, 159.9 cells/λ (in n_max = 1.00), 4.0 cells across feature (10 mm) |
| Boundary | -x face | CPML (6 cells, 79.2 mm) |
| Boundary | +x face | CPML (6 cells, 79.2 mm) |
| Boundary | -y face | CPML (6 cells, 79.2 mm) |
| Boundary | +y face | CPML (6 cells, 79.2 mm) |
| Boundary | -z face | CPML (6 cells, 73.4 mm) |
| Boundary | +z face | CPML (6 cells, 73.4 mm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: PEC wire loop top segment | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-12.5 mm, 12.5 mm] (d = 25 mm) |
| Material | Region 2: PEC wire loop left segment | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-13.8 mm, -11.3 mm] (d = 2.50 mm) |
| Material | Region 3: PEC wire loop right segment | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [11.3 mm, 13.8 mm] (d = 2.50 mm) |
| Material | Region 4: PEC wire loop bottom feed segment | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-12.5 mm, 12.5 mm] (d = 25 mm) |
| Source | 1 · LumpedPort | pos: x = 0 m, pol: E_x, pulse: f₀ = 750.0 MHz, Δf = 220.0 MHz |
| Monitor | 2 · Near2FarMonitor | Near2FarMonitor, pos: x ∈ [-150 mm, 150 mm], freqs: f = 750.0 MHz |
