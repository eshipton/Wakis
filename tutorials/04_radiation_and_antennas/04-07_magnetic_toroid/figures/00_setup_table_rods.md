| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 300 mm × 300 mm × 250 mm (z ∈ [-125 mm, 125 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 48 × 48 × 34 (78336 cells) |
| Discretization | Δx & resolution | Δ_min = 2.50 mm, 35.5 cells/λ (in n_max = 4.51), 4.0 cells across feature (10 mm) |
| Boundary | -x face | CPML (6 cells, 79.2 mm) |
| Boundary | +x face | CPML (6 cells, 79.2 mm) |
| Boundary | -y face | CPML (6 cells, 79.2 mm) |
| Boundary | +y face | CPML (6 cells, 79.2 mm) |
| Boundary | -z face | CPML (6 cells, 73.4 mm) |
| Boundary | +z face | CPML (6 cells, 73.4 mm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Permeable rod 1 (+x, +y magnetization) | ε_r = 1.00, μ_r = 20.30, model = Isotropic dielectric, x ∈ [7.50 mm, 17.5 mm] (d = 10 mm) |
| Material | Region 2: Permeable rod 2 (+y, -x magnetization) | ε_r = 1.00, μ_r = 20.30, model = Isotropic dielectric, x ∈ [-17.5 mm, 17.5 mm] (d = 35 mm) |
| Material | Region 3: Permeable rod 3 (-x, -y magnetization) | ε_r = 1.00, μ_r = 20.30, model = Isotropic dielectric, x ∈ [-17.5 mm, -7.50 mm] (d = 10 mm) |
| Material | Region 4: Permeable rod 4 (-y, +x magnetization) | ε_r = 1.00, μ_r = 20.30, model = Isotropic dielectric, x ∈ [-17.5 mm, 17.5 mm] (d = 35 mm) |
| Source | 1 · CurrentSource | pos: x = 0 m, y = 0 m, pol: J_z (E_z), pulse: f₀ = 750.0 MHz, Δf = 220.0 MHz |
| Monitor | 2 · Near2FarMonitor | Near2FarMonitor, pos: x ∈ [-150 mm, 150 mm], freqs: f = 750.0 MHz |
| Monitor | 3 · DFTMonitor | DFTMonitor (x ∈ [-22.5 mm, 22.5 mm], y ∈ [-22.5 mm, 22.5 mm]), pos: x ∈ [-22.5 mm, 22.5 mm], y ∈ [-22.5 mm, 22.5 mm], freqs: f = 750.0 MHz |
