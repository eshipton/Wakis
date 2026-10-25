| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 236 mm × 223.8 mm × 190.8 mm (z ∈ [-87.9 mm, 102.9 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 38 × 34 × 23 (29716 cells) |
| Discretization | Δx & resolution | Δ_min = 3 mm, 7369.5 cells/λ (in n_max = 1.00), 1.0 cells across feature (3 mm) |
| Boundary | -x face | PEC |
| Boundary | +x face | PEC |
| Boundary | -y face | PEC |
| Boundary | +y face | PEC |
| Boundary | -z face | PEC |
| Boundary | +z face | PEC |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: PEC chassis plate | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-39 mm, 39 mm] (d = 78 mm) |
| Material | Region 2: NiZn ferrite (μ'·t = 30 mm) | ε_r = 1.00, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-39 mm, 39 mm] (d = 78 mm) |
| Material | Region 3: Reader coil (3-turn PEC) | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-33 mm, 33 mm] (d = 66 mm) |
| Material | Region 4: ID-1 Tag coil (3-turn PEC) | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-39 mm, 39 mm] (d = 78 mm) |
| Source | 1 · LumpedPort | pos: x = 0 m, pol: E_x, pulse: f₀ = 13.56 MHz, Δf = 10.00 MHz |
| Source | 2 · LumpedPort | pos: x = 0 m, pol: E_x, pulse: continuous / custom |
