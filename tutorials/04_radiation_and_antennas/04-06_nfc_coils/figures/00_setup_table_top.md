| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 255 mm × 243 mm × 222 mm (z ∈ [-94.5 mm, 127.5 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 38 × 34 × 27 (34884 cells) |
| Discretization | Δx & resolution | Δ_min = 3 mm, 7369.5 cells/λ (in n_max = 1.00), 1.0 cells across feature (3 mm) |
| Boundary | -x face | PEC |
| Boundary | +x face | PEC |
| Boundary | -y face | PEC |
| Boundary | +y face | PEC |
| Boundary | -z face | PEC |
| Boundary | +z face | PEC |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Reader turn 1 (outer 66×66 mm) | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-33 mm, 33 mm] (d = 66 mm) |
| Material | Region 2: Reader turn 2 (middle 54×54 mm) | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-27 mm, 27 mm] (d = 54 mm) |
| Material | Region 3: Reader turn 3 (inner 42×42 mm) | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-21 mm, 21 mm] (d = 42 mm) |
| Source | 1 · LumpedPort | pos: x = 0 m, pol: E_x, pulse: f₀ = 13.56 MHz, Δf = 10.00 MHz |
| Source | 2 · LumpedElement | pos: x = 0 m, pol: E_x, pulse: continuous / custom |
