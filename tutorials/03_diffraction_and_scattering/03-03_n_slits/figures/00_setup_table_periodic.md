| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 1.08 µm × 2.42 µm × 210 nm (z ∈ [-105 nm, 105 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 42 × 98 × 1 (4116 cells) |
| Discretization | Δy & resolution | Δ_min = 21 nm, 28.6 cells/λ (in n_max = 1.00), 4.0 cells across feature (84 nm) |
| Boundary | -x face | Periodic |
| Boundary | +x face | Periodic |
| Boundary | -y face | CPML (10 cells, 249.7 nm) |
| Boundary | +y face | CPML (10 cells, 247.4 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: PEC half-bar | ε_r = inf, μ_r = 1.00, model = PEC, y ∈ [-42 nm, 42 nm] (d = 84 nm) |
| Material | Region 2: PEC half-bar | ε_r = inf, μ_r = 1.00, model = PEC, y ∈ [-42 nm, 42 nm] (d = 84 nm) |
| Source | 1 · CurrentSource | pos: y = -630 nm, pol: J_x (E_x), pulse: f₀ = 499.7 THz, Δf = 224.8 THz |
| Monitor | 2 · DiffractionMonitor | Near2FarMonitor, pos: y ∈ [-1.09 µm, 1.33 µm], freqs: f = 499.7 THz |
