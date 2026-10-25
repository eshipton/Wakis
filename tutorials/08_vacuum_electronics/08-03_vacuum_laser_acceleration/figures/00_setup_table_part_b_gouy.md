| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 33.6 µm × 33.6 µm × 4.64 µm (z ∈ [-2.32 µm, 2.32 µm]) |
| Domain | Grid cells | Nx × Ny × Nz = 60 × 60 × 116 (417600 cells) |
| Discretization | Δx & resolution | Δ_min = 40 nm, 20.0 cells/λ (in n_max = 1.00), 40.0 cells across feature (1.60 µm) |
| Boundary | -x face | CPML (6 cells, 13.5 µm) |
| Boundary | +x face | CPML (6 cells, 13.5 µm) |
| Boundary | -y face | CPML (6 cells, 13.5 µm) |
| Boundary | +y face | CPML (6 cells, 13.5 µm) |
| Boundary | -z face | CPML (6 cells, 240 nm) |
| Boundary | +z face | CPML (6 cells, 240 nm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Source | 1 · CurrentSource | pos: z = -2 µm, pol: J_x (E_x), pulse: f₀ = 374.7 THz, Δf = 44.97 THz |
| Monitor | 2 · FluxMonitor | FluxMonitor (x ∈ [-1.28 µm, 1.28 µm]), pos: x ∈ [-1.28 µm, 1.28 µm], freqs: f = 374.7 THz |
| Variation | Waist scan | w0 = 2λ_L and w0 = 3λ_L |
| Variation | Dispersion baseline | Plane wave on identical 3D FIT grid |
