| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 24.3 mm × 24.3 mm × 24.3 mm (z ∈ [-12.2 mm, 12.2 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 38 × 38 × 38 (54872 cells) |
| Discretization | Δx & resolution | Δ_min = 640 µm, 26.1 cells/λ (in n_max = 1.00), 7.0 cells across feature (4.48 mm) |
| Boundary | -x face | CPML (6 cells, 3.84 mm) |
| Boundary | +x face | CPML (6 cells, 3.84 mm) |
| Boundary | -y face | CPML (6 cells, 3.84 mm) |
| Boundary | +y face | CPML (6 cells, 3.84 mm) |
| Boundary | -z face | CPML (6 cells, 3.84 mm) |
| Boundary | +z face | CPML (6 cells, 3.84 mm) |
| Material | Background | Muscle (3-pole Debye), ε = 1.00 (isotropic) |
| Material | Region 1: Insulating Capsule (a) | ε_r = 1.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-2.24 mm, 2.24 mm] (d = 4.48 mm) |
| Source | 1 · CurrentSource | pos: x = 0 m, z = 0 m, pol: J_z (E_z), pulse: f₀ = 1.5 GHz, Δf = 1.8 GHz |
| Monitor | 2 · FluxMonitor | FluxMonitor (x ∈ [-3.20 mm, 3.20 mm]), pos: x ∈ [-3.20 mm, 3.20 mm], freqs: 2 freqs in [0.4, 2.5] GHz |
| Monitor | 3 · FluxMonitor | FluxMonitor (x ∈ [-6.40 mm, 6.40 mm]), pos: x ∈ [-6.40 mm, 6.40 mm], freqs: 2 freqs in [0.4, 2.5] GHz |
| Variation | Muscle model | 3-pole Debye fit to Gabriel (1996): ε_r(403 MHz) = 57.2 - 35.6j; ε_r(2.45 GHz) = 52.8 - 12.8j |
| Variation | Capsule & shell sweep | a ∈ {1.60, 2.24, 2.88} mm, ε_cap ∈ {1, 12}; r1 ∈ {2.56, 3.20, 3.84} mm, r2 = 6.40 mm |
| Variation | Antenna feeds | Ideal TM dipole, ideal 4-segment TE loop, gapped PEC loop, gapped PEC dipole, NiZn ferrite core |
