| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 11.7 µm × 11.7 µm × 3.36 µm (z ∈ [-1.68 µm, 1.68 µm]) |
| Domain | Grid cells | Nx × Ny × Nz = 54 × 54 × 84 (244944 cells) |
| Discretization | Δx & resolution | Δ_min = 40 nm, 11.5 cells/λ (in n_max = 1.73), 8.0 cells across feature (320 nm) |
| Boundary | -x face | CPML (6 cells, 4.63 µm) |
| Boundary | +x face | CPML (6 cells, 4.63 µm) |
| Boundary | -y face | CPML (6 cells, 4.63 µm) |
| Boundary | +y face | CPML (6 cells, 4.63 µm) |
| Boundary | -z face | CPML (6 cells, 240 nm) |
| Boundary | +z face | CPML (6 cells, 240 nm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Dielectric | ε_r = 1.50, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-160 nm, -120 nm] (d = 40 nm) |
| Material | Region 2: PEC screen (-L) | ε_r = 3.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-5.87 µm, 5.87 µm] (d = 11.7 µm) |
| Source | 1 · CurrentSource | pos: z = -1.28 µm, pol: J_x (E_x), pulse: f₀ = 374.7 THz, Δf = 37.47 THz |
| Monitor | 2 · Probe | Probe 1, pos: x = 0 m, freqs: time-domain |
| Variation | Cavity length scan | L / z0 in {0.25, 0.5, 1, 2, 4} (Eq. 76) |
| Variation | Aperture cutoff scan | a / (γ λ_bar) for γ = 3 and γ = 8 at L = z0 (Eq. 84) |
