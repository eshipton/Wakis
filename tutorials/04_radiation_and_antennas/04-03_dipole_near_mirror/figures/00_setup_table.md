| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 2.40 µm × 120 nm × 2.08 µm (z ∈ [0 m, 2.08 µm]) |
| Domain | Grid cells | Nx × Ny × Nz = 60 × 3 × 52 (9360 cells) |
| Discretization | Δx & resolution | Δ_min = 40 nm, 25.0 cells/λ (in n_max = 1.00), 6.0 cells across feature (240 nm) |
| Boundary | -x face | CPML (8 cells, 320 nm) |
| Boundary | +x face | CPML (8 cells, 320 nm) |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | PEC |
| Boundary | +z face | CPML (8 cells, 320 nm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Source | 1 · CurrentSource | pos: x = 0 m, z = 240 nm, pol: J_y (E_y), pulse: f₀ = 299.8 THz, Δf = 89.94 THz |
| Monitor | 2 · Near2FarMonitor | Near2FarMonitor box (ground_plane='-z'), pos: x ∈ [-0.28, 0.28] µm, z ∈ [0, 0.40] µm, freqs: f = 299.8 THz |
| Variation | Polarizations | param=component ∈ {'y' (TE_∥), 'x' (TM_∥), 'z' (TM_⊥)}, note=Exact 2D Bessel Purcell factors 1 - J₀(2kh), 1 - J₀(2kh) + J₂(2kh), 1 + J₀(2kh) + J₂(2kh) |
| Variation | Broadband & monochromatic kh sweep | param=kh ∈ [0.25, 9.5], h/λ₀ ∈ {0.24, 0.48, 0.76}, note=Local source work DFTMonitor at (0, y_mid, h) and Near2FarMonitor(ground_plane='-z') angular patterns |
