| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 3.19 µm × 3.19 µm × 13.2 nm (z ∈ [-6.59 nm, 6.59 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 242 × 242 × 1 (58564 cells) |
| Discretization | Δx & resolution | Δ_min = 13.2 nm, 21.7 cells/λ (in n_max = 3.37), 7.9 cells across feature (104.4 nm) |
| Boundary | -x face | CPML (8 cells, 105.5 nm) |
| Boundary | +x face | CPML (8 cells, 105.5 nm) |
| Boundary | -y face | CPML (8 cells, 105.5 nm) |
| Boundary | +y face | CPML (8 cells, 105.5 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: 10x10 Finite PCSEL Cavity (n_eff=3.323) | ε_r = 11.35, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.45 µm, 1.45 µm] (d = 2.90 µm) |
| Material | Region 2: Etched PhC air hole array (r/a=0.18) | ε_r = 8.33, μ_r = 1.00, model = Isotropic dielectric, x ∈ [92.8 nm, 197.2 nm] (d = 104.4 nm) |
| Source | 1 · CurrentSource | pos: x = 217.5 nm, y = 145 nm, pol: J_y (E_y), pulse: f₀ = 308.7 THz, Δf = 35.00 THz |
| Monitor | 2 · ResonanceMonitor | ResonanceMonitor (x = 217.5 nm), pos: x = 217.5 nm, freqs: time-domain |
| Monitor | 3 · DFTMonitor | DFTMonitor (x ∈ [-1.45 µm, 1.45 µm]), pos: x ∈ [-1.45 µm, 1.45 µm], freqs: f = 300.0 THz |
