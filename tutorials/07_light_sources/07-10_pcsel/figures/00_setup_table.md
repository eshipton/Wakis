| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 290 nm × 290 nm × 1.80 µm (z ∈ [-900 nm, 900 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 24 × 24 × 73 (42048 cells) |
| Discretization | Δx & resolution | Δ_min = 12 nm, 22.8 cells/λ (in n_max = 3.52), 8.7 cells across feature (104.4 nm) |
| Boundary | -x face | Periodic |
| Boundary | +x face | Periodic |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (8 cells, 277 nm) |
| Boundary | +z face | CPML (8 cells, 277 nm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Active InGaAs/GaAs QW core (n=3.52) | ε_r = 12.39, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-145 nm, 145 nm] (d = 290 nm) |
| Material | Region 2: PhC GaAs host layer (n=3.46) | ε_r = 11.97, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-145 nm, 145 nm] (d = 290 nm) |
| Material | Region 3: Circular air hole (r/a=0.18, n=1.00) | ε_r = 1.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-52.2 nm, 52.2 nm] (d = 104.4 nm) |
| Source | 1 · CurrentSource | pos: x = 72.5 nm, z = -60 nm, pol: J_y (E_y), pulse: f₀ = 311.0 THz, Δf = 18.00 THz |
| Source | 2 · CurrentSource | pos: x = -72.5 nm, z = -60 nm, pol: J_y (E_y), pulse: f₀ = 311.0 THz, Δf = 18.00 THz |
| Monitor | 3 · ResonanceMonitor | ResonanceMonitor (x = 72.5 nm), pos: x = 72.5 nm, freqs: time-domain |
| Monitor | 4 · DFTMonitor | DFTMonitor (z = 60 nm), pos: z = 60 nm, freqs: 3 freqs in [308.6, 313.2] THz |
