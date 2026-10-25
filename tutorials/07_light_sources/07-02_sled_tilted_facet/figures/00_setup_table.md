| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 6.50 µm × 12.6 µm × 28.6 nm (z ∈ [-14.3 nm, 14.3 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 288 × 478 × 1 (137664 cells) |
| Discretization | Δx & resolution | Δ_min = 21.7 nm, 20.6 cells/λ (in n_max = 3.34), 10.0 cells across feature (216.6 nm) |
| Boundary | -x face | CPML (10 cells, 216.6 nm) |
| Boundary | +x face | CPML (10 cells, 283 nm) |
| Boundary | -y face | CPML (10 cells, 285 nm) |
| Boundary | +y face | CPML (10 cells, 285 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Air exterior, ε = 1.00 (isotropic) |
| Material | Region 1: InP/InGaAsP cladding (n = 3.311) | ε_r = 10.96, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-4.20 µm, 0 m] (d = 4.20 µm) |
| Material | Region 2: Ridge core (w₁ = 1.75 µm, n = 3.337) | ε_r = 11.14, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-4.20 µm, 0 m] (d = 4.20 µm) |
| Material | Region 3: λ/4 AR film (θ_t = 7°, n = 1.72) | ε_r = 2.96, μ_r = 1.00, model = Isotropic dielectric, x ∈ [0 m, 216.6 nm] (d = 216.6 nm) |
| Source | 1 · ModeSource | pos: x = -3.29 µm, pol: mode 0, pulse: f₀ = 201.2 THz, Δf = 44.26 THz |
| Monitor | 2 · ModeMonitor | Upstream ModeMonitor (x = -2.60 µm, TE₀/TE₁), pos: x = -2.60 µm, freqs: f = 201.2 THz |
| Monitor | 3 · Near2FarMonitor | Near2FarMonitor in air (x = +0.85 µm, Snell 24°), pos: x = +0.85 µm, y ∈ [-4.80, 4.80] µm, freqs: f = 201.2 THz |
