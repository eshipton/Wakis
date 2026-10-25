| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 7.97 µm × 10.5 µm × 28.6 nm (z ∈ [-14.3 nm, 14.3 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 368 × 412 × 1 (151616 cells) |
| Discretization | Δx & resolution | Δ_min = 21.7 nm, 20.6 cells/λ (in n_max = 3.34), 129.3 cells across feature (2.80 µm) |
| Boundary | -x face | CPML (8 cells, 173.3 nm) |
| Boundary | +x face | CPML (8 cells, 173.3 nm) |
| Boundary | -y face | CPML (8 cells, 228 nm) |
| Boundary | +y face | CPML (8 cells, 228 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | InP/InGaAsP cladding (n = 3.311), ε = 10.96 (isotropic) |
| Material | Region 1: Bimodal active core (w₂ = 2.80 µm, n = 3.337) | ε_r = 11.14, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-3.81 µm, 3.81 µm] (d = 7.62 µm) |
| Source | 1 · CurrentSource | pos: x = -1.00 µm, 9 sites in y ∈ [-1.18, +1.18] µm (5 shown), pol: TE (J_y, random φ_j), pulse: f₀ = 201.2 THz, Δf = 50.30 THz |
| Monitor | 2 · ModeMonitor | Downstream 2-mode ModeMonitor (x = +1.00 µm, TE₀ & TE₁), pos: x = 1 µm, freqs: f = 201.2 THz |
