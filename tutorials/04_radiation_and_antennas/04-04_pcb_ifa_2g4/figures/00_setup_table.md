| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 94.4 mm × 104.3 mm × 69.6 mm (z ∈ [-34.3 mm, 35.3 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 72 × 74 × 36 (191808 cells) |
| Discretization | Δx & resolution | Δ_min = 115 µm, 522.6 cells/λ (in n_max = 2.04), 4.0 cells across feature (460 µm) |
| Boundary | -x face | CPML (6 cells, 16.4 mm) |
| Boundary | +x face | CPML (6 cells, 16.4 mm) |
| Boundary | -y face | CPML (6 cells, 16.4 mm) |
| Boundary | +y face | CPML (6 cells, 16.4 mm) |
| Boundary | -z face | CPML (6 cells, 18.3 mm) |
| Boundary | +z face | CPML (6 cells, 18.3 mm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: 1.00 mm FR-4 substrate | ε_r = 4.18, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-2 mm, 27.6 mm] (d = 29.6 mm) |
| Material | Region 2: Copper ground plane | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-2 mm, 27.6 mm] (d = 29.6 mm) |
| Material | Region 3: IFA radiating arm | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [0 m, 25.6 mm] (d = 25.6 mm) |
| Source | 1 · LumpedPort | pos: x = 7.94 mm, pol: E_y, pulse: f₀ = 2.4 GHz, Δf = 850.0 MHz |
| Monitor | 2 · Near2FarMonitor | Near2FarMonitor, pos: x ∈ [-10 mm, 35.6 mm], freqs: 3 freqs in [2.4, 2.5] GHz |
| Variation | 5-level uniform trace convergence | Δ_trace ∈ {0.50, 0.35, 0.25, 0.18, 0.1125} mm (up to 98×81×36 = 285,768 cells); continuum Richardson extrapolation f_∞ = 2.401 GHz |
| Variation | Controlled hypothesis & parameter study | FR-4 ε'_r ∈ {4.00, 4.28, 4.50}, ground-plane length (20–45 mm), tap spacing L_4, open arm length L_2, CPW gap L_8, and jog width W_2 |
