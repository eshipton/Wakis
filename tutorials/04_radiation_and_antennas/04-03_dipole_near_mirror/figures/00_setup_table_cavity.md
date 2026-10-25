| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 3 µm × 62.5 nm × 1 µm (z ∈ [0 m, 1 µm]) |
| Domain | Grid cells | Nx × Ny × Nz = 144 × 3 × 48 (20736 cells) |
| Discretization | Δx & resolution | Δ_min = 20.8 nm, 50.5 cells/λ (in n_max = 2.40), 24.0 cells across feature (500 nm) |
| Boundary | -x face | CPML (8 cells, 166.7 nm) |
| Boundary | +x face | CPML (8 cells, 166.7 nm) |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | PEC |
| Boundary | +z face | PEC |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Dielectric | ε_r = 5.76, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.50 µm, 1.50 µm] (d = 3 µm) |
| Source | 1 · CurrentSource | pos: x = 0 m, z = 500 nm, pol: J_y (E_y), pulse: f₀ = 118.7 THz, Δf = 23.73 THz |
| Monitor | 2 · DFTMonitor | DFTMonitor (x = 0 m), pos: x = 0 m, freqs: 15 freqs in [74.9, 169.9] THz |
| Variation | Band-limited optical thickness sweep | param=d = nL/λ₀ ∈ [0.22, 2.32] across 3 spectral windows, note=Avoids v_g = 0 Van Hove waveguide cutoffs (d = 0.5, 1.5 for J_y, J_x; d = 1.0, 2.0 for J_z) |
