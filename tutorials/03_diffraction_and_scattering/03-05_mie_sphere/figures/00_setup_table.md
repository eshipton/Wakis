| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 5.47 µm × 5.47 µm × 5.47 µm (z ∈ [-2.73 µm, 2.73 µm]) |
| Domain | Grid cells | Nx × Ny × Nz = 82 × 82 × 82 (551368 cells) |
| Discretization | Δx & resolution | Δ_min = 66.7 nm, 4.7 cells/λ (in n_max = 2.00), 30.0 cells across feature (2 µm) |
| Boundary | -x face | CPML (10 cells, 667 nm) |
| Boundary | +x face | CPML (10 cells, 667 nm) |
| Boundary | -y face | CPML (10 cells, 667 nm) |
| Boundary | +y face | CPML (10 cells, 667 nm) |
| Boundary | -z face | CPML (10 cells, 667 nm) |
| Boundary | +z face | CPML (10 cells, 667 nm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Dielectric sphere (n = 2) | ε_r = 4.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1 µm, 1 µm] (d = 2 µm) |
| Source | 1 · TFSFSource | pos: x ∈ [-1.27 µm, 1.27 µm], pol: E_x, pulse: f₀ = 286.3 THz, Δf = 95.43 THz |
| Monitor | 2 · Near2FarMonitor | Near2FarMonitor (P_sca, far field), pos: x, y, z ∈ [-1.47, 1.47] µm, freqs: 81 freqs in [95.4, 477.1] THz |
| Monitor | 3 · FluxMonitor | Inner FluxMonitor (P_abs), pos: x, y, z ∈ [-1.07, 1.07] µm, freqs: 81 freqs in [95.4, 477.1] THz |
| Variation | Field monitors | DFTMonitor y = 0 slice and 3D volume at x = 3, 6, 9; SliceRecorder (Ex, y = 0) |
| Variation | Material models | staircase (conformal=False), conformal ('all'), and tensor ('tensor') |
| Variation | Resolution modes | 15 cells/µm (quick mode, ~0.5 Mcells) and 40 cells/µm (full mode, ~6 Mcells) |
