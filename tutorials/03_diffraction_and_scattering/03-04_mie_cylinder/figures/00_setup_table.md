| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 5.10 µm × 5.10 µm × 200 nm (z ∈ [-100 nm, 100 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 102 × 102 × 4 (41616 cells) |
| Discretization | Δx & resolution | Δ_min = 50 nm, 6.3 cells/λ (in n_max = 2.00), 40.0 cells across feature (2 µm) |
| Boundary | -x face | CPML (10 cells, 500 nm) |
| Boundary | +x face | CPML (10 cells, 500 nm) |
| Boundary | -y face | CPML (10 cells, 500 nm) |
| Boundary | +y face | CPML (10 cells, 500 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Dielectric cylinder (n = 2) | ε_r = 4.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1 µm, 1 µm] (d = 2 µm) |
| Source | 1 · TFSFSource | pos: x ∈ [-1.25 µm, 1.25 µm], pol: E_z, pulse: f₀ = 286.3 THz, Δf = 95.43 THz |
| Monitor | 2 · FluxMonitor | Scattered FluxMonitor (P'_sca), pos: x, y ∈ [-1.45, 1.45] µm, freqs: 81 freqs in [95.4, 477.1] THz |
| Monitor | 3 · FluxMonitor | Inner FluxMonitor (P'_abs), pos: x, y ∈ [-1.10, 1.10] µm, freqs: 81 freqs in [95.4, 477.1] THz |
| Variation | Polarizations | TM (E_z along cylinder axis) and TE (E_y across axis, H_z along axis) |
| Variation | Material models | staircase (conformal=False), conformal ('all'), and tensor ('tensor') |
| Variation | Resolution sweep | 10–80 cells/µm (20 cells/µm quick mode, 60 cells/µm full mode) |
