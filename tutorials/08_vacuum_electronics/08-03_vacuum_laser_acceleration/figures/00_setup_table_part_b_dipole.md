| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 12.2 µm × 12.2 µm × 3.43 µm (z ∈ [-1.71 µm, 1.71 µm]) |
| Domain | Grid cells | Nx × Ny × Nz = 60 × 60 × 60 (216000 cells) |
| Discretization | Δx & resolution | Δ_min = 57.1 nm, 14.0 cells/λ (in n_max = 1.00), 14.0 cells across feature (800 nm) |
| Boundary | -x face | CPML (6 cells, 3.66 µm) |
| Boundary | +x face | CPML (6 cells, 3.66 µm) |
| Boundary | -y face | CPML (6 cells, 3.66 µm) |
| Boundary | +y face | CPML (6 cells, 3.66 µm) |
| Boundary | -z face | CPML (6 cells, 343 nm) |
| Boundary | +z face | CPML (6 cells, 343 nm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Source | 1 · Dipole | pos: x = 0 nm, z = 0 nm, pol: J_x (E_x), pulse: f₀ = 374.7 THz, Δf = 74.95 THz |
| Monitor | 2 · Near2FarMonitor | Near2FarMonitor, pos: x ∈ [-6.12 µm, 6.12 µm], freqs: f = 374.7 THz |
| Variation | Run (a): Focused laser | TEM00 (w0 = 1.5λ) projected to far sphere |
| Variation | Run (b): Free-electron dipole | wakis.sources.Dipole at focus driven by d_ddot = (e^2/m) E_L(0, t) |
