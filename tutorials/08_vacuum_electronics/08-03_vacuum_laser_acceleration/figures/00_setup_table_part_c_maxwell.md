| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 8.91 µm × 8.91 µm × 5.28 µm (z ∈ [-2.64 µm, 2.64 µm]) |
| Domain | Grid cells | Nx × Ny × Nz = 68 × 68 × 132 (610368 cells) |
| Discretization | Δx & resolution | Δ_min = 40 nm, 11.5 cells/λ (in n_max = 1.73), 1.6 cells across feature (64 nm) |
| Boundary | -x face | CPML (6 cells, 3.11 µm) |
| Boundary | +x face | CPML (6 cells, 3.11 µm) |
| Boundary | -y face | CPML (6 cells, 3.11 µm) |
| Boundary | +y face | CPML (6 cells, 3.11 µm) |
| Boundary | -z face | CPML (6 cells, 240 nm) |
| Boundary | +z face | CPML (6 cells, 240 nm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Dielectric | ε_r = 3.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-4.45 µm, 4.45 µm] (d = 8.91 µm) |
| Source | 1 · Beam | pos: x = 0 m, y = 0 m, pol: J_z, pulse: continuous / custom |
| Monitor | 2 · Near2FarMonitor | Near2FarMonitor, pos: x ∈ [-4.45 µm, 4.45 µm], freqs: f = 374.7 THz |
| Variation | Run 1: Laser only (L) | Focused TEM00 drive field (E_L, H_L) |
| Variation | Run 2: Beam only (B) | Prescribed crossing beam current J_B along x_e(z) = -theta_c z |
| Variation | Run 3: Laser + Beam (L+B) | Superposition -> W = -[U_int(t_end) + Phi_int(0->t_end)] |
