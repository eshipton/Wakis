| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly = 5.40 µm × 5.40 µm (x ∈ [-2.70 µm, 2.70 µm], y ∈ [-2.70 µm, 2.70 µm]) |
| Domain | Grid cells | Nx × Ny = 1068 × 1068 (1140624 cells) |
| Discretization | Δx & resolution | Δ_min = 119.6 nm, 2.6 cells/λ (in n_max = 2.00), 16.7 cells across feature (2 µm) |
| Boundary | -x face | Mur ABC |
| Boundary | +x face | Mur ABC |
| Boundary | -y face | Mur ABC |
| Boundary | +y face | Mur ABC |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Dielectric cylinder (n = 2) | ε_r = 4.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1 µm, 1 µm] (d = 2 µm) |
| Source | 1 · PlaneWave | pos: r <= 1 µm (cylinder interior), pol: TM (E_z) / TE (H_z), pulse: f₀ = 286.3 THz, Δf = 95.43 THz |
| Monitor | 2 · HarmonicRings | Cylindrical harmonic rings (r = 1.9, 2.2, 2.5 µm), pos: r = 1.9, 2.2, 2.5 µm (256 pts/ring), freqs: 95.4–477.1 THz (81 freqs) |
| Variation | Harmonic extraction | Cylindrical harmonic projection on r = 1.9, 2.2, 2.5 µm with incoming-wave de-embedding |
