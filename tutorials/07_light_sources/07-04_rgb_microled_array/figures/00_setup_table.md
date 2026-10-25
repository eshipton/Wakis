| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 5.70 µm × 1.96 µm × 1 µm (z ∈ [-500 nm, 500 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 630 × 282 × 1 (177660 cells) |
| Discretization | Δx & resolution | Δ_min = 4.88 nm, 25.6 cells/λ (in n_max = 4.21), 41.0 cells across feature (200 nm) |
| Boundary | -x face | CPML (8 cells, 72.5 nm) |
| Boundary | +x face | CPML (8 cells, 72.5 nm) |
| Boundary | -y face | CPML (8 cells, 39.3 nm) |
| Boundary | +y face | CPML (8 cells, 71.9 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Air superstrate (n = 1.00), ε = 1.00 (isotropic) |
| Material | Region 1: Si substrate (4.22 + 0.048i) | ε_r = 17.76, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-2.85 µm, 2.85 µm] (d = 5.70 µm) |
| Material | Region 2: Polyimide trench (n = 1.50) | ε_r = 2.25, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-2.85 µm, 2.85 µm] (d = 5.70 µm) |
| Material | Region 3: B n-GaN (n = 2.49) | ε_r = 6.20, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-2.02 µm, -975 nm] (d = 1.05 µm) |
| Material | Region 4: B MQW (n = 2.51, 457 nm) | ε_r = 6.30, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-2.02 µm, -975 nm] (d = 1.05 µm) |
| Material | Region 5: B p-GaN (n = 2.49) | ε_r = 6.20, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-2.02 µm, -975 nm] (d = 1.05 µm) |
| Material | Region 6: G n-GaN (n = 2.38) | ε_r = 5.66, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-525 nm, 525 nm] (d = 1.05 µm) |
| Material | Region 7: G MQW (n = 2.41, 527 nm) | ε_r = 5.81, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-525 nm, 525 nm] (d = 1.05 µm) |
| Material | Region 8: G p-GaN (n = 2.38) | ε_r = 5.66, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-525 nm, 525 nm] (d = 1.05 µm) |
| Material | Region 9: R n-GaN (n = 2.35) | ε_r = 5.52, μ_r = 1.00, model = Isotropic dielectric, x ∈ [975 nm, 2.02 µm] (d = 1.05 µm) |
| Material | Region 10: R MQW (n = 2.37, 611 nm) | ε_r = 5.62, μ_r = 1.00, model = Isotropic dielectric, x ∈ [975 nm, 2.02 µm] (d = 1.05 µm) |
| Material | Region 11: R p-GaN (n = 2.35) | ε_r = 5.52, μ_r = 1.00, model = Isotropic dielectric, x ∈ [975 nm, 2.02 µm] (d = 1.05 µm) |
| Source | 1 · CurrentSource | pos: G MQW: x ∈ [-0.42, +0.42] µm, y ∈ [-0.38, -0.22] µm (5×4 = 20 dipoles, 5 shown), pol: J_x (horizontal), pulse: f₀ = 568.9 THz, Δf = 199.1 THz |
| Monitor | 2 · Near2FarMonitor | Near2FarMonitor + subpixel exit windows W_B, W_G, W_R (y = +0.25 µm; normalized by MQW dipole box), pos: y = +0.25 µm, x ∈ [-2.60, +2.60] µm, freqs: f = 568.9 THz |
| Variation | Task 0 — Blue Method A (`method_a_B`) | notes=Blue mesa active (λ_B = 457 nm, x_c = -1.50 µm); Method A exact 20-run incoherent dipole sum |
| Variation | Task 1 — Green Method A (`method_a_G`) | notes=Green mesa active (λ_G = 527 nm, x_c = 0.00 µm); Method A exact 20-run incoherent dipole sum |
| Variation | Task 2 — Red Method A (`method_a_R`) | notes=Red mesa active (λ_R = 611 nm, x_c = +1.50 µm); Method A exact 20-run incoherent dipole sum |
| Variation | Task 3 — Green Method B (`method_b_G`) | notes=Simultaneous 20-dipole random-phase realizations (K = 1, 2, 4, 8, 16) vs. exact Method A |
| Variation | Tasks 5–9 — Grid & Si skin-depth convergence (`conv_cpw10..28`) | notes=cpw ∈ {10, 12, 16, 20, 28} (56.9–449.9 cells/δ_P in absorbing Si substrate) |
