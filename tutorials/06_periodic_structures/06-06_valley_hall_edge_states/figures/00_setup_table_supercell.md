| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 1 m × 1.73 m × 35.7 mm (z ∈ [0 m, 35.7 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 28 × 48 × 1 (1344 cells) |
| Discretization | Δx & resolution | Δ_min = 35.7 mm, 28.1 cells/λ (in n_max = 3.46), 8.9 cells across feature (317 mm) |
| Boundary | -x face | Bloch (k_x = 4.19 rad/m) |
| Boundary | +x face | Bloch (k_x = 4.19 rad/m) |
| Boundary | -y face | Bloch (k_y = 0 rad/m) |
| Boundary | +y face | Bloch (k_y = 0 rad/m) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Air (eps_r=1), ε = 1.00 (isotropic) |
| Material | Region 1: A1 | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [298.4 mm, 702 mm] (d = 403 mm) |
| Material | Region 2: A2 | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-201.6 mm, 201.6 mm] (d = 403 mm) |
| Material | Region 3: B1 | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [341.6 mm, 658 mm] (d = 317 mm) |
| Material | Region 4: B2 | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-158.4 mm, 158.4 mm] (d = 317 mm) |
| Source | 1 · CurrentSource | pos: x = 500 mm, y = 288.7 mm, pol: J_z (E_z), pulse: f₀ = 86.34 MHz, Δf = 23.98 MHz |
| Monitor | 2 · Probe | Probe 1, pos: x = 440 mm, freqs: time-domain |
| Monitor | 3 · Probe | Probe 2, pos: x = 940 mm, freqs: time-domain |
