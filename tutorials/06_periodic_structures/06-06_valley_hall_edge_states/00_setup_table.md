| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 9 m × 12.1 m × 10 m (z ∈ [0 m, 10 m]) |
| Domain | Grid cells | Nx × Ny × Nz = 198 × 252 × 1 (49896 cells) |
| Discretization | Δx & resolution | Δ_min = 45.5 mm, 22.3 cells/λ (in n_max = 3.46), 7.0 cells across feature (317 mm) |
| Boundary | -x face | CPML (6 cells, 272.7 mm) |
| Boundary | +x face | CPML (6 cells, 273 mm) |
| Boundary | -y face | CPML (6 cells, 288.7 mm) |
| Boundary | +y face | CPML (6 cells, 288.7 mm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Air (eps_r=1), ε = 1.00 (isotropic) |
| Material | Region 1: lower | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [298.4 mm, 702 mm] (d = 403 mm) |
| Material | Region 2: lower | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-201.6 mm, 201.6 mm] (d = 403 mm) |
| Material | Region 3: lower | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [341.6 mm, 658 mm] (d = 317 mm) |
| Material | Region 4: lower | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-158.4 mm, 158.4 mm] (d = 317 mm) |
| Material | Region 5: lower | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [298.4 mm, 702 mm] (d = 403 mm) |
| Material | Region 6: lower | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-201.6 mm, 201.6 mm] (d = 403 mm) |
| Material | Region 7: lower | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [341.6 mm, 658 mm] (d = 317 mm) |
| Material | Region 8: lower | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-158.4 mm, 158.4 mm] (d = 317 mm) |
| Material | Region 9: lower | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [298.4 mm, 702 mm] (d = 403 mm) |
| Material | Region 10: lower | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-201.6 mm, 201.6 mm] (d = 403 mm) |
| Material | Region 11: lower | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [341.6 mm, 658 mm] (d = 317 mm) |
| Material | Region 12: lower | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-158.4 mm, 158.4 mm] (d = 317 mm) |
| Material | Region 13: lower | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [298.4 mm, 702 mm] (d = 403 mm) |
| Material | Region 14: lower | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-201.6 mm, 201.6 mm] (d = 403 mm) |
| Material | Region 15: lower | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [341.6 mm, 658 mm] (d = 317 mm) |
| Material | Region 16: lower | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-158.4 mm, 158.4 mm] (d = 317 mm) |
| Material | Region 17: upper | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [341.6 mm, 658 mm] (d = 317 mm) |
| Material | Region 18: upper | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-158.4 mm, 158.4 mm] (d = 317 mm) |
| Material | Region 19: upper | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [298.4 mm, 702 mm] (d = 403 mm) |
| Material | Region 20: upper | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-201.6 mm, 201.6 mm] (d = 403 mm) |
| Material | Region 21: upper | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [341.6 mm, 658 mm] (d = 317 mm) |
| Material | Region 22: upper | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-158.4 mm, 158.4 mm] (d = 317 mm) |
| Material | Region 23: upper | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [298.4 mm, 702 mm] (d = 403 mm) |
| Material | Region 24: upper | ε_r = 12.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-201.6 mm, 201.6 mm] (d = 403 mm) |
| Source | 1 · CurrentSource | pos: x = 1.50 m, pol: J_z (E_z), pulse: f₀ = 85.44 MHz, Δf = 4.8 MHz |
| Monitor | 2 · FluxMonitor | FluxMonitor (x = 2.50 m), pos: x = 2.50 m, freqs: 11 freqs in [83.9, 86.9] MHz |
| Monitor | 3 · FluxMonitor | FluxMonitor (x = 7.50 m), pos: x = 7.50 m, freqs: 11 freqs in [83.9, 86.9] MHz |
