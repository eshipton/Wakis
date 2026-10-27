| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 14.5 mm × 14.5 mm × 14.5 mm (z ∈ [-7.25 mm, 7.25 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 34 × 18 × 32 (19584 cells) |
| Discretization | Δx & resolution | Δ_min = 250 µm, 396.4 cells/λ (in n_max = 1.00), 4.0 cells across feature (1 mm) |
| Boundary | -x face | PEC |
| Boundary | +x face | PEC |
| Boundary | -y face | PEC |
| Boundary | +y face | PEC |
| Boundary | -z face | Bloch (k_z = -115 rad/m) |
| Boundary | +z face | Bloch (k_z = -115 rad/m) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: PEC | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-7.25 mm, -6.75 mm] (d = 500 µm) |
| Material | Region 2: PEC | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [6.75 mm, 7.25 mm] (d = 500 µm) |
| Material | Region 3: PEC | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-6.75 mm, 6.75 mm] (d = 13.5 mm) |
| Material | Region 4: PEC | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-6.75 mm, 6.75 mm] (d = 13.5 mm) |
| Material | Region 5: PEC | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-750 µm, 750 µm] (d = 1.50 mm) |
| Material | Region 6: PEC | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-750 µm, 750 µm] (d = 1.50 mm) |
| Material | Region 7: PEC | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-5.75 mm, -1.75 mm] (d = 4 mm) |
| Material | Region 8: PEC | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-1.75 mm, -750 µm] (d = 1 mm) |
| Material | Region 9: PEC | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [1.75 mm, 5.75 mm] (d = 4 mm) |
| Material | Region 10: PEC | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [750 µm, 1.75 mm] (d = 1 mm) |
| Material | Region 11: PEC | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-5.75 mm, -1.75 mm] (d = 4 mm) |
| Material | Region 12: PEC | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-1.75 mm, -750 µm] (d = 1 mm) |
| Material | Region 13: PEC | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [1.75 mm, 5.75 mm] (d = 4 mm) |
| Material | Region 14: PEC | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [750 µm, 1.75 mm] (d = 1 mm) |
| Source | 1 · CurrentSource | pos: x = 500 µm, z = -2 mm, pol: J_z (E_z), pulse: f₀ = 3.1 GHz, Δf = 1.8 GHz |
| Source | 2 · CurrentSource | pos: x = 1.50 mm, z = 1 mm, pol: J_x (E_x), pulse: f₀ = 3.1 GHz, Δf = 1.8 GHz |
| Monitor | 3 · ResonanceMonitor | ResonanceMonitor (x = 0 m), pos: x = 0 m, freqs: time-domain |
| Monitor | 4 · HannDFTMonitor | HannDFTMonitor (x ∈ [-7.25 mm, 7.25 mm], z ∈ [-7.25 mm, 7.25 mm]), pos: x ∈ [-7.25 mm, 7.25 mm], z ∈ [-7.25 mm, 7.25 mm], freqs: f = 3 GHz |
| Variation | Task 1 · Bloch phase sweep | phi = beta_0 p in [15, 180] deg (Mode 1 LH backward band 2.848-3.314 GHz, Mode 2 RH band 4.08-4.65 GHz) |
| Variation | Task 1 · Plain-plate control | plain_plate=True removes CeSRR slots -> 0 modes in [2.5, 3.8] GHz (below 10.34 GHz waveguide cutoff) |
| Variation | Task 1 · 3-level mesh convergence | Graded slot-resolved grids + Richardson extrapolation on f_1 and <K_{c,0}> at phi = 95.7 deg |
