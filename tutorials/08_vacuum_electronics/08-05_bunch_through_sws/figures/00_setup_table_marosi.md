| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 1.40 mm × 800 µm × 9.97 mm (z ∈ [0 m, 9.97 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 20 × 28 × 190 (106400 cells) |
| Discretization | Δz & resolution | Δ_min = 25 µm, 184.0 cells/λ (in n_max = 1.00), 4.0 cells across feature (100 µm) |
| Boundary | -x face | PEC |
| Boundary | +x face | PEC |
| Boundary | -y face | PEC |
| Boundary | +y face | PEC |
| Boundary | -z face | CPML (8 cells, 420 µm) |
| Boundary | +z face | CPML (8 cells, 420 µm) |
| Material | Background | Vacuum V-band enclosure & sheet-beam tunnel (ε_r = 1.00, 1.40 × 0.80 mm²) (isotropic) |
| Material | Region 1: −x PEC pillar pair (z₁ = d/4) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [915 µm, 1.18 mm] (d = 270 µm) |
| Material | Region 2: +x PEC pillar pair (z₂ = 3d/4, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [1.44 mm, 1.71 mm] (d = 270 µm) |
| Material | Region 3: −x PEC pillar pair (z₁ = d/4) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [915 µm, 1.18 mm] (d = 270 µm) |
| Material | Region 4: +x PEC pillar pair (z₂ = 3d/4, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [1.44 mm, 1.71 mm] (d = 270 µm) |
| Material | Region 5: −x PEC pillar pair (z₁ = d/4) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [1.96 mm, 2.23 mm] (d = 270 µm) |
| Material | Region 6: +x PEC pillar pair (z₂ = 3d/4, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [2.49 mm, 2.76 mm] (d = 270 µm) |
| Material | Region 7: −x PEC pillar pair (z₁ = d/4) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [1.96 mm, 2.23 mm] (d = 270 µm) |
| Material | Region 8: +x PEC pillar pair (z₂ = 3d/4, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [2.49 mm, 2.76 mm] (d = 270 µm) |
| Material | Region 9: −x PEC pillar pair (z₁ = d/4) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [3.01 mm, 3.29 mm] (d = 270 µm) |
| Material | Region 10: +x PEC pillar pair (z₂ = 3d/4, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [3.54 mm, 3.81 mm] (d = 270 µm) |
| Material | Region 11: −x PEC pillar pair (z₁ = d/4) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [3.01 mm, 3.29 mm] (d = 270 µm) |
| Material | Region 12: +x PEC pillar pair (z₂ = 3d/4, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [3.54 mm, 3.81 mm] (d = 270 µm) |
| Material | Region 13: −x PEC pillar pair (z₁ = d/4) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [4.07 mm, 4.33 mm] (d = 270 µm) |
| Material | Region 14: +x PEC pillar pair (z₂ = 3d/4, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [4.59 mm, 4.86 mm] (d = 270 µm) |
| Material | Region 15: −x PEC pillar pair (z₁ = d/4) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [4.07 mm, 4.33 mm] (d = 270 µm) |
| Material | Region 16: +x PEC pillar pair (z₂ = 3d/4, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [4.59 mm, 4.86 mm] (d = 270 µm) |
| Material | Region 17: −x PEC pillar pair (z₁ = d/4) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [5.11 mm, 5.38 mm] (d = 270 µm) |
| Material | Region 18: +x PEC pillar pair (z₂ = 3d/4, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [5.64 mm, 5.91 mm] (d = 270 µm) |
| Material | Region 19: −x PEC pillar pair (z₁ = d/4) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [5.11 mm, 5.38 mm] (d = 270 µm) |
| Material | Region 20: +x PEC pillar pair (z₂ = 3d/4, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [5.64 mm, 5.91 mm] (d = 270 µm) |
| Material | Region 21: −x PEC pillar pair (z₁ = d/4) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [6.16 mm, 6.43 mm] (d = 270 µm) |
| Material | Region 22: +x PEC pillar pair (z₂ = 3d/4, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [6.69 mm, 6.96 mm] (d = 270 µm) |
| Material | Region 23: −x PEC pillar pair (z₁ = d/4) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [6.16 mm, 6.43 mm] (d = 270 µm) |
| Material | Region 24: +x PEC pillar pair (z₂ = 3d/4, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [6.69 mm, 6.96 mm] (d = 270 µm) |
| Material | Region 25: −x PEC pillar pair (z₁ = d/4) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [7.21 mm, 7.48 mm] (d = 270 µm) |
| Material | Region 26: +x PEC pillar pair (z₂ = 3d/4, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [7.74 mm, 8.01 mm] (d = 270 µm) |
| Material | Region 27: −x PEC pillar pair (z₁ = d/4) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [7.21 mm, 7.48 mm] (d = 270 µm) |
| Material | Region 28: +x PEC pillar pair (z₂ = 3d/4, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [7.74 mm, 8.01 mm] (d = 270 µm) |
| Material | Region 29: −x PEC pillar pair (z₁ = d/4) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [8.27 mm, 8.53 mm] (d = 270 µm) |
| Material | Region 30: +x PEC pillar pair (z₂ = 3d/4, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [8.79 mm, 9.06 mm] (d = 270 µm) |
| Material | Region 31: −x PEC pillar pair (z₁ = d/4) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [8.27 mm, 8.53 mm] (d = 270 µm) |
| Material | Region 32: +x PEC pillar pair (z₂ = 3d/4, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [8.79 mm, 9.06 mm] (d = 270 µm) |
| Source | 1 · Beam | pos: (x, y) = (0, 0), z ∈ [0, L_tot], pol: Jz (3D hybrid glide mode), pulse: f₀ = 65.17 GHz, Δf = 28.00 GHz |
| Monitor | 2 · WakeSolver | 3D longitudinal wake integration line (N = 6, L = 6.30 mm) → W_∥(s), Re Z_∥(f), pos: (x, y) = (0, 0), z ∈ [1.84, 8.14] mm, freqs: 45–80 GHz (V-band forward passband, f_s = 65.17 GHz) |
| Variation | 3D V-Band Staggered-Pillar Glide SWS (Marosi 2025) | Housing a = 1.40 mm × b = 0.80 mm, period d = 1.05 mm, pillars w = 1.08 mm × l = 0.27 mm × t = 0.10 mm, 5.2 kV beam (u₀ = 0.1416c, v_g = +0.1306c) |
