| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 1.45 mm × 850 µm × 1.05 mm (z ∈ [0 m, 1.05 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 29 × 17 × 20 (9860 cells) |
| Discretization | Δz & resolution | Δ_min = 50 µm, 99.9 cells/λ (in n_max = 1.00), 2.0 cells across feature (100 µm) |
| Boundary | -x face | PEC |
| Boundary | +x face | PEC |
| Boundary | -y face | PEC |
| Boundary | +y face | PEC |
| Boundary | -z face | Bloch (k_z = -1.5e+03 rad/m) |
| Boundary | +z face | Bloch (k_z = -1.5e+03 rad/m) |
| Material | Background | Vacuum waveguide & sheet-beam tunnel (ε_r = 1.00) (isotropic) |
| Material | Region 1: −x PEC pillar pair (z₁ = d/4 = 262.5 µm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [127.5 µm, 397.5 µm] (d = 270 µm) |
| Material | Region 2: −x PEC pillar pair (z₁ = d/4 = 262.5 µm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [127.5 µm, 397.5 µm] (d = 270 µm) |
| Material | Region 3: +x PEC pillar pair (z₂ = 3d/4 = 787.5 µm, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [652 µm, 922 µm] (d = 270 µm) |
| Material | Region 4: +x PEC pillar pair (z₂ = 3d/4 = 787.5 µm, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [652 µm, 922 µm] (d = 270 µm) |
| Source | 1 · CurrentSource | pos: (x, y, z) = (40, 20, 420) µm, pol: Jz, Jy, pulse: f₀ = 60.00 GHz, Δf = 30.00 GHz |
| Monitor | 2 · ResonanceMonitor | ResonanceMonitor (harminv) at (x, y, z) = (0, 0, 150) µm, pos: (x, y, z) = (0, 0, 150) µm, freqs: 40–85 GHz (V-band passband) |
| Monitor | 3 · DFTMonitor | 3D HannDFTMonitor over V-band unit cell, pos: x ∈ [−0.70, +0.70] mm, y ∈ [−0.40, +0.40] mm, z ∈ [0, 1.05] mm, freqs: 49–74 GHz |
| Variation | Housing, pillar & sheet-beam geometry (Marosi 2022) | Housing 1.40 × 0.80 × 1.05 mm³ (PEC ±x, ±y), pillars w = 1.08 mm × l = 270 µm × t = 100 µm (−x pair at z₁ = d/4, +x pair at z₂ = 3d/4), tunnel |y| ≤ 150 µm (5.2 kV sheet beam) |
| Variation | Glide vs broken-glide symmetry (Tasks 10–11) | Glide-symmetric (+x pillars at z₂ = 3d/4, gapless at β_n d = 3π, f = 60.86 GHz) vs mirror control (17.81 GHz stopband) |
