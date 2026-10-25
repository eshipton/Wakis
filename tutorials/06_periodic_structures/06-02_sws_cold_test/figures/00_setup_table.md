| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 1 mm × 829 µm × 500 µm (z ∈ [0 m, 500 µm]) |
| Domain | Grid cells | Nx × Ny × Nz = 1 × 45 × 36 (1620 cells) |
| Discretization | Δz & resolution | Δ_min = 11.3 µm, 220.6 cells/λ (in n_max = 1.00), 8.8 cells across feature (100 µm) |
| Boundary | -x face | Periodic |
| Boundary | +x face | Periodic |
| Boundary | -y face | PEC |
| Boundary | +y face | PEC |
| Boundary | -z face | Bloch (k_z = -3.14e+03 rad/m) |
| Boundary | +z face | Bloch (k_z = -3.14e+03 rad/m) |
| Material | Background | Vacuum interaction tunnel & groove slots (ε_r = 1.00, μ_r = 1.00) (isotropic) |
| Material | Region 1: Bottom PEC vane (z₁ = d/4 = 125 µm, l×h = 100×300 µm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [75 µm, 175 µm] (d = 100 µm) |
| Material | Region 2: Top PEC vane (z₂ = 3d/4 = 375 µm, glide +d/2) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [325 µm, 425 µm] (d = 100 µm) |
| Source | 1 · CurrentSource | pos: (z, y) = (210, 25) µm (off-axis to excite even & odd Bloch branches), pol: Jz, Jy (TM / LSE₁₀), pulse: f₀ = 120.0 GHz, Δf = 70.00 GHz |
| Monitor | 2 · ResonanceMonitor | Stage-1 ResonanceMonitor (harminv) at (z, y) = (70, 0) µm, pos: (z, y) = (70, 0) µm (beam axis y = 0), freqs: 10–245 GHz (Stage-1 eigenfrequency & Q extraction) |
| Monitor | 3 · DFTMonitor | Stage-2 HannDFTMonitor (3D unit-cell volume), pos: z ∈ [0, 500] µm, y ∈ [−400, +400] µm (full unit cell), freqs: Stage-1 eigenfrequencies f_m(β₀) (free ring-down) |
| Variation | Floquet–Bloch phase & symmetry sweep (Tasks 3–8) | Δφ = β₀ d ∈ [0.05π, 0.95π] (k_z = −β₀, E(z+d) = E(z) e^{−jβ₀d}), mirror (δz = 0) vs glide (δz = d/2 = 250 µm) |
| Variation | Vane / groove & sheet-beam geometry (Karetnikova 2018) | Period d = 500 µm, vane l = 100 µm × h = 300 µm, groove s = 400 µm, tunnel 2a = 200 µm, 20 kV beam (H_b = 150 µm, u₀ = 0.272c) |
| Variation | 3D LSE₁₀ rectangular housing (Task 9) | PEC side walls at x = 0, b = 850 µm (f_{c,TE10} = 176.35 GHz, 87,040 uniform / 53,856 graded cells) |
| Variation | 3D V-Band Marosi staggered-pillar SWS (Tasks 10–11) | Housing 1.40 × 0.80 × 1.05 mm³, pillars w = 1.08 mm, l = 0.27 mm, t = 0.10 mm, 5.2 kV beam (u₀ = 0.1416c) — see 00a/00b_setup_*_marosi.png |
