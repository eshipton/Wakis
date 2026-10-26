| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 680 nm × 21.2 nm × 2.60 µm (z ∈ [-1.30 µm, 1.30 µm]) |
| Domain | Grid cells | Nx × Ny × Nz = 32 × 1 × 74 (2368 cells) |
| Discretization | Δz & resolution | Δ_min = 15 nm, 30.2 cells/λ (in n_max = 3.48), 5.7 cells across feature (85 nm) |
| Boundary | -x face | Periodic |
| Boundary | +x face | Periodic |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (10 cells, 448 nm) |
| Boundary | +z face | CPML (10 cells, 448 nm) |
| Material | Background | Air / vacuum cladding (ε_bg = 1.00, n = 1.00) (isotropic) |
| Material | Region 1: Active bar 1 (w₁ = 170 nm, SaturableGain / FourLevelGain) | ε_r = 12.11, μ_r = 1.00, model = Isotropic dielectric, z ∈ [-95 nm, 95 nm] (d = 190 nm) |
| Material | Region 2: Active bar 2 (w₂ = 170 nm, SaturableGain / FourLevelGain) | ε_r = 12.11, μ_r = 1.00, model = Isotropic dielectric, z ∈ [-95 nm, 95 nm] (d = 190 nm) |
| Source | 1 · CurrentSource | pos: (x, z) = (−127.5, −25) nm (inside active Bar 1), pol: +Jy (TE, C₂-odd BIC seed), pulse: f₀ = 190.5 THz, Δf = 15.24 THz |
| Source | 2 · CurrentSource | pos: (x, z) = (+127.5, −25) nm (inside active Bar 2), pol: −Jy (TE, C₂-odd BIC seed), pulse: f₀ = 190.5 THz, Δf = 15.24 THz |
| Monitor | 3 · ResonanceMonitor | ResonanceMonitor (harminv Ey & ΔN(t) probe) at (x₁, z) = (−127.5, +25) nm, pos: (x, z) = (−127.5, +25) nm, freqs: 0.90 f₀ – 1.10 f₀ (Q(k_x), γ_eff(g₀), and L–I steady state) |
| Monitor | 4 · DFTMonitor | Radiated far-field monitor plane at z = +580 nm, pos: z = +580 nm (x ∈ [−340, +340] nm), freqs: f₀ = 190.5 THz (λ_BIC = 1573.9 nm) |
| Variation | Part 1 — Infinite lattice Bloch dispersion & vortex (Task 0) | k_x p / (2π) ∈ [0, 0.10] (Q ∝ k_x⁻²), bar width w₀ ∈ [140, 200] nm BIC sweep, 3D Bloch A₁ BIC vortex (q = +1) |
| Variation | Part 2 — Finite N-period strip array Q scaling & mode selection (Task 1) | N ∈ {4, 6, 8, 10, 12} strip with open CPML BCs (Q₁ ∝ N^p; mode m = 1 vs m = 2, 3) — see 00a/00b_*_finite.png |
| Variation | Part 3 — Active BIC lasing & doughnut far field (Task 2) | SaturableGain & FourLevelGain (spont. emission, s=80, L–I curve, ΔN clamping, Near2Far doughnut I(θ=0°)≈0) |
| Variation | Part 4 — Symmetry-breaking mutation & convergence (Tasks 3–4) | Finite N=8 strip across α = Δw / w₀ ∈ [0, 0.20] lowers Q and raises g_th(α) = g_th(0) + κ_th α²; mesh n_x ∈ {32, 48, 64} |
