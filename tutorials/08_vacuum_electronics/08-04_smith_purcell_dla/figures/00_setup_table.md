| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 4.40 µm × 50 nm × 22.7 µm (z ∈ [-11.3 µm, 11.3 µm]) |
| Domain | Grid cells | Nx × Ny × Nz = 79 × 1 × 462 (36498 cells) |
| Discretization | Δz & resolution | Δ_min = 30.8 nm, 46.7 cells/λ (in n_max = 1.00), 8.1 cells across feature (250 nm) |
| Boundary | -x face | CPML (10 cells, 600 nm) |
| Boundary | +x face | CPML (10 cells, 672 nm) |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (10 cells, 491 nm) |
| Boundary | +z face | CPML (10 cells, 491 nm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: PEC grating | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [-11.3 µm, 11.3 µm] (d = 22.7 µm) |
| Source | 1 · Beam | pos: x = 250 nm, y = 0 m, pol: J_z, pulse: f₀ = 143.6 THz, Δf = 78.99 THz |
| Monitor | 2 · DFTMonitor | DFTMonitor (z ∈ [-9.35 µm, 9.35 µm]), pos: z ∈ [-9.35 µm, 9.35 µm], freqs: 116 freqs in [73.1, 461.2] THz |
| Variation | Part 1 · Dielectric grating | SiO₂ grating (ε_r = 2.10, n = 1.45), adding internal Cherenkov leakage + guided slab modes |
| Variation | Part 1 · Uniform Fine baseline | Uniform Δx = Δz = 50.0 nm (88 × 462 = 40,656 cells) vs Graded 79 × 462 = 36,498 cells (−10.2%) |
| Variation | Part 2 · Inverse process (DLA) | Dual SiO₂ grating (λ_p = λ_0 = 800 nm, gap 2a = 400 nm, periodic ±z, ±x CPML, 7,600 graded vs 8,800 uniform cells) |
