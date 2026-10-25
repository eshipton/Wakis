| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 400 nm × 9.14 nm × 1.55 µm (z ∈ [-613 nm, 933 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 43 × 1 × 169 (7267 cells) |
| Discretization | Δz & resolution | Δ_min = 9.14 nm, 24.4 cells/λ (in n_max = 2.35), 19.7 cells across feature (180 nm) |
| Boundary | -x face | Periodic |
| Boundary | +x face | Periodic |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (14 cells, 128 nm) |
| Boundary | +z face | CPML (14 cells, 128 nm) |
| Material | Background | Air cladding / groove (n = 1.00, ε_r = 1.00) (isotropic) |
| Material | Region 1: N-LAF2 glass waveguide substrate (n_s = 1.74, ε_r = 3.03) | ε_r = 3.03, μ_r = 1.00, model = Isotropic dielectric, z ∈ [320 nm, 933 nm] (d = 613 nm) |
| Material | Region 2: Slanted TiO₂ ridge (α = 25° ≈ α_Bragg, f = 0.55, d = 320 nm, n = 2.35, ε_r = 5.52, Kottke subpixel) | ε_r = 5.52, μ_r = 1.00, model = Isotropic dielectric, z ∈ [0 m, 320 nm] (d = 320 nm) |
| Source | 1 · CurrentSource | pos: z = -374 nm (x ∈ [−200, +200] nm), pol: Ey (TE, θ_in = 0° normal & oblique AngleSweep), pulse: f₀ = 571.0 THz, Δf = 68.52 THz |
| Monitor | 2 · FloquetPort | Port 1 Reflected FloquetPort at z = -193 nm (R_m in air), pos: z = -193 nm (in air), freqs: f₀ = 571.0 THz (λ₀ = 525 nm) |
| Monitor | 3 · FloquetPort | Port 2 Transmitted FloquetPort at z = +537 nm (T₋₁ ≫ T₊₁ in glass), pos: z = +537 nm (in N-LAF2 glass), freqs: f₀ = 571.0 THz (λ₀ = 525 nm) |
| Variation | Slant angle & profile comparison (Task 0) | α ∈ [0°, 40°] (α_Bragg = 24.5°) and Binary (α = 0°) vs Sawtooth Blazed vs Slanted (α = 25°) |
| Variation | Groove depth & fill factor sweeps (Task 1) | d ∈ [100, 500] nm and f ∈ [0.25, 0.80] vs 28-slice multi-layer RCWA |
| Variation | Oblique FOV & broadband AngleSweep (Task 2) | θ_in ∈ [−20°, +20°] (Bloch ±x, TE & TM) and broadband λ ∈ [490, 565] nm at θ_in = 8° |
| Variation | Staircase vs Kottke conformal convergence (Task 3) | cpw ∈ {14, 18, 24, 32} cells/λ comparing staircase point-sampling vs 16×16 Kottke tensor smoothing |
