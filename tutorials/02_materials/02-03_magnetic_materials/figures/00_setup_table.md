| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 10 mm × 10 mm × 600 mm (z ∈ [-270 mm, 330 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 2 × 2 × 480 (1920 cells) |
| Discretization | Δz & resolution | Δ_min = 1.25 mm, 171.3 cells/λ (in n_max = 1.00), 40.0 cells across feature (50 mm) |
| Boundary | -x face | Periodic |
| Boundary | +x face | Periodic |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (15 cells, 18.7 mm) |
| Boundary | +z face | CPML (15 cells, 18.7 mm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Lorentz-mu ferrite slab | ε_r = 1.00, μ_r = 1.00, model = DispersiveMaterial, z ∈ [0 m, 50 mm] (d = 50 mm) |
| Source | 1 · CurrentSource | pos: z = -160 mm, pol: J_x (E_x), pulse: f₀ = 1.4 GHz, Δf = 950.0 MHz |
| Monitor | 2 · FluxMonitor | FluxMonitor (z = -80 mm), pos: z = -80 mm, freqs: 37 freqs in [0.3, 2.7] GHz |
| Monitor | 3 · FluxMonitor | FluxMonitor (z = 110 mm), pos: z = 110 mm, freqs: 37 freqs in [0.3, 2.7] GHz |
| Variation | Task 0 · Literature ferrites & periodic H-edge ADE check | Analytic LLG / Snoek / Acher / hesitivity audit + periodic 2×2×16 GridFIT3D H-edge ADE check against mu_discrete(f, dt) |
| Variation | Task 2 · Debye-μ relaxation ferrite slab | Debye magnetic relaxation slab (μ_∞ = 1.0, Δμ = 4.0, f_rel = 1.0 GHz, ε_r = 1.5, d = 40 mm, Lz = 560 mm, Δz = 1.25 mm) |
| Variation | Task 3 · Impedance-matched magneto-dielectric slab (ε(ω) = μ(ω)) | Matched Lorentz ε(ω) = μ(ω) absorber slab (ε_∞ = μ_∞ = 1.0, Δε = Δμ = 2.5, f₀ = 1.50 GHz, γ/2π = 0.40 GHz, d = 50 mm, Δz = 1.0 mm) |
| Variation | Task 4 · 3D PEC ferrite-loaded rectangular cavity | Closed PEC cavity (120 × 60 × 120 mm³, 16 × 8 × 16 cells, Lorentz-μ ferrite μ_∞ = 1.0, Δμ = 1.25, f_r ∈ [1.20, 2.50] GHz) |
