| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 17.9 nm × 17.9 nm × 2.84 µm (z ∈ [0 m, 2.84 µm]) |
| Domain | Grid cells | Nx × Ny × Nz = 1 × 1 × 159 (159 cells) |
| Discretization | Δz & resolution | Δ_min = 17.9 nm, 28.0 cells/λ (in n_max = 2.00), 35.0 cells across feature (625 nm) |
| Boundary | -x face | Periodic |
| Boundary | +x face | Periodic |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (10 cells, 178.6 nm) |
| Boundary | +z face | CPML (10 cells, 178.6 nm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Dielectric slab (n = 2.0) | ε_r = 4.00, μ_r = 1.00, model = Isotropic dielectric, z ∈ [1 µm, 1.62 µm] (d = 625 nm) |
| Source | 1 · CurrentSource | pos: z = 536 nm, pol: J_x (E_x), pulse: f₀ = 299.8 THz, Δf = 113.9 THz |
| Monitor | 2 · FluxMonitor | FluxMonitor (z = 857 nm), pos: z = 857 nm, freqs: 41 freqs in [179.9, 419.7] THz |
| Monitor | 3 · FluxMonitor | FluxMonitor (z = 1.97 µm), pos: z = 1.97 µm, freqs: 41 freqs in [179.9, 419.7] THz |
| Variation | Run 1a: Reference run (§1.1) | Vacuum everywhere (ε_r = 1.00); records incident flux Φ_ref(f) and E_ref(z_r, t) |
| Variation | Run 1b: Slab run (§1.1–1.2) | Dielectric slab ε_r = 4.00 (n = 2.00, d = 0.625 µm); R = 1 − Φ_r/Φ_ref, T = Φ_t/Φ_ref |
| Variation | Run 2: Oblique / Brewster scan (§1.3) | 2D (x, z) domain; Bloch ±x boundary (k_x = k₀ sin θ_i), θ_i ∈ [0°, 75°], glass n = 1.50 (θ_B = 56.31°) |
| Variation | Run 3: Boundary benchmark (§1.4) | Vacuum domain; +z test boundary ∈ {CPML (N_pml = 4, 8, 12, 16), Mur ABC, PEC} |
| Variation | Run 4: Subpixel convergence (§1.5) | Off-grid slab d = 0.655 µm; Kottke tensor vs staircase at cpw ∈ {10, 15, 20, 30, 40} |
