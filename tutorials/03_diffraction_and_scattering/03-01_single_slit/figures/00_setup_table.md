| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 7.10 µm × 4.10 µm × 225 nm (z ∈ [-112.5 nm, 112.5 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 284 × 165 × 1 (46860 cells) |
| Discretization | Δy & resolution | Δ_min = 22.5 nm, 26.7 cells/λ (in n_max = 1.00), 4.0 cells across feature (90 nm) |
| Boundary | -x face | CPML (10 cells, 250 nm) |
| Boundary | +x face | CPML (10 cells, 250 nm) |
| Boundary | -y face | CPML (10 cells, 248.5 nm) |
| Boundary | +y face | CPML (10 cells, 249 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: PEC screen (left jaw) | ε_r = inf, μ_r = 1.00, model = PEC, y ∈ [-45 nm, 45 nm] (d = 90 nm) |
| Material | Region 2: PEC screen (right jaw) | ε_r = inf, μ_r = 1.00, model = PEC, y ∈ [-45 nm, 45 nm] (d = 90 nm) |
| Source | 1 · CurrentSource | pos: y = -630 nm, pol: J_z (E_z), pulse: f₀ = 499.7 THz, Δf = 249.8 THz |
| Monitor | 2 · FluxMonitor | FluxMonitor (y = 0 m), pos: y = 0 m, freqs: f = 499.7 THz |
| Monitor | 3 · Near2FarMonitor | Near2FarMonitor (y = 101 nm), pos: y = 101 nm, freqs: f = 499.7 THz |
| Monitor | 4 · DFTMonitor | DFTMonitor (y = 1.24 µm), pos: y = 1.24 µm, freqs: f = 499.7 THz |
| Variation | Task 1 · Plane-wave source & Fresnel near-field sweep | Full-width vs PML-truncated source benchmark + Fresnel line monitors at N_F ∈ {2.0, 1.0, 0.5, 0.25} (a = 2λ, E_z) |
| Variation | Task 2 · Far-field angular patterns (Near2FarMonitor) | Slit width sweep a/λ ∈ {4.0, 2.0, 1.0, 0.6} for both E_z (TM_z, RS-I) and H_z (TE_z, RS-II) polarizations |
| Variation | Task 3 · Subwavelength transmission & waveguide cutoff | Normalized transmission τ(a/λ) for a/λ ∈ [0.30, 2.00] comparing thick PEC screen (t = 0.20λ) vs zero-thickness sheet (t = 0) |
| Variation | Task 4 · Mesh resolution convergence study | Background resolution sweep cpw ∈ {12, 16, 20, 28, 36} cells/λ for a = 2λ, t = 0.15λ |
