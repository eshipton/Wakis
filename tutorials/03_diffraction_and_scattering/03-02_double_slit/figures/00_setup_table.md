| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 7.15 µm × 3.66 µm × 225 nm (z ∈ [-112.5 nm, 112.5 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 262 × 136 × 1 (35632 cells) |
| Discretization | Δy & resolution | Δ_min = 22.5 nm, 26.7 cells/λ (in n_max = 1.00), 4.0 cells across feature (90 nm) |
| Boundary | -x face | CPML (10 cells, 272.7 nm) |
| Boundary | +x face | CPML (10 cells, 272.7 nm) |
| Boundary | -y face | CPML (10 cells, 270.5 nm) |
| Boundary | +y face | CPML (10 cells, 271 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: PEC screen (left jaw) | ε_r = inf, μ_r = 1.00, model = PEC, y ∈ [-45 nm, 45 nm] (d = 90 nm) |
| Material | Region 2: PEC central bridge (b = d - a) | ε_r = inf, μ_r = 1.00, model = PEC, y ∈ [-45 nm, 45 nm] (d = 90 nm) |
| Material | Region 3: PEC screen (right jaw) | ε_r = inf, μ_r = 1.00, model = PEC, y ∈ [-45 nm, 45 nm] (d = 90 nm) |
| Source | 1 · CurrentSource | pos: y = -510 nm, pol: J_x (E_x), pulse: f₀ = 499.7 THz, Δf = 249.8 THz |
| Monitor | 2 · FluxMonitor | FluxMonitor (y = 101.2 nm), pos: y = 101.2 nm, freqs: f = 499.7 THz |
| Monitor | 3 · Near2FarMonitor | Near2FarMonitor (y = 101 nm), pos: y = 101 nm, freqs: f = 499.7 THz |
| Monitor | 4 · DFTMonitor | DFTMonitor (y = 2.14 µm), pos: y = 2.14 µm, freqs: f = 499.7 THz |
| Variation | Task 1 (Missing orders & 2D near field) | a = 1.5λ₀ (900 nm), d/a ∈ {2, 3, 4} (d ∈ {1.80, 2.70, 3.60} µm), t = 0.14λ₀ (84 nm), E ⊥ slit edges (H_z, J_x source) |
| Variation | Task 2 (Fringe spacing scaling Δ(sin θ) = λ₀/d) | a = 0.8λ₀ (480 nm), d/λ₀ ∈ {1.8, 2.5, 3.2, 4.0}, t = 0.14λ₀ (84 nm), E ∥ slit edges (E_z, J_z source) |
| Variation | Task 3 (Inter-slit electromagnetic coupling) | a = 0.55λ₀ (330 nm), d/λ₀ ∈ [0.72, 2.25] (bridge b = d - a ∈ [0.17λ₀, 1.70λ₀]), t = 0.18λ₀ (108 nm), both E_z and H_z polarizations |
| Variation | Task 4 (Mesh convergence study) | a = 1.0λ₀ (600 nm), d = 3.0λ₀ (1.80 µm), t = 0.15λ₀ (90 nm), cpw ∈ {12, 16, 20, 28, 36} |
