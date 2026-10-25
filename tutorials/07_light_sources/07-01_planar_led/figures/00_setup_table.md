| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 3.60 µm × 1 µm × 1.25 µm (z ∈ [-850 nm, 400 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 746 × 1 × 280 (208880 cells) |
| Discretization | Δx & resolution | Δ_min = 3.12 nm, 30.9 cells/λ (in n_max = 2.90), 8.0 cells across feature (25 nm) |
| Boundary | -x face | CPML (8 cells, 38.6 nm) |
| Boundary | +x face | CPML (8 cells, 38.6 nm) |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (8 cells, 38.1 nm) |
| Boundary | +z face | CPML (8 cells, 38.0 nm) |
| Material | Background | Air superstrate, ε = 1.00 (isotropic) |
| Material | Region 1: Sapphire (d_z = 300 nm, n = 1.80) | ε_r = 3.24, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.80 µm, 1.80 µm] (d = 3.60 µm) |
| Material | Region 2: n-AlGaN (d_z = 450 nm, n = 2.60) | ε_r = 6.76, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.80 µm, 1.80 µm] (d = 3.60 µm) |
| Material | Region 3: MQW active (d_z = 50 nm, n = 2.60, α = 10³ cm⁻¹) | ε_r = 6.76, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.80 µm, 1.80 µm] (d = 3.60 µm) |
| Material | Region 4: p-AlGaN (d_z = 50 nm, n = 2.60) | ε_r = 6.76, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.80 µm, 1.80 µm] (d = 3.60 µm) |
| Material | Region 5: p-GaN absorber (d_z = 50 nm, n = 2.90, α = 1.7×10⁵ cm⁻¹) | ε_r = 8.41, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.80 µm, 1.80 µm] (d = 3.60 µm) |
| Source | 1 · CurrentSource | pos: z = -75 nm, 9 sites in x ∈ [-1.05, 1.05] µm (5 shown), pol: TE (J_x) & TM (J_z), pulse: f₀ = 1070.7 THz, Δf = 192.7 THz |
| Monitor | 2 · Near2FarMonitor | Top Near2FarMonitor in air (z = +80 nm), pos: z = +80 nm, x ∈ [-1.75, 1.75] µm, freqs: f = 1070.7 THz |
| Monitor | 3 · FluxMonitor | Bottom + lateral Poynting FluxMonitors (z_bot = -0.80 µm), pos: z = -802 nm, freqs: f = 1070.7 THz |
| Variation | Task 0 — Semi-infinite dielectric slab validation | notes=n_slab ∈ {1.50, 2.00, 2.60} → Air (n = 1.00), z_dip = -180 nm, 4-face FluxMonitor box vs exact 2D Fresnel integral |
| Variation | Task 1 — p-GaN thickness & AlGaN index sweeps (Ryu 2014 Figs. 3 & 7) | notes=t_p-GaN ∈ {0, 25, 50, 100, 150, 200} nm (α = 1.7×10⁵ cm⁻¹, δ = 58.82 nm), n_AlGaN ∈ {2.50, 2.60, 2.70} |
| Variation | Task 2 — Incoherent dipole ensemble & Near2Far pattern | notes=N_pos = 5 lateral positions x_i ∈ {0.0, 0.25, 0.50, 0.75, 1.05} µm (9 symmetric sites) × TE (J_x) / TM (J_z) = 18 dipoles |
| Variation | Tasks 3–6 — Kottke subpixel smoothing vs staircase convergence ladder | notes=cpw ∈ {10 (demo, Δz = 25 nm), 14 (Δz = 4.17 nm), 20 (accepted, Δz = 3.125 nm, 18.8 cells/δ), 28 (reference, Δz = 2.50 nm, 23.5 cells/δ)} |
