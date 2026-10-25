| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 8 nm × 8 nm × 240 nm (z ∈ [-108 nm, 132 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 2 × 2 × 240 (960 cells) |
| Discretization | Δz & resolution | Δ_min = 1 nm, 188.3 cells/λ (in n_max = 2.84), 40.0 cells across feature (40 nm) |
| Boundary | -x face | Periodic |
| Boundary | +x face | Periodic |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (15 cells, 15 nm) |
| Boundary | +z face | CPML (15 cells, 15 nm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Au thin film (J&C fit) | ε_r = 8.08, μ_r = 1.00, model = DispersiveMaterial, z ∈ [0 m, 40 nm] (d = 40 nm) |
| Source | 1 · CurrentSource | pos: z = -50 nm, pol: J_x (E_x), pulse: f₀ = 560.0 THz, Δf = 360.0 THz |
| Monitor | 2 · FluxMonitor | FluxMonitor (z = -25 nm), pos: z = -25 nm, freqs: 17 freqs in [336.1, 814.9] THz |
| Monitor | 3 · FluxMonitor | FluxMonitor (z = 65 nm), pos: z = 65 nm, freqs: 17 freqs in [336.1, 814.9] THz |
| Variation | Task 0 & Task 3 · Pole fitting & library audit | Non-FIT nonlinear least-squares pole fitting (fit_poles N = 1..6 on J&C Au/Ag; N = 1..5 Debye on Gabriel 1996 tissues) |
| Variation | Task 1b · Silver thin film (Ag, J&C 1972) | Ag slab (1 Drude + 2 Lorentz fit, d = 36 nm, Lz = 240 nm, Δz = 1.0 nm, λ ∈ [310, 900] nm) |
| Variation | Task 2 · Fused-silica etalon (SiO₂, Malitson 1965) | SiO₂ Fabry–Pérot slab (2-pole folded Lorentz fit, ε_∞ = 1.696, d = 600 nm, Lz = 1.60 µm, Δz = 5.0 nm, λ ∈ [350, 950] nm) |
| Variation | Task 4 · Human muscle tissue slab (Gabriel 1996) | Muscle tissue slab (4-pole Debye + σ_i = 0.20 S/m fit, d = 8.0 mm, Lz = 32.0 mm, Δz = 0.10 mm, f ∈ [0.6, 6.0] GHz) |
