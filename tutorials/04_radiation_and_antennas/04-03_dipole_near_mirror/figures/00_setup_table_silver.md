| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 960 nm × 15 nm × 610 nm (z ∈ [-150 nm, 460 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 56 × 3 × 145 (24360 cells) |
| Discretization | Δx & resolution | Δ_min = 2.50 nm, 124.8 cells/λ (in n_max = 1.92), 9.8 cells across feature (24.4 nm) |
| Boundary | -x face | CPML (8 cells, 191 nm) |
| Boundary | +x face | CPML (8 cells, 191 nm) |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | PEC |
| Boundary | +z face | CPML (8 cells, 195.6 nm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Drude Ag mirror (δE = 24.4 nm) | ε_r = 3.70, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-480 nm, 480 nm] (d = 960 nm) |
| Source | 1 · CurrentSource | pos: x = 0 m, z = 47.5 nm, pol: J_y (E_y), pulse: f₀ = 499.7 THz, Δf = 209.9 THz |
| Monitor | 2 · FluxMonitor | Dipole FluxMonitor box (P_tot), pos: x ∈ [-10.0, 10.0] nm, z ∈ [42.5, 52.5] nm, freqs: f = 499.7 THz |
| Monitor | 3 · FluxMonitor | FluxMonitor (z = 2.50 nm), pos: z = 2.50 nm, freqs: f = 499.7 THz |
| Variation | Height & polarization sweep | param=h ∈ [20, 220] nm (kh ∈ [0.21, 2.30]), pol ∈ {'y', 'x', 'z'}, note=Benchmarked against 2D Chance–Prock–Silbey (CPS) Sommerfeld integral (F_tot, F_rad, F_nonrad, u_SPP = 1.0338) |
| Variation | Skin-depth resolution study | param=Δz_Ag ∈ [2.0, 24.0] nm (1.02 to 12.21 cells / δE), note=Electric-field skin depth δE = 24.41 nm, power skin depth δI = 12.21 nm at λ₀ = 600 nm |
