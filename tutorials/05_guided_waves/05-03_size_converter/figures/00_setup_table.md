| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 12.7 µm × 11.6 µm × 25.4 nm (z ∈ [117.3 nm, 142.7 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 511 × 458 × 1 (234038 cells) |
| Discretization | Δx & resolution | Δ_min = 6.25 nm, 83.9 cells/λ (in n_max = 2.95), 8.0 cells across feature (50 nm) |
| Boundary | -x face | CPML (10 cells, 253.5 nm) |
| Boundary | +x face | CPML (10 cells, 253.4 nm) |
| Boundary | -y face | CPML (10 cells, 253.6 nm) |
| Boundary | +y face | CPML (10 cells, 253.6 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Air cladding (n = 1.00, ε_r = 1.00; SiO₂ BOX n_sub = 1.44) (isotropic) |
| Material | Region 1: Si 6-segment semi-lens (t_Si = 260 nm, n_eff = 2.9545, ε_r = 8.73) | ε_r = 8.73, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.40 µm, 11.2 µm] (d = 12.6 µm) |
| Source | 1 · ModeSource | pos: x = -1.03 µm, y ∈ [-0.90, 0.90] µm, pol: TE₀ (E_y, H_z), pulse: f₀ = 193.6 THz, Δf = 15.61 THz |
| Monitor | 2 · ModeMonitor | Port 'in' ModeMonitor (S₁₁), pos: x = -0.95 µm, y ∈ [-0.90, 0.90] µm, freqs: 51 freqs in [187.4, 199.9] THz |
| Monitor | 3 · ModeMonitor | Port 'out' ModeMonitor (TE₀,₂,₄,₆,₈ + FluxMonitor), pos: x = 10.55 µm, y ∈ [-4.72, 4.72] µm, freqs: 51 freqs in [187.4, 199.9] THz |
| Monitor | 4 · FluxMonitor | 2D Poynting FluxMonitor box, pos: x ∈ [-1.03, 10.80] µm, y ∈ [-5.35, 5.35] µm, freqs: 51 freqs in [187.4, 199.9] THz |
| Variation | Design 1: Linear & non-linear tapers (tasks 10–14) | param=w_in = 10.0 µm → w_out = 0.50 µm (t_Si = 110 nm, n_eff,slab = 2.2166), L_t ∈ [0, 100] µm, note=Linear, parabolic, exponential, and cosine tapers vs Milton–Burns adiabatic scaling & abrupt-step overlap |
| Variation | Design 2: Graded-thickness Luneburg lens (task 9) | param=D_lens = 11.0 µm (R = 5.5 µm, 60 CircularPolySlab shells, t(r) ∈ [110, 356.1] nm), note=n_eff(r) = n_edge √(2 − (r/R)²) ∈ [2.2166, 3.1348] calibrated via 1D SOI slab cubic polynomial fit |
| Variation | Resolution & subpixel ladder (tasks 0–8) | param=(cpw, N_feat) ∈ {(10, 1) demo, (10, 4) L1, (14, 6) L2, (20, 8) L3, (40, 16) R}, note=Kottke subpixel smoothing vs staircasing across the L₃ = 50 nm Segment 3 flare |
| Variation | Graded non-uniform Cartesian mesh (task 15) | param=NonUniformSemilensPIC (424 × 407 × 1 = 172,568 cells vs 235,008 at L3), note=8.0 cells across L₃ = 50 nm (Δx_min = 6.25 nm), non-square lead cells (−26.6% cells, −33.2% time) |
