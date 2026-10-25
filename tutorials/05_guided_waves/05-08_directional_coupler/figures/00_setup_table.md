| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 17.8 µm × 6.49 µm × 24.6 nm (z ∈ [97.7 nm, 122.3 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 725 × 266 × 1 (192850 cells) |
| Discretization | Δx & resolution | Δ_min = 24.2 nm, 22.2 cells/λ (in n_max = 2.84), 8.6 cells across feature (207 nm) |
| Boundary | -x face | CPML (10 cells, 245.3 nm) |
| Boundary | +x face | CPML (10 cells, 245.3 nm) |
| Boundary | -y face | CPML (10 cells, 244.4 nm) |
| Boundary | +y face | CPML (10 cells, 244.4 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Air lateral/upper clad (n=1.00, ε_r=1.00; SiO₂ BOX n_sub=1.45) (isotropic) |
| Material | Region 1: Si EIM core (n_Si=3.48, h=220 nm → n_eff,slab=2.8424, w_eff=343 nm) | ε_r = 8.08, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-10.5 µm, 10.5 µm] (d = 21 µm) |
| Material | Region 2: Si EIM core (n_Si=3.48, h=220 nm → n_eff,slab=2.8424, w_eff=343 nm) | ε_r = 8.08, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-10.5 µm, 10.5 µm] (d = 21 µm) |
| Source | 1 · ModeSource | pos: x = -8.27 µm, y = -2.00 µm, pol: TE₀ (E_y, H_z), pulse: f₀ = 195.2 THz, Δf = 18.89 THz |
| Monitor | 2 · ModeMonitor | Port 1 'in_bot' (S₁₁), pos: x = -8.15 µm, y = -2.00 µm, freqs: 51 freqs in [176.3, 214.1] THz |
| Monitor | 3 · ModeMonitor | Port 3 'out_top' (Cross S₃₁), pos: x = 8.15 µm, y = 2.00 µm, freqs: 51 freqs in [176.3, 214.1] THz |
| Monitor | 4 · ModeMonitor | Port 4 'out_bot' (Bar S₄₁), pos: x = 8.15 µm, y = -2.00 µm, freqs: 51 freqs in [176.3, 214.1] THz |
| Monitor | 5 · ModeMonitor | Port 2 'in_top' (S₂₁), pos: x = -8.15 µm, y = 2.00 µm, freqs: 51 freqs in [176.3, 214.1] THz |
| Monitor | 6 · FluxMonitor | 4-wall Poynting FluxMonitor box, pos: x ∈ [-8.85, 8.85] µm, y ∈ [-3.20, 3.20] µm, freqs: 51 freqs in [176.3, 214.1] THz |
| Variation | Dual-port quadrature routing (task_9) | Simultaneous ModeSource on in_bot (φ = 0) and in_top (φ = −π/2) via PhasedGaussianSource |
| Variation | Coupling-length sweep (task_10..task_13) | L_c ∈ {0, 2, 4, 6, 8} µm at θ_sw = 30° |
| Variation | Sidewall-angle sweep (task_14..task_19) | θ_sw ∈ {0°, 5°, 10°, 15°, 20°, 25°, 30°} (g_eff ∈ [80.0, 207.0] nm) |
| Variation | Graded non-uniform Cartesian mesh (task_20) | Mirror-symmetric graded Δy (10.00 cells across g_eff) & non-square Δx > Δy_min (119,637 cells vs 193,842 at L3, −38.3%) |
