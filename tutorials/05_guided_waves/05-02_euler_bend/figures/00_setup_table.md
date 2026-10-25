| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 7.06 µm × 7.06 µm × 50 nm (z ∈ [-25 nm, 25 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 265 × 265 × 1 (70225 cells) |
| Discretization | Δx & resolution | Δ_min = 26.6 nm, 20.7 cells/λ (in n_max = 2.81), 15.0 cells across feature (400 nm) |
| Boundary | -x face | CPML (10 cells, 266.4 nm) |
| Boundary | +x face | CPML (10 cells, 266.4 nm) |
| Boundary | -y face | CPML (10 cells, 266.4 nm) |
| Boundary | +y face | CPML (10 cells, 266.4 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | SiO₂ cladding (n = 1.444, ε_r = 2.09) (isotropic) |
| Material | Region 1: Si core (w = 400 nm, 2D EIM n_eff = 2.813) | ε_r = 7.91, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-3 µm, 4.20 µm] (d = 7.20 µm) |
| Source | 1 · ModeSource | pos: —, pol: —, pulse: f₀ = 193.6 THz, Δf = 15.61 THz |
| Monitor | 2 · ModeMonitor | Port 1 ModeMonitor (S₁₁), pos: —, freqs: time-domain |
| Monitor | 3 · ModeMonitor | Port 2 ModeMonitor (S₂₁, +y), pos: —, freqs: time-domain |
| Monitor | 4 · FluxMonitor | 2D Poynting FluxMonitor box, pos: —, freqs: time-domain |
| Variation | kottke_L1..L3, staircase_L1..L3 (tasks 0–5) | param=cpw ∈ {10, 14, 20}, N_feat ∈ {4, 6, 8}, averaging ∈ {kottke, staircase}, note=Uniform square-cell resolution ladder at R_eff = 4.0 µm |
| Variation | kottke_reference (task 6) | param=cpw = 40, N_feat = 16 (509 × 509 × 1 = 259,081 cells), note=Independent 2× fine Kottke reference |
| Variation | sweep_circular_R*, sweep_euler_R* (tasks 7–14) | param=R_eff ∈ {2.5, 3.0, 3.5, 4.0} µm × {circular, hybrid Euler}, note=8-run radius sweep comparing circular vs hybrid Euler 90° bends |
| Variation | kottke_nonuniform (task 15) | param=Mirror-symmetric graded Cartesian mesh (238 × 238 × 1 = 56,644 cells), note=15.0 cells across w = 400 nm (Δu_core = 26.6 nm), Δu_lead = 32.0 nm, Δu_clad ≤ 68.0 nm (−19.3% cells, −43.5% time) |
