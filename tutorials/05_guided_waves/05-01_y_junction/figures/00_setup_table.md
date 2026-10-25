| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 6.49 µm × 4.71 µm × 18.8 nm (z ∈ [-9.41 nm, 9.41 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 298 × 162 × 1 (48276 cells) |
| Discretization | Δx & resolution | Δ_min = 16.7 nm, 32.7 cells/λ (in n_max = 2.85), 12.0 cells across feature (200 nm) |
| Boundary | -x face | CPML (10 cells, 244.7 nm) |
| Boundary | +x face | CPML (10 cells, 243 nm) |
| Boundary | -y face | CPML (10 cells, 504 nm) |
| Boundary | +y face | CPML (10 cells, 504 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | SiO₂ cladding (n = 1.444, ε_r = 2.09) (isotropic) |
| Material | Region 1: Si core (n = 3.476, 2D EIM n_eff = 2.848) | ε_r = 8.11, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.40 µm, 5.20 µm] (d = 6.60 µm) |
| Source | 1 · ModeSource | pos: —, pol: —, pulse: f₀ = 193.6 THz, Δf = 15.61 THz |
| Monitor | 2 · ModeMonitor | Port 'in' ModeMonitor (S₁₁), pos: —, freqs: time-domain |
| Monitor | 3 · ModeMonitor | Port 'out1' ModeMonitor (S₂₁), pos: —, freqs: time-domain |
| Monitor | 4 · ModeMonitor | Port 'out2' ModeMonitor (S₃₁), pos: —, freqs: time-domain |
| Monitor | 5 · FluxMonitor | 2D Poynting FluxMonitor box, pos: —, freqs: time-domain |
| Variation | kottke_coarse..fine (tasks 0–5) | param=cpw ∈ {14, 20, 28}, averaging ∈ {kottke, staircase}, note=Uniform square-cell resolution ladder (14, 20, 28 cells/λ) |
| Variation | kottke_reference (task 6) | param=cpw = 56 (657 × 412 × 1 = 270,684 cells), note=Independent 2× fine Kottke reference |
| Variation | kottke_nonuniform (task 7) | param=Graded non-square Cartesian mesh (298 × 162 × 1 = 48,276 cells), note=12 cells across 200 nm tip gap (Δy_tip = 16.7 nm), Δx_lead = 24.5 nm, Δy_clad ≤ 50.4 nm (−34% cells) |
