| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 13.8 µm × 6.39 µm × 26.3 nm (z ∈ [-13.2 nm, 13.2 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 424 × 170 × 1 (72080 cells) |
| Discretization | Δx & resolution | Δ_min = 25 nm, 21.7 cells/λ (in n_max = 2.85), 8.0 cells across feature (200 nm) |
| Boundary | -x face | CPML (10 cells, 378 nm) |
| Boundary | +x face | CPML (10 cells, 378 nm) |
| Boundary | -y face | CPML (10 cells, 697 nm) |
| Boundary | +y face | CPML (10 cells, 697 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | SiO₂ cladding (n = 1.44, ε_r = 2.07) (isotropic) |
| Material | Region 1: SOI 220 nm EIM (n_eff = 2.85) | ε_r = 8.13, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-7.50 µm, 7.50 µm] (d = 15 µm) |
| Source | 1 · ModeSource | pos: —, pol: —, pulse: f₀ = 193.6 THz, Δf = 6.2 THz |
| Monitor | 2 · ModeMonitor | Ports 'in1'/'in2' ModeMonitors (S₁₁, S₂₁), pos: —, freqs: time-domain |
| Monitor | 3 · ModeMonitor | Ports 'out1' (bar) / 'out2' (cross) ModeMonitors (S₃₁, S₄₁), pos: —, freqs: time-domain |
| Monitor | 4 · FluxBox | 2D Poynting FluxBox, pos: —, freqs: time-domain |
| Variation | kottke_L1..L3 / staircase_L1..L3 (tasks 0–5) | param=L1=(10 cpw, N=4), L2=(14, 6), L3=(20, 8), note=Resolution ladder with throat-gap mesh_overrides at L_mmi = 3.00 µm |
| Variation | kottke_reference (task 6) | param=R=(40 cpw, N=16) (1016 × 403 × 1 = 409,448 cells), note=Independent 2× fine Kottke reference |
| Variation | length_*um (tasks 7–10) | param=L_mmi ∈ {2.50, 3.00, 3.25, 3.50, 3.80} µm at L3=(20, 8), note=General-interference sweep (2D EIM 50/50 split at L_mmi = 3.25 µm) |
| Variation | kottke_nonuniform (task 11) | param=Graded non-square Cartesian mesh (425 × 171 × 1 = 72,675 cells), note=Exact nodes at x = ±1.50 µm, y = ±0.10, ±0.50 µm; 8.0 cells across 200 nm gap |
