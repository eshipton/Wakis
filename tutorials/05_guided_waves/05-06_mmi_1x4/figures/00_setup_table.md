| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 21.9 µm × 8.37 µm × 25.1 nm (z ∈ [-12.6 nm, 12.6 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 731 × 256 × 1 (187136 cells) |
| Discretization | Δx & resolution | Δ_min = 20 nm, 26.0 cells/λ (in n_max = 2.98), 20.0 cells across feature (400 nm) |
| Boundary | -x face | CPML (10 cells, 377 nm) |
| Boundary | +x face | CPML (10 cells, 380 nm) |
| Boundary | -y face | CPML (10 cells, 734 nm) |
| Boundary | +y face | CPML (10 cells, 734 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | SiO₂ cladding (n = 1.444, ε_r = 2.09) (isotropic) |
| Material | Region 1: Si–GaN–Si slot EIM (n_eff = 2.98) | ε_r = 8.91, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-2.50 µm, 20.7 µm] (d = 23.1 µm) |
| Source | 1 · ModeSource | pos: —, pol: —, pulse: f₀ = 193.6 THz, Δf = 6.2 THz |
| Monitor | 2 · ModeMonitor | Port 'in' ModeMonitor (S₁₁), pos: —, freqs: time-domain |
| Monitor | 3 · ModeMonitor | Ports 'out1'–'out4' ModeMonitors (S₂₁..S₅₁), pos: —, freqs: time-domain |
| Monitor | 4 · FluxBox | 2D Poynting FluxBox, pos: —, freqs: time-domain |
| Variation | kottke_coarse..fine (tasks 0–5) | param=cpw ∈ {10, 14, 20}, averaging ∈ {kottke, staircase}, note=Uniform resolution ladder at (W_MMI, L_MMI) = (4.90, 11.15) µm |
| Variation | kottke_reference (task 6) | param=cpw = 40 (1705 × 571 × 1 = 973,555 cells), note=Independent 2× fine Kottke reference |
| Variation | sweep_W* (tasks 7–36) | param=W_MMI ∈ [4.80, 5.20] µm × L_MMI ∈ [11.10, 11.35] µm, note=30-point 5×6 design sweep at 20 cpw (min EL at 5.00×11.20, min imb at 5.00×11.30 µm) |
| Variation | kottke_nonuniform (task 37) | param=Graded non-square Cartesian mesh (731 × 256 × 1 = 187,136 cells), note=Exact sidewall nodes (y = ±2.45 µm), dy_min = 20.0 nm in gaps, dx_max = 38.0 nm in leads |
