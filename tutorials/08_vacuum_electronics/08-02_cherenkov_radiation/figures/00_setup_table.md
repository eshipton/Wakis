| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 22 mm × 22 mm × 42 mm (z ∈ [0 m, 42 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 68 × 68 × 131 (605744 cells) |
| Discretization | Δz & resolution | Δ_min = 320.6 µm, 20.8 cells/λ (in n_max = 1.50), 4.7 cells across feature (1.50 mm) |
| Boundary | -x face | CPML (8 cells, 2.59 mm) |
| Boundary | +x face | CPML (8 cells, 2.59 mm) |
| Boundary | -y face | CPML (8 cells, 2.59 mm) |
| Boundary | +y face | CPML (8 cells, 2.59 mm) |
| Boundary | -z face | CPML (8 cells, 2.56 mm) |
| Boundary | +z face | CPML (8 cells, 2.56 mm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Dielectric (n = 1.50) | ε_r = 2.25, μ_r = 1.00, model = Isotropic dielectric, z ∈ [2.60 mm, 39.4 mm] (d = 36.8 mm) |
| Source | 1 · Beam | pos: x = 0 mm, y = 0 mm, pol: J_z, pulse: f₀ = 25.00 GHz, Δf = 63.62 GHz |
| Monitor | 2 · FluxMonitor | FluxMonitor (r = 3.2 mm), pos: z ∈ [15.0, 27.0] mm, x ∈ [−3.2, +3.2] mm, freqs: 69 freqs in [12.0, 80.0] GHz |
| Monitor | 3 · Probe | Coulomb probe (r = 3.5 mm), pos: z = 21 mm, freqs: time-domain |
| Variation | Task 1: Dielectric Cherenkov vs Coulomb (§2, §4) | 3D homogeneous dielectric n = 1.50 (ε_r = 2.25), 22×22×42 mm, CPML (N_pml = 8) on all faces; super-threshold β = 0.90 (θ_c = 42.2°) vs sub-threshold β = 0.60, plus β ∈ [0.60, 0.96] sweep |
| Variation | Task 2: Dispersive Lorentz medium (§2.3, §4) | Single-pole Lorentz dielectric (ε_∞ = 1.0, Δε = 1.25, f₀ = 38 GHz, γ/2π = 1.5 GHz), β = 0.85; Cherenkov passband below f₀ vs Reststrahlen stopband above f₀ |
| Variation | Task 3: Vacuum numerical Cherenkov: Yee vs TETE (§2.4, §4) | Smooth PEC drift pipe 16×16×60 mm (PEC ±x, ±y, CPML ±z), vacuum (n = 1.00), ultra-relativistic bunch β = 0.999 (γ = 22.4, σ_z = 1.8 mm), Δx = Δy = 2Δz, magic step cΔt = Δz; scheme='yee' vs scheme='tete' |
| Variation | Task 4: 3D mesh convergence study (§4) | 3-level 3D grid refinement Δx = Δz ∈ {0.56, 0.42, 0.32} mm (11.9, 15.9, 20.8 cells/λ at 30 GHz in n = 1.50, σ_z = 1.2 mm) |
