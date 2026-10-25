| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 200 nm × 200 nm × 16 µm (z ∈ [-7.20 µm, 8.80 µm]) |
| Domain | Grid cells | Nx × Ny × Nz = 2 × 2 × 640 (2560 cells) |
| Discretization | Δz & resolution | Δ_min = 25 nm, 272.9 cells/λ (in n_max = 2.31), 100.0 cells across feature (2.50 µm) |
| Boundary | -x face | Periodic |
| Boundary | +x face | Periodic |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (15 cells, 375 nm) |
| Boundary | +z face | CPML (15 cells, 375 nm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: GaN Lorentz slab | ε_r = 5.35, μ_r = 1.00, model = DispersiveMaterial, z ∈ [0 m, 2.50 µm] (d = 2.50 µm) |
| Source | 1 · CurrentSource | pos: z = -4.50 µm, pol: J_x (E_x), pulse: f₀ = 19.00 THz, Δf = 6.5 THz |
| Monitor | 2 · FluxMonitor | FluxMonitor (z = -2.20 µm), pos: z = -2.20 µm, freqs: 141 freqs in [10.0, 28.0] THz |
| Monitor | 3 · FluxMonitor | FluxMonitor (z = 4.70 µm), pos: z = 4.70 µm, freqs: 141 freqs in [10.0, 28.0] THz |
| Variation | Task 0 · Bulk 1-cell periodic ADE & KK/f-sum | 1 × 1 × 1 periodic cell (k = 0 ODE testbench, Δt = 10 ps) + analytical Kramers–Kronig & TRK f-sum integrals |
| Variation | Task 2 · Drude ENZ slab (drude_enz_slab) | Drude carrier slab (ε_∞ = 4.0, f_ENZ = 2.0 THz, d = 40 µm, Lz = 320 µm, Δz = 0.5 µm, f ∈ [0.8, 3.5] THz) |
| Variation | Task 3 · Debye liquid-water slab (debye_water_slab) | Debye orientational slab (ε_∞ = 4.9, Δε = 75.2, τ = 9.3 ps, d = 0.5 mm, Lz = 6.0 mm, Δz = 10 µm, f ∈ [2, 40] GHz) |
| Variation | Task 4 · Temporal O(Δt²) convergence (dt_convergence_and_stability) | Lorentz slab (d = 40 mm, Lz = 480 mm, Δz ∈ [1.0, 3.2] mm, f ∈ [0.5, 2.5] GHz) + ADE root locus (ω₀Δt < 2) |
