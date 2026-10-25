| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 120 mm × 60 mm × 120 mm (z ∈ [0 m, 120 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 16 × 8 × 16 (2048 cells) |
| Discretization | Δz & resolution | Δ_min = 7.50 mm, 22.6 cells/λ (in n_max = 1.00), 8.0 cells across feature (60 mm) |
| Boundary | -x face | PEC |
| Boundary | +x face | PEC |
| Boundary | -y face | PEC |
| Boundary | +y face | PEC |
| Boundary | -z face | PEC |
| Boundary | +z face | PEC |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Lorentz-mu ferrite cavity fill | ε_r = 1.00, μ_r = 1.00, model = DispersiveMaterial, z ∈ [0 m, 120 mm] (d = 120 mm) |
| Source | 1 · CurrentSource | pos: z = 42 mm, x = 78 mm, pol: J_y (E_y), pulse: f₀ = 1.8 GHz, Δf = 1.5 GHz |
| Monitor | 2 · Probe | TE_101 mode ring-down probe (E_y), pos: z = 78 mm, freqs: time-domain |
| Variation | Vacuum reference cavity | Unloaded PEC cavity (μ = 1.0, bare TE_101 resonance at f_vac = 1.7665 GHz) |
| Variation | Static permeability reference | Non-dispersive ferrite cavity (μ = μ_s = 2.25, static resonance at f_vac / √μ_s = 1.1777 GHz) |
| Variation | Polariton avoided-crossing sweep | Lorentz-μ ferrite with f_r ∈ {1.20, 1.50, 1.76, 2.10, 2.50} GHz, γ_m/2π = 25 MHz |
