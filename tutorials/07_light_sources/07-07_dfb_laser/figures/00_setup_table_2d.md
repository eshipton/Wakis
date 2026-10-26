| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 800 nm × 691 nm × 8.62 µm (z ∈ [-4.31 µm, 4.31 µm]) |
| Domain | Grid cells | Nx × Ny × Nz = 40 × 1 × 624 (24960 cells) |
| Discretization | Δz & resolution | Δ_min = 13.8 nm, 36.2 cells/λ (in n_max = 3.00), 10.0 cells across feature (138.2 nm) |
| Boundary | -x face | PEC |
| Boundary | +x face | PEC |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | Mur ABC |
| Boundary | +z face | Mur ABC |
| Material | Background | Dielectric waveguide cladding (n_clad = 2.40, ε_r = 5.76) (isotropic) |
| Material | Region 1: Corrugated active slab core (n_core = 3.0, d = 320 nm, φ = -π/2) | ε_r = 9.00, μ_r = 1.00, model = Isotropic dielectric, z ∈ [-4.15 µm, 0 m] (d = 4.15 µm) |
| Material | Region 2: Corrugated active slab core (n_core = 3.0, φ = +π/2, λ/4 shift at z = 0) | ε_r = 9.00, μ_r = 1.00, model = Isotropic dielectric, z ∈ [0 m, 4.15 µm] (d = 4.15 µm) |
| Source | 1 · CurrentSource | pos: z = 0 m, pol: J_y (E_y), pulse: f₀ = 199.9 THz, Δf = 15.99 THz |
| Monitor | 2 · ResonanceMonitor | ResonanceMonitor (z = -1.38 µm), pos: z = -1.38 µm, freqs: time-domain |
