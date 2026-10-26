| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 300 mm × 300 mm × 250 mm (z ∈ [-125 mm, 125 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 48 × 48 × 34 (78336 cells) |
| Discretization | Δx & resolution | Δ_min = 2.50 mm, 35.5 cells/λ (in n_max = 4.51), 4.0 cells across feature (10 mm) |
| Boundary | -x face | CPML (6 cells, 79.2 mm) |
| Boundary | +x face | CPML (6 cells, 79.2 mm) |
| Boundary | -y face | CPML (6 cells, 79.2 mm) |
| Boundary | +y face | CPML (6 cells, 79.2 mm) |
| Boundary | -z face | CPML (6 cells, 73.4 mm) |
| Boundary | +z face | CPML (6 cells, 73.4 mm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Lorentz magnetic torus (-x cross-section, μ_DC = 20) | ε_r = 1.00, μ_r = 20.30, model = Isotropic dielectric, x ∈ [-20 mm, -10 mm] (d = 10 mm) |
| Material | Region 2: Lorentz magnetic torus (+x cross-section, μ_DC = 20) | ε_r = 1.00, μ_r = 20.30, model = Isotropic dielectric, x ∈ [10 mm, 20 mm] (d = 10 mm) |
| Source | 1 · CurrentSource | pos: x = 0 m, z = 0 m, pol: J_z (E_z), pulse: f₀ = 750.0 MHz, Δf = 220.0 MHz |
| Monitor | 2 · Near2FarMonitor | Near2FarMonitor, pos: x ∈ [-150 mm, 150 mm], freqs: f = 750.0 MHz |
| Monitor | 3 · DFTMonitor | DFTMonitor (x ∈ [-22.5 mm, 22.5 mm], z ∈ [-7.50 mm, 7.50 mm]), pos: x ∈ [-22.5 mm, 22.5 mm], z ∈ [-7.50 mm, 7.50 mm], freqs: f = 750.0 MHz |
| Variation | Nominal closed torus | N = 8 in-phase poloidal windings, μ_DC = 20, Δ_core = 2.5 mm |
| Variation | Ideal magnetic current ring | Azimuthal MagneticCurrentSource ring at ρ = R |
| Variation | Broken-symmetry counter-cases | 60° core gap, 7/8 feeds, Mode 1 antiphase, helical turn |
