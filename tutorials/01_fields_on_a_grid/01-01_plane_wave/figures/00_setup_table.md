| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 35.7 nm × 35.7 nm × 4.71 µm (z ∈ [0 m, 4.71 µm]) |
| Domain | Grid cells | Nx × Ny × Nz = 1 × 1 × 132 (132 cells) |
| Discretization | Δz & resolution | Δ_min = 35.7 nm, 28.0 cells/λ (in n_max = 1.00), 28.0 cells across λ_min (1 µm) |
| Boundary | -x face | Periodic |
| Boundary | +x face | Periodic |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (10 cells, 357 nm) |
| Boundary | +z face | CPML (10 cells, 357 nm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Source | 1 · CurrentSource | pos: z = 1.38 µm, pol: J_x (E_x), pulse: f₀ = 299.8 THz, Δf = 104.9 THz |
| Source | 2 · TFSFSource | pos: z ∈ [1.14 µm, 3.57 µm], pol: E_x, pulse: f₀ = 299.8 THz, Δf = 104.9 THz |
| Monitor | 3 · DFTMonitor | DFTMonitor (z ∈ [1.98 µm, 3.48 µm]), pos: z ∈ [1.98 µm, 3.48 µm], freqs: 25 freqs in [224.8, 374.7] THz |
| Monitor | 4 · Probe | Probe z₁ (1.98 µm), pos: z = 1.98 µm, freqs: time-domain |
| Monitor | 5 · Probe | Probe z₂ (3.48 µm), pos: z = 3.48 µm, freqs: time-domain |
| Variation | Soft-source run (§1.2) | CurrentSource J_x sheet at z = 1.50 µm; bilateral ±z radiation |
| Variation | TFSF run (§1.2) | TFSFSource box z ∈ [1.12 µm, 3.38 µm]; unilateral +z injection |
| Variation | Dispersion sweep (§1.3) | cpw ∈ {10, 20, 40, 80} cells/λ₀; periodic z boundary |
| Variation | Courant sweep (§1.4) | CFL S ∈ {0.50, 0.99, 1.005, 1.02} |
