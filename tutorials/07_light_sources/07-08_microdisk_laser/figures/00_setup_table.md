| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 3.58 µm × 3.58 µm × 280 nm (z ∈ [-140 nm, 140 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 128 × 128 × 1 (16384 cells) |
| Discretization | Δx & resolution | Δ_min = 28 nm, 26.6 cells/λ (in n_max = 2.00), 7.9 cells across feature (220 nm) |
| Boundary | -x face | CPML (10 cells, 280 nm) |
| Boundary | +x face | CPML (10 cells, 280 nm) |
| Boundary | -y face | CPML (10 cells, 280 nm) |
| Boundary | +y face | CPML (10 cells, 280 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Microdisk (n = 2.0, R = 1.0 µm) | ε_r = 4.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1 µm, 1 µm] (d = 2 µm) |
| Material | Region 2: Bus waveguide (w = 220 nm, g = 300 nm) | ε_r = 4.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.79 µm, 1.79 µm] |
| Source | 1 · CurrentSource | pos: x = 720 nm, y = 450 nm, pol: J_z (E_z), pulse: f₀ = 201.1 THz, Δf = 16.08 THz |
| Monitor | 2 · DFTMonitor | DFTMonitor (x ∈ [-1.30 µm, 1.30 µm], y ∈ [-1.30 µm, 1.30 µm]), pos: x ∈ [-1.30 µm, 1.30 µm], y ∈ [-1.30 µm, 1.30 µm], freqs: f = 201.1 THz |
| Monitor | 3 · Probe | WGM rim probe, pos: (x, y) = (-720 nm, 450 nm), freqs: time-domain |
| Monitor | 4 · Probe | Bus outport probe, pos: (x, y) = (850 nm, -1.41 µm), freqs: time-domain |
| Variation | Part 1 · Cold cavity | Passive subpixel vs staircase microdisk (R = 1.0 µm, n = 2.0), TM & TE WGMs (m = 4..9, q = 1, 2) |
| Variation | Part 2 · Linear gain threshold | Unsaturated Lorentz gain pole scanning g across g_th = n ω / (c Γ Q) for m = 4..8 |
| Variation | Part 3 · FourLevelGain lasing from noise | Spontaneous-emission noise startup (spontaneous_emission=True, s = 100) in annular ring (r ∈ [0.55 R, R]) vs full-disk & core pump, coupled to 220 nm bus across 300 nm gap |
