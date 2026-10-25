| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 9.18 µm × 3.42 µm × 210 nm (z ∈ [-105 nm, 105 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 345 × 127 × 1 (43815 cells) |
| Discretization | Δx & resolution | Δ_min = 21 nm, 28.6 cells/λ (in n_max = 1.00), 4.0 cells across feature (84 nm) |
| Boundary | -x face | CPML (10 cells, 271.7 nm) |
| Boundary | +x face | CPML (10 cells, 271.7 nm) |
| Boundary | -y face | CPML (10 cells, 272 nm) |
| Boundary | +y face | CPML (10 cells, 272 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: PEC outer jaw | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-4.59 µm, -2.88 µm] (d = 1.71 µm) |
| Material | Region 2: PEC grating bar | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-2.52 µm, -1.80 µm] (d = 720 nm) |
| Material | Region 3: PEC grating bar | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-1.44 µm, -720 nm] (d = 720 nm) |
| Material | Region 4: PEC grating bar | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-360 nm, 360 nm] (d = 720 nm) |
| Material | Region 5: PEC grating bar | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [720 nm, 1.44 µm] (d = 720 nm) |
| Material | Region 6: PEC grating bar | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [1.80 µm, 2.52 µm] (d = 720 nm) |
| Material | Region 7: PEC outer jaw | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [2.88 µm, 4.59 µm] (d = 1.71 µm) |
| Source | 1 · CurrentSource | pos: y = -510 nm, pol: J_x (E_x), pulse: f₀ = 499.7 THz, Δf = 224.8 THz |
| Monitor | 2 · FluxMonitor | FluxMonitor (y = 94.5 nm), pos: y = 94.5 nm, freqs: f = 499.7 THz |
| Monitor | 3 · Near2FarMonitor | Near2FarMonitor, pos: x ∈ [-4.59 µm, 4.59 µm], freqs: f = 499.7 THz |
| Variation | Slit count sweep (patterns task) | N ∈ {2, 3, 4, 6, 8}, a = 360 nm (0.6λ), d = 1.08 µm (1.8λ), t = 84 nm (0.14λ) |
| Variation | Rayleigh doublet (resolving_power task) | N = 6, order m = +1 (R = 6), broadband DFT at Δλ/λ ∈ {0.55/R, 1.00/R, 1.45/R} |
| Variation | Periodic unit cell (periodic_vs_finite task) | N → ∞ single period x ∈ [-d/2, +d/2] with periodic ±x BCs and DiffractionMonitor |
| Variation | Resolution sweep (resolution task) | N = 4, cpw ∈ {12, 16, 20, 28, 36} |
