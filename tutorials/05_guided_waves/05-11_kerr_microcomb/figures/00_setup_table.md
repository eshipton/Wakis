| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 5.60 µm × 5.60 µm × 200 nm (z ∈ [-100 nm, 100 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 199 × 204 × 2 (81192 cells) |
| Discretization | Δx & resolution | Δ_min = 22 nm, 30.4 cells/λ (in n_max = 2.09), 7.3 cells across feature (160 nm) |
| Boundary | -x face | CPML (10 cells, 341.6 nm) |
| Boundary | +x face | CPML (10 cells, 341.6 nm) |
| Boundary | -y face | CPML (10 cells, 341.6 nm) |
| Boundary | +y face | CPML (10 cells, 243 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Kerr Bus Waveguide | ε_r = 4.35, μ_r = 1.00, model = KerrMaterial, x ∈ [-2.80 µm, 2.80 µm] (d = 5.60 µm) |
| Material | Region 2: Kerr Microring Core | ε_r = 4.35, μ_r = 1.00, model = KerrMaterial, x ∈ [-2.30 µm, 2.30 µm] (d = 4.60 µm) |
| Material | Region 3: Inner Air Cladding | ε_r = 1.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.70 µm, 1.70 µm] (d = 3.40 µm) |
| Source | 1 · CurrentSource | pos: x = -1.80 µm, pol: J_z (E_z), pulse: f₀ = 214.5 THz, Δf = 35.00 THz |
| Monitor | 2 · DFTMonitor | Through Port (DFTMonitor), pos: x = 1.80 µm, freqs: 41 freqs in [170.0, 250.0] THz |
| Monitor | 3 · ResonanceMonitor | Ring WGM Probe (ResonanceMonitor), pos: x = 0 m, freqs: time-domain |
| Variation | Cold-cavity dispersion extraction (task_0) | Azimuthal orders m ∈ {14..20} (m₀ = 17, f₀ = 214.5 THz), anomalous (D₂ > 0) vs normal (D₂ < 0) |
| Variation | Bus-coupled Q extraction & Kerr shift (task_1, task_2) | 160 nm bus–ring gap (Q₀, Q_ex, κ) and short 2D FIT Kerr ring resonance shift vs V_eff,NL |
| Variation | 3-level convergence + graded mesh (task_3) | Uniform Δx ∈ {30, 25, 20} nm (≥ 20 cells/λ_core) + locally refined graded mesh & Richardson extrapolation |
| Variation | FIT-informed Lugiato–Lefever microcomb (task_4) | 256-mode split-step Fourier scan (MI → chaotic breather → single DKS soliton step) + sech² spectrum |
