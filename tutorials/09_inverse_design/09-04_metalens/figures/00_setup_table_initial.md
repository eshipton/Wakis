| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 16 µm × 6.40 µm × 20 nm (z ∈ [-10 nm, 10 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 800 × 320 × 1 (256000 cells) |
| Discretization | Δx & resolution | Δ_min = 20 nm, 11.5 cells/λ (in n_max = 2.40), 5.0 cells across feature (100 nm) |
| Boundary | -x face | Periodic |
| Boundary | +x face | Periodic |
| Boundary | -y face | CPML (20 cells, 400 nm) |
| Boundary | +y face | CPML (20 cells, 400 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Air superstrate (ε = 1.00, n = 1.00) (isotropic) |
| Material | Region 1: Fused-silica substrate (SiO₂, ε = 2.25, n = 1.50) | ε_r = 2.25, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-8 µm, 8 µm] (d = 16 µm) |
| Material | Region 2: TiO₂ metasurface design region Ω_d (initial ρ ≈ 0.48, 10.0 µm × 1.0 µm) | ε_r = 5.76, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-5 µm, 5 µm] (d = 10 µm) |
| Source | 1 · CurrentSource | pos: x ∈ [-7.50, +7.50] µm, y = -1.00 µm, pol: TM (E_z), pulse: f₀ = 557.5 THz, Δf = 153.7 THz |
| Monitor | 2 · DFTMonitor | Focal-point Objective DFTMonitor (E_z, NA = 0.9015), pos: (x_f, y_f) = (0.00, +3.40) µm, freqs: 3 RGB channels λ ∈ {450, 550, 650} nm (blue, green, red) |
| Monitor | 3 · DFTMonitor | Focal-plane Line DFTMonitor (E_z lateral profile), pos: x ∈ [-7.50, +7.50] µm, y = +3.40 µm, freqs: 3 RGB channels λ ∈ {450, 550, 650} nm (blue, green, red) |
| Variation | Initial design (_initial) | Stage-1 iter 0 initial greyscale TiO₂ density (mean ρ ≈ 0.48) in 500 × 50 active design box Ω_d (10.0 µm × 1.0 µm) |
| Variation | Final Round-2 design (_final) | Stage-2 STE binary design (ste_beta64:14, 5/5 px = 100 nm, official fmmax = 8.87×, Wakis = 8.43×) from round2_metalens/design.npy |
| Variation | Cross-solver check (fmmax) | Non-FIT Fourier modal method (FMM/RCWA) reference evaluator (Rasmus123nm parity: Wakis 14.27× vs fmmax 14.30×) |
