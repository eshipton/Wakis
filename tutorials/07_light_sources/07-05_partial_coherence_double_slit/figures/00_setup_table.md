| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 10.4 µm × 15.3 µm × 250 nm (z ∈ [-125 nm, 125 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 416 × 612 × 1 (254592 cells) |
| Discretization | Δx & resolution | Δ_min = 25 nm, 20.0 cells/λ (in n_max = 1.00), 8.0 cells across feature (200 nm) |
| Boundary | -x face | CPML (8 cells, 200 nm) |
| Boundary | +x face | CPML (8 cells, 200 nm) |
| Boundary | -y face | CPML (8 cells, 200 nm) |
| Boundary | +y face | CPML (8 cells, 200 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Vacuum / Air (n = 1.00), ε = 1.00 (isotropic) |
| Material | Region 1: PEC double-slit screen (t = 200 nm) | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-5.20 µm, -1.20 µm] (d = 4 µm) |
| Material | Region 2: PEC double-slit screen (t = 200 nm) | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-800 nm, 800 nm] (d = 1.60 µm) |
| Material | Region 3: PEC double-slit screen (t = 200 nm) | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [1.20 µm, 5.20 µm] (d = 4 µm) |
| Source | 1 · CurrentSource | pos: y = -8.00 µm, N_s = 37 dipoles in x ∈ [-1.80, +1.80] µm (5 shown), pol: J_z (TM), pulse: f₀ = 599.6 THz, Δf = 119.9 THz |
| Monitor | 2 · DFTMonitor | Slit-plane DFTMonitor (y = 0.00 µm; records E_z at slits x = ±1.00 µm for μ_12), pos: y = 0.00 µm, x ∈ [-4.60, +4.60] µm (slits at x = ±1.00 µm, a = 0.40 µm), freqs: f = 599.6 THz |
| Monitor | 3 · DFTMonitor | Downstream detector DFTMonitor (y = +6.00 µm; records interference profile I(x)), pos: y = +6.00 µm, x ∈ [-4.60, +4.60] µm (L_2 = 6.00 µm, Λ_f = 1.50 µm), freqs: f = 599.6 THz |
| Variation | Tasks 0–1 — Method A VCZ strip (`vcz_batch_a`, `vcz_batch_b`) | notes=N_s = 37 uncorrelated J_z line dipoles (19 symmetric runs at cpw = 20) spanning D ∈ [0, 3.60] µm (D_0 = 2.00 µm) |
| Variation | Tasks 6–7 — Spatial mesh convergence (`convergence_coarse`, `convergence_ref`) | notes=PEC double-slit mesh sweep cpw ∈ {10, 14, 20} vs. cpw = 40 reference |
