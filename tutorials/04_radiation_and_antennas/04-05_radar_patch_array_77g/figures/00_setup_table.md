| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 29.6 mm × 6.76 mm × 3.86 mm (z ∈ [-1.81 mm, 2.06 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 241 × 60 × 30 (433800 cells) |
| Discretization | Δx & resolution | Δ_min = 47.1 µm, 55.7 cells/λ (in n_max = 1.48), 5.4 cells across feature (254 µm) |
| Boundary | -x face | CPML (6 cells, 860 µm) |
| Boundary | +x face | CPML (6 cells, 860 µm) |
| Boundary | -y face | CPML (6 cells, 860 µm) |
| Boundary | +y face | CPML (6 cells, 860 µm) |
| Boundary | -z face | CPML (6 cells, 955 µm) |
| Boundary | +z face | CPML (6 cells, 955 µm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: 254 µm Rogers 5880 substrate | ε_r = 2.20, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.47 mm, 24.7 mm] (d = 26.2 mm) |
| Material | Region 2: Copper ground plane | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-1.47 mm, 24.7 mm] (d = 26.2 mm) |
| Material | Region 3: 10-element series-fed patch column | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-718 µm, 23.9 mm] (d = 24.7 mm) |
| Source | 1 · LumpedPort | pos: x = -638 µm, pol: E_z, pulse: f₀ = 77.00 GHz, Δf = 6 GHz |
| Monitor | 2 · Near2FarMonitor | Near2FarMonitor, pos: x ∈ [-1.89 mm, 25.1 mm], freqs: 7 freqs in [75.0, 78.0] GHz |
| Variation | Array topologies | Uniform '1111111111' (10 × W_1 = 1540 µm), binary GA '1011111111' (element #2 omitted, 9 × W_1 = 1540 µm), and raised-cosine width-taper variation |
| Variation | 4-level mesh convergence & SLL gap study | coarse (157k cells), medium (282k), fine (434k, 20.8 cpw, 4 cells across h & W_2), and gpu_superfine_32um (1.39M, 40.4 cpw, 8 cells across h & W_2); lossy Rogers 5880 (σ_d = 0.00846 S/m) and feed/termination variants |
