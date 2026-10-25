| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 600 mm × 600 mm × 600 mm (z ∈ [-300 mm, 300 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 52 × 52 × 53 (143312 cells) |
| Discretization | Δx & resolution | Δ_min = 4.37 mm, 68.5 cells/λ (in n_max = 1.00), 1.1 cells across feature (5 mm) |
| Boundary | -x face | CPML (6 cells, 83.1 mm) |
| Boundary | +x face | CPML (6 cells, 83.1 mm) |
| Boundary | -y face | CPML (6 cells, 83.1 mm) |
| Boundary | +y face | CPML (6 cells, 83.1 mm) |
| Boundary | -z face | CPML (6 cells, 83.1 mm) |
| Boundary | +z face | CPML (6 cells, 83.1 mm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Source | 1 · CurrentSource | pos: x = 0 m, z = 0 m, pol: J_z (E_z), pulse: f₀ = 1 GHz, Δf = 600.0 MHz |
| Monitor | 2 · Near2FarMonitor | Near2FarMonitor, pos: x ∈ [-175.3 mm, 175.3 mm], freqs: f = 1 GHz |
| Variation | Dipole length sweep | $\ell / \lambda_0 \in [0.04, 0.50]$ ($N_{\mathrm{seg}} \in \{1, \dots, 25\}$) |
| Variation | CPML clearance study | $d_{\mathrm{PML}} / \lambda_0 \in [0.045, 0.420]$ ($kd_{\mathrm{PML}} \in [0.28, 2.64]$) |
| Variation | Uniform mesh convergence | $\lambda_0 / \Delta \in \{10, 14, 20, 26\}$ cells/$\lambda_0$ |
| Variation | Extended radiation movie | $6.63\,\lambda_0 \times 2.93\,\lambda_0 \times 6.63\,\lambda_0$ at $20.5$ cells/$\lambda_0$ ($r_{\max} = 3.32\,\lambda_0$) |
