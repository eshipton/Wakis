| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 327.5 mm × 327.5 mm × 383.4 mm (z ∈ [-191.7 mm, 191.7 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 28 × 28 × 39 (30576 cells) |
| Discretization | Δx & resolution | Δ_min = 7.89 mm, 38.0 cells/λ (in n_max = 1.00), 1.0 cells across feature (7.89 mm) |
| Boundary | -x face | CPML (6 cells, 86.5 mm) |
| Boundary | +x face | CPML (6 cells, 86.5 mm) |
| Boundary | -y face | CPML (6 cells, 86.5 mm) |
| Boundary | +y face | CPML (6 cells, 86.5 mm) |
| Boundary | -z face | CPML (6 cells, 79.7 mm) |
| Boundary | +z face | CPML (6 cells, 79.7 mm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: PEC dipole arm | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-2.50 mm, 2.50 mm] (d = 5 mm) |
| Material | Region 2: PEC dipole arm | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-2.50 mm, 2.50 mm] (d = 5 mm) |
| Source | 1 · LumpedPort | pos: x = 0 m, pol: E_z, pulse: f₀ = 1 GHz, Δf = 500.0 MHz |
| Monitor | 2 · Near2FarMonitor | Near2FarMonitor, pos: x ∈ [-54.5 mm, 54.5 mm], freqs: f = 1 GHz |
| Variation | Wire radius & shortening sweep | $a \in \{1.50, 0.50, 0.15\}\;\mathrm{mm}$ ($\Omega \in \{9.2, 11.4, 13.8\}$) and trimmed $L = 0.470\,\lambda_0$ |
| Variation | Resolved 4×4 PEC vs. subcell wire | $\Delta_{\perp,\mathrm{core}} = 0.76\;\mathrm{mm}$ ($4\times 4$ cells across $2w = 3.05\;\mathrm{mm}$) vs. subcell $m = \frac{1}{2}\ln(\Delta_\perp / a)$ |
| Variation | Quarter-wave monopole (infinite PEC) | $h = \lambda_0 / 4 = 75.0\;\mathrm{mm}$ on $z = 0$ PEC boundary (`ground_plane='-z'` in `Near2FarMonitor`) |
| Variation | Finite PEC ground plane sweep | Square PEC sheet $W_{\mathrm{gnd}} / \lambda_0 \in \{0.40, 0.80, 1.20\}$ in 6-face CPML |
| Variation | Extended radiation movie | $6.68\,\lambda_0 \times 2.55\,\lambda_0 \times 6.68\,\lambda_0$ at $22.0$ cells/$\lambda_0$ ($r_{\max} = 3.34\,\lambda_0$) |
