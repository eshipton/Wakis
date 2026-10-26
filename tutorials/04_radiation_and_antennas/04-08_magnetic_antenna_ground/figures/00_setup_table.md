| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 467 mm × 419 mm × 235.4 mm (z ∈ [0 m, 235.4 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 43 × 31 × 19 (25327 cells) |
| Discretization | Δx & resolution | Δ_min = 4 mm, 42.3 cells/λ (in n_max = 6.32), 6.0 cells across feature (24 mm) |
| Boundary | -x face | CPML (4 cells, 109 mm) |
| Boundary | +x face | CPML (4 cells, 112 mm) |
| Boundary | -y face | CPML (4 cells, 109 mm) |
| Boundary | +y face | CPML (4 cells, 112 mm) |
| Boundary | -z face | PEC |
| Boundary | +z face | CPML (4 cells, 112 mm) |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Permeable rod (CZN Lorentz mu) | ε_r = 1.00, μ_r = 40.00, model = DispersiveMaterial, x ∈ [-36 mm, 36 mm] (d = 72 mm) |
| Source | 1 · CurrentSource | pos: z = 24 mm, pol: J_y (E_y), pulse: f₀ = 280.0 MHz, Δf = 126.0 MHz |
| Monitor | 2 · Near2FarMonitor | Near2FarMonitor, pos: x ∈ [-91 mm, 91 mm], freqs: f = 280.0 MHz |
| Variation | 1D Magnetic ADE slab | TEM waveguide ($2\times 2\times N_z$), Lorentz & Debye $\mu(\omega)$ slabs |
| Variation | 4-orientation PEC image sweep | HMD, VMD, VED, HED over $h/\lambda_0 \in [0.04, 0.50]$ |
| Variation | Volume convention & lift | Half-rod on PEC vs full free-space rod ($2 V_{\mathrm{phys}}$) vs lifted rod |
| Variation | Equal-$h_m$ Lorentz & Debye families | 4 Lorentz poles ($h_m=3.46\times 10^5\,\Omega$/m) & 3 Debye poles ($h_m=2.36\times 10^5\,\Omega$/m) |
| Variation | PEC/PMC duality & finite ground | Horiz permeable rod vs vert dielectric monopole; finite PEC $L_{\mathrm{gnd}}/\lambda_0$ |
