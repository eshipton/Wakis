| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 380 nm × 10.7 nm × 1.40 µm (z ∈ [-619 nm, 779 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 35 × 1 × 131 (4585 cells) |
| Discretization | Δz & resolution | Δ_min = 10.7 nm, 24.8 cells/λ (in n_max = 2.00), 15.0 cells across feature (160 nm) |
| Boundary | -x face | Bloch (k_x = -2.06e+06 rad/m) |
| Boundary | +x face | Bloch (k_x = -2.06e+06 rad/m) |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (14 cells, 149.3 nm) |
| Boundary | +z face | CPML (14 cells, 149.3 nm) |
| Material | Background | Air cladding / groove (n = 1.00, ε_r = 1.00) (isotropic) |
| Material | Region 1: High-index glass waveguide (n_s = 1.90, ε_r = 3.61, θ_c = 31.8°) | ε_r = 3.61, μ_r = 1.00, model = Isotropic dielectric, z ∈ [160 nm, 779 nm] (d = 619 nm) |
| Material | Region 2: Si₃N₄ in-coupler ridge (w = 190 nm, h = 160 nm, n_SiN = 2.00, ε_r = 4.00) | ε_r = 4.00, μ_r = 1.00, model = Isotropic dielectric, z ∈ [0 m, 160 nm] (d = 160 nm) |
| Source | 1 · CurrentSource | pos: z = -378 nm (x ∈ [−190, +190] nm), pol: Ey (TE, θ_in = 10°), pulse: f₀ = 565.6 THz, Δf = 67.88 THz |
| Monitor | 2 · FloquetPort | Port 1 Reflected FloquetPort / DiffractionMonitor at z = -195 nm, pos: z = -195 nm (in air), freqs: λ_B = 442 nm, λ_G = 530 nm, λ_R = 663 nm |
| Monitor | 3 · FloquetPort | Port 2 Transmitted FloquetPort / DiffractionMonitor at z = +379 nm, pos: z = +379 nm (in glass waveguide), freqs: λ_B = 442 nm, λ_G = 530 nm, λ_R = 663 nm |
| Variation | RGB FOV AngleSweep (Task 0) | θ_in ∈ [−20°, +20°] across Blue (442 nm), Green (530 nm), Red (663 nm) vs 1D RCWA |
| Variation | Achromatic (4, 5, 6) supercell (Task 1) | Shared period Λ = 1900 nm with 6 Si₃N₄ ridges matching m_R λ_R ≈ m_G λ_G ≈ m_B λ_B — see *_supercell.png |
| Variation | Out-coupler TIR bounce sweep (Task 2) | Illumination from glass (n_s = 1.90) at θ_wg ∈ [35°, 55°] > θ_c = 31.8° — see *_outcoupler.png |
| Variation | 4-level FIT convergence study (Task 3) | cpw ∈ {16, 20, 26, 34} cells / λ_min,mat with 16×16 subpixel permittivity smoothing |
