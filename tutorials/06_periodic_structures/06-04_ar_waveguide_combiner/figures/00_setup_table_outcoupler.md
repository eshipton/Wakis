| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 380 nm × 11.7 nm × 1.38 µm (z ∈ [-618 nm, 758 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 32 × 1 × 118 (3776 cells) |
| Discretization | Δz & resolution | Δ_min = 11.7 nm, 22.7 cells/λ (in n_max = 2.00), 7.2 cells across feature (83.6 nm) |
| Boundary | -x face | Bloch (k_x = -1.59e+07 rad/m) |
| Boundary | +x face | Bloch (k_x = -1.59e+07 rad/m) |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (14 cells, 163.3 nm) |
| Boundary | +z face | CPML (14 cells, 163.3 nm) |
| Material | Background | Air cladding (eye-box extraction side, n = 1.00, ε_r = 1.00) (isotropic) |
| Material | Region 1: High-index glass waveguide (incident TIR side, n_s = 1.90, ε_r = 3.61) | ε_r = 3.61, μ_r = 1.00, model = Isotropic dielectric, z ∈ [-618 nm, 0 m] (d = 618 nm) |
| Material | Region 2: Si₃N₄ out-coupler tooth (w = 83.6 nm, h = 140 nm, f = 0.22, n_SiN = 2.00) | ε_r = 4.00, μ_r = 1.00, model = Isotropic dielectric, z ∈ [0 m, 140 nm] (d = 140 nm) |
| Source | 1 · CurrentSource | pos: z = -378 nm (in glass waveguide), pol: Ey (TE, θ_wg = 45°), pulse: f₀ = 565.6 THz, Δf = 67.88 THz |
| Monitor | 2 · FloquetPort | Port 1 Reflected FloquetPort in glass at z = -195 nm (R₀ ≈ 80% TIR + R₋₁), pos: z = -195 nm (in glass waveguide), freqs: f₀ = 565.6 THz (λ_G = 530 nm) |
| Monitor | 3 · FloquetPort | Port 2 Transmitted FloquetPort in air at z = +359 nm (T₀ evanescent, T₋₁ to eye), pos: z = +359 nm (in air toward eye-box), freqs: f₀ = 565.6 THz (λ_G = 530 nm) |
| Variation | TIR bounce angle sweep (Task 2) | θ_wg ∈ {35°, 40°, 45°, 50°, 55°} (> θ_c = 31.8°; T₀ ≡ NaN evanescent in air, T₋₁ extracted) |
| Variation | Lorentz reciprocity verification (Task 2) | FloquetPort S-matrix reciprocity between air→glass in-coupling and glass→air out-coupling at θ_air = 8° |
