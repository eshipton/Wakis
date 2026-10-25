| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 400 nm × 9.14 nm × 1.55 µm (z ∈ [-613 nm, 933 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 43 × 1 × 169 (7267 cells) |
| Discretization | Δz & resolution | Δ_min = 9.14 nm, 24.4 cells/λ (in n_max = 2.35), 6.6 cells across feature (60 nm) |
| Boundary | -x face | Bloch (k_x = -1.67e+06 rad/m) |
| Boundary | +x face | Bloch (k_x = -1.67e+06 rad/m) |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (14 cells, 128 nm) |
| Boundary | +z face | CPML (14 cells, 128 nm) |
| Material | Background | Air cladding / groove (n = 1.00, ε_r = 1.00) (isotropic) |
| Material | Region 1: N-LAF2 glass waveguide substrate (n_s = 1.74, ε_r = 3.03) | ε_r = 3.03, μ_r = 1.00, model = Isotropic dielectric, z ∈ [320 nm, 933 nm] (d = 613 nm) |
| Material | Region 2: Sawtooth blazed TiO₂ tooth (f = 0.85, d = 320 nm, n = 2.35, ε_r = 5.52, Kottke subpixel) | ε_r = 5.52, μ_r = 1.00, model = Isotropic dielectric, z ∈ [0 m, 320 nm] (d = 320 nm) |
| Source | 1 · CurrentSource | pos: z = -374 nm (x ∈ [−200, +200] nm), pol: Ey (TE, θ_in = 8°), pulse: f₀ = 571.0 THz, Δf = 68.52 THz |
| Monitor | 2 · FloquetPort | Port 1 Reflected FloquetPort at z = -193 nm (R_m in air), pos: z = -193 nm (in air), freqs: f₀ = 571.0 THz (λ₀ = 525 nm) |
| Monitor | 3 · FloquetPort | Port 2 Transmitted FloquetPort at z = +537 nm (T₋₁ > T₊₁ in glass), pos: z = +537 nm (in N-LAF2 glass), freqs: f₀ = 571.0 THz (λ₀ = 525 nm) |
| Variation | Sawtooth blazed vs binary & slanted (Tasks 0 & 2) | Right-angled triangular tooth (x_fill(z) = f·(1 − z/d), f = 0.85) across θ_in ∈ [−20°, +20°] |
