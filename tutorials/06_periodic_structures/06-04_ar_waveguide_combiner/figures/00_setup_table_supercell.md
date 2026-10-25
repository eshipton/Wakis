| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 1.90 µm × 12.3 nm × 1.39 µm (z ∈ [-615 nm, 775 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 144 × 1 × 113 (16272 cells) |
| Discretization | Δx & resolution | Δ_min = 12.3 nm, 21.5 cells/λ (in n_max = 2.00), 10.7 cells across feature (131.7 nm) |
| Boundary | -x face | Bloch (k_x = -1.24e+06 rad/m) |
| Boundary | +x face | Bloch (k_x = -1.24e+06 rad/m) |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (14 cells, 172.3 nm) |
| Boundary | +z face | CPML (14 cells, 172.3 nm) |
| Material | Background | Air cladding / grooves (n = 1.00, ε_r = 1.00) (isotropic) |
| Material | Region 1: High-index glass waveguide (n_s = 1.90, ε_r = 3.61) | ε_r = 3.61, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-950 nm, 950 nm] (d = 1.90 µm) |
| Material | Region 2: Si₃N₄ 6-ridge achromatic supercell (Λ = 1900 nm, h = 160 nm, n_SiN = 2.00) | ε_r = 4.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-950 nm, 950 nm] (d = 1.90 µm) |
| Source | 1 · CurrentSource | pos: z = -378 nm (x ∈ [−950, +950] nm), pol: Ey (TE, θ_in = 6°), pulse: f₀ = 565.6 THz, Δf = 67.88 THz |
| Monitor | 2 · FloquetPort | Port 1 Reflected FloquetPort at z = -195 nm, pos: z = -195 nm (in air), freqs: RGB orders (±4, ±5, ±6) |
| Monitor | 3 · FloquetPort | Port 2 Transmitted FloquetPort at z = +379 nm, pos: z = +379 nm (in glass waveguide), freqs: RGB orders (±4, ±5, ±6) |
| Variation | Achromatic RGB (4, 5, 6) AngleSweep (Task 1) | Λ = 1900 nm with m_R λ_R = 2652 nm, m_G λ_G = 2650 nm, m_B λ_B = 2652 nm (Δθ_wg < 0.08°) |
