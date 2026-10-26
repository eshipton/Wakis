| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 700 µm × 860 µm × 13.0 mm (z ∈ [-1 mm, 12.0 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 2 × 25 × 677 (33850 cells) |
| Discretization | Δz, Δy & resolution | Δz_max = 20.0 µm (21.8–39.2 cells/λ_g at λ_g = 436.1 µm, 5–9 cells across 100 µm vane); Δy_min = 6.25–3.57 µm (4–7 cells across 25 µm beam–wall gap) |
| Boundary | -x face | Bloch (k_x = 4.02e+03 rad/m) |
| Boundary | +x face | Bloch (k_x = 4.02e+03 rad/m) |
| Boundary | -y face | PEC |
| Boundary | +y face | PEC |
| Boundary | -z face | CPML (8 cells, 155 µm) |
| Boundary | +z face | CPML (8 cells, 153.7 µm) |
| Material | Background | Vacuum tunnel & slots (eps_r = 1.00), ε = 1.00 (isotropic) |
| Material | Region 1: Lower PEC grating (teeth at z_m + d/4) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [-1 mm, 12 mm] (L_z = 13.0 mm; grating z ∈ [0, 11.0] mm, period d = 0.50 mm) |
| Material | Region 2: Upper PEC grating (glide +d/2, z_m + 3d/4) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [-1 mm, 12 mm] (L_z = 13.0 mm; grating z ∈ [0, 11.0] mm, period d = 0.50 mm) |
| Source | 1 · CurrentSource | pos: z = -0.60 mm, y in [-0.10, +0.10] mm, pol: Jy (excites LSE_10 Ez, Ey, Hx), pulse: f₀ = 186.0 GHz, Δf = 18.00 GHz |
| Source | 2 · LinearizedBeam | pos: y in [-75, +75] um, z in [0.5, 10.5] mm (active in 22-period grating), pol: J1z = rho0 v1z + u0 rho1, pulse: continuous / custom |
| Monitor | 3 · DFTMonitor | On-axis phasor monitor Ez(z, f0), I1(z, f0) along y = 0, pos: y = 0 mm, z in [0.5, 10.5] mm (periods m = 1..20), freqs: 180-210 GHz |
| Variation | 2D Karetnikova 0.2 THz staggered grating (Parts 2-3, N = 22 periods) | Domain z in [-1.0, 12.0] mm (13.0 mm total), grating z in [0, 11.0] mm, d = 0.50 mm, s = 0.10 mm, h = 0.30 mm, 2a = 0.20 mm, Hb = 0.15 mm (25 um beam-wall gap resolved with 4-7 cells), Nz/d in {26, 32, 40, 46} |
| Variation | 3D Marosi V-band glide-symmetric pillar SWS (Part 4) | a = 1.40 mm, b = 0.80 mm, d = 1.05 mm, pillars 1.08 x 0.27 x 0.10 mm^3, sheet beam 0.55 x 0.10 mm^2, V0 = 5.2 kV, I0 = 11 mA |
