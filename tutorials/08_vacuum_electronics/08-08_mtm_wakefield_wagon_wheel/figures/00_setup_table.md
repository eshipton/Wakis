| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 18 mm × 18 mm × 40 mm (z ∈ [0 m, 40 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 64 × 64 × 160 (655360 cells) |
| Discretization | Δz & resolution | Δ_min = 250 µm, 105.0 cells/λ (in n_max = 1.00), 4.0 cells across feature (1 mm) |
| Boundary | -x face | PEC |
| Boundary | +x face | PEC |
| Boundary | -y face | PEC |
| Boundary | +y face | PEC |
| Boundary | -z face | CPML (10 cells, 2.50 mm) |
| Boundary | +z face | CPML (10 cells, 2.50 mm) |
| Material | Background | Vacuum beam tunnel (r_b = 3 mm, PRL line 122) & inter-spoke/inter-plate vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Outer circular PEC waveguide wall (ID = 16 mm, PRL line 121) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [0 m, 40 mm] (d = 40 mm) |
| Material | Region 2: Outer circular PEC waveguide wall (ID = 16 mm, PRL line 121) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [0 m, 40 mm] (d = 40 mm) |
| Material | Region 3: Adiabatic coaxial TEM coupler shield (r ∈ [3.0, 3.3] mm, ASSUMED) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [0 m, 8 mm] (d = 8 mm) |
| Material | Region 4: Adiabatic coaxial TEM coupler shield (r ∈ [3.0, 3.3] mm, ASSUMED) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [0 m, 8 mm] (d = 8 mm) |
| Material | Region 5: Adiabatic coaxial TEM coupler shield (r ∈ [3.0, 3.3] mm, ASSUMED) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [32 mm, 40 mm] (d = 8 mm) |
| Material | Region 6: Adiabatic coaxial TEM coupler shield (r ∈ [3.0, 3.3] mm, ASSUMED) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [32 mm, 40 mm] (d = 8 mm) |
| Material | Region 7: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [8.50 mm, 9.50 mm] (d = 1 mm) |
| Material | Region 8: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [8.50 mm, 9.50 mm] (d = 1 mm) |
| Material | Region 9: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [10.5 mm, 11.5 mm] (d = 1 mm) |
| Material | Region 10: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [10.5 mm, 11.5 mm] (d = 1 mm) |
| Material | Region 11: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [12.5 mm, 13.5 mm] (d = 1 mm) |
| Material | Region 12: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [12.5 mm, 13.5 mm] (d = 1 mm) |
| Material | Region 13: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [14.5 mm, 15.5 mm] (d = 1 mm) |
| Material | Region 14: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [14.5 mm, 15.5 mm] (d = 1 mm) |
| Material | Region 15: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [16.5 mm, 17.5 mm] (d = 1 mm) |
| Material | Region 16: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [16.5 mm, 17.5 mm] (d = 1 mm) |
| Material | Region 17: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [18.5 mm, 19.5 mm] (d = 1 mm) |
| Material | Region 18: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [18.5 mm, 19.5 mm] (d = 1 mm) |
| Material | Region 19: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [20.5 mm, 21.5 mm] (d = 1 mm) |
| Material | Region 20: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [20.5 mm, 21.5 mm] (d = 1 mm) |
| Material | Region 21: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [22.5 mm, 23.5 mm] (d = 1 mm) |
| Material | Region 22: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [22.5 mm, 23.5 mm] (d = 1 mm) |
| Material | Region 23: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [24.5 mm, 25.5 mm] (d = 1 mm) |
| Material | Region 24: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [24.5 mm, 25.5 mm] (d = 1 mm) |
| Material | Region 25: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [26.5 mm, 27.5 mm] (d = 1 mm) |
| Material | Region 26: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [26.5 mm, 27.5 mm] (d = 1 mm) |
| Material | Region 27: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [28.5 mm, 29.5 mm] (d = 1 mm) |
| Material | Region 28: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [28.5 mm, 29.5 mm] (d = 1 mm) |
| Material | Region 29: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [30.5 mm, 31.5 mm] (d = 1 mm) |
| Material | Region 30: Wagon-wheel PEC plates (hub r ∈ [3, 4] mm + 4 radial spokes, t = 1 mm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [30.5 mm, 31.5 mm] (d = 1 mm) |
| Source | 1 · Beam | pos: (x, y) = (0, 0) inside r_b = 3.0 mm (PRL line 122) beam tunnel, z ∈ [0, 40] mm, pol: Jz (TM₀₁-like backward slow wave), pulse: rigid baseband Gaussian bunch λ_q(z), σ_z = 1.2 mm (2σ_z = 2.4 mm, PRL line 151), |F(f)|² = exp(-(2π f σ_z / u_0)²) |
| Monitor | 2 · Poynting | Upstream (−z) coaxial port Poynting plane z_up (backward Cherenkov output), pos: z_up = 6.0 mm (upstream TEM port before −z CPML), r ∈ [3.3, 8.0] mm, freqs: Passband-filtered Poynting flux P_up(t) [MW] |
| Monitor | 3 · WakeSolver | On-axis wake integration line (L = N p) → W_∥(s), Re Z_∥(f), pos: (x, y) = (0, 0), z ∈ [12.0, 28.0] mm, freqs: 0–25 GHz (reversed-Cherenkov peak at f_pk = 11.11 GHz) |
| Monitor | 4 · Poynting | Downstream (+z) coaxial port Poynting plane z_dn (forward leakage check), pos: z_dn = 34.0 mm (downstream TEM port before +z CPML), r ∈ [3.3, 8.0] mm, freqs: Passband-filtered Poynting flux P_dn(t) [MW] |
| Variation | 3D Bloch unit-cell cold test & mesh convergence (Part 2) | Single period p = 2.0 mm (PRL line 140) with Bloch BC along z (φ ∈ [12°, 72°]) and n_feat ∈ {4, 6, 8} → crossing 11.53 GHz at φ = 27.69° (11.58 GHz at φ_s = 27.43°, derived from ω_s p/u_0), v_E = -0.162c, r/Q = 22.0 kΩ/m |
| Variation | Finite structure length scaling & forward-wave control (Part 3) | N ∈ {6, 8, 12} periods (L = 12–24 mm, short-N pre-asymptotic regime vs experimental N = 40, L = 80 mm) and non-inductive disk-loaded forward-wave control (v_g > 0) |
| Variation | Two-bunch linear superposition & charge-scaled estimate (Part 4) | Exact 3D FIT linear superposition check and single-cycle (Δt = 1/f_s = 90.0 ps) charge-scaled single-bunch estimate for Q₁ + Q₂ = 85 nC vs experimental 1.3 GHz (PRL line 143) linac spacing (769 ps; 9th harmonic of 1.27 GHz bunch train with T_b = 0.79 ns, thesis lines 2340–2341, 2365) |
