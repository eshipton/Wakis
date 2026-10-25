| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 2.70 µm × 742 nm × 9.18 nm (z ∈ [0 m, 9.18 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 294 × 111 × 1 (32634 cells) |
| Discretization | Δx & resolution | Δ_min = 2.58 nm, 71.3 cells/λ (in n_max = 2.45), 10.4 cells across feature (26.9 nm) |
| Boundary | -x face | CPML (10 cells, 91.8 nm) |
| Boundary | +x face | CPML (10 cells, 91.8 nm) |
| Boundary | -y face | PEC |
| Boundary | +y face | CPML (10 cells, 90 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Air cladding (n = 1.00), ε = 1.00 (isotropic) |
| Material | Region 1: Drude Ag back-reflector (δ_s = 26.9 nm) | ε_r = 3.70, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-945 nm, 945 nm] (d = 1.89 µm) |
| Material | Region 2: GaN unetched base slab (n = 2.45, y_base = 153 nm) | ε_r = 6.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-945 nm, 945 nm] (d = 1.89 µm) |
| Material | Region 3: GaN 7-period trapezoidal grating (n = 2.45, Λ = 270 nm, h_g = 72 nm) | ε_r = 6.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-945 nm, 945 nm] (d = 1.89 µm) |
| Source | 1 · CurrentSource | pos: y_0 = 49.0 nm, N_sites = 25 × 3 orientations = 75 dipoles in x ∈ [-0.81, +0.81] µm (5 shown), pol: J_x, J_y, J_z, pulse: f₀ = 666.2 THz, Δf = 133.2 THz |
| Monitor | 2 · Near2FarMonitor2D | Cylindrical far-field line Near2FarMonitor2D (y_mon = 378.0 nm; projects U_2D(θ) into air), pos: y_mon = 378.0 nm, x ∈ [-1.08, +1.08] µm, freqs: f = 666.2 THz |
| Variation | Task 1 — Direct vs. reciprocal ensemble (`direct_vs_reciprocity_ensemble`) | notes=N_pos = 25 dipole sites × 3 orientations (J_x, J_y, J_z) = 75 forward solves vs. 29 reciprocal angles (cpw = 20, L2 diff = 0.066%) |
| Variation | Tasks 9–10 — Drude Ag skin-depth resolution ladder (`ag_grating_resolution_study`) | notes=Graded Ag collar 4, 6, 8, 10, 12 cells/δ_s (accepted Δ_min = 2.69 nm, 10.0 cells/δ_s) vs. under-resolved demo (1.6, 2.6 cells/δ_s) |
