| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 9.59 mm × 4 mm × 150 µm (z ∈ [-75 µm, 75 µm]) |
| Domain | Grid cells | Nx × Ny × Nz = 64 × 128 × 1 (8192 cells) |
| Discretization | Δx & resolution | Δ_min = 31.2 µm, 307.0 cells/λ (in n_max = 1.00), 16.0 cells across feature (500 µm) |
| Boundary | -x face | Bloch (k_x = -655 rad/m) |
| Boundary | +x face | Bloch (k_x = -655 rad/m) |
| Boundary | -y face | PEC |
| Boundary | +y face | PEC |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Drifting e⁻ sheet beam (b = 0.5 mm, u₀ = 0.16c, V₀ = 6.70 kV, f_p = 1.35 GHz) | ε_r = 1.00, μ_r = 1.00, model = Custom, x ∈ [0 m, 9.59 mm] (d = 9.59 mm) |
| Source | 1 · LinearizedBeam | pos: y ∈ [−0.50 mm, +0.50 mm], x ∈ [0 mm, 9.59 mm], pol: J_{1x}, E_x, pulse: f₀ = 5 GHz, Δf = 2.7 GHz |
| Monitor | 2 · EigenmodeSlice | TM₁₀ eigenmode slice (x = 2.40 mm), pos: x = 2.40 mm, y ∈ [−2.0 mm, +2.0 mm], freqs: 6 freqs in [1.5, 10.0] GHz |
| Monitor | 3 · Probe | Doppler doublet FFT probe E_x(x_p, 0, t) (x_p = 2.40 mm), pos: x = 2.40 mm, freqs: time-domain |
| Variation | Tasks 0–2: 2D Bloch dispersion & convergence (§5, §7–8) | 2D Bloch cell L_x = λ_e = 9.59 mm, 2a = 4.0 mm, PEC ±y walls, sheet e⁻ beam b = 0.5 mm (u₀ = 0.16c, V₀ = 6.70 kV, f_p = 1.35 GHz); grids 16×32, 24×48, 64×128 |
| Variation | Tasks 3–4: Frequency & filling-ratio sweeps (§7) | 2D Bloch cell (24×40); f₀ ∈ {1.5, 3.0, 5.0, 7.0, 8.5, 10.0} GHz and b/a ∈ {0.25, 0.375, 0.50, 0.75, 1.00} |
| Variation | Task 5: 3D square PEC drift tunnel (§7) | 3D domain 4.0 mm × 4.0 mm × 9.59 mm (40×40×24), PEC ±x, ±y walls, Bloch ±z, pencil beam b = 0.5 mm (f_p = 1.65 GHz) |
| Variation | Task 6: 2D klystron velocity-modulated drift tube (§7) | Open tube L_z = 115 mm × 4.0 mm (360×32), CPML ±z (N_pml = 6, 1.92 mm) + 10-cell ρ₀→0 taper, PEC ±y walls, RF gap V_m = 100 V at z_g = 3.51 mm, beam b = 0.8 mm |
| Variation | Task 7: Chu kinetic power & energy audit (§3, §7) | Periodic ±x, PEC ±y (48×48, 2400 steps); fast wave (P_k > 0) vs slow wave (P_k < 0) and coupled Chu–Poynting conservation |
