| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 1 mm × 800 µm × 6.50 mm (z ∈ [0 m, 6.50 mm]) |
| Domain | Grid cells | Nx × Ny × Nz = 1 × 24 × 260 (6240 cells) |
| Discretization | Δz & resolution | Δ_min = 25 µm, 105.1 cells/λ (in n_max = 1.00), 4.0 cells across feature (100 µm) |
| Boundary | -x face | PMC |
| Boundary | +x face | PMC |
| Boundary | -y face | PEC |
| Boundary | +y face | PEC |
| Boundary | -z face | CPML (10 cells, 250 µm) |
| Boundary | +z face | CPML (10 cells, 250 µm) |
| Material | Background | Vacuum beam tunnel & grooves (ε_r = 1.00, H_b = 150 µm sheet beam, W_x = 1.0 mm) (isotropic) |
| Material | Region 1: Bottom PEC vanes (staggered grating + 2-stage taper) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [950 µm, 1.05 mm] (d = 100 µm) |
| Material | Region 2: Top PEC vanes (glide shift d/2 = 250 µm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [725 µm, 825 µm] (d = 100 µm) |
| Material | Region 3: Bottom PEC vanes (staggered grating + 2-stage taper) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [1.45 mm, 1.55 mm] (d = 100 µm) |
| Material | Region 4: Top PEC vanes (glide shift d/2 = 250 µm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [1.23 mm, 1.32 mm] (d = 100 µm) |
| Material | Region 5: Bottom PEC vanes (staggered grating + 2-stage taper) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [1.95 mm, 2.05 mm] (d = 100 µm) |
| Material | Region 6: Top PEC vanes (glide shift d/2 = 250 µm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [1.73 mm, 1.82 mm] (d = 100 µm) |
| Material | Region 7: Bottom PEC vanes (staggered grating + 2-stage taper) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [2.45 mm, 2.55 mm] (d = 100 µm) |
| Material | Region 8: Top PEC vanes (glide shift d/2 = 250 µm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [2.23 mm, 2.33 mm] (d = 100 µm) |
| Material | Region 9: Bottom PEC vanes (staggered grating + 2-stage taper) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [2.95 mm, 3.05 mm] (d = 100 µm) |
| Material | Region 10: Top PEC vanes (glide shift d/2 = 250 µm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [2.72 mm, 2.82 mm] (d = 100 µm) |
| Material | Region 11: Bottom PEC vanes (staggered grating + 2-stage taper) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [3.45 mm, 3.55 mm] (d = 100 µm) |
| Material | Region 12: Top PEC vanes (glide shift d/2 = 250 µm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [3.23 mm, 3.33 mm] (d = 100 µm) |
| Material | Region 13: Bottom PEC vanes (staggered grating + 2-stage taper) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [3.95 mm, 4.05 mm] (d = 100 µm) |
| Material | Region 14: Top PEC vanes (glide shift d/2 = 250 µm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [3.72 mm, 3.82 mm] (d = 100 µm) |
| Material | Region 15: Bottom PEC vanes (staggered grating + 2-stage taper) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [4.45 mm, 4.55 mm] (d = 100 µm) |
| Material | Region 16: Top PEC vanes (glide shift d/2 = 250 µm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [4.23 mm, 4.33 mm] (d = 100 µm) |
| Material | Region 17: Bottom PEC vanes (staggered grating + 2-stage taper) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [4.95 mm, 5.05 mm] (d = 100 µm) |
| Material | Region 18: Top PEC vanes (glide shift d/2 = 250 µm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [4.72 mm, 4.82 mm] (d = 100 µm) |
| Material | Region 19: Bottom PEC vanes (staggered grating + 2-stage taper) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [5.45 mm, 5.55 mm] (d = 100 µm) |
| Material | Region 20: Top PEC vanes (glide shift d/2 = 250 µm) | ε_r = inf, μ_r = 1.00, model = PEC, z ∈ [5.23 mm, 5.33 mm] (d = 100 µm) |
| Source | 1 · Beam | pos: y = 0 (sheet-beam channel |y| ≤ 75 µm), z ∈ [0, L_tot], pol: Jz (TM slow space harmonic), pulse: f₀ = 114.1 GHz, Δf = 45.00 GHz |
| Monitor | 2 · WakeSolver | Longitudinal wake integration line (L = N d) → W_∥(s), Re Z_∥(f), pos: y = 0, z ∈ [1.75, 4.75] mm, freqs: 0–200 GHz (synchronous crossing f_s = 114.09 GHz) |
| Monitor | 3 · Poynting | Upstream smooth-guide Poynting plane z_in (full height |y| ≤ 400 µm), pos: z_in = 0.50 mm (smooth guide before −z CPML), y ∈ [−400, +400] µm, freqs: Passband-filtered Poynting flux P_z(t) |
| Monitor | 4 · Poynting | Downstream smooth-guide Poynting plane z_out (full height |y| ≤ 400 µm), pos: z_out = 6.00 mm (smooth guide before +z CPML), y ∈ [−400, +400] µm, freqs: Passband-filtered Poynting flux P_z(t) |
| Variation | Interaction length & off-axis sheet-beam sweep (Task 2) | N ∈ {5, 10, 20, 40} periods (L = 2.5–20.0 mm) + 2-stage vane tapers [0.25h, 0.55h] into CPML (|Γ(f_s)| = 18.0%), plus y₀ ∈ {0, 25, 50, 75} µm sheet-beam profile |
| Variation | Backward-wave vs forward-wave synchronism & energy balance (Task 3) | Glide n = −1 backward wave (20 kV, u₀ = 0.2719c, v_g = −0.544c < 0, f_s = 114.29 GHz) vs mirror n = 0 forward wave (109.9 kV, u₀ = 0.5689c, v_g = +0.325c > 0, f_s = 127.92 GHz, y₀ = 50 µm) |
| Variation | Detuned beam & shorted PEC cavity mutations (Task 4) | Velocity detuning u₀ ± 5% shifts f_s by Δf_s; replacing CPML + tapers with PEC end walls produces standing-wave cavity modes at β₀ d = mπ/N |
