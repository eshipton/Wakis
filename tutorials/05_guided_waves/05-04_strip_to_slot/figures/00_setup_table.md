| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 11.9 µm × 4.23 µm × 25.6 nm (z ∈ [-12.8 nm, 12.8 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 366 × 110 × 1 (40260 cells) |
| Discretization | Δx & resolution | Δ_min = 12.5 nm, 42.3 cells/λ (in n_max = 2.93), 8.0 cells across feature (100 nm) |
| Boundary | -x face | CPML (10 cells, 337.4 nm) |
| Boundary | +x face | CPML (10 cells, 338 nm) |
| Boundary | -y face | CPML (10 cells, 636 nm) |
| Boundary | +y face | CPML (10 cells, 636 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | SiO₂ cladding & 100 nm slot (ε_r = 2.07) (isotropic) |
| Material | Region 1: Si MMI core (ε_r = 8.59) | ε_r = 8.59, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-3 µm, 9 µm] (d = 12 µm) |
| Source | 1 · ModeSource | pos: x = -1.60 µm, y ∈ [-0.90, +0.90] µm, pol: TE₀ (Ey, Hz, +x), pulse: f₀ = 193.6 THz, Δf = 6.2 THz |
| Monitor | 2 · ModeMonitor | Port 'in' ModeMonitor (S₁₁), pos: x = -1.55 µm, y ∈ [-0.90, +0.90] µm, freqs: 187.4–199.9 THz (51 DFT pts) |
| Monitor | 3 · ModeMonitor | Port 'out' ModeMonitor (slot TE₀, S₂₁), pos: x = 7.55 µm, y ∈ [-0.90, +0.90] µm, freqs: 187.4–199.9 THz (51 DFT pts) |
| Monitor | 4 · FluxMonitor | 2D Poynting FluxMonitor box, pos: x ∈ [-2.87, 8.87] µm, y ∈ [-2.07, 2.07] µm, freqs: 187.4–199.9 THz (51 DFT pts) |
| Variation | Design 2: Symmetric Tapered-Finger (Wang et al. 2009) | param=L = 8.0 µm, d_finger = 60 nm, g_slot = 100 nm, note=Three-PolySlab double-tip finger coupler; T(1550 nm) = 99.80%, N = 8 cells across 60 nm finger gap (Δy_min = 7.14 nm) |
| Variation | Design 3: Asymmetric Lateral Taper (Feng et al. 2007) | param=L = 20.0 µm, g_slot = 100 nm, note=Two-PolySlab continuous upper rail + tapered lower rail; T(1550 nm) = 99.66%, N = 8 cells across 100 nm slot |
| Variation | Design 1 MMI length sweep | param=L ∈ {1.6, 2.2, 3.0, 4.5, 6.0} µm (L_mmi = 1.38 µm), note=Coupled-mode adiabatic saturation of slot-rail taper section L_taper = L - L_mmi |
| Variation | Kottke vs staircase convergence ladder | param=Under-resolved (10, 0), L1=(10, 4), L2=(14, 6), L3=(20, 8), R=(40, 16), note=Compares Kottke subpixel smoothing against staircase snapping across 100 nm oxide slot |
| Variation | Graded non-uniform mesh upgrade (kottke_nonuniform) | param=367 × 111 × 1 = 40,737 cells (-38.1% vs uniform L3), note=Δy = 12.5 nm (8 cells across 100 nm slot) with non-square Δx ≤ 34 nm along taper |
