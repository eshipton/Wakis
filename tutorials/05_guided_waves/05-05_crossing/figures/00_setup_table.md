| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 14.1 µm × 14.1 µm × 21.5 nm (z ∈ [-10.8 nm, 10.8 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 657 × 657 × 1 (431649 cells) |
| Discretization | Δx & resolution | Δ_min = 21.5 nm, 20.8 cells/λ (in n_max = 2.93), 16.3 cells across feature (350 nm) |
| Boundary | -x face | CPML (10 cells, 215 nm) |
| Boundary | +x face | CPML (10 cells, 215 nm) |
| Boundary | -y face | CPML (10 cells, 215 nm) |
| Boundary | +y face | CPML (10 cells, 215 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | SiO₂ cladding (ε_r = 2.10) (isotropic) |
| Material | Region 1: Si crossing core (ε_r = 8.57) | ε_r = 8.57, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-7.07 µm, 7.07 µm] (d = 14.1 µm) |
| Source | 1 · ModeSource | pos: x = -6.43 µm, y ∈ [-0.80, +0.80] µm, pol: TE₀ (Ey, Hz, +x), pulse: f₀ = 229.2 THz, Δf = 8.7 THz |
| Monitor | 2 · ModeMonitor | Port 'west' ModeMonitor (reflection S₁₁), pos: x = -6.35 µm, y ∈ [-0.80, +0.80] µm, freqs: 220.4–237.9 THz (51 DFT pts) |
| Monitor | 3 · ModeMonitor | Port 'east' ModeMonitor (thru S₂₁), pos: x = 6.35 µm, y ∈ [-0.80, +0.80] µm, freqs: 220.4–237.9 THz (51 DFT pts) |
| Monitor | 4 · ModeMonitor | Port 'north' ModeMonitor (+y cross S₃₁), pos: x ∈ [-0.80, +0.80] µm, y = 6.35 µm, freqs: 220.4–237.9 THz (51 DFT pts) |
| Monitor | 5 · ModeMonitor | Port 'south' ModeMonitor (-y cross S₄₁), pos: x ∈ [-0.80, +0.80] µm, y = -6.35 µm, freqs: 220.4–237.9 THz (51 DFT pts) |
| Monitor | 6 · FluxMonitor | 4-wall Poynting FluxMonitor box, pos: x ∈ [-7.02, 7.02] µm, y ∈ [-7.02, 7.02] µm, freqs: 220.4–237.9 THz (51 DFT pts) |
| Variation | Linear-tapered crossing (linear_taper_crossing) | param=w_in = 0.35 µm → w_out = 1.10 µm, l_t = 5.30 µm, note=Monotonic linear taper without TE₀+TE₂ self-imaging bulge; IL(1310 nm) = 0.65 dB, XT = -37.4 dB |
| Variation | Plain untapered crossing (plain_crossing) | param=w = 0.35 µm straight orthogonal intersection, note=Unfocused diffraction into cross arms; IL(1310 nm) = 0.95 dB, XT = -14.9 dB |
| Variation | 4-port D₄ symmetry & reciprocity audit (symmetry_4port) | param=Drives all 4 ports (west, east, north, south) at L3 = (20, 8), note=Verifies C₄ rotational invariance (5.6e-7), reciprocity (3.0e-7), and north/south mirror balance (1.1e-9) |
| Variation | Kottke vs staircase convergence ladder | param=L1=(10, 4), L2=(14, 6), L3=(20, 8), R=(40, 16), note=8.15 to 32.55 cells across w_in = 350 nm; Kottke converges at O(Δx^1.78) vs staircase O(Δx^0.82) |
| Variation | D₄-graded non-uniform mesh upgrade (kottke_nonuniform) | param=438 × 438 × 1 = 191,844 cells (-55.5% vs uniform L3), note=Δu = 21.0 nm in core/bulge graded to 42 nm in outer cladding (-65.9% runtime, IL = 0.087 dB) |
