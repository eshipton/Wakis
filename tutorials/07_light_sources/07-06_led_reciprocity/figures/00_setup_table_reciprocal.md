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
| Source | 1 · CurrentSource | pos: y = 378.0 nm, x ∈ [-1.08, +1.08] µm (θ ∈ [-70°, +70°]), pol: s (TM) + p (TE), pulse: f₀ = 666.2 THz, Δf = 133.2 THz |
| Monitor | 2 · DFTMonitor | Active-QW DFTMonitor (y_0 = 49.0 nm; integrates |E_rec(x, y_0; θ) / E_inc|² across 7 grating periods), pos: y_0 = 49.0 nm, x ∈ [-0.94, +0.94] µm (records E_x, E_y, E_z), freqs: f = 666.2 THz |
| Variation | Task 0 — 1D Bloch flat-slab reciprocity vs. TMM (`flat_slab_bloch_tmm`) | notes=Bloch unit cell on PEC & Drude Ag mirrors (cpw ∈ {10, 14, 20, 28, 40}, cpsd ∈ [1.6, 10.0]) |
| Variation | Tasks 2–8 — Kottke subpixel vs. snapped staircase convergence (`kottke_*`, `staircase_*`) | notes=7-period GaN grating reciprocal angular sweep cpw ∈ {14, 20, 28} vs. cpw = 40 reference |
| Variation | Section 5 — Janssen et al. (2010) unit-cell optimization & discrete adjoint | notes=2D Bloch FIT + `ResonanceMonitor.harminv` + `AdjointSimulation` shape gradients on Ag (`silver jc`, ~2× extraction) |
