| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly = 280 nm × 280 nm (x ∈ [-140 nm, 140 nm], y ∈ [-140 nm, 140 nm]) |
| Domain | Grid cells | Nx × Ny = 4448 × 4448 (19784704 cells) |
| Discretization | Δx & resolution | Δ_min = 1.61 nm, 88.7 cells/λ (in n_max = 2.11), 14.0 cells across feature (22.5 nm) |
| Boundary | -x face | PEC |
| Boundary | +x face | PEC |
| Boundary | -y face | PEC |
| Boundary | +y face | PEC |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Ag cylinder (N = 2 Drude–Lorentz) | ε_r = 4.44, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-20 nm, 20 nm] (d = 40 nm) |
| Source | 1 · PlaneWave | pos: r <= 20 nm (Ag cylinder interior), pol: TE (E_x, E_y in-plane, H_z axial), pulse: f₀ = 666.2 THz, Δf = 166.6 THz |
| Monitor | 2 · HarmonicRings | Cylindrical harmonic rings (r = 45, 55, 65 nm), pos: r = 45, 55, 65 nm (inside radial CFS-PML r ∈ [80, 140] nm), freqs: 241 freqs in [333.1, 999.3] THz (300–900 nm) |
| Variation | Radial CFS-PML | 10 annular layers in r ∈ [80, 140] nm with PEC outer termination |
| Variation | Local mesh refinement | h_s = 2.0 nm (4448 tris) and h_s = 1.25 nm (7097 tris) on r = 20 nm boundary |
