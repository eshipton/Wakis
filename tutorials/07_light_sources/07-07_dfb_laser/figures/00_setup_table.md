| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 521 nm × 521 nm × 10.3 µm (z ∈ [-5.17 µm, 5.17 µm]) |
| Domain | Grid cells | Nx × Ny × Nz = 1 × 1 × 992 (992 cells) |
| Discretization | Δz & resolution | Δ_min = 10.4 nm, 48.0 cells/λ (in n_max = 3.00), 12.0 cells across feature (125 nm) |
| Boundary | -x face | Periodic |
| Boundary | +x face | Periodic |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | Mur ABC |
| Boundary | +z face | Mur ABC |
| Material | Background | Index-matched waveguide cladding (n₀ = 3.00, ε_r = 9.00) (isotropic) |
| Material | Region 1: Left DFB grating (z ∈ [-L/2, 0], φ = -π/2, SaturableGain) | ε_r = 9.00, μ_r = 1.00, model = Isotropic dielectric, z ∈ [-5 µm, 0 m] (d = 5 µm) |
| Material | Region 2: Right DFB grating (z ∈ [0, +L/2], φ = +π/2, λ/4 shift at z = 0) | ε_r = 9.00, μ_r = 1.00, model = Isotropic dielectric, z ∈ [0 m, 5 µm] (d = 5 µm) |
| Source | 1 · CurrentSource | pos: z = 0 m, pol: J_x (E_x), pulse: f₀ = 199.9 THz, Δf = 15.99 THz |
| Monitor | 2 · ResonanceMonitor | ResonanceMonitor (z = -1.67 µm), pos: z = -1.67 µm, freqs: time-domain |
| Monitor | 3 · Probe | Facet output probe z_out, pos: z = 5.08 µm, freqs: time-domain |
| Variation | Run variation | Case=λ/4-shifted DFB (Φ = π/2), Grating phase=φ(z) = sign(z) π/2, Gain model=Linear (I_sat = ∞) & Saturable (I_sat = 10 MW/m²), Mode behavior=Single Bragg mode at δL = 0 |
| Variation | Run variation | Case=Uniform DFB (Φ = 0), Grating phase=φ(z) = 0, Gain model=Linear (I_sat = ∞) & Saturable (I_sat = 10 MW/m²), Mode behavior=Degenerate stopband doublet at ±|δ₁L| |
