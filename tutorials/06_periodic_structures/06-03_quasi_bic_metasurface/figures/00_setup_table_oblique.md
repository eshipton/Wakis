| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 680 nm × 21.2 nm × 1.22 µm (z ∈ [-608 nm, 608 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 32 × 1 × 64 (2048 cells) |
| Discretization | Δz & resolution | Δ_min = 19 nm, 23.8 cells/λ (in n_max = 3.48), 4.5 cells across feature (85 nm) |
| Boundary | -x face | Bloch (k_x = -6.93e+05 rad/m) |
| Boundary | +x face | Bloch (k_x = -6.93e+05 rad/m) |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (10 cells, 190 nm) |
| Boundary | +z face | CPML (10 cells, 190 nm) |
| Material | Background | Air / vacuum background (ε_bg = 1.00, n = 1.00) (isotropic) |
| Material | Region 1: Si bar 1 (w₁ = w₀ = 170.0 nm, h = 190 nm) | ε_r = 12.11, μ_r = 1.00, model = Isotropic dielectric, z ∈ [-95 nm, 95 nm] (d = 190 nm) |
| Material | Region 2: Si bar 2 (w₂ = w₀ = 170.0 nm, h = 190 nm) | ε_r = 12.11, μ_r = 1.00, model = Isotropic dielectric, z ∈ [-95 nm, 95 nm] (d = 190 nm) |
| Source | 1 · FloquetCellSource | pos: z = −320 nm (x ∈ [−340, +340] nm), pol: Ey (TE, θ_inc = 10°), pulse: f₀ = 190.5 THz, Δf = 15.24 THz |
| Source | 2 · CurrentSource | pos: (x, z) = (−127.5, −25) nm, pol: Jy, pulse: f₀ = 190.5 THz, Δf = 15.24 THz |
| Monitor | 3 · DiffractionMonitor | Reflected Floquet DiffractionMonitor at z = −220 nm, pos: z = −220 nm (x ∈ [−340, +340] nm), freqs: λ ∈ [1560, 1620] nm (AngleSweep resampling) |
| Monitor | 4 · DiffractionMonitor | Transmitted Floquet DiffractionMonitor at z = +220 nm, pos: z = +220 nm (x ∈ [−340, +340] nm), freqs: λ ∈ [1560, 1620] nm (AngleSweep resampling) |
| Monitor | 5 · ResonanceMonitor | Bloch ResonanceMonitor (harminv Ey) at (x₁, z) = (−127.5, +25) nm, pos: (x, z) = (−127.5, +25) nm, freqs: Q_rad(k_∥) ∝ sin⁻² θ_inc |
| Variation | Oblique angle & Bloch wavenumber sweep (Task 3) | θ_inc ∈ {1°, 2°, 3°, 5°, 7°, 10°} (harminv Q_rad(k_∥) ∝ k_∥⁻²) and AngleSweep θ_inc ∈ {0°, 6°, 9°, 12°} |
