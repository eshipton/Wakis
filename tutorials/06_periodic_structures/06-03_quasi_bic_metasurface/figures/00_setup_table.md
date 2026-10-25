| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 680 nm × 21.2 nm × 2.60 µm (z ∈ [-1.30 µm, 1.30 µm]) |
| Domain | Grid cells | Nx × Ny × Nz = 32 × 1 × 74 (2368 cells) |
| Discretization | Δz & resolution | Δ_min = 15 nm, 30.2 cells/λ (in n_max = 3.48), 5.7 cells across feature (85 nm) |
| Boundary | -x face | Periodic |
| Boundary | +x face | Periodic |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (10 cells, 448 nm) |
| Boundary | +z face | CPML (10 cells, 448 nm) |
| Material | Background | Air / vacuum background (ε_bg = 1.00, n = 1.00) (isotropic) |
| Material | Region 1: Si bar 1 (w₁ = 181.9 nm, h = 190 nm) | ε_r = 12.11, μ_r = 1.00, model = Isotropic dielectric, z ∈ [-95 nm, 95 nm] (d = 190 nm) |
| Material | Region 2: Si bar 2 (w₂ = 158.1 nm, h = 190 nm) | ε_r = 12.11, μ_r = 1.00, model = Isotropic dielectric, z ∈ [-95 nm, 95 nm] (d = 190 nm) |
| Source | 1 · CurrentSource | pos: z = −580 nm (full x ∈ [−340, +340] nm), pol: Ey (TE, +z), pulse: f₀ = 190.5 THz, Δf = 15.24 THz |
| Source | 2 · CurrentSource | pos: (x, z) = (−127.5, −25) nm (inside Si bar 1), pol: Jy (couples to C₂-odd BIC & quasi-BIC), pulse: f₀ = 190.5 THz, Δf = 15.24 THz |
| Monitor | 3 · DiffractionMonitor | Reflected DiffractionMonitor (R₀₀) at z = −420 nm, pos: z = −420 nm (x ∈ [−340, +340] nm), freqs: λ ∈ [1520, 1578] nm (251 DFT frequencies) |
| Monitor | 4 · DiffractionMonitor | Transmitted DiffractionMonitor (T₀₀) at z = +580 nm, pos: z = +580 nm (x ∈ [−340, +340] nm), freqs: λ ∈ [1520, 1578] nm (251 DFT frequencies) |
| Monitor | 5 · ResonanceMonitor | ResonanceMonitor (harminv Ey) at (x₁, z) = (−127.5, +25) nm, pos: (x, z) = (−127.5, +25) nm, freqs: 0.90 f₀ – 1.10 f₀ (38 optical cycles ring-down) |
| Variation | Asymmetry sweep (Tasks 0–1) | α = Δw / w₀ ∈ {0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20} (subpixel dual-cell averaging; Q_rad = Q₀ α⁻²) |
| Variation | Absorption loss saturation (Task 2) | Conductivity σ_Si inside bars giving Q_abs ∈ {5×10³, 1.5×10⁴, 5×10⁴, 10⁵} (Q_tot⁻¹ = Q_rad⁻¹ + Q_abs⁻¹) |
| Variation | Oblique Floquet angle sweep (Task 3) | Symmetric dimer (α = 0) with ±x Bloch BCs (k_x = k₀ sin θ_inc, θ_inc ∈ [1°, 12°]) — see 00a/00b_*_oblique.png |
| Variation | Resolution & CPML proximity study (Task 4) | n_x ∈ {16, 24, 32, 48, 64}, subpixel vs staircase quantization, and d_air ∈ [150, 850] nm |
