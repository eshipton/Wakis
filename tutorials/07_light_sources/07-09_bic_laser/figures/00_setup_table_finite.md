| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 6.76 µm × 21.2 nm × 2.30 µm (z ∈ [-1.15 µm, 1.15 µm]) |
| Domain | Grid cells | Nx × Ny × Nz = 318 × 1 × 68 (21624 cells) |
| Discretization | Δx & resolution | Δ_min = 15 nm, 30.2 cells/λ (in n_max = 3.48), 5.7 cells across feature (85 nm) |
| Boundary | -x face | CPML (10 cells, 212.5 nm) |
| Boundary | +x face | CPML (10 cells, 212.5 nm) |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (10 cells, 441 nm) |
| Boundary | +z face | CPML (10 cells, 441 nm) |
| Material | Background | Air cladding (ε_bg = 1.00, open CPML on ±x and ±z) (isotropic) |
| Material | Region 1: Active resonator array (N = 8) | ε_r = 12.11, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-2.59 µm, 2.59 µm] (d = 5.18 µm) |
| Source | 1 · CurrentSource | pos: z = 0 nm, x ∈ [−2.55, +2.55] µm (16 active bars), pol: ±Jy (intra-cell odd, envelope m = 1, 2, 3), pulse: f₀ = 190.5 THz, Δf = 15.24 THz |
| Monitor | 2 · Near2FarMonitor | Near2FarMonitor radiation aperture at z = +650 nm (projects doughnut beam I(θ)), pos: z = +650 nm, x ∈ [−3.10, +3.10] µm, freqs: λ₁(N=8) = 1582.5 nm (fundamental BIC lasing line) |
| Variation | Finite N-period strip size sweep (Task 1) | N ∈ {4, 6, 8, 10, 12} unit cells (Q₁(N) ∝ N^2.55; fundamental m = 1 has highest Q) |
| Variation | Envelope mode competition (Task 2) | Modes m = 1, 2, 3 at N = 8 (Q₁ = 528.3, Q₂ = 138.1, Q₃ = 65.3; single-mode lasing at m = 1) |
