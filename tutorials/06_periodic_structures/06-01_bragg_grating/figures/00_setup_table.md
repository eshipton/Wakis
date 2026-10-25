| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 34.7 µm × 2.19 µm × 37.8 nm (z ∈ [91.1 nm, 129 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 1264 × 85 × 1 (107440 cells) |
| Discretization | Δx & resolution | Δ_min = 8.33 nm, 64.9 cells/λ (in n_max = 2.83), 6.0 cells across feature (50 nm) |
| Boundary | -x face | CPML (10 cells, 366 nm) |
| Boundary | +x face | CPML (10 cells, 366 nm) |
| Boundary | -y face | CPML (10 cells, 376 nm) |
| Boundary | +y face | CPML (10 cells, 376 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Air lateral/upper clad (n = 1.00, ε_r = 1.00; SiO₂ BOX n_sub = 1.44) (isotropic) |
| Material | Region 1: Si EIM core (n_eff,slab = 2.8335) | ε_r = 8.03, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-19.2 µm, 19.2 µm] (d = 38.4 µm) |
| Source | 1 · ModeSource | pos: x = -16.67 µm, y ∈ [-0.55, 0.55] µm, pol: TE₀ (in-plane E_y, H_z), pulse: f₀ = 192.5 THz, Δf = 7.4 THz |
| Monitor | 2 · ModeMonitor | Port 'in' ModeMonitor (S₁₁), pos: x = -16.60 µm, y ∈ [-0.55, 0.55] µm, freqs: 185.1–199.9 THz (λ ∈ [1500, 1620] nm, 81 pts) |
| Monitor | 3 · ModeMonitor | Port 'out' ModeMonitor (S₂₁), pos: x = 16.60 µm, y ∈ [-0.55, 0.55] µm, freqs: 185.1–199.9 THz (λ ∈ [1500, 1620] nm, 81 pts) |
| Monitor | 4 · FluxBox | 2D Poynting FluxBox, pos: x ∈ [-17.00, 17.00] µm, y ∈ [-0.72, 0.72] µm, freqs: 185.1–199.9 THz (81 pts) |
| Variation | Periodic unit cell & Bragg Bloch condition | Λ = 324 nm (D = 50%, l_tooth = 162 nm, w_core = 450 nm, Δw = 50 nm), Bloch phase Δφ = β_B Λ = π at λ_B; ±z faces: Periodic (2D EIM) |
| Variation | Sidewall phase offset (task_9) | Aligned (Δx_shift = 0 → κ = κ₀, R_max > 99.7%) vs half-period shifted (Δx_shift = Λ/2 = 162 nm → κ = 0, R_max = 0.0003) |
| Variation | Parametric sweeps (tasks 10–14) | Period count N ∈ {25, 50, 100}, tooth depth Δw ∈ {25, 50, 75} nm, Gaussian apodization (α = 4.0, −10.3 dB sidelobe reduction) |
| Variation | Resolution & averaging ladder (tasks 0–8) | Kottke vs staircase across L1 (10 cpw, 4/tooth), L2 (14 cpw, 6/tooth), L3 (20 cpw, 8/tooth), and reference R (40 cpw, 16/tooth) |
