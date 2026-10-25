| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 26.7 mm × 26.7 mm × 3.20 m (z ∈ [0 m, 3.20 m]) |
| Domain | Grid cells | Nx × Ny × Nz = 1 × 1 × 480 (480 cells) |
| Discretization | Δz & resolution | Δ_min = 6.67 mm, 30.0 cells/λ (in n_max = 2.50), 21.0 cells across feature (140 mm) |
| Boundary | -x face | Periodic |
| Boundary | +x face | Periodic |
| Boundary | -y face | Periodic |
| Boundary | +y face | Periodic |
| Boundary | -z face | CPML (16 cells, 106.7 mm) |
| Boundary | +z face | CPML (16 cells, 106.7 mm) |
| Material | Background | Pre-switch medium (ε₁ = 2.25) (isotropic) |
| Material | Region 1: Time-varying medium (ε₁=2.25 → ε₂=6.25) | ε_r = 6.25, μ_r = 1.00, model = Isotropic dielectric, z ∈ [106.7 mm, 3.09 m] (d = 2.99 m) |
| Source | 1 · CurrentSource | pos: z = 1.10 m, pol: J_x (E_x), pulse: f₀ = 600.0 MHz, Δf = 227.2 MHz |
| Monitor | 2 · Probe | Time-reflected probe M_R (z = 0.85 m), pos: z = 850 mm, freqs: time-domain |
| Monitor | 3 · Probe | Time-refracted probe M_T (z = 2.35 m), pos: z = 2.35 m, freqs: time-domain |
| Variation | Task 0b · Periodic single-mode Fresnel & D/B continuity sweep | 1D periodic SolverFIT3D (Lz = 4.0 m, Nz = 160, m = 4) sweeping ε₂/ε₁ ∈ [0.44, 3.60] vs Morgenthaler (1958) & Galiffi (2022) |
| Variation | Task 1 · Frequency conversion & flash-ionized TimeVaryingDrude plasma | Dielectric redshift (2.25 → 4.00) & blueshift (4.00 → 1.44), flash-ionized Drude plasma ω_p2/ω₁ ∈ [0, 1.4] (Wilks 1988), and σ(t) step |
| Variation | Task 2 · Photonic time crystal (PTC) momentum band structure | CosineModulation ε_r(t) = 2.25(1 + 0.25 cos Ωt) across 21 wavenumbers k/k_c ∈ [0.55, 1.45] vs Floquet monodromy & Mathieu theory (Lyubarov 2022) |
| Variation | Task 3 · PTC growth scaling vs m & Moussa et al. (2023) temporal slab | PTC peak growth γ_max/Ω vs m ∈ [0, 0.40] + temporal Fabry–Pérot slab ε₁ → ε₂ → ε₁ (SlabTimeProfile, Moussa et al. 2023) |
| Variation | Task 4 · Spatial & temporal O(Δz² + Δt²) convergence & dynamic CFL map | Resolution sweep cpw ∈ {20, 28, 40, 56, 80} in highest-index state confirming 2nd-order convergence & step-down CFL boundary |
