| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 5.19 µm × 4.29 µm × 500 nm (z ∈ [-250 nm, 250 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 746 × 608 × 1 (453568 cells) |
| Discretization | Δx & resolution | Δ_min = 3 nm, 61.2 cells/λ (in n_max = 2.45), 5.1 cells across feature (15.4 nm) |
| Boundary | -x face | CPML (8 cells, 59.9 nm) |
| Boundary | +x face | CPML (8 cells, 59.9 nm) |
| Boundary | -y face | CPML (8 cells, 59.7 nm) |
| Boundary | +y face | CPML (8 cells, 60 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Air superstrate (n = 1.00), ε = 1.00 (isotropic) |
| Material | Region 1: Ag reflector & mesa | ε_r = 4.00, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-2.59 µm, 2.59 µm] (d = 5.19 µm) |
| Material | Region 2: SiO₂ passivation (n = 1.46) | ε_r = 2.13, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-500 nm, 500 nm] (d = 1 µm) |
| Material | Region 3: GaN active core (n = 2.45) | ε_r = 6.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-300 nm, 300 nm] (d = 600 nm) |
| Material | Region 4: dGRIN L1: Nb2O5 (n = 2.37) | ε_r = 5.62, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-789 nm, 789 nm] (d = 1.58 µm) |
| Material | Region 5: dGRIN L2: Si3N4 (n = 2.05) | ε_r = 4.20, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.08 µm, 1.08 µm] (d = 2.15 µm) |
| Material | Region 6: dGRIN L3: Al2O3 (n = 1.77) | ε_r = 3.13, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.37 µm, 1.37 µm] (d = 2.73 µm) |
| Material | Region 7: dGRIN L4: SiO2 (n = 1.46) | ε_r = 2.13, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.65 µm, 1.65 µm] (d = 3.31 µm) |
| Material | Region 8: dGRIN L5: MgF2 (n = 1.38) | ε_r = 1.90, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1.94 µm, 1.94 µm] (d = 3.89 µm) |
| Material | Region 9: Ag tapered µHorn (ϑ = 60°) | ε_r = 4.00, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-2.59 µm, -500 nm] (d = 2.09 µm) |
| Material | Region 10: Ag tapered µHorn (ϑ = 60°) | ε_r = 4.00, μ_r = 1.00, model = DispersiveMaterial, x ∈ [500 nm, 2.59 µm] (d = 2.09 µm) |
| Source | 1 · CurrentSource | pos: y = 150 nm, M = 17 sites in x ∈ [-240, +240] nm (5 shown), pol: TE (J_x, J_y) & TM (J_z), pulse: f₀ = 666.2 THz, Δf = 119.9 THz |
| Monitor | 2 · Near2FarMonitor2D | Near2FarMonitor2D aperture plane (y = 3.46 µm, 21 freqs in 425–475 nm), pos: y = 3.46 µm, x ∈ [-2.52, +2.52] µm, freqs: f = 666.2 THz |
| Variation | Task 0 — Bare reference pixel | notes=h = 0, w_top = 1.00 µm, M = 9 incoherent QW dipoles (y_QW = 150 nm) |
| Variation | Task 1 — Air-filled µHorn | notes=h = 3.0 µm, ϑ = 72° (w_top = 2.95 µm), M = 17 incoherent QW dipoles |
| Variation | Task 2 — SiO₂-filled µHorn | notes=h = 3.0 µm, ϑ = 64° (w_top = 3.93 µm, n = 1.46), M = 17 incoherent QW dipoles |
| Variation | Task 3 — 6-layer dGRIN µHorn (primary) | notes=h = 3.0 µm, ϑ = 60° (w_top = 4.46 µm, Nb₂O₅/Si₃N₄/Al₂O₃/SiO₂/MgF₂/Air), M = 17 dipoles |
| Variation | Tasks 4–6 — Angle/height, QW offset & skin-depth convergence | notes=ϑ ∈ [50°, 78°], h ∈ [0.6, 3.0] µm, Δy_QW ∈ [-40, +40] nm, Δx_min ∈ {15, 7.5, 5.0, 3.75, 2.5} nm |
