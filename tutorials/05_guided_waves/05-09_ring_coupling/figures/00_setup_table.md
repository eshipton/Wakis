| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 12.2 µm × 7.47 µm × 26.3 nm (z ∈ [-13.2 nm, 13.2 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 713 × 368 × 1 (262384 cells) |
| Discretization | Δx & resolution | Δ_min = 6.24 nm, 87.3 cells/λ (in n_max = 2.85), 8.0 cells across feature (50 nm) |
| Boundary | -x face | CPML (10 cells, 262 nm) |
| Boundary | +x face | CPML (10 cells, 262 nm) |
| Boundary | -y face | CPML (10 cells, 256 nm) |
| Boundary | +y face | CPML (10 cells, 263 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | SiO₂ cladding (n=1.444, ε_r=2.09) (isotropic) |
| Material | Region 1: Si EIM bus/ring core (n_Si=3.476, h=220 nm → n_eff,slab=2.8478, w=500 nm) | ε_r = 8.11, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-7.50 µm, 7.50 µm] (d = 15 µm) |
| Material | Region 2: Si EIM bus/ring core (n_Si=3.476, h=220 nm → n_eff,slab=2.8478, w=500 nm) | ε_r = 8.11, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-5.25 µm, 5.25 µm] (d = 10.5 µm) |
| Source | 1 · ModeSource | pos: x = -5.51 µm, y = 0.00 µm, pol: TE₀ (E_y, H_z), pulse: f₀ = 193.6 THz, Δf = 6.2 THz |
| Monitor | 2 · ModeMonitor | Port 1 'in' (S₁₁), pos: x = -5.39 µm, y = 0.00 µm, freqs: 51 freqs in [187.4, 199.9] THz |
| Monitor | 3 · ModeMonitor | Port 2 'through' (t = S₂₁), pos: x = 5.39 µm, y = 0.00 µm, freqs: 51 freqs in [187.4, 199.9] THz |
| Monitor | 4 · ModeMonitor | Port 3 'drop' (κ = S₃₁), pos: x = 5.00 µm, y = 5.80 µm, freqs: 51 freqs in [187.4, 199.9] THz |
| Monitor | 5 · ModeMonitor | Port 4 'add' (S₄₁), pos: x = -5.00 µm, y = 5.80 µm, freqs: 51 freqs in [187.4, 199.9] THz |
| Variation | Coupling-gap sweep (task_9..task_12) | g ∈ {50, 100, 150, 200, 250} nm at L3=(20 cpw, 8 cells/gap) |
| Variation | 3-port Lorentz reciprocity & CPML extrusion audit (task_4) | Drives on 'in', 'through', 'drop' + companion unextruded-CPML run |
| Variation | Graded non-uniform Cartesian & hybrid unstructured ring mesh (task_13) | Mirror-symmetric graded Δy (8.00 cells across 50 nm slot) & non-square Δx > Δy_min (165,487 cells, −37.2% cells, 650.2 s, −54.8% wall time vs single-drive L3) + wakis.unstructured subtri ring rim |
