| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 13.2 µm × 14.4 µm × 26.3 nm (z ∈ [-13.2 nm, 13.2 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 503 × 560 × 1 (281680 cells) |
| Discretization | Δx & resolution | Δ_min = 12.5 nm, 43.5 cells/λ (in n_max = 2.85), 8.0 cells across feature (100 nm) |
| Boundary | -x face | CPML (12 cells, 316 nm) |
| Boundary | +x face | CPML (12 cells, 316 nm) |
| Boundary | -y face | CPML (12 cells, 313.7 nm) |
| Boundary | +y face | CPML (12 cells, 313.7 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | SiO₂ cladding (n=1.444, ε_r=2.09) (isotropic) |
| Material | Region 1: Si EIM bus/ring core (n_Si=3.476, h=220 nm → n_eff,slab=2.8478, w=430 nm) | ε_r = 8.11, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-6.30 µm, 6.30 µm] (d = 12.6 µm) |
| Material | Region 2: Si EIM bus/ring core (n_Si=3.476, h=220 nm → n_eff,slab=2.8478, w=430 nm) | ε_r = 8.11, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-6.30 µm, 6.30 µm] (d = 12.6 µm) |
| Material | Region 3: Si EIM bus/ring core (n_Si=3.476, h=220 nm → n_eff,slab=2.8478, w=430 nm) | ε_r = 8.11, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-5.21 µm, 5.21 µm] (d = 10.4 µm) |
| Source | 1 · ModeSource | pos: x = -5.14 µm, y = 5.53 µm, pol: TE₀ (E_y, H_z), pulse: f₀ = 193.6 THz, Δf = 6.2 THz |
| Monitor | 2 · ModeMonitor | Port 1 'in' (S_in,in), pos: x = -5.00 µm, y = 5.53 µm, freqs: 1001 freqs in [187.4, 199.9] THz |
| Monitor | 3 · ModeMonitor | Port 2 'through' (S_through,in), pos: x = 5.00 µm, y = 5.53 µm, freqs: 1001 freqs in [187.4, 199.9] THz |
| Monitor | 4 · ModeMonitor | Port 3 'drop' (S_drop,in), pos: x = -5.00 µm, y = -5.53 µm, freqs: 1001 freqs in [187.4, 199.9] THz |
| Monitor | 5 · ModeMonitor | Port 4 'add' (S_add,in), pos: x = 5.00 µm, y = -5.53 µm, freqs: 1001 freqs in [187.4, 199.9] THz |
| Variation | Resolution & subpixel convergence ladder (task_0..task_6) | Kottke & staircase at L1=(10,4), L2=(14,6), L3=(20,8) vs Kottke reference R=(40,16) |
| Variation | Coupling-gap parametric sweep (task_4, task_7, task_8) | g ∈ {60, 100, 150} nm at L3=(20 cpw, 8 cells/gap) — over- vs under-coupling |
| Variation | Graded non-uniform Cartesian & hybrid unstructured ring mesh (task_11) | Mirror-symmetric graded Δy (8.00 cells across both 100 nm gaps) & non-square Δx, Δy + wakis.unstructured subtri ring annulus |
