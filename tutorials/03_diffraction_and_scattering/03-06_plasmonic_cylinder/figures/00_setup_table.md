| Category | Component / Face | Specification |
| :--- | :--- | :--- |
| Domain | Size & bounds | Lx × Ly × Lz = 184 nm × 184 nm × 160 nm (z ∈ [-80 nm, 80 nm]) |
| Domain | Grid cells | Nx × Ny × Nz = 46 × 46 × 4 (8464 cells) |
| Discretization | Δx & resolution | Δ_min = 4 nm, 35.6 cells/λ (in n_max = 2.11), 5.6 cells across feature (22.5 nm) |
| Boundary | -x face | CPML (9 cells, 36 nm) |
| Boundary | +x face | CPML (9 cells, 36 nm) |
| Boundary | -y face | CPML (9 cells, 36 nm) |
| Boundary | +y face | CPML (9 cells, 36 nm) |
| Boundary | -z face | Periodic |
| Boundary | +z face | Periodic |
| Material | Background | Vacuum, ε = 1.00 (isotropic) |
| Material | Region 1: Ag cylinder (N = 2 Drude–Lorentz) | ε_r = 4.44, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-20 nm, 20 nm] (d = 40 nm) |
| Source | 1 · TFSFSource | pos: x ∈ [-36 nm, 36 nm], pol: E_y, pulse: f₀ = 666.2 THz, Δf = 166.6 THz |
| Monitor | 2 · FluxMonitor | Outer FluxMonitor (P'_sca), pos: x, y ∈ [-44, 44] nm, freqs: 241 freqs in [333.1, 999.3] THz (300–900 nm) |
| Monitor | 3 · FluxMonitor | Inner FluxMonitor (P'_abs), pos: x, y ∈ [-28, 28] nm, freqs: 241 freqs in [333.1, 999.3] THz (300–900 nm) |
| Variation | Metals & diameters | Ag (2a = 40 nm and 80 nm) and Au (2a = 40 nm) |
| Variation | Dispersive pole models | N = 1, 2, 3, 4, 6 fitted to CC0 Johnson & Christy (1972) + Rakić (1998) + constant-ε baseline |
| Variation | Resolutions & subpixel | h = 4 nm (quick) and h = 2 nm (full), Kottke subpixel averaging (wakis.tensor_dispersive) |
