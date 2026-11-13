# Unstructured adjoint: focusing into the shadow of a PEC cylinder

This study is a worked inverse design with the exact discrete adjoint of `UnstructuredFIT2D` (`wakis.unstructured.adjoint`; physics guide section `docs/physicsguide_sections/unstructured_adjoint.md`). A z-directed line source sits 2 λ0 in front of a PEC cylinder of radius 0.5 λ0. A dielectric annulus 0.5 λ0 < r < 1.25 λ0 around the cylinder (ε_r ∈ [1, 4] per triangle) is optimized to maximize |E_z(f0)|² at a focus 1.75 λ0 behind the cylinder, deep in its shadow. Both the cylinder boundary and the annulus are conforming rings of triangle edges, so no staircase is involved. No Cartesian run was made: the only baseline is the bare cylinder on the same mesh.

## Setup

| item | value |
| :--- | :--- |
| mesh | `graded_disk_mesh`: 20172 nodes, 40027 triangles, 15429 design triangles |
| resolution | h = λ0/40 (20 cells per wavelength in ε = 4) in the annulus; graded ×1.2 per ring to λ0/20 in the outer vacuum |
| boundary | radial CFS-PML, 8 polar-aligned rings (dr = 0.06 λ0, R = 1e-6), backed by PEC |
| source | Gaussian-modulated sine at f0 (τ = 4/f0, t0 = 3τ) at (−2, 0) λ0 |
| run | Δt = 5.32e-11 s (frozen, built at ε_min), N_t = 3763 steps (60 periods) |
| objective | J = \|X(f0)\|² at (1.75, 0) λ0, normalized by the bare cylinder, J0 |
| parametrization | cone filter R = 0.08 λ0; tanh projection with β = 1, 4, 8, 16, 32; L-BFGS-B on −log J, 12 iterations per stage (68 evaluations) |
| compute | Cloud Batch n2-standard-4, job `uadj-demo-1`: 795 s for the whole script (FD check in 4 processes) |

## Results (`results.json`, `window.json`)

* **Gradient check** at the start design (ρ = 0.5, β = 1). Adjoint vs central FD (h = 1e-3) on 12 triangles: the 6 with the largest |g| and 6 at random.
  * The absolute difference is 1–3e-10 for every cell, independent of |g| (0.02 to 0.55). This is FD round-off (J ≈ 73: 3e-15 relative noise in J over 2h), not a gradient error, which would scale with |g|.
  * In relative terms that is 5e-11 to 7e-10 for the large-gradient cells and 1.4e-8 for the smallest (|g| = 0.02).
* **Checkpointing**: the gradient of the final design with `memory="checkpoint"` (4 slots) is bitwise equal to the stored-row gradient. Peak memory is 19.7 MB instead of 235 MB of stored rows, at a cost of 6024 recomputed steps.
* **Objective**:
  * The bare shadow is J0/J_free = 0.0191 (−17.2 dB) of the free-space intensity at the focus.
  * Over the 60-period window that was optimized, J/J0 = 1517 for the binarized design (1524 at β = 32).
  * **The window is not converged for the optimized design**: it is a resonant ring and keeps ringing. With 2, 4 and 8 N_t, J/J0 = 1384.0, 1392.14 and 1392.11. **The converged value is J/J0 = 1392 (+31.4 dB), i.e. 26.5× (+14.2 dB) the free-space intensity at the focus.** The 60-period figure overstates it by 9.0 %.
  * The bare J0 is window-converged to 5e-7.
* The design is mirror-symmetric about the source–focus axis: symmetry is not imposed, but it is preserved by the symmetric problem and gradient.

## Figures (`figures/`)

* `00_setup_mesh.png`: setup (PEC, design annulus, vacuum, PML, source, focus) and a zoom of the conforming triangles at the cylinder.
* `01_design_evolution.png`: ε_r per triangle at β = 1, 8 and 32.
* `02_objective.png`: J/J0 per L-BFGS-B evaluation, with the β stages shaded, the binarized value and the free-space level.
* `03_gradient_vs_fd.png`: adjoint vs FD, and the relative differences with the truncation estimate and the Richardson value.
* `04_fields.png`: |E_z(f0)|² in dB relative to free space at the focus, for the bare cylinder and the binarized design.

## Reproduce

```bash
bin/batch-run uadj-demo-1 WT studies/unstructured-adjoint -- bash -c \
  "python studies/unstructured-adjoint/shadow_focus.py && python studies/unstructured-adjoint/make_figures.py"
bin/batch-run uadj-window-1 WT studies/unstructured-adjoint/window.json -- bash -c \
  "cd studies/unstructured-adjoint && python check_window.py"
```

`shadow_focus.py --quick --procs 1` is a coarse, 2-iterations-per-stage smoke run of about 10 s.

## Lessons (for the project logs)

* **Degenerate test.** A barycentric-vs-Voronoi mutation of the design map passed the FD test on the hexagonal `disk_mesh`, where almost all triangles are equilateral and the two partial areas coincide. The test mesh is now jittered.
* **Non-deterministic Δt.** `UnstructuredFIT3D` with massless edges computes λ_max by an unseeded `eigsh`, so two builds of the same mesh differ in Δt at the Lanczos tolerance. Pass `dt` explicitly; a seeded `v0` in solver3d would fix it.
* **LTS and a frozen dt.** For a frozen dt with LTS, pass `lts=dict(dt=...)`. A constructor `dt` is the global step, and the coarse step becomes p·dt.
* **Window truncation.** A high-Q design can exploit ringing that the DFT window cuts off. Always check the objective of the final design at 2–4 N_t, and consider optimizing with a longer window or a decaying apodization.
