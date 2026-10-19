# Local Cartesian refinement (subgridding) for the Mie cylinder

**Question.** A graded rectilinear grid must keep the fine spacing along every row and column that crosses the scatterer. At res 20, 78 % of its cells are fine along at least one axis. Graded triangles avoid this, but they cost about 2 unknowns per triangle, 4.6/h² per area, against 3/h² for squares. Can a Cartesian grid with nested 2:1 patches, a provably stable interface and an exact adjoint get both advantages?

**Method.** `wakis.subgrid`: `QuadMesh` builds the patch mesh, `SubgridFIT2D` steps it, and the physics guide section "Local Cartesian refinement" gives the derivation.
- Hanging nodes take the linear interpolant of their coarse side, through the prolongation $\mathbf P$. The fine-to-coarse restriction is the exact transpose $\mathbf P^T$, and the mass is row-sum lumped. This is the conforming bilinear finite-element scheme with trapezoidal lumping, i.e. Yee/FIT on every uniform block.
- The operator is symmetric positive semi-definite in the mass inner product. The discrete energy is conserved, the scheme is stable iff $\Delta t < 2/\sqrt{\lambda_\max}$ (computed at setup), and the transposed step is exact.
- Materials are averaged sub-pixel: arithmetic node masses; in TE, a series-across / parallel-along rule for the edge Hodge.
- Plane wave in scattered-field form (no TF/SF box), split-field PML on the coarse frame, Drude/Lorentz ADE.
- Optional two-level local time stepping (`lts=2`, the stabilized leapfrog–Chebyshev scheme of `wakis.unstructured.lts`).

**Problem (benchmark B1).** Dielectric cylinder n = 2, a = 1 µm, x = 2–10 (41 points), TM and TE. The pulse, the 1000 fs window, the extraction (`wakis.cylinder_harmonics`, 3 circles, S-matrix de-embedding) and the error metric (RMS of Q_sca/Q_exact − 1) are identical to `../mie-cylinder-graded`.

## Arms

h_in = λ_min,diel / res with res = 10, 14, 20, 28, 40.
- **Coarse grid.** Uniform with H = 2 h_in = n h_in, i.e. equal cells per *local* wavelength, the same h_out as the graded arms.
- **Patch.** One 2:1 level on every coarse cell that comes closer than a + 2 h_in to the axis. This is the fine band of the graded arms, staircased in coarse cells.
- **PML.** 10 coarse cells of split-field PML.

| arm | domain | time step |
|---|---|---|
| sg | circles 1.9/2.2/2.5 µm, PML from 2.7 µm (as `cart-graded`, `tri-graded`) | global |
| sg-compact | circles 1.25/1.4/1.55 µm, PML from 1.7 µm (as `tri-graded-compact`; no TF/SF box limits it) | global |
| sg-lts, sg-compact-lts | same | two-level LTS, p = 2 |

**Baselines.**
- The graded Cartesian (`SolverFIT3D`, conformal 'all', one-cell slab) and graded triangle arms of `../mie-cylinder-graded`, taken as rerun on branch `feature/tri-fixes` (commit 75f603d, radial-PML triangle arm `tri-graded-pml`). Their JSON files are copied into `data/baselines/`.
- All runs, baselines and subgrid alike, used one uncontended n2-standard-2 Cloud Batch machine each.

## Results

![](figures/a_subgrid_mesh.png)
**(a)** The compact res-10 mesh (one quadrant): hanging nodes (orange) on the 2:1 interface, the cylinder, the extraction circles and the PML start.

![](figures/b_error_vs_cost.png)
**(b)** RMS error of Q_sca vs field unknowns (left) and vs work (right). For LTS, work counts the extra fine-set product: unknowns × steps × (1 + fine fraction).

![](figures/c_cost_to_target.png)
**(c)** Unknowns and work to reach 2 % (light) and 1 % (dark) RMS error (log-log interpolation).

![](figures/d_qsca_subgrid.png)
**(d)** Q_sca vs x at res 20 for the compact subgrid arms, against the exact series.

![](figures/e_verification.png)
**(e)** Verification (`verification.py`):
- reflection of a straight 2:1 interface vs coarse cells per wavelength and angle: order 2 at every angle;
- 3:1 interface, the first-order "inject" mutation, and the measurement floor;
- cavity eigenfrequencies with a patch: order 2, between coarse and fine; the mutation is first order;
- energy over 10⁵ steps: drift 2×10⁻¹⁵; the non-transposed (injection) restriction explodes.

RESULTS_TABLE

## Interpretation

INTERPRETATION

## Limits & caveats

- **One smooth convex scatterer.** The patch is a staircase of coarse cells around a circle. Features that need deeper refinement are the subject of the B3 nanogap case.
- **Domain shapes differ.** The subgrid and Cartesian arms use a square with a PML; the triangle arms use a disk.
- **Interpolated targets.** Values come from log-log interpolation between neighbouring resolutions, so treat ratios as ±15 %.
- **Time per update is implementation-specific.** The subgrid runs on SciPy sparse (one CSR product each for K and Kᵀ). The per-step DFT on 3 × 256 circle points (41 frequencies) is the same in every arm and is a large share of each step at these sizes. Work (unknowns × steps) is the implementation-independent comparison.
- **The LTS accuracy gain is not specific to subgridding.** With p = 2, the dielectric runs at twice the Courant number, where the leapfrog's temporal dispersion cancels more of the spatial dispersion. Global-Δt runs at Courant 0.5/0.7/0.95 (res 20, compact, TM) give 2.36/2.31/2.20 %. The triangle LTS study (`feature/unstructured-lts`) sees the same effect. Compare LTS arms with LTS arms.
- **Spurious interface reflection.** 0.2–0.7 % at 20 coarse cells per wavelength (0–60°), order 2, larger at grazing incidence.
- **The PML, ADE and source are not in the transposed step.** Only the lossless core has an exact adjoint so far (dot-product test).

## Reproduce

```bash
python subgrid_study.py list                                     # 40 tasks
python subgrid_study.py run --shard I --nshards 40 --outdir out  # Cloud Batch, one task per n2-standard-2
python subgrid_study.py plot                                     # figures + data/summary.md
python verification.py                                           # figure (e), < 10 s
```

`data/run1/` holds the first sweep. It is numerically identical except for a slower incident-field evaluation that was replaced by delay tabulation; the times in the tables come from the second sweep, `data/sg*.json`.

Tests: `tests/test_052_subgrid2d.py` covers eigenvalue order, energy over 10⁵ steps, interface reflection order, the dot-product adjoint, LTS energy, Mie res 10 and a Lorentz ADE cylinder. Each test has a mutation that makes it fail.
