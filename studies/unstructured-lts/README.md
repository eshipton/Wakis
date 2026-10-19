# Local time stepping for unstructured FIT (2D)

**Question.** Graded triangle meshes cut the unknowns of the Mie-cylinder problem 2.5–3× (`../mie-cylinder-graded`), but the global leapfrog still takes the time step of the finest region. Can a local time stepping (LTS) scheme let most of the mesh take a larger step, keep the accuracy, stay energy-conserving and keep an exact adjoint?

**Method** (`wakis/unstructured/lts.py`; physics guide, "Local time stepping"). This is the leapfrog LTS of Diaz & Grote (2009), stabilized with a damped Chebyshev polynomial in the sense of Grote, Michel & Sauter (2021) and Carle, Hochbruck & Sturm (2020). In second-order form:

y^{n+1} − 2y^n + y^{n−1} = −dt² A r(dt² P A) y^n,  A = M⁻¹K,

- P selects the fine nodes F, and r is a polynomial of degree p − 1 with r(0) = 1.
- K r(X) is symmetric, so the scheme is a leapfrog with a modified stiffness. It conserves an exact discrete energy, is stable iff ρ = dt/2 √λ_max(M⁻¹K r(X)) < 1, is second order, and its transpose r(X)ᵀ = r(Xᵀ) gives an exact adjoint.
- In the solver only the flux update changes: v −= dt G r(X) u.
- At setup, Lanczos computes ρ. The partition is accepted only if ρ ≤ 0.98; otherwise F grows by one layer of neighbours (an interface mode) or dt shrinks (a coarse-region mode).
- Usage: `UnstructuredFIT2D(mesh, ..., lts=2)`.

## Results

![](figures/a_partition.png)
**(a)** LTS levels on the graded Mie mesh (res 10). Inside the n = 2 cylinder and in the far vacuum (h = 2 h_in), the mesh has equal cells per *local* wavelength, so both regions can take about twice the global step. Only the vacuum band of size h_in around the cylinder, plus the interface ring, is fine: 10.5 % of the nodes. That band is two layers of 2 h_in plus the 1.2× growth zone. The coarse step is 1.94× the global step. The earlier estimate of 20–25 % saving assumed the cylinder interior was fine as well. It is not, because the local CFL limit scales with h·n.

![](figures/b_energy.png)
**(b)** PEC cavity (rod ε = 4), random initial field, 10⁵ coarse steps, TM.
- The exact LTS energy stays constant to 1e-14, the same as the global leapfrog.
- Two mutations drift by 0.2 and 2e-3 and fail the test: decoupling F from its coarse neighbours, and r(0) = 1.01.
- The unstabilized ν = 0 polynomial still conserves its energy, but ρ = 1.000. The quadratic form is then no longer a norm, and the field energy beats up to about 300×.

![](figures/c_speedup.png)
**(c)** Mie cylinder, global stepping vs LTS on the same meshes. Panels: Q_sca error vs unknowns, error vs uncontended time, and speed-up.

### Graded Mie meshes (Cloud Batch, one run per n2-standard-2)

The full table is in [`data/summary.md`](data/summary.md). "Work" is unknown-weighted: global = unknowns × steps; LTS = steps × (unknowns + fine nodes + their edges). Global means uniform stepping at the global (minimum) dt on the same mesh.

| pol | arm | res | fine % | dt_LTS/dt | RMS Q_sca global → LTS | work ratio | time ratio (field update) |
|---|---|---|---|---|---|---|---|
| TM | tri-graded | 10 | 10.5 | 1.94 | 0.0688 → 0.0318 | 1.75 | 1.56 |
| TM | tri-graded | 14 | 10.0 | 1.93 | 0.0355 → 0.0234 | 1.74 | 1.42 |
| TM | tri-graded | 20 | 7.1 | 1.95 | 0.0231 → 0.0134 | 1.81 | 2.14 |
| TM | compact | 10 | 23.3 | 1.95 | 0.0581 → 0.0265 | 1.56 | 2.41 |
| TM | compact | 14 | 17.2 | 1.95 | 0.0298 → 0.0211 | 1.65 | 2.31 |
| TM | compact | 20 | 9.3 | 1.95 | 0.0212 → 0.0114 | 1.77 | 1.96 |
| TE | tri-graded | 10 | 10.9 | 1.96 | 0.0915 → 0.0511 | 1.76 | 1.93 |
| TE | tri-graded | 14 | 7.9 | 1.94 | 0.0531 → 0.0247 | 1.79 | 2.62 |
| TE | compact | 10 | 18.3 | 1.96 | 0.0853 → 0.0434 | 1.63 | 1.19 |
| TE | compact | 14 | 13.4 | 1.97 | 0.0482 → 0.0250 | 1.72 | 1.77 |

**Status: partial.** 39 of the 40 runs finished, but only 23 were collected before the gcloud credentials expired. The res 20 TE and the res 28/40 rows are still in `gs://cloud-code-511302-autoresearch/sweeps/unstr-lts-20261019-0615/`. To complete the table: `gcloud storage cp "gs://…/unstr-lts-20261019-0615/shard_*/*.json" data/ && python lts_study.py plot`.

### Interpretation

- **Work saving.** At equal mesh, the unknown-weighted work drops 1.56–1.81×. The saving grows with resolution, because the fine band is a fixed number of cells and so a shrinking fraction of the mesh. The bound is 1.95× (the dt ratio).
- **Wall time is noisy.**
  - Single uncontended runs give time ratios of 1.19–2.62×.
  - A ratio above the dt ratio of 1.95 cannot come from the algorithm. It shows host-to-host speed differences of up to ±30 % between Spot machines, even though one run per machine removes contention.
  - The work ratio is the reliable number. Repeated timing pairs on one machine are still needed (open).
- **Accuracy: equal or better, for a known reason.**
  - LTS lowers the Q_sca error by 1.4–2.2×. A result this good was checked as a possible bug with control runs (`dt_control.py`, [`data/dt_control.md`](data/dt_control.md), res 10):
    - LTS with its coarse step forced equal to the global step reproduces the global error exactly (TM 0.0688, TE 0.0915), so the LTS machinery adds no error;
    - global stepping at half the step is *worse* (0.0762, 0.0990).
  - The error therefore falls as dt grows. This is the known partial cancellation of the spatial dispersion (numerical phase lag) by the leapfrog temporal dispersion (phase lead), which is strongest near the CFL limit (the "magic time step" in 1D).
  - With global stepping, the coarse regions run at about 0.49 of their own limit. With LTS they run at about 0.95, so their dispersion errors cancel better.
  - This is specific to dispersion-dominated errors. It is not a general accuracy gain of LTS: in the cavity eigenvalue test, the LTS time error is 3.4–3.8× the global one (dt ratio squared), as expected.
- **Against the earlier estimate.** The graded study estimated a 20–25 % saving for LTS (half the nodes coarse). The measured work saving is 36–45 % (ratio 1.56–1.81), because only 7–23 % of the nodes are fine.

## Tests (`tests/test_047_unstructured_lts.py`; 15 tests, all passing on Cloud Batch with the slow ones)

| property | result | mutation that fails it |
|---|---|---|
| `apply` = reference recurrence = sparse matrix (p = 2, 3, 4); r(X)y = y off F | 1e-13 | — |
| K r(X) symmetric | < 1e-13 relative | — |
| exact energy, 10⁵ steps, TM and TE | drift < 1e-11 (observed 2e-14) | buffer decoupled: 0.19; r(0) = 1.01: 2e-3 (test requires > 1e-4) |
| ν = 0 | ρ > 0.999, field energy grows > 10× | (the test asserts the instability) |
| eigenfrequency (rod ε = 4 in PEC cavity, graded h = 0.02/0.01) | LTS time error 2nd order (1.6 < order < 2.4) and 2.5–5× the global one; total error < global + 2 × spatial, < 1e-4 | — |
| `one_step` = `linear_step` (the analysed operator) | 1e-12 | — |
| adjoint dot product, one step TM and TE | < 1e-14 | r(X) used instead of r(X)ᵀ: > 1e-6 |

## Reproduce

```bash
python lts_study.py list                      # 40 tasks
python lts_study.py run --shard I --nshards 40 --outdir out   # Cloud Batch, one per n2-standard-2
python energy_runs.py                          # 5 x 1e5 steps (cloud or VM; ~15 s on one core)
python lts_study.py plot                       # figures + data/summary.md
```

The Batch job used the autoresearch image and service account, with the code shipped as a `git archive` tarball through the project bucket (as in `../mie-cylinder-graded`).

## Limits

These are summarized here; the physics guide has the full list.
- Two levels only; no dispersive media; no loss on fine nodes. The ABC is in the coarse region here.
- Sources and monitors run at the coarse rate.
- The coarse time error is about 4× larger (dt²). On problems dominated by temporal error, LTS loses accuracy.
- One smooth scatterer and one grading. Meshes with slivers would create scattered small F sets.
- The 3D tetrahedral and Cartesian versions are not implemented. The physics guide lists what they need.
