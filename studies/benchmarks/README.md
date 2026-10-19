# Benchmark suite: baseline arms

**Question.** On cases that stress meshing (metal particles with a surface plasmon resonance, a nanometre gap, a wire antenna), how many unknowns, how much work (unknowns × steps) and how much uncontended wall time does each existing mesh/scheme need to reach a given accuracy?

The cases, references and metrics are defined in [`wakis/benchmarks`](../../wakis/benchmarks/README.md).
- **Simulations** (markers in the figures) are labelled "Wakis FIT (mesh)".
- **References** (lines) are labelled "exact": the series, the T-matrix or the Hallén MoM.

**Compute.**
- Every simulation ran as one Cloud Batch task on its own spot machine: n2-standard-2 for B2/B3, n2-highmem-2 for B4.
- The code was shipped as a tarball through the project bucket.
- Every record is in `data/<phase>/` and in the shared store `gs://cloud-code-511302-autoresearch/bench-results/<case>/<arm>/` (`wakis.benchmarks.results.fetch(case)`).
- Run times scatter by 2–5× between spot machines at equal size, so the *work* column is the implementation-independent comparison.

## Arms

| arm | mesh | absorber / source |
|---|---|---|
| cart-staircase-uniform / -graded | rectilinear, legacy STL cell masks | TF/SF box, CPML |
| cart-conformal-all-uniform / -graded | rectilinear, diagonal subpixel averaging; dispersive cut cells via the passive sub-capacitor network | TF/SF box, CPML |
| tri-uniform / -graded | `cluster_mesh`: interface rings, greedy Poisson-disk fill, Voronoi Hodge | scattered-field plane wave (dispersive source), radial CFS-PML |

The resolution label is h, the cell or edge size on the particles: a/5 … a/40, i.e. 4 → 0.5 nm.
- **Graded meshes** keep h on the particles plus a 2h band, then grow by 1.2 per cell up to min(4h, λ_host/20).
- **Uniform meshes** use h everywhere.

## B2: Au and Ag nanocylinders (168 runs)

![](figures/b2_qext_spectra.png)
**(a)** Q_ext at h = 1 nm, against the exact series with the same Rakić ε(ω).
- Staircase: red-shifted shoulder from corner modes plus the jagged perimeter.
- Conformal: small excess on the red side, from mixed metal/host cells.
- Triangles: on the line.

![](figures/b2_convergence.png)
**(b)** Q_ext NRMS error and LSPR-wavelength error vs field unknowns (TM: Q_abs, since there is no peak).
- Triangles converge at second order.
- Conformal Cartesian TE converges non-monotonically.
- Staircase converges at first order.

![](figures/b2_error_vs_time.png)
**(c)** The same error against uncontended run time.

![](figures/b2_qabs_spectra.png)
**(d)** Q_abs at h = 1 nm.

### Cost to target (TE, from `data/summary_B2.md`)

| case, target | tri-graded | tri-uniform | cart-conformal-graded | cart-conformal-uniform | staircase |
|---|---|---|---|---|---|
| Au, Q_ext NRMS 1 % | 4.8k unknowns, 6.5e7 work, 5 s | ≤ 13k (met at the coarsest run) | 46k, 2.0e9, 86 s | 170k, 7.4e9, 193 s | not reached (2.7 % at 0.5 nm) |
| Au, LSPR λ 0.2 % | 5.0k | 14k | 67k | 244k | not reached |
| Au, FWHM 2 % | ≤ 4.3k | ≤ 13k | 56k | 235k | not reached |
| Ag, Q_ext NRMS 1 % | 4.8k | ≤ 13k | 42k | 154k | not reached |
| Ag, FWHM 2 % | ≤ 3.9k | ≤ 13k | 76k | 318k | not reached |

**Main result.** At 1 % on the metal TE spectra, graded triangles need about 10× fewer unknowns and 30× less work than the best Cartesian arm (graded conformal). The Cartesian slab counts 3 unknowns per cell, as in the earlier Mie studies.

**Why:**
- **Triangles.** The triangle mesh follows the circle. The series (edge-normal) Hodge then keeps D_n continuous across the metal surface, so the surface-charge (LSPR) physics is represented without geometric error at first order.
- **Cartesian conformal.** Kottke averaging of a metal (ε ≈ −4 … −30) and its host produces cells whose parallel-averaged ε(ω) crosses zero, or whose series average has a pole, somewhere in the band. These cells add weak spurious resonances, and the TE error is not monotone in h (B2-Au-TE: 2.3, 2.0, 2.9, 1.3, 1.4, 0.8, 0.6 %).
- **Cartesian staircase.** It is first order, with extra corner modes.
- **TM (E along the axis).** TM is easy for every conformal arm (≤ 0.05 % NRMS at h = 4 nm, uniform). In the graded arms the coarse outer region (4h) dominates the error once the metal's strong scattered field reaches the extraction circles, so graded is *worse* than uniform at equal h. This is an outer-mesh effect, not a metal effect.

## B3: Au dimer gap (102 runs)

![](figures/b3_reference_field_map.png)
**(a)** Exact T-matrix log10 |E/E0|² at the gap-mode peak for g = 5 nm and g = 2 nm, and the graded triangle mesh (h = 2 nm, polar PML rings in green).

![](figures/b3_gap_enhancement_spectra.png)
**(b)** Gap enhancement (mean |E/E0|² over the gap stencil) at h = 1 nm against the T-matrix.
- Staircase splits and shifts the gap mode: at g = 5 nm it shows a second, red-shifted peak.
- Conformal Cartesian under-shoots the peak by 5–10 %.
- Triangles sit on the line.

![](figures/b3_qext_spectra.png)
**(c)** Q_ext (per 4a) at h = 1 nm.

![](figures/b3_convergence.png)
**(d)** Peak gap-enhancement error and Q_ext NRMS vs unknowns.

### Cost to target (from `data/summary_B3.md`)

| case, target | tri-graded | tri-uniform | cart-conformal-graded | staircase |
|---|---|---|---|---|
| g = 10 nm, gap enhancement 5 % | met at coarsest run (≤ 6.1k) | ≤ 18k | 141k, 659 s | not reached |
| g = 10 nm, Q_ext NRMS 1 % | 7.0k | 22k | 64k | not reached |
| g = 5 nm, gap enhancement 5 % | 7.2k, 1.2e8 work, 11 s | 28k | 111k, 8.0e9, 481 s | not reached |
| g = 5 nm, Q_ext NRMS 1 % | 10.5k | 35k | 109k | not reached |
| g = 2 nm, gap enhancement 5 % | met at coarsest (h = 1 nm) run (≤ 35k) | ≤ 204k | 143k | not reached |
| g = 2 nm, Q_ext NRMS 1 % | met at coarsest run (≤ 35k) | ≤ 204k | not reached (1.7 % at 0.5 nm) | not reached |

**Main result.** For the 5 % gap-enhancement target, graded triangles need 10–20× fewer unknowns and about 65× less work than graded conformal Cartesian (g = 5 nm: 7.2k vs 111k unknowns, 1.2e8 vs 8.0e9 work). Staircase never reaches 5 % on these ladders: its errors are 12–120 % and non-monotone.

**Triangle floor at 1–2 %.** The triangle gap-enhancement error levels off at about 1–2 % (non-monotone), while Q_ext keeps converging. That is about twice the PML reflection the triangle arms report (3.5–5e-3 in field), and |E|² responds to it at first order. A better triangle absorber is needed for the 2 % target, and B3 uses only the raw outgoing coefficients (no de-embedding).

**Metric definition.** The metric is the mean over the gap stencil, not the pointwise centre value.
- The first B3 ladder used the centre value, plus a mesher that let the two outer interface rings collide in the gap (a 2.8° sliver at h = 2.86 nm).
- That combination gave 1–9 % non-monotone errors.
- The stencil (centre plus 8 points at g/4) and the priority-ring mesher remove the cell-position noise. The centre value is still reported (`enh_centre`).

**DFT window.** The window floor is ≤ 8e-6 on every B2/B3 metric (`data/window_floor.md`: 1× vs 3× window, finest arms). The plasmon modes (Q ≈ 3–9) ring down within a few fs, unlike the high-Q TE mode that limits B1, whose window is now 3000 fs.

## B4: dipole

(see below)

## Limits & caveats

- **B2 is 2D.** Spheres (−2ε_b resonance, 3D staircase) are harder for the Cartesian arms and are not covered.
- **Ladder too fine for some targets.** Some targets are met at the coarsest run (marked "≤"), so the ladder does not bracket them. The TM ladder in particular should be extended to coarser h.
- **Triangle absorber.** The triangle arms use a 10-ring CFS-PML of h_out cells; its reflection (3–8e-3, reported per record) sets an error floor of about 1e-3 in the B3 gap metrics.
- **Spot-machine timing.** Run times come from single spot runs; the work column is the comparison to trust.

## Reproduce

```bash
python bench_study.py list --phase B2                       # 168 tasks
python bench_study.py run --phase B2 --shard I --nshards N --outdir out [--publish]
python bench_study.py plot --phase B2                       # figures + data/summary_B2.md
```

Tests: `tests/test_bench_multicylinder.py`, `test_bench_cases.py`, `test_bench_dispersive_planewave.py`, `test_bench_thin_wire.py`.
