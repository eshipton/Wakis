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
> **Note (`m2` rerun pending):** The B3 tables and figures above come from the pre-`m2` runs (pointwise Raviart–Thomas gap reconstruction and 10-ring triangle PML); they will be replaced by the `--phase B3m2` rerun using the default `m2` metric (second-order 12-edge patch reconstruction on triangles, bilinear on Cartesian, and 40-ring triangle PML).

**Metric definition.** The metric is the mean over the gap stencil, not the pointwise centre value.
- The first B3 ladder used the centre value, plus a mesher that let the two outer interface rings collide in the gap (a 2.8° sliver at h = 2.86 nm).
- That combination gave 1–9 % non-monotone errors.
- The stencil (centre plus 8 points at g/4) and the priority-ring mesher remove the cell-position noise. On triangles, `e_gap` now uses a second-order least-squares linear patch fit over the 12 nearest host edges (`e_gap_rt` retains the pointwise Raviart–Thomas diagnostic, `enh_rt_peak`). The centre value is still reported (`enh_centre`).

**DFT window.** The window floor is ≤ 8e-6 on every B2/B3 metric (`data/window_floor.md`: 1× vs 3× window, finest arms). The plasmon modes (Q ≈ 3–9) ring down within a few fs, unlike the high-Q TE mode that limits B1, whose window is now 3000 fs.

## B4: centre-fed dipole (Cartesian 3D)

![](figures/b4_dipole_zin.png)
**(a)** Z_in(f): Hallén MoM reference (lines) vs the finest run of each Cartesian arm (markers).
**(b)** Relative errors of f_res and R_res vs the transverse cell h_t.
- The axial cell is h_z = min(g, 4 h_t): 1 mm for h_t ≥ a/2 and 0.5 mm for a/3 and a/4.
- Rungs h_t ≥ a/1.5 ran on n2-highmem-2 CPUs (50–120 min); the finer rungs ran on L4 GPUs (6–23 min).

| arm | h_t | h_z | f_res err | R_res err | directivity err |
|---|---|---|---|---|---|
| edge-wise staircase | a (0.5 mm) | 1 mm | 1.16 % | 0.8 % | 0.10 % |
| edge-wise staircase | a/1.5 | 1 mm | 4.8 % | 7.5 % | 0.25 % |
| edge-wise staircase | a/2 | 1 mm | 3.6 % | 5.5 % | 0.17 % |
| edge-wise staircase | a/3 | 0.5 mm | 1.74 % | 3.5 % | 0.00 % |
| edge-wise staircase | a/4 | 0.5 mm | 1.79 % | 3.6 % | 0.03 % |
| edge-wise staircase | a/3 | 0.25 mm | 1.69 % | 3.5 % | 0.04 % |
| edge-wise staircase | a/3 | 0.125 mm | 1.67 % | 3.5 % | 0.03 % |
| edge-wise, 2× domain / λ/40 outer mesh | a/2 | 1 mm | 3.59 % / 3.62 % | 5.5 % / 5.6 % | 6 % / 0.3 % |
| legacy STL staircase | a, a/1.5 | 1 mm | 6.2 %, 5.2 % | 8.6 %, 7.7 % | 0.4 % |
| conformal PEC (STL; **mask bug**) | a, a/1.5, a/2 | 1 mm | 6.0 %, 4.5 %, 4.0 % | 7.7 %, 5.8 %, 5.2 % | 0.3 % |

**Reading the ladder:**
- **The h_z dependence is first order down to 0.5 mm.** Halving the axial (gap) cell from 1 mm to 0.5 mm halves the error (3.6 % → 1.7 %). Below 0.5 mm the edge-wise staircase stops changing:
  - h_z = 0.5, 0.25, 0.125 mm give 1.74 %, 1.69 %, 1.67 %;
  - h_t = a/3 and a/4 give 1.74 % and 1.79 %.
- **Converged result.** The edge-wise Cartesian FIT dipole converges to f_res 1.7 % below, and R_res 3.5 % below, the reduced-kernel Hallén reference. The broadside directivity agrees to 3e-4.
- **The h_t = a point (1.16 %) is a lucky cancellation**, not convergence. The coarse cross-shaped PEC wire is effectively thicker, which pulls the error the other way. The cost-to-target rule ("stays below target") does not count it.
- **The error is not from the domain.** Doubling the domain or the outer resolution changes f_res by less than 0.03 %. One exception: the near-to-far box of the 2× domain run gives a 6 % directivity error, which is unexplained (the box is still outside the CPML); Z_in is unaffected.
- **The legacy STL staircase and the conformal-PEC arm are both thickened.** They carry the same thickened wire (physics guide; `problems_solutions.md`). The conformal-PEC records are flagged in the store until `fix/conformal-pec-legacy-mask` lands.

**Open: the 1.7 % / 3.5 % converged offset.** Two candidates:
1. **The reference.** Thin-wire theory differs from the solved geometry:
   - the reduced kernel puts the current on the axis;
   - the wire has no end caps;
   - the gap is modelled as a uniform axial field, whereas FIT solves flat-ended 0.5 mm cylinders with a real 1 mm gap between flat faces.
2. **FIT.** A non-vanishing staircase/feed-model error that does not shrink with h_t or h_z.

The MoM's own N-spread is 0.09 % / 0.06 %, so the offset is not MoM discretization. Two cheap hypotheses were tested with the MoM, and neither closes the offset:
- **Equivalent radius of the staircase wire.** At h_t = a/3 the PEC node set (29 nodes) is a staircased disk, not a square. Its outermost node lies at r = a, and the area-equivalent radius is 0.506 mm. Matching f_res would need a_eq ≈ 2a, and a thicker wire *raises* R_in (+1.5 % at a_eq = 1 mm) while FIT's R is lower.
- **Flat end caps.** Adding a or 2a to the length explains −0.33 … −0.65 % of the f_res offset and none of the R_in offset.
- **Feed width.** It matches: the FIT port spans exactly the 1 mm gap between the face planes, as in the MoM's uniform gap field.

The R_in offset (−3.5 %) is therefore not geometric thickening. It is logged in `open_questions.md`. Two independent checks would settle it: an exact-kernel (or body-of-revolution) reference, and the conformal-PEC arm after the mask fix.
Until then, the B4 targets are scored against the Hallén reference, and the offset is reported as part of the baseline.

**Cost.**
- Converged rung (h_t = a/3, h_z = 0.5 mm): 2.4 M cells and about 55k steps. It took 15 min on an L4 GPU; on one CPU core it would be about 7 h, so the CPU rungs beyond a/2 timed out at 330 min.
- h_t = a: 0.85 M cells, 19k steps, 49 min on n2-highmem-2.

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
