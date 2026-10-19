# B3 gap-enhancement plateau of graded triangles: PML or reconstruction?

## 1. Question

Graded triangles (suite arm `tri-graded`) on B3 (Au dimer, a = 20 nm, host n = 2, TE, gaps 2/5/10 nm) show `Q_ext` converging but the gap enhancement error `enh_peak` levelling at ~1-2 %.

Hypothesis from the benchmark agent: radial PML field reflection 3.5-5e-3 (the `boundary_reflection` fit metric).

## 2. Diagnosis

Runs on B3-g5 (Cloud Batch, one task per `n2-standard-2`; scripts `b3_pml2.py --phase diag|diag2` and `../unstructured-pml/b3_pml_reflection.py`).

### Table A: PML variants and gap reconstruction on B3-g5

`h` in nm; `enh_peak` error with the suite's pointwise Raviart-Thomas (RT) 9-point stencil vs a second-order least-squares patch reconstruction (12 nearest host edges, linear fit) on the same stencil; `boundary_reflection` fit metric:

| variant | h | RT enh_peak | patch enh_peak | q_ext_nrms | boundary_reflection |
|---|---|---|---|---|---|
| base (10 PML rings of h_out) | 1.0 | 0.10 % | 0.57 % | 0.31 % | 4.5e-3 |
| base | 0.71 | 1.23 % | 0.42 % | 0.17 % | 3.4e-3 |
| base | 0.5 | 0.70 % | 0.45 % | 0.13 % | 3.7e-3 |
| base | 0.36 | 1.33 % | 0.27 % | 0.10 % | 4.1e-3 |
| 20 rings | 0.71 | 1.09 % | 0.27 % | 0.16 % | 8.1e-4 |
| 20 rings | 0.36 | 1.15 % | 0.09 % | 0.08 % | 7.2e-4 |
| PML at 200 nm | 0.36 | 0.22 % | 0.04 % | 0.10 % | 1.5e-3 |

### Table B: Boundary influence on B3-g5

Boundary influence = max over the band of `|enh(variant) - enh(80 rings)| / max enh(80 rings)` (identical for RT and patch):

| PML | h = 0.71 nm | h = 0.5 nm | unknowns at 0.5 nm |
|---|---|---|---|
| 10 rings (old) | 3.5e-3 | 3.8e-3 | 114k |
| 20 rings | 7.0e-4 | 8.2e-4 | 128k |
| 40 rings | 1.3e-4 | 1.3e-4 | 156k |
| 20 rings in 80 nm | 6.9e-4 | 7.8e-4 | 128k |
| 30 rings in 120 nm | 2.9e-4 | 4.3e-4 | 142k |

### Findings

- The old 10-ring PML perturbs the gap enhancement by ~0.35 % (field reflection ~3.5e-3, first order in |E|^2 as suspected), but this is not the 1-2 % plateau: with 40 or 80 rings the RT error is still 1.08 % (h = 0.71 nm) and 0.50 % (0.5 nm).
- The plateau is the pointwise lowest-order RT reconstruction of E in the gap (first order, mesh-dependent sign: 0.1-1.7 % scatter between neighbouring h). The patch reconstruction is second order (test `tests/test_bench_gap_patch.py`: exact for linear fields, RT is not) and gives 0.26-0.57 %, decreasing with h.
- Boundary influence falls ~5x per doubling of the ring count (3.6e-3, 7.5e-4, 1.3e-4); it depends on the number of rings, not on R target, polynomial order, CFS alpha/kappa or physical thickness (B3-scale vacuum sweep, `b3_pml_reflection.py`).
- The `boundary_reflection` fit metric has a floor of ~6-8e-4 (fit conditioning at k r ~ 1-2.4) and cannot resolve 1e-4; the enh difference to an 80-ring reference is the reliable measure.
- Choice: 40 rings of h_out (boundary influence 1.3e-4, ~1e-4 field reflection equivalent), +37 % unknowns at h = 0.5 nm.

## 3. Results

Arms `tri-graded-pml2` (40 rings, suite RT gap metric, comparable with all store arms) and `tri-graded-pml2-patch` (same runs, patch gap metric), 17 Batch runs, published to `gs://cloud-code-511302-autoresearch/bench-results/<case>/<arm>/`.

### Table C: Unknowns / work to reach target `enh_peak` error

Unknowns / work (unknowns x steps) to reach `enh_peak` 5 %, 2 %, 1 % (log-log interpolation; `<=` = already below at the coarsest run; `n/r` = not reached):

| case | arm | 5 % | 2 % | 1 % | best |
|---|---|---|---|---|---|
| B3-g10 | cart-conformal-all-graded | 141k / 1.2e10 | n/r | n/r | 4.6 % |
| B3-g10 | tri-graded (suite, 10 rings) | <=6.0k / 9.5e7 | 40k / 2.1e9 | 81k / 6.2e9 | 0.43 % |
| B3-g10 | tri-graded-pml2 | <=13.4k / 2.1e8 | 61k / 3.1e9 | 111k / 8.3e9 | 0.36 % |
| B3-g10 | tri-graded-pml2-patch | 14.4k / 2.4e8 | 16.5k / 2.9e8 | 18.1k / 3.5e8 | <0.01 % |
| B3-g5 | cart-conformal-all-graded | 111k / 8.0e9 | n/r | n/r | 4.1 % |
| B3-g5 | tri-graded (suite) | 7.2k / 1.2e8 | 9.0k / 1.7e8 | 77k / 5.7e9 | 0.10 % |
| B3-g5 | tri-graded-pml2 | 15.0k / 2.5e8 | 17.6k / 3.3e8 | 95k / 6.4e9 | <0.01 % |
| B3-g5 | tri-graded-pml2-patch | 21.9k / 4.8e8 | 30.5k / 9.0e8 | 40.5k / 1.5e9 | 0.26 % |
| B3-g2 | cart-conformal-all-graded | 143k / 1.2e10 | n/r | n/r | 3.6 % |
| B3-g2 | tri-graded (suite) | <=35k / 1.7e9 | <=35k / 1.7e9 | n/r | 0.98 % |
| B3-g2 | tri-graded-pml2 | <=56k / 2.7e9 | <=56k / 2.7e9 | n/r | 0.76 % |
| B3-g2 | tri-graded-pml2-patch | 59k / 2.9e9 | 78k / 4.7e9 | 107k / 7.8e9 | 0.59 % |

### Interpretation

- With the RT metric, "cost to 1 %" is dominated by the erratic first-order reconstruction (lucky cancellations at single resolutions, e.g. 0.10 % for `tri-graded` at g5); the patch metric converges monotonically and is the meaningful 1 % number. Gap-enhancement convergence below 1 % is reached with the patch metric for all three gaps (g10 at 18k unknowns, g5 at 40k, g2 at 107k).
- vs graded conformal Cartesian at 5 %: g5 7.4x fewer unknowns / 32x less work (RT), 5.1x / 17x (patch); Cartesian never reaches 2 %.
- Recommendation: the suite should adopt a second-order gap reconstruction for all arms (the Cartesian arms interpolate too); the pointwise RT number should not be used for sub-2 % comparisons.

## 4. Limits

- Single host index.
- Patch fit uses host edges only (needs >= 12 host edges inside the gap, i.e. h <~ g/3).
- Costs from single spot-VM runs (+-30 % time noise, use work).
- Records of `tri-graded-pml` / `tri-graded-repro` in the store predate the stencil gap metric and are not comparable.

## 5. Reproduce

```bash
python b3_pml2.py run --phase diag  --shard I --nshards 20 --outdir DIR
python b3_pml2.py run --phase diag2 --shard I --nshards 12 --outdir DIR
python b3_pml2.py run --phase final --final '{"n_pml": 40}' --shard I --nshards 17 --outdir DIR
python ../unstructured-pml/b3_pml_reflection.py run --set fine
```
