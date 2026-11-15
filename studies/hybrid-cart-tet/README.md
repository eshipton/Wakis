# Hybrid Cartesian / tetrahedral FIT (`wakis.hybrid`), milestone 1

A tetrahedral box patch inside the Cartesian FIT grid. The coupling uses hanging tetrahedral edges on the hexahedral Nédélec trace of the Cartesian squares. Derivation, proofs, results and limits are in [`docs/physicsguide_sections/hybrid_cart_tet.md`](../../docs/physicsguide_sections/hybrid_cart_tet.md); figures are catalogued in `docs/figures.md`.

## Reproduce

All runs go to Cloud Batch. The local tests are `tests/test_105_hybrid_cart_tet.py` (16 tests, about 8 s).

```bash
# from the repo root of the CEM project, worktree WT
bin/batch-run JOB WT studies/hybrid-cart-tet --machine n2-highmem-16 -- \
    bash -c "python studies/hybrid-cart-tet/run_study.py --parts A --procs 14"
# parts: A (TE10 |R|, fixed patch length), B (cavity), C (PEC sphere; --only 8,10,12 / 16,20,24),
#        D (time domain: lambda bounds, cost, energy; --only 10 / 16), E and F (patch-length envelope,
#        N <= 24 and N = 28..40); extend_A.py: part A at N = 20, 24, 28
python studies/hybrid-cart-tet/plot_figures.py   # N = 10 solve for the mesh/field figures (~2 GB);
                                                 # HYBRID_SKIP_SOLVE=1 renders only the data figures
```

| Part | Batch job(s) | Machine | Result file |
|---|---|---|---|
| A | `w2hyb-a`, `w2hyb-a2`, `w2hyb-a4` | n2-highmem-16 | `results_A.json`, `results_A_8-12-16.json`, `results_A_ext.json` |
| B | `w2hyb-b` | n2-standard-16 | `results_B.json` |
| C | `w2hyb-c4`, `w2hyb-c5` | n2-highmem-16 / -32 | `results_C_8-10-12.json`, `results_C_16-20-24.json` |
| D | `w2hyb-d2`, `w2hyb-d3` | n2-highmem-8 | `results_D_10.json`, `results_D_16.json` |
| E, F | `w2hyb-e` (cross-checked by `w2hyb-e3`), `w2hyb-f` | n2-highmem-32 | `results_E.json`, `results_E_fine.json` |
| figures | `w2hyb-fig3` | n2-highmem-8 | `figures/fig_hybrid_*.png` |

The first C and D jobs (`w2hyb-c1`/`-c2`/`-c3`, `w2hyb-d`) were stopped: their direct solves with the MMD ordering were too slow. The geometric nested-dissection ordering now in `solve_frequency` replaced it, and it agrees with the pivoted MMD solve to 9e-9 on all 90 cases of part E.

## Headline numbers

* **Discrete exactness** (tests): de Rham sequence $10^{-15}$, interface flux bookkeeping $10^{-15}$, energy drift $<10^{-15}$ (200 steps) and $9\times10^{-15}$ (20 000 steps), dot-product test $10^{-16}$, $\lambda_\max\le\max(\lambda_\text{cart},\lambda_\text{tet})$. Every check has a mutation that fails it.
* **Cavity.** Second order (observed 1.91–2.09). The hybrid's error is within 7 % of the pure-Cartesian error at the same $h$.
* **PEC sphere, ka = 2, far zone.** Hybrid 6.6 % at 67k unknowns against 20 % for the staircase at 502k. At about equal unknowns the error is 12× lower (2.4 % against 29 %). All-tetrahedral meshes need 1.4–1.5× more unknowns than the hybrid at equal error.
* **Open.** The TE₁₀ interface reflection flattens at about $10^{-4}$ (Kuhn) and $2$–$3\times10^{-4}$ (Delaunay) beyond 20 cells per guide width.
* **Open.** Time-domain work: $\Delta t$ is 5.8–6.1× smaller than the Cartesian step. Each step costs about 200× ($N=10$) to 580× ($N=16$) more per unknown, because the Chebyshev degree of the consistent-mass solve grows from 204 to 502 with sliver conditioning.

Note: `bin/batch-run --fetch JOB WT` unpacks the whole study directory, scripts included, as shipped at submission time. Fetch into a scratch directory and copy only `results_*.json` and `figures/*.png`.
