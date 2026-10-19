# High-Order DGTD Benchmark Study: B2 Plasmonic Cylinders and B3 Dimer Gaps (`studies/highorder-bench`)

## Purpose

This study evaluates the 2D nodal Discontinuous Galerkin Time-Domain (`wakis.highorder.dgtd`) solver on two dispersive plasmonic benchmarks from `wakis.benchmarks`:

- **B2 (`B2-Au-TE`, `B2-Au-TM`, `B2-Ag-TE`, `B2-Ag-TM`)**: Localized surface plasmon resonance (LSPR) extinction efficiency $Q_{\mathrm{ext}}(\lambda)$ across $\lambda \in [300, 800]\,\mathrm{nm}$ for an $a = 20\,\mathrm{nm}$ gold or silver cylinder in vacuum ($80\,\mathrm{fs}$ window), compared against the exact analytical Mie series with the Rakić et al. (1998) 6-pole Drude–Lorentz permittivity.
- **B3 (`B3-g10`, `B3-g5`, `B3-g2`)**: TE-polarized gold nanowire dimer ($a = 20\,\mathrm{nm}$ cylinders separated by surface-to-surface gaps $g \in \{10, 5, 2\}\,\mathrm{nm}$, $100\,\mathrm{fs}$ window), measuring both far-field extinction NRMS error (`q_ext_nrms`) and midpoint gap-enhancement peak relative error (`enh_peak`) against a converged boundary-element reference.

## Tested Arms

All DGTD runs (`ho_bench.py`) use Hesthaven–Warburton nodal triangles with five-stage fourth-order Carpenter–Kennedy LSRK4 time integration, exact-exponential auxiliary differential equations (ADEs) for the 6-pole Rakić Drude–Lorentz dispersion inside the metal, and a rectangular unsplit Complex-Frequency-Shifted Perfectly Matched Layer (CFS-PML) outer termination:

- `dg1-pml`, `dg2-pml`, `dg3-pml`, `dg4-pml` (published to the shared benchmark store as `dgtd-p1` … `dgtd-p4`): polynomial orders $p \in \{1, 2, 3, 4\}$ with **curved isoparametric elements** (`curved=True`, Szabó–Babuška transfinite blending) snapped to the exact cylinder boundary $r = a$.
- `dg4-pml-straight` (published as `dgtd-p4-straight`): polynomial order $p = 4$ on the identical mesh topology with **straight-sided** affine triangles (`curved=False`), isolating the geometric boundary-approximation error on plasmonic interfaces.
- **Wakis FIT baselines** (loaded from `baseline_store/`): second-order Finite Integration Technique arms `tri-uniform`, `tri-graded`, `tri-graded-pml`, `cart-conformal-all-graded`, `hybrid-quadtri`, and `subgrid-2to1`.

## Convergence Figures

![B2 error vs unknowns](figures/B2_error_vs_unknowns.png)

![B3 error vs unknowns](figures/B3_error_vs_unknowns.png)

## Cost-to-Target Tables

### B2: Cost to Target (`data/B2/cost_to_target.md`)

Log-log interpolated cost from which the error stays at or below target (`≤` = coarsest run already meets target; `—` = target not reached).

| case | metric | target | arm | unknowns | work (unknowns × steps) | run [s] |
|---|---|---|---|---|---|---|
| B2-Ag-TE | q_ext_nrms | 1 % | cart-conformal-all-graded | 42,258 | 1.77e+09 | 78 |
| B2-Ag-TE | q_ext_nrms | 1 % | dg1-pml | — | — | — |
| B2-Ag-TE | q_ext_nrms | 1 % | dg2-pml | ≤ 5,328 | ≤ 8.84e+07 | ≤ 52 |
| B2-Ag-TE | q_ext_nrms | 1 % | dg3-pml | ≤ 8,880 | ≤ 2.30e+08 | ≤ 98 |
| B2-Ag-TE | q_ext_nrms | 1 % | dg4-pml | ≤ 13,320 | ≤ 4.94e+08 | ≤ 161 |
| B2-Ag-TE | q_ext_nrms | 1 % | dg4-pml-straight | 21,586 | 1.09e+09 | 372 |
| B2-Ag-TE | q_ext_nrms | 1 % | hybrid-quadtri | 5,335 | 5.47e+07 | 4 |
| B2-Ag-TE | q_ext_nrms | 1 % | subgrid-2to1 | 20,898 | 1.21e+09 | 188 |
| B2-Ag-TE | q_ext_nrms | 1 % | tri-graded | 4,796 | 6.47e+07 | 6 |
| B2-Ag-TE | q_ext_nrms | 1 % | tri-graded-pml | 4,796 | 6.47e+07 | 6 |
| B2-Ag-TE | q_ext_nrms | 1 % | tri-uniform | ≤ 13,195 | ≤ 1.67e+08 | ≤ 7 |
| B2-Ag-TE | q_ext_nrms | 0.1 % | cart-conformal-all-graded | — | — | — |
| B2-Ag-TE | q_ext_nrms | 0.1 % | dg1-pml | — | — | — |
| B2-Ag-TE | q_ext_nrms | 0.1 % | dg2-pml | 5,901 | 1.05e+08 | 72 |
| B2-Ag-TE | q_ext_nrms | 0.1 % | dg3-pml | ≤ 8,880 | ≤ 2.30e+08 | ≤ 98 |
| B2-Ag-TE | q_ext_nrms | 0.1 % | dg4-pml | ≤ 13,320 | ≤ 4.94e+08 | ≤ 161 |
| B2-Ag-TE | q_ext_nrms | 0.1 % | dg4-pml-straight | — | — | — |
| B2-Ag-TE | q_ext_nrms | 0.1 % | hybrid-quadtri | — | — | — |
| B2-Ag-TE | q_ext_nrms | 0.1 % | subgrid-2to1 | — | — | — |
| B2-Ag-TE | q_ext_nrms | 0.1 % | tri-graded | — | — | — |
| B2-Ag-TE | q_ext_nrms | 0.1 % | tri-graded-pml | — | — | — |
| B2-Ag-TE | q_ext_nrms | 0.1 % | tri-uniform | 82,680 | 2.33e+09 | 95 |
| B2-Ag-TM | q_ext_nrms | 1 % | cart-conformal-all-graded | ≤ 8,112 | ≤ 7.26e+07 | ≤ 8 |
| B2-Ag-TM | q_ext_nrms | 1 % | dg2-pml | ≤ 6,408 | ≤ 1.21e+08 | ≤ 91 |
| B2-Ag-TM | q_ext_nrms | 1 % | dg3-pml | ≤ 10,680 | ≤ 3.11e+08 | ≤ 124 |
| B2-Ag-TM | q_ext_nrms | 1 % | dg4-pml | ≤ 16,020 | ≤ 6.65e+08 | ≤ 211 |
| B2-Ag-TM | q_ext_nrms | 1 % | dg4-pml-straight | — | — | — |
| B2-Ag-TM | q_ext_nrms | 1 % | hybrid-quadtri | ≤ 4,653 | ≤ 3.98e+07 | ≤ 2 |
| B2-Ag-TM | q_ext_nrms | 1 % | subgrid-2to1 | ≤ 5,881 | ≤ 7.33e+07 | ≤ 3 |
| B2-Ag-TM | q_ext_nrms | 1 % | tri-graded | 5,340 | 7.55e+07 | 5 |
| B2-Ag-TM | q_ext_nrms | 1 % | tri-graded-pml | 5,340 | 7.55e+07 | 5 |
| B2-Ag-TM | q_ext_nrms | 1 % | tri-uniform | ≤ 13,053 | ≤ 1.65e+08 | ≤ 6 |
| B2-Ag-TM | q_ext_nrms | 0.1 % | cart-conformal-all-graded | 16,395 | 3.41e+08 | 16 |
| B2-Ag-TM | q_ext_nrms | 0.1 % | dg2-pml | ≤ 6,408 | ≤ 1.21e+08 | ≤ 91 |
| B2-Ag-TM | q_ext_nrms | 0.1 % | dg3-pml | ≤ 10,680 | ≤ 3.11e+08 | ≤ 124 |
| B2-Ag-TM | q_ext_nrms | 0.1 % | dg4-pml | ≤ 16,020 | ≤ 6.65e+08 | ≤ 211 |
| B2-Ag-TM | q_ext_nrms | 0.1 % | dg4-pml-straight | — | — | — |
| B2-Ag-TM | q_ext_nrms | 0.1 % | hybrid-quadtri | 13,088 | 3.69e+08 | 10 |
| B2-Ag-TM | q_ext_nrms | 0.1 % | subgrid-2to1 | 25,899 | 1.79e+09 | 55 |
| B2-Ag-TM | q_ext_nrms | 0.1 % | tri-graded | 10,872 | 2.53e+08 | 11 |
| B2-Ag-TM | q_ext_nrms | 0.1 % | tri-graded-pml | 10,872 | 2.53e+08 | 12 |
| B2-Ag-TM | q_ext_nrms | 0.1 % | tri-uniform | 15,464 | 2.01e+08 | 6 |
| B2-Au-TE | q_ext_nrms | 1 % | cart-conformal-all-graded | 45,783 | 2.02e+09 | 86 |
| B2-Au-TE | q_ext_nrms | 1 % | dg1-pml | — | — | — |
| B2-Au-TE | q_ext_nrms | 1 % | dg2-pml | ≤ 5,328 | ≤ 8.00e+07 | ≤ 47 |
| B2-Au-TE | q_ext_nrms | 1 % | dg3-pml | ≤ 8,880 | ≤ 1.99e+08 | ≤ 110 |
| B2-Au-TE | q_ext_nrms | 1 % | dg4-pml | ≤ 13,320 | ≤ 4.34e+08 | ≤ 143 |
| B2-Au-TE | q_ext_nrms | 1 % | dg4-pml-straight | 23,462 | 1.27e+09 | 438 |
| B2-Au-TE | q_ext_nrms | 1 % | hybrid-quadtri | 5,607 | 4.83e+07 | 3 |
| B2-Au-TE | q_ext_nrms | 1 % | subgrid-2to1 | 24,852 | 1.56e+09 | 162 |
| B2-Au-TE | q_ext_nrms | 1 % | tri-graded | 4,845 | 6.47e+07 | 6 |
| B2-Au-TE | q_ext_nrms | 1 % | tri-graded-pml | 4,845 | 6.47e+07 | 5 |
| B2-Au-TE | q_ext_nrms | 1 % | tri-uniform | ≤ 13,195 | ≤ 1.67e+08 | ≤ 6 |
| B2-Au-TE | q_ext_nrms | 0.1 % | cart-conformal-all-graded | — | — | — |
| B2-Au-TE | q_ext_nrms | 0.1 % | dg1-pml | — | — | — |
| B2-Au-TE | q_ext_nrms | 0.1 % | dg2-pml | 5,959 | 9.26e+07 | 75 |
| B2-Au-TE | q_ext_nrms | 0.1 % | dg3-pml | ≤ 8,880 | ≤ 1.99e+08 | ≤ 110 |
| B2-Au-TE | q_ext_nrms | 0.1 % | dg4-pml | ≤ 13,320 | ≤ 4.34e+08 | ≤ 143 |
| B2-Au-TE | q_ext_nrms | 0.1 % | dg4-pml-straight | — | — | — |
| B2-Au-TE | q_ext_nrms | 0.1 % | hybrid-quadtri | — | — | — |
| B2-Au-TE | q_ext_nrms | 0.1 % | subgrid-2to1 | — | — | — |
| B2-Au-TE | q_ext_nrms | 0.1 % | tri-graded | — | — | — |
| B2-Au-TE | q_ext_nrms | 0.1 % | tri-graded-pml | — | — | — |
| B2-Au-TE | q_ext_nrms | 0.1 % | tri-uniform | 87,971 | 2.58e+09 | 64 |
| B2-Au-TM | q_ext_nrms | 1 % | cart-conformal-all-graded | ≤ 8,112 | ≤ 7.26e+07 | ≤ 5 |
| B2-Au-TM | q_ext_nrms | 1 % | dg2-pml | ≤ 6,300 | ≤ 1.10e+08 | ≤ 60 |
| B2-Au-TM | q_ext_nrms | 1 % | dg3-pml | ≤ 10,500 | ≤ 2.82e+08 | ≤ 110 |
| B2-Au-TM | q_ext_nrms | 1 % | dg4-pml | ≤ 15,750 | ≤ 6.02e+08 | ≤ 193 |
| B2-Au-TM | q_ext_nrms | 1 % | dg4-pml-straight | 49,239 | 4.42e+09 | 1621 |
| B2-Au-TM | q_ext_nrms | 1 % | hybrid-quadtri | 8,848 | 1.67e+08 | 6 |
| B2-Au-TM | q_ext_nrms | 1 % | subgrid-2to1 | ≤ 5,957 | ≤ 5.24e+07 | ≤ 3 |
| B2-Au-TM | q_ext_nrms | 1 % | tri-graded | 6,905 | 1.14e+08 | 8 |
| B2-Au-TM | q_ext_nrms | 1 % | tri-graded-pml | 6,905 | 1.14e+08 | 7 |
| B2-Au-TM | q_ext_nrms | 1 % | tri-uniform | ≤ 13,053 | ≤ 1.65e+08 | ≤ 9 |
| B2-Au-TM | q_ext_nrms | 0.1 % | cart-conformal-all-graded | 20,944 | 5.39e+08 | 29 |
| B2-Au-TM | q_ext_nrms | 0.1 % | dg2-pml | ≤ 6,300 | ≤ 1.10e+08 | ≤ 60 |
| B2-Au-TM | q_ext_nrms | 0.1 % | dg3-pml | ≤ 10,500 | ≤ 2.82e+08 | ≤ 110 |
| B2-Au-TM | q_ext_nrms | 0.1 % | dg4-pml | ≤ 15,750 | ≤ 6.02e+08 | ≤ 193 |
| B2-Au-TM | q_ext_nrms | 0.1 % | dg4-pml-straight | — | — | — |
| B2-Au-TM | q_ext_nrms | 0.1 % | hybrid-quadtri | 15,594 | 4.71e+08 | 13 |
| B2-Au-TM | q_ext_nrms | 0.1 % | subgrid-2to1 | — | — | — |
| B2-Au-TM | q_ext_nrms | 0.1 % | tri-graded | 27,361 | 1.20e+09 | 38 |
| B2-Au-TM | q_ext_nrms | 0.1 % | tri-graded-pml | 27,361 | 1.20e+09 | 37 |
| B2-Au-TM | q_ext_nrms | 0.1 % | tri-uniform | 21,029 | 2.89e+08 | 11 |

### B3: Cost to Target (`data/B3/cost_to_target.md`)

Log-log interpolated cost from which the error stays at or below target (`≤` = coarsest run already meets target; `—` = target not reached).

| case | metric | target | arm | unknowns | work (unknowns × steps) | run [s] |
|---|---|---|---|---|---|---|
| B3-g10 | enh_peak | 5 % | cart-conformal-all-graded | 140,869 | 1.15e+10 | 659 |
| B3-g10 | enh_peak | 5 % | dg2-pml | ≤ 8,136 | ≤ 1.82e+08 | ≤ 95 |
| B3-g10 | enh_peak | 5 % | dg3-pml | ≤ 13,560 | ≤ 4.60e+08 | ≤ 197 |
| B3-g10 | enh_peak | 5 % | dg4-pml | ≤ 20,340 | ≤ 9.45e+08 | ≤ 312 |
| B3-g10 | enh_peak | 5 % | dg4-pml-straight | 44,571 | 3.64e+09 | 956 |
| B3-g10 | enh_peak | 5 % | hybrid-quadtri | 8,837 | 1.43e+08 | 15 |
| B3-g10 | enh_peak | 5 % | subgrid-2to1 | — | — | — |
| B3-g10 | enh_peak | 5 % | tri-graded | ≤ 6,048 | ≤ 9.54e+07 | ≤ 14 |
| B3-g10 | enh_peak | 5 % | tri-graded-pml | 9,095 | 1.70e+08 | 16 |
| B3-g10 | enh_peak | 5 % | tri-uniform | ≤ 18,024 | ≤ 2.84e+08 | ≤ 16 |
| B3-g10 | enh_peak | 2 % | cart-conformal-all-graded | — | — | — |
| B3-g10 | enh_peak | 2 % | dg2-pml | ≤ 8,136 | ≤ 1.82e+08 | ≤ 95 |
| B3-g10 | enh_peak | 2 % | dg3-pml | ≤ 13,560 | ≤ 4.60e+08 | ≤ 197 |
| B3-g10 | enh_peak | 2 % | dg4-pml | ≤ 20,340 | ≤ 9.45e+08 | ≤ 312 |
| B3-g10 | enh_peak | 2 % | dg4-pml-straight | — | — | — |
| B3-g10 | enh_peak | 2 % | hybrid-quadtri | 53,241 | 3.40e+09 | 186 |
| B3-g10 | enh_peak | 2 % | subgrid-2to1 | — | — | — |
| B3-g10 | enh_peak | 2 % | tri-graded | 39,938 | 2.06e+09 | 132 |
| B3-g10 | enh_peak | 2 % | tri-graded-pml | 44,492 | 2.43e+09 | 144 |
| B3-g10 | enh_peak | 2 % | tri-uniform | 31,679 | 5.52e+08 | 28 |
| B3-g10 | q_ext_nrms | 1 % | cart-conformal-all-graded | 63,478 | 3.19e+09 | 122 |
| B3-g10 | q_ext_nrms | 1 % | dg2-pml | ≤ 8,136 | ≤ 1.82e+08 | ≤ 95 |
| B3-g10 | q_ext_nrms | 1 % | dg3-pml | ≤ 13,560 | ≤ 4.60e+08 | ≤ 197 |
| B3-g10 | q_ext_nrms | 1 % | dg4-pml | ≤ 20,340 | ≤ 9.45e+08 | ≤ 312 |
| B3-g10 | q_ext_nrms | 1 % | dg4-pml-straight | 55,448 | 5.30e+09 | 1462 |
| B3-g10 | q_ext_nrms | 1 % | hybrid-quadtri | 9,007 | 1.49e+08 | 16 |
| B3-g10 | q_ext_nrms | 1 % | subgrid-2to1 | 50,454 | 4.48e+09 | 695 |
| B3-g10 | q_ext_nrms | 1 % | tri-graded | 6,945 | 1.15e+08 | 12 |
| B3-g10 | q_ext_nrms | 1 % | tri-graded-pml | 6,945 | 1.15e+08 | 11 |
| B3-g10 | q_ext_nrms | 1 % | tri-uniform | 21,998 | 3.59e+08 | 20 |
| B3-g2 | enh_peak | 5 % | cart-conformal-all-graded | 142,601 | 1.20e+10 | 643 |
| B3-g2 | enh_peak | 5 % | dg2-pml | ≤ 8,028 | ≤ 4.74e+08 | ≤ 254 |
| B3-g2 | enh_peak | 5 % | dg3-pml | ≤ 13,380 | ≤ 1.18e+09 | ≤ 551 |
| B3-g2 | enh_peak | 5 % | dg4-pml | ≤ 20,070 | ≤ 2.51e+09 | ≤ 1094 |
| B3-g2 | enh_peak | 5 % | dg4-pml-straight | ≤ 20,070 | ≤ 2.54e+09 | ≤ 1118 |
| B3-g2 | enh_peak | 5 % | hybrid-quadtri | 13,238 | 7.55e+08 | 52 |
| B3-g2 | enh_peak | 5 % | subgrid-2to1 | — | — | — |
| B3-g2 | enh_peak | 5 % | tri-graded | ≤ 34,756 | ≤ 1.69e+09 | ≤ 97 |
| B3-g2 | enh_peak | 5 % | tri-graded-pml | ≤ 34,780 | ≤ 1.69e+09 | ≤ 85 |
| B3-g2 | enh_peak | 5 % | tri-uniform | ≤ 203,709 | ≤ 9.88e+09 | ≤ 213 |
| B3-g2 | enh_peak | 2 % | cart-conformal-all-graded | — | — | — |
| B3-g2 | enh_peak | 2 % | dg2-pml | ≤ 8,028 | ≤ 4.74e+08 | ≤ 254 |
| B3-g2 | enh_peak | 2 % | dg3-pml | ≤ 13,380 | ≤ 1.18e+09 | ≤ 551 |
| B3-g2 | enh_peak | 2 % | dg4-pml | ≤ 20,070 | ≤ 2.51e+09 | ≤ 1094 |
| B3-g2 | enh_peak | 2 % | dg4-pml-straight | — | — | — |
| B3-g2 | enh_peak | 2 % | hybrid-quadtri | — | — | — |
| B3-g2 | enh_peak | 2 % | subgrid-2to1 | — | — | — |
| B3-g2 | enh_peak | 2 % | tri-graded | ≤ 34,756 | ≤ 1.69e+09 | ≤ 97 |
| B3-g2 | enh_peak | 2 % | tri-graded-pml | — | — | — |
| B3-g2 | enh_peak | 2 % | tri-uniform | ≤ 203,709 | ≤ 9.88e+09 | ≤ 213 |
| B3-g2 | q_ext_nrms | 1 % | cart-conformal-all-graded | — | — | — |
| B3-g2 | q_ext_nrms | 1 % | dg2-pml | ≤ 8,028 | ≤ 4.74e+08 | ≤ 254 |
| B3-g2 | q_ext_nrms | 1 % | dg3-pml | ≤ 13,380 | ≤ 1.18e+09 | ≤ 551 |
| B3-g2 | q_ext_nrms | 1 % | dg4-pml | ≤ 20,070 | ≤ 2.51e+09 | ≤ 1094 |
| B3-g2 | q_ext_nrms | 1 % | dg4-pml-straight | — | — | — |
| B3-g2 | q_ext_nrms | 1 % | hybrid-quadtri | 12,009 | 6.86e+08 | 48 |
| B3-g2 | q_ext_nrms | 1 % | subgrid-2to1 | — | — | — |
| B3-g2 | q_ext_nrms | 1 % | tri-graded | ≤ 34,756 | ≤ 1.69e+09 | ≤ 97 |
| B3-g2 | q_ext_nrms | 1 % | tri-graded-pml | ≤ 34,780 | ≤ 1.69e+09 | ≤ 85 |
| B3-g2 | q_ext_nrms | 1 % | tri-uniform | ≤ 203,709 | ≤ 9.88e+09 | ≤ 213 |
| B3-g5 | enh_peak | 5 % | cart-conformal-all-graded | 111,342 | 8.00e+09 | 481 |
| B3-g5 | enh_peak | 5 % | dg2-pml | ≤ 8,496 | ≤ 2.13e+08 | ≤ 112 |
| B3-g5 | enh_peak | 5 % | dg3-pml | ≤ 14,160 | ≤ 5.34e+08 | ≤ 255 |
| B3-g5 | enh_peak | 5 % | dg4-pml | ≤ 21,240 | ≤ 1.14e+09 | ≤ 379 |
| B3-g5 | enh_peak | 5 % | dg4-pml-straight | 66,809 | 7.38e+09 | 2067 |
| B3-g5 | enh_peak | 5 % | hybrid-quadtri | 12,592 | 3.03e+08 | 29 |
| B3-g5 | enh_peak | 5 % | subgrid-2to1 | — | — | — |
| B3-g5 | enh_peak | 5 % | tri-graded | 7,230 | 1.22e+08 | 11 |
| B3-g5 | enh_peak | 5 % | tri-graded-pml | 20,237 | 6.71e+08 | 34 |
| B3-g5 | enh_peak | 5 % | tri-uniform | 27,456 | 4.65e+08 | 17 |
| B3-g5 | enh_peak | 2 % | cart-conformal-all-graded | — | — | — |
| B3-g5 | enh_peak | 2 % | dg2-pml | ≤ 8,496 | ≤ 2.13e+08 | ≤ 112 |
| B3-g5 | enh_peak | 2 % | dg3-pml | ≤ 14,160 | ≤ 5.34e+08 | ≤ 255 |
| B3-g5 | enh_peak | 2 % | dg4-pml | ≤ 21,240 | ≤ 1.14e+09 | ≤ 379 |
| B3-g5 | enh_peak | 2 % | dg4-pml-straight | — | — | — |
| B3-g5 | enh_peak | 2 % | hybrid-quadtri | 17,126 | 5.05e+08 | 45 |
| B3-g5 | enh_peak | 2 % | subgrid-2to1 | — | — | — |
| B3-g5 | enh_peak | 2 % | tri-graded | 8,981 | 1.68e+08 | 15 |
| B3-g5 | enh_peak | 2 % | tri-graded-pml | 26,607 | 1.07e+09 | 59 |
| B3-g5 | enh_peak | 2 % | tri-uniform | 36,337 | 6.71e+08 | 22 |
| B3-g5 | q_ext_nrms | 1 % | cart-conformal-all-graded | 109,242 | 7.76e+09 | 468 |
| B3-g5 | q_ext_nrms | 1 % | dg2-pml | ≤ 8,496 | ≤ 2.13e+08 | ≤ 112 |
| B3-g5 | q_ext_nrms | 1 % | dg3-pml | ≤ 14,160 | ≤ 5.34e+08 | ≤ 255 |
| B3-g5 | q_ext_nrms | 1 % | dg4-pml | ≤ 21,240 | ≤ 1.14e+09 | ≤ 379 |
| B3-g5 | q_ext_nrms | 1 % | dg4-pml-straight | — | — | — |
| B3-g5 | q_ext_nrms | 1 % | hybrid-quadtri | ≤ 6,969 | ≤ 1.58e+08 | ≤ 12 |
| B3-g5 | q_ext_nrms | 1 % | subgrid-2to1 | — | — | — |
| B3-g5 | q_ext_nrms | 1 % | tri-graded | 10,502 | 2.15e+08 | 19 |
| B3-g5 | q_ext_nrms | 1 % | tri-graded-pml | 9,589 | 2.38e+08 | 26 |
| B3-g5 | q_ext_nrms | 1 % | tri-uniform | 34,737 | 6.29e+08 | 20 |

## Observations

1. **Curved isoparametric vs straight-sided elements at $p = 4$**:
   - At equal mesh spacing $h$ and identical unknown count, straight-sided $p = 4$ (`dg4-pml-straight`) is limited by polygonal boundary approximation on the plasmonic cylinder, whereas curved $p = 4$ (`dg4-pml`) reduces error by two to three orders of magnitude.
   - On B2 at $h = 10\,\mathrm{nm}$ ($15,750$–$16,020$ unknowns), `dg4-pml-straight` gives `q_ext_nrms` errors of $1.416\,\%$ (`B2-Ag-TE`), $2.533\,\%$ (`B2-Ag-TM`), $1.627\,\%$ (`B2-Au-TE`), and $3.898\,\%$ (`B2-Au-TM`), while curved `dg4-pml` at the same $h = 10\,\mathrm{nm}$ reaches $\le 0.001\,\%$ across all four cases (and already reaches $0.002$–$0.003\,\%$ on TE at $h = 13.33\,\mathrm{nm}$ with $13,320$ unknowns).
   - On B3 at $h = 6.67\,\mathrm{nm}$ ($40,500$–$44,100$ unknowns), `dg4-pml-straight` has `q_ext_nrms` / `enh_peak` errors of $1.349\,\% / 5.455\,\%$ (`B3-g10`), $2.113\,\% / 12.033\,\%$ (`B3-g5`), and $1.496\,\% / 4.624\,\%$ (`B3-g2`), compared with $0.010\,\% / 0.012\,\%$, $0.021\,\% / 0.070\,\%$, and $0.008\,\% / 0.002\,\%$ for curved `dg4-pml`.
2. **Cost to $0.1\,\%$ $Q_{\mathrm{ext}}$ NRMS on B2**:
   - For TE polarization (`B2-Ag-TE`, `B2-Au-TE`), none of the graded FIT or conformal Cartesian baselines reach $0.1\,\%$ down to $h = 0.50\,\mathrm{nm}$; only `tri-uniform` reaches $0.1\,\%$, requiring $82,680$ unknowns ($2.33 \times 10^9$ unknown-steps, $95\,\mathrm{s}$) on `B2-Ag-TE` and $87,971$ unknowns ($2.58 \times 10^9$ unknown-steps, $64\,\mathrm{s}$) on `B2-Au-TE`. Curved `dg2-pml` reaches $0.1\,\%$ with $5,901$ unknowns ($1.05 \times 10^8$ unknown-steps, $72\,\mathrm{s}$) on `B2-Ag-TE` (**$14.0\times$ fewer unknowns**, **$22.2\times$ fewer unknown-steps**) and $5,959$ unknowns ($9.26 \times 10^7$ unknown-steps, $75\,\mathrm{s}$) on `B2-Au-TE` (**$14.8\times$ fewer unknowns**, **$27.9\times$ fewer unknown-steps**). At the coarsest $h = 13.33\,\mathrm{nm}$ mesh, curved `dg3-pml` ($\le 8,880$ unknowns) is already at $0.021\,\%$ and `dg4-pml` ($\le 13,320$ unknowns) is at $0.002$–$0.003\,\%$.
   - For TM polarization (`B2-Ag-TM`, `B2-Au-TM`), where the electric field is axial and continuous across the metal–vacuum interface, the lowest-unknowns FIT baselines at $0.1\,\%$ are `tri-graded` ($10,872$ unknowns on `B2-Ag-TM`) and `hybrid-quadtri` ($15,594$ unknowns on `B2-Au-TM`), while `dg2-pml` already meets $0.1\,\%$ at its coarsest $h = 10\,\mathrm{nm}$ mesh ($\le 6,408$ unknowns at $0.018\,\%$ on `B2-Ag-TM`, $\le 6,300$ unknowns at $0.012\,\%$ on `B2-Au-TM`, i.e., **$\ge 1.7\times$** and **$\ge 2.5\times$ fewer unknowns**).
3. **Cost to target on B3 dimer gaps**:
   - All curved DGTD arms (`dg2-pml`, `dg3-pml`, `dg4-pml`) already meet the $5\,\%$ and $2\,\%$ `enh_peak` targets and the $1\,\%$ `q_ext_nrms` target at the coarsest tested resolution $h = 10\,\mathrm{nm}$ ($\le 8,028$–$8,496$ unknowns for $p = 2$, $\le 13,380$–$14,160$ for $p = 3$, and $\le 20,070$–$21,240$ for $p = 4$).
   - At the $2\,\%$ `enh_peak` target, `dg2-pml` requires $\le 8,136$ unknowns vs $31,679$ (`tri-uniform`) or $39,938$ (`tri-graded`) on `B3-g10` (**$\ge 3.9\times$ fewer unknowns**), $\le 8,496$ vs $8,981$ (`tri-graded`) or $17,126$ (`hybrid-quadtri`) on `B3-g5`, and $\le 8,028$ vs $\le 34,756$ (`tri-graded`) on `B3-g2` (**$\ge 4.3\times$ fewer unknowns**, **$\ge 3.6\times$ fewer unknown-steps**).

## Limitations and Notes

- **Curved-element geometric limit at $h = a$ ($20\,\mathrm{nm}$)**: The isoparametric curved-triangle mapping assumes at most one curved boundary face per triangle and explicitly rejects elements with two curved faces (`RuntimeError("element with two curved faces")`). On an $a = 20\,\mathrm{nm}$ cylinder, a coarse $h = 20\,\mathrm{nm}$ mesh produces triangles spanning two segments of the cylinder boundary and fails by design (the 8 $h = 20\,\mathrm{nm}$ B2c records are skipped during publication); the coarsest valid resolution is $h = 13.33\,\mathrm{nm}$ ($a / h = 1.5$).
- **Batch timeouts on finest B3 $p = 4$ meshes**: Three of the 60 B3 tasks timed out on Google Cloud Batch at $h = 2.50\,\mathrm{nm}$: `B3-g2_dg4-pml_2.500nm`, `B3-g5_dg4-pml_2.500nm`, and `B3-g5_dg4-pml-straight_2.500nm`.
- **Global time step constrained by the gap element**: Because time integration uses an explicit global LSRK4 step, the smallest triangle inside the dimer gap sets $\Delta t$ for the entire mesh. At the same outer resolution $h = 10\,\mathrm{nm}$, reducing the gap from $g = 10\,\mathrm{nm}$ (`B3-g10`) to $g = 2\,\mathrm{nm}$ (`B3-g2`) increases the required step count from $22,330 \to 58,984$ ($p = 2$), $33,922 \to 88,339$ ($p = 3$), and $46,479 \to 125,011$ ($p = 4$), a $\approx 2.6$–$2.7\times$ increase in steps at nearly identical unknown count ($\sim 8.0\text{k}$–$20.3\text{k}$).
- **PML late-time growth**: As characterized in `studies/highorder-mie`, the unsplit rectangular CFS-PML exhibits a late-time discrete instability on fine meshes at $p \ge 3$ after a large number of time steps. In these $80$–$100\,\mathrm{fs}$ B2/B3 windows, all $p = 2$ runs down to $h = 2.50\,\mathrm{nm}$ and $p = 3, 4$ runs at $h \ge 5.0$–$6.67\,\mathrm{nm}$ remain stable (reaching $0.0002$–$0.003\,\%$ on B2 and $0.005$–$0.077\,\%$ on B3), whereas the finest $p = 3, 4$ runs ($h \le 3.33\,\mathrm{nm}$, and $h = 5.00\,\mathrm{nm}$ on `B2-Ag-*` and `B3-g2`, where step counts exceed $\sim 8.6 \times 10^4$) blow up late in the window and are excluded from the plots and tables (listed at the bottom of `data/B2/summary.md` and `data/B3/summary.md`).
- **Hardware and commit provenance**: Timings were measured on single-core `n2-standard-2` Batch VMs and carry $\pm 30\,\%$ host-to-host wall-clock variation. In addition, the main B2 records ($h \in [2.5, 10]\,\mathrm{nm}$) were produced with pre-optimization commit `4bb0c4c` ($\approx 1.6\times$ slower per unknown-step than the final vectorized DG kernel), whereas the coarse B2c records ($h = 13.33\,\mathrm{nm}$, commit `bdd0676`) and all B3 records (commit `f2419a7`) used the final optimized code.
