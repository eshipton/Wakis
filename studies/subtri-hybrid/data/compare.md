# Equal-Work Benchmark Comparison (`subtri-hybrid`)

Cells report `unknowns / work / time` interpolated log-log to each target (`≤` = upper bound at coarsest rung, `—` = target not reached).

## B1-TE

| target | subtri-hybrid-r2 | subtri-hybrid-r4 | tri-graded-r2 | tri-graded-r4 | cart-conformal-all-graded-r2 | cart-conformal-all-graded-r4 | best baseline (work) | ours / best (work) | best baseline (time): ours / best |
|---|---|---|---|---|---|---|---|---|---|
| q_sca_rms_rel ≤ 2 % | 121k / 5.2e9 / 79 s | 147k / 9.6e9 / 114 s | 304k / 3.0e10 / 764 s | 265k / 3.4e10 / 904 s | 246k / 2.3e10 / 506 s | 253k / 3.1e10 / 826 s | cart-conformal-all-graded-r2 (2.3e10) | 0.22 | cart-conformal-all-graded-r2 (506 s): 0.16 |
| q_sca_rms_rel ≤ 1 % | 213k / 1.2e10 / 209 s | 263k / 2.3e10 / 366 s | 506k / 6.7e10 / 1647 s | — | 452k / 5.9e10 / 1384 s | — | cart-conformal-all-graded-r2 (5.9e10) | 0.20 | cart-conformal-all-graded-r2 (1384 s): 0.15 |

## B1-TM

| target | subtri-hybrid-r2 | subtri-hybrid-r4 | tri-graded-r2 | tri-graded-r4 | cart-conformal-all-graded-r2 | cart-conformal-all-graded-r4 | best baseline (work) | ours / best (work) | best baseline (time): ours / best |
|---|---|---|---|---|---|---|---|---|---|
| q_sca_rms_rel ≤ 2 % | 107k / 4.2e9 / 52 s | 162k / 1.1e10 / 138 s | 276k / 2.5e10 / 495 s | 254k / 3.2e10 / 522 s | 263k / 2.6e10 / 687 s | 294k / 3.9e10 / 904 s | tri-graded-r2 (2.5e10) | 0.16 | tri-graded-r2 (495 s): 0.11 |
| q_sca_rms_rel ≤ 1 % | 197k / 1.1e10 / 148 s | — | 521k / 7.0e10 / 1307 s | — | 511k / 7.2e10 / 1995 s | — | tri-graded-r2 (7.0e10) | 0.15 | tri-graded-r2 (1307 s): 0.11 |

## B2-Au-TE

| target | subtri-hybrid | subtri-hybrid-nolts | hybrid-quadtri | tri-graded | tri-graded-pml | tri-graded-pml-lts2 | cart-conformal-all-graded | nfit-ogrid | dgtd-p2 | dgtd-p3 | subgrid-2to1 | voronoi-poly | best baseline (work) | ours / best (work) | best baseline (time): ours / best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q_ext_nrms ≤ 2 % | ≤7.3k / ≤6.2e7 / ≤2.9 s | ≤7.3k / ≤6.2e7 / ≤3.1 s | ≤5.6k / ≤4.8e7 / ≤3.0 s | ≤4.3k / ≤5.5e7 / ≤5.5 s | ≤4.3k / ≤5.5e7 / ≤4.6 s | ≤4.3k / ≤2.7e7 / ≤2.6 s | 16.5k / 3.5e8 / 22 s | ≤2.4k / ≤4.3e7 / ≤2.8 s | ≤5.3k / ≤8.0e7 / ≤47 s | ≤8.9k / ≤2.0e8 / ≤110 s | 12.5k / 3.5e8 / 20 s | 5.5k / 2.1e8 / 15 s | tri-graded-pml-lts2 (≤2.7e7) | 2.28 | tri-graded-pml-lts2 (3 s): 1.14 |
| q_ext_nrms ≤ 1 % | 7.7k / 6.3e7 / 3.2 s | 7.7k / 7.4e7 / 3.5 s | 5.6k / 4.8e7 / 3.0 s | 4.8k / 6.5e7 / 7.3 s | 4.8k / 6.5e7 / 5.1 s | 4.8k / 3.2e7 / 3.4 s | 45.8k / 2.0e9 / 86 s | 2.9k / 5.7e7 / 3.4 s | ≤5.3k / ≤8.0e7 / ≤47 s | ≤8.9k / ≤2.0e8 / ≤110 s | 24.9k / 1.6e9 / 162 s | 7.7k / 3.8e8 / 22 s | tri-graded-pml-lts2 (3.2e7) | 1.99 | hybrid-quadtri (3 s): 1.08 |
| q_ext_nrms ≤ 0.5 % | 10.7k / 1.0e8 / 3.8 s | 11.4k / 2.3e8 / 11 s | 14.0k / 3.9e8 / 22 s | 7.9k / 1.4e8 / 9.2 s | 7.9k / 1.4e8 / 8.3 s | 7.7k / 6.7e7 / 4.8 s | — | 6.6k / 2.2e8 / 10 s | ≤5.3k / ≤8.0e7 / ≤47 s | ≤8.9k / ≤2.0e8 / ≤110 s | — | — | tri-graded-pml-lts2 (6.7e7) | 1.51 | tri-graded-pml-lts2 (5 s): 0.80 |
| q_ext_nrms ≤ 0.1 % | — | — | — | — | — | — | — | 42.2k / 4.6e9 / 261 s | 6.0k / 9.3e7 / 75 s | ≤8.9k / ≤2.0e8 / ≤110 s | — | — | dgtd-p2 (9.3e7) | — | — |
| q_ext_lam_peak ≤ 0.5 % | ≤7.3k / ≤6.2e7 / ≤2.9 s | ≤7.3k / ≤6.2e7 / ≤3.1 s | ≤5.6k / ≤4.8e7 / ≤3.0 s | ≤4.3k / ≤5.5e7 / ≤5.5 s | ≤4.3k / ≤5.5e7 / ≤4.6 s | ≤4.3k / ≤2.7e7 / ≤2.6 s | 12.0k / 1.8e8 / 12 s | ≤2.4k / ≤4.3e7 / ≤2.8 s | ≤5.3k / ≤8.0e7 / ≤47 s | ≤8.9k / ≤2.0e8 / ≤110 s | 10.5k / 1.4e8 / 5.1 s | 5.5k / 2.2e8 / 15 s | tri-graded-pml-lts2 (≤2.7e7) | 2.28 | tri-graded-pml-lts2 (3 s): 1.14 |
| q_ext_lam_peak ≤ 0.2 % | 8.2k / 6.5e7 / 3.4 s | 8.3k / 9.5e7 / 4.1 s | 6.1k / 7.2e7 / 4.2 s | 5.0k / 6.8e7 / 7.6 s | 5.0k / 6.8e7 / 5.2 s | 4.9k / 3.3e7 / 3.7 s | 67.0k / 5.5e9 / 220 s | 3.4k / 7.0e7 / 4.0 s | ≤5.3k / ≤8.0e7 / ≤47 s | ≤8.9k / ≤2.0e8 / ≤110 s | 12.2k / 3.0e8 / 16 s | 17.4k / 1.4e9 / 104 s | tri-graded-pml-lts2 (3.3e7) | 1.94 | tri-graded-pml-lts2 (4 s): 0.90 |

## B2-Au-TM

| target | subtri-hybrid | subtri-hybrid-nolts | hybrid-quadtri | tri-graded | tri-graded-pml | tri-graded-pml-lts2 | cart-conformal-all-graded | nfit-ogrid | dgtd-p2 | dgtd-p3 | subgrid-2to1 | voronoi-poly | best baseline (work) | ours / best (work) | best baseline (time): ours / best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q_ext_nrms ≤ 2 % | ≤7.1k / ≤6.1e7 / ≤2.0 s | ≤7.1k / ≤6.1e7 / ≤2.1 s | ≤5.4k / ≤4.7e7 / ≤2.0 s | 5.1k / 7.0e7 / 7.3 s | 5.1k / 7.0e7 / 6.7 s | 5.1k / 3.6e7 / 3.5 s | ≤8.1k / ≤7.3e7 / ≤4.9 s | ≤2.4k / ≤4.2e7 / ≤3.3 s | ≤6.3k / ≤1.1e8 / ≤60 s | ≤10.5k / ≤2.8e8 / ≤110 s | ≤6.0k / ≤5.2e7 / ≤2.6 s | 3.4k / 8.7e7 / 5.3 s | tri-graded-pml-lts2 (3.6e7) | 1.70 | hybrid-quadtri (2 s): 1.03 |
| q_ext_nrms ≤ 1 % | ≤7.1k / ≤6.1e7 / ≤2.0 s | ≤7.1k / ≤6.1e7 / ≤2.1 s | 8.8k / 1.7e8 / 5.6 s | 6.9k / 1.1e8 / 9.2 s | 6.9k / 1.1e8 / 6.8 s | 6.8k / 5.6e7 / 3.7 s | ≤8.1k / ≤7.3e7 / ≤4.9 s | ≤2.4k / ≤4.2e7 / ≤3.3 s | ≤6.3k / ≤1.1e8 / ≤60 s | ≤10.5k / ≤2.8e8 / ≤110 s | ≤6.0k / ≤5.2e7 / ≤2.6 s | 4.1k / 1.2e8 / 7.5 s | nfit-ogrid (≤4.2e7) | 1.43 | subgrid-2to1 (3 s): 0.79 |
| q_ext_nrms ≤ 0.5 % | ≤7.1k / ≤6.1e7 / ≤2.0 s | ≤7.1k / ≤6.1e7 / ≤2.1 s | 10.2k / 2.2e8 / 7.0 s | 8.6k / 1.7e8 / 11 s | 8.6k / 1.7e8 / 7.6 s | 8.4k / 8.0e7 / 4.1 s | 12.6k / 2.0e8 / 14 s | ≤2.4k / ≤4.2e7 / ≤3.3 s | ≤6.3k / ≤1.1e8 / ≤60 s | ≤10.5k / ≤2.8e8 / ≤110 s | ≤6.0k / ≤5.2e7 / ≤2.6 s | 4.8k / 1.7e8 / 11 s | nfit-ogrid (≤4.2e7) | 1.43 | subgrid-2to1 (3 s): 0.79 |
| q_ext_nrms ≤ 0.1 % | 9.8k / 1.4e8 / 2.8 s | 14.6k / 3.9e8 / 10.0 s | 15.6k / 4.7e8 / 13 s | 27.4k / 1.2e9 / 38 s | 27.4k / 1.2e9 / 37 s | 26.0k / 5.5e8 / 24 s | 20.9k / 5.4e8 / 29 s | 7.0k / 2.4e8 / 8.8 s | ≤6.3k / ≤1.1e8 / ≤60 s | ≤10.5k / ≤2.8e8 / ≤110 s | — | — | dgtd-p2 (≤1.1e8) | 1.27 | nfit-ogrid (9 s): 0.32 |
| q_ext_lam_peak ≤ 0.5 % | ≤7.1k / ≤6.1e7 / ≤2.0 s | ≤7.1k / ≤6.1e7 / ≤2.1 s | ≤5.4k / ≤4.7e7 / ≤2.0 s | ≤4.3k / ≤5.4e7 / ≤4.9 s | ≤4.3k / ≤5.4e7 / ≤5.9 s | ≤4.3k / ≤2.7e7 / ≤2.7 s | ≤8.1k / ≤7.3e7 / ≤4.9 s | ≤2.4k / ≤4.2e7 / ≤3.3 s | ≤6.3k / ≤1.1e8 / ≤60 s | ≤10.5k / ≤2.8e8 / ≤110 s | ≤6.0k / ≤5.2e7 / ≤2.6 s | ≤2.1k / ≤2.8e7 / ≤3.3 s | tri-graded-pml-lts2 (≤2.7e7) | 2.25 | hybrid-quadtri (2 s): 1.03 |
| q_ext_lam_peak ≤ 0.2 % | ≤7.1k / ≤6.1e7 / ≤2.0 s | ≤7.1k / ≤6.1e7 / ≤2.1 s | ≤5.4k / ≤4.7e7 / ≤2.0 s | ≤4.3k / ≤5.4e7 / ≤4.9 s | ≤4.3k / ≤5.4e7 / ≤5.9 s | ≤4.3k / ≤2.7e7 / ≤2.7 s | ≤8.1k / ≤7.3e7 / ≤4.9 s | ≤2.4k / ≤4.2e7 / ≤3.3 s | ≤6.3k / ≤1.1e8 / ≤60 s | ≤10.5k / ≤2.8e8 / ≤110 s | ≤6.0k / ≤5.2e7 / ≤2.6 s | ≤2.1k / ≤2.8e7 / ≤3.3 s | tri-graded-pml-lts2 (≤2.7e7) | 2.25 | hybrid-quadtri (2 s): 1.03 |

## B2-Ag-TE

| target | subtri-hybrid | subtri-hybrid-nolts | hybrid-quadtri | tri-graded | tri-graded-pml | tri-graded-pml-lts2 | cart-conformal-all-graded | nfit-ogrid | dgtd-p2 | dgtd-p3 | subgrid-2to1 | voronoi-poly | best baseline (work) | ours / best (work) | best baseline (time): ours / best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q_ext_nrms ≤ 2 % | ≤7.0k / ≤5.8e7 / ≤2.7 s | ≤7.0k / ≤6.4e7 / ≤3.0 s | ≤4.8k / ≤4.1e7 / ≤2.8 s | ≤3.9k / ≤4.9e7 / ≤4.6 s | ≤3.9k / ≤4.9e7 / ≤4.3 s | ≤3.9k / ≤2.4e7 / ≤2.5 s | 16.1k / 3.3e8 / 20 s | ≤2.4k / ≤4.3e7 / ≤3.0 s | ≤5.3k / ≤8.8e7 / ≤52 s | ≤8.9k / ≤2.3e8 / ≤98 s | 9.0k / 2.2e8 / 16 s | 5.2k / 2.0e8 / 13 s | tri-graded-pml-lts2 (≤2.4e7) | 2.37 | tri-graded-pml-lts2 (2 s): 1.10 |
| q_ext_nrms ≤ 1 % | 7.6k / 6.9e7 / 2.9 s | 7.6k / 8.4e7 / 4.4 s | 5.3k / 5.5e7 / 4.3 s | 4.8k / 6.5e7 / 6.5 s | 4.8k / 6.5e7 / 6.2 s | 5.0k / 3.6e7 / 3.2 s | 42.3k / 1.8e9 / 78 s | 2.8k / 5.3e7 / 3.4 s | ≤5.3k / ≤8.8e7 / ≤52 s | ≤8.9k / ≤2.3e8 / ≤98 s | 20.9k / 1.2e9 / 188 s | 7.3k / 3.5e8 / 26 s | tri-graded-pml-lts2 (3.6e7) | 1.89 | tri-graded-pml-lts2 (3 s): 0.92 |
| q_ext_nrms ≤ 0.5 % | 10.0k / 1.1e8 / 3.7 s | 10.0k / 2.0e8 / 9.7 s | 12.3k / 3.3e8 / 18 s | 7.5k / 1.3e8 / 11 s | 7.5k / 1.3e8 / 11 s | 7.7k / 6.8e7 / 4.7 s | — | 6.7k / 2.2e8 / 10 s | ≤5.3k / ≤8.8e7 / ≤52 s | ≤8.9k / ≤2.3e8 / ≤98 s | — | 23.5k / 2.3e9 / 117 s | tri-graded-pml-lts2 (6.8e7) | 1.67 | tri-graded-pml-lts2 (5 s): 0.79 |
| q_ext_nrms ≤ 0.1 % | — | — | — | — | — | — | — | 37.9k / 3.8e9 / 160 s | 5.9k / 1.1e8 / 72 s | ≤8.9k / ≤2.3e8 / ≤98 s | — | — | dgtd-p2 (1.1e8) | — | — |
| q_ext_lam_peak ≤ 0.5 % | ≤7.0k / ≤5.8e7 / ≤2.7 s | ≤7.0k / ≤6.4e7 / ≤3.0 s | ≤4.8k / ≤4.1e7 / ≤2.8 s | ≤3.9k / ≤4.9e7 / ≤4.6 s | ≤3.9k / ≤4.9e7 / ≤4.3 s | ≤3.9k / ≤2.4e7 / ≤2.5 s | ≤8.1k / ≤7.3e7 / ≤5.3 s | ≤2.4k / ≤4.3e7 / ≤3.0 s | ≤5.3k / ≤8.8e7 / ≤52 s | ≤8.9k / ≤2.3e8 / ≤98 s | ≤6.2k / ≤6.1e7 / ≤3.0 s | 3.2k / 7.4e7 / 6.6 s | tri-graded-pml-lts2 (≤2.4e7) | 2.37 | tri-graded-pml-lts2 (2 s): 1.10 |
| q_ext_lam_peak ≤ 0.2 % | ≤7.0k / ≤5.8e7 / ≤2.7 s | ≤7.0k / ≤6.4e7 / ≤3.0 s | ≤4.8k / ≤4.1e7 / ≤2.8 s | ≤3.9k / ≤4.9e7 / ≤4.6 s | ≤3.9k / ≤4.9e7 / ≤4.3 s | ≤3.9k / ≤2.4e7 / ≤2.5 s | 12.1k / 1.8e8 / 11 s | ≤2.4k / ≤4.3e7 / ≤3.0 s | ≤5.3k / ≤8.8e7 / ≤52 s | ≤8.9k / ≤2.3e8 / ≤98 s | 6.5k / 1.2e8 / 6.5 s | 6.2k / 2.6e8 / 19 s | tri-graded-pml-lts2 (≤2.4e7) | 2.37 | tri-graded-pml-lts2 (2 s): 1.10 |

## B2-Ag-TM

| target | subtri-hybrid | subtri-hybrid-nolts | hybrid-quadtri | tri-graded | tri-graded-pml | tri-graded-pml-lts2 | cart-conformal-all-graded | nfit-ogrid | dgtd-p2 | dgtd-p3 | subgrid-2to1 | voronoi-poly | best baseline (work) | ours / best (work) | best baseline (time): ours / best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q_ext_nrms ≤ 2 % | ≤6.9k / ≤5.7e7 / ≤1.8 s | ≤6.9k / ≤6.2e7 / ≤2.1 s | ≤4.7k / ≤4.0e7 / ≤1.9 s | 4.1k / 5.3e7 / 4.3 s | 4.1k / 5.3e7 / 4.0 s | 4.2k / 2.8e7 / 2.7 s | ≤8.1k / ≤7.3e7 / ≤7.4 s | ≤2.4k / ≤4.2e7 / ≤3.3 s | ≤6.4k / ≤1.2e8 / ≤91 s | ≤10.7k / ≤3.1e8 / ≤124 s | ≤5.9k / ≤5.3e7 / ≤2.5 s | ≤1.8k / ≤2.3e7 / ≤2.0 s | voronoi-poly (≤2.3e7) | 2.46 | hybrid-quadtri (2 s): 0.97 |
| q_ext_nrms ≤ 1 % | ≤6.9k / ≤5.7e7 / ≤1.8 s | ≤6.9k / ≤6.2e7 / ≤2.1 s | ≤4.7k / ≤4.0e7 / ≤1.9 s | 5.3k / 7.6e7 / 5.1 s | 5.3k / 7.6e7 / 5.2 s | 5.6k / 4.3e7 / 3.7 s | ≤8.1k / ≤7.3e7 / ≤7.4 s | ≤2.4k / ≤4.2e7 / ≤3.3 s | ≤6.4k / ≤1.2e8 / ≤91 s | ≤10.7k / ≤3.1e8 / ≤124 s | ≤5.9k / ≤5.3e7 / ≤2.5 s | 3.5k / 8.7e7 / 7.1 s | hybrid-quadtri (≤4.0e7) | 1.43 | hybrid-quadtri (2 s): 0.97 |
| q_ext_nrms ≤ 0.5 % | 7.0k / 5.9e7 / 1.9 s | 7.0k / 1.0e8 / 3.5 s | 9.2k / 2.0e8 / 8.4 s | 7.6k / 1.3e8 / 9.6 s | 7.6k / 1.3e8 / 8.9 s | 7.5k / 6.6e7 / 5.1 s | ≤8.1k / ≤7.3e7 / ≤7.4 s | ≤2.4k / ≤4.2e7 / ≤3.3 s | ≤6.4k / ≤1.2e8 / ≤91 s | ≤10.7k / ≤3.1e8 / ≤124 s | ≤5.9k / ≤5.3e7 / ≤2.5 s | 4.1k / 1.2e8 / 7.1 s | nfit-ogrid (≤4.2e7) | 1.40 | subgrid-2to1 (2 s): 0.78 |
| q_ext_nrms ≤ 0.1 % | 11.6k / 1.5e8 / 2.6 s | 8.6k / 1.6e8 / 4.8 s | 13.1k / 3.7e8 / 10 s | 10.9k / 2.5e8 / 11 s | 10.9k / 2.5e8 / 12 s | 10.3k / 1.2e8 / 7.7 s | 16.4k / 3.4e8 / 16 s | 6.3k / 2.1e8 / 7.8 s | ≤6.4k / ≤1.2e8 / ≤91 s | ≤10.7k / ≤3.1e8 / ≤124 s | 25.9k / 1.8e9 / 55 s | — | tri-graded-pml-lts2 (1.2e8) | 1.31 | tri-graded-pml-lts2 (8 s): 0.34 |
| q_ext_lam_peak ≤ 0.5 % | ≤6.9k / ≤5.7e7 / ≤1.8 s | ≤6.9k / ≤6.2e7 / ≤2.1 s | ≤4.7k / ≤4.0e7 / ≤1.9 s | ≤3.8k / ≤4.8e7 / ≤4.2 s | ≤3.8k / ≤4.8e7 / ≤3.7 s | ≤3.8k / ≤2.4e7 / ≤2.4 s | ≤8.1k / ≤7.3e7 / ≤7.4 s | ≤2.4k / ≤4.2e7 / ≤3.3 s | ≤6.4k / ≤1.2e8 / ≤91 s | ≤10.7k / ≤3.1e8 / ≤124 s | ≤5.9k / ≤5.3e7 / ≤2.5 s | ≤1.8k / ≤2.3e7 / ≤2.0 s | voronoi-poly (≤2.3e7) | 2.46 | hybrid-quadtri (2 s): 0.97 |
| q_ext_lam_peak ≤ 0.2 % | ≤6.9k / ≤5.7e7 / ≤1.8 s | ≤6.9k / ≤6.2e7 / ≤2.1 s | ≤4.7k / ≤4.0e7 / ≤1.9 s | ≤3.8k / ≤4.8e7 / ≤4.2 s | ≤3.8k / ≤4.8e7 / ≤3.7 s | ≤3.8k / ≤2.4e7 / ≤2.4 s | ≤8.1k / ≤7.3e7 / ≤7.4 s | ≤2.4k / ≤4.2e7 / ≤3.3 s | ≤6.4k / ≤1.2e8 / ≤91 s | ≤10.7k / ≤3.1e8 / ≤124 s | ≤5.9k / ≤5.3e7 / ≤2.5 s | ≤1.8k / ≤2.3e7 / ≤2.0 s | voronoi-poly (≤2.3e7) | 2.46 | hybrid-quadtri (2 s): 0.97 |

## B3-g10

| target | subtri-hybrid | subtri-hybrid-nolts | tri-graded-pml2-patch | hybrid-quadtri | tri-graded | cart-conformal-all-graded | nfit-boxed | dgtd-p3 | subgrid-2to1-gap4-lts8 | best baseline (work) | ours / best (work) | best baseline (time): ours / best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| enh_peak ≤ 5 % | ≤9.2k / ≤8.6e7 / ≤5.6 s | ≤9.2k / ≤1.0e8 / ≤6.2 s | 14.4k / 2.4e8 / 17 s | 8.8k / 1.4e8 / 15 s | 14.4k / 2.4e8 / 13 s | 141k / 1.1e10 / 731 s | 13.7k / 4.6e8 / 19 s | ≤13.6k / ≤4.6e8 / ≤197 s | — | hybrid-quadtri (1.4e8) | 0.60 | tri-graded (13 s): 0.43 |
| enh_peak ≤ 2 % | 10.3k / 1.2e8 / 7.2 s | 10.6k / 1.8e8 / 12 s | 16.5k / 2.9e8 / 19 s | 53.2k / 3.4e9 / 186 s | 16.5k / 2.9e8 / 33 s | — | 46.1k / 3.8e9 / 127 s | ≤13.6k / ≤4.6e8 / ≤197 s | — | tri-graded-pml2-patch (2.9e8) | 0.42 | tri-graded-pml2-patch (19 s): 0.38 |
| enh_peak ≤ 1 % | 13.6k / 1.9e8 / 9.8 s | 14.5k / 3.4e8 / 21 s | 18.1k / 3.4e8 / 21 s | 66.7k / 4.9e9 / 324 s | 18.1k / 3.4e8 / 40 s | — | 65.2k / 6.7e9 / 236 s | ≤13.6k / ≤4.6e8 / ≤197 s | — | tri-graded-pml2-patch (3.4e8) | 0.54 | tri-graded-pml2-patch (21 s): 0.47 |
| q_ext_nrms ≤ 2 % | ≤9.2k / ≤8.6e7 / ≤5.6 s | ≤9.2k / ≤1.0e8 / ≤6.2 s | ≤13.4k / ≤2.1e8 / ≤15 s | ≤6.8k / ≤7.6e7 / ≤5.6 s | ≤13.4k / ≤2.1e8 / ≤12 s | 18.2k / 3.5e8 / 29 s | 13.2k / 4.3e8 / 17 s | ≤13.6k / ≤4.6e8 / ≤197 s | ≤29.8k / ≤2.3e8 / ≤56 s | hybrid-quadtri (≤7.6e7) | 1.12 | hybrid-quadtri (6 s): 1.00 |
| q_ext_nrms ≤ 1 % | 9.9k / 1.1e8 / 6.4 s | 10.1k / 1.6e8 / 11 s | 14.5k / 2.4e8 / 21 s | 9.0k / 1.5e8 / 16 s | 14.5k / 2.4e8 / 18 s | 63.5k / 3.2e9 / 170 s | 20.4k / 9.5e8 / 35 s | ≤13.6k / ≤4.6e8 / ≤197 s | 50.3k / 5.6e8 / 97 s | hybrid-quadtri (1.5e8) | 0.72 | hybrid-quadtri (16 s): 0.41 |

## B3-g5

| target | subtri-hybrid | subtri-hybrid-nolts | tri-graded-pml2-patch | hybrid-quadtri | tri-graded | cart-conformal-all-graded | nfit-boxed | dgtd-p3 | subgrid-2to1-gap4-lts8 | best baseline (work) | ours / best (work) | best baseline (time): ours / best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| enh_peak ≤ 5 % | 10.4k / 1.2e8 / 5.6 s | 10.3k / 2.3e8 / 15 s | 21.9k / 4.8e8 / 22 s | 12.6k / 3.0e8 / 29 s | 21.9k / 4.8e8 / 31 s | 111k / 8.0e9 / 448 s | 23.1k / 1.2e9 / 54 s | ≤14.2k / ≤5.3e8 / ≤255 s | — | hybrid-quadtri (3.0e8) | 0.39 | tri-graded-pml2-patch (22 s): 0.25 |
| enh_peak ≤ 2 % | 13.6k / 2.0e8 / 12 s | 13.4k / 3.1e8 / 19 s | 30.5k / 9.0e8 / 42 s | 17.1k / 5.0e8 / 45 s | 30.5k / 9.0e8 / 57 s | — | 34.7k / 2.3e9 / 90 s | ≤14.2k / ≤5.3e8 / ≤255 s | — | hybrid-quadtri (5.0e8) | 0.39 | tri-graded-pml2-patch (42 s): 0.28 |
| enh_peak ≤ 1 % | 15.5k / 2.5e8 / 17 s | 15.0k / 3.8e8 / 23 s | 40.5k / 1.5e9 / 71 s | 25.5k / 1.0e9 / 89 s | 40.5k / 1.5e9 / 84 s | — | 92.1k / 1.2e10 / 716 s | ≤14.2k / ≤5.3e8 / ≤255 s | — | dgtd-p3 (≤5.3e8) | 0.46 | tri-graded-pml2-patch (71 s): 0.23 |
| q_ext_nrms ≤ 2 % | ≤9.4k / ≤8.7e7 / ≤4.8 s | ≤9.4k / ≤2.1e8 / ≤13 s | 13.9k / 2.3e8 / 13 s | ≤7.0k / ≤1.6e8 / ≤12 s | 13.9k / 2.3e8 / 13 s | 17.7k / 3.3e8 / 29 s | 14.6k / 5.1e8 / 20 s | ≤14.2k / ≤5.3e8 / ≤255 s | 25.5k / 2.0e8 / 27 s | hybrid-quadtri (≤1.6e8) | 0.55 | hybrid-quadtri (12 s): 0.40 |
| q_ext_nrms ≤ 1 % | ≤9.4k / ≤8.7e7 / ≤4.8 s | ≤9.4k / ≤2.1e8 / ≤13 s | 20.0k / 4.1e8 / 20 s | ≤7.0k / ≤1.6e8 / ≤12 s | 20.0k / 4.1e8 / 26 s | 109k / 7.8e9 / 431 s | 31.8k / 2.0e9 / 82 s | ≤14.2k / ≤5.3e8 / ≤255 s | — | hybrid-quadtri (≤1.6e8) | 0.55 | hybrid-quadtri (12 s): 0.40 |

## B3-g2

| target | subtri-hybrid | subtri-hybrid-nolts | tri-graded-pml2-patch | hybrid-quadtri | tri-graded | cart-conformal-all-graded | nfit-boxed | dgtd-p3 | subgrid-2to1-gap4-lts8 | best baseline (work) | ours / best (work) | best baseline (time): ours / best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| enh_peak ≤ 5 % | 20.8k / 5.6e8 / 20 s | 20.5k / 1.2e9 / 63 s | 58.7k / 2.9e9 / 132 s | 13.2k / 7.6e8 / 52 s | 58.7k / 2.9e9 / 118 s | 143k / 1.2e10 / 570 s | 56.3k / 5.3e9 / 252 s | ≤13.4k / ≤1.2e9 / ≤551 s | — | hybrid-quadtri (7.6e8) | 0.74 | hybrid-quadtri (52 s): 0.37 |
| enh_peak ≤ 2 % | 34.4k / 1.2e9 / 41 s | 33.9k / 2.0e9 / 101 s | 78.2k / 4.7e9 / 239 s | — | 78.2k / 4.7e9 / 198 s | — | 148k / 2.6e10 / 1816 s | ≤13.4k / ≤1.2e9 / ≤551 s | — | dgtd-p3 (≤1.2e9) | 0.99 | tri-graded (198 s): 0.21 |
| enh_peak ≤ 1 % | 44.2k / 1.6e9 / 47 s | 42.3k / 2.6e9 / 136 s | 107k / 7.8e9 / 421 s | — | 107k / 7.8e9 / 354 s | — | — | ≤13.4k / ≤1.2e9 / ≤551 s | — | dgtd-p3 (≤1.2e9) | 1.39 | tri-graded (354 s): 0.13 |
| q_ext_nrms ≤ 2 % | 10.1k / 1.9e8 / 18 s | 10.1k / 5.8e8 / 38 s | ≤55.5k / ≤2.7e9 / ≤118 s | 8.7k / 5.0e8 / 41 s | ≤55.5k / ≤2.7e9 / ≤107 s | 115k / 8.6e9 / 404 s | 23.3k / 1.2e9 / 37 s | ≤13.4k / ≤1.2e9 / ≤551 s | 47.5k / 1.0e9 / 163 s | hybrid-quadtri (5.0e8) | 0.37 | nfit-boxed (37 s): 0.48 |
| q_ext_nrms ≤ 1 % | 14.9k / 3.6e8 / 18 s | 14.7k / 8.4e8 / 51 s | ≤55.5k / ≤2.7e9 / ≤118 s | 12.0k / 6.9e8 / 48 s | ≤55.5k / ≤2.7e9 / ≤107 s | — | 117k / 1.7e10 / 945 s | ≤13.4k / ≤1.2e9 / ≤551 s | — | hybrid-quadtri (6.9e8) | 0.52 | hybrid-quadtri (48 s): 0.37 |

