# B3 baseline summary

Cost from which each target is met and stays met on finer meshes (log-log interpolation along the h ladder; '—' = not reached; '<=' = already met by the coarsest run). Unknowns: 3 per Cartesian cell, nodes + edges for triangles. Time: time-stepping wall time on one uncontended n2-standard-2 core.

## B3-g10

| metric | target | arm | unknowns | work | time [s] |
|---|---|---|---|---|---|
| q_ext_nrms | 0.02 | cart-conformal-all-graded | 1.82e+04 | 3.49e+08 | 20 |
| q_ext_nrms | 0.02 | cart-conformal-all-uniform | 4.72e+04 | 8.85e+08 | 25 |
| q_ext_nrms | 0.02 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.02 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.02 | tri-graded | <= 6.05e+03 | <= 9.54e+07 | <= 14 |
| q_ext_nrms | 0.02 | tri-uniform | <= 1.8e+04 | <= 2.84e+08 | <= 16 |
| q_ext_nrms | 0.01 | cart-conformal-all-graded | 6.35e+04 | 3.19e+09 | 122 |
| q_ext_nrms | 0.01 | cart-conformal-all-uniform | 2.09e+05 | 1.04e+10 | 438 |
| q_ext_nrms | 0.01 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.01 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.01 | tri-graded | 6.95e+03 | 1.15e+08 | 12 |
| q_ext_nrms | 0.01 | tri-uniform | 2.2e+04 | 3.59e+08 | 20 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-graded | 4.03e+04 | 1.49e+09 | 70 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-uniform | 1.33e+05 | 5.09e+09 | 193 |
| q_ext_lam_peak | 0.005 | cart-staircase-graded | — | — | — |
| q_ext_lam_peak | 0.005 | cart-staircase-uniform | — | — | — |
| q_ext_lam_peak | 0.005 | tri-graded | <= 6.05e+03 | <= 9.54e+07 | <= 14 |
| q_ext_lam_peak | 0.005 | tri-uniform | 1.91e+04 | 3.05e+08 | 17 |
| q_ext_lam_peak | 0.002 | cart-conformal-all-graded | 7.25e+04 | 3.95e+09 | 162 |
| q_ext_lam_peak | 0.002 | cart-conformal-all-uniform | 2.53e+05 | 1.39e+10 | 522 |
| q_ext_lam_peak | 0.002 | cart-staircase-graded | — | — | — |
| q_ext_lam_peak | 0.002 | cart-staircase-uniform | — | — | — |
| q_ext_lam_peak | 0.002 | tri-graded | 1.27e+04 | 2.88e+08 | 27 |
| q_ext_lam_peak | 0.002 | tri-uniform | 4.47e+04 | 9e+08 | 38 |
| enh_peak | 0.05 | cart-conformal-all-graded | 1.41e+05 | 1.15e+10 | 659 |
| enh_peak | 0.05 | cart-conformal-all-uniform | 5.2e+05 | 4.19e+10 | 1530 |
| enh_peak | 0.05 | cart-staircase-graded | — | — | — |
| enh_peak | 0.05 | cart-staircase-uniform | — | — | — |
| enh_peak | 0.05 | tri-graded | <= 6.05e+03 | <= 9.54e+07 | <= 14 |
| enh_peak | 0.05 | tri-uniform | <= 1.8e+04 | <= 2.84e+08 | <= 16 |
| enh_peak | 0.02 | cart-conformal-all-graded | — | — | — |
| enh_peak | 0.02 | cart-conformal-all-uniform | — | — | — |
| enh_peak | 0.02 | cart-staircase-graded | — | — | — |
| enh_peak | 0.02 | cart-staircase-uniform | — | — | — |
| enh_peak | 0.02 | tri-graded | 3.99e+04 | 2.06e+09 | 132 |
| enh_peak | 0.02 | tri-uniform | 3.17e+04 | 5.52e+08 | 28 |
| enh_lam | 0.005 | cart-conformal-all-graded | 6.22e+04 | 3.09e+09 | 117 |
| enh_lam | 0.005 | cart-conformal-all-uniform | 2.08e+05 | 1.03e+10 | 437 |
| enh_lam | 0.005 | cart-staircase-graded | — | — | — |
| enh_lam | 0.005 | cart-staircase-uniform | — | — | — |
| enh_lam | 0.005 | tri-graded | <= 6.05e+03 | <= 9.54e+07 | <= 14 |
| enh_lam | 0.005 | tri-uniform | <= 1.8e+04 | <= 2.84e+08 | <= 16 |
| enh_lam | 0.002 | cart-conformal-all-graded | 8.15e+04 | 4.78e+09 | 208 |
| enh_lam | 0.002 | cart-conformal-all-uniform | 2.81e+05 | 1.63e+10 | 574 |
| enh_lam | 0.002 | cart-staircase-graded | — | — | — |
| enh_lam | 0.002 | cart-staircase-uniform | — | — | — |
| enh_lam | 0.002 | tri-graded | 7.56e+03 | 1.29e+08 | 11 |
| enh_lam | 0.002 | tri-uniform | 3.35e+04 | 5.99e+08 | 29 |

Per-run records:

| arm | h [nm] | unknowns | steps | run [s] | Q_ext NRMS | lambda_peak err | height err | FWHM err | enh peak err | reflection |
|---|---|---|---|---|---|---|---|---|---|---|
| cart-conformal-all-graded | 2.86 | 14592 | 15658 | 18 | 2.53e-02 | 1.31e-03 | 5.11e-03 | — | 2.34e-01 | 2.63e-03 |
| cart-conformal-all-graded | 2 | 21432 | 22369 | 22 | 1.67e-02 | 5.31e-03 | 1.93e-02 | — | 1.69e-01 | 4.02e-03 |
| cart-conformal-all-graded | 1.43 | 31860 | 31338 | 60 | 1.83e-02 | 5.63e-03 | 2.73e-02 | — | 9.24e-02 | 2.60e-03 |
| cart-conformal-all-graded | 1 | 52896 | 44810 | 83 | 1.23e-02 | 4.35e-03 | 2.17e-02 | — | 7.96e-02 | 5.39e-04 |
| cart-conformal-all-graded | 0.714 | 90900 | 62762 | 263 | 6.66e-03 | 1.15e-03 | 4.70e-03 | — | 5.95e-02 | 1.17e-03 |
| cart-conformal-all-graded | 0.5 | 165600 | 89680 | 926 | 4.28e-03 | 9.80e-04 | 3.41e-03 | — | 4.69e-02 | 1.32e-03 |
| cart-conformal-all-uniform | 2.86 | 37632 | 15658 | 19 | 2.50e-02 | 1.48e-03 | 6.46e-03 | — | 2.34e-01 | 4.50e-04 |
| cart-conformal-all-uniform | 2 | 58800 | 22369 | 32 | 1.61e-02 | 5.55e-03 | 2.03e-02 | — | 1.69e-01 | 5.82e-04 |
| cart-conformal-all-uniform | 1.43 | 95052 | 31338 | 85 | 1.78e-02 | 5.79e-03 | 2.86e-02 | — | 9.20e-02 | 1.65e-03 |
| cart-conformal-all-uniform | 1 | 172800 | 44810 | 369 | 1.21e-02 | 4.46e-03 | 2.23e-02 | — | 7.93e-02 | 2.61e-03 |
| cart-conformal-all-uniform | 0.714 | 322752 | 62762 | 651 | 6.47e-03 | 1.20e-03 | 5.20e-03 | — | 5.91e-02 | 2.94e-03 |
| cart-conformal-all-uniform | 0.5 | 634800 | 89680 | 2189 | 4.11e-03 | 1.02e-03 | 3.63e-03 | — | 4.66e-02 | 3.28e-03 |
| cart-staircase-graded | 2.86 | 14592 | 15658 | 18 | 2.82e-01 | 5.04e-03 | 8.14e-02 | — | 9.14e-01 | 2.44e-03 |
| cart-staircase-graded | 2 | 21432 | 22369 | 32 | 1.95e-01 | 5.44e-02 | 1.52e-02 | — | 1.77e-01 | 3.92e-03 |
| cart-staircase-graded | 1.43 | 31860 | 31317 | 58 | 1.34e-01 | 2.35e-02 | 3.59e-02 | — | 1.52e-01 | 2.60e-03 |
| cart-staircase-graded | 1 | 52896 | 44739 | 122 | 7.95e-02 | 1.24e-02 | 3.75e-02 | — | 3.38e-01 | 5.31e-04 |
| cart-staircase-graded | 0.714 | 90900 | 62635 | 223 | 5.74e-02 | 8.71e-03 | 3.17e-02 | — | 2.63e-01 | 1.17e-03 |
| cart-staircase-graded | 0.5 | 165600 | 89479 | 475 | 4.10e-02 | 6.97e-03 | 2.30e-02 | — | 2.02e-01 | 1.31e-03 |
| cart-staircase-uniform | 2.86 | 37632 | 15658 | 30 | 2.80e-01 | 4.76e-03 | 8.36e-02 | — | 9.11e-01 | 4.52e-04 |
| cart-staircase-uniform | 2 | 58800 | 22369 | 29 | 1.93e-01 | 5.40e-02 | 1.63e-02 | — | 1.77e-01 | 5.33e-04 |
| cart-staircase-uniform | 1.43 | 95052 | 31317 | 158 | 1.32e-01 | 2.34e-02 | 3.79e-02 | — | 1.53e-01 | 1.59e-03 |
| cart-staircase-uniform | 1 | 172800 | 44739 | 259 | 7.87e-02 | 1.23e-02 | 3.83e-02 | — | 3.38e-01 | 2.59e-03 |
| cart-staircase-uniform | 0.714 | 322752 | 62635 | 699 | 5.68e-02 | 8.65e-03 | 3.23e-02 | — | 2.63e-01 | 2.92e-03 |
| cart-staircase-uniform | 0.5 | 634800 | 89479 | 1398 | 4.06e-02 | 6.94e-03 | 2.33e-02 | — | 2.02e-01 | 3.25e-03 |
| tri-graded | 4 | 6048 | 15774 | 14 | 1.46e-02 | 4.96e-03 | 9.18e-03 | — | 4.33e-02 | 8.09e-03 |
| tri-graded | 2.86 | 7988 | 17344 | 10 | 6.80e-03 | 2.07e-03 | 2.50e-03 | — | 1.34e-02 | 1.37e-02 |
| tri-graded | 2 | 12489 | 22478 | 27 | 7.63e-03 | 2.03e-03 | 5.83e-03 | — | 4.29e-03 | 1.31e-02 |
| tri-graded | 1.43 | 20356 | 32944 | 49 | 2.53e-03 | 1.28e-03 | 2.55e-04 | — | 7.17e-03 | 4.88e-03 |
| tri-graded | 1 | 35480 | 48514 | 124 | 3.18e-03 | 1.05e-03 | 3.04e-03 | — | 2.28e-02 | 5.37e-03 |
| tri-graded | 0.714 | 61731 | 64846 | 165 | 1.71e-03 | 8.33e-04 | 1.72e-03 | — | 1.23e-02 | 3.70e-03 |
| tri-graded | 0.5 | 114355 | 93166 | 418 | 1.40e-03 | 6.56e-04 | 1.71e-03 | — | 7.73e-03 | 4.16e-03 |
| tri-uniform | 4 | 18024 | 15774 | 16 | 1.29e-02 | 5.34e-03 | 9.15e-03 | — | 4.79e-02 | 5.28e-03 |
| tri-uniform | 2.86 | 31331 | 17344 | 27 | 6.37e-03 | 2.91e-03 | 4.51e-03 | — | 2.09e-02 | 3.70e-03 |
| tri-uniform | 2 | 57895 | 22478 | 49 | 3.34e-03 | 1.52e-03 | 2.93e-03 | — | 1.68e-03 | 4.16e-03 |
| tri-uniform | 1.43 | 105685 | 32944 | 64 | 1.69e-03 | 8.73e-04 | 1.74e-03 | — | 7.17e-03 | 4.88e-03 |
| tri-uniform | 1 | 203661 | 48514 | 215 | 1.05e-03 | 5.40e-04 | 1.38e-03 | — | 8.86e-04 | 5.31e-03 |
| tri-uniform | 0.714 | 383401 | 64846 | 723 | 8.00e-04 | 3.69e-04 | 1.14e-03 | — | 2.69e-03 | 5.83e-03 |
| tri-uniform | 0.5 | 758534 | 93166 | 1421 | 7.30e-04 | 2.81e-04 | 1.05e-03 | — | 1.22e-03 | 6.19e-03 |

## B3-g5

| metric | target | arm | unknowns | work | time [s] |
|---|---|---|---|---|---|
| q_ext_nrms | 0.02 | cart-conformal-all-graded | 1.77e+04 | 3.33e+08 | 22 |
| q_ext_nrms | 0.02 | cart-conformal-all-uniform | 4.78e+04 | 9.06e+08 | 25 |
| q_ext_nrms | 0.02 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.02 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.02 | tri-graded | 6.51e+03 | 1.06e+08 | 10 |
| q_ext_nrms | 0.02 | tri-uniform | 1.97e+04 | 3.15e+08 | 16 |
| q_ext_nrms | 0.01 | cart-conformal-all-graded | 1.09e+05 | 7.76e+09 | 468 |
| q_ext_nrms | 0.01 | cart-conformal-all-uniform | 4.25e+05 | 3.08e+10 | 997 |
| q_ext_nrms | 0.01 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.01 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.01 | tri-graded | 1.05e+04 | 2.15e+08 | 19 |
| q_ext_nrms | 0.01 | tri-uniform | 3.47e+04 | 6.29e+08 | 20 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-graded | 2.71e+04 | 7.54e+08 | 51 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-uniform | 8.23e+04 | 2.33e+09 | 89 |
| q_ext_lam_peak | 0.005 | cart-staircase-graded | — | — | — |
| q_ext_lam_peak | 0.005 | cart-staircase-uniform | — | — | — |
| q_ext_lam_peak | 0.005 | tri-graded | 7.88e+03 | 1.37e+08 | 13 |
| q_ext_lam_peak | 0.005 | tri-uniform | 3.09e+04 | 5.34e+08 | 17 |
| q_ext_lam_peak | 0.002 | cart-conformal-all-graded | — | — | — |
| q_ext_lam_peak | 0.002 | cart-conformal-all-uniform | — | — | — |
| q_ext_lam_peak | 0.002 | cart-staircase-graded | — | — | — |
| q_ext_lam_peak | 0.002 | cart-staircase-uniform | — | — | — |
| q_ext_lam_peak | 0.002 | tri-graded | 1.56e+04 | 4.24e+08 | 40 |
| q_ext_lam_peak | 0.002 | tri-uniform | 6.58e+04 | 1.6e+09 | 59 |
| enh_peak | 0.05 | cart-conformal-all-graded | 1.11e+05 | 8e+09 | 481 |
| enh_peak | 0.05 | cart-conformal-all-uniform | 4.14e+05 | 2.96e+10 | 959 |
| enh_peak | 0.05 | cart-staircase-graded | — | — | — |
| enh_peak | 0.05 | cart-staircase-uniform | — | — | — |
| enh_peak | 0.05 | tri-graded | 7.23e+03 | 1.22e+08 | 11 |
| enh_peak | 0.05 | tri-uniform | 2.75e+04 | 4.65e+08 | 17 |
| enh_peak | 0.02 | cart-conformal-all-graded | — | — | — |
| enh_peak | 0.02 | cart-conformal-all-uniform | — | — | — |
| enh_peak | 0.02 | cart-staircase-graded | — | — | — |
| enh_peak | 0.02 | cart-staircase-uniform | — | — | — |
| enh_peak | 0.02 | tri-graded | 8.98e+03 | 1.68e+08 | 15 |
| enh_peak | 0.02 | tri-uniform | 3.63e+04 | 6.71e+08 | 22 |
| enh_lam | 0.005 | cart-conformal-all-graded | 2.34e+04 | 5.67e+08 | 41 |
| enh_lam | 0.005 | cart-conformal-all-uniform | 6.66e+04 | 1.63e+09 | 47 |
| enh_lam | 0.005 | cart-staircase-graded | 1.18e+05 | 8.73e+09 | 311 |
| enh_lam | 0.005 | cart-staircase-uniform | 4.42e+05 | 3.27e+10 | 1126 |
| enh_lam | 0.005 | tri-graded | <= 6e+03 | <= 9.47e+07 | <= 9 |
| enh_lam | 0.005 | tri-uniform | <= 1.8e+04 | <= 2.84e+08 | <= 16 |
| enh_lam | 0.002 | cart-conformal-all-graded | 9.61e+04 | 6.31e+09 | 389 |
| enh_lam | 0.002 | cart-conformal-all-uniform | 3.52e+05 | 2.32e+10 | 743 |
| enh_lam | 0.002 | cart-staircase-graded | 1.44e+05 | 1.21e+10 | 503 |
| enh_lam | 0.002 | cart-staircase-uniform | 5.57e+05 | 4.66e+10 | 1668 |
| enh_lam | 0.002 | tri-graded | 1.61e+04 | 4.45e+08 | 42 |
| enh_lam | 0.002 | tri-uniform | 7.46e+04 | 1.97e+09 | 76 |

Per-run records:

| arm | h [nm] | unknowns | steps | run [s] | Q_ext NRMS | lambda_peak err | height err | FWHM err | enh peak err | reflection |
|---|---|---|---|---|---|---|---|---|---|---|
| cart-conformal-all-graded | 4 | 10752 | 11184 | 14 | 3.81e-02 | 1.36e-02 | 8.36e-02 | 9.22e-02 | 8.42e-01 | 4.54e-03 |
| cart-conformal-all-graded | 2.86 | 14592 | 15658 | 13 | 2.70e-02 | 9.71e-03 | 9.85e-03 | 3.20e-02 | 4.22e-01 | 2.38e-03 |
| cart-conformal-all-graded | 2 | 21432 | 22369 | 37 | 1.50e-02 | 5.70e-03 | 3.73e-02 | 3.39e-02 | 2.15e-01 | 3.93e-03 |
| cart-conformal-all-graded | 1.43 | 30780 | 31338 | 61 | 1.43e-02 | 4.65e-03 | 1.08e-02 | 4.82e-03 | 2.42e-01 | 1.07e-03 |
| cart-conformal-all-graded | 1 | 52200 | 44810 | 191 | 1.51e-02 | 2.90e-03 | 1.76e-02 | 9.53e-04 | 8.34e-02 | 5.56e-04 |
| cart-conformal-all-graded | 0.714 | 89100 | 62762 | 348 | 1.08e-02 | 3.81e-03 | 1.18e-02 | 2.60e-03 | 5.47e-02 | 1.04e-03 |
| cart-conformal-all-graded | 0.5 | 160800 | 89680 | 822 | 8.58e-03 | 2.28e-03 | 4.85e-03 | 6.55e-03 | 4.31e-02 | 1.32e-03 |
| cart-conformal-all-uniform | 4 | 23232 | 11184 | 19 | 3.77e-02 | 1.25e-02 | 8.52e-02 | 9.05e-02 | 8.43e-01 | 7.27e-04 |
| cart-conformal-all-uniform | 2.86 | 37632 | 15658 | 19 | 2.79e-02 | 9.93e-03 | 1.13e-02 | 3.27e-02 | 4.23e-01 | 4.04e-04 |
| cart-conformal-all-uniform | 2 | 58800 | 22369 | 33 | 1.50e-02 | 5.45e-03 | 3.89e-02 | 3.32e-02 | 2.15e-01 | 4.09e-04 |
| cart-conformal-all-uniform | 1.43 | 95052 | 31338 | 135 | 1.47e-02 | 4.82e-03 | 1.18e-02 | 5.50e-03 | 2.42e-01 | 1.22e-03 |
| cart-conformal-all-uniform | 1 | 172800 | 44810 | 314 | 1.53e-02 | 3.02e-03 | 1.83e-02 | 5.05e-04 | 8.33e-02 | 1.89e-03 |
| cart-conformal-all-uniform | 0.714 | 322752 | 62762 | 647 | 1.10e-02 | 3.88e-03 | 1.23e-02 | 2.90e-03 | 5.46e-02 | 2.12e-03 |
| cart-conformal-all-uniform | 0.5 | 634800 | 89680 | 1882 | 8.67e-03 | 2.32e-03 | 5.02e-03 | 6.83e-03 | 4.30e-02 | 2.35e-03 |
| cart-staircase-graded | 4 | 10752 | 11184 | 12 | 2.66e-01 | 5.01e-01 | 1.54e-01 | — | 8.70e-01 | 7.23e-03 |
| cart-staircase-graded | 2.86 | 14592 | 15658 | 19 | 1.12e-01 | 6.59e-03 | 2.12e-01 | — | 8.62e-01 | 5.38e-03 |
| cart-staircase-graded | 2 | 21432 | 22369 | 32 | 2.06e-01 | 7.96e-02 | 2.95e-02 | 3.94e-01 | 3.84e-01 | 3.83e-03 |
| cart-staircase-graded | 1.43 | 30780 | 31317 | 47 | 2.24e-01 | 6.55e-02 | 1.90e-01 | — | 5.70e-01 | 1.02e-03 |
| cart-staircase-graded | 1 | 52200 | 44739 | 111 | 1.37e-01 | 3.30e-02 | 8.01e-02 | — | 1.18e-01 | 6.71e-04 |
| cart-staircase-graded | 0.714 | 89100 | 62635 | 161 | 1.03e-01 | 2.22e-02 | 7.98e-02 | — | 1.63e-01 | 1.01e-03 |
| cart-staircase-graded | 0.5 | 160800 | 89479 | 648 | 4.98e-02 | 5.96e-03 | 4.46e-02 | 1.34e-01 | 3.22e-01 | 1.29e-03 |
| cart-staircase-uniform | 4 | 23232 | 11184 | 10 | 2.66e-01 | 5.01e-01 | 1.55e-01 | — | 8.70e-01 | 8.61e-04 |
| cart-staircase-uniform | 2.86 | 37632 | 15658 | 30 | 1.12e-01 | 6.94e-03 | 2.14e-01 | — | 8.61e-01 | 5.75e-04 |
| cart-staircase-uniform | 2 | 58800 | 22369 | 69 | 2.05e-01 | 7.93e-02 | 3.09e-02 | 3.92e-01 | 3.84e-01 | 3.56e-04 |
| cart-staircase-uniform | 1.43 | 95052 | 31317 | 122 | 2.23e-01 | 6.54e-02 | 1.91e-01 | — | 5.71e-01 | 9.51e-04 |
| cart-staircase-uniform | 1 | 172800 | 44739 | 334 | 1.36e-01 | 3.29e-02 | 8.10e-02 | — | 1.18e-01 | 1.74e-03 |
| cart-staircase-uniform | 0.714 | 322752 | 62635 | 658 | 1.03e-01 | 2.22e-02 | 8.03e-02 | — | 1.62e-01 | 2.00e-03 |
| cart-staircase-uniform | 0.5 | 634800 | 89479 | 2080 | 4.95e-02 | 5.92e-03 | 4.48e-02 | 1.33e-01 | 3.22e-01 | 2.28e-03 |
| tri-graded | 4 | 6004 | 15774 | 9 | 2.33e-02 | 9.34e-03 | 1.13e-02 | 1.82e-02 | 1.54e-01 | 9.73e-03 |
| tri-graded | 2.86 | 7864 | 17344 | 13 | 1.40e-02 | 5.02e-03 | 1.84e-03 | 1.66e-02 | 3.00e-02 | 9.12e-03 |
| tri-graded | 2 | 12309 | 22478 | 23 | 8.30e-03 | 2.70e-03 | 4.09e-03 | 6.82e-03 | 7.62e-03 | 1.02e-02 |
| tri-graded | 1.43 | 20052 | 32944 | 70 | 3.37e-03 | 1.47e-03 | 3.19e-04 | 2.21e-03 | 8.59e-03 | 5.44e-03 |
| tri-graded | 1 | 35160 | 48514 | 141 | 3.12e-03 | 1.09e-03 | 2.90e-03 | 1.05e-03 | 1.04e-03 | 4.47e-03 |
| tri-graded | 0.714 | 61411 | 64846 | 333 | 1.68e-03 | 7.63e-04 | 1.63e-03 | 1.54e-04 | 1.23e-02 | 3.39e-03 |
| tri-graded | 0.5 | 114003 | 93166 | 705 | 1.31e-03 | 5.88e-04 | 1.66e-03 | 5.45e-04 | 6.97e-03 | 3.73e-03 |
| tri-uniform | 4 | 18020 | 15774 | 16 | 2.23e-02 | 9.32e-03 | 7.77e-03 | 1.60e-02 | 1.61e-01 | 4.49e-03 |
| tri-uniform | 2.86 | 31355 | 17344 | 17 | 1.14e-02 | 4.91e-03 | 4.39e-03 | 7.26e-03 | 3.46e-02 | 3.39e-03 |
| tri-uniform | 2 | 57983 | 22478 | 46 | 5.29e-03 | 2.30e-03 | 2.84e-03 | 2.50e-03 | 3.54e-03 | 3.73e-03 |
| tri-uniform | 1.43 | 105681 | 32944 | 152 | 2.54e-03 | 1.18e-03 | 1.84e-03 | 2.99e-04 | 1.19e-02 | 4.14e-03 |
| tri-uniform | 1 | 203753 | 48514 | 326 | 1.38e-03 | 6.73e-04 | 1.46e-03 | 5.84e-04 | 1.34e-02 | 4.45e-03 |
| tri-uniform | 0.714 | 383417 | 64846 | 762 | 8.98e-04 | 4.10e-04 | 1.23e-03 | 1.08e-03 | 1.25e-03 | 4.78e-03 |
| tri-uniform | 0.5 | 758486 | 93166 | 1891 | 7.60e-04 | 2.81e-04 | 1.15e-03 | 1.32e-03 | 2.57e-04 | 5.03e-03 |

## B3-g2

| metric | target | arm | unknowns | work | time [s] |
|---|---|---|---|---|---|
| q_ext_nrms | 0.02 | cart-conformal-all-graded | 1.15e+05 | 8.57e+09 | 494 |
| q_ext_nrms | 0.02 | cart-conformal-all-uniform | 4.52e+05 | 3.39e+10 | 1219 |
| q_ext_nrms | 0.02 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.02 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.02 | tri-graded | <= 3.48e+04 | <= 1.69e+09 | <= 97 |
| q_ext_nrms | 0.02 | tri-uniform | <= 2.04e+05 | <= 9.88e+09 | <= 213 |
| q_ext_nrms | 0.01 | cart-conformal-all-graded | — | — | — |
| q_ext_nrms | 0.01 | cart-conformal-all-uniform | — | — | — |
| q_ext_nrms | 0.01 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.01 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.01 | tri-graded | <= 3.48e+04 | <= 1.69e+09 | <= 97 |
| q_ext_nrms | 0.01 | tri-uniform | <= 2.04e+05 | <= 9.88e+09 | <= 213 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-graded | 1.33e+05 | 1.07e+10 | 590 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-uniform | 5.34e+05 | 4.37e+10 | 1534 |
| q_ext_lam_peak | 0.005 | cart-staircase-graded | — | — | — |
| q_ext_lam_peak | 0.005 | cart-staircase-uniform | — | — | — |
| q_ext_lam_peak | 0.005 | tri-graded | <= 3.48e+04 | <= 1.69e+09 | <= 97 |
| q_ext_lam_peak | 0.005 | tri-uniform | <= 2.04e+05 | <= 9.88e+09 | <= 213 |
| q_ext_lam_peak | 0.002 | cart-conformal-all-graded | — | — | — |
| q_ext_lam_peak | 0.002 | cart-conformal-all-uniform | — | — | — |
| q_ext_lam_peak | 0.002 | cart-staircase-graded | — | — | — |
| q_ext_lam_peak | 0.002 | cart-staircase-uniform | — | — | — |
| q_ext_lam_peak | 0.002 | tri-graded | <= 3.48e+04 | <= 1.69e+09 | <= 97 |
| q_ext_lam_peak | 0.002 | tri-uniform | <= 2.04e+05 | <= 9.88e+09 | <= 213 |
| enh_peak | 0.05 | cart-conformal-all-graded | 1.43e+05 | 1.2e+10 | 643 |
| enh_peak | 0.05 | cart-conformal-all-uniform | 5.63e+05 | 4.73e+10 | 1651 |
| enh_peak | 0.05 | cart-staircase-graded | — | — | — |
| enh_peak | 0.05 | cart-staircase-uniform | — | — | — |
| enh_peak | 0.05 | tri-graded | <= 3.48e+04 | <= 1.69e+09 | <= 97 |
| enh_peak | 0.05 | tri-uniform | <= 2.04e+05 | <= 9.88e+09 | <= 213 |
| enh_peak | 0.02 | cart-conformal-all-graded | — | — | — |
| enh_peak | 0.02 | cart-conformal-all-uniform | — | — | — |
| enh_peak | 0.02 | cart-staircase-graded | — | — | — |
| enh_peak | 0.02 | cart-staircase-uniform | — | — | — |
| enh_peak | 0.02 | tri-graded | <= 3.48e+04 | <= 1.69e+09 | <= 97 |
| enh_peak | 0.02 | tri-uniform | <= 2.04e+05 | <= 9.88e+09 | <= 213 |
| enh_lam | 0.005 | cart-conformal-all-graded | 1.39e+05 | 1.15e+10 | 623 |
| enh_lam | 0.005 | cart-conformal-all-uniform | 5.51e+05 | 4.58e+10 | 1602 |
| enh_lam | 0.005 | cart-staircase-graded | — | — | — |
| enh_lam | 0.005 | cart-staircase-uniform | — | — | — |
| enh_lam | 0.005 | tri-graded | <= 3.48e+04 | <= 1.69e+09 | <= 97 |
| enh_lam | 0.005 | tri-uniform | <= 2.04e+05 | <= 9.88e+09 | <= 213 |
| enh_lam | 0.002 | cart-conformal-all-graded | — | — | — |
| enh_lam | 0.002 | cart-conformal-all-uniform | — | — | — |
| enh_lam | 0.002 | cart-staircase-graded | — | — | — |
| enh_lam | 0.002 | cart-staircase-uniform | — | — | — |
| enh_lam | 0.002 | tri-graded | 3.89e+04 | 2e+09 | 110 |
| enh_lam | 0.002 | tri-uniform | 2.13e+05 | 1.06e+10 | 228 |

Per-run records:

| arm | h [nm] | unknowns | steps | run [s] | Q_ext NRMS | lambda_peak err | height err | FWHM err | enh peak err | reflection |
|---|---|---|---|---|---|---|---|---|---|---|
| cart-conformal-all-graded | 1 | 50808 | 44810 | 118 | 3.47e-02 | 9.47e-04 | 3.15e-02 | 9.58e-01 | 1.32e-01 | 5.37e-04 |
| cart-conformal-all-graded | 0.714 | 87300 | 62762 | 348 | 2.40e-02 | 6.77e-03 | 3.66e-02 | 9.61e-01 | 1.48e-01 | 1.21e-03 |
| cart-conformal-all-graded | 0.5 | 158400 | 89680 | 734 | 1.63e-02 | 4.41e-03 | 2.13e-02 | 4.26e-02 | 3.97e-02 | 8.94e-04 |
| cart-conformal-all-uniform | 1 | 172800 | 44810 | 200 | 3.51e-02 | 1.07e-03 | 3.23e-02 | 9.58e-01 | 1.32e-01 | 1.39e-03 |
| cart-conformal-all-uniform | 0.714 | 322752 | 62762 | 764 | 2.43e-02 | 6.88e-03 | 3.70e-02 | 9.60e-01 | 1.47e-01 | 1.54e-03 |
| cart-conformal-all-uniform | 0.5 | 634800 | 89680 | 1951 | 1.65e-02 | 4.48e-03 | 2.15e-02 | 4.18e-02 | 3.95e-02 | 1.72e-03 |
| cart-staircase-graded | 1 | 50808 | 44739 | 115 | 1.14e-01 | 3.92e-02 | 7.21e-02 | 1.34e+00 | 5.36e-01 | 6.04e-04 |
| cart-staircase-graded | 0.714 | 87300 | 62635 | 219 | 2.10e-01 | 1.32e-01 | 1.85e-01 | — | 1.54e-01 | 1.16e-03 |
| cart-staircase-graded | 0.5 | 158400 | 89479 | 729 | 8.55e-02 | 3.29e-02 | 1.33e-01 | — | 4.62e-01 | 8.60e-04 |
| cart-staircase-uniform | 1 | 172800 | 44739 | 193 | 1.13e-01 | 3.91e-02 | 7.29e-02 | 1.34e+00 | 5.36e-01 | 1.26e-03 |
| cart-staircase-uniform | 0.714 | 322752 | 62635 | 523 | 2.09e-01 | 1.32e-01 | 1.86e-01 | — | 1.54e-01 | 1.34e-03 |
| cart-staircase-uniform | 0.5 | 634800 | 89479 | 1702 | 8.52e-02 | 3.27e-02 | 1.33e-01 | — | 4.62e-01 | 1.62e-03 |
| tri-graded | 1 | 34756 | 48514 | 97 | 5.40e-03 | 1.70e-03 | 2.03e-03 | 3.09e-03 | 1.18e-02 | 4.00e-03 |
| tri-graded | 0.714 | 60867 | 64846 | 180 | 2.52e-03 | 8.94e-04 | 1.27e-03 | 3.42e-03 | 9.85e-03 | 3.23e-03 |
| tri-graded | 0.5 | 113279 | 93166 | 706 | 1.58e-03 | 5.56e-04 | 1.44e-03 | 2.78e-03 | 1.80e-02 | 3.49e-03 |
| tri-uniform | 1 | 203709 | 48514 | 213 | 4.02e-03 | 1.36e-03 | 8.19e-04 | 5.37e-03 | 1.07e-02 | 3.94e-03 |
| tri-uniform | 0.714 | 383453 | 64846 | 563 | 1.81e-03 | 6.59e-04 | 1.06e-03 | 3.95e-03 | 1.07e-02 | 4.15e-03 |
| tri-uniform | 0.5 | 758526 | 93166 | 2083 | 1.04e-03 | 3.56e-04 | 1.10e-03 | 3.25e-03 | 1.87e-02 | 4.32e-03 |
