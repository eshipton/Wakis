# B3m2 baseline summary

Cost from which each target is met and stays met on finer meshes (log-log interpolation along the h ladder; '—' = not reached; '<=' = already met by the coarsest run). Unknowns: 3 per Cartesian cell, nodes + edges for triangles. Time: time-stepping wall time on one uncontended n2-standard-2 core.

## B3-g10

| metric | target | arm | unknowns | work | time [s] |
|---|---|---|---|---|---|
| q_ext_nrms | 0.02 | cart-conformal-all-graded | 1.82e+04 | 3.49e+08 | 29 |
| q_ext_nrms | 0.02 | cart-conformal-all-uniform | 4.72e+04 | 8.85e+08 | 27 |
| q_ext_nrms | 0.02 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.02 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.02 | tri-graded | <= 1.34e+04 | <= 2.11e+08 | <= 12 |
| q_ext_nrms | 0.02 | tri-uniform | <= 3.88e+04 | <= 6.12e+08 | <= 21 |
| q_ext_nrms | 0.01 | cart-conformal-all-graded | 6.35e+04 | 3.19e+09 | 170 |
| q_ext_nrms | 0.01 | cart-conformal-all-uniform | 2.09e+05 | 1.04e+10 | 342 |
| q_ext_nrms | 0.01 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.01 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.01 | tri-graded | 1.45e+04 | 2.39e+08 | 19 |
| q_ext_nrms | 0.01 | tri-uniform | 4.54e+04 | 7.42e+08 | 24 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-graded | 4.03e+04 | 1.49e+09 | 62 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-uniform | 1.33e+05 | 5.09e+09 | 197 |
| q_ext_lam_peak | 0.005 | cart-staircase-graded | — | — | — |
| q_ext_lam_peak | 0.005 | cart-staircase-uniform | — | — | — |
| q_ext_lam_peak | 0.005 | tri-graded | <= 1.34e+04 | <= 2.11e+08 | <= 12 |
| q_ext_lam_peak | 0.005 | tri-uniform | 3.93e+04 | 6.22e+08 | 21 |
| q_ext_lam_peak | 0.002 | cart-conformal-all-graded | 7.25e+04 | 3.95e+09 | 238 |
| q_ext_lam_peak | 0.002 | cart-conformal-all-uniform | 2.53e+05 | 1.39e+10 | 501 |
| q_ext_lam_peak | 0.002 | cart-staircase-graded | — | — | — |
| q_ext_lam_peak | 0.002 | cart-staircase-uniform | — | — | — |
| q_ext_lam_peak | 0.002 | tri-graded | 1.57e+04 | 2.72e+08 | 29 |
| q_ext_lam_peak | 0.002 | tri-uniform | 7.42e+04 | 1.43e+09 | 40 |
| enh_peak | 0.05 | cart-conformal-all-graded | 1.41e+05 | 1.15e+10 | 731 |
| enh_peak | 0.05 | cart-conformal-all-uniform | 5.2e+05 | 4.19e+10 | 1419 |
| enh_peak | 0.05 | cart-staircase-graded | — | — | — |
| enh_peak | 0.05 | cart-staircase-uniform | — | — | — |
| enh_peak | 0.05 | tri-graded | 1.44e+04 | 2.36e+08 | 18 |
| enh_peak | 0.05 | tri-uniform | 4.83e+04 | 7.98e+08 | 26 |
| enh_peak | 0.02 | cart-conformal-all-graded | — | — | — |
| enh_peak | 0.02 | cart-conformal-all-uniform | — | — | — |
| enh_peak | 0.02 | cart-staircase-graded | — | — | — |
| enh_peak | 0.02 | cart-staircase-uniform | — | — | — |
| enh_peak | 0.02 | tri-graded | 1.65e+04 | 2.94e+08 | 30 |
| enh_peak | 0.02 | tri-uniform | 6.96e+04 | 1.3e+09 | 37 |
| enh_peak | 0.01 | cart-conformal-all-graded | — | — | — |
| enh_peak | 0.01 | cart-conformal-all-uniform | — | — | — |
| enh_peak | 0.01 | cart-staircase-graded | — | — | — |
| enh_peak | 0.01 | cart-staircase-uniform | — | — | — |
| enh_peak | 0.01 | tri-graded | 1.81e+04 | 3.45e+08 | 28 |
| enh_peak | 0.01 | tri-uniform | 8.78e+04 | 1.85e+09 | 48 |
| enh_lam | 0.005 | cart-conformal-all-graded | 6.22e+04 | 3.09e+09 | 161 |
| enh_lam | 0.005 | cart-conformal-all-uniform | 2.08e+05 | 1.03e+10 | 339 |
| enh_lam | 0.005 | cart-staircase-graded | — | — | — |
| enh_lam | 0.005 | cart-staircase-uniform | — | — | — |
| enh_lam | 0.005 | tri-graded | <= 1.34e+04 | <= 2.11e+08 | <= 12 |
| enh_lam | 0.005 | tri-uniform | <= 3.88e+04 | <= 6.12e+08 | <= 21 |
| enh_lam | 0.002 | cart-conformal-all-graded | 8.15e+04 | 4.78e+09 | 319 |
| enh_lam | 0.002 | cart-conformal-all-uniform | 2.81e+05 | 1.63e+10 | 616 |
| enh_lam | 0.002 | cart-staircase-graded | — | — | — |
| enh_lam | 0.002 | cart-staircase-uniform | — | — | — |
| enh_lam | 0.002 | tri-graded | <= 1.34e+04 | <= 2.11e+08 | <= 12 |
| enh_lam | 0.002 | tri-uniform | <= 3.88e+04 | <= 6.12e+08 | <= 21 |

Per-run records:

| arm | h [nm] | unknowns | steps | run [s] | Q_ext NRMS | lambda_peak err | height err | FWHM err | enh peak err | reflection |
|---|---|---|---|---|---|---|---|---|---|---|
| cart-conformal-all-graded | 2.86 | 14592 | 15658 | 26 | 2.53e-02 | 1.31e-03 | 5.11e-03 | — | 2.34e-01 | 2.63e-03 |
| cart-conformal-all-graded | 2 | 21432 | 22369 | 31 | 1.67e-02 | 5.31e-03 | 1.93e-02 | — | 1.69e-01 | 4.02e-03 |
| cart-conformal-all-graded | 1.43 | 31860 | 31338 | 39 | 1.83e-02 | 5.63e-03 | 2.73e-02 | — | 9.24e-02 | 2.60e-03 |
| cart-conformal-all-graded | 1 | 52896 | 44810 | 107 | 1.23e-02 | 4.35e-03 | 2.17e-02 | — | 7.96e-02 | 5.39e-04 |
| cart-conformal-all-graded | 0.714 | 90900 | 62762 | 420 | 6.66e-03 | 1.15e-03 | 4.70e-03 | — | 5.95e-02 | 1.17e-03 |
| cart-conformal-all-graded | 0.5 | 165600 | 89680 | 897 | 4.28e-03 | 9.80e-04 | 3.41e-03 | — | 4.69e-02 | 1.32e-03 |
| cart-conformal-all-uniform | 2.86 | 37632 | 15658 | 21 | 2.50e-02 | 1.48e-03 | 6.46e-03 | — | 2.34e-01 | 4.50e-04 |
| cart-conformal-all-uniform | 2 | 58800 | 22369 | 36 | 1.61e-02 | 5.55e-03 | 2.03e-02 | — | 1.69e-01 | 5.82e-04 |
| cart-conformal-all-uniform | 1.43 | 95052 | 31338 | 158 | 1.78e-02 | 5.79e-03 | 2.86e-02 | — | 9.20e-02 | 1.65e-03 |
| cart-conformal-all-uniform | 1 | 172800 | 44810 | 235 | 1.21e-02 | 4.46e-03 | 2.23e-02 | — | 7.93e-02 | 2.61e-03 |
| cart-conformal-all-uniform | 0.714 | 322752 | 62762 | 812 | 6.47e-03 | 1.20e-03 | 5.20e-03 | — | 5.91e-02 | 2.94e-03 |
| cart-conformal-all-uniform | 0.5 | 634800 | 89680 | 1793 | 4.11e-03 | 1.02e-03 | 3.63e-03 | — | 4.66e-02 | 3.28e-03 |
| cart-staircase-graded | 2.86 | 14592 | 15658 | 21 | 2.82e-01 | 5.04e-03 | 8.14e-02 | — | 9.14e-01 | 2.44e-03 |
| cart-staircase-graded | 2 | 21432 | 22369 | 20 | 1.95e-01 | 5.44e-02 | 1.52e-02 | — | 1.77e-01 | 3.92e-03 |
| cart-staircase-graded | 1.43 | 31860 | 31317 | 75 | 1.34e-01 | 2.35e-02 | 3.59e-02 | — | 1.52e-01 | 2.60e-03 |
| cart-staircase-graded | 1 | 52896 | 44739 | 143 | 7.95e-02 | 1.24e-02 | 3.75e-02 | — | 3.38e-01 | 5.31e-04 |
| cart-staircase-graded | 0.714 | 90900 | 62635 | 135 | 5.74e-02 | 8.71e-03 | 3.17e-02 | — | 2.63e-01 | 1.17e-03 |
| cart-staircase-graded | 0.5 | 165600 | 89479 | 754 | 4.10e-02 | 6.97e-03 | 2.30e-02 | — | 2.02e-01 | 1.31e-03 |
| cart-staircase-uniform | 2.86 | 37632 | 15658 | 35 | 2.80e-01 | 4.76e-03 | 8.36e-02 | — | 9.11e-01 | 4.52e-04 |
| cart-staircase-uniform | 2 | 58800 | 22369 | 44 | 1.93e-01 | 5.40e-02 | 1.63e-02 | — | 1.77e-01 | 5.33e-04 |
| cart-staircase-uniform | 1.43 | 95052 | 31317 | 121 | 1.32e-01 | 2.34e-02 | 3.79e-02 | — | 1.53e-01 | 1.59e-03 |
| cart-staircase-uniform | 1 | 172800 | 44739 | 362 | 7.87e-02 | 1.23e-02 | 3.83e-02 | — | 3.38e-01 | 2.59e-03 |
| cart-staircase-uniform | 0.714 | 322752 | 62635 | 516 | 5.68e-02 | 8.65e-03 | 3.23e-02 | — | 2.63e-01 | 2.92e-03 |
| cart-staircase-uniform | 0.5 | 634800 | 89479 | 1942 | 4.06e-02 | 6.94e-03 | 2.33e-02 | — | 2.02e-01 | 3.25e-03 |
| tri-graded | 4 | 13368 | 15774 | 12 | 1.45e-02 | 4.87e-03 | 9.52e-03 | — | 8.02e-02 | 7.72e-03 |
| tri-graded | 2.86 | 15788 | 17344 | 30 | 6.70e-03 | 1.94e-03 | 2.25e-03 | — | 2.73e-02 | 1.35e-02 |
| tri-graded | 2 | 23049 | 22478 | 24 | 7.71e-03 | 1.81e-03 | 5.63e-03 | — | 1.71e-03 | 1.27e-02 |
| tri-graded | 1.43 | 34876 | 32944 | 58 | 2.30e-03 | 1.05e-03 | 2.00e-04 | — | 2.78e-03 | 4.27e-03 |
| tri-graded | 1 | 56240 | 48514 | 128 | 3.26e-03 | 8.24e-04 | 2.43e-03 | — | 2.48e-03 | 4.19e-03 |
| tri-graded | 0.714 | 90771 | 64846 | 210 | 1.58e-03 | 6.11e-04 | 9.79e-04 | — | 6.49e-06 | 6.14e-04 |
| tri-graded | 0.5 | 155875 | 93166 | 808 | 1.33e-03 | 4.49e-04 | 9.64e-04 | — | 7.10e-04 | 1.09e-03 |
| tri-uniform | 4 | 38784 | 15774 | 21 | 1.29e-02 | 5.10e-03 | 8.56e-03 | — | 8.10e-02 | 4.08e-03 |
| tri-uniform | 2.86 | 60371 | 17344 | 31 | 6.34e-03 | 2.69e-03 | 3.82e-03 | — | 3.05e-02 | 6.13e-04 |
| tri-uniform | 2 | 99415 | 22478 | 56 | 3.30e-03 | 1.31e-03 | 2.17e-03 | — | 6.90e-03 | 1.09e-03 |
| tri-uniform | 1.43 | 163765 | 32944 | 278 | 1.55e-03 | 6.75e-04 | 9.61e-04 | — | 5.66e-03 | 2.08e-04 |
| tri-uniform | 1 | 286701 | 48514 | 299 | 8.17e-04 | 3.47e-04 | 5.53e-04 | — | 2.81e-03 | 3.00e-04 |
| tri-uniform | 0.714 | 499561 | 64846 | 741 | 3.91e-04 | 1.80e-04 | 2.82e-04 | — | 2.02e-03 | 1.82e-04 |
| tri-uniform | 0.5 | 924494 | 93166 | 3257 | 1.97e-04 | 9.45e-05 | 1.76e-04 | — | 7.84e-04 | 1.96e-04 |

## B3-g5

| metric | target | arm | unknowns | work | time [s] |
|---|---|---|---|---|---|
| q_ext_nrms | 0.02 | cart-conformal-all-graded | 1.77e+04 | 3.33e+08 | 29 |
| q_ext_nrms | 0.02 | cart-conformal-all-uniform | 4.78e+04 | 9.06e+08 | 33 |
| q_ext_nrms | 0.02 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.02 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.02 | tri-graded | 1.39e+04 | 2.26e+08 | 13 |
| q_ext_nrms | 0.02 | tri-uniform | 4.17e+04 | 6.68e+08 | 23 |
| q_ext_nrms | 0.01 | cart-conformal-all-graded | 1.09e+05 | 7.76e+09 | 431 |
| q_ext_nrms | 0.01 | cart-conformal-all-uniform | 4.25e+05 | 3.08e+10 | 894 |
| q_ext_nrms | 0.01 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.01 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.01 | tri-graded | 2e+04 | 4.11e+08 | 26 |
| q_ext_nrms | 0.01 | tri-uniform | 6.59e+04 | 1.2e+09 | 33 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-graded | 2.71e+04 | 7.54e+08 | 37 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-uniform | 8.23e+04 | 2.33e+09 | 60 |
| q_ext_lam_peak | 0.005 | cart-staircase-graded | — | — | — |
| q_ext_lam_peak | 0.005 | cart-staircase-uniform | — | — | — |
| q_ext_lam_peak | 0.005 | tri-graded | 1.56e+04 | 2.69e+08 | 15 |
| q_ext_lam_peak | 0.005 | tri-uniform | 5.82e+04 | 1e+09 | 28 |
| q_ext_lam_peak | 0.002 | cart-conformal-all-graded | — | — | — |
| q_ext_lam_peak | 0.002 | cart-conformal-all-uniform | — | — | — |
| q_ext_lam_peak | 0.002 | cart-staircase-graded | — | — | — |
| q_ext_lam_peak | 0.002 | cart-staircase-uniform | — | — | — |
| q_ext_lam_peak | 0.002 | tri-graded | 2.62e+04 | 6.66e+08 | 43 |
| q_ext_lam_peak | 0.002 | tri-uniform | 1.04e+05 | 2.43e+09 | 64 |
| enh_peak | 0.05 | cart-conformal-all-graded | 1.11e+05 | 8e+09 | 448 |
| enh_peak | 0.05 | cart-conformal-all-uniform | 4.14e+05 | 2.96e+10 | 862 |
| enh_peak | 0.05 | cart-staircase-graded | — | — | — |
| enh_peak | 0.05 | cart-staircase-uniform | — | — | — |
| enh_peak | 0.05 | tri-graded | 2.19e+04 | 4.78e+08 | 31 |
| enh_peak | 0.05 | tri-uniform | 9.47e+04 | 2.08e+09 | 53 |
| enh_peak | 0.02 | cart-conformal-all-graded | — | — | — |
| enh_peak | 0.02 | cart-conformal-all-uniform | — | — | — |
| enh_peak | 0.02 | cart-staircase-graded | — | — | — |
| enh_peak | 0.02 | cart-staircase-uniform | — | — | — |
| enh_peak | 0.02 | tri-graded | 3.05e+04 | 8.96e+08 | 57 |
| enh_peak | 0.02 | tri-uniform | 1.45e+05 | 4.37e+09 | 153 |
| enh_peak | 0.01 | cart-conformal-all-graded | — | — | — |
| enh_peak | 0.01 | cart-conformal-all-uniform | — | — | — |
| enh_peak | 0.01 | cart-staircase-graded | — | — | — |
| enh_peak | 0.01 | cart-staircase-uniform | — | — | — |
| enh_peak | 0.01 | tri-graded | 4.05e+04 | 1.51e+09 | 84 |
| enh_peak | 0.01 | tri-uniform | 2.03e+05 | 7.78e+09 | 278 |
| enh_lam | 0.005 | cart-conformal-all-graded | 2.34e+04 | 5.67e+08 | 33 |
| enh_lam | 0.005 | cart-conformal-all-uniform | 6.66e+04 | 1.63e+09 | 44 |
| enh_lam | 0.005 | cart-staircase-graded | 1.18e+05 | 8.73e+09 | 271 |
| enh_lam | 0.005 | cart-staircase-uniform | 4.42e+05 | 3.27e+10 | 772 |
| enh_lam | 0.005 | tri-graded | 1.4e+04 | 2.28e+08 | 13 |
| enh_lam | 0.005 | tri-uniform | 4.46e+04 | 7.25e+08 | 24 |
| enh_lam | 0.002 | cart-conformal-all-graded | 9.61e+04 | 6.31e+09 | 334 |
| enh_lam | 0.002 | cart-conformal-all-uniform | 3.52e+05 | 2.32e+10 | 685 |
| enh_lam | 0.002 | cart-staircase-graded | 1.44e+05 | 1.21e+10 | 309 |
| enh_lam | 0.002 | cart-staircase-uniform | 5.57e+05 | 4.66e+10 | 1248 |
| enh_lam | 0.002 | tri-graded | 1.46e+04 | 2.42e+08 | 14 |
| enh_lam | 0.002 | tri-uniform | 4.91e+04 | 8.16e+08 | 26 |

Per-run records:

| arm | h [nm] | unknowns | steps | run [s] | Q_ext NRMS | lambda_peak err | height err | FWHM err | enh peak err | reflection |
|---|---|---|---|---|---|---|---|---|---|---|
| cart-conformal-all-graded | 4 | 10752 | 11184 | 8 | 3.81e-02 | 1.36e-02 | 8.36e-02 | 9.22e-02 | 8.42e-01 | 4.54e-03 |
| cart-conformal-all-graded | 2.86 | 14592 | 15658 | 26 | 2.70e-02 | 9.71e-03 | 9.85e-03 | 3.20e-02 | 4.22e-01 | 2.38e-03 |
| cart-conformal-all-graded | 2 | 21432 | 22369 | 31 | 1.50e-02 | 5.70e-03 | 3.73e-02 | 3.39e-02 | 2.15e-01 | 3.93e-03 |
| cart-conformal-all-graded | 1.43 | 30780 | 31338 | 40 | 1.43e-02 | 4.65e-03 | 1.08e-02 | 4.82e-03 | 2.42e-01 | 1.07e-03 |
| cart-conformal-all-graded | 1 | 52200 | 44810 | 82 | 1.51e-02 | 2.90e-03 | 1.76e-02 | 9.53e-04 | 8.34e-02 | 5.56e-04 |
| cart-conformal-all-graded | 0.714 | 89100 | 62762 | 287 | 1.08e-02 | 3.81e-03 | 1.18e-02 | 2.60e-03 | 5.47e-02 | 1.04e-03 |
| cart-conformal-all-graded | 0.5 | 160800 | 89680 | 935 | 8.58e-03 | 2.28e-03 | 4.85e-03 | 6.55e-03 | 4.31e-02 | 1.32e-03 |
| cart-conformal-all-uniform | 4 | 23232 | 11184 | 19 | 3.77e-02 | 1.25e-02 | 8.52e-02 | 9.05e-02 | 8.43e-01 | 7.27e-04 |
| cart-conformal-all-uniform | 2.86 | 37632 | 15658 | 29 | 2.79e-02 | 9.93e-03 | 1.13e-02 | 3.27e-02 | 4.23e-01 | 4.04e-04 |
| cart-conformal-all-uniform | 2 | 58800 | 22369 | 36 | 1.50e-02 | 5.45e-03 | 3.89e-02 | 3.32e-02 | 2.15e-01 | 4.09e-04 |
| cart-conformal-all-uniform | 1.43 | 95052 | 31338 | 74 | 1.47e-02 | 4.82e-03 | 1.18e-02 | 5.50e-03 | 2.42e-01 | 1.22e-03 |
| cart-conformal-all-uniform | 1 | 172800 | 44810 | 186 | 1.53e-02 | 3.02e-03 | 1.83e-02 | 5.05e-04 | 8.33e-02 | 1.89e-03 |
| cart-conformal-all-uniform | 0.714 | 322752 | 62762 | 604 | 1.10e-02 | 3.88e-03 | 1.23e-02 | 2.90e-03 | 5.46e-02 | 2.12e-03 |
| cart-conformal-all-uniform | 0.5 | 634800 | 89680 | 1588 | 8.67e-03 | 2.32e-03 | 5.02e-03 | 6.83e-03 | 4.30e-02 | 2.35e-03 |
| cart-staircase-graded | 4 | 10752 | 11184 | 7 | 2.66e-01 | 5.01e-01 | 1.54e-01 | — | 8.70e-01 | 7.23e-03 |
| cart-staircase-graded | 2.86 | 14592 | 15658 | 11 | 1.12e-01 | 6.59e-03 | 2.12e-01 | — | 8.62e-01 | 5.38e-03 |
| cart-staircase-graded | 2 | 21432 | 22369 | 38 | 2.06e-01 | 7.96e-02 | 2.95e-02 | 3.94e-01 | 3.84e-01 | 3.83e-03 |
| cart-staircase-graded | 1.43 | 30780 | 31317 | 33 | 2.24e-01 | 6.55e-02 | 1.90e-01 | — | 5.70e-01 | 1.02e-03 |
| cart-staircase-graded | 1 | 52200 | 44739 | 85 | 1.37e-01 | 3.30e-02 | 8.01e-02 | — | 1.18e-01 | 6.71e-04 |
| cart-staircase-graded | 0.714 | 89100 | 62635 | 227 | 1.03e-01 | 2.22e-02 | 7.98e-02 | — | 1.63e-01 | 1.01e-03 |
| cart-staircase-graded | 0.5 | 160800 | 89479 | 331 | 4.98e-02 | 5.96e-03 | 4.46e-02 | 1.34e-01 | 3.22e-01 | 1.29e-03 |
| cart-staircase-uniform | 4 | 23232 | 11184 | 9 | 2.66e-01 | 5.01e-01 | 1.55e-01 | — | 8.70e-01 | 8.61e-04 |
| cart-staircase-uniform | 2.86 | 37632 | 15658 | 19 | 1.12e-01 | 6.94e-03 | 2.14e-01 | — | 8.61e-01 | 5.75e-04 |
| cart-staircase-uniform | 2 | 58800 | 22369 | 80 | 2.05e-01 | 7.93e-02 | 3.09e-02 | 3.92e-01 | 3.84e-01 | 3.56e-04 |
| cart-staircase-uniform | 1.43 | 95052 | 31317 | 60 | 2.23e-01 | 6.54e-02 | 1.91e-01 | — | 5.71e-01 | 9.51e-04 |
| cart-staircase-uniform | 1 | 172800 | 44739 | 314 | 1.36e-01 | 3.29e-02 | 8.10e-02 | — | 1.18e-01 | 1.74e-03 |
| cart-staircase-uniform | 0.714 | 322752 | 62635 | 400 | 1.03e-01 | 2.22e-02 | 8.03e-02 | — | 1.62e-01 | 2.00e-03 |
| cart-staircase-uniform | 0.5 | 634800 | 89479 | 1636 | 4.95e-02 | 5.92e-03 | 4.48e-02 | 1.33e-01 | 3.22e-01 | 2.28e-03 |
| tri-graded | 4 | 13324 | 15774 | 12 | 2.31e-02 | 9.22e-03 | 1.15e-02 | 1.73e-02 | 6.19e-01 | 9.05e-03 |
| tri-graded | 2.86 | 15664 | 17344 | 15 | 1.39e-02 | 4.87e-03 | 1.92e-03 | 1.59e-02 | 1.48e-01 | 8.89e-03 |
| tri-graded | 2 | 22869 | 22478 | 34 | 8.39e-03 | 2.50e-03 | 3.69e-03 | 7.13e-03 | 4.34e-02 | 9.82e-03 |
| tri-graded | 1.43 | 34572 | 32944 | 72 | 3.19e-03 | 1.26e-03 | 3.14e-04 | 3.08e-03 | 1.43e-02 | 4.77e-03 |
| tri-graded | 1 | 55920 | 48514 | 117 | 3.25e-03 | 9.07e-04 | 2.14e-03 | 2.22e-03 | 4.79e-03 | 3.01e-03 |
| tri-graded | 0.714 | 90451 | 64846 | 274 | 1.65e-03 | 5.92e-04 | 8.01e-04 | 1.18e-03 | 2.57e-03 | 5.96e-04 |
| tri-graded | 0.5 | 155523 | 93166 | 810 | 1.30e-03 | 4.28e-04 | 7.84e-04 | 8.57e-04 | 2.57e-03 | 8.51e-04 |
| tri-uniform | 4 | 38780 | 15774 | 22 | 2.23e-02 | 9.13e-03 | 7.04e-03 | 1.71e-02 | 6.36e-01 | 3.04e-03 |
| tri-uniform | 2.86 | 60395 | 17344 | 29 | 1.14e-02 | 4.73e-03 | 3.57e-03 | 8.59e-03 | 1.49e-01 | 5.95e-04 |
| tri-uniform | 2 | 99503 | 22478 | 57 | 5.36e-03 | 2.14e-03 | 1.97e-03 | 3.90e-03 | 4.44e-02 | 8.51e-04 |
| tri-uniform | 1.43 | 163761 | 32944 | 209 | 2.53e-03 | 1.03e-03 | 9.30e-04 | 1.73e-03 | 1.56e-02 | 2.03e-04 |
| tri-uniform | 1 | 286793 | 48514 | 440 | 1.29e-03 | 5.24e-04 | 5.22e-04 | 8.59e-04 | 4.95e-03 | 2.63e-04 |
| tri-uniform | 0.714 | 499577 | 64846 | 815 | 6.26e-04 | 2.64e-04 | 2.74e-04 | 3.62e-04 | 3.15e-03 | 1.81e-04 |
| tri-uniform | 0.5 | 924446 | 93166 | 2133 | 3.12e-04 | 1.37e-04 | 1.72e-04 | 1.27e-04 | 2.10e-03 | 1.94e-04 |

## B3-g2

| metric | target | arm | unknowns | work | time [s] |
|---|---|---|---|---|---|
| q_ext_nrms | 0.02 | cart-conformal-all-graded | 1.15e+05 | 8.57e+09 | 404 |
| q_ext_nrms | 0.02 | cart-conformal-all-uniform | 4.52e+05 | 3.39e+10 | 1253 |
| q_ext_nrms | 0.02 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.02 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.02 | tri-graded | <= 5.55e+04 | <= 2.69e+09 | <= 107 |
| q_ext_nrms | 0.02 | tri-uniform | <= 2.87e+05 | <= 1.39e+10 | <= 702 |
| q_ext_nrms | 0.01 | cart-conformal-all-graded | — | — | — |
| q_ext_nrms | 0.01 | cart-conformal-all-uniform | — | — | — |
| q_ext_nrms | 0.01 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.01 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.01 | tri-graded | <= 5.55e+04 | <= 2.69e+09 | <= 107 |
| q_ext_nrms | 0.01 | tri-uniform | <= 2.87e+05 | <= 1.39e+10 | <= 702 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-graded | 1.33e+05 | 1.07e+10 | 509 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-uniform | 5.34e+05 | 4.37e+10 | 1604 |
| q_ext_lam_peak | 0.005 | cart-staircase-graded | — | — | — |
| q_ext_lam_peak | 0.005 | cart-staircase-uniform | — | — | — |
| q_ext_lam_peak | 0.005 | tri-graded | <= 5.55e+04 | <= 2.69e+09 | <= 107 |
| q_ext_lam_peak | 0.005 | tri-uniform | <= 2.87e+05 | <= 1.39e+10 | <= 702 |
| q_ext_lam_peak | 0.002 | cart-conformal-all-graded | — | — | — |
| q_ext_lam_peak | 0.002 | cart-conformal-all-uniform | — | — | — |
| q_ext_lam_peak | 0.002 | cart-staircase-graded | — | — | — |
| q_ext_lam_peak | 0.002 | cart-staircase-uniform | — | — | — |
| q_ext_lam_peak | 0.002 | tri-graded | <= 5.55e+04 | <= 2.69e+09 | <= 107 |
| q_ext_lam_peak | 0.002 | tri-uniform | <= 2.87e+05 | <= 1.39e+10 | <= 702 |
| enh_peak | 0.05 | cart-conformal-all-graded | 1.43e+05 | 1.2e+10 | 570 |
| enh_peak | 0.05 | cart-conformal-all-uniform | 5.63e+05 | 4.73e+10 | 1736 |
| enh_peak | 0.05 | cart-staircase-graded | — | — | — |
| enh_peak | 0.05 | cart-staircase-uniform | — | — | — |
| enh_peak | 0.05 | tri-graded | 5.87e+04 | 2.94e+09 | 118 |
| enh_peak | 0.05 | tri-uniform | 3.05e+05 | 1.53e+10 | 738 |
| enh_peak | 0.02 | cart-conformal-all-graded | — | — | — |
| enh_peak | 0.02 | cart-conformal-all-uniform | — | — | — |
| enh_peak | 0.02 | cart-staircase-graded | — | — | — |
| enh_peak | 0.02 | cart-staircase-uniform | — | — | — |
| enh_peak | 0.02 | tri-graded | 7.82e+04 | 4.67e+09 | 198 |
| enh_peak | 0.02 | tri-uniform | 4.26e+05 | 2.54e+10 | 962 |
| enh_peak | 0.01 | cart-conformal-all-graded | — | — | — |
| enh_peak | 0.01 | cart-conformal-all-uniform | — | — | — |
| enh_peak | 0.01 | cart-staircase-graded | — | — | — |
| enh_peak | 0.01 | cart-staircase-uniform | — | — | — |
| enh_peak | 0.01 | tri-graded | 1.07e+05 | 7.82e+09 | 354 |
| enh_peak | 0.01 | tri-uniform | 6.17e+05 | 4.53e+10 | 1348 |
| enh_lam | 0.005 | cart-conformal-all-graded | 1.39e+05 | 1.15e+10 | 547 |
| enh_lam | 0.005 | cart-conformal-all-uniform | 5.51e+05 | 4.58e+10 | 1682 |
| enh_lam | 0.005 | cart-staircase-graded | — | — | — |
| enh_lam | 0.005 | cart-staircase-uniform | — | — | — |
| enh_lam | 0.005 | tri-graded | <= 5.55e+04 | <= 2.69e+09 | <= 107 |
| enh_lam | 0.005 | tri-uniform | <= 2.87e+05 | <= 1.39e+10 | <= 702 |
| enh_lam | 0.002 | cart-conformal-all-graded | — | — | — |
| enh_lam | 0.002 | cart-conformal-all-uniform | — | — | — |
| enh_lam | 0.002 | cart-staircase-graded | — | — | — |
| enh_lam | 0.002 | cart-staircase-uniform | — | — | — |
| enh_lam | 0.002 | tri-graded | <= 5.55e+04 | <= 2.69e+09 | <= 107 |
| enh_lam | 0.002 | tri-uniform | <= 2.87e+05 | <= 1.39e+10 | <= 702 |

Per-run records:

| arm | h [nm] | unknowns | steps | run [s] | Q_ext NRMS | lambda_peak err | height err | FWHM err | enh peak err | reflection |
|---|---|---|---|---|---|---|---|---|---|---|
| cart-conformal-all-graded | 1 | 50808 | 44810 | 81 | 3.47e-02 | 9.47e-04 | 3.15e-02 | 9.58e-01 | 1.32e-01 | 5.37e-04 |
| cart-conformal-all-graded | 0.714 | 87300 | 62762 | 257 | 2.40e-02 | 6.77e-03 | 3.66e-02 | 9.61e-01 | 1.48e-01 | 1.21e-03 |
| cart-conformal-all-graded | 0.5 | 158400 | 89680 | 676 | 1.63e-02 | 4.41e-03 | 2.13e-02 | 4.26e-02 | 3.97e-02 | 8.94e-04 |
| cart-conformal-all-uniform | 1 | 172800 | 44810 | 219 | 3.51e-02 | 1.07e-03 | 3.23e-02 | 9.58e-01 | 1.32e-01 | 1.39e-03 |
| cart-conformal-all-uniform | 0.714 | 322752 | 62762 | 757 | 2.43e-02 | 6.88e-03 | 3.70e-02 | 9.60e-01 | 1.47e-01 | 1.54e-03 |
| cart-conformal-all-uniform | 0.5 | 634800 | 89680 | 2079 | 1.65e-02 | 4.48e-03 | 2.15e-02 | 4.18e-02 | 3.95e-02 | 1.72e-03 |
| cart-staircase-graded | 1 | 50808 | 44739 | 73 | 1.14e-01 | 3.92e-02 | 7.21e-02 | 1.34e+00 | 5.36e-01 | 6.04e-04 |
| cart-staircase-graded | 0.714 | 87300 | 62635 | 163 | 2.10e-01 | 1.32e-01 | 1.85e-01 | — | 1.54e-01 | 1.16e-03 |
| cart-staircase-graded | 0.5 | 158400 | 89479 | 417 | 8.55e-02 | 3.29e-02 | 1.33e-01 | — | 4.62e-01 | 8.60e-04 |
| cart-staircase-uniform | 1 | 172800 | 44739 | 357 | 1.13e-01 | 3.91e-02 | 7.29e-02 | 1.34e+00 | 5.36e-01 | 1.26e-03 |
| cart-staircase-uniform | 0.714 | 322752 | 62635 | 637 | 2.09e-01 | 1.32e-01 | 1.86e-01 | — | 1.54e-01 | 1.34e-03 |
| cart-staircase-uniform | 0.5 | 634800 | 89479 | 1321 | 8.52e-02 | 3.27e-02 | 1.33e-01 | — | 4.62e-01 | 1.62e-03 |
| tri-graded | 1 | 55516 | 48514 | 107 | 5.58e-03 | 1.60e-03 | 1.17e-03 | 6.76e-04 | 5.95e-02 | 2.31e-03 |
| tri-graded | 0.714 | 89907 | 64846 | 254 | 2.61e-03 | 8.09e-04 | 4.65e-04 | 1.02e-03 | 1.29e-02 | 5.70e-04 |
| tri-graded | 0.5 | 154799 | 93166 | 710 | 1.66e-03 | 4.75e-04 | 5.33e-04 | 2.86e-04 | 5.92e-03 | 6.88e-04 |
| tri-uniform | 1 | 286749 | 48514 | 702 | 4.10e-03 | 1.29e-03 | 1.49e-04 | 2.96e-03 | 5.94e-02 | 2.41e-04 |
| tri-uniform | 0.714 | 499613 | 64846 | 1091 | 1.79e-03 | 5.85e-04 | 6.92e-05 | 1.55e-03 | 1.29e-02 | 1.81e-04 |
| tri-uniform | 0.5 | 924486 | 93166 | 2020 | 8.59e-04 | 2.83e-04 | 7.84e-05 | 8.70e-04 | 6.12e-03 | 1.93e-04 |
