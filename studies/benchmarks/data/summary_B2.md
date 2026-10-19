# B2 baseline summary

Cost from which each target is met and stays met on finer meshes (log-log interpolation along the h ladder; '—' = not reached; '<=' = already met by the coarsest run). Unknowns: 3 per Cartesian cell, nodes + edges for triangles. Time: time-stepping wall time on one uncontended n2-standard-2 core.

## B2-Au-TE

| metric | target | arm | unknowns | work | time [s] |
|---|---|---|---|---|---|
| q_ext_nrms | 0.02 | cart-conformal-all-graded | 1.65e+04 | 3.46e+08 | 22 |
| q_ext_nrms | 0.02 | cart-conformal-all-uniform | 4.82e+04 | 9.98e+08 | 28 |
| q_ext_nrms | 0.02 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.02 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.02 | tri-graded | <= 4.32e+03 | <= 5.45e+07 | <= 5 |
| q_ext_nrms | 0.02 | tri-uniform | <= 1.32e+04 | <= 1.67e+08 | <= 10 |
| q_ext_nrms | 0.01 | cart-conformal-all-graded | 4.58e+04 | 2.02e+09 | 86 |
| q_ext_nrms | 0.01 | cart-conformal-all-uniform | 1.7e+05 | 7.4e+09 | 193 |
| q_ext_nrms | 0.01 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.01 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.01 | tri-graded | 4.84e+03 | 6.47e+07 | 5 |
| q_ext_nrms | 0.01 | tri-uniform | <= 1.32e+04 | <= 1.67e+08 | <= 10 |
| q_ext_nrms | 0.005 | cart-conformal-all-graded | — | — | — |
| q_ext_nrms | 0.005 | cart-conformal-all-uniform | — | — | — |
| q_ext_nrms | 0.005 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.005 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.005 | tri-graded | 7.91e+03 | 1.41e+08 | 14 |
| q_ext_nrms | 0.005 | tri-uniform | 1.98e+04 | 2.69e+08 | 9 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-graded | 1.2e+04 | 1.75e+08 | 12 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-uniform | 2.96e+04 | 4.11e+08 | 13 |
| q_ext_lam_peak | 0.005 | cart-staircase-graded | 8.89e+04 | 6.01e+09 | 201 |
| q_ext_lam_peak | 0.005 | cart-staircase-uniform | 3.69e+05 | 2.42e+10 | 589 |
| q_ext_lam_peak | 0.005 | tri-graded | <= 4.32e+03 | <= 5.45e+07 | <= 5 |
| q_ext_lam_peak | 0.005 | tri-uniform | <= 1.32e+04 | <= 1.67e+08 | <= 10 |
| q_ext_lam_peak | 0.002 | cart-conformal-all-graded | 6.7e+04 | 3.8e+09 | 151 |
| q_ext_lam_peak | 0.002 | cart-conformal-all-uniform | 2.44e+05 | 1.29e+10 | 341 |
| q_ext_lam_peak | 0.002 | cart-staircase-graded | — | — | — |
| q_ext_lam_peak | 0.002 | cart-staircase-uniform | — | — | — |
| q_ext_lam_peak | 0.002 | tri-graded | 4.99e+03 | 6.75e+07 | 6 |
| q_ext_lam_peak | 0.002 | tri-uniform | 1.39e+04 | 1.77e+08 | 10 |
| q_ext_height | 0.02 | cart-conformal-all-graded | 1.64e+04 | 3.41e+08 | 21 |
| q_ext_height | 0.02 | cart-conformal-all-uniform | 5.88e+04 | 1.39e+09 | 36 |
| q_ext_height | 0.02 | cart-staircase-graded | 1.71e+04 | 3.68e+08 | 15 |
| q_ext_height | 0.02 | cart-staircase-uniform | 5.47e+04 | 1.23e+09 | 48 |
| q_ext_height | 0.02 | tri-graded | 4.35e+03 | 5.51e+07 | 5 |
| q_ext_height | 0.02 | tri-uniform | <= 1.32e+04 | <= 1.67e+08 | <= 10 |
| q_ext_height | 0.01 | cart-conformal-all-graded | 5.59e+04 | 2.82e+09 | 115 |
| q_ext_height | 0.01 | cart-conformal-all-uniform | 2.76e+05 | 1.56e+10 | 430 |
| q_ext_height | 0.01 | cart-staircase-graded | 4.47e+04 | 1.94e+09 | 67 |
| q_ext_height | 0.01 | cart-staircase-uniform | 2.01e+05 | 9.54e+09 | 213 |
| q_ext_height | 0.01 | tri-graded | 5.96e+03 | 8.94e+07 | 8 |
| q_ext_height | 0.01 | tri-uniform | 1.38e+04 | 1.76e+08 | 10 |
| q_ext_fwhm | 0.05 | cart-conformal-all-graded | 8.5e+03 | 8.04e+07 | 8 |
| q_ext_fwhm | 0.05 | cart-conformal-all-uniform | 1.71e+04 | 1.63e+08 | 6 |
| q_ext_fwhm | 0.05 | cart-staircase-graded | — | — | — |
| q_ext_fwhm | 0.05 | cart-staircase-uniform | — | — | — |
| q_ext_fwhm | 0.05 | tri-graded | <= 4.32e+03 | <= 5.45e+07 | <= 5 |
| q_ext_fwhm | 0.05 | tri-uniform | <= 1.32e+04 | <= 1.67e+08 | <= 10 |
| q_ext_fwhm | 0.02 | cart-conformal-all-graded | 5.56e+04 | 2.8e+09 | 114 |
| q_ext_fwhm | 0.02 | cart-conformal-all-uniform | 2.35e+05 | 1.22e+10 | 319 |
| q_ext_fwhm | 0.02 | cart-staircase-graded | — | — | — |
| q_ext_fwhm | 0.02 | cart-staircase-uniform | — | — | — |
| q_ext_fwhm | 0.02 | tri-graded | <= 4.32e+03 | <= 5.45e+07 | <= 5 |
| q_ext_fwhm | 0.02 | tri-uniform | <= 1.32e+04 | <= 1.67e+08 | <= 10 |

Per-run records:

| arm | h [nm] | unknowns | steps | run [s] | Q_ext NRMS | lambda_peak err | height err | FWHM err | enh peak err | reflection |
|---|---|---|---|---|---|---|---|---|---|---|
| cart-conformal-all-graded | 4 | 8112 | 8947 | 7 | 2.30e-02 | 4.34e-03 | 3.94e-02 | 5.89e-02 | — | 1.47e-02 |
| cart-conformal-all-graded | 2.86 | 10800 | 12527 | 11 | 1.96e-02 | 6.41e-03 | 1.59e-02 | 2.17e-02 | — | 1.20e-02 |
| cart-conformal-all-graded | 2 | 13872 | 17895 | 13 | 2.86e-02 | 3.57e-03 | 2.33e-02 | 2.91e-02 | — | 5.60e-03 |
| cart-conformal-all-graded | 1.43 | 20172 | 25070 | 41 | 1.33e-02 | 3.16e-03 | 1.65e-02 | 4.09e-02 | — | 1.64e-03 |
| cart-conformal-all-graded | 1 | 33708 | 35848 | 54 | 1.43e-02 | 1.01e-03 | 1.53e-02 | 1.61e-02 | — | 1.46e-03 |
| cart-conformal-all-graded | 0.714 | 55488 | 50209 | 114 | 8.00e-03 | 2.27e-03 | 1.00e-02 | 2.00e-02 | — | 2.46e-03 |
| cart-conformal-all-graded | 0.5 | 97200 | 71744 | 261 | 5.76e-03 | 1.56e-03 | 7.62e-03 | 1.51e-02 | — | 5.31e-04 |
| cart-conformal-all-uniform | 4 | 15552 | 8947 | 6 | 2.29e-02 | 3.89e-03 | 4.26e-02 | 5.97e-02 | — | 7.08e-04 |
| cart-conformal-all-uniform | 2.86 | 26508 | 12527 | 10 | 1.89e-02 | 6.03e-03 | 2.00e-02 | 2.23e-02 | — | 1.77e-03 |
| cart-conformal-all-uniform | 2 | 38988 | 17895 | 22 | 2.84e-02 | 3.13e-03 | 2.65e-02 | 3.04e-02 | — | 5.56e-04 |
| cart-conformal-all-uniform | 1.43 | 63948 | 25070 | 40 | 1.25e-02 | 2.82e-03 | 1.89e-02 | 4.20e-02 | — | 8.33e-04 |
| cart-conformal-all-uniform | 1 | 120000 | 35848 | 115 | 1.42e-02 | 7.83e-04 | 1.67e-02 | 1.69e-02 | — | 5.81e-04 |
| cart-conformal-all-uniform | 0.714 | 221952 | 50209 | 286 | 7.69e-03 | 2.11e-03 | 1.09e-02 | 2.05e-02 | — | 5.91e-04 |
| cart-conformal-all-uniform | 0.5 | 433200 | 71744 | 993 | 5.52e-03 | 1.44e-03 | 8.33e-03 | 1.54e-02 | — | 6.28e-04 |
| cart-staircase-graded | 4 | 8112 | 8947 | 4 | 1.55e-01 | 2.56e-03 | 9.85e-02 | — | — | 1.38e-02 |
| cart-staircase-graded | 2.86 | 10800 | 12527 | 11 | 1.31e-01 | 1.41e-02 | 6.87e-02 | — | — | 1.26e-02 |
| cart-staircase-graded | 2 | 13872 | 17895 | 11 | 1.04e-01 | 2.61e-02 | 3.84e-02 | — | — | 5.66e-03 |
| cart-staircase-graded | 1.43 | 20172 | 25054 | 19 | 7.65e-02 | 1.82e-02 | 1.18e-02 | — | — | 1.70e-03 |
| cart-staircase-graded | 1 | 33708 | 35791 | 34 | 5.34e-02 | 1.01e-02 | 1.24e-02 | 1.60e-01 | — | 1.46e-03 |
| cart-staircase-graded | 0.714 | 55488 | 50108 | 113 | 3.81e-02 | 6.66e-03 | 8.51e-03 | 1.11e-01 | — | 2.47e-03 |
| cart-staircase-graded | 0.5 | 97200 | 71583 | 225 | 2.75e-02 | 4.74e-03 | 5.78e-03 | 7.82e-02 | — | 5.37e-04 |
| cart-staircase-uniform | 4 | 15552 | 8947 | 5 | 1.53e-01 | 2.99e-03 | 1.02e-01 | — | — | 8.10e-04 |
| cart-staircase-uniform | 2.86 | 26508 | 12527 | 17 | 1.28e-01 | 1.35e-02 | 7.28e-02 | — | — | 1.89e-03 |
| cart-staircase-uniform | 2 | 38988 | 17895 | 20 | 1.01e-01 | 2.55e-02 | 4.15e-02 | — | — | 5.95e-04 |
| cart-staircase-uniform | 1.43 | 63948 | 25054 | 72 | 7.48e-02 | 1.78e-02 | 1.43e-02 | — | — | 8.44e-04 |
| cart-staircase-uniform | 1 | 120000 | 35791 | 151 | 5.25e-02 | 9.85e-03 | 1.37e-02 | 1.61e-01 | — | 5.63e-04 |
| cart-staircase-uniform | 0.714 | 221952 | 50108 | 227 | 3.75e-02 | 6.48e-03 | 9.41e-03 | 1.11e-01 | — | 5.74e-04 |
| cart-staircase-uniform | 0.5 | 433200 | 71583 | 798 | 2.71e-02 | 4.60e-03 | 6.49e-03 | 7.86e-02 | — | 6.31e-04 |
| tri-graded | 4 | 4321 | 12619 | 5 | 1.28e-02 | 2.66e-03 | 2.03e-02 | 1.41e-02 | — | 1.92e-02 |
| tri-graded | 2.86 | 5233 | 13875 | 6 | 8.49e-03 | 1.82e-03 | 1.34e-02 | 8.30e-03 | — | 1.82e-02 |
| tri-graded | 2 | 8014 | 17983 | 15 | 4.91e-03 | 1.26e-03 | 5.07e-03 | 3.19e-03 | — | 7.80e-03 |
| tri-graded | 1.43 | 12842 | 26355 | 19 | 4.05e-03 | 9.40e-04 | 5.14e-03 | 1.88e-03 | — | 4.84e-03 |
| tri-graded | 1 | 22003 | 38811 | 36 | 2.17e-03 | 5.92e-04 | 2.76e-03 | 1.90e-03 | — | 4.58e-03 |
| tri-graded | 0.714 | 37723 | 51877 | 120 | 1.67e-03 | 4.24e-04 | 2.23e-03 | 1.27e-03 | — | 3.37e-03 |
| tri-graded | 0.5 | 69018 | 74533 | 192 | 1.14e-03 | 2.76e-04 | 1.70e-03 | 9.09e-04 | — | 3.73e-03 |
| tri-uniform | 4 | 13195 | 12619 | 10 | 7.84e-03 | 2.12e-03 | 1.05e-02 | 8.98e-03 | — | 4.58e-03 |
| tri-uniform | 2.86 | 22523 | 13875 | 9 | 4.34e-03 | 1.15e-03 | 5.88e-03 | 3.84e-03 | — | 3.36e-03 |
| tri-uniform | 2 | 40958 | 17982 | 17 | 2.38e-03 | 6.14e-04 | 3.39e-03 | 2.12e-03 | — | 3.73e-03 |
| tri-uniform | 1.43 | 73745 | 26355 | 67 | 1.22e-03 | 3.22e-04 | 1.69e-03 | 1.21e-03 | — | 3.46e-03 |
| tri-uniform | 1 | 140731 | 38811 | 89 | 5.91e-04 | 1.53e-04 | 8.11e-04 | 5.51e-04 | — | 3.63e-03 |
| tri-uniform | 0.714 | 263001 | 51877 | 288 | 2.62e-04 | 6.92e-05 | 3.58e-04 | 2.44e-04 | — | 3.63e-03 |
| tri-uniform | 0.5 | 516254 | 74533 | 831 | 1.40e-04 | 3.69e-05 | 1.90e-04 | 1.34e-04 | — | 3.66e-03 |

## B2-Au-TM

| metric | target | arm | unknowns | work | time [s] |
|---|---|---|---|---|---|
| q_ext_nrms | 0.02 | cart-conformal-all-graded | <= 8.11e+03 | <= 7.26e+07 | <= 5 |
| q_ext_nrms | 0.02 | cart-conformal-all-uniform | <= 1.56e+04 | <= 1.39e+08 | <= 10 |
| q_ext_nrms | 0.02 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.02 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.02 | tri-graded | 5.07e+03 | 6.96e+07 | 5 |
| q_ext_nrms | 0.02 | tri-uniform | <= 1.31e+04 | <= 1.65e+08 | <= 6 |
| q_ext_nrms | 0.01 | cart-conformal-all-graded | <= 8.11e+03 | <= 7.26e+07 | <= 5 |
| q_ext_nrms | 0.01 | cart-conformal-all-uniform | <= 1.56e+04 | <= 1.39e+08 | <= 10 |
| q_ext_nrms | 0.01 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.01 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.01 | tri-graded | 6.9e+03 | 1.14e+08 | 6 |
| q_ext_nrms | 0.01 | tri-uniform | <= 1.31e+04 | <= 1.65e+08 | <= 6 |
| q_ext_nrms | 0.005 | cart-conformal-all-graded | 1.26e+04 | 1.97e+08 | 14 |
| q_ext_nrms | 0.005 | cart-conformal-all-uniform | <= 1.56e+04 | <= 1.39e+08 | <= 10 |
| q_ext_nrms | 0.005 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.005 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.005 | tri-graded | 8.65e+03 | 1.66e+08 | 8 |
| q_ext_nrms | 0.005 | tri-uniform | <= 1.31e+04 | <= 1.65e+08 | <= 6 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-graded | <= 8.11e+03 | <= 7.26e+07 | <= 5 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-uniform | <= 1.56e+04 | <= 1.39e+08 | <= 10 |
| q_ext_lam_peak | 0.005 | cart-staircase-graded | <= 8.11e+03 | <= 7.26e+07 | <= 5 |
| q_ext_lam_peak | 0.005 | cart-staircase-uniform | <= 1.56e+04 | <= 1.39e+08 | <= 9 |
| q_ext_lam_peak | 0.005 | tri-graded | <= 4.27e+03 | <= 5.39e+07 | <= 4 |
| q_ext_lam_peak | 0.005 | tri-uniform | <= 1.31e+04 | <= 1.65e+08 | <= 6 |
| q_ext_lam_peak | 0.002 | cart-conformal-all-graded | <= 8.11e+03 | <= 7.26e+07 | <= 5 |
| q_ext_lam_peak | 0.002 | cart-conformal-all-uniform | <= 1.56e+04 | <= 1.39e+08 | <= 10 |
| q_ext_lam_peak | 0.002 | cart-staircase-graded | <= 8.11e+03 | <= 7.26e+07 | <= 5 |
| q_ext_lam_peak | 0.002 | cart-staircase-uniform | <= 1.56e+04 | <= 1.39e+08 | <= 9 |
| q_ext_lam_peak | 0.002 | tri-graded | <= 4.27e+03 | <= 5.39e+07 | <= 4 |
| q_ext_lam_peak | 0.002 | tri-uniform | <= 1.31e+04 | <= 1.65e+08 | <= 6 |
| q_ext_height | 0.02 | cart-conformal-all-graded | <= 8.11e+03 | <= 7.26e+07 | <= 5 |
| q_ext_height | 0.02 | cart-conformal-all-uniform | <= 1.56e+04 | <= 1.39e+08 | <= 10 |
| q_ext_height | 0.02 | cart-staircase-graded | — | — | — |
| q_ext_height | 0.02 | cart-staircase-uniform | — | — | — |
| q_ext_height | 0.02 | tri-graded | 4.92e+03 | 6.65e+07 | 5 |
| q_ext_height | 0.02 | tri-uniform | <= 1.31e+04 | <= 1.65e+08 | <= 6 |
| q_ext_height | 0.01 | cart-conformal-all-graded | <= 8.11e+03 | <= 7.26e+07 | <= 5 |
| q_ext_height | 0.01 | cart-conformal-all-uniform | <= 1.56e+04 | <= 1.39e+08 | <= 10 |
| q_ext_height | 0.01 | cart-staircase-graded | — | — | — |
| q_ext_height | 0.01 | cart-staircase-uniform | — | — | — |
| q_ext_height | 0.01 | tri-graded | 7.72e+03 | 1.36e+08 | 7 |
| q_ext_height | 0.01 | tri-uniform | <= 1.31e+04 | <= 1.65e+08 | <= 6 |
| q_ext_fwhm | 0.05 | cart-conformal-all-graded | — | — | — |
| q_ext_fwhm | 0.05 | cart-conformal-all-uniform | — | — | — |
| q_ext_fwhm | 0.05 | cart-staircase-graded | — | — | — |
| q_ext_fwhm | 0.05 | cart-staircase-uniform | — | — | — |
| q_ext_fwhm | 0.05 | tri-graded | — | — | — |
| q_ext_fwhm | 0.05 | tri-uniform | — | — | — |
| q_ext_fwhm | 0.02 | cart-conformal-all-graded | — | — | — |
| q_ext_fwhm | 0.02 | cart-conformal-all-uniform | — | — | — |
| q_ext_fwhm | 0.02 | cart-staircase-graded | — | — | — |
| q_ext_fwhm | 0.02 | cart-staircase-uniform | — | — | — |
| q_ext_fwhm | 0.02 | tri-graded | — | — | — |
| q_ext_fwhm | 0.02 | tri-uniform | — | — | — |

Per-run records:

| arm | h [nm] | unknowns | steps | run [s] | Q_ext NRMS | lambda_peak err | height err | FWHM err | enh peak err | reflection |
|---|---|---|---|---|---|---|---|---|---|---|
| cart-conformal-all-graded | 4 | 8112 | 8947 | 5 | 6.62e-03 | 0.00e+00 | 5.59e-03 | — | — | 1.91e-02 |
| cart-conformal-all-graded | 2.86 | 10800 | 12527 | 13 | 8.81e-03 | 0.00e+00 | 8.27e-03 | — | — | 2.79e-02 |
| cart-conformal-all-graded | 2 | 13872 | 17895 | 15 | 3.54e-03 | 0.00e+00 | 2.73e-03 | — | — | 2.51e-03 |
| cart-conformal-all-graded | 1.43 | 20172 | 25070 | 26 | 1.10e-03 | 0.00e+00 | 4.72e-04 | — | — | 5.94e-03 |
| cart-conformal-all-graded | 1 | 33708 | 35848 | 81 | 2.84e-04 | 0.00e+00 | 1.26e-04 | — | — | 2.40e-03 |
| cart-conformal-all-graded | 0.714 | 55488 | 50209 | 102 | 1.42e-04 | 0.00e+00 | 1.93e-04 | — | — | 2.09e-03 |
| cart-conformal-all-graded | 0.5 | 97200 | 71744 | 398 | 1.20e-04 | 0.00e+00 | 1.80e-04 | — | — | 9.66e-04 |
| cart-conformal-all-uniform | 4 | 15552 | 8947 | 10 | 3.57e-04 | 0.00e+00 | 7.38e-04 | — | — | 3.12e-03 |
| cart-conformal-all-uniform | 2.86 | 26508 | 12527 | 15 | 3.83e-04 | 0.00e+00 | 6.12e-04 | — | — | 1.63e-03 |
| cart-conformal-all-uniform | 2 | 38988 | 17895 | 19 | 4.67e-05 | 0.00e+00 | 7.25e-05 | — | — | 1.24e-03 |
| cart-conformal-all-uniform | 1.43 | 63948 | 25070 | 42 | 8.20e-05 | 0.00e+00 | 1.14e-04 | — | — | 7.25e-04 |
| cart-conformal-all-uniform | 1 | 120000 | 35848 | 181 | 1.06e-05 | 0.00e+00 | 3.57e-06 | — | — | 8.54e-04 |
| cart-conformal-all-uniform | 0.714 | 221952 | 50209 | 371 | 2.23e-05 | 0.00e+00 | 1.24e-05 | — | — | 9.42e-04 |
| cart-conformal-all-uniform | 0.5 | 433200 | 71744 | 892 | 1.44e-05 | 0.00e+00 | 1.53e-05 | — | — | 1.00e-03 |
| cart-staircase-graded | 4 | 8112 | 8947 | 5 | 1.13e-01 | 0.00e+00 | 1.40e-01 | — | — | 1.91e-02 |
| cart-staircase-graded | 2.86 | 10800 | 12527 | 7 | 1.09e-01 | 0.00e+00 | 1.35e-01 | — | — | 2.79e-02 |
| cart-staircase-graded | 2 | 13872 | 17895 | 11 | 9.31e-02 | 0.00e+00 | 1.15e-01 | — | — | 2.51e-03 |
| cart-staircase-graded | 1.43 | 20172 | 25054 | 23 | 7.88e-02 | 0.00e+00 | 9.66e-02 | — | — | 5.93e-03 |
| cart-staircase-graded | 1 | 33708 | 35791 | 37 | 5.44e-02 | 0.00e+00 | 6.65e-02 | — | — | 2.40e-03 |
| cart-staircase-graded | 0.714 | 55488 | 50108 | 79 | 4.01e-02 | 0.00e+00 | 4.90e-02 | — | — | 2.09e-03 |
| cart-staircase-graded | 0.5 | 97200 | 71583 | 233 | 2.96e-02 | 0.00e+00 | 3.60e-02 | — | — | 9.66e-04 |
| cart-staircase-uniform | 4 | 15552 | 8947 | 9 | 1.20e-01 | 0.00e+00 | 1.46e-01 | — | — | 3.12e-03 |
| cart-staircase-uniform | 2.86 | 26508 | 12527 | 19 | 1.19e-01 | 0.00e+00 | 1.45e-01 | — | — | 1.63e-03 |
| cart-staircase-uniform | 2 | 38988 | 17895 | 24 | 9.66e-02 | 0.00e+00 | 1.17e-01 | — | — | 1.24e-03 |
| cart-staircase-uniform | 1.43 | 63948 | 25054 | 51 | 7.98e-02 | 0.00e+00 | 9.71e-02 | — | — | 7.26e-04 |
| cart-staircase-uniform | 1 | 120000 | 35791 | 103 | 5.45e-02 | 0.00e+00 | 6.64e-02 | — | — | 8.55e-04 |
| cart-staircase-uniform | 0.714 | 221952 | 50108 | 259 | 4.01e-02 | 0.00e+00 | 4.88e-02 | — | — | 9.43e-04 |
| cart-staircase-uniform | 0.5 | 433200 | 71583 | 854 | 2.94e-02 | 0.00e+00 | 3.58e-02 | — | — | 1.00e-03 |
| tri-graded | 4 | 4269 | 12619 | 4 | 3.12e-02 | 0.00e+00 | 2.98e-02 | — | — | 1.62e-02 |
| tri-graded | 2.86 | 5181 | 13875 | 5 | 1.89e-02 | 0.00e+00 | 1.72e-02 | — | — | 1.28e-02 |
| tri-graded | 2 | 7943 | 17968 | 7 | 7.33e-03 | 0.00e+00 | 9.62e-03 | — | — | 4.89e-03 |
| tri-graded | 1.43 | 12743 | 26355 | 13 | 8.71e-04 | 0.00e+00 | 1.09e-03 | — | — | 3.21e-03 |
| tri-graded | 1 | 21861 | 38811 | 23 | 1.48e-03 | 0.00e+00 | 1.96e-03 | — | — | 2.58e-03 |
| tri-graded | 0.714 | 37525 | 51868 | 75 | 5.75e-04 | 0.00e+00 | 7.10e-04 | — | — | 2.40e-03 |
| tri-graded | 0.5 | 68735 | 74528 | 89 | 2.97e-04 | 0.00e+00 | 3.06e-04 | — | — | 2.20e-03 |
| tri-uniform | 4 | 13053 | 12619 | 6 | 2.24e-03 | 0.00e+00 | 2.99e-03 | — | — | 2.58e-03 |
| tri-uniform | 2.86 | 22325 | 13875 | 10 | 9.04e-04 | 0.00e+00 | 1.17e-03 | — | — | 2.39e-03 |
| tri-uniform | 2 | 40675 | 17968 | 15 | 1.47e-04 | 0.00e+00 | 8.17e-05 | — | — | 2.20e-03 |
| tri-uniform | 1.43 | 73349 | 26355 | 33 | 2.90e-04 | 0.00e+00 | 3.85e-04 | — | — | 2.35e-03 |
| tri-uniform | 1 | 140165 | 38811 | 106 | 1.15e-04 | 0.00e+00 | 1.53e-04 | — | — | 2.34e-03 |
| tri-uniform | 0.714 | 262209 | 51868 | 316 | 4.77e-05 | 0.00e+00 | 6.86e-05 | — | — | 2.34e-03 |
| tri-uniform | 0.5 | 515123 | 74528 | 601 | 3.70e-05 | 0.00e+00 | 4.90e-05 | — | — | 2.35e-03 |

## B2-Ag-TE

| metric | target | arm | unknowns | work | time [s] |
|---|---|---|---|---|---|
| q_ext_nrms | 0.02 | cart-conformal-all-graded | 1.61e+04 | 3.3e+08 | 20 |
| q_ext_nrms | 0.02 | cart-conformal-all-uniform | 4.65e+04 | 9.42e+08 | 31 |
| q_ext_nrms | 0.02 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.02 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.02 | tri-graded | <= 3.88e+03 | <= 4.89e+07 | <= 4 |
| q_ext_nrms | 0.02 | tri-uniform | <= 1.32e+04 | <= 1.67e+08 | <= 6 |
| q_ext_nrms | 0.01 | cart-conformal-all-graded | 4.23e+04 | 1.77e+09 | 78 |
| q_ext_nrms | 0.01 | cart-conformal-all-uniform | 1.54e+05 | 6.32e+09 | 202 |
| q_ext_nrms | 0.01 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.01 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.01 | tri-graded | 4.8e+03 | 6.47e+07 | 5 |
| q_ext_nrms | 0.01 | tri-uniform | <= 1.32e+04 | <= 1.67e+08 | <= 6 |
| q_ext_nrms | 0.005 | cart-conformal-all-graded | — | — | — |
| q_ext_nrms | 0.005 | cart-conformal-all-uniform | — | — | — |
| q_ext_nrms | 0.005 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.005 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.005 | tri-graded | 7.52e+03 | 1.31e+08 | 8 |
| q_ext_nrms | 0.005 | tri-uniform | 1.87e+04 | 2.51e+08 | 8 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-graded | <= 8.11e+03 | <= 7.26e+07 | <= 5 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-uniform | <= 1.83e+04 | <= 1.64e+08 | <= 7 |
| q_ext_lam_peak | 0.005 | cart-staircase-graded | 2.95e+04 | 9.65e+08 | 30 |
| q_ext_lam_peak | 0.005 | cart-staircase-uniform | 9.85e+04 | 3.15e+09 | 76 |
| q_ext_lam_peak | 0.005 | tri-graded | <= 3.88e+03 | <= 4.89e+07 | <= 4 |
| q_ext_lam_peak | 0.005 | tri-uniform | <= 1.32e+04 | <= 1.67e+08 | <= 6 |
| q_ext_lam_peak | 0.002 | cart-conformal-all-graded | 1.21e+04 | 1.8e+08 | 11 |
| q_ext_lam_peak | 0.002 | cart-conformal-all-uniform | 3.02e+04 | 4.27e+08 | 23 |
| q_ext_lam_peak | 0.002 | cart-staircase-graded | — | — | — |
| q_ext_lam_peak | 0.002 | cart-staircase-uniform | 4.31e+05 | 3.08e+10 | 801 |
| q_ext_lam_peak | 0.002 | tri-graded | <= 3.88e+03 | <= 4.89e+07 | <= 4 |
| q_ext_lam_peak | 0.002 | tri-uniform | <= 1.32e+04 | <= 1.67e+08 | <= 6 |
| q_ext_height | 0.02 | cart-conformal-all-graded | 1.77e+04 | 3.95e+08 | 27 |
| q_ext_height | 0.02 | cart-conformal-all-uniform | 7.11e+04 | 1.9e+09 | 54 |
| q_ext_height | 0.02 | cart-staircase-graded | 1.67e+04 | 3.53e+08 | 17 |
| q_ext_height | 0.02 | cart-staircase-uniform | 5.37e+04 | 1.19e+09 | 37 |
| q_ext_height | 0.02 | tri-graded | 4.7e+03 | 6.3e+07 | 5 |
| q_ext_height | 0.02 | tri-uniform | <= 1.32e+04 | <= 1.67e+08 | <= 6 |
| q_ext_height | 0.01 | cart-conformal-all-graded | 7.38e+04 | 4.44e+09 | 204 |
| q_ext_height | 0.01 | cart-conformal-all-uniform | 3.81e+05 | 2.55e+10 | 728 |
| q_ext_height | 0.01 | cart-staircase-graded | 3.78e+04 | 1.46e+09 | 42 |
| q_ext_height | 0.01 | cart-staircase-uniform | 1.66e+05 | 7.07e+09 | 167 |
| q_ext_height | 0.01 | tri-graded | 6.44e+03 | 1.02e+08 | 7 |
| q_ext_height | 0.01 | tri-uniform | 1.5e+04 | 1.94e+08 | 7 |
| q_ext_fwhm | 0.05 | cart-conformal-all-graded | 9.43e+03 | 1.01e+08 | 7 |
| q_ext_fwhm | 0.05 | cart-conformal-all-uniform | 2.17e+04 | 2.27e+08 | 12 |
| q_ext_fwhm | 0.05 | cart-staircase-graded | — | — | — |
| q_ext_fwhm | 0.05 | cart-staircase-uniform | — | — | — |
| q_ext_fwhm | 0.05 | tri-graded | <= 3.88e+03 | <= 4.89e+07 | <= 4 |
| q_ext_fwhm | 0.05 | tri-uniform | <= 1.32e+04 | <= 1.67e+08 | <= 6 |
| q_ext_fwhm | 0.02 | cart-conformal-all-graded | 7.62e+04 | 4.69e+09 | 210 |
| q_ext_fwhm | 0.02 | cart-conformal-all-uniform | 3.18e+05 | 1.94e+10 | 554 |
| q_ext_fwhm | 0.02 | cart-staircase-graded | — | — | — |
| q_ext_fwhm | 0.02 | cart-staircase-uniform | — | — | — |
| q_ext_fwhm | 0.02 | tri-graded | <= 3.88e+03 | <= 4.89e+07 | <= 4 |
| q_ext_fwhm | 0.02 | tri-uniform | <= 1.32e+04 | <= 1.67e+08 | <= 6 |

Per-run records:

| arm | h [nm] | unknowns | steps | run [s] | Q_ext NRMS | lambda_peak err | height err | FWHM err | enh peak err | reflection |
|---|---|---|---|---|---|---|---|---|---|---|
| cart-conformal-all-graded | 4 | 8112 | 8947 | 5 | 4.31e-02 | 1.73e-03 | 4.00e-02 | 6.28e-02 | — | 1.19e-02 |
| cart-conformal-all-graded | 2.86 | 10800 | 12527 | 9 | 1.79e-02 | 2.92e-03 | 1.98e-02 | 4.07e-02 | — | 1.39e-02 |
| cart-conformal-all-graded | 2 | 13872 | 17906 | 12 | 2.74e-02 | 1.30e-03 | 2.53e-02 | 3.20e-02 | — | 6.74e-03 |
| cart-conformal-all-graded | 1.43 | 20172 | 25088 | 40 | 1.24e-02 | 1.39e-03 | 1.76e-02 | 4.93e-02 | — | 1.71e-03 |
| cart-conformal-all-graded | 1 | 33708 | 35859 | 42 | 1.25e-02 | 4.85e-04 | 1.63e-02 | 1.71e-02 | — | 1.49e-03 |
| cart-conformal-all-graded | 0.714 | 55488 | 50216 | 164 | 7.62e-03 | 9.27e-04 | 1.16e-02 | 2.39e-02 | — | 2.74e-03 |
| cart-conformal-all-graded | 0.5 | 97200 | 71748 | 252 | 5.38e-03 | 6.19e-04 | 8.67e-03 | 1.74e-02 | — | 3.33e-04 |
| cart-conformal-all-uniform | 4 | 18252 | 8968 | 7 | 4.27e-02 | 1.50e-03 | 4.39e-02 | 6.12e-02 | — | 6.76e-04 |
| cart-conformal-all-uniform | 2.86 | 26508 | 12556 | 21 | 1.69e-02 | 2.72e-03 | 2.46e-02 | 3.95e-02 | — | 1.99e-03 |
| cart-conformal-all-uniform | 2 | 38988 | 17938 | 25 | 2.71e-02 | 1.09e-03 | 2.91e-02 | 3.09e-02 | — | 9.06e-04 |
| cart-conformal-all-uniform | 1.43 | 63948 | 25114 | 44 | 1.16e-02 | 1.23e-03 | 2.05e-02 | 4.84e-02 | — | 4.29e-04 |
| cart-conformal-all-uniform | 1 | 120000 | 35878 | 148 | 1.23e-02 | 3.78e-04 | 1.79e-02 | 1.67e-02 | — | 1.74e-04 |
| cart-conformal-all-uniform | 0.714 | 221952 | 50230 | 321 | 7.32e-03 | 8.50e-04 | 1.26e-02 | 2.38e-02 | — | 2.19e-04 |
| cart-conformal-all-uniform | 0.5 | 433200 | 71758 | 883 | 5.16e-03 | 5.63e-04 | 9.47e-03 | 1.73e-02 | — | 1.99e-04 |
| cart-staircase-graded | 4 | 8112 | 8947 | 4 | 1.38e-01 | 6.25e-04 | 8.26e-02 | 5.88e-01 | — | 1.17e-02 |
| cart-staircase-graded | 2.86 | 10800 | 12527 | 7 | 1.17e-01 | 6.19e-03 | 5.56e-02 | 4.88e-01 | — | 1.46e-02 |
| cart-staircase-graded | 2 | 13872 | 17895 | 15 | 9.32e-02 | 9.29e-03 | 3.65e-02 | 3.46e-01 | — | 6.06e-03 |
| cart-staircase-graded | 1.43 | 20172 | 25054 | 19 | 6.80e-02 | 7.31e-03 | 1.08e-02 | 2.35e-01 | — | 1.75e-03 |
| cart-staircase-graded | 1 | 33708 | 35791 | 35 | 4.75e-02 | 4.39e-03 | 1.11e-02 | 1.60e-01 | — | 1.49e-03 |
| cart-staircase-graded | 0.714 | 55488 | 50108 | 83 | 3.37e-02 | 2.94e-03 | 7.10e-03 | 1.12e-01 | — | 2.75e-03 |
| cart-staircase-graded | 0.5 | 97200 | 71583 | 182 | 2.43e-02 | 2.06e-03 | 4.76e-03 | 7.94e-02 | — | 3.40e-04 |
| cart-staircase-uniform | 4 | 18252 | 8947 | 6 | 1.35e-01 | 4.17e-04 | 8.69e-02 | 5.86e-01 | — | 8.14e-04 |
| cart-staircase-uniform | 2.86 | 26508 | 12527 | 10 | 1.14e-01 | 5.94e-03 | 6.03e-02 | 4.85e-01 | — | 2.15e-03 |
| cart-staircase-uniform | 2 | 38988 | 17895 | 31 | 9.11e-02 | 9.02e-03 | 4.03e-02 | 3.44e-01 | — | 8.95e-04 |
| cart-staircase-uniform | 1.43 | 63948 | 25054 | 40 | 6.64e-02 | 7.10e-03 | 1.36e-02 | 2.34e-01 | — | 4.38e-04 |
| cart-staircase-uniform | 1 | 120000 | 35791 | 100 | 4.67e-02 | 4.26e-03 | 1.26e-02 | 1.60e-01 | — | 1.54e-04 |
| cart-staircase-uniform | 0.714 | 221952 | 50108 | 266 | 3.32e-02 | 2.85e-03 | 8.10e-03 | 1.11e-01 | — | 2.06e-04 |
| cart-staircase-uniform | 0.5 | 433200 | 71583 | 807 | 2.39e-02 | 2.00e-03 | 5.56e-03 | 7.92e-02 | — | 2.00e-04 |
| tri-graded | 4 | 3875 | 12619 | 4 | 1.55e-02 | 1.62e-03 | 2.93e-02 | 5.87e-03 | — | 2.88e-02 |
| tri-graded | 2.86 | 5229 | 13875 | 5 | 8.37e-03 | 9.76e-04 | 1.62e-02 | 2.50e-03 | — | 2.25e-02 |
| tri-graded | 2 | 8014 | 18149 | 9 | 4.57e-03 | 6.34e-04 | 6.01e-03 | 2.34e-03 | — | 7.60e-03 |
| tri-graded | 1.43 | 12842 | 26355 | 26 | 3.84e-03 | 4.63e-04 | 6.43e-03 | 1.94e-03 | — | 3.49e-03 |
| tri-graded | 1 | 22003 | 38811 | 37 | 1.96e-03 | 2.85e-04 | 2.97e-03 | 6.28e-04 | — | 2.18e-03 |
| tri-graded | 0.714 | 37723 | 51985 | 76 | 1.53e-03 | 2.04e-04 | 2.50e-03 | 5.69e-04 | — | 2.54e-03 |
| tri-graded | 0.5 | 69018 | 74590 | 338 | 1.04e-03 | 1.34e-04 | 1.83e-03 | 3.33e-04 | — | 2.97e-03 |
| tri-uniform | 4 | 13195 | 12619 | 6 | 7.29e-03 | 1.05e-03 | 1.15e-02 | 2.17e-03 | — | 2.16e-03 |
| tri-uniform | 2.86 | 22523 | 13875 | 8 | 4.09e-03 | 5.83e-04 | 6.49e-03 | 1.25e-03 | — | 2.53e-03 |
| tri-uniform | 2 | 40958 | 18660 | 16 | 2.23e-03 | 3.13e-04 | 3.67e-03 | 6.30e-04 | — | 2.97e-03 |
| tri-uniform | 1.43 | 73745 | 26355 | 33 | 1.14e-03 | 1.65e-04 | 1.79e-03 | 3.02e-04 | — | 2.58e-03 |
| tri-uniform | 1 | 140731 | 38811 | 165 | 5.53e-04 | 7.88e-05 | 8.81e-04 | 1.53e-04 | — | 2.71e-03 |
| tri-uniform | 0.714 | 263001 | 52550 | 330 | 2.47e-04 | 3.60e-05 | 3.88e-04 | 6.52e-05 | — | 2.72e-03 |
| tri-uniform | 0.5 | 516254 | 75126 | 1097 | 1.32e-04 | 1.91e-05 | 2.05e-04 | 3.53e-05 | — | 2.73e-03 |

## B2-Ag-TM

| metric | target | arm | unknowns | work | time [s] |
|---|---|---|---|---|---|
| q_ext_nrms | 0.02 | cart-conformal-all-graded | <= 8.11e+03 | <= 7.26e+07 | <= 8 |
| q_ext_nrms | 0.02 | cart-conformal-all-uniform | <= 1.83e+04 | <= 1.64e+08 | <= 6 |
| q_ext_nrms | 0.02 | cart-staircase-graded | 9.02e+04 | 6.16e+09 | 168 |
| q_ext_nrms | 0.02 | cart-staircase-uniform | 3.93e+05 | 2.67e+10 | 811 |
| q_ext_nrms | 0.02 | tri-graded | 4.11e+03 | 5.3e+07 | 6 |
| q_ext_nrms | 0.02 | tri-uniform | <= 1.31e+04 | <= 1.65e+08 | <= 6 |
| q_ext_nrms | 0.01 | cart-conformal-all-graded | <= 8.11e+03 | <= 7.26e+07 | <= 8 |
| q_ext_nrms | 0.01 | cart-conformal-all-uniform | <= 1.83e+04 | <= 1.64e+08 | <= 6 |
| q_ext_nrms | 0.01 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.01 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.01 | tri-graded | 5.34e+03 | 7.55e+07 | 5 |
| q_ext_nrms | 0.01 | tri-uniform | <= 1.31e+04 | <= 1.65e+08 | <= 6 |
| q_ext_nrms | 0.005 | cart-conformal-all-graded | <= 8.11e+03 | <= 7.26e+07 | <= 8 |
| q_ext_nrms | 0.005 | cart-conformal-all-uniform | <= 1.83e+04 | <= 1.64e+08 | <= 6 |
| q_ext_nrms | 0.005 | cart-staircase-graded | — | — | — |
| q_ext_nrms | 0.005 | cart-staircase-uniform | — | — | — |
| q_ext_nrms | 0.005 | tri-graded | 7.61e+03 | 1.34e+08 | 10 |
| q_ext_nrms | 0.005 | tri-uniform | <= 1.31e+04 | <= 1.65e+08 | <= 6 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-graded | <= 8.11e+03 | <= 7.26e+07 | <= 8 |
| q_ext_lam_peak | 0.005 | cart-conformal-all-uniform | <= 1.83e+04 | <= 1.64e+08 | <= 6 |
| q_ext_lam_peak | 0.005 | cart-staircase-graded | <= 8.11e+03 | <= 7.26e+07 | <= 5 |
| q_ext_lam_peak | 0.005 | cart-staircase-uniform | <= 1.83e+04 | <= 1.63e+08 | <= 8 |
| q_ext_lam_peak | 0.005 | tri-graded | <= 3.83e+03 | <= 4.84e+07 | <= 6 |
| q_ext_lam_peak | 0.005 | tri-uniform | <= 1.31e+04 | <= 1.65e+08 | <= 6 |
| q_ext_lam_peak | 0.002 | cart-conformal-all-graded | <= 8.11e+03 | <= 7.26e+07 | <= 8 |
| q_ext_lam_peak | 0.002 | cart-conformal-all-uniform | <= 1.83e+04 | <= 1.64e+08 | <= 6 |
| q_ext_lam_peak | 0.002 | cart-staircase-graded | <= 8.11e+03 | <= 7.26e+07 | <= 5 |
| q_ext_lam_peak | 0.002 | cart-staircase-uniform | <= 1.83e+04 | <= 1.63e+08 | <= 8 |
| q_ext_lam_peak | 0.002 | tri-graded | <= 3.83e+03 | <= 4.84e+07 | <= 6 |
| q_ext_lam_peak | 0.002 | tri-uniform | <= 1.31e+04 | <= 1.65e+08 | <= 6 |
| q_ext_height | 0.02 | cart-conformal-all-graded | <= 8.11e+03 | <= 7.26e+07 | <= 8 |
| q_ext_height | 0.02 | cart-conformal-all-uniform | <= 1.83e+04 | <= 1.64e+08 | <= 6 |
| q_ext_height | 0.02 | cart-staircase-graded | — | — | — |
| q_ext_height | 0.02 | cart-staircase-uniform | — | — | — |
| q_ext_height | 0.02 | tri-graded | <= 3.83e+03 | <= 4.84e+07 | <= 6 |
| q_ext_height | 0.02 | tri-uniform | <= 1.31e+04 | <= 1.65e+08 | <= 6 |
| q_ext_height | 0.01 | cart-conformal-all-graded | <= 8.11e+03 | <= 7.26e+07 | <= 8 |
| q_ext_height | 0.01 | cart-conformal-all-uniform | <= 1.83e+04 | <= 1.64e+08 | <= 6 |
| q_ext_height | 0.01 | cart-staircase-graded | — | — | — |
| q_ext_height | 0.01 | cart-staircase-uniform | — | — | — |
| q_ext_height | 0.01 | tri-graded | 4.43e+03 | 5.85e+07 | 6 |
| q_ext_height | 0.01 | tri-uniform | <= 1.31e+04 | <= 1.65e+08 | <= 6 |
| q_ext_fwhm | 0.05 | cart-conformal-all-graded | — | — | — |
| q_ext_fwhm | 0.05 | cart-conformal-all-uniform | — | — | — |
| q_ext_fwhm | 0.05 | cart-staircase-graded | — | — | — |
| q_ext_fwhm | 0.05 | cart-staircase-uniform | — | — | — |
| q_ext_fwhm | 0.05 | tri-graded | — | — | — |
| q_ext_fwhm | 0.05 | tri-uniform | — | — | — |
| q_ext_fwhm | 0.02 | cart-conformal-all-graded | — | — | — |
| q_ext_fwhm | 0.02 | cart-conformal-all-uniform | — | — | — |
| q_ext_fwhm | 0.02 | cart-staircase-graded | — | — | — |
| q_ext_fwhm | 0.02 | cart-staircase-uniform | — | — | — |
| q_ext_fwhm | 0.02 | tri-graded | — | — | — |
| q_ext_fwhm | 0.02 | tri-uniform | — | — | — |

Per-run records:

| arm | h [nm] | unknowns | steps | run [s] | Q_ext NRMS | lambda_peak err | height err | FWHM err | enh peak err | reflection |
|---|---|---|---|---|---|---|---|---|---|---|
| cart-conformal-all-graded | 4 | 8112 | 8947 | 8 | 4.62e-03 | 0.00e+00 | 5.47e-03 | — | — | 1.85e-03 |
| cart-conformal-all-graded | 2.86 | 10800 | 12527 | 7 | 4.98e-03 | 0.00e+00 | 4.49e-03 | — | — | 2.06e-03 |
| cart-conformal-all-graded | 2 | 13872 | 17906 | 13 | 1.80e-03 | 0.00e+00 | 1.62e-03 | — | — | 7.03e-04 |
| cart-conformal-all-graded | 1.43 | 20172 | 25088 | 21 | 4.83e-04 | 0.00e+00 | 5.64e-04 | — | — | 2.11e-04 |
| cart-conformal-all-graded | 1 | 33708 | 35859 | 42 | 1.00e-04 | 0.00e+00 | 1.98e-04 | — | — | 8.11e-05 |
| cart-conformal-all-graded | 0.714 | 55488 | 50216 | 84 | 6.71e-05 | 0.00e+00 | 4.23e-05 | — | — | 9.33e-05 |
| cart-conformal-all-graded | 0.5 | 97200 | 71748 | 225 | 7.34e-05 | 0.00e+00 | 3.14e-05 | — | — | 2.24e-05 |
| cart-conformal-all-uniform | 4 | 18252 | 8968 | 6 | 4.69e-04 | 0.00e+00 | 1.82e-03 | — | — | 1.45e-04 |
| cart-conformal-all-uniform | 2.86 | 26508 | 12556 | 11 | 3.22e-04 | 0.00e+00 | 8.84e-04 | — | — | 3.98e-05 |
| cart-conformal-all-uniform | 2 | 38988 | 17938 | 22 | 1.05e-04 | 0.00e+00 | 4.83e-04 | — | — | 3.73e-05 |
| cart-conformal-all-uniform | 1.43 | 63948 | 25114 | 66 | 7.05e-05 | 0.00e+00 | 2.29e-04 | — | — | 1.75e-05 |
| cart-conformal-all-uniform | 1 | 120000 | 35878 | 109 | 2.91e-05 | 0.00e+00 | 1.37e-04 | — | — | 2.04e-05 |
| cart-conformal-all-uniform | 0.714 | 221952 | 50230 | 272 | 1.93e-05 | 0.00e+00 | 7.54e-05 | — | — | 2.08e-05 |
| cart-conformal-all-uniform | 0.5 | 433200 | 71758 | 838 | 1.33e-05 | 0.00e+00 | 4.64e-05 | — | — | 1.86e-05 |
| cart-staircase-graded | 4 | 8112 | 8947 | 5 | 7.33e-02 | 0.00e+00 | 9.26e-02 | — | — | 1.94e-03 |
| cart-staircase-graded | 2.86 | 10800 | 12527 | 8 | 7.19e-02 | 0.00e+00 | 9.12e-02 | — | — | 2.06e-03 |
| cart-staircase-graded | 2 | 13872 | 17895 | 11 | 6.10e-02 | 0.00e+00 | 7.63e-02 | — | — | 7.65e-04 |
| cart-staircase-graded | 1.43 | 20172 | 25054 | 30 | 5.14e-02 | 0.00e+00 | 6.39e-02 | — | — | 2.12e-04 |
| cart-staircase-graded | 1 | 33708 | 35791 | 50 | 3.54e-02 | 0.00e+00 | 4.41e-02 | — | — | 8.22e-05 |
| cart-staircase-graded | 0.714 | 55488 | 50108 | 93 | 2.61e-02 | 0.00e+00 | 3.26e-02 | — | — | 9.92e-05 |
| cart-staircase-graded | 0.5 | 97200 | 71583 | 183 | 1.92e-02 | 0.00e+00 | 2.40e-02 | — | — | 2.26e-05 |
| cart-staircase-uniform | 4 | 18252 | 8947 | 8 | 7.84e-02 | 0.00e+00 | 9.64e-02 | — | — | 1.46e-04 |
| cart-staircase-uniform | 2.86 | 26508 | 12527 | 10 | 7.74e-02 | 0.00e+00 | 9.50e-02 | — | — | 4.32e-05 |
| cart-staircase-uniform | 2 | 38988 | 17895 | 18 | 6.28e-02 | 0.00e+00 | 7.75e-02 | — | — | 3.72e-05 |
| cart-staircase-uniform | 1.43 | 63948 | 25054 | 71 | 5.19e-02 | 0.00e+00 | 6.43e-02 | — | — | 1.92e-05 |
| cart-staircase-uniform | 1 | 120000 | 35791 | 122 | 3.54e-02 | 0.00e+00 | 4.42e-02 | — | — | 2.06e-05 |
| cart-staircase-uniform | 0.714 | 221952 | 50108 | 264 | 2.60e-02 | 0.00e+00 | 3.25e-02 | — | — | 2.09e-05 |
| cart-staircase-uniform | 0.5 | 433200 | 71583 | 982 | 1.91e-02 | 0.00e+00 | 2.40e-02 | — | — | 1.88e-05 |
| tri-graded | 4 | 3833 | 12619 | 6 | 2.41e-02 | 0.00e+00 | 1.49e-02 | — | — | 2.13e-02 |
| tri-graded | 2.86 | 5177 | 13875 | 5 | 1.06e-02 | 0.00e+00 | 6.49e-03 | — | — | 6.83e-03 |
| tri-graded | 2 | 7943 | 18149 | 11 | 4.60e-03 | 0.00e+00 | 3.62e-03 | — | — | 4.29e-03 |
| tri-graded | 1.43 | 12743 | 26355 | 12 | 4.62e-04 | 0.00e+00 | 7.07e-04 | — | — | 2.83e-03 |
| tri-graded | 1 | 21861 | 38811 | 35 | 9.05e-04 | 0.00e+00 | 5.22e-04 | — | — | 2.00e-03 |
| tri-graded | 0.714 | 37525 | 51985 | 42 | 3.26e-04 | 0.00e+00 | 2.55e-04 | — | — | 1.84e-03 |
| tri-graded | 0.5 | 68735 | 74590 | 124 | 1.87e-04 | 0.00e+00 | 1.84e-04 | — | — | 1.67e-03 |
| tri-uniform | 4 | 13053 | 12619 | 6 | 1.34e-03 | 0.00e+00 | 1.88e-04 | — | — | 1.99e-03 |
| tri-uniform | 2.86 | 22325 | 13875 | 8 | 5.30e-04 | 0.00e+00 | 4.34e-04 | — | — | 1.83e-03 |
| tri-uniform | 2 | 40675 | 18660 | 19 | 1.50e-04 | 0.00e+00 | 5.75e-04 | — | — | 1.67e-03 |
| tri-uniform | 1.43 | 73349 | 26355 | 30 | 1.76e-04 | 0.00e+00 | 1.21e-04 | — | — | 1.80e-03 |
| tri-uniform | 1 | 140165 | 38811 | 90 | 6.88e-05 | 0.00e+00 | 6.14e-05 | — | — | 1.79e-03 |
| tri-uniform | 0.714 | 262209 | 52550 | 319 | 3.08e-05 | 0.00e+00 | 4.40e-05 | — | — | 1.79e-03 |
| tri-uniform | 0.5 | 515123 | 75126 | 775 | 2.25e-05 | 0.00e+00 | 1.48e-05 | — | — | 1.80e-03 |
