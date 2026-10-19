# B4 baseline summary

Cost from which each target is met and stays met on finer meshes (log-log interpolation along the h ladder; '—' = not reached; '<=' = already met by the coarsest run). Unknowns: 3 per Cartesian cell, nodes + edges for triangles. Time: time-stepping wall time on one uncontended n2-standard-2 core.

## B4

| metric | target | arm | unknowns | work | time [s] |
|---|---|---|---|---|---|
| f_res | 0.01 | cart-conformal-pec-graded | — | — | — |
| f_res | 0.01 | cart-staircase-edges-graded | — | — | — |
| f_res | 0.01 | cart-staircase-edges-graded-bigdomain | — | — | — |
| f_res | 0.01 | cart-staircase-edges-graded-hz0.75ht | — | — | — |
| f_res | 0.01 | cart-staircase-edges-graded-hz1.5ht | — | — | — |
| f_res | 0.01 | cart-staircase-edges-graded-ppw40 | — | — | — |
| f_res | 0.01 | cart-staircase-graded | — | — | — |
| f_res | 0.005 | cart-conformal-pec-graded | — | — | — |
| f_res | 0.005 | cart-staircase-edges-graded | — | — | — |
| f_res | 0.005 | cart-staircase-edges-graded-bigdomain | — | — | — |
| f_res | 0.005 | cart-staircase-edges-graded-hz0.75ht | — | — | — |
| f_res | 0.005 | cart-staircase-edges-graded-hz1.5ht | — | — | — |
| f_res | 0.005 | cart-staircase-edges-graded-ppw40 | — | — | — |
| f_res | 0.005 | cart-staircase-graded | — | — | — |
| r_res | 0.05 | cart-conformal-pec-graded | — | — | — |
| r_res | 0.05 | cart-staircase-edges-graded | 4.07e+06 | 1.61e+11 | 4633 |
| r_res | 0.05 | cart-staircase-edges-graded-bigdomain | — | — | — |
| r_res | 0.05 | cart-staircase-edges-graded-hz0.75ht | <= 2.52e+07 | <= 1.86e+12 | <= 4779 |
| r_res | 0.05 | cart-staircase-edges-graded-hz1.5ht | <= 1.32e+07 | <= 7.84e+11 | <= 1934 |
| r_res | 0.05 | cart-staircase-edges-graded-ppw40 | — | — | — |
| r_res | 0.05 | cart-staircase-graded | — | — | — |
| r_res | 0.02 | cart-conformal-pec-graded | — | — | — |
| r_res | 0.02 | cart-staircase-edges-graded | — | — | — |
| r_res | 0.02 | cart-staircase-edges-graded-bigdomain | — | — | — |
| r_res | 0.02 | cart-staircase-edges-graded-hz0.75ht | — | — | — |
| r_res | 0.02 | cart-staircase-edges-graded-hz1.5ht | — | — | — |
| r_res | 0.02 | cart-staircase-edges-graded-ppw40 | — | — | — |
| r_res | 0.02 | cart-staircase-graded | — | — | — |
| z_nrms | 0.05 | cart-conformal-pec-graded | — | — | — |
| z_nrms | 0.05 | cart-staircase-edges-graded | 6.84e+06 | 3.66e+11 | 1046 |
| z_nrms | 0.05 | cart-staircase-edges-graded-bigdomain | — | — | — |
| z_nrms | 0.05 | cart-staircase-edges-graded-hz0.75ht | <= 2.52e+07 | <= 1.86e+12 | <= 4779 |
| z_nrms | 0.05 | cart-staircase-edges-graded-hz1.5ht | <= 1.32e+07 | <= 7.84e+11 | <= 1934 |
| z_nrms | 0.05 | cart-staircase-edges-graded-ppw40 | — | — | — |
| z_nrms | 0.05 | cart-staircase-graded | — | — | — |
| z_nrms | 0.02 | cart-conformal-pec-graded | — | — | — |
| z_nrms | 0.02 | cart-staircase-edges-graded | — | — | — |
| z_nrms | 0.02 | cart-staircase-edges-graded-bigdomain | — | — | — |
| z_nrms | 0.02 | cart-staircase-edges-graded-hz0.75ht | — | — | — |
| z_nrms | 0.02 | cart-staircase-edges-graded-hz1.5ht | — | — | — |
| z_nrms | 0.02 | cart-staircase-edges-graded-ppw40 | — | — | — |
| z_nrms | 0.02 | cart-staircase-graded | — | — | — |
| directivity | 0.02 | cart-conformal-pec-graded | <= 2.64e+06 | <= 4.99e+10 | <= 2675 |
| directivity | 0.02 | cart-staircase-edges-graded | <= 2.64e+06 | <= 4.99e+10 | <= 2936 |
| directivity | 0.02 | cart-staircase-edges-graded-bigdomain | — | — | — |
| directivity | 0.02 | cart-staircase-edges-graded-hz0.75ht | <= 2.52e+07 | <= 1.86e+12 | <= 4779 |
| directivity | 0.02 | cart-staircase-edges-graded-hz1.5ht | <= 1.32e+07 | <= 7.84e+11 | <= 1934 |
| directivity | 0.02 | cart-staircase-edges-graded-ppw40 | <= 4.34e+06 | <= 1.57e+11 | <= 387 |
| directivity | 0.02 | cart-staircase-graded | <= 2.64e+06 | <= 4.99e+10 | <= 3179 |
| directivity | 0.01 | cart-conformal-pec-graded | <= 2.64e+06 | <= 4.99e+10 | <= 2675 |
| directivity | 0.01 | cart-staircase-edges-graded | <= 2.64e+06 | <= 4.99e+10 | <= 2936 |
| directivity | 0.01 | cart-staircase-edges-graded-bigdomain | — | — | — |
| directivity | 0.01 | cart-staircase-edges-graded-hz0.75ht | <= 2.52e+07 | <= 1.86e+12 | <= 4779 |
| directivity | 0.01 | cart-staircase-edges-graded-hz1.5ht | <= 1.32e+07 | <= 7.84e+11 | <= 1934 |
| directivity | 0.01 | cart-staircase-edges-graded-ppw40 | <= 4.34e+06 | <= 1.57e+11 | <= 387 |
| directivity | 0.01 | cart-staircase-graded | <= 2.64e+06 | <= 4.99e+10 | <= 3179 |

Per-run records:

| arm | h [nm] | unknowns | steps | run [s] | Q_ext NRMS | lambda_peak err | height err | FWHM err | enh peak err | reflection |
|---|---|---|---|---|---|---|---|---|---|---|
| cart-conformal-pec-graded | 5e+05 | 2636400 | 18934 | 2675 | — | — | — | — | — | — |
| cart-conformal-pec-graded | 3.33e+05 | 3325296 | 27510 | 5920 | — | — | — | — | — | — |
| cart-conformal-pec-graded | 2.5e+05 | 3510000 | 36256 | 5339 | — | — | — | — | — | — |
| cart-staircase-edges-graded | 5e+05 | 2636400 | 18934 | 2936 | — | — | — | — | — | — |
| cart-staircase-edges-graded | 3.33e+05 | 3325296 | 27510 | 4787 | — | — | — | — | — | — |
| cart-staircase-edges-graded | 2.5e+05 | 3510000 | 36256 | 7098 | — | — | — | — | — | — |
| cart-staircase-edges-graded | 1.67e+05 | 7184295 | 55021 | 909 | — | — | — | — | — | — |
| cart-staircase-edges-graded | 1.25e+05 | 8288055 | 72512 | 1374 | — | — | — | — | — | — |
| cart-staircase-edges-graded-bigdomain | 2.5e+05 | 5180334 | 36256 | 440 | — | — | — | — | — | — |
| cart-staircase-edges-graded-hz0.75ht | 1.67e+05 | 25213923 | 73603 | 4779 | — | — | — | — | — | — |
| cart-staircase-edges-graded-hz1.5ht | 1.67e+05 | 13246659 | 59206 | 1934 | — | — | — | — | — | — |
| cart-staircase-edges-graded-ppw40 | 2.5e+05 | 4340070 | 36256 | 387 | — | — | — | — | — | — |
| cart-staircase-graded | 5e+05 | 2636400 | 18934 | 3179 | — | — | — | — | — | — |
| cart-staircase-graded | 3.33e+05 | 3325296 | 27510 | 5464 | — | — | — | — | — | — |
