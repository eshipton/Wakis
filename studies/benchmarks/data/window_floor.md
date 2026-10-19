# DFT-window floor (1x vs 3x window, same arm and mesh)

Floor = change of the metric when the window is tripled. Error vs reference at 1x and 3x for scale.

| case | arm | window 1x [fs] | Q_ext NRMS (1x vs 3x) | peak lambda | peak height | FWHM | enh peak | Q_ext NRMS vs ref 1x / 3x | enh_peak vs ref 1x / 3x |
|---|---|---|---|---|---|---|---|---|---|
| B2-Ag-TE | cart-conformal-all-graded | 80 | 1.4e-07 | 8.2e-10 | 8.1e-10 | 1.1e-08 | — | 5.4e-03 / 5.4e-03 | — |
| B2-Ag-TE | tri-uniform | 80 | 8.7e-10 | 4.2e-10 | 1.4e-09 | 1.6e-09 | — | 2.5e-04 / 2.5e-04 | — |
| B2-Ag-TM | cart-conformal-all-graded | 80 | 2.1e-09 | — | — | — | — | 7.3e-05 / 7.3e-05 | — |
| B2-Ag-TM | tri-uniform | 80 | 1.1e-09 | — | — | — | — | 3.1e-05 / 3.1e-05 | — |
| B2-Au-TE | cart-conformal-all-graded | 80 | 2.8e-06 | 1.1e-07 | 1.9e-07 | 1.9e-07 | — | 5.8e-03 / 5.8e-03 | — |
| B2-Au-TE | tri-uniform | 80 | 7.5e-11 | 7.5e-11 | 1.8e-11 | 5.0e-11 | — | 2.6e-04 / 2.6e-04 | — |
| B2-Au-TM | cart-conformal-all-graded | 80 | 9.9e-09 | — | — | — | — | 1.2e-04 / 1.2e-04 | — |
| B2-Au-TM | tri-uniform | 80 | 2.4e-09 | — | — | — | — | 4.8e-05 / 4.8e-05 | — |
| B3-g10 | cart-conformal-all-graded | 100 | 4.2e-06 | 3.4e-08 | 3.9e-07 | — | 5.6e-07 | 4.3e-03 / 4.3e-03 | 4.6e-02 / 4.6e-02 |
| B3-g10 | tri-uniform | 100 | 1.7e-10 | 1.7e-10 | 7.0e-11 | — | 7.9e-11 | 8.0e-04 / 8.0e-04 | 1.6e-04 / 1.6e-04 |
| B3-g2 | cart-conformal-all-graded | 100 | 8.3e-06 | 6.0e-07 | 9.9e-07 | 4.1e-06 | 1.5e-06 | 1.6e-02 / 1.6e-02 | 3.6e-02 / 3.6e-02 |
| B3-g2 | tri-uniform | 100 | 4.6e-07 | 1.6e-07 | 5.4e-07 | 2.2e-07 | 3.6e-11 | 1.8e-03 / 1.8e-03 | 2.5e-03 / 2.5e-03 |
| B3-g5 | cart-conformal-all-graded | 100 | 5.5e-06 | 1.8e-08 | 5.8e-07 | 2.1e-06 | 1.8e-07 | 8.6e-03 / 8.6e-03 | 4.1e-02 / 4.1e-02 |
| B3-g5 | tri-uniform | 100 | 9.2e-09 | 6.6e-09 | 4.3e-09 | 2.7e-09 | 4.0e-11 | 9.0e-04 / 9.0e-04 | 6.9e-03 / 6.9e-03 |
