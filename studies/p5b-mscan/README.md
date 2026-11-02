# P5 part B diagnostic study: m-scan, settle time, aliased numerical Cherenkov

Coarse grid (10 x 1 x 474), deterministic `sheet_lattice` beam, 16 macroparticles per step,
q/m x1 unless stated. One Cloud Batch job (n2-standard-2) per point, script `run_point.py`,
post-processing `analyze.py`. Raw outputs (`out*/`, npz + json) are not committed; the json
files are small and are kept in `out*/` locally. Numbers below are copied from those json files.

## 1. Round-5 failures (settle 1.01 Lz/u_dep, shape 1)
- Raw particle fit at m = 0.015 sat 2.4e-4 below the modal prediction; the 3-point m->0
  Richardson extrapolation moved it to 1.56e-3 (bound 7.35e-4). beta(m) steps 0.0275, 0.0219 rad/m
  (ratio 1.25; an m^2 law gives 4).
- Subtracting the m = 0 run from each I1 (out/): residuals at m = 0.00375, 0.0075, 0.015 became
  -3.0e-4, -2.9e-4, -1.3e-4 (fit sigma 1.3e-4): an m-independent background, bias ~ 1/m.
- delta_meas - delta_disc = -2.1e-5 (q/m x1) and -8.3e-5 (q/m x2), independent of m and settle
  (open question: O(delta^2) term missing from the discrete DC model).

## 2. Settle time (out/, out3/)
- n_settle = 1.01 Lz/u_dep ignores the 2-period entrance ramp: DC velocity ripple (peak-to-peak / mean
  in the fit window) 6.0e-4 at settle 1.01 vs 3.6e-5 at 1.05 (q/m x1); 9.2e-3 vs 1.9e-3 (q/m x2).
- Derived settle (`settle_steps`, slow-wave group-velocity bound) = 1.23-1.30 x the old value.

## 3. The DC beam is unstable with shape 1 (out3/)
Settle-factor scan, m = 0.015 residual vs modal prediction, and the m = 0 run's I_z fluctuation rms
in the first and last sixth of the 6-period DFT window:

| settle factor | beta_q rel. residual | m = 0 rms first -> last sixth [A] |
|---|---|---|
| 1.05 | -2.05e-4 | 3.50e-3 -> 1.49e-3 (transient decaying) |
| 1.10 | -4.63e-4 | 7.10e-4 -> 1.91e-3 |
| 1.15 | -1.16e-3 | 4.65e-4 -> 2.73e-3 |
| 1.20 | -2.83e-3 | 7.05e-4 -> 3.40e-3 |
| 1.30 | -1.04e-2 | 1.68e-3 -> 5.18e-3 |
| 1.50 | -3.62e-2 | 4.81e-3 -> 9.23e-3 |

2D FFT of the m = 0 I_z(t, z) history (settle 1.5): peak at f = 111.5 GHz, k_z = 2336 rad/m (after
undoing the 8-edge decimation alias), harmonic at 222 GHz; rms peaks near the entrance (14.8 mA at
z ~ 0.05 m) and falls to ~4 mA at the exit.

Interpretation: the m = -1 spatial alias of the drifting beam, omega = (2 pi/dz - k) u, crosses the
backward light line omega = c k at k = (2 pi/dz) u/(c + u) = 2336 rad/m, f = c k/(2 pi) = 111.5 GHz
(u = 5.78e7 m/s, dz = 0.186/428 m). Energy flows upstream (backward wave) and reflects at the PEC
entrance wall: an absolute, BWO-like numerical instability. Note k dz = 2 pi u/(c + u) is independent
of dz, so grid refinement does not remove it.

## 4. Mitigation scan (out4/, settle 1.5)
| configuration | m = 0 rms first -> last sixth [A] | growth [1/s] | run time [s] |
|---|---|---|---|
| shape 1 | 4.81e-3 -> 9.23e-3 | 8.5e8 | 1928 |
| shape 2 | 1.38e-5 -> 1.41e-5 | ~0 (2.9e7) | 3806 |
| shape 3 | 1.29e-5 -> 1.26e-5 | ~0 (-2.8e7) | 7218 |
| shape 1 + binomial | 1.59e-3 -> 6.36e-3 | 1.8e9 | 1982 |
| shape 2 + binomial | 8.74e-6 -> 9.50e-6 | ~0 | 3906 |
| shape 3 + binomial | 8.70e-6 -> 8.32e-6 | ~0 | 7410 |

(Growth = ln(rms_last/rms_first) over 5/6 of the window.) The binomial filter alone does not help:
its symbol cos^2(k dz/2) is 0.76 at k dz = 1.015. Shape 2/3 beta_q residuals in this scan are not
meaningful yet: the modal and DC models hard-code CIC (shape 1) weights.

Decision (owner, 2026-11-02): run V7 with shape 2 and generalise the modal/DC model to shape p.
