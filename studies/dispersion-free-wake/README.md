# Dispersion-free wake computation: Yee vs explicit TE/TM

Study for `SolverFIT3D(scheme='tete')` (`wakis/tete.py`): the explicit TE/TM
scheme of Dohlus & Zagorodnov, J. Comput. Phys. 228, 2822 (2009)
(arXiv:0901.1053), dispersion-free along the beam axis at `c dt = dz`.
Physics and limits: physics guide, section "Dispersion-free wake computation".

All runs: NVIDIA L4 VM (`cem-tete`, g2-standard-8), Python 3.12, CuPy 14.2,
float64, q = 1 nC, beta = 1, beam and test particle on axis.

## Cases and baselines

| case | geometry | boundary | reference |
|---|---|---|---|
| `pipe` square | 16 mm × 16 mm PEC pipe (domain walls), L = 5–80 cm | CPML at both z ends + TFSF self-field | exact: W = 0 |
| `pipe` round | conformal round pipe r = 6 mm, L = 10, 40 cm | same | exact: W = 0 |
| `box` | closed 32 × 32 × 96 mm PEC cavity, beam through the end walls | all PEC, direct current source | analytic modal wake (`analytic_box.py`) |
| `pillbox` | conformal pillbox R = 16 mm, gap 16 mm, pipes r = 5 mm, 20 mm each side | CPML + TFSF | finest TE/TM run (dz = 0.1 mm), finest Yee shown |

Baselines (Rule 2): Yee at cfln 0.99 (best Yee) and Yee at cfln 0.5 (the
Wakis default), on the **same grid** (dx = dy = 2 dz) as TE/TM; the
equal-work comparison plots the error against cells × time steps.

Post-processing: the Wakis wake integral (`WakeSolver`) samples `Ez` at
`int(t/dt)`, a first-order time error that hits Yee (t/dt not integer) but
not TE/TM at `c dt = dz`. Every run therefore also stores the same integral
with linear time interpolation (`wake_interp.py`), which is the primary
metric below; the Wakis values are kept (`k_loss`, `WP`).

## Results

### Smooth PEC pipe

16 mm square pipe, `dz = 0.4 mm`, `dx = dy = 2 dz`, CPML + TFSF self-field (exact wake 0), `max |W(s)|` [V/pC]:

| sigma_z | L = 0.05 m | 0.1 m | 0.2 m | 0.4 m | 0.8 m |
|---|---|---|---|---|---|
| 1.6 mm, TE/TM | 3.7e-10 | 8.3e-10 | 1.6e-9 | 3.3e-9 | 6.5e-9 |
| 1.6 mm, Yee 0.99 | 1.18 | 2.82 | 6.08 | 12.4 | 25.1 |
| 1.6 mm, Yee 0.5 | 2.86 | 6.79 | 15.1 | 31.5 | 64.3 |
| 0.8 mm, TE/TM | 1.4e-9 | 3.1e-9 | 6.9e-9 | 1.4e-8 | 2.7e-8 |
| 0.8 mm, Yee 0.99 | 15.6 | 33.8 | 71.0 | 148 | 301 |
| 0.8 mm, Yee 0.5 | 37.6 | 78.5 | 162 | 328 | 660 |

Conformal round pipe `r = 6 mm`, `sigma_z = 0.8 mm`: TE/TM 3.1e-9 (`L = 0.1 m`) and 1.2e-8 (`0.4 m`); Yee 0.99 58.7 and 253; Yee 0.5 86.4 and 365 V/pC. TE/TM residuals are round-off (they grow linearly with the number of steps); `|loss factor|` TE/TM <= 3e-10 V/pC.

![](figures/pipe_spurious_wake.png)

Spurious longitudinal wake potential in a smooth PEC pipe as a function of pipe length `L` and wake coordinate `s`, comparing Yee (`cfln = 0.99` and `0.5`) against TE/TM at `c dt = dz` on staircase square and conformal round pipes.

### Closed rectangular PEC cavity

Closed 32 × 32 × 96 mm PEC cavity, beam through the end walls, analytic modal wake (`analytic_box.py`). Relative loss-factor error `|k/k_analytic - 1|`:

| sigma_z | sigma_z/dz | TE/TM | Yee 0.99 | Yee 0.5 |
|---|---|---|---|---|
| 3.2 mm | 4 / 8 / 24 | 9.5e-3 / 2.3e-3 / 2.6e-4 | 1.5e-2 / 3.9e-3 / 4.2e-4 | 3.4e-2 / 8.4e-3 / 9.3e-4 |
| 1.6 mm | 2 / 4 / 12 | 2.6e-2 / 5.4e-3 / 5.7e-4 | 6.5e-2 / 1.4e-2 / 1.5e-3 | 1.5e-1 / 3.1e-2 / 3.2e-3 |
| 0.8 mm | 1 / 2 / 6 | 1.0e-1 / 2.5e-2 / 2.3e-3 | 4.8e-1 / 1.0e-1 / 9.5e-3 | 1.5 / 2.7e-1 / 2.3e-2 |

Long-range wake (RMS error over `s` in `[5 sigma_z, 0.29 m]`) at `sigma_z = 0.8 mm`, `sigma_z/dz = 3`: TE/TM 0.12, Yee 0.99 0.21, Yee 0.5 0.48; at `sigma_z = 3.2 mm`, `sigma_z/dz = 8`: 0.033 / 0.029 / 0.057 (cavity-mode dispersion, set by the transverse mesh, dominates there: no TE/TM advantage).

Equal work (cells × steps) for a 1 % loss-factor error at `sigma_z = 0.8 mm`: TE/TM 2.0e9 (`sigma_z/dz = 3`), Yee 0.99 3.9e10 (`sigma_z/dz = 6`), i.e. 19x; Yee 0.5 above 7.8e10 (2.3 % there). At `sigma_z = 3.2 mm` and 1e-3: TE/TM 2.1e9, Yee 0.99 about 8e9 (4x).

On the box at `sigma_z = 3.2 mm`, `sigma_z/dz = 4`, the Wakis `int(t/dt)` wake integral raises the Yee 0.99 loss-factor error from 1.5 % (interpolated, `wake_interp.py`) to 7.8 %.

![](figures/box_cavity_convergence.png)

Convergence of the loss-factor error against longitudinal resolution `sigma_z/dz` and computational work (cells × steps), together with short- and long-range wake waveforms against the analytic modal sum for the closed rectangular PEC cavity.

### Conformal pillbox cavity

Conformal pillbox (`R = 16 mm`, gap 16 mm, pipes `r = 5 mm`, CPML + TFSF). Reference: finest runs, `dz = 0.1 mm`. Finest TE/TM vs finest Yee loss factors [V/pC]: `sigma_z = 3.2 mm` 1.940 / 1.927; 1.6 mm 3.197 / 3.174; 0.8 mm 5.340 / 5.384; 0.4 mm 6.671 / 9.205 (Yee not converged: `sigma_z/dz = 4`). So the reference uncertainty is about 0.7 % for `sigma_z >= 0.8 mm` (the two conformal models differ at first order). Error vs the finest TE/TM run:

| sigma_z | sigma_z/dz | TE/TM | Yee 0.99 |
|---|---|---|---|
| 3.2 mm | 4 / 8 / 16 | 7.6e-3 / 1.1e-2 / 5.1e-3 | 2.4e-2 / 6.0e-3 / 4.4e-3 |
| 1.6 mm | 2 / 4 / 8 | 2.3e-2 / 1.1e-2 / 5.6e-3 | 1.4e-1 / 8.5e-3 / 1.1e-3 |
| 0.8 mm | 1 / 2 / 4 | 5.5e-2 / 3.3e-2 / 1.1e-2 | 2.8 / 4.2e-1 / 8.0e-2 |
| 0.4 mm | 0.5 / 1 / 2 | 1.4e-1 / 9.8e-2 / 2.1e-2 | 27 / 10 / 2.1 |

For long bunches (`sigma_z/dz >= 4`) both schemes agree within the 0.7 % reference uncertainty and the TE/TM conformal recipe (trapezoidal areas, mass scaling) converges no faster than the Yee conformal model; for short bunches (`sigma_z/dz <= 4`) Yee errors grow to 8 %–1000 %, TE/TM stays at 1–14 %.

![](figures/pillbox_convergence.png)

Loss-factor error relative to the finest (`dz = 0.1 mm`) TE/TM reference and wake potential profiles across bunch lengths `sigma_z = 0.4–3.2 mm` for the conformal pillbox cavity.

### Cost per step and off-axis dispersion

Cost per step (median, same grid, PEC box, `dx = dy = 2 dz`): L4 GPU 1.05 M cells: Yee 2.85 ms, TE/TM 4.63 ms (1.63x); 8.4 M cells: 26.1 ms vs 61.9 ms (2.37x); CPU (SciPy, one VM core set) 1.05 M cells: 67 ms vs 117 ms (1.74x). TE/TM takes a 1.24x larger `dt` than Yee 0.99 (2.47x larger than Yee 0.5), so per simulated time it costs 1.32x (GPU 1 M) to 1.92x (GPU 8 M) of Yee 0.99. The TE/TM GPU step is not yet fused into custom kernels (the Yee step is), which explains most of the gap to the 4/3 sparse-work ratio. Combined with the 19x work saving, the wall-time saving at equal accuracy for short bunches is about 10x.

![](figures/dispersion_vs_angle.png)

Numerical phase-velocity error `1 - v_ph/c` as a function of propagation angle from the `z` axis on a `dx = dy = 2 dz` grid, showing exact longitudinal propagation (`theta = 0`) for TE/TM at `c dt = dz` and comparable off-axis dispersion to Yee.

## Caveats

- Reference uncertainty of the conformal pillbox is about 0.7 % (`sigma_z >= 0.8 mm`), because the TE/TM and Yee conformal cut-cell models differ at first order.
- Only one cavity geometry of each type (one closed rectangular box, one conformal pillbox) was tested.
- The TE/TM GPU time step is not fused into custom kernels (unlike the Yee GPU step).
- The 16.4 M-cell TE/TM step benchmark did not run (only 1.05 M and 8.4 M cells were benchmarked).
- The Arnoldi stability check was replaced by Gershgorin certificates (`wakis.tete.gershgorin_limits`).
- Yee runs use the Wakis staircase TEM template on staircase grids and the operator template (`Beam.tfsf_template='operator'`) on conformal grids (the same template on conformal grids for both schemes).

## Files

- `run_case.py` one run → `results/<name>.json`; `make_cases.py` case lists;
  `run_queue.sh` / VM queue; `analyze.py` figures + `results/summary.json`;
  `bench_step.py` cost per step; `dispersion_vs_angle.py` (local, tiny).
- `analytic_box.py` closed-cavity modal wake (Faddeeva form), self-check
  `python analytic_box.py` (sum k_n e^{-k^2 sigma^2} = int W lambda ds).
- Reproduce: on a GPU machine with the fork installed,
  `python make_cases.py; ./run_queue.sh cases_small.txt 4; ./run_queue.sh cases_large.txt 1; python analyze.py`.
