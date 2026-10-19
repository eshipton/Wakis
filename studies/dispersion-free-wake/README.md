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

RESULTS

## Files

- `run_case.py` one run → `results/<name>.json`; `make_cases.py` case lists;
  `run_queue.sh` / VM queue; `analyze.py` figures + `results/summary.json`;
  `bench_step.py` cost per step; `dispersion_vs_angle.py` (local, tiny).
- `analytic_box.py` closed-cavity modal wake (Faddeeva form), self-check
  `python analytic_box.py` (sum k_n e^{-k^2 sigma^2} = int W lambda ds).
- Reproduce: on a GPU machine with the fork installed,
  `python make_cases.py; ./run_queue.sh cases_small.txt 4; ./run_queue.sh cases_large.txt 1; python analyze.py`.
