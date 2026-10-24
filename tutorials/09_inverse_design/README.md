# Chapter 9 · Inverse Design

Gradient-based electromagnetic topology and shape optimization using Wakis's exact discrete adjoint solver (`wakis.inverse`) and JAX integration (`jax_simulation`).

## Tutorials

- **[9.1 Adjoint gradients](09-01_adjoint_gradients/adjoint_gradients.ipynb)** ([`09-01_adjoint_gradients`](09-01_adjoint_gradients/adjoint_gradients.py)) — Exact discrete transpose of the leapfrog FIT time-stepper (`AdjointSimulation`, `AdjointStepper`), why discretize-then-differentiate is exact to round-off vs continuous-adjoint mismatch, `MaterialGrid` density parameterization with conic filtering, $\tanh$ projection ($\beta$-continuation) and subpixel-smoothed tensor averaging (`smoothing="tensor"`), operator dot-product and central finite-difference V-curve verification, the four-rung memory ladder (`store`, Griewank–Walther `checkpoint`, CPML-interface `reverse`, and `nyquist`), and a live 2D waveguide-bend topology optimization.
- **[9.2 Waveguide bend](09-02_waveguide_bend/waveguide_bend.ipynb)** ([`09-02_waveguide_bend`](09-02_waveguide_bend/waveguide_bend.py)) — Topology optimization of the `invrs-gym` `ceviche_waveguide_bend` challenge (10 nm grid, $312\times 312$ pixels, $160\times 160$ design region, 6 wavelengths, 8 px minimum feature length scale), two-stage `FeasibleDensity` L-BFGS-B ($\beta=4..32$) + morphological brush STE ($\beta=64$) pipeline, broadband Wakis re-simulation vs stored `dense_S` and `ceviche` FDFD cross-validation ($\max|\Delta|S|^2| = 3.77\times 10^{-4}$ at $10\text{ nm}$), and like-for-like leaderboard comparison ($+0.6427$ and $+0.5793$ at $8/8\text{ px}$ vs $-1.2504$ at $9/7\text{ px}$).
- **9.3 Power splitter and mode converter** (`09-03_splitter_and_mode_converter`) *(in progress; round-2 run data in `data/`)* — Multi-objective broadband S-parameter optimization of a $1\times 2$ power splitter (`ceviche_power_splitter`) and a $\text{TE}_0 \to \text{TE}_1$ waveguide mode converter (`ceviche_mode_converter`).
- **9.4 Metalens** (`09-04_metalens`) *(in progress)* — The `invrs-gym` `metalens` challenge (2D x–z cross-section, periodic 16 µm, focal-intensity enhancement at 450/550/650 nm, official fmmax evaluator): our round-2 lens with guaranteed length scale (enhancement 8.87), the round-1 lens without it, and the Wakis vs fmmax cross-check on the reference design.

## Results behind this chapter

Our designs for the `invrs-gym` challenges, scored with the gym's own evaluators and compared like-for-like with the
published leaderboard solutions (designs and drivers in [`invrs-wakis`](https://github.com/eshipton/invrs-wakis)):

| Challenge | Ours (min. width/spacing) | Leaderboard best |
|---|---|---|
| Waveguide bend | **+0.643** (8/8 px) | −1.250 (9/7 px) |
| Power splitter | **+0.068** (8/8 px) | +0.009 (9/8 px) |
| Mode converter | +0.312 (8/8 px) | 0.854 (8/8 px) — behind |
| Metalens | 8.87 enhancement (guaranteed length scale) | — |

Related studies: [`studies/adjoint-memory`](../../studies/adjoint-memory/) (exact gradients in 30–60× less memory),
[`studies/ssp`](../../studies/ssp/) (subpixel-smoothed projection), [`studies/gpu-perf`](../../studies/gpu-perf/)
(fused GPU step).
