# Falsifier F-b1: Velocity-Shear Term vs Mode-3 Leakage

## Purpose

This study (`studies/p5b-shear/shear_fb1.py`) implements falsifier F-b1 for the M1 residual across the three grid and charge-to-mass cases `(nx, nz, qm) in [(10, 474, 1), (20, 948, 1), (10, 474, 2)]` using `run_particle_beam_drift(..., seed=42, m_mod=MODULATION_INDEX, profile="sheet_lattice")`.

On the test fit window `fit_window(params)`, the script fits the four-wave variable-projection model to `I1(z)`:
- Mode-3 fast and slow wavenumbers fixed at the exact-shear roots (and also reported with mode-3 fixed at the first-order roots),
- Mode-1 `kc` and `bq` free (`scipy.optimize.least_squares` starting at the first-order model values, with linear least squares for the four complex wave amplitudes at each `(kc, bq)`),
- Alongside the two-wave `fit_beta_q_from_envelope` value.

## Hypotheses

1. **Higher-order per-line velocity shear**: `discrete_modal_prediction("particles", ...)` treats the per-line DC velocity shear `u_l` to first order (mode-averaged `be_j`). The exact per-line cold-stream problem `det[diag((k - Om/u_l)^2) - diag(sigma_l S_z^2) G(k)] = 0` (`studies/p5b-shear/qep_ref.py`) has mode-1 roots that give a larger `beta_q`. Predicted mode-1 beta (rad/m), first-order model vs exact shear:
   - `qm=1 10x474`: `53.2413` vs `53.2527`
   - `qm=1 20x948`: `54.2462` vs `54.2583`
   - `qm=2 10x474`: `77.0475` vs `77.0818`
2. **Mode-3 under-excitation (competing hypothesis)**: Mode 3 is under-excited by 10-15 % at `qm=1`.

## Wave Amplitude Normalisation and Self-Check

In the modal sum `I1(z) = I1_0 * sum_j c_j * cos(bq_j * z) * exp(-i * kc_j * z)`, each standing cosine envelope decomposes into fast (`k_{j-} = kc_j - bq_j`) and slow (`k_{j+} = kc_j + bq_j`) waves via `cos(bq_j * z) = 0.5 * (exp(i * bq_j * z) + exp(-i * bq_j * z))`. Defining `A_n = 2 * a_n` where `I1(z) = sum_n a_n exp(-i * k_n * z) = sum_n A_n * [0.5 * exp(-i * k_n * z)]`, an unperturbed mode `j` has `|A_{j+}| = |A_{j-}| = I1_0 * |c_j|`, so:

```text
c3_meas = 0.5 * (|A3+| + |A3-|) / I1_0
c3_model = |c_modes[mode3]|
```

gives `|c3_meas / c3_model| = 1` on a two-mode model `I1(z)`. Before running particle cases, `shear_fb1.py` applies the four-wave fit to the synthetic first-order and exact-shear `I1(z)` signals and checks recovery of `beta_1` to `rel < 1e-6` and `|c3|` ratio to `< 1e-4`.

## Pre-Registered Decision Rule

Decision rule (pre-registered, copy verbatim): shear confirmed if |rel to exact| <= 9e-5 on all three cases and
|rel to first-order| > 1.5e-4 on qm=2; mode-3 masking supported if |c3_meas/c3_model| in [0.85, 0.90] on both qm=1
grids; ratio 1.00 +/- 0.03 rules it out.

## How to Run

Run all cases (logs `SELF_CHECK {json}` and `FB1 {json}` to stdout):

```bash
python studies/p5b-shear/shear_fb1.py
```

Run a single case `i` in `{0, 1, 2}`:

```bash
python studies/p5b-shear/shear_fb1.py --case 0
```

Run only the synthetic CPU self-check or the normalisation mutation check:

```bash
python studies/p5b-shear/shear_fb1.py --self-check-only
python studies/p5b-shear/shear_fb1.py --mutate-check
```

Run the QEP root reference script on a grid `(nx, nz)`:

```bash
python studies/p5b-shear/qep_ref.py 10 474
python studies/p5b-shear/qep_ref.py 20 948
```
