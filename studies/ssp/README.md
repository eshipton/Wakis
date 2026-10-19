# Subpixel-smoothed projection (SSP) study

This study covers `wakis.inverse.smoothed_projection` (JAX), `wakis.inverse.projection` (NumPy, with an explicit VJP) and `MaterialGrid(projection_beta=...)`. The method is Hammond et al., *Opt. Express* **33**, 33620 (2025) ([arXiv:2503.20189](https://arxiv.org/abs/2503.20189)). It was implemented from the paper's Eqs. 7, 10–12; Meep's (GPL) code was not used. The derivation, properties and **Limits & Failure Modes** are in the physics guide section "Subpixel-smoothed projection (SSP)".

```bash
PYTHONPATH=../.. python ssp_study.py pixel    # < 5 s: profiles, slope ripple vs R, convergence to binary
PYTHONPATH=../.. python ssp_study.py offset   # ~25 s, 1 process: Wakis FIT objective/gradient vs interface offset
PYTHONPATH=../.. python ssp_study.py pilot <collected Batch dir>   # invrs-gym pilot table + figure
```

## A. The gradient at β = ∞ as a level set moves through a cell

![](figures/a_offset_objective_gradient.png)

**Setup.**
- Wakis FIT, 2D TM (Ez), 40×40 cells of 7.5 mm, CPML, point source at 3 GHz.
- 6×6 cm `MaterialGrid` (17×17 nodes, ε 1 → 4) holding a straight level set ρ̃ = ½ + (x0 − x)/w, with w = 0.1 m.
- The offset x0 is swept over ±1 cell in 81 steps.
- Objective: F = |Ez(f0)|² at a probe behind the design. dF/dx0 comes from the exact adjoint (Σ dF/dρ / w).

**Arms**, all evaluated at the E edges:
- SSP at β = ∞ (R = 0.55 Δx);
- step projection at β = ∞ (R → 0);
- tanh projection at β = 64.

**Results.**
- **Step projection:** F is piecewise constant, it jumps when an edge sample changes side, and its gradient is exactly zero everywhere.
- **SSP at β = ∞:** F is smooth, and the adjoint gradient matches central finite differences of F(x0) to 2e-6–1e-4 relative (circles).
- **tanh at β = 64:** in this setup it is close to SSP. Its gray width 1/(β|∇ρ̃|) = 1.6 mm ≈ 0.2 Δx is still resolved by the edge samples.

The tanh result is the honest baseline. tanh's transition width depends on the local |∇ρ̃|, so it becomes a step (zero, spiky gradients) where the filtered field is steep, and as β grows. SSP's transition is always ~one cell, independent of β and |∇ρ̃|.

## B. Pixel-level properties

![](figures/b_pixel_profiles_ripple_convergence.png)

**(a) Profiles.** These are the analogue of the paper's Fig. 4: β → 0 gives ρ̃, and β = ∞ gives the C² fill F(d) inside |d| < R.

**(b) Shape-derivative ripple.** At β = ∞, a straight interface moving through the pixels changes the material at the rate dM/dx0 = Σ 15/(16R) (1 − u²)². This is analytic, and the code matches it to 1e-12. Its mean over a cell is exactly 1, but its spread depends on R:

| R | dM/dx0 min | dM/dx0 max | ratio |
|---|---|---|---|
| 0.45 Δx | 0 (frozen gradient: interface midway between samples) | 2.08 | ∞ |
| **0.55 Δx (paper)** | 0.10 | 1.70 | 17 |
| 0.75 Δx | 0.77 | 1.25 | 1.6 |
| 1.0 Δx | 0.94 | 1.06 | 1.1 |

The paper's choice avoids frozen gradients but leaves a 17× modulation of the shape derivative with sub-pixel position. R ≈ 0.75 Δx is a cheap improvement, at the cost of a slightly thicker gray layer. This is an open question for the optimizer pilots.

**(c) Convergence to the binary design.**
- Disk level set, filter width fixed in physical units, 32–1024 px grids.
- SSP at β = ∞: the gray-pixel fraction and the L¹ distance to the binary indicator fall as O(Δx).
- tanh at β = 32: about 6 % of the pixels stay gray at every resolution, and the L¹ distance stalls at 6e-3.

## C. invrs-gym pilot (ceviche lightweight bend and mode converter)

**Branch and code.** invrs-wakis branch `ssp`:
- `FeasibleDensity.projected(..., projection='ssp')`;
- `configs/ssp.json`;
- arm `ssp` of `scripts/run_challenge.py`.

**Batch job.** `scripts/ssp_batch.sh` runs one spec per n2-standard-2 Cloud Batch machine (spot, with GCS checkpoint/resume). The run ID is `20261019-0633-ssp-pilot`: 16 tasks = 2 challenges × seeds 0–3 × {round-2, SSP}.

**Arms, at equal iteration budget** (bend 135 steps, mode converter 200 steps, as in `configs/round2.json`):

| arm | stages |
|---|---|
| round-2 (`robust`) | tanh at β = 4, 8, 16, 32 (L-BFGS-B), then STE at β = 64 (Adam, objective r = 1 + r = 2) |
| SSP | SSP at β = 8, 32, ∞ (L-BFGS-B), then STE (Adam, r = 1 + r = 2) that simulates the repaired binary design and back-propagates the SSP(β = ∞) gradient |

Everything else is identical: the filter, the length-scale repair, candidate selection by the official backend, and the 10 nm re-check.

Results: *pending* (see `data/pilot.md` once collected).
