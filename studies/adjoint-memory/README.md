# Exact-gradient memory ladder for the Wakis adjoint

`AdjointSimulation(memory=...)` (`wakis/inverse/memory.py`) offers four ways to get the per-step design rows that the backward sweep needs. The theory and the Limits & Failure Modes are in `docs/physicsguide.md`, section "Memory ladder".

| rung | exact | memory | extra forward steps |
|---|---|---|---|
| `store` (default) | yes | Nt × N_design | 0 |
| `checkpoint` (binomial, Griewank & Walther 2000) | yes, bitwise replay | s snapshots + one block of rows | `plan_checkpoints` |
| `reverse` (time-reversed leapfrog + recorded CPML interface layers) | yes, round-off ~1e-15 | Nt × (interface layers + source supports) | 0 |
| `nyquist` | **no** (approximate, opt-in) | Nt × N_design / d | 0 |

## Files
- `bend2d.py`: memory vs recompute time on a ceviche-bend-sized 2D problem (313² cells, 25 600 design edges, Nt = 20 000), CPU (tracemalloc) and L4 GPU (CuPy memory hook). Writes `bend_cpu.json` / `bend_gpu.json`.
- `roundoff.py`: gradient error of `reverse` vs `store` for Nt = 250 to 64 000 in an open CPML domain and a closed lossless PEC cavity. Writes `roundoff.json`.
- `project.py`: projects the ladder to the full ceviche bend and WDM (r = 1, sizes from `invrs_wakis.resources`) and to a 3D metalens-sized case. Snapshot sizes and interface layers are counted on real solvers. Writes `projection.json`.
- `make_figures.py`: renders all figures from the JSON files.

All runs used the VM `cem-adjmem` (g2-standard-8, one L4, us-central1-b). Nothing heavy ran on the laptop.

## Results

### Correctness (`tests/test_045_adjoint_memory.py`, CPU on the VM: 39 passed, 1 GPU-only skip)
- **Checkpoint:** every replayed row is bitwise equal to the forward row, and the gradient is bitwise equal to `store`. This holds in 14 physics cases: CPML, lossy PEC, flux, near2far, mode, magnetic current, lumped RLC and ports, dispersive medium, dispersive design, Bloch, tensor smoothing (PEC and CPML), dispersive tensor network, TF/SF. It holds for 3 slot/block settings and for a second `vjp`.
  - Exception: the `mode` case agrees to < 1e-12, because its forward run is itself not reproducible (ARPACK random start; store vs store differs by 2e-17).
- **Reverse:** agrees with `store` to 3e-16 to 2e-15 on the 7 lossless cases. The 7 lossy, dispersive, lumped and tensor cases fall back to checkpointing with a warning and stay bitwise.
- **Mutations** (each check can fail, and the break was confirmed):
  - Corrupting one recorded H or E interface step changes the gradient by 7e-3; one source-current step, by 3e-5.
  - Leaving the CPML psi or the source state out of the snapshot breaks the replay.
  - Halving splits are worse than the DP optimum.
- **Measured memory (tracemalloc):** the checkpoint saving is 6.77 MB, against 6.80 MB predicted.

### Round-off of the reversed leapfrog (`roundoff.png`)
![roundoff](roundoff.png)

The error stays flat at (0.3–5)e-15 from Nt = 250 to 64 000, in both the open and the closed domain. That is below the eps·sqrt(Nt) random-walk line. This is a float64 result only; float32 is untested.

### What the ceviche bend/WDM and a 3D metalens need (`projection.png`)
![projection](projection.png)

Rows are float64, which is what makes the result bitwise equal to the exact adjoint. "Extra" counts forward steps recomputed during the backward sweep, divided by Nt; the recording replays are included.

| problem | store f64 | store f32 (1e-7 error) | checkpoint | reverse | nyquist (approx.) |
|---|---|---|---|---|---|
| ceviche bend r=1 (0.10 Mcells, Nt 56 587, 25 600 edges as counted by the bridge) | 11.6 GB | 5.8 GB | 0.49 GB at 2.25×; 1.05 GB at 1.75×; 3.9 GB at 1.0× | 1.1 GB, 0× | 0.25 GB (d = 47) |
| ceviche WDM r=1 (0.63 Mcells, Nt 56 587, 409 600 edges) | 185 GB | 92.7 GB | 1.2 GB at 3.6×; 4.3 GB at 2.4×; 15 GB at 1.8× | 2.9 GB, 0× | 4.0 GB (d = 47) |
| 3D metalens-sized (4.84 Mcells, Nt 4000, 1.2 M edges) | 38.4 GB | 19.2 GB | 2.0 GB at 6.1×; 4.2 GB at 2.8×; 6.5 GB at 2.0×; 13.4 GB at 1.0× | 26 GB (TF/SF faces dominate), 0× | 0.94 GB (d = 41) |

The memory figures cover the gradient storage only; the solver itself comes on top (WDM about 1 GB host). Snapshots are 7.9 MB (bend), 47 MB (WDM) and 0.56 GB (3D).

The 3D metalens case is an assumption, not an existing challenge:
- a 10 λ × 10 λ aperture, 20 cells/λ, 4 λ tall, with 10-cell CPML (220 × 220 × 100 cells);
- a 10-cell pillar layer as the design (3 × 200 × 200 × 10 E edges);
- a TF/SF box of 6 × 200² × 2 support rows;
- Nt = 4000.

The bridge's own estimator gives 92.7 GB of stored float32 increments for the WDM at r = 1. That is not the 34 GB quoted in the task, which presumably came from other run settings.

**Reading:**
- **WDM.** The rung that makes it fit on one machine without approximation is `reverse`: 2.9 GB, no recomputation, provided the problem is lossless with CPML and the sources are mode or current sources. Otherwise `checkpoint` takes 4.3 GB at 2.4 extra forward passes.
- **3D lens.** `checkpoint` is the right rung: 6.5 GB at 2 extra passes, or 13 GB at 1. `reverse` pays for recording the TF/SF face currents every step.

### Memory vs recompute time, measured (`bend_cpu.png`, `bend_gpu.png`)
![bend cpu](bend_cpu.png)

The test problem is a 2D bend of ceviche-bend size: 313² cells, CPML, Nt = 20 000, run on one core of the g2-standard-8 VM. The design region has 77 120 design edges, because MaterialGrid also selects Ex and Ey. The baseline is `store` with float64 rows, at 398 s for forward + gradient. "Time" below is forward + gradient wall time divided by that baseline.

| rung | rung memory (predicted) | tracemalloc peak | time / store | gradient vs store |
|---|---|---|---|---|
| store f64 | 12.34 GB | 12.44 GB | 1.00 | (reference) |
| store f32 | 6.17 GB | 6.27 GB | 0.97 | 1.1e-9 |
| checkpoint s=2 | 2.07 GB | 2.18 GB | 1.62 | bitwise |
| checkpoint s=4 | 0.85 GB | 0.96 GB | 1.76 | bitwise |
| checkpoint s=8 | 0.34 GB | 0.44 GB | 1.83 | bitwise |
| checkpoint s=16 | 0.20 GB | 0.30 GB | 1.84 | bitwise |
| checkpoint s=16, block 1 (pure revolve) | 0.12 GB | 0.22 GB | 3.57 | bitwise |
| auto, 1 GB budget | 0.99 GB | 1.09 GB | 1.49 | bitwise |
| auto, 0.25 GB budget | 0.25 GB | 0.35 GB | 1.93 | bitwise |
| reverse | 0.39 GB | 0.51 GB | 1.22 | 2.5e-15 |
| nyquist d=8 (approx.) | 1.54 GB | 1.64 GB | 1.00 | 1.0e-4 |
| nyquist d=32 (approx.) | 0.39 GB | 0.49 GB | 1.00 | 6.5e-4 |

- Measured peak = predicted + 0.10 GB in every row; the 0.10 GB is the solver and the monitors.
- With a 0.25–1 GB budget, checkpointing cuts the gradient memory 12–60× for 1.5–1.9× the wall time. Pure revolve (block = 1) needs 3.6× for a further 1.6× memory cut, so blocks matter when a design row is much smaller than a snapshot.
- `reverse` costs 1.22× (one reversed step per adjoint step).
- **GPU** (`bend_gpu.png`, Nt = 5000, CuPy pool peak via a memory hook):
  - The measured device peak is the predicted rung bytes plus 0.06–0.14 GB of solver arrays.
  - Gradients are bitwise for checkpointing and 1.5e-15 for reverse.
- **Side finding:** the L4 runs this 98k-cell 2D CPML problem at about 0.1 s/step, 10× slower than one CPU core (GPU utilization ~1 %). The GPU step is latency-bound on small 2D grids. This is unrelated to the ladder and is reported for the GPU-performance work.
