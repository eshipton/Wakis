# ⏱️ Performance

Forward and adjoint cost per time step of `SolverFIT3D` on CPU, GPU and MPI, measured with the scripts in `benchmarks/`:

```bash
python benchmarks/bench_timestep.py --sizes 64 128 192            # CPU
python benchmarks/bench_timestep.py --sizes 64 128 192 --gpu      # GPU
mpiexec -n 4 python benchmarks/bench_timestep.py --sizes 128 --mpi
python benchmarks/profile_step.py --n 128 --steps 30 [--gpu]      # cProfile (+ CuPy kernel timings)
```

**Case.** A vacuum PEC box of $n^3$ cells (1 m side), a `CurrentSource` with a `GaussianSource`, a `DFTMonitor` plane ($E_z$, $H_y$) and a `FluxMonitor` plane of $0.8\times0.8$ m (3 frequencies each); for the full adjoint run a `MaterialGrid` cube of 0.2 m ($5^3$ density nodes; 6 084, 48 750 and 173 394 design edges at $n$ = 64, 128, 192) and a flux objective. Reported quantities:

- **Mcells/s**: $n^3$ divided by the time of one step of the `emsolve` loop (source update, `one_step`, monitor updates);
- **adjoint/forward**: time of `AdjointStepper.step` over `one_step` (bare transposed step), and time per step of `AdjointSimulation.vjp` over that of `AdjointSimulation.forward` (with stored design increments, monitor adjoint sources and gradient accumulation; solver construction excluded);
- **memory**: CuPy pool bytes in use (GPU) or resident memory summed over ranks (CPU), after building the solver and after building the `AdjointStepper` (which adds the transposed operators $\mathbf A^H$, $\mathbf B^H$).

**Machine.** Google Cloud g2-standard-16: 16 vCPU (Intel Xeon @ 2.20 GHz, 8 physical cores), 64 GB RAM, one NVIDIA L4 (24 GB, ~300 GB/s), CUDA 12.9, CuPy 14.2, NumPy/SciPy from PyPI, Python 3.12, Open MPI 4.1. CPU runs use one thread (`OMP_NUM_THREADS=1`; SciPy sparse products are single-threaded) except the MKL row, which uses `sparse_dot_mkl` with all cores. All runs in float64. Times are means over 20 (CPU) or 50 (GPU) steps after a warm-up step; repeated runs on this shared cloud VM vary by up to ~±20 % on CPU (e.g. the bare CPU adjoint/forward step ratio at $128^3$ is reproducibly 1.57 while it is ~1 at $64^3$ and $192^3$; we did not find the cause). The single-core CPU rows were measured with the final code; the MKL, MPI, CPML and GPU `AdjointSimulation` columns were measured before the last two changes (in-place transposed step, faster `MaterialGrid.apply` on CPU), which only shorten the adjoint and forward runs (GPU $128^3$ re-run: 318 Mcells/s, ratios 1.15 / 1.10).

## Results

| Backend | $n$ | Mcells/s (develop) | Mcells/s | adjoint/forward (step) | adjoint/forward (`AdjointSimulation`) | memory solver → + adjoint [GB] |
|---|---|---|---|---|---|---|
| CPU, NumPy/SciPy, 1 core | 64 | 11.1 | 19.1 | 1.04 | 1.19 | 0.62 → 0.73 |
| | 128 | 6.9 | 14.2 | 1.57 | 1.03 | 2.13 → 2.73 |
| | 192 | 5.4 | 8.3 | 0.90 | 1.05 | 6.23 → 8.51 |
| CPU, MKL (`sparse_dot_mkl`), 8 cores | 64 | – | 29.6 | 0.75 | 0.99 | 0.47 → 0.58 |
| | 128 | – | 17.1 | 0.70 | 1.07 | 2.34 → 2.96 |
| | 192 | – | 16.3 | 0.64 | 0.73 | 8.99 → 11.3 |
| GPU, NVIDIA L4 | 64 | 151 | 185 | 1.10 | 1.93 | 0.12 → 0.25 |
| | 128 | 169 | 318 | 1.09 | 1.01 | 1.21 → 2.22 |
| | 192 | 165 | 319 | 1.09 | 1.07 | 5.19 → 8.59 |
| CPU CPML (8 cells) | 128 | – | 7.8 | 1.32 | 1.32 | 2.07 → 2.76 |
| GPU CPML (8 cells) | 128 | – | 188 | 1.02 | 1.30 | 1.31 → 2.38 |

MPI ($z$ slabs, one core per rank, NumPy/SciPy):

| ranks | Mcells/s, $128^3$ | speed-up | adjoint/forward (step / sim) | Mcells/s, $192^3$ | speed-up | adjoint/forward (step / sim) | memory $192^3$ (all ranks) [GB] |
|---|---|---|---|---|---|---|---|
| 1 | 13.9 | 1.0 | 1.96 / 0.98 | 8.5 | 1.0 | 1.17 / 1.03 | 8.7 → 11.1 |
| 2 | 26.5 | 1.9 | 1.30 / 0.94 | 16.4 | 1.9 | 1.17 / 1.03 | 9.4 → 11.7 |
| 4 | 48.1 | 3.5 | 1.15 / 0.92 | 31.5 | 3.7 | 1.20 / 0.87 | 10.8 → 12.8 |
| 8 | 71.8 | 5.2 | 1.11 / 0.92 | 75.9 | 8.9 | 1.19 / 0.89 | 13.0 → 16.0 |

**Reading the numbers.**

- One adjoint step costs about one forward step on every backend (the transposed step does the same two sparse products and a few more vector operations; the MKL adjoint is faster than the MKL forward because the forward MKL path still copies full fields). A full gradient (forward with stored increments plus adjoint) therefore costs about 2–2.3 forward runs. The larger ratios at $64^3$ on the GPU come from per-step kernel-launch overheads (monitor and adjoint-source updates), which dominate small grids: the bare step takes 0.37 ms there, the loop with source and monitors 1.4 ms.
- The GPU runs ~320 Mcells/s from $128^3$ up, 25–40× the single-core CPU and ~20× the 8-core MKL run. The two sparse matrix-vector products take 4.9 ms of the 6.5 ms step at $128^3$ (cuSPARSE CSR, 2.4 ms each, bandwidth bound); a matrix-free stencil kernel would be the next step for a large gain (not implemented).
- MPI scales with 89 % efficiency at 8 ranks for $192^3$; at $128^3$ the ghost exchanges (3 per forward step, 3 reversed per adjoint step) weigh more. Each rank stores its own operators, so memory grows slightly with the rank count (ghost planes, Python/MPI runtime per rank).
- Memory is dominated by the sparse operators: ~730 B per cell for the forward GPU solver and ~1.2 kB per cell with the adjoint's transposed operators, so a $256^3$ adjoint run (16.8 Mcells) does not fit the 24 GB L4 (out of memory at 20 GB), while it does fit with MPI over several GPUs or on the CPU. The stored increments add $N_t \times N_\text{design} \times 8$ bytes (e.g. 2 000 steps × 173 394 edges = 2.8 GB; half with `store_dtype=np.float32`, or kept in host memory with `store_on_host`).

## Optimizations found by profiling

cProfile (CPU) and `cupyx.profiler.benchmark` (GPU) of the forward and adjoint loops (`benchmarks/profile_step.py`) showed, on the develop version:

1. **Sparse matrix copied every step.** `H - dt * A * E` evaluates `dt * A` first: SciPy and CuPy build a scaled copy of the whole operator (data and indices) each step, ~16 % of the CPU step at $64^3$. `SolverFIT3D._update_H` now computes `H -= dt * (A @ E)` and `_update_E` computes `E += dt * (B @ H - ieps * J)` in place, with the same arithmetic for E and without full-size copies (`Field.fromarray`).
2. **GPU: fused updates and CSR.** On the GPU both updates are single CuPy elementwise kernels, and the operators are stored as CSR instead of CSC (cuSPARSE runs the CSR product directly; 2.4 ms instead of 2.8 ms at $128^3$). Step time at $128^3$: 12.3 ms → 6.5 ms.
3. **GPU: host-to-device copies in the time loop.** `DFTMonitor`, `FluxMonitor` (and `Near2FarMonitor`) and `ModeMonitor` computed their DFT kernels with NumPy and copied them to the GPU every step; each copy synchronizes the device. The kernels are now computed on the device. Magnetic currents copied the whole $\mu^{-1}$ array to the GPU every step; `imu` is now moved to the GPU with the other material arrays. With MPI, GPU fields were moved entirely to the host and back for every ghost exchange (3 per step); only the exchanged $z$-planes are copied now (`SolverFIT3D._mpi_communicate_gpu`).
4. **Adjoint.** Monitor adjoint sources touch only the sampled entries ($O(m)$ instead of a full-size sparse product and vector addition per monitor term and step), the transposed step works in place, no auxiliary arrays are allocated without lumped or dispersive corrections, and on CPU with MKL the transposed operators use MKL as well.

These changes are covered by the existing regression and validation tests (CERN reference data, analytic resonators, CPML, sources and monitors; all non-slow tests pass on CPU and the GPU-enabled ones with `--gpu`) and by `tests/test_020_adjoint.py::test_gpu_matches_cpu`, which checks that GPU outputs and gradients equal the CPU ones to $10^{-10}$ relative.

**Subnormal numbers.** The CPU sparse product is ~15× slower when its input holds subnormal floats (91 ms vs 1.35 s at $128^3$ for an input filled with $10^{-310}$). Fields can decay into the subnormal range ahead of a wavefront or in long ring-downs; if a CPU run slows down for no other reason, this is a likely cause.


## Estimating memory and run time before launching

Simulations grow quickly with resolution (cells scale as the cube of the cells per wavelength in 3D),
so estimate resources before launching, locally or in the cloud:

```bash
python -m wakis.estimate --shape 400,400,400 --steps 8000 --backend gpu --cpml --machine g2-standard-32
# -> peak host RAM 76.6 GB, peak GPU memory 59.0 GB ... DOES NOT FIT: GPU memory > 90% of 24 GB
```

```python
from wakis.estimate import estimate, preflight
est = estimate((Nx, Ny, Nz), steps=Nt, backend="gpu", cpml=True,
               monitors=[(n_freq, n_samples, n_components)],
               adjoint={"design_edges": n_design, "store_bytes": 4})
preflight(est, "g2-standard-16")   # raises MemoryError if it does not fit
```

The base cost was calibrated with `benchmarks/calibrate_memory.py` (48 runs, 16^3 to 128^3 cells, GCP
g2-standard-16 with an NVIDIA L4): peak host RAM is about 0.4 GB + 1.1 kB/cell (CPU, float64), 0.8 kB/cell
in float32; GPU memory about 0.9 kB/cell (float64). Held-out 160^3 runs agree within ~4 %; `estimate` adds a
25 % safety margin. Monitors, adjoint storage (steps x design edges) and dispersive state are added
analytically; complex (Bloch) fields are counted twice. Recalibrate after major solver changes.
