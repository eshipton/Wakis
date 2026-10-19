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


## Fused GPU time stepping (`fused=True`)

`SolverFIT3D(..., use_gpu=True, fused=True)` replaces the CuPy array expressions of the time loop by a handful of
hand-written kernels (`wakis/gpu_fused.py`). It is opt-in while it is validated; the CPU path and the default GPU
path are unchanged.

**Why.** The CuPy path issues one kernel (plus Python dispatch, ~10–20 µs) per array operation. A CPML step is
~330 launches (6 + 6 sparse curls, gathers/scatters of the 24 CPML faces, temporaries), a running DFT ~5 per
monitor term, a TF/SF source ~25: below a few million cells the GPU waits on the host, not on memory
(11–22 % utilization). Profiles and numbers: `studies/gpu-perf/README.md`.

**What the kernels do.**

- *Step* (two kernels, one per half step; one thread per edge $i$ of component $T$). The two derivative
  terms of the curl of $T$ are stored as one merged CSR row, split in the $a$ ($s=+1$) and $b$ ($s=-1$) parts,
  so the thread forms $p_a = (D_a E_{S_a})_i$, $p_b = (D_b E_{S_b})_i$, updates the CPML auxiliary variables
  it owns, $\psi_a \leftarrow b\,\psi_a + c\,p_a$ (same for $b$; each $\psi$ entry belongs to exactly one edge),
  and the field, $H_i \leftarrow H_i - \Delta t\,\mu_i^{-1}(p_a - p_b + \psi_a - \psi_b) - \Delta t\,\mu_i^{-1} M_i$.
  The E kernel adds the conduction current ($J \mathrel{+}= \sigma E - J_\text{old}$) and
  $E_i \leftarrow E_i + \Delta t\,\varepsilon_i^{-1}(q_a - q_b + \psi_a - \psi_b) - \Delta t\,\varepsilon_i^{-1} J_i$.
  Without CPML the kernels read the solver's own curl operators $\mathbf A$, $\mathbf B$ and compute
  $H \mathrel{-}= \Delta t\,\mathbf A E$, $E \mathrel{+}= \Delta t(\mathbf B H - \varepsilon^{-1} J)$ without the
  temporary curl vector.
- *Corrections*: tensor averaging (gather, increment, CSR product) and the ADE of each dispersive group
  (gather of $E^n$, one kernel for all poles) run as hooks around the step kernels.
- *Sources*: `CurrentSource`, `MagneticCurrentSource`, `ModeSource` add $w\,s(t) - J_\text{old}$ in one kernel each;
  `TFSFSource` advances its auxiliary line and applies its corrections in five.
- *Monitors*: `DFTMonitor`, `FluxMonitor`, `ModeMonitor` accumulate all terms and frequencies of a monitor in
  two kernels (phasors $\Delta t\,e^{-2\pi i f t}$ of each term, then $\text{acc} \mathrel{+}= \text{phasor}\cdot v$);
  their accumulators become views into one buffer, so `get_field`, `get_flux` and the adjoint readouts are
  unchanged.
- *Adjoint* (`AdjointStepper`, `AdjointSimulation` on a fused solver): the transposed step is four kernels
  (per half step: $\nu = \Delta t\,\varepsilon^{-1}\lambda_e$, the transposed $\psi$ update and
  $\beta = s\,\nu + c\,\tilde\mu$; then one product with the stored merged transpose
  $[\mathbf D_a^H\;\mathbf D_b^H]$), each adjoint DFT source two, the gradient accumulation one, and the
  forward design increments two (gather, scaled difference).
- *CUDA graphs*: when every per-step operation is device resident (fused step without the ABC wrapper,
  the sources and monitors above, no lumped elements), `emsolve` and `AdjointSimulation` capture 64 steps into a
  CUDA graph and replay it. Time-dependent scalars come from device tables (source amplitudes, TF/SF injection
  terms indexed by a device step counter), so nothing changes between replays. `solver.use_graphs = False`
  (or `AdjointSimulation.use_graphs`) keeps the per-step Python loop.

**Results** (NVIDIA L4, float64, uncontended; `studies/gpu-perf`, develop = 663536f):

| case | cells | develop GPU | fused (+ CUDA graphs) | speed-up | CPU 1 core |
|---|---|---|---|---|---|
| plain 3D PEC box | 275 k | 0.42 ms | 0.45 ms | 0.93× | 13.8 ms |
| plain 3D PEC box | 2.15 M | 6.47 ms | 4.59 ms | 1.41× | 147 ms |
| 3D CPML | 275 k | 7.33 ms | 0.68 ms | 10.8× | 36.5 ms |
| 2D rod, conformal `tensor` | 66 k | 0.73 ms | 0.057 ms | 12.7× | 2.3 ms |
| 3D CPML + ADE block | 275 k | 7.86 ms | 0.73 ms | 10.8× | 38.8 ms |
| 2D TF/SF + 2 flux boxes (81 f) + DFT plane | 44 k | 11.7 ms | 0.63 ms | 18.5× | 18.9 ms |
| `AdjointSimulation` 2D CPML: forward / vjp | 26 k | 5.97 / 5.73 ms | 0.101 / 0.053 ms | 59× / 108× | 1.91 / 1.97 ms |
| invrs-gym ceviche bend, forward + adjoint (56 587 steps each) | 98 k | 753 s | 15.3–16.6 s | 45–49× | – |

The CPML, monitor- and adjoint-heavy runs were launch-bound (11–22 % GPU utilization) and are now
bandwidth-bound (nsys kernel time ≈ wall time; bend 93 % utilization). The plain step was already
GPU-bound: there the fused curl kernels are 7 % slower than cuSPARSE at 0.3 Mcells and 1.4× faster at
2 Mcells (no temporary curl vector). `wakis.estimate.estimate(..., backend="gpu", fused=True)` uses these
measured rates (no CPML launch floor).

**Exactness.** The fused step performs the same operations as the CuPy path except for the order of a few
floating-point additions (curl rows summed in CSR order, CPML terms added in one expression): fields, DFT and
flux spectra agree to $10^{-11}$ relative after a few hundred steps on PEC, lossy, periodic, CPML, mixed and
Bloch (complex) cases. Source updates, the running DFT and the TF/SF line reproduce the CuPy expressions bit
for bit (kernels compiled without FMA contraction), and the graph replay is bitwise the per-step fused loop.
The fused transposed step is the exact transpose of the fused forward step: the dot-product test gives
$<10^{-12}$ relative on all cases, including tensor-averaged and dispersive solvers (also with states
supported on $\psi$ only), fails for a 0.1 % change of one
transposed entry, of the transposed $\psi$ coefficient or a $10^{-6}$ change of the loss factor, and gradients
of `AdjointSimulation` agree with the CuPy path to $10^{-10}$ (round-off growth over 250 steps) and with central
finite differences (`tests/test_045_gpu_fused.py`).

**Limits & failure modes.**

- Fused kernels: GPU, float64 or complex128 fields, no MPI (`fused=True` is ignored otherwise, with a
  message). Legacy in-step TF/SF injection (`source_type='tfsf'` with an active `Beam`) falls back to the CuPy
  step. Operators above $2^{31}$ nonzeros are rejected (32-bit indices).
- Tensor averaging (`conformal='tensor'`) and ADE dispersive groups (real fields) run as fused kernels
  around the step and keep the CUDA graphs. The Mur ABC, complex-field ADE and the dispersive tensor network
  still wrap the step with CuPy operations, and lumped elements, `Near2FarMonitor`, `SliceRecorder`,
  plotting, saving and callbacks in `emsolve` keep the per-step Python loop (with the fused step).
- The CPML $\psi$ arrays of the solver are views into fused buffers: arrays assigned to `solver.psi*` are copied
  in at the next step. `AdjointStepper.step` on a fused solver returns arrays from two alternating buffers,
  overwritten two calls later (copy them to keep them).
- CUDA graphs bake device pointers: reallocating solver arrays inside a graph loop is detected and raises.
- Memory: a fused CPML solver keeps merged curl rows, $\psi$ slot maps and buffers next to the solver's
  operators (~380 B per cell, +60 % GPU memory at 275 kcells); the adjoint adds the transposed rows.
- The DFT accumulation is $O(N_f N_\text{samples})$ work per step like before; for very large monitors
  (> ~10⁷ accumulator entries) the GPU time of the monitors, not the launches, dominates.


## Conformal setup

The conformal geometry (`ConformalGeometry`: open edge lengths and face areas inside PEC, dual volume fractions, interface normals), the merged-face operators, the averaged materials, the tensor couplings and the stable time step are computed once when `SolverFIT3D` is built. Measured with `benchmarks/bench_conformal_setup.py` (fresh process per point, same g2-standard-16 VM, shared with other jobs, CPU only):

```bash
python benchmarks/bench_conformal_setup.py --sizes 64,100,128,160 [--profile 30]
python benchmarks/conformal_regression.py --save ref.npz --n 24     # with the old code
python benchmarks/conformal_regression.py --compare ref.npz --n 24  # with the new code
```

Cases in a 0.2 m box: `sphere_pec` (vacuum sphere cavity, radius 83 mm, in a PEC background, `conformal='all'`), `sphere_tensor` (eps_r = 4 sphere, radius 60 mm, `'tensor'`), `box` (rotated PEC box, `'pec'`), `mixed` (tilted dielectric cylinder, PEC ball and a thin sheet, `'all'`). "Conformal" is the time of `apply_conformal_materials` + `apply_conformal_timestep` (everything the conformal option adds), "solver" the whole `SolverFIT3D` construction; develop = c7a4c6d.

| case | cells | conformal develop → now [s] | speed-up | solver develop → now [s] | `GridFIT3D` [s] | peak RSS develop → now [GB] |
|---|---|---|---|---|---|---|
| sphere_tensor | 0.27 M | 7.9 → 2.2 | 3.7× | 9.8 → 2.7 | 1.0 | 0.88 → 0.83 |
| | 1.03 M | 29.0 → 5.5 | 5.3× | 33.9 → 7.4 | 3.5 | 2.06 → 1.91 |
| | 2.15 M | 84.0 → 9.9 | 8.5× | 93.9 → 14.3 | 6.0 | 3.65 → 3.44 |
| | 4.17 M | 156.5 → 16.2 | 9.7× | 173.7 → 25.2 | 12.0 | 6.71 → 6.32 |
| sphere_pec | 1.03 M | 14.4 → 4.1 | 3.5× | 15.9 → 5.6 | 8.0 | 1.99 → 1.66 |
| | 2.15 M | 27.3 → 6.6 | 4.1× | 30.7 → 10.1 | 14.4 | 3.63 → 2.88 |
| | 4.17 M | 52.4 → 11.0 | 4.8× | 59.4 → 17.7 | 28.5 | 6.61 → 5.24 |
| box | 1.03 M | 10.4 → 3.2 | 3.2× | 12.2 → 5.0 | 0.8 | 1.93 → 1.76 |
| | 2.15 M | 22.3 → 5.3 | 4.2× | 26.3 → 9.4 | 1.7 | 3.58 → 3.13 |
| | 4.17 M | 66.7 → 11.9 | 5.6× | 74.7 → 20.0 | 3.3 | 6.55 → 5.70 |
| mixed | 1.03 M | 9.9 → 3.8 | 2.6× | 11.9 → 5.9 | 4.0 | 2.07 → 1.83 |
| | 2.15 M | 15.6 → 6.6 | 2.4× | 20.3 → 11.3 | 7.4 | 3.78 → 3.32 |
| | 4.17 M | 39.0 → 11.9 | 3.3× | 48.2 → 20.9 | 14.5 | 6.63 → 6.01 |

All metric, material and operator arrays are bitwise identical to develop for the four cases at 14³, 24³ and 48³ (`tests/test_028_conformal_setup_regression.py` keeps the 14³ data). The stable time step is an iterative estimate (residual tolerance $10^{-3}$ plus a 1 % margin); it differs from develop by $<10^{-4}$ relative, the same order as the scatter of the previous ARPACK estimate between runs ($\sim4\times10^{-5}$).

What changed (profiling with cProfile and wall-time wrappers):

1. **Stable time step** (was 60–95 % of the conformal time). ARPACK (`eigsh`; Arnoldi `eigs` for `'tensor'`, ~120 products) on all $3N$ edges is replaced by a Lanczos recurrence for $\mathbf P\mathbf Q$ in the $\mathbf P$-inner product (physics guide, *Tensor subpixel averaging*), symmetric also with the tensor couplings, on the edges with $P_{ee} > 0$ only (e.g. outside a PEC background), with five stored vectors instead of a 20-vector Krylov basis, in-place multi-threaded CSR products (row blocks, same per-row summation order) and BLAS vector updates. The start vector adds the grid's Nyquist mode (top eigenvector of the uniform curl-curl operator) to the random vector: 36 instead of 102 iterations for `sphere_tensor` at 2.15 M cells. On the GPU (`use_gpu=True`) the iterations run with CuPy (2.5 s instead of 5.6 s for that case). Threads: `OMP_NUM_THREADS`, else the CPU count, at most 8; one per MPI rank.
2. **Tensor assembly** was run three times per setup (time step, `install`, `install` after the dispersive edges); it is reused when its inputs are unchanged, and `np.unique(..., axis=0)` is replaced by a 1D key with the same order.
3. **Plane sections** (`_section_areas`, used for the PEC metrics and the volume fractions): the planes are processed in blocks on threads, each restricted to the node range of its segments; only crossed grid lines are accumulated; the x and y edges share their sample planes.
4. **Merged faces** (`_redistribution`): the stability test runs only on faces with $A_\text{open} < 0.25\,A_\text{full}$ and an open edge, and $\mathbf G = \mathbf T^T\mathbf D\mathbf T$ is assembled from its identity and unstable-column parts instead of two full-size sparse products.
5. **Interface normals**: one closest-cell search for the three components.

The remaining time is spread over the closest-cell search of the normals (VTK, one Python call per cut edge: 2.2 s for 2.3×10⁵ edges), the dual volume fractions and PEC metrics (1–2 s at 4 M cells), and the dense material and operator arrays. The STL cell masks of `GridFIT3D` (`select_interior_points`, not part of the conformal code) now take longer than the conformal setup for large solids (28 s for `sphere_pec` at 4.17 M cells); restricting them to the bounding box of each solid would not be bitwise identical (VTK's ray directions depend on the point index).

## Estimating memory and run time before launching

Simulations grow quickly with resolution (cells scale as the cube of the cells per wavelength in 3D),
so estimate resources before launching, locally or in the cloud:

```bash
python -m wakis.estimate --shape 400,400,400 --steps 8000 --backend gpu --cpml --machine g2-standard-32
# -> peak host RAM 75.93 GB, peak GPU memory 58.88 GB, run time 55.5 min (setup 131 s, 400 ms/step)
#    DOES NOT FIT: GPU memory 58.9 GB > 90% of 24 GB on g2-standard-32
python -m wakis.estimate --shape 128,128,128 --steps 5000 --conformal tensor --machine g2-standard-16
# -> peak host RAM 4.34 GB, run time 13.0 min (setup 19 s, 152.56 ms/step) ... FITS
```

```python
from wakis.estimate import estimate, estimate_unstructured, interface_cells_from_area, preflight
est = estimate((Nx, Ny, Nz), steps=Nt, backend="gpu", cpml=True,
               monitors=[(n_freq, n_samples, n_components)],
               adjoint={"design_edges": n_design, "store_bytes": 4},
               conformal="tensor",                      # or 'pec', 'all'
               interface_cells=interface_cells_from_area(area_m2, h, "tensor"),
               dispersive_edges=n_edges, n_poles=2)
print(est)            # peak host / GPU memory, setup time, time per step, breakdown
preflight(est, "g2-standard-16")   # raises MemoryError if it does not fit
estimate_unstructured(n_tets, steps=Nt, eigen=False)     # tetrahedral FIT
```

**Model.** Peak host RAM = fixed + per-cell cost of the solver (CPU/GPU, PEC/CPML, float64/float32;
complex fields twice), or for conformal grids the larger of that and the **setup peak**
fixed + per cell + per interface cell (geometry, merged faces and Lanczos time step; it dominates up to
at least 4 Mcells), plus monitors, adjoint storage and dispersive state; filling many dispersive edges
adds a transient of 160 B per edge on top of the steady solver. Interface cells are the cells holding an
E edge whose length (PEC) or dual cell (dielectric) is cut by a surface, about 1.1 (PEC) or 1.9
(dielectric) times the surface area over $h^2$ (`interface_cells_from_area`). The time per step is
interpolated in a measured throughput table per backend (one CPU core with SciPy, the L4 GPU; plain and
conformal grids; CPML with a 7 ms launch floor on the GPU), plus 12 + 25 $n_\text{poles}$ ns per
dispersive edge; the setup time is linear in cells and interface cells (it includes the STL masks of
`GridFIT3D`, which scale with the triangle count; calibrated with a 28 800-triangle sphere). The
unstructured solver costs 0.40 GB (`import wakis`) + 2.5 kB and 52 ns per tetrahedron and step, plus the
sparse-LU fill-in of an eigen-solve (`wakis.unstructured.estimate`, not re-measured). `estimate` adds a
25 % safety margin to the memory.

**Calibration** (`benchmarks/calibrate_memory.py --suite plain|conformal|dispersive|unstructured`, fresh
process per run, 20 warmed-up steps; g2-standard-16 VM shared with other jobs): fits on 32³, 64³, 96³,
128³ and 160³ grids (unstructured: 2×10⁴–3.7×10⁵ tetrahedra), validated on held-out runs
(48³, 80³, 112³, 136³, 144³, 176³, 200³; conformal 'pec'/'all'/'tensor' with a dielectric or PEC sphere,
also on the GPU; Drude/Lorentz fills of 10–50 % with 1–3 poles) with `benchmarks/validate_estimate.py`:

| quantity (78 held-out runs) | median error | max error | within 10 % | within 30 % |
|---|---|---|---|---|
| peak host RAM | 0.9 % | 9.9 % (dispersive, 50 % fill) | 78 | 78 |
| peak GPU memory (30 runs) | 0.7 % | 6.1 % | 30 | 30 |
| time per step | 8.1 % | 44 % | 47 | 72 |
| setup time (76 runs) | 11.8 % | 52 % | 34 | 68 |

The step-time outliers are single slow runs on the shared VM (a conformal 112³ run at 8 instead of
14 Mcells/s between two normal neighbours), the CPML CPU rate beyond the calibrated range (4.4 Mcells/s
at 5.5–8 Mcells, estimated 6.8: −35 %) and one GPU CPML run in the launch-bound regime. The previous
model (one throughput per backend: 8, 16 and 75 Mcells/s) overestimated the time per step by 1.8× on the
CPU and 2–4× on the GPU at 1–4 Mcells; its memory model (plain grids only) missed the conformal setup
peak by 12–20 % at these sizes (more for large, finely triangulated STL solids, whose VTK masks are not
modelled). Recalibrate after major solver changes.

**Observation.** At 3–8 Mcells a plain-grid CPU step runs at 8.2 Mcells/s while a conformal grid with the
same operators (same CSR layout and nonzeros) runs at 13–15 Mcells/s; the two sparse products take the same
time (0.18 s each at 4.2 Mcells), so the difference lies in the rest of the plain step (not investigated).
