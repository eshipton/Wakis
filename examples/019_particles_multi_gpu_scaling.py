# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

r"""Multi-GPU and MPI scaling benchmark for ``wakis.particles``.

Benchmarks the real coupled electromagnetic FIT + relativistic Boris/Esirkepov
particle-in-cell loop across:

1. **1 L4, this code, size=1** (``use_gpu=True, use_mpi=False``) — single-GPU
   baseline on the exact same grid, boundary conditions, and global particle set.
2. **2 L4 GPUs with NCCL** (``use_gpu=True, use_mpi=True, size=2,
   particles_comm="nccl"``) — GPU-to-GPU NCCL point-to-point communication for
   field ghosts, Esirkepov current guard reduction, and particle migration.
3. **2 L4 GPUs with pinned-host MPI fallback** (``use_gpu=True, use_mpi=True,
   size=2, particles_comm="mpi"``) — explicit device-to-pinned-host staging +
   ``MPI_Isend`` / ``MPI_Irecv`` + pinned-host-to-device copy.
4. **CPU MPI scaling** (``use_gpu=False, size in {1, 2, 4}``) at
   :math:`N_p \in \{10^5, 10^6\}`.

Equal-work benchmark design
---------------------------
- **Identical boundary conditions**: every configuration (``size=1``, ``size=2``,
  ``size=4``, CPU and GPU) uses ``bc_low=["periodic", "periodic", "pec"]`` and
  ``bc_high=["periodic", "periodic", "pec"]`` on the same :math:`64 \times 64
  \times 128` grid (:math:`5.24 \times 10^5` cells globally).
- **Identical global particle set**: for each total particle count :math:`N_p`,
  macroparticles are drawn from a single global pseudo-random stream
  (``np.random.default_rng(SEED)``) and then filtered per rank by its owned
  :math:`z`-slab :math:`[z_{\mathrm{own,min}}, z_{\mathrm{own,max}})`. The union
  of initial particles across all MPI ranks is therefore bit-for-bit identical
  for ``size=1``, ``size=2``, and ``size=4``.
- **Best-case load balance**: the uniform spatial particle distribution across
  :math:`z` assigns :math:`\approx N_p / \mathrm{size}` particles to every equal-height
  slab and represents the best case for parallel load balance (as opposed to a
  localized bunch concentrated inside a single slab).
- **Timing protocol**: each benchmark point executes ``n_warmup >= 3`` warm-up
  steps (including the initial half-step velocity back-push on step 0) followed
  by ``n_repeats >= 5`` independent repeats of ``n_steps >= 10`` unprofiled
  ``solver.one_step()`` calls, reporting the **median**, **min**, and **max**
  step wall time and throughput.
- **Separate profiled stage breakdown**: per-stage timings (Boris push +
  Esirkepov deposit, particle migration, guard current reduction, FIT :math:`B`
  update, FIT :math:`E` update, and FIT field ghost exchange) are measured in a
  **separate** pass with ``ps.profile = True`` and ``solver._profile_mpi = True``.
  Because per-stage timing inserts ``cudaStreamSynchronize()`` barriers around
  each stage, the profiled stage sum is inflated relative to the unprofiled
  coupled step time.

Usage
-----
Run each MPI rank count (or run the single-process entry point, which also
spawns the ``mpiexec`` passes automatically when permitted by the environment)::

    python examples/019_particles_multi_gpu_scaling.py
    mpiexec --allow-run-as-root --oversubscribe -n 2 python examples/019_particles_multi_gpu_scaling.py
    mpiexec --allow-run-as-root --oversubscribe -n 4 python examples/019_particles_multi_gpu_scaling.py --cpu-only
    python examples/019_particles_multi_gpu_scaling.py --plot-only

Run a quick smoke check on the smallest particle count::

    python examples/019_particles_multi_gpu_scaling.py --smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import warnings

# Pin host thread pools before importing NumPy/SciPy so CPU MPI ranks do not
# oversubscribe cores or contend in OpenMP/BLAS thread pools.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from scipy.constants import c, e as q_e, epsilon_0, m_e

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wakis import GridFIT3D, SolverFIT3D
from wakis.estimate import estimate as wakis_estimate, preflight as wakis_preflight
from wakis.particles import ParticleSolver, Species

DATA_PATH = REPO_ROOT / "examples" / "019_particles_multi_gpu_scaling.json"
FIG_SCALING_PATH = REPO_ROOT / "docs" / "img" / "particles_multi_gpu_scaling.png"
FIG_BREAKDOWN_PATH = (
    REPO_ROOT / "docs" / "img" / "particles_multi_gpu_stage_breakdown.png"
)
FIG_WORKING_SET_PATH = (
    REPO_ROOT / "docs" / "img" / "particles_multi_gpu_working_set.png"
)

PREFLIGHT_MACHINE = "g2-standard-24"
BASELINE_LABEL = "1 L4, this code, size=1"
SEED = 202603

DEFAULT_NX = 64
DEFAULT_NY = 64
DEFAULT_NZ = 128
DEFAULT_N_WARMUP = 3
DEFAULT_N_REPEATS = 5
DEFAULT_N_STEPS = 10
# Sorted-control periodic cell-sorting interval (matching the ParticleSolver
# library default sort_every so amortised sort overhead is measured over at
# least one full sort period; recorded in configuration["control_sort_every"]).
CONTROL_SORT_EVERY = 50

GPU_SWEEP_COUNTS = [100_000, 1_000_000, 10_000_000, 50_000_000]
CPU_SWEEP_COUNTS = [100_000, 1_000_000]
BREAKDOWN_N_PARTICLES = 10_000_000
WORKING_SET_SWEEP_NZ = [16, 32, 48, 64, 96, 128, 192, 256]
WORKING_SET_SWEEP_EXECUTION_ORDER_NZ = [16, 256, 32, 192, 48, 128, 64, 96]
WORKING_SET_SWEEP_DUPLICATE_NZ = [128, 64]
# Pre-registered 3x multiplier over the within-sweep back-to-back duplicate
# spreads (unsorted noise_within_sweep_ns_per_particle and sorted
# noise_within_sweep_sorted_ns_per_particle):
# (i) an unsorted L2-bracketing jump counts as a resolved step only if it
#     exceeds k * noise_within_sweep_ns_per_particle and the neighbour
#     log-slope jumps;
# (ii) the cell-sorted curve counts as flat over the rising regime iff its span
#      over the rising regime is at most
#      k * noise_within_sweep_sorted_ns_per_particle; and
# (iii) the onset of the sustained unsorted rise is the first interval i such
#       that every jump from i onward is > -k * noise_within_sweep_ns_per_particle
#       (a monotone tail up to noise) and the jump at i is
#       > +k * noise_within_sweep_ns_per_particle.
L2_STEP_NOISE_FACTOR_K = 3.0
N_SORT_SAMPLES = 5
GLOBAL_BATCH_SIZE = 5_000_000

STAGE_KEYS = (
    "boris_push_deposit_s",
    "particle_migration_s",
    "guard_current_sum_s",
    "fit_b_advance_s",
    "fit_e_advance_s",
    "fit_ghost_comm_s",
)


def _fmt_sig2(val: float) -> str:
    """Format a floating-point value to 2 significant figures without trailing dot."""
    v = float(val)
    if v == 0.0:
        return "0.0"
    if 100.0 <= abs(v) < 1000.0:
        return f"{round(v, -1):.0f}"
    s = f"{v:#.2g}"
    if s.endswith("."):
        s = s[:-1]
    return s


def _fmt_pct0(pct: float, *, signed: bool = False) -> str:
    """Format a percentage to 0 decimal places."""
    v = float(pct)
    if abs(v) < 0.5:
        v = 0.0
    return f"{v:+.0f}%" if signed else f"{v:.0f}%"


def compute_field_current_working_set_bytes(
    solver: SolverFIT3D, ps: ParticleSolver
) -> tuple[int, int, dict[str, int]]:
    """Compute per-rank field and current working-set bytes from live solver arrays.

    Inspects the actual device/host arrays touched during a coupled particle +
    FIT step (`E`, `H`, `H_prev`, time-centred `B_n`, Esirkepov flux buffers
    `Wx/Wy/Wz`, `J_particles`, `J_prev`, `solver.J`, and `itA`) and sums
    `arr.size * arr.dtype.itemsize` deduplicated by underlying buffer pointer.

    The push+deposit kernel working set (`hot_names`) contains only the field
    and current arrays directly accessed by the per-particle GPU kernels:
    - `gather_and_push_kernel` (`wakis/particles/gpu_kernels.py:398-511`,
      launched at `wakis/particles/solver.py:2390-2405`) reads `E_flat`
      (`solver.E.toarray()`, components `Ex_f, Ey_f, Ez_f` at
      `wakis/particles/gpu_kernels.py:414, 445-447, 461-466`) and time-centred
      `B_flat` (`ps._B_n_buf`, components `Bx_f, By_f, Bz_f` at
      `wakis/particles/gpu_kernels.py:415, 448-450, 461-466`).
    - `deposit_esirkepov_kernel` (`wakis/particles/gpu_kernels.py:553-696`,
      launched at `wakis/particles/solver.py:2450-2471` after zeroing at
      `wakis/particles/solver.py:2344-2346`) writes particle current fluxes via
      `atomicAdd` into `Wx, Wy, Wz` (`ps._Wx_buf, ps._Wy_buf, ps._Wz_buf` at
      `wakis/particles/gpu_kernels.py:562-564, 634, 662, 690`).
    Cell-only arrays (`H_curr`, `H_prev`, `J_particles`, `J_prev`, `solver_J`,
    `itA`) are read/written by cell-wise kernels (`scale_deposited_current_kernel`
    at `wakis/particles/gpu_kernels.py:861-889`, `finalize_current_update_kernel`
    at `wakis/particles/gpu_kernels.py:892-960`, and FIT curl updates) and are
    included in `total_working_set_bytes` but excluded from the per-particle
    kernel working set.

    Parameters
    ----------
    solver : SolverFIT3D
        Active electromagnetic FIT solver.
    ps : ParticleSolver
        Attached particle solver.

    Returns
    -------
    total_working_set_bytes : int
        Total unique bytes across all field and current arrays touched per step [B].
    gather_deposit_working_set_bytes : int
        Unique bytes of arrays directly accessed by the gather/push and
        Esirkepov deposit kernels (`E`, `B_n`, `Wx`, `Wy`, `Wz`) [B].
    breakdown_bytes : dict of str to int
        Per-buffer byte counts [B].
    """

    def _buf_ptr_and_nbytes(arr) -> tuple[int, int]:
        if arr is None:
            return 0, 0
        nbytes = int(arr.size * arr.dtype.itemsize)
        if hasattr(arr, "data") and hasattr(arr.data, "ptr"):
            return int(arr.data.ptr), nbytes
        if hasattr(arr, "__array_interface__"):
            return int(arr.__array_interface__["data"][0]), nbytes
        return id(arr), nbytes

    named_arrays = [
        ("E", solver.E.toarray()),
        ("B_n", ps._B_n_buf),
        ("Wx", ps._Wx_buf),
        ("Wy", ps._Wy_buf),
        ("Wz", ps._Wz_buf),
        ("H_curr", solver.H.toarray()),
        ("H_prev", ps._H_prev if ps._H_prev is not None else getattr(solver, "_H_before_update", None)),
        ("J_particles", ps.J_particles),
        ("J_prev", ps._J_prev),
        ("solver_J", solver.J.toarray()),
        ("itA", ps._itA_arr),
    ]
    # Derived directly from `gather_and_push_kernel` (`wakis/particles/gpu_kernels.py:398-511`:
    # reads `E_flat` at l.414, 445-447, 461-466 and `B_flat` at l.415, 448-450, 461-466)
    # and `deposit_esirkepov_kernel` (`wakis/particles/gpu_kernels.py:553-696`: writes
    # `Wx, Wy, Wz` at l.562-564, 634, 662, 690).
    hot_names = {"E", "B_n", "Wx", "Wy", "Wz"}
    seen_total: set[int] = set()
    seen_hot: set[int] = set()
    total_bytes = 0
    hot_bytes = 0
    breakdown: dict[str, int] = {}

    for name, arr in named_arrays:
        ptr, nbytes = _buf_ptr_and_nbytes(arr)
        breakdown[name] = nbytes
        if nbytes > 0 and ptr not in seen_total:
            seen_total.add(ptr)
            total_bytes += nbytes
        if name in hot_names and nbytes > 0 and ptr not in seen_hot:
            seen_hot.add(ptr)
            hot_bytes += nbytes

    breakdown["W_dep_total"] = int(
        breakdown["Wx"] + breakdown["Wy"] + breakdown["Wz"]
    )
    return int(total_bytes), int(hot_bytes), breakdown


def species_bytes_per_particle(sp: Species) -> int:
    """Compute allocated bytes per particle slot directly from ``sp`` buffers.

    Inspects every array attribute on ``sp`` whose shape matches
    ``(sp._capacity,)`` and deduplicates by underlying buffer address so that
    aliased scratch buffers (such as ``_xi1_dep_scratch`` aliasing
    ``_xi1_scratch`` above ``sp.dep_scratch_alias_threshold``) are counted
    exactly once.

    Parameters
    ----------
    sp : Species
        Particle species instance.

    Returns
    -------
    int
        Allocated bytes per particle capacity slot [B/particle].
    """
    cap = int(sp._capacity)
    seen_ptrs: set[int] = set()
    bytes_per_slot = 0
    for val in sp.__dict__.values():
        if hasattr(val, "shape") and hasattr(val, "dtype") and val.shape == (cap,):
            if hasattr(val, "data") and hasattr(val.data, "ptr"):
                ptr = int(val.data.ptr)
            elif hasattr(val, "__array_interface__"):
                ptr = int(val.__array_interface__["data"][0])
            else:
                ptr = id(val)
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                bytes_per_slot += int(val.dtype.itemsize)
    return bytes_per_slot


def probe_species_bytes_per_particle(
    *,
    use_gpu: bool,
    use_mpi: bool,
    size: int,
    n_local_est: int,
    track_ids: bool = False,
    has_absorbing_boundary: bool = True,
    sort_every: int = 0,
) -> int:
    """Measure ``Species`` bytes per particle on a minimal 16-slot probe instance.

    Parameters
    ----------
    use_gpu : bool
        Whether the species resides on GPU.
    use_mpi : bool
        Whether MPI slab decomposition is active.
    size : int
        Number of MPI ranks.
    n_local_est : int
        Estimated local particle capacity on one rank.
    track_ids : bool, optional
        Whether 64-bit particle IDs are allocated. Default is ``False``.
    has_absorbing_boundary : bool, optional
        Whether any domain boundary or collector can absorb particles.
        Default is ``True`` (for ``pec`` z boundaries).
    sort_every : int, optional
        Cell-sorting period. Default is ``0`` (disabled).

    Returns
    -------
    int
        Allocated bytes per particle slot [B/particle] measured from ``Species``.
    """
    alias_thresh = (
        0
        if int(n_local_est) > Species.DEFAULT_DEP_SCRATCH_ALIAS_THRESHOLD
        else Species.DEFAULT_DEP_SCRATCH_ALIAS_THRESHOLD
    )
    probe = Species(
        name="probe",
        charge=-q_e,
        mass=m_e,
        capacity=16,
        track_ids=track_ids,
        use_gpu=use_gpu,
        dep_scratch_alias_threshold=alias_thresh,
    )
    if use_gpu:
        need_tmp = (
            bool(has_absorbing_boundary)
            or (bool(use_mpi) and int(size) > 1)
            or (int(sort_every) > 0)
        )
        probe._ensure_gpu_scratch(
            need_tmp=need_tmp,
            need_boundary=bool(has_absorbing_boundary),
            need_sort=int(sort_every) > 0,
        )
    bpp = species_bytes_per_particle(probe)
    del probe
    return bpp


def estimate_particle_and_solver_bytes(
    *,
    use_gpu: bool,
    use_mpi: bool,
    size: int,
    nx: int,
    ny: int,
    nz: int,
    n_particles: int,
    probe_bpp: int,
    sort_every: int = 0,
    capacity_headroom: float = 0.0,
) -> tuple[float, int]:
    """Compute conservative upper-bound bytes for ``Species`` and ``ParticleSolver`` buffers.

    Counts every capacity-allocated ``Species`` array (via ``probe_bpp``),
    the ``_scan_block_sums`` / ``_scan_block_offsets`` compaction arrays,
    the ``cp.argsort`` radix-sort workspace when ``sort_every > 0``, and all
    cell-sized ``ParticleSolver`` field/current/guard/migration buffers.

    Returns
    -------
    total_extra_bytes : float
        Total estimated bytes added by ``Species`` and ``ParticleSolver`` on one rank.
    cap_est : int
        Estimated per-rank ``Species`` capacity slot count.
    """
    n_mean = float(n_particles) / max(int(size), 1)
    if use_mpi and int(size) > 1:
        # 4-sigma binomial upper bound on slab population + capacity_headroom
        sigma = float(np.sqrt(max(n_mean * (1.0 - 1.0 / size), 1.0)))
        n_local_upper = int(np.ceil(n_mean + 4.0 * sigma))
        eff_headroom = max(float(capacity_headroom), 0.05)
        cap_est = max(
            int(np.ceil(n_local_upper * (1.0 + eff_headroom))),
            n_local_upper + 64,
            16,
        )
    else:
        cap_est = max(int(np.ceil(n_mean)), 16)

    # Sort workspace: cp.argsort(int64) allocates perm (8 B) + radix-sort buffers (24 B)
    sort_workspace_bpp = 32 if (use_gpu and int(sort_every) > 0) else 0
    n_blocks = max((cap_est + 255) // 256, 1)
    scan_bytes = 2 * n_blocks * 4 + 4096
    species_bytes = float(cap_est * (int(probe_bpp) + sort_workspace_bpp) + scan_bytes)

    nz_local = (int(nz) // max(int(size), 1)) + (
        2 if (use_mpi and int(size) > 1) else 0
    )
    nz_dep = nz_local + (1 if (use_mpi and int(size) > 1) else 0)
    n_cells_local = int(nx * ny * nz_local)
    n_cells_dep = int(nx * ny * nz_dep)

    if use_gpu:
        # J_particles (24B), _B_n_buf (24B), _J_prev (24B), _q0_buf (8B),
        # _itA_arr (24B), _H_prev (24B), _H_before_update (24B) = 152 B/cell
        # plus _Wx_buf, _Wy_buf, _Wz_buf = 24 B/dep_cell
        ps_cell_bytes = float(152 * n_cells_local + 24 * n_cells_dep)
        if use_mpi and int(size) > 1:
            mig_cap = max(4096, int(np.ceil(0.05 * cap_est)))
            guard_bytes = float(18 * nx * ny * 8 + 24 * nx * ny * 8)
            mig_bytes = float(4 * 8 * mig_cap * 8 + 4096)
            ps_mpi_bytes = guard_bytes + mig_bytes
        else:
            ps_mpi_bytes = 0.0
        # Include 1% arena rounding margin for 512B CuPy memory pool blocks
        total_extra = 1.01 * (species_bytes + ps_cell_bytes + ps_mpi_bytes)
    else:
        # Host ParticleSolver arrays: J_particles (24B), _B_n_buf (24B), _q0_buf (8B), _itA_arr (24B)
        ps_cell_bytes = float(80 * n_cells_local + 24 * n_cells_dep)
        total_extra = species_bytes + ps_cell_bytes

    return float(total_extra), int(cap_est)


def _compute_stages_ns_per_particle(
    stages_s: dict[str, float],
    *,
    step_median_s: float,
    n_local_max: int,
    n_particles_actual: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute per-local-particle and per-global-particle stage costs [ns/particle]."""
    denom_local = float(max(int(n_local_max), 1))
    denom_global = float(max(int(n_particles_actual), 1))
    ns_local: dict[str, float] = {}
    ns_global: dict[str, float] = {}
    for k, val_s in stages_s.items():
        if isinstance(val_s, (int, float)):
            ns_local[k] = float(val_s) * 1.0e9 / denom_local
            ns_global[k] = float(val_s) * 1.0e9 / denom_global
    ns_local["unprofiled_step_median_s"] = float(step_median_s) * 1.0e9 / denom_local
    ns_global["unprofiled_step_median_s"] = float(step_median_s) * 1.0e9 / denom_global
    return ns_local, ns_global


def populate_global_particles_filtered_by_slab(
    sp: Species,
    ps: ParticleSolver,
    *,
    n_total: int,
    lx: float,
    ly: float,
    lz: float,
    dx: float,
    dy: float,
    dz: float,
    v_th: float,
    w_macro: float,
    seed: int = SEED,
) -> None:
    r"""Generate one global particle set from ``seed`` and filter per rank by z-slab.

    Every rank advances an identical ``np.random.default_rng(seed)`` through
    fixed global batches (size independent of ``size`` and ``rank``) and retains
    only particles whose initial :math:`z` falls in ``ps``'s owned slab
    :math:`[z_{\mathrm{own,min}}, z_{\mathrm{own,max}})`. This guarantees that
    the global initial particle state is bit-for-bit identical for ``size=1``,
    ``size=2``, and ``size=4``.

    Parameters
    ----------
    sp : Species
        Target species (already placed on GPU if ``ps.solver.use_gpu``).
    ps : ParticleSolver
        Attached particle solver defining the rank's owned :math:`z`-slab.
    n_total : int
        Total global number of macroparticles across the full domain.
    lx, ly, lz : float
        Global domain lengths in :math:`x, y, z` [m].
    dx, dy, dz : float
        Cell sizes in :math:`x, y, z` [m].
    v_th : float
        Thermal proper-velocity standard deviation [m/s].
    w_macro : float
        Macroparticle weight (real particles per macroparticle) [-].
    seed : int, optional
        Global random seed. Default is ``SEED``.
    """
    rng = np.random.default_rng(seed)
    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    chunks_z: list[np.ndarray] = []
    chunks_ux: list[np.ndarray] = []
    chunks_uy: list[np.ndarray] = []
    chunks_uz: list[np.ndarray] = []

    for start in range(0, int(n_total), GLOBAL_BATCH_SIZE):
        m = min(GLOBAL_BATCH_SIZE, int(n_total) - start)
        x_g = rng.uniform(dx, lx - dx, m)
        y_g = rng.uniform(dy, ly - dy, m)
        z_g = rng.uniform(dz, lz - dz, m)
        ux_g = rng.normal(0.0, v_th, m)
        uy_g = rng.normal(0.0, v_th, m)
        uz_g = rng.normal(0.0, v_th, m)

        if ps.use_mpi and int(getattr(ps.solver, "size", 1)) > 1:
            mask = (z_g >= ps._z_own_min) & (z_g < ps._z_own_max)
            if np.any(mask):
                chunks_x.append(np.ascontiguousarray(x_g[mask]))
                chunks_y.append(np.ascontiguousarray(y_g[mask]))
                chunks_z.append(np.ascontiguousarray(z_g[mask]))
                chunks_ux.append(np.ascontiguousarray(ux_g[mask]))
                chunks_uy.append(np.ascontiguousarray(uy_g[mask]))
                chunks_uz.append(np.ascontiguousarray(uz_g[mask]))
            del mask
        else:
            chunks_x.append(x_g)
            chunks_y.append(y_g)
            chunks_z.append(z_g)
            chunks_ux.append(ux_g)
            chunks_uy.append(uy_g)
            chunks_uz.append(uz_g)
        del x_g, y_g, z_g, ux_g, uy_g, uz_g

    n_local = sum(int(c_arr.size) for c_arr in chunks_z)
    headroom = float(getattr(ps, "capacity_headroom", 0.0))
    if n_local > 0 and headroom > 0.0:
        target_cap = max(
            int(np.ceil(n_local * (1.0 + headroom))),
            n_local + 64,
            16,
        )
    else:
        target_cap = max(n_local, 16)
    sp._ensure_capacity(target_cap, exact=True)

    offset = 0
    for x_c, y_c, z_c, ux_c, uy_c, uz_c in zip(
        chunks_x, chunks_y, chunks_z, chunks_ux, chunks_uy, chunks_uz
    ):
        k = int(z_c.size)
        end = offset + k
        if sp.on_gpu:
            sp._x[offset:end].set(x_c)
            sp._y[offset:end].set(y_c)
            sp._z[offset:end].set(z_c)
            sp._ux[offset:end].set(ux_c)
            sp._uy[offset:end].set(uy_c)
            sp._uz[offset:end].set(uz_c)
        else:
            sp._x[offset:end] = x_c
            sp._y[offset:end] = y_c
            sp._z[offset:end] = z_c
            sp._ux[offset:end] = ux_c
            sp._uy[offset:end] = uy_c
            sp._uz[offset:end] = uz_c
        offset = end

    if n_local > 0:
        sp._w[:n_local].fill(float(w_macro))
    sp.n_particles = n_local
    sp.u_is_synced = True


def _run_profiled_stage_pass(
    solver: SolverFIT3D,
    ps: ParticleSolver,
    *,
    n_steps: int,
    use_gpu: bool,
    use_mpi: bool,
    comm,
) -> dict[str, float]:
    """Run a separate profiled pass to measure per-stage times.

    Because ``ps.profile = True`` and ``solver._profile_mpi = True`` insert
    device synchronizations around individual stages, this pass is executed
    separately after the unprofiled timing repeats so it never perturbs the
    headline coupled step timings.

    When ``ps.sort_every > 0``, the profiled pass runs ``max(n_steps, sort_every)``
    steps so that it samples the complete ``sort_every``-step sort phase cycle
    (identical to the unprofiled repeats) and measures ``sort_by_cell`` after a
    full ``sort_every``-step particle evolution window rather than immediately
    after a prior sort.
    """
    from mpi4py import MPI

    cp = None
    if use_gpu:
        import cupy as cp

    orig_sort_every = int(ps.sort_every)
    n_steps_prof = (
        max(int(n_steps), orig_sort_every) if orig_sort_every > 0 else int(n_steps)
    )
    cell_sort_call_samples_s: list[float] = []
    cell_sort_call_s = 0.0
    cell_sort_call_min_s = 0.0
    cell_sort_call_max_s = 0.0
    if orig_sort_every > 0:
        cpu_presort_snapshots: list[tuple[Species, dict[str, np.ndarray], np.ndarray | None]] = []
        if not use_gpu:
            for sp_i in ps.species:
                n_act = int(sp_i.n_particles)
                if n_act > 0:
                    fl_snap = {
                        attr: getattr(sp_i, attr)[:n_act].copy()
                        for attr in sp_i._FLOAT_FIELDS
                    }
                    id_snap = (
                        sp_i._id[:n_act].copy() if sp_i.track_ids else None
                    )
                    cpu_presort_snapshots.append((sp_i, fl_snap, id_snap))

        for sample_idx in range(N_SORT_SAMPLES):
            if sample_idx > 0:
                if use_gpu:
                    # On GPU, sort_species_by_cell_gpu writes the sorted result
                    # into _*_tmp and swaps pointers, leaving the pre-sort
                    # evolved particle permutation intact in _*_tmp. Swapping
                    # back restores the exact pre-sort state in O(1) with zero
                    # additional device allocations.
                    for sp_i in ps.species:
                        if sp_i._n_launch_bound > 0:
                            sp_i._x, sp_i._x_tmp = sp_i._x_tmp, sp_i._x
                            sp_i._y, sp_i._y_tmp = sp_i._y_tmp, sp_i._y
                            sp_i._z, sp_i._z_tmp = sp_i._z_tmp, sp_i._z
                            sp_i._ux, sp_i._ux_tmp = sp_i._ux_tmp, sp_i._ux
                            sp_i._uy, sp_i._uy_tmp = sp_i._uy_tmp, sp_i._uy
                            sp_i._uz, sp_i._uz_tmp = sp_i._uz_tmp, sp_i._uz
                            sp_i._w, sp_i._w_tmp = sp_i._w_tmp, sp_i._w
                            if sp_i.track_ids:
                                sp_i._id, sp_i._id_tmp = sp_i._id_tmp, sp_i._id
                else:
                    for sp_i, fl_snap, id_snap in cpu_presort_snapshots:
                        n_act = int(sp_i.n_particles)
                        for attr, arr_snap in fl_snap.items():
                            getattr(sp_i, attr)[:n_act] = arr_snap
                        if id_snap is not None:
                            sp_i._id[:n_act] = id_snap
            if use_gpu:
                cp.cuda.Stream.null.synchronize()
            if use_mpi and comm is not None:
                comm.Barrier()
            ts0 = time.perf_counter()
            for sp_i in ps.species:
                if (sp_i._n_launch_bound if use_gpu else sp_i.n_particles) > 0:
                    sp_i.sort_by_cell(ps.grid_map)
            if use_gpu:
                cp.cuda.Stream.null.synchronize()
            sample_s = float(time.perf_counter() - ts0)
            if use_mpi and comm is not None:
                sample_s = float(comm.allreduce(sample_s, op=MPI.MAX))
            cell_sort_call_samples_s.append(sample_s)

        cell_sort_call_s = float(np.median(cell_sort_call_samples_s))
        cell_sort_call_min_s = float(np.min(cell_sort_call_samples_s))
        cell_sort_call_max_s = float(np.max(cell_sort_call_samples_s))
    cell_sort_amortised_s = (
        float(cell_sort_call_s / float(orig_sort_every))
        if orig_sort_every > 0
        else 0.0
    )

    ps.profile = True
    ps.sort_every = 0
    solver._profile_mpi = True

    stage_accum = {k: 0.0 for k in STAGE_KEYS}
    orig_update_h = solver._update_H
    orig_update_e = solver._update_E

    def _timed_update_h():
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
        th0 = time.perf_counter()
        orig_update_h()
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
        stage_accum["fit_b_advance_s"] += time.perf_counter() - th0

    def _timed_update_e():
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
        te0 = time.perf_counter()
        orig_update_e()
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
        stage_accum["fit_e_advance_s"] += time.perf_counter() - te0

    solver._update_H = _timed_update_h
    solver._update_E = _timed_update_e

    try:
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
        if use_mpi and comm is not None:
            comm.Barrier()

        t0 = time.perf_counter()
        for _ in range(n_steps_prof):
            solver._last_ghost_exchange_s = 0.0
            solver.one_step()
            stage_accum["boris_push_deposit_s"] += float(
                ps._last_push_gather_s + ps._last_deposit_s
            )
            stage_accum["particle_migration_s"] += float(ps._last_migrate_s)
            stage_accum["guard_current_sum_s"] += float(ps._last_guard_reduce_s)
            stage_accum["fit_ghost_comm_s"] += float(
                getattr(solver, "_last_ghost_exchange_s", 0.0)
            )

        if use_gpu:
            cp.cuda.Stream.null.synchronize()
        wall_profiled = time.perf_counter() - t0
        if use_mpi and comm is not None:
            wall_profiled = float(comm.allreduce(wall_profiled, op=MPI.MAX))
    finally:
        solver._update_H = orig_update_h
        solver._update_E = orig_update_e
        ps.sort_every = orig_sort_every
        ps.profile = False
        solver._profile_mpi = False

    stage_raw = {k: float(stage_accum[k] / n_steps_prof) for k in STAGE_KEYS}
    if use_mpi and comm is not None:
        for k in STAGE_KEYS:
            stage_raw[k] = float(comm.allreduce(stage_raw[k], op=MPI.MAX))

    stage_raw["cell_sort_call_s"] = float(cell_sort_call_s)
    stage_raw["cell_sort_call_median_s"] = float(cell_sort_call_s)
    stage_raw["cell_sort_call_min_s"] = float(cell_sort_call_min_s)
    stage_raw["cell_sort_call_max_s"] = float(cell_sort_call_max_s)
    stage_raw["cell_sort_call_samples_s"] = [
        float(v) for v in cell_sort_call_samples_s
    ]
    stage_raw["cell_sort_amortised_s"] = float(cell_sort_amortised_s)
    stage_raw["profiled_n_steps"] = int(n_steps_prof)
    stage_raw["profiled_step_total_s"] = float(
        wall_profiled / n_steps_prof + cell_sort_amortised_s
    )
    stage_raw["profiled_stage_sum_s"] = float(
        sum(stage_raw[k] for k in STAGE_KEYS) + cell_sort_amortised_s
    )
    return stage_raw


def run_single_benchmark(
    *,
    n_particles: int,
    use_gpu: bool,
    use_mpi: bool,
    particles_comm: str,
    nx: int = DEFAULT_NX,
    ny: int = DEFAULT_NY,
    nz: int = DEFAULT_NZ,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_repeats: int = DEFAULT_N_REPEATS,
    n_steps: int = DEFAULT_N_STEPS,
    sort_every: int = 0,
    parent_n_particles_requested: int | None = None,
) -> dict | None:
    """Execute one coupled FIT + particle benchmark configuration on the real solver path.

    Parameters
    ----------
    n_particles : int
        Total global number of macroparticles across all MPI ranks.
    use_gpu : bool
        If ``True``, run both ``SolverFIT3D`` and ``ParticleSolver`` on GPU.
    use_mpi : bool
        If ``True``, enable MPI z-slab decomposition.
    particles_comm : str
        Particle communication backend: ``"none"``, ``"auto"``, ``"nccl"``, or ``"mpi"``.
    nx, ny, nz : int, optional
        Global grid cell counts. Default is ``(64, 64, 128)``.
    n_warmup : int, optional
        Number of warm-up steps before timed repeats (must be ``>= 3``).
        Default is ``3``.
    n_repeats : int, optional
        Number of timed repeats (must be ``>= 5``). Default is ``5``.
    n_steps : int, optional
        Number of steps per timed repeat (must be ``>= 10``; automatically
        raised to ``max(n_steps, sort_every)`` when ``sort_every > 0`` so each
        repeat includes the amortised sort cost). Default is ``10``.
    sort_every : int, optional
        Cell-sorting period in steps (``0`` disables sorting). Default is ``0``.
    parent_n_particles_requested : int or None, optional
        Reference global particle count for half-domain control runs.

    Returns
    -------
    dict or None
        Benchmark record on rank 0, or ``None`` on non-root ranks.
    """
    if int(n_warmup) < 3:
        raise ValueError(f"n_warmup must be >= 3, got {n_warmup}.")
    if int(n_repeats) < 5:
        raise ValueError(f"n_repeats must be >= 5, got {n_repeats}.")
    if int(n_steps) < 10:
        raise ValueError(f"n_steps must be >= 10, got {n_steps}.")

    n_steps_eff = (
        max(int(n_steps), int(sort_every)) if int(sort_every) > 0 else int(n_steps)
    )

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank()) if use_mpi else 0
    size = int(comm.Get_size()) if use_mpi else 1

    cp = None
    if use_gpu:
        import cupy as cp

        n_gpus = int(cp.cuda.runtime.getDeviceCount())
        shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED) if use_mpi else None
        try:
            local_rank = int(shm_comm.Get_rank()) if shm_comm is not None else 0
        finally:
            if shm_comm is not None:
                shm_comm.Free()
        if n_gpus > 0:
            cp.cuda.Device(local_rank % n_gpus).use()
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    # Preflight against g2-standard-24 counting every Species and ParticleSolver buffer
    cells_global = int(nx * ny * nz)
    cells_per_rank = int(nx * ny * (nz // size))
    n_local_est = int(np.ceil(n_particles / size))
    total_steps = int(n_warmup + (n_repeats + 1) * n_steps_eff)
    probe_bpp = probe_species_bytes_per_particle(
        use_gpu=use_gpu,
        use_mpi=use_mpi,
        size=size,
        n_local_est=n_local_est,
        track_ids=False,
        has_absorbing_boundary=True,
        sort_every=int(sort_every),
    )
    est = wakis_estimate(
        cells=cells_per_rank,
        steps=total_steps,
        backend="gpu" if use_gpu else "cpu",
    )
    extra_particle_bytes, cap_est = estimate_particle_and_solver_bytes(
        use_gpu=use_gpu,
        use_mpi=use_mpi,
        size=size,
        nx=nx,
        ny=ny,
        nz=nz,
        n_particles=int(n_particles),
        probe_bpp=probe_bpp,
        sort_every=int(sort_every),
        capacity_headroom=0.05 if (use_mpi and size > 1) else 0.0,
    )
    if use_gpu:
        est.gpu_bytes = float(est.gpu_bytes + extra_particle_bytes)
    else:
        est.host_bytes = float(est.host_bytes + extra_particle_bytes)
    wakis_preflight(est, PREFLIGHT_MACHINE)

    lx = 0.001 * float(nx)
    ly = 0.001 * float(ny)
    lz = 0.001 * float(nz)
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz

    # Equal-work boundary conditions across all rank counts (PEC z everywhere)
    bc_low = ["periodic", "periodic", "pec"]
    bc_high = ["periodic", "periodic", "pec"]

    grid = GridFIT3D(
        0.0,
        lx,
        0.0,
        ly,
        0.0,
        lz,
        nx,
        ny,
        nz,
        use_mpi=use_mpi,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        cfln=0.45,
        bc_low=bc_low,
        bc_high=bc_high,
        use_mpi=use_mpi,
        use_gpu=use_gpu,
        verbose=0,
    )

    n0 = 1.0e14
    vol = lx * ly * lz
    w_macro = float(n0 * vol / max(n_particles, 1))
    v_th = 0.04 * c

    sp = Species(
        name="electron",
        charge=-q_e,
        mass=m_e,
        capacity=16,
        track_ids=False,
        use_gpu=use_gpu,
    )
    ps = ParticleSolver(
        solver,
        species=[sp],
        deposit_scheme="esirkepov",
        shape=1,
        sort_every=int(sort_every),
        profile=False,
        particles_comm=None if particles_comm == "auto" else particles_comm,
    )

    populate_global_particles_filtered_by_slab(
        sp,
        ps,
        n_total=int(n_particles),
        lx=lx,
        ly=ly,
        lz=lz,
        dx=dx,
        dy=dy,
        dz=dz,
        v_th=v_th,
        w_macro=w_macro,
        seed=SEED,
    )

    ps.attach()
    bpp_actual = species_bytes_per_particle(sp)
    ws_total_bytes, ws_hot_bytes, ws_breakdown_bytes = (
        compute_field_current_working_set_bytes(solver, ps)
    )

    if use_gpu:
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    n_local_init = int(sp._n_launch_bound if use_gpu else sp.n_particles)
    n_total_actual = (
        int(comm.allreduce(n_local_init, op=MPI.SUM)) if use_mpi else n_local_init
    )
    n_local_min = (
        int(comm.allreduce(n_local_init, op=MPI.MIN)) if use_mpi else n_local_init
    )
    n_local_max = (
        int(comm.allreduce(n_local_init, op=MPI.MAX)) if use_mpi else n_local_init
    )

    # Warm-up steps (includes the real half-step velocity back-push on step 0).
    # Pre-sort before warm-up so warm-up warms up the sort kernel and evolves
    # particles for n_warmup steps before Repeat 0's initial sort at step 0.
    if int(sort_every) > 0:
        for sp_i in ps.species:
            if (sp_i._n_launch_bound if use_gpu else sp_i.n_particles) > 0:
                sp_i.sort_by_cell(ps.grid_map)
    ps.sort_every = 0
    for _ in range(n_warmup):
        solver.one_step()
    if int(sort_every) > 0:
        # Align ps.n_particle_steps = sort_every so step 0 of every n_steps_eff
        # repeat triggers sort_by_cell inside solver.one_step(), steps 0..sort_every-1
        # sample the full post-sort cycle, and at the end of the final repeat the
        # particles have evolved a full sort_every steps before the profiled pass.
        ps.sort_every = int(sort_every)
        ps.n_particle_steps = int(sort_every)
    if use_gpu:
        cp.cuda.Stream.null.synchronize()
    if use_mpi:
        comm.Barrier()

    # Unprofiled timed repeats: n_repeats repeats of n_steps_eff coupled steps
    repeat_step_s: list[float] = []
    for _ in range(n_repeats):
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
        if use_mpi:
            comm.Barrier()
        t_rep0 = time.perf_counter()
        for _ in range(n_steps_eff):
            solver.one_step()
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
        wall_rep = time.perf_counter() - t_rep0
        if use_mpi:
            wall_rep = float(comm.allreduce(wall_rep, op=MPI.MAX))
        repeat_step_s.append(float(wall_rep / n_steps_eff))

    step_median_s = float(np.median(repeat_step_s))
    step_min_s = float(np.min(repeat_step_s))
    step_max_s = float(np.max(repeat_step_s))

    throughput_median_mp_s = float((n_total_actual * 1.0e-6) / step_median_s)
    throughput_min_mp_s = float((n_total_actual * 1.0e-6) / step_max_s)
    throughput_max_mp_s = float((n_total_actual * 1.0e-6) / step_min_s)

    # Separate profiled run for stage breakdown (device syncs inflate stage sum)
    stage_breakdown = _run_profiled_stage_pass(
        solver,
        ps,
        n_steps=n_steps_eff,
        use_gpu=use_gpu,
        use_mpi=use_mpi,
        comm=comm if use_mpi else None,
    )

    gpu_pool_used_bytes = 0
    gpu_pool_peak_bytes = 0
    gpu_driver_used_bytes = 0
    gpu_mem_total_bytes = 0
    if use_gpu:
        cp.cuda.Stream.null.synchronize()
        pool = cp.get_default_memory_pool()
        used_pool_b = int(pool.used_bytes())
        peak_pool_b = int(pool.total_bytes())
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        drv_used_b = int(total_b - free_b)
        gpu_pool_used_bytes = (
            int(comm.allreduce(used_pool_b, op=MPI.MAX)) if use_mpi else used_pool_b
        )
        gpu_pool_peak_bytes = (
            int(comm.allreduce(peak_pool_b, op=MPI.MAX)) if use_mpi else peak_pool_b
        )
        gpu_driver_used_bytes = (
            int(comm.allreduce(drv_used_b, op=MPI.MAX)) if use_mpi else drv_used_b
        )
        gpu_mem_total_bytes = int(total_b)
        assert est.gpu_bytes >= gpu_pool_peak_bytes, (
            f"Preflight GPU memory estimate ({est.gpu_bytes / 1e9:.3f} GB) is smaller "
            f"than measured CuPy pool peak ({gpu_pool_peak_bytes / 1e9:.3f} GB)."
        )

    active_comm = ps.particles_comm
    nccl_active = bool(getattr(solver, "_nccl_comm", None) is not None)
    if active_comm == "nccl" and not nccl_active:
        raise RuntimeError(
            "particles_comm_active is 'nccl' but solver._nccl_comm is None."
        )

    del ps, sp, solver, grid
    gc.collect()
    if use_gpu:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    if rank != 0:
        return None

    stages_ns_local, stages_ns_global = _compute_stages_ns_per_particle(
        stage_breakdown,
        step_median_s=step_median_s,
        n_local_max=n_local_max,
        n_particles_actual=n_total_actual,
    )
    sort_call_s = float(stage_breakdown.get("cell_sort_call_s", 0.0))
    sort_call_min_s = float(
        stage_breakdown.get("cell_sort_call_min_s", sort_call_s)
    )
    sort_call_max_s = float(
        stage_breakdown.get("cell_sort_call_max_s", sort_call_s)
    )
    sort_call_samples_s = [
        float(v)
        for v in stage_breakdown.get("cell_sort_call_samples_s", [])
    ]
    sort_amort_s = float(stage_breakdown.get("cell_sort_amortised_s", 0.0))

    rec = {
        "backend": "gpu" if use_gpu else "cpu",
        "use_gpu": bool(use_gpu),
        "use_mpi": bool(use_mpi),
        "size": int(size),
        "particles_comm_requested": str(particles_comm),
        "particles_comm_active": str(active_comm),
        "nccl_active": bool(nccl_active),
        "sort_every": int(sort_every),
        "grid_shape": [int(nx), int(ny), int(nz)],
        "bc_low": bc_low,
        "bc_high": bc_high,
        "cells_global": cells_global,
        "cells_per_rank": cells_per_rank,
        "n_particles_requested": int(n_particles),
        "n_particles_actual": int(n_total_actual),
        "n_local_min": int(n_local_min),
        "n_local_max": int(n_local_max),
        "species_capacity_est_per_rank": int(cap_est),
        "n_warmup": int(n_warmup),
        "n_repeats": int(n_repeats),
        "n_steps_per_repeat": int(n_steps_eff),
        "repeat_step_s": [float(v) for v in repeat_step_s],
        "coupled_one_step_s": step_median_s,
        "step_median_s": step_median_s,
        "step_min_s": step_min_s,
        "step_max_s": step_max_s,
        "cell_sort_call_s": sort_call_s,
        "cell_sort_call_median_s": sort_call_s,
        "cell_sort_call_min_s": sort_call_min_s,
        "cell_sort_call_max_s": sort_call_max_s,
        "cell_sort_call_samples_s": sort_call_samples_s,
        "cell_sort_amortised_s": sort_amort_s,
        "cell_sort_amortised_share_of_step_median": float(
            sort_amort_s / max(step_median_s, 1.0e-30)
        ),
        "cell_sort_amortised_share_of_profiled_stage_sum": float(
            sort_amort_s
            / max(float(stage_breakdown["profiled_stage_sum_s"]), 1.0e-30)
        ),
        "working_set_bytes": int(ws_total_bytes),
        "working_set_mb": float(ws_total_bytes / 1.0e6),
        "kernel_working_set_bytes": int(ws_hot_bytes),
        "kernel_working_set_mb": float(ws_hot_bytes / 1.0e6),
        "gather_deposit_kernel_working_set_bytes": int(ws_hot_bytes),
        "gather_deposit_kernel_working_set_mb": float(ws_hot_bytes / 1.0e6),
        "working_set_breakdown_bytes": ws_breakdown_bytes,
        "throughput_mparticles_per_s": throughput_median_mp_s,
        "throughput_median_mparticles_per_s": throughput_median_mp_s,
        "throughput_min_mparticles_per_s": throughput_min_mp_s,
        "throughput_max_mparticles_per_s": throughput_max_mp_s,
        "cell_throughput_mcells_per_s": float(
            (cells_global * 1.0e-6) / step_median_s
        ),
        "species_bytes_per_particle": int(bpp_actual),
        "gpu_pool_used_bytes_per_rank": int(gpu_pool_used_bytes),
        "gpu_pool_peak_bytes_per_rank": int(gpu_pool_peak_bytes),
        "gpu_pool_used_gb_per_rank": float(gpu_pool_used_bytes / 1.0e9),
        "gpu_pool_peak_gb_per_rank": float(gpu_pool_peak_bytes / 1.0e9),
        "gpu_mem_used_gb_per_rank": float(gpu_pool_used_bytes / 1.0e9),
        "gpu_driver_mem_used_gb_per_rank": float(gpu_driver_used_bytes / 1.0e9),
        "gpu_mem_total_gb_per_rank": float(gpu_mem_total_bytes / 1.0e9),
        "preflight_machine": PREFLIGHT_MACHINE,
        "preflight_est_ram_gb": float(est.host_bytes / 1.0e9),
        "preflight_est_gpu_gb": float(est.gpu_bytes / 1.0e9),
        "stages_s": stage_breakdown,
        "stages_ns_per_particle": stages_ns_local,
        "stages_ns_per_local_particle": stages_ns_local,
        "stages_ns_per_global_particle": stages_ns_global,
    }
    if parent_n_particles_requested is not None:
        rec["parent_n_particles_requested"] = int(parent_n_particles_requested)
    return rec


def _backfill_record_derived_fields(rec: dict) -> None:
    """Populate derived per-particle stage and working-set metrics on loaded JSON records if missing."""
    if "stages_s" in rec:
        rec["stages_s"].setdefault("cell_sort_call_s", 0.0)
        rec["stages_s"].setdefault("cell_sort_amortised_s", 0.0)
        sort_call_s = float(rec["stages_s"]["cell_sort_call_s"])
        rec["stages_s"].setdefault("cell_sort_call_median_s", sort_call_s)
        rec["stages_s"].setdefault("cell_sort_call_min_s", sort_call_s)
        rec["stages_s"].setdefault("cell_sort_call_max_s", sort_call_s)
        sort_amort_s = float(rec["stages_s"]["cell_sort_amortised_s"])
        rec.setdefault("cell_sort_call_s", sort_call_s)
        rec.setdefault(
            "cell_sort_call_median_s",
            float(rec["stages_s"]["cell_sort_call_median_s"]),
        )
        rec.setdefault(
            "cell_sort_call_min_s",
            float(rec["stages_s"]["cell_sort_call_min_s"]),
        )
        rec.setdefault(
            "cell_sort_call_max_s",
            float(rec["stages_s"]["cell_sort_call_max_s"]),
        )
        rec.setdefault("cell_sort_amortised_s", sort_amort_s)
        rec.setdefault(
            "cell_sort_amortised_share_of_step_median",
            float(sort_amort_s / max(float(rec["step_median_s"]), 1.0e-30)),
        )
        rec.setdefault(
            "cell_sort_amortised_share_of_profiled_stage_sum",
            float(
                sort_amort_s
                / max(float(rec["stages_s"]["profiled_stage_sum_s"]), 1.0e-30)
            ),
        )
        ns_local, ns_global = _compute_stages_ns_per_particle(
            rec["stages_s"],
            step_median_s=float(rec["step_median_s"]),
            n_local_max=int(rec["n_local_max"]),
            n_particles_actual=int(rec["n_particles_actual"]),
        )
        rec["stages_ns_per_particle"] = ns_local
        rec["stages_ns_per_local_particle"] = ns_local
        rec["stages_ns_per_global_particle"] = ns_global

    if "working_set_bytes" not in rec and "grid_shape" in rec:
        nx_r, ny_r, nz_r = [int(v) for v in rec["grid_shape"]]
        size_r = max(int(rec.get("size", 1)), 1)
        use_mpi_r = bool(rec.get("use_mpi", False))
        use_gpu_r = bool(rec.get("use_gpu", True))
        nz_loc = (nz_r // size_r) + (2 if (use_mpi_r and size_r > 1) else 0)
        nz_dep = nz_loc + (1 if (use_mpi_r and size_r > 1) else 0)
        n_loc = int(nx_r * ny_r * nz_loc)
        n_dep = int(nx_r * ny_r * nz_dep)
        if use_gpu_r:
            ws_tot = int(8 * 24 * n_loc + 24 * n_dep)
            ws_hot = int(2 * 24 * n_loc + 24 * n_dep)
        else:
            ws_tot = int(5 * 24 * n_loc + (24 * n_dep if (use_mpi_r and size_r > 1) else 0))
            ws_hot = int(2 * 24 * n_loc + 24 * n_dep)
        rec["working_set_bytes"] = ws_tot
        rec["working_set_mb"] = float(ws_tot / 1.0e6)
        rec["gather_deposit_kernel_working_set_bytes"] = ws_hot
        rec["gather_deposit_kernel_working_set_mb"] = float(ws_hot / 1.0e6)

    if "gather_deposit_kernel_working_set_bytes" in rec:
        ws_hot_val = int(rec["gather_deposit_kernel_working_set_bytes"])
        rec.setdefault("kernel_working_set_bytes", ws_hot_val)
        rec.setdefault("kernel_working_set_mb", float(ws_hot_val / 1.0e6))


def _check_sorted_control_consistency(
    unsorted_records: list[dict],
    sorted_records: list[dict],
    *,
    series_label: str,
) -> list[dict]:
    """Check that sorted controls include sort cost and compare against unsorted profiling overhead.

    The consistency tolerance is computed strictly from the paired unsorted run:
    ``tolerance_s = |t_prof_u - t_med_u| + (step_max_u - step_min_u)``.
    Nothing from the sorted run enters ``tolerance_s``. Per-control difference,
    tolerance, and ``consistency_pass`` are recorded on ``r_s`` and returned
    without raising on a tolerance failure so benchmark data are never lost.

    Parameters
    ----------
    unsorted_records : list of dict
        Unsorted benchmark records (`sort_every == 0`).
    sorted_records : list of dict
        Sorted benchmark records (`sort_every > 0`).
    series_label : str
        Human-readable series identifier.

    Returns
    -------
    list of dict
        Per-control consistency check records.

    Raises
    ------
    AssertionError
        If `n_steps_per_repeat < sort_every` or if `cell_sort_call_s <= 0`.
    """
    u_map = {
        (tuple(r["grid_shape"]), int(r["n_particles_requested"])): r
        for r in unsorted_records
    }
    check_records: list[dict] = []
    for r_s in sorted_records:
        key = (tuple(r_s["grid_shape"]), int(r_s["n_particles_requested"]))
        if key not in u_map:
            continue
        r_u = u_map[key]
        sort_ev = int(r_s.get("sort_every", 0))
        n_steps_rep = int(r_s.get("n_steps_per_repeat", 0))
        if sort_ev > 0 and n_steps_rep < sort_ev:
            raise AssertionError(
                f"{series_label} at grid={list(key[0])}, N_p={key[1]:,}: "
                f"n_steps_per_repeat ({n_steps_rep}) must be >= sort_every ({sort_ev}) "
                f"so the amortised sort cost is included in every timed repeat."
            )
        if sort_ev > 0 and float(r_s.get("cell_sort_call_s", 0.0)) <= 0.0:
            raise AssertionError(
                f"{series_label} at grid={list(key[0])}, N_p={key[1]:,}: "
                f"expected positive measured cell_sort_call_s for sort_every={sort_ev}."
            )
        t_med_u = float(r_u["step_median_s"])
        t_prof_u = float(r_u["stages_s"]["profiled_step_total_s"])
        unsorted_prof_overhead_s = abs(t_prof_u - t_med_u)
        unsorted_step_jitter_s = float(r_u["step_max_s"]) - float(r_u["step_min_s"])
        tolerance_s = float(unsorted_prof_overhead_s + unsorted_step_jitter_s)

        t_med_s = float(r_s["step_median_s"])
        t_prof_s = float(r_s["stages_s"]["profiled_step_total_s"])
        sorted_prof_diff_s = abs(t_prof_s - t_med_s)
        consistency_pass = bool(sorted_prof_diff_s <= tolerance_s)
        consistency_excess_s = float(sorted_prof_diff_s - tolerance_s)

        r_s["sorted_prof_diff_s"] = float(sorted_prof_diff_s)
        r_s["unsorted_prof_overhead_s"] = float(unsorted_prof_overhead_s)
        r_s["unsorted_step_jitter_s"] = float(unsorted_step_jitter_s)
        r_s["consistency_tolerance_s"] = float(tolerance_s)
        r_s["consistency_pass"] = bool(consistency_pass)
        r_s["consistency_excess_s"] = float(consistency_excess_s)

        check_records.append(
            {
                "series": str(series_label),
                "grid_shape": [int(v) for v in key[0]],
                "n_particles_requested": int(key[1]),
                "n_particles_actual": int(r_s["n_particles_actual"]),
                "sorted_step_median_s": float(t_med_s),
                "sorted_profiled_step_total_s": float(t_prof_s),
                "sorted_prof_diff_s": float(sorted_prof_diff_s),
                "unsorted_step_median_s": float(t_med_u),
                "unsorted_profiled_step_total_s": float(t_prof_u),
                "unsorted_prof_overhead_s": float(unsorted_prof_overhead_s),
                "unsorted_step_jitter_s": float(unsorted_step_jitter_s),
                "consistency_tolerance_s": float(tolerance_s),
                "consistency_pass": bool(consistency_pass),
                "consistency_excess_s": float(consistency_excess_s),
            }
        )
    return check_records


def _interleaved_sweep_order(nz_values: list[int]) -> list[int]:
    """Return a deterministic permutation alternating small and large ``nz`` values."""
    ordered = sorted(int(v) for v in nz_values)
    interleaved: list[int] = []
    lo, hi = 0, len(ordered) - 1
    while lo <= hi:
        interleaved.append(ordered[lo])
        lo += 1
        if lo <= hi:
            interleaved.append(ordered[hi])
            hi -= 1
    return interleaved


def _find_l2_bracketing_step(
    ws_bytes_list: list[int],
    ns_list: list[float],
    *,
    l2_bytes: int,
    threshold_jump_ns: float | None,
) -> dict[str, float | int | bool | list[float] | None]:
    """Check the interval bracketing ``l2_bytes`` against ``k * noise`` and neighbour log-slopes."""
    empty_res: dict[str, float | int | bool | list[float] | None] = {
        "has_bracket": False,
        "bracket_idx": None,
        "interval_mb": [],
        "jump_ns_per_particle": None,
        "slope_per_ln_ws": None,
        "prev_slope_per_ln_ws": None,
        "next_slope_per_ln_ws": None,
        "implied_jump_prev_ns_per_particle": None,
        "implied_jump_next_ns_per_particle": None,
        "exceeds_k_noise": None if threshold_jump_ns is None else False,
        "exceeds_neighbour_slopes": False,
        "shows_step": None if threshold_jump_ns is None else False,
    }
    if len(ws_bytes_list) < 2 or l2_bytes <= 0:
        return empty_res

    candidates: list[tuple[int, float]] = []
    for idx in range(len(ws_bytes_list) - 1):
        w_lo = int(ws_bytes_list[idx])
        w_hi = int(ws_bytes_list[idx + 1])
        if w_lo <= l2_bytes <= w_hi:
            jump = float(ns_list[idx + 1] - ns_list[idx])
            candidates.append((idx, jump))

    if not candidates:
        return empty_res

    # If l2_bytes coincides with an interior grid point, prefer the interval
    # crossing above l2_bytes (or the larger positive jump).
    best_idx, best_jump = max(candidates, key=lambda item: (item[1], item[0]))
    w_lo = float(ws_bytes_list[best_idx])
    w_hi = float(ws_bytes_list[best_idx + 1])
    interval_mb = [float(w_lo / 1.0e6), float(w_hi / 1.0e6)]
    dlog_curr = float(np.log(max(w_hi, 1.0) / max(w_lo, 1.0)))
    slope_curr = (
        float(best_jump / dlog_curr) if dlog_curr > 0.0 else None
    )

    prev_slope: float | None = None
    implied_jump_prev: float | None = None
    if best_idx >= 1:
        w_prev = float(ws_bytes_list[best_idx - 1])
        dlog_prev = float(np.log(max(w_lo, 1.0) / max(w_prev, 1.0)))
        if dlog_prev > 0.0:
            prev_slope = float(
                (ns_list[best_idx] - ns_list[best_idx - 1]) / dlog_prev
            )
            implied_jump_prev = float(prev_slope * dlog_curr)

    next_slope: float | None = None
    implied_jump_next: float | None = None
    if best_idx + 2 < len(ws_bytes_list):
        w_next = float(ws_bytes_list[best_idx + 2])
        dlog_next = float(np.log(max(w_next, 1.0) / max(w_hi, 1.0)))
        if dlog_next > 0.0:
            next_slope = float(
                (ns_list[best_idx + 2] - ns_list[best_idx + 1]) / dlog_next
            )
            implied_jump_next = float(next_slope * dlog_curr)

    exceeds_k_noise: bool | None = (
        bool(best_jump > threshold_jump_ns)
        if threshold_jump_ns is not None
        else None
    )
    exceeds_prev = (
        bool(best_jump > implied_jump_prev)
        if implied_jump_prev is not None
        else True
    )
    exceeds_next = (
        bool(best_jump > implied_jump_next)
        if implied_jump_next is not None
        else True
    )
    exceeds_neighbour_slopes = bool(exceeds_prev and exceeds_next)
    shows_step: bool | None = (
        bool(exceeds_k_noise and exceeds_neighbour_slopes)
        if exceeds_k_noise is not None
        else None
    )

    return {
        "has_bracket": True,
        "bracket_idx": int(best_idx),
        "interval_mb": interval_mb,
        "jump_ns_per_particle": float(best_jump),
        "slope_per_ln_ws": slope_curr,
        "prev_slope_per_ln_ws": prev_slope,
        "next_slope_per_ln_ws": next_slope,
        "implied_jump_prev_ns_per_particle": implied_jump_prev,
        "implied_jump_next_ns_per_particle": implied_jump_next,
        "exceeds_k_noise": exceeds_k_noise,
        "exceeds_neighbour_slopes": exceeds_neighbour_slopes,
        "shows_step": shows_step,
    }


def _analyze_working_set_sweep(
    ws_block: dict,
    *,
    baseline_records: list[dict] | None = None,
    half_domain_records: list[dict] | None = None,
    sorted_control_records: list[dict] | None = None,
) -> dict:
    """Analyze the 1-L4 fixed-density working-set sweep for L2-cache step behavior."""
    if not ws_block:
        return {}
    l2_bytes = int(ws_block.get("l2_cache_bytes", 0))
    unsorted_list = sorted(
        ws_block.get("unsorted", []),
        key=lambda r: int(r.get("working_set_bytes", 0)),
    )
    sorted_list = sorted(
        ws_block.get("sorted", []),
        key=lambda r: int(r.get("working_set_bytes", 0)),
    )
    dup_unsorted_list = list(ws_block.get("independent_duplicates_unsorted", []))
    dup_sorted_list = list(ws_block.get("independent_duplicates_sorted", []))
    for rec in (
        unsorted_list
        + sorted_list
        + dup_unsorted_list
        + dup_sorted_list
        + list(baseline_records or [])
        + list(half_domain_records or [])
        + list(sorted_control_records or [])
    ):
        _backfill_record_derived_fields(rec)

    error_bar_def = (
        "whole-step repeat min/max divided by local particle count N_p "
        "around profiled push+deposit time"
    )
    points_summary: list[dict] = []
    s_by_nz = {int(r["grid_shape"][2]): r for r in sorted_list}
    for r_u in unsorted_list:
        nz_v = int(r_u["grid_shape"][2])
        ws_b = int(r_u["working_set_bytes"])
        ws_hot_b = int(r_u["gather_deposit_kernel_working_set_bytes"])
        ns_u = float(r_u["stages_ns_per_particle"]["boris_push_deposit_s"])
        denom_u = float(max(int(r_u.get("n_local_max", r_u["n_particles_actual"])), 1))
        step_med_u = float(r_u["step_median_s"])
        step_min_u = float(r_u["step_min_s"])
        step_max_u = float(r_u["step_max_s"])
        err_low_u_ns = float(max(step_med_u - step_min_u, 0.0) * 1.0e9 / denom_u)
        err_high_u_ns = float(max(step_max_u - step_med_u, 0.0) * 1.0e9 / denom_u)
        span_u_ns = float(max(step_max_u - step_min_u, 0.0) * 1.0e9 / denom_u)

        pt: dict[str, float | int | str | list[int]] = {
            "grid_shape": [int(v) for v in r_u["grid_shape"]],
            "nz": nz_v,
            "cells_per_rank": int(r_u["cells_per_rank"]),
            "n_particles": int(r_u["n_particles_actual"]),
            "working_set_bytes": ws_b,
            "working_set_mb": float(ws_b / 1.0e6),
            "kernel_working_set_bytes": ws_hot_b,
            "kernel_working_set_mb": float(ws_hot_b / 1.0e6),
            "gather_deposit_kernel_working_set_bytes": ws_hot_b,
            "gather_deposit_kernel_working_set_mb": float(ws_hot_b / 1.0e6),
            "error_bar_definition": error_bar_def,
            "unsorted_push_deposit_ns_per_particle": ns_u,
            "unsorted_whole_step_err_low_ns_per_particle": err_low_u_ns,
            "unsorted_whole_step_err_high_ns_per_particle": err_high_u_ns,
            "unsorted_whole_step_repeat_span_ns_per_particle": span_u_ns,
            "unsorted_step_median_s": step_med_u,
            "unsorted_step_min_s": step_min_u,
            "unsorted_step_max_s": step_max_u,
        }
        if nz_v in s_by_nz:
            r_s = s_by_nz[nz_v]
            denom_s = float(
                max(int(r_s.get("n_local_max", r_s["n_particles_actual"])), 1)
            )
            step_med_s = float(r_s["step_median_s"])
            step_min_s = float(r_s["step_min_s"])
            step_max_s = float(r_s["step_max_s"])
            pt["sorted_push_deposit_ns_per_particle"] = float(
                r_s["stages_ns_per_particle"]["boris_push_deposit_s"]
            )
            pt["sorted_whole_step_err_low_ns_per_particle"] = float(
                max(step_med_s - step_min_s, 0.0) * 1.0e9 / denom_s
            )
            pt["sorted_whole_step_err_high_ns_per_particle"] = float(
                max(step_max_s - step_med_s, 0.0) * 1.0e9 / denom_s
            )
            pt["sorted_whole_step_repeat_span_ns_per_particle"] = float(
                max(step_max_s - step_min_s, 0.0) * 1.0e9 / denom_s
            )
            pt["sorted_step_median_s"] = step_med_s
            pt["sorted_step_min_s"] = step_min_s
            pt["sorted_step_max_s"] = step_max_s
            pt["sorted_cell_sort_call_s"] = float(r_s.get("cell_sort_call_s", 0.0))
            pt["sorted_cell_sort_call_median_s"] = float(
                r_s.get("cell_sort_call_median_s", r_s.get("cell_sort_call_s", 0.0))
            )
            pt["sorted_cell_sort_call_min_s"] = float(
                r_s.get("cell_sort_call_min_s", r_s.get("cell_sort_call_s", 0.0))
            )
            pt["sorted_cell_sort_call_max_s"] = float(
                r_s.get("cell_sort_call_max_s", r_s.get("cell_sort_call_s", 0.0))
            )
            pt["sorted_cell_sort_amortised_s"] = float(
                r_s.get("cell_sort_amortised_s", 0.0)
            )
        points_summary.append(pt)

    # Estimate single-run noise from duplicate points, classifying each pair into:
    # - unsorted back_to_back (noise_within_sweep_ns_per_particle): independent
    #   unsorted duplicates run at the end of the sweep against the matching
    #   unsorted sweep point;
    # - sorted back_to_back (noise_within_sweep_sorted_ns_per_particle):
    #   independent sorted duplicates run at the end of the sweep against the
    #   matching sorted sweep point;
    # - cross_phase (noise_cross_phase_ns_per_particle): 1-L4 baseline and 1-L4
    #   half-domain control against the matching unsorted sweep point.
    # Unsorted within-sweep criteria (L2-step k*noise test and rise-onset
    # detection) use noise_within_sweep_ns_per_particle; the sorted-flatness
    # criterion uses noise_within_sweep_sorted_ns_per_particle.
    # A second within-sweep noise diagnostic — the median whole-step repeat span
    # across sweep points — is recorded and quoted alongside the back-to-back
    # pair count so the reader can see how optimistic a small-n back-to-back
    # estimate is. If no back-to-back pairs exist, a warning is emitted, the
    # method is recorded as "unavailable", and noise-based tests are skipped.
    duplicate_comparisons: list[dict] = []
    duplicate_spreads_ns: list[float] = []
    back_to_back_comparisons: list[dict] = []
    back_to_back_spreads_ns: list[float] = []
    back_to_back_sorted_comparisons: list[dict] = []
    back_to_back_sorted_spreads_ns: list[float] = []
    cross_phase_comparisons: list[dict] = []
    cross_phase_spreads_ns: list[float] = []
    external_unsorted: list[tuple[str, str, dict]] = []
    for r_d in dup_unsorted_list:
        external_unsorted.append(
            ("sweep_independent_duplicate", "back_to_back", r_d)
        )
    for r_b in baseline_records or []:
        external_unsorted.append(("gpu_1l4_baseline", "cross_phase", r_b))
    for r_h in half_domain_records or []:
        external_unsorted.append(
            ("gpu_1l4_half_domain_control", "cross_phase", r_h)
        )

    for r_u in unsorted_list:
        key_u = (tuple(int(v) for v in r_u["grid_shape"]), int(r_u["n_particles_requested"]))
        ns_u = float(r_u["stages_ns_per_particle"]["boris_push_deposit_s"])
        for src_name, pair_class, r_ext in external_unsorted:
            key_ext = (
                tuple(int(v) for v in r_ext["grid_shape"]),
                int(r_ext["n_particles_requested"]),
            )
            if key_ext == key_u:
                ns_ext = float(
                    r_ext["stages_ns_per_particle"]["boris_push_deposit_s"]
                )
                diff_ns = float(abs(ns_u - ns_ext))
                comp_rec = {
                    "source": src_name,
                    "pair_class": pair_class,
                    "grid_shape": list(key_u[0]),
                    "nz": int(key_u[0][2]),
                    "n_particles_requested": int(key_u[1]),
                    "sweep_unsorted_push_deposit_ns_per_particle": ns_u,
                    "duplicate_unsorted_push_deposit_ns_per_particle": ns_ext,
                    "abs_diff_ns_per_particle": diff_ns,
                }
                duplicate_spreads_ns.append(diff_ns)
                duplicate_comparisons.append(comp_rec)
                if pair_class == "cross_phase":
                    cross_phase_spreads_ns.append(diff_ns)
                    cross_phase_comparisons.append(comp_rec)
                else:
                    back_to_back_spreads_ns.append(diff_ns)
                    back_to_back_comparisons.append(comp_rec)

    for r_s in sorted_list:
        key_s = (
            tuple(int(v) for v in r_s["grid_shape"]),
            int(r_s["n_particles_requested"]),
        )
        ns_s = float(r_s["stages_ns_per_particle"]["boris_push_deposit_s"])
        for r_ds in dup_sorted_list:
            key_ds = (
                tuple(int(v) for v in r_ds["grid_shape"]),
                int(r_ds["n_particles_requested"]),
            )
            if key_ds == key_s:
                ns_ds = float(
                    r_ds["stages_ns_per_particle"]["boris_push_deposit_s"]
                )
                diff_s_ns = float(abs(ns_s - ns_ds))
                comp_s_rec = {
                    "source": "sweep_independent_duplicate_sorted",
                    "pair_class": "back_to_back_sorted",
                    "grid_shape": list(key_s[0]),
                    "nz": int(key_s[0][2]),
                    "n_particles_requested": int(key_s[1]),
                    "sweep_sorted_push_deposit_ns_per_particle": ns_s,
                    "duplicate_sorted_push_deposit_ns_per_particle": ns_ds,
                    "abs_diff_ns_per_particle": diff_s_ns,
                }
                back_to_back_sorted_spreads_ns.append(diff_s_ns)
                back_to_back_sorted_comparisons.append(comp_s_rec)

    n_back_to_back_pairs_unsorted = int(len(back_to_back_spreads_ns))
    n_back_to_back_pairs_sorted = int(len(back_to_back_sorted_spreads_ns))
    n_cross_phase_pairs_unsorted = int(len(cross_phase_spreads_ns))

    within_record_median_span_ns = (
        float(
            np.median(
                [
                    float(p["unsorted_whole_step_repeat_span_ns_per_particle"])
                    for p in points_summary
                ]
            )
        )
        if points_summary
        else 0.0
    )
    sorted_spans_ns = [
        float(p["sorted_whole_step_repeat_span_ns_per_particle"])
        for p in points_summary
        if "sorted_whole_step_repeat_span_ns_per_particle" in p
    ]
    within_record_sorted_median_span_ns: float | None = (
        float(np.median(sorted_spans_ns)) if sorted_spans_ns else None
    )

    if back_to_back_spreads_ns:
        noise_within_sweep_ns: float | None = float(max(back_to_back_spreads_ns))
        noise_within_sweep_method = "max_back_to_back_duplicate_spread"
    else:
        warnings.warn(
            "No unsorted back-to-back duplicate pairs in working_set_sweep; "
            "noise_within_sweep_method is 'unavailable' and noise-based L2-step, "
            "rise-onset, and flatness tests are skipped.",
            RuntimeWarning,
            stacklevel=2,
        )
        noise_within_sweep_ns = None
        noise_within_sweep_method = "unavailable"

    if back_to_back_sorted_spreads_ns:
        noise_within_sweep_sorted_ns: float | None = float(
            max(back_to_back_sorted_spreads_ns)
        )
        noise_within_sweep_sorted_method = (
            "max_back_to_back_sorted_duplicate_spread"
        )
    else:
        if sorted_list:
            warnings.warn(
                "No sorted back-to-back duplicate pairs in working_set_sweep; "
                "noise_within_sweep_sorted_method is 'unavailable' and the "
                "sorted-flatness test is skipped.",
                RuntimeWarning,
                stacklevel=2,
            )
        noise_within_sweep_sorted_ns = None
        noise_within_sweep_sorted_method = "unavailable"

    if cross_phase_spreads_ns:
        noise_cross_phase_ns: float | None = float(max(cross_phase_spreads_ns))
        noise_cross_phase_method = "max_cross_phase_duplicate_spread"
    else:
        noise_cross_phase_ns = None
        noise_cross_phase_method = "unavailable"

    cross_phase_to_within_sweep_noise_ratio = (
        float(noise_cross_phase_ns / noise_within_sweep_ns)
        if (
            noise_cross_phase_ns is not None
            and noise_within_sweep_ns is not None
            and noise_within_sweep_ns > 0.0
        )
        else None
    )

    k_factor = float(L2_STEP_NOISE_FACTOR_K)
    threshold_jump_ns: float | None = (
        float(k_factor * noise_within_sweep_ns)
        if noise_within_sweep_ns is not None
        else None
    )
    sorted_threshold_jump_ns: float | None = (
        float(k_factor * noise_within_sweep_sorted_ns)
        if noise_within_sweep_sorted_ns is not None
        else None
    )
    criterion_noise_class = {
        "l2_step_k_noise": "noise_within_sweep_ns_per_particle",
        "sorted_flatness": "noise_within_sweep_sorted_ns_per_particle",
        "rise_onset": "noise_within_sweep_ns_per_particle",
        "sweep_vs_baseline_or_half_domain_control": (
            "noise_cross_phase_ns_per_particle"
        ),
    }
    rise_onset_criterion = (
        "first interval i such that every later jump from i onward is > "
        "-l2_step_noise_factor_k * noise_within_sweep_ns_per_particle "
        "and the jump at i is > +l2_step_noise_factor_k * noise_within_sweep_ns_per_particle"
    )

    total_step_info = _find_l2_bracketing_step(
        [], [], l2_bytes=l2_bytes, threshold_jump_ns=threshold_jump_ns
    )
    kernel_step_info = _find_l2_bracketing_step(
        [], [], l2_bytes=l2_bytes, threshold_jump_ns=threshold_jump_ns
    )
    max_adj_jump_ns = 0.0
    max_adj_jump_interval_mb: list[float] = []
    max_adj_jump_kernel_interval_mb: list[float] = []
    unsorted_min_ns: float | None = None
    unsorted_max_ns: float | None = None
    total_span_ns = 0.0
    unsorted_full_sweep_rel_rise: float | None = None
    unsorted_rise_min_ns: float | None = None
    unsorted_rise_max_ns: float | None = None
    unsorted_rise_span_ns: float | None = None
    unsorted_rise_rel_rise: float | None = None
    is_monotonic = False
    rise_start_idx: int | None = None
    rise_start_nz_interval: list[int] = []
    rise_start_kernel_ws_interval_mb: list[float] = []
    rise_start_total_ws_interval_mb: list[float] = []
    sorted_min_ns: float | None = None
    sorted_max_ns: float | None = None
    sorted_span_ns: float | None = None
    sorted_full_sweep_rel_rise: float | None = None
    sorted_rise_min_ns: float | None = None
    sorted_rise_max_ns: float | None = None
    sorted_rise_span_ns: float | None = None
    sorted_rise_rel_rise: float | None = None
    unsorted_to_sorted_rise_span_ratio: float | None = None
    unsorted_to_sorted_rise_rel_ratio: float | None = None
    sorted_flatness_criterion = (
        "sorted_rise_regime_span_ns_per_particle <= "
        "l2_step_noise_factor_k * noise_within_sweep_sorted_ns_per_particle"
    )
    sorted_is_flat: bool | None = None
    sorted_flatness_branch = "no_sorted_sweep_data"

    if len(points_summary) >= 2:
        ns_vals = [
            float(p["unsorted_push_deposit_ns_per_particle"]) for p in points_summary
        ]
        ws_vals = [int(p["working_set_bytes"]) for p in points_summary]
        kws_vals = [int(p["kernel_working_set_bytes"]) for p in points_summary]
        unsorted_min_ns = float(min(ns_vals))
        unsorted_max_ns = float(max(ns_vals))
        total_span_ns = float(unsorted_max_ns - unsorted_min_ns)
        unsorted_full_sweep_rel_rise = float(
            total_span_ns / max(unsorted_min_ns, 1.0e-30)
        )
        is_monotonic = bool(
            all(ns_vals[idx] <= ns_vals[idx + 1] for idx in range(len(ns_vals) - 1))
        )

        # Noise-aware rise-onset rule: first interval i such that every later jump
        # from i onward is > -k * noise_within_sweep (a monotone tail up to noise)
        # and the jump at i is > +k * noise_within_sweep.
        if threshold_jump_ns is not None:
            for idx in range(len(ns_vals) - 1):
                jump_i = ns_vals[idx + 1] - ns_vals[idx]
                if jump_i > threshold_jump_ns and all(
                    (ns_vals[j + 1] - ns_vals[j]) > -threshold_jump_ns
                    for j in range(idx, len(ns_vals) - 1)
                ):
                    rise_start_idx = idx
                    break
            if rise_start_idx is None:
                # No interval meets the noise-aware onset rule: record that and
                # skip the rise-regime and flatness numbers that depend on it.
                warnings.warn(
                    "working-set sweep: no rise onset meets the noise-aware rule; "
                    "rise-regime and sorted-flatness numbers are not evaluated"
                )
            else:
                u_rise_vals = ns_vals[rise_start_idx:]
                unsorted_rise_min_ns = float(min(u_rise_vals))
                unsorted_rise_max_ns = float(max(u_rise_vals))
                unsorted_rise_span_ns = float(
                    unsorted_rise_max_ns - unsorted_rise_min_ns
                )
                unsorted_rise_rel_rise = float(
                    unsorted_rise_span_ns / max(unsorted_rise_min_ns, 1.0e-30)
                )

                rise_start_nz_interval = [
                    int(points_summary[rise_start_idx]["nz"]),
                    int(points_summary[rise_start_idx + 1]["nz"]),
                ]
                rise_start_kernel_ws_interval_mb = [
                    float(points_summary[rise_start_idx]["kernel_working_set_mb"]),
                    float(
                        points_summary[rise_start_idx + 1]["kernel_working_set_mb"]
                    ),
                ]
                rise_start_total_ws_interval_mb = [
                    float(points_summary[rise_start_idx]["working_set_mb"]),
                    float(points_summary[rise_start_idx + 1]["working_set_mb"]),
                ]

        best_idx = 0
        max_adj_jump_ns = float(ns_vals[1] - ns_vals[0])
        for idx in range(len(points_summary) - 1):
            jump = float(ns_vals[idx + 1] - ns_vals[idx])
            if jump > max_adj_jump_ns:
                max_adj_jump_ns = jump
                best_idx = idx
        max_adj_jump_interval_mb = [
            float(ws_vals[best_idx] / 1.0e6),
            float(ws_vals[best_idx + 1] / 1.0e6),
        ]
        max_adj_jump_kernel_interval_mb = [
            float(kws_vals[best_idx] / 1.0e6),
            float(kws_vals[best_idx + 1] / 1.0e6),
        ]
        if l2_bytes > 0:
            total_step_info = _find_l2_bracketing_step(
                ws_vals,
                ns_vals,
                l2_bytes=l2_bytes,
                threshold_jump_ns=threshold_jump_ns,
            )
            kernel_step_info = _find_l2_bracketing_step(
                kws_vals,
                ns_vals,
                l2_bytes=l2_bytes,
                threshold_jump_ns=threshold_jump_ns,
            )

        if all("sorted_push_deposit_ns_per_particle" in p for p in points_summary):
            s_vals = [
                float(p["sorted_push_deposit_ns_per_particle"])
                for p in points_summary
            ]
            sorted_min_ns = float(min(s_vals))
            sorted_max_ns = float(max(s_vals))
            sorted_span_ns = float(sorted_max_ns - sorted_min_ns)
            sorted_full_sweep_rel_rise = float(
                sorted_span_ns / max(sorted_min_ns, 1.0e-30)
            )
            if rise_start_idx is not None and unsorted_rise_span_ns is not None and unsorted_rise_rel_rise is not None:
                s_rise_vals = s_vals[rise_start_idx:]
                sorted_rise_min_ns = float(min(s_rise_vals))
                sorted_rise_max_ns = float(max(s_rise_vals))
                sorted_rise_span_ns = float(
                    sorted_rise_max_ns - sorted_rise_min_ns
                )
                sorted_rise_rel_rise = float(
                    sorted_rise_span_ns / max(sorted_rise_min_ns, 1.0e-30)
                )
                unsorted_to_sorted_rise_span_ratio = float(
                    unsorted_rise_span_ns / max(sorted_rise_span_ns, 1.0e-30)
                )
                unsorted_to_sorted_rise_rel_ratio = float(
                    unsorted_rise_rel_rise / max(sorted_rise_rel_rise, 1.0e-30)
                )
                # Criterion: sorted counts as flat over the rising regime iff
                # sorted_rise_span <= k * noise_within_sweep_sorted, with k = 3 and
                # the sorted within-sweep back-to-back duplicate noise estimate.
                if sorted_threshold_jump_ns is not None:
                    sorted_is_flat = bool(
                        unsorted_rise_span_ns > 0.0
                        and sorted_rise_span_ns <= sorted_threshold_jump_ns
                    )
                    sorted_flatness_branch = (
                        "flat_within_k_noise"
                        if sorted_is_flat
                        else "smaller_rise_not_flat"
                    )
                else:
                    sorted_is_flat = None
                    sorted_flatness_branch = "not_evaluated_noise_unavailable"
            else:
                sorted_is_flat = None
                sorted_flatness_branch = "not_evaluated_noise_unavailable"

    shows_total_step = total_step_info["shows_step"]
    shows_kernel_step = kernel_step_info["shows_step"]
    if shows_total_step is None or shows_kernel_step is None:
        shows_any_step: bool | None = None
    else:
        shows_any_step = bool(shows_total_step or shows_kernel_step)

    ws_block.pop("rise_onset_noise_fraction", None)
    ws_block.setdefault(
        "execution_order_nz",
        [int(r["grid_shape"][2]) for r in ws_block.get("unsorted", [])],
    )
    ws_block["error_bar_definition"] = error_bar_def
    ws_block["points_summary"] = points_summary
    ws_block["duplicate_comparisons"] = duplicate_comparisons
    # Spreads are kept per noise class only (back-to-back vs cross-phase); the
    # legacy pooled list is removed so the two classes are never mixed.
    ws_block.pop("duplicate_spreads_ns_per_particle", None)
    ws_block["n_back_to_back_pairs_unsorted"] = n_back_to_back_pairs_unsorted
    ws_block["n_back_to_back_pairs_sorted"] = n_back_to_back_pairs_sorted
    ws_block["n_cross_phase_pairs_unsorted"] = n_cross_phase_pairs_unsorted
    ws_block["back_to_back_duplicate_comparisons"] = back_to_back_comparisons
    ws_block["back_to_back_duplicate_spreads_ns_per_particle"] = (
        back_to_back_spreads_ns
    )
    ws_block["back_to_back_sorted_duplicate_comparisons"] = (
        back_to_back_sorted_comparisons
    )
    ws_block["back_to_back_sorted_duplicate_spreads_ns_per_particle"] = (
        back_to_back_sorted_spreads_ns
    )
    ws_block["cross_phase_duplicate_comparisons"] = cross_phase_comparisons
    ws_block["cross_phase_duplicate_spreads_ns_per_particle"] = (
        cross_phase_spreads_ns
    )
    ws_block["median_whole_step_repeat_span_ns_per_particle"] = float(
        within_record_median_span_ns
    )
    ws_block["within_record_median_repeat_span_ns_per_particle"] = float(
        within_record_median_span_ns
    )
    ws_block["median_whole_step_repeat_span_sorted_ns_per_particle"] = (
        within_record_sorted_median_span_ns
    )
    ws_block["within_record_sorted_median_repeat_span_ns_per_particle"] = (
        within_record_sorted_median_span_ns
    )
    ws_block["noise_within_sweep_method"] = noise_within_sweep_method
    ws_block["noise_within_sweep_ns_per_particle"] = noise_within_sweep_ns
    ws_block["noise_within_sweep_sorted_method"] = (
        noise_within_sweep_sorted_method
    )
    ws_block["noise_within_sweep_sorted_ns_per_particle"] = (
        noise_within_sweep_sorted_ns
    )
    ws_block["noise_cross_phase_method"] = noise_cross_phase_method
    ws_block["noise_cross_phase_ns_per_particle"] = noise_cross_phase_ns
    ws_block["cross_phase_to_within_sweep_noise_ratio"] = (
        cross_phase_to_within_sweep_noise_ratio
    )
    ws_block["criterion_noise_class"] = criterion_noise_class
    ws_block["l2_step_noise_class"] = "noise_within_sweep_ns_per_particle"
    ws_block["sorted_flatness_noise_class"] = (
        "noise_within_sweep_sorted_ns_per_particle"
    )
    # Legacy single-noise keys removed: use noise_within_sweep_* and
    # noise_cross_phase_* (see criterion_noise_class).
    ws_block.pop("noise_estimation_method", None)
    ws_block.pop("noise_estimate_ns_per_particle", None)
    ws_block["rise_onset_found"] = rise_start_idx is not None
    ws_block["l2_step_noise_factor_k"] = float(k_factor)
    ws_block["l2_step_threshold_ns_per_particle"] = threshold_jump_ns
    ws_block["rise_onset_criterion"] = rise_onset_criterion
    ws_block["rise_onset_threshold_ns_per_particle"] = threshold_jump_ns
    ws_block["total_ws_bracketing_interval_mb"] = total_step_info["interval_mb"]
    ws_block["total_ws_bracketing_jump_ns_per_particle"] = total_step_info[
        "jump_ns_per_particle"
    ]
    ws_block["total_ws_bracketing_implied_jump_prev_ns_per_particle"] = (
        total_step_info["implied_jump_prev_ns_per_particle"]
    )
    ws_block["total_ws_bracketing_implied_jump_next_ns_per_particle"] = (
        total_step_info["implied_jump_next_ns_per_particle"]
    )
    ws_block["total_ws_exceeds_k_noise"] = total_step_info["exceeds_k_noise"]
    ws_block["total_ws_exceeds_neighbour_slopes"] = bool(
        total_step_info["exceeds_neighbour_slopes"]
    )
    ws_block["shows_step_near_l2_cache_total_ws"] = shows_total_step
    ws_block["kernel_ws_bracketing_interval_mb"] = kernel_step_info[
        "interval_mb"
    ]
    ws_block["kernel_ws_bracketing_jump_ns_per_particle"] = kernel_step_info[
        "jump_ns_per_particle"
    ]
    ws_block["kernel_ws_bracketing_implied_jump_prev_ns_per_particle"] = (
        kernel_step_info["implied_jump_prev_ns_per_particle"]
    )
    ws_block["kernel_ws_bracketing_implied_jump_next_ns_per_particle"] = (
        kernel_step_info["implied_jump_next_ns_per_particle"]
    )
    ws_block["kernel_ws_exceeds_k_noise"] = kernel_step_info["exceeds_k_noise"]
    ws_block["kernel_ws_exceeds_neighbour_slopes"] = bool(
        kernel_step_info["exceeds_neighbour_slopes"]
    )
    ws_block["shows_step_near_l2_cache_kernel_ws"] = shows_kernel_step
    ws_block["shows_step_near_l2_cache"] = shows_any_step
    ws_block["is_monotonically_increasing_with_nz"] = bool(is_monotonic)
    ws_block["rise_start_nz_interval"] = rise_start_nz_interval
    ws_block["rise_start_kernel_ws_interval_mb"] = (
        rise_start_kernel_ws_interval_mb
    )
    ws_block["rise_start_total_ws_interval_mb"] = (
        rise_start_total_ws_interval_mb
    )
    ws_block["max_adjacent_unsorted_jump_ns_per_particle"] = float(max_adj_jump_ns)
    ws_block["max_adjacent_unsorted_jump_working_set_mb"] = max_adj_jump_interval_mb
    ws_block["max_adjacent_unsorted_jump_kernel_working_set_mb"] = (
        max_adj_jump_kernel_interval_mb
    )
    ws_block["unsorted_push_deposit_min_ns_per_particle"] = unsorted_min_ns
    ws_block["unsorted_push_deposit_max_ns_per_particle"] = unsorted_max_ns
    ws_block["unsorted_push_deposit_span_ns_per_particle"] = float(total_span_ns)
    ws_block["unsorted_full_sweep_rel_rise"] = unsorted_full_sweep_rel_rise
    ws_block["unsorted_rise_regime_min_ns_per_particle"] = unsorted_rise_min_ns
    ws_block["unsorted_rise_regime_max_ns_per_particle"] = unsorted_rise_max_ns
    ws_block["unsorted_rise_regime_span_ns_per_particle"] = unsorted_rise_span_ns
    ws_block["unsorted_rise_regime_rel_rise"] = unsorted_rise_rel_rise
    ws_block["sorted_push_deposit_min_ns_per_particle"] = sorted_min_ns
    ws_block["sorted_push_deposit_max_ns_per_particle"] = sorted_max_ns
    ws_block["sorted_push_deposit_span_ns_per_particle"] = sorted_span_ns
    ws_block["sorted_full_sweep_rel_rise"] = sorted_full_sweep_rel_rise
    ws_block["sorted_rise_regime_min_ns_per_particle"] = sorted_rise_min_ns
    ws_block["sorted_rise_regime_max_ns_per_particle"] = sorted_rise_max_ns
    ws_block["sorted_rise_regime_span_ns_per_particle"] = sorted_rise_span_ns
    ws_block["sorted_rise_regime_rel_rise"] = sorted_rise_rel_rise
    ws_block["unsorted_to_sorted_rise_span_ratio"] = (
        unsorted_to_sorted_rise_span_ratio
    )
    ws_block["unsorted_to_sorted_rise_rel_ratio"] = (
        unsorted_to_sorted_rise_rel_ratio
    )
    ws_block["sorted_flatness_criterion"] = sorted_flatness_criterion
    ws_block["sorted_flatness_threshold_ns_per_particle"] = (
        sorted_threshold_jump_ns
    )
    ws_block["sorted_flatness_branch"] = sorted_flatness_branch
    ws_block["sorted_is_flat_relative_to_unsorted"] = sorted_is_flat
    return ws_block


def _load_repeat_json(repeat_json_path: Path | None) -> dict | None:
    """Load a repeat benchmark JSON file, raising ``FileNotFoundError`` if missing."""
    if repeat_json_path is None:
        return None
    if not repeat_json_path.exists():
        raise FileNotFoundError(
            f"Repeat benchmark JSON file not found: {repeat_json_path}"
        )
    with repeat_json_path.open("r", encoding="utf-8") as rfh:
        return json.load(rfh)


def _validate_repeat_configuration(primary: dict, repeat: dict) -> None:
    """Assert that ``primary`` and ``repeat`` share the same benchmark configuration."""
    p_cfg = primary["configuration"]
    r_cfg = repeat["configuration"]

    checks = [
        (
            "grid_shape",
            [int(v) for v in p_cfg.get("grid_shape", [])],
            [int(v) for v in r_cfg.get("grid_shape", [])],
        ),
        (
            "gpu_n_particles_list",
            [
                int(r["n_particles_requested"])
                for r in primary.get("gpu_1l4_baseline", [])
            ],
            [
                int(r["n_particles_requested"])
                for r in repeat.get("gpu_1l4_baseline", [])
            ],
        ),
        (
            "sort_every",
            int(p_cfg.get("sort_every", 0)),
            int(r_cfg.get("sort_every", 0)),
        ),
        (
            "control_sort_every",
            int(p_cfg["control_sort_every"]),
            int(r_cfg["control_sort_every"]),
        ),
        (
            "n_steps_per_repeat",
            int(p_cfg.get("n_steps_per_repeat", 0)),
            int(r_cfg.get("n_steps_per_repeat", 0)),
        ),
        (
            "n_repeats",
            int(p_cfg.get("n_repeats", 0)),
            int(r_cfg.get("n_repeats", 0)),
        ),
        (
            "working_set_sweep.nz_values",
            [
                int(v)
                for v in primary.get("working_set_sweep", {}).get("nz_values", [])
            ],
            [
                int(v)
                for v in repeat.get("working_set_sweep", {}).get("nz_values", [])
            ],
        ),
    ]
    for field_name, p_val, r_val in checks:
        if p_val != r_val:
            raise ValueError(
                f"Repeat JSON configuration mismatch for '{field_name}': "
                f"primary={p_val!r} vs repeat={r_val!r}."
            )


def _compute_cross_vm_working_set_diagnostic(
    primary_ws: dict, repeat_ws: dict
) -> dict:
    """Compute cross-VM per-nz working-set differences as a separate VM-offset diagnostic.

    Cross-VM differences between independent VMs measure systematic VM-to-VM offset
    (mean difference and sign consistency across ``nz``) and are never pooled into
    either run's ``k * noise`` step/flatness threshold.
    """
    p_pts = {
        int(p["nz"]): p
        for p in (primary_ws.get("points_summary", []) if isinstance(primary_ws, dict) else [])
    }
    r_pts = {
        int(p["nz"]): p
        for p in (repeat_ws.get("points_summary", []) if isinstance(repeat_ws, dict) else [])
    }
    common_nz = sorted(set(p_pts) & set(r_pts))
    per_nz: list[dict] = []
    u_diffs: list[float] = []
    s_diffs: list[float] = []
    for nz_v in common_nz:
        pp = p_pts[nz_v]
        rp = r_pts[nz_v]
        u_p = float(pp["unsorted_push_deposit_ns_per_particle"])
        u_r = float(rp["unsorted_push_deposit_ns_per_particle"])
        u_diff = float(u_r - u_p)
        u_diffs.append(u_diff)
        row: dict[str, int | float] = {
            "nz": int(nz_v),
            "primary_unsorted_push_deposit_ns_per_particle": u_p,
            "repeat_unsorted_push_deposit_ns_per_particle": u_r,
            "unsorted_diff_repeat_minus_primary_ns_per_particle": u_diff,
            "unsorted_abs_diff_ns_per_particle": float(abs(u_diff)),
        }
        if (
            "sorted_push_deposit_ns_per_particle" in pp
            and "sorted_push_deposit_ns_per_particle" in rp
        ):
            s_p = float(pp["sorted_push_deposit_ns_per_particle"])
            s_r = float(rp["sorted_push_deposit_ns_per_particle"])
            s_diff = float(s_r - s_p)
            s_diffs.append(s_diff)
            row["primary_sorted_push_deposit_ns_per_particle"] = s_p
            row["repeat_sorted_push_deposit_ns_per_particle"] = s_r
            row["sorted_diff_repeat_minus_primary_ns_per_particle"] = s_diff
            row["sorted_abs_diff_ns_per_particle"] = float(abs(s_diff))
        per_nz.append(row)

    def _summarize_diffs(diffs: list[float]) -> dict[str, float | int | bool]:
        if not diffs:
            return {
                "count": 0,
                "mean_signed_diff_ns_per_particle": 0.0,
                "mean_abs_diff_ns_per_particle": 0.0,
                "max_abs_diff_ns_per_particle": 0.0,
                "min_signed_diff_ns_per_particle": 0.0,
                "max_signed_diff_ns_per_particle": 0.0,
                "n_positive": 0,
                "n_negative": 0,
                "n_zero": 0,
                "sign_consistency_fraction": 0.0,
                "all_same_sign": False,
            }
        n_pos = sum(1 for d in diffs if d > 0.0)
        n_neg = sum(1 for d in diffs if d < 0.0)
        n_zero = len(diffs) - n_pos - n_neg
        return {
            "count": int(len(diffs)),
            "mean_signed_diff_ns_per_particle": float(np.mean(diffs)),
            "mean_abs_diff_ns_per_particle": float(np.mean(np.abs(diffs))),
            "max_abs_diff_ns_per_particle": float(np.max(np.abs(diffs))),
            "min_signed_diff_ns_per_particle": float(min(diffs)),
            "max_signed_diff_ns_per_particle": float(max(diffs)),
            "n_positive": int(n_pos),
            "n_negative": int(n_neg),
            "n_zero": int(n_zero),
            "sign_consistency_fraction": float(max(n_pos, n_neg) / len(diffs)),
            "all_same_sign": bool(n_pos == len(diffs) or n_neg == len(diffs)),
        }

    u_summary = _summarize_diffs(u_diffs)
    s_summary = _summarize_diffs(s_diffs)
    p_within_ns = (
        float(primary_ws["noise_within_sweep_ns_per_particle"])
        if isinstance(primary_ws, dict)
        and primary_ws.get("noise_within_sweep_ns_per_particle") is not None
        else None
    )
    p_cross_ns = (
        float(primary_ws["noise_cross_phase_ns_per_particle"])
        if isinstance(primary_ws, dict)
        and primary_ws.get("noise_cross_phase_ns_per_particle") is not None
        else None
    )
    p_n_cross_pairs = (
        int(primary_ws.get("n_cross_phase_pairs_unsorted", 0))
        if isinstance(primary_ws, dict)
        else 0
    )
    p_noise_ratio = (
        float(primary_ws["cross_phase_to_within_sweep_noise_ratio"])
        if isinstance(primary_ws, dict)
        and primary_ws.get("cross_phase_to_within_sweep_noise_ratio") is not None
        else None
    )
    u_mean_abs = float(u_summary["mean_abs_diff_ns_per_particle"])
    u_max_abs = float(u_summary["max_abs_diff_ns_per_particle"])
    s_mean_abs = float(s_summary["mean_abs_diff_ns_per_particle"])
    s_max_abs = float(s_summary["max_abs_diff_ns_per_particle"])
    cross_vm_lt_cross_phase = (
        bool(
            u_mean_abs < p_cross_ns
            and u_max_abs < p_cross_ns
            and (
                int(s_summary["count"]) == 0
                or (s_mean_abs < p_cross_ns and s_max_abs < p_cross_ns)
            )
        )
        if p_cross_ns is not None and p_cross_ns > 0.0 and int(u_summary["count"]) > 0
        else None
    )
    cross_phase_to_cross_vm_mean_abs_ratio = (
        float(p_cross_ns / u_mean_abs)
        if p_cross_ns is not None and u_mean_abs > 0.0
        else None
    )
    cross_phase_to_cross_vm_max_abs_ratio = (
        float(p_cross_ns / u_max_abs)
        if p_cross_ns is not None and u_max_abs > 0.0
        else None
    )

    return {
        "difference_convention": "repeat_minus_primary_ns_per_particle",
        "pooled_into_k_noise": False,
        "primary_noise_within_sweep_ns_per_particle": p_within_ns,
        "primary_noise_cross_phase_ns_per_particle": p_cross_ns,
        "primary_n_cross_phase_pairs_unsorted": p_n_cross_pairs,
        "primary_cross_phase_to_within_sweep_noise_ratio": p_noise_ratio,
        "cross_vm_offset_smaller_than_cross_phase_offset": cross_vm_lt_cross_phase,
        "primary_cross_phase_to_unsorted_cross_vm_mean_abs_ratio": (
            cross_phase_to_cross_vm_mean_abs_ratio
        ),
        "primary_cross_phase_to_unsorted_cross_vm_max_abs_ratio": (
            cross_phase_to_cross_vm_max_abs_ratio
        ),
        "per_nz": per_nz,
        "unsorted": u_summary,
        "sorted": s_summary,
    }


def _compute_repeat_comparison(primary: dict, repeat: dict) -> dict:
    """Compute two-run min-max ranges, relative spreads, and cross-VM offset diagnostics."""
    _validate_repeat_configuration(primary, repeat)
    p_sp = {
        int(s["n_particles"]): s for s in primary.get("gpu_speedup_summary", [])
    }
    r_sp = {
        int(s["n_particles"]): s for s in repeat.get("gpu_speedup_summary", [])
    }
    per_np: list[dict] = []
    speedup_efficiency_keys = (
        "headline_speedup_2gpu_nccl_sorted_default",
        "speedup_2gpu_nccl_vs_1gpu_unsorted",
        "speedup_2gpu_pinned_mpi_vs_1gpu_unsorted",
        "efficiency_2gpu_nccl_vs_half_domain_control",
    )
    migration_share_keys = (
        "migration_share_of_2gpu_profiled_stage_sum",
        "migration_share_of_2gpu_step_median",
    )
    keys_to_compare = speedup_efficiency_keys + migration_share_keys
    for n_p in sorted(set(p_sp) & set(r_sp)):
        s1 = p_sp[n_p]
        s2 = r_sp[n_p]
        entry: dict[str, int | float | list[float]] = {"n_particles": int(n_p)}
        for k in keys_to_compare:
            v1 = s1.get(k)
            v2 = s2.get(k)
            if v1 is not None and v2 is not None:
                f1, f2 = float(v1), float(v2)
                lo, hi = min(f1, f2), max(f1, f2)
                mean_v = 0.5 * (abs(f1) + abs(f2))
                entry[f"{k}_min_max"] = [lo, hi]
                entry[f"{k}_rel_spread"] = float(
                    (hi - lo) / max(mean_v, 1.0e-30)
                )
                if k in migration_share_keys or k.startswith("efficiency_"):
                    entry[f"{k}_abs_pct_point_spread"] = float((hi - lo) * 100.0)
        per_np.append(entry)

    ref_entry = next(
        (e for e in per_np if int(e["n_particles"]) == BREAKDOWN_N_PARTICLES),
        per_np[-1] if per_np else {},
    )
    low_entry = per_np[0] if per_np else {}

    ref_sp_eff_spreads = [
        float(ref_entry[f"{k}_rel_spread"])
        for k in speedup_efficiency_keys
        if f"{k}_rel_spread" in ref_entry
    ]
    caption_sp_eff_spreads = list(ref_sp_eff_spreads) + [
        float(low_entry[f"{k}_rel_spread"])
        for k in (
            "speedup_2gpu_nccl_vs_1gpu_unsorted",
            "speedup_2gpu_pinned_mpi_vs_1gpu_unsorted",
        )
        if f"{k}_rel_spread" in low_entry
    ]
    all_sp_eff_spreads = [
        float(e[f"{k}_rel_spread"])
        for e in per_np
        for k in speedup_efficiency_keys
        if f"{k}_rel_spread" in e
    ]
    ref_mig_rel_spreads = [
        float(ref_entry[f"{k}_rel_spread"])
        for k in migration_share_keys
        if f"{k}_rel_spread" in ref_entry
    ]
    ref_mig_pp_spreads = [
        float(ref_entry[f"{k}_abs_pct_point_spread"])
        for k in migration_share_keys
        if f"{k}_abs_pct_point_spread" in ref_entry
    ]
    all_mig_rel_spreads = [
        float(e[f"{k}_rel_spread"])
        for e in per_np
        for k in migration_share_keys
        if f"{k}_rel_spread" in e
    ]

    cross_vm_ws_diag = _compute_cross_vm_working_set_diagnostic(
        primary.get("working_set_sweep", {}),
        repeat.get("working_set_sweep", {}),
    )
    if isinstance(primary.get("working_set_sweep"), dict) and primary["working_set_sweep"]:
        primary["working_set_sweep"]["cross_vm_offset_diagnostic"] = cross_vm_ws_diag

    return {
        "primary_environment": dict(primary.get("environment", {})),
        "repeat_environment": dict(repeat.get("environment", {})),
        "relative_spread_definition": "(max - min) / (0.5 * (|r_1| + |r_2|))",
        "quoted_speedup_efficiency_keys": list(speedup_efficiency_keys),
        "migration_share_keys": list(migration_share_keys),
        "max_rel_spread_quoted_speedup_efficiency_at_ref_np": (
            float(max(ref_sp_eff_spreads)) if ref_sp_eff_spreads else 0.0
        ),
        "max_rel_spread_quoted_speedup_efficiency": (
            float(max(caption_sp_eff_spreads)) if caption_sp_eff_spreads else 0.0
        ),
        "max_rel_spread_all_speedup_efficiency": (
            float(max(all_sp_eff_spreads)) if all_sp_eff_spreads else 0.0
        ),
        "max_rel_spread_migration_share_at_ref_np": (
            float(max(ref_mig_rel_spreads)) if ref_mig_rel_spreads else 0.0
        ),
        "max_abs_pct_point_spread_migration_share_at_ref_np": (
            float(max(ref_mig_pp_spreads)) if ref_mig_pp_spreads else 0.0
        ),
        "max_rel_spread_all_migration_share": (
            float(max(all_mig_rel_spreads)) if all_mig_rel_spreads else 0.0
        ),
        "working_set_cross_vm_offset_diagnostic": cross_vm_ws_diag,
        "per_n_particles": per_np,
    }


def _recompute_summaries(
    payload: dict, *, repeat_payload: dict | None = None
) -> None:
    """Recompute GPU, CPU, control-run summaries, and JSON-derived captions."""
    for series_key in (
        "gpu_1l4_baseline",
        "gpu_2l4_nccl",
        "gpu_2l4_pinned_mpi",
        "gpu_1l4_half_domain_control",
        "gpu_1l4_sorted_control",
        "gpu_2l4_nccl_sorted_control",
    ):
        for rec in payload.get(series_key, []):
            _backfill_record_derived_fields(rec)
    for cpu_list in payload.get("cpu_mpi_scaling", {}).values():
        if isinstance(cpu_list, list):
            for rec in cpu_list:
                _backfill_record_derived_fields(rec)

    consistency_records: list[dict] = []
    if payload.get("gpu_1l4_sorted_control"):
        consistency_records.extend(
            _check_sorted_control_consistency(
                payload.get("gpu_1l4_baseline", []),
                payload.get("gpu_1l4_sorted_control", []),
                series_label="gpu_1l4_sorted_control",
            )
        )
    if payload.get("gpu_2l4_nccl_sorted_control"):
        consistency_records.extend(
            _check_sorted_control_consistency(
                payload.get("gpu_2l4_nccl", []),
                payload.get("gpu_2l4_nccl_sorted_control", []),
                series_label="gpu_2l4_nccl_sorted_control",
            )
        )

    ws_block = payload.get("working_set_sweep", {})
    if isinstance(ws_block, dict) and ws_block:
        if ws_block.get("sorted"):
            ws_checks = _check_sorted_control_consistency(
                ws_block.get("unsorted", []),
                ws_block.get("sorted", []),
                series_label="working_set_sweep",
            )
            ws_block["sorted_control_consistency"] = ws_checks
            consistency_records.extend(ws_checks)
        _analyze_working_set_sweep(
            ws_block,
            baseline_records=payload.get("gpu_1l4_baseline", []),
            half_domain_records=payload.get("gpu_1l4_half_domain_control", []),
            sorted_control_records=payload.get("gpu_1l4_sorted_control", []),
        )

    payload["sorted_control_consistency"] = consistency_records

    r1_map = {
        int(r["n_particles_requested"]): r
        for r in payload.get("gpu_1l4_baseline", [])
    }
    r_nccl_map = {
        int(r["n_particles_requested"]): r
        for r in payload.get("gpu_2l4_nccl", [])
    }
    r_mpi_map = {
        int(r["n_particles_requested"]): r
        for r in payload.get("gpu_2l4_pinned_mpi", [])
    }
    r_half_map = {
        int(r.get("parent_n_particles_requested", int(r["n_particles_requested"]) * 2)): r
        for r in payload.get("gpu_1l4_half_domain_control", [])
    }
    r1_sort_map = {
        int(r["n_particles_requested"]): r
        for r in payload.get("gpu_1l4_sorted_control", [])
    }
    r2_sort_map = {
        int(r["n_particles_requested"]): r
        for r in payload.get("gpu_2l4_nccl_sorted_control", [])
    }

    speedup_summary: list[dict] = []
    control_summary: list[dict] = []
    for n_req in sorted(r1_map):
        if n_req in r_nccl_map and n_req in r_mpi_map:
            r1 = r1_map[n_req]
            r_nccl = r_nccl_map[n_req]
            r_mpi = r_mpi_map[n_req]
            t1 = float(r1["step_median_s"])
            t_nccl = float(r_nccl["step_median_s"])
            t_mpi = float(r_mpi["step_median_s"])

            push_1gpu_ns = float(
                r1["stages_ns_per_particle"]["boris_push_deposit_s"]
            )
            push_2gpu_ns = float(
                r_nccl["stages_ns_per_particle"]["boris_push_deposit_s"]
            )
            push_rel_change = float(
                (push_2gpu_ns - push_1gpu_ns) / max(push_1gpu_ns, 1.0e-30)
            )
            mig_s = float(r_nccl["stages_s"]["particle_migration_s"])
            stage_sum_2gpu_s = float(r_nccl["stages_s"]["profiled_stage_sum_s"])
            stage_sum_1gpu_s = float(r1["stages_s"]["profiled_stage_sum_s"])
            fields_1gpu_s = float(
                r1["stages_s"]["fit_b_advance_s"] + r1["stages_s"]["fit_e_advance_s"]
            )
            fields_2gpu_s = float(
                r_nccl["stages_s"]["fit_b_advance_s"]
                + r_nccl["stages_s"]["fit_e_advance_s"]
            )
            mig_share_unprofiled = float(mig_s / max(t_nccl, 1.0e-30))
            mig_share_profiled = float(mig_s / max(stage_sum_2gpu_s, 1.0e-30))
            field_share_1gpu = float(
                fields_1gpu_s / max(stage_sum_1gpu_s, 1.0e-30)
            )
            field_share_2gpu = float(
                fields_2gpu_s / max(stage_sum_2gpu_s, 1.0e-30)
            )

            sp_unsorted_nccl = float(t1 / t_nccl)
            sp_unsorted_mpi = float(t1 / t_mpi)
            eff_half_nccl = (
                float(float(r_half_map[n_req]["step_median_s"]) / t_nccl)
                if n_req in r_half_map
                else None
            )
            eff_half_mpi = (
                float(float(r_half_map[n_req]["step_median_s"]) / t_mpi)
                if n_req in r_half_map
                else None
            )

            sp_entry: dict[str, float | int | None] = {
                "n_particles": int(r1["n_particles_actual"]),
                "t_1gpu_median_s": t1,
                "t_1gpu_min_s": float(r1["step_min_s"]),
                "t_1gpu_max_s": float(r1["step_max_s"]),
                "t_2gpu_nccl_median_s": t_nccl,
                "t_2gpu_nccl_min_s": float(r_nccl["step_min_s"]),
                "t_2gpu_nccl_max_s": float(r_nccl["step_max_s"]),
                "t_2gpu_pinned_mpi_median_s": t_mpi,
                "t_2gpu_pinned_mpi_min_s": float(r_mpi["step_min_s"]),
                "t_2gpu_pinned_mpi_max_s": float(r_mpi["step_max_s"]),
                "speedup_2gpu_nccl_vs_1gpu": sp_unsorted_nccl,
                "speedup_2gpu_nccl_vs_1gpu_unsorted": sp_unsorted_nccl,
                "speedup_2gpu_pinned_mpi_vs_1gpu": sp_unsorted_mpi,
                "speedup_2gpu_pinned_mpi_vs_1gpu_unsorted": sp_unsorted_mpi,
                "nccl_over_pinned_mpi_speedup": float(t_mpi / t_nccl),
                "parallel_efficiency_2gpu_nccl": eff_half_nccl,
                "parallel_efficiency_2gpu_pinned_mpi": eff_half_mpi,
                "efficiency_2gpu_nccl_vs_half_domain_control": eff_half_nccl,
                "efficiency_2gpu_pinned_mpi_vs_half_domain_control": eff_half_mpi,
                "apparent_efficiency_2gpu_nccl_vs_full_1gpu_unsorted": float(
                    sp_unsorted_nccl / 2.0
                ),
                "apparent_efficiency_2gpu_pinned_mpi_vs_full_1gpu_unsorted": float(
                    sp_unsorted_mpi / 2.0
                ),
                "push_deposit_1gpu_ns_per_particle": push_1gpu_ns,
                "push_deposit_2gpu_nccl_ns_per_particle": push_2gpu_ns,
                "push_deposit_per_particle_rel_change_2gpu_vs_1gpu": push_rel_change,
                "migration_share_of_2gpu_step_median": mig_share_unprofiled,
                "migration_share_of_2gpu_profiled_stage_sum": mig_share_profiled,
                "field_advance_share_of_1gpu_profiled_stage_sum": field_share_1gpu,
                "field_advance_share_of_2gpu_profiled_stage_sum": field_share_2gpu,
            }

            ctrl_entry: dict[str, float | int | bool | None | list[int]] = {
                "n_particles": int(r1["n_particles_actual"]),
                "unsorted_1gpu_push_deposit_ns_per_particle": push_1gpu_ns,
                "unsorted_2gpu_nccl_push_deposit_ns_per_particle": push_2gpu_ns,
                "unsorted_push_deposit_rel_change_2gpu_vs_1gpu": push_rel_change,
                "unsorted_speedup_2gpu_nccl_vs_1gpu": sp_unsorted_nccl,
                "efficiency_2gpu_nccl_vs_half_domain_control": eff_half_nccl,
                "efficiency_2gpu_pinned_mpi_vs_half_domain_control": eff_half_mpi,
                "migration_share_of_2gpu_step_median": mig_share_unprofiled,
                "migration_share_of_2gpu_profiled_stage_sum": mig_share_profiled,
                "field_advance_share_of_1gpu_profiled_stage_sum": field_share_1gpu,
                "field_advance_share_of_2gpu_profiled_stage_sum": field_share_2gpu,
            }
            if n_req in r_half_map:
                r_half = r_half_map[n_req]
                t_half = float(r_half["step_median_s"])
                push_half_ns = float(
                    r_half["stages_ns_per_particle"]["boris_push_deposit_s"]
                )
                sp_entry["t_1gpu_half_domain_ctrl_median_s"] = t_half
                ctrl_entry["half_domain_1gpu_grid_shape"] = [
                    int(v) for v in r_half["grid_shape"]
                ]
                ctrl_entry["half_domain_1gpu_n_particles"] = int(
                    r_half["n_particles_actual"]
                )
                ctrl_entry["half_domain_1gpu_step_median_s"] = t_half
                ctrl_entry["half_domain_1gpu_push_deposit_ns_per_particle"] = (
                    push_half_ns
                )
                ctrl_entry[
                    "half_domain_1gpu_push_deposit_rel_change_vs_full_1gpu"
                ] = float((push_half_ns - push_1gpu_ns) / max(push_1gpu_ns, 1.0e-30))
            if n_req in r1_sort_map:
                r1_s = r1_sort_map[n_req]
                ctrl_entry["sorted_1gpu_prof_diff_s"] = float(
                    r1_s.get("sorted_prof_diff_s", 0.0)
                )
                ctrl_entry["sorted_1gpu_consistency_tolerance_s"] = float(
                    r1_s.get("consistency_tolerance_s", 0.0)
                )
                ctrl_entry["sorted_1gpu_consistency_pass"] = bool(
                    r1_s.get("consistency_pass", True)
                )
            if n_req in r2_sort_map:
                r2_s = r2_sort_map[n_req]
                ctrl_entry["sorted_2gpu_nccl_prof_diff_s"] = float(
                    r2_s.get("sorted_prof_diff_s", 0.0)
                )
                ctrl_entry["sorted_2gpu_nccl_consistency_tolerance_s"] = float(
                    r2_s.get("consistency_tolerance_s", 0.0)
                )
                ctrl_entry["sorted_2gpu_nccl_consistency_pass"] = bool(
                    r2_s.get("consistency_pass", True)
                )
            if n_req in r1_sort_map and n_req in r2_sort_map:
                r1_s = r1_sort_map[n_req]
                r2_s = r2_sort_map[n_req]
                t1_s = float(r1_s["step_median_s"])
                t2_s = float(r2_s["step_median_s"])
                sp_sorted = float(t1_s / t2_s)
                p1_s_ns = float(
                    r1_s["stages_ns_per_particle"]["boris_push_deposit_s"]
                )
                p2_s_ns = float(
                    r2_s["stages_ns_per_particle"]["boris_push_deposit_s"]
                )
                s1_call_s = float(r1_s.get("cell_sort_call_s", 0.0))
                s2_call_s = float(r2_s.get("cell_sort_call_s", 0.0))
                s1_call_min_s = float(
                    r1_s.get("cell_sort_call_min_s", s1_call_s)
                )
                s1_call_max_s = float(
                    r1_s.get("cell_sort_call_max_s", s1_call_s)
                )
                s2_call_min_s = float(
                    r2_s.get("cell_sort_call_min_s", s2_call_s)
                )
                s2_call_max_s = float(
                    r2_s.get("cell_sort_call_max_s", s2_call_s)
                )
                s1_amort_s = float(r1_s.get("cell_sort_amortised_s", 0.0))
                s2_amort_s = float(r2_s.get("cell_sort_amortised_s", 0.0))
                s1_share = float(
                    r1_s.get(
                        "cell_sort_amortised_share_of_step_median",
                        s1_amort_s / max(t1_s, 1.0e-30),
                    )
                )
                s2_share = float(
                    r2_s.get(
                        "cell_sort_amortised_share_of_step_median",
                        s2_amort_s / max(t2_s, 1.0e-30),
                    )
                )
                sp_entry["t_1gpu_sorted_median_s"] = t1_s
                sp_entry["t_2gpu_nccl_sorted_median_s"] = t2_s
                sp_entry["headline_speedup_2gpu_nccl_sorted_default"] = sp_sorted
                sp_entry["speedup_2gpu_nccl_vs_1gpu_sorted_default"] = sp_sorted
                sp_entry[
                    "apparent_efficiency_2gpu_nccl_vs_full_1gpu_sorted_default"
                ] = float(sp_sorted / 2.0)
                ctrl_entry["sorted_1gpu_step_median_s"] = t1_s
                ctrl_entry["sorted_2gpu_nccl_step_median_s"] = t2_s
                ctrl_entry["sorted_speedup_2gpu_nccl_vs_1gpu"] = sp_sorted
                ctrl_entry["headline_speedup_2gpu_nccl_sorted_default"] = sp_sorted
                ctrl_entry["sorted_1gpu_push_deposit_ns_per_particle"] = p1_s_ns
                ctrl_entry["sorted_2gpu_nccl_push_deposit_ns_per_particle"] = p2_s_ns
                ctrl_entry["sorted_push_deposit_rel_change_2gpu_vs_1gpu"] = float(
                    (p2_s_ns - p1_s_ns) / max(p1_s_ns, 1.0e-30)
                )
                ctrl_entry["sorted_1gpu_cell_sort_call_s"] = s1_call_s
                ctrl_entry["sorted_1gpu_cell_sort_call_median_s"] = s1_call_s
                ctrl_entry["sorted_1gpu_cell_sort_call_min_s"] = s1_call_min_s
                ctrl_entry["sorted_1gpu_cell_sort_call_max_s"] = s1_call_max_s
                ctrl_entry["sorted_2gpu_nccl_cell_sort_call_s"] = s2_call_s
                ctrl_entry["sorted_2gpu_nccl_cell_sort_call_median_s"] = s2_call_s
                ctrl_entry["sorted_2gpu_nccl_cell_sort_call_min_s"] = s2_call_min_s
                ctrl_entry["sorted_2gpu_nccl_cell_sort_call_max_s"] = s2_call_max_s
                ctrl_entry["sorted_1gpu_cell_sort_amortised_s"] = s1_amort_s
                ctrl_entry["sorted_2gpu_nccl_cell_sort_amortised_s"] = s2_amort_s
                ctrl_entry["sorted_1gpu_cell_sort_amortised_share"] = s1_share
                ctrl_entry["sorted_2gpu_nccl_cell_sort_amortised_share"] = s2_share
            speedup_summary.append(sp_entry)
            control_summary.append(ctrl_entry)

    payload["gpu_speedup_summary"] = speedup_summary
    payload["control_analysis"] = control_summary

    if repeat_payload is not None:
        _recompute_summaries(repeat_payload, repeat_payload=None)
        payload["repeat_comparison"] = _compute_repeat_comparison(
            payload, repeat_payload
        )
    else:
        payload.pop("repeat_comparison", None)
        if isinstance(ws_block, dict):
            ws_block.pop("cross_vm_offset_diagnostic", None)

    cpu_rec = payload.get("cpu_mpi_scaling", {})
    c1_map = {
        int(r["n_particles_requested"]): r for r in cpu_rec.get("size_1", [])
    }
    c2_map = {
        int(r["n_particles_requested"]): r for r in cpu_rec.get("size_2", [])
    }
    c4_map = {
        int(r["n_particles_requested"]): r for r in cpu_rec.get("size_4", [])
    }
    cpu_speedup_summary: list[dict] = []
    for n_req in sorted(c1_map):
        if n_req in c2_map and n_req in c4_map:
            r1 = c1_map[n_req]
            r2 = c2_map[n_req]
            r4 = c4_map[n_req]
            t1 = float(r1["step_median_s"])
            t2 = float(r2["step_median_s"])
            t4 = float(r4["step_median_s"])
            cpu_speedup_summary.append(
                {
                    "n_particles": int(r1["n_particles_actual"]),
                    "t_cpu_1rank_median_s": t1,
                    "t_cpu_2rank_median_s": t2,
                    "t_cpu_4rank_median_s": t4,
                    "speedup_2rank_vs_1rank": float(t1 / t2),
                    "speedup_4rank_vs_1rank": float(t1 / t4),
                }
            )
    payload["cpu_speedup_summary"] = cpu_speedup_summary

    # Build figure caption strings strictly from the computed JSON summary records
    if speedup_summary:
        cfg_block = payload["configuration"]
        unsorted_sort_every = int(cfg_block.get("sort_every", 0))
        ctrl_sort_every = int(cfg_block["control_sort_every"])
        s_low = speedup_summary[0]
        s_ref = next(
            (s for s in speedup_summary if int(s["n_particles"]) == BREAKDOWN_N_PARTICLES),
            speedup_summary[-1],
        )
        s_ge_1e6 = [s for s in speedup_summary if int(s["n_particles"]) >= 1_000_000]
        max_field_share_ge_1e6_pct = (
            max(
                float(s["field_advance_share_of_1gpu_profiled_stage_sum"]) * 100.0
                for s in s_ge_1e6
            )
            if s_ge_1e6
            else float(s_ref["field_advance_share_of_1gpu_profiled_stage_sum"]) * 100.0
        )
        ref_np = int(s_ref["n_particles"])
        ref_sp_nccl_unsorted = float(s_ref["speedup_2gpu_nccl_vs_1gpu_unsorted"])
        ref_sp_mpi_unsorted = float(s_ref["speedup_2gpu_pinned_mpi_vs_1gpu_unsorted"])
        ref_push_1 = float(s_ref["push_deposit_1gpu_ns_per_particle"])
        ref_push_2 = float(s_ref["push_deposit_2gpu_nccl_ns_per_particle"])
        ref_push_chg_pct = (
            float(s_ref["push_deposit_per_particle_rel_change_2gpu_vs_1gpu"]) * 100.0
        )
        ref_mig_step_pct = (
            float(s_ref["migration_share_of_2gpu_step_median"]) * 100.0
        )
        ref_mig_prof_pct = (
            float(s_ref["migration_share_of_2gpu_profiled_stage_sum"]) * 100.0
        )
        ref_field_pct = (
            float(s_ref["field_advance_share_of_1gpu_profiled_stage_sum"]) * 100.0
        )
        low_np = int(s_low["n_particles"])
        low_sp_nccl = float(s_low["speedup_2gpu_nccl_vs_1gpu_unsorted"])
        low_sp_mpi = float(s_low["speedup_2gpu_pinned_mpi_vs_1gpu_unsorted"])

        ctrl_ref = next(
            (c for c in control_summary if int(c["n_particles"]) == ref_np),
            control_summary[-1] if control_summary else {},
        )
        rep_comp = payload.get("repeat_comparison", {})
        rep_ref = next(
            (
                r
                for r in rep_comp.get("per_n_particles", [])
                if int(r["n_particles"]) == ref_np
            ),
            {},
        )

        def _rep_range_speedup(key: str) -> str:
            mm = rep_ref.get(f"{key}_min_max")
            if mm and len(mm) == 2:
                return f" (two-run min-max {_fmt_sig2(mm[0])}x-{_fmt_sig2(mm[1])}x)"
            return ""

        def _rep_range_pct(key: str) -> str:
            mm = rep_ref.get(f"{key}_min_max")
            if mm and len(mm) == 2:
                return (
                    f" (two-run min-max {_fmt_pct0(mm[0] * 100.0)}-"
                    f"{_fmt_pct0(mm[1] * 100.0)})"
                )
            return ""

        # (b) Headline speedup at the library default (sort cost included)
        if "sorted_speedup_2gpu_nccl_vs_1gpu" in ctrl_ref:
            sort_sp = float(ctrl_ref["sorted_speedup_2gpu_nccl_vs_1gpu"])
            sort_p1 = float(ctrl_ref["sorted_1gpu_push_deposit_ns_per_particle"])
            sort_p2 = float(ctrl_ref["sorted_2gpu_nccl_push_deposit_ns_per_particle"])
            sort_c1_ms = float(ctrl_ref.get("sorted_1gpu_cell_sort_call_s", 0.0)) * 1e3
            sort_c2_ms = (
                float(ctrl_ref.get("sorted_2gpu_nccl_cell_sort_call_s", 0.0)) * 1e3
            )
            sort_sh1_pct = (
                float(ctrl_ref.get("sorted_1gpu_cell_sort_amortised_share", 0.0))
                * 100.0
            )
            sort_sh2_pct = (
                float(ctrl_ref.get("sorted_2gpu_nccl_cell_sort_amortised_share", 0.0))
                * 100.0
            )
            headline_clause = (
                f"At N_p = {ref_np:,}, the headline 2-GPU NCCL speedup at the library "
                f"default (periodic cell sorting sort_every={ctrl_sort_every}, "
                f"amortised sort cost included) is (b) {_fmt_sig2(sort_sp)}x"
                f"{_rep_range_speedup('headline_speedup_2gpu_nccl_sorted_default')} "
                f"(sorted push+deposit {_fmt_sig2(sort_p2)} vs {_fmt_sig2(sort_p1)} ns/particle; "
                f"sort call {_fmt_sig2(sort_c1_ms)} ms / {_fmt_pct0(sort_sh1_pct)} amortised "
                f"share on 1 L4 and {_fmt_sig2(sort_c2_ms)} ms / {_fmt_pct0(sort_sh2_pct)} "
                f"on 2 L4), compared with (a) {_fmt_sig2(ref_sp_nccl_unsorted)}x"
                f"{_rep_range_speedup('speedup_2gpu_nccl_vs_1gpu_unsorted')} with NCCL "
                f"and {_fmt_sig2(ref_sp_mpi_unsorted)}x"
                f"{_rep_range_speedup('speedup_2gpu_pinned_mpi_vs_1gpu_unsorted')} with "
                f"pinned-host MPI against the unsorted 1-L4 baseline "
                f"(sort_every={unsorted_sort_every})"
            )
        else:
            headline_clause = (
                f"At N_p = {ref_np:,}, the 2-GPU speedup against the unsorted 1-L4 "
                f"baseline (sort_every={unsorted_sort_every}) is (a) "
                f"{_fmt_sig2(ref_sp_nccl_unsorted)}x"
                f"{_rep_range_speedup('speedup_2gpu_nccl_vs_1gpu_unsorted')} with NCCL "
                f"and {_fmt_sig2(ref_sp_mpi_unsorted)}x"
                f"{_rep_range_speedup('speedup_2gpu_pinned_mpi_vs_1gpu_unsorted')} with "
                f"pinned-host MPI"
            )

        # (c) Efficiency vs the equal-work half-domain control, plus separate sentence
        # for 2-GPU and half-domain control per-particle push+deposit numbers
        cfg_grid = cfg_block.get(
            "grid_shape", [DEFAULT_NX, DEFAULT_NY, DEFAULT_NZ]
        )
        nx_c, ny_c, nz_c = int(cfg_grid[0]), int(cfg_grid[1]), int(cfg_grid[2])
        eff_val = ctrl_ref.get("efficiency_2gpu_nccl_vs_half_domain_control")
        has_half_ctrl = (
            "half_domain_1gpu_push_deposit_ns_per_particle" in ctrl_ref
            and eff_val is not None
        )
        if has_half_ctrl:
            gh = ctrl_ref.get(
                "half_domain_1gpu_grid_shape", [nx_c, ny_c, max(nz_c // 2, 2)]
            )
            gh_str = f"{int(gh[0])}x{int(gh[1])}x{int(gh[2])}"
            half_ns = float(ctrl_ref["half_domain_1gpu_push_deposit_ns_per_particle"])
            half_chg_pct = (
                float(
                    ctrl_ref[
                        "half_domain_1gpu_push_deposit_rel_change_vs_full_1gpu"
                    ]
                )
                * 100.0
            )
            eff_half = float(eff_val)
            app_eff_unsorted = float(
                s_ref["apparent_efficiency_2gpu_nccl_vs_full_1gpu_unsorted"]
            )
            eff_clause = (
                f", and (c) parallel efficiency vs the equal-work 1-L4 {gh_str} N_p/2 "
                f"half-domain control (t_half_ctrl / t_2gpu) is "
                f"{_fmt_pct0(eff_half * 100.0)}"
                f"{_rep_range_pct('efficiency_2gpu_nccl_vs_half_domain_control')} "
                f"(whereas apparent efficiency (t_1gpu / t_2gpu) / 2 against the "
                f"full-domain unsorted 1-L4 baseline is "
                f"{_fmt_pct0(app_eff_unsorted * 100.0)})."
            )
            push_compare_sentence = (
                f" At N_p = {ref_np:,}, unsorted per-particle push+deposit changes by "
                f"{_fmt_pct0(ref_push_chg_pct, signed=True)} on 2 GPUs "
                f"({_fmt_sig2(ref_push_2)} vs {_fmt_sig2(ref_push_1)} ns/particle on "
                f"1 L4) and by {_fmt_pct0(half_chg_pct, signed=True)} "
                f"({_fmt_sig2(half_ns)} ns/particle) on the 1-L4 {gh_str} N_p/2 "
                f"half-domain control."
            )
        else:
            eff_clause = "."
            push_compare_sentence = (
                f" At N_p = {ref_np:,}, unsorted per-particle push+deposit changes by "
                f"{_fmt_pct0(ref_push_chg_pct, signed=True)} on 2 GPUs "
                f"({_fmt_sig2(ref_push_2)} vs {_fmt_sig2(ref_push_1)} ns/particle on "
                f"1 L4)."
            )

        # Working-set sweep mechanism clause strictly from data
        ws_pts = ws_block.get("points_summary", []) if isinstance(ws_block, dict) else []
        if len(ws_pts) >= 2:
            l2_mb = float(ws_block.get("l2_cache_bytes", 0)) / 1.0e6
            nz_first = int(ws_pts[0]["nz"])
            nz_last = int(ws_pts[-1]["nz"])
            ws_first_mb = float(ws_pts[0]["working_set_mb"])
            ws_last_mb = float(ws_pts[-1]["working_set_mb"])
            kws_first_mb = float(ws_pts[0]["kernel_working_set_mb"])
            kws_last_mb = float(ws_pts[-1]["kernel_working_set_mb"])
            ns_u_vals = [
                float(p["unsorted_push_deposit_ns_per_particle"]) for p in ws_pts
            ]
            ns_u_first = ns_u_vals[0]
            ns_u_last = ns_u_vals[-1]
            min_idx = int(np.argmin(ns_u_vals))
            nz_min = int(ws_pts[min_idx]["nz"])
            ns_u_min = ns_u_vals[min_idx]
            k_val = float(ws_block.get("l2_step_noise_factor_k", L2_STEP_NOISE_FACTOR_K))
            n_btb_u = int(ws_block.get("n_back_to_back_pairs_unsorted", 0))
            n_btb_s = int(ws_block.get("n_back_to_back_pairs_sorted", 0))
            n_cp_u = int(ws_block.get("n_cross_phase_pairs_unsorted", 0))
            med_span_u_ns = float(
                ws_block.get(
                    "median_whole_step_repeat_span_ns_per_particle",
                    ws_block.get("within_record_median_repeat_span_ns_per_particle", 0.0),
                )
            )
            med_span_s_val = ws_block.get(
                "median_whole_step_repeat_span_sorted_ns_per_particle",
                ws_block.get("within_record_sorted_median_repeat_span_ns_per_particle"),
            )
            noise_within_val = ws_block.get("noise_within_sweep_ns_per_particle")
            noise_within_ns = (
                float(noise_within_val) if noise_within_val is not None else None
            )
            noise_within_sorted_val = ws_block.get(
                "noise_within_sweep_sorted_ns_per_particle"
            )
            noise_within_sorted_ns = (
                float(noise_within_sorted_val)
                if noise_within_sorted_val is not None
                else None
            )
            noise_cross_val = ws_block.get("noise_cross_phase_ns_per_particle")
            noise_cross_ns = (
                float(noise_cross_val) if noise_cross_val is not None else None
            )
            noise_ratio = ws_block.get("cross_phase_to_within_sweep_noise_ratio")
            noise_within_method_str = str(
                ws_block.get(
                    "noise_within_sweep_method",
                    "max_back_to_back_duplicate_spread",
                )
            )
            noise_within_sorted_method_str = str(
                ws_block.get("noise_within_sweep_sorted_method", "unavailable")
            )
            noise_desc = (
                f"unsorted within-sweep noise (max of {n_btb_u} back-to-back pairs)"
            )
            btb_vs_med_u_str = (
                f"the back-to-back estimate rests on {n_btb_u} pairs; "
                f"the median repeat span is {_fmt_sig2(med_span_u_ns)} ns/particle"
            )
            if (
                noise_ratio is not None
                and noise_cross_ns is not None
                and noise_within_ns is not None
                and noise_cross_ns > 0.0
            ):
                noise_classes_clause = (
                    f" (unsorted within-sweep noise is "
                    f"{_fmt_sig2(noise_within_ns)} ns/particle, max of {n_btb_u} "
                    f"back-to-back pairs — {btb_vs_med_u_str} — vs unsorted "
                    f"cross-phase offset {_fmt_sig2(noise_cross_ns)} ns/particle, "
                    f"max of {n_cp_u} cross-phase pairs against the 1-L4 baseline "
                    f"and half-domain control, so the two noise classes differ by "
                    f"{_fmt_sig2(float(noise_ratio))}x)"
                )
            elif noise_within_ns is not None:
                noise_classes_clause = f" ({btb_vs_med_u_str})"
            else:
                noise_classes_clause = ""
            thresh_val = ws_block.get("l2_step_threshold_ns_per_particle")
            thresh_ns = float(thresh_val) if thresh_val is not None else None
            s_thresh_val = ws_block.get("sorted_flatness_threshold_ns_per_particle")
            s_thresh_ns = float(s_thresh_val) if s_thresh_val is not None else None
            is_mono = bool(ws_block.get("is_monotonically_increasing_with_nz", False))
            rise_onset_found = bool(ws_block.get("rise_onset_found", True))
            rise_nz_iv = ws_block.get("rise_start_nz_interval", [])
            rise_kws_iv = ws_block.get("rise_start_kernel_ws_interval_mb", [])
            rise_ws_iv = ws_block.get("rise_start_total_ws_interval_mb", [])
            kws_br_iv = ws_block.get("kernel_ws_bracketing_interval_mb", [])
            kws_br_jump = ws_block.get("kernel_ws_bracketing_jump_ns_per_particle")
            kws_imp_prev = ws_block.get(
                "kernel_ws_bracketing_implied_jump_prev_ns_per_particle"
            )
            kws_imp_next = ws_block.get(
                "kernel_ws_bracketing_implied_jump_next_ns_per_particle"
            )
            kws_exceeds_noise = bool(ws_block.get("kernel_ws_exceeds_k_noise", False))
            kws_exceeds_trend = bool(
                ws_block.get("kernel_ws_exceeds_neighbour_slopes", False)
            )
            tws_br_iv = ws_block.get("total_ws_bracketing_interval_mb", [])
            tws_br_jump = ws_block.get("total_ws_bracketing_jump_ns_per_particle")

            if 0 < min_idx < len(ws_pts) - 1:
                shape_desc = (
                    f"starts at {_fmt_sig2(ns_u_first)} ns/particle (nz={nz_first}), "
                    f"reaches an interior minimum of {_fmt_sig2(ns_u_min)} ns/particle "
                    f"(nz={nz_min}, so small nz is slower than intermediate nz), and "
                    f"rises to {_fmt_sig2(ns_u_last)} ns/particle (nz={nz_last})"
                )
            elif is_mono:
                shape_desc = (
                    f"rises monotonically from {_fmt_sig2(ns_u_first)} ns/particle "
                    f"(nz={nz_first}) to {_fmt_sig2(ns_u_last)} ns/particle (nz={nz_last})"
                )
            else:
                shape_desc = (
                    f"starts at {_fmt_sig2(ns_u_first)} ns/particle (nz={nz_first}, "
                    f"push+deposit kernel working set {kws_first_mb:.1f} MB, total "
                    f"field+current working set {ws_first_mb:.1f} MB) and rises to "
                    f"{_fmt_sig2(ns_u_last)} ns/particle (nz={nz_last}, push+deposit "
                    f"kernel working set {kws_last_mb:.1f} MB, total field+current "
                    f"working set {ws_last_mb:.1f} MB)"
                )

            sorted_clause = ""
            if ws_block.get("sorted_push_deposit_min_ns_per_particle") is not None:
                s_min_ns = float(
                    ws_block.get("sorted_push_deposit_min_ns_per_particle", 0.0)
                )
                s_max_ns = float(
                    ws_block.get("sorted_push_deposit_max_ns_per_particle", 0.0)
                )
                s_full_pct = (
                    float(ws_block.get("sorted_full_sweep_rel_rise", 0.0)) * 100.0
                )
                u_full_pct = (
                    float(ws_block.get("unsorted_full_sweep_rel_rise", 0.0)) * 100.0
                )
                med_span_s_str = (
                    f"; for sorted, the back-to-back estimate rests on {n_btb_s} pairs; "
                    f"the median repeat span is {_fmt_sig2(float(med_span_s_val))} ns/particle"
                    if med_span_s_val is not None
                    else f"; for sorted, the back-to-back estimate rests on {n_btb_s} pairs"
                )
                if (
                    noise_within_method_str == "unavailable"
                    or noise_within_sorted_method_str == "unavailable"
                    or s_thresh_ns is None
                    or noise_within_sorted_ns is None
                ):
                    sorted_clause = (
                        f" For the cell-sorted curve (sort_every={ctrl_sort_every}; "
                        f"over the full sweep sorted rises {_fmt_sig2(s_full_pct)}% from "
                        f"{_fmt_sig2(s_min_ns)} to {_fmt_sig2(s_max_ns)} ns/particle vs "
                        f"unsorted {_fmt_sig2(u_full_pct)}%), sorted within-sweep "
                        f"back-to-back duplicate noise is unavailable (n={n_btb_s} "
                        f"sorted back-to-back pairs{med_span_s_str}), so the "
                        f"noise-based sorted-flatness test was not evaluated."
                    )
                elif (
                    not rise_onset_found
                    or len(rise_kws_iv) < 2
                    or len(rise_ws_iv) < 2
                ):
                    sorted_clause = (
                        f" For the cell-sorted curve (sort_every={ctrl_sort_every}; "
                        f"over the full sweep sorted rises {_fmt_sig2(s_full_pct)}% from "
                        f"{_fmt_sig2(s_min_ns)} to {_fmt_sig2(s_max_ns)} ns/particle vs "
                        f"unsorted {_fmt_sig2(u_full_pct)}%), no rise onset met the "
                        f"noise-aware rule (k·noise_within_sweep), so the "
                        f"rise-regime and sorted-flatness numbers are not evaluated."
                    )
                else:
                    s_rmin_ns = float(
                        ws_block.get(
                            "sorted_rise_regime_min_ns_per_particle", s_min_ns
                        )
                    )
                    s_rmax_ns = float(
                        ws_block.get(
                            "sorted_rise_regime_max_ns_per_particle", s_max_ns
                        )
                    )
                    s_rspan_ns = float(
                        ws_block.get(
                            "sorted_rise_regime_span_ns_per_particle",
                            s_rmax_ns - s_rmin_ns,
                        )
                    )
                    u_rspan_ns = float(
                        ws_block.get("unsorted_rise_regime_span_ns_per_particle", 0.0)
                    )
                    s_rise_pct = (
                        float(ws_block.get("sorted_rise_regime_rel_rise", 0.0)) * 100.0
                    )
                    u_rise_pct = (
                        float(ws_block.get("unsorted_rise_regime_rel_rise", 0.0)) * 100.0
                    )
                    span_ratio = float(
                        ws_block.get("unsorted_to_sorted_rise_span_ratio", 0.0)
                    )
                    rel_ratio = float(
                        ws_block.get("unsorted_to_sorted_rise_rel_ratio", 0.0)
                    )
                    if bool(ws_block.get("sorted_is_flat_relative_to_unsorted", False)):
                        sorted_clause = (
                            f" By contrast, the cell-sorted curve "
                            f"(sort_every={ctrl_sort_every}) is flat within sorted "
                            f"within-sweep k·noise over the rising regime (span "
                            f"{_fmt_sig2(s_rspan_ns)} <= sorted within-sweep k·noise = "
                            f"{_fmt_sig2(s_thresh_ns)} ns/particle, where k={k_val:.1f} "
                            f"times sorted within-sweep noise "
                            f"{_fmt_sig2(noise_within_sorted_ns)} ns/particle, max of "
                            f"{n_btb_s} sorted back-to-back pairs{med_span_s_str}, from "
                            f"{_fmt_sig2(s_rmin_ns)} to {_fmt_sig2(s_rmax_ns)} ns/particle, "
                            f"where sorted rises {_fmt_sig2(s_rise_pct)}% vs unsorted "
                            f"{_fmt_sig2(u_rise_pct)}% across push+deposit kernel working "
                            f"set {rise_kws_iv[0]:.1f}..{kws_last_mb:.1f} MB and total "
                            f"field+current working set "
                            f"{rise_ws_iv[0]:.1f}..{ws_last_mb:.1f} MB; over the full "
                            f"sweep sorted rises {_fmt_sig2(s_full_pct)}% from "
                            f"{_fmt_sig2(s_min_ns)} to {_fmt_sig2(s_max_ns)} ns/particle "
                            f"vs unsorted {_fmt_sig2(u_full_pct)}%), demonstrating that "
                            f"the unsorted penalty is locality-dependent and consistent "
                            f"with an L2 cache-residency effect as a stated hypothesis."
                        )
                    else:
                        sorted_clause = (
                            f" By contrast, over the rising regime (push+deposit kernel "
                            f"working set {rise_kws_iv[0]:.1f}..{kws_last_mb:.1f} MB, "
                            f"total field+current working set "
                            f"{rise_ws_iv[0]:.1f}..{ws_last_mb:.1f} MB), the cell-sorted "
                            f"curve (sort_every={ctrl_sort_every}) rises "
                            f"{_fmt_sig2(s_rise_pct)}% ({_fmt_sig2(s_rmin_ns)} to "
                            f"{_fmt_sig2(s_rmax_ns)} ns/particle, span "
                            f"{_fmt_sig2(s_rspan_ns)} ns/particle > sorted within-sweep "
                            f"k·noise = {_fmt_sig2(s_thresh_ns)} ns/particle, where "
                            f"k={k_val:.1f} times sorted within-sweep noise "
                            f"{_fmt_sig2(noise_within_sorted_ns)} ns/particle, max of "
                            f"{n_btb_s} sorted back-to-back pairs{med_span_s_str}) vs "
                            f"unsorted {_fmt_sig2(u_rise_pct)}% (span "
                            f"{_fmt_sig2(u_rspan_ns)} ns/particle), so the sorted curve "
                            f"is not flat within sorted within-sweep k·noise (max of "
                            f"{n_btb_s} sorted back-to-back pairs) but its rise is much "
                            f"smaller ({_fmt_sig2(rel_ratio)}x smaller in relative rise, "
                            f"{_fmt_sig2(span_ratio)}x smaller in span; over the full "
                            f"sweep sorted rises {_fmt_sig2(s_full_pct)}% from "
                            f"{_fmt_sig2(s_min_ns)} to {_fmt_sig2(s_max_ns)} ns/particle "
                            f"vs unsorted {_fmt_sig2(u_full_pct)}%), demonstrating that "
                            f"the unsorted penalty is locality-dependent and consistent "
                            f"with an L2 cache-residency effect as a stated hypothesis."
                        )

            if (
                noise_within_method_str == "unavailable"
                or noise_within_ns is None
                or thresh_ns is None
            ):
                mech_clause = (
                    f"{push_compare_sentence} In the 1-L4 fixed-density working-set "
                    f"sweep (nz = {nz_first}..{nz_last}, push+deposit kernel working "
                    f"set {kws_first_mb:.1f}..{kws_last_mb:.1f} MB and total "
                    f"field+current working set {ws_first_mb:.1f}..{ws_last_mb:.1f} MB "
                    f"vs {l2_mb:.1f} MB L2 cache), unsorted push+deposit {shape_desc}; "
                    f"unsorted within-sweep back-to-back duplicate noise is unavailable "
                    f"(n={n_btb_u} back-to-back pairs; {btb_vs_med_u_str}), so the "
                    f"noise-based L2-step, rise-onset, and flatness tests were not "
                    f"evaluated.{sorted_clause}"
                )
            elif (
                not rise_onset_found
                or len(rise_nz_iv) < 2
                or len(rise_kws_iv) < 2
                or len(rise_ws_iv) < 2
            ):
                mech_clause = (
                    f"{push_compare_sentence} In the 1-L4 fixed-density working-set "
                    f"sweep (nz = {nz_first}..{nz_last}, push+deposit kernel working "
                    f"set {kws_first_mb:.1f}..{kws_last_mb:.1f} MB and total "
                    f"field+current working set {ws_first_mb:.1f}..{ws_last_mb:.1f} MB "
                    f"vs {l2_mb:.1f} MB L2 cache), unsorted push+deposit {shape_desc}; "
                    f"no rise onset met the noise-aware rule (k·noise_within_sweep) "
                    f"(using {noise_desc} = {_fmt_sig2(noise_within_ns)} ns/particle, "
                    f"k={k_val:.1f}, threshold {_fmt_sig2(thresh_ns)} ns/particle; "
                    f"{btb_vs_med_u_str}), so the rise-regime numbers are not "
                    f"evaluated.{sorted_clause}"
                )
            elif bool(ws_block.get("shows_step_near_l2_cache", False)):
                which_ws = (
                    "push+deposit kernel working set"
                    if bool(ws_block.get("shows_step_near_l2_cache_kernel_ws", False))
                    else "total field+current working set"
                )
                mech_clause = (
                    f"{push_compare_sentence} In the 1-L4 fixed-density working-set "
                    f"sweep (nz = {nz_first}..{nz_last}, push+deposit kernel working "
                    f"set {kws_first_mb:.1f}..{kws_last_mb:.1f} MB and total "
                    f"field+current working set {ws_first_mb:.1f}..{ws_last_mb:.1f} MB "
                    f"vs {l2_mb:.1f} MB L2 cache), unsorted push+deposit {shape_desc} "
                    f"with a step on the {which_ws} across the {l2_mb:.1f} MB L2 size "
                    f"exceeding both k={k_val:.1f} times the {noise_desc} "
                    f"({_fmt_sig2(noise_within_ns)} ns/particle, threshold "
                    f"{_fmt_sig2(thresh_ns)} ns/particle) and the jumps implied by "
                    f"the two neighbouring intervals' slopes per log working set"
                    f"{noise_classes_clause}."
                    f"{sorted_clause}"
                )
            else:
                br_details: list[str] = []
                if len(kws_br_iv) == 2 and kws_br_jump is not None:
                    imp_parts: list[str] = []
                    if kws_imp_prev is not None:
                        imp_parts.append(
                            f"{_fmt_sig2(float(kws_imp_prev))} ns/particle from lower neighbour"
                        )
                    if kws_imp_next is not None:
                        imp_parts.append(
                            f"{_fmt_sig2(float(kws_imp_next))} ns/particle from upper neighbour"
                        )
                    imp_str = (
                        f" vs {', '.join(imp_parts)} implied by neighbour log-slopes"
                        if imp_parts
                        else ""
                    )
                    br_details.append(
                        f"across the L2-bracketing push+deposit kernel working-set "
                        f"interval {kws_br_iv[0]:.1f}..{kws_br_iv[1]:.1f} MB the "
                        f"jump is {_fmt_sig2(float(kws_br_jump))} ns/particle{imp_str}"
                    )
                if len(tws_br_iv) == 2 and tws_br_jump is not None:
                    br_details.append(
                        f"across the L2-bracketing total field+current working-set "
                        f"interval {tws_br_iv[0]:.1f}..{tws_br_iv[1]:.1f} MB the "
                        f"change is {_fmt_sig2(float(tws_br_jump))} ns/particle"
                    )
                br_detail_str = (
                    f" ({'; '.join(br_details)})" if br_details else ""
                )
                if kws_exceeds_noise and not kws_exceeds_trend:
                    step_outcome_clause = (
                        f"no discrete step at the {l2_mb:.1f} MB L2 cache size is "
                        f"resolved because across the L2-bracketing push+deposit "
                        f"kernel working-set interval the jump exceeds k={k_val:.1f} "
                        f"times the {noise_desc} ({_fmt_sig2(noise_within_ns)} "
                        f"ns/particle, threshold {_fmt_sig2(thresh_ns)} ns/particle) "
                        f"but does not exceed the jumps implied by the two "
                        f"neighbouring intervals' slopes per log working set — the "
                        f"jump exceeds the noise (unsorted within-sweep noise, max of "
                        f"{n_btb_u} back-to-back pairs) but not the trend"
                        f"{noise_classes_clause}{br_detail_str}."
                    )
                else:
                    step_outcome_clause = (
                        f"no discrete step at the {l2_mb:.1f} MB L2 cache size is "
                        f"resolved under the criterion requiring the L2-bracketing "
                        f"jump to exceed both k={k_val:.1f} times the {noise_desc} "
                        f"({_fmt_sig2(noise_within_ns)} ns/particle, threshold "
                        f"{_fmt_sig2(thresh_ns)} ns/particle) and the jumps implied by "
                        f"the two neighbouring intervals' slopes per log working set"
                        f"{noise_classes_clause}{br_detail_str}."
                    )
                mech_clause = (
                    f"{push_compare_sentence} In the 1-L4 fixed-density working-set "
                    f"sweep (nz = {nz_first}..{nz_last}, push+deposit kernel working "
                    f"set {kws_first_mb:.1f}..{kws_last_mb:.1f} MB and total "
                    f"field+current working set {ws_first_mb:.1f}..{ws_last_mb:.1f} MB "
                    f"vs {l2_mb:.1f} MB L2 cache), unsorted push+deposit {shape_desc}: "
                    f"under the noise-aware rise-onset rule (first interval i with "
                    f"jump > +k·noise_within_sweep and all jumps from i onward > "
                    f"-k·noise_within_sweep, using {noise_desc} = "
                    f"{_fmt_sig2(noise_within_ns)} ns/particle, k={k_val:.1f}, "
                    f"threshold {_fmt_sig2(thresh_ns)} ns/particle), the sustained "
                    f"rise is gradual, beginning in the push+deposit kernel "
                    f"working-set interval {rise_kws_iv[0]:.1f}..{rise_kws_iv[1]:.1f} MB "
                    f"(nz={rise_nz_iv[0]}..{rise_nz_iv[1]}, total field+current "
                    f"working set {rise_ws_iv[0]:.1f}..{rise_ws_iv[1]:.1f} MB), and "
                    f"{step_outcome_clause}{sorted_clause}"
                )
        else:
            mech_clause = (
                f"{push_compare_sentence} Without the working-set sweep, the cause "
                f"of this per-particle difference is not identified."
            )

        failed_checks = [
            c for c in consistency_records if not bool(c.get("consistency_pass", True))
        ]
        if failed_checks:
            fail_parts = [
                f"{c['series']} at grid={c['grid_shape'][0]}x{c['grid_shape'][1]}x"
                f"{c['grid_shape'][2]}, N_p={int(c['n_particles_requested']):,} "
                f"(|t_prof_s - t_med_s| = {_fmt_sig2(float(c['sorted_prof_diff_s']) * 1e3)} ms vs "
                f"unsorted tolerance {_fmt_sig2(float(c['consistency_tolerance_s']) * 1e3)} ms, "
                f"excess +{_fmt_sig2(float(c['consistency_excess_s']) * 1e3)} ms)"
                for c in failed_checks
            ]
            consistency_clause = (
                f" Sorted-control profiling consistency check "
                f"(|t_prof_s - t_med_s| <= |t_prof_u - t_med_u| + jitter_u): "
                f"failed on {'; '.join(fail_parts)}."
            )
        elif consistency_records:
            consistency_clause = (
                f" All {len(consistency_records)} sorted controls passed the "
                f"unsorted-referenced profiling consistency check "
                f"(|t_prof_s - t_med_s| <= |t_prof_u - t_med_u| + jitter_u)."
            )
        else:
            consistency_clause = ""

        if rep_comp.get("per_n_particles"):
            max_sp_eff_ref_pct = (
                float(
                    rep_comp.get(
                        "max_rel_spread_quoted_speedup_efficiency_at_ref_np", 0.0
                    )
                )
                * 100.0
            )
            max_sp_eff_quote_pct = (
                float(rep_comp.get("max_rel_spread_quoted_speedup_efficiency", 0.0))
                * 100.0
            )
            max_mig_ref_rel_pct = (
                float(rep_comp.get("max_rel_spread_migration_share_at_ref_np", 0.0))
                * 100.0
            )
            max_mig_ref_pp = float(
                rep_comp.get(
                    "max_abs_pct_point_spread_migration_share_at_ref_np", 0.0
                )
            )
            ws_vm_diag = rep_comp.get("working_set_cross_vm_offset_diagnostic", {})
            u_vm = ws_vm_diag.get("unsorted", {})
            s_vm = ws_vm_diag.get("sorted", {})
            vm_offset_clause = ""
            if int(u_vm.get("count", 0)) > 0:
                u_cnt = int(u_vm["count"])
                u_mean = float(u_vm["mean_signed_diff_ns_per_particle"])
                u_max_abs = float(
                    u_vm.get("max_abs_diff_ns_per_particle", abs(u_mean))
                )
                u_same = max(int(u_vm["n_positive"]), int(u_vm["n_negative"]))
                s_cnt = int(s_vm.get("count", 0))
                p_cross_phase_ns = ws_vm_diag.get(
                    "primary_noise_cross_phase_ns_per_particle"
                )
                p_n_cp_u = int(
                    ws_vm_diag.get(
                        "primary_n_cross_phase_pairs_unsorted",
                        ws_block.get("n_cross_phase_pairs_unsorted", 0)
                        if isinstance(ws_block, dict)
                        else 0,
                    )
                )
                lt_cross_phase = ws_vm_diag.get(
                    "cross_vm_offset_smaller_than_cross_phase_offset"
                )
                if p_cross_phase_ns is not None and lt_cross_phase is not None:
                    cmp_word = "smaller" if bool(lt_cross_phase) else "not smaller"
                    phase_cmp_str = (
                        f", which is {cmp_word} than the unsorted cross-phase offset "
                        f"({_fmt_sig2(float(p_cross_phase_ns))} ns/particle, max of "
                        f"{p_n_cp_u} cross-phase pairs)"
                    )
                else:
                    phase_cmp_str = ""
                if s_cnt > 0:
                    s_mean = float(s_vm["mean_signed_diff_ns_per_particle"])
                    s_max_abs = float(
                        s_vm.get("max_abs_diff_ns_per_particle", abs(s_mean))
                    )
                    s_same = max(int(s_vm["n_positive"]), int(s_vm["n_negative"]))
                    vm_offset_clause = (
                        f"; cross-VM per-nz working-set offset (repeat minus primary, "
                        f"not pooled into k·noise) has mean {u_mean:+.2f} ns/particle "
                        f"(max |diff| {_fmt_sig2(u_max_abs)} ns/particle, "
                        f"{u_same}/{u_cnt} same sign) on unsorted and "
                        f"{s_mean:+.2f} ns/particle (max |diff| "
                        f"{_fmt_sig2(s_max_abs)} ns/particle, {s_same}/{s_cnt} same sign) "
                        f"on sorted push+deposit{phase_cmp_str}"
                    )
                else:
                    vm_offset_clause = (
                        f"; cross-VM per-nz working-set offset (repeat minus primary, "
                        f"not pooled into k·noise) has mean {u_mean:+.2f} ns/particle "
                        f"(max |diff| {_fmt_sig2(u_max_abs)} ns/particle, "
                        f"{u_same}/{u_cnt} same sign) on unsorted push+deposit"
                        f"{phase_cmp_str}"
                    )
            run_variation_sentence = (
                f"Run-to-run relative spread between the two independent cloud VMs "
                f"is at most {_fmt_sig2(max_sp_eff_ref_pct)}% over the N_p = {ref_np:,} "
                f"speedup and efficiency quantities ({_fmt_sig2(max_sp_eff_quote_pct)}% "
                f"including the N_p = {low_np:,} speedups), and "
                f"{_fmt_sig2(max_mig_ref_rel_pct)}% relative "
                f"(<={_fmt_sig2(max_mig_ref_pp)} percentage points) on the "
                f"N_p = {ref_np:,} 2-GPU migration share{vm_offset_clause}; "
                f"quoted values are from the primary run, with two-run min-max "
                f"ranges shown in parentheses."
            )
        else:
            run_variation_sentence = "Quoted values are from a single run."

        payload["caption"] = (
            f"{headline_clause}{eff_clause}{mech_clause} "
            f"Two-GPU execution pays {_fmt_pct0(ref_mig_prof_pct)}"
            f"{_rep_range_pct('migration_share_of_2gpu_profiled_stage_sum')} of the "
            f"profiled stage sum ({_fmt_pct0(ref_mig_step_pct)}"
            f"{_rep_range_pct('migration_share_of_2gpu_step_median')} of the unprofiled "
            f"step median) in particle migration and compaction. In the separate "
            f"profiled pass, each stage time is reduced as the maximum over MPI ranks "
            f"and then summed across stages, which is why the 2-GPU stage sum can "
            f"exceed the profiled step total.{consistency_clause} "
            f"Electromagnetic FIT field updates (B + E advance) are ~{_fmt_pct0(ref_field_pct)} "
            f"of the 1-GPU profiled stage sum at N_p = {ref_np:,} "
            f"(<= {_fmt_pct0(max_field_share_ge_1e6_pct)} across N_p >= 10^6), so the "
            f"field solver's own scaling is not tested in the particle-dominated "
            f"regime. At N_p = {low_np:,} (field- and communication-latency-bound), "
            f"the 2-GPU speedups vs unsorted 1 L4 are {_fmt_sig2(low_sp_nccl)}x with "
            f"NCCL and {_fmt_sig2(low_sp_mpi)}x with pinned-host MPI. "
            f"{run_variation_sentence} Comparing 2 GPUs against 1 GPU on a "
            f"single node measures a 2-GPU speedup, not a multi-rank strong-scaling curve."
        )


def run_benchmark_suite(
    *,
    nx: int = DEFAULT_NX,
    ny: int = DEFAULT_NY,
    nz: int = DEFAULT_NZ,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_repeats: int = DEFAULT_N_REPEATS,
    n_steps: int = DEFAULT_N_STEPS,
    gpu_counts: list[int] | None = None,
    cpu_counts: list[int] | None = None,
    cpu_only: bool = False,
    data_path: Path = DATA_PATH,
    repeat_json_path: Path | None = None,
) -> dict | None:
    """Run the benchmark suite for the active MPI communicator size and merge JSON."""
    import cupy as cp
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())

    gpu_sweep = list(GPU_SWEEP_COUNTS if gpu_counts is None else gpu_counts)
    cpu_sweep = list(CPU_SWEEP_COUNTS if cpu_counts is None else cpu_counts)

    n_gpus = int(cp.cuda.runtime.getDeviceCount())
    dev_props = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = (
        dev_props["name"].decode("utf-8")
        if isinstance(dev_props["name"], bytes)
        else str(dev_props["name"])
    )
    l2_cache_bytes = int(dev_props.get("l2CacheSize", 0))
    _, total_vram = cp.cuda.runtime.memGetInfo()

    if rank == 0:
        print("=" * 78)
        print(
            f"019 Multi-GPU and MPI Coupled Particle Benchmark "
            f"({gpu_name}, MPI size={size})"
        )
        print(
            f"Grid: {nx}x{ny}x{nz} ({nx * ny * nz:,} cells), BC: periodic x/y, PEC z everywhere"
        )
        print(
            f"Protocol: {n_warmup} warm-up steps, {n_repeats} repeats x {n_steps} steps "
            f"(median and min/max reported)"
        )
        print("=" * 78)

    gpu_1l4_records: list[dict] = []
    gpu_2l4_nccl_records: list[dict] = []
    gpu_2l4_mpi_records: list[dict] = []
    gpu_1l4_half_records: list[dict] = []
    gpu_1l4_sorted_records: list[dict] = []
    gpu_2l4_nccl_sorted_records: list[dict] = []
    ws_unsorted_records: list[dict] = []
    ws_sorted_records: list[dict] = []
    ws_dup_unsorted_records: list[dict] = []
    ws_dup_sorted_records: list[dict] = []
    cpu_size_records: list[dict] = []

    if size == 1 and not cpu_only:
        for n_p in gpu_sweep:
            if rank == 0:
                print(f"\n[GPU Sweep, size=1] N_p = {n_p:,}")
            rec_1gpu = run_single_benchmark(
                n_particles=n_p,
                use_gpu=True,
                use_mpi=False,
                particles_comm="none",
                nx=nx,
                ny=ny,
                nz=nz,
                n_warmup=n_warmup,
                n_repeats=n_repeats,
                n_steps=n_steps,
                sort_every=0,
            )
            if rank == 0 and rec_1gpu is not None:
                gpu_1l4_records.append(rec_1gpu)
                print(
                    f"  {BASELINE_LABEL:<26s}: median = {rec_1gpu['step_median_s'] * 1e3:8.2f} ms "
                    f"[{rec_1gpu['step_min_s'] * 1e3:.2f}, {rec_1gpu['step_max_s'] * 1e3:.2f}] ms | "
                    f"{rec_1gpu['throughput_median_mparticles_per_s']:7.2f} Mparticles/s"
                )

            # Control (i): 1 L4 on nx x ny x (nz//2) with N_p/2 particles (same per-GPU state as 1 rank of size=2)
            rec_half = run_single_benchmark(
                n_particles=max(n_p // 2, 1),
                use_gpu=True,
                use_mpi=False,
                particles_comm="none",
                nx=nx,
                ny=ny,
                nz=max(nz // 2, 2),
                n_warmup=n_warmup,
                n_repeats=n_repeats,
                n_steps=n_steps,
                sort_every=0,
                parent_n_particles_requested=n_p,
            )
            if rank == 0 and rec_half is not None:
                gpu_1l4_half_records.append(rec_half)
                gh_c = rec_half["grid_shape"]
                gh_lbl = f"1 L4, {gh_c[0]}x{gh_c[1]}x{gh_c[2]} N_p/2"
                print(
                    f"  {gh_lbl:<26s}: median = {rec_half['step_median_s'] * 1e3:8.2f} ms | "
                    f"push+dep = {rec_half['stages_ns_per_particle']['boris_push_deposit_s']:5.2f} ns/particle"
                )

            # Control (ii): 1 L4 with sort_every=CONTROL_SORT_EVERY (pre-sorted + timed over >= sort_every steps)
            rec_1gpu_sort = run_single_benchmark(
                n_particles=n_p,
                use_gpu=True,
                use_mpi=False,
                particles_comm="none",
                nx=nx,
                ny=ny,
                nz=nz,
                n_warmup=n_warmup,
                n_repeats=n_repeats,
                n_steps=n_steps,
                sort_every=CONTROL_SORT_EVERY,
            )
            if rank == 0 and rec_1gpu_sort is not None:
                gpu_1l4_sorted_records.append(rec_1gpu_sort)
                sort_lbl_1 = f"1 L4, sort_every={CONTROL_SORT_EVERY}"
                print(
                    f"  {sort_lbl_1:<26s}: median = {rec_1gpu_sort['step_median_s'] * 1e3:8.2f} ms | "
                    f"push+dep = {rec_1gpu_sort['stages_ns_per_particle']['boris_push_deposit_s']:5.2f} ns/particle | "
                    f"sort = {rec_1gpu_sort['cell_sort_call_s'] * 1e3:5.2f} ms/call"
                )

        if rank == 0:
            _check_sorted_control_consistency(
                gpu_1l4_records,
                gpu_1l4_sorted_records,
                series_label="gpu_1l4_sorted_control",
            )

        # 1-L4 fixed-particle-density working-set sweep across nz in WORKING_SET_SWEEP_NZ
        # Run sweep points in an interleaved order (alternating small and large nz)
        # followed by two independent duplicate points at the end to measure cross-run noise.
        ref_np_ws = (
            BREAKDOWN_N_PARTICLES
            if BREAKDOWN_N_PARTICLES in gpu_sweep
            else gpu_sweep[-1]
        )
        exec_order_nz = (
            list(WORKING_SET_SWEEP_EXECUTION_ORDER_NZ)
            if int(nz) == DEFAULT_NZ
            else _interleaved_sweep_order(WORKING_SET_SWEEP_NZ)
        )
        dup_nz_values = (
            list(WORKING_SET_SWEEP_DUPLICATE_NZ)
            if int(nz) == DEFAULT_NZ
            else [int(nz), int(max(nz // 2, 2))]
        )
        if rank == 0:
            print(
                f"\n[1-L4 Working-Set Sweep, fixed density from N_p={ref_np_ws:,} at nz={nz}, "
                f"interleaved order={exec_order_nz}, duplicates={dup_nz_values}]"
            )
        for nz_sw in exec_order_nz:
            n_p_sw = max(int(round(ref_np_ws * (float(nz_sw) / float(nz)))), 1)
            rec_ws_u = run_single_benchmark(
                n_particles=n_p_sw,
                use_gpu=True,
                use_mpi=False,
                particles_comm="none",
                nx=nx,
                ny=ny,
                nz=int(nz_sw),
                n_warmup=n_warmup,
                n_repeats=n_repeats,
                n_steps=n_steps,
                sort_every=0,
            )
            rec_ws_s = run_single_benchmark(
                n_particles=n_p_sw,
                use_gpu=True,
                use_mpi=False,
                particles_comm="none",
                nx=nx,
                ny=ny,
                nz=int(nz_sw),
                n_warmup=n_warmup,
                n_repeats=n_repeats,
                n_steps=n_steps,
                sort_every=CONTROL_SORT_EVERY,
            )
            if rank == 0 and rec_ws_u is not None and rec_ws_s is not None:
                ws_unsorted_records.append(rec_ws_u)
                ws_sorted_records.append(rec_ws_s)
                print(
                    f"  nz={nz_sw:3d}, N_p={n_p_sw:10,d}, "
                    f"total_ws={rec_ws_u['working_set_mb']:6.2f} MB, "
                    f"kernel_ws={rec_ws_u['kernel_working_set_mb']:6.2f} MB | "
                    f"unsorted={rec_ws_u['stages_ns_per_particle']['boris_push_deposit_s']:5.2f} ns/p | "
                    f"sorted={rec_ws_s['stages_ns_per_particle']['boris_push_deposit_s']:5.2f} ns/p"
                )
        for nz_dup in dup_nz_values:
            n_p_dup = max(int(round(ref_np_ws * (float(nz_dup) / float(nz)))), 1)
            rec_dup_u = run_single_benchmark(
                n_particles=n_p_dup,
                use_gpu=True,
                use_mpi=False,
                particles_comm="none",
                nx=nx,
                ny=ny,
                nz=int(nz_dup),
                n_warmup=n_warmup,
                n_repeats=n_repeats,
                n_steps=n_steps,
                sort_every=0,
            )
            rec_dup_s = run_single_benchmark(
                n_particles=n_p_dup,
                use_gpu=True,
                use_mpi=False,
                particles_comm="none",
                nx=nx,
                ny=ny,
                nz=int(nz_dup),
                n_warmup=n_warmup,
                n_repeats=n_repeats,
                n_steps=n_steps,
                sort_every=CONTROL_SORT_EVERY,
            )
            if rank == 0 and rec_dup_u is not None and rec_dup_s is not None:
                ws_dup_unsorted_records.append(rec_dup_u)
                ws_dup_sorted_records.append(rec_dup_s)
                print(
                    f"  [duplicate] nz={nz_dup:3d}, N_p={n_p_dup:10,d} | "
                    f"unsorted={rec_dup_u['stages_ns_per_particle']['boris_push_deposit_s']:5.2f} ns/p | "
                    f"sorted={rec_dup_s['stages_ns_per_particle']['boris_push_deposit_s']:5.2f} ns/p"
                )
        if rank == 0:
            _check_sorted_control_consistency(
                ws_unsorted_records,
                ws_sorted_records,
                series_label="working_set_sweep",
            )
            if ws_dup_unsorted_records and ws_dup_sorted_records:
                _check_sorted_control_consistency(
                    ws_dup_unsorted_records,
                    ws_dup_sorted_records,
                    series_label="working_set_sweep_duplicates",
                )

    elif size == 2 and not cpu_only and n_gpus >= 2:
        for n_p in gpu_sweep:
            if rank == 0:
                print(f"\n[GPU Sweep, size=2] N_p = {n_p:,}")
            rec_nccl = run_single_benchmark(
                n_particles=n_p,
                use_gpu=True,
                use_mpi=True,
                particles_comm="nccl",
                nx=nx,
                ny=ny,
                nz=nz,
                n_warmup=n_warmup,
                n_repeats=n_repeats,
                n_steps=n_steps,
                sort_every=0,
            )
            if rank == 0 and rec_nccl is not None:
                gpu_2l4_nccl_records.append(rec_nccl)
                print(
                    f"  {'2 L4, NCCL, size=2':<26s}: median = {rec_nccl['step_median_s'] * 1e3:8.2f} ms "
                    f"[{rec_nccl['step_min_s'] * 1e3:.2f}, {rec_nccl['step_max_s'] * 1e3:.2f}] ms | "
                    f"{rec_nccl['throughput_median_mparticles_per_s']:7.2f} Mparticles/s"
                )

            rec_mpi = run_single_benchmark(
                n_particles=n_p,
                use_gpu=True,
                use_mpi=True,
                particles_comm="mpi",
                nx=nx,
                ny=ny,
                nz=nz,
                n_warmup=n_warmup,
                n_repeats=n_repeats,
                n_steps=n_steps,
                sort_every=0,
            )
            if rank == 0 and rec_mpi is not None:
                gpu_2l4_mpi_records.append(rec_mpi)
                print(
                    f"  {'2 L4, pinned MPI, size=2':<26s}: median = {rec_mpi['step_median_s'] * 1e3:8.2f} ms "
                    f"[{rec_mpi['step_min_s'] * 1e3:.2f}, {rec_mpi['step_max_s'] * 1e3:.2f}] ms | "
                    f"{rec_mpi['throughput_median_mparticles_per_s']:7.2f} Mparticles/s"
                )

            # Control (ii): 2 L4 NCCL with sort_every=CONTROL_SORT_EVERY (pre-sorted + timed over >= sort_every steps)
            rec_nccl_sort = run_single_benchmark(
                n_particles=n_p,
                use_gpu=True,
                use_mpi=True,
                particles_comm="nccl",
                nx=nx,
                ny=ny,
                nz=nz,
                n_warmup=n_warmup,
                n_repeats=n_repeats,
                n_steps=n_steps,
                sort_every=CONTROL_SORT_EVERY,
            )
            if rank == 0 and rec_nccl_sort is not None:
                gpu_2l4_nccl_sorted_records.append(rec_nccl_sort)
                sort_lbl_2 = f"2 L4, NCCL, sort={CONTROL_SORT_EVERY}"
                print(
                    f"  {sort_lbl_2:<26s}: median = {rec_nccl_sort['step_median_s'] * 1e3:8.2f} ms | "
                    f"push+dep = {rec_nccl_sort['stages_ns_per_particle']['boris_push_deposit_s']:5.2f} ns/particle | "
                    f"sort = {rec_nccl_sort['cell_sort_call_s'] * 1e3:5.2f} ms/call"
                )

        if rank == 0:
            _check_sorted_control_consistency(
                gpu_2l4_nccl_records,
                gpu_2l4_nccl_sorted_records,
                series_label="gpu_2l4_nccl_sorted_control",
            )

    if size in (1, 2, 4):
        for n_p in cpu_sweep:
            if rank == 0:
                print(f"\n[CPU MPI Sweep, size={size}] N_p = {n_p:,}")
            rec_cpu = run_single_benchmark(
                n_particles=n_p,
                use_gpu=False,
                use_mpi=(size > 1),
                particles_comm="mpi" if size > 1 else "none",
                nx=nx,
                ny=ny,
                nz=nz,
                n_warmup=n_warmup,
                n_repeats=n_repeats,
                n_steps=n_steps,
                sort_every=0,
            )
            if rank == 0 and rec_cpu is not None:
                cpu_size_records.append(rec_cpu)
                print(
                    f"  CPU size={size}                : median = {rec_cpu['step_median_s'] * 1e3:8.2f} ms "
                    f"[{rec_cpu['step_min_s'] * 1e3:.2f}, {rec_cpu['step_max_s'] * 1e3:.2f}] ms | "
                    f"{rec_cpu['throughput_median_mparticles_per_s']:7.2f} Mparticles/s"
                )

    if rank != 0:
        return None

    # Measure Species bytes per particle directly from Species buffers for metadata
    bpp_below_thresh = probe_species_bytes_per_particle(
        use_gpu=True,
        use_mpi=True,
        size=2,
        n_local_est=Species.DEFAULT_DEP_SCRATCH_ALIAS_THRESHOLD,
        track_ids=False,
        has_absorbing_boundary=True,
        sort_every=0,
    )
    bpp_above_thresh = probe_species_bytes_per_particle(
        use_gpu=True,
        use_mpi=True,
        size=2,
        n_local_est=Species.DEFAULT_DEP_SCRATCH_ALIAS_THRESHOLD + 1,
        track_ids=False,
        has_absorbing_boundary=True,
        sort_every=0,
    )
    bpp_sorted_below = probe_species_bytes_per_particle(
        use_gpu=True,
        use_mpi=True,
        size=2,
        n_local_est=Species.DEFAULT_DEP_SCRATCH_ALIAS_THRESHOLD,
        track_ids=False,
        has_absorbing_boundary=True,
        sort_every=CONTROL_SORT_EVERY,
    )
    bpp_sorted_above = probe_species_bytes_per_particle(
        use_gpu=True,
        use_mpi=True,
        size=2,
        n_local_est=Species.DEFAULT_DEP_SCRATCH_ALIAS_THRESHOLD + 1,
        track_ids=False,
        has_absorbing_boundary=True,
        sort_every=CONTROL_SORT_EVERY,
    )

    existing: dict = {}
    if data_path.exists():
        try:
            with data_path.open("r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except Exception:
            existing = {}

    if existing and (
        (size == 1 and not cpu_only)
        or not any(
            "working_set_bytes" in r for r in existing.get("gpu_1l4_baseline", [])
        )
    ):
        existing = {}

    cpu_records = existing.get(
        "cpu_mpi_scaling", {"size_1": [], "size_2": [], "size_4": []}
    )
    if cpu_size_records:
        cpu_records[f"size_{size}"] = cpu_size_records

    if ws_unsorted_records:
        ref_np_ws = (
            BREAKDOWN_N_PARTICLES
            if BREAKDOWN_N_PARTICLES in gpu_sweep
            else gpu_sweep[-1]
        )
        exec_order_nz = (
            list(WORKING_SET_SWEEP_EXECUTION_ORDER_NZ)
            if int(nz) == DEFAULT_NZ
            else _interleaved_sweep_order(WORKING_SET_SWEEP_NZ)
        )
        dup_nz_values = (
            list(WORKING_SET_SWEEP_DUPLICATE_NZ)
            if int(nz) == DEFAULT_NZ
            else [int(nz), int(max(nz // 2, 2))]
        )
        ws_payload = {
            "l2_cache_bytes": int(l2_cache_bytes),
            "l2_cache_mb": float(l2_cache_bytes / 1.0e6),
            "l2_cache_mib": float(l2_cache_bytes / (1024.0 * 1024.0)),
            "reference_n_particles_at_default_nz": int(ref_np_ws),
            "particles_per_cell": float(ref_np_ws / float(nx * ny * nz)),
            "nz_values": [int(v) for v in WORKING_SET_SWEEP_NZ],
            "execution_order_nz": [int(v) for v in exec_order_nz],
            "duplicate_nz_values": [int(v) for v in dup_nz_values],
            "unsorted": ws_unsorted_records,
            "sorted": ws_sorted_records,
            "independent_duplicates_unsorted": ws_dup_unsorted_records,
            "independent_duplicates_sorted": ws_dup_sorted_records,
        }
    else:
        ws_payload = existing.get("working_set_sweep", {})
        if isinstance(ws_payload, dict) and l2_cache_bytes > 0:
            ws_payload.setdefault("l2_cache_bytes", int(l2_cache_bytes))
            ws_payload.setdefault("l2_cache_mb", float(l2_cache_bytes / 1.0e6))
            ws_payload.setdefault(
                "l2_cache_mib", float(l2_cache_bytes / (1024.0 * 1024.0))
            )

    repeat_payload = _load_repeat_json(repeat_json_path)

    payload = {
        "benchmark": "019_particles_multi_gpu_scaling",
        "baseline_label": BASELINE_LABEL,
        "environment": {
            "preflight_machine": PREFLIGHT_MACHINE,
            "gpu_name": gpu_name,
            "gpu_count": n_gpus,
            "gpu_vram_gb_per_device": float(total_vram / 1.0e9),
            "gpu_l2_cache_bytes": int(l2_cache_bytes),
            "gpu_l2_cache_mb": float(l2_cache_bytes / 1.0e6),
            "mpi_vendor": str(MPI.get_vendor()),
            "mpi_library_version": str(MPI.Get_library_version()).strip(),
            "python_version": sys.version.split()[0],
            "omp_num_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
            "mkl_num_threads": int(os.environ.get("MKL_NUM_THREADS", "1")),
            "openblas_num_threads": int(os.environ.get("OPENBLAS_NUM_THREADS", "1")),
        },
        "configuration": {
            "grid_shape": [int(nx), int(ny), int(nz)],
            "cells_global": int(nx * ny * nz),
            "bc_low": ["periodic", "periodic", "pec"],
            "bc_high": ["periodic", "periodic", "pec"],
            "cfln": 0.45,
            "deposit_scheme": "esirkepov",
            "shape_order": 1,
            "sort_every": 0,
            "control_sort_every": CONTROL_SORT_EVERY,
            "track_ids": False,
            "seed": SEED,
            "n_warmup": int(n_warmup),
            "n_repeats": int(n_repeats),
            "n_steps_per_repeat": int(n_steps),
            "n_steps_per_repeat_sorted": max(int(n_steps), CONTROL_SORT_EVERY),
            "dep_scratch_alias_threshold": int(
                Species.DEFAULT_DEP_SCRATCH_ALIAS_THRESHOLD
            ),
            "species_gpu_bytes_per_particle_below_alias_threshold": int(
                bpp_below_thresh
            ),
            "species_gpu_bytes_per_particle_above_alias_threshold": int(
                bpp_above_thresh
            ),
            "species_gpu_bytes_per_particle_sorted_below_alias_threshold": int(
                bpp_sorted_below
            ),
            "species_gpu_bytes_per_particle_sorted_above_alias_threshold": int(
                bpp_sorted_above
            ),
            "load_balance_note": (
                "Uniform spatial particle initialization across z assigns "
                "approximately N_p / size particles to each equal-height slab "
                "and represents the best-case scenario for parallel load balance."
            ),
            "profiled_stage_note": (
                "Per-stage breakdown timings are measured in a separate profiled "
                "pass (ps.profile=True, solver._profile_mpi=True). Per-stage "
                "cudaStreamSynchronize() barriers inflate stage times relative to "
                "the unprofiled coupled step median, and for MPI runs (size > 1) "
                "each stage is reduced as the maximum over ranks (MPI.MAX) and "
                "then summed across stages (sum_s max_r t_{r,s} >= max_r sum_s t_{r,s}), "
                "which is why the 2-GPU stage sum can exceed the profiled step total."
            ),
            "baseline_and_efficiency_note": (
                f"The library default is periodic cell sorting "
                f"sort_every={CONTROL_SORT_EVERY}; headline 2-GPU speedup (b) "
                f"compares 2 L4 NCCL against 1 L4 with "
                f"sort_every={CONTROL_SORT_EVERY} (pre-sorted after warm-up and "
                f"timed over >= sort_every steps per repeat so amortised sort "
                f"cost is included). Speedup (a) compares unsorted runs "
                f"(sort_every=0). Parallel efficiency (c) is defined against the "
                f"equal-work 1-L4 half-domain N_p/2 control (t_half_ctrl / t_2gpu), "
                f"isolating communication and migration overhead from single-GPU "
                f"domain-size effects; apparent efficiency against the full-domain "
                f"1-L4 baseline ((t_1gpu / t_2gpu) / 2) is reported separately."
            ),
        },
        "gpu_1l4_baseline": (
            gpu_1l4_records
            if gpu_1l4_records
            else existing.get("gpu_1l4_baseline", [])
        ),
        "gpu_2l4_nccl": (
            gpu_2l4_nccl_records
            if gpu_2l4_nccl_records
            else existing.get("gpu_2l4_nccl", [])
        ),
        "gpu_2l4_pinned_mpi": (
            gpu_2l4_mpi_records
            if gpu_2l4_mpi_records
            else existing.get("gpu_2l4_pinned_mpi", [])
        ),
        "gpu_1l4_half_domain_control": (
            gpu_1l4_half_records
            if gpu_1l4_half_records
            else existing.get("gpu_1l4_half_domain_control", [])
        ),
        "gpu_1l4_sorted_control": (
            gpu_1l4_sorted_records
            if gpu_1l4_sorted_records
            else existing.get("gpu_1l4_sorted_control", [])
        ),
        "gpu_2l4_nccl_sorted_control": (
            gpu_2l4_nccl_sorted_records
            if gpu_2l4_nccl_sorted_records
            else existing.get("gpu_2l4_nccl_sorted_control", [])
        ),
        "working_set_sweep": ws_payload,
        "gpu_speedup_summary": [],
        "control_analysis": [],
        "sorted_control_consistency": [],
        "cpu_mpi_scaling": cpu_records,
        "cpu_speedup_summary": [],
        "caption": "",
    }
    _recompute_summaries(payload, repeat_payload=repeat_payload)

    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved benchmark results to {data_path}")
    return payload


def plot_scaling_figures(
    data_path: Path = DATA_PATH,
    fig_scaling_path: Path = FIG_SCALING_PATH,
    fig_breakdown_path: Path = FIG_BREAKDOWN_PATH,
    fig_working_set_path: Path = FIG_WORKING_SET_PATH,
    repeat_json_path: Path | None = None,
) -> None:
    """Generate multi-GPU scaling, stage-breakdown, and working-set figures strictly from the saved JSON data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wakis.figstyle import (
        FULL_WIDTH_IN,
        PALETTE as CATEGORICAL,
        apply_style,
        save as savefig,
    )

    def format_axes(
        ax,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
    ) -> None:
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title, pad=7.0)

    DOUBLE_COL = (FULL_WIDTH_IN, 4.15)

    with data_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    repeat_payload = _load_repeat_json(repeat_json_path)

    _recompute_summaries(data, repeat_payload=repeat_payload)

    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    apply_style()
    cfg = data["configuration"]
    ctrl_sort_every = int(cfg["control_sort_every"])
    nx, ny, nz = cfg["grid_shape"]
    n_warmup = int(cfg["n_warmup"])
    n_repeats = int(cfg["n_repeats"])
    n_steps = int(cfg["n_steps_per_repeat"])
    n_steps_sort = int(
        cfg.get("n_steps_per_repeat_sorted", max(n_steps, ctrl_sort_every))
    )

    r_1gpu = data["gpu_1l4_baseline"]
    r_nccl = data["gpu_2l4_nccl"]
    r_mpi = data["gpu_2l4_pinned_mpi"]
    r_1gpu_sort = data.get("gpu_1l4_sorted_control", [])
    r_nccl_sort = data.get("gpu_2l4_nccl_sorted_control", [])
    r_half = data.get("gpu_1l4_half_domain_control", [])
    r_cpu = data["cpu_mpi_scaling"]

    # ------------------------------------------------------------------
    # Figure 1: Coupled throughput and step wall time (median + min/max)
    # ------------------------------------------------------------------
    fig, (ax_tp, ax_dt) = plt.subplots(
        1, 2, figsize=DOUBLE_COL, constrained_layout=True
    )
    fig.suptitle(
        f"{nx}x{ny}x{nz} FIT grid (PEC z), {n_warmup} warm-up + "
        f"{n_repeats}x{n_steps} ({n_steps_sort} sorted) steps (median & min/max)",
        fontsize=9.0,
    )

    def _plot_band_series(
        ax,
        records: list[dict],
        y_med_key: str,
        y_min_key: str,
        y_max_key: str,
        *,
        scale: float = 1.0,
        color: str,
        marker: str,
        linestyle: str,
        label: str,
    ):
        if not records:
            return
        xs = np.array([r["n_particles_actual"] for r in records], dtype=float)
        ymed = np.array([r[y_med_key] for r in records], dtype=float) * scale
        ymin = np.array([r[y_min_key] for r in records], dtype=float) * scale
        ymax = np.array([r[y_max_key] for r in records], dtype=float) * scale
        yerr = np.vstack([np.maximum(ymed - ymin, 0.0), np.maximum(ymax - ymed, 0.0)])
        ax.errorbar(
            xs,
            ymed,
            yerr=yerr,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.6,
            markersize=5.2,
            capsize=2.8,
            label=label,
        )

    series_defs = [
        (
            r_1gpu_sort,
            CATEGORICAL[6],
            "X",
            "-",
            f"1 L4, sort={ctrl_sort_every} (default)",
        ),
        (
            r_nccl_sort,
            CATEGORICAL[2],
            "*",
            "-",
            f"2 L4, NCCL, sort={ctrl_sort_every} (default)",
        ),
        (r_1gpu, CATEGORICAL[0], "o", "--", "1 L4, sort=0 (unsorted)"),
        (r_nccl, CATEGORICAL[1], "s", "--", "2 L4, NCCL, sort=0"),
        (r_mpi, CATEGORICAL[4], "^", "-.", "2 L4, pinned MPI, sort=0"),
        (r_cpu.get("size_1", []), CATEGORICAL[3], "D", ":", "CPU 1 rank, size=1"),
        (r_cpu.get("size_2", []), CATEGORICAL[5], "v", "-.", "CPU MPI, size=2"),
        (r_cpu.get("size_4", []), CATEGORICAL[7], "P", "-", "CPU MPI, size=4"),
    ]

    for recs, col, mk, ls, lbl in series_defs:
        _plot_band_series(
            ax_tp,
            recs,
            "throughput_median_mparticles_per_s",
            "throughput_min_mparticles_per_s",
            "throughput_max_mparticles_per_s",
            color=col,
            marker=mk,
            linestyle=ls,
            label=lbl,
        )
        _plot_band_series(
            ax_dt,
            recs,
            "step_median_s",
            "step_min_s",
            "step_max_s",
            scale=1.0e3,
            color=col,
            marker=mk,
            linestyle=ls,
            label=lbl,
        )

    ax_tp.set_xscale("log")
    ax_tp.set_yscale("log")
    format_axes(
        ax_tp,
        xlabel=r"Global macroparticles $N_p$ [-]",
        ylabel=r"Throughput [Mparticles/s]",
        title="(a) Coupled Throughput",
    )

    ax_dt.set_xscale("log")
    ax_dt.set_yscale("log")
    ymin_dt, ymax_dt = ax_dt.get_ylim()
    ax_dt.set_ylim(ymin_dt * 0.25, ymax_dt * 1.40)

    for idx_sp, sp_rec in enumerate(data.get("gpu_speedup_summary", [])):
        n_p = sp_rec["n_particles"]
        if "headline_speedup_2gpu_nccl_sorted_default" in sp_rec:
            t_target_ms = float(sp_rec["t_2gpu_nccl_sorted_median_s"]) * 1.0e3
            sp_val = float(sp_rec["headline_speedup_2gpu_nccl_sorted_default"])
            ann_col = CATEGORICAL[2]
        else:
            t_target_ms = float(sp_rec["t_2gpu_nccl_median_s"]) * 1.0e3
            sp_val = float(sp_rec["speedup_2gpu_nccl_vs_1gpu_unsorted"])
            ann_col = CATEGORICAL[1]
        # Offset the first point's annotation horizontally inside the axes so it
        # is not clipped against the left axis spine.
        xytext_offset = (4, -11) if idx_sp == 0 else (0, -11)
        ha_align = "left" if idx_sp == 0 else "center"
        ax_dt.annotate(
            f"{_fmt_sig2(sp_val)}x",
            xy=(n_p, t_target_ms),
            xytext=xytext_offset,
            textcoords="offset points",
            ha=ha_align,
            va="top",
            fontsize=8.5,
            color=ann_col,
        )

    format_axes(
        ax_dt,
        xlabel=r"Global macroparticles $N_p$ [-]",
        ylabel=r"Coupled step time [ms/step]",
        title="(b) Coupled Step Time",
    )

    # Single shared legend across both panels of Figure 1
    handles_1, labels_1 = ax_tp.get_legend_handles_labels()
    if handles_1:
        fig.legend(
            handles_1,
            labels_1,
            loc="outside lower center",
            ncol=min(len(labels_1), 3),
            fontsize=8.5,
            frameon=False,
        )

    fig_scaling_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, str(fig_scaling_path), auto_legend_below=False)
    plt.close(fig)
    print(f"Saved scaling figure to {fig_scaling_path}")

    # ------------------------------------------------------------------
    # Figure 2: Stage breakdown + 2-GPU speedup & efficiency
    # ------------------------------------------------------------------
    fig2, (ax_bar, ax_sp) = plt.subplots(
        1, 2, figsize=DOUBLE_COL, constrained_layout=True
    )

    def _pick_breakdown_record(
        records: list[dict], target_np: int, *, use_parent_np: bool = False
    ) -> dict | None:
        if not records:
            return None
        for r in records:
            key_np = (
                int(r.get("parent_n_particles_requested", r["n_particles_requested"]))
                if use_parent_np
                else int(r["n_particles_requested"])
            )
            if key_np == int(target_np):
                return r
        return records[-1]

    rec_b_1 = _pick_breakdown_record(r_1gpu, BREAKDOWN_N_PARTICLES)
    rec_b_half = _pick_breakdown_record(
        r_half, BREAKDOWN_N_PARTICLES, use_parent_np=True
    )
    rec_b_1_sort = _pick_breakdown_record(r_1gpu_sort, BREAKDOWN_N_PARTICLES)
    rec_b_nccl = _pick_breakdown_record(r_nccl, BREAKDOWN_N_PARTICLES)
    rec_b_nccl_sort = _pick_breakdown_record(r_nccl_sort, BREAKDOWN_N_PARTICLES)
    rec_b_mpi = _pick_breakdown_record(r_mpi, BREAKDOWN_N_PARTICLES)

    if rec_b_half is not None:
        gh = rec_b_half["grid_shape"]
        half_bar_lbl = f"1L4 N/2 (nz={gh[2]})"
    else:
        half_bar_lbl = "1L4 N/2 (half)"

    bar_entries = [
        ("1L4, sort=0", rec_b_1),
        (half_bar_lbl, rec_b_half),
        (f"1L4, sort={ctrl_sort_every}", rec_b_1_sort),
        ("2L4 NCCL, sort=0", rec_b_nccl),
        (f"2L4 NCCL, sort={ctrl_sort_every}", rec_b_nccl_sort),
        ("2L4 MPI, sort=0", rec_b_mpi),
    ]
    bar_entries = [(lbl, r) for lbl, r in bar_entries if r is not None]
    np_label = (
        rec_b_1["n_particles_actual"]
        if rec_b_1 is not None
        else (
            bar_entries[0][1]["n_particles_actual"]
            if bar_entries
            else BREAKDOWN_N_PARTICLES
        )
    )

    fig2.suptitle(
        f"Stage Breakdown ($N_p = {np_label:,}$, per-stage max over ranks) "
        "& 2-GPU Speedup",
        fontsize=9.0,
    )

    stage_meta = [
        ("boris_push_deposit_s", "Boris push + deposit", CATEGORICAL[0]),
        ("cell_sort_amortised_s", "Cell sort (amortised)", CATEGORICAL[6]),
        ("particle_migration_s", "Particle migration", CATEGORICAL[1]),
        ("guard_current_sum_s", "Guard current reduction", CATEGORICAL[2]),
        ("fit_b_advance_s", "FIT B update", CATEGORICAL[3]),
        ("fit_e_advance_s", "FIT E update", CATEGORICAL[4]),
        ("fit_ghost_comm_s", "FIT field ghost comm", CATEGORICAL[5]),
    ]

    x_pos = np.arange(len(bar_entries), dtype=float)
    bottoms = np.zeros(len(bar_entries), dtype=float)
    for key, label, col in stage_meta:
        vals_ms = np.array(
            [float(r["stages_s"].get(key, 0.0)) * 1.0e3 for _, r in bar_entries],
            dtype=float,
        )
        ax_bar.bar(
            x_pos,
            vals_ms,
            bottom=bottoms,
            width=0.52,
            color=col,
            label=label,
        )
        bottoms += vals_ms

    # Mark unprofiled median step time and profiled step total for direct comparison
    unprofiled_ms = np.array(
        [float(r["step_median_s"]) * 1.0e3 for _, r in bar_entries], dtype=float
    )
    profiled_tot_ms = np.array(
        [float(r["stages_s"]["profiled_step_total_s"]) * 1.0e3 for _, r in bar_entries],
        dtype=float,
    )
    ax_bar.plot(
        x_pos,
        unprofiled_ms,
        linestyle="none",
        marker="D",
        markersize=5.2,
        color="black",
        label="Unprofiled step median",
    )
    ax_bar.plot(
        x_pos,
        profiled_tot_ms,
        linestyle="none",
        marker="o",
        markersize=5.2,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.2,
        label="Profiled step total",
    )

    max_bar_y = (
        float(np.max(np.maximum(np.maximum(bottoms, unprofiled_ms), profiled_tot_ms)))
        if len(bar_entries) > 0
        else 1.0
    )
    ax_bar.set_ylim(0.0, max_bar_y * 1.18)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(
        [lbl for lbl, _ in bar_entries], rotation=26, ha="right", fontsize=7.4
    )
    format_axes(
        ax_bar,
        ylabel="Time per step [ms]",
        title="(a) Stage Breakdown (max over ranks)",
    )
    ax_bar.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=2,
        fontsize=7.2,
        columnspacing=0.8,
        handletextpad=0.4,
        frameon=False,
    )

    # Panel (b): 2-GPU speedup on left axis (with 2x reference), parallel efficiency
    # on its own autoscaled right twin axis so efficiency 0.63 at 10^5 is unclipped.
    ax_eff = ax_sp.twinx()
    sp_summary = data.get("gpu_speedup_summary", [])
    if sp_summary:
        xs_sp = np.array([s["n_particles"] for s in sp_summary], dtype=float)
        if all("headline_speedup_2gpu_nccl_sorted_default" in s for s in sp_summary):
            sp_sort_arr = np.array(
                [s["headline_speedup_2gpu_nccl_sorted_default"] for s in sp_summary],
                dtype=float,
            )
            ax_sp.plot(
                xs_sp,
                sp_sort_arr,
                color=CATEGORICAL[2],
                marker="*",
                linestyle="-",
                linewidth=1.8,
                label=f"(b) NCCL, sort={ctrl_sort_every}",
            )
        sp_nccl_arr = np.array(
            [s["speedup_2gpu_nccl_vs_1gpu_unsorted"] for s in sp_summary], dtype=float
        )
        sp_mpi_arr = np.array(
            [s["speedup_2gpu_pinned_mpi_vs_1gpu_unsorted"] for s in sp_summary],
            dtype=float,
        )
        ax_sp.plot(
            xs_sp,
            sp_nccl_arr,
            color=CATEGORICAL[1],
            marker="s",
            linestyle="--",
            linewidth=1.7,
            label="(a) NCCL, sort=0",
        )
        ax_sp.plot(
            xs_sp,
            sp_mpi_arr,
            color=CATEGORICAL[4],
            marker="^",
            linestyle="-.",
            linewidth=1.7,
            label="Pinned MPI, sort=0",
        )
        eff_pts = [
            (
                float(s["n_particles"]),
                float(s["efficiency_2gpu_nccl_vs_half_domain_control"]),
            )
            for s in sp_summary
            if s.get("efficiency_2gpu_nccl_vs_half_domain_control") is not None
        ]
        if eff_pts:
            xs_eff = np.array([p[0] for p in eff_pts], dtype=float)
            eff_half_arr = np.array([p[1] for p in eff_pts], dtype=float)
            ax_eff.plot(
                xs_eff,
                eff_half_arr,
                color=CATEGORICAL[0],
                marker="o",
                linestyle=":",
                linewidth=1.6,
                label=r"(c) Eff. $t_{\mathrm{half}} / t_{2\mathrm{GPU}}$",
            )
    ax_sp.axhline(
        2.0,
        color="0.4",
        linestyle=":",
        linewidth=1.1,
        label="2x speedup ref",
    )
    ax_sp.axhline(
        1.0,
        color="0.7",
        linestyle="--",
        linewidth=0.9,
        label="1x parity",
    )
    ax_sp.set_xscale("log")
    ax_sp.margins(y=0.14)
    ax_eff.margins(y=0.16)
    ax_eff.grid(False)
    ax_eff.set_ylabel("Parallel efficiency [-]", color=CATEGORICAL[0])
    ax_eff.tick_params(axis="y", labelcolor=CATEGORICAL[0])
    format_axes(
        ax_sp,
        xlabel=r"Global macroparticles $N_p$ [-]",
        ylabel="2-GPU speedup [-]",
        title="(b) 2-GPU Speedup & Efficiency",
    )
    h_sp, l_sp = ax_sp.get_legend_handles_labels()
    h_eff, l_eff = ax_eff.get_legend_handles_labels()
    ax_sp.legend(
        h_sp + h_eff,
        l_sp + l_eff,
        loc="upper center",
        bbox_to_anchor=(0.45, -0.24),
        ncol=2,
        fontsize=7.3,
        columnspacing=0.9,
        handletextpad=0.4,
        frameon=False,
    )

    fig_breakdown_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig2, str(fig_breakdown_path), auto_legend_below=False)
    plt.close(fig2)
    print(f"Saved stage breakdown figure to {fig_breakdown_path}")

    # ------------------------------------------------------------------
    # Figure 3: 1-L4 kernel and total working-set sweeps
    # ------------------------------------------------------------------
    fig3, (ax_kws, ax_tws) = plt.subplots(
        1, 2, figsize=DOUBLE_COL, constrained_layout=True
    )
    fig3.suptitle(
        f"1-L4 Fixed-Density Working-Set Sweeps ({nx}x{ny}x$N_z$ FIT grid)",
        fontsize=9.0,
    )

    ws_block = data.get("working_set_sweep", {})
    ws_pts = ws_block.get("points_summary", []) if isinstance(ws_block, dict) else []
    if ws_pts:
        ws_mb = np.array([float(p["working_set_mb"]) for p in ws_pts], dtype=float)
        kws_mb = np.array(
            [
                float(
                    p.get(
                        "kernel_working_set_mb",
                        p["gather_deposit_kernel_working_set_mb"],
                    )
                )
                for p in ws_pts
            ],
            dtype=float,
        )
        ns_u = np.array(
            [float(p["unsorted_push_deposit_ns_per_particle"]) for p in ws_pts],
            dtype=float,
        )
        err_u_low = np.array(
            [
                float(
                    p.get(
                        "unsorted_whole_step_err_low_ns_per_particle",
                        p.get("unsorted_err_low_ns_per_particle", 0.0),
                    )
                )
                for p in ws_pts
            ],
            dtype=float,
        )
        err_u_high = np.array(
            [
                float(
                    p.get(
                        "unsorted_whole_step_err_high_ns_per_particle",
                        p.get("unsorted_err_high_ns_per_particle", 0.0),
                    )
                )
                for p in ws_pts
            ],
            dtype=float,
        )
        yerr_u = np.vstack([err_u_low, err_u_high])

        # Panel (a): Kernel working set (E, B_n, Wx, Wy, Wz) WITH L2 cache line
        ax_kws.errorbar(
            kws_mb,
            ns_u,
            yerr=yerr_u,
            color=CATEGORICAL[1],
            marker="s",
            linestyle="-",
            linewidth=1.6,
            markersize=4.6,
            capsize=2.5,
            label=r"sort=0 ($\pm$whole-step min/max/$N_p$)",
        )
        # Panel (b): Total field+current working set WITHOUT L2 cache line
        ax_tws.errorbar(
            ws_mb,
            ns_u,
            yerr=yerr_u,
            color=CATEGORICAL[0],
            marker="o",
            linestyle="-",
            linewidth=1.6,
            markersize=4.6,
            capsize=2.5,
            label=r"sort=0 ($\pm$whole-step min/max/$N_p$)",
        )
        if all("sorted_push_deposit_ns_per_particle" in p for p in ws_pts):
            ns_s = np.array(
                [float(p["sorted_push_deposit_ns_per_particle"]) for p in ws_pts],
                dtype=float,
            )
            err_s_low = np.array(
                [
                    float(
                        p.get(
                            "sorted_whole_step_err_low_ns_per_particle",
                            p.get("sorted_err_low_ns_per_particle", 0.0),
                        )
                    )
                    for p in ws_pts
                ],
                dtype=float,
            )
            err_s_high = np.array(
                [
                    float(
                        p.get(
                            "sorted_whole_step_err_high_ns_per_particle",
                            p.get("sorted_err_high_ns_per_particle", 0.0),
                        )
                    )
                    for p in ws_pts
                ],
                dtype=float,
            )
            yerr_s = np.vstack([err_s_low, err_s_high])
            ax_kws.errorbar(
                kws_mb,
                ns_s,
                yerr=yerr_s,
                color=CATEGORICAL[2],
                marker="d",
                linestyle="--",
                linewidth=1.5,
                markersize=4.6,
                capsize=2.5,
                label=rf"sort={ctrl_sort_every} ($\pm$whole-step min/max/$N_p$)",
            )
            ax_tws.errorbar(
                ws_mb,
                ns_s,
                yerr=yerr_s,
                color=CATEGORICAL[6],
                marker="X",
                linestyle="--",
                linewidth=1.5,
                markersize=4.6,
                capsize=2.5,
                label=rf"sort={ctrl_sort_every} ($\pm$whole-step min/max/$N_p$)",
            )
        l2_bytes = int(ws_block.get("l2_cache_bytes", 0))
        if l2_bytes > 0:
            l2_mb = float(l2_bytes) / 1.0e6
            ax_kws.axvline(
                l2_mb,
                color="0.25",
                linestyle=":",
                linewidth=1.3,
                label=f"L2 ({l2_mb:.1f} MB)",
            )
    ws_ylabel = r"Push+deposit [ns/particle]" + "\n" + r"($\pm$whole-step min/max/$N_p$)"
    format_axes(
        ax_kws,
        xlabel="Kernel working set (E, B, W) [MB]",
        ylabel=ws_ylabel,
        title="(a) 1-L4 Kernel Working Set",
    )
    ax_kws.yaxis.label.set_fontsize(8.0)
    ax_kws.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=1,
        fontsize=7.3,
        columnspacing=0.8,
        handletextpad=0.4,
        frameon=False,
    )
    format_axes(
        ax_tws,
        xlabel="Total field+current working set [MB]",
        ylabel=ws_ylabel,
        title="(b) 1-L4 Total Working Set",
    )
    ax_tws.yaxis.label.set_fontsize(8.0)
    ax_tws.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=1,
        fontsize=7.3,
        columnspacing=0.8,
        handletextpad=0.4,
        frameon=False,
    )

    fig_working_set_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig3, str(fig_working_set_path), auto_legend_below=False)
    plt.close(fig3)
    print(f"Saved working-set figure to {fig_working_set_path}")
    checks = data.get("sorted_control_consistency", [])
    if checks:
        print("\nSorted-control consistency check (tolerance = |t_prof_u - t_med_u| + jitter_u):")
        for c_rec in checks:
            status_str = "PASS" if c_rec["consistency_pass"] else "FAIL"
            print(
                f"  [{status_str}] {c_rec['series']} grid={c_rec['grid_shape']} "
                f"N_p={c_rec['n_particles_requested']:,}: "
                f"|t_prof_s - t_med_s| = {c_rec['sorted_prof_diff_s'] * 1e3:.3f} ms vs "
                f"tol = {c_rec['consistency_tolerance_s'] * 1e3:.3f} ms "
                f"(|t_prof_u - t_med_u| = {c_rec['unsorted_prof_overhead_s'] * 1e3:.3f} ms + "
                f"jitter_u = {c_rec['unsorted_step_jitter_s'] * 1e3:.3f} ms; "
                f"excess = {c_rec['consistency_excess_s'] * 1e3:+.3f} ms)"
            )
    if data.get("caption"):
        print(f"\nCaption (from JSON):\n  {data['caption']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-GPU and MPI scaling benchmark for wakis.particles."
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Internal worker mode: run a single benchmark configuration.",
    )
    parser.add_argument("--n-particles", type=int, default=100_000)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--use-mpi", action="store_true")
    parser.add_argument(
        "--particles-comm",
        type=str,
        default="auto",
        choices=("none", "auto", "nccl", "mpi"),
    )
    parser.add_argument("--nx", type=int, default=DEFAULT_NX)
    parser.add_argument("--ny", type=int, default=DEFAULT_NY)
    parser.add_argument("--nz", type=int, default=DEFAULT_NZ)
    parser.add_argument("--n-warmup", type=int, default=DEFAULT_N_WARMUP)
    parser.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--sort-every", type=int, default=0)
    parser.add_argument("--worker-out", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--repeat-json",
        type=Path,
        default=None,
        help="Optional repeat JSON path to include two-run min-max ranges in JSON and caption.",
    )
    parser.add_argument("--fig-scaling-path", type=Path, default=FIG_SCALING_PATH)
    parser.add_argument("--fig-breakdown-path", type=Path, default=FIG_BREAKDOWN_PATH)
    parser.add_argument(
        "--fig-working-set-path", type=Path, default=FIG_WORKING_SET_PATH
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run only the CPU MPI sweep for the active MPI communicator size.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Re-generate figures from existing JSON without running simulations.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a quick smoke benchmark on the smallest particle count (10^5) only.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.worker:
        rec = run_single_benchmark(
            n_particles=args.n_particles,
            use_gpu=args.use_gpu,
            use_mpi=args.use_mpi,
            particles_comm=args.particles_comm,
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            n_warmup=args.n_warmup,
            n_repeats=args.n_repeats,
            n_steps=args.n_steps,
            sort_every=args.sort_every,
        )
        if rec is not None and args.worker_out is not None:
            args.worker_out.parent.mkdir(parents=True, exist_ok=True)
            with args.worker_out.open("w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=2)
        return

    if args.plot_only:
        rank = 0
        size = 1
    else:
        from mpi4py import MPI

        rank = int(MPI.COMM_WORLD.Get_rank())
        size = int(MPI.COMM_WORLD.Get_size())

        gpu_counts = [100_000] if args.smoke else GPU_SWEEP_COUNTS
        cpu_counts = [100_000] if args.smoke else CPU_SWEEP_COUNTS
        run_benchmark_suite(
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            n_warmup=args.n_warmup,
            n_repeats=args.n_repeats,
            n_steps=args.n_steps,
            gpu_counts=gpu_counts,
            cpu_counts=cpu_counts,
            cpu_only=args.cpu_only,
            data_path=args.data_path,
            repeat_json_path=args.repeat_json,
        )

    if rank == 0 and args.data_path.exists():
        plot_scaling_figures(
            data_path=args.data_path,
            fig_scaling_path=args.fig_scaling_path,
            fig_breakdown_path=args.fig_breakdown_path,
            fig_working_set_path=args.fig_working_set_path,
            repeat_json_path=args.repeat_json,
        )


if __name__ == "__main__":
    main()
