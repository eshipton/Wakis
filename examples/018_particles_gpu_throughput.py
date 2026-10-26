# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Example 018 — GPU (CuPy RawKernel) vs 1-Thread NumPy CPU Particle-in-Cell Throughput
====================================================================================

Benchmarks milestone M2 of `wakis.particles` on an NVIDIA L4 GPU (`use_gpu=True`,
CuPy CUDA `RawKernel` implementation) against the 1-process, 1-thread unoptimised
vectorised NumPy M1 CPU baseline (`MKL_NUM_THREADS=OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1`)
on the same virtual machine across:

- Particle counts: ``N_p in {10^4, 10^5, 10^6, 10^7}`` macroparticles
- Grid sizes: ``32^3`` (``32,768`` cells) and ``128^3`` (``2,097,152`` cells)
- B-spline shape orders: ``p = 1`` (CIC) and ``p = 2`` (TSC), Galerkin gather,
  relativistic Boris pusher, and exact 3D Esirkepov current deposition
- Coupled ``solver.one_step()`` execution with an attached ``ParticleSolver`` on
  both CPU and GPU, alongside individual stage timings (``push_gather``,
  ``deposit``, ``sort``, ``field_step``) and their sum (``stage_sum_step_s``,
  labelled as sum of stage times).

Also quantifies:
- The effect of periodic cell sorting on GPU ``float64`` ``atomicAdd`` deposition
  throughput across particle-per-cell densities (``32^3`` and ``128^3`` for both
  ``p = 1`` and ``p = 2``).
- Per-edge Higham floating-point summation round-off bounds for ``atomicAdd`` vs
  CPU ``numpy.bincount`` across ``p in {1, 2, 3}``, coupled V2 self-consistent
  trajectories, and 1000-step discrete Gauss's law residuals.

Saves two publication-style figures in ``docs/img/`` (via `wakis.figstyle`) and
writes all hardware metadata, thread settings, validation bounds, and measured
timings to ``examples/018_particles_gpu_throughput.json``:
  1. ``docs/img/particles_gpu_throughput_scaling.png``
  2. ``docs/img/particles_gpu_kernel_breakdown.png``
"""

from __future__ import annotations

import os

# Pin BLAS/OpenMP/MKL thread counts to 1 BEFORE importing NumPy/SciPy so the
# 1-process CPU baseline is strictly 1-thread (including MKL sparse SpMV).
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import ctypes
import gc
import json
import logging
from pathlib import Path
import subprocess
import time

import cupy as cp
import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as e_charge
from scipy.constants import m_e

from wakis import GridFIT3D, SolverFIT3D, figstyle
from wakis.particles import (
    ParticleSolver,
    Species,
    deposit_current_esirkepov,
    gauss_residual,
)
from wakis.particles.gpu_kernels import (
    compute_centred_B_gpu,
    deposit_current_esirkepov_gpu,
    finalize_j_delta_gpu,
    launch_gather_and_push_gpu,
    launch_periodic_wrap_gpu,
)
from wakis.particles.shapes import esirkepov_edge_abs_sum_and_counts

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = REPO_ROOT / "docs" / "img"
JSON_PATH = Path(__file__).resolve().with_suffix(".json")

PARTICLE_COUNTS = [10_000, 100_000, 1_000_000, 10_000_000]
GRID_SIZES = [32, 128]
SHAPE_ORDERS = [1, 2]
SORT_EVERY = 50


def _pin_mkl_single_thread() -> int:
    """Enforce 1 thread in Intel MKL runtime if loaded and return max threads."""
    for libname in ("libmkl_rt.so", "libmkl_rt.so.2", "libmkl_rt.so.1"):
        try:
            mkl_lib = ctypes.CDLL(libname)
            if hasattr(mkl_lib, "MKL_Set_Num_Threads"):
                mkl_lib.MKL_Set_Num_Threads(ctypes.c_int(1))
            if hasattr(mkl_lib, "MKL_Get_Max_Threads"):
                mkl_lib.MKL_Get_Max_Threads.restype = ctypes.c_int
                return int(mkl_lib.MKL_Get_Max_Threads())
        except OSError:
            continue
    return 1


def _trim_host_memory() -> None:
    """Release Python GC objects and return free glibc heap pages to the OS."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _release_zero_pages(*arrays: np.ndarray) -> None:
    """Return physical pages of anonymous zeroed buffers to the OS via MADV_DONTNEED."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        madvise = libc.madvise
        madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        madvise.restype = ctypes.c_int
        page = 4096
        for arr in arrays:
            if isinstance(arr, np.ndarray) and arr.nbytes >= 2 * page:
                addr = int(arr.ctypes.data)
                aligned = (addr + page - 1) & ~(page - 1)
                end = (addr + int(arr.nbytes)) & ~(page - 1)
                if end > aligned:
                    madvise(aligned, end - aligned, 4)  # MADV_DONTNEED = 4
    except OSError:
        for arr in arrays:
            if isinstance(arr, np.ndarray):
                arr.fill(0.0)


def query_hardware_environment(mkl_max_threads: int) -> dict:
    """Query CPU, GPU (`nvidia-smi`), CuPy, CUDA runtime, and thread settings."""
    cpu_model = "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f_cpu:
            for line in f_cpu:
                if "model name" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        cpu_model = "unknown"

    gpu_name = "unknown"
    gpu_mem_mib = 0
    gpu_driver = "unknown"
    try:
        smi = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                timeout=5,
            )
            .decode("utf-8")
            .strip()
            .splitlines()[0]
        )
        parts = [p.strip() for p in smi.split(",")]
        if len(parts) >= 3:
            gpu_name = parts[0]
            gpu_mem_mib = int(float(parts[1]))
            gpu_driver = parts[2]
    except Exception:
        dev_props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = dev_props["name"].decode("utf-8")

    cuda_rt = cp.cuda.runtime.runtimeGetVersion()
    cuda_major = cuda_rt // 1000
    cuda_minor = (cuda_rt % 1000) // 10

    return {
        "machine": cpu_model,
        "cpu_count": int(os.cpu_count() or 1),
        "processes": 1,
        "thread_pinning": {
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
            "mkl_get_max_threads": int(mkl_max_threads),
        },
        "cpu_baseline_description": (
            f"1-process, 1-thread unoptimised vectorised NumPy M1 CPU baseline on this VM "
            f"({cpu_model}, MKL_NUM_THREADS=OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1)"
        ),
        "gpu_model": gpu_name,
        "gpu_memory_total_mib": gpu_mem_mib,
        "gpu_driver_version": gpu_driver,
        "cupy_version": str(cp.__version__),
        "cuda_runtime_version": f"{cuda_major}.{cuda_minor} ({cuda_rt})",
        "memory_bytes_per_particle": {
            "active_soa_bytes_with_ids": 64,
            "active_soa_bytes_without_ids": 56,
            "gpu_preallocated_scratch_bytes": 168,
            "total_gpu_bytes_with_ids": 232,
            "total_gpu_bytes_without_ids": 224,
            "field_buffers_128cubed_bytes": int(8 * 3 * (128**3) * 8),
            "field_buffers_128cubed_gb": float(8 * 3 * (128**3) * 8 / 1e9),
            "field_buffers_128cubed_gib": float(8 * 3 * (128**3) * 8 / (1024**3)),
        },
        "cpu_stripped_attributes": [
            "solver.imu",
            "solver.sigma",
            "solver.itA",
            "solver.J_old",
            "grid.L",
            "grid.iA",
            "grid.tL",
            "grid.itA",
            "ps._itA_arr",
            "ps._q0_buf",
            "ps._H_prev",
            "solver._particles_q_bg",
        ],
        "cpu_shared_csr_indptr": True,
        "coupled_timing_excludes_sorting": True,
        "coupled_timing_sort_note": (
            "ps.n_particle_steps is reset to 1 before timing < 50 coupled steps "
            "(sort_every = 50), so coupled_one_step_s excludes periodic cell sorting "
            "while stage_sum_step_s includes sort_amortized_s = sort_call_s / 50"
        ),
    }


def _populate_cpu_species(sp: Species, n_p: int, seed: int = 42) -> None:
    """Populate a CPU `Species` in chunks to minimise peak host allocation."""
    rng = np.random.default_rng(seed)
    sp.n_particles = n_p
    chunk = 250_000
    for s0 in range(0, n_p, chunk):
        s1 = min(s0 + chunk, n_p)
        m = s1 - s0
        sp._x[s0:s1] = rng.uniform(0.0, 1.0, size=m)
        sp._y[s0:s1] = rng.uniform(0.0, 1.0, size=m)
        sp._z[s0:s1] = rng.uniform(0.0, 1.0, size=m)
        sp._ux[s0:s1] = rng.normal(0.0, 0.08 * c_light, size=m)
        sp._uy[s0:s1] = rng.normal(0.0, 0.08 * c_light, size=m)
        sp._uz[s0:s1] = rng.normal(0.35 * c_light, 0.04 * c_light, size=m)
    sp._w[:n_p].fill(1.0)
    sp.u_is_synced = False


def _populate_gpu_species(sp: Species, n_p: int, seed: int = 42) -> None:
    """Populate a GPU `Species` in chunks on the device."""
    rs = cp.random.RandomState(seed)
    sp.n_particles = n_p
    chunk = 1_000_000
    for s0 in range(0, n_p, chunk):
        s1 = min(s0 + chunk, n_p)
        m = s1 - s0
        sp._x[s0:s1] = rs.uniform(0.0, 1.0, size=m, dtype=np.float64)
        sp._y[s0:s1] = rs.uniform(0.0, 1.0, size=m, dtype=np.float64)
        sp._z[s0:s1] = rs.uniform(0.0, 1.0, size=m, dtype=np.float64)
        sp._ux[s0:s1] = rs.normal(0.0, 0.08 * c_light, size=m, dtype=np.float64)
        sp._uy[s0:s1] = rs.normal(0.0, 0.08 * c_light, size=m, dtype=np.float64)
        sp._uz[s0:s1] = rs.normal(0.35 * c_light, 0.04 * c_light, size=m, dtype=np.float64)
    sp._w[:n_p].fill(1.0)
    sp.u_is_synced = False
    sp._ensure_gpu_scratch()


def benchmark_cpu_grid(n_grid: int) -> tuple[dict, str]:
    """Measure 1-thread NumPy M1 CPU coupled `solver.one_step()` and stage timings on an `n_grid^3` grid."""
    grid = GridFIT3D(
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        n_grid,
        n_grid,
        n_grid,
        boundary_at_domain_edge=False,
        verbose=0,
    )
    grid.grid = None
    _trim_host_memory()
    solver = SolverFIT3D(
        grid,
        cfln=0.45,
        bc_low=["periodic"] * 3,
        bc_high=["periodic"] * 3,
        use_gpu=False,
        verbose=0,
    )
    cpu_step_variant = getattr(
        solver.one_step, "__name__", type(solver.one_step).__name__
    )
    # Warmup step 0 (triggers _attrcleanup)
    solver.one_step()
    for attr in ("iDeps", "iDmu", "Dsigma"):
        if hasattr(solver, attr):
            setattr(solver, attr, None)
    _trim_host_memory()

    # Measure pure electromagnetic field-step time (H + E updates) before attaching particles
    n_field_steps = 20 if n_grid == 32 else 5
    t0 = time.perf_counter()
    for _ in range(n_field_steps):
        solver.one_step()
    field_step_s = (time.perf_counter() - t0) / n_field_steps

    # Attach ParticleSolver to solver so solver.one_step() executes the genuine
    # coupled H -> ParticleSolver._mid_step_hook -> E leapfrog cycle.
    ps = ParticleSolver(
        solver,
        species=[],
        pusher="boris",
        shape=1,
        gather="galerkin",
        sort_every=SORT_EVERY,
        profile=True,
    ).attach()

    # Free setup attributes not used during time-stepping so 128^3 + 10^7 particles
    # stays comfortably below the 2000 MB memguard cap while keeping tDsiDmuiDaC,
    # itDaiDepsDstC, E, H, J, ieps, and ps._itA_arr intact for genuine solver.one_step() execution.
    solver.imu = None
    solver.sigma = None
    solver.itA = None
    solver.J_old = None
    if hasattr(solver, "tDsiDmuiDaC") and hasattr(solver, "itDaiDepsDstC"):
        # sharing indptr is only valid when both operators have the same row structure
        assert np.array_equal(solver.itDaiDepsDstC.indptr, solver.tDsiDmuiDaC.indptr)
        solver.itDaiDepsDstC.indptr = solver.tDsiDmuiDaC.indptr
    grid.L = None
    grid.iA = None
    grid.tL = None
    grid.itA = None
    ps._itA_arr = float(n_grid * n_grid)
    ps._q0_buf = None
    ps._H_prev = None
    solver._particles_q_bg = None
    _trim_host_memory()

    out_by_order = {}
    for order in SHAPE_ORDERS:
        order_key = f"p{order}"
        out_by_order[order_key] = {}
        ps.shape = order
        ps._p_stag = order - 1
        for n_p in PARTICLE_COUNTS:
            n_reps = 5 if n_p == 10_000 else 3
            _release_zero_pages(
                solver.E.array,
                solver.H.array,
                solver.J.array,
                ps.J_particles,
                ps._B_n_buf,
            )
            ps._j_in_solver = False
            _trim_host_memory()
            sp = Species(
                name="electron",
                charge=-e_charge,
                mass=m_e,
                capacity=n_p,
                track_ids=False,
                use_gpu=False,
            )
            _populate_cpu_species(sp, n_p, seed=42 + order)

            # Measure sort_by_cell time (and leave particles sorted for steady-state timing)
            t_s0 = time.perf_counter()
            sp.sort_by_cell(ps.grid_map)
            sort_call_s = time.perf_counter() - t_s0
            sort_amortized_s = sort_call_s / SORT_EVERY
            _trim_host_memory()

            ps.species = [sp]
            ps._initialized_half_step = True
            ps.n_particle_steps = 1

            # Warmup coupled solver.one_step() on small counts
            if n_p <= 100_000:
                solver.one_step()
                ps.n_particle_steps = 1

            # Measure genuine coupled solver.one_step() on CPU and accumulate
            # ParticleSolver's internal push+gather and deposit stage times.
            pg_acc = 0.0
            dep_acc = 0.0
            coupled_acc = 0.0
            for r_i in range(n_reps):
                if n_grid == 128 and n_p == 10_000_000:
                    _release_zero_pages(
                        solver.E.array,
                        solver.H.array,
                        solver.J.array,
                        ps.J_particles,
                        ps._B_n_buf,
                    )
                    ps._j_in_solver = False
                    _trim_host_memory()
                t_c0 = time.perf_counter()
                solver.one_step()
                coupled_acc += time.perf_counter() - t_c0
                pg_acc += ps._last_push_gather_s
                dep_acc += ps._last_deposit_s
            coupled_step_s = coupled_acc / n_reps
            pg_s = pg_acc / n_reps
            dep_s = dep_acc / n_reps

            particle_only_s = pg_s + dep_s + sort_amortized_s
            stage_sum_step_s = particle_only_s + field_step_s

            out_by_order[order_key][str(n_p)] = {
                "n_particles": int(n_p),
                "n_steps_timed": int(n_reps),
                "push_gather_s": float(pg_s),
                "deposit_s": float(dep_s),
                "sort_call_s": float(sort_call_s),
                "sort_amortized_s": float(sort_amortized_s),
                "field_step_s": float(field_step_s),
                "particle_only_step_s": float(particle_only_s),
                "stage_sum_step_s": float(stage_sum_step_s),
                "coupled_one_step_s": float(coupled_step_s),
                "push_gather_particles_per_s": float(n_p / max(pg_s, 1e-12)),
                "deposit_particles_per_s": float(n_p / max(dep_s, 1e-12)),
                "particle_only_steps_per_s": float(n_p / max(particle_only_s, 1e-12)),
                "stage_sum_particle_steps_per_s": float(
                    n_p / max(stage_sum_step_s, 1e-12)
                ),
                "coupled_particle_steps_per_s": float(n_p / max(coupled_step_s, 1e-12)),
            }
            print(
                f"[CPU {n_grid}^3 p={order} N_p={n_p:8d}] "
                f"coupled={n_p / coupled_step_s:.3e} part-steps/s "
                f"(coupled_step={coupled_step_s*1e3:.2f} ms, stage_sum={stage_sum_step_s*1e3:.2f} ms, "
                f"pg={pg_s*1e3:.2f} ms, dep={dep_s*1e3:.2f} ms, "
                f"sort_amort={sort_amortized_s*1e3:.2f} ms, field={field_step_s*1e3:.2f} ms)"
            )
            ps.species = []
            del sp
            _trim_host_memory()

    del ps, solver, grid
    _trim_host_memory()
    return out_by_order, cpu_step_variant


def benchmark_gpu_grid(n_grid: int) -> tuple[dict, dict, str]:
    """Measure NVIDIA L4 GPU CuPy RawKernel timings on an `n_grid^3` periodic grid."""
    grid = GridFIT3D(
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        n_grid,
        n_grid,
        n_grid,
        boundary_at_domain_edge=False,
        verbose=0,
    )
    grid.grid = None
    _trim_host_memory()
    solver = SolverFIT3D(
        grid,
        cfln=0.45,
        bc_low=["periodic"] * 3,
        bc_high=["periodic"] * 3,
        use_gpu=True,
        verbose=0,
    )
    gpu_step_variant = getattr(
        solver.one_step, "__name__", type(solver.one_step).__name__
    )
    # Warmup step 0 (triggers _attrcleanup)
    solver.one_step()
    for attr in ("iDeps", "iDmu", "Dsigma"):
        if hasattr(solver, attr):
            setattr(solver, attr, None)
    grid.L = None
    grid.iA = None
    grid.tL = None
    _trim_host_memory()
    cp.cuda.Stream.null.synchronize()

    # Measure pure GPU electromagnetic field-step time (H + E updates)
    n_field_steps = 40 if n_grid == 32 else 20
    ev_start = cp.cuda.Event()
    ev_end = cp.cuda.Event()
    ev_start.record()
    for _ in range(n_field_steps):
        solver.one_step()
    ev_end.record()
    ev_end.synchronize()
    field_step_s = float(cp.cuda.get_elapsed_time(ev_start, ev_end) * 1e-3 / n_field_steps)

    Nx, Ny, Nz = solver.Nx, solver.Ny, solver.Nz
    N = solver.N
    dt = float(solver.dt)

    out_by_order = {}
    sorting_study = {}

    for order in SHAPE_ORDERS:
        order_key = f"p{order}"
        out_by_order[order_key] = {}
        sorting_study[order_key] = {}

        for n_p in PARTICLE_COUNTS:
            n_reps = 30 if n_p <= 100_000 else (15 if n_p == 1_000_000 else 10)
            sp = Species(
                name="electron",
                charge=-e_charge,
                mass=m_e,
                capacity=n_p,
                track_ids=False,
                use_gpu=True,
            )
            _populate_gpu_species(sp, n_p, seed=42 + order)
            ps = ParticleSolver(
                solver,
                species=[sp],
                pusher="boris",
                shape=order,
                gather="galerkin",
                sort_every=SORT_EVERY,
            ).attach()

            # 1. Measure unsorted Esirkepov deposit time before cell sorting
            launch_gather_and_push_gpu(
                sp,
                ps.grid_map,
                solver.E.toarray(),
                ps._B_n_buf,
                order=order,
                p_stag=order - 1,
                periodic=(True, True, True),
                dt=dt,
                pusher="boris",
                advance_pos=True,
            )
            cp.cuda.Stream.null.synchronize()
            ev_start.record()
            for _ in range(n_reps):
                ps._Wx_buf.fill(0.0)
                ps._Wy_buf.fill(0.0)
                ps._Wz_buf.fill(0.0)
                deposit_current_esirkepov_gpu(
                    sp._xi0_scratch,
                    sp._eta0_scratch,
                    sp._zeta0_scratch,
                    sp._xi1_scratch,
                    sp._eta1_scratch,
                    sp._zeta1_scratch,
                    sp._w,
                    sp.charge,
                    Nx,
                    Ny,
                    Nz,
                    order=order,
                    periodic=(True, True, True),
                    Wx_out=ps._Wx_buf,
                    Wy_out=ps._Wy_buf,
                    Wz_out=ps._Wz_buf,
                    n_launch=n_p,
                )
                finalize_j_delta_gpu(
                    ps._Wx_buf,
                    ps._Wy_buf,
                    ps._Wz_buf,
                    ps._itA_arr,
                    ps.J_particles,
                    ps._J_prev,
                    solver.J.array,
                    1.0 / dt,
                    N,
                )
            ev_end.record()
            ev_end.synchronize()
            dep_unsorted_s = float(cp.cuda.get_elapsed_time(ev_start, ev_end) * 1e-3 / n_reps)
            launch_periodic_wrap_gpu(ps.boundaries, sp)

            # 2. Measure sort_by_cell time on GPU
            sp.sort_by_cell(ps.grid_map)  # warmup
            _populate_gpu_species(sp, n_p, seed=100 + order)
            cp.cuda.Stream.null.synchronize()
            n_sort_reps = 5 if n_p <= 1_000_000 else 3
            t_sort_ms = 0.0
            for r_s in range(n_sort_reps):
                if r_s > 0:
                    _populate_gpu_species(sp, n_p, seed=100 + order + r_s)
                    cp.cuda.Stream.null.synchronize()
                ev_start.record()
                sp.sort_by_cell(ps.grid_map)
                ev_end.record()
                ev_end.synchronize()
                t_sort_ms += float(cp.cuda.get_elapsed_time(ev_start, ev_end))
            sort_call_s = (t_sort_ms * 1e-3) / n_sort_reps
            sort_amortized_s = sort_call_s / SORT_EVERY

            # 3. Measure separate GPU stages (push+gather vs sorted deposit)
            H_curr = solver.H.toarray()
            ev_start.record()
            for _ in range(n_reps):
                compute_centred_B_gpu(H_curr, H_curr, ps._B_n_buf)
                launch_gather_and_push_gpu(
                    sp,
                    ps.grid_map,
                    solver.E.toarray(),
                    ps._B_n_buf,
                    order=order,
                    p_stag=order - 1,
                    periodic=(True, True, True),
                    dt=dt,
                    pusher="boris",
                    advance_pos=True,
                )
            ev_end.record()
            ev_end.synchronize()
            pg_s = float(cp.cuda.get_elapsed_time(ev_start, ev_end) * 1e-3 / n_reps)

            ev_start.record()
            for _ in range(n_reps):
                ps._Wx_buf.fill(0.0)
                ps._Wy_buf.fill(0.0)
                ps._Wz_buf.fill(0.0)
                deposit_current_esirkepov_gpu(
                    sp._xi0_scratch,
                    sp._eta0_scratch,
                    sp._zeta0_scratch,
                    sp._xi1_scratch,
                    sp._eta1_scratch,
                    sp._zeta1_scratch,
                    sp._w,
                    sp.charge,
                    Nx,
                    Ny,
                    Nz,
                    order=order,
                    periodic=(True, True, True),
                    Wx_out=ps._Wx_buf,
                    Wy_out=ps._Wy_buf,
                    Wz_out=ps._Wz_buf,
                    n_launch=n_p,
                )
                finalize_j_delta_gpu(
                    ps._Wx_buf,
                    ps._Wy_buf,
                    ps._Wz_buf,
                    ps._itA_arr,
                    ps.J_particles,
                    ps._J_prev,
                    solver.J.array,
                    1.0 / dt,
                    N,
                )
            ev_end.record()
            ev_end.synchronize()
            dep_s = float(cp.cuda.get_elapsed_time(ev_start, ev_end) * 1e-3 / n_reps)
            launch_periodic_wrap_gpu(ps.boundaries, sp)

            # 4. Measure full coupled solver.one_step() on GPU
            ps.n_particle_steps = 1
            solver.one_step()  # warmup
            ps.n_particle_steps = 1
            cp.cuda.Stream.null.synchronize()
            ev_start.record()
            for _ in range(n_reps):
                solver.one_step()
            ev_end.record()
            ev_end.synchronize()
            coupled_step_s = float(cp.cuda.get_elapsed_time(ev_start, ev_end) * 1e-3 / n_reps)

            particle_only_s = pg_s + dep_s + sort_amortized_s
            stage_sum_step_s = particle_only_s + field_step_s

            out_by_order[order_key][str(n_p)] = {
                "n_particles": int(n_p),
                "n_steps_timed": int(n_reps),
                "push_gather_s": float(pg_s),
                "deposit_s": float(dep_s),
                "deposit_unsorted_s": float(dep_unsorted_s),
                "sort_call_s": float(sort_call_s),
                "sort_amortized_s": float(sort_amortized_s),
                "field_step_s": float(field_step_s),
                "particle_only_step_s": float(particle_only_s),
                "stage_sum_step_s": float(stage_sum_step_s),
                "coupled_one_step_s": float(coupled_step_s),
                "push_gather_particles_per_s": float(n_p / max(pg_s, 1e-12)),
                "deposit_particles_per_s": float(n_p / max(dep_s, 1e-12)),
                "deposit_unsorted_particles_per_s": float(n_p / max(dep_unsorted_s, 1e-12)),
                "particle_only_steps_per_s": float(n_p / max(particle_only_s, 1e-12)),
                "stage_sum_particle_steps_per_s": float(
                    n_p / max(stage_sum_step_s, 1e-12)
                ),
                "coupled_particle_steps_per_s": float(n_p / max(coupled_step_s, 1e-12)),
            }
            sorting_study[order_key][str(n_p)] = {
                "sorted_deposit_s": float(dep_s),
                "unsorted_deposit_s": float(dep_unsorted_s),
                "sorted_deposit_particles_per_s": float(n_p / max(dep_s, 1e-12)),
                "unsorted_deposit_particles_per_s": float(n_p / max(dep_unsorted_s, 1e-12)),
                "sorting_deposit_speedup": float(dep_unsorted_s / max(dep_s, 1e-12)),
            }
            print(
                f"[GPU {n_grid}^3 p={order} N_p={n_p:8d}] "
                f"coupled={n_p / coupled_step_s:.3e} part-steps/s "
                f"(coupled_step={coupled_step_s*1e3:.2f} ms, stage_sum={stage_sum_step_s*1e3:.2f} ms, "
                f"pg={pg_s*1e3:.2f} ms, dep={dep_s*1e3:.2f} ms, "
                f"sort_amort={sort_amortized_s*1e3:.2f} ms, field={field_step_s*1e3:.2f} ms)"
            )

            solver._mid_step_hooks.clear()
            solver._mid_step_hook_takes_arg.clear()
            del ps, sp
            cp.get_default_memory_pool().free_all_blocks()
            _trim_host_memory()

    del solver, grid
    cp.get_default_memory_pool().free_all_blocks()
    _trim_host_memory()
    return out_by_order, sorting_study, gpu_step_variant


def compute_speedups(cpu_data: dict, gpu_data: dict) -> dict:
    """Compute GPU vs 1-thread NumPy M1 CPU speedups for every configuration."""
    speedups = {}
    for order_key in ("p1", "p2"):
        speedups[order_key] = {}
        for np_key in [str(n) for n in PARTICLE_COUNTS]:
            c = cpu_data[order_key][np_key]
            g = gpu_data[order_key][np_key]
            speedups[order_key][np_key] = {
                "baseline": "1-process 1-thread unoptimised vectorised NumPy M1 CPU on this VM",
                "coupled_speedup": float(
                    c["coupled_one_step_s"] / g["coupled_one_step_s"]
                ),
                "stage_sum_speedup": float(
                    c["stage_sum_step_s"] / g["stage_sum_step_s"]
                ),
                "particle_only_speedup": float(
                    c["particle_only_step_s"] / g["particle_only_step_s"]
                ),
                "push_gather_speedup": float(c["push_gather_s"] / g["push_gather_s"]),
                "deposit_speedup": float(c["deposit_s"] / g["deposit_s"]),
                "sort_speedup": float(c["sort_call_s"] / g["sort_call_s"]),
                "field_step_speedup": float(c["field_step_s"] / g["field_step_s"]),
            }
    return speedups


def compute_validation_metrics() -> dict:
    """Compute and record Higham round-off bounds, coupled V2 parity, and 1000-step Gauss residuals."""
    u_eps = float(np.finfo(np.float64).eps) * 0.5  # 2^-53

    # 1. Single-step Esirkepov atomicAdd vs bincount round-off bound (p in {1, 2, 3})
    Nx, Ny, Nz = 8, 8, 8
    n_p_dep = 500
    single_step = {}
    for order in (1, 2, 3):
        rng = np.random.default_rng(100 + order)
        xi0 = rng.uniform(0.0, Nx, size=n_p_dep)
        eta0 = rng.uniform(0.0, Ny, size=n_p_dep)
        zeta0 = rng.uniform(0.0, Nz, size=n_p_dep)
        xi1 = xi0 + rng.uniform(-0.35, 0.35, size=n_p_dep)
        eta1 = eta0 + rng.uniform(-0.35, 0.35, size=n_p_dep)
        zeta1 = zeta0 + rng.uniform(-0.35, 0.35, size=n_p_dep)
        qw = -e_charge * rng.uniform(0.5e6, 2.0e6, size=n_p_dep)

        Wx_c, Wy_c, Wz_c = deposit_current_esirkepov(
            xi0, eta0, zeta0, xi1, eta1, zeta1, qw, Nx, Ny, Nz, order=order, xp=np
        )
        Wx_g, Wy_g, Wz_g = deposit_current_esirkepov(
            cp.asarray(xi0),
            cp.asarray(eta0),
            cp.asarray(zeta0),
            cp.asarray(xi1),
            cp.asarray(eta1),
            cp.asarray(zeta1),
            cp.asarray(qw),
            Nx,
            Ny,
            Nz,
            order=order,
            xp=cp,
        )
        abs_c, cnt_c = esirkepov_edge_abs_sum_and_counts(
            xi0, eta0, zeta0, xi1, eta1, zeta1, qw, Nx, Ny, Nz, order=order
        )
        W_cpu = np.concatenate([Wx_c, Wy_c, Wz_c])
        W_gpu = np.concatenate([cp.asnumpy(Wx_g), cp.asnumpy(Wy_g), cp.asnumpy(Wz_g)])
        abs_all = np.concatenate(abs_c)
        cnt_all = np.concatenate(cnt_c).astype(np.float64)
        n_m1 = np.maximum(cnt_all - 1.0, 0.0)
        two_gamma_nm1 = 2.0 * (n_m1 * u_eps) / np.maximum(1.0 - n_m1 * u_eps, 0.5)
        edge_bound = two_gamma_nm1 * abs_all
        edge_diff = np.abs(W_gpu - W_cpu)
        nz = edge_bound > 0.0
        single_step[f"p{order}"] = {
            "grid": [Nx, Ny, Nz],
            "n_particles": int(n_p_dep),
            "max_n_contrib": int(np.max(cnt_all)),
            "mean_active_n_contrib": float(np.mean(cnt_all[cnt_all > 0])),
            "rel_linf_diff": float(np.max(edge_diff) / np.max(np.abs(W_cpu))),
            "max_edge_abs_diff_C": float(np.max(edge_diff)),
            "max_higham_edge_bound_C": float(np.max(edge_bound)),
            "max_bound_utilization": float(np.max(edge_diff[nz] / edge_bound[nz])),
        }

    # 2. Coupled 60-step relativistic V2 plasma benchmark on 8x8x8 periodic grid
    coupled_v2 = {}
    for order in (1, 2):
        rng = np.random.default_rng(202604 + order)
        n_p_v2 = 256
        n_steps_v2 = 60
        x0 = rng.uniform(0.01, 0.07, size=n_p_v2)
        y0 = rng.uniform(0.01, 0.07, size=n_p_v2)
        z0 = rng.uniform(0.01, 0.09, size=n_p_v2)
        ux0 = rng.normal(0.0, 0.6 * c_light, size=n_p_v2)
        uy0 = rng.normal(0.0, 0.6 * c_light, size=n_p_v2)
        uz0 = rng.normal(0.2 * c_light, 0.6 * c_light, size=n_p_v2)
        w0 = rng.uniform(1.0e11, 2.0e11, size=n_p_v2)

        def _run_v2(on_gpu: bool):
            g_v2 = GridFIT3D(0.0, 0.08, 0.0, 0.08, 0.0, 0.10, 8, 8, 8, verbose=0)
            s_v2 = SolverFIT3D(
                g_v2,
                cfln=0.45,
                bc_low=["periodic"] * 3,
                bc_high=["periodic"] * 3,
                use_gpu=on_gpu,
                verbose=0,
            )
            sp_v2 = Species(name="e", charge=-e_charge, mass=m_e, use_gpu=on_gpu)
            sp_v2.add_particles(
                x0.copy(),
                y0.copy(),
                z0.copy(),
                ux0.copy(),
                uy0.copy(),
                uz0.copy(),
                w0.copy(),
            )
            ps_v2 = ParticleSolver(
                s_v2,
                species=[sp_v2],
                pusher="boris",
                shape=order,
                gather="galerkin",
                sort_every=15,
            ).attach()
            for _ in range(n_steps_v2):
                s_v2.one_step()
            return s_v2, sp_v2, ps_v2

        s_c, sp_c, ps_c = _run_v2(False)
        s_g, sp_g, ps_g = _run_v2(True)

        xi1_c, eta1_c, zeta1_c = ps_c.grid_map.to_logical(sp_c.x, sp_c.y, sp_c.z)
        vx_c = sp_c.ux / np.sqrt(
            1.0 + (sp_c.ux**2 + sp_c.uy**2 + sp_c.uz**2) / c_light**2
        )
        vy_c = sp_c.uy / np.sqrt(
            1.0 + (sp_c.ux**2 + sp_c.uy**2 + sp_c.uz**2) / c_light**2
        )
        vz_c = sp_c.uz / np.sqrt(
            1.0 + (sp_c.ux**2 + sp_c.uy**2 + sp_c.uz**2) / c_light**2
        )
        xi0_c = xi1_c - vx_c * float(s_c.dt) / float(ps_c.grid_map.dx[0])
        eta0_c = eta1_c - vy_c * float(s_c.dt) / float(ps_c.grid_map.dy[0])
        zeta0_c = zeta1_c - vz_c * float(s_c.dt) / float(ps_c.grid_map.dz[0])
        abs_c, cnt_c = esirkepov_edge_abs_sum_and_counts(
            xi0_c,
            eta0_c,
            zeta0_c,
            xi1_c,
            eta1_c,
            zeta1_c,
            sp_c.charge * sp_c.w,
            8,
            8,
            8,
            order=order,
        )
        cnt_all = np.concatenate(cnt_c).astype(np.float64)
        abs_all = np.concatenate(abs_c)
        n_m1 = np.maximum(cnt_all - 1.0, 0.0)
        two_gamma_nm1 = 2.0 * (n_m1 * u_eps) / np.maximum(1.0 - n_m1 * u_eps, 0.5)
        step_rel_bound = float(
            np.max(two_gamma_nm1 * abs_all) / max(float(np.max(abs_all)), 1e-30)
        )
        empirical_linear_budget = float(n_steps_v2 * step_rel_bound)

        x_c = np.column_stack([sp_c.x, sp_c.y, sp_c.z])
        x_g = np.column_stack(
            [cp.asnumpy(sp_g.x), cp.asnumpy(sp_g.y), cp.asnumpy(sp_g.z)]
        )
        u_c = np.column_stack([sp_c.ux, sp_c.uy, sp_c.uz])
        u_g = np.column_stack(
            [cp.asnumpy(sp_g.ux), cp.asnumpy(sp_g.uy), cp.asnumpy(sp_g.uz)]
        )
        E_c = s_c.E.toarray()
        E_g = cp.asnumpy(s_g.E.toarray())
        H_c = s_c.H.toarray()
        H_g = cp.asnumpy(s_g.H.toarray())
        J_c = s_c.J.toarray()
        J_g = cp.asnumpy(s_g.J.toarray())

        coupled_v2[f"p{order}"] = {
            "grid": [8, 8, 8],
            "domain_lengths_m": [0.08, 0.08, 0.10],
            "n_steps": int(n_steps_v2),
            "n_particles": int(n_p_v2),
            "max_n_contrib": int(np.max(cnt_all)),
            "rel_x_diff": float(np.max(np.abs(x_g - x_c)) / np.max(np.abs(x_c))),
            "rel_u_diff": float(np.max(np.abs(u_g - u_c)) / np.max(np.abs(u_c))),
            "rel_E_diff": float(np.max(np.abs(E_g - E_c)) / np.max(np.abs(E_c))),
            "rel_H_diff": float(np.max(np.abs(H_g - H_c)) / np.max(np.abs(H_c))),
            "rel_J_diff": float(np.max(np.abs(J_g - J_c)) / np.max(np.abs(J_c))),
            "empirical_linear_growth_budget": empirical_linear_budget,
            "gauss_residual_gpu": float(gauss_residual(s_g, [sp_g], order=order)),
            "gauss_residual_cpu": float(gauss_residual(s_c, [sp_c], order=order)),
        }

    # 3. 1000-step GPU Gauss's law residuals on 6x6x6 uniform and graded periodic grids
    gauss_1000 = {}
    for order in (1, 2):
        for graded in (False, True):
            tag = f"p{order}_{'graded' if graded else 'uniform'}"
            if graded:
                x_n = np.array([0.0, 0.011, 0.025, 0.042, 0.056, 0.068, 0.08], dtype=float)
                y_n = np.array([0.0, 0.012, 0.027, 0.041, 0.058, 0.070, 0.08], dtype=float)
                z_n = np.array([0.0, 0.015, 0.032, 0.051, 0.068, 0.085, 0.10], dtype=float)
                g_g = GridFIT3D(
                    0.0,
                    0.08,
                    0.0,
                    0.08,
                    0.0,
                    0.10,
                    6,
                    6,
                    6,
                    x=x_n,
                    y=y_n,
                    z=z_n,
                    verbose=0,
                )
            else:
                g_g = GridFIT3D(0.0, 0.08, 0.0, 0.08, 0.0, 0.10, 6, 6, 6, verbose=0)
            s_g = SolverFIT3D(
                g_g,
                cfln=0.45,
                bc_low=["periodic"] * 3,
                bc_high=["periodic"] * 3,
                use_gpu=True,
                verbose=0,
            )
            rng = np.random.default_rng(77 + 10 * order + int(graded))
            n_p_g = 64
            sp_g = Species(name="e", charge=-e_charge, mass=m_e, use_gpu=True)
            sp_g.add_particles(
                rng.uniform(0.01, 0.07, size=n_p_g),
                rng.uniform(0.01, 0.07, size=n_p_g),
                rng.uniform(0.01, 0.09, size=n_p_g),
                rng.normal(0.0, 0.55 * c_light, size=n_p_g),
                rng.normal(0.0, 0.55 * c_light, size=n_p_g),
                rng.normal(0.0, 0.55 * c_light, size=n_p_g),
                rng.uniform(0.5e6, 1.5e6, size=n_p_g),
            )
            ps_g = ParticleSolver(
                s_g,
                species=[sp_g],
                pusher="boris",
                shape=order,
                gather="galerkin",
                sort_every=25,
            ).attach()
            for _ in range(1000):
                s_g.one_step()
            gauss_1000[tag] = float(ps_g.gauss_residual())

    cp.get_default_memory_pool().free_all_blocks()
    _trim_host_memory()
    return {
        "esirkepov_single_step_roundoff": single_step,
        "coupled_v2_8x8x8_60steps": coupled_v2,
        "gauss_1000_steps_6x6x6": gauss_1000,
    }


def make_figure_scaling(results: dict, out_path: Path) -> None:
    """Create Figure 1: Coupled `one_step()` particle steps/s and GPU speedup vs N_p."""
    fig, axes = figstyle.new_figure(
        width_in=figstyle.FULL_WIDTH_IN,
        height_in=4.0,
        nrows=1,
        ncols=2,
        gridspec_kw={"wspace": 0.50},
    )
    ax_rate, ax_spd = axes[0], axes[1]
    counts = np.array(PARTICLE_COUNTS, dtype=float)

    # Panel (a): Coupled one_step() throughput on GPU (L4) vs 1-thread NumPy CPU
    # Use palette indices 0..3 for GPU series and 4..7 for CPU series so no series share colours.
    gpu_configs = [
        ("32x32x32", "p1", r"GPU $32^3,p=1$", 0, "-"),
        ("128x128x128", "p1", r"GPU $128^3,p=1$", 1, "-"),
        ("32x32x32", "p2", r"GPU $32^3,p=2$", 2, "-"),
        ("128x128x128", "p2", r"GPU $128^3,p=2$", 3, "-"),
    ]
    for g_key, o_key, lbl, idx, ls in gpu_configs:
        rates = [
            results["grids"][g_key]["gpu"][o_key][str(n)][
                "coupled_particle_steps_per_s"
            ]
            for n in PARTICLE_COUNTS
        ]
        st = figstyle.series_style(idx, with_marker=True, ms=5.0)
        st["linestyle"] = ls
        ax_rate.loglog(counts, rates, label=lbl, **st)

    cpu_configs = [
        ("32x32x32", "p1", r"CPU $32^3,p=1$", 4),
        ("128x128x128", "p1", r"CPU $128^3,p=1$", 5),
        ("32x32x32", "p2", r"CPU $32^3,p=2$", 6),
        ("128x128x128", "p2", r"CPU $128^3,p=2$", 7),
    ]
    for g_key, o_key, lbl, idx in cpu_configs:
        rates = [
            results["grids"][g_key]["cpu"][o_key][str(n)][
                "coupled_particle_steps_per_s"
            ]
            for n in PARTICLE_COUNTS
        ]
        st = figstyle.series_style(idx, with_marker=True, ms=4.5)
        st["linestyle"] = "--"
        ax_rate.loglog(counts, rates, label=lbl, **st)

    ax_rate.set_xlabel(r"Macroparticle count $N_p$")
    ax_rate.set_ylabel(r"Coupled throughput [particle steps / s]")
    ax_rate.set_title("(a) Coupled step throughput")
    ax_rate.set_ylim(1e4, 1e9)
    figstyle.spectrum_grid(ax_rate)
    figstyle.place_legend_below(
        ax_rate,
        ncol=2,
        y_offset=-0.24,
        fontsize=8.5,
        handlelength=1.4,
        columnspacing=0.7,
        handletextpad=0.35,
    )

    # Panel (b): Coupled solver.one_step() speedup of L4 GPU over 1-thread NumPy CPU
    for g_key, o_key, lbl, idx, _ in gpu_configs:
        spds = [
            results["grids"][g_key]["speedup_vs_1proc_cpu"][o_key][str(n)][
                "coupled_speedup"
            ]
            for n in PARTICLE_COUNTS
        ]
        st = figstyle.series_style(idx, with_marker=True, ms=5.0)
        short_lbl = lbl.replace("GPU ", "")
        ax_spd.semilogx(counts, spds, label=short_lbl, **st)

    ax_spd.axhline(1.0, color=figstyle.INK, linestyle=":", linewidth=1.0)
    ax_spd.set_xlabel(r"Macroparticle count $N_p$")
    ax_spd.set_ylabel(r"Coupled speedup vs 1-thread CPU [$\times$]")
    ax_spd.set_title("(b) Coupled speedup vs 1T CPU")
    figstyle.spectrum_grid(ax_spd)
    figstyle.place_legend_below(
        ax_spd,
        ncol=2,
        y_offset=-0.24,
        fontsize=8.5,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    figstyle.save(fig, out_path, auto_legend_below=False)
    matplotlib.pyplot.close(fig)


def make_figure_breakdown(results: dict, out_path: Path) -> None:
    """Create Figure 2: Kernel stage time breakdown and cell-sorting deposit speedup."""
    fig, axes = figstyle.new_figure(
        width_in=figstyle.FULL_WIDTH_IN,
        height_in=4.0,
        nrows=1,
        ncols=2,
        gridspec_kw={"wspace": 0.54},
    )
    ax_bar, ax_sort = axes[0], axes[1]

    # Panel (a): Stacked stage breakdown [ms/step] on L4 GPU at N_p = 10^7
    cases = [
        ("32x32x32", "p1", r"$32^3$" + "\n" + r"$p=1$"),
        ("32x32x32", "p2", r"$32^3$" + "\n" + r"$p=2$"),
        ("128x128x128", "p1", r"$128^3$" + "\n" + r"$p=1$"),
        ("128x128x128", "p2", r"$128^3$" + "\n" + r"$p=2$"),
    ]
    x_pos = np.arange(len(cases))
    pg_ms = np.array(
        [
            results["grids"][g]["gpu"][o]["10000000"]["push_gather_s"] * 1e3
            for g, o, _ in cases
        ]
    )
    dep_ms = np.array(
        [
            results["grids"][g]["gpu"][o]["10000000"]["deposit_s"] * 1e3
            for g, o, _ in cases
        ]
    )
    sort_ms = np.array(
        [
            results["grids"][g]["gpu"][o]["10000000"]["sort_amortized_s"] * 1e3
            for g, o, _ in cases
        ]
    )
    fld_ms = np.array(
        [
            results["grids"][g]["gpu"][o]["10000000"]["field_step_s"] * 1e3
            for g, o, _ in cases
        ]
    )

    c0 = figstyle.series_style(0)["color"]
    c1 = figstyle.series_style(1)["color"]
    c2 = figstyle.series_style(2)["color"]
    c3 = figstyle.series_style(3)["color"]

    width = 0.52
    ax_bar.bar(x_pos, pg_ms, width, label="Gather+push", color=c0)
    ax_bar.bar(x_pos, dep_ms, width, bottom=pg_ms, label="Deposit", color=c1)
    ax_bar.bar(
        x_pos,
        sort_ms,
        width,
        bottom=pg_ms + dep_ms,
        label=f"Sort (1/{SORT_EVERY})",
        color=c2,
    )
    ax_bar.bar(
        x_pos,
        fld_ms,
        width,
        bottom=pg_ms + dep_ms + sort_ms,
        label="Field (H+E)",
        color=c3,
    )

    totals_ms = pg_ms + dep_ms + sort_ms + fld_ms
    coupled_ms = np.array(
        [
            results["grids"][g]["gpu"][o]["10000000"]["coupled_one_step_s"] * 1e3
            for g, o, _ in cases
        ]
    )
    ax_bar.plot(
        x_pos,
        coupled_ms,
        linestyle="none",
        marker="D",
        markersize=5.2,
        color=figstyle.INK,
        zorder=5,
        label="Coupled one_step",
    )
    top_ms = np.maximum(totals_ms, coupled_ms)
    for i, (g, o, _) in enumerate(cases):
        spd = results["grids"][g]["speedup_vs_1proc_cpu"][o]["10000000"][
            "coupled_speedup"
        ]
        ax_bar.text(
            x_pos[i],
            top_ms[i] * 1.04,
            f"{spd:.0f}x coupled\nvs 1T CPU",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=figstyle.INK,
        )

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([lbl for _, _, lbl in cases])
    ax_bar.set_ylabel(r"L4 GPU step time [ms]")
    ax_bar.set_title(r"(a) Stage sum & coupled step ($N_p = 10^7$)")
    ax_bar.set_ylim(0.0, float(np.max(top_ms)) * 1.38)
    figstyle.spectrum_grid(ax_bar)
    figstyle.place_legend_below(
        ax_bar,
        ncol=2,
        y_offset=-0.24,
        fontsize=8.5,
        handlelength=1.2,
        columnspacing=0.6,
        handletextpad=0.35,
    )

    # Panel (b): Sorted vs unsorted GPU deposit throughput vs N_p (all 4 grid/order combinations)
    counts = np.array(PARTICLE_COUNTS, dtype=float)
    sort_series = [
        ("32x32x32", "p1", "sorted", r"$32^3,p=1$ sort", 0, "-"),
        ("32x32x32", "p1", "unsorted", r"$32^3,p=1$ unsort", 0, "--"),
        ("128x128x128", "p1", "sorted", r"$128^3,p=1$ sort", 1, "-"),
        ("128x128x128", "p1", "unsorted", r"$128^3,p=1$ unsort", 1, "--"),
        ("32x32x32", "p2", "sorted", r"$32^3,p=2$ sort", 2, "-"),
        ("32x32x32", "p2", "unsorted", r"$32^3,p=2$ unsort", 2, "--"),
        ("128x128x128", "p2", "sorted", r"$128^3,p=2$ sort", 3, "-"),
        ("128x128x128", "p2", "unsorted", r"$128^3,p=2$ unsort", 3, "--"),
    ]
    for g_key, o_key, mode, lbl, idx, ls in sort_series:
        metric_name = (
            "sorted_deposit_particles_per_s"
            if mode == "sorted"
            else "unsorted_deposit_particles_per_s"
        )
        vals = [
            results["grids"][g_key]["sorting_study"][o_key][str(n)][metric_name]
            for n in PARTICLE_COUNTS
        ]
        st = figstyle.series_style(idx, with_marker=True, ms=4.5)
        st["linestyle"] = ls
        ax_sort.loglog(counts, vals, label=lbl, **st)

    ax_sort.set_xlabel(r"Macroparticle count $N_p$")
    ax_sort.set_ylabel(r"GPU deposit rate [particles / s]")
    ax_sort.set_title("(b) Sorted vs unsorted deposit")
    figstyle.spectrum_grid(ax_sort)
    figstyle.place_legend_below(
        ax_sort,
        ncol=2,
        y_offset=-0.24,
        fontsize=8.5,
        handlelength=1.3,
        columnspacing=0.6,
        handletextpad=0.35,
    )

    figstyle.save(fig, out_path, auto_legend_below=False)
    matplotlib.pyplot.close(fig)


def main() -> None:
    mkl_threads = _pin_mkl_single_thread()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    env = query_hardware_environment(mkl_threads)
    print(
        f"Benchmarking on CPU: {env['machine']} "
        f"(1 process, MKL_NUM_THREADS={env['thread_pinning']['MKL_NUM_THREADS']}, "
        f"OMP_NUM_THREADS={env['thread_pinning']['OMP_NUM_THREADS']})"
    )
    print(
        f"Benchmarking on GPU: {env['gpu_model']} "
        f"(CuPy {env['cupy_version']}, CUDA {env['cuda_runtime_version']})"
    )

    # Run all CPU grids before initialising CUDA/cuSPARSE/cuRAND device contexts
    # so host RSS stays well below the 2000 MB memguard limit.
    cpu_by_grid = {}
    cpu_step_variant = "unknown"
    for n_grid in GRID_SIZES:
        cpu_res, cpu_step_variant = benchmark_cpu_grid(n_grid)
        cpu_by_grid[n_grid] = cpu_res

    gpu_by_grid = {}
    sort_by_grid = {}
    gpu_step_variant = "unknown"
    # Run 128^3 before 32^3 so SolverFIT3D(128^3, use_gpu=True) builds and frees
    # its host setup matrices before cuRAND/Thrust/NVRTC libraries are paged into VmRSS.
    for n_grid in reversed(GRID_SIZES):
        gpu_res, sort_res, gpu_step_variant = benchmark_gpu_grid(n_grid)
        gpu_by_grid[n_grid] = gpu_res
        sort_by_grid[n_grid] = sort_res

    env["cpu_one_step_variant"] = cpu_step_variant
    env["gpu_one_step_variant"] = gpu_step_variant

    val_metrics = compute_validation_metrics()

    grids_payload = {}
    for n_grid in GRID_SIZES:
        g_key = f"{n_grid}x{n_grid}x{n_grid}"
        cpu_res = cpu_by_grid[n_grid]
        gpu_res = gpu_by_grid[n_grid]
        sort_res = sort_by_grid[n_grid]
        spd_res = compute_speedups(cpu_res, gpu_res)
        grids_payload[g_key] = {
            "grid_shape": [n_grid, n_grid, n_grid],
            "n_cells": int(n_grid**3),
            "cpu": cpu_res,
            "gpu": gpu_res,
            "speedup_vs_1proc_cpu": spd_res,
            "sorting_study": sort_res,
        }

    fig1_path = IMG_DIR / "particles_gpu_throughput_scaling.png"
    fig2_path = IMG_DIR / "particles_gpu_kernel_breakdown.png"

    payload = {
        "environment": env,
        "problem_sizes": {
            "grids": [[g, g, g] for g in GRID_SIZES],
            "particle_counts": PARTICLE_COUNTS,
            "shape_orders": SHAPE_ORDERS,
            "sort_every": SORT_EVERY,
        },
        "validation_metrics": val_metrics,
        "grids": grids_payload,
        "figures": [
            str(fig1_path.relative_to(REPO_ROOT)),
            str(fig2_path.relative_to(REPO_ROOT)),
        ],
    }

    make_figure_scaling(payload, fig1_path)
    make_figure_breakdown(payload, fig2_path)

    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved figures to {fig1_path} and {fig2_path}")
    print(f"Saved benchmark metrics to {JSON_PATH}")


if __name__ == "__main__":
    main()
