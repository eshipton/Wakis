# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Milestone M2 validation suite for GPU (`cupy`) kernels in `wakis.particles`.

Tests:
1. Separate GPU = CPU kernel equivalence to IEEE-754 float64 round-off:
   - Fused gather + push (`gather_and_push_kernel`) for Boris, Vay, and
     Higuera-Cary pushers, Galerkin and momentum-conserving gather, shape
     orders ``p = 1, 2, 3``, on uniform and graded 3D grids.
   - 3D Esirkepov current deposition (`deposit_esirkepov_kernel`) for
     ``p = 1, 2, 3`` against the Higham floating-point atomic summation bound
     ``N_contrib * eps_mach * |J|`` and discrete continuity identity
     ``S_tilde W = -(q^{n+1} - q^n)``.
   - Stream compaction (`compact_scan_*` / `compact_scatter_kernel`) and
     boundary path preparation (`boundary_prepare_paths_kernel`).
   - Periodic cell sorting (`compute_cell_index_kernel` + `cupy.argsort` +
     `permute_particles_kernel`).
2. Fused GPU field step (`wakis.gpu_fused.FusedStep`) mid-step hook ordering
   between ``H^{n-1/2} -> H^{n+1/2}`` and ``E^n -> E^{n+1}``, plus ``attach()``
   validation of hooked vs unhooked GPU step variants.
3. Coupled GPU = CPU equivalence for V2 (relativistic bunch, ``p = 1, 2``) and
   V3 (cold plasma oscillation, ``p = 1, 2``) with identical seeds.
4. Discrete Gauss's law residual ``<= 1e-12`` over ``>= 1000`` steps on GPU for
   both uniform and smoothly graded 3D grids.
5. GPU kernel source mutation checks:
   - Dropping the ``(1/3) * dS_a * dS_b`` Esirkepov transverse term in
     ``gpu_kernels._SOURCE`` must fail both GPU = CPU equivalence and the
     GPU Gauss residual test.
   - Swapping the primal/staggered B-spline flags in ``gpu_kernels._SOURCE``
     must fail GPU = CPU gather+push equivalence.
6. Asynchronous execution (zero device-to-host synchronization via a ~60 ms
   device spin kernel + CUDA event ``not ev.done`` detector, plus a code-path
   mutation inserting ``float(sp._x[0])`` in ``_mid_step_hook_gpu`` that makes
   the test fail) and CuPy memory-pool allocation checks (``cp.cuda.MemoryHook``
   verifying zero sub-allocations on non-sorting steps and zero pool growth on
   sorting steps).
7. Relativistic single-cell CFL guard at ``ParticleSolver.attach()``
   (``c * dt <= min_cell_size``) and device CFL error flag ``_d_cfl_error`` in
   ``esirkepov_1d_window_dev`` checked in ``gauss_residual()``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import c as c_light
from scipy.constants import e as e_charge
from scipy.constants import epsilon_0, m_e, mu_0

cp = pytest.importorskip("cupy")

from wakis import GridFIT3D, SolverFIT3D
from wakis.gpu_fused import FusedStep
from wakis.particles import (
    BeamInjector,
    Collector,
    LogicalGridMap,
    ParticleBoundaries,
    ParticleSolver,
    Species,
    cold_plasma_dispersion_discrete,
    deposit_charge,
    deposit_current_esirkepov,
    gather_fields,
    gauss_residual,
    push_momentum,
    push_position,
)
from wakis.particles.shapes import esirkepov_edge_abs_sum_and_counts
import wakis.particles.gpu_kernels as gpu_kernels


pytestmark = pytest.mark.skipif(
    not cp.cuda.is_available(), reason="CUDA GPU not available"
)

_EPS64 = float(np.finfo(np.float64).eps)  # 2^-52 ~= 2.220446e-16
_U64 = 0.5 * _EPS64  # IEEE-754 float64 unit roundoff 2^-53 ~= 1.110223e-16


def _higham_gamma(m_arr):
    """Higham (2002, Theorem 3.1) summation factor ``gamma_m = (m * u) / (1 - m * u)``."""
    mu = np.maximum(np.asarray(m_arr, dtype=np.float64), 0.0) * _U64
    return mu / (1.0 - mu)


def _make_grid(graded: bool = False, Nx: int = 8, Ny: int = 8, Nz: int = 10):
    if not graded:
        return GridFIT3D(
            xmin=0.0,
            xmax=0.08,
            ymin=0.0,
            ymax=0.08,
            zmin=0.0,
            zmax=0.10,
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            verbose=0,
        )
    s_x = np.linspace(0.0, 1.0, Nx + 1)
    s_y = np.linspace(0.0, 1.0, Ny + 1)
    s_z = np.linspace(0.0, 1.0, Nz + 1)
    x_nodes = 0.08 * (s_x + 0.18 * np.sin(2.0 * np.pi * s_x) / (2.0 * np.pi))
    y_nodes = 0.08 * (s_y + 0.15 * np.sin(2.0 * np.pi * s_y) / (2.0 * np.pi))
    z_nodes = 0.10 * (s_z + 0.20 * np.sin(2.0 * np.pi * s_z) / (2.0 * np.pi))
    return GridFIT3D(
        xmin=0.0,
        xmax=0.08,
        ymin=0.0,
        ymax=0.08,
        zmin=0.0,
        zmax=0.10,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        verbose=0,
    )


# ============================================================================
# 1. Separate GPU = CPU kernel equivalence tests
# ============================================================================


@pytest.mark.parametrize("pusher", ["boris", "vay", "hc"])
@pytest.mark.parametrize("gather", ["galerkin", "momentum"])
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("graded", [False, True])
def test_gpu_vs_cpu_fused_gather_and_push_kernel(pusher, gather, order, graded):
    """Verify fused ``gather_and_push_kernel`` on GPU matches CPU to IEEE-754 round-off.

    Tests all three relativistic pushers (Boris, Vay, Higuera-Cary), both
    Galerkin (energy-conserving, staggered order ``p - 1``) and momentum
    (staggered order ``p``) gather schemes, B-spline orders ``p = 1, 2, 3``,
    and both uniform and non-uniform (graded) tensor-product meshes.
    """
    grid = _make_grid(graded=graded, Nx=8, Ny=8, Nz=10)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    N = Nx * Ny * Nz

    rng = np.random.default_rng(100 + order * 10 + int(graded))
    E_flat_cpu = rng.normal(0.0, 2.5e5, size=3 * N).astype(np.float64)
    B_flat_cpu = rng.normal(0.0, 0.25, size=3 * N).astype(np.float64)
    E_flat_gpu = cp.asarray(E_flat_cpu)
    B_flat_gpu = cp.asarray(B_flat_cpu)

    n_p = 256
    x_init = rng.uniform(0.005, 0.075, size=n_p)
    y_init = rng.uniform(0.005, 0.075, size=n_p)
    z_init = rng.uniform(0.005, 0.095, size=n_p)
    ux_init = rng.normal(0.0, 1.2 * c_light, size=n_p)
    uy_init = rng.normal(0.0, 1.2 * c_light, size=n_p)
    uz_init = rng.normal(0.8 * c_light, 1.5 * c_light, size=n_p)
    w_init = rng.uniform(1e5, 1e6, size=n_p)

    sp_cpu = Species(use_gpu=False)
    sp_cpu.add_particles(x_init, y_init, z_init, ux_init, uy_init, uz_init, w_init)

    sp_gpu = Species(use_gpu=True)
    sp_gpu.add_particles(x_init, y_init, z_init, ux_init, uy_init, uz_init, w_init)

    gmap_cpu = LogicalGridMap.from_grid(grid, xp=np)
    gmap_gpu = LogicalGridMap.from_grid(grid, xp=cp)

    dt = 1.5e-12
    periodic = (True, True, False)
    E_ext = (1.5e4, -2.0e4, 3.0e4)
    B_ext = (0.02, -0.03, 0.12)

    # --- CPU reference step ---
    xi0_c, eta0_c, zeta0_c = gmap_cpu.to_logical(sp_cpu.x, sp_cpu.y, sp_cpu.z)
    E_tup_c = (E_flat_cpu[:N], E_flat_cpu[N : 2 * N], E_flat_cpu[2 * N : 3 * N])
    B_tup_c = (B_flat_cpu[:N], B_flat_cpu[N : 2 * N], B_flat_cpu[2 * N : 3 * N])
    Ex_c, Ey_c, Ez_c, Bx_c, By_c, Bz_c = gather_fields(
        xi0_c,
        eta0_c,
        zeta0_c,
        E_tup_c,
        B_tup_c,
        Nx,
        Ny,
        Nz,
        order=order,
        gather=gather,
        periodic=periodic,
        xp=np,
    )
    push_momentum(
        sp_cpu,
        Ex_c + E_ext[0],
        Ey_c + E_ext[1],
        Ez_c + E_ext[2],
        Bx_c + B_ext[0],
        By_c + B_ext[1],
        Bz_c + B_ext[2],
        dt=dt,
        pusher=pusher,
    )
    push_position(sp_cpu, dt=dt)
    xi1_c, eta1_c, zeta1_c = gmap_cpu.to_logical(sp_cpu.x, sp_cpu.y, sp_cpu.z)

    # --- GPU standalone gather_fields check ---
    E_tup_g = (E_flat_gpu[:N], E_flat_gpu[N : 2 * N], E_flat_gpu[2 * N : 3 * N])
    B_tup_g = (B_flat_gpu[:N], B_flat_gpu[N : 2 * N], B_flat_gpu[2 * N : 3 * N])
    Ex_g, Ey_g, Ez_g, Bx_g, By_g, Bz_g = gather_fields(
        cp.asarray(xi0_c),
        cp.asarray(eta0_c),
        cp.asarray(zeta0_c),
        E_tup_g,
        B_tup_g,
        Nx,
        Ny,
        Nz,
        order=order,
        gather=gather,
        periodic=periodic,
        xp=cp,
    )
    np.testing.assert_allclose(cp.asnumpy(Ex_g), Ex_c, rtol=4 * _EPS64, atol=4 * _EPS64)
    np.testing.assert_allclose(cp.asnumpy(Ey_g), Ey_c, rtol=4 * _EPS64, atol=4 * _EPS64)
    np.testing.assert_allclose(cp.asnumpy(Ez_g), Ez_c, rtol=4 * _EPS64, atol=4 * _EPS64)
    np.testing.assert_allclose(cp.asnumpy(Bx_g), Bx_c, rtol=4 * _EPS64, atol=4 * _EPS64)
    np.testing.assert_allclose(cp.asnumpy(By_g), By_c, rtol=4 * _EPS64, atol=4 * _EPS64)
    np.testing.assert_allclose(cp.asnumpy(Bz_g), Bz_c, rtol=4 * _EPS64, atol=4 * _EPS64)

    # --- GPU fused gather + push kernel ---
    p_stag = (order - 1) if gather == "galerkin" else order
    gpu_kernels.launch_gather_and_push_gpu(
        sp_gpu,
        gmap_gpu,
        E_flat_gpu,
        B_flat_gpu,
        order=order,
        p_stag=p_stag,
        periodic=periodic,
        dt=dt,
        pusher=pusher,
        advance_pos=True,
        E_ext_uniform=E_ext,
        B_ext_uniform=B_ext,
    )

    np.testing.assert_allclose(
        cp.asnumpy(sp_gpu.ux), sp_cpu.ux, rtol=8 * _EPS64, atol=8 * _EPS64
    )
    np.testing.assert_allclose(
        cp.asnumpy(sp_gpu.uy), sp_cpu.uy, rtol=8 * _EPS64, atol=8 * _EPS64
    )
    np.testing.assert_allclose(
        cp.asnumpy(sp_gpu.uz), sp_cpu.uz, rtol=8 * _EPS64, atol=8 * _EPS64
    )
    np.testing.assert_allclose(
        cp.asnumpy(sp_gpu.x), sp_cpu.x, rtol=8 * _EPS64, atol=8 * _EPS64
    )
    np.testing.assert_allclose(
        cp.asnumpy(sp_gpu.y), sp_cpu.y, rtol=8 * _EPS64, atol=8 * _EPS64
    )
    np.testing.assert_allclose(
        cp.asnumpy(sp_gpu.z), sp_cpu.z, rtol=8 * _EPS64, atol=8 * _EPS64
    )
    np.testing.assert_allclose(
        cp.asnumpy(sp_gpu._xi1_scratch[:n_p]),
        xi1_c,
        rtol=8 * _EPS64,
        atol=8 * _EPS64,
    )
    np.testing.assert_allclose(
        cp.asnumpy(sp_gpu._eta1_scratch[:n_p]),
        eta1_c,
        rtol=8 * _EPS64,
        atol=8 * _EPS64,
    )
    np.testing.assert_allclose(
        cp.asnumpy(sp_gpu._zeta1_scratch[:n_p]),
        zeta1_c,
        rtol=8 * _EPS64,
        atol=8 * _EPS64,
    )


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("periodic", [(True, True, True), (False, True, False)])
def test_gpu_vs_cpu_esirkepov_deposit_and_roundoff_bound(order, periodic):
    """Verify GPU Esirkepov current deposition matches CPU within Higham's pointwise per-edge bound.

    Round-off bound derivation
    --------------------------
    In ``deposit_esirkepov_kernel``, one CUDA thread per particle computes the
    3D Esirkepov flux weights on its ``(p + 2)^3`` window and accumulates into
    global ``float64`` edge arrays ``(Wx, Wy, Wz)`` via hardware ``atomicAdd``.
    Because warp scheduling across thread blocks is non-deterministic, an edge
    ``e`` receiving ``n(e)`` non-zero stencil contributions ``w_{e,k}`` sums
    those ``n(e)`` IEEE-754 ``float64`` values in an arbitrary order.
    By Higham's floating-point summation theorem (N. J. Higham, *Accuracy and
    Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002, Theorem 3.1), each
    summation order of ``n(e)`` terms is within ``gamma_{n(e)-1} * sum_k |w_{e,k}|``
    of the exact sum, so the difference between any two summation orders on edge
    ``e`` satisfies:

        |W_e^GPU - W_e^CPU| <= 2 * gamma_{n(e)-1} * sum_{k in C(e)} |w_{e,k}|,

    where ``gamma_m = (m * u) / (1 - m * u)`` and ``u = 2^-53 ~= 1.11e-16`` is
    the IEEE-754 ``float64`` unit roundoff. Furthermore, on the GPU itself, the
    discrete continuity identity ``(S_tilde W)_{i,j,k} = -(q^{n+1} - q^n)``
    must hold to round-off.
    """
    Nx, Ny, Nz = 8, 8, 8
    rng = np.random.default_rng(2026 + order)
    n_p = 500

    # Keep particles inside interior cells when non-periodic so no boundary clipping is needed
    xi0_c = rng.uniform(2.1, 5.9, size=n_p)
    eta0_c = rng.uniform(2.1, 5.9, size=n_p)
    zeta0_c = rng.uniform(2.1, 5.9, size=n_p)
    xi1_c = xi0_c + rng.uniform(-0.35, 0.35, size=n_p)
    eta1_c = eta0_c + rng.uniform(-0.35, 0.35, size=n_p)
    zeta1_c = zeta0_c + rng.uniform(-0.35, 0.35, size=n_p)
    qw_c = -e_charge * rng.uniform(0.5e6, 1.5e6, size=n_p)

    Wx_c, Wy_c, Wz_c = deposit_current_esirkepov(
        xi0_c,
        eta0_c,
        zeta0_c,
        xi1_c,
        eta1_c,
        zeta1_c,
        qw_c,
        Nx,
        Ny,
        Nz,
        order=order,
        periodic=periodic,
        xp=np,
    )

    xi0_g = cp.asarray(xi0_c)
    eta0_g = cp.asarray(eta0_c)
    zeta0_g = cp.asarray(zeta0_c)
    xi1_g = cp.asarray(xi1_c)
    eta1_g = cp.asarray(eta1_c)
    zeta1_g = cp.asarray(zeta1_c)
    qw_g = cp.asarray(qw_c)

    Wx_g, Wy_g, Wz_g = deposit_current_esirkepov(
        xi0_g,
        eta0_g,
        zeta0_g,
        xi1_g,
        eta1_g,
        zeta1_g,
        qw_g,
        Nx,
        Ny,
        Nz,
        order=order,
        periodic=periodic,
        xp=cp,
    )

    # Compute exact per-edge sum_{k in C(e)} |w_{e,k}| and true contribution count n(e)
    (abs_Wx, abs_Wy, abs_Wz), (cnt_Wx, cnt_Wy, cnt_Wz) = (
        esirkepov_edge_abs_sum_and_counts(
            xi0_c,
            eta0_c,
            zeta0_c,
            xi1_c,
            eta1_c,
            zeta1_c,
            qw_c,
            Nx,
            Ny,
            Nz,
            order=order,
            periodic=periodic,
        )
    )
    assert int(max(cnt_Wx.max(), cnt_Wy.max(), cnt_Wz.max())) > 10

    diff_x = np.abs(cp.asnumpy(Wx_g) - Wx_c)
    diff_y = np.abs(cp.asnumpy(Wy_g) - Wy_c)
    diff_z = np.abs(cp.asnumpy(Wz_g) - Wz_c)
    assert float(max(diff_x.max(), diff_y.max(), diff_z.max())) > 0.0

    assert np.all(diff_x <= 2.0 * _higham_gamma(cnt_Wx - 1) * abs_Wx)
    assert np.all(diff_y <= 2.0 * _higham_gamma(cnt_Wy - 1) * abs_Wy)
    assert np.all(diff_z <= 2.0 * _higham_gamma(cnt_Wz - 1) * abs_Wz)

    # Verify exact discrete continuity S_tilde W = -(q1 - q0) on the GPU
    q0_g = deposit_charge(
        xi0_g, eta0_g, zeta0_g, qw_g, Nx, Ny, Nz, order=order, periodic=periodic, xp=cp
    )
    q1_g = deposit_charge(
        xi1_g, eta1_g, zeta1_g, qw_g, Nx, Ny, Nz, order=order, periodic=periodic, xp=cp
    )
    Wx_3d = Wx_g.reshape((Nx, Ny, Nz), order="F")
    Wy_3d = Wy_g.reshape((Nx, Ny, Nz), order="F")
    Wz_3d = Wz_g.reshape((Nx, Ny, Nz), order="F")

    if periodic[0]:
        div_x = Wx_3d - cp.roll(Wx_3d, 1, axis=0)
    else:
        div_x = cp.zeros_like(Wx_3d)
        div_x[1:, :, :] = Wx_3d[1:, :, :] - Wx_3d[:-1, :, :]
        div_x[0, :, :] = Wx_3d[0, :, :]
    if periodic[1]:
        div_y = Wy_3d - cp.roll(Wy_3d, 1, axis=1)
    else:
        div_y = cp.zeros_like(Wy_3d)
        div_y[:, 1:, :] = Wy_3d[:, 1:, :] - Wy_3d[:, :-1, :]
        div_y[:, 0, :] = Wy_3d[:, 0, :]
    if periodic[2]:
        div_z = Wz_3d - cp.roll(Wz_3d, 1, axis=2)
    else:
        div_z = cp.zeros_like(Wz_3d)
        div_z[:, :, 1:] = Wz_3d[:, :, 1:] - Wz_3d[:, :, :-1]
        div_z[:, :, 0] = Wz_3d[:, :, 0]

    continuity_err = float(
        cp.max(cp.abs((div_x + div_y + div_z) + (q1_g - q0_g))) / cp.max(cp.abs(q0_g))
    )
    assert continuity_err <= 1e-14


def test_gpu_vs_cpu_stream_compaction_and_boundary_paths():
    """Verify GPU prefix-sum stream compaction and boundary path preparation match CPU."""
    rng = np.random.default_rng(777)
    n_p = 600  # spans multiple 256-thread blocks
    x = rng.uniform(0.0, 0.08, size=n_p)
    y = rng.uniform(0.0, 0.08, size=n_p)
    z = rng.uniform(0.0, 0.10, size=n_p)
    ux = rng.normal(0.0, 0.5 * c_light, size=n_p)
    uy = rng.normal(0.0, 0.5 * c_light, size=n_p)
    uz = rng.normal(0.0, 0.5 * c_light, size=n_p)
    w = rng.uniform(1e4, 1e5, size=n_p)
    keep_mask = rng.uniform(0.0, 1.0, size=n_p) > 0.37

    sp_cpu = Species(use_gpu=False)
    sp_cpu.add_particles(x, y, z, ux, uy, uz, w)
    rem_cpu = sp_cpu.keep(keep_mask)

    sp_gpu = Species(use_gpu=True)
    sp_gpu.add_particles(x, y, z, ux, uy, uz, w)
    rem_gpu = sp_gpu.keep(cp.asarray(keep_mask))

    assert rem_gpu == rem_cpu
    assert sp_gpu.n_particles == sp_cpu.n_particles
    np.testing.assert_array_equal(cp.asnumpy(sp_gpu.id), sp_cpu.id)
    np.testing.assert_array_equal(cp.asnumpy(sp_gpu.x), sp_cpu.x)
    np.testing.assert_array_equal(cp.asnumpy(sp_gpu.y), sp_cpu.y)
    np.testing.assert_array_equal(cp.asnumpy(sp_gpu.z), sp_cpu.z)
    np.testing.assert_array_equal(cp.asnumpy(sp_gpu.ux), sp_cpu.ux)
    np.testing.assert_array_equal(cp.asnumpy(sp_gpu.uy), sp_cpu.uy)
    np.testing.assert_array_equal(cp.asnumpy(sp_gpu.uz), sp_cpu.uz)
    np.testing.assert_array_equal(cp.asnumpy(sp_gpu.w), sp_cpu.w)


@pytest.mark.parametrize("graded", [False, True])
def test_gpu_vs_cpu_cell_sort(graded):
    """Verify GPU periodic cell sorting matches CPU stable cell sorting identically."""
    grid = _make_grid(graded=graded, Nx=8, Ny=8, Nz=10)
    gmap_cpu = LogicalGridMap.from_grid(grid, xp=np)
    gmap_gpu = LogicalGridMap.from_grid(grid, xp=cp)

    rng = np.random.default_rng(888 + int(graded))
    n_p = 512
    x = rng.uniform(0.002, 0.078, size=n_p)
    y = rng.uniform(0.002, 0.078, size=n_p)
    z = rng.uniform(0.002, 0.098, size=n_p)
    ux = rng.normal(0.0, 0.3 * c_light, size=n_p)
    uy = rng.normal(0.0, 0.3 * c_light, size=n_p)
    uz = rng.normal(0.0, 0.3 * c_light, size=n_p)
    w = rng.uniform(1e4, 1e5, size=n_p)

    sp_cpu = Species(use_gpu=False)
    sp_cpu.add_particles(x, y, z, ux, uy, uz, w)
    sp_cpu.sort_by_cell(gmap_cpu)

    sp_gpu = Species(use_gpu=True)
    sp_gpu.add_particles(x, y, z, ux, uy, uz, w)
    sp_gpu.sort_by_cell(gmap_gpu)

    np.testing.assert_array_equal(cp.asnumpy(sp_gpu.id), sp_cpu.id)
    np.testing.assert_array_equal(cp.asnumpy(sp_gpu.x), sp_cpu.x)
    np.testing.assert_array_equal(cp.asnumpy(sp_gpu.y), sp_cpu.y)
    np.testing.assert_array_equal(cp.asnumpy(sp_gpu.z), sp_cpu.z)
    np.testing.assert_array_equal(cp.asnumpy(sp_gpu.ux), sp_cpu.ux)
    np.testing.assert_array_equal(cp.asnumpy(sp_gpu.uy), sp_cpu.uy)
    np.testing.assert_array_equal(cp.asnumpy(sp_gpu.uz), sp_cpu.uz)
    np.testing.assert_array_equal(cp.asnumpy(sp_gpu.w), sp_cpu.w)


# ============================================================================
# 2. Fused GPU field step (`wakis/gpu_fused.py`) mid-step hook & attach()
# ============================================================================


@pytest.mark.parametrize("use_cpml", [False, True])
def test_gpu_fused_step_mid_hook_order_and_attach(use_cpml):
    """Confirm FusedStep executes mid_step between H and E updates and attach() accepts FusedStep."""
    grid = GridFIT3D(
        xmin=0.0,
        xmax=0.06,
        ymin=0.0,
        ymax=0.06,
        zmin=0.0,
        zmax=0.06,
        Nx=6,
        Ny=6,
        Nz=6,
        verbose=0,
    )
    bc = ["cpml", "cpml", "cpml"] if use_cpml else ["periodic", "periodic", "periodic"]
    solver = SolverFIT3D(
        grid,
        cfln=0.45,
        bc_low=bc,
        bc_high=bc,
        n_pml=2,
        use_gpu=True,
        verbose=0,
    )
    # Activate FusedStep
    solver._fused_step = FusedStep(solver)
    solver.one_step = solver._fused_step

    # Seed non-trivial E^0 and H^{-1/2} so H updates during the first half-step
    rng = np.random.default_rng(42)
    solver.E.array[:] = cp.asarray(rng.normal(0.0, 1e3, size=3 * solver.N))
    solver.H.array[:] = cp.asarray(rng.normal(0.0, 1.0, size=3 * solver.N))
    solver._set_ghosts_to_0()

    H_init = solver.H.toarray().copy()
    E_init = solver.E.toarray().copy()

    observed = {}

    def _inspect_mid_state(s):
        # At mid_step:
        # 1. _H_before_update must equal H_init (H^{n-1/2})
        # 2. s.H must already have been advanced to H^{n+1/2} != H_init
        # 3. s.E must still equal E_init (E^n before the E update)
        observed["H_prev_matches_init"] = bool(
            cp.all(s._H_before_update == H_init).get()
        )
        observed["H_curr_updated"] = bool(cp.any(s.H.toarray() != H_init).get())
        observed["E_still_at_n"] = bool(cp.all(s.E.toarray() == E_init).get())

    sp = Species(use_gpu=True)
    sp.add_particles(
        x=0.03, y=0.03, z=0.03, ux=0.0, uy=0.0, uz=0.2 * c_light, w=1.0e7
    )
    ps = ParticleSolver(solver, species=[sp], shape=1).attach()
    solver.register_mid_step(_inspect_mid_state)

    solver.one_step()

    assert observed["H_prev_matches_init"] is True
    assert observed["H_curr_updated"] is True
    assert observed["E_still_at_n"] is True
    # After the second half-step (fused_e), E^{n+1} has been updated
    assert bool(cp.any(solver.E.toarray() != E_init).get()) is True
    assert ps.n_particle_steps == 1

    # Also verify _one_step_with_abc wrapping _fused_step is accepted by attach()
    solver._one_step_backend = solver._fused_step
    solver.one_step = solver._one_step_with_abc
    ParticleSolver(solver, species=[sp], shape=1).attach()

    # And unhooked GPU step variants are rejected by attach()
    for attr in (
        "_one_step_dispersive",
        "_one_step_nonlinear",
        "_one_step_time_varying",
        "_one_step_tete",
    ):
        solver.one_step = getattr(solver, attr)
        with pytest.raises(ValueError, match="mid_step hooks"):
            ParticleSolver(solver, species=[sp], shape=1).attach()


# ============================================================================
# 3. Coupled V2 and V3 GPU = CPU equivalence and >= 1000-step GPU Gauss tests
# ============================================================================


@pytest.mark.parametrize("shape", [1, 2])
@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("use_fused", [False, True])
def test_gpu_vs_cpu_coupled_v2_and_gauss(shape, graded, use_fused):
    """Coupled V2 relativistic bunch run: GPU = CPU within empirical linear-growth round-off budget for p = 1, 2."""
    Nx, Ny, Nz = 8, 8, 8
    N = Nx * Ny * Nz
    n_steps = 60

    def _run(on_gpu: bool):
        grid = _make_grid(graded=graded, Nx=Nx, Ny=Ny, Nz=Nz)
        solver = SolverFIT3D(
            grid,
            cfln=0.45,
            bc_low=["periodic", "periodic", "periodic"],
            bc_high=["periodic", "periodic", "periodic"],
            use_gpu=on_gpu,
            verbose=0,
        )
        if on_gpu and use_fused:
            solver._fused_step = FusedStep(solver)
            solver.one_step = solver._fused_step

        rng = np.random.default_rng(202604 + shape)
        n_p = 256
        x0 = rng.uniform(0.01, 0.07, size=n_p)
        y0 = rng.uniform(0.01, 0.07, size=n_p)
        z0 = rng.uniform(0.01, 0.09, size=n_p)
        ux0 = rng.normal(0.0, 0.6 * c_light, size=n_p)
        uy0 = rng.normal(0.0, 0.6 * c_light, size=n_p)
        uz0 = rng.normal(0.8 * c_light, 0.5 * c_light, size=n_p)
        # Strong self-field charge weight so self-consistent E, B perturb x and u above machine zero
        w0 = rng.uniform(0.8e11, 1.2e11, size=n_p)

        sp = Species(use_gpu=on_gpu)
        sp.add_particles(x0, y0, z0, ux0, uy0, uz0, w0)
        ps = ParticleSolver(
            solver,
            species=[sp],
            pusher="boris",
            shape=shape,
            gather="galerkin",
            sort_every=20,
            B_ext=(0.0, 0.0, 0.08),
        ).attach()

        for _ in range(n_steps - 1):
            solver.one_step()
        if not on_gpu:
            x_prev, y_prev, z_prev = sp.x.copy(), sp.y.copy(), sp.z.copy()
        else:
            x_prev = y_prev = z_prev = None
        solver.one_step()
        return solver, sp, ps, (x_prev, y_prev, z_prev)

    s_cpu, sp_cpu, ps_cpu, prev_cpu = _run(on_gpu=False)
    s_gpu, sp_gpu, ps_gpu, _ = _run(on_gpu=True)

    assert ps_gpu.gauss_residual() <= 1e-12
    assert ps_cpu.gauss_residual() <= 1e-12

    # Compute exact single-step per-edge 2 * gamma_{n(e)-1} bound on the final step,
    # then form the empirical linear-growth budget n_steps * single_step_rel_bound
    # (an empirical multi-step tolerance budget rather than a formal Higham bound,
    # since coupled steps also include cuSPARSE SpMV and trajectory feedback).
    x_prev, y_prev, z_prev = prev_cpu
    inv_g = 1.0 / np.sqrt(
        1.0 + (sp_cpu.ux**2 + sp_cpu.uy**2 + sp_cpu.uz**2) / (c_light**2)
    )
    x_unw = x_prev + float(s_cpu.dt) * sp_cpu.ux * inv_g
    y_unw = y_prev + float(s_cpu.dt) * sp_cpu.uy * inv_g
    z_unw = z_prev + float(s_cpu.dt) * sp_cpu.uz * inv_g
    xi0_c, eta0_c, zeta0_c = ps_cpu.grid_map.to_logical(x_prev, y_prev, z_prev)
    xi1_c, eta1_c, zeta1_c = ps_cpu.grid_map.to_logical(x_unw, y_unw, z_unw)
    (abs_Wx, abs_Wy, abs_Wz), (cnt_Wx, cnt_Wy, cnt_Wz) = (
        esirkepov_edge_abs_sum_and_counts(
            xi0_c,
            eta0_c,
            zeta0_c,
            xi1_c,
            eta1_c,
            zeta1_c,
            sp_cpu.charge * sp_cpu.w,
            Nx,
            Ny,
            Nz,
            order=shape,
            periodic=(True, True, True),
        )
    )
    itA_c = ps_cpu._itA_arr
    inv_dt = 1.0 / float(s_cpu.dt)
    bound_Jx = 2.0 * _higham_gamma(cnt_Wx - 1) * abs_Wx * itA_c[:N] * inv_dt
    bound_Jy = 2.0 * _higham_gamma(cnt_Wy - 1) * abs_Wy * itA_c[N : 2 * N] * inv_dt
    bound_Jz = 2.0 * _higham_gamma(cnt_Wz - 1) * abs_Wz * itA_c[2 * N : 3 * N] * inv_dt
    J_cpu = ps_cpu.J_particles
    single_step_rel_bound = float(
        max(bound_Jx.max(), bound_Jy.max(), bound_Jz.max()) / np.max(np.abs(J_cpu))
    )
    empirical_linear_budget_rel = float(n_steps) * single_step_rel_bound

    E_cpu = s_cpu.E.toarray()
    H_cpu = s_cpu.H.toarray()
    E_gpu = cp.asnumpy(s_gpu.E.toarray())
    H_gpu = cp.asnumpy(s_gpu.H.toarray())
    J_gpu = cp.asnumpy(ps_gpu.J_particles)

    assert np.max(np.abs(J_gpu - J_cpu)) <= empirical_linear_budget_rel * np.max(
        np.abs(J_cpu)
    )
    assert np.max(np.abs(E_gpu - E_cpu)) <= empirical_linear_budget_rel * np.max(
        np.abs(E_cpu)
    )
    assert np.max(np.abs(H_gpu - H_cpu)) <= empirical_linear_budget_rel * np.max(
        np.abs(H_cpu)
    )

    # Compare particle positions and momenta sorted by particle ID
    ord_c = np.argsort(sp_cpu.id)
    ord_g = np.argsort(cp.asnumpy(sp_gpu.id))
    x_c, y_c, z_c = sp_cpu.x[ord_c], sp_cpu.y[ord_c], sp_cpu.z[ord_c]
    ux_c, uy_c, uz_c = sp_cpu.ux[ord_c], sp_cpu.uy[ord_c], sp_cpu.uz[ord_c]
    x_g = cp.asnumpy(sp_gpu.x)[ord_g]
    y_g = cp.asnumpy(sp_gpu.y)[ord_g]
    z_g = cp.asnumpy(sp_gpu.z)[ord_g]
    ux_g = cp.asnumpy(sp_gpu.ux)[ord_g]
    uy_g = cp.asnumpy(sp_gpu.uy)[ord_g]
    uz_g = cp.asnumpy(sp_gpu.uz)[ord_g]

    # Self-fields must be strong enough that particle x and u differences are non-zero
    assert float(max(np.max(np.abs(x_g - x_c)), np.max(np.abs(y_g - y_c)), np.max(np.abs(z_g - z_c)))) > 0.0
    assert float(max(np.max(np.abs(ux_g - ux_c)), np.max(np.abs(uy_g - uy_c)), np.max(np.abs(uz_g - uz_c)))) > 0.0

    # Empirical domain-length scaling factors Lx = Ly = 0.08 m, Lz = 0.10 m
    assert np.max(np.abs(x_g - x_c)) <= empirical_linear_budget_rel * 0.08
    assert np.max(np.abs(y_g - y_c)) <= empirical_linear_budget_rel * 0.08
    assert np.max(np.abs(z_g - z_c)) <= empirical_linear_budget_rel * 0.10
    u_scale = float(np.max(np.sqrt(ux_c**2 + uy_c**2 + uz_c**2)))
    assert np.max(np.abs(ux_g - ux_c)) <= empirical_linear_budget_rel * u_scale
    assert np.max(np.abs(uy_g - uy_c)) <= empirical_linear_budget_rel * u_scale
    assert np.max(np.abs(uz_g - uz_c)) <= empirical_linear_budget_rel * u_scale


@pytest.mark.parametrize("graded", [False, True])
def test_gpu_gauss_residual_1000_steps(graded):
    """Gauss residual on GPU must stay <= 1e-12 over >= 1000 steps (uniform and graded)."""
    Nx, Ny, Nz = 6, 6, 6
    grid = _make_grid(graded=graded, Nx=Nx, Ny=Ny, Nz=Nz)
    solver = SolverFIT3D(
        grid,
        cfln=0.45,
        bc_low=["periodic", "periodic", "periodic"],
        bc_high=["periodic", "periodic", "periodic"],
        use_gpu=True,
        verbose=0,
    )
    solver._fused_step = FusedStep(solver)
    solver.one_step = solver._fused_step

    rng = np.random.default_rng(1000 + int(graded))
    n_p = 48
    sp = Species(use_gpu=True)
    sp.add_particles(
        x=rng.uniform(0.005, 0.075, size=n_p),
        y=rng.uniform(0.005, 0.075, size=n_p),
        z=rng.uniform(0.005, 0.095, size=n_p),
        ux=rng.normal(0.0, 0.8 * c_light, size=n_p),
        uy=rng.normal(0.0, 0.8 * c_light, size=n_p),
        uz=rng.normal(0.5 * c_light, 0.8 * c_light, size=n_p),
        w=rng.uniform(1e6, 2e6, size=n_p),
    )
    ps = ParticleSolver(
        solver,
        species=[sp],
        pusher="boris",
        shape=2,
        gather="galerkin",
        sort_every=100,
    ).attach()

    n_steps = 1000
    max_gauss = 0.0
    for step in range(1, n_steps + 1):
        solver.one_step()
        if step in (1, 250, 500, 750, 1000):
            max_gauss = max(max_gauss, ps.gauss_residual())

    assert ps.n_particle_steps == n_steps
    assert max_gauss <= 1e-12


@pytest.mark.parametrize("shape", [1, 2])
def test_gpu_vs_cpu_coupled_v3_cold_plasma(shape):
    """Coupled V3 cold plasma oscillation: GPU = CPU to round-off for p = 1 and p = 2."""
    Nx, Ny, Nz = 2, 2, 16
    Lx, Ly, Lz = 0.02, 0.02, 0.16
    dz = Lz / Nz
    k_mode = 2.0 * np.pi / Lz
    n0 = 2.0e16
    omega_p0 = float(np.sqrt(n0 * e_charge**2 / (epsilon_0 * m_e)))
    n_ppc = 4
    n_p = Nx * Ny * Nz * n_ppc
    V_domain = Lx * Ly * Lz
    w_macro = (n0 * V_domain) / n_p
    delta_z = 1.0e-5 * dz
    n_steps = 80

    def _run_v3(on_gpu: bool):
        grid = GridFIT3D(
            xmin=0.0,
            xmax=Lx,
            ymin=0.0,
            ymax=Ly,
            zmin=0.0,
            zmax=Lz,
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            verbose=0,
        )
        solver = SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["periodic", "periodic", "periodic"],
            bc_high=["periodic", "periodic", "periodic"],
            use_gpu=on_gpu,
            verbose=0,
        )
        if on_gpu:
            solver._fused_step = FusedStep(solver)
            solver.one_step = solver._fused_step

        ix, iy, iz, ip = np.meshgrid(
            np.arange(Nx),
            np.arange(Ny),
            np.arange(Nz),
            np.arange(n_ppc),
            indexing="ij",
        )
        x0 = ((ix.ravel() + 0.5) * (Lx / Nx)).astype(np.float64)
        y0 = ((iy.ravel() + 0.5) * (Ly / Ny)).astype(np.float64)
        z0 = ((iz.ravel() + (ip.ravel() + 0.5) / n_ppc) * dz).astype(np.float64)
        uz0 = (1.0e-4 * c_light * np.sin(k_mode * z0)).astype(np.float64)

        sp = Species(use_gpu=on_gpu)
        sp.add_particles(
            x=x0,
            y=y0,
            z=z0,
            ux=np.zeros(n_p),
            uy=np.zeros(n_p),
            uz=uz0,
            w=np.full(n_p, w_macro),
        )
        ps = ParticleSolver(
            solver,
            species=[sp],
            pusher="boris",
            shape=shape,
            gather="galerkin",
            sort_every=25,
        ).attach()

        ez_mode_hist = []
        z_nodes = np.arange(Nz) * dz + 0.5 * dz
        sin_basis = np.sin(k_mode * z_nodes)
        for _ in range(n_steps):
            solver.one_step()
            Ez_3d = solver.E.to_matrix("z")
            if on_gpu:
                Ez_3d = cp.asnumpy(Ez_3d)
            ez_prof = np.mean(Ez_3d, axis=(0, 1))
            ez_mode_hist.append(float(np.dot(ez_prof, sin_basis) / (0.5 * Nz)))

        return solver, sp, ps, np.array(ez_mode_hist)

    s_cpu, sp_cpu, ps_cpu, hist_cpu = _run_v3(on_gpu=False)
    s_gpu, sp_gpu, ps_gpu, hist_gpu = _run_v3(on_gpu=True)

    assert ps_gpu.gauss_residual() <= 1e-12
    omega_disp = cold_plasma_dispersion_discrete(
        omega_p0=omega_p0,
        dt=s_cpu.dt,
        dz=dz,
        k=k_mode,
        order=shape,
        gather="galerkin",
    )
    assert omega_disp > 0.0

    # Form empirical linear-growth round-off budget n_steps * single_step_rel_v3
    # from the single-step 2 * gamma_{n(e)-1} per-edge deposit bound
    assert float(np.max(np.abs(hist_cpu))) > 1.0
    xi1_v3, eta1_v3, zeta1_v3 = ps_cpu.grid_map.to_logical(
        sp_cpu.x, sp_cpu.y, sp_cpu.z
    )
    xi0_v3 = xi1_v3 - float(s_cpu.dt) * sp_cpu.ux / (Lx / Nx)
    eta0_v3 = eta1_v3 - float(s_cpu.dt) * sp_cpu.uy / (Ly / Ny)
    zeta0_v3 = zeta1_v3 - float(s_cpu.dt) * sp_cpu.uz / dz
    (_, _, abs_Wz_v3), (_, _, cnt_Wz_v3) = esirkepov_edge_abs_sum_and_counts(
        xi0_v3,
        eta0_v3,
        zeta0_v3,
        xi1_v3,
        eta1_v3,
        zeta1_v3,
        sp_cpu.charge * sp_cpu.w,
        Nx,
        Ny,
        Nz,
        order=shape,
        periodic=(True, True, True),
    )
    single_step_rel_v3 = float(
        np.max(2.0 * _higham_gamma(cnt_Wz_v3 - 1) * abs_Wz_v3) / np.max(abs_Wz_v3)
    )
    empirical_linear_budget_rel = float(n_steps) * single_step_rel_v3
    assert np.max(np.abs(hist_gpu - hist_cpu)) <= empirical_linear_budget_rel * np.max(
        np.abs(hist_cpu)
    )

    E_cpu = s_cpu.E.toarray()
    E_gpu = cp.asnumpy(s_gpu.E.toarray())
    assert np.max(np.abs(E_gpu - E_cpu)) <= empirical_linear_budget_rel * np.max(
        np.abs(E_cpu)
    )

    ord_c = np.argsort(sp_cpu.id)
    ord_g = np.argsort(cp.asnumpy(sp_gpu.id))
    uz_c = sp_cpu.uz[ord_c]
    uz_g = cp.asnumpy(sp_gpu.uz)[ord_g]
    assert np.max(np.abs(uz_g - uz_c)) <= empirical_linear_budget_rel * np.max(
        np.abs(uz_c)
    )


def test_gpu_absorption_and_injection_bookkeeping():
    """Verify GPU boundary/collector absorption and BeamInjector match CPU and conserve Gauss."""
    L = 0.08
    Nx = Ny = Nz = 8

    def _run(on_gpu: bool):
        grid = GridFIT3D(
            xmin=-0.04,
            xmax=0.04,
            ymin=-0.04,
            ymax=0.04,
            zmin=0.0,
            zmax=L,
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            verbose=0,
        )
        solver = SolverFIT3D(
            grid,
            cfln=0.45,
            bc_low=["pec", "pec", "pec"],
            bc_high=["pec", "pec", "pec"],
            use_gpu=on_gpu,
            verbose=0,
        )
        sp = Species(name="electron", charge=-e_charge, mass=m_e, use_gpu=on_gpu)
        injector = BeamInjector(
            species=sp,
            z0=0.0,
            current=0.5,
            gamma=2.0,
            profile="uniform",
            radius=0.015,
            n_macro_per_step=8,
            rise_time=10 * solver.dt,
            thermal_u=0.08 * c_light,
            seed=1234,
        )
        collector = Collector(name="anode_collector", plane=("z", 0.055, 1))
        ps = ParticleSolver(
            solver,
            species=[sp],
            injectors=[injector],
            collectors=[collector],
            pusher="boris",
            shape=1,
            gather="galerkin",
        ).attach()

        for _ in range(60):
            solver.one_step()
        return solver, sp, ps, collector

    _, sp_cpu, ps_cpu, col_cpu = _run(on_gpu=False)
    _, sp_gpu, ps_gpu, col_gpu = _run(on_gpu=True)

    assert ps_gpu.gauss_residual() <= 1e-12
    assert sp_gpu.n_particles == sp_cpu.n_particles
    assert col_gpu.absorbed_count == col_cpu.absorbed_count
    assert abs(col_gpu.absorbed_charge - col_cpu.absorbed_charge) / abs(
        col_cpu.absorbed_charge
    ) <= 1e-12
    q_inj = ps_gpu.injected_charge()
    assert (
        abs(q_inj - (ps_gpu.total_charge() + ps_gpu.absorbed_charge())) / abs(q_inj)
        <= 1e-14
    )


# ============================================================================
# 4. GPU kernel code-path mutation checks
# ============================================================================


def test_gpu_kernel_mutation_drop_esirkepov_third_term_fails(monkeypatch):
    """Monkeypatching ``gpu_kernels._SOURCE`` to drop the 1/3 dS dS Esirkepov term must fail GPU=CPU and Gauss."""
    orig_src = gpu_kernels._SOURCE
    mut_src = orig_src.replace(
        "const double esirkepov_dd_coeff = 1.0 / 3.0;",
        "const double esirkepov_dd_coeff = 0.0;",
    )
    assert mut_src != orig_src

    with monkeypatch.context() as m:
        m.setattr(gpu_kernels, "_SOURCE", mut_src)

        grid = _make_grid(graded=False, Nx=6, Ny=6, Nz=6)
        solver = SolverFIT3D(
            grid,
            cfln=0.45,
            bc_low=["periodic", "periodic", "periodic"],
            bc_high=["periodic", "periodic", "periodic"],
            use_gpu=True,
            verbose=0,
        )
        rng = np.random.default_rng(999)
        n_p = 32
        x0 = rng.uniform(0.01, 0.07, size=n_p)
        y0 = rng.uniform(0.01, 0.07, size=n_p)
        z0 = rng.uniform(0.01, 0.09, size=n_p)
        ux0 = rng.normal(0.5 * c_light, 0.5 * c_light, size=n_p)
        uy0 = rng.normal(0.5 * c_light, 0.5 * c_light, size=n_p)
        uz0 = rng.normal(0.5 * c_light, 0.5 * c_light, size=n_p)
        w0 = rng.uniform(1e6, 2e6, size=n_p)

        sp = Species(use_gpu=True)
        sp.add_particles(x0, y0, z0, ux0, uy0, uz0, w0)
        ps = ParticleSolver(solver, species=[sp], shape=1).attach()

        for _ in range(15):
            solver.one_step()

        # 1. Gauss's law residual must fail (>> 1e-12) when the 1/3 dS dS term is dropped
        assert ps.gauss_residual() > 1e-4

        # 2. GPU = CPU deposit equivalence must also fail
        Wx_c, _, _ = deposit_current_esirkepov(
            x0 * 60.0,
            y0 * 60.0,
            z0 * 60.0,
            x0 * 60.0 + 0.25,
            y0 * 60.0 - 0.20,
            z0 * 60.0 + 0.15,
            -e_charge * w0,
            6,
            6,
            6,
            order=1,
            xp=np,
        )
        Wx_g, _, _ = deposit_current_esirkepov(
            cp.asarray(x0 * 60.0),
            cp.asarray(y0 * 60.0),
            cp.asarray(z0 * 60.0),
            cp.asarray(x0 * 60.0 + 0.25),
            cp.asarray(y0 * 60.0 - 0.20),
            cp.asarray(z0 * 60.0 + 0.15),
            cp.asarray(-e_charge * w0),
            6,
            6,
            6,
            order=1,
            xp=cp,
        )
        rel_err = float(
            np.max(np.abs(cp.asnumpy(Wx_g) - Wx_c)) / np.max(np.abs(Wx_c))
        )
        assert rel_err > 1e-3


def test_gpu_kernel_mutation_swap_gather_staggering_fails(monkeypatch):
    """Monkeypatching ``gpu_kernels._SOURCE`` to swap primal/staggered gather flags must fail GPU=CPU."""
    orig_src = gpu_kernels._SOURCE
    mut_src = orig_src.replace(
        "const int stag_flag = 1;\n    const int nod_flag = 0;",
        "const int stag_flag = 0;\n    const int nod_flag = 1;",
    )
    assert mut_src != orig_src

    with monkeypatch.context() as m:
        m.setattr(gpu_kernels, "_SOURCE", mut_src)
        with pytest.raises(AssertionError):
            test_gpu_vs_cpu_fused_gather_and_push_kernel(
                pusher="boris", gather="galerkin", order=1, graded=False
            )


# ============================================================================
# 5. Asynchronous execution (no host sync) and memory-pool allocation checks
# ============================================================================

_SPIN_KERNEL = cp.RawKernel(
    r"""
extern "C" __global__ void _device_spin_ns(long long duration_ns, int* out) {
    long long start = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    long long now = start;
    while (now - start < duration_ns) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    }
    if (out != nullptr) {
        out[0] = (int)(now - start);
    }
}
""",
    "_device_spin_ns",
)


class _AllocCounterHook(cp.cuda.MemoryHook):
    """CuPy MemoryHook counting both pool-cache sub-allocations (`malloc`) and new device allocations (`alloc`)."""

    def __init__(self) -> None:
        self.malloc_calls = 0
        self.alloc_calls = 0

    def malloc_preprocess(self, **kwargs) -> None:
        self.malloc_calls += 1

    def alloc_preprocess(self, **kwargs) -> None:
        self.alloc_calls += 1


def test_gpu_step_performs_no_device_to_host_transfer_or_allocations(monkeypatch):
    """Verify ``solver.one_step()`` never synchronises the host stream and performs zero pool sub-allocations on non-sort steps.

    Detection method
    ----------------
    1. Launch a ~60 ms device spin kernel (``_device_spin_ns``) on the current
       CUDA stream and record a ``cp.cuda.Event`` immediately after it.
    2. Call ``solver.one_step()`` on the host and assert ``not ev.done`` upon
       return — proving ``one_step()`` enqueues all GPU work asynchronously and
       never blocks on device-to-host copies (``.get()``, ``float(...)``,
       ``int(...)``, ``bool(...)``, or synchronous ``cudaMemcpy``).
    3. Use ``cp.cuda.MemoryHook`` to verify that non-sorting steps perform
       ``malloc_calls == 0`` and ``alloc_calls == 0`` (zero pool sub-allocations),
       while sorting steps (which call ``cp.argsort``) reuse cached pool blocks
       with ``alloc_calls == 0`` (no pool growth).
    4. Mutation check: monkeypatch ``ParticleSolver._mid_step_hook_gpu`` to read
       ``float(sp._x[0])`` during the step and assert that the ``not ev.done``
       check fails.
    """
    grid = GridFIT3D(
        xmin=0.0,
        xmax=0.06,
        ymin=0.0,
        ymax=0.06,
        zmin=0.0,
        zmax=0.08,
        Nx=6,
        Ny=6,
        Nz=8,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        cfln=0.45,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_gpu=True,
        verbose=0,
    )
    solver._fused_step = FusedStep(solver)
    solver.one_step = solver._fused_step

    rng = np.random.default_rng(12345)
    n_p = 128
    sp = Species(use_gpu=True)
    sp.add_particles(
        x=rng.uniform(0.01, 0.05, size=n_p),
        y=rng.uniform(0.01, 0.05, size=n_p),
        z=rng.uniform(0.01, 0.075, size=n_p),
        ux=rng.normal(0.0, 0.3 * c_light, size=n_p),
        uy=rng.normal(0.0, 0.3 * c_light, size=n_p),
        uz=rng.normal(0.5 * c_light, 0.3 * c_light, size=n_p),
        w=rng.uniform(1e5, 2e5, size=n_p),
    )
    collector = Collector(name="top_col", plane=("z", 0.072, 1))
    ps = ParticleSolver(
        solver,
        species=[sp],
        collectors=[collector],
        pusher="boris",
        shape=2,
        gather="galerkin",
        sort_every=5,
    ).attach()

    dummy_spin = cp.zeros(1, dtype=cp.int32)
    _SPIN_KERNEL((1,), (1,), (np.int64(1000), dummy_spin))
    # Warm up step 0 (builds FusedStep CSR cache) and one sort step (step 5)
    for _ in range(6):
        solver.one_step()
    cp.cuda.Stream.null.synchronize()

    pool = cp.get_default_memory_pool()
    total_bytes_before = pool.total_bytes()

    def _assert_step_does_not_synchronize(n_steps_to_enqueue: int = 1) -> None:
        ev_spin = cp.cuda.Event()
        _SPIN_KERNEL((1,), (1,), (np.int64(60_000_000), dummy_spin))
        ev_spin.record()
        for _ in range(n_steps_to_enqueue):
            solver.one_step()
        # Host must return while the 60 ms spin kernel is still running on the GPU
        assert not ev_spin.done, (
            "solver.one_step() synchronized the CUDA stream (device-to-host transfer or sync detected)!"
        )
        cp.cuda.Stream.null.synchronize()
        assert ev_spin.done

    # 1. Verify non-sorting steps (steps 6, 7, 8, 9) perform zero pool sub-allocations
    #    and zero host synchronizations (including boundary ray-casting & collector compaction)
    with _AllocCounterHook() as hook_nonsort:
        _assert_step_does_not_synchronize(n_steps_to_enqueue=4)
    assert hook_nonsort.malloc_calls == 0
    assert hook_nonsort.alloc_calls == 0

    # 2. Verify sorting step (step 10, where cp.argsort runs) also never synchronizes
    #    the host stream and reuses cached pool blocks without growing the pool
    with _AllocCounterHook() as hook_sort:
        _assert_step_does_not_synchronize(n_steps_to_enqueue=1)
    assert hook_sort.alloc_calls == 0
    assert hook_sort.malloc_calls > 0
    assert pool.total_bytes() == total_bytes_before
    assert ps.n_particle_steps == 11

    # 3. Code-path mutation check: inserting float(sp._x[0]) inside _mid_step_hook_gpu
    #    forces a device-to-host scalar copy that synchronizes the stream and MUST fail
    orig_hook = ParticleSolver._mid_step_hook_gpu

    def _mutated_hook_with_dtoh(self_ps, s_arg):
        _ = float(self_ps.species[0]._x[0])
        return orig_hook(self_ps, s_arg)

    with monkeypatch.context() as m:
        m.setattr(ParticleSolver, "_mid_step_hook_gpu", _mutated_hook_with_dtoh)
        with pytest.raises(AssertionError, match="synchronized the CUDA stream"):
            _assert_step_does_not_synchronize(n_steps_to_enqueue=1)


def test_gpu_esirkepov_cfl_guard_and_attach_dt_check(monkeypatch):
    """Verify host-side ``c * dt <= min_cell_size`` check at ``attach()`` and device ``_d_cfl_error`` flag."""
    grid = _make_grid(graded=False, Nx=6, Ny=6, Nz=6)
    solver = SolverFIT3D(
        grid,
        cfln=0.45,
        bc_low=["periodic", "periodic", "periodic"],
        bc_high=["periodic", "periodic", "periodic"],
        use_gpu=True,
        verbose=0,
    )
    sp = Species(use_gpu=True)
    sp.add_particles(x=0.03, y=0.03, z=0.03, ux=0.0, uy=0.0, uz=0.1 * c_light, w=1.0e5)

    # 1. Host-side dt check at attach(): setting c * dt > min_cell_size must raise ValueError
    dx_min = float(min(np.min(grid.dx), np.min(grid.dy), np.min(grid.dz)))
    orig_dt = solver.dt
    solver.dt = 1.25 * dx_min / c_light
    with pytest.raises(ValueError, match="c \\* dt <= min_cell_size"):
        ParticleSolver(solver, species=[sp], shape=1).attach()

    # Restore valid dt and attach
    solver.dt = orig_dt
    ps = ParticleSolver(solver, species=[sp], shape=1).attach()
    assert int(ps._d_cfl_error[0].get()) == 0

    # 1b. Host-side dt re-check at the start of one_step()/step() when dt changes after attach()
    solver.dt = 1.25 * dx_min / c_light
    H_before = solver.H.toarray().copy()
    with pytest.raises(ValueError, match="c \\* dt <= min_cell_size"):
        solver.one_step()
    # the check runs before the H update: a rejected step must leave H untouched
    assert bool(cp.all(solver.H.toarray() == H_before))
    with pytest.raises(ValueError, match="c \\* dt <= min_cell_size"):
        ps._mid_step_hook()
    solver.dt = orig_dt
    solver.one_step()
    assert ps.gauss_residual() <= 1e-12

    # 2. Device CFL error flag: passing a displacement |b1 - b0| > 1 to
    #    deposit_current_esirkepov_gpu sets _d_cfl_error and causes gauss_residual() to raise
    xi0 = cp.array([1.2], dtype=np.float64)
    eta0 = cp.array([1.2], dtype=np.float64)
    zeta0 = cp.array([1.2], dtype=np.float64)
    xi1_bad = cp.array([3.7], dtype=np.float64)  # |b1 - b0| = 2 > 1
    w_arr = cp.array([1.0e5], dtype=np.float64)
    gpu_kernels.deposit_current_esirkepov_gpu(
        xi0,
        eta0,
        zeta0,
        xi1_bad,
        eta0,
        zeta0,
        w_arr,
        -e_charge,
        6,
        6,
        6,
        order=1,
        periodic=(True, True, True),
        Wx_out=ps._Wx_buf,
        Wy_out=ps._Wy_buf,
        Wz_out=ps._Wz_buf,
        d_cfl_error=ps._d_cfl_error,
    )
    assert int(ps._d_cfl_error[0].get()) > 0
    with pytest.raises(ValueError, match="crossed more than 1 cell"):
        ps.gauss_residual()

    # 3. Resetting _d_cfl_error via ps.reset_cfl_error() clears the device counter
    ps.reset_cfl_error()
    assert int(ps._d_cfl_error[0].get()) == 0
    assert ps.gauss_residual() <= 1e-12


def test_usersguide_particles_gpu_snippet_executes():
    """Execute the ``use_gpu=True`` snippet from ``docs/usersguide.md`` on the GPU."""
    from pathlib import Path
    import re

    ug_path = Path(__file__).resolve().parents[1] / "docs" / "usersguide.md"
    text = ug_path.read_text(encoding="utf-8")
    sec_match = re.search(
        r"### GPU-Accelerated Execution \(`use_gpu=True`\)(.*?)```python\n(.*?)```",
        text,
        flags=re.DOTALL,
    )
    assert sec_match is not None, (
        "Could not find GPU-Accelerated Execution (`use_gpu=True`) snippet in docs/usersguide.md"
    )
    code = sec_match.group(2)
    assert "use_gpu=True" in code
    ns = {}
    exec(compile(code, str(ug_path), "exec"), ns)
    res = float(
        ns["gauss_residual"](ns["solver_gpu"], [ns["electrons_gpu"]])
    )
    assert res < 1e-12
    assert ns["electrons_gpu"].n_particles == 1024

