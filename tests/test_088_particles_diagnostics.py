# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Validation suite for P8 GPU-resident, MPI-aware particle diagnostics (`wakis.particles`).

Covers requirements (a)-(f):
(a) PhaseSpaceMonitor and EnergySpectrumMonitor match numpy.histogram2d /
    numpy.histogram (exact counts, weighted sums within Higham round-off bound,
    including edge-bin and out-of-range overflow cases).
(b) PlaneCurrentProbe charge conservation in an injector + collector beam run
    with accelerating field, plus wrong-time-level mutation check
    (z_old = z - v_z^{n-1/2} * dt).
(c) EmittanceProbe on correlated bivariate Gaussian distributions with known
    (sigma_x, sigma_xp, rho) matching analytic RMS and normalized emittance
    within the derived Wishart standard error eps / sqrt(N), plus mutation check
    dropping the <x x'>^2 covariance term.
(d) TrackRecorder for 3 gyrating particles in uniform B matching the analytic
    V1 relativistic gyration trajectory, plus absorbed-particle NaN handling.
(e) GPU no-sync and zero-allocation verification for every monitor using the
    60 ms device spin kernel and CuPy MemoryHook, plus GPU == CPU parity.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import c as c_light
from scipy.constants import e as e_charge
from scipy.constants import m_e

from wakis import GridFIT3D, SolverFIT3D
from wakis.particles import (
    BeamInjector,
    Collector,
    EmittanceProbe,
    EnergySpectrumMonitor,
    ParticleSolver,
    PhaseSpaceMonitor,
    PlaneCurrentProbe,
    Species,
    TrackRecorder,
)

try:
    import cupy as cp

    HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # noqa: BLE001 - import or CUDA runtime/driver errors both mean no GPU
    cp = None
    HAS_GPU = False

try:
    from mpi4py import MPI

    HAS_MPI_MULTI = MPI.COMM_WORLD.Get_size() >= 2
except Exception:  # noqa: BLE001 - import or MPI init errors both mean no MPI
    MPI = None
    HAS_MPI_MULTI = False

if HAS_GPU:
    from wakis.gpu_fused import FusedStep

    _SPIN_KERNEL = cp.RawKernel(
        r"""
        extern "C" __global__ void _device_spin_ns(long long min_ns, int* out) {
            long long start = clock64();
            long long now = start;
            while ((now - start) < min_ns) {
                now = clock64();
            }
            if (out != nullptr && threadIdx.x == 0) {
                out[0] = (int)(now - start);
            }
        }
        """,
        "_device_spin_ns",
    )

    class _AllocCounterHook(cp.cuda.MemoryHook):
        """Count CuPy pool sub-allocations (malloc) and driver allocations (alloc)."""

        def __init__(self) -> None:
            super().__init__()
            self.malloc_calls = 0
            self.alloc_calls = 0

        def malloc_preprocess(self, **kwargs) -> None:
            self.malloc_calls += 1

        def alloc_preprocess(self, **kwargs) -> None:
            self.alloc_calls += 1


def _higham_gamma(n: int) -> float:
    """Higham (2002) Theorem 4.2 summation round-off factor n*u / (1 - n*u)."""
    u = 0.5 * float(np.finfo(np.float64).eps)
    nu = max(int(n), 1) * u
    return nu / max(1.0 - nu, 0.5)


# ===========================================================================
# (a) Histogram accuracy vs numpy.histogram2d and numpy.histogram
# ===========================================================================


@pytest.mark.parametrize("use_gpu", [False, True])
def test_histograms_match_numpy_with_edge_and_overflow_cases(
    use_gpu: bool, monkeypatch
):
    """Verify PhaseSpaceMonitor and EnergySpectrumMonitor match NumPy histograms.

    Checks:
    1. Unweighted counts (weighted=False) agree with numpy.histogram2d and
       numpy.histogram with zero error (exact integer match).
    2. Weighted sums (weighted=True) agree with numpy.histogram2d and
       numpy.histogram within Higham (2002) Theorem 4.2 summation bound
       gamma_{m_b} * H_b for m_b particles in bin b.
    3. Particles lying directly on the lower outer edge, interior bin boundaries,
       and the upper outer edge (closed last bin) are binned identically to
       NumPy, while out-of-range particles are accumulated in overflow.
    4. Ring-buffer mode (ring_size=3) retains the last 3 frames chronologically.
    5. Mutation check: corrupting the upper-edge inclusive rule causes the test
       to fail.
    """
    if use_gpu and not HAS_GPU:
        pytest.skip("CuPy / CUDA GPU not available")

    rng = np.random.default_rng(2026088)
    nb_z, nb_uz = 16, 12
    z_range = (0.0, 0.08)
    uz_range = (-0.2 * c_light, 0.6 * c_light)
    z_edges = np.linspace(z_range[0], z_range[1], nb_z + 1)
    uz_edges = np.linspace(uz_range[0], uz_range[1], nb_uz + 1)

    # Interior random samples + explicit boundary/edge samples + out-of-range samples
    n_rand = 500
    z_rand = rng.uniform(0.001, 0.079, size=n_rand)
    uz_rand = rng.uniform(-0.18 * c_light, 0.58 * c_light, size=n_rand)

    # Exact edge values: lower edge, interior edges, upper edge (closed last bin)
    z_edge_pts = np.array(
        [
            z_edges[0],
            z_edges[3],
            z_edges[8],
            z_edges[-1],
            z_edges[0],
            z_edges[-1],
        ]
    )
    uz_edge_pts = np.array(
        [
            uz_edges[0],
            uz_edges[4],
            uz_edges[7],
            uz_edges[-1],
            uz_edges[-1],
            uz_edges[0],
        ]
    )

    # Out-of-range overflow points (below min, above max on z, uz, or both)
    z_ovf_pts = np.array([-0.01, 0.09, 0.04, 0.04, -0.02])
    uz_ovf_pts = np.array(
        [0.1 * c_light, 0.1 * c_light, -0.3 * c_light, 0.8 * c_light, 0.9 * c_light]
    )

    z_all = np.concatenate([z_rand, z_edge_pts, z_ovf_pts])
    uz_all = np.concatenate([uz_rand, uz_edge_pts, uz_ovf_pts])
    n_tot = z_all.size
    x_all = rng.uniform(0.01, 0.05, size=n_tot)
    y_all = rng.uniform(0.01, 0.05, size=n_tot)
    ux_all = rng.normal(0.0, 0.05 * c_light, size=n_tot)
    uy_all = rng.normal(0.0, 0.05 * c_light, size=n_tot)
    w_all = rng.uniform(1.0e4, 5.0e5, size=n_tot)

    sp = Species(name="electrons", charge=-e_charge, mass=m_e, use_gpu=use_gpu)
    sp.add_particles(
        x=x_all, y=y_all, z=z_all, ux=ux_all, uy=uy_all, uz=uz_all, w=w_all
    )

    # Reference 2D histograms via numpy.histogram2d
    ref_counts_2d, _, _ = np.histogram2d(
        z_all, uz_all, bins=(nb_z, nb_uz), range=(z_range, uz_range)
    )
    ref_weights_2d, _, _ = np.histogram2d(
        z_all,
        uz_all,
        bins=(nb_z, nb_uz),
        range=(z_range, uz_range),
        weights=w_all,
    )
    in_2d = (
        (z_all >= z_range[0])
        & (z_all <= z_range[1])
        & (uz_all >= uz_range[0])
        & (uz_all <= uz_range[1])
    )
    ref_ovf_cnt_2d = int(np.sum(~in_2d))
    ref_ovf_w_2d = float(np.sum(w_all[~in_2d]))

    mon_2d_cnt = PhaseSpaceMonitor(
        axes=("z", "uz"),
        bins=(nb_z, nb_uz),
        ranges=(z_range, uz_range),
        weighted=False,
    )
    mon_2d_w = PhaseSpaceMonitor(
        axes=("z", "uz"),
        bins=(nb_z, nb_uz),
        ranges=(z_range, uz_range),
        weighted=True,
        ring_size=3,
    )
    mon_2d_cnt.sample(sp, t=1.0e-11, step=0)
    mon_2d_w.sample(sp, t=1.0e-11, step=0)

    res_2d_cnt = mon_2d_cnt.result()
    res_2d_w = mon_2d_w.result()

    # Exact integer equality for unweighted counts
    assert np.array_equal(res_2d_cnt["hist"], ref_counts_2d)
    assert res_2d_cnt["overflow_count"] == ref_ovf_cnt_2d
    assert res_2d_cnt["overflow"] == float(ref_ovf_cnt_2d)

    # Weighted 2D histogram within Higham Theorem 4.2 bound per bin
    # Each bin b has m_b = ref_counts_2d[b] positive summands
    bin_gamma_2d = np.vectorize(_higham_gamma)(
        np.maximum(ref_counts_2d.astype(int), 1)
    )
    bound_2d = 2.0 * bin_gamma_2d * np.maximum(ref_weights_2d, 1.0)
    assert np.all(np.abs(res_2d_w["hist"] - ref_weights_2d) <= bound_2d)
    assert res_2d_w["overflow_count"] == ref_ovf_cnt_2d
    assert abs(res_2d_w["overflow"] - ref_ovf_w_2d) <= (
        2.0 * _higham_gamma(ref_ovf_cnt_2d) * ref_ovf_w_2d
    )

    # Now test EnergySpectrumMonitor (1D) including exact edge and overflow energies
    u2_all = ux_all**2 + uy_all**2 + uz_all**2
    gamma_all = np.sqrt(1.0 + u2_all / (c_light**2))
    ek_eV_all = (m_e / e_charge) * (u2_all / (gamma_all + 1.0))

    nb_e = 20
    # Choose energy range so some particles fall on edges and some overflow above/below
    e_min = float(np.percentile(ek_eV_all, 5.0))
    e_max = float(np.percentile(ek_eV_all, 92.0))
    e_edges = np.linspace(e_min, e_max, nb_e + 1)

    # Construct exact kinetic energies at e_edges[0], e_edges[5], e_edges[-1]
    target_ek = np.array([e_edges[0], e_edges[5], e_edges[-1]])
    target_gamma = 1.0 + target_ek * (e_charge / m_e) / (c_light**2)
    target_uz = c_light * np.sqrt(np.maximum(target_gamma**2 - 1.0, 0.0))
    sp.add_particles(
        x=np.full(3, 0.03),
        y=np.full(3, 0.03),
        z=np.full(3, 0.04),
        ux=np.zeros(3),
        uy=np.zeros(3),
        uz=target_uz,
        w=np.array([1.2e5, 2.3e5, 3.4e5]),
    )
    w_ext = np.concatenate([w_all, np.array([1.2e5, 2.3e5, 3.4e5])])
    u2_ext = np.concatenate([u2_all, target_uz**2])
    gamma_ext = np.sqrt(1.0 + u2_ext / (c_light**2))
    ek_eV_ext = (m_e / e_charge) * (u2_ext / (gamma_ext + 1.0))

    ref_counts_1d, _ = np.histogram(ek_eV_ext, bins=nb_e, range=(e_min, e_max))
    ref_weights_1d, _ = np.histogram(
        ek_eV_ext, bins=nb_e, range=(e_min, e_max), weights=w_ext
    )
    in_1d = (ek_eV_ext >= e_min) & (ek_eV_ext <= e_max)
    ref_ovf_cnt_1d = int(np.sum(~in_1d))
    ref_ovf_w_1d = float(np.sum(w_ext[~in_1d]))

    mon_1d_cnt = EnergySpectrumMonitor(
        bins=nb_e, range_eV=(e_min, e_max), weighted=False
    )
    mon_1d_w = EnergySpectrumMonitor(
        bins=nb_e, range_eV=(e_min, e_max), weighted=True
    )
    mon_1d_cnt.sample(sp, t=1.0e-11, step=0)
    mon_1d_w.sample(sp, t=1.0e-11, step=0)

    res_1d_cnt = mon_1d_cnt.result()
    res_1d_w = mon_1d_w.result()

    assert np.array_equal(res_1d_cnt["hist"], ref_counts_1d)
    assert res_1d_cnt["overflow_count"] == ref_ovf_cnt_1d
    assert res_1d_cnt["overflow"] == float(ref_ovf_cnt_1d)

    bin_gamma_1d = np.vectorize(_higham_gamma)(
        np.maximum(ref_counts_1d.astype(int), 1)
    )
    bound_1d = 2.0 * bin_gamma_1d * np.maximum(ref_weights_1d, 1.0)
    assert np.all(np.abs(res_1d_w["hist"] - ref_weights_1d) <= bound_1d)
    assert res_1d_w["overflow_count"] == ref_ovf_cnt_1d
    assert abs(res_1d_w["overflow"] - ref_ovf_w_1d) <= (
        2.0 * _higham_gamma(ref_ovf_cnt_1d) * ref_ovf_w_1d
    )

    # Ring buffer chronological ordering over 4 additional samples (total 5 > ring_size=3)
    for st in range(1, 5):
        mon_2d_w.sample(sp, t=float((st + 1) * 1.0e-11), step=st)
    res_ring = mon_2d_w.result()
    assert res_ring["frames"].shape == (3, nb_z, nb_uz)
    assert np.array_equal(res_ring["steps"], np.array([2, 3, 4]))

    # Mutation check on the production binning path:
    # - On CPU: monkeypatch _find_bins_cpu so it clips overflow into the end bins
    #   and skips edge refinement.
    # - On GPU: perturb _d_edges0 on the bound monitor.
    # The numpy-equality assertion on hist must fail.
    mon_2d_mut = PhaseSpaceMonitor(
        axes=("z", "uz"),
        bins=(nb_z, nb_uz),
        ranges=(z_range, uz_range),
        weighted=False,
    )
    if use_gpu:
        mon_2d_mut._bind(sp)
        mon_2d_mut._d_edges0 = mon_2d_mut._d_edges0 + 0.02
    else:

        def _clipped_no_refine_bins_cpu(
            vals, first_edge, last_edge, n_bins, edges
        ):
            in_range = np.ones(vals.shape, dtype=bool)
            f_idx = ((vals - first_edge) / (last_edge - first_edge)) * n_bins
            idx = np.clip(f_idx.astype(np.intp), 0, n_bins - 1)
            return idx, in_range

        monkeypatch.setattr(
            "wakis.particles.diagnostics._find_bins_cpu",
            _clipped_no_refine_bins_cpu,
        )
    mon_2d_mut.sample(sp, t=1.0e-11, step=0)
    res_2d_mut = mon_2d_mut.result()
    with pytest.raises(AssertionError):
        assert np.array_equal(res_2d_mut["hist"], ref_counts_2d)


# ===========================================================================
# (b) PlaneCurrentProbe charge conservation & wrong-time-level mutation
# ===========================================================================


@pytest.mark.parametrize("use_gpu", [False, True])
def test_plane_current_probe_charge_conservation_and_mutation(
    use_gpu: bool, monkeypatch
):
    """Verify PlaneCurrentProbe conserves injected beam charge and fails under v_z^{n-1/2}.

    Physical setup:
    An electron beam pulse is injected at z0 = 0.012 m over the first 5 steps and
    accelerated in +z by a uniform external electric field E_ext = (0, 0, -3.5e7) V/m
    across the measurement plane z_plane toward a collector at z = 0.018 m.
    Every injected macroparticle transits across z_plane before the end of the run,
    so the time-integrated probe charge sum_n dQ_n must equal the total injected
    charge Q_inj up to IEEE-754 summation round-off (including when every=3 or when
    the sample count exceeds the ring-buffer capacity).

    Mutation check:
    Monkeypatching _compute_z_old_cpu (CPU) / _compute_z_old_vel_gpu (GPU) to use
    the wrong leapfrog time level v_z^{n-1/2} instead of v_z^{n+1/2} (while keeping
    the production mask, sum, and slot code under test) shifts the reconstructed
    start position by (v_z^{n+1/2} - v_z^{n-1/2}) * dt > 0, missing particles that
    cross within that acceleration layer and violating charge conservation.
    """
    if use_gpu and not HAS_GPU:
        pytest.skip("CuPy / CUDA GPU not available")

    def _run_beam_probe(
        use_wrong_time_level: bool = False,
        sampled_only_every3: bool = False,
        check_warnings: bool = False,
    ):
        grid = GridFIT3D(
            xmin=0.0,
            xmax=0.04,
            ymin=0.0,
            ymax=0.04,
            zmin=0.0,
            zmax=0.064,
            Nx=4,
            Ny=4,
            Nz=16,
            verbose=0,
        )
        solver = SolverFIT3D(
            grid,
            cfln=0.45,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            use_gpu=use_gpu,
            verbose=0,
        )
        if use_gpu:
            solver._fused_step = FusedStep(solver)
            solver.one_step = solver._fused_step

        sp = Species(
            name="electrons", charge=-e_charge, mass=m_e, use_gpu=use_gpu
        )
        # Choose initial velocity and accelerating field E_z so that on the crossing
        # step, z^n lies inside [z_plane - (v_z^{n+1/2} - v_z^{n-1/2})*dt, z_plane).
        dt = float(solver.dt)
        z0 = 0.012
        v0 = 0.25 * c_light
        g0 = 1.0 / np.sqrt(1.0 - (v0 / c_light) ** 2)
        uz0 = g0 * v0
        Ez_ext = -3.5e7  # accelerates electrons in +z

        # Analytically compute trajectory of a particle injected at z0 at step 0
        # under Boris push in uniform E_z to place z_plane inside the step-2
        # acceleration window [z^2, z^2 + (v_z^{5/2} - v_z^{3/2}) * dt).
        uz_half = uz0
        z_pos = z0
        v_prev = uz_half / np.sqrt(1.0 + (uz_half / c_light) ** 2)
        for step_k in range(3):
            uz_next = uz_half + (-e_charge / m_e) * Ez_ext * dt
            v_next = uz_next / np.sqrt(1.0 + (uz_next / c_light) ** 2)
            z_next = z_pos + v_next * dt
            if step_k == 2:
                # Midpoint of [z_pos, z_pos + (v_next - v_prev) * dt]
                z_plane_target = z_pos + 0.5 * (v_next - v_prev) * dt
            z_pos = z_next
            uz_half = uz_next
            v_prev = v_next

        inj = BeamInjector(
            species=sp,
            z0=z0,
            current=1.0e-9,
            gamma=g0,
            radius=0.006,
            x0=0.02,
            y0=0.02,
            n_macro_per_step=8,
            seed=901,
        )
        col = Collector(name="anode", plane=("z", 0.018, 1))
        ps = ParticleSolver(
            solver,
            species=[sp],
            injectors=[inj],
            collectors=[col],
            E_ext=(0.0, 0.0, Ez_ext),
            pusher="boris",
            shape=1,
        ).attach()

        if check_warnings:
            with pytest.warns(UserWarning, match="equals injector"):
                ps.add_diagnostic(PlaneCurrentProbe(z_plane=z0))
            with pytest.warns(UserWarning, match="collector"):
                ps.add_diagnostic(
                    PlaneCurrentProbe(z_plane=0.018 - 0.5 * c_light * dt)
                )
            ps.diagnostics.clear()
            with monkeypatch.context() as mp_blk:
                mp_blk.setattr("wakis.particles.diagnostics._BLOCK_SIZE", 200)
                with pytest.raises(AssertionError, match="power of two"):
                    from wakis.particles.diagnostics import _get_diag_kernel

                    _get_diag_kernel("plane_current_probe_kernel")

        probe = PlaneCurrentProbe(z_plane=z_plane_target, every=1)
        probe_every3 = PlaneCurrentProbe(z_plane=z_plane_target, every=3)
        probe_cap4 = PlaneCurrentProbe(
            z_plane=z_plane_target, every=1, capacity=4
        )
        ps.add_diagnostic(probe)
        ps.add_diagnostic(probe_every3)
        ps.add_diagnostic(probe_cap4)
        assert probe._placement_checked is True

        placement_calls = 0
        orig_check = probe._check_placement_warnings

        def _counted_check(dt_val):
            nonlocal placement_calls
            placement_calls += 1
            return orig_check(dt_val)

        monkeypatch.setattr(probe, "_check_placement_warnings", _counted_check)

        if sampled_only_every3:
            monkeypatch.setattr(probe_every3, "_always_record", False)

        if use_wrong_time_level:
            # Monkeypatch _compute_z_old_cpu (CPU) and _compute_z_old_vel_gpu (GPU)
            # to use the n - 1/2 velocity, keeping the production mask, sum, and
            # slot code under test.
            if use_gpu:
                def _wrong_z_old_vel_gpu(sp_arg: Species, dt_step: float):
                    uz_prev = sp_arg._uz - (-e_charge / m_e) * Ez_ext * dt_step
                    return sp_arg._ux, sp_arg._uy, uz_prev

                monkeypatch.setattr(
                    probe, "_compute_z_old_vel_gpu", _wrong_z_old_vel_gpu
                )
            else:
                def _wrong_z_old_cpu(sp_arg: Species, dt_step: float):
                    n_p = sp_arg.n_particles
                    ux_arr = sp_arg._ux[:n_p]
                    uy_arr = sp_arg._uy[:n_p]
                    uz_prev = (
                        sp_arg._uz[:n_p] - (-e_charge / m_e) * Ez_ext * dt_step
                    )
                    inv_g_prev = 1.0 / np.sqrt(
                        1.0
                        + (ux_arr * ux_arr + uy_arr * uy_arr + uz_prev * uz_prev)
                        / (c_light**2)
                    )
                    return sp_arg._z[:n_p], (dt_step * inv_g_prev) * uz_prev

                monkeypatch.setattr(
                    probe, "_compute_z_old_cpu", _wrong_z_old_cpu
                )

        # Inject for 5 steps, then turn off injector and drift/accelerate for 25 steps
        # so all injected particles cross z_plane and reach the collector at z = 0.018.
        for s_idx in range(30):
            if s_idx == 5:
                inj.current = 0.0
            solver.one_step()
        assert placement_calls == 0

        res = probe.result()
        res_every3 = probe_every3.result()
        res_cap4 = probe_cap4.result()
        q_inj = ps.injected_charge()
        q_col = ps.absorbed_charge("anode")
        return res, res_every3, res_cap4, q_inj, q_col

    res_exact, res_every3, res_cap4, q_inj, q_col = _run_beam_probe(
        use_wrong_time_level=False, check_warnings=True
    )
    assert abs(q_inj) > 0.0
    # Every injected particle crossed z_plane and was absorbed by the collector
    n_tot_macro = 5 * 8
    roundoff_bound = 4.0 * _higham_gamma(n_tot_macro) * abs(q_inj)
    assert abs(res_exact["Q_total"] - q_inj) <= roundoff_bound
    assert abs(res_every3["Q_total"] - res_exact["Q_total"]) <= roundoff_bound
    assert res_cap4["dQ"].size == 4
    assert abs(res_cap4["Q_total"] - q_inj) <= roundoff_bound
    assert abs(res_cap4["Q_cumulative"][-1] - q_inj) <= roundoff_bound
    assert abs(q_col - q_inj) <= roundoff_bound
    assert np.allclose(
        res_exact["I"] * res_exact["dt"], res_exact["dQ"], rtol=0.0, atol=1e-30
    )
    assert res_every3["dQ"].size == 10
    assert np.allclose(
        res_every3["dt"], 3.0 * res_exact["dt"][0], rtol=0.0, atol=1e-30
    )
    assert np.allclose(
        res_every3["I"] * res_every3["dt"],
        res_every3["dQ"],
        rtol=0.0,
        atol=1e-30,
    )

    # Mutation check (every=3 sampled-only detection): restoring sampled-only
    # detection drops crossings on steps where step % 3 != 0 and must fail.
    _, res_every3_mut, _, _, _ = _run_beam_probe(sampled_only_every3=True)
    with pytest.raises(AssertionError):
        assert (
            abs(res_every3_mut["Q_total"] - res_exact["Q_total"])
            <= roundoff_bound
        )

    # Mutation check: wrong time level v_z^{n-1/2} must fail the conservation identity
    res_mut, _, _, q_inj_mut, _ = _run_beam_probe(use_wrong_time_level=True)
    with pytest.raises(AssertionError):
        assert abs(res_mut["Q_total"] - q_inj_mut) <= roundoff_bound


@pytest.mark.parametrize("use_gpu", [False, True])
def test_plane_current_probe_periodic_wrap(use_gpu: bool):
    """Verify PlaneCurrentProbe counts particles wrapping across a periodic z boundary."""
    if use_gpu and not HAS_GPU:
        pytest.skip("CuPy / CUDA GPU not available")

    grid = GridFIT3D(
        xmin=0.0,
        xmax=0.04,
        ymin=0.0,
        ymax=0.04,
        zmin=0.0,
        zmax=0.08,
        Nx=4,
        Ny=4,
        Nz=8,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        cfln=0.45,
        bc_low=["periodic", "periodic", "periodic"],
        bc_high=["periodic", "periodic", "periodic"],
        use_gpu=use_gpu,
        verbose=0,
    )
    if use_gpu:
        solver._fused_step = FusedStep(solver)
        solver.one_step = solver._fused_step

    dt = float(solver.dt)
    vz = 0.4 * c_light
    gamma = 1.0 / np.sqrt(1.0 - (vz / c_light) ** 2)
    uz = gamma * vz
    dz = vz * dt

    # Particle 0 starts near zmax = 0.08 and wraps past zmin = 0.0 into z > 0.001
    # Particle 1 moves in -z starting near zmin = 0.0 and wraps into z < 0.079
    z_start = np.array([0.08 - 0.3 * dz, 0.0 + 0.3 * dz])
    uz_start = np.array([uz, -uz])
    w_start = np.array([2.0e5, 1.5e5])

    sp = Species(name="beam", charge=e_charge, mass=m_e, use_gpu=use_gpu)
    sp.add_particles(
        x=np.array([0.02, 0.02]),
        y=np.array([0.02, 0.02]),
        z=z_start,
        ux=np.zeros(2),
        uy=np.zeros(2),
        uz=uz_start,
        w=w_start,
    )
    ps = ParticleSolver(solver, species=[sp], pusher="boris", shape=1).attach()

    # Probe near z = 0.0001 m (crossed by both wrapping particles in opposite directions)
    probe = PlaneCurrentProbe(z_plane=0.1 * dz)
    ps.add_diagnostic(probe)
    solver.one_step()

    res = probe.result()
    expected_q = e_charge * (w_start[0] - w_start[1])
    assert res["counts"][0] == 0  # +1 and -1
    assert abs(res["Q_total"] - expected_q) <= 4.0 * _higham_gamma(2) * abs(expected_q)


# ===========================================================================
# (c) EmittanceProbe on correlated Gaussian & covariance mutation
# ===========================================================================


@pytest.mark.parametrize("use_gpu", [False, True])
def test_emittance_probe_correlated_gaussian_and_mutation(
    use_gpu: bool, monkeypatch
):
    """Verify EmittanceProbe on correlated bivariate Gaussians (uniform and varying u_z).

    Statistical tolerance derivation:
    For N weighted draws from a bivariate Gaussian (x, x') ~ N(mu, Sigma) with
    Sigma = [[sigma_x^2, rho*sigma_x*sigma_xp], [rho*sigma_x*sigma_xp, sigma_xp^2]],
    the exact population geometric RMS emittance is
        eps_x = sqrt(det(Sigma)) = sigma_x * sigma_xp * sqrt(1 - rho^2).
    By Wishart distribution theory (Anderson 2003, Theorem 7.5.3) with Kish
    effective sample size N_eff = (sum w_p)^2 / sum(w_p^2), the sample
    generalized variance det(S) has asymptotic relative standard error
    sqrt(2 / N_eff) for eps_x = sqrt(det(S)), i.e. SE(eps_x) = eps_x / sqrt(N_eff).
    We also include a large centroid offset mu_x = 10.0 m >> sigma_x = 1.2e-3 m
    (condition number kappa = mu_x / sigma_x = 8333) to verify the two-pass
    central-moment algorithm is immune to catastrophic cancellation, and test
    both monoenergetic u_z and varying u_z (where trace-space normalized
    emittance <|u|/c> * eps_x differs from momentum-space emittance eps_{n,u,x}).

    Mutation check:
    Setting probe._drop_cov_term = True drops the <x x'>^2 term, yielding
    sigma_x * sigma_xp instead of sigma_x * sigma_xp * sqrt(1 - rho^2), which
    must fail the statistical tolerance check.
    """
    if use_gpu and not HAS_GPU:
        pytest.skip("CuPy / CUDA GPU not available")

    rng = np.random.default_rng(314159)
    n_p = 20000
    sigma_x = 1.2e-3
    sigma_xp = 2.5e-3
    rho = 0.80
    mu_x = 10.0  # large centroid offset to test numerical stability (kappa = 8333)
    mu_xp = -0.05

    cov = np.array(
        [
            [sigma_x**2, rho * sigma_x * sigma_xp],
            [rho * sigma_x * sigma_xp, sigma_xp**2],
        ]
    )
    samples = rng.multivariate_normal([mu_x, mu_xp], cov, size=n_p)
    x_in = samples[:, 0]
    xp_in = samples[:, 1]
    y_in = rng.normal(0.0, 0.8e-3, size=n_p)
    yp_in = rng.normal(0.0, 1.5e-3, size=n_p)

    # Put 20000 particles inside z_window = [0.02, 0.06] and 500 outlier particles outside
    z_in = rng.uniform(0.021, 0.059, size=n_p)
    uz_in = np.full(n_p, 1.5 * c_light)
    ux_in = xp_in * uz_in
    uy_in = yp_in * uz_in
    w_in = rng.uniform(0.8, 1.2, size=n_p)
    n_eff = float((np.sum(w_in) ** 2) / np.sum(w_in * w_in))

    n_out = 500
    x_all = np.concatenate([x_in, np.full(n_out, 50.0)])
    y_all = np.concatenate([y_in, np.full(n_out, 50.0)])
    z_all = np.concatenate([z_in, np.full(n_out, 0.005)])
    ux_all = np.concatenate([ux_in, np.full(n_out, 0.5 * c_light)])
    uy_all = np.concatenate([uy_in, np.full(n_out, 0.5 * c_light)])
    uz_all = np.concatenate([uz_in, np.full(n_out, 1.5 * c_light)])
    w_all = np.concatenate([w_in, np.full(n_out, 100.0)])

    sp = Species(name="beam", charge=-e_charge, mass=m_e, use_gpu=use_gpu)
    sp.add_particles(
        x=x_all, y=y_all, z=z_all, ux=ux_all, uy=uy_all, uz=uz_all, w=w_all
    )

    probe = EmittanceProbe(z_window=(0.02, 0.06))
    probe.sample(sp, t=0.0, step=0)
    res = probe.result()

    eps_x_analytic = sigma_x * sigma_xp * np.sqrt(1.0 - rho**2)
    u_mag_in = np.sqrt(ux_in**2 + uy_in**2 + uz_in**2)
    bg_analytic = float(np.average(u_mag_in / c_light, weights=w_in))
    norm_eps_x_analytic = bg_analytic * eps_x_analytic

    # Derived statistical error: SE(eps_x) = eps_x / sqrt(N_eff); 3.5-sigma bound
    stat_tol = 3.5 * eps_x_analytic / np.sqrt(n_eff)
    assert res["counts"][0] == n_p
    assert abs(res["emittance_x"][0] - eps_x_analytic) <= stat_tol
    assert (
        abs(res["normalized_emittance_x"][0] - norm_eps_x_analytic)
        <= bg_analytic * stat_tol
    )

    # Also verify exact agreement with direct double-precision central moments on the sample
    w_norm = w_in / np.sum(w_in)
    dx = x_in - np.sum(w_norm * x_in)
    dxp = xp_in - np.sum(w_norm * xp_in)
    dux = ux_in - np.sum(w_norm * ux_in)
    s_xx = float(np.sum(w_norm * dx * dx))
    s_xpxp = float(np.sum(w_norm * dxp * dxp))
    s_xxp = float(np.sum(w_norm * dx * dxp))
    s_uxux = float(np.sum(w_norm * dux * dux))
    s_xux = float(np.sum(w_norm * dx * dux))
    eps_x_sample = float(np.sqrt(s_xx * s_xpxp - s_xxp**2))
    eps_n_u_x_sample = float(np.sqrt(s_xx * s_uxux - s_xux**2) / c_light)
    # Two-pass central moments on the same sample must match to Higham round-off
    # amplified by condition number kappa = |mu_x| / sigma_x
    kappa = abs(mu_x) / sigma_x
    sample_roundoff = 8.0 * _higham_gamma(n_p) * kappa * eps_x_sample
    assert abs(res["emittance_x"][0] - eps_x_sample) <= sample_roundoff
    assert (
        abs(res["momentum_emittance_x"][0] - eps_n_u_x_sample)
        <= 1.5 * sample_roundoff
    )

    # Case 2: Varying u_z across the beam
    uz_var = rng.uniform(0.8 * c_light, 2.4 * c_light, size=n_p)
    ux_var = xp_in * uz_var
    uy_var = yp_in * uz_var
    sp_var = Species(
        name="beam_var_uz", charge=-e_charge, mass=m_e, use_gpu=use_gpu
    )
    sp_var.add_particles(
        x=np.concatenate([x_in, np.full(n_out, 50.0)]),
        y=np.concatenate([y_in, np.full(n_out, 50.0)]),
        z=z_all,
        ux=np.concatenate([ux_var, np.full(n_out, 0.5 * c_light)]),
        uy=np.concatenate([uy_var, np.full(n_out, 0.5 * c_light)]),
        uz=np.concatenate([uz_var, np.full(n_out, 1.5 * c_light)]),
        w=w_all,
    )
    probe_var = EmittanceProbe(z_window=(0.02, 0.06))
    probe_var.sample(sp_var, t=0.0, step=0)
    res_var = probe_var.result()

    dux_var = ux_var - np.sum(w_norm * ux_var)
    s_uxux_var = float(np.sum(w_norm * dux_var * dux_var))
    s_xux_var = float(np.sum(w_norm * dx * dux_var))
    eps_n_u_x_var = float(
        np.sqrt(s_xx * s_uxux_var - s_xux_var**2) / c_light
    )
    assert abs(res_var["emittance_x"][0] - eps_x_analytic) <= stat_tol
    assert abs(res_var["emittance_x"][0] - eps_x_sample) <= sample_roundoff
    assert (
        abs(res_var["momentum_emittance_x"][0] - eps_n_u_x_var)
        <= 8.0 * _higham_gamma(n_p) * kappa * eps_n_u_x_var
    )
    # Because u_z varies and mu_xp != 0, <|u|/c> * eps_x != eps_{n,u,x}
    assert (
        abs(
            res_var["normalized_emittance_x"][0]
            - res_var["momentum_emittance_x"][0]
        )
        > 10.0 * stat_tol
    )

    # Mutation check: dropping <x x'>^2 term must fail the statistical tolerance check
    monkeypatch.setattr(probe, "_drop_cov_term", True)
    res_mut = probe.result()
    with pytest.raises(AssertionError):
        assert abs(res_mut["emittance_x"][0] - eps_x_analytic) <= stat_tol


# ===========================================================================
# (d) TrackRecorder in uniform B vs V1 analytic gyration + absorption NaN
# ===========================================================================


@pytest.mark.parametrize("use_gpu", [False, True])
def test_track_recorder_uniform_b_gyration_and_absorption(
    use_gpu: bool, monkeypatch
):
    """Verify TrackRecorder on 3 gyrating particles in uniform B0 plus an absorbed particle.

    Physical setup (matching V1 in test_079_particles_core.py):
    Three particles with distinct initial transverse momenta and pitches gyrate
    in a uniform magnetic field B = (0, 0, B0). At the end of step m (time
    t_{m+1} = (m + 1) * dt), positions (x, y, z) are at t_{m+1} and proper
    velocities (ux, uy, uz) are at t_{m+1/2} = (m + 0.5) * dt.
    Discrete Boris rotation rotates (ux, uy) by exact angle
        dtheta = -2 * arctan(q * B0 * dt / (2 * gamma * m))
    per step, so the recorded trajectories match the exact discrete Boris
    analytical solution to machine round-off (and the continuous cyclotron
    orbit to O(dt^2)). A 4th tracked particle hits a collector plane and is
    recorded as NaN on all subsequent frames.
    """
    if use_gpu and not HAS_GPU:
        pytest.skip("CuPy / CUDA GPU not available")

    # 3 gyrating particles + 1 particle near collector z = 0.072 that gets absorbed
    x0 = np.array([0.005, -0.008, 0.002, 0.0])
    y0 = np.array([-0.003, 0.006, 0.004, 0.0])
    z0 = np.array([0.020, 0.035, 0.050, 0.070])
    ux0 = np.array([0.20 * c_light, 0.0, -0.15 * c_light, 0.0])
    uy0 = np.array([0.0, 0.30 * c_light, 0.25 * c_light, 0.0])
    uz0 = np.array([0.05 * c_light, -0.04 * c_light, 0.02 * c_light, 0.45 * c_light])
    # Negligible macro-weight so self-fields are ~0 and motion is pure external B0
    w0 = np.full(4, 1.0e-20)
    B0 = 0.15  # Tesla
    n_steps = 25

    def _run_gyration(permute_ids: bool = False):
        grid = GridFIT3D(
            xmin=-0.05,
            xmax=0.05,
            ymin=-0.05,
            ymax=0.05,
            zmin=0.0,
            zmax=0.08,
            Nx=8,
            Ny=8,
            Nz=8,
            verbose=0,
        )
        solver = SolverFIT3D(
            grid,
            cfln=0.45,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            use_gpu=use_gpu,
            verbose=0,
        )
        if use_gpu:
            solver._fused_step = FusedStep(solver)
            solver.one_step = solver._fused_step

        sp = Species(
            name="electrons",
            charge=-e_charge,
            mass=m_e,
            use_gpu=use_gpu,
            track_ids=True,
        )
        sp.add_particles(x=x0, y=y0, z=z0, ux=ux0, uy=uy0, uz=uz0, w=w0)
        tracked_ids = sp.id.get().copy() if use_gpu else sp.id.copy()
        if permute_ids:
            # Permute sp._id before the run so particle 0 and particle 1 swap IDs
            perm = sp.xp.asarray([1, 0, 2, 3], dtype=np.int64)
            sp._id[:4] = sp._id[:4][perm]

        col = Collector(name="top", plane=("z", 0.0715, 1))
        ps = ParticleSolver(
            solver,
            species=[sp],
            collectors=[col],
            B_ext=(0.0, 0.0, B0),
            pusher="boris",
            shape=1,
        ).attach()

        recorder = TrackRecorder(ids=tracked_ids, every=1, ring_size=64)
        ps.add_diagnostic(recorder)

        for _ in range(n_steps):
            solver.one_step()

        return recorder.result(), float(solver.dt)

    res, dt = _run_gyration(permute_ids=False)
    assert res["tracks"].shape == (n_steps, 4, 6)

    # Analytical V1 discrete Boris trajectory for the 3 surviving particles (k = 0, 1, 2)
    q_over_m = -e_charge / m_e
    x_ref_0 = None
    for k in range(3):
        u2_k = ux0[k] ** 2 + uy0[k] ** 2 + uz0[k] ** 2
        gamma_k = np.sqrt(1.0 + u2_k / (c_light**2))
        # Step 0 initializes u^0 -> u^{-1/2} by -dt/2, then each step pushes by +dt
        dtheta_half = 2.0 * np.arctan(0.5 * q_over_m * B0 * (-0.5 * dt) / gamma_k)
        dtheta_step = 2.0 * np.arctan(0.5 * q_over_m * B0 * dt / gamma_k)

        # Rotate u^0 by dtheta_half to get u^{-1/2}
        c_h, s_h = np.cos(-dtheta_half), np.sin(-dtheta_half)
        ux_curr = c_h * ux0[k] - s_h * uy0[k]
        uy_curr = s_h * ux0[k] + c_h * uy0[k]
        uz_curr = uz0[k]
        x_curr, y_curr, z_curr = x0[k], y0[k], z0[k]

        c_s, s_s = np.cos(-dtheta_step), np.sin(-dtheta_step)
        x_ref = np.zeros(n_steps)
        y_ref = np.zeros(n_steps)
        z_ref = np.zeros(n_steps)
        ux_ref = np.zeros(n_steps)
        uy_ref = np.zeros(n_steps)

        for m in range(n_steps):
            ux_next = c_s * ux_curr - s_s * uy_curr
            uy_next = s_s * ux_curr + c_s * uy_curr
            ux_curr, uy_curr = ux_next, uy_next
            x_curr += (ux_curr / gamma_k) * dt
            y_curr += (uy_curr / gamma_k) * dt
            z_curr += (uz_curr / gamma_k) * dt
            x_ref[m] = x_curr
            y_ref[m] = y_curr
            z_ref[m] = z_curr
            ux_ref[m] = ux_curr
            uy_ref[m] = uy_curr

        if k == 0:
            x_ref_0 = x_ref.copy()

        pos_tol = 16.0 * _higham_gamma(n_steps) * 0.05
        vel_tol = 16.0 * _higham_gamma(n_steps) * c_light
        assert np.all(np.abs(res["x"][:, k] - x_ref) <= pos_tol)
        assert np.all(np.abs(res["y"][:, k] - y_ref) <= pos_tol)
        assert np.all(np.abs(res["z"][:, k] - z_ref) <= pos_tol)
        assert np.all(np.abs(res["ux"][:, k] - ux_ref) <= vel_tol)
        assert np.all(np.abs(res["uy"][:, k] - uy_ref) <= vel_tol)

    # Particle 3 hits the collector at z = 0.0715 and must be NaN after absorption
    assert not np.isnan(res["z"][0, 3])
    assert np.all(np.isnan(res["tracks"][-5:, 3, :]))

    # Mutation check: permuting sp._id before the run must fail the gyration assertion
    res_mut, _ = _run_gyration(permute_ids=True)
    with pytest.raises(AssertionError):
        assert np.all(np.abs(res_mut["x"][:, 0] - x_ref_0) <= pos_tol)

    # Duplicate-ID and multi-species checks:
    # 1. Multiple species without explicit species=... must raise ValueError
    grid_dup = GridFIT3D(
        xmin=-0.05,
        xmax=0.05,
        ymin=-0.05,
        ymax=0.05,
        zmin=0.0,
        zmax=0.08,
        Nx=4,
        Ny=4,
        Nz=4,
        verbose=0,
    )
    solver_dup = SolverFIT3D(
        grid_dup,
        cfln=0.45,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_gpu=use_gpu,
        verbose=0,
    )
    sp_a = Species(name="e1", use_gpu=use_gpu, track_ids=True)
    sp_b = Species(name="e2", use_gpu=use_gpu, track_ids=True)
    sp_a.add_particles(
        x=x0[:2], y=y0[:2], z=z0[:2], ux=ux0[:2], uy=uy0[:2], uz=uz0[:2], w=w0[:2]
    )
    sp_b.add_particles(
        x=x0[:2], y=y0[:2], z=z0[:2], ux=ux0[:2], uy=uy0[:2], uz=uz0[:2], w=w0[:2]
    )
    ps_two = ParticleSolver(solver_dup, species=[sp_a, sp_b]).attach()
    with pytest.raises(ValueError, match="single species"):
        ps_two.add_diagnostic(TrackRecorder(ids=[0]))

    # 2. Duplicated particle IDs inside a single species must raise ValueError in .result()
    dup_id_arr = sp_a.xp.asarray([0, 0], dtype=np.int64)
    sp_a._id[:2] = dup_id_arr
    tr_dup = TrackRecorder(ids=[0], species="e1")
    tr_dup.sample(sp_a, t=0.0, step=0)
    with pytest.raises(ValueError, match="duplicate particle id"):
        tr_dup.result()


# ===========================================================================
# (e) GPU no-sync & zero-allocation test for every monitor + GPU == CPU parity
# ===========================================================================


@pytest.mark.skipif(not HAS_GPU, reason="CuPy / CUDA GPU not available")
def test_gpu_monitors_perform_no_host_sync_and_no_allocations(monkeypatch):
    """Prove every diagnostic monitor adds zero host stream syncs and zero pool allocations.

    Uses the 6e7-cycle _SPIN_KERNEL + cp.cuda.Event pattern and _AllocCounterHook
    from tests/test_080_particles_gpu.py across all 5 monitors simultaneously
    and individually, including collector compaction (where sp._n_dirty_gpu is True).
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

    rng = np.random.default_rng(88001)
    n_p = 256
    sp = Species(use_gpu=True, track_ids=True)
    sp.add_particles(
        x=rng.uniform(0.01, 0.05, size=n_p),
        y=rng.uniform(0.01, 0.05, size=n_p),
        z=rng.uniform(0.01, 0.075, size=n_p),
        ux=rng.normal(0.0, 0.2 * c_light, size=n_p),
        uy=rng.normal(0.0, 0.2 * c_light, size=n_p),
        uz=rng.normal(0.4 * c_light, 0.1 * c_light, size=n_p),
        w=rng.uniform(1e4, 2e5, size=n_p),
    )
    collector = Collector(name="top_col", plane=("z", 0.072, 1))
    ps = ParticleSolver(
        solver,
        species=[sp],
        collectors=[collector],
        pusher="boris",
        shape=2,
        gather="galerkin",
        sort_every=0,
    ).attach()

    monitors = [
        PhaseSpaceMonitor(
            axes=("z", "uz"),
            bins=(16, 16),
            ranges=((0.0, 0.08), (-0.2 * c_light, 0.8 * c_light)),
            ring_size=8,
        ),
        EnergySpectrumMonitor(bins=24, range_eV=(0.0, 5.0e5), ring_size=8),
        PlaneCurrentProbe(z_plane=0.04),
        EmittanceProbe(z_window=(0.01, 0.06)),
        TrackRecorder(ids=[0, 5, 10, 20], every=1, ring_size=32),
    ]
    for mon in monitors:
        ps.add_diagnostic(mon)

    dummy_spin = cp.zeros(1, dtype=cp.int32)
    _SPIN_KERNEL((1,), (1,), (np.int64(1000), dummy_spin))
    # Warm up 2 steps (particles starting at z > 0.072 are absorbed, setting _n_dirty_gpu = True)
    for _ in range(2):
        solver.one_step()
    cp.cuda.Stream.null.synchronize()
    assert sp._n_dirty_gpu is True

    # Verify each monitor individually and all monitors together neither sync nor allocate
    for mon in monitors:
        ps.diagnostics = [mon]
        with _AllocCounterHook() as hook_single:
            ev_spin = cp.cuda.Event()
            _SPIN_KERNEL((1,), (1,), (np.int64(60_000_000), dummy_spin))
            ev_spin.record()
            solver.one_step()
            assert sp._n_dirty_gpu is True
            assert not ev_spin.done, (
                f"{type(mon).__name__} synchronized the CUDA stream during one_step()!"
            )
            cp.cuda.Stream.null.synchronize()
            assert ev_spin.done
        assert hook_single.malloc_calls == 0, (
            f"{type(mon).__name__} performed {hook_single.malloc_calls} pool mallocs!"
        )
        assert hook_single.alloc_calls == 0, (
            f"{type(mon).__name__} performed {hook_single.alloc_calls} driver allocs!"
        )

    # All 5 monitors active simultaneously over 4 back-to-back steps
    ps.diagnostics = list(monitors)
    with _AllocCounterHook() as hook_all:
        ev_spin = cp.cuda.Event()
        _SPIN_KERNEL((1,), (1,), (np.int64(60_000_000), dummy_spin))
        ev_spin.record()
        for _ in range(4):
            solver.one_step()
        assert sp._n_dirty_gpu is True
        assert not ev_spin.done
        cp.cuda.Stream.null.synchronize()
        assert ev_spin.done
    assert hook_all.malloc_calls == 0
    assert hook_all.alloc_calls == 0

    # Mutation check: injecting a device-to-host read inside a monitor's _record
    # must cause the not ev_spin.done assertion to fail.
    orig_record = monitors[0]._record

    def _syncing_record(source, t: float, dt: float, step: int):
        orig_record(source, t, dt, step)
        _ = float(sp._z[0].get())

    monkeypatch.setattr(monitors[0], "_record", _syncing_record)
    ev_mut = cp.cuda.Event()
    _SPIN_KERNEL((1,), (1,), (np.int64(60_000_000), dummy_spin))
    ev_mut.record()
    solver.one_step()
    with pytest.raises(AssertionError):
        assert not ev_mut.done
    cp.cuda.Stream.null.synchronize()


@pytest.mark.skipif(not HAS_GPU, reason="CuPy / CUDA GPU not available")
def test_gpu_equals_cpu_all_monitors():
    """Verify GPU and CPU diagnostic results agree on an identical multi-step run (including absorbed NaN tracks)."""
    n_p = 300
    rng = np.random.default_rng(99088)
    x0 = rng.uniform(0.01, 0.05, size=n_p)
    y0 = rng.uniform(0.01, 0.05, size=n_p)
    z0 = rng.uniform(0.01, 0.068, size=n_p)
    ux0 = rng.normal(0.0, 0.15 * c_light, size=n_p)
    uy0 = rng.normal(0.0, 0.15 * c_light, size=n_p)
    uz0 = rng.uniform(0.2 * c_light, 0.5 * c_light, size=n_p)
    # Place tracked particle id=25 near collector z = 0.072 so it is absorbed mid-run
    z0[25] = 0.0695
    ux0[25] = 0.0
    uy0[25] = 0.0
    uz0[25] = 0.45 * c_light
    w0 = rng.uniform(1e4, 1e5, size=n_p)

    def _run(use_gpu: bool):
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
            bc_low=["periodic", "periodic", "periodic"],
            bc_high=["periodic", "periodic", "periodic"],
            use_gpu=use_gpu,
            verbose=0,
        )
        if use_gpu:
            solver._fused_step = FusedStep(solver)
            solver.one_step = solver._fused_step

        sp = Species(
            name="electrons",
            charge=-e_charge,
            mass=m_e,
            use_gpu=use_gpu,
            track_ids=True,
        )
        sp.add_particles(
            x=x0.copy(),
            y=y0.copy(),
            z=z0.copy(),
            ux=ux0.copy(),
            uy=uy0.copy(),
            uz=uz0.copy(),
            w=w0.copy(),
        )
        col = Collector(name="abs_top", plane=("z", 0.072, 1))
        ps = ParticleSolver(
            solver,
            species=[sp],
            collectors=[col],
            B_ext=(0.0, 0.0, 0.08),
            pusher="boris",
            shape=2,
        ).attach()

        m_ps = PhaseSpaceMonitor(
            axes=("z", "uz"),
            bins=(12, 10),
            ranges=((0.0, 0.08), (0.1 * c_light, 0.6 * c_light)),
            weighted=True,
            every=2,
            ring_size=4,
        )
        m_ps_cnt = PhaseSpaceMonitor(
            axes=("z", "uz"),
            bins=(12, 10),
            ranges=((0.0, 0.08), (0.1 * c_light, 0.6 * c_light)),
            weighted=False,
            every=2,
        )
        m_es = EnergySpectrumMonitor(
            bins=16, range_eV=(0.0, 3.0e5), weighted=True, every=1, ring_size=4
        )
        m_pc = PlaneCurrentProbe(z_plane=0.04, every=1)
        m_em = EmittanceProbe(z_window=(0.015, 0.065), every=1)
        m_tr = TrackRecorder(ids=[0, 25, 100], every=1, ring_size=16)

        for d in (m_ps, m_ps_cnt, m_es, m_pc, m_em, m_tr):
            ps.add_diagnostic(d)

        for _ in range(10):
            solver.one_step()

        return {
            "ps": m_ps.result(),
            "ps_cnt": m_ps_cnt.result(),
            "es": m_es.result(),
            "pc": m_pc.result(),
            "em": m_em.result(),
            "tr": m_tr.result(),
        }

    cpu_out = _run(use_gpu=False)
    gpu_out = _run(use_gpu=True)

    gamma_n = _higham_gamma(n_p * 10)
    assert np.array_equal(cpu_out["ps_cnt"]["hist"], gpu_out["ps_cnt"]["hist"])
    assert (
        cpu_out["ps_cnt"]["overflow_count"]
        == gpu_out["ps_cnt"]["overflow_count"]
    )
    assert np.allclose(
        cpu_out["ps"]["hist"],
        gpu_out["ps"]["hist"],
        rtol=8.0 * gamma_n,
        atol=1e-9,
    )
    assert np.allclose(
        cpu_out["ps"]["frames"],
        gpu_out["ps"]["frames"],
        rtol=8.0 * gamma_n,
        atol=1e-9,
    )
    assert np.allclose(
        cpu_out["es"]["hist"],
        gpu_out["es"]["hist"],
        rtol=8.0 * gamma_n,
        atol=1e-9,
    )
    assert np.array_equal(cpu_out["pc"]["counts"], gpu_out["pc"]["counts"])
    assert np.allclose(
        cpu_out["pc"]["I"], gpu_out["pc"]["I"], rtol=8.0 * gamma_n, atol=1e-9
    )
    assert np.array_equal(cpu_out["em"]["counts"], gpu_out["em"]["counts"])
    assert np.allclose(
        cpu_out["em"]["emittance_x"],
        gpu_out["em"]["emittance_x"],
        rtol=16.0 * gamma_n,
        atol=1e-14,
    )
    assert np.allclose(
        cpu_out["em"]["normalized_emittance_x"],
        gpu_out["em"]["normalized_emittance_x"],
        rtol=16.0 * gamma_n,
        atol=1e-14,
    )
    assert np.allclose(
        cpu_out["em"]["momentum_emittance_x"],
        gpu_out["em"]["momentum_emittance_x"],
        rtol=16.0 * gamma_n,
        atol=1e-14,
    )
    # Tracked particle id=25 (column 1) was absorbed by the collector -> NaN entries present
    assert not np.isnan(cpu_out["tr"]["z"][0, 1])
    assert np.all(np.isnan(cpu_out["tr"]["tracks"][-3:, 1, :]))
    assert np.allclose(
        cpu_out["tr"]["tracks"],
        gpu_out["tr"]["tracks"],
        rtol=16.0 * gamma_n,
        atol=1e-14,
        equal_nan=True,
    )


# ===========================================================================
# (f) MPI at -n 2 on CPU: .result() of each monitor equals serial run
# ===========================================================================


@pytest.mark.skipif(
    not HAS_MPI_MULTI, reason="MPI test requires mpiexec -n >= 2"
)
def test_mpi_two_ranks_equals_serial_all_monitors(monkeypatch):
    """Verify 2-rank MPI .result() equals serial run (exact counts, float round-off).

    Includes:
    - Particles migrating in both +z and -z across the z = 0.06 m slab boundary.
    - PlaneCurrentProbe placed directly on the slab boundary (z = 0.06 m) and
      in the slab interior (z = 0.045 m).
    - EmittanceProbe with z_window = (0.03, 0.09) straddling both MPI ranks.
    - TrackRecorder following particles that migrate across MPI slabs as well
      as a particle absorbed by a Collector at z = 0.11 m.
    - Mutation check: disabling MPI reduction in .result() causes the test to fail.
    """
    rng = np.random.default_rng(20260886)
    n_p = 120
    x0 = rng.uniform(0.01, 0.07, size=n_p)
    y0 = rng.uniform(0.01, 0.07, size=n_p)
    # Place particles across both slabs [0.0, 0.06) and [0.06, 0.12)
    z0 = rng.uniform(0.018, 0.102, size=n_p)
    ux0 = rng.normal(0.0, 0.08 * c_light, size=n_p)
    uy0 = rng.normal(0.0, 0.08 * c_light, size=n_p)
    # Bidirectional uz so particles cross the slab boundary z = 0.06 in both directions
    uz0 = np.where(
        np.arange(n_p) % 2 == 0,
        rng.uniform(0.25 * c_light, 0.45 * c_light, size=n_p),
        rng.uniform(-0.45 * c_light, -0.25 * c_light, size=n_p),
    )
    # Ensure particle 0 starts at z = 0.052 with +uz (migrates rank 0 -> rank 1),
    # particle 1 starts at z = 0.068 with -uz (migrates rank 1 -> rank 0),
    # and particle 2 starts at z = 0.106 with +uz (absorbed at collector z = 0.110).
    z0[0], uz0[0] = 0.052, 0.40 * c_light
    z0[1], uz0[1] = 0.068, -0.40 * c_light
    z0[2], uz0[2] = 0.106, 0.42 * c_light
    # Negligible macro-weight so trajectories are governed by external B0 + drift
    # and self-field round-off differences do not shift bin or crossing boundaries.
    w0 = rng.uniform(1.0, 5.0, size=n_p) * 1.0e-18

    def _run_sim(use_mpi: bool):
        grid = GridFIT3D(
            xmin=0.0,
            xmax=0.08,
            ymin=0.0,
            ymax=0.08,
            zmin=0.0,
            zmax=0.12,
            Nx=6,
            Ny=6,
            Nz=12,
            use_mpi=use_mpi,
            verbose=0,
        )
        solver = SolverFIT3D(
            grid,
            cfln=0.45,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            use_mpi=use_mpi,
            use_gpu=False,
            verbose=0,
        )
        sp = Species(
            name="electrons",
            charge=-e_charge,
            mass=m_e,
            use_gpu=False,
            track_ids=True,
        )
        sp.add_particles(
            x=x0.copy(),
            y=y0.copy(),
            z=z0.copy(),
            ux=ux0.copy(),
            uy=uy0.copy(),
            uz=uz0.copy(),
            w=w0.copy(),
        )
        col = Collector(name="top_abs", plane=("z", 0.110, 1))
        ps = ParticleSolver(
            solver,
            species=[sp],
            collectors=[col],
            B_ext=(0.0, 0.0, 0.05),
            pusher="boris",
            shape=1,
        ).attach()

        # Range chosen so some particles fall inside and some overflow
        m_ps_cnt = PhaseSpaceMonitor(
            axes=("z", "uz"),
            bins=(12, 10),
            ranges=((0.02, 0.10), (-0.42 * c_light, 0.42 * c_light)),
            weighted=False,
            every=1,
            ring_size=6,
        )
        m_ps_w = PhaseSpaceMonitor(
            axes=("z", "uz"),
            bins=(12, 10),
            ranges=((0.02, 0.10), (-0.42 * c_light, 0.42 * c_light)),
            weighted=True,
            every=1,
            ring_size=6,
        )
        m_es_cnt = EnergySpectrumMonitor(
            bins=15,
            range_eV=(1.5e4, 5.5e4),
            weighted=False,
            every=1,
            ring_size=6,
        )
        m_es_w = EnergySpectrumMonitor(
            bins=15,
            range_eV=(1.5e4, 5.5e4),
            weighted=True,
            every=1,
            ring_size=6,
        )
        m_pc_slab = PlaneCurrentProbe(z_plane=0.06, every=1)
        m_pc_int = PlaneCurrentProbe(z_plane=0.045, every=1)
        m_em = EmittanceProbe(z_window=(0.03, 0.09), every=1)
        m_tr = TrackRecorder(ids=[0, 1, 2, 10, 25], every=1, ring_size=32)

        for d in (
            m_ps_cnt,
            m_ps_w,
            m_es_cnt,
            m_es_w,
            m_pc_slab,
            m_pc_int,
            m_em,
            m_tr,
        ):
            ps.add_diagnostic(d)

        for _ in range(25):
            solver.one_step()

        mig_lo = ps._n_migrated_lo
        mig_hi = ps._n_migrated_hi
        if use_mpi:
            mig_lo = int(solver.comm.allreduce(mig_lo, op=MPI.SUM))
            mig_hi = int(solver.comm.allreduce(mig_hi, op=MPI.SUM))

        return (
            {
                "ps_cnt": m_ps_cnt.result(),
                "ps_w": m_ps_w.result(),
                "es_cnt": m_es_cnt.result(),
                "es_w": m_es_w.result(),
                "pc_slab": m_pc_slab.result(),
                "pc_int": m_pc_int.result(),
                "em": m_em.result(),
                "tr": m_tr.result(),
                "mig_lo": mig_lo,
                "mig_hi": mig_hi,
            },
            m_ps_cnt,
        )

    ser, _ = _run_sim(use_mpi=False)
    mpi_out, m_ps_cnt_mpi = _run_sim(use_mpi=True)

    # Confirm particles actually migrated across the z = 0.06 slab boundary in both directions
    assert mpi_out["mig_lo"] > 0
    assert mpi_out["mig_hi"] > 0

    # 1. Exact integer equality for all counts
    assert np.array_equal(ser["ps_cnt"]["hist"], mpi_out["ps_cnt"]["hist"])
    assert np.array_equal(ser["ps_cnt"]["frames"], mpi_out["ps_cnt"]["frames"])
    assert (
        ser["ps_cnt"]["overflow_count"] == mpi_out["ps_cnt"]["overflow_count"]
    )
    assert np.array_equal(
        ser["ps_cnt"]["frame_overflow_count"],
        mpi_out["ps_cnt"]["frame_overflow_count"],
    )
    assert np.array_equal(ser["es_cnt"]["hist"], mpi_out["es_cnt"]["hist"])
    assert np.array_equal(ser["es_cnt"]["frames"], mpi_out["es_cnt"]["frames"])
    assert (
        ser["es_cnt"]["overflow_count"] == mpi_out["es_cnt"]["overflow_count"]
    )
    assert np.array_equal(
        ser["pc_slab"]["counts"], mpi_out["pc_slab"]["counts"]
    )
    assert np.array_equal(ser["pc_int"]["counts"], mpi_out["pc_int"]["counts"])
    assert np.array_equal(ser["em"]["counts"], mpi_out["em"]["counts"])

    # 2. Floating-point outputs agree within Higham summation round-off bound
    gamma_tot = _higham_gamma(n_p * 25)
    w_scale = float(np.max(w0)) * n_p
    q_scale = e_charge * w_scale

    assert np.all(
        np.abs(ser["ps_w"]["hist"] - mpi_out["ps_w"]["hist"])
        <= 8.0 * gamma_tot * w_scale
    )
    assert np.all(
        np.abs(ser["ps_w"]["frames"] - mpi_out["ps_w"]["frames"])
        <= 8.0 * gamma_tot * w_scale
    )
    assert (
        abs(ser["ps_w"]["overflow"] - mpi_out["ps_w"]["overflow"])
        <= 8.0 * gamma_tot * w_scale
    )
    assert np.all(
        np.abs(ser["es_w"]["hist"] - mpi_out["es_w"]["hist"])
        <= 8.0 * gamma_tot * w_scale
    )
    assert (
        abs(ser["es_w"]["overflow"] - mpi_out["es_w"]["overflow"])
        <= 8.0 * gamma_tot * w_scale
    )

    for key_pc in ("pc_slab", "pc_int"):
        assert np.all(
            np.abs(ser[key_pc]["dQ"] - mpi_out[key_pc]["dQ"])
            <= 8.0 * gamma_tot * q_scale
        )
        assert (
            abs(ser[key_pc]["Q_total"] - mpi_out[key_pc]["Q_total"])
            <= 8.0 * gamma_tot * q_scale
        )

    for key_em in (
        "emittance_x",
        "emittance_y",
        "normalized_emittance_x",
        "normalized_emittance_y",
        "momentum_emittance_x",
        "momentum_emittance_y",
        "mean_x",
        "mean_xp",
        "var_x",
        "var_xp",
        "cov_xxp",
    ):
        ref_arr = ser["em"][key_em]
        mpi_arr = mpi_out["em"][key_em]
        scale = np.maximum(np.abs(ref_arr), 1.0e-15)
        assert np.all(np.abs(ref_arr - mpi_arr) <= 32.0 * gamma_tot * scale)

    # 3. TrackRecorder: exact NaN pattern for absorbed particle 2 and seamless
    #    trajectory agreement across rank migrations for particles 0 and 1.
    assert np.array_equal(
        np.isnan(ser["tr"]["tracks"]), np.isnan(mpi_out["tr"]["tracks"])
    )
    assert np.any(np.isnan(mpi_out["tr"]["z"][:, 2]))
    finite_mask = ~np.isnan(ser["tr"]["tracks"])
    assert np.allclose(
        ser["tr"]["tracks"][finite_mask],
        mpi_out["tr"]["tracks"][finite_mask],
        rtol=32.0 * gamma_tot,
        atol=1.0e-15,
    )

    # 4. Duplicate ID across MPI ranks (valid_sum > 1) must raise ValueError in .result()
    grid_dup = GridFIT3D(
        xmin=0.0,
        xmax=0.08,
        ymin=0.0,
        ymax=0.08,
        zmin=0.0,
        zmax=0.12,
        Nx=4,
        Ny=4,
        Nz=8,
        use_mpi=True,
        verbose=0,
    )
    solver_dup = SolverFIT3D(
        grid_dup,
        cfln=0.45,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_mpi=True,
        use_gpu=False,
        verbose=0,
    )
    sp_dup = Species(name="electrons", use_gpu=False, track_ids=True)
    # Seed one particle inside each rank's slab so both ranks assign id=0 independently
    rank = solver_dup.comm.Get_rank()
    z_rank = 0.03 if rank == 0 else 0.09
    sp_dup.add_particles(
        x=np.array([0.04]),
        y=np.array([0.04]),
        z=np.array([z_rank]),
        ux=np.zeros(1),
        uy=np.zeros(1),
        uz=np.zeros(1),
        w=np.ones(1),
    )
    ps_dup = ParticleSolver(solver_dup, species=[sp_dup]).attach()
    tr_mpi_dup = TrackRecorder(ids=[0])
    ps_dup.add_diagnostic(tr_mpi_dup)
    tr_mpi_dup.sample(ps_dup, t=0.0, step=0)
    with pytest.raises(ValueError, match="duplicate particle id"):
        tr_mpi_dup.result()

    # 5. Mutation check: disabling MPI reduction (_get_comm -> None) must fail
    monkeypatch.setattr(m_ps_cnt_mpi, "_get_comm", lambda: None)
    mut_res = m_ps_cnt_mpi.result()
    with pytest.raises(AssertionError):
        assert np.array_equal(ser["ps_cnt"]["hist"], mut_res["hist"])

