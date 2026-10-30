# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Milestone M3 (V8) MPI validation for ``wakis.particles``.

Run with 2 or 4 MPI ranks::

    mpiexec --allow-run-as-root --oversubscribe --timeout 180 -n 2 python -m pytest -v tests/test_082_particles_mpi.py
    mpiexec --allow-run-as-root --oversubscribe --timeout 180 -n 4 python -m pytest -v tests/test_082_particles_mpi.py

Covers:
1. V2 discrete charge conservation and parallel-vs-serial equivalence over
   ``>= 500`` coupled steps on uniform and smoothly graded ``z`` meshes with
   relativistic particles crossing internal ``z``-slab boundaries repeatedly.
2. V3 cold-plasma oscillation dispersion, energy conservation, and
   parallel-vs-serial equivalence across ``z``-slabs.
3. Exact particle count, ID set, and total charge conservation at every step
   for particles initialized directly on internal slab faces ``z = z_face``
   and at ``z = nextafter(z_face, -inf)``.
4. Per-rank ``BeamInjector`` and ``Collector`` / PEC wall diagnostics reduced
   via ``MPI.Allreduce`` matching a serial reference run.
5. Code-path mutations that must fail:
   (a) overwriting instead of adding in ``_mpi_sum_guards``,
   (b) dropping migrating particles in ``_mpi_migrate_particles``,
   (c) migrating before instead of after the particle push.
"""

from __future__ import annotations

import contextlib
import faulthandler
import time

import numpy as np
import pytest
from scipy.constants import c, e as e_charge, epsilon_0 as eps_0, m_e

from wakis import GridFIT3D, SolverFIT3D
from wakis.particles import (
    BeamInjector,
    Collector,
    ParticleCFLError,
    ParticleSolver,
    Species,
    cold_plasma_dispersion_discrete,
)

MPI = pytest.importorskip("mpi4py.MPI")

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() < 2,
    reason="MPI particle tests require mpiexec -n >= 2",
)

_TEST_HANG_TIMEOUT_S = 45


@contextlib.contextmanager
def _hang_guard(timeout_s: float):
    """Arm C-level ``faulthandler.dump_traceback_later(timeout_s, exit=True)`` for a block.

    Unlike Python ``signal.SIGALRM``, ``faulthandler.dump_traceback_later`` runs
    on a dedicated C watchdog thread and terminates the process with nonzero
    exit status even when a rank is blocked inside a C MPI or NCCL collective.
    """
    faulthandler.dump_traceback_later(timeout_s, exit=True)
    try:
        yield
    finally:
        faulthandler.dump_traceback_later(_TEST_HANG_TIMEOUT_S, exit=True)


@pytest.fixture(autouse=True)
def _mpi_sync_between_tests():
    """Guard each test with C-level faulthandler timeout and synchronize all MPI ranks at test end."""
    faulthandler.dump_traceback_later(_TEST_HANG_TIMEOUT_S, exit=True)
    try:
        yield
        if MPI.COMM_WORLD.Get_size() >= 2:
            MPI.COMM_WORLD.Barrier()
    finally:
        faulthandler.cancel_dump_traceback_later()


def _assert_interior_bidirectional_migration(
    sent_lo: int, sent_hi: int, comm
) -> None:
    """Assert per interior rank (``has_lo and has_hi``) that particles were sent both down and up during the run."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    has_lo = rank > 0
    has_hi = rank < size - 1
    if has_lo and has_hi:
        local_ok = 1 if (sent_lo > 0 and sent_hi > 0) else 0
    elif has_hi:
        local_ok = 1 if (sent_lo == 0 and sent_hi > 0) else 0
    elif has_lo:
        local_ok = 1 if (sent_lo > 0 and sent_hi == 0) else 0
    else:
        local_ok = 1
    all_ok = int(comm.allreduce(local_ok, op=MPI.MIN))
    tot_lo = int(comm.allreduce(int(sent_lo), op=MPI.SUM))
    tot_hi = int(comm.allreduce(int(sent_hi), op=MPI.SUM))
    assert all_ok == 1, (
        f"Rank {rank}/{size} (has_lo={has_lo}, has_hi={has_hi}) did not send "
        f"particles in both required directions: sent_lo={sent_lo}, sent_hi={sent_hi}"
    )
    assert tot_lo > 0 and tot_hi > 0



# IEEE 754 binary64 unit roundoff u = 2^-53 (half machine epsilon) from
# Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., Sec. 3.1.
_IEEE754_BINARY64_UNIT_ROUNDOFF = float(2.0**-53)

# Pre-registered random seeds for the 3-seed per-step rounding-perturbation envelope.
_PERTURBATION_SEEDS = (101, 202, 303)

# Pre-registered heuristic coherence multiplier applied to max_seed_deviation:
# injected perturbations have independent random signs (+-1), whereas actual
# hardware rounding errors (FMA contraction, summation order, slab association)
# can grow coherently over N time steps by up to sqrt(N) relative to a random walk.
# The factor 2.0 is a heuristic, not a worst-case derivation.
_PERTURBATION_COHERENCE_FACTOR = 2.0

# Smallest power-of-ten relative guard-current scaling epsilon for which
# scaling the received guard plane by (1 + epsilon) in
# ParticleSolver._accumulate_guard_plane causes the 2-GPU V3 cold-plasma run to
# exceed derived_mpi_roundoff_bound (obs/bound > 1 at epsilon, vs obs/bound <= 1
# at 0.1 * epsilon) while staying below the 1e-12 Gauss-residual tolerance used by
# these tests. The mutation is Gauss-silent only to that tolerance: it raises
# gauss_res well above the unmutated round-off floor (both values are printed), so a
# Gauss check set near the floor would be the more sensitive detector of this defect.
# The interpolated threshold epsilon_crit and its ratio to
# gamma_n are printed at run time by test_multi_gpu_v3_cold_plasma_matches_1gpu_and_cpu_serial.
_V3_GUARD_SCALE_MUTATION_EPS = 1.0e-8


def _max_relative_run_deviation(ser_ref: dict, run_cmp: dict) -> float:
    """Return the maximum dimensionless relative deviation between ``ser_ref`` and ``run_cmp``."""
    eps_mach = float(np.finfo(np.float64).eps)
    E_a, E_b = ser_ref["E"], run_cmp["E"]
    H_a, H_b = ser_ref["H"], run_cmp["H"]

    e_norm = max(
        float(np.max(np.abs(_field_comp_host(E_a, c_name)))) for c_name in "xyz"
    )
    h_norm = max(
        float(np.max(np.abs(_field_comp_host(H_a, c_name)))) for c_name in "xyz"
    )
    rel_env = eps_mach
    if e_norm > 0.0:
        for c_name in "xyz":
            diff_e = float(
                np.max(
                    np.abs(
                        _field_comp_host(E_a, c_name)
                        - _field_comp_host(E_b, c_name)
                    )
                )
            )
            rel_env = max(rel_env, diff_e / e_norm)
    if h_norm > 0.0:
        for c_name in "xyz":
            diff_h = float(
                np.max(
                    np.abs(
                        _field_comp_host(H_a, c_name)
                        - _field_comp_host(H_b, c_name)
                    )
                )
            )
            rel_env = max(rel_env, diff_h / h_norm)

    sp_a, sp_b = ser_ref["sp"], run_cmp["sp"]
    ord_a = np.argsort(sp_a.id)
    ord_b = np.argsort(sp_b.id)
    dz_scale = float(ser_ref.get("L_z_min", ser_ref.get("dz", 1.0)))
    for pos_attr in ("x", "y", "z"):
        diff_p = float(
            np.max(
                np.abs(
                    getattr(sp_a, pos_attr)[ord_a]
                    - getattr(sp_b, pos_attr)[ord_b]
                )
            )
        )
        rel_env = max(rel_env, diff_p / dz_scale)

    u_norm = float(np.max(np.sqrt(sp_a.ux**2 + sp_a.uy**2 + sp_a.uz**2)))
    if u_norm > 0.0:
        for vel_attr in ("ux", "uy", "uz"):
            diff_u = float(
                np.max(
                    np.abs(
                        getattr(sp_a, vel_attr)[ord_a]
                        - getattr(sp_b, vel_attr)[ord_b]
                    )
                )
            )
            rel_env = max(rel_env, diff_u / u_norm)

    if "ke_tot" in ser_ref and ser_ref["ke_tot"] > 0.0:
        rel_env = max(
            rel_env,
            abs(ser_ref["ke_tot"] - run_cmp["ke_tot"]) / ser_ref["ke_tot"],
        )
    if "ue_tot" in ser_ref and ser_ref["ue_tot"] > 0.0:
        rel_env = max(
            rel_env,
            abs(ser_ref["ue_tot"] - run_cmp["ue_tot"]) / ser_ref["ue_tot"],
        )
    if "ke" in ser_ref and "ue" in ser_ref:
        e_tot_a = ser_ref["ke"] + ser_ref["ue"]
        e_tot_b = run_cmp["ke"] + run_cmp["ue"]
        e_scale = max(float(np.max(e_tot_a)), 1e-30)
        drift_a = float(np.max(np.abs(e_tot_a - e_tot_a[0])) / e_tot_a[0])
        drift_b = float(np.max(np.abs(e_tot_b - e_tot_b[0])) / e_tot_b[0])
        rel_env = max(
            rel_env,
            float(np.max(np.abs(ser_ref["ke"] - run_cmp["ke"])) / e_scale),
            float(np.max(np.abs(ser_ref["ue"] - run_cmp["ue"])) / e_scale),
            abs(drift_a - drift_b),
        )
    return float(rel_env)


def _run_serial_perturbation_envelope(
    run_fn,
    *args,
    seeds: tuple[int, ...] = _PERTURBATION_SEEDS,
    deposit_rel_override: float | None = None,
    **kwargs,
) -> tuple[dict, list[dict]]:
    """Run unperturbed CPU serial reference (recording ``n``, ``k``, and ``N``) and perturbed runs over ``seeds``."""
    import wakis.particles.shapes as shapes_mod
    import wakis.particles.solver as solver_mod

    u = _IEEE754_BINARY64_UNIT_ROUNDOFF
    max_n = 0
    max_k = 0
    n_steps_counted = 0
    orig_acc = shapes_mod._accumulate_flat
    orig_pm = solver_mod.push_momentum
    orig_pp = solver_mod.push_position
    orig_uh = SolverFIT3D._update_H
    orig_ue = SolverFIT3D._update_E

    def _track_acc(target_flat, idx, weights, n_total, xp=np):
        nonlocal max_n
        if idx.size > 0 and xp is np:
            nz = weights != 0.0
            if np.any(nz):
                c_max = int(np.max(np.bincount(idx[nz], minlength=n_total)))
                if c_max > max_n:
                    max_n = c_max
        return orig_acc(target_flat, idx, weights, n_total, xp=xp)

    def _track_uh(self):
        nonlocal max_k, n_steps_counted
        n_steps_counted += 1
        if not self.use_gpu:
            for mat in (self.tDsiDmuiDaC, self.itDaiDepsDstC):
                k_row = np.diff(mat.indptr)
                if k_row.size > 0:
                    k_m = int(np.max(k_row))
                    if k_m > max_k:
                        max_k = k_m
        return orig_uh(self)

    shapes_mod._accumulate_flat = _track_acc
    SolverFIT3D._update_H = _track_uh
    try:
        ser_ref = run_fn(*args, **kwargs)
    finally:
        shapes_mod._accumulate_flat = orig_acc
        SolverFIT3D._update_H = orig_uh

    n_val = max(int(max_n), 1)
    k_val = max(int(max_k), 1)
    gamma_n = float((n_val * u) / (1.0 - n_val * u))
    gamma_k = float((k_val * u) / (1.0 - k_val * u))
    dep_scale = (
        float(deposit_rel_override)
        if deposit_rel_override is not None
        else gamma_n
    )
    ser_ref["max_edge_contributions"] = n_val
    ser_ref["gamma_n"] = gamma_n
    ser_ref["max_curl_row_nnz"] = k_val
    ser_ref["gamma_k"] = gamma_k
    ser_ref["n_steps"] = int(kwargs.get("n_steps", n_steps_counted))
    ser_ref["perturbation_seeds"] = [int(s) for s in seeds]

    pert_runs: list[dict] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        in_step = [False]

        def _pert_acc(target_flat, idx, weights, n_total, xp=np):
            orig_acc(target_flat, idx, weights, n_total, xp=xp)
            if in_step[0] and idx.size > 0 and xp is np:
                sum_abs_w = np.bincount(
                    idx, weights=np.abs(weights), minlength=n_total
                )
                signs = np.where(
                    rng.integers(0, 2, size=n_total) == 0, -1.0, 1.0
                )
                target_flat += signs * (dep_scale * sum_abs_w)

        def _pert_pm(sp, Ex, Ey, Ez, Bx, By, Bz, dt, pusher="boris"):
            in_step[0] = True
            orig_pm(sp, Ex, Ey, Ez, Bx, By, Bz, dt=dt, pusher=pusher)
            n_p = sp.n_particles
            if n_p > 0:
                for attr in ("_ux", "_uy", "_uz"):
                    arr = getattr(sp, attr)[:n_p]
                    signs = np.where(
                        rng.integers(0, 2, size=n_p) == 0, -1.0, 1.0
                    )
                    arr += signs * (u * np.abs(arr))

        def _pert_pp(sp, dt):
            orig_pp(sp, dt=dt)
            n_p = sp.n_particles
            if n_p > 0:
                for attr in ("_x", "_y", "_z"):
                    arr = getattr(sp, attr)[:n_p]
                    signs = np.where(
                        rng.integers(0, 2, size=n_p) == 0, -1.0, 1.0
                    )
                    arr += signs * (u * np.abs(arr))

        def _pert_uh(self):
            orig_uh(self)
            if not self.use_gpu:
                cache = getattr(self, "_abs_ch_pert_cache", None)
                if cache is None:
                    ch = self.tDsiDmuiDaC
                    k_row = np.diff(ch.indptr)
                    gk = (k_row * u) / (1.0 - k_row * u)
                    cache = (ch, abs(ch), gk)
                    self._abs_ch_pert_cache = cache
                ch, abs_ch, gk = cache
                e_arr = self.E.toarray()
                abs_curl = abs_ch * np.abs(e_arr)
                signs = np.where(
                    rng.integers(0, 2, size=abs_curl.size) == 0, -1.0, 1.0
                )
                self.H.array += signs * (self.dt * gk * abs_curl)
                abs_dt_curl = self.dt * np.abs(ch * e_arr)
                signs_upd = np.where(
                    rng.integers(0, 2, size=abs_curl.size) == 0, -1.0, 1.0
                )
                self.H.array += signs_upd * (
                    u * (abs_dt_curl + np.abs(self.H.array))
                )

        def _pert_ue(self):
            orig_ue(self)
            if not self.use_gpu:
                cache = getattr(self, "_abs_ce_pert_cache", None)
                if cache is None:
                    ce = self.itDaiDepsDstC
                    k_row = np.diff(ce.indptr)
                    gk = (k_row * u) / (1.0 - k_row * u)
                    cache = (ce, abs(ce), gk)
                    self._abs_ce_pert_cache = cache
                ce, abs_ce, gk = cache
                h_arr = self.H.toarray()
                abs_curl = abs_ce * np.abs(h_arr)
                signs = np.where(
                    rng.integers(0, 2, size=abs_curl.size) == 0, -1.0, 1.0
                )
                self.E.array += signs * (self.dt * gk * abs_curl)
                abs_dt_curl = self.dt * np.abs(ce * h_arr)
                signs_upd = np.where(
                    rng.integers(0, 2, size=abs_curl.size) == 0, -1.0, 1.0
                )
                self.E.array += signs_upd * (
                    u * (abs_dt_curl + np.abs(self.E.array))
                )

        shapes_mod._accumulate_flat = _pert_acc
        solver_mod.push_momentum = _pert_pm
        solver_mod.push_position = _pert_pp
        SolverFIT3D._update_H = _pert_uh
        SolverFIT3D._update_E = _pert_ue
        try:
            pert_runs.append(run_fn(*args, **kwargs))
        finally:
            shapes_mod._accumulate_flat = orig_acc
            solver_mod.push_momentum = orig_pm
            solver_mod.push_position = orig_pp
            SolverFIT3D._update_H = orig_uh
            SolverFIT3D._update_E = orig_ue

    return ser_ref, pert_runs


def derived_mpi_roundoff_bound(
    ser_ref: dict,
    pert_runs: list[dict] | tuple[dict, ...] | dict,
    safety_factor: float = _PERTURBATION_COHERENCE_FACTOR,
) -> float:
    """Return a floating-point round-off bound from a 3-seed per-step perturbation-amplification envelope.

    Methodology
    -----------
    Discrepancies between CPU serial, CPU MPI ``z``-slab, single-GPU, and
    multi-GPU trajectories arise from multiple floating-point sources — not
    only particle summation order:
    1. Non-associative summation order during current deposition (deterministic
       particle-order ``np.bincount`` on CPU vs warp-scheduled ``atomicAdd`` on
       GPU, plus arbitrary particle permutations across MPI slabs).
    2. Slab-boundary association in ``_mpi_sum_guards`` (accumulating owned
       slab partials and guard-plane partials separately before summing).
    3. Hardware fused multiply-add (``FMA``, ``fl(a * b + c)``) contraction in
       CUDA device kernels vs separate IEEE 754 multiply and add on CPU during
       B-spline weight evaluation, field gathering, relativistic momentum/position
       pushes, and Esirkepov current deposition.
    4. Sparse curl matrix-vector products ``C * x``, ``curl *= dt`` scaling, and
       ``H -= curl`` / ``E += curl`` updates in ``_update_H`` and ``_update_E``
       (SciPy CSR vs cuSPARSE reduction order and FMA).

    By standard floating-point error analysis (Higham, *Accuracy and Stability
    of Numerical Algorithms*, 2nd ed., Sec. 3.1), per-step rounding differences
    in these operations are bounded by and injected as independent random-sign
    (``+-1``) terms:
    - ``gamma_n * sum(|w|)`` on each deposited current edge, where
      ``u = 2^-53`` is the IEEE 754 binary64 unit roundoff, ``n`` is the
      maximum number of nonzero contributions per edge measured over the run,
      and ``gamma_n = n * u / (1 - n * u)`` (which also bounds FMA rounding
      ``|fl(a * b + c) - (a * b + c)| <= u * |a * b + c|``);
    - ``u * |q|`` on each particle position and velocity/momentum component ``q``;
    - ``dt * gamma_k * sum_j(|C_ij * x_j|)`` on each ``H`` and ``E`` sparse
      curl product row ``i`` (where ``k <= 4`` is the number of nonzeros per row
      of the FIT curl matrix ``C`` and ``gamma_k = k * u / (1 - k * u)``), plus
      ``u * (|dt * (C * E)_i| + |H_i^{n+1/2}|)`` and
      ``u * (|dt * (C * H)_i| + |E_i^{n+1}|)`` for the ``curl *= dt`` scaling
      and final ``H -= curl`` / ``E += curl`` field-update additions.
    Still omitted from the per-step perturbation model are CPML auxiliary field
    updates (if any; inactive in these PEC/periodic tests) and the separate
    rounding of the source / ``J``-to-Ampère subtraction ``-dt * ieps * J``
    (beyond the ``gamma_n * sum(|w|)`` perturbation already injected into ``J``
    and the final ``u * |E^{n+1}|`` update term).

    The CPU serial reference is re-run across ``3`` pre-registered random seeds
    ``(101, 202, 303)`` injecting these per-step perturbations with independent
    random signs ``+-1`` over ``N`` steps. The bound is the maximum final
    relative deviation across the 3 seeds multiplied by a pre-registered
    **heuristic ×2** factor (``safety_factor = 2.0``, not a worst-case
    derivation: coherent rounding-error growth over ``N`` steps can exceed
    random-sign ``+-1`` growth by up to ``sqrt(N)``; ``N`` is recorded in
    ``ser_ref["roundoff_envelope"]["n_steps"]``).

    Parameters
    ----------
    ser_ref : dict
        Output dictionary from the unperturbed CPU serial reference run.
    pert_runs : sequence of dict or dict
        Sequence of output dictionaries from the 3 perturbed serial runs.
    safety_factor : float, optional
        Pre-registered heuristic coherence multiplier. Default is ``2.0``.

    Returns
    -------
    bound : float
        Dimensionless relative round-off bound.
    """
    runs_seq = (
        [pert_runs] if isinstance(pert_runs, dict) else list(pert_runs)
    )
    seed_devs = [
        _max_relative_run_deviation(ser_ref, pr) for pr in runs_seq
    ]
    max_dev = max(seed_devs) if seed_devs else float(np.finfo(np.float64).eps)
    bound = float(safety_factor * max_dev)
    ser_ref["roundoff_envelope"] = {
        "u": float(_IEEE754_BINARY64_UNIT_ROUNDOFF),
        "n": int(ser_ref.get("max_edge_contributions", 0)),
        "gamma_n": float(ser_ref.get("gamma_n", 0.0)),
        "k": int(ser_ref.get("max_curl_row_nnz", 0)),
        "gamma_k": float(ser_ref.get("gamma_k", 0.0)),
        "n_steps": int(ser_ref.get("n_steps", 0)),
        "seeds": list(ser_ref.get("perturbation_seeds", _PERTURBATION_SEEDS)),
        "seed_deviations": [float(d) for d in seed_devs],
        "coherence_factor": float(safety_factor),
        "bound": bound,
    }
    return bound


def _report_roundoff_comparison(
    label: str, ser_ref: dict, run_cmp: dict, bound: float
) -> float:
    """Compute, record, and print the ``observed / bound`` ratio for ``run_cmp`` vs ``ser_ref``."""
    obs = _max_relative_run_deviation(ser_ref, run_cmp)
    ratio = float(obs / bound) if bound > 0.0 else float("inf")
    env = ser_ref.get("roundoff_envelope", {})
    info = {
        "observed": float(obs),
        "bound": float(bound),
        "observed_to_bound_ratio": ratio,
        "n": int(env.get("n", ser_ref.get("max_edge_contributions", 0))),
        "gamma_n": float(env.get("gamma_n", ser_ref.get("gamma_n", 0.0))),
        "k": int(env.get("k", ser_ref.get("max_curl_row_nnz", 0))),
        "gamma_k": float(env.get("gamma_k", ser_ref.get("gamma_k", 0.0))),
        "n_steps": int(env.get("n_steps", ser_ref.get("n_steps", 0))),
        "seeds": list(env.get("seeds", _PERTURBATION_SEEDS)),
    }
    ser_ref.setdefault("roundoff_comparisons", {})[label] = info
    print(
        f"[roundoff {label}] obs={obs:.4e}, bound={bound:.4e}, "
        f"obs/bound={ratio:.4f} (N={info['n_steps']}, n={info['n']}, "
        f"gamma_n={info['gamma_n']:.4e}, k={info['k']}, "
        f"gamma_k={info['gamma_k']:.4e}, seeds={info['seeds']})"
    )
    return ratio


def _make_v2_grid(graded: bool, use_mpi: bool, Nx: int = 8, Ny: int = 8, Nz: int = 12):
    """Construct a 3D grid with uniform or smoothly graded z spacing."""
    Lx, Ly, Lz = 0.08, 0.08, 0.12
    xg = np.linspace(0.0, Lx, Nx + 1)
    yg = np.linspace(0.0, Ly, Ny + 1)
    if graded:
        s = np.linspace(0.0, 1.0, Nz + 1)
        zg = Lz * (0.8 * s + 0.2 * s * s)
        return GridFIT3D(
            xmin=0.0,
            xmax=Lx,
            ymin=0.0,
            ymax=Ly,
            zmin=0.0,
            zmax=Lz,
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            x=xg,
            y=yg,
            z=zg,
            use_mpi=use_mpi,
            verbose=0,
        )
    return GridFIT3D(
        xmin=0.0,
        xmax=Lx,
        ymin=0.0,
        ymax=Ly,
        zmin=0.0,
        zmax=Lz,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        use_mpi=use_mpi,
        verbose=0,
    )


def _run_v2_coupled(
    graded: bool,
    use_mpi: bool,
    n_steps: int = 500,
    use_gpu: bool = False,
    particles_comm: str | None = None,
    reverse_particles: bool = False,
):
    """Run V2 relativistic particle gyration across z-slab faces for ``n_steps``."""
    grid = _make_v2_grid(graded=graded, use_mpi=use_mpi)
    solver = SolverFIT3D(
        grid,
        cfln=0.5,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_mpi=use_mpi,
        use_gpu=use_gpu,
        verbose=0,
    )

    rng = np.random.default_rng(202605)
    n_p = 48
    x0 = rng.uniform(0.0, 0.08, n_p)
    y0 = rng.uniform(0.0, 0.08, n_p)
    z0 = rng.uniform(0.024, 0.096, n_p)
    ux0 = rng.uniform(-0.20, 0.20, n_p) * c
    uy0 = rng.uniform(-0.20, 0.20, n_p) * c
    uz0 = rng.uniform(-0.32, 0.32, n_p) * c
    w0 = np.full(n_p, 5.0e5)
    ids0 = np.arange(n_p, dtype=np.int64)
    if reverse_particles:
        x0 = x0[::-1].copy()
        y0 = y0[::-1].copy()
        z0 = z0[::-1].copy()
        ux0 = ux0[::-1].copy()
        uy0 = uy0[::-1].copy()
        uz0 = uz0[::-1].copy()
        w0 = w0[::-1].copy()
        ids0 = ids0[::-1].copy()

    # Transverse external magnetic field confines Larmor orbits axially (~0.015 m)
    # so particles cross internal z-slab faces (z = 0.03, 0.06, 0.09 m) hundreds
    # of times over 500 steps while staying inside the cavity.
    B_ext = (0.045, -0.035, 0.0)
    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    sp.add_particles(x=x0, y=y0, z=z0, ux=ux0, uy=uy0, uz=uz0, w=w0)
    sp._id[:n_p] = sp.xp.asarray(ids0, dtype=np.int64)
    ps = ParticleSolver(
        solver,
        species=[sp],
        pusher="boris",
        shape=1,
        gather="galerkin",
        B_ext=B_ext,
        sort_every=50,
        particles_comm=particles_comm,
    )

    ps.attach()
    for _ in range(n_steps):
        solver.one_step()

    migrations = (
        int(
            solver.comm.allreduce(
                ps._n_migrated_lo + ps._n_migrated_hi, op=MPI.SUM
            )
        )
        if use_mpi
        else 0
    )
    gauss_res = ps.gauss_residual()
    q_tot = ps.total_charge()
    ke_tot = ps.kinetic_energy(synchronised=True)
    ue_tot = solver.field_energy()
    E_full = solver.mpi_gather_asField("E") if use_mpi else solver.E
    H_full = solver.mpi_gather_asField("H") if use_mpi else solver.H
    sp_full = ps.mpi_gather_species(0) if use_mpi else sp.copy().to_cpu()
    dz_all = (
        np.diff(grid.Z_nodes)
        if getattr(grid, "Z_nodes", None) is not None
        else grid.dz
    )
    return {
        "gauss_res": gauss_res,
        "q_tot": q_tot,
        "ke_tot": ke_tot,
        "ue_tot": ue_tot,
        "E": E_full,
        "H": H_full,
        "sp": sp_full,
        "migrations": migrations,
        "sent_lo": ps._n_migrated_lo,
        "sent_hi": ps._n_migrated_hi,
        "L_z_min": float(np.min(dz_all)),
        "particles_comm": ps.particles_comm,
        "nccl_comm": solver._nccl_comm,
    }


@pytest.mark.parametrize("graded", [False, True])
def test_v2_mpi_vs_serial_and_gauss_500_steps(graded):
    """V2/V8: 500-step coupled run on uniform and graded z meshes matches serial to round-off.

    Physical setup
    --------------
    48 relativistic macroparticles gyrate in a transverse magnetic field across
    internal ``z``-slab interfaces for ``500`` time steps on an ``8 x 8 x 12``
    mesh (both uniform and quadratically graded in ``z``). Verifies that:
    - particles cross internal slab boundaries ``>= 100`` times over the run,
    - the global gathered Gauss residual stays ``<= 1e-12``,
    - gathered ``E``, ``H``, total charge, kinetic/field energy, and the
      ``id``-sorted particle phase-space coordinates match the serial run
      within the 3-seed perturbation-amplification round-off bound
      ``derived_mpi_roundoff_bound(ser, pert_runs)``.
    """
    comm = MPI.COMM_WORLD
    n_steps = 500

    par = _run_v2_coupled(graded=graded, use_mpi=True, n_steps=n_steps)
    assert par["migrations"] >= 100
    assert par["gauss_res"] <= 1.0e-12
    _assert_interior_bidirectional_migration(par["sent_lo"], par["sent_hi"], comm)

    if comm.Get_rank() != 0:
        return

    ser, pert_runs = _run_serial_perturbation_envelope(
        _run_v2_coupled, graded=graded, use_mpi=False, n_steps=n_steps
    )
    assert ser["gauss_res"] <= 1.0e-12
    bound = derived_mpi_roundoff_bound(ser, pert_runs)
    ratio = _report_roundoff_comparison(
        f"v2_cpu_mpi_vs_serial(graded={graded})", ser, par, bound
    )
    assert ratio <= 1.0

    _assert_v2_matches_reference(par, ser, bound=bound)


def _run_v3_cold_plasma(
    use_mpi: bool,
    n_steps: int = 240,
    use_gpu: bool = False,
    particles_comm: str | None = None,
    reverse_particles: bool = False,
):
    """Run a 1D-in-3D standing cold-plasma oscillation across MPI z-slabs."""
    Nx, Ny, Nz = 2, 2, 32
    Lx, Ly, Lz = 0.02, 0.02, 0.32
    dz = Lz / Nz
    # Standing mode m_z = 3 has sin(k_mode * z) = 0 at z = 0, Lz and near-maximal
    # velocity across internal 2-rank and 4-rank slab faces (z = Lz/4, Lz/2, 3*Lz/4).
    k_mode = 3.0 * np.pi / Lz

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
        use_mpi=use_mpi,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        cfln=0.5,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_mpi=use_mpi,
        use_gpu=use_gpu,
        verbose=0,
    )

    omega_p0 = 2.0 * np.pi * 1.2e9
    n0 = eps_0 * m_e * (omega_p0**2) / (e_charge**2)
    vol = Lx * Ly * Lz

    n_ppc_z = 8
    n_z_tot = Nz * n_ppc_z
    # Place the symmetric particle pairs straddling z = Lz/4, Lz/2, 3*Lz/4 at
    # z_face +/- 15 um (preserving each pair's centroid at z_face while sitting well
    # inside the ~48 um plasma oscillation excursion) so particles cross every
    # internal 2-rank and 4-rank slab face in both directions each period.
    z_1d = (np.arange(n_z_tot) + 0.5) * (Lz / n_z_tot)
    for idx_lo, z_face in (
        (n_z_tot // 4 - 1, 0.25 * Lz),
        (n_z_tot // 2 - 1, 0.50 * Lz),
        (3 * n_z_tot // 4 - 1, 0.75 * Lz),
    ):
        z_1d[idx_lo] = z_face - 1.5e-5
        z_1d[idx_lo + 1] = z_face + 1.5e-5
    x_c = (np.arange(Nx) + 0.5) * (Lx / Nx)
    y_c = (np.arange(Ny) + 0.5) * (Ly / Ny)
    xx, yy, zz = np.meshgrid(x_c, y_c, z_1d, indexing="ij")
    x_p, y_p, z_p = xx.ravel(), yy.ravel(), zz.ravel()

    w_macro = (n0 * vol) / x_p.size
    delta_v = 2.0e-3 * (omega_p0 / k_mode)
    uz_p = delta_v * np.sin(k_mode * z_p)
    n_tot = x_p.size
    ids_p = np.arange(n_tot, dtype=np.int64)
    if reverse_particles:
        x_p = x_p[::-1].copy()
        y_p = y_p[::-1].copy()
        z_p = z_p[::-1].copy()
        uz_p = uz_p[::-1].copy()
        ids_p = ids_p[::-1].copy()

    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    sp.add_particles(x=x_p, y=y_p, z=z_p, ux=0.0, uy=0.0, uz=uz_p, w=w_macro)
    sp._id[:n_tot] = sp.xp.asarray(ids_p, dtype=np.int64)
    ps = ParticleSolver(
        solver,
        species=[sp],
        pusher="boris",
        shape=1,
        gather="galerkin",
        sort_every=0,
        particles_comm=particles_comm,
    ).attach()

    ke_hist = np.zeros(n_steps + 1)
    ue_hist = np.zeros(n_steps + 1)
    ke_hist[0] = ps.kinetic_energy(synchronised=True)
    ue_hist[0] = solver.field_energy()

    for step in range(1, n_steps + 1):
        solver.one_step()
        ke_hist[step] = ps.kinetic_energy(synchronised=True)
        ue_hist[step] = solver.field_energy()

    E_full = solver.mpi_gather_asField("E") if use_mpi else solver.E
    H_full = solver.mpi_gather_asField("H") if use_mpi else solver.H
    sp_full = ps.mpi_gather_species(0) if use_mpi else sp.copy().to_cpu()
    omega_pred = cold_plasma_dispersion_discrete(
        omega_p0=omega_p0,
        dt=solver.dt,
        dz=dz,
        k=k_mode,
        order=1,
        gather="galerkin",
    )
    return {
        "gauss_res": ps.gauss_residual(),
        "E": E_full,
        "H": H_full,
        "sp": sp_full,
        "ke": ke_hist,
        "ue": ue_hist,
        "omega_pred": omega_pred,
        "dt": float(solver.dt),
        "dz": dz,
        "sent_lo": ps._n_migrated_lo,
        "sent_hi": ps._n_migrated_hi,
        "particles_comm": ps.particles_comm,
        "nccl_comm": solver._nccl_comm,
    }


def test_v3_cold_plasma_mpi_vs_serial():
    """V3/V8: Cold plasma oscillation across MPI z-slabs matches serial and conserves energy."""
    comm = MPI.COMM_WORLD
    n_steps = 240

    par = _run_v3_cold_plasma(use_mpi=True, n_steps=n_steps)
    assert par["gauss_res"] <= 1.0e-12
    _assert_interior_bidirectional_migration(par["sent_lo"], par["sent_hi"], comm)

    # Derivation of the staggered leapfrog energy oscillation bound:
    # For a harmonic plasma mode zeta^n = A cos(omega * t^n), averaging adjacent
    # half-step velocities gives v^n = (v^{n+1/2} + v^{n-1/2})/2 = -A omega cos(omega dt / 2) sin(omega t^n),
    # so the sum of synchronised kinetic and electrostatic energy is proportional to
    # cos^2(omega t^n) + cos^2(omega dt / 2) sin^2(omega t^n) = 1 - sin^2(omega dt / 2) sin^2(omega t^n)
    # = 1 - (1/4) (omega dt)^2 sin^2(omega t^n) + O((omega dt)^4) (Birdsall & Langdon, 1991, Sec. 4-2).
    # Multiplying the leading-order amplitude 0.25 * (omega_pred * dt)^2 by a safety factor of 2.0
    # for finite-grid CIC spline harmonic terms gives the same 0.5 * (omega_pred * dt)^2 bound
    # used in test_079_particles_core.py.
    e_tot = par["ke"] + par["ue"]
    rel_e_drift = float(np.max(np.abs(e_tot - e_tot[0])) / e_tot[0])
    safety_factor_energy = 2.0
    leapfrog_energy_bound = (
        safety_factor_energy * 0.25 * ((par["omega_pred"] * par["dt"]) ** 2)
    )
    assert rel_e_drift <= leapfrog_energy_bound

    if comm.Get_rank() != 0:
        return

    ser, pert_runs = _run_serial_perturbation_envelope(
        _run_v3_cold_plasma, use_mpi=False, n_steps=n_steps
    )
    assert ser["gauss_res"] <= 1.0e-12
    bound = derived_mpi_roundoff_bound(ser, pert_runs)
    ratio = _report_roundoff_comparison("v3_cpu_mpi_vs_serial", ser, par, bound)
    assert ratio <= 1.0

    e_tot_ser = ser["ke"] + ser["ue"]
    ser_e_drift = float(np.max(np.abs(e_tot_ser - e_tot_ser[0])) / e_tot_ser[0])
    assert abs(rel_e_drift - ser_e_drift) <= bound

    ez_par = par["E"][:, :, :, "z"]
    ez_ser = ser["E"][:, :, :, "z"]
    ez_norm = float(np.max(np.abs(ez_ser)))
    assert ez_norm > 0.0
    assert float(np.max(np.abs(ez_par - ez_ser))) / ez_norm <= bound

    sp_par, sp_ser = par["sp"], ser["sp"]
    assert sp_par.n_particles == sp_ser.n_particles
    order_s = np.argsort(sp_ser.id)
    np.testing.assert_array_equal(sp_par.id, sp_ser.id[order_s])
    assert (
        float(np.max(np.abs(sp_par.z - sp_ser.z[order_s]))) / par["dz"] <= bound
    )
    assert (
        float(np.max(np.abs(sp_par.uz - sp_ser.uz[order_s])))
        / float(np.max(np.abs(sp_ser.uz)))
        <= bound
    )
    np.testing.assert_allclose(
        par["ke"], ser["ke"], rtol=bound, atol=bound * float(e_tot_ser[0])
    )
    np.testing.assert_allclose(
        par["ue"], ser["ue"], rtol=bound, atol=bound * float(e_tot_ser[0])
    )


def test_slab_face_and_nextafter_migration_conservation():
    """Particles placed right on internal slab faces and at ``nextafter(z_hi, -inf)`` never duplicate or vanish.

    Physical setup
    --------------
    Places pairs of particles on every internal ``z``-slab interface
    ``z_face = k * Lz / size`` (one moving in ``+z``, one in ``-z``) and at
    ``np.nextafter(z_face, -np.inf)`` and ``np.nextafter(z_face, +np.inf)``.
    Verifies that half-open ownership ``[z_lo, z_hi)`` assigns each particle to
    a single rank initially and preserves the exact particle count, unique IDs,
    total charge, and ``gauss_residual() <= 1e-12`` at every step.
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    Lz = 0.12
    grid = GridFIT3D(
        0.0, 0.06, 0.0, 0.06, 0.0, Lz, 6, 6, 12, use_mpi=True, verbose=0
    )
    solver = SolverFIT3D(
        grid,
        cfln=0.5,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_mpi=True,
        verbose=0,
    )

    z_list = []
    uz_list = []
    for r in range(1, size):
        z_face = r * (Lz / size)
        z_minus = float(np.nextafter(z_face, -np.inf))
        z_plus = float(np.nextafter(z_face, np.inf))
        for z_val in (z_face, z_minus, z_plus):
            z_list.extend([z_val, z_val])
            uz_list.extend([0.25 * c, -0.25 * c])

    z0 = np.asarray(z_list, dtype=np.float64)
    uz0 = np.asarray(uz_list, dtype=np.float64)
    n_expected = z0.size
    x0 = np.linspace(0.015, 0.045, n_expected)
    y0 = np.linspace(0.045, 0.015, n_expected)
    ux0 = np.full(n_expected, 0.05 * c)
    uy0 = np.full(n_expected, -0.05 * c)
    w0 = np.linspace(1.0e6, 2.0e6, n_expected)
    q_expected = float(-e_charge * np.sum(w0))

    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    sp.add_particles(x=x0, y=y0, z=z0, ux=ux0, uy=uy0, uz=uz0, w=w0)
    ps = ParticleSolver(solver, species=[sp], shape=1, sort_every=5).attach()

    # Initial ownership check before any step
    n_init = int(comm.allreduce(sp.n_particles, op=MPI.SUM))
    assert n_init == n_expected
    assert np.isclose(ps.total_charge(), q_expected, rtol=1e-15, atol=0.0)

    for _ in range(25):
        solver.one_step()
        n_now = int(comm.allreduce(sp.n_particles, op=MPI.SUM))
        assert n_now == n_expected
        assert np.isclose(ps.total_charge(), q_expected, rtol=1e-15, atol=0.0)
        assert ps.gauss_residual() <= 1.0e-12
        # Every rank's particles must strictly satisfy the ownership invariant
        if sp.n_particles > 0:
            assert bool(np.all(sp.z >= ps._z_own_min))
            assert bool(np.all(sp.z < ps._z_own_max))

    g_sp = ps.mpi_gather_species(0)
    if comm.Get_rank() == 0:
        np.testing.assert_array_equal(
            g_sp.id, np.arange(n_expected, dtype=np.int64)
        )


def test_mpi_injection_and_collector_absorption_vs_serial():
    """Per-rank BeamInjector and Collector/PEC absorption reductions match serial."""
    comm = MPI.COMM_WORLD

    def _run_inj_abs(use_mpi: bool):
        grid = GridFIT3D(
            0.0, 0.06, 0.0, 0.06, 0.0, 0.12, 6, 6, 12, use_mpi=use_mpi, verbose=0
        )
        solver = SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            use_mpi=use_mpi,
            verbose=0,
        )
        sp = Species(name="electron", charge=-e_charge, mass=m_e)
        col = Collector(name="mid_screen", plane=("z", 0.095, +1))

        inj_low = BeamInjector(
            species=sp,
            z0=0.015,
            current=0.05,
            gamma=1.15,
            x0=0.03,
            y0=0.03,
            radius=0.004,
            n_macro_per_step=4,
            direction=-1,
            seed=101,
        )
        inj_high = BeamInjector(
            species=sp,
            z0=0.055,
            current=0.08,
            gamma=1.15,
            x0=0.03,
            y0=0.03,
            radius=0.004,
            n_macro_per_step=4,
            direction=1,
            seed=202,
        )
        ps = ParticleSolver(
            solver,
            species=[sp],
            injectors=[inj_low, inj_high],
            collectors=[col],
            shape=1,
            sort_every=0,
        ).attach()

        for _ in range(30):
            solver.one_step()

        return {
            "q_inj": ps.injected_charge(),
            "e_inj": ps.injected_energy(),
            "q_abs_tot": ps.absorbed_charge(),
            "q_abs_z_low": ps.absorbed_charge("z_low"),
            "q_abs_col": ps.absorbed_charge("mid_screen"),
            "e_abs_tot": ps.absorbed_energy(),
            "q_active": ps.total_charge(),
            "gauss_res": ps.gauss_residual(),
        }

    par = _run_inj_abs(use_mpi=True)
    # Charge conservation: injected == active + absorbed
    assert np.isclose(
        par["q_inj"], par["q_active"] + par["q_abs_tot"], rtol=1e-14, atol=0.0
    )
    assert abs(par["q_abs_z_low"]) > 0.0
    assert abs(par["q_abs_col"]) > 0.0

    if comm.Get_rank() != 0:
        return

    ser = _run_inj_abs(use_mpi=False)
    for key in (
        "q_inj",
        "e_inj",
        "q_abs_tot",
        "q_abs_z_low",
        "q_abs_col",
        "e_abs_tot",
        "q_active",
    ):
        assert np.isclose(par[key], ser[key], rtol=1e-12, atol=0.0), key


@pytest.mark.parametrize(
    "mutation",
    [
        "overwrite_guards",
        "zero_second_high_guard",
        "skip_low_guard",
        "drop_migrating",
        "migrate_before_push",
    ],
)
def test_mpi_mutations_fail(mutation, monkeypatch):
    """Required mutations on the MPI code path must fail validation.

    Mutations
    ---------
    - ``overwrite_guards``: monkeypatches ``ParticleSolver._accumulate_guard_plane``
      to overwrite ``target[...] = guard`` instead of adding ``target += guard``.
    - ``zero_second_high_guard``: monkeypatches ``ParticleSolver._pack_second_high_guard_plane``
      to zero ``send_hi[:, 1, :] = 0`` (dropping the second high guard plane ``k = k_own_end + 1``).
    - ``skip_low_guard``: monkeypatches ``ParticleSolver._accumulate_low_guard_plane``
      to a no-op (skipping addition of the neighbour's low guard plane ``k = 0``).
    - ``drop_migrating``: monkeypatches ``ParticleSolver._mpi_migrate_particles``
      to filter out leaving particles without receiving incoming particles.
    - ``migrate_before_push``: monkeypatches ``ParticleSolver._mpi_pre_push_hook``
      and ``_mpi_post_push_hook`` so migration executes before rather than after
      the particle push and Esirkepov current deposition.
    """
    grid = GridFIT3D(
        0.0, 0.06, 0.0, 0.06, 0.0, 0.12, 6, 6, 12, use_mpi=True, verbose=0
    )
    solver = SolverFIT3D(
        grid,
        cfln=0.5,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_mpi=True,
        verbose=0,
    )
    z0 = np.array([0.02985, 0.03015, 0.05985, 0.06015, 0.08985, 0.09015])
    x0 = np.full_like(z0, 0.03)
    y0 = np.full_like(z0, 0.03)
    ux0 = np.full_like(z0, 0.25 * c)
    uy0 = np.full_like(z0, -0.25 * c)
    uz0 = np.array([0.35 * c, -0.35 * c, 0.35 * c, -0.35 * c, 0.35 * c, -0.35 * c])
    w0 = np.full_like(z0, 1.0e7)
    q_init = float(-e_charge * np.sum(w0))

    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    sp.add_particles(x=x0, y=y0, z=z0, ux=ux0, uy=uy0, uz=uz0, w=w0)
    ps = ParticleSolver(solver, species=[sp], shape=1, sort_every=0)

    if mutation == "overwrite_guards":
        monkeypatch.setattr(
            ps,
            "_accumulate_guard_plane",
            lambda target_slice, guard_slice: target_slice.__setitem__(
                Ellipsis, guard_slice
            ),
        )
    elif mutation == "zero_second_high_guard":
        monkeypatch.setattr(
            ps,
            "_pack_second_high_guard_plane",
            lambda send_hi, N, nxy: send_hi[:, 1, :].fill(0.0),
        )
    elif mutation == "skip_low_guard":
        monkeypatch.setattr(
            ps,
            "_accumulate_low_guard_plane",
            lambda target_slice, guard_slice: None,
        )
    elif mutation == "drop_migrating":
        monkeypatch.setattr(
            ps,
            "_mpi_migrate_particles",
            lambda sp_i: sp_i.keep(ps._owned_z_mask(sp_i.z)),
        )
    elif mutation == "migrate_before_push":
        monkeypatch.setattr(ps, "_mpi_pre_push_hook", ps._mpi_migrate_particles)
        monkeypatch.setattr(ps, "_mpi_post_push_hook", lambda sp_i: None)

    ps.attach()
    solver.one_step()

    res = ps.gauss_residual()
    q_now = ps.total_charge()

    # Every mutation must violate discrete Gauss law by > 10 orders of magnitude
    assert res > 1.0e-3
    if mutation == "drop_migrating":
        assert abs(q_now) < abs(q_init)
    if mutation == "migrate_before_push":
        # Leaving particles remain unmigrated on their old rank at step end
        out_of_bounds = int(
            solver.comm.allreduce(
                int(np.sum(~ps._owned_z_mask(sp.z))), op=MPI.SUM
            )
        )
        assert out_of_bounds > 0


def test_mpi_minimal_nz_two_per_rank_and_validation():
    """Minimal valid z-slab ``NZ = 2 * size`` conserves charge and matches serial; ``NZ // size < 2`` raises."""
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    # 1. Validation checks under real MPI communicator:
    #    - Nz = size (1 owned z-cell per rank) must raise ValueError
    #    - Nz = 2 * size + 1 (not divisible by size) must raise ValueError
    #    - filter="binomial" with use_mpi=True must raise NotImplementedError
    grid_one_cell = GridFIT3D(
        0.0, 0.04, 0.0, 0.04, 0.0, 0.04 * size, 4, 4, size, use_mpi=True, verbose=0
    )
    solver_one_cell = SolverFIT3D(
        grid_one_cell,
        cfln=0.5,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_mpi=True,
        verbose=0,
    )
    with pytest.raises(ValueError, match="at least 2 owned z-cells"):
        ParticleSolver(solver_one_cell, species=[Species()], shape=1)

    grid_non_div = GridFIT3D(
        0.0,
        0.04,
        0.0,
        0.04,
        0.0,
        0.04 * (2 * size + 1),
        4,
        4,
        2 * size + 1,
        use_mpi=True,
        verbose=0,
    )
    solver_non_div = SolverFIT3D(
        grid_non_div,
        cfln=0.5,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_mpi=True,
        verbose=0,
    )
    with pytest.raises(ValueError, match="divisible by solver.size"):
        ParticleSolver(solver_non_div, species=[Species()], shape=1)

    grid_min = GridFIT3D(
        0.0,
        0.04,
        0.0,
        0.04,
        0.0,
        0.02 * (2 * size),
        4,
        4,
        2 * size,
        use_mpi=True,
        verbose=0,
    )
    solver_min_chk = SolverFIT3D(
        grid_min,
        cfln=0.5,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_mpi=True,
        verbose=0,
    )
    with pytest.raises(NotImplementedError, match="Binomial current filtering"):
        ParticleSolver(
            solver_min_chk, species=[Species()], shape=1, filter="binomial"
        )

    # 2. Minimal valid case NZ = 2 * size: run coupled steps with particles
    #    crossing every internal slab interface in both +z and -z with nonzero
    #    transverse velocity so both high guard planes (k_own_end, k_own_end+1)
    #    and the low guard plane (k=0) are exercised.
    Nz_tot = 2 * size
    Lz = 0.01 * Nz_tot
    n_steps = 30
    rng = np.random.default_rng(202610)
    n_p = 16
    x0 = rng.uniform(0.005, 0.035, n_p)
    y0 = rng.uniform(0.005, 0.035, n_p)
    z0 = rng.uniform(0.25 * Lz, 0.75 * Lz, n_p)
    ux0 = rng.uniform(-0.18, 0.18, n_p) * c
    uy0 = rng.uniform(-0.18, 0.18, n_p) * c
    uz0 = rng.uniform(-0.28, 0.28, n_p) * c
    w0 = np.full(n_p, 4.0e6)

    def _run_min(use_mpi: bool, rev: bool = False):
        g = GridFIT3D(
            0.0,
            0.04,
            0.0,
            0.04,
            0.0,
            Lz,
            4,
            4,
            Nz_tot,
            use_mpi=use_mpi,
            verbose=0,
        )
        s = SolverFIT3D(
            g,
            cfln=0.45,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            use_mpi=use_mpi,
            verbose=0,
        )
        sp = Species(name="electron", charge=-e_charge, mass=m_e)
        ids = np.arange(n_p, dtype=np.int64)
        if rev:
            sp.add_particles(
                x=x0[::-1],
                y=y0[::-1],
                z=z0[::-1],
                ux=ux0[::-1],
                uy=uy0[::-1],
                uz=uz0[::-1],
                w=w0[::-1],
            )
            sp._id[:n_p] = ids[::-1]
        else:
            sp.add_particles(x=x0, y=y0, z=z0, ux=ux0, uy=uy0, uz=uz0, w=w0)
        ps = ParticleSolver(
            s,
            species=[sp],
            B_ext=(0.06, -0.05, 0.0),
            shape=1,
            sort_every=0,
        ).attach()
        for _ in range(n_steps):
            s.one_step()
        return {
            "gauss_res": ps.gauss_residual(),
            "q_tot": ps.total_charge(),
            "ke_tot": ps.kinetic_energy(synchronised=True),
            "ue_tot": s.field_energy(),
            "E": s.mpi_gather_asField("E") if use_mpi else s.E,
            "H": s.mpi_gather_asField("H") if use_mpi else s.H,
            "sp": ps.mpi_gather_species(0) if use_mpi else sp.copy(),
            "L_z_min": 0.01,
        }

    par = _run_min(use_mpi=True)
    assert par["gauss_res"] <= 1.0e-12
    if comm.Get_rank() != 0:
        return
    ser, pert_runs = _run_serial_perturbation_envelope(_run_min, use_mpi=False)
    assert ser["gauss_res"] <= 1.0e-12
    bound = derived_mpi_roundoff_bound(ser, pert_runs)
    ratio = _report_roundoff_comparison("v2_min_nz_mpi_vs_serial", ser, par, bound)
    assert ratio <= 1.0
    assert par["sp"].n_particles == ser["sp"].n_particles == n_p
    for c_name in "xyz":
        e_par = _field_comp_host(par["E"], c_name)
        e_ser = _field_comp_host(ser["E"], c_name)
        e_norm = max(
            float(np.max(np.abs(_field_comp_host(ser["E"], cn)))) for cn in "xyz"
        )
        assert float(np.max(np.abs(e_par - e_ser))) / e_norm <= bound


def test_mpi_gather_as_grid_graded():
    """``grid.mpi_gather_asGrid()`` preserves custom graded node coordinates on rank 0."""
    comm = MPI.COMM_WORLD
    grid_mpi = _make_v2_grid(graded=True, use_mpi=True)
    grid_gathered = grid_mpi.mpi_gather_asGrid()
    if comm.Get_rank() != 0:
        return
    grid_ser = _make_v2_grid(graded=True, use_mpi=False)
    assert grid_gathered._custom_xyz is True
    np.testing.assert_allclose(grid_gathered.x, grid_ser.x, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(grid_gathered.y, grid_ser.y, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(grid_gathered.z, grid_ser.z, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(grid_gathered.dz, grid_ser.dz, rtol=0.0, atol=1e-15)
    # Confirm the grid is genuinely non-uniform (dz varies by > 30% from zmin to zmax)
    assert float(np.max(grid_gathered.dz) / np.min(grid_gathered.dz)) > 1.3


def test_mpi_single_rank_cfl_error_raises_on_all_ranks_and_nccl_collective_fallback(
    monkeypatch,
):
    """A single-rank CFL/multi-hop or OOM error raises on ALL ranks without hanging; NCCL availability is decided collectively."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Re-budget T = 20 s for the collective error/OOM/NCCL blocks covered here
    with _hang_guard(20):
        # 1. CPU multi-hop violation on rank 0 only must raise ParticleCFLError on every rank
        grid = GridFIT3D(
            0.0, 0.06, 0.0, 0.06, 0.0, 0.12, 6, 6, 12, use_mpi=True, verbose=0
        )
        solver = SolverFIT3D(
            grid,
            cfln=0.45,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            use_mpi=True,
            verbose=0,
        )
        sp = Species(name="electron", charge=-e_charge, mass=m_e)
        sp.add_particles(
            x=np.array([0.03]),
            y=np.array([0.03]),
            z=np.array([0.5 * (grid.zmin + grid.zmax)]),
            ux=np.array([0.0]),
            uy=np.array([0.0]),
            uz=np.array([0.01 * c]),
            w=np.array([1.0e6]),
        )
        ps = ParticleSolver(
            solver, species=[sp], shape=1, deposit_scheme="naive"
        ).attach()
        # Verify CPU _mpi_sum_guards preallocates and reuses guard buffers across steps
        # and reallocates only when (nxy, dtype) changes.
        ps._mpi_sum_guards(ps.J_particles)
        buf_ids_first = (
            id(ps._cpu_guard_send_lo),
            id(ps._cpu_guard_recv_lo),
            id(ps._cpu_guard_send_hi),
            id(ps._cpu_guard_recv_hi),
        )
        ps._mpi_sum_guards(ps.J_particles)
        buf_ids_second = (
            id(ps._cpu_guard_send_lo),
            id(ps._cpu_guard_recv_lo),
            id(ps._cpu_guard_send_hi),
            id(ps._cpu_guard_recv_hi),
        )
        assert buf_ids_first == buf_ids_second
        ps._ensure_cpu_mpi_guard_buffers(
            int(solver.Nx * solver.Ny) + 4, np.float64
        )
        assert id(ps._cpu_guard_send_lo) != buf_ids_first[0]
        assert ps._cpu_guard_send_lo.shape == (3, int(solver.Nx * solver.Ny) + 4)

        # Corrupt z on rank 0 only right before migration via a post-push wrapper
        orig_mig = ps._mpi_migrate_particles

        def _corrupt_rank0_and_migrate(sp_i):
            if rank == 0 and sp_i.n_particles > 0:
                sp_i._z[0] = ps._z_ghost_hi + 0.01
            return orig_mig(sp_i)

        monkeypatch.setattr(ps, "_mpi_migrate_particles", _corrupt_rank0_and_migrate)
        with pytest.raises(ParticleCFLError, match="more than 1 cell"):
            solver.one_step()

        # 2. CPU single-cell Esirkepov CFL violation on rank 0 only must raise
        #    ParticleCFLError on every rank and chain __cause__ on the failing rank
        grid2 = GridFIT3D(
            0.0, 0.06, 0.0, 0.06, 0.0, 0.12, 6, 6, 12, use_mpi=True, verbose=0
        )
        solver2 = SolverFIT3D(
            grid2,
            cfln=0.45,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            use_mpi=True,
            verbose=0,
        )
        sp2 = Species(name="electron", charge=-e_charge, mass=m_e)
        sp2.add_particles(
            x=np.array([0.01]),
            y=np.array([0.03]),
            z=np.array([0.5 * (grid2.zmin + grid2.zmax)]),
            ux=np.array([0.0]),
            uy=np.array([0.0]),
            uz=np.array([0.0]),
            w=np.array([1.0e6]),
        )
        ps2 = ParticleSolver(
            solver2, species=[sp2], shape=1, deposit_scheme="esirkepov"
        ).attach()
        orig_prep = ps2.boundaries.prepare_step_paths

        def _corrupt_xi1_on_rank0(
            sp_i, x0, y0, z0, xi0, eta0, zeta0, xi1, eta1, zeta1, t=0.0
        ):
            if rank == 0 and xi1.size > 0:
                xi1 = xi1.copy()
                xi1[0] = xi0[0] + 2.5  # |b1 - b0| >= 2 on rank 0 only
            return orig_prep(
                sp_i, x0, y0, z0, xi0, eta0, zeta0, xi1, eta1, zeta1, t=t
            )

        with monkeypatch.context() as m:
            m.setattr(
                ps2.boundaries, "prepare_step_paths", _corrupt_xi1_on_rank0
            )
            with pytest.raises(
                ParticleCFLError, match="crossed more than 1 cell|more than 1 cell"
            ) as cfl_exc_info:
                solver2.one_step()
            assert type(cfl_exc_info.value) is ParticleCFLError
            if rank == 0:
                assert type(cfl_exc_info.value.__cause__) is ParticleCFLError
            else:
                assert cfl_exc_info.value.__cause__ is None

        # 2b. Non-CFL ValueError on rank 0 during a step must re-raise ValueError
        #     (NOT RuntimeError and NOT ParticleCFLError) on all ranks and chain __cause__
        def _raise_val_err_rank0(_sp_i):
            if rank == 0:
                raise ValueError("Simulated non-CFL ValueError on rank 0")

        with monkeypatch.context() as m:
            m.setattr(ps2, "_mpi_pre_push_hook", _raise_val_err_rank0)
            with pytest.raises(ValueError, match="non-CFL ValueError|peer MPI rank") as val_exc_info:
                solver2.one_step()
            assert type(val_exc_info.value) is ValueError
            if rank == 0:
                assert type(val_exc_info.value.__cause__) is ValueError
            else:
                assert val_exc_info.value.__cause__ is None

        # 2c. Mixed failures across ranks: rank 0 has a ParticleCFLError (code 1)
        #     and rank 1 has a non-CFL ValueError (code 2). Every rank must raise
        #     the uniform class implied by max_code == 2 (exact type ValueError,
        #     NOT ParticleCFLError on rank 0), with __cause__ set on failing ranks.
        def _raise_mixed_cfl_and_val(_sp_i):
            if rank == 0:
                raise ParticleCFLError("Simulated CFL error on rank 0")
            if rank == 1:
                raise ValueError("Simulated non-CFL ValueError on rank 1")

        with monkeypatch.context() as m:
            m.setattr(ps2, "_mpi_pre_push_hook", _raise_mixed_cfl_and_val)
            with pytest.raises(ValueError) as mix_exc_info:
                solver2.one_step()
            assert type(mix_exc_info.value) is ValueError
            if rank == 0:
                assert type(mix_exc_info.value.__cause__) is ParticleCFLError
            elif rank == 1:
                assert type(mix_exc_info.value.__cause__) is ValueError
            else:
                assert mix_exc_info.value.__cause__ is None

        # Also verify mixed CFL + ValueError in _collective_check
        with pytest.raises(ValueError) as mix_cc_info:
            ps2._collective_check(
                ParticleCFLError("rank-0 CFL")
                if rank == 0
                else (ValueError("rank-1 ValueError") if rank == 1 else None)
            )
        assert type(mix_cc_info.value) is ValueError
        if rank == 0:
            assert type(mix_cc_info.value.__cause__) is ParticleCFLError
        elif rank == 1:
            assert type(mix_cc_info.value.__cause__) is ValueError
        else:
            assert mix_cc_info.value.__cause__ is None

        # 2d. Local TypeError (code 3 -> max_code == 3): every rank must raise
        #     RuntimeError and the failing rank must preserve TypeError in __cause__.
        def _raise_type_err_rank0(_sp_i):
            if rank == 0:
                raise TypeError("Simulated TypeError on rank 0")

        with monkeypatch.context() as m:
            m.setattr(ps2, "_mpi_pre_push_hook", _raise_type_err_rank0)
            with pytest.raises(RuntimeError) as type_exc_info:
                solver2.one_step()
            assert type(type_exc_info.value) is RuntimeError
            if rank == 0:
                assert type(type_exc_info.value.__cause__) is TypeError
            else:
                assert type_exc_info.value.__cause__ is None

        with pytest.raises(RuntimeError) as type_cc_info:
            ps2._collective_check(
                TypeError("Simulated TypeError in _collective_check")
                if rank == 0
                else None
            )
        assert type(type_cc_info.value) is RuntimeError
        if rank == 0:
            assert type(type_cc_info.value.__cause__) is TypeError
        else:
            assert type_cc_info.value.__cause__ is None

        # 3. Collective _ensure_nccl_comm decision: if one rank fails during
        #    NCCL communicator setup, all ranks must collectively return None
        #    and fall back to pinned-host MPI without hanging.
        solver2._nccl_comm = None
        solver2._nccl_unavailable = False
        solver2.use_gpu = True
        import cupy as cp
        from cupy.cuda import nccl

        orig_get_uid = nccl.get_unique_id

        def _fail_on_rank0():
            if rank == 0:
                raise RuntimeError("Simulated rank-0 NCCL get_unique_id failure")
            return orig_get_uid()

        with monkeypatch.context() as m:
            m.setattr(SolverFIT3D, "_nccl_comm_cache", {}, raising=False)
            m.setattr(
                cp.cuda.runtime,
                "getDeviceCount",
                lambda: max(2, comm.Get_size()),
            )
            m.setattr(
                cp.cuda,
                "Device",
                lambda dev_id: type("_DummyDev", (), {"use": lambda self: None})(),
            )
            m.setattr(nccl, "get_unique_id", _fail_on_rank0)
            res_comm = solver2._ensure_nccl_comm()
            assert res_comm is None
            assert solver2._nccl_unavailable is True

        # Also test failure on rank > 0 during device selection and NcclCommunicator construction
        solver2._nccl_comm = None
        solver2._nccl_unavailable = False

        def _fail_device_on_last_rank(dev_id):
            if rank == comm.Get_size() - 1:
                raise RuntimeError("Simulated last-rank CUDA device failure")
            return type("_DummyDev", (), {"use": lambda self: None})()

        with monkeypatch.context() as m:
            m.setattr(SolverFIT3D, "_nccl_comm_cache", {}, raising=False)
            m.setattr(
                cp.cuda.runtime,
                "getDeviceCount",
                lambda: max(2, comm.Get_size()),
            )
            m.setattr(cp.cuda, "Device", _fail_device_on_last_rank)
            res_comm_dev = solver2._ensure_nccl_comm()
            assert res_comm_dev is None
            assert solver2._nccl_unavailable is True

        solver2._nccl_comm = None
        solver2._nccl_unavailable = False

        # Verify _abort_world helper writes to stderr and calls comm.Abort(1)
        import wakis.solverFIT3D as sfit_mod
        from wakis.solverFIT3D import _abort_world

        class _DummyAbortComm:
            def __init__(self):
                self.aborted_code = None

            def Abort(self, code):
                self.aborted_code = code

        dummy_abort_comm = _DummyAbortComm()
        _abort_world(dummy_abort_comm, "unit-test abort check")
        assert dummy_abort_comm.aborted_code == 1

        # NcclCommunicator (ncclCommInitRank) is a blocking NCCL collective;
        # if it raises on rank 1 only, the failing rank must invoke _abort_world
        # rather than waiting in comm.allreduce while healthy ranks block in NCCL.
        class _AbortSentinel(RuntimeError):
            pass

        abort_calls = []

        def _sentinel_abort_world(c_obj, msg):
            abort_calls.append((c_obj, msg))
            raise _AbortSentinel(msg)

        def _fail_on_rank1(*args, **kwargs):
            if rank == 1:
                raise RuntimeError("Simulated rank-1 NcclCommunicator failure")
            return object()

        with monkeypatch.context() as m:
            m.setattr(SolverFIT3D, "_nccl_comm_cache", {}, raising=False)
            m.setattr(
                cp.cuda.runtime,
                "getDeviceCount",
                lambda: max(2, comm.Get_size()),
            )
            m.setattr(
                cp.cuda,
                "Device",
                lambda dev_id: type("_DummyDev", (), {"use": lambda self: None})(),
            )
            m.setattr(nccl, "NcclCommunicator", _fail_on_rank1)
            m.setattr(sfit_mod, "_abort_world", _sentinel_abort_world)
            if rank == 1:
                with pytest.raises(_AbortSentinel, match="Rank 1"):
                    solver2._ensure_nccl_comm()
                assert len(abort_calls) == 1
                assert abort_calls[0][0] is solver2.comm
            else:
                res_comm2 = solver2._ensure_nccl_comm()
                assert res_comm2 is not None
                assert len(abort_calls) == 0

        # Verify node-local size check: when world size > n_gpus (e.g. multi-node)
        # but node-local _local_size <= n_gpus, _ensure_nccl_comm succeeds.
        solver2._nccl_comm = None
        solver2._nccl_unavailable = False
        with monkeypatch.context() as m:
            m.setattr(SolverFIT3D, "_nccl_comm_cache", {}, raising=False)
            m.setattr(cp.cuda.runtime, "getDeviceCount", lambda: 2)
            m.setattr(
                cp.cuda,
                "Device",
                lambda dev_id: type("_DummyDev", (), {"use": lambda self: None})(),
            )
            m.setattr(nccl, "get_unique_id", lambda: b"dummy_uid")
            m.setattr(nccl, "NcclCommunicator", lambda *a, **kw: object())
            m.setattr(solver2, "_local_size", 2)
            m.setattr(solver2, "_local_rank", rank % 2)
            res_multinode = solver2._ensure_nccl_comm()
            assert res_multinode is not None

        # Verify collective _nccl_comm_cache check, deferred destroy on eviction,
        # and communicator identity tag verification:
        # (i) Register two real SolverFIT3D owners through _ensure_nccl_comm(),
        #     force an eviction, verify the evicted communicator is not destroyed
        #     while the owners are alive, delete both owners, run gc.collect(),
        #     and assert that the communicator was destroyed exactly once.
        # (ii) When this solver is the last owner, evicting the stale cached entry
        #      calls destroy() immediately and re-initializes collectively.
        # (iii) If every rank has an entry under key_r (has_cached == 1 everywhere)
        #       but with mismatched communicator tags (e.g. after py2f() handle reuse),
        #       _ensure_nccl_comm detects the tag mismatch, destroys the mismatched
        #       entries, and rebuilds a unified communicator.
        import gc
        import sys

        solver2._nccl_comm = None
        solver2._nccl_unavailable = False
        destroyed_comms = []

        class _CountingNcclComm:
            def __init__(self, inner=None):
                self._inner = inner
                self.destroy_count = 0

            def destroy(self):
                self.destroy_count += 1
                if self._inner is not None and hasattr(self._inner, "destroy"):
                    self._inner.destroy()
                destroyed_comms.append(self)

        with monkeypatch.context() as m:
            dev_id_r = rank % 2
            key_r = (
                int(solver2.comm.py2f()),
                int(solver2.size),
                int(solver2.rank),
                int(dev_id_r),
            )
            m.setattr(SolverFIT3D, "_nccl_comm_cache", {}, raising=False)
            m.setattr(SolverFIT3D, "_nccl_evicted_comms", {}, raising=False)
            m.setattr(cp.cuda.runtime, "getDeviceCount", lambda: 2)
            m.setattr(
                cp.cuda,
                "Device",
                lambda dev_id: type("_DummyDev", (), {"use": lambda self: None})(),
            )
            m.setattr(cp.cuda.driver, "ctxGetCurrent", lambda: 1)
            m.setattr(nccl, "get_unique_id", lambda: b"dummy_uid_shared")
            shared_comm = _CountingNcclComm()
            m.setattr(nccl, "NcclCommunicator", lambda *a, **kw: shared_comm)

            owner_1 = SolverFIT3D(
                grid2,
                cfln=0.45,
                bc_low=["periodic", "periodic", "pec"],
                bc_high=["periodic", "periodic", "pec"],
                use_mpi=True,
                verbose=0,
            )
            owner_1.use_gpu = True
            owner_1._local_size = 2
            owner_1._local_rank = dev_id_r
            owner_2 = SolverFIT3D(
                grid2,
                cfln=0.45,
                bc_low=["periodic", "periodic", "pec"],
                bc_high=["periodic", "periodic", "pec"],
                use_mpi=True,
                verbose=0,
            )
            owner_2.use_gpu = True
            owner_2._local_size = 2
            owner_2._local_rank = dev_id_r

            assert owner_1._ensure_nccl_comm() is shared_comm
            assert owner_2._ensure_nccl_comm() is shared_comm
            assert SolverFIT3D._nccl_comm_cache[key_r][1] == {
                id(owner_1),
                id(owner_2),
            }

            # Force eviction on all ranks (via tag mismatch across ranks) while
            # owner_1 and owner_2 remain alive
            _, live_owners, _ = SolverFIT3D._nccl_comm_cache[key_r]
            SolverFIT3D._nccl_comm_cache[key_r] = (
                shared_comm,
                live_owners,
                500 + rank,
            )
            fresh_obj_shared = _CountingNcclComm()
            m.setattr(nccl, "NcclCommunicator", lambda *a, **kw: fresh_obj_shared)
            m.setattr(solver2, "_local_size", 2)
            m.setattr(solver2, "_local_rank", dev_id_r)
            res_shared = solver2._ensure_nccl_comm()
            assert res_shared is fresh_obj_shared
            # Not destroyed while owner_1 and owner_2 are still alive
            assert shared_comm.destroy_count == 0
            assert len(destroyed_comms) == 0
            # Deleting only the first owner leaves owner_2 holding the communicator
            del owner_1
            gc.collect()
            assert shared_comm.destroy_count == 0
            # Deleting the second owner and running gc.collect() destroys it exactly once
            del owner_2
            gc.collect()
            assert shared_comm.destroy_count == 1
            assert destroyed_comms == [shared_comm]
            destroyed_comms.clear()

            # Also verify finalizer-driven destroy() skips when sys.is_finalizing()
            # is True or when the CUDA driver context is gone (ctxGetCurrent() == 0).
            for skip_mode in ("finalizing", "no_cuda_ctx"):
                SolverFIT3D._nccl_comm_cache.clear()
                SolverFIT3D._nccl_evicted_comms.clear()
                skip_comm = _CountingNcclComm()
                m.setattr(nccl, "NcclCommunicator", lambda *a, **kw: skip_comm)
                owner_skip = SolverFIT3D(
                    grid2,
                    cfln=0.45,
                    bc_low=["periodic", "periodic", "pec"],
                    bc_high=["periodic", "periodic", "pec"],
                    use_mpi=True,
                    verbose=0,
                )
                owner_skip.use_gpu = True
                owner_skip._local_size = 2
                owner_skip._local_rank = dev_id_r
                assert owner_skip._ensure_nccl_comm() is skip_comm
                _, skip_owners, _ = SolverFIT3D._nccl_comm_cache[key_r]
                SolverFIT3D._nccl_comm_cache[key_r] = (
                    skip_comm,
                    skip_owners,
                    600 + rank,
                )
                solver2._nccl_comm = None
                m.setattr(nccl, "NcclCommunicator", lambda *a, **kw: object())
                solver2._ensure_nccl_comm()
                with monkeypatch.context() as m_skip:
                    if skip_mode == "finalizing":
                        m_skip.setattr(sys, "is_finalizing", lambda: True)
                    else:
                        m_skip.setattr(cp.cuda.driver, "ctxGetCurrent", lambda: 0)
                    del owner_skip
                    gc.collect()
                assert skip_comm.destroy_count == 0
            destroyed_comms.clear()

            solver2._nccl_comm = None
            stale_comm = _CountingNcclComm()
            asym_cache = (
                {key_r: (stale_comm, {id(solver2)}, 88)} if rank == 0 else {}
            )
            m.setattr(SolverFIT3D, "_nccl_comm_cache", asym_cache, raising=False)
            m.setattr(nccl, "get_unique_id", lambda: b"dummy_uid_2")
            fresh_obj = object()
            m.setattr(nccl, "NcclCommunicator", lambda *a, **kw: fresh_obj)
            res_asym = solver2._ensure_nccl_comm()
            assert res_asym is fresh_obj
            assert stale_comm.destroy_count == (1 if rank == 0 else 0)
            assert destroyed_comms == ([stale_comm] if rank == 0 else [])
            destroyed_comms.clear()

            # (iii) All ranks have key_r in cache, but with mismatched tags
            solver2._nccl_comm = None
            mismatch_comm = _CountingNcclComm()
            mismatch_cache = {
                key_r: (mismatch_comm, {id(solver2)}, 1000 + rank)
            }
            m.setattr(SolverFIT3D, "_nccl_comm_cache", mismatch_cache, raising=False)
            m.setattr(nccl, "get_unique_id", lambda: b"dummy_uid_tag_mismatch")
            fresh_obj_tag = object()
            m.setattr(nccl, "NcclCommunicator", lambda *a, **kw: fresh_obj_tag)
            res_tag = solver2._ensure_nccl_comm()
            assert res_tag is fresh_obj_tag
            assert mismatch_comm.destroy_count == 1
            assert destroyed_comms == [mismatch_comm]
            rebuilt_tag = int(SolverFIT3D._nccl_comm_cache[key_r][2])
            assert rebuilt_tag > 0
            assert int(comm.allreduce(rebuilt_tag, op=MPI.MIN)) == int(
                comm.allreduce(rebuilt_tag, op=MPI.MAX)
            )

        # 4. S1: Explicit particles_comm="nccl" must raise collectively when NCCL
        #    is unavailable, whereas "auto"/None warns and falls back to "mpi".
        solver2._nccl_comm = None
        solver2._nccl_unavailable = True
        with pytest.raises(RuntimeError, match="particles_comm='nccl'"):
            ParticleSolver(
                solver2, species=[sp2], shape=1, particles_comm="nccl"
            )
        with pytest.warns(RuntimeWarning, match="falling back to particles_comm='mpi'"):
            ps_fallback = ParticleSolver(
                solver2, species=[sp2], shape=1, particles_comm="auto"
            )
        assert ps_fallback.particles_comm == "mpi"

        # 5. Single-rank OOM during migration regrow (CPU and GPU), attach() headroom,
        #    injection, sort_by_cell, and step-0 _filter_species_to_owned_slab must
        #    raise RuntimeError (and MemoryError) collectively across ALL ranks without hanging.
        grid_oom = GridFIT3D(
            0.0, 0.06, 0.0, 0.06, 0.0, 0.12, 6, 6, 12, use_mpi=True, verbose=0
        )
        solver_oom = SolverFIT3D(
            grid_oom,
            cfln=0.45,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            use_mpi=True,
            verbose=0,
        )
        sp_oom = Species(name="electron", charge=-e_charge, mass=m_e, capacity=1)
        # Place particle just below rank 0's upper slab boundary moving +z into rank 1
        z_b01 = float(0.12 / comm.Get_size())
        sp_oom.add_particles(
            x=np.array([0.03]),
            y=np.array([0.03]),
            z=np.array([z_b01 - 1.0e-4]),
            ux=np.array([0.0]),
            uy=np.array([0.0]),
            uz=np.array([0.35 * c]),
            w=np.array([1.0e6]),
        )
        ps_oom = ParticleSolver(
            solver_oom, species=[sp_oom], shape=1, capacity_headroom=0.0
        )
        # Inject OOM on rank 0 during attach() headroom reservation
        orig_ensure_cap = sp_oom._ensure_capacity

        def _oom_on_rank0_attach(req, exact=False):
            if rank == 0:
                raise MemoryError("Simulated rank-0 OOM in attach headroom")
            return orig_ensure_cap(req, exact=exact)

        ps_oom.capacity_headroom = 0.05
        with monkeypatch.context() as m:
            m.setattr(sp_oom, "_ensure_capacity", _oom_on_rank0_attach)
            with pytest.raises(RuntimeError):
                ps_oom.attach()

        # Now attach normally with capacity_headroom=0.0 and inject OOM on rank 1
        # when the particle migrates from rank 0 -> rank 1 and rank 1 regrows capacity
        ps_oom.capacity_headroom = 0.0
        ps_oom.attach()

        def _oom_on_rank1_regrow(req, exact=False):
            if rank == 1:
                raise MemoryError("Simulated rank-1 OOM in migration regrow")
            return orig_ensure_cap(req, exact=exact)

        with monkeypatch.context() as m:
            m.setattr(sp_oom, "_ensure_capacity", _oom_on_rank1_regrow)
            with pytest.raises(RuntimeError):
                solver_oom.one_step()

        # Helper to verify that disabling _enable_step_status_piggyback causes
        # rank 0 to raise raw MemoryError (not collective RuntimeError) while
        # peer ranks do not receive a collective RuntimeError (assert not peer_raised_runtime).
        def _assert_piggyback_disabled_skips_collective_error(
            solver_inst, ps_inst, sp_inst
        ):
            n_steps_before = ps_inst.n_particle_steps
            try:
                ps_inst._enable_step_status_piggyback = False
                if rank == 0:
                    raw_oom_caught = False
                    try:
                        solver_inst.one_step()
                    except MemoryError as exc:
                        # Must be raw MemoryError, NOT collective RuntimeError
                        assert not isinstance(exc, RuntimeError)
                        raw_oom_caught = True
                    assert raw_oom_caught
                    # Drain the remaining step collectives so peer ranks unblock
                    ps_inst._mpi_post_push_hook(sp_inst)
                    ps_inst._mpi_sum_guards(ps_inst.J_particles)
                    solver_inst._mpi_communicate(solver_inst.E)
                else:
                    peer_raised_runtime = False
                    try:
                        solver_inst.one_step()
                    except RuntimeError:
                        peer_raised_runtime = True
                    assert not peer_raised_runtime
                comm.Barrier()
            finally:
                ps_inst._enable_step_status_piggyback = True
                ps_inst.n_particle_steps = n_steps_before

        # 6. Single-rank OOM during injection (inj.inject -> add_particles -> _ensure_capacity,
        #    called on the owning rank only) and during sp.sort_by_cell, plus step-0
        #    _filter_species_to_owned_slab(), must raise RuntimeError collectively across ALL ranks.
        for use_gpu_mode in (
            [False, True] if int(cp.cuda.runtime.getDeviceCount()) >= 1 else [False]
        ):
            grid_inj = GridFIT3D(
                0.0, 0.06, 0.0, 0.06, 0.0, 0.12, 6, 6, 12, use_mpi=True, verbose=0
            )
            solver_inj = SolverFIT3D(
                grid_inj,
                cfln=0.45,
                bc_low=["periodic", "periodic", "pec"],
                bc_high=["periodic", "periodic", "pec"],
                use_mpi=True,
                use_gpu=use_gpu_mode,
                verbose=0,
            )
            sp_inj = Species(
                name="electron",
                charge=-e_charge,
                mass=m_e,
                capacity=1,
                use_gpu=use_gpu_mode,
            )
            sp_inj.add_particles(
                x=np.array([0.03, 0.03]),
                y=np.array([0.03, 0.03]),
                z=np.array([0.015, 0.075]),
                ux=np.array([0.0, 0.0]),
                uy=np.array([0.0, 0.0]),
                uz=np.array([0.01 * c, 0.01 * c]),
                w=np.array([1.0e6, 1.0e6]),
            )
            ps_inj = ParticleSolver(
                solver_inj,
                species=[sp_inj],
                shape=1,
                sort_every=1,
                particles_comm="mpi" if use_gpu_mode else None,
                capacity_headroom=0.0,
            ).attach()
            # Emitter at z0 = 0.01 m is owned by rank 0 only
            inj = BeamInjector(
                species=sp_inj,
                z0=0.01,
                current=0.05,
                gamma=1.15,
                x0=0.03,
                y0=0.03,
                radius=0.004,
                n_macro_per_step=4,
                direction=1,
                seed=303,
            )
            ps_inj.injectors.append(inj)

            # (a) Step-0 _filter_species_to_owned_slab OOM on rank 0
            ps_inj._initialized_half_step = False
            ps_inj._init_gauss_counts = (-1,)
            orig_filter = ps_inj._filter_species_to_owned_slab

            def _filter_oom_rank0():
                if rank == 0:
                    raise MemoryError("Simulated rank-0 OOM in _filter_species_to_owned_slab")
                return orig_filter()

            with monkeypatch.context() as m:
                m.setattr(ps_inj, "_filter_species_to_owned_slab", _filter_oom_rank0)
                with pytest.raises(RuntimeError, match="OOM|memory") as flt_exc_info:
                    solver_inj.one_step()
                if rank == 0:
                    assert isinstance(flt_exc_info.value.__cause__, MemoryError)
                # Mutation check: disabling piggyback leaves peer ranks without collective RuntimeError
                ps_inj._initialized_half_step = False
                ps_inj._init_gauss_counts = (-1,)
                _assert_piggyback_disabled_skips_collective_error(
                    solver_inj, ps_inj, sp_inj
                )
            ps_inj._initialized_half_step = True
            ps_inj._init_gauss_counts = (sp_inj._n_launch_bound,)

            # (b) Injection OOM on owning rank 0 (inj.inject -> add_particles -> _ensure_capacity)
            orig_inj_cap = sp_inj._ensure_capacity

            def _inj_oom_rank0(req, exact=False):
                if rank == 0:
                    raise MemoryError("Simulated rank-0 OOM in inj.inject _ensure_capacity")
                return orig_inj_cap(req, exact=exact)

            with monkeypatch.context() as m:
                m.setattr(sp_inj, "_ensure_capacity", _inj_oom_rank0)
                with pytest.raises(RuntimeError, match="OOM|memory"):
                    solver_inj.one_step()
                # Mutation check: disabling piggyback leaves peer ranks without collective RuntimeError
                _assert_piggyback_disabled_skips_collective_error(
                    solver_inj, ps_inj, sp_inj
                )

            # Remove injector so injection succeeds, advance 1 step so n_particle_steps >= 1,
            # and test (c) sp.sort_by_cell OOM on rank 0
            ps_inj.injectors.clear()
            solver_inj.one_step()
            orig_sort = sp_inj.sort_by_cell

            def _sort_oom_rank0(gmap):
                if rank == 0:
                    raise MemoryError("Simulated rank-0 OOM in sp.sort_by_cell")
                return orig_sort(gmap)

            with monkeypatch.context() as m:
                m.setattr(sp_inj, "sort_by_cell", _sort_oom_rank0)
                with pytest.raises(RuntimeError, match="OOM|memory"):
                    solver_inj.one_step()
                # Mutation check: disabling piggyback leaves peer ranks without collective RuntimeError
                _assert_piggyback_disabled_skips_collective_error(
                    solver_inj, ps_inj, sp_inj
                )

        # Also test GPU migration regrow OOM on rank 1 when CuPy is available
        if int(cp.cuda.runtime.getDeviceCount()) >= 1:
            grid_goom = GridFIT3D(
                0.0, 0.06, 0.0, 0.06, 0.0, 0.12, 6, 6, 12, use_mpi=True, verbose=0
            )
            solver_goom = SolverFIT3D(
                grid_goom,
                cfln=0.45,
                bc_low=["periodic", "periodic", "pec"],
                bc_high=["periodic", "periodic", "pec"],
                use_mpi=True,
                use_gpu=True,
                verbose=0,
            )
            sp_goom = Species(
                name="electron", charge=-e_charge, mass=m_e, capacity=1, use_gpu=True
            )
            sp_goom.add_particles(
                x=np.array([0.03]),
                y=np.array([0.03]),
                z=np.array([z_b01 - 1.0e-4]),
                ux=np.array([0.0]),
                uy=np.array([0.0]),
                uz=np.array([0.35 * c]),
                w=np.array([1.0e6]),
            )
            ps_goom = ParticleSolver(
                solver_goom,
                species=[sp_goom],
                shape=1,
                particles_comm="mpi",
                capacity_headroom=0.0,
            ).attach()
            orig_gpu_bufs = ps_goom._ensure_gpu_mpi_buffers

            def _gpu_buf_oom_on_rank0(
                pack_cap=4096, recv_lo_cap=4096, recv_hi_cap=4096
            ):
                if rank == 0 and pack_cap > 0:
                    raise cp.cuda.memory.OutOfMemoryError(1024, 2048)
                return orig_gpu_bufs(
                    pack_cap=pack_cap,
                    recv_lo_cap=recv_lo_cap,
                    recv_hi_cap=recv_hi_cap,
                )

            if rank == 0:
                sp_goom._z[:1] = z_b01 + 1.0e-4
            ps_goom._gpu_mig_pack_cap = 0
            with monkeypatch.context() as m:
                m.setattr(ps_goom, "_ensure_gpu_mpi_buffers", _gpu_buf_oom_on_rank0)
                with pytest.raises(RuntimeError):
                    ps_goom._mpi_migrate_particles_gpu(sp_goom)
            ps_goom._gpu_mig_pack_cap = 4096
            if rank == 0:
                sp_goom._z[:1] = z_b01 - 1.0e-4

            orig_gpu_cap = sp_goom._ensure_capacity

            def _gpu_oom_on_rank1(req, exact=False):
                if rank == 1:
                    raise cp.cuda.memory.OutOfMemoryError(1024, 2048)
                return orig_gpu_cap(req, exact=exact)

            with monkeypatch.context() as m:
                m.setattr(sp_goom, "_ensure_capacity", _gpu_oom_on_rank1)
                with pytest.raises(RuntimeError):
                    solver_goom.one_step()


def _require_multi_gpu():
    """Skip if fewer than 2 CUDA GPUs are available or MPI size exceeds GPU count."""
    cp = pytest.importorskip("cupy")
    try:
        n_gpus = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    if n_gpus < 2:
        pytest.skip(f"Multi-GPU tests require >= 2 GPUs (found {n_gpus})")
    size = MPI.COMM_WORLD.Get_size()
    if size > n_gpus:
        pytest.skip(
            f"Multi-GPU tests require one MPI rank per GPU (size={size} > n_gpus={n_gpus})"
        )
    return cp


def _field_comp_host(field, comp: str) -> np.ndarray:
    """Return component ``comp`` of ``field`` as a host NumPy array."""
    arr = field[:, :, :, comp]
    return np.asarray(arr.get() if hasattr(arr, "get") else arr)


def _assert_v2_matches_reference(run_a, run_b, bound: float):
    """Assert two V2 run dicts agree within ``bound`` on fields, energies, and particles."""
    E_a, E_b = run_a["E"], run_b["E"]
    H_a, H_b = run_a["H"], run_b["H"]
    e_norm = max(
        float(np.max(np.abs(_field_comp_host(E_b, c_name)))) for c_name in "xyz"
    )
    h_norm = max(
        float(np.max(np.abs(_field_comp_host(H_b, c_name)))) for c_name in "xyz"
    )
    assert e_norm > 0.0 and h_norm > 0.0

    rel_E = (
        max(
            float(
                np.max(
                    np.abs(
                        _field_comp_host(E_a, c_name)
                        - _field_comp_host(E_b, c_name)
                    )
                )
            )
            for c_name in "xyz"
        )
        / e_norm
    )
    rel_H = (
        max(
            float(
                np.max(
                    np.abs(
                        _field_comp_host(H_a, c_name)
                        - _field_comp_host(H_b, c_name)
                    )
                )
            )
            for c_name in "xyz"
        )
        / h_norm
    )
    assert rel_E <= bound
    assert rel_H <= bound

    sp_a, sp_b = run_a["sp"], run_b["sp"]
    assert sp_a.n_particles == sp_b.n_particles == 48
    order_a = np.argsort(sp_a.id)
    order_b = np.argsort(sp_b.id)
    np.testing.assert_array_equal(sp_a.id[order_a], sp_b.id[order_b])
    np.testing.assert_allclose(
        sp_a.w[order_a], sp_b.w[order_b], rtol=0.0, atol=1e-15
    )

    dz_min = run_b["L_z_min"]
    for pos_attr in ("x", "y", "z"):
        diff_pos = float(
            np.max(
                np.abs(
                    getattr(sp_a, pos_attr)[order_a]
                    - getattr(sp_b, pos_attr)[order_b]
                )
            )
        )
        assert diff_pos / dz_min <= bound

    u_norm = float(np.max(np.sqrt(sp_b.ux**2 + sp_b.uy**2 + sp_b.uz**2)))
    for vel_attr in ("ux", "uy", "uz"):
        diff_u = float(
            np.max(
                np.abs(
                    getattr(sp_a, vel_attr)[order_a]
                    - getattr(sp_b, vel_attr)[order_b]
                )
            )
        )
        assert diff_u / u_norm <= bound

    assert abs(run_a["q_tot"] - run_b["q_tot"]) / abs(run_b["q_tot"]) <= 1.0e-14
    assert abs(run_a["ke_tot"] - run_b["ke_tot"]) / run_b["ke_tot"] <= bound
    assert abs(run_a["ue_tot"] - run_b["ue_tot"]) / run_b["ue_tot"] <= bound


@pytest.mark.parametrize("graded", [False, True])
def test_multi_gpu_v2_matches_1gpu_and_cpu_serial(graded, monkeypatch):
    """Multi-GPU V2/V8: 2-GPU NCCL run matches 1-GPU and CPU serial within round-off over 500 steps."""
    _require_multi_gpu()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_steps = 500

    mut_v2_2gpu = None
    if not graded:
        # Charge-conserving parallel GPU/MPI mutation (b) (run first):
        # Zero plane k = Nz - 1 of E_flat and B_n_flat passed to
        # launch_gather_and_push_gpu on rank 0 at every step (does not skip the
        # halo exchange). Esirkepov deposition conserves charge for any
        # trajectory so gauss_res stays <= 1e-12, while _report_roundoff_comparison
        # yields obs/bound > 1.0, proving that the trajectory comparison check is
        # wired up (not how sensitive the bound is).
        import wakis.particles.gpu_kernels as gk_mod

        orig_lgp = gk_mod.launch_gather_and_push_gpu

        def _zero_upper_gather_plane_rank0(
            sp_i, grid_map, E_flat, B_n_flat, **kw
        ):
            """Zero plane ``k = Nz - 1`` of ``E_flat`` and ``B_n_flat`` on rank 0 every step.

            Proves that the multi-GPU trajectory / round-off comparison check is
            wired up and catches a charge-conserving gather corruption
            (``gauss_res <= 1e-12``, ``obs/bound > 1``), not how sensitive the
            bound is.
            """
            if rank == 0:
                nxy = int(grid_map.Nx * grid_map.Ny)
                Nz = int(grid_map.Nz)
                E_mut = E_flat.copy().reshape((3, Nz, nxy))
                B_mut = B_n_flat.copy().reshape((3, Nz, nxy))
                E_mut[:, Nz - 1, :] = 0.0
                B_mut[:, Nz - 1, :] = 0.0
                return orig_lgp(
                    sp_i, grid_map, E_mut.ravel(), B_mut.ravel(), **kw
                )
            return orig_lgp(sp_i, grid_map, E_flat, B_n_flat, **kw)

        with monkeypatch.context() as m:
            m.setattr(
                gk_mod,
                "launch_gather_and_push_gpu",
                _zero_upper_gather_plane_rank0,
            )
            mut_v2_2gpu = _run_v2_coupled(
                graded=graded,
                use_mpi=True,
                n_steps=n_steps,
                use_gpu=True,
                particles_comm="nccl",
            )
        assert mut_v2_2gpu["gauss_res"] <= 1.0e-12

    par_2gpu = _run_v2_coupled(
        graded=graded,
        use_mpi=True,
        n_steps=n_steps,
        use_gpu=True,
        particles_comm="nccl",
    )
    assert par_2gpu["particles_comm"] == "nccl"
    assert par_2gpu["nccl_comm"] is not None
    assert par_2gpu["migrations"] >= 100
    assert par_2gpu["gauss_res"] <= 1.0e-12
    _assert_interior_bidirectional_migration(
        par_2gpu["sent_lo"], par_2gpu["sent_hi"], comm
    )

    if rank != 0:
        return

    ser_1gpu = _run_v2_coupled(
        graded=graded, use_mpi=False, n_steps=n_steps, use_gpu=True
    )
    ser_cpu, pert_runs = _run_serial_perturbation_envelope(
        _run_v2_coupled,
        graded=graded,
        use_mpi=False,
        n_steps=n_steps,
        use_gpu=False,
    )
    assert ser_1gpu["gauss_res"] <= 1.0e-12
    assert ser_cpu["gauss_res"] <= 1.0e-12
    bound = derived_mpi_roundoff_bound(ser_cpu, pert_runs)
    if mut_v2_2gpu is not None:
        gauss_v2_unmut = float(par_2gpu["gauss_res"])
        gauss_v2_mut = float(mut_v2_2gpu["gauss_res"])
        gauss_v2_ratio = gauss_v2_mut / max(gauss_v2_unmut, 1.0e-30)
        # Heuristic band, not derived: zeroing one gather plane must leave the
        # Gauss residual at its round-off floor (the mutation is charge-conserving),
        # so the mutated/unmutated ratio should be O(1); a factor 2 either way is
        # allowed for run-to-run round-off differences.
        assert 0.5 <= gauss_v2_ratio <= 2.0
        mut_v2_label = (
            "v2_2gpu_zero_upper_gather_plane_mutation_vs_cpu_serial"
        )
        ratio_mut_v2 = _report_roundoff_comparison(
            mut_v2_label,
            ser_cpu,
            mut_v2_2gpu,
            bound,
        )
        ser_cpu["roundoff_comparisons"][mut_v2_label].update(
            {
                "unmutated_gauss_res": gauss_v2_unmut,
                "mutated_gauss_res": gauss_v2_mut,
                "mutated_to_unmutated_gauss_res_ratio": float(gauss_v2_ratio),
            }
        )
        print(
            f"[roundoff {mut_v2_label}_gauss] "
            f"unmutated_gauss_res={gauss_v2_unmut:.4e}, "
            f"mutated_gauss_res={gauss_v2_mut:.4e} "
            f"(mut/unmut={gauss_v2_ratio:.4f})"
        )
        assert ratio_mut_v2 > 1.0
    ratio_2gpu_cpu = _report_roundoff_comparison(
        f"v2_2gpu_vs_cpu_serial(graded={graded})", ser_cpu, par_2gpu, bound
    )
    ratio_1gpu_cpu = _report_roundoff_comparison(
        f"v2_1gpu_vs_cpu_serial(graded={graded})", ser_cpu, ser_1gpu, bound
    )
    assert ratio_2gpu_cpu <= 1.0
    assert ratio_1gpu_cpu <= 1.0

    _assert_v2_matches_reference(par_2gpu, ser_1gpu, bound=bound)
    _assert_v2_matches_reference(par_2gpu, ser_cpu, bound=bound)
    _assert_v2_matches_reference(ser_1gpu, ser_cpu, bound=bound)


@pytest.mark.parametrize("particles_comm", ["nccl", "mpi"])
def test_multi_gpu_v3_cold_plasma_matches_1gpu_and_cpu_serial(
    particles_comm, monkeypatch
):
    """Multi-GPU V3/V8: 2-GPU NCCL and pinned-host MPI cold-plasma runs match 1-GPU and CPU serial within round-off."""
    _require_multi_gpu()
    comm = MPI.COMM_WORLD
    n_steps = 240
    eps_hi = _V3_GUARD_SCALE_MUTATION_EPS
    eps_lo = 0.1 * eps_hi

    # Real parallel GPU/MPI mutation check on both NCCL and pinned-host MPI guard paths:
    # Scale the received guard current by (1 + eps_hi) in
    # ParticleSolver._accumulate_guard_plane during the 2-GPU run, and also run at
    # (1 + eps_lo) = (1 + 0.1 * eps_hi) to verify that eps_hi is the smallest
    # power of ten that fails (obs/bound > 1 at eps_hi, obs/bound <= 1 at eps_lo)
    # and to linearly interpolate the threshold eps_crit at run time.
    # Gauss-silent only to the 1e-12 tolerance: the mutation raises gauss_res above
    # the unmutated round-off floor (asserted ordering below, values printed), so the
    # Gauss residual is the more sensitive check for this defect; this test shows
    # that the trajectory envelope catches it as well, independently.
    with monkeypatch.context() as m:
        m.setattr(
            ParticleSolver,
            "_accumulate_guard_plane",
            staticmethod(
                lambda dst, src: dst.__iadd__((1.0 + eps_hi) * src)
            ),
        )
        mut_2gpu = _run_v3_cold_plasma(
            use_mpi=True,
            n_steps=n_steps,
            use_gpu=True,
            particles_comm=particles_comm,
        )
    assert mut_2gpu["gauss_res"] <= 1.0e-12

    with monkeypatch.context() as m:
        m.setattr(
            ParticleSolver,
            "_accumulate_guard_plane",
            staticmethod(
                lambda dst, src: dst.__iadd__((1.0 + eps_lo) * src)
            ),
        )
        mut_2gpu_uncaught = _run_v3_cold_plasma(
            use_mpi=True,
            n_steps=n_steps,
            use_gpu=True,
            particles_comm=particles_comm,
        )
    assert mut_2gpu_uncaught["gauss_res"] <= 1.0e-12

    par_2gpu = _run_v3_cold_plasma(
        use_mpi=True,
        n_steps=n_steps,
        use_gpu=True,
        particles_comm=particles_comm,
    )
    assert par_2gpu["particles_comm"] == particles_comm
    if particles_comm == "nccl":
        assert par_2gpu["nccl_comm"] is not None
    else:
        assert par_2gpu["nccl_comm"] is None
    assert par_2gpu["gauss_res"] <= 1.0e-12
    gauss_unmut = float(par_2gpu["gauss_res"])
    gauss_mut_lo = float(mut_2gpu_uncaught["gauss_res"])
    gauss_mut = float(mut_2gpu["gauss_res"])
    gauss_ratio = gauss_mut / max(gauss_unmut, 1.0e-30)
    assert 0.0 < gauss_unmut <= gauss_mut_lo <= gauss_mut <= 1.0e-12
    _assert_interior_bidirectional_migration(
        par_2gpu["sent_lo"], par_2gpu["sent_hi"], comm
    )

    # Same staggered leapfrog modified-Hamiltonian bound 2.0 * 0.25 * (omega_pred * dt)^2
    # = 0.5 * (omega_pred * dt)^2 derived in test_v3_cold_plasma_mpi_vs_serial above
    e_tot = par_2gpu["ke"] + par_2gpu["ue"]
    rel_e_drift = float(np.max(np.abs(e_tot - e_tot[0])) / e_tot[0])
    safety_factor_energy = 2.0
    leapfrog_energy_bound = (
        safety_factor_energy
        * 0.25
        * ((par_2gpu["omega_pred"] * par_2gpu["dt"]) ** 2)
    )
    assert rel_e_drift <= leapfrog_energy_bound

    if comm.Get_rank() != 0:
        return

    ser_1gpu = _run_v3_cold_plasma(
        use_mpi=False, n_steps=n_steps, use_gpu=True
    )
    ser_cpu, pert_runs = _run_serial_perturbation_envelope(
        _run_v3_cold_plasma, use_mpi=False, n_steps=n_steps, use_gpu=False
    )
    assert ser_1gpu["gauss_res"] <= 1.0e-12
    assert ser_cpu["gauss_res"] <= 1.0e-12
    bound = derived_mpi_roundoff_bound(ser_cpu, pert_runs)
    mut_label_hi = (
        f"v3_2gpu_{particles_comm}_guard_scale_mutation(eps={eps_hi:.0e})_vs_cpu_serial"
    )
    mut_label_lo = (
        f"v3_2gpu_{particles_comm}_guard_scale_mutation_uncaught(eps={eps_lo:.0e})_vs_cpu_serial"
    )
    ratio_mut_2gpu = _report_roundoff_comparison(
        mut_label_hi,
        ser_cpu,
        mut_2gpu,
        bound,
    )
    ratio_mut_2gpu_lo = _report_roundoff_comparison(
        mut_label_lo,
        ser_cpu,
        mut_2gpu_uncaught,
        bound,
    )
    eps_crit = float(
        eps_lo
        + (1.0 - ratio_mut_2gpu_lo)
        * (eps_hi - eps_lo)
        / (ratio_mut_2gpu - ratio_mut_2gpu_lo)
    )
    gamma_n = float(ser_cpu["roundoff_envelope"]["gamma_n"])
    eps_crit_over_gamma_n = float(eps_crit / gamma_n)
    ser_cpu["roundoff_comparisons"][mut_label_hi].update(
        {
            "eps_caught": float(eps_hi),
            "eps_uncaught": float(eps_lo),
            "obs_to_bound_uncaught": float(ratio_mut_2gpu_lo),
            "eps_crit": eps_crit,
            "eps_crit_over_gamma_n": eps_crit_over_gamma_n,
            "unmutated_gauss_res": gauss_unmut,
            "uncaught_mutated_gauss_res": gauss_mut_lo,
            "mutated_gauss_res": gauss_mut,
            "mutated_to_unmutated_gauss_res_ratio": float(gauss_ratio),
        }
    )
    print(
        f"[roundoff v3_2gpu_{particles_comm}_guard_scale_sensitivity] "
        f"eps_uncaught={eps_lo:.1e} (obs/bound={ratio_mut_2gpu_lo:.4f}, "
        f"gauss_res={gauss_mut_lo:.4e}), "
        f"eps_caught={eps_hi:.1e} (obs/bound={ratio_mut_2gpu:.4f}, "
        f"gauss_res={gauss_mut:.4e}), "
        f"eps_crit={eps_crit:.4e} ({eps_crit_over_gamma_n:.2f} * gamma_n, "
        f"gamma_n={gamma_n:.4e}), "
        f"unmutated_gauss_res={gauss_unmut:.4e}, "
        f"mutated_gauss_res={gauss_mut:.4e} "
        f"(mut/unmut={gauss_ratio:.4f})"
    )
    assert ratio_mut_2gpu > 1.0
    assert ratio_mut_2gpu_lo <= 1.0
    ratio_2gpu_cpu = _report_roundoff_comparison(
        f"v3_2gpu_{particles_comm}_vs_cpu_serial", ser_cpu, par_2gpu, bound
    )
    ratio_1gpu_cpu = _report_roundoff_comparison(
        f"v3_1gpu_vs_cpu_serial({particles_comm})", ser_cpu, ser_1gpu, bound
    )
    assert ratio_2gpu_cpu <= 1.0
    assert ratio_1gpu_cpu <= 1.0

    # Mutation check (item 3d): a 1e-9 relative perturbation on deposited current
    # edges must exceed the round-off bound and be caught.
    _, mut_runs = _run_serial_perturbation_envelope(
        _run_v3_cold_plasma,
        use_mpi=False,
        n_steps=n_steps,
        use_gpu=False,
        seeds=(101,),
        deposit_rel_override=1.0e-9,
    )
    mut_dev = _max_relative_run_deviation(ser_cpu, mut_runs[0])
    assert mut_dev > bound

    e_tot_ser = ser_cpu["ke"] + ser_cpu["ue"]
    ser_e_drift = float(np.max(np.abs(e_tot_ser - e_tot_ser[0])) / e_tot_ser[0])
    assert abs(rel_e_drift - ser_e_drift) <= bound

    ez_2gpu = _field_comp_host(par_2gpu["E"], "z")
    for ref in (ser_1gpu, ser_cpu):
        ez_ref = _field_comp_host(ref["E"], "z")
        ez_norm = float(np.max(np.abs(ez_ref)))
        assert ez_norm > 0.0
        assert float(np.max(np.abs(ez_2gpu - ez_ref))) / ez_norm <= bound

        sp_2gpu, sp_ref = par_2gpu["sp"], ref["sp"]
        assert sp_2gpu.n_particles == sp_ref.n_particles
        order_r = np.argsort(sp_ref.id)
        np.testing.assert_array_equal(sp_2gpu.id, sp_ref.id[order_r])
        assert (
            float(np.max(np.abs(sp_2gpu.z - sp_ref.z[order_r])))
            / par_2gpu["dz"]
            <= bound
        )
        assert (
            float(np.max(np.abs(sp_2gpu.uz - sp_ref.uz[order_r])))
            / float(np.max(np.abs(sp_ref.uz)))
            <= bound
        )
        np.testing.assert_allclose(
            par_2gpu["ke"], ref["ke"], rtol=bound, atol=bound * float(e_tot_ser[0])
        )
        np.testing.assert_allclose(
            par_2gpu["ue"], ref["ue"], rtol=bound, atol=bound * float(e_tot_ser[0])
        )


def test_multi_gpu_nccl_matches_pinned_mpi_fallback():
    """Multi-GPU NCCL path and pinned-host MPI fallback agree within the measured reordering round-off bound."""
    _require_multi_gpu()
    comm = MPI.COMM_WORLD
    n_steps = 200

    run_nccl = _run_v2_coupled(
        graded=False,
        use_mpi=True,
        n_steps=n_steps,
        use_gpu=True,
        particles_comm="nccl",
    )
    run_mpi = _run_v2_coupled(
        graded=False,
        use_mpi=True,
        n_steps=n_steps,
        use_gpu=True,
        particles_comm="mpi",
    )
    assert run_nccl["particles_comm"] == "nccl"
    assert run_nccl["nccl_comm"] is not None
    assert run_mpi["particles_comm"] == "mpi"
    assert run_mpi["nccl_comm"] is None
    assert run_nccl["gauss_res"] <= 1.0e-12
    assert run_mpi["gauss_res"] <= 1.0e-12
    _assert_interior_bidirectional_migration(
        run_nccl["sent_lo"], run_nccl["sent_hi"], comm
    )
    _assert_interior_bidirectional_migration(
        run_mpi["sent_lo"], run_mpi["sent_hi"], comm
    )

    if comm.Get_rank() != 0:
        return

    ser, pert_runs = _run_serial_perturbation_envelope(
        _run_v2_coupled,
        graded=False,
        use_mpi=False,
        n_steps=n_steps,
        use_gpu=False,
    )
    bound = derived_mpi_roundoff_bound(ser, pert_runs)
    run_mpi["roundoff_envelope"] = dict(ser["roundoff_envelope"])
    ratio = _report_roundoff_comparison(
        "v2_nccl_vs_pinned_mpi", run_mpi, run_nccl, bound
    )
    assert ratio <= 1.0
    _assert_v2_matches_reference(run_nccl, run_mpi, bound=bound)


def test_multi_gpu_nccl_vs_pinned_mpi_single_step_bit_exact():
    """Single-step unit test: given identical deterministic inputs, NCCL and pinned-MPI guard/ghost/migration exchanges are bit-exact."""
    cp = _require_multi_gpu()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    def _run_comm_once(comm_backend: str):
        grid = GridFIT3D(
            0.0, 0.06, 0.0, 0.06, 0.0, 0.12, 6, 6, 12, use_mpi=True, verbose=0
        )
        solver = SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            use_mpi=True,
            use_gpu=True,
            verbose=0,
        )
        z0 = np.array([0.02985, 0.03015, 0.05985, 0.06015, 0.08985, 0.09015])
        x0 = np.full_like(z0, 0.03)
        y0 = np.full_like(z0, 0.03)
        ux0 = np.full_like(z0, 0.25 * c)
        uy0 = np.full_like(z0, -0.25 * c)
        uz0 = np.array([0.35 * c, -0.35 * c, 0.35 * c, -0.35 * c, 0.35 * c, -0.35 * c])
        w0 = np.full_like(z0, 1.0e7)
        sp = Species(name="electron", charge=-e_charge, mass=m_e, use_gpu=True)
        sp.add_particles(x=x0, y=y0, z=z0, ux=ux0, uy=uy0, uz=uz0, w=w0)
        ps = ParticleSolver(
            solver,
            species=[sp],
            shape=1,
            sort_every=0,
            particles_comm=comm_backend,
        ).attach()

        # Advance local z across internal slab faces so _mpi_migrate_particles_gpu
        # sends particles both down and up across all internal slab boundaries
        n_loc = sp.n_particles
        if n_loc > 0:
            sp._z[:n_loc] = sp._z[:n_loc] + solver.dt * sp._uz[:n_loc]
        ps._mpi_migrate_particles_gpu(sp)
        _assert_interior_bidirectional_migration(
            ps._n_migrated_lo, ps._n_migrated_hi, comm
        )

        nxy = int(solver.Nx * solver.Ny)
        N = int(nxy * solver.Nz)
        rng = np.random.default_rng(9001 + rank)
        j_in = rng.standard_normal(3 * N).astype(np.float64)
        wx_extra = rng.standard_normal(nxy).astype(np.float64)
        wy_extra = rng.standard_normal(nxy).astype(np.float64)
        wz_extra = rng.standard_normal(nxy).astype(np.float64)
        e_in = rng.standard_normal(3 * N).astype(np.float64)

        ps.J_particles[...] = cp.asarray(j_in)
        if ps._Wx_buf.size >= N + nxy:
            ps._Wx_buf[N : N + nxy] = cp.asarray(wx_extra)
            ps._Wy_buf[N : N + nxy] = cp.asarray(wy_extra)
            ps._Wz_buf[N : N + nxy] = cp.asarray(wz_extra)
        solver.E.fromarray(cp.asarray(e_in))

        ps._mpi_sum_guards(ps.J_particles)
        solver._mpi_communicate(solver.E)
        return (
            cp.asnumpy(ps.J_particles).reshape((3, solver.Nz, nxy)),
            cp.asnumpy(solver.E.toarray()).reshape((3, solver.Nz, nxy)),
            cp.asnumpy(sp._z[: sp.n_particles]),
            ps.particles_comm,
            solver._nccl_comm,
        )

    j_nccl, e_nccl, z_nccl, mode_nccl, nccl_comm_obj = _run_comm_once("nccl")
    j_mpi, e_mpi, z_mpi, mode_mpi, mpi_nccl_obj = _run_comm_once("mpi")
    assert mode_nccl == "nccl"
    assert nccl_comm_obj is not None
    assert mode_mpi == "mpi"
    assert mpi_nccl_obj is None
    # Owned planes receiving guard contributions, communicated field ghost planes, and migrated particles must be bit-identical
    np.testing.assert_array_equal(j_nccl, j_mpi)
    np.testing.assert_array_equal(e_nccl, e_mpi)
    np.testing.assert_array_equal(z_nccl, z_mpi)


@pytest.mark.parametrize(
    "mutation",
    [
        "overwrite_guards",
        "zero_second_high_guard",
        "skip_low_guard",
        "drop_migrating",
    ],
)
def test_multi_gpu_mutations_fail(mutation, monkeypatch):
    """Mutations on the multi-GPU NCCL guard-reduction and migration paths must fail."""
    cp = _require_multi_gpu()
    grid = GridFIT3D(
        0.0, 0.06, 0.0, 0.06, 0.0, 0.12, 6, 6, 12, use_mpi=True, verbose=0
    )
    solver = SolverFIT3D(
        grid,
        cfln=0.5,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_mpi=True,
        use_gpu=True,
        verbose=0,
    )
    z0 = np.array([0.02985, 0.03015, 0.05985, 0.06015, 0.08985, 0.09015])
    x0 = np.full_like(z0, 0.03)
    y0 = np.full_like(z0, 0.03)
    ux0 = np.full_like(z0, 0.25 * c)
    uy0 = np.full_like(z0, -0.25 * c)
    uz0 = np.array([0.35 * c, -0.35 * c, 0.35 * c, -0.35 * c, 0.35 * c, -0.35 * c])
    w0 = np.full_like(z0, 1.0e7)
    q_init = float(-e_charge * np.sum(w0))

    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    sp.add_particles(x=x0, y=y0, z=z0, ux=ux0, uy=uy0, uz=uz0, w=w0)
    ps = ParticleSolver(
        solver, species=[sp], shape=1, sort_every=0, particles_comm="nccl"
    )
    assert solver._nccl_comm is not None

    if mutation == "overwrite_guards":
        monkeypatch.setattr(
            ps,
            "_accumulate_guard_plane",
            lambda target_slice, guard_slice: target_slice.__setitem__(
                Ellipsis, guard_slice
            ),
        )
    elif mutation == "zero_second_high_guard":
        monkeypatch.setattr(
            ps,
            "_pack_second_high_guard_plane",
            lambda send_hi, N, nxy: send_hi[:, 1, :].fill(0.0),
        )
    elif mutation == "skip_low_guard":
        monkeypatch.setattr(
            ps,
            "_accumulate_low_guard_plane",
            lambda target_slice, guard_slice: None,
        )
    elif mutation == "drop_migrating":

        def _drop_and_count_migrating(sp_i):
            n_loc = sp_i.n_particles
            z_loc = sp_i._z[:n_loc]
            if solver.rank > 0:
                ps._n_migrated_lo += int(cp.sum(z_loc < ps._z_own_min).get())
            if solver.rank < solver.size - 1:
                ps._n_migrated_hi += int(cp.sum(z_loc >= ps._z_own_max).get())
            sp_i.keep(ps._owned_z_mask(sp_i.z))

        monkeypatch.setattr(
            ps,
            "_mpi_migrate_particles_gpu",
            _drop_and_count_migrating,
        )

    ps.attach()
    solver.one_step()
    _assert_interior_bidirectional_migration(
        ps._n_migrated_lo, ps._n_migrated_hi, solver.comm
    )

    res = ps.gauss_residual()
    q_now = ps.total_charge()
    assert res > 1.0e-3
    if mutation == "drop_migrating":
        assert abs(q_now) < abs(q_init)


def test_gpu_mpi_device_sharing_and_interior_ranks():
    """GPU MPI with ``particles_comm='mpi'`` works when ``size > n_gpus`` (ranks share GPUs via ``local_rank % n_gpus``) and exercises interior ranks (``has_lo`` and ``has_hi``)."""
    cp = pytest.importorskip("cupy")
    try:
        n_gpus = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    if n_gpus < 1:
        pytest.skip("Requires at least 1 CUDA GPU")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1. When size > n_gpus, explicit particles_comm="nccl" must raise RuntimeError
    #    collectively, while "auto" warns RuntimeWarning and selects "mpi".
    if size > n_gpus:
        g_chk = GridFIT3D(
            0.0, 0.06, 0.0, 0.06, 0.0, 0.12, 6, 6, 12, use_mpi=True, verbose=0
        )
        s_chk = SolverFIT3D(
            g_chk,
            cfln=0.45,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            use_mpi=True,
            use_gpu=True,
            verbose=0,
        )
        assert int(cp.cuda.Device().id) == (s_chk._local_rank % n_gpus)
        with pytest.raises(RuntimeError, match="particles_comm='nccl'"):
            ParticleSolver(
                s_chk, species=[Species(use_gpu=True)], shape=1, particles_comm="nccl"
            )
        with pytest.warns(RuntimeWarning, match="falling back to particles_comm='mpi'"):
            ps_auto = ParticleSolver(
                s_chk, species=[Species(use_gpu=True)], shape=1, particles_comm="auto"
            )
        assert ps_auto.particles_comm == "mpi"

    # 2. Force GPU migration buffer regrow (shrink pre-step cap to 1 via exact=True)
    #    with bidirectional crossings across all internal slab faces (exercising interior
    #    ranks with both has_lo=True and has_hi=True at -n 4).
    g_reg = GridFIT3D(
        0.0, 0.06, 0.0, 0.06, 0.0, 0.12, 6, 6, 12, use_mpi=True, verbose=0
    )
    s_reg = SolverFIT3D(
        g_reg,
        cfln=0.45,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_mpi=True,
        use_gpu=True,
        verbose=0,
    )
    # Place 2 +z and 2 -z particles at every internal slab face so >1 particle migrates
    # in both directions across each face on step 0, triggering _ensure_gpu_mpi_buffers regrow.
    faces = [k * (0.12 / size) for k in range(1, size)]
    z_list, uz_list = [], []
    for zf in faces:
        for _ in range(2):
            z_list.extend([zf - 1.5e-4, zf + 1.5e-4])
            uz_list.extend([0.35 * c, -0.35 * c])
    z0_r = np.array(z_list, dtype=np.float64)
    uz0_r = np.array(uz_list, dtype=np.float64)
    n_tot_r = z0_r.size
    sp_reg = Species(name="electron", charge=-e_charge, mass=m_e, use_gpu=True)
    sp_reg.add_particles(
        x=np.full(n_tot_r, 0.03),
        y=np.full(n_tot_r, 0.03),
        z=z0_r,
        ux=np.full(n_tot_r, 0.2 * c),
        uy=np.full(n_tot_r, -0.2 * c),
        uz=uz0_r,
        w=np.full(n_tot_r, 1.0e7),
    )
    ps_reg = ParticleSolver(
        s_reg,
        species=[sp_reg],
        shape=1,
        sort_every=0,
        particles_comm="mpi",
        capacity_headroom=0.0,
    ).attach()
    ps_reg._ensure_gpu_mpi_buffers(pack_cap=1, recv_lo_cap=1, recv_hi_cap=1)
    ps_reg._gpu_mig_pack_cap = 1
    ps_reg._gpu_mig_recv_lo_cap = 1
    ps_reg._gpu_mig_recv_hi_cap = 1
    pre_pack_cap = ps_reg._gpu_mig_pack_cap
    pre_recv_lo_cap = ps_reg._gpu_mig_recv_lo_cap
    pre_recv_hi_cap = ps_reg._gpu_mig_recv_hi_cap
    s_reg.one_step()
    assert ps_reg.gauss_residual() <= 1.0e-12
    sp_reg_all = ps_reg.mpi_gather_species(0, root=None)
    assert sp_reg_all.n_particles == n_tot_r
    assert ps_reg._gpu_mig_pack_cap > pre_pack_cap
    if rank > 0:
        assert ps_reg._gpu_mig_recv_lo_cap > pre_recv_lo_cap
    if rank < size - 1:
        assert ps_reg._gpu_mig_recv_hi_cap > pre_recv_hi_cap
    _assert_interior_bidirectional_migration(
        ps_reg._n_migrated_lo, ps_reg._n_migrated_hi, comm
    )

    # 3. Coupled V2 multi-step run with particles_comm="mpi" across all ranks (including
    #    interior ranks at -n 4 sharing GPUs via rank % n_gpus) must match serial within round-off.
    n_steps = 120
    par_mpi = _run_v2_coupled(
        graded=False,
        use_mpi=True,
        n_steps=n_steps,
        use_gpu=True,
        particles_comm="mpi",
    )
    assert par_mpi["particles_comm"] == "mpi"
    assert par_mpi["migrations"] >= 20
    assert par_mpi["gauss_res"] <= 1.0e-12
    _assert_interior_bidirectional_migration(
        par_mpi["sent_lo"], par_mpi["sent_hi"], comm
    )

    if rank != 0:
        return

    ser_cpu, pert_runs = _run_serial_perturbation_envelope(
        _run_v2_coupled,
        graded=False,
        use_mpi=False,
        n_steps=n_steps,
        use_gpu=False,
    )
    bound = derived_mpi_roundoff_bound(ser_cpu, pert_runs)
    ratio = _report_roundoff_comparison(
        "v2_gpu_mpi_shared_vs_cpu_serial", ser_cpu, par_mpi, bound
    )
    assert ratio <= 1.0
    _assert_v2_matches_reference(par_mpi, ser_cpu, bound=bound)


