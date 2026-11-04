# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Tests for PEC wall-closure current and Gauss's law near PEC walls for shape >= 2.

Covers:
1. Single macroparticle and 16-particle sheet injected by ``BeamInjector`` at a
   PEC ``z``-low wall with ``B_ext`` along ``z``, drifting through a ``6 x 1 x 40``
   vacuum box (periodic ``y``) and absorbed at the PEC ``z``-high wall: for
   ``p = 1, 2, 3``, ``gauss_residual(..., include_wall_planes=True) <= 1e-12`` at
   every recorded step through transit and after absorption. Also covers
   absorption on all six PEC domain faces (``x_low``, ``x_high``, ``y_low``,
   ``y_high``, ``z_low``, ``z_high``) and both ``GridFIT3D`` boundary layouts
   (``boundary_at_domain_edge=True`` and ``False``), as well as high-face
   injection (``z0 = zmax``, ``direction = -1``) and ``_injector_pec_wall_face``
   orientation validation.
2. Mutation check: monkeypatching the wall-closure current off for ``p = 2``
   causes ``gauss_residual(..., include_wall_planes=True)`` to exceed ``1e-2``
   (analytically ``1/8 = 0.125`` relative at ``k = 1`` after step 1).
3. Algebraic identity check of the ``_pec_wall_closure_1d`` helper: for ``p = 1``
   at ``zeta = 0`` (and ``zeta = Nz``), the wall-closure edge charge flux is
   identically zero (``atol = 0``).
4. ``_particles_q_bg`` after PEC-wall injection of ``N`` particles is nonzero
   only on the wall node plane ``k_w = 0`` with total equal to
   ``-injected_charge`` for ``p = 1, 2, 3`` to ``<= 1e-12`` relative; both the
   whole-domain Gauss sum ``sum(S_tilde d_tilde) == sum(q_nodes + q_bg)`` and
   the wall-node identity ``(S_tilde d_tilde)[k_w] == (q_nodes + q_bg)[k_w]``
   hold through injection, transit, and absorption to ``<= 1e-12`` relative
   while leaving ``E``, ``J_particles``, and particle trajectories bit-identical.
5. Energy-balance wall-closure work ``wall_closure_work()`` / ``"W_wall_closure"``:
   identically zero for ``p = 1`` with injection at ``zeta = 0``; nonzero for
   ``p = 2`` with the energy residual including ``W_wall_closure`` at least 10x
   smaller than without it; mesh-refinement check showing
   ``|W_wall_closure| / K_injected`` decreases as ``h -> h/2``; mutation check
   monkeypatching ``wall_closure_work`` to ``0.0`` breaks monotonic decrease and
   Richardson ``dt``-convergence for ``p = 2``.
6. Checkpoint persistence of ``W_wall_closure`` across restart, plus a
   ``RuntimeWarning`` when loading an older checkpoint archive that lacks
   ``cl_work_vec`` while PEC wall closure is active.
7. CPU vs GPU agreement of ``_particles_q_bg``, ``J_particles``, and
   ``W_wall_closure`` to ``<= 1e-12`` relative for ``p = 2`` (skipped without CuPy).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from scipy.constants import c as c_light
from scipy.constants import e as e_charge
from scipy.constants import m_e

import wakis.particles.solver as solver_mod
from wakis.gridFIT3D import GridFIT3D
from wakis.particles import (
    BeamInjector,
    ParticleSolver,
    Species,
    deposit_charge,
    gauss_residual,
)
from wakis.particles.solver import _pec_wall_closure_1d, compute_dual_divergence_d
from wakis.solverFIT3D import SolverFIT3D


def _run_z_wall_to_wall_transit(
    order: int,
    n_macro: int,
    *,
    n_steps_inject: int = 1,
) -> tuple[SolverFIT3D, ParticleSolver, Species, BeamInjector, list[tuple[int, float]]]:
    """Run a z-low PEC injection to z-high PEC absorption transit in a 6 x 1 x 40 box."""
    nx, ny, nz = 6, 1, 40
    xmin, xmax = -0.03, 0.03
    ymin, ymax = -0.005, 0.005
    zmin, zmax = 0.0, 0.04

    grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz, verbose=0)
    solver = SolverFIT3D(
        grid,
        cfln=0.5,
        bc_low=["pec", "periodic", "pec"],
        bc_high=["pec", "periodic", "pec"],
        use_stl=False,
        verbose=0,
    )

    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    gamma0 = 2.5
    beta0 = float(np.sqrt(1.0 - 1.0 / (gamma0 * gamma0)))
    vz0 = beta0 * c_light

    dt = float(solver.dt)
    t_inject_end = (n_steps_inject - 0.25) * dt

    def _pulse_current(t: float) -> float:
        return 0.2 if t <= t_inject_end else 0.0

    inj = BeamInjector(
        species=sp,
        z0=zmin,
        current=_pulse_current,
        gamma=gamma0,
        profile="sheet_lattice",
        radius=0.008,
        n_macro_per_step=n_macro,
        seed=1234,
    )
    ps = ParticleSolver(
        solver,
        species=[sp],
        injectors=[inj],
        pusher="boris",
        shape=order,
        gather="momentum",
        B_ext=(0.0, 0.0, 0.2),
    ).attach()

    n_transit_steps = int(np.ceil((zmax - zmin) / (vz0 * dt))) + 12
    history: list[tuple[int, float]] = []
    for step in range(1, n_transit_steps + 1):
        solver.one_step()
        res = gauss_residual(solver, sp, order=order, include_wall_planes=True)
        history.append((step, float(res)))

    return solver, ps, sp, inj, history


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("n_macro", [1, 16])
def test_pec_wall_injection_transit_and_absorption_gauss(order: int, n_macro: int):
    """Test 1: Single macroparticle and 16-particle sheet satisfy Gauss's law <= 1e-12
    on all interior vacuum planes at every step from PEC z-low birth through
    PEC z-high absorption for ``p = 1, 2, 3``.
    """
    _solver, ps, sp, inj, history = _run_z_wall_to_wall_transit(
        order=order, n_macro=n_macro
    )
    assert inj.emitter_kind == "pec_wall"
    assert inj.injected_count == n_macro
    assert sp.n_particles == 0
    assert ps.boundaries.absorbed_count["z_high"] == n_macro

    residuals = [r for _, r in history]
    max_res = max(residuals)
    step1_res = residuals[0]
    final_res = residuals[-1]
    print(
        f"[p={order}, N={n_macro:2d}] step1={step1_res:.3e}, "
        f"max_transit={max_res:.3e}, post_absorb={final_res:.3e}"
    )
    assert max_res <= 1e-12, (
        f"Gauss residual exceeded 1e-12 for p={order}, N={n_macro}: max={max_res:.3e}"
    )
    assert ps.gauss_residual(include_wall_planes=True) <= 1e-12


def test_pec_wall_closure_mutation_fails(monkeypatch):
    """Test 2: Disabling PEC wall-closure current for ``p = 2`` leaves the ``1/8``
    birth stencil tail at ``k = 1`` uncompensated in ``W_z[0]`` and produces a
    relative Gauss residual ``> 1e-2`` (analytically ``~1/8 = 0.125``).
    """
    # Mutation A: module flag _ENABLE_PEC_WALL_CLOSURE = False
    with monkeypatch.context() as m:
        m.setattr(solver_mod, "_ENABLE_PEC_WALL_CLOSURE", False)
        _, _, _, _, history_flag = _run_z_wall_to_wall_transit(order=2, n_macro=1)
        step1_res_flag = history_flag[0][1]
        max_res_flag = max(r for _, r in history_flag)
        print(
            f"[mutation flag=False, p=2, N=1] step1_res={step1_res_flag:.6e}, "
            f"max_res={max_res_flag:.6e}"
        )
        assert step1_res_flag > 1e-2
        assert max_res_flag > 1e-2

    # Mutation B: patch _pec_wall_closure_1d helper to return zeros while keeping
    # wall-node-only _particles_q_bg bookkeeping
    with monkeypatch.context() as m:
        m.setattr(
            solver_mod,
            "_pec_wall_closure_1d",
            lambda q_vis, axis, side, k_w=None, *, event="inject", xp=np: xp.zeros_like(
                q_vis
            ),
        )
        _, _, _, _, history_zero = _run_z_wall_to_wall_transit(order=2, n_macro=16)
        step1_res_zero = history_zero[0][1]
        max_res_zero = max(r for _, r in history_zero)
        print(
            f"[mutation zero_dW, p=2, N=16] step1_res={step1_res_zero:.6e}, "
            f"max_res={max_res_zero:.6e}"
        )
        assert step1_res_zero > 1e-2
        assert max_res_zero > 1e-2


def test_pec_wall_closure_1d_helper_p1_identity():
    """Test 3: Algebraic identity check of ``_pec_wall_closure_1d`` for ``p = 1``.

    Because a ``p = 1`` (CIC) particle on the wall plane ``zeta = 0`` (or
    ``zeta = Nz``) has zero B-spline weight on interior vacuum nodes ``k != k_w``,
    the helper ``_pec_wall_closure_1d`` returns an identically zero array
    (``atol = 0``).
    """
    nx, ny, nz = 6, 4, 12
    q_vis_low = np.zeros((nx, ny, nz + 1), dtype=np.float64, order="F")
    xi = np.array([1.2, 2.7, 4.1], dtype=np.float64)
    eta = np.array([0.5, 1.8, 2.3], dtype=np.float64)
    zeta_low = np.zeros(3, dtype=np.float64)
    qw = np.array([-1.0e-12, -2.5e-12, -0.8e-12], dtype=np.float64)

    deposit_charge(
        xi,
        eta,
        zeta_low,
        qw,
        nx,
        ny,
        nz + 1,
        order=1,
        periodic=(False, True, False),
        q_nodes=q_vis_low,
        xp=np,
    )
    dW_low = _pec_wall_closure_1d(q_vis_low, axis=2, side="low", k_w=0, event="inject")
    assert np.max(np.abs(dW_low)) == 0.0
    assert np.all(dW_low == 0.0)

    q_vis_high = np.zeros((nx, ny, nz + 1), dtype=np.float64, order="F")
    zeta_high = np.full(3, float(nz), dtype=np.float64)
    deposit_charge(
        xi,
        eta,
        zeta_high,
        qw,
        nx,
        ny,
        nz + 1,
        order=1,
        periodic=(False, True, False),
        q_nodes=q_vis_high,
        xp=np,
    )
    dW_high = _pec_wall_closure_1d(
        q_vis_high, axis=2, side="high", k_w=nz, event="absorb"
    )
    assert np.max(np.abs(dW_high)) == 0.0
    assert np.all(dW_high == 0.0)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_q_bg_after_pec_wall_injection_on_wall_plane_only(order: int):
    """Test 4: After injecting ``N`` particles at a PEC wall, ``_particles_q_bg``
    is nonzero only on the wall node plane ``k_w = 0``, with total equal to
    ``-injected_charge`` for ``p = 1, 2, 3`` to ``<= 1e-12`` relative.
    """
    nx, ny, nz = 6, 1, 20
    grid = GridFIT3D(-0.03, 0.03, -0.005, 0.005, 0.0, 0.02, nx, ny, nz, verbose=0)
    solver = SolverFIT3D(
        grid,
        cfln=0.5,
        bc_low=["pec", "periodic", "pec"],
        bc_high=["pec", "periodic", "pec"],
        use_stl=False,
        verbose=0,
    )
    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    n_macro = 16
    inj = BeamInjector(
        species=sp,
        z0=0.0,
        current=0.25,
        gamma=2.2,
        profile="sheet_lattice",
        radius=0.008,
        n_macro_per_step=n_macro,
        seed=99,
    )
    ps = ParticleSolver(
        solver,
        species=[sp],
        injectors=[inj],
        pusher="boris",
        shape=order,
        gather="momentum",
    ).attach()

    solver.one_step()
    assert sp.n_particles == n_macro
    q_bg = np.asarray(solver._particles_q_bg)

    # Nonzero only on the wall node plane k_w = 0:
    assert np.all(q_bg[:, :, 1:] == 0.0)
    assert np.any(q_bg[:, :, 0] != 0.0)

    q_inj = float(inj.injected_charge)
    expected_q_bg_total = -q_inj
    actual_q_bg_total = float(np.sum(q_bg[:, :, 0]))
    rel_err = abs(actual_q_bg_total - expected_q_bg_total) / abs(expected_q_bg_total)
    print(
        f"[q_bg p={order}] sum(q_bg)={actual_q_bg_total:.6e}, "
        f"expected={expected_q_bg_total:.6e}, rel_err={rel_err:.3e}"
    )
    assert rel_err <= 1e-12
    assert ps.gauss_residual(include_wall_planes=True) <= 1e-12


@pytest.mark.parametrize("order", [1, 2, 3])
def test_whole_domain_and_wall_node_gauss_identity_and_field_invariance(
    order: int, monkeypatch
):
    """Verify whole-domain ``sum(S_tilde d_tilde) == sum(q_nodes + q_bg)`` and
    wall-node ``(S_tilde d_tilde)[k_w] == (q_nodes + q_bg)[k_w]`` through
    injection, transit, and absorption for ``p = 1, 2, 3``, and assert that
    ``E``, ``J_particles``, and particle dynamics are bit-identical to the
    unmodified ``W``/``J`` deposition.
    """

    def _run_case(use_legacy_q_bg: bool, m_ctx=None):
        nx, ny, nz = 1, 1, 40
        grid = GridFIT3D(-0.005, 0.005, -0.005, 0.005, 0.0, 0.04, nx, ny, nz, verbose=0)
        solver = SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            use_stl=False,
            verbose=0,
        )
        sp = Species(name="electron", charge=-e_charge, mass=m_e)
        gamma0 = 2.5
        beta0 = float(np.sqrt(1.0 - 1.0 / (gamma0 * gamma0)))
        vz0 = beta0 * c_light
        dt = float(solver.dt)
        t_inject_end = 0.75 * dt

        inj = BeamInjector(
            species=sp,
            z0=0.0,
            current=lambda t: 0.2 if t <= t_inject_end else 0.0,
            gamma=gamma0,
            profile="sheet_lattice",
            radius=0.001,
            n_macro_per_step=1,
            seed=1234,
        )
        ps = ParticleSolver(
            solver,
            species=[sp],
            injectors=[inj],
            pusher="boris",
            shape=order,
            gather="momentum",
        ).attach()

        if use_legacy_q_bg:
            orig_apply = ParticleSolver._apply_pec_wall_event

            def _legacy_apply(self_ps, q_vis, axis, side, *, event, W_tuple, q_bg):
                orig_apply(
                    self_ps,
                    q_vis,
                    axis,
                    side,
                    event=event,
                    W_tuple=W_tuple,
                    q_bg=q_bg,
                )
                k_w = self_ps._pec_wall_node_index(axis, side)
                sl_w = [slice(None), slice(None), slice(None)]
                sl_w[axis] = k_w
                sl_w_t = tuple(sl_w)
                q_ghost = self_ps.xp.sum(q_vis, axis=axis) - q_vis[sl_w_t]
                if event == "inject":
                    q_bg[sl_w_t] += q_ghost
                else:
                    q_bg[sl_w_t] -= q_ghost

            m_ctx.setattr(ParticleSolver, "_apply_pec_wall_event", _legacy_apply)

        def _measure_residuals():
            div_d = np.asarray(compute_dual_divergence_d(solver))
            q_nodes = np.zeros((solver.Nx, solver.Ny, solver.Nz), order="F")
            if sp.n_particles > 0:
                xi, eta, zeta = ps.grid_map.to_logical(sp.x, sp.y, sp.z)
                deposit_charge(
                    xi,
                    eta,
                    zeta,
                    sp.charge * sp.w,
                    solver.Nx,
                    solver.Ny,
                    solver.Nz,
                    order=order,
                    periodic=ps.boundaries.periodic,
                    q_nodes=q_nodes,
                )
            q_tot = q_nodes + np.asarray(solver._particles_q_bg)
            q_ref = abs(float(inj.injected_charge))
            whole_rel = abs(float(np.sum(div_d) - np.sum(q_tot))) / q_ref
            wall_lo_rel = abs(float(div_d[0, 0, 0] - q_tot[0, 0, 0])) / q_ref
            wall_hi_rel = (
                abs(float(div_d[0, 0, solver.Nz - 1] - q_tot[0, 0, solver.Nz - 1]))
                / q_ref
            )
            return {
                "whole": whole_rel,
                "wall_lo": wall_lo_rel,
                "wall_hi": wall_hi_rel,
            }

        n_transit_steps = int(np.ceil(0.04 / (vz0 * dt))) + 12
        snaps = {}
        e_mid = None
        j_mid = None
        z_mid = None
        uz_mid = None
        for step in range(1, n_transit_steps + 1):
            solver.one_step()
            if step == 3:
                snaps["post_inject"] = _measure_residuals()
            elif step == 20:
                snaps["mid_transit"] = _measure_residuals()
                e_mid = solver.E.toarray().copy()
                j_mid = ps.J_particles.copy()
                z_mid = sp.z.copy()
                uz_mid = sp.uz.copy()
        snaps["post_absorb"] = _measure_residuals()
        return (
            snaps,
            solver.E.toarray().copy(),
            ps.J_particles.copy(),
            e_mid,
            j_mid,
            z_mid,
            uz_mid,
        )

    with monkeypatch.context() as m_ctx:
        snaps_old, e_old_end, j_old_end, e_old_mid, j_old_mid, z_old_mid, uz_old_mid = (
            _run_case(use_legacy_q_bg=True, m_ctx=m_ctx)
        )
    snaps_new, e_new_end, j_new_end, e_new_mid, j_new_mid, z_new_mid, uz_new_mid = (
        _run_case(use_legacy_q_bg=False)
    )

    for stage in ("post_inject", "mid_transit", "post_absorb"):
        o = snaps_old[stage]
        n = snaps_new[stage]
        print(
            f"[p={order} {stage}] BEFORE whole={o['whole']:.3e} kw0={o['wall_lo']:.3e} "
            f"kwN={o['wall_hi']:.3e} | AFTER whole={n['whole']:.3e} "
            f"kw0={n['wall_lo']:.3e} kwN={n['wall_hi']:.3e}"
        )
        assert n["whole"] <= 1e-12
        assert n["wall_lo"] <= 1e-12
        assert n["wall_hi"] <= 1e-12
        if order >= 2 and stage in ("post_inject", "mid_transit"):
            assert o["whole"] > 0.1
            assert o["wall_lo"] > 0.1
        if order >= 2 and stage == "post_absorb":
            assert o["wall_lo"] > 0.1
            assert o["wall_hi"] > 0.1

    # Bit-identical E, J_particles, and particle trajectories at mid-transit and end:
    assert np.array_equal(e_new_mid, e_old_mid)
    assert np.array_equal(j_new_mid, j_old_mid)
    assert np.array_equal(z_new_mid, z_old_mid)
    assert np.array_equal(uz_new_mid, uz_old_mid)
    assert np.array_equal(e_new_end, e_old_end)
    assert np.array_equal(j_new_end, j_old_end)


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize(
    "face", ["x_low", "x_high", "y_low", "y_high", "z_low", "z_high"]
)
def test_pec_wall_absorption_all_six_faces(order: int, face: str):
    """Verify axis-agnostic PEC wall-closure current on absorption at all 6 PEC faces."""
    nx, ny, nz = 10, 10, 10
    grid = GridFIT3D(0.0, 0.05, 0.0, 0.05, 0.0, 0.05, nx, ny, nz, verbose=0)
    solver = SolverFIT3D(
        grid,
        cfln=0.5,
        bc_low=["pec", "pec", "pec"],
        bc_high=["pec", "pec", "pec"],
        use_stl=False,
        verbose=0,
    )
    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    u0 = 1.5 * c_light
    pos = [0.025, 0.025, 0.025]
    vel = [0.0, 0.0, 0.0]
    axis_map = {"x": 0, "y": 1, "z": 2}
    ax_char, side = face.split("_")
    ax = axis_map[ax_char]
    pos[ax] = 0.007 if side == "low" else 0.043
    vel[ax] = -u0 if side == "low" else +u0

    sp.add_particles(
        x=np.array([pos[0]]),
        y=np.array([pos[1]]),
        z=np.array([pos[2]]),
        ux=np.array([vel[0]]),
        uy=np.array([vel[1]]),
        uz=np.array([vel[2]]),
        w=np.array([1.0e6]),
    )
    ps = ParticleSolver(
        solver, species=[sp], pusher="boris", shape=order, gather="momentum"
    ).attach()

    max_res = 0.0
    for _ in range(12):
        solver.one_step()
        max_res = max(max_res, ps.gauss_residual(include_wall_planes=True))

    assert sp.n_particles == 0
    assert ps.boundaries.absorbed_count[face] == 1
    print(f"[absorb face={face}, p={order}] max_gauss_res={max_res:.3e}")
    assert max_res <= 1e-12


@pytest.mark.parametrize("boundary_at_domain_edge", [True, False])
@pytest.mark.parametrize("order", [1, 2])
def test_pec_wall_absorption_z_high_grid_layouts(
    boundary_at_domain_edge: bool, order: int
):
    """Verify PEC ``z_high`` absorption with ``boundary_at_domain_edge=True`` and ``False``
    for ``p = 1, 2`` raises no ``IndexError`` and satisfies Gauss's law ``<= 1e-12``.
    """
    grid = GridFIT3D(
        0.0,
        0.06,
        0.0,
        0.06,
        0.0,
        0.10,
        6,
        6,
        20,
        boundary_at_domain_edge=boundary_at_domain_edge,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        cfln=0.4,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_stl=False,
        verbose=0,
    )
    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    z_wall_hi = float(grid.z[solver.Nz - 1])
    sp.add_particles(
        x=np.array([0.025]),
        y=np.array([0.035]),
        z=np.array([z_wall_hi - 2.2 * float(grid.dz[0])]),
        ux=np.array([0.0]),
        uy=np.array([0.0]),
        uz=np.array([1.2 * c_light]),
        w=np.array([1.0e8]),
    )
    ps = ParticleSolver(
        solver,
        species=[sp],
        pusher="boris",
        shape=order,
        gather="momentum",
        track_energy=True,
    ).attach()

    assert ps._pec_wall_node_index(2, "high") == solver.Nz - 1
    max_res = 0.0
    for _ in range(25):
        solver.one_step()
        max_res = max(max_res, ps.gauss_residual(include_wall_planes=True))

    assert sp.n_particles == 0
    assert ps.boundaries.absorbed_count["z_high"] == 1
    print(
        f"[z_high layout boundary_at_domain_edge={boundary_at_domain_edge}, p={order}] "
        f"max_gauss_res={max_res:.3e}"
    )
    assert max_res <= 1e-12


@pytest.mark.parametrize("order", [1, 2, 3])
def test_pec_wall_high_face_injection_and_orientation_validation(order: int):
    """Verify ``BeamInjector`` emitting toward ``-z`` from ``z = zmax`` (``z_high``
    PEC wall) satisfies Gauss's law ``<= 1e-12`` through transit and ``z_low``
    absorption, and that ``_injector_pec_wall_face`` raises ``ValueError`` for
    unsupported orientations.
    """
    nx, ny, nz = 6, 1, 24
    zmin, zmax = 0.0, 0.024
    grid = GridFIT3D(-0.03, 0.03, -0.005, 0.005, zmin, zmax, nx, ny, nz, verbose=0)
    solver = SolverFIT3D(
        grid,
        cfln=0.5,
        bc_low=["pec", "periodic", "pec"],
        bc_high=["pec", "periodic", "pec"],
        use_stl=False,
        verbose=0,
    )
    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    gamma0 = 2.5
    beta0 = float(np.sqrt(1.0 - 1.0 / (gamma0 * gamma0)))
    vz0 = beta0 * c_light
    dt = float(solver.dt)
    t_inject_end = 0.75 * dt

    inj = BeamInjector(
        species=sp,
        z0=zmax,
        current=lambda t: 0.2 if t <= t_inject_end else 0.0,
        gamma=gamma0,
        direction=-1,
        profile="sheet_lattice",
        radius=0.008,
        n_macro_per_step=8,
        seed=321,
    )
    ps = ParticleSolver(
        solver,
        species=[sp],
        injectors=[inj],
        pusher="boris",
        shape=order,
        gather="momentum",
        track_energy=True,
    ).attach()

    assert ps._injector_pec_wall_face(inj) == (2, "high")
    n_steps = int(np.ceil((zmax - zmin) / (vz0 * dt))) + 10
    max_res = 0.0
    for _ in range(n_steps):
        solver.one_step()
        max_res = max(max_res, ps.gauss_residual(include_wall_planes=True))

    assert inj.emitter_kind == "pec_wall"
    assert inj.injected_count == 8
    assert sp.n_particles == 0
    assert ps.boundaries.absorbed_count["z_low"] == 8
    print(f"[high-face inject p={order}] max_gauss_res={max_res:.3e}")
    assert max_res <= 1e-12

    if order == 1:

        class _DummyInj:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        with pytest.raises(ValueError, match="axis"):
            ps._injector_pec_wall_face(_DummyInj(axis=0, z0=0.0))
        with pytest.raises(ValueError, match="axis"):
            ps._injector_pec_wall_face(_DummyInj(axis=5, side="low"))
        with pytest.raises(ValueError, match="side"):
            ps._injector_pec_wall_face(_DummyInj(axis=2, side="middle"))
        with pytest.raises(ValueError, match="direction"):
            ps._injector_pec_wall_face(_DummyInj(axis=2, z0=zmin, direction=-1))
        with pytest.raises(ValueError, match="direction"):
            ps._injector_pec_wall_face(_DummyInj(axis=2, z0=zmax, direction=1))
        with pytest.raises(ValueError, match="z0"):
            ps._injector_pec_wall_face(
                _DummyInj(axis=2, z0=0.5 * (zmin + zmax), direction=1)
            )


def test_wall_closure_work_p1_zero_p2_residual_and_mutation(monkeypatch):
    """Test 5: Verify ``wall_closure_work`` / ``W_wall_closure`` in ``energy_balance()``.

    Covers:
    (a) For ``p = 1`` with injection at ``zeta = 0``, ``W_wall_closure == 0.0`` exactly.
    (b) For ``p = 2``, ``W_wall_closure`` is nonzero and the energy residual with the
        term is at least 10x smaller than the residual without it.
    (c) Mutation: monkeypatching ``ParticleSolver.wall_closure_work`` to return ``0.0``
        destroys monotonic decrease and Richardson ``dt``-convergence for ``p = 2``.
    """
    from tests.test_083_particles_beams import (
        _assert_richardson_order,
        _passes_richardson_order,
        _run_dt_closure_sweep,
    )

    # (a) p = 1 with injection at zeta = 0 (z0 = 0.0) -> W_wall_closure == 0.0 exactly
    nx, ny, nz = 6, 1, 20
    grid_p1 = GridFIT3D(-0.03, 0.03, -0.005, 0.005, 0.0, 0.04, nx, ny, nz, verbose=0)
    solver_p1 = SolverFIT3D(
        grid_p1,
        cfln=0.5,
        bc_low=["pec", "periodic", "pec"],
        bc_high=["pec", "periodic", "pec"],
        use_stl=False,
        verbose=0,
    )
    sp_p1 = Species(name="electron", charge=-e_charge, mass=m_e)
    inj_p1 = BeamInjector(
        species=sp_p1,
        z0=0.0,
        current=0.25,
        gamma=2.0,
        profile="sheet_lattice",
        radius=0.008,
        n_macro_per_step=16,
        seed=42,
    )
    ps_p1 = ParticleSolver(
        solver_p1,
        species=[sp_p1],
        injectors=[inj_p1],
        pusher="boris",
        shape=1,
        gather="momentum",
        track_energy=True,
    ).attach()
    for _ in range(8):
        solver_p1.one_step()
    eb_p1 = ps_p1.energy_balance()
    assert ps_p1.wall_closure_work() == 0.0
    assert eb_p1["W_wall_closure"] == 0.0
    print(
        f"[energy p=1, zeta=0] W_wall_closure={eb_p1['W_wall_closure']:.6e}, "
        f"residual={eb_p1['residual']:.6e}"
    )

    # (b) p = 2: closure work is nonzero and residual with term is >= 10x smaller than without
    dts_pre, _, ebs_pre, kinds_pre = _run_dt_closure_sweep(
        z0=0.0,
        z_col=0.075,
        n_steps_base=8,
        cfl_factors=(0.4, 0.2),
        shape=2,
    )
    assert all(k == "pec_wall" for k in kinds_pre)
    for dt_val, eb in zip(dts_pre, ebs_pre):
        w_cl = eb["W_wall_closure"]
        res_with = abs(eb["residual"])
        res_without = abs(eb["residual"] + w_cl)
        print(
            f"[energy p=2, dt={dt_val:.3e}] W_wall_closure={w_cl:.6e}, "
            f"res_with={res_with:.6e}, res_without={res_without:.6e}, "
            f"ratio={res_without / res_with:.2f}x"
        )
        assert w_cl != 0.0
        assert res_with * 10.0 <= res_without

    # Verify the p = 2 dt-convergence check from test_083 passes when wall_closure_work is active
    dts_p2, ress_p2, ebs_p2, _ = _run_dt_closure_sweep(
        z0=0.0,
        z_col=0.010,
        n_steps_base=10,
        cfl_factors=(0.8, 0.4, 0.2, 0.1),
        init_cavity_ez=1.0e4,
        shape=2,
        gamma=2.0,
    )
    print(f"[energy p=2 sweep] ress_with={ress_p2}")
    assert all(eb["W_wall_closure"] != 0.0 for eb in ebs_p2)
    assert np.all(ress_p2 > 0.0)
    assert np.all(np.diff(ress_p2) < 0.0)
    _assert_richardson_order(dts_p2, ress_p2, order=1.0)

    # (c) Mutation: monkeypatch wall_closure_work -> 0.0 and verify that the
    # mutated residuals are neither monotonically decreasing nor Richardson-convergent
    with monkeypatch.context() as m:
        m.setattr(ParticleSolver, "wall_closure_work", lambda self_ps: 0.0)
        dts_mut, ress_mut, _, _ = _run_dt_closure_sweep(
            z0=0.0,
            z_col=0.010,
            n_steps_base=10,
            cfl_factors=(0.8, 0.4, 0.2, 0.1),
            init_cavity_ez=1.0e4,
            shape=2,
            gamma=2.0,
        )
        print(f"[mutation wall_closure_work=0, p=2] ress_mut={ress_mut}")
        assert not np.all(np.diff(ress_mut) < 0.0)
        assert not _passes_richardson_order(dts_mut, ress_mut, order=1.0)


def test_wall_closure_work_mesh_refinement_p2():
    """Verify ``|W_wall_closure| / K_injected`` decreases under mesh refinement
    ``h -> h / 2`` at fixed physical box, physics, and physical time ``T`` for ``p = 2``.
    """
    ratios = []
    hs = []
    for scale in (1, 2):
        nx, ny, nz = 1, 1, 20 * scale
        grid = GridFIT3D(-0.005, 0.005, -0.005, 0.005, 0.0, 0.04, nx, ny, nz, verbose=0)
        solver = SolverFIT3D(
            grid,
            cfln=0.4,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            use_stl=False,
            verbose=0,
        )
        if scale == 1:
            dt_coarse = float(solver.dt)
        else:
            solver.dt = 0.5 * dt_coarse

        sp = Species(name="electron", charge=-e_charge, mass=m_e)
        inj = BeamInjector(
            species=sp,
            z0=0.0,
            current=0.25,
            gamma=2.0,
            profile="sheet_lattice",
            radius=0.001,
            n_macro_per_step=1,
            seed=42,
        )
        ps = ParticleSolver(
            solver,
            species=[sp],
            injectors=[inj],
            pusher="boris",
            shape=2,
            gather="momentum",
            track_energy=True,
        ).attach()

        n_steps = 20 * scale
        for _ in range(n_steps):
            solver.one_step()
        eb = ps.energy_balance()
        ratio = abs(eb["W_wall_closure"]) / eb["K_injected"]
        dz = float(grid.dz[0])
        hs.append(dz)
        ratios.append(ratio)

    order_obs = float(np.log2(ratios[0] / ratios[1]))
    print(
        f"[refinement p=2] h={hs[0]:.4e} -> |W_cl|/K_inj={ratios[0]:.6e}; "
        f"h/2={hs[1]:.4e} -> |W_cl|/K_inj={ratios[1]:.6e}; "
        f"observed_order={order_obs:.3f}"
    )
    assert ratios[1] < ratios[0]


def _build_p2_energy_case():
    """p = 2 pec_wall injection with energy tracking (closure work nonzero from step 1)."""
    grid = GridFIT3D(-0.03, 0.03, -0.005, 0.005, 0.0, 0.04, 6, 1, 20, verbose=0)
    solver = SolverFIT3D(
        grid,
        cfln=0.5,
        bc_low=["pec", "periodic", "pec"],
        bc_high=["pec", "periodic", "pec"],
        use_stl=False,
        verbose=0,
    )
    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    inj = BeamInjector(
        species=sp,
        z0=0.0,
        current=0.25,
        gamma=2.0,
        profile="sheet_lattice",
        radius=0.008,
        n_macro_per_step=16,
        seed=42,
    )
    ps = ParticleSolver(
        solver,
        species=[sp],
        injectors=[inj],
        pusher="boris",
        shape=2,
        gather="momentum",
        track_energy=True,
    ).attach()
    return solver, ps


def test_wall_closure_work_survives_checkpoint_restart(tmp_path, monkeypatch):
    """Test 6: ``W_wall_closure`` and the residual are bitwise continuous across a restart.

    Mutation: a restore that drops the saved closure-work accumulator must break equality.
    """
    import wakis.particles.checkpoint as ckpt_mod

    n_a, n_b = 6, 6
    solver, ps = _build_p2_energy_case()
    for _ in range(n_a):
        solver.one_step()
    ps.save_checkpoint(str(tmp_path / "ck"))
    for _ in range(n_b):
        solver.one_step()
    eb_ref = ps.energy_balance()
    assert eb_ref["W_wall_closure"] != 0.0

    def _restart(mutate: bool):
        s2, ps2 = _build_p2_energy_case()
        orig = ckpt_mod._restore_rank_arrays

        def _restore(ps_t, data):
            orig(ps_t, data)
            if mutate:
                ps_t._cl_work_vec.fill(0.0)

        monkeypatch.setattr(ckpt_mod, "_restore_rank_arrays", _restore)
        ps2.load_checkpoint(str(tmp_path / "ck"))
        monkeypatch.setattr(ckpt_mod, "_restore_rank_arrays", orig)
        for _ in range(n_b):
            s2.one_step()
        return ps2.energy_balance()

    eb_rs = _restart(False)
    eb_mut = _restart(True)
    print(
        f"[restart p=2] W_cl ref={eb_ref['W_wall_closure']:.6e} restart={eb_rs['W_wall_closure']:.6e} "
        f"mutated={eb_mut['W_wall_closure']:.6e}; residual ref={eb_ref['residual']:.6e} "
        f"restart={eb_rs['residual']:.6e} mutated={eb_mut['residual']:.6e}"
    )
    for key in ("W_wall_closure", "residual", "field_work", "W_field", "K_particles"):
        assert eb_rs[key] == eb_ref[key], key
    assert eb_mut["W_wall_closure"] != eb_ref["W_wall_closure"]
    assert eb_mut["residual"] != eb_ref["residual"]


def test_old_checkpoint_without_cl_work_vec_warns(tmp_path):
    """Loading a checkpoint that lacks ``cl_work_vec`` when ``track_energy=True``
    and PEC wall closure is active emits a ``RuntimeWarning`` and restarts
    ``W_wall_closure`` from zero.
    """
    ckpt_dir = tmp_path / "ck_legacy"
    solver, ps = _build_p2_energy_case()
    for _ in range(4):
        solver.one_step()
    w_cl_before = ps.wall_closure_work()
    assert w_cl_before != 0.0
    ps.save_checkpoint(str(ckpt_dir))

    manifest_path = ckpt_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rank_fname = next(iter(manifest["files"].keys()))
    rank_npz = ckpt_dir / rank_fname
    with np.load(rank_npz, allow_pickle=False) as data:
        arrays = {
            k: data[k]
            for k in data.files
            if k not in ("cl_work_vec", "cl_J", "cl_pending")
        }
    np.savez(rank_npz, **arrays)
    manifest["files"][rank_fname] = hashlib.sha256(rank_npz.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _s2, ps2 = _build_p2_energy_case()
    with pytest.warns(RuntimeWarning, match="W_wall_closure"):
        ps2.load_checkpoint(str(ckpt_dir))
    assert ps2.wall_closure_work() == 0.0


def test_pec_wall_closure_cpu_vs_gpu_p2():
    """Compare CPU and GPU after injection, transit, and absorption for ``p = 2``:
    ``_particles_q_bg``, ``J_particles``, and ``W_wall_closure`` agree to ``<= 1e-12`` relative.
    """
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA GPU device available")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    def _run(use_gpu: bool):
        nx, ny, nz = 6, 1, 24
        zmin, zmax = 0.0, 0.024
        grid = GridFIT3D(-0.03, 0.03, -0.005, 0.005, zmin, zmax, nx, ny, nz, verbose=0)
        solver = SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["pec", "periodic", "pec"],
            bc_high=["pec", "periodic", "pec"],
            use_stl=False,
            use_gpu=use_gpu,
            verbose=0,
        )
        sp = Species(name="electron", charge=-e_charge, mass=m_e)
        gamma0 = 2.5
        beta0 = float(np.sqrt(1.0 - 1.0 / (gamma0 * gamma0)))
        vz0 = beta0 * c_light
        dt = float(solver.dt)
        t_inject_end = 1.75 * dt

        inj = BeamInjector(
            species=sp,
            z0=zmin,
            current=lambda t: 0.25 if t <= t_inject_end else 0.0,
            gamma=gamma0,
            profile="sheet_lattice",
            radius=0.008,
            n_macro_per_step=16,
            seed=777,
        )
        ps = ParticleSolver(
            solver,
            species=[sp],
            injectors=[inj],
            pusher="boris",
            shape=2,
            gather="momentum",
            track_energy=True,
        ).attach()

        n_steps = int(np.ceil((zmax - zmin) / (vz0 * dt))) + 8
        j_accum = np.zeros(3 * solver.N, dtype=np.float64)
        for _ in range(n_steps):
            solver.one_step()
            j_step = (
                cp.asnumpy(ps.J_particles)
                if hasattr(ps.J_particles, "get")
                else np.asarray(ps.J_particles)
            )
            j_accum += np.abs(j_step)

        q_bg = (
            cp.asnumpy(solver._particles_q_bg)
            if hasattr(solver._particles_q_bg, "get")
            else np.asarray(solver._particles_q_bg).copy()
        )
        j_final = (
            cp.asnumpy(ps.J_particles)
            if hasattr(ps.J_particles, "get")
            else np.asarray(ps.J_particles).copy()
        )
        eb = ps.energy_balance()
        return (
            q_bg,
            j_accum,
            j_final,
            eb["W_wall_closure"],
            ps.boundaries.absorbed_count["z_high"],
        )

    q_bg_cpu, j_acc_cpu, j_fin_cpu, w_cl_cpu, n_abs_cpu = _run(False)
    q_bg_gpu, j_acc_gpu, j_fin_gpu, w_cl_gpu, n_abs_gpu = _run(True)

    assert n_abs_cpu == 32
    assert n_abs_gpu == 32
    q_bg_rel = float(np.linalg.norm(q_bg_cpu - q_bg_gpu) / np.linalg.norm(q_bg_cpu))
    j_acc_rel = float(np.linalg.norm(j_acc_cpu - j_acc_gpu) / np.linalg.norm(j_acc_cpu))
    j_fin_diff = float(np.max(np.abs(j_fin_cpu - j_fin_gpu)))
    w_cl_rel = abs(w_cl_cpu - w_cl_gpu) / abs(w_cl_cpu)
    print(
        f"[CPU vs GPU p=2] q_bg_rel={q_bg_rel:.3e}, j_acc_rel={j_acc_rel:.3e}, "
        f"j_fin_diff={j_fin_diff:.3e}, w_cl_rel={w_cl_rel:.3e}"
    )
    assert q_bg_rel <= 1e-12
    assert j_acc_rel <= 1e-12
    assert j_fin_diff <= 1e-12
    assert w_cl_rel <= 1e-12


def test_interior_mpi_rank_has_pec_domain_wall_from_global_bc():
    """Verify ``_compute_has_pec_domain_wall`` returns ``True`` on an interior MPI
    rank whose local ``bc_low``/``bc_high`` on ``z`` are ``'mpi'`` while
    ``solver._global_bc_z`` is ``('pec', 'pec')`` and ``x``/``y`` are periodic.
    """
    grid = GridFIT3D(-0.03, 0.03, -0.03, 0.03, 0.0, 0.04, 6, 6, 12, verbose=0)
    solver = SolverFIT3D(
        grid,
        cfln=0.45,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_stl=False,
        verbose=0,
    )
    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    ps = ParticleSolver(
        solver, species=[sp], pusher="boris", shape=1, track_energy=True
    ).attach()
    assert ps._has_pec_domain_wall is True

    # Simulate an interior rank (0 < rank < size - 1) where z faces are "mpi"
    # and only solver._global_bc_z records the global ("pec", "pec") z walls:
    ps.boundaries.bc_low = ("periodic", "periodic", "mpi")
    ps.boundaries.bc_high = ("periodic", "periodic", "mpi")
    solver._global_bc_z = ("pec", "pec")
    assert ps._compute_has_pec_domain_wall() is True
    solver._global_bc_z = ("periodic", "periodic")
    assert ps._compute_has_pec_domain_wall() is False
