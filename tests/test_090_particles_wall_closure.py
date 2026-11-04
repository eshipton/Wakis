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
   ``y_high``, ``z_low``, ``z_high``).
2. Mutation check: monkeypatching the wall-closure current off for ``p = 2``
   causes ``gauss_residual(..., include_wall_planes=True)`` to exceed ``1e-2``
   (analytically ``1/8 = 0.125`` relative at ``k = 1`` after step 1).
3. ``p = 1`` wall-closure current at ``zeta = 0`` (and ``zeta = Nz``) is
   identically zero (``atol = 0``).
4. ``_particles_q_bg`` after PEC-wall injection of ``N`` particles is nonzero
   only on the wall node plane ``k_w = 0``, with total equal to
   ``-injected_charge`` (for ``p = 1``) and ``-sum(q_vis)`` (for ``p = 1, 2, 3``)
   to ``<= 1e-12`` relative.
5. Energy-balance wall-closure work ``wall_closure_work()`` / ``"W_wall_closure"``:
   identically zero for ``p = 1`` with injection at ``zeta = 0``; nonzero for
   ``p = 2`` with the energy residual including ``W_wall_closure`` at least 10x
   smaller than without it; mutation check monkeypatching ``wall_closure_work``
   to ``0.0`` causes the ``p = 2`` ``dt``-convergence check to fail.
"""

from __future__ import annotations

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
from wakis.particles.solver import _pec_wall_closure_1d
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
    with ``include_wall_planes=True`` at every step from PEC z-low birth through
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
        f"Gauss residual with include_wall_planes=True exceeded 1e-12 for "
        f"p={order}, N={n_macro}: max={max_res:.3e}"
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
        max_res_flag = max(r for _, r in history_flag)
        print(f"[mutation flag=False, p=2, N=1] max_res={max_res_flag:.6e}")
        assert max_res_flag > 1e-2
        with pytest.raises(AssertionError):
            assert max_res_flag <= 1e-12

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


def test_p1_closure_at_wall_is_identically_zero():
    """Test 3: For ``p = 1`` at ``zeta = 0`` (and ``zeta = Nz``), the wall-closure
    current is identically zero (``atol = 0``).
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
    ``-injected_charge`` (for ``p = 1``) and ``-sum(q_vis)`` (for ``p = 1, 2, 3``,
    where ``sum(q_vis) = (1 - w_{-1}) * injected_charge``) to ``<= 1e-12`` relative.
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

    # At zeta = 0, the B-spline weight on the exterior ghost node k = -1 is
    # 0 (p = 1), 1/8 (p = 2), 1/6 (p = 3), so sum(q_vis) = (1 - w_{-1}) * q_inj.
    w_ghost = {1: 0.0, 2: 0.125, 3: 1.0 / 6.0}[order]
    q_inj = float(inj.injected_charge)
    expected_q_bg_total = -(1.0 - w_ghost) * q_inj
    actual_q_bg_total = float(np.sum(q_bg[:, :, 0]))
    rel_err = abs(actual_q_bg_total - expected_q_bg_total) / abs(expected_q_bg_total)
    print(
        f"[q_bg p={order}] sum(q_bg)={actual_q_bg_total:.6e}, "
        f"expected={expected_q_bg_total:.6e}, rel_err={rel_err:.3e}"
    )
    assert rel_err <= 1e-12
    if order == 1:
        assert abs(actual_q_bg_total - (-q_inj)) / abs(q_inj) <= 1e-12
    assert ps.gauss_residual(include_wall_planes=True) <= 1e-12


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


def test_wall_closure_work_p1_zero_p2_residual_and_mutation(monkeypatch):
    """Test 5: Verify ``wall_closure_work`` / ``W_wall_closure`` in ``energy_balance()``.

    Covers:
    (a) For ``p = 1`` with injection at ``zeta = 0``, ``W_wall_closure == 0.0`` exactly.
    (b) For ``p = 2``, ``W_wall_closure`` is nonzero and the energy residual with the
        term is at least 10x smaller than the residual without it.
    (c) Mutation: monkeypatching ``ParticleSolver.wall_closure_work`` to return ``0.0``
        causes the ``test_083`` dt-convergence check for ``p = 2`` to fail with
        ``AssertionError``.
    """
    from tests.test_083_particles_beams import (
        _assert_richardson_order,
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

    # (c) Mutation: monkeypatch wall_closure_work -> 0.0 and show test_083 dt-convergence fails for p = 2
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
        with pytest.raises(AssertionError):
            assert np.all(ress_mut > 0.0)
            assert np.all(np.diff(ress_mut) < 0.0)
            _assert_richardson_order(dts_mut, ress_mut, order=1.0)
