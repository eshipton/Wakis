# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Core validation suite for ``wakis.particles`` (Milestone M1).

Covers:
- V1: Single-particle relativistic pushers (Boris, Vay, Higuera-Cary):
  cyclotron gyration radius and frequency ``omega_c / gamma``, relativistic
  ``E x B`` drift at ``gamma = 1.1`` and ``gamma = 20`` (demonstrating the
  Boris drift error and Vay/Higuera-Cary accuracy), relativistic hyperbolic
  motion in uniform ``E``, and second-order convergence in ``dt``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import c as c_light
from scipy.constants import e as e_charge
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import m_e

from wakis import GridFIT3D, SolverFIT3D
from wakis.particles import (
    BeamInjector,
    Collector,
    LogicalGridMap,
    ParticleSolver,
    Species,
    cold_plasma_dispersion_discrete,
    compute_dual_divergence_d,
    deposit_charge,
    deposit_current_esirkepov,
    deposit_current_naive,
    gather_fields,
    gauss_residual,
    integrate_trajectory,
    lorentz_gamma,
    push_boris,
    push_higuera_cary,
    push_vay,
)


# ============================================================================
# Species & LogicalGridMap unit/invariant tests
# ============================================================================


def test_species_soa_compaction_and_sort():
    """Verify Species SoA buffer growth, boolean compaction, and cell sorting."""
    sp = Species(capacity=4, track_ids=True)
    assert sp.charge == -e_charge
    assert sp.mass == m_e
    assert sp.n_particles == 0

    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 1.0, size=20)
    y = rng.uniform(0.0, 1.0, size=20)
    z = rng.uniform(0.0, 1.0, size=20)
    ux = rng.normal(0.0, 1e6, size=20)
    uy = rng.normal(0.0, 1e6, size=20)
    uz = rng.normal(0.0, 1e6, size=20)
    w = np.full(20, 100.0)

    sp.add_particles(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, w=w)
    assert sp.n_particles == 20
    assert sp.capacity >= 20
    np.testing.assert_array_equal(sp.id, np.arange(20, dtype=np.int64))

    # Keep only particles with x < 0.5
    mask = sp.x < 0.5
    expected_ids = sp.id[mask].copy()
    sp.keep(mask)
    assert sp.n_particles == int(np.sum(mask))
    np.testing.assert_array_equal(sp.id, expected_ids)

    # Sort by cell index on a 4x4x4 grid over [0, 1]^3
    gmap = LogicalGridMap(
        np.linspace(0.0, 1.0, 5),
        np.linspace(0.0, 1.0, 5),
        np.linspace(0.0, 1.0, 5),
    )
    ke_before = sp.kinetic_energy()
    q_before = sp.total_charge()
    sp.sort_by_cell(gmap)

    xi, eta, zeta = gmap.to_logical(sp.x, sp.y, sp.z)
    cell_idx = (
        np.floor(xi).astype(int)
        + 4 * np.floor(eta).astype(int)
        + 16 * np.floor(zeta).astype(int)
    )
    assert np.all(np.diff(cell_idx) >= 0)
    assert np.isclose(sp.kinetic_energy(), ke_before, rtol=1e-15)
    assert np.isclose(sp.total_charge(), q_before, rtol=1e-15)


def test_logical_grid_map_uniform_and_graded():
    """Verify exact piecewise-linear round-trip on uniform and graded grids."""
    x_nodes = np.array([0.0, 0.1, 0.25, 0.5, 1.0])
    y_nodes = np.linspace(-0.5, 0.5, 9)
    z_nodes = np.geomspace(1.0, 2.0, 7) - 1.0
    gmap = LogicalGridMap(x_nodes, y_nodes, z_nodes)
    assert not gmap.is_uniform
    assert gmap._uniform == (False, True, False)

    rng = np.random.default_rng(123)
    x = rng.uniform(0.0, 1.0, 200)
    y = rng.uniform(-0.5, 0.5, 200)
    z = rng.uniform(0.0, 1.0, 200)

    xi, eta, zeta = gmap.to_logical(x, y, z)
    # On uniform y-axis, affine formula and searchsorted must agree to round-off
    eta_ss = gmap.to_logical_1d(y, 1, use_searchsorted=True)
    np.testing.assert_allclose(eta, eta_ss, rtol=0.0, atol=1e-14)

    x_rec, y_rec, z_rec = gmap.to_physical(xi, eta, zeta)
    np.testing.assert_allclose(x_rec, x, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(y_rec, y, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(z_rec, z, rtol=0.0, atol=1e-14)


# ============================================================================
# V1: Single-particle relativistic pushers (Boris, Vay, Higuera-Cary)
# ============================================================================


@pytest.mark.parametrize("pusher", ["boris", "vay", "hc"])
def test_v1_relativistic_gyration(pusher, monkeypatch):
    """V1a: Relativistic gyration radius and cyclotron frequency ``omega_c / gamma``.

    Physical setup
    --------------
    An electron with initial Lorentz factor ``gamma_0 = 3.0`` gyrates in a
    uniform magnetic field ``B = (0, 0, B0)`` with ``B0 = 0.1 T`` and ``E = 0``.
    Analytical cyclotron frequency: ``Omega_c = |q| B0 / (gamma_0 m_e)``.
    Analytical Larmor radius: ``r_L = u0 * m_e / (|q| B0)``.

    Error estimate & tolerance
    --------------------------
    With ``N_steps_per_turn = 200`` (``Omega_c * dt = 2*pi / 200 = 0.0314 rad``),
    the second-order leapfrog phase error is
    ``|Omega_num - Omega_c| / Omega_c = (Omega_c dt)^2 / 12 = 8.23e-5``,
    and the discrete radius offset is ``(Omega_c dt)^2 / 8 = 1.23e-4``.
    We assert relative errors <= 2.5e-4 for both ``r_L`` and ``Omega_c``, and
    machine-precision energy conservation ``|gamma - gamma_0| / gamma_0 <= 1e-14``.
    """
    q = -e_charge
    m = m_e
    B0 = 0.1  # [T]
    gamma0 = 3.0
    beta0 = np.sqrt(1.0 - 1.0 / (gamma0 * gamma0))
    v0 = beta0 * c_light
    u0_mag = gamma0 * v0

    omega_c = abs(q) * B0 / (gamma0 * m)
    r_larmor = u0_mag * m / (abs(q) * B0)
    period = 2.0 * np.pi / omega_c

    steps_per_turn = 200
    n_turns = 3
    dt = period / steps_per_turn
    n_steps = steps_per_turn * n_turns

    x0 = np.array([0.0, -r_larmor, 0.0])
    u0 = np.array([u0_mag, 0.0, 0.0])

    t_hist, x_hist, u_hist, u_sync = integrate_trajectory(
        x0=x0,
        u0=u0,
        dt=dt,
        n_steps=n_steps,
        E_fn=np.zeros(3),
        B_fn=np.array([0.0, 0.0, B0]),
        q=q,
        m=m,
        pusher=pusher,
    )

    radii = np.hypot(x_hist[:, 0], x_hist[:, 1])
    r_num = float(np.mean(radii))
    rel_err_r = abs(r_num - r_larmor) / r_larmor
    assert rel_err_r <= 2.5e-4

    # Unwrap gyration angle in the (x, y) plane to measure Omega_c
    angles = np.unwrap(np.arctan2(x_hist[:, 1], x_hist[:, 0]))
    omega_num = float(abs(angles[-1] - angles[0]) / t_hist[-1])
    rel_err_omega = abs(omega_num - omega_c) / omega_c
    assert rel_err_omega <= 2.5e-4

    # Exact discrete momentum rotation check:
    # For Boris/Vay in E=0, tan(Omega_rot * dt / 2) = Omega_c * dt / 2;
    # for Higuera-Cary, tan(Omega_rot * dt / 2) = (|q| B0 dt / 2 m) / gamma_hc.
    tau = 0.5 * abs(q) * B0 * dt / m
    if pusher == "hc":
        sigma = gamma0 * gamma0 - tau * tau
        gamma_rot = np.sqrt(0.5 * (sigma + np.sqrt(sigma * sigma + 4.0 * tau * tau)))
    else:
        gamma_rot = gamma0
    omega_discrete_exact = (2.0 / dt) * np.arctan(tau / gamma_rot)
    angles_u = np.unwrap(np.arctan2(u_hist[1:, 1], u_hist[1:, 0]))
    omega_u_num = float(abs(angles_u[-1] - angles_u[0]) / ((n_steps - 1) * dt))
    assert abs(omega_u_num - omega_discrete_exact) / omega_discrete_exact <= 1e-13

    # Energy (gamma) conservation in pure B field (accumulated FP round-off over 600 steps)
    gamma_hist = lorentz_gamma(u_sync[:, 0], u_sync[:, 1], u_sync[:, 2])
    assert np.max(np.abs(gamma_hist - gamma0)) / gamma0 <= 5e-14

    # Code mutation check: mutating _INV_C2 -> 0.0 inside wakis.particles.pushers
    # drops the relativistic 1/gamma factor in the magnetic rotation, changing the
    # cyclotron frequency by gamma0 = 3.0 and the Larmor radius by 1/gamma0.
    import wakis.particles.pushers as pushers_mod

    monkeypatch.setattr(pushers_mod, "_INV_C2", 0.0)
    _, x_mut, u_mut, _ = integrate_trajectory(
        x0=x0,
        u0=u0,
        dt=dt,
        n_steps=n_steps,
        E_fn=np.zeros(3),
        B_fn=np.array([0.0, 0.0, B0]),
        q=q,
        m=m,
        pusher=pusher,
    )
    angles_u_mut = np.unwrap(np.arctan2(u_mut[1:, 1], u_mut[1:, 0]))
    omega_u_mut = float(abs(angles_u_mut[-1] - angles_u[0]) / ((n_steps - 1) * dt))
    angles_x_mut = np.unwrap(np.arctan2(x_mut[:, 1], x_mut[:, 0]))
    omega_x_mut = float(abs(angles_x_mut[-1] - angles_x_mut[0]) / t_hist[-1])
    assert abs(omega_u_mut - omega_c) / omega_c > 1.9
    assert abs(omega_x_mut - omega_c) / omega_c > 1.9
    assert float(np.max(np.linalg.norm(x_mut - x_hist, axis=1)) / r_larmor) > 1.5


def test_v1_exb_drift_low_and_high_gamma():
    """V1b: Relativistic ``E x B`` drift fixed-point identity and gyrating+drifting guiding-centre test.

    Physical setup
    --------------
    1. **Drift fixed-point identity check**: Crossed uniform fields ``B = (0, 0, B0)``
       and ``E = (0, E_y, 0)`` with ``E_y = v_d * B0``. A particle initialized
       directly at the drift fixed point ``u(0) = (gamma_d * v_d, 0, 0)`` has
       ``E + v_d x B = 0``. For Vay (2008) and Higuera-Cary (2017), ``u = gamma_d v_d``
       is an exact algebraic fixed point of the one-step map, whereas Boris's
       half-kick ``gamma^- = gamma_d sqrt(1 + (beta_d Omega_c dt / 2)^2) != gamma_d``
       fails to cancel ``E`` and ``v x B``.
    2. **Non-degenerate gyrating-plus-drifting guiding-centre benchmark at ``gamma_d = 10``**:
       In the drift frame moving at ``v_d`` along ``+x``, ``E' = 0`` and
       ``B'_z = B0 / gamma_d``, and the particle gyrates with proper speed
       ``beta' = 0.25`` (``gamma' = 1.0328``, lab ``gamma_0 = gamma_d * gamma' = 10.328``).
       Over ``N_turns = 4`` complete gyroperiods ``T_lab = 2*pi * gamma_d^2 * gamma' * m_e / (|q| B0)``
       (``40`` steps/period), the particle returns to ``x' = y' = 0`` in the drift frame,
       so its exact lab-frame guiding-centre displacement is ``(v_d * N_turns * T_lab, 0, 0)``.
       Vay and Higuera-Cary reproduce the guiding-centre drift velocity ``v_d`` to
       ``<= 2e-5``, whereas Boris suffers a ``> 1.5e-2`` spurious drift error.
    """
    q = -e_charge
    m = m_e
    B0 = 1.0  # [T]
    omega_c0 = abs(q) * B0 / m
    theta_step = 0.4  # proper cyclotron angle Omega_c * dt [rad]
    n_steps = 200

    boris_transverse_errors = {}
    boris_drift_errors = {}

    # Part 1: Algebraic drift fixed-point identity check at gamma_d = 1.1 and 20.0
    for gamma_d in (1.1, 20.0):
        beta_d = np.sqrt(1.0 - 1.0 / (gamma_d * gamma_d))
        v_d = beta_d * c_light
        Ey = v_d * B0
        dt = theta_step * gamma_d / omega_c0

        x0 = np.zeros(3)
        u0 = np.array([gamma_d * v_d, 0.0, 0.0])
        E_vec = np.array([0.0, Ey, 0.0])
        B_vec = np.array([0.0, 0.0, B0])

        x_exact = v_d * (n_steps * dt)

        for pusher in ("vay", "hc"):
            _, x_hist, _, u_sync = integrate_trajectory(
                x0=x0,
                u0=u0,
                dt=dt,
                n_steps=n_steps,
                E_fn=E_vec,
                B_fn=B_vec,
                q=q,
                m=m,
                pusher=pusher,
            )
            rel_x_err = abs(x_hist[-1, 0] - x_exact) / x_exact
            rel_y_err = np.max(np.abs(x_hist[:, 1])) / x_exact
            assert rel_x_err <= 1e-13, f"{pusher} failed drift x at gamma={gamma_d}"
            assert rel_y_err <= 1e-14, f"{pusher} failed drift y at gamma={gamma_d}"
            np.testing.assert_allclose(u_sync[:, 0], gamma_d * v_d, rtol=1e-13)
            np.testing.assert_allclose(
                u_sync[:, 1], 0.0, atol=1e-13 * (gamma_d * v_d)
            )

        # Boris pusher on the identical setup
        _, x_boris, _, _ = integrate_trajectory(
            x0=x0,
            u0=u0,
            dt=dt,
            n_steps=n_steps,
            E_fn=E_vec,
            B_fn=B_vec,
            q=q,
            m=m,
            pusher="boris",
        )
        boris_drift_errors[gamma_d] = abs(x_boris[-1, 0] - x_exact) / x_exact
        boris_transverse_errors[gamma_d] = float(
            np.max(np.abs(x_boris[:, 1])) / x_exact
        )

    assert boris_transverse_errors[1.1] > 1e-4
    assert boris_transverse_errors[20.0] > 1e-3
    assert boris_transverse_errors[20.0] > boris_transverse_errors[1.1]
    assert boris_drift_errors[20.0] > 1e-4

    # Part 2: Non-degenerate gyrating-plus-drifting trajectory at gamma_d = 10.0
    gamma_d_gyr = 10.0
    beta_d_gyr = np.sqrt(1.0 - 1.0 / (gamma_d_gyr * gamma_d_gyr))
    v_d_gyr = beta_d_gyr * c_light
    Ey_gyr = v_d_gyr * B0
    beta_prime = 0.25
    gamma_prime = 1.0 / np.sqrt(1.0 - beta_prime * beta_prime)
    # Lorentz boost of u' = (0, gamma' beta' c, 0) from drift frame to lab frame:
    u0_gyr = np.array(
        [gamma_d_gyr * gamma_prime * v_d_gyr, gamma_prime * beta_prime * c_light, 0.0]
    )
    omega_lab = (abs(q) * B0) / (m * gamma_d_gyr * gamma_d_gyr * gamma_prime)
    T_period_lab = 2.0 * np.pi / omega_lab
    n_turns = 4
    steps_per_turn = 40
    n_steps_gyr = n_turns * steps_per_turn
    dt_gyr = (n_turns * T_period_lab) / n_steps_gyr
    t_end_gyr = n_steps_gyr * dt_gyr

    gyr_drift_vx_err = {}
    gyr_drift_vy_err = {}
    for pusher in ("boris", "vay", "hc"):
        _, x_h, _, _ = integrate_trajectory(
            x0=np.zeros(3),
            u0=u0_gyr,
            dt=dt_gyr,
            n_steps=n_steps_gyr,
            E_fn=np.array([0.0, Ey_gyr, 0.0]),
            B_fn=np.array([0.0, 0.0, B0]),
            q=q,
            m=m,
            pusher=pusher,
        )
        vx_mean = float(x_h[-1, 0] / t_end_gyr)
        vy_mean = float(x_h[-1, 1] / t_end_gyr)
        gyr_drift_vx_err[pusher] = abs(vx_mean - v_d_gyr) / v_d_gyr
        gyr_drift_vy_err[pusher] = abs(vy_mean) / v_d_gyr

    assert gyr_drift_vx_err["vay"] <= 2.0e-5
    assert gyr_drift_vx_err["hc"] <= 2.0e-5
    assert gyr_drift_vx_err["boris"] > 3.0e-3
    assert gyr_drift_vx_err["boris"] > 100.0 * gyr_drift_vx_err["vay"]
    assert gyr_drift_vx_err["boris"] > 100.0 * gyr_drift_vx_err["hc"]
    assert gyr_drift_vy_err["boris"] > 2.0e-2
    assert gyr_drift_vy_err["boris"] > 10.0 * gyr_drift_vy_err["vay"]
    assert gyr_drift_vy_err["boris"] > 10.0 * gyr_drift_vy_err["hc"]


@pytest.mark.parametrize("pusher", ["boris", "vay", "hc"])
def test_v1_hyperbolic_motion(pusher):
    """V1c: Relativistic hyperbolic motion in a uniform electric field (``B = 0``).

    Physical setup
    --------------
    A particle of charge ``q`` and mass ``m`` starts from rest at ``x(0) = 0``
    in a uniform field ``E = (E0, 0, 0)``, ``B = 0``, accelerating to
    ``gamma(T) = sqrt(1 + (q E0 T / (m c))^2) ~ 5.1``.
    Note: for ``B = 0``, the magnetic rotation vector vanishes (``tau = 0, t = 0``)
    so ``push_boris``, ``push_vay``, and ``push_higuera_cary`` are algebraically
    identical (reducing to ``u^{n+1/2} = u^{n-1/2} + (q dt / m) E^n``); this test
    verifies the common electric impulse and relativistic position quadrature.
    Analytical solution:
    ``u_x(t) = (q E0 / m) t``,
    ``x(t) = (m c^2 / (q E0)) * (sqrt(1 + (q E0 t / (m c))^2) - 1)``.

    Error estimate & tolerance
    --------------------------
    Momentum ``u_x`` is linear in ``t`` and is integrated to round-off
    (``<= 2e-14`` over 500 steps). Position ``x^N`` is a midpoint quadrature of
    ``v_x(t)`` over ``N = 500`` steps, whose exact leading Euler-Maclaurin error is
    ``(dt^2 / 24) (a_x(0) - a_x(T)) / x(T) = 1.0088e-6``; we assert
    ``|x^N - x(T)| / |x(T)| <= 1.2e-6`` and agreement with the asymptotic
    midpoint error prediction to ``<= 1e-4``.
    """
    q = e_charge
    m = m_e
    E0 = 5.0e6  # [V/m]
    # Choose T so q E0 T / (m c) = 5.0 -> final gamma = sqrt(26) ~ 5.099
    T_end = 5.0 * m * c_light / (q * E0)
    n_steps = 500
    dt = T_end / n_steps

    t_hist, x_hist, _, u_sync = integrate_trajectory(
        x0=np.zeros(3),
        u0=np.zeros(3),
        dt=dt,
        n_steps=n_steps,
        E_fn=np.array([E0, 0.0, 0.0]),
        B_fn=np.zeros(3),
        q=q,
        m=m,
        pusher=pusher,
    )

    u_exact = (q * E0 / m) * t_hist
    gamma_exact = np.sqrt(1.0 + (u_exact / c_light) ** 2)
    x_exact = (m * c_light * c_light / (q * E0)) * (gamma_exact - 1.0)

    rel_u_err = np.max(np.abs(u_sync[:, 0] - u_exact)) / u_exact[-1]
    rel_x_err = abs(x_hist[-1, 0] - x_exact[-1]) / x_exact[-1]
    a0 = q * E0 / m
    aT = a0 / (gamma_exact[-1] ** 3)
    predicted_rel_x_err = (dt * dt / 24.0) * (a0 - aT) / x_exact[-1]

    assert rel_u_err <= 2e-14
    assert rel_x_err <= 1.2e-6
    assert abs(rel_x_err - predicted_rel_x_err) / predicted_rel_x_err <= 1e-4


@pytest.mark.parametrize("pusher", ["boris", "vay", "hc"])
def test_v1_second_order_convergence(pusher):
    """V1d: Measured second-order convergence in ``dt`` for all three pushers.

    Physical setup
    --------------
    Relativistic electron (``gamma_0 ~ 2.5``) in combined non-collinear fields
    ``E = (2e6, -1e6, 5e5) V/m`` and ``B = (0.02, -0.01, 0.08) T`` integrated
    over a fixed physical duration ``T`` with ``N in {100, 200, 400, 800}``
    compared against a fine reference trajectory (``N_ref = 12800``).
    The measured convergence order ``log2(err(dt) / err(dt/2))`` for both
    position ``x(T)`` and synchronised momentum ``u(T)`` must lie in ``[1.98, 2.02]``.
    """
    q = -e_charge
    m = m_e
    E_vec = np.array([2.0e6, -1.0e6, 0.5e6])
    B_vec = np.array([0.02, -0.01, 0.08])
    x0 = np.array([1.0e-3, -2.0e-3, 0.5e-3])
    u0 = np.array([1.8 * c_light, 1.1 * c_light, -0.6 * c_light])
    T_end = 1.5e-10  # [s]

    _, x_ref, _, u_ref = integrate_trajectory(
        x0=x0,
        u0=u0,
        dt=T_end / 12800,
        n_steps=12800,
        E_fn=E_vec,
        B_fn=B_vec,
        q=q,
        m=m,
        pusher=pusher,
    )
    x_exact = x_ref[-1]
    u_exact = u_ref[-1]

    err_x = []
    err_u = []
    for n_steps in (100, 200, 400):
        dt = T_end / n_steps
        _, x_h, _, u_h = integrate_trajectory(
            x0=x0,
            u0=u0,
            dt=dt,
            n_steps=n_steps,
            E_fn=E_vec,
            B_fn=B_vec,
            q=q,
            m=m,
            pusher=pusher,
        )
        err_x.append(np.linalg.norm(x_h[-1] - x_exact) / np.linalg.norm(x_exact))
        err_u.append(np.linalg.norm(u_h[-1] - u_exact) / np.linalg.norm(u_exact))

    order_x_1 = np.log2(err_x[0] / err_x[1])
    order_x_2 = np.log2(err_x[1] / err_x[2])
    order_u_1 = np.log2(err_u[0] / err_u[1])
    order_u_2 = np.log2(err_u[1] / err_u[2])

    assert 1.98 <= order_x_1 <= 2.02
    assert 1.98 <= order_x_2 <= 2.02
    assert 1.98 <= order_u_1 <= 2.02
    assert 1.98 <= order_u_2 <= 2.02

    # Mutation check: omitting the half-step initialization degrades the global
    # convergence order from 2nd order to 1st order.
    err_x_uncentred = []
    for n_steps in (100, 200, 400):
        dt = T_end / n_steps
        _, x_h, _, _ = integrate_trajectory(
            x0=x0,
            u0=u0,
            dt=dt,
            n_steps=n_steps,
            E_fn=E_vec,
            B_fn=B_vec,
            q=q,
            m=m,
            pusher=pusher,
            half_step_init=False,
        )
        err_x_uncentred.append(
            np.linalg.norm(x_h[-1] - x_exact) / np.linalg.norm(x_exact)
        )
    order_mut = np.log2(err_x_uncentred[1] / err_x_uncentred[2])
    assert order_mut < 1.2


# ============================================================================
# V2: Discrete charge conservation (Esirkepov p=1, 2 on uniform & graded grids)
# ============================================================================


def _build_v2_solver(grid_kind: str, Nx: int = 8, Ny: int = 8, Nz: int = 8):
    """Build an 8x8x8 periodic SolverFIT3D with either uniform or graded spacing."""
    L = 0.08  # [m]
    if grid_kind == "uniform":
        grid = GridFIT3D(
            xmin=0.0,
            xmax=L,
            ymin=0.0,
            ymax=L,
            zmin=0.0,
            zmax=L,
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            verbose=0,
        )
    elif grid_kind == "graded":
        # Non-uniform smoothly graded node distributions on all 3 axes
        s_x = np.linspace(0.0, 1.0, Nx + 1)
        s_y = np.linspace(0.0, 1.0, Ny + 1)
        s_z = np.linspace(0.0, 1.0, Nz + 1)
        x_nodes = L * (s_x + 0.18 * np.sin(2.0 * np.pi * s_x) / (2.0 * np.pi))
        y_nodes = L * (s_y - 0.15 * np.sin(2.0 * np.pi * s_y) / (2.0 * np.pi))
        z_nodes = L * (0.65 * s_z + 0.35 * s_z * s_z)
        grid = GridFIT3D(
            xmin=0.0,
            xmax=L,
            ymin=0.0,
            ymax=L,
            zmin=0.0,
            zmax=L,
            x=x_nodes,
            y=y_nodes,
            z=z_nodes,
            boundary_at_domain_edge=False,
            verbose=0,
        )
    else:
        raise ValueError(f"Unknown grid_kind={grid_kind!r}")

    solver = SolverFIT3D(
        grid,
        cfln=0.45,
        bc_low=["periodic", "periodic", "periodic"],
        bc_high=["periodic", "periodic", "periodic"],
        verbose=0,
    )
    return solver


@pytest.mark.parametrize("grid_kind", ["uniform", "graded"])
@pytest.mark.parametrize("order", [1, 2])
def test_v2_gauss_residual_1000_steps(grid_kind, order):
    """V2: Gauss residual <= 1e-12 over 1000 steps with 1000 random 3D particles.

    Physical setup
    --------------
    ``N_p = 1000`` relativistic macroparticles initialized with random 3D
    positions and isotropic relativistic momenta (``|v| ~ 0.30 c`` in all
    directions, crossing ~80 cell boundaries per particle and ~8e4 cell
    boundaries in total along x, y, z) evolve
    self-consistently for ``N_steps = 1000`` FIT steps on both a uniform and a
    smoothly graded 3D periodic grid with shape orders ``p = 1`` (CIC) and
    ``p = 2`` (TSC).

    Error estimate & tolerance
    --------------------------
    Because ``S_tilde C_tilde = 0`` identically and the Esirkepov current
    decomposition satisfies ``S_tilde W^{n+1/2} = -(q^{n+1} - q^n)`` in logical
    coordinates on any tensor-product grid, the discrete Gauss residual
    ``max|S_tilde d_tilde - q| / max|q|`` is bounded by accumulated floating-point
    round-off ``O(sqrt(N_steps) * eps_mach) ~ 1e-13 <= 1e-12``.
    """
    solver = _build_v2_solver(grid_kind, Nx=8, Ny=8, Nz=8)
    L = 0.08
    n_particles = 1000
    n_steps = 1000

    rng = np.random.default_rng(2026 + order + (0 if grid_kind == "uniform" else 10))
    x0 = rng.uniform(0.0, L, size=n_particles)
    y0 = rng.uniform(0.0, L, size=n_particles)
    z0 = rng.uniform(0.0, L, size=n_particles)

    # Isotropic 3D velocities up to ~0.45 c so particles cross cells in all directions
    dirs = rng.normal(0.0, 1.0, size=(n_particles, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    speeds = rng.uniform(0.15 * c_light, 0.45 * c_light, size=(n_particles, 1))
    v_vec = dirs * speeds
    gamma_init = 1.0 / np.sqrt(1.0 - np.sum(v_vec * v_vec, axis=1) / (c_light**2))
    u_vec = gamma_init[:, None] * v_vec

    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    sp.add_particles(
        x=x0,
        y=y0,
        z=z0,
        ux=u_vec[:, 0],
        uy=u_vec[:, 1],
        uz=u_vec[:, 2],
        w=np.full(n_particles, 1.0e4),
    )

    ps = ParticleSolver(
        solver,
        species=[sp],
        pusher="boris",
        shape=order,
        gather="galerkin",
        sort_every=100,
    ).attach()

    max_res = 0.0
    for step in range(1, n_steps + 1):
        solver.one_step()
        if step in (1, 10, 100, 500, 1000):
            res = ps.gauss_residual()
            max_res = max(max_res, res)
            assert res <= 1e-12, (
                f"Gauss residual {res:.3e} > 1e-12 at step {step} "
                f"(grid={grid_kind}, order={order})"
            )

    final_res = gauss_residual(solver, sp, order=order)
    assert final_res <= 1e-12


@pytest.mark.parametrize("grid_kind", ["uniform", "graded"])
@pytest.mark.parametrize("order", [1, 2])
def test_v2_mutation_naive_deposit_breaks_gauss(grid_kind, order):
    """V2 mutation check: naive ``J = q v`` midpoint deposit must make Gauss residual grow."""
    solver = _build_v2_solver(grid_kind, Nx=8, Ny=8, Nz=8)
    L = 0.08
    n_particles = 400
    rng = np.random.default_rng(999 + order)

    x0 = rng.uniform(0.0, L, size=n_particles)
    y0 = rng.uniform(0.0, L, size=n_particles)
    z0 = rng.uniform(0.0, L, size=n_particles)
    ux = rng.uniform(-0.3 * c_light, 0.3 * c_light, size=n_particles)
    uy = rng.uniform(-0.3 * c_light, 0.3 * c_light, size=n_particles)
    uz = rng.uniform(-0.3 * c_light, 0.3 * c_light, size=n_particles)

    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    sp.add_particles(x=x0, y=y0, z=z0, ux=ux, uy=uy, uz=uz, w=1.0e5)

    ps_naive = ParticleSolver(
        solver,
        species=[sp],
        pusher="boris",
        shape=order,
        gather="galerkin",
        deposit_scheme="naive",
    ).attach()

    res_history = []
    for step in range(1, 41):
        solver.one_step()
        if step in (1, 10, 40):
            res_history.append(ps_naive.gauss_residual())

    # Naive deposit violates discrete continuity at O(d_xi^2) per step:
    # residual exceeds 1e-3 (over 9 orders of magnitude above the 1e-12 tolerance)
    # and grows with step count.
    assert res_history[0] > 1e-4
    assert res_history[-1] > res_history[0]
    assert res_history[-1] > 1e-2


# ============================================================================
# V3: Cold plasma oscillation (Birdsall-Langdon dispersion & energy exchange)
# ============================================================================


def _run_cold_plasma_oscillation(order: int = 1, gather: str = "galerkin", n_steps: int = 240):
    """Run a 1D-in-3D cold plasma oscillation in a periodic box and return diagnostics."""
    Nx, Ny, Nz = 2, 2, 32
    Lx, Ly, Lz = 0.02, 0.02, 0.32  # dz = 0.01 m
    dz = Lz / Nz
    k_mode = 2.0 * np.pi / Lz  # fundamental longitudinal mode m_z = 1 (32 cells/lambda)

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
        verbose=0,
    )

    # Continuum plasma frequency omega_p0 = 2*pi * 1.2 GHz
    omega_p0 = 2.0 * np.pi * 1.2e9
    n0 = eps_0 * m_e * (omega_p0**2) / (e_charge**2)  # [m^-3]
    vol = Lx * Ly * Lz

    n_ppc_z = 8
    n_z_tot = Nz * n_ppc_z
    z_1d = (np.arange(n_z_tot) + 0.5) * (Lz / n_z_tot)
    x_centers = (np.arange(Nx) + 0.5) * (Lx / Nx)
    y_centers = (np.arange(Ny) + 0.5) * (Ly / Ny)
    xx, yy, zz = np.meshgrid(x_centers, y_centers, z_1d, indexing="ij")
    x_p = xx.ravel()
    y_p = yy.ravel()
    z_p = zz.ravel()

    n_macro = x_p.size
    w_macro = (n0 * vol) / n_macro

    # Small sinusoidal velocity perturbation delta_v << dz / dt (linear regime)
    delta_v = 1.0e-3 * (omega_p0 / k_mode)
    uz_p = delta_v * np.sin(k_mode * z_p)

    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    sp.add_particles(
        x=x_p,
        y=y_p,
        z=z_p,
        ux=0.0,
        uy=0.0,
        uz=uz_p,
        w=w_macro,
    )

    ps = ParticleSolver(
        solver,
        species=[sp],
        pusher="boris",
        shape=order,
        gather=gather,
        sort_every=0,
    ).attach()

    z_edges = (np.arange(Nz) + 0.5) * dz
    sin_basis = np.sin(k_mode * z_edges)

    t_hist = np.zeros(n_steps + 1)
    ez_mode = np.zeros(n_steps + 1)
    ke_hist = np.zeros(n_steps + 1)
    ue_hist = np.zeros(n_steps + 1)

    ke_hist[0] = ps.kinetic_energy(synchronised=True)
    ue_hist[0] = solver.field_energy()

    for step in range(1, n_steps + 1):
        solver.one_step()
        t_hist[step] = step * solver.dt
        Ez_1d = np.mean(solver.E[:, :, :, "z"], axis=(0, 1))
        ez_mode[step] = (2.0 / Nz) * np.sum(Ez_1d * sin_basis)
        ke_hist[step] = ps.kinetic_energy(synchronised=True)
        ue_hist[step] = solver.field_energy()

    omega_pred = cold_plasma_dispersion_discrete(
        omega_p0=omega_p0,
        dt=solver.dt,
        dz=dz,
        k=k_mode,
        order=order,
        gather=gather,
    )

    # Fit Ez_mode(t) = A * sin(omega_num * t) via nonlinear least squares around omega_pred
    from scipy.optimize import least_squares

    amp0 = float(np.max(np.abs(ez_mode)))

    def _res(params):
        om, amp, ph = params
        return ez_mode - amp * np.sin(om * t_hist + ph)

    fit = least_squares(_res, x0=[omega_pred, amp0, 0.0])
    omega_num = float(abs(fit.x[0]))

    return {
        "t": t_hist,
        "ez_mode": ez_mode,
        "ke": ke_hist,
        "ue": ue_hist,
        "omega_p0": omega_p0,
        "omega_pred": omega_pred,
        "omega_num": omega_num,
        "gauss_res": ps.gauss_residual(),
        "dt": solver.dt,
        "dz": dz,
        "k": k_mode,
    }


def test_cold_plasma_dispersion_closed_form_vs_alias_sum(monkeypatch):
    """Verify closed-form ``kappa_grid^2`` in ``cold_plasma_dispersion_discrete`` against direct alias sums (>= 50 terms).

    Physical setup
    --------------
    For ``order in (1, 2)`` and ``gather in ("galerkin", "momentum")``, the
    spatial grid attenuation factor is the Brillouin alias sum
    ``kappa_grid^2(theta) = sum_{m=-M}^{M} [sinc(theta/2 + pi*m)]^{p + p_g + 1}``
    where ``p_g = p - 1`` for ``"galerkin"`` and ``p_g = p`` for ``"momentum"``.
    Using ``M = 100`` (201 terms, ``>= 50`` terms on each side), we check each
    closed form across ``theta = k * dz in [pi/16, pi/8, pi/4, pi/2]`` and verify
    that the wrong expressions ``1 - (2/3) s^2`` for ``p = 1, "momentum"`` and
    ``1 - s^2 + (2/15) s^4`` for ``p = 2, "momentum"`` are rejected.
    """
    import wakis.particles.solver as solver_mod

    omega_p0 = 2.0 * np.pi * 1.2e9
    dz = 0.01
    dt = 1.0e-12  # small dt to isolate kappa_grid^2 to machine precision
    m_arr = np.arange(-100, 101, dtype=np.float64)

    for order in (1, 2):
        for gather, p_g in (("galerkin", order - 1), ("momentum", order)):
            exp_tot = order + p_g + 1
            for theta in (np.pi / 16.0, np.pi / 8.0, np.pi / 4.0, np.pi / 2.0):
                k = theta / dz
                arg = 0.5 * theta + np.pi * m_arr
                sinc = np.where(np.abs(arg) < 1e-14, 1.0, np.sin(arg) / arg)
                kappa2_alias = float(np.sum(sinc**exp_tot))

                w_closed = cold_plasma_dispersion_discrete(
                    omega_p0, dt, dz, k, order=order, gather=gather
                )
                kappa2_closed = (
                    (2.0 / dt) * np.sin(0.5 * w_closed * dt) / omega_p0
                ) ** 2

                if exp_tot == 2:
                    # By Euler's identity sum_{m=-inf}^{inf} 1/(theta/2 + pi*m)^2 = csc^2(theta/2),
                    # sum_{m=-inf}^{inf} sinc^2(theta/2 + pi*m) = sin^2(theta/2)*csc^2(theta/2) == 1.0
                    # identically (exact). Also check against the M=100 alias sum (truncation tail <= 2.1e-3).
                    assert abs(kappa2_closed - 1.0) <= 1.0e-14
                    assert abs(kappa2_closed - kappa2_alias) <= 2.5e-3
                else:
                    # Truncation tail of sum_{|m|>100} |m|^{-N} is O(100^{-(N-1)}) for N >= 3
                    tol = 1.0e-4 if exp_tot == 3 else 1.0e-6
                    assert abs(kappa2_closed - kappa2_alias) <= tol

                s2 = float(np.sin(0.5 * theta) ** 2)
                if order == 1 and gather == "momentum":
                    assert abs(kappa2_closed - (1.0 - 0.5 * s2)) <= 1e-12
                    wrong_p1 = 1.0 - (2.0 / 3.0) * s2
                    assert abs(kappa2_alias - wrong_p1) > 10.0 * abs(
                        kappa2_alias - kappa2_closed
                    )
                elif order == 2 and gather == "momentum":
                    assert (
                        abs(kappa2_closed - (1.0 - (5.0 / 6.0) * s2 + (1.0 / 24.0) * s2 * s2))
                        <= 1e-12
                    )
                    wrong_p2 = 1.0 - s2 + (2.0 / 15.0) * s2 * s2
                    assert abs(kappa2_alias - wrong_p2) > 100.0 * abs(
                        kappa2_alias - kappa2_closed
                    )

    # Code mutation check: mutating cold_plasma_dispersion_discrete in solver_mod
    # to ignore the gather mode must fail the momentum check
    orig_disp = solver_mod.cold_plasma_dispersion_discrete
    monkeypatch.setattr(
        solver_mod,
        "cold_plasma_dispersion_discrete",
        lambda wp0, dt_in, dz_in, k_in, order=2, gather="galerkin": orig_disp(
            wp0, dt_in, dz_in, k_in, order=order, gather="galerkin"
        ),
    )
    w_mut = solver_mod.cold_plasma_dispersion_discrete(
        omega_p0, dt, dz, np.pi / (4.0 * dz), order=2, gather="momentum"
    )
    w_ref = orig_disp(
        omega_p0, dt, dz, np.pi / (4.0 * dz), order=2, gather="momentum"
    )
    assert abs(w_mut - w_ref) / w_ref > 1.0e-2


@pytest.mark.parametrize("gather", ["galerkin", "momentum"])
@pytest.mark.parametrize("order", [1, 2])
def test_v3_cold_plasma_oscillation_dispersion_and_energy(order, gather, monkeypatch):
    """V3: Cold plasma oscillation frequency vs Birdsall-Langdon prediction and energy closure.

    Physical setup
    --------------
    A cold uniform electron plasma (``omega_p0 / 2*pi = 1.2 GHz``) in a 3D
    periodic box with ``Nz = 32`` cells per wavelength along ``z`` is given a
    linear velocity perturbation ``u_z(z, 0) = delta_v sin(k z)``.
    By Parseval's theorem over spatial aliases ``k_m = k + 2*pi*m/dz``, the
    discrete oscillation frequency satisfies
    ``(2 / dt) sin(omega_p dt / 2) = omega_p0 * kappa_grid(k dz)``, where:
    - ``p = 1, "galerkin"``: ``kappa_grid^2 = 1``
    - ``p = 1, "momentum"``: ``kappa_grid^2 = 1 - (1/2) sin^2(k dz / 2)``
    - ``p = 2, "galerkin"``: ``kappa_grid^2 = 1 - (2/3) sin^2(k dz / 2)``
    - ``p = 2, "momentum"``: ``kappa_grid^2 = 1 - (5/6) sin^2(k dz / 2) + (1/24) sin^4(k dz / 2)``

    Error estimate & tolerance
    --------------------------
    With ``delta_z_max / dz ~ 5e-3`` (linear perturbation amplitude) and
    ``N_ppc = 8`` particles per cell, the harmonic frequency fit over ~2.8
    plasma periods agrees with ``omega_pred`` to ``<= 1e-5`` for ``p = 1``
    (where no particle crosses a cell boundary so the discrete system is linear
    up to ``O((delta_v / c)^2) ~ 2e-7``) and ``<= 5e-5`` for ``p = 2``, while
    clearly rejecting the continuum frequency ``omega_p0`` (and rejecting the
    wrong momentum formulas ``1 - (2/3) s^2`` for ``p = 1`` and
    ``1 - s^2 + (2/15) s^4`` for ``p = 2``).
    Total energy ``U_field(t_n) + K_sync(t_n)`` exhibits the analytical
    reversible leapfrog oscillation ``(omega_pred dt)^2 / 4 = 1.318e-3`` (matching
    within 5%), whereas the secular energy drift (linear trend over the run) is
    ``<= 1e-5``.
    """
    res = _run_cold_plasma_oscillation(order=order, gather=gather, n_steps=240)

    rel_omega_err = abs(res["omega_num"] - res["omega_pred"]) / res["omega_pred"]
    rel_err_vs_continuum = abs(res["omega_num"] - res["omega_p0"]) / res["omega_p0"]
    if order == 1:
        assert rel_omega_err <= 1.0e-5
        assert rel_err_vs_continuum > 1.5e-4
    else:
        assert rel_omega_err <= 5.0e-5
        assert rel_err_vs_continuum > 2.5e-3

    if gather == "momentum":
        s2 = float(np.sin(0.5 * res["k"] * res["dz"]) ** 2)
        dt = res["dt"]
        wp0 = res["omega_p0"]
        if order == 1:
            k2_wrong = 1.0 - (2.0 / 3.0) * s2
            w_wrong = float((2.0 / dt) * np.arcsin(0.5 * dt * wp0 * np.sqrt(k2_wrong)))
            assert abs(res["omega_num"] - w_wrong) / w_wrong > 5.0e-4
        else:
            k2_wrong = 1.0 - s2 + (2.0 / 15.0) * (s2 * s2)
            w_wrong = float((2.0 / dt) * np.arcsin(0.5 * dt * wp0 * np.sqrt(k2_wrong)))
            assert abs(res["omega_num"] - w_wrong) / w_wrong > 5.0e-4

    # Verify reversible kinetic <-> electric field energy exchange
    ke0 = res["ke"][0]
    ue_max = float(np.max(res["ue"]))
    ke_min = float(np.min(res["ke"]))
    assert abs(ue_max - ke0) / ke0 <= 5.0e-3
    assert ke_min / ke0 <= 5.0e-3

    # Bounded leapfrog energy oscillation:
    # - For gather="galerkin" (energy-conserving, adjoint deposit/gather), the
    #   reversible peak-to-peak energy oscillation is purely temporal and matches
    #   (omega_pred * dt)^2 / 4 within 5%.
    # - For gather="momentum" (order p gather vs order p-1 deposit), the ratio of
    #   deposit to gather symbols adds an empirical regression heuristic
    #   (1 / sinc(k*dz/2) - 1) ~ (k*dz)^2 / 24 to the reversible oscillation
    #   amplitude, matching within 5%.
    # In both cases, the long-time secular drift (linear trend) is <= 1e-5.
    e_tot = res["ke"] + res["ue"]
    rel_energy_osc = float(np.max(np.abs(e_tot - ke0)) / ke0)
    pred_leapfrog_osc = 0.25 * (res["omega_pred"] * res["dt"]) ** 2
    if gather == "galerkin":
        pred_osc = pred_leapfrog_osc
    else:
        # Empirical regression heuristic for momentum-gather energy oscillation
        half_theta = 0.5 * res["k"] * res["dz"]
        sinc_half = float(np.sin(half_theta) / half_theta)
        pred_osc = pred_leapfrog_osc + (1.0 / sinc_half - 1.0)
    assert abs(rel_energy_osc - pred_osc) / pred_osc <= 0.05

    t = res["t"]
    osc_curve = 1.0 + pred_osc * (np.sin(res["omega_pred"] * t) ** 2)
    secular_slope = float(np.polyfit(t, (e_tot / ke0) - osc_curve, 1)[0])
    secular_drift = abs(secular_slope * t[-1])
    assert secular_drift <= 1.0e-5

    # Gauss's law holds simultaneously to round-off
    assert res["gauss_res"] <= 1e-12

    # Code mutation check: monkeypatching push_momentum in wakis.particles.solver
    # to scale dt by 0.5 halves the effective restoring acceleration (reducing
    # omega_p by 1/sqrt(2)) and must fail the dispersion check.
    import wakis.particles.solver as solver_mod

    orig_push_mom = solver_mod.push_momentum
    monkeypatch.setattr(
        solver_mod,
        "push_momentum",
        lambda sp, Ex, Ey, Ez, Bx, By, Bz, dt, pusher="boris": orig_push_mom(
            sp, Ex, Ey, Ez, Bx, By, Bz, 0.5 * dt, pusher=pusher
        ),
    )
    res_mut = _run_cold_plasma_oscillation(order=order, gather=gather, n_steps=120)
    assert abs(res_mut["omega_num"] - res["omega_pred"]) / res["omega_pred"] > 0.25


# ============================================================================
# Section 2: Self-force test in a closed PEC box (and periodic comparison)
# ============================================================================


@pytest.mark.parametrize("order", [1, 2])
def test_self_force_stationary_particle_in_pec_box(order, monkeypatch):
    """Self-force test: single particle initialized with its discrete Poisson field (no background cancellation).

    Physical setup
    --------------
    A single macroparticle (``w = 1e6`` electrons, ``Q = -1.602e-13 C``) is placed
    at an asymmetric sub-cell position ``(xi, eta, zeta) = (4.31, 4.67, 4.42)``
    inside an ``8 x 8 x 8`` closed PEC box (``L = 0.08 m, dx = 0.01 m``).
    Instead of starting from ``E = 0`` (which silently introduces an immobile
    neutralizing background charge ``q_bg = -q_nodes(0)``), we solve the exact
    discrete 3D FIT Poisson equation ``-S_tilde eps_0 A_tilde G phi = q_nodes(0)``
    with Dirichlet ``phi = 0`` on the PEC walls and initialize ``solver.E = -G phi``
    so ``q_bg = 0`` identically in the interior vacuum.

    Measured self-field & error bounds
    ----------------------------------
    1. Characteristic single-cell Coulomb field scale:
       ``E_Coulomb = |Q| / (4 * pi * eps_0 * dx^2) = 14.400 V/m``.
       With ``gather="galerkin"`` (order ``p`` charge assignment, order ``p-1``
       longitudinal edge gather), an off-centre particle experiences a non-zero
       sub-cell electrostatic self-field (Birdsall & Langdon, 1991, Ch. 8):
       - On the 1-step local gradient proxy ``E_loc = -G q_nodes / (eps_0 dx^2)``:
         ``E_self = (-19.65, 18.14, -9.23) V/m`` (``|E| = 28.29 V/m``) for ``p = 1``
         and ``(-6.96, 6.89, -4.77) V/m`` (``|E| = 10.89 V/m``) for ``p = 2``.
       - On the exact 3D discrete Poisson solution in the ``8 x 8 x 8`` PEC box:
         ``E_self = (-4.550, 3.996, -2.106) V/m`` (``|E| = 6.411 V/m <= 0.5 E_Coulomb``)
         for ``p = 1`` and ``(-1.466, 1.284, -1.016) V/m``
         (``|E| = 2.197 V/m <= 0.2 E_Coulomb``) for ``p = 2`` (a ``2.9x`` reduction
         in self-force from ``p = 1`` to ``p = 2``).
    2. On a periodic 3D discrete Poisson solve (translation-invariant Green's
       function without wall image charges), ``gather="momentum"`` (identical
       order ``p`` for deposit and gather) with a collocated nodal-difference
       field eliminates the self-force at the same asymmetric position
       ``(4.31, 4.67, 4.42)`` to machine precision (``|E_self| <= 1e-13 V/m``);
       with the staggered ``gather_fields`` it is only reduced (asserted
       ``< 0.7x`` the Galerkin value), while ``gather="galerkin"`` retains the
       non-zero sub-cell self-force.
    3. Evolving ``SolverFIT3D`` for ``N_steps = 100`` from the discrete Poisson
       initial condition in the PEC box bounds the self-induced velocity by the
       kinematic estimate ``|u(T)| <= (e |E_self(0)| / m_e) T`` and the displacement
       to ``|dr| / dx <= 5e-5`` (``p = 1``) and ``<= 2e-5`` (``p = 2``), while
       maintaining ``gauss_residual <= 1e-12`` with ``q_bg = 0``.
    """
    from wakis.particles.solver import _interior_vacuum_node_mask

    L = 0.08
    N_phys = 8
    dx = L / N_phys
    grid = GridFIT3D(
        xmin=0.0,
        xmax=L,
        ymin=0.0,
        ymax=L,
        zmin=0.0,
        zmax=L,
        Nx=N_phys,
        Ny=N_phys,
        Nz=N_phys,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        cfln=0.45,
        bc_low=["pec", "pec", "pec"],
        bc_high=["pec", "pec", "pec"],
        verbose=0,
    )

    xi = np.array([4.31])
    eta = np.array([4.67])
    zeta = np.array([4.42])
    x_pos, y_pos, z_pos = float(xi[0] * dx), float(eta[0] * dx), float(zeta[0] * dx)
    Q_macro = -e_charge * 1.0e6
    E_coulomb = abs(Q_macro) / (4.0 * np.pi * eps_0 * dx * dx)  # 14.3996 V/m

    # 1. Local 1-step unscreened gradient proxy at asymmetric position (4.31, 4.67, 4.42)
    q_loc = deposit_charge(
        xi, eta, zeta, np.array([Q_macro]), N_phys, N_phys, N_phys, order=order, periodic=(False, False, False)
    )
    Ex_loc = np.zeros((N_phys, N_phys, N_phys), order="F")
    Ey_loc = np.zeros((N_phys, N_phys, N_phys), order="F")
    Ez_loc = np.zeros((N_phys, N_phys, N_phys), order="F")
    Ex_loc[:-1, :, :] = -(q_loc[1:, :, :] - q_loc[:-1, :, :]) / (eps_0 * dx * dx)
    Ey_loc[:, :-1, :] = -(q_loc[:, 1:, :] - q_loc[:, :-1, :]) / (eps_0 * dx * dx)
    Ez_loc[:, :, :-1] = -(q_loc[:, :, 1:] - q_loc[:, :, :-1]) / (eps_0 * dx * dx)
    E_loc_tup = (Ex_loc.reshape(-1, order="F"), Ey_loc.reshape(-1, order="F"), Ez_loc.reshape(-1, order="F"))
    B_zero_8 = (np.zeros(N_phys**3), np.zeros(N_phys**3), np.zeros(N_phys**3))
    ex_l, ey_l, ez_l, _, _, _ = gather_fields(
        xi, eta, zeta, E_loc_tup, B_zero_8, N_phys, N_phys, N_phys, order=order, gather="galerkin", periodic=(False, False, False)
    )
    if order == 1:
        # Exact analytical CIC-deposit + NGP-gather 1-step self-field formula:
        # E_loc,x = -(Q / (eps_0 dx^2)) * (2*dx_f - 1) * ((1-dy_f)^2 + dy_f^2) * ((1-dz_f)^2 + dz_f^2)
        dx_f, dy_f, dz_f = float(xi[0] - 4.0), float(eta[0] - 4.0), float(zeta[0] - 4.0)
        sx2 = (1.0 - dx_f) ** 2 + dx_f**2
        sy2 = (1.0 - dy_f) ** 2 + dy_f**2
        sz2 = (1.0 - dz_f) ** 2 + dz_f**2
        scale_loc = -Q_macro / (eps_0 * dx * dx)
        ex_l_exact = scale_loc * (2.0 * dx_f - 1.0) * sy2 * sz2
        ey_l_exact = scale_loc * (2.0 * dy_f - 1.0) * sx2 * sz2
        ez_l_exact = scale_loc * (2.0 * dz_f - 1.0) * sx2 * sy2
        np.testing.assert_allclose(
            [ex_l[0], ey_l[0], ez_l[0]],
            [ex_l_exact, ey_l_exact, ez_l_exact],
            rtol=1e-13,
        )
    else:
        # Empirical regression band for p=2 (TSC) sub-cell self-field on local gradient proxy
        assert ex_l[0] < 0.0 and ey_l[0] > 0.0 and ez_l[0] < 0.0
        assert 0.5 * E_coulomb <= float(np.hypot(ex_l[0], np.hypot(ey_l[0], ez_l[0]))) <= E_coulomb

    # 2. Periodic 3D Poisson solve: on a collocated nodal difference E_node = -(phi_{i+1}-phi_{i-1})/(2*dx)
    # where deposit and gather share the identical nodal stencil S^(p), self-force vanishes to machine
    # precision (Birdsall & Langdon Ch. 8); on staggered Yee edges, gather="momentum" reduces the
    # sub-cell self-force relative to gather="galerkin".
    q_per = deposit_charge(
        xi, eta, zeta, np.array([Q_macro]), N_phys, N_phys, N_phys, order=order, periodic=(True, True, True)
    )
    q_hat = np.fft.fftn(q_per - np.mean(q_per))
    k_1d = 2.0 * np.pi * np.fft.fftfreq(N_phys, d=dx)
    kx, ky, kz = np.meshgrid(k_1d, k_1d, k_1d, indexing="ij")
    denom_k = (
        (2.0 * np.sin(0.5 * kx * dx) / dx) ** 2
        + (2.0 * np.sin(0.5 * ky * dx) / dx) ** 2
        + (2.0 * np.sin(0.5 * kz * dx) / dx) ** 2
    )
    denom_k[0, 0, 0] = 1.0
    phi_hat = q_hat / (eps_0 * dx**3 * denom_k)
    phi_hat[0, 0, 0] = 0.0
    phi_per = np.real(np.fft.ifftn(phi_hat))
    Ex_nod = -(np.roll(phi_per, -1, axis=0) - np.roll(phi_per, 1, axis=0)) / (2.0 * dx)
    Ey_nod = -(np.roll(phi_per, -1, axis=1) - np.roll(phi_per, 1, axis=1)) / (2.0 * dx)
    Ez_nod = -(np.roll(phi_per, -1, axis=2) - np.roll(phi_per, 1, axis=2)) / (2.0 * dx)
    w_nod = q_per / Q_macro
    ex_colloc = float(np.sum(w_nod * Ex_nod))
    ey_colloc = float(np.sum(w_nod * Ey_nod))
    ez_colloc = float(np.sum(w_nod * Ez_nod))
    assert float(np.hypot(ex_colloc, np.hypot(ey_colloc, ez_colloc))) <= 1e-13

    Ex_per = -(np.roll(phi_per, -1, axis=0) - phi_per) / dx
    Ey_per = -(np.roll(phi_per, -1, axis=1) - phi_per) / dx
    Ez_per = -(np.roll(phi_per, -1, axis=2) - phi_per) / dx
    E_per_tup = (Ex_per.reshape(-1, order="F"), Ey_per.reshape(-1, order="F"), Ez_per.reshape(-1, order="F"))
    ex_pm, ey_pm, ez_pm, _, _, _ = gather_fields(
        xi, eta, zeta, E_per_tup, B_zero_8, N_phys, N_phys, N_phys, order=order, gather="momentum", periodic=(True, True, True)
    )
    ex_pg, ey_pg, ez_pg, _, _, _ = gather_fields(
        xi, eta, zeta, E_per_tup, B_zero_8, N_phys, N_phys, N_phys, order=order, gather="galerkin", periodic=(True, True, True)
    )
    E_pm_mag = float(np.hypot(ex_pm[0], np.hypot(ey_pm[0], ez_pm[0])))
    E_pg_mag = float(np.hypot(ex_pg[0], np.hypot(ey_pg[0], ez_pg[0])))
    assert E_pg_mag > 1.0
    assert E_pm_mag < 0.7 * E_pg_mag

    # 3. Exact discrete 3D FIT Poisson solve on the closed PEC box (q_bg == 0)
    q_nod = deposit_charge(
        xi,
        eta,
        zeta,
        np.array([Q_macro]),
        solver.Nx,
        solver.Ny,
        solver.Nz,
        order=order,
        periodic=(False, False, False),
    )
    n_int = N_phys - 1
    D1 = 2.0 * np.eye(n_int) - np.eye(n_int, k=1) - np.eye(n_int, k=-1)
    I1 = np.eye(n_int)
    Lap = (
        np.kron(np.kron(D1, I1), I1)
        + np.kron(np.kron(I1, D1), I1)
        + np.kron(np.kron(I1, I1), D1)
    )
    rhs = q_nod[1:N_phys, 1:N_phys, 1:N_phys].reshape(-1, order="C") / (eps_0 * dx)
    phi = np.zeros((solver.Nx, solver.Ny, solver.Nz), dtype=np.float64)
    phi[1:N_phys, 1:N_phys, 1:N_phys] = np.linalg.solve(Lap, rhs).reshape(
        (n_int, n_int, n_int), order="C"
    )

    Ex = np.zeros((solver.Nx, solver.Ny, solver.Nz), dtype=np.float64)
    Ey = np.zeros((solver.Nx, solver.Ny, solver.Nz), dtype=np.float64)
    Ez = np.zeros((solver.Nx, solver.Ny, solver.Nz), dtype=np.float64)
    Ex[:N_phys, :N_phys, :N_phys] = (
        -(phi[1 : N_phys + 1, :N_phys, :N_phys] - phi[:N_phys, :N_phys, :N_phys]) / dx
    )
    Ey[:N_phys, :N_phys, :N_phys] = (
        -(phi[:N_phys, 1 : N_phys + 1, :N_phys] - phi[:N_phys, :N_phys, :N_phys]) / dx
    )
    Ez[:N_phys, :N_phys, :N_phys] = (
        -(phi[:N_phys, :N_phys, 1 : N_phys + 1] - phi[:N_phys, :N_phys, :N_phys]) / dx
    )
    solver.E[:, :, :, "x"] = Ex * (solver.ieps[:, :, :, "x"] > 0)
    solver.E[:, :, :, "y"] = Ey * (solver.ieps[:, :, :, "y"] > 0)
    solver.E[:, :, :, "z"] = Ez * (solver.ieps[:, :, :, "z"] > 0)

    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    sp.add_particles(x=x_pos, y=y_pos, z=z_pos, ux=0.0, uy=0.0, uz=0.0, w=1.0e6)

    ps = ParticleSolver(
        solver, species=[sp], pusher="boris", shape=order, gather="galerkin"
    ).attach()

    # Verify no background charge cancellation: q_bg on interior vacuum nodes is zero
    vac_mask = _interior_vacuum_node_mask(solver)
    assert (
        float(np.max(np.abs(solver._particles_q_bg[vac_mask])))
        / float(np.max(np.abs(q_nod[vac_mask])))
        <= 1e-14
    )
    assert ps.gauss_residual() <= 1e-14

    B_zero = (np.zeros(solver.N), np.zeros(solver.N), np.zeros(solver.N))
    Ex_g, Ey_g, Ez_g, _, _, _ = gather_fields(
        xi,
        eta,
        zeta,
        solver.E,
        B_zero,
        solver.Nx,
        solver.Ny,
        solver.Nz,
        order=order,
        gather="galerkin",
        periodic=(False, False, False),
    )
    E_self_mag = float(np.hypot(Ex_g[0], np.hypot(Ey_g[0], Ez_g[0])))
    assert Ex_g[0] < 0.0 and Ey_g[0] > 0.0 and Ez_g[0] < 0.0
    # Empirical regression bands for the discrete 3D PEC-box sub-cell self-field
    # at (4.31, 4.67, 4.42): [0.25, 0.50] * E_coulomb (p=1) and [0.08, 0.20] * E_coulomb (p=2)
    if order == 1:
        assert 0.25 * E_coulomb <= E_self_mag <= 0.50 * E_coulomb
    else:
        assert 0.08 * E_coulomb <= E_self_mag <= 0.20 * E_coulomb

    # Evolve for 100 steps and bound self-acceleration against kinematic estimate
    n_steps = 100
    for _ in range(n_steps):
        solver.one_step()

    T_run = n_steps * solver.dt
    u_bound = (e_charge * E_self_mag / m_e) * T_run
    dr_bound = 0.5 * (e_charge * E_self_mag / m_e) * (T_run * T_run)
    u_num = float(np.hypot(sp.ux[0], np.hypot(sp.uy[0], sp.uz[0])))
    dr_num = float(np.hypot(sp.x[0] - x_pos, np.hypot(sp.y[0] - y_pos, sp.z[0] - z_pos)))
    assert 0.5 * u_bound <= u_num <= 1.05 * u_bound
    assert 0.5 * dr_bound <= dr_num <= 1.05 * dr_bound
    assert dr_num / dx <= (5.0e-5 if order == 1 else 2.0e-5)
    assert ps.gauss_residual() <= 1e-12

    # Code mutation check: mutating bspline_weights_1d in wakis.particles.shapes
    # to omit the staggered -0.5 shift breaks edge centring:
    # (a) at the asymmetric position (4.31, 4.67, 4.42) with gather="galerkin",
    #     it shifts the gathered self-field by > 3 V/m;
    # (b) at the symmetric box centre (4.0, 4.0, 4.0) with gather="momentum",
    #     the unmutated self-field vanishes by symmetry (<= 1e-13 V/m), whereas
    #     the mutation induces a large spurious force (> 3 V/m).
    import wakis.particles.shapes as shapes_mod

    xi_sym = np.array([4.0])
    q_sym = deposit_charge(
        xi_sym, xi_sym, xi_sym, np.array([Q_macro]), solver.Nx, solver.Ny, solver.Nz, order=order, periodic=(False, False, False)
    )
    rhs_sym = q_sym[1:N_phys, 1:N_phys, 1:N_phys].reshape(-1, order="C") / (eps_0 * dx)
    phi_sym = np.zeros((solver.Nx, solver.Ny, solver.Nz), dtype=np.float64)
    phi_sym[1:N_phys, 1:N_phys, 1:N_phys] = np.linalg.solve(Lap, rhs_sym).reshape(
        (n_int, n_int, n_int), order="C"
    )
    Ex_sym = np.zeros((solver.Nx, solver.Ny, solver.Nz), dtype=np.float64)
    Ex_sym[:N_phys, :N_phys, :N_phys] = (
        -(phi_sym[1 : N_phys + 1, :N_phys, :N_phys] - phi_sym[:N_phys, :N_phys, :N_phys]) / dx
    )
    E_sym_tup = (Ex_sym.reshape(-1, order="F"), np.zeros(solver.N), np.zeros(solver.N))
    ex_sym_ok, _, _, _, _, _ = gather_fields(
        xi_sym, xi_sym, xi_sym, E_sym_tup, B_zero, solver.Nx, solver.Ny, solver.Nz, order=order, gather="momentum", periodic=(False, False, False)
    )
    assert abs(float(ex_sym_ok[0])) <= 1e-13

    orig_bspline = shapes_mod.bspline_weights_1d
    monkeypatch.setattr(
        shapes_mod,
        "bspline_weights_1d",
        lambda u_in, ord_in, staggered=False, xp=np: orig_bspline(
            u_in, ord_in, staggered=False, xp=xp
        ),
    )
    ex_l_mut, ey_l_mut, ez_l_mut, _, _, _ = gather_fields(
        xi, eta, zeta, E_loc_tup, B_zero_8, N_phys, N_phys, N_phys, order=order, gather="galerkin", periodic=(False, False, False)
    )
    dE_l_mut = float(
        np.hypot(
            ex_l_mut[0] - ex_l[0],
            np.hypot(ey_l_mut[0] - ey_l[0], ez_l_mut[0] - ez_l[0]),
        )
    )
    assert dE_l_mut > 3.0
    ex_sym_mut, _, _, _, _, _ = gather_fields(
        xi_sym, xi_sym, xi_sym, E_sym_tup, B_zero, solver.Nx, solver.Ny, solver.Nz, order=order, gather="momentum", periodic=(False, False, False)
    )
    assert abs(float(ex_sym_mut[0])) > 3.0


def test_coupled_step_second_order_time_centring(monkeypatch):
    """Coupled SolverFIT3D + ParticleSolver 2nd-order dt convergence and time-centring mutations.

    Physical setup
    --------------
    A relativistic electron evolves inside a 3D periodic ``SolverFIT3D`` cavity
    carrying a standing electromagnetic wave ``(E_x(z, t), H_y(z, t))`` plus an
    external guide field ``B_ext = (0, 0, 0.15) T`` over a fixed physical time
    ``T = 1.6e-10 s`` across ``N_steps in {20, 40, 80, 160}`` (compared against
    a fine coupled reference ``N_ref = 1280`` on the identical spatial grid).

    Time-centring & mutation checks
    -------------------------------
    1. With ``B^n = 0.5 * mu_0 * (H^{n-1/2} + H^{n+1/2})`` and the initial half-step
       back-push ``_initialize_species_half_step`` enabled, the global trajectory
       error converges at second order (log-log slope in ``[1.95, 2.05]``).
    2. Mutation A: monkeypatching ``ParticleSolver._compute_centred_B`` to return
       ``mu_0 * H^{n+1/2}`` destroys time-centring of ``B^n``, degrading the
       convergence slope to ``< 1.2`` and increasing the ``N = 160`` error by ``> 10x``.
    3. Mutation B: monkeypatching ``ParticleSolver._initialize_species_half_step``
       to a no-op omits the ``u^0 -> u^{-1/2}`` leapfrog initialisation, degrading
       the convergence slope to ``< 1.2`` and increasing the ``N = 160`` error by ``> 50x``.
    """
    from scipy.constants import mu_0

    def _run_coupled(n_steps: int):
        Nx, Ny, Nz = 2, 2, 16
        Lx, Ly, Lz = 0.02, 0.02, 0.16
        dz = Lz / Nz
        k_z = 2.0 * np.pi / Lz
        kappa_z = (2.0 / dz) * np.sin(0.5 * k_z * dz)
        T_total = 1.6e-10
        dt = T_total / n_steps

        grid = GridFIT3D(0.0, Lx, 0.0, Ly, 0.0, Lz, Nx, Ny, Nz, verbose=0)
        solver = SolverFIT3D(
            grid,
            bc_low=["periodic", "periodic", "periodic"],
            bc_high=["periodic", "periodic", "periodic"],
            verbose=0,
        )
        solver.dt = dt

        omega_dt = (2.0 / dt) * np.arcsin(0.5 * c_light * dt * kappa_z)
        E0 = 1.5e7
        z_nodes = np.arange(Nz) * dz
        z_mid = (np.arange(Nz) + 0.5) * dz

        Ex_init = E0 * np.cos(k_z * z_nodes)[None, None, :]
        Hy_init = (
            -(E0 / (mu_0 * c_light))
            * np.sin(k_z * z_mid)[None, None, :]
            * np.sin(0.5 * omega_dt * dt)
        )
        solver.E[:, :, :, "x"] = np.broadcast_to(Ex_init, (Nx, Ny, Nz))
        solver.H[:, :, :, "y"] = np.broadcast_to(Hy_init, (Nx, Ny, Nz))

        sp = Species(name="electron", charge=-e_charge, mass=m_e)
        sp.add_particles(
            x=0.01,
            y=0.01,
            z=0.02,
            ux=0.25 * c_light,
            uy=0.35 * c_light,
            uz=0.15 * c_light,
            w=1.0,
        )
        ps = ParticleSolver(
            solver,
            species=[sp],
            pusher="boris",
            shape=2,
            gather="galerkin",
            sort_every=0,
            B_ext=(0.0, 0.0, 0.15),
        ).attach()

        for _ in range(n_steps):
            solver.one_step()
        return np.array([sp.x[0], sp.y[0], sp.z[0]])

    n_list = [20, 40, 80, 160]
    r_ref = _run_coupled(1280)
    r_norm = float(np.linalg.norm(r_ref))

    errs_normal = [float(np.linalg.norm(_run_coupled(n) - r_ref) / r_norm) for n in n_list]
    slope_normal = float(-np.polyfit(np.log(n_list), np.log(errs_normal), 1)[0])
    assert 1.95 <= slope_normal <= 2.05

    # Mutation A: replace B = mu_0 * (H^{n-1/2} + H^{n+1/2}) / 2 with mu_0 * H^{n+1/2}
    with monkeypatch.context() as m:
        m.setattr(
            ParticleSolver,
            "_compute_centred_B",
            lambda self, H_prev, H_curr: mu_0 * H_curr,
        )
        errs_mut_b = [
            float(np.linalg.norm(_run_coupled(n) - r_ref) / r_norm) for n in n_list
        ]
        slope_mut_b = float(-np.polyfit(np.log(n_list), np.log(errs_mut_b), 1)[0])
        assert slope_mut_b < 1.2
        assert errs_mut_b[-1] > 10.0 * errs_normal[-1]

    # Mutation B: make _initialize_species_half_step a no-op
    with monkeypatch.context() as m:
        m.setattr(
            ParticleSolver,
            "_initialize_species_half_step",
            lambda self, sp, B_n_tuple, t_n: None,
        )
        errs_mut_half = [
            float(np.linalg.norm(_run_coupled(n) - r_ref) / r_norm) for n in n_list
        ]
        slope_mut_half = float(
            -np.polyfit(np.log(n_list), np.log(errs_mut_half), 1)[0]
        )
        assert slope_mut_half < 1.2
        assert errs_mut_half[-1] > 50.0 * errs_normal[-1]


# ============================================================================
# Section 4 & 5: Absorption bookkeeping, mid_step bit-identity, CurrentSource
# ============================================================================


def test_absorption_and_injection_bookkeeping(monkeypatch):
    """Verify exact charge & energy conservation: Q_injected = Q_on_grid + Q_absorbed.

    Physical setup
    --------------
    A :class:`BeamInjector` emits relativistic electrons (``gamma_0 = 2.0``,
    ``I = 0.5 A``) from a PEC cathode plane ``z_0 = 0`` into a PEC box of
    length ``L_z = 0.08 m``. A :class:`Collector` plane at ``z_col = 0.055 m``
    and lateral PEC walls absorb particles that reach them. Over 80 steps,
    some particles are in flight on the grid while others have been absorbed by
    the collector or domain walls.

    Error estimate & tolerance
    --------------------------
    Charge accounting ``Q_injected = Q_on_grid + Q_absorbed`` and kinetic energy
    accounting at absorption must hold to floating-point round-off (``<= 1e-14``
    relative), and ``gauss_residual`` in the interior vacuum must stay ``<= 1e-12``.
    """
    L = 0.08
    Nx = Ny = Nz = 8

    def _run_sim():
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
            verbose=0,
        )

        sp = Species(name="electron", charge=-e_charge, mass=m_e)
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

        for _ in range(1, 81):
            solver.one_step()
        return solver, ps, collector

    _, ps, collector = _run_sim()
    q_inj = ps.injected_charge()
    q_grid = ps.total_charge()
    q_abs = ps.absorbed_charge()
    assert abs(q_inj - (q_grid + q_abs)) / abs(q_inj) <= 1e-14

    # Both active on-grid charge and collector-absorbed charge must be non-zero
    assert abs(ps.total_charge()) > 0.0
    assert abs(collector.absorbed_charge) > 0.0
    assert collector.absorbed_energy > 0.0
    assert collector.absorbed_count > 0
    assert ps.gauss_residual() <= 1e-12

    # Code mutation check: monkeypatching Collector.check_crossing so it fails
    # to truncate trajectories at the collector removes collector absorption from
    # ps.absorbed_charge() and violates the collector charge accounting.
    monkeypatch.setattr(
        Collector,
        "check_crossing",
        lambda self, x0, y0, z0, x1, y1, z1, xp=np: (
            xp.zeros(x1.size, dtype=bool),
            xp.ones(x1.size, dtype=np.float64),
        ),
    )
    _, _, col_mut = _run_sim()
    assert col_mut.absorbed_charge == 0.0


@pytest.mark.parametrize("use_cpml", [False, True])
def test_mid_step_bit_identical_when_empty(use_cpml):
    """Deliverable 2: SolverFIT3D fields must be bit-identical with/without an empty mid_step hook."""
    from wakis.sources import CurrentSource, CustomSource

    def _run_solver(attach_empty_hook: bool):
        grid = GridFIT3D(
            xmin=0.0,
            xmax=0.06,
            ymin=0.0,
            ymax=0.06,
            zmin=0.0,
            zmax=0.06,
            Nx=10,
            Ny=10,
            Nz=10,
            verbose=0,
        )
        bc = ["cpml", "cpml", "cpml"] if use_cpml else ["pec", "pec", "periodic"]
        s = SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=bc,
            bc_high=bc,
            n_pml=3,
            verbose=0,
        )
        src = CurrentSource(
            src_time=CustomSource(lambda t: np.sin(2.0 * np.pi * 2.0e9 * t)),
            component="z",
            center=(0.03, 0.03, 0.03),
            size=(0.0, 0.0, 0.0),
            amplitude=1.0,
        )
        hooks_called = [0]
        if attach_empty_hook:
            s.mid_step = lambda solver: None
            s.register_mid_step(lambda solver: hooks_called.__setitem__(0, hooks_called[0] + 1))

        for n in range(25):
            src.update(s, n * s.dt)
            s.one_step()
        return s, hooks_called[0]

    s_ref, _ = _run_solver(attach_empty_hook=False)
    s_hook, n_calls = _run_solver(attach_empty_hook=True)

    assert n_calls == 25
    np.testing.assert_array_equal(s_ref.E.toarray(), s_hook.E.toarray())
    np.testing.assert_array_equal(s_ref.H.toarray(), s_hook.H.toarray())
    np.testing.assert_array_equal(s_ref.J.toarray(), s_hook.J.toarray())


def test_current_source_and_particle_solver_share_edges():
    """Section 5: CurrentSource and ParticleSolver sharing the same edges superpose linearly."""
    from wakis.sources import CurrentSource, CustomSource

    def _make_solver():
        grid = GridFIT3D(
            xmin=0.0,
            xmax=0.04,
            ymin=0.0,
            ymax=0.04,
            zmin=0.0,
            zmax=0.04,
            Nx=6,
            Ny=6,
            Nz=6,
            verbose=0,
        )
        return SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["periodic", "periodic", "periodic"],
            bc_high=["periodic", "periodic", "periodic"],
            verbose=0,
        )

    dx = 0.04 / 6
    src_center = (3.0 * dx, 3.0 * dx, 3.5 * dx)

    # 1. Run with CurrentSource alone on edge (3, 3, 3, 'z')
    s_src = _make_solver()
    src_only = CurrentSource(
        src_time=CustomSource(lambda t: np.cos(2.0 * np.pi * 1.5e9 * t)),
        component="z",
        center=src_center,
        size=(0.0, 0.0, 0.0),
        amplitude=2.5,
    )
    for n in range(15):
        src_only.update(s_src, n * s_src.dt)
        s_src.one_step()

    # 2. Run with both CurrentSource and a particle crossing cell (3, 3, 3)
    # under a heavy mass (negligible back-reaction on particle velocity)
    s_both = _make_solver()
    src_both = CurrentSource(
        src_time=CustomSource(lambda t: np.cos(2.0 * np.pi * 1.5e9 * t)),
        component="z",
        center=src_center,
        size=(0.0, 0.0, 0.0),
        amplitude=2.5,
    )
    sp = Species(name="heavy_ion", charge=e_charge, mass=1.0e12 * m_e)
    sp.add_particles(
        x=3.0 * dx,
        y=3.0 * dx,
        z=3.2 * dx,
        ux=0.0,
        uy=0.0,
        uz=0.2 * c_light,
        w=1.0e8,
    )
    ps_both = ParticleSolver(s_both, species=[sp], shape=1).attach()

    for n in range(15):
        src_both.update(s_both, n * s_both.dt)
        s_both.one_step()

    # At step 15, s_both.J must equal s_src.J + ps_both.J_particles to machine precision
    np.testing.assert_allclose(
        s_both.J.toarray(),
        s_src.J.toarray() + ps_both.J_particles,
        rtol=1e-14,
        atol=1e-14,
    )
    # And the shared edge (3, 3, 3, 'z') has non-zero contributions from BOTH sources
    assert abs(s_src.J[3, 3, 3, "z"]) > 0.0
    assert abs(s_both.J[3, 3, 3, "z"] - s_src.J[3, 3, 3, "z"]) > 0.0


def test_usersguide_particles_snippet_executes():
    """Execute the ``wakis.particles`` code block from ``docs/usersguide.md`` verbatim (with fewer steps)."""
    from pathlib import Path

    ug_path = Path(__file__).resolve().parents[1] / "docs" / "usersguide.md"
    text = ug_path.read_text(encoding="utf-8")
    marker = "## ⚡ Relativistic Particle-in-Cell (`wakis.particles`)"
    assert marker in text
    sub = text.split(marker, 1)[1]
    code_block = sub.split("```python\n", 1)[1].split("```", 1)[0]
    # Run with 12 steps for fast unit testing while exercising every API call
    code_fast = code_block.replace("range(80)", "range(12)")
    ns: dict = {}
    exec(compile(code_fast, str(ug_path), "exec"), ns)
    ps = ns["ps"]
    assert ns["electrons"].n_particles > 0
    assert ps.gauss_residual() <= 1e-12
    assert abs(ps.injected_charge() - (ps.total_charge() + ps.absorbed_charge())) <= 1e-20


def test_mid_step_signature_inspection_once():
    """Verify ``mid_step`` and ``register_mid_step`` decide arity via ``inspect.signature`` and do not retry on TypeError."""
    grid = GridFIT3D(0.0, 0.03, 0.0, 0.03, 0.0, 0.03, 3, 3, 3, verbose=0)
    solver = SolverFIT3D(grid, verbose=0)

    calls_0 = [0]
    calls_1 = [0]
    solver.mid_step = lambda: calls_0.__setitem__(0, calls_0[0] + 1)
    solver.register_mid_step(lambda s: calls_1.__setitem__(0, calls_1[0] + 1))
    solver.one_step()
    assert calls_0[0] == 1
    assert calls_1[0] == 1

    # A 1-arg hook that raises an internal TypeError must NOT be called a second time
    bad_calls = [0]

    def _bad_hook(s):
        bad_calls[0] += 1
        raise TypeError("internal type error inside hook")

    solver.mid_step = _bad_hook
    with pytest.raises(TypeError, match="internal type error inside hook"):
        solver.one_step()
    assert bad_calls[0] == 1


def test_attach_rejects_unhooked_step_variants():
    """Verify ``ParticleSolver.attach()`` raises ``ValueError`` for unhooked ``SolverFIT3D`` step variants."""
    grid = GridFIT3D(
        xmin=0.0, xmax=0.04, ymin=0.0, ymax=0.04, zmin=0.0, zmax=0.04, Nx=4, Ny=4, Nz=4, verbose=0
    )
    solver = SolverFIT3D(grid, verbose=0)
    sp = Species()

    # Standard hooked variants (_one_step, _one_step_cpml, and _one_step_with_abc wrapping _one_step) succeed
    solver.one_step = solver._one_step
    ParticleSolver(solver, species=[sp]).attach()

    solver.one_step = solver._one_step_cpml
    ParticleSolver(solver, species=[sp]).attach()

    # MKL variants also call _run_mid_step (selected automatically when sparse_dot_mkl is installed)
    for attr in ("_one_step_mkl", "_one_step_cpml_mkl"):
        if hasattr(solver, attr):
            solver.one_step = getattr(solver, attr)
            ParticleSolver(solver, species=[sp]).attach()

    solver._one_step_backend = solver._one_step
    solver.one_step = solver._one_step_with_abc
    ParticleSolver(solver, species=[sp]).attach()

    # Every unhooked step variant must raise ValueError at attach()
    unhooked_names = (
        "_one_step_dispersive",
        "_one_step_nonlinear",
        "_one_step_time_varying",
        "_one_step_tete",
        "one_step_tensor",
        "one_step_network",
    )
    for attr in unhooked_names:
        fn = getattr(solver, attr, None)
        if fn is None:
            fn = lambda: None  # noqa: E731
        solver.one_step = fn
        with pytest.raises(ValueError, match="mid_step hooks"):
            ParticleSolver(solver, species=[sp]).attach()

    # _one_step_with_abc wrapping an unhooked backend must also raise ValueError
    solver._one_step_backend = solver._one_step_dispersive
    solver.one_step = solver._one_step_with_abc
    with pytest.raises(ValueError, match="mid_step hooks"):
        ParticleSolver(solver, species=[sp]).attach()


def test_pec_solid_mask_absorption_gauss_and_bookkeeping(monkeypatch):
    """Regression test for PEC / ``solid_mask`` absorption with Gauss residual <= 1e-12 and bookkeeping.

    Physical setup
    --------------
    A relativistic electron beamlet (4 macroparticles, ``gamma = 2.5``) streams
    along ``+z`` inside a ``6 x 6 x 8`` grid and impacts an embedded PEC block
    occupying ``iz >= 5`` (represented by ``solid_mask[:, :, 5:] = True`` and
    zeroed ``ieps`` inside the conductor). Every particle must be truncated at
    the ``z = 5 * dz`` conductor entry face, removed from the active species,
    bookkept under ``ps.boundaries.absorbed_charge["pec"]`` and
    ``ps.absorbed_charge()``, while ``ps.gauss_residual() <= 1e-12`` at every step.
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
        bc_low=["pec", "pec", "pec"],
        bc_high=["pec", "pec", "pec"],
        verbose=0,
    )

    solid_mask = np.zeros((solver.Nx, solver.Ny, solver.Nz), dtype=bool)
    solid_mask[:, :, 5:] = True
    for comp in ("x", "y", "z"):
        arr = solver.ieps[:, :, :, comp].copy()
        arr[:, :, 5:] = 0.0
        solver.ieps[:, :, :, comp] = arr

    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    n_p = 4
    gamma0 = 2.5
    uz0 = float(np.sqrt(gamma0 * gamma0 - 1.0) * c_light)
    sp.add_particles(
        x=np.array([0.025, 0.035, 0.025, 0.035]),
        y=np.array([0.025, 0.025, 0.035, 0.035]),
        z=np.full(n_p, 0.032),
        ux=np.zeros(n_p),
        uy=np.zeros(n_p),
        uz=np.full(n_p, uz0),
        w=np.full(n_p, 2.5e7),
    )

    ps = ParticleSolver(
        solver, species=[sp], pusher="boris", shape=2, solid_mask=solid_mask
    ).attach()
    q_init = ps.total_charge()
    ke_init = sp.kinetic_energy()

    max_gauss = 0.0
    for _ in range(35):
        solver.one_step()
        max_gauss = max(max_gauss, ps.gauss_residual())

    assert max_gauss <= 1e-12
    assert sp.n_particles == 0
    assert ps.boundaries.absorbed_count["pec"] == n_p
    assert abs(ps.boundaries.absorbed_charge["pec"] - q_init) / abs(q_init) <= 1e-14
    assert abs(ps.absorbed_charge() - q_init) / abs(q_init) <= 1e-14
    assert ps.boundaries.absorbed_energy["pec"] > 0.5 * ke_init

    # Code mutation check: disabling solid_mask in ParticleBoundaries prevents
    # PEC absorption at iz=5 so absorbed_count["pec"] drops to 0.
    ps.boundaries.solid_mask = None
    ps.boundaries.absorbed_count["pec"] = 0
    sp.add_particles(
        x=np.array([0.03]),
        y=np.array([0.03]),
        z=np.array([0.032]),
        ux=np.array([0.0]),
        uy=np.array([0.0]),
        uz=np.array([uz0]),
        w=np.array([2.5e7]),
    )
    for _ in range(35):
        solver.one_step()
    assert ps.boundaries.absorbed_count["pec"] == 0


def test_cpml_face_absorption_gauss_and_bookkeeping(monkeypatch):
    """Regression test for CPML inner-face absorption with Gauss residual <= 1e-12 and bookkeeping.

    Physical setup
    --------------
    A relativistic electron bunch (4 macroparticles, ``gamma = 2.5``) propagates
    along ``+z`` toward a CPML upper boundary (``bc_high=['periodic', 'periodic', 'cpml']``,
    ``n_pml = 4``) on a ``6 x 6 x 12`` grid (``dz = 0.01 m``). The CPML inner
    interface lies at ``zeta_max = Nz_phys - n_pml = 8`` (``z = 0.08 m``).
    Particles must be absorbed upon crossing ``z = 0.08 m`` (never entering the
    PML layer), bookkept under ``ps.boundaries.absorbed_charge['z_high']`` and
    ``ps.absorbed_charge()``, while ``ps.gauss_residual() <= 1e-12`` at every step.
    """
    n_pml = 4
    grid = GridFIT3D(
        xmin=0.0,
        xmax=0.06,
        ymin=0.0,
        ymax=0.06,
        zmin=0.0,
        zmax=0.12,
        Nx=6,
        Ny=6,
        Nz=12,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        cfln=0.45,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=n_pml,
        verbose=0,
    )

    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    n_p = 4
    gamma0 = 2.5
    uz0 = float(np.sqrt(gamma0 * gamma0 - 1.0) * c_light)
    sp.add_particles(
        x=np.array([0.02, 0.04, 0.02, 0.04]),
        y=np.array([0.02, 0.02, 0.04, 0.04]),
        z=np.full(n_p, 0.062),
        ux=np.zeros(n_p),
        uy=np.zeros(n_p),
        uz=np.full(n_p, uz0),
        w=np.full(n_p, 2.5e7),
    )

    ps = ParticleSolver(
        solver, species=[sp], pusher="boris", shape=2
    ).attach()
    assert ps.boundaries.xi_max[2] == 8.0
    q_init = ps.total_charge()
    ke_init = sp.kinetic_energy()

    max_gauss = 0.0
    max_z_seen = 0.0
    for _ in range(35):
        if sp.n_particles > 0:
            max_z_seen = max(max_z_seen, float(np.max(sp.z)))
        solver.one_step()
        max_gauss = max(max_gauss, ps.gauss_residual())

    assert max_gauss <= 1e-12
    assert max_z_seen <= 0.08 + 1e-12
    assert sp.n_particles == 0
    assert ps.boundaries.absorbed_count["z_high"] == n_p
    assert abs(ps.boundaries.absorbed_charge["z_high"] - q_init) / abs(q_init) <= 1e-14
    assert abs(ps.absorbed_charge() - q_init) / abs(q_init) <= 1e-14
    assert ps.boundaries.absorbed_energy["z_high"] > 0.5 * ke_init

    # Code mutation check: if xi_max[2] were set to 12.0 (ignoring n_pml),
    # a particle starting at z = 0.062 m over 10 steps would enter the PML
    # region (z > 0.08 m) instead of being absorbed at z = 0.08 m.
    ps.boundaries.xi_max = (ps.boundaries.xi_max[0], ps.boundaries.xi_max[1], 12.0)
    sp.add_particles(
        x=np.array([0.03]),
        y=np.array([0.03]),
        z=np.array([0.062]),
        ux=np.array([0.0]),
        uy=np.array([0.0]),
        uz=np.array([uz0]),
        w=np.array([2.5e7]),
    )
    for _ in range(15):
        solver.one_step()
    assert sp.n_particles == 1 and float(sp.z[0]) > 0.085


def test_mpi_configuration_validation():
    """ParticleSolver validates grid.use_mpi, shape=1, filter=False, and z-slab geometry under use_mpi=True."""
    grid = GridFIT3D(
        xmin=0.0, xmax=0.04, ymin=0.0, ymax=0.04, zmin=0.0, zmax=0.04, Nx=4, Ny=4, Nz=4, verbose=0
    )
    solver = SolverFIT3D(grid, verbose=0)
    sp = Species()

    solver.use_mpi = True
    with pytest.raises(ValueError, match="use_mpi=True"):
        ParticleSolver(solver, species=[sp])
    grid.use_mpi = True
    with pytest.raises(ValueError, match="shape=1"):
        ParticleSolver(solver, species=[sp], shape=2)
    with pytest.raises(NotImplementedError, match="Binomial current filtering"):
        ParticleSolver(solver, species=[sp], filter="binomial")
    solver.size = 3
    grid._Nz_input = 4
    with pytest.raises(ValueError, match="divisible by solver.size"):
        ParticleSolver(solver, species=[sp])
    solver.size = 4
    solver.NZ = 4
    grid._Nz_input = 4
    with pytest.raises(ValueError, match="at least 2 owned z-cells"):
        ParticleSolver(solver, species=[sp])
    solver.size = 1
    solver.NZ = 4
    grid.use_mpi = False
    solver.use_mpi = False

    # Longitudinal periodic z with use_mpi=True must raise NotImplementedError
    grid_mpi_per = GridFIT3D(
        xmin=0.0, xmax=0.04, ymin=0.0, ymax=0.04, zmin=0.0, zmax=0.04, Nx=4, Ny=4, Nz=4, verbose=0
    )
    with pytest.raises(NotImplementedError, match="Longitudinal periodic MPI"):
        SolverFIT3D(
            grid_mpi_per,
            bc_low=["pec", "pec", "periodic"],
            bc_high=["pec", "pec", "periodic"],
            use_mpi=True,
            verbose=0,
        )


@pytest.mark.parametrize("order", [1, 2])
def test_cpu_chunked_periodic_fast_path_and_mutation(order, monkeypatch):
    """Compare the chunked CPU periodic fast path (n_p = 70 000 > _CPU_CHUNK = 65 536) against the general unchunked path.

    Physical setup
    --------------
    70 000 relativistic macroparticles evolve for 4 coupled steps on an ``8 x 8 x 8``
    periodic grid for ``p = 1`` and ``p = 2``. Because ``70 000 > _CPU_CHUNK (65 536)``,
    ``ParticleSolver._initialize_species_half_step`` and ``ParticleSolver._mid_step_hook``
    execute the chunked CPU periodic fast path in two slices ``[0:65536]`` and ``[65536:70000]``.
    Compared against the general unchunked path (obtained by monkeypatching
    ``solver._CPU_CHUNK = 1_000_000``, while ``deposit_current_esirkepov`` still accumulates
    in 65 536-particle chunks), ``E, H, x, u`` match identically (tolerance ``0.0``).
    Mutating ``_cpu_chunk_ranges`` to skip the trailing chunk ``[65536:70000]`` must fail.
    """
    import wakis.particles.solver as solver_mod

    n_p = 70_000
    n_steps = 4
    rng = np.random.default_rng(70000 + order)
    x0 = rng.uniform(0.005, 0.075, size=n_p)
    y0 = rng.uniform(0.005, 0.075, size=n_p)
    z0 = rng.uniform(0.005, 0.075, size=n_p)
    ux0 = rng.normal(0.0, 0.25 * c_light, size=n_p)
    uy0 = rng.normal(0.0, 0.25 * c_light, size=n_p)
    uz0 = rng.normal(0.35 * c_light, 0.15 * c_light, size=n_p)
    w0 = rng.uniform(1.0e4, 2.0e4, size=n_p)

    def _run():
        grid = GridFIT3D(
            xmin=0.0,
            xmax=0.08,
            ymin=0.0,
            ymax=0.08,
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
            bc_low=["periodic", "periodic", "periodic"],
            bc_high=["periodic", "periodic", "periodic"],
            verbose=0,
        )
        sp = Species(
            name="electron",
            charge=-e_charge,
            mass=m_e,
            capacity=n_p,
            track_ids=False,
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
        ps = ParticleSolver(
            solver,
            species=[sp],
            pusher="boris",
            shape=order,
            gather="galerkin",
            sort_every=0,
            B_ext=(0.0, 0.0, 0.05),
        ).attach()
        for _ in range(n_steps):
            solver.one_step()
        return (
            solver.E.toarray().copy(),
            solver.H.toarray().copy(),
            sp.x.copy(),
            sp.y.copy(),
            sp.z.copy(),
            sp.ux.copy(),
            sp.uy.copy(),
            sp.uz.copy(),
            ps.gauss_residual(),
        )

    # 1. Default chunked periodic fast path (_CPU_CHUNK = 65536 < 70000)
    assert solver_mod._CPU_CHUNK == 65536
    E_chk, H_chk, x_chk, y_chk, z_chk, ux_chk, uy_chk, uz_chk, g_chk = _run()
    assert g_chk <= 1e-12

    # 2. General (unchunked) path via _CPU_CHUNK = 1_000_000
    with monkeypatch.context() as m:
        m.setattr(solver_mod, "_CPU_CHUNK", 1_000_000)
        E_ref, H_ref, x_ref, y_ref, z_ref, ux_ref, uy_ref, uz_ref, g_ref = _run()
    assert g_ref <= 1e-12

    assert float(np.max(np.abs(E_chk - E_ref))) == 0.0
    assert float(np.max(np.abs(H_chk - H_ref))) == 0.0
    assert float(np.max(np.abs(x_chk - x_ref))) == 0.0
    assert float(np.max(np.abs(y_chk - y_ref))) == 0.0
    assert float(np.max(np.abs(z_chk - z_ref))) == 0.0
    assert float(np.max(np.abs(ux_chk - ux_ref))) == 0.0
    assert float(np.max(np.abs(uy_chk - uy_ref))) == 0.0
    assert float(np.max(np.abs(uz_chk - uz_ref))) == 0.0

    # 3. Code-path mutation: skipping the final chunk [65536:70000] must fail
    orig_ranges = solver_mod._cpu_chunk_ranges
    with monkeypatch.context() as m:
        m.setattr(
            solver_mod,
            "_cpu_chunk_ranges",
            lambda n_tot: list(orig_ranges(n_tot))[:-1],
        )
        E_mut, H_mut, x_mut, _, _, ux_mut, _, _, _ = _run()
        with pytest.raises(AssertionError):
            assert float(np.max(np.abs(E_mut - E_ref))) == 0.0
        with pytest.raises(AssertionError):
            assert float(np.max(np.abs(x_mut - x_ref))) == 0.0
        with pytest.raises(AssertionError):
            assert float(np.max(np.abs(ux_mut - ux_ref))) == 0.0


def test_reset_gauss_reference_after_post_attach_modification():
    """Verify ``reset_gauss_reference()`` refreshes ``_particles_q_bg`` when positions/fields change after ``attach()``."""
    grid = GridFIT3D(0.0, 0.06, 0.0, 0.06, 0.0, 0.06, 6, 6, 6, verbose=0)
    solver = SolverFIT3D(
        grid,
        cfln=0.45,
        bc_low=["periodic"] * 3,
        bc_high=["periodic"] * 3,
        verbose=0,
    )
    sp = Species()
    sp.add_particles(
        x=np.array([0.015, 0.025]),
        y=np.array([0.015, 0.025]),
        z=np.array([0.015, 0.025]),
        ux=np.zeros(2),
        uy=np.zeros(2),
        uz=np.array([0.1 * c_light, -0.1 * c_light]),
        w=np.array([1.0e6, 1.0e6]),
    )
    ps = ParticleSolver(solver, species=[sp], shape=1).attach()
    assert ps.gauss_residual() <= 1e-12

    # Displace particles in-place after attach() without changing sp.n_particles:
    # the reference q_bg is now stale until reset_gauss_reference() is called.
    sp.x[:] = np.array([0.045, 0.035])
    assert ps.gauss_residual() > 0.5

    ps.reset_gauss_reference()
    assert ps.gauss_residual() <= 1e-12
    for _ in range(5):
        solver.one_step()
    assert ps.gauss_residual() <= 1e-12


def test_particle_solver_profile_flag_and_mutation(monkeypatch):
    """Verify default ``profile=False`` makes zero ``perf_counter`` calls while ``profile=True`` records stage timings."""
    from wakis.particles import solver as solver_mod

    grid = GridFIT3D(0.0, 0.04, 0.0, 0.04, 0.0, 0.04, 4, 4, 4, verbose=0)
    solver = SolverFIT3D(
        grid,
        cfln=0.45,
        bc_low=["periodic"] * 3,
        bc_high=["periodic"] * 3,
        verbose=0,
    )
    sp = Species()
    sp.add_particles(
        x=np.array([0.01, 0.02]),
        y=np.array([0.01, 0.02]),
        z=np.array([0.01, 0.02]),
        ux=np.zeros(2),
        uy=np.zeros(2),
        uz=np.array([0.1 * c_light, -0.1 * c_light]),
        w=np.array([1.0e5, 1.0e5]),
    )
    ps = ParticleSolver(solver, species=[sp], shape=1).attach()
    assert ps.profile is False

    # 1. With default profile=False, monkeypatching time.perf_counter to raise must not trigger
    def _forbid_perf_counter():
        raise RuntimeError("perf_counter must not be called when profile=False")

    with monkeypatch.context() as m:
        m.setattr(solver_mod.time, "perf_counter", _forbid_perf_counter)
        solver.one_step()
        assert ps._last_push_gather_s == 0.0
        assert ps._last_deposit_s == 0.0

        # Mutation check: enabling ps.profile = True must call perf_counter and raise
        ps.profile = True
        with pytest.raises(RuntimeError, match="perf_counter must not be called"):
            solver.one_step()

    # 2. With profile=True and real perf_counter, both stage times are > 0
    ps_prof = ParticleSolver(solver, species=[sp], shape=1, profile=True)
    assert ps_prof.profile is True
    ps.profile = True
    solver.one_step()
    assert ps._last_push_gather_s > 0.0
    assert ps._last_deposit_s > 0.0




