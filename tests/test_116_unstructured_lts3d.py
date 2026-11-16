# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Validation and mutation tests for 3D tetrahedral local time stepping (`wakis.unstructured.lts3d`).

All test-bed inputs (mesh sizes h, jitter amplitudes, RNG seeds, and step
multipliers such as 2.2) are test design choices and not study results.
"""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
from scipy.constants import c as c_0
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from wakis.dispersive import DispersiveMaterial, Lorentz
from wakis.resonances import harminv
from wakis.unstructured.adjoint import (
    CellDFTMonitor,
    CellMaterialGrid,
    UnstructuredAdjointSimulation,
    UnstructuredAdjointStepper,
)
from wakis.unstructured.gpu import DeviceStepper, massless_schur_operator
from wakis.unstructured.lts import chebyshev_coefficients
from wakis.unstructured.lts3d import (
    TetLTS,
    effective_stiffness,
    element_levels,
    frequency_shift,
    global_limit_ratio,
    is_stable,
    lts_dt_max,
    mass_matrix,
    modified_energy,
    nest_levels,
    r_prime_0,
    stability_ratio,
    theta_factor,
)
from wakis.unstructured.meshers import ball_mesh, box_mesh
from wakis.unstructured.mpi import DistributedFIT, LocalCluster
from wakis.unstructured.pml3d import ball_with_pml_mesh
from wakis.unstructured.ports3d import LumpedPort, attach_ports
from wakis.unstructured.quality import element_stability
from wakis.unstructured.solver3d import UnstructuredFIT3D

DOT_TOL = 1e-14
EPS = np.finfo(float).eps


def blockwise_dot(st, seed=0):
    """Max scale-invariant dot-product error over all (input, output) blocks."""
    rng = np.random.default_rng(seed)
    zero = st.zero_state()
    keys = list(zero)
    fwd, adj = {}, {}
    for k in keys:
        x = {kk: v.copy() for kk, v in zero.items()}
        x[k] = rng.standard_normal(x[k].size)
        fwd[k] = (x, st.step(x, 0, sources=False)[0])
        y = {kk: v.copy() for kk, v in zero.items()}
        y[k] = rng.standard_normal(y[k].size)
        adj[k] = (y, st.step_T(y)[0])
    worst = 0.0
    for i in keys:
        x, Ax = fwd[i]
        for j in keys:
            y, ATy = adj[j]
            a = float(Ax[j] @ y[j])
            b = float(x[i] @ ATy[i])
            scale = np.linalg.norm(Ax[j]) * np.linalg.norm(y[j]) + np.linalg.norm(
                x[i]
            ) * np.linalg.norm(ATy[i])
            if scale > 0:
                worst = max(worst, abs(a - b) / scale)
    return worst


# ------------------------------------------------------------------ meshes
_CACHE = {}


def _small_box(jitter=0.2, seed=1):
    """Test-bed PEC box 1.0 x 0.75 x 0.5 m with non-power-of-two divisions (test design choice)."""
    key = ("small_box", jitter, seed)
    if key not in _CACHE:
        xs = np.linspace(0.0, 1.0, 7)
        ys = np.linspace(0.0, 0.75, 6)
        zs = np.linspace(0.0, 0.5, 5)
        _CACHE[key] = box_mesh(xs, ys, zs, jitter=jitter, seed=seed)
    return _CACHE[key]


def _graded_box(seed=1):
    """Test-bed graded PEC box refined toward x = 0 (test design choice)."""
    key = ("graded_box", seed)
    if key not in _CACHE:
        s = np.linspace(0.0, 1.0, 7)
        xs = 1.0 * (s**1.7)
        ys = np.linspace(0.0, 0.75, 6)
        zs = np.linspace(0.0, 0.5, 5)
        _CACHE[key] = box_mesh(xs, ys, zs, jitter=0.18, seed=seed)
    return _CACHE[key]


def _cavity_box(seed=1):
    """Test-bed PEC box 1.0 x 0.75 x 0.5 m for TM110 frequency check (test design choice)."""
    key = ("cavity_box", seed)
    if key not in _CACHE:
        xs = np.linspace(0.0, 1.0, 10)
        ys = np.linspace(0.0, 0.75, 8)
        zs = np.linspace(0.0, 0.5, 6)
        _CACHE[key] = box_mesh(xs, ys, zs, jitter=0.2, seed=seed)
    return _CACHE[key]


def _small_ball():
    """Test-bed ODT-smoothed ball mesh (test design choice)."""
    key = ("small_ball",)
    if key not in _CACHE:
        _CACHE[key] = ball_mesh([1.0], 0.35)
    return _CACHE[key]


def _nested_test_levels(solver, ratios):
    """Construct a deterministic non-trivial nested partition on `solver`."""
    L = len(ratios)
    K = effective_stiffness(solver)
    M, M_diag = mass_matrix(solver)
    m_diag = M_diag if M_diag is not None else np.asarray(M.diagonal(), dtype=float)
    rate = np.asarray(K.diagonal(), dtype=float) / np.maximum(m_diag, 1e-30)
    q = np.quantile(rate, np.linspace(0.70, 0.92, L))
    lvl = np.zeros(solver.free.size, dtype=int)
    for lv in range(1, L + 1):
        lvl[rate >= q[lv - 1]] = lv
    return nest_levels(K, lvl)


# ------------------------------------------------------------------ 1. known limits
@pytest.mark.parametrize("hodge", ["voronoi", "whitney"])
def test_known_limits(hodge):
    """
    Known algebraic limits of `TetLTS`:
    (i) F_1 = empty reproduces `lts=None` bitwise over 50 steps;
    (ii) F_1 = all, L = 1, nu = 0, p in {2, 3} obeys the Chebyshev identity
         dt^2 S_0 v = 2 M (I - T_p(I - tau^2 M^-1 K / 2)) v to <= 1e-12.
    Mutation: `mutate='consistency'` breaks the identity (> 1e-3).
    """
    m = _small_box()
    s_ref = UnstructuredFIT3D(m, hodge=hodge)
    zero_lvl = np.zeros(s_ref.free.size, dtype=int)
    s_lts = UnstructuredFIT3D(
        m, hodge=hodge, dt=s_ref.dt, lts={"ratios": [2], "edge_level": zero_lvl}
    )
    rng = np.random.default_rng(1)
    u0 = rng.standard_normal(s_ref.free.size)
    b0 = rng.standard_normal(m.n_faces)
    s_ref.u[:], s_ref.b[:] = u0, b0
    s_lts.u[:], s_lts.b[:] = u0, b0
    s_ref.run(50)
    s_lts.run(50)
    assert np.array_equal(s_lts.u, s_ref.u)
    assert np.array_equal(s_lts.b, s_ref.b)

    K = effective_stiffness(s_ref)
    M, M_diag = mass_matrix(s_ref)
    Minv = (lambda x: x / M_diag) if M_diag is not None else s_ref._M1.solve
    all_lvl = np.ones(s_ref.free.size, dtype=int)

    max_rel = 0.0
    min_mut = np.inf
    for p in (2, 3):
        dt = s_ref.dt
        tau = dt / p
        lts = TetLTS(s_ref, dt, all_lvl, [p], nu=0.0)
        lts_mut = TetLTS(s_ref, dt, all_lvl, [p], nu=0.0, mutate="consistency")

        def Z_op(w, tau=tau):
            return w - 0.5 * (tau**2) * Minv(K @ w)

        for _ in range(3):
            v = rng.standard_normal(s_ref.free.size)
            lhs = (dt**2) * (K @ lts.apply(v))
            z1 = Z_op(v)
            z2 = Z_op(z1)
            if p == 2:
                tp_v = 2.0 * z2 - v
            else:
                tp_v = 4.0 * Z_op(z2) - 3.0 * z1
            rhs = 2.0 * (M @ (v - tp_v))
            rel = float(np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs))
            max_rel = max(max_rel, rel)
            assert rel <= 1e-12, f"Chebyshev identity failed: rel={rel:.3e}"

            lhs_mut = (dt**2) * (K @ lts_mut.apply(v))
            rel_mut = float(np.linalg.norm(lhs_mut - rhs) / np.linalg.norm(rhs))
            min_mut = min(min_mut, rel_mut)
            assert rel_mut > 1e-3, (
                f"Mutation 'consistency' did not fail: rel_mut={rel_mut:.3e}"
            )

    print(
        f"test_known_limits[{hodge}]: bitwise_empty=True, max_cheb_rel={max_rel:.3e}, "
        f"min_mut_consistency={min_mut:.3e}"
    )


# ------------------------------------------------------------------ 2. fast == ref == dense
def _dense_apply(solver, dt_s, edge_level, ratios, nus):
    """Explicit dense evaluation of R_0 using numpy.linalg.inv(M_ll)."""
    K = effective_stiffness(solver).toarray()
    M, _ = mass_matrix(solver)
    M = M.toarray()
    n = solver.free.size
    L = len(ratios)
    tau_s = [float(dt_s)]
    for p in ratios:
        tau_s.append(tau_s[-1] / p)

    def R_dense(lv):
        if lv >= L:
            return np.eye(n)
        m = lv + 1
        p = ratios[m - 1]
        nu = nus[m - 1]
        Rm = R_dense(m)
        F = np.flatnonzero(edge_level >= m)
        if F.size == 0 or p == 1:
            return Rm
        M_ll_inv = np.linalg.inv(M[np.ix_(F, F)])
        Q = np.zeros((n, n))
        Q[np.ix_(F, F)] = M_ll_inv
        X = (tau_s[m - 1] ** 2) * (Q @ K @ Rm)
        delta, beta, T, _, out = chebyshev_coefficients(p, nu)
        eye = np.eye(n)
        d_old = np.zeros((n, n))
        dk = eye.copy()
        for k in range(1, p):
            d_new = 2.0 * delta * dk - 2.0 * beta * (X @ dk) + 2.0 * T[k] * eye - d_old
            d_old, dk = dk, d_new
        return Rm @ (out * dk)

    return R_dense(0)


@pytest.mark.parametrize("hodge", ["voronoi", "whitney"])
@pytest.mark.parametrize("ratios", [[2], [2, 2], [2, 3]])
def test_fast_equals_reference_and_dense(hodge, ratios):
    """
    Fast `apply` matches `apply_full` (<= 1e-13), `matrix()` (<= 1e-13), and an
    independent dense matrix-inverse evaluation (<= 1e-12).
    Mutations: `mutate='interface'` and `mutate='drop_buffer'` both fail (> 1e-6).
    """
    m = _small_box()
    s = UnstructuredFIT3D(m, hodge=hodge)
    lvl = _nested_test_levels(s, ratios)
    dt = 0.8 * s.dt_max
    lts = TetLTS(s, dt, lvl, ratios, nu=0.05)
    assert all(
        lts.levels[lv].F.size > 0 and lts.levels[lv].C.size > 0
        for lv in range(1, lts.L + 1)
    )

    rng = np.random.default_rng(2)
    u = rng.standard_normal(s.free.size)
    ref = lts.apply_full(u)
    fast = lts.apply(u)
    mat_u = lts.matrix() @ u
    R_den = _dense_apply(s, dt, lvl, ratios, lts.nus)
    den_u = R_den @ u

    nrm = np.linalg.norm(ref)
    err_fast = float(np.linalg.norm(fast - ref) / nrm)
    err_mat = float(np.linalg.norm(mat_u - ref) / nrm)
    err_den = float(np.linalg.norm(den_u - ref) / nrm)
    assert err_fast <= 1e-13, f"fast vs ref: {err_fast:.3e}"
    assert err_mat <= 1e-13, f"matrix vs ref: {err_mat:.3e}"
    assert err_den <= 1e-12, f"dense vs ref: {err_den:.3e}"

    lts_int = TetLTS(s, dt, lvl, ratios, nu=0.05, mutate="interface")
    lts_drp = TetLTS(s, dt, lvl, ratios, nu=0.05, mutate="drop_buffer")
    err_int = float(np.linalg.norm(lts_int.apply(u) - ref) / nrm)
    err_drp = float(np.linalg.norm(lts_drp.apply(u) - ref) / nrm)
    assert err_int > 1e-6, f"mutation 'interface' did not fail: {err_int:.3e}"
    assert err_drp > 1e-6, f"mutation 'drop_buffer' did not fail: {err_drp:.3e}"

    print(
        f"test_fast_equals_reference_and_dense[{hodge},{ratios}]: "
        f"fast={err_fast:.3e}, mat={err_mat:.3e}, dense={err_den:.3e}; "
        f"mut_interface={err_int:.3e}, mut_drop_buffer={err_drp:.3e}"
    )


# ------------------------------------------------------------------ 3. symmetry & dense criterion
@pytest.mark.parametrize("hodge", ["voronoi", "whitney"])
def test_stiffness_symmetric_and_criterion_matches_dense(hodge):
    """
    `S_0 = K R_0` is symmetric to <= 1e-13 relative max-norm, and `is_stable(tol=1e-12)`
    matches dense generalized eigenvalues at 0.99 and 1.02 * `lts_dt_max` with
    |rho_Lanczos - rho_dense| <= 1e-8.
    Mutation: `mutate='interface'` breaks symmetry (> 1e-4 relative).
    """
    m = _small_ball()
    s = UnstructuredFIT3D(m, hodge=hodge)
    ratios = [2]
    lvl = _nested_test_levels(s, ratios)
    dt_lts, _ = lts_dt_max(s, 2.2 * s.dt_max, rtol=1e-4, edge_level=lvl, ratios=ratios)

    max_asym = 0.0
    max_rho_diff = 0.0
    for fac, expected_ok in ((0.99, True), (1.02, False)):
        L_op = TetLTS(s, fac * dt_lts, lvl, ratios)
        S0, M = L_op.stiffness()
        A = S0.toarray()
        M_den = M.toarray()
        asym = float(np.abs(A - A.T).max() / np.abs(A).max())
        max_asym = max(max_asym, asym)
        assert asym <= 1e-13, f"asymmetry={asym:.3e}"

        eigs = la.eigvalsh(0.5 * (A + A.T), M_den)
        rho_dense = float(0.5 * L_op.dt_s * np.sqrt(max(eigs[-1], 0.0)))
        dense_ok = bool(eigs[0] >= -1e-11 * eigs[-1] and rho_dense < 1.0)
        ok, rho_lanczos, marg = is_stable(L_op, tol=1e-12)
        rho_diff = abs(rho_lanczos - rho_dense)
        max_rho_diff = max(max_rho_diff, rho_diff)
        assert ok == expected_ok, (
            f"fac={fac}: ok={ok}, margins={marg}, rho={rho_lanczos}"
        )
        assert dense_ok == expected_ok, (
            f"fac={fac}: dense_ok={dense_ok}, min_eig={eigs[0]:.3e}, rho={rho_dense}"
        )
        assert rho_diff <= 1e-8, f"rho_diff={rho_diff:.3e}"

    L_mut = TetLTS(s, 0.99 * dt_lts, lvl, ratios, mutate="interface")
    A_mut = L_mut.stiffness()[0].toarray()
    asym_mut = float(np.abs(A_mut - A_mut.T).max() / np.abs(A_mut).max())
    assert asym_mut > 1e-4, f"mutation 'interface' asymmetry too small: {asym_mut:.3e}"

    print(
        f"test_stiffness_symmetric[{hodge}]: dt_lts/dt_max={dt_lts / s.dt_max:.4f}, "
        f"max_asym={max_asym:.3e}, max_rho_diff={max_rho_diff:.3e}, mut_asym={asym_mut:.3e}"
    )


# ------------------------------------------------------------------ 4. modified energy conserved
@pytest.mark.parametrize("hodge", ["voronoi", "whitney"])
def test_modified_energy_conserved(hodge):
    """
    Modified energy `E^{n+1/2}` is conserved over 2000 coarse steps at
    0.95 * `lts_dt_max` (`L = 2`) within `50 * eps * sqrt(n_steps)`.
    For Voronoi, asserts `n_massless > 0`.
    Mutations: `interface` and `drop_buffer` drift by > 1e-6.
    """
    m = _graded_box(seed=2)
    s0 = UnstructuredFIT3D(m, hodge=hodge, massless_tol=5e-2)
    if hodge == "voronoi":
        assert s0.massless.size > 0, "Expected massless edges on graded Voronoi box"
    ratios = [2, 2]
    lvl = _nested_test_levels(s0, ratios)
    dt_lts, _ = lts_dt_max(
        s0, 3.2 * s0.dt_max, rtol=1e-3, edge_level=lvl, ratios=ratios
    )
    dt = 0.95 * dt_lts

    s = UnstructuredFIT3D(
        m,
        hodge=hodge,
        massless_tol=5e-2,
        dt=dt,
        lts={"ratios": ratios, "edge_level": lvl},
    )
    rng = np.random.default_rng(4)
    u_init = rng.standard_normal(s.free.size)
    s.u[:] = u_init

    n_steps = 2000
    energies = np.empty(n_steps)
    for k in range(n_steps):
        u_prev = s.u.copy()
        s.one_step()
        energies[k] = modified_energy(s, u_prev, s.u)
    drift = float(np.max(np.abs(energies - energies[0])) / np.abs(energies[0]))
    tol = 50.0 * EPS * np.sqrt(n_steps)
    assert drift < tol, f"Energy drift {drift:.3e} >= {tol:.3e}"

    mut_drifts = {}
    for mut in ("interface", "drop_buffer"):
        sm = UnstructuredFIT3D(
            m,
            hodge=hodge,
            massless_tol=5e-2,
            dt=0.5 * s0.dt_max,
            lts={"ratios": ratios, "edge_level": lvl, "mutate": mut, "_unsafe": True},
        )
        sm.u[:] = u_init
        e_mut = []
        for _ in range(250):
            up = sm.u.copy()
            sm.one_step()
            e_mut.append(modified_energy(sm, up, sm.u))
        e_mut = np.asarray(e_mut)
        d_mut = float(np.max(np.abs(e_mut - e_mut[0])) / np.abs(e_mut[0]))
        mut_drifts[mut] = d_mut
        assert d_mut > 1e-6, f"Mutation {mut} did not drift: {d_mut:.3e}"

    print(
        f"test_modified_energy_conserved[{hodge}]: massless={s0.massless.size}, "
        f"drift={drift:.3e} < tol={tol:.3e}, mut_interface={mut_drifts['interface']:.3e}, "
        f"mut_drop_buffer={mut_drifts['drop_buffer']:.3e}"
    )


# ------------------------------------------------------------------ 5. energy conserved while unstable
def test_energy_conserved_while_unstable():
    """
    At 1.05 * `lts_dt_max` (`_unsafe=True`), `||u||` grows by > 1e3x while
    `|E - E_0|` stays <= 1e-10 * (|E_kin| + |E_pot|), showing that the
    modified energy becomes indefinite rather than drifting when unstable.
    """
    m = _small_box()
    s0 = UnstructuredFIT3D(m, hodge="voronoi")
    ratios = [2]
    lvl = element_levels(s0, 1.9 * s0.dt_max, ratios, [0.05])
    dt_lts, _ = lts_dt_max(
        s0, 2.2 * s0.dt_max, rtol=1e-4, edge_level=lvl, ratios=ratios
    )

    s = UnstructuredFIT3D(
        m,
        hodge="voronoi",
        dt=1.05 * dt_lts,
        lts={"ratios": ratios, "edge_level": lvl, "_unsafe": True},
    )
    rng = np.random.default_rng(5)
    s.u[:] = 1e-3 * rng.standard_normal(s.free.size)
    u_norm0 = float(np.linalg.norm(s.u))

    E0 = None
    max_rel_err = 0.0
    growth = 1.0
    for step in range(1, 2000):
        u_prev = s.u.copy()
        s.one_step()
        E, e_kin, e_pot = modified_energy(s, u_prev, s.u, return_terms=True)
        if E0 is None:
            E0 = E
        denom = abs(e_kin) + abs(e_pot)
        rel_err = abs(E - E0) / max(denom, 1e-30)
        max_rel_err = max(max_rel_err, rel_err)
        growth = float(np.linalg.norm(s.u) / u_norm0)
        if growth > 1e4:
            break

    print(
        f"test_energy_conserved_while_unstable: steps={step}, growth={growth:.3e}, "
        f"max_rel_err={max_rel_err:.3e}"
    )
    assert growth > 1e3, f"Expected unstable growth > 1e3, got {growth:.3e}"
    assert max_rel_err <= 1e-10, (
        f"Modified energy drifted while unstable: {max_rel_err:.3e}"
    )


# ------------------------------------------------------------------ 6. stability enforced
def test_stability_enforced():
    """
    `UnstructuredFIT3D` rejects `dt = 1.0001 * dt_lts` with `ValueError`,
    accepts `0.999 * dt_lts` and stays finite for 5000 steps, and `lts_dt_max`
    rejects an unstable lower bracket `dt_lo_s = 1.02 * dt_lts`.
    Mutation: `mutate='nonest'` raises `RuntimeError` at construction.
    """
    m = _small_box()
    s0 = UnstructuredFIT3D(m, hodge="voronoi")
    ratios = [2]
    lvl = element_levels(s0, 1.8 * s0.dt_max, ratios, [0.05])
    dt_lts, _ = lts_dt_max(
        s0, 2.2 * s0.dt_max, rtol=5e-5, edge_level=lvl, ratios=ratios
    )

    with pytest.raises(ValueError, match="unstable|failed stability"):
        UnstructuredFIT3D(
            m,
            hodge="voronoi",
            dt=1.0001 * dt_lts,
            lts={"ratios": ratios, "edge_level": lvl},
        )

    with pytest.raises(ValueError, match="unstable"):
        lts_dt_max(
            s0,
            2.2 * s0.dt_max,
            dt_lo_s=1.02 * dt_lts,
            rtol=1e-3,
            edge_level=lvl,
            ratios=ratios,
        )

    s_ok = UnstructuredFIT3D(
        m,
        hodge="voronoi",
        dt=0.999 * dt_lts,
        lts={"ratios": ratios, "edge_level": lvl},
    )
    rng = np.random.default_rng(6)
    s_ok.u[:] = rng.standard_normal(s_ok.free.size)
    s_ok.run(5000)
    u_norm = float(np.linalg.norm(s_ok.u))
    assert np.all(np.isfinite(s_ok.u)) and np.all(np.isfinite(s_ok.b))

    mg = _graded_box(seed=3)
    sg = UnstructuredFIT3D(mg, hodge="voronoi")
    with pytest.raises(RuntimeError, match="nested"):
        UnstructuredFIT3D(
            mg,
            hodge="voronoi",
            dt=2.2 * sg.dt_max,
            lts={"ratios": [2, 2], "mutate": "nonest"},
        )

    print(
        f"test_stability_enforced: dt_lts/dt_global={dt_lts / s0.dt_max:.5f}, "
        f"norm_5000={u_norm:.3e}, nonest_raised=RuntimeError"
    )


# ------------------------------------------------------------------ 7. global limit relation
@pytest.mark.parametrize("ratios", [[2], [3], [2, 2]])
def test_global_limit_relation(ratios):
    """
    When the global top mode lies in `F_L` (M-energy fraction >= 0.99),
    `dt_lts / dt_global` matches `global_limit_ratio(ratios, nus)` within 2e-3.
    Mutation: equal damping on nested levels (`nu_l = nu`) fails `is_stable`
    at the predicted step.
    """
    m = _graded_box(seed=3)
    s0 = UnstructuredFIT3D(m, hodge="voronoi")
    nus = [0.05 * lv for lv in range(1, len(ratios) + 1)]
    pred = global_limit_ratio(ratios, nus)
    lvl = element_levels(s0, 1.05 * pred * s0.dt_max, ratios, nus)

    # Top global mode M-energy fraction on F_L
    L = len(ratios)
    F_L = np.flatnonzero(lvl >= L)
    lts_zero = TetLTS(s0, s0.dt_max, np.zeros(s0.free.size, dtype=int), [])
    _, top_energy = stability_ratio(lts_zero, tol=1e-10, return_mode=True)
    frac_FL = float(np.sum(top_energy[F_L]))
    assert frac_FL >= 0.99, (
        f"Precondition failed: top-mode M-energy on F_L is {frac_FL:.4f} < 0.99"
    )

    dt_lts, _ = lts_dt_max(
        s0,
        1.05 * pred * s0.dt_max,
        dt_lo_s=0.95 * pred * s0.dt_max,
        rtol=5e-4,
        edge_level=lvl,
        ratios=ratios,
        nu=nus,
    )
    meas = dt_lts / s0.dt_max
    rel_err = abs(meas / pred - 1.0)
    assert rel_err <= 2e-3, (
        f"ratios={ratios}: meas={meas:.5f}, pred={pred:.5f}, rel_err={rel_err:.3e}"
    )

    mut_msg = "n/a"
    if len(ratios) >= 2:
        lts_eq = TetLTS(s0, pred * s0.dt_max, lvl, ratios, nu=[0.05] * len(ratios))
        ok_eq, _rho_eq, marg_eq = is_stable(lts_eq, tol=1e-5)
        mut_msg = f"ok={ok_eq}, margins={marg_eq}"
        assert not ok_eq, (
            f"Equal damping on nested levels should be unstable: {mut_msg}"
        )

    print(
        f"test_global_limit_relation[{ratios}]: frac_FL={frac_FL:.5f}, "
        f"meas={meas:.5f}, pred={pred:.5f}, rel_err={rel_err:.3e}, equal_damping={mut_msg}"
    )


# ------------------------------------------------------------------ 8. box frequency
def _lts_mode_near(lts, lam_target, x_ref):
    """Find the generalized eigenvalue of (S_0, M) closest to `lam_target`."""
    S0, M = lts.stiffness()
    S0 = 0.5 * (S0 + S0.T)
    A = (S0 - lam_target * M).tocsc()
    lu = splu(A)
    n = lts.n
    OPinv = LinearOperator((n, n), matvec=lu.solve, dtype=float)
    v0 = np.random.default_rng(7).standard_normal(n)
    vals, vecs = eigsh(
        S0, k=4, M=M, sigma=lam_target, which="LM", OPinv=OPinv, v0=v0, tol=1e-12
    )
    overlaps = [abs(float(vecs[:, i] @ (M @ x_ref))) for i in range(vals.size)]
    idx = int(np.argmax(overlaps))
    return float(np.real(vals[idx]))


@pytest.mark.parametrize("hodge", ["voronoi", "whitney"])
def test_box_frequency(hodge):
    """
    PEC box 1 x 0.75 x 0.5 m TM110 frequency check (`L = 1, p = 2`, `theta_F >= 0.05`):
    (a) `f_td` (harminv, >= 40 periods) matches `f_pred` from `(S_0, M)` within 1e-5;
    (b) `lambda_0 / lambda - 1` matches first-order `r'(0) dt^2 lambda theta_F` within
        `10 * |r'(0)| dt^2 lambda * (lambda / gap) * |pred|`;
    (c) `|f_td / f_analytic - 1| <= |e_semi| + delta_t + |delta_L| + 1e-5`.
    Mutation: `mutate='consistency'` in the time-domain run breaks (a) (> 1e-4).
    """
    a, b = 1.0, 0.75
    f_analytic = 0.5 * c_0 * np.sqrt(1.0 / a**2 + 1.0 / b**2)
    m = _cavity_box(seed=1)
    s0 = UnstructuredFIT3D(m, hodge=hodge)

    f_modes, e_modes = s0.eigenmodes(n_modes=6, f_target=0.85 * f_analytic)
    idx = int(np.argmin(np.abs(f_modes - f_analytic)))
    f_semi = float(f_modes[idx])
    omega = 2.0 * np.pi * f_semi
    lam = omega**2
    x = e_modes[s0.free, idx]
    other_lams = np.delete((2.0 * np.pi * f_modes) ** 2, idx)
    gap = float(np.min(np.abs(other_lams - lam)))

    # Partition including stiff elements and central region so theta_F >= 0.05
    p, nu = 2, 0.05
    dt = 1.35 * s0.dt_max
    lvl = element_levels(s0, 1.55 * s0.dt_max, [p], [nu], indicator="element")
    mid = m.edge_midpoint[s0.free]
    in_center = (
        (mid[:, 0] > 0.3) & (mid[:, 0] < 0.7) & (mid[:, 1] > 0.2) & (mid[:, 1] < 0.55)
    )
    lvl[in_center] = 1
    lvl = nest_levels(effective_stiffness(s0), lvl)

    s_lts = UnstructuredFIT3D(
        m, hodge=hodge, dt=dt, lts={"ratios": [p], "nu": nu, "edge_level": lvl}
    )
    lts = s_lts.lts
    d1 = lts.levels[1]
    F1 = d1.F
    M, M_diag = mass_matrix(s0)
    Mx = (M_diag * x) if M_diag is not None else (M @ x)
    theta_F = float((Mx[F1] @ d1.solve_M(Mx[F1])) / (x @ Mx))
    assert theta_F >= 0.05, f"Expected theta_F >= 0.05, got {theta_F:.4f}"

    lam_0 = _lts_mode_near(lts, lam, x)
    f_pred = float(np.arcsin(0.5 * dt * np.sqrt(lam_0)) / (np.pi * dt))

    # (b) Perturbation check
    rp0 = r_prime_0(p, nu)
    pred_lam_shift = float(rp0 * (dt**2) * lam * theta_F)
    meas_lam_shift = float(lam_0 / lam - 1.0)
    assert abs(pred_lam_shift) >= 1e-6, (
        f"Precondition |pred| >= 1e-6 failed: {pred_lam_shift:.3e}"
    )
    diff_b = abs(meas_lam_shift - pred_lam_shift)
    bound_b = 10.0 * abs(rp0) * (dt**2) * lam * (lam / gap) * abs(pred_lam_shift)
    assert diff_b <= bound_b, f"(b) failed: diff={diff_b:.3e} > bound={bound_b:.3e}"

    # (a) Time-domain run (>= 42 periods) initialized with TM110 eigenvector
    probe_pt = [[0.48, 0.36, 0.24]]
    pr = s_lts.add_probe(probe_pt, component=2)
    s_lts.u[:] = x
    n_steps = int(np.ceil(45.0 / (f_pred * dt)))
    s_lts.run(n_steps)
    sig = s_lts.probe_signal(pr)[:, 0]
    res = harminv(
        sig, dt, fmin=0.8 * f_analytic, fmax=1.2 * f_analytic, method="pencil"
    )
    f_td = float(res.frequency[np.argmin(np.abs(res.frequency - f_pred))])
    err_a = abs(f_td / f_pred - 1.0)
    assert err_a < 1e-5, f"(a) f_td vs f_pred: {err_a:.3e} >= 1e-5"

    # (c) Analytic bound
    e_semi = float(f_semi / f_analytic - 1.0)
    delta_t = float(np.arcsin(0.5 * omega * dt) / (0.5 * omega * dt) - 1.0)
    delta_L = float(
        frequency_shift(omega, dt, theta_F, p, nu)
        - frequency_shift(omega, dt, 0.0, p, nu)
    )
    err_c = abs(f_td / f_analytic - 1.0)
    bound_c = abs(e_semi) + abs(delta_t) + abs(delta_L) + 1e-5
    assert err_c <= bound_c, f"(c) err_c={err_c:.3e} > bound_c={bound_c:.3e}"

    # Mutation: consistency in run only
    s_mut = UnstructuredFIT3D(
        m,
        hodge=hodge,
        dt=dt,
        lts={
            "ratios": [p],
            "nu": nu,
            "edge_level": lvl,
            "mutate": "consistency",
            "_unsafe": True,
        },
    )
    pr_m = s_mut.add_probe(probe_pt, component=2)
    s_mut.u[:] = x
    s_mut.run(n_steps)
    sig_m = s_mut.probe_signal(pr_m)[:, 0]
    res_m = harminv(
        sig_m, dt, fmin=0.7 * f_analytic, fmax=1.3 * f_analytic, method="pencil"
    )
    f_td_mut = float(res_m.frequency[np.argmin(np.abs(res_m.frequency - f_pred))])
    err_mut = abs(f_td_mut / f_pred - 1.0)
    assert err_mut > 1e-4, f"Mutation 'consistency' did not fail (a): {err_mut:.3e}"

    print(
        f"test_box_frequency[{hodge}]: theta_F={theta_F:.4f}, f_analytic={f_analytic:.4e}, "
        f"f_semi={f_semi:.4e}, f_pred={f_pred:.4e}, f_td={f_td:.4e}, "
        f"err_a={err_a:.3e}, meas_lam_shift={meas_lam_shift:.3e}, pred_lam_shift={pred_lam_shift:.3e}, "
        f"diff_b={diff_b:.3e} <= {bound_b:.3e}, e_semi={e_semi:.3e}, delta_t={delta_t:.3e}, "
        f"delta_L={delta_L:.3e}, err_c={err_c:.3e} <= {bound_c:.3e}, mut_err_a={err_mut:.3e}"
    )


# ------------------------------------------------------------------ 9. dot product & bitwise forward
@pytest.mark.parametrize("case", ["voronoi_pec", "whitney_pec", "voronoi_pml"])
def test_dot_product(case):
    """
    `blockwise_dot` <= 1e-14 and adjoint forward step is bitwise identical to
    `one_step` with LTS for Voronoi PEC (with massless > 0), Whitney PEC, and
    Voronoi spherical PML.
    Mutation: `mutate='transpose'` fails with dot error > 1e-6.
    """
    if case in ("voronoi_pec", "whitney_pec"):
        hodge = "voronoi" if case == "voronoi_pec" else "whitney"
        m = _graded_box(seed=2)
        s0 = UnstructuredFIT3D(m, hodge=hodge, massless_tol=5e-2)
        if hodge == "voronoi":
            assert s0.massless.size > 0
        ratios = [2, 2]
        lvl = _nested_test_levels(s0, ratios)
        dt = 0.85 * s0.dt_max
        s = UnstructuredFIT3D(
            m,
            hodge=hodge,
            massless_tol=5e-2,
            dt=dt,
            lts={"ratios": ratios, "edge_level": lvl},
        )
        s_mut = UnstructuredFIT3D(
            m,
            hodge=hodge,
            massless_tol=5e-2,
            dt=dt,
            lts={
                "ratios": ratios,
                "edge_level": lvl,
                "mutate": "transpose",
                "_unsafe": True,
            },
        )
    else:
        m, pml_cfg = ball_with_pml_mesh(2.5, 0.65, 3)
        s0 = UnstructuredFIT3D(m, pml=pml_cfg)
        rc = np.linalg.norm(m.nodes[m.cells].mean(axis=1), axis=1)
        l_T = (rc < 0.85).astype(int)
        edge_lvl = np.zeros(m.n_edges, dtype=int)
        np.maximum.at(edge_lvl, m.cell_edges, l_T[:, None])
        lvl = nest_levels(effective_stiffness(s0), edge_lvl[s0.free])
        dt = 0.9 * s0.dt_max
        s = UnstructuredFIT3D(
            m, pml=pml_cfg, dt=dt, lts={"ratios": [2], "edge_level": lvl}
        )
        s_mut = UnstructuredFIT3D(
            m,
            pml=pml_cfg,
            dt=dt,
            lts={
                "ratios": [2],
                "edge_level": lvl,
                "mutate": "transpose",
                "_unsafe": True,
            },
        )

    st = UnstructuredAdjointStepper(s)
    dot_err = blockwise_dot(st, seed=9)
    assert dot_err <= DOT_TOL, f"{case}: dot_err={dot_err:.3e} > {DOT_TOL:.3e}"

    # Bitwise forward equivalence between `one_step` and `st.step`
    np.random.default_rng(11)
    s.reset()
    s.add_point_dipole([0.35, 0.31, 0.22], [0.0, 0.0, 1.0], lambda t: np.sin(1e9 * t))
    st_f = UnstructuredAdjointStepper(s)
    x = st_f.zero_state()
    for n in range(10):
        s.one_step()
        x, _ = st_f.step(x, n)
        assert np.array_equal(x["u"], s.u)
        assert np.array_equal(x["b"], s.b)

    st_mut = UnstructuredAdjointStepper(s_mut)
    dot_mut = blockwise_dot(st_mut, seed=9)
    assert dot_mut > 1e-6, f"{case}: mutation 'transpose' did not fail: {dot_mut:.3e}"

    print(
        f"test_dot_product[{case}]: dot_err={dot_err:.3e} <= {DOT_TOL:.1e}, "
        f"bitwise_forward=True, mut_transpose={dot_mut:.3e}"
    )


# ------------------------------------------------------------------ 10. gradient vs FD
def _objective(out):
    X = out["m"]["dft"]
    return float(np.sum(np.abs(X) ** 2)), {"m": {"dft": 2 * X}}


def _fd_check(sim, rho, g, cells, h=1e-3):
    F0_ = _objective(sim.forward(rho, record=False))[0]
    out = []
    for k in cells:
        fd = []
        for hh in (h, 2 * h):
            rp, rm = rho.copy(), rho.copy()
            rp[k] += hh
            rm[k] -= hh
            Fp = _objective(sim.forward(rp, record=False))[0]
            Fm = _objective(sim.forward(rm, record=False))[0]
            fd.append((Fp - Fm) / (2 * hh))
        trunc = abs(fd[1] - fd[0]) / 3
        roundoff = 1e3 * np.finfo(float).eps * abs(F0_) / h
        out.append((g[k], fd[0], trunc, roundoff))
    return out


def test_gradient_vs_fd(monkeypatch):
    """
    Adjoint material gradient matches second-order central finite differences
    (`err <= 1.5 * trunc + roundoff` and `rel_err < 1e-5`) on the Voronoi box
    when the design region is disjoint from `F_1` (`|F_1| >= 10%` of `n_free`).
    A design region touching `F_1` raises `NotImplementedError`.
    Mutation: dropping `apply_T` in `_step3d_T` gives gradient error > 1e-3.
    """
    m = _small_box()
    s0 = UnstructuredFIT3D(m, hodge="voronoi")
    K = effective_stiffness(s0)

    # Put F_1 near x < 0.35 so |F_1| >= 10% of n_free and design cells near x > 0.55
    mid_e = m.edge_midpoint[s0.free]
    lvl = (mid_e[:, 0] < 0.32).astype(int)
    lvl = nest_levels(K, lvl)
    f1_frac = float(np.sum(lvl >= 1) / s0.free.size)
    assert f1_frac >= 0.10, f"Expected |F_1| >= 10% of n_free, got {f1_frac:.3f}"

    dt = 0.85 * s0.dt_max

    def make_solver():
        s = UnstructuredFIT3D(
            m, hodge="voronoi", dt=dt, lts={"ratios": [2], "edge_level": lvl}
        )
        s.add_point_dipole(
            [0.72, 0.35, 0.25], [0.0, 0.0, 1.0], lambda t: np.sin(1.5e9 * t)
        )
        return s

    s = make_solver()
    bed = np.zeros(m.n_edges, dtype=bool)
    bed[m.boundary_edges] = True
    f1_global = np.zeros(m.n_edges, dtype=bool)
    f1_global[s.free[s.lts.levels[1].F]] = True
    cell_touch_bd = np.any(bed[m.cell_edges], axis=1)
    cell_touch_f1 = np.any(f1_global[m.cell_edges], axis=1)
    min_partial = np.asarray(m.dual_area_partial.min(axis=0).todense()).ravel()
    cent = m.nodes[m.cells].mean(axis=1)
    design_cells = np.flatnonzero(
        ~cell_touch_bd & ~cell_touch_f1 & (min_partial >= 0.0) & (cent[:, 0] > 0.5)
    )
    assert design_cells.size >= 3

    # Verify design touching F_1 raises NotImplementedError
    bad_cells = np.flatnonzero(~cell_touch_bd & cell_touch_f1)
    assert bad_cells.size >= 1
    with pytest.raises(NotImplementedError, match="LTS"):
        UnstructuredAdjointStepper(s, design_cells=bad_cells[:2])

    design = CellMaterialGrid(m, design_cells, 1.0, 3.0)
    # Place monitor inside F_1 (x < 0.32) so the adjoint wave traverses R_0^T
    mon = CellDFTMonitor([200e6, 300e6], points=[[0.22, 0.38, 0.25]], component=2)
    sim = UnstructuredAdjointSimulation(make_solver, design, {"m": mon}, Nt=40)

    rng = np.random.default_rng(12)
    rho = rng.uniform(0.2, 0.8, design.shape)
    F_val, g = sim.value_and_grad(rho, _objective)
    assert F_val > 0 and np.all(np.isfinite(g)) and np.abs(g).max() > 0

    cells_idx = np.arange(min(3, rho.size))
    checks = _fd_check(sim, rho, g, cells_idx)
    max_rel = 0.0
    for adj, fd, trunc, roundoff in checks:
        err = abs(adj - fd)
        rel = err / abs(fd)
        max_rel = max(max_rel, rel)
        assert err <= 1.5 * trunc + roundoff, (adj, fd, trunc, roundoff)
        assert rel < 1e-5, f"rel={rel:.3e} >= 1e-5"

    # Mutation: drop apply_T in _step3d_T
    monkeypatch.setattr(sim.solver.lts, "apply_T", lambda z: z)
    _, g_mut = sim.value_and_grad(rho, _objective)
    mut_rels = [
        abs(g_mut[k] - fd) / abs(fd) for k, (_, fd, _, _) in zip(cells_idx, checks)
    ]
    max_mut_rel = max(mut_rels)
    assert max_mut_rel > 1e-3, (
        f"Dropping apply_T did not break gradient: {max_mut_rel:.3e}"
    )

    print(
        f"test_gradient_vs_fd: f1_frac={f1_frac:.3f}, max_rel={max_rel:.3e}, "
        f"mut_no_apply_T={max_mut_rel:.3e}"
    )


# ------------------------------------------------------------------ 11. PML with LTS
def _pml_probe_matrix(mesh, points):
    """Whitney interpolation of E (rows 3 i + c) from all edge voltages (as in test_098)."""
    cell, W = mesh.whitney_edge(points)
    rows = np.repeat(3 * np.arange(len(cell)), 6)
    cols = mesh.cell_edges[cell].ravel()
    shape = (3 * len(cell), mesh.n_edges)
    return sum(
        sp.csr_matrix((W[:, :, c].ravel(), (rows + c, cols)), shape=shape)
        for c in range(3)
    ).tocsr()


def _pml_lts_signals(mesh, cfg, dt, rb=4.0, h=1.0):
    """Run normal-incidence spherical PML helper of test_098 with interior LTS (`rc < 1.5` m)."""
    lam0 = 12.0 * h
    tau = lam0 / (np.pi * c_0)
    t0 = 4.0 * tau

    def amp(t):
        x = (t - t0) / tau
        return -x * np.exp(-(x**2))

    th = np.radians([90.0, 45.0])
    rp = rb - 1.0
    src = np.array([0.31, 0.17, 0.23])
    probes = rp * np.stack([np.sin(th), 0.0 * th, np.cos(th)], axis=1)
    r1 = rb + 10.0 * h
    t_win = 2.0 * t0 + (2.0 * r1 - (rb - 1.0)) / c_0

    s0 = UnstructuredFIT3D(mesh, bc="pec", dt=dt, pml=cfg)
    rc = np.linalg.norm(mesh.nodes[mesh.cells].mean(axis=1), axis=1)
    l_T = (rc < 1.5).astype(int)
    edge_lvl = np.zeros(mesh.n_edges, dtype=int)
    np.maximum.at(edge_lvl, mesh.cell_edges, l_T[:, None])
    lvl = nest_levels(effective_stiffness(s0), edge_lvl[s0.free])

    s = UnstructuredFIT3D(
        mesh, bc="pec", dt=dt, pml=cfg, lts={"ratios": [2], "edge_level": lvl}
    )
    s.add_point_dipole(src, [0.0, 0.0, 1.0], amp)
    P = _pml_probe_matrix(s.mesh, probes)
    volt = s._pml.edge_voltages
    out = []
    for _ in range(round(t_win / s.dt)):
        s.one_step()
        out.append((P @ volt()).reshape(len(probes), 3))
    return np.array(out)


def test_pml_with_lts():
    """
    Spherical CFS-PML with interior LTS (`rc < 1.5` m) achieves reflection
    `< TARGET = 1e-2` (of `test_098`) against the `n_extra` reference; a
    partition touching the PML layer raises `ValueError`.
    Mutation: PML `_mutation='drop_aux'` increases reflection by > 10x.
    """
    rb, h, n_layers, n_ref, n_extra = 4.0, 1.0, 10, 6, 24
    target = 1e-2
    m_pml, pml = ball_with_pml_mesh(rb, h, n_layers)
    m_ref, pml_ref = ball_with_pml_mesh(rb, h, n_ref, n_extra=n_extra)

    s0 = UnstructuredFIT3D(m_pml, bc="pec", pml=pml)
    s0_ref = UnstructuredFIT3D(m_ref, bc="pec", pml=pml_ref)
    dt = 0.9 * min(s0.dt_max, s0_ref.dt_max)

    sig_ref = _pml_lts_signals(m_ref, pml_ref, dt, rb=rb, h=h)
    sig_ok = _pml_lts_signals(m_pml, pml, dt, rb=rb, h=h)
    ref_norm = np.linalg.norm(sig_ref, axis=-1).max(axis=0)
    r_ok_vec = np.linalg.norm(sig_ok - sig_ref, axis=-1).max(axis=0) / ref_norm
    r_ok = float(np.max(r_ok_vec))
    assert r_ok < target, f"PML+LTS reflection {r_ok:.3e} >= {target:.3e}"

    # Partition touching the PML layer must raise ValueError
    bad_lvl = np.zeros(s0.free.size, dtype=int)
    bad_lvl[s0._pml.ue[:3]] = 1
    bad_lvl = nest_levels(effective_stiffness(s0), bad_lvl)
    with pytest.raises(ValueError, match="PML"):
        UnstructuredFIT3D(
            m_pml, bc="pec", pml=pml, dt=dt, lts={"ratios": [2], "edge_level": bad_lvl}
        )

    # Mutation: _mutation="drop_aux"
    pml_mut = dict(pml, _mutation="drop_aux")
    sig_mut = _pml_lts_signals(m_pml, pml_mut, dt, rb=rb, h=h)
    r_mut_vec = np.linalg.norm(sig_mut - sig_ref, axis=-1).max(axis=0) / ref_norm
    r_mut = float(np.max(r_mut_vec))
    assert r_mut > 10.0 * r_ok, (
        f"PML drop_aux mutation did not fail: r_mut={r_mut:.3e} vs r_ok={r_ok:.3e}"
    )

    print(
        f"test_pml_with_lts: r_ok={r_ok:.3e} < target={target:.3e}, "
        f"r_mut={r_mut:.3e} ({r_mut / r_ok:.1f}x)"
    )


# ------------------------------------------------------------------ 12. PML late-time (slow)
@pytest.mark.slow
def test_pml_lts_late_time():
    """
    20 000 coarse steps after the source with CFS-PML and interior LTS:
    interior energy satisfies `E2 <= E1 * (1 + 1e-3)`.
    """
    m, pml = ball_with_pml_mesh(2.5, 0.65, 4)
    pml = dict(pml, alpha=2.0 * np.pi * c_0 / 12.0 / 10.0)
    s0 = UnstructuredFIT3D(m, pml=pml)
    rc = np.linalg.norm(m.nodes[m.cells].mean(axis=1), axis=1)
    l_T = (rc < 0.85).astype(int)
    edge_lvl = np.zeros(m.n_edges, dtype=int)
    np.maximum.at(edge_lvl, m.cell_edges, l_T[:, None])
    lvl = nest_levels(effective_stiffness(s0), edge_lvl[s0.free])

    dt = 0.92 * s0.dt_max
    s = UnstructuredFIT3D(m, pml=pml, dt=dt, lts={"ratios": [2], "edge_level": lvl})
    lam0 = 12.0 * 0.65
    tau = lam0 / (np.pi * c_0)
    t0 = 4.0 * tau
    s.add_point_dipole(
        [0.15, -0.1, 0.2],
        [0.0, 0.0, 1.0],
        lambda t: -((t - t0) / tau) * np.exp(-(((t - t0) / tau) ** 2)),
    )

    n_pulse = int((t0 + 6.0 * tau) / dt)
    s.run(n_pulse)
    E1 = float(s._pml.interior_energy())
    s.run(10000)
    E_mid = float(s._pml.interior_energy())
    s.run(10000)
    E2 = float(s._pml.interior_energy())
    print(
        f"test_pml_lts_late_time: E_pulse={E1:.3e}, E_10k={E_mid:.3e}, E_20k={E2:.3e}, "
        f"E_20k/E_10k={E2 / max(E_mid, 1e-30):.6f}, E_20k/E_pulse={E2 / max(E1, 1e-30):.3e}"
    )
    assert np.all(np.isfinite(s.u)) and np.all(np.isfinite(s.b))
    assert E2 <= E1 * (1.0 + 1e-3), f"Late-time energy grew: E2={E2:.3e} > E1={E1:.3e}"
    assert E2 <= E_mid * (1.0 + 1e-3), (
        f"Late-time energy grew from 10k to 20k: E2={E2:.3e} > E_mid={E_mid:.3e}"
    )


# ------------------------------------------------------------------ 13. partition rules
@pytest.mark.parametrize("hodge", ["voronoi", "whitney"])
def test_partition_rules(hodge):
    """
    Verify level assignment and nesting rules:
    (i) every `C_l subset F_{l-1}`;
    (ii) every free edge of an element with `tau_{l-1} > theta * dt_T` is at level >= l;
    (iii) coarse block satisfies `lambda_max(K_cc, M_cc) <= (2 / dt)^2 * (1 + 1e-6)`
          for the element indicator.
    Mutation: `mutate='nonest'` violates `C_2 subset F_1`.
    """
    m = _graded_box(seed=3)
    s = UnstructuredFIT3D(m, hodge=hodge)
    ratios = [2, 2]
    nus = [0.05, 0.10]
    dt_s = 2.2 * s.dt_max
    stab = element_stability(s)
    lvl = element_levels(s, dt_s, ratios, nus, indicator="element", stab=stab)
    lts = TetLTS(s, dt_s, lvl, ratios, nu=nus)

    # (i) C_l subset F_{l-1}
    for lv in range(1, lts.L + 1):
        C_l = lts.levels[lv].C
        assert np.all(lvl[C_l] >= lv - 1), f"Level {lv} buffer not in F_{lv - 1}"

    # (ii) element level dominance
    theta = theta_factor(ratios, nus)
    dt_T = stab["dt_T"]
    pos = np.full(m.n_edges, -1, dtype=int)
    pos[s.free] = np.arange(s.free.size)
    for lv in range(1, lts.L + 1):
        flagged_cells = np.flatnonzero(lts.tau_s[lv - 1] > theta * dt_T)
        if flagged_cells.size:
            fe = pos[m.cell_edges[flagged_cells].ravel()]
            fe = fe[fe >= 0]
            assert np.all(lvl[fe] >= lv)

    # (iii) coarse block stability bound
    coarse = np.flatnonzero(lvl == 0)
    assert coarse.size > 0
    K_cc = lts.K[coarse, :][:, coarse].toarray()
    M_cc = lts.M[coarse, :][:, coarse].toarray()
    lam_cc = float(la.eigvalsh(0.5 * (K_cc + K_cc.T), M_cc)[-1])
    bound_cc = (2.0 / dt_s) ** 2
    ratio_cc = lam_cc / bound_cc
    assert ratio_cc <= 1.0 + 1e-6, f"Coarse block unstable: ratio={ratio_cc:.6f}"

    # Mutation: nonest violates C_2 subset F_1
    lvl_mut = element_levels(
        s, dt_s, ratios, nus, indicator="element", stab=stab, mutate="nonest"
    )
    F2_mut = np.flatnonzero(lvl_mut >= 2)
    cols2 = np.unique(lts.K[F2_mut, :].indices)
    n_viol = int(np.sum(lvl_mut[cols2] < 1))
    assert n_viol > 0, "Expected nonest mutation to leave C_2 edges outside F_1"
    with pytest.raises(RuntimeError, match="nested"):
        TetLTS(s, dt_s, lvl_mut, ratios, nu=nus)

    print(
        f"test_partition_rules[{hodge}]: ratio_cc={ratio_cc:.6f} <= 1, "
        f"nonest_violations={n_viol}"
    )


# ------------------------------------------------------------------ 14. guards
def test_guards():
    """
    Unsupported features raise `NotImplementedError` when 3D LTS is enabled:
    GPU `DeviceStepper`, MPI `LocalCluster` and `DistributedFIT`, lumped ports,
    dispersive materials, conductivity `sigma > 0`, and `bc='abc'`.
    """
    m = _small_box()
    s_lts = UnstructuredFIT3D(m, lts=2)

    with pytest.raises(NotImplementedError, match="LTS"):
        DeviceStepper(s_lts, xp=np)

    with pytest.raises(NotImplementedError, match="LTS"):
        LocalCluster(s_lts, 2)

    with pytest.raises(NotImplementedError, match="LTS"):
        DistributedFIT(s_lts, None)

    e_free = int(s_lts.free[0])
    port = LumpedPort(s_lts, [e_free], [1.0], R=50.0)
    with pytest.raises(NotImplementedError, match="ports"):
        attach_ports(s_lts, [port])

    with pytest.raises(NotImplementedError, match="dispersive"):
        UnstructuredFIT3D(
            m,
            materials={0: DispersiveMaterial(1.0, [Lorentz(1.0, 2e9, 1e8)])},
            lts=2,
        )

    with pytest.raises(NotImplementedError, match="conductivity|sigma"):
        UnstructuredFIT3D(m, materials={0: [1.0, 1.0, 0.05]}, lts=2)

    with pytest.raises(NotImplementedError, match="abc"):
        UnstructuredFIT3D(m, bc="abc", lts=2)

    print("test_guards: all 7 unsupported configurations raised NotImplementedError")


# ------------------------------------------------------------------ 15. work accounting
@pytest.mark.parametrize("hodge", ["voronoi", "whitney"])
def test_work_accounting(hodge):
    """
    `work_per_step()` matches an independent count of nonzeros from the stored
    sparse matrices on `solver` and `TetLTS`.
    """
    m = _graded_box(seed=2)
    s = UnstructuredFIT3D(m, hodge=hodge, massless_tol=5e-2)
    ratios = [2, 3]
    lvl = _nested_test_levels(s, ratios)
    lts = TetLTS(s, 0.9 * s.dt_max, lvl, ratios)
    w = lts.work_per_step()

    if hodge == "voronoi" and s.massless.size > 0:
        _, Sop, _, _ = massless_schur_operator(s)
        c_eff = (s.Cf - s.Cs @ Sop).tocsr()
        c_eff.eliminate_zeros()
        ceff_nnz = int(c_eff.nnz)
    else:
        ceff_nnz = int(s.Cf.nnz)

    full_ind = (
        ceff_nnz
        + int(s.Cf.T.nnz)
        + int(s.free.size)
        + (int(s.M1.nnz) if hodge == "whitney" else 0)
    )
    d1, d2 = lts.levels[1], lts.levels[2]
    row1 = int(
        d1.K_FF.nnz
        + d1.K_FC.nnz
        + d1.F.size
        + (d1.M_ll.nnz if hodge == "whitney" else 0)
    )
    row2 = int(
        d2.K_FF.nnz
        + d2.K_FC.nnz
        + d2.F.size
        + (d2.M_ll.nnz if hodge == "whitney" else 0)
    )
    w1_ind = (ratios[0] - 1) * row1
    w2_ind = ratios[0] * (ratios[1] - 1) * row2

    assert w["full"] == full_ind
    assert w["per_level"] == [w1_ind, w2_ind]
    assert w["total"] == full_ind + w1_ind + w2_ind

    print(
        f"test_work_accounting[{hodge}]: full={w['full']}, "
        f"per_level={w['per_level']}, total={w['total']}"
    )
