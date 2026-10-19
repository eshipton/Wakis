# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Tests for quadtree + conforming triangle-band ("subtri") meshes and solvers."""

import numpy as np
import pytest
from scipy.constants import c as C0
from scipy.spatial import cKDTree

from wakis.sources import GaussianSource
from wakis.unstructured.lts import modified_energy
from wakis.unstructured.polymesh import PolyMesh
from wakis.unstructured.polymeshers import hybrid_circles_mesh
from wakis.unstructured.polysolver2d import PolygonFIT2D
from wakis.unstructured.subtri import (
    SubTriFIT2D,
    linear_step,
    linear_step_T,
    subtri_circles_mesh,
)


def _leapfrog_energy(solver, u, v_old, v_new):
    """Exact two-time-level leapfrog invariant 1/2 u^T D u + 1/2 v_old^T S v_new."""
    return float(0.5 * (u @ (solver.mass * u)) + 0.5 * (v_old @ (solver.stiff * v_new)))


def _step_loop(solver, n_steps):
    """Run ``n_steps`` of loss-free leapfrog; return ``(u_prev, u, v_old, v_new)``."""
    u = solver.u.copy()
    v = solver.v.copy()
    dt = solver.dt
    inv_m = dt / solver.mass
    stiff = solver.stiff
    sign = -1.0 if solver.pol == "TM" else 1.0
    u_prev = u.copy()
    v_old = v.copy()
    for _ in range(n_steps):
        u_prev = u
        v_old = v
        grad_u = solver._grad(u)
        v = v + (sign * dt) * grad_u
        u = u - (sign * inv_m) * (solver.GT @ (stiff * v))
    solver.u[:] = u
    solver.v[:] = v
    return u_prev, u, v_old, v


def test_mesh_conforming():
    """Dimer mesh with L=2 satisfies tiling, positivity, and linear exactness."""
    h = 0.05
    r = 0.25
    gap = 3.0 * h
    offset = r + 0.5 * gap
    circles = [(-offset, 0.0, r), (offset, 0.0, r)]
    mesh, P, info = subtri_circles_mesh(circles, h=h, levels=2, half_width=1.2)

    # Domain tiling check
    expected_area = (2.0 * info["half_width_pml"]) ** 2
    assert abs(mesh.cell_area.sum() - expected_area) / expected_area < 1e-9

    # No hanging node on any triangle
    col_nnz = np.diff(P.tocsc().indptr)
    hanging = set(np.flatnonzero(col_nnz == 0).tolist())
    assert len(hanging) == info["n_hanging"] > 0
    tri_mask = mesh.cell_size == 3
    assert int(tri_mask.sum()) == info["n_triangles"] > 0
    for ci in np.flatnonzero(tri_mask):
        assert hanging.isdisjoint(mesh.cell(ci))

    # Every triangle circumcentre-dual length > 0
    dual_len = np.asarray(mesh.dual_edge_partial().sum(axis=1)).ravel()
    tri_corners = tri_mask[mesh.corner_cell]
    tri_edges = np.unique(mesh.corner_edge[tri_corners])
    assert (dual_len[tri_edges] > 0.0).all()
    assert (dual_len > 0.0).all()

    # Every row of P sums to 1
    row_sums = np.asarray(P.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    # G @ P @ x_linear reproduces exact edge differences of a linear function
    # on every edge (including edges incident to hanging nodes).
    x_lin = 1.7 * mesh.nodes[:, 0] - 2.3 * mesh.nodes[:, 1] + 0.4
    exact_diff = x_lin[mesh.edges[:, 1]] - x_lin[mesh.edges[:, 0]]
    gp_diff = (mesh.G @ (P @ x_lin)).ravel()
    rel_err = np.linalg.norm(gp_diff - exact_diff) / np.linalg.norm(exact_diff)
    assert rel_err < 1e-12

    # Mutation check: perturbing a hanging-node weight in P breaks linear exactness.
    P_mut = P.tolil(copy=True)
    h_node = next(iter(hanging))
    P_mut.rows[h_node] = P_mut.rows[h_node][:1]
    P_mut.data[h_node] = [1.0]
    mut_diff = (mesh.G @ (P_mut.tocsr() @ x_lin)).ravel()
    mut_rel_err = np.linalg.norm(mut_diff - exact_diff) / np.linalg.norm(exact_diff)
    assert mut_rel_err > 1e-3


def test_energy_long():
    """Long 1e5-step leapfrog preserves exact discrete energy in TM and TE."""
    h = 0.055
    circles = [(0.0, 0.0, 0.25)]
    mesh, P, _ = subtri_circles_mesh(circles, h=h, levels=2, half_width=0.88)
    materials = {0: (4.0, 1.0), 1: (1.0, 1.0)}
    n_steps = 100_000

    for pol in ("TM", "TE"):
        solver = SubTriFIT2D(mesh, P, materials=materials, polarization=pol, cfl=0.9)
        assert 1500 <= solver.n_unknowns <= 5000
        rng = np.random.default_rng(42 if pol == "TM" else 43)
        solver.u[:] = rng.standard_normal(solver.free.size)

        # Take one step to initialize v at t = dt/2 and record E0
        u0 = solver.u.copy()
        v0 = solver.v.copy()
        u_prev1, _, v_old, v_new = _step_loop(solver, 1)
        e0 = _leapfrog_energy(solver, u_prev1, v_old, v_new)
        assert e0 > 0.0

        u_prev_end, _, v_old_end, v_new_end = _step_loop(solver, n_steps - 1)
        e_end = _leapfrog_energy(solver, u_prev_end, v_old_end, v_new_end)
        rel_drift = abs(e_end - e0) / e0
        assert rel_drift < 1e-10

        # Mutation check: drop prolongation P in transpose operator GT only
        solver_mut = SubTriFIT2D(
            mesh, P, materials=materials, polarization=pol, cfl=0.9
        )
        solver_mut.u[:] = u0
        solver_mut.v[:] = v0
        solver_mut.GT = mesh.G[:, solver_mut.free].T.tocsr()

        u_prev_m0, _, v_old_m0, v_new_m0 = _step_loop(solver_mut, 1)
        e0_mut = _leapfrog_energy(solver_mut, u_prev_m0, v_old_m0, v_new_m0)
        # Run in chunks and check for blow-up or large energy drift
        blown = False
        for _ in range(10):
            u_prev_m, u_m, v_old_m, v_new_m = _step_loop(solver_mut, n_steps // 10)
            if not np.all(np.isfinite(u_m)) or np.max(np.abs(u_m)) > 1e6:
                blown = True
                break
        if not blown:
            e_mut = _leapfrog_energy(solver_mut, u_prev_m, v_old_m, v_new_m)
            drift_mut = abs(e_mut - e0_mut) / abs(e_mut)
            assert drift_mut > 1e-6


def test_lts_energy():
    """Two-level LTS on a narrow-gap dimer mesh preserves bounded modified energy."""
    h = 0.08
    r = 0.22
    gap = 0.04
    offset = r + 0.5 * gap
    circles = [(-offset, 0.0, r), (offset, 0.0, r)]
    mesh, P, _ = subtri_circles_mesh(
        circles,
        h=h,
        levels=2,
        half_width=0.96,
        gap_cells=6,
        h_min=gap / 6.0,
    )
    materials = {0: (4.0, 1.0), 1: (4.0, 1.0), 2: (1.0, 1.0)}

    solver_global = SubTriFIT2D(
        mesh, P, materials=materials, polarization="TM", lts=None, cfl=0.9
    )
    solver = SubTriFIT2D(
        mesh, P, materials=materials, polarization="TM", lts=2, cfl=0.9
    )
    assert solver.dt / solver_global.dt > 1.5
    assert solver.lts_work_per_step() > solver.n_unknowns

    rng = np.random.default_rng(123)
    u_init = rng.standard_normal(solver.free.size)
    solver.u[:] = u_init

    u_prev1, u1, _, _ = _step_loop(solver, 1)
    e0 = modified_energy(solver, u_prev1, u1)
    assert e0 > 0.0

    u_prev_end, u_end, _, _ = _step_loop(solver, 20_000 - 1)
    assert np.all(np.isfinite(u_end))
    e_end = modified_energy(solver, u_prev_end, u_end)
    rel_drift = abs(e_end - e0) / e0
    assert rel_drift < 1e-9

    # Mutation check: disabling the LTS correction filter destabilizes the coarse step
    solver_mut = SubTriFIT2D(
        mesh, P, materials=materials, polarization="TM", lts=2, cfl=0.9
    )
    solver_mut.u[:] = u_init
    solver_mut.lts = None
    u_prev_m, u_mut, v_old_m, v_new_m = _step_loop(solver_mut, 500)
    if np.all(np.isfinite(u_mut)) and np.max(np.abs(u_mut)) < 1e6:
        e_mut = _leapfrog_energy(solver_mut, u_prev_m, v_old_m, v_new_m)
        assert abs(e_mut - e0) / e0 > 1.0
    else:
        assert not np.all(np.isfinite(u_mut)) or np.max(np.abs(u_mut)) >= 1e6


def test_cavity_convergence():
    """TM PEC cavity with a virtual circle converges at 2nd order; injection degrades it.

    Honest convergence report on [-a, a]^2 with R = 0.45*a, L = 1, band = 2, grade = 2:
      - h = a/6:  max rel error = 1.207e-2 (no coarse cells yet since R + 6h = 1.45a > a)
      - h = a/12: max rel error = 5.361e-3 (order a/6  -> a/12 = 1.171; 13.9% coarse cells)
      - h = a/24: max rel error = 1.696e-3 (order a/12 -> a/24 = 1.661; 55.6% coarse cells,
        so 41.7% of the domain transitions from fine size a/12 to coarse size 2*(a/24)=a/12)
      - h = a/48: max rel error = 4.544e-4 (order a/24 -> a/48 = 1.900; 76.2% coarse cells)
    Mutant ((1, 0) injection on hanging nodes):
      - h = a/6:  max rel error = 1.207e-2
      - h = a/12: max rel error = 1.334e-2 (order a/6  -> a/12 = -0.144)
      - h = a/24: max rel error = 8.339e-3 (order a/12 -> a/24 =  0.678 < 1.5, 4.92x > 2x)
      - h = a/48: max rel error = 3.920e-3 (order a/24 -> a/48 =  1.089 < 1.5, 8.63x > 2x)
    """
    a = 1.0
    # Lowest 4 non-zero TM PEC modes on [-a, a]^2 (side 2a): (1,1), (1,2), (2,1), (2,2)
    f_exact = (
        C0
        / (4.0 * a)
        * np.array([np.hypot(1, 1), np.hypot(1, 2), np.hypot(1, 2), np.hypot(2, 2)])
    )
    hs = [a / 6.0, a / 12.0, a / 24.0, a / 48.0]

    errs = []
    errs_mut = []
    for h in hs:
        mesh, P, _ = subtri_circles_mesh(
            [(0.0, 0.0, 0.45 * a)], h=h, levels=1, half_width=a
        )
        materials = {0: (1.0, 1.0), 1: (1.0, 1.0)}
        solver = SubTriFIT2D(mesh, P, materials=materials, polarization="TM")
        freqs, _ = solver.eigenmodes(n_modes=6)
        err = float(np.max(np.abs(freqs[:4] - f_exact) / f_exact))
        errs.append(err)

        # Mutant: replace hanging-node (0.5, 0.5) interpolation by (1.0, 0.0) injection
        P_mut = P.tolil(copy=True)
        col_nnz = np.diff(P.tocsc().indptr)
        for h_node in np.flatnonzero(col_nnz == 0):
            P_mut.rows[h_node] = P_mut.rows[h_node][:1]
            P_mut.data[h_node] = [1.0]
        solver_mut = SubTriFIT2D(
            mesh, P_mut.tocsr(), materials=materials, polarization="TM"
        )
        freqs_mut, _ = solver_mut.eigenmodes(n_modes=6)
        err_mut = float(np.max(np.abs(freqs_mut[:4] - f_exact) / f_exact))
        errs_mut.append(err_mut)

    order_12_24 = np.log2(errs[1] / errs[2])
    order_24_48 = np.log2(errs[2] / errs[3])
    mut_order_12_24 = np.log2(errs_mut[1] / errs_mut[2])
    mut_order_24_48 = np.log2(errs_mut[2] / errs_mut[3])

    # Pre-asymptotic (a/12 -> a/24) order is 1.661; asymptotic (a/24 -> a/48) order is 1.900 >= 1.7
    assert order_12_24 > 1.6
    assert order_24_48 >= 1.7

    # Mutant order drops below 1.5 AND finest error grows by > 2x on both pairs
    assert mut_order_12_24 < 1.5 and errs_mut[2] > 2.0 * errs[2]
    assert mut_order_24_48 < 1.5 and errs_mut[3] > 2.0 * errs[3]


def test_default_unchanged():
    """PolygonFIT2D with default kwargs is bit-identical to explicit None hooks."""
    mesh, _ = hybrid_circles_mesh(
        [(0.0, 0.0, 0.25)], h_in=0.05, h_out=0.08, half_width=1.0
    )
    materials = {0: (3.0, 1.0), 1: (1.0, 1.0)}

    s_default = PolygonFIT2D(mesh, materials=materials, polarization="TM")
    s_probe = PolygonFIT2D(
        mesh, materials=materials, polarization="TM", prolongation=None, lts=None
    )
    # eigsh in max_eigenvalue uses unseeded Fortran ARPACK (tol=1e-4); passing the
    # same dt makes the time-stepping trajectory bit-identical.
    assert np.isclose(s_default.dt_max, s_probe.dt_max, rtol=1e-4)
    s_explicit = PolygonFIT2D(
        mesh,
        materials=materials,
        polarization="TM",
        dt=s_default.dt,
        prolongation=None,
        lts=None,
    )
    assert np.array_equal(s_default.mass, s_explicit.mass)
    assert np.array_equal(s_default.stiff, s_explicit.stiff)
    assert np.array_equal(s_default.K.data, s_explicit.K.data)

    rng = np.random.default_rng(7)
    u0 = rng.standard_normal(s_default.free.size)
    s_default.u[:] = u0
    s_explicit.u[:] = u0

    for _ in range(50):
        s_default.one_step()
        s_explicit.one_step()

    assert np.array_equal(s_default.u, s_explicit.u)
    assert np.array_equal(s_default.v, s_explicit.v)
    assert s_default.lts_work_per_step() == s_default.n_unknowns

    # Mutation check: perturbing one entry of u0 produces a measurably different final state
    s_mut = PolygonFIT2D(mesh, materials=materials, polarization="TM", dt=s_default.dt)
    u0_mut = u0.copy()
    u0_mut[0] += 1e-6
    s_mut.u[:] = u0_mut
    for _ in range(50):
        s_mut.one_step()
    assert not np.array_equal(s_default.u, s_mut.u)
    assert np.max(np.abs(s_default.u - s_mut.u)) > 1e-8


def test_dot_product():
    """Adjoint dot-product test for linear_step / linear_step_T (TM and TE, with/without LTS)."""
    h = 0.08
    r = 0.22
    gap = 0.04
    offset = r + 0.5 * gap
    circles = [(-offset, 0.0, r), (offset, 0.0, r)]
    mesh, P, _ = subtri_circles_mesh(
        circles,
        h=h,
        levels=2,
        half_width=0.96,
        gap_cells=6,
        h_min=gap / 6.0,
    )
    materials = {0: (4.0, 1.0), 1: (4.0, 1.0), 2: (1.0, 1.0)}

    # Reject PML solvers
    s_pml = SubTriFIT2D(
        mesh,
        P,
        materials=materials,
        polarization="TM",
        pml=dict(x=(-0.96, 0.96), y=(-0.96, 0.96), thickness=0.20),
    )
    with pytest.raises(ValueError, match="lossless"):
        linear_step(s_pml, s_pml.u, s_pml.v)
    with pytest.raises(ValueError, match="lossless"):
        linear_step_T(s_pml, s_pml.u, s_pml.v)

    for pol in ("TM", "TE"):
        for lts in (None, 2):
            s = SubTriFIT2D(
                mesh, P, materials=materials, polarization=pol, lts=lts, cfl=0.9
            )
            rng = np.random.default_rng(
                100 + (0 if pol == "TM" else 10) + (0 if lts is None else 2)
            )
            n, m = s.free.size, mesh.n_edges
            # Balanced physical scales: dt * G u ~ v and dt^2 * S / M ~ 1
            u, ub = rng.standard_normal((2, n))
            v = s.dt * rng.standard_normal(m)
            vb = rng.standard_normal(m) / s.dt

            # 1. Exact agreement between solver.one_step() and linear_step(s, u, v)
            s.u[:] = u
            s.v[:] = v
            s.one_step()
            u1, v1 = linear_step(s, u, v)
            rel_u = np.linalg.norm(s.u - u1) / np.linalg.norm(u1)
            rel_v = np.linalg.norm(s.v - v1) / np.linalg.norm(v1)
            assert max(rel_u, rel_v) < 1e-14

            # 2. Adjoint dot-product identity: |<A x, y> - <x, A^T y>| / (|A x||y|) < 1e-13
            ua, va = linear_step_T(s, ub, vb)
            lhs = float(ub @ u1 + vb @ v1)
            rhs = float(ua @ u + va @ v)
            scale = float(
                np.linalg.norm(u1) * np.linalg.norm(ub)
                + np.linalg.norm(v1) * np.linalg.norm(vb)
            )
            assert abs(lhs - rhs) / scale < 1e-13

            # Mutation 1: transpose uses mesh.G[:, free] instead of G P
            G_true, GT_true = s.G, s.GT
            G_bad = mesh.G[:, s.free].tocsr()
            s.G, s.GT = G_bad, G_bad.T.tocsr()
            ua_m1, va_m1 = linear_step_T(s, ub, vb)
            s.G, s.GT = G_true, GT_true
            rhs_m1 = float(ua_m1 @ u + va_m1 @ v)
            assert abs(lhs - rhs_m1) / scale > 1e-6

            # Mutation 2 (for LTS): drop apply_T in transpose
            if lts is not None:
                apply_T_orig = s.lts.apply_T
                s.lts.apply_T = lambda z: z
                ua_m2, va_m2 = linear_step_T(s, ub, vb)
                s.lts.apply_T = apply_T_orig
                rhs_m2 = float(ua_m2 @ u + va_m2 @ v)
                assert abs(lhs - rhs_m2) / scale > 1e-6


def test_one_step_energy():
    """Real solver.one_step() conserves leapfrog energy over 2e4 steps in TM and TE.

    Note on hanging-node prolongation scaling: scaling one hanging node's row in P by
    1.1 breaks partition of unity (rows no longer sum to 1), but since the solver
    assembles K_grad = G P, GT = K_grad^T, and D = P^T m consistently from P, the
    spatial operator D^{-1} K_grad^T S K_grad remains self-adjoint in the D inner
    product and the leapfrog invariant is still conserved to round-off (< 1e-10).
    Perturbing a single entry of solver.GT by 1% breaks K_grad^T == GT symmetry and
    causes energy drift > 1e-6.
    """
    h = 0.055
    circles = [(0.0, 0.0, 0.25)]
    mesh, P, _ = subtri_circles_mesh(circles, h=h, levels=2, half_width=0.88)
    materials = {0: (4.0, 1.0), 1: (1.0, 1.0)}
    n_steps = 20_000

    def step_invariant(solver, u_prev, v_old):
        """Leapfrog invariant after one_step() transitions (u_prev, v_old) -> (u, v).

        TM updates v first (v^{n-1/2} -> v^{n+1/2} using u^n = u_prev) then u^n -> u^{n+1};
        TE updates u first (u^{n-1} -> u^n = solver.u using v^{n-1/2}) then v^{n-1/2} -> v^{n+1/2}.
        Thus the primary field at the integer time level straddled by (v_old, solver.v) is
        u_prev in TM and solver.u in TE.
        """
        u_mid = u_prev if solver.pol == "TM" else solver.u
        return _leapfrog_energy(solver, u_mid, v_old, solver.v)

    col_nnz = np.diff(P.tocsc().indptr)
    h_node = int(np.flatnonzero(col_nnz == 0)[0])
    P_scaled = P.tolil(copy=True)
    P_scaled.data[h_node] = [1.1 * w for w in P_scaled.data[h_node]]
    P_scaled = P_scaled.tocsr()

    for pol in ("TM", "TE"):
        solver = SubTriFIT2D(mesh, P, materials=materials, polarization=pol, cfl=0.9)
        rng = np.random.default_rng(301 if pol == "TM" else 302)
        u0 = rng.standard_normal(solver.free.size)
        solver.u[:] = u0

        u_prev = solver.u.copy()
        v_old = solver.v.copy()
        solver.one_step()
        e0 = step_invariant(solver, u_prev, v_old)
        assert e0 > 0.0

        for _ in range(n_steps - 1):
            u_prev = solver.u.copy()
            v_old = solver.v.copy()
            solver.one_step()
        e_end = step_invariant(solver, u_prev, v_old)
        drift = abs(e_end - e0) / e0
        assert drift < 1e-10

        # Documented non-failure mode: scaling one hanging node's P row by 1.1
        # preserves operator symmetry (GT = (G P)^T), so energy is still conserved.
        s_sym = SubTriFIT2D(
            mesh, P_scaled, materials=materials, polarization=pol, cfl=0.9
        )
        s_sym.u[:] = u0
        u_prev_s = s_sym.u.copy()
        v_old_s = s_sym.v.copy()
        s_sym.one_step()
        e0_s = step_invariant(s_sym, u_prev_s, v_old_s)
        for _ in range(n_steps - 1):
            u_prev_s = s_sym.u.copy()
            v_old_s = s_sym.v.copy()
            s_sym.one_step()
        e_end_s = step_invariant(s_sym, u_prev_s, v_old_s)
        assert abs(e_end_s - e0_s) / e0_s < 1e-10

        # True failure mode: perturb one non-zero entry of solver.GT by 1%
        s_mut = SubTriFIT2D(
            mesh, P, materials=materials, polarization=pol, dt=solver.dt
        )
        s_mut.u[:] = u0
        s_mut.GT = s_mut.GT.copy()
        s_mut.GT.data[0] *= 1.01
        u_prev_m = s_mut.u.copy()
        v_old_m = s_mut.v.copy()
        s_mut.one_step()
        e0_m = step_invariant(s_mut, u_prev_m, v_old_m)
        blown = False
        for _ in range(n_steps - 1):
            u_prev_m = s_mut.u.copy()
            v_old_m = s_mut.v.copy()
            s_mut.one_step()
            if not np.all(np.isfinite(s_mut.u)) or np.max(np.abs(s_mut.u)) > 1e6:
                blown = True
                break
        if not blown:
            e_end_m = step_invariant(s_mut, u_prev_m, v_old_m)
            drift_m = abs(e_end_m - e0_m) / abs(e0_m)
            assert drift_m > 1e-8  # correct scheme drifts ~1e-15; mutant ~1e-6


def test_interface_reflection_order():
    """Spurious reflection from a vacuum subtri patch decays as H^p (p >= 1.5); injection fails.

    Measured reflection R(H) = max_{t >= t_gate} |r(t)| / max |y_ref(t)| for
    H in {lam/8, lam/12, lam/16} (levels=2, h = H/4):
      - Correct (0.5, 0.5) hanging-node interpolation:
          R(lam/8)  = 1.614e-2
          R(lam/12) = 5.913e-3
          R(lam/16) = 3.729e-3
          fitted slope = 2.136 (>= 1.5)
      - Mutant (1.0, 0.0) hanging-node injection:
          R_mut(lam/8)  = 1.750e-2
          R_mut(lam/12) = 1.338e-2
          R_mut(lam/16) = 7.724e-3 (2.07x > 2x larger at finest H)
          fitted slope  = 1.146 (< 1.2)
    """
    lam = 1.0
    f0 = C0 / lam
    # Differentiated Gaussian of centre wavelength lam; spectrum in [lam/1.5, 1.5*lam]
    src = GaussianSource(f0, 0.20 * f0, cutoff=4.0)
    src_pt = (-1.6 * lam, 0.0)
    prb_pt = (-1.2 * lam, 0.0)
    # Direct pulse centre passes prb_pt at src.t0 + 0.4*lam/C0; gate after main lobe (+1.5*sigma_t)
    t_gate = src.t0 + 0.4 * lam / C0 + 1.5 * src.sigma_t
    t_end = src.end_time + 4.5 * lam / C0

    Hs = []
    Rs = []
    Rs_mut = []
    for N in (8, 12, 16):
        H = lam / N
        h = H / 4.0
        L = 2.5 * lam
        mesh, P, info = subtri_circles_mesh(
            [(0.0, 0.0, 0.6 * lam)],
            h=h,
            levels=2,
            band=2,
            grade=2,
            half_width=L,
            pml_cells=10,
        )
        assert np.isclose(info["H"], H, rtol=0.0, atol=1e-15)
        assert np.isclose(info["half_width"], L, rtol=0.0, atol=1e-15)

        # Reference coarse square grid with spacing H and identical outer domain
        M = int(round(info["half_width_pml"] / H))
        xc = H * np.arange(-M, M + 1)
        ref_mesh = PolyMesh.rectilinear(xc, xc)

        # Check node coordinates coincide far from the patch (|x| >= 1.2*lam or |y| >= 1.2*lam)
        far_comp = mesh.nodes[
            np.maximum(np.abs(mesh.nodes[:, 0]), np.abs(mesh.nodes[:, 1])) >= 1.2 * lam
        ]
        far_ref = ref_mesh.nodes[
            np.maximum(np.abs(ref_mesh.nodes[:, 0]), np.abs(ref_mesh.nodes[:, 1]))
            >= 1.2 * lam
        ]
        assert far_comp.shape == far_ref.shape
        dist_far, _ = cKDTree(far_ref).query(far_comp)
        assert dist_far.max() < 1e-12 * H

        pml = dict(x=(-L, L), y=(-L, L), thickness=10 * H)
        materials = {0: (1.0, 1.0), 1: (1.0, 1.0)}
        s_comp = SubTriFIT2D(
            mesh, P, materials=materials, polarization="TM", pml=pml, cfl=0.9
        )
        dt = s_comp.dt
        s_ref = PolygonFIT2D(
            ref_mesh, materials={0: (1.0, 1.0)}, polarization="TM", pml=pml, dt=dt
        )

        # Mutant: hanging nodes with (1, 0) injection
        P_mut = P.tolil(copy=True)
        col_nnz = np.diff(P.tocsc().indptr)
        for h_node in np.flatnonzero(col_nnz == 0):
            P_mut.rows[h_node] = P_mut.rows[h_node][:1]
            P_mut.data[h_node] = [1.0]
        s_mut = SubTriFIT2D(
            mesh,
            P_mut.tocsr(),
            materials=materials,
            polarization="TM",
            pml=pml,
            dt=dt,
        )

        for s in (s_comp, s_ref, s_mut):
            s.add_point_source(src_pt, src)
        p_comp = s_comp.add_probe([prb_pt])
        p_ref = s_ref.add_probe([prb_pt])
        p_mut = s_mut.add_probe([prb_pt])

        n_steps = int(np.ceil(t_end / dt))
        s_comp.run(n_steps)
        s_ref.run(n_steps)
        s_mut.run(n_steps)

        y_comp = s_comp.probe_signal(p_comp).ravel()
        y_ref = s_ref.probe_signal(p_ref).ravel()
        y_mut = s_mut.probe_signal(p_mut).ravel()
        t_arr = np.arange(1, n_steps + 1) * dt
        mask = t_arr >= t_gate
        peak_dir = float(np.max(np.abs(y_ref)))

        Hs.append(H)
        Rs.append(float(np.max(np.abs(y_comp[mask] - y_ref[mask])) / peak_dir))
        Rs_mut.append(float(np.max(np.abs(y_mut[mask] - y_ref[mask])) / peak_dir))

    slope = float(np.polyfit(np.log(Hs), np.log(Rs), 1)[0])
    slope_mut = float(np.polyfit(np.log(Hs), np.log(Rs_mut), 1)[0])

    assert slope >= 1.5
    assert Rs_mut[-1] > 2.0 * Rs[-1]
    assert slope_mut < 1.2
