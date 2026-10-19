"""
Local time stepping (stabilized two-level leapfrog LTS) for UnstructuredFIT2D.

Mesh: a rod (eps 4, radius 0.5) in a PEC cavity (radius 1), graded so that
h is constant inside the rod and in a 2h band around it, then grows to 2h.
The vacuum band sets the global step; the rod (n = 2) and the coarse vacuum
can take twice that step, so p = 2.

Every test has a mutation that makes it fail:

* energy: the exactly conserved LTS energy drifts by < 1e-11 over 1e5 steps;
  dropping the coupling of F to its coarse neighbours (the implicit buffer)
  or perturbing the polynomial normalization by 1 % drifts it by > 1e-4;
  the unstabilized Diaz-Grote polynomial (nu = 0) is marginal (rho = 1) and
  the physical energy grows by > 10x.
* eigenfrequency: the time error of the LTS converges at 2nd order and is
  about (dt_lts / dt_global)^2 = 4x the global one; the total error matches
  the global stepping within 2x of the spatial error.
* adjoint: one LTS step and its transpose pass the dot-product test at 1e-14;
  an unmodified transpose (r^T replaced by r) fails it.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.constants import c
from scipy.sparse.linalg import eigsh

from wakis.unstructured import UnstructuredFIT2D, analytic, graded_disk_mesh
from wakis.unstructured import lts as L

B_ROD, A_CAV, EPS = 0.5, 1.0, 4.0


def graded_rod_mesh(h):
    def size(r):
        d = r - B_ROD - 2 * h
        return h if d <= 0 else min(2 * h, h * 1.2 ** (d / h))

    return graded_disk_mesh([B_ROD, A_CAV], size)


@pytest.fixture(scope="module")
def mesh08():
    return graded_rod_mesh(0.08)


def solver(mesh, pol, **lts):
    return UnstructuredFIT2D(
        mesh, {0: [EPS, 1.0]}, polarization=pol, lts=dict(p=2, **lts)
    )


def run_energy(s, n_steps, mutate=None):
    if mutate == "buffer":
        s.lts.kappa[:] = 0  # F no longer sees its coarse neighbours
    elif mutate == "normalization":
        s.lts.out *= 1.01  # r(0) != 1
    s.u = np.random.default_rng(0).standard_normal(s.free.size)
    E, P = [], []
    for n in range(n_steps):
        u0 = s.u.copy()
        s.one_step()
        if n % 1000 == 0 or n == n_steps - 1:
            E.append(L.modified_energy(s, u0, s.u))
            P.append(s.energy())
    E, P = np.array(E), np.array(P)
    return np.abs(E / E[0] - 1).max(), P.max() / P[len(P) // 10 :].min()


class TestOperator:
    @pytest.mark.parametrize("p", [2, 3, 4])
    def test_apply_matches_reference_and_transpose(self, mesh08, p):
        s = UnstructuredFIT2D(mesh08, {0: [EPS, 1.0]}, lts=dict(p=p, rho_max=0.99))
        lts = s.lts
        rng = np.random.default_rng(1)
        y, z = rng.standard_normal((2, lts.n))
        assert np.allclose(lts.apply(y), lts.apply_full(y), rtol=0, atol=1e-13)
        assert np.allclose(lts.apply(y), L.lts_matrix(lts) @ y, rtol=0, atol=1e-12)
        # r(X) y = y off F; r(0) = 1
        off = np.ones(lts.n, bool)
        off[lts.F] = False
        assert np.array_equal(lts.apply(y)[off], y[off])
        dot = (z @ lts.apply(y) - y @ lts.apply_T(z)) / (
            np.linalg.norm(z) * np.linalg.norm(lts.apply(y))
        )
        assert abs(dot) < 1e-14
        # K r(X) is symmetric (energy conservation rests on this)
        Kp = s.K @ L.lts_matrix(lts)
        assert abs(Kp - Kp.T).max() < 1e-13 * abs(Kp).max()

    def test_stability_polynomial(self):
        """nu = 0 touches 4 inside the range (marginal); nu > 0 stays below."""
        x = np.linspace(0, 15, 3001)
        assert L.stability_polynomial(x, 2, 0.0).max() == pytest.approx(4, abs=1e-6)
        # nu = 0.15: valid range x <= 14.94 (fine substep at 0.966 of its limit)
        xs = np.linspace(0, 14.9, 3001)
        phi = L.stability_polynomial(xs, 2, 0.15)
        assert phi.max() < 3.75 and phi.min() >= -1e-12
        assert L.stability_polynomial(15.2, 2, 0.15) < 0  # beyond: unstable

    def test_partition(self, mesh08):
        s = solver(mesh08, "TM")
        assert s.dt == pytest.approx(
            2 * UnstructuredFIT2D(mesh08, {0: [EPS, 1.0]}).dt, rel=1e-3
        )
        assert s.courant < 0.98
        assert 0.1 < s.lts.fine_fraction < 0.9
        # fine nodes sit in the vacuum band around the rod
        r = np.hypot(*mesh08.nodes[s.free[s.lts.F]].T)
        assert np.median(r) > B_ROD and np.median(r) < B_ROD + 0.4


class TestEnergy:
    @pytest.mark.parametrize("pol", ["TM", "TE"])
    def test_conserved_1e5_steps(self, mesh08, pol):
        drift, growth = run_energy(solver(mesh08, pol), 100_000)
        assert drift < 1e-11
        assert growth < 2.5  # physical energy bounded (TE starts with v = 0)

    @pytest.mark.parametrize("mutate", ["buffer", "normalization"])
    def test_mutations_drift(self, mesh08, mutate):
        drift, _ = run_energy(solver(mesh08, "TM"), 20_000, mutate)
        assert drift > 1e-4

    def test_unstabilized_is_marginal(self, mesh08):
        s = solver(mesh08, "TM", nu=0.0, rho_max=1.0)
        assert s.courant > 0.999
        drift, growth = run_energy(s, 100_000)
        assert drift < 1e-10  # still conserved, but not positive definite
        assert growth > 10


def k_modes(s, use_lts, semi=False, k=4):
    K = s.K @ L.lts_matrix(s.lts) if use_lts else s.K
    K = 0.5 * (K + K.T)
    shift = -1e-3 * s.K.diagonal().mean() / s.mass.mean()
    w2 = np.sort(eigsh(K, k=k, M=sp.diags(s.mass), sigma=shift, which="LM")[0])
    w2 = w2[w2 > 1e-6 * w2.max()]
    if semi:
        return np.sqrt(w2) / c
    # leapfrog dispersion: sin(w dt / 2) = dt/2 sqrt(lambda)
    return 2 / s.dt * np.arcsin(s.dt / 2 * np.sqrt(w2)) / c


class TestEigenfrequency:
    @pytest.mark.parametrize("pol", ["TM", "TE"])
    def test_second_order_like_global(self, pol):
        ref = analytic.rod_in_cavity(A_CAV, B_ROD, EPS, pol, 1)[0]
        t_glob, t_lts, tot = [], [], []
        for h in (0.02, 0.01):  # 4.5k and 17k nodes
            m = graded_rod_mesh(h)
            g = UnstructuredFIT2D(m, {0: [EPS, 1.0]}, polarization=pol)
            s = solver(m, pol)
            semi = k_modes(g, False, semi=True)[0]
            kg, kl = k_modes(g, False)[0], k_modes(s, True)[0]
            t_glob.append(kg / semi - 1)
            t_lts.append(kl / semi - 1)
            tot.append((abs(semi / ref - 1), abs(kg / ref - 1), abs(kl / ref - 1)))
        order = np.log2(t_lts[0] / t_lts[1])
        assert 1.6 < order < 2.4, t_lts
        assert 2.5 < t_lts[1] / t_glob[1] < 5  # ~ (2 dt)^2 / dt^2
        space, glob, lts = tot[1]
        assert lts < glob + 2 * space and lts < 1e-4

    def test_time_domain_matches_operator(self, mesh08):
        """one_step is the operator analysed above (no hidden extra terms)."""
        for pol in ("TM", "TE"):
            s = solver(mesh08, pol)
            rng = np.random.default_rng(2)
            s.u, s.v = rng.standard_normal(s.free.size), rng.standard_normal(s.v.size)
            u1, v1 = L.linear_step(s, s.u, s.v)
            s.one_step()
            assert np.allclose(s.u, u1, rtol=0, atol=1e-12 * np.abs(u1).max())
            assert np.allclose(s.v, v1, rtol=0, atol=1e-12 * np.abs(v1).max())


class TestAdjoint:
    @pytest.mark.parametrize("pol", ["TM", "TE"])
    def test_dot_product(self, mesh08, pol):
        s = solver(mesh08, pol)
        rng = np.random.default_rng(3)
        n, m = s.free.size, s.v.size
        # balanced scales: dt G u ~ v and dt^2 S / M ~ 1
        u, ub = rng.standard_normal((2, n))
        v, vb = s.dt * rng.standard_normal(m), rng.standard_normal(m) / s.dt
        u1, v1 = L.linear_step(s, u, v)
        ua, va = L.linear_step_T(s, ub, vb)
        lhs, rhs = ub @ u1 + vb @ v1, ua @ u + va @ v
        scale = np.linalg.norm(ub) * np.linalg.norm(u1)
        scale += np.linalg.norm(vb) * np.linalg.norm(v1)
        assert abs(lhs - rhs) / scale < 1e-14
        # mutation: forget to transpose r(X)
        lts = s.lts
        apply_T = lts.apply_T
        lts.apply_T = lts.apply
        ua, va = L.linear_step_T(s, ub, vb)
        lts.apply_T = apply_T
        assert abs(lhs - (ua @ u + va @ v)) / scale > 1e-6
