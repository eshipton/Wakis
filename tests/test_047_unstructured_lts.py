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
    @pytest.mark.slow
    @pytest.mark.parametrize("pol", ["TM", "TE"])
    def test_conserved_1e5_steps(self, mesh08, pol):
        drift, growth = run_energy(solver(mesh08, pol), 100_000)
        assert drift < 1e-11
        assert growth < 2.5  # physical energy bounded (TE starts with v = 0)

    @pytest.mark.parametrize("mutate", ["buffer", "normalization"])
    def test_mutations_drift(self, mesh08, mutate):
        drift, _ = run_energy(solver(mesh08, "TM"), 20_000, mutate)
        assert drift > 1e-4

    @pytest.mark.slow
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
    @pytest.mark.slow
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


# ------------------------------------------------------------------ dispersive
def _state(s):
    """Views of all state arrays (fields + ADE polarizations), TE and TM."""
    arrs = [("u", None), ("v", None)]
    if s.pol == "TE":
        for gi, g in enumerate(s._te_groups):
            arrs.append(("teE", gi))
            for k in range(len(g["lorentz"])):
                arrs += [("teP", (gi, k, 3)), ("teP", (gi, k, 4))]
    else:
        for gi, g in enumerate(s._update.groups):
            for k in range(len(g["lorentz"])):
                arrs += [("tmP", (gi, k, 3)), ("tmP", (gi, k, 4))]
    return arrs


def _get(s, key):
    kind, i = key
    if kind in ("u", "v"):
        return getattr(s, kind)
    if kind == "teE":
        return s._te_groups[i]["E"]
    gi, k, j = i
    grp = s._te_groups if kind == "teP" else s._update.groups
    return grp[gi]["lorentz"][k][j]


def _set(s, key, x):
    kind, i = key
    if kind in ("u", "v"):
        setattr(s, kind, x.copy())
    elif kind == "teE":
        s._te_groups[i]["E"] = x.copy()
    else:
        gi, k, j = i
        grp = s._te_groups if kind == "teP" else s._update.groups
        t = list(grp[gi]["lorentz"][k])
        t[j] = x.copy()
        grp[gi]["lorentz"][k] = tuple(t)


def spectral_radius(s):
    """max |eigenvalue| of the full one-step map (fields + ADE), dense.

    A lossless leapfrog has its whole spectrum on the unit circle (Arnoldi
    does not converge), so the map is assembled column by column (small
    meshes only)."""
    keys = _state(s)
    sizes = [_get(s, key).size for key in keys]
    n = sum(sizes)
    T = np.zeros((n, n))
    for col in range(n):
        x = np.zeros(n)
        x[col] = 1.0
        o = 0
        for key, m in zip(keys, sizes):
            _set(s, key, x[o : o + m])
            o += m
        s.one_step()
        T[:, col] = np.concatenate([_get(s, key) for key in keys])
    # balance units: diagonal similarity does not change the eigenvalues
    return np.abs(np.linalg.eigvals(T)).max()


def drude_solver(pol, mesh, gamma, kp=4.0, **lts):
    from wakis.dispersive import DispersiveMaterial, Drude

    drude = DispersiveMaterial(1.0, [Drude(kp * c, gamma)])
    return UnstructuredFIT2D(mesh, {0: drude}, polarization=pol, lts=lts or None)


class TestDispersive:
    """Lorentz/Drude cells inside F: the poles are a force frozen over the
    coarse step. No proof; the one-step map (fields + ADE states) must have
    spectral radius <= 1, and the plasmon resonance must match. The pole is
    stepped at the coarse dt: omega_p dt_coarse must pass the von Neumann
    check (`DispersiveMaterial.is_stable` at the LTS ratio rho), which the
    solver enforces; omega_p dt_c = 0.8 at rho = 0.966 gives |lambda| = 1.8 (h = 0.16)."""

    @pytest.fixture(scope="class")
    def mesh16(self):
        return graded_rod_mesh(0.16)  # ~1.4k state variables: dense map

    @pytest.mark.parametrize("pol", ["TE", "TM"])
    @pytest.mark.parametrize("gamma", [0.0, 0.3 * c])
    def test_amplification_spectral_radius(self, mesh16, pol, gamma):
        s = drude_solver(pol, mesh16, gamma, kp=2.0, p=2)  # omega_p dt_c = 0.4
        # the metal rod (eps_inf = 1, h) is fine: it is the CFL bottleneck
        r = np.hypot(*mesh16.nodes[s.free[s.lts.F]].T)
        assert (r < B_ROD).sum() >= 5  # metal-interior nodes are fine
        # round-off on the unit circle of a non-normal map: ~1e-7 (global too)
        assert spectral_radius(s) < 1 + 2e-6

    def test_mutation_no_fine_region_is_unstable(self, mesh16):
        s = drude_solver("TE", mesh16, 0.0, kp=2.0, p=2)
        s.lts.correction = lambda y: np.zeros(s.lts.F.size)  # no fine substeps
        assert spectral_radius(s) > 1.01

    def test_pole_too_fast_for_coarse_step_is_refused(self, mesh08):
        with pytest.raises(ValueError, match="unstable"):
            drude_solver("TE", mesh08, 0.0, kp=8.0, p=2)  # omega_p dt_c = 1.6

    @pytest.mark.slow
    def test_plasmon_matches_eigenmode(self):
        """Time-domain plasmon (TE Drude rod, k_p a = 4) with LTS vs global.

        The n = 1 plasmon is weak at the probe (|a| ~ 2e-3) next to a mode
        at 1.033 f_sp (|a| ~ 2e-2): fit the whole record after the source
        ends in 0.9-1.1 f_sp (a later start and a 0.97-1.03 window found no
        mode, for global stepping too: studies/unstructured-lts,
        plasmon_diag.py). Expected: leapfrog time error ~ dt^2, i.e. the
        LTS shift from the eigenfrequency ~ 4x the global one."""
        from wakis.resonances import harminv

        m = graded_rod_mesh(0.04)
        f_an = analytic.drude_rod_in_cavity_te(A_CAV, B_ROD, 4.0, n=1)[1] * c
        shifts = {}
        for mode in ("global", "lts"):
            s = drude_solver("TE", m, 0.0, **({"p": 2} if mode == "lts" else {}))
            f_sp = s.dispersive_eigenmode(f_an / (2 * np.pi))
            s0 = 1 / (2 * np.pi * 0.3 * f_sp)

            def pulse(t, s0=s0, f0=f_sp):
                return np.exp(-(((t - 5 * s0) / s0) ** 2) / 2) * np.sin(
                    2 * np.pi * f0 * (t - 5 * s0)
                )

            s.add_point_source([0.6, 0.1], pulse, [1, 0.3])
            probe = s.add_probe([[0.55, -0.2]])
            n = int(60 / f_sp / s.dt)
            s.run(n)
            sig = s.probe_signal(probe)[:, 0]
            i0 = int(10 * s0 / s.dt) + 1  # source off
            res = harminv(sig[i0:], s.dt, 0.9 * f_sp, 1.1 * f_sp, t0=(i0 + 1) * s.dt)
            f = np.real(np.atleast_1d(res.frequency))
            shifts[mode] = f[np.argmin(np.abs(f / f_sp - 1))] / f_sp - 1
            # bounded (lossless cavity): no growth over 60 periods
            assert np.abs(sig[-n // 10 :]).max() < 2 * np.abs(sig[i0 : 2 * i0]).max()
        # observed: global +3.6e-4, LTS +1.2e-3 (h = 0.04)
        assert abs(shifts["global"]) < 1e-3
        assert abs(shifts["lts"]) < 2e-3
        assert abs(shifts["lts"]) < 6 * abs(shifts["global"]) + 1e-4
