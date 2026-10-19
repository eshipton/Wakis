"""
Curvilinear (body-fitted O-grid) time-domain NFIT, `wakis.curvilinear`.

Every check has a failure mode it can detect (mutations included):
* energy: the leapfrog conserves the discrete energy to round-off with the
  symmetric Hodge; an asymmetric Hodge (mutation) does not;
* eigenmodes: rod in a circular cavity (TM, TE) converge at second order to
  the exact Bessel solution; an empty square cavity meshed with the
  multi-block O-grid + Cartesian mesh matches the analytic square modes;
* PML: radial complex-stretching layer reflects < 2e-3 against a large
  domain reference; dropping the curvature factor s_phi = r~/r (mutation)
  is > 10x worse and no layer reflects O(1);
* plane wave: the scattered-field source reproduces the Mie series of a
  dispersive (Lorentz) cylinder in TE; removing the incident term fails;
* the coated-cylinder series reduces to the homogeneous one.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.constants import c as C0

from wakis.curvilinear import (
    CurvilinearFIT2D,
    hodge_components,
    ogrid_cartesian_mesh,
    ogrid_mesh,
    radial_nodes,
)


def _cavity(nt, polar=True):
    a, b = 1.0, 0.5
    h = 2 * np.pi * a / nt
    return ogrid_mesh(radial_nodes([0.9 * b, b, a], lambda r: 0.8 * h), nt,
                      interfaces=[b], polar=polar)  # fmt: skip


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_energy_conserved_and_asymmetric_mutation(pol):
    mesh = _cavity(32)
    s = CurvilinearFIT2D(mesh, {1: [4.0, 1.0]}, polarization=pol)
    W = s.W
    assert abs(W - W.T).max() < 1e-12 * abs(W).max()
    rng = np.random.default_rng(1)

    def drift(solver):
        solver.u = rng.standard_normal(solver.u.size)
        solver.u_old = solver.u.copy()
        solver.one_step()
        e0 = solver.energy()
        es = []
        for _ in range(1500):
            solver.one_step()
            es.append(solver.energy())
        es = np.array(es)
        return np.nanmax(np.abs(es / e0 - 1)) if np.all(np.isfinite(es)) else np.inf

    assert drift(s) < 1e-12
    # mutation: non-symmetric metric coupling (upper triangle scaled)
    Wm = W + 0.5 * sp.triu(W, 1)
    sm = CurvilinearFIT2D(mesh, {1: [4.0, 1.0]}, polarization=pol, W_override=Wm,
                          dt=s.dt)  # fmt: skip
    assert drift(sm) > 1e-4


def test_corner_hodge_is_standard_fit_on_rectangles():
    from wakis.curvilinear.mesh import QuadMesh

    x = np.array([0.0, 1.0, 3.0])
    y = np.array([0.0, 2.0, 2.5])
    X, Y = np.meshgrid(x, y, indexing="ij")
    idx = np.arange(9).reshape(3, 3)
    cells = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    mesh = QuadMesh(np.stack([X.ravel(), Y.ravel()], 1), cells, np.zeros(4))
    hc = hodge_components(mesh)
    W = (hc["R"].T @ sp.diags(hc["w"]) @ hc["R"]).toarray()
    assert np.allclose(W, np.diag(np.diag(W)))  # orthogonal: diagonal
    # x-edge from (0,0) to (1,0): dual length 1 (half of y-cell 2), length 1
    L = np.linalg.norm(mesh.nodes[mesh.edges[:, 1]] - mesh.nodes[mesh.edges[:, 0]], axis=1)
    e = np.flatnonzero((mesh.edges[:, 0] == 0) & (mesh.edges[:, 1] == 3))[0]
    assert np.isclose(W[e, e] * L[e], 1.0)
    # skewed quad: corner coupling appears and W stays SPD
    nodes = mesh.nodes.copy()
    nodes[4] += [0.3, 0.2]
    W2 = hodge_components(QuadMesh(nodes, cells, np.zeros(4)))
    W2 = (W2["R"].T @ sp.diags(W2["w"]) @ W2["R"]).toarray()
    assert np.abs(W2 - np.diag(np.diag(W2))).max() > 1e-3
    assert np.linalg.eigvalsh(W2).min() > 0


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_rod_in_cavity_second_order(pol):
    from wakis.unstructured.analytic import rod_in_cavity

    ref = rod_in_cavity(1.0, 0.5, 6.0, pol, n_modes=4)
    errs = []
    for nt in (64, 128, 256):
        s = CurvilinearFIT2D(_cavity(nt), {1: [6.0, 1.0]}, polarization=pol)
        errs.append(np.abs(s.eigenmodes(4) - ref) / ref)
    errs = np.array(errs)
    rms = np.sqrt(np.mean(errs**2, axis=1))
    order = np.log2(rms[:-1] / rms[1:])
    # pre-asymptotic at coarse grids (measured orders 1.1, 1.5, 2.5 from
    # n_theta = 32 to 256): require second order on the finest pair
    assert rms[-1] < 3e-4, rms
    assert order[-1] > 1.8 and order[0] > 1.2, order


def test_multiblock_square_cavity():
    """O-grid annulus + transition + Cartesian block, PEC square box of side
    2 L: TM modes k = (pi / 2L) sqrt(m^2 + n^2)."""
    nt, L = 48, 1.0
    radii = radial_nodes([0.3, 0.6], lambda r: 2 * np.pi * 0.6 / nt)
    S = 0.7
    hc = 2 * S / (nt // 4)
    xo = S + hc * np.arange(1, int(round((L - S) / hc)) + 1)
    xo = S + (xo - S) * (L - S) / (xo[-1] - S)
    mesh = ogrid_cartesian_mesh(radii, nt, S, xo, m_tr_out=2)
    s = CurvilinearFIT2D(mesh, {}, polarization="TM")
    m, n = np.meshgrid(np.arange(1, 5), np.arange(1, 5))
    ref = np.sort((np.pi / (2 * L) * np.hypot(m, n)).ravel())[:6]
    k = s.eigenmodes(6)
    assert np.max(np.abs(k - ref) / ref) < 1e-2
    # energy is conserved on the multi-block mesh as well
    s.u = np.random.default_rng(0).standard_normal(s.u.size)
    s.u_old = s.u.copy()
    s.one_step()
    e0 = s.energy()
    s.run(500)
    assert abs(s.energy() / e0 - 1) < 1e-12


def _pml_reflection(pml, pol="TM"):
    from wakis.sources import GaussianSource

    lam = 1e-6
    h, R0 = lam / 12, 1.0e-6
    nt = 4 * int(np.ceil(2 * np.pi * R0 / h / 4 / 1.5))
    src = GaussianSource(C0 / lam, 0.5 * C0 / lam)

    def run(r_out, pml, n, dt=None):
        radii = radial_nodes([0.5e-6, R0, r_out], lambda r: h)
        mesh = ogrid_mesh(radii, nt)
        s = CurvilinearFIT2D(mesh, {}, polarization=pol, pml=pml, dt=dt)
        s.add_point_source((0.3e-6, 0.2e-6), src)
        ring, _ = mesh.ring(radii[np.argmin(np.abs(radii - 0.85e-6))])
        loc = np.searchsorted(s.free, ring)
        out = []
        for _ in range(n or 1):
            s.one_step()
            out.append(s.u[loc].copy())
        return np.array(out), s

    big = 4.5e-6
    _, s0 = run(big, None, 1)
    n = int(2 * (big - R0) / C0 / s0.dt)
    ref, _ = run(big, None, n, s0.dt)
    r_out = pml["r1"] if pml else R0 + 8 * h
    u, _ = run(r_out, pml, n, s0.dt)
    return np.abs(u - ref).max() / np.abs(ref).max()


def test_radial_pml_and_curvature_mutation():
    R0, h = 1.0e-6, 1e-6 / 12
    good = _pml_reflection(dict(kind="radial", r0=R0, r1=R0 + 12 * h))
    bad = _pml_reflection(dict(kind="radial", r0=R0, r1=R0 + 12 * h, curvature=False))
    none = _pml_reflection(None)
    assert good < 2e-3, good
    assert bad > 10 * good, (good, bad)
    assert none > 0.3


def test_dispersive_te_plane_wave_mie():
    """Lorentz cylinder (eps ~ 4.0..6.7, lossy), TE scattered-field source vs
    the series with the same eps(omega); low resolution, so a loose bound,
    and the mutation without the incident term must fail it."""
    from wakis.analytic import cylinder_coefficients
    from wakis.cylinder_harmonics import (
        fit_harmonics,
        mie_coefficients_from_fit,
        project_harmonics,
    )
    from wakis.dispersive import DispersiveMaterial, Lorentz
    from wakis.sources import GaussianSource

    a = 1e-6
    x = np.linspace(2, 5, 7)
    freqs = x * C0 / (2 * np.pi * a)
    w0 = 2 * np.pi * freqs[-1] * 1.3
    mat = DispersiveMaterial(2.0, [Lorentz(2.0, w0, 0.1 * w0)])
    h = 2 * np.pi * a / (5 * 2.6) / 12
    nt = 8 * int(np.ceil(2 * np.pi * a / h / 8))  # node at theta = 0
    radii_x = [1.2e-6, 1.4e-6, 1.6e-6]

    def size(r):
        return h if r < a + 2 * h else min(2 * h, h * 1.2 ** ((r - a - 2 * h) / h))

    r0 = 1.7e-6
    radii = radial_nodes([0.9 * a, a] + radii_x + [r0, r0 + 10 * 2 * h],
                         lambda r: 2 * h if r >= r0 else size(r))  # fmt: skip
    mesh = ogrid_mesh(radii, nt, interfaces=[a])
    q_err = []
    for use_source in (True, False):
        s = CurvilinearFIT2D(mesh, {1: mat}, polarization="TE",
                             pml=dict(kind="radial", r0=r0, r1=radii[-1]))  # fmt: skip
        f0, df = 0.5 * (freqs[0] + freqs[-1]), 0.3 * (freqs[-1] - freqs[0])
        s.add_plane_wave(GaussianSource(f0, df))
        if not use_source:
            s.plane_wave["GQ"] = 0 * s.plane_wave["GQ"]
            for g in s._te_groups:
                g["I"] = 0 * g["I"]
        locs, ths = zip(*[mesh.ring(r) for r in radii_x])
        locs = [np.searchsorted(s.free, i) for i in locs]
        n_steps = int(600e-15 / s.dt)
        times = (np.arange(n_steps) + 0.5) * s.dt
        acc = [np.zeros((len(freqs), len(i)), complex) for i in locs]
        ker = np.exp(-2j * np.pi * np.outer(times, freqs)) * s.dt
        for n in range(n_steps):
            s.one_step()
            for A_, i in zip(acc, locs):
                A_ += np.outer(ker[n], s.u[i])
        m_pts = len(locs[0])
        assert all(len(i) == m_pts for i in locs)
        n_max = 12
        F = project_harmonics(np.stack(acc, 1), n_max)
        k = 2 * np.pi * freqs / C0
        al, ga, _ = fit_harmonics(F, k, radii_x, n_max)
        u0 = (np.exp(-2j * np.pi * np.outer(freqs, times)) @ s.incident_field(np.zeros((1, 2)), times)) * s.dt
        c = mie_coefficients_from_fit(al, ga, np.conj(u0), n_max)
        q = 2 / x * np.sum(np.abs(c) ** 2, 1)
        eps = mat.epsilon(freqs)
        q_ref = []
        for xx, e in zip(x, eps):
            cc = cylinder_coefficients(np.sqrt(e), xx, "TE", n_terms=n_max)
            q_ref.append(2 / xx * (abs(cc[0]) ** 2 + 2 * np.sum(np.abs(cc[1:]) ** 2)))
        q_err.append(np.sqrt(np.mean((q / np.array(q_ref) - 1) ** 2)))
    assert q_err[0] < 0.03, q_err
    assert q_err[1] > 0.5, q_err


def test_coated_cylinder_series_reduces_to_homogeneous():
    from wakis.analytic import cylinder_coefficients
    from wakis.analytic.coated_cylinder import coated_cylinder_coefficients

    for pol in ("TM", "TE"):
        for m in (2.0, 1.5 - 0.1j, 0.3 - 3j):
            c1 = coated_cylinder_coefficients(m, m, 3.0, 5.0, pol)
            c2 = cylinder_coefficients(m, 5.0, pol)
            assert np.abs(c1 - c2).max() < 1e-12
        # a real shell changes the answer (the test can fail)
        c3 = coated_cylinder_coefficients(2.0, 3.5, 4.9, 5.0, pol)
        assert np.abs(c3 - cylinder_coefficients(2.0, 5.0, pol)).max() > 1e-2
