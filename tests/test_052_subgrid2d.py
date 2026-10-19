"""
Local Cartesian refinement (subgridding) for 2D FIT: `wakis.subgrid`.

Each test has a mutation that must make it fail:

* eigenfrequencies of a rectangular PEC cavity with an off-centre 2:1 patch
  converge with order 2 (mutation: hanging nodes *copy* a coarse neighbour
  instead of interpolating -> order < 1);
* the discrete energy of a closed cavity with a 3-level patch and dielectrics
  crossing the interfaces is conserved to round-off over 1e5 steps
  (mutation: a non-transposed (injection) fine-to-coarse restriction grows
  exponentially, even at half the CFL limit; dt 2 % above the computed limit
  blows up);
* the plane-wave reflection of a straight interface (Bloch strip, exact
  time-discrete steady state) falls as N^-2 at 0/30/60 deg (mutation:
  injection falls as N^-1 at oblique incidence and is > 5x larger);
* the one-step transpose passes the dot-product test to 1e-13 (mutation: a
  transpose missing the hanging-node prolongation fails);
* Mie cylinder Q_sca (n = 2, x = 2..10, res 10, compact domain) within 9 %
  RMS of the exact series (mutation: without the patch the error doubles).
"""

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.constants import c as C0

from wakis.subgrid import Circle, Geometry, QuadMesh, Rect, SubgridFIT2D, refine_near
from wakis.subgrid.interface import _inject, bloch_reflection

A_CAV, B_CAV = 1.0, 0.7


def cavity(N, pol, mutate=False):
    x = np.linspace(0, A_CAV, N + 1)
    y = np.linspace(0, B_CAV, int(round(N * B_CAV / A_CAV)) + 1)
    f = refine_near(
        [lambda xm, ym, hd: (np.abs(xm - 0.37) < 0.18) & (np.abs(ym - 0.31) < 0.13)]
    )
    m = QuadMesh(x, y, ratio=2, max_level=1, refine=f)
    if mutate:
        m.P = _inject(m, 1.0).real.tocsr()
    s = SubgridFIT2D(m, polarization=pol)
    K, D = s.stiffness()
    lam = np.sort(sla.eigsh(K, k=8, M=sp.diags(D), sigma=-1.0, which="LM")[0])
    lo = 1 if pol == "TM" else 0
    exact = sorted(C0 / 2 * np.hypot(i / A_CAV, j / B_CAV)
                   for i in range(lo, 8) for j in range(lo, 8) if i + j > 0)  # fmt: skip
    fr = np.sqrt(np.abs(lam)) / (2 * np.pi)
    fr = fr[fr > 1e3][:5]
    return np.abs(fr / np.array(exact[:5]) - 1).max()


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_cavity_eigenfrequencies_second_order(pol):
    e20, e40 = cavity(20, pol), cavity(40, pol)
    order = np.log2(e20 / e40)
    assert 1.8 < order < 2.3, order
    # mutation: injection instead of interpolation at the hanging nodes
    m20, m40 = cavity(20, pol, True), cavity(40, pol, True)
    assert np.log2(m20 / m40) < 1.0
    assert m40 > 2 * e40


def closed_solver(pol, cfl=0.99):
    x = np.linspace(0, 1e-6, 17)
    y = np.linspace(0, 0.8e-6, 14)
    f = refine_near([
        lambda xm, ym, hd: np.hypot(xm - 0.45e-6, ym - 0.4e-6) < 0.25e-6,
        lambda xm, ym, hd: np.hypot(xm - 0.45e-6, ym - 0.4e-6) < 0.12e-6,
    ])  # fmt: skip
    m = QuadMesh(x, y, ratio=2, max_level=2, refine=f)
    geo = Geometry([Circle((0.5e-6, 0.42e-6), 0.2e-6, 4.0),
                    Rect(0.2e-6, 0.3e-6, 0.1e-6, 0.7e-6, 2.25)])  # fmt: skip
    return m, SubgridFIT2D(m, geo, polarization=pol, cfl=cfl)


def run_energy(s, steps, step=None):
    rng = np.random.default_rng(0)
    s.u = rng.standard_normal(s.n_dof)
    step = step or s.one_step
    step()
    e0 = s.energy()
    worst = 0.0
    for n in range(steps):
        step()
        if n % 1000 == 0:
            e = s.energy()
            if not np.isfinite(e):
                return np.inf
            worst = max(worst, abs(e / e0 - 1))
            if worst > 1e6:
                return worst
    return worst


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_energy_conserved_1e5_steps(pol):
    m, s = closed_solver(pol)
    assert m.level_counts().tolist()[2] > 0 and m.is_balanced()
    assert run_energy(s, 100_000) < 1e-12


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_energy_mutations_blow_up(pol):
    # dt 2 % above the computed limit (bypass the constructor check)
    m, s = closed_solver(pol)
    s.dt = 1.02 * s.dt_max
    assert run_energy(s, 20_000) > 1e3
    # non-reciprocal interface: the fine-to-coarse restriction injects
    m, s = closed_solver(pol, cfl=0.5)
    keep = np.isin(m.free, s.dof)
    KQ = (m.gradient() @ _inject(m, 1.0).real[:, keep]).tocsr()[s.edge_idx]
    KQt = KQ.T.tocsr()

    def step():
        f1 = s.f - s.dt * (s.K @ s.u)
        s.u_prev, s.u = s.u, s.u + s.dt * (KQt @ (s.W * f1)) / s.D
        s.f_prev, s.f = s.f, f1

    assert run_energy(s, 20_000, step) > 1e3


def test_interface_reflection_second_order():
    for th in (0, 30, 60):
        r10, r20 = bloch_reflection(10, th), bloch_reflection(20, th)
        assert r20 < 0.01
        assert r10 / r20 > 3.3, (th, r10, r20)
    # floor of the measurement (no interface)
    assert bloch_reflection(20, 30, refine=False) < 1e-4
    # 3:1 ratio is also second order
    assert bloch_reflection(10, 30, ratio=3) / bloch_reflection(20, 30, ratio=3) > 3.3
    # mutation: injection is first order at oblique incidence
    i10, i20 = (
        bloch_reflection(10, 30, mutate="inject"),
        bloch_reflection(20, 30, mutate="inject"),
    )
    assert i10 / i20 < 2.5
    assert i20 > 5 * bloch_reflection(20, 30)


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_adjoint_dot_product(pol):
    m, s = closed_solver(pol)
    rng = np.random.default_rng(1)
    u, f = rng.standard_normal(s.n_dof), rng.standard_normal(s.n_edges)
    ub, fb = rng.standard_normal(s.n_dof), rng.standard_normal(s.n_edges)
    u1, f1 = s.linear_step(u, f)
    uT, fT = s.linear_step_T(ub, fb)
    lhs, rhs = u1 @ ub + f1 @ fb, u @ uT + f @ fT
    assert abs(lhs - rhs) < 1e-13 * (abs(lhs) + abs(rhs))
    # mutation: transpose built with the injection prolongation
    keep = np.isin(m.free, s.dof)
    K_bad = (m.gradient() @ _inject(m, 1.0).real[:, keep]).tocsr()[s.edge_idx]
    K_true = s.K
    s.K = K_bad
    uT, fT = s.linear_step_T(ub, fb)
    s.K = K_true
    rhs = u @ uT + f @ fT
    assert abs(lhs - rhs) > 1e-6 * (abs(lhs) + abs(rhs))


def test_lts_energy_and_ratio():
    """Two-level LTS on the subgrid: modified energy conserved, fewer steps."""
    x = np.linspace(0, 2e-6, 33)
    y = np.linspace(0, 2e-6, 33)
    f = refine_near([lambda xm, ym, hd: np.hypot(xm - 1e-6, ym - 1e-6) < 0.4e-6])
    m = QuadMesh(x, y, ratio=2, max_level=1, refine=f)
    geo = Geometry([Circle((1e-6, 1e-6), 0.3e-6, 4.0)])
    s = SubgridFIT2D(m, geo, "TM", cfl=0.95, lts=2)
    g = SubgridFIT2D(m, geo, "TM", cfl=0.95)
    assert s.dt > 1.6 * g.dt
    assert s.lts_info["rho"] < 1
    rng = np.random.default_rng(0)
    s.u = rng.standard_normal(s.n_dof)
    K = (s.K.T @ sp.diags(s.W) @ s.K).tocsr()
    dt = s.dt

    def modified_energy(y0, y1):
        w, yb = (y1 - y0) / dt, 0.5 * (y1 + y0)
        Kp = lambda z: K @ s.lts.apply(z)  # noqa: E731
        return 0.5 * (w @ (s.D * w) - dt**2 / 4 * w @ Kp(w) + yb @ Kp(yb))

    y0 = s.u.copy()
    s.one_step()
    e0 = modified_energy(y0, s.u)
    for _ in range(20_000):
        y0 = s.u.copy()
        s.one_step()
    assert abs(modified_energy(y0, s.u) / e0 - 1) < 1e-10


@pytest.mark.slow
def test_mie_cylinder_res10():
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parents[1] / "studies"
    sys.path.insert(0, str(here / "mie-cylinder-subgrid"))
    sys.path.insert(0, str(here / "mie-cylinder-mesh"))
    import subgrid_study as SS

    SS.S.T_RUN = 1000e-15

    def rms(band):
        old = SS.build

        def build(*a, **k):
            k["band"] = band
            return old(*a, **k)

        SS.build = build
        try:
            out = SS.run_sg("compact", "TM", 10)
        finally:
            SS.build = old
        q = np.array(out["q"]) / np.array(out["q_ref"]) - 1
        return np.sqrt(np.mean(q**2))

    good = rms(2.0)
    assert good < 0.09
    assert rms(-1e9) > 2 * good  # mutation: no patch (coarse grid only)


@pytest.mark.slow
@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_lorentz_cylinder_ade(pol):
    """Lorentz-dispersive cylinder (ADE on nodes in TM, on edges in TE)."""
    import sys
    from pathlib import Path

    from wakis.dispersive import DispersiveMaterial, Lorentz

    here = Path(__file__).resolve().parents[1] / "studies"
    sys.path.insert(0, str(here / "mie-cylinder-subgrid"))
    sys.path.insert(0, str(here / "mie-cylinder-mesh"))
    import subgrid_study as SS

    SS.S.T_RUN = 1000e-15
    TH = 2 * np.pi * 1e12
    lor = DispersiveMaterial(2.0, [Lorentz(2.5, 900 * TH, 30 * TH)])

    def rms(material):
        out = SS.run_sg("compact", pol, 10, material=material)
        q = np.array(out["q"]) / np.array(out["q_ref"]) - 1
        return np.sqrt(np.mean(q**2))

    good = rms(lor)
    assert good < 0.12
    # mutation: the pole dynamics switched off (eps_inf only) vs the same ref
    no_pole = DispersiveMaterial(2.0, [])
    no_pole.epsilon = lor.epsilon
    assert rms(no_pole) > 3 * good
