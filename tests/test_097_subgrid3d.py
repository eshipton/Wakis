"""
3D Cartesian subgridding for FIT: `wakis.subgrid3d` (milestone 1, CPU).

Every check comes with a mutation that was run and seen to fail
(docs/physicsguide_sections/subgrid3d.md, "Verification"):

* without a patch, the operator is SolverFIT3D's (graded grid, eps, mu): the
  nonzero spectra agree to 1e-12;
* discrete energy of a closed cavity with two nested patches (2:1 then 3:1,
  graded coarse grid, eps/mu boxes crossing interfaces, PEC and PMC cells)
  is conserved to round-off over 1e4 steps at 0.99 of the Lanczos limit;
  mutations: P_e^T replaced by injection drifts at 0.5 of the limit; 2 %
  above the limit blows up. The Lanczos limit obeys the derived bound
  lambda_max <= max over blocks (natural boundaries) and the finest-cell
  Yee bound;
* div b = 0 in every leaf cell and coarse-face flux = sum of its fine
  sub-face fluxes to round-off (mutation: wrong P_e fraction breaks the
  flux balance; the missing transverse term does not, as derived);
* cavity eigenfrequencies with an off-centre 2:1 or 3:1 patch converge with
  order 2 (mutations: wrong fraction -> order < 1; missing transverse term
  -> O(1) errors);
* plane-wave reflection: normal incidence equals the exact 1D interface
  formula to the fit floor; s-polarisation equals the 2D TM result to
  1e-9; |R| falls as N^-2 at 0/30/60 deg for s and p, 2:1 and 3:1
  (mutations: wrong fraction, missing transverse term -> no N^-2);
* the one-step transpose passes the dot-product test to 1e-14 and the
  adjoint gradient matches central finite differences (mutation: a source
  transpose without P_e fails);
* time-domain probe + DFT resonance = the semi-discrete eigenfrequency
  mapped through the leapfrog dispersion, to 1e-5.
"""

import numpy as np
import pytest
import scipy.linalg as la
from scipy.constants import c as C0

from wakis.subgrid3d import Box, SubgridFIT3D, SubgridMesh3D
from wakis.subgrid3d.interface import bloch_reflection_3d, reflection_1d_exact

A_CAV, B_CAV, D_CAV = 1.0, 0.75, 0.5
PATCH = ((0.25, 0.625), (0.25, 0.5), (0.125, 0.375))


def exact_modes(k, a=A_CAV, b=B_CAV, d=D_CAV):
    """Lowest k PEC-box resonances [Hz] with multiplicity (TE + TM)."""
    out = []
    for i in range(10):
        for j in range(10):
            for m in range(10):
                nz = (i > 0) + (j > 0) + (m > 0)
                if nz >= 2:
                    f = C0 / 2 * np.sqrt((i / a) ** 2 + (j / b) ** 2 + (m / d) ** 2)
                    out += [f] * (2 if nz == 3 else 1)
    return np.sort(out)[:k]


def cavity_mesh(N, ratio=2, patch=True, nested=False):
    m = SubgridMesh3D(
        np.linspace(0, A_CAV, N + 1),
        np.linspace(0, B_CAV, 3 * N // 4 + 1),
        np.linspace(0, D_CAV, N // 2 + 1),
    )
    if patch:
        p = m.add_box(PATCH, ratio=ratio)
        if nested:
            q = 1.0 / N  # one coarse cell inside the first patch
            inner = tuple((lo + q, hi - q) for lo, hi in PATCH)
            m.add_box(inner, ratio=2, parent=p)
    return m


def cavity_error(N, ratio=2, mutate=None, patch=True, nested=False, k=6):
    s = SubgridFIT3D(cavity_mesh(N, ratio, patch, nested), mutate=mutate, dt=1.0)
    ex = exact_modes(k)
    return np.abs(s.eigenfrequencies(k, ex[0]) / ex - 1).max()


def closed_solver(mutate=None, cfl=0.99, dt=None):
    """Graded coarse grid, 2:1 patch with a nested 3:1 patch, eps/mu boxes
    crossing the interfaces, a PEC cube and a PMC cell block."""
    x = np.array([0, 0.1, 0.2, 0.3, 0.42, 0.55, 0.65, 0.75, 0.85, 1.0])
    y = np.linspace(0, 0.8, 9)
    z = np.linspace(0, 0.6, 7)
    m = SubgridMesh3D(x, y, z)
    p1 = m.add_patch((2, 2, 1), (7, 6, 5), ratio=2)
    m.add_patch((3, 2, 2), (7, 6, 6), ratio=3, parent=p1)

    def eps(X, Y, Z):
        return np.where((X > 0.3) & (X < 0.55) & (Y > 0.1) & (Y < 0.5), 4.0, 1.0)

    def mu(X, Y, Z):
        return np.where((Z > 0.3) & (Y > 0.4), 2.0, 1.0)

    pec = Box(((0.42, 0.55), (0.35, 0.45), (0.25, 0.35)))
    pmc = Box(((0.75, 0.85), (0.1, 0.2), (0.0, 0.1)))
    return SubgridFIT3D(
        m, eps_r=eps, mu_r=mu, pec=pec, pmc=pmc, cfl=cfl, dt=dt, mutate=mutate
    )


def energy_run(s, steps, seed=0):
    rng = np.random.default_rng(seed)
    s.reset()
    s.e[:] = rng.standard_normal(s.n_free)
    E = s.run(steps, energy=True)
    return E


# --------------------------------------------------------------- reuse check
def test_without_patch_equals_solverfit3d():
    from wakis import GridFIT3D, SolverFIT3D

    x = np.array([0, 0.1, 0.25, 0.35, 0.5, 0.6])
    y = np.array([0, 0.12, 0.2, 0.33, 0.4])
    z = np.array([0, 0.1, 0.22, 0.3])
    g = GridFIT3D(x=x, y=y, z=z, verbose=0)
    sv = SolverFIT3D(
        g, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0, bg=[2.0, 1.5, 0.0]
    )
    A = (sv.itDaiDepsDstC @ sv.tDsiDmuiDaC).toarray()
    w_ref = np.sort(la.eigvals(A).real)
    w_ref = w_ref[w_ref > 1e-8 * w_ref.max()]
    s = SubgridFIT3D(SubgridMesh3D(g), eps_r=2.0, mu_r=1.5, dt=1.0)
    K, D = s.stiffness()
    w = la.eigh(K.toarray(), np.diag(D), eigvals_only=True)
    w = w[w > 1e-8 * w.max()]
    assert len(w) == len(w_ref)
    # measured 8e-15; dense eigensolver round-off
    assert np.abs(w / w_ref - 1).max() < 1e-12


# ------------------------------------------------------ energy and stability
def test_energy_conserved_nested_patches():
    s = closed_solver()
    assert s.mesh.n_blocks == 3
    E = energy_run(s, 10_000)
    drift = np.abs(E / E[0] - 1).max()
    # measured 4e-16; bound: random walk of O(eps) per step, 50 eps sqrt(n)
    assert drift < 50 * np.finfo(float).eps * np.sqrt(len(E)), drift


def test_cfl_bound_and_blowup():
    s = closed_solver()
    lam = s.lambda_max
    bounds = s.block_lambda_bounds()
    # theorem: lambda_max(composite) <= max_b lambda_max(block b, natural BC)
    assert lam <= bounds.max() * (1 + 1e-10)
    # ... <= the Yee bound c^2 sum 4/h^2 of the finest vacuum cells
    hf = [np.min(h) for h in s.mesh.blocks[-1].h]
    assert lam <= C0**2 * sum(4 / h**2 for h in hf) * (1 + 1e-10)
    # stable just below the limit (energy test), unstable 2 % above
    dtm = s.dt_max
    s_hi = closed_solver(dt=1.02 * dtm)
    with np.errstate(all="ignore"):
        E = energy_run(s_hi, 1500)
    assert not np.isfinite(E[-1]) or abs(E[-1]) > 1e6 * abs(E[0])


def test_mutation_injection_breaks_energy():
    s = closed_solver()
    s_inj = closed_solver(mutate="inject", dt=0.5 * s.dt_max)
    E = energy_run(s_inj, 4000)
    # correct operator: < 1e-13 (test above); injection drifts by O(1)
    assert np.abs(E / E[0] - 1).max() > 0.1


# --------------------------------------------------------------- divergence
def _driven(mutate=None):
    s = closed_solver(mutate=mutate)
    rng = np.random.default_rng(3)
    s.e[:] = rng.standard_normal(s.n_free)
    # a source on a hanging edge region (inside the 3:1 patch surface)
    s.add_current((0.36, 0.4, 0.3), 2, lambda t: np.sin(2 * np.pi * 3e8 * t))
    s.run(300)
    return s


def test_divergence_free_and_flux_balance():
    s = _driven()
    bmax = np.abs(s.b).max()
    assert np.abs(s.divergence()).max() < 1e-13 * bmax
    assert np.abs(s.interface_flux_mismatch()).max() < 1e-13 * bmax


def test_mutations_flux_balance():
    s = _driven("fraction")
    assert np.abs(s.interface_flux_mismatch()).max() > 1e-2 * np.abs(s.b).max()
    assert np.abs(s.divergence()).max() < 1e-13 * np.abs(s.b).max()  # per cell still 0
    # derived: face-interior fine edges cancel in the sum over sub-faces, so the
    # missing transverse term leaves the flux balance intact (other tests catch it)
    s = _driven("missing_transverse")
    assert np.abs(s.interface_flux_mismatch()).max() < 1e-13 * np.abs(s.b).max()


# ------------------------------------------------------------ cavity modes
@pytest.mark.parametrize("ratio", [2, 3])
def test_cavity_second_order(ratio):
    e8, e16 = cavity_error(8, ratio), cavity_error(16, ratio)
    order = np.log2(e8 / e16)
    assert 1.8 < order < 2.2, order  # measured 1.98 (2:1 and 3:1)


@pytest.mark.parametrize("ratio", [2, 3])
def test_cavity_mutations(ratio):
    good = cavity_error(16, ratio)
    f8, f16 = cavity_error(8, ratio, "fraction"), cavity_error(16, ratio, "fraction")
    assert np.log2(f8 / f16) < 1.0  # measured 0.40
    assert f16 > 5 * good
    assert cavity_error(16, ratio, "missing_transverse") > 0.1  # measured 0.53-0.58


@pytest.mark.slow
@pytest.mark.parametrize("ratio,nested,Ns", [(2, True, (16, 24, 32)), (3, False, (8, 16, 32)),
                                             (3, True, (16, 24))])  # fmt: skip
def test_cavity_second_order_fine(ratio, nested, Ns):
    # the nested box sits one coarse cell inside the first patch: N >= 16
    e = [cavity_error(N, ratio, nested=nested) for N in Ns]
    orders = np.log(np.array(e[:-1]) / np.array(e[1:])) / np.log(
        np.array(Ns[1:]) / np.array(Ns[:-1])
    )
    assert np.all((orders > 1.8) & (orders < 2.2)), orders


# ------------------------------------------------------------- reflection
@pytest.mark.parametrize("ratio", [2, 3])
@pytest.mark.parametrize("pol", ["s", "p"])
def test_normal_incidence_equals_1d_exact(ratio, pol):
    floor = bloch_reflection_3d(10, 0, ratio, pol, refine=False)
    for N in (10, 20):
        R = bloch_reflection_3d(N, 0, ratio, pol)
        assert abs(R - reflection_1d_exact(N, ratio)) < floor, (N, R)
    # mutation: missing transverse term (normal incidence still sees it)
    Rm = bloch_reflection_3d(10, 0, ratio, pol, mutate="missing_transverse")
    assert Rm > 10 * reflection_1d_exact(10, ratio)


@pytest.mark.parametrize("theta", [30, 60])
def test_s_polarisation_equals_2d_tm(theta):
    from wakis.subgrid.interface import bloch_reflection

    r3 = bloch_reflection_3d(10, theta, 2, "s", bloch_source=False)
    r2 = bloch_reflection(10, theta, 2, "TM")
    assert abs(r3 / r2 - 1) < 1e-9  # measured 1.6e-12


@pytest.mark.parametrize("ratio", [2, 3])
@pytest.mark.parametrize("pol", ["s", "p"])
@pytest.mark.parametrize("theta", [0, 30, 60])
def test_reflection_second_order(ratio, pol, theta):
    r10, r20 = (
        bloch_reflection_3d(10, theta, ratio, pol),
        bloch_reflection_3d(20, theta, ratio, pol),
    )
    assert 1.8 < np.log2(r10 / r20) < 2.25, (r10, r20)


@pytest.mark.parametrize("mutate", ["fraction", "missing_transverse"])
def test_reflection_mutations(mutate):
    r10 = bloch_reflection_3d(10, 60, 2, "s", mutate=mutate)
    r20 = bloch_reflection_3d(20, 60, 2, "s", mutate=mutate)
    good = bloch_reflection_3d(20, 60, 2, "s")
    assert np.log2(r10 / r20) < 1.0  # no convergence
    assert r20 > 10 * good


@pytest.mark.slow
@pytest.mark.parametrize("ratio", [2, 3])
@pytest.mark.parametrize("pol", ["s", "p"])
@pytest.mark.parametrize("theta", [0, 30, 60])
def test_reflection_second_order_fine(ratio, pol, theta):
    r = [bloch_reflection_3d(N, theta, ratio, pol) for N in (20, 40)]
    assert 1.9 < np.log2(r[0] / r[1]) < 2.1, r


# ---------------------------------------------------------------- adjoint
def _random_state(s, rng):
    return (rng.standard_normal(s.n_faces), rng.standard_normal(s.n_free),
            rng.standard_normal(s.mesh.n_edges))  # fmt: skip


def test_dot_product_transpose():
    s = closed_solver()
    rng = np.random.default_rng(7)
    x = _random_state(s, rng)
    y = _random_state(s, rng)[:2]
    fx = s.linear_step(*x)
    gy = s.linear_step_T(*y)
    lhs = fx[0] @ y[0] + fx[1] @ y[1]
    rhs = x[0] @ gy[0] + x[1] @ gy[1] + x[2] @ gy[2]
    assert abs(lhs - rhs) < 1e-14 * abs(lhs)  # measured ~1e-16
    # source part alone (its scale is far below the field part above), and the
    # mutation: a source transpose that ignores the hanging rows of P_e
    i_src = np.zeros(s.mesh.n_edges)
    i_src[s.hanging] = rng.standard_normal(s.hanging.sum())
    fs = s.linear_step(0 * x[0], 0 * x[1], i_src)
    lhs = fs[0] @ y[0] + fs[1] @ y[1]
    assert abs(lhs - i_src @ gy[2]) < 1e-14 * abs(lhs)
    ib_bad = gy[2].copy()
    ib_bad[s.hanging] = 0.0
    assert abs(lhs - i_src @ ib_bad) > 0.5 * abs(lhs)


def test_adjoint_gradient_matches_finite_differences():
    s = closed_solver()
    rng = np.random.default_rng(11)
    # initial fields small against the driven ones, so the central difference
    # of the quadratic J does not cancel catastrophically
    b0 = 1e-12 * rng.standard_normal(s.n_faces)
    e0 = 1e-3 * rng.standard_normal(s.n_free)
    shape = np.zeros(s.mesh.n_edges)
    k = s.locate_edge((0.36, 0.4, 0.3), 2)
    assert s.hanging[k]
    shape[k] = 1.0
    nst = 60
    wave = np.sin(0.3 * np.arange(nst))

    def J(alpha):
        b, e = b0, e0
        for n in range(nst):
            b, e = s.linear_step(b, e, alpha * wave[n] * shape)
        return 0.5 * e @ e, (b, e)

    j0, (bN, eN) = J(0.7)
    bb, eb = np.zeros_like(bN), eN.copy()
    grad = 0.0
    for n in reversed(range(nst)):
        bb, eb, ib = s.linear_step_T(bb, eb)
        grad += wave[n] * (ib @ shape)
    d = 1e-3
    fd = (J(0.7 + d)[0] - J(0.7 - d)[0]) / (2 * d)  # exact for a quadratic J
    assert abs(grad / fd - 1) < 1e-8


# ------------------------------------------------------- time-domain run
def test_time_domain_resonance_probe_dft():
    s = SubgridFIT3D(cavity_mesh(8, 2), cfl=0.9)
    fe = s.eigenfrequencies(1, exact_modes(1)[0])[0]
    f0, tau = 3.0e8, 4 / 3.0e8
    s.add_current((0.31, 0.36, 0.27), 2,
                  lambda t: np.exp(-(((t - 3 * tau) / tau) ** 2)) * np.sin(2 * np.pi * f0 * t))  # fmt: skip
    p = s.add_probe((0.4, 0.33, 0.2), 2, freqs=[fe, 1.1 * fe])
    nst = 20_000
    s.run(nst)
    v = np.array(s.probes[p]["values"])
    t = np.arange(len(v)) * s.dt
    # built-in running DFT = rectangular DFT of the record
    ref = (np.exp(-2j * np.pi * np.outer([fe, 1.1 * fe], t)) @ v) * s.dt
    assert np.allclose(s.probes[p]["dft"], ref, rtol=1e-10, atol=0)
    # leapfrog maps w_semi to w_td with 2 sin(w_td dt/2)/dt = w_semi
    ftd = np.arcsin(np.pi * fe * s.dt) * 2 / s.dt / (2 * np.pi)
    w = np.hanning(len(v))
    fs = ftd * (1 + np.linspace(-0.005, 0.005, 1001))
    X = np.abs(np.exp(-2j * np.pi * np.outer(fs, t)) @ (w * v))
    i = int(np.argmax(X))
    y0, y1, y2 = X[i - 1], X[i], X[i + 1]
    fpk = fs[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (fs[1] - fs[0])
    # Hann + parabolic peak of an isolated line: bias << 1 % of a bin
    # (bin = 1.7e-3 f); measured 2.6e-7
    assert abs(fpk / ftd - 1) < 1e-5
