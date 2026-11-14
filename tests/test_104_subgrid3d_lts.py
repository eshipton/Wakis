"""
3D subgridding, wave 2: local time stepping, CPML and block/GPU kernels
(`wakis.subgrid3d.SubgridLTSFIT3D`).

Every check has a mutation that was run and seen to fail
(docs/physicsguide_sections/subgrid3d.md, "Wave 2 verification"):

* the fast LTS operator (F_l + buffer only) equals the full-length
  recursion and the sparse matrix to round-off (mutation: wrong buffer
  closed form);
* S_0 = K^H W K R_0 is Hermitian; the stability criterion (Lanczos rho < 1
  and the level positivity margins <= 1) agrees with the dense spectrum on
  both sides of the bisected limit;
* the LTS limit equals the coarse-grid limit up to the damping floor
  min_l sqrt(x_+(p_l, nu_l)) / (2 p_l) (mutation: coarse parents of hanging
  edges left out of F_1 -> below the floor);
* the modified energy is conserved to round-off over 4000 coarse steps at
  0.99 of the LTS limit with nested 2:1 and 3:1 patches, eps/mu/PEC/PMC
  (mutations: 1.03 x the limit; parents left out of F_1);
* the step transpose passes the dot-product test with LTS and CPML
  (mutation: R_0 in place of R_0^T);
* cavity resonances with LTS converge at order 2 (mutation: r(0) = 1.05);
* time-domain probe + DFT = the LTS eigenvalue through the leapfrog
  dispersion;
* CPML: residual energy after the pulse leaves <= 1e-6 with a patch one
  coarse cell from the layer, global and LTS steps (mutation: sigma = 0);
  late-time stable (mutation: a patch inside the layer without a CFS shift
  grows); refusals for patches / LTS fine edges in the layer;
* block kernels = CSR operators to round-off, and runs agree within the
  derived round-off bound (mutation: hanging rows dropped); GPU (CuPy) =
  CPU within the same bound (skipped without CuPy).
"""

import numpy as np
import pytest
import scipy.linalg as la
from scipy.constants import c as C0

from wakis.subgrid3d import Box, SubgridLTS, SubgridLTSFIT3D, SubgridMesh3D
from wakis.subgrid3d.lts import is_stable, lts_dt_max

EPS = np.finfo(float).eps
A_CAV, B_CAV, D_CAV = 1.0, 0.75, 0.5
PATCH = ((0.25, 0.625), (0.25, 0.5), (0.125, 0.375))


def exact_modes(k, a=A_CAV, b=B_CAV, d=D_CAV):
    out = []
    for i in range(10):
        for j in range(10):
            for m in range(10):
                nz = (i > 0) + (j > 0) + (m > 0)
                if nz >= 2:
                    f = C0 / 2 * np.sqrt((i / a) ** 2 + (j / b) ** 2 + (m / d) ** 2)
                    out += [f] * (2 if nz == 3 else 1)
    return np.sort(out)[:k]


def cavity_mesh(N, patch=True, nested=False, ratio=2):
    m = SubgridMesh3D(
        np.linspace(0, A_CAV, N + 1),
        np.linspace(0, B_CAV, 3 * N // 4 + 1),
        np.linspace(0, D_CAV, N // 2 + 1),
    )
    if patch:
        p = m.add_box(PATCH, ratio=ratio)
        if nested:
            q = 1.0 / N
            m.add_box(tuple((lo + q, hi - q) for lo, hi in PATCH), ratio=2, parent=p)
    return m


def closed_mesh():
    """Graded coarse grid, 2:1 patch with a nested 3:1 patch (as test_097)."""
    x = np.array([0, 0.1, 0.2, 0.3, 0.42, 0.55, 0.65, 0.75, 0.85, 1.0])
    m = SubgridMesh3D(x, np.linspace(0, 0.8, 9), np.linspace(0, 0.6, 7))
    p1 = m.add_patch((2, 2, 1), (7, 6, 5), ratio=2)
    m.add_patch((3, 2, 2), (7, 6, 6), ratio=3, parent=p1)
    return m


def closed_materials():
    def eps(X, Y, Z):
        return np.where((X > 0.3) & (X < 0.55) & (Y > 0.1) & (Y < 0.5), 4.0, 1.0)

    def mu(X, Y, Z):
        return np.where((Z > 0.3) & (Y > 0.4), 2.0, 1.0)

    return dict(eps_r=eps, mu_r=mu, pec=Box(((0.42, 0.55), (0.35, 0.45), (0.25, 0.35))),
                pmc=Box(((0.75, 0.85), (0.1, 0.2), (0.0, 0.1))))  # fmt: skip


def damping_floor(lts):
    """
    min_l sqrt(2 delta_l / beta_l) / (2 p_l): the fraction of its own Yee
    limit a level-l mode may use. At ``x = 2 delta / beta`` the argument of
    T_p reaches ``-delta``: r_l changes sign there for even p (positivity)
    and x r(x) reaches 4 for odd p (rho).
    """
    from wakis.unstructured.lts import chebyshev_coefficients

    out = []
    for d in lts.levels[1:]:
        delta, beta, _, _, _ = chebyshev_coefficients(d.p, d.nu)
        out.append(np.sqrt(2 * delta / beta) / (2 * d.p))
    return min(out)


# ------------------------------------------------------------- operator
def test_fast_operator_equals_reference_and_matrix():
    s = SubgridLTSFIT3D(closed_mesh(), lts=False, **closed_materials())
    L = SubgridLTS(s, 2.5 * s.dt_max)
    assert L.L == 2 and L.ratios == [2, 3]
    rng = np.random.default_rng(0)
    e = rng.standard_normal(s.n_free)
    fast, full = L.apply(e), L.apply_full(e)
    sc = np.abs(full).max()
    assert np.abs(fast - full).max() / sc < 1e-13
    assert np.abs(L.matrix() @ e - full).max() / sc < 1e-13
    # mutation: wrong closed form on the buffer (kappa_k y_C)
    d = L.levels[1]
    d.kappa = 1.1 * d.kappa
    assert np.abs(L.apply(e) - full).max() / sc > 1e-6  # measured 7e-4


def test_stiffness_hermitian_and_criterion_matches_dense():
    s = SubgridLTSFIT3D(cavity_mesh(8), lts=False)
    dt_max, _ = lts_dt_max(s, 2.2 * s.dt_max, rtol=1e-3)
    for f, stable in ((0.99, True), (1.02, False)):
        L = SubgridLTS(s, f * dt_max)
        S0, D = L.stiffness()
        A = S0.toarray()
        assert np.abs(A - A.T).max() < 1e-13 * np.abs(A).max()
        sc = 1 / np.sqrt(D)
        lam = la.eigvalsh(sc[:, None] * (0.5 * (A + A.T)) * sc[None, :])
        dense_ok = lam.min() > -1e-12 * lam.max() and L.dt / 2 * np.sqrt(lam.max()) < 1
        ok, rho, marg = is_stable(L)
        assert ok == stable and dense_ok == stable
        assert abs(rho - L.dt / 2 * np.sqrt(lam.max())) < 1e-8


@pytest.mark.parametrize("nested", [False, True])
def test_lts_limit_equals_coarse_grid_limit(nested):
    s0 = SubgridLTSFIT3D(cavity_mesh(16, patch=False), lts=False)
    s = SubgridLTSFIT3D(cavity_mesh(16, nested=nested), lts=True)
    ratio = s.lts_info["dt_max"] / s0.dt_max
    floor = damping_floor(s.lts) * (1 - 2e-3)
    assert floor <= ratio <= 1.05, (ratio, floor)
    assert s.lts_info["dt_max"] / s.dt_max > (3.8 if nested else 1.9)
    # mutation: parents of hanging edges outside F_1 -> limit below the floor
    if not nested:
        dt_mut, _ = lts_dt_max(s, 2.2 * s.dt_max, interface_parents=False)
        assert dt_mut / s0.dt_max < floor


# ---------------------------------------------------------------- energy
def test_modified_energy_conserved_nested_2_3():
    s = SubgridLTSFIT3D(closed_mesh(), lts=True, cfl=0.99, **closed_materials())
    assert s.lts.ratios == [2, 3]
    s.e[:] = np.random.default_rng(1).standard_normal(s.n_free)
    E = s.run(4000, energy=True)
    drift = np.abs(E / E[0] - 1).max()
    assert drift < 50 * EPS * np.sqrt(len(E)), drift
    dt_ok = s.dt
    # mutation 1: 3 % above the limit
    m = SubgridLTSFIT3D(closed_mesh(), lts=True, dt=1.03 * s.lts_info["dt_max"],
                        **closed_materials())  # fmt: skip
    m.e[:] = np.random.default_rng(1).standard_normal(m.n_free)
    with np.errstate(all="ignore"):
        Em = m.run(4000, energy=True)
    assert not np.all(np.isfinite(Em)) or np.abs(Em / Em[0] - 1).max() > 1e-3
    # mutation 2: coarse parents of hanging edges left out of F_1
    m = SubgridLTSFIT3D(closed_mesh(), lts=True, dt=dt_ok,
                        lts_kw=dict(interface_parents=False), **closed_materials())  # fmt: skip
    m.e[:] = np.random.default_rng(1).standard_normal(m.n_free)
    with np.errstate(all="ignore"):
        m.run(4000)
    assert not np.all(np.isfinite(m.e)) or np.abs(m.e).max() > 1e3


# ------------------------------------------------------------- transpose
def _dot(s, mutate=None):
    """Relative dot-product defect of `step_vec_T`; the b part of the random
    vectors is scaled by sqrt(D/W) so that b and e weigh alike."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(s.n_state)
    y = rng.standard_normal(s.n_state)
    i = rng.standard_normal(s.mesh.n_edges)
    W = np.asarray(s._Wb)
    sb = np.sqrt(np.median(s.D) / np.median(W[W > 0]))
    x[: s._nb] *= sb
    y[: s._nb] *= sb
    fx = s.step_vec(x, i)
    if mutate == "R":  # R_0 in place of R_0^T
        s.lts.apply_T = s.lts.apply
    if mutate == "pml":  # CPML coefficient off by 10 % in the transpose only
        for p in s._pml:
            p["ah"] = 1.1 * p["ah"]
    xb, ib = s.step_vec_T(y)
    scale = np.linalg.norm(fx) * np.linalg.norm(y)
    return abs(fx @ y - (x @ xb + i @ ib)) / scale


def _open_box(patch=True, lts=True, kernels="csr", pml=True, R0=1e-6, **kw):
    H, n = 0.04, 24
    x = np.arange(n + 1) * H
    m = SubgridMesh3D(x, x, x)
    if patch:
        m.add_box(((0.4, 0.68), (0.4, 0.6), (0.4, 0.6)), ratio=2)
    th = [(6 * H, 6 * H)] * 3
    return SubgridLTSFIT3D(m, lts=lts, kernels=kernels, dt_rtol=1e-2,
                           pml=dict(thickness=th, R0=R0) if pml else None, **kw)  # fmt: skip


def test_dot_product_lts_cpml():
    s = SubgridLTSFIT3D(closed_mesh(), lts=True, dt=1.0e-10, **closed_materials())
    assert _dot(s) < 1e-15
    s = _open_box(dt=7.0e-11)
    assert s.lts is not None and s.cpml is not None
    assert _dot(s) < 1e-15
    s = _open_box(dt=7.0e-11, kernels="block")
    assert _dot(s) < 1e-15
    # mutations (measured ~1e-7 against ~1e-18)
    assert _dot(_open_box(dt=7.0e-11), mutate="R") > 1e-10
    assert _dot(_open_box(dt=7.0e-11), mutate="pml") > 1e-10


# ------------------------------------------------------------ convergence
def _lts_error(N, mutate=None):
    s = SubgridLTSFIT3D(cavity_mesh(N), lts=True, cfl=0.95, dt_rtol=1e-2)
    if mutate:
        s.lts = SubgridLTS(s, s.dt, mutate=mutate)
    ex = exact_modes(6)
    return np.abs(s.lts_eigenfrequencies(6, ex[0]) / ex - 1).max()


def test_lts_cavity_converges_second_order():
    e8, e16 = _lts_error(8), _lts_error(16)
    order = np.log2(e8 / e16)
    assert 1.8 <= order <= 2.2, (e8, e16, order)
    m8, m16 = _lts_error(8, "consistency"), _lts_error(16, "consistency")
    assert np.log2(m8 / m16) < 1.0 and m16 > 5 * e16


def test_time_domain_matches_lts_eigenvalue():
    s = SubgridLTSFIT3D(cavity_mesh(8), lts=True, cfl=0.95, dt_rtol=1e-2)
    f1 = s.lts_eigenfrequencies(1, exact_modes(1)[0])[0]
    tau = 2.0 / f1
    s.add_current((0.41, 0.33, 0.27), 2,
                  lambda t: np.exp(-(((t - 4 * tau) / tau) ** 2)) * np.sin(2 * np.pi * f1 * t))  # fmt: skip
    s.add_probe((0.37, 0.41, 0.23), 2)
    n = int(np.ceil(300 / f1 / s.dt))
    s.run(n)
    v = np.array(s.probes[0]["values"])
    pad = 16 * len(v)
    X = np.abs(np.fft.rfft(np.hanning(len(v)) * v, n=pad))
    f = np.fft.rfftfreq(pad, s.dt)
    sel = np.flatnonzero(np.abs(f / f1 - 1) < 0.1)
    i = sel[np.argmax(X[sel])]
    y0, y1, y2 = X[i - 1], X[i], X[i + 1]
    fp = f[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (f[1] - f[0])
    assert abs(fp / f1 - 1) < 1e-5


# ------------------------------------------------------------------ CPML
def _residual(s, t_end=40e-9):
    f0 = 1.0e9
    tau = 0.5 / f0
    s.add_current((0.48, 0.5, 0.5), 2,
                  lambda t: -(t - 4 * tau) / tau * np.exp(-(((t - 4 * tau) / tau) ** 2)))  # fmt: skip
    Emax = 0.0
    for k in range(int(t_end / s.dt)):
        s.one_step()
        if k % 10 == 0:
            e, b = s.e_host(), s.b_faces()
            E = 0.5 * np.sum(b * s.W * b) + 0.5 * np.sum(e * s.D * e)
            Emax = max(Emax, E)
    return E / Emax


@pytest.mark.parametrize("lts", [False, True])
def test_cpml_absorbs_with_patch_near_layer(lts):
    s = _open_box(lts=lts)
    assert (s.lts is not None) == lts
    r = _residual(s)
    assert r < 1e-6, r
    # mutation: sigma = 0 (R0 = 1)
    assert _residual(_open_box(lts=lts, R0=1.0)) > 1e-3


def test_cpml_late_time_stability_and_refusals():
    s = _open_box(lts=False)
    rng = np.random.default_rng(4)
    s.e[:] = rng.standard_normal(s.n_free)
    a0 = np.abs(s.e).max()
    s.run(6000)
    assert np.abs(s.e).max() <= a0
    # refusals
    H, n = 0.04, 24
    x = np.arange(n + 1) * H
    m = SubgridMesh3D(x, x, x)
    m.add_box(((0.4, 0.88), (0.4, 0.6), (0.4, 0.6)), ratio=2)
    th = [(0, 6 * H), (0, 0), (0, 0)]
    with pytest.raises(ValueError, match="CPML"):
        SubgridLTSFIT3D(m, lts=False, pml=dict(thickness=th))
    with pytest.raises(ValueError, match="CPML"):
        SubgridLTSFIT3D(m, lts=True, dt=7e-11, pml=dict(thickness=th, allow_patches=True))
    # mutation (documented failure mode): patch inside the layer, no CFS shift
    s = SubgridLTSFIT3D(m, lts=False, pml=dict(thickness=th, allow_patches=True))
    s.e[:] = rng.standard_normal(s.n_free)
    a0 = np.abs(s.e).max()
    with np.errstate(all="ignore"):
        s.run(1500)
    assert not np.isfinite(np.abs(s.e).max()) or np.abs(s.e).max() > 1e3 * a0


# ---------------------------------------------------------- block / GPU
def _roundoff_bound(s, n_steps):
    """n u (m_K + m_KH) kappa_E, kappa_E = sqrt((1 + rho)/(1 - rho))."""
    mK = np.diff(s.K.indptr).max()
    mKH = np.diff(s.KH.tocsr().indptr).max()
    rho = min(s.stability_ratio(), 0.999)
    return n_steps * EPS / 2 * (mK + mKH) * np.sqrt((1 + rho) / (1 - rho))


def _rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def test_block_kernels_equal_csr():
    kw = closed_materials()
    sc = SubgridLTSFIT3D(closed_mesh(), lts=True, cfl=0.95, kernels="csr", **kw)
    sb = SubgridLTSFIT3D(closed_mesh(), lts=True, dt=sc.dt, kernels="block", **kw)
    rng = np.random.default_rng(5)
    e = rng.standard_normal(sc.n_free)
    assert _rel(sb.ops.K(e, np.empty(sb.mesh.n_faces))[sb.faces], sc.K @ e) < 4 * EPS
    y = rng.standard_normal(sc.n_faces)
    assert _rel(sb.ops.KH(sb.ops.from_faces(y)), sc.KH @ y) < 8 * EPS
    n = 500
    for s in (sc, sb):
        s.e[:] = e
        s.run(n)
    bound = _roundoff_bound(sc, n)
    err = _rel(sb.e, sc.e)
    assert err < bound, (err, bound)
    # mutation: block kernels without the hanging rows of P_e
    sb.reset()
    sb.e[:] = e
    sb.ops.hang = sb.ops.hang[:0]
    sb.run(10)
    sc.reset()
    sc.e[:] = e
    sc.run(10)
    assert _rel(sb.e, sc.e) > 1e-3


def test_gpu_equals_cpu():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    kw = closed_materials()
    sc = SubgridLTSFIT3D(closed_mesh(), lts=True, cfl=0.95, kernels="csr", **kw)
    sg = SubgridLTSFIT3D(closed_mesh(), lts=True, dt=sc.dt, kernels="block",
                         backend="cupy", **kw)  # fmt: skip
    e = np.random.default_rng(6).standard_normal(sc.n_free)
    n = 500
    sc.e[:] = e
    sg.e[:] = cp.asarray(e)
    sc.run(n)
    sg.run(n)
    err = _rel(sg.e_host(), sc.e)
    bound = _roundoff_bound(sc, n)
    assert err < bound, (err, bound)
    E0 = sg.modified_energy(sg.e, sg.e)  # smoke: energy on device
    assert np.isfinite(E0)
