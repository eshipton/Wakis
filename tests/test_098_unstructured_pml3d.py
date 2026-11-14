"""
Spherical CFS-PML of `UnstructuredFIT3D` (`wakis.unstructured.pml3d`).

Setup: tetrahedral ball (radius 4 h) + radially extruded shell of frustum
prisms (`spherical_shell_mesh`); a derivative-of-Gaussian point dipole
(centre wavelength 12 h). The reference run has the *same* ball and the same
first layers, continued by 24 vacuum layers before its own PML, so no
reference echo reaches a probe in the window; the probe-field difference
test - reference is the reflection of the test boundary alone (the interior
discretisation error cancels exactly). All runs share one dt.

* Mesh: shell volume equals the exact polyhedral frustum volume; diagonals
  have zero dual area, inner prism faces zero dual length, every coupling
  dof in the shell is radial or tangential.
* Identity filters (sigma = 0, kappa = 1) reproduce the plain solver step to
  round-off; a dropped massless term would not.
* Filter: the cascade's discrete response equals the bilinear-transform
  value of s_n / s_d (exact in the discrete sense, ~1e-12).
* Transpose: <S x, y> = <x, S^T y> (~1e-15); the reversed stage order fails.
* Massless constraint C_s^T h = 0 holds with the stretched h (round-off);
  the unstretched elimination (mutation) violates it.
* Reflection (normal incidence and oblique, up to ~57 deg): 10 layers below
  -40 dB (target) and > 40 dB below the bare PEC wall; the first-order
  Silver-Mueller boundary on the same ball (named baseline) >= 10 x worse.
* Layers: the reflection falls with the layer count N at fixed dr.
* Mutations that must fail the -40 dB target: negative sigma, radial /
  tangential filters swapped, the auxiliary (convolution) term of the
  denominators dropped. Omitting the tangential stretch s_t (planar stretch
  on a sphere) stays below -40 dB here (-47 dB) but must reflect > 10x more
  than the correct layer (measured 35-40x).
* Energy: after the source is off, the ball energy (leapfrog-conserved form)
  falls below 1e-8 of its peak and never rises by more than R^2 = 1e-4 of the
  peak (the energy a -40 dB reflection can return); in a PEC ball the same
  energy is conserved to round-off (control of the energy form).
* Floor: the ~1e-4 reflection reported against the vacuum-shell
  reference is that reference's error (a 20-layer PML baseline halves it;
  the reference differs from that baseline by the floor itself); the
  backing-wall echo (E_PEC - E_PMC) / 2 is < 1e-5 (mutation R = 1e-3: > 1e-5).
* Long run (slow): 1e5 steps with CFS alpha > 0, no late-time growth (the
  late residual stays < 1e-6 of the peak); mutation alpha < 0 must fail it.
"""

import functools

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.constants import c as C0

from wakis.unstructured import UnstructuredFIT3D, ball_mesh
from wakis.unstructured.pml3d import (
    _Cascade,
    _Layout,
    _stage,
    _sel,
    ball_with_pml_mesh,
    bilinear_response,
)

H = 1.0
RB = 4.0
LAM0 = 12.0
TAU = LAM0 / (np.pi * C0)
T0 = 4 * TAU
N_EXTRA = 24
N_REF = 6
TARGET = 1e-2  # -40 dB


def amp(t):
    x = (t - T0) / TAU
    return -x * np.exp(-(x**2))


def _geometry(kind):
    if kind == "normal":
        th = np.radians([90, 45])
        rp = RB - 1.0
        src = np.array([0.31, 0.17, 0.23])  # off-node (see the study README)
        return src, rp * np.stack([np.sin(th), 0 * th, np.cos(th)], 1)
    rs = RB - 0.7  # radial dipole just below the PML surface
    ps = np.radians([40, 70, 110])
    return np.array([0, 0, rs]), rs * np.stack([np.sin(ps), 0 * ps, np.cos(ps)], 1)


def _window(kind):
    r1 = RB + 10 * H
    path = 2 * r1 - (RB - 1.0) if kind == "normal" else 2 * (r1 - RB + 0.7) + 2 * RB
    return 2 * T0 + path / C0


@functools.lru_cache(None)
def _mesh(n_extra, n_layers):
    return ball_with_pml_mesh(RB, H, n_layers, n_extra=n_extra)


@functools.lru_cache(None)
def _dt():
    m, _ = _mesh(N_EXTRA, N_REF)
    d1 = UnstructuredFIT3D(m, bc="pec").dt_max
    d2 = UnstructuredFIT3D(ball_mesh([RB], H), bc="abc").dt_max
    return 0.9 * min(d1, d2)


def _solver(label, n_layers=10, **opts):
    dt = _dt()
    if label in ("abc", "pec"):
        return UnstructuredFIT3D(ball_mesh([RB], H), bc=label, dt=dt)
    m, cfg = _mesh(N_EXTRA if label == "ref" else 0, n_layers)
    return UnstructuredFIT3D(m, bc="pec", dt=dt, pml=dict(cfg, **opts))


def _probe_matrix(mesh, points):
    """Whitney interpolation of E (rows 3 i + c) from *all* edge voltages."""
    cell, W = mesh.whitney_edge(points)
    rows = np.repeat(3 * np.arange(len(cell)), 6)
    cols = mesh.cell_edges[cell].ravel()
    shape = (3 * len(cell), mesh.n_edges)
    return sum(
        sp.csr_matrix((W[:, :, c].ravel(), (rows + c, cols)), shape=shape) for c in range(3)
    ).tocsr()


def _signals(s, kind):
    src, probes = _geometry(kind)
    s.add_point_dipole(src, [0, 0, 1], amp)
    P = _probe_matrix(s.mesh, probes)
    volt = s._pml.edge_voltages if s._pml is not None else (lambda: s.e)
    out = []
    for _ in range(int(round(_window(kind) / s.dt))):
        s.one_step()
        if not np.isfinite(s.u[:10]).all():
            return np.full((1, len(probes), 3), np.inf)
        out.append((P @ volt()).reshape(len(probes), 3))
    return np.array(out)


@functools.lru_cache(None)
def _ref(kind):
    return _signals(_solver("ref", N_REF), kind)


def _reflection(kind, label, n_layers=10, **opts):
    sig = _signals(_solver(label, n_layers, **opts), kind)
    ref = _ref(kind)
    if not np.isfinite(sig).all() or sig.shape != ref.shape:
        return np.full(ref.shape[1], np.inf)
    num = np.linalg.norm(sig - ref, axis=-1).max(0)
    return num / np.linalg.norm(ref, axis=-1).max(0)


# ------------------------------------------------------------------ mesh
def test_shell_mesh_geometry():
    nl = 3
    m, cfg = _mesh(0, nl)
    ball = ball_mesh([RB], H)
    assert np.array_equal(m.nodes[: ball.n_nodes], ball.nodes)
    # Exact volume of the polyhedral shell: cones over the boundary triangles
    F = ball.nodes[ball.faces[ball.boundary_faces]]
    v0 = np.abs(np.einsum("ij,ij->i", F[:, 0], np.cross(F[:, 1], F[:, 2]))).sum() / 6
    v_shell = v0 * ((cfg["r1"] / RB) ** 3 - 1)
    shell = np.arange(ball.n_cells, m.n_cells)
    assert m.cell_volume[shell].min() > 0
    assert m.cell_volume[shell].sum() == pytest.approx(v_shell, rel=1e-12)
    # Co-spherical prisms: the 3 tets of a prism share one circumcentre
    cc = m.cell_circumcenter[shell].reshape(-1, 3, 3)
    assert np.abs(cc - cc[:, :1]).max() < 1e-9 * cfg["r1"]
    s = UnstructuredFIT3D(m, bc="pec", pml=cfg)
    p = s._pml
    # Diagonals massless (exact zero area), inner faces inert (zero length)
    rn = np.linalg.norm(m.nodes, axis=1)
    in_e = rn[m.edges].min(1) > RB - 1e-9
    area = m.dual_measure()
    ra, rb = rn[m.edges[:, 0]], rn[m.edges[:, 1]]
    cos = np.abs(np.einsum("ij,ij->i", m.edge_tangent, m.edge_midpoint))
    cos /= np.linalg.norm(m.edge_midpoint, axis=1)
    diag = in_e & (cos < 1 - 1e-9) & (np.abs(ra - rb) > 1e-9)
    assert diag.sum() == 3 * nl * ball.boundary_faces.size // 2  # one per side quad
    assert np.all(area[diag] == 0)
    assert np.all(area[p.spokes] > 0) and np.all(area[p.chords] > 0)
    assert np.all(m.dual_length()[p.sphere_faces] > 0)
    assert np.all(m.dual_length()[p.side_faces] > 0)


def test_unlayered_mesh_rejected():
    m = ball_mesh([RB + 2, RB + 4], H)  # tetrahedra (not prisms) in the "layer"
    with pytest.raises(ValueError, match="neither"):
        UnstructuredFIT3D(m, bc="pec", pml=dict(r0=RB + 2, r1=RB + 4))


# ------------------------------------------------------------------ algebra
def test_identity_filters_match_plain_solver():
    m, cfg = _mesh(0, 4)
    s0 = UnstructuredFIT3D(m, bc="pec")
    s1 = UnstructuredFIT3D(m, bc="pec", pml=dict(cfg, sigma_max=0.0))
    assert s0.massless.size > 0  # the elimination is exercised
    for s in (s0, s1):
        s.add_point_dipole([0.3, 0.1, -0.2], [1, 0.5, 0.2], amp)
    for _ in range(200):
        s0.one_step()
        s1.one_step()
    # round-off of 200 steps through two algebraically equal paths
    assert np.abs(s1.u - s0.u).max() < 1e-11 * np.abs(s0.u).max()
    assert np.abs(s1.b - s0.b).max() < 1e-10 * np.abs(s0.b).max()


def test_filter_matches_bilinear_response():
    dt, alpha, nper = 1e-3, 3.0, 400
    f = 1 / (nper * dt) * np.array([1.0, 7.0, 40.0])
    coeffs = [
        (np.full(3, 2.0), np.full(3, 50.0), np.full(3, 1.5), np.full(3, 20.0)),
        (np.ones(3), np.zeros(3), np.full(3, 3.0), np.full(3, 80.0)),
    ]
    L = _Layout()
    xin = L.add("x", 3)
    cas = _Cascade("c", L, coeffs, alpha, dt)
    S = _stage(L.n, cas.updates(_sel(xin, L.n), L.n))
    state = np.zeros(L.n)
    n_steps = 40 * nper
    t = np.arange(n_steps) * dt
    ys = []
    for k in range(n_steps):
        state[xin] = np.cos(2 * np.pi * f * t[k])
        state = S @ state
        ys.append(state[cas.out].copy())
    ys = np.array(ys)[-10 * nper :]
    tt = t[-10 * nper :]
    # complex amplitude of the steady state over 10 periods of the lowest f
    z = 2 * np.mean(ys * np.exp(-2j * np.pi * f * tt[:, None]), axis=0)
    expect = bilinear_response(coeffs, alpha, dt, f)
    assert np.abs(z - expect).max() < 1e-6 * np.abs(expect).max()
    # the plain continuous value differs at the highest frequency (warping)
    cont = np.prod(
        [(kn + sn / (alpha + 2j * np.pi * f)) / (kd + sd / (alpha + 2j * np.pi * f))
         for kn, sn, kd, sd in coeffs], axis=0)  # fmt: skip
    assert np.abs(cont - expect).max() > 1e-4 * np.abs(expect).max()


def test_step_transpose_dot_product():
    m, cfg = _mesh(0, 4)
    s = UnstructuredFIT3D(m, bc="pec", pml=dict(cfg, kappa_max=3.0, alpha=1e8))
    p = s._pml
    rng = np.random.default_rng(1)
    n, nu = p.layout.n, s.free.size
    x, y, src = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(nu)
    lhs = p.apply(x, src) @ y
    xt, gs = p.step_T(y)
    rhs = x @ xt + src @ gs
    assert abs(lhs - rhs) < 1e-13 * (abs(lhs) + np.linalg.norm(p.apply(x, src)) * np.linalg.norm(y))
    # Mutation: transposed stages in forward order -> not the transpose
    S1, S2, S3, S4 = p.stages
    bad = S4.T @ (S3.T @ (S2.T @ (S1.T @ y)))
    assert abs(lhs - (x @ bad + src @ gs)) > 1e-6 * abs(lhs)
    # Gradient of J = <w, x_3> wrt a source amplitude vs finite differences
    w = rng.standard_normal(n)
    g_src = rng.standard_normal(nu)

    def J(a):
        z = np.zeros(n)
        for _ in range(3):
            z = p.apply(z, a * g_src)
        return w @ z

    lam, dJ = w, 0.0
    for _ in range(3):
        lam, gs = p.step_T(lam)
        dJ += gs @ g_src
    fd = (J(1e-3) - J(-1e-3)) / 2e-3
    assert dJ == pytest.approx(fd, rel=1e-9)


@pytest.mark.parametrize("mutation", [None, "plain_massless"])
def test_massless_constraint_stretched(mutation):
    m, cfg = _mesh(0, 6)
    opts = dict(kappa_max=5.0, alpha=2e7)
    if mutation:
        opts["_mutation"] = mutation
    s = UnstructuredFIT3D(m, bc="pec", pml=dict(cfg, **opts))
    p = s._pml
    s.add_point_dipole([0.0, 0, 0], [0, 0, 1], amp)
    Mnu = np.asarray(s.Mnu_diag)
    worst, scale = 0.0, 0.0
    for _ in range(int(2 * T0 / s.dt)):
        s.one_step()
        h = Mnu * s.b
        h[p.f_pml] = p.x[p.ff.out]
        worst = max(worst, np.abs(s.Cs.T @ h).max())
        scale = max(scale, np.abs(s.Cf.T @ h).max())
    print(mutation, "max |C_s^T h| / max |C_f^T h|", worst / scale)
    if mutation is None:
        assert worst < 1e-10 * scale
    else:
        assert worst > 1e-6 * scale


# ------------------------------------------------------------------ physics
@pytest.mark.parametrize("kind", ["normal", "oblique"])
def test_reflection_and_baselines(kind):
    r_pml = _reflection(kind, "pml", 10)
    r_abc = _reflection(kind, "abc")
    r_pec = _reflection(kind, "pec")
    print(kind, "PML10", r_pml, "Silver-Mueller", r_abc, "PEC", r_pec)
    assert np.all(r_pml < TARGET)
    # the window contains the wall echo, suppressed by > 40 dB vs the total reflector
    assert np.all(r_pec > 100 * r_pml)
    assert np.all(r_abc > 10 * r_pml)


def test_reflection_falls_with_layers():
    r = {n: _reflection("normal", "pml", n).max() for n in (3, 6, 12)}
    print("max reflection vs layers", r)
    assert r[3] > r[6] > r[12]
    # No saturation from 6 to 12 layers. With the reference floor (massless
    # decision per chord) the ratio was 0.57 (1.3e-4 / 2.3e-4); a grading-
    # limited layer keeps falling (measured 0.10). 0.25 separates the two.
    assert r[12] < 0.25 * r[6]


# ------------------------------------------------------------------ the floor
N_THICK = 20  # baseline layer whose backing echo is outside the window


@functools.lru_cache(None)
def _sig(n_layers, bc="pec", per_chord=False, n_extra=0, n_pml=None, **opts):
    """Normal-incidence probe signals. per_chord=True restores the massless
    decision per chord (before the fix): more chords of a radial column turn
    massless at larger r."""
    m, cfg = ball_with_pml_mesh(RB, H, n_layers if n_pml is None else n_pml, n_extra=n_extra)
    if per_chord:
        m.massless_area_factor = np.ones(m.n_edges)
    s = UnstructuredFIT3D(m, bc=bc, dt=_dt(), pml=dict(cfg, **opts))
    return _signals(s, "normal")


def _rel(a, b):
    return np.linalg.norm(a - b, axis=-1).max(0) / np.linalg.norm(b, axis=-1).max(0)


def test_massless_decision_is_per_column():
    m, cfg = ball_with_pml_mesh(RB, H, N_REF, n_extra=N_EXTRA)  # fresh (mutated below)
    rn = np.linalg.norm(m.nodes, axis=1)
    ra, rb = rn[m.edges].T
    radii = m.shell_radii[1:-1]  # inner rings of the shell (the last is PEC)

    def per_ring():
        ml = np.zeros(m.n_edges, bool)
        ml[UnstructuredFIT3D(m, bc="pec").massless] = True
        return [int(ml[(np.abs(ra - r) < 1e-6) & (np.abs(rb - r) < 1e-6)].sum()) for r in radii]

    c1 = per_ring()
    print("massless chords per ring", c1)
    assert len(set(c1)) == 1
    # mutation: the per-chord decision lets the count grow with r
    m.massless_area_factor = np.ones(m.n_edges)
    c2 = per_ring()
    print("per-chord decision", c2)
    assert c2[-1] > 2 * c2[0]


def test_floor_was_the_reference_shell():
    """
    The ~1e-4 "floor" of the reflection against the vacuum-shell reference
    (10 and 12 layers equal) was the reference's own error: with a massless
    decision per chord, the number of massless chords of a radial column grows
    with r, so the vacuum shell is a radially inhomogeneous discrete medium
    and reflects into the window. Falsifiable checks (a 20-layer PML, whose
    backing echo is outside the window and whose shell echo is attenuated,
    is the baseline): per chord, the vacuum reference differs from that
    baseline by about the floor; per column (default) by < 1/10 of that, and
    the 12-layer reflection measured against the vacuum reference falls by
    more than 2x. Both fail if the floor were a property of the PML.
    """
    out = {}
    for pc in (True, False):
        vac = _sig(N_REF, per_chord=pc, n_extra=N_EXTRA, n_pml=N_REF)
        thick = _sig(N_THICK, per_chord=pc)
        r12 = _sig(12, per_chord=pc)
        out[pc] = dict(ref_err=_rel(vac, thick).max(), r12_vac=_rel(r12, vac).max(),
                       r12_thick=_rel(r12, thick).max())  # fmt: skip
        print("per chord" if pc else "per column", out[pc])
    assert out[True]["ref_err"] > 0.5 * out[True]["r12_vac"]  # the floor is the reference
    assert out[False]["ref_err"] < 0.1 * out[True]["ref_err"]
    assert out[False]["r12_vac"] < 0.5 * out[True]["r12_vac"]


@pytest.mark.parametrize("R", [1e-8, 1e-3])
def test_backing_echo_far_below_floor(R):
    """
    The wave reflected by the wall behind the layer flips sign with the wall
    type, so (E_PEC - E_PMC) / 2 is the backing echo. With the design
    R = 1e-8 it is < 1e-5 of the peak (>= 10x below the 1e-4 floor: the floor
    is not the backing echo); with R = 1e-3 (mutation) it must exceed 1e-5.
    """
    e = np.linalg.norm(_sig(10, "pec", R=R) - _sig(10, "pmc", R=R), axis=-1).max(0) / 2
    e = (e / np.linalg.norm(_sig(N_THICK), axis=-1).max(0)).max()
    print("R", R, "backing echo", e)
    if R == 1e-8:
        assert e < 1e-5
    else:
        assert e > 1e-5


@pytest.mark.parametrize(
    "mutation", ["negative_sigma", "swap", "drop_aux", "no_tangential"]
)
def test_mutations_fail(mutation):
    if mutation == "negative_sigma":
        r = _reflection("normal", "pml", 10, sigma_max=-2e9)
    else:
        r = _reflection("normal", "pml", 10, _mutation=mutation)
    print(mutation, r)
    if mutation == "no_tangential":
        # A planar (uniaxial) stretch on the curved shell still absorbs: its
        # extra reflection is a curvature error (measured 3-4.5e-3, -47 dB,
        # below the target), so compare it with the correct layer instead
        r_ok = _reflection("normal", "pml", 10)
        assert r.max() > 10 * r_ok.max()
    else:
        assert not np.all(r < TARGET)


def _ball_energy(s, u_prev):
    """Leapfrog-conserved energy of the dofs in r <= RB (same mask in every run)."""
    m = s.mesh
    rn = np.linalg.norm(m.nodes, axis=1)
    ein = (rn[m.edges].max(1) <= RB + 1e-7)[s.free]
    fin = rn[m.faces].max(1) <= RB + 1e-7
    we = 0.5 * np.sum(s.cap[ein] * s.u[ein] * u_prev[ein])
    return we + 0.5 * np.sum(s.Mnu_diag[fin] * s.b[fin] ** 2)


@pytest.mark.parametrize("label", ["pml", "pec"])
def test_interior_energy_decays(label):
    opts = dict(kappa_max=5.0, alpha=2 * np.pi * C0 / LAM0 / 10)
    s = _solver(label, 10, **(opts if label == "pml" else {}))
    s.add_point_dipole([0.2, -0.1, 0.3], [0, 0, 1], amp)
    t_off = T0 + 6 * TAU  # |I l| < 2e-15 of its peak afterwards
    w, t = [], []
    for _ in range(int((2 * T0 + 40 * H / C0) / s.dt)):
        u_prev = s.u.copy()
        s.one_step()
        w.append(_ball_energy(s, u_prev))
        t.append(s.n * s.dt)
    w, t = np.array(w), np.array(t)
    peak = w.max()
    after = w[t > t_off]
    # largest rise above the running minimum after the source is off
    rise = np.max(after - np.minimum.accumulate(after))
    print(label, "W(t_off)/peak", after[0] / peak, "W_end/peak", after[-1] / peak,
          "max rise/peak", rise / peak)  # fmt: skip
    if label == "pml":
        assert after[-1] < 1e-8 * peak
        # monotone up to the energy a -40 dB reflection can return (R^2)
        assert rise < TARGET**2 * peak
    else:  # control: the closed lossless PEC ball conserves this energy exactly
        assert np.abs(after - after[0]).max() < 1e-9 * after[0]


def _long_run(n, alpha):
    m, cfg = ball_with_pml_mesh(3.0, H, 6)
    s = UnstructuredFIT3D(m, bc="pec", pml=dict(cfg, kappa_max=5.0, alpha=alpha))
    s.add_point_dipole([0.2, 0.1, 0.3], [0.3, 0.2, 1.0], amp)
    a = np.empty(n // 100)
    for k in range(n):
        s.one_step()
        if k % 100 == 0:
            a[k // 100] = np.abs(s.u).max()
            if not np.isfinite(a[k // 100]):
                a[k // 100 :] = np.inf
                break
    return a


def _growth(a):
    """max over the last 10 % of the run / max over 40-50 %, and late level."""
    q = a.size // 10
    return a[-q:].max() / a[4 * q : 5 * q].max(), a[a.size // 2 :].max() / a.max()


A_CFS = 2 * np.pi * C0 / LAM0 / 10


@pytest.mark.slow
def test_long_run_stability():
    # 1e5 steps (~ 3.3e4 h/c, ~ 2800 periods at the centre wavelength)
    g, late = _growth(_long_run(100_000, A_CFS))
    print("late growth", g, "late max / peak", late)
    assert np.isfinite(g)
    # the bounded residual (~1e-7 of peak, a non-radiating field at the
    # source) fluctuates by ~10 %; a factor 2 (10x that) flags growth
    assert g < 2.0
    assert late < 1e-6


def test_long_run_mutation_unstable_pole_fails():
    # alpha < 0 makes P' + alpha P = f an exponentially growing pole
    g, late = _growth(_long_run(10_000, -A_CFS))
    print("alpha < 0: growth", g, "late", late)
    assert not (np.isfinite(g) and g < 2.0 and late < 1e-6)
