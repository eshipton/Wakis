"""
Hybrid Cartesian / tetrahedral FIT (`wakis.hybrid`, milestone 1).

Every property is checked together with a mutation that must break it:

* discrete de Rham sequence: curl(P grad phi) = 0 on the composite mesh
  (round-off), broken by the 'nearest' and 'inject' trace mutations;
* interface flux bookkeeping: square flux minus the sum of its triangle
  fluxes stays constant (round-off), broken by the 'scale' trace mutation;
* exact energy conservation of the leapfrog (LU and Chebyshev mass solves),
  broken by the 'asym' mutation (e-update with a different trace);
* exact transpose of one step (dot-product test), broken by a non-symmetric
  perturbation of the mass block that the Chebyshev transpose assumes away;
* stability bound lambda_max <= max(lambda_cart, lambda_tet);
* TE10 reflection of a vacuum patch decreases at least like h^2 and sits
  far above the fit floor; the 'nearest' trace gives > 10x more;
* PEC-cavity resonances converge at second order like the pure Cartesian
  grid;
* the near-field Mie reference: incident series = plane wave, tangential
  total field = 0 on the sphere, far field = `mie_s1_s2`;
* the mesh checks reject a patch whose boundary triangles straddle squares.

Tolerances: round-off checks use 1e3 x the double-precision epsilon times
the measured growth (measured values are ~1e-15); convergence checks use the
derived order (2) with the margin stated at each assertion.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from wakis.hybrid import HybridFIT3D, box_trace, delaunay_box_mesh, kuhn_box_mesh
from wakis.hybrid.mie_pec import pec_sphere_field, plane_wave_line_integrals
from wakis.hybrid.verification import cavity_modes, waveguide_reflection

H = 0.1
XN = np.arange(0, 1.0 + 1e-9, H)
YN = np.arange(0, 0.8 + 1e-9, H)
ZN = np.arange(0, 0.6 + 1e-9, H)
BOX = ((3, 7), (2, 6), (2, 4))
LINES = (XN[3:8], YN[2:7], ZN[2:5])


@pytest.fixture(scope="module")
def meshes():
    return {
        "kuhn": kuhn_box_mesh(*LINES, r=2),
        "delaunay": delaunay_box_mesh(*LINES, h=0.06, r=2, seed=0),
    }


def _phi(X):
    """Potential vanishing on the PEC walls of the 1 x 0.8 x 0.6 cavity."""
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return (np.sin(3 * x) + y**2 * z + 0.3 * x * y + 1) * x * (1 - x) * y * (0.8 - y) * z * (0.6 - z)


def _gradient_voltages(s, tm):
    """Edge differences of phi; box-surface tet nodes take the bilinear (Q1)
    trace of phi on their Cartesian square, as the hexahedral element does."""
    mid, L, d = s.edge_mid, s.edge_len, s.edge_dir
    ar = np.arange(len(d))
    p, q = mid.copy(), mid.copy()
    p[ar, d] -= L / 2
    q[ar, d] += L / 2
    X = tm.nodes
    ph = _phi(X)
    lines = (XN, YN, ZN)
    lo = [XN[3], YN[2], ZN[2]]
    hi = [XN[7], YN[6], ZN[4]]
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        for pl in (lo[a], hi[a]):
            on = np.abs(X[:, a] - pl) < 1e-12
            u, v = X[on, b], X[on, c]
            ib = np.clip(np.searchsorted(lines[b], u, "right") - 1, 0, len(lines[b]) - 2)
            ic = np.clip(np.searchsorted(lines[c], v, "right") - 1, 0, len(lines[c]) - 2)
            u0, u1 = lines[b][ib], lines[b][ib + 1]
            v0, v1 = lines[c][ic], lines[c][ic + 1]
            tu, tv = (u - u0) / (u1 - u0), (v - v0) / (v1 - v0)

            def corner(uu, vv):
                Y = np.zeros((on.sum(), 3))
                Y[:, a], Y[:, b], Y[:, c] = pl, uu, vv
                return _phi(Y)

            ph[on] = ((1 - tu) * (1 - tv) * corner(u0, v0) + tu * (1 - tv) * corner(u1, v0)
                      + (1 - tu) * tv * corner(u0, v1) + tu * tv * corner(u1, v1))  # fmt: skip
    e_all = np.r_[_phi(q) - _phi(p), ph[tm.edges[:, 1]] - ph[tm.edges[:, 0]]]
    return e_all[s.free_edges], e_all


@pytest.mark.parametrize("kind", ["kuhn", "delaunay"])
def test_de_rham_and_trace_mutations(meshes, kind):
    """curl(P grad phi) = 0 to round-off; trace mutations break it."""
    tm = meshes[kind]
    res = {}
    for mut in (None, "nearest", "inject"):
        s = HybridFIT3D(XN, YN, ZN, box=BOX, tet_mesh=tm, mutate=mut, dt=1.0)
        ef, e_all = _gradient_voltages(s, tm)
        res[mut] = np.abs(s.Cf @ (s.P @ ef)).max() / np.abs(ef).max()
        if mut is None:
            # hanging edges reproduce the potential differences exactly
            hang = s.nEc + s.tet_hanging
            assert np.abs((s.P @ ef)[hang] - e_all[hang]).max() < 1e-13 * np.abs(ef).max()
    assert res[None] < 1e-13  # measured ~1e-15
    assert res["nearest"] > 1e-3 and res["inject"] > 1e-3  # measured 0.04-0.29


def test_trace_line_integral_of_nedelec_field():
    """box_trace integrates the hexahedral trace exactly along a diagonal."""
    from wakis.subgrid3d.block import CartesianBlock

    blk = CartesianBlock(XN, YN, ZN)
    lo, hi = np.array([3, 2, 2]), np.array([7, 6, 4])
    # face z = ZN[2], square x in [0.4, 0.5], y in [0.3, 0.4]; segment inside
    p = np.array([[0.42, 0.31, ZN[2]]])
    q = np.array([[0.47, 0.38, ZN[2]]])
    rows, cols, vals = box_trace(blk, lo, hi, p, q)
    rng = np.random.default_rng(5)
    e = rng.standard_normal(blk.n_edges)
    got = vals @ e[cols]
    # reference: E_x(y), E_y(x) of the trace, integrated by quadrature
    ix, iy = blk.edge_index(0, 4, 3, 2)[0], blk.edge_index(0, 4, 4, 2)[0]
    jx, jy = blk.edge_index(1, 4, 3, 2)[0], blk.edge_index(1, 5, 3, 2)[0]
    t = np.linspace(0, 1, 2001)
    P = p + t[:, None] * (q - p)
    Ex = (e[ix] * (0.4 - P[:, 1]) + e[iy] * (P[:, 1] - 0.3)) / H**2
    Ey = (e[jx] * (0.5 - P[:, 0]) + e[jy] * (P[:, 0] - 0.4)) / H**2
    ref = np.trapezoid(Ex * (q - p)[0, 0] + Ey * (q - p)[0, 1], t)
    assert abs(got - ref) < 1e-12 * max(1.0, abs(ref))
    with pytest.raises(ValueError):  # straddles two squares
        box_trace(blk, lo, hi, np.array([[0.42, 0.31, ZN[2]]]), np.array([[0.55, 0.38, ZN[2]]]))


@pytest.mark.parametrize("kind", ["kuhn", "delaunay"])
def test_flux_bookkeeping_and_divergence(meshes, kind):
    """Square flux - sum of triangle fluxes and every cell divergence stay 0."""
    tm = meshes[kind]
    out = {}
    for mut in (None, "scale"):
        s = HybridFIT3D(XN, YN, ZN, box=BOX, tet_mesh=tm, mutate=mut)
        s.add_current((0.45, 0.35, 0.3), 2,
                      lambda t: np.exp(-(((t - 1e-9) / 3e-10) ** 2)) * np.sin(4e9 * np.pi * t))  # fmt: skip
        s.run(120)
        bmax = np.abs(s.b).max()
        out[mut] = np.abs(s.interface_flux_mismatch()).max() / bmax
        assert np.abs(s.divergence()).max() < 1e-12 * bmax  # topological: D C = 0
    assert out[None] < 1e-12  # measured ~1e-15
    assert out["scale"] > 1e-3  # measured 1e-2


@pytest.mark.parametrize("kind,solver", [("kuhn", "lu"), ("kuhn", "chebyshev"),
                                         ("delaunay", "chebyshev")])  # fmt: skip
def test_energy_conservation(meshes, kind, solver):
    """1/2 b.W b + 1/2 e^n.M e^{n+1} is constant; 'asym' breaks it."""
    tm = meshes[kind]
    rng = np.random.default_rng(1)
    drift = {}
    for mut in (None, "asym"):
        s = HybridFIT3D(XN, YN, ZN, box=BOX, tet_mesh=tm, mutate=mut, mass_solver=solver)
        s.e = rng.standard_normal(s.n_free)
        s.b = rng.standard_normal(s.n_faces) * 1e-9
        E = s.run(200, energy=True)
        drift[mut] = np.ptp(E) / np.abs(E).mean()
    # measured 4e-16 (LU) and < 1e-15 (Chebyshev, polynomial error <= 1e-16)
    assert drift[None] < 1e-12
    assert drift["asym"] > 1e-4


def test_stability_bound(meshes):
    """lambda_max <= max of the two regions' own lambda_max (mediant bound)."""
    for tm in meshes.values():
        s = HybridFIT3D(XN, YN, ZN, box=BOX, tet_mesh=tm)
        lc, lt = s.block_lambda_bounds()
        assert s.lambda_max <= max(lc, lt) * (1 + 1e-8)
        assert s.lambda_max >= 0.5 * max(lc, lt)  # the bound is not vacuous here
        assert s.dt < 2 / np.sqrt(s.lambda_max)


@pytest.mark.parametrize("solver", ["lu", "chebyshev"])
def test_dot_product(meshes, solver):
    """<step(x), y> = <x, step_T(y)> to round-off; a non-symmetric mass
    perturbation (which the symmetric Chebyshev transpose ignores) fails."""
    tm = meshes["kuhn"]
    s = HybridFIT3D(XN, YN, ZN, box=BOX, tet_mesh=tm, mass_solver=solver)
    rng = np.random.default_rng(2)
    b, e = rng.standard_normal(s.n_faces), rng.standard_normal(s.n_free)
    i_all = rng.standard_normal(s.nEc + s.nEt)
    bb, eb = rng.standard_normal(s.n_faces), rng.standard_normal(s.n_free)

    def mismatch(s):
        b1, e1 = s.linear_step(b, e, i_all)
        tb, te, ti = s.linear_step_T(bb, eb)
        lhs = b1 @ bb + e1 @ eb
        rhs = b @ tb + e @ te + i_all @ ti
        return abs(lhs - rhs) / (abs(lhs) + abs(rhs))

    assert mismatch(s) < 1e-13  # measured ~1e-16
    # mutation: non-symmetric coupled mass block
    M = s.M.tolil()
    S = s.msolver.S
    M[S[0], S[1]] = M[S[0], S[1]] + 0.3 * M[S[0], S[0]]
    s.M = M.tocsr()
    s._msolver = None
    if solver == "chebyshev":
        assert mismatch(s) > 1e-8
    else:  # LU solves the transpose exactly: the test still passes
        assert mismatch(s) < 1e-13


def test_waveguide_reflection_converges():
    """TE10 through a vacuum Kuhn patch: |R| far above the floor, falls at
    least like h^2 (measured order ~3), and the 'nearest' trace is > 10x worse."""
    floor = waveguide_reflection(8, patch=None)
    R8 = waveguide_reflection(8, patch="kuhn")
    R12 = waveguide_reflection(12, patch="kuhn")
    Rn = waveguide_reflection(8, patch="kuhn", mutate="nearest")
    assert floor < 1e-4 and R8 > 30 * floor  # measured 5.5e-6 vs 3.0e-3
    assert R12 < R8 * (8 / 12) ** 2  # measured 8.2e-4 vs bound 1.3e-3
    assert Rn > 10 * R8  # measured 4.2e-2


def test_cavity_second_order():
    """Lowest PEC-cavity resonance with a Kuhn or Delaunay patch converges at
    second order (observed order between 1.7 and 2.5)."""
    for patch in ("kuhn", "delaunay"):
        errs = []
        for N in (8, 16):
            f, fe, _ = cavity_modes(N, patch=patch, n_modes=3)
            errs.append(abs(f[0] - fe[0]) / fe[0])
        order = np.log(errs[0] / errs[1]) / np.log(2)
        assert 1.7 < order < 2.5, (patch, errs, order)


def test_mie_pec_reference():
    """Incident series = plane wave; total tangential E = 0 on r = a;
    far field = S2 of `mie_s1_s2` at large |m| (PEC limit, O(1/|m|))."""
    from wakis.analytic import mie_s1_s2

    k, a = 2 * np.pi, 0.3
    rng = np.random.default_rng(0)
    n = rng.standard_normal((100, 3))
    n /= np.linalg.norm(n, axis=1)[:, None]
    Et = pec_sphere_field(a * n, k, a, "total")
    tang = Et - np.einsum("ij,ij->i", Et, n)[:, None] * n
    assert np.abs(tang).max() < 1e-12
    Q = 0.7 * n
    Ei = pec_sphere_field(Q, k, a, "incident")
    assert np.abs(Ei[:, 0] - np.exp(1j * k * Q[:, 2])).max() < 1e-9
    assert np.abs(Ei[:, 1:]).max() < 1e-9
    r, th = 200.0, np.linspace(0.2, 3.0, 5)
    pts = np.stack([r * np.sin(th), 0 * th, r * np.cos(th)], 1)
    Es = pec_sphere_field(pts, k, a)
    eth = np.stack([np.cos(th), 0 * th, -np.sin(th)], 1)
    S2 = np.einsum("ij,ij->i", Es, eth) / (np.exp(1j * k * r) / (-1j * k * r))
    _, S2ref = mie_s1_s2(1e6 + 1e6j, k * a, th)
    assert np.abs(S2 - S2ref).max() < 5e-3 * np.abs(S2ref).max()  # measured 1.3e-3
    # exact plane-wave line integral vs quadrature
    p, q = rng.uniform(-1, 1, (5, 3)), rng.uniform(-1, 1, (5, 3))
    t = np.linspace(0, 1, 4001)
    ref = [np.trapezoid(np.exp(1j * k * (pp[2] + t * (qq[2] - pp[2]))) * (qq[0] - pp[0]), t)
           for pp, qq in zip(p, q)]  # fmt: skip
    assert np.allclose(plane_wave_line_integrals(k, p, q), ref, atol=1e-6)


def test_patch_must_refine_the_squares():
    """A Kuhn patch on node lines that are not the Cartesian ones is rejected."""
    shifted = (LINES[0], LINES[1] + 0.03 * (np.arange(5) % 2), LINES[2])
    tm = kuhn_box_mesh(*shifted)
    with pytest.raises(ValueError):
        HybridFIT3D(XN, YN, ZN, box=BOX, tet_mesh=tm)
    with pytest.raises(ValueError):  # box touching the outer wall
        HybridFIT3D(XN, YN, ZN, box=((0, 4), (2, 6), (2, 4)), tet_mesh=kuhn_box_mesh(XN[0:5], *LINES[1:]))


def test_loose_chebyshev_tolerance_keeps_stability(meshes):
    """mass_tol = 1e-4: the degree drops ~4x, the step stays exactly
    transposable, and the M-energy stays within ~mass_tol of its start
    (it is exactly conserved for the modified SPD mass)."""
    tm = meshes["delaunay"]
    ref = HybridFIT3D(XN, YN, ZN, box=BOX, tet_mesh=tm, mass_solver="chebyshev")
    s = HybridFIT3D(XN, YN, ZN, box=BOX, tet_mesh=tm, mass_solver="chebyshev",
                    mass_tol=1e-4, dt=ref.dt)  # fmt: skip
    assert s.msolver.degree < 0.35 * ref.msolver.degree
    rng = np.random.default_rng(4)
    x, y = rng.standard_normal(s.n_free), rng.standard_normal(s.n_free)
    sym = abs(s.msolver.solve(x) @ y - x @ s.msolver.solve(y)) / abs(x @ s.msolver.solve(y))
    assert sym < 1e-13
    s.e = rng.standard_normal(s.n_free)
    s.b = rng.standard_normal(s.n_faces) * 1e-9
    E = s.run(400, energy=True)
    dev = np.abs(E / E[0] - 1)
    # |e^n.(M - M~) e^{n+1}| <= mass_tol |e^n|_M |e^{n+1}|_M <= 2 mass_tol E: bound 2e-4
    # (measured 1.1e-6)
    assert dev.max() < 2e-4
    assert dev[200:].max() < 3 * dev[:200].max() + 1e-12  # no secular growth
