"""
Polygonal 2D FIT (`PolyMesh`, `PolygonFIT2D`, `hybrid_circles_mesh`,
`voronoi_disk_mesh`).

Each test states the failure it guards against and includes a mutation that
must make it fail (Rule 1: a test is only evidence if it could fail).

* Triangles: on a Delaunay mesh the polygon solver reproduces
  `UnstructuredFIT2D` (masses, stiffness, time step and fields after 300
  plane-wave steps, TM and TE) to round-off. The mimetic Hodge gives the
  same (cotangent) stiffness on triangles (G u is uniform per cell, so the
  stability term never acts); mutation: its median-dual masses differ by
  > 1 %.
* Mimetic local matrices: exact for uniform fields (F^T M F = |C| |u|^2,
  M N = R) and SPD on random convex polygons. Mutation: dropping the
  stability term leaves a singular matrix; dropping the consistency term
  breaks exactness.
* Cavities (closed PEC/PMC, eigenmodes vs analytic): rectangles and graded
  rectilinear grids (diagonal Hodge) and randomly distorted quadrilaterals
  (mimetic Hodge) converge at second order; the diagonal Hodge is rejected
  on the distorted mesh (non-orthogonal).
* Hybrid mesh: tiles the domain, orthogonal (defect < 1e-6), positive masses;
  with eps = 1 everywhere the square-cavity modes converge at second order
  (the triangle patch is transparent). Mutation: removing the triangle
  contributions on the staircase edges shifts the modes by > 10x the error.
* Voronoi mesh: conforming interface (polygon area = pi a^2 to 1e-3),
  positive dual lengths and masses; circular PEC cavity TM_01/TM_11 modes
  converge at about second order.
* PML (split field, squares): reflection of a point-source pulse < 1 % of
  the direct signal; mutation: without the magnetic (edge) loss the layer
  is not matched and reflects > 5x more.
* Dispersive plane wave (Drude cylinder, TE LSPR and TM): Q_sca within
  3 % RMS of the series; mutation: dropping the background term
  eps_b E_inc . n from the segment D (TE) gives > 10 % error.
* Mie (n = 2, x = 2..10) on a coarse hybrid disk mesh: RMS Q_sca error
  < 10 % (recorded 8.6 % at res 10, TM).
"""

import numpy as np
import pytest
from scipy.constants import c as C0
from scipy.special import jn_zeros

from wakis.sources import GaussianSource
from wakis.unstructured import UnstructuredFIT2D, disk_mesh, graded_disk_mesh
from wakis.unstructured.polymesh import PolyMesh
from wakis.unstructured.polymeshers import hybrid_circles_mesh, voronoi_disk_mesh
from wakis.unstructured.polysolver2d import PolygonFIT2D


# ---------------------------------------------------------------- triangles
@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_reproduces_triangle_solver(pol):
    m = disk_mesh([1e-6, 2e-6], 0.12e-6)
    a = UnstructuredFIT2D(m, {0: [4.0, 1.0]}, polarization=pol, bc="abc")
    b = PolygonFIT2D(PolyMesh.from_simplex(m), {0: [4.0, 1.0]}, polarization=pol,
                     bc="abc", hodge="orthogonal", dt=a.dt)  # fmt: skip
    assert np.allclose(a.mass, b.mass, rtol=1e-12)
    assert abs(a.K - b.K).max() < 1e-12 * abs(a.K).max()
    src = GaussianSource(1.5e14, 0.5e14)
    a.add_plane_wave(src)
    b.add_plane_wave(src)
    for _ in range(300):
        a.one_step()
        b.one_step()
    assert np.abs(a.u - b.u).max() < 1e-9 * np.abs(a.u).max()
    # On triangles G u is a uniform field per cell, so the mimetic Hodge gives
    # the same (cotangent) stiffness; only its median-dual node masses differ.
    c = PolygonFIT2D(PolyMesh.from_simplex(m), {0: [4.0, 1.0]}, polarization=pol,
                     bc="abc", hodge="mfd")  # fmt: skip
    assert abs(c.K - a.K).max() < 1e-12 * abs(a.K).max()
    # mutation: median-dual masses differ from the circumcentric ones
    assert np.abs(c.mass / a.mass - 1).max() > 1e-2


# ---------------------------------------------------------------- mimetic
def _random_convex_polygon(rng, n):
    th = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = rng.uniform(0.8, 1.2)
    return np.stack([r * np.cos(th), r * np.sin(th) * rng.uniform(0.5, 1.5)], 1)


def test_mimetic_local_matrix_exact_and_spd():
    rng = np.random.default_rng(1)
    for n in (3, 4, 5, 7):
        P = _random_convex_polygon(rng, n) + rng.normal(size=2)
        pm = PolyMesh(P, [list(range(n))])
        M = pm.mfd_local_matrices()[0]
        e, sg = pm.corner_edge, pm.corner_sign
        N = pm.edge_length[e][:, None] * (pm.edge_normal[e] * sg[:, None])
        R = pm.edge_midpoint[e] - pm.cell_centroid[0]
        for _ in range(3):
            u = rng.normal(size=2)
            F = N @ u
            assert np.isclose(F @ M @ F, pm.cell_area[0] * u @ u, rtol=1e-12)
        assert np.allclose(M @ N, R, atol=1e-12)
        assert np.linalg.eigvalsh(M).min() > 0
        # mutation 1: consistency part only -> rank 2, singular for n > 2
        cons = R @ R.T / pm.cell_area[0]
        assert np.linalg.eigvalsh(cons).min() < 1e-12 * np.abs(cons).max()
        # mutation 2: stability part only -> not exact
        stab = M - cons
        u = np.array([1.0, 0.3])
        assert not np.isclose((N @ u) @ stab @ (N @ u), pm.cell_area[0] * u @ u, rtol=1e-2)


# ---------------------------------------------------------------- cavities
def _rect_modes(a, b, n):
    f = [C0 / 2 * np.hypot(i / a, j / b) for i in range(1, 6) for j in range(1, 6)]
    return np.sort(f)[:n]


def _cavity_error(mesh, hodge, n=3):
    s = PolygonFIT2D(mesh, polarization="TM", bc="pec", hodge=hodge)
    f, _ = s.eigenmodes(n)
    return np.abs(f / _rect_modes(1.0, 0.5, n) - 1).max()


def test_rectangles_second_order():
    errs = []
    for n in (10, 20):
        x = np.linspace(0, 1, n + 1)
        y = np.linspace(0, 0.5, n // 2 + 1)
        # smooth grading along x (non-uniform but orthogonal)
        x = x + 0.03 * np.sin(2 * np.pi * x)
        errs.append(_cavity_error(PolyMesh.rectilinear(x, y), "orthogonal"))
    assert errs[1] < errs[0] / 3.3, errs


def _distorted_quads(n, amp, seed=0):
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 0.5, n // 2 + 1)
    pm = PolyMesh.rectilinear(x, y)
    rng = np.random.default_rng(seed)
    X = pm.nodes.copy()
    inner = (X[:, 0] > 0) & (X[:, 0] < 1) & (X[:, 1] > 0) & (X[:, 1] < 0.5)
    X[inner] += amp / n * rng.uniform(-1, 1, (inner.sum(), 2))
    return PolyMesh(X, [pm.cell(i) for i in range(pm.n_cells)])


def test_mimetic_distorted_quads_converge():
    with pytest.raises(ValueError, match="not orthogonal"):
        PolygonFIT2D(_distorted_quads(10, 0.25), hodge="orthogonal")
    # recorded: 1.9 % (n = 20), 0.44 % (n = 40)
    e1 = _cavity_error(_distorted_quads(20, 0.25), "mfd")
    e2 = _cavity_error(_distorted_quads(40, 0.25), "mfd")
    assert e2 < e1 / 3.0, (e1, e2)
    assert e2 < 0.006


# ---------------------------------------------------------------- hybrid
def _hybrid_square_cavity(h):
    # eps = 1 everywhere: the triangle patch around the (virtual) circle must
    # be invisible; square cavity of side 2 L
    mesh, info = hybrid_circles_mesh([(0.07, -0.05, 0.3)], h, 2 * h, 1.0)
    return mesh, info


def test_hybrid_mesh_geometry_and_cavity():
    errs = []
    for h in (0.04, 0.02):
        mesh, info = _hybrid_square_cavity(h)
        d = mesh.diagnostics()
        assert d["max_orthogonality_defect"] < 1e-6
        assert d["min_dual_edge"] > 0 and d["min_node_area"] > 0
        assert info["n_triangles"] > 100 and info["n_squares"] > 100
        L = 2 * info["half_width"]
        s = PolygonFIT2D(mesh, polarization="TM", bc="pec")
        f, _ = s.eigenmodes(4)
        fe = np.sort([C0 / 2 * np.hypot(i / L, j / L) for i in range(1, 4) for j in range(1, 4)])[:4]
        errs.append(np.abs(f / fe - 1).max())
    assert errs[1] < errs[0] / 3.0, errs
    # mutation: drop the triangle half of the staircase edges' dual lengths
    mesh, info = _hybrid_square_cavity(0.02)
    s = PolygonFIT2D(mesh, polarization="TM", bc="pec")
    sq = mesh.cell_size[mesh.corner_cell] == 4
    tri_on_shared = (~sq) & np.isin(mesh.corner_edge, mesh.corner_edge[sq])
    mesh.corner_h = np.where(tri_on_shared, 0.0, mesh.corner_h)
    s2 = PolygonFIT2D(mesh, polarization="TM", bc="pec")
    f1, _ = s.eigenmodes(4)
    f2, _ = s2.eigenmodes(4)
    assert np.abs(f2 / f1 - 1).max() > 10 * errs[1]


# ---------------------------------------------------------------- Voronoi
def test_voronoi_mesh_conforms_and_converges():
    a, R = 0.4, 1.0
    errs = []
    for s in (0.08, 0.04):
        mesh = voronoi_disk_mesh([a, R], s, collapse=0.1, lloyd=5)
        d = mesh.diagnostics()
        assert d["min_dual_edge"] > 0 and d["min_node_area"] > 0
        assert abs(mesh.cell_area[mesh.cell_tags == 0].sum() / (np.pi * a**2) - 1) < 1e-3
        assert abs(mesh.cell_area.sum() / (np.pi * R**2) - 1) < 1e-3
        sol = PolygonFIT2D(mesh, polarization="TM", bc="pec", ortho_tol=np.inf)
        f, _ = sol.eigenmodes(3)
        fe = C0 / (2 * np.pi * R) * np.array([jn_zeros(0, 1)[0], jn_zeros(1, 1)[0], jn_zeros(1, 1)[0]])
        errs.append(np.abs(f / fe - 1).max())
    assert errs[1] < errs[0] / 2.5, errs
    assert errs[1] < 0.01


# ---------------------------------------------------------------- PML
def _pml_run(n_in, n_pml, matched=True):
    h = 0.01
    L = n_in * h / 2
    x = h * np.arange(-n_in // 2 - n_pml, n_in // 2 + n_pml + 1)
    mesh = PolyMesh.rectilinear(x, x)
    pml = dict(x=(-L, L), y=(-L, L), thickness=n_pml * h) if n_pml else None
    s = PolygonFIT2D(mesh, polarization="TM", bc="pec", pml=pml, cfl=0.9)
    if pml is not None and not matched:
        s.edge_rate[:] = 0
        s._ce[:] = 0
        s._lossy_edges = np.array([], int)
    src = GaussianSource(1.5e9, 0.75e9)
    s.add_point_source((0.0, 0.0), src)
    pr = s.add_probe(np.array([[0.26, 0.0], [0.26, 0.26]]))  # same points in all runs
    s.run(int(5e-9 / s.dt))
    return s.probe_signal(pr)


def test_split_field_pml():
    ref = _pml_run(240, 0)  # PEC walls 1.2 m away: no reflection within 5 ns
    sig = _pml_run(60, 12)
    bad = _pml_run(60, 12, matched=False)
    n = min(len(ref), len(sig))
    r_ok = np.abs(sig[:n] - ref[:n]).max() / np.abs(ref[:n]).max()
    r_bad = np.abs(bad[:n] - ref[:n]).max() / np.abs(ref[:n]).max()
    assert r_ok < 0.01, r_ok
    assert r_bad > 5 * r_ok, (r_ok, r_bad)


# ---------------------------------------------------------------- dispersive
def _drude_qsca(pol, mutate=False):
    from wakis.analytic import cylinder_efficiencies
    from wakis.cylinder_harmonics import (
        circle_points, fit_harmonics, mie_coefficients_from_fit, project_harmonics,
        q_sca_from_coefficients,
    )  # fmt: skip
    from wakis.dispersive import DispersiveMaterial, Drude

    a, R = 30e-9, 400e-9
    radii = np.array([150e-9, 200e-9, 250e-9])
    mat = DispersiveMaterial(eps_inf=4.0, poles=[Drude(1.37e16, 1e14)])
    lam = np.linspace(280e-9, 600e-9, 9)
    f = C0 / lam
    hin = 3e-9

    def size(r):
        return hin if r < a + 3 * hin else min(25e-9, hin * 1.2 ** ((r - a - 3 * hin) / hin))

    mesh = PolyMesh.from_simplex(graded_disk_mesh([a, R], size))
    s = PolygonFIT2D(mesh, {0: mat}, polarization=pol, bc="abc")
    s.add_plane_wave(GaussianSource(0.5 * (f[0] + f[-1]), 0.3 * (f[0] - f[-1])))
    if mutate:
        for g in s._te_groups:
            g["eps_b"] = 0.0
    M = 64
    pts = np.vstack([circle_points(r, M)[0] for r in radii])
    P = s._interp(pts)
    n = int(120e-15 / s.dt)
    dt = s.dt
    times = (np.arange(n) + 1) * dt + (0 if pol == "TM" else -0.5 * dt)
    acc = np.zeros((len(f), len(pts)), complex)
    for i in range(n):
        s.one_step()
        acc += np.outer(np.exp(-2j * np.pi * f * times[i]), P @ s.u) * dt
    u0 = (np.exp(-2j * np.pi * np.outer(f, times)) @ s.incident_field(np.zeros((1, 2)), times)) * dt
    k = 2 * np.pi * f / C0
    F = project_harmonics(acc.reshape(len(f), 3, M), 4)
    al, ga, _ = fit_harmonics(F, k, radii, 4)
    q = q_sca_from_coefficients(mie_coefficients_from_fit(al, ga, np.conj(u0), 4), k * a)
    qr = np.array(
        [cylinder_efficiencies(np.sqrt(e), x, pol)[1] for e, x in zip(mat.epsilon(f), k * a)]
    )
    return np.sqrt(np.mean((q / qr - 1) ** 2))


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_dispersive_plane_wave(pol):
    err = _drude_qsca(pol)
    assert err < 0.03, err
    if pol == "TE":
        assert _drude_qsca(pol, mutate=True) > 0.1


# ---------------------------------------------------------------- Mie smoke
def test_mie_hybrid_disk_coarse():
    import sys
    from pathlib import Path

    study = Path(__file__).resolve().parents[1] / "studies" / "mie-cylinder-polyfit"
    sys.path.insert(0, str(study))
    sys.path.insert(0, str(study.parent / "mie-cylinder-mesh"))
    import polyfit_study as P

    out = P.run("hybrid-disk-compact", "TM", 10, 1000e-15)
    q, qr = np.array(out["q"]), np.array(out["q_ref"])
    rms = np.sqrt(np.mean((q / qr - 1) ** 2))
    assert rms < 0.10, rms
