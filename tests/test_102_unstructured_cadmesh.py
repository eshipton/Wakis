"""
CAD -> tetrahedral mesh import, tetrahedral mesh quality and the element-wise
stability bound of the 3D unstructured FIT, and the time-step-driven node
relocation (`wakis.unstructured.cad`, `.quality`, `.improve`).

Physics guide: "CAD meshes and the tetrahedral time step"
(docs/physicsguide_sections/unstructured_cadmesh.md).

gmsh (GPL) is an optional external tool: every test that needs it calls
``pytest.importorskip("gmsh")`` and is skipped without it; the rest only need
NumPy/SciPy.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.constants import c

from wakis.unstructured import SimplexMesh, UnstructuredFIT3D, ball_mesh
from wakis.unstructured.analytic import layered_pillbox_tm0, sphere_in_cavity
from wakis.unstructured.cad import region_sizes, size_from_cpw
from wakis.unstructured.improve import edge_rates, improve_dt, tet_partials
from wakis.unstructured.quality import (
    _element_lambda,
    dihedral_angles,
    element_stability,
    lts_partition,
    tet_quality,
    top_mode_localization,
)

X01 = 2.404825557695773


def _assemble(mesh, blocks, free):
    ce = mesh.cell_edges
    r = np.repeat(ce, 6, axis=1).ravel()
    cc = np.tile(ce, (1, 6)).ravel()
    A = sp.csr_matrix((blocks.ravel(), (r, cc)), shape=(mesh.n_edges,) * 2)
    return A[free][:, free]


@pytest.fixture(scope="module")
def ball():
    # dielectric ball (eps 4) inside a PEC sphere: interfaces, Delaunay, 2.1e3 tets
    return ball_mesh([0.5, 1.0], 0.25)


MATS = {0: [4.0, 1.0], 1: [1.0, 1.0]}


# ----------------------------------------------------------------- sizes
def test_size_from_cpw():
    """h = c0 / (f n cpw); per-material sizes use each material's index."""
    assert size_from_cpw(1e9, 1.0, 10) == pytest.approx(c / 1e10, rel=1e-15)
    s = region_sizes({"a": [4.0, 1.0], "b": [1.0, 1.0], "m": "pec"}, 1e9, cpw=20)
    assert s["a"] == pytest.approx(c / (1e9 * 2 * 20), rel=1e-15)
    assert s["b"] == pytest.approx(2 * s["a"], rel=1e-15)
    assert s["m"] == s["a"]  # PEC regions take the finest dielectric size
    u = region_sizes({"a": [4.0, 1.0], "b": [1.0, 1.0]}, 1e9, 20, "uniform")
    assert u["a"] == u["b"] == s["a"]
    with pytest.raises(ValueError):
        size_from_cpw(1e9, 1.0, 0)


# ----------------------------------------------------------------- shape metrics
def test_regular_and_corner_tet_metrics():
    """Closed-form metrics of the regular and the trirectangular tetrahedron."""
    reg = SimplexMesh(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], [[0, 1, 2, 3]]
    )
    q = tet_quality(reg)
    assert q["radius_ratio"][0] == pytest.approx(1.0, abs=1e-14)
    assert q["edge_ratio"][0] == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(
        np.degrees(dihedral_angles(reg)[0]), np.degrees(np.arccos(1 / 3)), atol=1e-12
    )
    assert q["circumcenter_bary"][0] == pytest.approx(0.25, abs=1e-14)
    corner = SimplexMesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 2, 3]])
    d = np.sort(np.degrees(dihedral_angles(corner)[0]))
    np.testing.assert_allclose(
        d, [np.degrees(np.arccos(1 / np.sqrt(3)))] * 3 + [90.0] * 3, atol=1e-12
    )
    # circumcentre (1/2, 1/2, 1/2) is outside: barycentric 1 - 3/2 = -1/2
    assert tet_quality(corner)["circumcenter_bary"][0] == pytest.approx(-0.5, abs=1e-14)


def test_tet_partials_match_simplexmesh():
    """The per-tetrahedron dual measures of `improve` equal SimplexMesh's (one material)."""
    m = ball_mesh([1.0], 0.3)
    A, h, area, vol = tet_partials(m.nodes[m.cells])
    h2 = m.edge_length.mean() ** 2
    np.testing.assert_allclose(A, m.cell_dual_area, rtol=0, atol=1e-12 * h2)
    np.testing.assert_allclose(h, m.cell_dual_edge, rtol=0, atol=1e-12 * np.sqrt(h2))
    np.testing.assert_allclose(vol, m.cell_volume, rtol=1e-12)
    np.testing.assert_allclose(area, m.face_area[m.cell_faces], rtol=1e-12)


# ----------------------------------------------------------------- element split
@pytest.mark.parametrize("hodge", ["voronoi", "whitney"])
def test_element_matrices_assemble_exactly(ball, hodge):
    """sum_T K_T = K and sum_T M_T = M on the free edges, to round-off."""
    s = UnstructuredFIT3D(ball, MATS, hodge=hodge)
    st = element_stability(s)
    assert st["n_negative_faces"] == 0
    K = _assemble(ball, st["K_T"], s.free)
    M = _assemble(ball, st["M_T"], s.free)
    Mref = s.M1 if hodge == "whitney" else sp.diags(s.cap).tocsr()
    assert abs(K - s.K).max() <= 1e-12 * abs(s.K).max()
    assert abs(M - Mref).max() <= 1e-12 * abs(Mref).max()
    # Mutation (negative control): a 1 % error in one element's mass breaks it
    Mbad = st["M_T"].copy()
    Mbad[np.argmax(st["lam_T"])] *= 1.01
    assert abs(_assemble(ball, Mbad, s.free) - Mref).max() > 1e-6 * abs(Mref).max()


@pytest.mark.parametrize("hodge", ["voronoi", "whitney"])
def test_bounds_bracket_lambda_max(ball, hodge):
    """
    max_e K_ee / M_ee <= lambda_max <= max_T lambda_T (derived; the Lanczos
    estimate is accurate to its tolerance 1e-6). For the Voronoi Hodge with
    stiffness-proportional mass split the upper bound is within nu <= 6 of the
    lower one (tightness <= sqrt 6); measured 1.07 here.
    """
    s = UnstructuredFIT3D(ball, MATS, hodge=hodge)
    st = element_stability(s)
    tol = 1e-5
    assert st["lam_lower"] <= st["lam_max"] * (1 + tol)
    assert st["lam_max"] <= st["lam_bound"] * (1 + tol)
    assert st["dt_bound"] <= st["dt_max"] * (1 + tol)
    if hodge == "voronoi":
        assert st["tightness"] <= np.sqrt(6)
        assert st["lam_max"] <= st["lam_gershgorin"] * (1 + tol)
    # Negative control: a mass split that does not sum to M (x4) gives a
    # "bound" below lambda_max, which this test detects
    wrong = _element_lambda(st["K_T"], 4 * st["M_T"]).max()
    assert wrong < st["lam_max"]


def test_whitney_bound_on_distorted_mesh():
    """The Whitney element bound holds on a non-Delaunay (randomly moved) mesh."""
    m = ball_mesh([1.0], 0.3)
    rng = np.random.default_rng(3)
    X = m.nodes.copy()
    inner = np.ones(m.n_nodes, bool)
    inner[m.boundary_nodes] = False
    X[inner] += 0.12 * m.edge_length.mean() * rng.standard_normal((inner.sum(), 3))
    m2 = SimplexMesh(X, m.cells, m.cell_tags)
    assert m2.diagnostics()["negative_dual_area"] > 0  # really non-Delaunay
    s = UnstructuredFIT3D(m2, hodge="whitney")
    st = element_stability(s)
    assert st["lam_lower"] <= st["lam_max"] * (1 + 1e-5)
    assert st["lam_max"] <= st["lam_bound"] * (1 + 1e-5)


def test_voronoi_limiting_mode_is_local(ball):
    """
    The dt-limiting eigenvector of the Voronoi FIT lives on the elements of
    smallest dt_T: more than half of its element energy on the worst 1 %
    (measured 0.998 here), so dt is set by individual near-degenerate
    Voronoi faces, not by a global property of the mesh.
    """
    s = UnstructuredFIT3D(ball, MATS)
    st = element_stability(s)
    loc = top_mode_localization(s, st)
    assert loc["share_worst"][0.01] > 0.5
    assert loc["rank_of_peak"] < 0.01 * ball.n_cells


def test_lts_partition_coarse_block_is_stable(ball):
    """
    With the elements of dt_T < dt_c flagged fine, the coarse block (edges of
    no fine element) satisfies lambda_max(K_cc, M_cc) <= (2 / dt_c)^2: every
    element touching a coarse-only vector has dt_T >= dt_c (derived).
    """
    s = UnstructuredFIT3D(ball, MATS, massless_tol=1e-12)  # no Schur elimination
    st = element_stability(s)
    dt_c = 2.0 * st["dt_max"]
    fine, bad = lts_partition(s, st, dt_c)
    assert 0 < fine.mean() < 0.9
    coarse = np.flatnonzero(~fine)
    Kcc = s.K[coarse][:, coarse]
    from scipy.sparse.linalg import eigsh

    minv = 1 / np.sqrt(s.cap[coarse])
    lam = eigsh(sp.diags(minv) @ Kcc @ sp.diags(minv), k=1, which="LA",
                return_eigenvectors=False, tol=1e-8)[0]  # fmt: skip
    assert lam <= (2 / dt_c) ** 2 * (1 + 1e-6)


# ----------------------------------------------------------------- improvement
def _sphere_error(mesh, s):
    k = sphere_in_cavity(1.0, 0.5, 4.0, "all", n_modes=1)[0]
    f, _ = s.eigenmodes(1, f_target=0.7 * k * c / (2 * np.pi))
    return f[0] / (k * c / (2 * np.pi)) - 1


def test_improve_dt_ball(ball):
    """
    Node relocation on the dielectric ball: the solver's time step grows
    (baseline: the same ball_mesh, same h, solver defaults), the cells, tags,
    boundary and interface nodes are unchanged, and the first resonance
    (exact: Mie-type root of `sphere_in_cavity`) is as accurate as on the
    baseline mesh: |error| <= 2 x baseline error (same h, so the error stays
    O(h^2) with a different constant; a factor 2 is the definition of "equal
    accuracy" used throughout the study).
    """
    s0 = UnstructuredFIT3D(ball, MATS)
    new, hist = improve_dt(ball, MATS, n_sweeps=6)
    s1 = UnstructuredFIT3D(new, MATS)
    np.testing.assert_array_equal(new.cells, ball.cells)
    np.testing.assert_array_equal(new.cell_tags, ball.cell_tags)
    from wakis.unstructured.improve import movable_nodes

    fixed = ~movable_nodes(ball)
    np.testing.assert_array_equal(new.nodes[fixed], ball.nodes[fixed])
    assert hist[-1]["max_rate"] < hist[0]["max_rate"]
    gain = s1.dt_max / s0.dt_max
    assert gain > 1.1, gain
    e0, e1 = _sphere_error(ball, s0), _sphere_error(new, s1)
    assert abs(e1) <= 2 * abs(e0), (e0, e1)


def test_edge_rates_give_lambda_lower(ball):
    """The unclipped edge rates of `improve` equal K_ee / M_ee away from interfaces."""
    m = ball_mesh([1.0], 0.3)
    s = UnstructuredFIT3D(m, massless_tol=1e-12)
    r = edge_rates(m)[s.free]
    np.testing.assert_allclose(r, s.K.diagonal() / s.cap, rtol=1e-10)


# ----------------------------------------------------------------- gmsh (optional)
A_PB, L_PB = 0.1, 0.1


def _pillbox(gmsh):
    gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L_PB, A_PB)


def test_mesh_cad_pillbox_converges():
    """
    TM010 of a CAD pillbox (gmsh, Delaunay output) against j01 c / (2 pi a):
    the error falls by more than 2x from 10 to 20 cells per wavelength at
    f_max = 1.5 GHz (first order at least; measured -3.5e-3 -> -3.4e-4).
    """
    pytest.importorskip("gmsh")
    from wakis.unstructured.cad import mesh_cad

    f0 = X01 * c / (2 * np.pi * A_PB)
    err = []
    for cpw in (10, 20):
        m, info = mesh_cad(_pillbox, {"vac": (0, 0, L_PB / 2)}, {"vac": [1, 1]},
                           f_max=1.5e9, cpw=cpw, optimize=None)  # fmt: skip
        s = UnstructuredFIT3D(m, info.materials)
        f, _ = s.eigenmodes(1, f_target=0.7 * f0)
        err.append(f[0] / f0 - 1)
    assert abs(err[1]) < abs(err[0]) / 2, err


def test_step_layered_pillbox(tmp_path):
    """
    STEP round trip with two materials: a pillbox (a = d = 0.1 m) with an
    eps = 4 layer of 30 mm is written as STEP, imported by `mesh_cad` with
    point-selected regions and per-material sizes, and its TM_01 mode is
    compared with the transverse-resonance root (0.9365 GHz). At cpw 10
    (counted in the dielectric) |error| < 1.5e-2: four times the empty
    pillbox error at the same cpw, since the vacuum region is twice as
    coarse (second order). The improved mesh raises dt by > 20 % at
    |error| <= 2 x baseline.
    """
    gmsh = pytest.importorskip("gmsh")
    from wakis.unstructured.cad import mesh_cad

    a, d, t, eps = 0.1, 0.1, 0.03, 4.0
    step = str(tmp_path / "layered.step")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, t, a)
        gmsh.model.occ.addCylinder(0, 0, t, 0, 0, d - t, a)
        gmsh.model.occ.synchronize()
        gmsh.write(step)
    finally:
        gmsh.finalize()
    fx = layered_pillbox_tm0(a, d, t, eps)[0] * c / (2 * np.pi)
    regions = {"diel": (0, 0, t / 2), "vac": (0, 0, (d + t) / 2)}
    mats = {"diel": [eps, 1], "vac": [1, 1]}
    out = {}
    for imp in (None, True):
        m, info = mesh_cad(step, regions, mats, f_max=1.5e9, cpw=10,
                           optimize=None, improve=imp)  # fmt: skip
        assert info.region_tags == {"diel": 0, "vac": 1}
        # the dielectric is the layer: volume within the polygon chord error
        vd = m.cell_volume[m.cell_tags == 0].sum() / (np.pi * a**2 * t)
        assert abs(vd - 1) < 2e-2
        assert info.sizes["vac"] == pytest.approx(2 * info.sizes["diel"])
        s = UnstructuredFIT3D(m, info.materials)
        f, _ = s.eigenmodes(1, f_target=0.6 * fx)
        out[imp] = (f[0] / fx - 1, s.dt_max)
    e0, dt0 = out[None]
    e1, dt1 = out[True]
    assert abs(e0) < 1.5e-2, e0
    assert dt1 / dt0 > 1.2, dt1 / dt0
    assert abs(e1) <= 2 * abs(e0), (e0, e1)


def test_real_step_regions_by_label():
    """
    The muon-cooling cavity STEP (tests/stl/006_muonCavity.stp, mm): the two
    vacuum half cells are selected by their STEP label, copper and beryllium
    windows are dropped (their surfaces become the PEC boundary) and the
    meshed volume matches the CAD volume to the curved-surface chord error.
    """
    pytest.importorskip("gmsh")
    from wakis.unstructured.cad import mesh_cad

    m, info = mesh_cad("tests/stl/006_muonCavity.stp", {"vacuum": r"Vacuum$"},
                       {"vacuum": [1, 1]}, f_max=1e9, cpw=10, scale=1e-3,
                       optimize=None)  # fmt: skip
    assert len(info.volumes["vacuum"]) == 2
    v_cad = 2 * 3.516147158909837e-3  # m^3 (gmsh OCC mass of each half cell)
    assert abs(m.cell_volume.sum() / v_cad - 1) < 5e-2
    assert np.all(m.cell_tags == 0)
    with pytest.raises(ValueError):
        mesh_cad("tests/stl/006_muonCavity.stp", {"x": r"NoSuchLabel"},
                 {"x": [1, 1]}, f_max=1e9, cpw=10, scale=1e-3)  # fmt: skip


def test_preflight_refuses_huge_mesh():
    """A forgotten mm -> m scale is caught before meshing (estimated tets)."""
    pytest.importorskip("gmsh")
    from wakis.unstructured.cad import mesh_cad

    with pytest.raises(MemoryError):
        mesh_cad("tests/stl/006_muonCavity.stp", {"vacuum": r"Vacuum$"},
                 {"vacuum": [1, 1]}, f_max=1e9, cpw=10, scale=1.0)  # fmt: skip


def test_read_mesh_roundtrip(tmp_path):
    """A .msh written by gmsh reads back with its named groups -> materials."""
    pytest.importorskip("gmsh")
    pytest.importorskip("meshio")
    from wakis.unstructured.cad import mesh_cad, read_mesh

    fn = str(tmp_path / "pb.msh")
    m, info = mesh_cad(_pillbox, {"vac": (0, 0, L_PB / 2)}, {"vac": [1, 1]},
                       f_max=1.5e9, cpw=10, optimize=None, filename=fn)  # fmt: skip
    r, mats = read_mesh(fn, materials={"vac": [2.0, 1.0]})
    assert r.n_cells == m.n_cells
    assert r.region_names == {"vac": 1}
    assert mats == {1: [2.0, 1.0]}
    np.testing.assert_allclose(r.cell_volume.sum(), m.cell_volume.sum(), rtol=1e-12)


def test_mesh_stl_body_in_box():
    """
    A closed STL sphere (tests/stl/003_sphere.stl, mm) meshed inside a padding
    box: the body is remeshed at h = 5 mm with its nodes on the STL sphere
    (R = 25 mm), so it is inscribed: the volume deficit is at most the cap
    volume under every triangle, 3 r_c^2 / (2 R^2) with circumradius
    r_c = h / sqrt(3), i.e. h^2 / (2 R^2) = 2 % (measured 1.3 %); the box
    region fills the rest exactly.
    """
    pytest.importorskip("gmsh")
    meshio = pytest.importorskip("meshio")
    from wakis.unstructured.cad import mesh_stl

    raw = meshio.read("tests/stl/003_sphere.stl")
    P, T = raw.points.astype(float) * 1e-3, raw.cells_dict["triangle"]
    v_stl = abs(np.einsum("ij,ij->i", P[T[:, 0]], np.cross(P[T[:, 1]], P[T[:, 2]])).sum() / 6)
    m, info = mesh_stl("tests/stl/003_sphere.stl", [4, 1], f_max=3e9, cpw=10,
                       padding=15, scale=1e-3)  # fmt: skip
    vb = m.cell_volume[m.cell_tags == 0].sum()
    h, R = info.sizes["body"], 25e-3
    assert 0 < 1 - vb / v_stl < h**2 / (2 * R**2)
    lo, hi = P.min(0) - 15e-3, P.max(0) + 15e-3
    # STL coordinates are float32 (relative rounding 6e-8 per bound)
    np.testing.assert_allclose(m.cell_volume.sum(), np.prod(hi - lo), rtol=1e-6)


def test_improve_dt_repairs_non_delaunay_faces():
    """
    Faces with a negative dual length make K indefinite (exponential growth at
    any dt). On a Delaunay ball with interior nodes moved at random (some
    faces become non-Delaunay), the lexicographic objective of `improve_dt`
    removes most of them (fewer than half remain) before reducing rates.
    """
    m = ball_mesh([1.0], 0.3)
    rng = np.random.default_rng(5)
    X = m.nodes.copy()
    inner = np.ones(m.n_nodes, bool)
    inner[m.boundary_nodes] = False
    X[inner] += 0.08 * m.edge_length.mean() * rng.standard_normal((inner.sum(), 3))
    m2 = SimplexMesh(X, m.cells, m.cell_tags)
    new, hist = improve_dt(m2, n_sweeps=8)
    assert hist[0]["negative_faces"] > 5
    assert hist[-1]["negative_faces"] < hist[0]["negative_faces"] / 2, hist
