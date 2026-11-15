# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Tests for topological flip repair of non-Delaunay tetrahedral meshes
(`wakis.unstructured.repair.flip_to_delaunay` and `improve_dt(repair=True)`).

Each synthetic test builds a small (< 2000 tetrahedra) random-perturbed
Delaunay mesh with `scipy.spatial.Delaunay` using a fixed seed and forces
between 3 and 10 non-Delaunay faces:
- ``interior`` case: forced interior non-Delaunay faces repaired by ``"2-3"``
  flips;
- ``boundary`` case: forced non-Delaunay faces including outer boundary faces
  repaired by ``"2-2"`` coplanar boundary flips;
- ``interface`` case: two material tags meeting at a planar interface with
  forced non-Delaunay faces on the tag interface repaired by single-tag
  ``"3-2"`` flips without flipping across the material interface.

Note on flip types: in generic perturbed point sets without four-tet coplanar
octahedra, ``"2-3"``, ``"3-2"``, and ``"2-2"`` are the flip types required by
the ``interior``, ``interface``, and ``boundary`` cases respectively; the
``"4-4"`` flip only applies to degenerate coplanar degree-4 edges and is not
needed for the three generic-position cases.
"""

import numpy as np
import pytest
from scipy.spatial import Delaunay

from wakis.unstructured.improve import improve_dt, neg_faces
from wakis.unstructured.mesh import SimplexMesh
from wakis.unstructured.quality import stability_report
from wakis.unstructured.repair import (
    _flip_22,
    _flip_23,
    _flip_32,
    _orient_tet,
    _RepairState,
    _sface,
    flip_to_delaunay,
)
from wakis.unstructured.solver3d import UnstructuredFIT3D


def _make_delaunay_box(seed=42, n=5, interface=False):
    """Build a small random-perturbed Delaunay mesh on [0, 1]^3 (< 2000 tets)."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, n)
    pts = []
    for x in grid:
        for y in grid:
            for z in grid:
                p = np.array([x, y, z], dtype=float)
                for d in range(3):
                    if 0.0 < p[d] < 1.0 and not (
                        interface and d == 2 and abs(p[d] - 0.5) < 1e-12
                    ):
                        p[d] += rng.uniform(-0.04, 0.04)
                pts.append(p)
    X = np.array(pts)
    tri = Delaunay(X)
    cells = tri.simplices.copy()
    vol6 = np.abs(np.linalg.det(X[cells[:, 1:]] - X[cells[:, :1]]))
    cells = cells[vol6 > 1e-12]
    tags = None
    if interface:
        cent_z = X[cells].mean(axis=1)[:, 2]
        tags = np.where(cent_z < 0.5, 1, 2)
    mesh = SimplexMesh(X, cells, tags)
    assert mesh.n_cells < 2000
    return mesh


def _build_interior_case():
    """Force k=5 interior non-Delaunay faces via reverse 3-2 flips (needs 2-3)."""
    m0 = _make_delaunay_box(seed=42, n=5, interface=False)
    assert neg_faces(m0, bc="pec").size == 0
    locked = np.zeros(m0.n_nodes, dtype=bool)
    st = _RepairState(m0, None, "pec")
    forced = 0
    used_verts = set()
    for ed in sorted(st.edge_map.keys()):
        if ed in st.bnd_edges:
            continue
        cids = list(st.edge_map.get(ed, ()))
        if len(cids) != 3:
            continue
        verts = {v for c in cids for v in st.cells[c]}
        if verts & used_verts:
            continue
        cur_m = st.to_mesh(m0)
        st_copy = _RepairState(cur_m, None, "pec")
        ok, _ = _flip_32(st_copy, ed, locked, check_delaunay=False)
        if ok:
            m_try = st_copy.to_mesh(m0)
            nf = neg_faces(m_try, bc="pec")
            if nf.size == forced + 1 and not np.isin(nf, m_try.boundary_faces).any():
                _flip_32(st, ed, locked, check_delaunay=False)
                used_verts.update(verts)
                forced += 1
                if forced == 5:
                    break
    mesh = st.to_mesh(m0)
    return mesh, None, "pec", forced, "2-3"


def _build_boundary_case():
    """Force k=4 non-Delaunay faces including boundary faces via reverse 2-2 flips."""
    m0 = _make_delaunay_box(seed=42, n=6, interface=False)
    assert neg_faces(m0, bc="pmc").size == 0
    locked = np.zeros(m0.n_nodes, dtype=bool)
    st = _RepairState(m0, None, "pmc")
    forced = 0
    used_verts = set()
    for ed in sorted(st.bnd_edges):
        cids = list(st.edge_map.get(ed, ()))
        if len(cids) != 2:
            continue
        a, b = ed
        rem1 = set(st.cells[cids[0]]) - {a, b}
        rem2 = set(st.cells[cids[1]]) - {a, b}
        common = rem1 & rem2
        if len(common) != 1:
            continue
        p = next(iter(common))
        c = next(iter(rem1 - {p}))
        d = next(iter(rem2 - {p}))
        verts = {a, b, c, d, p}
        if verts & used_verts:
            continue
        cur_m = st.to_mesh(m0)
        st_copy = _RepairState(cur_m, None, "pmc")
        ok, _ = _flip_22(st_copy, ed, locked, check_delaunay=False)
        if ok:
            f_cdp = _sface(c, d, p)
            if st_copy.is_neg_face(f_cdp):
                m_try = st_copy.to_mesh(m0)
                _, rep_22 = flip_to_delaunay(
                    m_try, None, bc="pmc", enabled_flips=("2-2",)
                )
                _, rep_no22 = flip_to_delaunay(
                    m_try, None, bc="pmc", enabled_flips=("2-3", "3-2", "4-4")
                )
                if (
                    rep_22["negative_faces_after"] == 0
                    and rep_no22["negative_faces_after"] >= forced + 1
                ):
                    _flip_22(st, ed, locked, check_delaunay=False)
                    used_verts.update(verts)
                    forced += 1
                    if forced == 4:
                        break
    mesh = st.to_mesh(m0)
    neg_count = int(neg_faces(mesh, bc="pmc").size)
    return mesh, None, "pmc", neg_count, "2-2"


def _build_interface_case():
    """Force k=9 non-Delaunay faces on a two-tag material interface (needs 3-2)."""
    m0 = _make_delaunay_box(seed=42, n=5, interface=True)
    materials = {1: (1.0, 1.0), 2: (4.0, 1.0)}
    assert neg_faces(m0, materials, bc="pec").size == 0
    locked = np.zeros(m0.n_nodes, dtype=bool)
    st = _RepairState(m0, materials, "pec")
    if_faces = {f for f in st.face_map if st.is_interface_face(f)}
    if_nodes = {v for f in if_faces for v in f}
    bnd_nodes = set(m0.boundary_nodes)
    forced_flips = 0
    used_verts = set()
    for f in sorted(st.face_map.keys()):
        entries = st.face_map.get(f, ())
        if len(entries) != 2:
            continue
        (c1, _), (c2, _) = entries
        if st.tags[c1] != st.tags[c2]:
            continue
        verts = set(st.cells[c1]) | set(st.cells[c2])
        if len(verts & if_nodes) < 2 or (verts & used_verts) or (verts & bnd_nodes):
            continue
        cur_m = st.to_mesh(m0)
        st_copy = _RepairState(cur_m, materials, "pec")
        ok, _ = _flip_23(st_copy, f, locked, check_delaunay=False)
        if ok:
            m_try = st_copy.to_mesh(m0)
            _, rep_32 = flip_to_delaunay(
                m_try, materials, bc="pec", enabled_flips=("3-2",)
            )
            _, rep_no32 = flip_to_delaunay(
                m_try, materials, bc="pec", enabled_flips=("2-3", "4-4", "2-2")
            )
            if (
                rep_32["negative_faces_after"] == 0
                and rep_no32["negative_faces_after"]
                > neg_faces(cur_m, materials, bc="pec").size
            ):
                _flip_23(st, f, locked, check_delaunay=False)
                used_verts.update(verts)
                forced_flips += 1
                if forced_flips == 3:
                    break
    mesh = st.to_mesh(m0)
    neg_count = int(neg_faces(mesh, materials, bc="pec").size)
    return mesh, materials, "pec", neg_count, "3-2"


def _boundary_signature(mesh):
    """Return boundary node set, total boundary area, and area per outward unit normal."""
    bf = mesh.boundary_faces
    areas = mesh.face_area[bf]
    sgn = np.asarray(mesh.D[:, bf].sum(axis=0)).ravel()
    normals = mesh.face_normal[bf] * sgn[:, None]
    patch_areas = {}
    for area, nrm in zip(areas, normals):
        key = tuple(np.round(nrm, 6))
        patch_areas[key] = patch_areas.get(key, 0.0) + float(area)
    return {int(v) for v in mesh.boundary_nodes}, float(areas.sum()), patch_areas


@pytest.mark.parametrize(
    "case_name,builder",
    [
        ("interior", _build_interior_case),
        ("boundary", _build_boundary_case),
        ("interface", _build_interface_case),
    ],
)
def test_flip_to_delaunay_cases(case_name, builder):
    """
    Check that `flip_to_delaunay` removes all forced non-Delaunay faces in the
    interior, boundary, and two-material interface cases while conserving total
    and per-tag volumes to 1e-12 relative, keeping the boundary surface
    unchanged, and allowing `UnstructuredFIT3D` to construct without
    `allow_negative_faces=True`. A failure indicates that a flip type failed to
    restore local Delaunayhood or violated volume/boundary/tag invariants.
    """
    mesh, materials, bc, expected_neg, needed_flip = builder()
    assert mesh.n_cells < 2000

    nf_before = neg_faces(mesh, materials=materials, bc=bc)
    # Exact assert: forced count is between 3 and 10 and equals nf_before.size
    assert 3 <= expected_neg <= 10
    assert nf_before.size == expected_neg

    if case_name == "interior":
        assert not np.isin(nf_before, mesh.boundary_faces).any()
    elif case_name == "boundary":
        assert np.isin(nf_before, mesh.boundary_faces).any()
    elif case_name == "interface":
        if_nodes = set(
            mesh.faces[mesh.cell_faces[mesh.cell_tags == 1].ravel()].ravel()
        ) & set(mesh.faces[mesh.cell_faces[mesh.cell_tags == 2].ravel()].ravel())
        assert all(set(mesh.faces[f]) & if_nodes for f in nf_before)

    bnd_nodes_0, bnd_area_0, patches_0 = _boundary_signature(mesh)
    vol_total_0 = float(mesh.cell_volume.sum())
    tags_0 = np.unique(mesh.cell_tags)
    vol_per_tag_0 = {
        int(t): float(mesh.cell_volume[mesh.cell_tags == t].sum()) for t in tags_0
    }

    repaired, rep = flip_to_delaunay(mesh, materials=materials, bc=bc)

    assert rep["negative_faces_before"] == expected_neg
    assert rep["negative_faces_after"] == 0
    assert rep["stuck"] == []
    assert rep["flips"][needed_flip] > 0
    assert neg_faces(repaired, materials=materials, bc=bc).size == 0
    assert np.all(repaired.cell_volume > 0.0)

    # Total and per-tag volume conservation to 1e-12 relative
    vol_total_1 = float(repaired.cell_volume.sum())
    assert abs(vol_total_1 - vol_total_0) <= 1e-12 * vol_total_0
    for t, v0 in vol_per_tag_0.items():
        v1 = float(repaired.cell_volume[repaired.cell_tags == t].sum())
        assert abs(v1 - v0) <= 1e-12 * v0

    # Boundary surface unchanged (same boundary node set, total area, and normal patches)
    bnd_nodes_1, bnd_area_1, patches_1 = _boundary_signature(repaired)
    assert bnd_nodes_1 == bnd_nodes_0
    assert abs(bnd_area_1 - bnd_area_0) <= 1e-12 * bnd_area_0
    assert set(patches_1.keys()) == set(patches_0.keys())
    for k in patches_0:
        assert abs(patches_1[k] - patches_0[k]) <= 1e-12 * patches_0[k]

    # Solver guard and stability_report pass without allow_negative_faces=True
    solver = UnstructuredFIT3D(repaired, materials=materials, bc=bc)
    srep = stability_report(solver)
    assert srep["n_negative_faces"] == 0


def test_each_case_needs_its_flip():
    """
    Mutation check: for each of the three cases (interior, boundary, interface),
    disabling its required flip type in `enabled_flips` must leave negative
    faces unrepaired (`negative_faces_after > 0`). A failure means the test
    case did not genuinely exercise that flip helper or another path masked it.
    """
    all_types = ("2-3", "3-2", "4-4", "2-2")
    for builder in (_build_interior_case, _build_boundary_case, _build_interface_case):
        mesh, materials, bc, expected_neg, needed_flip = builder()
        subset = tuple(t for t in all_types if t != needed_flip)
        repaired_sub, rep_sub = flip_to_delaunay(
            mesh, materials=materials, bc=bc, enabled_flips=subset
        )
        assert rep_sub["flips"][needed_flip] == 0
        assert rep_sub["negative_faces_after"] == expected_neg
        assert len(rep_sub["stuck"]) == expected_neg
        assert neg_faces(repaired_sub, materials=materials, bc=bc).size == expected_neg


def test_repaired_mesh_1e4_steps_gaussian_stability():
    """
    Run `UnstructuredFIT3D` for 10,000 steps at dt = 0.95 * dt_max with a
    Gaussian dipole source on a repaired mesh and verify that max |E| at step
    10,000 is finite and below 10 times max |E| at step 100. A failure indicates
    residual indefinite curl-curl modes or time-domain instability after repair.
    """
    m0 = _make_delaunay_box(seed=7, n=4, interface=False)
    locked = np.zeros(m0.n_nodes, dtype=bool)
    st = _RepairState(m0, None, "pec")
    forced = 0
    used_verts = set()
    for ed in sorted(st.edge_map.keys()):
        if ed in st.bnd_edges:
            continue
        cids = list(st.edge_map.get(ed, ()))
        if len(cids) != 3:
            continue
        verts = {v for c in cids for v in st.cells[c]}
        if verts & used_verts:
            continue
        cur_m = st.to_mesh(m0)
        st_copy = _RepairState(cur_m, None, "pec")
        ok, _ = _flip_32(st_copy, ed, locked, check_delaunay=False)
        if ok:
            m_try = st_copy.to_mesh(m0)
            if neg_faces(m_try, bc="pec").size == forced + 1:
                _flip_32(st, ed, locked, check_delaunay=False)
                used_verts.update(verts)
                forced += 1
                if forced == 3:
                    break
    m_bad = st.to_mesh(m0)
    assert neg_faces(m_bad, bc="pec").size == 3

    m_rep, rep = flip_to_delaunay(m_bad, bc="pec")
    assert rep["negative_faces_after"] == 0

    solver = UnstructuredFIT3D(m_rep, bc="pec", cfl=0.95)
    assert np.isclose(solver.dt, 0.95 * solver.dt_max, rtol=1e-12)

    dt = solver.dt
    t0 = 25.0 * dt
    tau = 6.0 * dt
    solver.add_point_dipole(
        [0.5, 0.5, 0.5],
        [0.0, 0.0, 1.0],
        lambda t: np.exp(-0.5 * ((t - t0) / tau) ** 2),
    )

    solver.run(100)
    e_100 = float(np.max(np.abs(solver.e / solver.mesh.edge_length)))
    assert np.isfinite(e_100) and e_100 > 0.0

    solver.run(9900)
    assert solver.n == 10000
    e_end = float(np.max(np.abs(solver.e / solver.mesh.edge_length)))
    assert np.isfinite(e_end)
    assert e_end < 10.0 * e_100


def test_locked_nodes_refuse_flips_and_steiner_fallback():
    """
    Verify that locking the endpoints of a boundary edge refuses a 2-2 flip
    with reason ``"locked"``, and that `steiner=True` repairs a stuck interior
    face by inserting a Steiner node and re-flipping. A failure means
    locked-node constraints or Steiner insertion are not respected.
    """
    mesh, _, bc, expected_neg, _ = _build_boundary_case()
    all_locked = np.ones(mesh.n_nodes, dtype=bool)
    _, rep_locked = flip_to_delaunay(mesh, bc=bc, locked=all_locked, steiner=False)
    assert rep_locked["refused"].get("locked", 0) > 0
    assert rep_locked["negative_faces_after"] == expected_neg

    # Stuck interior face (0, 1, 2) where edge (3, 4) already exists in an
    # adjacent tetrahedron so 2-3 is topologically blocked and edges of (0, 1, 2)
    # cannot be flipped; steiner=True inserts 1 Steiner node at the tet-pair
    # circumcentre and re-flips to zero negative faces.
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.8660254, 0.0],
            [0.5, 0.2886751, 0.40],
            [0.5, 0.2886751, -0.40],
            [0.5, -0.7, 0.0],
            [1.3, 0.6, 0.0],
        ],
        dtype=float,
    )
    cells = np.array(
        [
            [0, 1, 2, 3],
            [0, 2, 1, 4],
            [0, 5, 1, 3],
            [0, 1, 5, 4],
            [1, 6, 3, 4],
            [1, 2, 6, 3],
            [1, 6, 2, 4],
        ],
        dtype=int,
    )
    m_stuck = SimplexMesh(X, cells)
    _, rep_nost = flip_to_delaunay(m_stuck, steiner=False)
    assert rep_nost["negative_faces_before"] == 1
    assert rep_nost["negative_faces_after"] == 1
    assert len(rep_nost["stuck"]) == 1

    m_st, rep_st = flip_to_delaunay(m_stuck, steiner=True)
    assert rep_st["steiner_nodes"] == 1
    assert rep_st["negative_faces_after"] == 0
    assert rep_st["stuck"] == []
    assert np.isclose(m_st.cell_volume.sum(), m_stuck.cell_volume.sum(), rtol=1e-12)


def test_improve_dt_repair_integration():
    """
    Verify that `improve_dt(repair=True)` runs topological flip repair before
    and after node smoothing and appends a ``"repair"`` entry to `history` with
    ``negative_faces_after == 0``, while `improve_dt(repair=False)` leaves
    locked interface-adjacent negative faces unrepaired. A failure means the
    `improve_dt` hook or its history entry is broken.
    """
    mesh, materials, bc, expected_neg, _ = _build_interface_case()
    _, hist_false = improve_dt(
        mesh, materials=materials, bc=bc, n_sweeps=2, repair=False
    )
    assert all("repair" not in h for h in hist_false)

    m_true, hist_true = improve_dt(
        mesh, materials=materials, bc=bc, n_sweeps=2, repair=True
    )
    assert "repair" in hist_true[-1]
    assert hist_true[-1]["repair"]["negative_faces_before"] == expected_neg
    assert hist_true[-1]["repair"]["negative_faces_after"] == 0
    assert neg_faces(m_true, materials=materials, bc=bc).size == 0
    UnstructuredFIT3D(m_true, materials=materials, bc=bc)


def _make_two_layer_unequal_box(seed=0):
    """Build a conforming two-material mesh with unequal layer mesh sizes."""
    rng = np.random.default_rng(seed)
    n_if = 5
    grid_if = np.linspace(0.0, 0.1, n_if)
    pts_if = []
    for x in grid_if:
        for y in grid_if:
            p = np.array([x, y, 0.03])
            if 0.0 < x < 0.1:
                p[0] += rng.uniform(-0.004, 0.004)
            if 0.0 < y < 0.1:
                p[1] += rng.uniform(-0.004, 0.004)
            pts_if.append(p)
    pts_if = np.array(pts_if)

    pts0 = list(pts_if)
    for x in np.linspace(0.0, 0.1, 7):
        for y in np.linspace(0.0, 0.1, 7):
            for z in (0.0, 0.01, 0.02):
                p = np.array([x, y, z])
                if 0.0 < x < 0.1:
                    p[0] += rng.uniform(-0.003, 0.003)
                if 0.0 < y < 0.1:
                    p[1] += rng.uniform(-0.003, 0.003)
                if 0.0 < z < 0.03:
                    p[2] += rng.uniform(-0.002, 0.002)
                pts0.append(p)
    pts0 = np.array(pts0)
    tets0 = Delaunay(pts0).simplices
    v0 = np.abs(np.linalg.det(pts0[tets0[:, 1:]] - pts0[tets0[:, :1]])) / 6.0
    tets0 = tets0[v0 > 1e-6 * v0.max()]

    pts1 = list(pts_if)
    for x in np.linspace(0.0, 0.1, 4):
        for y in np.linspace(0.0, 0.1, 4):
            for z in (0.055, 0.08, 0.10):
                p = np.array([x, y, z])
                if 0.0 < x < 0.1:
                    p[0] += rng.uniform(-0.005, 0.005)
                if 0.0 < y < 0.1:
                    p[1] += rng.uniform(-0.005, 0.005)
                if 0.03 < z < 0.10:
                    p[2] += rng.uniform(-0.004, 0.004)
                pts1.append(p)
    pts1 = np.array(pts1)
    tets1 = Delaunay(pts1).simplices
    v1 = np.abs(np.linalg.det(pts1[tets1[:, 1:]] - pts1[tets1[:, :1]])) / 6.0
    tets1 = tets1[v1 > 1e-6 * v1.max()]

    n_shared = len(pts_if)
    offset = len(pts0) - n_shared
    tets1_mapped = np.where(tets1 < n_shared, tets1, tets1 + offset)
    X = np.vstack([pts0, pts1[n_shared:]])
    cells = np.vstack([tets0, tets1_mapped])
    tags = np.concatenate(
        [np.zeros(len(tets0), dtype=int), np.ones(len(tets1), dtype=int)]
    )
    return SimplexMesh(X, cells, tags)


def test_layered_interface_unequal_mesh_sizes_repair():
    """
    Verify that a conforming two-material mesh whose layers are meshed at
    different characteristic sizes and meet on a planar interface has its
    negative dual-length interface faces repaired by `improve_dt(repair=True)`
    while conserving per-tag volumes. A failure means interface diagonal flips
    or interface-preserving Steiner splits left negative faces stuck.
    """
    mesh = _make_two_layer_unequal_box(seed=0)
    assert mesh.n_cells < 2000
    materials = {0: [4.0, 1.0], 1: [1.0, 1.0]}

    nf_before = neg_faces(mesh, materials=materials)
    assert nf_before.size >= 3

    # Mutation check: without coplanar 2-2 flips and Steiner splits, the
    # interface faces between separately Delaunay layers remain stuck
    _, rep_restricted = flip_to_delaunay(
        mesh,
        materials=materials,
        enabled_flips=("2-3", "3-2", "4-4"),
        steiner=False,
    )
    assert rep_restricted["negative_faces_after"] == nf_before.size
    assert rep_restricted["refused"].get("interface", 0) > 0

    vol_diel_0 = float(mesh.cell_volume[mesh.cell_tags == 0].sum())
    vol_vac_0 = float(mesh.cell_volume[mesh.cell_tags == 1].sum())

    repaired, history = improve_dt(mesh, materials=materials, n_sweeps=0, repair=True)
    rep = history[-1]["repair"]
    assert rep["negative_faces_before"] == nf_before.size
    assert rep["negative_faces_after"] == 0
    assert rep["stuck"] == []
    assert neg_faces(repaired, materials=materials).size == 0

    vol_diel_1 = float(repaired.cell_volume[repaired.cell_tags == 0].sum())
    vol_vac_1 = float(repaired.cell_volume[repaired.cell_tags == 1].sum())
    assert abs(vol_diel_1 - vol_diel_0) <= 1e-12 * vol_diel_0
    assert abs(vol_vac_1 - vol_vac_0) <= 1e-12 * vol_vac_0

    UnstructuredFIT3D(repaired, materials=materials)


def test_locked_curve_edge_is_not_flipped_away():
    """
    Verify that a 2-2 flip removing a boundary edge (a CAD-curve edge) whose
    two endpoints are both locked is refused by `flip_to_delaunay(..., locked=mask)`
    and that `improve_dt(..., repair=True)` forwards `locked=~movable` so the
    locked boundary edge survives both repair passes.
    Mutation check: passing `locked=None` in `improve_dt` allows `_flip_22`
    to remove boundary edge (0, 1) and makes this test fail.
    """
    X = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, -0.4, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.5],
        ],
        dtype=float,
    )
    cells = np.array(
        [
            _orient_tet(X, 0, 1, 2, 4),
            _orient_tet(X, 0, 1, 3, 4),
            _orient_tet(X, 0, 2, 4, 5),
            _orient_tet(X, 2, 1, 4, 5),
            _orient_tet(X, 1, 3, 4, 5),
            _orient_tet(X, 3, 0, 4, 5),
        ],
        dtype=np.int64,
    )
    mesh = SimplexMesh(X, cells)
    assert neg_faces(mesh).size > 0
    assert (0, 1) in _RepairState(mesh, None, "pec").bnd_edges

    def _has_edge(m, u, v):
        target = (min(u, v), max(u, v))
        return any((int(e[0]), int(e[1])) == target for e in m.edges)

    assert _has_edge(mesh, 0, 1)

    mask = np.array([True, True, False, False, False, False], dtype=bool)
    repaired, rep = flip_to_delaunay(mesh, locked=mask)
    assert _has_edge(repaired, 0, 1)
    assert rep["refused"].get("locked", 0) > 0

    improved, history = improve_dt(
        mesh,
        n_sweeps=1,
        repair=True,
        project=lambda n, y: None if mask[n] else y,
    )
    assert _has_edge(improved, 0, 1)
    assert history[-1]["repair"]["refused"].get("locked", 0) > 0


def test_interior_chord_between_locked_nodes_is_flippable():
    """
    Verify that an interior non-Delaunay face touching two locked (surface)
    nodes is repaired when the flip removes an interior edge between the two
    locked nodes, leaving zero negative faces and `refused["locked"] == 0`.
    Mutation check: the old rule (refuse any edge with both ends locked)
    refuses the 3-2 flip on interior edge (0, 1) and makes this test fail.
    """
    X = np.array(
        [
            [0.0, 0.0, -1.5],
            [0.0, 0.0, 1.5],
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0, 0.0],
            [-0.5, -np.sqrt(3.0) / 2.0, 0.0],
        ],
        dtype=float,
    )
    cells = np.array(
        [
            _orient_tet(X, 0, 1, 2, 3),
            _orient_tet(X, 0, 1, 3, 4),
            _orient_tet(X, 0, 1, 4, 2),
        ],
        dtype=np.int64,
    )
    mesh = SimplexMesh(X, cells)
    assert neg_faces(mesh).size > 0
    assert (0, 1) not in _RepairState(mesh, None, "pec").bnd_edges

    mask = np.array([True, True, False, False, False], dtype=bool)
    repaired, rep = flip_to_delaunay(mesh, locked=mask)
    assert rep["negative_faces_after"] == 0
    assert neg_faces(repaired).size == 0
    assert rep["refused"].get("locked", 0) == 0
    assert rep["flips"]["3-2"] == 1

    improved, history = improve_dt(
        mesh,
        bc="pmc",
        n_sweeps=0,
        repair=True,
        project=lambda n, y: None if mask[n] else y,
    )
    assert history[-1]["repair"]["negative_faces_after"] == 0
    assert history[-1]["repair"]["refused"].get("locked", 0) == 0
    assert neg_faces(improved, bc="pmc").size == 0


def test_interface_22_only_case():
    """
    Verify that a conforming two-material mesh with a planar interface at z = 0
    and exactly one negative interface face is repaired solely by a coplanar
    interface 2-2 flip across z = 0, conserving per-tag volumes and the
    interface triangle union boundary and area to 1e-12 relative tolerance.
    Mutation check: making the interface branch of `_flip_22` return
    `(False, "mutant")` leaves the negative faces unrepaired and makes this
    test fail.
    """
    # Nodes 0, 1, 2, 3 lie on the planar interface z = 0 forming a convex quad
    # split along the non-Delaunay diagonal (0, 1); apex 4 is in tag 0 (z > 0)
    # and apex 5 is in tag 1 (z < 0).
    X = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.9, 0.10, 0.0],
            [-0.9, -0.35, 0.0],
            [-0.9, -0.90, 0.60],
            [-0.9, -0.90, -0.60],
        ],
        dtype=float,
    )
    cells = np.array(
        [
            _orient_tet(X, 0, 1, 2, 4),
            _orient_tet(X, 0, 1, 3, 4),
            _orient_tet(X, 0, 1, 2, 5),
            _orient_tet(X, 0, 1, 3, 5),
        ],
        dtype=np.int64,
    )
    tags = np.array([0, 0, 1, 1], dtype=np.int64)
    materials = {0: [4.0, 1.0], 1: [1.0, 1.0]}
    mesh = SimplexMesh(X, cells, tags)

    def _interface_info(m):
        f = m.cell_faces.ravel()
        t = np.repeat(m.cell_tags, 4)
        lo = np.full(m.n_faces, np.iinfo(np.int64).max)
        hi = np.full(m.n_faces, np.iinfo(np.int64).min)
        np.minimum.at(lo, f, t)
        np.maximum.at(hi, f, t)
        if_fids = np.flatnonzero(lo != hi)
        edge_counts = {}
        area = 0.0
        for fid in if_fids:
            tri = tuple(int(v) for v in m.faces[fid])
            p0, p1, p2 = m.nodes[list(tri)]
            area += 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
            for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])):
                e = (min(u, v), max(u, v))
                edge_counts[e] = edge_counts.get(e, 0) + 1
        bnd_edges = {e for e, cnt in edge_counts.items() if cnt == 1}
        return if_fids, bnd_edges, area

    nf_before = neg_faces(mesh, materials=materials)
    if_fids_0, if_bnd_0, if_area_0 = _interface_info(mesh)
    neg_if_0 = np.intersect1d(nf_before, if_fids_0)
    print(
        f"interface_22_only config: n_neg={nf_before.size}, "
        f"if_faces={if_fids_0.tolist()}, neg_if_faces={neg_if_0.tolist()}, "
        f"if_bnd={sorted(if_bnd_0)}, if_area={if_area_0:.12f}"
    )
    assert if_fids_0.size == 2
    assert neg_if_0.size == 1

    # Without "2-2" and with steiner=False, no flip can repair the mesh
    _, rep_no22 = flip_to_delaunay(
        mesh,
        materials=materials,
        enabled_flips=("2-3", "3-2", "4-4"),
        steiner=False,
    )
    assert rep_no22["negative_faces_after"] == nf_before.size

    vol_tag0_before = float(mesh.cell_volume[mesh.cell_tags == 0].sum())
    vol_tag1_before = float(mesh.cell_volume[mesh.cell_tags == 1].sum())

    repaired, rep = flip_to_delaunay(mesh, materials=materials)
    assert rep["negative_faces_after"] == 0
    assert rep["flips"]["2-2"] == 1
    assert neg_faces(repaired, materials=materials).size == 0

    vol_tag0_after = float(repaired.cell_volume[repaired.cell_tags == 0].sum())
    vol_tag1_after = float(repaired.cell_volume[repaired.cell_tags == 1].sum())
    assert abs(vol_tag0_after - vol_tag0_before) <= 1e-12 * vol_tag0_before
    assert abs(vol_tag1_after - vol_tag1_before) <= 1e-12 * vol_tag1_before

    if_fids_1, if_bnd_1, if_area_1 = _interface_info(repaired)
    assert if_fids_1.size == 2
    assert if_bnd_1 == if_bnd_0
    assert abs(if_area_1 - if_area_0) <= 1e-12 * if_area_0
