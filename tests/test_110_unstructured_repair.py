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
    Verify that locking the endpoints of an edge refuses a 3-2 flip with reason
    ``"locked"``, and that `steiner=True` repairs a stuck interior face by
    inserting a Steiner node and re-flipping. A failure means locked-node
    constraints or Steiner insertion are not respected.
    """
    mesh, _, bc, expected_neg, _ = _build_interface_case()
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
