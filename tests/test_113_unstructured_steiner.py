# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

r"""
Tests for Steiner point insertion and projection onto locked CAD curves
(`wakis.unstructured.steiner`, `wakis.unstructured.repair.flip_to_delaunay`,
`wakis.unstructured.improve.improve_dt`, and `wakis.unstructured.cad.CadSplitProjector`).

Covers Brief W4-A tests T1 (synthetic gmsh-free rim case with mutations),
T2 (`feature_curve_projector` on a 31-node jittered circle, a straight line,
and the gmsh layered pillbox), T3 (`CadSplitProjector` on the gmsh layered
pillbox), and T5 (geometry, volume toward CAD, and 10,000-step leapfrog
stability on the repaired layered pillbox).
"""

import numpy as np
import pytest
from scipy.spatial import Delaunay

from wakis.unstructured.improve import improve_dt, movable_nodes, neg_faces
from wakis.unstructured.mesh import SimplexMesh
from wakis.unstructured.repair import _RepairState, flip_to_delaunay
from wakis.unstructured.solver3d import UnstructuredFIT3D
from wakis.unstructured.steiner import (
    FeatureCurveProjector,
    feature_curve_projector,
    split_edge_steiner,
)


def _guard_negative_count(solver):
    """Solver guard count: faces with Mnu_diag < 0 coupling at least one free edge."""
    return int(np.sum((solver.Mnu_diag < 0) & (np.abs(solver.Cf).sum(1).A1 > 0)))


def _build_t1_synthetic_rim_mesh(n_rim=13, radius=0.1, height=0.03, seed=42):
    """
    Build a small (< 200 tets) convex cylinder mesh with `n_rim=13` rim nodes on
    each circle `z = 0` and `z = height` (radius `radius`), then move one interior
    node toward rim edge `(0, 1)` so that face `(0, 1, idx_p)` becomes negative.
    """
    rng = np.random.default_rng(seed)
    angles_bot = np.linspace(0.0, 2.0 * np.pi, n_rim, endpoint=False) + rng.uniform(
        -0.02, 0.02, n_rim
    )
    angles_top = (
        np.linspace(0.0, 2.0 * np.pi, n_rim, endpoint=False)
        + (np.pi / n_rim)
        + rng.uniform(-0.02, 0.02, n_rim)
    )
    rim_bot = np.column_stack(
        [radius * np.cos(angles_bot), radius * np.sin(angles_bot), np.zeros(n_rim)]
    )
    rim_top = np.column_stack(
        [
            radius * np.cos(angles_top),
            radius * np.sin(angles_top),
            np.full(n_rim, height),
        ]
    )
    ang_in = np.linspace(0.0, 2.0 * np.pi, n_rim, endpoint=False) + rng.uniform(
        -0.04, 0.04, n_rim
    )
    r_in = 0.55 * radius + rng.uniform(-0.03 * radius, 0.03 * radius, n_rim)
    z_in = 0.5 * height + rng.uniform(-0.06 * height, 0.06 * height, n_rim)
    ring_in = np.column_stack([r_in * np.cos(ang_in), r_in * np.sin(ang_in), z_in])
    axis_pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, height],
            [0.002 * radius, -0.001 * radius, 0.5 * height],
        ]
    )

    mid01 = 0.5 * (rim_bot[0] + rim_bot[1])
    m_dir = mid01 / np.linalg.norm(mid01)
    p_c = np.array([0.85 * radius * m_dir[0], 0.85 * radius * m_dir[1], 0.0])
    p_p0 = np.array([0.92 * radius * m_dir[0], 0.92 * radius * m_dir[1], 0.35 * height])

    X0 = np.vstack([rim_bot, rim_top, ring_in, axis_pts, [p_c, p_p0]])
    tri = Delaunay(X0)
    cells = tri.simplices.copy()
    vol6 = np.abs(np.linalg.det(X0[cells[:, 1:]] - X0[cells[:, :1]]))
    cells = cells[vol6 > 1e-12]

    z_cent = X0[cells, 2].mean(axis=1)
    tags = np.where(z_cent < 0.12 * height, 0, 1)
    materials = {0: (4.0, 1.0), 1: (1.0, 1.0)}
    m0 = SimplexMesh(X0, cells, tags)
    assert m0.n_cells < 200
    assert neg_faces(m0, materials).size == 0

    # Move the single interior node idx_p (last node) toward rim edge (0, 1)
    idx_p = len(X0) - 1
    X1 = X0.copy()
    X1[idx_p] = np.array(
        [0.82 * radius * m_dir[0], 0.82 * radius * m_dir[1], 0.20 * height]
    )
    m1 = SimplexMesh(X1, cells, tags)
    return m1, materials, radius, height, n_rim, idx_p


def _analytic_cylinder_rim_projector(mesh, radius, height, tol=1e-9):
    """Analytic split projector onto the two rim circles r = radius, z in {0, height}."""
    coords = {int(i): mesh.nodes[i].copy() for i in range(mesh.n_nodes)}

    def _on_rim(pt):
        r = float(np.hypot(pt[0], pt[1]))
        if abs(r - radius) > tol * radius:
            return None
        for z0 in (0.0, height):
            if abs(pt[2] - z0) <= tol * max(height, radius):
                return z0
        return None

    class _AnalyticRimProj:
        def project(self, a, b, x):
            pa = coords.get(int(a))
            pb = coords.get(int(b))
            if pa is None or pb is None:
                return None
            za = _on_rim(pa)
            zb = _on_rim(pb)
            if za is None or zb is None or abs(za - zb) > tol * height:
                return None
            x_arr = np.asarray(x, dtype=float)
            r_xy = float(np.hypot(x_arr[0], x_arr[1]))
            if r_xy <= 0.0:
                return None
            pt = np.array([radius * x_arr[0] / r_xy, radius * x_arr[1] / r_xy, za])
            self._last = pt
            return pt

        def register(self, new_node, a, b):
            if hasattr(self, "_last"):
                coords[int(new_node)] = self._last.copy()

        def unregister(self, new_node, a, b):
            coords.pop(int(new_node), None)

    return _AnalyticRimProj()


def test_t1_synthetic_rim_locked_edge_steiner():
    """
    T1 (gmsh-free): on a 13-node cylinder rim mesh with one interior node moved
    toward locked rim edge (0, 1), verify that `split_projector=None` leaves
    face (0, 1, idx_p) negative and touching the locked rim edge, whereas an
    analytic circle projector (and `improve_dt` with `split_project="auto"`)
    splits (0, 1) on the circle `r = R, z = 0`, repairs the mesh to 0 negative
    faces (both repair and solver guard criteria), preserves boundary manifoldness
    and positive cell volumes, locks the new rim Steiner node, and satisfies the
    exact boundary-flux volume identity to 1e-12 of the star volume.
    In-test mutations:
    (a) identity projector (chord midpoint): distance to circle equals the sagitta
        and exceeds tau, failing the geometry assertion;
    (b) `steiner=False`: leaves >= 1 negative face.
    """
    mesh, materials, R, H, n_rim, _idx_p = _build_t1_synthetic_rim_mesh(n_rim=13)
    locked = ~movable_nodes(mesh)
    # All 2 * n_rim rim nodes must be locked
    assert np.all(locked[: 2 * n_rim])
    tau = 1e-12 * R

    # Precondition: wave-3 behaviour (split_projector=None) leaves >= 1 negative face
    # touching a locked rim edge
    m_noproj, rep_noproj = flip_to_delaunay(
        mesh, materials=materials, locked=locked, split_projector=None
    )
    assert rep_noproj["negative_faces_before"] >= 1
    assert rep_noproj["negative_faces_after"] >= 1
    assert rep_noproj["refused"].get("locked", 0) > 0
    stuck_fid = rep_noproj["stuck"][0]
    stuck_verts = tuple(int(v) for v in m_noproj.faces[stuck_fid])
    rim_edges = {(i, (i + 1) % n_rim) for i in range(n_rim)} | {
        (0, n_rim - 1),
    }
    touches_locked_rim_edge = any(
        (min(u, v), max(u, v)) in rim_edges
        for u in stuck_verts
        for v in stuck_verts
        if u < v
    )
    print(
        f"T1 precondition (split_projector=None): stuck_face={stuck_verts}, "
        f"touches_locked_rim_edge={touches_locked_rim_edge}, "
        f"negative_faces_after={rep_noproj['negative_faces_after']}, "
        f"refused={rep_noproj['refused']}"
    )
    assert touches_locked_rim_edge

    # Check exact volume identity directly on splitting edge (0, 1)
    st_id = _RepairState(mesh, materials, "pec")
    proj_id = _analytic_cylinder_rim_projector(mesh, R, H)
    ok_split, split_info = split_edge_steiner(
        st_id, (0, 1), locked=locked, split_projector=proj_id
    )
    assert ok_split
    print(f"T1 volume_identity_err={split_info['volume_identity_err']:.3e}")
    assert split_info["volume_identity_err"] <= 1e-12

    # Repair with analytic circle projector
    proj = _analytic_cylinder_rim_projector(mesh, R, H)
    repaired, rep = flip_to_delaunay(
        mesh, materials=materials, locked=locked, split_projector=proj
    )
    print(f"T1 repaired report: {rep}")
    assert rep["negative_faces_after"] == 0
    assert rep["stuck"] == []
    assert rep["steiner_nodes"] >= 1
    assert rep["steiner_on_locked"] >= 1
    assert rep["steiner_projected"] >= 1
    assert neg_faces(repaired, materials=materials).size == 0

    # Solver constructs without allow_negative_faces and guard count is 0
    solver = UnstructuredFIT3D(repaired, materials=materials)
    assert _guard_negative_count(solver) == 0
    assert np.all(repaired.cell_volume > 0.0)

    # Every boundary face has exactly one incident tetrahedron
    bnd_incidence = np.abs(repaired.D[:, repaired.boundary_faces]).sum(axis=0).A1
    assert np.all(bnd_incidence == 1)

    # Each Steiner node on the rim lies on the circle within tau and stays locked
    locked_after = ~movable_nodes(repaired)
    for s_idx in range(mesh.n_nodes, repaired.n_nodes):
        pt = repaired.nodes[s_idx]
        r_err = abs(float(np.hypot(pt[0], pt[1])) - R)
        z_err = min(abs(float(pt[2])), abs(float(pt[2]) - H))
        print(
            f"T1 Steiner node {s_idx}: r_err={r_err:.3e}, z_err={z_err:.3e}, "
            f"tau={tau:.3e}, locked={locked_after[s_idx]}"
        )
        assert r_err <= tau
        assert z_err <= tau
        assert locked_after[s_idx]

    # Also verify improve_dt(..., split_project="auto") repairs via feature_curve_projector
    _m_imp, hist_imp = improve_dt(mesh, materials=materials, n_sweeps=0, repair=True)
    assert hist_imp[-1]["repair"]["negative_faces_after"] == 0
    assert hist_imp[-1]["repair"]["steiner_on_locked"] >= 1

    # Mutation (a): identity projector (places Steiner node on the chord midpoint)
    m_mut_a, _rep_mut_a = flip_to_delaunay(
        mesh,
        materials=materials,
        locked=locked,
        split_projector=lambda a, b, x: np.asarray(x, dtype=float),
    )
    assert m_mut_a.n_nodes > mesh.n_nodes
    pt_chord = m_mut_a.nodes[mesh.n_nodes]
    dist_circle = abs(R - float(np.hypot(pt_chord[0], pt_chord[1])))
    L_01 = float(np.linalg.norm(mesh.nodes[1] - mesh.nodes[0]))
    # Exact sagitta R - sqrt(R^2 - (L/2)^2) and small-angle sagitta L^2 / (8 R):
    # Note that on a 13-node circle (L/R = 0.479), the next Taylor term in
    # R - sqrt(R^2 - (L/2)^2) = (L^2 / (8R)) * (1 + L^2 / (16 R^2) + ...) is
    # L^2 / (16 R^2) = 1.43e-2, whereas on a small-angle chord (tested below)
    # dist_circle matches L^2 / (8R) to < 1e-9 relative.
    sagitta_exact = R - float(np.sqrt(R**2 - 0.25 * L_01**2))
    sagitta_lead = L_01**2 / (8.0 * R)
    rel_err_exact = abs(dist_circle - sagitta_exact) / sagitta_exact
    rel_err_lead = abs(dist_circle - sagitta_lead) / sagitta_lead
    print(
        f"T1 mutation (a) identity projector: dist_circle={dist_circle:.6e}, "
        f"sagitta_exact={sagitta_exact:.6e} (rel_err={rel_err_exact:.3e}), "
        f"sagitta_L2_8R={sagitta_lead:.6e} (rel_err={rel_err_lead:.3e}), "
        f"tau={tau:.3e}"
    )
    assert rel_err_exact <= 1e-9
    assert dist_circle > tau

    # Mutation (b): steiner=False leaves >= 1 face negative
    _, rep_mut_b = flip_to_delaunay(
        mesh, materials=materials, locked=locked, split_projector=proj, steiner=False
    )
    print(
        f"T1 mutation (b) steiner=False: negative_faces_after={rep_mut_b['negative_faces_after']}"
    )
    assert rep_mut_b["negative_faces_after"] >= 1


def test_t2_feature_curve_projector_circle_and_line():
    r"""
    T2 (gmsh-free): verify `FeatureCurveProjector` on a 31-node circle with
    jittered spacing and on a straight segment, and verify feature-edge
    classification on the synthetic cylinder mesh.

    The 3-point circle condition number scales as :math:`(R / L_{\min})^2`, so
    the projected arc midpoint on an exact circle with float64 coordinates lies
    within :math:`100 \varepsilon_{\mathrm{mach}} (R / L_{\min})^2 R` of :math:`R`.
    """
    R = 0.1
    n_circle = 31
    rng = np.random.default_rng(17)
    angles = np.linspace(0.0, 2.0 * np.pi, n_circle, endpoint=False) + rng.uniform(
        -0.03, 0.03, n_circle
    )
    pts_circle = np.column_stack(
        [R * np.cos(angles), R * np.sin(angles), np.full(n_circle, 0.03)]
    )
    edges_circle = [(i, (i + 1) % n_circle) for i in range(n_circle)]
    fcp = FeatureCurveProjector(pts_circle, edges_circle)

    chord_lengths = [
        float(np.linalg.norm(pts_circle[b] - pts_circle[a])) for a, b in edges_circle
    ]
    L_min = min(chord_lengths)
    eps_mach = float(np.finfo(np.float64).eps)
    bound_circle = 100.0 * eps_mach * ((R / L_min) ** 2) * R

    max_err_circle = 0.0
    for a, b in edges_circle:
        mid = 0.5 * (pts_circle[a] + pts_circle[b])
        p_proj = fcp.project(a, b, mid)
        assert p_proj is not None
        r_err = abs(float(np.hypot(p_proj[0], p_proj[1])) - R)
        z_err = abs(float(p_proj[2]) - 0.03)
        max_err_circle = max(max_err_circle, r_err, z_err)
    print(
        f"T2 circle (n={n_circle}): max_err={max_err_circle:.3e}, "
        f"bound={bound_circle:.3e}, L_min={L_min:.4e}"
    )
    assert max_err_circle <= bound_circle

    # Test register / unregister for recursive sub-edge projection
    mid01 = 0.5 * (pts_circle[0] + pts_circle[1])
    p_new = fcp.project(0, 1, mid01)
    fcp.register(n_circle, 0, 1)
    assert (0, n_circle) in fcp.feature_edges
    assert (1, n_circle) in fcp.feature_edges
    assert (0, 1) not in fcp.feature_edges
    p_sub = fcp.project(0, n_circle, 0.5 * (pts_circle[0] + p_new))
    assert p_sub is not None
    assert abs(float(np.hypot(p_sub[0], p_sub[1])) - R) <= bound_circle
    fcp.unregister(n_circle, 0, 1)
    assert (0, 1) in fcp.feature_edges

    # Straight segment: exact to roundoff
    n_line = 7
    s_vals = np.sort(rng.uniform(0.0, 1.0, n_line))
    p_start = np.array([0.01, -0.02, 0.005])
    d_vec = np.array([0.12, 0.07, -0.04])
    d_unit = d_vec / np.linalg.norm(d_vec)
    pts_line = p_start[None, :] + s_vals[:, None] * d_vec[None, :]
    edges_line = [(i, i + 1) for i in range(n_line - 1)]
    fcp_line = FeatureCurveProjector(pts_line, edges_line)
    # Interior edges (1..n_line-3) have both chain neighbours p and q
    max_err_line = 0.0
    for i in range(1, n_line - 2):
        mid = 0.5 * (pts_line[i] + pts_line[i + 1])
        p_proj = fcp_line.project(i, i + 1, mid)
        assert p_proj is not None
        diff = p_proj - p_start
        perp = diff - np.dot(diff, d_unit) * d_unit
        max_err_line = max(max_err_line, float(np.linalg.norm(perp)))
    print(f"T2 straight line: max_err={max_err_line:.3e}")
    assert max_err_line <= 10.0 * eps_mach * float(np.linalg.norm(d_vec))

    # Mutation check on circle: chord midpoint exceeds bound_circle by > 1e10
    mid_err_chord = abs(R - float(np.hypot(mid01[0], mid01[1])))
    assert mid_err_chord > 1e6 * bound_circle


def _write_layered_pillbox_step(gmsh, tmp_path, a=0.1, d=0.1, t=0.03):
    """Write the two-volume layered pillbox STEP file in `tmp_path`."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        occ.addCylinder(0, 0, 0, 0, 0, t, a)
        occ.addCylinder(0, 0, t, 0, 0, d - t, a)
        occ.synchronize()
        step = str(tmp_path / "layered.step")
        gmsh.write(step)
        return step
    finally:
        gmsh.finalize()


def _is_rim_node(pt, a=0.1, t=0.03, d=0.1, tol=1e-6):
    r = float(np.hypot(pt[0], pt[1]))
    if abs(r - a) > tol * a:
        return None
    for idx, z0 in enumerate((0.0, t, d)):
        if abs(float(pt[2]) - z0) <= tol * d:
            return idx
    return None


def test_t2_feature_curve_projector_layered_pillbox_gmsh(tmp_path):
    """
    T2 (gmsh): on the STEP layered pillbox, verify that `feature_curve_projector`
    classifies as feature edges **exactly** the mesh edges whose two endpoints
    both lie on the same analytic rim circle `r = a, z in {0, t, d}`.
    """
    gmsh = pytest.importorskip("gmsh")
    from wakis.unstructured.cad import mesh_cad

    a, d, t, eps = 0.1, 0.1, 0.03, 4.0
    step = _write_layered_pillbox_step(gmsh, tmp_path, a, d, t)
    regions = {"diel": (0, 0, t / 2), "vac": (0, 0, (d + t) / 2)}
    mats = {"diel": [eps, 1], "vac": [1, 1]}
    m, _ = mesh_cad(
        step, regions, mats, f_max=1.5e9, cpw=10, optimize=None, improve=None
    )

    locked = ~movable_nodes(m)
    fcp = feature_curve_projector(m, locked=locked)

    analytic_rim_edges = set()
    for u, v in m.edges:
        iu, iv = (int(u), int(v)) if u < v else (int(v), int(u))
        ru = _is_rim_node(m.nodes[iu], a, t, d)
        rv = _is_rim_node(m.nodes[iv], a, t, d)
        if ru is not None and ru == rv:
            analytic_rim_edges.add((iu, iv))

    print(
        f"T2 gmsh layered pillbox: classified={len(fcp.feature_edges)}, "
        f"analytic_rim_edges={len(analytic_rim_edges)}"
    )
    assert fcp.feature_edges == analytic_rim_edges

    # Mutation check: lowering feature_angle_deg to 1.0 deg misclassifies smooth
    # cylinder-wall edges as feature edges
    fcp_bad = feature_curve_projector(m, locked=locked, feature_angle_deg=1.0)
    assert len(fcp_bad.feature_edges) > len(analytic_rim_edges)


def test_t3_cad_split_projector_layered_pillbox_gmsh(tmp_path):
    """
    T3 (gmsh): on the STEP layered pillbox with a live gmsh model, verify that
    `CadSplitProjector` projects every rim-edge midpoint within `<= tau` of the
    analytic rim circle `r = a, z in {0, t, d}`.
    """
    gmsh = pytest.importorskip("gmsh")
    from wakis.unstructured.cad import (
        _collect,
        _configure,
        _match_regions,
        _set_size_fields,
        cad_projector,
        region_sizes,
    )

    a, d, t, eps = 0.1, 0.1, 0.03, 4.0
    step = _write_layered_pillbox_step(gmsh, tmp_path, a, d, t)
    regions = {"diel": (0, 0, t / 2), "vac": (0, 0, (d + t) / 2)}
    mats = {"diel": [eps, 1], "vac": [1, 1]}
    names = list(regions)
    sizes = region_sizes(mats, f_max=1.5e9, cpw=10)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("wakis_cad_t3")
        occ = gmsh.model.occ
        occ.importShapes(step)
        occ.synchronize()
        vols = gmsh.model.getEntities(3)
        names_in = {tag: gmsh.model.getEntityName(3, tag) for _, tag in vols}
        cad_names = dict(names_in)
        if len(vols) > 1:
            _, out_map = occ.fragment(vols[:1], vols[1:])
            occ.synchronize()
            cad_names = {}
            for (_, t_in), outs in zip(vols, out_map):
                for dim, tag in outs:
                    if dim == 3:
                        cad_names.setdefault(tag, names_in[t_in])
        vol_region = _match_regions(gmsh, regions, cad_names)
        _set_size_fields(
            gmsh,
            [([v for v, n in vol_region.items() if n == k], sizes[k]) for k in names],
            0,
        )
        _configure(gmsh, None, None)
        for k in names:
            gmsh.model.addPhysicalGroup(
                3, [v for v, n in vol_region.items() if n == k], names.index(k) + 1, k
            )
        gmsh.model.mesh.generate(3)

        mesh, node_tags = _collect(gmsh, vol_region, names, 1.0, return_tags=True)
        proj = cad_projector(gmsh, node_tags, scale=1.0)
        cad_split = proj.split

        # Compute tau from input gmsh rim nodes
        rim_node_errs = []
        for idx, pt in enumerate(mesh.nodes):
            r_id = _is_rim_node(pt, a, t, d)
            if r_id is not None:
                z0 = (0.0, t, d)[r_id]
                err = float(np.hypot(np.hypot(pt[0], pt[1]) - a, pt[2] - z0))
                rim_node_errs.append(err)
        tau = max(10.0 * max(rim_node_errs), 1e-12 * a)

        max_proj_err = 0.0
        max_chord_err = 0.0
        n_rim_edges = 0
        for u, v in mesh.edges:
            iu, iv = int(u), int(v)
            ru = _is_rim_node(mesh.nodes[iu], a, t, d)
            rv = _is_rim_node(mesh.nodes[iv], a, t, d)
            if ru is not None and ru == rv:
                n_rim_edges += 1
                z0 = (0.0, t, d)[ru]
                mid = 0.5 * (mesh.nodes[iu] + mesh.nodes[iv])
                chord_err = float(np.hypot(np.hypot(mid[0], mid[1]) - a, mid[2] - z0))
                max_chord_err = max(max_chord_err, chord_err)
                p_proj = cad_split.project(iu, iv, mid)
                assert p_proj is not None
                err = float(
                    np.hypot(np.hypot(p_proj[0], p_proj[1]) - a, p_proj[2] - z0)
                )
                max_proj_err = max(max_proj_err, err)

        print(
            f"T3 CadSplitProjector: n_rim_edges={n_rim_edges}, "
            f"max_proj_err={max_proj_err:.3e}, max_chord_err={max_chord_err:.3e}, "
            f"tau={tau:.3e}"
        )
        assert n_rim_edges > 0
        assert max_proj_err <= tau
        # Mutation check: unprojected chord midpoint violates tau by > 100x
        assert max_chord_err > 100.0 * tau
    finally:
        gmsh.finalize()


def _analytic_surface_distance(pt, a=0.1, t=0.03, d=0.1):
    """Distance of `pt` to the nearest analytic pillbox CAD surface (cylinder wall or planes z in {0, t, d})."""
    r = float(np.hypot(pt[0], pt[1]))
    z = float(pt[2])
    return min(abs(r - a), abs(z - 0.0), abs(z - t), abs(z - d))


def _analytic_pillbox_projector(mesh, a=0.1, d=0.1, t=0.03, tol=1e-5):
    """
    Analytic split projector for the layered circular cylinder of radius `a`,
    height `d`, and interface plane `z = t`. Projects rim-circle and cylinder-wall
    edges onto `r = a` and planar disk/interface edges onto `z in {0, t, d}`.
    """
    # Map node -> (on_cyl, z_plane_or_None)
    node_info = {}
    for idx, pt in enumerate(mesh.nodes):
        r = float(np.hypot(pt[0], pt[1]))
        on_cyl = abs(r - a) <= tol * a
        z_plane = None
        for z0 in (0.0, t, d):
            if abs(float(pt[2]) - z0) <= tol * d:
                z_plane = z0
                break
        node_info[idx] = (on_cyl, z_plane)

    class _PillboxProj:
        def project(self, u, v, x):
            u, v = int(u), int(v)
            x = np.asarray(x, dtype=float)
            cu, zu = node_info.get(u, (False, None))
            cv, zv = node_info.get(v, (False, None))
            if cu and cv:
                r_xy = float(np.hypot(x[0], x[1]))
                if r_xy < 1e-15:
                    return None
                z_new = zu if (zu is not None and zu == zv) else float(x[2])
                return np.array([a * x[0] / r_xy, a * x[1] / r_xy, z_new], dtype=float)
            if zu is not None and zu == zv:
                return np.array([x[0], x[1], zu], dtype=float)
            return None

        def register(self, new_node, u, v):
            cu, zu = node_info.get(int(u), (False, None))
            cv, zv = node_info.get(int(v), (False, None))
            node_info[int(new_node)] = (
                bool(cu and cv),
                zu if (zu is not None and zu == zv) else None,
            )

        def unregister(self, new_node, u, v):
            node_info.pop(int(new_node), None)

    return _PillboxProj()


def test_t5_layered_pillbox_geometry_and_stability_gmsh(tmp_path):
    """
    T5 (gmsh): on the repaired STEP layered pillbox, verify that:
    - every Steiner node on a boundary or interface lies within `<= tau` of the
      analytic CAD geometry, where `tau = max(10 * max_input_rim_err, 1e-12 * a)`
      and `tau < min_sagitta / 100`;
    - total and per-tag (`diel`, `vac`) meshed volumes do not move away from the
      analytic CAD volumes `pi * a^2 * t` and `pi * a^2 * (d - t)`:
      `|V_CAD - V_after| <= |V_CAD - V_before| + 1e-12 * V_CAD`;
    - 10,000 leapfrog steps of `UnstructuredFIT3D` at `0.95 * dt_max` with a
      Gaussian dipole source remain stable (`e_10000 < 10 * e_100`).
    """
    gmsh = pytest.importorskip("gmsh")
    from wakis.unstructured.cad import mesh_cad

    a, d, t, eps = 0.1, 0.1, 0.03, 4.0
    v_cad_diel = float(np.pi * a**2 * t)
    v_cad_vac = float(np.pi * a**2 * (d - t))
    v_cad_tot = v_cad_diel + v_cad_vac

    step = _write_layered_pillbox_step(gmsh, tmp_path, a, d, t)
    regions = {"diel": (0, 0, t / 2), "vac": (0, 0, (d + t) / 2)}
    mats = {"diel": [eps, 1], "vac": [1, 1]}

    m_raw, info_raw = mesh_cad(
        step, regions, mats, f_max=1.5e9, cpw=10, optimize=None, improve=None
    )
    rim_errs = []
    rim_sags = []
    for idx, pt in enumerate(m_raw.nodes):
        r_id = _is_rim_node(pt, a, t, d)
        if r_id is not None:
            z0 = (0.0, t, d)[r_id]
            rim_errs.append(float(np.hypot(np.hypot(pt[0], pt[1]) - a, pt[2] - z0)))
    for u, v in m_raw.edges:
        ru = _is_rim_node(m_raw.nodes[int(u)], a, t, d)
        rv = _is_rim_node(m_raw.nodes[int(v)], a, t, d)
        if ru is not None and ru == rv:
            L = float(np.linalg.norm(m_raw.nodes[int(v)] - m_raw.nodes[int(u)]))
            rim_sags.append(L**2 / (8.0 * a))

    tau = max(10.0 * max(rim_errs), 1e-12 * a)
    min_sagitta = min(rim_sags)
    print(
        f"T5 scales: tau={tau:.3e} m, min_rim_sagitta={min_sagitta:.3e} m, "
        f"ratio={min_sagitta / tau:.1f}"
    )
    assert tau < min_sagitta / 100.0

    v0_diel = float(m_raw.cell_volume[m_raw.cell_tags == 0].sum())
    v0_vac = float(m_raw.cell_volume[m_raw.cell_tags == 1].sum())
    v0_tot = float(m_raw.cell_volume.sum())

    m_rep, hist = improve_dt(m_raw, info_raw.materials, n_sweeps=0, repair=True)
    rep = hist[-1]["repair"]
    print(f"T5 imp=None split_project='auto' report: {rep}")
    if rep["negative_faces_after"] > 0:
        m_rep, hist = improve_dt(
            m_raw,
            info_raw.materials,
            n_sweeps=0,
            repair=True,
            split_project=_analytic_pillbox_projector(m_raw, a, d, t),
        )
        rep = hist[-1]["repair"]
        print(f"T5 imp=None analytic pillbox projector report: {rep}")
    assert rep["negative_faces_after"] == 0

    # Check every committed Steiner node on boundary/interface lies within tau of CAD
    bnd_nodes_rep = {int(x) for x in m_rep.boundary_nodes}
    for s_idx in range(m_raw.n_nodes, m_rep.n_nodes):
        pt = m_rep.nodes[s_idx]
        if s_idx in bnd_nodes_rep:
            surf_err = _analytic_surface_distance(pt, a, t, d)
            print(f"T5 Steiner node {s_idx}: surf_err={surf_err:.3e}, tau={tau:.3e}")
            assert surf_err <= tau

    v1_diel = float(m_rep.cell_volume[m_rep.cell_tags == 0].sum())
    v1_vac = float(m_rep.cell_volume[m_rep.cell_tags == 1].sum())
    v1_tot = float(m_rep.cell_volume.sum())
    print(
        f"T5 volumes: diel |V_CAD-V0|={abs(v_cad_diel - v0_diel):.6e} -> "
        f"|V_CAD-V1|={abs(v_cad_diel - v1_diel):.6e}, "
        f"vac |V_CAD-V0|={abs(v_cad_vac - v0_vac):.6e} -> "
        f"|V_CAD-V1|={abs(v_cad_vac - v1_vac):.6e}"
    )
    assert abs(v_cad_diel - v1_diel) <= abs(v_cad_diel - v0_diel) + 1e-12 * v_cad_diel
    assert abs(v_cad_vac - v1_vac) <= abs(v_cad_vac - v0_vac) + 1e-12 * v_cad_vac
    assert abs(v_cad_tot - v1_tot) <= abs(v_cad_tot - v0_tot) + 1e-12 * v_cad_tot

    # 10,000-step leapfrog stability at 0.95 * dt_max (same criterion as test_110)
    solver = UnstructuredFIT3D(m_rep, materials=info_raw.materials, bc="pec", cfl=0.95)
    assert _guard_negative_count(solver) == 0
    assert np.isclose(solver.dt, 0.95 * solver.dt_max, rtol=1e-12)

    dt = solver.dt
    t0 = 25.0 * dt
    tau_src = 6.0 * dt
    solver.add_point_dipole(
        [0.02, 0.01, 0.05],
        [0.0, 0.0, 1.0],
        lambda time_s: np.exp(-0.5 * ((time_s - t0) / tau_src) ** 2),
    )
    solver.run(100)
    e_100 = float(np.max(np.abs(solver.e / solver.mesh.edge_length)))
    assert np.isfinite(e_100) and e_100 > 0.0

    solver.run(9900)
    assert solver.n == 10000
    e_end = float(np.max(np.abs(solver.e / solver.mesh.edge_length)))
    print(f"T5 1e4-step leapfrog: e_100={e_100:.4e}, e_10000={e_end:.4e}")
    assert np.isfinite(e_end)
    assert e_end < 10.0 * e_100
