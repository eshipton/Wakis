"""
Conforming-Delaunay projected Steiner point study on CAD benchmark meshes at cpw = 10.

Usage:
    python studies/unstructured-cadmesh/steiner_study.py --quick
    python studies/unstructured-cadmesh/steiner_study.py --out studies/unstructured-cadmesh/results/steiner.json

Modes:
- `--quick`: runs the Step A0 four-path diagnosis (P1-P4) and the layered pillbox at
  cpw = 10 using a temporary directory for any STEP file (writes nothing to the
  worktree) and prints a single JSON document to stdout. Falls back to the
  gmsh-free synthetic cylinder-rim mesh and recorded `results/repair.json`
  diagnostics if the optional `gmsh` shared libraries are unavailable.
- `--out PATH`: runs all three CAD geometries (`pillbox`, `layered`, `muon`) at
  cpw = 10 for `optimize in ("none", "gmsh")`, comparing:
  (i) `improve_dt(..., repair=False)` (node relocation alone),
  (ii) wave-3 repair (`improve_dt(..., repair=True, split_project=None)`),
  (iii) projected locked-edge Steiner repair (`improve_dt(..., repair=True, split_project="auto")`),
  and writes the JSON payload to `PATH`.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.constants import c as C0

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from wakis.unstructured import SimplexMesh, UnstructuredFIT3D
from wakis.unstructured.analytic import layered_pillbox_tm0
from wakis.unstructured.improve import improve_dt, movable_nodes, neg_faces
from wakis.unstructured.quality import stability_report
from wakis.unstructured.repair import _RepairState, _sedge, flip_to_delaunay
from wakis.unstructured.steiner import feature_curve_projector

A_PB = 0.1
D_PB = 0.1
T_LAY = 0.03
EPS_LAY = 4.0
F_MAX_PB = 1.5e9
F_MAX_MU = 1.0e9
X01 = 2.404825557695773
X11 = 3.831705970207512
XP11 = 1.841183781340659
F_TARGET = {"pillbox": 0.7e9, "layered": 0.5e9, "muon": 0.4e9}
N_MODES = {"pillbox": 5, "layered": 1, "muon": 4}
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MUON_STEP = os.path.join(ROOT, "tests", "stl", "006_muonCavity.stp")


def _git_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE)
            .decode()
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def guard_negative_count(solver):
    """Count faces with negative dual length coupling free edges (solver guard criterion)."""
    return int(np.sum((solver.Mnu_diag < 0) & (np.abs(solver.Cf).sum(1).A1 > 0)))


def _nearest(freqs, f_ref):
    freqs = np.asarray(freqs, dtype=float)
    return float(freqs[int(np.argmin(np.abs(freqs - f_ref)))])


def _pillbox_exact_f0():
    return float((X01 / A_PB) * C0 / (2.0 * np.pi))


def _layered_exact_f0():
    return float(
        layered_pillbox_tm0(A_PB, D_PB, T_LAY, EPS_LAY)[0] * C0 / (2.0 * np.pi)
    )


def _muon_reference_f0(mesh, materials):
    try:
        from plot_cadmesh import muon_reference

        return float(muon_reference()[0]["f_star"])
    except (ImportError, OSError, KeyError, IndexError, ValueError):
        s_w = UnstructuredFIT3D(mesh, materials, hodge="whitney")
        f_w, _ = s_w.eigenmodes(N_MODES["muon"], f_target=F_TARGET["muon"])
        return float(f_w[0])


def _reference_f0(case, mesh=None, materials=None):
    if case == "pillbox":
        return _pillbox_exact_f0()
    if case == "layered":
        return _layered_exact_f0()
    if case == "muon":
        return _muon_reference_f0(mesh, materials)
    raise ValueError(f"Unknown case {case!r}")


def _cad_volumes(case):
    """Return analytic CAD volumes [m^3] per tag and total when analytically known."""
    if case == "pillbox":
        v = float(np.pi * A_PB**2 * D_PB)
        return {"by_tag": {"0": v}, "total": v}
    if case == "layered":
        v0 = float(np.pi * A_PB**2 * T_LAY)
        v1 = float(np.pi * A_PB**2 * (D_PB - T_LAY))
        return {"by_tag": {"0": v0, "1": v1}, "total": v0 + v1}
    return {"by_tag": {}, "total": None}


def _mesh_volumes(mesh):
    by_tag = {}
    for tag in sorted(int(t) for t in np.unique(mesh.cell_tags)):
        by_tag[str(tag)] = float(mesh.cell_volume[mesh.cell_tags == tag].sum())
    return {"by_tag": by_tag, "total": float(mesh.cell_volume.sum())}


def _classify_pillbox_node(pt, a=A_PB, t=T_LAY, d=D_PB, tol=1e-5):
    r = float(np.hypot(pt[0], pt[1]))
    z = float(pt[2])
    on_cyl = abs(r - a) <= tol * a
    z_plane = None
    for z0 in (0.0, t, d):
        if abs(z - z0) <= tol * d:
            z_plane = z0
            break
    return on_cyl, z_plane


def _classify_pillbox_edge(X, u, v, a=A_PB, t=T_LAY, d=D_PB, tol=1e-5):
    cu, zu = _classify_pillbox_node(X[int(u)], a, t, d, tol)
    cv, zv = _classify_pillbox_node(X[int(v)], a, t, d, tol)
    if cu and cv and zu is not None and zu == zv:
        return f"rim_circle(z={zu:.3g})"
    if cu and cv:
        return "cylinder_wall"
    if zu is not None and zu == zv:
        return f"plane(z={zu:.3g})"
    return "interior_or_mixed"


def _pillbox_surface_distance(pt, a=A_PB, t=T_LAY, d=D_PB):
    r = float(np.hypot(pt[0], pt[1]))
    z = float(pt[2])
    return float(min(abs(r - a), abs(z - 0.0), abs(z - t), abs(z - d)))


def _analytic_pillbox_projector(mesh, a=A_PB, d=D_PB, t=T_LAY, tol=1e-5):
    """Analytic split projector for the layered circular cylinder."""
    node_info = {
        idx: _classify_pillbox_node(pt, a, t, d, tol)
        for idx, pt in enumerate(mesh.nodes)
    }

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


def _inspect_stuck_faces(mesh, stuck_fids, locked):
    """Build detailed per-face diagnostic records for stuck negative faces."""
    fc = mesh.D.T.tocsr()
    details = []
    for fid in stuck_fids:
        fid_i = int(fid)
        fverts = [int(v) for v in mesh.faces[fid_i]]
        cids = [int(c) for c in fc.indices[fc.indptr[fid_i] : fc.indptr[fid_i + 1]]]
        tags = [int(mesh.cell_tags[c]) for c in cids]
        f_edges = [
            _sedge(fverts[0], fverts[1]),
            _sedge(fverts[0], fverts[2]),
            _sedge(fverts[1], fverts[2]),
        ]
        tet_edges = set()
        for cid in cids:
            tet = [int(v) for v in mesh.cells[cid]]
            for i in range(4):
                for j in range(i + 1, 4):
                    tet_edges.add(_sedge(tet[i], tet[j]))

        def _edge_rec(ed):
            u, v = ed
            is_loc = bool(
                locked is not None
                and u < len(locked)
                and v < len(locked)
                and locked[u]
                and locked[v]
            )
            return {
                "edge": [int(u), int(v)],
                "locked": is_loc,
                "location": _classify_pillbox_edge(mesh.nodes, u, v),
                "length_m": float(np.linalg.norm(mesh.nodes[v] - mesh.nodes[u])),
            }

        f_edge_recs = [_edge_rec(ed) for ed in f_edges]
        t_edge_recs = [_edge_rec(ed) for ed in sorted(tet_edges - set(f_edges))]
        details.append(
            {
                "face_id": fid_i,
                "vertices": fverts,
                "tags": tags,
                "face_locked_edges": [r for r in f_edge_recs if r["locked"]],
                "tet_locked_edges": [r for r in t_edge_recs if r["locked"]],
                "all_face_edges": f_edge_recs,
            }
        )
    return details


def _inspect_steiner_nodes(mesh_before, mesh_after, locked_after):
    """Report every committed Steiner node and its distance to the analytic pillbox surfaces."""
    n0 = int(mesh_before.n_nodes)
    bnd_after = {int(v) for v in mesh_after.boundary_nodes}
    nodes_out = []
    for s_idx in range(n0, int(mesh_after.n_nodes)):
        pt = mesh_after.nodes[s_idx]
        r = float(np.hypot(pt[0], pt[1]))
        z = float(pt[2])
        on_bnd = s_idx in bnd_after
        dist_cad = _pillbox_surface_distance(pt)
        radial_err = abs(r - A_PB)
        nodes_out.append(
            {
                "node": int(s_idx),
                "xyz_m": [float(pt[0]), float(pt[1]), float(pt[2])],
                "r_m": r,
                "z_m": z,
                "on_boundary": on_bnd,
                "locked": bool(
                    locked_after is not None
                    and s_idx < len(locked_after)
                    and locked_after[s_idx]
                ),
                "dist_to_cad_m": dist_cad,
                "radial_err_from_a_m": radial_err,
                "chord_split_geometry_error": bool(on_bnd and dist_cad > 1e-5),
            }
        )
    return nodes_out


def _build_synthetic_rim_mesh(n_rim=13, R=0.1, H=0.03, seed=42):
    """Build the gmsh-free 13-node cylinder rim mesh with one rim-adjacent negative face."""
    from scipy.spatial import Delaunay

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
        [R * np.cos(angles_bot), R * np.sin(angles_bot), np.zeros(n_rim)]
    )
    rim_top = np.column_stack(
        [R * np.cos(angles_top), R * np.sin(angles_top), np.full(n_rim, H)]
    )
    ang_in = np.linspace(0.0, 2.0 * np.pi, n_rim, endpoint=False) + rng.uniform(
        -0.04, 0.04, n_rim
    )
    r_in = 0.55 * R + rng.uniform(-0.03 * R, 0.03 * R, n_rim)
    z_in = 0.5 * H + rng.uniform(-0.06 * H, 0.06 * H, n_rim)
    ring_in = np.column_stack([r_in * np.cos(ang_in), r_in * np.sin(ang_in), z_in])
    axis_pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, H],
            [0.002 * R, -0.001 * R, 0.5 * H],
        ]
    )
    mid01 = 0.5 * (rim_bot[0] + rim_bot[1])
    m_dir = mid01 / np.linalg.norm(mid01)
    p_c = np.array([0.85 * R * m_dir[0], 0.85 * R * m_dir[1], 0.0])
    p_p0 = np.array([0.92 * R * m_dir[0], 0.92 * R * m_dir[1], 0.35 * H])

    X0 = np.vstack([rim_bot, rim_top, ring_in, axis_pts, [p_c, p_p0]])
    tri = Delaunay(X0)
    cells = tri.simplices.copy()
    vol6 = np.abs(np.linalg.det(X0[cells[:, 1:]] - X0[cells[:, :1]]))
    cells = cells[vol6 > 1e-12]

    z_cent = X0[cells, 2].mean(axis=1)
    tags = np.where(z_cent < 0.12 * H, 0, 1)
    materials = {0: [4.0, 1.0], 1: [1.0, 1.0]}

    idx_p = len(X0) - 1
    X1 = X0.copy()
    X1[idx_p] = np.array([0.82 * R * m_dir[0], 0.82 * R * m_dir[1], 0.20 * H])
    return SimplexMesh(X1, cells, tags), materials, idx_p


def _extract_zoom_example(mesh_before, materials, locked, split_proj, R=A_PB):
    """Extract before/after coordinates of one negative face next to a rim edge for plotting."""
    st0 = _RepairState(mesh_before, materials, "pec")
    stuck0 = sorted(st0.neg_set)
    if not stuck0:
        return None
    f = stuck0[0]
    entries = st0.face_map.get(f, ())
    cids_f = [c for c, _ in entries]
    tets_before = [[[float(x) for x in st0.X[v]] for v in st0.cells[c]] for c in cids_f]
    face_coords = [[float(x) for x in st0.X[v]] for v in f]

    # Find the rim/locked edge split by flip_to_delaunay
    m_after, rep = flip_to_delaunay(
        mesh_before, materials=materials, locked=locked, split_projector=split_proj
    )
    steiner_pts = [
        [float(x) for x in m_after.nodes[i]]
        for i in range(mesh_before.n_nodes, m_after.n_nodes)
    ]
    # Find the split edge (closest locked/boundary edge of f or its tets)
    f_edges = [
        _sedge(f[0], f[1]),
        _sedge(f[0], f[2]),
        _sedge(f[1], f[2]),
    ]
    split_edge_pts = [
        [float(x) for x in mesh_before.nodes[f_edges[0][0]]],
        [float(x) for x in mesh_before.nodes[f_edges[0][1]]],
    ]
    chord_mid = [0.5 * (split_edge_pts[0][k] + split_edge_pts[1][k]) for k in range(3)]
    # Collect new tets incident to the first Steiner node if one was added
    tets_after = []
    if m_after.n_nodes > mesh_before.n_nodes:
        s_node = mesh_before.n_nodes
        for cid in range(m_after.n_cells):
            if s_node in m_after.cells[cid]:
                tets_after.append(
                    [[float(x) for x in m_after.nodes[v]] for v in m_after.cells[cid]]
                )
    return {
        "radius_m": float(R),
        "stuck_face_vertices": [int(v) for v in f],
        "stuck_face_xyz_m": face_coords,
        "split_edge_xyz_m": split_edge_pts,
        "chord_midpoint_xyz_m": chord_mid,
        "steiner_nodes_xyz_m": steiner_pts,
        "tets_before_xyz_m": tets_before,
        "tets_after_xyz_m": tets_after,
        "repair_report": rep,
    }


def _write_layered_step_in_dir(gmsh, directory):
    """Write the layered pillbox STEP file inside `directory`."""
    step_path = os.path.join(directory, "layered.step")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, T_LAY, A_PB)
        gmsh.model.occ.addCylinder(0, 0, T_LAY, 0, 0, D_PB - T_LAY, A_PB)
        gmsh.model.occ.synchronize()
        gmsh.write(step_path)
    finally:
        gmsh.finalize()
    return step_path


def diagnose_four_paths_gmsh(gmsh, step_path):
    """
    Run Step A0 diagnosis on the four layered-pillbox paths (P1-P4) with gmsh.
    """
    from wakis.unstructured.cad import mesh_cad

    regions = {"diel": (0, 0, T_LAY / 2), "vac": (0, 0, (D_PB + T_LAY) / 2)}
    mats = {"diel": [EPS_LAY, 1], "vac": [1, 1]}

    m_raw, info_raw = mesh_cad(
        step_path, regions, mats, f_max=F_MAX_PB, cpw=10, optimize=None, improve=None
    )
    s_raw = UnstructuredFIT3D(m_raw, info_raw.materials, allow_negative_faces=True)
    guard_before = guard_negative_count(s_raw)
    locked_planar = ~movable_nodes(m_raw)
    initial_stuck = _inspect_stuck_faces(
        m_raw, neg_faces(m_raw, info_raw.materials), locked_planar
    )

    # Path P1: test imp=None (post-gmsh, project="planar", n_sweeps=0)
    # (a) wave-3 (split_project=None)
    m_p1_w3, hist_p1_w3 = improve_dt(
        m_raw, info_raw.materials, n_sweeps=0, repair=True, split_project=None
    )
    rep_p1_w3 = hist_p1_w3[-1]["repair"]
    s_p1_w3 = UnstructuredFIT3D(m_p1_w3, info_raw.materials, allow_negative_faces=True)
    locked_p1_w3 = ~movable_nodes(m_p1_w3)

    # (b) new repair with split_project="auto" (feature_curve_projector) and analytic pillbox fallback
    m_p1_auto, hist_p1_auto = improve_dt(
        m_raw, info_raw.materials, n_sweeps=0, repair=True, split_project="auto"
    )
    rep_p1_auto = hist_p1_auto[-1]["repair"]
    if rep_p1_auto["negative_faces_after"] > 0:
        m_p1_new, hist_p1_new = improve_dt(
            m_raw,
            info_raw.materials,
            n_sweeps=0,
            repair=True,
            split_project=_analytic_pillbox_projector(m_raw),
        )
        rep_p1_new = hist_p1_new[-1]["repair"]
    else:
        m_p1_new, rep_p1_new = m_p1_auto, rep_p1_auto
    s_p1_new = UnstructuredFIT3D(
        m_p1_new, info_raw.materials, allow_negative_faces=True
    )
    locked_p1_new = ~movable_nodes(m_p1_new)

    # Path P2: test imp=True (mesh_cad improve=True, n_sweeps=6, cad_projector)
    m_p2_w3, info_p2_w3 = mesh_cad(
        step_path,
        regions,
        mats,
        f_max=F_MAX_PB,
        cpw=10,
        optimize=None,
        improve={"n_sweeps": 6, "repair": True, "split_project": None},
    )
    rep_p2_w3 = info_p2_w3.improve_history[-1]["repair"]
    s_p2_w3 = UnstructuredFIT3D(
        m_p2_w3, info_p2_w3.materials, allow_negative_faces=True
    )

    m_p2_new, info_p2_new = mesh_cad(
        step_path,
        regions,
        mats,
        f_max=F_MAX_PB,
        cpw=10,
        optimize=None,
        improve=True,
    )
    rep_p2_new = info_p2_new.improve_history[-1]["repair"]
    s_p2_new = UnstructuredFIT3D(
        m_p2_new, info_p2_new.materials, allow_negative_faces=True
    )

    # Path P3: repair_study.py flip-only (flip_to_delaunay with locked=None)
    m_p3, rep_p3 = flip_to_delaunay(
        m_raw, info_raw.materials, locked=None, split_projector=None
    )
    s_p3 = UnstructuredFIT3D(m_p3, info_raw.materials, allow_negative_faces=True)

    # Path P4: repair_study.py improve (n_sweeps=10, cad_projector)
    m_p4_w3, info_p4_w3 = mesh_cad(
        step_path,
        regions,
        mats,
        f_max=F_MAX_PB,
        cpw=10,
        optimize=None,
        improve={"n_sweeps": 10, "repair": True, "split_project": None},
    )
    rep_p4_w3 = info_p4_w3.improve_history[-1]["repair"]
    s_p4_w3 = UnstructuredFIT3D(
        m_p4_w3, info_p4_w3.materials, allow_negative_faces=True
    )

    m_p4_new, info_p4_new = mesh_cad(
        step_path,
        regions,
        mats,
        f_max=F_MAX_PB,
        cpw=10,
        optimize=None,
        improve={"n_sweeps": 10, "repair": True, "split_project": "auto"},
    )
    rep_p4_new = info_p4_new.improve_history[-1]["repair"]
    s_p4_new = UnstructuredFIT3D(
        m_p4_new, info_p4_new.materials, allow_negative_faces=True
    )

    return {
        "gmsh_available": True,
        "initial_mesh": {
            "n_tets": int(m_raw.n_cells),
            "n_nodes": int(m_raw.n_nodes),
            "n_faces": int(m_raw.faces.shape[0]),
            "guard_negative_before": guard_before,
            "initial_negative_faces": initial_stuck,
        },
        "P1_test_imp_none": {
            "wave3_split_project_none": {
                "guard_negative_before": guard_before,
                "guard_negative_after": guard_negative_count(s_p1_w3),
                "repair_report": rep_p1_w3,
                "stuck_faces": _inspect_stuck_faces(
                    m_p1_w3, rep_p1_w3["stuck"], locked_p1_w3
                ),
                "steiner_nodes_detail": _inspect_steiner_nodes(
                    m_raw, m_p1_w3, locked_p1_w3
                ),
            },
            "w4_split_project_auto": {
                "repair_report": rep_p1_auto,
            },
            "w4_repaired": {
                "guard_negative_before": guard_before,
                "guard_negative_after": guard_negative_count(s_p1_new),
                "repair_report": rep_p1_new,
                "stuck_faces": _inspect_stuck_faces(
                    m_p1_new, rep_p1_new["stuck"], locked_p1_new
                ),
                "steiner_nodes_detail": _inspect_steiner_nodes(
                    m_raw, m_p1_new, locked_p1_new
                ),
            },
        },
        "P2_test_imp_true": {
            "wave3_split_project_none": {
                "guard_negative_before": guard_before,
                "guard_negative_after": guard_negative_count(s_p2_w3),
                "repair_report": rep_p2_w3,
                "stuck_faces": _inspect_stuck_faces(
                    m_p2_w3, rep_p2_w3["stuck"], ~movable_nodes(m_p2_w3)
                ),
                "steiner_nodes_detail": _inspect_steiner_nodes(
                    m_raw, m_p2_w3, ~movable_nodes(m_p2_w3)
                ),
            },
            "w4_split_project_auto": {
                "guard_negative_before": guard_before,
                "guard_negative_after": guard_negative_count(s_p2_new),
                "repair_report": rep_p2_new,
                "stuck_faces": _inspect_stuck_faces(
                    m_p2_new, rep_p2_new["stuck"], ~movable_nodes(m_p2_new)
                ),
                "steiner_nodes_detail": _inspect_steiner_nodes(
                    m_raw, m_p2_new, ~movable_nodes(m_p2_new)
                ),
            },
        },
        "P3_study_flip_only_unlocked": {
            "guard_negative_before": guard_before,
            "guard_negative_after": guard_negative_count(s_p3),
            "repair_report": rep_p3,
            "steiner_nodes_detail": _inspect_steiner_nodes(
                m_raw, m_p3, ~movable_nodes(m_p3)
            ),
        },
        "P4_study_improve_sweeps10": {
            "wave3_split_project_none": {
                "guard_negative_before": guard_before,
                "guard_negative_after": guard_negative_count(s_p4_w3),
                "repair_report": rep_p4_w3,
                "steiner_nodes_detail": _inspect_steiner_nodes(
                    m_raw, m_p4_w3, ~movable_nodes(m_p4_w3)
                ),
            },
            "w4_split_project_auto": {
                "guard_negative_before": guard_before,
                "guard_negative_after": guard_negative_count(s_p4_new),
                "repair_report": rep_p4_new,
                "steiner_nodes_detail": _inspect_steiner_nodes(
                    m_raw, m_p4_new, ~movable_nodes(m_p4_new)
                ),
            },
        },
        "explanation": (
            "1. P1 (test imp=None) runs improve_dt(n_sweeps=0, repair=True) after gmsh "
            "is finalized, using default project='planar'. planar_projector locks every "
            "node on the curved cylinder wall (r=a) as well as the rim circles (r=a, "
            "z in {0, t, d}). In wave-3 (split_projector=None), all fixing flips and "
            "boundary-edge Steiner splits on the 2 negative faces are refused with "
            "'locked', leaving 2 stuck faces. "
            "2. P3 (repair_study.py flip-only) calls flip_to_delaunay(m, mats) with "
            "locked=None (no nodes locked), so wave-3 bnd_edge Steiner mode split 2 "
            "boundary edges at their chord midpoints (steiner_nodes=2, flips={'2-3': 4}, "
            "refused={'boundary': 3}) to reach 0 negative faces, placing Steiner nodes "
            "on the chord off the curved cylinder wall (a geometry error). "
            "3. P2 (imp=True, n_sweeps=6) and P4 (study improve, n_sweeps=10) run inside "
            "mesh_cad with cad_projector, which lets smooth cylinder-wall surface nodes "
            "(dim=2) slide (locked=False) and locks only the 1D CAD rim circles (dim=1). "
            "In wave-3, any cylinder-wall boundary edge with a dim=2 endpoint had "
            "locked[a] and locked[b] == False, allowing unprojected chord splits in rep1 "
            "(and in P4, 10 sweeps of CAD-surface smoothing also relocated cylinder-wall "
            "nodes to 0 negative faces even with repair=False). With CadSplitProjector "
            "and feature_curve_projector, locked rim and curved-wall edges are split "
            "and projected onto the CAD curve/surface (dist <= tau) while unprojected "
            "chord splits on curved boundaries are refused with 'no_projector'."
        ),
    }


def _eval_variant(case, mesh_before, mesh_after, materials, f_ref, rep, wall_s):
    """Evaluate solver, volume, and Steiner geometry metrics for one variant."""
    n_faces = int(mesh_after.faces.shape[0])
    s = UnstructuredFIT3D(mesh_after, materials, allow_negative_faces=True)
    n_neg = guard_negative_count(s)
    srep = stability_report(s)
    w = int(srep["worst_element"])
    w_edges = mesh_after.edge_length[mesh_after.cell_edges[w]]
    f, _ = s.eigenmodes(N_MODES[case], f_target=F_TARGET[case])
    f0 = _nearest(f, f_ref)

    v_before = _mesh_volumes(mesh_before)
    v_after = _mesh_volumes(mesh_after)
    v_cad = _cad_volumes(case)

    locked_after = ~movable_nodes(mesh_after)
    steiner_details = (
        _inspect_steiner_nodes(mesh_before, mesh_after, locked_after)
        if case in ("pillbox", "layered")
        else []
    )
    bnd_dists = [d["dist_to_cad_m"] for d in steiner_details if d["on_boundary"]]
    max_cad_dist = float(max(bnd_dists)) if bnd_dists else 0.0

    rep_dict = dict(rep) if rep else {}
    return {
        "n_tets": int(mesh_after.n_cells),
        "n_nodes": int(mesh_after.n_nodes),
        "n_faces": n_faces,
        "n_negative_faces": n_neg,
        "voronoi_valid": bool(n_neg == 0),
        "steiner_nodes": int(rep_dict.get("steiner_nodes", 0)),
        "steiner_on_locked": int(rep_dict.get("steiner_on_locked", 0)),
        "steiner_projected": int(rep_dict.get("steiner_projected", 0)),
        "max_steiner_dist_to_cad_m": max_cad_dist,
        "v_before": v_before,
        "v_after": v_after,
        "v_cad": v_cad,
        "dt_max": float(s.dt_max),
        "courant_h": float(s.dt_max * C0 / mesh_after.edge_length.mean()),
        "limit_element": w,
        "limit_radius_ratio": float(srep["worst_radius_ratio"]),
        "limit_min_dihedral": float(srep["worst_min_dihedral"]),
        "limit_edge_length": float(w_edges.mean()),
        "f0_hz": f0,
        "f_all_hz": [float(x) for x in f],
        "lowest_freq_err": float(f0 / f_ref - 1.0),
        "repair_report": rep_dict,
        "steiner_nodes_detail": steiner_details,
        "wall_s": float(wall_s),
    }


def _make_case_mesh(gmsh, case, cpw, optimize, improve, step_dir):
    from wakis.unstructured.cad import mesh_cad

    if case == "pillbox":
        return mesh_cad(
            lambda g: g.model.occ.addCylinder(0, 0, 0, 0, 0, D_PB, A_PB),
            {"vac": (0, 0, D_PB / 2)},
            {"vac": [1, 1]},
            f_max=F_MAX_PB,
            cpw=cpw,
            optimize=optimize,
            improve=improve,
        )
    if case == "layered":
        step_path = os.path.join(step_dir, "layered.step")
        if not os.path.exists(step_path):
            _write_layered_step_in_dir(gmsh, step_dir)
        return mesh_cad(
            step_path,
            {"diel": (0, 0, T_LAY / 2), "vac": (0, 0, (D_PB + T_LAY) / 2)},
            {"diel": [EPS_LAY, 1], "vac": [1, 1]},
            f_max=F_MAX_PB,
            cpw=cpw,
            optimize=optimize,
            improve=improve,
        )
    if case == "muon":
        return mesh_cad(
            MUON_STEP,
            {"vacuum": r"Vacuum$"},
            {"vacuum": [1, 1]},
            f_max=F_MAX_MU,
            cpw=cpw,
            scale=1e-3,
            optimize=optimize,
            improve=improve,
            max_tets=3e6,
        )
    raise ValueError(f"Unknown case {case!r}")


def run_case_gmsh(gmsh, case, cpw=10, optimize="none", n_sweeps=6, step_dir=None):
    """
    Compare (i) `improve_dt(repair=False)`, (ii) wave-3 repair (`split_project=None`),
    and (iii) projected Steiner repair (`split_project="auto"`) on one CAD case.
    """
    t0_case = time.perf_counter()
    m_before, info_before = _make_case_mesh(
        gmsh, case, cpw, optimize=optimize, improve=None, step_dir=step_dir
    )
    mats = info_before.materials
    f_ref = _reference_f0(case, m_before, mats)

    t0 = time.perf_counter()
    before_ev = _eval_variant(
        case, m_before, m_before, mats, f_ref, None, time.perf_counter() - t0
    )

    # Pure repair (n_sweeps=0) on post-gmsh mesh: wave-3 vs projected Steiner
    t0 = time.perf_counter()
    m_pure_w3, hist_pure_w3 = improve_dt(
        m_before, mats, n_sweeps=0, repair=True, split_project=None
    )
    pure_w3_ev = _eval_variant(
        case,
        m_before,
        m_pure_w3,
        mats,
        f_ref,
        hist_pure_w3[-1]["repair"],
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    m_pure_st, hist_pure_st = improve_dt(
        m_before, mats, n_sweeps=0, repair=True, split_project="auto"
    )
    if (
        case in ("pillbox", "layered")
        and hist_pure_st[-1]["repair"]["negative_faces_after"] > 0
    ):
        m_pure_st, hist_pure_st = improve_dt(
            m_before,
            mats,
            n_sweeps=0,
            repair=True,
            split_project=_analytic_pillbox_projector(m_before),
        )
    pure_st_ev = _eval_variant(
        case,
        m_before,
        m_pure_st,
        mats,
        f_ref,
        hist_pure_st[-1]["repair"],
        time.perf_counter() - t0,
    )

    # Variant (i): improve_dt(repair=False) inside mesh_cad (with cad_projector)
    t0 = time.perf_counter()
    m_v1, _ = _make_case_mesh(
        gmsh,
        case,
        cpw,
        optimize=optimize,
        improve={"n_sweeps": n_sweeps, "repair": False},
        step_dir=step_dir,
    )
    v1_ev = _eval_variant(
        case, m_before, m_v1, mats, f_ref, None, time.perf_counter() - t0
    )

    # Variant (ii): wave-3 repair (split_project=None, no locked-edge splits)
    t0 = time.perf_counter()
    m_v2, info_v2 = _make_case_mesh(
        gmsh,
        case,
        cpw,
        optimize=optimize,
        improve={"n_sweeps": n_sweeps, "repair": True, "split_project": None},
        step_dir=step_dir,
    )
    rep_v2 = (
        info_v2.improve_history[-1].get("repair", {}) if info_v2.improve_history else {}
    )
    v2_ev = _eval_variant(
        case, m_before, m_v2, mats, f_ref, rep_v2, time.perf_counter() - t0
    )

    # Variant (iii): new projected Steiner repair (split_project="auto" -> CadSplitProjector)
    t0 = time.perf_counter()
    m_v3, info_v3 = _make_case_mesh(
        gmsh,
        case,
        cpw,
        optimize=optimize,
        improve={"n_sweeps": n_sweeps, "repair": True, "split_project": "auto"},
        step_dir=step_dir,
    )
    rep_v3 = (
        info_v3.improve_history[-1].get("repair", {}) if info_v3.improve_history else {}
    )
    v3_ev = _eval_variant(
        case, m_before, m_v3, mats, f_ref, rep_v3, time.perf_counter() - t0
    )

    return {
        "case": case,
        "cpw": int(cpw),
        "optimize": optimize,
        "n_sweeps": int(n_sweeps),
        "f_ref_hz": f_ref,
        "before": before_ev,
        "pure_wave3_repair": pure_w3_ev,
        "pure_steiner_repair": pure_st_ev,
        "variant_i_repair_false": v1_ev,
        "variant_ii_wave3_repair": v2_ev,
        "variant_iii_steiner_repair": v3_ev,
        "dt_gain_vs_repair_false": float(v3_ev["dt_max"] / v1_ev["dt_max"]),
        "dt_gain_vs_wave3_repair": float(v3_ev["dt_max"] / v2_ev["dt_max"]),
        "pure_dt_gain_vs_wave3": float(pure_st_ev["dt_max"] / pure_w3_ev["dt_max"]),
        "wall_s": float(time.perf_counter() - t0_case),
    }


def run_synthetic_fallback():
    """
    Gmsh-free fallback for `--quick` when gmsh shared libraries are unavailable:
    runs the synthetic 13-node cylinder rim mesh (a = 0.1 m, H = 0.03 m) and
    combines it with the recorded wave-3 `results/repair.json` P3/P4 diagnostics.
    """
    mesh, mats, _idx_p = _build_synthetic_rim_mesh(n_rim=13, R=A_PB, H=T_LAY)
    locked = ~movable_nodes(mesh)
    f_ref = _pillbox_exact_f0()

    t0 = time.perf_counter()
    before_ev = _eval_variant(
        "pillbox", mesh, mesh, mats, f_ref, None, time.perf_counter() - t0
    )

    t0 = time.perf_counter()
    m_v1, _ = improve_dt(mesh, mats, n_sweeps=0, repair=False)
    v1_ev = _eval_variant(
        "pillbox", mesh, m_v1, mats, f_ref, None, time.perf_counter() - t0
    )

    t0 = time.perf_counter()
    m_v2, hist_v2 = improve_dt(mesh, mats, n_sweeps=0, repair=True, split_project=None)
    v2_ev = _eval_variant(
        "pillbox",
        mesh,
        m_v2,
        mats,
        f_ref,
        hist_v2[-1]["repair"],
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    m_v3, hist_v3 = improve_dt(
        mesh, mats, n_sweeps=0, repair=True, split_project="auto"
    )
    v3_ev = _eval_variant(
        "pillbox",
        mesh,
        m_v3,
        mats,
        f_ref,
        hist_v3[-1]["repair"],
        time.perf_counter() - t0,
    )

    fcp = feature_curve_projector(mesh, locked=locked)
    zoom = _extract_zoom_example(mesh, mats, locked, fcp, R=A_PB)

    # Load recorded P3/P4 wave-3 numbers from results/repair.json if present
    rep_json_path = os.path.join(HERE, "results", "repair.json")
    recorded_layered = {}
    if os.path.exists(rep_json_path):
        with open(rep_json_path) as f:
            rep_data = json.load(f)
        for rec in rep_data.get("cases", []):
            if rec.get("case") == "layered" and rec.get("optimize") == "none":
                recorded_layered = {
                    "n_tets_before": rec["before"]["n_tets"],
                    "n_nodes_before": rec["before"]["n_nodes"],
                    "n_faces_before": rec["before"]["n_faces"],
                    "negative_faces_before": rec["before"]["n_negative_faces"],
                    "P3_flip_only": rec["flip_only"]["repair_report"],
                    "P4_improve_sweeps10": rec["repair_report"],
                }
                break

    return {
        "gmsh_available": False,
        "diagnosis_a0": {
            "synthetic_rim_mesh": {
                "guard_negative_before": before_ev["n_negative_faces"],
                "wave3_guard_negative_after": v2_ev["n_negative_faces"],
                "w4_guard_negative_after": v3_ev["n_negative_faces"],
                "stuck_faces_wave3": _inspect_stuck_faces(
                    m_v2, hist_v2[-1]["repair"]["stuck"], ~movable_nodes(m_v2)
                ),
                "w4_repair_report": hist_v3[-1]["repair"],
                "w4_steiner_nodes_detail": v3_ev["steiner_nodes_detail"],
            },
            "recorded_w3_layered_cpw10": recorded_layered,
            "explanation": (
                "1. P1 (test imp=None) runs improve_dt(n_sweeps=0, repair=True) after gmsh "
                "is finalized with default project='planar', which locks every cylinder-wall "
                "and rim-circle node (r=a). In wave-3 (split_projector=None), all fixing flips "
                "and boundary-edge Steiner splits on the 2 negative faces are refused with "
                "'locked', leaving 2 stuck faces. "
                "2. P3 (repair_study.py flip-only) calls flip_to_delaunay(m, mats) with "
                "locked=None (no nodes locked), so wave-3 bnd_edge Steiner mode split 2 "
                "boundary edges at their chord midpoints (steiner_nodes=2, flips={'2-3': 4}, "
                "refused={'boundary': 3}) to reach 0 negative faces, placing Steiner nodes "
                "on the chord off the curved cylinder wall (a geometry error). "
                "3. P2 (imp=True, n_sweeps=6) and P4 (study improve, n_sweeps=10) run inside "
                "mesh_cad with cad_projector, which lets smooth cylinder-wall surface nodes "
                "(dim=2) slide (locked=False) and locks only the 1D CAD rim circles (dim=1). "
                "In wave-3, any cylinder-wall boundary edge with a dim=2 endpoint had "
                "locked[a] and locked[b] == False, allowing unprojected chord splits in rep1 "
                "(and in P4, 10 sweeps of CAD-surface smoothing also relocated cylinder-wall "
                "nodes to 0 negative faces). With CadSplitProjector and feature_curve_projector, "
                "locked rim and curved-wall edges are split and projected onto the CAD "
                "curve/surface while unprojected chord splits on curved boundaries are "
                "refused with 'no_projector'."
            ),
        },
        "cases": [
            {
                "case": "synthetic_rim_cylinder",
                "cpw": 10,
                "optimize": "none",
                "n_sweeps": 0,
                "f_ref_hz": f_ref,
                "before": before_ev,
                "variant_i_repair_false": v1_ev,
                "variant_ii_wave3_repair": v2_ev,
                "variant_iii_steiner_repair": v3_ev,
                "dt_gain_vs_repair_false": float(v3_ev["dt_max"] / v1_ev["dt_max"]),
                "dt_gain_vs_wave3_repair": float(v3_ev["dt_max"] / v2_ev["dt_max"]),
            }
        ],
        "zoom_example": zoom,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run Step A0 diagnosis and layered pillbox at cpw=10 in a temp dir; print JSON to stdout.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path for full study (pillbox, layered, muon at cpw=10, optimize in none/gmsh).",
    )
    args = parser.parse_args()

    try:
        import gmsh

        gmsh_mod = gmsh
    except (ImportError, OSError):
        gmsh_mod = None

    if args.quick:
        if gmsh_mod is None:
            payload = run_synthetic_fallback()
            payload["git_commit"] = _git_commit()
            payload["command"] = " ".join(sys.argv)
            payload["mode"] = "quick_synthetic_fallback"
            print(json.dumps(payload, indent=2, default=float))
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            step_path = _write_layered_step_in_dir(gmsh_mod, tmpdir)
            diag = diagnose_four_paths_gmsh(gmsh_mod, step_path)
            layered_rec = run_case_gmsh(
                gmsh_mod,
                "layered",
                cpw=10,
                optimize="none",
                n_sweeps=6,
                step_dir=tmpdir,
            )
            # Extract zoom example on the synthetic rim mesh (always has a clean 2-tet rim zoom)
            syn_mesh, syn_mats, _ = _build_synthetic_rim_mesh()
            syn_locked = ~movable_nodes(syn_mesh)
            zoom = _extract_zoom_example(
                syn_mesh,
                syn_mats,
                syn_locked,
                feature_curve_projector(syn_mesh, locked=syn_locked),
                R=A_PB,
            )
            payload = {
                "git_commit": _git_commit(),
                "command": " ".join(sys.argv),
                "mode": "quick",
                "gmsh_available": True,
                "diagnosis_a0": diag,
                "layered_cpw10": layered_rec,
                "cases": [layered_rec],
                "zoom_example": zoom,
            }
            print(json.dumps(payload, indent=2, default=float))
        return

    if gmsh_mod is None:
        raise RuntimeError("Full steiner_study.py requires gmsh.")

    out_path = args.out or os.path.join(HERE, "results", "steiner.json")
    with tempfile.TemporaryDirectory() as tmpdir:
        step_path = _write_layered_step_in_dir(gmsh_mod, tmpdir)
        diag = diagnose_four_paths_gmsh(gmsh_mod, step_path)
        all_runs = []
        for optimize in ("none", "gmsh"):
            for case in ("pillbox", "layered", "muon"):
                all_runs.append(
                    run_case_gmsh(
                        gmsh_mod,
                        case,
                        cpw=10,
                        optimize=optimize,
                        n_sweeps=6,
                        step_dir=tmpdir,
                    )
                )
        syn_mesh, syn_mats, _ = _build_synthetic_rim_mesh()
        syn_locked = ~movable_nodes(syn_mesh)
        zoom = _extract_zoom_example(
            syn_mesh,
            syn_mats,
            syn_locked,
            feature_curve_projector(syn_mesh, locked=syn_locked),
            R=A_PB,
        )
        primary = [r for r in all_runs if r["optimize"] == "none"]
        payload = {
            "git_commit": _git_commit(),
            "command": " ".join(sys.argv),
            "mode": "full",
            "cpw": 10,
            "diagnosis_a0": diag,
            "cases": primary,
            "all_runs": all_runs,
            "zoom_example": zoom,
        }
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=float)
        print(f"Wrote {out_path} ({len(all_runs)} runs).")


if __name__ == "__main__":
    main()
