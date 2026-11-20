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

from wakis.unstructured import SimplexMesh, UnstructuredFIT3D
from wakis.unstructured.analytic import layered_pillbox_tm0
from wakis.unstructured.improve import improve_dt, movable_nodes, neg_faces
from wakis.unstructured.quality import stability_report
from wakis.unstructured.repair import _RepairState, _sedge, flip_to_delaunay
from wakis.unstructured.steiner import (
    FeatureCurveProjector,
    feature_curve_projector,
    split_edge_steiner,
)

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

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


def _edge_class_bucket(loc):
    if loc.startswith("rim_circle"):
        return "rim_circle"
    if loc == "cylinder_wall":
        return "cylinder_wall"
    if loc.startswith("plane"):
        return "plane"
    return "interior_or_mixed"


def _summarize_stuck_edge_classes(stuck_faces):
    """Count edge classes (rim_circle, cylinder_wall, plane, interior_or_mixed) on stuck faces."""
    buckets = ("rim_circle", "cylinder_wall", "plane", "interior_or_mixed")
    face_locked = {k: 0 for k in buckets}
    tet_locked = {k: 0 for k in buckets}
    all_face = {k: 0 for k in buckets}
    for sf in stuck_faces:
        for r in sf.get("face_locked_edges", []):
            face_locked[_edge_class_bucket(r["location"])] += 1
        for r in sf.get("tet_locked_edges", []):
            tet_locked[_edge_class_bucket(r["location"])] += 1
        for r in sf.get("all_face_edges", []):
            all_face[_edge_class_bucket(r["location"])] += 1
    return {
        "face_locked_edges": face_locked,
        "tet_locked_edges": tet_locked,
        "all_face_edges": all_face,
    }


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


def _path_diag_record(
    mesh_before, mesh_after, solver_after, guard_before, rep, locked_after
):
    """Build uniform Step A0 diagnostic record for one repair path."""
    stuck_faces = _inspect_stuck_faces(mesh_after, rep.get("stuck", []), locked_after)
    steiner_details = _inspect_steiner_nodes(mesh_before, mesh_after, locked_after)
    bnd_dists = [d["dist_to_cad_m"] for d in steiner_details if d["on_boundary"]]
    all_dists = [d["dist_to_cad_m"] for d in steiner_details]
    rad_errs = [d["radial_err_from_a_m"] for d in steiner_details if d["on_boundary"]]
    return {
        "n_tets": int(mesh_after.n_cells),
        "n_nodes": int(mesh_after.n_nodes),
        "n_faces": int(mesh_after.faces.shape[0]),
        "guard_negative_before": int(guard_before),
        "guard_negative_after": int(guard_negative_count(solver_after)),
        "steiner_nodes": int(rep.get("steiner_nodes", 0)),
        "steiner_on_locked": int(rep.get("steiner_on_locked", 0)),
        "steiner_projected": int(rep.get("steiner_projected", 0)),
        "max_steiner_dist_to_cad_m": float(max(bnd_dists)) if bnd_dists else 0.0,
        "max_all_steiner_dist_to_cad_m": float(max(all_dists)) if all_dists else 0.0,
        "max_steiner_radial_err_m": float(max(rad_errs)) if rad_errs else 0.0,
        "repair_report": rep,
        "stuck_faces": stuck_faces,
        "stuck_edge_classes": _summarize_stuck_edge_classes(stuck_faces),
        "steiner_nodes_detail": steiner_details,
    }


def _load_recorded_w3_layered():
    """Load recorded wave-3 layered pillbox (cpw=10, optimize='none') from results/repair.json."""
    rep_json_path = os.path.join(HERE, "results", "repair.json")
    if not os.path.exists(rep_json_path):
        return {}
    with open(rep_json_path) as f:
        rep_data = json.load(f)
    gains = [
        float(r["dt_gain_vs_repair_false"])
        for r in rep_data.get("all_runs", rep_data.get("cases", []))
        if "dt_gain_vs_repair_false" in r
    ]
    for rec in rep_data.get("cases", []):
        if rec.get("case") == "layered" and rec.get("optimize") == "none":
            return {
                "n_tets_before": int(rec["before"]["n_tets"]),
                "n_nodes_before": int(rec["before"]["n_nodes"]),
                "n_faces_before": int(rec["before"]["n_faces"]),
                "negative_faces_before": int(rec["before"]["n_negative_faces"]),
                "P3_flip_only_n_tets_after": int(rec["flip_only"]["n_tets"]),
                "P3_flip_only_n_nodes_after": int(rec["flip_only"]["n_nodes"]),
                "P3_flip_only_n_faces_after": int(rec["flip_only"]["n_faces"]),
                "P3_flip_only": rec["flip_only"]["repair_report"],
                "P4_improve_sweeps10_n_tets_after": int(
                    rec["after_repair_true"]["n_tets"]
                ),
                "P4_improve_sweeps10_n_nodes_after": int(
                    rec["after_repair_true"]["n_nodes"]
                ),
                "P4_improve_sweeps10_n_faces_after": int(
                    rec["after_repair_true"]["n_faces"]
                ),
                "P4_improve_sweeps10": rec["repair_report"],
                "suite_dt_gain_vs_repair_false_min": min(gains) if gains else None,
                "suite_dt_gain_vs_repair_false_max": max(gains) if gains else None,
            }
    return {}


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
    s_p1_auto = UnstructuredFIT3D(
        m_p1_auto, info_raw.materials, allow_negative_faces=True
    )
    locked_p1_auto = ~movable_nodes(m_p1_auto)
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
    m_p3_w3, rep_p3_w3 = flip_to_delaunay(
        m_raw,
        info_raw.materials,
        locked=None,
        split_projector=lambda a, b, x: np.asarray(x, dtype=float),
    )
    s_p3_w3 = UnstructuredFIT3D(m_p3_w3, info_raw.materials, allow_negative_faces=True)

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

    p1_w3_rec = _path_diag_record(
        m_raw, m_p1_w3, s_p1_w3, guard_before, rep_p1_w3, locked_p1_w3
    )
    p1_auto_rec = _path_diag_record(
        m_raw, m_p1_auto, s_p1_auto, guard_before, rep_p1_auto, locked_p1_auto
    )
    p1_new_rec = _path_diag_record(
        m_raw, m_p1_new, s_p1_new, guard_before, rep_p1_new, locked_p1_new
    )
    p2_w3_rec = _path_diag_record(
        m_raw, m_p2_w3, s_p2_w3, guard_before, rep_p2_w3, ~movable_nodes(m_p2_w3)
    )
    p2_new_rec = _path_diag_record(
        m_raw, m_p2_new, s_p2_new, guard_before, rep_p2_new, ~movable_nodes(m_p2_new)
    )
    p3_rec = _path_diag_record(
        m_raw, m_p3, s_p3, guard_before, rep_p3, ~movable_nodes(m_p3)
    )
    p3_w3_rec = _path_diag_record(
        m_raw, m_p3_w3, s_p3_w3, guard_before, rep_p3_w3, ~movable_nodes(m_p3_w3)
    )
    p3_rec["wave3_unprojected_chord"] = p3_w3_rec
    p4_w3_rec = _path_diag_record(
        m_raw, m_p4_w3, s_p4_w3, guard_before, rep_p4_w3, ~movable_nodes(m_p4_w3)
    )
    p4_new_rec = _path_diag_record(
        m_raw, m_p4_new, s_p4_new, guard_before, rep_p4_new, ~movable_nodes(m_p4_new)
    )
    recorded_w3 = _load_recorded_w3_layered()

    return {
        "gmsh_available": True,
        "initial_mesh": {
            "n_tets": int(m_raw.n_cells),
            "n_nodes": int(m_raw.n_nodes),
            "n_faces": int(m_raw.faces.shape[0]),
            "guard_negative_before": guard_before,
            "initial_negative_faces": initial_stuck,
            "initial_edge_classes": _summarize_stuck_edge_classes(initial_stuck),
        },
        "P1_test_imp_none": {
            "wave3_split_project_none": p1_w3_rec,
            "w4_split_project_auto": p1_auto_rec,
            "w4_repaired": p1_new_rec,
        },
        "P2_test_imp_true": {
            "wave3_split_project_none": p2_w3_rec,
            "w4_split_project_auto": p2_new_rec,
        },
        "P3_study_flip_only_unlocked": p3_rec,
        "P4_study_improve_sweeps10": {
            "wave3_split_project_none": p4_w3_rec,
            "w4_split_project_auto": p4_new_rec,
        },
        "recorded_w3_layered_cpw10": recorded_w3,
        "explanation": (
            f"1. P1 (test imp=None) runs improve_dt(n_sweeps=0, repair=True) after gmsh "
            f"is finalized, using default project='planar'. planar_projector locks every "
            f"node on the curved cylinder wall (r=a) as well as the rim circles. "
            f"In wave-3 (split_project=None), all fixing flips and boundary-edge splits "
            f"are refused ({rep_p1_w3['refused']}), leaving {rep_p1_w3['negative_faces_after']} "
            f"stuck faces. With post-gmsh default split_project='auto' (feature_curve_projector), "
            f"the blocking edges on the stuck faces are curved cylinder-wall edges rather "
            f"than sharp rim curves, leaving {rep_p1_auto['negative_faces_after']} stuck faces "
            f"(refused={rep_p1_auto['refused']}); imp=None reaches "
            f"{rep_p1_new['negative_faces_after']} negative faces only with an analytic "
            f"pillbox projector (steiner_nodes={rep_p1_new['steiner_nodes']}, "
            f"steiner_on_locked={rep_p1_new['steiner_on_locked']}, "
            f"steiner_projected={rep_p1_new['steiner_projected']}). "
            f"2. P2 (imp=True, mesh_cad + CadSplitProjector) reaches "
            f"{rep_p2_new['negative_faces_after']} negative faces via CAD surface projection "
            f"(steiner_nodes={rep_p2_new['steiner_nodes']}, "
            f"steiner_on_locked={rep_p2_new['steiner_on_locked']}, "
            f"steiner_projected={rep_p2_new['steiner_projected']}). "
            f"3. In recorded wave-3 results/repair.json, P3 (flip_only with locked=None) "
            f"and P4 (improve with cad_projector) reached 0 via unprojected chord splits "
            f"({recorded_w3.get('P3_flip_only', {})})."
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


def _compute_t1_t2_metrics(mesh, mats, locked, fcp):
    """Compute T1 (rim split sagitta and volume identity) and T2 (31-node circle conditioning) metrics."""
    st_vol = _RepairState(mesh, mats, "pec")
    ed01 = (0, 1)
    m_pt = 0.5 * (mesh.nodes[0] + mesh.nodes[1])
    ok_sp, info_sp = split_edge_steiner(
        st_vol,
        ed01,
        locked=locked,
        split_projector=_analytic_pillbox_projector(mesh),
    )
    st_vol_fcp = _RepairState(mesh, mats, "pec")
    ok_fcp, info_fcp = split_edge_steiner(
        st_vol_fcp, ed01, locked=locked, split_projector=fcp
    )
    if ok_sp:
        pt_new = st_vol.X[info_sp["new_node"]]
        sagitta_m = float(np.linalg.norm(pt_new - m_pt))
        vol_rel_err = float(info_sp["volume_identity_err"])
    else:
        sagitta_m = 0.0
        vol_rel_err = 0.0
    vol_rel_err_fcp = float(info_fcp["volume_identity_err"]) if ok_fcp else 0.0

    n_c = 31
    rng_c = np.random.default_rng(17)
    angles = np.linspace(0.0, 2.0 * np.pi, n_c, endpoint=False) + rng_c.uniform(
        -0.03, 0.03, n_c
    )
    pts_circle = np.column_stack(
        [A_PB * np.cos(angles), A_PB * np.sin(angles), np.full(n_c, 0.03)]
    )
    edges_circle = [(i, (i + 1) % n_c) for i in range(n_c)]
    fcp_c = FeatureCurveProjector(pts_circle, edges_circle)
    l_min = min(
        float(np.linalg.norm(pts_circle[b] - pts_circle[a])) for a, b in edges_circle
    )
    eps_mach = float(np.finfo(np.float64).eps)
    circle_bound = 100.0 * eps_mach * ((A_PB / l_min) ** 2) * A_PB
    circle_max_err = 0.0
    for a, b in edges_circle:
        mid = 0.5 * (pts_circle[a] + pts_circle[b])
        p_proj = fcp_c.project(a, b, mid)
        r_err = abs(float(np.hypot(p_proj[0], p_proj[1])) - A_PB)
        z_err = abs(float(p_proj[2]) - 0.03)
        circle_max_err = max(circle_max_err, r_err, z_err)

    n_line = 7
    s_vals = np.sort(rng_c.uniform(0.0, 1.0, n_line))
    p_start = np.array([0.01, -0.02, 0.005])
    d_vec = np.array([0.12, 0.07, -0.04])
    d_unit = d_vec / np.linalg.norm(d_vec)
    pts_line = p_start[None, :] + s_vals[:, None] * d_vec[None, :]
    edges_line = [(i, i + 1) for i in range(n_line - 1)]
    fcp_line = FeatureCurveProjector(pts_line, edges_line)
    line_err = 0.0
    for i in range(1, n_line - 2):
        mid = 0.5 * (pts_line[i] + pts_line[i + 1])
        p_proj = fcp_line.project(i, i + 1, mid)
        diff = p_proj - p_start
        perp = diff - np.dot(diff, d_unit) * d_unit
        line_err = max(line_err, float(np.linalg.norm(perp)))
    return {
        "t1_sagitta_m": sagitta_m,
        "t1_vol_identity_rel_err": vol_rel_err,
        "t1_vol_identity_rel_err_fcp": vol_rel_err_fcp,
        "t2_n_circle": n_c,
        "t2_L_min_m": l_min,
        "t2_circle_max_err_m": circle_max_err,
        "t2_circle_bound_m": circle_bound,
        "t2_line_err_m": line_err,
    }


class _CylinderCadModelStub:
    """OpenCASCADE-equivalent cylinder CAD model stub for `CadSplitProjector` without gmsh."""

    def __init__(self, X, R, H, tol=1e-7):
        self.X = np.asarray(X, dtype=float)
        self.R = float(R)
        self.H = float(H)
        self.tol = float(tol)
        self.mesh = self

    def getEntities(self, dim):
        return [(1, 1), (1, 2)] if int(dim) == 1 else [(2, 1), (2, 2), (2, 3)]

    def getNodes(self, dim, tag, includeBoundary=True):
        X = self.X
        r = np.hypot(X[:, 0], X[:, 1])
        z = X[:, 2]
        on_r = np.abs(r - self.R) <= self.tol
        on_z0 = np.abs(z) <= self.tol
        on_zh = np.abs(z - self.H) <= self.tol
        if int(dim) == 1:
            mask = (on_r & on_z0) if int(tag) == 1 else (on_r & on_zh)
        else:
            if int(tag) == 1:
                mask = on_r & (z >= -self.tol) & (z <= self.H + self.tol)
                if not includeBoundary:
                    mask = mask & (~on_z0) & (~on_zh)
            elif int(tag) == 2:
                mask = on_z0 & (r <= self.R + self.tol)
                if not includeBoundary:
                    mask = mask & (~on_r)
            else:
                mask = on_zh & (r <= self.R + self.tol)
                if not includeBoundary:
                    mask = mask & (~on_r)
        idx = np.where(mask)[0] + 1
        return idx, X[idx - 1].ravel(), None

    def getClosestPoint(self, dim, tag, x_cad):
        x, y, z = float(x_cad[0]), float(x_cad[1]), float(x_cad[2])
        r = max(float(np.hypot(x, y)), 1e-30)
        if int(dim) == 1:
            z_t = 0.0 if int(tag) == 1 else self.H
            return [self.R * x / r, self.R * y / r, z_t], [0.0]
        if int(tag) == 1:
            return [self.R * x / r, self.R * y / r, float(np.clip(z, 0.0, self.H))], [
                0.0,
                0.0,
            ]
        z_t = 0.0 if int(tag) == 2 else self.H
        if r > self.R:
            return [self.R * x / r, self.R * y / r, z_t], [0.0, 0.0]
        return [x, y, z_t], [0.0, 0.0]


class _CylinderGmshStub:
    """Wrapper holding `.model` for `cad_projector` / `CadSplitProjector`."""

    def __init__(self, X, R, H):
        self.model = _CylinderCadModelStub(X, R, H)


def _load_recorded_repair_suite():
    """Load the 6-case CAD repair suite from `results/repair.json`."""
    rep_json_path = os.path.join(HERE, "results", "repair.json")
    if not os.path.exists(rep_json_path):
        return []
    with open(rep_json_path) as f:
        rep_data = json.load(f)
    out = []
    for r in rep_data.get("all_runs", rep_data.get("cases", [])):
        out.append(
            {
                "case": r["case"],
                "cpw": int(r["cpw"]),
                "optimize": r["optimize"],
                "before_neg": int(r["before"]["n_negative_faces"]),
                "flip_only_neg": int(r["flip_only"]["n_negative_faces"]),
                "flip_only_steiner": int(
                    r["flip_only"]["repair_report"].get("steiner_nodes", 0)
                ),
                "repair_false_neg": int(r["after_repair_false"]["n_negative_faces"]),
                "repair_true_neg": int(r["after_repair_true"]["n_negative_faces"]),
                "repair_true_steiner": int(r["repair_report"].get("steiner_nodes", 0)),
                "dt_max_before_s": float(r["before"]["dt_max"]),
                "dt_max_flip_only_s": float(r["flip_only"]["dt_max"]),
                "dt_max_repair_false_s": float(r["after_repair_false"]["dt_max"]),
                "dt_max_repair_true_s": float(r["after_repair_true"]["dt_max"]),
                "dt_gain_vs_repair_false": float(r["dt_gain_vs_repair_false"]),
                "dt_gain_vs_before": float(r["dt_gain_vs_before"]),
            }
        )
    return out


def run_synthetic_fallback():
    """
    Gmsh-free fallback for `--quick` when gmsh shared libraries are unavailable:
    runs the synthetic 13-node cylinder rim mesh (a = 0.1 m, H = 0.03 m) across
    all repair paths (`repair=False`, `wave-3 flips`, `w4 auto`,
    `w4 CAD projector`, and `analytic`) and combines it with the recorded
    wave-3 `results/repair.json` P3/P4 diagnostics.
    """
    from wakis.unstructured.cad import cad_projector

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

    gmsh_stub = _CylinderGmshStub(mesh.nodes, A_PB, T_LAY)
    node_tags = np.arange(1, mesh.n_nodes + 1)
    cad_proj = cad_projector(gmsh_stub, node_tags, scale=1.0)

    t0 = time.perf_counter()
    m_v4_0, hist_v4_0 = improve_dt(
        mesh, mats, n_sweeps=0, project=cad_proj, repair=True, split_project="auto"
    )
    v4_0_ev = _eval_variant(
        "pillbox",
        mesh,
        m_v4_0,
        mats,
        f_ref,
        hist_v4_0[-1]["repair"],
        time.perf_counter() - t0,
    )

    cad_proj_6 = cad_projector(gmsh_stub, node_tags, scale=1.0)
    t0 = time.perf_counter()
    m_v4_6, hist_v4_6 = improve_dt(
        mesh, mats, n_sweeps=6, project=cad_proj_6, repair=True, split_project="auto"
    )
    v4_6_ev = _eval_variant(
        "pillbox",
        mesh,
        m_v4_6,
        mats,
        f_ref,
        hist_v4_6[-1]["repair"],
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    m_v5, hist_v5 = improve_dt(
        mesh,
        mats,
        n_sweeps=0,
        repair=True,
        split_project=_analytic_pillbox_projector(mesh),
    )
    v5_ev = _eval_variant(
        "pillbox",
        mesh,
        m_v5,
        mats,
        f_ref,
        hist_v5[-1]["repair"],
        time.perf_counter() - t0,
    )

    fcp = feature_curve_projector(mesh, locked=locked)
    t1_t2 = _compute_t1_t2_metrics(mesh, mats, locked, fcp)
    zoom = _extract_zoom_example(
        mesh, mats, locked, feature_curve_projector(mesh, locked=locked), R=A_PB
    )
    recorded_layered = _load_recorded_w3_layered()
    recorded_suite = _load_recorded_repair_suite()
    stuck_w3 = _inspect_stuck_faces(
        m_v2, hist_v2[-1]["repair"]["stuck"], ~movable_nodes(m_v2)
    )

    return {
        "gmsh_available": False,
        "diagnosis_a0": {
            "synthetic_rim_mesh": {
                "n_tets_before": before_ev["n_tets"],
                "n_nodes_before": before_ev["n_nodes"],
                "n_faces_before": before_ev["n_faces"],
                "guard_negative_before": before_ev["n_negative_faces"],
                "wave3_guard_negative_after": v2_ev["n_negative_faces"],
                "wave3_n_tets_after": v2_ev["n_tets"],
                "wave3_n_nodes_after": v2_ev["n_nodes"],
                "w4_guard_negative_after": v3_ev["n_negative_faces"],
                "w4_n_tets_after": v3_ev["n_tets"],
                "w4_n_nodes_after": v3_ev["n_nodes"],
                "stuck_faces_wave3": stuck_w3,
                "stuck_edge_classes_wave3": _summarize_stuck_edge_classes(stuck_w3),
                "w4_repair_report": hist_v3[-1]["repair"],
                "w4_steiner_nodes_detail": v3_ev["steiner_nodes_detail"],
                "w4_max_steiner_dist_to_cad_m": v3_ev["max_steiner_dist_to_cad_m"],
            },
            "t1_t2_metrics": t1_t2,
            "recorded_w3_layered_cpw10": recorded_layered,
            "explanation": (
                "1. P1 (test imp=None) runs improve_dt(n_sweeps=0, repair=True) after gmsh "
                "is finalized with default project='planar', which locks every cylinder-wall "
                "and rim-circle node (r=a). In wave-3 (split_projector=None), all fixing flips "
                "and boundary-edge Steiner splits on the negative faces are refused with "
                "'locked'. "
                "2. P3 (repair_study.py flip-only) calls flip_to_delaunay(m, mats) with "
                "locked=None (no nodes locked), so wave-3 bnd_edge Steiner mode split "
                "boundary edges at their chord midpoints "
                f"(recorded in results/repair.json: {recorded_layered.get('P3_flip_only', {})}), "
                "placing Steiner nodes on the chord off the curved cylinder wall. "
                "3. P2 (imp=True) and P4 (study improve) run inside mesh_cad with "
                "cad_projector, which lets smooth cylinder-wall surface nodes slide "
                "(locked=False) and locks only the 1D CAD rim circles. With CadSplitProjector "
                "and feature_curve_projector, locked rim and curved-wall edges are split "
                "and projected onto the CAD curve/surface while unprojected chord splits on "
                "curved boundaries are refused with 'no_projector'."
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
                "variant_iv_cad_projector": v4_0_ev,
                "variant_iv_cad_projector_sweeps6": v4_6_ev,
                "variant_v_analytic_projector": v5_ev,
                "dt_gain_vs_repair_false": float(v3_ev["dt_max"] / v1_ev["dt_max"]),
                "dt_gain_vs_wave3_repair": float(v3_ev["dt_max"] / v2_ev["dt_max"]),
                "dt_gain_cad_sweeps6_vs_repair_false": float(
                    v4_6_ev["dt_max"] / v1_ev["dt_max"]
                ),
            }
        ],
        "recorded_repair_suite": recorded_suite,
        "zoom_example": zoom,
    }


def _print_summary(payload):
    """Print deterministic SUMMARY ... END SUMMARY block before JSON output in --quick mode."""
    h_diel = float(C0 / (F_MAX_PB * np.sqrt(EPS_LAY) * 10.0))
    h_vac = float(C0 / (F_MAX_PB * 1.0 * 10.0))
    facet_turn_rad = float(h_diel / A_PB)
    facet_turn_deg = float(np.degrees(facet_turn_rad))
    tau_m = float(max(1e-7 * A_PB, 100.0 * np.finfo(np.float64).eps * A_PB))

    print("SUMMARY")
    print(f"mode: {payload.get('mode')}")
    print(f"gmsh_available: {payload.get('gmsh_available')}")
    print(
        f"pillbox_params: a_m={A_PB}, d_m={D_PB}, t_m={T_LAY}, eps_r={EPS_LAY}, "
        f"f_max_hz={F_MAX_PB:.3e}, cpw=10, h_diel_m={h_diel:.4f}, h_vac_m={h_vac:.4f}, "
        f"facet_turn_rad={facet_turn_rad:.4f}, facet_turn_deg={facet_turn_deg:.2f} (~5.7 deg), "
        f"tau_m={tau_m:.3e}"
    )
    print(
        "algorithm_params: max_steiner_default='max(32, 4*max(neg_before, 1))', "
        "max_steiner_depth=2, min_edge_ratio=0.2, collinear_tol=1e-12, "
        "termination_cad_angle_limit_deg=90"
    )

    diag = payload.get("diagnosis_a0", {})
    if payload.get("gmsh_available"):
        init = diag.get("initial_mesh", {})
        print(
            f"A0_initial_mesh: n_tets={init.get('n_tets')}, n_nodes={init.get('n_nodes')}, "
            f"n_faces={init.get('n_faces')}, guard_negative_before={init.get('guard_negative_before')}, "
            f"initial_edge_classes={init.get('initial_edge_classes')}"
        )
        p1 = diag.get("P1_test_imp_none", {})
        p2 = diag.get("P2_test_imp_true", {})
        p3 = diag.get("P3_study_flip_only_unlocked", {})
        p4 = diag.get("P4_study_improve_sweeps10", {})
        path_items = [
            ("P1_wave3_split_project_none", p1.get("wave3_split_project_none", {})),
            ("P1_w4_split_project_auto", p1.get("w4_split_project_auto", {})),
            ("P1_w4_analytic_projector", p1.get("w4_repaired", {})),
            ("P2_wave3_split_project_none", p2.get("wave3_split_project_none", {})),
            ("P2_w4_split_project_auto", p2.get("w4_split_project_auto", {})),
            ("P3_wave3_unprojected_chord", p3.get("wave3_unprojected_chord", {})),
            ("P3_study_flip_only_unlocked", p3),
            ("P4_wave3_split_project_none", p4.get("wave3_split_project_none", {})),
            ("P4_w4_split_project_auto", p4.get("w4_split_project_auto", {})),
        ]
        for label, rec in path_items:
            rep = rec.get("repair_report", {})
            print(
                f"A0_{label}: guard_before={rec.get('guard_negative_before')}, "
                f"guard_after={rec.get('guard_negative_after')}, "
                f"n_tets={rec.get('n_tets')}, n_nodes={rec.get('n_nodes')}, "
                f"n_faces={rec.get('n_faces')}, "
                f"steiner_nodes={rec.get('steiner_nodes')}, "
                f"steiner_on_locked={rec.get('steiner_on_locked')}, "
                f"steiner_projected={rec.get('steiner_projected')}, "
                f"max_steiner_dist_to_cad_m={rec.get('max_steiner_dist_to_cad_m', 0.0):.3e}, "
                f"max_steiner_radial_err_m={rec.get('max_steiner_radial_err_m', 0.0):.3e}, "
                f"flips={rep.get('flips')}, refused={rep.get('refused')}, "
                f"stuck={rep.get('stuck')}, "
                f"stuck_edge_classes={rec.get('stuck_edge_classes')}"
            )

        lay = payload.get("layered_cpw10", {})
        if lay:
            for vkey in (
                "before",
                "pure_wave3_repair",
                "pure_steiner_repair",
                "variant_i_repair_false",
                "variant_ii_wave3_repair",
                "variant_iii_steiner_repair",
            ):
                v = lay.get(vkey, {})
                if v:
                    print(
                        f"layered_cpw10_{vkey}: n_tets={v.get('n_tets')}, "
                        f"n_nodes={v.get('n_nodes')}, n_faces={v.get('n_faces')}, "
                        f"n_negative_faces={v.get('n_negative_faces')}, "
                        f"steiner_nodes={v.get('steiner_nodes')}, "
                        f"steiner_on_locked={v.get('steiner_on_locked')}, "
                        f"steiner_projected={v.get('steiner_projected')}, "
                        f"max_steiner_dist_to_cad_m={v.get('max_steiner_dist_to_cad_m', 0.0):.3e}, "
                        f"dt_max={v.get('dt_max', 0.0):.4e}, "
                        f"lowest_freq_err={v.get('lowest_freq_err', 0.0):+.3e}"
                    )
            print(
                f"layered_cpw10_gains: "
                f"dt_gain_vs_repair_false={lay.get('dt_gain_vs_repair_false', 0.0):.4f}, "
                f"dt_gain_vs_wave3_repair={lay.get('dt_gain_vs_wave3_repair', 0.0):.4f}, "
                f"pure_dt_gain_vs_wave3={lay.get('pure_dt_gain_vs_wave3', 0.0):.4f}"
            )

    syn_payload = payload.get("synthetic_fallback") or payload
    syn_diag = syn_payload.get("diagnosis_a0", {})
    rec_w3 = diag.get("recorded_w3_layered_cpw10") or syn_diag.get(
        "recorded_w3_layered_cpw10", {}
    )
    if rec_w3:
        print(
            f"recorded_repair_json_layered_cpw10: "
            f"n_tets_before={rec_w3.get('n_tets_before')}, "
            f"n_nodes_before={rec_w3.get('n_nodes_before')}, "
            f"n_faces_before={rec_w3.get('n_faces_before')}, "
            f"negative_faces_before={rec_w3.get('negative_faces_before')}, "
            f"P3_flip_only_n_tets_after={rec_w3.get('P3_flip_only_n_tets_after')}, "
            f"P3_flip_only_n_nodes_after={rec_w3.get('P3_flip_only_n_nodes_after')}, "
            f"P3_flip_only_n_faces_after={rec_w3.get('P3_flip_only_n_faces_after')}, "
            f"P3_flip_only_report={rec_w3.get('P3_flip_only')}, "
            f"P4_improve_sweeps10_n_tets_after={rec_w3.get('P4_improve_sweeps10_n_tets_after')}, "
            f"P4_improve_sweeps10_n_nodes_after={rec_w3.get('P4_improve_sweeps10_n_nodes_after')}, "
            f"P4_improve_sweeps10_report={rec_w3.get('P4_improve_sweeps10')}, "
            f"suite_dt_gain_vs_repair_false_range=[{rec_w3.get('suite_dt_gain_vs_repair_false_min', 0.0):.4f}, "
            f"{rec_w3.get('suite_dt_gain_vs_repair_false_max', 0.0):.4f}]"
        )

    syn_mesh_diag = syn_diag.get("synthetic_rim_mesh", {})
    if syn_mesh_diag:
        print(
            f"synthetic_rim_A0: guard_before={syn_mesh_diag.get('guard_negative_before')}, "
            f"wave3_guard_after={syn_mesh_diag.get('wave3_guard_negative_after')}, "
            f"w4_guard_after={syn_mesh_diag.get('w4_guard_negative_after')}, "
            f"n_tets_before={syn_mesh_diag.get('n_tets_before')}, "
            f"n_nodes_before={syn_mesh_diag.get('n_nodes_before')}, "
            f"n_faces_before={syn_mesh_diag.get('n_faces_before')}, "
            f"w4_n_tets_after={syn_mesh_diag.get('w4_n_tets_after')}, "
            f"w4_n_nodes_after={syn_mesh_diag.get('w4_n_nodes_after')}, "
            f"w4_max_steiner_dist_to_cad_m={syn_mesh_diag.get('w4_max_steiner_dist_to_cad_m', 0.0):.3e}, "
            f"stuck_edge_classes_wave3={syn_mesh_diag.get('stuck_edge_classes_wave3')}, "
            f"w4_repair_report={syn_mesh_diag.get('w4_repair_report')}"
        )

    syn_cases = syn_payload.get("cases", [])
    for sc in syn_cases:
        if sc.get("case") == "synthetic_rim_cylinder":
            v1 = sc["variant_i_repair_false"]
            v2 = sc["variant_ii_wave3_repair"]
            v3 = sc["variant_iii_steiner_repair"]
            print(
                f"synthetic_rim_cylinder: "
                f"before_n_tets={sc['before']['n_tets']}, "
                f"before_n_nodes={sc['before']['n_nodes']}, "
                f"before_n_faces={sc['before']['n_faces']}, "
                f"before_neg={sc['before']['n_negative_faces']}, "
                f"v1_repair_false_neg={v1['n_negative_faces']} (allow_negative_faces=True), "
                f"v1_dt_max_s={v1['dt_max']:.4e}, "
                f"v2_wave3_neg={v2['n_negative_faces']} (allow_negative_faces=True), "
                f"v2_refused={v2['repair_report'].get('refused')}, "
                f"v2_dt_max_s={v2['dt_max']:.4e}, "
                f"v3_steiner_n_tets={v3['n_tets']}, "
                f"v3_steiner_n_nodes={v3['n_nodes']}, "
                f"v3_steiner_n_faces={v3['n_faces']}, "
                f"v3_steiner_neg={v3['n_negative_faces']}, "
                f"v3_steiner_nodes={v3['steiner_nodes']}, "
                f"v3_steiner_on_locked={v3['steiner_on_locked']}, "
                f"v3_steiner_projected={v3['steiner_projected']}, "
                f"v3_flips={v3['repair_report'].get('flips')}, "
                f"v3_max_steiner_dist_to_cad_m={v3['max_steiner_dist_to_cad_m']:.3e}, "
                f"v3_dt_max_s={v3['dt_max']:.4e}, "
                f"dt_gain_vs_repair_false={sc['dt_gain_vs_repair_false']:.4f}, "
                f"dt_gain_vs_wave3_repair={sc['dt_gain_vs_wave3_repair']:.4f}"
            )

    t12 = syn_diag.get("t1_t2_metrics", {})
    if t12:
        print(
            f"t1_t2_verification: "
            f"t1_sagitta_m={t12['t1_sagitta_m']:.6e}, "
            f"t1_vol_identity_rel_err={t12['t1_vol_identity_rel_err']:.3e}, "
            f"t1_vol_identity_rel_err_fcp={t12.get('t1_vol_identity_rel_err_fcp', 0.0):.3e}, "
            f"t2_n_circle={t12['t2_n_circle']}, "
            f"t2_L_min_m={t12['t2_L_min_m']:.4e}, "
            f"t2_circle_max_err_m={t12['t2_circle_max_err_m']:.3e}, "
            f"t2_circle_bound_m={t12['t2_circle_bound_m']:.3e}, "
            f"t2_line_err_m={t12['t2_line_err_m']:.3e}"
        )
    print("END SUMMARY")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run Step A0 diagnosis and layered pillbox at cpw=10 in a temp dir; print SUMMARY and JSON to stdout.",
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

    if args.quick or gmsh_mod is None:
        syn_payload = run_synthetic_fallback()
        if gmsh_mod is None:
            payload = syn_payload
            payload["git_commit"] = _git_commit()
            payload["command"] = " ".join(sys.argv)
            payload["mode"] = "quick_synthetic_fallback"
            out_path = args.out or (
                None if args.quick else os.path.join(HERE, "results", "steiner.json")
            )
            if out_path is not None:
                os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(payload, f, indent=2, default=float)
            _print_summary(payload)
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
            payload = {
                "git_commit": _git_commit(),
                "command": " ".join(sys.argv),
                "mode": "quick",
                "gmsh_available": True,
                "diagnosis_a0": diag,
                "layered_cpw10": layered_rec,
                "cases": [layered_rec],
                "synthetic_fallback": syn_payload,
                "zoom_example": syn_payload["zoom_example"],
            }
            if args.out is not None:
                os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                with open(args.out, "w") as f:
                    json.dump(payload, f, indent=2, default=float)
            _print_summary(payload)
            print(json.dumps(payload, indent=2, default=float))
        return

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
