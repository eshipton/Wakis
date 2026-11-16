# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Adaptive mesh refinement (AMR) for 3D tetrahedral FIT (`UnstructuredFIT3D`).

Provides:
* Freudenthal / Kuhn structured tetrahedral meshes (`kuhn_mesh`, `l_prism_mesh`);
* Rivara longest-edge bisection with iterative conformity closure and boundary
  projection (`bisect`, `refine_uniform`, `label_constraint_faces`,
  `sphere_projector`);
* Whitney 1-form field and curl reconstruction (`whitney_fields`);
* Coefficient-weighted residual and Zienkiewicz-Zhu curl error indicators
  (`residual_indicator`, `zz_curl_indicator`) and Dörfler bulk marking
  (`dorfler_mark`);
* Adaptive cavity eigenmode loop (`adapt_eigen`, `AdaptResult`) and final-mesh
  Voronoi export (`voronoi_export`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.constants import c as C0

from ._core import cell_materials
from .estimate import estimate_memory
from .improve import constraint_faces, improve_dt, neg_faces
from .mesh import SimplexMesh
from .solver3d import UnstructuredFIT3D

__all__ = [
    "AdaptResult",
    "adapt_eigen",
    "bisect",
    "dorfler_mark",
    "kuhn_mesh",
    "l_prism_mesh",
    "label_constraint_faces",
    "refine_uniform",
    "residual_indicator",
    "sphere_projector",
    "voronoi_export",
    "whitney_fields",
    "zz_curl_indicator",
]

# Local edge pairs of a tetrahedron (matching SimplexMesh._local_edges)
_LOC_E = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], dtype=np.int64)

# Local faces opposite vertex 0, 1, 2, 3 (matching SimplexMesh._local_faces)
_LOC_F = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int64)

# Even permutations (a, b, c, d) of (0, 1, 2, 3) for each of the 6 local edges (a, b)
_EVEN_PERM = np.array(
    [
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        [1, 2, 0, 3],
        [1, 3, 2, 0],
        [2, 3, 0, 1],
    ],
    dtype=np.int64,
)

# Kuhn (Freudenthal) 6 axis-permutation paths from (0,0,0) to (1,1,1)
_KUHN_PATHS = np.array(
    [
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]],
        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]],
        [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]],
    ],
    dtype=np.int64,
)

_EDGE_CODE_STRIDE = 1 << 30


def kuhn_mesh(n_xyz, h_m, origin_m=(0.0, 0.0, 0.0), keep=None, tag=None):
    """
    Freudenthal (Kuhn) tetrahedral mesh of a Cartesian grid of cubes.

    Each kept cube is subdivided into 6 Kuhn tetrahedra along the 6 axis
    permutations from its lower corner ``v000`` to its opposite corner ``v111``.

    Parameters
    ----------
    n_xyz : tuple of 3 int or int
        Number of cubes ``(nx, ny, nz)`` along each axis.
    h_m : float
        Cube edge length [m].
    origin_m : tuple of 3 float, optional
        Lower-corner coordinates ``(x0, y0, z0)`` [m].
    keep : callable, optional
        Vectorized predicate ``keep(xc, yc, zc) -> bool`` evaluated at cube
        centres [m] to select active cubes.
    tag : callable, optional
        Vectorized function ``tag(xc, yc, zc) -> int`` evaluated at kept cube
        centres [m] to assign ``cell_tags``.

    Returns
    -------
    SimplexMesh
        Conforming tetrahedral mesh.
    """
    if np.ndim(n_xyz) == 0:
        nx = ny = nz = int(n_xyz)
    else:
        nx, ny, nz = (int(v) for v in n_xyz)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("n_xyz counts must be positive")
    h = float(h_m)
    if h <= 0.0:
        raise ValueError("h_m must be positive")
    ox, oy, oz = (float(v) for v in origin_m)

    xs = ox + np.arange(nx + 1, dtype=float) * h
    ys = oy + np.arange(ny + 1, dtype=float) * h
    zs = oz + np.arange(nz + 1, dtype=float) * h
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    ix, iy, iz = np.meshgrid(
        np.arange(nx, dtype=np.int64),
        np.arange(ny, dtype=np.int64),
        np.arange(nz, dtype=np.int64),
        indexing="ij",
    )
    ix, iy, iz = ix.ravel(), iy.ravel(), iz.ravel()
    xc = ox + (ix + 0.5) * h
    yc = oy + (iy + 0.5) * h
    zc = oz + (iz + 0.5) * h

    if keep is not None:
        mask = np.asarray(keep(xc, yc, zc), dtype=bool)
        ix, iy, iz = ix[mask], iy[mask], iz[mask]
        xc, yc, zc = xc[mask], yc[mask], zc[mask]

    stride_x = (ny + 1) * (nz + 1)
    stride_y = nz + 1
    base = np.stack([ix, iy, iz], axis=1)[:, None, None, :]  # (ncube, 1, 1, 3)
    verts_ijk = base + _KUHN_PATHS[None, :, :, :]  # (ncube, 6, 4, 3)
    cells = (
        verts_ijk[..., 0] * stride_x + verts_ijk[..., 1] * stride_y + verts_ijk[..., 2]
    ).reshape(-1, 4)

    cell_tags = None
    if tag is not None:
        cube_tags = np.asarray(tag(xc, yc, zc), dtype=np.int64)
        cell_tags = np.repeat(cube_tags, 6)

    return SimplexMesh(nodes, cells, cell_tags)


def l_prism_mesh(n_per_unit, n_z, height_m=0.6):
    """
    Kuhn tetrahedral mesh of the re-entrant L-shaped prism cavity.

    Cross-section ``L = (-1, 1)^2 \\ [0, 1] x [-1, 0]`` [m] extruded over
    ``z in [0, height_m]`` [m] with cube size ``h = 1 / n_per_unit`` [m].

    Parameters
    ----------
    n_per_unit : int
        Cubes per metre in the transverse plane (``h_m = 1 / n_per_unit``).
    n_z : int
        Number of cube layers along ``z``.
    height_m : float, optional
        Prism height ``d`` [m] (default 0.6 m). Must equal ``n_z / n_per_unit``
        to round-off.

    Returns
    -------
    SimplexMesh
        Tetrahedral mesh of the L-prism.
    """
    n_per_unit = int(n_per_unit)
    n_z = int(n_z)
    height_m = float(height_m)
    if n_per_unit <= 0 or n_z <= 0:
        raise ValueError("n_per_unit and n_z must be positive")
    h = 1.0 / n_per_unit
    if abs(n_z * h - height_m) > 1e-12 * max(1.0, abs(height_m)):
        raise ValueError(
            f"n_z * (1 / n_per_unit) = {n_z * h:.15g} m does not match "
            f"height_m = {height_m:.15g} m"
        )
    return kuhn_mesh(
        (2 * n_per_unit, 2 * n_per_unit, n_z),
        h,
        origin_m=(-1.0, -1.0, 0.0),
        keep=lambda xc, yc, zc: ~((xc > 0.0) & (yc < 0.0)),
    )


def label_constraint_faces(mesh, classify=None):
    """
    Assign integer patch labels to boundary and material-interface faces.

    Parameters
    ----------
    mesh : SimplexMesh
        3D tetrahedral mesh.
    classify : callable, optional
        ``classify(centroids, normals) -> array_like of int`` returning a
        non-negative patch label for each constraint face given its centroid
        [m] and unit normal. Defaults to ``0`` on all constraint faces.

    Returns
    -------
    ndarray of int, shape (n_faces,)
        ``-1`` on unconstrained interior faces; patch labels ``>= 0`` on
        constraint faces.
    """
    patch = np.full(mesh.n_faces, -1, dtype=np.int64)
    cf = np.flatnonzero(constraint_faces(mesh))
    if cf.size == 0:
        return patch
    if classify is None:
        patch[cf] = 0
    else:
        centroids = mesh.nodes[mesh.faces[cf]].mean(axis=1)
        normals = mesh.face_normal[cf]
        patch[cf] = np.asarray(classify(centroids, normals), dtype=np.int64)
    return patch


def sphere_projector(center_m, radius_m, patch=0):
    """
    Radial projector onto a sphere for boundary nodes of a given patch label.

    Parameters
    ----------
    center_m : array_like, shape (3,)
        Sphere centre [m].
    radius_m : float
        Sphere radius [m].
    patch : int, optional
        Patch label to project (default 0); nodes with other labels are
        returned unchanged.

    Returns
    -------
    callable
        ``project(points, patches) -> projected_points`` [m].
    """
    center = np.asarray(center_m, dtype=float).reshape(1, 3)
    radius = float(radius_m)
    target_patch = int(patch)

    def project(points, patches):
        pts = np.asarray(points, dtype=float).copy()
        sel = np.asarray(patches) == target_patch
        if np.any(sel):
            d = pts[sel] - center
            r = np.linalg.norm(d, axis=1, keepdims=True)
            pts[sel] = center + radius * (d / r)
        return pts

    return project


def _pick_bisect_edge(nodes, active_cells, l2_ref):
    """
    Select the local edge (0..5) of each active cell with the largest global key.

    Global edge key: ``(round(L_e^2 / l2_ref, 10), lo_node, hi_node)``.
    """
    pairs = active_cells[:, _LOC_E]
    lo = np.minimum(pairs[..., 0], pairs[..., 1])
    hi = np.maximum(pairs[..., 0], pairs[..., 1])
    codes = lo * _EDGE_CODE_STRIDE + hi
    u_codes, inv = np.unique(codes.ravel(), return_inverse=True)
    u_lo = u_codes // _EDGE_CODE_STRIDE
    u_hi = u_codes % _EDGE_CODE_STRIDE
    d = nodes[u_hi] - nodes[u_lo]
    u_l2_round = np.round(np.einsum("ij,ij->i", d, d) / l2_ref, 10)
    order = np.lexsort((u_hi, u_lo, u_l2_round))
    rank = np.empty(len(u_codes), dtype=np.int64)
    rank[order] = np.arange(len(u_codes), dtype=np.int64)
    cell_ranks = rank[inv].reshape(len(active_cells), 6)
    return np.argmax(cell_ranks, axis=1)


def _split_cells(
    active_cells, active_tags, active_parent, active_orig_face, loc_edge, mid_nodes
):
    """Split each active tetrahedron (a, b, c, d) at midpoint m on (a, b)."""
    idx = np.arange(len(active_cells))[:, None]
    perm = _EVEN_PERM[loc_edge]
    abcd = active_cells[idx, perm]
    f_abcd = active_orig_face[idx, perm]

    c1 = abcd.copy()
    c1[:, 1] = mid_nodes
    f1 = f_abcd.copy()
    f1[:, 0] = -1

    c2 = abcd.copy()
    c2[:, 0] = mid_nodes
    f2 = f_abcd.copy()
    f2[:, 1] = -1

    return (
        np.vstack([c1, c2]),
        np.concatenate([active_tags, active_tags]),
        np.concatenate([active_parent, active_parent]),
        np.vstack([f1, f2]),
    )


def _find_nonconforming_cells(cells, split_code_arr):
    """Indices of cells that still have at least one edge in `split_code_arr`."""
    if split_code_arr.size == 0:
        return np.empty(0, dtype=np.int64)
    pairs = cells[:, _LOC_E]
    lo = np.minimum(pairs[..., 0], pairs[..., 1])
    hi = np.maximum(pairs[..., 0], pairs[..., 1])
    codes = lo * _EDGE_CODE_STRIDE + hi
    pos = np.searchsorted(split_code_arr, codes)
    pos_clip = np.minimum(pos, split_code_arr.size - 1)
    has_split = split_code_arr[pos_clip] == codes
    return np.flatnonzero(np.any(has_split, axis=1))


def _apply_projector(project, pts, patches):
    """Apply the boundary/interface projector to candidate midpoint coordinates."""
    return np.asarray(project(pts, patches), dtype=float)


def _tet_signed_volume(nodes, cells):
    """Signed volumes of tetrahedra `cells` with vertex coordinates `nodes` [m^3]."""
    P = nodes[cells]
    return np.linalg.det(P[:, 1:] - P[:, :1]) / 6.0


def bisect(mesh, marked, face_patch=None, project=None, max_rounds=1000):
    """
    Rivara longest-edge bisection (LEB) with iterative conformity closure.

    Geometry is frozen during the conformity closure; optional boundary/interface
    node projection is applied once after the final closure round, with per-node
    back-off (1 -> 1/2 -> 1/4 -> 0) if any incident tetrahedron would invert.

    Parameters
    ----------
    mesh : SimplexMesh
        Input 3D tetrahedral mesh.
    marked : array_like of int or bool
        Indices or boolean mask of tetrahedra to bisect in the initial round.
    face_patch : array_like of int, shape (n_faces,), optional
        Constraint-face patch labels (default from `label_constraint_faces`).
    project : callable, optional
        ``project(points (k, 3), patch (k,)) -> (k, 3)`` moving new nodes whose
        incident constraint faces all carry the same patch label ``>= 0``.
    max_rounds : int, optional
        Maximum closure rounds before raising `RuntimeError` (default 1000).

    Returns
    -------
    new_mesh : SimplexMesh
        Conforming refined tetrahedral mesh.
    info : dict
        ``rounds``, ``n_bisections``, ``parent``, ``n_new_nodes``,
        ``n_projected``, ``n_backoff``, ``face_patch``.
    """
    if mesh.dim != 3:
        raise ValueError("bisect requires a 3D tetrahedral SimplexMesh")
    if face_patch is None:
        face_patch = label_constraint_faces(mesh)
    else:
        face_patch = np.asarray(face_patch, dtype=np.int64).copy()
        if face_patch.shape != (mesh.n_faces,):
            raise ValueError("face_patch must have shape (mesh.n_faces,)")

    marked_arr = np.asarray(marked)
    if marked_arr.dtype == bool:
        if marked_arr.shape != (mesh.n_cells,):
            raise ValueError("boolean marked mask must have length mesh.n_cells")
        active = np.flatnonzero(marked_arr)
    else:
        active = np.unique(marked_arr.astype(np.int64))

    if active.size == 0:
        new_mesh = SimplexMesh(
            mesh.nodes.copy(), mesh.cells.copy(), mesh.cell_tags.copy()
        )
        if hasattr(mesh, "region_names"):
            new_mesh.region_names = dict(mesh.region_names)
        if hasattr(mesh, "surface_groups"):
            new_mesh.surface_groups = {
                k: np.asarray(v, dtype=np.int64).copy()
                for k, v in mesh.surface_groups.items()
            }
        info = {
            "rounds": 0,
            "n_bisections": 0,
            "parent": np.arange(mesh.n_cells, dtype=np.int64),
            "n_new_nodes": 0,
            "n_projected": 0,
            "n_backoff": 0,
            "face_patch": face_patch.copy(),
        }
        return new_mesh, info

    l2_ref = float(np.max(mesh.edge_length**2))
    nodes = mesh.nodes.copy()
    cells = mesh.cells.copy()
    tags = mesh.cell_tags.copy()
    parent = np.arange(mesh.n_cells, dtype=np.int64)
    cell_orig_face = mesh.cell_faces.copy()
    n_nodes_init = len(nodes)
    node_patches = []

    # Map edge_code -> midpoint node id across all rounds of this call
    split_code_arr = np.empty(0, dtype=np.int64)
    split_mid_arr = np.empty(0, dtype=np.int64)

    n_bisections = 0
    rounds = 0
    while active.size > 0:
        if rounds >= max_rounds:
            raise RuntimeError(
                f"bisect closure did not terminate within max_rounds={max_rounds}"
            )
        rounds += 1
        act_cells = cells[active]
        loc_edge = _pick_bisect_edge(nodes, act_cells, l2_ref)
        idx = np.arange(len(active))
        e_nodes = act_cells[idx[:, None], _LOC_E[loc_edge]]
        lo = np.minimum(e_nodes[:, 0], e_nodes[:, 1])
        hi = np.maximum(e_nodes[:, 0], e_nodes[:, 1])
        codes = lo * _EDGE_CODE_STRIDE + hi

        u_codes, inv = np.unique(codes, return_inverse=True)
        if split_code_arr.size > 0:
            pos = np.searchsorted(split_code_arr, u_codes)
            pos_clip = np.minimum(pos, split_code_arr.size - 1)
            known = split_code_arr[pos_clip] == u_codes
        else:
            known = np.zeros(len(u_codes), dtype=bool)
            pos = np.zeros(len(u_codes), dtype=np.int64)

        u_mids = np.empty(len(u_codes), dtype=np.int64)
        if np.any(known):
            u_mids[known] = split_mid_arr[pos[known]]

        new_mask = ~known
        if np.any(new_mask):
            new_codes = u_codes[new_mask]
            new_lo = new_codes // _EDGE_CODE_STRIDE
            new_hi = new_codes % _EDGE_CODE_STRIDE
            new_pts = 0.5 * (nodes[new_lo] + nodes[new_hi])
            n_cur = len(nodes)
            new_ids = np.arange(n_cur, n_cur + len(new_codes), dtype=np.int64)
            u_mids[new_mask] = new_ids
            nodes = np.vstack([nodes, new_pts])

            # Determine constraint patch labels of newly split edges from current constraint faces
            valid_f = cell_orig_face >= 0
            orig_f = cell_orig_face[valid_f]
            fp = face_patch[orig_f]
            has_p = fp >= 0
            if np.any(has_p):
                c_idx, k_idx = np.nonzero(valid_f)
                c_idx, k_idx = c_idx[has_p], k_idx[has_p]
                f_patches = fp[has_p]
                tris = cells[c_idx[:, None], _LOC_F[k_idx]]  # (nf_c, 3)
                f_e0 = np.sort(tris[:, [0, 1]], axis=1)
                f_e1 = np.sort(tris[:, [1, 2]], axis=1)
                f_e2 = np.sort(tris[:, [0, 2]], axis=1)
                fe_all = np.vstack([f_e0, f_e1, f_e2])
                fe_codes = fe_all[:, 0] * _EDGE_CODE_STRIDE + fe_all[:, 1]
                fe_patches = np.tile(f_patches, 3)

                p_pos = np.searchsorted(new_codes, fe_codes)
                p_clip = np.minimum(p_pos, len(new_codes) - 1)
                hit = new_codes[p_clip] == fe_codes
                if np.any(hit):
                    hit_idx = p_pos[hit]
                    hit_patch = fe_patches[hit]
                    p_min = np.full(
                        len(new_codes), np.iinfo(np.int64).max, dtype=np.int64
                    )
                    p_max = np.full(len(new_codes), -1, dtype=np.int64)
                    np.minimum.at(p_min, hit_idx, hit_patch)
                    np.maximum.at(p_max, hit_idx, hit_patch)
                    new_np = np.where((p_max >= 0) & (p_min == p_max), p_min, -1)
                else:
                    new_np = np.full(len(new_codes), -1, dtype=np.int64)
            else:
                new_np = np.full(len(new_codes), -1, dtype=np.int64)
            node_patches.append(new_np)

            # Merge into sorted (split_code_arr, split_mid_arr)
            all_codes = np.concatenate([split_code_arr, new_codes])
            all_mids = np.concatenate([split_mid_arr, new_ids])
            s_ord = np.argsort(all_codes)
            split_code_arr = all_codes[s_ord]
            split_mid_arr = all_mids[s_ord]

        mid_nodes = u_mids[inv]
        n_bisections += len(active)

        child_cells, child_tags, child_parent, child_orig_face = _split_cells(
            act_cells,
            tags[active],
            parent[active],
            cell_orig_face[active],
            loc_edge,
            mid_nodes,
        )

        keep_mask = np.ones(len(cells), dtype=bool)
        keep_mask[active] = False
        cells = np.vstack([cells[keep_mask], child_cells])
        tags = np.concatenate([tags[keep_mask], child_tags])
        parent = np.concatenate([parent[keep_mask], child_parent])
        cell_orig_face = np.vstack([cell_orig_face[keep_mask], child_orig_face])

        active = _find_nonconforming_cells(cells, split_code_arr)

    n_new_nodes = len(nodes) - n_nodes_init
    n_projected = 0
    n_backoff = 0
    if project is not None and n_new_nodes > 0:
        new_node_patch = np.concatenate(node_patches)
        proj_rel = np.flatnonzero(new_node_patch >= 0)
        if proj_rel.size > 0:
            proj_nodes = n_nodes_init + proj_rel
            pts0 = nodes[proj_nodes].copy()
            target = _apply_projector(project, pts0, new_node_patch[proj_rel])
            delta = target - pts0
            moved = np.any(delta != 0.0, axis=1)
            n_projected = int(np.sum(moved))
            alpha = np.ones(len(proj_nodes), dtype=float)
            nodes[proj_nodes] = pts0 + alpha[:, None] * delta

            vol0_min = float(np.min(_tet_signed_volume(nodes, cells)))
            if vol0_min <= 0.0:
                # Back off incident projected nodes through 1/2 -> 1/4 -> 0
                inv_map = np.full(len(nodes), -1, dtype=np.int64)
                inv_map[proj_nodes] = np.arange(len(proj_nodes), dtype=np.int64)
                for next_alpha in (0.5, 0.25, 0.0):
                    vol = _tet_signed_volume(nodes, cells)
                    bad_cells = np.flatnonzero(vol <= 0.0)
                    if bad_cells.size == 0:
                        break
                    bad_p = inv_map[np.unique(cells[bad_cells].ravel())]
                    bad_p = bad_p[bad_p >= 0]
                    alpha[bad_p] = next_alpha
                    nodes[proj_nodes] = pts0 + alpha[:, None] * delta
                n_backoff = int(np.sum(alpha < 1.0))

    # Check if any cell would be re-oriented by SimplexMesh._orient
    vol_pre = _tet_signed_volume(nodes, cells)
    neg = vol_pre < 0.0
    if np.any(neg):
        cells = cells.copy()
        cells[neg, :2] = cells[neg, 1::-1]
        cell_orig_face = cell_orig_face.copy()
        cell_orig_face[neg, :2] = cell_orig_face[neg, 1::-1]

    new_mesh = SimplexMesh(nodes, cells, tags)

    # Map original face indices onto new_mesh.faces
    parent_face = np.full(new_mesh.n_faces, -1, dtype=np.int64)
    valid_f = cell_orig_face >= 0
    if np.any(valid_f):
        parent_face[new_mesh.cell_faces[valid_f]] = cell_orig_face[valid_f]
    new_face_patch = np.where(parent_face >= 0, face_patch[parent_face], -1)

    if hasattr(mesh, "region_names"):
        new_mesh.region_names = dict(mesh.region_names)
    if hasattr(mesh, "surface_groups"):
        new_mesh.surface_groups = {
            name: np.flatnonzero(np.isin(parent_face, np.asarray(fids, dtype=np.int64)))
            for name, fids in mesh.surface_groups.items()
        }

    info = {
        "rounds": int(rounds),
        "n_bisections": int(n_bisections),
        "parent": parent,
        "n_new_nodes": int(n_new_nodes),
        "n_projected": int(n_projected),
        "n_backoff": int(n_backoff),
        "face_patch": new_face_patch,
        "parent_face": parent_face,
    }
    return new_mesh, info


def refine_uniform(mesh, generations=3, return_info=False, **kw):
    """
    Uniformly bisect all tetrahedra of `mesh` for `generations` generations.

    Parameters
    ----------
    mesh : SimplexMesh
        Input 3D tetrahedral mesh.
    generations : int, optional
        Number of bisection generations (default 3; 3 generations halve ``h``
        and split each tetrahedron into 8).
    return_info : bool, optional
        If True, return ``(mesh, info)``; otherwise return ``mesh`` with
        ``mesh.bisect_info = info`` attached.
    **kw
        Forwarded to `bisect` (``face_patch``, ``project``, ``max_rounds``).

    Returns
    -------
    SimplexMesh or (SimplexMesh, dict)
    """
    generations = int(generations)
    face_patch = kw.pop("face_patch", None)
    if face_patch is None:
        face_patch = label_constraint_faces(mesh)
    parent = np.arange(mesh.n_cells, dtype=np.int64)
    parent_face = np.arange(mesh.n_faces, dtype=np.int64)
    tot_rounds = 0
    tot_bisections = 0
    tot_new_nodes = 0
    tot_projected = 0
    tot_backoff = 0
    for _ in range(generations):
        mesh, info = bisect(mesh, np.arange(mesh.n_cells), face_patch=face_patch, **kw)
        face_patch = info["face_patch"]
        parent = parent[info["parent"]]
        pf = info["parent_face"]
        parent_face = np.where(pf >= 0, parent_face[np.maximum(pf, 0)], -1)
        tot_rounds += info["rounds"]
        tot_bisections += info["n_bisections"]
        tot_new_nodes += info["n_new_nodes"]
        tot_projected += info["n_projected"]
        tot_backoff += info["n_backoff"]
    agg_info = {
        "rounds": tot_rounds,
        "n_bisections": tot_bisections,
        "parent": parent,
        "n_new_nodes": tot_new_nodes,
        "n_projected": tot_projected,
        "n_backoff": tot_backoff,
        "face_patch": face_patch,
        "parent_face": parent_face,
    }
    mesh.bisect_info = agg_info
    if return_info:
        return mesh, agg_info
    return mesh


# Exact P1 mass matrices on a tetrahedron (divide by V) and triangle (divide by A)
_P1_TET_MASS = (np.ones((4, 4)) + np.eye(4)) / 20.0
_P1_TRI_MASS = (np.ones((3, 3)) + np.eye(3)) / 12.0


def _local_edge_signs(cells):
    """Orientation signs +1 if cells[:, a] < cells[:, b] else -1 for local edges."""
    return np.stack(
        [np.where(cells[:, a] < cells[:, b], 1.0, -1.0) for a, b in _LOC_E], axis=1
    )


def whitney_fields(mesh, e):
    """
    Reconstruct the linear Whitney 1-form field and constant curl on each cell.

    Parameters
    ----------
    mesh : SimplexMesh
        3D tetrahedral mesh.
    e : array_like, shape (n_edges,)
        Oriented edge voltages [V].

    Returns
    -------
    U : ndarray, shape (n_cells, 4, 3)
        Electric field values at the 4 local vertices of each tetrahedron [V/m].
    B : ndarray, shape (n_cells, 3)
        Constant curl ``curl u_h`` on each tetrahedron [V/m^2].
    """
    if mesh.dim != 3:
        raise ValueError("whitney_fields requires a 3D tetrahedral mesh")
    e_arr = np.asarray(e, dtype=float)
    grad = mesh.barycentric_gradients()  # (nc, 4, 3)
    sgn = _local_edge_signs(mesh.cells)  # (nc, 6)
    ep = sgn * e_arr[mesh.cell_edges]  # (nc, 6)

    U = np.zeros((mesh.n_cells, 4, 3), dtype=float)
    B = np.zeros((mesh.n_cells, 3), dtype=float)
    for p, (a, b) in enumerate(_LOC_E):
        w = ep[:, p, None]
        U[:, a, :] += w * grad[:, b, :]
        U[:, b, :] -= w * grad[:, a, :]
        B += 2.0 * w * np.cross(grad[:, a, :], grad[:, b, :])
    return U, B


def _res_element_term(h_T, mu_T, lam_h, eps_T, u_l2_T):
    """Element residual contribution h_T^2 * mu_T * ||lam_h * eps_T * u_h||_T^2."""
    return (h_T**2) * mu_T * ((lam_h * eps_T) ** 2) * u_l2_T


def _res_face_jump_weight(h_F, mu_F, lam_h, eps_F):
    """Weights (h_F * mu_F, h_F * lam_h / eps_F) of the tangential and normal jumps."""
    return h_F * mu_F, h_F * (lam_h / eps_F)


def residual_indicator(mesh, e, k2_per_m2, materials=None, bc="pec"):
    """
    Coefficient-weighted residual a-posteriori error indicator for the Maxwell
    eigenproblem.

    Normalises ``e`` internally so that ``(eps_r u_h, u_h) = 1`` and evaluates
    for each tetrahedron ``T``::

        eta_T^2 = h_T^2 mu_T || lam_h eps_T u_h ||_T^2
                + sum_{F in dT, int} 1/2 h_F [ mu_F ||[n x mu^-1 curl u_h]||_F^2
                                             + (lam_h / eps_F) ||[eps u_h . n]||_F^2 ]
                + sum_{F in dT, PMC} h_F mu_T || n x mu_T^-1 curl u_h ||_F^2

    Parameters
    ----------
    mesh : SimplexMesh
        3D tetrahedral mesh.
    e : array_like, shape (n_edges,)
        Eigenmode edge voltages [V].
    k2_per_m2 : float
        Discrete eigenvalue ``lambda_h = k_h^2`` [1/m^2].
    materials : dict, optional
        ``{tag: material}`` mapping as in `UnstructuredFIT3D`.
    bc : {"pec", "pmc", "abc"}, optional
        Outer boundary condition (default ``"pec"``; ``"abc"`` raises
        `NotImplementedError`).

    Returns
    -------
    eta2 : ndarray, shape (n_cells,)
        Per-tetrahedron squared error indicator ``eta_T^2`` [1/m^2].
    """
    if bc == "abc":
        raise NotImplementedError("residual_indicator does not support lossy bc='abc'")
    if bc not in ("pec", "pmc"):
        raise ValueError("bc must be 'pec', 'pmc' or 'abc'")

    eps_r, mu_r, _, pec, _ = cell_materials(mesh, materials)
    live = ~pec
    live_f = live.astype(float)
    lam_h = float(k2_per_m2)

    U, B = whitney_fields(mesh, e)
    # Exact element L2 integral ||u_h||_T^2 with the tetrahedral P1 mass matrix
    u_l2_T = mesh.cell_volume * np.einsum("nik,ij,njk->n", U, _P1_TET_MASS, U)
    norm_sq = float(np.sum(eps_r * live_f * u_l2_T))
    if norm_sq <= 0.0:
        raise ValueError("eigenmode has zero electric energy norm")
    inv_norm = 1.0 / np.sqrt(norm_sq)
    U = U * inv_norm
    B = B * inv_norm
    u_l2_T = u_l2_T / norm_sq

    h_T = mesh.edge_length[mesh.cell_edges].max(axis=1)
    h_F = mesh.edge_length[mesh.face_edges].max(axis=1)

    eta2 = live_f * _res_element_term(h_T, mu_r, lam_h, eps_r, u_l2_T)

    # Sort cell-face incidences to pair interior faces and locate boundary faces
    nc = mesh.n_cells
    nf = mesh.n_faces
    f_all = mesh.cell_faces.ravel()
    c_all = np.repeat(np.arange(nc, dtype=np.int64), 4)
    k_all = np.tile(np.arange(4, dtype=np.int64), nc)
    order = np.argsort(f_all, kind="stable")
    f_sorted = f_all[order]
    c_sorted = c_all[order]
    k_sorted = k_all[order]
    counts = np.bincount(f_all, minlength=nf)

    # Interior faces (valence 2) where both cells are non-PEC
    m2 = counts[f_sorted] == 2
    f_int = f_sorted[m2][0::2]
    c0 = c_sorted[m2][0::2]
    k0 = k_sorted[m2][0::2]
    c1 = c_sorted[m2][1::2]
    k1 = k_sorted[m2][1::2]

    both_live = live[c0] & live[c1]
    if np.any(both_live):
        f_int = f_int[both_live]
        c0, k0 = c0[both_live], k0[both_live]
        c1, k1 = c1[both_live], k1[both_live]

        n_f = mesh.face_normal[f_int]
        A_f = mesh.face_area[f_int]

        # Tangential curl jump: [n x mu^-1 curl u_h] is constant on each face
        H0 = B[c0] / mu_r[c0, None]
        H1 = B[c1] / mu_r[c1, None]
        jmp_H_tan = np.cross(n_f, H0 - H1)
        curl_jump_sq = A_f * np.einsum("fi,fi->f", jmp_H_tan, jmp_H_tan)

        # Normal displacement jump: [eps u_h . n] is linear on F; match vertices by global node ID
        lv0 = _LOC_F[k0]  # (n_int, 3)
        gv0 = mesh.cells[c0[:, None], lv0]
        lv0_sorted = np.take_along_axis(lv0, np.argsort(gv0, axis=1), axis=1)

        lv1 = _LOC_F[k1]
        gv1 = mesh.cells[c1[:, None], lv1]
        lv1_sorted = np.take_along_axis(lv1, np.argsort(gv1, axis=1), axis=1)

        U0 = U[c0[:, None], lv0_sorted]  # (n_int, 3, 3)
        U1 = U[c1[:, None], lv1_sorted]
        dn0 = eps_r[c0, None] * np.einsum("fvi,fi->fv", U0, n_f)
        dn1 = eps_r[c1, None] * np.einsum("fvi,fi->fv", U1, n_f)
        jmp_dn = dn0 - dn1
        div_jump_sq = A_f * np.einsum("fi,ij,fj->f", jmp_dn, _P1_TRI_MASS, jmp_dn)

        mu_F = 0.5 * (mu_r[c0] + mu_r[c1])
        eps_F = 0.5 * (eps_r[c0] + eps_r[c1])
        w_curl, w_div = _res_face_jump_weight(h_F[f_int], mu_F, lam_h, eps_F)
        contrib = 0.5 * (w_curl * curl_jump_sq + w_div * div_jump_sq)
        np.add.at(eta2, c0, contrib)
        np.add.at(eta2, c1, contrib)

    # Outer PMC boundary faces (valence 1)
    if bc == "pmc":
        m1 = counts[f_sorted] == 1
        f_bnd = f_sorted[m1]
        c_bnd = c_sorted[m1]
        live_bnd = live[c_bnd]
        if np.any(live_bnd):
            f_bnd = f_bnd[live_bnd]
            c_bnd = c_bnd[live_bnd]
            n_f = mesh.face_normal[f_bnd]
            A_f = mesh.face_area[f_bnd]
            H_b = B[c_bnd] / mu_r[c_bnd, None]
            n_cross_H = np.cross(n_f, H_b)
            pmc_sq = A_f * np.einsum("fi,fi->f", n_cross_H, n_cross_H)
            w_pmc, _ = _res_face_jump_weight(
                h_F[f_bnd], mu_r[c_bnd], lam_h, eps_r[c_bnd]
            )
            np.add.at(eta2, c_bnd, w_pmc * pmc_sq)

    eta2[pec] = 0.0
    return eta2


def _zz_recover_curl(mesh, B, live):
    """
    Volume-weighted nodal curl recovery per material tag (never across tags).

    Returns vertex values `R_verts` of shape `(n_cells, 4, 3)`.
    """
    R_verts = np.zeros((mesh.n_cells, 4, 3), dtype=float)
    u_tags, tag_inv = np.unique(mesh.cell_tags, return_inverse=True)
    for ti in range(len(u_tags)):
        sel = live & (tag_inv == ti)
        if not np.any(sel):
            continue
        c_sel = np.flatnonzero(sel)
        v_sel = mesh.cell_volume[c_sel]
        vb_sel = v_sel[:, None] * B[c_sel]
        num = np.zeros((mesh.n_nodes, 3), dtype=float)
        den = np.zeros(mesh.n_nodes, dtype=float)
        for k in range(4):
            nk = mesh.cells[c_sel, k]
            np.add.at(num, nk, vb_sel)
            np.add.at(den, nk, v_sel)
        R_node = num / np.maximum(den[:, None], 1e-300)
        R_verts[c_sel] = R_node[mesh.cells[c_sel]]
    return R_verts


def zz_curl_indicator(mesh, e, materials=None):
    """
    Zienkiewicz-Zhu recovery error indicator on ``curl u_h``.

    Recovers a continuous piecewise-linear curl ``R`` inside each material tag
    by volume-weighted nodal averaging of ``B_T = curl u_h`` (never averaging
    across material tags) and integrates ``int_T mu_T^-1 |R - B_T|^2 dV``
    exactly with the P1 mass matrix for ``u_h`` normalised to
    ``(eps_r u_h, u_h) = 1``.

    Parameters
    ----------
    mesh : SimplexMesh
        3D tetrahedral mesh.
    e : array_like, shape (n_edges,)
        Eigenmode edge voltages [V].
    materials : dict, optional
        ``{tag: material}`` mapping as in `UnstructuredFIT3D`.

    Returns
    -------
    eta2_zz : ndarray, shape (n_cells,)
        Per-tetrahedron squared ZZ curl indicator [1/m^2].
    """
    eps_r, mu_r, _, pec, _ = cell_materials(mesh, materials)
    live = ~pec
    live_f = live.astype(float)

    U, B = whitney_fields(mesh, e)
    u_l2_T = mesh.cell_volume * np.einsum("nik,ij,njk->n", U, _P1_TET_MASS, U)
    norm_sq = float(np.sum(eps_r * live_f * u_l2_T))
    if norm_sq <= 0.0:
        raise ValueError("eigenmode has zero electric energy norm")
    B = B / np.sqrt(norm_sq)

    R_verts = _zz_recover_curl(mesh, B, live)
    diff = R_verts - B[:, None, :]
    eta2 = (
        (live_f / mu_r)
        * mesh.cell_volume
        * np.einsum("nik,ij,njk->n", diff, _P1_TET_MASS, diff)
    )
    eta2[pec] = 0.0
    return eta2


def _dorfler_order(eta2):
    """Sort cell indices by descending error indicator (stable)."""
    return np.argsort(-eta2, kind="stable")


def dorfler_mark(eta2, theta=0.5):
    """
    Dörfler bulk marking: smallest set ``M`` with
    ``sum_{T in M} eta_T^2 >= theta * sum_T eta_T^2``.

    Parameters
    ----------
    eta2 : array_like, shape (n_cells,)
        Per-cell squared error indicators ``eta_T^2`` [1/m^2].
    theta : float, optional
        Bulk marking fraction in ``(0, 1]`` (default 0.5).

    Returns
    -------
    ndarray of int
        Indices of marked cells (ordered by descending ``eta_T^2``).
    """
    theta = float(theta)
    if not (0.0 < theta <= 1.0):
        raise ValueError("theta must be in (0, 1]")
    eta2_arr = np.asarray(eta2, dtype=float)
    tot = float(np.sum(eta2_arr))
    if eta2_arr.size == 0 or tot <= 0.0:
        return np.empty(0, dtype=np.int64)
    order = _dorfler_order(eta2_arr)
    csum = np.cumsum(eta2_arr[order])
    target = theta * tot
    k = int(np.searchsorted(csum, target, side="left")) + 1
    k = min(k, eta2_arr.size)
    return order[:k]


@dataclass
class AdaptResult:
    """
    Result of an adaptive cavity eigenmode refinement run (`adapt_eigen`).

    Attributes
    ----------
    mesh : SimplexMesh
        Final solved tetrahedral mesh.
    e : ndarray, shape (n_edges,)
        Final eigenmode edge voltages [V].
    f_hz : float
        Final eigenfrequency [Hz].
    history : list of dict
        Per-iteration diagnostics, error indicators, mesh metrics, and wall times.
    stop_reason : {"max_iter", "max_dofs", "memory"}
        Criterion that terminated the loop.
    """

    mesh: SimplexMesh
    e: np.ndarray
    f_hz: float
    history: list[dict]
    stop_reason: str


def _count_free_edges(mesh, materials=None, bc="pec"):
    """Count free (non-fixed) edge degrees of freedom for Whitney FIT."""
    _, _, _, pec, _ = cell_materials(mesh, materials)
    fixed = np.zeros(mesh.n_edges, dtype=bool)
    fixed[mesh.cell_edges[pec].ravel()] = True
    if bc == "pec":
        fixed[mesh.boundary_edges] = True
    touched = np.zeros(mesh.n_edges, dtype=bool)
    touched[mesh.cell_edges[~pec].ravel()] = True
    fixed |= ~touched
    return int(np.sum(~fixed))


def adapt_eigen(
    mesh,
    materials=None,
    bc="pec",
    f_target_hz=None,
    mode=0,
    estimator="residual",
    marking="dorfler",
    theta=0.5,
    max_iter=20,
    max_dofs=None,
    max_memory_bytes=None,
    face_patch=None,
    project=None,
    mark_counts=None,
    seed=0,
    k2_ref_per_m2=None,
    callback=None,
    verbose=False,
):
    """
    Adaptive tetrahedral mesh refinement loop for a cavity eigenmode.

    Solves with ``hodge="whitney"`` at each iteration, computes both residual
    and ZZ error indicators, marks tetrahedra, and refines via longest-edge
    bisection (`bisect`).

    Parameters
    ----------
    mesh : SimplexMesh
        Initial 3D tetrahedral mesh.
    materials : dict, optional
        ``{tag: material}`` mapping as in `UnstructuredFIT3D`.
    bc : {"pec", "pmc"}, optional
        Outer boundary condition (default ``"pec"``).
    f_target_hz : float, optional
        Shift-invert target frequency [Hz].
    mode : int, optional
        Mode index (0-based, ascending frequency) selected at iteration 0;
        subsequent iterations track the mode closest in frequency to the
        previous ``f_hz``.
    estimator : {"residual", "zz"}, optional
        Error indicator driving Dörfler marking (default ``"residual"``).
    marking : {"dorfler", "uniform", "random"} or callable, optional
        Marking strategy: ``"dorfler"`` (`dorfler_mark`), ``"uniform"`` (one
        bisection generation per iteration), ``"random"`` (`mark_counts[it]`
        uniformly random cells with `seed`), or ``marking(mesh, eta2, it)``.
    theta : float, optional
        Dörfler bulk fraction in ``(0, 1]`` (default 0.5).
    max_iter : int, optional
        Maximum number of solve iterations (default 20).
    max_dofs : int, optional
        Stop when the free-edge DOF count reaches or when the next refined mesh
        would exceed `max_dofs`.
    max_memory_bytes : float, optional
        Stop when `estimate_memory(mesh, eigen=True)` exceeds this budget [bytes].
    face_patch : ndarray of int, optional
        Constraint-face patch labels (default from `label_constraint_faces`).
    project : callable, optional
        Boundary/interface node projector passed to `bisect`.
    mark_counts : sequence of int, optional
        Per-iteration cell counts required when ``marking="random"``.
    seed : int, optional
        Random seed for ``marking="random"`` (default 0).
    k2_ref_per_m2 : float, optional
        Reference eigenvalue ``k^2`` [1/m^2] for ``rel_err_signed``.
    callback : callable, optional
        Optional ``callback(it, mesh, e, record)`` called after each iteration.
    verbose : bool, optional
        If True, print per-iteration progress to stdout.

    Returns
    -------
    AdaptResult
    """
    if estimator not in ("residual", "zz"):
        raise ValueError("estimator must be 'residual' or 'zz'")
    if not (marking in ("dorfler", "uniform", "random") or callable(marking)):
        raise ValueError(
            "marking must be 'dorfler', 'uniform', 'random', or a callable"
        )
    if marking == "random" and mark_counts is None:
        raise ValueError("mark_counts is required when marking='random'")
    max_iter = int(max_iter)
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    mode = int(mode)
    if mode < 0:
        raise ValueError("mode must be non-negative")

    if f_target_hz is None:
        if k2_ref_per_m2 is not None:
            f_target_hz = 0.8 * (C0 / (2.0 * np.pi)) * np.sqrt(float(k2_ref_per_m2))
        else:
            span = np.ptp(mesh.nodes, axis=0)
            k_est = np.pi / float(np.max(span))
            f_target_hz = 0.5 * (C0 / (2.0 * np.pi)) * k_est
    else:
        f_target_hz = float(f_target_hz)

    cur_mesh = mesh
    cur_fp = (
        label_constraint_faces(cur_mesh)
        if face_patch is None
        else np.asarray(face_patch, dtype=np.int64).copy()
    )
    rng = np.random.default_rng(seed)
    history = []
    stop_reason = "max_iter"
    prev_f_hz = None
    last_mesh = cur_mesh
    last_e = np.zeros(cur_mesh.n_edges, dtype=float)
    last_f_hz = float("nan")
    n_modes_solve = max(mode + 2, 2)

    for it in range(max_iter):
        if cur_mesh.n_cells > 20000 or max_memory_bytes is not None:
            mem_est = estimate_memory(cur_mesh, eigen=True)
            if max_memory_bytes is not None and mem_est > float(max_memory_bytes):
                if verbose:
                    print(
                        f"[adapt_eigen] stopping before it={it}: estimated memory "
                        f"{mem_est / 1e9:.2f} GB exceeds budget "
                        f"{float(max_memory_bytes) / 1e9:.2f} GB"
                    )
                stop_reason = "memory"
                break

        t_iter0 = time.perf_counter()

        t0 = time.perf_counter()
        min_rr = float(cur_mesh.quality().min())
        h_min_m = float(cur_mesh.edge_length.min())
        h_max_m = float(cur_mesh.edge_length.max())
        _ = cur_mesh.C
        _ = cur_mesh.G
        t_mesh_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        sol = UnstructuredFIT3D(
            cur_mesh, materials=materials, bc=bc, hodge="whitney", stability=False
        )
        t_solver_build_s = time.perf_counter() - t0
        n_dofs = int(sol.free.size)

        t0 = time.perf_counter()
        freqs, E = sol.eigenmodes(n_modes=n_modes_solve, f_target=f_target_hz)
        t_eigen_s = time.perf_counter() - t0

        if it == 0:
            m_idx = mode
        else:
            m_idx = int(np.argmin(np.abs(freqs - prev_f_hz)))
        f_hz = float(freqs[m_idx])
        e_vec = E[:, m_idx].copy()
        prev_f_hz = f_hz
        other_idx = [j for j in range(len(freqs)) if j != m_idx]
        f_second_hz = (
            float(freqs[other_idx[int(np.argmin(np.abs(freqs[other_idx] - f_hz)))]])
            if other_idx
            else float("nan")
        )

        t0 = time.perf_counter()
        k2 = float((2.0 * np.pi * f_hz / C0) ** 2)
        eta2_res_cells = residual_indicator(
            cur_mesh, e_vec, k2, materials=materials, bc=bc
        )
        eta2_zz_cells = zz_curl_indicator(cur_mesh, e_vec, materials=materials)
        eta2_per_m2 = float(np.sum(eta2_res_cells))
        eta2_zz_per_m2 = float(np.sum(eta2_zz_cells))
        t_estimate_s = time.perf_counter() - t0

        last_mesh = cur_mesh
        last_e = e_vec
        last_f_hz = f_hz

        # Check if we should stop without further refinement
        at_max_dofs = max_dofs is not None and n_dofs >= int(max_dofs)
        at_max_iter = (it + 1) >= max_iter
        at_end_counts = (
            marking == "random"
            and mark_counts is not None
            and (it >= len(mark_counts) or int(mark_counts[it]) <= 0)
        )
        if at_max_dofs or at_max_iter or at_end_counts:
            rec = {
                "n_cells": int(cur_mesh.n_cells),
                "n_dofs": int(n_dofs),
                "f_hz": float(f_hz),
                "f_second_hz": float(f_second_hz),
                "k2_per_m2": float(k2),
                "eta2_per_m2": float(eta2_per_m2),
                "eta2_zz_per_m2": float(eta2_zz_per_m2),
                "n_marked": 0,
                "n_bisections": 0,
                "closure_rounds": 0,
                "min_radius_ratio": float(min_rr),
                "h_min_m": float(h_min_m),
                "h_max_m": float(h_max_m),
                "t_mesh_s": float(t_mesh_s),
                "t_solver_build_s": float(t_solver_build_s),
                "t_eigen_s": float(t_eigen_s),
                "t_estimate_s": float(t_estimate_s),
                "t_mark_s": 0.0,
                "t_refine_s": 0.0,
                "t_total_s": float(time.perf_counter() - t_iter0),
            }
            if k2_ref_per_m2 is not None:
                rec["rel_err_signed"] = float(
                    (k2 - float(k2_ref_per_m2)) / float(k2_ref_per_m2)
                )
            history.append(rec)
            if callback is not None:
                callback(it, cur_mesh, e_vec, rec)
            if verbose:
                print(
                    f"[adapt_eigen] it={it} cells={cur_mesh.n_cells} dofs={n_dofs} "
                    f"f={f_hz * 1e-9:.6f} GHz k2={k2:.6f} (stop)"
                )
            stop_reason = "max_dofs" if at_max_dofs else "max_iter"
            break

        # Mark cells for refinement
        t0 = time.perf_counter()
        eta2_drive = eta2_res_cells if estimator == "residual" else eta2_zz_cells
        if marking == "dorfler":
            marked = dorfler_mark(eta2_drive, theta=theta)
        elif marking == "uniform":
            marked = np.arange(cur_mesh.n_cells, dtype=np.int64)
        elif marking == "random":
            k_mark = min(int(mark_counts[it]), cur_mesh.n_cells)
            marked = rng.choice(cur_mesh.n_cells, size=k_mark, replace=False)
        else:
            marked = np.asarray(marking(cur_mesh, eta2_drive, it), dtype=np.int64)
        t_mark_s = time.perf_counter() - t0
        n_marked = len(marked)

        # Refine via longest-edge bisection
        t0 = time.perf_counter()
        next_mesh, info = bisect(cur_mesh, marked, face_patch=cur_fp, project=project)
        t_refine_s = time.perf_counter() - t0
        n_bisections = int(info["n_bisections"])
        closure_rounds = int(info["rounds"])

        rec = {
            "n_cells": int(cur_mesh.n_cells),
            "n_dofs": int(n_dofs),
            "f_hz": float(f_hz),
            "f_second_hz": float(f_second_hz),
            "k2_per_m2": float(k2),
            "eta2_per_m2": float(eta2_per_m2),
            "eta2_zz_per_m2": float(eta2_zz_per_m2),
            "n_marked": int(n_marked),
            "n_bisections": int(n_bisections),
            "closure_rounds": int(closure_rounds),
            "min_radius_ratio": float(min_rr),
            "h_min_m": float(h_min_m),
            "h_max_m": float(h_max_m),
            "t_mesh_s": float(t_mesh_s),
            "t_solver_build_s": float(t_solver_build_s),
            "t_eigen_s": float(t_eigen_s),
            "t_estimate_s": float(t_estimate_s),
            "t_mark_s": float(t_mark_s),
            "t_refine_s": float(t_refine_s),
            "t_total_s": float(time.perf_counter() - t_iter0),
        }
        if k2_ref_per_m2 is not None:
            rec["rel_err_signed"] = float(
                (k2 - float(k2_ref_per_m2)) / float(k2_ref_per_m2)
            )
        history.append(rec)
        if callback is not None:
            callback(it, cur_mesh, e_vec, rec)
        if verbose:
            print(
                f"[adapt_eigen] it={it} cells={cur_mesh.n_cells} dofs={n_dofs} "
                f"f={f_hz * 1e-9:.6f} GHz k2={k2:.6f} marked={n_marked} "
                f"bisect={n_bisections} t={rec['t_total_s']:.2f}s"
            )

        if max_dofs is not None:
            next_dofs = _count_free_edges(next_mesh, materials=materials, bc=bc)
            if next_dofs > int(max_dofs):
                if verbose:
                    print(
                        f"[adapt_eigen] stopping after it={it}: next_dofs={next_dofs} "
                        f"> max_dofs={int(max_dofs)}"
                    )
                stop_reason = "max_dofs"
                break

        if next_mesh.n_cells > 20000 or max_memory_bytes is not None:
            mem_next = estimate_memory(next_mesh, eigen=True)
            if max_memory_bytes is not None and mem_next > float(max_memory_bytes):
                if verbose:
                    print(
                        f"[adapt_eigen] stopping after it={it}: next estimated memory "
                        f"{mem_next / 1e9:.2f} GB exceeds budget "
                        f"{float(max_memory_bytes) / 1e9:.2f} GB"
                    )
                stop_reason = "memory"
                break

        cur_mesh = next_mesh
        cur_fp = info["face_patch"]

    return AdaptResult(
        mesh=last_mesh,
        e=last_e,
        f_hz=last_f_hz,
        history=history,
        stop_reason=stop_reason,
    )


def voronoi_export(mesh, materials=None, bc="pec", **improve_kw):
    """
    Export an adapted tetrahedral mesh for explicit Voronoi-Hodge time stepping.

    Counts negative-dual-length faces before and after calling
    `improve_dt(mesh, materials, bc=bc, repair=True, **improve_kw)`, then builds
    `UnstructuredFIT3D(..., hodge="voronoi")` and reports ``dt_max_s`` (or the
    solver guard's `ValueError` message when residual negative faces remain).

    Parameters
    ----------
    mesh : SimplexMesh
        Input 3D tetrahedral mesh.
    materials : dict, optional
        ``{tag: material}`` mapping as in `UnstructuredFIT3D`.
    bc : {"pec", "pmc", "abc"}, optional
        Outer boundary condition (default ``"pec"``).
    **improve_kw
        Additional keyword arguments forwarded to `improve_dt` (plus optional
        ``steiner`` and ``max_flips`` forwarded to `flip_to_delaunay`).

    Returns
    -------
    new_mesh : SimplexMesh
        Repaired and node-relocated mesh.
    report : dict
        ``negative_faces_before``, ``negative_faces_after``, ``dt_max_s``,
        ``dt_rate_s``, ``error``, and ``history``.
    """
    from . import repair as _repair_mod

    neg_before = len(neg_faces(mesh, materials=materials, bc=bc))
    repair_flag = bool(improve_kw.pop("repair", True))
    steiner_flag = bool(improve_kw.pop("steiner", False))
    max_flips = improve_kw.pop("max_flips", None)

    orig_ftd = _repair_mod.flip_to_delaunay
    if not steiner_flag or max_flips is not None:

        def _ftd_wrapper(*args, **kwargs):
            kwargs.setdefault("steiner", steiner_flag)
            if max_flips is not None:
                kwargs.setdefault("max_flips", max_flips)
            return orig_ftd(*args, **kwargs)

        _repair_mod.flip_to_delaunay = _ftd_wrapper
    try:
        new_mesh, hist = improve_dt(
            mesh, materials=materials, bc=bc, repair=repair_flag, **improve_kw
        )
    finally:
        _repair_mod.flip_to_delaunay = orig_ftd

    neg_after = len(neg_faces(new_mesh, materials=materials, bc=bc))
    max_rate = float(hist[-1]["max_rate"]) if hist else float("nan")
    dt_rate_s = (
        float(2.0 / np.sqrt(max_rate))
        if np.isfinite(max_rate) and max_rate > 0.0
        else float("nan")
    )

    dt_max_s = float("nan")
    err_msg = None
    try:
        sol = UnstructuredFIT3D(new_mesh, materials=materials, bc=bc, hodge="voronoi")
        dt_max_s = float(sol.dt_max)
    except (ValueError, RuntimeError) as exc:
        err_msg = str(exc)

    report = {
        "negative_faces_before": neg_before,
        "negative_faces_after": neg_after,
        "dt_max_s": dt_max_s,
        "dt_rate_s": dt_rate_s,
        "error": err_msg,
        "history": hist,
    }
    return new_mesh, report
