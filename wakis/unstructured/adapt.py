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

import numpy as np

from .improve import constraint_faces
from .mesh import SimplexMesh

__all__ = [
    "bisect",
    "kuhn_mesh",
    "l_prism_mesh",
    "label_constraint_faces",
    "refine_uniform",
    "sphere_projector",
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
