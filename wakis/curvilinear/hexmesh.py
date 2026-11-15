# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Conforming unstructured curvilinear hexahedral mesh for 3D nonorthogonal FIT.

A `HexMesh` is defined by a node coordinate array ``nodes`` of shape
``(n_nodes, 3)`` [m] and a hexahedral connectivity array ``hexes`` of shape
``(n_cells, 8)`` in VTK hexahedron vertex ordering:

* bottom face (``zeta = 0``, counter-clockwise in ``(xi, eta)``):
  ``0: (0, 0, 0)``, ``1: (1, 0, 0)``, ``2: (1, 1, 0)``, ``3: (0, 1, 0)``;
* top face (``zeta = 1``, same order in ``(xi, eta)``):
  ``4: (0, 0, 1)``, ``5: (1, 0, 1)``, ``6: (1, 1, 1)``, ``7: (0, 1, 1)``.

Global orientation conventions:

* **Primal edges**: each unique edge is identified by its sorted node pair
  ``(u, v)`` with ``u < v``, sorted lexicographically. The global edge
  orientation points from the lower global node index ``u`` to the higher
  global node index ``v``. Each local hex edge ``(a, db, dc)`` (oriented along
  ``+xi_a``) maps to a global edge with sign ``+1`` if its start node index is
  less than its end node index and ``-1`` otherwise.
* **Primal faces**: each unique quadrilateral face is identified by its sorted
  4-tuple of global node indices ``(n0, n1, n2, n3)`` with
  ``n0 < n1 < n2 < n3``, sorted lexicographically. Its canonical global cycle
  ``(q0, q1, q2, q3)`` is anchored at the lowest global node index ``q0 = n0``
  and follows the perimeter cycle of the first incident local face (ordered by
  ``(a, side, hex)``). Both the topological curl matrix ``C`` and the primal
  face shoelace vector area
  ``A_f = 1/2 (nodes[q2] - nodes[q0]) x (nodes[q3] - nodes[q1])`` use this
  exact same cycle ``(q0, q1, q2, q3)``. Each local hex face ``(a, side)``
  (oriented along ``+xi_a``) maps to its global face with sign ``+1`` if its
  ``q0``-anchored ``+xi_a`` cycle matches ``(q0, q1, q2, q3)`` and ``-1`` if
  reversed.
"""

import numpy as np
import scipy.sparse as sp

from .mesh3d import HexBlock3D

__all__ = ["HexMesh", "mesh_quality"]

# Mapping from logical corner (a, b, c) in {0, 1}^3 to VTK vertex index 0..7
_CORNER_VTK = np.array(
    [
        [[0, 4], [3, 7]],
        [[1, 5], [2, 6]],
    ],
    dtype=np.int64,
)

# Local +xi_a face corner VTK indices (p0, p1, p2, p3) for a in (0, 1, 2), side in (0, 1)
# where b = (a + 1) % 3, c = (a + 2) % 3, and (db, dc) = ((0,0), (1,0), (1,1), (0,1))
_LOCAL_FACE_VTK = np.empty((3, 2, 4), dtype=np.int64)
for _a in range(3):
    _b, _c = (_a + 1) % 3, (_a + 2) % 3
    for _side in (0, 1):
        for _m, (_db, _dc) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
            _idx = [0, 0, 0]
            _idx[_a] = _side
            _idx[_b] = _db
            _idx[_c] = _dc
            _LOCAL_FACE_VTK[_a, _side, _m] = _CORNER_VTK[tuple(_idx)]

# Local +xi_a edge VTK endpoints (start, end) for a in (0, 1, 2), db in (0, 1), dc in (0, 1)
_LOCAL_EDGE_VTK = np.empty((3, 2, 2, 2), dtype=np.int64)
for _a in range(3):
    _b, _c = (_a + 1) % 3, (_a + 2) % 3
    for _db in (0, 1):
        for _dc in (0, 1):
            for _end in (0, 1):
                _idx = [0, 0, 0]
                _idx[_a] = _end
                _idx[_b] = _db
                _idx[_c] = _dc
                _LOCAL_EDGE_VTK[_a, _db, _dc, _end] = _CORNER_VTK[tuple(_idx)]


def _rotate_quads_to_min(quads):
    """Rotate each 4-tuple row of ``quads`` (N, 4) cyclically so its minimum
    node index is at column 0."""
    pos = np.argmin(quads, axis=1)
    idx = (pos[:, None] + np.arange(4)[None, :]) % 4
    return np.take_along_axis(quads, idx, axis=1)


class HexMesh:
    """
    Conforming unstructured curvilinear hexahedral mesh.

    Parameters
    ----------
    nodes : array_like of shape (n_nodes, 3)
        Node coordinates [m].
    hexes : array_like of shape (n_cells, 8)
        Hexahedral vertex connectivity in VTK ordering.
    hex_tags : array_like of shape (n_cells,), optional
        Integer material tag per hexahedron (default 0).
    boundary_patch : array_like of shape (n_boundary_faces,), optional
        Integer label of the smooth wall patch each boundary face belongs to.
        When ``None``, assigns one patch per distinct unit outward face normal
        direction within cross-norm ``1e-12`` (on curved walls without explicit
        patch labels, this default classifies every multi-face boundary node as
        sharp).
    sharp_nodes : array_like of shape (n_nodes,) of bool, optional
        Generator-supplied boolean mask of boundary nodes that are sharp PEC
        features.
    _flip_face_sign : int, optional
        Private test-only mutation hook: flip the face-to-edge orientation of
        global face ``_flip_face_sign``.
    _ignore_sharp_nodes : bool, optional
        Private test-only mutation hook: ignore ``sharp_nodes`` in the boundary
        octant classification rule.

    Raises
    ------
    ValueError
        If coordinates are non-finite, any hexahedron is inverted/degenerate,
        duplicate hexahedra exist, or the mesh is non-conforming (reporting the
        offending hex index).
    """

    def __init__(
        self,
        nodes,
        hexes=None,
        hex_tags=None,
        boundary_patch=None,
        sharp_nodes=None,
        _flip_face_sign=None,
        _ignore_sharp_nodes=False,
    ):
        if isinstance(nodes, HexMesh) and hexes is None:
            if (
                hex_tags is None
                and boundary_patch is None
                and sharp_nodes is None
                and _flip_face_sign is None
                and not _ignore_sharp_nodes
            ):
                self.__dict__.update(nodes.__dict__)
                return
            src = nodes
            nodes = src.nodes
            hexes = src.hexes
            if hex_tags is None:
                hex_tags = src.hex_tags
            if boundary_patch is None:
                boundary_patch = src.boundary_patch
            if sharp_nodes is None:
                sharp_nodes = src.sharp_nodes
        if isinstance(nodes, tuple) and hexes is None:
            if len(nodes) == 4:
                nodes, hexes, boundary_patch, sharp_nodes = nodes
            elif len(nodes) == 3:
                nodes, hexes, boundary_patch = nodes
            elif len(nodes) == 2:
                nodes, hexes = nodes
        if isinstance(nodes, HexBlock3D) and hexes is None:
            built = self.from_block(
                nodes,
                _flip_face_sign=_flip_face_sign,
                _ignore_sharp_nodes=_ignore_sharp_nodes,
            )
            self.__dict__.update(built.__dict__)
            return

        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] < 8:
            raise ValueError(
                "nodes must have shape (n_nodes, 3) with n_nodes >= 8, "
                f"got {nodes.shape}"
            )
        if not np.all(np.isfinite(nodes)):
            raise ValueError("node coordinates must be finite")

        if hexes is None:
            raise ValueError("hexes connectivity array is required")
        hexes = np.asarray(hexes, dtype=np.int64)
        if hexes.ndim != 2 or hexes.shape[1] != 8 or hexes.shape[0] < 1:
            raise ValueError(
                "hexes must have shape (n_cells, 8) with n_cells >= 1, "
                f"got {hexes.shape}"
            )

        n_nodes = int(nodes.shape[0])
        n_cells = int(hexes.shape[0])

        bad_range = np.any((hexes < 0) | (hexes >= n_nodes), axis=1)
        if np.any(bad_range):
            h_bad = int(np.flatnonzero(bad_range)[0])
            raise ValueError(f"hex {h_bad} has out-of-range node index")

        hexes_sorted = np.sort(hexes, axis=1)
        bad_repeat = np.any(hexes_sorted[:, 1:] == hexes_sorted[:, :-1], axis=1)
        if np.any(bad_repeat):
            h_bad = int(np.flatnonzero(bad_repeat)[0])
            raise ValueError(f"degenerate hex at index {h_bad} with repeated vertex")

        _, uniq_idx, uniq_inv, uniq_cnt = np.unique(
            hexes_sorted,
            axis=0,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        if np.any(uniq_cnt > 1):
            dup_groups = np.flatnonzero(uniq_cnt > 1)
            first_idx = set(uniq_idx[dup_groups].tolist())
            dup_hexes = [
                h
                for h in range(n_cells)
                if uniq_cnt[uniq_inv[h]] > 1 and h not in first_idx
            ]
            raise ValueError(f"duplicate hex at index {int(dup_hexes[0])}")

        if (
            sharp_nodes is None
            and hex_tags is not None
            and boundary_patch is not None
            and np.asarray(boundary_patch).dtype == bool
            and np.asarray(boundary_patch).shape == (n_nodes,)
        ):
            sharp_nodes = boundary_patch
            boundary_patch = hex_tags
            hex_tags = None

        if hex_tags is None:
            hex_tags = np.zeros(n_cells, dtype=np.int64)
        else:
            hex_tags = np.asarray(hex_tags, dtype=np.int64)
            if hex_tags.shape != (n_cells,):
                raise ValueError(
                    f"hex_tags must have shape {(n_cells,)}, got {hex_tags.shape}"
                )

        self.nodes = nodes
        self.hexes = hexes
        self.hex_tags = hex_tags
        self.n_nodes = n_nodes
        self.n_cells = n_cells
        self.nc = (n_cells,)
        self._flip_face_sign = None if _flip_face_sign is None else int(_flip_face_sign)
        self._ignore_sharp_nodes = bool(_ignore_sharp_nodes)
        self._from_block_ref = None
        self.block_edge_perm = None
        self.block_face_perm = None

        self.corner_jacobians, self.scaled_corner_jacobians = (
            self._compute_corner_jacobians()
        )
        bad_jac = np.any(
            (~np.isfinite(self.scaled_corner_jacobians))
            | (self.scaled_corner_jacobians <= 0.0)
            | (self.corner_jacobians <= 0.0),
            axis=(1, 2, 3),
        )
        if np.any(bad_jac):
            h_bad = int(np.flatnonzero(bad_jac)[0])
            raise ValueError(f"inverted or degenerate hexahedron at hex index {h_bad}")

        self.cell_volumes = self._compute_cell_volumes()
        bad_vol = (~np.isfinite(self.cell_volumes)) | (self.cell_volumes <= 0.0)
        if np.any(bad_vol):
            h_bad = int(np.flatnonzero(bad_vol)[0])
            raise ValueError(
                f"non-positive hexahedral cell volume at hex index {h_bad}"
            )
        self.corner_volumes = self.cell_volumes / 8.0
        self.cell_centroids = self._compute_cell_centroids()

        self._build_faces_and_edges()
        self.face_areas, self.face_centroids = self._compute_face_geometry()
        self.A_f = self.face_areas

        self._build_boundary_patches_and_normals(boundary_patch, sharp_nodes)
        self.dual_face_pieces, self.dual_face_areas = self._compute_dual_faces()
        self.S_e = self.dual_face_areas

        self._solver_cache = None

    def __iter__(self):
        """Allow unpacking ``nodes, hexes, boundary_patch, sharp_nodes = mesh``."""
        return iter((self.nodes, self.hexes, self.boundary_patch, self.sharp_nodes))

    def total_volume(self):
        """
        Total volume ``sum_h V_hex`` [m^3] of the mesh.

        Returns
        -------
        float
        """
        return float(np.sum(self.cell_volumes))

    @classmethod
    def from_block(cls, block, _flip_face_sign=None, _ignore_sharp_nodes=False):
        """
        Construct an equivalent `HexMesh` from a structured `HexBlock3D`.

        Parameters
        ----------
        block : HexBlock3D
            Structured curvilinear hexahedral block.

        Returns
        -------
        HexMesh
        """
        if not isinstance(block, HexBlock3D):
            raise TypeError("block must be a HexBlock3D instance")

        Nx, Ny, Nz = block.nc
        nn = block.nn
        nodes = block.X.reshape(-1, 3)

        ijk = block.cell_grid()
        hexes = np.empty((block.n_cells, 8), dtype=np.int64)
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    v_idx = _CORNER_VTK[a, b, c]
                    n_idx = np.ravel_multi_index(
                        (ijk[0] + a, ijk[1] + b, ijk[2] + c), nn
                    ).ravel()
                    hexes[:, v_idx] = n_idx

        # Patch per local hex face (h, a, side): 2 * a + side in 0..5
        local_patch = np.full((block.n_cells, 3, 2), -1, dtype=np.int64)
        for a in range(3):
            for side in (0, 1):
                idx_a = 0 if side == 0 else block.nc[a] - 1
                mask = (ijk[a] == idx_a).ravel()
                local_patch[mask, a, side] = 2 * a + side

        # Sharp nodes from block wall normals (matches Job A classification)
        sharp_mask = np.zeros(nn, dtype=bool)
        w_cnt = block.node_wall_count
        multi_ijk = np.nonzero(w_cnt >= 2)
        if multi_ijk[0].size > 0:
            n_m = multi_ijk[0].size
            parallel = np.ones(n_m, dtype=bool)
            n_ref = np.zeros((n_m, 3), dtype=float)
            has_ref = np.zeros(n_m, dtype=bool)
            for ax, n_max in ((0, Nx), (1, Ny), (2, Nz)):
                for side, s_val in ((0, 0), (1, n_max)):
                    hit = multi_ijk[ax] == s_val
                    if not np.any(hit):
                        continue
                    n_k = block.wall_node_normals[
                        ax,
                        side,
                        multi_ijk[0][hit],
                        multi_ijk[1][hit],
                        multi_ijk[2][hit],
                    ]
                    first = ~has_ref[hit]
                    if np.any(first):
                        hit_first = np.flatnonzero(hit)[first]
                        n_ref[hit_first] = n_k[first]
                        has_ref[hit_first] = True
                    rest = has_ref[hit] & (~first)
                    if np.any(rest):
                        hit_rest = np.flatnonzero(hit)[rest]
                        cn = np.linalg.norm(
                            np.cross(n_ref[hit_rest], n_k[rest]), axis=1
                        )
                        parallel[hit_rest] &= cn <= 1e-12
            sharp_mask[
                multi_ijk[0][~parallel],
                multi_ijk[1][~parallel],
                multi_ijk[2][~parallel],
            ] = True

        mesh = cls(
            nodes,
            hexes,
            boundary_patch=local_patch,
            sharp_nodes=sharp_mask.ravel(),
            _flip_face_sign=_flip_face_sign,
            _ignore_sharp_nodes=_ignore_sharp_nodes,
        )
        mesh._from_block_ref = block

        # Explicit permutation mapping HexMesh edge/face indices to HexBlock3D indices:
        # block_edge[e_blk] == mesh_edge[block_edge_perm[e_blk]]
        blk_edges = []
        for a in range(3):
            eg = block.edge_grid(a)
            n0 = np.ravel_multi_index(tuple(eg), nn).ravel()
            eg1 = [x.copy() for x in eg]
            eg1[a] = eg1[a] + 1
            n1 = np.ravel_multi_index(tuple(eg1), nn).ravel()
            blk_edges.append(np.stack([n0, n1], axis=1))
        blk_edge_nodes = np.concatenate(blk_edges, axis=0)
        # Since mesh.edge_nodes is sorted lexicographically, searchsorted in 1D composite key
        key_mesh_e = mesh.edge_nodes[:, 0] * mesh.n_nodes + mesh.edge_nodes[:, 1]
        key_blk_e = blk_edge_nodes[:, 0] * mesh.n_nodes + blk_edge_nodes[:, 1]
        mesh.block_edge_perm = np.searchsorted(key_mesh_e, key_blk_e)

        blk_faces = []
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            fg = block.face_grid(a)
            corners = []
            for db, dc in ((0, 0), (1, 0), (1, 1), (0, 1)):
                q = [x.copy() for x in fg]
                q[b] = q[b] + db
                q[c] = q[c] + dc
                corners.append(np.ravel_multi_index(tuple(q), nn).ravel())
            blk_faces.append(np.sort(np.stack(corners, axis=1), axis=1))
        blk_face_nodes = np.concatenate(blk_faces, axis=0)
        # Lexicographical match between mesh.face_nodes_sorted and blk_face_nodes
        order_blk = np.lexsort(
            (
                blk_face_nodes[:, 3],
                blk_face_nodes[:, 2],
                blk_face_nodes[:, 1],
                blk_face_nodes[:, 0],
            )
        )
        inv_order_blk = np.empty_like(order_blk)
        inv_order_blk[order_blk] = np.arange(order_blk.size)
        mesh.block_face_perm = inv_order_blk

        return mesh

    # --------------------------------------------------------------- geometry
    def _compute_corner_jacobians(self):
        """Signed and normalised corner Jacobian determinants of the trilinear
        map at the 8 corners of every hex, shape ``(n_cells, 2, 2, 2)``."""
        X = self.nodes
        H = self.hexes
        J = np.empty((self.n_cells, 2, 2, 2), dtype=float)
        scaled_J = np.empty((self.n_cells, 2, 2, 2), dtype=float)
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    dxi = X[H[:, _CORNER_VTK[1, b, c]]] - X[H[:, _CORNER_VTK[0, b, c]]]
                    deta = X[H[:, _CORNER_VTK[a, 1, c]]] - X[H[:, _CORNER_VTK[a, 0, c]]]
                    dzeta = (
                        X[H[:, _CORNER_VTK[a, b, 1]]] - X[H[:, _CORNER_VTK[a, b, 0]]]
                    )
                    det = np.einsum("...i,...i->...", np.cross(dxi, deta), dzeta)
                    J[:, a, b, c] = det
                    denom = (
                        np.linalg.norm(dxi, axis=-1)
                        * np.linalg.norm(deta, axis=-1)
                        * np.linalg.norm(dzeta, axis=-1)
                    )
                    with np.errstate(divide="ignore", invalid="ignore"):
                        sj = np.where(denom > 0.0, det / denom, -np.inf)
                    scaled_J[:, a, b, c] = sj
        return J, scaled_J

    def _compute_cell_volumes(self):
        """Exact trilinear hexahedron volume [m^3] via 2x2x2 Gauss-Legendre
        quadrature on ``[0, 1]^3``, shape ``(n_cells,)``."""
        X = self.nodes
        H = self.hexes
        corners = np.empty((2, 2, 2, self.n_cells, 3), dtype=float)
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    corners[a, b, c] = X[H[:, _CORNER_VTK[a, b, c]]]

        dx = corners[1] - corners[0]
        dy = corners[:, 1] - corners[:, 0]
        dz = corners[:, :, 1] - corners[:, :, 0]

        s = 0.5 / np.sqrt(3.0)
        g_pts = (0.5 - s, 0.5 + s)
        vol = np.zeros(self.n_cells, dtype=float)
        for xi in g_pts:
            wx = (1.0 - xi, xi)
            for eta in g_pts:
                wy = (1.0 - eta, eta)
                for zeta in g_pts:
                    wz = (1.0 - zeta, zeta)
                    dxi = (
                        wy[0] * wz[0] * dx[0, 0]
                        + wy[0] * wz[1] * dx[0, 1]
                        + wy[1] * wz[0] * dx[1, 0]
                        + wy[1] * wz[1] * dx[1, 1]
                    )
                    deta = (
                        wx[0] * wz[0] * dy[0, 0]
                        + wx[0] * wz[1] * dy[0, 1]
                        + wx[1] * wz[0] * dy[1, 0]
                        + wx[1] * wz[1] * dy[1, 1]
                    )
                    dzeta = (
                        wx[0] * wy[0] * dz[0, 0]
                        + wx[0] * wy[1] * dz[0, 1]
                        + wx[1] * wy[0] * dz[1, 0]
                        + wx[1] * wy[1] * dz[1, 1]
                    )
                    vol += 0.125 * np.einsum(
                        "...i,...i->...", np.cross(dxi, deta), dzeta
                    )
        return vol

    def _compute_cell_centroids(self):
        """Hex centroids (average of the 8 corners) [m], shape ``(n_cells, 3)``."""
        X = self.nodes
        H = self.hexes
        corners = np.empty((self.n_cells, 2, 2, 2, 3), dtype=float)
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    corners[:, a, b, c] = X[H[:, _CORNER_VTK[a, b, c]]]
        c0 = 0.5 * (corners[:, 0] + corners[:, 1])
        c01 = 0.5 * (c0[:, 0] + c0[:, 1])
        return 0.5 * (c01[:, 0] + c01[:, 1])

    def _build_faces_and_edges(self):
        """Deduplicate primal faces and edges, verify conformity, and record
        local-to-global indices and orientation signs."""
        H = self.hexes
        M = self.n_cells

        # Order local faces as (a=0..2, side=0..1, h=0..M-1)
        loc_quads = np.empty((3, 2, M, 4), dtype=np.int64)
        for a in range(3):
            for side in (0, 1):
                loc_quads[a, side] = H[:, _LOCAL_FACE_VTK[a, side]]

        flat_quads = loc_quads.reshape(-1, 4)
        flat_rot = _rotate_quads_to_min(flat_quads)
        flat_sorted = np.sort(flat_quads, axis=1)

        bad_quad = np.any(flat_sorted[:, 1:] == flat_sorted[:, :-1], axis=1)
        if np.any(bad_quad):
            flat_idx = int(np.flatnonzero(bad_quad)[0])
            h_bad = flat_idx % M
            raise ValueError(f"degenerate face in hex at index {h_bad}")

        face_nodes_sorted, first_idx, inv_idx, face_counts = np.unique(
            flat_sorted,
            axis=0,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        if np.any(face_counts > 2):
            bad_f = int(np.flatnonzero(face_counts > 2)[0])
            hits = np.flatnonzero(inv_idx == bad_f)
            h_bad = int(hits[-1] % M)
            raise ValueError(
                f"non-conforming face shared by {int(face_counts[bad_f])} hexes "
                f"at hex index {h_bad}"
            )

        n_faces = int(face_nodes_sorted.shape[0])
        # Canonical global cycle (q0, q1, q2, q3) anchored at q0 = min(face_nodes)
        face_cycles = flat_rot[first_idx].copy()

        # Check every local face against its canonical cycle
        canon_for_flat = face_cycles[inv_idx]
        same_cycle = (
            (flat_rot[:, 1] == canon_for_flat[:, 1])
            & (flat_rot[:, 2] == canon_for_flat[:, 2])
            & (flat_rot[:, 3] == canon_for_flat[:, 3])
        )
        rev_cycle = (
            (flat_rot[:, 1] == canon_for_flat[:, 3])
            & (flat_rot[:, 2] == canon_for_flat[:, 2])
            & (flat_rot[:, 3] == canon_for_flat[:, 1])
        )
        valid_cycle = same_cycle | rev_cycle
        if not np.all(valid_cycle):
            flat_idx = int(np.flatnonzero(~valid_cycle)[0])
            h_bad = flat_idx % M
            raise ValueError(f"non-conforming twisted quad face at hex index {h_bad}")

        flat_signs = np.where(same_cycle, 1.0, -1.0)
        hex_faces = np.moveaxis(inv_idx.reshape(3, 2, M), 2, 0).copy()
        hex_face_signs = np.moveaxis(flat_signs.reshape(3, 2, M), 2, 0).copy()

        # Verify interior faces have opposite outward normals on their two sharing hexes
        out_sgn_grid = np.empty((3, 2, M), dtype=float)
        for a in range(3):
            for side in (0, 1):
                s_out = -1.0 if side == 0 else 1.0
                out_sgn_grid[a, side] = s_out * hex_face_signs[:, a, side]
        out_sum = np.zeros(n_faces, dtype=float)
        np.add.at(out_sum, inv_idx, out_sgn_grid.ravel())
        int_bad = (face_counts == 2) & (out_sum != 0.0)
        if np.any(int_bad):
            bad_f = int(np.flatnonzero(int_bad)[0])
            hits = np.flatnonzero(inv_idx == bad_f)
            h_bad = int(hits[-1] % M)
            raise ValueError(
                f"non-conforming interior face orientation at hex index {h_bad}"
            )

        self.n_faces = n_faces
        self.face_nodes_sorted = face_nodes_sorted
        self.face_cycles = face_cycles
        self.face_hex_count = face_counts
        self.hex_faces = hex_faces
        self.hex_face_signs = hex_face_signs
        self.boundary_face_mask = face_counts == 1
        self.boundary_faces = np.flatnonzero(self.boundary_face_mask)
        self.n_boundary_faces = int(self.boundary_faces.size)

        # Record single owner (h, a, side) for each boundary face
        bnd_flat_pos = first_idx[self.boundary_faces]
        self.boundary_face_hex = bnd_flat_pos % M
        rem = bnd_flat_pos // M
        self.boundary_face_axis = rem // 2
        self.boundary_face_side = rem % 2

        # Extract unique edges from all 12 local hex edges
        loc_edges = np.empty((M, 3, 2, 2, 2), dtype=np.int64)
        for a in range(3):
            for db in (0, 1):
                for dc in (0, 1):
                    loc_edges[:, a, db, dc] = H[:, _LOCAL_EDGE_VTK[a, db, dc]]

        n_start = loc_edges[..., 0]
        n_end = loc_edges[..., 1]
        e_min = np.minimum(n_start, n_end)
        e_max = np.maximum(n_start, n_end)
        e_pairs = np.stack([e_min, e_max], axis=-1).reshape(-1, 2)

        edge_nodes, edge_inv = np.unique(e_pairs, axis=0, return_inverse=True)
        self.n_edges = int(edge_nodes.shape[0])
        self.edge_nodes = edge_nodes
        self.hex_edges = edge_inv.reshape(M, 3, 2, 2)
        self.hex_edge_signs = np.where(n_start < n_end, 1.0, -1.0)

        # Map each face's 4 cycle edges (q_m -> q_{(m+1)%4}) to global edges and signs
        key_e = edge_nodes[:, 0] * self.n_nodes + edge_nodes[:, 1]
        face_edges = np.empty((n_faces, 4), dtype=np.int64)
        face_edge_signs = np.empty((n_faces, 4), dtype=float)
        for m in range(4):
            u = face_cycles[:, m]
            v = face_cycles[:, (m + 1) % 4]
            emin = np.minimum(u, v)
            emax = np.maximum(u, v)
            face_edges[:, m] = np.searchsorted(key_e, emin * self.n_nodes + emax)
            face_edge_signs[:, m] = np.where(u < v, 1.0, -1.0)

        if self._flip_face_sign is not None:
            k_flip = self._flip_face_sign
            if k_flip < 0 or k_flip >= n_faces:
                raise ValueError(
                    f"_flip_face_sign={k_flip} out of range [0, {n_faces})"
                )
            # Flip one edge orientation on face k_flip (breaks closed-loop orientation)
            face_edge_signs[k_flip, 0] *= -1.0
            self.hex_face_signs[self.hex_faces == k_flip] *= -1.0

        self.face_edges = face_edges
        self.face_edge_signs = face_edge_signs
        self.edges = self.edge_nodes
        self.faces = self.face_cycles

        # Conformity check on boundary: every boundary edge must belong to >= 2
        # boundary faces (exactly 2 on a manifold boundary)
        bnd_f_edges = face_edges[self.boundary_faces]
        bnd_edge_counts = np.zeros(self.n_edges, dtype=np.int64)
        np.add.at(bnd_edge_counts, bnd_f_edges.ravel(), 1)
        bad_bnd_e = (bnd_edge_counts > 0) & (bnd_edge_counts != 2)
        if np.any(bad_bnd_e):
            e_bad = int(np.flatnonzero(bad_bnd_e)[0])
            bf_hit = int(np.flatnonzero(np.any(bnd_f_edges == e_bad, axis=1))[0])
            h_bad = int(self.boundary_face_hex[bf_hit])
            raise ValueError(
                f"non-conforming boundary (hanging edge/face) at hex index {h_bad}"
            )

        self.boundary_edge_mask = bnd_edge_counts > 0
        self.unknown_edges = np.flatnonzero(~self.boundary_edge_mask)
        self.n_unknown_edges = int(self.unknown_edges.size)

        self.boundary_node_mask = np.zeros(self.n_nodes, dtype=bool)
        self.boundary_node_mask[self.face_nodes_sorted[self.boundary_faces].ravel()] = (
            True
        )

        p0 = self.nodes[self.edge_nodes[:, 0]]
        p1 = self.nodes[self.edge_nodes[:, 1]]
        self.edge_vectors = p1 - p0
        self.edge_midpoints = 0.5 * (p0 + p1)
        self.edge_lengths = np.linalg.norm(self.edge_vectors, axis=1)

    def _compute_face_geometry(self):
        """Primal face shoelace vector areas ``A_f`` [m^2] and centroids [m]."""
        X = self.nodes
        q = self.face_cycles
        p0 = X[q[:, 0]]
        p1 = X[q[:, 1]]
        p2 = X[q[:, 2]]
        p3 = X[q[:, 3]]
        af = 0.5 * np.cross(p2 - p0, p3 - p1)
        gf = 0.5 * (0.5 * (p0 + p1) + 0.5 * (p3 + p2))
        return af, gf

    def _build_boundary_patches_and_normals(self, boundary_patch, sharp_nodes):
        """Build boundary patch labels, per-patch node unit normals, combined
        unit node normals, and sharp node classification."""
        n_bf = self.n_boundary_faces
        bf_idx = self.boundary_faces
        h_own = self.boundary_face_hex
        a_own = self.boundary_face_axis
        s_own = self.boundary_face_side

        sgn_out = (
            np.where(s_own == 0, -1.0, 1.0) * self.hex_face_signs[h_own, a_own, s_own]
        )
        af_out = sgn_out[:, None] * self.face_areas[bf_idx]
        af_norms = np.linalg.norm(af_out, axis=1, keepdims=True)
        u_out = np.where(af_norms > 0.0, af_out / af_norms, 0.0)

        if boundary_patch is None:
            # Default: one patch per distinct face normal direction within 1e-12
            patch_ids = np.full(n_bf, -1, dtype=np.int64)
            reps = []
            for i in range(n_bf):
                ui = u_out[i]
                matched = -1
                for p_id, u_rep in enumerate(reps):
                    if (
                        np.dot(ui, u_rep) > 0.0
                        and np.linalg.norm(np.cross(ui, u_rep)) <= 1e-12
                    ):
                        matched = p_id
                        break
                if matched < 0:
                    matched = len(reps)
                    reps.append(ui)
                patch_ids[i] = matched
            self.boundary_patch = patch_ids
        else:
            bp = np.asarray(boundary_patch, dtype=np.int64)
            if bp.shape == (self.n_cells, 3, 2):
                self.boundary_patch = bp[h_own, a_own, s_own].copy()
            elif bp.shape == (self.n_cells, 6):
                self.boundary_patch = bp[h_own, 2 * a_own + s_own].copy()
            elif bp.shape == (n_bf,):
                self.boundary_patch = bp.copy()
            else:
                raise ValueError(
                    f"boundary_patch must have shape {(n_bf,)} or "
                    f"{(self.n_cells, 3, 2)}, got {bp.shape}"
                )

        unique_patches = np.unique(self.boundary_patch)
        self.unique_patches = unique_patches
        n_patches = int(unique_patches.size)

        patch_node_normals = np.zeros((n_patches, self.n_nodes, 3), dtype=float)
        patch_node_incident = np.zeros((n_patches, self.n_nodes), dtype=bool)
        total_area_normal = np.zeros((self.n_nodes, 3), dtype=float)

        # Accumulate boundary face outward area vectors per patch in deterministic
        # (a_own, s_own, h_own) order and corner order (p0, p3, p1, p2) matching HexBlock3D
        H = self.hexes
        for p_idx, p_label in enumerate(unique_patches):
            m_p = np.flatnonzero(self.boundary_patch == p_label)
            order = np.lexsort((h_own[m_p], s_own[m_p], a_own[m_p]))
            m_ord = m_p[order]

            n_sum = np.zeros((self.n_nodes, 3), dtype=float)
            af_p = af_out[m_ord]
            h_p = h_own[m_ord]
            a_p = a_own[m_ord]
            s_p = s_own[m_ord]
            # Local face corners (p0, p1, p2, p3); HexBlock3D adds (db=0,dc=0)->p0,
            # (db=0,dc=1)->p3, (db=1,dc=0)->p1, (db=1,dc=1)->p2
            for c_pos in (0, 3, 1, 2):
                v_local = _LOCAL_FACE_VTK[a_p, s_p, c_pos]
                nodes_c = H[h_p, v_local]
                np.add.at(n_sum, nodes_c, af_p)
                patch_node_incident[p_idx, nodes_c] = True

            total_area_normal += n_sum
            norm = np.linalg.norm(n_sum, axis=1, keepdims=True)
            valid = norm[:, 0] > 0.0
            patch_node_normals[p_idx, valid] = n_sum[valid] / norm[valid]

        total_norm = np.linalg.norm(total_area_normal, axis=1, keepdims=True)
        node_normals = np.zeros((self.n_nodes, 3), dtype=float)
        valid_tot = total_norm[:, 0] > 0.0
        node_normals[valid_tot] = total_area_normal[valid_tot] / total_norm[valid_tot]

        self.wall_node_normals = patch_node_normals
        self.patch_node_incident = patch_node_incident
        self.node_wall_count = np.sum(patch_node_incident, axis=0)
        self.node_normals = node_normals

        # Parallel patch normal check at each boundary node
        bnd_nodes = np.flatnonzero(self.boundary_node_mask)
        parallel_mask = np.ones(bnd_nodes.size, dtype=bool)
        n_dir = np.zeros((bnd_nodes.size, 3), dtype=float)
        has_ref = np.zeros(bnd_nodes.size, dtype=bool)
        for p_idx in range(n_patches):
            hit = patch_node_incident[p_idx, bnd_nodes]
            if not np.any(hit):
                continue
            n_k = patch_node_normals[p_idx, bnd_nodes[hit]]
            first = ~has_ref[hit]
            if np.any(first):
                hit_first = np.flatnonzero(hit)[first]
                n_dir[hit_first] = n_k[first]
                has_ref[hit_first] = True
            rest = has_ref[hit] & (~first)
            if np.any(rest):
                hit_rest = np.flatnonzero(hit)[rest]
                cross_norm = np.linalg.norm(
                    np.cross(n_dir[hit_rest], n_k[rest]), axis=1
                )
                dot_val = np.sum(n_dir[hit_rest] * n_k[rest], axis=1)
                parallel_mask[hit_rest] &= (cross_norm <= 1e-12) & (dot_val > 0.0)

        non_parallel_nodes = np.zeros(self.n_nodes, dtype=bool)
        non_parallel_nodes[bnd_nodes[~parallel_mask]] = True
        self.non_parallel_patch_nodes = non_parallel_nodes

        if sharp_nodes is None:
            self.sharp_nodes = non_parallel_nodes.copy()
            self._has_explicit_sharp_nodes = False
        else:
            sn = np.asarray(sharp_nodes, dtype=bool)
            if sn.shape != (self.n_nodes,):
                raise ValueError(
                    f"sharp_nodes must have shape {(self.n_nodes,)}, got {sn.shape}"
                )
            self.sharp_nodes = sn.copy()
            self._has_explicit_sharp_nodes = True

        if self._has_explicit_sharp_nodes and not self._ignore_sharp_nodes:
            self.effective_sharp_nodes = self.sharp_nodes & self.boundary_node_mask
        else:
            self.effective_sharp_nodes = non_parallel_nodes

        self.node_octant_counts = np.bincount(
            self.hexes.ravel(), minlength=self.n_nodes
        )
        self.n_sharp_octants = int(
            np.sum(self.node_octant_counts[self.effective_sharp_nodes])
        )
        self.n_wall_octants = int(
            np.sum(
                self.node_octant_counts[
                    self.boundary_node_mask & (~self.effective_sharp_nodes)
                ]
            )
        )

    def _compute_dual_faces(self):
        """Per-hex dual face quadrilateral pieces [m^2] and assembled whole dual
        face vector areas ``S_e`` [m^2] for each primal edge."""
        pieces = np.empty((self.n_cells, 3, 2, 2, 3), dtype=float)
        S_e = np.zeros((self.n_edges, 3), dtype=float)
        x_hex = self.cell_centroids

        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            for db in (0, 1):
                for dc in (0, 1):
                    e_idx = self.hex_edges[:, a, db, dc]
                    e_sgn = self.hex_edge_signs[:, a, db, dc]
                    fb_idx = self.hex_faces[:, b, db]
                    fc_idx = self.hex_faces[:, c, dc]

                    m_e = self.edge_midpoints[e_idx]
                    g_fb = self.face_centroids[fb_idx]
                    g_fc = self.face_centroids[fc_idx]

                    sgn = (1 - 2 * db) * (1 - 2 * dc)
                    piece = (0.5 * sgn * e_sgn)[:, None] * np.cross(
                        x_hex - m_e, g_fb - g_fc
                    )
                    pieces[:, a, db, dc, :] = piece
                    np.add.at(S_e, e_idx, piece)

        return pieces, S_e

    # -------------------------------------------------------------- operators
    def curl(self):
        """
        Topological curl matrix ``C`` (``n_faces x n_edges``) mapping primal
        edge voltages to primal face circulations, with entries in
        ``{-1, 0, +1}``.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        rows = np.repeat(np.arange(self.n_faces, dtype=np.int64), 4)
        cols = self.face_edges.ravel()
        vals = self.face_edge_signs.ravel()
        return sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(self.n_faces, self.n_edges),
        )

    def divergence(self):
        """
        Topological cell-face divergence matrix ``D`` (``n_cells x n_faces``)
        with entries in ``{-1, 0, +1}``.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        rows, cols, vals = [], [], []
        cells = np.arange(self.n_cells, dtype=np.int64)
        for a in range(3):
            for side, s_out in ((0, -1.0), (1, 1.0)):
                rows.append(cells)
                cols.append(self.hex_faces[:, a, side])
                vals.append(s_out * self.hex_face_signs[:, a, side])
        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_cells, self.n_faces),
        )

    def gradient(self):
        """
        Full topological node-to-edge gradient matrix ``G``
        (``n_edges x n_nodes``) with entries in ``{-1, 0, +1}``, satisfying
        ``C @ G == 0``.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        e_idx = np.arange(self.n_edges, dtype=np.int64)
        rows = np.concatenate([e_idx, e_idx])
        cols = np.concatenate([self.edge_nodes[:, 0], self.edge_nodes[:, 1]])
        vals = np.concatenate(
            [-np.ones(self.n_edges, dtype=float), np.ones(self.n_edges, dtype=float)]
        )
        return sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(self.n_edges, self.n_nodes),
        )

    def gradient_unknown(self):
        """
        Topological node-to-edge gradient matrix ``G_u`` restricted to unknown
        edges and interior nodes, satisfying ``C_u @ G_u == 0``.

        Returns
        -------
        scipy.sparse.csr_matrix
            Matrix of shape ``(n_unknown_edges, n_interior_nodes)``.
        """
        int_nodes = np.flatnonzero(~self.boundary_node_mask)
        n_int = int(int_nodes.size)
        if n_int == 0 or self.n_unknown_edges == 0:
            return sp.csr_matrix((self.n_unknown_edges, 0), dtype=float)

        node_map = -np.ones(self.n_nodes, dtype=np.int64)
        node_map[int_nodes] = np.arange(n_int, dtype=np.int64)

        u_edges = self.unknown_edges
        u_pairs = self.edge_nodes[u_edges]
        rows, cols, vals = [], [], []
        for end, sign in ((0, -1.0), (1, 1.0)):
            c_idx = node_map[u_pairs[:, end]]
            valid = c_idx >= 0
            rows.append(np.flatnonzero(valid))
            cols.append(c_idx[valid])
            vals.append(np.full(int(np.sum(valid)), sign, dtype=float))

        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_unknown_edges, n_int),
        )

    def _ensure_solver(self):
        if self._solver_cache is None:
            from .solver3d import CurvilinearFIT3D

            self._solver_cache = CurvilinearFIT3D(self)
        return self._solver_cache

    @property
    def C(self):
        """Topological curl matrix ``C`` (``n_faces x n_edges``)."""
        return self.curl()

    @property
    def W_nu(self):
        """Magnetic Hodge matrix ``W_nu`` (``n_faces x n_faces``) [1/H]."""
        return self._ensure_solver().W_nu

    @property
    def W_eps(self):
        """Restricted electric Hodge matrix ``W_eps`` [1/F]."""
        return self._ensure_solver().W_eps

    @property
    def W_eps_full(self):
        """Full electric Hodge matrix ``W_eps_full`` (``n_edges x n_edges``) [1/F]."""
        return self._ensure_solver().W_eps_full

    @property
    def Q_eps_sqrt(self):
        """Factored square-root ``Q_eps_sqrt`` of the electric Hodge."""
        return self._ensure_solver().Q_eps_sqrt


def mesh_quality(mesh):
    """
    Compute mesh quality and corner reconstruction metrics for a `HexMesh`
    (or `HexBlock3D`).

    Parameters
    ----------
    mesh : HexMesh or HexBlock3D
        Hexahedral mesh to inspect.

    Returns
    -------
    dict
        Dictionary containing:

        * ``'min_scaled_jacobian'``: minimum over all cells and 8 corners of
          ``det(J) / (||dxi|| ||deta|| ||dzeta||)``;
        * ``'max_cond_magnetic'``: maximum 2-norm condition number of the
          corner primal-face area matrix ``M_c``;
        * ``'max_cond_electric'``: maximum 2-norm condition number of the
          corner dual-face area matrix ``N_c``;
        * ``'max_corner_cond'``: ``max(max_cond_magnetic, max_cond_electric)``;
        * ``'n_hex'``: number of hexahedral cells;
        * ``'n_nodes'``: number of mesh nodes;
        * ``'n_sharp_octants'``: number of sharp PEC boundary octants;
        * ``'n_wall_octants'``: number of smooth PEC wall octants.
    """
    if isinstance(mesh, HexBlock3D):
        mesh = HexMesh.from_block(mesh)
    elif not isinstance(mesh, HexMesh):
        raise TypeError("mesh must be a HexMesh or HexBlock3D instance")

    min_sj = float(np.min(mesh.scaled_corner_jacobians))
    A_f = mesh.face_areas
    S_e = mesh.dual_face_areas

    max_cond_mag = 0.0
    max_cond_elec = 0.0
    n_wall_octants = 0
    n_sharp_octants = 0

    H = mesh.hexes
    bnd_n = mesh.boundary_node_mask
    sharp_n = mesh.effective_sharp_nodes

    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                f0 = mesh.hex_faces[:, 0, a]
                f1 = mesh.hex_faces[:, 1, b]
                f2 = mesh.hex_faces[:, 2, c]
                s_f0 = mesh.hex_face_signs[:, 0, a]
                s_f1 = mesh.hex_face_signs[:, 1, b]
                s_f2 = mesh.hex_face_signs[:, 2, c]
                M_c = np.stack(
                    [
                        s_f0[:, None] * A_f[f0],
                        s_f1[:, None] * A_f[f1],
                        s_f2[:, None] * A_f[f2],
                    ],
                    axis=1,
                )
                max_cond_mag = max(max_cond_mag, float(np.max(np.linalg.cond(M_c))))

                e0 = mesh.hex_edges[:, 0, b, c]
                e1 = mesh.hex_edges[:, 1, c, a]
                e2 = mesh.hex_edges[:, 2, a, b]
                s_e0 = mesh.hex_edge_signs[:, 0, b, c]
                s_e1 = mesh.hex_edge_signs[:, 1, c, a]
                s_e2 = mesh.hex_edge_signs[:, 2, a, b]
                N_c = np.stack(
                    [
                        s_e0[:, None] * S_e[e0],
                        s_e1[:, None] * S_e[e1],
                        s_e2[:, None] * S_e[e2],
                    ],
                    axis=1,
                )
                c_nodes = H[:, _CORNER_VTK[a, b, c]]
                on_b = bnd_n[c_nodes]
                int_m = ~on_b
                if np.any(int_m):
                    max_cond_elec = max(
                        max_cond_elec, float(np.max(np.linalg.cond(N_c[int_m])))
                    )
                else:
                    max_cond_elec = max(
                        max_cond_elec, float(np.max(np.linalg.cond(N_c)))
                    )
                is_sh = on_b & sharp_n[c_nodes]
                is_wl = on_b & (~sharp_n[c_nodes])
                n_sharp_octants += int(np.sum(is_sh))
                n_wall_octants += int(np.sum(is_wl))

    return {
        "min_scaled_jacobian": min_sj,
        "max_cond_magnetic": max_cond_mag,
        "max_cond_electric": max_cond_elec,
        "max_corner_cond": max(max_cond_mag, max_cond_elec),
        "n_hex": int(mesh.n_cells),
        "n_nodes": int(mesh.n_nodes),
        "n_sharp_octants": n_sharp_octants,
        "n_wall_octants": n_wall_octants,
    }
