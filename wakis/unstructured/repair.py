# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Topological flip repair of non-Delaunay tetrahedral meshes for Voronoi FIT.

With the Voronoi Hodge, the dual edge of a primal face f joins the circumcentres
of the two tetrahedra sharing f, and its material-weighted signed length H_f is
negative when f violates the local Delaunay condition, making the curl-curl
matrix indefinite. Local bistellar flips (2-3, 3-2, 4-4 in the interior and 2-2 on
planar boundaries) replace non-Delaunay simplices inside a single material tag;
coplanar diagonal flips also apply across a planar material interface when
adjacent coplanar interface triangles separate the same pair of material tags
and share one opposite apex on each side, preserving the interface plane and
per-tag volumes. Flips that would remove an edge whose two endpoints are both
locked (e.g. on fixed CAD curves or corners) are refused. Stuck interior or
interface faces (B. Joe, Construction of three-dimensional Delaunay
triangulations using local transformations, Computer Aided Geometric Design
8(2), 123-142 (1991), DOI 10.1016/0167-8396(91)90038-d; H. Edelsbrunner and
N. R. Shah, Incremental topological flipping works for regular triangulations,
Algorithmica 15(3), 223-241 (1996), DOI 10.1007/BF01975867) are resolved by
inserting a Steiner node at the tet-pair circumcentre or by bisecting an
unlocked edge of the offending face while preserving each cell's material tag.
"""

from collections import deque

import numpy as np

from ._core import cell_materials
from .improve import LOC_E, _default_free, _State, tet_partials
from .mesh import SimplexMesh

__all__ = ["flip_to_delaunay"]

_LOC_F = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))


def _sface(a, b, c):
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    return (int(a), int(b), int(c))


def _sedge(a, b):
    return (int(a), int(b)) if a < b else (int(b), int(a))


def _orient_tet(X, a, b, c, d):
    ab = X[b] - X[a]
    ac = X[c] - X[a]
    ad = X[d] - X[a]
    det = (
        ab[0] * (ac[1] * ad[2] - ac[2] * ad[1])
        - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
        + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0])
    )
    if det < 0.0:
        return (int(a), int(c), int(b), int(d))
    return (int(a), int(b), int(c), int(d))


def _tet_circumcentre(P):
    E = P[:, 1:] - P[:, :1]
    rhs = 0.5 * np.einsum("nij,nij->ni", E, E)
    return P[:, 0] + np.linalg.solve(E, rhs[..., None])[..., 0]


def _tet_volumes(P):
    return np.linalg.det(P[:, 1:] - P[:, :1]) / 6.0


class _RepairState:
    """Mutable tetrahedral mesh connectivity and dual lengths for flip repair."""

    def __init__(self, mesh, materials, bc):
        eps_r, mu_r, _, pec, _ = cell_materials(mesh, materials)
        live = (~pec).astype(float)
        free = _default_free(mesh, pec, bc)
        st = _State(mesh, eps_r, mu_r, live, free)

        self.X = mesh.nodes.copy()
        self.bc = bc
        self.materials = materials
        self.initial_neg_count = int(st.neg_faces().size)

        self.tag_inv_mu = {}
        self.tag_pec = {}
        for c in range(mesh.n_cells):
            t = int(mesh.cell_tags[c])
            if t not in self.tag_inv_mu:
                self.tag_inv_mu[t] = float(live[c] / mu_r[c])
                self.tag_pec[t] = bool(pec[c])

        self.cells = {}
        self.tags = {}
        self.vol = {}
        self.h = {}
        self.inv_mu = {}
        self.pec_cell = {}
        self.face_map = {}
        self.edge_map = {}

        _, h_all, _, vol_all = tet_partials(self.X[mesh.cells])
        for cid in range(mesh.n_cells):
            tet = tuple(int(v) for v in mesh.cells[cid])
            tag = int(mesh.cell_tags[cid])
            self.cells[cid] = tet
            self.tags[cid] = tag
            self.vol[cid] = float(vol_all[cid])
            self.h[cid] = h_all[cid].copy()
            self.inv_mu[cid] = self.tag_inv_mu[tag]
            self.pec_cell[cid] = self.tag_pec[tag]
            for k, (i0, i1, i2) in enumerate(_LOC_F):
                f = _sface(tet[i0], tet[i1], tet[i2])
                self.face_map.setdefault(f, []).append((cid, k))
            for i0, i1 in LOC_E:
                e = _sedge(tet[i0], tet[i1])
                self.edge_map.setdefault(e, set()).add(cid)

        self.next_cid = mesh.n_cells
        self.min_vol = 1e-11 * float(vol_all.max()) if vol_all.size else 0.0
        self.bnd_edges = set()
        for f, entries in self.face_map.items():
            if len(entries) == 1:
                a, b, c = f
                self.bnd_edges.add(_sedge(a, b))
                self.bnd_edges.add(_sedge(a, c))
                self.bnd_edges.add(_sedge(b, c))
        self.neg_set = {f for f in self.face_map if self.is_neg_face(f)}
        self._undo_log = None

    def edge_free(self, e, bnd_edges=None):
        cids = self.edge_map.get(e)
        if not cids:
            return False
        be = self.bnd_edges if bnd_edges is None else bnd_edges
        if self.bc == "pec" and e in be:
            return False
        has_live = False
        for c in cids:
            if self.pec_cell[c]:
                return False
            has_live = True
        return has_live

    def face_free(self, f, bnd_edges=None):
        a, b, c = f
        return (
            self.edge_free((a, b), bnd_edges)
            or self.edge_free((a, c), bnd_edges)
            or self.edge_free((b, c), bnd_edges)
        )

    def face_H(self, f):
        entries = self.face_map.get(f)
        if not entries:
            return 0.0
        return sum(self.inv_mu[cid] * float(self.h[cid][k]) for cid, k in entries)

    def is_neg_face(self, f):
        return f in self.face_map and self.face_free(f) and self.face_H(f) < 0.0

    def is_interface_face(self, f):
        entries = self.face_map.get(f)
        if not entries or len(entries) != 2:
            return False
        return self.tags[entries[0][0]] != self.tags[entries[1][0]]

    def evaluate_cavity(
        self,
        old_cids,
        new_tets,
        h_new,
        tag,
        target_face=None,
        removed_bnd_edges=(),
        added_bnd_edges=(),
    ):
        """Check signed dual lengths on the flip cavity before committing."""
        old_set = set(old_cids)
        if isinstance(tag, (int, np.integer)):
            tags_new = [int(tag)] * len(new_tets)
        else:
            tags_new = [int(t) for t in tag]

        # Collect old faces touched by old_cids
        old_faces = set()
        for cid in old_cids:
            tet = self.cells[cid]
            for i0, i1, i2 in _LOC_F:
                old_faces.add(_sface(tet[i0], tet[i1], tet[i2]))

        n_neg_old = sum(1 for f in old_faces if self.is_neg_face(f))

        # Build new face contributions from new_tets
        new_contrib = {}
        new_face_tags = {}
        for idx, tet in enumerate(new_tets):
            t_idx = tags_new[idx]
            w_idx = self.tag_inv_mu[t_idx]
            for k, (i0, i1, i2) in enumerate(_LOC_F):
                f = _sface(tet[i0], tet[i1], tet[i2])
                new_contrib.setdefault(f, []).append(w_idx * float(h_new[idx, k]))
                new_face_tags.setdefault(f, []).append(t_idx)

        # Updated boundary edges if 2-2 flip
        if removed_bnd_edges or added_bnd_edges:
            bnd_edges_after = (self.bnd_edges - set(removed_bnd_edges)) | set(
                added_bnd_edges
            )
        else:
            bnd_edges_after = self.bnd_edges

        def edge_free_after(e):
            if self.bc == "pec" and e in bnd_edges_after:
                return False
            cids = self.edge_map.get(e, set()) - old_set
            for c in cids:
                if self.pec_cell[c]:
                    return False
            for idx, tet in enumerate(new_tets):
                if self.tag_pec[tags_new[idx]] and e[0] in tet and e[1] in tet:
                    return False
            return True

        def face_free_after(f):
            a, b, c = f
            return (
                edge_free_after((a, b))
                or edge_free_after((a, c))
                or edge_free_after((b, c))
            )

        n_neg_new_int = 0
        n_neg_new_total = 0
        target_H_after = None

        for f, vals in new_contrib.items():
            ext_entries = [
                (cid, k) for cid, k in self.face_map.get(f, ()) if cid not in old_set
            ]
            H_ext = sum(
                self.inv_mu[cid] * float(self.h[cid][k]) for cid, k in ext_entries
            )
            H_after = H_ext + sum(vals)
            is_free = face_free_after(f)
            is_neg = is_free and (H_after < 0.0)

            if not ext_entries:
                # Internal face of new_tets (or new outer boundary face in 2-2)
                if is_neg:
                    n_neg_new_int += 1
            else:
                # Cavity boundary face shared with an external tet
                ext_cid = ext_entries[0][0]
                t_int = new_face_tags[f][0]
                if self.tags[ext_cid] != t_int and is_neg and not self.is_neg_face(f):
                    # Never make a non-negative interface face negative
                    return False
            if is_neg:
                n_neg_new_total += 1
            if f == target_face:
                target_H_after = H_after

        if n_neg_new_int > 0:
            return False

        if target_face is not None and target_face not in new_contrib:
            # Direct flip removed target_face
            return True

        if target_face is not None and target_face in new_contrib:
            # Indirect flip on a neighbour of target_face: must repair target_face
            # and strictly reduce negative faces on the cavity
            if target_H_after is None or target_H_after < 0.0:
                return False
            return n_neg_new_total < n_neg_old

        return n_neg_new_total < n_neg_old

    def replace_cells(
        self, old_cids, new_tets, h_new, vol_new, tag, removed_bnd=(), added_bnd=()
    ):
        if isinstance(tag, (int, np.integer)):
            tags_new = [int(tag)] * len(new_tets)
        else:
            tags_new = [int(t) for t in tag]

        touched_faces = set()
        old_records = []
        for cid in old_cids:
            tet = self.cells.pop(cid)
            t_old = self.tags.pop(cid)
            v_old = self.vol.pop(cid)
            h_old = self.h.pop(cid)
            self.inv_mu.pop(cid)
            self.pec_cell.pop(cid)
            if self._undo_log is not None:
                old_records.append((cid, tet, t_old, v_old, h_old))
            for k, (i0, i1, i2) in enumerate(_LOC_F):
                f = _sface(tet[i0], tet[i1], tet[i2])
                touched_faces.add(f)
                lst = self.face_map.get(f)
                if lst is not None:
                    lst[:] = [item for item in lst if item[0] != cid]
                    if not lst:
                        del self.face_map[f]
            for i0, i1 in LOC_E:
                e = _sedge(tet[i0], tet[i1])
                s = self.edge_map.get(e)
                if s is not None:
                    s.discard(cid)
                    if not s:
                        del self.edge_map[e]

        actual_removed_bnd = [e for e in removed_bnd if e in self.bnd_edges]
        actual_added_bnd = [e for e in added_bnd if e not in self.bnd_edges]
        for e in actual_removed_bnd:
            self.bnd_edges.discard(e)
        for e in actual_added_bnd:
            self.bnd_edges.add(e)

        new_cids = []
        for idx, tet in enumerate(new_tets):
            t_idx = tags_new[idx]
            cid = self.next_cid
            self.next_cid += 1
            tet_t = tuple(int(v) for v in tet)
            self.cells[cid] = tet_t
            self.tags[cid] = t_idx
            self.vol[cid] = float(vol_new[idx])
            self.h[cid] = h_new[idx].copy()
            self.inv_mu[cid] = self.tag_inv_mu[t_idx]
            self.pec_cell[cid] = self.tag_pec[t_idx]
            for k, (i0, i1, i2) in enumerate(_LOC_F):
                f = _sface(tet_t[i0], tet_t[i1], tet_t[i2])
                touched_faces.add(f)
                self.face_map.setdefault(f, []).append((cid, k))
            for i0, i1 in LOC_E:
                e = _sedge(tet_t[i0], tet_t[i1])
                self.edge_map.setdefault(e, set()).add(cid)
            new_cids.append(cid)

        for f in touched_faces:
            if self.is_neg_face(f):
                self.neg_set.add(f)
            else:
                self.neg_set.discard(f)

        if self._undo_log is not None:
            self._undo_log.append(
                (old_records, new_cids, actual_removed_bnd, actual_added_bnd)
            )
        return new_cids

    def checkpoint(self):
        if self._undo_log is None:
            self._undo_log = []
        return (self.X.shape[0], self.next_cid, len(self._undo_log))

    def rollback(self, cp):
        n_nodes0, next_cid0, log_len = cp
        while len(self._undo_log) > log_len:
            old_records, new_cids, actual_removed_bnd, actual_added_bnd = (
                self._undo_log.pop()
            )
            touched_faces = set()
            for cid in new_cids:
                tet = self.cells.pop(cid)
                self.tags.pop(cid)
                self.vol.pop(cid)
                self.h.pop(cid)
                self.inv_mu.pop(cid)
                self.pec_cell.pop(cid)
                for k, (i0, i1, i2) in enumerate(_LOC_F):
                    f = _sface(tet[i0], tet[i1], tet[i2])
                    touched_faces.add(f)
                    lst = self.face_map.get(f)
                    if lst is not None:
                        lst[:] = [item for item in lst if item[0] != cid]
                        if not lst:
                            del self.face_map[f]
                for i0, i1 in LOC_E:
                    e = _sedge(tet[i0], tet[i1])
                    s = self.edge_map.get(e)
                    if s is not None:
                        s.discard(cid)
                        if not s:
                            del self.edge_map[e]

            for e in actual_added_bnd:
                self.bnd_edges.discard(e)
            for e in actual_removed_bnd:
                self.bnd_edges.add(e)

            for cid, tet_t, t_old, v_old, h_old in old_records:
                self.cells[cid] = tet_t
                self.tags[cid] = t_old
                self.vol[cid] = v_old
                self.h[cid] = h_old
                self.inv_mu[cid] = self.tag_inv_mu[t_old]
                self.pec_cell[cid] = self.tag_pec[t_old]
                for k, (i0, i1, i2) in enumerate(_LOC_F):
                    f = _sface(tet_t[i0], tet_t[i1], tet_t[i2])
                    touched_faces.add(f)
                    self.face_map.setdefault(f, []).append((cid, k))
                for i0, i1 in LOC_E:
                    e = _sedge(tet_t[i0], tet_t[i1])
                    self.edge_map.setdefault(e, set()).add(cid)

            for f in touched_faces:
                if self.is_neg_face(f):
                    self.neg_set.add(f)
                else:
                    self.neg_set.discard(f)

        self.X = self.X[:n_nodes0]
        self.next_cid = next_cid0
        if log_len == 0:
            self._undo_log = None

    def cells_faces(self, cids):
        faces = set()
        for cid in cids:
            if cid in self.cells:
                tet = self.cells[cid]
                for i0, i1, i2 in _LOC_F:
                    faces.add(_sface(tet[i0], tet[i1], tet[i2]))
        return faces

    def to_mesh(self, mesh0):
        cids = sorted(self.cells.keys())
        cells_arr = np.array([self.cells[c] for c in cids], dtype=np.int64)
        tags_arr = np.array([self.tags[c] for c in cids], dtype=mesh0.cell_tags.dtype)
        new_mesh = SimplexMesh(self.X, cells_arr, tags_arr)
        for attr in ("region_names", "surface_groups"):
            if hasattr(mesh0, attr):
                setattr(new_mesh, attr, getattr(mesh0, attr))
        return new_mesh


def _flip_23(state, f, locked, target_face=None, check_delaunay=True):
    """Bistellar 2-3 flip replacing interior face f with a transverse edge."""
    entries = state.face_map.get(f)
    if not entries or len(entries) != 2:
        return False, "boundary"
    (c1, k1), (c2, k2) = entries
    tag = state.tags[c1]
    if state.tags[c2] != tag:
        return False, "interface"

    tet1 = state.cells[c1]
    tet2 = state.cells[c2]
    d = tet1[k1]
    e = tet2[k2]
    if d == e:
        return False, "topology"

    # Order (u, v, w) of f so that (u, v, w, d) has positive volume
    u, v, w = f
    if _orient_tet(state.X, u, v, w, d) != (u, v, w, d):
        v, w = w, v

    de_edge = _sedge(d, e)
    if de_edge in state.edge_map:
        return False, "topology"
    for x in (u, v, w):
        if _sface(x, d, e) in state.face_map:
            return False, "topology"

    new_tets = [
        (u, v, e, d),
        (v, w, e, d),
        (w, u, e, d),
    ]
    P = state.X[np.asarray(new_tets, dtype=np.int64)]
    vol_old = state.vol[c1] + state.vol[c2]
    vol_new = _tet_volumes(P)
    min_v = max(state.min_vol, 1e-9 * vol_old)
    if not (
        np.all(vol_new > min_v) and abs(vol_new.sum() - vol_old) <= 1e-12 * vol_old
    ):
        return False, "volume"
    _, h_new, _, vol_new = tet_partials(P)

    tf = f if target_face is None else target_face
    if check_delaunay and not state.evaluate_cavity(
        (c1, c2), new_tets, h_new, tag, target_face=tf
    ):
        return False, "delaunay"

    new_cids = state.replace_cells((c1, c2), new_tets, h_new, vol_new, tag)
    return True, new_cids


def _flip_32(state, edge, locked, target_face=None, check_delaunay=True):
    """Bistellar 3-2 flip replacing degree-3 interior edge with a triangle."""
    if edge in state.bnd_edges:
        return False, "boundary"
    cids = list(state.edge_map.get(edge, ()))
    if len(cids) != 3:
        return False, "topology"

    tags = {state.tags[c] for c in cids}
    if len(tags) != 1:
        return False, "interface"
    tag = next(iter(tags))

    a, b = edge
    if locked[a] and locked[b]:
        return False, "locked"

    other_verts = set()
    for cid in cids:
        for v in state.cells[cid]:
            if v != a and v != b:
                other_verts.add(v)
    if len(other_verts) != 3:
        return False, "topology"
    u, v, w = sorted(other_verts)

    # All 3 internal faces around edge (a, b) must belong only to the 3 tets
    cset = set(cids)
    for x in (u, v, w):
        f_abx = _sface(a, b, x)
        entries = state.face_map.get(f_abx, ())
        if len(entries) != 2 or not all(c in cset for c, _ in entries):
            return False, "topology"

    new_face = _sface(u, v, w)
    if new_face in state.face_map:
        return False, "topology"

    # Orient (u, v, w, a) positively and (u, w, v, b) on the opposite side
    t0 = _orient_tet(state.X, u, v, w, a)
    t1 = (t0[0], t0[2], t0[1], int(b))
    new_tets = [t0, t1]
    P = state.X[np.asarray(new_tets, dtype=np.int64)]
    vol_old = sum(state.vol[c] for c in cids)
    vol_new = _tet_volumes(P)
    min_v = max(state.min_vol, 1e-9 * vol_old)
    if not (
        np.all(vol_new > min_v) and abs(vol_new.sum() - vol_old) <= 1e-12 * vol_old
    ):
        return False, "volume"
    _, h_new, _, vol_new = tet_partials(P)

    if check_delaunay and not state.evaluate_cavity(
        cids, new_tets, h_new, tag, target_face=target_face
    ):
        return False, "delaunay"

    new_cids = state.replace_cells(cids, new_tets, h_new, vol_new, tag)
    return True, new_cids


def _flip_44(state, edge, locked, target_face=None, check_delaunay=True):
    """Bistellar 4-4 flip on a degree-4 interior edge in an octahedron."""
    if edge in state.bnd_edges:
        return False, "boundary"
    cids = list(state.edge_map.get(edge, ()))
    if len(cids) != 4:
        return False, "topology"

    tags = {state.tags[c] for c in cids}
    if len(tags) != 1:
        return False, "interface"
    tag = next(iter(tags))

    a, b = edge
    if locked[a] and locked[b]:
        return False, "locked"

    # Extract the 4-cycle of link edges around (a, b)
    cset = set(cids)
    link_adj = {}
    for cid in cids:
        rem = [v for v in state.cells[cid] if v != a and v != b]
        if len(rem) != 2:
            return False, "topology"
        p, q = rem
        link_adj.setdefault(p, []).append(q)
        link_adj.setdefault(q, []).append(p)

    if len(link_adj) != 4 or any(len(nb) != 2 for nb in link_adj.values()):
        return False, "topology"

    for v in link_adj:
        entries = state.face_map.get(_sface(a, b, v), ())
        if len(entries) != 2 or not all(c in cset for c, _ in entries):
            return False, "topology"

    v0 = min(link_adj.keys())
    v1 = link_adj[v0][0]
    v2 = next(x for x in link_adj[v1] if x != v0)
    v3 = next(x for x in link_adj[v0] if x != v1)
    if v3 not in link_adj[v2]:
        return False, "topology"

    vol_old = sum(state.vol[c] for c in cids)
    last_reason = "topology"

    # Try both diagonals (v0, v2) and (v1, v3) of the link 4-cycle
    for (u, w), (p, q) in (((v0, v2), (v1, v3)), ((v1, v3), (v0, v2))):
        uw_edge = _sedge(u, w)
        if uw_edge in state.edge_map:
            last_reason = "topology"
            continue
        if any(_sface(u, w, x) in state.face_map for x in (a, b, p, q)):
            last_reason = "topology"
            continue

        # On p-side: share face (u, w, p) with opposite vertices a, b
        # On q-side: share face (u, w, q) with opposite vertices a, b
        tp0 = _orient_tet(state.X, u, w, p, a)
        tp1 = (tp0[0], tp0[2], tp0[1], int(b))
        tq0 = _orient_tet(state.X, u, w, q, a)
        tq1 = (tq0[0], tq0[2], tq0[1], int(b))
        new_tets = [tp0, tp1, tq0, tq1]

        P = state.X[np.asarray(new_tets, dtype=np.int64)]
        vol_new = _tet_volumes(P)
        min_v = max(state.min_vol, 1e-9 * vol_old)
        if not (
            np.all(vol_new > min_v) and abs(vol_new.sum() - vol_old) <= 1e-12 * vol_old
        ):
            last_reason = "volume"
            continue

        # Also check that p and q lie on opposite sides of (u, w, a) and (u, w, b)
        ta0 = _orient_tet(state.X, u, w, a, p)
        ta1 = (ta0[0], ta0[2], ta0[1], int(q))
        tb0 = _orient_tet(state.X, u, w, b, p)
        tb1 = (tb0[0], tb0[2], tb0[1], int(q))
        v_ab_split = _tet_volumes(
            state.X[np.asarray([ta0, ta1, tb0, tb1], dtype=np.int64)]
        )
        if (
            np.any(v_ab_split <= min_v)
            or abs(v_ab_split.sum() - vol_old) > 1e-12 * vol_old
        ):
            last_reason = "volume"
            continue

        _, h_new, _, vol_new = tet_partials(P)

        if check_delaunay and not state.evaluate_cavity(
            cids, new_tets, h_new, tag, target_face=target_face
        ):
            last_reason = "delaunay"
            continue

        new_cids = state.replace_cells(cids, new_tets, h_new, vol_new, tag)
        return True, new_cids

    return False, last_reason


def _flip_22(state, edge, locked, target_face=None, check_delaunay=True):
    """Bistellar coplanar diagonal flip on a boundary or planar interface edge."""
    cids = list(state.edge_map.get(edge, ()))
    a, b = edge

    if edge in state.bnd_edges:
        if len(cids) != 2:
            return False, "topology"
        c1, c2 = cids
        tag = state.tags[c1]
        if state.tags[c2] != tag:
            return False, "interface"

        rem1 = set(state.cells[c1]) - {a, b}
        rem2 = set(state.cells[c2]) - {a, b}
        common = rem1 & rem2
        if len(common) != 1:
            return False, "topology"
        p = next(iter(common))
        c = next(iter(rem1 - {p}))
        d = next(iter(rem2 - {p}))

        fb1 = _sface(a, b, c)
        fb2 = _sface(a, b, d)
        if (
            len(state.face_map.get(fb1, ())) != 1
            or len(state.face_map.get(fb2, ())) != 1
        ):
            return False, "boundary"

        if (
            (locked[a] and locked[b])
            or all(locked[v] for v in fb1)
            or all(locked[v] for v in fb2)
        ):
            return False, "locked"

        cd_edge = _sedge(c, d)
        if cd_edge in state.edge_map:
            return False, "topology"
        for nf in (_sface(a, c, d), _sface(b, c, d), _sface(c, d, p)):
            if nf in state.face_map:
                return False, "topology"

        # Coplanarity and planar polygon area preservation to 1e-12 relative
        xa, xb, xc, xd = state.X[a], state.X[b], state.X[c], state.X[d]
        n1 = np.cross(xb - xa, xc - xa)
        n2 = np.cross(xd - xa, xb - xa)
        a1 = float(np.linalg.norm(n1))
        a2 = float(np.linalg.norm(n2))
        if a1 <= 0.0 or a2 <= 0.0:
            return False, "boundary"
        u1 = n1 / a1
        u2 = n2 / a2
        if float(np.dot(u1, u2)) < 1.0 - 1e-12:
            return False, "boundary"

        n1_new = np.cross(xa - xc, xd - xc)
        n2_new = np.cross(xb - xd, xc - xd)
        a1_new = float(np.linalg.norm(n1_new))
        a2_new = float(np.linalg.norm(n2_new))
        if a1_new <= 0.0 or a2_new <= 0.0:
            return False, "boundary"
        u1_new = n1_new / a1_new
        u2_new = n2_new / a2_new
        if (
            float(np.dot(u1_new, u1)) < 1.0 - 1e-12
            or float(np.dot(u2_new, u1)) < 1.0 - 1e-12
        ):
            return False, "boundary"

        area_old = 0.5 * (a1 + a2)
        area_new = 0.5 * (a1_new + a2_new)
        if abs(area_new - area_old) > 1e-12 * area_old:
            return False, "boundary"

        t0 = _orient_tet(state.X, c, d, p, a)
        t1 = (t0[0], t0[2], t0[1], int(b))
        new_tets = [t0, t1]
        P = state.X[np.asarray(new_tets, dtype=np.int64)]
        vol_old = state.vol[c1] + state.vol[c2]
        vol_new = _tet_volumes(P)
        min_v = max(state.min_vol, 1e-9 * vol_old)
        if not (
            np.all(vol_new > min_v) and abs(vol_new.sum() - vol_old) <= 1e-12 * vol_old
        ):
            return False, "volume"
        _, h_new, _, vol_new = tet_partials(P)

        tf = _sface(a, b, p) if target_face is None else target_face
        if check_delaunay and not state.evaluate_cavity(
            (c1, c2),
            new_tets,
            h_new,
            tag,
            target_face=tf,
            removed_bnd_edges=(edge,),
            added_bnd_edges=(cd_edge,),
        ):
            return False, "delaunay"

        new_cids = state.replace_cells(
            (c1, c2),
            new_tets,
            h_new,
            vol_new,
            tag,
            removed_bnd=(edge,),
            added_bnd=(cd_edge,),
        )
        return True, new_cids

    # Coplanar 2-2 flip on an interior material-interface edge shared by 4 tets
    # (2 in tag t1 sharing apex p1, 2 in tag t2 sharing apex p2)
    if len(cids) != 4:
        return False, "boundary"
    tags_set = {state.tags[c] for c in cids}
    if len(tags_set) != 2:
        return False, "boundary"
    t1, t2 = sorted(tags_set)
    cids1 = [c for c in cids if state.tags[c] == t1]
    cids2 = [c for c in cids if state.tags[c] == t2]
    if len(cids1) != 2 or len(cids2) != 2:
        return False, "interface"

    rem1a = set(state.cells[cids1[0]]) - {a, b}
    rem1b = set(state.cells[cids1[1]]) - {a, b}
    common1 = rem1a & rem1b
    if len(common1) != 1:
        return False, "topology"
    p1 = next(iter(common1))
    c = next(iter(rem1a - {p1}))
    d = next(iter(rem1b - {p1}))

    rem2a = set(state.cells[cids2[0]]) - {a, b}
    rem2b = set(state.cells[cids2[1]]) - {a, b}
    common2 = rem2a & rem2b
    if len(common2) != 1:
        return False, "topology"
    p2 = next(iter(common2))
    if (rem2a - {p2}) | (rem2b - {p2}) != {c, d} or p1 == p2:
        return False, "interface"

    fb1 = _sface(a, b, c)
    fb2 = _sface(a, b, d)
    if len(state.face_map.get(fb1, ())) != 2 or len(state.face_map.get(fb2, ())) != 2:
        return False, "topology"
    if (
        len(state.face_map.get(_sface(a, b, p1), ())) != 2
        or len(state.face_map.get(_sface(a, b, p2), ())) != 2
    ):
        return False, "topology"

    if (
        (locked[a] and locked[b])
        or all(locked[v] for v in fb1)
        or all(locked[v] for v in fb2)
    ):
        return False, "locked"

    cd_edge = _sedge(c, d)
    if cd_edge in state.edge_map:
        return False, "topology"
    for nf in (
        _sface(a, c, d),
        _sface(b, c, d),
        _sface(c, d, p1),
        _sface(c, d, p2),
    ):
        if nf in state.face_map:
            return False, "topology"

    xa, xb, xc, xd = state.X[a], state.X[b], state.X[c], state.X[d]
    n1 = np.cross(xb - xa, xc - xa)
    n2 = np.cross(xd - xa, xb - xa)
    a1 = float(np.linalg.norm(n1))
    a2 = float(np.linalg.norm(n2))
    if a1 <= 0.0 or a2 <= 0.0:
        return False, "boundary"
    u1 = n1 / a1
    u2 = n2 / a2
    if float(np.dot(u1, u2)) < 1.0 - 1e-12:
        return False, "boundary"

    n1_new = np.cross(xa - xc, xd - xc)
    n2_new = np.cross(xb - xd, xc - xd)
    a1_new = float(np.linalg.norm(n1_new))
    a2_new = float(np.linalg.norm(n2_new))
    if a1_new <= 0.0 or a2_new <= 0.0:
        return False, "boundary"
    u1_new = n1_new / a1_new
    u2_new = n2_new / a2_new
    if (
        float(np.dot(u1_new, u1)) < 1.0 - 1e-12
        or float(np.dot(u2_new, u1)) < 1.0 - 1e-12
    ):
        return False, "boundary"

    area_old = 0.5 * (a1 + a2)
    area_new = 0.5 * (a1_new + a2_new)
    if abs(area_new - area_old) > 1e-12 * area_old:
        return False, "boundary"

    t1_0 = _orient_tet(state.X, c, d, p1, a)
    t1_1 = (t1_0[0], t1_0[2], t1_0[1], int(b))
    t2_0 = _orient_tet(state.X, c, d, p2, a)
    t2_1 = (t2_0[0], t2_0[2], t2_0[1], int(b))
    new_tets = [t1_0, t1_1, t2_0, t2_1]
    tags_new = [t1, t1, t2, t2]

    P = state.X[np.asarray(new_tets, dtype=np.int64)]
    vol_old1 = state.vol[cids1[0]] + state.vol[cids1[1]]
    vol_old2 = state.vol[cids2[0]] + state.vol[cids2[1]]
    vol_new = _tet_volumes(P)
    min_v1 = max(state.min_vol, 1e-9 * vol_old1)
    min_v2 = max(state.min_vol, 1e-9 * vol_old2)
    if not (
        np.all(vol_new[:2] > min_v1)
        and abs(vol_new[:2].sum() - vol_old1) <= 1e-12 * vol_old1
        and np.all(vol_new[2:] > min_v2)
        and abs(vol_new[2:].sum() - vol_old2) <= 1e-12 * vol_old2
    ):
        return False, "volume"
    _, h_new, _, vol_new = tet_partials(P)

    old_all = cids1 + cids2
    tf = fb1 if target_face is None else target_face
    if check_delaunay and not state.evaluate_cavity(
        old_all, new_tets, h_new, tags_new, target_face=tf
    ):
        return False, "delaunay"

    new_cids = state.replace_cells(old_all, new_tets, h_new, vol_new, tags_new)
    return True, new_cids


def _insert_steiner(state, f, mode="1-4", host=0, alpha=0.0, locked=None):
    """
    Insert a Steiner node for a stuck face f while preserving material tags.

    Places the node at the circumcentre of the offending tet pair (or pulled
    toward the host centroid into the strict positive-volume kernel) or bisects
    an edge of f, assigning each resulting tetrahedron its parent cell's tag.
    """
    entries = state.face_map.get(f)
    if not entries or len(entries) != 2:
        return False, "boundary"
    (c1, k1), (c2, k2) = entries
    tag1 = state.tags[c1]
    tag2 = state.tags[c2]

    tet1 = state.cells[c1]
    tet2 = state.cells[c2]
    d = tet1[k1]
    e = tet2[k2]
    u, v, w = f
    if _orient_tet(state.X, u, v, w, d) != (u, v, w, d):
        v, w = w, v

    s_idx = state.X.shape[0]

    if mode in ("edge", "bnd_edge"):
        f_edges = sorted(
            (_sedge(u, v), _sedge(u, w), _sedge(v, w)),
            key=lambda ed: -float(np.linalg.norm(state.X[ed[1]] - state.X[ed[0]])),
        )
        ed = f_edges[host % 3]
        if mode == "edge" and ed in state.bnd_edges:
            return False, "boundary"
        if mode == "bnd_edge" and ed not in state.bnd_edges:
            return False, "boundary"
        cids = list(state.edge_map.get(ed, ()))
        if not cids:
            return False, "topology"
        a, b = ed
        if (
            tag1 == tag2
            and locked is not None
            and a < len(locked)
            and b < len(locked)
            and locked[a]
            and locked[b]
        ):
            return False, "locked"
        pt = (1.0 - alpha) * state.X[a] + alpha * state.X[b]
        X_try = np.vstack([state.X, pt[None, :]])
        new_tets = []
        tags_new = []
        for cid in cids:
            tet = state.cells[cid]
            rem = [x for x in tet if x != a and x != b]
            if len(rem) != 2:
                return False, "topology"
            p, q = rem
            t_a = _orient_tet(X_try, a, s_idx, p, q)
            t_b = _orient_tet(X_try, s_idx, b, p, q)
            P_ab = X_try[np.asarray([t_a, t_b], dtype=np.int64)]
            v_ab = _tet_volumes(P_ab)
            v_old = state.vol[cid]
            min_v = max(state.min_vol, 1e-6 * v_old)
            if not (np.all(v_ab > min_v) and abs(v_ab.sum() - v_old) <= 1e-12 * v_old):
                return False, "volume"
            new_tets.extend([t_a, t_b])
            tags_new.extend([state.tags[cid], state.tags[cid]])
        P = X_try[np.asarray(new_tets, dtype=np.int64)]
        _, h_new, _, vol_new = tet_partials(P)
        state.X = X_try
        removed_bnd = ()
        added_bnd = ()
        if ed in state.bnd_edges:
            removed_bnd = (ed,)
            added_bnd_list = [_sedge(a, s_idx), _sedge(s_idx, b)]
            for cid in cids:
                for x in state.cells[cid]:
                    if (
                        x != a
                        and x != b
                        and len(state.face_map.get(_sface(a, b, x), ())) == 1
                    ):
                        added_bnd_list.append(_sedge(s_idx, x))
            added_bnd = tuple(added_bnd_list)
        new_cids = state.replace_cells(
            cids,
            new_tets,
            h_new,
            vol_new,
            tags_new,
            removed_bnd=removed_bnd,
            added_bnd=added_bnd,
        )
        return True, new_cids

    P12 = state.X[np.asarray([tet1, tet2], dtype=np.int64)]
    cc = _tet_circumcentre(P12)
    c_pair = 0.5 * (cc[0] + cc[1])

    if mode == "1-4":
        cid = c1 if host == 0 else c2
        tag_cid = state.tags[cid]
        v0, v1, v2, v3 = state.cells[cid]
        c_tet = state.X[np.asarray([v0, v1, v2, v3], dtype=np.int64)].mean(axis=0)
        c_ref = c_pair if tag1 == tag2 else cc[host]
        pt = (1.0 - alpha) * c_ref + alpha * c_tet
        X_try = np.vstack([state.X, pt[None, :]])
        new_tets = [
            _orient_tet(X_try, v0, v1, v2, s_idx),
            _orient_tet(X_try, v0, v3, v1, s_idx),
            _orient_tet(X_try, v1, v3, v2, s_idx),
            _orient_tet(X_try, v2, v3, v0, s_idx),
        ]
        P = X_try[np.asarray(new_tets, dtype=np.int64)]
        vol_old = state.vol[cid]
        vol_new = _tet_volumes(P)
        min_v = max(state.min_vol, 1e-6 * vol_old)
        if np.all(vol_new > min_v) and abs(vol_new.sum() - vol_old) <= 1e-12 * vol_old:
            _, h_new, _, vol_new = tet_partials(P)
            state.X = X_try
            new_cids = state.replace_cells((cid,), new_tets, h_new, vol_new, tag_cid)
            return True, new_cids
        return False, "volume"

    c_face = (state.X[u] + state.X[v] + state.X[w]) / 3.0
    if tag1 != tag2:
        # Project c_pair onto the interface plane (u, v, w) to preserve per-tag volumes
        n_vec = np.cross(state.X[v] - state.X[u], state.X[w] - state.X[u])
        n_norm = float(np.linalg.norm(n_vec))
        if n_norm <= 0.0:
            return False, "volume"
        n_unit = n_vec / n_norm
        c_ref = c_pair - float(np.dot(c_pair - state.X[u], n_unit)) * n_unit
    else:
        c_ref = c_pair

    new_tets = [
        (u, v, s_idx, d),
        (v, w, s_idx, d),
        (w, u, s_idx, d),
        (u, w, s_idx, e),
        (w, v, s_idx, e),
        (v, u, s_idx, e),
    ]
    tags_new = [tag1, tag1, tag1, tag2, tag2, tag2]
    pt = (1.0 - alpha) * c_ref + alpha * c_face
    X_try = np.vstack([state.X, pt[None, :]])
    P = X_try[np.asarray(new_tets, dtype=np.int64)]
    vol_old1 = state.vol[c1]
    vol_old2 = state.vol[c2]
    vol_new = _tet_volumes(P)
    min_v1 = max(state.min_vol, 1e-6 * vol_old1)
    min_v2 = max(state.min_vol, 1e-6 * vol_old2)
    if (
        np.all(vol_new[:3] > min_v1)
        and abs(vol_new[:3].sum() - vol_old1) <= 1e-12 * vol_old1
        and np.all(vol_new[3:] > min_v2)
        and abs(vol_new[3:].sum() - vol_old2) <= 1e-12 * vol_old2
    ):
        _, h_new, _, vol_new = tet_partials(P)
        state.X = X_try
        new_cids = state.replace_cells((c1, c2), new_tets, h_new, vol_new, tags_new)
        return True, new_cids

    return False, "volume"


def _try_flips_for_face(state, f, locked, enabled_set, flips_count, refused_count):
    """Attempt enabled local flips directly on f or on the tets incident to f."""
    entries = state.face_map.get(f)
    if not entries:
        return None

    reasons = []
    a, b, c = f
    f_edges = (_sedge(a, b), _sedge(a, c), _sedge(b, c))

    # 1. Direct 2-3 flip on f
    if "2-3" in enabled_set:
        ok, res = _flip_23(state, f, locked, target_face=f, check_delaunay=True)
        if ok:
            flips_count["2-3"] += 1
            return res
        reasons.append(res)

    # 2. Direct 3-2, 4-4, 2-2 flips on edges of f
    for ed in f_edges:
        if "3-2" in enabled_set and ed not in state.bnd_edges:
            ok, res = _flip_32(state, ed, locked, target_face=f, check_delaunay=True)
            if ok:
                flips_count["3-2"] += 1
                return res
            if res in ("interface", "locked", "volume", "delaunay"):
                reasons.append(res)
        if "4-4" in enabled_set and ed not in state.bnd_edges:
            ok, res = _flip_44(state, ed, locked, target_face=f, check_delaunay=True)
            if ok:
                flips_count["4-4"] += 1
                return res
            if res in ("interface", "locked", "volume", "delaunay"):
                reasons.append(res)
        if "2-2" in enabled_set and (
            ed in state.bnd_edges or len(state.edge_map.get(ed, ())) == 4
        ):
            ok, res = _flip_22(state, ed, locked, target_face=f, check_delaunay=True)
            if ok:
                flips_count["2-2"] += 1
                return res
            if res in ("interface", "locked", "boundary", "volume", "delaunay"):
                reasons.append(res)

    # 3. Indirect single-tag flips on the other faces/edges of the tets incident to f
    #    (repairs interface faces or boundary-adjacent faces without crossing interfaces)
    inc_cids = [cid for cid, _ in entries]
    cand_faces = []
    cand_edges = []
    for cid in inc_cids:
        if cid not in state.cells:
            continue
        tet = state.cells[cid]
        for i0, i1, i2 in _LOC_F:
            g = _sface(tet[i0], tet[i1], tet[i2])
            if g != f and g not in cand_faces:
                cand_faces.append(g)
        for i0, i1 in LOC_E:
            ed = _sedge(tet[i0], tet[i1])
            if ed not in f_edges and ed not in cand_edges:
                cand_edges.append(ed)

    if "2-3" in enabled_set:
        for g in cand_faces:
            ok, res = _flip_23(state, g, locked, target_face=f, check_delaunay=True)
            if ok:
                flips_count["2-3"] += 1
                return res

    if "3-2" in enabled_set:
        for ed in cand_edges:
            if ed not in state.bnd_edges:
                ok, res = _flip_32(
                    state, ed, locked, target_face=f, check_delaunay=True
                )
                if ok:
                    flips_count["3-2"] += 1
                    return res

    if "4-4" in enabled_set:
        for ed in cand_edges:
            if ed not in state.bnd_edges:
                ok, res = _flip_44(
                    state, ed, locked, target_face=f, check_delaunay=True
                )
                if ok:
                    flips_count["4-4"] += 1
                    return res

    if "2-2" in enabled_set:
        for ed in cand_edges:
            if ed in state.bnd_edges or len(state.edge_map.get(ed, ())) == 4:
                ok, res = _flip_22(
                    state, ed, locked, target_face=f, check_delaunay=True
                )
                if ok:
                    flips_count["2-2"] += 1
                    return res

    # Record primary refusal reason for f
    for prio in ("interface", "locked", "boundary", "volume", "delaunay", "topology"):
        if prio in reasons:
            refused_count[prio] = refused_count.get(prio, 0) + 1
            break
    else:
        refused_count["topology"] = refused_count.get("topology", 0) + 1

    return None


def flip_to_delaunay(
    mesh,
    materials=None,
    locked=None,
    bc="pec",
    steiner=True,
    max_flips=None,
    verbose=False,
    enabled_flips=("2-3", "3-2", "4-4", "2-2"),
):
    """
    Topological flip repair of non-Delaunay faces for the Voronoi FIT Hodge.

    Parameters
    ----------
    mesh : SimplexMesh
        Input 3D tetrahedral mesh.
    materials : dict, optional
        ``{tag: material}`` mapping as for `UnstructuredFIT3D`.
    locked : ndarray of bool (n_nodes,), optional
        True for nodes on locked CAD curves or surfaces. A flip is refused if
        it removes an edge whose two endpoints are both locked, or a boundary
        face whose three nodes are all locked.
    bc : {"pec", "pmc", "abc"}, optional
        Outer boundary condition used to identify free edges (default ``"pec"``).
    steiner : bool, optional
        If True (default), insert a Steiner node at the circumcentre of a stuck
        interior tet pair and re-flip.
    max_flips : int, optional
        Maximum number of topological flips to perform (default: ``10 * n_cells``).
    verbose : bool, optional
        If True, print a one-line summary of flips and negative faces.
    enabled_flips : tuple of str, optional
        Subset of ``("2-3", "3-2", "4-4", "2-2")`` enabled during repair.

    Returns
    -------
    new_mesh : SimplexMesh
        Repaired tetrahedral mesh preserving total and per-tag volumes.
    report : dict
        Dictionary with keys ``"flips"``, ``"steiner_nodes"``, ``"refused"``,
        ``"negative_faces_before"``, ``"negative_faces_after"``, and ``"stuck"``.
    """
    if locked is None:
        locked_arr = np.zeros(mesh.n_nodes, dtype=bool)
    else:
        locked_arr = np.asarray(locked, dtype=bool).copy()
        if locked_arr.shape != (mesh.n_nodes,):
            raise ValueError("locked must be a boolean array of shape (n_nodes,)")

    enabled_set = set(enabled_flips)
    all_flips_enabled = enabled_set >= {"2-3", "3-2", "4-4", "2-2"}
    limit = int(10 * mesh.n_cells) if max_flips is None else int(max_flips)

    flips_count = {"2-3": 0, "3-2": 0, "4-4": 0, "2-2": 0}
    refused_count = {}
    steiner_count = 0

    state = _RepairState(mesh, materials, bc)
    neg_before = state.initial_neg_count

    if neg_before == 0 or limit <= 0:
        report = {
            "flips": flips_count,
            "steiner_nodes": 0,
            "refused": refused_count,
            "negative_faces_before": neg_before,
            "negative_faces_after": neg_before,
            "stuck": [int(i) for i in _get_neg_indices(mesh, materials, bc)]
            if neg_before
            else [],
        }
        return mesh, report

    queue = deque(f for f in state.face_map if state.is_neg_face(f))
    in_queue = set(queue)

    total_flips = 0
    while queue and total_flips < limit:
        f = queue.popleft()
        in_queue.discard(f)
        if not state.is_neg_face(f):
            continue

        new_cids = _try_flips_for_face(
            state, f, locked_arr, enabled_set, flips_count, refused_count
        )
        if new_cids is not None:
            total_flips += 1
            for nf in state.cells_faces(new_cids):
                if state.is_neg_face(nf) and nf not in in_queue:
                    queue.append(nf)
                    in_queue.add(nf)

    # Steiner fallback for stuck faces when all flip types are enabled
    if steiner and all_flips_enabled and total_flips < limit:
        for _pass in range(6):
            stuck_faces = sorted(state.neg_set)
            if not stuck_faces:
                break
            progress = False
            for f in stuck_faces:
                if total_flips >= limit or not state.is_neg_face(f):
                    continue
                cur_neg = len(state.neg_set)
                best_cand = None
                best_neg = cur_neg
                cell_splits = [
                    ("1-4", h, a)
                    for a in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
                    for h in (0, 1)
                ] + [("2-6", 0, a) for a in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)]
                edge_splits = [
                    ("edge", h, a) for a in (0.5, 0.4, 0.6) for h in (0, 1, 2)
                ] + [("bnd_edge", h, a) for a in (0.5, 0.4, 0.6) for h in (0, 1, 2)]
                if state.is_interface_face(f):
                    candidates = edge_splits + cell_splits
                else:
                    candidates = cell_splits + edge_splits
                for mode, host, alpha in candidates:
                    cp = state.checkpoint()
                    ok, nc0 = _insert_steiner(
                        state,
                        f,
                        mode=mode,
                        host=host,
                        alpha=alpha,
                        locked=locked_arr,
                    )
                    if not ok:
                        state.rollback(cp)
                        continue
                    flips_try = dict(flips_count)
                    refused_try = dict(refused_count)
                    locked_try = np.append(locked_arr, False)
                    t_flips = total_flips
                    q_try = deque(
                        nf for nf in state.cells_faces(nc0) if state.is_neg_face(nf)
                    )
                    in_q_try = set(q_try)
                    for nf in sorted(state.neg_set):
                        if nf not in in_q_try:
                            q_try.append(nf)
                            in_q_try.add(nf)
                    while q_try and t_flips < limit:
                        qf = q_try.popleft()
                        in_q_try.discard(qf)
                        if not state.is_neg_face(qf):
                            continue
                        new_cids = _try_flips_for_face(
                            state, qf, locked_try, enabled_set, flips_try, refused_try
                        )
                        if new_cids is not None:
                            t_flips += 1
                            for nf in state.cells_faces(new_cids):
                                if state.is_neg_face(nf) and nf not in in_q_try:
                                    q_try.append(nf)
                                    in_q_try.add(nf)
                    neg_try = len(state.neg_set)
                    state.rollback(cp)
                    if neg_try < best_neg:
                        best_neg = neg_try
                        best_cand = (mode, host, alpha)
                        if neg_try == 0:
                            break
                if best_cand is not None:
                    mode, host, alpha = best_cand
                    _, nc0 = _insert_steiner(
                        state,
                        f,
                        mode=mode,
                        host=host,
                        alpha=alpha,
                        locked=locked_arr,
                    )
                    locked_arr = np.append(locked_arr, False)
                    steiner_count += 1
                    progress = True
                    q_try = deque(
                        nf for nf in state.cells_faces(nc0) if state.is_neg_face(nf)
                    )
                    in_q_try = set(q_try)
                    for nf in sorted(state.neg_set):
                        if nf not in in_q_try:
                            q_try.append(nf)
                            in_q_try.add(nf)
                    while q_try and total_flips < limit:
                        qf = q_try.popleft()
                        in_q_try.discard(qf)
                        if not state.is_neg_face(qf):
                            continue
                        new_cids = _try_flips_for_face(
                            state,
                            qf,
                            locked_arr,
                            enabled_set,
                            flips_count,
                            refused_count,
                        )
                        if new_cids is not None:
                            total_flips += 1
                            for nf in state.cells_faces(new_cids):
                                if state.is_neg_face(nf) and nf not in in_q_try:
                                    q_try.append(nf)
                                    in_q_try.add(nf)
            if not progress:
                break
        state._undo_log = None

    if total_flips == 0 and steiner_count == 0:
        new_mesh = mesh
    else:
        new_mesh = state.to_mesh(mesh)

    stuck_idx = _get_neg_indices(new_mesh, materials, bc)
    report = {
        "flips": flips_count,
        "steiner_nodes": int(steiner_count),
        "refused": refused_count,
        "negative_faces_before": int(neg_before),
        "negative_faces_after": int(stuck_idx.size),
        "stuck": [int(i) for i in stuck_idx],
    }
    if verbose:
        print(
            f"flip_to_delaunay: negative faces {neg_before} -> {stuck_idx.size}, "
            f"flips {flips_count}, steiner {steiner_count}"
        )
    return new_mesh, report


def _get_neg_indices(mesh, materials, bc):
    eps_r, mu_r, _, pec, _ = cell_materials(mesh, materials)
    live = (~pec).astype(float)
    free = _default_free(mesh, pec, bc)
    return _State(mesh, eps_r, mu_r, live, free).neg_faces()
