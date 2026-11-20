# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Distributed-memory (MPI) time stepping for 3D Cartesian subgridding
(`wakis.subgrid3d`).

Active leaf cells of all octree blocks are partitioned across ``P`` ranks by
weighted recursive coordinate bisection (``method="rcb"``), weighted recursive
inertial bisection (``method="rib"``), or whole-block greedy assignment
(``method="blocks"``), with per-cell work weight ``1`` under a global step or
``prod_{m<=l} p_m`` at refinement level ``l`` under local time stepping (LTS).

Ownership rule:
* Every active leaf cell belongs to one part ``0 <= p < P``.
* Every free edge (``solver.free_edges``) is owned by the lowest part among the
  active leaf cells of its own block that contain it.
* Every active face (``solver.faces``) is owned by the lowest part among its
  ``<= 2`` active leaf cells.
* CPML auxiliary entries ``psi_d`` on ``pf[d]`` and ``phi_d`` on ``pe[d]``
  inherit the owner of their face or free edge.

Because ``K = C P_e`` already embeds the Nedelec hanging-edge trace couplings
as CSR matrix entries, each rank stores the serial CSR rows of its owned
unknowns with the identical column order and refreshes off-rank column entries
via point-to-point halo exchanges. Every owned unknown therefore undergoes the
exact same sequence of IEEE-754 floating-point operations as in the serial
`one_step`, giving bitwise equality (``np.array_equal``) for fields, probes and
running DFTs across any number of ranks and any partition.
"""

import numpy as np
import scipy.sparse as sp

from .mesh import SubgridMesh3D
from .stepper import SubgridLTSFIT3D

UNIT_ROUNDOFF = 2.0**-53

__all__ = [
    "UNIT_ROUNDOFF",
    "DistributedFIT",
    "DistributedSubgridFIT3D",
    "LocalCluster",
    "SubgridLocalCluster",
    "build_pieces",
    "partition_cells",
    "partition_stats",
]


def _check_supported(solver):
    """Validate that `solver` uses only features supported by the MPI engine."""
    if getattr(solver, "backend", "numpy") != "numpy":
        raise NotImplementedError(
            f"MPI subgrid3d supports backend='numpy' only, got {solver.backend!r}"
        )
    if getattr(solver, "kernels", "csr") != "csr":
        raise NotImplementedError(
            f"MPI subgrid3d supports kernels='csr' only, got {solver.kernels!r}"
        )
    if getattr(solver, "dispersive_groups", None):
        raise NotImplementedError(
            "MPI subgrid3d does not support dispersive media (dispersive_groups)"
        )
    cpml = getattr(solver, "cpml", None)
    n_aux = cpml.n_aux if cpml is not None else 0
    expected_n_state = int(solver.b.size + solver.e.size + n_aux)
    n_state_attr = solver.__dict__.get(
        "n_state", getattr(solver, "n_state", expected_n_state)
    )
    if int(n_state_attr) != expected_n_state:
        raise NotImplementedError(
            f"MPI subgrid3d: unsupported solver.n_state={n_state_attr} "
            f"(expected {expected_n_state} = len(b) + len(e) + CPML auxiliaries)"
        )


def _mesh_and_lts(solver_or_mesh):
    if isinstance(solver_or_mesh, SubgridMesh3D):
        return solver_or_mesh.build(), None
    return solver_or_mesh.mesh.build(), getattr(solver_or_mesh, "lts", None)


def _active_cell_geometry(solver_or_mesh, weights="work"):
    """
    Centres ``(n_active, 3)`` [m], block indices ``(n_active,)`` and per-cell
    weights ``(n_active,)`` of all active leaf cells in block order.
    """
    M, lts = _mesh_and_lts(solver_or_mesh)
    centres, blocks, w_list = [], [], []
    ratios = lts.ratios if (lts is not None and lts.L > 0) else []
    for bi, blk in enumerate(M.blocks):
        act = M.active[bi].ravel()
        n_act = int(act.sum())
        if n_act == 0:
            continue
        X, Y, Z = blk.cell_centres()
        centres.append(
            np.column_stack([X.ravel()[act], Y.ravel()[act], Z.ravel()[act]])
        )
        blocks.append(np.full(n_act, bi, dtype=np.int64))
        lv = int(M.level[bi])
        if weights == "work" and lv > 0 and ratios:
            w_bi = float(np.prod(ratios[:lv]))
        else:
            w_bi = 1.0
        w_list.append(np.full(n_act, w_bi, dtype=float))
    cent = np.concatenate(centres, axis=0)
    blk_id = np.concatenate(blocks)
    if isinstance(weights, str) or weights is None:
        if weights not in ("work", "uniform", "cells", None):
            raise ValueError(f"unknown weights specification {weights!r}")
        w = np.concatenate(w_list)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (cent.shape[0],):
            raise ValueError(f"weights must have shape ({cent.shape[0]},)")
    return M, cent, blk_id, w


def _bisect_weighted(points, weights, ids, n_parts, out, first, inertial):
    if n_parts == 1:
        out[ids] = first
        return
    P = points[ids]
    w = weights[ids]
    if inertial:
        X = P - np.average(P, axis=0, weights=w)
        _, V = np.linalg.eigh((X * w[:, None]).T @ X)
        key = X @ V[:, -1]
    else:
        key = P[:, int(np.argmax(P.max(0) - P.min(0)))]
    nl = n_parts // 2
    nr = n_parts - nl
    o = np.argsort(key, kind="stable")
    ids_s = ids[o]
    w_s = w[o]
    cw = np.cumsum(w_s)
    target = cw[-1] * (nl / n_parts)
    k = int(np.argmin(np.abs(cw[:-1] - target))) + 1
    k = max(nl, min(ids.size - nr, k))
    _bisect_weighted(points, weights, ids_s[:k], nl, out, first, inertial)
    _bisect_weighted(points, weights, ids_s[k:], nr, out, first + nl, inertial)


def partition_cells(solver, n_parts, method="rcb", weights="work"):
    """
    Partition the active leaf cells of all blocks into `n_parts` parts.

    Parameters
    ----------
    solver : SubgridFIT3D or SubgridLTSFIT3D or SubgridMesh3D
        Configured subgrid solver (or mesh).
    n_parts : int
        Number of MPI parts (ranks).
    method : {"rcb", "rib", "blocks"}
        ``"rcb"``: weighted recursive coordinate bisection on cell centres;
        ``"rib"``: weighted recursive inertial bisection on cell centres;
        ``"blocks"``: whole blocks per rank, assigned greedily by descending
        block work weight.
    weights : {"work", "uniform"} or ndarray
        ``"work"`` weights each active cell of level ``l`` by ``1`` under a
        global step and by ``prod_{m<=l} p_m`` under LTS.

    Returns
    -------
    ndarray of int64, shape (n_active_cells,)
        Part index ``0 <= p < n_parts`` of every active leaf cell.
    """
    n_parts = int(n_parts)
    M, cent, blk_id, w = _active_cell_geometry(solver, weights=weights)
    n_active = cent.shape[0]
    if n_parts < 1 or n_parts > n_active:
        raise ValueError(f"n_parts must be in [1, {n_active}]")
    if n_parts == 1:
        return np.zeros(n_active, dtype=np.int64)
    if method == "auto":
        method = "rcb"
    parts = np.empty(n_active, dtype=np.int64)
    if method in ("rcb", "rib"):
        _bisect_weighted(
            cent, w, np.arange(n_active), n_parts, parts, 0, method == "rib"
        )
    elif method == "blocks":
        block_items = []
        for bi in range(M.n_blocks):
            idx = np.flatnonzero(blk_id == bi)
            if idx.size > 0:
                block_items.append((float(w[idx].sum()), bi, idx))
        # Sort by descending block weight, tie-breaking by block index
        block_items.sort(key=lambda item: (-item[0], item[1]))
        load = np.zeros(n_parts, dtype=float)
        for w_bi, _, idx in block_items:
            r = int(np.argmin(load))
            parts[idx] = r
            load[r] += w_bi
    else:
        raise ValueError(f"unknown partition method {method!r}")
    return parts


def _normalize_parts(solver, parts, n_parts=None):
    """Accept active-cell `parts` (n_active,) or all-cell `parts` (n_cells,)."""
    M, _ = _mesh_and_lts(solver)
    act = np.concatenate([a.ravel() for a in M.active])
    n_active = int(act.sum())
    parts = np.asarray(parts, dtype=np.int64)
    if parts.shape == (M.n_cells,) and M.n_cells != n_active:
        parts = parts[act]
    if parts.shape != (n_active,):
        raise ValueError(
            f"parts must have length {n_active} (active cells), got {parts.shape}"
        )
    if n_parts is None:
        n_parts = int(parts.max()) + 1
    if np.any((parts < 0) | (parts >= n_parts)):
        raise ValueError("parts entries must lie in [0, n_parts)")
    return parts, int(n_parts)


def _entity_owners(solver, parts, n_parts):
    """
    Deterministic owner of every free edge and active face.

    Free edge owner: lowest part among the active leaf cells of its own block
    that contain it.
    Face owner (faces in ``solver.faces``): lowest part among its ``<= 2``
    active leaf cells.
    """
    M = solver.mesh
    part_all = np.full(M.n_cells, n_parts, dtype=np.int64)
    part_all[solver.active_cells] = parts
    own_e_all = np.full(M.n_edges, n_parts, dtype=np.int64)
    own_f_all = np.full(M.n_faces, n_parts, dtype=np.int64)
    for bi, blk in enumerate(M.blocks):
        act = M.active[bi].ravel()
        c0, e0, f0 = M.cell_off[bi], M.edge_off[bi], M.face_off[bi]
        c_e, e_loc, _ = blk.cell_edges()
        ok_e = act[c_e]
        np.minimum.at(own_e_all, e0 + e_loc[ok_e], part_all[c0 + c_e[ok_e]])
        c_f, f_loc, _ = blk.cell_faces()
        ok_f = act[c_f]
        np.minimum.at(own_f_all, f0 + f_loc[ok_f], part_all[c0 + c_f[ok_f]])
    owner_e = own_e_all[solver.free_edges]
    owner_f = own_f_all[solver.faces]
    if np.any(owner_e >= n_parts) or np.any(owner_f >= n_parts):
        raise RuntimeError("unowned free edge or active face (internal error)")
    return owner_e, owner_f


def partition_stats(solver, parts=None, n_parts=None, method="rcb", weights="work"):
    """
    Cell counts, work weights, imbalance and cut statistics of a partition.

    Parameters
    ----------
    solver : SubgridFIT3D or SubgridLTSFIT3D
    parts : ndarray of int or int, optional
        Active-cell partition array, or `n_parts` if an integer is passed.
    n_parts : int, optional
        Number of parts (inferred from `parts` if omitted).
    method : {"rcb", "rib", "blocks"}
        Partition method used when `parts` is not an array.
    weights : {"work", "uniform"} or ndarray
        Per-cell weighting for work imbalance.

    Returns
    -------
    dict
        ``n_parts``, ``active_cells``, ``cells`` (per part), ``work`` (per
        part), ``imbalance`` (weighted max/mean), ``work_imbalance``,
        ``cell_imbalance``, ``owned_edges`` (per part), ``owned_faces`` (per
        part), and ``cut_faces``.
    """
    if isinstance(parts, (int, np.integer)) and n_parts is None:
        n_parts = int(parts)
        parts = None
    if parts is None:
        if n_parts is None:
            raise ValueError("must provide parts or n_parts")
        parts = partition_cells(solver, n_parts, method=method, weights=weights)
    parts, n_parts = _normalize_parts(solver, parts, n_parts=n_parts)
    M, _, _, w = _active_cell_geometry(solver, weights=weights)
    cells = np.bincount(parts, minlength=n_parts)
    work = np.bincount(parts, weights=w, minlength=n_parts)
    work_imb = float(work.max() / work.mean()) if work.sum() > 0 else 1.0
    cell_imb = float(cells.max() / cells.mean()) if cells.sum() > 0 else 1.0

    # Cut faces inside blocks (shared by two active cells in different parts)
    part_all = np.full(M.n_cells, -1, dtype=np.int64)
    part_all[solver.active_cells] = parts
    f_min = np.full(M.n_faces, n_parts, dtype=np.int64)
    f_max = np.full(M.n_faces, -1, dtype=np.int64)
    for bi, blk in enumerate(M.blocks):
        act = M.active[bi].ravel()
        c0, f0 = M.cell_off[bi], M.face_off[bi]
        c_f, f_loc, _ = blk.cell_faces()
        ok = act[c_f]
        p_c = part_all[c0 + c_f[ok]]
        np.minimum.at(f_min, f0 + f_loc[ok], p_c)
        np.maximum.at(f_max, f0 + f_loc[ok], p_c)
    cut_faces = int(np.sum((f_max >= 0) & (f_min < n_parts) & (f_min != f_max)))

    out = {
        "n_parts": n_parts,
        "active_cells": int(parts.size),
        "cells": cells.tolist(),
        "work": work.tolist(),
        "imbalance": work_imb,
        "work_imbalance": work_imb,
        "cell_imbalance": cell_imb,
        "cut_faces": cut_faces,
    }
    if hasattr(solver, "free_edges") and hasattr(solver, "faces"):
        owner_e, owner_f = _entity_owners(solver, parts, n_parts)
        out["owned_edges"] = np.bincount(owner_e, minlength=n_parts).tolist()
        out["owned_faces"] = np.bincount(owner_f, minlength=n_parts).tolist()
    return out


# ================================================================ halo spaces
class _Space:
    """
    Layout of one distributed vector on one rank: ``[owned | halo]``.

    ``send[q]`` are positions in the owned segment sent to rank ``q``;
    ``recv[q]`` are positions in the halo segment received from rank ``q``
    (in matching global index order).
    """

    def __init__(self, owned, halo, send, recv, tag):
        self.owned = np.asarray(owned, dtype=np.int64)
        self.halo = np.asarray(halo, dtype=np.int64)
        self.send = send
        self.recv = recv
        self.tag = int(tag)
        self.n_own = int(self.owned.size)
        self.n = int(self.owned.size + self.halo.size)
        self.dropped = []

    def lookup(self, n_global):
        loc = np.full(n_global, -1, dtype=np.int64)
        loc[self.owned] = np.arange(self.n_own, dtype=np.int64)
        loc[self.halo] = self.n_own + np.arange(self.halo.size, dtype=np.int64)
        return loc

    @property
    def halo_size(self):
        return int(sum(v.size for v in self.recv.values()))

    @property
    def n_messages(self):
        return len(self.send)

    @property
    def neighbours(self):
        return sorted(set(self.send) | set(self.recv))


def _build_spaces(owner, needs, n_parts, tag):
    """Build `_Space` instances for all `n_parts` ranks."""
    owned = [np.flatnonzero(owner == r) for r in range(n_parts)]
    pos = np.empty(owner.size, dtype=np.int64)
    for o in owned:
        pos[o] = np.arange(o.size, dtype=np.int64)
    halos, recvs = [], []
    sends = [{} for _ in range(n_parts)]
    for r in range(n_parts):
        nd = np.unique(needs[r]) if len(needs[r]) else np.zeros(0, dtype=np.int64)
        h = np.setdiff1d(nd, owned[r], assume_unique=True)
        oh = owner[h]
        rv = {}
        for q in np.unique(oh):
            k = np.flatnonzero(oh == q)
            rv[int(q)] = k
            sends[int(q)][r] = pos[h[k]]
        halos.append(h)
        recvs.append(rv)
    return [_Space(owned[r], halos[r], sends[r], recvs[r], tag) for r in range(n_parts)]


def _drop_halo_entry(spaces):
    """
    Verification mutation: drop one halo entry on the first rank (preferring
    rank >= 1) that has a non-empty halo.
    """
    candidates = [r for r in range(1, len(spaces)) if spaces[r].recv]
    if not candidates:
        candidates = [r for r in range(len(spaces)) if spaces[r].recv]
    if not candidates:
        return
    rank = candidates[0]
    sp_r = spaces[rank]
    q = min(sp_r.recv)
    k = sp_r.recv[q]
    sp_r.dropped.append(int(sp_r.n_own + k[0]))
    if k.size == 1:
        del sp_r.recv[q]
        del spaces[q].send[rank]
    else:
        sp_r.recv[q] = k[1:]
        spaces[q].send[rank] = spaces[q].send[rank][1:]


def _ref_csr(A):
    """
    CSR matrix with the exact per-row entry order of SciPy's serial SpMV `A @ x`.

    When `A` is already CSR, SciPy sums each row in stored order. When `A` is
    CSC (such as `M.T` of a CSR matrix `M`), SciPy accumulates each output row
    in increasing column order, which equals the sorted-indices CSR row.
    """
    if sp.isspmatrix_csr(A) or isinstance(A, sp.csr_array):
        return sp.csr_matrix(A)
    A = sp.csr_matrix(A.tocsr())
    A.sort_indices()
    return A


def _cols(A, rows):
    if rows.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.unique(A[rows].indices).astype(np.int64)


def _local(A, rows, lookup, n_loc):
    """Extract `rows` of CSR `A` and renumber columns via `lookup` (order kept)."""
    if rows.size == 0:
        return sp.csr_matrix((0, n_loc), dtype=A.dtype)
    sub = A[rows]
    idx = lookup[sub.indices]
    if np.any(idx < 0):
        raise RuntimeError("halo does not cover all operator columns")
    return sp.csr_matrix(
        (sub.data.copy(), idx.astype(np.int64), sub.indptr.astype(np.int64)),
        shape=(rows.size, n_loc),
        dtype=A.dtype,
    )


def _expand_cols(M_csr, col_map, n_cols_full):
    """
    Embed a CSR matrix `M_csr` of shape ``(m, len(col_map))`` into shape
    ``(m, n_cols_full)`` with strictly increasing `col_map`, preserving row
    entry order.
    """
    M_csr = _ref_csr(M_csr)
    return sp.csr_matrix(
        (
            M_csr.data.copy(),
            col_map[M_csr.indices].astype(np.int64),
            M_csr.indptr.astype(np.int64),
        ),
        shape=(M_csr.shape[0], n_cols_full),
        dtype=M_csr.dtype,
    )


# ================================================================ build pieces
def build_pieces(
    solver,
    n_parts,
    parts=None,
    method="rcb",
    weights="work",
    ranks=None,
    _mutation=None,
):
    """
    Build per-rank operators, halo spaces, initial state and monitors.

    Parameters
    ----------
    solver : SubgridFIT3D or SubgridLTSFIT3D
        Configured serial solver.
    n_parts : int
        Number of parts (ranks).
    parts : ndarray, optional
        Active-cell partition (default: `partition_cells`).
    method : {"rcb", "rib", "blocks"}
        Partition method when `parts` is None.
    weights : {"work", "uniform"} or ndarray
        Per-cell work weights when `parts` is None.
    ranks : sequence of int, optional
        Build local pieces only for these ranks (halo spaces are built for all).
    _mutation : {None, "drop_halo", "skip_lts_exchange"}
        Verification-only mutation hooks.

    Returns
    -------
    pieces : list of dict
        Per-rank piece dictionaries (`None` for ranks not in `ranks`).
    parts : ndarray of int64
        Active-cell partition.
    """
    if _mutation not in (None, "drop_halo", "skip_lts_exchange"):
        raise ValueError(f"unknown _mutation {_mutation!r}")
    _check_supported(solver)
    n_parts = int(n_parts)
    if parts is None:
        parts = partition_cells(solver, n_parts, method=method, weights=weights)
    else:
        parts, n_parts = _normalize_parts(solver, parts, n_parts=n_parts)
    ranks = list(range(n_parts)) if ranks is None else [int(r) for r in ranks]

    owner_e, owner_f = _entity_owners(solver, parts, n_parts)
    own_e = [np.flatnonzero(owner_e == r) for r in range(n_parts)]
    own_f = [np.flatnonzero(owner_f == r) for r in range(n_parts)]

    K = _ref_csr(solver.K)
    KH = _ref_csr(solver.KH)
    KT = _ref_csr(solver.K.T)
    KHT = _ref_csr(solver.KH.T)

    cpml = getattr(solver, "cpml", None)
    pml_Kf, pml_Ke, pml_KfT, pml_KeT = [], [], [], []
    if cpml is not None:
        for d in range(3):
            Kf_d = _ref_csr(cpml.Kf[d])
            Ke_d = _ref_csr(cpml.Ke[d])
            pml_Kf.append(Kf_d)
            pml_Ke.append(Ke_d)
            pml_KfT.append(_expand_cols(_ref_csr(Kf_d.T), cpml.pf[d], solver.n_faces))
            pml_KeT.append(_expand_cols(_ref_csr(Ke_d.T), cpml.pe[d], solver.n_free))

    # Probes: owned by lowest owner among the free edges in the probe row
    probe_owners = []
    probe_rows = []
    for p in solver.probes:
        row = _ref_csr(p["row"])
        cols = row.indices
        po = int(owner_e[cols].min()) if cols.size else 0
        probe_owners.append(po)
        probe_rows.append(row)

    needs_e, needs_b = [], []
    for r in range(n_parts):
        ce = [_cols(K, own_f[r]), _cols(KHT, own_f[r])]
        cb = [_cols(KH, own_e[r]), _cols(KT, own_e[r])]
        if cpml is not None:
            for d in range(3):
                sel_f = np.flatnonzero(owner_f[cpml.pf[d]] == r)
                sel_e = np.flatnonzero(owner_e[cpml.pe[d]] == r)
                ce.append(_cols(pml_Kf[d], sel_f))
                ce.append(_cols(pml_KeT[d], own_f[r]))
                cb.append(_cols(pml_Ke[d], sel_e))
                cb.append(_cols(pml_KfT[d], own_e[r]))
        for row, po in zip(probe_rows, probe_owners):
            if po == r:
                ce.append(row.indices.astype(np.int64))
        needs_e.append(np.unique(np.concatenate(ce)))
        needs_b.append(np.unique(np.concatenate(cb)))

    sp_e = _build_spaces(owner_e, needs_e, n_parts, tag=10)
    sp_b = _build_spaces(owner_f, needs_b, n_parts, tag=11)

    # Probe-only halo space (used at end of LTS step only if a probe crosses ranks)
    needs_p = []
    for r in range(n_parts):
        cp = [np.zeros(0, dtype=np.int64)]
        for row, po in zip(probe_rows, probe_owners):
            if po == r:
                cp.append(row.indices.astype(np.int64))
        needs_p.append(np.unique(np.concatenate(cp)))
    sp_p = _build_spaces(owner_e, needs_p, n_parts, tag=12)
    has_probe_halo = any(s.halo_size > 0 for s in sp_p)

    # Multi-level LTS spaces and level operators
    lts = getattr(solver, "lts", None)
    lts_L = int(lts.L) if (lts is not None and lts.L > 0) else 0
    lts_spaces_fc = {}
    lts_spaces_phi = {}
    lts_level_meta = {}
    if lts_L > 0:
        for lv in range(1, lts_L + 1):
            d = lts.levels[lv]
            nF, nC, nPhi = int(d.F.size), int(d.C.size), int(d.Phi.size)
            own_FC = np.concatenate([owner_e[d.F], owner_e[d.C]])
            own_Phi = owner_f[d.Phi]
            own_F_r = [np.flatnonzero(owner_e[d.F] == r) for r in range(n_parts)]
            own_C_r = [np.flatnonzero(owner_e[d.C] == r) for r in range(n_parts)]
            own_Phi_r = [np.flatnonzero(own_Phi == r) for r in range(n_parts)]

            KF = _ref_csr(d.K_F)
            KC = _ref_csr(d.K_C)
            KHF = _ref_csr(d.KH_F)
            KFT = _ref_csr(d.K_F.T)
            KCT = _ref_csr(d.K_C.T)
            KHFT = _ref_csr(d.KH_F.T)

            needs_fc, needs_phi = [], []
            for r in range(n_parts):
                c_fc = [
                    _cols(KF, own_Phi_r[r]),
                    _cols(KHFT, own_Phi_r[r]),
                    nF + _cols(KC, own_Phi_r[r]),
                ]
                needs_fc.append(np.unique(np.concatenate(c_fc)))
                c_phi = [
                    _cols(KHF, own_F_r[r]),
                    _cols(KFT, own_F_r[r]),
                    _cols(KCT, own_C_r[r]),
                ]
                needs_phi.append(np.unique(np.concatenate(c_phi)))

            lts_spaces_fc[lv] = _build_spaces(
                own_FC, needs_fc, n_parts, tag=20 + 2 * lv
            )
            lts_spaces_phi[lv] = _build_spaces(
                own_Phi, needs_phi, n_parts, tag=21 + 2 * lv
            )
            lts_level_meta[lv] = {
                "d": d,
                "nF": nF,
                "nC": nC,
                "nPhi": nPhi,
                "own_F_r": own_F_r,
                "own_C_r": own_C_r,
                "own_Phi_r": own_Phi_r,
                "KF": KF,
                "KC": KC,
                "KHF": KHF,
                "KFT": KFT,
                "KCT": KCT,
                "KHFT": KHFT,
            }

    if _mutation == "drop_halo":
        _drop_halo_entry(sp_e)

    e_host = solver.e_host() if hasattr(solver, "e_host") else np.asarray(solver.e)
    b_host = solver.b_faces() if hasattr(solver, "b_faces") else np.asarray(solver.b)
    is_lts_cls = isinstance(solver, SubgridLTSFIT3D)

    out = [None] * n_parts
    for r in ranks:
        se, sb, spp = sp_e[r], sp_b[r], sp_p[r]
        le = se.lookup(solver.n_free)
        lb = sb.lookup(solver.n_faces)
        pos_own_e = np.full(solver.n_free, -1, dtype=np.int64)
        pos_own_e[se.owned] = np.arange(se.n_own, dtype=np.int64)
        pos_own_f = np.full(solver.n_faces, -1, dtype=np.int64)
        pos_own_f[sb.owned] = np.arange(sb.n_own, dtype=np.int64)

        spaces_r = {"e": se, "b": sb}
        if has_probe_halo:
            spaces_r["p"] = spp

        pc = {
            "rank": r,
            "n_parts": n_parts,
            "parts": parts,
            "dt": float(solver.dt),
            "n": int(solver.n),
            "dtype": solver.dtype,
            "n_free": int(solver.n_free),
            "n_faces": int(solver.n_faces),
            "is_lts_cls": is_lts_cls,
            "has_lts": (lts_L > 0),
            "lts_L": lts_L,
            "has_cpml": (cpml is not None),
            "has_probe_halo": has_probe_halo,
            "mutation": _mutation,
            "spaces": spaces_r,
            "K": _local(K, sb.owned, le, se.n),
            "KH": _local(KH, se.owned, lb, sb.n),
            "KT": _local(KT, se.owned, lb, sb.n),
            "KHT": _local(KHT, sb.owned, le, se.n),
            "W_own": np.asarray(solver.W)[sb.owned].copy(),
            "W_ext": np.asarray(solver.W)[np.r_[sb.owned, sb.halo]].copy(),
            "D_own": np.asarray(solver.D)[se.owned].copy(),
            "Dinv_own": np.asarray(solver.Dinv)[se.owned].copy(),
            "e0": np.asarray(e_host, dtype=solver.dtype)[
                np.r_[se.owned, se.halo]
            ].copy(),
            "b0": np.asarray(b_host, dtype=solver.dtype)[
                np.r_[sb.owned, sb.halo]
            ].copy(),
        }
        for p_drop in se.dropped:
            pc["e0"][p_drop] = 0

        # CPML local data
        if cpml is not None:
            pml_loc = []
            for d in range(3):
                pf_d = cpml.pf[d]
                pe_d = cpml.pe[d]
                sel_f = np.flatnonzero(owner_f[pf_d] == r)
                sel_e = np.flatnonzero(owner_e[pe_d] == r)
                pf_own_glob = pf_d[sel_f]
                pe_own_glob = pe_d[sel_e]
                psi0 = np.asarray(solver.psi[d], dtype=solver.dtype)[sel_f].copy()
                phi0 = np.asarray(solver.phi[d], dtype=solver.dtype)[sel_e].copy()
                pml_loc.append(
                    {
                        "sel_f": sel_f,
                        "sel_e": sel_e,
                        "n_pf_glob": int(pf_d.size),
                        "n_pe_glob": int(pe_d.size),
                        "loc_pf": pos_own_f[pf_own_glob],
                        "loc_pe": pos_own_e[pe_own_glob],
                        "loc_pf_in_b": lb[pf_own_glob],
                        "loc_pe_in_e": le[pe_own_glob],
                        "bh": np.asarray(cpml.bh[d])[sel_f].copy(),
                        "ah": np.asarray(cpml.ah[d])[sel_f].copy(),
                        "be": np.asarray(cpml.be[d])[sel_e].copy(),
                        "ae": np.asarray(cpml.ae[d])[sel_e].copy(),
                        "Kf": _local(pml_Kf[d], sel_f, le, se.n),
                        "Ke": _local(pml_Ke[d], sel_e, lb, sb.n),
                        "KfT": _local(pml_KfT[d], se.owned, lb, sb.n),
                        "KeT": _local(pml_KeT[d], sb.owned, le, se.n),
                        "psi0": psi0,
                        "phi0": phi0,
                    }
                )
            pc["pml"] = pml_loc
        else:
            pc["pml"] = None

        # Multi-level LTS local data
        if lts_L > 0:
            lts_loc = [None]
            prev_owned_edges = se.owned
            for lv in range(1, lts_L + 1):
                meta = lts_level_meta[lv]
                d = meta["d"]
                nF, nC, nPhi = meta["nF"], meta["nC"], meta["nPhi"]
                sfc = lts_spaces_fc[lv][r]
                sphi = lts_spaces_phi[lv][r]
                spaces_r[f"fc{lv}"] = sfc
                spaces_r[f"phi{lv}"] = sphi

                own_F = meta["own_F_r"][r]
                own_C = meta["own_C_r"][r]
                own_Phi = meta["own_Phi_r"][r]

                # In sfc, owned entries are [own_F, nF + own_C] in increasing order
                l_fc = sfc.lookup(nF + nC)
                l_F = l_fc[:nF]
                l_C = l_fc[nF:]
                l_phi = sphi.lookup(nPhi)

                # Local indices of owned F_lv and C_lv inside rank r's owned F_{lv-1}
                pos_prev = np.full(solver.n_free, -1, dtype=np.int64)
                pos_prev[prev_owned_edges] = np.arange(
                    prev_owned_edges.size, dtype=np.int64
                )
                iF_loc = pos_prev[d.F[own_F]]
                iC_loc = pos_prev[d.C[own_C]]
                if np.any(iF_loc < 0) or np.any(iC_loc < 0):
                    raise RuntimeError("LTS local nesting mismatch (internal error)")

                lts_loc.append(
                    {
                        "lv": lv,
                        "p": int(d.p),
                        "tau2": float(d.tau2),
                        "delta": float(d.delta),
                        "beta": float(d.beta),
                        "T": np.asarray(d.T, dtype=float).copy(),
                        "kappa": np.asarray(d.kappa, dtype=float).copy(),
                        "out": float(d.out),
                        "n_F_own": int(own_F.size),
                        "n_C_own": int(own_C.size),
                        "n_Phi_own": int(own_Phi.size),
                        "iF": iF_loc,
                        "iC": iC_loc,
                        "W_own": np.asarray(d.W)[own_Phi].copy(),
                        "Dinv_own": np.asarray(d.Dinv)[own_F].copy(),
                        "K_F": _local(meta["KF"], own_Phi, l_F, sfc.n),
                        "K_C": _local(meta["KC"], own_Phi, l_C, sfc.n),
                        "KH_F": _local(meta["KHF"], own_F, l_phi, sphi.n),
                        "K_F_T": _local(meta["KFT"], own_F, l_phi, sphi.n),
                        "K_C_T": _local(meta["KCT"], own_C, l_phi, sphi.n),
                        "KH_F_T": _local(meta["KHFT"], own_Phi, l_F, sfc.n),
                    }
                )
                prev_owned_edges = d.F[own_F]
            pc["lts"] = lts_loc
        else:
            pc["lts"] = None

        # Sources (restricted to owned free edges, preserving serial source order)
        srcs_loc = []
        for src in solver.sources:
            row_idx = src["row"]
            sel = owner_e[row_idx] == r
            srcs_loc.append(
                {
                    "loc_row": pos_own_e[row_idx[sel]],
                    "w": np.asarray(src["w"][sel], dtype=solver.dtype).copy(),
                    "f": src["f"],
                    "amp": float(src["amp"]),
                }
            )
        pc["has_sources"] = bool(solver.sources)
        pc["sources"] = srcs_loc

        # Probes (each probe is evaluated on its single owner rank in serial order)
        probes_loc = []
        lp = spp.lookup(solver.n_free) if has_probe_halo else le
        for i, (p, row, po) in enumerate(zip(solver.probes, probe_rows, probe_owners)):
            if po == r:
                loc_cols = lp[row.indices]
                if np.any(loc_cols < 0):
                    raise RuntimeError("probe halo missing column")
                row_loc = sp.csr_matrix(
                    (
                        row.data.copy(),
                        loc_cols.astype(np.int64),
                        row.indptr.astype(np.int64),
                    ),
                    shape=(1, spp.n if has_probe_halo else se.n),
                    dtype=row.dtype,
                )
                freqs = (
                    None if p["freqs"] is None else np.asarray(p["freqs"], float).copy()
                )
                dft0 = (
                    None if p["dft"] is None else np.asarray(p["dft"], complex).copy()
                )
                vals0 = list(p["values"])
            else:
                row_loc = None
                loc_cols = None
                freqs = (
                    None if p["freqs"] is None else np.asarray(p["freqs"], float).copy()
                )
                dft0 = None
                vals0 = []
            probes_loc.append(
                {
                    "owner": po,
                    "owned": (po == r),
                    "row_data": row.data.copy() if po == r else None,
                    "loc_cols": loc_cols,
                    "row_loc": row_loc,
                    "L": float(p["L"]),
                    "freqs": freqs,
                    "dft0": dft0,
                    "vals0": vals0,
                }
            )
        pc["probes"] = probes_loc
        out[r] = pc

    return out, parts


# ================================================================ rank engine
class _SubgridRank:
    """Per-rank state and local arithmetic (communication is driven externally)."""

    def __init__(self, pc):
        self.pc = pc
        self.rank = pc["rank"]
        self.dt = pc["dt"]
        self.n = pc["n"]
        self.t = self.n * self.dt
        self.dtype = pc["dtype"]
        self.spaces = pc["spaces"]
        self.se = self.spaces["e"]
        self.sb = self.spaces["b"]
        self._ne = self.se.n_own
        self._nb = self.sb.n_own
        self.use_sources = True

        self.buf = {
            "e": pc["e0"].copy(),
            "b": pc["b0"].copy(),
            "z": np.zeros(self.se.n, dtype=self.dtype),
            "ge_T": np.zeros(self.se.n, dtype=self.dtype),
            "b1bar_T": np.zeros(self.sb.n, dtype=self.dtype),
        }
        if pc["has_probe_halo"]:
            self.buf["p"] = np.zeros(self.spaces["p"].n, dtype=self.dtype)
        if pc["has_cpml"]:
            self.psi = [p["psi0"].copy() for p in pc["pml"]]
            self.phi = [p["phi0"].copy() for p in pc["pml"]]
            for d in range(3):
                self.buf[f"ae_T_{d}"] = np.zeros(self.se.n, dtype=self.dtype)
                self.buf[f"ah_T_{d}"] = np.zeros(self.sb.n, dtype=self.dtype)
        else:
            self.psi = []
            self.phi = []

        if pc["has_lts"]:
            for lv in range(1, pc["lts_L"] + 1):
                self.buf[f"fc{lv}"] = np.zeros(
                    self.spaces[f"fc{lv}"].n, dtype=self.dtype
                )
                self.buf[f"phi{lv}"] = np.zeros(
                    self.spaces[f"phi{lv}"].n, dtype=self.dtype
                )

        self.probe_values = [list(p["vals0"]) for p in pc["probes"]]
        self.probe_dft = [
            None if p["dft0"] is None else p["dft0"].copy() for p in pc["probes"]
        ]

        self.program = self._build_program()
        self.program_T = self._build_program_T()

    # ------------------------------------------------- step program builders
    def _build_program(self):
        prog = []
        if self.pc["has_lts"]:
            prog.append(self._lts_begin)
            prog.extend(self._lts_prog(0))
            prog.append(self._lts_end_to_z)
            prog.append(("e", "z"))
            prog.append(self._faraday_from_z)
        else:
            prog.append(self._faraday_from_e)
        prog.append(("b", "b"))
        prog.append(self._ampere)
        if self.pc["has_lts"]:
            if self.pc["has_probe_halo"]:
                prog.append(self._load_probe_buf)
                prog.append(("p", "p"))
        else:
            prog.append(("e", "e"))
        prog.append(self._record_probes)
        return prog

    def _lts_prog(self, lv):
        """Unrolled recursive program for `_R(lv, ...)`."""
        if lv >= self.pc["lts_L"]:
            return []
        m = lv + 1
        d = self.pc["lts"][m]
        ops = [lambda m=m: self._corr_begin(m)]
        for k in range(1, d["p"]):
            if m < self.pc["lts_L"]:
                ops.append(lambda m=m: self._push_sub_r(m))
                ops.extend(self._lts_prog(m))
                ops.append(lambda m=m: self._pop_sub_r(m))
            else:
                ops.append(lambda m=m: self._leaf_sub_r(m))
            ops.append(lambda m=m: self._load_fc(m))
            if not (self.pc["mutation"] == "skip_lts_exchange" and m == 1):
                ops.append((f"fc{m}", f"fc{m}"))
            ops.append(lambda m=m, k=k: self._eval_phi(m, k))
            if not (self.pc["mutation"] == "skip_lts_exchange" and m == 1):
                ops.append((f"phi{m}", f"phi{m}"))
            ops.append(lambda m=m, k=k: self._eval_cheb(m, k))
        ops.append(lambda m=m: self._corr_finish_and_push_tail(m))
        ops.extend(self._lts_prog(m))
        ops.append(lambda m=m: self._pop_tail(m))
        return ops

    # ---------------------------------------------------- LTS runtime state
    def _lts_begin(self):
        L = self.pc["lts_L"]
        self._r_vec = [None] * (L + 1)
        self._r_vec[0] = self.buf["e"][: self._ne].copy()
        self._c_state = [None] * (L + 1)

    def _corr_begin(self, m):
        d = self.pc["lts"][m]
        v_parent = self._r_vec[m - 1]
        yF = v_parent[d["iF"]].copy()
        yC = v_parent[d["iC"]].copy()
        self._c_state[m] = {
            "yF": yF,
            "yC": yC,
            "d_old": np.zeros_like(yF),
            "dk": yF.copy(),
            "z": None,
            "g": None,
        }

    def _push_sub_r(self, m):
        self._r_vec[m] = self._c_state[m]["dk"].copy()

    def _pop_sub_r(self, m):
        self._c_state[m]["z"] = self._r_vec[m]

    def _leaf_sub_r(self, m):
        self._c_state[m]["z"] = self._c_state[m]["dk"]

    def _load_fc(self, m):
        d = self.pc["lts"][m]
        st = self._c_state[m]
        nF, nC = d["n_F_own"], d["n_C_own"]
        buf = self.buf[f"fc{m}"]
        buf[:nF] = st["z"]
        buf[nF : nF + nC] = st["yC"]

    def _eval_phi(self, m, k):
        d = self.pc["lts"][m]
        st = self._c_state[m]
        buf_fc = self.buf[f"fc{m}"]
        if k == 1:
            st["g"] = d["K_C"] @ buf_fc
        w_phi = d["W_own"] * (d["K_F"] @ buf_fc + d["kappa"][k] * st["g"])
        self.buf[f"phi{m}"][: d["n_Phi_own"]] = w_phi

    def _eval_cheb(self, m, k):
        d = self.pc["lts"][m]
        st = self._c_state[m]
        buf_phi = self.buf[f"phi{m}"]
        Xd = d["tau2"] * d["Dinv_own"] * (d["KH_F"] @ buf_phi)
        d_new = (
            2 * d["delta"] * st["dk"]
            - 2 * d["beta"] * Xd
            + 2 * d["T"][k] * st["yF"]
            - st["d_old"]
        )
        st["d_old"], st["dk"] = st["dk"], d_new

    def _corr_finish_and_push_tail(self, m):
        d = self.pc["lts"][m]
        st = self._c_state[m]
        corr = (
            np.zeros_like(st["yF"]) if d["p"] == 1 else (d["out"] * st["dk"] - st["yF"])
        )
        self._r_vec[m] = st["yF"] + corr

    def _pop_tail(self, m):
        d = self.pc["lts"][m]
        self._r_vec[m - 1][d["iF"]] = self._r_vec[m]

    def _lts_end_to_z(self):
        self.buf["z"][: self._ne] = self._r_vec[0]

    # ---------------------------------------------------- Faraday & Ampere
    def _faraday_with(self, z_ext):
        pc, dt = self.pc, self.dt
        Kz = pc["K"] @ z_ext
        if pc["has_cpml"]:
            for d, p in enumerate(pc["pml"]):
                self.psi[d] = p["bh"] * self.psi[d] + p["ah"] * (p["Kf"] @ z_ext)
                Kz[p["loc_pf"]] += self.psi[d]
        self.buf["b"][: self._nb] -= dt * Kz

    def _faraday_from_e(self):
        for p_drop in self.se.dropped:
            self.buf["e"][p_drop] = 0
        self._faraday_with(self.buf["e"])

    def _faraday_from_z(self):
        for p_drop in self.se.dropped:
            self.buf["z"][p_drop] = 0
        self._faraday_with(self.buf["z"])

    def _source_own(self, t):
        if not self.use_sources or not self.pc["has_sources"]:
            return None
        s = np.zeros(self._ne, dtype=self.dtype)
        for src in self.pc["sources"]:
            s[src["loc_row"]] += src["amp"] * src["f"](t) * src["w"]
        return s

    def _ampere(self):
        pc, dt = self.pc, self.dt
        y_ext = pc["W_ext"] * self.buf["b"]
        rhs = pc["KH"] @ y_ext
        if pc["has_cpml"]:
            for d, p in enumerate(pc["pml"]):
                self.phi[d] = p["be"] * self.phi[d] + p["ae"] * (p["Ke"] @ y_ext)
                rhs[p["loc_pe"]] += self.phi[d]
        s = self._source_own(self.t + 0.5 * dt)
        if s is not None:
            rhs -= s
        self.buf["e"][: self._ne] += dt * pc["Dinv_own"] * rhs
        for p_drop in self.se.dropped:
            self.buf["e"][p_drop] = 0
        self.n += 1
        self.t = self.n * dt

    def _load_probe_buf(self):
        spp = self.spaces["p"]
        self.buf["p"][: spp.n_own] = self.buf["e"][: self._ne]

    def _record_probes(self):
        pc = self.pc
        e_src = (
            self.buf["p"] if (pc["has_lts"] and pc["has_probe_halo"]) else self.buf["e"]
        )
        for i, p in enumerate(pc["probes"]):
            if not p["owned"]:
                continue
            if pc["is_lts_cls"]:
                rdata = p["row_data"]
                v = (rdata * e_src[p["loc_cols"]]).sum()
                v = (
                    complex(v)
                    if np.iscomplexobj(rdata) or self.dtype is complex
                    else float(v)
                )
                v = v / p["L"]
            else:
                v = (p["row_loc"] @ e_src).item() / p["L"]
            self.probe_values[i].append(v)
            if self.probe_dft[i] is not None:
                self.probe_dft[i] += (
                    v * np.exp(-1j * 2 * np.pi * p["freqs"] * self.t) * self.dt
                )

    # ---------------------------------------------------- state & energy
    def state(self):
        out = {
            "b": self.buf["b"][: self._nb].copy(),
            "e": self.buf["e"][: self._ne].copy(),
        }
        if self.pc["has_cpml"]:
            out["psi"] = [v.copy() for v in self.psi]
            out["phi"] = [v.copy() for v in self.phi]
        return out

    def owned_index(self):
        out = {"b": self.sb.owned, "e": self.se.owned}
        if self.pc["has_cpml"]:
            out["psi"] = [p["sel_f"] for p in self.pc["pml"]]
            out["phi"] = [p["sel_e"] for p in self.pc["pml"]]
        return out

    def set_state(self, glob):
        self.buf["b"][:] = np.asarray(glob["b"], dtype=self.dtype)[
            np.r_[self.sb.owned, self.sb.halo]
        ]
        self.buf["e"][:] = np.asarray(glob["e"], dtype=self.dtype)[
            np.r_[self.se.owned, self.se.halo]
        ]
        for p_drop in self.se.dropped:
            self.buf["e"][p_drop] = 0
        if self.pc["has_cpml"]:
            for d, p in enumerate(self.pc["pml"]):
                if "psi" in glob:
                    self.psi[d][:] = np.asarray(glob["psi"][d], dtype=self.dtype)[
                        p["sel_f"]
                    ]
                else:
                    self.psi[d][:] = 0
                if "phi" in glob:
                    self.phi[d][:] = np.asarray(glob["phi"][d], dtype=self.dtype)[
                        p["sel_e"]
                    ]
                else:
                    self.phi[d][:] = 0

    def energy_partial(self):
        e_own = self.buf["e"][: self._ne]
        b_own = self.buf["b"][: self._nb]
        eb = 0.5 * float(np.real(np.vdot(b_own, self.pc["W_own"] * b_own)))
        ee = 0.5 * float(np.real(np.vdot(e_own, self.pc["D_own"] * e_own)))
        return eb + ee

    # ---------------------------------------------------- exact transpose
    def _build_program_T(self):
        prog = [self._t1_prepare_ge, ("e", "ge_T")]
        if self.pc["has_cpml"]:
            for d in range(3):
                prog.append(("e", f"ae_T_{d}"))
        prog.append(self._t2_eval_b1bar)
        prog.append(("b", "b1bar_T"))
        if self.pc["has_cpml"]:
            for d in range(3):
                prog.append(("b", f"ah_T_{d}"))
        prog.append(self._t3_eval_zbar)
        if self.pc["has_lts"]:
            prog.append(self._lts_T_begin)
            prog.extend(self._lts_T_prog(0))
            prog.append(self._lts_T_end)
        else:
            prog.append(self._t4_finish_no_lts)
        return prog

    def _lts_T_prog(self, lv):
        """Unrolled exact transpose of `_lts_prog(lv)` (`_R_T`)."""
        if lv >= self.pc["lts_L"]:
            return []
        m = lv + 1
        d = self.pc["lts"][m]
        ops = [lambda m=m: self._rt_push_head(m)]
        ops.extend(self._lts_T_prog(m))
        ops.append(lambda m=m: self._rt_after_head_and_corr_T_begin(m))
        for k in range(d["p"] - 1, 0, -1):
            ops.append(lambda m=m, k=k: self._rt_cheb_T_pre(m, k))
            ops.append((f"fc{m}", f"fc{m}"))
            ops.append(lambda m=m, k=k: self._rt_phi_T(m, k))
            ops.append((f"phi{m}", f"phi{m}"))
            ops.append(lambda m=m: self._rt_after_phi_T(m))
            if m < self.pc["lts_L"]:
                ops.append(lambda m=m: self._rt_push_sub(m))
                ops.extend(self._lts_T_prog(m))
                ops.append(lambda m=m: self._rt_pop_sub(m))
            else:
                ops.append(lambda m=m: self._rt_leaf_sub(m))
        if d["p"] > 1:
            ops.append(lambda m=m: self._rt_corr_T_load_g(m))
            ops.append((f"phi{m}", f"phi{m}"))
        ops.append(lambda m=m: self._rt_corr_T_finish(m))
        return ops

    def set_adjoint(self, lam):
        self._lam_b = np.asarray(lam["b"], dtype=self.dtype)[self.sb.owned].copy()
        self._lam_e = np.asarray(lam["e"], dtype=self.dtype)[self.se.owned].copy()
        if self.pc["has_cpml"]:
            self._lam_psi = [
                np.asarray(lam["psi"][d], dtype=self.dtype)[p["sel_f"]].copy()
                for d, p in enumerate(self.pc["pml"])
            ]
            self._lam_phi = [
                np.asarray(lam["phi"][d], dtype=self.dtype)[p["sel_e"]].copy()
                for d, p in enumerate(self.pc["pml"])
            ]
        else:
            self._lam_psi = []
            self._lam_phi = []

    def _t1_prepare_ge(self):
        pc, dt = self.pc, self.dt
        ge_own = dt * pc["Dinv_own"] * self._lam_e
        self.buf["ge_T"][:] = 0
        self.buf["ge_T"][: self._ne] = ge_own
        self._phib_T = []
        if pc["has_cpml"]:
            for d, p in enumerate(pc["pml"]):
                tot = self._lam_phi[d] + ge_own[p["loc_pe"]]
                self._phib_T.append(p["be"] * tot)
                buf_ae = self.buf[f"ae_T_{d}"]
                buf_ae[:] = 0
                buf_ae[p["loc_pe"]] = p["ae"] * tot

    def _t2_eval_b1bar(self):
        pc, dt = self.pc, self.dt
        ybar = pc["KHT"] @ self.buf["ge_T"]
        if pc["has_cpml"]:
            for d, p in enumerate(pc["pml"]):
                ybar = ybar + p["KeT"] @ self.buf[f"ae_T_{d}"]
        b1bar_own = self._lam_b + pc["W_own"] * ybar
        self._b1bar_own = b1bar_own
        self.buf["b1bar_T"][:] = 0
        self.buf["b1bar_T"][: self._nb] = b1bar_own
        self._psib_T = []
        if pc["has_cpml"]:
            for d, p in enumerate(pc["pml"]):
                tot = self._lam_psi[d] - dt * b1bar_own[p["loc_pf"]]
                self._psib_T.append(p["bh"] * tot)
                buf_ah = self.buf[f"ah_T_{d}"]
                buf_ah[:] = 0
                buf_ah[p["loc_pf"]] = p["ah"] * tot

    def _t3_eval_zbar(self):
        pc, dt = self.pc, self.dt
        zbar = -dt * (pc["KT"] @ self.buf["b1bar_T"])
        if pc["has_cpml"]:
            for d, p in enumerate(pc["pml"]):
                zbar = zbar + p["KfT"] @ self.buf[f"ah_T_{d}"]
        self._zbar_own = zbar

    def _t4_finish_no_lts(self):
        ebar = self._lam_e + self._zbar_own
        self._store_out_T(ebar)

    def _lts_T_begin(self):
        L = self.pc["lts_L"]
        self._rt_vec = [None] * (L + 1)
        self._rt_vec[0] = self._zbar_own.copy()
        self._rt_state = [None] * (L + 1)

    def _rt_push_head(self, m):
        d = self.pc["lts"][m]
        u_bar = self._rt_vec[m - 1]
        self._rt_vec[m] = u_bar[d["iF"]].copy()

    def _rt_after_head_and_corr_T_begin(self, m):
        d = self.pc["lts"][m]
        uF_bar = self._rt_vec[m]
        v_bar = self._rt_vec[m - 1].copy()
        v_bar[d["iF"]] = uF_bar
        self._rt_state[m] = {
            "v_bar": v_bar,
            "c_bar": uF_bar,
            "dk_bar": d["out"] * uF_bar if d["p"] > 1 else np.zeros_like(uF_bar),
            "yF_bar": -uF_bar.copy() if d["p"] > 1 else np.zeros_like(uF_bar),
            "d_old_bar": np.zeros_like(uF_bar),
            "g_bar": np.zeros(d["n_Phi_own"], dtype=self.dtype),
            "z_bar": None,
        }

    def _rt_cheb_T_pre(self, m, k):
        d = self.pc["lts"][m]
        st = self._rt_state[m]
        d_new_bar = st["dk_bar"]
        st["dk_bar"] = 2 * d["delta"] * d_new_bar + st["d_old_bar"]
        st["d_old_bar"] = -d_new_bar
        st["yF_bar"] = st["yF_bar"] + 2 * d["T"][k] * d_new_bar
        Xd_bar = -2 * d["beta"] * d_new_bar
        buf_fc = self.buf[f"fc{m}"]
        buf_fc[:] = 0
        buf_fc[: d["n_F_own"]] = d["tau2"] * d["Dinv_own"] * Xd_bar

    def _rt_phi_T(self, m, k):
        d = self.pc["lts"][m]
        st = self._rt_state[m]
        w_bar = d["W_own"] * (d["KH_F_T"] @ self.buf[f"fc{m}"])
        st["g_bar"] = st["g_bar"] + d["kappa"][k] * w_bar
        buf_phi = self.buf[f"phi{m}"]
        buf_phi[:] = 0
        buf_phi[: d["n_Phi_own"]] = w_bar

    def _rt_after_phi_T(self, m):
        d = self.pc["lts"][m]
        st = self._rt_state[m]
        st["z_bar"] = d["K_F_T"] @ self.buf[f"phi{m}"]

    def _rt_push_sub(self, m):
        self._rt_vec[m] = self._rt_state[m]["z_bar"].copy()

    def _rt_pop_sub(self, m):
        st = self._rt_state[m]
        st["dk_bar"] = st["dk_bar"] + self._rt_vec[m]

    def _rt_leaf_sub(self, m):
        st = self._rt_state[m]
        st["dk_bar"] = st["dk_bar"] + st["z_bar"]

    def _rt_corr_T_load_g(self, m):
        d = self.pc["lts"][m]
        st = self._rt_state[m]
        buf_phi = self.buf[f"phi{m}"]
        buf_phi[:] = 0
        buf_phi[: d["n_Phi_own"]] = st["g_bar"]

    def _rt_corr_T_finish(self, m):
        d = self.pc["lts"][m]
        st = self._rt_state[m]
        v_bar = st["v_bar"]
        if d["p"] > 1:
            yF_bar = st["yF_bar"] + st["dk_bar"]
            yC_bar = d["K_C_T"] @ self.buf[f"phi{m}"]
            v_bar[d["iF"]] = v_bar[d["iF"]] + yF_bar
            v_bar[d["iC"]] = v_bar[d["iC"]] + yC_bar
        self._rt_vec[m - 1] = v_bar

    def _lts_T_end(self):
        ebar = self._lam_e + self._rt_vec[0]
        self._store_out_T(ebar)

    def _store_out_T(self, ebar):
        out = {"b": self._b1bar_own.copy(), "e": ebar.copy()}
        if self.pc["has_cpml"]:
            out["psi"] = [v.copy() for v in self._psib_T]
            out["phi"] = [v.copy() for v in self._phib_T]
        self._out_T = out


# ================================================================ drivers
class _Driver:
    """Common interface of `SubgridLocalCluster` and `DistributedSubgridFIT3D`."""

    def _layout_sizes(self):
        pc0 = self._pc0
        sizes = [pc0["n_faces"], pc0["n_free"]]
        if pc0["has_cpml"]:
            sizes += [p["n_pf_glob"] for p in pc0["pml"]]
            sizes += [p["n_pe_glob"] for p in pc0["pml"]]
        return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)

    def _unpack_state(self, x):
        if isinstance(x, dict):
            return x, False
        x = np.asarray(x, dtype=self._pc0["dtype"])
        o = self._layout_sizes()
        parts = [x[o[i] : o[i + 1]] for i in range(len(o) - 1)]
        d = {"b": parts[0], "e": parts[1]}
        if self._pc0["has_cpml"]:
            d["psi"] = parts[2:5]
            d["phi"] = parts[5:8]
        return d, True

    def _pack_state(self, d):
        if d is None:
            return None
        vecs = [d["b"], d["e"]]
        if self._pc0["has_cpml"]:
            vecs += list(d["psi"]) + list(d["phi"])
        return np.concatenate(vecs)

    def run(self, n_steps):
        """Advance `n_steps` steps, recording probes at every integer time level."""
        self._maybe_record_initial()
        for _ in range(int(n_steps)):
            self.step()

    def apply(self, x):
        """
        One source-free step of global state `x` (dict or packed 1D ndarray).
        Returns the updated global state (on rank 0 under MPI) in the same format.
        """
        d_in, packed = self._unpack_state(x)
        saved = self._each(
            lambda r: (
                r.n,
                r.t,
                r.buf["b"].copy(),
                r.buf["e"].copy(),
                [v.copy() for v in r.psi],
                [v.copy() for v in r.phi],
                [list(v) for v in r.probe_values],
                [None if v is None else v.copy() for v in r.probe_dft],
            )
        )
        self._each(lambda r: r.set_state(d_in))
        self._each(lambda r: setattr(r, "use_sources", False))
        try:
            self.step()
            out = self.gather_state()
        finally:
            self._each(lambda r: setattr(r, "use_sources", True))
            self._restore(saved)
        return self._pack_state(out) if packed else out

    def apply_T(self, y):
        """
        Exact transpose of the source-free step `apply` on global adjoint `y`
        (dict or packed 1D ndarray).
        """
        d_in, packed = self._unpack_state(y)
        self._each(lambda r: r.set_adjoint(d_in))
        self._run_program("program_T")
        out = self._gather_state_from(lambda r: r._out_T)
        return self._pack_state(out) if packed else out

    def exchange_counts_per_step(self):
        """
        Number of collective halo exchanges per coarse step:
        ``2`` for a global step, ``2 + 2 * sum_{l=1..L} (p_l - 1) prod_{m<l} p_m``
        under LTS.
        """
        r0 = self.ranks[0] if hasattr(self, "ranks") else self.r
        exchanges = [
            item for item in r0.program if not callable(item) and item[0] != "p"
        ]
        return len(exchanges)


class SubgridLocalCluster(_Driver):
    """
    All ranks of a partition stepped in one process (no MPI).

    Executes the exact per-rank arithmetic and halo copies of
    `DistributedSubgridFIT3D`.

    Parameters
    ----------
    solver : SubgridFIT3D or SubgridLTSFIT3D
        Configured serial solver.
    n_parts : int
        Number of parts (ranks).
    parts : ndarray, optional
        Active-cell partition.
    method : {"rcb", "rib", "blocks"}
        Partition method when `parts` is None.
    weights : {"work", "uniform"} or ndarray
        Per-cell work weights when `parts` is None.
    _mutation : {None, "drop_halo", "skip_lts_exchange"}
        Verification-only mutation hook.
    """

    def __init__(
        self,
        solver,
        n_parts,
        parts=None,
        method="rcb",
        weights="work",
        partitioner=None,
        _mutation=None,
    ):
        if partitioner is not None:
            method = partitioner
        pieces, self.parts = build_pieces(
            solver,
            n_parts,
            parts=parts,
            method=method,
            weights=weights,
            _mutation=_mutation,
        )
        self.ranks = [_SubgridRank(pc) for pc in pieces]
        self._pc0 = pieces[0]
        self.n_parts = int(n_parts)
        self.method = method

    @property
    def n(self):
        return self.ranks[0].n

    def _each(self, fn):
        return [fn(r) for r in self.ranks]

    def _restore(self, saved):
        for r, (n, t, b_buf, e_buf, psi, phi, pv, pd) in zip(self.ranks, saved):
            r.n, r.t = n, t
            r.buf["b"][:] = b_buf
            r.buf["e"][:] = e_buf
            r.psi = psi
            r.phi = phi
            r.probe_values = pv
            r.probe_dft = pd

    def _maybe_record_initial(self):
        r0 = self.ranks[0]
        if r0.n == 0 and r0.pc["probes"]:
            owner0 = r0.pc["probes"][0]["owner"]
            if len(self.ranks[owner0].probe_values[0]) == 0:
                if r0.pc["has_lts"] and r0.pc["has_probe_halo"]:
                    for r in self.ranks:
                        r._load_probe_buf()
                    self._exchange("p", "p")
                for r in self.ranks:
                    r._record_probes()

    def _exchange(self, key, bkey):
        for r in self.ranks:
            sp_r = r.spaces[key]
            for q, pos in sp_r.recv.items():
                src = self.ranks[q]
                r.buf[bkey][sp_r.n_own + pos] = src.buf[bkey][
                    src.spaces[key].send[r.rank]
                ]

    def _run_program(self, name):
        prog0 = getattr(self.ranks[0], name)
        for k, item in enumerate(prog0):
            if callable(item):
                for r in self.ranks:
                    getattr(r, name)[k]()
            else:
                self._exchange(*item)

    def step(self):
        """One coarse step across all ranks."""
        self._run_program("program")

    def _gather_state_from(self, getter):
        vals = [getter(r) for r in self.ranks]
        inds = [r.owned_index() for r in self.ranks]
        pc0 = self._pc0
        dtp = pc0["dtype"]
        b = np.zeros(pc0["n_faces"], dtype=dtp)
        e = np.zeros(pc0["n_free"], dtype=dtp)
        for v, idx in zip(vals, inds):
            b[idx["b"]] = v["b"]
            e[idx["e"]] = v["e"]
        out = {"b": b, "e": e}
        if pc0["has_cpml"]:
            psi, phi = [], []
            for d, p in enumerate(pc0["pml"]):
                psi_d = np.zeros(p["n_pf_glob"], dtype=dtp)
                phi_d = np.zeros(p["n_pe_glob"], dtype=dtp)
                for v, idx in zip(vals, inds):
                    psi_d[idx["psi"][d]] = v["psi"][d]
                    phi_d[idx["phi"][d]] = v["phi"][d]
                psi.append(psi_d)
                phi.append(phi_d)
            out["psi"] = psi
            out["phi"] = phi
        return out

    def gather_state(self):
        """Gather global state dictionary ``{'b', 'e', ...}``."""
        return self._gather_state_from(lambda r: r.state())

    def gather(self, key):
        return self.gather_state()[key]

    def energy(self):
        """Global discrete energy ``1/2 b.W b + 1/2 e.D e`` [J]."""
        return float(sum(r.energy_partial() for r in self.ranks))

    def probe_signal(self, i=0):
        """Recorded time series of probe `i` [V/m]."""
        po = self._pc0["probes"][i]["owner"]
        return np.asarray(self.ranks[po].probe_values[i])

    def dft_data(self, i=0):
        """Running DFT array of probe `i` [V s / m]."""
        po = self._pc0["probes"][i]["owner"]
        dft = self.ranks[po].probe_dft[i]
        return None if dft is None else dft.copy()

    def comm_stats(self):
        """Halo sizes per rank and space."""
        return [{k: s.halo_size for k, s in r.spaces.items()} for r in self.ranks]


class DistributedSubgridFIT3D(_Driver):
    """
    One rank of an MPI-distributed 3D subgrid FIT simulation.

    Parameters
    ----------
    solver : SubgridFIT3D or SubgridLTSFIT3D or None
        In ``mode="replicated"``, every rank passes an identically built
        solver and constructs its own local piece. In ``mode="scatter"``,
        only `root` needs to pass the solver (non-root ranks may pass `None`);
        `root` builds all pieces and scatters them (source waveforms must be
        picklable).
    comm : mpi4py.MPI.Comm
        MPI communicator.
    parts : ndarray, optional
        Active-cell partition (computed on `root` and broadcast if omitted).
    method : {"rcb", "rib", "blocks"}
        Partition method when `parts` is None.
    weights : {"work", "uniform"} or ndarray
        Per-cell work weights when `parts` is None.
    mode : {"replicated", "scatter"}
        Setup distribution mode.
    root : int
        Root rank index (default 0).
    timers : bool
        Accumulate per-step compute and communication wall times using
        `MPI.Wtime()` in ``self.t``.
    _mutation : {None, "drop_halo", "skip_lts_exchange"}
        Verification-only mutation hook.
    """

    def __init__(
        self,
        solver,
        comm,
        parts=None,
        method="rcb",
        weights="work",
        partitioner=None,
        mode="replicated",
        root=0,
        timers=False,
        _mutation=None,
    ):
        from mpi4py import MPI

        if partitioner is not None:
            method = partitioner
        self._MPI = MPI
        self.comm = comm
        self.root = int(root)
        P, me = comm.Get_size(), comm.Get_rank()
        self.n_parts = P
        self.method = method

        if mode == "replicated":
            if parts is None:
                parts = (
                    partition_cells(solver, P, method=method, weights=weights)
                    if me == self.root
                    else None
                )
                parts = comm.bcast(parts, root=self.root)
            pieces, self.parts = build_pieces(
                solver,
                P,
                parts=parts,
                method=method,
                weights=weights,
                ranks=[me],
                _mutation=_mutation,
            )
            pc = pieces[me]
        elif mode == "scatter":
            pieces = None
            if me == self.root:
                pieces, _ = build_pieces(
                    solver,
                    P,
                    parts=parts,
                    method=method,
                    weights=weights,
                    _mutation=_mutation,
                )
            pc = comm.scatter(pieces, root=self.root)
            self.parts = pc["parts"]
        else:
            raise ValueError("mode must be 'replicated' or 'scatter'")

        self.r = _SubgridRank(pc)
        self._pc0 = pc
        self.timers = bool(timers)
        self.t = {"compute": 0.0, "exchange": 0.0}
        self.bytes_sent = 0
        self.messages_sent = 0

    @property
    def n(self):
        return self.r.n

    def _each(self, fn):
        return [fn(self.r)]

    def _restore(self, saved):
        n, t, b_buf, e_buf, psi, phi, pv, pd = saved[0]
        self.r.n, self.r.t = n, t
        self.r.buf["b"][:] = b_buf
        self.r.buf["e"][:] = e_buf
        self.r.psi = psi
        self.r.phi = phi
        self.r.probe_values = pv
        self.r.probe_dft = pd

    def _maybe_record_initial(self):
        if self.r.n == 0 and self.r.pc["probes"]:
            owner0 = self.r.pc["probes"][0]["owner"]
            already = (
                len(self.r.probe_values[0]) > 0
                if self.comm.Get_rank() == owner0
                else False
            )
            already = self.comm.bcast(already, root=owner0)
            if not already:
                if self.r.pc["has_lts"] and self.r.pc["has_probe_halo"]:
                    self.r._load_probe_buf()
                    self._exchange("p", "p")
                self.r._record_probes()

    def _exchange(self, key, bkey):
        MPI = self._MPI
        sp_ = self.r.spaces[key]
        buf = self.r.buf[bkey]
        reqs, rbufs, sbufs = [], {}, []
        for q, pos in sp_.recv.items():
            rbufs[q] = np.empty(pos.size, dtype=buf.dtype)
            reqs.append(self.comm.Irecv(rbufs[q], source=q, tag=sp_.tag))
        for q, pos in sp_.send.items():
            sb = np.ascontiguousarray(buf[pos])
            sbufs.append(sb)
            reqs.append(self.comm.Isend(sb, dest=q, tag=sp_.tag))
            self.bytes_sent += int(sb.nbytes)
            self.messages_sent += 1
        if reqs:
            MPI.Request.Waitall(reqs)
        for q, pos in sp_.recv.items():
            buf[sp_.n_own + pos] = rbufs[q]

    def _run_program(self, name):
        wtime = self._MPI.Wtime
        for item in getattr(self.r, name):
            t0 = wtime() if self.timers else 0.0
            if callable(item):
                item()
                if self.timers:
                    self.t["compute"] += wtime() - t0
            else:
                self._exchange(*item)
                if self.timers:
                    self.t["exchange"] += wtime() - t0

    def step(self):
        """One coarse step on this rank."""
        self._run_program("program")

    def _gather_state_from(self, getter):
        parts = self.comm.gather((getter(self.r), self.r.owned_index()), root=self.root)
        if self.comm.Get_rank() != self.root:
            return None
        pc0 = self._pc0
        dtp = pc0["dtype"]
        b = np.zeros(pc0["n_faces"], dtype=dtp)
        e = np.zeros(pc0["n_free"], dtype=dtp)
        for v, idx in parts:
            b[idx["b"]] = v["b"]
            e[idx["e"]] = v["e"]
        out = {"b": b, "e": e}
        if pc0["has_cpml"]:
            psi, phi = [], []
            for d, p in enumerate(pc0["pml"]):
                psi_d = np.zeros(p["n_pf_glob"], dtype=dtp)
                phi_d = np.zeros(p["n_pe_glob"], dtype=dtp)
                for v, idx in parts:
                    psi_d[idx["psi"][d]] = v["psi"][d]
                    phi_d[idx["phi"][d]] = v["phi"][d]
                psi.append(psi_d)
                phi.append(phi_d)
            out["psi"] = psi
            out["phi"] = phi
        return out

    def gather_state(self):
        """Gather global state dictionary onto `root` (`None` on other ranks)."""
        return self._gather_state_from(lambda r: r.state())

    def gather(self, key):
        st = self.gather_state()
        return None if st is None else st[key]

    def energy(self):
        """Global discrete energy (`Allreduce` of per-rank partial sums) [J]."""
        return float(self.comm.allreduce(float(self.r.energy_partial())))

    def probe_signal(self, i=0):
        """Gather recorded time series of probe `i` onto `root` [V/m]."""
        p_meta = self.r.pc["probes"][i]
        po = p_meta["owner"]
        me = self.comm.Get_rank()
        if po == self.root:
            return np.asarray(self.r.probe_values[i]) if me == self.root else None
        if me == po:
            self.comm.send(
                np.asarray(self.r.probe_values[i]), dest=self.root, tag=90 + i
            )
            return None
        if me == self.root:
            return self.comm.recv(source=po, tag=90 + i)
        return None

    def dft_data(self, i=0):
        """Gather running DFT array of probe `i` onto `root` [V s / m]."""
        p_meta = self.r.pc["probes"][i]
        po = p_meta["owner"]
        me = self.comm.Get_rank()
        if po == self.root:
            if me != self.root:
                return None
            dft = self.r.probe_dft[i]
            return None if dft is None else dft.copy()
        if me == po:
            dft = self.r.probe_dft[i]
            self.comm.send(
                None if dft is None else dft.copy(), dest=self.root, tag=190 + i
            )
            return None
        if me == self.root:
            return self.comm.recv(source=po, tag=190 + i)
        return None

    def comm_stats(self):
        """Gather per-rank halo size dictionaries onto `root`."""
        st = {k: s.halo_size for k, s in self.r.spaces.items()}
        return self.comm.gather(st, root=self.root)


LocalCluster = SubgridLocalCluster
DistributedFIT = DistributedSubgridFIT3D
