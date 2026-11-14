# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Distributed-memory (MPI) time stepping of the unstructured FIT solvers.

The cells of the primal mesh are partitioned (METIS through the optional
`pymetis`, or a pure-NumPy recursive coordinate / inertial bisection) and
every degree of freedom is *owned* by one rank (the lowest part among the
cells that contain it). Each rank keeps the rows of the solver's operators
that update its owned dofs, with their columns renumbered into a local
vector ``[owned | halo]``; the halo holds copies of the off-rank dofs those
rows read and is refreshed by one neighbour exchange per vector and step::

    s = UnstructuredFIT3D(mesh, materials, bc="abc")      # serial set-up
    s.add_point_dipole(...); s.add_probe(...); s.add_dft(f)
    d = DistributedFIT(s, comm=MPI.COMM_WORLD)            # one per rank
    d.run(n_steps)
    u = d.gather("u")                                     # on rank 0

    c = LocalCluster(s, n_parts=4)                        # all ranks in one
    c.run(n_steps)                                        # process (tests)

Step of `UnstructuredFIT3D` (Voronoi Hodge) and 2D TM, per rank::

    b[own] -= dt A[own, :] u[own | halo]           A = C_f (C_eff with massless edges)
    exchange b halo
    r = C_f^T[own, :] (M_nu b)[own | halo] - j[own]
    u[own] = capacitive update (local per dof: conductivity, ABC, ADE poles)
    exchange u halo

Each row is the same CSR row as the serial operator with the same entry
order, so each owned entry is the *same* sequence of floating-point
operations as in the serial step: the distributed run is bitwise equal to
`one_step` (and to `DeviceStepper(xp=numpy)` with massless edges, whose
fused ``C_eff = C_f - C_s K_ss^-1 K_sf`` it shares; see the physics guide
section ``unstructured_mpi.md`` for the derivation and the round-off bound
against the serial LU elimination).

The PML of `UnstructuredFIT3D` (`pml3d.PML3D`, four sparse stages on the
extended state ``x = [u, b, d, filter states]``) is distributed stage by
stage with one exchange per stage. Both engines have the exact transpose
(`DistributedFIT.step_T`) for adjoint runs.

Not supported (``NotImplementedError``): ``hodge="whitney"`` (a global
sparse solve per step), the 2D PML, LTS and plane waves, 2D TE with
dispersive media, and the transpose with dispersive media.
"""

import time

import numpy as np
import scipy.sparse as sp

UNIT_ROUNDOFF = 2.0**-53

__all__ = [
    "cell_adjacency",
    "partition_cells",
    "partition_stats",
    "build_pieces",
    "LocalCluster",
    "DistributedFIT",
]


# ================================================================ partitioning
def cell_adjacency(mesh):
    """
    Dual graph of the mesh: cells sharing a facet (face in 3D, edge in 2D).

    Returns
    -------
    csr_matrix (n_cells, n_cells)
        Symmetric 0/1 adjacency without self loops.
    """
    facets = mesh.cell_faces if mesh.dim == 3 else mesh.cell_edges
    nc, k = facets.shape
    f = facets.ravel()
    c = np.repeat(np.arange(nc), k)
    o = np.argsort(f, kind="stable")
    f, c = f[o], c[o]
    same = f[1:] == f[:-1]
    a, b = c[:-1][same], c[1:][same]
    A = sp.csr_matrix(
        (np.ones(2 * a.size), (np.r_[a, b], np.r_[b, a])), shape=(nc, nc)
    )
    A.data[:] = 1.0
    A.sort_indices()
    return A


def _bisect(points, ids, n_parts, out, first, inertial):
    if n_parts == 1:
        out[ids] = first
        return
    P = points[ids]
    if inertial:
        X = P - P.mean(0)
        _, V = np.linalg.eigh(X.T @ X)
        key = X @ V[:, -1]
    else:
        key = P[:, np.argmax(P.max(0) - P.min(0))]
    nl = n_parts // 2
    k = int(round(ids.size * nl / n_parts))
    o = np.argsort(key, kind="stable")
    _bisect(points, ids[o[:k]], nl, out, first, inertial)
    _bisect(points, ids[o[k:]], n_parts - nl, out, first + nl, inertial)


def partition_cells(mesh, n_parts, method="auto"):
    """
    Partition the cells into `n_parts` parts.

    Parameters
    ----------
    mesh : SimplexMesh
    n_parts : int
    method : {"auto", "metis", "rcb", "rib"}
        ``metis``: k-way METIS on the dual graph (`pymetis`, optional, MIT
        bindings to the Apache-2 METIS); ``rcb``: recursive coordinate
        bisection of the cell centroids (longest extent, balanced counts);
        ``rib``: recursive inertial bisection (principal axis). ``auto`` is
        METIS when `pymetis` is importable, else RCB.

    Returns
    -------
    ndarray of int (n_cells,)
        Part of every cell, every part non-empty.
    """
    n_parts = int(n_parts)
    nc = mesh.n_cells
    if n_parts < 1 or n_parts > nc:
        raise ValueError(f"n_parts must be in [1, {nc}]")
    if n_parts == 1:
        return np.zeros(nc, dtype=np.int64)
    if method == "auto":
        try:
            import pymetis  # noqa: F401

            method = "metis"
        except ImportError:
            method = "rcb"
    if method == "metis":
        import pymetis

        A = cell_adjacency(mesh)
        _, parts = pymetis.part_graph(
            n_parts, xadj=A.indptr.astype(np.int64), adjncy=A.indices.astype(np.int64)
        )
        parts = np.asarray(parts, dtype=np.int64)
    elif method in ("rcb", "rib"):
        cent = mesh.nodes[mesh.cells].mean(1)
        parts = np.empty(nc, dtype=np.int64)
        _bisect(cent, np.arange(nc), n_parts, parts, 0, method == "rib")
    else:
        raise ValueError(f"unknown partition method {method!r}")
    if np.bincount(parts, minlength=n_parts).min() == 0:
        raise ValueError("empty part: use fewer parts or another method")
    return parts


def partition_stats(mesh, parts):
    """
    Size and cut of a cell partition.

    Returns
    -------
    dict
        ``n_parts``, ``cells`` (per part), ``imbalance`` (max / mean),
        ``cut_facets`` (facets between cells of different parts: the
        partition surface in facets).
    """
    A = sp.triu(cell_adjacency(mesh), 1).tocoo()
    n_parts = int(parts.max()) + 1
    cells = np.bincount(parts, minlength=n_parts)
    return dict(
        n_parts=n_parts,
        cells=cells.tolist(),
        imbalance=float(cells.max() / cells.mean()),
        cut_facets=int(np.sum(parts[A.row] != parts[A.col])),
    )


def _owner(entities, parts, n_entities, n_parts):
    """Lowest part among the cells containing each entity."""
    own = np.full(n_entities, n_parts, dtype=np.int64)
    np.minimum.at(own, entities.ravel(), np.repeat(parts, entities.shape[1]))
    return own


# ================================================================ halo spaces
class _Space:
    """
    Layout of one distributed vector on one rank: ``[owned | halo]``.

    ``send[q]`` are positions in the owned block sent to rank q,
    ``recv[q]`` positions in the halo block received from rank q (the same
    global entries in the same order).
    """

    def __init__(self, owned, halo, send, recv, tag):
        self.owned, self.halo = owned, halo
        self.send, self.recv = send, recv
        self.tag = int(tag)
        self.n_own = owned.size
        self.n = owned.size + halo.size
        self.dropped = []  # halo positions never refreshed (mutation hook)

    def lookup(self, n_global):
        loc = np.full(n_global, -1, dtype=np.int64)
        loc[self.owned] = np.arange(self.n_own)
        loc[self.halo] = self.n_own + np.arange(self.halo.size)
        return loc

    @property
    def halo_size(self):
        return int(sum(v.size for v in self.recv.values()))

    @property
    def neighbours(self):
        return sorted(set(self.send) | set(self.recv))


def _build_spaces(owner, needs, n_parts, tag):
    """Halo spaces of all ranks from the global entries each rank reads."""
    owned = [np.flatnonzero(owner == r) for r in range(n_parts)]
    pos = np.empty(owner.size, dtype=np.int64)
    for o in owned:
        pos[o] = np.arange(o.size)
    halos, recvs = [], []
    sends = [dict() for _ in range(n_parts)]
    for r in range(n_parts):
        h = np.setdiff1d(needs[r], owned[r])
        oh = owner[h]
        rv = {}
        for q in np.unique(oh):
            k = np.flatnonzero(oh == q)
            rv[int(q)] = k
            sends[int(q)][r] = pos[h[k]]
        halos.append(h)
        recvs.append(rv)
    return [_Space(owned[r], halos[r], sends[r], recvs[r], tag) for r in range(n_parts)]


def _drop_halo_entry(spaces, rank):
    """Mutation: rank `rank` never receives its first halo entry."""
    sp_r = spaces[rank]
    q = min(sp_r.recv)
    k = sp_r.recv[q]
    sp_r.dropped.append(int(sp_r.n_own + k[0]))
    sp_r.recv[q] = k[1:]
    spaces[q].send[rank] = spaces[q].send[rank][1:]


def _ref_csr(A):
    """
    CSR with the entry order of the serial SpMV of `A`.

    SciPy's CSR SpMV sums each row in stored order; its CSC SpMV (also the
    transpose ``A.T`` of a CSR matrix) accumulates every row in increasing
    column order, i.e. the sorted CSR row. Keeping that order keeps every
    distributed row sum bitwise equal to the serial one.
    """
    if sp.isspmatrix_csr(A) or isinstance(A, sp.csr_array):
        return sp.csr_matrix(A)
    A = sp.csr_matrix(A.tocsr())
    A.sort_indices()
    return A


def _cols(A, rows):
    return np.unique(A[rows].indices)


def _local(A, rows, lookup, n_loc):
    """Rows `rows` of CSR `A` with columns renumbered by `lookup` (order kept)."""
    sub = A[rows]
    idx = lookup[sub.indices]
    if np.any(idx < 0):
        raise RuntimeError("halo does not cover the operator columns")
    return sp.csr_matrix(
        (sub.data.copy(), idx.astype(np.int64), sub.indptr.astype(np.int64)),
        shape=(rows.size, n_loc),
    )


# ================================================================ pieces
def _solver_kind(s):
    if s.mesh.dim == 3:
        if s.hodge != "voronoi":
            raise NotImplementedError("MPI: hodge='whitney' needs a global solve per step")
        return "pml3d" if s._pml is not None else "3d"
    if s._pml is not None or s.lts is not None or s.plane_wave is not None:
        raise NotImplementedError("MPI: 2D PML / LTS / plane wave")
    if s.pol == "TE" and s.dispersive:
        raise NotImplementedError("MPI: 2D TE with dispersive media")
    return "tm" if s.pol == "TM" else "te"


def _probe_owner(P, owner_cols):
    P = sp.csr_matrix(P)
    out = np.zeros(P.shape[0], dtype=np.int64)
    for i in range(P.shape[0]):
        c = P.indices[P.indptr[i] : P.indptr[i + 1]]
        out[i] = owner_cols[c].min() if c.size else 0
    return out


def _slice_capacitive(up, own, pos_own, owner_u, rank):
    """`CapacitiveUpdate` restricted to the owned dofs (same arithmetic)."""
    from ._core import CapacitiveUpdate

    loc = CapacitiveUpdate.__new__(CapacitiveUpdate)
    loc.dt = up.dt
    loc.a1, loc.den = up.a1[own].copy(), up.den[own].copy()
    loc.groups = []
    for g in up.groups:
        sel = owner_u[g["idx"]] == rank
        loc.groups.append(
            {
                "idx": pos_own[g["idx"][sel]],
                "c": g["c"][sel].copy(),
                "lorentz": [
                    (a, x, z, p[sel].copy(), q[sel].copy()) for a, x, z, p, q in g["lorentz"]
                ],
                "debye": [(k, b, p[sel].copy()) for k, b, p in g["debye"]],
            }
        )
    return loc


def _field_ops(s, kind, fuse=True):
    """Global operators of the field engine (serial entry order)."""
    ops = {}
    if kind == "3d":
        if s.massless.size and fuse:
            from .gpu import massless_schur_operator

            _, S, _, _ = massless_schur_operator(s)
            Ceff = (s.Cf - s.Cs @ S).tocsr()
            Ceff.sort_indices()  # as DeviceStepper._mat
            ops["A"] = Ceff
        else:
            ops["A"] = _ref_csr(s.Cf)
        ops["AT"] = _ref_csr(ops["A"].T)  # transpose step: A^T
        ops["B"] = _ref_csr(s.Cf)  # transpose step: C_f r
        ops["BT"] = _ref_csr(s.Cf.T)  # forward Ampere: C_f^T h
        wf = np.asarray(s.Mnu_diag, float)
        mass = np.asarray(s.cap, float)
    else:
        ops["A"] = _ref_csr(s.G)
        ops["AT"] = _ref_csr(s.G.T)
        ops["B"], ops["BT"] = ops["A"], ops["AT"]
        wf = np.asarray(s.stiff, float)
        mass = np.asarray(s.mass, float)
    return ops, wf, mass


def build_pieces(solver, n_parts, parts=None, partitioner="auto", ranks=None,
                 _mutation=None):  # fmt: skip
    """
    Per-rank data (local operators, halo spaces, initial state, monitors).

    Parameters
    ----------
    solver : UnstructuredFIT3D or UnstructuredFIT2D
        Fully configured serial solver (its current state is distributed).
    n_parts : int
    parts : ndarray, optional
        Cell partition (default `partition_cells(mesh, n_parts, partitioner)`).
    ranks : list of int, optional
        Build only these ranks' pieces (the halo spaces of all ranks are
        always derived, which is O(nnz)).
    _mutation : str, optional
        ``"drop_halo"``: rank 1 (rank 0 if alone) never receives its first
        halo entry of the field vector u (3D/2D) or of the first stage
        (PML); a test hook that must break the agreement with serial.

    Returns
    -------
    list
        Piece dicts (None for ranks not in `ranks`) and the partition.
    """
    s = solver
    kind = _solver_kind(s)
    mesh = s.mesh
    if parts is None:
        parts = partition_cells(mesh, n_parts, partitioner)
    parts = np.asarray(parts, dtype=np.int64)
    ranks = list(range(n_parts)) if ranks is None else list(ranks)
    if mesh.dim == 3:
        own_e = _owner(mesh.cell_edges, parts, mesh.n_edges, n_parts)
        own_f = _owner(mesh.cell_faces, parts, mesh.n_faces, n_parts)
        owner_u, owner_f = own_e[s.free], own_f
    else:
        own_n = _owner(mesh.cells, parts, mesh.n_nodes, n_parts)
        own_e = _owner(mesh.cell_edges, parts, mesh.n_edges, n_parts)
        owner_u, owner_f = own_n[s.free], own_e
    common = dict(kind=kind, n_parts=n_parts, dt=float(s.dt), n=int(s.n),
                  n_u=s.free.size, parts=parts)  # fmt: skip
    if kind == "pml3d":
        return _pml_pieces(s, common, owner_u, owner_f, ranks, _mutation), parts
    return _field_pieces(s, common, owner_u, owner_f, ranks, _mutation), parts


def _field_pieces(s, common, owner_u, owner_f, ranks, mutation):
    kind, n_parts = common["kind"], common["n_parts"]
    ops, wf, mass = _field_ops(s, kind)
    own_u = [np.flatnonzero(owner_u == r) for r in range(n_parts)]
    own_f = [np.flatnonzero(owner_f == r) for r in range(n_parts)]
    probes = [sp.csr_matrix(rec["P"]) for rec in s.recorder.probes]
    probe_owner = [_probe_owner(P, owner_u) for P in probes]
    # u is read by A (flux rows), B (transpose) and the probes; f by B^T and A^T
    needs_u, needs_f = [], []
    for r in range(n_parts):
        cu = [_cols(ops["A"], own_f[r]), _cols(ops["B"], own_f[r])]
        cu += [_cols(P, np.flatnonzero(po == r)) for P, po in zip(probes, probe_owner)]
        needs_u.append(np.unique(np.concatenate(cu)))
        needs_f.append(np.unique(np.r_[_cols(ops["BT"], own_u[r]), _cols(ops["AT"], own_u[r])]))
    sp_u = _build_spaces(owner_u, needs_u, n_parts, tag=11)
    sp_f = _build_spaces(owner_f, needs_f, n_parts, tag=12)
    if mutation == "drop_halo":
        _drop_halo_entry(sp_u, 1 if n_parts > 1 else 0)
    out = [None] * n_parts
    for r in ranks:
        su, sf = sp_u[r], sp_f[r]
        lu, lf = su.lookup(owner_u.size), sf.lookup(owner_f.size)
        pos_own = np.full(owner_u.size, -1, dtype=np.int64)
        pos_own[su.owned] = np.arange(su.n_own)
        u0 = np.asarray(s.u, float)
        f0 = np.asarray(s.b if kind == "3d" else s.v, float)
        pc = dict(common, rank=r, spaces={"u": su, "f": sf})
        pc["A"] = _local(ops["A"], sf.owned, lu, su.n)
        pc["B"] = _local(ops["B"], sf.owned, lu, su.n)
        pc["BT"] = _local(ops["BT"], su.owned, lf, sf.n)
        pc["AT"] = _local(ops["AT"], su.owned, lf, sf.n)
        pc["wf"] = wf[np.r_[sf.owned, sf.halo]]
        pc["mass"] = mass[su.owned]
        pc["u0"] = u0[np.r_[su.owned, su.halo]]
        pc["f0"] = f0[np.r_[sf.owned, sf.halo]]
        for p in su.dropped:
            pc["u0"][p] = 0.0
        if kind in ("3d", "tm"):
            pc["cap"] = _slice_capacitive(s._update, su.owned, pos_own, owner_u, r)
        else:
            pc["a"] = (s.mass - s.loss * s.dt / 2)[su.owned]
            pc["bb"] = (s.mass + s.loss * s.dt / 2)[su.owned]
        pc["ce"] = None
        if kind in ("tm", "te") and s._absorber is not None:
            pc["ce"] = (0.5 * s.dt * s.edge_rate)[sf.owned]
        srcs = []
        for src in s.sources:
            if len(src) == 3:
                sk, w, f = src
            else:
                (w, f), sk = src, "node"
            w = np.asarray(w, float)
            srcs.append((sk, w[su.owned] if sk == "node" else w[sf.owned], f))
        pc["sources"] = srcs
        pc["probes"] = []
        for P, po in zip(probes, probe_owner):
            rows = np.flatnonzero(po == r)
            pc["probes"].append((rows, _local(P, rows, lu, su.n)))
        pc["dfts"] = [np.asarray(rec["freqs"], float) for rec in s.recorder.dfts]
        pc["dft0"] = [np.asarray(rec["data"])[:, su.owned].copy() for rec in s.recorder.dfts]
        out[r] = pc
    return out


def _pml_entry_owner(pml, owner_u, owner_f, edge_owner_free):
    """Owner of every entry of the PML extended state."""
    L = pml.layout
    out = np.empty(L.n, dtype=np.int64)
    e_own = edge_owner_free(pml.e_pml)
    f_own = owner_f[pml.f_pml]
    for name, idx in L.blocks.items():
        if name == "u":
            out[idx] = owner_u
        elif name == "b":
            out[idx] = owner_f
        elif name == "d" or name.startswith("fe"):
            out[idx] = e_own
        elif name.startswith("ff"):
            out[idx] = f_own
        else:
            raise RuntimeError(f"unknown PML block {name}")
    return out


def _pml_pieces(s, common, owner_u, owner_f, ranks, mutation):
    n_parts = common["n_parts"]
    pml = s._pml
    mesh = s.mesh
    edge_owner = _owner(mesh.cell_edges, common["parts"], mesh.n_edges, n_parts)
    own_x = _pml_entry_owner(pml, owner_u, owner_f, lambda e: edge_owner[e])
    n = pml.layout.n
    nu = s.free.size
    S = [_ref_csr(M) for M in pml.stages]
    ST = [_ref_csr(M.T) for M in pml.stages]
    B = _ref_csr(pml.B)  # (n, nu)
    BT = _ref_csr(pml.B.T)  # (nu, n)
    owned = [np.flatnonzero(own_x == r) for r in range(n_parts)]
    own_u = [np.flatnonzero(owner_u == r) for r in range(n_parts)]
    # u entries of x are the first nu entries and keep their owners
    probes = []
    for rec in s.recorder.probes:
        P = sp.csr_matrix(rec["P"])
        probes.append(sp.csr_matrix((P.data, pml.layout["u"][P.indices], P.indptr),
                                    shape=(P.shape[0], n)))  # fmt: skip
    probe_owner = [_probe_owner(P, own_x) for P in probes]
    spaces = {}
    for k, M in enumerate(S):
        spaces[f"s{k}"] = _build_spaces(own_x, [_cols(M, o) for o in owned], n_parts, 20 + k)
    for k, M in enumerate(ST):
        spaces[f"t{k}"] = _build_spaces(own_x, [_cols(M, o) for o in owned], n_parts, 30 + k)
    need_p = [
        np.unique(np.concatenate([np.zeros(0, np.int64)] + [
            _cols(P, np.flatnonzero(po == r)) for P, po in zip(probes, probe_owner)]))
        for r in range(n_parts)
    ]  # fmt: skip
    spaces["p"] = _build_spaces(own_x, need_p, n_parts, 40)
    if mutation == "drop_halo":
        _drop_halo_entry(spaces["s0"], 1 if n_parts > 1 else 0)
    x0 = np.asarray(pml.x, float)
    out = [None] * n_parts
    for r in ranks:
        pc = dict(common, rank=r, n_x=n)
        pc["spaces"] = {k: v[r] for k, v in spaces.items()}
        own = owned[r]
        pc["own_x"] = own
        pc["n_u_own"] = int(np.sum(own < nu))
        if not np.array_equal(own[: pc["n_u_own"]], own_u[r]):
            raise RuntimeError("PML: u block ownership mismatch")
        pc["S"] = []
        for k, M in enumerate(S):
            spk = pc["spaces"][f"s{k}"]
            pc["S"].append(_local(M, spk.owned, spk.lookup(n), spk.n))
        pc["ST"] = []
        for k, M in enumerate(ST):
            spk = pc["spaces"][f"t{k}"]
            pc["ST"].append(_local(M, spk.owned, spk.lookup(n), spk.n))
        # B: owned x rows read owned src entries only; B^T: owned src rows
        # read owned x entries only (same edge owner)
        lu = np.full(nu, -1, dtype=np.int64)
        lu[own_u[r]] = np.arange(own_u[r].size)
        lx = np.full(n, -1, dtype=np.int64)
        lx[own] = np.arange(own.size)
        pc["B"] = _local(B, own, lu, own_u[r].size)
        pc["BT"] = _local(BT, own_u[r], lx, own.size)
        pc["x0"] = x0[own].copy()
        pc["mass"] = np.asarray(s.cap, float)[own_u[r]]
        pc["Mnu"] = np.asarray(s.Mnu_diag, float)[own[(own >= nu) & (own < nu + mesh.n_faces)] - nu]
        sp0 = pc["spaces"]["s0"]
        pc["dropped0"] = list(sp0.dropped)
        srcs = []
        for w, f in s.sources:
            srcs.append(("node", np.asarray(w, float)[own_u[r]], f))
        pc["sources"] = srcs
        pc["probes"] = []
        spp = pc["spaces"]["p"]
        lp = spp.lookup(n)
        for P, po in zip(probes, probe_owner):
            rows = np.flatnonzero(po == r)
            pc["probes"].append((rows, _local(P, rows, lp, spp.n)))
        pc["dfts"] = [np.asarray(rec["freqs"], float) for rec in s.recorder.dfts]
        pc["dft0"] = [np.asarray(rec["data"])[:, own_u[r]].copy() for rec in s.recorder.dfts]
        out[r] = pc
    return out


# ================================================================ rank engines
class _Rank:
    """Local state and step program of one rank (no communication here)."""

    def __init__(self, pc):
        self.pc = pc
        self.rank = pc["rank"]
        self.kind = pc["kind"]
        self.dt = pc["dt"]
        self.n = pc["n"]
        self.spaces = pc["spaces"]
        self.use_sources = True
        self.samples = [[] for _ in pc["probes"]]
        self.dft = [d.copy() for d in pc["dft0"]]
        self.buf = {}

    # -------------------------------------------------- sources / monitors
    def _src(self, kind="node"):
        t_half = (self.n + 0.5) * self.dt
        n = self._n_src(kind)
        out = np.zeros(n)
        if not self.use_sources:
            return out
        for sk, w, f in self.pc["sources"]:
            if sk == kind:
                out += w * f(t_half)
        return out

    def _record(self, u_ext, u_own):
        self.n += 1
        t, dt = self.n * self.dt, self.dt
        for k, (_, P) in enumerate(self.pc["probes"]):
            self.samples[k].append(P @ u_ext)
        for k, fr in enumerate(self.pc["dfts"]):
            self.dft[k] += np.exp(-2j * np.pi * fr * t)[:, None] * u_own[None] * dt


class _FieldRank(_Rank):
    """3D Voronoi / 2D TM / 2D TE leapfrog on [owned | halo] vectors."""

    def __init__(self, pc):
        super().__init__(pc)
        self.su, self.sf = self.spaces["u"], self.spaces["f"]
        self.buf["u"] = pc["u0"].copy()
        self.buf["f"] = pc["f0"].copy()
        self.buf["ru"] = np.zeros(self.su.n)  # transpose: r-bar on u
        self.buf["rf"] = np.zeros(self.sf.n)  # transpose: b-bar on f
        self.cap = pc.get("cap")
        nu, nf = self.su.n_own, self.sf.n_own
        self._nu, self._nf = nu, nf
        if self.kind in ("3d", "tm"):
            self.program = [self._flux, ("f", "f"), self._ampere, ("u", "u"), self._rec]
        else:
            self.program = [self._te_h, ("u", "u"), self._te_d, ("f", "f"), self._rec]
        self.program_T = [self._t1, ("u", "ru"), self._t2, ("f", "rf"), self._t3]

    def _n_src(self, kind):
        return self._nu if kind == "node" else self._nf

    def _flux(self):
        pc, dt = self.pc, self.dt
        u, b, nf = self.buf["u"], self.buf["f"], self._nf
        if pc["ce"] is None:
            b[:nf] -= dt * (pc["A"] @ u)
        else:
            ce = pc["ce"]
            b[:nf] = ((1 - ce) * b[:nf] - dt * (pc["A"] @ u)) / (1 + ce)

    def _ampere(self):
        pc = self.pc
        u, b = self.buf["u"], self.buf["f"]
        r = pc["BT"] @ (pc["wf"] * b) - self._src("node")
        u[: self._nu] = self.cap.step(u[: self._nu], r)

    def _te_h(self):
        pc, dt, nu = self.pc, self.dt, self._nu
        u, v = self.buf["u"], self.buf["f"]
        vol = pc["wf"] * v
        u[:nu] = (pc["a"] * u[:nu] - dt * (pc["BT"] @ vol + self._src("node"))) / pc["bb"]

    def _te_d(self):
        pc, dt, nf = self.pc, self.dt, self._nf
        u, v = self.buf["u"], self.buf["f"]
        es = self._src("edge")
        if pc["ce"] is None:
            v[:nf] += dt * (pc["A"] @ u - es)
        else:
            ce = pc["ce"]
            v[:nf] = ((1 - ce) * v[:nf] + dt * (pc["A"] @ u - es)) / (1 + ce)

    def _rec(self):
        self._record(self.buf["u"], self.buf["u"][: self._nu])

    # -------------------------------------------------- state
    def state(self):
        return {"u": self.buf["u"][: self._nu].copy(), "f": self.buf["f"][: self._nf].copy()}

    def owned_index(self):
        return {"u": self.su.owned, "f": self.sf.owned}

    def set_state(self, glob):
        """Owned and halo entries from global vectors {'u', 'f'}."""
        self.buf["u"][:] = glob["u"][np.r_[self.su.owned, self.su.halo]]
        self.buf["f"][:] = glob["f"][np.r_[self.sf.owned, self.sf.halo]]
        for p in self.su.dropped:
            self.buf["u"][p] = 0.0
        if self.cap is not None and self.cap.groups:
            # ADE states are not part of the linear-map state here
            self.cap.reset()

    def energy_partial(self):
        u, b = self.buf["u"][: self._nu], self.buf["f"][: self._nf]
        return 0.5 * self.pc["mass"] @ u**2 + 0.5 * self.pc["wf"][: self._nf] @ b**2

    # -------------------------------------------------- transpose (source-free)
    def set_adjoint(self, lam):
        """Owned entries of the global adjoint vectors {'u', 'f'}."""
        self._lam = {"u": np.asarray(lam["u"], float)[self.su.owned],
                     "f": np.asarray(lam["f"], float)[self.sf.owned]}  # fmt: skip

    def _t1(self):
        if self.kind == "te":
            raise NotImplementedError("MPI transpose: 2D TE")
        if self.cap.groups:
            raise NotImplementedError("MPI transpose with dispersive media")
        acc = self._lam["u"] / self.cap.den
        self._ub = np.zeros(self._nu)
        self._ub += self.cap.a1 * acc
        self.buf["ru"][: self._nu] = self.cap.dt * acc

    def _t2(self):
        pc = self.pc
        h = pc["B"] @ self.buf["ru"]
        self.buf["rf"][: self._nf] = self._lam["f"] + pc["wf"][: self._nf] * h

    def _t3(self):
        pc, dt = self.pc, self.dt
        bt = self.buf["rf"]
        if self.kind == "3d":
            ub = self._ub - dt * (pc["AT"] @ bt)
            fb = bt[: self._nf].copy()
        elif pc["ce"] is None:
            ub = self._ub - pc["AT"] @ (dt * bt)
            fb = bt[: self._nf].copy()
        else:
            # needs the f halo of v1b_tot / (1 + ce): ce on the halo
            raise NotImplementedError("MPI transpose: 2D absorber")
        self._out_T = {"u": ub, "f": fb}


class _StageRank(_Rank):
    """Sparse-stage step x <- S4 (S3 S2 S1 x + B src) of `PML3D`."""

    def __init__(self, pc):
        super().__init__(pc)
        self.x = pc["x0"].copy()
        self._nu = pc["n_u_own"]
        for k in range(4):
            self.buf[f"s{k}"] = np.zeros(self.spaces[f"s{k}"].n)
            self.buf[f"t{k}"] = np.zeros(self.spaces[f"t{k}"].n)
        self.buf["p"] = np.zeros(self.spaces["p"].n)
        self.program = []
        for k in range(4):
            self.program += [self._loader(f"s{k}", lambda: self.x), (f"s{k}", f"s{k}"),
                             self._stage(k)]  # fmt: skip
        self.program += [self._loader("p", lambda: self.x), ("p", "p"), self._rec]
        self.program_T = []
        for k in (3, 2, 1, 0):
            self.program_T += [self._loader(f"t{k}", lambda: self._y), (f"t{k}", f"t{k}"),
                               self._stage_T(k)]  # fmt: skip

    def _n_src(self, kind):
        return self.pc["mass"].size

    def _loader(self, key, get):
        sp_ = self.spaces[key]

        def load():
            self.buf[key][: sp_.n_own] = get()
            for p in sp_.dropped:
                self.buf[key][p] = 0.0

        return load

    def _stage(self, k):
        def run():
            y = self.pc["S"][k] @ self.buf[f"s{k}"]
            if k == 2:
                y = y + self.pc["B"] @ self._src("node")
            self.x = y

        return run

    def _stage_T(self, k):
        def run():
            y = self.pc["ST"][k] @ self.buf[f"t{k}"]
            if k == 3:
                self._gsrc = self.pc["BT"] @ y
            self._y = y

        return run

    def _rec(self):
        self._record(self.buf["p"], self.x[: self._nu])

    def state(self):
        return {"x": self.x.copy()}

    def owned_index(self):
        return {"x": self.pc["own_x"]}

    def set_state(self, glob):
        self.x = np.asarray(glob["x"], float)[self.pc["own_x"]].copy()

    def set_adjoint(self, lam):
        self._y = np.asarray(lam["x"], float)[self.pc["own_x"]].copy()

    def energy_partial(self):
        nu, nf = self._nu, self.pc["Mnu"].size
        u = self.x[:nu]
        b = self.x[nu : nu + nf]
        return 0.5 * self.pc["mass"] @ u**2 + 0.5 * self.pc["Mnu"] @ b**2

    @property
    def _out_T(self):
        return {"x": self._y, "src": self._gsrc}


def _make_rank(pc):
    return _StageRank(pc) if pc["kind"] == "pml3d" else _FieldRank(pc)


# ================================================================ drivers
class _Driver:
    """Common API of `LocalCluster` and `DistributedFIT`."""

    def run(self, n_steps):
        """Advance `n_steps` steps."""
        for _ in range(int(n_steps)):
            self.step()

    def apply(self, x):
        """
        One source-free step of global state `x` (dict of global vectors,
        keys ``u``, ``f`` or ``x`` for the PML); no monitors. Returns the
        new global state (on rank 0 under MPI). The ADE states of dispersive
        groups are reset (not part of this map).
        """
        self._each(lambda r: r.set_state(x))
        self._each(lambda r: setattr(r, "use_sources", False))
        saved = self._each(lambda r: (r.n, r.samples, r.dft))
        self._each(lambda r: setattr(r, "_record", lambda *a: None))
        try:
            self.step()
        finally:
            self._each(lambda r: r.__dict__.pop("_record", None))
            self._each(lambda r: setattr(r, "use_sources", True))
        self._restore(saved)
        return self.gather_state()

    def apply_T(self, lam):
        """
        Exact transpose of `apply` (source-free step) on global `lam`.

        Returns the global adjoint state (dict like `lam`; the PML returns
        also ``src``, the sensitivity to the source vector).
        """
        self._each(lambda r: r.set_adjoint(lam))
        self._run_program("program_T")
        return self._gather_dict(lambda r: r._out_T, lambda r: dict(
            r.owned_index(), **({"src": r.spaces["s0"].owned[: r._nu]}
                                if r.kind == "pml3d" else {})))  # fmt: skip


class LocalCluster(_Driver):
    """
    All ranks of a partition stepped in one process (no MPI).

    Executes exactly the per-rank arithmetic and halo copies of
    `DistributedFIT`; used by the tests and for partition studies.

    Parameters
    ----------
    solver : configured serial solver
    n_parts : int
    parts : ndarray, optional
        Cell partition.
    partitioner : str
        `partition_cells` method.
    """

    def __init__(self, solver, n_parts, parts=None, partitioner="auto", _mutation=None):
        pieces, self.parts = build_pieces(solver, n_parts, parts, partitioner,
                                          _mutation=_mutation)  # fmt: skip
        self.ranks = [_make_rank(pc) for pc in pieces]
        self.n_parts = n_parts
        self.kind = self.ranks[0].kind
        self._n_u = pieces[0]["n_u"]

    @property
    def n(self):
        return self.ranks[0].n

    def _each(self, fn):
        return [fn(r) for r in self.ranks]

    def _restore(self, saved):
        for r, (n, s, d) in zip(self.ranks, saved):
            r.n, r.samples, r.dft = n, s, d

    def _exchange(self, key, bkey):
        for r in self.ranks:
            sp_r = r.spaces[key]
            for q, pos in sp_r.recv.items():
                src = self.ranks[q]
                r.buf[bkey][sp_r.n_own + pos] = src.buf[bkey][src.spaces[key].send[r.rank]]

    def _run_program(self, name):
        prog0 = getattr(self.ranks[0], name)
        for k, item in enumerate(prog0):
            if callable(item):
                for r in self.ranks:
                    getattr(r, name)[k]()
            else:
                self._exchange(*item)

    def step(self):
        """One step on every rank."""
        self._run_program("program")

    def _gather_dict(self, get, idx):
        vals = [get(r) for r in self.ranks]
        inds = [idx(r) for r in self.ranks]
        out = {}
        for k in vals[0]:
            n = sum(v[k].size for v in vals)
            g = np.zeros(n)
            for v, i in zip(vals, inds):
                g[i[k]] = v[k]
            out[k] = g
        return out

    def gather_state(self):
        """Global state vectors ({'u', 'f'} or {'x'})."""
        return self._gather_dict(lambda r: r.state(), lambda r: r.owned_index())

    def gather(self, key):
        return self.gather_state()[key]

    def energy(self):
        """Sum of the per-rank energies (owned dofs) [J]."""
        return float(sum(r.energy_partial() for r in self.ranks))

    def probe_signal(self, i=0):
        """(n_steps, n_points) samples of probe i, rows in serial order."""
        rows = [r.pc["probes"][i][0] for r in self.ranks]
        n_pts = sum(x.size for x in rows)
        n_t = len(self.ranks[0].samples[i])
        out = np.zeros((n_t, n_pts))
        for r, rw in zip(self.ranks, rows):
            if rw.size:
                out[:, rw] = np.asarray(r.samples[i]).reshape(n_t, rw.size)
        return out

    def dft_data(self, i=0):
        """(n_freqs, n_u) DFT of probe-free u dofs, serial order."""
        nf = self.ranks[0].dft[i].shape[0]
        out = np.zeros((nf, self._n_u), complex)
        for r in self.ranks:
            idx = r.owned_index()
            own_u = idx["u"] if "u" in idx else idx["x"][: r._nu]
            out[:, own_u] = r.dft[i]
        return out

    def comm_stats(self):
        """Halo entries per rank and space (doubles received per exchange)."""
        return [{k: s.halo_size for k, s in r.spaces.items()} for r in self.ranks]


class DistributedFIT(_Driver):
    """
    One rank of an MPI-distributed unstructured FIT run.

    Parameters
    ----------
    solver : configured serial solver or None
        ``mode="scatter"``: needed on `root` only (others pass None); the
        root builds every piece and scatters them (source amplitude
        functions must be picklable: module-level functions or callable
        objects). ``mode="replicated"``: every rank passes an identically
        built solver and builds its own piece (no pickling).
    comm : mpi4py communicator
    parts : ndarray, optional
        Cell partition (default: computed on `root`, broadcast).
    partitioner : str
    mode : {"scatter", "replicated"}
    timers : bool
        Accumulate wall time in compute and exchange phases (``self.t``).
    """

    def __init__(self, solver, comm, parts=None, partitioner="auto", mode="scatter",
                 root=0, timers=False, _mutation=None):  # fmt: skip
        from mpi4py import MPI

        self._MPI = MPI
        self.comm, self.root = comm, root
        P, me = comm.Get_size(), comm.Get_rank()
        self.n_parts = P
        if mode == "replicated":
            if parts is None:
                parts = (partition_cells(solver.mesh, P, partitioner) if me == root else None)
                parts = comm.bcast(parts, root=root)
            pieces, self.parts = build_pieces(solver, P, parts, ranks=[me], _mutation=_mutation)
            pc = pieces[me]
        elif mode == "scatter":
            pieces = None
            if me == root:
                pieces, _ = build_pieces(solver, P, parts, partitioner, _mutation=_mutation)
            pc = comm.scatter(pieces, root=root)
            self.parts = pc["parts"]
        else:
            raise ValueError("mode must be 'scatter' or 'replicated'")
        self.r = _make_rank(pc)
        self.kind = self.r.kind
        self._n_u = pc["n_u"]
        self.timers = timers
        self.t = {"compute": 0.0, "exchange": 0.0}
        self.bytes_sent = 0

    @property
    def n(self):
        return self.r.n

    def _each(self, fn):
        return [fn(self.r)]

    def _restore(self, saved):
        self.r.n, self.r.samples, self.r.dft = saved[0]

    def _exchange(self, key, bkey):
        MPI = self._MPI
        sp_ = self.r.spaces[key]
        buf = self.r.buf[bkey]
        reqs, rbufs, sbufs = [], {}, []
        for q, pos in sp_.recv.items():
            rbufs[q] = np.empty(pos.size)
            reqs.append(self.comm.Irecv(rbufs[q], source=q, tag=sp_.tag))
        for q, pos in sp_.send.items():
            sb = np.ascontiguousarray(buf[pos])
            sbufs.append(sb)
            reqs.append(self.comm.Isend(sb, dest=q, tag=sp_.tag))
            self.bytes_sent += sb.nbytes
        MPI.Request.Waitall(reqs)
        for q, pos in sp_.recv.items():
            buf[sp_.n_own + pos] = rbufs[q]

    def _run_program(self, name):
        tm = time.perf_counter
        for item in getattr(self.r, name):
            t0 = tm() if self.timers else 0.0
            if callable(item):
                item()
                if self.timers:
                    self.t["compute"] += tm() - t0
            else:
                self._exchange(*item)
                if self.timers:
                    self.t["exchange"] += tm() - t0

    def step(self):
        self._run_program("program")

    def _gather_dict(self, get, idx):
        parts = self.comm.gather((get(self.r), idx(self.r)), root=self.root)
        if self.comm.Get_rank() != self.root:
            return None
        out = {}
        for k in parts[0][0]:
            n = sum(v[k].size for v, _ in parts)
            g = np.zeros(n)
            for v, i in parts:
                g[i[k]] = v[k]
            out[k] = g
        return out

    def gather_state(self):
        return self._gather_dict(lambda r: r.state(), lambda r: r.owned_index())

    def gather(self, key):
        st = self.gather_state()
        return None if st is None else st[key]

    def energy(self):
        """Global energy (Allreduce of the owned partial sums) [J]."""
        return self.comm.allreduce(float(self.r.energy_partial()))

    def probe_signal(self, i=0):
        rows, P = self.r.pc["probes"][i]
        data = np.asarray(self.r.samples[i]).reshape(len(self.r.samples[i]), rows.size)
        parts = self.comm.gather((rows, data), root=self.root)
        if self.comm.Get_rank() != self.root:
            return None
        n_pts = sum(rw.size for rw, _ in parts)
        out = np.zeros((data.shape[0], n_pts))
        for rw, d in parts:
            if rw.size:
                out[:, rw] = d
        return out

    def dft_data(self, i=0):
        idx = self.r.owned_index()
        own_u = idx["u"] if "u" in idx else idx["x"][: self.r._nu]
        parts = self.comm.gather((own_u, self.r.dft[i]), root=self.root)
        if self.comm.Get_rank() != self.root:
            return None
        out = np.zeros((parts[0][1].shape[0], self._n_u), complex)
        for o, d in parts:
            out[:, o] = d
        return out

    def comm_stats(self):
        st = {k: s.halo_size for k, s in self.r.spaces.items()}
        return self.comm.gather(st, root=self.root)
