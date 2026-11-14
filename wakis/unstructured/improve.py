# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Time-step-driven node relocation for tetrahedral meshes (Voronoi FIT).

For the diagonal (Voronoi) Hodge, the stability limit of the leapfrog is
set, to within a few per cent on the meshes we measured, by the largest
edge rate

    rate_e = K_ee / M_e = (sum_{f > e} M_nu,f) / M_eps,e
           = L_e sum_{f > e} (H_f / A_f) / (eps_0 Aeps_e),

H_f = sum_T h_{f,T} / mu_T the material-weighted dual edge length of face f,
A_f its area and Aeps_e = sum_T eps_T A_{e,T} the material-weighted dual
face area of edge e (lambda_max >= max_e rate_e by the Rayleigh quotient of
a unit vector; the element bound of `wakis.unstructured.quality` gives the
matching upper bound). Edges with a nearly degenerate Voronoi face
(Aeps_e -> 0: four nearly co-spherical nodes, "slivers" of the Delaunay
mesh) have huge rates; the solver eliminates those below
``massless_tol * L_e^2`` and the next ones set dt.

`improve_dt` moves interior nodes (boundary and material-interface nodes
stay fixed, so the geometry and the material regions are unchanged; the
connectivity is kept) to minimize the largest edge rate in their
neighbourhood: greedy min-max local search over the optimal-Delaunay (ODT)
point of the node (Chen & Xu 2004) and random trial points, accepting a move
lexicographically better: first fewer faces with a negative dual length
around the node (non-Delaunay faces make K indefinite, an instability no
time step cures), then a smaller largest rate among the edges it touches;
tetrahedra must keep a positive volume. Every
quantity a move changes lives on the edges of the tetrahedra around the
node, so each trial costs O(1) and the search is exact for the unclipped
dual measures (material-interface clipping is re-applied when the mesh is
rebuilt; interface nodes do not move).
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0

from ._core import cell_materials
from .mesh import SimplexMesh

__all__ = [
    "tet_partials",
    "edge_rates",
    "movable_nodes",
    "constraint_faces",
    "planar_projector",
    "improve_dt",
]

LOC_E = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])


def _cc_tri(P):
    a, b = P[..., 1, :] - P[..., 0, :], P[..., 2, :] - P[..., 0, :]
    aa, bb, ab = (a * a).sum(-1), (b * b).sum(-1), (a * b).sum(-1)
    den = 2 * (aa * bb - ab**2)
    return (
        P[..., 0, :]
        + (bb * (aa - ab) / den)[..., None] * a
        + (aa * (bb - ab) / den)[..., None] * b
    )


def tet_partials(P):
    """
    Circumcentric partial dual measures of tetrahedra.

    Parameters
    ----------
    P : ndarray (n, 4, 3)
        Vertex coordinates [m].

    Returns
    -------
    A : ndarray (n, 6)
        Signed part of the dual face of each local edge (order
        (01, 02, 03, 12, 13, 23)) inside the tetrahedron [m^2].
    h : ndarray (n, 4)
        Signed distance from the circumcentre of the face opposite local
        vertex k to the tetrahedron circumcentre (part of the dual edge) [m].
    area : ndarray (n, 4)
        Face areas [m^2].
    vol : ndarray (n,)
        Signed volume [m^3].

    Same formulas as `SimplexMesh` (before material-interface clipping).
    """
    E = P[:, 1:] - P[:, :1]
    vol = np.linalg.det(E) / 6
    rhs = 0.5 * np.einsum("nij,nij->ni", E, E)
    cT = P[:, 0] + np.linalg.solve(E, rhs[..., None])[..., 0]
    others = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    cf, h, area = [], np.empty(P.shape[:1] + (4,)), np.empty(P.shape[:1] + (4,))
    for k in range(4):
        F = P[:, others[k]]
        c = _cc_tri(F)
        nv = np.cross(F[:, 1] - F[:, 0], F[:, 2] - F[:, 0])
        a2 = np.linalg.norm(nv, axis=1)
        area[:, k] = 0.5 * a2
        nv = nv / a2[:, None]
        sgn = np.sign(np.einsum("ij,ij->i", nv, P[:, k] - c))
        h[:, k] = sgn * np.einsum("ij,ij->i", nv, cT - c)
        cf.append(c)
    A = np.zeros(P.shape[:1] + (6,))
    for p, (a, b) in enumerate(LOC_E):
        m = 0.5 * (P[:, a] + P[:, b])
        tv = P[:, b] - P[:, a]
        tv = tv / np.linalg.norm(tv, axis=1)[:, None]
        for o in set(range(4)) - {a, b}:
            third = ({0, 1, 2, 3} - {a, b, o}).pop()
            w = P[:, third] - m
            w = w - np.einsum("ij,ij->i", w, tv)[:, None] * tv
            w = w / np.linalg.norm(w, axis=1)[:, None]
            s_ef = np.einsum("ij,ij->i", w, cf[o] - m)
            A[:, p] += 0.5 * s_ef * h[:, o]
    return A, h, area, vol


def constraint_faces(mesh):
    """Boolean mask of faces on the outer boundary or on a material interface."""
    f = mesh.cell_faces.ravel()
    t = np.repeat(mesh.cell_tags, 4)
    lo = np.full(mesh.n_faces, np.iinfo(np.int64).max)
    hi = np.full(mesh.n_faces, np.iinfo(np.int64).min)
    np.minimum.at(lo, f, t)
    np.maximum.at(hi, f, t)
    mask = lo != hi
    mask[mesh.boundary_faces] = True
    return mask


def planar_projector(mesh, tol=1e-9):
    """
    Sliding rule for nodes on flat parts of the boundary and interfaces.

    A constrained node (on the outer boundary or a material interface) whose
    constraint faces are all coplanar may slide in that plane: the geometry
    and the material regions are unchanged. Nodes on curved surfaces, edges
    and corners stay fixed (pass a CAD projector to `improve_dt` to slide
    them on curved surfaces).

    Returns
    -------
    project : callable
        ``project(n, Y)`` -> Y projected on the plane of node n, or None if
        node n must not move.
    """
    cf = constraint_faces(mesh)
    fid = np.flatnonzero(cf)
    rows = mesh.faces[fid].ravel()
    nf = sp.csr_matrix(
        (np.ones(rows.size), (rows, np.repeat(fid, 3))), shape=(mesh.n_nodes, mesh.n_faces)
    ).tocsr()
    normal = mesh.face_normal
    plane = {}
    for n in np.unique(rows):
        fs = nf.indices[nf.indptr[n] : nf.indptr[n + 1]]
        nv = normal[fs]
        if np.all(np.abs(nv @ nv[0]) > 1 - tol):
            plane[int(n)] = nv[0]

    def project(n, Y):
        v = plane.get(int(n))
        if v is None:
            return None
        x0 = mesh.nodes[n]
        return Y - np.outer((Y - x0) @ v, v)

    project.n_sliding = len(plane)
    return project


def movable_nodes(mesh):
    """Nodes off the outer boundary and off every material interface."""
    fix = np.zeros(mesh.n_nodes, bool)
    fix[mesh.boundary_nodes] = True
    f = mesh.cell_faces.ravel()
    t = np.repeat(mesh.cell_tags, 4)
    lo = np.full(mesh.n_faces, np.iinfo(np.int64).max)
    hi = np.full(mesh.n_faces, np.iinfo(np.int64).min)
    np.minimum.at(lo, f, t)
    np.maximum.at(hi, f, t)
    fix[mesh.faces[lo != hi].ravel()] = True
    return ~fix


class _State:
    """Incrementally updated dual sums of a tetrahedral mesh."""

    def __init__(self, mesh, eps_r, mu_r, live, free_edge):
        self.mesh = mesh
        self.X = mesh.nodes.copy()
        self.cells = mesh.cells
        self.ce, self.cf = mesh.cell_edges, mesh.cell_faces
        self.eps = eps_r * live
        self.inv_mu = live / mu_r
        self.free_edge = free_edge
        nc = mesh.n_cells
        A, h, area, vol = tet_partials(self.X[self.cells])
        self.A_eT = A * self.eps[:, None]
        self.h_fT = h * self.inv_mu[:, None]
        self.vol = vol
        self.Aeps = np.zeros(mesh.n_edges)
        np.add.at(self.Aeps, self.ce.ravel(), self.A_eT.ravel())
        self.H = np.zeros(mesh.n_faces)
        np.add.at(self.H, self.cf.ravel(), self.h_fT.ravel())
        self.Af = mesh.face_area.copy()
        self.L = mesh.edge_length.copy()
        # face -> edges and edge -> faces
        self.face_edges = mesh.face_edges
        fe = sp.csr_matrix(
            (np.ones(3 * mesh.n_faces), (np.repeat(np.arange(mesh.n_faces), 3),
                                          mesh.face_edges.ravel())),
            shape=(mesh.n_faces, mesh.n_edges),
        )  # fmt: skip
        self.edge_faces = fe.T.tocsr()
        self.Kd = np.asarray(fe.T @ (self.H / self.Af)).ravel()
        # node -> cells
        nt = sp.csr_matrix(
            (np.ones(4 * nc), (self.cells.ravel(), np.repeat(np.arange(nc), 4))),
            shape=(mesh.n_nodes, nc),
        )
        self.node_cells = nt.tocsr()
        self.vol_ref = np.abs(vol).mean()
        # Faces that couple free edges: negative H makes K indefinite
        self.face_free = np.asarray(fe @ free_edge.astype(float)).ravel() > 0

    def rates(self, e=None):
        e = np.arange(self.mesh.n_edges) if e is None else e
        with np.errstate(divide="ignore", invalid="ignore"):
            r = self.Kd[e] * self.L[e] / (mu_0 * eps_0 * self.Aeps[e])
        r = np.where(self.Aeps[e] > 0, r, np.inf)
        return np.where(self.free_edge[e], r, 0.0)

    def trial(self, n, Y):
        """
        Rates of the affected edges if node n moves to each row of Y.

        Returns (max_rate (m,), ok (m,), payload) for the m trial points.
        """
        T = self.node_cells.indices[self.node_cells.indptr[n] : self.node_cells.indptr[n + 1]]
        E = np.unique(self.ce[T].ravel())
        F = np.unique(self.cf[T].ravel())
        m, nt = len(Y), len(T)
        P = np.broadcast_to(self.X[self.cells[T]], (m, nt, 4, 3)).copy()
        loc = self.cells[T] == n  # (nt, 4)
        P[:, loc] = np.repeat(Y[:, None, :], loc.sum(), axis=1)
        A, h, area, vol = tet_partials(P.reshape(-1, 4, 3))
        A = A.reshape(m, nt, 6) * self.eps[T][None, :, None]
        h = h.reshape(m, nt, 4) * self.inv_mu[T][None, :, None]
        area = area.reshape(m, nt, 4)
        vol = vol.reshape(m, nt)
        # local index maps
        eidx = np.searchsorted(E, self.ce[T])  # (nt, 6)
        fidx = np.searchsorted(F, self.cf[T])  # (nt, 4)
        Aeps = np.broadcast_to(self.Aeps[E], (m, len(E))).copy()
        H = np.broadcast_to(self.H[F], (m, len(F))).copy()
        for k in range(6):
            np.add.at(Aeps, (slice(None), eidx[:, k]), A[:, :, k] - self.A_eT[T, k])
        for k in range(4):
            np.add.at(H, (slice(None), fidx[:, k]), h[:, :, k] - self.h_fT[T, k])
        Af = np.broadcast_to(self.Af[F], (m, len(F))).copy()
        for k in range(4):
            Af[:, fidx[:, k]] = area[:, :, k]
        # Edge lengths
        Le = np.broadcast_to(self.L[E], (m, len(E))).copy()
        ed = self.mesh.edges[E]
        has = (ed == n).any(1)
        other = np.where(ed[:, 0] == n, ed[:, 1], ed[:, 0])[has]
        Le[:, has] = np.linalg.norm(Y[:, None, :] - self.X[other][None], axis=2)
        # K_ee change from the faces in F that contain e
        Mf_new = H / Af
        Mf_old = self.H[F] / self.Af[F]
        dK = np.zeros((m, len(E)))
        fe = np.searchsorted(E, self.face_edges[F])  # (len F, 3)
        for k in range(3):
            np.add.at(dK, (slice(None), fe[:, k]), Mf_new - Mf_old)
        Kd = self.Kd[E][None] + dK
        free = self.free_edge[E]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = Kd * Le / (mu_0 * eps_0 * Aeps)
        r = np.where(Aeps > 0, r, np.inf)
        r = np.where(free[None], r, 0.0)
        ok = (vol > 1e-3 * self.vol_ref).all(1)
        # faces with a negative dual length that couple free edges (K indefinite)
        n_neg = ((H < 0) & self.face_free[F][None]).sum(1)
        payload = (T, E, F, A, h, area, vol, Aeps, H, Af, Le, Kd, n_neg)
        return r.max(1), ok, payload

    def accept(self, n, y, j, payload):
        T, E, F, A, h, area, vol, Aeps, H, Af, Le, Kd, _ = payload
        self.X[n] = y
        self.A_eT[T] = A[j]
        self.h_fT[T] = h[j]
        self.vol[T] = vol[j]
        self.Aeps[E] = Aeps[j]
        self.H[F] = H[j]
        self.Af[F] = Af[j]
        self.L[E] = Le[j]
        self.Kd[E] = Kd[j]

    def odt_point(self, n):
        T = self.node_cells.indices[self.node_cells.indptr[n] : self.node_cells.indptr[n + 1]]
        P = self.X[self.cells[T]]
        E = P[:, 1:] - P[:, :1]
        rhs = 0.5 * np.einsum("nij,nij->ni", E, E)
        cT = P[:, 0] + np.linalg.solve(E, rhs[..., None])[..., 0]
        w = np.abs(self.vol[T])
        return (w[:, None] * cT).sum(0) / w.sum()


def edge_rates(mesh, materials=None, free=None):
    """
    Unclipped edge rates K_ee / M_ee [1/s^2] of the Voronoi FIT operator.

    Parameters
    ----------
    mesh : SimplexMesh
    materials : dict, optional
        As for `UnstructuredFIT3D`.
    free : ndarray of bool (n_edges,), optional
        Edges that carry a dof (default: interior edges not touching PEC).
    """
    eps_r, mu_r, _, pec, _ = cell_materials(mesh, materials)
    live = (~pec).astype(float)
    if free is None:
        free = _default_free(mesh, pec)
    st = _State(mesh, eps_r, mu_r, live, free)
    return st.rates()


def _default_free(mesh, pec, bc="pec"):
    fixed = np.zeros(mesh.n_edges, bool)
    fixed[mesh.cell_edges[pec].ravel()] = True
    if bc == "pec":
        fixed[mesh.boundary_edges] = True
    touched = np.zeros(mesh.n_edges, bool)
    touched[mesh.cell_edges[~pec].ravel()] = True
    return ~fixed & touched


def improve_dt(
    mesh,
    materials=None,
    bc="pec",
    n_sweeps=6,
    target_quantile=0.98,
    n_random=24,
    step=0.25,
    seed=0,
    massless_tol=1e-2,
    project="planar",
    verbose=False,
):
    """
    Relocate interior nodes to lower the largest edge rates (raise dt).

    Parameters
    ----------
    mesh : SimplexMesh
        Tetrahedral mesh (connectivity is kept).
    materials : dict, optional
        ``{tag: material}`` as for `UnstructuredFIT3D`.
    bc : {"pec", "pmc", "abc"}
        Outer boundary of the intended solver (PEC fixes the boundary edges).
    n_sweeps : int
        Number of sweeps over the target nodes.
    target_quantile : float
        Each sweep treats the nodes of the tetrahedra around the edges whose
        rate exceeds this quantile of the free-edge rates.
    n_random : int
        Random trial points per node and sweep (in a ball of radius
        ``step`` x mean incident edge length), besides the ODT point and
        halfway to it.
    step : float
        Trial radius in units of the local edge length.
    seed : int
        Random seed (the result is deterministic for a given seed).
    massless_tol : float or None
        Edges whose material-weighted dual area is below
        ``massless_tol * L_e**2`` are eliminated by `UnstructuredFIT3D`
        (same default, 1e-2) and are ignored in the objective, which then
        targets the edges that limit dt in the solver. None includes every
        edge (pushes all dual areas away from zero).
    project : "planar", callable or None
        Sliding rule for boundary/interface nodes: ``project(n, Y)`` returns
        the points Y (m, 3) projected on the constraint surface of node n,
        or None when node n is fixed. "planar" (default) uses
        `planar_projector`; None keeps every constrained node fixed;
        `mesh_cad` passes a projector onto the CAD surfaces.
    verbose : bool

    Returns
    -------
    new_mesh : SimplexMesh
        Same cells and tags, moved interior nodes.
    history : list of dict
        Per sweep: ``max_rate`` (largest free-edge rate, unclipped),
        ``moved`` (accepted moves), ``targets``.
    """
    rng = np.random.default_rng(seed)
    eps_r, mu_r, _, pec, _ = cell_materials(mesh, materials)
    live = (~pec).astype(float)
    free = _default_free(mesh, pec, bc)
    st = _State(mesh, eps_r, mu_r, live, free)
    interior = movable_nodes(mesh)
    if project == "planar":
        project = planar_projector(mesh)
    movable = interior.copy()
    if project is not None:
        for n in np.flatnonzero(~interior):
            if project(n, mesh.nodes[n : n + 1]) is not None:
                movable[n] = True
    edges = mesh.edges

    def objective_rates():
        r = st.rates()
        if massless_tol is not None:
            # edges the solver eliminates (material-weighted test of the solver)
            r = np.where(st.Aeps < massless_tol * st.L**2, 0.0, r)
        return r

    def neg_faces():
        return np.flatnonzero((st.H < 0) & st.face_free)

    history = []
    r = objective_rates()
    history.append(dict(max_rate=float(r[free].max()), moved=0, targets=0,
                        negative_faces=int(neg_faces().size)))  # fmt: skip
    for sweep in range(int(n_sweeps)):
        thr = np.quantile(r[free & np.isfinite(r)], target_quantile)
        bad = np.flatnonzero(r > thr)
        # edges of the non-Delaunay faces are targets too (they make K indefinite)
        nf = neg_faces()
        if nf.size:
            bad = np.union1d(bad, st.face_edges[nf].ravel())
        # nodes of the tetrahedra around the bad edges
        ec = sp.csr_matrix(
            (np.ones(mesh.cell_edges.size), (mesh.cell_edges.ravel(),
                                              np.repeat(np.arange(mesh.n_cells), 6))),
            shape=(mesh.n_edges, mesh.n_cells),
        )  # fmt: skip
        cells = np.unique(ec[bad].indices)
        nodes = np.unique(mesh.cells[cells].ravel())
        nodes = nodes[movable[nodes]]
        # worst first
        worst = np.zeros(mesh.n_nodes)
        np.maximum.at(worst, edges[bad, 0], np.minimum(r[bad], 1e300))
        np.maximum.at(worst, edges[bad, 1], np.minimum(r[bad], 1e300))
        if nf.size:
            worst[mesh.faces[nf].ravel()] = np.inf
        nodes = nodes[np.argsort(-worst[nodes], kind="stable")]
        moved = 0
        for n in nodes:
            x0 = st.X[n]
            T = st.node_cells.indices[st.node_cells.indptr[n] : st.node_cells.indptr[n + 1]]
            Eall = np.unique(st.ce[T].ravel())
            hloc = st.L[Eall].mean()
            y_odt = st.odt_point(n)
            d = rng.standard_normal((n_random, 3))
            d /= np.linalg.norm(d, axis=1)[:, None]
            rad = step * hloc * rng.random(n_random) ** (1 / 3)
            Y = np.vstack([x0, y_odt, 0.5 * (x0 + y_odt), x0 + d * rad[:, None]])
            if not interior[n]:
                Y = np.vstack([x0, project(n, Y[1:])])
            obj, ok, payload = st.trial(n, Y)
            if massless_tol is not None:
                E, Aeps, Le, Kd = payload[1], payload[7], payload[10], payload[11]
                with np.errstate(divide="ignore", invalid="ignore"):
                    rr = Kd * Le / (mu_0 * eps_0 * Aeps)
                rr = np.where(Aeps > 0, rr, np.inf)
                rr = np.where(Aeps < massless_tol * Le**2, 0.0, rr)
                rr = np.where(free[E][None], rr, 0.0)
                obj = rr.max(1)
            n_neg = np.where(ok, payload[12], np.iinfo(np.int64).max)
            obj = np.where(ok, obj, np.inf)
            # lexicographic: fewer non-Delaunay faces first, then the largest rate
            j = int(np.lexsort((obj, n_neg))[0])
            better = n_neg[j] < n_neg[0] or (
                n_neg[j] == n_neg[0] and obj[j] < obj[0] * (1 - 1e-9)
            )
            if j > 0 and better:
                st.accept(n, Y[j], j, payload)
                moved += 1
        r = objective_rates()
        history.append(
            dict(max_rate=float(r[free].max()), moved=moved, targets=int(nodes.size),
                 negative_faces=int(neg_faces().size))
        )  # fmt: skip
        if verbose:
            print(f"sweep {sweep}: max rate {history[-1]['max_rate']:.4e}, "
                  f"moved {moved}/{nodes.size}, negative faces "
                  f"{history[-1]['negative_faces']}")  # fmt: skip
        if moved == 0:
            break
    new = SimplexMesh(st.X, mesh.cells, mesh.cell_tags)
    for attr in ("region_names", "surface_groups"):
        if hasattr(mesh, attr):
            setattr(new, attr, getattr(mesh, attr))
    return new, history
