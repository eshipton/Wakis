# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
3D FIT on extruded prism meshes: a 2D triangulation times a 1D z grid.

The primal complex is the tensor product of a triangle mesh (nodes n, edges
e, triangles t; incidences G2 = edge-node, C2 = triangle-edge) with a 1D grid
of levels z_0 < ... < z_K (layers k + 1/2 between z_k and z_{k+1}):

==========================  ====================  ===========================
primal object               field                 dual object (Hodge)
==========================  ====================  ===========================
horizontal edge (e, l)      e_h  [V]              2D dual edge x dz*_l
vertical edge (n, k+1/2)    e_v  [V]              2D dual cell x (dz_k)
triangle face (t, l)        b_z  [Wb]             vertical segment dz*_l
vertical quad (e, k+1/2)    b_q  [Wb]             2D dual edge at mid layer
prism (t, k+1/2)            material              --
==========================  ====================  ===========================

with dz*_l the dual (half-sum) spacing at level l. The circumcentric (Voronoi)
2D dual is orthogonal to the triangles and the z dual is orthogonal to the
levels, so every dual edge is parallel to its primal face normal and every
dual face is perpendicular to its primal edge: the Hodge (material) matrices
are diagonal, exactly as for the Cartesian Yee/FIT grid. The curl is the
Kronecker assembly

    C = [[ C2 x I_L,     0      ],     (triangle faces)
         [ I_E2 x S,  G2 x I_K  ]]     (vertical quads)

with S the (K x L) level difference (S[k, k] = 1, S[k, k+1] = -1). With the
nodal gradient G = [[G2 x I_L], [I_N2 x Dz]], Dz = -S, C G = 0 exactly, so the
discrete magnetic flux is divergence free to round-off.

Material averaging (each prism holds one material, interfaces are unions of
primal faces): the capacitance of an edge sums eps over the parts of its dual
face in each prism (parallel averaging, exact for the tangential E carried by
the edges of a conforming mesh); the reluctance of a face sums h / mu over
the parts of its dual edge (series averaging, exact for the normal B).

See the physics guide, "Prism FIT (extruded triangle meshes)".
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from ..dispersive import DispersiveMaterial
from ._core import ETA_0, CapacitiveUpdate, check_dispersive_stability, max_eigenvalue


class PolygonComplex2D:
    """
    2D polygonal cell complex obtained by merging triangles across edges whose
    circumcentric dual edge is (nearly) degenerate.

    Two triangles sharing an edge e with dual length h_e = 0 are cocircular:
    their union is a quadrilateral with a single circumcentre, and the
    Voronoi-Delaunay pair stays orthogonal with e removed. For a small
    |h_e| <= merge_tol * L_e (or a locally non-Delaunay h_e < 0) the merge is
    a local O(h_e / L_e) perturbation of the dual geometry, which removes the
    nearly massless horizontal edges that would otherwise collapse the
    leapfrog timestep. Edges on material interfaces (tags differing in any
    layer) and boundary edges are never removed.

    The attributes mirror those of `SimplexMesh` used by `PrismFIT3D`:
    C (cell-edge), G (edge-node), edge_length, edge_midpoint, edge_tangent,
    dual_edge_partial (n_edges x n_cells), dual_area_partial,
    dual_area_barycentric (n_nodes x n_cells), cell_volume, boundary_edges,
    boundary_nodes, edges, nodes and the maps ``tri2cell`` (triangle ->
    polygon) and ``edge_map`` (triangle-mesh edge -> kept edge, -1 removed).
    """

    def __init__(self, m2, tags, merge_tol=0.0):
        nT, nE = m2.n_cells, m2.n_edges
        Hsum = np.asarray(m2.dual_edge_partial.sum(1)).ravel()
        bnd = np.zeros(nE, bool)
        bnd[m2.boundary_edges] = True
        # the two triangles of every interior edge
        ce = m2.cell_edges.ravel()
        tri = np.repeat(np.arange(nT), 3)
        o = np.argsort(ce, kind="stable")
        ce_s, tri_s = ce[o], tri[o]
        first = np.searchsorted(ce_s, np.arange(nE))
        cnt = np.bincount(ce_s, minlength=nE)
        t1 = tri_s[first]
        t2 = np.where(cnt == 2, tri_s[np.minimum(first + 1, ce_s.size - 1)], -1)
        same = np.zeros(nE, bool)
        inter = cnt == 2
        same[inter] = np.all(tags[t1[inter]] == tags[t2[inter]], axis=1)
        drop = (~bnd) & inter & same & (Hsum <= merge_tol * m2.edge_length)
        # Every removed edge needs a triangle whose two other edges are kept:
        # its (virtual) voltage is then the path sum through that triangle,
        # which keeps C G = 0 for the quad that the removed edge retains.
        cells, cell_edges = m2.cells, m2.cell_edges
        path = {}
        for _ in range(10):
            changed = False
            path = {}
            for e in np.flatnonzero(drop):
                a, b = m2.edges[e]
                for t in (t1[e], t2[e]):
                    others = [x for x in cell_edges[t] if x != e]
                    if not drop[others].any():
                        cn = [n for n in cells[t] if n != a and n != b][0]
                        terms = []
                        for x in others:
                            u, v = m2.edges[x]  # oriented u -> v (u < v)
                            # path a -> c -> b: +1 if traversed along u -> v
                            if {u, v} == {a, cn}:
                                terms.append((x, 1.0 if u == a else -1.0))
                            else:
                                terms.append((x, 1.0 if v == b else -1.0))
                        path[e] = terms
                        break
                else:
                    drop[e] = False
                    changed = True
            if not changed:
                break
        # union-find over triangles joined by dropped edges
        parent = np.arange(nT)

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for e in np.flatnonzero(drop):
            ra, rb = find(t1[e]), find(t2[e])
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        roots = np.array([find(i) for i in range(nT)])
        _, tri2cell = np.unique(roots, return_inverse=True)
        nP = tri2cell.max() + 1
        Mg = sp.csr_matrix((np.ones(nT), (tri2cell, np.arange(nT))), shape=(nP, nT))
        keep = np.flatnonzero(~drop)
        edge_map = -np.ones(nE, dtype=np.int64)
        edge_map[keep] = np.arange(keep.size)
        self.tri2cell, self.edge_map, self.kept_edges = tri2cell, edge_map, keep
        self.n_removed = int(drop.sum())
        self.nodes = m2.nodes
        self.edges = m2.edges[keep]
        self.C = (Mg @ m2.C)[:, keep].tocsr()
        self.C.eliminate_zeros()
        self.G = m2.G.tocsr()[keep]
        self.edge_length = m2.edge_length[keep]
        self.edge_midpoint = m2.edge_midpoint[keep]
        self.edge_tangent = m2.edge_tangent[keep]
        self.dual_edge_partial = (m2.dual_edge_partial.tocsr()[keep] @ Mg.T).tocsr()
        self.dual_area_partial = (m2.dual_area_partial @ Mg.T).tocsr()
        self.dual_area_barycentric = (m2.dual_area_barycentric @ Mg.T).tocsr()
        self.cell_volume = Mg @ m2.cell_volume
        self.cell_centroid = (Mg @ (m2.cell_volume[:, None] * m2.nodes[m2.cells].mean(1))) / self.cell_volume[:, None]
        # Quads (vertical faces) are kept for *all* 2D edges: a merged edge
        # loses its horizontal voltage e_h (nearly massless) but keeps its
        # vertical quad, whose flux couples the vertical edges through the
        # in-plane gradient only. This keeps the node (TM) Laplacian exact
        # (its cotangent weight h_e / L_e is not dropped); only the small
        # z-difference coupling of the removed e_h is lost.
        rem = np.flatnonzero(drop)
        qorder = np.r_[keep, rem]
        self.quad_edges = qorder  # triangle-mesh edge of every quad
        rows, cols, vals = [], [], []
        for i, e in enumerate(rem):
            for x, sgn in path[e]:
                rows.append(i)
                cols.append(edge_map[x])
                vals.append(sgn)
        # virtual voltages of the removed edges from the kept ones (n_rem x nk)
        self.P_removed = sp.csr_matrix((vals, (rows, cols)), shape=(rem.size, keep.size))
        self.n_quads = qorder.size
        self.G_quad = m2.G.tocsr()[qorder]
        self.dual_edge_partial_quad = (m2.dual_edge_partial.tocsr()[qorder] @ Mg.T).tocsr()
        self.edge_length_quad = m2.edge_length[qorder]
        self.edge_midpoint_quad = m2.edge_midpoint[qorder]
        self.boundary_edges = edge_map[m2.boundary_edges]
        self.boundary_nodes = m2.boundary_nodes
        first_tri = np.zeros(nP, dtype=np.int64)
        first_tri[tri2cell[::-1]] = np.arange(nT)[::-1]
        self.first_tri = first_tri
        # unsigned cell-edge and cell-node incidences
        self.cell_edge_abs = abs(self.C).tocsr()
        cn = (abs(Mg) @ sp.csr_matrix(
            (np.ones(3 * nT), (np.repeat(np.arange(nT), 3), m2.cells.ravel())),
            shape=(nT, m2.n_nodes),
        )).tocsr()
        cn.data[:] = 1.0
        self.cell_node_abs = cn

    @property
    def n_cells(self):
        return self.cell_volume.size

    @property
    def n_edges(self):
        return self.edge_length.size

    @property
    def n_nodes(self):
        return self.nodes.shape[0]


class PrismMesh:
    """
    Extrusion of a triangular `SimplexMesh` along z.

    Parameters
    ----------
    mesh2d : SimplexMesh
        2D triangulation of the cross-section (x, y) [m].
    z : array_like
        Increasing z levels [m] (K + 1 values, K layers); non-uniform spacing
        is allowed. For a periodic z direction the last level is identified
        with the first (period z[-1] - z[0]).
    tags : array_like of int (n_tri, K) or callable, optional
        Material tag of every prism. A callable is called as
        ``tags(tri_tags, z_mid)`` with the 2D triangle tags (n_tri,) and the
        layer mid-points (K,) and must return an (n_tri, K) array (it is
        broadcast). Default: the 2D tags in every layer.
    periodic_z : bool, optional
        Identify the top and bottom levels.
    merge_tol : float, optional
        Merge the two triangles of every non-interface edge with dual length
        h_e <= merge_tol * L_e into one polygonal cell (see
        `PolygonComplex2D`); 0 merges only exactly cocircular pairs.

    Attributes
    ----------
    n_levels : int
        Number of distinct levels L (K + 1, or K if periodic).
    dz : ndarray (K,)
        Layer thicknesses [m].
    dz_dual : ndarray (L,)
        Dual spacing at each level [m] (half sum of the adjacent layers).
    """

    def __init__(self, mesh2d, z, tags=None, periodic_z=False, merge_tol=0.05):
        if mesh2d.dim != 2:
            raise ValueError("PrismMesh extrudes a 2D triangular mesh")
        z = np.asarray(z, dtype=float)
        if z.ndim != 1 or z.size < 2 or np.any(np.diff(z) <= 0):
            raise ValueError("z must be an increasing 1D array with >= 2 levels")
        self.m2, self.z, self.periodic_z = mesh2d, z, bool(periodic_z)
        self.dz = np.diff(z)
        K = self.dz.size
        self.K = K
        self.z_mid = 0.5 * (z[1:] + z[:-1])
        if self.periodic_z:
            self.n_levels = K
            self.dz_dual = 0.5 * (self.dz + np.roll(self.dz, 1))
            self.z_level = z[:-1]
        else:
            self.n_levels = K + 1
            self.dz_dual = 0.5 * (np.r_[0.0, self.dz] + np.r_[self.dz, 0.0])
            self.z_level = z
        nt = mesh2d.n_cells
        if tags is None:
            tags = np.repeat(mesh2d.cell_tags[:, None], K, 1)
        elif callable(tags):
            tags = tags(mesh2d.cell_tags[:, None], self.z_mid[None, :])
        tags = np.broadcast_to(np.asarray(tags, dtype=np.int64), (nt, K)).copy()
        self.tri_tags = tags
        self.c2 = PolygonComplex2D(mesh2d, tags, merge_tol)
        self.tags = tags[self.c2.first_tri]  # (n_cells2d, K)

        # Level-difference operators (K x L)
        L = self.n_levels
        k = np.arange(K)
        up = (k + 1) % L
        self.S = sp.csr_matrix(
            (np.r_[np.ones(K), -np.ones(K)], (np.r_[k, k], np.r_[k, up])), shape=(K, L)
        )
        self._up = up

    # ------------------------------------------------------------ counts
    @property
    def n_hedges(self):
        return self.c2.n_edges * self.n_levels

    @property
    def n_vedges(self):
        return self.c2.n_nodes * self.K

    @property
    def n_edges(self):
        return self.n_hedges + self.n_vedges

    @property
    def n_faces(self):
        return self.c2.n_cells * self.n_levels + self.c2.n_quads * self.K

    @property
    def n_cells(self):
        return self.c2.n_cells * self.K

    @property
    def n_nodes(self):
        return self.c2.n_nodes * self.n_levels

    # ------------------------------------------------------------ global ops
    def curl(self):
        """Global face-edge incidence C (faces: [b_z, b_q], edges: [e_h, e_v])."""
        m, L, K = self.c2, self.n_levels, self.K
        nk = m.n_edges
        top = sp.hstack(
            [sp.kron(m.C, sp.eye(L)), sp.csr_matrix((m.n_cells * L, m.n_nodes * K))]
        )
        zs = sp.vstack([sp.kron(sp.eye(nk), self.S), sp.kron(m.P_removed, self.S)])
        bot = sp.hstack([zs, sp.kron(m.G_quad, sp.eye(K))])
        return sp.vstack([top, bot]).tocsr()

    def grad(self):
        """Global edge-node incidence G (nodes indexed n * L + l)."""
        m, L = self.c2, self.n_levels
        return sp.vstack(
            [sp.kron(m.G, sp.eye(L)), sp.kron(sp.eye(m.n_nodes), -self.S)]
        ).tocsr()

    def div(self):
        """Global cell-face incidence D (outward fluxes; D C = 0)."""
        m = self.c2
        Cq = sp.hstack([m.C, sp.csr_matrix((m.n_cells, m.n_quads - m.n_edges))])
        # b_z: +top level, -bottom level of each layer = -S
        return sp.hstack(
            [sp.kron(sp.eye(m.n_cells), -self.S), sp.kron(Cq, sp.eye(self.K))]
        ).tocsr()

    def edge_points(self):
        """Midpoints (n_edges, 3) of [e_h, e_v] in the solver ordering [m]."""
        m, L, K = self.c2, self.n_levels, self.K
        hm = np.repeat(m.edge_midpoint, L, 0)
        hz = np.tile(self.z_level, m.n_edges)
        vm = np.repeat(m.nodes, K, 0)
        vz = np.tile(self.z_mid, m.n_nodes)
        return np.vstack([np.c_[hm, hz], np.c_[vm, vz]])

    def locate(self, points):
        """Triangle, barycentric coordinates and (layer, fraction) of points."""
        pts = np.atleast_2d(np.asarray(points, float))
        tri, lam = self.m2.locate(pts[:, :2])
        if np.any(tri < 0):
            raise ValueError("point outside the cross-section")
        zz = pts[:, 2]
        if np.any(zz < self.z[0]) or np.any(zz > self.z[-1]):
            raise ValueError("point outside the z range")
        k = np.clip(np.searchsorted(self.z, zz, side="right") - 1, 0, self.K - 1)
        frac = (zz - self.z[k]) / self.dz[k]
        return tri, lam, k, frac


class PrismFIT3D:
    """
    Time-domain (leapfrog) and eigenmode FIT solver on an extruded prism mesh.

    Parameters
    ----------
    mesh : PrismMesh
        Prism mesh with material tags per prism.
    materials : dict, optional
        ``{tag: material}``: ``[eps_r, mu_r]``, ``[eps_r, mu_r, sigma]``, a
        `wakis.dispersive.DispersiveMaterial` or ``"pec"``. Missing tags are
        vacuum.
    bc_xy : {"pec", "pmc", "abc"} or callable
        Lateral boundary of the cross-section: PEC, PMC (natural) or the
        first-order Silver-Mueller absorbing condition; a callable maps the
        (n, 2) midpoints of the boundary edges to one of these per edge
        (e.g. symmetry planes plus an absorbing outer arc). A node is PEC
        if any of its boundary edges is.
    bc_z : {"pec", "pmc", "periodic"} or (low, high)
        Boundaries at z[0] and z[-1] (pec: tangential E = 0, pmc: natural).
    absorber_xy : dict, optional
        Matched isotropic loss annulus ``dict(r0, r1, s_max, order=2)``,
        sigma_e / eps = sigma_m / mu = s(r) = s_max ((r - r0)/(r1 - r0))^order
        [1/s] for r = sqrt(x^2 + y^2) > r0 (as `UnstructuredFIT2D`).
    pml_z : dict, optional
        Berenger split-field PML along z, ``dict(n_low=..., n_high=...,
        s_max=..., order=3)``: the z-derivative parts of e_h and b_q decay at
        the rate s(z) = s_max (d / D)^order [1/s] in the first `n_low` and last
        `n_high` layers (d: depth into the layer stack of thickness D). The
        matched condition sigma_e / eps = sigma_m / mu holds in any medium,
        so layers may cross the PML. Non-dispersive, lossless media only
        inside the PML.
    hodge : {"voronoi", "barycentric"}
        2D dual of the vertical-edge capacitances (barycentric: positive on
        any mesh, first-order accurate). Horizontal edges always use the
        Voronoi dual edges (requires a Delaunay cross-section).
    dual_floor : float, optional
        Lower bound of the dual-edge length of horizontal edges and vertical
        quads, in units of the primal edge length (0 disables the clamp).
    dt, cfl : float, optional
        Timestep [s] or fraction of the leapfrog stability limit.
    use_gpu : bool, optional
        Run the time stepping with CuPy (non-dispersive, lossless interior;
        absorbers and PML supported).

    Attributes
    ----------
    eh : ndarray (n_edges2d, n_levels)
        Horizontal edge voltages [V].
    ev : ndarray (n_nodes2d, K)
        Vertical edge voltages [V].
    bz : ndarray (n_tri, n_levels)
        Magnetic flux through the triangles [Wb] (along +z).
    bq : ndarray (n_edges2d, K)
        Magnetic flux through the vertical quads [Wb] (along t_e x z).
    """

    def __init__(
        self,
        mesh,
        materials=None,
        bc_xy="pec",
        bc_z="pec",
        absorber_xy=None,
        pml_z=None,
        hodge="voronoi",
        dual_floor=0.05,
        dt=None,
        cfl=0.95,
        use_gpu=False,
    ):
        if not callable(bc_xy) and bc_xy not in ("pec", "pmc", "abc"):
            raise ValueError("bc_xy must be 'pec', 'pmc', 'abc' or a callable")
        if isinstance(bc_z, str):
            bc_z = (bc_z, bc_z)
        bc_z = tuple(b.lower() for b in bc_z)
        if ("periodic" in bc_z) != mesh.periodic_z or (
            mesh.periodic_z and bc_z != ("periodic", "periodic")
        ):
            raise ValueError("bc_z='periodic' requires PrismMesh(periodic_z=True)")
        for b in bc_z:
            if b not in ("pec", "pmc", "periodic"):
                raise ValueError("bc_z entries must be 'pec', 'pmc' or 'periodic'")
        if hodge not in ("voronoi", "barycentric"):
            raise ValueError("hodge must be 'voronoi' or 'barycentric'")
        self.mesh, self.bc_xy, self.bc_z, self.hodge = mesh, bc_xy, bc_z, hodge
        m2, K, Lv = mesh.c2, mesh.K, mesh.n_levels
        nT, nE, nN = m2.n_cells, m2.n_edges, m2.n_nodes
        nQ = m2.n_quads
        self._nk = nE
        self._shape = dict(eh=(nE, Lv), ev=(nN, K), bz=(nT, Lv), bq=(nQ, K))

        # ---- materials per prism
        eps_r, mu_r, sigma = np.ones((nT, K)), np.ones((nT, K)), np.zeros((nT, K))
        pec = np.zeros((nT, K), bool)
        dispersive = []
        for tag, mat in (materials or {}).items():
            sel = mesh.tags == tag
            if not np.any(sel):
                continue
            if isinstance(mat, str):
                if mat.lower() != "pec":
                    raise ValueError(f"unknown material {mat!r}")
                pec |= sel
            elif isinstance(mat, DispersiveMaterial):
                eps_r[sel] = mat.eps_inf
                sigma[sel] = mat.sigma
                if mat.poles:
                    dispersive.append((mat, sel))
            else:
                mat = list(mat)
                eps_r[sel] = mat[0]
                mu_r[sel] = mat[1] if len(mat) > 1 else 1.0
                sigma[sel] = mat[2] if len(mat) > 2 else 0.0
        live = ~pec
        self.eps_r, self.mu_r, self.sigma, self.pec = eps_r, mu_r, sigma, pec
        self.dispersive = dispersive

        # ---- 2D metric
        Le = m2.edge_length
        He = m2.dual_edge_partial  # (nE, nT): h_{e,t}
        Pn = m2.dual_area_partial if hodge == "voronoi" else m2.dual_area_barycentric
        At = m2.cell_volume
        dz, dzd = mesh.dz, mesh.dz_dual

        def level_sum(W):
            """(n, K) per-layer half-contributions -> (n, L) per level."""
            out = np.zeros((W.shape[0], Lv))
            out[:, :K] += W
            out[:, mesh._up] += W
            return out

        def h_cap(cellval):
            # sum over the two adjacent layers of h_{e,t} * value * dz / 2
            return level_sum(He @ (cellval * live * dz / 2)) / Le[:, None]

        def v_cap(cellval):
            return (Pn @ (cellval * live)) / dz[None, :]

        self._h_cap, self._v_cap = h_cap, v_cap
        cap_h = eps_0 * h_cap(eps_r)
        cap_v = eps_0 * v_cap(eps_r)
        cond_h = h_cap(sigma)
        cond_v = v_cap(sigma)
        nu = live / (mu_0 * mu_r)
        Mz = level_sum(nu * dz / 2) / At[:, None]
        Hq, Lq = m2.dual_edge_partial_quad, m2.edge_length_quad
        Mq = (Hq @ nu) / (Lq[:, None] * dz[None, :])
        # Dual-edge floor: cocircular (h_e = 0) or locally non-Delaunay edges
        # (a circumcentre beyond the edge, possibly in the high-eps cell) give
        # non-positive Hodge entries, i.e. massless e_h or negative magnetic
        # energy. They are clamped to the value of a dual edge of length
        # dual_floor * L_e in the adjacent media (a local O(1) Hodge
        # perturbation on a few edges; counted in `n_floored`).
        Ha = abs(He)
        if dual_floor > 0:
            wsum = level_sum(Ha @ (live * dz / 2))
            eps_avg = level_sum(Ha @ (eps_r * live * dz / 2)) / np.maximum(wsum, 1e-300)
            cmin = eps_0 * eps_avg * dual_floor * mesh.dz_dual[None, :]
            low_h = (cap_h < cmin) & (wsum > 0)
            cap_h = np.where(low_h, cmin, cap_h)
            Hqa = abs(Hq)
            nu_avg = (Hqa @ nu) / np.maximum(Hqa @ live.astype(float), 1e-300)
            qmin = nu_avg * dual_floor / dz[None, :]
            low_q = (Mq < qmin) & ((Hqa @ live.astype(float)) > 0)
            Mq = np.where(low_q, qmin, Mq)
            self.n_floored = int(low_h.sum() + low_q.sum())
        else:
            self.n_floored = 0
        self.Mz, self.Mq = Mz, Mq

        # ---- fixed (PEC) edges: free if touched by a live prism, fixed if
        # touched by a PEC prism (tangential E = 0 on the conductor)
        CEa, CNa = m2.cell_edge_abs.T.tocsr(), m2.cell_node_abs.T.tocsr()
        up = mesh._up

        def touched(sel):
            th = CEa @ sel.astype(float)  # (nE, K) per layer
            lev = np.zeros((nE, Lv))
            lev[:, :K] += th
            lev[:, up] += th
            return lev > 0, (CNa @ sel.astype(float)) > 0

        th_live, tv_live = touched(live)
        th_pec, tv_pec = touched(pec)
        fix_h = ~th_live | th_pec
        fix_v = ~tv_live | tv_pec
        # per-boundary-edge type (a callable maps edge midpoints to types)
        be_all = m2.boundary_edges
        if callable(bc_xy):
            kind = np.asarray(bc_xy(m2.edge_midpoint[be_all]))
            if not np.all(np.isin(kind, ("pec", "pmc", "abc"))):
                raise ValueError("bc_xy(points) must return 'pec', 'pmc' or 'abc'")
        else:
            kind = np.full(be_all.size, bc_xy)
        self.boundary_kind = kind
        be_pec = be_all[kind == "pec"]
        fix_h[be_pec, :] = True
        fix_v[np.unique(m2.edges[be_pec].ravel()), :] = True
        if not mesh.periodic_z:
            if bc_z[0] == "pec":
                fix_h[:, 0] = True
            if bc_z[1] == "pec":
                fix_h[:, -1] = True
        self.fix_h, self.fix_v = fix_h, fix_v

        # ---- lateral ABC (Silver-Mueller): G = l_surface / (eta L)
        if np.any(kind == "abc"):
            be = be_all[kind == "abc"]
            tb = CEa[be].indices  # the single cell of each boundary edge
            if tb.size != be.size:
                raise RuntimeError("boundary edge with != 1 cell")
            eta_e = ETA_0 * np.sqrt(mu_r[tb] / eps_r[tb])  # (nbe, K)
            # horizontal boundary edges: surface dual length dz*/2 per layer
            g = np.zeros((be.size, Lv))
            g[:, :K] += dz / 2 / eta_e
            g[:, mesh._up] += dz / 2 / eta_e
            cond_h[be] += g / Le[be, None]
            # vertical boundary edges: half of each adjacent boundary edge
            gv = np.zeros((nN, K))
            for s in (0, 1):
                np.add.at(gv, m2.edges[be, s], 0.5 * Le[be, None] / eta_e)
            cond_v += gv / dz[None, :]

        # ---- matched lateral absorber
        self.rate_bz = self.rate_bq = None
        if absorber_xy is not None:
            r0, r1 = absorber_xy["r0"], absorber_xy["r1"]
            order = absorber_xy.get("order", 2)

            def rate(r):
                return absorber_xy["s_max"] * np.clip((r - r0) / (r1 - r0), 0, None) ** order

            r_e = np.hypot(*m2.edge_midpoint.T)
            r_q = np.hypot(*m2.edge_midpoint_quad.T)
            r_n = np.hypot(*m2.nodes.T)
            r_t = np.hypot(*m2.cell_centroid.T)
            cond_h = cond_h + cap_h * rate(r_e)[:, None]
            cond_v = cond_v + cap_v * rate(r_n)[:, None]
            self.rate_bz = np.repeat(rate(r_t)[:, None], Lv, 1)
            self.rate_bq = np.repeat(rate(r_q)[:, None], K, 1)

        # ---- flatten (edge order [e_h, e_v])
        self.free = np.flatnonzero(~np.r_[fix_h.ravel(), fix_v.ravel()])
        self.fixed_mask = np.r_[fix_h.ravel(), fix_v.ravel()]
        cap = np.r_[cap_h.ravel(), cap_v.ravel()]
        cond = np.r_[cond_h.ravel(), cond_v.ravel()]
        bad = (cap <= 0) & ~self.fixed_mask
        if np.any(bad):
            raise ValueError(
                f"{bad.sum()} free edges with non-positive capacitance: the "
                "cross-section is not Delaunay (or PEC cuts dual faces)"
            )
        cap = np.where(self.fixed_mask, 1.0, cap)
        cond = np.where(self.fixed_mask, 0.0, cond)
        self.cap, self.cond = cap, cond
        self.icap = np.where(self.fixed_mask, 0.0, 1.0 / cap)
        self.Mnu = np.r_[Mz.ravel(), Mq.ravel()]
        self._freef = (~self.fixed_mask).astype(float)
        self.pml = None
        if pml_z is not None:
            self._init_pml(pml_z)

        # ---- stability limit (non-dispersive leapfrog, without losses)
        C = mesh.curl()
        self._C = C
        Cf = C[:, self.free]
        Kf = (Cf.T @ sp.diags(self.Mnu) @ Cf).tocsr()
        lam = max_eigenvalue(Kf, M_inv_sqrt=1 / np.sqrt(cap[self.free]))
        self.lambda_max = lam
        self.dt_max = 2 / np.sqrt(lam)
        self.dt = cfl * self.dt_max if dt is None else float(dt)
        self.courant = self.dt * np.sqrt(lam) / 2

        # ---- dispersive groups / losses -> semi-implicit update
        groups = []
        if dispersive:
            check_dispersive_stability([m for m, _ in dispersive], self.dt, self.courant)
            for mat, sel in dispersive:
                c = eps_0 * np.r_[h_cap(sel.astype(float)).ravel(), v_cap(sel.astype(float)).ravel()]
                idx = np.flatnonzero((c > 0) & ~self.fixed_mask)
                groups.append((mat, idx, c[idx]))
        self._lossy = bool(groups) or np.any(cond != 0)
        self._update = CapacitiveUpdate(cap, cond, groups, self.dt) if self._lossy else None

        # ---- 2D operators for the layered update
        self.C2 = m2.C.tocsr()
        self.C2T = m2.C.T.tocsr()
        self.G2 = m2.G_quad.tocsr()  # quad-node incidence (all 2D edges)
        self.G2T = m2.G_quad.T.tocsr()
        self.Prem = m2.P_removed.tocsr()  # virtual e_h of merged edges
        self.PremT = m2.P_removed.T.tocsr()
        self.sources = []
        self.ports = []
        self.probes = []
        self.dfts = []
        self.use_gpu = use_gpu
        self.xp = np
        if use_gpu:
            self._to_gpu()
        self._prepare_pml()
        self.reset()

    # ------------------------------------------------------------ PML
    def _init_pml(self, p):
        mesh = self.mesh
        if mesh.periodic_z:
            raise ValueError("pml_z needs non-periodic z")
        z, K = mesh.z, mesh.K
        nl, nh = int(p.get("n_low", 0)), int(p.get("n_high", 0))
        order = p.get("order", 3)
        smax = float(p["s_max"])
        zl = z[nl] if nl else -np.inf
        zh = z[K - nh] if nh else np.inf
        Dl = z[nl] - z[0] if nl else 1.0
        Dh = z[-1] - z[K - nh] if nh else 1.0

        def rate(zz):
            d = np.where(zz < zl, (zl - zz) / Dl, 0.0) + np.where(zz > zh, (zz - zh) / Dh, 0.0)
            return smax * np.clip(d, 0, 1) ** order

        s_lev = rate(mesh.z_level)
        s_lay = rate(mesh.z_mid)
        lev = np.flatnonzero(s_lev > 0)
        lay = np.flatnonzero(s_lay > 0)
        if self.dispersive and any(np.any(sel[:, lay]) for _, sel in self.dispersive):
            raise NotImplementedError("pml_z: non-dispersive media only")
        self.pml = dict(lev=lev, lay=lay, s_lev=s_lev[lev], s_lay=s_lay[lay])

    def _prepare_pml(self):
        """Per-edge coefficients of the split e_h update in the PML levels."""
        p = self.pml
        if p is None:
            return
        xp, lev = self.xp, p["lev"]
        sh = self._shape["eh"]
        nh = self.mesh.n_hedges
        C0 = self.cap[:nh].reshape(sh)[:, lev]
        G = self.cond[:nh].reshape(sh)[:, lev]
        if self._update is not None:
            a1 = self._update.a1[:nh].reshape(sh)[:, lev]
            den = self._update.den[:nh].reshape(sh)[:, lev]
        else:
            a1 = den = C0
        p.update(
            C0=xp.asarray(C0), g=xp.asarray(G / C0), a1=xp.asarray(a1),
            den=xp.asarray(den),
            free_lev=xp.asarray(self._freef[:nh].reshape(sh)[:, lev]),
            s_lev_x=xp.asarray(p["s_lev"]), s_lay_x=xp.asarray(p["s_lay"]),
        )  # fmt: skip

    # ------------------------------------------------------------ GPU
    def _to_gpu(self):
        import cupy as cp
        import cupyx.scipy.sparse as csp

        if self._lossy and self._update.groups:
            raise NotImplementedError("use_gpu with dispersive materials")
        self.xp = cp
        for name in ("C2", "C2T", "G2", "G2T", "Prem", "PremT"):
            setattr(self, name, csp.csr_matrix(getattr(self, name)))
        self._g = {
            "icap": cp.asarray(self.icap),
            "a1": cp.asarray(self._update.a1 if self._lossy else self.cap),
            "den": cp.asarray(self._update.den if self._lossy else self.cap),
            "Mz": cp.asarray(self.Mz),
            "Mq": cp.asarray(self.Mq),
            "freef": cp.asarray(self._freef),
        }

    # ------------------------------------------------------------ state
    def reset(self):
        """Zero fields, auxiliary variables, probes, ports and DFTs."""
        xp = self.xp
        nh = self.mesh.n_hedges
        self.e = xp.zeros(self.mesh.n_edges)
        self.eh = self.e[:nh].reshape(self._shape["eh"])
        self.ev = self.e[nh:].reshape(self._shape["ev"])
        self.bz = xp.zeros(self._shape["bz"])
        self.bq = xp.zeros(self._shape["bq"])
        self.n = 0
        if self._update is not None:
            self._update.reset()
        if self.pml is not None:
            p = self.pml
            nE, nQ = self._shape["eh"][0], self._shape["bq"][0]
            p["eh_z"] = xp.zeros((nE, p["lev"].size))
            p["bq_z"] = xp.zeros((nQ, p["lay"].size))
        for pr in self.probes:
            pr["samples"] = []
        for d in self.dfts:
            d["data"][:] = 0
        for port in self.ports:
            port.update(V=[], I=[], t=[])

    @property
    def b(self):
        """All face fluxes [b_z, b_q] (flattened) [Wb]."""
        return self.xp.concatenate([self.bz.ravel(), self.bq.ravel()])

    def curl_e(self, eh, ev):
        """Discrete curl of edge voltages: (C e) split into (b_z, b_q) shapes."""
        cz = self.C2 @ eh
        if self.mesh.periodic_z:
            dzv = eh - eh[:, self.mesh._up]
        else:
            dzv = eh[:, :-1] - eh[:, 1:]
        cq = self.G2 @ ev
        cq[: self._nk] += dzv
        if self._nk < cq.shape[0]:
            cq[self._nk :] += self.Prem @ dzv
        return cz, cq

    def curl_t_h(self, hz, hq):
        """Transposed curl of face voltages: (C^T h) split into (e_h, e_v)."""
        rh = self.C2T @ hz
        K = self.mesh.K
        hk = self._hk(hq)
        rh[:, :K] += hk
        if self.mesh.periodic_z:
            rh -= self.xp.roll(hk, 1, axis=1)
        else:
            rh[:, 1:] -= hk
        rv = self.G2T @ hq
        return rh, rv

    def _hk(self, hq):
        """z-coupling seen by the kept horizontal edges (C^T restricted)."""
        nk = self._nk
        if nk == hq.shape[0]:
            return hq
        return hq[:nk] + self.PremT @ hq[nk:]

    def energy(self):
        """
        Exactly conserved discrete energy [J] of the lossless leapfrog,
        W^n = e^n M_eps e^n / 2 + b^{n-1/2} M_nu b^{n+1/2} / 2.
        """
        xp = self.xp
        cz, cq = self.curl_e(self.eh, self.ev)
        cap = xp.asarray(self.cap) * xp.asarray(self._freef)
        we = 0.5 * xp.sum(cap * self.e**2)
        wm = 0.5 * (
            xp.sum(self._Mz() * self.bz * (self.bz - self.dt * cz))
            + xp.sum(self._Mq() * self.bq * (self.bq - self.dt * cq))
        )
        return float(we + wm)

    def divergence_b(self):
        """Net magnetic flux out of every prism [Wb], shape (n_tri, K)."""
        if self.mesh.periodic_z:
            dbz = self.bz[:, self.mesh._up] - self.bz
        else:
            dbz = self.bz[:, 1:] - self.bz[:, :-1]
        return dbz + self.C2 @ self.bq[: self._nk]

    # ------------------------------------------------------------ sources
    def _whitney(self, points):
        """
        Prism Whitney edge functions at points: sparse (n_points*3 x n_edges)
        rows ordered (point, component x/y/z) [1/m].

        In-plane: W_e(x, y) (2D Whitney) times (1 - f) at level k and f at
        level k + 1; vertical: lambda_n(x, y) / dz_k z-hat.
        """
        mesh, m2 = self.mesh, self.mesh.m2
        emap = mesh.c2.edge_map
        tri, lam, k, f = mesh.locate(points)
        _, W = m2.whitney_edge(np.atleast_2d(points)[:, :2])  # (n, 3, 2)
        L, K = mesh.n_levels, mesh.K
        nh = mesh.n_hedges
        rows, cols, vals = [], [], []
        for i in range(len(tri)):
            for j in range(3):
                e = emap[m2.cell_edges[tri[i], j]]
                if e < 0:
                    continue  # merged (removed) edge: no degree of freedom
                for lev, w in ((k[i], 1 - f[i]), (mesh._up[k[i]], f[i])):
                    for c in (0, 1):
                        rows.append(3 * i + c)
                        cols.append(e * L + lev)
                        vals.append(W[i, j, c] * w)
                n = m2.cells[tri[i], j]
                rows.append(3 * i + 2)
                cols.append(nh + n * K + k[i])
                vals.append(lam[i, j] / mesh.dz[k[i]])
        return sp.csr_matrix((vals, (rows, cols)), shape=(3 * len(tri), mesh.n_edges))

    def add_point_dipole(self, position, direction, amp_func):
        """
        Electric point dipole J = I l(t) d delta(r - r0), distributed on the
        edges of the containing prism with the Whitney functions
        (i_e = I l(t) d . W_e(r0)).

        Parameters
        ----------
        position : (3,) float
            Location [m].
        direction : (3,) float
            Dipole direction.
        amp_func : callable
            t [s] -> current moment I l [A m].
        """
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        W = self._whitney([position])
        w = np.asarray(W.T @ d).ravel()
        w[self.fixed_mask] = 0
        self.sources.append((self.xp.asarray(w), amp_func))

    def add_current_density(self, J_func, amp_func):
        """
        Distributed current density J(r) amp(t) [A/m^2]: edge currents
        i_e = A*_e J(m_e) . t_e through the (live) dual faces.
        """
        mesh = self.mesh
        pts = mesh.edge_points()
        J = np.asarray(J_func(pts))
        m2 = mesh.c2
        t_h = np.repeat(np.c_[m2.edge_tangent, np.zeros(m2.n_edges)], mesh.n_levels, 0)
        nv = mesh.n_vedges
        t = np.vstack([t_h, np.tile([0.0, 0.0, 1.0], (nv, 1))])
        area = self.cap / eps_0  # vacuum-weighted dual area / length ... use eps-free measure
        ones = np.ones_like(self.eps_r)
        A = np.r_[
            (self._h_cap(ones) * m2.edge_length[:, None]).ravel(),
            (self._v_cap(ones) * mesh.dz[None, :]).ravel(),
        ]
        del area
        w = A * np.einsum("ij,ij->i", J, t)
        w[self.fixed_mask] = 0
        self.sources.append((self.xp.asarray(w), amp_func))

    def add_lumped_port(self, edges, R, amp_func=None, length=None):
        """
        Lumped (discrete) port: resistor R in parallel with a Norton current
        source V_s(t) / R across a gap spanned by vertical edges.

        The gap is a block of vertical edges: edges in different layers are
        in series, edges at different nodes in parallel. The port is a
        uniform conductivity sigma = l / (R A) over the block (l: gap length,
        A: total dual area of one layer), applied semi-implicitly (stable for
        any R), plus the impressed current density sigma V_s / l. The port
        voltage is V = -sum(e) / n_parallel weighted by the dual areas (the
        potential of the top minus the bottom), the port current I = (V_s -
        V) / R flows into the structure.

        Parameters
        ----------
        edges : array_like of int
            Indices of vertical edges in the solver edge ordering (see
            `vertical_edge_index`), all in the gap.
        R : float
            Port (reference) resistance [Ohm].
        amp_func : callable, optional
            t [s] -> source voltage V_s [V]; None for a passive resistor.
        length : float, optional
            Gap length l [m] (default: summed layer thicknesses of the edges).

        Returns
        -------
        dict
            Port record with time series ``t``, ``V``, ``I`` (filled while
            stepping); see `port_impedance`.
        """
        if self.use_gpu:
            raise NotImplementedError("lumped ports on GPU")
        mesh = self.mesh
        edges = np.unique(np.asarray(edges, int))
        nh = mesh.n_hedges
        if np.any(edges < nh) or np.any(self.fixed_mask[edges]):
            raise ValueError("port edges must be free vertical edges")
        n, k = np.divmod(edges - nh, mesh.K)
        layers = np.unique(k)
        l = float(mesh.dz[layers].sum()) if length is None else float(length)
        ones = np.ones_like(self.eps_r)
        Adual = (self._v_cap(ones) * mesh.dz[None, :]).ravel()[edges - nh]
        A = Adual.sum() / layers.size
        sigma = l / (R * A)
        g = sigma * Adual / mesh.dz[k]  # conductance per edge [S]
        if self._update is None:
            self._update = CapacitiveUpdate(self.cap, self.cond.copy(), [], self.dt)
            self._lossy = True
        # add conductance to the semi-implicit update
        up = self._update
        up.a1[edges] -= g * self.dt / 2
        up.den[edges] += g * self.dt / 2
        port = dict(
            edges=edges, g=g, R=float(R), l=l, src=amp_func,
            wV=Adual / A, inj=sigma * Adual / l, t=[], V=[], I=[],
        )  # fmt: skip
        self.ports.append(port)
        self._prepare_pml()
        return port

    def vertical_edge_index(self, nodes2d, layers):
        """Solver indices of the vertical edges at 2D `nodes2d` x `layers`."""
        n, k = np.meshgrid(np.atleast_1d(nodes2d), np.atleast_1d(layers), indexing="ij")
        return (self.mesh.n_hedges + n * self.mesh.K + k).ravel()

    @staticmethod
    def port_impedance(port, freqs):
        """Input impedance Z(f) = V(f) / I(f) [Ohm] of a port (DTFT of records)."""
        t = np.asarray(port["t"])
        ph = np.exp(-2j * np.pi * np.outer(freqs, t))
        V = ph @ np.asarray(port["V"])
        I = ph @ np.asarray(port["I"])
        return V / I

    # ------------------------------------------------------------ monitors
    def add_probe(self, points, component):
        """Record E_component ('x', 'y' or 'z') at points every step (Whitney)."""
        c = "xyz".index(component) if isinstance(component, str) else int(component)
        W = self._whitney(points)[c::3]
        if self.use_gpu:
            import cupyx.scipy.sparse as csp

            W = csp.csr_matrix(W)
        rec = {"P": W, "samples": []}
        self.probes.append(rec)
        return rec

    def add_dft(self, freqs):
        """Running DFT (kernel exp(-i w t) dt) of all edge voltages at `freqs`."""
        freqs = np.atleast_1d(np.asarray(freqs, float))
        rec = {"freqs": freqs, "data": self.xp.zeros((freqs.size, self.mesh.n_edges), complex)}
        self.dfts.append(rec)
        return rec

    @staticmethod
    def probe_signal(probe):
        """Recorded samples, shape (n_steps, n_points)."""
        return np.asarray([np.asarray(getattr(s, "get", lambda: s)()) for s in probe["samples"]])

    def probe_times(self):
        """Sample times [s] (field at the end of each step)."""
        return (np.arange(self.n) + 1) * self.dt

    # ------------------------------------------------------------ stepping
    def one_step(self):
        """Leapfrog: b to n+1/2, then e to n+1 (losses CN-averaged)."""
        xp, dt = self.xp, self.dt
        t_half = (self.n + 0.5) * dt
        pml = self.pml
        cz = self.C2 @ self.eh
        if self.mesh.periodic_z:
            cqz = self.eh - self.eh[:, self.mesh._up]
        else:
            cqz = self.eh[:, :-1] - self.eh[:, 1:]
        cq = self.G2 @ self.ev
        nk = self._nk
        if nk < cq.shape[0]:
            cqz = xp.concatenate([cqz, self.Prem @ cqz])
        cq += cqz
        if self.rate_bz is None:
            if pml is not None:
                lay = pml["lay"]
                old_z = pml["bq_z"]
                rz = cqz[:, lay]
            self.bz -= dt * cz
            self.bq -= dt * cq
            if pml is not None:
                # split b_q = b_xy + b_z in PML layers: b_z' + s b_z = -dz e_h
                s = 0.5 * dt * pml["s_lay_x"]
                new_z = ((1 - s) * old_z - dt * rz) / (1 + s)
                self.bq[:, lay] += (new_z - old_z) + dt * rz
                pml["bq_z"] = new_z
        else:
            a = 0.5 * dt * self._rate("bz")
            self.bz[:] = ((1 - a) * self.bz - dt * cz) / (1 + a)
            a = 0.5 * dt * self._rate("bq")
            if pml is not None:
                lay = pml["lay"]
                old_z = pml["bq_z"]
                rz = cqz[:, lay]
                old_xy = self.bq[:, lay] - old_z
            self.bq[:] = ((1 - a) * self.bq - dt * cq) / (1 + a)
            if pml is not None:
                al = a[:, lay]
                sz = al + 0.5 * dt * pml["s_lay_x"]
                new_z = ((1 - sz) * old_z - dt * rz) / (1 + sz)
                new_xy = ((1 - al) * old_xy - dt * (cq[:, lay] - rz)) / (1 + al)
                self.bq[:, lay] = new_xy + new_z
                pml["bq_z"] = new_z

        hz = self._Mz() * self.bz
        hq = self._Mq() * self.bq
        rh, rv = self.curl_t_h(hz, hq)
        r = xp.concatenate([rh.ravel(), rv.ravel()])
        for w, f in self.sources:
            r -= w * float(f(t_half))
        for p in self.ports:
            if p["src"] is not None:
                p["_vs"] = float(p["src"](t_half))
                # Norton current J = sigma V_s / l along +z: r = C^T h - j
                r[p["edges"]] -= p["inj"] * p["_vs"]
            p["_vn"] = -float(p["wV"] @ self.e[p["edges"]])
        if pml is not None:
            lev, K = pml["lev"], self.mesh.K
            # z-derivative part of C^T h at the PML levels
            rz = xp.zeros(self.eh.shape)
            hk = self._hk(hq)
            rz[:, :K] += hk
            rz[:, 1:] -= hk
            rz = rz[:, lev] * pml["free_lev"]
            old_z = pml["eh_z"]
        if self._lossy and self.use_gpu:
            g = self._g
            self.e[:] = (g["a1"] * self.e + dt * g["freef"] * r) / g["den"]
        elif self._lossy:
            r *= self._freef
            self.e[:] = self._update.step(self.e, r)
        elif self.use_gpu:
            self.e += dt * self._g["icap"] * r
        else:
            self.e += dt * self.icap * r
        if pml is not None:
            # e = e_full - (a1 e_z + dt r_z) / den + e_z_new, where e_z decays
            # at the rate g + s (g = G / C0 of the lateral losses)
            C0, g, a1, den = pml["C0"], pml["g"], pml["a1"], pml["den"]
            sz = 0.5 * dt * (g + pml["s_lev_x"])
            new_z = ((1 - sz) * old_z + dt * rz / C0) / (1 + sz)
            self.eh[:, lev] += new_z - (a1 * old_z + dt * rz) / den
            pml["eh_z"] = new_z
        self.n += 1
        t = self.n * dt
        for p in self.ports:
            vn1 = -float(p["wV"] @ self.e[p["edges"]])
            V = 0.5 * (p["_vn"] + vn1)
            vs = p.get("_vs", 0.0) if p["src"] is not None else 0.0
            p["t"].append(t_half)
            p["V"].append(V)
            p["I"].append((vs - V) / p["R"])
        for rec in self.probes:
            rec["samples"].append(rec["P"] @ self.e)
        for rec in self.dfts:
            ph = xp.exp(-2j * np.pi * xp.asarray(rec["freqs"]) * t) * dt
            rec["data"] += ph[:, None] * self.e[None]

    def _rate(self, which):
        key = "_g_rate_" + which
        if not hasattr(self, key):
            setattr(self, key, self.xp.asarray(getattr(self, "rate_" + which)))
        return getattr(self, key)

    def _Mz(self):
        return self._g["Mz"] if self.use_gpu else self.Mz

    def _Mq(self):
        return self._g["Mq"] if self.use_gpu else self.Mq

    def _icap_h(self):
        if not hasattr(self, "_icap_h_arr"):
            self._icap_h_arr = self.xp.asarray(
                self.icap[: self.mesh.n_hedges].reshape(self._shape["eh"])
            )
        return self._icap_h_arr

    def run(self, n_steps):
        """Advance `n_steps` steps."""
        for _ in range(int(n_steps)):
            self.one_step()

    # ------------------------------------------------------------ eigenmodes
    def eigenmodes(self, n_modes=6, f_target=None, tol=1e-10):
        """
        Resonant frequencies of the closed, lossless structure.

        Shift-invert Lanczos for C^T M_nu C e = w^2 M_eps e on the free edges,
        with the static gradient null space projected out (as
        `UnstructuredFIT3D.eigenmodes`).

        Parameters
        ----------
        n_modes : int
        f_target : float
            Shift [Hz] (> 0), below the modes of interest.

        Returns
        -------
        f : ndarray
            The `n_modes` frequencies closest to `f_target` [Hz], ascending.
        e : ndarray (n_edges, n_modes)
            Edge voltages.
        """
        if not f_target:
            raise ValueError("f_target > 0 is required")
        if self.pml is not None or np.any(self.cond):
            raise ValueError("eigenmodes need a closed lossless structure")
        idx = self.free
        C = self._C[:, idx]
        Kf = (C.T @ sp.diags(self.Mnu) @ C).tocsc()
        M = sp.diags(self.cap[idx])
        sigma = (2 * np.pi * f_target) ** 2
        lu = splu((Kf - sigma * M).tocsc())
        Gfull = self.mesh.grad().tocsr()
        G = Gfull[idx]
        touch = np.zeros(self.mesh.n_nodes, bool)
        touch[np.unique(Gfull[np.flatnonzero(self.fixed_mask)].indices)] = True
        Gn = G[:, np.flatnonzero(~touch)].tocsc()
        MG = (M @ Gn).tocsc()
        lap = splu((Gn.T @ MG).tocsc())

        def op(x):
            y = lu.solve(x)
            return y - Gn @ lap.solve(MG.T @ y)

        n = Kf.shape[0]
        w2, vec = eigsh(
            Kf, k=n_modes + 2, M=M, sigma=sigma, which="LM",
            OPinv=LinearOperator((n, n), matvec=op, dtype=float),
            ncv=max(4 * n_modes + 10, 40), tol=tol,
        )  # fmt: skip
        keep = np.flatnonzero(w2 > 1e-6 * sigma)
        w2, vec = w2[keep], vec[:, keep]
        o = np.argsort(np.abs(w2 - sigma))[:n_modes]
        o = o[np.argsort(w2[o])]
        out = np.zeros((self.mesh.n_edges, o.size))
        out[idx] = vec[:, o]
        return np.sqrt(w2[o]) / (2 * np.pi), out

    # ------------------------------------------------------------ info
    @property
    def n_unknowns(self):
        """Field unknowns: all edges plus all faces."""
        return self.mesh.n_edges + self.mesh.n_faces
