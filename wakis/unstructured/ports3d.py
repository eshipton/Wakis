# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Lumped and waveguide ports for the tetrahedral FIT solver `UnstructuredFIT3D`.

Every port is a rank-one boundary admittance on the edge voltages,

    j = w (g V - I_s(t)),   V = w^T u,

with a weight vector w on the edges, a terminating conductance g [S] and an
impressed (Norton) current I_s [A]:

* a *lumped* port is an edge path from its + to its - terminal, w = s (the
  orientation signs of the path edges), so V = int_+^- E . dl is the port
  voltage and g = 1 / R its internal conductance;
* a *waveguide* (modal) port lives on a planar set of boundary triangles;
  w = M_G phi is the port Hodge (2D Voronoi or Whitney mass on the port
  triangles) times a 2D eigenmode phi of the port cross-section, normalized
  phi^T M_G phi = 1, so V = phi^T M_G u is the modal amplitude and the
  current pattern w excites that mode only.

The conductance is integrated with the trapezoidal rule (V at n + 1/2 is the
average of the voltages at n and n + 1, the time level of the source and of
the curl term), which keeps the update linear, symmetric and unconditionally
stable for any g; the resulting low-rank implicit term is solved exactly by
the Woodbury identity on top of the solver's diagonal update. The port
records V^{n+1/2} and I_s^{n+1/2}; the current delivered into the structure
is I = I_s - g V, and S-parameters follow from power waves referred to the
admittance of the port's (discrete) mode.

Derivation, discrete dispersion and limits: physics guide section
``docs/physicsguide_sections/unstructured_ports3d.md``.
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import splu
from scipy.spatial import Delaunay

from .mesh import SimplexMesh
from .pml3d import _snap_degenerate_duals, _split_prisms

__all__ = [
    "rectangle_section",
    "coax_section",
    "extrude_section",
    "edge_path",
    "PortPlane",
    "LumpedPort",
    "ModalPort",
    "PortSet",
    "attach_ports",
    "dtft",
    "power_waves",
    "s_matrix",
    "ExtrudedLine",
]


# =============================================================== meshing
def _close(tri_nodes, tri):
    P = tri_nodes[tri]
    u, v = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    area = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    return tri[area > 1e-12 * area.max()]


def rectangle_section(a, b, h, wall=None, jitter=0.2, seed=0):
    """
    Triangulated rectangle [0, a] x [0, b] with an optional conductor frame.

    Nodes on a rectangular lattice (spacing <= h); interior nodes are
    jittered by up to ``jitter`` times the spacing (seeded), boundary nodes
    stay on their side, and the triangulation is Delaunay. With ``wall``,
    a frame of thickness `wall` (two triangles per boundary segment) is added
    outside the rectangle with tag 0, the inside carries tag 1.

    Returns
    -------
    nodes : (n, 2) float, tris : (m, 3) int, tags : (m,) int
    """
    nx, ny = int(np.ceil(a / h)), int(np.ceil(b / h))
    xs, ys = np.linspace(0, a, nx + 1), np.linspace(0, b, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], 1)
    rng = np.random.default_rng(seed)
    inner = (X.ravel() > 0) & (X.ravel() < a) & (Y.ravel() > 0) & (Y.ravel() < b)
    d = min(a / nx, b / ny)
    P[inner] += jitter * d * rng.uniform(-1, 1, (inner.sum(), 2))
    tri = _close(P, Delaunay(P).simplices)
    tags = np.ones(len(tri), int)
    if wall is None:
        return P, tri, tags
    # Perimeter loop (counter-clockwise) and the outward offset frame
    ii = [(i, 0) for i in range(nx)] + [(nx, j) for j in range(ny)]
    ii += [(i, ny) for i in range(nx, 0, -1)] + [(0, j) for j in range(ny, 0, -1)]
    loop = np.array([i * (ny + 1) + j for i, j in ii])
    q = P[loop].copy()
    q[:, 0] = np.where(q[:, 0] <= 0, -wall, np.where(q[:, 0] >= a, a + wall, q[:, 0]))
    q[:, 1] = np.where(q[:, 1] <= 0, -wall, np.where(q[:, 1] >= b, b + wall, q[:, 1]))
    off = len(P) + np.arange(len(loop))
    nodes = np.vstack([P, q])
    m = len(loop)
    ring = []
    for k in range(m):
        p0, p1, q0, q1 = loop[k], loop[(k + 1) % m], off[k], off[(k + 1) % m]
        ring += [(p0, q0, q1), (p0, q1, p1)]
    tri = np.vstack([tri, np.array(ring)])
    tags = np.r_[tags, np.zeros(len(ring), int)]
    return nodes, tri, tags


def coax_section(r_in, r_out, h, wall=None):
    """
    Annulus r_in < r < r_out (tag 1) with the inner disk (tag 0) and an
    optional outer conductor ring of thickness `wall` (tag 2); see
    `wakis.unstructured.disk_mesh` (conforming rings, area-preserving).
    """
    from .meshers import disk_mesh

    radii = [r_in, r_out] + ([r_out + wall] if wall else [])
    m = disk_mesh(radii, h)
    return m.nodes, m.cells, m.cell_tags


def extrude_section(nodes2d, tris2d, tags2d, z, rtol=1e-9, massless_rtol=0.0):
    """
    Tetrahedral mesh of a cross-section extruded along z.

    Every triangle becomes a column of right prisms between the levels `z`,
    each split into three tetrahedra by the min-index diagonal rule
    (`pml3d._split_prisms`). The six vertices of a right prism over a
    triangle lie on one sphere, so the three tetrahedra share one
    circumcentre: the quad diagonals have zero dual area (massless) and the
    inner prism faces zero dual length; these are snapped to exact zeros.
    The circumcentric dual is then the tensor product of the 2D Voronoi dual
    of the section and the 1D dual of `z` (see the physics guide).

    Parameters
    ----------
    nodes2d : (n, 2) float
    tris2d : (m, 3) int
    tags2d : (m,) int
    z : (nz,) float
        Increasing levels [m].
    rtol : float
        Snapping tolerance of the vanishing dual measures.
    massless_rtol : float
        Section edges whose 2D dual length is below ``massless_rtol * L_e``
        get a zero dual area on every level (massless in the 3D solver and
        in `PortPlane`), which removes the timestep collapse of nearly
        co-circular section triangles. Their side faces keep the small dual
        length h_e, so the z-coupling of these edges is not removed and the
        mode separation of `ExtrudedLine` is no longer exact (an O(h_e / L_e)
        perturbation; measured 2e-3 in |S11| on the coax of the tests at
        0.05). Zeroing the side faces too would leave the quad diagonals
        without any reluctance (singular massless block). Default 0.

    Returns
    -------
    SimplexMesh
        Node index k * n + i is section node i at level z[k]; cell tags are
        inherited from the section.
    """
    nodes2d = np.asarray(nodes2d, float)
    tris2d = np.asarray(tris2d, int)
    z = np.asarray(z, float)
    n, nz = len(nodes2d), len(z)
    nodes = np.column_stack(
        [np.tile(nodes2d, (nz, 1)), np.repeat(z, n)]
    )
    key = np.arange(len(nodes))
    tets = [_split_prisms(tris2d + k * n, tris2d + (k + 1) * n, key) for k in range(nz - 1)]
    tags = np.repeat(np.tile(np.asarray(tags2d, int), nz - 1), 3)
    # _split_prisms returns 3 consecutive tets per prism
    mesh = SimplexMesh(nodes, np.vstack(tets), tags)
    mesh.n_snapped = _snap_degenerate_duals(mesh, rtol)
    if massless_rtol > 0:
        m2 = SimplexMesh(nodes2d, tris2d)
        small = m2.dual_length() < massless_rtol * m2.edge_length
        keys2 = m2.edges[:, 0] * n + m2.edges[:, 1]
        E = mesh.edges
        same = (E[:, 0] // n) == (E[:, 1] // n)
        k3 = (E[:, 0] % n) * n + (E[:, 1] % n)
        pos = np.searchsorted(keys2, k3)
        pos = np.minimum(pos, keys2.size - 1)
        hit = same & (keys2[pos] == k3) & small[pos]
        P = mesh.dual_area_partial.tocsr()
        mesh.dual_area_partial = (sp.diags((~hit).astype(float)) @ P).tocsr()
        mesh.n_massless_section = int(small.sum())
    mesh.section_nodes = n
    mesh.z_levels = z
    return mesh


# =============================================================== paths
def edge_path(solver, p_plus, p_minus):
    """
    Shortest path of free edges from the node nearest `p_plus` to the node
    nearest `p_minus` (a lumped-port or lumped-element path).

    Returns
    -------
    edges : ndarray of int
        Global edge indices along the path.
    signs : ndarray of float
        +1 where the edge orientation (low -> high node index) runs along the
        path from + to -, -1 otherwise; V = sum(signs * e[edges]).
    """
    m = solver.mesh
    X = m.nodes
    a = int(np.argmin(np.linalg.norm(X - np.asarray(p_plus), axis=1)))
    b = int(np.argmin(np.linalg.norm(X - np.asarray(p_minus), axis=1)))
    E = m.edges[solver.free]
    L = m.edge_length[solver.free]
    A = sp.coo_matrix((L, (E[:, 0], E[:, 1])), shape=(m.n_nodes,) * 2).tocsr()
    _, pred = shortest_path(A, directed=False, indices=a, return_predecessors=True)
    if pred[b] < 0:
        raise ValueError("no path of free edges between the two points")
    nodes = [b]
    while nodes[-1] != a:
        nodes.append(pred[nodes[-1]])
    nodes = nodes[::-1]
    lookup = {tuple(e): k for k, e in enumerate(map(tuple, m.edges))}
    edges, signs = [], []
    for p, q in zip(nodes[:-1], nodes[1:]):
        edges.append(lookup[(min(p, q), max(p, q))])
        signs.append(1.0 if p < q else -1.0)
    return np.array(edges), np.array(signs)


# =============================================================== port plane
class PortPlane:
    """
    A planar set of boundary triangles of a tetrahedral mesh and its 2D
    edge-element (Whitney 1-form) discretization.

    The tangential traces of the 3D Whitney edge functions on the port
    triangles are the 2D Whitney edge functions of the port mesh, so the 3D
    edge voltages of the port edges are directly the 2D dofs (same edges,
    same orientation). The 2D Hodge is either the diagonal circumcentric one
    (``hodge="voronoi"``, M_G = diag(h_e / L_e), h_e the 2D dual edge length,
    which equals the 3D boundary half-cell Hodge on extruded meshes) or the
    consistent Whitney mass (``"whitney"``).

    Parameters
    ----------
    solver : UnstructuredFIT3D
    point, normal : (3,) float
        A point of the port plane and its normal (pointing into the
        structure or out of it; only the plane matters).
    hodge : {"voronoi", "whitney"}
    tol : float
        Relative plane tolerance.

    Attributes
    ----------
    faces : ndarray of int
        Global (boundary) face indices of the port triangles (faces of PEC
        cells are excluded).
    edges : ndarray of int
        Global edge indices of the port triangle edges (2D edge order).
    mesh2d : SimplexMesh
        Port triangles in in-plane coordinates.
    free : ndarray of bool
        Port edges that are free (non-massless) dofs of the 3D solver.
    dof : ndarray of bool
        Port edges that are not PEC (free or massless).
    M : sparse matrix
        Port Hodge on the port edges (dimensionless: [m / m]).
    K : sparse matrix
        Port curl-curl C2^T diag(1 / A_t) C2 [1/m^2].
    """

    def __init__(self, solver, point, normal, hodge="voronoi", tol=1e-9):
        m = solver.mesh
        n = np.asarray(normal, float)
        n = n / np.linalg.norm(n)
        p = np.asarray(point, float)
        bf = m.boundary_faces
        fc = m.nodes[m.faces[bf]].mean(1)
        scale = np.ptp(m.nodes, axis=0).max()
        on = np.abs((fc - p) @ n) < tol * scale
        # Exclude faces of PEC cells
        bcell = np.zeros(m.n_faces, int)
        bcell[m.cell_faces.ravel()] = np.repeat(np.arange(m.n_cells), 4)
        on &= ~solver.pec_cells[bcell[bf]]
        self.faces = bf[on]
        if self.faces.size == 0:
            raise ValueError("no boundary triangles in the port plane")
        self.solver, self.normal, self.point, self.hodge = solver, n, p, hodge
        tri3 = m.faces[self.faces]
        gnodes = np.unique(tri3)  # sorted -> orientation preserved
        loc = np.full(m.n_nodes, -1)
        loc[gnodes] = np.arange(gnodes.size)
        u1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(n, u1)
        X = m.nodes[gnodes] - p
        self.nodes3d = gnodes
        self.mesh2d = SimplexMesh(np.stack([X @ u1, X @ u2], 1), loc[tri3])
        lookup = {tuple(e): k for k, e in enumerate(map(tuple, m.edges))}
        self.edges = np.array(
            [lookup[(gnodes[i], gnodes[j])] for i, j in self.mesh2d.edges]
        )
        is_free = np.zeros(m.n_edges, bool)
        is_free[solver.free] = True
        self.free = is_free[self.edges]
        # Dofs of the port problem: free and massless edges (not PEC)
        self.dof = ~np.isin(self.edges, solver.fixed)
        m2 = self.mesh2d
        if hodge == "voronoi":
            d = m2.dual_length() / m2.edge_length
            # Edges that are massless in the 3D solver are massless here too
            d[np.isin(self.edges, solver.massless)] = 0.0
            self.M = sp.diags(d).tocsr()
        elif hodge == "whitney":
            self.M = _whitney_mass_2d(m2)
        else:
            raise ValueError("hodge must be 'voronoi' or 'whitney'")
        self.K = (m2.C.T @ sp.diags(1 / m2.cell_volume) @ m2.C).tocsr()

    # ----------------------------------------------------------- modes
    def te_modes(self, n_modes=1, massless_rtol=1e-9):
        """
        Transverse-electric modes of a homogeneous port cross-section.

        Solves K phi = k_c^2 M phi on the free port edges (dense; port meshes
        are small), discarding the gradient null space, and normalizes
        phi^T M phi = 1. Edges with a zero diagonal Hodge (degenerate
        Voronoi faces) are eliminated by static condensation, as in the 3D
        solver.

        Returns
        -------
        kc : ndarray (n_modes,)
            Discrete cutoff wavenumbers [1/m].
        phi : ndarray (n_port_edges, n_modes)
            Mode voltages on the port edges (0 on fixed edges) [V per unit
            modal amplitude].
        """
        from scipy.linalg import eigh

        f = np.flatnonzero(self.dof)
        K = self.K[f][:, f].toarray()
        M = self.M[f][:, f].toarray()
        d = np.diag(M)
        live = np.abs(d) > massless_rtol * np.abs(d).max()
        if self.hodge == "whitney":
            live[:] = True
        a, s = np.flatnonzero(live), np.flatnonzero(~live)
        if s.size:
            Kss = K[np.ix_(s, s)]
            T = -np.linalg.solve(Kss, K[np.ix_(s, a)])
            Ka = K[np.ix_(a, a)] + K[np.ix_(a, s)] @ T
        else:
            Ka, T = K[np.ix_(a, a)], None
        w2, V = eigh(Ka, M[np.ix_(a, a)])
        keep = np.flatnonzero(w2 > 1e-8 * w2.max())[:n_modes]
        phi = np.zeros((self.edges.size, keep.size))
        sub = np.zeros((f.size, keep.size))
        sub[a] = V[:, keep]
        if T is not None:
            sub[s] = T @ V[:, keep]
        phi[f] = sub
        phi /= np.sqrt(np.einsum("ik,ik->k", phi, self.M @ phi))
        return np.sqrt(w2[keep]), phi

    def tem_mode(self):
        """
        Transverse-electromagnetic mode of a two-conductor port.

        The conductors are the connected sets of PEC port edges; the mode
        is minus the discrete gradient of the M-harmonic potential psi
        (psi = 1 on the conductor containing the lowest node index, 0 on
        the other, G^T M G psi = 0 elsewhere). Normalized phi^T M phi = 1.

        Returns
        -------
        phi : ndarray (n_port_edges,)
        volts : float
            Line integral of E from conductor 1 (psi = 1) to conductor 0 per
            unit modal amplitude: V_phys = volts * V_modal.
        """
        m2 = self.mesh2d
        G = m2.G.tocsr()
        fixed = ~self.dof
        E = m2.edges[fixed]
        A = sp.coo_matrix(
            (np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(m2.n_nodes,) * 2
        )
        ncomp, lab = connected_components(A, directed=False)
        on_cond = np.zeros(m2.n_nodes, bool)
        on_cond[E.ravel()] = True
        conds = np.unique(lab[on_cond])
        if conds.size != 2:
            raise ValueError(f"TEM mode needs exactly 2 conductors, found {conds.size}")
        c1 = on_cond & (lab == conds[0])
        c0 = on_cond & (lab == conds[1])
        psi = np.zeros(m2.n_nodes)
        psi[c1] = 1.0
        inner = ~on_cond
        Lap = (G.T @ self.M @ G).tocsr()
        i = np.flatnonzero(inner)
        rhs = -Lap[i][:, np.flatnonzero(c1)] @ np.ones(c1.sum())
        psi[i] = splu(Lap[i][:, i].tocsc()).solve(rhs)
        phi = -(G @ psi)
        phi[fixed] = 0.0
        nrm = np.sqrt(phi @ (self.M @ phi))
        return phi / nrm, 1.0 / nrm

    def to_global(self, phi):
        """Port-edge vector -> global edge vector."""
        out = np.zeros(self.solver.mesh.n_edges)
        out[self.edges] = phi
        return out


def _whitney_mass_2d(m2):
    """Consistent Whitney 1-form mass int W_e . W_e' dA on a triangle mesh."""
    grad = m2.barycentric_gradients()
    A = m2.cell_volume
    Mll = (np.ones((3, 3)) + np.eye(3)) / 12  # int l_i l_j / A
    loc = m2._local_edges
    sgn = np.stack(
        [np.where(m2.cells[:, a] < m2.cells[:, b], 1.0, -1.0) for a, b in loc], 1
    )

    def dot(i, j):
        return np.einsum("ni,ni->n", grad[:, i], grad[:, j])

    rows, cols, vals = [], [], []
    for p, (a, b) in enumerate(loc):
        for q, (c, d) in enumerate(loc):
            val = (
                Mll[a, c] * dot(b, d)
                - Mll[a, d] * dot(b, c)
                - Mll[b, c] * dot(a, d)
                + Mll[b, d] * dot(a, c)
            )
            rows.append(m2.cell_edges[:, p])
            cols.append(m2.cell_edges[:, q])
            vals.append(A * val * sgn[:, p] * sgn[:, q])
    # W_e has units 1/m; the edge dof is a voltage, so M = int W W dA is
    # dimensionless like diag(h / L)
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(m2.n_edges, m2.n_edges),
    )


# =============================================================== ports
class _Port:
    """Rank-one port j = w (g V - I_s), V = w^T u (see module docstring)."""

    def __init__(self, w_global, g, src=None, name=None):
        if g < 0:
            raise ValueError("conductance must be >= 0")
        self.w_global = np.asarray(w_global, float)
        self.g = float(g)
        self.src = src
        self.name = name
        self.V, self.Is = [], []

    def source(self, t):
        return 0.0 if self.src is None else float(self.src(t))

    def reset(self):
        self.V, self.Is = [], []

    def signals(self):
        """(t, V, I_s, I) at the half steps; I = I_s - g V into the structure."""
        V = np.asarray(self.V)
        Is = np.asarray(self.Is)
        t = (np.arange(V.size) + 0.5) * self._dt
        return t, V, Is, Is - self.g * V

    def spectra(self, freqs):
        """DTFT of V and of the current I into the structure at `freqs`."""
        t, V, _, I = self.signals()
        return dtft(V, t, freqs), dtft(I, t, freqs)


class LumpedPort(_Port):
    """
    Discrete port on an edge path: a resistor R (internal resistance) in
    parallel with an impressed current I_s = v_s(t) / R (Norton form of a
    voltage source v_s with series resistance R).

    Parameters
    ----------
    solver : UnstructuredFIT3D
    edges, signs : array_like
        Path edges and orientation signs from the + to the - terminal
        (`edge_path`). V = sum(signs * e[edges]) = phi_+ - phi_- [V].
    R : float
        Internal resistance [Ohm] (also the S-parameter reference).
    v_src : callable, optional
        Open-circuit source voltage v_s(t) [V]; None for a passive load.
    """

    def __init__(self, solver, edges, signs, R=50.0, v_src=None, name=None):
        if R <= 0:
            raise ValueError("R must be > 0")
        w = np.zeros(solver.mesh.n_edges)
        np.add.at(w, np.asarray(edges), np.asarray(signs, float))
        src = None if v_src is None else (lambda t, f=v_src, R=R: f(t) / R)
        super().__init__(w, 1.0 / R, src, name)
        self.R = R
        self.edges = np.asarray(edges)

    def reference_admittance(self, freqs):
        return np.full(np.shape(freqs), 1.0 / self.R)


class ModalPort(_Port):
    """
    Waveguide port on a `PortPlane` with one mode phi (phi^T M phi = 1).

    The weight w = M phi makes the port voltage the modal amplitude
    V = phi^T M u and the impressed current pattern excite mode phi only
    (M-orthogonality of the modes). `g` is the terminating modal
    conductance [S] (e.g. the mode admittance at the centre frequency); the
    S-parameter reference is the admittance `y_ref(f)` of the mode
    (discrete or continuum), not g.

    Parameters
    ----------
    plane : PortPlane
    phi : ndarray (n_port_edges,)
    y_ref : callable
        f [Hz] -> reference modal admittance [S] (real above cutoff).
    g : float, optional
        Terminating conductance [S] (default y_ref(f0) with `f0`).
    i_src : callable, optional
        Impressed modal current I_s(t) [A]; None for a passive port.
    voltage_scale : float
        Physical volts per unit modal amplitude (TEM ports).
    """

    def __init__(self, plane, phi, y_ref, g=None, f0=None, i_src=None, name=None,
                 voltage_scale=1.0):
        # Massless (zero-capacitance) edges cannot carry the port current;
        # the Voronoi port Hodge is already zero there, a Whitney one is not
        w = plane.to_global(np.where(plane.free, plane.M @ phi, 0.0))
        if g is None:
            if f0 is None:
                raise ValueError("pass g or f0")
            g = float(np.real(y_ref(f0)))
        super().__init__(w, g, i_src, name)
        self.plane, self.phi, self.y_ref = plane, phi, y_ref
        self.voltage_scale = voltage_scale

    def reference_admittance(self, freqs):
        return np.asarray(self.y_ref(np.asarray(freqs, float)))


# =============================================================== stepping
class PortSet:
    """
    Port-augmented capacitive update of `UnstructuredFIT3D`.

    Replaces ``solver._update`` (the diagonal semi-implicit update
    D+ u' = a1 u + dt r) by

        (D+ + dt/2 W g W^T) u' = (a1 - dt/2 W g W^T) u + dt (r + W I_s)

    solved exactly with the Woodbury identity (W: n_free x n_ports). The
    ports' edges must be free, non-massless and non-dispersive.
    """

    def __init__(self, solver, ports):
        if solver.hodge != "voronoi":
            raise NotImplementedError("ports need hodge='voronoi'")
        if getattr(solver, "_pml", None) is not None:
            raise NotImplementedError("ports with the spherical PML are not supported")
        if isinstance(solver._update, PortSet):
            raise ValueError("ports already attached")
        self.solver, self.ports = solver, list(ports)
        self.base = solver._update
        pos = np.full(solver.mesh.n_edges, -1)
        pos[solver.free] = np.arange(solver.free.size)
        cols = []
        for p in self.ports:
            nz = np.flatnonzero(p.w_global)
            if np.any(pos[nz] < 0):
                raise ValueError(f"port {p.name!r} touches non-free (PEC or massless) edges")
            cols.append(sp.csc_matrix((p.w_global[nz], (pos[nz], np.zeros(nz.size, int))),
                                      shape=(solver.free.size, 1)))
            p._dt = solver.dt
        for g in self.base.groups:
            hit = np.zeros(solver.free.size, bool)
            hit[g["idx"]] = True
            for c in cols:
                if np.any(hit[c.indices]):
                    raise NotImplementedError("port edges must not be dispersive")
        self.W = sp.hstack(cols).tocsc()  # (n_free, P)
        self.g = np.array([p.g for p in self.ports])
        dt = solver.dt
        self.dt = dt
        self.dinv = dt / self.base.den  # sensitivity of base.step to r
        Q = (self.W.T @ sp.diags(self.dinv) @ self.W).toarray()
        self.Q = Q
        self.S = np.eye(len(self.ports)) + 0.5 * self.g[:, None] * Q
        self._Sinv = np.linalg.inv(self.S)

    # -- interface used by UnstructuredFIT3D.one_step
    def step(self, u, r):
        t = (self.solver.n + 0.5) * self.dt
        Is = np.array([p.source(t) for p in self.ports])
        Vn = self.W.T @ u
        u_star = self.base.step(u, r + self.W @ (Is - 0.5 * self.g * Vn))
        alpha = self._Sinv @ (0.5 * self.g * (self.W.T @ u_star))
        u_new = u_star - self.dinv * (self.W @ alpha)
        Vavg = 0.5 * (Vn + self.W.T @ u_new)
        for k, p in enumerate(self.ports):
            p.V.append(Vavg[k])
            p.Is.append(Is[k])
        return u_new

    def reset(self):
        self.base.reset()
        for p in self.ports:
            p.reset()

    def __getattr__(self, name):  # den, a1, groups, ... of the base update
        if name in ("base",):
            raise AttributeError(name)
        return getattr(self.base, name)

    # -- exact linear step and its transpose (adjoint use, dot-product tests)
    def _massless(self, u):
        s = self.solver
        return -s._Kss.solve(s._Ksf @ u) if s.massless.size else None

    def linear_step(self, u, b, Is):
        """
        One full leapfrog step (b to n+1/2, u to n+1) of the solver with the
        ports, as a linear function of (u, b, I_s); no dispersive media.

        Returns
        -------
        u_new, b_new : ndarray
        """
        s = self.solver
        dt = self.dt
        b = b - dt * (s.Cf @ u)
        es = self._massless(u)
        if es is not None:
            b = b - dt * (s.Cs @ es)
        r = s.Cf.T @ (s.Mnu @ b)
        Vn = self.W.T @ u
        rhs = self.base.a1 * u + dt * (r + self.W @ (Is - 0.5 * self.g * Vn))
        u_star = rhs / self.base.den
        alpha = self._Sinv @ (0.5 * self.g * (self.W.T @ u_star))
        return u_star - self.dinv * (self.W @ alpha), b

    def linear_step_T(self, ub, bb):
        """
        Exact transpose of `linear_step`: given adjoint states (ub, bb) at
        n+1, return (ub, bb, Isb) at n.
        """
        s = self.solver
        dt = self.dt
        # u_new = A^-1 rhs with A = D + dt/2 W g W^T symmetric; A^-1 = (I - Dinv W Sinv g/2 W^T) Dinv
        z = self.W.T @ (self.dinv * ub)
        y = ub / self.base.den  # (D^-1)^T ub
        y = y - (self.W @ (0.5 * self.g * (self._Sinv.T @ z))) / self.base.den
        # rhs = a1 u + dt r + dt W (Is - g/2 W^T u)
        u_b = self.base.a1 * y - dt * (self.W @ (0.5 * self.g * (self.W.T @ y)))
        Is_b = dt * (self.W.T @ y)
        r_b = dt * y
        b_b = bb + s.Mnu @ (s.Cf @ r_b)
        # b_half = b - dt Cf u - dt Cs es(u)
        u_b = u_b - dt * (s.Cf.T @ b_b)
        if s.massless.size:
            cs = s.Cs.T @ b_b
            u_b = u_b + dt * (s._Ksf.T @ s._Kss.solve(cs, trans="T"))
        return u_b, b_b, Is_b

    # -- frequency-domain solve of the same discrete equations
    def frequency_response(self, freqs, Is):
        """
        Port voltages of the time-stepped system at `freqs` from a direct
        solve of its exact discrete-time frequency response.

        With u^n = U exp(i w n dt), b^{n+1/2} = B exp(i w (n+1/2) dt) and
        w~ = 2 sin(w dt / 2) / dt, c = cos(w dt / 2):

            [i w~ D0 + c (G + W g W^T) + K_eff / (i w~)] U = W I_s

        (K_eff: curl-curl with the massless edges condensed), and the
        recorded voltage V^{n+1/2} is c W^T U (time convention exp(+i w t)
        of `dtft`). No dispersive media.

        Parameters
        ----------
        freqs : array_like [Hz]
        Is : array_like (n_ports,)
            Impressed current phasors (at the half steps) [A].

        Returns
        -------
        V : ndarray (n_freqs, n_ports)
            Phasors of the recorded port voltages V^{n+1/2}.
        """
        s = self.solver
        dt = self.dt
        if self.base.groups:
            raise NotImplementedError("dispersive media")
        C0, G = s.cap, s.cond
        nf, ns = s.free.size, s.massless.size
        Cfull = sp.hstack([s.Cf, s.Cs]).tocsr()
        K = (Cfull.T @ s.Mnu @ Cfull).tocsc()
        WgW = (self.W @ sp.diags(self.g) @ self.W.T)
        out = []
        for f in np.atleast_1d(freqs):
            w = 2 * np.pi * f
            wt = 2 * np.sin(w * dt / 2) / dt
            c = np.cos(w * dt / 2)
            Dff = sp.diags(1j * wt * C0 + c * G) + c * WgW
            Z = sp.bmat([[Dff, None], [None, sp.csr_matrix((ns, ns))]]) if ns else Dff
            A = (Z + K / (1j * wt)).tocsc()
            rhs = np.zeros(nf + ns, complex)
            rhs[:nf] = self.W @ np.asarray(Is, complex)
            U = splu(A).solve(rhs)[:nf]
            out.append(c * (self.W.T @ U))
        return np.array(out)


def attach_ports(solver, ports):
    """Attach `ports` to `solver` (replaces its capacitive update); returns the `PortSet`."""
    ps = PortSet(solver, ports)
    solver._update = ps
    solver.ports = ps
    return ps


# =============================================================== S-parameters
def dtft(x, t, freqs):
    """X(f) = sum_n x_n exp(-2 pi i f t_n) dt (time convention exp(+i w t))."""
    t = np.asarray(t, float)
    dt = t[1] - t[0]
    f = np.atleast_1d(np.asarray(freqs, float))
    return np.exp(-2j * np.pi * f[:, None] * t[None]) @ np.asarray(x, float) * dt


def power_waves(V, I, Y):
    """
    Power waves a = (sqrt(Y) V + I / sqrt(Y)) / 2, b = (sqrt(Y) V - I / sqrt(Y)) / 2
    for a real reference admittance Y (Kurokawa 1965); I flows into the
    structure. |a|^2 - |b|^2 = Re(V I*) is the delivered power.
    """
    rY = np.sqrt(np.asarray(Y, complex))
    return (rY * V + I / rY) / 2, (rY * V - I / rY) / 2


def s_matrix(V, I, Y):
    """
    S-matrix from P independent excitations.

    Parameters
    ----------
    V, I : ndarray (n_freqs, n_runs, n_ports)
        Port voltage and current spectra for each run.
    Y : ndarray (n_freqs, n_ports)
        Reference admittances.

    Returns
    -------
    S : ndarray (n_freqs, n_ports, n_ports)
        S = B A^-1 with A[:, port, run], B[:, port, run].
    """
    a, b = power_waves(V, I, np.asarray(Y)[:, None, :])
    A = np.transpose(a, (0, 2, 1))
    B = np.transpose(b, (0, 2, 1))
    return B @ np.linalg.inv(A)


# =============================================================== reduced line
class ExtrudedLine:
    """
    Exact discrete transmission line of one TE / TEM mode on an extruded
    mesh (`extrude_section`) with z-layered homogeneous materials.

    For a section mode with K phi = k_c^2 M phi the leapfrog FIT equations
    separate exactly (tensor-product dual) into a ladder network with, at
    angular frequency w and w~ = 2 sin(w dt / 2) / dt,

        series impedance of layer l:  Z_l = i w~ mu_l D_l
        shunt admittance of node k:   Y_k = sum_{l adj k} (D_l / 2)
                                             (i w~ eps_l + k_c^2 / (i w~ mu_l))

    (D_l layer thickness; half cells at the end nodes). Voltages are at the
    integer steps, currents at the half steps (physics guide, section 3).

    Parameters
    ----------
    z : (N+1,) float
        Node levels [m].
    eps_r, mu_r : (N,) float
        Relative permittivity / permeability of each layer.
    kc : float
        Discrete cutoff wavenumber of the section mode [1/m] (0 for TEM).
    dt : float
        Timestep [s] (0 for the semi-discrete limit).
    """

    def __init__(self, z, eps_r, mu_r, kc, dt):
        self.z = np.asarray(z, float)
        self.D = np.diff(self.z)
        N = self.D.size
        self.eps = eps_0 * np.broadcast_to(np.asarray(eps_r, float), (N,))
        self.mu = mu_0 * np.broadcast_to(np.asarray(mu_r, float), (N,))
        self.kc2 = float(kc) ** 2
        self.dt = float(dt)

    def wt(self, f):
        w = 2 * np.pi * np.asarray(f, float)
        return w if self.dt == 0 else 2 * np.sin(w * self.dt / 2) / self.dt

    def cos_half(self, f):
        return np.cos(np.pi * np.asarray(f, float) * self.dt)

    def theta(self, f, layer=0):
        """Phase per layer theta = beta_h D of a uniform continuation of `layer`."""
        wt = self.wt(f)
        D, eps, mu = self.D[layer], self.eps[layer], self.mu[layer]
        x = (D / 2) * np.sqrt((wt**2 * eps * mu - self.kc2).astype(complex))
        return 2 * np.arcsin(x)

    def beta(self, f, layer=0):
        """Discrete propagation constant beta_h [1/m] of a uniform section."""
        return self.theta(f, layer) / self.D[layer]

    def port_admittance(self, f, side=0):
        """
        Admittance I / V (V integer-level, I half-level) of a semi-infinite
        uniform continuation of the end layer, seen from its end node:
        Y_h = sin(theta) / (w~ mu D).
        """
        layer = 0 if side == 0 else -1
        th = self.theta(f, layer)
        return np.sin(th) / (self.wt(f) * self.mu[layer] * self.D[layer])

    def y_matrix(self, f):
        """2x2 admittance matrix between the end nodes (I_k into node k)."""
        wt = self.wt(f)
        N = self.D.size
        out = []
        for w in np.atleast_1d(wt):
            Y = np.zeros((N + 1, N + 1), complex)
            yl = 1 / (1j * w * self.mu * self.D)
            sh = (self.D / 2) * (1j * w * self.eps + self.kc2 / (1j * w * self.mu))
            for l in range(N):
                Y[l, l] += yl[l] + sh[l]
                Y[l + 1, l + 1] += yl[l] + sh[l]
                Y[l, l + 1] -= yl[l]
                Y[l + 1, l] -= yl[l]
            b, i = [0, N], np.arange(1, N)
            Yr = Y[np.ix_(b, b)] - Y[np.ix_(b, i)] @ np.linalg.solve(
                Y[np.ix_(i, i)], Y[np.ix_(i, b)]
            )
            out.append(Yr)
        return np.array(out)

    def s_matrix(self, f):
        """Discrete-exact S-matrix referred to the end-layer mode admittances."""
        f = np.atleast_1d(np.asarray(f, float))
        Ym = self.y_matrix(f)
        Yref = np.stack([self.port_admittance(f, 0), self.port_admittance(f, 1)], 1)
        out = []
        for k in range(f.size):
            r = np.sqrt(Yref[k].astype(complex))
            A = (np.diag(r) + np.diag(1 / r) @ Ym[k]) / 2
            B = (np.diag(r) - np.diag(1 / r) @ Ym[k]) / 2
            out.append(B @ np.linalg.inv(A))
        return np.array(out)

    def input_reflection(self, f, g_load):
        """
        Reflection at port 0 (reference: end-layer admittance) with node N
        terminated by a trapezoidal conductance g_load (admittance
        g_load cos(w dt / 2) on integer-level voltages).
        """
        f = np.atleast_1d(np.asarray(f, float))
        Ym = self.y_matrix(f)
        yl = g_load * self.cos_half(f)
        yin = Ym[:, 0, 0] - Ym[:, 0, 1] * Ym[:, 1, 0] / (Ym[:, 1, 1] + yl)
        y0 = self.port_admittance(f, 0)
        return (y0 - yin) / (y0 + yin)
