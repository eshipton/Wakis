# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Spherical CFS-PML for the tetrahedral FIT solver `UnstructuredFIT3D`.

The absorbing layer is a shell r0 < r < r1 meshed by radial extrusion of the
outer surface triangulation of an interior ball mesh (`spherical_shell_mesh`).
Every shell prism is a frustum of a cone with its apex at the centre; its six
vertices lie on one sphere, so the three tetrahedra it is split into share
one circumcentre. As a consequence (see the physics guide section
"Spherical CFS-PML for tetrahedral FIT"):

* every edge with a non-zero dual area is radial (a *spoke*) or a chord of a
  sphere r = r_k (tangential at its midpoint); the quad diagonals created by
  the prism split have zero dual area (massless, eliminated by the solver);
* every face with a non-zero dual length is a *sphere face* (normal radial)
  or a *side face* in a plane through the centre (normal tangential); the
  faces inside a prism have zero dual length.

Each coupling dof is therefore aligned with an eigenvector of the
complex-stretching tensor Lambda = diag(s_t^2 / s_r, s_r, s_r) of the
spherical PML (Teixeira & Chew 1997), so the stretched Hodge stays diagonal.
With s = kappa + sigma / (alpha + i w):

    spoke edges   e = (d / M_eps) s_r / s_t^2      (D_r = eps s_t^2 / s_r E_r)
    chord edges   e = (d / M_eps) / s_r
    sphere faces  h = M_nu b s_r / s_t^2
    side faces    h = M_nu b / s_r

where d is the electric flux of the edge's dual face (accumulated from
Ampere's law, d' = C^T h - j) and b the magnetic face flux. Every ratio is a
cascade of two first-order bilinear-transform (trapezoidal) filters, the 3D
counterpart of `solver2d._CFSRatio`.

The whole step (solver leapfrog plus filters) is assembled as four fixed
sparse matrices acting on an extended state, so the update is linear and its
exact transpose (`PML3D.step_T`) is available for adjoint use.
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0

from .mesh import SimplexMesh

__all__ = ["spherical_shell_mesh", "ball_with_pml_mesh", "PML3D", "cfs_profiles"]


# ---------------------------------------------------------------- meshing
def _split_prisms(bot, top, key):
    """
    Split prisms (bot[i], top[i]) into 3 tetrahedra each, conformingly.

    Every quadrilateral side is cut by the diagonal through its vertex of
    smallest `key` (Dompierre, Labbe, Vallet & Camarero 1999): the rule only
    depends on the quad, so neighbouring prisms agree, and it never gives
    the three cyclic diagonals that cannot be tetrahedralized.

    Parameters
    ----------
    bot, top : ndarray (n, 3) of int
        Bottom triangle and the vertices above it (top[:, k] over bot[:, k]).
    key : ndarray of int
        Total order of the vertices (global node index).
    """
    V = np.concatenate([bot, top], 1)  # (n, 6): v0 v1 v2 / v3 v4 v5
    # Symmetries of the prism that move vertex m to position 0
    perm = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 0, 4, 5, 3],
            [2, 0, 1, 5, 3, 4],
            [3, 5, 4, 0, 2, 1],
            [4, 3, 5, 1, 0, 2],
            [5, 4, 3, 2, 1, 0],
        ]
    )
    m = np.argmin(key[V], axis=1)
    W = np.take_along_axis(V, perm[m], axis=1)
    k = key[W]
    # Quad (v1, v2, v5, v4) opposite v0: diagonal v1-v5 or v2-v4
    d15 = np.minimum(k[:, 1], k[:, 5]) < np.minimum(k[:, 2], k[:, 4])
    t15 = np.stack(
        [W[:, [0, 1, 2, 5]], W[:, [0, 1, 5, 4]], W[:, [0, 4, 5, 3]]], 1
    )
    t24 = np.stack(
        [W[:, [0, 1, 2, 4]], W[:, [0, 4, 2, 5]], W[:, [0, 4, 5, 3]]], 1
    )
    return np.where(d15[:, None, None], t15, t24).reshape(-1, 4)


def _snap_degenerate_duals(mesh, rtol=1e-9):
    """
    Zero dual measures that vanish up to round-off.

    In a co-spherical prism the quad diagonals have zero dual area and the
    inner faces zero dual length exactly; the per-tetrahedron pieces cancel
    only to round-off (~1e-17 relative), which would leave tiny, sign-random
    capacitances and reluctances. They are set to exact zeros so that the
    diagonals are massless and the inner faces inert.
    """
    L = mesh.edge_length
    area = mesh.dual_measure()
    zero_e = np.abs(area) < rtol * L**2
    P = mesh.dual_area_partial.tocsr()
    mesh.dual_area_partial = (sp.diags((~zero_e).astype(float)) @ P).tocsr()
    lf = np.sqrt(mesh.face_area)
    length = mesh.dual_length()
    zero_f = np.abs(length) < rtol * lf
    P = mesh.dual_edge_partial.tocsr()
    mesh.dual_edge_partial = (sp.diags((~zero_f).astype(float)) @ P).tocsr()
    return int(zero_e.sum()), int(zero_f.sum())


def spherical_shell_mesh(mesh, radii, tag=None, center=(0.0, 0.0, 0.0), rtol=1e-9):
    """
    Add a layered spherical shell to a tetrahedral ball mesh.

    The outer boundary of `mesh` must be a triangulated sphere (all boundary
    nodes at one radius r0 about `center`, e.g. the outer surface of
    `ball_mesh`). Every boundary node is extruded radially to the radii in
    `radii`, every boundary triangle becomes a column of frustum prisms and
    each prism is split into three tetrahedra (`_split_prisms`). The six
    vertices of every prism are co-spherical, so the circumcentric dual of
    the shell is exactly radial / tangential (see module docstring).

    Parameters
    ----------
    mesh : SimplexMesh
        Interior tetrahedral mesh of the ball r <= r0.
    radii : sequence of float
        Increasing layer radii r_1 < r_2 < ... (> r0) [m].
    tag : int, optional
        Material tag of the shell cells (default: the tag of the boundary
        cells of `mesh`, which must then be unique).
    center : (3,) float
        Centre of the sphere [m].
    rtol : float
        Relative tolerance of the "one radius" check and of the snapping of
        degenerate dual measures.

    Returns
    -------
    SimplexMesh
        Interior plus shell; the interior nodes and cells keep their indices
        and come first. The attribute ``shell_radii`` holds (r0, r_1, ...).
    """
    c = np.asarray(center, float)
    X = mesh.nodes - c
    bf = mesh.faces[mesh.boundary_faces]
    bn = np.unique(bf)
    rb = np.linalg.norm(X[bn], axis=1)
    r0 = rb.mean()
    if np.abs(rb - r0).max() > 1e-6 * r0:
        raise ValueError(
            "spherical_shell_mesh: the boundary nodes are not on one sphere "
            f"(radius {rb.min():.6g} .. {rb.max():.6g})"
        )
    radii = np.asarray(radii, float)
    if radii.ndim != 1 or radii[0] <= r0 * (1 + 1e-9) or np.any(np.diff(radii) <= 0):
        raise ValueError("radii must increase and start above the ball radius")
    if tag is None:
        bcell = np.zeros(mesh.n_faces, int)
        bcell[mesh.cell_faces.ravel()] = np.repeat(np.arange(mesh.n_cells), 4)
        tags = np.unique(mesh.cell_tags[bcell[mesh.boundary_faces]])
        if tags.size != 1:
            raise ValueError("boundary cells carry several tags: pass `tag`")
        tag = int(tags[0])
    nb, nn = bn.size, mesh.n_nodes
    loc = np.full(nn, -1)
    loc[bn] = np.arange(nb)
    dirs = X[bn] / rb[:, None]
    new_nodes = [mesh.nodes]
    for rk in radii:
        new_nodes.append(c + dirs * rk)
    nodes = np.vstack(new_nodes)

    def gid(layer, v):  # global index of boundary node v (global id) at layer
        return v if layer == 0 else nn + (layer - 1) * nb + loc[v]

    key = np.arange(len(nodes))
    tets = [mesh.cells]
    for j in range(1, radii.size + 1):
        bot = gid(j - 1, bf)
        top = gid(j, bf)
        tets.append(_split_prisms(bot, top, key))
    cells = np.vstack(tets)
    ctags = np.r_[mesh.cell_tags, np.full(len(cells) - mesh.n_cells, tag)]
    out = SimplexMesh(nodes, cells, ctags)
    out.n_snapped = _snap_degenerate_duals(out, rtol)
    out.shell_radii = np.r_[r0, radii]
    out.shell_center = c
    out.massless_area_factor = _column_massless_factor(out, bf, gid, radii.size)
    return out


def _column_massless_factor(mesh, bf, gid, n_layers):
    """
    Make the massless decision of the shell chords the same along every
    radial column.

    The solver calls an edge massless when its dual area is below
    ``massless_tol * L^2``. For the chord of one surface edge at radius r
    the dual area grows as r dr and L^2 as r^2, so the ratio falls as dr / r
    and more chords of the same column turn massless the further out they
    are: the discrete medium of the shell changes with r and reflects
    (studies/unstructured-pml3d, the vacuum-shell reference floor). The
    returned factor f (``mesh.massless_area_factor``, used by
    `UnstructuredFIT3D` as ``small = massless_tol L^2 / f``) rescales every
    shell chord so that its test value equals that of its column's chord
    on the first shell sphere r_1. The decision is then made once per column;
    the local stiffness / capacitance ratio of a chord is independent of r
    along a column (both scale as 1 / dr), so keeping a column does not
    lower the stable dt.
    """
    E = mesh.edges
    nn = mesh.n_nodes
    keys = np.minimum(E[:, 0], E[:, 1]) * nn + np.maximum(E[:, 0], E[:, 1])
    order = np.argsort(keys)
    pairs = np.unique(np.sort(np.concatenate([bf[:, [0, 1]], bf[:, [1, 2]], bf[:, [0, 2]]]), axis=1), axis=0)
    area, L = mesh.dual_measure(), mesh.edge_length
    factor = np.ones(mesh.n_edges)
    ref = None
    for j in range(1, n_layers + 1):
        a, b = gid(j, pairs[:, 0]), gid(j, pairs[:, 1])
        k = np.minimum(a, b) * nn + np.maximum(a, b)
        pos = np.searchsorted(keys, k, sorter=order)
        idx = order[pos]
        if not np.array_equal(keys[idx], k):
            raise RuntimeError("shell chord lookup failed")
        q = area[idx] / L[idx] ** 2
        if ref is None:
            ref = q
        ok = q > 0
        factor[idx[ok]] = ref[ok] / q[ok]
    return factor


def ball_with_pml_mesh(radius, h, n_layers, dr=None, n_extra=0, **ball_kw):
    """
    `ball_mesh` of radius `radius` plus `n_extra` vacuum layers and a PML.

    Parameters
    ----------
    radius : float
        Radius of the tetrahedral ball [m].
    h : float
        Target edge length of the ball [m].
    n_layers : int
        Number of shell layers in the PML.
    dr : float, optional
        Radial thickness of every layer (default `h`).
    n_extra : int
        Plain layers between the ball and the PML (same spacing), e.g. to
        build an identical-interior reference domain.

    Returns
    -------
    mesh : SimplexMesh
    pml : dict
        ``dict(r0=..., r1=...)`` for the solver's `pml` argument.
    """
    from .meshers import ball_mesh

    dr = h if dr is None else dr
    ball = ball_mesh([radius], h, **ball_kw)
    n = int(n_extra) + int(n_layers)
    radii = radius + dr * np.arange(1, n + 1)
    m = spherical_shell_mesh(ball, radii)
    r0 = radius + dr * n_extra
    return m, dict(r0=r0, r1=radius + dr * n)


# ---------------------------------------------------------------- profiles
def cfs_profiles(r, r0, r1, sigma_max, kappa_max=1.0, order=3):
    """
    Radial and tangential CFS stretching parameters of the spherical PML.

    s_r(r) = kappa_r + sigma_r / (alpha + i w) with polynomial grading
    xi^m, xi = (r - r0) / (r1 - r0), and s_t = r~ / r with
    r~ = r0 + int_{r0}^r s_r dr', which keeps the CFS form
    s_t = kappa_t + sigma_t / (alpha + i w) for a uniform alpha:

        kappa_t = 1 + (kappa_max - 1) d xi^(m+1) / ((m + 1) r)
        sigma_t = sigma_max d xi^(m+1) / ((m + 1) r)

    Returns
    -------
    kr, sr, kt, st : ndarray
        (sigma in 1/s.)
    """
    r = np.asarray(r, float)
    d = r1 - r0
    xi = np.clip((r - r0) / d, 0, None)
    m = float(order)
    sr = sigma_max * xi**m
    kr = 1 + (kappa_max - 1) * xi**m
    w = d * xi ** (m + 1) / ((m + 1) * r)
    return kr, sr, 1 + (kappa_max - 1) * w, sigma_max * w


# ---------------------------------------------------------------- linear algebra
def _sel(idx, n):
    """Selection rows: (len(idx), n) sparse with ones at (k, idx[k])."""
    idx = np.asarray(idx)
    return sp.csr_matrix((np.ones(idx.size), (np.arange(idx.size), idx)), shape=(idx.size, n))


def _D(v):
    return sp.diags(np.asarray(v, float))


class _Layout:
    """Index blocks of the extended state vector."""

    def __init__(self):
        self.n = 0
        self.blocks = {}

    def add(self, name, size):
        self.blocks[name] = np.arange(self.n, self.n + size)
        self.n += size
        return self.blocks[name]

    def __getitem__(self, name):
        return self.blocks[name]


def _stage(n, updates):
    """
    Sparse stage matrix: identity except the rows in `updates`.

    `updates` is a list of (row indices, sparse expression (len(rows), n));
    every expression refers to the state before the stage.
    """
    keep = np.ones(n)
    mats = []
    for rows, expr in updates:
        keep[rows] = 0
        mats.append(_sel(rows, n).T @ sp.csr_matrix(expr))
    M = _D(keep)
    for E in mats:
        M = M + E
    return M.tocsr()


class _Cascade:
    """
    Two cascaded per-dof bilinear CFS filters y = (s_n1 / s_d1)(s_n2 / s_d2) x.

    Each stage solves kappa_d y + sigma_d P[y] = kappa_n x + sigma_n P[x],
    P' + alpha P = f, trapezoidal: P^n = c1 P^{n-1} + c2 (f^n + f^{n-1})
    (as `solver2d._CFSRatio`). State per stage: P[x], P[y], x^{n-1}, y^{n-1};
    the output y^n is the last block ("y0" of stage 2).
    """

    def __init__(self, name, layout, coeffs, alpha, dt):
        self.name = name
        self.coeffs = coeffs  # [(kn, sn, kd, sd), (kn, sn, kd, sd)]
        m = np.size(coeffs[0][0])
        self.m = m
        a = np.broadcast_to(np.asarray(alpha, float), (m,))
        self.c1 = (1 - a * dt / 2) / (1 + a * dt / 2)
        self.c2 = (dt / 2) / (1 + a * dt / 2)
        self.slots = [
            {k: layout.add(f"{name}{j}_{k}", m) for k in ("Px", "Py", "x0", "y0")}
            for j in range(2)
        ]

    @property
    def out(self):
        return self.slots[1]["y0"]

    def gain(self):
        """Instantaneous gain dy^n/dx^n of the cascade (> 0 for CFS)."""
        g = 1.0
        for kn, sn, kd, sd in self.coeffs:
            g = g * (np.asarray(kn) + np.asarray(sn) * self.c2) / (
                np.asarray(kd) + np.asarray(sd) * self.c2
            )
        return g

    def history(self, n):
        """Output y^n for zero input x^n = 0: the state-dependent part."""
        zero = sp.csr_matrix((self.m, n))
        return [e for rows, e in self.updates(zero, n) if rows is self.out][0]

    def updates(self, x_expr, n):
        """Stage-row updates for input expression x (m x n) of the old state."""
        c1, c2 = _D(self.c1), _D(self.c2)
        ups = []
        x = sp.csr_matrix(x_expr)
        for (kn, sn, kd, sd), sl in zip(self.coeffs, self.slots):
            Px, Py, x0, y0 = (_sel(sl[k], n) for k in ("Px", "Py", "x0", "y0"))
            Px_new = c1 @ Px + c2 @ (x + x0)
            den = 1 / (np.asarray(kd, float) + np.asarray(sd, float) * self.c2)
            y = _D(den) @ (_D(kn) @ x + _D(sn) @ Px_new - _D(sd) @ (c1 @ Py + c2 @ y0))
            Py_new = c1 @ Py + c2 @ (y + y0)
            ups += [(sl["Px"], Px_new), (sl["Py"], Py_new), (sl["x0"], x), (sl["y0"], y)]
            x = y.tocsr()
        return ups


def bilinear_response(coeffs, alpha, dt, f):
    """
    Exact discrete response y/x of a `_Cascade` at frequency f [Hz].

    The trapezoidal rule maps i w to i W = (2 / dt) i tan(w dt / 2), so
    y/x = prod (kn + sn / (alpha + iW)) / (kd + sd / (alpha + iW)).
    """
    iW = 2j / dt * np.tan(np.pi * np.asarray(f) * dt)
    out = 1.0
    for kn, sn, kd, sd in coeffs:
        g = 1 / (alpha + iW)
        out = out * (kn + sn * g) / (kd + sd * g)
    return out


# ---------------------------------------------------------------- PML
class PML3D:
    """
    Spherical CFS-PML attached to an `UnstructuredFIT3D` solver.

    Created by ``UnstructuredFIT3D(mesh, ..., bc="pec", pml=cfg)`` with

    cfg : dict
        ``r0``, ``r1`` (inner / outer radius of the layer [m], the outer one
        backed by the solver's PEC/PMC wall), and optionally ``order`` (3),
        ``R`` (design normal-incidence reflection, 1e-8, sets ``sigma_max``
        = (m + 1) c ln(1/R) / (2 d)), ``sigma_max`` [1/s], ``kappa_max`` (1),
        ``alpha`` [1/s] (CFS pole, uniform; 0 is the plain stretched PML),
        ``center`` ((0, 0, 0)).

    The mesh must be layered in r0 < r < r1 (`spherical_shell_mesh`); the
    constructor classifies every dof and raises if a coupling edge or face
    is neither radial nor tangential.

    The solver's own update (`CapacitiveUpdate`) is replaced by
    ``x <- S4 S3 S2 S1 x + B src`` on the extended state
    x = [u, b, d_pml, filter states], with fixed sparse stages:
    S1 Faraday (b), S2 face filters (h of PML faces), S3 Ampere
    (u outside the layer, flux d of PML edges), S4 edge filters (u of PML
    edges). Conductivity is supported outside the layer, dispersive
    materials are not (next milestone).
    """

    def __init__(self, solver, cfg):
        s = self.solver = solver
        mesh, dt = s.mesh, s.dt
        if s.hodge != "voronoi":
            raise NotImplementedError("pml needs hodge='voronoi'")
        if s.dispersive:
            raise NotImplementedError("pml with dispersive materials: not implemented")
        cfg = dict(cfg)
        self.cfg = cfg
        r0, r1 = float(cfg["r0"]), float(cfg["r1"])
        m = float(cfg.get("order", 3))
        d = r1 - r0
        c = np.asarray(cfg.get("center", (0.0, 0.0, 0.0)), float)
        sig = cfg.get("sigma_max")
        if sig is None:
            free_cells = ~s.pec_cells
            v = 1 / np.sqrt(mu_0 * eps_0 * np.min(s.eps_r[free_cells] * s.mu_r[free_cells]))
            sig = (m + 1) * v * np.log(1 / cfg.get("R", 1e-8)) / (2 * d)
        self.sigma_max = float(sig)
        kmax = float(cfg.get("kappa_max", 1.0))
        alpha = float(cfg.get("alpha", 0.0))
        self.alpha = alpha
        prof = lambda r: cfs_profiles(r, r0, r1, self.sigma_max, kmax, m)  # noqa: E731

        X = mesh.nodes - c
        rn = np.linalg.norm(X, axis=1)
        tol = 1e-7 * r1
        ra, rb = rn[mesh.edges[:, 0]], rn[mesh.edges[:, 1]]
        mid = mesh.edge_midpoint - c
        rmid = np.linalg.norm(mid, axis=1)
        cos_e = np.abs(np.einsum("ij,ij->i", mesh.edge_tangent, mid)) / rmid
        # Edges in the layer: not entirely on or inside the r0 sphere
        in_e = (np.minimum(ra, rb) > r0 - tol) & (np.maximum(ra, rb) > r0 + tol)
        free = s.free
        pos = np.full(mesh.n_edges, -1)
        pos[free] = np.arange(free.size)
        cap_all = np.zeros(mesh.n_edges)
        cap_all[free] = s.cap
        coupling = pos >= 0  # massless and fixed edges carry no capacitance dof
        spoke = in_e & (cos_e > 1 - 1e-9) & coupling
        chord = in_e & (np.abs(ra - rb) < tol) & coupling
        bad = in_e & coupling & ~spoke & ~chord
        if np.any(bad):
            raise ValueError(
                f"pml: {bad.sum()} edges in the layer are neither radial nor "
                "tangential (use spherical_shell_mesh)"
            )
        self.spokes = np.flatnonzero(spoke)
        self.chords = np.flatnonzero(chord)
        if self.spokes.size + self.chords.size == 0:
            raise ValueError("pml: no edges in the layer r0 < r < r1")
        # faces
        F = mesh.faces
        rf = rn[F]
        Mnu = np.asarray(s.Mnu_diag, float)
        fc = X[F].mean(1)
        cos_f = np.abs(np.einsum("ij,ij->i", mesh.face_normal, fc)) / np.linalg.norm(fc, axis=1)
        in_f = (rf.min(1) > r0 - tol) & (rf.max(1) > r0 + tol)
        live_f = np.abs(Mnu) > 0
        sph = in_f & live_f & (rf.max(1) - rf.min(1) < tol)
        side = in_f & live_f & (cos_f < 1e-9) & ~sph
        bad = in_f & live_f & ~sph & ~side
        if np.any(bad):
            raise ValueError(
                f"pml: {bad.sum()} faces in the layer are neither sphere nor "
                "side faces (use spherical_shell_mesh)"
            )
        self.sphere_faces = np.flatnonzero(sph)
        self.side_faces = np.flatnonzero(side)

        one = lambda a: np.ones(np.size(a))  # noqa: E731
        zero = lambda a: np.zeros(np.size(a))  # noqa: E731

        def ratio_r_over_t2(r):  # s_r / s_t^2
            kr, sr, kt, st = prof(r)
            return [(kr, sr, kt, st), (one(r), zero(r), kt, st)]

        def inv_r(r):  # 1 / s_r
            kr, sr, _, _ = prof(r)
            return [(one(r), zero(r), kr, sr), (one(r), zero(r), one(r), zero(r))]

        # Evaluation radii: spoke midpoint, chord node radius (centre of its
        # dual face), sphere-face radius, side-face mid-layer radius
        r_sp = 0.5 * (ra + rb)[self.spokes]
        r_ch = ra[self.chords]
        r_sf = rf[self.sphere_faces].mean(1)
        r_sd = 0.5 * (rf.min(1) + rf.max(1))[self.side_faces]
        self.r_eval = dict(spokes=r_sp, chords=r_ch, sphere=r_sf, side=r_sd)

        e_pml = np.r_[self.spokes, self.chords]
        f_pml = np.r_[self.sphere_faces, self.side_faces]
        # Test hooks (mutations that must make the validation tests fail)
        mut = cfg.get("_mutation")
        if mut == "swap":  # wrong normal projection: radial <-> tangential
            coef_e = _concat_coeffs(inv_r(r_sp), ratio_r_over_t2(r_ch))
            coef_f = _concat_coeffs(inv_r(r_sf), ratio_r_over_t2(r_sd))
        else:
            coef_e = _concat_coeffs(ratio_r_over_t2(r_sp), inv_r(r_ch))
            coef_f = _concat_coeffs(ratio_r_over_t2(r_sf), inv_r(r_sd))
        if mut == "drop_aux":  # auxiliary (convolution) term of s_d dropped
            coef_e = [(kn, sn, kd, 0 * sd) for kn, sn, kd, sd in coef_e]
            coef_f = [(kn, sn, kd, 0 * sd) for kn, sn, kd, sd in coef_f]
        if mut == "no_tangential":  # s_t = 1: planar (uniaxial) stretch only
            coef_e = [(kn, sn, kd, sd) for kn, sn, kd, sd in coef_e]
            for co in (coef_e, coef_f):
                n_sp = r_sp.size if co is coef_e else r_sf.size
                kn, sn, kd, sd = co[0]
                kd, sd = kd.copy(), sd.copy()
                kd[:n_sp], sd[:n_sp] = 1.0, 0.0
                co[0] = (kn, sn, kd, sd)
                kn, sn, kd, sd = co[1]
                kd, sd = kd.copy(), sd.copy()
                kd[:n_sp], sd[:n_sp] = 1.0, 0.0
                co[1] = (kn, sn, kd, sd)
        self._plain_massless = mut == "plain_massless"
        self.e_pml, self.f_pml = e_pml, f_pml
        self.ue = pos[e_pml]  # positions of the PML edges in u

        # ---------------------------------------------------------- layout
        L = self.layout = _Layout()
        nu, nf = free.size, mesh.n_faces
        L.add("u", nu)
        L.add("b", nf)
        L.add("d", e_pml.size)
        self.fe = _Cascade("fe", L, coef_e, alpha, dt)
        self.ff = _Cascade("ff", L, coef_f, alpha, dt)
        n = L.n
        Su, Sb, Sd = _sel(L["u"], n), _sel(L["b"], n), _sel(L["d"], n)

        # S1: Faraday b <- b - dt (C_f u + C_s e_s). The massless voltages
        # e_s enforce C_s^T h^{n+1/2} = 0 with the *stretched* h: the face
        # filter output is affine, h = G M_nu b + eta(filter state), so
        # (C_s^T G M_nu C_s) e_s dt = C_s^T (G M_nu (b - dt C_f u) + eta)
        w = Sb - dt * s.Cf @ Su  # b - dt C_f u
        if s.massless.size:
            G = np.ones(nf)
            G[f_pml] = self.ff.gain()
            eta = _sel(f_pml, nf).T @ self.ff.history(n)  # (nf, n)
            if self._plain_massless:  # mutation: unstretched constraint
                G[:], eta = 1.0, sp.csr_matrix((nf, n))
            GM = _D(G * Mnu)
            Kss = (s.Cs.T @ GM @ s.Cs).tocsc()
            self._Kinv = _sparse_inverse(Kss)
            z = s.Cs.T @ (GM @ w + eta)  # (n_s, n)
            self._Es = (self._Kinv @ z / dt).tocsr()  # e_s as a state expression
            b_new = w - dt * s.Cs @ self._Es
        else:
            self._Es = None
            b_new = w
        S1 = _stage(n, [(L["b"], b_new)])
        # S2: face filters, input M_nu b of the PML faces
        S2 = _stage(n, self.ff.updates(_D(Mnu[f_pml]) @ _sel(L["b"][f_pml], n), n))
        # S3: Ampere. h = M_nu b outside, filter output on the PML faces
        keep_f = np.ones(nf)
        keep_f[f_pml] = 0
        H = _D(Mnu * keep_f) @ Sb + _sel(f_pml, nf).T @ _sel(self.ff.out, n)
        self._R = (s.Cf.T @ H).tocsr()  # r = C^T h - src
        upd = s._update
        S3, self.B = self.ampere_stage(upd.a1, upd.den)
        # S4: edge filters, input d / M_eps; output -> u of the PML edges
        x_e = _D(1 / s.cap[self.ue]) @ Sd
        ups = self.fe.updates(x_e, n)
        S4 = _stage(n, ups)
        S4u = _stage(n, [(L["u"][self.ue], _sel(self.fe.out, n))])
        self.stages = [S1, S2, S3, (S4u @ S4).tocsr()]
        self.x = np.zeros(n)

    def ampere_stage(self, a1, den):
        """
        Stage S3 (Ampere) and the source matrix B for capacitive coefficients
        a1 = C0 - G dt / 2, den = C0 + G dt / 2 of the free edges.

        Outside the layer u <- (a1 u + dt r) / den, r = C^T h - src; PML
        edges accumulate d <- d + dt r (their u rows are overwritten by S4).
        Called once by the constructor and by the adjoint for every design
        (the coefficients of the non-PML edges depend on the permittivity);
        the same inputs give the same matrices bit for bit.
        """
        s, L, dt = self.solver, self.layout, self.solver.dt
        n, nu = L.n, s.free.size
        Su, Sd = _sel(L["u"], n), _sel(L["d"], n)
        R = self._R
        U3 = _D(a1 / den) @ Su + _D(dt / den) @ R
        D3 = Sd + dt * _sel(self.ue, nu) @ R
        S3 = _stage(n, [(L["u"], U3), (L["d"], D3)])
        # source: r -= src -> u -= dt src / den, d -= dt src[ue]
        B = (
            _sel(L["u"], n).T @ _D(-dt / den) + _sel(L["d"], n).T @ (-dt * _sel(self.ue, nu))
        ).tocsr()
        return S3, B

    # ------------------------------------------------------------ state
    def reset(self):
        self.x = np.zeros(self.layout.n)
        self._push()

    def _push(self):
        s, L = self.solver, self.layout
        s.u = self.x[L["u"]].copy()
        s.b = self.x[L["b"]].copy()

    def apply(self, x, src):
        """One step of the extended state: S4 S3 S2 S1 x + B src (pure)."""
        S1, S2, S3, S4 = self.stages
        y = S3 @ (S2 @ (S1 @ x)) + self.B @ src
        return S4 @ y

    def step_T(self, lam):
        """
        Transposed step: returns (S4 S3 S2 S1)^T lam and B^T S4^T lam.

        The second output is the sensitivity to the source vector `src` of
        `apply` (the right-hand side current at the half step).
        """
        S1, S2, S3, S4 = self.stages
        y = S4.T @ lam
        g_src = self.B.T @ y
        return S1.T @ (S2.T @ (S3.T @ y)), g_src

    def step(self, src):
        """Advance the solver one step with source vector `src` (free edges)."""
        self.x = self.apply(self.x, src)
        self._push()

    # ------------------------------------------------------------ diagnostics
    def edge_voltages(self):
        """
        Voltages on all edges [V], massless ones included.

        The massless voltages come from the stretched elimination (the
        solver's own `e` property uses the unstretched one, which is wrong
        for massless edges inside the layer).
        """
        s = self.solver
        out = np.zeros(s.mesh.n_edges)
        out[s.free] = s.u
        if self._Es is not None:
            out[s.massless] = self._Es @ self.x
        return out

    def interior_mask(self):
        """(edges in u, faces) whose primal object lies in r <= r0."""
        s, mesh = self.solver, self.solver.mesh
        c = np.asarray(self.cfg.get("center", (0.0, 0.0, 0.0)), float)
        rn = np.linalg.norm(mesh.nodes - c, axis=1)
        r0 = float(self.cfg["r0"])
        tol = 1e-7 * r0
        ein = rn[mesh.edges].max(1) <= r0 + tol
        fin = rn[mesh.faces].max(1) <= r0 + tol
        return ein[s.free], fin

    def interior_energy(self, u_prev=None):
        """
        Electromagnetic energy [J] of the dofs inside r <= r0.

        With `u_prev` (u one step earlier) the electric part is
        u^n M_eps u^{n+1} / 2, which with b^{n+1/2} M_nu b^{n+1/2} / 2 is the
        energy conserved exactly by the leapfrog in a closed lossless cavity.
        """
        s = self.solver
        ein, fin = self.interior_mask()
        up = s.u if u_prev is None else u_prev
        we = 0.5 * np.sum(s.cap[ein] * s.u[ein] * up[ein])
        Mnu = np.asarray(s.Mnu_diag)
        wm = 0.5 * np.sum(Mnu[fin] * s.b[fin] ** 2)
        return we + wm


def _concat_coeffs(a, b):
    """Concatenate two per-dof cascade coefficient lists [(kn, sn, kd, sd)] * 2."""
    return [tuple(np.r_[x, y] for x, y in zip(sa, sb)) for sa, sb in zip(a, b)]


def _sparse_inverse(K):
    """Inverse of a sparse SPD matrix whose graph has small components."""
    K = sp.csr_matrix(K)
    if K.nnz == np.count_nonzero(K.diagonal()) and np.all(K.diagonal() != 0):
        return sp.diags(1 / K.diagonal()).tocsr()
    from scipy.sparse.csgraph import connected_components

    ncomp, lab = connected_components(K, directed=False)
    order = np.argsort(lab, kind="stable")
    counts = np.bincount(lab)
    if counts.max() > 200:
        raise ValueError("pml: massless edges form a large coupled block")
    rows, cols, vals = [], [], []
    start = 0
    for cnt in counts:
        idx = order[start : start + cnt]
        start += cnt
        blk = np.linalg.inv(K[idx][:, idx].toarray())
        rows.append(np.repeat(idx, cnt))
        cols.append(np.tile(idx, cnt))
        vals.append(blk.ravel())
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=K.shape
    )
