# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

r"""
Conforming-Delaunay Steiner edge splitting and projection onto locked CAD curves.

When topological flips (`2-3`, `3-2`, `4-4`, coplanar `2-2`) cannot eliminate an
interior face with negative Voronoi dual length because every fixing flip would
remove a locked CAD-curve or boundary edge, the conforming-Delaunay remedy is to
split an encroached constrained edge at its midpoint projected onto the geometry
it discretises (Ruppert 1995; Shewchuk 1998; Cheng, Dey & Shewchuk 2012).

Split Projector Protocol
------------------------
A split projector ``P`` provides:

* ``P.project(a, b, x) -> np.ndarray (shape (3,)) or None``: project the trial
  split point ``x`` (typically the chord midpoint ``0.5 * (X[a] + X[b])``) of
  edge ``(a, b)`` onto the underlying curve or surface, or return ``None`` when
  the projector cannot classify or project the edge.
* ``P.register(new_node, a, b)``: record that node ``new_node`` was inserted by
  splitting edge ``(a, b)``, inheriting its geometric entity so that the two
  sub-edges ``(a, new_node)`` and ``(new_node, b)`` can be split recursively.

Three-Point Circle Conditioning
-------------------------------
In `feature_curve_projector`, a circular feature curve of radius :math:`R` is
reconstructed locally from three consecutive vertices :math:`(p, a, b)` (and
verified against :math:`(a, b, q)`) with chord lengths :math:`L \ll R`. The
sagitta across a two-edge span is :math:`s \approx L^2 / (2R)`. A coordinate
perturbation of magnitude :math:`\delta x \sim \varepsilon_{\mathrm{mach}} R`
perturbs the sagitta by

.. math::

   \frac{\delta s}{s} \sim \frac{\varepsilon_{\mathrm{mach}} R}{L^2 / R}
   = \varepsilon_{\mathrm{mach}} \left(\frac{R}{L}\right)^2,

so the condition number of the 3-point circle centre and radius grows as
:math:`(R / L_{\min})^2`, bounding the arc-midpoint error on an exact circle by
:math:`\mathcal{O}(\varepsilon_{\mathrm{mach}} (R / L_{\min})^2 R)`.
"""

import numpy as np

from .improve import tet_partials


def _sedge(a, b):
    return (int(a), int(b)) if a < b else (int(b), int(a))


def _sface(a, b, c):
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    return (int(a), int(b), int(c))


def _constraint_face_mask(mesh):
    """Boolean mask (n_faces,) of outer-boundary and material-interface faces."""
    f = mesh.cell_faces.ravel()
    t = np.repeat(mesh.cell_tags, 4)
    lo = np.full(mesh.n_faces, np.iinfo(np.int64).max)
    hi = np.full(mesh.n_faces, np.iinfo(np.int64).min)
    np.minimum.at(lo, f, t)
    np.maximum.at(hi, f, t)
    mask = lo != hi
    mask[mesh.boundary_faces] = True
    return mask


def _halfedge_unit_normal(X, a, b, c, reverse=False):
    """
    Unit normal of triangle (a, b, c) oriented by the half-edge (a -> b).

    For two triangles (a, b, c1) and (a, b, c2) unfolding across their shared
    edge (a, b), calling with ``reverse=False`` on c1 and ``reverse=True`` on c2
    yields identical normals (angle 0) when the two triangles are coplanar and
    unfolded, and the geometric dihedral turn angle otherwise.
    """
    ab = X[b] - X[a]
    ac = X[c] - X[a]
    n_vec = np.cross(ac, ab) if reverse else np.cross(ab, ac)
    n_norm = float(np.linalg.norm(n_vec))
    if n_norm <= 0.0:
        return None
    return n_vec / n_norm


def circle_from_3pts(p1, p2, p3, collinear_tol=1e-12):
    """
    Classify three consecutive curve points (p1, p2, p3) as a line or circle.

    Parameters
    ----------
    p1, p2, p3 : ndarray, shape (3,)
        Cartesian coordinates [m] of three consecutive points along a chain.
    collinear_tol : float, optional
        Relative cross-product threshold below which the three points are
        classified as collinear. Default is ``1e-12``.

    Returns
    -------
    kind : {"line", "circle", "degenerate"}
        Geometric classification of the triple.
    center : ndarray of shape (3,) or None
        Circumcircle centre [m] when ``kind == "circle"``, else ``None``.
    radius : float or None
        Circumcircle radius [m] when ``kind == "circle"``, else ``None``.
    """
    u = p2 - p1
    v = p3 - p2
    lu = float(np.linalg.norm(u))
    lv = float(np.linalg.norm(v))
    if lu <= 0.0 or lv <= 0.0:
        return "degenerate", None, None

    cross_uv = np.cross(u, v)
    sin_theta = float(np.linalg.norm(cross_uv)) / (lu * lv)
    if sin_theta <= collinear_tol:
        if float(np.dot(u, v)) > 0.0:
            return "line", None, None
        return "degenerate", None, None

    a = p2 - p1
    b = p3 - p1
    aa = float(np.dot(a, a))
    bb = float(np.dot(b, b))
    ab = float(np.dot(a, b))
    den = 2.0 * (aa * bb - ab * ab)
    if den <= 0.0:
        return "degenerate", None, None
    alpha = bb * (aa - ab) / den
    beta = aa * (bb - ab) / den
    center = p1 + alpha * a + beta * b
    radius = float(np.linalg.norm(p2 - center))
    return "circle", center, radius


class CallableSplitProjector:
    """Adapter wrapping a plain ``fn(a, b, x)`` callable into the split projector protocol."""

    def __init__(self, fn):
        self._fn = fn

    def project(self, a, b, x):
        return self._fn(int(a), int(b), np.asarray(x, dtype=float))

    def __call__(self, a, b, x):
        return self.project(a, b, x)

    def register(self, new_node, a, b):
        if hasattr(self._fn, "register"):
            self._fn.register(int(new_node), int(a), int(b))

    def unregister(self, new_node, a, b):
        if hasattr(self._fn, "unregister"):
            self._fn.unregister(int(new_node), int(a), int(b))


def normalize_split_projector(proj):
    """Return a split projector object with `.project` and `.register`, or None."""
    if proj is None:
        return None
    if hasattr(proj, "project") and hasattr(proj, "register"):
        return proj
    if callable(proj):
        return CallableSplitProjector(proj)
    raise TypeError(
        "split_projector must be None, a callable(a, b, x), or an object with "
        ".project(a, b, x) and .register(new_node, a, b)"
    )


class FeatureCurveProjector:
    """
    Geometry-only split projector for straight and circular feature curves.

    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 3)
        Mesh node coordinates [m].
    feature_edges : iterable of (int, int)
        Undirected edges belonging to sharp feature curves.
    locked : ndarray of bool, optional
        Node lock mask; if provided, only feature edges whose endpoints are
        both locked are retained as feature edges.
    circle_tol : float, optional
        Relative tolerance on circle centre and radius agreement between the
        two adjacent 3-point circles ``(p, a, b)`` and ``(a, b, q)``.
        Default is ``1e-9``.
    collinear_tol : float, optional
        Relative sine-angle threshold for collinearity. Default is ``1e-12``.
    """

    def __init__(
        self,
        nodes,
        feature_edges,
        locked=None,
        circle_tol=1e-9,
        collinear_tol=1e-12,
    ):
        self.coords = {
            int(i): np.asarray(pt, dtype=float).copy() for i, pt in enumerate(nodes)
        }
        self.circle_tol = float(circle_tol)
        self.collinear_tol = float(collinear_tol)
        self.adj = {}
        self.feature_edges = set()
        self._last_projected = {}

        n_lock = len(locked) if locked is not None else 0
        for u, v in feature_edges:
            a, b = (int(u), int(v)) if u < v else (int(v), int(u))
            if (
                locked is not None
                and a < n_lock
                and b < n_lock
                and not (locked[a] and locked[b])
            ):
                continue
            self.feature_edges.add((a, b))
            self.adj.setdefault(a, set()).add(b)
            self.adj.setdefault(b, set()).add(a)

    def _chain_neighbour(self, node, other):
        """Return the continuation neighbour of `node` opposite `other` along a feature chain."""
        nbrs = [n for n in self.adj.get(node, ()) if n != other]
        if len(nbrs) == 1:
            return nbrs[0]
        if not nbrs:
            return None
        # At a junction of multiple feature curves, select the neighbour most nearly
        # continuing the tangent from `other` through `node` (if within 45 degrees).
        t_in = self.coords[node] - self.coords[other]
        l_in = float(np.linalg.norm(t_in))
        if l_in <= 0.0:
            return None
        t_in = t_in / l_in
        best_n = None
        best_cos = 0.7071067811865475  # cos(45 deg)
        for cand in nbrs:
            t_out = self.coords[cand] - self.coords[node]
            l_out = float(np.linalg.norm(t_out))
            if l_out <= 0.0:
                continue
            c = float(np.dot(t_in, t_out / l_out))
            if c > best_cos:
                best_cos = c
                best_n = cand
        return best_n

    def project(self, a, b, x):
        """
        Project split point `x` of edge `(a, b)` onto the reconstructed feature curve.

        Returns ``ndarray(3)`` on the line or circle when both adjacent chain
        triples ``(p, a, b)`` and ``(a, b, q)`` exist and agree to ``circle_tol``,
        or ``None`` otherwise.
        """
        u, v = (int(a), int(b)) if a < b else (int(b), int(a))
        if (u, v) not in self.feature_edges:
            return None
        if u not in self.coords or v not in self.coords:
            return None

        p = self._chain_neighbour(u, v)
        q = self._chain_neighbour(v, u)
        if p is None or q is None:
            return None

        xp = self.coords[p]
        xu = self.coords[u]
        xv = self.coords[v]
        xq = self.coords[q]

        k1, c1, r1 = circle_from_3pts(xp, xu, xv, collinear_tol=self.collinear_tol)
        k2, c2, r2 = circle_from_3pts(xu, xv, xq, collinear_tol=self.collinear_tol)
        if k1 != k2 or k1 == "degenerate":
            return None

        x_arr = np.asarray(x, dtype=float)
        if k1 == "line":
            # Exact for straight segments: project x onto the line through u and v
            d_uv = xv - xu
            l_uv2 = float(np.dot(d_uv, d_uv))
            if l_uv2 <= 0.0:
                return None
            t = np.clip(float(np.dot(x_arr - xu, d_uv)) / l_uv2, 0.0, 1.0)
            pt = xu + t * d_uv
            self._last_projected[(u, v)] = pt.copy()
            return pt

        # k1 == "circle" == k2: check relative agreement of centre and radius
        r_ref = max(r1, r2)
        if r_ref <= 0.0:
            return None
        if abs(r1 - r2) / r_ref > self.circle_tol:
            return None
        if float(np.linalg.norm(c1 - c2)) / r_ref > self.circle_tol:
            return None

        center = 0.5 * (c1 + c2)
        radius = 0.5 * (r1 + r2)
        radial = x_arr - center
        r_norm = float(np.linalg.norm(radial))
        if r_norm <= 0.0:
            return None
        pt = center + (radius / r_norm) * radial
        self._last_projected[(u, v)] = pt.copy()
        return pt

    def __call__(self, a, b, x):
        return self.project(a, b, x)

    def register(self, new_node, a, b, pt=None):
        """Record `new_node` as subdividing feature edge `(a, b)` into two sub-edges."""
        u, v = (int(a), int(b)) if a < b else (int(b), int(a))
        s = int(new_node)
        if (u, v) not in self.feature_edges:
            return
        if pt is None:
            pt = self._last_projected.get((u, v))
        if pt is None and u in self.coords and v in self.coords:
            pt = 0.5 * (self.coords[u] + self.coords[v])
        if pt is not None:
            self.coords[s] = np.asarray(pt, dtype=float).copy()

        self.feature_edges.discard((u, v))
        if u in self.adj:
            self.adj[u].discard(v)
            self.adj[u].add(s)
        if v in self.adj:
            self.adj[v].discard(u)
            self.adj[v].add(s)
        self.adj.setdefault(s, set()).update((u, v))
        self.feature_edges.add((min(u, s), max(u, s)))
        self.feature_edges.add((min(s, v), max(s, v)))

    def unregister(self, new_node, a, b):
        """Undo a previous `register(new_node, a, b)` call during trial rollback."""
        u, v = (int(a), int(b)) if a < b else (int(b), int(a))
        s = int(new_node)
        e1 = (min(u, s), max(u, s))
        e2 = (min(s, v), max(s, v))
        if e1 not in self.feature_edges and e2 not in self.feature_edges:
            return
        self.feature_edges.discard(e1)
        self.feature_edges.discard(e2)
        self.feature_edges.add((u, v))
        if u in self.adj:
            self.adj[u].discard(s)
            self.adj[u].add(v)
        if v in self.adj:
            self.adj[v].discard(s)
            self.adj[v].add(u)
        self.adj.pop(s, None)
        self.coords.pop(s, None)


def feature_curve_projector(
    mesh,
    locked=None,
    feature_angle_deg=45.0,
    circle_tol=1e-9,
    collinear_tol=1e-12,
):
    """
    Build a geometry-only split projector for CAD-free meshes.

    Feature edges are constraint edges (outer boundary or material interface)
    incident to ``>= 3`` constraint faces (e.g. a triple junction where a
    material interface meets an outer boundary wall) or to ``2`` constraint
    faces whose half-edge oriented normals turn by more than
    ``feature_angle_deg``.

    Choice of default ``feature_angle_deg = 45.0``: on a smooth curved wall of
    radius of curvature :math:`R` discretised at mesh spacing :math:`h`,
    adjacent facets turn by :math:`\\Delta\\theta \\approx h / R` rad
    (:math:`\\approx 5.7^\\circ` at 10 cells per wavelength on the pillbox, and
    :math:`\\le 30^\\circ` on coarse 12-sided cylinders), whereas the CAD
    dihedral at pillbox and cavity rims is :math:`90^\\circ`. Setting the
    threshold to :math:`45^\\circ` separates smooth curved walls from
    :math:`\\ge 90^\\circ` feature curves with a factor-of-two margin on both
    sides.

    Parameters
    ----------
    mesh : SimplexMesh
        Tetrahedral mesh.
    locked : ndarray of bool, optional
        Boolean mask of locked nodes (length ``mesh.n_nodes``).
    feature_angle_deg : float, optional
        Minimum normal turn angle [deg] between two constraint faces to
        classify their shared edge as a feature edge. Default is ``45.0``.
    circle_tol : float, optional
        Relative tolerance on circle centre and radius consistency.
        Default is ``1e-9``.
    collinear_tol : float, optional
        Relative collinearity tolerance. Default is ``1e-12``.

    Returns
    -------
    FeatureCurveProjector
        Projector implementing ``.project(a, b, x)`` and ``.register(new, a, b)``.
    """
    cf_mask = _constraint_face_mask(mesh)
    cf_ids = np.flatnonzero(cf_mask)
    X = mesh.nodes

    # Map each undirected constraint edge (a, b) -> list of third vertices c in incident constraint faces
    edge_opp = {}
    for fid in cf_ids:
        i, j, k = (int(v) for v in mesh.faces[fid])
        edge_opp.setdefault((i, j), []).append(k)
        edge_opp.setdefault((j, k), []).append(i)
        edge_opp.setdefault((i, k), []).append(j)

    cos_thresh = float(np.cos(np.deg2rad(feature_angle_deg)))
    feat_edges = []
    for (a, b), opps in edge_opp.items():
        if len(opps) >= 3:
            feat_edges.append((a, b))
        elif len(opps) == 2:
            n1 = _halfedge_unit_normal(X, a, b, opps[0], reverse=False)
            n2 = _halfedge_unit_normal(X, a, b, opps[1], reverse=True)
            if n1 is not None and n2 is not None and float(np.dot(n1, n2)) < cos_thresh:
                feat_edges.append((a, b))

    return FeatureCurveProjector(
        X,
        feat_edges,
        locked=locked,
        circle_tol=circle_tol,
        collinear_tol=collinear_tol,
    )


def is_edge_encroached(X, cells, edge, cids):
    """
    Return True if any vertex in the edge star of `edge` lies inside its diametral sphere.

    An edge :math:`e = (a, b)` with midpoint :math:`\\mathbf{m}` and length
    :math:`L = \\|\\mathbf{x}_b - \\mathbf{x}_a\\|` is encroached by an apex
    vertex :math:`x` of its edge star when :math:`\\|\\mathbf{x} - \\mathbf{m}\\| < L / 2`,
    equivalently when the angle :math:`\\angle a x b > 90^\\circ`.
    """
    a, b = int(edge[0]), int(edge[1])
    xa, xb = X[a], X[b]
    m = 0.5 * (xa + xb)
    half_len = 0.5 * float(np.linalg.norm(xb - xa))
    if half_len <= 0.0:
        return False
    thresh = half_len * (1.0 - 1e-12)
    seen = {a, b}
    for cid in cids:
        for v in cells[cid]:
            vi = int(v)
            if vi not in seen:
                seen.add(vi)
                if float(np.linalg.norm(X[vi] - m)) < thresh:
                    return True
    return False


def _is_edge_locked(state, edge, locked):
    """Return True if `edge` is a locked boundary edge or locked non-coplanar interface edge."""
    if locked is None:
        return False
    a, b = int(edge[0]), int(edge[1])
    n = len(locked)
    if not (a < n and b < n and locked[a] and locked[b]):
        return False
    ed = _sedge(a, b)
    if ed in state.bnd_edges:
        return True
    cids = state.edge_map.get(ed, ())
    if len(cids) >= 2:
        tags = {state.tags[c] for c in cids}
        if len(tags) > 1 and not is_coplanar_constraint_edge(state, ed):
            return True
    return False


def is_constraint_edge(state, edge):
    """Return True if `edge` lies on the outer boundary or on a material interface."""
    ed = _sedge(int(edge[0]), int(edge[1]))
    if ed in state.bnd_edges:
        return True
    cids = state.edge_map.get(ed, ())
    if len(cids) >= 2:
        tags = {state.tags[c] for c in cids}
        if len(tags) > 1:
            return True
    return False


def is_coplanar_constraint_edge(state, edge, tol=1e-9):
    """
    Return True if all constraint faces (boundary or interface) incident to `edge` are coplanar.

    For a purely interior single-tag edge (zero incident constraint faces),
    returns ``True``. For an edge on the outer boundary or a material interface,
    returns ``True`` only when the incident constraint faces unfold in a single
    common plane within ``tol``, so an unprojected chord-midpoint split does
    not perturb the boundary or interface surface.
    """
    a, b = _sedge(int(edge[0]), int(edge[1]))
    cids = list(state.edge_map.get((a, b), ()))
    if not cids:
        return True

    # Collect third vertices of all constraint faces (a, b, x) in the edge star
    opps = set()
    for cid in cids:
        for x in state.cells[cid]:
            xi = int(x)
            if xi != a and xi != b:
                f = _sface(a, b, xi)
                entries = state.face_map.get(f, ())
                if len(entries) == 1 or state.is_interface_face(f):
                    opps.add(xi)

    if not opps:
        return True
    opps = sorted(opps)
    if len(opps) != 2:
        # A triple junction (>= 3 constraint faces) is only planar if all lie in one plane
        n0 = _halfedge_unit_normal(state.X, a, b, opps[0], reverse=False)
        if n0 is None:
            return False
        for xi in opps[1:]:
            ni = _halfedge_unit_normal(state.X, a, b, xi, reverse=False)
            if ni is None or abs(float(np.dot(ni, n0))) <= 1.0 - tol:
                return False
        return True

    n1 = _halfedge_unit_normal(state.X, a, b, opps[0], reverse=False)
    n2 = _halfedge_unit_normal(state.X, a, b, opps[1], reverse=True)
    if n1 is None or n2 is None:
        return False
    return float(np.dot(n1, n2)) > 1.0 - tol


def candidate_edges_for_face(state, f, locked):
    """
    Build the ordered candidate edge list :math:`E_f` for a stuck negative face `f`.

    Ordering (Brief W4-A Section 3.1):
    1. Locked edges of ``f``,
    2. Locked edges of the two tetrahedra incident to ``f`` (not in ``f``),
    3. Unlocked edges of ``f`` and of the two incident tetrahedra.

    Within each group, edges are ordered with encroached edges first
    (:math:`\\|\\mathbf{x} - \\mathbf{m}\\| < L / 2` for some apex vertex
    :math:`x` in the edge star), then by edge length descending.
    """
    entries = state.face_map.get(f)
    if not entries or len(entries) != 2:
        return []
    (c1, _), (c2, _) = entries
    u, v, w = (int(x) for x in f)
    f_edges = [_sedge(u, v), _sedge(u, w), _sedge(v, w)]
    f_edge_set = set(f_edges)

    tet_edges = []
    seen = set(f_edges)
    for cid in (c1, c2):
        tet = [int(x) for x in state.cells[cid]]
        for i in range(4):
            for j in range(i + 1, 4):
                ed = _sedge(tet[i], tet[j])
                if ed not in seen:
                    seen.add(ed)
                    tet_edges.append(ed)

    all_edges = f_edges + tet_edges
    groups = {0: [], 1: [], 2: []}
    for ed in all_edges:
        cids = list(state.edge_map.get(ed, ()))
        if not cids:
            continue
        is_loc = _is_edge_locked(state, ed, locked)
        if is_loc and ed in f_edge_set:
            grp = 0
        elif is_loc:
            grp = 1
        else:
            grp = 2
        enc = is_edge_encroached(state.X, state.cells, ed, cids)
        length = float(np.linalg.norm(state.X[ed[1]] - state.X[ed[0]]))
        groups[grp].append((0 if enc else 1, -length, ed, is_loc))

    ordered = []
    for grp in (0, 1, 2):
        groups[grp].sort(key=lambda item: (item[0], item[1], item[2]))
        for enc_rank, neg_len, ed, is_loc in groups[grp]:
            ordered.append((ed, is_loc, enc_rank == 0, -neg_len))
    return ordered


def star_volume_change_identity(X, cells_star, tags_star, edge, pt):
    """
    Evaluate the exact boundary-flux volume identity for splitting `edge` at `pt`.

    For an edge :math:`e = (a, b)` with chord midpoint
    :math:`\\mathbf{m} = \\frac{1}{2}(\\mathbf{x}_a + \\mathbf{x}_b)` split at
    :math:`\\mathbf{p}`, each parent tetrahedron :math:`T = (a, b, p_v, q_v)`
    (oriented positively so :math:`\\det(\\mathbf{x}_b - \\mathbf{x}_a,
    \\mathbf{x}_{p_v} - \\mathbf{x}_a, \\mathbf{x}_{q_v} - \\mathbf{x}_a) > 0`)
    is replaced by :math:`T_a = (a, s, p_v, q_v)` and :math:`T_b = (s, b, p_v, q_v)`.
    Because :math:`(\\mathbf{x}_p - \\mathbf{x}_a) - (\\mathbf{x}_p - \\mathbf{x}_b) = \\mathbf{x}_b - \\mathbf{x}_a`,
    the signed volume change of :math:`T` equals

    .. math::

       V(T_a) + V(T_b) - V(T)
       = \\frac{1}{6}\\det\\!\\big(\\mathbf{x}_b - \\mathbf{x}_a,\\,
         \\mathbf{x}_{q_v} - \\mathbf{x}_{p_v},\\,\\mathbf{p} - \\mathbf{m}\\big)
       = \\Big(\\frac{1}{3} A_{F_{p_v}}\\mathbf{n}_{F_{p_v}}
         + \\frac{1}{3} A_{F_{q_v}}\\mathbf{n}_{F_{q_v}}\\Big)\\cdot(\\mathbf{p} - \\mathbf{m}),

    where :math:`F_{p_v} = (a, b, p_v)` and :math:`F_{q_v} = (a, b, q_v)` are the
    two faces of :math:`T` containing :math:`(a, b)` with outward unit normals
    :math:`\\mathbf{n}_{F}` from :math:`T`. Summing over all :math:`T \\in \\mathcal{S}_t(e)`
    cancels every interior face shared by two tetrahedra of the same tag :math:`t`,
    leaving only the outer and material-interface boundary triangles
    :math:`F = (a, b, x) \\subset \\partial\\mathcal{S}_t(e)`:

    .. math::

       \\Delta V_t = \\sum_{F=(a,b,x)\\subset\\partial\\mathcal{S}_t(e)}
       \\frac{1}{3} A_F\\,\\mathbf{n}_{F,t}\\cdot(\\mathbf{p} - \\mathbf{m}).

    Returns
    -------
    dv_total : float
        Total volume change across the star predicted by the outer-boundary sum.
    dv_per_tag : dict of int -> float
        Per-tag volume change predicted by each tag sub-star's boundary sum.
    """
    a, b = int(edge[0]), int(edge[1])
    xa, xb = X[a], X[b]
    ab = xb - xa
    m = 0.5 * (xa + xb)
    disp = np.asarray(pt, dtype=float) - m

    # Accumulate outward area-normal vectors (1/2) * (xb - xa) x (xx - xa)
    # per face x, globally and per tag.
    global_face_flux = {}
    tag_face_flux = {}

    for tet, tag in zip(cells_star, tags_star):
        t_id = int(tag)
        rem = [int(v) for v in tet if int(v) != a and int(v) != b]
        p_v, q_v = rem[0], rem[1]
        # Orient (a, b, p_v, q_v) positively: det(xb - xa, xp - xa, xq - xa) > 0
        mat = np.stack([ab, X[p_v] - xa, X[q_v] - xa], axis=0)
        if float(np.linalg.det(mat)) < 0.0:
            p_v, q_v = q_v, p_v

        # For positively oriented (a, b, p_v, q_v), det(ab, xp - xa, xq - xa) > 0, so
        # ab x (xp - xa) points inward toward q_v and ab x (xq - xa) points outward away from p_v:
        # - Face (a, b, p_v) opposite q_v has outward area normal -0.5 * (ab x (xp - xa))
        # - Face (a, b, q_v) opposite p_v has outward area normal +0.5 * (ab x (xq - xa))
        an_p = -0.5 * np.cross(ab, X[p_v] - xa)
        an_q = 0.5 * np.cross(ab, X[q_v] - xa)

        global_face_flux[p_v] = global_face_flux.get(p_v, 0.0) + an_p
        global_face_flux[q_v] = global_face_flux.get(q_v, 0.0) + an_q

        t_dict = tag_face_flux.setdefault(t_id, {})
        t_dict[p_v] = t_dict.get(p_v, 0.0) + an_p
        t_dict[q_v] = t_dict.get(q_v, 0.0) + an_q

    # Only boundary triangles of the star have non-zero net area-normal
    dv_total = 0.0
    for an_net in global_face_flux.values():
        dv_total += (1.0 / 3.0) * float(np.dot(an_net, disp))

    dv_per_tag = {}
    for t_id, t_dict in tag_face_flux.items():
        dv_t = 0.0
        for an_net in t_dict.values():
            dv_t += (1.0 / 3.0) * float(np.dot(an_net, disp))
        dv_per_tag[t_id] = dv_t

    return dv_total, dv_per_tag


def split_edge_steiner(
    state,
    edge,
    locked=None,
    split_projector=None,
    alpha=0.5,
    min_edge_ratio=0.2,
    orig_edge_len=None,
):
    """
    Split every tetrahedron in the star of `edge` into two at a (possibly projected) point.

    Parameters
    ----------
    state : _RepairState
        Mutable repair state.
    edge : tuple of (int, int)
        Sorted node pair ``(a, b)`` to split.
    locked : ndarray of bool or None
        Node lock flags.
    split_projector : SplitProjector or callable or None
        Split projector ``P`` with ``.project(a, b, x)`` and ``.register(new, a, b)``.
    alpha : float, optional
        Chord fraction in ``(0, 1)`` for the initial split point
        ``(1 - alpha) * X[a] + alpha * X[b]``. Default is ``0.5``.
    min_edge_ratio : float, optional
        Minimum allowed sub-edge length relative to ``orig_edge_len`` (default ``0.2``).
    orig_edge_len : float, optional
        Original unsplit parent edge length [m]; defaults to current length of ``edge``.

    Returns
    -------
    ok : bool
        True if the split succeeded and modified ``state`` in-place.
    info : dict or str
        On failure, the refusal reason string (``"no_projector"``, ``"locked"``,
        ``"volume"``, ``"min_edge"``, ``"topology"``). On success, a dictionary with
        keys ``"new_cids"``, ``"new_node"``, ``"locked_new"``, ``"projected"``,
        ``"on_locked"``, ``"sub_edges"``, ``"orig_edge_len"``, ``"volume_identity_err"``.
    """
    a, b = _sedge(int(edge[0]), int(edge[1]))
    cids = list(state.edge_map.get((a, b), ()))
    if not cids:
        return False, "topology"

    proj = normalize_split_projector(split_projector)
    is_loc = _is_edge_locked(state, (a, b), locked)
    is_constr = is_constraint_edge(state, (a, b))

    xa = state.X[a]
    xb = state.X[b]
    chord_len = float(np.linalg.norm(xb - xa))
    if chord_len <= 0.0:
        return False, "topology"
    ref_len = chord_len if orig_edge_len is None else float(orig_edge_len)

    m_pt = (1.0 - alpha) * xa + alpha * xb
    pt = None
    projected = False

    if is_loc:
        if proj is None:
            return False, "locked"
        pt = proj.project(a, b, m_pt)
        if pt is None:
            return False, "no_projector"
        projected = True
    elif is_constr:
        if proj is not None:
            pt = proj.project(a, b, m_pt)
        if pt is not None:
            projected = True
        else:
            if not is_coplanar_constraint_edge(state, (a, b)):
                return False, "no_projector"
            pt = m_pt
    else:
        pt = m_pt

    pt = np.asarray(pt, dtype=float)
    sub_len_a = float(np.linalg.norm(pt - xa))
    sub_len_b = float(np.linalg.norm(xb - pt))
    if min(sub_len_a, sub_len_b) < min_edge_ratio * ref_len:
        return False, "min_edge"

    s_idx = state.X.shape[0]
    X_try = np.vstack([state.X, pt[None, :]])

    new_tets = []
    tags_new = []
    cells_star = [state.cells[cid] for cid in cids]
    tags_star = [state.tags[cid] for cid in cids]
    v_star_old = float(sum(state.vol[cid] for cid in cids))

    tag_vol_old = {}
    tag_vol_new = {}

    for cid in cids:
        tet = state.cells[cid]
        tag_c = int(state.tags[cid])
        rem = [int(x) for x in tet if int(x) != a and int(x) != b]
        if len(rem) != 2:
            return False, "topology"
        p_v, q_v = rem[0], rem[1]
        # Orient parent (a, b, p_v, q_v) positively on state.X; do NOT re-orient children
        # so that any child tet inverted by projection has signed volume <= 0 < min_v.
        mat_parent = np.stack([xb - xa, state.X[p_v] - xa, state.X[q_v] - xa], axis=0)
        if float(np.linalg.det(mat_parent)) < 0.0:
            p_v, q_v = q_v, p_v

        t_a = (a, s_idx, p_v, q_v)
        t_b = (s_idx, b, p_v, q_v)
        P_ab = X_try[np.asarray([t_a, t_b], dtype=np.int64)]
        v_ab = np.linalg.det(P_ab[:, 1:] - P_ab[:, :1]) / 6.0
        v_old = float(state.vol[cid])
        min_v = max(state.min_vol, 1e-6 * v_old)
        if not np.all(v_ab > min_v):
            return False, "volume"
        if not projected and abs(float(v_ab.sum()) - v_old) > 1e-12 * v_old:
            return False, "volume"

        tag_vol_old[tag_c] = tag_vol_old.get(tag_c, 0.0) + v_old
        tag_vol_new[tag_c] = tag_vol_new.get(tag_c, 0.0) + float(v_ab.sum())
        new_tets.extend([t_a, t_b])
        tags_new.extend([tag_c, tag_c])

    # Verify the exact volume identity (Brief W4-A Section 3.3) to 1e-12 of star volume
    dv_id_total, dv_id_tags = star_volume_change_identity(
        state.X, cells_star, tags_star, (a, b), pt
    )
    v_star_new = float(sum(tag_vol_new.values()))
    dv_actual_total = v_star_new - v_star_old
    vol_id_err = abs(dv_actual_total - dv_id_total)
    if vol_id_err > 1e-12 * v_star_old:
        return False, "volume"
    for t_id, dv_t_id in dv_id_tags.items():
        dv_t_act = tag_vol_new[t_id] - tag_vol_old[t_id]
        t_err = abs(dv_t_act - dv_t_id)
        vol_id_err = max(vol_id_err, t_err)
        if t_err > 1e-12 * v_star_old:
            return False, "volume"

    P = X_try[np.asarray(new_tets, dtype=np.int64)]
    _, h_new, _, vol_new = tet_partials(P)
    state.X = X_try

    removed_bnd = ()
    added_bnd = ()
    if (a, b) in state.bnd_edges:
        removed_bnd = ((a, b),)
        added_bnd_list = [_sedge(a, s_idx), _sedge(s_idx, b)]
        for cid in cids:
            for x in state.cells[cid]:
                xi = int(x)
                if (
                    xi != a
                    and xi != b
                    and len(state.face_map.get(_sface(a, b, xi), ())) == 1
                ):
                    added_bnd_list.append(_sedge(s_idx, xi))
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
    return True, {
        "new_cids": new_cids,
        "new_node": s_idx,
        "locked_new": bool(is_loc),
        "projected": bool(projected),
        "on_locked": bool(is_loc),
        "sub_edges": (_sedge(a, s_idx), _sedge(s_idx, b)),
        "orig_edge_len": ref_len,
        "volume_identity_err": float(vol_id_err / max(v_star_old, 1e-30)),
    }


__all__ = [
    "CallableSplitProjector",
    "FeatureCurveProjector",
    "candidate_edges_for_face",
    "circle_from_3pts",
    "feature_curve_projector",
    "is_constraint_edge",
    "is_coplanar_constraint_edge",
    "is_edge_encroached",
    "normalize_split_projector",
    "split_edge_steiner",
    "star_volume_change_identity",
]
