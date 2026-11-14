# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Tetrahedral meshes of an axis-aligned box whose surface refines a Cartesian grid.

The hybrid solver (`HybridFIT3D`) needs a tetrahedral patch that fills a box
of whole Cartesian cells and whose boundary triangles each lie inside one
Cartesian boundary square (closed). Two generators:

* `kuhn_box_mesh`: structured Freudenthal/Kuhn split of every (sub)cube into
  six tetrahedra around its main diagonal. Deterministic, good quality,
  used by the fast tests.
* `delaunay_box_mesh`: unstructured Delaunay mesh of the box with an optional
  spherical hole (PEC) or tagged spherical inclusion. Box-surface nodes are the
  (optionally refined) Cartesian lattice of the box faces; interior nodes are a
  body-centred cubic lattice smoothed by optimal-Delaunay sweeps (Chen & Xu
  2004) with the box-surface nodes fixed and the sphere nodes kept on the
  sphere.
"""

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree

from ..unstructured.mesh import SimplexMesh
from ..unstructured.meshers import _circumcenter_3d, _fibonacci_sphere

__all__ = ["kuhn_box_mesh", "delaunay_box_mesh", "refine_lines", "box_surface_check"]


def refine_lines(v, r):
    """Node lines `v` with every interval split into `r` equal parts."""
    v = np.asarray(v, float)
    if r == 1:
        return v.copy()
    t = np.arange(r) / r
    return np.r_[(v[:-1, None] + np.diff(v)[:, None] * t).ravel(), v[-1]]


def kuhn_box_mesh(xs, ys, zs, r=1, tag=0):
    """
    Kuhn (Freudenthal) tetrahedralization of the tensor grid ``xs x ys x zs``.

    Every cell (after splitting each interval into `r` parts) is cut into the
    six tetrahedra ``x0 -> x0 + u_p1 -> x0 + u_p1 + u_p2 -> x1`` over the
    permutations ``p`` of the axes. Neighbouring cells use the same main
    diagonal direction, so the mesh is conforming, and every boundary square
    is split by one diagonal into two triangles.

    Parameters
    ----------
    xs, ys, zs : array_like
        Node lines of the box [m] (the Cartesian node lines inside the box).
    r : int
        Refinement of each Cartesian interval.
    tag : int
        Cell tag of all tetrahedra.

    Returns
    -------
    SimplexMesh
    """
    g = [refine_lines(v, r) for v in (xs, ys, zs)]
    n = [len(v) for v in g]
    X, Y, Z = np.meshgrid(*g, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    I, J, K = np.meshgrid(*[np.arange(m - 1) for m in n], indexing="ij")
    base = np.stack([I.ravel(), J.ravel(), K.ravel()], 1)
    unit = np.eye(3, dtype=int)
    tets = []
    from itertools import permutations

    for p in permutations(range(3)):
        c = [base]
        for a in p:
            c.append(c[-1] + unit[a])
        tets.append(
            np.stack([np.ravel_multi_index(tuple(q.T), n) for q in c], 1)
        )
    tets = np.concatenate(tets)
    return SimplexMesh(nodes, tets, np.full(len(tets), tag))


def _box_lattice_surface(g, rng, jitter=0.15):
    """
    Box-surface nodes for a tensor lattice `g`.

    The lattice nodes on the six faces, the midpoints of the surface lattice
    edges, and one node near the centre of every surface square (moved by
    up to ``jitter`` of the square size per direction). The half-edges
    between lattice nodes and midpoints are Gabriel edges (their diametral
    circles, radius h/4, are empty: the nearest other node is at least
    ``(1/2 - jitter) h`` from the edge), so they belong to the 2D Delaunay
    triangulation of the face, which the boundary of the 3D Delaunay mesh
    reproduces. Every boundary triangle therefore lies inside one square,
    and the jittered centres put the face nodes in general position (no flat
    tetrahedra from cocircular or cross-edge cospherical sets).
    """
    g2 = [refine_lines(v, 2) for v in g]
    X, Y, Z = np.meshgrid(*g2, indexing="ij")
    I = np.meshgrid(*[np.arange(len(v)) for v in g2], indexing="ij")
    on = np.zeros(X.shape, bool)
    centre = np.zeros(X.shape, bool)
    for a, A in enumerate((X, Y, Z)):
        face = (I[a] == 0) | (I[a] == len(g2[a]) - 1)
        on |= face
        b, c = (a + 1) % 3, (a + 2) % 3
        centre |= face & (I[b] % 2 == 1) & (I[c] % 2 == 1)
    keep = on & ~centre
    pts = [np.stack([X[keep], Y[keep], Z[keep]], 1)]
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        cb, cc = 0.5 * (g[b][1:] + g[b][:-1]), 0.5 * (g[c][1:] + g[c][:-1])
        hb, hc = np.diff(g[b]), np.diff(g[c])
        U, V = np.meshgrid(cb, cc, indexing="ij")
        HU, HV = np.meshgrid(hb, hc, indexing="ij")
        for plane in (g[a][0], g[a][-1]):
            P = np.zeros((U.size, 3))
            P[:, a] = plane
            P[:, b] = U.ravel() + jitter * HU.ravel() * rng.uniform(-1, 1, U.size)
            P[:, c] = V.ravel() + jitter * HV.ravel() * rng.uniform(-1, 1, U.size)
            pts.append(P)
    return np.vstack(pts)


def _odt_box(nodes, kind, lo, hi, sphere, gap):
    """
    One ODT smoothing sweep (Chen & Xu 2004) for a box with a sphere.

    kind: 0 interior (moves, kept `gap` inside the box and away from the
    sphere), 1 box surface (fixed), 2 sphere surface (moves, projected back).
    """
    tet = Delaunay(nodes).simplices
    P = nodes[tet]
    A = P[:, 1:] - P[:, :1]
    vol = np.abs(np.linalg.det(A)) / 6
    ok = vol > 1e-12 * vol.max()
    rhs = 0.5 * np.einsum("nij,nij->ni", A, A)
    cc = np.zeros((len(tet), 3))
    cc[ok] = P[ok, 0] + np.linalg.solve(A[ok], rhs[ok, :, None])[..., 0]
    num = np.zeros_like(nodes)
    den = np.zeros(len(nodes))
    for k in range(4):
        np.add.at(num, tet[ok, k], vol[ok, None] * cc[ok])
        np.add.at(den, tet[ok, k], vol[ok])
    new = num / np.maximum(den, 1e-300)[:, None]
    new[kind == 1] = nodes[kind == 1]
    inner = kind == 0
    new[inner] = np.clip(new[inner], lo + gap, hi - gap)
    if sphere is not None:
        c, R = sphere
        d = new - c
        rn = np.linalg.norm(d, axis=1)
        sph = kind == 2
        new[sph] = c + d[sph] * (R / rn[sph])[:, None]
        # interior nodes: keep `gap` away from the sphere surface
        side = np.linalg.norm(nodes[inner] - c, axis=1) > R
        target = np.where(
            side,
            np.maximum(rn[inner], R + gap),
            np.minimum(rn[inner], R - gap),
        )
        new[inner] = c + d[inner] * (target / np.maximum(rn[inner], 1e-300))[:, None]
    return new


def _resolve_flat(tet, nodes, vtol):
    """
    Remove zero-volume tetrahedra of a degenerate (cospherical) Delaunay mesh
    without breaking conformity.

    A flat tetrahedron F = {a, b, c, d} is a planar convex quad with
    diagonals ac and bd; its neighbours on one side use the faces of ac, on
    the other those of bd. If the two ac-side neighbours share an apex p
    (a 2-2 configuration), they are re-split along bd ({a, b, d, p},
    {b, c, d, p}: positive volume since the quad is convex and p is off its
    plane), else the bd side is flipped to ac. F is then dropped. A side
    without neighbours lies on the mesh boundary (no flip needed). Raises if
    neither side is a 2-2 configuration.
    """
    tet = [tuple(t) for t in tet]
    for _ in range(4):
        P = nodes[np.array(tet)]
        vol = np.abs(np.linalg.det(P[:, 1:] - P[:, :1])) / 6
        flat = np.flatnonzero(vol <= vtol)
        if not flat.size:
            return np.array(tet)
        face_map = {}
        for i, t in enumerate(tet):
            for k in range(4):
                f = tuple(sorted(t[:k] + t[k + 1 :]))
                face_map.setdefault(f, []).append(i)
        alive = np.ones(len(tet), bool)
        new = []
        for i in flat:
            if not alive[i]:
                continue
            q = list(tet[i])
            X = nodes[q]
            nrm = np.cross(X[1] - X[0], X[2] - X[0])
            if np.linalg.norm(nrm) < 1e-14 * np.linalg.norm(X[1] - X[0]) ** 2:
                nrm = np.cross(X[1] - X[0], X[3] - X[0])
            diag = None
            for (i1, i2), (i3, i4) in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
                def side(u, v, w):
                    return np.dot(np.cross(X[v] - X[u], X[w] - X[u]), nrm)
                if side(i1, i2, i3) * side(i1, i2, i4) < 0 and side(i3, i4, i1) * side(i3, i4, i2) < 0:
                    diag = ((q[i1], q[i2]), (q[i3], q[i4]))
            if diag is None:
                raise RuntimeError("flat tetrahedron that is not a convex quad")
            done = False
            for (u, v), (w, x) in (diag, diag[::-1]):
                f1 = tuple(sorted((u, v, w)))
                f2 = tuple(sorted((u, v, x)))
                n1 = [j for j in face_map.get(f1, []) if j != i and alive[j]]
                n2 = [j for j in face_map.get(f2, []) if j != i and alive[j]]
                if not n1 and not n2:  # boundary side: dropping F is enough
                    done = True
                    break
                if len(n1) == 1 and len(n2) == 1:
                    p1 = set(tet[n1[0]]) - {u, v, w}
                    p2 = set(tet[n2[0]]) - {u, v, x}
                    if p1 == p2:
                        pp = p1.pop()
                        alive[n1[0]] = alive[n2[0]] = False
                        new += [(w, x, u, pp), (w, x, v, pp)]
                        done = True
                        break
            if not done:
                raise RuntimeError("flat tetrahedron without a 2-2 neighbourhood")
            alive[i] = False
        tet = [t for t, a in zip(tet, alive) if a] + new
    raise RuntimeError("could not resolve flat tetrahedra")


def delaunay_box_mesh(xs, ys, zs, h, r=1, sphere=None, sphere_tag=1, hole=True,
                      clearance=0.5, smooth=6, seed=0):  # fmt: skip
    """
    Unstructured Delaunay tetrahedral mesh of a box with an optional sphere.

    Parameters
    ----------
    xs, ys, zs : array_like
        Cartesian node lines inside the box [m]; the box surface nodes are
        the lattice ``refine_lines(., r)`` on the six faces, so every boundary
        triangle lies inside one Cartesian square (checked), plus the
        midpoints of the surface lattice edges and one jittered node inside
        every surface square (`_box_lattice_surface`).
    h : float
        Target interior edge length [m] (BCC lattice cube edge ``h``).
    r : int
        Refinement of the box-surface lattice.
    sphere : (center (3,), radius) or None
        Sphere fully inside the box (at least ``2 h`` from its faces).
    sphere_tag : int
        Tag of the tetrahedra inside the sphere (if `hole` is False).
    hole : bool
        True: the sphere is removed (a PEC obstacle: its surface edges are
        boundary edges of the mesh). False: kept and tagged `sphere_tag`.
    clearance : float
        Minimum distance of interior nodes from the box faces and the sphere,
        in units of `h`.
    smooth : int
        ODT smoothing sweeps.
    seed : int
        Seed of the jitter that breaks cospherical degeneracies (interior
        lattice: 1e-3 h; one node per surface square: up to 15 % of the
        square from its centre).

    Returns
    -------
    SimplexMesh
        Tags: 0 outside the sphere, `sphere_tag` inside (hole=False).
    """
    g = [refine_lines(v, r) for v in (xs, ys, zs)]
    lo = np.array([v[0] for v in g])
    hi = np.array([v[-1] for v in g])
    rng = np.random.default_rng(seed)
    surf = _box_lattice_surface(g, rng)
    # BCC lattice of interior nodes
    axes = [np.arange(lo[a] + h / 2, hi[a], h) for a in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    cube = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    pts = np.vstack([cube, cube + h / 2])
    pts += 1e-3 * h * rng.standard_normal(pts.shape)
    gap = clearance * h
    keep = np.all((pts > lo + gap) & (pts < hi - gap), axis=1)
    sph_pts = np.zeros((0, 3))
    if sphere is not None:
        c, R = np.asarray(sphere[0], float), float(sphere[1])
        if np.any(c - R < lo + 2 * h) or np.any(c + R > hi - 2 * h):
            raise ValueError("the sphere must be at least 2 h inside the box")
        rr = np.linalg.norm(pts - c, axis=1)
        keep &= np.abs(rr - R) > gap
        if hole:
            keep &= rr > R
        sph_pts = c + _fibonacci_sphere(R, h)
        sphere = (c, R)
    pts = pts[keep]
    nodes = np.vstack([surf, sph_pts, pts])
    kind = np.r_[np.ones(len(surf), int), np.full(len(sph_pts), 2), np.zeros(len(pts), int)]
    for _ in range(smooth):
        nodes = _odt_box(nodes, kind, lo, hi, sphere, gap)
    if sphere is not None:
        # recover the sphere surface: drop interior nodes inside the diametral
        # balls of the sphere's hull triangles (as `ball_mesh`)
        c, R = sphere
        tri = nodes[kind == 2][ConvexHull(nodes[kind == 2]).simplices]
        cc = _circumcenter_3d(tri)
        rho = np.linalg.norm(tri[:, 0] - cc, axis=1)
        inner = np.flatnonzero(kind == 0)
        tree = cKDTree(nodes[inner])
        drop = np.zeros(len(inner), bool)
        for idx in tree.query_ball_point(cc, rho * (1 + 1e-9)):
            drop[idx] = True
        keepn = np.ones(len(nodes), bool)
        keepn[inner[drop]] = False
        nodes, kind = nodes[keepn], kind[keepn]
    tet = _resolve_flat(Delaunay(nodes).simplices, nodes, 1e-9 * h**3)
    P = nodes[tet]
    vol = np.abs(np.linalg.det(P[:, 1:] - P[:, :1])) / 6
    if np.any(vol <= 1e-9 * h**3):
        raise RuntimeError("degenerate tetrahedra left after flips")
    tags = np.zeros(len(tet), int)
    if sphere is not None:
        c, R = sphere
        inside = np.linalg.norm(nodes[tet].mean(1) - c, axis=1) < R
        if hole:
            tet = tet[~inside]
            tags = tags[~inside]
        else:
            tags[inside] = sphere_tag
    mesh = SimplexMesh(nodes, tet, tags)
    box_surface_check(mesh, xs, ys, zs, allow_other=sphere is not None and hole)
    return mesh


def box_surface_check(mesh, xs, ys, zs, allow_other=False, tol=1e-9):
    """
    Check that `mesh` fills the box ``[xs0, xs-1] x ...`` and that every
    boundary triangle on the box surface lies inside one Cartesian square.

    Returns the boundary-face indices on the box surface. Raises ValueError.
    """
    lines = [np.asarray(v, float) for v in (xs, ys, zs)]
    lo = np.array([v[0] for v in lines])
    hi = np.array([v[-1] for v in lines])
    scale = np.max(hi - lo)
    tol = tol * scale
    X = mesh.nodes
    if np.any(X < lo - tol) or np.any(X > hi + tol):
        raise ValueError("tetrahedral mesh leaves the box")
    bf = mesh.boundary_faces
    F = X[mesh.faces[bf]]  # (nb, 3, 3)
    on_box = np.zeros(len(bf), bool)
    area = np.zeros(len(bf))
    for a in range(3):
        for plane in (lo[a], hi[a]):
            sel = np.all(np.abs(F[:, :, a] - plane) < tol, axis=1)
            on_box |= sel
    if not allow_other and not on_box.all():
        raise ValueError("tetrahedral mesh has boundary faces off the box surface")
    area = mesh.face_area[bf[on_box]]
    box_area = 2 * (
        (hi[0] - lo[0]) * (hi[1] - lo[1])
        + (hi[1] - lo[1]) * (hi[2] - lo[2])
        + (hi[0] - lo[0]) * (hi[2] - lo[2])
    )
    if abs(area.sum() - box_area) > 1e-9 * box_area:
        raise ValueError(
            f"box surface not covered: triangles {area.sum():.6g} vs box {box_area:.6g}"
        )
    # each box triangle within one Cartesian square
    Fb = F[on_box]
    for a in range(3):
        on_a = np.zeros(len(Fb), bool)
        for plane in (lo[a], hi[a]):
            on_a |= np.all(np.abs(Fb[:, :, a] - plane) < tol, axis=1)
        for b in ((a + 1) % 3, (a + 2) % 3):
            cen = Fb[on_a, :, b].mean(1)
            k = np.clip(np.searchsorted(lines[b], cen) - 1, 0, len(lines[b]) - 2)
            v0, v1 = lines[b][k], lines[b][k + 1]
            inside = (Fb[on_a, :, b] >= v0[:, None] - tol) & (
                Fb[on_a, :, b] <= v1[:, None] + tol
            )
            if not inside.all():
                raise ValueError("a box-surface triangle straddles two Cartesian squares")
    return bf[on_box]
