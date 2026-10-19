# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Interface-conforming triangle meshes for circles in a box (or a disk), for
the high-order DGTD solver.

Algorithm (written from the published idea of force-equilibrium smoothing,
Persson & Strang, SIAM Rev. 46, 329 (2004); no code reused):

1. fixed nodes on every circle (spacing ~h on the circle) and on the outer
   boundary (box sides and corners, or an outer circle);
2. interior nodes from a hexagonal lattice thinned with probability
   (h_min / h(x))^2;
3. a few rounds of spring smoothing (bars push when shorter than the
   target), re-triangulating with Delaunay each time. Free nodes may not
   come closer than `clearance` h to a circle or the box, so every circle
   chord stays a Delaunay edge (no node inside its diametral circle) and
   the mesh conforms to every interface.

Each triangle gets the tag of the circle that contains its centroid
(0 = background, k = circle k - 1 + 1).
"""

import numpy as np
from scipy.spatial import Delaunay

__all__ = ["TriMesh", "circles_mesh"]


class TriMesh:
    """
    Plain triangle mesh: ``nodes`` (n, 2) [m], counter-clockwise ``cells``
    (K, 3), ``tags`` (K,), and the circles [(cx, cy, R), ...] whose arcs are
    represented by mesh edges (used to curve the elements).
    """

    def __init__(self, nodes, cells, tags, circles=(), box=None, outer_circle=None):
        self.nodes = np.asarray(nodes, float)
        cells = np.asarray(cells, np.int64)
        P = self.nodes[cells]
        area2 = (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (
            P[:, 2, 0] - P[:, 0, 0]
        ) * (P[:, 1, 1] - P[:, 0, 1])
        cells[area2 < 0] = cells[area2 < 0][:, [0, 2, 1]]
        self.cells = cells
        self.tags = np.asarray(tags, np.int64)
        self.circles = [tuple(map(float, c)) for c in circles]
        self.box = box
        self.outer_circle = outer_circle

    @property
    def n_cells(self):
        return len(self.cells)

    def quality(self):
        """Radius ratio 2 r_in / r_circ per triangle (1 = equilateral)."""
        P = self.nodes[self.cells]
        a = np.linalg.norm(P[:, 1] - P[:, 2], axis=1)
        b = np.linalg.norm(P[:, 2] - P[:, 0], axis=1)
        c = np.linalg.norm(P[:, 0] - P[:, 1], axis=1)
        s = 0.5 * (a + b + c)
        area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0))
        return 2 * (area / s) / (a * b * c / (4 * area))

    def inradius(self):
        P = self.nodes[self.cells]
        a = np.linalg.norm(P[:, 1] - P[:, 2], axis=1)
        b = np.linalg.norm(P[:, 2] - P[:, 0], axis=1)
        c = np.linalg.norm(P[:, 0] - P[:, 1], axis=1)
        s = 0.5 * (a + b + c)
        area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0))
        return area / s


def _circle_pts(cx, cy, R, h, phase=0.0):
    n = max(8, int(np.ceil(2 * np.pi * R / h)))
    th = phase + 2 * np.pi * np.arange(n) / n
    return np.stack([cx + R * np.cos(th), cy + R * np.sin(th)], 1)


def _edge_pts(p0, p1, size, include_end=False):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    # integrate 1/h along the segment to get the node count
    t = np.linspace(0, 1, 401)
    h = np.array([size(p0 + tt * (p1 - p0)) for tt in t])
    cum = np.concatenate([[0], np.cumsum(0.5 * (1 / h[1:] + 1 / h[:-1]) * np.diff(t) * L)])
    n = max(1, int(np.round(cum[-1])))
    tk = np.interp(np.linspace(0, cum[-1], n + 1), cum, t)
    tk = tk if include_end else tk[:-1]
    return p0 + tk[:, None] * (p1 - p0)


def circles_mesh(
    circles,
    size,
    box=None,
    outer_circle=None,
    clearance=0.6,
    n_smooth=40,
    seed=0,
):
    """
    Interface-conforming triangle mesh of circular inclusions.

    Parameters
    ----------
    circles : list of (cx, cy, R)
        Non-overlapping circles [m] (material interfaces).
    size : callable
        h(point (2,)) -> target edge length [m]. Must be vectorizable over
        points of shape (n, 2) (return (n,)).
    box : (xmin, xmax, ymin, ymax), optional
        Rectangular outer boundary.
    outer_circle : (cx, cy, R), optional
        Circular outer boundary (instead of `box`).
    clearance : float
        Minimum distance of free nodes to circles and boundary, in units of
        the local h.
    n_smooth : int
        Spring-smoothing iterations.

    Returns
    -------
    TriMesh
    """
    rng = np.random.default_rng(seed)
    circles = [tuple(map(float, c)) for c in circles]

    def hv(P):
        return np.asarray(size(np.atleast_2d(P)), float).reshape(-1)

    def h1(p):
        return float(hv(np.asarray(p)[None])[0])

    fixed = []
    for cx, cy, R in circles:
        # spacing from the size field on the circle (minimum over the arc)
        ring = _circle_pts(cx, cy, R, 1e9)
        th = np.linspace(0, 2 * np.pi, 721)[:-1]
        arc = np.stack([cx + R * np.cos(th), cy + R * np.sin(th)], 1)
        harc = hv(arc)
        cum = np.concatenate([[0], np.cumsum(R * (th[1] - th[0]) / harc)])
        n = max(8, int(np.ceil(cum[-1])))
        thk = np.interp(np.linspace(0, cum[-1], n + 1)[:-1], cum, np.append(th, 2 * np.pi))
        ring = np.stack([cx + R * np.cos(thk), cy + R * np.sin(thk)], 1)
        fixed.append(ring)
    if box is not None:
        x0, x1, y0, y1 = box
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for i in range(4):
            fixed.append(_edge_pts(corners[i], corners[(i + 1) % 4], h1))
        bbox = box
    elif outer_circle is not None:
        cx, cy, R = outer_circle
        th = np.linspace(0, 2 * np.pi, 721)[:-1]
        arc = np.stack([cx + R * np.cos(th), cy + R * np.sin(th)], 1)
        cum = np.concatenate([[0], np.cumsum(R * (th[1] - th[0]) / hv(arc))])
        n = max(8, int(np.ceil(cum[-1])))
        thk = np.interp(np.linspace(0, cum[-1], n + 1)[:-1], cum, np.append(th, 2 * np.pi))
        fixed.append(np.stack([cx + R * np.cos(thk), cy + R * np.sin(thk)], 1))
        bbox = (cx - R, cx + R, cy - R, cy + R)
    else:
        raise ValueError("give box or outer_circle")
    fixed = np.vstack(fixed)
    nf = len(fixed)

    # signed distances: to circles (|r - R|) and to the outer boundary
    def dist_circles(P):
        if not circles:
            return np.full(len(P), np.inf), np.zeros(len(P), int), np.zeros(len(P))
        D = np.stack(
            [np.hypot(P[:, 0] - cx, P[:, 1] - cy) - R for cx, cy, R in circles], 1
        )
        k = np.argmin(np.abs(D), 1)
        return np.abs(D[np.arange(len(P)), k]), k, D[np.arange(len(P)), k]

    def dist_outer(P):
        if box is not None:
            x0, x1, y0, y1 = box
            return np.minimum.reduce(
                [P[:, 0] - x0, x1 - P[:, 0], P[:, 1] - y0, y1 - P[:, 1]]
            )
        cx, cy, R = outer_circle
        return R - np.hypot(P[:, 0] - cx, P[:, 1] - cy)

    # initial interior lattice
    x0, x1, y0, y1 = bbox
    hmin = 0.5 * min(hv(fixed).min(), h1(((x0 + x1) / 2, (y0 + y1) / 2)))
    hmin = max(hmin, 1e-3 * max(x1 - x0, y1 - y0))
    xs = np.arange(x0, x1 + hmin, hmin)
    ys = np.arange(y0, y1 + hmin, hmin * np.sqrt(3) / 2)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    X = X + (np.arange(len(ys)) % 2)[:, None] * hmin / 2
    P = np.stack([X.ravel(), Y.ravel()], 1)
    hP = hv(P)
    P = P[rng.random(len(P)) < (hmin / hP) ** 2]
    hP = hv(P)
    dc, _, _ = dist_circles(P)
    P = P[(dc > clearance * hP) & (dist_outer(P) > clearance * hP)]
    side0 = None

    def constrain(P, side):
        hP = hv(P)
        dc, k, sd = dist_circles(P)
        bad = dc < clearance * hP
        if np.any(bad) and circles:
            C = np.array([c[:2] for c in circles])[k[bad]]
            R = np.array([c[2] for c in circles])[k[bad]]
            v = P[bad] - C
            nv = np.linalg.norm(v, axis=1)[:, None]
            s = side[bad] if side is not None else np.sign(sd[bad])
            P[bad] = C + v / np.maximum(nv, 1e-300) * (R + s * clearance * hP[bad])[:, None]
        do = dist_outer(P)
        bad = do < clearance * hP
        if np.any(bad):
            if box is not None:
                bx0, bx1, by0, by1 = box
                m = clearance * hP[bad]
                P[bad, 0] = np.clip(P[bad, 0], bx0 + m, bx1 - m)
                P[bad, 1] = np.clip(P[bad, 1], by0 + m, by1 - m)
            else:
                cx, cy, R = outer_circle
                v = P[bad] - (cx, cy)
                nv = np.linalg.norm(v, axis=1)[:, None]
                P[bad] = (cx, cy) + v / nv * (R - clearance * hP[bad])[:, None]
        return P

    _, kk, sd = dist_circles(P)
    side0 = np.sign(sd) if circles else None
    for it in range(n_smooth):
        allp = np.vstack([fixed, P])
        tri = Delaunay(allp).simplices
        bars = np.unique(np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1), axis=0)
        bv = allp[bars[:, 0]] - allp[bars[:, 1]]
        L = np.linalg.norm(bv, axis=1)
        hb = hv(0.5 * (allp[bars[:, 0]] + allp[bars[:, 1]]))
        L0 = hb * 1.2 * np.sqrt(np.sum(L**2) / np.sum(hb**2))
        F = np.maximum(L0 - L, 0)
        Fv = (F / L)[:, None] * bv
        tot = np.zeros_like(allp)
        np.add.at(tot, bars[:, 0], Fv)
        np.add.at(tot, bars[:, 1], -Fv)
        move = 0.2 * tot[nf:]
        P = P + move
        P = constrain(P, side0)
        if np.max(np.linalg.norm(move, axis=1) / hv(P)) < 1e-3:
            break
    allp = np.vstack([fixed, P])
    tri = Delaunay(allp).simplices
    # drop degenerate hull slivers (zero-area)
    Q = allp[tri]
    a2 = np.abs(
        (Q[:, 1, 0] - Q[:, 0, 0]) * (Q[:, 2, 1] - Q[:, 0, 1])
        - (Q[:, 2, 0] - Q[:, 0, 0]) * (Q[:, 1, 1] - Q[:, 0, 1])
    )
    tri = tri[a2 > 1e-10 * np.median(a2)]
    cen = allp[tri].mean(1)
    tags = np.zeros(len(tri), np.int64)
    for i, (cx, cy, R) in enumerate(circles):
        inside = np.hypot(cen[:, 0] - cx, cen[:, 1] - cy) < R
        tags[inside] = i + 1
        # conformity: no triangle has vertices strictly on both sides
        rv = np.hypot(allp[tri][..., 0] - cx, allp[tri][..., 1] - cy) - R
        tolr = 1e-9 * R
        straddle = (rv > tolr).any(1) & (rv < -tolr).any(1)
        if np.any(straddle):
            raise RuntimeError(f"{straddle.sum()} triangles straddle circle {i}")
    return TriMesh(allp, tri, tags, circles, box, outer_circle)
