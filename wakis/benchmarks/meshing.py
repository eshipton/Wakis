# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Graded triangular meshes of cylinder clusters (for the triangle baseline arm).

`cluster_mesh` builds an interface-conforming Delaunay mesh of several
circular cylinders inside a disk, with a size field graded with the distance
to the nearest cylinder surface and polar-aligned rings outside (optionally
the polar PML rings of `UnstructuredFIT2D`):

1. every cylinder gets its surface ring (area-preserving polygon) plus one
   staggered ring just inside (fixed) and one just outside (first in the
   greedy pass, so that the rings of two cylinders facing each other across
   a narrow gap cannot collide into slivers), so that the polygon edges are
   Delaunay edges and the interface strip is made of near-equilateral
   triangles (as `wakis.unstructured.graded_disk_mesh`), and concentric
   staggered rings inwards with spacing sqrt(3)/2 h;
2. outside the cluster, beyond the radius R_c where the size field reaches
   h_out, concentric staggered rings about the origin up to the outer
   radius (and the PML rings);
3. the region in between is filled by a greedy Poisson-disk selection from
   candidate rings around each cylinder and around the origin: a candidate
   is accepted if it is farther than 0.8 h(p) from every accepted point
   (fixed points first, then by increasing h).

The node masses of the Voronoi Hodge need a Delaunay mesh without obtuse
boundary triangles; `UnstructuredFIT2D` checks positivity.
"""

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from ..unstructured.mesh import SimplexMesh

__all__ = ["cluster_mesh", "size_field"]


def size_field(cylinders, h_s, h_out, growth=1.2, band=2.0):
    """
    h(p) = h_s within `band` h_s of a cylinder surface (and inside), then
    geometric growth (ratio `growth` per cell) up to h_out.
    """
    C = np.array([cy["center"] for cy in cylinders], float)
    R = np.array([cy["radius"] for cy in cylinders], float)

    def size(p):
        p = np.atleast_2d(p)
        d = np.min(
            np.hypot(p[:, None, 0] - C[None, :, 0], p[:, None, 1] - C[None, :, 1])
            - R[None],
            1,
        )
        d = np.maximum(d - band * h_s, 0.0)
        # cumulative growth: h(d) with dh/dd = (growth - 1)
        h = h_s + (growth - 1) * d
        return np.minimum(h, h_out)

    return size


def _ring(center, r, h, rot=0.0, preserve_area=False):
    n = max(8, int(np.ceil(2 * np.pi * r / h)))
    th = 2 * np.pi * np.arange(n) / n + rot * np.pi / n
    if preserve_area:
        r = r * np.sqrt(2 * np.pi / (n * np.sin(2 * np.pi / n)))
    return np.asarray(center) + r * np.stack([np.cos(th), np.sin(th)], 1)


def cluster_mesh(cylinders, h_s, h_out, r_outer, growth=1.2, pml_layers=0,
                 pml_dr=None, accept=0.8):  # fmt: skip
    """
    Graded triangular mesh of a cylinder cluster in a disk.

    Parameters
    ----------
    cylinders : list of dict
        ``dict(center=(x, y), radius=a)`` [m].
    h_s : float
        Edge length on and inside the cylinders [m].
    h_out : float
        Largest edge length (far region) [m].
    r_outer : float
        Radius of the outer boundary / PML start [m].
    growth : float
        Size growth per cell away from the surfaces.
    pml_layers : int
        Polar-aligned rings appended outside r_outer (radial PML).
    pml_dr : float, optional
        Radial spacing of the PML rings (default h_out).
    accept : float
        Minimum point spacing in units of the local h.

    Returns
    -------
    SimplexMesh
        Tags: 0 host, j + 1 inside cylinder j, len(cylinders) + 1 PML.
    """
    size = size_field(cylinders, h_s, h_out, growth)
    fixed, cand, prio = [], [], []
    dr = np.sqrt(3) / 2 * h_s
    for cy in cylinders:
        c, a = np.asarray(cy["center"], float), float(cy["radius"])
        fixed.append(_ring(c, a, h_s, 0, preserve_area=True))
        fixed.append(_ring(c, a - dr, h_s, 1))
        # the ring just outside is a priority candidate, not fixed: in a gap
        # narrower than ~2 dr the rings of the two cylinders would collide
        # (slivers); the greedy pass keeps the compatible points
        ring = _ring(c, a + dr, h_s, 1)
        for other in cylinders:
            if other is cy:
                continue
            d = np.hypot(*(ring - np.asarray(other["center"])).T) - other["radius"]
            ring = ring[d > 0.6 * h_s]
        prio.append(ring)
        r, k = a - 2 * dr, 0
        while r > 0.6 * h_s:
            fixed.append(_ring(c, r, h_s, k % 2))
            r -= dr
            k += 1
        fixed.append(c[None, :])
        # candidate rings outside, spacing from the size field
        r, k = a + dr, 0
        rmax = r_outer
        while r < rmax:
            hh = float(size(c + np.array([r, 0.0]))[0])
            r += np.sqrt(3) / 2 * hh
            cand.append(_ring(c, r, hh, k % 2))
            k += 1
    # polar rings about the origin from R_c to r_outer
    Rb = max(np.hypot(*cy["center"]) + cy["radius"] for cy in cylinders)
    rc = Rb + (h_out - h_s) / (growth - 1) + 2 * h_s if growth > 1 else Rb + 4 * h_s
    rings, r, k = [], r_outer, 0
    while r > rc:
        rings.append(r)
        r -= np.sqrt(3) / 2 * h_out
    for k, r in enumerate(rings):
        fixed.append(
            _ring(
                (0.0, 0.0), r, h_out, k % 2, preserve_area=(k == 0 and not pml_layers)
            )
        )
    r_in = rings[-1] if rings else r_outer
    # candidates about the origin (hex lattice of varying density)
    for hh in np.unique(np.minimum(h_out, h_s * growth ** np.arange(0, 40))):
        xs = np.arange(-r_in, r_in + hh, hh)
        ys = np.arange(-r_in, r_in + hh, hh * np.sqrt(3) / 2)
        X, Y = np.meshgrid(xs, ys)
        X = X + (np.arange(len(ys)) % 2)[:, None] * hh / 2
        P = np.stack([X.ravel(), Y.ravel()], 1)
        hs = size(P)
        P = P[
            (hs >= hh / growth)
            & (hs <= hh * growth)
            & (np.hypot(*P.T) < r_in - 0.5 * hh)
        ]
        cand.append(P)
    fixed = np.vstack(fixed)
    cand = np.vstack(cand)
    # drop candidates inside cylinders or beyond the inner polar ring
    inside = np.zeros(len(cand), bool)
    for cy in cylinders:
        inside |= (
            np.hypot(*(cand - np.asarray(cy["center"])).T) < cy["radius"] + 1.5 * dr
        )
    cand = cand[~inside & (np.hypot(*cand.T) < r_in - 0.5 * size(cand))]
    hc = size(cand)
    order = np.argsort(hc, kind="stable")
    cand, hc = cand[order], hc[order]
    if prio:
        pr = np.vstack(prio)
        cand = np.vstack([pr, cand])
        hc = np.concatenate([np.full(len(pr), h_s), hc])
    # greedy Poisson-disk selection
    tree_fixed = cKDTree(fixed)
    dfix, _ = tree_fixed.query(cand)
    keep_mask = dfix > accept * hc
    cand, hc = cand[keep_mask], hc[keep_mask]
    accepted = []
    grid = {}
    cell = accept * h_out
    for p, hp in zip(cand, hc):
        key = (int(np.floor(p[0] / cell)), int(np.floor(p[1] / cell)))
        ok = True
        rad = int(np.ceil(accept * hp / cell))
        for di in range(-rad, rad + 1):
            for dj in range(-rad, rad + 1):
                for q in grid.get((key[0] + di, key[1] + dj), ()):
                    if (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 < (
                        accept * min(hp, q[2])
                    ) ** 2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            accepted.append(p)
            grid.setdefault(key, []).append((p[0], p[1], hp))
    pts = [fixed] + ([np.array(accepted)] if accepted else [])
    if pml_layers:
        outer = _ring((0.0, 0.0), r_outer, h_out, 0)
        d = h_out if pml_dr is None else pml_dr
        for j in range(1, int(pml_layers) + 1):
            pts.append(outer * (r_outer + j * d) / r_outer)
    nodes = np.vstack(pts)
    nodes = np.unique(nodes, axis=0)
    tri = Delaunay(nodes).simplices
    cent = nodes[tri].mean(1)
    tags = np.zeros(len(tri), int)
    for j, cy in enumerate(cylinders):
        tags[np.hypot(*(cent - np.asarray(cy["center"])).T) < cy["radius"]] = j + 1
    tags[np.hypot(*cent.T) > r_outer] = len(cylinders) + 1
    return SimplexMesh(nodes, tri, tags)
