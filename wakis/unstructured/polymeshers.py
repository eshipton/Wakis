# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Generators of polygonal meshes for `PolygonFIT2D`.

* `hybrid_circles_mesh`: axis-aligned squares fill the bulk (coarse, h_out)
  and the inside of each circular scatterer (fine, h_in); a conforming
  Delaunay triangle patch fills only the band around every material
  interface. Square and triangle edges are shared at the staircase
  boundaries of the patch, and every dual point (square centre, triangle
  circumcentre) projects onto the edge midpoint, so the diagonal FIT Hodge
  stays exact (orthogonal) across the whole hybrid mesh.
* `voronoi_disk_mesh`: the primal cells are the Voronoi polygons of a point
  set (hexagons in the bulk), with the Delaunay triangulation as the dual
  grid (dual points = generators). Mirrored generator pairs across each
  circle make Voronoi edges conform to it.

Both are pure NumPy/SciPy (qhull); no external mesher.
"""

import numpy as np
from scipy.spatial import Delaunay, Voronoi, cKDTree

from .polymesh import PolyMesh

__all__ = ["hybrid_circles_mesh", "voronoi_disk_mesh", "graded_size"]


def graded_size(h_in, h_out, band, growth=1.2):
    """Size field h(d) [m] of the distance d to the nearest interface:
    h_in for d <= band, then geometric growth (`growth` per h) up to h_out."""

    def size(d):
        d = np.asarray(d, float)
        x = np.maximum(d - band, 0.0) / h_in
        return np.minimum(h_out, h_in * growth**x)

    return size


def _ring_pts(center, r, n, phase=0.0):
    th = 2 * np.pi * (np.arange(n) + phase) / n
    return np.asarray(center) + r * np.stack([np.cos(th), np.sin(th)], 1)


def _square_grid(h, lo, hi, origin=0.0):
    """Node coordinates k h + origin covering [lo, hi]."""
    k0 = int(np.floor((lo - origin) / h + 1e-9))
    k1 = int(np.ceil((hi - origin) / h - 1e-9))
    return origin + h * np.arange(k0, k1 + 1)


def hybrid_circles_mesh(
    circles,
    h_in,
    h_out,
    half_width,
    band=None,
    growth=1.2,
    patch_margin=1.5,
    inner_squares=True,
    inner_band=None,
    min_spacing=0.75,
    smooth=4,
    pml_cells=0,
    outer_radius=None,
):
    """
    Hybrid square/triangle mesh of circular scatterers in a square domain.

    Parameters
    ----------
    circles : sequence of (cx, cy, radius)
        Material interfaces [m] (non-overlapping).
    h_in : float
        Edge length at the interfaces and inside the scatterers [m].
    h_out : float
        Edge length of the background squares [m] (e.g. n h_in for equal
        cells per local wavelength).
    half_width : float
        The bulk squares cover [-half_width, half_width]^2 [m] (rounded up to
        a multiple of h_out), plus `pml_cells` extra square layers.
    band : float, optional
        Width with h = h_in on both sides of each interface (default 2 h_in).
    growth : float
        Geometric growth of the triangle size per h beyond the band.
    patch_margin : float
        Extra distance of the patch boundary, in units of h_out, beyond the
        point where the size field reaches h_out.
    inner_squares : bool
        Fill each scatterer with h_in squares up to `inner_band` (default
        2 h_in) inside the interface; False: triangles everywhere inside.
    min_spacing : float
        Minimum point distance in units of the local size.
    smooth : int
        Laplacian smoothing sweeps of the free triangle points.
    pml_cells : int
        Square layers added outside the domain (for a PML).
    outer_radius : float, optional
        Disk-shaped domain of this radius [m] instead of the square: the
        squares stop about 1.5 h_out inside it and a thin conforming
        triangle annulus (size h_out) reaches the circular boundary (for
        ``bc='abc'``, the same domain as the triangle arms).

    Returns
    -------
    mesh : PolyMesh
        Tags: i for cells inside circle i, len(circles) for the background.
    info : dict
        Square/triangle counts and the domain extent.
    """
    circles = np.atleast_2d(np.asarray(circles, float))
    C, R = circles[:, :2], circles[:, 2]
    band = 2 * h_in if band is None else band
    inner_band = 2 * h_in if inner_band is None else inner_band
    size = graded_size(h_in, h_out, band, growth)
    d_grow = band + h_in * np.log(h_out / h_in) / np.log(growth)
    d_patch = d_grow + patch_margin * h_out

    def iface_dist(P):
        r = np.linalg.norm(P[:, None, :] - C[None], axis=2)
        d = np.abs(r - R[None])
        return d.min(1), d.argmin(1), r

    # ---- outer squares
    L = h_out * np.ceil(half_width / h_out - 1e-9)
    Lp = L + pml_cells * h_out
    xs = _square_grid(h_out, -Lp, Lp)
    nx = len(xs)
    X0, Y0 = np.meshgrid(xs[:-1], xs[:-1], indexing="xy")
    cx0, cy0 = X0.ravel(), Y0.ravel()
    # distance from each circle centre to the square (clamped)
    dmin = np.full(cx0.size, np.inf)
    for (ccx, ccy), rr in zip(C, R):
        dx = np.maximum(np.maximum(cx0 - ccx, ccx - (cx0 + h_out)), 0)
        dy = np.maximum(np.maximum(cy0 - ccy, ccy - (cy0 + h_out)), 0)
        dmin = np.minimum(dmin, np.hypot(dx, dy) - rr)
    keep_out = dmin > d_patch
    if outer_radius is not None:
        far = np.max(
            [np.hypot(cx0 + dx, cy0 + dy) for dx in (0, h_out) for dy in (0, h_out)], 0
        )
        keep_out &= far < outer_radius - 1.5 * h_out
    out_cells = np.flatnonzero(keep_out)

    # ---- inner squares (per circle, aligned to its centre)
    inner = []  # (node coords array, cells as 4 coords)
    if inner_squares:
        for (ccx, ccy), rr in zip(C, R):
            g = _square_grid(h_in, -rr, rr)
            gx, gy = np.meshgrid(g[:-1], g[:-1], indexing="xy")
            gx, gy = gx.ravel(), gy.ravel()
            corners = np.stack(
                [np.hypot(gx + dx, gy + dy) for dx in (0, h_in) for dy in (0, h_in)], 1
            )
            ok = corners.max(1) < rr - inner_band
            inner.append((ccx + gx[ok], ccy + gy[ok]))

    # ---- assemble square nodes
    quad_pts, quads = [], []
    # outer
    iy, ix = np.divmod(out_cells, nx - 1)
    n0 = ix + nx * iy
    used = np.zeros(nx * nx, bool)
    oq = np.stack([n0, n0 + 1, n0 + 1 + nx, n0 + nx], 1)
    used[oq.ravel()] = True
    XX, YY = np.meshgrid(xs, xs, indexing="xy")
    grid_pts = np.stack([XX.ravel(), YY.ravel()], 1)
    remap = -np.ones(nx * nx, np.int64)
    remap[used] = np.arange(used.sum())
    quad_pts.append(grid_pts[used])
    quads.append(remap[oq])
    offset = int(used.sum())
    for gx, gy in inner:
        if gx.size == 0:
            continue
        P = np.concatenate(
            [np.stack([gx, gy], 1), np.stack([gx + h_in, gy], 1),
             np.stack([gx + h_in, gy + h_in], 1), np.stack([gx, gy + h_in], 1)]
        )  # fmt: skip
        key = np.round(P / (1e-3 * h_in)).astype(np.int64)
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        pts = np.zeros((len(uk), 2))
        pts[inv] = P
        quad_pts.append(pts)
        quads.append(offset + inv.reshape(4, -1).T)
        offset += len(uk)
    sq_nodes = np.vstack(quad_pts)
    sq_cells = np.vstack(quads)
    n_sq_nodes = len(sq_nodes)
    sq_centers = sq_nodes[sq_cells].mean(1)
    sq_size = np.linalg.norm(sq_nodes[sq_cells[:, 0]] - sq_nodes[sq_cells[:, 1]], axis=1)

    # ---- triangle patch points
    sq_tree = cKDTree(sq_centers)
    fixed = [sq_nodes]
    iface = []
    for (ccx, ccy), rr in zip(C, R):
        n = max(8, int(np.ceil(2 * np.pi * rr / h_in)))
        rho = rr * np.sqrt(2 * np.pi / (n * np.sin(2 * np.pi / n)))  # area-preserving
        iface.append(_ring_pts((ccx, ccy), rho, n))
    if not inner_squares:
        iface.append(C)  # centre points
    if outer_radius is not None:
        n_o = max(8, int(np.ceil(2 * np.pi * outer_radius / h_out)))
        rho_o = outer_radius * np.sqrt(2 * np.pi / (n_o * np.sin(2 * np.pi / n_o)))
        iface.append(_ring_pts((0.0, 0.0), rho_o, n_o))
    iface = np.vstack(iface)
    accepted = np.vstack([sq_nodes, iface])
    n_fixed = len(accepted)
    tree = cKDTree(accepted)
    new_pts = []
    for ci, ((ccx, ccy), rr) in enumerate(zip(C, R)):
        for sgn in (+1, -1):
            r, k = rr, 0
            while True:
                d = abs(r - rr)
                step = np.sqrt(3) / 2 * float(size(d + 0.5 * size(d)))
                r = r + sgn * step
                k += 1
                if r <= 0.5 * h_in:
                    break
                if abs(r - rr) > d_patch + 2 * h_out and (sgn > 0 or inner_squares):
                    break
                h_loc = float(size(abs(r - rr)))
                n = max(6, int(np.ceil(2 * np.pi * r / h_loc)))
                P = _ring_pts((ccx, ccy), r, n, phase=0.5 * (k % 2))
                dI, own, _ = iface_dist(P)
                P = P[own == ci]
                if len(P) == 0:
                    continue
                hP = size(dI[own == ci])
                # outside the circumcircles of the squares (keeps square edges Delaunay)
                dd, jj = sq_tree.query(P)
                ok = dd > 0.72 * sq_size[jj] + 0.3 * hP
                P, hP = P[ok], hP[ok]
                if len(P) == 0:
                    continue
                if new_pts:
                    tree = cKDTree(np.vstack([accepted] + new_pts))
                dd, _ = tree.query(P)
                ok = dd > min_spacing * hP
                P = P[ok]
                if len(P):
                    new_pts.append(P)
    if outer_radius is not None:
        r, k = rho_o, 0
        while r > outer_radius - 1.5 * h_out - 2 * h_out:
            r -= np.sqrt(3) / 2 * h_out
            k += 1
            n = max(6, int(np.ceil(2 * np.pi * r / h_out)))
            P = _ring_pts((0.0, 0.0), r, n, phase=0.5 * (k % 2))
            dd, jj = sq_tree.query(P)
            P = P[dd > 0.72 * sq_size[jj] + 0.3 * h_out]
            if len(P) == 0:
                continue
            tree = cKDTree(np.vstack([accepted] + new_pts))
            dd, _ = tree.query(P)
            P = P[dd > min_spacing * h_out]
            if len(P):
                new_pts.append(P)
    free_pts = np.vstack(new_pts) if new_pts else np.zeros((0, 2))
    pts = np.vstack([accepted, free_pts])

    in_square = _square_membership(sq_nodes, sq_cells, sq_size)

    def triangulate(pts):
        tri = Delaunay(pts).simplices
        cen = pts[tri].mean(1)
        keep = ~in_square(cen)
        keep &= np.all(np.abs(cen) < Lp, axis=1)
        if outer_radius is not None:
            keep &= np.hypot(*cen.T) < outer_radius
        return tri[keep]

    tri = triangulate(pts)
    for _ in range(smooth):
        # Laplacian smoothing of the free points (indices >= n_fixed)
        nb_sum = np.zeros_like(pts)
        cnt = np.zeros(len(pts))
        for a, b in ((0, 1), (1, 2), (2, 0)):
            np.add.at(nb_sum, tri[:, a], pts[tri[:, b]])
            np.add.at(nb_sum, tri[:, b], pts[tri[:, a]])
            np.add.at(cnt, tri[:, a], 1)
            np.add.at(cnt, tri[:, b], 1)
        mv = np.arange(n_fixed, len(pts))
        mv = mv[cnt[mv] > 0]
        target = nb_sum[mv] / cnt[mv, None]
        new = 0.5 * pts[mv] + 0.5 * target
        # do not cross an interface or enter a square circumcircle
        dI_old = iface_dist(pts[mv])
        dI_new = iface_dist(new)
        same_side = np.all(
            np.sign(dI_old[2] - R[None]) == np.sign(dI_new[2] - R[None]), axis=1
        )
        dd, jj = sq_tree.query(new)
        ok = same_side & (dd > 0.72 * sq_size[jj] + 0.2 * size(dI_new[0]))
        if outer_radius is not None:
            ok &= np.hypot(*new.T) < outer_radius - 0.3 * h_out
        pts[mv[ok]] = new[ok]
        tri = triangulate(pts)

    # ---- assemble
    nodes = pts
    cells = [list(c) for c in sq_cells] + [list(t) for t in tri]
    allc = np.vstack([nodes[sq_cells].mean(1), nodes[tri].mean(1)])
    r = np.linalg.norm(allc[:, None, :] - C[None], axis=2)
    inside = r < R[None]
    tags = np.where(inside.any(1), inside.argmax(1), len(R))
    mesh = PolyMesh(nodes, cells, tags)
    if outer_radius is None:
        area_ok = abs(mesh.cell_area.sum() / (2 * Lp) ** 2 - 1)
    else:
        area_ok = abs(mesh.cell_area.sum() / (np.pi * outer_radius**2) - 1)
    if area_ok > 1e-9:
        raise RuntimeError(f"hybrid mesh does not tile the domain (area error {area_ok:.2e})")
    info = dict(
        n_squares=int(len(sq_cells)),
        n_triangles=int(len(tri)),
        half_width=float(L),
        half_width_pml=float(Lp),
        d_patch=float(d_patch),
        n_square_nodes=int(n_sq_nodes),
    )
    return mesh, info


def _square_membership(nodes, cells, size):
    """Function testing whether points lie strictly inside any square."""
    lo = nodes[cells].min(1)
    hi = nodes[cells].max(1)
    cen = 0.5 * (lo + hi)
    tree = cKDTree(cen)

    def test(P):
        _, J = tree.query(P, k=min(4, len(cen)))
        J = np.atleast_2d(J.T).T if J.ndim == 1 else J
        out = np.zeros(len(P), bool)
        for j in J.T:
            tol = 1e-9 * size[j][:, None]
            out |= np.all((P > lo[j] + tol) & (P < hi[j] - tol), axis=1)
        return out

    return test


# ----------------------------------------------------------------------------
def voronoi_disk_mesh(
    radii, size, merge_tol=1e-6, collapse=0.0, center=(0.0, 0.0), lattice=True, lloyd=0
):
    """
    Voronoi-primal mesh of concentric disks, conforming to every circle.

    Generators are placed on concentric rings (radial spacing sqrt(3)/2 s(r),
    azimuthal spacing s(r), staggered by half a spacing: hexagonal Voronoi
    cells) and, inside the first ring, on a hexagonal lattice. Every circle
    in `radii` is straddled by a *mirrored* ring pair at r_k -/+ delta with
    equal angles: the Voronoi edge of each pair lies on the tangent of the
    circle, so the cells conform to the interface (the polygon is
    area-preserving). The last radius is the outer boundary: the cells of
    the outer mirror ring are discarded.

    Primal: Voronoi vertices/edges/cells; dual: Delaunay (dual points = the
    generators), so the diagonal FIT Hodge is exact (orthogonal) with
    h_{e,C} = |p_i - p_j| / 2 > 0 on every edge and positive node masses
    (Delaunay triangle areas) without any Delaunay-quality condition.
    Voronoi vertices closer than ``merge_tol * s`` (cocircular generators)
    are merged. With ``collapse > 0`` every Voronoi edge shorter than
    ``collapse`` times its dual (Delaunay) edge is collapsed to a point: a
    short edge carries a stiffness h/L that sets the global time step, and
    collapsing it is the L -> 0 limit of that stiffness (equal node values).
    The merged vertex moves by at most L/2, which leaves the adjacent edges
    orthogonal to O(collapse) (reported by `PolyMesh.orthogonality_defect`).

    Parameters
    ----------
    radii : sequence of float
        Interface radii, last = outer boundary [m].
    size : callable or float
        Generator spacing s(r) [m]. The Voronoi edges are ~ s / sqrt(3).
    lattice : bool
        Hexagonal lattice of generators in the centre (else rings to r ~ s).
    collapse : float
        Relative edge-collapse threshold (0: off).
    lloyd : int
        Lloyd (centroidal Voronoi) sweeps of the free generators; the
        mirrored interface rings stay fixed and every generator stays in its
        material region.

    Returns
    -------
    PolyMesh
        Tags k for r_{k-1} < r < r_k (0 innermost).
    """
    radii = np.sort(np.atleast_1d(np.asarray(radii, float)))
    if not callable(size):
        s0 = float(size)

        def size(r):
            return s0

    ctr = np.asarray(center, float)
    gens = []
    tags = []
    fixed = []
    # radial positions: per annulus, mirrored pair around each radius
    r_prev = 0.0
    first_inner_ring = None
    for k, rk in enumerate(radii):
        s_k = size(rk)
        delta = np.sqrt(3) / 4 * s_k  # half the ring spacing
        r_lo = rk - delta
        # rings between r_prev(+delta) and r_lo, staggered
        start = r_prev + (np.sqrt(3) / 4 * size(r_prev) if k > 0 else 0.0)
        rings = []
        r = r_lo
        r_core = 0.4 * rk if (k == 0 and lattice) else 0.0
        while True:
            dr = np.sqrt(3) / 2 * size(r)
            if r - dr < max(start, r_core) + 0.5 * dr:
                break
            r -= dr
            rings.append(r)
        rings = rings[::-1]
        n_k = max(8, int(np.ceil(2 * np.pi * rk / s_k)))
        # area-preserving tangent polygon: n rho^2 tan(pi/n) = pi r^2
        rho = rk * np.sqrt(np.pi / (n_k * np.tan(np.pi / n_k)))
        mirror_in = _ring_pts(ctr, rho - delta, n_k)
        mirror_out = _ring_pts(ctr, rho + delta, n_k)
        if k == 0 and lattice:
            s_c = size(0.0)
            lim = (rings[0] if rings else rho - delta) - 0.75 * s_c
            ext = lim + s_c
            xs = np.arange(-ext, ext + s_c, s_c)
            ys = np.arange(-ext, ext + s_c, s_c * np.sqrt(3) / 2)
            X, Y = np.meshgrid(xs, ys, indexing="xy")
            X = X + (np.arange(len(ys)) % 2)[:, None] * s_c / 2
            P = np.stack([X.ravel(), Y.ravel()], 1)
            P = P[np.hypot(*P.T) < lim]
            gens.append(P + ctr)
            tags.append(np.full(len(P), k))
            fixed.append(np.zeros(len(P), bool))
        for j, r in enumerate(rings):
            n = max(6, int(np.ceil(2 * np.pi * r / size(r))))
            gens.append(_ring_pts(ctr, r, n, phase=0.5 * ((len(rings) - j) % 2)))
            tags.append(np.full(n, k))
            fixed.append(np.zeros(n, bool))
        gens.append(mirror_in)
        tags.append(np.full(n_k, k))
        fixed.append(np.ones(n_k, bool))
        if k < len(radii) - 1:
            gens.append(mirror_out)
            tags.append(np.full(n_k, k + 1))
            r_prev = rho + delta
        else:
            gens.append(mirror_out)
            tags.append(np.full(n_k, -1))  # outside: discarded
            # a second outside ring bounds the mirror cells
            gens.append(_ring_pts(ctr, rho + 3 * delta, n_k, 0.5))
            tags.append(np.full(n_k, -1))
            fixed.append(np.ones(n_k, bool))
        fixed.append(np.ones(n_k, bool))
    G = np.vstack(gens)
    T = np.concatenate(tags)
    fixed = np.concatenate(fixed)
    # radial limits of each region (between the mirror rings)
    rG = np.hypot(*(G - ctr).T)
    for _ in range(lloyd):
        vor = Voronoi(G)
        mv = np.flatnonzero(~fixed)
        new = G.copy()
        for gi in mv:
            reg = vor.regions[vor.point_region[gi]]
            if -1 in reg or len(reg) < 3:
                continue
            P = vor.vertices[reg]
            x, y = P[:, 0], P[:, 1]
            x1, y1 = np.roll(x, -1), np.roll(y, -1)
            cr = x * y1 - x1 * y
            Ar = 0.5 * cr.sum()
            new[gi] = [np.sum((x + x1) * cr) / (6 * Ar), np.sum((y + y1) * cr) / (6 * Ar)]
        # stay inside the own region: between the neighbouring mirror rings
        rn = np.hypot(*(new - ctr).T)
        lo = np.zeros(len(G))
        hi = np.full(len(G), np.inf)
        for k, rk in enumerate(radii):
            s_k = size(rk)
            d = np.sqrt(3) / 4 * s_k
            n_k = max(8, int(np.ceil(2 * np.pi * rk / s_k)))
            rho = rk * np.sqrt(np.pi / (n_k * np.tan(np.pi / n_k)))
            hi = np.where(T == k, np.minimum(hi, rho - 1.5 * d), hi)
            lo = np.where(T == k + 1, np.maximum(lo, rho + 1.5 * d), lo)
        ok = (rn > lo) & (rn < hi)
        G[mv[ok[mv]]] = new[mv[ok[mv]]]
        rG = np.hypot(*(G - ctr).T)
    return _voronoi_polymesh(G, T, merge_tol * size(radii[-1]), collapse)


def _voronoi_polymesh(G, T, merge_tol, collapse=0.0):
    """PolyMesh of the Voronoi cells of the generators G with T >= 0."""
    vor = Voronoi(G)
    V = vor.vertices
    # merge (near-)coincident vertices of cocircular generators
    tree = cKDTree(V)
    pairs = tree.query_pairs(merge_tol, output_type="ndarray")
    parent = np.arange(len(V))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if collapse > 0:
        # collapse short Voronoi edges: L_e < collapse * |p_i - p_j|
        rv = np.array(vor.ridge_vertices)
        rp = vor.ridge_points
        fin = np.all(rv >= 0, axis=1)
        rv, rp = rv[fin], rp[fin]
        Le = np.linalg.norm(V[rv[:, 0]] - V[rv[:, 1]], axis=1)
        De = np.linalg.norm(G[rp[:, 0]] - G[rp[:, 1]], axis=1)
        pairs = np.vstack([pairs.reshape(-1, 2), rv[Le < collapse * De]])
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    root = np.array([find(i) for i in range(len(V))])
    cells, keep_gen = [], []
    for gi in np.flatnonzero(T >= 0):
        reg = vor.regions[vor.point_region[gi]]
        if -1 in reg or len(reg) == 0:
            raise RuntimeError("unbounded Voronoi cell inside the domain")
        poly = [root[v] for v in reg]
        dedup = [p for i, p in enumerate(poly) if p != poly[i - 1]]
        if len(dedup) < 3:
            raise RuntimeError("edge collapse degenerated a cell; lower `collapse`")
        cells.append(dedup)
        keep_gen.append(gi)
    keep_gen = np.array(keep_gen)
    # merged vertex position: mean of the cluster
    Vm = np.zeros_like(V)
    cnt = np.bincount(root, minlength=len(V))
    np.add.at(Vm, root, V)
    Vm[cnt > 0] /= cnt[cnt > 0, None]
    return PolyMesh(Vm, cells, T[keep_gen], dual_points=G[keep_gen])
