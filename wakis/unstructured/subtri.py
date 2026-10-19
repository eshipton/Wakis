# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Quadtree squares + conforming Delaunay triangle band ("subtri") for 2D FIT.

The bulk and (optionally) the deep interior of circular scatterers are meshed
with a 2:1-balanced quadtree of axis-aligned squares (`QuadMesh`, coarse size
``H = 2**levels * h`` down to finest size ``h``). A narrow band of finest-level
squares around each circular interface (and across narrow inter-circle gaps) is
removed and replaced by a conforming Delaunay triangle patch whose outer and
inner staircase boundaries share the exact finest-level square vertices (no
hanging nodes touch the triangle patch).

Discrete formulation (`SubTriFIT2D` / `PolygonFIT2D` with ``prolongation=P``):

* Unknowns ``u`` live on the master (free) nodes; the full nodal field is
  ``P u``, where each hanging-node row of ``P`` holds the linear interpolation
  weights on its two master nodes and each hanging-node column is zero.
* Restricted gradient ``K_grad = G P`` (stored as ``solver.G``), diagonal
  orthogonal-dual edge Hodge ``S = diag(sum_C w_C h_{e,C} / L_e)``, and lumped
  diagonal node mass ``D = P^T m`` (stored as ``solver.mass``).
* The lossless leapfrog update reads::

      D (u^{n+1} - 2 u^n + u^{n-1}) = -dt^2 K_grad^T S K_grad u^n

  with stiffness ``K = K_grad^T S K_grad``. The evolution operator
  ``A = D^{-1} K`` is self-adjoint in the ``D`` inner product, and the staggered
  quadratic form::

      E = 1/2 u^T D u + 1/2 v^{n-1/2,T} S v^{n+1/2}

  is an exact invariant of the leapfrog step, positive definite if and only if
  ``dt^2 lambda_max(A) < 4``.
"""

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay, cKDTree

from ..subgrid.mesh import QuadMesh
from .polymesh import PolyMesh
from .polymeshers import _graded_ring, _ring_pts
from .polysolver2d import PolygonFIT2D

__all__ = ["subtri_circles_mesh", "SubTriFIT2D"]


def _box_circle_dist(x0, x1, y0, y1, ccx, ccy):
    """Minimum and maximum distance from (ccx, ccy) to boxes [x0, x1] x [y0, y1]."""
    dx_min = np.maximum(np.maximum(x0 - ccx, ccx - x1), 0.0)
    dy_min = np.maximum(np.maximum(y0 - ccy, ccy - y1), 0.0)
    r_min = np.hypot(dx_min, dy_min)
    dx_max = np.maximum(np.abs(x0 - ccx), np.abs(x1 - ccx))
    dy_max = np.maximum(np.abs(y0 - ccy), np.abs(y1 - ccy))
    r_max = np.hypot(dx_max, dy_max)
    return r_min, r_max


def subtri_circles_mesh(
    circles,
    h,
    levels,
    half_width,
    band=2.0,
    grade=2.0,
    pml_cells=0,
    gap_cells=None,
    h_min=None,
    growth=1.2,
    min_spacing=0.75,
    smooth=4,
    inner_fine=True,
):
    """
    Quadtree square mesh with a conforming Delaunay triangle band around circles.

    Parameters
    ----------
    circles : sequence of (cx, cy, radius)
        Circular material interfaces [m] (non-overlapping).
    h : float
        Finest square size and interface triangle edge length [m].
    levels : int
        Number of quadtree refinement levels ``L >= 0``; coarse square size is
        ``H = 2**L * h`` [m].
    half_width : float
        Physical domain half-width [m], rounded up to a multiple of ``H``.
    band : float
        Half-width of the removed finest-square annulus around each circle, in
        units of ``h`` (default 2.0).
    grade : float
        Quadtree grading margin in units of the current cell size ``s``
        (default 2.0).
    pml_cells : int
        Number of coarse (size ``H``) square layers added outside ``half_width``
        for a PML (never refined).
    gap_cells : float, optional
        If given, refine narrow gaps between circles so the local size is at
        most ``max(h_min, (d_1 + d_2) / gap_cells)``.
    h_min : float, optional
        Minimum triangle size in refined gaps [m] (default ``h / 8``).
    growth : float
        Geometric growth factor per ``h`` (kept for API symmetry with
        `hybrid_circles_mesh`).
    min_spacing : float
        Minimum distance between accepted triangle points in units of the local
        target size (default 0.75).
    smooth : int
        Number of Laplacian smoothing sweeps on free triangle vertices
        (default 4).
    inner_fine : bool
        If True (default), refine all quadtree cells intersecting any disk to
        the finest level ``h``.

    Returns
    -------
    mesh : PolyMesh
        Assembled polygonal mesh (counter-clockwise squares and triangles).
        Tags: ``i`` inside circle ``i``, ``len(circles)`` in the background.
    P : scipy.sparse.csr_matrix (n_nodes, n_nodes)
        Prolongation matrix mapping master node values to all mesh nodes
        (hanging-node columns are all zero; triangle-node block is identity).
    info : dict
        Summary dictionary with keys ``n_squares``, ``n_squares_per_level``,
        ``n_triangles``, ``n_hanging``, ``min_dual_edge``, ``H``,
        ``half_width``, ``half_width_pml``, ``h``, and ``levels``.
    """
    circles = np.atleast_2d(np.asarray(circles, float))
    C, R = circles[:, :2], circles[:, 2]
    h = float(h)
    L = int(levels)
    if L < 0:
        raise ValueError("levels must be >= 0")
    H = (2**L) * h
    M_in = int(np.ceil(half_width / H - 1e-9))
    pml_cells = int(pml_cells)
    M = M_in + pml_cells
    L_in = M_in * H
    L_pml = M * H
    xc = H * np.arange(-M, M + 1)

    band_m = float(band) * h
    grade = float(grade)
    gap = gap_cells is not None and len(R) > 1
    h_min = h / 8.0 if h_min is None else float(h_min)

    def iface_dist(P):
        r = np.linalg.norm(P[:, None, :] - C[None], axis=2)
        d = np.abs(r - R[None])
        return d.min(1), d.argmin(1), r

    def gap_size(P):
        r = np.linalg.norm(P[:, None, :] - C[None], axis=2)
        d = np.sort(np.abs(r - R[None]), axis=1)
        return np.maximum(h_min, (d[:, 0] + d[:, 1]) / gap_cells)

    def size_at(P):
        P = np.atleast_2d(P)
        out = np.full(len(P), h)
        return np.minimum(out, gap_size(P)) if gap else out

    def refine(level, x0, x1, y0, y1):
        s = x1 - x0
        split = np.zeros(np.shape(x0), dtype=bool)
        for (ccx, ccy), rr in zip(C, R):
            r_min, r_max = _box_circle_dist(x0, x1, y0, y1, ccx, ccy)
            d_iface = np.maximum(np.maximum(r_min - rr, rr - r_max), 0.0)
            split |= d_iface < band_m + grade * s
            if inner_fine:
                split |= r_min <= rr
        if gap:
            cen = np.stack([0.5 * (x0 + x1), 0.5 * (y0 + y1)], axis=-1)
            split |= gap_size(cen) < s
        if pml_cells > 0 and np.any(split):
            in_pml = (np.maximum(np.abs(x0), np.abs(x1)) > L_in + 1e-9 * h) | (
                np.maximum(np.abs(y0), np.abs(y1)) > L_in + 1e-9 * h
            )
            if np.any(split & in_pml):
                raise ValueError(
                    "quadtree refinement reaches the PML layer; increase `half_width`"
                )
        return split

    qmesh = QuadMesh(xc, xc, ratio=2, max_level=L, refine=refine, balance=True)
    c = qmesh.cells
    lvl = c["level"]
    x0, x1, y0, y1 = c["x0"], c["x1"], c["y0"], c["y1"]

    if pml_cells > 0:
        in_pml = (np.maximum(np.abs(x0), np.abs(x1)) > L_in + 1e-9 * h) | (
            np.maximum(np.abs(y0), np.abs(y1)) > L_in + 1e-9 * h
        )
        if np.any(in_pml & (lvl > 0)):
            raise ValueError(
                "quadtree balancing reached the PML layer; increase `half_width`"
            )

    # Identify hole region: finest leaves intersecting (R - band*h, R + band*h)
    # or having gap_size(centre) < h.
    in_hole_region = np.zeros(qmesh.n_cells, dtype=bool)
    crossed = np.zeros(qmesh.n_cells, dtype=bool)
    for (ccx, ccy), rr in zip(C, R):
        r_min, r_max = _box_circle_dist(x0, x1, y0, y1, ccx, ccy)
        in_hole_region |= (r_min < rr + band_m - 1e-9 * h) & (
            r_max > rr - band_m + 1e-9 * h
        )
        crossed |= (r_min <= rr + 0.25 * h) & (r_max >= rr - 0.25 * h)
    if gap:
        cen_all = np.stack([0.5 * (x0 + x1), 0.5 * (y0 + y1)], axis=1)
        in_hole_region |= gap_size(cen_all) < h - 1e-9 * h

    if np.any((lvl < L) & in_hole_region):
        raise RuntimeError(
            "non-finest quadtree leaf intersects the triangle hole; raise `grade`"
        )

    hole = (lvl == L) & in_hole_region
    while True:
        kept_nodes = np.unique(qmesh.cell_nodes[~hole])
        pimple = hole & ~crossed & np.all(np.isin(qmesh.cell_nodes, kept_nodes), axis=1)
        if not np.any(pimple):
            break
        hole[pimple] = False
    kept = ~hole
    if not np.any(hole) or not np.any(kept):
        raise RuntimeError("degenerate hole/kept partition in subtri_circles_mesh")

    hole_nodes = np.unique(qmesh.cell_nodes[hole])
    kept_nodes = np.unique(qmesh.cell_nodes[kept])
    hole_bnd = np.intersect1d(hole_nodes, kept_nodes)

    # Assert no non-finest leaf touches the hole closure and no hole-boundary
    # node is a hanging node.
    kept_touches_hole = np.isin(qmesh.cell_nodes[kept], hole_nodes).any(axis=1)
    if np.any(lvl[kept][kept_touches_hole] < L) or np.any(qmesh.hanging[hole_nodes]):
        raise RuntimeError(
            "non-finest leaf or hanging node on the hole boundary; raise `grade`"
        )

    # Integer lattice lookup on the finest lattice for hole membership
    hole_grid = np.zeros((qmesh.NI, qmesh.NJ), dtype=bool)
    hole_grid[c["i"][hole], c["j"][hole]] = True
    x_min, y_min = float(qmesh.xl[0]), float(qmesh.yl[0])

    def in_hole(P):
        P = np.atleast_2d(P)
        ix = np.floor((P[:, 0] - x_min) / h).astype(np.int64)
        iy = np.floor((P[:, 1] - y_min) / h).astype(np.int64)
        valid = (ix >= 0) & (ix < qmesh.NI) & (iy >= 0) & (iy < qmesh.NJ)
        out = np.zeros(len(P), dtype=bool)
        out[valid] = hole_grid[ix[valid], iy[valid]]
        return out

    # Kept squares as counter-clockwise 4-node polygons: [BL, BR, TR, TL]
    sq_cells = qmesh.cell_nodes[kept][:, [0, 1, 3, 2]]
    sq_centers = np.stack(
        [0.5 * (x0[kept] + x1[kept]), 0.5 * (y0[kept] + y1[kept])], axis=1
    )
    sq_size = x1[kept] - x0[kept]
    sq_tree = cKDTree(sq_centers)

    # Triangle patch points: fixed hole-boundary nodes + fixed interface rings
    bnd_pts = qmesh.nodes[hole_bnd]
    iface = []
    for (ccx, ccy), rr in zip(C, R):
        if gap:
            P_ring = _graded_ring((ccx, ccy), rr, size_at, h_min)
            rx, ry = (P_ring - [ccx, ccy]).T
            area = 0.5 * np.sum(rx * np.roll(ry, -1) - np.roll(rx, -1) * ry)
            iface.append(
                [ccx, ccy] + (P_ring - [ccx, ccy]) * np.sqrt(np.pi * rr**2 / area)
            )
        else:
            n = max(8, int(np.ceil(2 * np.pi * rr / h)))
            rho = rr * np.sqrt(2 * np.pi / (n * np.sin(2 * np.pi / n)))
            iface.append(_ring_pts((ccx, ccy), rho, n))
        if in_hole(np.array([[ccx, ccy]]))[0]:
            iface.append(np.array([[ccx, ccy]]))
    iface = np.vstack(iface)
    accepted = np.vstack([bnd_pts, iface])
    n_bnd = len(bnd_pts)
    n_fixed = len(accepted)
    tree = cKDTree(accepted)

    new_pts = []
    max_d = (band + 2.0) * h if not gap else 2.0 * L_in
    for ci, ((ccx, ccy), rr) in enumerate(zip(C, R)):
        for sgn in (+1, -1):
            r, k = rr, 0
            while True:
                if gap:
                    r_probe = max(r + sgn * 0.5 * h, 0.5 * h_min)
                    ring = _ring_pts((ccx, ccy), r_probe, 512)
                    step = np.sqrt(3) / 2 * float(size_at(ring).min())
                else:
                    step = np.sqrt(3) / 2 * h
                r = r + sgn * step
                k += 1
                if r <= 0.5 * h:
                    break
                if abs(r - rr) > max_d:
                    break
                if gap:
                    P = _graded_ring((ccx, ccy), r, size_at, h_min, phase=0.5 * (k % 2))
                else:
                    n = max(6, int(np.ceil(2 * np.pi * r / h)))
                    P = _ring_pts((ccx, ccy), r, n, phase=0.5 * (k % 2))
                _, own, _ = iface_dist(P)
                P = P[(own == ci) & in_hole(P)]
                if len(P) == 0:
                    continue
                hP = size_at(P)
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

    free_pts = np.vstack(new_pts) if new_pts else np.zeros((0, 2))
    pts = np.vstack([accepted, free_pts])

    def triangulate(pts_arr):
        tri_all = Delaunay(pts_arr).simplices
        cen = pts_arr[tri_all].mean(axis=1)
        return tri_all[in_hole(cen)]

    tri = triangulate(pts)
    for _ in range(smooth):
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
        dI_old = iface_dist(pts[mv])
        dI_new = iface_dist(new)
        same_side = np.all(
            np.sign(dI_old[2] - R[None]) == np.sign(dI_new[2] - R[None]), axis=1
        )
        dd, jj = sq_tree.query(new)
        ok = same_side & in_hole(new) & (dd > 0.72 * sq_size[jj] + 0.2 * size_at(new))
        pts[mv[ok]] = new[ok]
        tri = triangulate(pts)

    # Assemble nodes = QuadMesh nodes + new triangle points (iface + free_pts)
    n_q = qmesh.n_nodes
    new_tri_pts = pts[n_bnd:]
    n_tri_new = len(new_tri_pts)
    nodes_raw = np.vstack([qmesh.nodes, new_tri_pts])

    tri_map = np.empty(len(pts), dtype=np.int64)
    tri_map[:n_bnd] = hole_bnd
    tri_map[n_bnd:] = n_q + np.arange(n_tri_new, dtype=np.int64)
    tri_global = tri_map[tri]

    cells_raw = [list(sq) for sq in sq_cells] + [list(t) for t in tri_global]
    allc = np.vstack([sq_centers, nodes_raw[tri_global].mean(axis=1)])
    r_c = np.linalg.norm(allc[:, None, :] - C[None], axis=2)
    inside = r_c < R[None]
    tags = np.where(inside.any(axis=1), inside.argmax(axis=1), len(R))

    # Build full prolongation matrix on nodes_raw, then restrict to nodes kept
    # by PolyMesh (which strips unused nodes inside the hole).
    used = np.zeros(len(nodes_raw), dtype=bool)
    used[sq_cells.ravel()] = True
    used[tri_global.ravel()] = True

    coo = qmesh.P.tocoo()
    P_q = sp.csr_matrix((coo.data, (coo.row, qmesh.free[coo.col])), shape=(n_q, n_q))
    P_raw = sp.block_diag([P_q, sp.eye(n_tri_new, format="csr")], format="csr")
    P = P_raw[used][:, used].tocsr()

    mesh = PolyMesh(nodes_raw, cells_raw, tags)

    # Validation checks
    area_err = abs(mesh.cell_area.sum() / (2.0 * L_pml) ** 2 - 1.0)
    if area_err > 1e-9:
        raise RuntimeError(
            f"subtri mesh does not tile the domain (relative area error {area_err:.2e})"
        )

    hanging_mesh = np.asarray((P != 0).sum(axis=0)).ravel() == 0
    n_hanging = int(hanging_mesh.sum())

    # Check single-cell edges: every non-outer single-cell edge must be a
    # quadtree coarse/fine interface edge.
    single = np.flatnonzero(mesh.edge_cells[:, 1] == -1)
    e_pts = mesh.nodes[mesh.edges[single]]
    tol = 1e-9 * h
    on_outer = (
        (np.abs(e_pts[:, 0, 0] - e_pts[:, 1, 0]) < tol)
        & (np.abs(np.abs(e_pts[:, 0, 0]) - L_pml) < tol)
    ) | (
        (np.abs(e_pts[:, 0, 1] - e_pts[:, 1, 1]) < tol)
        & (np.abs(np.abs(e_pts[:, 0, 1]) - L_pml) < tol)
    )
    int_single = single[~on_outer]
    if int_single.size > 0:
        if not np.array_equal(np.sort(int_single), np.sort(mesh.interface_edges)):
            raise RuntimeError(
                "interior single-cell edge is not on a coarse/fine quadtree side"
            )
        if np.any(mesh.cell_size[mesh.edge_cells[int_single, 0]] != 4):
            raise RuntimeError("triangle edge lies on an unmatched interior boundary")

    # Conforming edges must be orthogonal
    conforming = mesh.edge_cells[:, 1] >= 0
    if np.any(conforming) and mesh.orthogonality_defect[conforming].max() >= 1e-6:
        raise RuntimeError(
            f"orthogonality defect {mesh.orthogonality_defect[conforming].max():.2e} >= 1e-6"
        )

    dual_len = np.bincount(mesh.corner_edge, mesh.corner_h, minlength=mesh.n_edges)
    min_dual = float(dual_len.min())
    if min_dual <= 0:
        raise RuntimeError(f"non-positive dual-edge length: min = {min_dual:.3e}")
    if min_dual < 0.05 * h:
        warnings.warn(
            f"small minimum dual-edge length ({min_dual / h:.3f} h < 0.05 h)",
            RuntimeWarning,
            stacklevel=2,
        )

    n_sq_level = np.bincount(lvl[kept], minlength=L + 1).tolist()
    info = dict(
        n_squares=n_sq_level,
        n_squares_total=int(kept.sum()),
        n_triangles=int(len(tri)),
        n_hanging=n_hanging,
        min_dual_edge=min_dual,
        H=float(H),
        half_width=float(L_in),
        half_width_pml=float(L_pml),
        h=float(h),
        levels=int(L),
    )
    return mesh, P, info


class SubTriFIT2D(PolygonFIT2D):
    """
    2D FIT solver on a quadtree-square + conforming-triangle mesh (`subtri_circles_mesh`).

    Thin wrapper around `PolygonFIT2D` with ``prolongation=P``,
    ``hodge="orthogonal"``, and ``bc="pec"``.
    """

    def __init__(
        self,
        mesh,
        P,
        materials=None,
        polarization="TM",
        pml=None,
        lts=None,
        cfl=0.95,
        **kw,
    ):
        kw.setdefault("hodge", "orthogonal")
        kw.setdefault("bc", "pec")
        super().__init__(
            mesh,
            materials=materials,
            polarization=polarization,
            pml=pml,
            lts=lts,
            cfl=cfl,
            prolongation=P,
            **kw,
        )
