# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Plane-section conformal geometry engine on the Wakis staggered FIT grid.

Evaluates how closed triangulated solids and axis-aligned planar sheets cut
the primal edges, primal faces, and clipped dual cells of a Cartesian grid:
planar cross sections of the triangulated surface, line crossings for edge
lengths, Green's theorem for face areas, and section-area quadrature for dual
volume fractions.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

# Plane normal w -> in-plane axes (u, v), cyclic so that (u, v, w) is right-handed
_CYCLIC = {0: (1, 2), 1: (2, 0), 2: (0, 1)}

# Relative size of the geometry shifts that remove degenerate intersections
# (grid lines lying exactly on faces or through vertices of the surface)
_SHIFT = 1e-7 * np.array([1.0, 0.7548776662, 0.5698402910])

_POOL = []


def _default_threads():
    """Default thread count: OMP_NUM_THREADS if set, else min(os.cpu_count(), 8)."""
    env = os.environ.get("OMP_NUM_THREADS", "")
    n = int(env) if env.isdigit() and int(env) > 0 else (os.cpu_count() or 1)
    return max(1, min(n, 8))


def _pool():
    """Shared worker threads (created on first use)."""
    if not _POOL:
        _POOL.append(ThreadPoolExecutor(8, thread_name_prefix="wakis-sections"))
    return _POOL[0]


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_POOL.clear)


@dataclass
class SolidMeasures:
    """
    Geometric measures of a closed solid on a Cartesian FIT grid.

    Attributes
    ----------
    edge_length : tuple of 3 ndarray
        Length [m] of each primal edge inside the solid, for components
        ``(x, y, z)`` with shapes ``(Nx - 1, Ny, Nz)``, ``(Nx, Ny - 1, Nz)``,
        and ``(Nx, Ny, Nz - 1)``.
    face_area : tuple of 3 ndarray
        Area [m^2] of each primal face inside the solid, for faces normal to
        ``(x, y, z)`` with shapes ``(Nx, Ny - 1, Nz - 1)``,
        ``(Nx - 1, Ny, Nz - 1)``, and ``(Nx - 1, Ny - 1, Nz)``.
    dual_fraction : tuple of 3 ndarray or None
        Dimensionless volume fraction [-] of the solid inside the clipped dual
        cell of each primal edge, with the same shapes as ``edge_length``, or
        ``None`` when ``dual=False``.
    normal : tuple of 3 ndarray or None
        Unit outward surface normal [-] of shape ``(..., 3)`` at primal edges
        with ``1e-9 < dual_fraction < 1 - 1e-9`` (zeros elsewhere), or
        ``None`` when ``dual=False``.
    """

    edge_length: tuple
    face_area: tuple
    dual_fraction: tuple | None
    normal: tuple | None


@dataclass
class SheetMeasures:
    """
    Geometric measures of a planar axis-aligned sheet on a Cartesian FIT grid.

    Attributes
    ----------
    axis : int
        Sheet normal axis (``0`` for x, ``1`` for y, ``2`` for z).
    index : int
        Node index of the sheet plane along ``axis``.
    edge_length : tuple of 3 ndarray
        Covered length [m] of in-plane primal edges (zeros on all other planes
        and on the normal component), with full primal-edge shapes.
    face_area : tuple of 3 ndarray
        Covered area [m^2] of primal faces in the sheet plane (zeros on all
        other planes and on the other normal components), with full primal-face
        shapes.
    outline : tuple of 2 ndarray
        Start and end points ``(p, q)`` [m], each of shape ``(S, 2)``, of the
        sheet boundary segments in the in-plane ``(u, v)`` coordinates relative
        to the grid origin.
    """

    axis: int
    index: int
    edge_length: tuple
    face_area: tuple
    outline: tuple


def _triangles(surface):
    """(T, 3, 3) vertex coordinates of a triangulated surface."""
    surface = surface.triangulate()
    faces = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    return np.asarray(surface.points, float)[faces]


def _outward(tri):
    """Orient a closed surface so that its normals point outwards."""
    signed_volume = np.einsum(
        "ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])
    ).sum()
    return tri[:, [0, 2, 1]] if signed_volume < 0 else tri


def _pairs(lo, hi):
    """Expand index ranges [lo, hi) per item into (item, index) pairs."""
    count = np.maximum(hi - lo, 0)
    item = np.repeat(np.arange(len(lo)), count)
    start = np.repeat(np.cumsum(count) - count, count)
    return item, np.arange(item.size) - start + np.repeat(lo, count)


def _plane_segments(tri, w, planes):
    """
    Oriented cross-section segments of a closed surface with planes w = const.

    The segments are oriented counter-clockwise around the section in the
    (u, v) plane (section on their left), which follows from the outward
    triangle normal n as the direction e_w x n.

    Returns
    -------
    plane : ndarray of int
        Plane index of each segment.
    p, q : ndarray, shape (S, 2)
        (u, v) start and end points.
    """
    u, v = _CYCLIC[w]
    tw = tri[:, :, w]
    # A vertex on a plane counts as above it, so that the triangles sharing
    # it agree and every cut triangle has exactly two cut edges
    lo = np.searchsorted(planes, tw.min(axis=1), side="right")
    hi = np.searchsorted(planes, tw.max(axis=1), side="right")
    t, plane = _pairs(lo, hi)
    pts = tri[t]
    d = pts[:, :, w] - planes[plane][:, None]
    ends = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        cut = (d[:, a] < 0) != (d[:, b] < 0)
        s = d[:, a] / np.where(cut, d[:, a] - d[:, b], 1.0)
        point = pts[:, a] + s[:, None] * (pts[:, b] - pts[:, a])
        ends.append((cut, point[:, [u, v]]))
    # Exactly two edges are cut in general position
    (c0, p0), (_, p1), (c2, p2) = ends
    first = np.where(c0[:, None], p0, p1)
    second = np.where(c2[:, None], p2, p1)
    normal = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
    direction = np.stack([-normal[:, v], normal[:, u]], axis=1)
    flip = np.einsum("ij,ij->i", second - first, direction) < 0
    p = np.where(flip[:, None], second, first)
    q = np.where(flip[:, None], first, second)
    return plane, p, q


def _sheet_segments(tri, w):
    """Boundary edges of a planar triangulated sheet, counter-clockwise in (u, v)."""
    u, v = _CYCLIC[w]
    pts = tri[:, :, [u, v]]
    d1, d2 = pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]
    area = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    pts = np.where((area < 0)[:, None, None], pts[:, [0, 2, 1]], pts)
    p = pts.reshape(-1, 2)
    q = pts[:, [1, 2, 0]].reshape(-1, 2)
    # Interior edges appear twice with opposite directions: keep boundary ones
    key = np.round(np.concatenate([p, q], axis=1), 12)
    fwd = [tuple(r) for r in key]
    rev = {tuple(r) for r in key[:, [2, 3, 0, 1]]}
    keep = np.array([k not in rev for k in fwd], dtype=bool)
    return np.zeros(keep.sum(), int), p[keep], q[keep]


def _line_lengths(plane, p, q, n_planes, along, nodes, lines):
    """
    Length inside the section on each segment [nodes[i], nodes[i+1]] of the
    lines through `lines` parallel to in-plane axis `along` (0: u, 1: v).

    Uses that the inside length from -inf to x is
    F(x) = sum_k s_k (x - c_k)_+ over the line crossings c_k, s_k = +1 when
    the line enters the section and -1 when it leaves.

    Returns
    -------
    ndarray, shape (n_planes, len(lines), len(nodes) - 1)
    """
    a, b = along, 1 - along
    pa, pb, qa, qb = p[:, a], p[:, b], q[:, a], q[:, b]
    # Half-open convention (an end point on a line counts as above it):
    # a polygon vertex on a line is crossed once or not at all
    lo = np.searchsorted(lines, np.minimum(pb, qb), side="right")
    hi = np.searchsorted(lines, np.maximum(pb, qb), side="right")
    seg, line = _pairs(lo, hi)
    tb = qb[seg] - pb[seg]
    c = pa[seg] + (lines[line] - pb[seg]) * (qa[seg] - pa[seg]) / tb
    # Section on the left of the segment: entering along +u if t_v < 0,
    # entering along +v if t_u > 0
    s = np.where(tb < 0, 1.0, -1.0) if along == 0 else np.where(tb > 0, 1.0, -1.0)
    m = np.searchsorted(nodes, c, side="right")
    keep = m < len(nodes)
    out = np.zeros((n_planes, len(lines), len(nodes) - 1))
    if not np.any(keep):
        return out
    # Only the crossed lines (rows) are accumulated; the others stay zero.
    # bincount adds the weights in input order, like np.add.at.
    nn = len(nodes)
    rows, inv = np.unique(
        plane[seg][keep] * len(lines) + line[keep], return_inverse=True
    )
    flat = inv.ravel() * nn + m[keep]
    A = np.bincount(flat, weights=s[keep], minlength=rows.size * nn)
    B = np.bincount(flat, weights=(s * c)[keep], minlength=rows.size * nn)
    A, B = A.reshape(rows.size, nn), B.reshape(rows.size, nn)
    F = np.cumsum(A, axis=1) * nodes - np.cumsum(B, axis=1)
    out.reshape(-1, nn - 1)[rows] = np.diff(F, axis=1)
    return out


def _section_areas(plane, p, q, n_planes, u_nodes, v_nodes, threads=None):
    """
    Section area inside each rectangle [u_i, u_i+1] x [v_j, v_j+1] of every
    plane, exact for polygonal sections (Green's theorem).

    The planes holding segments are processed in contiguous blocks (on
    threads for large grids), each restricted to the node ranges covering
    its segments (one node of margin). This gives the same values as the
    whole grid at once: every sum involves segments of one plane only, in
    their original order, lines outside the range are not crossed and
    rectangles outside it are not covered.

    Returns
    -------
    area, lu, lv : see `_section_areas_full`.
    """
    nu, nv = len(u_nodes), len(v_nodes)
    area = np.zeros((n_planes, nu - 1, nv - 1))
    lu = np.zeros((n_planes, nv, nu - 1))
    lv = np.zeros((n_planes, nu, nv - 1))
    if len(plane) == 0 or nu < 2 or nv < 2:
        return area, lu, lv
    order = np.argsort(plane, kind="stable")
    sorted_planes = plane[order]
    k_min, k_max = int(sorted_planes[0]), int(sorted_planes[-1]) + 1
    n_threads = _default_threads() if threads is None else max(1, int(threads))
    work = (k_max - k_min) * nu * nv
    n_blocks = (
        1
        if n_threads == 1 or (threads is None and work < 400_000)
        else min(4 * n_threads, k_max - k_min)
    )
    edges = np.linspace(k_min, k_max, n_blocks + 1).round().astype(int)

    def run(block):
        k0, k1 = int(edges[block]), int(edges[block + 1])
        i0, i1 = np.searchsorted(sorted_planes, [k0, k1])
        if i1 == i0:
            return
        sel = order[i0:i1]
        pl, ps, qs = plane[sel], p[sel], q[sel]
        pts = np.concatenate([ps, qs])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        # Lines at or below the minimum and above the maximum are not crossed
        iu0 = max(int(np.searchsorted(u_nodes, lo[0], side="right")) - 1, 0)
        iu1 = min(int(np.searchsorted(u_nodes, hi[0], side="right")) + 1, nu)
        iv0 = max(int(np.searchsorted(v_nodes, lo[1], side="right")) - 1, 0)
        iv1 = min(int(np.searchsorted(v_nodes, hi[1], side="right")) + 1, nv)
        if iu1 - iu0 < 2 or iv1 - iv0 < 2:
            return
        a, b, c = _section_areas_full(
            pl - k0, ps, qs, k1 - k0, u_nodes[iu0:iu1], v_nodes[iv0:iv1]
        )
        area[k0:k1, iu0 : iu1 - 1, iv0 : iv1 - 1] = a
        lu[k0:k1, iv0:iv1, iu0 : iu1 - 1] = b
        lv[k0:k1, iu0:iu1, iv0 : iv1 - 1] = c

    if n_blocks == 1:
        run(0)
    else:
        list(_pool().map(run, range(n_blocks)))
    return area, lu, lv


def _section_areas_full(plane, p, q, n_planes, u_nodes, v_nodes):
    """
    Section area inside each rectangle [u_i, u_i+1] x [v_j, v_j+1] of every
    plane, exact for polygonal sections (Green's theorem), on the whole grid.

    The boundary of (section ∩ rectangle) consists of the section contour
    inside the rectangle and the parts of the rectangle sides inside the
    section. In coordinates local to the rectangle corner (u_i, v_j) the
    sides v = 0 and u = 0 do not contribute to 1/2 ∮ (u dv - v du), the side
    u = du contributes du times its inside length and the side v = dv
    contributes dv times its inside length.

    Returns
    -------
    area : ndarray, shape (n_planes, len(u_nodes) - 1, len(v_nodes) - 1)
    lu : ndarray, shape (n_planes, len(v_nodes), len(u_nodes) - 1)
        Inside lengths of the u-directed rectangle sides.
    lv : ndarray, shape (n_planes, len(u_nodes), len(v_nodes) - 1)
        Inside lengths of the v-directed rectangle sides.
    """
    lu = _line_lengths(plane, p, q, n_planes, 0, u_nodes, v_nodes)
    lv = _line_lengths(plane, p, q, n_planes, 1, v_nodes, u_nodes)

    # Split the contour segments where they cross grid lines
    seg_ids, params = (
        [np.arange(len(p))] * 2,
        [np.zeros(len(p)), np.ones(len(p))],
    )
    for k, nodes in ((0, u_nodes), (1, v_nodes)):
        lo = np.searchsorted(nodes, np.minimum(p[:, k], q[:, k]), side="right")
        hi = np.searchsorted(nodes, np.maximum(p[:, k], q[:, k]), side="left")
        seg, node = _pairs(lo, hi)
        seg_ids.append(seg)
        params.append((nodes[node] - p[seg, k]) / (q[seg, k] - p[seg, k]))
    seg = np.concatenate(seg_ids)
    t = np.concatenate(params)
    order = np.lexsort((t, seg))
    seg, t = seg[order], t[order]
    same = seg[1:] == seg[:-1]
    s, t0, t1 = seg[:-1][same], t[:-1][same], t[1:][same]
    d = q[s] - p[s]
    a = p[s] + t0[:, None] * d
    b = p[s] + t1[:, None] * d
    mid = 0.5 * (a + b)
    i = np.searchsorted(u_nodes, mid[:, 0], side="right") - 1
    j = np.searchsorted(v_nodes, mid[:, 1], side="right") - 1
    ok = (i >= 0) & (i < len(u_nodes) - 1) & (j >= 0) & (j < len(v_nodes) - 1)
    i, j, a, b = i[ok], j[ok], a[ok], b[ok]
    corner = np.stack([u_nodes[i], v_nodes[j]], axis=1)
    a, b = a - corner, b - corner
    contour = np.zeros((n_planes, len(u_nodes) - 1, len(v_nodes) - 1))
    np.add.at(
        contour,
        (plane[s][ok], i, j),
        a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1],
    )
    du, dv = np.diff(u_nodes), np.diff(v_nodes)
    area = 0.5 * (
        contour
        + du[None, :, None] * lv[:, 1:, :]
        + dv[None, None, :] * np.swapaxes(lu[:, 1:, :], 1, 2)
    )
    return area, lu, lv


def _to_xyz(array, w):
    """Reorder an array indexed (w, u, v) to (x, y, z)."""
    u, v = _CYCLIC[w]
    return np.transpose(array, np.argsort([w, u, v]))


def _pec_metrics(tri, nodes, closed=True, plane=None, threads=None):
    """
    Inside lengths of all primal edges and inside areas of all primal faces.

    Parameters
    ----------
    tri : ndarray (T, 3, 3)
        Triangles, already shifted to grid-local coordinates.
    nodes : list of 3 ndarrays
        Node coordinates (local) along x, y, z.
    closed : bool
        True for a closed solid, False for a planar sheet.
    plane : tuple (w, k), optional
        For sheets: normal axis and node index of the sheet plane.
    threads : int, optional
        Number of worker threads for `_section_areas`.

    Returns
    -------
    lengths, areas : list of 3 ndarrays
        lengths[c] has shape (cells along c, nodes, nodes) in (x, y, z) order;
        areas[w] has shape (nodes along w, cells, cells).
    """
    shape_n = [len(n) for n in nodes]
    lengths = [None] * 3
    areas = [None] * 3
    for w in range(3):
        u, v = _CYCLIC[w]
        if closed:
            seg = _plane_segments(tri, w, nodes[w])
            n_planes = shape_n[w]
        else:
            if w != plane[0]:
                continue
            seg = _sheet_segments(tri, w)
            n_planes = 1
        area, lu, lv = _section_areas(
            *seg, n_planes, nodes[u], nodes[v], threads=threads
        )
        if not closed:
            full = np.zeros((shape_n[w], *area.shape[1:]))
            full[plane[1]] = area[0]
            area = full
            pad = np.zeros((shape_n[w], *lu.shape[1:]))
            pad[plane[1]] = lu[0]
            lu = pad
            pad = np.zeros((shape_n[w], *lv.shape[1:]))
            pad[plane[1]] = lv[0]
            lv = pad
        areas[w] = _to_xyz(area, w)
        # lu: (w, v-node, u-cell) -> edges along u; lv: (w, u-node, v-cell)
        edges_u = _to_xyz(np.swapaxes(lu, 1, 2), w)
        edges_v = _to_xyz(lv, w)
        if lengths[u] is None:
            lengths[u] = edges_u
        if lengths[v] is None:
            lengths[v] = edges_v
    # A sheet only covers the edges and faces of its own plane
    for c in range(3):
        if lengths[c] is None:
            lengths[c] = np.zeros([shape_n[a] - (a == c) for a in range(3)])
        if areas[c] is None:
            areas[c] = np.zeros([shape_n[a] - (a != c) for a in range(3)])
    return lengths, areas


def _axis_array(values, axis, ndim):
    """Reshape a 1D array to broadcast along `axis`."""
    shape = [1] * ndim
    shape[axis] = -1
    return np.reshape(values, shape)


def _dual_bounds(nodes):
    """Clipped bounds of the dual intervals on each node."""
    mids = 0.5 * (nodes[1:] + nodes[:-1])
    return np.r_[nodes[0], mids, nodes[-1]]


def _sub_samples(bounds, n_sub):
    """Midpoint samples splitting each interval of `bounds` into n_sub parts."""
    frac = (np.arange(n_sub) + 0.5) / n_sub
    return (bounds[:-1, None] + np.diff(bounds)[:, None] * frac).ravel()


def _dual_fractions(tri, nodes, n_sub, threads=None):
    """
    Volume fraction of the solid in the clipped dual cell of every primal edge.

    The dual cell of an edge along c spans the edge and the clipped dual
    intervals on both transverse axes. Its volume is integrated exactly in x
    and y (section areas in planes z = const) and with the n_sub-point
    midpoint rule along z, which is exact for geometry extruded along z.

    Returns
    -------
    list of 3 ndarrays
        Arrays of shape (cells along c, nodes transversally) in (x, y, z)
        order.
    """
    dual = [_dual_bounds(n) for n in nodes]
    out = []
    sections = {}  # the x and y edges share the sample planes
    for c in range(3):
        u_nodes = nodes[0] if c == 0 else dual[0]
        v_nodes = nodes[1] if c == 1 else dual[1]
        bounds = nodes[2] if c == 2 else dual[2]
        samples = _sub_samples(bounds, n_sub)
        if c == 2 or "dual" not in sections:
            seg = _plane_segments(tri, 2, samples)
            if c != 2:
                sections["dual"] = seg
        else:
            seg = sections["dual"]
        area, _, _ = _section_areas(
            *seg, len(samples), u_nodes, v_nodes, threads=threads
        )
        area = area.reshape(len(bounds) - 1, n_sub, *area.shape[1:]).mean(axis=1)
        volume = area / (np.diff(u_nodes)[:, None] * np.diff(v_nodes)[None, :])
        out.append(_to_xyz(volume, 2))
    return out


def _surface_normals(surface, points):
    """Unit normal of the closest surface triangle to each point."""
    surface = surface.triangulate().compute_normals(
        cell_normals=True, point_normals=False, auto_orient_normals=True
    )
    if len(points) == 0:
        return np.zeros((0, 3))
    cells = np.atleast_1d(surface.find_closest_cell(points))
    return np.asarray(surface.cell_data["Normals"])[cells]


def _metrics(tri, nodes, shift, reduce, plane=None, threads=None):
    """
    Inside lengths and areas of a solid (plane None) or a sheet, reduced
    over two opposite shifts of the geometry: `np.maximum` counts edges
    and faces lying on the surface as inside, `np.minimum` as outside.
    """
    out_l, out_a = None, None
    for sign in (1.0, -1.0):
        moved = tri + sign * shift
        if plane is not None:
            moved[:, :, plane[0]] = nodes[plane[0]][plane[1]]
        lengths, areas = _pec_metrics(
            moved, nodes, plane is None, plane, threads=threads
        )
        if out_l is None:
            out_l, out_a = lengths, areas
        else:
            out_l = list(map(reduce, out_l, lengths))
            out_a = list(map(reduce, out_a, areas))
    return out_l, out_a


def _sheet_plane(tri, nodes):
    """Normal axis and node index of a planar axis-aligned sheet."""
    extent = np.ptp(tri.reshape(-1, 3), axis=0)
    flat = np.flatnonzero(extent <= 1e-9 * max(extent.max(), 1e-30))
    if flat.size != 1:
        raise ValueError(
            "Sheet must be planar and normal to x, y or z "
            "(oblique sheets are not supported)."
        )
    w = int(flat[0])
    position = tri[0, 0, w]
    axis_nodes = nodes[w]
    k = int(np.argmin(np.abs(axis_nodes - position)))
    cell = np.min(np.diff(axis_nodes))
    if abs(axis_nodes[k] - position) > 1e-6 * cell:
        raise ValueError(
            f"Sheet lies between the grid planes along "
            f"{'xyz'[w]}; place a mesh node on it."
        )
    return w, k


def _interface_normals(surface, axes, fractions):
    """Surface normals at the primal edges cut by the interface of a solid."""
    centres = [0.5 * (a[:-1] + a[1:]) for a in axes]
    indices, points = [], []
    for c in range(3):
        f = fractions[c]
        cut = (f > 1e-9) & (f < 1 - 1e-9)
        idx = np.nonzero(cut)
        coords = [centres[a] if a == c else axes[a] for a in range(3)]
        indices.append(idx)
        points.append(np.stack([coords[a][idx[a]] for a in range(3)], axis=1))
    # One closest-cell search for the three components
    found = _surface_normals(surface, np.concatenate(points))
    normals, start = [], 0
    for f, idx, pts in zip(fractions, indices, points):
        n = np.zeros((*f.shape, 3))
        n[idx] = found[start : start + len(pts)]
        start += len(pts)
        normals.append(n)
    return tuple(normals)


def _validate_axes(x, y, z):
    """Convert and validate strictly increasing 1D node coordinate arrays."""
    axes = [np.asarray(a, dtype=float) for a in (x, y, z)]
    for name, a in zip("xyz", axes):
        if a.ndim != 1 or a.size < 2 or np.any(np.diff(a) <= 0):
            raise ValueError(
                f"Axis '{name}' must be a strictly increasing 1D array with "
                "at least 2 nodes."
            )
    return axes


def solid_measures(
    surface,
    x,
    y,
    z,
    *,
    on_surface="inside",
    dual=True,
    n_sub=4,
    threads=None,
) -> SolidMeasures:
    """
    Measure how a closed triangulated solid cuts the primal and dual FIT grid.

    Parameters
    ----------
    surface : pyvista.PolyData
        Closed triangulated surface [m] with any facet orientation.
    x, y, z : array_like
        Strictly increasing 1D primal node coordinates [m] along x, y, and z,
        with lengths ``Nx``, ``Ny``, and ``Nz``.
    on_surface : {'inside', 'outside'}, optional
        How edges and faces lying on the surface are classified: ``'inside'``
        (default) counts them as inside the solid via the elementwise maximum
        of two opposite degeneracy shifts; ``'outside'`` counts them as
        outside via the elementwise minimum.
    dual : bool, optional
        If ``True`` (default), also compute clipped dual-cell volume fractions
        and unit outward surface normals at cut edges. If ``False``, set
        ``dual_fraction`` and ``normal`` to ``None``. The fractions use one
        degeneracy shift for both ``on_surface`` modes, so a surface lying
        exactly on a dual-cell boundary is not classified by ``on_surface``.
    n_sub : int, optional
        Number of midpoint quadrature subsamples per z interval for the dual
        volume fractions. Default is 4.
    threads : int or None, optional
        Worker thread count for plane-section evaluation. If ``None``
        (default), uses ``OMP_NUM_THREADS`` when set, else
        ``min(os.cpu_count(), 8)``.

    Returns
    -------
    SolidMeasures
        Primal edge lengths [m], primal face areas [m^2], and optionally
        clipped dual-cell volume fractions [-] and outward unit normals [-].

    Raises
    ------
    ValueError
        If ``on_surface`` is not ``'inside'`` or ``'outside'``, if any coordinate
        axis is not strictly increasing with at least 2 nodes, or if
        ``n_sub < 1``.
    """
    if on_surface == "inside":
        reduce = np.maximum
    elif on_surface == "outside":
        reduce = np.minimum
    else:
        raise ValueError(
            f"on_surface must be 'inside' or 'outside', got {on_surface!r}."
        )
    if int(n_sub) < 1:
        raise ValueError(f"n_sub must be at least 1, got {n_sub!r}.")
    n_sub = int(n_sub)

    axes = _validate_axes(x, y, z)
    origin = np.array([axes[0][0], axes[1][0], axes[2][0]])
    nodes = [a - origin[i] for i, a in enumerate(axes)]
    cell = min(np.min(np.diff(n)) for n in nodes)
    shift = _SHIFT * cell

    tri = _outward(_triangles(surface)) - origin
    lengths, areas = _metrics(tri, nodes, shift, reduce, plane=None, threads=threads)
    if not dual:
        return SolidMeasures(
            edge_length=tuple(lengths),
            face_area=tuple(areas),
            dual_fraction=None,
            normal=None,
        )

    fractions = tuple(_dual_fractions(tri + shift, nodes, n_sub, threads=threads))
    normals = _interface_normals(surface, axes, fractions)
    return SolidMeasures(
        edge_length=tuple(lengths),
        face_area=tuple(areas),
        dual_fraction=fractions,
        normal=normals,
    )


def sheet_measures(surface, x, y, z, *, threads=None) -> SheetMeasures:
    """
    Measure how a planar axis-aligned sheet covers edges and faces of a grid plane.

    Parameters
    ----------
    surface : pyvista.PolyData
        Planar triangulated surface [m] normal to x, y, or z and lying on a
        primal node plane.
    x, y, z : array_like
        Strictly increasing 1D primal node coordinates [m] along x, y, and z,
        with lengths ``Nx``, ``Ny``, and ``Nz``.
    threads : int or None, optional
        Worker thread count for plane-section evaluation. If ``None``
        (default), uses ``OMP_NUM_THREADS`` when set, else
        ``min(os.cpu_count(), 8)``.

    Returns
    -------
    SheetMeasures
        Sheet normal axis, node index, covered primal edge lengths [m],
        covered primal face areas [m^2], and boundary outline ``(p, q)`` [m].

    Raises
    ------
    ValueError
        If the sheet is not planar and normal to x, y, or z, if it lies
        between grid planes, or if any coordinate axis is invalid.
    """
    axes = _validate_axes(x, y, z)
    origin = np.array([axes[0][0], axes[1][0], axes[2][0]])
    nodes = [a - origin[i] for i, a in enumerate(axes)]
    cell = min(np.min(np.diff(n)) for n in nodes)
    shift = _SHIFT * cell

    tri = _triangles(surface) - origin
    plane = _sheet_plane(tri, nodes)
    # The outline is computed relative to the grid origin; return it in metres
    uv = list(_CYCLIC[plane[0]])
    outline = tuple(seg + origin[uv] for seg in _sheet_segments(tri, plane[0])[1:])
    lengths, areas = _metrics(
        tri, nodes, shift, np.maximum, plane=plane, threads=threads
    )
    return SheetMeasures(
        axis=plane[0],
        index=plane[1],
        edge_length=tuple(lengths),
        face_area=tuple(areas),
        outline=outline,
    )
