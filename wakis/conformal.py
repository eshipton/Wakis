# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Conformal FIT: partially filled cells at curved boundaries.

Three opt-in techniques modify only the metric and material coefficients of
the FIT update, so the time step stays linear in the fields:

* conformal PEC (Dey-Mittra / PBA-type): a primary edge enters Faraday's
  circulation with its length outside the conductor, and the circulation is
  divided by the face area outside the conductor;
* conformal dielectrics: per E edge an anisotropic (Kottke) average of the
  permittivity over the dual cell, diagonal part only;
* thin PEC sheets: zero-thickness planar metal that covers edges and faces of
  one grid plane partially.

The geometry is evaluated exactly for triangulated surfaces: planar cross
sections of the surface, line crossings for edge lengths and Green's theorem
for face areas. See "Conformal FIT" in the physics guide.
"""

import numpy as np
import pyvista as pv
from scipy.constants import epsilon_0 as eps_0
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, eigsh

# Plane normal w -> in-plane axes (u, v), cyclic so that (u, v, w) is right-handed
_CYCLIC = {0: (1, 2), 1: (2, 0), 2: (0, 1)}

# Faces with an open area below MERGE_THRESHOLD * A_stab are merged
MERGE_THRESHOLD = 0.5
# Faces normal to x and y with scheme='tete' are not redistributed: a merged
# face couples two (x, y) columns and the bunch self-field is then no longer
# an exact discrete travelling wave. Their small-face (transverse) stability
# is restored by `wakis.tete.stabilize_transverse` instead.
TETE_XY_THRESHOLD = 0.0

# Relative size of the geometry shifts that remove degenerate intersections
# (grid lines lying exactly on faces or through vertices of the surface)
_SHIFT = 1e-7 * np.array([1.0, 0.7548776662, 0.5698402910])


def rectangle_sheet(center, size):
    """
    Axis-aligned rectangular sheet as a triangulated surface.

    Parameters
    ----------
    center : sequence of float
        Centre (x, y, z) [m].
    size : sequence of float
        Extent (sx, sy, sz) [m]; exactly one entry must be zero (the sheet
        normal).

    Returns
    -------
    pyvista.PolyData
        Two triangles, usable in ``GridFIT3D(sheets=...)``.
    """
    center = np.asarray(center, float)
    size = np.asarray(size, float)
    if np.count_nonzero(size == 0) != 1:
        raise ValueError("A rectangle sheet needs exactly one zero size entry.")
    w = int(np.flatnonzero(size == 0)[0])
    u, v = _CYCLIC[w]
    corners = np.tile(center, (4, 1))
    for n, (su, sv) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        corners[n, u] += su * size[u] / 2
        corners[n, v] += sv * size[v] / 2
    return pv.PolyData(corners, faces=[3, 0, 1, 2, 3, 0, 2, 3])


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
    (c0, p0), (c1, p1), (c2, p2) = ends
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


# Threads for the plane sections (None: `wakis._lanczos.default_threads`);
# set to 1 by `ConformalGeometry` on MPI grids
_THREADS = [None]


def _section_areas(plane, p, q, n_planes, u_nodes, v_nodes):
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
    from ._lanczos import _pool, default_threads

    nu, nv = len(u_nodes), len(v_nodes)
    area = np.zeros((n_planes, nu - 1, nv - 1))
    lu = np.zeros((n_planes, nv, nu - 1))
    lv = np.zeros((n_planes, nu, nv - 1))
    if len(plane) == 0 or nu < 2 or nv < 2:
        return area, lu, lv
    order = np.argsort(plane, kind="stable")
    sorted_planes = plane[order]
    k_min, k_max = int(sorted_planes[0]), int(sorted_planes[-1]) + 1
    threads = _THREADS[0] or default_threads()
    work = (k_max - k_min) * nu * nv
    n_blocks = 1 if threads == 1 or work < 400_000 else min(4 * threads, k_max - k_min)
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
    seg_ids, params = [np.arange(len(p))] * 2, [np.zeros(len(p)), np.ones(len(p))]
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


def _pec_metrics(tri, nodes, closed=True, plane=None):
    """
    Inside lengths of all primary edges and inside areas of all primary faces.

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
        area, lu, lv = _section_areas(*seg, n_planes, nodes[u], nodes[v])
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
    """Bounds of the dual intervals centred on each node."""
    d = np.diff(nodes)
    mids = 0.5 * (nodes[1:] + nodes[:-1])
    return np.concatenate([[nodes[0] - d[0] / 2], mids, [nodes[-1] + d[-1] / 2]])


def _sub_samples(bounds, n_sub):
    """Midpoint samples splitting each interval of `bounds` into n_sub parts."""
    frac = (np.arange(n_sub) + 0.5) / n_sub
    return (bounds[:-1, None] + np.diff(bounds)[:, None] * frac).ravel()


def _dual_fractions(tri, nodes, n_sub):
    """
    Volume fraction of the solid in the dual cell of every primary edge.

    The dual cell of an edge along c spans the edge and half a cell on both
    sides transversally. Its volume is integrated exactly in x and y
    (section areas in planes z = const) and with the n_sub-point midpoint
    rule along z, which is exact for geometry extruded along z.

    Returns
    -------
    list of 3 ndarrays (cells along c, nodes transversally), to be cropped
    to the field shape.
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
        area, _, _ = _section_areas(*seg, len(samples), u_nodes, v_nodes)
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


class ConformalGeometry:
    """
    Conformal metric data of a `GridFIT3D`: per primary edge the length
    inside PEC, per primary face the area inside PEC, and per E edge the
    volume fractions and interface normals of the dielectric solids.

    Built (and cached on the grid) by `SolverFIT3D` for grids with
    ``conformal`` or ``sheets``; all arrays are in the field layout
    (Nx, Ny, Nz) per component, lengths in [m] and areas in [m^2].

    Parameters
    ----------
    grid : GridFIT3D
        Grid with STL solids, materials and/or sheets.
    background_pec : bool
        True when the solver background is PEC: everything outside the
        non-PEC solids is then conductor.
    mode : {'pec', 'all', 'tensor'}
        'pec': conformal PEC and sheets only, dielectric edges take the
        material with the largest share of their dual cell (staircase);
        'all': also the anisotropic averaging of dielectric solids;
        'tensor': as 'all', plus the off-diagonal couplings of the full
        averaged tensor (`wakis.tensor_averaging`).
    n_sub : int
        Midpoint samples per dual cell in the third direction for the volume
        fractions (the other two directions are exact).
    """

    def __init__(self, grid, background_pec=False, mode="all", n_sub=4):
        threads = _THREADS[0]
        if getattr(grid, "use_mpi", False):
            _THREADS[0] = 1  # one thread per MPI rank
        try:
            self._build(grid, background_pec, mode, n_sub)
        finally:
            _THREADS[0] = threads

    def _build(self, grid, background_pec, mode, n_sub):
        self.mode = mode
        self.n_sub = n_sub
        origin = np.array([grid.x[0], grid.y[0], grid.z[0]])
        nodes = [grid.x - origin[0], grid.y - origin[1], grid.z - origin[2]]
        self._nodes = nodes
        shape = (grid.Nx, grid.Ny, grid.Nz)
        self.shape = shape
        cell = min(np.min(grid.dx), np.min(grid.dy), np.min(grid.dz))
        shift = _SHIFT * cell
        zeros = [np.zeros(shape) for _ in range(3)]
        self.pec_length, self.pec_area = list(zeros), list(zeros)
        open_length, open_area = list(zeros), list(zeros)
        self.fractions = {}
        self.normals = {}
        self.sheet_outlines = []

        solids = grid.stl_solids or {}
        materials = grid.stl_materials or {}
        for key in solids:
            surface = grid.read_stl(key)
            tri = _outward(_triangles(surface)) - origin
            if np.isinf(materials[key][0]):
                # Points on the surface belong to the conductor (closed set)
                lengths, areas = self._metrics(tri, shift, np.maximum)
                self.pec_length = list(map(np.maximum, self.pec_length, lengths))
                self.pec_area = list(map(np.maximum, self.pec_area, areas))
                continue
            frac = _dual_fractions(tri + shift, nodes, n_sub)
            self.fractions[key] = [self._crop(f) for f in frac]
            self.normals[key] = self._interface_normals(surface, grid, key)
            if background_pec:
                # Open set: the surface of a hole in a conductor is metal
                lengths, areas = self._metrics(tri, shift, np.minimum)
                open_length = [a + b for a, b in zip(open_length, lengths)]
                open_area = [a + b for a, b in zip(open_area, areas)]

        if background_pec:
            d = [np.diff(n) for n in nodes]
            full_length = [self._crop(_axis_array(d[c], c, 3)) for c in range(3)]
            full_area = []
            for w in range(3):
                u, v = _CYCLIC[w]
                du, dv = _axis_array(d[u], u, 3), _axis_array(d[v], v, 3)
                full_area.append(self._crop(du * dv))
            for c in range(3):
                closed = full_length[c] - np.minimum(open_length[c], full_length[c])
                self.pec_length[c] = np.maximum(self.pec_length[c], closed)
                closed = full_area[c] - np.minimum(open_area[c], full_area[c])
                self.pec_area[c] = np.maximum(self.pec_area[c], closed)

        for key, surface in (grid.sheets or {}).items():
            tri = _triangles(surface) - origin
            plane = self._sheet_plane(tri, key)
            outline = _sheet_segments(tri, plane[0])[1:]
            self.sheet_outlines.append((plane, *outline))
            lengths, areas = self._metrics(tri, shift, np.maximum, plane)
            self.pec_length = list(map(np.maximum, self.pec_length, lengths))
            self.pec_area = list(map(np.maximum, self.pec_area, areas))

    def _crop(self, array):
        return array[: self.shape[0], : self.shape[1], : self.shape[2]]

    def _sheet_plane(self, tri, key):
        """Normal axis and node index of a planar axis-aligned sheet."""
        extent = np.ptp(tri.reshape(-1, 3), axis=0)
        flat = np.flatnonzero(extent <= 1e-9 * max(extent.max(), 1e-30))
        if flat.size != 1:
            raise ValueError(
                f"Sheet '{key}' must be planar and normal to x, y or z "
                "(oblique sheets are not supported)."
            )
        w = int(flat[0])
        position = tri[0, 0, w]
        nodes = self._nodes[w]
        k = int(np.argmin(np.abs(nodes - position)))
        cell = np.min(np.diff(nodes))
        if abs(nodes[k] - position) > 1e-6 * cell:
            raise ValueError(
                f"Sheet '{key}' lies between the grid planes along "
                f"{'xyz'[w]}; place a mesh node on it."
            )
        if k >= self.shape[w]:
            raise ValueError(f"Sheet '{key}' lies outside the field grid.")
        return w, k

    def _metrics(self, tri, shift, reduce, plane=None):
        """
        Inside lengths and areas of a solid (plane None) or a sheet, reduced
        over two opposite shifts of the geometry: `np.maximum` counts edges
        and faces lying on the surface as inside, `np.minimum` as outside.
        """
        out_l, out_a = None, None
        for sign in (1.0, -1.0):
            moved = tri + sign * shift
            if plane is not None:
                moved[:, :, plane[0]] = self._nodes[plane[0]][plane[1]]
            lengths, areas = _pec_metrics(moved, self._nodes, plane is None, plane)
            lengths = [self._crop(a) for a in lengths]
            areas = [self._crop(a) for a in areas]
            if out_l is None:
                out_l, out_a = lengths, areas
            else:
                out_l = list(map(reduce, out_l, lengths))
                out_a = list(map(reduce, out_a, areas))
        return out_l, out_a

    def _interface_normals(self, surface, grid, key):
        """Surface normals at the E edges cut by the interface of a solid."""
        centres = (
            grid.x[:-1] + grid.dx / 2,
            grid.y[:-1] + grid.dy / 2,
            grid.z[:-1] + grid.dz / 2,
        )
        nodes = (grid.x[:-1], grid.y[:-1], grid.z[:-1])
        indices, points = [], []
        for c in range(3):
            f = self.fractions[key][c]
            cut = (f > 1e-9) & (f < 1 - 1e-9)
            idx = np.nonzero(cut)
            coords = [centres[a] if a == c else nodes[a] for a in range(3)]
            indices.append(idx)
            points.append(np.stack([coords[a][idx[a]] for a in range(3)], axis=1))
        # One closest-cell search for the three components
        found = _surface_normals(surface, np.concatenate(points))
        normals, start = [], 0
        for idx, pts in zip(indices, points):
            n = np.zeros((*self.shape, 3))
            n[idx] = found[start : start + len(pts)]
            start += len(pts)
            normals.append(n)
        return normals


def _material_tensors(solver, geo):
    """
    Inverse permittivity and conductivity per E edge from the volume
    fractions of the non-PEC solids (Kottke averaging in modes 'all' and
    'tensor'). In mode 'tensor' also returns, per component, the
    anisotropy <1/eps> - 1/<eps> [m/F] and the full interface normal of
    every edge (for the off-diagonal tensor couplings), else None.
    """
    grid = solver.grid
    materials = grid.stl_materials or {}
    keys = list(geo.fractions)
    eps = np.array([materials[k][0] * eps_0 for k in keys])
    sig = np.array([materials[k][2] for k in keys])
    bg_pec = np.isinf(solver.eps_bg)
    ieps, sigma, aniso, normal = [], [], [], []
    # Per-edge mixture (fractions, background share, n_c^2), kept for the
    # dispersive subpixel averaging of `wakis.tensor_dispersive`
    mixture = {
        "keys": keys,
        "fs": [],
        "f_bg": [],
        "p": [],
        "normal": [],
        "eps_bg": None,
    }
    for c in range(3):
        # Solids listed later replace earlier ones where they overlap (as for
        # the cell masks): f_j <- f_j (1 - f_k) for j before k
        fs = np.zeros((len(keys), *geo.shape))
        n_c = np.zeros(geo.shape)
        n_full = np.zeros((*geo.shape, 3)) if geo.mode == "tensor" else None
        for i, key in enumerate(keys):
            f = np.clip(geo.fractions[key][c], 0.0, 1.0)
            fs[:i] *= 1.0 - f
            fs[i] = f
            # Interface normal of the last solid cutting the dual cell
            cut = (f > 1e-9) & (f < 1 - 1e-9)
            n_c = np.where(cut, geo.normals[key][c][..., c], n_c)
            if geo.mode == "tensor":
                n_full = np.where(cut[..., None], geo.normals[key][c], n_full)
        total = fs.sum(axis=0)
        if bg_pec:
            # The non-PEC part of the dual cell holds only solids
            fs = fs / np.where(total > 0, total, 1.0)
            f_bg = np.where(total > 0, 0.0, 1.0)
            eps_bg = eps_0
        else:
            f_bg = 1.0 - total
            eps_bg = solver.eps_bg
        if geo.mode in ("all", "tensor"):
            mean_eps = f_bg * eps_bg + np.tensordot(eps, fs, axes=1)
            mean_ieps = f_bg / eps_bg + np.tensordot(1.0 / eps, fs, axes=1)
            # Diagonal of P <1/eps> + (1 - P) <eps>^-1, P = n n^T [Kottke 2008]
            p = n_c**2
            ieps_c = p * mean_ieps + (1.0 - p) / mean_eps
            aniso.append(np.clip(mean_ieps - 1.0 / mean_eps, 0.0, None))
            normal.append(n_full)
            mixture["fs"].append(fs)
            mixture["f_bg"].append(f_bg * np.ones(geo.shape))
            mixture["p"].append(p)
            mixture["normal"].append(n_full)
            mixture["eps_bg"] = eps_bg
        else:
            # Majority material (staircase dielectric)
            share = np.concatenate([f_bg[None], fs])
            values = np.concatenate([[eps_bg], eps])
            ieps_c = 1.0 / values[np.argmax(share, axis=0)]
        sigma_c = f_bg * (0.0 if bg_pec else solver.sigma_bg) + np.tensordot(
            sig, fs, axes=1
        )
        ieps.append(ieps_c)
        sigma.append(sigma_c)
    if geo.mode in ("all", "tensor"):
        solver._conformal_mixture = mixture
    if geo.mode != "tensor":
        return ieps, sigma, None
    return ieps, sigma, (aniso, normal)


def _redistribution(
    lengths, areas, cell_lengths, ghosts, threshold, pattern=None, merge_only=()
):
    """
    Stabilized inverse face-area operator G (H = G b / mu for face fluxes b)
    per face component, replacing diag(1/A_open).

    A face is unstable when its open area is below `threshold` times
    A_stab = max(l_i h_i) / 2, the largest product of an open edge length
    and the face size perpendicular to it (the criterion of the legacy Wakis
    ECT solver, `grid2D.mark_cells`). Its flux is redistributed to stable
    in-plane neighbours along the direction of its longest open edge with
    weights T: (2, -1) to the first and second neighbour (linear
    extrapolation), or 1 to the first neighbour when the second is not
    available (cell merging). Next to a sheet edge, where H normal to the
    sheet is singular, the weight is instead the ratio H_u / H_n of the
    knife-edge field (`pattern`). An unstable face without stable neighbours
    keeps its own flux with the area raised to threshold * A_stab. With the
    receiving areas
    Â_n = A_n + sum_u T_nu A_u,

        G = T^T diag(1/Â) T,

    so stable faces carry H_n = (b_n + sum_u T_nu b_u) / Â_n and the
    unstable face H_u = sum_n T_nu H_n. G is symmetric positive
    semi-definite and the faces of one plane share their dual length, so the
    update keeps a conserved discrete energy; a uniform H is reproduced
    exactly (and the knife-edge field near sheet edges).

    Faces of the components in `merge_only` use weight 1 to the first
    neighbour only (cell merging, no extrapolation). `threshold` may be a
    sequence of three per-component values. The TE/TM scheme does not
    redistribute the faces normal to x and y (`TETE_XY_THRESHOLD` = 0): the
    (2, -1) extrapolation raises the longitudinal operator above
    4 c^2/dz^2 and breaks the magic time step, and a merged face couples two
    (x, y) columns so that the bunch self-field is no longer an exact
    discrete travelling wave.

    Returns
    -------
    list of 3 scipy.sparse matrices (N x N), number of unstable faces
    """
    from scipy.sparse import coo_matrix

    shape = areas[0].shape
    n = int(np.prod(shape))
    ops, n_unstable = [], 0
    thresholds = np.broadcast_to(np.asarray(threshold, float), (3,))
    for w in range(3):
        threshold = thresholds[w]
        u, v = _CYCLIC[w]
        Lu, Lv = lengths[u], lengths[v]
        A = areas[w]

        def at(array, ijk, axis=None, step=0):
            """array at the faces ijk shifted by step along axis (wrapping)."""
            if axis is None:
                return array[ijk]
            idx = list(ijk)
            idx[axis] = (idx[axis] + step) % shape[axis]
            return array[tuple(idx)]

        def stab_active(ijk):
            # Open lengths of the four edges: u-edges at v and v+1, v-edges
            # at u and u+1 (index v+1 / u+1 wraps around, as np.roll)
            lu0, lu1 = at(Lu, ijk), at(Lu, ijk, v, 1)
            lv0, lv1 = at(Lv, ijk), at(Lv, ijk, u, 1)
            active = (lu0 + lu1 + lv0 + lv1) > 0
            stab = 0.5 * np.maximum(
                np.maximum(lu0, lu1) * at(cell_lengths[v], ijk),
                np.maximum(lv0, lv1) * at(cell_lengths[u], ijk),
            )
            return active, stab

        # A face is unstable when A < threshold * A_stab; since the open
        # lengths do not exceed the cell lengths, A_stab <= A_full / 2, so
        # only faces with A < threshold * A_full / 2 need the full test
        small = A < threshold * (0.5 * (cell_lengths[u] * cell_lengths[v]))
        # ... and an open edge (active face; wrapping shifts, as above)
        open_u, open_v = Lu > 0, Lv > 0
        small &= (
            open_u | np.roll(open_u, -1, axis=v) | open_v | np.roll(open_v, -1, axis=u)
        )
        cand = np.flatnonzero(small.ravel(order="F"))
        ijk = np.unravel_index(cand, shape, order="F")
        active_c, stab_c = stab_active(ijk)
        unstable_c = active_c & (A[ijk] < threshold * stab_c)
        unstable_full = np.zeros(shape, dtype=bool)
        cand, stab_c = cand[unstable_c], stab_c[unstable_c]
        ijk = tuple(x[unstable_c] for x in ijk)
        unstable_full[ijk] = True

        def receiver(nijk):
            """Stable active faces inside the domain (not the ghost layer)."""
            active, _ = stab_active(nijk)
            ok = active & ~unstable_full[nijk]
            for axis in (u, v):
                if ghosts[axis]:
                    ok &= nijk[axis] != shape[axis] - 1
            return ok

        # Neighbour search on the unstable faces only (as on whole arrays:
        # first direction with the best score wins, no wrap-around)
        strides = (1, shape[0], shape[0] * shape[1])  # flat index, order "F"
        best = np.zeros(cand.size)
        first_c = np.full(cand.size, -1)
        second_c = np.full(cand.size, -1)
        for axis, step, shared in (
            (u, 1, at(Lv, ijk, u, 1)),
            (u, -1, at(Lv, ijk)),
            (v, 1, at(Lu, ijk, v, 1)),
            (v, -1, at(Lu, ijk)),
        ):
            ok = []
            for k in (1, 2):
                pos = ijk[axis] + k * step
                inside = (pos >= 0) & (pos < shape[axis])
                nijk = list(ijk)
                nijk[axis] = np.where(inside, pos, 0)
                ok.append(inside & receiver(tuple(nijk)))
            ok1, ok2 = ok
            n1 = cand + step * strides[axis]
            n2 = cand + 2 * step * strides[axis]
            score = np.where(ok1, shared * (1.0 + 2.0 * ok2), 0.0)
            take = score > best
            best = np.where(take, score, best)
            first_c = np.where(take, n1, first_c)
            second_c = np.where(take, np.where(ok2, n2, -1), second_c)
        fixed_c = first_c >= 0
        n_unstable += int(fixed_c.sum())
        a = A.ravel(order="F")
        # Faces next to a sheet edge: one partner, weight from the edge field
        fx, fst, snd = cand[fixed_c], first_c[fixed_c], second_c[fixed_c]
        if w in merge_only:
            snd = np.full_like(snd, -1)
        alpha = np.ones(fx.size)
        sing = np.zeros(fx.size, dtype=bool)
        if pattern is not None:
            field = pattern[0][w].ravel(order="F")
            segment = pattern[1][w].ravel(order="F")
            sing = (
                np.isfinite(field[fx])
                & np.isfinite(field[fst])
                & (segment[fx] == segment[fst])
            )
            alpha[sing] = field[fx][sing] / field[fst][sing]
        two = ~sing & (snd >= 0)
        one = sing | (snd < 0)
        keep = np.ones(n, dtype=bool)
        keep[fx] = False
        # Redistribution matrix T restricted to the unstable columns, E:
        # T = I on the kept faces + E; same entry order as the full T
        e_rows = np.concatenate([fst[one], fst[two], snd[two]])
        e_cols = np.concatenate([fx[one], fx[two], fx[two]])
        e_vals = np.concatenate(
            [alpha[one], 2 * np.ones(two.sum()), -np.ones(two.sum())]
        )
        # Receiving area a_hat = T a, rows summed in column order (as the
        # CSR product with T); rows without donors are a_n
        a_hat = np.where(keep, a, 0.0)
        recv = np.unique(e_rows)
        rows = np.concatenate([e_rows, recv])
        cols = np.concatenate([e_cols, recv])
        vals = np.concatenate([e_vals * a[e_cols], 1.0 * a[recv]])
        order = np.lexsort((cols, rows))
        a_hat[recv] = 0.0
        np.add.at(a_hat, rows[order], vals[order])
        # Reproduces H_u = alpha H_n, i.e. A_n + alpha^2 A_u
        np.add.at(a_hat, fst[sing], (alpha[sing] ** 2 - alpha[sing]) * a[fx][sing])
        # Unstable faces without stable neighbours (e.g. the tip of a section
        # touching a grid plane) fall back to the stable area itself
        clamp = np.zeros(n)
        clamp[cand[~fixed_c]] = threshold * stab_c[~fixed_c]
        a_hat = np.maximum(a_hat, clamp)
        # Faces without open area and edges (inside PEC) keep H = 0
        inv = np.where(a_hat > 0, 1.0 / np.where(a_hat > 0, a_hat, 1.0), 0.0)
        # G = T^T diag(inv) T = diag(inv) on kept faces + D E + (D E)^T
        # + E^T D E (E has kept rows and unstable columns)
        de = inv[e_rows] * e_vals
        E = coo_matrix((e_vals, (e_rows, e_cols)), shape=(n, n)).tocsr()
        ete = (E.T @ diags(inv) @ E).tocoo()
        rows = np.concatenate([e_rows, e_cols, ete.row])
        cols = np.concatenate([e_cols, e_rows, ete.col])
        vals = np.concatenate([de, de, ete.data])
        couple = coo_matrix((vals, (rows, cols)), shape=(n, n))
        ops.append((diags(np.where(keep, inv, 0.0)) + couple).tocsr())
    return ops, n_unstable


# Edges and faces within this many cells of a sheet edge get the
# singular-field correction
EDGE_RADIUS = 1.5


def _edge_potential(s, w):
    """
    Static field of a knife edge along t at s = w = 0, the metal half-plane
    at w = 0, s < 0: F = sqrt(s + i w) (principal branch, cut on the metal).
    Re F is the potential of the field with charge at the edge (and E_t of
    the field with current along the edge); Im F is its stream function.
    """
    return np.sqrt(s + 1j * w + 0j)


def _potential_integral(s1, s2, w1, w2):
    """Integral of Re sqrt(s + i w) over [s1, s2] x [w1, w2]."""

    def corner(s, w):
        z = s + 1j * w + 0j
        return (4.0 / 15.0) * z * z * np.sqrt(z) / 1j

    def part(a, b):  # on one side of the cut (antiderivative continuous)
        return (corner(s2, b) - corner(s1, b) - corner(s2, a) + corner(s1, a)).real

    upper = part(np.maximum(w1, 0.0), np.maximum(w2, 0.0))
    lower = part(np.minimum(w1, -0.0), np.minimum(w2, -0.0))
    return upper + lower


def _sheet_edge_factors(geo, L_open, A_open, L, tl):
    """
    Singular-field correction of the material matrices near sheet edges.

    Near the straight edge of a zero-thickness conductor the static field is
    F = sqrt(zeta), zeta = s + i w in the plane normal to the edge (s away
    from the metal, w normal to the sheet). The standard FIT metric
    (voltage = E x length, flux = D x area) under-represents the stored
    energy of this singular field, which makes a sheet edge act about a
    third of a cell larger. Here every primary edge and face within
    EDGE_RADIUS cells of an axis-aligned sheet edge gets the ratio of its
    exact voltage-to-flux relation for F to the standard one:

    * E across the edge (charge at the edge): e = Re F difference along the
      edge, d = Im F difference across its dual face;
    * E along the edge: e = Re F at the node, d = integral of Re F over the
      dual face;
    * H faces (current along the edge): b = Re F difference along the face,
      h = Im F difference along the dual edge.

    Dual faces and edges keep their FIT position; where one crosses the
    sheet, the principal branch goes around the tip, so that the discrete
    Gauss and Ampere laws exclude the charge and current on the sheet.

    Returns
    -------
    eps_factor, mu_factor : list of 3 ndarrays
        Factors on 1/eps of the E edges and on the inverse face areas.
    pattern : tuple of two lists of 3 ndarrays
        Knife-edge H on the corrected faces (NaN elsewhere) and the index of
        the sheet edge it belongs to, for `_redistribution`.
    """
    shape = geo.shape
    nodes = geo._nodes
    dual = [_dual_bounds(n) for n in nodes]
    eps_f = [np.ones(shape) for _ in range(3)]
    mu_f = [np.ones(shape) for _ in range(3)]
    best_e = [np.full(shape, np.inf) for _ in range(3)]
    best_m = [np.full(shape, np.inf) for _ in range(3)]
    # Knife-edge H per face (for the stabilization) and its segment
    h_field = [np.full(shape, np.nan) for _ in range(3)]
    h_segment = [np.full(shape, -1) for _ in range(3)]
    segment_id = 0

    def assign(factor, best, comp, idx, kappa, dist, limit, field=None):
        ok = np.isfinite(kappa) & (kappa > 0) & (dist <= limit)
        ok &= dist < best[comp][idx]
        idx = tuple(a[ok] for a in idx)
        factor[comp][idx] = kappa[ok]
        best[comp][idx] = dist[ok]
        if field is not None:
            h_field[comp][idx] = field[ok]
            h_segment[comp][idx] = segment_id

    for (w, k), p, q in geo.sheet_outlines:
        u, v = _CYCLIC[w]
        for P, Q in zip(p, q):
            segment_id += 1
            d = Q - P
            length = np.hypot(*d)
            if abs(d[0]) <= 1e-9 * length:
                t_in = 1
            elif abs(d[1]) <= 1e-9 * length:
                t_in = 0
            else:
                continue  # oblique sheet edges are not corrected
            T, S = (u, v)[t_in], (u, v)[1 - t_in]
            normal = np.array([d[1], -d[0]]) / length  # away from the metal
            sigma = np.sign(normal[1 - t_in])
            s_e = P[1 - t_in]
            t0, t1 = sorted((P[t_in], Q[t_in]))
            W = w
            ns, nw, nt = nodes[S], nodes[W], nodes[T]
            last = shape[S] - 1
            tol = 1e-9 * (ns[-1] - ns[0])
            if not (ns[0] + tol < s_e < ns[last] - tol):
                continue  # on or beyond the domain boundary
            i0 = int(np.clip(np.searchsorted(ns, s_e), 1, len(ns) - 1))
            h_ref = max(ns[i0] - ns[i0 - 1], nw[min(k + 1, len(nw) - 1)] - nw[k])
            limit = EDGE_RADIUS * h_ref
            i_n = np.flatnonzero(np.abs(ns[: shape[S]] - s_e) <= limit + 2 * h_ref)
            k_n = np.arange(max(k - 2, 0), min(k + 3, shape[W]))
            # Objects along the edge only: nodes strictly inside the segment
            # and cells not reaching beyond it (corners have another field)
            j_n = np.flatnonzero(
                (nt[: shape[T]] > t0 + tol) & (nt[: shape[T]] < t1 - tol)
            )
            j_c = np.flatnonzero(
                (nt[: shape[T]] >= t0 - tol) & (nt[1 : shape[T] + 1] <= t1 + tol)
            )
            i_c = i_n[i_n < len(ns) - 1]
            k_c = k_n[k_n < len(nw) - 1]

            def sp(x):
                return sigma * (x - s_e)

            def grid_idx(i, kk, j):
                I, K, J = np.meshgrid(i, kk, j, indexing="ij")
                idx = [None] * 3
                idx[S], idx[W], idx[T] = I, K, J
                return tuple(idx), I, K

            # Across-edge E along S and H faces normal W (cells along S)
            for comp, jj in ((S, j_n), (W, j_c)):
                idx, I, K = grid_idx(i_c, k_n, jj)
                a, b = sp(ns[I]), sp(ns[I + 1])
                lo, hi = np.minimum(a, b), np.maximum(a, b)
                on = K == k
                lo = np.where(on, np.maximum(lo, 0.0), lo)
                wp = nw[K] - nw[k]
                mid = sp(0.5 * (ns[I] + ns[I + 1]))
                wl, wh = dual[W][K] - nw[k], dual[W][K + 1] - nw[k]
                with np.errstate(divide="ignore", invalid="ignore"):
                    e = (_edge_potential(hi, wp) - _edge_potential(lo, wp)).real
                    flux = (_edge_potential(mid, wh) - _edge_potential(mid, wl)).imag
                    open_len = hi - lo
                    dist = np.hypot(np.clip(0.0, lo, hi), wp)
                    # Only where the cut matches this edge alone (not at
                    # corners, where the other edge opens the edge or face)
                    if comp == S:
                        match = np.abs(L_open[S][idx] - open_len) <= 1e-6 * L[S][idx]
                        kappa = np.abs(e / flux) * tl[W][K] / L_open[S][idx]
                        kappa = np.where(match, kappa, np.nan)
                        assign(eps_f, best_e, S, idx, kappa, dist, limit)
                    else:
                        area = open_len * L[T][idx]
                        match = (
                            np.abs(A_open[W][idx] - area)
                            <= 1e-6 * L[S][idx] * L[T][idx]
                        )
                        kappa = np.abs(flux / e) * open_len / tl[W][K]
                        kappa = np.where((open_len > 0) & match, kappa, np.nan)
                        field = np.abs(flux) / tl[W][K]
                        assign(mu_f, best_m, W, idx, kappa, dist, limit, field)

            # E along W and H faces normal S (cells along W)
            for comp, jj in ((W, j_n), (S, j_c)):
                idx, I, K = grid_idx(i_n, k_c, jj)
                sv = sp(ns[I])
                wa, wb = nw[K] - nw[k], nw[K + 1] - nw[k]
                sl, sh = sp(dual[S][I]), sp(dual[S][I + 1])
                wm = 0.5 * (wa + wb)
                with np.errstate(divide="ignore", invalid="ignore"):
                    e = (_edge_potential(sv, wb) - _edge_potential(sv, wa)).real
                    flux = (_edge_potential(sh, wm) - _edge_potential(sl, wm)).imag
                    dist = np.hypot(sv, np.clip(0.0, wa, wb))
                    if comp == W:
                        kappa = np.abs(e / flux) * tl[S][I] / L_open[W][idx]
                        assign(eps_f, best_e, W, idx, kappa, dist, limit)
                    else:
                        kappa = np.abs(flux / e) * (wb - wa) / tl[S][I]
                        field = np.abs(flux) / tl[S][I]
                        assign(mu_f, best_m, S, idx, kappa, dist, limit, field)

            # E along the edge (T) at the nodes; an edge reaching beyond the
            # segment end (corner) is corrected on the part beside the sheet
            j_o = np.flatnonzero(
                (nt[1 : shape[T] + 1] > t0 + tol) & (nt[: shape[T]] < t1 - tol)
            )
            idx, I, K = grid_idx(i_n, k_n, j_o)
            J = idx[T]
            beside = np.minimum(nt[J + 1], t1) - np.maximum(nt[J], t0)
            beside = np.clip(beside / (nt[J + 1] - nt[J]), 0.0, 1.0)
            sv, wv = sp(ns[I]), nw[K] - nw[k]
            sl, sh = np.sort([sp(dual[S][I]), sp(dual[S][I + 1])], axis=0)
            wl, wh = dual[W][K] - nw[k], dual[W][K + 1] - nw[k]
            phi = _edge_potential(sv, wv).real
            with np.errstate(divide="ignore", invalid="ignore"):
                integral = _potential_integral(sl, sh, wl, wh)
                kappa = phi * tl[S][I] * tl[W][K] / integral
            whole = L_open[T][idx] >= (1 - 1e-6) * L[T][idx]
            kappa = 1.0 - beside + beside * kappa
            kappa = np.where((phi > 0) & whole, kappa, np.nan)
            assign(eps_f, best_e, T, idx, kappa, np.hypot(sv, wv), limit)
    return eps_f, mu_f, (h_field, h_segment)


def apply_conformal_materials(solver):
    """
    Conformal metrics and materials of a solver whose grid was built with
    ``conformal`` or ``sheets``: called by `SolverFIT3D` after the STL
    materials. Replaces ``solver.L``, ``solver.Ds`` by the open edge lengths,
    ``solver.iA`` by the inverse open face areas and ``solver.iDa`` by the
    merged-cell operator ``M diag(1/A_open)`` (see `_merge_groups`), and
    ``ieps`` and ``sigma`` by the conformal values. ``imu`` keeps the
    cell-mask values (merged faces assume a uniform permeability).
    """
    from scipy.sparse import block_diag

    grid = solver.grid
    background_pec = bool(np.isinf(solver.eps_bg))
    cache = grid.__dict__.setdefault("_conformal_geometry", {})
    if background_pec not in cache:
        cache[background_pec] = ConformalGeometry(
            grid,
            background_pec=background_pec,
            mode=grid.conformal or "pec",
            n_sub=grid.conformal_subsamples,
        )
    geo = cache[background_pec]
    real = solver.ieps.array.dtype
    L = [grid.L.to_matrix(d) for d in "xyz"]
    A = [1.0 / grid.iA.to_matrix(d) for d in "xyz"]
    tol = 1e-6

    L_open = [np.clip(L[c] - geo.pec_length[c], 0.0, None) for c in range(3)]
    L_open = [np.where(L_open[c] < tol * L[c], 0.0, L_open[c]) for c in range(3)]
    A_open = [np.clip(A[c] - geo.pec_area[c], 0.0, None) for c in range(3)]
    A_open = [np.where(A_open[c] < tol**2 * A[c], 0.0, A_open[c]) for c in range(3)]
    if getattr(solver, "scheme", "yee") == "tete":
        A_open = _tete_face_areas(L_open, A_open, L)
    eps_factor, pattern = None, None
    if geo.sheet_outlines and grid.sheet_edge_correction:
        tl = [
            grid.tL.to_matrix("x")[:, 0, 0],
            grid.tL.to_matrix("y")[0, :, 0],
            grid.tL.to_matrix("z")[0, 0, :],
        ]
        eps_factor, mu_factor, pattern = _sheet_edge_factors(geo, L_open, A_open, L, tl)
        # The corrected inverse area acts as an effective open area
        A_open = [A_open[c] / mu_factor[c] for c in range(3)]
    tete = getattr(solver, "scheme", "yee") == "tete"
    threshold = (
        (TETE_XY_THRESHOLD, TETE_XY_THRESHOLD, MERGE_THRESHOLD)
        if tete
        else MERGE_THRESHOLD
    )
    ops, n_unstable = _redistribution(
        L_open, A_open, L, grid.edge_ghosts, threshold, pattern,
        merge_only=(0, 1) if tete else (),
    )  # fmt: skip
    solver.conformal_unstable_faces = n_unstable

    solver.L = solver.L.copy()
    solver.iA = solver.iA.copy()
    for c, d in enumerate("xyz"):
        setattr(solver.L, f"field_{d}", L_open[c])
        setattr(solver.iA, f"field_{d}", ops[c].diagonal())
    N3 = 3 * solver.N
    solver.Ds = diags(solver.L.toarray(), shape=(N3, N3), dtype=solver.dtype)
    solver.iDa = block_diag(ops, format="csr", dtype=solver.dtype)
    solver.conformal_face_ops = [op.astype(solver.dtype) for op in ops]

    if geo.fractions or np.isinf(solver.eps_bg):
        ieps, sigma, tensor = _material_tensors(solver, geo)
        for c, d in enumerate("xyz"):
            setattr(solver.ieps, f"field_{d}", ieps[c].astype(real))
            setattr(solver.sigma, f"field_{d}", sigma[c].astype(real))
        if tensor is not None:
            _set_tensor_edges(solver, *tensor)
        if np.any(solver.sigma.toarray() > 0):
            solver.use_conductivity = True
    # E edges completely inside PEC carry no voltage; the sheet-edge factors
    # are kept so that materials written later can be corrected the same way
    closed = np.concatenate([(L_open[c] == 0).ravel(order="F") for c in range(3)])
    factor = np.where(closed, 0.0, 1.0)
    if eps_factor is not None:
        factor *= np.concatenate([f.ravel(order="F") for f in eps_factor])
    solver.conformal_ieps_factor = factor
    solver.ieps.fromarray(solver.ieps.toarray() * factor)
    solver.sigma.fromarray(np.where(closed, 0.0, solver.sigma.toarray()))


def _tete_face_areas(L_open, A_open, L):
    """
    Open areas of the faces normal to x and y for the TE/TM scheme.

    The face normal to x between z_k and z_k+1 gets the trapezoidal area
    dz (l_y,k + l_y,k+1) / 2 from the open lengths of its two y edges (and
    likewise for y faces). It is exact for z-invariant walls (beam pipes) and
    for walls linear in z, and it bounds the longitudinal operator of every
    (x, y) column by Gershgorin's theorem: each row of B_L A_L sums to
    (tl / (eps mu tA)) [(l_k+1 + l_k)/A_r + (l_k + l_k-1)/A_l] = 4 c^2/dz^2,
    so the magic step c dt = dz stays stable on conformal grids. The
    subsampled areas of the Yee path instead differ from the edge-length
    trapezoid by several percent and lower the stable step below dz/c.
    Faces normal to z (transverse only) keep their areas.
    """
    out = list(A_open)
    dz = L[2]
    for face, edge in ((0, 1), (1, 0)):
        le = L_open[edge]
        upper = np.concatenate([le[:, :, 1:], le[:, :, -1:]], axis=2)
        out[face] = dz * 0.5 * (le + upper)
    return out


def _set_tensor_edges(solver, aniso, normal):
    """Store the anisotropic edges of mode 'tensor' in a `TensorOperator`."""
    from .tensor_averaging import TensorOperator

    a = np.concatenate([x.ravel(order="F") for x in aniso])
    n = np.concatenate([x.reshape(-1, 3, order="F") for x in normal])
    edges = np.flatnonzero(a > 0)
    op = TensorOperator(solver)
    op.set_edges(edges, a[edges], n[edges])
    solver.tensor_operator = op


def max_stable_dt(solver, tol=1e-3, method="lanczos"):
    """
    Largest stable leapfrog time step of the solver's (conformal) operators.

    The source-free update H' = H - dt A E, E' = E + dt B H' is stable for
    dt < 2 / sqrt(lambda_max(B A)). B A = D_1 C^H D_2 G C D_4, with the
    diagonal D_1 = ieps/tA, D_2 = tL, D_4 = L and G = iDa imu (diagonal, or
    block-symmetric for merged faces), is similar to the Hermitian positive
    semi-definite matrix R C^H D_2 G C R, R = (D_4 D_1)^{1/2}, whose largest
    eigenvalue is computed with Lanczos iterations.

    Parameters
    ----------
    tol : float
        Relative residual tolerance of the eigenvalue estimate.
    method : {'lanczos', 'arpack'}
        'lanczos' (default): `wakis._lanczos.stable_dt`, restricted to the
        open edges, multi-threaded, few stored vectors; 'arpack': SciPy
        `eigsh` on all edges (the original implementation, kept for
        comparison). Both agree within their tolerance (~1e-6 relative at
        tol=1e-3, to 1e-15 at tol=1e-8).

    Returns
    -------
    float
        2 / sqrt(lambda_max) [s].
    """
    if method == "lanczos":
        from ._lanczos import stable_dt

        return stable_dt(solver, tol=tol)
    d1 = solver.ieps.toarray() * solver.itDa.diagonal()
    d2 = solver.tDs.diagonal()
    right = np.sqrt(np.abs(solver.Ds.diagonal() * d1))
    imu = solver.imu.toarray()
    iDa = solver.iDa.tocsr()
    C = solver.C.tocsr()
    CH = C.conj().T.tocsr()

    def matvec(x):
        return right * (CH @ (d2 * (iDa @ (imu * (C @ (right * x))))))

    n = C.shape[1]
    op = LinearOperator((n, n), matvec=matvec, dtype=solver.dtype)
    lam = eigsh(op, k=1, which="LA", tol=tol, return_eigenvectors=False)[0]
    # The Lanczos estimate converges from below: add a margin of tol
    return 2.0 / np.sqrt(abs(lam) * (1.0 + 10 * tol))


def apply_conformal_timestep(solver, dt_given):
    """
    Reduce the solver time step to `cfln` times the stable limit of the
    conformal operators (when dt was not given) or warn when the given dt
    exceeds the limit.
    """
    op = getattr(solver, "tensor_operator", None)
    if op is not None:
        from . import tensor_averaging

        # Edges in PEC (open length 0) and ghost edges now have ieps = 0
        op.assemble(solver)
        dt_max = tensor_averaging.max_stable_dt(solver, op)
    else:
        dt_max = max_stable_dt(solver)
    if solver.use_mpi and solver.grid.use_mpi:
        # Every rank estimates its own slab: step with the smallest limit
        from mpi4py import MPI

        dt_max = solver.grid.comm.allreduce(dt_max, op=MPI.MIN)
    solver.conformal_dt_max = dt_max
    if dt_given:
        if solver.dt > dt_max:
            print(
                f"[!] Warning: dt={solver.dt:.3e} s exceeds the stable limit "
                f"{dt_max:.3e} s of the conformal operators."
            )
        return
    solver.dt = type(solver.dt)(min(solver.dt, solver.cfln * dt_max))
