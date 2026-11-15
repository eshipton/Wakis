# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Structured, body-fitted quadrilateral meshes for curvilinear (nonorthogonal)
FIT in 2D.

A `QuadMesh` is a conforming union of structured blocks (merged nodes).
Every cell is either

* a *polar* cell: an exact annular sector r0 < r < r1, th0 < th < th1 around
  the origin (curved sides; orthogonal, so its Hodge is diagonal), or
* a *general* quadrilateral with straight sides (its Hodge couples the two
  edges meeting at each corner: nonorthogonal FIT).

Generators:

* `ogrid_mesh` -- single-block O-grid: a smoothed "butterfly" core (square +
  four transition blocks) inside r_t, then polar rings out to r_out (the
  outermost rings can host a radial PML).
* `ogrid_cartesian_mesh` -- multi-block: core + polar rings to r_b, a
  transition block from the circle r_b to a square, and a Cartesian outer
  block (conforming at the square; a Cartesian PML can be put there).
* `radial_nodes` -- graded ring radii with exact break radii (interfaces,
  extraction circles, PML start).
"""

import numpy as np
import scipy.sparse as sp

__all__ = ["QuadMesh", "ogrid_mesh", "ogrid_cartesian_mesh", "boxed_cylinders_mesh", "radial_nodes"]


class QuadMesh:
    """
    Conforming 2D quadrilateral mesh.

    Parameters
    ----------
    nodes : ndarray (n_nodes, 2)
        Node coordinates [m].
    cells : ndarray of int (n_cells, 4)
        Counter-clockwise node indices; side k joins local nodes k and k+1.
    cell_tags : ndarray of int (n_cells,)
        Material tags (0 = background by convention).
    polar : ndarray (n_cells, 4) of float, optional
        (r0, r1, th0, th1) of polar cells, NaN for general cells. Polar cells
        are ordered (r0, th0), (r1, th0), (r1, th1), (r0, th1).
    polar_center : ndarray (n_cells, 2), optional
        Centre of each polar cell's circle (default: the origin).
    """

    def __init__(self, nodes, cells, cell_tags, polar=None, polar_center=None):
        self.nodes = np.asarray(nodes, float)
        self.cells = np.asarray(cells, np.int64)
        self.cell_tags = np.asarray(cell_tags, np.int64)
        nc = len(self.cells)
        self.polar = (
            np.full((nc, 4), np.nan) if polar is None else np.asarray(polar, float)
        )
        self.is_polar = ~np.isnan(self.polar[:, 0])
        self.polar_center = (
            np.zeros((nc, 2)) if polar_center is None else np.asarray(polar_center, float)
        )
        sides = np.stack([self.cells, np.roll(self.cells, -1, axis=1)], -1)
        key = np.sort(sides.reshape(-1, 2), axis=1)
        self.edges, inv = np.unique(key, axis=0, return_inverse=True)
        self.cell_edges = inv.reshape(nc, 4)
        ne, nn = len(self.edges), len(self.nodes)
        self.G = sp.csr_matrix(
            (
                np.r_[-np.ones(ne), np.ones(ne)],
                (np.r_[np.arange(ne), np.arange(ne)], self.edges.T.ravel()),
            ),
            shape=(ne, nn),
        )
        count = np.bincount(self.cell_edges.ravel(), minlength=ne)
        self.boundary_edges = np.flatnonzero(count == 1)
        self.boundary_nodes = np.unique(self.edges[self.boundary_edges])
        P = self.nodes[self.cells]
        self.cell_area = 0.5 * np.sum(
            P[..., 0] * np.roll(P[..., 1], -1, 1) - np.roll(P[..., 0], -1, 1) * P[..., 1],
            axis=1,
        )
        pa = self.polar[self.is_polar]
        self.cell_area[self.is_polar] = 0.5 * (pa[:, 1] ** 2 - pa[:, 0] ** 2) * (
            pa[:, 3] - pa[:, 2]
        )
        if np.any(self.cell_area <= 0):
            raise ValueError(f"{np.sum(self.cell_area <= 0)} inverted cells")

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def n_cells(self):
        return len(self.cells)

    def ring(self, radius, rtol=1e-9):
        """Node indices on the circle `radius`, sorted by angle in [0, 2 pi)."""
        r = np.hypot(*self.nodes.T)
        idx = np.flatnonzero(np.abs(r - radius) < rtol * radius)
        th = np.mod(np.arctan2(self.nodes[idx, 1], self.nodes[idx, 0]), 2 * np.pi)
        o = np.argsort(th)
        return idx[o], th[o]

    def quality(self):
        """Minimum and maximum interior corner angle [deg] of general cells."""
        P = self.nodes[self.cells[~self.is_polar]]
        if len(P) == 0:
            return 90.0, 90.0
        a = np.roll(P, 1, 1) - P
        b = np.roll(P, -1, 1) - P
        c = np.einsum("ijk,ijk->ij", a, b) / (
            np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)
        )
        ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
        return float(ang.min()), float(ang.max())


# ---------------------------------------------------------------------------
def radial_nodes(breaks, size):
    """
    Ring radii from breaks[0] to breaks[-1] that contain every break radius.

    Parameters
    ----------
    breaks : sequence of float
        Increasing radii [m] that must be rings (interfaces, extraction
        circles, PML start, outer radius).
    size : callable
        r [m] -> target radial spacing [m].

    Returns
    -------
    ndarray
        Increasing ring radii.
    """
    out = [breaks[0]]
    for r0, r1 in zip(breaks[:-1], breaks[1:]):
        xs = [r0]
        while xs[-1] < r1 - 1e-9 * r1:
            xs.append(xs[-1] + size(xs[-1]))
        n = max(1, len(xs) - 1)
        if n > 1 and (xs[-1] - r1) > 0.5 * (xs[-1] - xs[-2]):
            n -= 1  # last step overshoots by more than half: drop it
        xs = np.asarray(xs[: n + 1])
        if xs[-1] <= r0:
            xs = np.array([r0, r1])
        xs = r0 + (xs - r0) * (r1 - r0) / (xs[-1] - r0)
        out.extend(xs[1:])
    return np.asarray(out)


def _merge(blocks):
    """Merge structured blocks (list of (ni, nj, 2) node arrays, optional polar
    params) into nodes, cells, polar arrays."""
    nodes, cells, polar = [], [], []
    base = 0
    for B, pol in blocks:
        ni, nj = B.shape[:2]
        nodes.append(B.reshape(-1, 2))
        idx = base + np.arange(ni * nj).reshape(ni, nj)
        c = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1)
        cells.append(c.reshape(-1, 4))
        polar.append(
            np.full((c.shape[0] * c.shape[1], 4), np.nan) if pol is None else pol
        )
        base += ni * nj
    nodes = np.concatenate(nodes)
    cells = np.concatenate(cells)
    polar = np.concatenate(polar)
    scale = np.abs(nodes).max()
    key = np.round(nodes / (scale * 1e-10)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    merged = nodes[first]
    cells = inv[cells]
    P = merged[cells]
    sa = np.sum(
        P[..., 0] * np.roll(P[..., 1], -1, 1) - np.roll(P[..., 0], -1, 1) * P[..., 1],
        axis=1,
    )
    if np.any(sa <= 0):
        raise ValueError("block cells must be counter-clockwise")
    return merged, cells, polar


def _default_m_tr(n, core_frac):
    """Transition layers so that the layer thickness on the square's diagonal,
    (1 - sqrt(2) core_frac) r_t / m_tr, is about one square cell."""
    return max(1, int(round((1 - np.sqrt(2) * core_frac) * n / (2 * core_frac))))


def _butterfly(r_t, n, core_frac, m_tr):
    """Square (n x n cells, half side core_frac r_t) + four transition blocks
    (m_tr layers) to the circle r_t whose nodes sit at th = -pi/4 + j pi/(2n)
    (+ multiples of pi/2)."""
    s = core_frac * r_t
    u = np.linspace(-1, 1, n + 1)
    U, V = np.meshgrid(u, u, indexing="ij")
    blocks = [(np.stack([s * U, s * V], -1), None)]
    t = np.linspace(0, 1, m_tr + 1)[:, None, None]
    for rot in range(4):
        th = rot * np.pi / 2
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        side = np.stack([np.full_like(u, s), s * u], -1)
        phi = u * np.pi / 4
        arc = r_t * np.stack([np.cos(phi), np.sin(phi)], -1)
        blocks.append((((1 - t) * side[None] + t * arc[None]) @ R.T, None))
    return blocks


def _polar_block(radii, n_theta, th_offset):
    """Polar rings radii[0..] x full circle, n_theta sectors (periodic: the
    last angular column duplicates the first and is merged)."""
    th = th_offset + 2 * np.pi * np.arange(n_theta + 1) / n_theta
    Rr, Th = np.meshgrid(radii, th, indexing="ij")
    B = np.stack([Rr * np.cos(Th), Rr * np.sin(Th)], -1)
    r0, r1 = Rr[:-1, :-1], Rr[1:, :-1]
    t0, t1 = Th[:-1, :-1], Th[:-1, 1:]
    pol = np.stack([r0, r1, t0, t1], -1).reshape(-1, 4)
    return B, pol


def _smooth(nodes, cells, movable, n_iter):
    """Laplacian (neighbour-average) smoothing of the `movable` nodes."""
    if n_iter <= 0 or not np.any(movable):
        return nodes
    e = np.concatenate([cells[:, [k, (k + 1) % 4]] for k in range(4)])
    e = np.unique(np.sort(e, 1), axis=0)
    n = len(nodes)
    A = sp.csr_matrix((np.ones(2 * len(e)), (e.T.ravel(), e[:, ::-1].T.ravel())), (n, n))
    deg = np.asarray(A.sum(1)).ravel()
    x = nodes.copy()
    for _ in range(n_iter):
        avg = (A @ x) / deg[:, None]
        x[movable] = 0.5 * x[movable] + 0.5 * avg[movable]
    return x


def _tags(mesh_nodes, cells, polar, interfaces):
    """Tag k+1 inside interfaces[k] (innermost first), 0 outside the last."""
    rc = np.where(
        np.isnan(polar[:, 0]),
        np.hypot(*mesh_nodes[cells].mean(1).T),
        0.5 * (polar[:, 0] + polar[:, 1]),
    )
    tags = np.zeros(len(cells), np.int64)
    for k in range(len(interfaces) - 1, -1, -1):
        tags[rc < interfaces[k]] = k + 1
    return tags


def ogrid_mesh(
    radii, n_theta, interfaces=(), core_frac=0.55, m_tr=None, smooth=0,
    polar=True,
):  # fmt: skip
    """
    Single-block O-grid of a disk: butterfly core + polar rings.

    Parameters
    ----------
    radii : array_like
        Ring radii [m] from the core radius r_t = radii[0] to the outer
        radius; interfaces should be among them (see `radial_nodes`).
    n_theta : int
        Number of angular sectors (multiple of 4).
    interfaces : sequence of float
        Material interface radii (innermost first) for the cell tags: tag
        k+1 inside interfaces[k], 0 outside the last.
    core_frac : float
        Half side of the core square / r_t.
    m_tr : int, optional
        Layers between the square and the circle r_t (default from the
        spacing).
    smooth : int
        Laplacian smoothing iterations of the core nodes (inside r_t).
    polar : bool or float
        False: treat the rings as general (straight-sided) quads too, i.e.
        use the nonorthogonal corner Hodge everywhere (for comparisons); a
        radius: only rings beyond it stay polar (e.g. a radial PML).

    Returns
    -------
    QuadMesh
    """
    radii = np.asarray(radii, float)
    if n_theta % 4:
        raise ValueError("n_theta must be a multiple of 4")
    n = n_theta // 4
    r_t = radii[0]
    if m_tr is None:
        m_tr = _default_m_tr(n, core_frac)
    blocks = _butterfly(r_t, n, core_frac, m_tr)
    B, pol = _polar_block(radii, n_theta, -np.pi / 4)
    blocks.append((B, pol))
    nodes, cells, pol_all = _merge(blocks)
    r = np.hypot(*nodes.T)
    nodes = _smooth(nodes, cells, r < r_t * (1 - 1e-9), smooth)
    tags = _tags(nodes, cells, pol_all, list(interfaces))
    if polar is False:
        pol_all = np.full_like(pol_all, np.nan)
    elif polar is not True:  # general (straight-sided) cells for r1 <= polar
        pol_all[pol_all[:, 1] <= polar * (1 + 1e-12)] = np.nan
    return QuadMesh(nodes, cells, tags, pol_all)


def ogrid_cartesian_mesh(
    radii, n_theta, half_square, x_outer, m_tr_out, interfaces=(), core_frac=0.55,
    smooth=0,
):  # fmt: skip
    """
    Multi-block mesh: O-grid (core + polar rings up to radii[-1]), a
    transition block from that circle to the square |x|, |y| <= half_square,
    and a Cartesian outer block.

    Parameters
    ----------
    radii : array_like
        Polar ring radii [m] (core radius first, block-interface circle last).
    n_theta : int
        Angular sectors (multiple of 4); the square sides get n_theta/4
        uniform cells each.
    half_square : float
        Half side of the Cartesian block's inner boundary [m].
    x_outer : array_like
        Increasing node coordinates > half_square [m] of the Cartesian block
        (mirrored to negative x and to y).
    m_tr_out : int
        Layers of the circle-to-square transition block.
    interfaces, core_frac, smooth
        As in `ogrid_mesh`.

    Returns
    -------
    QuadMesh
    """
    radii = np.asarray(radii, float)
    n = n_theta // 4
    blocks = _butterfly(radii[0], n, core_frac, _default_m_tr(n, core_frac))
    B, pol = _polar_block(radii, n_theta, -np.pi / 4)
    blocks.append((B, pol))
    r_b, S = radii[-1], half_square
    u = np.linspace(-1, 1, n + 1)
    t = np.linspace(0, 1, m_tr_out + 1)[:, None, None]
    for rot in range(4):
        th = rot * np.pi / 2
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        arc = r_b * np.stack([np.cos(u * np.pi / 4), np.sin(u * np.pi / 4)], -1)
        side = np.stack([np.full_like(u, S), S * u], -1)
        blocks.append((((1 - t) * arc[None] + t * side[None]) @ R.T, None))
    xs = np.concatenate([-np.asarray(x_outer)[::-1], S * u, np.asarray(x_outer)])
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    Bc = np.stack([X, Y], -1)
    nodes, cells, pol_all = _merge(blocks)
    # Cartesian frame (cells outside the square)
    nx = len(xs)
    idx = np.arange(nx * nx).reshape(nx, nx)
    c = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1)
    cx = 0.5 * (xs[:-1] + xs[1:])
    CX, CY = np.meshgrid(cx, cx, indexing="ij")
    outside = (np.abs(CX) > S) | (np.abs(CY) > S)
    cart_cells = c[outside]
    allnodes = np.concatenate([nodes, Bc.reshape(-1, 2)])
    allcells = np.concatenate([cells, cart_cells + len(nodes)])
    pol_all = np.concatenate([pol_all, np.full((len(cart_cells), 4), np.nan)])
    scale = np.abs(allnodes).max()
    key = np.round(allnodes / (scale * 1e-10)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    nodes = allnodes[first]
    cells = inv[allcells]
    # drop Cartesian nodes not used by any cell (inside the square)
    used = np.zeros(len(nodes), bool)
    used[cells.ravel()] = True
    remap = -np.ones(len(nodes), np.int64)
    remap[used] = np.arange(used.sum())
    nodes, cells = nodes[used], remap[cells]
    r = np.hypot(*nodes.T)
    nodes = _smooth(nodes, cells, r < radii[0] * (1 - 1e-9), smooth)
    tags = _tags(nodes, cells, pol_all, list(interfaces))
    return QuadMesh(nodes, cells, tags, pol_all)


def _arc_to_box(center, r_b, n, box, m_tr):
    """Four transition blocks from the circle r_b around `center` (nodes at
    th = -pi/4 + j pi/(2n)) to the sides of the axis-aligned `box` = (xmin,
    xmax, ymin, ymax) (n uniform cells per side), m_tr layers each."""
    xmin, xmax, ymin, ymax = box
    u = np.linspace(0, 1, n + 1)
    sides = [
        np.stack([np.full_like(u, xmax), ymin + (ymax - ymin) * u], -1),  # right
        np.stack([xmax - (xmax - xmin) * u, np.full_like(u, ymax)], -1),  # top
        np.stack([np.full_like(u, xmin), ymax - (ymax - ymin) * u], -1),  # left
        np.stack([xmin + (xmax - xmin) * u, np.full_like(u, ymin)], -1),  # bottom
    ]
    t = np.linspace(0, 1, m_tr + 1)[:, None, None]
    blocks = []
    for q in range(4):
        th = -np.pi / 4 + q * np.pi / 2 + u * np.pi / 2
        arc = np.asarray(center) + r_b * np.stack([np.cos(th), np.sin(th)], -1)
        blocks.append(((1 - t) * arc[None] + t * sides[q][None], None))
    return blocks


def boxed_cylinders_mesh(centers, a, radii_in, n_theta, half_box, m_tr, x_outer, y_outer,
                         core_frac=0.55):  # fmt: skip
    """
    Multi-block mesh of equal cylinders in a row along x: around each
    cylinder a butterfly core and polar rings up to its surface, then a
    transition block from the circle to a square box of half side
    `half_box`; the boxes are adjacent (they tile one rectangle) and a
    Cartesian block surrounds them (rectilinear, conforming node by node).

    Parameters
    ----------
    centers : sequence of float
        x coordinates of the centres [m]; neighbouring centres must be
        2 half_box apart (adjacent boxes); all at y = 0.
    a : float
        Cylinder radius [m].
    radii_in : array_like
        Ring radii from the core radius to a (inclusive) [m].
    n_theta : int
        Angular sectors (multiple of 8: a node on each box side's midpoint).
    half_box : float
        Box half side [m] (> a).
    m_tr : int
        Layers between the circle and the box.
    x_outer, y_outer : array_like
        Increasing node coordinates of the Cartesian block beyond the boxes'
        rectangle, |x| > max(centers) + half_box and |y| > half_box [m].

    Returns
    -------
    QuadMesh
        Tags: 1 inside the cylinders, 0 outside.
    """
    if n_theta % 8:
        raise ValueError("n_theta must be a multiple of 8")
    n = n_theta // 4
    radii_in = np.asarray(radii_in, float)
    centers = np.sort(np.asarray(centers, float))
    if np.any(np.abs(np.diff(centers) - 2 * half_box) > 1e-9 * half_box):
        raise ValueError("boxes must be adjacent: centre spacing 2 half_box")
    blocks, pcen = [], []
    for cx in centers:
        c = np.array([cx, 0.0])
        for B, _ in _butterfly(radii_in[0], n, core_frac, _default_m_tr(n, core_frac)):
            blocks.append((B + c, None))
            pcen.append(None)
        B, pol = _polar_block(radii_in, n_theta, -np.pi / 4)
        blocks.append((B + c, pol))
        pcen.append(c)
        box = (cx - half_box, cx + half_box, -half_box, half_box)
        for blk in _arc_to_box(c, a, n, box, m_tr):
            blocks.append(blk)
            pcen.append(None)
    # polar centres follow the block order through _merge's concatenation
    cen_rows = []
    for (B, pol), c in zip(blocks, pcen):
        ncell = (B.shape[0] - 1) * (B.shape[1] - 1)
        cen_rows.append(np.tile(c if c is not None else [0.0, 0.0], (ncell, 1)))
    nodes, cells, pol_all = _merge(blocks)
    pc = np.concatenate(cen_rows)
    # Cartesian block outside the boxes' rectangle
    X0, Y0 = centers[-1] + half_box, half_box
    xin = np.unique(np.concatenate([cx + np.linspace(-half_box, half_box, n + 1) for cx in centers]))
    yin = np.linspace(-half_box, half_box, n + 1)
    xo, yo = np.asarray(x_outer), np.asarray(y_outer)
    xs = np.concatenate([-xo[::-1], xin, xo]) if centers[0] == -centers[-1] else None
    if xs is None:
        raise ValueError("centres must be symmetric about x = 0")
    ys = np.concatenate([-yo[::-1], yin, yo])
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nx, ny = len(xs), len(ys)
    idx = np.arange(nx * ny).reshape(nx, ny)
    c4 = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1)
    CX, CY = np.meshgrid(0.5 * (xs[:-1] + xs[1:]), 0.5 * (ys[:-1] + ys[1:]), indexing="ij")
    out = (np.abs(CX) > X0) | (np.abs(CY) > Y0)
    cart = c4[out]
    allnodes = np.concatenate([nodes, np.stack([X.ravel(), Y.ravel()], 1)])
    allcells = np.concatenate([cells, cart + len(nodes)])
    pol_all = np.concatenate([pol_all, np.full((len(cart), 4), np.nan)])
    pc = np.concatenate([pc, np.zeros((len(cart), 2))])
    scale = np.abs(allnodes).max()
    key = np.round(allnodes / (scale * 1e-10)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    nodes, cells = allnodes[first], inv[allcells]
    used = np.zeros(len(nodes), bool)
    used[cells.ravel()] = True
    remap = -np.ones(len(nodes), np.int64)
    remap[used] = np.arange(used.sum())
    nodes, cells = nodes[used], remap[cells]
    # tags: inside a cylinder
    is_pol = ~np.isnan(pol_all[:, 0])
    cen = nodes[cells].mean(1)
    d = np.min(np.abs(cen[:, None, 0] - centers[None, :]) ** 2 + cen[:, None, 1] ** 2, axis=1) ** 0.5
    rmid = 0.5 * (pol_all[:, 0] + pol_all[:, 1])
    inside = np.where(is_pol, rmid < a, d < a)
    mesh = QuadMesh(nodes, cells, inside.astype(np.int64), pol_all, polar_center=pc)
    mesh.cartesian_axes = (xs, ys)
    return mesh
