# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Geometry adapter and quadtree refinement predicates for PIC `PolySlab`s on
`SubgridFIT2D`.

`PolySlabGeometry` adapts a sequence of `wakis.templates.photonics.PolySlab`
(and `Box`, `Cylinder`, `Circle`, `Rect`) objects to the `Geometry` interface
expected by `SubgridFIT2D._sample_materials`:

* `material_id(x, y)` — vectorized painter's-order material lookup with
  axis-aligned rectangle fast paths and bounding-box pre-culling before
  polygon point-in-polygon tests;
* `cut(xm, ym, half_diag)` — vectorized cell-boundary intersection test;
* `eps`, `background_eps`, `dispersive`.

Refinement helpers build callables `refine(level, x0, x1, y0, y1) -> bool`
for `QuadMesh`:

* `refine_boxes([(x0, x1, y0, y1, level), ...])`;
* `refine_near_polygons(polys, distance, level=1, boundary_only=False)`;
* `choose_refinement(coarse_cell, target_boxes, allowed_ratios=(2, 3))` —
  picks `(ratio, max_level)` and builds the `refine` predicate so that a
  target cell size is reached inside each box from a coarse cell size set by
  `cells_per_wavelength`.
"""

from collections import namedtuple

import numpy as np

from scipy.constants import c as C0

from ..templates.photonics import PolySlab, _clean_polygon, _edges_and_normals, n_to_eps
from ..templates.shapes import Box, Cylinder, material_eps, resolve_material
from .shapes import Circle, Rect, _eps_inf, _is_pec, _sigma

__all__ = [
    "PolySlabGeometry",
    "SubgridPlan",
    "refine_boxes",
    "refine_near_polygons",
    "combine_refinements",
    "choose_refinement",
    "skin_depth",
    "refine_skin_depth",
]

SubgridPlan = namedtuple(
    "SubgridPlan", ["ratio", "max_level", "refine", "boxes", "fine_cell"]
)


def _as_axis_aligned_rect(poly_slab, z=None):
    """
    If `poly_slab` is a 4-vertex axis-aligned rectangle without holes or
    sidewall tilt, return `(x0, x1, y0, y1)`; otherwise return `None`.
    """
    if poly_slab.holes or (poly_slab._tan_angle != 0.0 and z is not None):
        return None
    v = poly_slab.vertices
    if len(v) != 4:
        return None
    x0, y0 = v.min(axis=0)
    x1, y1 = v.max(axis=0)
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    for c in corners:
        if not np.any(np.all(np.isclose(v, c, rtol=0.0, atol=1e-13), axis=1)):
            return None
    return float(x0), float(x1), float(y0), float(y1)


def _rect_sdf(x, y, x0, x1, y0, y1):
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    hx, hy = 0.5 * (x1 - x0), 0.5 * (y1 - y0)
    qx, qy = np.abs(x - cx) - hx, np.abs(y - cy) - hy
    out = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    return out + np.minimum(np.maximum(qx, qy), 0.0)


class _PolygonAdapter:
    """Vectorized 2D wrapper around a `PolySlab` or raw `(N, 2)` polygon."""

    def __init__(self, shape, material, z=None):
        self.material = material
        self.z = z
        if isinstance(shape, PolySlab):
            self.slab = shape
        else:
            verts = _clean_polygon(shape, ccw=True)
            self.slab = PolySlab(verts, -1.0, 1.0, material=1.0)
        self.rect = _as_axis_aligned_rect(self.slab, z=z)
        lo, hi = self.slab.bounds
        self.bbox = (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))

    def contains(self, x_flat, y_flat):
        """Boolean mask of points `(x_flat, y_flat)` inside the polygon."""
        if self.rect is not None:
            x0, x1, y0, y1 = self.rect
            return (x_flat >= x0) & (x_flat <= x1) & (y_flat >= y0) & (y_flat <= y1)
        x0, x1, y0, y1 = self.bbox
        in_box = (x_flat >= x0) & (x_flat <= x1) & (y_flat >= y0) & (y_flat <= y1)
        out = np.zeros(x_flat.shape, dtype=bool)
        if not np.any(in_box):
            return out
        pts = np.column_stack([x_flat[in_box], y_flat[in_box]])
        out[in_box] = self.slab._contains_xy(pts, z=self.z)
        return out

    def boundary_distance(self, x_flat, y_flat):
        """Unsigned distance [m] from `(x_flat, y_flat)` to the polygon boundary."""
        if self.rect is not None:
            return np.abs(_rect_sdf(x_flat, y_flat, *self.rect))
        d_xy, _, _ = self.slab._dist_and_normal_xy(x_flat, y_flat)
        return d_xy

    def cut(self, xm, ym, half_diag):
        """Boolean mask of cells `(xm, ym, half_diag)` crossed by the boundary."""
        tol = np.broadcast_to(half_diag * (1.0 + 1e-9), xm.shape)
        if self.rect is not None:
            return np.abs(_rect_sdf(xm, ym, *self.rect)) <= tol
        x0, x1, y0, y1 = self.bbox
        near_box = (
            (xm + tol >= x0)
            & (xm - tol <= x1)
            & (ym + tol >= y0)
            & (ym - tol <= y1)
        )
        out = np.zeros(xm.shape, dtype=bool)
        if not np.any(near_box):
            return out
        d_xy, _, _ = self.slab._dist_and_normal_xy(
            xm[near_box], ym[near_box]
        )
        out[near_box] = d_xy <= tol[near_box]
        return out


class _AnalyticAdapter:
    """Wrapper around `Circle`, `Rect`, `Box`, or z-oriented `Cylinder`."""

    def __init__(self, shape, material):
        self.material = material
        if isinstance(shape, Box):
            lo, hi = shape.bounds
            self.shape = Rect(lo[0], hi[0], lo[1], hi[1], material)
            self.bbox = (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))
        elif isinstance(shape, Cylinder):
            if shape.axis != 2:
                raise ValueError("Only z-axis Cylinders are supported in 2D.")
            c, r = shape.center[:2], shape.radius
            self.shape = Circle(c, r, material)
            self.bbox = (
                float(c[0] - r),
                float(c[0] + r),
                float(c[1] - r),
                float(c[1] + r),
            )
        elif isinstance(shape, Rect):
            self.shape = shape
            self.bbox = shape.b
        elif isinstance(shape, Circle):
            self.shape = shape
            c, r = shape.center, shape.radius
            self.bbox = (
                float(c[0] - r),
                float(c[0] + r),
                float(c[1] - r),
                float(c[1] + r),
            )
        else:
            self.shape = shape
            self.bbox = (-np.inf, np.inf, -np.inf, np.inf)

    def contains(self, x_flat, y_flat):
        x0, x1, y0, y1 = self.bbox
        in_box = (x_flat >= x0) & (x_flat <= x1) & (y_flat >= y0) & (y_flat <= y1)
        out = np.zeros(x_flat.shape, dtype=bool)
        if not np.any(in_box):
            return out
        idx = np.flatnonzero(in_box)
        out[idx] = self.shape.sdf(x_flat[idx], y_flat[idx]) < 0.0
        return out

    def boundary_distance(self, x_flat, y_flat):
        return np.abs(self.shape.sdf(x_flat, y_flat))

    def cut(self, xm, ym, half_diag):
        tol = half_diag * (1.0 + 1e-9)
        x0, x1, y0, y1 = self.bbox
        near = (
            (xm + tol >= x0)
            & (xm - tol <= x1)
            & (ym + tol >= y0)
            & (ym - tol <= y1)
        )
        out = np.zeros(xm.shape, dtype=bool)
        if not np.any(near):
            return out
        idx = np.flatnonzero(near)
        out.ravel()[idx] = (
            np.abs(self.shape.sdf(xm.ravel()[idx], ym.ravel()[idx]))
            <= tol.ravel()[idx]
        )
        return out


def _resolve_2d_material(mat, eim_map=None, frequency=None):
    """Normalize a shape material to float `eps_r`, `'pec'`, or `DispersiveMaterial`."""
    if _is_pec(mat):
        return "pec"
    if isinstance(mat, str):
        try:
            mat = resolve_material(mat)
        except KeyError:
            mat = n_to_eps(mat)
    else:
        mat = resolve_material(mat)
    if _is_pec(mat):
        return "pec"
    if eim_map is not None:
        eps_core_3d, eps_slab_2d = eim_map
        f_ref = 193.4e12 if frequency is None else float(frequency)
        if np.isclose(material_eps(mat, f_ref), eps_core_3d, rtol=1e-2):
            return float(eps_slab_2d)
    return mat


class PolySlabGeometry:
    """
    `Geometry`-compatible adapter built from kit `PolySlab`s for `SubgridFIT2D`.

    Parameters
    ----------
    structures : sequence of PolySlab, Box, Cylinder, Circle or Rect
        Shapes painted in order (later shapes override earlier ones).
    background_eps : float or str, optional
        Background relative permittivity (or `PIC_MATERIALS` key, converted via
        `n_to_eps`). Default 1.0 (vacuum).
    z : float, optional
        Midplane z-coordinate [m] used to slice 3D shapes (`Box`, `PolySlab`).
        When given, shapes whose `[z_min, z_max]` does not contain `z` are
        skipped, and tilted sidewalls (`sidewall_angle_deg != 0`) are evaluated
        at `z`.
    eim_map : tuple (eps_core_3d, eps_slab_2d), optional
        2D effective-index mapping: replaces any structure material matching
        `eps_core_3d` (within 1 %) by `eps_slab_2d`.
    frequency : float, optional
        Reference frequency [Hz] used when comparing dispersive materials in
        `eim_map`.

    Attributes
    ----------
    eps : ndarray (n_materials + 1,)
        Instantaneous relative permittivity per material id (`0` = background).
    background_eps : float
        Background relative permittivity.
    dispersive : list of (int, DispersiveMaterial)
        Dispersive materials and their 1-based material ids.
    """

    def __init__(
        self,
        structures=(),
        background_eps=1.0,
        z=None,
        eim_map=None,
        frequency=None,
    ):
        if isinstance(background_eps, str):
            background_eps = n_to_eps(background_eps)
        self.background_eps = float(background_eps)
        self.z = None if z is None else float(z)
        self.adapters = []
        mats = []
        self.shape_mat = []

        for s in structures:
            if self.z is not None and hasattr(s, "bounds"):
                lo, hi = s.bounds
                if len(lo) >= 3 and (self.z < lo[2] - 1e-15 or self.z > hi[2] + 1e-15):
                    continue
            raw_mat = getattr(s, "material", 1.0)
            mat = _resolve_2d_material(raw_mat, eim_map=eim_map, frequency=frequency)
            if isinstance(s, PolySlab):
                adapter = _PolygonAdapter(s, mat, z=self.z)
            elif isinstance(s, (Box, Cylinder, Circle, Rect)):
                adapter = _AnalyticAdapter(s, mat)
            elif hasattr(s, "sdf"):
                adapter = _AnalyticAdapter(s, mat)
            else:
                adapter = _PolygonAdapter(s, mat, z=self.z)
            self.adapters.append(adapter)

            for k, m in enumerate(mats):
                if (
                    m is mat
                    or (_is_pec(m) and _is_pec(mat))
                    or (
                        not _is_pec(m)
                        and not _is_pec(mat)
                        and not hasattr(m, "poles")
                        and not hasattr(mat, "poles")
                        and _sigma(m) == _sigma(mat)
                        and np.isclose(
                            _eps_inf(m), _eps_inf(mat), rtol=0.0, atol=1e-14
                        )
                    )
                ):
                    self.shape_mat.append(k + 1)
                    break
            else:
                mats.append(mat)
                self.shape_mat.append(len(mats))

        self.materials = mats
        self.eps = np.array([self.background_eps] + [_eps_inf(m) for m in mats], float)
        self.sigma = np.array([0.0] + [_sigma(m) for m in mats], float)
        self.pec_ids = [k + 1 for k, m in enumerate(mats) if _is_pec(m)]
        self.dispersive = [
            (k + 1, m)
            for k, m in enumerate(mats)
            if hasattr(m, "poles") and len(m.poles) > 0
        ]

    def material_id(self, x, y):
        """Material id at broadcast coordinates `(x, y)` [m] (painter's order)."""
        X, Y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        xf, yf = X.ravel(), Y.ravel()
        mid = np.zeros(xf.size, dtype=int)
        for adapter, k in zip(self.adapters, self.shape_mat):
            mid[adapter.contains(xf, yf)] = k
        return mid.reshape(X.shape)

    def pec_at_points(self, x, y, tol=1e-12):
        """Boolean mask of points `(x, y)` inside or on the boundary of a PEC shape."""
        pec_set = set(self.pec_ids)
        if not pec_set:
            return np.zeros(np.shape(x), bool)
        X, Y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        xf, yf = X.ravel(), Y.ravel()
        out = np.zeros(xf.size, dtype=bool)
        for adapter, k in zip(self.adapters, self.shape_mat):
            hit = adapter.contains(xf, yf) | (adapter.boundary_distance(xf, yf) <= tol)
            if k in pec_set:
                out |= hit
            else:
                out &= ~adapter.contains(xf, yf)
        return out.reshape(X.shape)

    def cut(self, xm, ym, half_diag):
        """Boolean mask of cells `(xm, ym, half_diag)` [m] crossed by an interface."""
        XM, YM, HD = np.broadcast_arrays(
            np.asarray(xm, float),
            np.asarray(ym, float),
            np.asarray(half_diag, float),
        )
        out = np.zeros(XM.shape, dtype=bool)
        for adapter in self.adapters:
            out |= adapter.cut(XM, YM, HD)
        return out

    def cell_fill(self, x0, x1, y0, y1, n_sub=8, mat_id=1):
        """
        Sub-pixel area fill fraction of `mat_id` in cells `[x0, x1] x [y0, y1]`
        using an `n_sub x n_sub` midpoint grid (matching `PolySlab.fill`).
        """
        x0, x1, y0, y1 = (
            np.asarray(x0, float),
            np.asarray(x1, float),
            np.asarray(y0, float),
            np.asarray(y1, float),
        )
        dx, dy = x1 - x0, y1 - y0
        xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        is_cut = self.cut(xm, ym, 0.5 * np.hypot(dx, dy))
        fill = (self.material_id(xm, ym) == mat_id).astype(float)
        idx = np.flatnonzero(is_cut)
        if idx.size == 0:
            return fill
        t = (np.arange(n_sub, dtype=float) + 0.5) / n_sub
        tx, ty = np.meshgrid(t, t, indexing="ij")
        tx_f, ty_f = tx.ravel()[None, :], ty.ravel()[None, :]
        k_sub = n_sub * n_sub
        chunk = max(1, 200_000 // k_sub)
        fill_flat = fill.ravel()
        x0_f, dx_f = x0.ravel(), dx.ravel()
        y0_f, dy_f = y0.ravel(), dy.ravel()
        for start in range(0, idx.size, chunk):
            sel = idx[start : start + chunk]
            px = x0_f[sel, None] + dx_f[sel, None] * tx_f
            py = y0_f[sel, None] + dy_f[sel, None] * ty_f
            mid = self.material_id(px, py)
            fill_flat[sel] = (mid == mat_id).mean(axis=1)
        return fill_flat.reshape(x0.shape)


# ---------------------------------------------------------- refinement helpers
def _parse_box_spec(b):
    """
    Normalize a box specification to `(x0, x1, y0, y1, level_or_none)`:
    * `(x0, x1, y0, y1)` -> `(x0, x1, y0, y1, None)`
    * `(x0, x1, y0, y1, level)` -> `(x0, x1, y0, y1, int(level))`
    * `(lo, hi)` or `(lo, hi, level)` with sequence `lo, hi` ->
      `(lo[0], hi[0], lo[1], hi[1], level)`
    """
    if len(b) == 2 and np.ndim(b[0]) > 0:
        lo, hi = b
        return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), None
    if len(b) == 3 and np.ndim(b[0]) > 0:
        lo, hi, lev = b
        return (
            float(lo[0]),
            float(hi[0]),
            float(lo[1]),
            float(hi[1]),
            None if lev is None else int(lev),
        )
    if len(b) == 4:
        x0, x1, y0, y1 = b
        return float(x0), float(x1), float(y0), float(y1), None
    if len(b) == 5:
        x0, x1, y0, y1, lev = b
        return (
            float(x0),
            float(x1),
            float(y0),
            float(y1),
            None if lev is None else int(lev),
        )
    raise ValueError(
        f"Expected box (x0, x1, y0, y1[, level]) or (lo, hi[, level]), got {b!r}."
    )


def refine_boxes(boxes):
    """
    Build a `QuadMesh(refine=...)` predicate from axis-aligned refinement boxes.

    Parameters
    ----------
    boxes : sequence of tuple
        Each entry is `(x0, x1, y0, y1, level)` or `(x0, x1, y0, y1)` [m].
        Cells overlapping `[x0, x1] x [y0, y1]` are split while
        `current_level < level` (or at all levels up to `mesh.max_level` when
        `level` is omitted).

    Returns
    -------
    callable
        `refine(level, x0, x1, y0, y1) -> ndarray of bool`.
    """
    parsed = [_parse_box_spec(b) for b in boxes]

    def refine(level, x0, x1, y0, y1):
        x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
        y0, y1 = np.asarray(y0, float), np.asarray(y1, float)
        tol_x = 1e-9 * (x1 - x0)
        tol_y = 1e-9 * (y1 - y0)
        out = np.zeros(x0.shape, dtype=bool)
        for bx0, bx1, by0, by1, target_lev in parsed:
            if target_lev is not None and level >= target_lev:
                continue
            out |= (
                (x1 > bx0 + tol_x)
                & (x0 < bx1 - tol_x)
                & (y1 > by0 + tol_y)
                & (y0 < by1 - tol_y)
            )
        return out

    return refine


def refine_near_polygons(polys, distance, level=1, boundary_only=False):
    """
    Build a `QuadMesh(refine=...)` predicate refining cells near `polys`.

    Parameters
    ----------
    polys : PolySlab, shape, (N, 2) array, or sequence thereof
        Polygons or 2D shapes to refine around.
    distance : float or sequence of float
        Refinement distance [m] from the polygon boundary. A scalar applies to
        all levels `< level`; a sequence `[d_0, d_1, ...]` sets the distance
        for each level `0, 1, ...` (`level` then defaults to `len(distance)`).
    level : int, optional
        Target refinement level when `distance` is a scalar. Default 1.
    boundary_only : bool, optional
        If False (default), refine both the polygon interior and cells within
        `distance` of its boundary. If True, refine only cells whose boundary
        distance is within `distance + half_diag`.

    Returns
    -------
    callable
        `refine(level, x0, x1, y0, y1) -> ndarray of bool`.
    """
    if isinstance(polys, (PolySlab, Box, Cylinder, Circle, Rect)) or (
        isinstance(polys, np.ndarray) and polys.ndim == 2
    ):
        polys = [polys]

    adapters = []
    for p in polys:
        if isinstance(p, PolySlab) or (isinstance(p, np.ndarray) and p.ndim == 2):
            adapters.append(_PolygonAdapter(p, 1.0))
        else:
            adapters.append(_AnalyticAdapter(p, 1.0))

    if np.ndim(distance) > 0:
        dists = [float(d) for d in distance]
        max_lev = len(dists)
    else:
        max_lev = int(level)
        dists = [float(distance)] * max_lev

    def refine(lev, x0, x1, y0, y1):
        x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
        y0, y1 = np.asarray(y0, float), np.asarray(y1, float)
        if lev >= max_lev:
            return np.zeros(x0.shape, dtype=bool)
        d_thresh = dists[lev]
        xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        hd = 0.5 * np.hypot(x1 - x0, y1 - y0)
        reach = d_thresh + hd * (1.0 + 1e-9)
        xf, yf, rf = xm.ravel(), ym.ravel(), reach.ravel()
        out = np.zeros(xf.size, dtype=bool)
        for ad in adapters:
            bx0, bx1, by0, by1 = ad.bbox
            cand = (
                (xf + rf >= bx0)
                & (xf - rf <= bx1)
                & (yf + rf >= by0)
                & (yf - rf <= by1)
            )
            if not np.any(cand):
                continue
            idx = np.flatnonzero(cand)
            hit = ad.boundary_distance(xf[idx], yf[idx]) <= rf[idx]
            if not boundary_only:
                hit = hit | ad.contains(xf[idx], yf[idx])
            out[idx] |= hit
        return out.reshape(x0.shape)

    return refine


def combine_refinements(*predicates):
    """Logical OR of multiple `QuadMesh(refine=...)` callables (ignoring `None`)."""
    preds = [p for p in predicates if p is not None]
    if not preds:
        return None
    if len(preds) == 1:
        return preds[0]

    def refine(level, x0, x1, y0, y1):
        out = np.zeros(np.shape(x0), dtype=bool)
        for p in preds:
            out |= np.asarray(p(level, x0, x1, y0, y1), dtype=bool)
        return out

    return refine


def _parse_target_box(entry, clip_box=None):
    """
    Normalize a target-cell box specification to `(x0, x1, y0, y1, h_target)`:
    * `(x0, x1, y0, y1, h_target)`
    * `((x0, y0[, z0]), (x1, y1[, z1]), h_target)` (`PICSimulation` `mesh_overrides`)
    """
    if len(entry) == 3 and np.ndim(entry[0]) > 0:
        lo, hi, h = entry
        x0, y0 = float(lo[0]), float(lo[1])
        x1, y1 = float(hi[0]), float(hi[1])
        h_arr = np.atleast_1d(np.asarray(h, dtype=float))
        h_target = float(np.min(h_arr[:2])) if h_arr.size >= 2 else float(h_arr[0])
    elif len(entry) == 5:
        x0, x1, y0, y1, h = entry
        x0, x1, y0, y1 = float(x0), float(x1), float(y0), float(y1)
        h_target = float(np.min(np.asarray(h, dtype=float)))
    else:
        raise ValueError(
            "Expected (x0, x1, y0, y1, max_cell) or (lo, hi, max_cell), "
            f"got {entry!r}."
        )
    if clip_box is not None:
        cx0, cx1, cy0, cy1 = clip_box
        x0, x1 = max(x0, float(cx0)), min(x1, float(cx1))
        y0, y1 = max(y0, float(cy0)), min(y1, float(cy1))
    return x0, x1, y0, y1, h_target


def choose_refinement(
    coarse_cell, target_boxes, allowed_ratios=(2, 3), clip_box=None
):
    """
    Choose `(ratio, max_level)` and build a `refine` callable so that a target
    cell size is reached inside each box from a coarse cell size.

    Parameters
    ----------
    coarse_cell : float or sequence of float
        Coarse background/waveguide cell size [m] (e.g. from
        `cells_per_wavelength`). If a `(dx, dy)` pair is given, `max(dx, dy)`
        is used as the conservative coarse reference.
    target_boxes : float or sequence of tuple
        Either:
        * a sequence of `(x0, x1, y0, y1, max_cell)` tuples [m];
        * a sequence of `PICSimulation`-style `mesh_overrides`
          `((x0, y0, z0), (x1, y1, z1), max_cell)` [m];
        * a single target cell size `max_cell` [m] (used with `clip_box` or the
          whole domain `(-inf, inf, -inf, inf)`).
    allowed_ratios : sequence of int, optional
        Candidate refinement ratios per level. Default `(2, 3)`.
    clip_box : tuple (xmin, xmax, ymin, ymax), optional
        Interior domain bounds [m] used to clip infinite or PML-overlapping
        box extents.

    Returns
    -------
    SubgridPlan
        Named tuple `(ratio, max_level, refine, boxes, fine_cell)` where
        `boxes` is the list of `(x0, x1, y0, y1, level)` tuples and `refine` is
        the predicate for `QuadMesh`.
    """
    h_c = float(np.max(np.asarray(coarse_cell, dtype=float)))
    if np.isscalar(target_boxes):
        bx = (-np.inf, np.inf, -np.inf, np.inf) if clip_box is None else clip_box
        parsed = [(*bx, float(target_boxes))]
    else:
        parsed = [
            _parse_target_box(b, clip_box=clip_box) for b in target_boxes
        ]
        parsed = [b for b in parsed if b[1] > b[0] and b[3] > b[2]]

    if not parsed:
        return SubgridPlan(
            ratio=int(allowed_ratios[0]),
            max_level=0,
            refine=None,
            boxes=[],
            fine_cell=h_c,
        )

    h_min = min(b[4] for b in parsed)
    factor_max = h_c / h_min
    if factor_max <= 1.0 + 1e-9:
        return SubgridPlan(
            ratio=int(allowed_ratios[0]),
            max_level=0,
            refine=None,
            boxes=[],
            fine_cell=h_c,
        )

    # Pick (ratio, max_level) with r**max_level >= factor_max that minimizes
    # the worst-case oversampling r**max_level (and on ties uses fewer levels)
    best = None
    for r in allowed_ratios:
        r = int(r)
        if r < 2:
            continue
        lev = max(1, int(np.ceil(np.log(factor_max) / np.log(r) - 1e-9)))
        total_ref = r**lev
        key = (total_ref, lev, r)
        if best is None or key < best[0]:
            best = (key, r, lev)
    _, ratio, max_level = best

    boxes_with_levels = []
    for x0, x1, y0, y1, h_t in parsed:
        fac = h_c / h_t
        if fac <= 1.0 + 1e-9:
            continue
        lev = int(np.clip(np.ceil(np.log(fac) / np.log(ratio) - 1e-9), 1, max_level))
        boxes_with_levels.append((x0, x1, y0, y1, lev))

    refine_fn = refine_boxes(boxes_with_levels) if boxes_with_levels else None
    fine_cell = h_c / (ratio**max_level)
    return SubgridPlan(
        ratio=ratio,
        max_level=max_level if boxes_with_levels else 0,
        refine=refine_fn,
        boxes=boxes_with_levels,
        fine_cell=fine_cell,
    )


def skin_depth(material, frequency):
    """
    Optical / RF field skin depth `delta = c / (omega * |Im sqrt(eps(f))|)` [m].

    Parameters
    ----------
    material : str, complex, float, or DispersiveMaterial
        Metal or lossy material (or its complex relative permittivity `eps_r`).
    frequency : float or array_like
        Frequency [Hz].

    Returns
    -------
    float or ndarray
        Field penetration depth `delta` [m] (`1 / e` amplitude decay length).
    """
    f = np.asarray(frequency, dtype=float)
    if isinstance(material, str):
        material = resolve_material(material)
    if hasattr(material, "epsilon"):
        eps = material.epsilon(f)
    else:
        eps = np.asarray(material, dtype=complex)
    kappa = np.abs(np.imag(np.sqrt(np.asarray(eps, dtype=complex))))
    omega = 2.0 * np.pi * f
    delta = C0 / (omega * kappa)
    return float(delta) if np.ndim(frequency) == 0 else delta


def refine_skin_depth(
    coarse_cell,
    interface_box,
    material,
    frequency,
    cells_per_skin_depth=4.0,
    extra_boxes=(),
    allowed_ratios=(2, 3),
    clip_box=None,
):
    """
    Build a `SubgridPlan` resolving the skin depth of `material` at `frequency`.

    Parameters
    ----------
    coarse_cell : float or sequence of float
        Coarse background cell size [m].
    interface_box : tuple (x0, x1, y0, y1) or sequence of tuples
        Patch box(es) `[x0, x1] x [y0, y1]` [m] around the metal interface
        (and optional emitter near-field region) to refine to
        `delta / cells_per_skin_depth`.
    material : str, complex, or DispersiveMaterial
        Metal material or complex permittivity at `frequency`.
    frequency : float or array_like
        Frequency [Hz] (if an array is passed, the minimum skin depth over the
        band is resolved).
    cells_per_skin_depth : float, optional
        Target fine cells per skin depth `delta / h_fine`. Default 4.0.
    extra_boxes : sequence of tuple, optional
        Additional `(x0, x1, y0, y1, max_cell)` target boxes [m].
    allowed_ratios : sequence of int, optional
        Allowed refinement ratios per level. Default `(2, 3)`.
    clip_box : tuple (xmin, xmax, ymin, ymax), optional
        Interior domain bounds [m] used to clip boxes.

    Returns
    -------
    SubgridPlan
        Refinement plan with `fine_cell <= delta / cells_per_skin_depth`.
    """
    delta = float(np.min(skin_depth(material, frequency)))
    h_target = delta / float(cells_per_skin_depth)
    if (
        isinstance(interface_box, (tuple, list))
        and len(interface_box) == 4
        and np.isscalar(interface_box[0])
    ):
        ib_list = [interface_box]
    else:
        ib_list = list(interface_box)
    target_boxes = [(*b[:4], h_target) for b in ib_list] + list(extra_boxes)
    return choose_refinement(
        coarse_cell=coarse_cell,
        target_boxes=target_boxes,
        allowed_ratios=allowed_ratios,
        clip_box=clip_box,
    )

