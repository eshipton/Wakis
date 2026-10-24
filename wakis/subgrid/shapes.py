# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
2D shapes and material lookup for `SubgridFIT2D` (sub-pixel averaging).

Each shape has a signed distance (negative inside) used to find the cells it
cuts, and a point-membership test used to sample those cells. Shapes are
painted in order: later shapes override earlier ones. The background
(material id 0) is a lossless dielectric, vacuum by default.
"""

import numpy as np

__all__ = ["Circle", "Rect", "Geometry"]


class Circle:
    """Disk of radius `radius` [m] centred at `center` [m]."""

    def __init__(self, center, radius, material):
        self.center = np.asarray(center, float)
        self.radius = float(radius)
        self.material = material

    def sdf(self, x, y):
        return np.hypot(x - self.center[0], y - self.center[1]) - self.radius


class Rect:
    """Axis-aligned rectangle [x0, x1] x [y0, y1] [m]."""

    def __init__(self, x0, x1, y0, y1, material):
        self.b = (float(x0), float(x1), float(y0), float(y1))
        self.material = material

    def sdf(self, x, y):
        x0, x1, y0, y1 = self.b
        cx, cy, hx, hy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        qx, qy = np.abs(x - cx) - hx, np.abs(y - cy) - hy
        out = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
        return out + np.minimum(np.maximum(qx, qy), 0)


class Geometry:
    """
    Ordered list of shapes with their materials.

    A material is a relative permittivity (float), or a
    `wakis.dispersive.DispersiveMaterial` (its ``eps_inf`` is the
    instantaneous permittivity; its poles become ADE updates).

    Attributes
    ----------
    eps : ndarray (n_materials + 1,)
        Instantaneous relative permittivity per material id (0 = the
        lossless background, `background_eps`, default vacuum).
    dispersive : list of (id, DispersiveMaterial)
    """

    def __init__(self, shapes=(), background_eps=1.0):
        self.shapes = list(shapes)
        self.background_eps = float(background_eps)
        mats = []
        self.shape_mat = []
        for s in self.shapes:
            for k, m in enumerate(mats):
                if m is s.material or (
                    _is_pec(m) and _is_pec(s.material)
                ):
                    self.shape_mat.append(k + 1)
                    break
            else:
                mats.append(s.material)
                self.shape_mat.append(len(mats))
        self.materials = mats
        self.eps = np.array([self.background_eps] + [_eps_inf(m) for m in mats])
        self.sigma = np.array([0.0] + [_sigma(m) for m in mats])
        self.pec_ids = [k + 1 for k, m in enumerate(mats) if _is_pec(m)]
        self.dispersive = [
            (k + 1, m)
            for k, m in enumerate(mats)
            if hasattr(m, "poles") and len(m.poles) > 0
        ]

    def material_id(self, x, y):
        """Material id at points (painter's order)."""
        mid = np.zeros(np.shape(x), int)
        for s, k in zip(self.shapes, self.shape_mat):
            mid[s.sdf(x, y) < 0] = k
        return mid

    def pec_at_points(self, x, y, tol=1e-12):
        """Boolean mask of points `(x, y)` inside or on the boundary of a PEC shape."""
        pec_set = set(self.pec_ids)
        if not pec_set:
            return np.zeros(np.shape(x), bool)
        out = np.zeros(np.shape(x), bool)
        for s, k in zip(self.shapes, self.shape_mat):
            if k in pec_set:
                out |= s.sdf(x, y) <= tol
            else:
                out &= ~(s.sdf(x, y) < -tol)
        return out

    def cut(self, xm, ym, half_diag):
        """Cells (centre, half diagonal) that an interface may cross."""
        out = np.zeros(np.shape(xm), bool)
        for s in self.shapes:
            out |= np.abs(s.sdf(xm, ym)) <= half_diag * (1 + 1e-9)
        return out


def _is_pec(m):
    return isinstance(m, str) and m.strip().lower() == "pec"


def _eps_inf(m):
    if _is_pec(m):
        return 1.0
    if hasattr(m, "eps_inf"):
        return float(m.eps_inf)
    if isinstance(m, (tuple, list)) and len(m) >= 1:
        return float(m[0])
    return float(m)


def _sigma(m):
    if _is_pec(m):
        return 0.0
    if hasattr(m, "sigma"):
        return float(m.sigma)
    if isinstance(m, (tuple, list)) and len(m) == 2:
        return float(m[1])
    if isinstance(m, (tuple, list)) and len(m) >= 3:
        return float(m[2])
    return 0.0
