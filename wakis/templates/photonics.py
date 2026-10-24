# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

# SPDX-License-Identifier: Apache-2.0

"""
Photonic-integrated-circuit (PIC) geometry primitives, subpixel-smoothed
polygonal slabs, effective-index 2D/3D simulation wrapper, S-parameter port
extraction, and plotting helpers.

Materials in `PIC_MATERIALS` provide non-dispersive refractive indices at
lambda = 1550 nm. Over 1500-1600 nm silicon changes by ~0.8 % and SiO2 by
~0.1 %; for broadband dispersive time-domain runs use the `'silicon'`
Sellmeier ADE model in `wakis.templates.shapes.resolve_material`.

References
----------
.. [1] H. H. Li, "Refractive index of silicon and germanium and its wavelength
   and temperature derivatives," J. Phys. Chem. Ref. Data 9, 561-658 (1980),
   DOI: 10.1063/1.555624.
.. [2] I. H. Malitson, "Interspecimen comparison of the refractive index of
   fused silica," J. Opt. Soc. Am. 55, 1205-1209 (1965),
   DOI: 10.1364/JOSA.55.001205.
.. [3] K. Luke et al., "Broadband mid-infrared frequency comb generation in a
   Si3N4 microresonator," Opt. Lett. 40, 4823-4826 (2015),
   DOI: 10.1364/OL.40.004823.
"""

import copy
from pathlib import Path as FilePath

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch, Polygon as MplPolygon
from matplotlib.path import Path as MplPath
from scipy.constants import c as c_light, epsilon_0
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from ..field_monitors import DFTMonitor, FluxMonitor
from ..mode_solver import ModeMonitor, ModeSource, solve_modes
from . import meshing
from .formulas import slab_waveguide_neff
from .shapes import Box, material_eps, resolve_material
from .simulation import Simulation

PIC_MATERIALS = {
    "Si": 3.476,
    "SiO2": 1.444,
    "Si3N4": 1.996,
    "air": 1.0,
}


def n_to_eps(n):
    """Convert refractive index (or `PIC_MATERIALS` key) to relative permittivity `n**2`."""
    if isinstance(n, str):
        if n in PIC_MATERIALS:
            n = PIC_MATERIALS[n]
        else:
            return resolve_material(n)
    return np.asarray(n, dtype=float) ** 2 if np.ndim(n) > 0 else float(n) ** 2


def _clean_polygon(vertices, ccw=True):
    """Return an (N, 2) float array oriented counter-clockwise without duplicate closing point."""
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3:
        raise ValueError("Polygon vertices must have shape (N, 2) with N >= 3.")
    if np.allclose(v[0], v[-1], atol=1e-15):
        v = v[:-1]
    if len(v) < 3:
        raise ValueError("Polygon needs at least 3 distinct vertices.")
    area2 = np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1])
    if (area2 < 0 and ccw) or (area2 > 0 and not ccw):
        v = v[::-1]
    return np.ascontiguousarray(v)


def _edges_and_normals(poly, outward_sign=1.0):
    """Segment start points, vectors, squared lengths, and unit outward normals for a CCW polygon."""
    p0 = poly
    p1 = np.roll(poly, -1, axis=0)
    seg = p1 - p0
    len2 = np.sum(seg * seg, axis=1)
    length = np.sqrt(np.maximum(len2, 1e-30))
    tx, ty = seg[:, 0] / length, seg[:, 1] / length
    normals = np.column_stack([outward_sign * ty, -outward_sign * tx])
    return p0, seg, len2, normals


class PolySlab:
    """
    Planar polygon in the x-y plane extruded along z from `z_min` to `z_max`.

    Parameters
    ----------
    vertices : array_like, shape (N, 2)
        Outer polygon vertices `(x, y)` [m]. Oriented counter-clockwise.
    z_min, z_max : float
        Bottom and top z coordinates [m] (`z_min < z_max`).
    material : float, str or DispersiveMaterial
        Relative permittivity `eps_r`, key of `PIC_MATERIALS`, `'pec'`, or
        key of `wakis.materials.dispersive_lib`.
    holes : sequence of array_like, optional
        Interior hole polygons `(M_i, 2)` [m] subtracted from the slab.
    sidewall_angle_deg : float, optional
        Sidewall tilt angle [deg] from vertical; offsets vertices inward by
        `(z - z_min) * tan(angle)` using a miter-free vertex-normal offset.
    """

    def __init__(
        self, vertices, z_min, z_max, material, holes=(), sidewall_angle_deg=0.0
    ):
        if z_max <= z_min:
            raise ValueError("Require z_min < z_max.")
        self.vertices = _clean_polygon(vertices, ccw=True)
        self.holes = [_clean_polygon(h, ccw=True) for h in holes]
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        if isinstance(material, str) and material in PIC_MATERIALS:
            material = n_to_eps(PIC_MATERIALS[material])
        self.material = resolve_material(material)
        self.sidewall_angle_deg = float(sidewall_angle_deg)
        self._tan_angle = float(np.tan(np.deg2rad(self.sidewall_angle_deg)))

        self._outer_path = MplPath(self.vertices)
        self._hole_paths = [MplPath(h) for h in self.holes]

        p0, seg, len2, nrm = _edges_and_normals(self.vertices, outward_sign=1.0)
        p0_list, seg_list, len2_list, n_list = [p0], [seg], [len2], [nrm]
        vn = 0.5 * (nrm + np.roll(nrm, 1, axis=0))
        self._outer_vertex_normals = vn / np.maximum(
            np.linalg.norm(vn, axis=1, keepdims=True), 1e-12
        )
        for h in self.holes:
            hp0, hseg, hlen2, hnrm = _edges_and_normals(h, outward_sign=-1.0)
            p0_list.append(hp0)
            seg_list.append(hseg)
            len2_list.append(hlen2)
            n_list.append(hnrm)

        self._edge_p0 = np.vstack(p0_list)
        self._edge_seg = np.vstack(seg_list)
        self._edge_len2 = np.concatenate(len2_list)
        self._edge_normals = np.vstack(n_list)

        vmin, vmax = self.vertices.min(axis=0), self.vertices.max(axis=0)
        self._lo = np.array([vmin[0], vmin[1], self.z_min], dtype=float)
        self._hi = np.array([vmax[0], vmax[1], self.z_max], dtype=float)

    @property
    def bounds(self):
        """`(lo, hi)` bounding-box corners [m]."""
        return self._lo.copy(), self._hi.copy()

    @property
    def center(self):
        """Bounding-box center `(x, y, z)` [m]."""
        return 0.5 * (self._lo + self._hi)

    @property
    def size(self):
        """Bounding-box extent `(sx, sy, sz)` [m]."""
        return self._hi - self._lo

    def faces(self, axis):
        """Mesh-snapping coordinates along `axis` [m]: `[]` for x/y, `[z_min, z_max]` for z."""
        return [] if axis in (0, 1) else [self.z_min, self.z_max]

    def _contains_xy(self, pts, z=None):
        if self._tan_angle != 0.0 and z is not None:
            offset = float(np.mean(z) - self.z_min) * self._tan_angle
            outer_path = MplPath(self.vertices - offset * self._outer_vertex_normals)
        else:
            outer_path = self._outer_path
        inside = outer_path.contains_points(pts)
        for h_path in self._hole_paths:
            if np.any(inside):
                inside[inside] &= ~h_path.contains_points(pts[inside])
        return inside

    def _dist_and_normal_xy(self, x_flat, y_flat, chunk_size=8192):
        m = x_flat.size
        min_d2 = np.empty(m, dtype=float)
        nx_out = np.empty(m, dtype=float)
        ny_out = np.empty(m, dtype=float)
        p0x, p0y = self._edge_p0[:, 0][None, :], self._edge_p0[:, 1][None, :]
        sx, sy = self._edge_seg[:, 0][None, :], self._edge_seg[:, 1][None, :]
        l2 = self._edge_len2[None, :]
        enx, eny = self._edge_normals[:, 0], self._edge_normals[:, 1]

        for start in range(0, m, chunk_size):
            end = min(start + chunk_size, m)
            dx = x_flat[start:end, None] - p0x
            dy = y_flat[start:end, None] - p0y
            t = np.clip((dx * sx + dy * sy) / np.maximum(l2, 1e-30), 0.0, 1.0)
            rx, ry = dx - t * sx, dy - t * sy
            d2 = rx * rx + ry * ry
            d2_min = np.min(d2, axis=1)
            tie = d2 <= d2_min[:, None] * (1.0 + 1e-9) + 1e-28
            nx_sum, ny_sum = tie @ enx, tie @ eny
            norm = np.hypot(nx_sum, ny_sum)
            safe_norm = np.where(norm > 1e-12, norm, 1.0)
            idx = np.argmin(d2, axis=1)
            min_d2[start:end] = d2_min
            nx_out[start:end] = np.where(norm > 1e-12, nx_sum / safe_norm, enx[idx])
            ny_out[start:end] = np.where(norm > 1e-12, ny_sum / safe_norm, eny[idx])
        return np.sqrt(min_d2), nx_out, ny_out

    def weight(self, x, y, z, tol):
        """Point-sample weight at `(x, y, z)`: 1 inside, 0.5 on boundary, 0 outside."""
        X, Y, Z = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )
        dist_z = 0.5 * (self.z_max - self.z_min) - np.abs(
            Z - 0.5 * (self.z_min + self.z_max)
        )
        wz = np.where(dist_z > tol, 1.0, np.where(dist_z >= -tol, 0.5, 0.0))
        xf, yf = X.ravel(), Y.ravel()
        d_xy, _, _ = self._dist_and_normal_xy(xf, yf)
        inside = self._contains_xy(np.column_stack([xf, yf]))
        sd = np.where(inside, d_xy, -d_xy).reshape(X.shape)
        wxy = np.where(sd > tol, 1.0, np.where(sd >= -tol, 0.5, 0.0))
        return wxy * wz

    def fill(self, X, Y, Z, hx, hy, hz, n_sub=6):
        """
        Volume fill fraction of the cell centered at `(X, Y, Z)` with sizes
        `(hx, hy, hz)` [m], using `n_sub x n_sub` in-plane supersampling and
        analytical z overlap.
        """
        X, Y, Z, hx, hy, hz = np.broadcast_arrays(
            np.asarray(X, dtype=float),
            np.asarray(Y, dtype=float),
            np.asarray(Z, dtype=float),
            np.asarray(hx, dtype=float),
            np.asarray(hy, dtype=float),
            np.asarray(hz, dtype=float),
        )
        out_shape = X.shape
        z_lo = np.maximum(Z - 0.5 * hz, self.z_min)
        z_hi = np.minimum(Z + 0.5 * hz, self.z_max)
        fz = np.where(
            hz > 0,
            np.clip((z_hi - z_lo) / np.where(hz > 0, hz, 1.0), 0.0, 1.0),
            np.where((Z >= self.z_min) & (Z <= self.z_max), 1.0, 0.0),
        )
        lo, hi = self._lo, self._hi
        active = (
            (fz > 0)
            & (X + 0.5 * hx >= lo[0])
            & (X - 0.5 * hx <= hi[0])
            & (Y + 0.5 * hy >= lo[1])
            & (Y - 0.5 * hy <= hi[1])
        )
        if not np.any(active):
            return np.zeros(out_shape, dtype=float)

        xa, ya, za = X[active].ravel(), Y[active].ravel(), Z[active].ravel()
        hxa, hya = hx[active].ravel(), hy[active].ravel()
        d_xy, _, _ = self._dist_and_normal_xy(xa, ya)
        far = d_xy > 0.5 * np.hypot(hxa, hya) * (1.0 + 1e-7)

        fxy = np.zeros(xa.size, dtype=float)
        if np.any(far):
            fxy[far] = self._contains_xy(
                np.column_stack([xa[far], ya[far]]), z=za[far]
            ).astype(float)

        cut_idx = np.flatnonzero(~far)
        if cut_idx.size > 0:
            u = (np.arange(n_sub, dtype=float) + 0.5) / n_sub - 0.5
            uu, vv = np.meshgrid(u, u, indexing="ij")
            uu_f, vv_f = uu.ravel()[None, :], vv.ravel()[None, :]
            k_sub = n_sub * n_sub
            chunk = max(1, 200_000 // k_sub)
            for start in range(0, cut_idx.size, chunk):
                sel = cut_idx[start : start + chunk]
                px = xa[sel, None] + hxa[sel, None] * uu_f
                py = ya[sel, None] + hya[sel, None] * vv_f
                ins = self._contains_xy(
                    np.column_stack([px.ravel(), py.ravel()]), z=za[sel]
                )
                fxy[sel] = ins.reshape(sel.size, k_sub).mean(axis=1)

        result = np.zeros(out_shape, dtype=float)
        result[active] = fxy * fz[active]
        return result

    def _normal_xy(self, X, Y):
        X, Y = np.broadcast_arrays(
            np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        )
        _, nx, ny = self._dist_and_normal_xy(X.ravel(), Y.ravel())
        out = np.zeros(X.shape + (3,), dtype=float)
        out[..., 0] = nx.reshape(X.shape)
        out[..., 1] = ny.reshape(X.shape)
        return out

    def normal(self, X, Y, Z):
        """Unit outward normal `(nx, ny, nz)` of the nearest boundary at `(X, Y, Z)` [m]."""
        X, Y, Z = np.broadcast_arrays(
            np.asarray(X, dtype=float),
            np.asarray(Y, dtype=float),
            np.asarray(Z, dtype=float),
        )
        xf, yf, zf = X.ravel(), Y.ravel(), Z.ravel()
        d_xy, nx, ny = self._dist_and_normal_xy(xf, yf)
        d_bot, d_top = np.abs(zf - self.z_min), np.abs(self.z_max - zf)
        d_z = np.minimum(d_bot, d_top)
        use_z = d_z < d_xy
        out = np.zeros((xf.size, 3), dtype=float)
        out[~use_z, 0] = nx[~use_z]
        out[~use_z, 1] = ny[~use_z]
        out[use_z, 2] = np.where(d_top <= d_bot, 1.0, -1.0)[use_z]
        return out.reshape(X.shape + (3,))

    def to_pyvista(self):
        """Extruded 3D surface mesh as `pyvista.PolyData` [m]."""
        v = self.vertices
        n = len(v)
        v_top = (
            v - (self.z_max - self.z_min) * self._tan_angle * self._outer_vertex_normals
            if self._tan_angle != 0.0
            else v
        )
        points = np.vstack(
            [
                np.column_stack([v, np.full(n, self.z_min)]),
                np.column_stack([v_top, np.full(n, self.z_max)]),
            ]
        )
        faces = [n] + list(range(n - 1, -1, -1)) + [n] + list(range(n, 2 * n))
        for i in range(n):
            j = (i + 1) % n
            faces.extend([4, i, j, n + j, n + i])
        return pv.PolyData(points, np.asarray(faces, dtype=int)).triangulate()


# --------------------------------------------------- path and geometry helpers
def straight(p0, p1, width):
    """Counter-clockwise `(4, 2)` vertices [m] of a straight waveguide from `p0` to `p1`."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    dp = p1 - p0
    length = np.linalg.norm(dp)
    if length == 0:
        raise ValueError("p0 and p1 must be distinct.")
    t = dp / length
    n = np.array([-t[1], t[0]]) * (0.5 * float(width))
    return _clean_polygon(np.array([p0 - n, p1 - n, p1 + n, p0 + n]), ccw=True)


def taper(p0, p1, w0, w1, profile="linear", n=64):
    """Symmetric waveguide taper between `(p0, w0)` and `(p1, w1)` [m]."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    dp = p1 - p0
    length = np.linalg.norm(dp)
    if length == 0:
        raise ValueError("p0 and p1 must be distinct.")
    t = dp / length
    normal = np.array([-t[1], t[0]])
    s = np.linspace(0.0, 1.0, 2 if profile == "linear" else max(int(n), 2))
    if profile == "linear":
        w = w0 + (w1 - w0) * s
    elif profile == "cosine":
        w = w0 + (w1 - w0) * 0.5 * (1.0 - np.cos(np.pi * s))
    elif callable(profile):
        vals = np.asarray(profile(s), dtype=float)
        w = (
            w0 + (w1 - w0) * vals
            if np.isclose(vals[0], 0.0, atol=1e-6)
            and np.isclose(vals[-1], 1.0, atol=1e-6)
            else vals
        )
    else:
        raise ValueError(f"Unknown taper profile {profile!r}.")
    center = p0[None, :] + s[:, None] * dp[None, :]
    lower = center - 0.5 * w[:, None] * normal[None, :]
    upper = center + 0.5 * w[:, None] * normal[None, :]
    return _clean_polygon(np.vstack([lower, upper[::-1]]), ccw=True)


def spline_taper(length, widths, x0=0.0, y0=0.0, n=200):
    """Smooth cubic-spline width profile through equally spaced control points [m]."""
    widths = np.asarray(widths, dtype=float)
    if widths.ndim != 1 or widths.size < 2:
        raise ValueError("widths must be a 1D sequence with at least 2 points.")
    x_knots = np.linspace(float(x0), float(x0) + float(length), widths.size)
    cs = CubicSpline(x_knots, widths)
    x = np.linspace(float(x0), float(x0) + float(length), int(n))
    half_w = 0.5 * cs(x)
    lower = np.column_stack([x, float(y0) - half_w])
    upper = np.column_stack([x[::-1], float(y0) + half_w[::-1]])
    return _clean_polygon(np.vstack([lower, upper]), ccw=True)


def arc(center, radius, width, theta0, theta1, n=64):
    """Circular arc waveguide of midline `radius` and transverse `width` [m]."""
    cx, cy = float(center[0]), float(center[1])
    r_in, r_out = float(radius) - 0.5 * float(width), float(radius) + 0.5 * float(width)
    if r_in <= 0:
        raise ValueError("radius must exceed width / 2.")
    theta = np.linspace(float(theta0), float(theta1), int(n))
    c, s = np.cos(theta), np.sin(theta)
    inner = np.column_stack([cx + r_in * c, cy + r_in * s])
    outer = np.column_stack([cx + r_out * c[::-1], cy + r_out * s[::-1]])
    return _clean_polygon(np.vstack([inner, outer]), ccw=True)


def euler_bend(radius_min, angle_deg, width, p=1.0, n=200):
    """
    Clothoid (Euler) bend with linear curvature in arc length up to `1 / radius_min`.

    Returns `(polygon, centerline)` in metres.
    """
    if not (0.0 < p <= 1.0):
        raise ValueError("Euler parameter p must lie in (0, 1].")
    if radius_min <= 0.5 * width:
        raise ValueError("radius_min must exceed width / 2.")
    angle_rad = np.deg2rad(float(angle_deg))
    if angle_rad == 0:
        raise ValueError("angle_deg must be non-zero.")
    sign = 1.0 if angle_rad >= 0 else -1.0
    theta_tot = abs(angle_rad)
    r_min = float(radius_min)
    theta_e = 0.5 * p * theta_tot
    L_e = 2.0 * r_min * theta_e
    L_c = r_min * (1.0 - p) * theta_tot
    L_tot = 2.0 * L_e + L_c

    n_fine = max(4001, 20 * int(n) + 1)
    s_fine = np.linspace(0.0, L_tot, n_fine)
    theta_fine = np.empty_like(s_fine)
    m1 = s_fine <= L_e
    m2 = (s_fine > L_e) & (s_fine <= L_e + L_c)
    m3 = s_fine > L_e + L_c
    theta_fine[m1] = 0.5 * s_fine[m1] ** 2 / (r_min * L_e)
    theta_fine[m2] = theta_e + (s_fine[m2] - L_e) / r_min
    theta_fine[m3] = theta_tot - 0.5 * (L_tot - s_fine[m3]) ** 2 / (r_min * L_e)
    theta_fine *= sign

    x_fine = cumulative_trapezoid(np.cos(theta_fine), s_fine, initial=0.0)
    y_fine = cumulative_trapezoid(np.sin(theta_fine), s_fine, initial=0.0)
    s = np.linspace(0.0, L_tot, int(n))
    centerline = np.column_stack(
        [np.interp(s, s_fine, x_fine), np.interp(s, s_fine, y_fine)]
    )
    theta = np.interp(s, s_fine, theta_fine)
    normal = np.column_stack([-np.sin(theta), np.cos(theta)])
    half_w = 0.5 * float(width)
    polygon = _clean_polygon(
        np.vstack([centerline - half_w * normal, (centerline + half_w * normal)[::-1]]),
        ccw=True,
    )
    return polygon, centerline


def sbend(p0, dx, dy, width, profile="cosine", n=100):
    """S-bend waveguide connecting `p0` to `(x0 + dx, y0 + dy)` [m] with horizontal tangents."""
    x0, y0 = float(p0[0]), float(p0[1])
    dx, dy = float(dx), float(dy)
    if dx <= 0:
        raise ValueError("dx must be positive.")
    u = np.linspace(0.0, 1.0, int(n))
    x = x0 + dx * u
    if profile == "cosine":
        y = y0 + 0.5 * dy * (1.0 - np.cos(np.pi * u))
        dydx = (0.5 * np.pi * dy / dx) * np.sin(np.pi * u)
    elif profile == "cubic":
        y = y0 + dy * (3.0 * u**2 - 2.0 * u**3)
        dydx = (6.0 * dy / dx) * u * (1.0 - u)
    else:
        raise ValueError(f"Unknown sbend profile {profile!r}.")
    inv_norm = 1.0 / np.hypot(1.0, dydx)
    nx, ny = -dydx * inv_norm, inv_norm
    half_w = 0.5 * float(width)
    lower = np.column_stack([x - half_w * nx, y - half_w * ny])
    upper = np.column_stack([x + half_w * nx, y + half_w * ny])[::-1]
    return _clean_polygon(np.vstack([lower, upper]), ccw=True)


def ring(center, radius, width, n=128):
    """Annular ring as `(outer, [inner])` polygons [m] for `PolySlab`."""
    cx, cy = float(center[0]), float(center[1])
    r_out, r_in = float(radius) + 0.5 * float(width), float(radius) - 0.5 * float(width)
    if r_in <= 0:
        raise ValueError("radius must exceed width / 2.")
    theta = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    c, s = np.cos(theta), np.sin(theta)
    outer = _clean_polygon(np.column_stack([cx + r_out * c, cy + r_out * s]), ccw=True)
    inner = _clean_polygon(np.column_stack([cx + r_in * c, cy + r_in * s]), ccw=True)
    return outer, [inner]


# ----------------------------------------- effective index & vertical slab mode
def slab_neff(n_core, n_clad, thickness, wavelength, n_sub=None, polarization="TE"):
    """Fundamental slab effective index `n_eff` of a 3-layer vertical stack."""
    if n_sub is None or np.isclose(n_sub, n_clad):
        return slab_waveguide_neff(
            n_core, n_clad, thickness, wavelength, polarization=polarization
        )
    pol = polarization.upper()
    n_low = max(float(n_clad), float(n_sub))
    if not (n_low < n_core):
        raise ValueError("Require max(n_clad, n_sub) < n_core.")
    k0 = 2.0 * np.pi / float(wavelength)

    def dispersion(neff):
        kappa = k0 * np.sqrt(max(n_core**2 - neff**2, 0.0))
        g_c = k0 * np.sqrt(max(neff**2 - n_clad**2, 0.0))
        g_s = k0 * np.sqrt(max(neff**2 - n_sub**2, 0.0))
        if pol == "TE":
            return kappa * thickness - np.arctan2(
                kappa * (g_c + g_s), kappa**2 - g_c * g_s
            )
        r_c = (n_core / n_clad) ** 2 * g_c
        r_s = (n_core / n_sub) ** 2 * g_s
        return kappa * thickness - np.arctan2(kappa * (r_c + r_s), kappa**2 - r_c * r_s)

    return float(brentq(dispersion, n_low + 1e-12, n_core - 1e-12))


def vertical_slab_mode(
    n_core=PIC_MATERIALS["Si"],
    n_clad=PIC_MATERIALS["SiO2"],
    thickness=220e-9,
    wavelength=1550e-9,
    n_sub=None,
    z_span=2.0e-6,
    n_pts=401,
):
    """Analytical 1D vertical TE0 slab mode profile `E_y(z)` and index profile `n(z)`."""
    n_sub = n_clad if n_sub is None else float(n_sub)
    neff = slab_neff(
        n_core, n_clad, thickness, wavelength, n_sub=n_sub, polarization="TE"
    )
    k0 = 2.0 * np.pi / float(wavelength)
    kappa = k0 * np.sqrt(n_core**2 - neff**2)
    g_c = k0 * np.sqrt(neff**2 - n_clad**2)
    g_s = k0 * np.sqrt(neff**2 - n_sub**2)
    z = np.linspace(-0.5 * z_span, 0.5 * z_span, int(n_pts))
    d2 = 0.5 * thickness
    phi = (
        0.5 * np.arctan2(g_s - g_c, kappa * (1.0 + g_s * g_c / kappa**2))
        if n_sub != n_clad
        else 0.0
    )
    field = np.empty_like(z)
    in_core, above, below = np.abs(z) <= d2, z > d2, z < -d2
    field[in_core] = np.cos(kappa * z[in_core] - phi)
    field[above] = np.cos(kappa * d2 - phi) * np.exp(-g_c * (z[above] - d2))
    field[below] = np.cos(-kappa * d2 - phi) * np.exp(g_s * (z[below] + d2))
    field /= np.max(np.abs(field))
    n_prof = np.where(in_core, n_core, np.where(above, n_clad, n_sub))
    confinement = float(np.sum(field[in_core] ** 2) / np.sum(field**2))
    return {
        "z": z,
        "n_profile": n_prof,
        "field": field,
        "neff": neff,
        "confinement": confinement,
    }


# ------------------------------------------------------- ports & PICSimulation
class Port:
    """Named waveguide mode port on a plane normal to x or y."""

    def __init__(self, name, center, size, direction=+1, mode_index=0):
        self.name = str(name)
        self.center = np.asarray(center, dtype=float)
        self.size = np.asarray(size, dtype=float)
        zero_axes = np.flatnonzero(self.size == 0)
        if zero_axes.size != 1 or zero_axes[0] not in (0, 1):
            raise ValueError(
                "Port `size` must have a single zero entry on axis x or y."
            )
        self.axis = int(zero_axes[0])
        if isinstance(direction, str):
            if direction not in ("+", "-"):
                raise ValueError("Port `direction` must be +1, -1, '+' or '-'.")
            self.direction = 1 if direction == "+" else -1
        else:
            if int(direction) not in (1, -1):
                raise ValueError("Port `direction` must be +1 or -1.")
            self.direction = int(direction)
        self.mode_index = int(mode_index)


class _SnapshotRecorder:
    """Lightweight time-domain monitor recording downsampled 2D midplane slices and a port probe."""

    def __init__(self, component="y", max_frames=50, stride_xy=1, n_steps_est=1000):
        self.component = component[-1].lower()
        self.max_frames = int(max_frames)
        self.stride_xy = max(1, int(stride_xy))
        self.step_interval = max(1, int(n_steps_est) // max(1, self.max_frames))
        self.frames, self.times = [], []
        self.step = 0
        self.x = self.y = None
        self.port_trace_t, self.port_trace_val = [], []
        self.probe_idx = None

    def update(self, solver, t):
        if self.x is None:
            g = solver.grid
            s = self.stride_xy
            self.x = (0.5 * (g.x[:-1] + g.x[1:]))[::s].astype(np.float32)
            self.y = (0.5 * (g.y[:-1] + g.y[1:]))[::s].astype(np.float32)

        need_frame = (
            self.max_frames > 0
            and self.step % self.step_interval == 0
            and len(self.frames) < self.max_frames
        )
        if self.probe_idx is not None or need_frame:
            mat = solver.E.to_matrix(self.component)
            if hasattr(mat, "get"):
                mat = mat.get()
            if self.probe_idx is not None:
                ix, iy, iz = self.probe_idx
                self.port_trace_t.append(float(t))
                self.port_trace_val.append(float(mat[ix, iy, iz]))
            if need_frame:
                iz = solver.Nz // 2
                s = self.stride_xy
                self.frames.append(
                    np.asarray(mat[::s, ::s, iz], dtype=np.float32).copy()
                )
                self.times.append(float(t))
        self.step += 1


class PICSimulation:
    """
    Photonic-integrated-circuit wrapper on top of `Simulation`.

    Supports both the 2D effective-index method (`two_d=True`, default) and 3D
    extruded `PolySlab` stacks, with subpixel Kottke permittivity smoothing and
    multi-port S-parameter extraction. Pass `backend="subgrid"` to dispatch to
    `wakis.subgrid.SubgridPICSimulation`.

    `mesh_overrides` (list of (lo, hi, max_cell) boxes [m]) refines the mesh
    locally, e.g. to put >= 4 cells across a coupler gap, slot or grating
    tooth narrower than lambda / (cells_per_wavelength n_eff); see `Simulation`.
    """

    def __new__(cls, *args, backend=None, **kwargs):
        if backend == "subgrid":
            from ..subgrid.pic import SubgridPICSimulation

            return SubgridPICSimulation(*args, **kwargs)
        if backend not in (None, "cartesian"):
            raise ValueError(
                f"Unknown PICSimulation backend {backend!r}; expected None, 'cartesian' or 'subgrid'."
            )
        return super().__new__(cls)

    def __init__(
        self,
        wavelength_min=1500e-9,
        wavelength_max=1600e-9,
        structures=(),
        ports=(),
        n_core=PIC_MATERIALS["Si"],
        n_clad=PIC_MATERIALS["SiO2"],
        n_sub=None,
        wavelength_ref=1550e-9,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=20,
        n_freqs=51,
        domain=None,
        padding=None,
        n_pml=10,
        cfln=0.5,
        mesh_overrides=(),
        field_wavelengths=(1550e-9,),
        record_movie=False,
        movie_frames=50,
        use_gpu=False,
        backend=None,
    ):
        if not 0 < wavelength_min < wavelength_max:
            raise ValueError("Require 0 < wavelength_min < wavelength_max.")
        self.wavelength_min = float(wavelength_min)
        self.wavelength_max = float(wavelength_max)
        self.wavelength_ref = float(wavelength_ref)
        self.f_min = c_light / self.wavelength_max
        self.f_max = c_light / self.wavelength_min
        self.f_center = 0.5 * (self.f_min + self.f_max)
        self.structures_3d = list(structures)
        self.ports = list(ports)
        self.port_map = {p.name: p for p in self.ports}
        self.n_core = float(n_core)
        self.n_clad = float(n_clad)
        self.n_sub = float(n_clad if n_sub is None else n_sub)
        self.two_d = bool(two_d)
        self.averaging = averaging
        self.cells_per_wavelength = float(cells_per_wavelength)
        self.n_freqs = int(n_freqs)
        self.padding = 0.8e-6 if padding is None else float(padding)
        self.n_pml = int(n_pml) if n_pml is not None else None
        self.cfln = float(cfln)
        self.mesh_overrides = list(mesh_overrides)
        self.field_wavelengths = tuple(float(w) for w in field_wavelengths)
        self.record_movie = bool(record_movie)
        self.movie_frames = min(int(movie_frames), 60)
        self.use_gpu = bool(use_gpu)

        z_mins = [s.bounds[0][2] for s in self.structures_3d]
        z_maxs = [s.bounds[1][2] for s in self.structures_3d]
        self.z_min = min(z_mins) if z_mins else -110e-9
        self.z_max = max(z_maxs) if z_maxs else 110e-9
        self.thickness = self.z_max - self.z_min
        self.n_eff_slab = slab_neff(
            self.n_core,
            self.n_clad,
            self.thickness,
            self.wavelength_ref,
            n_sub=self.n_sub,
            polarization="TE",
        )
        self._domain_input = domain
        self.sim = self.solver = self.grid = None
        self.frequencies = self.wavelengths = None
        self.dft_fields = {}
        self.movie_data = None
        self.port_time_trace = None
        self.flux_balance = None
        self.build()

    def _build_2d_structures_and_domain(self):
        if self._domain_input is not None:
            xmin, xmax, ymin, ymax = self._domain_input[:4]
        else:
            all_lo = np.min([s.bounds[0] for s in self.structures_3d], axis=0)
            all_hi = np.max([s.bounds[1] for s in self.structures_3d], axis=0)
            xmin, xmax = float(all_lo[0]), float(all_hi[0])
            ymin, ymax = (
                float(all_lo[1]) - self.padding,
                float(all_hi[1]) + self.padding,
            )

        h_core = meshing.cell_size(
            self.f_max, self.n_eff_slab**2, self.cells_per_wavelength
        )
        z0 = 0.5 * (self.z_min + self.z_max)
        zmin_2d, zmax_2d = z0 - 0.5 * h_core, z0 + 0.5 * h_core
        domain = (xmin, xmax, ymin, ymax, zmin_2d, zmax_2d)

        margin = 20.0 * self.wavelength_max
        bg_box = Box(
            center=(0.5 * (xmin + xmax), 0.5 * (ymin + ymax), z0),
            size=(xmax - xmin + margin, ymax - ymin + margin, 4.0 * h_core),
            material=n_to_eps(self.n_clad),
        )
        eps_core_3d = n_to_eps(self.n_core)
        eps_slab_2d = n_to_eps(self.n_eff_slab)

        structs_2d = [bg_box]
        for s in self.structures_3d:
            s_copy = copy.deepcopy(s)
            if isinstance(s_copy, PolySlab):
                s_copy.z_min, s_copy.z_max = zmin_2d, zmax_2d
                s_copy._lo[2], s_copy._hi[2] = zmin_2d, zmax_2d
                if np.isclose(
                    material_eps(s_copy.material, self.f_center), eps_core_3d, rtol=1e-2
                ):
                    s_copy.material = eps_slab_2d
            structs_2d.append(s_copy)
        return structs_2d, domain

    def build(self):
        """Build the mesh, solver, and subpixel-averaged permittivity tensor."""
        if self.two_d:
            structs, domain = self._build_2d_structures_and_domain()
            boundaries = (["cpml", "cpml", "periodic"], ["cpml", "cpml", "periodic"])
        else:
            structs = list(self.structures_3d)
            domain = self._domain_input
            boundaries = "cpml"

        self.sim = Simulation(
            self.f_min,
            self.f_max,
            structures=structs,
            boundaries=boundaries,
            domain=domain,
            cells_per_wavelength=self.cells_per_wavelength,
            padding=self.padding,
            n_pml=self.n_pml,
            n_freqs=self.n_freqs,
            cfln=self.cfln,
            mesh_overrides=self.mesh_overrides,
            use_gpu=self.use_gpu,
            averaging=self.averaging,
        )
        self.solver = self.sim.build()
        if self.two_d:
            # Restrict 2D EIM to in-plane quasi-TE (Ex, Ey, Hz) by zeroing ieps_z
            self.solver.ieps[:, :, :, "z"] = 0.0
        self.grid = self.sim.grid
        self.frequencies = self.sim.frequencies
        self.wavelengths = c_light / self.frequencies
        return self.solver

    def _port_plane_args(self, port):
        center, size = port.center.copy(), port.size.copy()
        if self.two_d:
            center[2] = 0.5 * (self.grid.zmin + self.grid.zmax)
            size[2] = np.inf
        a = port.axis
        mid_a = 0.5 * (
            (self.grid.xmin + self.grid.xmax)
            if a == 0
            else (self.grid.ymin + self.grid.ymax)
        )
        inward = 1 if center[a] <= mid_a else -1
        return center, size, inward

    def solve_port_modes(self, port_name, frequencies=None, n_modes=1):
        """Solve the eigenmodes on `port_name` at `frequencies` [Hz]."""
        port = self.port_map[port_name]
        center, size, _ = self._port_plane_args(port)
        nm = max(n_modes, port.mode_index + 1)
        if frequencies is None:
            return solve_modes(self.solver, self.f_center, center, size, n_modes=nm)
        if np.ndim(frequencies) == 0:
            return solve_modes(
                self.solver, float(frequencies), center, size, n_modes=nm
            )
        return [
            solve_modes(self.solver, float(f), center, size, n_modes=nm)
            for f in frequencies
        ]

    def run_s_parameters(
        self, drive=None, energy_decay=1e-4, max_time=None, collect_fields=True
    ):
        """
        Run one time-domain simulation per driven port and return `S[(out, in)](f)`.

        Normalized by the forward amplitude measured at the driven port's own
        `ModeMonitor`, so `|S[(out, in)]|^2` is the guided modal power fraction.
        """
        if not self.ports:
            raise ValueError("No ports defined on PICSimulation.")
        if drive is None:
            drive = [self.ports[0].name]
        elif isinstance(drive, str):
            drive = [p.name for p in self.ports] if drive == "all" else [drive]

        if max_time is None:
            lx = self.grid.xmax - self.grid.xmin
            ly = self.grid.ymax - self.grid.ymin
            transit = np.hypot(lx, ly) * self.n_core / c_light
            max_time = self.sim.pulse.end_time + 15.0 * transit

        s_matrix = {}
        for run_idx, in_name in enumerate(drive):
            in_port = self.port_map[in_name]
            in_center, in_size, in_inward = self._port_plane_args(in_port)
            a_in = in_port.axis
            nodes_a = self.grid.x if a_in == 0 else self.grid.y
            k_mon = int(np.argmin(np.abs(nodes_a[:-1] - in_center[a_in])))
            k_src = int(np.clip(k_mon - 2 * in_inward, 2, len(nodes_a) - 3))
            src_center = in_center.copy()
            src_center[a_in] = nodes_a[k_src]

            self.build()
            src = ModeSource(
                self.sim.pulse,
                center=src_center,
                size=in_size,
                direction="+" if in_inward > 0 else "-",
                mode_index=in_port.mode_index,
                mode_frequency=self.f_center,
            )

            port_monitors, monitors_list = {}, []
            for p in self.ports:
                p_center, p_size, _ = self._port_plane_args(p)
                mon = ModeMonitor(
                    self.frequencies,
                    center=p_center,
                    size=p_size,
                    n_modes=p.mode_index + 1,
                    mode_frequency=self.f_center,
                )
                port_monitors[p.name] = mon
                monitors_list.append(mon)

            dft_mon = snap_mon = flux_planes = None
            if collect_fields and run_idx == 0:
                f_dft = [c_light / w for w in self.field_wavelengths]
                stride = max(1, int(np.ceil(max(self.grid.Nx, self.grid.Ny) / 400)))
                z_mid = 0.5 * (self.grid.zmin + self.grid.zmax)
                dft_mon = DFTMonitor(
                    f_dft,
                    components=("Ex", "Ey"),
                    center=(
                        0.5 * (self.grid.xmin + self.grid.xmax),
                        0.5 * (self.grid.ymin + self.grid.ymax),
                        z_mid,
                    ),
                    size=(np.inf, np.inf, 0.0),
                )
                monitors_list.append(dft_mon)

                Nt_est = int(np.ceil(max_time / self.solver.dt))
                snap_mon = _SnapshotRecorder(
                    component="y",
                    max_frames=self.movie_frames if self.record_movie else 0,
                    stride_xy=stride,
                    n_steps_est=Nt_est,
                )
                trace_port = self.ports[1] if len(self.ports) > 1 else in_port
                tp_c, _, _ = self._port_plane_args(trace_port)
                ix = int(np.argmin(np.abs(self.grid.x[:-1] - tp_c[0])))
                iy = int(np.argmin(np.abs(self.grid.y[:-1] - tp_c[1])))
                snap_mon.probe_idx = (ix, iy, self.solver.Nz // 2)
                monitors_list.append(snap_mon)

                if self.two_d:
                    # 2D Poynting flux box inside CPML to measure lateral and unguided radiation
                    n_p = self.sim.n_pml + 1
                    xw = float(self.grid.x[max(1, k_src - 2)])
                    xe = float(self.grid.x[-1 - n_p])
                    ys = float(self.grid.y[n_p])
                    yn = float(self.grid.y[-1 - n_p])
                    xc_b, yc_b = 0.5 * (xw + xe), 0.5 * (ys + yn)
                    lx_b, ly_b = xe - xw, yn - ys
                    flux_planes = {
                        "west": FluxMonitor(
                            self.frequencies, (xw, yc_b, z_mid), (0.0, ly_b, np.inf)
                        ),
                        "east": FluxMonitor(
                            self.frequencies, (xe, yc_b, z_mid), (0.0, ly_b, np.inf)
                        ),
                        "south": FluxMonitor(
                            self.frequencies, (xc_b, ys, z_mid), (lx_b, 0.0, np.inf)
                        ),
                        "north": FluxMonitor(
                            self.frequencies, (xc_b, yn, z_mid), (lx_b, 0.0, np.inf)
                        ),
                    }
                    monitors_list.extend(flux_planes.values())

            Nt = int(np.ceil(max_time / self.solver.dt))
            self.solver.emsolve(
                Nt,
                source=[src],
                monitors=monitors_list,
                stop_when_energy_decayed=energy_decay,
            )

            a_plus_in, a_minus_in = port_monitors[in_name].get_amplitudes(
                in_port.mode_index
            )
            a_inc = a_plus_in if in_inward > 0 else a_minus_in

            for p in self.ports:
                _, _, p_inward = self._port_plane_args(p)
                a_p, a_m = port_monitors[p.name].get_amplitudes(p.mode_index)
                b_out = a_m if p_inward > 0 else a_p
                s_matrix[(p.name, in_name)] = b_out / a_inc

            if dft_mon is not None:
                nx, ny = self.grid.Nx, self.grid.Ny
                xc = 0.5 * (self.grid.x[:-1] + self.grid.x[1:])
                yc = 0.5 * (self.grid.y[:-1] + self.grid.y[1:])
                ex_raw, _ = dft_mon.get_field("Ex")
                ey_raw, _ = dft_mon.get_field("Ey")
                ex_c = ex_raw[:, :nx, :ny, 0]
                ey_c = ey_raw[:, :nx, :ny, 0]
                x_dft = xc[::stride].astype(np.float32)
                y_dft = yc[::stride].astype(np.float32)
                for idx_w, w_val in enumerate(self.field_wavelengths):
                    ex = ex_c[idx_w, ::stride, ::stride].astype(np.complex64)
                    ey = ey_c[idx_w, ::stride, ::stride].astype(np.complex64)
                    self.dft_fields[w_val] = {
                        "x": x_dft,
                        "y": y_dft,
                        "Ex": ex,
                        "Ey": ey,
                        "E2": (np.abs(ex) ** 2 + np.abs(ey) ** 2).astype(np.float32),
                    }

            if snap_mon is not None:
                self.port_time_trace = {
                    "t": np.asarray(snap_mon.port_trace_t, dtype=np.float32),
                    "Ey": np.asarray(snap_mon.port_trace_val, dtype=np.float32),
                }
                if self.record_movie and snap_mon.frames:
                    frames_f32 = np.stack(snap_mon.frames, axis=0)
                    peak = max(float(np.max(np.abs(frames_f32))), 1e-30)
                    self.movie_data = {
                        "x": snap_mon.x,
                        "y": snap_mon.y,
                        "times": np.asarray(snap_mon.times, dtype=np.float32),
                        "frames": (frames_f32 / peak).astype(np.float16),
                        "peak": np.float32(peak),
                    }

            if flux_planes is not None:
                p_inc = 2.0 * np.abs(a_inc) ** 2
                f_east = flux_planes["east"].get_flux() / p_inc
                f_west = -flux_planes["west"].get_flux() / p_inc
                f_north = flux_planes["north"].get_flux() / p_inc
                f_south = -flux_planes["south"].get_flux() / p_inc
                self.flux_balance = {
                    "east": f_east,
                    "west": f_west,
                    "lateral": f_north + f_south,
                    "total_box": f_east + f_west + f_north + f_south,
                }

        return s_matrix


# -------------------------------------------------------------- figure helpers
def _finalize_fig(fig, out_path, close=True):
    fig.tight_layout()
    if out_path is not None:
        FilePath(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if close:
            plt.close(fig)


def _poly_to_patch(poly_slab, facecolor="#2b5c8f", edgecolor="#16324f", alpha=0.85):
    verts_um = [np.vstack([poly_slab.vertices * 1e6, poly_slab.vertices[0] * 1e6])]
    codes = [
        [MplPath.MOVETO]
        + [MplPath.LINETO] * (len(poly_slab.vertices) - 1)
        + [MplPath.CLOSEPOLY]
    ]
    for h in poly_slab.holes:
        h_um = h[::-1] * 1e6
        verts_um.append(np.vstack([h_um, h_um[0]]))
        codes.append(
            [MplPath.MOVETO] + [MplPath.LINETO] * (len(h_um) - 1) + [MplPath.CLOSEPOLY]
        )
    path = MplPath(np.vstack(verts_um), np.concatenate(codes))
    return PathPatch(
        path, facecolor=facecolor, edgecolor=edgecolor, lw=1.2, alpha=alpha
    )


def plot_layout(
    structures,
    ports=(),
    domain=None,
    annotations=(),
    title="Device layout (top view)",
    out_path=None,
    ax=None,
):
    """Plot top-view x-y PIC layout with ports and dimensional annotations."""
    created = ax is None
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=150) if created else (ax.figure, ax)
    all_lo = np.min([s.bounds[0][:2] for s in structures], axis=0) * 1e6
    all_hi = np.max([s.bounds[1][:2] for s in structures], axis=0) * 1e6

    for s in structures:
        if isinstance(s, PolySlab):
            ax.add_patch(_poly_to_patch(s))

    colors = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a"]
    for i, p in enumerate(ports):
        col = colors[i % len(colors)]
        c_um, s_um = p.center[:2] * 1e6, p.size[:2] * 1e6
        if p.axis == 0:
            y0, y1 = c_um[1] - 0.5 * s_um[1], c_um[1] + 0.5 * s_um[1]
            ax.plot([c_um[0], c_um[0]], [y0, y1], color=col, lw=2.5, zorder=5)
            ax.annotate(
                "",
                xy=(c_um[0] + 0.35 * p.direction, c_um[1]),
                xytext=(c_um[0], c_um[1]),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.8),
            )
            ax.text(
                c_um[0],
                y1 + 0.12,
                f"Port '{p.name}'",
                color=col,
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="bottom",
            )
        else:
            x0, x1 = c_um[0] - 0.5 * s_um[0], c_um[0] + 0.5 * s_um[0]
            ax.plot([x0, x1], [c_um[1], c_um[1]], color=col, lw=2.5, zorder=5)
            ax.text(
                x1 + 0.1,
                c_um[1],
                f"Port '{p.name}'",
                color=col,
                fontsize=9,
                fontweight="bold",
            )

    for ann in annotations:
        ax.annotate(
            ann.get("text", ""),
            xy=ann["xy"],
            xytext=ann.get("xytext", ann["xy"]),
            fontsize=8.5,
            ha=ann.get("ha", "center"),
            va=ann.get("va", "center"),
            arrowprops=ann.get("arrowprops"),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
        )

    if domain is not None:
        x0, x1, y0, y1 = [v * 1e6 for v in domain[:4]]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    else:
        pad_x = 0.08 * (all_hi[0] - all_lo[0] + 1.0)
        pad_y = 0.25 * (all_hi[1] - all_lo[1] + 1.0)
        ax.set_xlim(all_lo[0] - pad_x, all_hi[0] + pad_x)
        ax.set_ylim(all_lo[1] - pad_y, all_hi[1] + pad_y)

    ax.set(aspect="equal", xlabel="x [µm]", ylabel="y [µm]", title=title)
    ax.grid(True, ls=":", alpha=0.4)
    _finalize_fig(fig, out_path, close=created)
    return fig, ax


def plot_eim_stack(
    n_core=PIC_MATERIALS["Si"],
    n_clad=PIC_MATERIALS["SiO2"],
    thickness=220e-9,
    wavelength_ref=1550e-9,
    wavelengths=None,
    out_path=None,
):
    """Plot the vertical SOI stack, 1D TE0 slab mode profile, and `n_eff,slab(lambda)`."""
    if wavelengths is None:
        wavelengths = np.linspace(1500e-9, 1600e-9, 51)
    vm = vertical_slab_mode(
        n_core=n_core, n_clad=n_clad, thickness=thickness, wavelength=wavelength_ref
    )
    neff_band = np.array(
        [
            slab_neff(n_core, n_clad, thickness, w, polarization="TE")
            for w in wavelengths
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), dpi=150)
    d_nm = thickness * 1e9
    ax0, ax1, ax2 = axes
    ax0.axhspan(-600, -0.5 * d_nm, color="#d9e6f2", label=f"SiO₂ BOX (n={n_clad:.3f})")
    ax0.axhspan(
        -0.5 * d_nm,
        0.5 * d_nm,
        color="#2b5c8f",
        alpha=0.85,
        label=f"Si core ({d_nm:.0f} nm, n={n_core:.3f})",
    )
    ax0.axhspan(0.5 * d_nm, 600, color="#eef4fa", label=f"SiO₂ clad (n={n_clad:.3f})")
    ax0.axhline(-0.5 * d_nm, color="#16324f", lw=1.2)
    ax0.axhline(0.5 * d_nm, color="#16324f", lw=1.2)
    ax0.set(
        ylim=(-600, 600),
        xlim=(0, 1),
        xticks=[],
        ylabel="Vertical coordinate z [nm]",
        title="(a) Vertical SOI layer stack",
    )
    ax0.legend(loc="upper right", fontsize=8)

    z_nm = vm["z"] * 1e9
    ax1.axvspan(-0.5 * d_nm, 0.5 * d_nm, color="#2b5c8f", alpha=0.18, label="Si core")
    ax1.plot(z_nm, vm["field"], color="#d95f02", lw=2.0, label=r"$|E_y(z)|$ (TE$_0$)")
    ax1.plot(
        z_nm,
        vm["field"] ** 2,
        color="#1b9e77",
        lw=1.6,
        ls="--",
        label=f"$|E_y|^2$ (Γ={vm['confinement'] * 100:.1f}%)",
    )
    ax1.set(
        xlim=(-600, 600),
        ylim=(0, 1.08),
        xlabel="Vertical coordinate z [nm]",
        ylabel="Normalized mode amplitude",
        title=f"(b) 1D slab TE₀ mode (n_eff = {vm['neff']:.4f} @ {wavelength_ref * 1e9:.0f} nm)",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8)

    wl_nm = np.asarray(wavelengths) * 1e9
    ax2.plot(
        wl_nm,
        neff_band,
        color="#2b5c8f",
        lw=2.0,
        label=r"$n_{\mathrm{eff,slab}}(\lambda)$",
    )
    ax2.axhline(
        vm["neff"],
        color="#d95f02",
        ls="--",
        lw=1.6,
        label=f"Constant @ 1550 nm ({vm['neff']:.4f})",
    )
    ax2.set(
        xlabel="Wavelength λ [nm]",
        ylabel=r"Vertical slab effective index $n_{\mathrm{eff,slab}}$",
        title="(c) Slab effective index dispersion",
    )
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8)

    _finalize_fig(fig, out_path)
    return fig, axes


def plot_3d(
    structures,
    out_path=None,
    n_core=PIC_MATERIALS["Si"],
    n_clad=PIC_MATERIALS["SiO2"],
    thickness=220e-9,
):
    """2D-scope replacement for 3D render: plots the EIM vertical stack and slab mode."""
    return plot_eim_stack(
        n_core=n_core, n_clad=n_clad, thickness=thickness, out_path=out_path
    )


def plot_cross_section_mesh(
    solver_staircase,
    solver_kottke,
    xlim_um=None,
    ylim_um=None,
    component="y",
    out_path=None,
):
    """Side-by-side in-plane permittivity map with mesh lines for staircase vs Kottke."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=150, sharey=True)
    titles = [
        "(a) Staircase permittivity (rounded fill)",
        f"(b) Kottke subpixel smoothing (eps_{component})",
    ]
    for ax, solver, ttl in zip(axes, (solver_staircase, solver_kottke), titles):
        g = solver.grid
        iz = solver.Nz // 2
        ieps = solver.ieps[:, :, iz, component]
        if hasattr(ieps, "get"):
            ieps = ieps.get()
        eps_r = np.where(
            ieps > 0, 1.0 / (np.where(ieps > 0, ieps, 1.0) * epsilon_0), 1.0
        )
        x_um, y_um = g.x * 1e6, g.y * 1e6
        pcm = ax.pcolormesh(x_um, y_um, eps_r.T, cmap="viridis", shading="flat")
        if xlim_um is not None:
            ax.set_xlim(*xlim_um)
        if ylim_um is not None:
            ax.set_ylim(*ylim_um)
        xl, yl = ax.get_xlim(), ax.get_ylim()
        xv = x_um[(x_um >= xl[0]) & (x_um <= xl[1])]
        yv = y_um[(y_um >= yl[0]) & (y_um <= yl[1])]
        vlines = [[(x, yl[0]), (x, yl[1])] for x in xv]
        hlines = [[(xl[0], y), (xl[1], y)] for y in yv]
        ax.add_collection(
            LineCollection(vlines + hlines, colors="white", linewidths=0.35, alpha=0.55)
        )
        ax.set(aspect="equal", xlabel="x [µm]", title=ttl)
        cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(rf"Relative permittivity $\varepsilon_{{r,{component}}}$")
    axes[0].set_ylabel("y [µm]")
    _finalize_fig(fig, out_path)
    return fig, axes


def plot_port_modes(
    y_in_um,
    ey_in,
    neff_in,
    y_out_um,
    ey_out,
    neff_out,
    structures_outline_y=None,
    out_path=None,
):
    """Plot 1D transverse port mode profiles (`|Ey|` and `Re(Ey)`) at input and output."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), dpi=150)
    panels = [
        (axes[0], y_in_um, ey_in, neff_in, "(a) Input port mode (x = x_in)"),
        (axes[1], y_out_um, ey_out, neff_out, "(b) Output arm port mode (x = x_out)"),
    ]
    for ax, y_um, ey, neff, ttl in panels:
        scale = np.max(np.abs(ey)) or 1.0
        ax.plot(
            y_um,
            np.abs(ey) / scale,
            color="#2b5c8f",
            lw=2.0,
            label=r"$|E_y| / \max|E_y|$",
        )
        ax.plot(
            y_um,
            np.real(ey) / scale,
            color="#d95f02",
            ls="--",
            lw=1.6,
            label=r"$\mathrm{Re}(E_y) / \max|E_y|$",
        )
        for y_lo, y_hi in structures_outline_y or ():
            ax.axvspan(y_lo, y_hi, color="0.85", alpha=0.5)
        ax.set(
            xlabel="Transverse coordinate y [µm]",
            ylabel="Normalized modal field",
            title=f"{ttl}\n$n_{{\\mathrm{{eff}}}}$ = {neff:.4f}",
        )
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8.5)
    _finalize_fig(fig, out_path)
    return fig, axes


def plot_neff_dispersion(
    wavelengths_nm, neff_2d, neff_slab=None, ng_2d=None, out_path=None
):
    """Plot waveguide mode effective index `n_eff(lambda)` and group index."""
    fig, ax1 = plt.subplots(figsize=(6.8, 3.8), dpi=150)
    ax1.plot(
        wavelengths_nm,
        neff_2d,
        color="#2b5c8f",
        lw=2.2,
        label=r"Channel mode $n_{\mathrm{eff}}(\lambda)$ (2D EIM)",
    )
    if neff_slab is not None:
        ax1.axhline(
            neff_slab,
            color="0.45",
            ls=":",
            lw=1.5,
            label=f"Core slab index $n_{{\\mathrm{{eff,slab}}}}$ = {neff_slab:.4f}",
        )
    ax1.set_xlabel("Wavelength λ [nm]")
    ax1.set_ylabel(r"Effective index $n_{\mathrm{eff}}$", color="#2b5c8f")
    ax1.grid(True, ls=":", alpha=0.4)
    if ng_2d is not None:
        ax2 = ax1.twinx()
        ax2.plot(
            wavelengths_nm,
            ng_2d,
            color="#d95f02",
            ls="--",
            lw=1.8,
            label=r"Group index $n_g(\lambda)$",
        )
        ax2.set_ylabel(r"Group index $n_g$", color="#d95f02")
        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lb1 + lb2, loc="center right", fontsize=8.5)
    else:
        ax1.legend(loc="upper right", fontsize=8.5)
    ax1.set_title("Port waveguide mode dispersion across the C-band")
    _finalize_fig(fig, out_path)
    return fig, ax1


def plot_field_map(
    x_m,
    y_m,
    field_2d,
    structures=(),
    kind="intensity_log",
    title=None,
    out_path=None,
    db_floor=-35.0,
):
    """Plot a 2D midplane field map (`intensity_log` in dB or symmetric `real`) with outlines."""
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=150)
    x_um, y_um = np.asarray(x_m) * 1e6, np.asarray(y_m) * 1e6
    if kind == "intensity_log":
        f_max = float(np.max(field_2d)) or 1.0
        data_db = 10.0 * np.log10(
            np.maximum(field_2d / f_max, 10.0 ** (db_floor / 10.0))
        )
        im = ax.pcolormesh(
            x_um,
            y_um,
            data_db.T,
            cmap="inferno",
            vmin=db_floor,
            vmax=0.0,
            shading="auto",
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"Normalized intensity $10\log_{10}(|E|^2 / |E|_{\max}^2)$ [dB]")
        outline_col = "white"
    else:
        vmax = float(np.percentile(np.abs(field_2d), 99.5)) or 1.0
        im = ax.pcolormesh(
            x_um,
            y_um,
            np.real(field_2d).T / vmax,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            shading="auto",
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"Normalized $\mathrm{Re}(E_y)$ [arb. units]")
        outline_col = "black"

    for s in structures:
        if isinstance(s, PolySlab):
            ax.add_patch(
                MplPolygon(
                    s.vertices * 1e6,
                    closed=True,
                    fill=False,
                    edgecolor=outline_col,
                    lw=1.1,
                    alpha=0.85,
                )
            )
    ax.set(
        aspect="equal",
        xlim=(x_um.min(), x_um.max()),
        ylim=(y_um.min(), y_um.max()),
        xlabel="x [µm]",
        ylabel="y [µm]",
    )
    if title:
        ax.set_title(title)
    _finalize_fig(fig, out_path)
    return fig, ax


def plot_spectra_db(
    wavelengths_nm,
    curves_db,
    title="S-parameter spectra vs wavelength",
    ylabel="Transmission / Reflection [dB]",
    ylim=None,
    reference_lines=(),
    out_path=None,
):
    """Plot multiple dB spectra against wavelength [nm]."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#2b5c8f"]
    styles = ["-", "--", "-.", ":"]
    for i, (label, y_db) in enumerate(curves_db.items()):
        ax.plot(
            wavelengths_nm,
            y_db,
            color=palette[i % len(palette)],
            ls=styles[i % len(styles)],
            lw=2.0,
            label=label,
        )
    for ref in reference_lines:
        ax.axhline(
            ref["y"],
            color=ref.get("color", "0.4"),
            ls=ref.get("ls", ":"),
            lw=1.3,
            label=ref.get("label"),
        )
    ax.set(xlabel="Wavelength λ [nm]", ylabel=ylabel, title=title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="best", fontsize=8.5)
    _finalize_fig(fig, out_path)
    return fig, ax


def plot_convergence(
    resolutions,
    err_staircase,
    err_kottke,
    ylabel="Relative error vs reference",
    title="Subpixel convergence: Kottke vs staircase",
    out_path=None,
):
    """Log-log convergence plot of error vs cells per wavelength with order guides."""
    res = np.asarray(resolutions, dtype=float)
    err_s = np.maximum(np.asarray(err_staircase, dtype=float), 1e-12)
    err_k = np.maximum(np.asarray(err_kottke, dtype=float), 1e-12)
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.loglog(
        res,
        err_s,
        "s-",
        color="#d95f02",
        lw=2.0,
        ms=6,
        label="Staircase (rounded fill)",
    )
    ax.loglog(
        res,
        err_k,
        "o-",
        color="#1b9e77",
        lw=2.0,
        ms=6,
        label="Kottke subpixel smoothing",
    )
    r0 = res[0]
    ax.loglog(
        res,
        err_s[0] * (r0 / res) ** 1,
        "--",
        color="#d95f02",
        alpha=0.55,
        lw=1.2,
        label=r"$\mathcal{O}(\Delta x^1)$ guide",
    )
    ax.loglog(
        res,
        err_k[0] * (r0 / res) ** 2,
        ":",
        color="#1b9e77",
        alpha=0.65,
        lw=1.4,
        label=r"$\mathcal{O}(\Delta x^2)$ guide",
    )
    ax.set(xlabel="Resolution [cells per wavelength]", ylabel=ylabel, title=title)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="best", fontsize=8.5)
    _finalize_fig(fig, out_path)
    return fig, ax


def make_movie(
    x_m, y_m, frames, times_s, structures=(), out_path="field_propagation.gif", fps=12
):
    """Render a time-domain pulse propagation GIF from downsampled `float16` midplane frames."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    x_um, y_um = np.asarray(x_m) * 1e6, np.asarray(y_m) * 1e6
    frames = np.asarray(frames, dtype=np.float32)
    vmax = float(np.percentile(np.abs(frames), 99.2)) or 1.0
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=110)
    im = ax.pcolormesh(
        x_um,
        y_um,
        frames[0].T / vmax,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        shading="auto",
    )
    for s in structures:
        if isinstance(s, PolySlab):
            ax.add_patch(
                MplPolygon(
                    s.vertices * 1e6, closed=True, fill=False, edgecolor="black", lw=1.0
                )
            )
    ax.set(
        aspect="equal",
        xlim=(x_um.min(), x_um.max()),
        ylim=(y_um.min(), y_um.max()),
        xlabel="x [µm]",
        ylabel="y [µm]",
    )
    title_obj = ax.set_title(f"Pulse propagation Ey (t = {times_s[0] * 1e15:.1f} fs)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"Normalized $E_y(x, y, t)$")
    fig.tight_layout()

    def _update(frame_idx):
        im.set_array((frames[frame_idx].T / vmax).ravel())
        title_obj.set_text(
            f"Pulse propagation Ey (t = {times_s[frame_idx] * 1e15:.1f} fs)"
        )
        return im, title_obj

    anim = FuncAnimation(
        fig, _update, frames=len(frames), interval=1000 // fps, blit=False
    )
    FilePath(out_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
