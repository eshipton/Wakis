# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Analytic validation of the plane-section conformal geometry engine
(`wakis.conformal.sections`) on the upstream staggered FIT grid layout.
"""

import numpy as np
import pytest
import pyvista as pv

from wakis.conformal import (
    SheetMeasures,
    SolidMeasures,
    sheet_measures,
    solid_measures,
)
from wakis.gridFIT3D import GridFIT3D


def _overlap_1d(lo, hi, a, b):
    """Exact 1D interval overlap max(0, min(b, hi) - max(a, lo))."""
    return np.maximum(0.0, np.minimum(b, hi) - np.maximum(a, lo))


def _grid_steps(x, y, z):
    """Minimum and maximum cell widths across all three axes."""
    diffs = np.concatenate([np.diff(x), np.diff(y), np.diff(z)])
    return float(diffs.min()), float(diffs.max())


def _octahedron_surface(center, a, inward=False):
    """Exact 8-triangle octahedron |x - cx| + |y - cy| + |z - cz| <= a."""
    cx, cy, cz = center
    vertices = np.array(
        [
            [cx + a, cy, cz],
            [cx - a, cy, cz],
            [cx, cy + a, cz],
            [cx, cy - a, cz],
            [cx, cy, cz + a],
            [cx, cy, cz - a],
        ],
        dtype=float,
    )
    # Outward-oriented faces in the 8 octants
    triangles = [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ]
    if inward:
        triangles = [[i, k, j] for i, j, k in triangles]
    faces = np.hstack([[3, *tri] for tri in triangles])
    return pv.PolyData(vertices, faces=faces)


def _rectangle_sheet_surface(u0, u1, v0, v1, w_val, axis=2):
    """Two-triangle axis-aligned rectangular sheet on plane w = w_val."""
    cyclic = {0: (1, 2), 1: (2, 0), 2: (0, 1)}
    u, v = cyclic[axis]
    corners = np.zeros((4, 3), dtype=float)
    corners[:, axis] = w_val
    corners[:, u] = [u0, u1, u1, u0]
    corners[:, v] = [v0, v0, v1, v1]
    return pv.PolyData(corners, faces=[3, 0, 1, 2, 3, 0, 2, 3])


def test_stretched_grid_box_overlaps():
    """
    Axis-aligned box with faces between nodes on a stretched grid: edge
    lengths, face areas, and clipped dual fractions equal the products of 1D
    interval overlaps.
    """
    tx_param = np.linspace(0.0, 1.0, 11)
    ty_param = np.linspace(-0.2, 1.0, 13)
    tz_param = np.linspace(0.1, 1.1, 10)
    x = np.sort(tx_param + 0.015 * np.sin(2.0 * np.pi * tx_param))
    y = np.sort(ty_param + 0.018 * np.sin(2.0 * np.pi * (ty_param + 0.2) / 1.2))
    z = np.sort(tz_param + 0.012 * np.sin(2.0 * np.pi * (tz_param - 0.1)))

    h_min, h_max = _grid_steps(x, y, z)
    atol_len = 1e-6 * h_min
    atol_area = 1e-6 * h_min * h_max
    atol_frac = 1e-6
    rtol = 1e-12

    # Place faces between nodes and inside boundary half-cells to test both
    # interior and clipped boundary dual intervals on the stretched grid
    bounds = (
        x[0] + 0.23 * (x[1] - x[0]),
        x[-2] + 0.77 * (x[-1] - x[-2]),
        y[0] + 0.31 * (y[1] - y[0]),
        y[-2] + 0.69 * (y[-1] - y[-2]),
        z[0] + 0.25 * (z[1] - z[0]),
        z[-1] - 0.25 * (z[-1] - z[-2]),
    )
    box = pv.Box(bounds=bounds)
    res = solid_measures(box, x, y, z, n_sub=4)
    assert isinstance(res, SolidMeasures)

    axes = (x, y, z)
    box_lo = bounds[0::2]
    box_hi = bounds[1::2]

    primal_ov = [
        _overlap_1d(a[:-1], a[1:], lo, hi) for a, lo, hi in zip(axes, box_lo, box_hi)
    ]
    primal_frac = [ov / (a[1:] - a[:-1]) for ov, a in zip(primal_ov, axes)]
    node_in = [
        ((a > lo) & (a < hi)).astype(float) for a, lo, hi in zip(axes, box_lo, box_hi)
    ]
    dual_bounds = [np.r_[a[0], 0.5 * (a[:-1] + a[1:]), a[-1]] for a in axes]
    dual_frac = [
        _overlap_1d(t[:-1], t[1:], lo, hi) / (t[1:] - t[:-1])
        for t, lo, hi in zip(dual_bounds, box_lo, box_hi)
    ]

    for c in range(3):
        factors_len = [primal_ov[a] if a == c else node_in[a] for a in range(3)]
        exp_len = (
            factors_len[0][:, None, None]
            * factors_len[1][None, :, None]
            * factors_len[2][None, None, :]
        )
        np.testing.assert_allclose(
            res.edge_length[c], exp_len, rtol=rtol, atol=atol_len
        )

        factors_area = [node_in[a] if a == c else primal_ov[a] for a in range(3)]
        exp_area = (
            factors_area[0][:, None, None]
            * factors_area[1][None, :, None]
            * factors_area[2][None, None, :]
        )
        np.testing.assert_allclose(
            res.face_area[c], exp_area, rtol=rtol, atol=atol_area
        )

        factors_dual = [primal_frac[a] if a == c else dual_frac[a] for a in range(3)]
        exp_dual = (
            factors_dual[0][:, None, None]
            * factors_dual[1][None, :, None]
            * factors_dual[2][None, None, :]
        )
        np.testing.assert_allclose(
            res.dual_fraction[c], exp_dual, rtol=rtol, atol=atol_frac
        )


def test_octahedron_edges_and_plane_areas():
    """
    Off-centre octahedron |x - cx| + |y - cy| + |z - cz| <= a built from its 8
    exact triangles: every x-edge inside length matches the analytic chord
    overlap and every z-plane total face area equals 2 (a - |z - cz|)_+^2.
    """
    x = np.linspace(-0.5, 0.6, 14)
    y = np.linspace(-0.45, 0.55, 13)
    z = np.linspace(-0.4, 0.6, 15)
    h_min, h_max = _grid_steps(x, y, z)
    atol_len = 1e-6 * h_min
    atol_area = 1e-6 * h_min * h_max
    rtol = 1e-12

    center = (0.043, -0.031, 0.067)
    a = 0.31
    # Strictly inside the domain on all axes
    assert center[0] - a > x[0] and center[0] + a < x[-1]
    assert center[1] - a > y[0] and center[1] + a < y[-1]
    assert center[2] - a > z[0] and center[2] + a < z[-1]

    # Pass inward-oriented triangles to verify _outward reorientation as well
    octa = _octahedron_surface(center, a, inward=True)
    res = solid_measures(octa, x, y, z, dual=False)
    assert res.dual_fraction is None
    assert res.normal is None

    cx, cy, cz = center
    r = np.maximum(a - np.abs(y[:, None] - cy) - np.abs(z[None, :] - cz), 0.0)
    exp_lx = _overlap_1d(
        x[:-1, None, None],
        x[1:, None, None],
        cx - r[None, :, :],
        cx + r[None, :, :],
    )
    np.testing.assert_allclose(res.edge_length[0], exp_lx, rtol=rtol, atol=atol_len)

    for k, zk in enumerate(z):
        d = max(a - abs(zk - cz), 0.0)
        exp_plane_area = 2.0 * d**2
        perimeter = 4.0 * np.sqrt(2.0) * d
        atol_plane = max(1e-6 * h_min * perimeter, atol_area)
        np.testing.assert_allclose(
            res.face_area[2][:, :, k].sum(),
            exp_plane_area,
            rtol=rtol,
            atol=atol_plane,
        )


def test_two_crossings_on_single_edge_and_face():
    """
    Two boxes along x separated by a gap g = 0.3 * dx strictly inside one
    x-edge: that edge has inside length dx - g and the corresponding faces
    have inside area (dx - g) * dy and (dx - g) * dz.
    """
    x = np.linspace(0.0, 1.0, 6)
    y = np.linspace(0.0, 0.8, 5)
    z = np.linspace(0.0, 0.6, 4)
    h_min, h_max = _grid_steps(x, y, z)
    atol_len = 1e-6 * h_min
    atol_area = 1e-6 * h_min * h_max
    rtol = 1e-12

    i_cut, j_cut, k_cut = 2, 1, 1
    dx = x[i_cut + 1] - x[i_cut]
    dy = y[j_cut + 1] - y[j_cut]
    dz = z[k_cut + 1] - z[k_cut]
    gap = 0.3 * dx
    gap_lo = x[i_cut] + 0.35 * dx
    gap_hi = gap_lo + gap

    ylo = y[j_cut] - 0.4 * dy
    yhi = y[j_cut + 1] + 0.4 * dy
    zlo = z[k_cut] - 0.4 * dz
    zhi = z[k_cut + 1] + 0.4 * dz

    box_left = pv.Box(bounds=(x[0] + 0.3 * dx, gap_lo, ylo, yhi, zlo, zhi))
    box_right = pv.Box(bounds=(gap_hi, x[-1] - 0.3 * dx, ylo, yhi, zlo, zhi))
    two_boxes = box_left.merge(box_right)

    res = solid_measures(two_boxes, x, y, z, dual=False)
    np.testing.assert_allclose(
        res.edge_length[0][i_cut, j_cut, k_cut],
        dx - gap,
        rtol=rtol,
        atol=atol_len,
    )
    np.testing.assert_allclose(
        res.face_area[2][i_cut, j_cut, k_cut],
        (dx - gap) * dy,
        rtol=rtol,
        atol=atol_area,
    )
    np.testing.assert_allclose(
        res.face_area[1][i_cut, j_cut, k_cut],
        (dx - gap) * dz,
        rtol=rtol,
        atol=atol_area,
    )


def test_box_on_node_planes_surface_modes():
    """
    Box whose faces lie on node planes: with on_surface='inside' the edges
    lying in the box faces (and the box faces themselves) have full measure;
    with 'outside' they have zero; interior elements are full and exterior
    elements are zero in both modes.
    """
    # Dyadic grid spacings so pv.Box (float32 points) places faces on exact nodes
    x = np.linspace(0.0, 1.5, 7)
    y = np.linspace(0.0, 1.75, 8)
    z = np.linspace(0.0, 1.25, 6)
    h_min, h_max = _grid_steps(x, y, z)
    atol_len = 1e-6 * h_min
    atol_area = 1e-6 * h_min * h_max
    rtol = 1e-12

    i0, i1 = 1, 5
    j0, j1 = 2, 6
    k0, k1 = 1, 4
    box = pv.Box(bounds=(x[i0], x[i1], y[j0], y[j1], z[k0], z[k1]))

    inside = solid_measures(box, x, y, z, on_surface="inside", dual=False)
    outside = solid_measures(box, x, y, z, on_surface="outside", dual=False)

    axes = (x, y, z)
    idx_lo = (i0, j0, k0)
    idx_hi = (i1, j1, k1)
    diffs = [np.diff(a) for a in axes]

    for c in range(3):
        u, v = {0: (1, 2), 1: (2, 0), 2: (0, 1)}[c]
        shape_l = inside.edge_length[c].shape
        ijk = np.ogrid[: shape_l[0], : shape_l[1], : shape_l[2]]
        full_len = np.broadcast_to(
            diffs[c].reshape([-1 if a == c else 1 for a in range(3)]), shape_l
        )

        # Along c the edge cell must be inside [lo_c, hi_c)
        in_c = (ijk[c] >= idx_lo[c]) & (ijk[c] < idx_hi[c])
        int_u = (ijk[u] > idx_lo[u]) & (ijk[u] < idx_hi[u])
        int_v = (ijk[v] > idx_lo[v]) & (ijk[v] < idx_hi[v])
        on_u = (ijk[u] == idx_lo[u]) | (ijk[u] == idx_hi[u])
        on_v = (ijk[v] == idx_lo[v]) | (ijk[v] == idx_hi[v])
        out_uv = (
            (ijk[u] < idx_lo[u])
            | (ijk[u] > idx_hi[u])
            | (ijk[v] < idx_lo[v])
            | (ijk[v] > idx_hi[v])
        )

        interior_edge = in_c & int_u & int_v
        face_edge = in_c & ((on_u & int_v) | (on_v & int_u))
        exterior_edge = (~in_c) | out_uv

        np.testing.assert_allclose(
            inside.edge_length[c][interior_edge],
            full_len[interior_edge],
            rtol=rtol,
            atol=atol_len,
        )
        np.testing.assert_allclose(
            outside.edge_length[c][interior_edge],
            full_len[interior_edge],
            rtol=rtol,
            atol=atol_len,
        )
        np.testing.assert_allclose(
            inside.edge_length[c][face_edge],
            full_len[face_edge],
            rtol=rtol,
            atol=atol_len,
        )
        np.testing.assert_allclose(
            outside.edge_length[c][face_edge],
            0.0,
            rtol=rtol,
            atol=atol_len,
        )
        np.testing.assert_allclose(
            inside.edge_length[c][exterior_edge],
            0.0,
            rtol=rtol,
            atol=atol_len,
        )
        np.testing.assert_allclose(
            outside.edge_length[c][exterior_edge],
            0.0,
            rtol=rtol,
            atol=atol_len,
        )

        # Faces normal to c: only one node coordinate (along c) can lie on a
        # box face while the two in-plane cell intervals lie inside the box
        cell_u = np.zeros(len(axes[u]) - 1, dtype=float)
        cell_u[idx_lo[u] : idx_hi[u]] = 1.0
        cell_u *= diffs[u]
        cell_v = np.zeros(len(axes[v]) - 1, dtype=float)
        cell_v[idx_lo[v] : idx_hi[v]] = 1.0
        cell_v *= diffs[v]

        for mode_res, closed in ((inside, True), (outside, False)):
            node_c = np.zeros(len(axes[c]), dtype=float)
            if closed:
                node_c[idx_lo[c] : idx_hi[c] + 1] = 1.0
            else:
                node_c[idx_lo[c] + 1 : idx_hi[c]] = 1.0
            factors = [None] * 3
            factors[c], factors[u], factors[v] = node_c, cell_u, cell_v
            exp_area = (
                factors[0][:, None, None]
                * factors[1][None, :, None]
                * factors[2][None, None, :]
            )
            np.testing.assert_allclose(
                mode_res.face_area[c], exp_area, rtol=rtol, atol=atol_area
            )

    with pytest.raises(ValueError, match="on_surface"):
        solid_measures(box, x, y, z, on_surface="invalid", dual=False)


def test_dual_fraction_clipped_at_domain_boundary():
    """
    Dual cells at the domain boundary have half-cell width inside the domain:
    a box covering [x[0] - dx, x[0] + 2.5 dx] has fraction 1 at x[0], and a
    box covering [x[0] + 0.25 dx, x[0] + 2.5 dx] has fraction 0.5 at x[0]
    (whereas unclipped extended bounds would give 0.25).
    """
    x = np.linspace(0.2, 1.2, 6)
    y = np.linspace(-0.3, 0.5, 5)
    z = np.linspace(0.0, 0.8, 5)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    atol_frac = 1e-6
    rtol = 1e-12

    box_full = pv.Box(
        bounds=(
            x[0] - dx,
            x[0] + 2.5 * dx,
            y[0] - dy,
            y[-1] + dy,
            z[0] - dz,
            z[-1] + dz,
        )
    )
    res_full = solid_measures(box_full, x, y, z)
    np.testing.assert_allclose(
        res_full.dual_fraction[1][0, :, :], 1.0, rtol=rtol, atol=atol_frac
    )
    np.testing.assert_allclose(
        res_full.dual_fraction[2][0, :, :], 1.0, rtol=rtol, atol=atol_frac
    )

    # Box starting midway through the clipped boundary dual interval [x[0], x[0] + 0.5 dx]
    box_half = pv.Box(
        bounds=(
            x[0] + 0.25 * dx,
            x[0] + 2.5 * dx,
            y[0] - dy,
            y[-1] + dy,
            z[0] - dz,
            z[-1] + dz,
        )
    )
    res_half = solid_measures(box_half, x, y, z)
    np.testing.assert_allclose(
        res_half.dual_fraction[1][0, :, :], 0.5, rtol=rtol, atol=atol_frac
    )
    np.testing.assert_allclose(
        res_half.dual_fraction[2][0, :, :], 0.5, rtol=rtol, atol=atol_frac
    )


def test_sheet_measures_and_validation():
    """
    Rectangle sheet on plane z = z[k] covering a non-aligned rectangle:
    in-plane edge lengths and face areas match 1D interval overlaps, all
    out-of-plane entries vanish, and sheets between planes or oblique sheets
    raise ValueError.
    """
    x = np.linspace(0.0, 1.0, 9)
    y = np.linspace(0.0, 1.2, 10)
    z = np.linspace(0.0, 0.8, 7)
    h_min, h_max = _grid_steps(x, y, z)
    atol_len = 1e-6 * h_min
    atol_area = 1e-6 * h_min * h_max
    rtol = 1e-12

    k = 3
    x0, x1 = 0.17, 0.79
    y0, y1 = 0.23, 0.91
    sheet = _rectangle_sheet_surface(x0, x1, y0, y1, z[k], axis=2)

    res = sheet_measures(sheet, x, y, z)
    assert isinstance(res, SheetMeasures)
    assert res.axis == 2
    assert res.index == k

    ov_x = _overlap_1d(x[:-1], x[1:], x0, x1)
    ov_y = _overlap_1d(y[:-1], y[1:], y0, y1)
    in_x = ((x > x0) & (x < x1)).astype(float)
    in_y = ((y > y0) & (y < y1)).astype(float)

    exp_lx = np.zeros((len(x) - 1, len(y), len(z)))
    exp_ly = np.zeros((len(x), len(y) - 1, len(z)))
    exp_lz = np.zeros((len(x), len(y), len(z) - 1))
    exp_lx[:, :, k] = ov_x[:, None] * in_y[None, :]
    exp_ly[:, :, k] = in_x[:, None] * ov_y[None, :]

    exp_ax = np.zeros((len(x), len(y) - 1, len(z) - 1))
    exp_ay = np.zeros((len(x) - 1, len(y), len(z) - 1))
    exp_az = np.zeros((len(x) - 1, len(y) - 1, len(z)))
    exp_az[:, :, k] = ov_x[:, None] * ov_y[None, :]

    np.testing.assert_allclose(res.edge_length[0], exp_lx, rtol=rtol, atol=atol_len)
    np.testing.assert_allclose(res.edge_length[1], exp_ly, rtol=rtol, atol=atol_len)
    np.testing.assert_array_equal(res.edge_length[2], exp_lz)
    np.testing.assert_array_equal(res.face_area[0], exp_ax)
    np.testing.assert_array_equal(res.face_area[1], exp_ay)
    np.testing.assert_allclose(res.face_area[2], exp_az, rtol=rtol, atol=atol_area)

    p, q = res.outline
    assert p.shape == (4, 2) and q.shape == (4, 2)
    seg_lengths = np.linalg.norm(q - p, axis=1).sum()
    np.testing.assert_allclose(
        seg_lengths, 2.0 * ((x1 - x0) + (y1 - y0)), rtol=rtol, atol=atol_len
    )

    sheet_between = _rectangle_sheet_surface(
        x0, x1, y0, y1, 0.5 * (z[k] + z[k + 1]), axis=2
    )
    with pytest.raises(ValueError, match="between the grid planes"):
        sheet_measures(sheet_between, x, y, z)

    oblique_pts = np.array(
        [
            [x0, y0, z[k]],
            [x1, y0, z[k]],
            [x1, y1, z[k + 1]],
            [x0, y1, z[k + 1]],
        ]
    )
    oblique_sheet = pv.PolyData(oblique_pts, faces=[3, 0, 1, 2, 3, 0, 2, 3])
    with pytest.raises(ValueError, match="planar and normal"):
        sheet_measures(oblique_sheet, x, y, z)


def test_sphere_normals():
    """
    On pv.Sphere(radius=R, theta_resolution=96, phi_resolution=96), every
    flagged edge normal is a unit outward vector within 5 degrees of the
    radial direction.
    """
    x = np.linspace(-0.5, 0.5, 11)
    y = np.linspace(-0.5, 0.5, 12)
    z = np.linspace(-0.5, 0.5, 13)
    center = np.array([0.023, -0.019, 0.031])
    radius = 0.33
    sphere = pv.Sphere(
        radius=radius, center=center, theta_resolution=96, phi_resolution=96
    )

    res = solid_measures(sphere, x, y, z, threads=1)

    centres = [0.5 * (a[:-1] + a[1:]) for a in (x, y, z)]
    axes = (x, y, z)
    cos_5deg = np.cos(np.deg2rad(5.0))

    for c in range(3):
        f = res.dual_fraction[c]
        n = res.normal[c]
        flagged = (f > 1e-9) & (f < 1.0 - 1e-9)
        assert np.any(flagged)
        np.testing.assert_array_equal(n[~flagged], 0.0)

        n_cut = n[flagged]
        np.testing.assert_allclose(
            np.linalg.norm(n_cut, axis=1), 1.0, rtol=1e-12, atol=1e-6
        )

        coords = [centres[a] if a == c else axes[a] for a in range(3)]
        idx = np.nonzero(flagged)
        pts = np.stack([coords[a][idx[a]] for a in range(3)], axis=1)
        radial = pts - center
        radial /= np.linalg.norm(radial, axis=1, keepdims=True)
        dots = np.einsum("ij,ij->i", n_cut, radial)
        assert np.all(dots > 0.0)
        assert np.all(dots >= cos_5deg)


def test_thread_independence():
    """
    Sphere measures with threads=1 and threads=4 are bitwise identical.
    """
    x = np.linspace(-0.5, 0.5, 11)
    y = np.linspace(-0.5, 0.5, 12)
    z = np.linspace(-0.5, 0.5, 13)
    sphere = pv.Sphere(
        radius=0.33,
        center=(0.023, -0.019, 0.031),
        theta_resolution=96,
        phi_resolution=96,
    )

    res1 = solid_measures(sphere, x, y, z, threads=1)
    res4 = solid_measures(sphere, x, y, z, threads=4)

    for c in range(3):
        np.testing.assert_array_equal(res1.edge_length[c], res4.edge_length[c])
        np.testing.assert_array_equal(res1.face_area[c], res4.face_area[c])
        np.testing.assert_array_equal(res1.dual_fraction[c], res4.dual_fraction[c])
        np.testing.assert_array_equal(res1.normal[c], res4.normal[c])


def test_upstream_gridfit3d_shapes():
    """
    Every returned array matches the physical component shapes of upstream
    GridFIT3D: grid.primal_coords[c] for edges/fractions/normals and
    grid.dual_coords[c] for faces.
    """
    grid = GridFIT3D(0.0, 1.0, -0.4, 0.6, 0.1, 0.9, 7, 9, 11, verbose=0)
    box = pv.Box(bounds=(0.15, 0.75, -0.2, 0.35, 0.25, 0.75))
    sheet = _rectangle_sheet_surface(0.2, 0.8, -0.1, 0.3, grid.z[4], axis=2)

    solid = solid_measures(box, grid.x, grid.y, grid.z)
    sheet_m = sheet_measures(sheet, grid.x, grid.y, grid.z)

    for c, comp in enumerate("xyz"):
        primal_shape = tuple(len(a) for a in grid.primal_coords[comp])
        dual_shape = tuple(len(a) for a in grid.dual_coords[comp])

        assert solid.edge_length[c].shape == primal_shape
        assert solid.dual_fraction[c].shape == primal_shape
        assert solid.normal[c].shape == (*primal_shape, 3)
        assert solid.face_area[c].shape == dual_shape

        assert sheet_m.edge_length[c].shape == primal_shape
        assert sheet_m.face_area[c].shape == dual_shape


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_sheet_outline_in_absolute_coordinates(axis):
    """The sheet outline is returned in metres, not relative to the grid origin."""
    cyclic = {0: (1, 2), 1: (2, 0), 2: (0, 1)}
    u, v = cyclic[axis]
    axes = [
        np.linspace(0.3, 1.3, 9),
        np.linspace(-0.7, 0.5, 10),
        np.linspace(2.0, 2.8, 7),
    ]
    k = 3
    lo = {u: axes[u][0] + 0.17, v: axes[v][0] + 0.23}
    hi = {u: axes[u][0] + 0.79, v: axes[v][0] + 0.61}
    sheet = _rectangle_sheet_surface(lo[u], hi[u], lo[v], hi[v], axes[axis][k], axis)

    res = sheet_measures(sheet, *axes)
    p, q = res.outline
    pts = np.concatenate([p, q])
    np.testing.assert_allclose(pts.min(axis=0), [lo[u], lo[v]], rtol=1e-12)
    np.testing.assert_allclose(pts.max(axis=0), [hi[u], hi[v]], rtol=1e-12)
