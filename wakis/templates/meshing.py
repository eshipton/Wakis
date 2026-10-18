# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Automatic setup rules: cell size from cells per wavelength, padding between
the structure and the absorbing boundary, CPML thickness and graded
(piecewise uniform) mesh axes snapped to the structure faces.

The rules are the usual engineering defaults of time-domain solvers:

* cell size: `cells_per_wavelength` cells per wavelength at `f_max` in each
  material, Delta <= c / (f_max n N). The FIT/FDTD phase error per wavelength
  scales as (pi / N)^2 / 6, ~0.4 % for N = 20 [Taflove & Hagness, 2005, Ch. 4];
* padding: a quarter wavelength at `f_min` between the structure and the CPML,
  so that the reactive near field does not reach the absorber;
* CPML thickness: at least 10 cells (polynomial grading of order 4, reflection
  ~1e-4..1e-6 for normal incidence) and at least lambda(f_min) / 8 for broadband
  runs with large wavelengths in cells.
"""

import numpy as np
from scipy.constants import c as c_light


def wavelength(frequency, eps_r=1.0, mu_r=1.0):
    """
    Wavelength [m] at `frequency` [Hz] in a medium of relative eps_r, mu_r.
    """
    return c_light / (frequency * np.sqrt(eps_r * mu_r))


def cell_size(f_max, eps_r=1.0, cells_per_wavelength=20, mu_r=1.0):
    """
    Largest cell size [m] resolving the wavelength at `f_max` [Hz] in a medium
    (eps_r, mu_r) with `cells_per_wavelength` cells.
    """
    return wavelength(f_max, eps_r, mu_r) / cells_per_wavelength


def padding(f_min, fraction=0.25):
    """
    Free-space distance [m] between the structure and the absorbing boundary:
    `fraction` of the vacuum wavelength at `f_min` [Hz] (default lambda/4).
    """
    return fraction * wavelength(f_min)


def pml_cells(f_min, dx, minimum=10, maximum=30):
    """
    CPML thickness in cells: max(`minimum`, lambda(f_min) / (8 dx)), clipped to
    `maximum`. dx is the cell size at the boundary [m].
    """
    n = int(np.ceil(wavelength(f_min) / (8 * dx)))
    return int(np.clip(n, minimum, maximum))


def graded_axis(points, sizes, min_cells=1, grading=1.4, samples=400):
    """
    Mesh axis with nodes on all `points` and smoothly graded cell sizes.

    Each interval [points[i], points[i+1]] has a target cell size sizes[i],
    reduced to the interval length / `min_cells`. The cell-size field is then
    limited so that adjacent cells grow by at most `grading` (slightly more
    where the cell count of an interval is rounded up),
    s(x) <= s_i + (grading - 1) dist(x, interval i), and the nodes of each
    interval are placed at equal increments of the integral of 1/s.

    Parameters
    ----------
    points : array_like
        Sorted interval end points [m] (structure faces, domain ends).
    sizes : array_like
        Target cell size of each interval [m], len(points) - 1 values.
    min_cells : int or array_like, optional
        Minimum number of cells per interval. Default 1.
    grading : float, optional
        Maximum ratio of adjacent cell sizes. Default 1.4.
    samples : int, optional
        Quadrature samples per interval. Default 400.

    Returns
    -------
    ndarray
        Node coordinates [m], including all `points`.
    """
    points = np.asarray(points, dtype=float)
    lengths = np.diff(points)
    if np.any(lengths <= 0):
        raise ValueError("points must be strictly increasing.")
    min_cells = np.broadcast_to(np.asarray(min_cells), lengths.shape)
    target = np.minimum(np.asarray(sizes, dtype=float), lengths / min_cells)

    def size_field(x):
        dist = np.maximum(points[:-1, None] - x, x - points[1:, None])
        return np.min(target[:, None] + (grading - 1) * np.maximum(dist, 0), axis=0)

    nodes = [points[:1]]
    for a, b in zip(points[:-1], points[1:]):
        x = np.linspace(a, b, samples)
        density = 1 / size_field(x)
        cumulative = np.concatenate(
            [[0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(x))]
        )
        n = max(1, int(np.ceil(cumulative[-1] - 1e-6)))
        fractions = np.arange(1, n + 1) / n * cumulative[-1]
        inner = np.interp(fractions[:-1], cumulative, x)
        nodes.append(np.append(inner, b))
    return np.concatenate(nodes)


def merge_points(points, tol):
    """Sort `points` and merge those closer than `tol` [m] (keeps the first)."""
    points = np.sort(np.asarray(points, dtype=float))
    merged = [points[0]]
    for p in points[1:]:
        if p - merged[-1] > tol:
            merged.append(p)
    return np.asarray(merged)
