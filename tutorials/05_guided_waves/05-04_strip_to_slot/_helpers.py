# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Tutorial 5.4 (Strip-to-Slot Waveguide Mode Converters)
that extend ``wakis.templates.photonics`` with:

1. ``asymmetric_taper`` — planar waveguide rail taper with independent lower
   and upper transverse edge profiles ``y_bot(x)`` and ``y_top(x)`` and
   automatic vertex deduplication at sharp (zero-width) tips.
2. ``slot_straight`` — straight dual-rail slot waveguide polygon pair.
3. ``slot_waveguide_neff`` and ``slot_waveguide_mode`` — exact 1D 5-layer
   symmetric slot waveguide eigenvalue and field-profile solver (Almeida
   et al., Opt. Lett. 29, 1209, 2004) for the in-plane quasi-TE mode
   (normal ``E_y``, tangential ``H_z``).
"""

import numpy as np
from scipy.optimize import brentq

from wakis.templates.photonics import _clean_polygon


def asymmetric_taper(
    x0,
    x1,
    y_bot0,
    y_top0,
    y_bot1,
    y_top1,
    profile="linear",
    n=64,
):
    """
    Counter-clockwise polygon vertices of a planar taper with independent
    lower and upper edges ``y_bot(x)`` and ``y_top(x)``.

    Automatically collapses pinched endpoints (``y_bot == y_top``) into a
    single vertex so that sharp taper tips do not produce degenerate
    zero-length polygon edges.

    Parameters
    ----------
    x0, x1 : float
        Start and end axial coordinates [m] (`x0 != x1`).
    y_bot0, y_top0 : float
        Lower and upper y coordinates at ``x0`` [m] (`y_bot0 <= y_top0`).
    y_bot1, y_top1 : float
        Lower and upper y coordinates at ``x1`` [m] (`y_bot1 <= y_top1`).
    profile : {'linear', 'cosine'} or callable, optional
        Interpolation profile along normalized axial coordinate ``s in [0, 1]``.
    n : int, optional
        Number of axial sample points when ``profile != 'linear'``.

    Returns
    -------
    vertices : ndarray, shape (M, 2)
        Counter-clockwise polygon vertices `(x, y)` [m].
    """
    x0, x1 = float(x0), float(x1)
    y_bot0, y_top0 = float(y_bot0), float(y_top0)
    y_bot1, y_top1 = float(y_bot1), float(y_top1)
    if np.isclose(x0, x1, atol=1e-15):
        raise ValueError("x0 and x1 must be distinct.")
    if y_bot0 > y_top0 + 1e-15 or y_bot1 > y_top1 + 1e-15:
        raise ValueError("Require y_bot <= y_top at both ends of asymmetric_taper.")
    if np.isclose(y_bot0, y_top0, atol=1e-15) and np.isclose(
        y_bot1, y_top1, atol=1e-15
    ):
        raise ValueError("Taper cannot have zero width at both ends.")

    n_pts = 2 if profile == "linear" else max(int(n), 2)
    s = np.linspace(0.0, 1.0, n_pts)
    if profile == "linear":
        u = s
    elif profile == "cosine":
        u = 0.5 * (1.0 - np.cos(np.pi * s))
    elif callable(profile):
        u = np.asarray(profile(s), dtype=float)
    else:
        raise ValueError(f"Unknown taper profile {profile!r}.")

    x = x0 + (x1 - x0) * s
    y_bot = y_bot0 + (y_bot1 - y_bot0) * u
    y_top = y_top0 + (y_top1 - y_top0) * u

    lower = np.column_stack([x, y_bot])
    upper = np.column_stack([x[::-1], y_top[::-1]])

    # Drop duplicate vertex if right end (x1) or left end (x0) is a sharp tip
    if np.allclose(lower[-1], upper[0], atol=1e-14):
        upper = upper[1:]
    if len(upper) > 0 and np.allclose(upper[-1], lower[0], atol=1e-14):
        upper = upper[:-1]

    return _clean_polygon(np.vstack([lower, upper]), ccw=True)


def slot_straight(x0, x1, width_slot, gap_slot, y0=0.0):
    """
    Return counter-clockwise `(verts_bot, verts_top)` polygon vertex arrays [m]
    for a straight symmetric dual-rail slot waveguide along `x`.

    Parameters
    ----------
    x0, x1 : float
        Start and end axial coordinates [m].
    width_slot : float
        Total outer width across both rails and the central slot [m].
    gap_slot : float
        Central low-index slot width [m] (`0 < gap_slot < width_slot`).
    y0 : float, optional
        Transverse centerline coordinate [m].

    Returns
    -------
    verts_bot, verts_top : tuple of ndarray, each shape (4, 2)
        Counter-clockwise vertices [m] of the lower and upper rails.
    """
    w_tot = float(width_slot)
    g = float(gap_slot)
    if not (0.0 < g < w_tot):
        raise ValueError("Require 0 < gap_slot < width_slot.")
    half_tot = 0.5 * w_tot
    half_gap = 0.5 * g
    yc = float(y0)
    verts_bot = asymmetric_taper(
        x0, x1, yc - half_tot, yc - half_gap, yc - half_tot, yc - half_gap
    )
    verts_top = asymmetric_taper(
        x0, x1, yc + half_gap, yc + half_tot, yc + half_gap, yc + half_tot
    )
    return verts_bot, verts_top


def slot_waveguide_neff(n_high, n_low, slot_width, rail_width, wavelength):
    """
    Analytical fundamental effective index `n_eff` of a symmetric 5-layer
    slot waveguide (`n_low / n_high / n_low / n_high / n_low`) for the
    in-plane quasi-TE mode (`E_y` normal to the slot walls, `H_z` tangential).

    Solves the exact 5-layer transcendental dispersion relation of
    Almeida et al., Opt. Lett. 29(11), 1209-1211 (2004).

    Parameters
    ----------
    n_high : float
        Refractive index of the two high-index rails (in 2D EIM, `n_eff,slab`).
    n_low : float
        Refractive index of the central slot and outer cladding (`n_SiO2`).
    slot_width : float
        Central low-index slot width `g` [m].
    rail_width : float
        Single high-index rail width `w_rail = (w_slot - g) / 2` [m].
    wavelength : float
        Vacuum wavelength [m].

    Returns
    -------
    neff : float
        Fundamental symmetric slot-mode effective index.
    """
    n_h = float(n_high)
    n_s = float(n_low)
    if not (n_s < n_h):
        raise ValueError("Require n_low < n_high.")
    a = 0.5 * float(slot_width)
    w_r = float(rail_width)
    if a < 0.0 or w_r <= 0.0:
        raise ValueError("Require slot_width >= 0 and rail_width > 0.")
    k0 = 2.0 * np.pi / float(wavelength)
    r = (n_h / n_s) ** 2

    def dispersion(neff):
        kh = k0 * np.sqrt(max(n_h**2 - neff**2, 0.0))
        gs = k0 * np.sqrt(max(neff**2 - n_s**2, 0.0))
        tanh_gsa = np.tanh(gs * a)
        return np.sin(kh * w_r) * (kh**2 - r**2 * gs**2 * tanh_gsa) - (
            np.cos(kh * w_r) * r * gs * kh * (1.0 + tanh_gsa)
        )

    n_grid = np.linspace(n_s + 1e-7, n_h - 1e-7, 600)
    vals = [dispersion(val) for val in n_grid]
    for i in range(len(n_grid) - 2, -1, -1):
        if vals[i] * vals[i + 1] <= 0.0:
            return float(brentq(dispersion, n_grid[i], n_grid[i + 1], xtol=1e-12))
    raise RuntimeError("No fundamental slot mode root found in (n_low, n_high).")


def slot_waveguide_mode(
    n_high,
    n_low,
    slot_width,
    rail_width,
    wavelength=1550e-9,
    y_span=1.6e-6,
    n_pts=801,
):
    """
    Analytical 1D transverse mode profile `E_y(y)`, `D_y(y)`, and `H_z(y)` of
    a symmetric 5-layer slot waveguide (Almeida et al., 2004).

    Parameters
    ----------
    n_high : float
        High-index rail refractive index (in 2D EIM, `n_eff,slab`).
    n_low : float
        Low-index slot and cladding refractive index (`n_SiO2`).
    slot_width : float
        Central low-index slot width `g` [m].
    rail_width : float
        Width of each high-index rail `w_rail` [m].
    wavelength : float, optional
        Vacuum wavelength [m].
    y_span : float, optional
        Total transverse coordinate span `[-y_span/2, +y_span/2]` [m].
    n_pts : int, optional
        Number of transverse sample points.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - `'y'`: transverse coordinates [m],
        - `'n_profile'`: refractive index profile `n(y)`,
        - `'Hz'`: normalized magnetic field `H_z(y)`,
        - `'Ey'`: normalized normal electric field `E_y(y)` (`max|E_y| = 1`),
        - `'Dy_norm'`: continuous normalized displacement `(n(y)/n_low)^2 E_y(y)`,
        - `'neff'`: fundamental mode effective index,
        - `'discontinuity_ratio'`: `(n_high / n_low)^2` jump at `y = +-slot_width/2`,
        - `'slot_confinement'`: fraction of `|E_y|^2` integral inside `|y| <= slot_width/2`.
    """
    n_h = float(n_high)
    n_s = float(n_low)
    a = 0.5 * float(slot_width)
    w_r = float(rail_width)
    b = a + w_r
    neff = slot_waveguide_neff(n_h, n_s, slot_width, rail_width, wavelength)

    k0 = 2.0 * np.pi / float(wavelength)
    kh = k0 * np.sqrt(max(n_h**2 - neff**2, 0.0))
    gs = k0 * np.sqrt(max(neff**2 - n_s**2, 0.0))
    r = (n_h / n_s) ** 2

    y = np.linspace(-0.5 * float(y_span), 0.5 * float(y_span), int(n_pts))
    ay = np.abs(y)

    in_slot = ay <= a
    in_rail = (ay > a) & (ay <= b)
    in_clad = ay > b

    # Continuity of Hz and (1/n^2) dHz/dy at |y| = a and |y| = b
    B = np.cosh(gs * a)
    C = r * (gs / max(kh, 1e-30)) * np.sinh(gs * a)
    D = B * np.cos(kh * w_r) + C * np.sin(kh * w_r)

    hz = np.empty_like(y)
    hz[in_slot] = np.cosh(gs * ay[in_slot])
    hz[in_rail] = B * np.cos(kh * (ay[in_rail] - a)) + C * np.sin(
        kh * (ay[in_rail] - a)
    )
    hz[in_clad] = D * np.exp(-gs * (ay[in_clad] - b))

    n_prof = np.where(in_rail, n_h, n_s)
    # Ey(y) is proportional to Hz(y) / n^2(y)
    ey_raw = hz / (n_prof**2)
    scale = float(np.max(np.abs(ey_raw))) or 1.0
    ey = ey_raw / scale
    hz_norm = hz / ( float(np.max(np.abs(hz))) or 1.0 )
    dy_norm = ((n_prof / n_s) ** 2) * ey

    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    slot_conf = float(np.sum(ey[in_slot] ** 2) * dy / (np.sum(ey**2) * dy))

    return {
        "y": y,
        "n_profile": n_prof,
        "Hz": hz_norm,
        "Ey": ey,
        "Dy_norm": dy_norm,
        "neff": neff,
        "discontinuity_ratio": r,
        "slot_confinement": slot_conf,
    }
