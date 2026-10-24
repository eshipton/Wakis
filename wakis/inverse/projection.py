# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Subpixel-smoothed projection (SSP) of a filtered density, NumPy version.

Hammond, Oskooi, Hammond, Chen, Ralph & Johnson, "Unifying and accelerating
level-set and density-based topology optimization by subpixel-smoothed
projection", Opt. Express 33, 33620 (2025), arXiv:2503.20189, Eqs. (7) and
(10)-(12). Implemented here from those equations (see "Subpixel-smoothed
projection" in the physics guide); the JAX twin is
`wakis.inverse.jax_tools.smoothed_projection`.

Given a filtered density rho_f and the norm of its gradient |grad rho_f|,
the signed distance to the planar approximation of the level set
rho_f = eta is d = (eta - rho_f) / |grad rho_f| (Eq. 7). Within the
smoothing radius R of the level set the output blends the projections of
two points on either side of it,

    out = (1 - F(d)) P(rho_f - R g F(d)) + F(d) P(rho_f + R g (1 - F(d))),

with g = |grad rho_f|, P the tanh projection of steepness beta and the
twice-differentiable fill fraction (Eq. 10)

    F(d) = 1/2 - 15/16 x + 5/8 x^3 - 3/16 x^5,   x = d / R,

of the parabolic kernel k(r) ~ 1 - (r/R)^2. Elsewhere out = P(rho_f). At
beta = inf the output is exactly F(d) near interfaces and 0 or 1 elsewhere:
binary except for a layer about 2R thick, yet a smooth function of the
interface position.
"""

import numpy as np

# Smoothing radius in units of the sample spacing (Hammond et al. 2025):
# 2R slightly above the spacing, so that every interface position has at
# least one sample with |d| < R (otherwise the gradient vanishes when the
# interface sits between samples)
RADIUS_FACTOR = 0.55


def tanh_projection(x, beta, eta=0.5):
    """
    Smoothed Heaviside projection (Wang, Lazarov & Sigmund 2011).

    P(x) = (tanh(beta eta) + tanh(beta (x - eta)))
           / (tanh(beta eta) + tanh(beta (1 - eta))),
    a step at eta (P = 1 for x > eta) for beta = inf.

    Parameters
    ----------
    x : array_like
        Density.
    beta : float
        Steepness (> 0), may be np.inf.
    eta : float, optional
        Threshold. Default 0.5.

    Returns
    -------
    ndarray
        P(x).
    """
    x = np.asarray(x, dtype=float)
    if np.isinf(beta):
        return np.where(x > eta, 1.0, 0.0)
    if beta <= 0:
        raise ValueError("beta must be positive.")
    num = np.tanh(beta * eta) + np.tanh(beta * (x - eta))
    return num / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))


def tanh_projection_derivative(x, beta, eta=0.5):
    """dP/dx of `tanh_projection` (zero almost everywhere for beta = inf)."""
    x = np.asarray(x, dtype=float)
    if np.isinf(beta):
        return np.zeros_like(x)
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return beta / np.cosh(beta * (x - eta)) ** 2 / den


def fill_fraction(x):
    """
    F_2 of Hammond et al. (Eq. 10) at x = d / R: 1 for x <= -1, 0 for
    x >= 1, C^2 in between, F(-x) = 1 - F(x).
    """
    x = np.clip(np.asarray(x, dtype=float), -1.0, 1.0)
    return 0.5 - 15 / 16 * x + 5 / 8 * x**3 - 3 / 16 * x**5


def fill_fraction_derivative(x):
    """dF/dx = -15/16 (1 - x^2)^2 for |x| < 1, else 0."""
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) < 1, -15 / 16 * (1 - x**2) ** 2, 0.0)


def smoothed_projection_pointwise(rho_f, grad_norm, beta, eta=0.5, radius=0.55):
    """
    SSP of samples with filtered density rho_f and gradient norm grad_norm.

    Parameters
    ----------
    rho_f : array_like
        Filtered density at the samples.
    grad_norm : array_like
        |grad rho_f| at the samples [1 / length unit of `radius`].
    beta : float
        Projection steepness (> 0, np.inf allowed).
    eta : float, optional
        Threshold. Default 0.5.
    radius : float, optional
        Smoothing radius R [length unit]; use RADIUS_FACTOR times the
        largest spacing of the samples. Default 0.55 (unit spacing).

    Returns
    -------
    out, d_rho, d_grad : ndarray
        The projected density and its partial derivatives with respect to
        rho_f and grad_norm (zero where no smoothing is applied).
    """
    rho_f = np.asarray(rho_f, dtype=float)
    g = np.asarray(grad_norm, dtype=float)
    R = float(radius)
    plain = tanh_projection(rho_f, beta, eta)
    d_plain = tanh_projection_derivative(rho_f, beta, eta)
    g_safe = np.where(g > 0, g, 1.0)
    d = (eta - rho_f) / g_safe
    smooth = (g > 0) & (np.abs(d) < R)
    x = np.where(smooth, d / R, 0.0)
    F = fill_fraction(x)
    dF = fill_fraction_derivative(x)
    rm = rho_f - R * g * F
    rp = rho_f + R * g * (1 - F)
    Pm, Pp = tanh_projection(rm, beta, eta), tanh_projection(rp, beta, eta)
    dPm = tanh_projection_derivative(rm, beta, eta)
    dPp = tanh_projection_derivative(rp, beta, eta)
    blend = (1 - F) * Pm + F * Pp
    # out(rho_f, g, F(x(rho_f, g)))
    d_F = Pp - Pm - R * g * ((1 - F) * dPm + F * dPp)
    d_rho_explicit = (1 - F) * dPm + F * dPp
    d_g_explicit = R * F * (1 - F) * (dPp - dPm)
    # x = (eta - rho_f) / (R g): dx/drho_f = -1/(R g), dx/dg = -x/g
    d_rho = d_rho_explicit - d_F * dF / (R * g_safe)
    d_g = d_g_explicit - d_F * dF * x / g_safe
    out = np.where(smooth, blend, plain)
    return out, np.where(smooth, d_rho, d_plain), np.where(smooth, d_g, 0.0)


def _axes(shape, periodic):
    periodic = np.broadcast_to(np.asarray(periodic, dtype=bool), (len(shape),))
    return [(a, bool(periodic[a])) for a in range(len(shape)) if shape[a] > 1]


def _difference(x, axis, periodic):
    """Central difference along axis (np.gradient stencil; wrap if periodic)."""
    if periodic:
        return 0.5 * (np.roll(x, -1, axis) - np.roll(x, 1, axis))
    return np.gradient(x, axis=axis, edge_order=1)


def _difference_transpose(c, axis, periodic):
    """Transpose of `_difference` applied to the cotangent c."""
    if periodic:
        return 0.5 * (np.roll(c, 1, axis) - np.roll(c, -1, axis))
    c = np.moveaxis(c, axis, 0)
    out = np.zeros_like(c)
    out[1] += c[0]
    out[0] -= c[0]
    out[-1] += c[-1]
    out[-2] -= c[-1]
    out[2:] += 0.5 * c[1:-1]
    out[:-2] -= 0.5 * c[1:-1]
    return np.moveaxis(out, 0, axis)


def _gradient_norm(rho_f, spacing, periodic):
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (rho_f.ndim,))
    comps = {
        a: _difference(rho_f, a, p) / spacing[a]
        for a, p in _axes(rho_f.shape, periodic)
    }
    norm = np.sqrt(sum(c**2 for c in comps.values())) if comps else np.zeros_like(rho_f)
    return norm, comps, spacing


def smoothed_projection(rho_f, beta, eta=0.5, spacing=1.0, radius=None, periodic=False):
    """
    Subpixel-smoothed projection of a filtered density on a regular grid.

    The gradient of rho_f is taken by central differences (one-sided at
    non-periodic ends, as np.gradient); axes of length 1 are ignored.

    Parameters
    ----------
    rho_f : array_like
        Filtered density on the grid.
    beta : float
        Projection steepness (> 0, np.inf allowed).
    eta : float, optional
        Threshold. Default 0.5.
    spacing : float or sequence of float, optional
        Grid spacing per axis [length unit]. Default 1 (pixels).
    radius : float, optional
        Smoothing radius R [length unit]. Default RADIUS_FACTOR times the
        largest spacing of the non-singleton axes.
    periodic : bool or sequence of bool, optional
        Periodic axes. Default False.

    Returns
    -------
    ndarray
        Projected density, same shape as rho_f.
    """
    rho_f = np.asarray(rho_f, dtype=float)
    norm, _, spacing = _gradient_norm(rho_f, spacing, periodic)
    R = _default_radius(rho_f.shape, spacing) if radius is None else radius
    return smoothed_projection_pointwise(rho_f, norm, beta, eta, R)[0]


def smoothed_projection_vjp(
    rho_f, cotangent, beta, eta=0.5, spacing=1.0, radius=None, periodic=False
):
    """
    Pull a cotangent dF/d(out) of `smoothed_projection` back to rho_f.

    Returns
    -------
    ndarray
        dF/drho_f, same shape as rho_f.
    """
    rho_f = np.asarray(rho_f, dtype=float)
    c = np.asarray(cotangent, dtype=float)
    norm, comps, spacing = _gradient_norm(rho_f, spacing, periodic)
    R = _default_radius(rho_f.shape, spacing) if radius is None else radius
    _, d_rho, d_g = smoothed_projection_pointwise(rho_f, norm, beta, eta, R)
    grad = c * d_rho
    cg = c * d_g / np.where(norm > 0, norm, 1.0)  # d|g| / dg_a = g_a / |g|
    periodic_axes = dict(_axes(rho_f.shape, periodic))
    for a, comp in comps.items():
        grad = grad + _difference_transpose(cg * comp, a, periodic_axes[a]) / spacing[a]
    return grad


def _default_radius(shape, spacing):
    active = [spacing[a] for a in range(len(shape)) if shape[a] > 1]
    return RADIUS_FACTOR * (max(active) if active else 1.0)
