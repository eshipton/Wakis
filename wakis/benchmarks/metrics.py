# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Error metrics of the benchmark suite.

All metrics compare a simulated spectrum y(lambda) with the reference
y_ref(lambda) sampled on the same wavelengths.

* `nrms`: RMS error normalised by the reference peak,
  sqrt(mean (y - y_ref)^2) / max |y_ref|. Unlike the pointwise relative error
  it stays meaningful where the spectrum is small (metal particles far from
  resonance).
* `rms_rel`: RMS of the pointwise relative error (the metric of the n = 2
  Mie case B1, kept for comparability with earlier studies).
* `peak`: peak wavelength (parabola through the three largest samples),
  peak height (parabola value) and full width at half maximum (linear
  interpolation of the half-maximum crossings; NaN when a crossing falls
  outside the band).
* `peak_errors`: relative errors of those three quantities.
"""

import numpy as np

__all__ = ["nrms", "rms_rel", "peak", "peak_errors", "interp_to_target"]


def nrms(y, y_ref):
    """sqrt(mean((y - y_ref)^2)) / max|y_ref|."""
    y, y_ref = np.asarray(y, float), np.asarray(y_ref, float)
    return float(np.sqrt(np.mean((y - y_ref) ** 2)) / np.max(np.abs(y_ref)))


def rms_rel(y, y_ref):
    """sqrt(mean((y / y_ref - 1)^2))."""
    y, y_ref = np.asarray(y, float), np.asarray(y_ref, float)
    return float(np.sqrt(np.mean((y / y_ref - 1) ** 2)))


def peak(lam, y):
    """
    Peak position, height and FWHM of a sampled spectrum.

    Parameters
    ----------
    lam : array_like
        Monotonic wavelengths (or any abscissa).
    y : array_like
        Spectrum.

    Returns
    -------
    dict
        ``lam_peak``, ``height``, ``fwhm`` (NaN if the half maximum is not
        crossed on both sides inside the band), ``interior`` (False when the
        maximum sits on a band edge).
    """
    lam, y = np.asarray(lam, float), np.asarray(y, float)
    if lam[0] > lam[-1]:
        lam, y = lam[::-1], y[::-1]
    i = int(np.argmax(y))
    interior = 0 < i < len(y) - 1
    if interior:
        x0, x1, x2 = lam[i - 1 : i + 2]
        y0, y1, y2 = y[i - 1 : i + 2]
        # parabola through three (possibly non-uniform) points
        d = (x0 - x1) * (x0 - x2) * (x1 - x2)
        A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
        B = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / d
        C = (
            x1 * x2 * (x1 - x2) * y0
            + x2 * x0 * (x2 - x0) * y1
            + x0 * x1 * (x0 - x1) * y2
        ) / d
        xp = -B / (2 * A) if A < 0 else x1
        xp = float(np.clip(xp, x0, x2))
        yp = float(A * xp**2 + B * xp + C) if A < 0 else float(y1)
    else:
        xp, yp = float(lam[i]), float(y[i])
    half = yp / 2
    fwhm = np.nan
    lo = i
    while lo > 0 and y[lo - 1] > half:
        lo -= 1
    hi = i
    while hi < len(y) - 1 and y[hi + 1] > half:
        hi += 1
    if lo > 0 and hi < len(y) - 1:
        xl = np.interp(half, [y[lo - 1], y[lo]], [lam[lo - 1], lam[lo]])
        xh = np.interp(half, [y[hi + 1], y[hi]], [lam[hi + 1], lam[hi]])
        fwhm = float(xh - xl)
    return dict(lam_peak=xp, height=yp, fwhm=fwhm, interior=bool(interior))


def peak_errors(lam, y, y_ref):
    """Relative errors of the peak wavelength, height and FWHM."""
    p, r = peak(lam, y), peak(lam, y_ref)
    return dict(
        lam_peak=abs(p["lam_peak"] / r["lam_peak"] - 1),
        height=abs(p["height"] / r["height"] - 1),
        fwhm=abs(p["fwhm"] / r["fwhm"] - 1) if np.isfinite(r["fwhm"]) else np.nan,
    )


def interp_to_target(cost, err, target):
    """
    Cost from which the error stays at or below `target` (log-log
    interpolation).

    The ladder is ordered by cost; the crossing is taken after the *last*
    point above the target, so a non-monotone ladder (an accidentally good
    coarse point) does not count as converged.

    Parameters
    ----------
    cost, err : array_like
        Cost (unknowns, work or time) and error of a resolution ladder.
    target : float

    Returns
    -------
    float
        Interpolated cost; ``-cost[0]`` (negative: an upper bound) if the
        whole ladder is below target; NaN if the finest point is above it.
    """
    cost, err = np.asarray(cost, float), np.asarray(err, float)
    o = np.argsort(cost)
    cost, err = cost[o], err[o]
    above = np.flatnonzero(~(err <= target))
    if above.size == 0:
        return -float(cost[0])
    j = int(above[-1])
    if j == len(cost) - 1:
        return np.nan
    lc0, lc1 = np.log(cost[j]), np.log(cost[j + 1])
    le0, le1 = np.log(err[j]), np.log(max(err[j + 1], 1e-300))
    t = (np.log(target) - le0) / (le1 - le0)
    return float(np.exp(lc0 + t * (lc1 - lc0)))
