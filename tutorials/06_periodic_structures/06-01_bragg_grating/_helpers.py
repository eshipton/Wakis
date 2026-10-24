# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Tutorial 6.1 (Waveguide Bragg gratings):
parametric sidewall-corrugated waveguide polygon generator (uniform, phase-shifted,
and apodized) and Coupled-Mode Theory (CMT) analytical spectra and stopband metrics.
"""

import numpy as np

from wakis.templates.photonics import PolySlab


class FastPolySlab(PolySlab):
    """
    `PolySlab` subclass with 1D bounding-interval spatial culling in
    `_dist_and_normal_xy` for long multi-period structures (such as 100-period
    Bragg gratings with > 800 polygon edges). Produces bit-identical `fill` and
    `normal` results to `PolySlab` at ~5x higher speed.
    """

    def __init__(self, vertices, z_min, z_max, material, **kwargs):
        super().__init__(vertices, z_min, z_max, material, **kwargs)
        p0x = self._edge_p0[:, 0]
        p0y = self._edge_p0[:, 1]
        p1x = p0x + self._edge_seg[:, 0]
        self._edge_xmin = np.minimum(p0x, p1x)
        self._edge_xmax = np.maximum(p0x, p1x)
        self._cull_margin = float(self._hi[1] - self._lo[1]) + 1.0e-6

        # If all edges are axis-aligned and horizontal edges partition [xmin, xmax]
        # on top and bottom without holes/tilt, enable O(log N) scanline point-in-polygon.
        enx, eny = self._edge_normals[:, 0], self._edge_normals[:, 1]
        axis_aligned = np.all((np.abs(enx) > 0.9999) | (np.abs(eny) > 0.9999))
        self._rectilinear_monotone = False
        if axis_aligned and self._tan_angle == 0.0 and len(self.holes) == 0:
            m_bot = eny < -0.9999
            m_top = eny > 0.9999
            if np.any(m_bot) and np.any(m_top):
                bx0, bx1, by = self._edge_xmin[m_bot], self._edge_xmax[m_bot], p0y[m_bot]
                ob = np.argsort(bx0)
                bx0, bx1, by = bx0[ob], bx1[ob], by[ob]
                tx0, tx1, ty = self._edge_xmin[m_top], self._edge_xmax[m_top], p0y[m_top]
                ot = np.argsort(tx0)
                tx0, tx1, ty = tx0[ot], tx1[ot], ty[ot]
                if np.allclose(bx1[:-1], bx0[1:], atol=1e-13) and np.allclose(
                    tx1[:-1], tx0[1:], atol=1e-13
                ):
                    self._bot_x0, self._bot_x1, self._bot_y = bx0, bx1, by
                    self._top_x0, self._top_x1, self._top_y = tx0, tx1, ty
                    self._rectilinear_monotone = True

    def _contains_xy(self, pts, z=None):
        if not self._rectilinear_monotone:
            return super()._contains_xy(pts, z=z)
        px = pts[:, 0]
        py = pts[:, 1]
        ib = np.clip(
            np.searchsorted(self._bot_x1, px, side="right"), 0, len(self._bot_y) - 1
        )
        it = np.clip(
            np.searchsorted(self._top_x1, px, side="right"), 0, len(self._top_y) - 1
        )
        return (
            (px > self._bot_x0[0])
            & (px <= self._bot_x1[-1])
            & (py > self._bot_y[ib])
            & (py <= self._top_y[it])
        )

    def _dist_and_normal_xy(self, x_flat, y_flat, chunk_size=1024):
        m = x_flat.size
        if m == 0 or len(self._edge_p0) < 64:
            return super()._dist_and_normal_xy(x_flat, y_flat, chunk_size=chunk_size)
        min_d2 = np.empty(m, dtype=float)
        nx_out = np.empty(m, dtype=float)
        ny_out = np.empty(m, dtype=float)
        margin = self._cull_margin
        for start in range(0, m, chunk_size):
            end = min(start + chunk_size, m)
            xc = x_flat[start:end]
            yc = y_flat[start:end]
            xlo, xhi = float(np.min(xc)) - margin, float(np.max(xc)) + margin
            mask = (self._edge_xmax >= xlo) & (self._edge_xmin <= xhi)
            if not np.any(mask):
                mask = slice(None)
            p0x = self._edge_p0[mask, 0][None, :]
            p0y = self._edge_p0[mask, 1][None, :]
            sx = self._edge_seg[mask, 0][None, :]
            sy = self._edge_seg[mask, 1][None, :]
            l2 = self._edge_len2[mask][None, :]
            enx = self._edge_normals[mask, 0]
            eny = self._edge_normals[mask, 1]
            dx = xc[:, None] - p0x
            dy = yc[:, None] - p0y
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


def bragg_grating(
    n_periods=100,
    pitch=0.324e-6,
    w_core=0.45e-6,
    dw=0.05e-6,
    duty_cycle=0.5,
    shift=0.0,
    apodization="uniform",
    apod_alpha=4.0,
    y0=0.0,
    pad_x=3.0e-6,
):
    """
    Construct a sidewall-corrugated Bragg grating waveguide as a single closed
    counter-clockwise polygon `(M, 2)` [m] suitable for `PolySlab`.

    Parameters
    ----------
    n_periods : int, optional
        Number of grating periods `N`. Default 100.
    pitch : float, optional
        Grating period `Lambda` [m]. Default `0.324e-6`.
    w_core : float, optional
        Inner uncorrugated core strip width `w_wg` [m]. Default `0.45e-6`.
    dw : float, optional
        Corrugation tooth depth `w_corrug` on each sidewall [m]. For a uniform
        grating, the waveguide width alternates between `w_core` and
        `w_core + 2 * dw` (average width `w_core + dw` at 50% duty cycle).
        Default `0.05e-6`.
    duty_cycle : float, optional
        Fraction of each period occupied by a tooth, in `(0, 1)`. Default `0.5`.
    shift : float, optional
        Longitudinal shift [m] applied to the `-y` (bottom) sidewall teeth
        relative to the `+y` (top) sidewall teeth. Use `0.0` for an aligned
        (in-phase) grating and `0.5 * pitch` for a half-period-shifted
        (anti-symmetric, reflection-suppressed) grating. Default `0.0`.
    apodization : {'uniform', 'gaussian', 'raised_cosine'}, optional
        Longitudinal envelope profile `f(x)` applied to the sidewall modulation.
        For `'gaussian'` and `'raised_cosine'`, the modulation is applied
        symmetrically (`+-0.5 * f(x) * dw` around the mean half-width
        `0.5 * w_core + 0.5 * dw`) so the local DC average effective index
        remains constant along the grating. Default `'uniform'`.
    apod_alpha : float, optional
        Gaussian roll-off parameter `alpha` in `exp(-alpha * (2*x/L)**2)`.
        Default `4.0`.
    y0 : float, optional
        Transverse center coordinate `y` [m] of the waveguide. Default `0.0`.
    pad_x : float, optional
        Length [m] of the straight input and output feed waveguides extending
        beyond the grating ends `x = +-0.5 * n_periods * pitch`. Default `3.0e-6`.

    Returns
    -------
    vertices : ndarray of float, shape (M, 2)
        Counter-clockwise polygon vertices `(x, y)` [m].
    """
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1.")
    if not (0.0 < duty_cycle < 1.0):
        raise ValueError("duty_cycle must lie in (0, 1).")
    if dw < 0.0:
        raise ValueError("dw must be non-negative.")

    length = float(n_periods) * float(pitch)
    x0 = -0.5 * length
    x_in = x0 - float(pad_x)
    x_out = -x0 + float(pad_x)
    tooth_len = float(duty_cycle) * float(pitch)

    apod = apodization.lower()
    if apod not in ("uniform", "gaussian", "raised_cosine"):
        raise ValueError(f"Unknown apodization {apodization!r}.")

    def _envelope(xc):
        u = np.clip((xc - x0) / max(length, 1e-30), 0.0, 1.0)
        s = 2.0 * u - 1.0  # in [-1, 1]
        if apod == "uniform":
            return 1.0
        if apod == "gaussian":
            return float(np.exp(-float(apod_alpha) * s * s))
        return float(0.5 * (1.0 + np.cos(np.pi * s)))

    if apod == "uniform":
        hw_feed = 0.5 * float(w_core)
    else:
        hw_feed = 0.5 * float(w_core) + float(duty_cycle) * float(dw)

    # Bottom (-y) boundary from x_in to x_out (CCW)
    bot = [[x_in, y0 - hw_feed]]
    for i in range(int(n_periods)):
        xs = x0 + i * float(pitch) + float(shift)
        xe = xs + tooth_len
        xn = xs + float(pitch)
        env = _envelope(0.5 * (xs + xe))
        if apod == "uniform":
            y_trench = y0 - 0.5 * float(w_core)
            y_tooth = y_trench - env * float(dw)
            bot.extend(
                [
                    [xs, y_trench],
                    [xs, y_tooth],
                    [xe, y_tooth],
                    [xe, y_trench],
                ]
            )
        else:
            y_mean = y0 - hw_feed
            y_tooth = y_mean - 0.5 * env * float(dw)
            y_trench = y_mean + 0.5 * env * float(dw)
            bot.extend(
                [
                    [xs, y_mean],
                    [xs, y_tooth],
                    [xe, y_tooth],
                    [xe, y_trench],
                    [xn, y_trench],
                    [xn, y_mean],
                ]
            )
    bot.append([x_out, y0 - hw_feed])

    # Top (+y) boundary from x_out to x_in (CCW)
    top = [[x_out, y0 + hw_feed]]
    for i in range(int(n_periods) - 1, -1, -1):
        xs = x0 + i * float(pitch)
        xe = xs + tooth_len
        xn = xs + float(pitch)
        env = _envelope(0.5 * (xs + xe))
        if apod == "uniform":
            y_trench = y0 + 0.5 * float(w_core)
            y_tooth = y_trench + env * float(dw)
            top.extend(
                [
                    [xe, y_trench],
                    [xe, y_tooth],
                    [xs, y_tooth],
                    [xs, y_trench],
                ]
            )
        else:
            y_mean = y0 + hw_feed
            y_tooth = y_mean + 0.5 * env * float(dw)
            y_trench = y_mean - 0.5 * env * float(dw)
            top.extend(
                [
                    [xn, y_mean],
                    [xn, y_trench],
                    [xe, y_trench],
                    [xe, y_tooth],
                    [xs, y_tooth],
                    [xs, y_mean],
                ]
            )
    top.append([x_in, y0 + hw_feed])

    verts = np.asarray(bot + top, dtype=float)
    # Remove any consecutive duplicate points
    keep = np.ones(len(verts), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(verts, axis=0)) > 1e-14, axis=1)
    return verts[keep]


def cmt_coupling_from_neff_step(
    delta_neff, wavelength_bragg, duty_cycle=0.5, shift_over_pitch=0.0
):
    """
    First-order Coupled-Mode Theory (CMT) coupling coefficient `kappa` [1/m]
    for a rectangular step-index Bragg grating with sidewall phase offset.

    Parameters
    ----------
    delta_neff : float
        Effective-index step `n_eff(w_max) - n_eff(w_min)` between the tooth and
        trench cross-sections [dimensionless].
    wavelength_bragg : float
        Bragg resonance wavelength `lambda_B = 2 * n_eff * Lambda` [m].
    duty_cycle : float, optional
        Tooth duty cycle `D` in `(0, 1)`. Default `0.5`.
    shift_over_pitch : float, optional
        Normalized longitudinal shift `Delta_x / Lambda` between upper and lower
        sidewall teeth (`0.0` for aligned, `0.5` for half-period shifted).
        Default `0.0`.

    Returns
    -------
    kappa : float
        Forward-to-backward mode coupling coefficient `kappa` [1/m].
    """
    kappa_0 = (
        2.0 * abs(float(delta_neff)) / float(wavelength_bragg)
    ) * float(np.sin(np.pi * float(duty_cycle)))
    return float(kappa_0 * abs(np.cos(np.pi * float(shift_over_pitch))))


def cmt_bragg_spectrum(
    wavelengths,
    n_eff,
    pitch,
    kappa,
    length,
    n_g=None,
    wavelength_bragg=None,
):
    """
    Analytical Coupled-Mode Theory (CMT) reflection and transmission spectra for
    a uniform lossless Bragg grating of length `L` and coupling `kappa`.

    Parameters
    ----------
    wavelengths : array_like
        Vacuum wavelengths `lambda` [m].
    n_eff : float or array_like
        Average modal effective index `n_eff` (scalar or array matching
        `wavelengths`) [dimensionless].
    pitch : float
        Grating period `Lambda` [m].
    kappa : float
        Coupling coefficient `kappa` [1/m].
    length : float
        Total grating length `L = N * Lambda` [m].
    n_g : float, optional
        Modal group index `n_g = n_eff - lambda * dn_eff/dlambda`. When provided
        together with scalar `n_eff`, detuning is linearized around
        `wavelength_bragg` using `n_g`:
        `delta = -2 * pi * n_g * (lambda - lambda_B) / lambda_B**2`.
    wavelength_bragg : float, optional
        Explicit Bragg center wavelength `lambda_B` [m]. Defaults to
        `2 * mean(n_eff) * pitch`.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - `'wavelengths'`: wavelength array [m]
        - `'lambda_bragg'`: Bragg wavelength [m]
        - `'detuning'`: phase detuning `delta(lambda)` [1/m]
        - `'r'`: complex reflection amplitude `r(lambda)`
        - `'t'`: complex transmission amplitude `t(lambda)`
        - `'R'`: power reflectivity `|r(lambda)|**2`
        - `'T'`: power transmittance `|t(lambda)|**2`
        - `'R_max'`: peak reflectivity `tanh(kappa * L)**2`
        - `'null_bandwidth'`: analytical first-null bandwidth [m]
    """
    wl = np.asarray(wavelengths, dtype=float)
    neff_arr = np.asarray(n_eff, dtype=float)
    neff_mean = float(np.mean(neff_arr))
    lam_b = (
        float(wavelength_bragg)
        if wavelength_bragg is not None
        else 2.0 * neff_mean * float(pitch)
    )
    if n_g is not None and neff_arr.ndim == 0:
        ng_eff = float(n_g)
        delta = -2.0 * np.pi * ng_eff * (wl - lam_b) / (lam_b * wl)
    else:
        ng_eff = float(n_g) if n_g is not None else neff_mean
        delta = 2.0 * np.pi * neff_arr / wl - np.pi / float(pitch)

    kap = float(kappa)
    L = float(length)
    gamma = np.sqrt(np.astype(kap * kap - delta * delta, np.complex128))
    sinh_gL = np.sinh(gamma * L)
    cosh_gL = np.cosh(gamma * L)
    denom = gamma * cosh_gL + 1j * delta * sinh_gL
    r = np.where(np.abs(denom) > 1e-30, -1j * kap * sinh_gL / denom, 0.0)
    t = np.where(np.abs(denom) > 1e-30, gamma / denom, 1.0)
    R = np.real(r * np.conj(r))
    T = np.real(t * np.conj(t))
    r_max = float(np.tanh(kap * L) ** 2)
    null_bw = float(
        (lam_b**2 / (np.pi * ng_eff * max(L, 1e-30)))
        * np.sqrt((kap * L) ** 2 + np.pi**2)
    )
    return {
        "wavelengths": wl,
        "lambda_bragg": lam_b,
        "detuning": delta,
        "r": r,
        "t": t,
        "R": R,
        "T": T,
        "R_max": r_max,
        "null_bandwidth": null_bw,
    }


def extract_stopband_metrics(wavelengths, R, T, length, n_g=None):
    """
    Extract stopband resonance wavelength, peak reflection, minimum transmission,
    bandwidths, sidelobe level, and CMT coupling coefficient from simulated spectra.

    Parameters
    ----------
    wavelengths : array_like
        Wavelengths [m].
    R, T : array_like
        Power reflection `|S11|^2` and transmission `|S21|^2` spectra.
    length : float
        Grating length `L` [m].
    n_g : float, optional
        Group index `n_g` used to invert the first-null bandwidth for `kappa`.

    Returns
    -------
    metrics : dict
        Stopband characterization metrics in SI units.
    """
    wl = np.asarray(wavelengths, dtype=float)
    order = np.argsort(wl)
    wl = wl[order]
    r_arr = np.clip(np.asarray(R, dtype=float)[order], 0.0, 1.0)
    t_arr = np.clip(np.asarray(T, dtype=float)[order], 0.0, 1.0)

    # Avoid 2 boundary points at each end of the Gaussian pulse spectrum
    margin = min(3, max(0, len(wl) // 10))
    sl = slice(margin, len(wl) - margin if margin > 0 else None)
    idx_min_t = int(np.argmin(t_arr[sl])) + margin
    idx_max_r = int(np.argmax(r_arr[sl])) + margin
    # Use the minimum-transmission point as the stopband center (immune to small port ripples)
    i0 = idx_min_t
    lam_b = float(wl[i0])
    r_max = float(r_arr[idx_max_r])
    t_min = float(t_arr[idx_min_t])
    ext_db = float(-10.0 * np.log10(max(t_min, 1e-12)))

    # Half-maximum (-3 dB) bandwidth of reflection peak and transmission rejection
    half_r = 0.5 * r_max
    left_wl = float(wl[0])
    for i in range(idx_max_r, 0, -1):
        if r_arr[i - 1] <= half_r <= r_arr[i]:
            frac = (half_r - r_arr[i - 1]) / max(r_arr[i] - r_arr[i - 1], 1e-30)
            left_wl = float(wl[i - 1] + frac * (wl[i] - wl[i - 1]))
            break
    right_wl = float(wl[-1])
    for i in range(idx_max_r, len(wl) - 1):
        if r_arr[i + 1] <= half_r <= r_arr[i]:
            frac = (half_r - r_arr[i + 1]) / max(r_arr[i] - r_arr[i + 1], 1e-30)
            right_wl = float(wl[i + 1] - frac * (wl[i + 1] - wl[i]))
            break
    fwhm_bw = max(0.0, right_wl - left_wl)

    # For a flat-bottomed stopband, use the sub-grid midpoint of the -3 dB
    # transmission-rejection edges (1 - T = 0.5 * (1 - T_min)) so lambda_B is
    # immune to floor ripples and discrete frequency-grid quantization.
    rej = 1.0 - t_arr
    half_rej = 0.5 * float(rej[idx_min_t])
    if half_rej > 0.05:
        left_t = None
        for i in range(idx_min_t, margin, -1):
            if rej[i - 1] <= half_rej <= rej[i]:
                frac = (half_rej - rej[i - 1]) / max(rej[i] - rej[i - 1], 1e-30)
                left_t = float(wl[i - 1] + frac * (wl[i] - wl[i - 1]))
                break
        right_t = None
        for i in range(idx_min_t, len(wl) - margin - 1):
            if rej[i + 1] <= half_rej <= rej[i]:
                frac = (half_rej - rej[i + 1]) / max(rej[i] - rej[i + 1], 1e-30)
                right_t = float(wl[i + 1] - frac * (wl[i + 1] - wl[i]))
                break
        if left_t is not None and right_t is not None and right_t > left_t:
            lam_b = 0.5 * (left_t + right_t)
            fwhm_bw = float(right_t - left_t)

    # First-null bandwidth (first local minima of R to the left and right of i0)
    i_null_left = 0
    for i in range(idx_max_r - 1, 0, -1):
        if r_arr[i] <= r_arr[i - 1] and r_arr[i] <= r_arr[i + 1] and r_arr[i] < 0.25 * r_max:
            i_null_left = i
            break
    i_null_right = len(wl) - 1
    for i in range(idx_max_r + 1, len(wl) - 1):
        if r_arr[i] <= r_arr[i - 1] and r_arr[i] <= r_arr[i + 1] and r_arr[i] < 0.25 * r_max:
            i_null_right = i
            break
    null_bw = float(wl[i_null_right] - wl[i_null_left])

    # Highest sidelobe outside [i_null_left, i_null_right]
    sl_candidates = []
    if i_null_left > margin:
        sl_candidates.append(float(np.max(r_arr[margin:i_null_left])))
    if i_null_right < len(wl) - margin - 1:
        sl_candidates.append(float(np.max(r_arr[i_null_right + 1 : len(wl) - margin])))
    sl_max = max(sl_candidates) if sl_candidates else 0.0
    sidelobe_db = float(10.0 * np.log10(max(sl_max, 1e-12) / max(r_max, 1e-12)))

    # CMT coupling coefficient from R_max = tanh^2(kappa * L) and from 1 - T_min = tanh^2(kappa * L)
    L = max(float(length), 1e-30)
    r_eff = min(max(1.0 - t_min, r_max), 0.999999)
    kappa_rmax = float(np.arctanh(np.sqrt(max(r_eff, 0.0))) / L)
    kappa_t = float(np.arctanh(np.sqrt(np.clip(1.0 - t_min, 0.0, 0.999999))) / L)

    kappa_bw = float("nan")
    if n_g is not None and null_bw > 0:
        arg = (np.pi * float(n_g) * L * null_bw / (lam_b**2)) ** 2 - np.pi**2
        if arg > 0:
            kappa_bw = float(np.sqrt(arg) / L)

    return {
        "lambda_bragg": lam_b,
        "R_max": r_max,
        "T_min": t_min,
        "extinction_db": ext_db,
        "fwhm_bandwidth": fwhm_bw,
        "null_bandwidth": null_bw,
        "sidelobe_db": sidelobe_db,
        "kappa_from_rmax": kappa_rmax,
        "kappa_from_tmin": kappa_t,
        "kappa_from_bw": kappa_bw,
    }


def build_bragg_mesh_overrides(
    n_periods=100,
    pitch=0.324e-6,
    w_core=0.45e-6,
    dw=0.05e-6,
    duty_cycle=0.5,
    shift=0.0,
    y0=0.0,
    n_feature=4,
    dx_background=None,
):
    """
    Construct per-axis `mesh_overrides` boxes `(lo, hi, (hx, hy, hz))` [m]
    resolving the two sidewall tooth bands in `y` (`hy = dw / n_feature`,
    spanning from the core edge to the tooth tip plus 1 tooth depth of margin)
    and the grating extent in `x` (`hx = min(dx_background, l_tooth / n_feature)`).

    Returns an empty list when `n_feature` is `None` or `<= 0` (used for the
    under-resolved demonstration runs).
    """
    if n_feature is None or int(n_feature) <= 0:
        return []
    n_feat = float(n_feature)
    dw = float(dw)
    w_core = float(w_core)
    pitch = float(pitch)
    tooth_len = float(duty_cycle) * pitch
    length = float(n_periods) * pitch

    hy = dw / n_feat
    hx = tooth_len / n_feat
    if dx_background is not None:
        hy = min(float(dx_background), hy)
        hx = min(float(dx_background), hx)

    # Each tooth band spans [0.5*w_core, 0.5*w_core + dw] with 0.5*dw margin
    # on each side (1 tooth depth of total margin across the band).
    y_in = 0.5 * w_core - 0.5 * dw
    y_out = 0.5 * w_core + 1.5 * dw
    x_lo = -0.5 * length
    x_hi = +0.5 * length + max(0.0, float(shift))

    return [
        ((-np.inf, y0 - y_out, -np.inf), (np.inf, y0 - y_in, np.inf), (1.0, hy, 1.0)),
        ((-np.inf, y0 + y_in, -np.inf), (np.inf, y0 + y_out, np.inf), (1.0, hy, 1.0)),
        ((x_lo, -np.inf, -np.inf), (x_hi, np.inf, np.inf), (hx, 1.0, 1.0)),
    ]


def measure_grating_mesh_metrics(
    x_nodes,
    y_nodes,
    z_nodes,
    dw=0.05e-6,
    n_periods=100,
    pitch=0.324e-6,
    w_core=0.45e-6,
    duty_cycle=0.5,
    shift=0.0,
    y0=0.0,
    n_pml=6,
):
    """
    Measure the actual mesh resolution metrics (`min_feature_nm`,
    `dx_background_nm`, `dx_min_nm`, `cells_across_feature`, `n_cells`)
    directly from the grid node arrays `(x_nodes, y_nodes, z_nodes)` [m].
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)
    dx = np.diff(x_nodes)
    dy = np.diff(y_nodes)
    dz = np.diff(z_nodes)

    dw = float(dw)
    w_core = float(w_core)
    pitch = float(pitch)
    tooth_len = float(duty_cycle) * pitch
    length = float(n_periods) * pitch

    # Background slab cell size measured in the unrefined feed waveguide section
    sl_x = slice(int(n_pml), len(dx) - int(n_pml)) if len(dx) > 2 * int(n_pml) else slice(None)
    dx_bg_nm = float(np.max(dx[sl_x]) * 1e9)
    dx_min_nm = float(min(np.min(dx), np.min(dy)) * 1e9)
    dx_min_x_nm = float(np.min(dx) * 1e9)
    dy_min_nm = float(np.min(dy) * 1e9)

    # Cells across the smallest feature (sidewall tooth depth dw in y)
    tol = 1e-12
    y_hi_lo, y_hi_hi = y0 + 0.5 * w_core, y0 + 0.5 * w_core + dw
    y_lo_lo, y_lo_hi = y0 - 0.5 * w_core - dw, y0 - 0.5 * w_core
    in_tooth_y = (
        ((y_nodes[1:] > y_hi_lo + tol) & (y_nodes[:-1] < y_hi_hi - tol))
        | ((y_nodes[1:] > y_lo_lo + tol) & (y_nodes[:-1] < y_lo_hi - tol))
    )
    max_dy_tooth = float(np.max(dy[in_tooth_y])) if np.any(in_tooth_y) else float(np.max(dy))
    cells_across_feature = float(dw / max_dy_tooth)

    # Cells across the tooth length l_corrug in x
    x_lo = -0.5 * length
    x_hi = +0.5 * length + max(0.0, float(shift))
    in_grat_x = (x_nodes[1:] > x_lo + tol) & (x_nodes[:-1] < x_hi - tol)
    max_dx_tooth = float(np.max(dx[in_grat_x])) if np.any(in_grat_x) else float(np.max(dx))
    cells_across_tooth_x = float(tooth_len / max_dx_tooth)

    n_cells = int(len(dx) * len(dy) * len(dz))
    return {
        "min_feature_nm": round(dw * 1e9, 2),
        "dx_background_nm": round(dx_bg_nm, 2),
        "dx_min_nm": round(dx_min_nm, 2),
        "dx_min_x_nm": round(dx_min_x_nm, 2),
        "dy_min_nm": round(dy_min_nm, 2),
        "cells_across_feature": round(cells_across_feature, 2),
        "cells_across_tooth_x": round(cells_across_tooth_x, 2),
        "n_cells": n_cells,
        "grid_shape": [int(len(dx)), int(len(dy)), int(len(dz))],
    }

