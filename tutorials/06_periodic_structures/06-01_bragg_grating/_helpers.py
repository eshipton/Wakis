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

    def __init__(self, vertices, z_min, z_max, material, name="Si EIM core", **kwargs):
        super().__init__(vertices, z_min, z_max, material, **kwargs)
        self.name = name
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
    in_core_y = (y_nodes[1:] > y0 - 0.5 * w_core + tol) & (
        y_nodes[:-1] < y0 + 0.5 * w_core - tol
    )
    dy_core_nm = (
        float(np.max(dy[in_core_y]) * 1e9)
        if np.any(in_core_y)
        else float(np.max(dy) * 1e9)
    )
    return {
        "min_feature_nm": round(dw * 1e9, 2),
        "dx_background_nm": round(dx_bg_nm, 2),
        "dx_min_nm": round(dx_min_nm, 2),
        "dx_min_x_nm": round(dx_min_x_nm, 2),
        "dy_min_nm": round(dy_min_nm, 2),
        "dy_core_nm": round(dy_core_nm, 2),
        "cells_across_feature": round(cells_across_feature, 2),
        "cells_across_tooth_x": round(cells_across_tooth_x, 2),
        "n_cells": n_cells,
        "grid_shape": [int(len(dx)), int(len(dy)), int(len(dz))],
    }


def build_tuned_bragg_mesh_overrides(
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
    """Construct tuned graded non-uniform ``mesh_overrides`` for a Bragg grating.

    Refines the sidewall tooth bands (``|y - y0| in [0.5*w_core - 0.5*dw,
    0.5*w_core + 1.5*dw]``) to ``hy_tooth = dw / n_feature``, the grating
    longitudinal extent (``x in [-L/2, +L/2]``) to isotropic ``hx_tooth =
    hy_tooth``, and the inner waveguide core (``|y - y0| <= 0.5*w_core -
    0.5*dw``) to ``hy_core = 2 * hy_tooth``, while grading smoothly into the
    cladding background cell size ``dx_background`` [m].

    Parameters
    ----------
    n_periods : int, optional
        Number of grating periods ``N`` [dimensionless]. Default is 100.
    pitch : float, optional
        Grating period ``Lambda`` [m]. Default is ``0.324e-6``.
    w_core : float, optional
        Inner uncorrugated core width ``w_core`` [m]. Default is ``0.45e-6``.
    dw : float, optional
        Sidewall tooth depth ``Delta_w`` per side [m]. Default is ``0.05e-6``.
    duty_cycle : float, optional
        Tooth duty cycle ``f_dc`` in ``(0, 1)`` [dimensionless]. Default is 0.5.
    shift : float, optional
        Longitudinal shift of the bottom teeth [m]. Default is 0.0.
    y0 : float, optional
        Transverse waveguide centerline coordinate [m]. Default is 0.0.
    n_feature : int or float, optional
        Target number of cells across the ``dw`` tooth depth. Default is 4.
    dx_background : float or None, optional
        Background slab cell size [m].

    Returns
    -------
    overrides : list of tuple
        List of ``(lo, hi, (hx, hy, hz))`` override boxes in meters [m].
    """
    if n_feature is None or float(n_feature) <= 0:
        return []
    n_feat = float(n_feature)
    dw = float(dw)
    w_core = float(w_core)
    pitch = float(pitch)
    length = float(n_periods) * pitch

    hy_tooth = dw / n_feat
    hx_tooth = hy_tooth
    hy_core = 2.0 * hy_tooth
    if dx_background is not None:
        hy_tooth = min(float(dx_background), hy_tooth)
        hx_tooth = min(float(dx_background), hx_tooth)
        hy_core = min(float(dx_background), hy_core)

    y_in = 0.5 * w_core - 0.5 * dw
    y_out = 0.5 * w_core + 1.5 * dw
    x_lo = -0.5 * length
    x_hi = +0.5 * length + max(0.0, float(shift))

    return [
        ((-np.inf, y0 - y_out, -np.inf), (np.inf, y0 - y_in, np.inf), (1.0, hy_tooth, 1.0)),
        ((-np.inf, y0 + y_in, -np.inf), (np.inf, y0 + y_out, np.inf), (1.0, hy_tooth, 1.0)),
        ((-np.inf, y0 - y_in, -np.inf), (np.inf, y0 + y_in, np.inf), (1.0, hy_core, 1.0)),
        ((x_lo, -np.inf, -np.inf), (x_hi, np.inf, np.inf), (hx_tooth, 1.0, 1.0)),
    ]


def build_uniform_bragg_mesh_overrides(
    dw=0.05e-6,
    n_feature=4,
    dx_background=None,
):
    """Construct a full-domain uniform-cell ``mesh_overrides`` box at tooth resolution.

    Applies ``hx = hy = dw / n_feature`` uniformly across the entire computational
    domain (including the cladding and access waveguides) to benchmark uniform
    meshing against graded non-uniform meshing at equal tooth resolution.

    Parameters
    ----------
    dw : float, optional
        Sidewall tooth depth ``Delta_w`` [m]. Default is ``0.05e-6``.
    n_feature : int or float, optional
        Target number of cells across ``dw``. Default is 4.
    dx_background : float or None, optional
        Background slab cell size [m].

    Returns
    -------
    overrides : list of tuple
        Single full-domain override box ``(lo, hi, (h, h, 1.0))`` [m].
    """
    if n_feature is None or float(n_feature) <= 0:
        return []
    h = float(dw) / float(n_feature)
    if dx_background is not None:
        h = min(float(dx_background), h)
    return [
        ((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf), (h, h, 1.0)),
    ]


def compare_bragg_mesh_strategies(
    n_periods=100,
    pitch=0.324e-6,
    w_core=0.45e-6,
    dw=0.05e-6,
    duty_cycle=0.5,
    cpw=14,
    n_feature=6,
    n_pml=6,
    wl_min=1500e-9,
    n_eff_slab=2.8335,
):
    """Compare uniform coarse, sidewall-strip graded, tuned graded, and uniform fine meshes.

    Parameters
    ----------
    n_periods : int, optional
        Number of grating periods ``N`` [dimensionless]. Default is 100.
    pitch : float, optional
        Grating period ``Lambda`` [m]. Default is ``0.324e-6``.
    w_core : float, optional
        Inner core width ``w_core`` [m]. Default is ``0.45e-6``.
    dw : float, optional
        Sidewall tooth depth ``Delta_w`` [m]. Default is ``0.05e-6``.
    duty_cycle : float, optional
        Tooth duty cycle ``f_dc`` [dimensionless]. Default is 0.5.
    cpw : float, optional
        Background cells per guided wavelength [dimensionless]. Default is 14.
    n_feature : int or float, optional
        Target cells across the ``dw`` tooth depth. Default is 6.
    n_pml : int, optional
        Number of CPML boundary layers. Default is 6.
    wl_min : float, optional
        Minimum wavelength [m]. Default is ``1500e-9``.
    n_eff_slab : float, optional
        Vertical slab effective index [dimensionless]. Default is 2.8335.

    Returns
    -------
    comparison : dict
        Dictionary mapping strategy names (``"uniform_coarse"``,
        ``"sidewall_graded"``, ``"tuned_graded"``, ``"uniform_fine"``) to their
        measured mesh metrics and cell-count reduction factors relative to
        ``"uniform_fine"``.
    """
    from scipy.constants import c as c_light

    from wakis.templates import meshing
    from wakis.templates.photonics import Box
    from wakis.templates.simulation import Simulation

    length = float(n_periods) * float(pitch)
    domain_x = 0.5 * length + 0.80e-6
    domain_y = 0.72e-6
    f_max = c_light / float(wl_min)
    f_min = c_light / (float(wl_min) * 1.08)
    dx_bg = float(wl_min) / (float(n_eff_slab) * float(cpw))
    h_core = meshing.cell_size(f_max, float(n_eff_slab) ** 2, float(cpw))
    domain = (-domain_x, +domain_x, -domain_y, +domain_y, -0.5 * h_core, +0.5 * h_core)

    bg_box = Box(
        center=(0.0, 0.0, 0.0),
        size=(2.0 * domain_x + 10e-6, 2.0 * domain_y + 10e-6, 4.0 * h_core),
        material=1.0,
    )
    verts = bragg_grating(
        n_periods=n_periods,
        pitch=pitch,
        w_core=w_core,
        dw=dw,
        duty_cycle=duty_cycle,
    )
    slab = FastPolySlab(
        verts, z_min=-0.5 * h_core, z_max=+0.5 * h_core, material=float(n_eff_slab) ** 2
    )
    structs = [bg_box, slab]
    boundaries = (["cpml", "cpml", "periodic"], ["cpml", "cpml", "periodic"])

    strategies = {
        "uniform_coarse": [],
        "sidewall_graded": build_bragg_mesh_overrides(
            n_periods=n_periods,
            pitch=pitch,
            w_core=w_core,
            dw=dw,
            duty_cycle=duty_cycle,
            n_feature=n_feature,
            dx_background=dx_bg,
        ),
        "tuned_graded": build_tuned_bragg_mesh_overrides(
            n_periods=n_periods,
            pitch=pitch,
            w_core=w_core,
            dw=dw,
            duty_cycle=duty_cycle,
            n_feature=n_feature,
            dx_background=dx_bg,
        ),
        "uniform_fine": build_uniform_bragg_mesh_overrides(
            dw=dw,
            n_feature=n_feature,
            dx_background=dx_bg,
        ),
    }

    out = {}
    for name, ov in strategies.items():
        sim = Simulation(
            f_min=f_min,
            f_max=f_max,
            structures=structs,
            boundaries=boundaries,
            domain=domain,
            cells_per_wavelength=float(cpw),
            mesh_overrides=ov,
            padding=0.0,
            n_pml=int(n_pml),
        )
        x_n, y_n, z_n = sim._axis(0), sim._axis(1), sim._axis(2)
        meta = measure_grating_mesh_metrics(
            x_n,
            y_n,
            z_n,
            dw=dw,
            n_periods=n_periods,
            pitch=pitch,
            w_core=w_core,
            duty_cycle=duty_cycle,
            n_pml=n_pml,
        )
        out[name] = meta

    n_unif = max(int(out["uniform_fine"]["n_cells"]), 1)
    for meta in out.values():
        meta["cell_reduction_vs_uniform_fine"] = round(n_unif / max(int(meta["n_cells"]), 1), 3)
    return out


def load_graded_vs_uniform_study_summary(repo_root=None):
    """Load the 100-period graded vs uniform mesh benchmark summary.

    Reads ``studies/subgrid-vs-graded-bragg/data/analysis_full.json`` when
    present and returns a structured dictionary comparing the baseline
    sidewall-strip graded mesh (``graded_t18``), the tuned core-and-tooth
    graded mesh (``graded_tuned``), and the uniform fine mesh (``uniform_N4``)
    in Bragg wavelength ``lambda_B_nm``, ``-3 dB`` stopband width
    ``bandwidth_3db_nm``, cell count ``n_cells``, time steps ``n_steps``, and
    wall time ``total_wall_time_s``.
    """
    import json
    from pathlib import Path

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    study_json = (
        Path(repo_root)
        / "studies"
        / "subgrid-vs-graded-bragg"
        / "data"
        / "analysis_full.json"
    )
    if study_json.exists():
        raw = json.loads(study_json.read_text(encoding="utf-8"))
        by_name = {t["name"]: t for t in raw.get("tasks", [])}
        return {
            "reference": raw.get("reference", {}),
            "richardson_limits_nm": raw.get("richardson_limits_nm", {}),
            "richardson_limits_bandwidth_nm": raw.get("richardson_limits_bandwidth_nm", {}),
            "uniform_N4": by_name.get("uniform_N4", raw.get("uniform_N4", {})),
            "graded_tuned": [
                by_name[k]
                for k in ("graded_tuned_L1", "graded_tuned_L2", "graded_tuned_L3", "graded_tuned_R")
                if k in by_name
            ],
            "graded_t18": [
                by_name[k]
                for k in ("graded_t18_L1", "graded_t18_L2", "graded_t18_L3", "graded_t18_R")
                if k in by_name
            ],
        }
    return {
        "reference": {"lambda_B_ref_nm": 1551.3084, "BW_ref_nm": 30.4094},
        "richardson_limits_nm": {"graded_tuned": 1551.3105, "graded_t18": 1551.4176},
        "richardson_limits_bandwidth_nm": {"graded_tuned": 30.3757, "graded_t18": 30.4233},
        "uniform_N4": {
            "name": "uniform_N4",
            "n_tooth_actual": 4.063,
            "dy_tooth_nm": 12.3077,
            "dx_tooth_nm": 12.3681,
            "dy_core_nm": 12.3077,
            "nx": 2769,
            "ny": 137,
            "n_cells": 379353,
            "n_steps": 68757,
            "total_wall_time_s": 1095.102,
            "lambda_B_nm": 1551.4229,
            "bandwidth_3db_nm": 30.6448,
            "err_lambda_B_nm": 0.1145,
            "err_bw_pct": 0.7741,
        },
        "graded_tuned": [
            {
                "name": "graded_tuned_L1",
                "n_tooth_actual": 4.0,
                "dy_tooth_nm": 12.5,
                "dx_tooth_nm": 12.5,
                "dy_core_nm": 24.9502,
                "nx": 2647,
                "ny": 74,
                "n_cells": 195878,
                "n_steps": 72820,
                "total_wall_time_s": 1516.5,
                "lambda_B_nm": 1553.0407,
                "bandwidth_3db_nm": 30.2867,
                "err_lambda_B_nm": 1.7323,
                "err_bw_pct": 0.4034,
            },
            {
                "name": "graded_tuned_L2",
                "n_tooth_actual": 6.0,
                "dy_tooth_nm": 8.3333,
                "dx_tooth_nm": 8.3333,
                "dy_core_nm": 16.6445,
                "nx": 3955,
                "ny": 96,
                "n_cells": 379680,
                "n_steps": 109039,
                "total_wall_time_s": 4617.53,
                "lambda_B_nm": 1552.1238,
                "bandwidth_3db_nm": 30.3142,
                "err_lambda_B_nm": 0.8154,
                "err_bw_pct": 0.313,
            },
            {
                "name": "graded_tuned_L3",
                "n_tooth_actual": 8.0,
                "dy_tooth_nm": 6.25,
                "dx_tooth_nm": 6.25,
                "dy_core_nm": 12.4877,
                "nx": 5269,
                "ny": 122,
                "n_cells": 642818,
                "n_steps": 145640,
                "total_wall_time_s": 8730.775,
                "lambda_B_nm": 1551.768,
                "bandwidth_3db_nm": 30.3411,
                "err_lambda_B_nm": 0.4596,
                "err_bw_pct": 0.2246,
            },
        ],
        "graded_t18": [],
    }


def build_bragg_setup_objects(
    n_periods=100,
    pitch=0.324e-6,
    w_core=0.45e-6,
    dw=0.05e-6,
    duty_cycle=0.5,
    shift=0.0,
    apodization="uniform",
    cpw=14,
    n_feature=6,
    n_pml=10,
    tuned_graded=False,
    n_core=3.47,
    n_clad=1.0,
    n_sub=1.44,
    h_si=220e-9,
    wl_min=1500e-9,
    wl_max=1620e-9,
    wl_ref=1532e-9,
):
    """Construct live Wakis ``PICSimulation``, ``SolverFIT3D``, ``ModeSource``, and ``ModeMonitor`` objects.

    Builds the 2D Effective-Index Method (EIM) Bragg grating setup without
    running time steps, suitable for passing directly to ``wakis.schematic``.
    """
    from wakis.mode_solver import ModeMonitor, ModeSource
    from wakis.templates.photonics import PICSimulation, Port, n_to_eps, slab_neff

    length = float(n_periods) * float(pitch)
    pad_x = 3.0e-6
    verts = bragg_grating(
        n_periods=n_periods,
        pitch=pitch,
        w_core=w_core,
        dw=dw,
        duty_cycle=duty_cycle,
        shift=shift,
        apodization=apodization,
        y0=0.0,
        pad_x=pad_x,
    )
    slab = FastPolySlab(
        verts,
        z_min=0.0,
        z_max=float(h_si),
        material=n_to_eps(float(n_core)),
    )
    x_port = 0.5 * length + 0.40e-6
    domain_x = 0.5 * length + 0.80e-6
    domain_y = 0.72e-6
    ports = [
        Port("in", center=(-x_port, 0.0, 0.5 * float(h_si)), size=(0.0, 1.10e-6, 0.8e-6), direction=+1),
        Port("out", center=(+x_port, 0.0, 0.5 * float(h_si)), size=(0.0, 1.10e-6, 0.8e-6), direction=+1),
    ]
    domain = (-domain_x, +domain_x, -domain_y, +domain_y)
    neff_slab = float(slab_neff(n_core, n_clad, h_si, wl_ref, n_sub=n_sub, polarization="TE"))
    dx_bg = float(wl_min) / (neff_slab * float(cpw))
    ov_builder = build_tuned_bragg_mesh_overrides if tuned_graded else build_bragg_mesh_overrides
    mesh_overrides = ov_builder(
        n_periods=n_periods,
        pitch=pitch,
        w_core=w_core,
        dw=dw,
        duty_cycle=duty_cycle,
        shift=shift,
        n_feature=n_feature,
        dx_background=dx_bg,
    )
    pic = PICSimulation(
        wavelength_min=wl_min,
        wavelength_max=wl_max,
        n_freqs=21,
        structures=[slab],
        ports=ports,
        n_core=n_core,
        n_clad=n_clad,
        n_sub=n_sub,
        wavelength_ref=wl_ref,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=cpw,
        domain=domain,
        n_pml=n_pml,
        cfln=0.85,
        mesh_overrides=mesh_overrides,
    )
    in_port = pic.port_map["in"]
    in_center, in_size, in_inward = pic._port_plane_args(in_port)
    nodes_x = pic.grid.x
    k_mon = int(np.argmin(np.abs(nodes_x[:-1] - in_center[0])))
    k_src = int(np.clip(k_mon - 2 * in_inward, 2, len(nodes_x) - 3))
    src_center = in_center.copy()
    src_center[0] = nodes_x[k_src]
    src = ModeSource(
        pic.sim.pulse,
        center=src_center,
        size=in_size,
        direction="+",
        mode_index=0,
        mode_frequency=pic.f_center,
    )
    src.normal = 0
    src.index = k_src
    src.label = f"ModeSource (TE₀, x = {src_center[0] * 1e6:.1f} µm)"
    monitors = []
    for p in pic.ports:
        p_center, p_size, _ = pic._port_plane_args(p)
        mon = ModeMonitor(
            pic.frequencies,
            center=p_center,
            size=p_size,
            n_modes=1,
            mode_frequency=pic.f_center,
        )
        mon.name = f"Port '{p.name}'"
        mon.label = f"ModeMonitor '{p.name}' (x = {p_center[0] * 1e6:.1f} µm)"
        monitors.append(mon)

    slab.name = "Si EIM core"
    structures_schematic = [slab]
    return pic, src, monitors, structures_schematic


class _PolygonSchematicProxy:
    """Proxy exposing ``.vertices``, ``.material``, and ``.name`` without ``.bounds`` for ``wakis.schematic``."""

    def __init__(self, slab, material_eps: float, name: str):
        self.vertices = np.asarray(slab.vertices, dtype=float)
        self.material = float(material_eps)
        self.name = str(name)


class _SchematicSolverProxy:
    """Proxy around ``SolverFIT3D`` that hides ``ieps`` so ``wakis.schematic`` uses explicit ``structures``."""

    def __init__(self, solver, eps_clad: float = 1.0):
        self._solver = solver
        self.grid = solver.grid
        self.Nx = solver.Nx
        self.Ny = solver.Ny
        self.Nz = solver.Nz
        self.dt = solver.dt
        self.n_pml = getattr(solver, "n_pml", 10)
        self.bc_low = getattr(solver, "bc_low", ["pml", "pml", "periodic"])
        self.bc_high = getattr(solver, "bc_high", ["pml", "pml", "periodic"])
        self.bg = [float(eps_clad), 1.0, 0.0]
        self._assigned_structures = ()


def plot_bragg_setup_schematic(
    out_path,
    n_periods=100,
    pitch=0.324e-6,
    w_core=0.45e-6,
    dw=0.05e-6,
    duty_cycle=0.5,
):
    """Render ``00a_setup_geometry.png``, ``00b_setup_mesh_and_excitation.png``, and ``00_setup_table.md`` for Tutorial 6.1."""
    from pathlib import Path
    from matplotlib.patches import Polygon as MplPolygon, Rectangle
    from wakis import figstyle, schematic

    out_p = Path(out_path)
    out_dir = out_p if out_p.suffix == "" else out_p.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pic_full, src_full, mons_full, structs_full = build_bragg_setup_objects(
        n_periods=n_periods,
        pitch=pitch,
        w_core=w_core,
        dw=dw,
        duty_cycle=duty_cycle,
        cpw=14,
        n_feature=6,
        n_pml=10,
        tuned_graded=False,
    )
    neff_slab = float(pic_full.n_eff_slab)
    eps_slab = round(neff_slab**2, 2)
    solver_proxy = _SchematicSolverProxy(pic_full.solver, eps_clad=1.0)
    poly_proxy = _PolygonSchematicProxy(
        structs_full[0],
        material_eps=eps_slab,
        name=f"Si EIM core (n_eff,slab = {neff_slab:.4f})",
    )
    mat_names = {
        "background": "Air lateral/upper clad (n = 1.00, ε_r = 1.00; SiO₂ BOX n_sub = 1.44)",
        "region_0": f"Si EIM core (n_Si = 3.47, h = 220 nm → n_eff,slab = {neff_slab:.4f}, ε_r = {eps_slab:.2f})",
    }

    xmin, xmax = float(pic_full.grid.xmin), float(pic_full.grid.xmax)
    ymin, ymax = float(pic_full.grid.ymin), float(pic_full.grid.ymax)
    n_pml = int(getattr(pic_full.solver, "n_pml", 10))
    th_x_low = float(pic_full.grid.x[n_pml] - pic_full.grid.x[0])
    th_x_high = float(pic_full.grid.x[-2] - pic_full.grid.x[-2 - n_pml])
    th_y_low = float(pic_full.grid.y[n_pml] - pic_full.grid.y[0])
    th_y_high = float(pic_full.grid.y[-2] - pic_full.grid.y[-2 - n_pml])

    pulse_obj = pic_full.sim.pulse
    f0 = float(pulse_obj.frequency)
    fwidth = float(pulse_obj.fwidth)
    x_src = float(src_full.center[0])
    hy_port = 0.5 * float(src_full.size[1])
    x_in_mon = float(mons_full[0].center[0])
    x_out_mon = float(mons_full[1].center[0])

    sources_s = [
        {
            "kind": "sheet",
            "type_name": "ModeSource",
            "short_word": "ModeSrc",
            "label": f"Port 'in' ModeSource (TE₀, +x, x = {x_src * 1e6:.2f} µm)",
            "desc": f"Port 'in' ModeSource at x = {x_src * 1e6:.2f} µm (TE₀/EIM E_y, +x)",
            "pos_str": f"x = {x_src * 1e6:.2f} µm, y ∈ [{-hy_port * 1e6:.2f}, {+hy_port * 1e6:.2f}] µm",
            "polarization": "TE₀ (in-plane E_y, H_z)",
            "h_bounds": (x_src, x_src),
            "v_bounds": (-hy_port, +hy_port),
            "e_vec": (0.0, 1.0, 0.0),
            "k_vec": (1.0, 0.0, 0.0),
            "h_vec": (0.0, 0.0, 1.0),
            "pulse": {
                "f0": f0,
                "fwidth": fwidth,
                "t_peak": float(4.0 / (np.pi * fwidth)),
                "t_end": float(pulse_obj.end_time),
                "src_time": pulse_obj,
            },
        }
    ]
    monitors_s = [
        {
            "kind": "plane",
            "type_name": "ModeMonitor",
            "short_word": "in mon",
            "label": "Port 'in' ModeMonitor (S₁₁)",
            "desc": f"Port 'in' ModeMonitor at x = {x_in_mon * 1e6:.2f} µm (S₁₁ reflection)",
            "pos_str": f"x = {x_in_mon * 1e6:.2f} µm, y ∈ [{-hy_port * 1e6:.2f}, {+hy_port * 1e6:.2f}] µm",
            "freq_str": "185.1–199.9 THz (λ ∈ [1500, 1620] nm, 81 pts)",
            "frequencies": pic_full.frequencies,
            "h_bounds": (x_in_mon, x_in_mon),
            "v_bounds": (-hy_port, +hy_port),
        },
        {
            "kind": "plane",
            "type_name": "ModeMonitor",
            "short_word": "out mon",
            "label": "Port 'out' ModeMonitor (S₂₁)",
            "desc": f"Port 'out' ModeMonitor at x = {x_out_mon * 1e6:.2f} µm (S₂₁ transmission)",
            "pos_str": f"x = {x_out_mon * 1e6:.2f} µm, y ∈ [{-hy_port * 1e6:.2f}, {+hy_port * 1e6:.2f}] µm",
            "freq_str": "185.1–199.9 THz (λ ∈ [1500, 1620] nm, 81 pts)",
            "frequencies": pic_full.frequencies,
            "h_bounds": (x_out_mon, x_out_mon),
            "v_bounds": (-hy_port, +hy_port),
        },
        {
            "kind": "box",
            "type_name": "FluxBox",
            "short_word": "FluxBox",
            "label": "2D Poynting FluxBox",
            "desc": "Closed 2D Poynting FluxBox (4 faces inside CPML)",
            "pos_str": f"x ∈ [{(xmin + th_x_low) * 1e6:.2f}, {(xmax - th_x_high) * 1e6:.2f}] µm, y ∈ [{(ymin + th_y_low) * 1e6:.2f}, {(ymax - th_y_high) * 1e6:.2f}] µm",
            "freq_str": "185.1–199.9 THz (81 pts)",
            "frequencies": pic_full.frequencies,
            "h_bounds": (xmin + 1.15 * th_x_low, xmax - 1.15 * th_x_high),
            "v_bounds": (ymin + 1.15 * th_y_low, ymax - 1.15 * th_y_high),
        },
    ]
    variations = [
        {
            "name": "Periodic unit cell & Bragg Bloch condition",
            "details": "Λ = 324 nm (D = 50%, l_tooth = 162 nm, w_core = 450 nm, Δw = 50 nm), Bloch phase Δφ = β_B Λ = π at λ_B; ±z faces: Periodic (2D EIM)",
        },
        {
            "name": "Sidewall phase offset (task_9)",
            "details": "Aligned (Δx_shift = 0 → κ = κ₀, R_max > 99.7%) vs half-period shifted (Δx_shift = Λ/2 = 162 nm → κ = 0, R_max = 0.0003)",
        },
        {
            "name": "Parametric sweeps (tasks 10–14)",
            "details": "Period count N ∈ {25, 50, 100}, tooth depth Δw ∈ {25, 50, 75} nm, Gaussian apodization (α = 4.0, −10.3 dB sidelobe reduction)",
        },
        {
            "name": "Resolution & averaging ladder (tasks 0–8)",
            "details": "Kottke vs staircase across L1 (10 cpw, 4/tooth), L2 (14 cpw, 6/tooth), L3 (20 cpw, 8/tooth), and reference R (40 cpw, 16/tooth)",
        },
    ]

    title_geom = "Sidewall-Corrugated Silicon Waveguide Bragg Grating (2D EIM Setup, N = 100, Λ = 324 nm)"
    title_mesh = "Graded Non-Uniform Mesh (1× Tooth Zoom), Spatial Wave Packet & Broadband Excitation"

    res = schematic.save_setup_figures(
        solver_proxy,
        out_dir,
        prefix="00",
        plane="xy",
        structures=[poly_proxy],
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        wavelength=1532e-9,
        min_feature=dw,
        variations=variations,
        title_geometry=title_geom,
        title_mesh=title_mesh,
    )

    # Re-render 00a with aspect='auto' so the 34.0 µm × 1.44 µm waveguide and Bloch/periodic annotations are clearly legible
    fig_a, ax_a = schematic.plot_schematic(
        solver_proxy,
        plane="xy",
        structures=[poly_proxy],
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        wavelength=1532e-9,
        title=title_geom,
        width_in=8.2,
        height_in=4.2,
    )
    ax_a.set_aspect("auto")
    x0_um, x1_um = xmin * 1e6, xmax * 1e6
    y0_um, y1_um = ymin * 1e6, ymax * 1e6
    dy_um = y1_um - y0_um
    for txt in ax_a.texts:
        t_str = txt.get_text()
        if getattr(txt, "_schematic_role", None) == "dimension" and "34." in t_str:
            txt.set_position((0.5 * (x0_um + x1_um), y1_um + 0.165 * dy_um))
        elif "3 · out mon" in t_str:
            txt.set_position((15.1, -0.52))
        elif "2 · in mon" in t_str:
            txt.set_position((-15.0, -0.52))
        elif "1 · ModeSrc" in t_str:
            txt.set_position((-14.9, +0.54))
    ax_a.set_title(title_geom, pad=36, fontsize=11.0, fontweight="bold")
    leg_a = ax_a.get_legend()
    if leg_a is not None:
        leg_a.set_bbox_to_anchor((0.5, -0.32))
    for yb_s, th_y in ((y0_um, th_y_low * 1e6), (y1_um - th_y_high * 1e6, th_y_high * 1e6)):
        ax_a.add_patch(
            Rectangle(
                (x0_um, yb_s),
                x1_um - x0_um,
                th_y,
                facecolor="#CBD5E1",
                edgecolor="#64748B",
                hatch="///",
                alpha=0.36,
                lw=0.8,
                zorder=2,
            )
        )
    ax_a.text(
        0.0,
        0.0,
        f"Si EIM core (n_eff,slab = {neff_slab:.4f}, ε_r = {eps_slab:.2f}) | ±z faces: Periodic\n"
        f"Periodic unit cell: Λ = {pitch * 1e9:.0f} nm (N = {n_periods}, Δw = {dw * 1e9:.0f} nm), Bloch phase Δφ = β_B Λ = π",
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFFEE", edgecolor="#94A3B8", lw=0.7),
        zorder=9,
    )
    figstyle.save(fig_a, res["geometry_figure"], close=True, auto_legend_below=False)

    # Re-render 00b with aspect='auto' on ax_mesh, polygon overlay, and 1x zoom on a 50 nm × 162 nm sidewall tooth
    zoom_bounds = (-0.24e-6, +0.24e-6, +0.20e-6, +0.30e-6)
    fig_b, axes_b = schematic.plot_mesh_and_excitation(
        solver_proxy,
        plane="xy",
        structures=[poly_proxy],
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        mesh_every=18,
        zoom_bounds=zoom_bounds,
        wavelength=1532e-9,
        min_feature=dw,
        title=title_mesh,
        width_in=8.2,
        height_in=5.6,
    )
    ax_mesh = axes_b[0]
    ax_mesh.set_aspect("auto")
    verts_um = np.asarray(structs_full[0].vertices, dtype=float) * 1e6
    ax_mesh.add_patch(
        MplPolygon(
            verts_um,
            closed=True,
            facecolor="#DCEAF7",
            edgecolor="#1E293B",
            lw=0.9,
            alpha=0.82,
            zorder=2,
        )
    )
    for child_ax in list(fig_b.axes) + list(getattr(ax_mesh, "child_axes", [])):
        if getattr(child_ax, "_is_inset", False):
            child_ax.add_patch(
                MplPolygon(
                    verts_um,
                    closed=True,
                    facecolor="#93C5FD",
                    edgecolor="#1D4ED8",
                    lw=1.6,
                    alpha=0.65,
                    zorder=1.5,
                )
            )
    figstyle.save(fig_b, res["mesh_figure"], close=True, auto_legend_below=False)
    return res


def render_bragg_pulse_gif(
    x_m,
    y_m,
    frames,
    times_s,
    solver,
    src,
    monitors,
    structures_schematic,
    out_path,
    fps=10,
):
    """Render ``09_pulse_propagation.gif`` with ``wakis.figstyle`` and ``schematic.overlay_geometry``."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib.pyplot as plt
    from pathlib import Path
    from wakis import figstyle, schematic

    x_um = np.asarray(x_m, dtype=float) * 1e6
    y_um = np.asarray(y_m, dtype=float) * 1e6
    stride = 2 if len(frames) > 20 else 1
    frames_f = np.asarray(frames[::stride], dtype=np.float32)
    times_f = np.asarray(times_s[::stride], dtype=float)
    vmax = float(np.percentile(np.abs(frames_f), 99.2)) or 1.0
    v_um = structures_schematic[0].vertices * 1e6

    with figstyle.apply_style():
        fig, ax = figstyle.new_figure(nrows=1, ncols=1, width=7.2, height=3.1, dpi=88)
        im = ax.pcolormesh(
            x_um,
            y_um,
            frames_f[0].T / vmax,
            cmap=figstyle.CMAP_DIVERGING,
            vmin=-1.0,
            vmax=1.0,
            shading="auto",
        )
        schematic.overlay_geometry(
            ax,
            solver,
            plane="xy",
            unit="µm",
            structures=structures_schematic,
            sources=[src],
            monitors=monitors,
            show_boundaries=True,
            show_materials=True,
            show_sources=True,
            show_monitors=True,
            label_materials=False,
            outline_color="none",
        )
        ax.plot(
            np.append(v_um[:, 0], v_um[0, 0]),
            np.append(v_um[:, 1], v_um[0, 1]),
            color="#0f172a",
            lw=0.95,
            clip_on=True,
            zorder=6,
        )
        ax.text(
            -11.2,
            0.35,
            "Si EIM core (ε_r = 8.03)",
            fontsize=7.8,
            fontweight="bold",
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.82),
            zorder=8,
        )
        ax.set(
            aspect="auto",
            xlim=(float(x_um.min()), float(x_um.max())),
            ylim=(float(y_um.min()), float(y_um.max())),
            xlabel="x [µm]",
            ylabel="y [µm]",
        )
        title_obj = ax.set_title(
            f"Bragg grating pulse propagation Ey(x, y, t) — t = {float(times_f[0]) * 1e15:.0f} fs"
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(r"Normalized $E_y(x, y, t)$")
        figstyle.finish(fig)

        def _update(frame_idx):
            im.set_array((frames_f[frame_idx].T / vmax).ravel())
            title_obj.set_text(
                f"Bragg grating pulse propagation Ey(x, y, t) — t = {float(times_f[frame_idx]) * 1e15:.0f} fs"
            )
            return im, title_obj

        anim = FuncAnimation(
            fig, _update, frames=len(frames_f), interval=1000 // fps, blit=False
        )
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(out_p), writer=PillowWriter(fps=fps))
        plt.close(fig)
    return out_p



