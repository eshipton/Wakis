# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Analytical and post-processing helpers for Tutorial 5.9 (Waveguide-to-Ring
Coupling in 2D EIM).

Provides:
1. 1D isolated 2D-EIM channel waveguide eigenmode solver (`Ey`, `Hz`, `i*Ex`)
   and evanescent decay constants (`gamma_evan`, `gamma_slab`).
2. Coupled-mode theory (CMT) for waveguide-to-ring point couplers:
   - Numerical mode-overlap integral of two isolated 2D-EIM modes,
   - Closed-form weak-coupling CMT (`mu_0 * exp(-gamma_evan * g)`) and
     Gaussian curved-coupler interaction length `L_eff = sqrt(2*pi*r / gamma)`,
   - Exact 5-layer coupled-slab even/odd supermode dispersion capturing the
     sub-wavelength slot-field enhancement across narrow gaps (`g = 50 nm`).
3. Exponential decay constant fitting (`fit_exponential_decay`) against gap.
4. Conformally mapped bent-waveguide radial eigenmode solver
   (`bent_waveguide_radial_mode`, Heiblum & Harris 1975) and angled (`theta = 45 deg`)
   cross-section DFT field sampler (`sample_angled_drop_slice`) — addressing the
   angled/bent mode-port gap in `wakis.mode_solver`.
5. All-pass microring resonator transfer-function synthesis (`ring_resonator_metrics`)
   computing FSR, finesse, loaded quality factor `Q_L`, and critical coupling
   from the extracted single-pass coupler coefficients `(t(lambda), kappa(lambda))`.
"""

from pathlib import Path

from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light, epsilon_0
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

from wakis import figstyle, schematic
from wakis.sources import GaussianSource
from wakis.templates import Simulation
from wakis.templates.meshing import graded_axis
from wakis.templates.photonics import (
    PIC_MATERIALS,
    PICSimulation,
    make_movie,
    n_to_eps,
    plot_field_map,
    slab_neff,
)
from wakis.unstructured import subtri_circles_mesh


def isolated_eim_channel_mode(
    n_core_2d,
    n_clad=PIC_MATERIALS["SiO2"],
    width=0.50e-6,
    wavelength=1550e-9,
    y_span=5.0e-6,
    n_pts=4001,
):
    """
    Analytical 1D in-plane quasi-TE channel mode of a straight waveguide in 2D EIM.

    In the 2D effective-index method (`Ex`, `Ey`, `Hz`), a waveguide of width
    `width` along `y` propagating along `+x` has magnetic field `Hz(y)` parallel
    to `z` and dominant transverse electric field `Ey(y)` normal to the `y = +-w/2`
    sidewalls (1D TM-like boundary conditions along `y`: `Hz` and `(1/n^2) dHz/dy`
    continuous).

    Parameters
    ----------
    n_core_2d : float
        2D EIM core effective index `n_eff,slab` [dimensionless].
    n_clad : float, optional
        Cladding refractive index [dimensionless]. Default 1.444 (`SiO2`).
    width : float, optional
        Waveguide width [m]. Default `0.50e-6`.
    wavelength : float, optional
        Free-space wavelength [m]. Default `1550e-9`.
    y_span : float, optional
        Transverse coordinate span `[-y_span/2, +y_span/2]` [m]. Default `5.0e-6`.
    n_pts : int, optional
        Number of grid points along `y`. Default 4001.

    Returns
    -------
    mode : dict
        Dictionary with keys:
        - `'y'` : ndarray, transverse coordinates [m]
        - `'Ey'` : ndarray, unit-power normalized transverse electric field [V/m / sqrt(W/m)]
        - `'Hz'` : ndarray, unit-power normalized magnetic field [A/m / sqrt(W/m)]
        - `'iEx'` : ndarray, quadrature longitudinal electric field `i * Ex(y)`
        - `'neff'` : float, 2D EIM channel effective index
        - `'beta'` : float, propagation constant [rad/m]
        - `'kx'` : float, core transverse wavenumber [1/m]
        - `'gamma_evan'` : float, channel mode evanescent decay constant [1/m]
        - `'gamma_slab'` : float, 1D slab evanescent decay constant `k0 * sqrt(n_core_2d^2 - n_clad^2)` [1/m]
    """
    n1 = float(n_core_2d)
    n2 = float(n_clad)
    w = float(width)
    lam = float(wavelength)
    k0 = 2.0 * np.pi / lam
    omega = k0 * c_light

    neff = slab_neff(n1, n2, w, lam, polarization="TM")
    beta = k0 * neff
    kx = k0 * np.sqrt(max(n1**2 - neff**2, 0.0))
    gamma_evan = k0 * np.sqrt(max(neff**2 - n2**2, 0.0))
    gamma_slab = k0 * np.sqrt(max(n1**2 - n2**2, 0.0))

    y = np.linspace(-0.5 * float(y_span), 0.5 * float(y_span), int(n_pts))
    d2 = 0.5 * w
    in_core = np.abs(y) <= d2
    hz = np.where(
        in_core,
        np.cos(kx * y),
        np.cos(kx * d2) * np.exp(-gamma_evan * (np.abs(y) - d2)),
    )
    n_y = np.where(in_core, n1, n2)
    ey = (beta / (omega * epsilon_0 * n_y**2)) * hz
    i_ex = np.where(
        in_core,
        (-kx * np.sin(kx * y)) / (omega * epsilon_0 * n1**2),
        (-np.sign(y) * gamma_evan * np.cos(kx * d2) * np.exp(-gamma_evan * (np.abs(y) - d2)))
        / (omega * epsilon_0 * n2**2),
    )

    p0 = float(0.5 * np.trapezoid(ey * hz, y))
    scale = 1.0 / np.sqrt(max(p0, 1e-30))
    return {
        "y": y,
        "Ey": ey * scale,
        "Hz": hz * scale,
        "iEx": i_ex * scale,
        "neff": float(neff),
        "beta": float(beta),
        "kx": float(kx),
        "gamma_evan": float(gamma_evan),
        "gamma_slab": float(gamma_slab),
    }


def _coupled_slab_supermode_neffs(n1, n2, width, gap, wavelength):
    """
    Exact even and odd supermode effective indices `(neff_even, neff_odd)` of two
    symmetric 1D TM-in-y waveguides of width `width` separated by edge-to-edge `gap` [m].
    """
    k0 = 2.0 * np.pi / float(wavelength)
    ratio = (n1 / n2) ** 2

    def dispersion(neff, parity):
        kx = k0 * np.sqrt(max(n1**2 - neff**2, 1e-30))
        gm = k0 * np.sqrt(max(neff**2 - n2**2, 1e-30))
        arg = 0.5 * gm * gap
        t_in = np.tanh(arg) if parity == "even" else 1.0 / np.maximum(np.tanh(arg), 1e-15)
        phi_in = np.arctan2(ratio * gm * t_in, kx)
        phi_out = np.arctan2(ratio * gm, kx)
        return kx * width - phi_in - phi_out

    neff_iso = slab_neff(n1, n2, width, wavelength, polarization="TM")
    neff_even = float(brentq(dispersion, neff_iso, n1 - 1e-12, args=("even",)))
    neff_odd = float(brentq(dispersion, n2 + 1e-12, neff_iso, args=("odd",)))
    return neff_even, neff_odd


def cmt_coupling_vs_gap(
    gaps_m,
    radius=5.0e-6,
    width=0.50e-6,
    n_core=PIC_MATERIALS["Si"],
    n_clad=PIC_MATERIALS["SiO2"],
    thickness=220e-9,
    wavelength=1550e-9,
    n_core_2d=None,
):
    """
    Coupled-mode theory (CMT) and 5-layer supermode predictions for waveguide-to-ring
    point coupling across a sequence of edge-to-edge gaps `gaps_m` [m].

    Parameters
    ----------
    gaps_m : array_like
        Edge-to-edge coupling gaps at `x = 0` [m].
    radius : float, optional
        Mean ring radius [m]. Default `5.0e-6`.
    width : float, optional
        Bus and ring waveguide width [m]. Default `0.50e-6`.
    n_core, n_clad : float, optional
        3D core and cladding refractive indices.
    thickness : float, optional
        Silicon slab thickness [m]. Default `220e-9`.
    wavelength : float, optional
        Free-space wavelength [m]. Default `1550e-9`.
    n_core_2d : float, optional
        If provided, overrides `slab_neff(n_core, n_clad, thickness, wavelength)`.

    Returns
    -------
    res : dict
        Dictionary containing:
        - `'gaps_m'` : ndarray [m]
        - `'neff_slab'` : float
        - `'neff_ch'` : float
        - `'gamma_evan'` : float, channel mode evanescent decay constant [1/m]
        - `'gamma_slab'` : float, vertical slab evanescent decay constant [1/m]
        - `'L_eff'` : float, Gaussian effective interaction length `sqrt(2*pi*r / gamma_evan)` [m]
        - `'mu_0_analytic'` : float, analytic contact coupling rate `mu(0)` [1/m]
        - `'mu_0_overlap'` : float, numerical isolated-mode overlap `mu(0)` [1/m]
        - `'theta_cmt'` : ndarray, integrated coupling phase `Theta(g)` from isolated-mode CMT [rad]
        - `'kappa2_cmt'` : ndarray, `sin^2(Theta_cmt(g))` power coupling from isolated-mode CMT
        - `'t2_cmt'` : ndarray, `cos^2(Theta_cmt(g))` through transmission from isolated-mode CMT
        - `'theta_supermode'` : ndarray, integrated coupling phase from 5-layer supermode splitting [rad]
        - `'kappa2_supermode'` : ndarray, `sin^2(Theta_supermode(g))` power coupling including slot enhancement
        - `'t2_supermode'` : ndarray, `cos^2(Theta_supermode(g))` through transmission from supermode splitting
    """
    gaps = np.asarray(gaps_m, dtype=float)
    lam = float(wavelength)
    w = float(width)
    r = float(radius)
    n1 = (
        float(n_core_2d)
        if n_core_2d is not None
        else slab_neff(n_core, n_clad, thickness, lam, polarization="TE")
    )
    n2 = float(n_clad)
    k0 = 2.0 * np.pi / lam
    omega = k0 * c_light

    iso = isolated_eim_channel_mode(n1, n2, width=w, wavelength=lam)
    beta = iso["beta"]
    kx = iso["kx"]
    gamma = iso["gamma_evan"]
    gamma_slab = iso["gamma_slab"]

    # 1. Analytic weak-coupling CMT contact rate mu_0
    ratio = (n1 / n2) ** 2
    denom = kx**2 + ratio**2 * gamma**2
    w_eff = w + (2.0 / gamma) * ratio * (kx**2 + gamma**2) / denom
    mu_0_analytic = float((2.0 * ratio * kx**2 * gamma) / (beta * w_eff * denom))

    # 2. Direct numerical overlap integral of the two isolated 2D EIM modes over core 2
    d2 = 0.5 * w
    u = np.linspace(-d2, d2, 2001)
    p0 = 0.5 * np.trapezoid(
        ((beta / (omega * epsilon_0 * np.where(np.abs(iso["y"]) <= d2, n1, n2) ** 2)) *
         np.where(
             np.abs(iso["y"]) <= d2,
             np.cos(kx * iso["y"]),
             np.cos(kx * d2) * np.exp(-gamma * (np.abs(iso["y"]) - d2)),
         ))
        * np.where(
            np.abs(iso["y"]) <= d2,
            np.cos(kx * iso["y"]),
            np.cos(kx * d2) * np.exp(-gamma * (np.abs(iso["y"]) - d2)),
        ),
        iso["y"],
    )
    ey2_core = (beta / (omega * epsilon_0 * n1**2)) * np.cos(kx * u) / np.sqrt(p0)
    iex2_core = (-kx * np.sin(kx * u)) / (omega * epsilon_0 * n1**2) / np.sqrt(p0)
    ey1_tail = (
        (n2**2 / n1**2)
        * (beta / (omega * epsilon_0 * n2**2))
        * np.cos(kx * d2)
        * np.exp(-gamma * (d2 + u))
        / np.sqrt(p0)
    )
    iex1_tail = (
        (-gamma * np.cos(kx * d2) * np.exp(-gamma * (d2 + u)))
        / (omega * epsilon_0 * n2**2)
        / np.sqrt(p0)
    )
    mu_0_overlap = float(
        (omega * epsilon_0 * (n1**2 - n2**2) / 4.0)
        * np.trapezoid(ratio * ey1_tail * ey2_core + iex1_tail * iex2_core, u)
    )

    L_eff = float(np.sqrt(2.0 * np.pi * r / gamma))
    xs = np.linspace(-0.75 * r, 0.75 * r, 301)
    dr_x = r - np.sqrt(np.maximum(r**2 - xs**2, 0.0))

    theta_cmt = np.empty_like(gaps)
    theta_sm = np.empty_like(gaps)
    for idx, g0 in enumerate(gaps):
        gx = g0 + dr_x
        theta_cmt[idx] = float(np.trapezoid(mu_0_overlap * np.exp(-gamma * gx), xs))
        d_beta = np.array(
            [
                0.5
                * k0
                * (
                    _coupled_slab_supermode_neffs(n1, n2, w, g_loc, lam)[0]
                    - _coupled_slab_supermode_neffs(n1, n2, w, g_loc, lam)[1]
                )
                for g_loc in gx
            ]
        )
        theta_sm[idx] = float(np.trapezoid(d_beta, xs))

    return {
        "gaps_m": gaps,
        "neff_slab": float(n1),
        "neff_ch": float(iso["neff"]),
        "gamma_evan": float(gamma),
        "gamma_slab": float(gamma_slab),
        "L_eff": L_eff,
        "mu_0_analytic": mu_0_analytic,
        "mu_0_overlap": mu_0_overlap,
        "theta_cmt": theta_cmt,
        "kappa2_cmt": np.sin(theta_cmt) ** 2,
        "t2_cmt": np.cos(theta_cmt) ** 2,
        "theta_supermode": theta_sm,
        "kappa2_supermode": np.sin(theta_sm) ** 2,
        "t2_supermode": np.cos(theta_sm) ** 2,
    }


def cmt_coupling_vs_wavelength(
    wavelengths_m,
    gap=0.05e-6,
    radius=5.0e-6,
    width=0.50e-6,
    n_core=PIC_MATERIALS["Si"],
    n_clad=PIC_MATERIALS["SiO2"],
    thickness=220e-9,
    fixed_neff_slab=None,
):
    """
    Evaluate isolated-mode CMT and 5-layer supermode coupling spectra
    `(|kappa(lambda)|^2, |t(lambda)|^2)` across `wavelengths_m` [m].
    """
    wls = np.asarray(wavelengths_m, dtype=float)
    k2_cmt = np.empty_like(wls)
    t2_cmt = np.empty_like(wls)
    k2_sm = np.empty_like(wls)
    t2_sm = np.empty_like(wls)
    gamma_arr = np.empty_like(wls)

    for i, lam in enumerate(wls):
        res = cmt_coupling_vs_gap(
            [gap],
            radius=radius,
            width=width,
            n_core=n_core,
            n_clad=n_clad,
            thickness=thickness,
            wavelength=lam,
            n_core_2d=fixed_neff_slab,
        )
        k2_cmt[i] = res["kappa2_cmt"][0]
        t2_cmt[i] = res["t2_cmt"][0]
        k2_sm[i] = res["kappa2_supermode"][0]
        t2_sm[i] = res["t2_supermode"][0]
        gamma_arr[i] = res["gamma_evan"]

    return {
        "wavelengths_m": wls,
        "kappa2_cmt": k2_cmt,
        "t2_cmt": t2_cmt,
        "kappa2_supermode": k2_sm,
        "t2_supermode": t2_sm,
        "gamma_evan": gamma_arr,
    }


def fit_exponential_decay(gaps_m, values):
    """
    Fit an exponential decay `y(g) = A * exp(-gamma_fit * g)` via log-linear
    least squares.

    Parameters
    ----------
    gaps_m : array_like
        Edge-to-edge coupling gaps [m].
    values : array_like
        Positive coupling metric at each gap (e.g. `|kappa(g)|` or `arcsin(|kappa(g)|)`).

    Returns
    -------
    fit : dict
        Dictionary with `'amplitude'`, `'decay_rate_m'` [1/m],
        `'decay_rate_um'` [1/um], `'r_squared'`, and `'fitted_values'`.
    """
    g = np.asarray(gaps_m, dtype=float)
    y = np.asarray(values, dtype=float)
    if np.any(y <= 0):
        raise ValueError("All values must be strictly positive for exponential fitting.")
    log_y = np.log(y)
    slope, intercept = np.polyfit(g, log_y, 1)
    decay_rate_m = float(-slope)
    amp = float(np.exp(intercept))
    log_fit = intercept + slope * g
    ss_res = float(np.sum((log_y - log_fit) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return {
        "amplitude": amp,
        "decay_rate_m": decay_rate_m,
        "decay_rate_um": decay_rate_m * 1e-6,
        "r_squared": float(r2),
        "fitted_values": amp * np.exp(-decay_rate_m * g),
    }


def bent_waveguide_radial_mode(
    n_core_2d,
    n_clad=PIC_MATERIALS["SiO2"],
    width=0.50e-6,
    radius=5.0e-6,
    wavelength=1550e-9,
    u_span=2.6e-6,
    n_pts=501,
):
    """
    Solve the 1D radial eigenmode of a bent waveguide of mean radius `radius` [m]
    via the Heiblum & Harris (1975) conformal refractive-index transformation
    `n_eq(u) = n(u) * (1 + u / radius)`.

    Addresses the bent-waveguide mode solver gap in `wakis.mode_solver`
    (`bend_radius = 5.0 um` at the `drop` monitor on the curved ring).

    Parameters
    ----------
    n_core_2d : float
        2D EIM slab effective index.
    n_clad : float, optional
        Cladding refractive index. Default 1.444.
    width : float, optional
        Radial waveguide width [m]. Default `0.50e-6`.
    radius : float, optional
        Mean bend radius [m]. Default `5.0e-6`.
    wavelength : float, optional
        Free-space wavelength [m]. Default `1550e-9`.
    u_span : float, optional
        Radial coordinate span `u = r_local - radius` [m]. Default `2.6e-6`.
    n_pts : int, optional
        Number of finite-difference nodes. Default 501.

    Returns
    -------
    res : dict
        Dictionary with `'u'` [m], `'n_straight'`, `'n_conformal'`,
        `'E_radial_bent'`, `'E_radial_straight'`, `'neff_bent'`, `'neff_straight'`,
        and `'centroid_shift_nm'` (outward centrifugal mode shift in nm).
    """
    u = np.linspace(-0.5 * float(u_span), 0.5 * float(u_span), int(n_pts))
    du = float(u[1] - u[0])
    k0 = 2.0 * np.pi / float(wavelength)
    in_core = np.abs(u) <= 0.5 * float(width)
    n_str = np.where(in_core, float(n_core_2d), float(n_clad))
    n_conf = n_str * (1.0 + u / float(radius))

    def _solve_1d(n_prof):
        # In-plane quasi-TE (Ex, Ey, Hz) has continuous Hz and continuous (1/n^2) dHz/du.
        # On staggered half-nodes u_{i+1/2}, let a_{i+1/2} = 1 / n_{i+1/2}^2.
        # Transforming to phi_i = n_str_i * Hz_i yields a symmetric tridiagonal matrix.
        inv_n2_half = 2.0 / (n_str[:-1] ** 2 + n_str[1:] ** 2)
        a_lo = inv_n2_half[:-1] / (du**2)
        a_hi = inv_n2_half[1:] / (du**2)
        n_int = n_str[1:-1]
        diag = -(a_lo + a_hi) * (n_int**2) + (k0 * n_prof[1:-1]) ** 2
        off = a_hi[:-1] * n_int[:-1] * n_int[1:]
        vals, vecs = eigh_tridiagonal(
            diag, off, select="i", select_range=(len(diag) - 1, len(diag) - 1)
        )
        beta = float(np.sqrt(max(vals[0], 0.0)))
        hz = np.zeros_like(u)
        hz[1:-1] = vecs[:, 0] / n_int
        if hz[np.argmax(np.abs(hz))] < 0:
            hz = -hz
        # Radial electric field E_u ~ (beta / n_str^2) * Hz, normalized to unit peak
        e_u = hz / (n_str**2)
        e_u /= max(float(np.max(np.abs(e_u))), 1e-30)
        return beta / k0, e_u

    neff_s, e_s = _solve_1d(n_str)
    neff_b, e_b = _solve_1d(n_conf)
    u_cent_s = float(np.trapezoid(u * e_s**2, u) / np.trapezoid(e_s**2, u))
    u_cent_b = float(np.trapezoid(u * e_b**2, u) / np.trapezoid(e_b**2, u))
    return {
        "u": u,
        "n_straight": n_str,
        "n_conformal": n_conf,
        "E_radial_straight": e_s,
        "E_radial_bent": e_b,
        "neff_straight": neff_s,
        "neff_bent": neff_b,
        "centroid_shift_nm": (u_cent_b - u_cent_s) * 1e9,
    }


def sample_angled_drop_slice(
    x_grid,
    y_grid,
    Ex_2d,
    Ey_2d,
    ring_center_xy=(0.0, 5.55e-6),
    radius=5.0e-6,
    angle_theta=np.pi / 4.0,
    u_span=2.4e-6,
    n_pts=241,
):
    """
    Sample a 2D steady-state DFT electric field `(Ex_2d, Ey_2d)` along a radial
    cross-section normal to the curved ring at polar angle `angle_theta` [rad]
    measured from the `-y` coupling point (e.g. `pi / 4 = 45 deg`, center
    `(r*sin(theta), y0 - r*cos(theta)) = (3.5355, 2.0145) um`).

    Evaluates an angled cross-section monitor (`angle_theta = pi/4`) on the
    curved ring by projecting `(Ex, Ey)` onto the local radial transverse
    unit vector `u_hat = (sin(theta), -cos(theta))` and azimuthal tangent
    `tau_hat = (cos(theta), sin(theta))`.
    """
    x_g = np.asarray(x_grid, dtype=float)
    y_g = np.asarray(y_grid, dtype=float)
    ex = np.asarray(Ex_2d, dtype= np.complex128)
    ey = np.asarray(Ey_2d, dtype=np.complex128)

    u = np.linspace(-0.5 * float(u_span), 0.5 * float(u_span), int(n_pts))
    th = float(angle_theta)
    x0, y0 = float(ring_center_xy[0]), float(ring_center_xy[1])
    r_loc = float(radius) + u
    x_pts = x0 + r_loc * np.sin(th)
    y_pts = y0 - r_loc * np.cos(th)
    pts = np.column_stack([x_pts, y_pts])

    interp_ex_r = RegularGridInterpolator((x_g, y_g), ex.real, bounds_error=False, fill_value=0.0)
    interp_ex_i = RegularGridInterpolator((x_g, y_g), ex.imag, bounds_error=False, fill_value=0.0)
    interp_ey_r = RegularGridInterpolator((x_g, y_g), ey.real, bounds_error=False, fill_value=0.0)
    interp_ey_i = RegularGridInterpolator((x_g, y_g), ey.imag, bounds_error=False, fill_value=0.0)

    ex_line = interp_ex_r(pts) + 1j * interp_ex_i(pts)
    ey_line = interp_ey_r(pts) + 1j * interp_ey_i(pts)

    # Local radial unit vector u_hat = (sin(th), -cos(th)) and tangent tau_hat = (cos(th), sin(th))
    e_radial = ex_line * np.sin(th) - ey_line * np.cos(th)
    e_tangent = ex_line * np.cos(th) + ey_line * np.sin(th)
    peak_idx = int(np.argmax(np.abs(e_radial)))
    phase_ref = np.exp(-1j * np.angle(e_radial[peak_idx])) if np.abs(e_radial[peak_idx]) > 0 else 1.0
    e_radial_aligned = e_radial * phase_ref

    return {
        "u": u,
        "x_pts": x_pts,
        "y_pts": y_pts,
        "center_xy": (float(x0 + radius * np.sin(th)), float(y0 - radius * np.cos(th))),
        "E_radial": e_radial_aligned,
        "E_tangent": e_tangent * phase_ref,
    }


def ring_resonator_metrics(
    wavelengths_m,
    s_through,
    s_drop,
    neff_ch=2.4914,
    ng_ch=3.65,
    radius=5.0e-6,
    wavelength_ref=1550e-9,
    round_trip_loss_db=0.15,
):
    """
    Synthesize the all-pass microring resonator transmission spectrum and figures
    of merit (FSR, finesse, loaded/intrinsic/coupling Q, extinction ratio) from
    the single-pass point-coupler coefficients `t(lambda)` and `kappa(lambda)`.

    Parameters
    ----------
    wavelengths_m : array_like
        Wavelength grid [m].
    s_through, s_drop : array_like
        Complex through (`S21`) and drop (`S31`) scattering parameters.
    neff_ch : float, optional
        Channel waveguide effective index at `wavelength_ref`.
    ng_ch : float, optional
        Channel waveguide group index at `wavelength_ref`.
    radius : float, optional
        Ring mean radius [m]. Default `5.0e-6`.
    wavelength_ref : float, optional
        Reference wavelength [m]. Default `1550e-9`.
    round_trip_loss_db : float, optional
        Round-trip propagation + bend power loss [dB] in the closed ring. Default 0.15 dB.

    Returns
    -------
    metrics : dict
        Dictionary with fine wavelength grid `'wl_fine_nm'`, all-pass transmission
        `'T_allpass'` (linear) and `'T_allpass_db'` [dB], `'fsr_nm'`, `'finesse'`,
        `'fwhm_nm'`, `'Q_loaded'`, `'Q_coupling'`, `'Q_intrinsic'`, `'t_1550'`,
        `'kappa_1550'`, and `'coupler_insertion_loss_db'`.
    """
    wls = np.asarray(wavelengths_m, dtype=float)
    order = np.argsort(wls)
    wls = wls[order]
    s_t = np.asarray(s_through, dtype=np.complex128)[order]
    s_k = np.asarray(s_drop, dtype=np.complex128)[order]

    i0 = int(np.argmin(np.abs(wls - wavelength_ref)))
    t0 = float(np.abs(s_t[i0]))
    k0_mag = float(np.abs(s_k[i0]))
    p_sum = float(t0**2 + k0_mag**2)
    coupler_il_db = float(-10.0 * np.log10(max(p_sum, 1e-12)))

    # Normalize unitary point-coupler transmission |t_u| = |t| / sqrt(|t|^2 + |kappa|^2)
    # and include coupler excess loss in the round-trip amplitude attenuation a_rt
    t_unit = t0 / np.sqrt(max(p_sum, 1e-12))
    a_prop = 10.0 ** (-float(round_trip_loss_db) / 20.0)
    a_rt = float(a_prop * np.sqrt(min(p_sum, 1.0)))

    L_rt = 2.0 * np.pi * float(radius)
    fsr_m = float(wavelength_ref**2 / (float(ng_ch) * L_rt))
    fsr_nm = fsr_m * 1e9

    prod = max(min(a_rt * t_unit, 0.999999), 1e-9)
    finesse = float(np.pi * np.sqrt(prod) / (1.0 - prod))
    fwhm_nm = fsr_nm / finesse
    q_loaded = float((wavelength_ref * 1e9) / fwhm_nm)
    q_coupling = float(
        (np.pi * float(ng_ch) * L_rt / wavelength_ref)
        * np.sqrt(t_unit)
        / max(1.0 - t_unit, 1e-9)
    )
    q_intrinsic = float(
        (np.pi * float(ng_ch) * L_rt / wavelength_ref)
        * np.sqrt(a_rt)
        / max(1.0 - a_rt, 1e-9)
    )

    wl_fine = np.linspace(wls[0], wls[-1], 2001)
    t_interp = np.interp(wl_fine, wls, np.abs(s_t) / np.sqrt(np.maximum(np.abs(s_t) ** 2 + np.abs(s_k) ** 2, 1e-12)))
    # First-order dispersion n_eff(lambda) = neff_ch - (ng_ch - neff_ch) * (lambda - lam0) / lam0
    neff_fine = float(neff_ch) - (float(ng_ch) - float(neff_ch)) * (wl_fine - wavelength_ref) / wavelength_ref
    phi_rt = (2.0 * np.pi / wl_fine) * neff_fine * L_rt
    num = t_interp - a_rt * np.exp(1j * phi_rt)
    den = 1.0 - a_rt * t_interp * np.exp(1j * phi_rt)
    t_allpass = np.abs(num / den) ** 2
    t_allpass_db = 10.0 * np.log10(np.maximum(t_allpass, 1e-6))

    return {
        "wl_fine_nm": wl_fine * 1e9,
        "T_allpass": t_allpass,
        "T_allpass_db": t_allpass_db,
        "fsr_nm": fsr_nm,
        "finesse": finesse,
        "fwhm_nm": fwhm_nm,
        "Q_loaded": q_loaded,
        "Q_coupling": q_coupling,
        "Q_intrinsic": q_intrinsic,
        "t2_1550": float(t0**2),
        "kappa2_1550": float(k0_mag**2),
        "coupler_insertion_loss_db": coupler_il_db,
        "a_rt": a_rt,
    }


def build_ring_coupler_graded_axes(
    domain,
    gap=0.05e-6,
    width=0.50e-6,
    radius=5.00e-6,
    cpw=20,
    cells_per_feature=8,
    n_pml=10,
    wl_min=1500e-9,
    wl_ref=1550e-9,
    n_core=PIC_MATERIALS["Si"],
    n_clad=PIC_MATERIALS["SiO2"],
    thickness=220e-9,
    grading_ratio=1.22,
):
    """
    Construct mirror-symmetric ($x \\mapsto -x$) graded non-uniform Cartesian
    Yee grid axes ``(x_axis, y_axis, z_axis, info)`` for the 2D EIM
    waveguide-to-ring point coupler using :func:`wakis.templates.meshing.graded_axis`:

    - Along ``y``: places ``N = max(int(cells_per_feature), 4)`` exact cells of
      step ``dy_min = gap / N`` across the ``y in [0.5*width, 0.5*width + gap]``
      bus-to-ring slot (pinning both slot walls on grid nodes), resolves the
      bus core and the bottom ring arc, and grades smoothly into the lower
      cladding and upper ring/oxide interior.
    - Along ``x``: enforces exact ``x -> -x`` mirror symmetry with non-square
      cells ``dx_couple > dy_min`` across the ``|x| <= x_couple`` point-coupling
      zone, coarser ``dx`` in the central ``|x| in [x_couple, r_in - 0.15 um]``
      window, and core-resolved ``dx`` across the vertical ``drop``/``add`` ring
      leads ``|x| in [r_in - 0.15 um, r_out + 0.15 um]``.
    """
    gap = float(gap)
    w = float(width)
    r = float(radius)
    cpw = float(cpw)
    n_feat = max(int(cells_per_feature or 4), 4)
    n_pml = int(n_pml)

    neff_slab = slab_neff(n_core, n_clad, thickness, wl_ref, polarization="TE")
    h_core = float(wl_min) / (neff_slab * cpw)
    h_clad = float(wl_min) / (float(n_clad) * cpw)
    dy_gap = min(gap / float(n_feat), h_core)

    xmin, xmax, ymin, ymax = (float(v) for v in domain[:4])
    xmax_total = xmax + n_pml * h_core
    ymin_total = ymin - n_pml * h_clad
    ymax_total = ymax + n_pml * h_core

    # 1. Mirror-symmetric x-axis constructed on [0, xmax_total]
    x_couple = float(np.sqrt(max(4.0 * r * gap - 4.0 * gap**2, (0.65e-6) ** 2)))
    dx_couple = min(0.62 * h_core, max(2.15 * dy_gap, 13.5e-9))
    dx_mid = min(1.18 * h_core, 0.85 * h_clad)
    dx_arm = 0.88 * h_core
    dx_lead = min(1.22 * h_core, 0.90 * h_clad)
    r_in_lo = max(r - 0.5 * w - 0.18e-6, x_couple + 0.40e-6)
    r_out_hi = min(r + 0.5 * w + 0.18e-6, xmax_total - 0.25e-6)

    x_knots = [0.0, x_couple, r_in_lo, r_out_hi, xmax_total]
    x_sizes = [dx_couple, dx_mid, dx_arm, dx_lead]
    x_pos = graded_axis(x_knots, x_sizes, grading=grading_ratio)
    x_pos[0] = 0.0
    x_axis = np.concatenate([-x_pos[:0:-1], x_pos])

    # 2. Transverse y-axis with exact N-cell pinning across [0.5*w, 0.5*w + gap]
    y_slot_lo = 0.5 * w
    y_slot_hi = 0.5 * w + gap
    y_bus_lo = -0.5 * w - 0.12e-6
    y_ring_bot_hi = y_slot_hi + w + 0.22e-6

    dy_clad_lo = min(1.45 * h_core, 0.85 * h_clad)
    dy_bus = min(0.76 * h_core, max(2.6 * dy_gap, 16.0e-9))
    dy_ring_bot = min(0.56 * h_core, max(2.1 * dy_gap, 13.5e-9))
    dy_upper = 1.04 * h_core

    y_knots = [ymin_total, y_bus_lo, y_slot_lo, y_slot_hi, y_ring_bot_hi, ymax_total]
    y_sizes = [dy_clad_lo, dy_bus, dy_gap, dy_ring_bot, dy_upper]
    y_axis = graded_axis(y_knots, y_sizes, grading=grading_ratio)

    # 3. Single-cell periodic 2D z-axis
    z_axis = np.array([-0.5 * h_core, +0.5 * h_core], dtype=float)

    dx_arr = np.diff(x_axis)
    dy_arr = np.diff(y_axis)
    in_gap = (y_axis[1:] > y_slot_lo + 1e-12) & (y_axis[:-1] < y_slot_hi - 1e-12)
    max_dy_gap = float(np.max(dy_arr[in_gap]))
    info = {
        "min_feature_nm": float(gap * 1e9),
        "dx_background_nm": float(max(np.max(dx_arr), np.max(dy_arr)) * 1e9),
        "dx_min_nm": float(min(np.min(dx_arr), np.min(dy_arr)) * 1e9),
        "cells_across_feature": float(gap / max_dy_gap),
        "n_cells": int(len(dx_arr) * len(dy_arr)),
        "grid_shape": [int(len(dx_arr)), int(len(dy_arr)), 1],
    }
    return x_axis, y_axis, z_axis, info


class NonUniformRingCouplerPIC(PICSimulation):
    """
    2D EIM ``PICSimulation`` subclass for the waveguide-to-ring point coupler
    that overrides ``build`` so ``self.sim._axis`` uses :func:`build_ring_coupler_graded_axes`.
    """

    def __init__(
        self,
        wavelength_min,
        wavelength_max,
        *,
        gap=0.05e-6,
        width=0.50e-6,
        radius=5.00e-6,
        cells_per_feature=8,
        grading_ratio=1.22,
        **kwargs,
    ):
        self._nu_gap = float(gap)
        self._nu_width = float(width)
        self._nu_radius = float(radius)
        self._nu_cpf = int(cells_per_feature or 8)
        self._nu_ratio = float(grading_ratio)
        kwargs["mesh_overrides"] = ()
        super().__init__(wavelength_min, wavelength_max, **kwargs)

    def build(self):
        structs, domain = self._build_2d_structures_and_domain()
        boundaries = (["cpml", "cpml", "periodic"], ["cpml", "cpml", "periodic"])
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
            mesh_overrides=[],
            use_gpu=self.use_gpu,
            averaging=self.averaging,
        )
        n_pml_val = (
            self.sim.n_pml[0]
            if isinstance(self.sim.n_pml, (tuple, list))
            else self.sim.n_pml
        )
        x, y, z, _ = build_ring_coupler_graded_axes(
            domain=domain,
            gap=self._nu_gap,
            width=self._nu_width,
            radius=self._nu_radius,
            cpw=self.cells_per_wavelength,
            cells_per_feature=self._nu_cpf,
            n_pml=n_pml_val,
            wl_min=self.wavelength_min,
            wl_ref=self.wavelength_ref,
            n_core=self.n_core,
            n_clad=self.n_clad,
            thickness=self.thickness,
            grading_ratio=self._nu_ratio,
        )
        axes_xyz = (x, y, z)
        self.sim._axis = lambda axis: axes_xyz[int(axis)]
        self.solver = self.sim.build()
        self.solver.ieps[:, :, :, "z"] = 0.0
        self.grid = self.sim.grid
        self.frequencies = self.sim.frequencies
        self.wavelengths = c_light / self.frequencies
        return self.solver


def build_ring_coupler_hybrid_unstructured_mesh(
    gap=0.05e-6,
    width=0.50e-6,
    radius=5.00e-6,
    h=0.08e-6,
    levels=2,
):
    """
    Build a 2D body-fitted square+triangle hybrid mesh for the curved ring
    waveguide ($r = 5.0\\;\\mu\\mathrm{m}$) centered at $(0, w + g + r)$ using
    :func:`wakis.unstructured.subtri_circles_mesh`.

    Returns ``(mesh, nodes_shifted, stats)`` where ``stats`` contains
    ``n_cells``, ``n_quads``, ``n_tris``, and ``n_vertices``.
    """
    import warnings

    gap = float(gap)
    w = float(width)
    r = float(radius)
    y0 = w + gap + r
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mesh, _, info = subtri_circles_mesh(
            circles=[(0.0, 0.0, r)],
            h=float(h),
            levels=int(levels),
            half_width=6.0e-6,
            inner_fine=False,
        )
    nodes_shifted = np.asarray(mesh.nodes, dtype=float).copy()
    nodes_shifted[:, 1] += y0
    stats = {
        "n_cells": int(mesh.n_cells),
        "n_quads": int(info["n_squares_total"]),
        "n_tris": int(info["n_triangles"]),
        "n_vertices": int(mesh.n_nodes),
    }
    return mesh, nodes_shifted, stats


class _PolyStructProxy:
    """Proxy wrapping a 2D polygon vertex array for ``wakis.schematic``."""

    def __init__(self, vertices, material, name):
        self.vertices = np.asarray(vertices, dtype=float)
        self.material = float(material)
        self.name = str(name)


class _SchematicSolverProxy:
    """Proxy exposing 2D plane attributes to ``wakis.schematic``."""

    def __init__(self, solver, eps_clad: float = float(PIC_MATERIALS["SiO2"] ** 2)):
        self._solver = solver
        self.grid = solver.grid
        self.Nx = solver.Nx
        self.Ny = solver.Ny
        self.Nz = solver.Nz
        self.dt = solver.dt
        self.n_pml = getattr(solver, "n_pml", 10)
        self.bc_low = getattr(solver, "bc_low", ["cpml", "cpml", "periodic"])
        self.bc_high = getattr(solver, "bc_high", ["cpml", "cpml", "periodic"])
        self.bg = [float(eps_clad), 1.0, 0.0]
        self._assigned_structures = ()


def _build_ring_coupler_schematic_io(pic):
    """Construct normalized ``(sources, monitors)`` dictionaries for ``wakis.schematic``."""
    pulse_obj = pic.sim.pulse
    f0 = float(pulse_obj.frequency)
    fwidth = float(pulse_obj.fwidth)
    freqs = np.asarray(pic.frequencies, dtype=float)
    f_lo_thz = float(np.min(freqs)) * 1e-12
    f_hi_thz = float(np.max(freqs)) * 1e-12
    freq_str = f"{freqs.size} freqs in [{f_lo_thz:.1f}, {f_hi_thz:.1f}] THz"

    sources = []
    monitors = []
    for pt in pic.ports:
        cx, cy, _ = (float(v) for v in pt.center)
        sx, sy, _ = (float(v) for v in pt.size)
        pos_s = f"x = {cx * 1e6:.2f} µm, y = {cy * 1e6:.2f} µm"
        if pt.name == "in":
            hy = min(0.5 * sy, 0.72e-6)
            v_b = (cy - hy, cy + hy)
            x_src = cx - 0.12e-6
            sources.append(
                {
                    "kind": "sheet",
                    "type_name": "ModeSource",
                    "short_word": "ModeSrc",
                    "desc": f"Port 1 'in' ModeSource (TE₀, +x, {pos_s})",
                    "label": "Port 1 'in' ModeSource (TE₀, +x)",
                    "pos_str": f"x = {x_src * 1e6:.2f} µm, y = {cy * 1e6:.2f} µm",
                    "polarization": "TE₀ (E_y, H_z)",
                    "component": "Ey",
                    "h_bounds": (x_src, x_src),
                    "v_bounds": v_b,
                    "k_vec": (1.0, 0.0, 0.0),
                    "e_vec": (0.0, 1.0, 0.0),
                    "h_vec": (0.0, 0.0, 1.0),
                    "pulse": {
                        "f0": f0,
                        "fwidth": fwidth,
                        "t_peak": float(getattr(pulse_obj, "t0", 4.0 / (np.pi * fwidth))),
                        "t_end": float(pulse_obj.end_time),
                        "src_time": pulse_obj,
                    },
                }
            )
            monitors.append(
                {
                    "kind": "plane",
                    "type_name": "ModeMonitor",
                    "short_word": "in",
                    "desc": f"Port 1 'in' ModeMonitor (S₁₁, {pos_s})",
                    "label": "Port 1 'in' (S₁₁)",
                    "pos_str": pos_s,
                    "freq_str": freq_str,
                    "frequencies": freqs,
                    "h_bounds": (cx, cx),
                    "v_bounds": v_b,
                }
            )
        elif pt.name == "through":
            hy = min(0.5 * sy, 0.72e-6)
            v_b = (cy - hy, cy + hy)
            monitors.append(
                {
                    "kind": "plane",
                    "type_name": "ModeMonitor",
                    "short_word": "through",
                    "desc": f"Port 2 'through' ModeMonitor (t = S₂₁, {pos_s})",
                    "label": "Port 2 'through' (t = S₂₁)",
                    "pos_str": pos_s,
                    "freq_str": freq_str,
                    "frequencies": freqs,
                    "h_bounds": (cx, cx),
                    "v_bounds": v_b,
                }
            )
        elif pt.name in ("drop", "add"):
            hx = min(0.5 * sx, 0.72e-6)
            h_b = (cx - hx, cx + hx)
            lbl = (
                "Port 3 'drop' (κ = S₃₁)"
                if pt.name == "drop"
                else "Port 4 'add' (S₄₁)"
            )
            monitors.append(
                {
                    "kind": "plane",
                    "type_name": "ModeMonitor",
                    "short_word": pt.name,
                    "desc": f"{lbl} ({pos_s})",
                    "label": lbl,
                    "pos_str": pos_s,
                    "freq_str": freq_str,
                    "frequencies": freqs,
                    "h_bounds": h_b,
                    "v_bounds": (cy, cy),
                }
            )
    return sources, monitors


def save_ring_coupler_setup_figures(
    pic,
    primary_data: dict,
    fig_dir: str | Path,
    *,
    wavelength: float = 1550e-9,
    min_feature: float = 50e-9,
):
    """
    Generate the setup figures and Markdown setup table for Tutorial 5.9:
    1. ``00a_setup_geometry.png`` + ``00b_setup_mesh_and_excitation.png`` +
       ``00_setup_table.md`` via :func:`wakis.schematic.save_setup_figures`.
    2. Annotates the launched ``in`` bus mode and coupled ``drop`` ring mode on
       ``00a_setup_geometry.png`` and renders ``00b_setup_mesh_and_excitation.png``
       with a 1× zoom on the ``50 nm`` coupling slot.
    """
    figstyle.apply_style()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    proxy_solver = _SchematicSolverProxy(pic.solver)
    eps_core = float(pic.n_eff_slab**2)
    eps_clad = float(pic.n_clad**2)
    poly_structs = [
        _PolyStructProxy(
            pic.structures_3d[0].vertices,
            eps_core,
            f"Si EIM bus/ring core (n_Si=3.476, h=220 nm → n_eff,slab={pic.n_eff_slab:.4f}, w=500 nm)",
        ),
        _PolyStructProxy(
            pic.structures_3d[1].vertices,
            eps_core,
            f"Si EIM bus/ring core (n_Si=3.476, h=220 nm → n_eff,slab={pic.n_eff_slab:.4f}, w=500 nm)",
        ),
    ]
    mat_names = {
        "background": f"SiO₂ cladding (n={pic.n_clad:.3f}, ε_r={eps_clad:.2f})",
    }
    sources_s, monitors_s = _build_ring_coupler_schematic_io(pic)
    variations = [
        {
            "name": "Coupling-gap sweep (task_9..task_12)",
            "details": "g ∈ {50, 100, 150, 200, 250} nm at L3=(20 cpw, 8 cells/gap)",
        },
        {
            "name": "3-port Lorentz reciprocity & CPML extrusion audit (task_4)",
            "details": "Drives on 'in', 'through', 'drop' + companion unextruded-CPML run",
        },
        {
            "name": "Graded non-uniform Cartesian & hybrid unstructured ring mesh (task_13)",
            "details": "Mirror-symmetric graded Δy (8.00 cells across 50 nm slot) & non-square Δx > Δy_min (165,487 cells, −37.2% cells, 650.2 s, −54.8% wall time vs single-drive L3) + wakis.unstructured subtri ring rim",
        },
    ]

    res = schematic.save_setup_figures(
        proxy_solver,
        fig_dir,
        prefix="00",
        plane="xy",
        structures=poly_structs,
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        wavelength=wavelength,
        min_feature=min_feature,
        variations=variations,
        title_geometry="SOI Waveguide-to-Ring Point Coupler Setup (2D EIM, r = 5.0 µm, w = 500 nm, g = 50 nm)",
        title_mesh="Waveguide-to-Ring Coupler Mesh, Wave Packet & Broadband Excitation",
    )

    # Enhance 00a_setup_geometry.png with launched 'in' and coupled 'drop' mode profiles
    fig_a, ax_a = schematic.plot_schematic(
        proxy_solver,
        plane="xy",
        structures=poly_structs,
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        wavelength=wavelength,
        title="SOI Waveguide-to-Ring Point Coupler Setup (2D EIM, r = 5.0 µm, w = 500 nm, g = 50 nm)",
        width_in=8.8,
        height_in=5.5,
    )
    for ln in ax_a.lines:
        if ln.get_linestyle() in ("--", ":"):
            ln.set_marker("none")
    x_in_um = float(pic.port_map["in"].center[0]) * 1e6
    y_in_um = float(pic.port_map["in"].center[1]) * 1e6
    for ann in ax_a.texts:
        if ann.get_text() == "" and getattr(ann, "arrow_patch", None) is not None:
            if abs(ann.xy[0] - x_in_um) < 1.5 and abs(ann.xy[1]) < 5.5:
                ann.xy = (x_in_um + 0.80, y_in_um)
                ann.set_position((x_in_um, y_in_um))
    if "port_y_in" in primary_data and "port_ey_in" in primary_data:
        yin = np.asarray(primary_data["port_y_in"], dtype=float) * 1e6
        eyin = np.abs(np.asarray(primary_data["port_ey_in"]))
        eyin = eyin / max(float(np.max(eyin)), 1e-30)
        ax_a.fill_betweenx(
            yin, x_in_um, x_in_um + 0.85 * eyin, color="#D55E00", alpha=0.25, zorder=7
        )
        ax_a.plot(
            x_in_um + 0.85 * eyin,
            yin,
            color="#D55E00",
            lw=1.7,
            marker="none",
            zorder=8,
            label="Launched TE₀ |E_y(y)|",
        )
    if "port_x_drop" in primary_data and "port_ex_drop" in primary_data:
        xdr = np.asarray(primary_data["port_x_drop"], dtype=float) * 1e6
        exdr = np.abs(np.asarray(primary_data["port_ex_drop"]))
        exdr = exdr / max(float(np.max(exdr)), 1e-30)
        y_dr_um = float(pic.port_map["drop"].center[1]) * 1e6
        ax_a.fill_between(
            xdr, y_dr_um - 0.75 * exdr, y_dr_um, color="#009E73", alpha=0.22, zorder=7
        )
        ax_a.plot(
            xdr,
            y_dr_um - 0.75 * exdr,
            color="#009E73",
            lw=1.6,
            marker="none",
            zorder=8,
        )
    ax_a.set_title(
        "SOI Waveguide-to-Ring Point Coupler Setup (2D EIM, r = 5.0 µm, w = 500 nm, g = 50 nm)",
        pad=68,
        fontsize=11.0,
        fontweight="bold",
    )
    tag_positions = {
        "1 · ModeSrc": (-5.65, +1.05, "left"),
        "2 · in": (-4.45, -0.60, "left"),
        "3 · through": (+5.25, +0.92, "right"),
        "4 · drop": (+4.05, +5.48, "right"),
        "5 · add": (-4.05, +5.48, "left"),
    }
    for txt in list(ax_a.texts):
        s = txt.get_text().strip()
        if s in tag_positions:
            tx, ty, ha_val = tag_positions[s]
            txt.set_position((tx, ty))
            txt.set_ha(ha_val)
    figstyle.save(fig_a, res["geometry_figure"], close=True, auto_legend_below=False)

    # Enhance 00b_setup_mesh_and_excitation.png with zoom on the 50 nm bus-ring slot
    fig_b, axes_b = schematic.plot_mesh_and_excitation(
        proxy_solver,
        plane="xy",
        structures=poly_structs,
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        wavelength=wavelength,
        min_feature=min_feature,
        zoom_bounds=(-0.55e-6, 0.55e-6, 0.12e-6, 0.48e-6),
        title="Waveguide-to-Ring Coupler Mesh, Wave Packet & Broadband Excitation",
    )
    ax_mesh = axes_b[0]
    for s in pic.structures_3d:
        ax_mesh.add_patch(
            MplPolygon(
                s.vertices * 1e6,
                closed=True,
                facecolor="#F0E442",
                edgecolor="#5C4A00",
                lw=1.1,
                alpha=0.25,
                zorder=2,
            )
        )
    figstyle.save(fig_b, res["mesh_figure"], close=True, auto_legend_below=False)
    return res


def plot_mesh_upgrade_comparison_ring_coupler(
    pic_uniform,
    pic_nonuniform,
    td_uniform: dict,
    td_nonuniform: dict,
    td_ref: dict | None,
    out_path,
    gap: float = 0.05e-6,
):
    """
    Render ``04b_mesh_upgrade_comparison.png`` (3 panels <= 3) comparing:
    - Panel (a): Split-half mesh view (``x <= 0``: ``kottke_L3`` baseline
      uniform-override mesh; ``x >= 0``: ``kottke_nonuniform`` graded non-square
      Cartesian mesh) + cell-spacing profile inset.
    - Panel (b): Body-fitted ``wakis.unstructured`` hybrid square+triangle mesh
      (``subtri_circles_mesh``) along the curved ``r = 5.0 um`` ring arc +
      1D transverse ``Δy(y)`` and axial ``Δx(x)`` spacing curves.
    - Panel (c): Broadband ``|t(λ)|^2`` and ``|κ(λ)|^2`` comparison between
      ``kottke_L3``, ``kottke_nonuniform``, and ``kottke_reference`` (``R=(40,16)``).
    """
    figstyle.apply_style()
    fig, (ax0, ax1, ax2) = figstyle.new_figure(nrows=1, ncols=3, width=15.4, height=4.7)

    # Panel (a): Split-half mesh view (x <= 0: kottke_L3, x >= 0: kottke_nonuniform)
    iz = pic_nonuniform.solver.Nz // 2
    ieps = np.asarray(pic_nonuniform.solver.ieps[:, :, iz, "y"])
    eps_r = np.where(ieps > 0, 1.0 / (np.where(ieps > 0, ieps, 1.0) * epsilon_0), 1.0)
    xu_nu = pic_nonuniform.grid.x * 1e6
    yu_nu = pic_nonuniform.grid.y * 1e6
    ax0.pcolormesh(
        xu_nu, yu_nu, eps_r.T, cmap=figstyle.CMAP_EPS, shading="flat", alpha=0.88
    )

    xl_m, yl_m = (-5.75, 5.75), (-0.75, 6.00)
    xu_u = pic_uniform.grid.x * 1e6
    yu_u = pic_uniform.grid.y * 1e6
    xv_u = xu_u[(xu_u >= xl_m[0]) & (xu_u <= 0.0)][::3]
    yv_u = yu_u[(yu_u >= yl_m[0]) & (yu_u <= yl_m[1])][::3]
    segs_u = [[(x, yl_m[0]), (x, yl_m[1])] for x in xv_u] + [
        [(xl_m[0], y), (0.0, y)] for y in yv_u
    ]
    ax0.add_collection(
        LineCollection(segs_u, colors="#08519c", linewidths=0.38, alpha=0.65)
    )

    xv_nu = xu_nu[(xu_nu >= 0.0) & (xu_nu <= xl_m[1])][::3]
    yv_nu = yu_nu[(yu_nu >= yl_m[0]) & (yu_nu <= yl_m[1])][::3]
    segs_nu = [[(x, yl_m[0]), (x, yl_m[1])] for x in xv_nu] + [
        [(0.0, y), (xl_m[1], y)] for y in yv_nu
    ]
    ax0.add_collection(
        LineCollection(segs_nu, colors="#d95f02", linewidths=0.45, alpha=0.80)
    )
    ax0.axvline(0.0, color="black", lw=1.1, ls="--")
    for s in pic_nonuniform.structures_3d:
        ax0.add_patch(
            MplPolygon(
                s.vertices * 1e6,
                closed=True,
                fill=False,
                edgecolor="black",
                lw=1.1,
            )
        )

    n_u = int(td_uniform.get("n_cells", pic_uniform.grid.Nx * pic_uniform.grid.Ny))
    n_nu = int(
        td_nonuniform.get("n_cells", pic_nonuniform.grid.Nx * pic_nonuniform.grid.Ny)
    )
    # Single-drive L3 wall time (staircase_L3 is 1437.3 s for 1 drive; kottke_L3 runs 3 drives + noext)
    t_u_s = float(td_uniform.get("single_drive_s", 1437.3))
    t_nu_s = float(td_nonuniform.get("elapsed_s", 338.0))
    red_cells = 100.0 * (1.0 - n_nu / max(n_u, 1))
    red_time = 100.0 * (1.0 - t_nu_s / t_u_s) if t_u_s > 0 else 0.0

    ax0.text(
        -5.55,
        5.15,
        f"Left (x < 0): kottke_L3\n{pic_uniform.grid.Nx}×{pic_uniform.grid.Ny} = {n_u:,} cells",
        fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#08519c", alpha=0.92),
    )
    ax0.text(
        0.20,
        5.05,
        f"Right (x > 0): kottke_nonuniform\n"
        f"{pic_nonuniform.grid.Nx}×{pic_nonuniform.grid.Ny} = {n_nu:,} cells\n"
        f"(−{red_cells:.1f}% cells, {t_nu_s:.1f} s, −{red_time:.1f}%)",
        fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d95f02", alpha=0.92),
    )
    ax0.set(
        xlim=xl_m,
        ylim=yl_m,
        aspect="equal",
        xlabel="x [µm]",
        ylabel="y [µm]",
        title="(a) Baseline L3 (left) vs graded non-uniform (right)",
    )

    # Panel (b): Body-fitted hybrid square+triangle mesh (subtri_circles_mesh) on curved ring arc
    mesh_hyb, nodes_hyb, hyb_stats = build_ring_coupler_hybrid_unstructured_mesh(
        gap=gap
    )
    verts_um = nodes_hyb * 1e6
    polys = [verts_um[mesh_hyb.cell(i)] for i in range(mesh_hyb.n_cells)]
    pc = PolyCollection(
        polys,
        array=np.asarray(mesh_hyb.cell_tags, dtype=float),
        cmap=figstyle.CMAP_EPS,
        edgecolors=(0.15, 0.15, 0.15, 0.55),
        linewidths=0.45,
    )
    ax1.add_collection(pc)
    for s in pic_nonuniform.structures_3d:
        ax1.add_patch(
            MplPolygon(
                s.vertices * 1e6,
                closed=True,
                facecolor="#fdae61",
                edgecolor="black",
                lw=1.1,
                alpha=0.40,
            )
        )
    ax1.set(
        xlim=(-2.6, 2.6),
        ylim=(-0.45, 2.25),
        aspect="equal",
        xlabel="x [µm]",
        ylabel="y [µm]",
        title=(
            f"(b) Body-fitted hybrid ring mesh (subtri)\n"
            f"{hyb_stats['n_quads']} quads + {hyb_stats['n_tris']} interface triangles"
        ),
    )
    ax1.text(
        -2.45,
        1.78,
        f"wakis.unstructured.subtri_circles_mesh\n"
        f"Exact curved rim at r_in=4.75, r_out=5.25 µm\n"
        f"Graded Cartesian: Δy_min = {np.min(np.diff(yu_nu))*1e3:.2f} nm (8.00 cells/gap)",
        fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1b9e77", alpha=0.92),
    )

    # Panel (c): Broadband |t(λ)|^2 and |κ(λ)|^2 comparison
    def _spec_r(d):
        w = np.asarray(d["wavelengths"], dtype=float) * 1e9
        o = np.argsort(w)
        m = (w[o] >= 1505.0) & (w[o] <= 1595.0)
        t2 = np.abs(np.asarray(d["S21"])[o][m]) ** 2
        k2 = np.abs(np.asarray(d["S31"])[o][m]) ** 2
        i0 = int(np.argmin(np.abs(w[o][m] - 1550.0)))
        return w[o][m], t2, k2, float(t2[i0]), float(k2[i0])

    w_u, t2_u, k2_u, t0_u, k0_u = _spec_r(td_uniform)
    w_nu, t2_nu, k2_nu, t0_nu, k0_nu = _spec_r(td_nonuniform)
    ax2.plot(
        w_u,
        t2_u,
        "-",
        color="#08519c",
        lw=2.0,
        label=f"kottke_L3 |t|² ({t0_u:.4f})",
    )
    ax2.plot(
        w_nu,
        t2_nu,
        "--",
        color="#d95f02",
        lw=2.1,
        label=f"kottke_nonuniform |t|² ({t0_nu:.4f})",
    )
    ax2.plot(
        w_u,
        k2_u,
        "-",
        color="#1b9e77",
        lw=2.0,
        label=f"kottke_L3 |κ|² ({k0_u:.4f})",
    )
    ax2.plot(
        w_nu,
        k2_nu,
        "--",
        color="#e7298a",
        lw=2.1,
        label=f"kottke_nonuniform |κ|² ({k0_nu:.4f})",
    )
    if td_ref is not None:
        w_r, t2_r, k2_r, t0_r, k0_r = _spec_r(td_ref)
        ax2.plot(
            w_r,
            t2_r,
            ":",
            color="black",
            lw=1.7,
            label=f"kottke_reference R |t|² ({t0_r:.4f})",
        )
        ax2.plot(
            w_r,
            k2_r,
            ":",
            color="0.35",
            lw=1.7,
            label=f"kottke_reference R |κ|² ({k0_r:.4f})",
        )
    ax2.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Modal power fraction |t|², |κ|²",
        ylim=(-0.02, 1.03),
        title="(c) Broadband coupling: graded vs baseline & ref",
    )
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="center right", fontsize=6.8)
    figstyle.save(fig, out_path, close=True, auto_legend_below=False)


def overlay_ring_coupler_geometry(
    ax,
    pic,
    *,
    outline_color: str = "white",
    include_labels: bool = True,
):
    """Overlay material outlines/labels, CPML boundaries, source, and monitors on ``ax``."""
    proxy_solver = _SchematicSolverProxy(pic.solver)
    eps_core = float(pic.n_eff_slab**2)
    poly_structs = [
        _PolyStructProxy(
            pic.structures_3d[0].vertices,
            eps_core,
            "Si bus/ring (n_eff=2.85)",
        ),
        _PolyStructProxy(
            pic.structures_3d[1].vertices,
            eps_core,
            "Si bus/ring (n_eff=2.85)",
        ),
    ]
    mat_names = {"background": "SiO₂ (n=1.444)"}
    sources_s, monitors_s = _build_ring_coupler_schematic_io(pic)
    schematic.overlay_geometry(
        ax,
        proxy_solver,
        plane="xy",
        structures=poly_structs,
        material_names=mat_names,
        sources=sources_s if include_labels else None,
        monitors=monitors_s if include_labels else None,
        unit="µm",
        outline_color=outline_color,
        label_materials=include_labels,
    )


def render_ring_coupler_field_map_with_overlay(
    x,
    y,
    field,
    *,
    pic,
    kind="intensity_log",
    title="",
    out_path=None,
):
    """Render a 2D field map with :func:`overlay_ring_coupler_geometry` and save via ``figstyle``."""
    figstyle.apply_style()
    fig, ax = plot_field_map(
        x,
        y,
        field,
        structures=pic.structures_3d,
        kind=kind,
        title=title,
        out_path=None,
    )
    outline_col = "white" if kind == "intensity_log" else "black"
    overlay_ring_coupler_geometry(
        ax, pic, outline_color=outline_col, include_labels=True
    )
    if out_path is not None:
        figstyle.save(fig, out_path, close=True, auto_legend_below=False)
    return fig, ax


def render_ring_coupler_movie_with_overlay(
    x,
    y,
    frames,
    times_s,
    *,
    pic,
    out_path,
    fps=12,
):
    """Render pulse propagation GIF (< 5 MB) with :func:`overlay_ring_coupler_geometry` on every frame."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    figstyle.apply_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=84)
    x_um = np.asarray(x) * 1e6
    y_um = np.asarray(y) * 1e6
    f_arr = np.asarray(frames, dtype=float)
    vmax = float(np.percentile(np.abs(f_arr), 99.6)) or 1.0

    im = ax.pcolormesh(
        x_um,
        y_um,
        (f_arr[0] / vmax).T,
        cmap=figstyle.CMAP_DIVERGING,
        vmin=-1.0,
        vmax=1.0,
        shading="auto",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cb.set_label(r"$E_y(x, y, t) / E_{\max}$")
    overlay_ring_coupler_geometry(ax, pic, outline_color="black", include_labels=True)
    ax.set(
        xlim=(float(x_um.min()), float(x_um.max())),
        ylim=(float(y_um.min()), float(y_um.max())),
        aspect="equal",
        xlabel="x [µm]",
        ylabel="y [µm]",
    )
    ttl = ax.set_title(
        f"Waveguide-to-ring coupler pulse propagation (t = {float(times_s[0])*1e15:.1f} fs)"
    )
    fig.tight_layout()

    def _update(frame_idx):
        im.set_array((f_arr[frame_idx] / vmax).T.ravel())
        ttl.set_text(
            f"Waveguide-to-ring coupler pulse propagation (t = {float(times_s[frame_idx])*1e15:.1f} fs)"
        )
        return (im, ttl)

    anim = FuncAnimation(
        fig, _update, frames=len(f_arr), interval=1000 // fps, blit=False
    )
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)

