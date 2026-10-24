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

import numpy as np
from scipy.constants import c as c_light, epsilon_0
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

from wakis.templates.photonics import PIC_MATERIALS, slab_neff


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
