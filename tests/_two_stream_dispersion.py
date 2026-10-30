# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Analytical and discrete FIT / leapfrog dispersion relations and window/tolerance helpers for V4 two-stream tests.

Provides:
- Analytic normal-mode decomposition (``two_stream_seed_mode_shares``) of the position-only
  sinusoidal initial seed into the growing mode (``a_grow``), decaying mode (``a_decay``),
  and non-decaying oscillatory ``omega_+`` branch (``a_osc``).
- Per-case rule-derived linear window bounds (``two_stream_window_bounds_wb``) setting
  ``t_start`` from ``(a_osc / a_grow) * exp(-gamma_disc(case) * t_start) <= TRANSIENT_TOL``
  and predicting ``t_end = ln(3 * ALPHA_SAT / (8 * eps * a_grow)) / gamma_disc(case)``
  (since ``E_0 / E_trap = 8 * eps / 3`` with ``gamma0^3`` cancelling at
  ``k_1 * v0 = (sqrt(3) / 2) * omega_{b,rel}``), with per-case integration horizon
  ``1.5 * t_end_pred(case)`` and suite maximum ``T_HORIZON_WB = 1.5 * max_case(t_end_pred)``
  (no window selection by trial runs).
- General two-beam continuum and discrete FIT / leapfrog dispersion solvers supporting
  symmetric counter-streaming beams ``(s_plus, s_minus) = (1, 1)``, scaled field-gather
  mutations ``(0.5, 0.5)``, opposite-force-sign pusher mutations ``(1, -1)``,
  asymmetric beam density ratios ``(1, eta)``, upper-half-plane root counting by the
  argument principle (``two_stream_discrete_uhp_root_count``), dominant-alias complex
  root searches (``two_stream_discrete_complex_roots``), and stable-mode beat envelope
  bounds (``two_stream_stable_beat_envelope_bound``).
- Tolerance bounds for the heuristic particle-lattice alias at ``k_{lat,m} = (1 + m * N_p) * k_1``
  and the order-of-magnitude quadratic ``(3/16) * |gamma_disc| * (|E_1| / E_trap)^2``
  weakly nonlinear amplitude detuning.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import c
from scipy.optimize import brentq, root

# Upper bound on the relative non-decaying omega_+ oscillatory transient share
# (a_osc / a_grow) * exp(-gamma_disc(case) * t_start) <= TRANSIENT_TOL = 1.0e-3
# defining t_start per case from that case's discrete growth rate gamma_disc(case).
TRANSIENT_TOL: float = 1.0e-3

# Upper end of the linear fit window |E_1(t_end)| <= ALPHA_SAT * E_trap, chosen so
# the quadratic O((|E_1| / E_trap)^2) weakly nonlinear growth-rate shift remains
# below 1e-4 of gamma_disc (no window selection by trial runs).
ALPHA_SAT: float = 1.0e-2

# Default initial sinusoidal density perturbation amplitude eps = delta_n / n_b
# and horizon safety multiplier T_HORIZON_WB = HORIZON_SAFETY_FACTOR * max_case(omega_b * t_end).
DEFAULT_EPS_SEED: float = 1.0e-8
HORIZON_SAFETY_FACTOR: float = 1.5


def two_stream_continuum_root(
    omega_b: float,
    k: float,
    v0: float,
    gamma0: float = 1.0,
    s_plus: float = 1.0,
    s_minus: float = 1.0,
) -> complex:
    """Solve the 1D two-beam continuum dispersion relation for the most unstable root ``omega(k)`` [rad/s].

    Solves the general cold two-beam electrostatic dispersion relation

    .. math::

        1 = \\omega_{b,\\mathrm{rel}}^2 \\left[
            \\frac{s_+}{(\\omega - k v_0)^2}
            + \\frac{s_-}{(\\omega + k v_0)^2}
        \\right],
        \\qquad \\omega_{b,\\mathrm{rel}} = \\gamma_0^{-3/2} \\omega_b,

    which expands to the quartic polynomial

    .. math::

        \\omega^4
        - \\left[2 k^2 v_0^2 + (s_+ + s_-)\\omega_{b,\\mathrm{rel}}^2\\right]\\omega^2
        - 2 (s_+ - s_-) \\omega_{b,\\mathrm{rel}}^2 k v_0\\,\\omega
        + k^2 v_0^2 \\left[k^2 v_0^2 - (s_+ + s_-)\\omega_{b,\\mathrm{rel}}^2\\right] = 0.

    Parameters
    ----------
    omega_b : float
        Reference single-beam non-relativistic plasma frequency
        ``sqrt(n_b e^2 / (eps_0 m_e))`` [rad/s].
    k : float
        Longitudinal wavenumber along ``z`` [rad/m].
    v0 : float
        Counter-streaming beam speed ``|v_z|`` [m/s].
    gamma0 : float, optional
        Unperturbed beam Lorentz factor ``(1 - (v0 / c)^2)^(-1/2)`` [-]. Default is ``1.0``.
    s_plus : float, optional
        Dimensionless susceptibility weight of the ``+v0`` beam [-]. Default is ``1.0``.
    s_minus : float, optional
        Dimensionless susceptibility weight of the ``-v0`` beam [-]. Default is ``1.0``
        (``-1.0`` when the ``-v0`` beam momentum push has reversed force sign ``-q E``;
        ``n_- / n_+`` for asymmetric beam densities).

    Returns
    -------
    complex
        Continuum root ``omega = omega_r + 1j * gamma`` [rad/s] with maximum growth
        rate ``gamma = Im(omega) >= 0``.
    """
    wb = float(omega_b) * (float(gamma0) ** (-1.5))
    kv = abs(float(k) * float(v0))
    sp = float(s_plus)
    sm = float(s_minus)

    if kv == 0.0:
        sq = (sp + sm) * wb * wb
        return complex(np.sqrt(sq), 0.0) if sq >= 0.0 else complex(0.0, np.sqrt(-sq))

    if sp == sm and sp > 0.0:
        wb_eff = np.sqrt(sp) * wb
        if kv >= np.sqrt(2.0) * wb_eff:
            om_minus_sq = kv * kv + wb_eff * wb_eff - wb_eff * np.sqrt(wb_eff * wb_eff + 4.0 * kv * kv)
            return complex(float(np.sqrt(max(0.0, om_minus_sq))), 0.0)
        disc = wb_eff * np.sqrt(wb_eff * wb_eff + 4.0 * kv * kv) - (kv * kv + wb_eff * wb_eff)
        return complex(0.0, float(np.sqrt(max(0.0, disc))))

    coeffs = np.array(
        [
            1.0,
            0.0,
            -(2.0 * kv * kv + (sp + sm) * wb * wb),
            -2.0 * (sp - sm) * wb * wb * kv,
            kv * kv * (kv * kv - (sp + sm) * wb * wb),
        ],
        dtype=np.float64,
    )
    roots = np.roots(coeffs)
    best = max(roots, key=lambda r: float(np.imag(r)))
    if float(np.imag(best)) < 1e-12 * wb:
        return complex(float(np.real(best)), 0.0)
    return complex(float(np.real(best)), float(np.imag(best)))


def two_stream_continuum_growth_rate(
    omega_b: float,
    k: float,
    v0: float,
    gamma0: float = 1.0,
    s_plus: float = 1.0,
    s_minus: float = 1.0,
) -> float:
    """Analytical continuum growth rate ``gamma(k) = Im(omega(k))`` [rad/s] of two counter-streaming beams.

    Parameters
    ----------
    omega_b : float
        Reference single-beam non-relativistic plasma frequency [rad/s].
    k : float
        Longitudinal wavenumber along ``z`` [rad/m].
    v0 : float
        Counter-streaming beam speed ``|v_z|`` [m/s].
    gamma0 : float, optional
        Unperturbed beam Lorentz factor [-]. Default is ``1.0``.
    s_plus : float, optional
        Susceptibility weight of the ``+v0`` beam [-]. Default is ``1.0``.
    s_minus : float, optional
        Susceptibility weight of the ``-v0`` beam [-]. Default is ``1.0``.

    Returns
    -------
    float
        Continuum linear growth rate ``gamma(k)`` [rad/s] (``0.0`` if stable).
    """
    return float(
        np.imag(
            two_stream_continuum_root(
                omega_b,
                k,
                v0,
                gamma0=gamma0,
                s_plus=s_plus,
                s_minus=s_minus,
            )
        )
    )


def two_stream_discrete_complex_roots(
    omega_b: float,
    k: float,
    v0: float,
    dt: float,
    dz: float,
    order: int = 1,
    gather: str = "galerkin",
    gamma0: float = 1.0,
    n_alias: int = 20,
    s_plus: float = 1.0,
    s_minus: float = 1.0,
    n_alias_roots: int = 4,
) -> list[complex]:
    """Search the complex ``omega``-plane for discrete FIT / leapfrog roots across dominant aliases ``|m| <= n_alias_roots`` (not exhaustive).

    Solves the alias-summed discrete dispersion relation from:
    1. All four continuum quartic roots (on-axis and displaced into the upper/lower
       half-plane by ``+/- 0.05j * omega_{b,rel}``);
    2. Positive imaginary-axis seeds ``+1j * gamma_0``; and
    3. Dominant finite-grid Brillouin-zone alias Doppler branches
       ``omega / omega_{b,rel} = s * (k_m * v0 / omega_{b,rel}) +/- w_m + 1j * delta_i * w_m``
       with ``w_m = |sinc(k_m * dz / 2)|^((p + p_g + 1) / 2) * sqrt(|s_s|)`` for
       ``m in {-n_alias_roots, ..., n_alias_roots} \\ {0}`` (dominant aliases
       ``|m| <= 4`` searched; not exhaustive — use ``two_stream_discrete_uhp_root_count``
       for an exhaustive upper-half-plane root count by the argument principle).

    Parameters
    ----------
    omega_b : float
        Reference single-beam non-relativistic plasma frequency [rad/s].
    k : float
        Longitudinal wavenumber [rad/m].
    v0 : float
        Beam streaming speed [m/s].
    dt : float
        Leapfrog time step [s].
    dz : float
        Longitudinal cell size [m].
    order : int, optional
        B-spline shape order ``p``. Default is ``1``.
    gather : str, optional
        Field gather mode (``"galerkin"`` or ``"momentum"``). Default is ``"galerkin"``.
    gamma0 : float, optional
        Unperturbed beam Lorentz factor [-]. Default is ``1.0``.
    n_alias : int, optional
        Half-width ``M`` of Brillouin alias summation. Default is ``20``.
    s_plus : float, optional
        Susceptibility weight of the ``+v0`` beam [-]. Default is ``1.0``.
    s_minus : float, optional
        Susceptibility weight of the ``-v0`` beam [-]. Default is ``1.0``.
    n_alias_roots : int, optional
        Number of finite-grid alias harmonics ``|m| <= n_alias_roots`` whose Doppler
        branches are seeded in the complex root search. Default is ``4``.

    Returns
    -------
    list of complex
        Unique converged complex roots ``omega`` [rad/s], sorted by ``Im(omega)`` descending.
    """
    wb = float(omega_b) * (float(gamma0) ** (-1.5))
    kv = float(k) * float(v0) / wb
    sp = float(s_plus)
    sm = float(s_minus)
    mode = str(gather).strip().lower()
    is_galerkin = mode in ("galerkin", "energy", "energy-conserving")
    p_g = (order - 1) if is_galerkin else order
    exp_tot = int(order + p_g + 1)

    m_arr = np.arange(-n_alias, n_alias + 1, dtype=np.float64)
    km = float(k) + 2.0 * np.pi * m_arr / float(dz)
    arg = 0.5 * km * float(dz)
    sinc_w = np.where(np.abs(arg) < 1e-14, 1.0, np.sin(arg) / arg) ** exp_tot

    def disp_complex_residual(xy: np.ndarray) -> list[float]:
        w = (xy[0] + 1j * xy[1]) * wb
        om_p = w - km * float(v0)
        om_m = w + km * float(v0)
        denom_p = ((2.0 / float(dt)) * np.sin(0.5 * om_p * float(dt))) ** 2
        denom_m = ((2.0 / float(dt)) * np.sin(0.5 * om_m * float(dt))) ** 2
        val = (wb * wb) * np.sum(sinc_w * (sp / denom_p + sm / denom_m)) - 1.0
        return [float(np.real(val)), float(np.imag(val))]

    coeffs = np.array(
        [
            1.0,
            0.0,
            -(2.0 * kv * kv + (sp + sm)),
            -2.0 * (sp - sm) * kv,
            kv * kv * (kv * kv - (sp + sm)),
        ],
        dtype=np.float64,
    )
    cont_roots = np.roots(coeffs)

    seeds: list[complex] = []
    for r in cont_roots:
        seeds.append(complex(r))
        seeds.append(complex(np.real(r), float(np.imag(r)) + 0.05))
        seeds.append(complex(np.real(r), float(np.imag(r)) - 0.05))
    for g0 in (0.05, 0.25, 0.5, 0.8):
        seeds.append(complex(0.0, g0))

    for m in range(-int(n_alias_roots), int(n_alias_roots) + 1):
        if m == 0:
            continue
        km_val = float(k) + 2.0 * np.pi * float(m) / float(dz)
        arg_m = 0.5 * km_val * float(dz)
        s_val = 1.0 if abs(arg_m) < 1e-14 else float(np.sin(arg_m) / arg_m)
        w_base = abs(s_val) ** (0.5 * exp_tot)
        kv_m = km_val * float(v0) / wb
        for beam_s, weight_s in ((1.0, sp), (-1.0, sm)):
            wm = max(w_base * float(np.sqrt(abs(weight_s))), 1e-6)
            for branch_sign in (-1.0, 1.0):
                seeds.append(complex(beam_s * kv_m + branch_sign * wm, 0.0))
                seeds.append(complex(beam_s * kv_m + branch_sign * wm, 0.02 * wm))

    nyquist_norm = float(np.pi / (wb * float(dt)))
    found: list[complex] = []
    for s0 in seeds:
        x0 = np.array([float(np.real(s0)), float(np.imag(s0))], dtype=np.float64)
        sol = root(disp_complex_residual, x0, tol=1e-12)
        res_norm = float(np.hypot(*disp_complex_residual(sol.x)))
        if res_norm < 1e-9 and abs(float(sol.x[0])) < nyquist_norm:
            re_w = float(sol.x[0]) * wb
            im_w = float(sol.x[1]) * wb
            if abs(im_w) < 1e-12 * wb:
                im_w = 0.0
            cand = complex(re_w, im_w)
            if not any(abs(cand - prev) < 1e-6 * wb for prev in found):
                found.append(cand)

    found.sort(key=lambda z: (-float(np.imag(z)), float(np.real(z))))
    return found


def two_stream_discrete_uhp_root_count(
    omega_b: float,
    k: float,
    v0: float,
    dt: float,
    dz: float,
    order: int = 1,
    gather: str = "galerkin",
    gamma0: float = 1.0,
    n_alias: int = 20,
    s_plus: float = 1.0,
    s_minus: float = 1.0,
    eps_im_wb: float = 1.0e-4,
    gamma_max_wb: float = 5.0,
) -> int:
    """Count discrete dispersion roots with ``Im(omega) > 0`` in the Nyquist strip by the argument principle.

    Evaluates the winding number of the pole-removed discrete dispersion function

    .. math::

        \\widetilde{D}_M(\\omega) = D_M(\\omega) \\prod_{m=-M}^{M} \\prod_{s \\in \\{+1, -1\\}}
            \\left[
                \\frac{\\sin\\!\\left(\\frac{1}{2}(\\omega - s k_m v_0)\\Delta t\\right)}
                     {\\sin\\!\\left(\\frac{1}{2}(\\omega - s k_m v_0 + i\\omega_{b,\\mathrm{rel}})\\Delta t\\right)}
            \\right]^2

    around the upper-half-plane rectangular contour
    ``[-omega_Nyq, +omega_Nyq] x [i * eps_im, i * gamma_max]`` bounded by the Nyquist
    frequencies ``omega_Nyq = pi / dt``. The factor multiplying ``D_M(omega)`` removes
    all real-axis double poles ``omega = +/- k_m * v0 (mod 2 * pi / dt)`` while
    introducing neither zeros nor poles anywhere in the closed upper half-plane
    ``Im(omega) >= 0`` (all denominator zeros lie at ``Im(omega) = -omega_{b,rel} < 0``).
    Adaptive midpoint subdivision refines any contour segment until the per-segment phase
    increment satisfies ``|Delta arg| <= 0.2`` rad.

    Parameters
    ----------
    omega_b : float
        Reference single-beam non-relativistic plasma frequency [rad/s].
    k : float
        Longitudinal wavenumber [rad/m].
    v0 : float
        Beam streaming speed [m/s].
    dt : float
        Leapfrog time step [s].
    dz : float
        Longitudinal cell size [m].
    order : int, optional
        B-spline shape order ``p``. Default is ``1``.
    gather : str, optional
        Field gather mode (``"galerkin"`` or ``"momentum"``). Default is ``"galerkin"``.
    gamma0 : float, optional
        Unperturbed beam Lorentz factor [-]. Default is ``1.0``.
    n_alias : int, optional
        Half-width ``M`` of Brillouin alias summation. Default is ``20``.
    s_plus : float, optional
        Susceptibility weight of the ``+v0`` beam [-]. Default is ``1.0``.
    s_minus : float, optional
        Susceptibility weight of the ``-v0`` beam [-]. Default is ``1.0``.
    eps_im_wb : float, optional
        Bottom contour height ``eps_im / omega_{b,rel}`` above the real axis [-].
        Default is ``1e-4``.
    gamma_max_wb : float, optional
        Top contour height ``gamma_max / omega_{b,rel}`` [-]. Default is ``5.0``.

    Returns
    -------
    int
        Exact count of roots with ``Im(omega) > eps_im`` in ``Re(omega) in [-pi/dt, +pi/dt]``.
        Roots with ``0 < Im(omega) < eps_im`` lie below the contour and are not counted;
        their growth over the test horizons is below detection anyway.

    Raises
    ------
    RuntimeError
        If adaptive refinement leaves any contour segment unresolved.
    """
    wb = float(omega_b) * (float(gamma0) ** (-1.5))
    sp = float(s_plus)
    sm = float(s_minus)
    mode = str(gather).strip().lower()
    is_galerkin = mode in ("galerkin", "energy", "energy-conserving")
    p_g = (order - 1) if is_galerkin else order
    exp_tot = int(order + p_g + 1)

    m_arr = np.arange(-int(n_alias), int(n_alias) + 1, dtype=np.float64)
    km = float(k) + 2.0 * np.pi * m_arr / float(dz)
    arg = 0.5 * km * float(dz)
    sinc_w = np.where(np.abs(arg) < 1e-14, 1.0, np.sin(arg) / arg) ** exp_tot

    w_nyq = float(np.pi / float(dt))
    eps_im = float(eps_im_wb) * wb
    g_max = float(gamma_max_wb) * wb

    def eval_disp(w_arr: np.ndarray) -> np.ndarray:
        w = w_arr[:, None]
        om_p = w - km[None, :] * float(v0)
        om_m = w + km[None, :] * float(v0)
        dp = ((2.0 / float(dt)) * np.sin(0.5 * om_p * float(dt))) ** 2
        dm = ((2.0 / float(dt)) * np.sin(0.5 * om_m * float(dt))) ** 2
        return 1.0 - (wb * wb) * np.sum(
            sinc_w[None, :] * (sp / dp + sm / dm), axis=1
        )

    def eval_pole_factor_ratios(w_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = w_arr[:, None]
        om_p = w - km[None, :] * float(v0)
        om_m = w + km[None, :] * float(v0)
        rp = np.sin(0.5 * om_p * float(dt)) / np.sin(
            0.5 * (om_p + 1j * wb) * float(dt)
        )
        rm = np.sin(0.5 * om_m * float(dt)) / np.sin(
            0.5 * (om_m + 1j * wb) * float(dt)
        )
        return rp, rm

    # Seed the bottom contour edge with all folded Doppler pole locations and nearby
    # real-branch offsets so no zero-pole pair can be straddled before adaptive refinement.
    features: list[float] = [0.0]
    for m_idx in range(km.size):
        wm = float(np.sqrt(abs(sinc_w[m_idx]))) * wb
        for beam_s in (-1.0, 1.0):
            p_loc = beam_s * float(km[m_idx]) * float(v0)
            for off in (-2.0 * wm, -wm, -0.5 * wm, 0.0, 0.5 * wm, wm, 2.0 * wm):
                pf = ((p_loc + off + w_nyq) % (2.0 * w_nyq)) - w_nyq
                features.extend(
                    [
                        pf - 2.0 * eps_im,
                        pf - 0.5 * eps_im,
                        pf,
                        pf + 0.5 * eps_im,
                        pf + 2.0 * eps_im,
                    ]
                )
    re_bot = np.unique(
        np.clip(
            np.concatenate(
                [np.linspace(-w_nyq, w_nyq, 2000), np.asarray(features, dtype=np.float64)]
            ),
            -w_nyq,
            w_nyq,
        )
    )
    w_bot = re_bot + 1j * eps_im
    w_right = w_nyq + 1j * np.linspace(eps_im, g_max, 200)
    w_top = np.linspace(w_nyq, -w_nyq, 200) + 1j * g_max
    w_left = -w_nyq + 1j * np.linspace(g_max, eps_im, 200)

    w_contour = np.concatenate([w_bot[:-1], w_right[:-1], w_top[:-1], w_left])
    d_val = eval_disp(w_contour)
    for _ in range(30):
        dtheta = np.angle(d_val[1:] / d_val[:-1])
        denom_min = np.maximum(
            np.minimum(np.abs(d_val[1:]), np.abs(d_val[:-1])), 1e-30
        )
        rel_jump = np.abs(d_val[1:] - d_val[:-1]) / denom_min
        bad = np.flatnonzero((np.abs(dtheta) > 0.2) | (rel_jump > 0.5))
        if bad.size == 0:
            break
        mids = 0.5 * (w_contour[bad] + w_contour[bad + 1])
        mid_d = eval_disp(mids)
        w_contour = np.insert(w_contour, bad + 1, mids)
        d_val = np.insert(d_val, bad + 1, mid_d)
    # The principal-value phase sum around a closed loop is always a multiple of 2*pi, so
    # the count is only correct if every segment's increment is resolved (|dtheta| < pi).
    # An unresolved contour would return a wrong integer silently: refuse instead.
    if bad.size > 0:
        raise RuntimeError(
            f"two_stream_discrete_uhp_root_count: {bad.size} contour segment(s) still "
            "unresolved after 30 refinement passes; the winding number is not reliable"
        )

    rp, rm = eval_pole_factor_ratios(w_contour)
    dtheta_d = np.angle(d_val[1:] / d_val[:-1])
    dtheta_poles = 2.0 * np.sum(
        np.angle(rp[1:] / rp[:-1]) + np.angle(rm[1:] / rm[:-1]), axis=1
    )
    winding = float(np.sum(dtheta_d + dtheta_poles) / (2.0 * np.pi))
    # Round-off guard only (an integer by construction once every segment is resolved).
    return int(round(winding))  # noqa: RUF046 - keep a Python int for numpy floats


def two_stream_discrete_root(
    omega_b: float,
    k: float,
    v0: float,
    dt: float,
    dz: float,
    order: int = 1,
    gather: str = "galerkin",
    gamma0: float = 1.0,
    n_alias: int = 20,
    s_plus: float = 1.0,
    s_minus: float = 1.0,
) -> complex:
    """Solve the discrete FIT / leapfrog two-stream dispersion relation for ``omega_disc(k)`` [rad/s].

    Solves the alias-summed discrete dispersion relation

    .. math::

        1 = \\omega_{b,\\mathrm{rel}}^2 \\sum_{m=-M}^{M}
            \\left[\\operatorname{sinc}\\!\\left(\\frac{k_m \\Delta z}{2}\\right)\\right]^{p + p_g + 1}
            \\left\\{
                \\frac{s_+}{\\Omega_t^2(\\omega - k_m v_0)}
                + \\frac{s_-}{\\Omega_t^2(\\omega + k_m v_0)}
            \\right\\},

    where ``k_m = k + 2 * pi * m / dz`` and ``Omega_t(w) = (2 / dt) * sin(w * dt / 2)``.

    Parameters
    ----------
    omega_b : float
        Reference single-beam non-relativistic plasma frequency [rad/s].
    k : float
        Longitudinal wavenumber [rad/m].
    v0 : float
        Beam streaming speed [m/s].
    dt : float
        Leapfrog time step [s].
    dz : float
        Longitudinal cell size [m].
    order : int, optional
        B-spline shape order ``p``. Default is ``1``.
    gather : str, optional
        Field gather mode (``"galerkin"`` or ``"momentum"``). Default is ``"galerkin"``.
    gamma0 : float, optional
        Unperturbed beam Lorentz factor [-]. Default is ``1.0``.
    n_alias : int, optional
        Half-width ``M`` of Brillouin alias summation. Default is ``20``.
    s_plus : float, optional
        Susceptibility weight of the ``+v0`` beam [-]. Default is ``1.0``.
    s_minus : float, optional
        Susceptibility weight of the ``-v0`` beam [-]. Default is ``1.0``.

    Returns
    -------
    complex
        Discrete complex root ``omega_disc = omega_r + 1j * gamma_disc`` [rad/s].
    """
    wb = float(omega_b) * (float(gamma0) ** (-1.5))
    sp = float(s_plus)
    sm = float(s_minus)
    mode = str(gather).strip().lower()
    is_galerkin = mode in ("galerkin", "energy", "energy-conserving")
    p_g = (order - 1) if is_galerkin else order
    exp_tot = int(order + p_g + 1)

    m_arr = np.arange(-n_alias, n_alias + 1, dtype=np.float64)
    km = float(k) + 2.0 * np.pi * m_arr / float(dz)
    arg = 0.5 * km * float(dz)
    sinc_w = np.where(np.abs(arg) < 1e-14, 1.0, np.sin(arg) / arg) ** exp_tot

    if sp == sm and sp > 0.0:
        wb_eff = np.sqrt(sp) * wb

        def disp_imag_residual(gamma: float) -> float:
            om_p = 1j * gamma - km * float(v0)
            om_m = 1j * gamma + km * float(v0)
            denom_p = ((2.0 / float(dt)) * np.sin(0.5 * om_p * float(dt))) ** 2
            denom_m = ((2.0 / float(dt)) * np.sin(0.5 * om_m * float(dt))) ** 2
            val = (wb_eff * wb_eff) * np.sum(sinc_w * (1.0 / denom_p + 1.0 / denom_m))
            return float(np.real(val) - 1.0)

        if disp_imag_residual(1e-9 * wb_eff) > 0.0:
            g_root = float(brentq(disp_imag_residual, 1e-9 * wb_eff, 0.999 * wb_eff))
            return complex(0.0, g_root)

        # When no purely imaginary root exists, search the complex plane including
        # off-axis continuum and finite-grid alias branches.
        c_roots = two_stream_discrete_complex_roots(
            omega_b,
            k,
            v0,
            dt,
            dz,
            order=order,
            gather=gather,
            gamma0=gamma0,
            n_alias=n_alias,
            s_plus=sp,
            s_minus=sm,
        )
        if c_roots:
            best = max(c_roots, key=lambda r: float(np.imag(r)))
            if float(np.imag(best)) > 0.0:
                return best
            pos_real = [float(np.real(r)) for r in c_roots if float(np.real(r)) > 0.0]
            if pos_real:
                return complex(min(pos_real), 0.0)
        return complex(0.0, 0.0)

    w_cont = two_stream_continuum_root(
        omega_b, k, v0, gamma0=gamma0, s_plus=sp, s_minus=sm
    )
    if np.imag(w_cont) <= 0.0:
        return complex(float(np.real(w_cont)), 0.0)

    def disp_complex_residual(xy: np.ndarray) -> list[float]:
        w = (xy[0] + 1j * xy[1]) * wb
        om_p = w - km * float(v0)
        om_m = w + km * float(v0)
        denom_p = ((2.0 / float(dt)) * np.sin(0.5 * om_p * float(dt))) ** 2
        denom_m = ((2.0 / float(dt)) * np.sin(0.5 * om_m * float(dt))) ** 2
        val = (wb * wb) * np.sum(sinc_w * (sp / denom_p + sm / denom_m)) - 1.0
        return [float(np.real(val)), float(np.imag(val))]

    x0 = np.array([np.real(w_cont) / wb, np.imag(w_cont) / wb], dtype=np.float64)
    sol = root(disp_complex_residual, x0, tol=1e-12)
    if not sol.success and float(np.hypot(*disp_complex_residual(sol.x))) > 1e-10:
        raise RuntimeError(f"Discrete two-stream root finder failed to converge: {sol.message}")
    return complex(float(sol.x[0] * wb), max(0.0, float(sol.x[1] * wb)))


def two_stream_stable_beat_envelope_bound(
    omega_b: float,
    k: float,
    v0: float,
    dt: float,
    dz: float,
    order: int = 1,
    gather: str = "galerkin",
    gamma0: float = 1.0,
) -> dict[str, float]:
    """Compute the two-frequency beat envelope and derived sub-window maximum tolerance at a stable ``k``.

    At a stable wavenumber ``k * v0 > sqrt(2) * omega_{b,rel}`` (such as ``k_2 = 2 * k_1``
    where ``k_2 * v0 = sqrt(3) * omega_{b,rel}``), all discrete roots are real and the
    position-only seed excites a two-frequency beat between the slow branch ``omega_-``
    and fast branch ``omega_+``:

    .. math::

        \\frac{\\hat{E}(t)}{i E_0}
        = A_-\\,\\cos(\\omega_- t - \\varphi_-) - A_+\\,\\cos(\\omega_+ t - \\varphi_+),
        \\qquad A_{\\max} = A_- + A_+,

    where

    .. math::

        \\frac{A_-}{E_0} = \\frac{k^2 v_0^2 - \\omega_-^2}{\\omega_+^2 - \\omega_-^2}
            \\sqrt{1 + \\frac{3 k^2 v_0^2}{\\omega_-^2}},
        \\qquad
        \\frac{A_+}{E_0} = \\frac{\\omega_+^2 - k^2 v_0^2}{\\omega_+^2 - \\omega_-^2}
            \\sqrt{1 + \\frac{3 k^2 v_0^2}{\\omega_+^2}}.

    Over any sub-window of length ``T_sub >= pi / omega_-`` (spanning at least one slow
    half-period), the fast oscillation ``-cos(omega_+ * t - phi_+)`` attains local extrema
    of the required same sign as the slow peak every ``2 * pi / omega_+`` (i.e. within
    ``|t - t_*| <= pi / omega_+`` of the slow peak ``t_*``). At the fast extremum of the
    required opposite sign closest to the slow peak, the slow-mode phase offset can be up
    to ``Delta_theta = pi * (omega_- / omega_+)``, bounding the relative difference
    between the last-third and first-third envelope maxima ``M_3`` and ``M_1`` by

    .. math::

        \\frac{|M_3 - M_1|}{\\max(M_1, M_3)} \\le \\mathrm{tol}_{\\mathrm{env}}
        \\equiv \\frac{A_-}{A_{\\max}}\\,\\left(1 - \\cos\\Delta\\theta\\right),
        \\qquad \\Delta\\theta = \\pi\\,\\frac{\\omega_-}{\\omega_+}.

    For a 3-third envelope test over ``[0, T_stable]``, the shortest horizon covering one
    slow half-period ``pi / omega_-`` per third (and thus ``(3/2) * (omega_+ / omega_- - 1)``
    beat periods ``2 * pi / (omega_+ - omega_-)``) is ``T_stable = 3 * pi / omega_-``,
    yielding minimum detectable growth rate
    ``gamma_min_detectable = ln(1 + tol_env) / ((2 / 3) * T_stable)``.

    Parameters
    ----------
    omega_b : float
        Reference single-beam non-relativistic plasma frequency [rad/s].
    k : float
        Stable longitudinal wavenumber ``k * v0 > sqrt(2) * omega_{b,rel}`` [rad/m].
    v0 : float
        Beam streaming speed [m/s].
    dt : float
        Leapfrog time step [s].
    dz : float
        Longitudinal cell size [m].
    order : int, optional
        B-spline shape order ``p``. Default is ``1``.
    gather : str, optional
        Field gather mode (``"galerkin"`` or ``"momentum"``). Default is ``"galerkin"``.
    gamma0 : float, optional
        Unperturbed beam Lorentz factor [-]. Default is ``1.0``.

    Returns
    -------
    dict of str to float
        Dictionary with keys ``"omega_minus_disc"``, ``"omega_plus_disc"``,
        ``"a_minus_over_e0"``, ``"a_plus_over_e0"``, ``"a_max_over_e0"``,
        ``"delta_theta"``, ``"tol_env_rel"``, ``"t_stable_wb"``, and
        ``"gamma_min_detectable"``.
    """
    wb_rel = float(omega_b) * (float(gamma0) ** (-1.5))
    kv = float(k) * float(v0)
    if kv <= np.sqrt(2.0) * wb_rel:
        raise ValueError(f"Wavenumber k*v0={kv} is not in the stable regime > sqrt(2)*wb_rel")

    om_m_cont = float(
        np.sqrt(
            kv * kv
            + wb_rel * wb_rel
            - wb_rel * np.sqrt(wb_rel * wb_rel + 4.0 * kv * kv)
        )
    )
    om_p_cont = float(
        np.sqrt(
            kv * kv
            + wb_rel * wb_rel
            + wb_rel * np.sqrt(wb_rel * wb_rel + 4.0 * kv * kv)
        )
    )

    c_roots = two_stream_discrete_complex_roots(
        omega_b,
        k,
        v0,
        dt,
        dz,
        order=order,
        gather=gather,
        gamma0=gamma0,
        n_alias_roots=2,
    )
    pos_real = [float(np.real(r)) for r in c_roots if float(np.real(r)) > 0.0]
    om_m_disc = min(pos_real, key=lambda w: abs(w - om_m_cont))
    om_p_disc = min(pos_real, key=lambda w: abs(w - om_p_cont))

    denom = om_p_disc * om_p_disc - om_m_disc * om_m_disc
    alpha_m = (kv * kv - om_m_disc * om_m_disc) / denom
    alpha_p = (om_p_disc * om_p_disc - kv * kv) / denom
    a_m = float(alpha_m * np.sqrt(1.0 + 3.0 * (kv * kv) / (om_m_disc * om_m_disc)))
    a_p = float(alpha_p * np.sqrt(1.0 + 3.0 * (kv * kv) / (om_p_disc * om_p_disc)))
    a_max = a_m + a_p

    delta_theta = float(np.pi * (om_m_disc / om_p_disc))
    tol_env_rel = float((a_m / a_max) * (1.0 - np.cos(delta_theta)))
    t_stable_wb = float(3.0 * np.pi * float(omega_b) / om_m_disc)
    dt_thirds_s = (2.0 / 3.0) * (t_stable_wb / float(omega_b))
    gamma_min_detectable = float(np.log(1.0 + tol_env_rel) / dt_thirds_s)
    return {
        "omega_minus_disc": om_m_disc,
        "omega_plus_disc": om_p_disc,
        "a_minus_over_e0": a_m,
        "a_plus_over_e0": a_p,
        "a_max_over_e0": a_max,
        "delta_theta": delta_theta,
        "tol_env_rel": tol_env_rel,
        "t_stable_wb": t_stable_wb,
        "gamma_min_detectable": gamma_min_detectable,
    }


def two_stream_discrete_growth_rate(
    omega_b: float,
    k: float,
    v0: float,
    dt: float,
    dz: float,
    order: int = 1,
    gather: str = "galerkin",
    gamma0: float = 1.0,
    n_alias: int = 20,
    s_plus: float = 1.0,
    s_minus: float = 1.0,
) -> float:
    """Discrete FIT / leapfrog growth rate ``gamma_disc(k) = Im(omega_disc(k))`` [rad/s].

    Parameters
    ----------
    omega_b : float
        Reference single-beam non-relativistic plasma frequency [rad/s].
    k : float
        Longitudinal wavenumber [rad/m].
    v0 : float
        Beam streaming speed [m/s].
    dt : float
        Leapfrog time step [s].
    dz : float
        Longitudinal cell size [m].
    order : int, optional
        B-spline shape order ``p``. Default is ``1``.
    gather : str, optional
        Field gather mode (``"galerkin"`` or ``"momentum"``). Default is ``"galerkin"``.
    gamma0 : float, optional
        Unperturbed beam Lorentz factor [-]. Default is ``1.0``.
    n_alias : int, optional
        Half-width ``M`` of Brillouin alias summation. Default is ``20``.
    s_plus : float, optional
        Susceptibility weight of the ``+v0`` beam [-]. Default is ``1.0``.
    s_minus : float, optional
        Susceptibility weight of the ``-v0`` beam [-]. Default is ``1.0``.

    Returns
    -------
    float
        Discrete linear growth rate ``gamma_disc(k)`` [rad/s] (``0.0`` if stable).
    """
    return float(
        np.imag(
            two_stream_discrete_root(
                omega_b,
                k,
                v0,
                dt,
                dz,
                order=order,
                gather=gather,
                gamma0=gamma0,
                n_alias=n_alias,
                s_plus=s_plus,
                s_minus=s_minus,
            )
        )
    )


def two_stream_lattice_alias_bound(
    gamma_disc: float,
    n_particles_beam: int,
    order: int = 1,
    gather: str = "galerkin",
    n_harmonics: int = 20,
) -> float:
    """Return the heuristic particle-lattice alias bound ``delta_gamma_lat`` [rad/s].

    On a quiet-start uniform lattice ``z_{p,0} = (p + 1/2) * Lz / N_p`` with
    ``N_p = Nz * n_ppc_z`` macroparticles per beam, Poisson summation aliases the
    seeded mode ``k_1 = 2 * pi / Lz`` to the particle-lattice harmonics
    ``k_{lat,m} = (1 + m * N_p) * k_1`` (``m != 0``). Since
    ``k_{lat,m} * dz / 2 = k_1 * dz / 2 + pi * m * n_ppc_z``, the order-``p`` deposit
    and order-``p_g`` gather B-spline symbol ``sinc^(p + p_g + 1)(k * dz / 2)``
    attenuates the ``m``-th particle-lattice alias relative to ``k_1`` by
    ``|1 + m * N_p|^(-(p + p_g + 1))``, motivating the heuristic particle-lattice alias
    bound:

    .. math::

        \\delta\\gamma_{\\mathrm{lat}} = |\\gamma_{\\mathrm{disc}}| \\sum_{m=1}^{M_{\\mathrm{lat}}}
            \\left[
                \\frac{1}{(m N_p - 1)^{p + p_g + 1}}
                + \\frac{1}{(m N_p + 1)^{p + p_g + 1}}
            \\right]
        \\simeq \\frac{2\\,\\zeta(p + p_g + 1)}{N_p^{p + p_g + 1}}\\,|\\gamma_{\\mathrm{disc}}|.

    Parameters
    ----------
    gamma_disc : float
        Discrete growth rate prediction [rad/s].
    n_particles_beam : int
        Total longitudinal macroparticles per beam ``N_p = Nz * n_ppc_z`` [-].
    order : int, optional
        B-spline shape order ``p``. Default is ``1``.
    gather : str, optional
        Field gather mode (``"galerkin"`` or ``"momentum"``). Default is ``"galerkin"``.
    n_harmonics : int, optional
        Number of particle-lattice harmonic pairs ``m = 1, ..., M_lat`` summed. Default is ``20``.

    Returns
    -------
    float
        Heuristic particle-lattice alias bound ``delta_gamma_lat`` [rad/s].
    """
    mode = str(gather).strip().lower()
    is_galerkin = mode in ("galerkin", "energy", "energy-conserving")
    p_g = (order - 1) if is_galerkin else order
    exp_tot = int(order + p_g + 1)
    n_p = float(max(int(n_particles_beam), 2))
    m_arr = np.arange(1, int(n_harmonics) + 1, dtype=np.float64)
    alias_sum = float(
        np.sum((m_arr * n_p - 1.0) ** (-exp_tot) + (m_arr * n_p + 1.0) ** (-exp_tot))
    )
    return abs(float(gamma_disc)) * alias_sum


def two_stream_nonlinear_bound(
    gamma_disc: float,
    amp_end: float,
    e_trap: float,
) -> float:
    """Return the order-of-magnitude quadratic ``(3/16) * |gamma_disc| * (|E_1| / E_trap)^2`` nonlinear detuning bound ``delta_gamma_nl`` [rad/s].

    By translational symmetry ``z -> z + pi / k_1`` (``E_1 -> -E_1``), the leading
    nonlinear correction to the fundamental mode ``k_1`` growth rate is quadratic in
    the mode amplitude. From ``-i * gamma0^3 * m_e * (omega - s * k_1 * v_0) * delta_v_{z,1} = -e * E_1``
    and ``E_trap = gamma0^3 * m_e * k_1 * v_0^2 / (2 * e)``, at the optimal wavenumber
    ``k_1 * v_0 = (sqrt(3) / 2) * omega_{b,rel}`` where ``gamma = 0.5 * omega_{b,rel}`` and
    ``|i * gamma - k_1 * v_0| = omega_{b,rel}``, the linear beam velocity modulation is
    ``|delta_v_{z,1}| / v_0 = (sqrt(3) / 4) * (|E_1| / E_trap)``. Using the
    order-of-magnitude bound ``delta_gamma_nl / |gamma_disc| ~ (|delta_v_{z,1}| / v_0)^2``
    gives:

    .. math::

        \\delta\\gamma_{\\mathrm{nl}}
        = |\\gamma_{\\mathrm{disc}}|\\,\\left(\\frac{|\\delta v_{z,1}(t_{\\mathrm{end}})|}{v_0}\\right)^2
        = \\frac{3}{16}\\,|\\gamma_{\\mathrm{disc}}|\\,\\left(\\frac{|E_1(t_{\\mathrm{end}})|}{E_{\\mathrm{trap}}}\\right)^2.

    Parameters
    ----------
    gamma_disc : float
        Discrete growth rate prediction [rad/s].
    amp_end : float
        Fundamental mode amplitude ``|E_{z,1}(t_end)|`` at the end of the fit window [V/m].
    e_trap : float
        Wave-particle trapping amplitude ``E_trap = gamma0^3 * m_e * k_1 * v0^2 / (2 * e)`` [V/m].

    Returns
    -------
    float
        Order-of-magnitude quadratic nonlinear detuning bound ``delta_gamma_nl`` [rad/s].
    """
    ratio = float(amp_end) / max(float(e_trap), 1e-30)
    return (3.0 / 16.0) * abs(float(gamma_disc)) * (ratio * ratio)


def two_stream_seed_mode_shares(
    s_plus: float = 1.0,
    s_minus: float = 1.0,
    density_ratio_minus: float = 1.0,
    kv_over_wb_rel: float = float(np.sqrt(3.0) / 2.0),
    phase_plus: float = float(np.pi / 3.0),
    phase_minus: float = float(-np.pi / 3.0),
) -> dict[str, float]:
    """Compute analytic normal-mode shares of the position-only initial seed via Laplace residues.

    For initial sinusoidal displacements ``delta_z_a(z, 0) = -(eps / k) * sin(k * z + phi_a)``
    and zero initial velocity perturbation ``delta_v_{z,a}(z, 0) = 0``, the initial complex
    electric field harmonic contributed by beam ``a`` (normalized by
    ``E_0 = e * n_b * eps / (eps_0 * k) = (8 * eps / 3) * E_trap`` at
    ``k * v_0 = (sqrt(3) / 2) * omega_{b,rel}``, since ``gamma0^3`` cancels between
    ``E_trap`` and ``k^2 * v_0^2 = (3 / 4) * gamma0^(-3) * omega_b^2``) is
    ``E_a(0) / E_0 = i * d_a * exp(i * phi_a)``
    with ``d_+ = 1`` and ``d_- = density_ratio_minus``. By Laplace inversion of the linearized
    continuity-momentum-Gauss system, the complex mode amplitude of root ``omega_j`` of the
    quartic dispersion polynomial ``P(omega) = 0`` is

    .. math::

        \\frac{c_j}{E_0} = \\frac{(\\hat{\\omega}_j^2 - \\hat{k}^2)
            \\left[(\\hat{\\omega}_j + \\hat{k})\\,i d_+ e^{i\\phi_+}
            + (\\hat{\\omega}_j - \\hat{k})\\,i d_- e^{i\\phi_-}\\right]}{P'(\\hat{\\omega}_j)},

    where ``\\hat{\\omega} = \\omega / \\omega_{b,\\mathrm{rel}}`` and ``\\hat{k} = k v_0 / \\omega_{b,\\mathrm{rel}}``.
    For equal symmetric beams (``s_+ = s_- = 1, d_+ = d_- = 1, phi_+/- = +/- pi/3``) at
    ``\\hat{k} = \\sqrt{3}/2``, this gives ``a_grow = 1/2``, ``a_decay = 1/4``, and
    non-decaying ``\\omega_+`` oscillatory envelope ``a_osc = 3 / \\sqrt{10}``.

    Parameters
    ----------
    s_plus : float, optional
        Susceptibility weight of the ``+v0`` beam [-]. Default is ``1.0``.
    s_minus : float, optional
        Susceptibility weight of the ``-v0`` beam [-]. Default is ``1.0``.
    density_ratio_minus : float, optional
        Initial density weight ratio ``n_- / n_+`` [-]. Default is ``1.0``.
    kv_over_wb_rel : float, optional
        Normalized wavenumber ``k * v_0 / omega_{b,rel}`` [-]. Default is ``sqrt(3) / 2``.
    phase_plus : float, optional
        Initial displacement phase ``phi_+`` [rad]. Default is ``+pi / 3``.
    phase_minus : float, optional
        Initial displacement phase ``phi_-`` [rad]. Default is ``-pi / 3``.

    Returns
    -------
    dict of str to float
        Dictionary with keys ``"a_grow"`` (growing mode share ``|c_grow| / E_0``),
        ``"a_osc"`` (non-decaying oscillatory ``omega_+`` envelope share
        ``sum_{j != grow, Im(omega_j) >= 0} |c_j| / E_0``), ``"a_decay"`` (decaying mode
        share ``sum_{Im(omega_j) < 0} |c_j| / E_0``), and ``"e_init_over_e0"``
        (``|E(0)| / E_0``).
    """
    kv = float(kv_over_wb_rel)
    sp = float(s_plus)
    sm = float(s_minus)
    dp = 1.0
    dm = float(density_ratio_minus)

    coeffs = np.array(
        [
            1.0,
            0.0,
            -(2.0 * kv * kv + (sp + sm)),
            -2.0 * (sp - sm) * kv,
            kv * kv * (kv * kv - (sp + sm)),
        ],
        dtype=np.float64,
    )
    roots = np.roots(coeffs)
    p_deriv = np.polyder(coeffs)

    ep0 = 1j * dp * np.exp(1j * float(phase_plus))
    em0 = 1j * dm * np.exp(1j * float(phase_minus))
    e_init_over_e0 = float(np.abs(ep0 + em0))

    c_j = np.array(
        [
            ((r * r - kv * kv) * ((r + kv) * ep0 + (r - kv) * em0))
            / np.polyval(p_deriv, r)
            for r in roots
        ],
        dtype=np.complex128,
    )
    idx_grow = int(np.argmax(np.imag(roots)))
    a_grow = float(np.abs(c_j[idx_grow]))
    osc_indices = [
        i for i in range(len(roots)) if i != idx_grow and float(np.imag(roots[i])) >= -1e-12
    ]
    dec_indices = [i for i in range(len(roots)) if float(np.imag(roots[i])) < -1e-12]
    a_osc = float(sum(np.abs(c_j[i]) for i in osc_indices))
    a_decay = float(sum(np.abs(c_j[i]) for i in dec_indices))
    return {
        "a_grow": a_grow,
        "a_osc": a_osc,
        "a_decay": a_decay,
        "e_init_over_e0": e_init_over_e0,
    }


def two_stream_window_bounds_wb(
    omega_b: float,
    gamma_disc: float,
    gamma0: float = 1.0,
    eps: float = DEFAULT_EPS_SEED,
    alpha_sat: float = ALPHA_SAT,
    transient_tol: float = TRANSIENT_TOL,
    s_plus: float = 1.0,
    s_minus: float = 1.0,
    density_ratio_minus: float = 1.0,
    kv_over_wb_rel: float = float(np.sqrt(3.0) / 2.0),
) -> dict[str, float]:
    """Compute per-case dimensionless linear fit window bounds ``[omega_b * t_start, omega_b * t_end_pred]``.

    Sets ``t_start`` per case from the requirement that the non-decaying ``omega_+``
    oscillatory share ``a_osc`` relative to the growing mode ``a_grow * exp(gamma_disc * t)``
    falls below ``transient_tol`` (``1e-3``):

    .. math::

        \\frac{a_{\\mathrm{osc}}}{a_{\\mathrm{grow}}}\\,e^{-\\gamma_{\\mathrm{disc}}\\,t_{\\mathrm{start}}} \\le \\mathrm{transient\\_tol}
        \\qquad \\Longleftrightarrow \\qquad
        t_{\\mathrm{start}} = \\frac{\\ln\\!\\left(\\dfrac{a_{\\mathrm{osc}}}{\\mathrm{transient\\_tol}\\,a_{\\mathrm{grow}}}\\right)}{\\gamma_{\\mathrm{disc}}},

    and derives the predicted saturation window end time ``t_end_pred`` at which the
    growing mode ``a_grow * E_0 * exp(gamma_disc * t)`` with ``E_0 / E_trap = 8 * eps / 3``
    (where ``gamma0^3`` cancels because ``k_1`` is chosen from ``omega_{b,rel}`` via
    ``k_1^2 * v_0^2 = (3 / 4) * omega_{b,rel}^2 = (3 / 4) * gamma0^(-3) * omega_b^2``)
    reaches ``alpha_sat * E_trap``:

    .. math::

        t_{\\mathrm{end,pred}} = \\frac{\\ln\\!\\left(\\dfrac{3\\,\\alpha_{\\mathrm{sat}}}{8\\varepsilon\\,a_{\\mathrm{grow}}}\\right)}{\\gamma_{\\mathrm{disc}}}.

    Parameters
    ----------
    omega_b : float
        Reference single-beam non-relativistic plasma frequency [rad/s].
    gamma_disc : float
        Case-specific discrete growth rate ``gamma_disc(case)`` [rad/s].
    gamma0 : float, optional
        Unperturbed beam Lorentz factor [-]. Default is ``1.0``.
    eps : float, optional
        Initial sinusoidal density perturbation amplitude ``delta_n / n_b`` [-].
        Default is ``DEFAULT_EPS_SEED`` (``1e-8``).
    alpha_sat : float, optional
        Trapping-field fraction defining the upper end of the linear window [-].
        Default is ``ALPHA_SAT`` (``1e-2``).
    transient_tol : float, optional
        Upper bound on the relative ``omega_+`` oscillatory transient share at ``t_start`` [-].
        Default is ``TRANSIENT_TOL`` (``1e-3``).
    s_plus : float, optional
        Susceptibility weight of the ``+v0`` beam [-]. Default is ``1.0``.
    s_minus : float, optional
        Susceptibility weight of the ``-v0`` beam [-]. Default is ``1.0``.
    density_ratio_minus : float, optional
        Initial density weight ratio ``n_- / n_+`` [-]. Default is ``1.0``.
    kv_over_wb_rel : float, optional
        Normalized wavenumber ``k * v_0 / omega_{b,rel}`` [-]. Default is ``sqrt(3) / 2``.

    Returns
    -------
    dict of str to float
        Dictionary with keys ``"t_start_wb"``, ``"t_end_pred_wb"``, ``"t_horizon_wb"``,
        ``"a_grow"``, ``"a_osc"``, ``"a_decay"``, and ``"e_init_over_e0"``.
    """
    shares = two_stream_seed_mode_shares(
        s_plus=s_plus,
        s_minus=s_minus,
        density_ratio_minus=density_ratio_minus,
        kv_over_wb_rel=kv_over_wb_rel,
    )
    a_grow = shares["a_grow"]
    a_osc = shares["a_osc"]
    g_over_wb = float(gamma_disc) / float(omega_b)
    if g_over_wb <= 0.0:
        raise ValueError(f"gamma_disc must be positive for an unstable mode, got {gamma_disc}")

    t_start_wb = float(np.log(a_osc / (float(transient_tol) * a_grow)) / g_over_wb)
    t_end_pred_wb = float(
        np.log((3.0 * float(alpha_sat)) / (8.0 * float(eps) * a_grow)) / g_over_wb
    )
    t_horizon_wb = float(HORIZON_SAFETY_FACTOR * t_end_pred_wb)
    return {
        "t_start_wb": t_start_wb,
        "t_end_pred_wb": t_end_pred_wb,
        "t_horizon_wb": t_horizon_wb,
        **shares,
    }


def two_stream_cfl_dt(
    dz: float,
    cfln: float = 0.8,
    transverse_aspect: float = 8.0,
) -> float:
    """Return the 3D FIT Courant time step ``dt`` [s] for ``dx = dy = transverse_aspect * dz``."""
    return (
        float(cfln)
        / float(np.sqrt(1.0 + 2.0 / (float(transverse_aspect) ** 2)))
    ) * (float(dz) / c)


def two_stream_suite_horizon_wb(
    omega_b: float = 2.0 * np.pi * 1.0e9,
    v0_over_c: float = 0.06,
    Nz: int = 16,
    cfln: float = 0.8,
    transverse_aspect: float = 8.0,
    eps: float = DEFAULT_EPS_SEED,
    alpha_sat: float = ALPHA_SAT,
    safety_factor: float = HORIZON_SAFETY_FACTOR,
) -> float:
    """Derive the suite maximum integration horizon ``T_HORIZON_WB = safety_factor * max_case(omega_b * t_end_pred)`` [-].

    Evaluates ``omega_b * t_end_pred`` from ``two_stream_window_bounds_wb`` across all
    verification cases (symmetric ``(1, 1)``, asymmetric ``(1, 0.5)``, opposite-force
    pusher mutation ``(1, -1)``, and halved-gather mutation ``(0.5, 0.5)``) using each
    case's discrete growth rate ``gamma_disc(case)``.
    """
    v0 = float(v0_over_c) * c
    gamma0 = 1.0 / float(np.sqrt(1.0 - float(v0_over_c) ** 2))
    wb_rel = float(omega_b) * (gamma0 ** (-1.5))
    k1 = (float(np.sqrt(3.0)) / 2.0) * wb_rel / v0
    lz = 2.0 * float(np.pi) / k1
    dz = lz / float(Nz)
    dt = two_stream_cfl_dt(dz, cfln=cfln, transverse_aspect=transverse_aspect)

    cases = (
        (1.0, 1.0, 1.0),
        (1.0, 0.5, 0.5),
        (1.0, -1.0, 1.0),
        (0.5, 0.5, 1.0),
    )
    max_t_end_wb = 0.0
    for sp, sm, dm in cases:
        g_disc = two_stream_discrete_growth_rate(
            omega_b,
            k1,
            v0,
            dt,
            dz,
            order=1,
            gather="galerkin",
            gamma0=gamma0,
            s_plus=sp,
            s_minus=sm,
        )
        win = two_stream_window_bounds_wb(
            omega_b,
            g_disc,
            gamma0=gamma0,
            eps=eps,
            alpha_sat=alpha_sat,
            s_plus=sp,
            s_minus=sm,
            density_ratio_minus=dm,
        )
        max_t_end_wb = max(max_t_end_wb, float(win["t_end_pred_wb"]))
    return float(safety_factor) * max_t_end_wb


T_HORIZON_WB: float = two_stream_suite_horizon_wb()
