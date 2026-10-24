# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Linearized relativistic cold-fluid electron beam (`LinearizedBeam`) and
analytic space-charge wave helpers for `SolverFIT3D`.

A magnetically confined (1D axial motion along ``axis`` in ``{'x', 'y', 'z'}``),
cold electron beam drifting at DC velocity ``u0`` (Lorentz factor ``gamma0``)
with unperturbed charge density ``rho0(r_perp) <= 0`` obeys, in the small-signal
Eulerian limit,

    (d/dt + u0 d/ds + nu_b) v1 = -(e / (gamma0^3 m_e)) E_s
    (d/dt + u0 d/ds) rho1      = -rho0 d v1/ds
    J1_s                       = rho0 v1 + u0 rho1

where ``v1`` [m/s] is the axial velocity perturbation, ``rho1`` [C/m^3] is the
space-charge density perturbation, and ``J1_s`` [A/m^2] is the AC beam current
density injected into Ampere's law on the primary ``E_s`` edges.

Discretization
--------------
The auxiliary unknowns ``(v1, rho1)`` live on the ``E_s`` edges inside the beam
mask at the half-integer time levels ``t^{n-1/2}`` and ``t^{n+1/2}`` used for
currents in the FIT leapfrog scheme:

1. At the start of step ``n`` (time ``t^n``), ``E_s^n`` is known at the exact
   midpoint of the interval ``[t^{n-1/2}, t^{n+1/2}]``.
2. ``v1`` and ``rho1`` advance from ``t^{n-1/2}`` to ``t^{n+1/2}`` via a
   trapezoidal (Crank-Nicolson) integrator in time coupled with second-order
   centred differences ``D_s`` along each 1D beam line (an exact tridiagonal or
   cyclic-tridiagonal solve per line).
3. The resulting current ``J1_s^{n+1/2} = rho0 v1^{n+1/2} + u0 rho1^{n+1/2}``
   is centred at ``t^{n+1/2}``, the exact midpoint of the electric-field update
   from ``t^n`` to ``t^{n+1}``.

For ``u0 = 0`` (``gamma0 = 1``), the update reduces identically to the
stationary Drude cold-plasma ADE (`wakis.plasma.cold_plasma`). For ``u0 > 0``
and ``nu_b = 0``, the scheme is second-order accurate in space and time,
non-dissipative for all 3D wavenumbers under the plasma-CFL condition
``courant^2 + (omega_p_rel dt / 2)^2 <= 1``, satisfies an exact discrete Chu
kinetic power theorem on the staggered grid, and has an exact Hermitian
transpose (`AdjointStepper`).
"""

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as q_e
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import m_e, mu_0
from scipy.optimize import brentq
from scipy.sparse import coo_matrix, csc_matrix, diags, eye
from scipy.sparse.linalg import splu
from scipy.special import i0, i1, ive, j0, j1, k0, k1, kve


def relativistic_gamma(u0=None, *, beta=None, voltage=None):
    """
    Lorentz factor ``gamma0`` and drift velocity ``u0`` [m/s] of an electron beam.

    Parameters
    ----------
    u0 : float, optional
        DC axial drift velocity [m/s], ``|u0| < c``.
    beta : float, optional
        Normalized drift velocity ``u0 / c`` in ``(-1, 1)``.
    voltage : float, optional
        Accelerating DC kinetic voltage ``V0 >= 0`` [V], related to ``gamma0``
        by ``gamma0 = 1 + e V0 / (m_e c^2)``.

    Returns
    -------
    u0 : float
        Drift velocity [m/s].
    gamma0 : float
        Lorentz factor ``(1 - (u0/c)^2)^(-1/2)`` [-].
    """
    given = [x is not None for x in (u0, beta, voltage)]
    if sum(given) != 1:
        raise ValueError("Specify exactly one of `u0`, `beta`, or `voltage`.")
    if voltage is not None:
        v0 = float(voltage)
        if v0 < 0:
            raise ValueError(f"voltage must be >= 0, got {v0}.")
        gamma0 = 1.0 + q_e * v0 / (m_e * c_light**2)
        beta0 = np.sqrt(1.0 - 1.0 / gamma0**2)
        return float(beta0 * c_light), float(gamma0)
    if beta is not None:
        beta0 = float(beta)
        if abs(beta0) >= 1.0:
            raise ValueError(f"|beta| must be < 1, got {beta0}.")
        u0_val = beta0 * c_light
    else:
        u0_val = float(u0)
        beta0 = u0_val / c_light
        if abs(beta0) >= 1.0:
            raise ValueError(f"|u0| must be < c ({c_light:.6e} m/s), got {u0_val}.")
    gamma0 = 1.0 / np.sqrt(1.0 - beta0**2)
    return float(u0_val), float(gamma0)


def beam_plasma_frequency(rho0=None, *, density=None, current=None, area=None, u0=None):
    """
    Non-relativistic electron plasma angular frequency ``omega_p`` [rad/s].

    Parameters
    ----------
    rho0 : float or array_like, optional
        Unperturbed DC charge density [C/m^3] (``<= 0`` for electrons).
    density : float or array_like, optional
        Electron number density ``n0`` [1/m^3] (``>= 0``).
    current : float, optional
        DC beam current magnitude ``I0 >= 0`` [A] (requires ``area`` and ``u0``).
    area : float, optional
        Beam cross-sectional area ``A_b > 0`` [m^2].
    u0 : float, optional
        DC drift velocity ``|u0| > 0`` [m/s].

    Returns
    -------
    omega_p : float or ndarray
        Plasma angular frequency ``sqrt(e |rho0| / (eps_0 m_e))`` [rad/s].
    """
    if rho0 is not None:
        rho_arr = np.asarray(rho0, dtype=float)
        if np.any(rho_arr > 0):
            raise ValueError("Electron beam charge density `rho0` must be <= 0.")
        return np.sqrt(q_e * np.abs(rho_arr) / (eps_0 * m_e))
    if density is not None:
        n_arr = np.asarray(density, dtype=float)
        if np.any(n_arr < 0):
            raise ValueError("Electron number density `density` must be >= 0.")
        return np.sqrt(n_arr * q_e**2 / (eps_0 * m_e))
    if current is not None:
        if area is None or u0 is None or area <= 0 or u0 == 0:
            raise ValueError("`current` requires positive `area` and non-zero `u0`.")
        rho_val = -abs(float(current)) / (float(area) * abs(float(u0)))
        return float(np.sqrt(q_e * abs(rho_val) / (eps_0 * m_e)))
    raise ValueError("Provide `rho0`, `density`, or (`current`, `area`, `u0`).")


def reduction_factor_planar(omega, u0, b, a=np.inf, ky=0.0, relativistic=True):
    """
    Plasma-frequency reduction factor ``R(omega)`` of a 2D planar sheet beam.

    Solves the symmetric TM transverse resonance equation (Branch & Mihran
    1955) for a sheet beam of half-thickness ``b`` (``|x| <= b``)
    between grounded parallel PEC plates at ``|x| = a``::

        T b tan(T b) = tau_x b coth(tau_x (a - b)),
        R = 1 / sqrt(1 + (T^2 + ky^2) / tau^2),

    where ``tau = (omega / u0) / gamma0`` (or ``omega / u0`` if
    ``relativistic=False``) and ``tau_x = sqrt(tau^2 + ky^2)``. When ``b == a``
    (beam filling the waveguide), ``T = pi / (2 a)`` in closed form.

    Parameters
    ----------
    omega : float or array_like
        Angular frequency [rad/s], ``> 0``.
    u0 : float
        DC beam drift velocity [m/s], ``0 < |u0| < c``.
    b : float
        Sheet-beam half-thickness [m], ``> 0``.
    a : float, optional
        Parallel-plate PEC half-spacing [m], ``>= b``. ``np.inf`` for an
        unbounded sheet beam in free space.
    ky : float, optional
        Transverse wavenumber [rad/m] along the second transverse axis (e.g.
        ``pi / W`` for the fundamental mode of a rectangular tunnel of width
        ``W``; ``0`` for an infinite-width 2D slab).
    relativistic : bool, optional
        Include the ``1 / gamma0`` slow-wave factor in ``tau``. Default True.

    Returns
    -------
    R : float or ndarray
        Dimensionless plasma-frequency reduction factor in ``(0, 1)``.
    """
    omega_arr = np.asarray(omega, dtype=float)
    scalar = omega_arr.ndim == 0
    omega_1d = np.atleast_1d(omega_arr)
    b = float(b)
    a = float(a)
    if b <= 0 or a < b:
        raise ValueError(f"Expected 0 < b <= a, got b={b}, a={a}.")
    _, gamma0 = relativistic_gamma(u0=u0)
    beta_e = np.abs(omega_1d) / abs(float(u0))
    tau = beta_e / gamma0 if relativistic else beta_e
    tau_x = np.sqrt(tau**2 + float(ky) ** 2)

    out = np.empty_like(omega_1d, dtype=float)
    for i, (t_val, tx_val) in enumerate(zip(tau, tau_x)):
        if t_val == 0:
            out[i] = 0.0
            continue
        if np.isclose(b, a):
            T = 0.5 * np.pi / b
        else:
            gap = tx_val * (a - b)
            coth_val = 1.0 / np.tanh(gap) if np.isfinite(gap) and gap < 50.0 else 1.0
            rhs = tx_val * b * coth_val
            x_root = brentq(
                lambda x: x * np.tan(x) - rhs, 0.0, 0.5 * np.pi * (1.0 - 1e-12)
            )
            T = x_root / b
        out[i] = 1.0 / np.sqrt(1.0 + (T**2 + float(ky) ** 2) / t_val**2)
    return float(out[0]) if scalar else out.reshape(omega_arr.shape)


def reduction_factor_cylindrical(
    omega, u0, r_beam, r_tunnel=np.inf, relativistic=True
):
    """
    Plasma-frequency reduction factor ``R(omega)`` of a cylindrical pencil beam.

    Solves the axisymmetric ``TM_01`` transverse resonance equation of Branch
    & Mihran (1955) and Mealy, Marosi & Capolino (2022, Eqs. 10-11) for a
    round beam of radius ``r_beam`` inside a concentric PEC drift tube of
    radius ``r_tunnel``::

        T r_b J_1(T r_b) / J_0(T r_b)
            = tau r_b [I_0(tau r_t) K_1(tau r_b) + K_0(tau r_t) I_1(tau r_b)]
                    / [I_0(tau r_t) K_0(tau r_b) - K_0(tau r_t) I_0(tau r_b)],
        R = 1 / sqrt(1 + (T / tau)^2).

    When ``r_beam == r_tunnel``, ``T = j_01 / r_beam`` with ``j_01 = 2.4048255577``.

    Parameters
    ----------
    omega : float or array_like
        Angular frequency [rad/s], ``> 0``.
    u0 : float
        DC beam drift velocity [m/s], ``0 < |u0| < c``.
    r_beam : float
        Beam radius ``R_b`` [m], ``> 0``.
    r_tunnel : float, optional
        PEC tunnel radius ``R_t`` [m], ``>= r_beam``. ``np.inf`` for an
        unbounded beam in free space.
    relativistic : bool, optional
        Include the ``1 / gamma0`` slow-wave factor in ``tau``. Default True.

    Returns
    -------
    R : float or ndarray
        Dimensionless plasma-frequency reduction factor in ``(0, 1)``.
    """
    J01 = 2.404825557695773
    omega_arr = np.asarray(omega, dtype=float)
    scalar = omega_arr.ndim == 0
    omega_1d = np.atleast_1d(omega_arr)
    rb = float(r_beam)
    rt = float(r_tunnel)
    if rb <= 0 or rt < rb:
        raise ValueError(f"Expected 0 < r_beam <= r_tunnel, got {rb}, {rt}.")
    _, gamma0 = relativistic_gamma(u0=u0)
    beta_e = np.abs(omega_1d) / abs(float(u0))
    tau = beta_e / gamma0 if relativistic else beta_e

    out = np.empty_like(omega_1d, dtype=float)
    for i, t_val in enumerate(tau):
        if t_val == 0:
            out[i] = 0.0
            continue
        if np.isclose(rb, rt):
            T = J01 / rb
        else:
            xb = t_val * rb
            if np.isinf(rt):
                rhs = xb * kve(1, xb) / kve(0, xb)
            else:
                xt = t_val * rt
                scale = np.exp(-2.0 * (xt - xb))
                num = ive(0, xt) * kve(1, xb) + scale * kve(0, xt) * ive(1, xb)
                den = ive(0, xt) * kve(0, xb) - scale * kve(0, xt) * ive(0, xb)
                rhs = xb * (num / den)
            x_root = brentq(
                lambda x: x * j1(x) / j0(x) - rhs, 0.0, J01 * (1.0 - 1e-12)
            )
            T = x_root / rb
        out[i] = 1.0 / np.sqrt(1.0 + (T / t_val) ** 2)
    return float(out[0]) if scalar else out.reshape(omega_arr.shape)


def reduction_factor_rectangular(
    omega, u0, bx, by, ax=None, ay=None, n_modes=12, relativistic=True
):
    """
    Plasma-frequency reduction factor ``R(omega)`` of a rectangular beam in a
    rectangular PEC waveguide.

    For a beam occupying ``[-bx, bx] x [-by, by]`` inside a grounded PEC guide
    ``[-ax, ax] x [-ay, ay]``, the fundamental even-even TM space-charge mode
    satisfies ``(-lap_perp + tau^2) E_s = (tau / R)^2 Theta_beam(x, y) E_s``.
    When the beam fills both walls (``bx == ax`` and ``by == ay``), the exact
    closed form ``T^2 = (pi/(2 ax))^2 + (pi/(2 ay))^2`` is used; when it fills
    one wall, `reduction_factor_planar` is used; otherwise the even Fourier-sine
    Galerkin eigenproblem in ``cos((2m-1) pi x / (2 ax)) cos((2n-1) pi y / (2 ay))``
    is solved.

    Parameters
    ----------
    omega : float or array_like
        Angular frequency [rad/s].
    u0 : float
        DC drift velocity [m/s].
    bx, by : float
        Beam half-widths along the two transverse axes [m].
    ax, ay : float, optional
        PEC tunnel half-widths [m] (default equal to ``bx, by``).
    n_modes : int, optional
        Number of even Fourier basis modes per transverse axis. Default 12.
    relativistic : bool, optional
        Include the ``1 / gamma0`` slow-wave factor in ``tau``. Default True.

    Returns
    -------
    R : float or ndarray
        Dimensionless reduction factor in ``(0, 1)``.
    """
    bx, by = float(bx), float(by)
    ax = bx if ax is None else float(ax)
    ay = by if ay is None else float(ay)
    if bx <= 0 or by <= 0 or ax < bx or ay < by:
        raise ValueError("Invalid rectangular beam/tunnel dimensions.")
    if np.isclose(bx, ax):
        return reduction_factor_planar(
            omega, u0, by, a=ay, ky=0.5 * np.pi / ax, relativistic=relativistic
        )
    if np.isclose(by, ay):
        return reduction_factor_planar(
            omega, u0, bx, a=ax, ky=0.5 * np.pi / ay, relativistic=relativistic
        )

    # 2D even-even cosine basis psi_m(x) = sqrt(1/ax) cos(kx_m x), kx_m = (2m+1)pi/(2ax)
    m = np.arange(int(n_modes))
    kx = (2 * m + 1) * np.pi / (2.0 * ax)
    ky = (2 * m + 1) * np.pi / (2.0 * ay)

    def overlap_1d(k, b, a):
        ki, kj = np.meshgrid(k, k, indexing="ij")
        diff = ki - kj
        summ = ki + kj
        t1 = np.where(diff == 0, b, np.sin(diff * b) / np.where(diff == 0, 1.0, diff))
        t2 = np.sin(summ * b) / summ
        return (t1 + t2) / a

    Ox = overlap_1d(kx, bx, ax)
    Oy = overlap_1d(ky, by, ay)
    # Generalized problem: M c = (R / tau)^2 (kx^2 + ky^2 + tau^2) c -> max eig of D^-1/2 O D^-1/2
    O2d = np.kron(Ox, Oy)
    KX2, KY2 = np.meshgrid(kx**2, ky**2, indexing="ij")
    kperp2 = (KX2 + KY2).ravel()

    omega_arr = np.asarray(omega, dtype=float)
    scalar = omega_arr.ndim == 0
    omega_1d = np.atleast_1d(omega_arr)
    _, gamma0 = relativistic_gamma(u0=u0)
    beta_e = np.abs(omega_1d) / abs(float(u0))
    tau = beta_e / gamma0 if relativistic else beta_e

    out = np.empty_like(omega_1d, dtype=float)
    for i, t_val in enumerate(tau):
        inv_sqrt = 1.0 / np.sqrt(kperp2 + t_val**2)
        A = (inv_sqrt[:, None] * O2d) * inv_sqrt[None, :]
        eig_max = float(np.max(np.linalg.eigvalsh(A)))
        out[i] = float(t_val * np.sqrt(max(eig_max, 0.0)))
    return float(out[0]) if scalar else out.reshape(omega_arr.shape)


def space_charge_wave_dispersion(
    beta, u0, omega_p, R=1.0, gamma0=None, gamma_power=3.0
):
    """
    Fast and slow space-charge wave angular frequencies ``omega_+(beta), omega_-(beta)``.

    .. math::

        \\omega_{\\pm}(\\beta) = \\beta u_0 \\pm R \\frac{\\omega_p}{\\gamma_0^{p/2}},

    where ``omega_+`` is the fast space-charge wave (phase velocity ``> u_0``,
    positive Chu kinetic power) and ``omega_-`` is the slow space-charge wave
    (phase velocity ``< u_0``, negative Chu kinetic power).

    Parameters
    ----------
    beta : float or array_like
        Axial wavenumber [rad/m].
    u0 : float
        DC drift velocity [m/s].
    omega_p : float
        Non-relativistic plasma angular frequency [rad/s].
    R : float or array_like, optional
        Plasma-frequency reduction factor in ``(0, 1]``. Default 1.0.
    gamma0 : float, optional
        Relativistic factor; computed from ``u0`` if None.
    gamma_power : float, optional
        Exponent ``p`` of ``gamma0`` in the longitudinal mass ``gamma0^p m_e``
        (physical value is ``3.0``).

    Returns
    -------
    omega_fast, omega_slow : float or ndarray
        Angular frequencies [rad/s] of the fast and slow space-charge waves.
    """
    if gamma0 is None:
        _, gamma0 = relativistic_gamma(u0=u0)
    beta_arr = np.asarray(beta, dtype=float)
    omega_q = np.asarray(R, dtype=float) * float(omega_p) / (float(gamma0) ** (0.5 * float(gamma_power)))
    omega_0 = beta_arr * float(u0)
    return omega_0 + omega_q, omega_0 - omega_q


def ballistic_bunching_analytic(
    z, f, u0, I0, Vm, R, omega_p, d_gap=0.0, gamma0=None, gamma_power=3.0
):
    """
    Analytic small-signal kinetic voltage ``V_b(z)`` and bunching current
    ``I_b(z)`` phasors downstream of an ideal velocity-modulating gap at ``z = 0``
    (Ramo 1939; Mealy, Marosi & Capolino 2022, Eqs. 1, 5 & Fig. 1b).

    For an input gap of width ``d_gap`` driven by AC voltage ``V_m cos(2 pi f t)``,
    the beam exits the gap with velocity modulation ``V_b(0^+) = M V_m`` and zero
    initial density modulation ``I_b(0^+) = 0``, where the gap coupling factor is
    ``M = sinc(theta_g / (2 pi)) = sin(theta_g / 2) / (theta_g / 2)`` with transit
    angle ``theta_g = omega d_gap / u0``. Downstream (``z >= 0``), interference
    between the equal-amplitude fast and slow space-charge waves produces a
    standing-wave envelope::

        V_b(z) = M V_m cos(beta_q z) exp(-i beta_e z),
        I_b(z) = -i (omega / omega_q) (I_0 / (2 V_0_eff)) M V_m sin(beta_q z) exp(-i beta_e z),

    where ``beta_e = omega / u0``, ``omega_q = R omega_p / gamma0^(p/2)``,
    ``beta_q = omega_q / u0``, and ``V_0_eff = gamma0^p m_e u0^2 / (2 e)``.

    Parameters
    ----------
    z : array_like
        Axial distance downstream of the modulating gap [m], ``>= 0``.
    f : float
        Modulation frequency [Hz].
    u0 : float
        DC drift velocity [m/s].
    I0 : float
        DC beam current magnitude [A], ``> 0``.
    Vm : float
        Peak AC modulating gap voltage [V].
    R : float
        Plasma-frequency reduction factor [-].
    omega_p : float
        Non-relativistic plasma angular frequency [rad/s].
    d_gap : float, optional
        Modulating gap spacing [m]. Default 0.0 (``M = 1``).
    gamma0 : float, optional
        Lorentz factor; computed from ``u0`` if None.
    gamma_power : float, optional
        Longitudinal mass exponent ``p`` (default ``3.0``).

    Returns
    -------
    dict
        ``{'Vb': V_b(z), 'Ib': I_b(z), 'beta_e': beta_e, 'beta_q': beta_q,
        'omega_q': omega_q, 'M': M, 'lambda_q': 2*pi/beta_q, 'I_peak': ...}``.
    """
    z_arr = np.asarray(z, dtype=float)
    omega = 2.0 * np.pi * float(f)
    if gamma0 is None:
        _, gamma0 = relativistic_gamma(u0=u0)
    m_par = (float(gamma0) ** float(gamma_power)) * m_e
    omega_q = float(R) * float(omega_p) / (float(gamma0) ** (0.5 * float(gamma_power)))
    beta_e = omega / float(u0)
    beta_q = omega_q / float(u0)
    theta_g = omega * float(d_gap) / float(u0)
    M = float(np.sinc(0.5 * theta_g / np.pi))
    # From J1 = -i (omega / omega_q) (e |rho0| / (m_par u0)) V_1 sin(beta_q z)
    # and I0 = |rho0| A_b u0: I_peak = (omega / omega_q) * (e I0 / (m_par u0^2)) * M * Vm
    v0_eff = 0.5 * m_par * float(u0) ** 2 / q_e
    g = 0.5 * float(I0) * beta_e / v0_eff
    i_peak = (g * M * float(Vm)) / beta_q
    phase = np.exp(-1j * beta_e * z_arr)
    Vb = M * float(Vm) * np.cos(beta_q * z_arr) * phase
    Ib = -1j * i_peak * np.sin(beta_q * z_arr) * phase
    return {
        "Vb": Vb,
        "Ib": Ib,
        "beta_e": float(beta_e),
        "beta_q": float(beta_q),
        "omega_q": float(omega_q),
        "M": float(M),
        "g": float(g),
        "v0_eff": float(v0_eff),
        "I_peak": float(i_peak),
        "lambda_q": float(2.0 * np.pi / beta_q) if beta_q > 0 else np.inf,
    }


class LinearizedBeam:
    """
    1D-confined linearized relativistic cold-fluid electron beam for `SolverFIT3D`.

    Integrates the small-signal Eulerian velocity perturbation ``v1`` [m/s] and
    space-charge density perturbation ``rho1`` [C/m^3] on the primary ``E_s``
    edges inside the beam cross-section at the half-integer time levels
    ``t^{n+1/2}``, and injects the AC beam current density
    ``J1_s = rho0 v1 + u0 rho1`` [A/m^2] into Ampere's law.

    Parameters
    ----------
    u0 : float, optional
        DC drift velocity [m/s] along ``axis``, ``|u0| < c``. Alternatively pass
        ``beta`` or ``voltage``.
    rho0 : float, ndarray or callable, optional
        Unperturbed DC charge density [C/m^3] (``<= 0`` for an electron beam).
        Alternatively pass ``omega_p`` or ``current``.
    axis : {'x', 'y', 'z'}, optional
        Beam propagation axis ``s``. Default is ``'z'``. Works in 3D and in 2D
        (e.g. ``axis='x'`` when ``Nz == 1``).
    beta : float, optional
        Normalized drift velocity ``u0 / c`` in ``(-1, 1)``.
    voltage : float, optional
        DC kinetic beam voltage ``V0 >= 0`` [V].
    omega_p : float, optional
        Non-relativistic plasma angular frequency [rad/s], setting
        ``rho0 = -eps_0 m_e omega_p^2 / e``.
    current : float, optional
        Total DC beam current magnitude ``I0 >= 0`` [A], setting
        ``rho0 = -I0 / (A_b |u0|)`` over the beam cross-sectional area ``A_b``.
    center : tuple of 3 float, optional
        Transverse/axial centre ``(x0, y0, z0)`` [m] of the beam. Default
        ``(0.0, 0.0, 0.0)``.
    radius : float, optional
        Beam radius ``R_b`` [m] for a circular beam in 3D (or sheet half-width
        in 2D if only one transverse axis is resolved).
    half_thickness : float or tuple of 2 float, optional
        Half-thickness ``b`` [m] of a planar sheet beam (along the first
        non-periodic transverse axis, or ``(b_a, b_b)`` along both transverse
        axes for a rectangular beam).
    size : tuple of 3 float, optional
        Full box size ``(sx, sy, sz)`` [m] selecting a rectangular beam mask.
    mask : ndarray or callable, optional
        Explicit fill-fraction mask in ``[0, 1]`` of shape ``(Nx, Ny, Nz)`` or
        transverse shape ``(Na, Nb)``, or a callable ``mask(x, y, z)``.
    s_min, s_max : float, optional
        Axial entrance and exit coordinates [m] along ``axis``. By default the
        beam spans the entire non-PML physical domain along ``axis``.
    nu_b : float, optional
        Collisional momentum damping rate [1/s], ``>= 0``. Default ``0.0``.
    ramp_cells : int, optional
        Number of cells over which ``rho0(s)`` is smoothly ramped to zero
        (cosine-squared taper) before any outflow PML/CPML boundary so no beam
        current enters the PML. Default is ``8`` when the outflow face is a
        PML/CPML and ``0`` otherwise.
    ramp_length : float, optional
        Physical ramp length [m] before an outflow boundary (overrides
        ``ramp_cells`` if given).
    gamma0 : float, optional
        Explicit Lorentz factor ``gamma0``. Default is ``(1 - (u0/c)^2)^(-1/2)``.
    gamma_power : float, optional
        Exponent ``p`` of ``gamma0`` in the longitudinal relativistic mass
        ``gamma0^p m_e``. Default is ``3.0``.
    coupling_sign : float, optional
        Sign multiplier of the Lorentz force ``-(e / (gamma0^p m_e)) E_s``.
        Default is ``1.0`` (``-1.0`` flips the coupling sign for mutation tests).
    solver : SolverFIT3D, optional
        If provided, immediately attaches the beam to ``solver``.
    """

    def __init__(
        self,
        u0=None,
        rho0=None,
        axis="z",
        *,
        beta=None,
        voltage=None,
        omega_p=None,
        current=None,
        center=(0.0, 0.0, 0.0),
        radius=None,
        half_thickness=None,
        size=None,
        mask=None,
        s_min=None,
        s_max=None,
        nu_b=0.0,
        ramp_cells=None,
        ramp_length=None,
        gamma0=None,
        gamma_power=3.0,
        coupling_sign=1.0,
        gap_modulation=None,
        solver=None,
    ):
        axis = str(axis)[-1].lower()
        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}.")
        self.axis = axis
        self.component = axis
        self.u0, calc_gamma0 = relativistic_gamma(u0=u0, beta=beta, voltage=voltage)
        self.beta0 = self.u0 / c_light
        self.gamma0 = float(calc_gamma0 if gamma0 is None else gamma0)
        if self.gamma0 <= 0:
            raise ValueError(f"gamma0 must be > 0, got {self.gamma0}.")
        self.gamma_power = float(gamma_power)
        self.coupling_sign = float(coupling_sign)
        self.gap_modulation = gap_modulation
        self._step_count = 0
        self.nu_b = float(nu_b)
        if self.nu_b < 0:
            raise ValueError(f"nu_b must be >= 0, got {self.nu_b}.")

        charge_specs = [x is not None for x in (rho0, omega_p, current)]
        if sum(charge_specs) != 1:
            raise ValueError(
                "Specify exactly one of `rho0`, `omega_p`, or `current`."
            )
        if rho0 is not None and not callable(rho0):
            rho_arr = np.asarray(rho0, dtype=float)
            if np.any(rho_arr > 0):
                raise ValueError(
                    "Electron beam DC charge density `rho0` must be <= 0."
                )
        if omega_p is not None and float(omega_p) < 0:
            raise ValueError(f"omega_p must be >= 0, got {omega_p}.")
        if current is not None:
            if float(current) < 0:
                raise ValueError(f"current must be >= 0, got {current}.")
            if self.u0 == 0.0:
                raise ValueError("`current` specification requires u0 != 0.")

        self._rho0_spec = rho0
        self._omega_p_spec = None if omega_p is None else float(omega_p)
        self._current_spec = None if current is None else float(current)

        self.center = np.asarray(center, dtype=float)
        self.radius = None if radius is None else float(radius)
        self.half_thickness = half_thickness
        self.size = None if size is None else np.asarray(size, dtype=float)
        self.mask_spec = mask
        self.s_min = None if s_min is None else float(s_min)
        self.s_max = None if s_max is None else float(s_max)
        self.ramp_cells = None if ramp_cells is None else int(ramp_cells)
        self.ramp_length = None if ramp_length is None else float(ramp_length)

        self.end_time = 0.0  # passive response medium for emsolve decay checks
        self.is_initialized = False
        self._solver = None
        self.v1 = None
        self.rho1 = None
        self.v1_prev = None
        self.rho1_prev = None
        self.J1 = None
        self.J_old = None

        if solver is not None:
            self.attach(solver)

    @property
    def m_eff(self):
        """Longitudinal relativistic electron mass ``gamma0^p m_e`` [kg]."""
        return (self.gamma0**self.gamma_power) * m_e

    @property
    def beam_voltage(self):
        """Relativistic DC kinetic voltage ``(gamma0 - 1) m_e c^2 / e`` [V]."""
        return (self.gamma0 - 1.0) * m_e * c_light**2 / q_e

    @property
    def omega_p(self):
        """Peak non-relativistic plasma angular frequency ``omega_p`` [rad/s]."""
        if self._omega_p_spec is not None:
            return self._omega_p_spec
        if self.is_initialized:
            return float(np.sqrt(q_e * abs(self.rho0_peak) / (eps_0 * m_e)))
        if self._rho0_spec is not None and np.ndim(self._rho0_spec) == 0:
            return float(np.sqrt(q_e * abs(float(self._rho0_spec)) / (eps_0 * m_e)))
        if self._current_spec is not None and self.radius is not None:
            area = np.pi * self.radius**2
            return beam_plasma_frequency(
                current=self._current_spec, area=area, u0=self.u0
            )
        raise RuntimeError("Attach the beam to a solver first to evaluate omega_p.")

    @property
    def omega_p_rel(self):
        """Relativistic longitudinal plasma frequency ``omega_p / gamma0^(p/2)`` [rad/s]."""
        return self.omega_p / (self.gamma0 ** (0.5 * self.gamma_power))

    def is_stable(self, dt, ds=None, courant=0.5):
        """
        Check the von Neumann stability of the coupled FIT + Crank-Nicolson beam step.

        For a lossless beam (``nu_b >= 0``), the coupled 3D Yee + Crank-Nicolson
        update has all amplification eigenvalues on (or inside) the unit circle
        when the plasma-CFL condition ``courant^2 + (omega_p_rel dt / 2)^2 <= 1``
        holds, and the single-mode amplification matrix has spectral radius ``<= 1``.

        Parameters
        ----------
        dt : float
            Timestep [s].
        ds : float, optional
            Grid spacing along the beam axis [m] (default: ``c dt / courant``).
        courant : float, optional
            Vacuum electromagnetic Courant number ``c dt sqrt(sum 1/dx_i^2)``.

        Returns
        -------
        bool
            True if the timestep is stable.
        """
        dt = float(dt)
        wp_rel = self.omega_p_rel
        if float(courant) ** 2 + (0.5 * wp_rel * dt) ** 2 > 1.0 + 1e-9:
            return False
        if ds is None:
            ds = c_light * dt / max(float(courant), 1e-6)
        ds = float(ds)
        q_max = 2.0 * float(courant)
        for ks_ds in np.linspace(-0.95 * np.pi, 0.95 * np.pi, 19):
            k_tilde = np.sin(ks_ds) / ds
            q_s = 2.0 * c_light * dt * np.sin(0.5 * ks_ds) / ds
            q_perp_max = np.sqrt(max(q_max**2 - q_s**2, 0.0))
            for q_perp in (0.0, 0.5 * q_perp_max, q_perp_max):
                G = _beam_amplification_matrix(
                    dt, q_perp, k_tilde, self.u0, wp_rel, self.nu_b, q_s=q_s
                )
                if np.max(np.abs(np.linalg.eigvals(G))) > 1.0 + 1e-6:
                    return False
        return True

    def attach(self, solver):
        """
        Attach this beam to ``solver`` and initialize its edge operators.

        Wraps ``solver.one_step`` so that every call to ``solver.one_step()``
        or ``solver.emsolve(...)`` advances ``(v1, rho1)`` and injects ``J1_s``
        into ``solver.J`` before the electric-field update.

        Parameters
        ----------
        solver : SolverFIT3D
            Target electromagnetic solver.

        Returns
        -------
        LinearizedBeam
            ``self``.
        """
        if self._solver is solver and self.is_initialized:
            return self
        self._initialize(solver)
        if not hasattr(solver, "_beams"):
            solver._beams = []
            solver._one_step_without_beams = solver.one_step

            def _step_with_beams():
                for b in solver._beams:
                    b._pre_step(solver)
                solver._one_step_without_beams()

            solver.one_step = _step_with_beams
        if self not in solver._beams:
            solver._beams.append(self)
        return self

    install = attach

    def update(self, solver, t):
        """
        Hook for ``SolverFIT3D.emsolve(..., source=[beam, ...])``.

        Ensures the beam is attached to ``solver``. Because ``attach`` wraps
        ``solver.one_step``, the actual state update happens once per step
        inside ``solver.one_step()``.
        """
        if not self.is_initialized or self._solver is not solver:
            self.attach(solver)

    def _initialize(self, solver):
        """Select active ``E_s`` edges, build ``D_s`` and factorize ``M_+``."""
        self._solver = solver
        grid = solver.grid
        c_ax = "xyz".index(self.axis)
        dims = (solver.Nx, solver.Ny, solver.Nz)
        nodes = (grid.x, grid.y, grid.z)
        Ns = dims[c_ax]

        bc_lo = str(solver.bc_low[c_ax]).lower()
        bc_hi = str(solver.bc_high[c_ax]).lower()
        periodic_axis = bc_lo in ("periodic", "bloch") or bc_hi in ("periodic", "bloch")

        # Primary edge midpoints along axis `c_ax`
        s_edges = 0.5 * (nodes[c_ax][:Ns] + nodes[c_ax][1 : Ns + 1])
        ds_edges = np.diff(nodes[c_ax])[:Ns]

        if periodic_axis and self.s_min is None and self.s_max is None:
            k_start, k_stop = 0, Ns
            self.periodic_s = True
        else:
            self.periodic_s = False
            k_min = 0
            # Non-periodic axes have a domain-edge ghost E_s sample at Ns - 1
            k_max = Ns if periodic_axis else Ns - 1
            n_pml = int(getattr(solver, "n_pml", 0))
            if bc_lo in ("cpml", "pml"):
                k_min = max(k_min, n_pml)
            if bc_hi in ("cpml", "pml"):
                k_max = min(k_max, (Ns - 1) - n_pml)
            if self.s_min is not None:
                k_min = max(k_min, int(np.searchsorted(s_edges, self.s_min, side="left")))
            if self.s_max is not None:
                k_max = min(k_max, int(np.searchsorted(s_edges, self.s_max, side="right")))
            if k_max <= k_min:
                raise ValueError(
                    f"No active E_{self.axis} edges in axial range "
                    f"[{self.s_min}, {self.s_max}] outside PML."
                )
            k_start, k_stop = k_min, k_max

        self.k_indices = np.arange(k_start, k_stop, dtype=int)
        self.s_coords = s_edges[self.k_indices]
        self.ds_line = ds_edges[self.k_indices]
        M = self.k_indices.size
        self.M = M

        # Transverse fill fractions w_perp in [0, 1] on the (a, b) plane
        w_perp_3d = self._build_transverse_weights(solver, c_ax)
        # Slice to active axial indices and move `c_ax` to axis 0 -> (M, Na, Nb)
        sl = [slice(None), slice(None), slice(None)]
        sl[c_ax] = self.k_indices
        w_sub = np.moveaxis(w_perp_3d[tuple(sl)], c_ax, 0)

        # Dual areas dA_perp and cell volumes dV on the E_s grid
        tA_all = np.where(grid.itA.toarray() != 0, 1.0 / grid.itA.toarray(), 0.0)
        dV_all = grid.L.toarray() * tA_all
        ieps_host = np.asarray(
            solver.ieps.array.get() if solver.use_gpu else solver.ieps.array
        )

        I, J, K = np.meshgrid(
            np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing="ij"
        )
        N = solver.N
        flat_3d = c_ax * N + I + dims[0] * J + dims[0] * dims[1] * K
        flat_sub = np.moveaxis(flat_3d[tuple(sl)], c_ax, 0).reshape(M, -1)
        w_sub_2d = w_sub.reshape(M, -1)
        ieps_sub = ieps_host[flat_sub]

        # Keep transverse lines that have at least one active non-PEC edge
        active_edge = (w_sub_2d > 0) & (ieps_sub != 0)
        keep_lines = np.any(active_edge, axis=0)
        if not np.any(keep_lines):
            raise ValueError("Beam mask contains no non-PEC E edges.")

        self.keep_lines = keep_lines
        self.flat_idx_2d = flat_sub[:, keep_lines]
        self.flat_idx_host = self.flat_idx_2d.ravel()
        self.N_lines = int(np.sum(keep_lines))
        w_perp_lines = w_sub_2d[:, keep_lines] * (ieps_sub[:, keep_lines] != 0)
        self.dA_perp = np.moveaxis(
            tA_all[c_ax * N : (c_ax + 1) * N].reshape(dims, order="F")[tuple(sl)],
            c_ax,
            0,
        ).reshape(M, -1)[:, keep_lines]
        self.dV = dV_all[self.flat_idx_2d]

        # Axial taper w_s(s) -> 0 before any outflow PML so no beam current enters PML
        w_s = np.ones(M, dtype=float)
        if not self.periodic_s:
            n_ramp = self.ramp_cells
            if self.ramp_length is not None and self.ramp_length > 0:
                n_ramp = int(np.round(self.ramp_length / np.mean(self.ds_line)))
            elif n_ramp is None:
                outflow_pml = (self.u0 >= 0 and bc_hi in ("cpml", "pml")) or (
                    self.u0 < 0 and bc_lo in ("cpml", "pml")
                )
                n_ramp = min(8, max(2, M // 6)) if (outflow_pml and M >= 4) else 0
            n_ramp = min(max(int(n_ramp), 0), max(M - 1, 0))
            self.n_ramp = n_ramp
            if n_ramp > 0:
                ramp_profile = np.sin(
                    0.5 * np.pi * np.arange(n_ramp, dtype=float) / float(n_ramp)
                )
                if self.u0 >= 0:
                    w_s[M - n_ramp :] = ramp_profile[::-1]
                else:
                    w_s[:n_ramp] = ramp_profile
        else:
            self.n_ramp = 0
        self.w_axial = w_s

        # Determine unperturbed DC charge density rho0 <= 0 [C/m^3]
        self.area = float(np.mean(np.sum(w_perp_lines * self.dA_perp, axis=1)))
        if self._omega_p_spec is not None:
            self.rho0_peak = -eps_0 * m_e * self._omega_p_spec**2 / q_e
            rho0_2d = self.rho0_peak * w_perp_lines
        elif self._current_spec is not None:
            # Analytic area if circular/sheet inside domain, else discrete area
            area_eff = self._analytic_area(solver, c_ax) or self.area
            self.rho0_peak = -self._current_spec / (area_eff * abs(self.u0))
            # Scale w_perp so discrete integral of rho0 u0 dA matches -I0
            scale = area_eff / self.area if self.area > 0 else 1.0
            rho0_2d = self.rho0_peak * (w_perp_lines * scale)
        elif callable(self._rho0_spec):
            X, Y, Z = np.meshgrid(
                0.5 * (grid.x[: dims[0]] + grid.x[1 : dims[0] + 1])
                if c_ax == 0
                else grid.x[: dims[0]],
                0.5 * (grid.y[: dims[1]] + grid.y[1 : dims[1] + 1])
                if c_ax == 1
                else grid.y[: dims[1]],
                0.5 * (grid.z[: dims[2]] + grid.z[1 : dims[2] + 1])
                if c_ax == 2
                else grid.z[: dims[2]],
                indexing="ij",
            )
            rho_eval = np.asarray(self._rho0_spec(X, Y, Z), dtype=float)
            if np.any(rho_eval > 0):
                raise ValueError("Callable `rho0` returned positive charge density.")
            rho_sub = np.moveaxis(rho_eval[tuple(sl)], c_ax, 0).reshape(M, -1)[
                :, keep_lines
            ]
            rho0_2d = rho_sub * w_perp_lines
            self.rho0_peak = float(np.min(rho0_2d))
        else:
            rho_arr = np.asarray(self._rho0_spec, dtype=float)
            if rho_arr.ndim == 0:
                self.rho0_peak = float(rho_arr)
                rho0_2d = self.rho0_peak * w_perp_lines
            else:
                rho_3d = np.broadcast_to(rho_arr, dims)
                rho_sub = np.moveaxis(rho_3d[tuple(sl)], c_ax, 0).reshape(M, -1)[
                    :, keep_lines
                ]
                rho0_2d = rho_sub * w_perp_lines
                self.rho0_peak = float(np.min(rho0_2d))

        self.rho0 = rho0_2d
        self.n0 = -self.rho0 / q_e

        # Check stability against solver timestep
        courant = (
            c_light
            * solver.dt
            * np.sqrt(
                np.sum([1.0 / np.min(d) ** 2 for d in (solver.dx, solver.dy, solver.dz)])
            )
        )
        if not self.is_stable(solver.dt, ds=float(np.min(self.ds_line)), courant=courant):
            raise ValueError(
                f"dt={solver.dt:.3e} s is unstable for LinearizedBeam "
                f"(omega_p_rel={self.omega_p_rel:.3e} rad/s); reduce `dt` or `cfln`."
            )

        # Build 1D axial derivative operator D_s of shape (M, M)
        op_dtype = (
            np.complex128
            if np.iscomplexobj(solver.E.array)
            else np.float64
        )
        Ds = self._build_axial_derivative(solver, c_ax, op_dtype)
        Im = eye(M, format="csc", dtype=op_dtype)
        dt = float(solver.dt)

        Mp_v = (Im + 0.5 * dt * (self.u0 * Ds + self.nu_b * Im)).tocsc()
        Mm_v = (Im - 0.5 * dt * (self.u0 * Ds + self.nu_b * Im)).tocsc()
        Mp_r = (Im + 0.5 * dt * (self.u0 * Ds)).tocsc()
        Mm_r = (Im - 0.5 * dt * (self.u0 * Ds)).tocsc()

        self.Ds = Ds
        self.Ds_H = Ds.conj().T.tocsc()
        self.Mp_v, self.Mm_v = Mp_v, Mm_v
        self.Mp_r, self.Mm_r = Mp_r, Mm_r
        self.Mm_v_H = Mm_v.conj().T.tocsc()
        self.Mm_r_H = Mm_r.conj().T.tocsc()
        self._lu_v = splu(Mp_v)
        self._lu_r = splu(Mp_r)
        self._lu_v_H = splu(Mp_v.conj().T.tocsc())
        self._lu_r_H = splu(Mp_r.conj().T.tocsc())

        # Coupling arrays of shape (M, N_lines)
        active_mask = (ieps_sub[:, keep_lines] != 0).astype(float)
        w_ax_2d = self.w_axial[:, None] * active_mask
        self._alpha_E = (
            -self.coupling_sign * dt * (q_e / self.m_eff) * w_ax_2d
        ).astype(op_dtype)
        self._beta_v = (-dt * self.rho0).astype(op_dtype)
        self._c_v = (w_ax_2d * self.rho0).astype(op_dtype)
        self._c_rho = (w_ax_2d * self.u0).astype(op_dtype)

        xp = solver.E.xp
        self.xp = xp
        self.dtype = solver.E.dtype
        self.flat_idx = xp.asarray(self.flat_idx_host)
        self.v1 = np.zeros((M, self.N_lines), dtype=self.dtype)
        self.rho1 = np.zeros((M, self.N_lines), dtype=self.dtype)
        self.v1_prev = np.zeros((M, self.N_lines), dtype=self.dtype)
        self.rho1_prev = np.zeros((M, self.N_lines), dtype=self.dtype)
        self.J1 = np.zeros((M, self.N_lines), dtype=self.dtype)
        self.J_old = xp.zeros(self.flat_idx_host.size, dtype=self.dtype)
        self.is_initialized = True

    def _analytic_area(self, solver, c_ax):
        """Exact geometric cross-sectional area when radius/half_thickness is given."""
        dims = (solver.Nx, solver.Ny, solver.Nz)
        nodes = (solver.grid.x, solver.grid.y, solver.grid.z)
        t_axes = [d for d in range(3) if d != c_ax]
        a_ax, b_ax = t_axes
        La = float(nodes[a_ax][-1] - nodes[a_ax][0])
        Lb = float(nodes[b_ax][-1] - nodes[b_ax][0])
        is_2d_a = dims[a_ax] == 1
        is_2d_b = dims[b_ax] == 1
        if self.radius is not None:
            if is_2d_a:
                return min(2.0 * self.radius, Lb) * La
            if is_2d_b:
                return min(2.0 * self.radius, La) * Lb
            return float(np.pi * self.radius**2)
        if self.half_thickness is not None:
            if np.ndim(self.half_thickness) == 0:
                b = float(self.half_thickness)
                if is_2d_b or str(solver.bc_low[b_ax]).lower() in ("periodic", "bloch"):
                    return min(2.0 * b, La) * Lb
                return min(2.0 * b, Lb) * La
            ba, bb = map(float, self.half_thickness)
            return min(2.0 * ba, La) * min(2.0 * bb, Lb)
        return None

    def _build_transverse_weights(self, solver, c_ax):
        """
        Compute smooth dual-cell overlap weights ``w_perp`` in ``[0, 1]`` on
        the 3D ``E_s`` grid of shape ``(Nx, Ny, Nz)``.
        """
        grid = solver.grid
        dims = (solver.Nx, solver.Ny, solver.Nz)
        nodes = (grid.x, grid.y, grid.z)
        if self.mask_spec is not None:
            if callable(self.mask_spec):
                coords = [
                    0.5 * (nodes[d][: dims[d]] + nodes[d][1 : dims[d] + 1])
                    if d == c_ax
                    else nodes[d][: dims[d]]
                    for d in range(3)
                ]
                X, Y, Z = np.meshgrid(*coords, indexing="ij")
                m = np.asarray(self.mask_spec(X, Y, Z), dtype=float)
                return np.clip(np.broadcast_to(m, dims), 0.0, 1.0)
            m = np.asarray(self.mask_spec, dtype=float)
            if m.ndim == 2:
                t_axes = [d for d in range(3) if d != c_ax]
                pad_2d = []
                for ax_i, d in enumerate(t_axes):
                    if m.shape[ax_i] == dims[d] - 1:
                        pad_2d.append((0, 1))
                    else:
                        pad_2d.append((0, 0))
                if any(p[1] > 0 for p in pad_2d):
                    m = np.pad(m, pad_2d, mode="constant")
                shape_3d = [dims[0], dims[1], dims[2]]
                shape_3d[c_ax] = 1
                m = m.reshape(shape_3d)
            elif m.ndim == 3:
                pad_3d = [
                    (0, 1) if m.shape[d] == dims[d] - 1 else (0, 0)
                    for d in range(3)
                ]
                if any(p[1] > 0 for p in pad_3d):
                    m = np.pad(m, pad_3d, mode="constant")
            return np.clip(np.broadcast_to(m, dims), 0.0, 1.0)

        # Dual-cell intervals [lo, hi] around each transverse node
        t_axes = [d for d in range(3) if d != c_ax]
        dual_intervals = {}
        for d in t_axes:
            n = dims[d]
            nd = nodes[d]
            if n == 1:
                dual_intervals[d] = (np.array([nd[0]]), np.array([nd[1]]))
            else:
                mid = 0.5 * (nd[:-1] + nd[1:])
                lo = np.concatenate([[nd[0]], mid[: n - 1]])
                hi = np.concatenate([mid[: n - 1], [nd[n]]])
                dual_intervals[d] = (lo, hi)

        def interval_overlap(lo, hi, c0, half_w):
            if np.isinf(half_w):
                return np.ones_like(lo, dtype=float)
            overlap = np.clip(
                np.minimum(hi, c0 + half_w) - np.maximum(lo, c0 - half_w), 0.0, None
            )
            width = np.maximum(hi - lo, 1e-30)
            return overlap / width

        a_ax, b_ax = t_axes
        alo, ahi = dual_intervals[a_ax]
        blo, bhi = dual_intervals[b_ax]
        ca, cb = self.center[a_ax], self.center[b_ax]

        if self.radius is not None:
            R = self.radius
            # In 2D (one transverse axis has 1 cell), treat radius as sheet half-width
            if dims[b_ax] == 1 and dims[a_ax] > 1:
                wa = interval_overlap(alo, ahi, ca, R)
                wb = np.ones(1, dtype=float)
                w2d = wa[:, None] * wb[None, :]
            elif dims[a_ax] == 1 and dims[b_ax] > 1:
                wa = np.ones(1, dtype=float)
                wb = interval_overlap(blo, bhi, cb, R)
                w2d = wa[:, None] * wb[None, :]
            else:
                # 8x8 subpixel quadrature on each dual cell in (a, b)
                q = (np.arange(8, dtype=float) + 0.5) / 8.0
                a_sub = alo[:, None] + (ahi - alo)[:, None] * q[None, :] - ca
                b_sub = blo[:, None] + (bhi - blo)[:, None] * q[None, :] - cb
                r2 = a_sub[:, None, :, None] ** 2 + b_sub[None, :, None, :] ** 2
                w2d = np.mean(r2 <= R**2, axis=(2, 3))
        elif self.half_thickness is not None:
            if np.ndim(self.half_thickness) == 0:
                b = float(self.half_thickness)
                # Sheet beam across the non-periodic / resolved transverse axis
                b_periodic = (
                    dims[b_ax] == 1
                    or str(solver.bc_low[b_ax]).lower() in ("periodic", "bloch")
                )
                a_periodic = (
                    dims[a_ax] == 1
                    or str(solver.bc_low[a_ax]).lower() in ("periodic", "bloch")
                )
                if b_periodic and not a_periodic:
                    wa = interval_overlap(alo, ahi, ca, b)
                    wb = np.ones_like(blo)
                elif a_periodic and not b_periodic:
                    wa = np.ones_like(alo)
                    wb = interval_overlap(blo, bhi, cb, b)
                else:
                    wa = interval_overlap(alo, ahi, ca, b)
                    wb = np.ones_like(blo)
            else:
                ba, bb = map(float, self.half_thickness)
                wa = interval_overlap(alo, ahi, ca, ba)
                wb = interval_overlap(blo, bhi, cb, bb)
            w2d = wa[:, None] * wb[None, :]
        elif self.size is not None:
            ha = 0.5 * self.size[a_ax] if self.size[a_ax] > 0 else np.inf
            hb = 0.5 * self.size[b_ax] if self.size[b_ax] > 0 else np.inf
            wa = interval_overlap(alo, ahi, ca, ha)
            wb = interval_overlap(blo, bhi, cb, hb)
            w2d = wa[:, None] * wb[None, :]
        else:
            # Default: full cross-section
            w2d = np.ones((dims[a_ax], dims[b_ax]), dtype=float)

        shape_3d = [dims[0], dims[1], dims[2]]
        shape_3d[c_ax] = 1
        return np.broadcast_to(w2d.reshape(shape_3d), dims)

    def _build_axial_derivative(self, solver, c_ax, dtype):
        """
        Build the ``M x M`` 1D spatial derivative matrix ``D_s`` along ``axis``.

        - Periodic/Bloch: second-order centred differences with Bloch phase
          wrap-around ``solver.bloch_phase[c_ax]``.
        - Open (inflow/outflow): unmodulated inflow ``v1 = rho1 = 0`` at the
          upstream ghost edge, second-order centred differences in the interior,
          and first-order upwind characteristic closure at the outflow edge.
        """
        M = self.M
        ds = self.ds_line
        if self.u0 == 0.0:
            return csc_matrix((M, M), dtype=dtype)

        if self.periodic_s:
            phase = complex(solver.bloch_phase[c_ax])
            if dtype == np.float64:
                phase = float(np.real(phase))
            if M == 1:
                val = (phase - np.conj(phase)) / (2.0 * ds[0])
                return csc_matrix(np.array([[val]], dtype=dtype))
            rows, cols, data = [], [], []
            for k in range(M):
                kp = (k + 1) % M
                km = (k - 1) % M
                ph_p = phase if k == M - 1 else 1.0
                ph_m = np.conj(phase) if k == 0 else 1.0
                rows.extend([k, k])
                cols.extend([kp, km])
                data.extend([ph_p / (2.0 * ds[k]), -ph_m / (2.0 * ds[k])])
            return coo_matrix((data, (rows, cols)), shape=(M, M), dtype=dtype).tocsc()

        if M == 1:
            return diags([1.0 / ds[0]], [0], shape=(1, 1), format="csc", dtype=dtype)

        rows, cols, data = [], [], []
        if self.u0 > 0:
            # Inflow at k = 0 (ghost k = -1 is 0): (psi_1 - 0) / (2 ds_0)
            rows.append(0)
            cols.append(1)
            data.append(1.0 / (2.0 * ds[0]))
            # Interior 1 <= k <= M - 2: centred difference
            for k in range(1, M - 1):
                rows.extend([k, k])
                cols.extend([k + 1, k - 1])
                data.extend([1.0 / (2.0 * ds[k]), -1.0 / (2.0 * ds[k])])
            # Outflow at k = M - 1: upwind (psi_{M-1} - psi_{M-2}) / ds_{M-1}
            rows.extend([M - 1, M - 1])
            cols.extend([M - 1, M - 2])
            data.extend([1.0 / ds[M - 1], -1.0 / ds[M - 1]])
        else:
            # u0 < 0: outflow at k = 0, inflow at k = M - 1
            rows.extend([0, 0])
            cols.extend([1, 0])
            data.extend([1.0 / ds[0], -1.0 / ds[0]])
            for k in range(1, M - 1):
                rows.extend([k, k])
                cols.extend([k + 1, k - 1])
                data.extend([1.0 / (2.0 * ds[k]), -1.0 / (2.0 * ds[k])])
            rows.append(M - 1)
            cols.append(M - 2)
            data.append(-1.0 / (2.0 * ds[M - 1]))
        return coo_matrix((data, (rows, cols)), shape=(M, M), dtype=dtype).tocsc()

    def _pre_step(self, solver):
        """
        Advance ``(v1, rho1)`` from ``t^{n-1/2}`` to ``t^{n+1/2}`` using
        ``E_s^n`` and inject ``J1_s^{n+1/2}`` into ``solver.J``.
        """
        xp = self.xp
        E_flat = solver.E.array[self.flat_idx]
        E_host = np.asarray(E_flat.get() if hasattr(E_flat, "get") else E_flat)
        E_b = E_host.reshape(self.M, self.N_lines)
        if self.gap_modulation is not None:
            t = self._step_count * float(solver.dt)
            e_mod = np.asarray(self.gap_modulation(self.s_coords, t))
            if e_mod.ndim == 1:
                e_mod = e_mod[:, None]
            E_b = E_b + e_mod
        self._step_count += 1

        v_old = self.v1
        rho_old = self.rho1
        self.v1_prev = v_old
        self.rho1_prev = rho_old

        rhs_v = self.Mm_v @ v_old + self._alpha_E * E_b
        v_new = self._lu_v.solve(rhs_v)
        v_bar = 0.5 * (v_new + v_old)

        rhs_r = self.Mm_r @ rho_old + self._beta_v * (self.Ds @ v_bar)
        rho_new = self._lu_r.solve(rhs_r)

        J_new = self._c_v * v_new + self._c_rho * rho_new
        self.v1 = v_new
        self.rho1 = rho_new
        self.J1 = J_new

        J_dev = xp.asarray(J_new.ravel(), dtype=self.dtype)
        solver.J.array[self.flat_idx] += J_dev - self.J_old
        self.J_old = J_dev

    def reset(self):
        """Zero the auxiliary beam velocity, charge density, and current arrays."""
        self._step_count = 0
        if not self.is_initialized:
            return
        self.v1.fill(0.0)
        self.rho1.fill(0.0)
        self.v1_prev.fill(0.0)
        self.rho1_prev.fill(0.0)
        self.J1.fill(0.0)
        self.J_old.fill(0.0)

    reset_state = reset

    def _to_3d_grid(self, arr_2d):
        """Map a ``(M, N_lines)`` beam array back to the 3D ``(Nx, Ny, Nz)`` grid."""
        s = self._solver
        c_ax = "xyz".index(self.axis)
        full = np.zeros(s.N, dtype=arr_2d.dtype)
        local_idx = self.flat_idx_host - c_ax * s.N
        full[local_idx] = np.asarray(arr_2d).ravel()
        return full.reshape((s.Nx, s.Ny, s.Nz), order="F")

    def velocity_field(self):
        """Return ``v1`` mapped onto the 3D grid of shape ``(Nx, Ny, Nz)`` [m/s]."""
        return self._to_3d_grid(self.v1)

    def charge_density_field(self):
        """Return ``rho1`` mapped onto the 3D grid of shape ``(Nx, Ny, Nz)`` [C/m^3]."""
        return self._to_3d_grid(self.rho1)

    def current_density_field(self):
        """Return ``J1_s`` mapped onto the 3D grid of shape ``(Nx, Ny, Nz)`` [A/m^2]."""
        return self._to_3d_grid(self.J1)

    # ------------------------------------------------------------------
    # Physical diagnostics & Chu kinetic power / energy balance
    # ------------------------------------------------------------------

    def kinetic_voltage(self, v1=None):
        """
        Chu small-signal kinetic voltage ``V1 = -(gamma0^p m_e u0 / e) v1`` [V].

        Parameters
        ----------
        v1 : ndarray, optional
            Velocity perturbation array (defaults to ``self.v1``).

        Returns
        -------
        ndarray
            Kinetic voltage [V] on the beam edges, shape ``(M, N_lines)``.
        """
        v = self.v1 if v1 is None else np.asarray(v1)
        return -self.coupling_sign * (self.m_eff * self.u0 / q_e) * v

    def axial_current(self, J1=None):
        """
        Cross-section-integrated AC beam current ``I1(s) = sum_perp J1_s dA_perp`` [A].

        Parameters
        ----------
        J1 : ndarray, optional
            Current density array of shape ``(M, N_lines)`` (defaults to ``self.J1``).

        Returns
        -------
        ndarray
            1D axial current profile ``I1(s_k)`` [A] of shape ``(M,)``.
        """
        J = self.J1 if J1 is None else np.asarray(J1)
        return np.sum(J * self.dA_perp, axis=1)

    def axial_kinetic_voltage(self, v1=None):
        """
        Cross-section-averaged Chu kinetic voltage ``V_b(s)`` [V] along ``s``.
        """
        V1 = self.kinetic_voltage(v1)
        w = np.abs(self.rho0) * self.dA_perp
        denom = np.maximum(np.sum(w, axis=1), 1e-30)
        return np.sum(V1 * w, axis=1) / denom

    def kinetic_energy_density(self, v1=None, rho1=None):
        """
        Chu small-signal kinetic energy density ``w_k`` [J/m^3] on the beam edges::

            w_k = 1/2 n0 m_eff |v1|^2 + Re(rho1^* V1).

        For a fast space-charge wave ``w_k > 0``; for a slow space-charge wave
        ``w_k < 0``. When ``u0 = 0``, ``V1 = 0`` and ``w_k = 1/2 n0 m_e |v1|^2 >= 0``.
        """
        v = self.v1 if v1 is None else np.asarray(v1)
        r = self.rho1 if rho1 is None else np.asarray(rho1)
        V1 = self.kinetic_voltage(v)
        return 0.5 * self.n0 * self.m_eff * np.abs(v) ** 2 + np.real(np.conj(r) * V1)

    def kinetic_energy(self, v1=None, rho1=None, k_slice=None):
        """
        Total integrated Chu kinetic energy ``W_k = sum dV w_k`` [J].

        Parameters
        ----------
        v1, rho1 : ndarray, optional
            Beam state arrays (default: current ``self.v1, self.rho1``).
        k_slice : slice, optional
            Axial index slice ``slice(k1, k2)`` over which to integrate.

        Returns
        -------
        float
            Integrated Chu kinetic energy [J].
        """
        wk = self.kinetic_energy_density(v1, rho1)
        sl = slice(None) if k_slice is None else k_slice
        return float(np.sum(self.dV[sl] * wk[sl]))

    def kinetic_power_density(self, v1=None, rho1=None):
        """
        Instantaneous Chu kinetic power flux density ``S_k = Re(V1^* J1)`` [W/m^2].
        """
        v = self.v1 if v1 is None else np.asarray(v1)
        r = self.rho1 if rho1 is None else np.asarray(rho1)
        V1 = self.kinetic_voltage(v)
        J1 = self._c_v * v + self._c_rho * r
        return np.real(np.conj(V1) * J1)

    def kinetic_power_profile(self, v1=None, rho1=None):
        """
        Cross-section-integrated Chu kinetic power ``P_k(s) = sum_perp dA_perp S_k`` [W].
        """
        return np.sum(self.dA_perp * self.kinetic_power_density(v1, rho1), axis=1)

    def kinetic_power(self, v1=None, rho1=None):
        """
        Spatially averaged Chu kinetic power ``<P_k>`` [W] along the active beam.

        Positive for the fast space-charge wave and negative for the slow
        space-charge wave.
        """
        return float(np.mean(self.kinetic_power_profile(v1, rho1)))

    def kinetic_power_spectrum(self, v1_hat, rho1_hat=None, J1_hat=None):
        """
        Time-averaged harmonic Chu kinetic power ``P_k(s, omega) = 1/2 Re sum dA V1 J1^*`` [W].

        Positive for the fast space-charge wave (``omega > beta u0``) and
        negative for the slow space-charge wave (``omega < beta u0``).
        """
        V1_hat = self.kinetic_voltage(v1_hat)
        if J1_hat is None:
            if rho1_hat is None:
                raise ValueError("Provide either `rho1_hat` or `J1_hat`.")
            J1_hat = self._c_v * v1_hat + self._c_rho * rho1_hat
        return 0.5 * np.real(np.sum(self.dA_perp * V1_hat * np.conj(J1_hat), axis=-1))

    def interface_kinetic_flux(self, k_interface, v1_bar=None, rho1_bar=None):
        """
        Exact discrete Chu kinetic power flux ``Phi_k^n(k + 1/2)`` [W] across the
        transverse plane between axial indices ``k = k_interface`` and ``k + 1``::

            Phi_k^n(k + 1/2) = sum_perp dA_perp 1/2 Re(V_bar_k^* J_bar_{k+1} + V_bar_{k+1}^* J_bar_k),

        which telescopes ``sum_{k=k1}^{k2} dV (J_bar D_s V_bar + V_bar D_s J_bar)``
        to machine precision on the centred-difference stencil.
        """
        if v1_bar is None:
            v1_bar = 0.5 * (self.v1 + self.v1_prev)
        if rho1_bar is None:
            rho1_bar = 0.5 * (self.rho1 + self.rho1_prev)
        V_bar = self.kinetic_voltage(v1_bar)
        J_bar = self._c_v * v1_bar + self._c_rho * rho1_bar
        k = int(k_interface)
        kp = (k + 1) % self.M
        s_int = 0.5 * np.real(
            np.conj(V_bar[k]) * J_bar[kp] + np.conj(V_bar[kp]) * J_bar[k]
        )
        return float(np.sum(self.dA_perp[k] * s_int))


def _beam_amplification_matrix(
    dt, q_perp, k_tilde, u0, wp_rel, nu_b=0.0, q_s=None
):
    """
    Single-mode von Neumann amplification matrix of the coupled TM Yee EM +
    Crank-Nicolson beam step for state
    ``[E_s^n, E_perp^n, h_perp^(n-1/2), v_hat^(n-1/2), r_hat^(n-1/2)]``.
    """
    if q_s is None:
        q_s = c_light * float(dt) * float(k_tilde)
    a_v = 0.5 * dt * (1j * u0 * k_tilde + nu_b)
    a_r = 0.5 * dt * (1j * u0 * k_tilde)
    gv = (1.0 - a_v) / (1.0 + a_v)
    bv = dt * wp_rel / (1.0 + a_v)
    gr = (1.0 - a_r) / (1.0 + a_r)
    br = (-1j * k_tilde * dt) / (1.0 + a_r)

    def step(x):
        Es, Ep_t, h, v, r = x
        hp = h - q_perp * Es + q_s * Ep_t
        Ep_tp = Ep_t - q_s * hp
        vp = gv * v + bv * Es
        vbar = 0.5 * (vp + v)
        rp = gr * r + br * vbar
        j_hat = wp_rel * (vp + u0 * rp)
        Esp = Es + q_perp * hp - dt * j_hat
        return np.array([Esp, Ep_tp, hp, vp, rp], dtype=np.complex128)

    return np.column_stack([step(col) for col in np.eye(5, dtype=np.complex128)])


class BeamTranspose:
    """
    Exact Hermitian transpose of one `LinearizedBeam` step for `AdjointStepper`.
    """

    def __init__(self, beam, key, dt, ieps):
        self.beam = beam
        self.v_key = f"{key}_v1"
        self.r_key = f"{key}_rho1"
        self.keys = [self.v_key, self.r_key]
        self.idx = beam.flat_idx
        self.idx_host = beam.flat_idx_host
        self.shape = (beam.M, beam.N_lines)
        ieps_host = np.asarray(ieps.get() if hasattr(ieps, "get") else ieps)
        self.dt_ieps = (float(dt) * ieps_host[self.idx_host]).reshape(self.shape)

    def get_state(self, state):
        xp = self.beam.xp
        state[self.v_key] = xp.asarray(self.beam.v1.ravel().copy(), dtype=self.beam.dtype)
        state[self.r_key] = xp.asarray(self.beam.rho1.ravel().copy(), dtype=self.beam.dtype)

    def set_state(self, state, xp):
        v_arr = state[self.v_key]
        r_arr = state[self.r_key]
        v_host = np.asarray(v_arr.get() if hasattr(v_arr, "get") else v_arr)
        r_host = np.asarray(r_arr.get() if hasattr(r_arr, "get") else r_arr)
        self.beam.v1 = v_host.reshape(self.shape).astype(self.beam.dtype, copy=True)
        self.beam.rho1 = r_host.reshape(self.shape).astype(self.beam.dtype, copy=True)
        self.beam.J_old = xp.zeros_like(self.beam.J_old)

    def transpose_pre(self, lam, lam_star, out):
        b = self.beam
        xp = b.xp
        le = lam_star[self.idx]
        le_host = np.asarray(le.get() if hasattr(le, "get") else le).reshape(self.shape)
        lv = lam[self.v_key]
        lr = lam[self.r_key]
        lv_host = np.asarray(lv.get() if hasattr(lv, "get") else lv).reshape(self.shape)
        lr_host = np.asarray(lr.get() if hasattr(lr, "get") else lr).reshape(self.shape)

        lam_J1 = -np.conj(self.dt_ieps) * le_host
        t_vp = lv_host + np.conj(b._c_v) * lam_J1
        t_rp = lr_host + np.conj(b._c_rho) * lam_J1

        xi_r = b._lu_r_H.solve(t_rp)
        out_r = b.Mm_r_H @ xi_r
        lam_vbar = b.Ds_H @ (np.conj(b._beta_v) * xi_r)

        h_vp = t_vp + 0.5 * lam_vbar
        xi_v = b._lu_v_H.solve(h_vp)
        out_v = b.Mm_v_H @ xi_v + 0.5 * lam_vbar
        delta_E = np.conj(b._alpha_E) * xi_v

        out[self.v_key] = xp.asarray(out_v.ravel(), dtype=b.dtype)
        out[self.r_key] = xp.asarray(out_r.ravel(), dtype=b.dtype)
        out["E"][self.idx] += xp.asarray(delta_E.ravel(), dtype=b.dtype)

