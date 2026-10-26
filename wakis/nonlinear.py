# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Kerr (chi^(3)) nonlinear media, delayed Raman response, and Lugiato-Lefever
microcomb solvers for the FIT time-domain solver.

Constitutive Relation and Wakis Phasor Convention
-------------------------------------------------
In a centrosymmetric optical medium with instantaneous electronic Kerr
nonlinearity and optional delayed nuclear (Raman) vibrational response, the
electric displacement field ``D(r, t)`` [C/m^2] is related to the real electric
field ``E(r, t)`` [V/m] by (Taflove & Hagness 2005 [1], Ch. 9; Joseph & Taflove
1997 [2]; Goorjian & Taflove 1992 [3]; Blow & Wood 1989 [4])::

    D(r, t) = eps_0 eps_inf E(r, t) + P_lin(r, t) + P_NL(r, t),
    P_NL(r, t) = eps_0 chi^(3) [ (1 - f_R) |E(r, t)|^2 + f_R S(r, t) ] E(r, t),

where:

* ``eps_inf`` [-] is the high-frequency relative permittivity (background
  refractive index ``n_0 = sqrt(eps_inf)`` in the absence of linear poles),
* ``chi^(3)`` [m^2/V^2] is the isotropic third-order susceptibility,
* ``f_R`` in ``[0, 1]`` is the fractional delayed Raman contribution (``f_R = 0``
  for purely instantaneous Kerr response), and
* ``S(r, t) = int_0^infty h_R(tau) |E(r, t - tau)|^2 d tau`` [V^2/m^2] is the
  causal Raman nuclear response driven by the Lorentz auxiliary differential
  equation (ADE)::

      d^2 S / dt^2 + 2 Gamma_R dS/dt + Omega_R^2 S = Omega_R^2 |E(r, t)|^2,

  with ``Gamma_R = 1 / tau_2`` [rad/s], ``Omega_R = sqrt(1/tau_1^2 + 1/tau_2^2)``
  [rad/s], and impulse response
  ``h_R(t) = ((tau_1^2 + tau_2^2) / (tau_1 tau_2^2)) exp(-t/tau_2) sin(t/tau_1) Theta(t)``
  normalized to ``int_0^infty h_R(t) dt = 1``.

Relation between ``chi^(3)`` and ``n_2`` (the 3/4 factor for real fields)
-------------------------------------------------------------------------
In Wakis's ``exp(+i omega t)`` engineering phasor convention, a monochromatic
linearly polarized optical wave ``E(z, t) = Re[E_tilde(z) exp(+i omega t)] =
|E_tilde| cos(omega t + phi)`` satisfies::

    E^3(z, t) = |E_tilde|^3 [ (3/4) cos(omega t + phi) + (1/4) cos(3 omega t + 3 phi) ]
              = Re[ (3/4) |E_tilde|^2 E_tilde exp(+i omega t)
                  + (1/4) E_tilde^3 exp(+3 i omega t) ].

The fundamental-frequency nonlinear polarization phasor is therefore
``P_tilde_NL(omega) = (3/4) eps_0 chi^(3) |E_tilde|^2 E_tilde``, producing an
effective permittivity ``eps_eff(omega) = n_0^2 + (3/4) chi^(3) |E_tilde|^2``
and refractive-index shift ``Delta n_NL = (3 chi^(3)) / (8 n_0) |E_tilde|^2``.
Equating ``Delta n_NL = n_2 I`` with the cycle-averaged intensity
``I = (1/2) c eps_0 n_0 |E_tilde|^2`` [W/m^2] gives (Boyd 2020 [5]; Agrawal
2019 [6])::

    n_2 = (3 chi^(3)) / (4 c eps_0 n_0^2)   [m^2/W],
    chi^(3) = (4/3) c eps_0 n_0^2 n_2       [m^2/V^2].

Time-Level Centering and Per-Edge Discretization
------------------------------------------------
On the staggered FIT leapfrog grid with timestep ``dt``:

* Dual magnetic field ``H^(n+1/2)`` lives at half-integer times ``t^(n+1/2)``.
* Primary electric field ``E^n``, linear polarizations ``P_lin^n``, Raman state
  ``S^n``, and nonlinear polarization ``P_NL^n`` are collocated at integer times
  ``t^n = n dt``.
* Within step ``n -> n+1``:
  1. ``S^(n+1)`` and any linear Lorentz poles ``p_lin^(n+1)`` are advanced from
     ``(t^n, t^(n-1))`` via explicit second-order central differences centred
     at ``t^n``.
  2. Ampère's law integrated over ``[t^n, t^(n+1)]`` yields the local algebraic
     equation on each nonlinear edge ``e`` with fill fraction ``w_e``::

         a_e E_e^(n+1) + b_e |E^(n+1)|_e^2 E_e^(n+1) = r_e^(n+1),

     where ``a_e = eps_lin,e + w_e chi^(3) f_R S_e^(n+1)`` and
     ``b_e = w_e chi^(3) (1 - f_R)``.
  3. Three time-stepping update schemes are supported (`method` parameter):
     - ``'newton'`` (default): vectorized per-edge (or per-cell isotropic)
       Newton-Raphson cubic solve converging quadratically to machine round-off
       in 3-5 iterations (Taflove & Hagness 2005 [1], Sec. 9.5).
     - ``'fixed_point'``: Picard fixed-point iteration
       ``E^(m+1) = r / (a + b |E^(m)|^2)`` (Joseph & Taflove 1997 [2]).
     - ``'explicit_ade'``: explicit predictor-corrector ``P_NL`` auxiliary
       differential equation update.

References
----------
[1] A. Taflove and S. C. Hagness, *Computational Electrodynamics: The
    Finite-Difference Time-Domain Method*, 3rd ed., Artech House, Ch. 9 (2005).
[2] R. M. Joseph and A. Taflove, "FDTD Maxwell's equations models for nonlinear
    electrodynamics and optics," *IEEE Trans. Antennas Propag.* 45(3), 364-374
    (1997), doi:10.1109/8.558652.
[3] P. M. Goorjian and A. Taflove, "Direct time integration of Maxwell's
    equations in nonlinear dispersive media for propagation and scattering of
    femtosecond electromagnetic solitons," *Opt. Lett.* 17(3), 180-182 (1992),
    doi:10.1364/OL.17.000180.
[4] K. J. Blow and D. Wood, "Theoretical description of transient stimulated
    Raman scattering in optical fibers," *IEEE J. Quantum Electron.* 25(12),
    2665-2673 (1989), doi:10.1109/3.40655.
[5] R. W. Boyd, *Nonlinear Optics*, 4th ed., Academic Press (2020).
[6] G. P. Agrawal, *Nonlinear Fiber Optics*, 6th ed., Academic Press (2019).
[7] L. A. Lugiato and R. Lefever, "Spatial dissipative structures in passive
    optical systems," *Phys. Rev. Lett.* 58(21), 2209-2211 (1987),
    doi:10.1103/PhysRevLett.58.2209.
[8] Y. K. Chembo and C. R. Menyuk, "Spatiotemporal Lugiato-Lefever formalism for
    Kerr-comb generation in whispering-gallery-mode resonators," *Phys. Rev. A*
    87(5), 053852 (2013), doi:10.1103/PhysRevA.87.053852.
[9] T. Herr et al., "Temporal solitons in optical microresonators," *Nat.
    Photonics* 8(2), 145-152 (2014), doi:10.1038/nphoton.2013.343.
[10] T. J. Kippenberg, A. L. Gaeta, M. Lipson, and M. L. Gorodetsky,
     "Dissipative Kerr solitons in optical microresonators," *Science*
     361(6402), eaan8083 (2018), doi:10.1126/science.aan8083.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import hbar
from scipy.constants import mu_0

from .dispersive import DispersiveMaterial, _ade_coefficients, _restrict_group

__all__ = [
    "NonlinearMaterial",
    "KerrMaterial",
    "NonlinearMixin",
    "chi3_to_n2",
    "n2_to_chi3",
    "optical_intensity",
    "field_from_intensity",
    "spm_phase_shift",
    "thg_conversion_efficiency",
    "effective_nonlinear_volume",
    "kerr_cavity_frequency_shift",
    "lle_cw_bistability",
    "lle_soliton_analytic",
    "solve_lle",
]


def _resolve_n0(eps_r=1.0, n0=None):
    """Resolve background refractive index ``n_0`` from ``n0`` or ``eps_r``."""
    if n0 is not None:
        n_val = float(n0)
        if n_val <= 0.0:
            raise ValueError(f"Refractive index n0 must be > 0, got {n0}.")
        return n_val
    eps_val = float(eps_r)
    if eps_val <= 0.0:
        raise ValueError(f"Relative permittivity eps_r must be > 0, got {eps_r}.")
    return float(np.sqrt(eps_val))


def chi3_to_n2(chi3, eps_r=1.0, *, n0=None, factor=0.75):
    """
    Convert third-order susceptibility ``chi^(3)`` [m^2/V^2] to Kerr index ``n_2`` [m^2/W].

    For a monochromatic real electric field ``E(t) = Re[E_0 exp(+i omega t)]``
    with cycle-averaged optical intensity ``I = 0.5 * c * eps_0 * n_0 * |E_0|^2``,
    the fundamental-frequency nonlinear polarization is
    ``P_NL(omega) = factor * eps_0 * chi^(3) * |E_0|^2 * E_0`` with
    ``factor = 3/4 = 0.75`` from ``cos^3(theta) = (3/4) cos(theta) + (1/4) cos(3 theta)``.
    Equating ``Delta n = (factor * chi^(3) / (2 * n_0)) |E_0|^2 = n_2 * I`` yields::

        n_2 = factor * chi^(3) / (c * eps_0 * n_0^2)
            = (3 / 4) * chi^(3) / (c * eps_0 * n_0^2).

    Parameters
    ----------
    chi3 : float or array_like
        Third-order susceptibility ``chi^(3)`` [m^2/V^2].
    eps_r : float, optional
        Linear relative permittivity ``n_0^2`` [-], ``> 0``. Default ``1.0``.
    n0 : float, optional
        Linear refractive index ``n_0`` [-], ``> 0``. Overrides ``eps_r`` if given.
    factor : float, optional
        Real-field degeneracy factor (``3/4 = 0.75`` for linearly polarized
        optical fields). Exposed for sensitivity and mutation testing.

    Returns
    -------
    n2 : float or ndarray
        Nonlinear Kerr refractive index ``n_2`` [m^2/W].
    """
    n_bg = _resolve_n0(eps_r=eps_r, n0=n0)
    chi3_arr = np.asarray(chi3, dtype=float)
    out = float(factor) * chi3_arr / (c_light * eps_0 * n_bg**2)
    return float(out) if chi3_arr.ndim == 0 else out


def n2_to_chi3(n2, eps_r=1.0, *, n0=None, factor=0.75):
    """
    Convert Kerr index ``n_2`` [m^2/W] to third-order susceptibility ``chi^(3)`` [m^2/V^2].

    Inverts :func:`chi3_to_n2`::

        chi^(3) = (1 / factor) * c * eps_0 * n_0^2 * n_2
                = (4 / 3) * c * eps_0 * n_0^2 * n_2.

    Parameters
    ----------
    n2 : float or array_like
        Nonlinear Kerr refractive index ``n_2`` [m^2/W].
    eps_r : float, optional
        Linear relative permittivity ``n_0^2`` [-], ``> 0``. Default ``1.0``.
    n0 : float, optional
        Linear refractive index ``n_0`` [-], ``> 0``. Overrides ``eps_r`` if given.
    factor : float, optional
        Real-field degeneracy factor (default ``3/4 = 0.75``).

    Returns
    -------
    chi3 : float or ndarray
        Third-order susceptibility ``chi^(3)`` [m^2/V^2].
    """
    if float(factor) == 0.0:
        raise ValueError("factor must be non-zero.")
    n_bg = _resolve_n0(eps_r=eps_r, n0=n0)
    n2_arr = np.asarray(n2, dtype=float)
    out = (c_light * eps_0 * n_bg**2 / float(factor)) * n2_arr
    return float(out) if n2_arr.ndim == 0 else out


def optical_intensity(E_amp, eps_r=1.0, *, n0=None):
    """
    Cycle-averaged optical intensity ``I = 0.5 * c * eps_0 * n_0 * |E_0|^2`` [W/m^2].

    Parameters
    ----------
    E_amp : float or array_like
        Peak electric-field amplitude ``|E_0|`` [V/m].
    eps_r : float, optional
        Linear relative permittivity ``n_0^2`` [-]. Default ``1.0``.
    n0 : float, optional
        Linear refractive index ``n_0`` [-]. Overrides ``eps_r`` if given.

    Returns
    -------
    I : float or ndarray
        Cycle-averaged optical intensity [W/m^2].
    """
    n_bg = _resolve_n0(eps_r=eps_r, n0=n0)
    e_arr = np.asarray(E_amp)
    out = 0.5 * c_light * eps_0 * n_bg * np.abs(e_arr) ** 2
    return float(out) if e_arr.ndim == 0 else out


def field_from_intensity(intensity, eps_r=1.0, *, n0=None):
    """
    Peak electric-field amplitude ``|E_0| = sqrt(2 * I / (c * eps_0 * n_0))`` [V/m].

    Parameters
    ----------
    intensity : float or array_like
        Cycle-averaged optical intensity ``I`` [W/m^2], ``>= 0``.
    eps_r : float, optional
        Linear relative permittivity ``n_0^2`` [-]. Default ``1.0``.
    n0 : float, optional
        Linear refractive index ``n_0`` [-]. Overrides ``eps_r`` if given.

    Returns
    -------
    E_amp : float or ndarray
        Peak electric-field amplitude [V/m].
    """
    n_bg = _resolve_n0(eps_r=eps_r, n0=n0)
    i_arr = np.asarray(intensity, dtype=float)
    if np.any(i_arr < 0.0):
        raise ValueError("intensity must be >= 0.")
    out = np.sqrt(2.0 * i_arr / (c_light * eps_0 * n_bg))
    return float(out) if i_arr.ndim == 0 else out


def spm_phase_shift(
    intensity_or_E0,
    length,
    wavelength,
    *,
    n2=None,
    chi3=None,
    eps_r=1.0,
    n0=None,
    input_is_field=False,
    factor=0.75,
):
    """
    Analytic 1D self-phase modulation (SPM) nonlinear phase shift ``Delta Phi_NL`` [rad].

    For a monochromatic wave at vacuum wavelength ``lambda_0`` (wavenumber
    ``k_0 = 2 * pi / lambda_0``) propagating a distance ``L`` through a Kerr
    medium with ``Delta n_NL = n_2 * I = (3 * chi^(3) / (8 * n_0)) * |E_0|^2``::

        Delta Phi_NL = k_0 * n_2 * I * L
                     = (factor / (2 * n_0)) * k_0 * chi^(3) * |E_0|^2 * L.

    Parameters
    ----------
    intensity_or_E0 : float or array_like
        Optical intensity ``I`` [W/m^2] (default) or peak electric field ``|E_0|``
        [V/m] when ``input_is_field=True``.
    length : float
        Propagation length ``L`` [m], ``>= 0``.
    wavelength : float
        Vacuum wavelength ``lambda_0`` [m], ``> 0``.
    n2 : float, optional
        Nonlinear Kerr index [m^2/W]. Either ``n2`` or ``chi3`` must be provided.
    chi3 : float, optional
        Third-order susceptibility ``chi^(3)`` [m^2/V^2].
    eps_r : float, optional
        Linear relative permittivity ``n_0^2`` [-]. Default ``1.0``.
    n0 : float, optional
        Linear refractive index ``n_0`` [-]. Overrides ``eps_r`` if given.
    input_is_field : bool, optional
        If True, interpret ``intensity_or_E0`` as peak field ``|E_0|`` [V/m].
        Default False.
    factor : float, optional
        Real-field degeneracy factor (default ``3/4 = 0.75``).

    Returns
    -------
    delta_phi : float or ndarray
        Nonlinear phase shift ``Delta Phi_NL`` [rad].
    """
    L = float(length)
    lam = float(wavelength)
    if L < 0.0:
        raise ValueError(f"length must be >= 0, got {L}.")
    if lam <= 0.0:
        raise ValueError(f"wavelength must be > 0, got {lam}.")
    n_bg = _resolve_n0(eps_r=eps_r, n0=n0)
    if n2 is None and chi3 is None:
        raise ValueError("Provide either `n2` or `chi3`.")
    if n2 is None:
        n2_val = chi3_to_n2(chi3, n0=n_bg, factor=factor)
    else:
        n2_val = float(n2) * (float(factor) / 0.75)

    val_arr = np.asarray(intensity_or_E0, dtype=float)
    if input_is_field:
        intensity = optical_intensity(val_arr, n0=n_bg)
    else:
        intensity = val_arr
    k0 = 2.0 * np.pi / lam
    out = n2_val * k0 * intensity * L
    return float(out) if val_arr.ndim == 0 else out


def thg_conversion_efficiency(
    intensity_or_E0,
    length,
    wavelength,
    chi3,
    *,
    n1=1.0,
    n3=None,
    delta_k=None,
    input_is_field=False,
    factor_thg=0.25,
):
    """
    Undepleted-pump analytical third-harmonic generation (THG) efficiency in a slab.

    For a fundamental pump wave ``E_1(z, t) = Re[E_1(0) exp(i(omega_1 t - k_1 z))]``
    entering a nonlinear slab of thickness ``L``, the ``cos(3 omega_1 t)``
    component of ``E_1^3`` has amplitude ``factor_thg * E_1^3`` with
    ``factor_thg = 1/4 = 0.25``. Solving the slowly varying envelope equation
    with phase mismatch ``Delta k = k_3 - 3 k_1 = (3 omega_1 / c) (n_3 - n_1)``
    in the undepleted-pump approximation yields::

        |E_3(L)| = (3 * factor_thg * omega_1 * |chi^(3)| / (2 * n_3 * c))
                   * |E_1(0)|^3 * L * |sinc(Delta k * L / 2)|
                 = (3 * omega_1 * |chi^(3)| / (8 * n_3 * c))
                   * |E_1(0)|^3 * L * |sinc(Delta k * L / 2)|,

        eta_THG = I_3(L) / I_1(0) = (n_3 / n_1) * (|E_3(L)| / |E_1(0)|)^2,

    where ``sinc(x) = sin(x) / x``.

    Parameters
    ----------
    intensity_or_E0 : float or array_like
        Pump optical intensity ``I_1(0)`` [W/m^2] (default) or peak pump field
        ``|E_1(0)|`` [V/m] when ``input_is_field=True``.
    length : float or array_like
        Nonlinear slab thickness ``L`` [m], ``>= 0``.
    wavelength : float
        Fundamental vacuum wavelength ``lambda_1`` [m], ``> 0``.
    chi3 : float
        Third-order susceptibility ``chi^(3)`` [m^2/V^2].
    n1 : float, optional
        Refractive index at the fundamental frequency ``omega_1`` [-]. Default ``1.0``.
    n3 : float, optional
        Refractive index at the third harmonic ``3 * omega_1`` [-]. Defaults to ``n1``.
    delta_k : float, optional
        Explicit wavevector mismatch ``Delta k = k_3 - 3 * k_1`` [rad/m]. If
        None, computed as ``(6 * pi / wavelength) * (n3 - n1)``.
    input_is_field : bool, optional
        If True, interpret ``intensity_or_E0`` as ``|E_1(0)|`` [V/m]. Default False.
    factor_thg : float, optional
        Third-harmonic degeneracy factor from ``cos^3(theta)`` (default ``1/4 = 0.25``).

    Returns
    -------
    dict
        Dictionary with keys:
        - ``'efficiency'``: power conversion efficiency ``I_3(L) / I_1(0)`` [-],
        - ``'field_ratio'``: field amplitude ratio ``|E_3(L)| / |E_1(0)|`` [-],
        - ``'E3_amp'``: generated third-harmonic peak field ``|E_3(L)|`` [V/m],
        - ``'I3'``: generated third-harmonic intensity ``I_3(L)`` [W/m^2],
        - ``'delta_k'``: phase mismatch ``Delta k`` [rad/m],
        - ``'sinc_sq'``: phase-matching factor ``sinc^2(Delta k * L / 2)`` [-].
    """
    lam1 = float(wavelength)
    if lam1 <= 0.0:
        raise ValueError(f"wavelength must be > 0, got {lam1}.")
    n1_val = float(n1)
    n3_val = n1_val if n3 is None else float(n3)
    if n1_val <= 0.0 or n3_val <= 0.0:
        raise ValueError("Refractive indices n1 and n3 must be > 0.")

    omega_1 = 2.0 * np.pi * c_light / lam1
    dk = (
        3.0 * omega_1 * (n3_val - n1_val) / c_light
        if delta_k is None
        else float(delta_k)
    )

    L_arr = np.asarray(length, dtype=float)
    val_arr = np.asarray(intensity_or_E0, dtype=float)
    if input_is_field:
        e1_amp = np.abs(val_arr)
        i1_val = optical_intensity(e1_amp, n0=n1_val)
    else:
        i1_val = val_arr
        e1_amp = field_from_intensity(i1_val, n0=n1_val)

    # np.sinc(x) computes sin(pi * x) / (pi * x)
    half_phase = 0.5 * dk * L_arr
    sinc_val = np.sinc(half_phase / np.pi)
    sinc_sq = sinc_val**2

    prefactor = (
        3.0 * float(factor_thg) * omega_1 * abs(float(chi3)) / (2.0 * n3_val * c_light)
    )
    e3_amp = prefactor * (e1_amp**3) * L_arr * np.abs(sinc_val)
    with np.errstate(divide="ignore", invalid="ignore"):
        field_ratio = np.where(e1_amp > 0.0, e3_amp / e1_amp, 0.0)
    eta = (n3_val / n1_val) * (field_ratio**2)
    i3_val = eta * i1_val

    scalar = (L_arr.ndim == 0) and (val_arr.ndim == 0)
    if scalar:
        return {
            "efficiency": float(eta),
            "field_ratio": float(field_ratio),
            "E3_amp": float(e3_amp),
            "I3": float(i3_val),
            "delta_k": float(dk),
            "sinc_sq": float(sinc_sq),
        }
    return {
        "efficiency": eta,
        "field_ratio": field_ratio,
        "E3_amp": e3_amp,
        "I3": i3_val,
        "delta_k": float(dk),
        "sinc_sq": sinc_sq,
    }


def effective_nonlinear_volume(e_field_sq, dV, eps_r=1.0, nl_mask=None):
    """
    Compute the effective nonlinear mode volume ``V_eff,NL`` [m^3] of a cavity mode.

    Defined from the modal electric-field intensity profile ``|E(r)|^2`` as
    (Kippenberg et al. 2018 [10]; Herr et al. 2014 [9])::

        V_eff,NL = ( int eps_r(r) |E(r)|^2 dV )^2
                   / ( int_{Omega_NL} eps_r(r)^2 |E(r)|^4 dV ).

    For a 1D standing-wave Fabry-Perot mode ``E(z) = E_0 sin(m pi z / L)`` in a
    uniform cavity of physical volume ``V_phys = A * L``,
    ``int_0^L sin^2 dz = L/2`` and ``int_0^L sin^4 dz = 3 L / 8``, giving
    ``V_eff,NL = (2/3) V_phys``.

    Parameters
    ----------
    e_field_sq : array_like
        Squared electric-field magnitude ``|E(r)|^2`` [V^2/m^2] on the mesh.
    dV : float or array_like
        Cell or edge volume weights [m^3].
    eps_r : float or array_like, optional
        Relative permittivity ``eps_r(r)`` [-]. Default ``1.0``.
    nl_mask : array_like of bool or float, optional
        Mask or fill-fraction weights in ``[0, 1]`` selecting the nonlinear
        region ``Omega_NL``. Default is the entire domain.

    Returns
    -------
    V_eff : float
        Effective nonlinear mode volume [m^3].
    """
    e2 = np.asarray(e_field_sq, dtype=float)
    vol = np.asarray(dV, dtype=float)
    eps = np.asarray(eps_r, dtype=float)
    w_nl = 1.0 if nl_mask is None else np.asarray(nl_mask, dtype=float)

    num_integral = float(np.sum(eps * e2 * vol))
    den_integral = float(np.sum(w_nl * (eps**2) * (e2**2) * vol))
    if den_integral <= 0.0:
        raise ValueError("Nonlinear mode overlap denominator is zero.")
    return float((num_integral**2) / den_integral)


def kerr_cavity_frequency_shift(
    omega_0,
    chi3,
    *,
    e_field=None,
    dV=None,
    eps_r=1.0,
    nl_mask=None,
    stored_energy=None,
    V_eff=None,
    E_peak=None,
    cavity_dim=1,
    travelling_wave=False,
    factor=0.375,
):
    """
    First-order perturbation-theory Kerr resonance shift ``Delta omega_NL`` [rad/s].

    For a cavity mode with real field ``E(r, t) = Re[E_tilde(r) exp(+i omega t)]``,
    the fundamental-frequency permittivity perturbation is
    ``Delta eps_r(r) = (3/4) chi^(3)(r) |E_tilde(r)|^2``. First-order cavity
    perturbation theory gives::

        Delta omega_NL / omega_0
            = - (1 / 2) * (int Delta eps_r(r) |E_tilde(r)|^2 dV)
                        / (int eps_r(r) |E_tilde(r)|^2 dV)
            = - (3 / 8) * (int_{Omega_NL} chi^(3)(r) |E_tilde(r)|^4 dV)
                        / (int eps_r(r) |E_tilde(r)|^2 dV)
            = - (3 * chi^(3) * U) / (4 * eps_0 * n_0^4 * V_eff,NL)
            = - (c * n_2 * U) / (n_0^2 * V_eff,NL),

    where ``U = (1/2) int eps_0 eps_r(r) |E_tilde(r)|^2 dV`` [J] is the stored
    electromagnetic energy of the mode.

    In a homogeneous ``D``-dimensional standing-wave cavity (``D = 1, 2, 3``)
    with ``E_tilde(r) = E_peak prod_{j=1}^D sin(k_j x_j)`` and uniform
    ``eps_r = n_0^2``, the spatial overlap ratio is
    ``(int sin^4 dx) / (int sin^2 dx) = (3/4)^D`` (or ``(3/4)^(D-1)`` for a
    unidirectional azimuthal travelling wave along one periodic coordinate),
    giving::

        Delta omega_NL / omega_0 = - (3/8) * (3/4)^D * (chi^(3) / eps_r) * E_peak^2
                                 = - (9/32) * (chi^(3) / eps_r) * E_peak^2   (for D = 1).

    Parameters
    ----------
    omega_0 : float
        Unperturbed cold-cavity angular resonance frequency [rad/s], ``> 0``.
    chi3 : float or array_like
        Third-order susceptibility ``chi^(3)`` [m^2/V^2].
    e_field : array_like, optional
        Modal electric-field profile ``E_tilde(r)`` [V/m] on the grid.
    dV : float or array_like, optional
        Cell/edge volume weights [m^3] when ``e_field`` is provided.
    eps_r : float or array_like, optional
        Relative permittivity ``eps_r`` [-]. Default ``1.0``.
    nl_mask : array_like, optional
        Nonlinear region mask or edge weights when ``e_field`` is provided.
    stored_energy : float or array_like, optional
        Total electromagnetic energy ``U`` [J] stored in the mode (used together
        with ``V_eff``).
    V_eff : float, optional
        Effective nonlinear mode volume ``V_eff,NL`` [m^3] (used together with
        ``stored_energy``).
    E_peak : float or array_like, optional
        Peak standing-wave electric field amplitude [V/m] for an analytical
        ``cavity_dim``-dimensional sin-mode cavity.
    cavity_dim : int, optional
        Number of standing-wave sine dimensions ``D`` in ``{0, 1, 2, 3}`` when
        ``E_peak`` is used. Default ``1`` (1D Fabry-Perot cavity).
    travelling_wave : bool, optional
        If True with ``E_peak``, reduce the number of standing-wave factors of
        ``3/4`` by one (e.g. a 1D ring WGM travelling wave has spatial ratio 1,
        and a 2D ring WGM with radial standing profile has one transverse factor).
    factor : float, optional
        Temporal perturbation factor ``(1/2) * (3/4) = 3/8 = 0.375``. Exposed
        for mutation testing.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``'delta_omega'``: angular frequency shift ``Delta omega_NL`` [rad/s],
        - ``'delta_f'``: frequency shift ``Delta f_NL = Delta omega_NL / (2 pi)`` [Hz],
        - ``'rel_shift'``: fractional shift ``Delta omega_NL / omega_0`` [-],
        - ``'V_eff'``: effective nonlinear mode volume [m^3] if computed or given,
        - ``'stored_energy'``: stored modal energy ``U`` [J] if computed or given.
    """
    w0 = float(omega_0)
    fac = float(factor)

    if e_field is not None:
        e_arr = np.asarray(e_field)
        e2 = np.abs(e_arr) ** 2
        vol = 1.0 if dV is None else np.asarray(dV, dtype=float)
        eps_arr = np.asarray(eps_r, dtype=float)
        w_nl = 1.0 if nl_mask is None else np.asarray(nl_mask, dtype=float)
        chi3_arr = np.asarray(chi3, dtype=float)

        denom = float(np.sum(eps_arr * e2 * vol))
        numer = np.sum(w_nl * chi3_arr * (e2**2) * vol)
        rel_shift = -fac * numer / denom
        v_eff_val = effective_nonlinear_volume(e2, vol, eps_r=eps_arr, nl_mask=w_nl)
        u_val = 0.5 * eps_0 * denom
    elif stored_energy is not None and V_eff is not None:
        u_val = np.asarray(stored_energy, dtype=float)
        v_eff_val = float(V_eff)
        eps_sc = float(eps_r)
        chi3_sc = float(chi3)
        # Since fac = 3/8 and U = 0.5 * eps_0 * int eps_r |E|^2 dV:
        # -fac * chi3 * int |E|^4 / int eps_r |E|^2 = -(2 * fac) * chi3 * U / (eps_0 * eps_r^2 * V_eff)
        rel_shift = -(2.0 * fac) * chi3_sc * u_val / (eps_0 * (eps_sc**2) * v_eff_val)
    elif E_peak is not None:
        e_pk = np.asarray(E_peak, dtype=float)
        eps_sc = float(eps_r)
        chi3_sc = float(chi3)
        n_sw = max(0, int(cavity_dim) - (1 if travelling_wave else 0))
        spatial_ratio = (0.75) ** n_sw
        rel_shift = -fac * spatial_ratio * (chi3_sc / eps_sc) * (e_pk**2)
        v_eff_val = None
        u_val = None
    else:
        raise ValueError(
            "Provide either `e_field`, `(stored_energy, V_eff)`, or `E_peak`."
        )

    d_omega = w0 * rel_shift
    d_f = d_omega / (2.0 * np.pi)
    if np.ndim(rel_shift) == 0:
        rel_shift = float(rel_shift)
        d_omega = float(d_omega)
        d_f = float(d_f)
    return {
        "delta_omega": d_omega,
        "delta_f": d_f,
        "rel_shift": rel_shift,
        "V_eff": v_eff_val,
        "stored_energy": u_val if u_val is None or np.ndim(u_val) > 0 else float(u_val),
    }


class NonlinearMaterial(list):
    """
    Base class for nonlinear optical materials in `SolverFIT3D`.

    Like `DispersiveMaterial` and `GainMaterial`, a `NonlinearMaterial` instance
    is also a 3-element list ``[eps_inf, 1.0, 0.0]`` so it can be passed wherever
    Wakis expects a static material triple ``[eps_r, mu_r, sigma]``.
    """

    is_nonlinear = True
    has_electric = True
    has_magnetic = False
    is_mu_anisotropic = False

    def __init__(self, eps_inf=1.0, sigma=0.0, name=None):
        if float(eps_inf) <= 0.0:
            raise ValueError(f"eps_inf must be > 0, got {eps_inf}.")
        if float(sigma) < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma}.")
        self.eps_inf = float(eps_inf)
        self.sigma = float(sigma)
        self.name = name
        super().__init__([self.eps_inf, 1.0, 0.0])

    @property
    def n0(self):
        """High-frequency linear refractive index ``sqrt(eps_inf)`` [-]."""
        return float(np.sqrt(self.eps_inf))


class KerrMaterial(NonlinearMaterial):
    """
    Instantaneous Kerr (``chi^(3)``) and delayed Raman nonlinear material for `SolverFIT3D`.

    Models the constitutive relation (Taflove & Hagness 2005 [1], Ch. 9; Joseph
    & Taflove 1997 [2])::

        D(t) = eps_0 eps_inf E(t) + P_lin(t)
               + eps_0 chi^(3) [ (1 - f_R) |E(t)|^2 + f_R S(t) ] E(t),

    where ``S(t)`` is the delayed Raman vibrational response obeying::

        d^2 S / dt^2 + 2 Gamma_R dS/dt + Omega_R^2 S = Omega_R^2 |E(t)|^2.

    Parameters
    ----------
    eps_inf : float, optional
        High-frequency relative permittivity ``eps_inf > 0`` [-]. Default ``1.0``.
    chi3 : float, optional
        Isotropic third-order nonlinear susceptibility ``chi^(3)`` [m^2/V^2].
        Default ``0.0``.
    n2 : float, optional
        Intensity-dependent Kerr refractive index ``n_2`` [m^2/W]. If provided,
        sets ``chi3 = (4/3) * c * eps_0 * eps_inf * n2``. Cannot be specified
        simultaneously with a non-zero ``chi3``.
    f_R : float, optional
        Fractional Raman contribution in ``[0, 1]`` [-]. Default ``0.0`` (purely
        instantaneous electronic Kerr response; ``f_R = 0.18`` for silica).
    tau_1 : float, optional
        Raman vibrational period parameter ``tau_1`` [s], ``> 0`` (Blow & Wood
        1989 [4]; ``12.2 fs`` for silica). Default ``12.2e-15``.
    tau_2 : float, optional
        Raman damping lifetime ``tau_2`` [s], ``> 0`` (``32.0 fs`` for silica).
        Default ``32.0e-15``.
    omega_R : float, optional
        Explicit Raman oscillator frequency ``Omega_R`` [rad/s], ``> 0``.
        Overrides ``sqrt(1/tau_1^2 + 1/tau_2^2)`` if given.
    gamma_R : float, optional
        Explicit Raman damping coefficient ``Gamma_R`` [rad/s], ``>= 0`` in
        ``d^2 S/dt^2 + 2 Gamma_R dS/dt + Omega_R^2 S = Omega_R^2 |E|^2``.
        Overrides ``1 / tau_2`` if given.
    poles : sequence of Pole, optional
        Optional linear dispersive poles (`Lorentz`, `Drude`, `Debye` from
        `wakis.dispersive`). Default ``()``.
    sigma : float, optional
        Static electric conductivity [S/m], ``>= 0``. Default ``0.0``.
    method : {'newton', 'fixed_point', 'explicit_ade'}, optional
        Time-domain nonlinear E-field update algorithm:
        - ``'newton'`` (default): per-edge cubic Newton-Raphson solver,
        - ``'fixed_point'``: Picard fixed-point iteration,
        - ``'explicit_ade'``: explicit predictor-corrector ``P_NL`` ADE update.
    max_iter : int, optional
        Maximum iterations per step for ``'newton'`` or ``'fixed_point'``.
        Default ``6``.
    tol : float, optional
        Relative convergence tolerance for ``'newton'`` or ``'fixed_point'``.
        Default ``1e-13``.
    coupling : {'edge', 'cell'}, optional
        Whether ``|E|^2`` in the instantaneous Kerr update is evaluated per edge
        (``'edge'``, default) or summed over ``E_x^2 + E_y^2 + E_z^2`` in each
        Yee cell (``'cell'``).
    degeneracy_factor : float, optional
        Multiplier on the instantaneous ``chi^(3)`` term (default ``1.0``; set to
        ``4.0 / 3.0`` to simulate dropping the ``3/4`` real-field factor in
        mutation checks).
    name : str, optional
        Optional material name.
    """

    def __init__(
        self,
        eps_inf=1.0,
        chi3=0.0,
        *,
        n2=None,
        f_R=0.0,
        tau_1=12.2e-15,
        tau_2=32.0e-15,
        omega_R=None,
        gamma_R=None,
        poles=(),
        sigma=0.0,
        method="newton",
        max_iter=6,
        tol=1e-13,
        coupling="edge",
        degeneracy_factor=1.0,
        name=None,
    ):
        super().__init__(eps_inf=eps_inf, sigma=sigma, name=name)
        if n2 is not None:
            if float(chi3) != 0.0:
                raise ValueError("Specify either `chi3` or `n2`, not both.")
            chi3 = n2_to_chi3(n2, eps_r=self.eps_inf)
        self.chi3 = float(chi3)
        self.degeneracy_factor = float(degeneracy_factor)
        self.chi3_eff = self.chi3 * self.degeneracy_factor

        self.f_R = float(f_R)
        if not (0.0 <= self.f_R <= 1.0):
            raise ValueError(f"f_R must be in [0, 1], got {self.f_R}.")

        self.tau_1 = float(tau_1)
        self.tau_2 = float(tau_2)
        if self.tau_1 <= 0.0 or self.tau_2 <= 0.0:
            raise ValueError("Raman time constants tau_1 and tau_2 must be > 0.")

        if omega_R is not None:
            self.omega_R = float(omega_R)
        else:
            self.omega_R = float(np.sqrt(1.0 / self.tau_1**2 + 1.0 / self.tau_2**2))
        if gamma_R is not None:
            self.gamma_R = float(gamma_R)
        else:
            self.gamma_R = float(1.0 / self.tau_2)
        if self.omega_R <= 0.0:
            raise ValueError(f"omega_R must be > 0, got {self.omega_R}.")
        if self.gamma_R < 0.0:
            raise ValueError(f"gamma_R must be >= 0, got {self.gamma_R}.")

        self.poles = tuple(poles)
        self._linear_dispersive = (
            DispersiveMaterial(self.eps_inf, list(self.poles), sigma=self.sigma)
            if (self.poles or self.sigma > 0.0)
            else None
        )

        method_clean = str(method).strip().lower()
        if method_clean not in ("newton", "fixed_point", "explicit_ade"):
            raise ValueError(
                f"Unknown nonlinear update method {method!r}; expected "
                "'newton', 'fixed_point', or 'explicit_ade'."
            )
        self.method = method_clean
        self.max_iter = int(max_iter)
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")
        self.tol = float(tol)

        coupling_clean = str(coupling).strip().lower()
        if coupling_clean not in ("edge", "cell"):
            raise ValueError(
                f"Unknown coupling mode {coupling!r}; expected 'edge' or 'cell'."
            )
        self.coupling = coupling_clean

    @property
    def n2(self):
        """Intensity-dependent Kerr refractive index ``n_2`` [m^2/W] at ``n_0 = sqrt(eps_inf)``."""
        return chi3_to_n2(self.chi3_eff, eps_r=self.eps_inf)

    @property
    def has_raman(self):
        """True if delayed Raman response is active (``f_R > 0`` and ``chi3 != 0``)."""
        return bool(self.f_R > 0.0 and self.chi3_eff != 0.0)

    def epsilon_linear(self, f):
        """
        Continuous linear relative permittivity ``eps_lin(f)`` in ``exp(+i w t)`` convention.

        Parameters
        ----------
        f : float or array_like
            Frequency [Hz].

        Returns
        -------
        eps : complex or ndarray
            Linear complex relative permittivity [-].
        """
        if self._linear_dispersive is not None:
            return self._linear_dispersive.epsilon(f)
        f_arr = np.asarray(f, dtype=float)
        out = np.full_like(f_arr, self.eps_inf, dtype=complex)
        return complex(out) if f_arr.ndim == 0 else out

    def epsilon_linear_discrete(self, f, dt):
        """
        Discrete linear relative permittivity ``eps_lin,d(f, dt)`` at timestep ``dt`` [s].
        """
        if self._linear_dispersive is not None:
            return self._linear_dispersive.epsilon_discrete(f, dt)
        f_arr = np.asarray(f, dtype=float)
        out = np.full_like(f_arr, self.eps_inf, dtype=complex)
        return complex(out) if f_arr.ndim == 0 else out

    epsilon = epsilon_linear
    epsilon_discrete = epsilon_linear_discrete

    def epsilon_kerr(self, f, E_amp=0.0, *, intensity=None, factor=0.75):
        """
        Effective continuous-wave relative permittivity including the Kerr shift.

        For a monochromatic wave of peak amplitude ``|E_0|`` (or intensity ``I``),
        ``eps_eff(f) = eps_lin(f) + factor * chi^(3) * |E_0|^2`` with ``factor = 3/4``
        (since both the instantaneous fraction ``1 - f_R`` and the DC component of
        the Raman response ``f_R * H_R(0) = f_R`` contribute to the CW index shift).
        """
        eps_lin = self.epsilon_linear(f)
        if intensity is not None:
            n_real = float(np.real(np.sqrt(eps_lin))) if np.ndim(eps_lin) == 0 else np.real(np.sqrt(eps_lin))
            e2 = 2.0 * np.asarray(intensity, dtype=float) / (c_light * eps_0 * n_real)
        else:
            e2 = np.abs(np.asarray(E_amp)) ** 2
        return eps_lin + float(factor) * self.chi3_eff * e2

    def raman_impulse_response(self, t):
        """
        Causal delayed Raman impulse response ``h_R(t)`` [1/s] for time ``t`` [s].

        Satisfies ``d^2 h_R/dt^2 + 2 Gamma_R dh_R/dt + Omega_R^2 h_R = Omega_R^2 delta(t)``
        and ``int_0^infty h_R(t) dt = 1``::

            h_R(t) = (Omega_R^2 / omega_d) exp(-Gamma_R t) sin(omega_d t) Theta(t),

        where ``omega_d = sqrt(Omega_R^2 - Gamma_R^2) = 1 / tau_1``.
        """
        t_arr = np.asarray(t, dtype=float)
        scalar = t_arr.ndim == 0
        t_pos = np.maximum(t_arr, 0.0)
        disc = self.omega_R**2 - self.gamma_R**2
        if disc > 0.0:
            omega_d = np.sqrt(disc)
            h = (self.omega_R**2 / omega_d) * np.exp(-self.gamma_R * t_pos) * np.sin(omega_d * t_pos)
        elif disc == 0.0:
            h = (self.omega_R**2) * t_pos * np.exp(-self.gamma_R * t_pos)
        else:
            kappa_d = np.sqrt(-disc)
            h = (self.omega_R**2 / kappa_d) * np.exp(-self.gamma_R * t_pos) * np.sinh(kappa_d * t_pos)
        h = np.where(t_arr >= 0.0, h, 0.0)
        return float(h) if scalar else h

    def raman_susceptibility(self, omega):
        """
        Continuous Raman transfer function ``H_R(omega)`` [-] in ``exp(+i w t)`` convention::

            H_R(omega) = Omega_R^2 / (Omega_R^2 - omega^2 + 2 i Gamma_R omega).
        """
        w = np.asarray(omega, dtype=float)
        out = (self.omega_R**2) / (self.omega_R**2 - w**2 + 2j * self.gamma_R * w)
        return complex(out) if w.ndim == 0 else out

    def raman_susceptibility_discrete(self, omega, dt):
        """
        Discrete ADE Raman transfer function ``H_{R,d}(omega, dt)`` [-] at timestep ``dt`` [s].
        """
        dt = float(dt)
        w = np.asarray(omega, dtype=float)
        w_tilde_sq = (2.0 / dt * np.sin(0.5 * w * dt)) ** 2
        damping = 2j * self.gamma_R * np.sin(w * dt) / dt
        out = (self.omega_R**2) / (self.omega_R**2 - w_tilde_sq + damping)
        return complex(out) if w.ndim == 0 else out

    chi_raman = raman_susceptibility
    chi_raman_discrete = raman_susceptibility_discrete

    def raman_step_coefficients(self, dt):
        """
        Second-order central-difference coefficients ``(alpha_R, xi_R, zeta_R)``
        for ``S^(n+1) = alpha_R S^n + xi_R S^(n-1) + zeta_R |E^n|^2``.
        """
        dt = float(dt)
        den = 1.0 + self.gamma_R * dt
        alpha_r = (2.0 - (self.omega_R * dt) ** 2) / den
        xi_r = (self.gamma_R * dt - 1.0) / den
        zeta_r = ((self.omega_R * dt) ** 2) / den
        return float(alpha_r), float(xi_r), float(zeta_r)

    def is_stable(self, dt, courant=1.0, max_E=0.0):
        """
        Verify whether timestep ``dt`` [s] resolves linear/Raman poles and hyperbolicity.

        Parameters
        ----------
        dt : float
            Solver timestep [s].
        courant : float, optional
            Vacuum Courant number ``c * dt * sqrt(sum 1/dx_i^2)``. Default ``1.0``.
        max_E : float, optional
            Expected peak electric field [V/m] to check hyperbolicity
            ``eps_inf + 3 * (1 - f_R) * chi^(3) * max_E^2 > 0`` for self-defocusing
            media (``chi^(3) < 0``). Default ``0.0``.

        Returns
        -------
        bool
        """
        dt = float(dt)
        if dt <= 0.0:
            return False
        if self.has_raman and self.omega_R * dt >= 2.0:
            return False
        if self._linear_dispersive is not None:
            if not self._linear_dispersive.is_stable(dt, courant=courant):
                return False
        else:
            if courant > np.sqrt(self.eps_inf) * (1.0 + 1e-12):
                return False
        if max_E > 0.0 and self.chi3_eff < 0.0:
            eps_diff = self.eps_inf + 3.0 * (1.0 - self.f_R) * self.chi3_eff * (float(max_E) ** 2)
            if eps_diff <= 0.0 or courant > np.sqrt(eps_diff):
                return False
        return True

    def __repr__(self):
        label = f"{self.name}: " if self.name else ""
        return (
            f"KerrMaterial({label}eps_inf={self.eps_inf:.6g}, "
            f"chi3={self.chi3:.6g}, f_R={self.f_R:.6g}, method={self.method!r})"
        )


def _solve_cubic_kerr_edge(a, b, rhs, E_init, max_iter=6, tol=1e-13):
    """
    Solve ``a * E + b * E^3 = rhs`` elementwise via Newton-Raphson for real fields.
    """
    if np.iscomplexobj(rhs) or np.iscomplexobj(E_init):
        raise NotImplementedError(
            "Kerr nonlinear updates require real-valued fields (float64/float32); "
            "complex-dtype fields are not supported because applying |E|^2 E to a "
            "complex phasor overpredicts SPM by 4/3 and omits 3*omega THG."
        )

    # Real fields: f(E) = a * E + b * E^3 - rhs = 0, f'(E) = a + 3 * b * E^2
    e2_init = E_init * E_init
    denom0 = a + b * e2_init
    E = np.where(denom0 > 1e-30, rhs / denom0, rhs / a)
    for _ in range(max_iter):
        e2 = E * E
        deriv = a + 3.0 * b * e2
        if np.any(deriv <= 0.0):
            raise FloatingPointError(
                "Non-hyperbolic Kerr state encountered (a + 3 * b * E^2 <= 0); "
                "reduce field amplitude or check self-defocusing chi3 < 0 limit."
            )
        f_val = (a + b * e2) * E - rhs
        dE = f_val / deriv
        E = E - dE
        if np.max(np.abs(dE)) <= tol * (np.max(np.abs(E)) + 1e-30):
            break
    return E


def _solve_cubic_kerr_cell(a, b, rhs, E_init, inv, n_cells, max_iter=6, tol=1e-13):
    """
    Solve ``(a_e + b_e * u_{cell(e)}) * E_e = rhs_e`` where ``u_c = sum_{e in c} |E_e|^2``.
    """
    e2_init = np.abs(E_init) ** 2
    u_cell = np.bincount(inv, weights=e2_init, minlength=n_cells)
    for _ in range(max_iter):
        u_edge = u_cell[inv]
        denom = a + b * u_edge
        if np.any(denom <= 0.0):
            raise FloatingPointError(
                "Non-hyperbolic Kerr cell state encountered (a + b * |E|^2 <= 0)."
            )
        E = rhs / denom
        e2 = np.abs(E) ** 2
        f_cell = u_cell - np.bincount(inv, weights=e2, minlength=n_cells)
        df_du = 1.0 + np.bincount(
            inv, weights=2.0 * b * e2 / denom, minlength=n_cells
        )
        if np.any(df_du <= 0.0):
            raise FloatingPointError(
                "Non-hyperbolic Kerr Jacobian encountered in cell-coupled solve."
            )
        du = f_cell / df_du
        u_cell = np.maximum(0.0, u_cell - du)
        if np.max(np.abs(du)) <= tol * (np.max(u_cell) + 1e-30):
            break
    return rhs / (a + b * u_cell[inv])


def _solve_fixed_point_kerr(a, b, rhs, E_init, max_iter=6, tol=1e-13, inv=None, n_cells=0):
    """Picard fixed-point iteration ``E^(m+1) = rhs / (a + b * |E^(m)|^2)``."""
    E = E_init.copy()
    for _ in range(max_iter):
        e2 = np.abs(E) ** 2
        if inv is not None:
            e2 = np.bincount(inv, weights=e2, minlength=n_cells)[inv]
        denom = a + b * e2
        if np.any(denom <= 0.0):
            raise FloatingPointError(
                "Non-hyperbolic Kerr state encountered in fixed-point iteration."
            )
        E_next = rhs / denom
        diff = np.max(np.abs(E_next - E))
        E = E_next
        if diff <= tol * (np.max(np.abs(E)) + 1e-30):
            break
    return E


class NonlinearMixin:
    """
    `SolverFIT3D` mixin for Kerr (``chi^(3)``) and delayed Raman nonlinear media.

    Wraps ``self.one_step`` once on first registration so that all CPU/MPI/MKL/CPML/TFSF
    stepper variants are supported with minimal hooks in `SolverFIT3D`.
    """

    def add_nonlinear_material(
        self, material, center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0)
    ):
        """
        Fill a box region (or entire domain when ``center=None, size=None``) with a
        `NonlinearMaterial` (`KerrMaterial`).

        Primary ``E`` edges strictly inside the box receive fill weight ``1.0``;
        tangential ``E`` edges lying on a box boundary face receive weight ``0.5``
        (arithmetic interface averaging with the background medium). Later calls
        override earlier ones on shared edges.

        Parameters
        ----------
        material : KerrMaterial
            Nonlinear optical material model.
        center, size : tuple of 3 float or None, optional
            Centre and extent ``(x, y, z)`` [m] of the box. If both are ``None``,
            all non-PEC edges in the domain receive unit weight.

        Raises
        ------
        TypeError
            If ``material`` is not a `NonlinearMaterial`.
        NotImplementedError
            If the solver uses ``use_gpu=True``, ``fused=True``, or
            ``scheme='tete'``.
        ValueError
            If the solver timestep ``dt`` violates stability for the material.
        """
        if not isinstance(material, NonlinearMaterial):
            raise TypeError(
                f"material must be a NonlinearMaterial (e.g. KerrMaterial), "
                f"got {type(material).__name__}."
            )
        if center is None and size is None:
            idx_e = np.arange(3 * self.N, dtype=int)
            w_e = np.ones(3 * self.N, dtype=float)
        else:
            if center is None:
                center = (0.0, 0.0, 0.0)
            if size is None:
                size = (0.0, 0.0, 0.0)
            idx_e, w_e = self._box_edges(center, size, dual=False)
        self._add_nonlinear_edges(material, idx_e, w_e)

    def _apply_stl_nonlinear(self):
        """Attach nonlinear updates to STL solids whose material is a `NonlinearMaterial`."""
        for key, material in self.grid.stl_materials.items():
            if not isinstance(material, NonlinearMaterial):
                continue
            mask = np.reshape(self.grid.grid[key], (self.Nx, self.Ny, self.Nz))
            fraction = np.reshape(mask, self.N, order="F").astype(float)
            fraction = np.tile(fraction, 3)
            idx = np.flatnonzero(fraction)
            self._add_nonlinear_edges(material, idx, fraction[idx], eps_bg=1.0)

    def _check_nonlinear_supported_and_stable(self, material):
        if not isinstance(material, NonlinearMaterial):
            raise TypeError("material must be a NonlinearMaterial")
        if getattr(self, "scheme", "yee") == "tete":
            raise NotImplementedError(
                "scheme='tete' does not support nonlinear materials; use scheme='yee'."
            )
        fused = getattr(self, "_fused_step", None)
        if getattr(self, "fused", False) or (
            fused is not None and self.one_step is fused
        ):
            raise NotImplementedError(
                "Nonlinear media do not support the fused GPU step (fused=True) yet; "
                "use fused=False."
            )
        if getattr(self, "use_gpu", False):
            raise NotImplementedError(
                "Nonlinear media do not support GPU execution (use_gpu=True) yet; "
                "use use_gpu=False."
            )
        if np.issubdtype(getattr(self, "dtype", float), np.complexfloating) or np.iscomplexobj(
            self.E.array
        ):
            raise NotImplementedError(
                "Nonlinear media require real-valued fields (float64/float32); "
                "complex-dtype fields are not supported because applying |E|^2 E to a "
                "complex phasor overpredicts SPM by 4/3 and omits 3*omega THG."
            )
        courant = (
            c_light
            * self.dt
            * np.sqrt(
                np.sum([1.0 / np.min(d) ** 2 for d in (self.dx, self.dy, self.dz)])
            )
        )
        if not material.is_stable(self.dt, courant=courant):
            raise ValueError(
                f"dt={self.dt:.3e} s is unstable for {material!r}; pass a "
                "smaller `dt` or `cfln` to SolverFIT3D (see KerrMaterial.is_stable)."
            )

    def _install_nonlinear_stepper(self):
        """Wrap ``self.one_step`` once with ``_one_step_nonlinear``."""
        if getattr(self, "_nonlinear_installed", False):
            return
        self._nonlinear_installed = True
        if not hasattr(self, "_nonlinear_groups"):
            self._nonlinear_groups = []
        if (
            getattr(self, "activate_abc", False)
            and self.one_step == self._one_step_with_abc
        ):
            self._one_step_nonnl = self._one_step_backend
            self._one_step_backend = self._one_step_nonlinear
        else:
            self._one_step_nonnl = self.one_step
            self.one_step = self._one_step_nonlinear

    def _add_nonlinear_edges(self, material, idx, weight, eps_bg=None):
        """Register flat E indices ``idx`` with fill fractions ``weight`` for a nonlinear material."""
        self._check_nonlinear_supported_and_stable(material)

        ieps = np.asarray(self.ieps.array)
        keep = (weight > 0.0) & (ieps[idx] != 0.0)
        idx, weight = idx[keep], weight[keep]
        if idx.size == 0:
            return

        self._install_nonlinear_stepper()

        for group in self._nonlinear_groups:
            old = ~np.isin(group["idx_host"], idx)
            if not old.all():
                _restrict_nonlinear_group(group, old)
        for group in getattr(self, "_dispersive_groups", ()):
            old = ~np.isin(group["idx_host"], idx)
            if not old.all():
                _restrict_group(group, old, self.E.xp)

        eps_s = 1.0 / (eps_0 * ieps[idx])
        bg = eps_s if eps_bg is None else np.full_like(eps_s, float(eps_bg))
        w_arr = np.asarray(weight, dtype=self.dtype)
        eps_e = (1.0 - w_arr) * bg + w_arr * material.eps_inf

        lorentz_cf, debye_cf = _ade_coefficients(material, self.dt)
        beta_sum = w_arr * sum(beta for _, beta in debye_cf)
        s_hat = 0.5 * w_arr * material.sigma * self.dt / eps_0

        cells, inv = np.unique(idx % self.N, return_inverse=True)
        n_cells = cells.size
        zeros_e = np.zeros(idx.size, dtype=self.dtype)
        zeros_real = np.zeros(
            n_cells if material.coupling == "cell" else idx.size, dtype=float
        )

        same_eps = bool(
            material.chi3_eff == 0.0
            and not lorentz_cf
            and not debye_cf
            and material.sigma == 0.0
            and np.allclose(eps_e, eps_s, rtol=0.0, atol=1e-15)
        )

        raman_state = None
        if material.has_raman:
            alpha_r, xi_r, zeta_r = material.raman_step_coefficients(self.dt)
            raman_state = [
                alpha_r,
                xi_r,
                zeta_r,
                zeros_real.copy(),
                zeros_real.copy(),
            ]

        group = {
            "material": material,
            "idx_host": idx,
            "idx": idx,
            "weight": w_arr,
            "eps_s": np.asarray(eps_s, dtype=self.dtype),
            "eps_e": np.asarray(eps_e, dtype=self.dtype),
            "a1": np.asarray(eps_e - s_hat - beta_sum, dtype=self.dtype),
            "den": np.asarray(eps_e + s_hat + beta_sum, dtype=self.dtype),
            "lorentz": [(*cf, zeros_e.copy(), zeros_e.copy()) for cf in lorentz_cf],
            "debye": [(*cf, zeros_e.copy()) for cf in debye_cf],
            "p_nl": zeros_e.copy(),
            "raman": raman_state,
            "cells": cells,
            "inv": inv,
            "n_cells": n_cells,
            "same_eps": same_eps,
            "initialized": False,
        }

        self._nonlinear_groups = [
            g for g in self._nonlinear_groups if g["idx_host"].size > 0
        ] + [group]

    def _one_step_nonlinear(self):
        """Run the underlying FIT step and apply the nonlinear Kerr/Raman E update."""
        groups = getattr(self, "_nonlinear_groups", None)
        if not groups:
            self._one_step_nonnl()
            return

        E_n = [None if g["same_eps"] else self.E.array[g["idx"]].copy() for g in groups]
        self._one_step_nonnl()
        E = self.E.array

        for g, En in zip(groups, E_n):
            if g["same_eps"]:
                # Exact bit-for-bit linear pass-through when chi3 == 0 and eps_e == eps_s
                continue

            mat = g["material"]
            w = g["weight"]
            inv = g["inv"]
            n_cells = g["n_cells"]

            # Linear curl + source + CPML/TFSF contribution:
            # (E* - E^n) * eps_s = (dt / eps_0) * (curl H^(n+1/2) - J^(n+1/2) + CPML)
            acc_lin = (E[g["idx"]] - En) * g["eps_s"] + g["a1"] * En

            for k, (alpha, xi, zeta, p, p_old) in enumerate(g["lorentz"]):
                p_new = alpha * p + xi * p_old + zeta * En
                acc_lin = acc_lin - w * (p_new - p)
                g["lorentz"][k] = (alpha, xi, zeta, p_new, p)

            for kappa, _, p_deb in g["debye"]:
                acc_lin = acc_lin - w * (kappa - 1.0) * p_deb

            if mat.chi3_eff == 0.0:
                E_new = acc_lin / g["den"]
                for k, (kappa, beta, p_deb) in enumerate(g["debye"]):
                    g["debye"][k] = (kappa, beta, kappa * p_deb + beta * (E_new + En))
                E[g["idx"]] = E_new
                continue

            # Advance delayed Raman vibrational variable S^(n+1) centered at t^n
            if g["raman"] is not None:
                alpha_r, xi_r, zeta_r, s_cur, s_old = g["raman"]
                e2_n = np.abs(En) ** 2
                if mat.coupling == "cell":
                    e2_n = np.bincount(inv, weights=e2_n, minlength=n_cells)
                s_new = alpha_r * s_cur + xi_r * s_old + zeta_r * e2_n
                g["raman"][3], g["raman"][4] = s_new, s_cur
                s_edge = s_new[inv] if mat.coupling == "cell" else s_new
            else:
                s_edge = 0.0

            chi3_inst = (1.0 - mat.f_R) * mat.chi3_eff
            chi3_ram = mat.f_R * mat.chi3_eff
            if not g["initialized"]:
                g["initialized"] = True
                if np.any(En != 0.0) and not np.any(g["p_nl"] != 0.0):
                    e2_init = np.abs(En) ** 2
                    if mat.coupling == "cell":
                        e2_init = np.bincount(
                            inv, weights=e2_init, minlength=n_cells
                        )[inv]
                    g["p_nl"] = chi3_inst * e2_init * En
            b_edge = w * chi3_inst
            a_edge = g["den"] + w * chi3_ram * s_edge
            rhs = acc_lin + w * g["p_nl"]

            if mat.method == "newton":
                if mat.coupling == "cell" and chi3_inst != 0.0:
                    E_new = _solve_cubic_kerr_cell(
                        a_edge,
                        b_edge,
                        rhs,
                        En,
                        inv,
                        n_cells,
                        max_iter=mat.max_iter,
                        tol=mat.tol,
                    )
                elif chi3_inst != 0.0:
                    E_new = _solve_cubic_kerr_edge(
                        a_edge,
                        b_edge,
                        rhs,
                        En,
                        max_iter=mat.max_iter,
                        tol=mat.tol,
                    )
                else:
                    E_new = rhs / a_edge
                e2_new = np.abs(E_new) ** 2
                if mat.coupling == "cell":
                    e2_new = np.bincount(inv, weights=e2_new, minlength=n_cells)[inv]
                p_nl_new = (chi3_inst * e2_new + chi3_ram * s_edge) * E_new

            elif mat.method == "fixed_point":
                E_new = _solve_fixed_point_kerr(
                    a_edge,
                    b_edge,
                    rhs,
                    En,
                    max_iter=mat.max_iter,
                    tol=mat.tol,
                    inv=inv if mat.coupling == "cell" else None,
                    n_cells=n_cells,
                )
                e2_new = np.abs(E_new) ** 2
                if mat.coupling == "cell":
                    e2_new = np.bincount(inv, weights=e2_new, minlength=n_cells)[inv]
                p_nl_new = (chi3_inst * e2_new + chi3_ram * s_edge) * E_new

            else:  # 'explicit_ade'
                # Linear predictor E_pred followed by explicit P_NL^(n+1) update
                e2_n_edge = np.abs(En) ** 2
                if mat.coupling == "cell":
                    e2_n_edge = np.bincount(inv, weights=e2_n_edge, minlength=n_cells)[inv]
                denom_pred = a_edge + b_edge * e2_n_edge
                if np.any(denom_pred <= 0.0):
                    raise FloatingPointError(
                        "Non-hyperbolic Kerr state encountered in explicit_ade predictor."
                    )
                E_pred = rhs / denom_pred
                e2_pred = np.abs(E_pred) ** 2
                if mat.coupling == "cell":
                    e2_pred = np.bincount(inv, weights=e2_pred, minlength=n_cells)[inv]
                p_nl_new = (chi3_inst * e2_pred + chi3_ram * s_edge) * E_pred
                E_new = (acc_lin - w * (p_nl_new - g["p_nl"])) / g["den"]

            for k, (kappa, beta, p_deb) in enumerate(g["debye"]):
                g["debye"][k] = (kappa, beta, kappa * p_deb + beta * (E_new + En))

            g["p_nl"] = p_nl_new
            E[g["idx"]] = E_new

        if getattr(self, "use_mpi", False):
            self._mpi_communicate(self.E)

    def get_nonlinear_polarization(self, group=0):
        """
        Return the nonlinear polarization ``P_NL = eps_0 * p_nl`` [C/m^2] on edges of ``group``.
        """
        groups = getattr(self, "_nonlinear_groups", None)
        if not groups:
            raise RuntimeError("No nonlinear material has been registered on this solver.")
        return eps_0 * groups[group]["p_nl"].copy()

    def get_raman_response(self, group=0):
        """
        Return the delayed Raman state ``S`` [V^2/m^2] of ``group`` (or zeros if ``f_R == 0``).
        """
        groups = getattr(self, "_nonlinear_groups", None)
        if not groups:
            raise RuntimeError("No nonlinear material has been registered on this solver.")
        g = groups[group]
        if g["raman"] is None:
            return np.zeros(g["idx"].size, dtype=float)
        return g["raman"][3].copy()

    def nonlinear_field_energy(self):
        """
        Total electromagnetic energy [J] including the Kerr nonlinear electric energy density::

            w_e(E) = (1/2) eps_0 eps_e |E|^2 + (3/4) eps_0 w_e (1 - f_R) chi^(3) |E|^4.
        """
        w_lin = float(self.field_energy())
        groups = getattr(self, "_nonlinear_groups", None)
        if not groups:
            return w_lin

        itA = self.grid.itA.toarray()
        dA = np.divide(1.0, itA, out=np.zeros_like(itA), where=itA != 0)
        vol_E = self.grid.L.toarray() * dA
        E_arr = self.E.array
        w_extra = 0.0
        for g in groups:
            idx = g["idx"]
            mat = g["material"]
            dv = vol_E[idx]
            e2 = np.abs(E_arr[idx]) ** 2
            # Correct linear permittivity difference if eps_e != eps_s
            w_extra += float(np.sum(0.5 * eps_0 * (g["eps_e"] - g["eps_s"]) * e2 * dv))
            chi3_inst = (1.0 - mat.f_R) * mat.chi3_eff
            if chi3_inst != 0.0:
                w_extra += float(
                    np.sum(0.75 * eps_0 * g["weight"] * chi3_inst * (e2**2) * dv)
                )
        return w_lin + w_extra

    def reset_nonlinear_state(self):
        """Zero all auxiliary nonlinear polarizations, Raman states, and linear poles."""
        for g in getattr(self, "_nonlinear_groups", ()):
            g["p_nl"].fill(0.0)
            g["initialized"] = False
            if g["raman"] is not None:
                g["raman"][3].fill(0.0)
                g["raman"][4].fill(0.0)
            g["lorentz"] = [
                (a, x, z, 0 * p, 0 * q) for a, x, z, p, q in g["lorentz"]
            ]
            g["debye"] = [(k, b, 0 * p) for k, b, p in g["debye"]]


def _restrict_nonlinear_group(group, keep):
    """Drop edges of a nonlinear group where ``keep`` is False and rebuild cell mapping."""
    group["idx_host"] = group["idx_host"][keep]
    group["idx"] = group["idx"][keep]
    for key in ("weight", "eps_s", "eps_e", "a1", "den", "p_nl"):
        group[key] = group[key][keep]
    group["lorentz"] = [
        (a, x, z, p[keep], q[keep]) for a, x, z, p, q in group["lorentz"]
    ]
    group["debye"] = [(k, b, p[keep]) for k, b, p in group["debye"]]
    if group["idx_host"].size == 0:
        return
    cell_of_kept = group["cells"][group["inv"][keep]]
    new_cells, new_inv = np.unique(cell_of_kept, return_inverse=True)
    old_cell_mask = np.isin(group["cells"], new_cells)
    group["cells"] = new_cells
    group["inv"] = new_inv
    group["n_cells"] = new_cells.size
    if group["raman"] is not None:
        if group["material"].coupling == "cell":
            group["raman"][3] = group["raman"][3][old_cell_mask]
            group["raman"][4] = group["raman"][4][old_cell_mask]
        else:
            group["raman"][3] = group["raman"][3][keep]
            group["raman"][4] = group["raman"][4][keep]


# ---------------------------------------------------------------------------
# Lugiato-Lefever Equation (LLE) solvers and analytic microcomb theory
# ---------------------------------------------------------------------------


def lle_cw_bistability(zeta_0, f_pump):
    """
    Solve the homogeneous CW steady state ``I_0 [1 + (zeta_0 - I_0)^2] = f^2`` of the LLE.

    Parameters
    ----------
    zeta_0 : float or array_like
        Normalized pump-cavity detuning ``zeta_0 = 2 (omega_0 - omega_p) / kappa`` [-].
    f_pump : float
        Normalized external pump amplitude ``f`` [-], ``>= 0``.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``'zeta_0'``: detuning array,
        - ``'lower_branch'``: lowest real root ``I_0`` at each ``zeta_0`` (always exists),
        - ``'middle_branch'``: unstable middle root (``np.nan`` outside bistable range),
        - ``'upper_branch'``: upper CW root (``np.nan`` when single-valued),
        - ``'zeta_bistable_min'``, ``'zeta_bistable_max'``: analytical spinodal bounds
          where three real roots exist (for ``f^2 > 8 / (3 sqrt(3))``),
        - ``'zeta_cw_max'``: resonance tip ``zeta_max^CW = f^2`` where ``I_{0,max} = f^2``.
    """
    f2 = float(f_pump) ** 2
    z_arr = np.atleast_1d(np.asarray(zeta_0, dtype=float))
    lower = np.empty_like(z_arr)
    middle = np.full_like(z_arr, np.nan)
    upper = np.full_like(z_arr, np.nan)

    for idx, z in enumerate(z_arr):
        # I^3 - 2*z*I^2 + (1 + z^2)*I - f^2 = 0
        roots = np.roots([1.0, -2.0 * z, 1.0 + z * z, -f2])
        real_roots = np.sort(np.real(roots[np.abs(np.imag(roots)) < 1e-9]))
        lower[idx] = real_roots[0]
        if real_roots.size >= 3:
            middle[idx] = real_roots[1]
            upper[idx] = real_roots[2]

    return {
        "zeta_0": float(z_arr[0]) if np.ndim(zeta_0) == 0 else z_arr,
        "lower_branch": float(lower[0]) if np.ndim(zeta_0) == 0 else lower,
        "middle_branch": float(middle[0]) if np.ndim(zeta_0) == 0 else middle,
        "upper_branch": float(upper[0]) if np.ndim(zeta_0) == 0 else upper,
        "zeta_cw_max": float(f2),
        "mi_threshold_f": 1.0,
        "bistability_threshold_f2": float(8.0 / (3.0 * np.sqrt(3.0))),
    }


def lle_soliton_analytic(phi, zeta_0, d2, f_pump, *, factor_max=None):
    """
    Analytical single dissipative Kerr soliton (DKS) profile and ``sech^2`` spectrum.

    From Herr et al., *Nat. Photonics* **8**, 145 (2014) [9], Eqs. (6)-(7), and
    Kippenberg et al., *Science* **361**, eaan8083 (2018) [10], for the
    dimensionless LLE::

        dPsi/dtau = -(1 + i zeta_0) Psi + i (d_2 / 2) d^2 Psi / dphi^2 + i |Psi|^2 Psi + f,

    a single stationary bright soliton on the low-intensity CW background
    ``Psi_bg`` exists for ``zeta_min(f) < zeta_0 <= zeta_max(f) = (pi^2 / 8) f^2``
    and is approximated by::

        Psi(phi) = Psi_bg + sqrt(2 zeta_0) exp(i phi_0) sech(phi / Delta phi_sol),
        Delta phi_sol = sqrt(d_2 / (2 zeta_0)),
        cos(phi_0) = sqrt(8 zeta_0) / (pi f),

    with comb-line spectral envelope (for ``mu != 0``)::

        |Psi_mu|^2 = (d_2 / 4) sech^2( (pi * mu / 2) * sqrt(d_2 / (2 zeta_0)) ).

    Parameters
    ----------
    phi : array_like
        Azimuthal angle grid in ``[-pi, pi)`` [rad].
    zeta_0 : float
        Normalized red detuning ``zeta_0 = 2 (omega_0 - omega_p) / kappa > 0`` [-].
    d2 : float
        Normalized anomalous second-order dispersion ``d_2 = 2 D_2 / kappa > 0`` [-].
    f_pump : float
        Normalized pump amplitude ``f`` [-], ``> 0``.
    factor_max : float, optional
        Prefactor in ``zeta_max = factor_max * f^2`` (default ``pi^2 / 8 = 1.23370055``).
        Exposed for mutation testing.

    Returns
    -------
    dict
        Analytical soliton waveform, spectrum, existence bounds, and widths.
    """
    z0 = float(zeta_0)
    d2_val = float(d2)
    f_val = float(f_pump)
    if z0 <= 0.0:
        raise ValueError(f"Soliton requires red detuning zeta_0 > 0, got {z0}.")
    if d2_val <= 0.0:
        raise ValueError(f"Bright DKS requires anomalous dispersion d2 > 0, got {d2_val}.")
    if f_val <= 0.0:
        raise ValueError(f"f_pump must be > 0, got {f_val}.")

    fac_max = float(np.pi**2 / 8.0) if factor_max is None else float(factor_max)
    zeta_max = fac_max * (f_val**2)

    phi_arr = np.asarray(phi, dtype=float)
    n_phi = phi_arr.size
    mu = np.fft.fftshift(np.fft.fftfreq(n_phi, d=1.0 / n_phi))

    # Low-intensity CW background root on the lower bistable branch
    cw = lle_cw_bistability(z0, f_val)
    i_bg = float(cw["lower_branch"])
    psi_bg = f_val / (1.0 + 1j * (z0 - i_bg))

    delta_phi = float(np.sqrt(d2_val / (2.0 * z0)))
    amp_peak = float(np.sqrt(2.0 * z0))
    cos_phi0 = np.clip(np.sqrt(8.0 * z0) / (np.pi * f_val), -1.0, 1.0)
    phi_0 = float(np.arccos(cos_phi0))

    sech_phi = 1.0 / np.cosh(phi_arr / delta_phi)
    psi_sol = psi_bg + amp_peak * np.exp(1j * phi_0) * sech_phi

    alpha_mu = float(0.5 * np.pi * delta_phi)
    sech_mu = 1.0 / np.cosh(np.clip(alpha_mu * mu, -300.0, 300.0))
    spec_sech2 = 0.25 * d2_val * (sech_mu**2)

    # 3 dB (half-maximum) full spectral width in mode numbers mu:
    # sech^2(alpha_mu * mu_hwhm) = 1/2  =>  mu_fwhm = 2 * arccosh(sqrt(2)) / alpha_mu
    arccosh_sqrt2 = float(np.log(1.0 + np.sqrt(2.0)))
    fwhm_mu = float(2.0 * arccosh_sqrt2 / alpha_mu)
    fwhm_phi = float(2.0 * arccosh_sqrt2 * delta_phi)

    return {
        "phi": phi_arr,
        "psi": psi_sol,
        "psi_bg": complex(psi_bg),
        "amplitude_peak": amp_peak,
        "phase_0": phi_0,
        "delta_phi_sol": delta_phi,
        "fwhm_phi": fwhm_phi,
        "mu": mu,
        "spectrum_sech2": spec_sech2,
        "alpha_mu": alpha_mu,
        "fwhm_mu": fwhm_mu,
        "zeta_max": zeta_max,
        "in_existence_range": bool(z0 <= zeta_max),
    }


def solve_lle(
    d2,
    f_pump,
    zeta_0,
    *,
    n_modes=256,
    dtau=0.01,
    n_steps=4000,
    psi_init=None,
    d_int_norm=None,
    tau_R=0.0,
    record_every=0,
    seed=42,
    noise_amp=1e-4,
):
    """
    Integrate the dimensionless Lugiato-Lefever equation (LLE) via symmetric
    split-step Fourier (Strang splitting) with exact exponential linear propagator.

    Solves (Herr et al. 2014 [9], Eq. 6; Chembo & Menyuk 2013 [8])::

        dPsi/dtau = -(1 + i zeta_0(tau)) Psi
                    - i D_int,norm(mu) Psi
                    + i |Psi|^2 Psi
                    - i tau_R Psi d(|Psi|^2)/dphi
                    + f,

    where ``D_int,norm(mu) = (1/2) d_2 mu^2 + ...`` in azimuthal mode number ``mu``.

    Parameters
    ----------
    d2 : float
        Normalized second-order dispersion ``d_2 = 2 D_2 / kappa`` [-].
    f_pump : float
        Normalized pump amplitude ``f`` [-].
    zeta_0 : float or callable or array_like
        Normalized detuning ``zeta_0 = 2 (omega_0 - omega_p) / kappa`` [-].
        May be a constant float, a callable ``zeta_0(tau)``, or a 1D array of
        length ``n_steps`` for a laser detuning scan.
    n_modes : int, optional
        Number of azimuthal grid points / Fourier modes (even integer). Default ``256``.
    dtau : float, optional
        Dimensionless timestep ``Delta tau = (kappa / 2) Delta t``. Default ``0.01``.
    n_steps : int, optional
        Number of integration steps. Default ``4000``.
    psi_init : array_like, optional
        Initial intracavity envelope ``Psi(phi, 0)`` of shape ``(n_modes,)``.
        If None, initializes to small complex noise of amplitude ``noise_amp``.
    d_int_norm : array_like, optional
        Explicit normalized integrated dispersion ``D_int(mu) / (kappa / 2)`` on
        the unshifted FFT mode order ``np.fft.fftfreq(n_modes, d=1/n_modes)``.
        If None, uses ``0.5 * d2 * mu^2``.
    tau_R : float, optional
        Normalized Raman shock time ``tau_R`` [-]. Default ``0.0``.
    record_every : int, optional
        If ``> 0``, store snapshots every ``record_every`` steps. Default ``0``.
    seed : int, optional
        Random seed for default initial noise. Default ``42``.
    noise_amp : float, optional
        Noise amplitude when ``psi_init`` is None. Default ``1e-4``.

    Returns
    -------
    dict
        Dictionary containing:
        - ``'phi'``: azimuthal coordinate array in ``[-pi, pi)`` [rad],
        - ``'mu'``: fftshifted mode indices ``[-n_modes/2, ..., n_modes/2 - 1]``,
        - ``'psi'``: final field ``Psi(phi)`` of shape ``(n_modes,)``,
        - ``'spectrum'``: fftshifted modal power spectrum ``|Psi_mu|^2``,
        - ``'tau'``: time array of recorded steps,
        - ``'zeta_trace'``: detuning ``zeta_0`` at every step,
        - ``'power_trace'``: mean intracavity power ``mean(|Psi|^2)`` at every step,
        - ``'peak_trace'``: peak intracavity intensity ``max(|Psi|^2)`` at every step,
        - ``'snapshots'``: recorded ``Psi`` snapshots if ``record_every > 0``.
    """
    n_phi = int(n_modes)
    dt = float(dtau)
    nt = int(n_steps)
    f_val = float(f_pump)

    phi = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    mu_fft = np.fft.fftfreq(n_phi, d=1.0 / n_phi)
    mu_shifted = np.fft.fftshift(mu_fft)

    if d_int_norm is None:
        disp_mu = 0.5 * float(d2) * (mu_fft**2)
    else:
        disp_mu = np.asarray(d_int_norm, dtype=float)
        if disp_mu.shape != (n_phi,):
            raise ValueError(f"Expected d_int_norm of shape ({n_phi},), got {disp_mu.shape}.")

    if psi_init is None:
        rng = np.random.default_rng(seed)
        psi = noise_amp * (rng.standard_normal(n_phi) + 1j * rng.standard_normal(n_phi))
    else:
        psi = np.asarray(psi_init, dtype=np.complex128).copy()

    if callable(zeta_0):
        zeta_arr = np.array([float(zeta_0((k + 0.5) * dt)) for k in range(nt)], dtype=float)
        const_zeta = False
    elif np.ndim(zeta_0) == 0:
        z_const = float(zeta_0)
        zeta_arr = np.full(nt, z_const, dtype=float)
        const_zeta = True
    else:
        zeta_arr = np.asarray(zeta_0, dtype=float)
        if zeta_arr.shape != (nt,):
            raise ValueError(f"Expected zeta_0 array of length {nt}, got {zeta_arr.shape}.")
        const_zeta = False

    def _linear_op(z_val):
        op = -1.0 - 1j * (z_val + disp_mu)
        exp_half = np.exp(0.5 * dt * op)
        op0 = op[0]
        pump_half = f_val * n_phi * (exp_half[0] - 1.0) / op0
        return exp_half, pump_half

    if const_zeta:
        exp_half_cached, pump_half_cached = _linear_op(zeta_arr[0])

    power_trace = np.empty(nt, dtype=float)
    peak_trace = np.empty(nt, dtype=float)
    snapshots = []
    tau_snaps = []

    for step in range(nt):
        if const_zeta:
            exp_half, pump_half = exp_half_cached, pump_half_cached
        else:
            exp_half, pump_half = _linear_op(zeta_arr[step])

        # First linear half-step in Fourier space
        psi_hat = np.fft.fft(psi)
        psi_hat *= exp_half
        psi_hat[0] += pump_half
        psi = np.fft.ifft(psi_hat)

        # Full nonlinear step in real space
        abs2 = np.abs(psi) ** 2
        if tau_R != 0.0:
            d_abs2_dphi = np.real(np.fft.ifft(1j * mu_fft * np.fft.fft(abs2)))
            nl_phase = (abs2 - float(tau_R) * d_abs2_dphi) * dt
        else:
            nl_phase = abs2 * dt
        psi *= np.exp(1j * nl_phase)

        # Second linear half-step in Fourier space
        psi_hat = np.fft.fft(psi)
        psi_hat *= exp_half
        psi_hat[0] += pump_half
        psi = np.fft.ifft(psi_hat)

        abs2_end = np.abs(psi) ** 2
        power_trace[step] = float(np.mean(abs2_end))
        peak_trace[step] = float(np.max(abs2_end))

        if record_every > 0 and ((step + 1) % record_every == 0 or step == nt - 1):
            snapshots.append(psi.copy())
            tau_snaps.append((step + 1) * dt)

    psi_modes = np.fft.fftshift(np.fft.fft(psi) / n_phi)
    spectrum = np.abs(psi_modes) ** 2

    return {
        "phi": phi,
        "mu": mu_shifted,
        "psi": psi,
        "psi_modes": psi_modes,
        "spectrum": spectrum,
        "zeta_trace": zeta_arr,
        "power_trace": power_trace,
        "peak_trace": peak_trace,
        "tau_snapshots": np.asarray(tau_snaps, dtype=float),
        "snapshots": np.asarray(snapshots) if snapshots else np.empty((0, n_phi), dtype=np.complex128),
    }
