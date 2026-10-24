# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Time-varying electromagnetic media for the FIT time-domain solver.

Physical continuity across a temporal interface
-----------------------------------------------
In a spatially homogeneous medium whose constitutive parameters change in time
(for instance an abrupt switch ``(eps_1, mu_1) -> (eps_2, mu_2)`` at ``t = t_0``),
Maxwell's curl equations are

    curl E = -dB/dt,        curl H = dD/dt + J_c + J_src.

Integrating both equations over an infinitesimal time interval
``[t_0 - delta_t, t_0 + delta_t]`` as ``delta_t -> 0+`` shows that while ``E``,
``H`` and ``J_c`` remain bounded, their spatial curls and conduction currents
integrate to zero across the zero-duration step. Consequently, the fundamental
conserved quantities across a temporal interface are the **electric displacement
field** ``D`` and the **magnetic flux density** ``B`` (Morgenthaler 1958 [1];
Mendonca & Shukla 2002 [2]; Galiffi et al. 2022 [3]):

    D(r, t_0+) = D(r, t_0-),        B(r, t_0+) = B(r, t_0-),

whereas the electric and magnetic field intensities jump discontinuously:

    E(r, t_0+) = (eps_1 / eps_2) E(r, t_0-),
    H(r, t_0+) = (mu_1 / mu_2)   H(r, t_0-).

Because spatial translation symmetry holds at every instant, the spatial
wavevector ``k`` of a plane wave is conserved (``k_2 = k_1 = k``), while its
angular frequency shifts according to the new phase velocity:

    omega_2 = omega_1 * sqrt((eps_1 * mu_1) / (eps_2 * mu_2)).

Time-level centering and D-conserving FIT update
------------------------------------------------
On the staggered FIT leapfrog grid with timestep ``dt``:

* Primary electric field ``E^n`` and displacement ``D^n = eps_0 eps_r(t^n) E^n``
  are defined at integer time levels ``t^n = n * dt``.
* Dual magnetic field ``H^(n+1/2)`` and flux density
  ``B^(n+1/2) = mu_0 mu_r(t^(n+1/2)) H^(n+1/2)`` are defined at half-integer
  time levels ``t^(n+1/2) = (n + 1/2) * dt``.

Integrating Ampere's law from ``t^n`` to ``t^(n+1)`` centered at ``t^(n+1/2)``
yields the conservative update

    eps_0 [eps_eff(t^(n+1)) E^(n+1) - eps_eff(t^n) E^n] / dt
    + sigma_eff(t^(n+1/2)) (E^(n+1) + E^n) / 2
    + w * sum_p (P_p^(n+1) - P_p^n) / dt
        = (curl H)^(n+1/2) - J_src^(n+1/2),

where ``eps_eff(t) = (1 - w) eps_bg + w eps_r(t)`` and ``sigma_eff(t) = w sigma(t)``
for edge fill fraction ``w``. When ``sigma = 0`` and no dispersive poles are
present, this simplifies to

    D^(n+1) = D^n + dt * [(curl H)^(n+1/2) - J_src^(n+1/2)],
    E^(n+1) = D^(n+1) / (eps_0 eps_eff(t^(n+1))),

so ``D`` is updated purely by the curl of ``H`` and is conserved to machine
round-off across any abrupt jump in ``eps_r(t)``.

Time-varying Drude plasma
-------------------------
For a cold electron plasma with time-varying density ``n_e(t)`` (plasma
frequency ``omega_p(t) = sqrt(n_e(t) e^2 / (eps_0 m_e))``, e.g. rapid
photoionization where newly created electrons are born at rest [2, 5]), the
linearized momentum equation is

    dJ_p/dt + gamma(t) J_p = eps_0 omega_p^2(t) E(t),        J_p = dP_p/dt.

Discretizing ``P_p = eps_0 p_p`` at integer time levels ``t^n`` with central
differences gives the explicit second-order recursion

    p_p^(n+1) = alpha(t^n) p_p^n + xi(t^n) p_p^(n-1) + zeta(t^n) E^n,

which preserves continuity of ``E``, ``H``, and ``J_p`` across an ionization
front while upshifting the wave frequency to ``sqrt(omega_1^2 + omega_p2^2)``.

References
----------
[1] F. R. Morgenthaler, "Velocity modulation of electromagnetic waves,"
    IRE Trans. Microw. Theory Tech. 6(2), 167-172 (1958),
    doi:10.1109/TMTT.1958.1124533.
[2] J. T. Mendonca and P. K. Shukla, "Time refraction and time reflection:
    two basic concepts," Phys. Scr. 65(2), 160-163 (2002),
    doi:10.1238/Physica.Regular.065a00160.
[3] E. Galiffi et al., "Photonics of time-varying media," Adv. Photonics 4(1),
    014002 (2022), doi:10.1117/1.AP.4.1.014002.
[4] M. Lyubarov et al., "Amplified emission and lasing in photonic time
    crystals," Science 377(6604), 425-428 (2022), doi:10.1126/science.abo3324.
[5] S. C. Wilks, J. M. Dawson and W. B. Mori, "Frequency up-conversion of
    electromagnetic radiation with use of an overdense plasma," Phys. Rev.
    Lett. 61(3), 337-340 (1988), doi:10.1103/PhysRevLett.61.337.
[6] H. Moussa et al., "Observation of temporal reflection and broadband
    frequency translation at photonic time interfaces," Nat. Phys. 19,
    863-868 (2023), doi:10.1038/s41567-023-01975-y.
"""

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0

from .dispersive import Debye, Lorentz, _restrict_group
from .field import Field


def _eval_param(param, t):
    """Evaluate a scalar or callable time profile `param` at time `t` [s]."""
    if callable(param):
        val = param(t)
        return float(val) if np.ndim(val) == 0 else np.asarray(val, dtype=float)
    return float(param)


class StepProfile:
    """
    Temporal step profile between two values `v1` and `v2` at `t_switch`.

    For ``rise_time == 0`` the transition is an abrupt Heaviside step::

        v(t) = v1   for t < t_switch,
        v(t) = v2   for t >= t_switch.

    For ``rise_time > 0`` the transition is a smooth hyperbolic-tangent ramp
    of 10%-90% duration ``rise_time`` centered at ``t_switch``::

        v(t) = v1 + 0.5 * (v2 - v1) * (1 + tanh(ln(9) * (t - t_switch) / rise_time)).

    Parameters
    ----------
    v1, v2 : float
        Initial (`t < t_switch`) and final (`t >= t_switch`) parameter values.
    t_switch : float
        Switching instant [s].
    rise_time : float, optional
        10%-90% transition duration [s], >= 0. Default is 0.0 (abrupt step).
    """

    def __init__(self, v1, v2, t_switch, rise_time=0.0):
        self.v1 = float(v1)
        self.v2 = float(v2)
        self.t_switch = float(t_switch)
        self.rise_time = float(rise_time)
        if self.rise_time < 0.0:
            raise ValueError(f"rise_time must be >= 0, got {rise_time}.")

    def __call__(self, t):
        t_arr = np.asarray(t, dtype=float)
        if self.rise_time == 0.0:
            out = np.where(t_arr >= self.t_switch, self.v2, self.v1)
        else:
            # ln(9) / 2 ~ 1.09861229 maps [-rise_time/2, +rise_time/2] to 10%-90%
            arg = np.clip(
                np.log(9.0) * (t_arr - self.t_switch) / self.rise_time, -60.0, 60.0
            )
            out = self.v1 + 0.5 * (self.v2 - self.v1) * (1.0 + np.tanh(arg))
        return float(out) if np.ndim(t) == 0 else out

    def __repr__(self):
        return (
            f"StepProfile(v1={self.v1:.6g}, v2={self.v2:.6g}, "
            f"t_switch={self.t_switch:.6g}, rise_time={self.rise_time:.6g})"
        )


class SlabTimeProfile:
    """
    Temporal slab (double temporal interface) active on ``[t_start, t_end)``.

    Models a parameter that switches from ``v_bg`` to ``v_slab`` at ``t_start``
    and returns to ``v_end`` (defaults to ``v_bg``) at ``t_end``, forming a
    temporal Fabry-Perot resonator (Moussa et al. 2023 [6]).

    Parameters
    ----------
    v_bg : float
        Background value for ``t < t_start``.
    v_slab : float
        Slab value during ``t_start <= t < t_end``.
    t_start, t_end : float
        Start and end times of the temporal slab [s] (``t_end >= t_start``).
    v_end : float, optional
        Value for ``t >= t_end``; defaults to ``v_bg``.
    rise_time : float, optional
        10%-90% switching rise time [s], >= 0. Default 0.0.
    """

    def __init__(self, v_bg, v_slab, t_start, t_end, v_end=None, rise_time=0.0):
        self.v_bg = float(v_bg)
        self.v_slab = float(v_slab)
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.v_end = float(v_bg if v_end is None else v_end)
        self.rise_time = float(rise_time)
        if self.t_end < self.t_start:
            raise ValueError("t_end must be >= t_start.")
        if self.rise_time < 0.0:
            raise ValueError("rise_time must be >= 0.")

    def __call__(self, t):
        t_arr = np.asarray(t, dtype=float)
        if self.rise_time == 0.0:
            out = np.where(
                t_arr < self.t_start,
                self.v_bg,
                np.where(t_arr < self.t_end, self.v_slab, self.v_end),
            )
        else:
            s1 = 0.5 * (
                1.0
                + np.tanh(
                    np.clip(
                        np.log(9.0) * (t_arr - self.t_start) / self.rise_time,
                        -60.0,
                        60.0,
                    )
                )
            )
            s2 = 0.5 * (
                1.0
                + np.tanh(
                    np.clip(
                        np.log(9.0) * (t_arr - self.t_end) / self.rise_time,
                        -60.0,
                        60.0,
                    )
                )
            )
            out = (
                self.v_bg
                + (self.v_slab - self.v_bg) * s1
                + (self.v_end - self.v_slab) * s2
            )
        return float(out) if np.ndim(t) == 0 else out

    def __repr__(self):
        return (
            f"SlabTimeProfile(v_bg={self.v_bg:.6g}, v_slab={self.v_slab:.6g}, "
            f"t_start={self.t_start:.6g}, t_end={self.t_end:.6g})"
        )


class CosineModulation:
    """
    Periodic sinusoidal modulation for a Photonic Time Crystal (PTC)::

        v(t) = v0 * (1 + modulation_depth * cos(omega_mod * (t - t_start) + phase))

    for ``t_start <= t <= t_end``, and ``v0`` outside that window.

    Parameters
    ----------
    v0 : float
        Unmodulated average value.
    modulation_depth : float
        Relative modulation amplitude ``m`` (requires ``|m| < 1`` when
        representing a positive permittivity ``eps_r(t) > 0``).
    omega_mod : float
        Modulation angular frequency ``Omega`` [rad/s].
    phase : float, optional
        Initial modulation phase [rad]. Default is 0.0.
    t_start : float, optional
        Modulation turn-on time [s]. Default is 0.0.
    t_end : float, optional
        Modulation turn-off time [s]. Default is ``np.inf``.
    """

    def __init__(
        self,
        v0,
        modulation_depth,
        omega_mod,
        phase=0.0,
        t_start=0.0,
        t_end=np.inf,
    ):
        self.v0 = float(v0)
        self.modulation_depth = float(modulation_depth)
        self.omega_mod = float(omega_mod)
        self.phase = float(phase)
        self.t_start = float(t_start)
        self.t_end = float(t_end)

    def __call__(self, t):
        t_arr = np.asarray(t, dtype=float)
        active = (t_arr >= self.t_start) & (t_arr <= self.t_end)
        mod = self.v0 * (
            1.0
            + self.modulation_depth
            * np.cos(self.omega_mod * (t_arr - self.t_start) + self.phase)
        )
        out = np.where(active, mod, self.v0)
        return float(out) if np.ndim(t) == 0 else out

    def __repr__(self):
        return (
            f"CosineModulation(v0={self.v0:.6g}, m={self.modulation_depth:.6g}, "
            f"omega_mod={self.omega_mod:.6g})"
        )


class TimeVaryingDrude:
    """
    Time-varying Drude plasma pole with ``omega_p(t)`` and ``gamma(t)``.

    Integrates the cold-plasma auxiliary differential equation::

        dJ_p/dt + gamma(t) J_p = eps_0 omega_p^2(t) E(t),        J_p = dP_p/dt,

    using the central-difference polarization update at integer time levels
    ``t^n = n * dt``.

    Parameters
    ----------
    omega_p : float or callable
        Plasma angular frequency ``omega_p(t)`` [rad/s], >= 0.
    gamma : float or callable, optional
        Electron collision rate ``gamma(t)`` [rad/s], >= 0. Default is 0.0.
    omega_0 : float or callable, optional
        Resonance angular frequency ``omega_0(t)`` [rad/s] (0.0 for a Drude
        plasma; > 0 for a time-varying Lorentz oscillator). Default is 0.0.
    """

    def __init__(self, omega_p, gamma=0.0, omega_0=0.0):
        self.omega_p = omega_p
        self.gamma = gamma
        self.omega_0 = omega_0

    def omega_p_at(self, t):
        """Instantaneous plasma angular frequency [rad/s] at time `t` [s]."""
        val = _eval_param(self.omega_p, t)
        if np.any(val < 0):
            raise ValueError(f"omega_p(t) must be >= 0 at t={t}, got {val}.")
        return val

    def gamma_at(self, t):
        """Instantaneous collision rate [rad/s] at time `t` [s]."""
        val = _eval_param(self.gamma, t)
        if np.any(val < 0):
            raise ValueError(f"gamma(t) must be >= 0 at t={t}, got {val}.")
        return val

    def omega_0_at(self, t):
        """Instantaneous resonance frequency [rad/s] at time `t` [s]."""
        val = _eval_param(self.omega_0, t)
        if np.any(val < 0):
            raise ValueError(f"omega_0(t) must be >= 0 at t={t}, got {val}.")
        return val

    def ade_coefficients(self, t, dt):
        """
        Return ``(alpha, xi, zeta)`` for ``p^(n+1) = alpha p^n + xi p^(n-1) + zeta E^n``.
        """
        wp = self.omega_p_at(t)
        gam = self.gamma_at(t)
        w0 = self.omega_0_at(t)
        d = 1.0 + 0.5 * gam * dt
        alpha = (2.0 - (w0 * dt) ** 2) / d
        xi = (0.5 * gam * dt - 1.0) / d
        zeta = (wp * dt) ** 2 / d
        return alpha, xi, zeta

    def __repr__(self):
        return f"TimeVaryingDrude(omega_p={self.omega_p!r}, gamma={self.gamma!r})"


class TimeVaryingMaterial(list):
    """
    Material with time-varying permittivity ``eps_r(t)``, conductivity
    ``sigma(t)``, permeability ``mu_r(t)``, and/or ``TimeVaryingDrude`` poles.

    Subclasses ``list`` (initialized to the ``t = 0`` triple
    ``[eps_r(0), mu_r(0), 0.0]``) so it can be assigned in
    ``GridFIT3D(stl_materials={...})`` or registered on a box/domain via
    ``SolverFIT3D.add_time_varying_material``.

    Parameters
    ----------
    eps_r : float or callable, optional
        Instantaneous relative permittivity ``eps_r(t)`` (> 0). Default is 1.0.
    sigma : float or callable, optional
        Instantaneous electric conductivity ``sigma(t)`` [S/m] (>= 0).
        Default is 0.0.
    poles : sequence of TimeVaryingDrude, Lorentz or Debye, optional
        Electric dispersive poles (static or time-varying).
    mu_r : float or callable, optional
        Instantaneous relative permeability ``mu_r(t)`` (> 0). Default is 1.0.
    sigma_m : float or callable, optional
        Instantaneous magnetic conductivity ``sigma_m(t)`` [Ohm/m] (>= 0).
        Default is 0.0.
    name : str, optional
        Descriptive name.
    """

    def __init__(
        self,
        eps_r=1.0,
        sigma=0.0,
        poles=(),
        mu_r=1.0,
        sigma_m=0.0,
        name=None,
    ):
        self.eps_r = eps_r
        self.sigma = sigma
        self.mu_r = mu_r
        self.sigma_m = sigma_m
        self.name = name

        self.poles = []
        for pole in poles:
            if isinstance(pole, TimeVaryingDrude):
                self.poles.append(pole)
            elif isinstance(pole, Lorentz):
                self.poles.append(
                    TimeVaryingDrude(
                        omega_p=np.sqrt(pole.strength),
                        gamma=pole.gamma,
                        omega_0=pole.omega_0,
                    )
                )
            elif isinstance(pole, Debye):
                self.poles.append(pole)
            else:
                raise TypeError(f"Unsupported pole type {type(pole).__name__}")

        eps0_val = self.eps_at(0.0)
        mu0_val = self.mu_at(0.0)
        self.sigma_at(0.0)
        self.sigma_m_at(0.0)
        super().__init__([float(eps0_val), float(mu0_val), 0.0])

    @property
    def has_electric(self):
        """True if electric properties differ from static vacuum or vary in time."""
        return (
            callable(self.eps_r)
            or callable(self.sigma)
            or float(self.eps_r) != 1.0
            or float(self.sigma) != 0.0
            or bool(self.poles)
        )

    @property
    def has_magnetic(self):
        """True if magnetic properties differ from static vacuum or vary in time."""
        return (
            callable(self.mu_r)
            or callable(self.sigma_m)
            or float(self.mu_r) != 1.0
            or float(self.sigma_m) != 0.0
        )

    def eps_at(self, t):
        """Evaluate relative permittivity ``eps_r(t)`` at time `t` [s]."""
        val = _eval_param(self.eps_r, t)
        if np.any(val <= 0):
            raise ValueError(f"eps_r(t) must be > 0 at t={t}, got {val}.")
        return val

    def sigma_at(self, t):
        """Evaluate electric conductivity ``sigma(t)`` [S/m] at time `t` [s]."""
        val = _eval_param(self.sigma, t)
        if np.any(val < 0):
            raise ValueError(f"sigma(t) must be >= 0 at t={t}, got {val}.")
        return val

    def mu_at(self, t):
        """Evaluate relative permeability ``mu_r(t)`` at time `t` [s]."""
        val = _eval_param(self.mu_r, t)
        if np.any(val <= 0):
            raise ValueError(f"mu_r(t) must be > 0 at t={t}, got {val}.")
        return val

    def sigma_m_at(self, t):
        """Evaluate magnetic conductivity ``sigma_m(t)`` [Ohm/m] at time `t` [s]."""
        val = _eval_param(self.sigma_m, t)
        if np.any(val < 0):
            raise ValueError(f"sigma_m(t) must be >= 0 at t={t}, got {val}.")
        return val

    def is_stable_at(self, t, dt, courant=0.5):
        """
        Check local Courant and ADE pole stability at instant `t` [s].

        Requires ``courant^2 <= eps_r(t) * mu_r(t)`` (wave speed does not exceed
        the grid Courant limit) and, for each ``TimeVaryingDrude`` pole,
        ``(omega_0(t)^2 + omega_p(t)^2 / eps_r(t)) * (dt / 2)^2 + courant^2 / (eps_r(t) * mu_r(t)) <= 1``.
        """
        eps_t = float(self.eps_at(t))
        mu_t = float(self.mu_at(t))
        c2_eff = (courant**2) / (eps_t * mu_t)
        if c2_eff > 1.0 + 1e-9:
            return False
        for pole in self.poles:
            if isinstance(pole, TimeVaryingDrude):
                wp = float(pole.omega_p_at(t))
                w0 = float(pole.omega_0_at(t))
                if (w0 * dt / 2.0) ** 2 + (wp * dt / 2.0) ** 2 / eps_t + c2_eff > 1.0 + 1e-6:
                    return False
        return True

    def __repr__(self):
        label = f"{self.name}: " if self.name else ""
        return (
            f"TimeVaryingMaterial({label}eps_r={self.eps_r!r}, "
            f"sigma={self.sigma!r}, poles={self.poles!r}, mu_r={self.mu_r!r})"
        )


# ---------------------------------------------------------------------------
# Analytical theory helpers for temporal interfaces and photonic time crystals
# ---------------------------------------------------------------------------


def temporal_fresnel_coefficients(eps1, eps2, mu1=1.0, mu2=1.0):
    """
    Analytical temporal Fresnel coefficients at an abrupt switch
    ``(eps1, mu1) -> (eps2, mu2)`` (Morgenthaler 1958 [1]; Mendonca & Shukla
    2002 [2]; Galiffi et al. 2022 [3]).

    Across a spatially homogeneous temporal interface at ``t = t_0``, the
    wavevector ``k`` is conserved while the frequency converts as
    ``omega_2 / omega_1 = n_1 / n_2``. Continuity of ``D`` and ``B`` splits an
    initial forward wave into a forward ("time-refracted", ``+k``) wave and a
    backward ("time-reflected", ``-k``) wave.

    Parameters
    ----------
    eps1, eps2 : float or array_like
        Initial and final relative permittivities (> 0).
    mu1, mu2 : float or array_like, optional
        Initial and final relative permeabilities (> 0). Default is 1.0.

    Returns
    -------
    dict
        Dictionary containing:

        * ``n1``, ``n2``: refractive indices ``sqrt(eps * mu)``.
        * ``eta1``, ``eta2``: normalized wave impedances ``sqrt(mu / eps)``.
        * ``omega_ratio``: frequency conversion factor ``omega_2 / omega_1 = n_1 / n_2``.
        * ``k_ratio``: wavenumber ratio ``k_2 / k_1 = 1.0``.
        * ``tau_D``, ``rho_D``: forward and backward ``D`` amplitude ratios
          (``tau_D + rho_D = 1``).
        * ``tau_B``, ``rho_B``: forward and backward ``B`` amplitude ratios
          (``tau_B + rho_B = 1``).
        * ``tau_E``, ``rho_E``: forward and backward ``E`` amplitude ratios
          (``tau_E + rho_E = eps1 / eps2``).
        * ``tau_H``, ``rho_H``: forward and backward ``H`` amplitude ratios
          (``tau_H + rho_H = mu1 / mu2``).
        * ``energy_forward_ratio``: forward wave energy ratio ``W_+ / W_1``.
        * ``energy_backward_ratio``: backward wave energy ratio ``W_- / W_1``.
        * ``energy_total_ratio``: total post-switch energy ratio
          ``W_2 / W_1 = (W_+ + W_-) / W_1 = 0.5 * (eps1 / eps2 + mu1 / mu2)``.
        * ``action_ratio``: wave-action (photon-number difference) ratio
          ``((W_+ - W_-) / omega_2) / (W_1 / omega_1) = 1.0``.
    """
    eps1 = np.asarray(eps1, dtype=float)
    eps2 = np.asarray(eps2, dtype=float)
    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    if np.any(eps1 <= 0) or np.any(eps2 <= 0) or np.any(mu1 <= 0) or np.any(mu2 <= 0):
        raise ValueError("Permittivities and permeabilities must be > 0.")

    n1 = np.sqrt(eps1 * mu1)
    n2 = np.sqrt(eps2 * mu2)
    eta1 = np.sqrt(mu1 / eps1)
    eta2 = np.sqrt(mu2 / eps2)

    omega_ratio = n1 / n2
    tau_D = 0.5 * (1.0 + eta1 / eta2)
    rho_D = 0.5 * (1.0 - eta1 / eta2)
    tau_B = 0.5 * (1.0 + eta2 / eta1)
    rho_B = 0.5 * (1.0 - eta2 / eta1)

    tau_E = (eps1 / eps2) * tau_D
    rho_E = (eps1 / eps2) * rho_D
    tau_H = (mu1 / mu2) * tau_B
    rho_H = -(mu1 / mu2) * rho_B

    # Energy ratio of each travelling mode: u = eps_2 E^2 = D^2 / eps_2
    # For mu1 = mu2, W_+- / W_1 = (eps1 / eps2) * (mu1 / mu2)... wait:
    # In general: u_+ = 0.5 * (D_+^2 / eps2 + B_+^2 / mu2) = D_+^2 / eps2
    # since B_+ = eta2 D_+ => B_+^2 / mu2 = (mu2 / eps2) D_+^2 / mu2 = D_+^2 / eps2.
    # Thus W_+ / W_1 = (D_+ / D_0)^2 * (eps1 / eps2) = (eps1 / eps2) * tau_D^2!
    # Let's check: tau_D^2 + rho_D^2 = 0.5 * (1 + (eta1/eta2)^2) = 0.5 * (1 + (mu1*eps2)/(eps1*mu2)),
    # so (eps1/eps2)*(tau_D^2 + rho_D^2) = 0.5 * (eps1/eps2 + mu1/mu2)!
    energy_forward = (eps1 / eps2) * (tau_D**2)
    energy_backward = (eps1 / eps2) * (rho_D**2)
    energy_total = energy_forward + energy_backward
    action_ratio = (energy_forward - energy_backward) / omega_ratio

    return {
        "n1": n1,
        "n2": n2,
        "eta1": eta1,
        "eta2": eta2,
        "omega_ratio": omega_ratio,
        "k_ratio": np.ones_like(omega_ratio),
        "tau_D": tau_D,
        "rho_D": rho_D,
        "tau_B": tau_B,
        "rho_B": rho_B,
        "tau_E": tau_E,
        "rho_E": rho_E,
        "tau_H": tau_H,
        "rho_H": rho_H,
        "energy_forward_ratio": energy_forward,
        "energy_backward_ratio": energy_backward,
        "energy_total_ratio": energy_total,
        "energy_ratio": energy_total,
        "action_ratio": action_ratio,
    }


def temporal_slab_coefficients(
    eps1, eps2, delta_t, omega1, eps3=None, mu1=1.0, mu2=1.0, mu3=None
):
    """
    Analytical scattering coefficients of a temporal slab (double temporal
    interface ``(eps1, mu1) -> (eps2, mu2) -> (eps3, mu3)`` of duration
    ``delta_t``; Galiffi et al. 2022 [3], Moussa et al. 2023 [6]).

    Parameters
    ----------
    eps1, eps2 : float
        Initial and slab relative permittivities (> 0).
    delta_t : float or array_like
        Slab duration ``t_2 - t_1`` [s] (>= 0).
    omega1 : float or array_like
        Initial wave angular frequency ``omega_1`` [rad/s].
    eps3 : float, optional
        Final relative permittivity after the slab; defaults to ``eps1``.
    mu1, mu2, mu3 : float, optional
        Relative permeabilities; ``mu3`` defaults to ``mu1``.

    Returns
    -------
    dict
        ``{'omega2': omega2, 'omega3': omega3, 'tau_D': ..., 'rho_D': ...,
        'tau_E': ..., 'rho_E': ..., 'T_D': |tau_D|^2, 'R_D': |rho_D|^2}``.
    """
    if eps3 is None:
        eps3 = eps1
    if mu3 is None:
        mu3 = mu1
    delta_t = np.asarray(delta_t, dtype=float)
    omega1 = np.asarray(omega1, dtype=float)

    s12 = temporal_fresnel_coefficients(eps1, eps2, mu1, mu2)
    s23 = temporal_fresnel_coefficients(eps2, eps3, mu2, mu3)

    omega2 = omega1 * s12["omega_ratio"]
    omega3 = omega2 * s23["omega_ratio"]
    phi = omega2 * delta_t

    # Forward (+k) wave accumulates exp(-i phi) and backward (-k) wave exp(+i phi)
    # Matching at t1 and t2:
    # [D_3+; D_3-] = [[tau23, rho23], [rho23, tau23]] [[e^(-i phi), 0], [0, e^(+i phi)]] [[tau12], [rho12]]
    exp_m = np.exp(-1j * phi)
    exp_p = np.exp(1j * phi)
    tau_D = s23["tau_D"] * s12["tau_D"] * exp_m + s23["rho_D"] * s12["rho_D"] * exp_p
    rho_D = s23["rho_D"] * s12["tau_D"] * exp_m + s23["tau_D"] * s12["rho_D"] * exp_p

    scale_e = float(eps1) / float(eps3)
    tau_E = scale_e * tau_D
    rho_E = scale_e * rho_D

    return {
        "omega2": omega2,
        "omega3": omega3,
        "phase": phi,
        "tau_D": tau_D,
        "rho_D": rho_D,
        "tau_E": tau_E,
        "rho_E": rho_E,
        "T_D": np.abs(tau_D) ** 2,
        "R_D": np.abs(rho_D) ** 2,
        "T_E": np.abs(tau_E) ** 2,
        "R_E": np.abs(rho_E) ** 2,
    }


def ptc_mathieu_gap_analytic(eps_avg, modulation_depth, omega_mod, k=None):
    """
    Analytical weak-modulation (Mathieu / two-wave coupled-mode) theory for a
    Photonic Time Crystal ``eps_r(t) = eps_avg * (1 + m * cos(Omega * t))``
    around the fundamental momentum gap at ``omega_0(k) = Omega / 2``
    (Lyubarov et al. 2022 [4]; Galiffi et al. 2022 [3]).

    Parameters
    ----------
    eps_avg : float
        Average relative permittivity ``bar{eps}_r > 0``.
    modulation_depth : float
        Modulation depth ``m`` in ``[0, 1)``.
    omega_mod : float
        Modulation angular frequency ``Omega`` [rad/s].
    k : float or array_like, optional
        Wavenumber(s) [rad/m] at which to evaluate the coupled-mode dispersion
        ``omega_re(k)`` and growth rate ``gamma(k)``.

    Returns
    -------
    dict
        Dictionary with ``k_center``, ``k_center_2nd`` (including the
        ``sqrt(1 - m^2)`` harmonic-mean shift), ``omega_gap`` (``Omega / 2``),
        ``delta_k``, ``k_edges``, ``gamma_max`` (``m * Omega / 8``), and
        (if ``k`` is provided) ``omega_re`` and ``gamma``.
    """
    eps_avg = float(eps_avg)
    m = abs(float(modulation_depth))
    omega_mod = float(omega_mod)
    if eps_avg <= 0 or m >= 1.0 or omega_mod <= 0:
        raise ValueError("Require eps_avg > 0, 0 <= |m| < 1, and omega_mod > 0.")

    v_avg = c_light / np.sqrt(eps_avg)
    omega_gap = 0.5 * omega_mod
    k_center = omega_gap / v_avg
    # Mean of 1 / (1 + m cos(Omega t)) is 1 / sqrt(1 - m^2), shifting the gap
    # center to omega_0(k) = (Omega / 2) * (1 - m^2)^(1/4)
    k_center_2nd = k_center * ((1.0 - m**2) ** 0.25)
    gamma_max = 0.125 * m * omega_mod
    delta_k = 2.0 * gamma_max / v_avg

    out = {
        "k_center": float(k_center),
        "k_center_2nd": float(k_center_2nd),
        "omega_gap": float(omega_gap),
        "delta_k": float(delta_k),
        "k_edges": (float(k_center_2nd - 0.5 * delta_k), float(k_center_2nd + 0.5 * delta_k)),
        "gamma_max": float(gamma_max),
    }
    if k is not None:
        k_arr = np.asarray(k, dtype=float)
        # Effective unperturbed frequency including the 0th Fourier harmonic of 1/eps(t)
        w0_eff = v_avg * k_arr / ((1.0 - m**2) ** 0.25) if m > 0 else v_avg * k_arr
        detuning = w0_eff - omega_gap
        inside = np.abs(detuning) < gamma_max
        gamma_k = np.where(
            inside, np.sqrt(np.maximum(0.0, gamma_max**2 - detuning**2)), 0.0
        )
        sign = np.where(detuning >= 0.0, 1.0, -1.0)
        omega_re = np.where(
            inside,
            omega_gap,
            omega_gap + sign * np.sqrt(np.maximum(0.0, detuning**2 - gamma_max**2)),
        )
        out["omega_re"] = omega_re
        out["gamma"] = gamma_k
    return out


def ptc_floquet_bandstructure(
    k,
    eps_avg,
    modulation_depth,
    omega_mod,
    dt=None,
    dx=None,
    n_steps_per_period=400,
):
    """
    Exact Floquet band structure ``(Re omega_F(k), Im omega_F(k))`` of a
    Photonic Time Crystal ``eps_r(t) = eps_avg * (1 + m * cos(Omega * t))``
    by symplectic monodromy integration of Hill's equation::

        d^2 D_k / dt^2 + (c^2 k^2 / eps_r(t)) D_k = 0

    over one modulation period ``T_mod = 2 * pi / Omega``.

    When ``dt`` (and optionally ``dx``) are supplied, the exact discrete
    monodromy matrix of the FIT leapfrog scheme is integrated with the grid
    wavenumber ``K = (2 / dx) * sin(k * dx / 2)``.

    Parameters
    ----------
    k : float or array_like
        Spatial wavenumber(s) [rad/m].
    eps_avg : float
        Average relative permittivity ``bar{eps}_r > 0``.
    modulation_depth : float
        Modulation depth ``m`` in ``(-1, 1)``.
    omega_mod : float
        Modulation angular frequency ``Omega`` [rad/s].
    dt : float, optional
        FIT timestep [s]. If None, uses ``T_mod / n_steps_per_period``.
    dx : float, optional
        Grid cell size [m] for the discrete spatial operator.
    n_steps_per_period : int, optional
        Number of symplectic steps per modulation period when ``dt`` is None.
        Default is 400.

    Returns
    -------
    dict
        ``{'k': k_arr, 'omega_re': ..., 'gamma': ..., 'trace_half': ...,
        'multiplier_abs': ..., 'in_gap': ...}`` where ``omega_re`` [rad/s] is
        the real Floquet frequency (unwrapped to the positive branch) and
        ``gamma = max Im(omega_F) >= 0`` [rad/s] is the exponential growth rate.
    """
    k_arr = np.atleast_1d(np.asarray(k, dtype=float))
    eps_avg = float(eps_avg)
    m = float(modulation_depth)
    omega_mod = float(omega_mod)
    if eps_avg <= 0 or abs(m) >= 1.0 or omega_mod <= 0:
        raise ValueError("Require eps_avg > 0, |m| < 1, and omega_mod > 0.")

    t_mod = 2.0 * np.pi / omega_mod
    if dt is None:
        n_steps = int(n_steps_per_period)
        h = t_mod / n_steps
    else:
        n_steps = max(1, int(round(t_mod / float(dt))))
        h = t_mod / n_steps

    if dx is not None and float(dx) > 0:
        k_eff = (2.0 / float(dx)) * np.sin(0.5 * k_arr * float(dx))
    else:
        k_eff = k_arr
    c2_k2 = (c_light * k_eff) ** 2

    # Symplectic leapfrog monodromy matrix over one period t in [0, T_mod):
    # p^(n+1/2) = p^(n-1/2) - h * (c^2 K^2 / eps_r(t^n)) * D^n
    # D^(n+1)   = D^n       + h * p^(n+1/2)
    # State vector [D^n; p^(n-1/2)] has transfer matrix
    # [[1 - h^2 w^2(t^n), h], [-h w^2(t^n), 1]] of determinant 1.
    m11 = np.ones_like(k_arr)
    m12 = np.zeros_like(k_arr)
    m21 = np.zeros_like(k_arr)
    m22 = np.ones_like(k_arr)

    for step_idx in range(n_steps):
        t_n = step_idx * h
        eps_n = eps_avg * (1.0 + m * np.cos(omega_mod * t_n))
        w2 = c2_k2 / eps_n
        a11 = 1.0 - (h**2) * w2
        a21 = -h * w2
        # Multiply A_n @ M
        new_m11 = a11 * m11 + h * m21
        new_m12 = a11 * m12 + h * m22
        new_m21 = a21 * m11 + m21
        new_m22 = a21 * m12 + m22
        m11, m12, m21, m22 = new_m11, new_m12, new_m21, new_m22

    half_tr = 0.5 * (m11 + m22)
    in_gap = np.abs(half_tr) > 1.0 + 1e-12

    # Eigenvalues of 2x2 unimodular matrix: lambda = half_tr +- sqrt(half_tr^2 - 1)
    disc_pos = np.maximum(0.0, half_tr**2 - 1.0)
    lam_max = np.where(
        in_gap,
        np.abs(half_tr) + np.sqrt(disc_pos),
        1.0,
    )
    gamma = np.log(lam_max) / t_mod

    # Real part of Floquet frequency:
    # In the first gap (half_tr <= -1), the phase per period is pi => omega_re = Omega / 2.
    # Outside the gap, cos(omega_re * T_mod) = half_tr; unwrap continuously around
    # the unmodulated dispersion omega_0(k) = c k / sqrt(eps_avg * sqrt(1 - m^2)).
    cos_arg = np.clip(half_tr, -1.0, 1.0)
    theta_principal = np.arccos(cos_arg)  # in [0, pi]
    w0_approx = c_light * k_eff / np.sqrt(eps_avg * np.sqrt(1.0 - m**2))
    theta_approx = w0_approx * t_mod
    # Pick branch 2*pi*m +- theta_principal closest to theta_approx
    branch_m = np.round(theta_approx / (2.0 * np.pi))
    cand_plus = 2.0 * np.pi * branch_m + theta_principal
    cand_minus = 2.0 * np.pi * branch_m - theta_principal
    theta_unwrapped = np.where(
        np.abs(cand_plus - theta_approx) <= np.abs(cand_minus - theta_approx),
        cand_plus,
        cand_minus,
    )
    omega_re = theta_unwrapped / t_mod

    return {
        "k": k_arr,
        "omega_re": omega_re,
        "omega_real": omega_re,
        "gamma": gamma,
        "trace_half": half_tr,
        "multiplier_abs": lam_max,
        "in_gap": in_gap,
    }


def plasma_ionization_fresnel(omega1, omega_p1, omega_p2, eps_inf=1.0):
    """
    Analytical frequency conversion and mode amplitudes across an abrupt
    plasma density jump ("flash ionization", ``omega_p1 -> omega_p2`` at
    ``t = t_0``) in a collisionless Drude plasma (Wilks, Dawson & Mori 1988
    [5]; Mendonca 2001 [2]).

    Because newly ionized electrons are born with zero initial velocity at
    ``t = t_0``, the electric field ``E``, magnetic field ``H``, and plasma
    current ``J_p`` are all continuous at ``t = t_0``, while ``dJ_p/dt`` jumps.
    A forward wave of frequency ``omega_1`` (wavenumber
    ``c k = sqrt(eps_inf * omega_1^2 - omega_p1^2)``) splits into:

    1. A forward upshifted wave at ``+omega_2`` with ``E_+ / E_0 = 0.5 * (1 + omega_1 / omega_2)``,
    2. A backward upshifted wave at ``-omega_2`` with ``E_- / E_0 = 0.5 * (1 - omega_1 / omega_2)``,
    3. A spatially periodic, zero-frequency (``omega_3 = 0``) static magnetic
       field and DC plasma current mode with
       ``H_dc / H_0 = (omega_p2^2 - omega_p1^2) / (eps_inf * omega_2^2)``.

    Parameters
    ----------
    omega1 : float or array_like
        Initial wave angular frequency [rad/s] (> ``omega_p1 / sqrt(eps_inf)``).
    omega_p1, omega_p2 : float or array_like
        Initial and final plasma angular frequencies [rad/s] (>= 0).
    eps_inf : float, optional
        High-frequency background permittivity. Default is 1.0.

    Returns
    -------
    dict
        ``{'omega1': ..., 'omega2': ..., 'omega_ratio': ..., 'tau_E': ...,
        'rho_E': ..., 'tau_H': ..., 'rho_H': ..., 'H_dc_ratio': ...}``.
    """
    omega1 = np.asarray(omega1, dtype=float)
    omega_p1 = np.asarray(omega_p1, dtype=float)
    omega_p2 = np.asarray(omega_p2, dtype=float)
    eps_inf = float(eps_inf)
    if np.any(eps_inf * omega1**2 <= omega_p1**2):
        raise ValueError("Initial wave must be propagating: eps_inf * omega1^2 > omega_p1^2.")

    omega2 = np.sqrt(omega1**2 + (omega_p2**2 - omega_p1**2) / eps_inf)
    ratio = omega2 / omega1
    tau_E = 0.5 * (1.0 + omega1 / omega2)
    rho_E = 0.5 * (1.0 - omega1 / omega2)
    tau_H = 0.5 * (omega1 / omega2) * (1.0 + omega1 / omega2)
    rho_H = -0.5 * (omega1 / omega2) * (1.0 - omega1 / omega2)
    h_dc_ratio = (omega_p2**2 - omega_p1**2) / (eps_inf * omega2**2)

    return {
        "omega1": omega1,
        "omega_1": omega1,
        "omega2": omega2,
        "omega_2": omega2,
        "omega_ratio": ratio,
        "tau_E": tau_E,
        "rho_E": rho_E,
        "tau_H": tau_H,
        "rho_H": rho_H,
        "H_dc_ratio": h_dc_ratio,
    }


# ---------------------------------------------------------------------------
# SolverFIT3D Mixin
# ---------------------------------------------------------------------------


class TimeVaryingMixin:
    """
    `SolverFIT3D` methods for time-varying materials (`eps_r(t)`, `sigma(t)`,
    `mu_r(t)`, and `TimeVaryingDrude` poles).

    Integrates Ampere's law in conservative flux form so that the electric
    displacement `D^n = eps_0 eps_eff(t^n) E^n` is continuous across abrupt
    temporal interfaces.
    """

    def _check_time_varying_supported(self):
        if getattr(self, "use_gpu", False):
            raise NotImplementedError(
                "Time-varying materials are currently supported on CPU only "
                "(use_gpu=False)."
            )
        if getattr(self, "use_mpi", False):
            raise NotImplementedError(
                "Time-varying materials are currently supported in serial CPU "
                "mode only (use_mpi=False)."
            )
        if getattr(self, "scheme", "yee") == "tete":
            raise NotImplementedError(
                "scheme='tete' does not support time-varying materials."
            )
        if getattr(self, "fused", False):
            raise NotImplementedError(
                "fused=True does not support time-varying materials."
            )

    def _check_time_varying_stable(self, material, t=None):
        if not isinstance(material, TimeVaryingMaterial):
            raise TypeError("material must be a TimeVaryingMaterial")
        dt = float(self.dt)
        courant = (
            c_light
            * dt
            * np.sqrt(
                np.sum([1.0 / np.min(d) ** 2 for d in (self.dx, self.dy, self.dz)])
            )
        )
        probe_times = [0.0] if t is None else [float(t)]
        if t is None:
            for param in (
                material.eps_r,
                material.mu_r,
                *(p.omega_p for p in material.poles if isinstance(p, TimeVaryingDrude)),
            ):
                if isinstance(param, StepProfile):
                    probe_times.extend([param.t_switch - dt, param.t_switch + dt])
                elif isinstance(param, SlabTimeProfile):
                    probe_times.extend(
                        [
                            param.t_start - dt,
                            0.5 * (param.t_start + param.t_end),
                            param.t_end + dt,
                        ]
                    )
                elif isinstance(param, CosineModulation) and param.omega_mod > 0:
                    t_min = param.t_start + (np.pi - param.phase) / param.omega_mod
                    t_max = param.t_start - param.phase / param.omega_mod
                    probe_times.extend([t_min, t_max])
        for tp in probe_times:
            if not material.is_stable_at(max(0.0, float(tp)), dt, courant=courant):
                raise ValueError(
                    f"dt={dt:.3e} s (vacuum Courant={courant:.3f}) violates the "
                    f"Courant stability limit for {material!r} at t={tp:.3e} s; "
                    "pass a smaller `dt` or `cfln` to SolverFIT3D."
                )

    def add_time_varying_material(self, material, center=None, size=None):
        """
        Register a `TimeVaryingMaterial` on the entire domain (`center=None`
        and `size=None`) or inside a box `(center, size)`.

        Parameters
        ----------
        material : TimeVaryingMaterial
            Time-varying material model.
        center, size : tuple of 3 float, optional
            Centre and extent of the box `(x, y, z)` [m]. If both are `None`,
            all non-PEC edges in the domain are assigned the material with unit
            weight.
        """
        if not isinstance(material, TimeVaryingMaterial):
            raise TypeError("material must be a TimeVaryingMaterial")
        self._check_time_varying_supported()
        self._check_time_varying_stable(material)

        if material.has_electric or not material.has_magnetic:
            if center is None and size is None:
                idx_e = np.arange(3 * self.N, dtype=int)
                w_e = np.ones(3 * self.N, dtype=float)
            else:
                if center is None:
                    center = (0.0, 0.0, 0.0)
                if size is None:
                    size = (0.0, 0.0, 0.0)
                idx_e, w_e = self._box_edges(center, size, dual=False)
            self._add_time_varying_edges(material, idx_e, w_e)

        if material.has_magnetic:
            if center is None and size is None:
                idx_h = np.arange(3 * self.N, dtype=int)
                w_h = np.ones(3 * self.N, dtype=float)
            else:
                if center is None:
                    center = (0.0, 0.0, 0.0)
                if size is None:
                    size = (0.0, 0.0, 0.0)
                idx_h, w_h = self._box_edges(center, size, dual=True)
            self._add_time_varying_h_edges(material, idx_h, w_h)

    def _apply_stl_time_varying(self):
        """Attach time-varying updates to STL solids with a `TimeVaryingMaterial`."""
        for key, material in self.grid.stl_materials.items():
            if not isinstance(material, TimeVaryingMaterial):
                continue
            self._check_time_varying_supported()
            mask = np.reshape(self.grid.grid[key], (self.Nx, self.Ny, self.Nz))
            fraction = np.reshape(mask, self.N, order="F").astype(float)
            fraction = np.tile(fraction, 3)
            idx = np.flatnonzero(fraction)
            if material.has_electric or not material.has_magnetic:
                self._add_time_varying_edges(
                    material, idx, fraction[idx], eps_bg=1.0
                )
            if material.has_magnetic:
                self._add_time_varying_h_edges(
                    material, idx, fraction[idx], mu_bg=1.0
                )

    def _install_time_varying_stepper(self):
        """Wrap `self.one_step` once with `_one_step_time_varying`."""
        if hasattr(self, "_time_varying_installed") and self._time_varying_installed:
            return
        self._time_varying_installed = True
        self._tv_step = 0
        if not hasattr(self, "_time_varying_groups"):
            self._time_varying_groups = []
        if not hasattr(self, "_time_varying_h_groups"):
            self._time_varying_h_groups = []

        if (
            getattr(self, "activate_abc", False)
            and self.one_step == self._one_step_with_abc
        ):
            self._one_step_nontv = self._one_step_backend
            self._one_step_backend = self._one_step_time_varying
        else:
            self._one_step_nontv = self.one_step
            self.one_step = self._one_step_time_varying

    def _add_time_varying_edges(self, material, idx, weight, eps_bg=None):
        """Register flat E indices `idx` with fill fractions `weight`."""
        self._check_time_varying_supported()
        self._check_time_varying_stable(material)
        net = getattr(self, "dispersive_network", None)
        if net is not None and np.any(np.isin(idx, net.rows)):
            raise NotImplementedError(
                "Time-varying regions must not overlap conformal dispersive solids."
            )
        ieps = np.asarray(self.ieps.array)
        keep = (weight > 0) & (ieps[idx] != 0)
        idx, weight = idx[keep], weight[keep]
        self._install_time_varying_stepper()

        for group in self._time_varying_groups:
            old = ~np.isin(group["idx_host"], idx)
            if not old.all():
                _restrict_tv_group(group, old)

        eps_s = 1.0 / (eps_0 * ieps[idx])
        bg = eps_s if eps_bg is None else np.full_like(eps_s, float(eps_bg))
        w_arr = np.asarray(weight, dtype=self.dtype)
        eps0_val = material.eps_at(0.0)
        eps_eff_0 = (1.0 - w_arr) * bg + w_arr * eps0_val

        drude_poles = []
        debye_poles = []
        zeros = np.zeros(idx.size, dtype=self.dtype)
        for pole in material.poles:
            if isinstance(pole, TimeVaryingDrude):
                drude_poles.append([pole, zeros.copy(), zeros.copy()])
            elif isinstance(pole, Debye):
                r = float(self.dt) / (2.0 * pole.tau)
                kappa = (1.0 - r) / (1.0 + r)
                beta = pole.delta_eps * r / (1.0 + r)
                debye_poles.append([kappa, beta, zeros.copy()])

        group = {
            "material": material,
            "idx_host": idx,
            "idx": idx,
            "weight": w_arr,
            "eps_s": np.asarray(eps_s, dtype=self.dtype),
            "bg": np.asarray(bg, dtype=self.dtype),
            "eps_eff_current": np.asarray(eps_eff_0, dtype=self.dtype),
            "drude": drude_poles,
            "debye": debye_poles,
            "D_before_switch": zeros.copy(),
            "D_after_switch": zeros.copy(),
        }
        self._time_varying_groups = [
            g for g in self._time_varying_groups if g["idx_host"].size
        ] + [group]

    def _add_time_varying_h_edges(self, material, idx, weight, mu_bg=None):
        """Register flat H indices `idx` with fill fractions `weight`."""
        self._check_time_varying_supported()
        self._check_time_varying_stable(material)
        imu = np.asarray(self.imu.array)
        keep = (weight > 0) & (imu[idx] != 0)
        idx, weight = idx[keep], weight[keep]
        self._install_time_varying_stepper()

        for group in self._time_varying_h_groups:
            old = ~np.isin(group["idx_host"], idx)
            if not old.all():
                _restrict_tv_group(group, old)

        mu_s = 1.0 / (mu_0 * imu[idx])
        bg = mu_s if mu_bg is None else np.full_like(mu_s, float(mu_bg))
        w_arr = np.asarray(weight, dtype=self.dtype)
        mu0_val = material.mu_at(0.0)
        mu_eff_0 = (1.0 - w_arr) * bg + w_arr * mu0_val
        zeros = np.zeros(idx.size, dtype=self.dtype)
        group = {
            "material": material,
            "idx_host": idx,
            "idx": idx,
            "weight": w_arr,
            "mu_s": np.asarray(mu_s, dtype=self.dtype),
            "bg": np.asarray(bg, dtype=self.dtype),
            "mu_eff_current": np.asarray(mu_eff_0, dtype=self.dtype),
            "B_before_switch": zeros.copy(),
            "B_after_switch": zeros.copy(),
        }
        self._time_varying_h_groups = [
            g for g in self._time_varying_h_groups if g["idx_host"].size
        ] + [group]

    def _one_step_time_varying(self):
        """
        Advance one FIT step and apply the D/B-conserving time-varying material
        update.

        Time-level centering at step `n = self._tv_step`:
        * `E^n` and `D^n = eps_0 eps_eff(t^n) E^n` are at `t^n = n * dt`.
        * `H^(n-1/2)` is at `t^(n-1/2) = (n - 0.5) * dt`, updated to
          `t^(n+1/2) = (n + 0.5) * dt` using `E^n`.
        * `E^(n+1)` and `D^(n+1) = eps_0 eps_eff(t^(n+1)) E^(n+1)` are at
          `t^(n+1) = (n + 1) * dt`, with conductivity `sigma(t^(n+1/2))`
          centered at `t^(n+1/2)`.
        """
        dt = float(self.dt)
        n = self._tv_step
        t_n = n * dt
        t_half = (n + 0.5) * dt
        t_np1 = (n + 1) * dt
        t_mhalf = (n - 0.5) * dt

        e_groups = getattr(self, "_time_varying_groups", ())
        h_groups = getattr(self, "_time_varying_h_groups", ())

        E_n = [self.E.array[g["idx"]].copy() for g in e_groups]
        H_n = [self.H.array[g["idx"]].copy() for g in h_groups]

        self._one_step_nontv()

        # If any magnetic group has time-varying mu_r(t) or sigma_m(t):
        if h_groups:
            H = self.H.array
            for g, Hn in zip(h_groups, H_n):
                mat = g["material"]
                w = g["weight"]
                mu_n = (1.0 - w) * g["bg"] + w * mat.mu_at(t_mhalf)
                mu_np1 = (1.0 - w) * g["bg"] + w * mat.mu_at(t_half)
                sm_hat = w * mat.sigma_m_at(t_n) * dt / (2.0 * mu_0)
                delta_b_curl = (H[g["idx"]] - Hn) * g["mu_s"]
                b_np1_minus = mu_n * Hn + delta_b_curl
                H_new = (b_np1_minus - sm_hat * Hn) / (mu_np1 + sm_hat)
                g["B_before_switch"] = mu_0 * b_np1_minus
                g["B_after_switch"] = mu_0 * mu_np1 * H_new
                g["mu_eff_current"] = mu_np1
                H[g["idx"]] = H_new

        if e_groups:
            E = self.E.array
            for g, En in zip(e_groups, E_n):
                mat = g["material"]
                w = g["weight"]
                eps_n = (1.0 - w) * g["bg"] + w * mat.eps_at(t_n)
                eps_np1 = (1.0 - w) * g["bg"] + w * mat.eps_at(t_np1)
                s_hat = w * mat.sigma_at(t_half) * dt / (2.0 * eps_0)
                beta_sum = w * sum(beta for _, beta, _ in g["debye"])

                # Exact curl + source + CPML/TFSF increment over [t^n, t^(n+1)]:
                # (E* - E^n) * eps_s = (dt / eps_0) * (curl H - J_src + CPML)
                delta_d_curl = (E[g["idx"]] - En) * g["eps_s"]
                d_np1_minus = eps_n * En + delta_d_curl
                acc = d_np1_minus - (s_hat + beta_sum) * En

                for k_idx, (pole, p, p_old) in enumerate(g["drude"]):
                    alpha, xi, zeta = pole.ade_coefficients(t_n, dt)
                    p_new = alpha * p + xi * p_old + zeta * En
                    acc = acc - w * (p_new - p)
                    g["drude"][k_idx][1] = p_new
                    g["drude"][k_idx][2] = p

                for kappa, _, p_deb in g["debye"]:
                    acc = acc - w * (kappa - 1.0) * p_deb

                den = eps_np1 + s_hat + beta_sum
                E_new = acc / den

                for k_idx, (kappa, beta, p_deb) in enumerate(g["debye"]):
                    g["debye"][k_idx][2] = kappa * p_deb + beta * (E_new + En)

                g["D_before_switch"] = eps_0 * d_np1_minus
                g["D_after_switch"] = eps_0 * eps_np1 * E_new
                g["eps_eff_current"] = eps_np1
                E[g["idx"]] = E_new

        self._tv_step = n + 1
        self.nstep = self._tv_step

    def get_D(self, t=None):
        """
        Return the electric displacement field ``D = eps_0 eps_eff(t) E`` [C/m^2]
        as a `wakis.Field`.

        Parameters
        ----------
        t : float, optional
            Time [s] at which to evaluate ``eps_r(t)``. If None (default), uses
            the permittivity synchronized with the current integer time level
            ``t^n = _tv_step * dt`` of ``self.E``.
        """
        ieps = np.asarray(self.ieps.toarray())
        eps = np.where(ieps != 0, 1.0 / np.where(ieps != 0, ieps, 1.0), 0.0)
        for g in getattr(self, "_time_varying_groups", ()):
            if t is None:
                eps_eff = g["eps_eff_current"]
            else:
                w = g["weight"]
                eps_eff = (1.0 - w) * g["bg"] + w * g["material"].eps_at(float(t))
            eps[g["idx"]] = eps_0 * eps_eff
        D = Field(self.Nx, self.Ny, self.Nz, use_gpu=False, dtype=self.dtype)
        D.fromarray(eps * self.E.toarray())
        return D

    def get_B(self, t=None):
        """
        Return the magnetic flux density ``B = mu_0 mu_eff(t) H`` [T] as a
        `wakis.Field`.
        """
        imu = np.asarray(self.imu.toarray())
        mu = np.where(imu != 0, 1.0 / np.where(imu != 0, imu, 1.0), 0.0)
        for g in getattr(self, "_time_varying_h_groups", ()):
            if t is None:
                mu_eff = g["mu_eff_current"]
            else:
                w = g["weight"]
                mu_eff = (1.0 - w) * g["bg"] + w * g["material"].mu_at(float(t))
            mu[g["idx"]] = mu_0 * mu_eff
        B = Field(self.Nx, self.Ny, self.Nz, use_gpu=False, dtype=self.dtype)
        B.fromarray(mu * self.H.toarray())
        return B

    def reset_time_varying_state(self):
        """Zero auxiliary polarizations and reset the time-varying step counter."""
        self._tv_step = 0
        for g in getattr(self, "_time_varying_groups", ()):
            w = g["weight"]
            g["eps_eff_current"] = (1.0 - w) * g["bg"] + w * g["material"].eps_at(0.0)
            g["D_before_switch"].fill(0.0)
            g["D_after_switch"].fill(0.0)
            for pole_entry in g["drude"]:
                pole_entry[1].fill(0.0)
                pole_entry[2].fill(0.0)
            for deb_entry in g["debye"]:
                deb_entry[2].fill(0.0)
        for g in getattr(self, "_time_varying_h_groups", ()):
            w = g["weight"]
            g["mu_eff_current"] = (1.0 - w) * g["bg"] + w * g["material"].mu_at(0.0)
            g["B_before_switch"].fill(0.0)
            g["B_after_switch"].fill(0.0)


def _restrict_tv_group(group, keep):
    """Drop the edges of a time-varying group where `keep` is False."""
    group["idx_host"] = group["idx_host"][keep]
    for key in (
        "idx",
        "weight",
        "eps_s",
        "mu_s",
        "bg",
        "eps_eff_current",
        "mu_eff_current",
        "D_before_switch",
        "D_after_switch",
        "B_before_switch",
        "B_after_switch",
    ):
        if key in group:
            group[key] = group[key][keep]
    if "drude" in group:
        group["drude"] = [
            [pole, p[keep], p_old[keep]] for pole, p, p_old in group["drude"]
        ]
    if "debye" in group:
        group["debye"] = [
            [k, b, p_deb[keep]] for k, b, p_deb in group["debye"]
        ]
