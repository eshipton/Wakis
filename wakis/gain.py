# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Active gain media and laser rate-equation models for the FIT time-domain solver.

Provides two time-domain auxiliary differential equation (ADE) models coupled to
`SolverFIT3D`:

1. `FourLevelGain`: Semiclassical four-level two-electron atomic system of
   S.-H. Chang and A. Taflove, *Opt. Express* **12**(16), 3827-3833 (2004),
   doi:10.1364/OPEX.12.003827. Macroscopic polarizations ``P_a`` (lasing
   transition 1 <-> 2 at angular frequency ``omega_a``) and ``P_b`` (optional
   pump transition 0 <-> 3 at angular frequency ``omega_b``) obey Lorentz
   oscillator equations driven by the instantaneous population differences
   ``(N_1 - N_2)`` and ``(N_0 - N_3)``::

       d^2 P_a / dt^2 + gamma_a dP_a / dt + omega_a^2 P_a = kappa_a (N_1 - N_2) E
       d^2 P_b / dt^2 + gamma_b dP_b / dt + omega_b^2 P_b = kappa_b (N_0 - N_3) E

   with classical oscillator coupling ``kappa_a = 6 pi eps_0 c^3 / (omega_a^2 tau_21)``
   and ``kappa_b = 6 pi eps_0 c^3 / (omega_b^2 tau_30)``. The level population
   densities ``N_0, N_1, N_2, N_3`` [1/m^3] evolve according to coupled rate
   equations including external pumping ``W_p`` (0 -> 3), nonradiative/spontaneous
   decays ``tau_32, tau_30, tau_21, tau_10``, field-driven stimulated/absorption
   transitions ``(1 / (hbar omega)) E . dP/dt``, and optional Pauli exclusion
   blocking factors ``(1 - N_j / N_site)``.

2. `SaturableGain`: Teaching saturable-gain Lorentz polarization model with
   homogeneous intensity saturation ``g(I) = g0 / (1 + I / I_sat)``
   (A. E. Siegman, *Lasers*, University Science Books, 1986, Chs. 2, 7, 13)::

       d^2 P / dt^2 + gamma dP / dt + omega_0^2 P
           = eps_0 (delta_eps_0 omega_0^2 / (1 + I / I_sat)) E

   where ``delta_eps_0 = -c sqrt(eps_inf) gamma g0 / omega_0^2`` and ``I`` [W/m^2]
   is the local optical intensity.

Conventions and Discretization
------------------------------
* **Units**: All quantities are in SI units (`omega` in [rad/s], `gamma` in
  [rad/s], `tau` in [s], `N_i` in [1/m^3], `W_p` in [1/s], `g0` in [1/m],
  `I_sat` in [W/m^2]).
* **Sign convention**: Wakis transforms time signals with ``exp(-i w t)``
  (phasors ``Re[X exp(+i w t)]``). Passive absorption corresponds to
  ``Im eps(w) < 0`` (normal population ``N_1 > N_2``, ``delta_eps > 0``), and
  active amplification (gain) corresponds to ``Im eps(w) > 0`` (population
  inversion ``N_2 > N_1``, ``delta_eps < 0``).
* **Time-level centering and second-order accuracy**:
  - ``H`` is stored at half-integer times ``t^(n-1/2)`` and advanced to
    ``t^(n+1/2)`` by Faraday's law centred at ``t^n``.
  - ``E^n``, polarizations ``P_a^n, P_b^n``, and populations ``N_i^n`` (or
    intensity ``I^n``) are collocated at integer times ``t^n = n dt``.
  - Within step ``n``:
    1. ``P^(n+1)`` is advanced from ``P^n, P^(n-1)`` using second-order central
       differences centred at ``t^n`` driven by ``N^n`` and ``E^n``.
    2. ``E^(n+1)`` is advanced from ``E^n`` using Ampère's law centred at the
       half-step ``t^(n+1/2)`` with ``(P^(n+1) - P^n) / dt``.
    3. ``N^(n+1)`` (or ``I^(n+1)``) is advanced from ``N^n`` using the
       trapezoidal (Crank-Nicolson) rule centred at the half-step ``t^(n+1/2)``,
       with the half-step electric field ``E^(n+1/2) = (E^(n+1) + E^n) / 2``
       and half-step polarization current ``dP/dt|^(n+1/2) = (P^(n+1) - P^n) / dt``.
       The scheme is second-order accurate in time, ``O(dt^2)``, and conserves
       total population ``N_0 + N_1 + N_2 + N_3 = N_total`` to machine precision
       at every step.
"""

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import hbar
from scipy.optimize import brentq
from scipy.special import lambertw

from .dispersive import DispersiveMaterial, Lorentz, _restrict_group


class GainMaterial(list):
    """
    Base class for active gain materials in `SolverFIT3D`.

    Like `DispersiveMaterial`, a `GainMaterial` instance is also a 3-element
    list ``[eps_inf, 1.0, 0.0]`` so it can be passed wherever Wakis expects a
    static material triple ``[eps_r, mu_r, sigma]``.
    """

    is_gain = True
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
    def n_bg(self):
        """Background refractive index ``sqrt(eps_inf)`` [-]."""
        return float(np.sqrt(self.eps_inf))


class FourLevelGain(GainMaterial):
    """
    Four-level two-electron atomic gain medium coupled to 3D FIT.

    Implements the semiclassical Maxwell-rate-equation model of S.-H. Chang and
    A. Taflove, *Opt. Express* **12**(16), 3827-3833 (2004),
    doi:10.1364/OPEX.12.003827.

    Parameters
    ----------
    omega_a : float
        Resonance angular frequency [rad/s] of the 1 <-> 2 lasing transition,
        ``> 0``.
    gamma_a : float
        Homogeneous damping (dephasing) rate [rad/s] of the 1 <-> 2 transition,
        ``>= 0``.
    tau_21 : float
        Decay lifetime [s] from level 2 to level 1, ``> 0``.
    tau_10 : float
        Decay lifetime [s] from level 1 to level 0, ``> 0`` (use ``np.inf`` to
        disable 1 -> 0 relaxation).
    tau_32 : float
        Decay lifetime [s] from level 3 to level 2, ``> 0``.
    N_total : float, optional
        Total atomic population density ``N_0 + N_1 + N_2 + N_3`` [1/m^3],
        ``>= 0``. Alias ``N_density`` is also accepted. Default ``1.0e24``.
    pump_rate : float, optional
        External pumping transition rate ``W_p`` [1/s] from level 0 to level 3,
        ``>= 0``. Alias ``W_p`` is also accepted. Default ``0.0``.
    eps_inf : float, optional
        Host high-frequency relative permittivity [-], ``> 0``. Default ``1.0``.
    tau_30 : float, optional
        Decay lifetime [s] from level 3 directly to level 0, ``> 0``. Default
        ``np.inf`` (no direct 3 -> 0 decay).
    omega_b : float, optional
        Resonance angular frequency [rad/s] of the 0 <-> 3 pump transition. If
        None (default), only the 1 <-> 2 polarization ``P_a`` is integrated and
        pumping proceeds via ``pump_rate``.
    gamma_b : float, optional
        Homogeneous damping rate [rad/s] of the 0 <-> 3 transition, ``>= 0``.
        Default ``0.0``.
    kappa_a : float, optional
        Coupling constant [C^2 / (kg m^3)] in
        ``d^2 P_a/dt^2 + gamma_a dP_a/dt + omega_a^2 P_a = kappa_a (N_1 - N_2) E``.
        Defaults to the classical dipole value
        ``6 * pi * eps_0 * c^3 / (omega_a^2 * tau_21)``.
    kappa_b : float, optional
        Coupling constant [C^2 / (kg m^3)] for the 0 <-> 3 transition. Defaults
        to ``6 * pi * eps_0 * c^3 / (omega_b^2 * tau_30)`` when ``omega_b`` is
        set and ``tau_30`` is finite, else ``0.0``.
    initial_populations : sequence of 4 float or {'ground', 'steady_state'}, optional
        Initial population densities ``(N_0, N_1, N_2, N_3)`` [1/m^3].
        ``'ground'`` (default) sets ``(N_total, 0, 0, 0)``; ``'steady_state'``
        initializes to the zero-field rate-equation steady state
        `steady_state_populations`.
    pauli_exclusion : bool, optional
        If True, multiply each transition ``i -> j`` in the rate equations by
        the Pauli exclusion blocking factor ``(1 - N_j / N_site)`` (Chang &
        Taflove 2004, Eqs. 5a-5d). Default False.
    N_site : float, optional
        Maximum single-level site density [1/m^3] for Pauli blocking when
        ``pauli_exclusion=True``. Defaults to ``N_total``.
    enable_stimulated : bool, optional
        If True (default), include the field-driven transition terms
        ``(1 / (hbar * omega)) E . dP/dt`` in the rate equations.
    spatial_averaging : bool, optional
        If False (default), each grid cell evolves its own local population
        vector ``N_i(r)`` and captures standing-wave spatial hole burning. If
        True, the stimulated transition rate is spatially averaged across the
        gain region at each step (fast carrier diffusion / homogenized-cavity
        limit).
    sigma : float, optional
        Host static electric conductivity [S/m], ``>= 0``. Default ``0.0``.
    name : str, optional
        Optional material name.
    """

    def __init__(
        self,
        omega_a,
        gamma_a,
        tau_21,
        tau_10,
        tau_32,
        N_total=1.0e24,
        pump_rate=0.0,
        *,
        N_density=None,
        W_p=None,
        eps_inf=1.0,
        tau_30=np.inf,
        omega_b=None,
        gamma_b=0.0,
        kappa_a=None,
        kappa_b=None,
        initial_populations="ground",
        pauli_exclusion=False,
        N_site=None,
        enable_stimulated=True,
        spatial_averaging=False,
        sigma=0.0,
        name=None,
    ):
        super().__init__(eps_inf=eps_inf, sigma=sigma, name=name)
        if N_density is not None:
            N_total = N_density
        if W_p is not None:
            pump_rate = W_p

        self.omega_a = float(omega_a)
        self.gamma_a = float(gamma_a)
        self.tau_21 = float(tau_21)
        self.tau_10 = float(tau_10)
        self.tau_32 = float(tau_32)
        self.tau_30 = float(tau_30)
        self.N_total = float(N_total)
        self.pump_rate = float(pump_rate)

        if self.omega_a <= 0.0:
            raise ValueError(f"omega_a must be > 0, got {self.omega_a}.")
        if self.gamma_a < 0.0:
            raise ValueError(f"gamma_a must be >= 0, got {self.gamma_a}.")
        if self.tau_21 <= 0.0 or self.tau_10 <= 0.0 or self.tau_32 <= 0.0 or self.tau_30 <= 0.0:
            raise ValueError("All decay lifetimes tau_ij must be > 0.")
        if self.N_total < 0.0:
            raise ValueError(f"N_total must be >= 0, got {self.N_total}.")
        if self.pump_rate < 0.0:
            raise ValueError(f"pump_rate must be >= 0, got {self.pump_rate}.")

        if kappa_a is None:
            self.kappa_a = float(
                6.0 * np.pi * eps_0 * c_light**3 / (self.omega_a**2 * self.tau_21)
            )
        else:
            self.kappa_a = float(kappa_a)

        self.omega_b = None if omega_b is None else float(omega_b)
        self.gamma_b = float(gamma_b)
        if self.omega_b is not None:
            if self.omega_b <= 0.0:
                raise ValueError(f"omega_b must be > 0, got {self.omega_b}.")
            if self.gamma_b < 0.0:
                raise ValueError(f"gamma_b must be >= 0, got {self.gamma_b}.")
            if kappa_b is None:
                self.kappa_b = (
                    0.0
                    if np.isinf(self.tau_30)
                    else float(
                        6.0
                        * np.pi
                        * eps_0
                        * c_light**3
                        / (self.omega_b**2 * self.tau_30)
                    )
                )
            else:
                self.kappa_b = float(kappa_b)
        else:
            self.kappa_b = 0.0 if kappa_b is None else float(kappa_b)

        self.pauli_exclusion = bool(pauli_exclusion)
        self.N_site = (
            float(self.N_total if self.N_total > 0 else 1.0)
            if N_site is None
            else float(N_site)
        )
        if self.N_site <= 0.0:
            raise ValueError(f"N_site must be > 0, got {self.N_site}.")
        self.enable_stimulated = bool(enable_stimulated)
        self.spatial_averaging = bool(spatial_averaging)

        if isinstance(initial_populations, str):
            mode = initial_populations.lower()
            if mode not in ("ground", "steady_state"):
                raise ValueError(
                    "initial_populations must be 'ground', 'steady_state', or a "
                    "4-sequence (N0, N1, N2, N3)."
                )
            self._init_mode = mode
            self._init_tuple = None
        else:
            pops = tuple(float(v) for v in initial_populations)
            if len(pops) != 4:
                raise ValueError("initial_populations must have 4 elements (N0, N1, N2, N3).")
            self._init_mode = "explicit"
            self._init_tuple = pops
            self.N_total = float(sum(pops))

    @property
    def W_p(self):
        """External pumping rate ``pump_rate`` [1/s]."""
        return self.pump_rate

    @W_p.setter
    def W_p(self, value):
        val = float(value)
        if val < 0.0:
            raise ValueError(f"W_p must be >= 0, got {val}.")
        self.pump_rate = val

    @property
    def N_density(self):
        """Total atomic population density ``N_total`` [1/m^3]."""
        return self.N_total

    @property
    def has_transition_b(self):
        """Whether the 0 <-> 3 polarization ``P_b`` is active."""
        return self.omega_b is not None and self.kappa_b != 0.0

    @property
    def tau_3(self):
        """Total lifetime ``(1/tau_32 + 1/tau_30)^(-1)`` [s] of level 3."""
        rate_3 = (1.0 / self.tau_32) + (0.0 if np.isinf(self.tau_30) else 1.0 / self.tau_30)
        return np.inf if rate_3 == 0.0 else float(1.0 / rate_3)

    @property
    def emission_cross_section(self):
        """
        Stimulated emission cross-section at line center ``omega_a`` [m^2].

        Defined such that the line-center power gain coefficient is
        ``g_0 = sigma_21 * (N_2 - N_1)``, with
        ``sigma_21 = kappa_a / (eps_0 * c * sqrt(eps_inf) * gamma_a)``.
        """
        if self.gamma_a <= 0.0:
            return np.inf
        return float(self.kappa_a / (eps_0 * c_light * self.n_bg * self.gamma_a))

    def initial_population_vector(self):
        """Return the initial population vector ``[N_0, N_1, N_2, N_3]`` [1/m^3]."""
        if self._init_mode == "ground":
            return np.array([self.N_total, 0.0, 0.0, 0.0], dtype=float)
        if self._init_mode == "steady_state":
            return self.steady_state_populations()
        return np.array(self._init_tuple, dtype=float)

    def rate_matrix(self, W_p=None, stim_rate_a=0.0):
        """
        Linear rate matrix ``M`` [1/s] for the state vector ``[N_0, N_1, N_2, N_3]``.

        Parameters
        ----------
        W_p : float, optional
            Pumping rate [1/s]. Defaults to ``self.pump_rate``.
        stim_rate_a : float, optional
            Continuous-wave stimulated transition rate ``W_stim = sigma_21 * I / (hbar * omega_a)``
            [1/s] between levels 1 and 2. Default ``0.0``.

        Returns
        -------
        M : ndarray of shape (4, 4)
            Rate matrix whose columns sum to zero (exact conservation of ``sum N_i``).
        """
        wp = self.pump_rate if W_p is None else float(W_p)
        g10 = 0.0 if np.isinf(self.tau_10) else 1.0 / self.tau_10
        g21 = 0.0 if np.isinf(self.tau_21) else 1.0 / self.tau_21
        g32 = 0.0 if np.isinf(self.tau_32) else 1.0 / self.tau_32
        g30 = 0.0 if np.isinf(self.tau_30) else 1.0 / self.tau_30
        ws = float(stim_rate_a)
        return np.array(
            [
                [-wp, g10, 0.0, g30],
                [0.0, -(g10 + ws), g21 + ws, 0.0],
                [0.0, ws, -(g21 + ws), g32],
                [wp, 0.0, 0.0, -(g32 + g30)],
            ],
            dtype=float,
        )

    def steady_state_populations(self, W_p=None, intensity=0.0):
        """
        Analytic steady-state population densities ``[N_0, N_1, N_2, N_3]`` [1/m^3].

        Parameters
        ----------
        W_p : float, optional
            Pumping rate [1/s]. Defaults to ``self.pump_rate``.
        intensity : float, optional
            Continuous-wave optical intensity ``I`` [W/m^2] at ``omega_a``.
            Default ``0.0`` (unsaturated steady state).

        Returns
        -------
        N_ss : ndarray of shape (4,)
            Steady-state populations ``[N_0, N_1, N_2, N_3]`` [1/m^3] summing
            to ``N_total``.
        """
        wp = self.pump_rate if W_p is None else float(W_p)
        if self.N_total == 0.0:
            return np.zeros(4, dtype=float)
        if wp == 0.0 and intensity == 0.0:
            if np.isinf(self.tau_10) and self._init_mode == "explicit":
                # If 1 -> 0 relaxation is disabled, level 1 population stays trapped
                n0, n1, n2, n3 = self._init_tuple
                return np.array([n0 + n3, n1 + n2, 0.0, 0.0], dtype=float)
            return np.array([self.N_total, 0.0, 0.0, 0.0], dtype=float)

        ws = (
            float(intensity) * self.emission_cross_section / (hbar * self.omega_a)
            if (intensity > 0.0 and self.enable_stimulated and self.gamma_a > 0.0)
            else 0.0
        )

        if not self.pauli_exclusion:
            t3 = self.tau_3
            r3 = wp * t3  # N_3 / N_0
            g32 = 1.0 / self.tau_32
            g21 = 1.0 / self.tau_21
            g10 = 0.0 if np.isinf(self.tau_10) else 1.0 / self.tau_10
            if g10 == 0.0:
                return np.array([0.0, self.N_total, 0.0, 0.0], dtype=float)
            # Steady state of dN_3/dt = 0 and dN_1/dt + dN_2/dt = 0 -> g32*N_3 = g10*N_1
            r1 = (g32 / g10) * r3  # N_1 / N_0
            r2 = (g32 * r3 + ws * r1) / (g21 + ws)  # N_2 / N_0
            n0 = self.N_total / (1.0 + r1 + r2 + r3)
            return np.array([n0, r1 * n0, r2 * n0, r3 * n0], dtype=float)

        # Pauli exclusion: solve in normalized occupancy u_i = N_i / N_site
        ns = self.N_site
        u_tot = self.N_total / ns
        g10 = 0.0 if np.isinf(self.tau_10) else 1.0 / self.tau_10
        g21 = 1.0 / self.tau_21
        g32 = 1.0 / self.tau_32
        g30 = 0.0 if np.isinf(self.tau_30) else 1.0 / self.tau_30

        # Direct Newton/root solve on the 3 independent variables (u1, u2, u3) with u0 = u_tot - u1 - u2 - u3
        from scipy.optimize import root

        def residual(u123):
            u1, u2, u3 = u123
            u0 = u_tot - (u1 + u2 + u3)
            rp = wp * u0 * (1.0 - u3)
            r32 = g32 * u3 * (1.0 - u2)
            r30 = g30 * u3 * (1.0 - u0)
            r21 = g21 * u2 * (1.0 - u1) + ws * (u2 - u1)
            r10 = g10 * u1 * (1.0 - u0)
            return np.array([rp - r32 - r30, r32 - r21, r21 - r10])

        # Initial guess from the linear steady state
        self.pauli_exclusion = False
        guess = self.steady_state_populations(W_p=wp, intensity=intensity) / ns
        self.pauli_exclusion = True
        sol = root(residual, guess[1:], tol=1e-14)
        u1, u2, u3 = sol.x
        u0 = u_tot - (u1 + u2 + u3)
        return ns * np.array([u0, u1, u2, u3], dtype=float)

    def steady_state_inversion(self, W_p=None, intensity=0.0):
        """
        Steady-state population inversion ``Delta N_a = N_2 - N_1`` [1/m^3].

        Parameters
        ----------
        W_p : float, optional
            Pumping rate [1/s]. Defaults to ``self.pump_rate``.
        intensity : float, optional
            Continuous-wave intensity ``I`` [W/m^2] at ``omega_a``. Default ``0.0``.

        Returns
        -------
        delta_N : float
            Population inversion ``N_2 - N_1`` [1/m^3] (positive for gain).
        """
        n_ss = self.steady_state_populations(W_p=W_p, intensity=intensity)
        return float(n_ss[2] - n_ss[1])

    def saturation_time(self, W_p=None):
        """
        Effective saturation recovery time ``tau_sat`` [s] of the 4-level system.

        In steady state under monochromatic illumination at ``omega_a``, the
        inversion obeys ``Delta N_a(I) = Delta N_a(0) / (1 + I / I_sat)`` with
        ``I_sat = hbar * omega_a / (sigma_21 * tau_sat)`` and::

            tau_sat = (tau_21 * (1 + W_p * tau_3 * (1 + tau_10 / tau_32)))
                      / (1 + W_p * tau_3 * (1 + (tau_10 + tau_21) / tau_32))

        which reduces to ``tau_21`` in the weak-pump or fast-relaxation limit
        ``tau_10, tau_32 << tau_21``.
        """
        wp = self.pump_rate if W_p is None else float(W_p)
        t3 = self.tau_3
        t10 = 0.0 if np.isinf(self.tau_10) else self.tau_10
        num = self.tau_21 * (1.0 + wp * t3 * (1.0 + t10 / self.tau_32))
        den = 1.0 + wp * t3 * (1.0 + (t10 + self.tau_21) / self.tau_32)
        return float(num / den)

    def saturation_intensity(self, W_p=None):
        """
        Homogeneous saturation intensity ``I_sat`` [W/m^2] at line center ``omega_a``.

        Returns
        -------
        I_sat : float
            ``hbar * omega_a / (sigma_21 * tau_sat)`` [W/m^2].
        """
        if not self.enable_stimulated or self.kappa_a == 0.0:
            return np.inf
        return float(
            hbar
            * self.omega_a
            / (self.emission_cross_section * self.saturation_time(W_p=W_p))
        )

    def small_signal_gain(self, f=None, W_p=None, populations=None):
        """
        Small-signal power gain coefficient ``g(f)`` [1/m].

        Parameters
        ----------
        f : float or array_like, optional
            Frequency [Hz]. Defaults to the lasing line center
            ``f_a = omega_a / (2 * pi)``.
        W_p : float, optional
            Pumping rate [1/s] used when ``populations`` is None. Defaults to
            ``self.pump_rate``.
        populations : sequence of 4 float, optional
            Explicit populations ``(N_0, N_1, N_2, N_3)`` [1/m^3]. If None, uses
            `steady_state_populations`.

        Returns
        -------
        g : float or ndarray
            Power gain coefficient ``g(f) = 2 Im k(f)`` [1/m] (positive for
            amplification, negative for absorption).
        """
        if f is None:
            f_arr = np.asarray(self.omega_a / (2.0 * np.pi), dtype=float)
            scalar = True
        else:
            f_arr = np.asarray(f, dtype=float)
            scalar = f_arr.ndim == 0
        pops = (
            self.steady_state_populations(W_p=W_p, intensity=0.0)
            if populations is None
            else np.asarray(populations, dtype=float)
        )
        eps_r = self.epsilon(f_arr, populations=pops)
        omega = 2.0 * np.pi * f_arr
        # Complex refractive index n_c = n_r + i n_i in exp(+i w t - i k z) convention:
        # k = (w / c) * sqrt(eps_r) chosen with Re(k) >= 0; power grows as exp(+2 Im(k) z)
        n_c = np.sqrt(eps_r.astype(complex))
        n_c = np.where(n_c.real < 0, -n_c, n_c)
        g = 2.0 * (omega / c_light) * n_c.imag
        return float(g) if scalar else g

    def gain_coefficient(self, intensity=0.0, f=None, W_p=None):
        """
        Saturated power gain coefficient ``g(f, I)`` [1/m] at intensity ``I`` [W/m^2].
        """
        pops = self.steady_state_populations(W_p=W_p, intensity=intensity)
        return self.small_signal_gain(f=f, populations=pops)

    def threshold_pump_rate(self, g_th):
        """
        Threshold pumping rate ``W_{p,th}`` [1/s] required for line-center gain ``g_th``.

        Solves ``sigma_21 * Delta N_a(W_{p,th}) = g_th`` in closed form.

        Parameters
        ----------
        g_th : float
            Required threshold power gain coefficient [1/m], ``>= 0``.

        Returns
        -------
        W_p_th : float
            Threshold pumping rate [1/s].

        Raises
        ------
        ValueError
            If ``tau_21 <= tau_10`` (no population inversion possible) or if
            ``g_th`` exceeds the maximum achievable gain at infinite pump.
        """
        g_th = float(g_th)
        if g_th < 0.0:
            raise ValueError(f"g_th must be >= 0, got {g_th}.")
        if g_th == 0.0:
            return 0.0
        if self.tau_21 <= self.tau_10:
            raise ValueError(
                f"Population inversion requires tau_21 > tau_10, got "
                f"tau_21={self.tau_21}, tau_10={self.tau_10}."
            )
        dn_th = g_th / self.emission_cross_section
        t3 = self.tau_3
        a = self.N_total * (t3 / self.tau_32) * (self.tau_21 - self.tau_10)
        b = (t3 / self.tau_32) * (self.tau_10 + self.tau_21) + t3
        if dn_th * b >= a:
            raise ValueError(
                f"Threshold inversion Delta N_th={dn_th:.3e} m^-3 exceeds the "
                f"maximum inversion {a / b:.3e} m^-3 of this medium."
            )
        if not self.pauli_exclusion:
            return float(dn_th / (a - dn_th * b))

        # With Pauli exclusion, refine with Brent's method
        wp_lin = float(dn_th / (a - dn_th * b))
        return float(
            brentq(
                lambda wp: self.steady_state_inversion(W_p=wp) - dn_th,
                0.0,
                max(10.0 * wp_lin, 1.0 / self.tau_21),
                rtol=1e-12,
            )
        )

    def _effective_populations(self, populations=None, intensity=0.0):
        if populations is not None:
            p = np.asarray(populations, dtype=float)
            if p.shape != (4,):
                raise ValueError("populations must have shape (4,): (N0, N1, N2, N3).")
            return p
        if self._init_mode == "explicit" and self.pump_rate == 0.0 and intensity == 0.0:
            return np.asarray(self._init_tuple, dtype=float)
        return self.steady_state_populations(intensity=intensity)

    def chi(self, omega, populations=None, intensity=0.0):
        """
        Continuous electric susceptibility ``chi(omega)`` in ``exp(+i w t)`` convention.

        Parameters
        ----------
        omega : float or array_like
            Angular frequency [rad/s].
        populations : sequence of 4 float, optional
            Populations ``(N_0, N_1, N_2, N_3)`` [1/m^3]. Defaults to the
            steady-state populations at ``self.pump_rate`` and ``intensity``.
        intensity : float, optional
            Optical intensity [W/m^2] used when ``populations`` is None.

        Returns
        -------
        chi : complex ndarray
        """
        omega = np.asarray(omega, dtype=float)
        n0, n1, n2, n3 = self._effective_populations(populations, intensity)
        num_a = (self.kappa_a / eps_0) * (n1 - n2)
        chi_val = num_a / (self.omega_a**2 - omega**2 + 1j * self.gamma_a * omega)
        if self.has_transition_b:
            num_b = (self.kappa_b / eps_0) * (n0 - n3)
            chi_val = chi_val + num_b / (
                self.omega_b**2 - omega**2 + 1j * self.gamma_b * omega
            )
        return chi_val

    def chi_discrete(self, omega, dt, populations=None, intensity=0.0):
        """
        Discrete ADE electric susceptibility ``chi_d(omega, dt)`` at timestep ``dt`` [s].
        """
        omega = np.asarray(omega, dtype=float)
        n0, n1, n2, n3 = self._effective_populations(populations, intensity)
        w2 = (2.0 / dt * np.sin(omega * dt / 2.0)) ** 2
        damp_a = 1j * self.gamma_a * np.sin(omega * dt) / dt
        num_a = (self.kappa_a / eps_0) * (n1 - n2)
        chi_val = num_a / (self.omega_a**2 - w2 + damp_a)
        if self.has_transition_b:
            damp_b = 1j * self.gamma_b * np.sin(omega * dt) / dt
            num_b = (self.kappa_b / eps_0) * (n0 - n3)
            chi_val = chi_val + num_b / (self.omega_b**2 - w2 + damp_b)
        return chi_val

    def epsilon(self, f, populations=None, intensity=0.0):
        """
        Complex relative permittivity ``eps(f)`` in ``exp(+i w t)`` convention.
        """
        omega = 2.0 * np.pi * np.asarray(f, dtype=float)
        eps = self.eps_inf + self.chi(omega, populations=populations, intensity=intensity)
        if self.sigma:
            eps = eps + self.sigma / (1j * omega * eps_0)
        return eps

    def epsilon_discrete(self, f, dt, populations=None, intensity=0.0):
        """
        Discrete complex relative permittivity ``eps_d(f, dt)`` at timestep ``dt`` [s].
        """
        omega = 2.0 * np.pi * np.asarray(f, dtype=float)
        eps = self.eps_inf + self.chi_discrete(
            omega, dt, populations=populations, intensity=intensity
        )
        if self.sigma:
            w_tilde = 2.0 / dt * np.sin(omega * dt / 2.0)
            eps = eps + self.sigma * np.cos(omega * dt / 2.0) / (1j * w_tilde * eps_0)
        return eps

    def is_stable(self, dt, courant=1.0):
        """
        Check whether timestep ``dt`` [s] resolves the atomic oscillator poles.

        Evaluates the von Neumann stability of the passive unpumped ground-state
        Lorentz oscillators and checks ``omega_a * dt < 2`` (and ``omega_b * dt < 2``).
        """
        dt = float(dt)
        if dt <= 0.0 or self.omega_a * dt >= 2.0:
            return False
        if self.omega_b is not None and self.omega_b * dt >= 2.0:
            return False
        n_init = self.initial_population_vector()
        n_ss = self.steady_state_populations()
        dn_a = max(abs(float(n_init[1] - n_init[2])), abs(float(n_ss[1] - n_ss[2])))
        dn_b = max(abs(float(n_init[0] - n_init[3])), abs(float(n_ss[0] - n_ss[3])))
        poles = []
        if dn_a > 0.0 and self.kappa_a > 0.0:
            de_a = self.kappa_a * dn_a / (eps_0 * self.omega_a**2)
            poles.append(Lorentz(de_a, self.omega_a, self.gamma_a))
        if self.has_transition_b and dn_b > 0.0 and self.kappa_b > 0.0:
            de_b = self.kappa_b * dn_b / (eps_0 * self.omega_b**2)
            poles.append(Lorentz(de_b, self.omega_b, self.gamma_b))
        if not poles:
            return True
        passive = DispersiveMaterial(self.eps_inf, poles, sigma=self.sigma)
        return passive.is_stable(dt, courant=courant)

    def step_rate_equations(self, Nt, dt, N_init=None, W_p=None):
        """
        Integrate the standalone zero-field rate equations for ``Nt`` steps of ``dt`` [s].

        Uses the exact same second-order Crank-Nicolson (or Heun for Pauli
        blocking) time discretization centred at ``t^(n+1/2)`` as the 3D solver.

        Parameters
        ----------
        Nt : int
            Number of timesteps.
        dt : float
            Timestep [s].
        N_init : sequence of 4 float, optional
            Initial populations ``(N_0, N_1, N_2, N_3)`` [1/m^3]. Defaults to
            `initial_population_vector`.
        W_p : float, optional
            Pumping rate [1/s]. Defaults to ``self.pump_rate``.

        Returns
        -------
        t : ndarray of shape (Nt + 1,)
            Time array [s].
        N_hist : ndarray of shape (Nt + 1, 4)
            Population history ``[N_0, N_1, N_2, N_3]`` [1/m^3] at each step.
        """
        Nt = int(Nt)
        dt = float(dt)
        n_vec = (
            self.initial_population_vector()
            if N_init is None
            else np.asarray(N_init, dtype=float).copy()
        )
        wp = self.pump_rate if W_p is None else float(W_p)
        hist = np.empty((Nt + 1, 4), dtype=float)
        hist[0] = n_vec
        if not self.pauli_exclusion:
            a_mat, _ = _rate_cn_matrices(self, dt, W_p=wp)
            for n in range(Nt):
                n_vec = a_mat @ n_vec
                hist[n + 1] = n_vec
        else:
            s_zero = np.zeros((4, 1), dtype=float)
            n_col = n_vec.reshape(4, 1)
            for n in range(Nt):
                n_col = _step_pauli_rates(self, n_col, s_zero, dt, wp=wp)
                hist[n + 1] = n_col[:, 0]
        return np.arange(Nt + 1) * dt, hist

    def __repr__(self):
        label = f"{self.name}: " if self.name else ""
        return (
            f"FourLevelGain({label}eps_inf={self.eps_inf:.6g}, "
            f"omega_a={self.omega_a:.6g}, gamma_a={self.gamma_a:.6g}, "
            f"tau_21={self.tau_21:.6g}, tau_10={self.tau_10:.6g}, "
            f"tau_32={self.tau_32:.6g}, N_total={self.N_total:.6g}, "
            f"pump_rate={self.pump_rate:.6g})"
        )


class SaturableGain(GainMaterial):
    """
    Saturable-gain Lorentz polarization model ``g(I) = g0 / (1 + I / I_sat)``.

    Integrates a single Lorentz polarization ``P`` whose oscillator strength
    saturates homogeneously with the local optical intensity ``I`` [W/m^2]::

        d^2 P / dt^2 + gamma dP / dt + omega_0^2 P
            = eps_0 (delta_eps_0 omega_0^2 / (1 + I / I_sat)) E

    with ``delta_eps_0 = -c sqrt(eps_inf) gamma g0 / omega_0^2``.

    Parameters
    ----------
    omega_0 : float
        Resonance angular frequency [rad/s], ``> 0``.
    gamma : float
        Homogeneous linewidth (damping rate) [rad/s], ``> 0``.
    g0 : float, optional
        Unsaturated line-center power gain coefficient [1/m] (``> 0`` for gain,
        ``< 0`` for saturable absorption, ``0`` for passive background). Default
        ``0.0``.
    I_sat : float, optional
        Saturation intensity [W/m^2], ``> 0``. Default ``np.inf`` (unsaturated
        linear Lorentz response).
    eps_inf : float, optional
        Host high-frequency relative permittivity [-], ``> 0``. Default ``1.0``.
    delta_eps : float, optional
        Explicit unsaturated Lorentz susceptibility strength ``delta_eps_0`` [-].
        If provided, overrides ``g0`` via
        ``g0 = -omega_0^2 * delta_eps / (c * sqrt(eps_inf) * gamma)``.
    tau_I : float, optional
        Averaging time constant [s] of the cycle-averaged intensity filter
        ``tau_I dI/dt + I = c eps_0 sqrt(eps_inf) |E|^2``. If None (default),
        uses 5 optical periods ``5 * (2 * pi / omega_0)``. Set ``0.0`` for
        instantaneous ``I(t) = 0.5 * c * eps_0 * sqrt(eps_inf) * |E(t)|^2``.
    sigma : float, optional
        Host static electric conductivity [S/m], ``>= 0``. Default ``0.0``.
    name : str, optional
        Optional material label.
    """

    def __init__(
        self,
        omega_0,
        gamma,
        g0=0.0,
        I_sat=np.inf,
        *,
        eps_inf=1.0,
        delta_eps=None,
        tau_I=None,
        sigma=0.0,
        name=None,
    ):
        super().__init__(eps_inf=eps_inf, sigma=sigma, name=name)
        self.omega_0 = float(omega_0)
        self.gamma = float(gamma)
        if self.omega_0 <= 0.0:
            raise ValueError(f"omega_0 must be > 0, got {self.omega_0}.")
        if self.gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {self.gamma}.")

        if delta_eps is not None:
            self.delta_eps = float(delta_eps)
            self.g0 = (
                0.0
                if self.gamma == 0.0
                else float(
                    -self.omega_0**2
                    * self.delta_eps
                    / (c_light * self.n_bg * self.gamma)
                )
            )
        else:
            self.g0 = float(g0)
            self.delta_eps = float(
                -c_light * self.n_bg * self.gamma * self.g0 / (self.omega_0**2)
            )

        self.I_sat = float(I_sat)
        if self.I_sat <= 0.0:
            raise ValueError(f"I_sat must be > 0, got {self.I_sat}.")

        if tau_I is None:
            self.tau_I = float(5.0 * (2.0 * np.pi / self.omega_0))
        else:
            self.tau_I = float(tau_I)
            if self.tau_I < 0.0:
                raise ValueError(f"tau_I must be >= 0, got {self.tau_I}.")

    def _sat_factor(self, intensity=0.0):
        if np.isinf(self.I_sat):
            return 1.0
        return 1.0 / (1.0 + np.asarray(intensity, dtype=float) / self.I_sat)

    def chi(self, omega, intensity=0.0):
        """
        Continuous susceptibility ``chi(omega, I)`` in ``exp(+i w t)`` convention.
        """
        omega = np.asarray(omega, dtype=float)
        strength = self.delta_eps * self._sat_factor(intensity) * self.omega_0**2
        return strength / (self.omega_0**2 - omega**2 + 1j * self.gamma * omega)

    def chi_discrete(self, omega, dt, intensity=0.0):
        """
        Discrete ADE susceptibility ``chi_d(omega, dt, I)`` at timestep ``dt`` [s].
        """
        omega = np.asarray(omega, dtype=float)
        w2 = (2.0 / dt * np.sin(omega * dt / 2.0)) ** 2
        damping = 1j * self.gamma * np.sin(omega * dt) / dt
        strength = self.delta_eps * self._sat_factor(intensity) * self.omega_0**2
        return strength / (self.omega_0**2 - w2 + damping)

    def epsilon(self, f, intensity=0.0):
        """
        Complex relative permittivity ``eps(f, I)`` in ``exp(+i w t)`` convention.
        """
        omega = 2.0 * np.pi * np.asarray(f, dtype=float)
        eps = self.eps_inf + self.chi(omega, intensity=intensity)
        if self.sigma:
            eps = eps + self.sigma / (1j * omega * eps_0)
        return eps

    def epsilon_discrete(self, f, dt, intensity=0.0):
        """
        Discrete complex relative permittivity ``eps_d(f, dt, I)`` at timestep ``dt`` [s].
        """
        omega = 2.0 * np.pi * np.asarray(f, dtype=float)
        eps = self.eps_inf + self.chi_discrete(omega, dt, intensity=intensity)
        if self.sigma:
            w_tilde = 2.0 / dt * np.sin(omega * dt / 2.0)
            eps = eps + self.sigma * np.cos(omega * dt / 2.0) / (1j * w_tilde * eps_0)
        return eps

    def gain_coefficient(self, intensity=0.0, f=None):
        """
        Power gain coefficient ``g(f, I) = g(f, 0) / (1 + I / I_sat)`` [1/m].
        """
        if f is None:
            return float(self.g0 * self._sat_factor(intensity))
        f_arr = np.asarray(f, dtype=float)
        scalar = f_arr.ndim == 0
        omega = 2.0 * np.pi * f_arr
        lorentz_shape = (self.gamma * omega) ** 2 / (
            (self.omega_0**2 - omega**2) ** 2 + (self.gamma * omega) ** 2
        )
        g = self.g0 * self._sat_factor(intensity) * lorentz_shape
        return float(g) if scalar else g

    def small_signal_gain(self, f=None):
        """Unsaturated small-signal power gain coefficient ``g(f, 0)`` [1/m]."""
        return self.gain_coefficient(intensity=0.0, f=f)

    def is_stable(self, dt, courant=1.0):
        """Check whether timestep ``dt`` [s] resolves the Lorentz pole."""
        dt = float(dt)
        if dt <= 0.0 or self.omega_0 * dt >= 2.0:
            return False
        if self.delta_eps == 0.0:
            return True
        passive = DispersiveMaterial(
            self.eps_inf,
            [Lorentz(abs(self.delta_eps), self.omega_0, self.gamma)],
            sigma=self.sigma,
        )
        return passive.is_stable(dt, courant=courant)

    def __repr__(self):
        label = f"{self.name}: " if self.name else ""
        return (
            f"SaturableGain({label}eps_inf={self.eps_inf:.6g}, "
            f"omega_0={self.omega_0:.6g}, gamma={self.gamma:.6g}, "
            f"g0={self.g0:.6g}, I_sat={self.I_sat:.6g})"
        )


def amplified_intensity(I_in, length, g0, I_sat=np.inf):
    """
    Exact output intensity of a homogeneous saturable amplifier of length ``length``.

    Integrates ``dI/dz = g0 * I / (1 + I / I_sat)`` from ``z = 0`` to ``z = length``::

        ln(I_out / I_in) + (I_out - I_in) / I_sat = g0 * length

    via the principal Lambert W branch ``I_out = I_sat * W_0((I_in / I_sat) *
    exp(I_in / I_sat + g0 * length))``.

    Parameters
    ----------
    I_in : float or array_like
        Input optical intensity [W/m^2], ``>= 0``.
    length : float
        Amplifier length ``L`` [m], ``>= 0``.
    g0 : float
        Small-signal power gain coefficient [1/m].
    I_sat : float, optional
        Saturation intensity [W/m^2], ``> 0``. Default ``np.inf`` (unsaturated
        exponential growth ``I_in * exp(g0 * length)``).

    Returns
    -------
    I_out : float or ndarray
        Output optical intensity [W/m^2].
    """
    i_arr = np.asarray(I_in, dtype=float)
    scalar = i_arr.ndim == 0
    i_1d = np.atleast_1d(i_arr)
    if np.any(i_1d < 0.0):
        raise ValueError("I_in must be >= 0.")
    L = float(length)
    g0 = float(g0)
    if np.isinf(I_sat):
        out = i_1d * np.exp(g0 * L)
        return float(out[0]) if scalar else out.reshape(i_arr.shape)

    isat = float(I_sat)
    if isat <= 0.0:
        raise ValueError(f"I_sat must be > 0, got {isat}.")
    out = np.zeros_like(i_1d, dtype=float)
    pos = i_1d > 0.0
    if np.any(pos):
        u0 = i_1d[pos] / isat
        y = np.log(u0) + u0 + g0 * L
        # For moderate y, use lambertw(exp(y)); for large y, use asymptotic + Halley/Newton
        u = np.empty_like(y)
        small = y < 500.0
        if np.any(small):
            u[small] = np.real(lambertw(np.exp(y[small])))
        if np.any(~small):
            yl = y[~small]
            ln_y = np.log(yl)
            u_est = yl - ln_y + ln_y / yl
            for _ in range(3):
                f_val = u_est + np.log(u_est) - yl
                u_est -= f_val / (1.0 + 1.0 / u_est)
            u[~small] = u_est
        # One Newton polish on moderate values where u > 0
        nz = u > 0.0
        if np.any(nz):
            un = u[nz]
            yn = y[nz]
            u[nz] = un - (un + np.log(un) - yn) / (1.0 + 1.0 / un)
        out[pos] = isat * u
    return float(out[0]) if scalar else out.reshape(i_arr.shape)


def fabry_perot_threshold_gain(length, R1, R2=None, alpha_int=0.0, confinement=1.0):
    """
    Cold-cavity threshold power gain coefficient ``g_th`` [1/m] of a Fabry-Perot laser.

    Parameters
    ----------
    length : float
        Gain-medium length ``L`` [m] inside the cavity, ``> 0``.
    R1 : float
        Power reflectance of mirror 1 in ``(0, 1]``.
    R2 : float, optional
        Power reflectance of mirror 2 in ``(0, 1]``. Defaults to ``R1``.
    alpha_int : float, optional
        Distributed internal power loss coefficient [1/m], ``>= 0``. Default ``0.0``.
    confinement : float, optional
        Longitudinal/modal confinement factor ``Gamma`` in ``(0, 1]``. Default ``1.0``.

    Returns
    -------
    g_th : float
        Threshold power gain coefficient
        ``(alpha_int + ln(1 / (R1 * R2)) / (2 * length)) / confinement`` [1/m].
    """
    L = float(length)
    r1 = float(R1)
    r2 = r1 if R2 is None else float(R2)
    gamma_c = float(confinement)
    if L <= 0.0:
        raise ValueError(f"length must be > 0, got {L}.")
    if not (0.0 < r1 <= 1.0 and 0.0 < r2 <= 1.0):
        raise ValueError(f"Mirror reflectances R1, R2 must be in (0, 1], got {r1}, {r2}.")
    if gamma_c <= 0.0:
        raise ValueError(f"confinement must be > 0, got {gamma_c}.")
    return float((float(alpha_int) + np.log(1.0 / (r1 * r2)) / (2.0 * L)) / gamma_c)


def dbr_reflectance(n_high, n_low, n_pairs, n_in=1.0, n_out=1.0, extra_high=False):
    """
    Bragg-wavelength normal-incidence power reflectance ``R`` of a quarter-wave DBR.

    At the centre wavelength ``lambda_0`` where each layer has optical thickness
    ``lambda_0 / 4``, ``n_pairs`` pairs of ``(n_high, n_low)`` layers (plus one
    optional terminating ``n_high`` quarter-wave layer when ``extra_high=True``)
    transform the exit admittance by ``(n_high / n_low)**(2 * n_pairs)``.

    Parameters
    ----------
    n_high, n_low : float
        Refractive indices of the quarter-wave layers, ``> 0``.
    n_pairs : int
        Number of ``(H, L)`` pairs, ``>= 0``.
    n_in, n_out : float, optional
        Refractive indices of the incident (cavity) and exit media. Default ``1.0``.
    extra_high : bool, optional
        If True, append one terminating ``n_high`` quarter-wave layer ``(H, L)^N H``.

    Returns
    -------
    R : float
        Power reflectance in ``[0, 1)``.
    """
    nh, nl = float(n_high), float(n_low)
    nin, nout = float(n_in), float(n_out)
    N = int(n_pairs)
    ratio = (nh / nl) ** (2 * N)
    if extra_high:
        y_eff = (nh**2 * ratio) / (nin * nout)
    else:
        y_eff = (nout / nin) * ratio
    r = (1.0 - y_eff) / (1.0 + y_eff)
    return float(r * r)


def fabry_perot_laser_output(
    pump_ratio, I_sat, R1, R2=None, spatial_hole_burning=True
):
    """
    Analytic steady-state single-side output intensity ``I_out`` [W/m^2] of a Fabry-Perot laser.

    Parameters
    ----------
    pump_ratio : float or array_like
        Normalized unsaturated inversion ratio ``r = Delta N_0(W_p) / Delta N_th``
        (equal to ``W_p / W_{p,th}`` in the linear-pump regime).
    I_sat : float
        Homogeneous saturation intensity [W/m^2], ``> 0``.
    R1 : float
        Power reflectance of the output mirror in ``(0, 1)``.
    R2 : float, optional
        Power reflectance of the second mirror in ``(0, 1]``. Defaults to ``R1``.
    spatial_hole_burning : bool, optional
        If True (default), include the standing-wave spatial hole burning
        reduction ``s(r) = (4 r - 1 - sqrt(8 r + 1)) / 4`` (Siegman 1986,
        Sec. 8.3). If False, use the spatially homogenized linear law
        ``s(r) = r - 1``.

    Returns
    -------
    I_out : float or ndarray
        Steady-state output intensity [W/m^2] exiting through mirror 1.
    """
    r_arr = np.asarray(pump_ratio, dtype=float)
    scalar = r_arr.ndim == 0
    r1 = float(R1)
    r2 = r1 if R2 is None else float(R2)
    isat = float(I_sat)
    if not (0.0 < r1 < 1.0 and 0.0 < r2 <= 1.0):
        raise ValueError(f"Expected 0 < R1 < 1 and 0 < R2 <= 1, got {r1}, {r2}.")
    if isat <= 0.0:
        raise ValueError(f"I_sat must be > 0, got {isat}.")

    r_pos = np.maximum(r_arr, 1.0)
    if spatial_hole_burning:
        s = 0.25 * (4.0 * r_pos - 1.0 - np.sqrt(8.0 * r_pos + 1.0))
    else:
        s = r_pos - 1.0
    s = np.where(r_arr > 1.0, s, 0.0)

    # Rigrod outcoupling fraction through mirror 1 of the total round-trip mirror loss
    t1 = 1.0 - r1
    t2 = 1.0 - r2
    frac_1 = (t1 * np.sqrt(r2)) / (t1 * np.sqrt(r2) + t2 * np.sqrt(r1))
    total_mirror_loss = 0.5 * np.log(1.0 / (r1 * r2))
    out = frac_1 * total_mirror_loss * isat * s
    return float(out) if scalar else out


# ---------------------------------------------------------------------------
# Internal time-stepping helpers
# ---------------------------------------------------------------------------


def _lorentz_step_coeffs(omega_0, gamma, strength_per_eps0, dt):
    """Second-order central-difference coefficients (alpha, xi, zeta) at t^n."""
    d = 1.0 + 0.5 * gamma * dt
    alpha = (2.0 - (omega_0 * dt) ** 2) / d
    xi = (0.5 * gamma * dt - 1.0) / d
    zeta = strength_per_eps0 * dt**2 / d
    return float(alpha), float(xi), float(zeta)


def _rate_cn_matrices(material, dt, W_p=None):
    """Precompute Crank-Nicolson matrices (A_rate, B_rate) for 4-level populations."""
    m_mat = material.rate_matrix(W_p=W_p)
    eye4 = np.eye(4, dtype=float)
    lhs_inv = np.linalg.inv(eye4 - 0.5 * dt * m_mat)
    a_rate = lhs_inv @ (eye4 + 0.5 * dt * m_mat)
    b_rate = dt * lhs_inv
    return a_rate, b_rate


def _pauli_rate_rhs(material, n_arr, wp):
    """Evaluate the Pauli-blocked relaxation/pumping rate vector for N of shape (4, M)."""
    ns = material.N_site
    n0, n1, n2, n3 = n_arr[0], n_arr[1], n_arr[2], n_arr[3]
    u0 = np.clip(n0 / ns, 0.0, 1.0)
    u1 = np.clip(n1 / ns, 0.0, 1.0)
    u2 = np.clip(n2 / ns, 0.0, 1.0)
    u3 = np.clip(n3 / ns, 0.0, 1.0)
    g10 = 0.0 if np.isinf(material.tau_10) else 1.0 / material.tau_10
    g21 = 0.0 if np.isinf(material.tau_21) else 1.0 / material.tau_21
    g32 = 0.0 if np.isinf(material.tau_32) else 1.0 / material.tau_32
    g30 = 0.0 if np.isinf(material.tau_30) else 1.0 / material.tau_30

    r_pump = wp * n0 * (1.0 - u3)
    r32 = g32 * n3 * (1.0 - u2)
    r30 = g30 * n3 * (1.0 - u0)
    r21 = g21 * n2 * (1.0 - u1)
    r10 = g10 * n1 * (1.0 - u0)
    return np.stack(
        [
            -r_pump + r30 + r10,
            r21 - r10,
            r32 - r21,
            r_pump - r32 - r30,
        ],
        axis=0,
    )


def _step_pauli_rates(material, n_arr, s_half, dt, wp=None):
    """Second-order Heun (trapezoidal predictor-corrector) step for Pauli-blocked rates."""
    if wp is None:
        wp = material.pump_rate
    k1 = _pauli_rate_rhs(material, n_arr, wp) + s_half
    n_pred = n_arr + dt * k1
    k2 = _pauli_rate_rhs(material, n_pred, wp) + s_half
    return n_arr + 0.5 * dt * (k1 + k2)


class GainMixin:
    """
    `SolverFIT3D` mixin for active gain media (`FourLevelGain`, `SaturableGain`).

    Hooks around the electric-field update in every CPU time-stepping variant via
    `_capture_gain_E` (before ``E`` is updated) and `_update_gain_E` (immediately
    after ``E`` is updated).
    """

    def add_gain_material(
        self, material, center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0)
    ):
        """
        Fill a box region with an active gain material (`FourLevelGain` or `SaturableGain`).

        Primary ``E`` edges strictly inside the box receive fill weight ``1.0``;
        tangential ``E`` edges lying on a box boundary face receive weight ``0.5``
        (arithmetic interface averaging with the background medium). Later calls
        override earlier ones on shared edges.

        Parameters
        ----------
        material : FourLevelGain or SaturableGain
            Active gain material model.
        center, size : tuple of 3 float, optional
            Centre and extent ``(x, y, z)`` [m] of the box. A zero extent along
            an axis selects the nearest grid sample along that axis.

        Raises
        ------
        TypeError
            If ``material`` is not a `GainMaterial`.
        NotImplementedError
            If the solver uses ``use_gpu=True``, ``fused=True``, or
            ``scheme='tete'``.
        ValueError
            If the solver timestep ``dt`` does not resolve the material poles.
        """
        if not isinstance(material, GainMaterial):
            raise TypeError(
                f"material must be a GainMaterial (FourLevelGain or SaturableGain), "
                f"got {type(material).__name__}."
            )
        idx_e, w_e = self._box_edges(center, size, dual=False)
        self._add_gain_edges(material, idx_e, w_e)

    def _apply_stl_gain(self):
        """Attach gain ADE updates to STL solids whose material is a `GainMaterial`."""
        for key, material in self.grid.stl_materials.items():
            if not isinstance(material, GainMaterial):
                continue
            mask = np.reshape(self.grid.grid[key], (self.Nx, self.Ny, self.Nz))
            fraction = np.reshape(mask, self.N, order="F").astype(float)
            fraction = np.tile(fraction, 3)
            idx = np.flatnonzero(fraction)
            self._add_gain_edges(material, idx, fraction[idx], eps_bg=1.0)

    def _check_gain_supported_and_stable(self, material):
        if not isinstance(material, GainMaterial):
            raise TypeError("material must be a GainMaterial")
        if getattr(self, "scheme", "yee") == "tete":
            raise NotImplementedError(
                "scheme='tete' does not support gain (ADE) materials; use scheme='yee'."
            )
        fused = getattr(self, "_fused_step", None)
        if getattr(self, "fused", False) or (
            fused is not None and self.one_step is fused
        ):
            raise NotImplementedError(
                "Gain media do not support the fused GPU step (fused=True) yet; "
                "use fused=False."
            )
        if getattr(self, "use_gpu", False):
            raise NotImplementedError(
                "Gain media do not support GPU execution (use_gpu=True) yet; "
                "use use_gpu=False."
            )
        courant = (
            c_light
            * self.dt
            * np.sqrt(
                np.sum([1.0 / np.min(d) ** 2 for d in (self.dx, self.dy, self.dz)])
            )
        )
        if not material.is_stable(self.dt, courant):
            raise ValueError(
                f"dt={self.dt:.3e} s is unstable for {material!r}; pass a "
                "smaller `dt` or `cfln` to SolverFIT3D (see GainMaterial.is_stable)."
            )

    def _add_gain_edges(self, material, idx, weight, eps_bg=None):
        """Register flat E indices ``idx`` with fill fractions ``weight`` for a gain material."""
        self._check_gain_supported_and_stable(material)

        ieps = np.asarray(self.ieps.array)
        keep = (weight > 0.0) & (ieps[idx] != 0.0)  # skip PEC edges
        idx, weight = idx[keep], weight[keep]
        if idx.size == 0:
            return

        if not hasattr(self, "_gain_groups"):
            self._gain_groups = []

        # Restrict any overlapping existing gain groups or passive dispersive groups
        for group in self._gain_groups:
            old = ~np.isin(group["idx_host"], idx)
            if not old.all():
                _restrict_gain_group(group, old)
        for group in getattr(self, "_dispersive_groups", ()):
            old = ~np.isin(group["idx_host"], idx)
            if not old.all():
                _restrict_group(group, old, self.E.xp)

        eps_s = 1.0 / (eps_0 * ieps[idx])
        bg = eps_s if eps_bg is None else eps_bg
        eps_e = (1.0 - weight) * bg + weight * material.eps_inf
        s_hat = 0.5 * weight * material.sigma * self.dt / eps_0

        cells, inv = np.unique(idx % self.N, return_inverse=True)
        n_cells = cells.size
        cell_w = np.bincount(inv, weights=weight, minlength=n_cells) / np.maximum(
            np.bincount(inv, minlength=n_cells), 1
        )
        zeros_e = np.zeros(idx.size, dtype=self.dtype)

        group = {
            "material": material,
            "idx_host": idx,
            "idx": idx,
            "weight": np.asarray(weight, dtype=self.dtype),
            "eps_s": np.asarray(eps_s, dtype=self.dtype),
            "a1": np.asarray(eps_e - s_hat, dtype=self.dtype),
            "den": np.asarray(eps_e + s_hat, dtype=self.dtype),
            "cells": cells,
            "inv": inv,
            "n_cells": n_cells,
            "cell_weight": cell_w,
            "eff_n_cells": float(np.sum(cell_w)),
        }

        if isinstance(material, FourLevelGain):
            alpha_a, xi_a, zeta_a0 = _lorentz_step_coeffs(
                material.omega_a,
                material.gamma_a,
                material.kappa_a / eps_0,
                self.dt,
            )
            group["kind"] = "four_level"
            group["lorentz_a"] = [
                alpha_a,
                xi_a,
                zeta_a0,
                zeros_e.copy(),
                zeros_e.copy(),
            ]
            if material.has_transition_b:
                alpha_b, xi_b, zeta_b0 = _lorentz_step_coeffs(
                    material.omega_b,
                    material.gamma_b,
                    material.kappa_b / eps_0,
                    self.dt,
                )
                group["lorentz_b"] = [
                    alpha_b,
                    xi_b,
                    zeta_b0,
                    zeros_e.copy(),
                    zeros_e.copy(),
                ]
            else:
                group["lorentz_b"] = None

            n_init = material.initial_population_vector()
            group["N"] = np.broadcast_to(n_init[:, None], (4, n_cells)).copy()
            group["W_p_cached"] = material.pump_rate
            group["A_rate"], group["B_rate"] = _rate_cn_matrices(
                material, self.dt, W_p=material.pump_rate
            )
        elif isinstance(material, SaturableGain):
            alpha, xi, zeta0 = _lorentz_step_coeffs(
                material.omega_0,
                material.gamma,
                material.delta_eps * material.omega_0**2,
                self.dt,
            )
            group["kind"] = "saturable"
            group["lorentz"] = [alpha, xi, zeta0, zeros_e.copy(), zeros_e.copy()]
            group["I"] = np.zeros(n_cells, dtype=float)
            if material.tau_I > 0.0:
                r = 0.5 * self.dt / material.tau_I
                group["kappa_I"] = (1.0 - r) / (1.0 + r)
                group["beta_I"] = (2.0 * r) / (1.0 + r)
            else:
                group["kappa_I"] = 0.0
                group["beta_I"] = 1.0

        self._gain_groups = [
            g for g in self._gain_groups if g["idx_host"].size > 0
        ] + [group]

    def _capture_gain_E(self):
        """Capture ``E^n`` on active gain edges prior to the ``E`` update."""
        groups = getattr(self, "_gain_groups", None)
        if not groups:
            return None
        return [self.E.array[g["idx"]].copy() for g in groups]

    def _update_gain_E(self, E_n):
        """
        Apply the local gain ADE polarization, ``E^(n+1)``, and population updates.

        Order within each step ``n``:
        1. Advance ``p^(n+1)`` from ``(p^n, p^(n-1), E^n, N^n)`` centred at ``t^n``.
        2. Advance ``E^(n+1)`` from ``E^n`` using ``(p^(n+1) - p^n) / dt`` centred
           at ``t^(n+1/2)``.
        3. Advance ``N^(n+1)`` (or intensity ``I^(n+1)``) from ``N^n`` centred at
           ``t^(n+1/2)`` using ``E^(n+1/2) = 0.5 * (E^(n+1) + E^n)`` and
           ``dP/dt|^(n+1/2) = eps_0 * (p^(n+1) - p^n) / dt``.
        """
        if E_n is None:
            return
        E = self.E.array
        dt = float(self.dt)

        for g, En in zip(self._gain_groups, E_n):
            w = g["weight"]
            mat = g["material"]
            inv = g["inv"]
            n_cells = g["n_cells"]

            # Recover (dt / eps_0) * (curl H - J_src + CPML/TFSF) = (E* - E^n) * eps_s
            acc = (E[g["idx"]] - En) * g["eps_s"] + g["a1"] * En

            if g["kind"] == "four_level":
                N = g["N"]
                alpha_a, xi_a, zeta_a0, pa, pa_old = g["lorentz_a"]
                dn12 = N[1, inv] - N[2, inv]
                pa_new = alpha_a * pa + xi_a * pa_old + (zeta_a0 * dn12) * En
                dpa = pa_new - pa
                acc -= w * dpa
                g["lorentz_a"][3], g["lorentz_a"][4] = pa_new, pa

                if g["lorentz_b"] is not None:
                    alpha_b, xi_b, zeta_b0, pb, pb_old = g["lorentz_b"]
                    dn03 = N[0, inv] - N[3, inv]
                    pb_new = alpha_b * pb + xi_b * pb_old + (zeta_b0 * dn03) * En
                    dpb = pb_new - pb
                    acc -= w * dpb
                    g["lorentz_b"][3], g["lorentz_b"][4] = pb_new, pb
                else:
                    dpb = None

                E_new = acc / g["den"]
                E[g["idx"]] = E_new

                # Half-step E^(n+1/2) and dP/dt|^(n+1/2) drive N^n -> N^(n+1)
                if mat.enable_stimulated:
                    e_half = 0.5 * (np.real(E_new) + np.real(En))
                    work_a = e_half * np.real(dpa) * (eps_0 / (dt * hbar * mat.omega_a))
                    if mat.spatial_averaging and n_cells > 1:
                        sa = np.full(
                            n_cells,
                            float(np.sum(w * work_a) / g["eff_n_cells"]),
                            dtype=float,
                        )
                    else:
                        sa = np.bincount(inv, weights=work_a, minlength=n_cells)
                    if dpb is not None:
                        work_b = e_half * np.real(dpb) * (
                            eps_0 / (dt * hbar * mat.omega_b)
                        )
                        if mat.spatial_averaging and n_cells > 1:
                            sb = np.full(
                                n_cells,
                                float(np.sum(w * work_b) / g["eff_n_cells"]),
                                dtype=float,
                            )
                        else:
                            sb = np.bincount(inv, weights=work_b, minlength=n_cells)
                    else:
                        sb = 0.0
                else:
                    sa = 0.0
                    sb = 0.0

                if not mat.pauli_exclusion:
                    if mat.pump_rate != g["W_p_cached"]:
                        g["A_rate"], g["B_rate"] = _rate_cn_matrices(
                            mat, dt, W_p=mat.pump_rate
                        )
                        g["W_p_cached"] = mat.pump_rate
                    a_rate, b_rate = g["A_rate"], g["B_rate"]
                    n_next = a_rate @ N
                    if mat.enable_stimulated:
                        n_next = n_next + (b_rate[:, 2:3] - b_rate[:, 1:2]) * sa
                        if dpb is not None:
                            n_next = n_next + (b_rate[:, 3:4] - b_rate[:, 0:1]) * sb
                    g["N"] = n_next
                else:
                    s_vec = np.zeros((4, n_cells), dtype=float)
                    if mat.enable_stimulated:
                        s_vec[1] -= sa
                        s_vec[2] += sa
                        if dpb is not None:
                            s_vec[0] -= sb
                            s_vec[3] += sb
                    g["N"] = _step_pauli_rates(mat, N, s_vec, dt, wp=mat.pump_rate)

            elif g["kind"] == "saturable":
                I_cell = g["I"]
                alpha, xi, zeta0, p, p_old = g["lorentz"]
                sat = (
                    1.0
                    if np.isinf(mat.I_sat)
                    else 1.0 / (1.0 + I_cell[inv] / mat.I_sat)
                )
                p_new = alpha * p + xi * p_old + (zeta0 * sat) * En
                dp = p_new - p
                acc -= w * dp
                g["lorentz"][3], g["lorentz"][4] = p_new, p

                E_new = acc / g["den"]
                E[g["idx"]] = E_new

                # Half-step intensity update centred at t^(n+1/2)
                e2_edge = 0.5 * (np.abs(E_new) ** 2 + np.abs(En) ** 2)
                e2_cell = np.bincount(inv, weights=e2_edge, minlength=n_cells)
                if mat.tau_I > 0.0:
                    # Cycle average of c * eps_0 * n_bg * E^2(t) equals 0.5 * c * eps_0 * n_bg * |E_0|^2
                    i_inst = c_light * eps_0 * mat.n_bg * e2_cell
                    g["I"] = g["kappa_I"] * I_cell + g["beta_I"] * i_inst
                else:
                    g["I"] = 0.5 * c_light * eps_0 * mat.n_bg * e2_cell

    def get_gain_populations(self, group=0, cell_average=False):
        """
        Return the level populations ``(N_0, N_1, N_2, N_3)`` [1/m^3] of a gain group.

        Parameters
        ----------
        group : int, optional
            Index of the registered gain group. Default ``0``.
        cell_average : bool, optional
            If True, return the spatial average ``(N0, N1, N2, N3)`` of shape
            ``(4,)`` across the gain region. If False (default), return the
            per-cell array of shape ``(4, n_cells)``.

        Returns
        -------
        N : ndarray
            Population densities [1/m^3].
        """
        groups = getattr(self, "_gain_groups", None)
        if not groups:
            raise RuntimeError("No gain material has been registered on this solver.")
        g = groups[group]
        if g["kind"] != "four_level":
            raise TypeError(f"Gain group {group} is not a FourLevelGain material.")
        if cell_average:
            return np.average(g["N"], axis=1, weights=g["cell_weight"]).copy()
        return g["N"].copy()

    def set_gain_populations(self, populations, group=0):
        """
        Set the level populations ``(N_0, N_1, N_2, N_3)`` [1/m^3] of a `FourLevelGain` group.
        """
        groups = getattr(self, "_gain_groups", None)
        if not groups:
            raise RuntimeError("No gain material has been registered on this solver.")
        g = groups[group]
        if g["kind"] != "four_level":
            raise TypeError(f"Gain group {group} is not a FourLevelGain material.")
        mat = g["material"]
        if isinstance(populations, str):
            mode = populations.lower()
            if mode == "ground":
                vec = np.array([mat.N_total, 0.0, 0.0, 0.0], dtype=float)
            elif mode == "steady_state":
                vec = mat.steady_state_populations()
            else:
                raise ValueError(f"Unknown populations mode {populations!r}.")
            g["N"][:] = vec[:, None]
            return
        arr = np.asarray(populations, dtype=float)
        if arr.shape == (4,):
            g["N"][:] = arr[:, None]
        elif arr.shape == g["N"].shape:
            g["N"][:] = arr
        else:
            raise ValueError(
                f"Expected populations of shape (4,) or {g['N'].shape}, got {arr.shape}."
            )

    def reset_gain_state(self):
        """Reset all auxiliary gain polarizations, intensities, and populations."""
        for g in getattr(self, "_gain_groups", ()):
            if g["kind"] == "four_level":
                g["lorentz_a"][3][:] = 0.0
                g["lorentz_a"][4][:] = 0.0
                if g["lorentz_b"] is not None:
                    g["lorentz_b"][3][:] = 0.0
                    g["lorentz_b"][4][:] = 0.0
                n_init = g["material"].initial_population_vector()
                g["N"][:] = n_init[:, None]
            elif g["kind"] == "saturable":
                g["lorentz"][3][:] = 0.0
                g["lorentz"][4][:] = 0.0
                g["I"][:] = 0.0


def _restrict_gain_group(group, keep):
    """Drop edges of a gain group where ``keep`` is False and rebuild cell mapping."""
    group["idx_host"] = group["idx_host"][keep]
    group["idx"] = group["idx"][keep]
    for key in ("weight", "eps_s", "a1", "den"):
        group[key] = group[key][keep]
    if group["idx_host"].size == 0:
        return
    # Recompute unique cells from the remaining flat indices
    # Note: idx_host % N is preserved by indexing into group["cells"][group["inv"][keep]]
    cell_of_kept = group["cells"][group["inv"][keep]]
    new_cells, new_inv = np.unique(cell_of_kept, return_inverse=True)
    old_cell_mask = np.isin(group["cells"], new_cells)
    group["cells"] = new_cells
    group["inv"] = new_inv
    group["n_cells"] = new_cells.size
    cell_w = np.bincount(
        new_inv, weights=group["weight"], minlength=new_cells.size
    ) / np.maximum(np.bincount(new_inv, minlength=new_cells.size), 1)
    group["cell_weight"] = cell_w
    group["eff_n_cells"] = float(np.sum(cell_w))
    if group["kind"] == "four_level":
        group["lorentz_a"][3] = group["lorentz_a"][3][keep]
        group["lorentz_a"][4] = group["lorentz_a"][4][keep]
        if group["lorentz_b"] is not None:
            group["lorentz_b"][3] = group["lorentz_b"][3][keep]
            group["lorentz_b"][4] = group["lorentz_b"][4][keep]
        group["N"] = group["N"][:, old_cell_mask]
    elif group["kind"] == "saturable":
        group["lorentz"][3] = group["lorentz"][3][keep]
        group["lorentz"][4] = group["lorentz"][4][keep]
        group["I"] = group["I"][old_cell_mask]
