# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Semiconductor active gain media and diode-laser rate-equation models for `SolverFIT3D`.

Provides `SemiconductorGain`, a carrier-density-driven active material coupled to the
3D Finite Integration Technique (FIT) time-domain solver via `GainMixin` (`wakis.gain`):

* **Carrier rate equation** (Coldren, Corzine & Mašanović, *Diode Lasers and Photonic
  Integrated Circuits*, 2nd ed., Wiley, 2012, Chs. 2, 4, 5, doi:10.1002/9781118148167)::

      dN/dt = eta_i J(t) / (q d) - (A N + B N^2 + C N^3)
              + (1 / (hbar omega_0)) E . dP_gain/dt + D nabla^2 N

  where ``N(r, t)`` [1/m^3] is the electron-hole pair (carrier) density, ``J(t)``
  [A/m^2] is the injection current density across an active layer of thickness ``d``
  [m] with internal injection efficiency ``eta_i`` [-], ``A`` [1/s] is the
  Shockley-Read-Hall (SRH) nonradiative recombination coefficient, ``B`` [m^3/s] is
  the bimolecular radiative recombination coefficient, ``C`` [m^6/s] is the Auger
  recombination coefficient, ``-(1 / (hbar omega_0)) E . dP_gain/dt`` is the
  stimulated emission rate density driven by the electromagnetic field work, and
  ``D`` [m^2/s] is the optional ambipolar carrier diffusion coefficient along the
  active layer.

* **Gain spectrum and ADE polarization poles**: One or more Lorentzian auxiliary
  differential equation (ADE) poles ``P_p`` whose line-center power gain depends on
  ``N`` through either the linear law ``g_peak(N) = g' (N - N_tr)`` or the
  logarithmic law ``g_peak(N) = g0 ln((N + N_s) / (N_tr + N_s))``, with an optional
  bandfilling peak shift ``omega_0(N) = omega_0 + (domega_0/dN) (N - N_tr)``::

      d^2 P_p / dt^2 + gamma_p dP_p / dt + omega_p(N)^2 P_p
          = -eps_0 c n_bg gamma_p g_p(N) E

* **Henry linewidth enhancement factor** ``alpha_H`` (C. H. Henry, *IEEE J. Quantum
  Electron.* **18**(2), 259-264, 1982, doi:10.1109/JQE.1982.1071522; T. L. Koch and
  J. E. Bowers, "Nature of wavelength chirping in directly modulated semiconductor
  lasers", *Electron. Lett.* **20**(25), 1038-1040, 1984, doi:10.1049/el:19840709):
  Implemented via a carrier-dependent real background
  susceptibility ``Delta chi_bg(N)`` such that at the lasing angular frequency
  ``omega_0`` in Wakis's ``exp(+i omega t)`` convention (where gain has
  ``Im chi > 0`` and carrier-induced index depression has ``d(Re chi)/dN < 0``)::

      d(Re chi) / dN = -alpha_H * d(Im chi) / dN   (at omega = omega_0).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.constants import c as c_light
from scipy.constants import e as q_e
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import hbar
from scipy.optimize import brentq

from .dispersive import DispersiveMaterial, Lorentz
from .gain import GainMaterial, _lorentz_step_coeffs, fabry_perot_threshold_gain


class SemiconductorGain(GainMaterial):
    """
    Carrier-density-driven semiconductor gain medium with Henry ``alpha_H`` factor.

    Couples the semiconductor carrier rate equation (SRH ``A``, radiative ``B``,
    Auger ``C`` recombination, current injection ``eta_i J / (q d)``, stimulated
    emission ``-(1 / (hbar omega_0)) E . dP/dt``, and optional 1D/2D/3D carrier
    diffusion ``D nabla^2 N``) to one or more Lorentzian ADE polarization poles
    and a carrier-dependent background susceptibility ``Delta chi_bg(N)`` in
    `SolverFIT3D`.

    Parameters
    ----------
    omega_0 : float
        Reference gain-peak resonance angular frequency [rad/s] at transparency,
        ``> 0``.
    gamma : float
        Homogeneous dephasing rate (Lorentzian FWHM linewidth) [rad/s] of the
        primary gain pole, ``> 0``.
    N_tr : float, optional
        Transparency carrier density [1/m^3], ``> 0``. Default ``1.0e24``.
    differential_gain : float, optional
        Differential material gain ``g' = dg/dN`` [m^2], ``>= 0``. Alias
        ``g_prime`` is also accepted. Default ``3.0e-20``.
    g0 : float, optional
        Logarithmic gain coefficient ``g_0`` [1/m] used when
        ``gain_model='logarithmic'``, where
        ``g_peak(N) = g_0 * ln((N + N_s) / (N_tr + N_s))``. If None, defaults to
        ``differential_gain * (N_tr + N_s)`` so that ``dg/dN|_{N_tr} = g'``.
    N_s : float, optional
        Logarithmic gain linearity offset carrier density [1/m^3], ``> -N_tr``.
        Default ``0.0``.
    gain_model : {'linear', 'logarithmic'}, optional
        Functional dependence of the peak material gain ``g_peak(N)`` [1/m] on
        carrier density ``N``:
        - ``'linear'`` (default): ``g_peak(N) = g' * (N - N_tr)``.
        - ``'logarithmic'``: ``g_peak(N) = g_0 * ln((N + N_s) / (N_tr + N_s))``.
    alpha_H : float, optional
        Henry linewidth enhancement factor ``alpha_H`` [-] at ``omega_0``,
        defined by ``d(Re chi)/dN = -alpha_H * d(Im chi)/dN`` in Wakis's
        ``exp(+i omega t)`` convention. Default ``0.0``.
    J : float or callable, optional
        Injection current density [A/m^2], or callable ``J(t)`` returning the
        current density [A/m^2] at time ``t`` [s]. Alias ``current_density`` is
        also accepted. Default ``0.0``.
    eta_i : float, optional
        Internal current injection efficiency [-], in ``[0, 1]``. Default ``1.0``.
    active_thickness : float, optional
        Active-layer thickness ``d`` [m] in the injection rate
        ``eta_i * J / (q * d)``, ``> 0``. Alias ``d_active`` is also accepted.
        Default ``1.0e-7`` (100 nm).
    A : float, optional
        Shockley-Read-Hall (SRH) monomolecular nonradiative recombination rate
        coefficient [1/s], ``>= 0``. If ``tau_n`` is given, ``A`` is set to
        ``1.0 / tau_n``. Default ``1.0e9``.
    B : float, optional
        Bimolecular radiative recombination coefficient [m^3/s], ``>= 0``.
        Default ``0.0``.
    C : float, optional
        Auger recombination coefficient [m^6/s], ``>= 0``. Default ``0.0``.
    tau_n : float, optional
        Carrier recombination lifetime [s], ``> 0``. When provided, sets
        ``A = 1.0 / tau_n``.
    diffusion_coeff : float, optional
        Ambipolar carrier diffusion coefficient ``D`` [m^2/s], ``>= 0``. Alias
        ``D`` is also accepted. Default ``0.0``.
    diffusion_axes : sequence of str, optional
        Grid axes along which the carrier diffusion Laplacian is applied, subset
        of ``('x', 'y', 'z')``. Default ``('x', 'y', 'z')``.
    domega_dN : float, optional
        Optional bandfilling shift of the gain peak angular frequency with
        carrier density, ``d(omega_0)/dN`` [rad m^3 / s], so that
        ``omega_0(N) = omega_0 + domega_dN * (N - N_tr)``. Default ``0.0``.
    eps_nl : float, optional
        Nonlinear gain compression factor ``eps_nl`` [m^3] in the photon density
        saturation ``g(N, S) = g(N) / (1 + eps_nl * S)`` for coupled rate-equation
        analyses, ``>= 0``. Default ``0.0``.
    extra_poles : sequence of dict, optional
        Optional additional Lorentzian ADE poles to shape a multi-pole gain
        spectrum. Each dict specifies ``'omega_0'`` [rad/s], ``'gamma'`` [rad/s],
        and either ``'weight'`` [-] (gain relative to the primary pole) or
        ``'g_prime'`` [m^2], plus optional ``'N_tr'`` [1/m^3] and
        ``'domega_dN'`` [rad m^3 / s].
    initial_carriers : float, ndarray, or {'steady_state', 'transparency', 'ground'}, optional
        Initial carrier density ``N(0)`` [1/m^3]. ``'steady_state'`` (default)
        initializes to the zero-field steady state ``steady_state_carriers(J(0))``
        when ``J(0) > 0`` (and ``N_tr`` when ``J(0) == 0`` and ``A == B == C == 0``);
        ``'transparency'`` initializes to ``N_tr``; ``'ground'`` initializes to
        ``0.0``; a float or 1D array sets ``N(0)`` explicitly.
    enable_stimulated : bool, optional
        If True (default), include the stimulated emission term
        ``-(1 / (hbar * omega_p)) * E . dP_p/dt`` in the carrier rate equation.
    spatial_averaging : bool, optional
        If False (default), each grid cell evolves its own local carrier density
        ``N(r, t)``. If True, the stimulated transition rate is spatially
        homogenized across the active region at each step.
    eps_inf : float, optional
        Background high-frequency relative permittivity ``n_bg^2`` [-] at
        transparency, ``> 0``. Default ``12.25`` (``n_bg = 3.5``).
    sigma : float, optional
        Host static electric conductivity [S/m] representing distributed internal
        cavity loss ``alpha_i = sigma / (c * n_bg * eps_0)``, ``>= 0``.
        Default ``0.0``.
    name : str, optional
        Optional material name.
    """

    _gain_kind = "semiconductor"

    def __init__(
        self,
        omega_0,
        gamma,
        N_tr=1.0e24,
        differential_gain=3.0e-20,
        *,
        g_prime=None,
        g0=None,
        N_s=0.0,
        gain_model="linear",
        alpha_H=0.0,
        J=0.0,
        current_density=None,
        eta_i=1.0,
        active_thickness=1.0e-7,
        d_active=None,
        A=1.0e9,
        B=0.0,
        C=0.0,
        tau_n=None,
        diffusion_coeff=0.0,
        D=None,
        diffusion_axes=("x", "y", "z"),
        domega_dN=0.0,
        eps_nl=0.0,
        extra_poles=(),
        initial_carriers="steady_state",
        enable_stimulated=True,
        spatial_averaging=False,
        eps_inf=12.25,
        sigma=0.0,
        name=None,
    ):
        super().__init__(eps_inf=eps_inf, sigma=sigma, name=name)
        if g_prime is not None:
            differential_gain = g_prime
        if current_density is not None:
            J = current_density
        if d_active is not None:
            active_thickness = d_active
        if D is not None:
            diffusion_coeff = D
        if tau_n is not None:
            tau_n = float(tau_n)
            if tau_n <= 0.0:
                raise ValueError(f"tau_n must be > 0, got {tau_n}.")
            A = 0.0 if np.isinf(tau_n) else 1.0 / tau_n

        self.omega_0 = float(omega_0)
        self.gamma = float(gamma)
        self.N_tr = float(N_tr)
        self.differential_gain = float(differential_gain)
        self.N_s = float(N_s)

        if self.omega_0 <= 0.0:
            raise ValueError(f"omega_0 must be > 0, got {self.omega_0}.")
        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be > 0, got {self.gamma}.")
        if self.N_tr <= 0.0:
            raise ValueError(f"N_tr must be > 0, got {self.N_tr}.")
        if self.differential_gain < 0.0:
            raise ValueError(
                f"differential_gain must be >= 0, got {self.differential_gain}."
            )
        if self.N_tr + self.N_s <= 0.0:
            raise ValueError(
                f"N_tr + N_s must be > 0, got N_tr={self.N_tr}, N_s={self.N_s}."
            )

        model = str(gain_model).lower()
        if model not in ("linear", "logarithmic"):
            raise ValueError(
                f"gain_model must be 'linear' or 'logarithmic', got {gain_model!r}."
            )
        self.gain_model = model
        if g0 is None:
            self.g0 = float(self.differential_gain * (self.N_tr + self.N_s))
        else:
            self.g0 = float(g0)
            if self.g0 < 0.0:
                raise ValueError(f"g0 must be >= 0, got {self.g0}.")
            if g_prime is None and model == "logarithmic":
                self.differential_gain = float(self.g0 / (self.N_tr + self.N_s))

        self.alpha_H = float(alpha_H)
        self.J = J if callable(J) else float(J)
        self.eta_i = float(eta_i)
        self.active_thickness = float(active_thickness)
        self.A = float(A)
        self.B = float(B)
        self.C = float(C)
        self.diffusion_coeff = float(diffusion_coeff)
        self.domega_dN = float(domega_dN)
        self.eps_nl = float(eps_nl)
        self.enable_stimulated = bool(enable_stimulated)
        self.spatial_averaging = bool(spatial_averaging)

        if not (0.0 <= self.eta_i <= 1.0):
            raise ValueError(f"eta_i must be in [0, 1], got {self.eta_i}.")
        if self.active_thickness <= 0.0:
            raise ValueError(
                f"active_thickness must be > 0, got {self.active_thickness}."
            )
        if self.A < 0.0 or self.B < 0.0 or self.C < 0.0:
            raise ValueError("Recombination coefficients A, B, C must be >= 0.")
        if self.diffusion_coeff < 0.0:
            raise ValueError(
                f"diffusion_coeff must be >= 0, got {self.diffusion_coeff}."
            )
        if self.eps_nl < 0.0:
            raise ValueError(f"eps_nl must be >= 0, got {self.eps_nl}.")

        axes = tuple(str(a).lower() for a in diffusion_axes)
        if any(a not in ("x", "y", "z") for a in axes):
            raise ValueError(
                f"diffusion_axes must be a subset of ('x', 'y', 'z'), got {diffusion_axes!r}."
            )
        self.diffusion_axes = axes

        self.poles = [
            {
                "omega_0": self.omega_0,
                "gamma": self.gamma,
                "weight": 1.0,
                "g_prime": None,
                "N_tr": self.N_tr,
                "domega_dN": self.domega_dN,
            }
        ]
        for ep in extra_poles:
            w_p = float(ep["omega_0"])
            gam_p = float(ep["gamma"])
            if w_p <= 0.0 or gam_p <= 0.0:
                raise ValueError("Each extra pole must have omega_0 > 0 and gamma > 0.")
            self.poles.append(
                {
                    "omega_0": w_p,
                    "gamma": gam_p,
                    "weight": float(ep.get("weight", 1.0)),
                    "g_prime": (
                        None if ep.get("g_prime") is None else float(ep["g_prime"])
                    ),
                    "N_tr": float(ep.get("N_tr", self.N_tr)),
                    "domega_dN": float(ep.get("domega_dN", self.domega_dN)),
                }
            )

        if isinstance(initial_carriers, str):
            mode = initial_carriers.lower()
            if mode not in ("steady_state", "transparency", "ground"):
                raise ValueError(
                    "initial_carriers must be 'steady_state', 'transparency', "
                    "'ground', or a non-negative float/array."
                )
            self._init_mode = mode
            self._init_value = None
        else:
            arr = np.asarray(initial_carriers, dtype=float)
            if np.any(arr < 0.0):
                raise ValueError("initial_carriers must be >= 0.")
            self._init_mode = "explicit"
            self._init_value = float(arr) if arr.ndim == 0 else arr.copy()

    @property
    def g_prime(self):
        """Differential gain ``differential_gain`` [m^2]."""
        return self.differential_gain

    @property
    def d_active(self):
        """Active-layer thickness ``active_thickness`` [m]."""
        return self.active_thickness

    @property
    def D(self):
        """Ambipolar carrier diffusion coefficient ``diffusion_coeff`` [m^2/s]."""
        return self.diffusion_coeff

    @property
    def v_g(self):
        """Background group/phase velocity ``c / sqrt(eps_inf)`` [m/s]."""
        return float(c_light / self.n_bg)

    @property
    def alpha_internal(self):
        """Host internal power loss coefficient ``sigma / (c * n_bg * eps_0)`` [1/m]."""
        return float(self.sigma / (c_light * self.n_bg * eps_0))

    def current_density_at(self, t=0.0):
        """Return the injection current density ``J(t)`` [A/m^2] at time ``t`` [s]."""
        if callable(self.J):
            return float(self.J(float(t)))
        return float(self.J)

    def injection_rate(self, J=None, t=0.0):
        """
        Carrier injection rate density ``R_inj = eta_i * J / (q * d)`` [1/(m^3 s)].

        Parameters
        ----------
        J : float or callable, optional
            Current density [A/m^2]. Defaults to ``self.J``.
        t : float, optional
            Time [s] used when ``J`` is callable. Default ``0.0``.

        Returns
        -------
        R_inj : float
            Carrier injection rate density [1/(m^3 s)].
        """
        if J is None:
            j_val = self.current_density_at(t)
        elif callable(J):
            j_val = float(J(float(t)))
        else:
            j_val = float(J)
        return float(self.eta_i * j_val / (q_e * self.active_thickness))

    def recombination_rate(self, N):
        """
        Spontaneous and nonradiative recombination rate ``A N + B N^2 + C N^3`` [1/(m^3 s)].
        """
        n_arr = np.asarray(N, dtype=float)
        scalar = n_arr.ndim == 0
        rate = self.A * n_arr + self.B * n_arr**2 + self.C * n_arr**3
        return float(rate) if scalar else rate

    def differential_lifetime(self, N):
        """
        Differential carrier lifetime ``tau_d(N) = (A + 2 B N + 3 C N^2)^(-1)`` [s].
        """
        n_arr = np.asarray(N, dtype=float)
        scalar = n_arr.ndim == 0
        rate_deriv = self.A + 2.0 * self.B * n_arr + 3.0 * self.C * n_arr**2
        with np.errstate(divide="ignore"):
            tau = np.where(rate_deriv > 0.0, 1.0 / rate_deriv, np.inf)
        return float(tau) if scalar else tau

    def steady_state_carriers(self, J=None, t=0.0):
        """
        Zero-field steady-state carrier density ``N_0(J)`` [1/m^3].

        Solves ``A N + B N^2 + C N^3 = eta_i J / (q d)`` for ``N >= 0``.
        """
        r_inj = self.injection_rate(J=J, t=t)
        if r_inj <= 0.0:
            return 0.0
        if self.A == 0.0 and self.B == 0.0 and self.C == 0.0:
            return float(self.N_tr)
        if self.B == 0.0 and self.C == 0.0:
            return float(r_inj / self.A)
        if self.C == 0.0:
            # A N + B N^2 = r_inj
            return float(
                (-self.A + np.sqrt(self.A**2 + 4.0 * self.B * r_inj)) / (2.0 * self.B)
            )
        # General cubic A N + B N^2 + C N^3 = r_inj (monotone for N >= 0)
        n_hi = max(
            (r_inj / self.C) ** (1.0 / 3.0),
            r_inj / self.A if self.A > 0.0 else 0.0,
            1.0,
        )
        while self.recombination_rate(n_hi) < r_inj:
            n_hi *= 2.0
        return float(
            brentq(
                lambda n: self.recombination_rate(n) - r_inj,
                0.0,
                n_hi,
                rtol=1e-14,
            )
        )

    def initial_carrier_density(self, n_cells=None):
        """Return initial carrier density ``N(0)`` [1/m^3] as scalar or array of length ``n_cells``."""
        if self._init_mode == "ground":
            val = 0.0
        elif self._init_mode == "transparency":
            val = self.N_tr
        elif self._init_mode == "steady_state":
            j0 = self.current_density_at(0.0)
            if j0 == 0.0 and self.A == 0.0 and self.B == 0.0 and self.C == 0.0:
                val = self.N_tr
            else:
                val = self.steady_state_carriers(J=j0)
        else:
            val = self._init_value

        if n_cells is None:
            return float(np.mean(val)) if np.ndim(val) > 0 else float(val)
        if np.ndim(val) == 0:
            return np.full(int(n_cells), float(val), dtype=float)
        arr = np.asarray(val, dtype=float)
        if arr.shape != (int(n_cells),):
            raise ValueError(
                f"initial_carriers array shape {arr.shape} does not match n_cells={n_cells}."
            )
        return arr.copy()

    def _effective_carriers(self, N=None, J=None):
        if N is not None:
            return np.asarray(N, dtype=float)
        if J is not None:
            return np.asarray(self.steady_state_carriers(J=J), dtype=float)
        if self._init_mode == "explicit":
            return np.asarray(self._init_value, dtype=float)
        if self._init_mode == "transparency":
            return np.asarray(self.N_tr, dtype=float)
        if self._init_mode == "ground" and self.current_density_at(0.0) == 0.0:
            return np.asarray(0.0, dtype=float)
        return np.asarray(self.steady_state_carriers(), dtype=float)

    def _pole_gain(self, pole, N):
        """Evaluate line-center power gain ``g_p(N)`` [1/m] of ``pole`` at carrier density ``N``."""
        n_arr = np.asarray(N, dtype=float)
        ntr = pole["N_tr"]
        gp = pole["g_prime"]
        w = pole["weight"]
        if self.gain_model == "linear":
            slope = self.differential_gain if gp is None else gp
            return w * slope * (n_arr - ntr)
        # Logarithmic gain model: g_0 * ln((N + N_s) / (N_tr + N_s))
        g0_p = self.g0 if gp is None else gp * (ntr + self.N_s)
        n_floor = max(1.0e-6 * (ntr + self.N_s), 1.0)
        num = np.maximum(n_arr + self.N_s, n_floor)
        return w * g0_p * np.log(num / (ntr + self.N_s))

    def _pole_omega(self, pole, N):
        """Evaluate carrier-dependent resonance angular frequency ``omega_p(N)`` [rad/s]."""
        beta = pole["domega_dN"]
        if beta == 0.0:
            return pole["omega_0"]
        return pole["omega_0"] + beta * (np.asarray(N, dtype=float) - pole["N_tr"])

    def peak_gain(self, N=None, J=None):
        """
        Peak material power gain coefficient ``g_peak(N)`` [1/m] of the primary pole.
        """
        n_arr = self._effective_carriers(N=N, J=J)
        scalar = n_arr.ndim == 0
        g_val = self._pole_gain(self.poles[0], n_arr)
        return float(g_val) if scalar else g_val

    def differential_gain_at(self, N=None):
        """
        Differential gain ``dg_peak / dN`` [m^2] at carrier density ``N`` [1/m^3].
        """
        if self.gain_model == "linear":
            return float(self.differential_gain)
        n_arr = self._effective_carriers(N=N)
        scalar = n_arr.ndim == 0
        n_floor = max(1.0e-6 * (self.N_tr + self.N_s), 1.0)
        dg = self.g0 / np.maximum(n_arr + self.N_s, n_floor)
        return float(dg) if scalar else dg

    def omega_peak(self, N=None):
        """
        Carrier-dependent gain-peak angular frequency ``omega_0(N)`` [rad/s].
        """
        n_arr = self._effective_carriers(N=N)
        scalar = n_arr.ndim == 0
        w_val = self._pole_omega(self.poles[0], n_arr)
        return float(w_val) if scalar else w_val

    def _chi_poles(self, omega, N):
        """Resonant Lorentzian susceptibility sum ``sum_p chi_p(omega, N)``."""
        omega_arr = np.asarray(omega, dtype=float)
        n_arr = np.asarray(N, dtype=float)
        total = np.zeros(np.broadcast_shapes(omega_arr.shape, n_arr.shape), dtype=complex)
        for pole in self.poles:
            g_p = self._pole_gain(pole, n_arr)
            w_p = self._pole_omega(pole, n_arr)
            gam_p = pole["gamma"]
            strength = -c_light * self.n_bg * gam_p * g_p
            total = total + strength / (w_p**2 - omega_arr**2 + 1j * gam_p * omega_arr)
        return total

    def delta_chi_bg(self, N=None, J=None):
        """
        Real carrier-dependent background susceptibility ``Delta chi_bg(N)`` [-].

        Implements the Henry linewidth enhancement factor ``alpha_H`` so that at
        ``omega = omega_0`` in Wakis's ``exp(+i omega t)`` convention::

            Re chi(omega_0, N) = -alpha_H * Im chi(omega_0, N),
            d(Re chi(omega_0, N)) / dN = -alpha_H * d(Im chi(omega_0, N)) / dN.
        """
        n_arr = self._effective_carriers(N=N, J=J)
        scalar = n_arr.ndim == 0
        if (
            len(self.poles) == 1
            and self.domega_dN == 0.0
            and self.alpha_H == 0.0
        ):
            return 0.0 if scalar else np.zeros_like(n_arr, dtype=float)
        chi_w0 = self._chi_poles(self.omega_0, n_arr)
        dchi = -self.alpha_H * np.imag(chi_w0) - np.real(chi_w0)
        return float(dchi) if scalar else dchi

    def chi(self, omega, N=None, J=None):
        """
        Continuous complex electric susceptibility ``chi(omega, N)`` in ``exp(+i w t)`` convention.

        Parameters
        ----------
        omega : float or array_like
            Angular frequency [rad/s].
        N : float or array_like, optional
            Carrier density [1/m^3]. Defaults to the steady-state or initial
            carrier density.
        J : float, optional
            Current density [A/m^2] used when ``N`` is None.

        Returns
        -------
        chi : complex or complex ndarray
            Susceptibility ``Delta chi_bg(N) + sum_p chi_p(omega, N)`` [-].
        """
        omega_arr = np.asarray(omega, dtype=float)
        n_arr = self._effective_carriers(N=N, J=J)
        scalar = omega_arr.ndim == 0 and n_arr.ndim == 0
        val = self.delta_chi_bg(N=n_arr) + self._chi_poles(omega_arr, n_arr)
        return complex(val) if scalar else val

    susceptibility = chi

    def chi_discrete(self, omega, dt, N=None, J=None):
        """
        Discrete ADE electric susceptibility ``chi_d(omega, dt, N)`` at timestep ``dt`` [s].
        """
        omega_arr = np.asarray(omega, dtype=float)
        n_arr = self._effective_carriers(N=N, J=J)
        scalar = omega_arr.ndim == 0 and n_arr.ndim == 0
        dt = float(dt)
        w_tilde2 = (2.0 / dt * np.sin(omega_arr * dt / 2.0)) ** 2
        sin_wdt = np.sin(omega_arr * dt) / dt
        total = np.zeros(
            np.broadcast_shapes(omega_arr.shape, n_arr.shape), dtype=complex
        )
        for pole in self.poles:
            g_p = self._pole_gain(pole, n_arr)
            w_p = self._pole_omega(pole, n_arr)
            gam_p = pole["gamma"]
            strength = -c_light * self.n_bg * gam_p * g_p
            total = total + strength / (w_p**2 - w_tilde2 + 1j * gam_p * sin_wdt)
        val = self.delta_chi_bg(N=n_arr) + total
        return complex(val) if scalar else val

    def epsilon(self, f, N=None, J=None):
        """
        Complex relative permittivity ``eps_r(f, N)`` in ``exp(+i w t)`` convention.
        """
        f_arr = np.asarray(f, dtype=float)
        omega = 2.0 * np.pi * f_arr
        eps = self.eps_inf + self.chi(omega, N=N, J=J)
        if self.sigma:
            eps = eps + self.sigma / (1j * omega * eps_0)
        return eps

    def epsilon_discrete(self, f, dt, N=None, J=None):
        """
        Discrete complex relative permittivity ``eps_d(f, dt, N)`` at timestep ``dt`` [s].
        """
        f_arr = np.asarray(f, dtype=float)
        omega = 2.0 * np.pi * f_arr
        dt = float(dt)
        eps = self.eps_inf + self.chi_discrete(omega, dt, N=N, J=J)
        if self.sigma:
            w_tilde = 2.0 / dt * np.sin(omega * dt / 2.0)
            eps = eps + self.sigma * np.cos(omega * dt / 2.0) / (1j * w_tilde * eps_0)
        return eps

    def refractive_index(self, f=None, N=None, J=None, dt=None):
        """
        Complex refractive index ``n_c(f, N) = n_r + i n_i`` with ``n_r > 0``.

        In Wakis's ``exp(+i w t - i k z)`` convention, ``n_i > 0`` corresponds to
        amplification (gain) and ``n_i < 0`` to absorption.
        """
        if f is None:
            f_arr = np.asarray(self.omega_0 / (2.0 * np.pi), dtype=float)
        else:
            f_arr = np.asarray(f, dtype=float)
        eps = (
            self.epsilon(f_arr, N=N, J=J)
            if dt is None
            else self.epsilon_discrete(f_arr, dt, N=N, J=J)
        )
        n_c = np.sqrt(np.asarray(eps, dtype=complex))
        n_c = np.where(n_c.real < 0.0, -n_c, n_c)
        return complex(n_c) if np.ndim(n_c) == 0 else n_c

    def small_signal_gain(self, f=None, N=None, J=None, dt=None):
        """
        Small-signal power gain coefficient ``g(f, N) = (4 pi f / c) Im n_c(f, N)`` [1/m].

         Excludes host conductivity ``sigma`` so ``small_signal_gain`` returns the
        active material gain ``g(f, N)`` (positive above transparency, negative
        below transparency).
        """
        if f is None:
            f_arr = np.asarray(self.omega_0 / (2.0 * np.pi), dtype=float)
        else:
            f_arr = np.asarray(f, dtype=float)
        omega = 2.0 * np.pi * f_arr
        chi_val = (
            self.chi(omega, N=N, J=J)
            if dt is None
            else self.chi_discrete(omega, dt, N=N, J=J)
        )
        n_c = np.sqrt(np.asarray(self.eps_inf + chi_val, dtype=complex))
        n_c = np.where(n_c.real < 0.0, -n_c, n_c)
        g = 2.0 * (omega / c_light) * n_c.imag
        return float(g) if np.ndim(g) == 0 else g

    def henry_factor(self, f=None, N=None, dN=None):
        """
        Evaluate ``- (d Re chi / dN) / (d Im chi / dN)`` at frequency ``f`` [Hz].

        At the reference lasing frequency ``f_0 = omega_0 / (2 pi)``, equals ``alpha_H``.
        """
        omega = self.omega_0 if f is None else 2.0 * np.pi * float(f)
        n0 = float(self._effective_carriers(N=N))
        dn = max(1.0e-5 * self.N_tr, 1.0e18) if dN is None else float(dN)
        chi_plus = self.chi(omega, N=n0 + 0.5 * dn)
        chi_minus = self.chi(omega, N=n0 - 0.5 * dn)
        d_chi = (chi_plus - chi_minus) / dn
        return float(-np.real(d_chi) / np.imag(d_chi))

    def threshold_carriers(self, g_th):
        """
        Threshold carrier density ``N_th`` [1/m^3] required for peak material gain ``g_th`` [1/m].
        """
        g_th = float(g_th)
        if g_th < 0.0:
            raise ValueError(f"g_th must be >= 0, got {g_th}.")
        if self.gain_model == "linear":
            if self.differential_gain <= 0.0:
                raise ValueError("Cannot reach positive threshold gain with differential_gain == 0.")
            return float(self.N_tr + g_th / self.differential_gain)
        if self.g0 <= 0.0:
            raise ValueError("Cannot reach positive threshold gain with g0 == 0.")
        return float((self.N_tr + self.N_s) * np.exp(g_th / self.g0) - self.N_s)

    def threshold_current_density(self, g_th):
        """
        Threshold injection current density ``J_th`` [A/m^2] for threshold gain ``g_th`` [1/m].

        Evaluates ``J_th = (q * d / eta_i) * (A N_th + B N_th^2 + C N_th^3)``.
        """
        if self.eta_i <= 0.0:
            raise ValueError("Cannot reach threshold when eta_i == 0.")
        n_th = self.threshold_carriers(g_th)
        r_th = self.recombination_rate(n_th)
        return float((q_e * self.active_thickness / self.eta_i) * r_th)

    def slope_efficiency(self, alpha_m, alpha_i=None):
        """
        Differential quantum efficiency ``eta_d = eta_i * alpha_m / (alpha_i + alpha_m)`` [-].

        Parameters
        ----------
        alpha_m : float
            Mirror loss coefficient ``ln(1 / (R1 * R2)) / (2 * L)`` [1/m], ``> 0``.
        alpha_i : float, optional
            Internal modal loss coefficient [1/m], ``>= 0``. Defaults to
            `alpha_internal`.

        Returns
        -------
        eta_d : float
            Differential quantum efficiency [-].
        """
        am = float(alpha_m)
        ai = self.alpha_internal if alpha_i is None else float(alpha_i)
        if am <= 0.0 or ai < 0.0:
            raise ValueError(f"Expected alpha_m > 0 and alpha_i >= 0, got {am}, {ai}.")
        return float(self.eta_i * am / (ai + am))

    def steady_state_laser(
        self,
        J=None,
        length=1.0e-6,
        R1=0.8,
        R2=None,
        alpha_i=None,
        confinement=1.0,
        spatial_hole_burning=False,
        L_eff=None,
    ):
        """
        Analytic steady-state Fabry-Perot diode laser operating point and dynamics.

        Parameters
        ----------
        J : float or array_like, optional
            Injection current density [A/m^2]. Defaults to ``self.J(0)``.
        length : float, optional
            Active gain section length ``L`` [m], ``> 0``.
        R1, R2 : float, optional
            Facet power reflectances in ``(0, 1]``. ``R2`` defaults to ``R1``.
        alpha_i : float, optional
            Internal cavity loss coefficient [1/m]. Defaults to `alpha_internal`.
        confinement : float, optional
            Optical confinement factor ``Gamma`` in ``(0, 1]``. Default ``1.0``.
        spatial_hole_burning : bool, optional
            If True, apply the standing-wave spatial hole burning factor
            ``s(r) / (r - 1)`` to the above-threshold photon density and output
            intensity. Default False (carrier diffusion / homogenized limit).
        L_eff : float, optional
            Effective cavity length ``L_eff >= L`` [m] including distributed-
            Bragg-reflector (DBR) phase penetration ``L_eff = L + 2 * L_pen``.
            When provided, the longitudinal confinement factor
            ``Gamma_z = L / L_eff`` scales the cold-cavity photon lifetime
            ``tau_p_eff = tau_p / Gamma_z`` and the small-signal relaxation
            oscillation frequency ``omega_R = sqrt(Gamma_z) * omega_R_bare``
            while preserving threshold gain ``g_th`` and steady-state output.

        Returns
        -------
        res : dict
            Dictionary containing:
            - ``'alpha_m'``: mirror loss coefficient [1/m]
            - ``'alpha_i'``: internal loss coefficient [1/m]
            - ``'g_th'``: threshold material gain [1/m]
            - ``'N_th'``: threshold carrier density [1/m^3]
            - ``'J_th'``: threshold current density [A/m^2]
            - ``'tau_p'``: bare active-cavity photon lifetime [s]
            - ``'tau_p_eff'``: effective photon lifetime including ``L_eff`` [s]
            - ``'gamma_z'``: longitudinal confinement factor ``L / L_eff`` [-]
            - ``'eta_d'``: differential quantum efficiency [-]
            - ``'responsivity_W_per_A'``: total slope ``(hbar * omega_0 / q) * eta_d`` [W/A]
            - ``'N_ss'``: steady-state carrier density [1/m^3]
            - ``'S_ss'``: steady-state intracavity photon density [1/m^3]
            - ``'I_out_total'``: total two-facet output optical intensity [W/m^2]
            - ``'I_out_1'``: facet-1 output optical intensity [W/m^2]
            - ``'I_out_2'``: facet-2 output optical intensity [W/m^2]
            - ``'omega_R'``: relaxation-oscillation angular frequency [rad/s]
            - ``'f_R'``: relaxation-oscillation frequency ``omega_R / (2 pi)`` [Hz]
            - ``'gamma_RO'``: relaxation-oscillation damping rate [1/s]
        """
        L = float(length)
        l_eff = L if L_eff is None else float(L_eff)
        if l_eff <= 0.0:
            raise ValueError(f"L_eff must be > 0, got {l_eff}.")
        gamma_z = L / l_eff
        r1 = float(R1)
        r2 = r1 if R2 is None else float(R2)
        ai = self.alpha_internal if alpha_i is None else float(alpha_i)
        gamma_c = float(confinement)
        alpha_m = float(np.log(1.0 / (r1 * r2)) / (2.0 * L))
        g_th = fabry_perot_threshold_gain(
            L, r1, r2, alpha_int=ai, confinement=gamma_c
        )
        n_th = self.threshold_carriers(g_th)
        j_th = self.threshold_current_density(g_th)
        tau_p = float(1.0 / (self.v_g * (ai + alpha_m)))
        tau_p_eff = float(tau_p / gamma_z)
        eta_d = self.slope_efficiency(alpha_m, alpha_i=ai)

        j_in = self.current_density_at(0.0) if J is None else J
        j_arr = np.asarray(j_in, dtype=float)
        scalar = j_arr.ndim == 0
        j_1d = np.atleast_1d(j_arr)

        n_ss = np.empty_like(j_1d, dtype=float)
        s_ss = np.zeros_like(j_1d, dtype=float)
        for idx_j, jv in enumerate(j_1d):
            if jv <= j_th:
                n_ss[idx_j] = self.steady_state_carriers(J=float(jv))
            else:
                n_ss[idx_j] = n_th
                # Homogenized rate-equation steady state:
                # S_ss = Gamma * tau_p * eta_i * (J - J_th) / (q * d)
                s_hom = (
                    gamma_c
                    * tau_p
                    * self.eta_i
                    * (float(jv) - j_th)
                    / (q_e * self.active_thickness)
                )
                if spatial_hole_burning:
                    n0_unsat = self.steady_state_carriers(J=float(jv))
                    g0_unsat = float(self.peak_gain(N=n0_unsat))
                    r_pump = max(g0_unsat / g_th, 1.0)
                    if r_pump > 1.0:
                        s_shb = 0.25 * (
                            4.0 * r_pump - 1.0 - np.sqrt(8.0 * r_pump + 1.0)
                        )
                        s_ss[idx_j] = s_hom * (s_shb / (r_pump - 1.0))
                        n_ss[idx_j] = self.N_tr + (n0_unsat - self.N_tr) / np.sqrt(
                            1.0 + 2.0 * s_shb
                        )
                else:
                    s_ss[idx_j] = s_hom

        # Total intensity exiting both mirrors: hbar * omega_0 * v_g * alpha_m * (L / Gamma) * S_ss
        i_tot = hbar * self.omega_0 * self.v_g * alpha_m * (L / gamma_c) * s_ss
        t1 = 1.0 - r1
        t2 = 1.0 - r2
        denom = t1 * np.sqrt(r2) + t2 * np.sqrt(r1)
        frac_1 = (t1 * np.sqrt(r2) / denom) if denom > 0.0 else 0.5
        i_1 = frac_1 * i_tot
        i_2 = (1.0 - frac_1) * i_tot

        # Small-signal relaxation oscillation frequency (with eps_nl = 0):
        # omega_R^2 = Gamma_z * Gamma * v_g * g'(N_th) * S_ss / tau_p
        gp_th = self.differential_gain_at(N=n_th)
        omega_r = np.sqrt(gamma_z * gamma_c * self.v_g * gp_th * s_ss / tau_p)
        f_r = omega_r / (2.0 * np.pi)
        tau_d_th = self.differential_lifetime(n_th)
        gamma_ro = (0.0 if np.isinf(tau_d_th) else 1.0 / tau_d_th) + (
            self.v_g * gp_th * s_ss
        )

        def _unwrap(arr):
            return float(arr[0]) if scalar else arr.reshape(j_arr.shape)

        return {
            "alpha_m": alpha_m,
            "alpha_i": ai,
            "g_th": g_th,
            "N_th": n_th,
            "J_th": j_th,
            "tau_p": tau_p,
            "tau_p_eff": tau_p_eff,
            "gamma_z": gamma_z,
            "eta_d": eta_d,
            "responsivity_W_per_A": float((hbar * self.omega_0 / q_e) * eta_d),
            "N_ss": _unwrap(n_ss),
            "S_ss": _unwrap(s_ss),
            "I_out_total": _unwrap(i_tot),
            "I_out_1": _unwrap(i_1),
            "I_out_2": _unwrap(i_2),
            "omega_R": _unwrap(omega_r),
            "f_R": _unwrap(f_r),
            "gamma_RO": _unwrap(gamma_ro),
        }

    def modulation_transfer_function(
        self,
        f_mod,
        J=None,
        length=1.0e-6,
        R1=0.8,
        R2=None,
        alpha_i=None,
        confinement=1.0,
    ):
        """
        Small-signal intensity modulation transfer function ``H(f_mod)`` [-].

        With gain compression ``eps_nl = 0``, linearizing the coupled carrier and
        photon rate equations around the steady state ``(N_th, S_0)`` gives::

            H(omega_m) = omega_R^2 / (omega_R^2 - omega_m^2 + i * gamma_RO * omega_m)

        where ``omega_R^2 = Gamma * v_g * g' * S_0 / tau_p`` and
        ``gamma_RO = 1 / tau_d(N_th) + v_g * g' * S_0``.
        """
        ss = self.steady_state_laser(
            J=J,
            length=length,
            R1=R1,
            R2=R2,
            alpha_i=alpha_i,
            confinement=confinement,
        )
        f_arr = np.asarray(f_mod, dtype=float)
        scalar = f_arr.ndim == 0
        wm = 2.0 * np.pi * f_arr
        wr2 = ss["omega_R"] ** 2
        gro = ss["gamma_RO"]
        h_val = wr2 / (wr2 - wm**2 + 1j * gro * wm)
        return complex(h_val) if scalar else h_val

    def is_stable(self, dt, courant=1.0):
        """
        Check whether timestep ``dt`` [s] resolves the semiconductor gain poles.
        """
        dt = float(dt)
        if dt <= 0.0:
            return False
        n_ref = max(
            self.N_tr,
            self.initial_carrier_density(),
            self.steady_state_carriers(),
            2.0 * self.N_tr,
        )
        poles_lorentz = []
        for pole in self.poles:
            w_p = max(
                float(pole["omega_0"]),
                float(abs(self._pole_omega(pole, n_ref))),
            )
            if w_p * dt >= 2.0:
                return False
            g_mag = max(
                abs(float(self._pole_gain(pole, 0.0))),
                abs(float(self._pole_gain(pole, n_ref))),
            )
            if g_mag > 0.0:
                de_p = c_light * self.n_bg * pole["gamma"] * g_mag / (w_p**2)
                poles_lorentz.append(Lorentz(de_p, w_p, pole["gamma"]))
        if not poles_lorentz:
            return True
        passive = DispersiveMaterial(self.eps_inf, poles_lorentz, sigma=self.sigma)
        return passive.is_stable(dt, courant=courant)

    # -----------------------------------------------------------------------
    # Standalone carrier and coupled diode-laser rate-equation integrators
    # -----------------------------------------------------------------------

    def _step_carrier_implicit_midpoint(
        self, n_curr, dt, r_inj_half, r_stim_half, diff_op=None, diff_solver=None
    ):
        """
        Advance ``N^n -> N^(n+1)`` via the second-order implicit midpoint rule at ``t^(n+1/2)``.

        Solves::

            (N^(n+1) - N^n) / dt
                = R_inj^(n+1/2) - R_rec(N^(n+1/2)) - R_stim^(n+1/2) + D nabla^2 N^(n+1/2)

        for ``N^(n+1/2) = 0.5 * (N^n + N^(n+1))``, returning ``(N^(n+1), R_rec(N^(n+1/2)))``.
        Because ``1^T nabla^2 = 0``, the discrete conservation identity
        ``sum V_m (N_m^(n+1) - N_m^n) = dt * sum V_m (R_inj - R_rec - R_stim)``
        holds to machine precision at every step.
        """
        rhs_const = n_curr + 0.5 * dt * (r_inj_half - r_stim_half)
        a_c, b_c, c_c = self.A, self.B, self.C

        if diff_op is None:
            if b_c == 0.0 and c_c == 0.0:
                n_half = rhs_const / (1.0 + 0.5 * dt * a_c)
            else:
                n_half = rhs_const / (
                    1.0 + 0.5 * dt * (a_c + b_c * n_curr + c_c * n_curr**2)
                )
                for _ in range(3):
                    rec = a_c * n_half + b_c * n_half**2 + c_c * n_half**3
                    drec = a_c + 2.0 * b_c * n_half + 3.0 * c_c * n_half**2
                    res = n_half + 0.5 * dt * rec - rhs_const
                    n_half = n_half - res / (1.0 + 0.5 * dt * drec)
        else:
            # Implicit Crank-Nicolson diffusion + Newton polish for nonlinear B, C
            n_half = diff_solver(rhs_const)
            if b_c != 0.0 or c_c != 0.0:
                for _ in range(3):
                    nl_rec = b_c * n_half**2 + c_c * n_half**3
                    n_half = diff_solver(rhs_const - 0.5 * dt * nl_rec)

        r_rec_half = a_c * n_half + b_c * n_half**2 + c_c * n_half**3
        diff_term = diff_op.dot(n_half) if diff_op is not None else 0.0
        n_next = n_curr + dt * (r_inj_half - r_rec_half - r_stim_half + diff_term)
        return n_next, r_rec_half

    def step_rate_equations(
        self, Nt, dt, N_init=None, J=None, dx=None, periodic=False
    ):
        """
        Integrate the standalone zero-field carrier rate equation for ``Nt`` steps of ``dt`` [s].

        Includes current injection ``eta_i J(t) / (q d)``, recombination
        ``A N + B N^2 + C N^3``, and optional 1D carrier diffusion ``D d^2N/dx^2``
        when ``N_init`` is a 1D spatial profile and ``dx`` [m] is provided.

        Parameters
        ----------
        Nt : int
            Number of timesteps.
        dt : float
            Timestep [s].
        N_init : float or array_like, optional
            Initial carrier density [1/m^3]. Defaults to `initial_carrier_density`.
        J : float or callable, optional
            Current density [A/m^2]. Defaults to ``self.J``.
        dx : float, optional
            Spatial grid spacing [m] for 1D carrier diffusion when ``N_init`` is
            a 1D array and ``self.diffusion_coeff > 0``.
        periodic : bool, optional
            If True, use periodic boundary conditions for 1D diffusion; if False
            (default), use zero-flux Neumann boundary conditions.

        Returns
        -------
        t : ndarray of shape (Nt + 1,)
            Time array [s].
        N_hist : ndarray of shape (Nt + 1,) or (Nt + 1, M)
            Carrier density history [1/m^3].
        """
        Nt = int(Nt)
        dt = float(dt)
        if N_init is None:
            n_curr = np.asarray(self.initial_carrier_density(), dtype=float)
        else:
            n_curr = np.asarray(N_init, dtype=float).copy()

        diff_op = None
        diff_solver = None
        if n_curr.ndim == 1 and n_curr.size > 1 and self.diffusion_coeff > 0.0 and dx is not None:
            m = n_curr.size
            h2 = float(dx) ** 2
            diag = np.full(m, -2.0 / h2)
            off = np.full(m - 1, 1.0 / h2)
            if not periodic:
                diag[0] = -1.0 / h2
                diag[-1] = -1.0 / h2
                lap = sp.diags([off, diag, off], [-1, 0, 1], shape=(m, m), format="csc")
            else:
                lap = sp.diags(
                    [off, diag, off, [1.0 / h2], [1.0 / h2]],
                    [-1, 0, 1, m - 1, -(m - 1)],
                    shape=(m, m),
                    format="csc",
                )
            diff_op = self.diffusion_coeff * lap
            lhs = (
                sp.eye(m, format="csc") * (1.0 + 0.5 * dt * self.A)
                - 0.5 * dt * diff_op
            )
            diff_solver = spla.factorized(lhs)

        hist = np.empty((Nt + 1,) + n_curr.shape, dtype=float)
        hist[0] = n_curr
        for n in range(Nt):
            t_half = (n + 0.5) * dt
            r_inj = self.injection_rate(J=J, t=t_half)
            n_curr, _ = self._step_carrier_implicit_midpoint(
                n_curr,
                dt,
                r_inj,
                0.0,
                diff_op=diff_op,
                diff_solver=diff_solver,
            )
            hist[n + 1] = n_curr
        return np.arange(Nt + 1) * dt, hist

    def step_laser_rate_equations(
        self,
        Nt,
        dt,
        J=None,
        length=1.0e-6,
        R1=0.8,
        R2=None,
        alpha_i=None,
        confinement=1.0,
        N_init=None,
        S_init=1.0e15,
        beta_sp=0.0,
    ):
        """
        Integrate the coupled carrier-photon-phase semiconductor laser rate equations.

        Integrates::

            dN/dt = eta_i J(t) / (q d) - (A N + B N^2 + C N^3)
                    - v_g g_peak(N) S / (1 + eps_nl S)
            dS/dt = (Gamma v_g g_peak(N) / (1 + eps_nl S) - 1 / tau_p) S
                    + beta_sp B N^2
            dphi/dt = 0.5 * alpha_H * Gamma * v_g * (g_peak(N) - g_th)

        using classic 4th-order Runge-Kutta in ``(N, ln S, phi)`` so that rapid
        many-decade photon turn-on transients and small-signal modulations are
        resolved accurately.

        Returns
        -------
        res : dict
            Dictionary with keys ``'t'``, ``'N'``, ``'S'``, ``'I_out_total'``,
            ``'I_out_1'``, ``'delta_nu'`` [Hz], ``'dlnP_dt'`` [1/s],
            ``'chirp_adiabatic'`` (``(alpha_H / (4 pi)) * d(ln P)/dt`` [Hz]), and
            ``'phi'`` [rad].
        """
        Nt = int(Nt)
        dt = float(dt)
        L = float(length)
        r1 = float(R1)
        r2 = r1 if R2 is None else float(R2)
        ai = self.alpha_internal if alpha_i is None else float(alpha_i)
        gamma_c = float(confinement)
        alpha_m = float(np.log(1.0 / (r1 * r2)) / (2.0 * L))
        g_th = (ai + alpha_m) / gamma_c
        tau_p = float(1.0 / (self.v_g * (ai + alpha_m)))

        n_val = (
            float(self.initial_carrier_density())
            if N_init is None
            else float(N_init)
        )
        u_val = float(np.log(max(float(S_init), 1.0e-30)))  # u = ln(S)
        phi_val = 0.0

        t_arr = np.arange(Nt + 1) * dt
        n_hist = np.empty(Nt + 1, dtype=float)
        s_hist = np.empty(Nt + 1, dtype=float)
        phi_hist = np.empty(Nt + 1, dtype=float)
        dnu_hist = np.empty(Nt + 1, dtype=float)
        dlnp_hist = np.empty(Nt + 1, dtype=float)

        def _rhs(t_cur, n_c, u_c):
            s_c = np.exp(u_c)
            r_inj = self.injection_rate(J=J, t=t_cur)
            r_rec = self.recombination_rate(n_c)
            g_n = float(self.peak_gain(N=n_c))
            g_sat = g_n / (1.0 + self.eps_nl * s_c)
            dn_dt = r_inj - r_rec - self.v_g * g_sat * s_c
            du_dt = (
                gamma_c * self.v_g * g_sat
                - 1.0 / tau_p
                + (beta_sp * self.B * n_c**2) / max(s_c, 1.0e-30)
            )
            dphi_dt = 0.5 * self.alpha_H * gamma_c * self.v_g * (g_n - g_th)
            return dn_dt, du_dt, dphi_dt

        for idx_t in range(Nt + 1):
            t_cur = t_arr[idx_t]
            n_hist[idx_t] = n_val
            s_hist[idx_t] = np.exp(u_val)
            phi_hist[idx_t] = phi_val
            _, du_cur, dphi_cur = _rhs(t_cur, n_val, u_val)
            dlnp_hist[idx_t] = du_cur
            dnu_hist[idx_t] = dphi_cur / (2.0 * np.pi)
            if idx_t == Nt:
                break

            k1_n, k1_u, k1_p = _rhs(t_cur, n_val, u_val)
            k2_n, k2_u, k2_p = _rhs(
                t_cur + 0.5 * dt, n_val + 0.5 * dt * k1_n, u_val + 0.5 * dt * k1_u
            )
            k3_n, k3_u, k3_p = _rhs(
                t_cur + 0.5 * dt, n_val + 0.5 * dt * k2_n, u_val + 0.5 * dt * k2_u
            )
            k4_n, k4_u, k4_p = _rhs(
                t_cur + dt, n_val + dt * k3_n, u_val + dt * k3_u
            )
            n_val += (dt / 6.0) * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n)
            u_val += (dt / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
            phi_val += (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)

        i_tot = hbar * self.omega_0 * self.v_g * alpha_m * (L / gamma_c) * s_hist
        t1 = 1.0 - r1
        t2 = 1.0 - r2
        denom = t1 * np.sqrt(r2) + t2 * np.sqrt(r1)
        frac_1 = (t1 * np.sqrt(r2) / denom) if denom > 0.0 else 0.5

        return {
            "t": t_arr,
            "N": n_hist,
            "S": s_hist,
            "I_out_total": i_tot,
            "I_out_1": frac_1 * i_tot,
            "phi": phi_hist,
            "delta_nu": dnu_hist,
            "dlnP_dt": dlnp_hist,
            "chirp_adiabatic": (self.alpha_H / (4.0 * np.pi)) * dlnp_hist,
        }

    # -----------------------------------------------------------------------
    # SolverFIT3D (GainMixin) registration and time-stepping hooks
    # -----------------------------------------------------------------------

    def _build_cell_diffusion_operator(self, cells, solver, dt, cell_vol=None):
        """Build finite-volume sparse 1D/2D/3D Laplacian ``D nabla^2`` on active ``cells``."""
        if self.diffusion_coeff <= 0.0 or cells.size <= 1 or not self.diffusion_axes:
            return None, None

        nx, ny, nz = int(solver.Nx), int(solver.Ny), int(solver.Nz)
        n_cells = int(cells.size)
        v_arr = (
            np.ones(n_cells, dtype=float)
            if cell_vol is None
            else np.asarray(cell_vol, dtype=float)
        )
        # Map flat cell index c = i + nx * j + nx * ny * k to (i, j, k)
        iz = cells // (nx * ny)
        rem = cells % (nx * ny)
        iy = rem // nx
        ix = rem % nx
        coords = {"x": ix, "y": iy, "z": iz}
        dims = {"x": nx, "y": ny, "z": nz}
        strides = {"x": 1, "y": nx, "z": nx * ny}
        spacings = {
            "x": np.asarray(solver.dx, dtype=float),
            "y": np.asarray(solver.dy, dtype=float),
            "z": np.asarray(solver.dz, dtype=float),
        }
        bc_low = [str(b).lower() for b in getattr(solver, "bc_low", ("pec",) * 3)]
        bc_period = {
            "x": bc_low[0] == "periodic",
            "y": bc_low[1] == "periodic",
            "z": bc_low[2] == "periodic",
        }

        rows = []
        cols = []
        vals = []
        diag = np.zeros(n_cells, dtype=float)

        for ax in self.diffusion_axes:
            c_ax = coords[ax]
            n_ax = dims[ax]
            stride = strides[ax]
            h_arr = spacings[ax]
            h_cell = h_arr[np.minimum(c_ax, h_arr.size - 1)]

            for step_dir in (-1, +1):
                nbr_coord = c_ax + step_dir
                if bc_period[ax]:
                    nbr_coord_mod = nbr_coord % n_ax
                    nbr_flat = cells + (nbr_coord_mod - c_ax) * stride
                    in_bounds = np.ones(n_cells, dtype=bool)
                else:
                    in_bounds = (nbr_coord >= 0) & (nbr_coord < n_ax)
                    nbr_flat = cells + step_dir * stride

                pos = np.searchsorted(cells, nbr_flat)
                pos_clipped = np.clip(pos, 0, n_cells - 1)
                valid = (
                    in_bounds
                    & (cells[pos_clipped] == nbr_flat)
                    & (pos_clipped != np.arange(n_cells))
                )
                src_idx = np.flatnonzero(valid)
                dst_idx = pos_clipped[valid]
                if src_idx.size > 0:
                    v_src = v_arr[src_idx]
                    v_dst = v_arr[dst_idx]
                    h_pair = 0.5 * (h_cell[src_idx] + h_cell[dst_idx])
                    g_cond = (
                        (2.0 * v_src * v_dst / (v_src + v_dst))
                        * self.diffusion_coeff
                        / (h_pair**2)
                    )
                    w_edge = g_cond / v_src
                    rows.append(src_idx)
                    cols.append(dst_idx)
                    vals.append(w_edge)
                    diag[src_idx] -= w_edge

        if not rows:
            return None, None

        all_r = np.concatenate(rows + [np.arange(n_cells)])
        all_c = np.concatenate(cols + [np.arange(n_cells)])
        all_v = np.concatenate(vals + [diag])
        diff_op = sp.csc_matrix((all_v, (all_r, all_c)), shape=(n_cells, n_cells))
        lhs = (
            sp.eye(n_cells, format="csc") * (1.0 + 0.5 * dt * self.A)
            - 0.5 * dt * diff_op
        )
        return diff_op, spla.factorized(lhs)

    def _init_solver_group(self, group, solver):
        """Initialize ADE state, geometry volumes, diffusion, and bookkeeping on ``group``."""
        dt = float(solver.dt)
        idx = group["idx_host"]
        cells = group["cells"]
        inv = group["inv"]
        n_cells = group["n_cells"]
        zeros_e = np.zeros(idx.size, dtype=solver.dtype)

        group["kind"] = "semiconductor"
        group["step_index"] = 0

        # Compute edge volumes dV_E and effective cell volumes V_cell so that
        # sum_m V_cell[m] * R_stim[m] == sum_e w_e * dV_E[e] * w_stim[e] identically
        itA = solver.grid.itA.toarray()
        L_arr = solver.grid.L.toarray()
        tA = np.divide(1.0, itA, out=np.zeros_like(itA), where=itA != 0.0)
        dV_E = L_arr[idx] * tA[idx]
        w_dV_edge = np.asarray(group["weight"], dtype=float) * dV_E
        # Cell volume from mean weighted edge volume per cell
        edge_counts = np.maximum(np.bincount(inv, minlength=n_cells), 1)
        cell_dV = np.bincount(inv, weights=dV_E, minlength=n_cells) / edge_counts
        cell_vol = group["cell_weight"] * cell_dV
        cell_vol = np.where(cell_vol > 0.0, cell_vol, 1.0)
        group["w_dV_edge"] = w_dV_edge
        group["cell_vol"] = cell_vol
        group["total_vol"] = float(np.sum(cell_vol))

        # Initialize ADE poles: for each pole p, driving strength is
        # S_p(N) = -c * n_bg * gamma_p * g_p(N)
        pole_states = []
        for pole in self.poles:
            alpha_p, xi_p, zeta_p_unit = _lorentz_step_coeffs(
                pole["omega_0"],
                pole["gamma"],
                -c_light * self.n_bg * pole["gamma"],
                dt,
            )
            pole_states.append(
                {
                    "spec": pole,
                    "alpha0": alpha_p,
                    "xi": xi_p,
                    "zeta_unit": zeta_p_unit,
                    "inv_denom": 1.0 / (1.0 + 0.5 * pole["gamma"] * dt),
                    "p": zeros_e.copy(),
                    "p_old": zeros_e.copy(),
                }
            )
        group["poles"] = pole_states

        n_init = self.initial_carrier_density(n_cells=n_cells)
        group["N"] = n_init.copy()
        group["N_initial_total"] = float(np.sum(cell_vol * n_init))
        group["carriers_injected"] = 0.0
        group["carriers_recombined"] = 0.0
        group["carriers_stimulated"] = 0.0
        group["photons_generated"] = 0.0
        group["em_work_joules"] = 0.0
        group["dchi_work_joules"] = 0.0
        group["dchi_parametric_joules"] = 0.0
        group["dchi_stored_joules"] = 0.0

        diff_op, diff_solver = self._build_cell_diffusion_operator(
            cells, solver, dt, cell_vol=cell_vol
        )
        group["diff_op"] = diff_op
        group["diff_solver"] = diff_solver
        group["_solver_ref"] = solver

    def _step_solver_group(self, g, En, E, acc, dt):
        """Advance polarization poles, ``E^(n+1)``, carrier density ``N^(n+1)``, and bookkeeping."""
        w = g["weight"]
        inv = g["inv"]
        n_cells = g["n_cells"]
        N_cell = g["N"]
        step_idx = g["step_index"]

        # 1. Carrier-dependent background susceptibility Delta chi_bg(N^n) for Henry alpha_H
        has_dchi = self.alpha_H != 0.0 or len(self.poles) > 1 or self.domega_dN != 0.0
        if has_dchi:
            dchi_cell = self.delta_chi_bg(N=N_cell)
            dchi_edge = dchi_cell[inv]
            w_dchi = w * dchi_edge
            acc = acc + w_dchi * En
            den = g["den"] + w_dchi
        else:
            dchi_cell = None
            dchi_edge = None
            den = g["den"]

        # 2. Advance Lorentzian gain polarization poles p_p^(n+1) centred at t^n
        dp_list = []
        for p_state in g["poles"]:
            spec = p_state["spec"]
            g_p_cell = self._pole_gain(spec, N_cell)
            g_p_edge = g_p_cell[inv]
            if spec["domega_dN"] == 0.0:
                alpha_p = p_state["alpha0"]
            else:
                w_p_cell = self._pole_omega(spec, N_cell)
                alpha_p = (2.0 - (w_p_cell[inv] * dt) ** 2) * p_state["inv_denom"]
            p_cur = p_state["p"]
            p_old = p_state["p_old"]
            p_new = (
                alpha_p * p_cur
                + p_state["xi"] * p_old
                + (p_state["zeta_unit"] * g_p_edge) * En
            )
            dp = p_new - p_cur
            acc = acc - w * dp
            p_state["p"], p_state["p_old"] = p_new, p_cur
            dp_list.append((spec["omega_0"], dp))

        # 3. Advance E^(n+1) centred at t^(n+1/2)
        E_new = acc / den
        E[g["idx"]] = E_new

        # 4. Stimulated emission work and carrier update centred at t^(n+1/2)
        e_half = 0.5 * (np.real(E_new) + np.real(En))
        w_dV = g["w_dV_edge"]
        cell_vol = g["cell_vol"]

        stim_edge_rate = np.zeros_like(e_half, dtype=float)
        em_power_edge = np.zeros_like(e_half, dtype=float)
        for omega_p, dp in dp_list:
            # Power density delivered from polarization to EM field: -E^(n+1/2) . dP/dt|^(n+1/2)
            p_dot = (eps_0 / dt) * np.real(dp)
            pow_density = -e_half * p_dot
            em_power_edge += pow_density
            stim_edge_rate += pow_density / (hbar * omega_p)

        step_em_work = float(dt * np.sum(w_dV * em_power_edge))
        step_photons = float(dt * np.sum(w_dV * stim_edge_rate))
        g["em_work_joules"] += step_em_work
        g["photons_generated"] += step_photons

        if self.enable_stimulated:
            if self.spatial_averaging and n_cells > 1:
                r_stim_cell = np.full(
                    n_cells,
                    float(np.sum(w_dV * stim_edge_rate) / g["total_vol"]),
                    dtype=float,
                )
            else:
                cell_photons_rate = np.bincount(
                    inv, weights=w_dV * stim_edge_rate, minlength=n_cells
                )
                r_stim_cell = cell_photons_rate / cell_vol
        else:
            r_stim_cell = np.zeros(n_cells, dtype=float)

        t_half = (step_idx + 0.5) * dt
        r_inj_half = self.injection_rate(t=t_half)
        n_next, r_rec_half = self._step_carrier_implicit_midpoint(
            N_cell,
            dt,
            r_inj_half,
            r_stim_cell,
            diff_op=g["diff_op"],
            diff_solver=g["diff_solver"],
        )
        g["N"] = n_next
        g["step_index"] = step_idx + 1

        if has_dchi:
            e_new_sq = np.real(E_new) ** 2
            e_old_sq = np.real(En) ** 2
            # Work done on the host-permittivity EM field by -eps_0 * Delta chi_bg(N^n) * dE/dt:
            step_dchi_work = float(
                -0.5 * eps_0 * np.sum(w_dV * dchi_edge * (e_new_sq - e_old_sq))
            )
            dchi_next_edge = self.delta_chi_bg(N=n_next)[inv]
            # Parametric index-modulation work 0.5 * eps_0 * (Delta chi_bg^(n+1) - Delta chi_bg^n) * (E^(n+1))^2:
            step_dchi_param = float(
                0.5 * eps_0 * np.sum(w_dV * (dchi_next_edge - dchi_edge) * e_new_sq)
            )
            g["dchi_work_joules"] += step_dchi_work
            g["dchi_parametric_joules"] += step_dchi_param
            g["dchi_stored_joules"] = float(
                0.5 * eps_0 * np.sum(w_dV * dchi_next_edge * e_new_sq)
            )

        g["carriers_injected"] += float(dt * np.sum(cell_vol * r_inj_half))
        g["carriers_recombined"] += float(dt * np.sum(cell_vol * r_rec_half))
        g["carriers_stimulated"] += float(dt * np.sum(cell_vol * r_stim_cell))

    def _restrict_solver_group(self, group, keep, old_cell_mask):
        """Restrict a semiconductor gain group when edges are overridden."""
        for p_state in group["poles"]:
            p_state["p"] = p_state["p"][keep]
            p_state["p_old"] = p_state["p_old"][keep]
        group["N"] = group["N"][old_cell_mask]
        group["w_dV_edge"] = group["w_dV_edge"][keep]
        group["cell_vol"] = group["cell_vol"][old_cell_mask]
        group["total_vol"] = float(np.sum(group["cell_vol"]))
        group["N_initial_total"] = float(np.sum(group["cell_vol"] * group["N"]))
        solver = group.get("_solver_ref")
        if solver is not None:
            diff_op, diff_solver = self._build_cell_diffusion_operator(
                group["cells"],
                solver,
                float(solver.dt),
                cell_vol=group["cell_vol"],
            )
            group["diff_op"] = diff_op
            group["diff_solver"] = diff_solver

    def _reset_solver_group(self, group):
        """Reset polarizations, carrier density, and bookkeeping counters."""
        for p_state in group["poles"]:
            p_state["p"][:] = 0.0
            p_state["p_old"][:] = 0.0
        n_init = self.initial_carrier_density(n_cells=group["n_cells"])
        group["N"][:] = n_init
        group["step_index"] = 0
        group["N_initial_total"] = float(np.sum(group["cell_vol"] * n_init))
        group["carriers_injected"] = 0.0
        group["carriers_recombined"] = 0.0
        group["carriers_stimulated"] = 0.0
        group["photons_generated"] = 0.0
        group["em_work_joules"] = 0.0
        group["dchi_work_joules"] = 0.0
        group["dchi_parametric_joules"] = 0.0
        group["dchi_stored_joules"] = 0.0

    def __repr__(self):
        label = f"{self.name}: " if self.name else ""
        return (
            f"SemiconductorGain({label}eps_inf={self.eps_inf:.6g}, "
            f"omega_0={self.omega_0:.6g}, gamma={self.gamma:.6g}, "
            f"N_tr={self.N_tr:.6g}, g_prime={self.differential_gain:.6g}, "
            f"gain_model={self.gain_model!r}, alpha_H={self.alpha_H:.6g})"
        )


def adiabatic_chirp(t, power, alpha_H):
    """
    Evaluate the dynamic/adiabatic frequency chirp ``Delta nu(t) = (alpha_H / (4 pi)) d(ln P)/dt`` [Hz].

    Follows T. L. Koch and J. E. Bowers, "Nature of wavelength chirping in
    directly modulated semiconductor lasers", *Electron. Lett.* **20**(25),
    1038-1040 (1984), doi:10.1049/el:19840709 (with nonlinear gain compression
    ``eps_nl = 0``).

    Parameters
    ----------
    t : array_like
        Time array [s] (or scalar timestep ``dt`` [s]).
    power : array_like
        Optical power or intensity envelope ``P(t)`` [W or W/m^2], ``> 0``.
    alpha_H : float
        Henry linewidth enhancement factor [-].

    Returns
    -------
    delta_nu : ndarray
        Frequency chirp ``Delta nu(t)`` [Hz].
    """
    p_arr = np.asarray(power, dtype=float)
    ln_p = np.log(np.maximum(p_arr, 1.0e-300))
    t_arr = np.asarray(t, dtype=float)
    dlnp_dt = (
        np.gradient(ln_p, float(t_arr))
        if t_arr.ndim == 0
        else np.gradient(ln_p, t_arr)
    )
    return (float(alpha_H) / (4.0 * np.pi)) * dlnp_dt
