# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Frequency-dependent (dispersive) materials for the FIT time-domain solver.

The relative permittivity is a sum of poles plus a static conductivity::

    eps(w) = eps_inf + sum_p chi_p(w) + sigma / (i w eps_0)

with Lorentz, Drude and Debye susceptibilities chi_p. Wakis transforms time
signals with the kernel exp(-i w t) (see `wakis.DFTMonitor`), i.e. phasors
follow the engineering convention x(t) = Re[X exp(+i w t)]. In this convention
a passive (lossy) material has ``Im eps(w) <= 0``. Optical data in the physics
convention (n + i k, k >= 0) must be conjugated, see `eps_from_nk`.

In the time domain every pole is integrated with an auxiliary differential
equation (ADE) for its polarization P_p, which enters Ampere's law as the
polarization current dP_p/dt::

    eps_0 eps_inf dE/dt = curl H - J - sigma E - sum_p dP_p/dt

The update is local to the dispersive edges and linear in all fields, see the
physics guide ("Dispersive materials") for the discretization, its discrete
permittivity and the stability limit.
"""

import warnings

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as q_e
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import hbar
from scipy.constants import mu_0
from scipy.optimize import least_squares

#: Angular frequency [rad/s] of a photon energy of 1 eV (hbar w = 1 eV).
EV = q_e / hbar


class Lorentz:
    """
    Lorentz oscillator pole.

    chi(w) = delta_eps w_0^2 / (w_0^2 - w^2 + i gamma w), in the exp(+i w t)
    convention, from d^2P/dt^2 + gamma dP/dt + w_0^2 P = eps_0 delta_eps w_0^2 E.

    Parameters
    ----------
    delta_eps : float
        Oscillator strength (static susceptibility), >= 0 for a passive medium.
    omega_0 : float
        Resonance angular frequency [rad/s].
    gamma : float
        Damping rate [rad/s], >= 0.
    """

    def __init__(self, delta_eps, omega_0, gamma):
        self.delta_eps = float(delta_eps)
        self.omega_0 = float(omega_0)
        self.gamma = float(gamma)

    @property
    def strength(self):
        """Numerator delta_eps w_0^2 of the susceptibility [rad^2/s^2]."""
        return self.delta_eps * self.omega_0**2

    def chi(self, omega):
        """Continuous susceptibility at angular frequency `omega` [rad/s]."""
        omega = np.asarray(omega, dtype=float)
        return self.strength / (self.omega_0**2 - omega**2 + 1j * self.gamma * omega)

    def chi_discrete(self, omega, dt):
        """Susceptibility of the time-discrete ADE with step `dt` [s]."""
        omega = np.asarray(omega, dtype=float)
        w2 = (2 / dt * np.sin(omega * dt / 2)) ** 2
        damping = 1j * self.gamma * np.sin(omega * dt) / dt
        return self.strength / (self.omega_0**2 - w2 + damping)

    def __repr__(self):
        return (
            f"Lorentz(delta_eps={self.delta_eps:.6g}, omega_0={self.omega_0:.6g}, "
            f"gamma={self.gamma:.6g})"
        )


class Drude(Lorentz):
    """
    Drude (free-carrier) pole: a Lorentz pole with w_0 = 0.

    chi(w) = -w_p^2 / (w^2 - i gamma w), from dJ/dt + gamma J = eps_0 w_p^2 E
    with J = dP/dt.

    Parameters
    ----------
    omega_p : float
        Plasma angular frequency [rad/s].
    gamma : float
        Collision rate [rad/s], >= 0.
    """

    def __init__(self, omega_p, gamma):
        self.omega_p = float(omega_p)
        self.omega_0 = 0.0
        self.gamma = float(gamma)

    @property
    def strength(self):
        return self.omega_p**2

    def __repr__(self):
        return f"Drude(omega_p={self.omega_p:.6g}, gamma={self.gamma:.6g})"


class Debye:
    """
    Debye relaxation pole.

    chi(w) = delta_eps / (1 + i w tau), from tau dP/dt + P = eps_0 delta_eps E.

    Parameters
    ----------
    delta_eps : float
        Relaxation strength eps_static - eps_inf of this pole, >= 0.
    tau : float
        Relaxation time [s], > 0.
    """

    def __init__(self, delta_eps, tau):
        self.delta_eps = float(delta_eps)
        self.tau = float(tau)

    def chi(self, omega):
        """Continuous susceptibility at angular frequency `omega` [rad/s]."""
        omega = np.asarray(omega, dtype=float)
        return self.delta_eps / (1 + 1j * omega * self.tau)

    def chi_discrete(self, omega, dt):
        """Susceptibility of the Crank-Nicolson ADE with step `dt` [s]."""
        omega = np.asarray(omega, dtype=float)
        return self.delta_eps / (1 + 2j * self.tau / dt * np.tan(omega * dt / 2))

    def __repr__(self):
        return f"Debye(delta_eps={self.delta_eps:.6g}, tau={self.tau:.6g})"


class DispersiveMaterial(list):
    """
    Material with dispersive permittivity eps(w) and/or permeability mu(w).

    The object is also the static triple ``[eps_inf, mu_inf, 0.0]`` (a list), so
    it can be used wherever Wakis expects a material ``[eps_r, mu_r, sigma]``,
    e.g. in ``GridFIT3D(stl_materials={...})``; `SolverFIT3D` then adds the
    pole dynamics to the cells of that solid automatically.

    Parameters
    ----------
    eps_inf : float, optional
        High-frequency relative permittivity (>= 1 is recommended).
    poles : sequence of Lorentz, Drude or Debye, optional
        Electric dispersive poles.
    sigma : float, optional
        Static electric conductivity [S/m], integrated semi-implicitly (no
        timestep restriction from the relaxation time eps/sigma).
    name : str, optional
        Label used in representations.
    mu_inf : float or 3-tuple of float, optional
        High-frequency relative permeability (scalar for isotropic, or
        ``(mu_x, mu_y, mu_z)`` for diagonal anisotropy).
    mu_poles : sequence of poles or dict of {'x', 'y', 'z'} to sequence, optional
        Magnetic dispersive poles (`Lorentz`, `Drude` or `Debye`, whose
        ``delta_eps`` is the magnetic susceptibility strength ``delta_mu``). A
        sequence applies to all three axes; a dict ``{'x': [...], 'y': [...],
        'z': [...]}`` specifies diagonal anisotropy (missing keys default to
        empty).
    sigma_m : float or 3-tuple of float, optional
        Static magnetic conductivity [Ohm/m], integrated semi-implicitly.

    Examples
    --------
    >>> drude = DispersiveMaterial(1.0, [Drude(omega_p=2e15, gamma=1e13)])
    >>> eps = drude.epsilon(f=[1e14, 3e14])
    """

    def __init__(
        self,
        eps_inf=1.0,
        poles=(),
        sigma=0.0,
        name=None,
        mu_inf=1.0,
        mu_poles=(),
        sigma_m=0.0,
    ):
        self.eps_inf = float(eps_inf)
        self.poles = list(poles)
        self.sigma = float(sigma)
        self.name = name
        for pole in self.poles:
            if not isinstance(pole, (Lorentz, Debye)):
                raise TypeError(f"Unknown pole type {type(pole).__name__}")

        if np.isscalar(mu_inf):
            self.mu_inf = float(mu_inf)
            self._mu_inf_xyz = (self.mu_inf,) * 3
        else:
            vals = tuple(float(v) for v in mu_inf)
            if len(vals) != 3:
                raise ValueError("mu_inf must be a float or a 3-tuple of floats")
            self.mu_inf = vals
            self._mu_inf_xyz = vals

        if np.isscalar(sigma_m):
            self.sigma_m = float(sigma_m)
            self._sigma_m_xyz = (self.sigma_m,) * 3
        else:
            svals = tuple(float(v) for v in sigma_m)
            if len(svals) != 3:
                raise ValueError("sigma_m must be a float or a 3-tuple of floats")
            self.sigma_m = svals
            self._sigma_m_xyz = svals

        if isinstance(mu_poles, dict):
            invalid = set(mu_poles) - {"x", "y", "z"}
            if invalid:
                raise ValueError(
                    f"Unknown mu_poles keys {sorted(invalid)}; expected a subset "
                    "of {'x', 'y', 'z'}"
                )
            self._mu_poles_xyz = {c: list(mu_poles.get(c, ())) for c in "xyz"}
            self.mu_poles = {c: list(self._mu_poles_xyz[c]) for c in "xyz"}
        else:
            plist = list(mu_poles)
            self._mu_poles_xyz = {c: list(plist) for c in "xyz"}
            self.mu_poles = plist

        for c in "xyz":
            for pole in self._mu_poles_xyz[c]:
                if not isinstance(pole, (Lorentz, Debye)):
                    raise TypeError(f"Unknown pole type {type(pole).__name__}")

        self.is_mu_anisotropic = (
            (not np.isscalar(mu_inf))
            or isinstance(mu_poles, dict)
            or (not np.isscalar(sigma_m))
        )
        super().__init__([self.eps_inf, float(self._mu_inf_xyz[0]), 0.0])

    @property
    def has_electric(self):
        """Whether the material has a non-vacuum electric response."""
        return self.eps_inf != 1.0 or bool(self.poles) or self.sigma != 0.0

    @property
    def has_magnetic(self):
        """Whether the material has a non-vacuum magnetic response."""
        return (
            any(m != 1.0 for m in self._mu_inf_xyz)
            or any(bool(self._mu_poles_xyz[c]) for c in "xyz")
            or any(s != 0.0 for s in self._sigma_m_xyz)
        )

    @property
    def lorentz_poles(self):
        """Second-order electric poles (Lorentz and Drude)."""
        return [p for p in self.poles if isinstance(p, Lorentz)]

    @property
    def debye_poles(self):
        """First-order (Debye) electric poles."""
        return [p for p in self.poles if isinstance(p, Debye)]

    def mu_lorentz_poles(self, component="x"):
        """Second-order magnetic poles (Lorentz and Drude) on `component`."""
        return [p for p in self._mu_poles_xyz[component] if isinstance(p, Lorentz)]

    def mu_debye_poles(self, component="x"):
        """First-order (Debye) magnetic poles on `component`."""
        return [p for p in self._mu_poles_xyz[component] if isinstance(p, Debye)]

    def epsilon(self, f):
        """
        Complex relative permittivity in the exp(+i w t) convention.

        Parameters
        ----------
        f : float or array_like
            Frequency [Hz], > 0 if `sigma` or a Drude pole is present.

        Returns
        -------
        eps : complex ndarray
            eps(f); ``Im eps <= 0`` for passive materials.
        """
        omega = 2 * np.pi * np.asarray(f, dtype=float)
        eps = self.eps_inf + 0j * omega
        for pole in self.poles:
            eps = eps + pole.chi(omega)
        if self.sigma:
            eps = eps + self.sigma / (1j * omega * eps_0)
        return eps

    def epsilon_discrete(self, f, dt):
        """
        Permittivity realised by the time-discrete ADE update with step `dt`.

        Defined such that the discrete Ampere law reads
        curl H = i W eps_0 eps_d E with W = 2 sin(w dt / 2) / dt, i.e. the
        scheme behaves as a Yee scheme in a medium of permittivity eps_d. It
        tends to `epsilon` with second-order accuracy in `dt`.

        Parameters
        ----------
        f : float or array_like
            Frequency [Hz].
        dt : float
            Timestep [s].

        Returns
        -------
        eps : complex ndarray
        """
        omega = 2 * np.pi * np.asarray(f, dtype=float)
        eps = self.eps_inf + 0j * omega
        for pole in self.poles:
            eps = eps + pole.chi_discrete(omega, dt)
        if self.sigma:
            w_tilde = 2 / dt * np.sin(omega * dt / 2)
            eps = eps + self.sigma * np.cos(omega * dt / 2) / (1j * w_tilde * eps_0)
        return eps

    def _mu_comp(self, f, component, dt=None):
        idx = "xyz".index(component)
        omega = 2 * np.pi * np.asarray(f, dtype=float)
        mu_val = self._mu_inf_xyz[idx] + 0j * omega
        for pole in self._mu_poles_xyz[component]:
            mu_val = mu_val + (
                pole.chi(omega) if dt is None else pole.chi_discrete(omega, dt)
            )
        sig_m = self._sigma_m_xyz[idx]
        if sig_m:
            if dt is None:
                mu_val = mu_val + sig_m / (1j * omega * mu_0)
            else:
                w_tilde = 2 / dt * np.sin(omega * dt / 2)
                mu_val = mu_val + (
                    sig_m * np.cos(omega * dt / 2) / (1j * w_tilde * mu_0)
                )
        return mu_val

    def mu(self, f, component=None):
        """
        Complex relative permeability in the exp(+i w t) convention.

        Parameters
        ----------
        f : float or array_like
            Frequency [Hz], > 0 if `sigma_m` or a Drude pole is present.
        component : {'x', 'y', 'z'}, optional
            Axis to evaluate. If None, returns a scalar/array of shape
            ``np.shape(f)`` when isotropic, or ``(3, *np.shape(f))`` when
            diagonally anisotropic.

        Returns
        -------
        mu : complex ndarray
            mu(f); ``Im mu <= 0`` for passive materials.
        """
        if component is not None:
            return self._mu_comp(f, component)
        if self.is_mu_anisotropic:
            return np.stack([self._mu_comp(f, c) for c in "xyz"], axis=0)
        return self._mu_comp(f, "x")

    def mu_discrete(self, f, dt, component=None):
        """
        Permeability realised by the time-discrete ADE update with step `dt`.

        Defined such that the discrete Faraday law reads
        -curl E = i W mu_0 mu_d H with W = 2 sin(w dt / 2) / dt, i.e. the
        scheme behaves as a Yee scheme in a medium of permeability mu_d. It
        tends to `mu` with second-order accuracy in `dt`.

        Parameters
        ----------
        f : float or array_like
            Frequency [Hz].
        dt : float
            Timestep [s].
        component : {'x', 'y', 'z'}, optional
            Axis to evaluate. If None, returns a scalar/array of shape
            ``np.shape(f)`` when isotropic, or ``(3, *np.shape(f))`` when
            diagonally anisotropic.

        Returns
        -------
        mu : complex ndarray
        """
        if component is not None:
            return self._mu_comp(f, component, dt=dt)
        if self.is_mu_anisotropic:
            return np.stack([self._mu_comp(f, c, dt=dt) for c in "xyz"], axis=0)
        return self._mu_comp(f, "x", dt=dt)

    def is_stable(self, dt, courant=1.0):
        """
        Check the von Neumann stability of the coupled Yee + ADE update.

        The spectral radius of the single-mode amplification matrix is
        evaluated for all discrete wavenumbers allowed on the grid, accounting
        jointly for the electric and magnetic poles. The vacuum Courant number
        is ``c dt sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)`` (<= 1).

        Parameters
        ----------
        dt : float
            Timestep [s].
        courant : float, optional
            Vacuum Courant number of the grid with this `dt`.

        Returns
        -------
        bool
            True if no mode grows.

        Notes
        -----
        For a single lossless Drude pole the condition reduces to
        (w_p dt / 2)^2 + courant^2 <= eps_inf, and for a lossless Lorentz pole
        to (w_0 dt / 2)^2 <= (eps_inf - courant^2) / (eps_s - courant^2) (and
        dually for a purely magnetic material).
        """
        comps = ("x", "y", "z") if self.is_mu_anisotropic else ("x",)
        for comp in comps:
            for q in np.linspace(0.0, 2 * courant, 101):
                A = _amplification_matrix(self, dt, q, mu_comp=comp)
                rho = np.max(np.abs(np.linalg.eigvals(A)))
                if rho > 1 + 1e-6:
                    return False
        return True

    def __repr__(self):
        label = f"{self.name}: " if self.name else ""
        if not self.has_magnetic:
            return (
                f"DispersiveMaterial({label}eps_inf={self.eps_inf:.6g}, "
                f"poles={self.poles}, sigma={self.sigma:.6g})"
            )
        return (
            f"DispersiveMaterial({label}eps_inf={self.eps_inf:.6g}, "
            f"poles={self.poles}, sigma={self.sigma:.6g}, "
            f"mu_inf={self.mu_inf}, mu_poles={self.mu_poles}, "
            f"sigma_m={self.sigma_m})"
        )


def _ade_coefficients_poles(poles, dt):
    """Per-pole update coefficients for a sequence of Lorentz/Drude/Debye poles."""
    lorentz = []
    debye = []
    for pole in poles:
        if isinstance(pole, Lorentz):
            d = 1 + pole.gamma * dt / 2
            lorentz.append(
                (
                    (2 - (pole.omega_0 * dt) ** 2) / d,  # alpha: p^n
                    (pole.gamma * dt / 2 - 1) / d,  # xi: p^(n-1)
                    pole.strength * dt**2 / d,  # zeta: E^n
                )
            )
        elif isinstance(pole, Debye):
            r = dt / (2 * pole.tau)
            debye.append(((1 - r) / (1 + r), pole.delta_eps * r / (1 + r)))  # kappa, beta
    return lorentz, debye


def _ade_coefficients(material, dt):
    """Per-pole electric update coefficients (normalized polarization p = P / eps_0)."""
    return _ade_coefficients_poles(material.poles, dt)


def _ade_h_coefficients(material, dt, component="x"):
    """Per-pole magnetic update coefficients (normalized magnetization m = M / mu_0)."""
    return _ade_coefficients_poles(material._mu_poles_xyz[component], dt)


def _amplification_matrix(material, dt, q, mu_comp="x"):
    """
    One-step amplification matrix of a plane wave in the dispersive medium.

    State [E^n, h^(n-1/2), p_L^n, p_L^(n-1), ..., p_D^n, m_L^(n-1/2),
    m_L^(n-3/2), ..., m_D^(n-1/2)] in units with eps_0 = mu_0 = 1;
    `q` = c K dt in [0, 2 courant], K the discrete wavenumber, so
    mu_inf (h' - h) = -q E - ... and eps_inf (E' - E) = q h' - ...
    """
    lorentz_e, debye_e = _ade_coefficients(material, dt)
    s_hat_e = material.sigma * dt / eps_0
    beta_sum_e = sum(beta for _, beta in debye_e)
    den_e = material.eps_inf + s_hat_e / 2 + beta_sum_e
    a1_e = material.eps_inf - s_hat_e / 2 - beta_sum_e

    c_idx = "xyz".index(mu_comp)
    mu_inf = material._mu_inf_xyz[c_idx]
    sigma_m = material._sigma_m_xyz[c_idx]
    lorentz_m, debye_m = _ade_h_coefficients(material, dt, mu_comp)
    s_hat_m = sigma_m * dt / mu_0
    beta_sum_m = sum(beta for _, beta in debye_m)
    den_m = mu_inf + s_hat_m / 2 + beta_sum_m
    a1_m = mu_inf - s_hat_m / 2 - beta_sum_m

    n_e = 2 * len(lorentz_e) + len(debye_e)
    n_m = 2 * len(lorentz_m) + len(debye_m)
    n = 2 + n_e + n_m

    def step(x):
        E, h = x[0], x[1]
        y = np.zeros(n)

        # Magnetic half-step + ADE: h^(n-1/2) -> h^(n+1/2)
        acc_h = a1_m * h - q * E
        off_m = 2 + n_e
        for k, (alpha, xi, zeta) in enumerate(lorentz_m):
            m, m_old = x[off_m + 2 * k], x[off_m + 2 * k + 1]
            m_new = alpha * m + xi * m_old + zeta * h
            acc_h -= m_new - m
            y[off_m + 2 * k], y[off_m + 2 * k + 1] = m_new, m
        off_md = off_m + 2 * len(lorentz_m)
        for k, (kappa, _) in enumerate(debye_m):
            acc_h -= (kappa - 1) * x[off_md + k]
        y[1] = acc_h / den_m
        for k, (kappa, beta) in enumerate(debye_m):
            y[off_md + k] = kappa * x[off_md + k] + beta * (y[1] + h)

        # Electric half-step + ADE: E^n -> E^(n+1)
        acc_e = a1_e * E + q * y[1]
        for k, (alpha, xi, zeta) in enumerate(lorentz_e):
            p, p_old = x[2 + 2 * k], x[3 + 2 * k]
            p_new = alpha * p + xi * p_old + zeta * E
            acc_e -= p_new - p
            y[2 + 2 * k], y[3 + 2 * k] = p_new, p
        off_ed = 2 + 2 * len(lorentz_e)
        for k, (kappa, _) in enumerate(debye_e):
            acc_e -= (kappa - 1) * x[off_ed + k]
        y[0] = acc_e / den_e
        for k, (kappa, beta) in enumerate(debye_e):
            y[off_ed + k] = kappa * x[off_ed + k] + beta * (y[0] + E)
        return y

    return np.column_stack([step(col) for col in np.eye(n)])


def eps_from_nk(n, k):
    """
    Relative permittivity in the Wakis exp(+i w t) convention from optical
    constants n + i k (physics convention, k >= 0 for absorption).

    Returns
    -------
    eps : complex ndarray
        (n - i k)^2, with ``Im eps <= 0`` for absorbing media.
    """
    return (np.asarray(n) - 1j * np.asarray(k)) ** 2


def fit_poles(
    f, eps, n_lorentz=1, n_drude=0, n_debye=0, eps_inf=None, n_starts=4, name=None
):
    """
    Fit a passive pole model to tabulated complex permittivity data.

    Minimizes the relative complex error ``(eps_model - eps) / |eps|`` with
    `scipy.optimize.least_squares`. Every strength, resonance, damping and
    relaxation time is parameterized by its logarithm, so the result is
    passive (``Im eps <= 0`` at all frequencies) and causal by construction.

    Parameters
    ----------
    f : array_like
        Frequencies [Hz] of the data.
    eps : array_like of complex
        Relative permittivity in the Wakis exp(+i w t) convention (use
        `eps_from_nk` or ``np.conj`` to convert optical data).
    n_lorentz, n_drude, n_debye : int, optional
        Number of poles of each kind.
    eps_inf : float, optional
        Fixed high-frequency permittivity; fitted (> 0) if None.
    n_starts : int, optional
        Number of deterministic multi-start initial guesses (different
        initial damping); the best fit is returned.
    name : str, optional
        Name of the returned material.

    Returns
    -------
    material : DispersiveMaterial
        Fitted model; the relative RMS error is stored in ``material.fit_error``.
    """
    f = np.asarray(f, dtype=float)
    eps = np.asarray(eps, dtype=complex)
    omega = 2 * np.pi * f
    w_ref = omega.max()
    w_lo = omega.min()
    weight = 1 / np.maximum(np.abs(eps), 1e-12)

    def unpack(x):
        x = np.exp(x)
        i = 0
        if eps_inf is None:
            e_inf, i = x[0], 1
        else:
            e_inf = eps_inf
        poles = []
        for _ in range(n_lorentz):
            poles.append(Lorentz(x[i], x[i + 1] * w_ref, x[i + 2] * w_ref))
            i += 3
        for _ in range(n_drude):
            poles.append(Drude(x[i] * w_ref, x[i + 1] * w_ref))
            i += 2
        for _ in range(n_debye):
            poles.append(Debye(x[i], x[i + 1] / w_ref))
            i += 2
        return DispersiveMaterial(e_inf, poles, name=name)

    def residual(x):
        r = (unpack(x).epsilon(f) - eps) * weight
        return np.concatenate([r.real, r.imag])

    def initial_guess(damping):
        x0 = [] if eps_inf is not None else [max(np.real(eps[-1]), 1.0)]
        centres = np.geomspace(w_lo, w_ref, n_lorentz + 2)[1:-1] / w_ref
        for w0 in centres:
            x0 += [1.0, w0, damping * w0]
        for _ in range(n_drude):
            x0 += [1.0, damping]
        taus = np.geomspace(w_ref / w_lo, 1.0, n_debye + 2)[1:-1]
        for tau in taus:
            x0 += [1.0, tau]
        return np.log(np.asarray(x0, dtype=float))

    best = None
    for damping in np.geomspace(0.01, 1.0, n_starts):
        sol = least_squares(residual, initial_guess(damping), method="lm")
        if best is None or sol.cost < best.cost:
            best = sol
    material = unpack(best.x)
    material.fit_error = np.sqrt(2 * best.cost / f.size)
    return material


def _rakic(name, omega_p, f, gamma, omega):
    """Lorentz-Drude model of Rakic et al. (1998), parameters in eV."""
    poles = [Drude(np.sqrt(f[0]) * omega_p * EV, gamma[0] * EV)]
    for fk, gk, wk in zip(f[1:], gamma[1:], omega[1:]):
        poles.append(Lorentz(fk * omega_p**2 / wk**2, wk * EV, gk * EV))
    return DispersiveMaterial(1.0, poles, name=name)


def _sellmeier(name, B, C_um):
    """Lossless Sellmeier model n^2 = 1 + sum B lambda^2/(lambda^2 - C^2)."""
    poles = [
        Lorentz(b, 2 * np.pi * c_light / (cc * 1e-6), 0.0) for b, cc in zip(B, C_um)
    ]
    return DispersiveMaterial(1.0, poles, name=name)


# fmt: off
#: Library of dispersive materials. See the physics guide for the validity
#: range of each model.
dispersive_lib = {
    # Rakic et al., Appl. Opt. 37, 5271 (1998), Lorentz-Drude, 0.1-6 eV
    'gold': _rakic('gold', 9.03,
        [0.760, 0.024, 0.010, 0.071, 0.601, 4.384],
        [0.053, 0.241, 0.345, 0.870, 2.494, 2.214],
        [0.000, 0.415, 0.830, 2.969, 4.304, 13.32]),
    'silver': _rakic('silver', 9.01,
        [0.845, 0.065, 0.124, 0.011, 0.840, 5.646],
        [0.048, 3.886, 0.452, 0.065, 0.916, 2.419],
        [0.000, 0.816, 4.481, 8.185, 9.083, 20.29]),
    'aluminium': _rakic('aluminium', 14.98,
        [0.523, 0.227, 0.050, 0.166, 0.030],
        [0.047, 0.333, 0.312, 1.351, 3.382],
        [0.000, 0.162, 1.544, 1.808, 3.473]),
    # Malitson, J. Opt. Soc. Am. 55, 1205 (1965), 0.21-6.7 um, lossless
    'fused silica': _sellmeier('fused silica',
        [0.6961663, 0.4079426, 0.8974794],
        [0.0684043, 0.1162414, 9.896161]),
    # Salzberg & Villa, J. Opt. Soc. Am. 47, 244 (1957), 1.36-11 um, lossless
    'silicon': _sellmeier('silicon',
        [10.6684293, 0.0030434748, 1.54133408],
        [0.301516485, 1.13475115, 1104.0]),
    # 3-pole Debye fits (Wakis, fit_poles, 0.1-20 GHz, rel. error ~1e-3) to
    # Djordjevic-Sarkar wideband models (m1 = 4, m2 = 12) through nominal
    # laminate data: eps' = 4.4, tan d = 0.02 at 1 GHz (FR-4) and
    # eps' = 3.66, tan d = 0.0037 at 10 GHz (Rogers RO4350B design values)
    'fr4': DispersiveMaterial(4.17903, [
        Debye(0.150892, 1.20864e-09),
        Debye(0.124324, 1.17325e-10),
        Debye(0.141714, 1.18854e-11)], name='fr4'),
    'ro4350b': DispersiveMaterial(3.64546, [
        Debye(0.0240563, 1.22302e-09),
        Debye(0.0198475, 1.19157e-10),
        Debye(0.0226761, 1.20224e-11)], name='ro4350b'),
}
# fmt: on


class DispersiveMixin:
    """
    `SolverFIT3D` methods for dispersive materials (ADE on selected E and H edges).

    The non-dispersive H and E half-steps are run unchanged; after each half-step
    the new field on the dispersive edges is recomputed from the curl
    contribution of that half-step and the auxiliary polarizations or
    magnetizations. The update is local, linear and explicit.
    """

    def _box_edges(self, center, size, dual=False):
        """Flat indices and weights of E (`dual=False`) or H (`dual=True`) in a box."""
        flat, weights = [], []
        nodes = (self.grid.x, self.grid.y, self.grid.z)
        for c, _ in enumerate("xyz"):
            axis_idx, axis_w = [], []
            for d in range(3):
                if dual:
                    # H components live on dual edges: nodes along their own axis,
                    # cell midpoints along the two transverse axes
                    pos = (
                        nodes[d][:-1]
                        if d == c
                        else (nodes[d][:-1] + nodes[d][1:]) / 2
                    )
                else:
                    pos = (
                        (nodes[d][:-1] + nodes[d][1:]) / 2
                        if d == c
                        else nodes[d][:-1]
                    )
                if size[d] == 0:
                    i = np.array([np.argmin(np.abs(pos - center[d]))])
                    axis_idx.append(i)
                    axis_w.append(np.ones(1))
                    continue
                tol = 1e-6 * np.min(np.diff(nodes[d]))
                dist = size[d] / 2 - np.abs(pos - center[d])
                i = np.flatnonzero(dist > -tol)
                axis_idx.append(i)
                axis_w.append(np.where(dist[i] < tol, 0.5, 1.0))
            ii, jj, kk = np.meshgrid(*axis_idx, indexing="ij")
            wx, wy, wz = np.meshgrid(*axis_w, indexing="ij")
            n = ii + jj * self.Nx + kk * self.Nx * self.Ny + c * self.N
            flat.append(n.ravel())
            weights.append((wx * wy * wz).ravel())
        return np.concatenate(flat), np.concatenate(weights)

    def add_dispersive_material(
        self, material, center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0)
    ):
        """
        Fill a box with a dispersive material.

        E and H edges strictly inside the box get the material; edges lying on
        a face of the box get the arithmetic mean of material and background
        (weight 1/2 per face, as for tangential fields on a material
        interface). Later calls override earlier ones on shared edges.

        Parameters
        ----------
        material : DispersiveMaterial
            Material model.
        center, size : tuple of float
            Centre and extent of the box (x, y, z) [m]. A zero size selects
            the single nearest sample along that direction.

        Raises
        ------
        ValueError
            If the solver timestep is unstable for this material.
        """
        if getattr(material, "is_gain", False):
            return self.add_gain_material(material, center=center, size=size)
        if not isinstance(material, DispersiveMaterial):
            raise TypeError("material must be a DispersiveMaterial")
        if getattr(self, "scheme", "yee") == "tete":
            raise NotImplementedError(
                "scheme='tete' does not support dispersive (ADE) materials."
            )
        if material.has_electric or not material.has_magnetic:
            idx_e, w_e = self._box_edges(center, size, dual=False)
            self._add_dispersive_edges(material, idx_e, w_e)
        if material.has_magnetic:
            idx_h, w_h = self._box_edges(center, size, dual=True)
            self._add_dispersive_h_edges(material, idx_h, w_h)

    def _apply_stl_dispersive(self):
        """Attach ADE updates to STL solids whose material is dispersive."""
        conformal = getattr(self.grid, "conformal", False) in ("all", "tensor")
        warned_mu = False
        for key, material in self.grid.stl_materials.items():
            if not isinstance(material, DispersiveMaterial):
                continue
            if not material.has_magnetic:
                continue
            if conformal and not warned_mu:
                warnings.warn(
                    "Subpixel averaging of dispersive permeability (mu) is not "
                    "implemented; applying volume-fraction mapping on H.",
                    UserWarning,
                    stacklevel=2,
                )
                warned_mu = True
            mask = np.reshape(self.grid.grid[key], (self.Nx, self.Ny, self.Nz))
            fraction = np.reshape(mask, self.N, order="F").astype(float)
            fraction = np.tile(fraction, 3)
            idx = np.flatnonzero(fraction)
            self._add_dispersive_h_edges(material, idx, fraction[idx], mu_bg=1.0)

        if conformal:
            # Subpixel (Kottke) averaging of the electric dispersive solids
            from .tensor_dispersive import apply_conformal_dispersive

            if apply_conformal_dispersive(self):
                return
            if not any(
                isinstance(m, DispersiveMaterial) and m.has_electric
                for m in self.grid.stl_materials.values()
            ):
                return

        for key, material in self.grid.stl_materials.items():
            if not isinstance(material, DispersiveMaterial):
                continue
            if not (material.has_electric or not material.has_magnetic):
                continue
            # Same volume fraction on the three components and the same
            # (Nx, Ny, Nz) mapping as _apply_stl_materials uses for ieps
            mask = np.reshape(self.grid.grid[key], (self.Nx, self.Ny, self.Nz))
            fraction = np.reshape(mask, self.N, order="F").astype(float)
            fraction = np.tile(fraction, 3)
            idx = np.flatnonzero(fraction)
            # _apply_stl_materials mixes eps with vacuum: eps_bg = 1
            self._add_dispersive_edges(material, idx, fraction[idx], eps_bg=1.0)

    def _check_dispersive_stable(self, material):
        if not isinstance(material, DispersiveMaterial):
            raise TypeError("material must be a DispersiveMaterial")
        courant = (
            c_light
            * self.dt
            * np.sqrt(np.sum([1 / np.min(d) ** 2 for d in (self.dx, self.dy, self.dz)]))
        )
        if not material.is_stable(self.dt, courant):
            raise ValueError(
                f"dt={self.dt:.3e} s is unstable for {material!r}; pass a "
                "smaller `dt` or `cfln` to SolverFIT3D (see "
                "DispersiveMaterial.is_stable)."
            )

    def _add_dispersive_edges(self, material, idx, weight, eps_bg=None):
        """Register flat E indices `idx` with fill fractions `weight`."""
        self._check_dispersive_stable(material)

        net = getattr(self, "dispersive_network", None)
        if net is not None and np.any(np.isin(idx, net.rows)):
            raise NotImplementedError(
                "Dispersive regions must not overlap conformal dispersive solids."
            )
        ieps = np.asarray(self.ieps.array.get() if self.use_gpu else self.ieps.array)
        keep = (weight > 0) & (ieps[idx] != 0)  # skip PEC edges
        idx, weight = idx[keep], weight[keep]
        if not hasattr(self, "_dispersive_groups"):
            self._dispersive_groups = []
            fused = getattr(self, "_fused_step", None)
            if (
                fused is not None
                and self.one_step is fused
                and np.dtype(self.dtype) == np.float64
            ):
                # Fused GPU step: the ADE runs as kernels around the step
                from .gpu_fused import ADEHooks

                hooks = ADEHooks(self)
                fused.pre.append(hooks.pre)
                fused.post.append(hooks.post)
            elif (
                getattr(self, "activate_abc", False)
                and self.one_step == self._one_step_with_abc
            ):
                self._one_step_nondispersive = self._one_step_backend
                self._one_step_backend = self._one_step_dispersive
            else:
                self._one_step_nondispersive = self.one_step
                self.one_step = self._one_step_dispersive
        # Later assignments override earlier ones
        for group in self._dispersive_groups:
            old = ~np.isin(group["idx_host"], idx)
            if not old.all():
                _restrict_group(group, old, self.E.xp)
        if getattr(self, "_gain_groups", None):
            from .gain import _restrict_gain_group

            for group in self._gain_groups:
                old = ~np.isin(group["idx_host"], idx)
                if not old.all():
                    _restrict_gain_group(group, old)
            self._gain_groups = [
                g for g in self._gain_groups if g["idx_host"].size > 0
            ]

        eps_s = 1 / (eps_0 * ieps[idx])  # permittivity used by the solver step
        bg = eps_s if eps_bg is None else eps_bg
        eps_e = (1 - weight) * bg + weight * material.eps_inf
        lorentz, debye = _ade_coefficients(material, self.dt)
        beta_sum = weight * sum(beta for _, beta in debye)
        s_hat = weight * material.sigma * self.dt / eps_0 / 2
        xp = self.E.xp
        zeros = xp.zeros(idx.size, dtype=self.dtype)
        group = {
            "material": material,
            "idx_host": idx,
            "idx": xp.asarray(idx),
            "weight": xp.asarray(weight, dtype=self.dtype),
            "eps_s": xp.asarray(eps_s, dtype=self.dtype),
            "a1": xp.asarray(eps_e - s_hat - beta_sum, dtype=self.dtype),
            "den": xp.asarray(eps_e + s_hat + beta_sum, dtype=self.dtype),
            "lorentz": [(*cf, zeros.copy(), zeros.copy()) for cf in lorentz],
            "debye": [(*cf, zeros.copy()) for cf in debye],
        }
        self._dispersive_groups = [
            g for g in self._dispersive_groups if g["idx_host"].size
        ] + [group]

    def _add_dispersive_h_edges(self, material, idx, weight, mu_bg=None):
        """Register flat H indices `idx` with fill fractions `weight`."""
        self._check_dispersive_stable(material)
        fused = getattr(self, "_fused_step", None)
        if getattr(self, "fused", False) or (
            fused is not None and self.one_step is fused
        ):
            raise NotImplementedError(
                "The fused GPU step does not support dispersive permeability "
                "(mu) yet; use fused=False."
            )
        imu = np.asarray(self.imu.array.get() if self.use_gpu else self.imu.array)
        keep = (weight > 0) & (imu[idx] != 0)
        idx, weight = idx[keep], weight[keep]
        if not hasattr(self, "_dispersive_h_groups"):
            self._dispersive_h_groups = []
        # Later assignments override earlier ones
        for group in self._dispersive_h_groups:
            old = ~np.isin(group["idx_host"], idx)
            if not old.all():
                _restrict_group(group, old, self.H.xp)

        xp = self.H.xp
        new_groups = []
        comps = (
            [(c, comp, (idx // self.N) == c) for c, comp in enumerate("xyz")]
            if material.is_mu_anisotropic
            else [(0, "x", np.ones(idx.size, dtype=bool))]
        )
        for c_idx, comp, mask_c in comps:
            idx_c = idx[mask_c]
            if idx_c.size == 0:
                continue
            w_c = weight[mask_c]
            mu_s = 1 / (mu_0 * imu[idx_c])  # permeability used by the solver step
            bg = mu_s if mu_bg is None else mu_bg
            mu_inf_c = material._mu_inf_xyz[c_idx]
            sigma_m_c = material._sigma_m_xyz[c_idx]
            mu_e = (1 - w_c) * bg + w_c * mu_inf_c
            lorentz, debye = _ade_h_coefficients(material, self.dt, comp)
            beta_sum = w_c * sum(beta for _, beta in debye)
            s_hat = w_c * sigma_m_c * self.dt / mu_0 / 2
            zeros = xp.zeros(idx_c.size, dtype=self.dtype)
            new_groups.append(
                {
                    "material": material,
                    "comp": comp,
                    "idx_host": idx_c,
                    "idx": xp.asarray(idx_c),
                    "weight": xp.asarray(w_c, dtype=self.dtype),
                    "mu_s": xp.asarray(mu_s, dtype=self.dtype),
                    "a1": xp.asarray(mu_e - s_hat - beta_sum, dtype=self.dtype),
                    "den": xp.asarray(mu_e + s_hat + beta_sum, dtype=self.dtype),
                    "lorentz": [(*cf, zeros.copy(), zeros.copy()) for cf in lorentz],
                    "debye": [(*cf, zeros.copy()) for cf in debye],
                }
            )
        self._dispersive_h_groups = [
            g for g in self._dispersive_h_groups if g["idx_host"].size
        ] + new_groups

    def _capture_dispersive_H(self):
        """Capture H^(n-1/2) on magnetic dispersive edges before the H update."""
        groups = getattr(self, "_dispersive_h_groups", None)
        if not groups:
            return None
        return [self.H.array[g["idx"]].copy() for g in groups]

    def _update_dispersive_H(self, H_n):
        """Apply the local ADE correction to H^(n+1/2) after the H update."""
        if H_n is None:
            return
        H = self.H.array
        for g, Hn in zip(self._dispersive_h_groups, H_n):
            w = g["weight"]
            # (H* - H^(n-1/2)) mu_s = -(dt / mu_0) (curl E + M + PML/TFSF terms)
            acc = (H[g["idx"]] - Hn) * g["mu_s"] + g["a1"] * Hn
            for k, (alpha, xi, zeta, m, m_old) in enumerate(g["lorentz"]):
                m_new = alpha * m + xi * m_old + zeta * Hn
                acc -= w * (m_new - m)
                g["lorentz"][k] = (alpha, xi, zeta, m_new, m)
            for kappa, _, m in g["debye"]:
                acc -= w * (kappa - 1) * m
            H_new = acc / g["den"]
            for k, (kappa, beta, m) in enumerate(g["debye"]):
                g["debye"][k] = (kappa, beta, kappa * m + beta * (H_new + Hn))
            H[g["idx"]] = H_new

    def _one_step_dispersive(self):
        """Non-dispersive step followed by the local ADE correction of E."""
        E_n = [self.E.array[g["idx"]].copy() for g in self._dispersive_groups]
        self._one_step_nondispersive()
        E = self.E.array
        for g, En in zip(self._dispersive_groups, E_n):
            w = g["weight"]
            # (E* - E^n) eps_s = (dt / eps_0) (curl H - J + PML/TFSF terms)
            acc = (E[g["idx"]] - En) * g["eps_s"] + g["a1"] * En
            for k, (alpha, xi, zeta, p, p_old) in enumerate(g["lorentz"]):
                p_new = alpha * p + xi * p_old + zeta * En
                acc -= w * (p_new - p)
                g["lorentz"][k] = (alpha, xi, zeta, p_new, p)
            for kappa, _, p in g["debye"]:
                acc -= w * (kappa - 1) * p
            E_new = acc / g["den"]
            for k, (kappa, beta, p) in enumerate(g["debye"]):
                g["debye"][k] = (kappa, beta, kappa * p + beta * (E_new + En))
            E[g["idx"]] = E_new

    def reset_dispersive_state(self):
        """Zero the auxiliary polarizations and magnetizations."""
        if getattr(self, "dispersive_network", None) is not None:
            self.dispersive_network.reset()
        for g in getattr(self, "_dispersive_groups", []):
            g["lorentz"] = [(a, x, z, 0 * p, 0 * q) for a, x, z, p, q in g["lorentz"]]
            g["debye"] = [(k, b, 0 * p) for k, b, p in g["debye"]]
        for g in getattr(self, "_dispersive_h_groups", []):
            g["lorentz"] = [(a, x, z, 0 * m, 0 * q) for a, x, z, m, q in g["lorentz"]]
            g["debye"] = [(k, b, 0 * m) for k, b, m in g["debye"]]


def _restrict_group(group, keep, xp):
    """Drop the edges of a dispersive group where `keep` is False."""
    keep_dev = xp.asarray(keep)
    group["idx_host"] = group["idx_host"][keep]
    for key in ("idx", "weight", "eps_s", "mu_s", "a1", "den"):
        if key in group:
            group[key] = group[key][keep_dev]
    group["lorentz"] = [
        (a, x, z, p[keep_dev], q[keep_dev]) for a, x, z, p, q in group["lorentz"]
    ]
    group["debye"] = [(k, b, p[keep_dev]) for k, b, p in group["debye"]]


from . import material_library as _material_library  # noqa: E402, F401

