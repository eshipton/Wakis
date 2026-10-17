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

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as q_e
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import hbar
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
    Material with eps(w) = eps_inf + sum of Lorentz/Drude/Debye poles + sigma.

    The object is also the static triple ``[eps_inf, 1.0, 0.0]`` (a list), so
    it can be used wherever Wakis expects a material ``[eps_r, mu_r, sigma]``,
    e.g. in ``GridFIT3D(stl_materials={...})``; `SolverFIT3D` then adds the
    pole dynamics to the cells of that solid automatically.

    Parameters
    ----------
    eps_inf : float
        High-frequency relative permittivity (>= 1 is recommended).
    poles : list of Lorentz, Drude or Debye, optional
        Dispersive poles.
    sigma : float, optional
        Static conductivity [S/m], integrated semi-implicitly (no timestep
        restriction from the relaxation time eps/sigma).
    name : str, optional
        Label used in representations.

    Examples
    --------
    >>> drude = DispersiveMaterial(1.0, [Drude(omega_p=2e15, gamma=1e13)])
    >>> eps = drude.epsilon(f=[1e14, 3e14])
    """

    def __init__(self, eps_inf=1.0, poles=(), sigma=0.0, name=None):
        super().__init__([float(eps_inf), 1.0, 0.0])
        self.eps_inf = float(eps_inf)
        self.poles = list(poles)
        self.sigma = float(sigma)
        self.name = name
        for pole in self.poles:
            if not isinstance(pole, (Lorentz, Debye)):
                raise TypeError(f"Unknown pole type {type(pole).__name__}")

    @property
    def lorentz_poles(self):
        """Second-order poles (Lorentz and Drude)."""
        return [p for p in self.poles if isinstance(p, Lorentz)]

    @property
    def debye_poles(self):
        """First-order (Debye) poles."""
        return [p for p in self.poles if isinstance(p, Debye)]

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

    def is_stable(self, dt, courant=1.0):
        """
        Check the von Neumann stability of the coupled Yee + ADE update.

        The spectral radius of the single-mode amplification matrix is
        evaluated for all discrete wavenumbers allowed on the grid. The vacuum
        Courant number is ``c dt sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)`` (<= 1).

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
        to (w_0 dt / 2)^2 <= (eps_inf - courant^2) / (eps_s - courant^2).
        """
        for q in np.linspace(0.0, 2 * courant, 101):
            rho = np.max(np.abs(np.linalg.eigvals(_amplification_matrix(self, dt, q))))
            if rho > 1 + 1e-6:
                return False
        return True

    def __repr__(self):
        label = f"{self.name}: " if self.name else ""
        return (
            f"DispersiveMaterial({label}eps_inf={self.eps_inf:.6g}, "
            f"poles={self.poles}, sigma={self.sigma:.6g})"
        )


def _ade_coefficients(material, dt):
    """Per-pole update coefficients (normalized polarization p = P / eps_0)."""
    lorentz = []
    for pole in material.lorentz_poles:
        d = 1 + pole.gamma * dt / 2
        lorentz.append(
            (
                (2 - (pole.omega_0 * dt) ** 2) / d,  # alpha: p^n
                (pole.gamma * dt / 2 - 1) / d,  # xi: p^(n-1)
                pole.strength * dt**2 / d,  # zeta: E^n
            )
        )
    debye = []
    for pole in material.debye_poles:
        r = dt / (2 * pole.tau)
        debye.append(((1 - r) / (1 + r), pole.delta_eps * r / (1 + r)))  # kappa, beta
    return lorentz, debye


def _amplification_matrix(material, dt, q):
    """
    One-step amplification matrix of a plane wave in the dispersive medium.

    State [E^n, h^(n-1/2), p_L^n, p_L^(n-1), ..., p_D^n] in units with
    eps_0 = mu_0 = 1; `q` = c K dt in [0, 2 courant], K the discrete
    wavenumber, so h' = h - q E and eps_inf (E' - E) = q h' - ...
    """
    lorentz, debye = _ade_coefficients(material, dt)
    s_hat = material.sigma * dt / eps_0
    beta_sum = sum(beta for _, beta in debye)
    den = material.eps_inf + s_hat / 2 + beta_sum
    a1 = material.eps_inf - s_hat / 2 - beta_sum
    n = 2 + 2 * len(lorentz) + len(debye)

    def step(x):
        E, h = x[0], x[1]
        y = np.zeros(n)
        y[1] = h - q * E
        acc = a1 * E + q * y[1]
        for k, (alpha, xi, zeta) in enumerate(lorentz):
            p, p_old = x[2 + 2 * k], x[3 + 2 * k]
            p_new = alpha * p + xi * p_old + zeta * E
            acc -= p_new - p
            y[2 + 2 * k], y[3 + 2 * k] = p_new, p
        off = 2 + 2 * len(lorentz)
        for k, (kappa, _) in enumerate(debye):
            acc -= (kappa - 1) * x[off + k]
        y[0] = acc / den
        for k, (kappa, beta) in enumerate(debye):
            y[off + k] = kappa * x[off + k] + beta * (y[0] + E)
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
    `SolverFIT3D` methods for dispersive materials (ADE on selected E edges).

    The non-dispersive time step is run unchanged; afterwards the new E on the
    dispersive edges is recomputed from the curl contribution of that step and
    the auxiliary polarizations. The update is local, linear and explicit.
    """

    def add_dispersive_material(
        self, material, center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0)
    ):
        """
        Fill a box with a dispersive material.

        E edges strictly inside the box get the material; edges lying on a
        face of the box get the arithmetic mean of material and background
        permittivity (weight 1/2 per face, as for tangential E on a material
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
        flat, weights = [], []
        nodes = (self.grid.x, self.grid.y, self.grid.z)
        for c, comp in enumerate("xyz"):
            axis_idx, axis_w = [], []
            for d in range(3):
                pos = (nodes[d][:-1] + nodes[d][1:]) / 2 if d == c else nodes[d][:-1]
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
        self._add_dispersive_edges(
            material, np.concatenate(flat), np.concatenate(weights)
        )

    def _apply_stl_dispersive(self):
        """Attach ADE updates to STL solids whose material is dispersive."""
        for key, material in self.grid.stl_materials.items():
            if not isinstance(material, DispersiveMaterial):
                continue
            # Same volume fraction on the three components and the same
            # (Nx, Ny, Nz) mapping as _apply_stl_materials uses for ieps
            mask = np.reshape(self.grid.grid[key], (self.Nx, self.Ny, self.Nz))
            fraction = np.reshape(mask, self.N, order="F").astype(float)
            fraction = np.tile(fraction, 3)
            idx = np.flatnonzero(fraction)
            # _apply_stl_materials mixes eps with vacuum: eps_bg = 1
            self._add_dispersive_edges(material, idx, fraction[idx], eps_bg=1.0)

    def _add_dispersive_edges(self, material, idx, weight, eps_bg=None):
        """Register flat E indices `idx` with fill fractions `weight`."""
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

        ieps = np.asarray(self.ieps.array.get() if self.use_gpu else self.ieps.array)
        keep = (weight > 0) & (ieps[idx] != 0)  # skip PEC edges
        idx, weight = idx[keep], weight[keep]
        if not hasattr(self, "_dispersive_groups"):
            self._dispersive_groups = []
            self._one_step_nondispersive = self.one_step
            self.one_step = self._one_step_dispersive
        # Later assignments override earlier ones
        for group in self._dispersive_groups:
            old = ~np.isin(group["idx_host"], idx)
            if not old.all():
                _restrict_group(group, old, self.E.xp)

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

    def _one_step_dispersive(self):
        """Non-dispersive step followed by the local ADE correction of E."""
        E_n = [self.E.array[g["idx"]] for g in self._dispersive_groups]
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
        """Zero the auxiliary polarizations (call together with reset_fields)."""
        for g in getattr(self, "_dispersive_groups", []):
            g["lorentz"] = [(a, x, z, 0 * p, 0 * q) for a, x, z, p, q in g["lorentz"]]
            g["debye"] = [(k, b, 0 * p) for k, b, p in g["debye"]]


def _restrict_group(group, keep, xp):
    """Drop the edges of a dispersive group where `keep` is False."""
    keep_dev = xp.asarray(keep)
    group["idx_host"] = group["idx_host"][keep]
    for key in ("idx", "weight", "eps_s", "a1", "den"):
        group[key] = group[key][keep_dev]
    group["lorentz"] = [
        (a, x, z, p[keep_dev], q[keep_dev]) for a, x, z, p, q in group["lorentz"]
    ]
    group["debye"] = [(k, b, p[keep_dev]) for k, b, p in group["debye"]]
