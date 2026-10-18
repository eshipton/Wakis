# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Cold-plasma helpers and the linear plasma wakefield of a rigid bunch.

A cold, collisionless, unmagnetized electron plasma with immobile ions is the
Drude medium eps(w) = 1 - w_p^2 / w^2 (`cold_plasma`). A rigid bunch moving
at v = c along z drives a plasma wake that depends on x, y and the co-moving
coordinate xi = z - c t only. For a bunch of line density lambda(xi) at the
transverse position r0 inside a perfectly conducting pipe of radius `a`
filled with plasma, linear cold-fluid theory gives (see the physics guide,
"Linear plasma wakefield")::

    E_z(r, xi) = -(k_p^2 / (2 pi eps_0)) g(r; r0)
                 * int_xi^inf lambda(xi') cos(k_p (xi - xi')) dxi'

    F_perp(r, xi) = (k_p / (2 pi eps_0)) grad_perp g(r; r0)
                    * int_xi^inf lambda(xi') sin(k_p (xi' - xi)) dxi'

with k_p = w_p / c and the transverse Green function
g = K_0(k_p |r - r0|) - sum_m eps_m cos(m (phi - phi0)) I_m(k_p r0)
I_m(k_p r) K_m(k_p a) / I_m(k_p a) (eps_0 = 1, eps_m = 2), which vanishes at
the pipe wall. F_perp = E_perp + c e_z x B is the transverse Lorentz force
per unit charge on a test particle moving at c.
"""

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as q_e
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import m_e
from scipy.special import erfc, ive, k0, kve

from .dispersive import DispersiveMaterial, Drude


def plasma_frequency(density):
    """
    Electron plasma angular frequency w_p = sqrt(n e^2 / (eps_0 m_e)).

    Parameters
    ----------
    density : float or array_like
        Electron density n [1/m^3].

    Returns
    -------
    omega_p : float or ndarray
        Plasma angular frequency [rad/s].
    """
    return np.sqrt(np.asarray(density, dtype=float) * q_e**2 / (eps_0 * m_e))


def cold_plasma(omega_p, collision_rate=0.0, name="cold plasma"):
    """
    Cold electron plasma with immobile ions as a Drude `DispersiveMaterial`.

    eps(w) = 1 - w_p^2 / (w^2 - i nu w) in the Wakis exp(+i w t) convention.
    The Drude ADE is the linearized cold-fluid momentum equation
    dJ/dt + nu J = eps_0 w_p^2 E for the plasma current J.

    Parameters
    ----------
    omega_p : float
        Plasma angular frequency [rad/s] (see `plasma_frequency`).
    collision_rate : float, optional
        Electron collision rate nu [1/s]; 0 for a collisionless plasma.
    name : str, optional
        Material name.

    Returns
    -------
    DispersiveMaterial
    """
    return DispersiveMaterial(1.0, [Drude(omega_p, collision_rate)], name=name)


def pipe_green_function(x, y, k_p, radius=np.inf, x0=0.0, y0=0.0, m_max=80):
    """
    Transverse Green function g of the screened Poisson operator in a pipe.

    g solves (lap_perp - k_p^2) g = -2 pi delta(r - r0) with g = 0 at the
    wall r = `radius`; g = K_0(k_p |r - r0|) without a wall.

    Parameters
    ----------
    x, y : float or array_like
        Observation point [m] (inside the pipe, r != r0).
    k_p : float
        Plasma wavenumber w_p / c [1/m].
    radius : float, optional
        Pipe radius a [m]; ``np.inf`` for an unbounded plasma.
    x0, y0 : float, optional
        Source (bunch) position [m].
    m_max : int, optional
        Highest azimuthal order of the image series (it converges like
        (r r0 / a^2)^m).

    Returns
    -------
    g : ndarray
        Dimensionless Green function.
    """
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    g = k0(k_p * np.hypot(x - x0, y - y0))
    if np.isinf(radius):
        return g
    r, r0 = np.hypot(x, y), np.hypot(x0, y0)
    dphi = np.arctan2(y, x) - np.arctan2(y0, x0)
    orders = np.arange(m_max + 1) if r0 > 0 else np.arange(1)
    for m in orders:
        # I_m(k r0) I_m(k r) K_m(k a) / I_m(k a) with exponentially scaled
        # Bessel functions, so that large k_p a does not overflow
        term = (
            ive(m, k_p * r0)
            * ive(m, k_p * r)
            * kve(m, k_p * radius)
            / ive(m, k_p * radius)
            * np.exp(k_p * (r + r0 - 2 * radius))
        )
        g = g - (1.0 if m == 0 else 2.0) * np.cos(m * dphi) * term
    return g


def _gaussian_integrals(xi, k_p, sigmaz):
    """
    int_xi^inf lambda(xi') exp(-i k_p xi') dxi' for a unit-charge Gaussian
    line density centred at 0, times exp(i k_p xi): returns (C, S) with
    C = int lambda cos(k_p (xi - xi')), S = int lambda sin(k_p (xi' - xi)).
    """
    xi = np.asarray(xi, dtype=float)
    damp = np.exp(-((k_p * sigmaz) ** 2) / 2)
    tail = 0.5 * erfc((xi + 1j * k_p * sigmaz**2) / (np.sqrt(2) * sigmaz))
    integral = np.exp(1j * k_p * xi) * damp * tail
    return integral.real, -integral.imag


def linear_wake(x, y, xi, k_p, sigmaz, q=1.0, radius=np.inf, x0=0.0, y0=0.0, m_max=80):
    """
    Linear plasma wakefield of a rigid Gaussian bunch moving at c.

    Parameters
    ----------
    x, y : float or array_like
        Transverse observation point [m] (r != r0); broadcast with `xi`.
    xi : float or array_like
        Co-moving coordinate z - c t [m] measured from the bunch centre
        (xi < 0 behind the bunch).
    k_p : float
        Plasma wavenumber w_p / c [1/m].
    sigmaz : float
        RMS bunch length [m].
    q : float, optional
        Bunch charge [C] (negative for electrons).
    radius : float, optional
        Radius of the perfectly conducting pipe [m]; ``np.inf`` for an
        unbounded plasma.
    x0, y0 : float, optional
        Transverse bunch position [m].
    m_max : int, optional
        Azimuthal orders of the image series (see `pipe_green_function`).

    Returns
    -------
    Ez, Fx, Fy : ndarray
        Longitudinal field and transverse force per unit test charge
        F_x = E_x - c B_y, F_y = E_y + c B_x [V/m].

    Notes
    -----
    Far behind the bunch the wake oscillates without decay with amplitude
    q k_p^2 g / (2 pi eps_0) exp(-k_p^2 sigmaz^2 / 2), maximal at
    k_p sigmaz = sqrt(2) for fixed bunch length and fixed k_p r.
    """
    C, S = _gaussian_integrals(xi, k_p, sigmaz)
    args = dict(k_p=k_p, radius=radius, x0=x0, y0=y0, m_max=m_max)
    g = pipe_green_function(x, y, **args)
    # grad g by central differences of the closed form (error ~ h^2 g''')
    h = 1e-5 / k_p
    gx = pipe_green_function(x + h, y, **args) - pipe_green_function(x - h, y, **args)
    gy = pipe_green_function(x, y + h, **args) - pipe_green_function(x, y - h, **args)
    scale = q * k_p / (2 * np.pi * eps_0)
    Ez = -scale * k_p * g * C
    Fx = scale * gx / (2 * h) * S
    Fy = scale * gy / (2 * h) * S
    return Ez, Fx, Fy


def wake_amplitude(r, k_p, sigmaz, q=1.0, radius=np.inf):
    """
    Amplitude of the on-axis-driven longitudinal wake far behind the bunch,
    q k_p^2 g(r) exp(-k_p^2 sigmaz^2 / 2) / (2 pi eps_0) [V/m].
    """
    g = pipe_green_function(r, 0.0, k_p, radius)
    return q * k_p**2 * g * np.exp(-((k_p * sigmaz) ** 2) / 2) / (2 * np.pi * eps_0)


def plasma_wavelength(omega_p):
    """Plasma wavelength 2 pi c / w_p [m]."""
    return 2 * np.pi * c_light / omega_p
