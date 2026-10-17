# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Helpers for Bloch-periodic (Floquet) unit-cell simulations.

A Bloch-periodic cell is set up with ``bc_low``/``bc_high`` = ``'bloch'`` on
the periodic axes, a Bloch wavevector ``k_bloch`` [rad/m] and complex fields
(``dtype=np.complex128``) in ``SolverFIT3D``. An oblique plane wave is then
excited by a sheet ``CurrentSource`` with ``amp_func = exp(1j k_par . r)``
and ``k_bloch = k_par``. Because ``k_par`` is fixed, the angle of incidence
varies with frequency; these helpers convert between the two.
"""

import numpy as np
from scipy.constants import c


def k_parallel(frequency, theta, phi=0.0, n=1.0):
    """
    Tangential wavevector of a plane wave incident at a given angle.

    Parameters
    ----------
    frequency : float
        Frequency at which the wave hits the cell at angle ``theta`` [Hz].
    theta : float
        Polar angle of incidence from the cell normal (z) [rad].
    phi : float, optional
        Azimuth of the plane of incidence from the x axis [rad]. Default 0.
    n : float, optional
        Refractive index of the incidence medium. Default 1.

    Returns
    -------
    ndarray
        (kx, ky, 0) [rad/m], to pass as ``k_bloch`` and use in the source
        profile exp(1j (kx x + ky y)).
    """
    k = 2 * np.pi * frequency * n / c * np.sin(theta)
    return np.array([k * np.cos(phi), k * np.sin(phi), 0.0])


def incidence_angle(k_par, frequency, n=1.0):
    """
    Angle of incidence at each frequency for a fixed tangential wavevector.

    Phase matching fixes the tangential wavevector, so the polar angle is
    theta(f) = arcsin(c |k_par| / (2 pi f n)). Below the cutoff frequency
    f_c = c |k_par| / (2 pi n) the wave is evanescent and NaN is returned.

    Parameters
    ----------
    k_par : sequence of float
        Tangential wavevector (kx, ky[, kz]) [rad/m]; only its norm is used.
    frequency : float or array_like
        Frequency [Hz]. Negative frequencies give the same angle.
    n : float, optional
        Refractive index of the incidence medium. Default 1.

    Returns
    -------
    float or ndarray
        Polar angle of incidence [rad].
    """
    s = c * np.linalg.norm(k_par) / (2 * np.pi * np.abs(np.asarray(frequency)) * n)
    return np.arcsin(np.where(s <= 1, s, np.nan))
