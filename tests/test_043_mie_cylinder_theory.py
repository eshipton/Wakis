"""
2D Mie theory (infinite cylinder, normal incidence; wakis.analytic.mie_cylinder)
against independent values.

* Thin-cylinder limits: Q_sca = pi^2 x^3 |m^2-1|^2 / 8 (TM, E along the axis)
  and pi^2 x^3 |(m^2-1)/(m^2+1)|^2 / 4 (TE), derived from line-dipole
  radiation (not from the series).
* Coefficients against a direct evaluation with SciPy's J_n(m x) (no
  recurrence).
* PEC limit (|m| -> infinity, lossy): b_n -> J_n(x)/H_n(x) (TM) and
  a_n -> J_n'(x)/H_n'(x) (TE), which also exercises the downward recurrence
  at |m x| ~ 3e4 where a direct J_n(m x) overflows.
* Energy: Q_ext = Q_sca for real m; the angular integral of |T|^2 equals
  Q_sca; absorption > 0 for lossy m in either time convention.
* Extinction paradox: Q_ext - 2 ~ x^(-2/3) for large x.
"""

import numpy as np
import pytest
from scipy.special import jv, jvp, yv, yvp

from wakis.analytic.mie_cylinder import (
    cylinder_amplitude,
    cylinder_coefficients,
    cylinder_efficiencies,
    cylinder_rayleigh_efficiencies,
    cylinder_scattering_width,
)


@pytest.mark.parametrize("pol", ["TM", "TE"])
@pytest.mark.parametrize("m", [2.0, 1.5 + 0.2j])
def test_thin_cylinder_limit(pol, m):
    for x in (1e-3, 1e-2):
        q_sca = cylinder_efficiencies(m, x, pol)[1]
        ref = cylinder_rayleigh_efficiencies(m, x, pol)
        assert abs(q_sca / ref - 1) < 20 * (abs(m) * x) ** 2 * (1 + abs(np.log(x)))
    # x^3 law (2D) rather than the x^4 law of a sphere
    r = (
        cylinder_efficiencies(2.0, 1e-2, pol)[1]
        / cylinder_efficiencies(2.0, 1e-3, pol)[1]
    )
    assert abs(np.log10(r) - 3) < 2e-3


def _direct(m, x, n_max):
    n = np.arange(n_max + 1)
    J, dJ, Jm, dJm = jv(n, x), jvp(n, x), jv(n, m * x), jvp(n, m * x)
    H, dH = J + 1j * yv(n, x), dJ + 1j * yvp(n, x)
    b = (Jm * dJ - m * dJm * J) / (Jm * dH - m * dJm * H)
    a = (m * dJ * Jm - J * dJm) / (m * Jm * dH - dJm * H)
    return b, a


@pytest.mark.parametrize("m, x", [(2.0, 0.8), (2.0, 7.0), (1.33, 25.0), (3.5, 4.0)])
def test_coefficients_against_direct_bessel(m, x):
    b = cylinder_coefficients(m, x, "TM")
    a = cylinder_coefficients(m, x, "TE")
    b_ref, a_ref = _direct(m, x, b.size - 1)
    np.testing.assert_allclose(b, b_ref, rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(a, a_ref, rtol=1e-9, atol=1e-13)


def test_pec_limit():
    x = 2.0
    m = 1e4 * (1 + 1j)  # good conductor: skin depth << radius
    b = cylinder_coefficients(m, x, "TM")
    a = cylinder_coefficients(m, x, "TE")
    n = np.arange(b.size)
    np.testing.assert_allclose(b, jv(n, x) / (jv(n, x) + 1j * yv(n, x)), atol=2e-4)
    np.testing.assert_allclose(a, jvp(n, x) / (jvp(n, x) + 1j * yvp(n, x)), atol=2e-4)


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_energy_and_angular_integral(pol):
    q_ext, q_sca, q_abs = cylinder_efficiencies(2.0, 6.0, pol)
    assert abs(q_abs) < 1e-12 and q_sca > 0.5
    theta = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    for m in (2.0, 1.5 + 0.2j):
        q_ext, q_sca, q_abs = cylinder_efficiencies(m, 6.0, pol)
        T = cylinder_amplitude(m, 6.0, theta, pol)
        q_int = np.mean(np.abs(T) ** 2) * 2 * np.pi / (np.pi * 6.0)
        assert np.isclose(q_int, q_sca, rtol=1e-10)
        # forward amplitude: Q_ext = (2/x) Re T(0)
        assert np.isclose(2 / 6.0 * cylinder_amplitude(m, 6.0, 0.0, pol).real, q_ext)
    lossy = cylinder_efficiencies(1.5 + 0.2j, 6.0, pol)
    assert lossy[2] > 0.1
    np.testing.assert_allclose(cylinder_efficiencies(1.5 - 0.2j, 6.0, pol), lossy)
    # Scattering width [m] = Q * 2a, and the differential width integrates to it
    a, lam = 1e-6, 1.2e-6
    C = cylinder_scattering_width(2.0, a, lam, polarization=pol)
    x = 2 * np.pi * a / lam
    assert np.isclose(C, cylinder_efficiencies(2.0, x, pol)[1] * 2 * a)
    dC = cylinder_scattering_width(2.0, a, lam, theta, polarization=pol)
    assert np.isclose(np.mean(dC) * 2 * np.pi, C, rtol=1e-10)


def test_polarizations_differ_and_extinction_paradox():
    # TE and TM must not be confused (rule 1: a swapped label fails here)
    assert (
        abs(
            cylinder_efficiencies(2.0, 0.3, "TM")[1]
            / cylinder_efficiencies(2.0, 0.3, "TE")[1]
        )
        > 5
    )
    x = np.array([1000.0, 16000.0])
    for pol in ("TM", "TE"):
        excess = cylinder_efficiencies(1.5 + 0.05j, x, pol)[0] - 2
        assert np.all(excess > 0)
        slope = np.log(excess[1] / excess[0]) / np.log(x[1] / x[0])
        assert abs(slope + 2 / 3) < 0.05
