"""
Mie theory reference (wakis.analytic.mie) against independent values.

* Rayleigh limit (x -> 0) of Q_sca and Q_abs, with the leading O(x^2)
  correction bounding the tolerance.
* Coefficients a_n, b_n against a direct evaluation with SciPy's spherical
  Bessel functions (no logarithmic-derivative recurrence).
* Optical theorem Q_ext = 4 Re S(0) / x^2 and energy balance Q_sca = the
  integral of the differential cross section over the sphere.
* Bohren & Huffman's BHMIE example (m = 1.55, r = 0.525 um, lambda =
  0.6328 um): Q_sca = 3.10543, Q_back = 2.92534 (sample output of the program
  in appendix A; values cross-checked here against miepython to 1e-6).
* Extinction paradox: Q_ext -> 2 for large x, with the x^(-2/3) approach.
* miepython (MIT; optional) across dielectric and metallic spheres.

Each test has a failure regime: e.g. replacing the downward recurrence by an
upward one breaks the metal and large-|m x| cases, and a sign error in either
convention branch breaks the absorbing cases.
"""

import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn

from wakis.analytic import (
    mie_coefficients,
    mie_cross_sections,
    mie_differential_cross_section,
    mie_efficiencies,
    mie_s1_s2,
    rayleigh_efficiencies,
)


@pytest.mark.parametrize("m", [2.0, 1.5 + 0.1j, 0.27 + 3.4j])
def test_rayleigh_limit(m):
    """Q_sca and Q_abs approach the dipole limit with an O(x^2) correction."""
    for x in (1e-3, 1e-2):
        q_ext, q_sca, q_abs, _ = mie_efficiencies(m, x)
        r_sca, r_abs = rayleigh_efficiencies(m, x)
        tol = 10 * (abs(m) * x) ** 2
        assert abs(q_sca / r_sca - 1) < tol
        if np.imag(m) != 0:
            assert abs(q_abs / r_abs - 1) < tol
        else:
            assert abs(q_abs) < 1e-12 * q_sca
    # The x^4 law: one decade in x is four decades in Q_sca (n = 2)
    ratio = mie_efficiencies(2.0, 1e-2)[1] / mie_efficiencies(2.0, 1e-3)[1]
    assert abs(np.log10(ratio) - 4) < 1e-3


def _coefficients_direct(m, x, n_max):
    """a_n, b_n from Bohren & Huffman's formulas with SciPy Bessel functions
    (exp(-iwt), Im m >= 0), real m only (SciPy's jn is real-argument)."""
    n = np.arange(1, n_max + 1)
    mx = m * x
    jx, jmx = spherical_jn(n, x), spherical_jn(n, mx)
    djx, djmx = spherical_jn(n, x, True), spherical_jn(n, mx, True)
    hx = jx + 1j * spherical_yn(n, x)
    dhx = djx + 1j * spherical_yn(n, x, True)
    # [x j_n(x)]' = j_n + x j_n'
    dpsi_x, dpsi_mx = jx + x * djx, jmx + mx * djmx
    dxi_x = hx + x * dhx
    a = (m**2 * jmx * dpsi_x - jx * dpsi_mx) / (m**2 * jmx * dxi_x - hx * dpsi_mx)
    b = (jmx * dpsi_x - jx * dpsi_mx) / (jmx * dxi_x - hx * dpsi_mx)
    return a, b


@pytest.mark.parametrize("m, x", [(2.0, 0.7), (2.0, 6.0), (1.33, 20.0), (4.0, 3.0)])
def test_coefficients_against_scipy_bessel(m, x):
    a, b = mie_coefficients(m, x)
    a_ref, b_ref = _coefficients_direct(m, x, a.size)
    np.testing.assert_allclose(a, a_ref, rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(b, b_ref, rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize("m, x", [(2.0, 4.0), (1.5 + 0.1j, 3.0), (0.2 + 3.0j, 1.5)])
def test_optical_theorem_and_angular_integral(m, x):
    q_ext, q_sca, _, q_back = mie_efficiencies(m, x)
    S1, S2 = mie_s1_s2(m, x, 0.0)
    assert np.isclose(S1, S2, rtol=1e-12)
    assert np.isclose(4 * S1.real / x**2, q_ext, rtol=1e-10)
    # Q_sca = (1/x^2) int (|S1|^2 + |S2|^2) sin(theta) dtheta (phi-averaged)
    theta, w = np.polynomial.legendre.leggauss(200)
    S1, S2 = mie_s1_s2(m, x, np.arccos(theta))
    q_int = np.sum(w * (np.abs(S1) ** 2 + np.abs(S2) ** 2)) / x**2
    assert np.isclose(q_int, q_sca, rtol=1e-10)
    # Backscatter: S2(pi) = -S1(pi), Q_back = 4 |S1(pi)|^2 / x^2
    S1, S2 = mie_s1_s2(m, x, np.pi)
    assert np.isclose(S1, -S2, rtol=1e-10)
    assert np.isclose(4 * abs(S1) ** 2 / x**2, q_back, rtol=1e-10)


def test_bohren_huffman_example():
    x = 2 * np.pi * 0.525 / 0.6328
    q_ext, q_sca, q_abs, q_back = mie_efficiencies(1.55, x)
    assert abs(q_sca - 3.10543) < 1e-5
    assert abs(q_ext - 3.10543) < 1e-5
    assert abs(q_back - 2.92534) < 1e-5
    assert abs(q_abs) < 1e-12


def test_extinction_paradox():
    # Q_ext -> 2 with an edge (glory/surface-wave) term ~ 2 x^(-2/3)
    # (van de Hulst's asymptotics): fit the slope between x = 500 and 8000
    x = np.array([500.0, 8000.0])
    excess = mie_efficiencies(1.5 + 0.01j, x)[0] - 2
    assert np.all(excess > 0) and np.all(excess < 2.5 * x ** (-2 / 3))
    slope = np.log(excess[1] / excess[0]) / np.log(x[1] / x[0])
    assert abs(slope + 2 / 3) < 0.05
    # Not trivially 2 at moderate x (rule 1: the check can fail)
    assert abs(mie_efficiencies(1.5, 5.0)[0] - 2) > 0.5


def test_conventions_and_cross_sections():
    """exp(+iwt) input (Im m < 0) gives the same efficiencies and conjugate
    amplitudes; cross sections scale with pi r^2 and the medium index."""
    q_bh = mie_efficiencies(0.2 + 3.0j, 1.5)
    q_eng = mie_efficiencies(0.2 - 3.0j, 1.5)
    np.testing.assert_allclose(q_bh, q_eng, rtol=1e-14)
    assert q_bh[2] > 0.1  # absorbing
    a, b = mie_coefficients(0.2 + 3.0j, 1.5)
    a2, b2 = mie_coefficients(0.2 - 3.0j, 1.5)
    np.testing.assert_allclose(a2, a.conj())
    r, lam = 1e-6, np.array([2e-6, 3e-6])
    sig = mie_cross_sections(2.0, r, lam)
    q = mie_efficiencies(2.0, 2 * np.pi * r / lam)
    np.testing.assert_allclose(sig[1], q[1] * np.pi * r**2)
    # In water (n = 1.33) the size parameter and relative index both change
    sig_w = mie_cross_sections(2.0, r, lam, n_medium=1.33)
    q_w = mie_efficiencies(2.0 / 1.33, 2 * np.pi * 1.33 * r / lam)
    np.testing.assert_allclose(sig_w[1], q_w[1] * np.pi * r**2)
    # Differential cross section integrates to sigma_sca
    mu, w = np.polynomial.legendre.leggauss(120)
    phi = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    dsig = mie_differential_cross_section(
        2.0, r, 3e-6, np.arccos(mu)[:, None], phi[None, :]
    )
    total = np.sum(w[:, None] * dsig) * 2 * np.pi / phi.size
    assert np.isclose(total, sig[1][1], rtol=1e-10)


@pytest.mark.parametrize(
    "m, x",
    [
        (2.0, 0.5),
        (2.0, 8.0),
        (1.5 - 0.1j, 10.0),
        (0.05 - 4.0j, 0.8),  # silver-like (exp(+iwt) sign as in miepython)
        (0.2 - 3.0j, 30.0),
        (10.0 - 10.0j, 5.0),  # large |m x|: needs the downward recurrence
    ],
)
def test_against_miepython(m, x):
    miepython = pytest.importorskip("miepython")
    q_ext, q_sca, q_back, _ = miepython.efficiencies_mx(m, x)
    mine = mie_efficiencies(m, x)
    np.testing.assert_allclose(
        [mine[0], mine[1], mine[3]], [q_ext, q_sca, q_back], rtol=2e-5
    )
