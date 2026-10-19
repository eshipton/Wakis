"""
Thin-wire dipole reference (wakis.analytic.thin_wire) for benchmark B4:
induced-EMF limit, energy balance (input power = radiated power from the
current), convergence in N, the finite-gap source and mutation checks.
"""

import numpy as np
import pytest
from scipy.constants import c

from wakis.analytic import thin_wire as tw


def test_induced_emf_half_wave_classic_value():
    # Balanis sec. 4.6: 73.1 + j 42.5 Ohm for an infinitely thin lambda/2 dipole
    z = tw.induced_emf_impedance(0.5, 1e-9, c)
    assert abs(z.real - 73.08) < 0.05 and abs(z.imag - 42.51) < 0.05


def test_hallen_approaches_induced_emf_for_thin_wires():
    # The sinusoidal current is the thin-wire limit: the MoM impedance of a
    # lambda/2 dipole tends to 73 + j 42.5 as a -> 0 (error ~ 1 / Omega).
    errs = []
    for a in (1e-3, 1e-4, 1e-6):
        z = tw.hallen_dipole(0.5, a, c, 100)[0][0]
        errs.append(abs(z - (73.08 + 42.51j)) / 84.5)
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 0.06


@pytest.mark.parametrize("gap", [0.0, 2e-3])
def test_energy_balance_and_directivity(gap):
    f = 0.95 * c / 0.3
    z, zn, I = tw.hallen_dipole(0.15, 0.5e-3, f, 150, gap=gap)
    U, P = tw.dipole_far_field(zn, I[0], f, [np.pi / 2])
    p_in = 0.5 * np.real(1.0 * np.conj(I[0][len(zn) // 2]))
    assert abs(P / p_in - 1) < 2e-3
    assert 1.60 < 4 * np.pi * U[0] / P < 1.66


def test_mutation_wrong_source_strength_breaks_energy_balance(monkeypatch):
    # A wrong generator strength in Hallen's right-hand side (eta doubled)
    # scales I by 1/2: P_rad ~ |I|^2 no longer equals Re(V I*)/2. (A wrong
    # *reactive* kernel term is not caught by this test: power balance only
    # probes the radiative part, hence the separate convergence tests.)
    f = 0.95 * c / 0.3
    monkeypatch.setattr(tw, "ETA0", 2 * tw.ETA0)
    z, zn, I = tw.hallen_dipole(0.15, 0.5e-3, f, 150)
    monkeypatch.setattr(tw, "ETA0", tw.ETA0 / 2)
    U, P = tw.dipole_far_field(zn, I[0], f, [np.pi / 2])
    p_in = 0.5 * np.real(np.conj(I[0][len(zn) // 2]))
    assert abs(P / p_in - 1) > 0.3


def test_gap_source_limits():
    k, z = 20.0, np.linspace(-0.07, 0.07, 41)
    assert np.allclose(tw.gap_excitation(z, k, 1e-9), 0.5 * np.sin(k * np.abs(z)),
                       atol=1e-9)  # fmt: skip
    # continuity at the gap edge
    g = 0.02
    e = tw.gap_excitation(np.array([g / 2 - 1e-12, g / 2 + 1e-12]), k, g)
    assert abs(e[0] - e[1]) < 1e-9


def test_resonance_converges_in_n():
    from scipy.optimize import brentq

    fr = [brentq(lambda f: tw.hallen_dipole(0.15, 0.5e-3, f, n, gap=1e-3)[0][0].imag,
                 0.85e9, 1.0e9, xtol=1e3) for n in (150, 200, 300)]  # fmt: skip
    assert max(fr) / min(fr) - 1 < 2e-3
    assert 0.46 < 0.15 * fr[1] / c < 0.48  # thick-wire shortening (Omega = 11.4)
