"""
Scattered-field plane wave on dispersive scatterers in UnstructuredFIT2D
(wakis.unstructured.dispersive_planewave), on the B2-Au case (Rakic gold
cylinder, a = 20 nm, host n = 2) at a coarse mesh (h = 4 nm, ~1 s per run):

* spectra against the exact series (tolerances = 2x the errors measured at
  this mesh; the error converges with h, see studies/benchmarks);
* energy balance: the power absorbed in the metal, (w eps_0 / 2) int
  (-Im eps) |E|^2 dA from the DFT of the *total* field inside, equals the
  series C_abs and the extracted C_ext - C_sca;
* a lossless Lorentz scatterer (eps ~ 9) scatters as the series predicts
  (Q_sca; its Q_ext - Q_sca is a small difference of nearly equal terms for
  this weak scatterer, set by the phase error of the coarse outer mesh, so
  it is not a usable energy check at this resolution);
* mutation: driving the poles with the scattered field only (the incident
  field removed from the ADE drive) must fail the comparison.
"""

import numpy as np
import pytest
from scipy.constants import epsilon_0 as eps0
from scipy.constants import mu_0

from wakis.benchmarks import get_case
from wakis.benchmarks.arms.triangles2d import Triangles2DArm
from wakis.cylinder_harmonics import circle_points
from wakis.dispersive import DispersiveMaterial, Lorentz
from wakis.unstructured.dispersive_planewave import absorbed_power_weights


def _run(pol, material=None, mutate=False, h=4e-9):
    case = get_case(f"B2-Au-{pol}")
    case.wavelengths = case.wavelengths[::10]
    case.t_run = 40e-15
    if material is not None:
        case.materials = {"Au": material}
    mesh, s, _ = Triangles2DArm().build(case, h)
    if mutate:
        if pol == "TM":
            for tb in s._update._tables:
                tb.w[:] = 0
        else:
            for g in s._te_groups:
                g["einc"].w[:] = 0
    pts = np.vstack([circle_points(r, case.n_points)[0] for r in case.circles])
    P = s._interp(pts)
    f, dt = case.freqs, s.dt
    shift = 0.0 if pol == "TM" else -0.5 * dt
    mat, _, idx, w = absorbed_power_weights(s)[0]
    acc = np.zeros((len(f), len(pts)), complex)
    acc_v = np.zeros((len(f), len(idx)), complex)
    if pol == "TM":
        xn = mesh.nodes[s.free[s._update.groups[0]["idx"]]]
    n = int(case.t_run / dt)
    for i in range(n):
        s.one_step()
        t = (i + 1) * dt
        acc += np.outer(np.exp(-2j * np.pi * f * (t + shift)), P @ s.u) * dt
        if pol == "TM":
            e = s.u[s._update.groups[0]["idx"]] + s.incident_field(xn, t)
        else:
            e = s._te_groups[0]["E"]
        acc_v += np.outer(np.exp(-2j * np.pi * f * t), e) * dt
    times = (np.arange(n) + 1) * dt + shift
    u0 = (
        np.exp(-2j * np.pi * np.outer(f, times))
        @ s.incident_field(np.zeros((1, 2)), times)
    ) * dt
    sp = case.spectra_from_result(
        dict(samples=acc.reshape(len(f), len(case.circles), case.n_points),
             u0=u0, radii=case.circles)
    )  # fmt: skip
    eta = np.sqrt(mu_0 / (eps0 * case.n_background**2))
    E0 = u0 if pol == "TM" else u0 * eta
    p_abs = (
        0.5 * 2 * np.pi * f * eps0 * (-mat.epsilon(f).imag) * (np.abs(acc_v) ** 2 @ w)
    )
    c_abs_vol = p_abs / (np.abs(E0) ** 2 / (2 * eta))
    case.__dict__.pop("reference", None)
    return case, sp, c_abs_vol


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_gold_cylinder_against_series_and_energy_balance(pol):
    case, sp, c_vol = _run(pol)
    ref = case.reference
    err_ext = np.abs(sp["c_ext"] / ref["c_ext"] - 1).max()
    assert err_ext < 0.05, err_ext
    # absorbed power inside the metal = exact C_abs
    assert np.allclose(c_vol, ref["c_abs"], rtol=0.03), c_vol / ref["c_abs"]
    # ... and = extinction - scattering of the same simulation
    tol = 0.05 if pol == "TE" else 0.12  # TM: C_abs = small difference of large terms
    assert np.allclose(c_vol, sp["c_abs"], rtol=tol), c_vol / sp["c_abs"]


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_mutation_scattered_field_drive_fails(pol):
    case, sp, c_vol = _run(pol, mutate=True)
    ref = case.reference
    assert np.abs(sp["c_ext"] / ref["c_ext"] - 1).max() > 0.3


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_lossless_lorentz_scatterer(pol):
    lor = DispersiveMaterial(1.0, [Lorentz(8.0, 2 * np.pi * 1500e12, 0.0)])
    case, sp, c_vol = _run(pol, material=lor)
    ref = case.reference
    assert np.allclose(sp["c_sca"], ref["c_sca"], rtol=0.08)  # measured 3.8 %
    assert np.all(c_vol == 0)  # no loss term at all
