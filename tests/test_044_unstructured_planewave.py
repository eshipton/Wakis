"""
Scattered-field plane wave on the triangular FIT mesh
(`UnstructuredFIT2D.add_plane_wave`) against the 2D Mie series.

A dielectric cylinder (n = 2, radius a) on an interface-conforming Delaunay
mesh, first-order ABC on a circle; the scattered field is DFT'd on three
circles and decomposed into outgoing/incoming cylindrical harmonics
(`wakis.cylinder_harmonics`). Checks:

* x = ka = 0.1, 0.3, 1.5: Q_sca matches the series to 3 %;
* thin cylinder (x = 0.1): Q_sca matches the Rayleigh (line-dipole) limit
  to 12 % (the series itself deviates from it by +7.9 % TM, +1.1 % TE), TM
  and TE: the TE value carries the 2D depolarization factor
  |(m^2-1)/(m^2+1)|^2, which a source that ignores normal-D continuity
  misses;
* x = 1.5: the complex coefficients c_0, c_1 (phase included) match the
  series to 5 %;
* mutations fail: a flipped TE edge normal (sign of c_n), the TE source
  weighted as if the normal E were continuous, and a TM source weighted by
  eps instead of the contrast eps - eps_b;
* no scatterer: the scattered field stays exactly zero.
"""

import numpy as np
import pytest
from scipy.constants import c as C0

from wakis.analytic import cylinder_coefficients, cylinder_rayleigh_efficiencies
from wakis.cylinder_harmonics import (
    circle_points,
    fit_harmonics,
    mie_coefficients_from_fit,
    project_harmonics,
    q_sca_from_coefficients,
)
from wakis.sources import GaussianSource
from wakis.unstructured import UnstructuredFIT2D
from wakis.unstructured.meshers import disk_mesh

A = 1.0e-6
N_CYL = 2.0
RADII = np.array([2.0, 2.5, 3.0]) * A
M_PTS = 64
N_MAX = 6
X_TEST = np.array([0.1, 0.3, 1.5])


def _run(pol, mutate=None, eps=N_CYL**2, h=A / 10, mesh=None):
    mesh = disk_mesh([A, 3.5 * A], h) if mesh is None else mesh
    solver = UnstructuredFIT2D(mesh, {0: [eps, 1.0]}, polarization=pol, bc="abc")
    f = X_TEST * C0 / (2 * np.pi * A)
    src = GaussianSource(0.9 * C0 / (2 * np.pi * A), 0.45 * C0 / (2 * np.pi * A))
    solver.add_plane_wave(src)
    pw = solver.plane_wave
    if mutate == "te_normal":
        pw["coef"] = -pw["coef"]
    elif mutate == "te_continuous_E":
        # weight (eps_T - eps_b) / eps_b: as if E_n (not D_n) were continuous
        T = solver.mesh.dual_edge_partial.tocoo()
        keep = (~np.isclose(solver.eps_r[T.col], 1.0)) & (T.data != 0)
        e, c, h_ = T.row[keep], T.col[keep], T.data[keep]
        t = solver.mesh.edge_tangent[e]
        n_e = np.stack([t[:, 1], -t[:, 0]], 1)
        pw["coef"] = h_ * (solver.eps_r[c] - 1.0) * (n_e @ np.array([0.0, 1.0]))
    elif mutate == "tm_full_mass":
        # current weighted by eps_T instead of the contrast eps_T - eps_b
        pw["w"] = solver.mass[pw["idx"]]
    if mutate is not None:
        solver._build_pw_table()  # the source table folds coef / w in
    pts = np.vstack([circle_points(r, M_PTS)[0] for r in RADII])
    P = solver._interp(pts)
    dt = solver.dt
    shift = 0.0 if pol == "TM" else -0.5 * dt
    n_steps = int((src.end_time + 3e-13) / dt)
    acc = np.zeros((f.size, len(pts)), complex)
    times = (np.arange(n_steps) + 1) * dt + shift
    peak = 0.0
    for n in range(n_steps):
        solver.one_step()
        acc += np.outer(np.exp(-2j * np.pi * f * times[n]), P @ solver.u) * dt
        peak = max(peak, np.abs(solver.u).max())
    u_inc = solver.incident_field(np.zeros((1, 2)), times)
    u0 = np.exp(-2j * np.pi * np.outer(f, times)) @ u_inc * dt
    F = project_harmonics(acc.reshape(f.size, len(RADII), M_PTS), N_MAX)
    k = 2 * np.pi * f / C0
    alpha, gamma, _ = fit_harmonics(F, k, RADII, N_MAX)
    c = mie_coefficients_from_fit(alpha, gamma, np.conj(u0), N_MAX)
    return c, q_sca_from_coefficients(c, X_TEST), peak


def _series(pol):
    c = np.array([cylinder_coefficients(N_CYL, x, pol, n_terms=N_MAX) for x in X_TEST])
    w = np.where(np.arange(N_MAX + 1) > 0, 2.0, 1.0)
    return c, 2 / X_TEST * np.sum(w * np.abs(c) ** 2, 1)


def _checks(pol, c, q):
    c_ref, q_ref = _series(pol)
    q_ray = cylinder_rayleigh_efficiencies(N_CYL, X_TEST[0], pol)
    c_pos = c[:, N_MAX:]  # n = 0..N_MAX
    err_q = np.abs(q / q_ref - 1)
    err_ray = abs(q[0] / q_ray - 1)
    err_c = np.abs(c_pos[-1, :2] - c_ref[-1, :2]) / np.abs(c_ref[-1, :2]).max()
    return err_q, err_ray, err_c


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_planewave_mie_series(pol):
    c, q, _ = _run(pol)
    err_q, err_ray, err_c = _checks(pol, c, q)
    print(f"{pol}: Q_sca err {err_q}, Rayleigh err {err_ray:.3f}, c_0,1 err {err_c}")
    assert np.all(err_q < 0.03)
    assert err_ray < 0.12
    assert np.all(err_c < 0.05)


@pytest.mark.parametrize(
    "pol, mutate",
    [("TE", "te_normal"), ("TE", "te_continuous_E"), ("TM", "tm_full_mass")],
)
def test_planewave_mutations_fail(pol, mutate):
    c, q, _ = _run(pol, mutate)
    err_q, err_ray, err_c = _checks(pol, c, q)
    print(f"{pol} {mutate}: Q_sca err {err_q}, Rayleigh {err_ray:.3f}, c err {err_c}")
    assert np.any(err_q > 0.05) or np.any(err_c > 0.1)


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_planewave_no_scatterer(pol):
    _, _, peak = _run(pol, eps=1.0, h=A / 5)
    assert peak == 0.0
