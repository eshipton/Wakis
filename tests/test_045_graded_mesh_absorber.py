"""
Graded triangle meshes (`graded_disk_mesh`) and the matched absorbing layer
of `UnstructuredFIT2D` (``absorber=``).

* Graded mesh (h inside the n = 2 cylinder, 2h in vacuum): Delaunay with
  positive Voronoi masses, at least 2x fewer nodes than the uniform
  `disk_mesh`, interface ring on r = a, and plane-wave Q_sca within 1 % of
  the series at x = 0.1, 0.3, 1.5 (recorded 0.1-0.4 %), c_0, c_1 within 3 %.
  An abrupt 4x size jump at the interface (no grading) must be > 3x worse at
  x = 0.1, 0.3 (recorded ~6x; mutation).
* Absorber (0.8 a thick, s_max = 6 c / d, quadratic): for an outgoing
  cylindrical wave from a line source the reflection |gamma_0 / alpha_0| at
  x = 6..10 drops at least 3x against the bare first-order ABC (recorded
  1.4-2.1 % -> 0.1-0.3 %); an electric-only (unmatched) layer reflects more
  than twice as much as the matched one (mutation).
"""

import numpy as np
from scipy.constants import c as C0
from test_044_unstructured_planewave import A, _checks, _run

from wakis.cylinder_harmonics import circle_points, fit_harmonics, project_harmonics
from wakis.sources import GaussianSource
from wakis.unstructured import UnstructuredFIT2D, disk_mesh, graded_disk_mesh

R_OUT = 3.5 * A


def _size(h, h_out, band=None, q=1.2):
    band = 2 * h if band is None else band

    def size(r):
        d = r - A - band
        return h if d <= 0 else min(h_out, h * q ** (d / h))

    return size


def test_graded_mesh_quality():
    h = A / 10
    m = graded_disk_mesh([A, R_OUT], _size(h, 2 * h))
    u = disk_mesh([A, R_OUT], h)
    s = UnstructuredFIT2D(m, {0: [4.0, 1.0]}, polarization="TM", bc="abc")
    assert np.all(s.mass > 0)
    assert m.n_nodes < 0.5 * u.n_nodes
    ring = m.nodes[np.isclose(np.hypot(*m.nodes.T), np.hypot(*m.nodes.T)[0])]
    assert len(ring) > 8
    area = m.cell_volume[m.cell_tags == 0].sum()
    assert abs(area / (np.pi * A**2) - 1) < 1e-10  # area-preserving ring


def test_graded_mesh_mie():
    h = A / 10
    errs = {}
    for pol in ("TM", "TE"):
        m = graded_disk_mesh([A, R_OUT], _size(h, 2 * h))
        c, q, _ = _run(pol, mesh=m)
        err_q, _, err_c = _checks(pol, c, q)
        print(f"graded {pol}: Q err {err_q}, c err {err_c}")
        assert np.all(err_q < 0.01) and np.all(err_c < 0.03)
        errs[pol] = err_q
    # mutation: no grading, the vacuum mesh jumps to 4h at the interface
    m = graded_disk_mesh([A, R_OUT], lambda r: h if r <= A else 4 * h)
    c, q, _ = _run("TE", mesh=m)
    err_q, _, err_c = _checks("TE", c, q)
    print(f"4x jump TE: Q err {err_q}, c err {err_c}")
    assert err_q[:2].min() > 3 * errs["TE"][:2].max()


def _reflection(absorber, electric_only=False):
    h = 60e-9
    x = np.array([6.0, 8.0, 10.0])
    f = x * C0 / (2 * np.pi * A)
    radii = [A, 1.75 * A] if absorber is None else [A, absorber["r0"], absorber["r1"]]
    s = UnstructuredFIT2D(graded_disk_mesh(radii, h), {}, polarization="TM",
                          bc="abc", absorber=absorber)  # fmt: skip
    if electric_only:
        s.edge_rate[:] = 0
    s.add_point_source([0, 0], GaussianSource(f.mean(), 0.3 * f.mean()))
    rc = np.array([1.2, 1.4, 1.6]) * A
    pts = np.vstack([circle_points(r, 16)[0] for r in rc])
    P = s._interp(pts)
    acc = np.zeros((f.size, len(pts)), complex)
    for i in range(int(250e-15 / s.dt)):
        s.one_step()
        acc += np.outer(np.exp(-2j * np.pi * f * (i + 1) * s.dt), P @ s.u)
    F = project_harmonics(acc.reshape(f.size, 3, 16), 2)
    al, ga, _ = fit_harmonics(F, 2 * np.pi * f / C0, rc, 2)
    return np.abs(ga[:, 2] / al[:, 2])


def test_absorber_reflection():
    d = 0.8 * A
    ab = dict(r0=1.75 * A, r1=1.75 * A + d, s_max=6 * C0 / d, order=2)
    r_abc = _reflection(None)
    r_abs = _reflection(ab)
    r_e = _reflection(ab, electric_only=True)
    print("ABC", r_abc, "matched layer", r_abs, "electric only", r_e)
    assert np.all(r_abs < r_abc / 3)
    assert np.all(r_e > 2 * r_abs)
