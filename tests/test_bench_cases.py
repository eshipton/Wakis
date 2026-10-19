"""
Benchmark-suite plumbing: case references, metrics and the circle
extraction, checked against exact synthetic fields (tests that can fail:
a wrong direction phase, a missing de-embedding or a shifted spectrum must be
detected).
"""

import numpy as np
import pytest
from scipy.constants import c as c_light
from scipy.special import hankel1, hankel2

from wakis.benchmarks import get_case, list_cases
from wakis.benchmarks import metrics as M
from wakis.benchmarks.extraction import cross_sections_from_circles
from wakis.cylinder_harmonics import circle_points


def test_registry():
    names = list_cases()
    for n in ("B1-TM", "B2-Au-TE", "B2-Ag-TE", "B3-g5", "B4", "B5-TE"):
        assert n in names
    with pytest.raises(KeyError):
        get_case("nope")


def test_b2_reference_has_clear_resonance():
    # The design choice of B2-Au: a clear LSPR (host n = 2) near 553 nm.
    case = get_case("B2-Au-TE")
    ref = case.reference
    p = M.peak(case.wavelengths, ref["q_ext"])
    assert p["interior"] and 540e-9 < p["lam_peak"] < 565e-9
    assert np.isfinite(p["fwhm"]) and p["fwhm"] < 200e-9
    assert np.all(ref["c_abs"] > 0)
    # TM has no plasmon: no interior maximum above the band edges
    tm = get_case("B2-Au-TM").reference["q_ext"]
    assert tm.max() < 1.05 * max(tm[0], tm[-1])


def test_b5_pec_limit():
    # A dielectric with m -> large imaginary approaches the PEC series
    from wakis.analytic.mie_cylinder import cylinder_efficiencies

    case = get_case("B5-TM")
    ref = case.reference
    x = case.k_background[10] * 1e-6
    q = cylinder_efficiencies(1e-3 - 3e4j, x, "TM")[0]
    assert np.isclose(q, ref["q_ext"][10], rtol=2e-3)


def _synthetic_samples(case, f_index, extra_incoming=0.0):
    """Exact scattered field (Wakis convention) on the case circles."""
    f = case.freqs[f_index]
    cl = case.cluster(f)
    pts = np.vstack([circle_points(r, case.n_points)[0] for r in case.circles])
    _, us = cl.field(pts)  # Wakis convention for Wakis-convention m
    if extra_incoming:
        # add an incoming (H2) wave as from an imperfect outer boundary
        k = 2 * np.pi * f * case.n_background / c_light
        r = np.hypot(*pts.T)
        th = np.arctan2(pts[:, 1], pts[:, 0])
        us = us + np.conj(extra_incoming * hankel2(1, k * r) * np.exp(1j * th))
    return us.reshape(len(case.circles), case.n_points)


@pytest.mark.parametrize("name", ["B3-g5", "B2-Au-TE"])
def test_extraction_reproduces_exact_cross_sections(name):
    case = get_case(name)
    idx = [0, len(case.freqs) // 2, len(case.freqs) - 1]
    S = np.array([_synthetic_samples(case, i) for i in idx])
    sp = cross_sections_from_circles(
        S, np.ones(len(idx)), case.freqs[idx], case.circles, case.n_background,
        case.direction, deembed=case.deembed,
    )  # fmt: skip
    ref = case.reference
    assert np.allclose(sp["c_ext"], ref["c_ext"][idx], rtol=1e-6)
    assert np.allclose(sp["c_sca"], ref["c_sca"][idx], rtol=1e-6)
    # mutation: the wrong propagation direction breaks C_ext of the dimer
    if name.startswith("B3"):
        bad = cross_sections_from_circles(
            S, np.ones(len(idx)), case.freqs[idx], case.circles,
            case.n_background, 0.0,
        )  # fmt: skip
        assert not np.allclose(bad["c_ext"], ref["c_ext"][idx], rtol=1e-2)


def test_deembedding_removes_boundary_reflections():
    # Centred cylinder re-illuminated by an incoming wave gamma_n (as from a
    # reflecting outer boundary): outgoing O_n = S_n I_n with S_n = 1 - 2 c_n.
    from wakis.analytic.mie_cylinder import cylinder_coefficients

    case = get_case("B2-Au-TE")
    i = len(case.freqs) // 2
    f = case.freqs[i]
    k = case.k_background[i]
    eps = case.materials["Au"].epsilon(f)
    N = 12
    c = cylinder_coefficients(np.sqrt(eps) / 2, k * 20e-9, "TE", n_terms=N)
    c = np.conj(np.concatenate([c[:0:-1], c]))  # exp(-i w t)
    n = np.arange(-N, N + 1)
    gamma = 0.05 * np.exp(1j * n) * (np.abs(n) <= 3)
    I = 1j**n / 2 + gamma
    O = (1 - 2 * c) * I
    pts = np.vstack([circle_points(r, case.n_points)[0] for r in case.circles])
    r, th = np.hypot(*pts.T), np.arctan2(pts[:, 1], pts[:, 0])
    E = np.exp(1j * np.outer(th, n))
    us = (hankel2(n[None], k * r[:, None]) * gamma + hankel1(n[None], k * r[:, None])
          * (O - 1j**n / 2)) * E  # fmt: skip
    S = np.conj(us.sum(1)).reshape(1, len(case.circles), case.n_points)
    args = (np.ones(1), case.freqs[[i]], case.circles, case.n_background, 0.0)
    good = cross_sections_from_circles(S, *args, deembed=True, n_max=N)
    bad = cross_sections_from_circles(S, *args, deembed=False, n_max=N)
    ref = case.reference["c_ext"][i]
    assert good["refl"][0] > 1e-2
    assert np.isclose(good["c_ext"][0], ref, rtol=1e-8)
    assert not np.isclose(bad["c_ext"][0], ref, rtol=1e-2)


def test_peak_metrics():
    lam = np.linspace(400, 800, 161)
    y = 1 / (1 + ((lam - 600.3) / 25) ** 2)
    p = M.peak(lam, y)
    assert abs(p["lam_peak"] - 600.3) < 0.05
    assert abs(p["fwhm"] - 50) < 0.3
    assert abs(p["height"] - 1) < 2e-3
    e = M.peak_errors(lam, 1 / (1 + ((lam - 606.3) / 25) ** 2), y)
    assert abs(e["lam_peak"] - 6 / 600.3) < 1e-3
    assert M.nrms(y, y) == 0 and M.nrms(1.02 * y, y) > 0.005
    # cost interpolation
    assert np.isclose(M.interp_to_target([1, 10, 100], [0.1, 0.01, 0.001], 0.0316),
                      10 ** 0.5, rtol=1e-2)  # fmt: skip
    assert np.isnan(M.interp_to_target([1, 10], [0.1, 0.05], 0.01))
    # a lucky coarse point does not count: the error must stay below target
    assert np.isclose(M.interp_to_target([1, 10, 100], [0.005, 0.1, 0.001], 0.01),
                      10 * 10 ** 0.5, rtol=1e-2)  # fmt: skip
    assert M.interp_to_target([1, 10], [0.001, 0.0005], 0.01) == -1.0


def test_cluster_mesh_has_no_gap_slivers():
    # With the outer rings fixed, h = 2.857 nm in the 5 nm gap left 0.05 nm
    # between the two rings and a 2.8 deg sliver (and a 6.5 % gap-field
    # error); the greedy priority ring keeps every angle above 15 deg.
    from wakis.benchmarks.meshing import cluster_mesh

    case = get_case("B3-g5")
    for h in (4e-9, 2.857e-9, 1.429e-9):
        ho = min(4 * h, 12.5e-9)  # as Triangles2DArm
        m = cluster_mesh(case.cylinders, h, ho, 110e-9, pml_layers=4, pml_dr=ho)
        P = m.nodes[m.cells]
        P = P[np.hypot(*P.mean(1).T) < 100e-9]
        e = [
            np.linalg.norm(P[:, i] - P[:, j], axis=1)
            for i, j in ((1, 2), (0, 2), (0, 1))
        ]
        ang = [np.arccos(np.clip((e[(k + 1) % 3] ** 2 + e[(k + 2) % 3] ** 2 - e[k] ** 2)
                                 / (2 * e[(k + 1) % 3] * e[(k + 2) % 3]), -1, 1))
               for k in range(3)]  # fmt: skip
        assert np.degrees(np.min(ang)) > 15, h
        # the gap is meshed: nodes inside the 5 nm gap near y = 0 (h <= 2 nm)
        gap = (np.abs(m.nodes[:, 0]) < 2.4e-9) & (np.abs(m.nodes[:, 1]) < h)
        assert gap.sum() >= 1 or h > 2e-9


def test_gap_stencil_reference():
    case = get_case("B3-g5")
    pts = case.gap_points
    assert pts.shape == (9, 2) and np.allclose(np.hypot(*pts[1:].T), 1.25e-9)
    ref = case.reference
    # the stencil mean and the centre differ (the field is not uniform)
    assert 0.001 < abs(ref["enh"].max() / ref["enh_centre"].max() - 1) < 0.2
