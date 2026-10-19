"""
Exact 2D multi-cylinder T-matrix (wakis.analytic.multi_cylinder): Graf's
theorem, reduction to the single-cylinder series, optical theorem / energy
balance, reciprocity, mirror symmetry, convergence in the multipole order,
and mutation checks (a wrong translation phase must fail these tests).
"""

import numpy as np
import pytest
from scipy.special import hankel1, jv

from wakis.analytic.mie_cylinder import cylinder_coefficients, cylinder_efficiencies
from wakis.analytic.multi_cylinder import CylinderCluster

TRIO = dict(centers=[[-1.2, 0.0], [1.2, 0.1], [0.0, 2.0]], radii=[1.0, 0.8, 0.5], k=1.7)


def test_graf_addition_theorem():
    # H_m(k|r - r_l|) e^{i m theta_l} = sum_n H_{m-n}(k d) e^{i(m-n) phi} J_n(k rho) e^{i n theta}
    k, rl, rj = 1.3, np.array([0.3, -0.2]), np.array([2.0, 1.1])
    p = rj + np.array([0.2, -0.35])
    R = rj - rl
    for m in (-2, 0, 3):
        q = p - rl
        lhs = hankel1(m, k * np.hypot(*q)) * np.exp(1j * m * np.arctan2(q[1], q[0]))
        n = np.arange(-40, 41)
        s = p - rj
        rhs = np.sum(
            hankel1(m - n, k * np.hypot(*R)) * np.exp(1j * (m - n) * np.arctan2(R[1], R[0]))
            * jv(n, k * np.hypot(*s)) * np.exp(1j * n * np.arctan2(s[1], s[0]))
        )  # fmt: skip
        assert abs(lhs - rhs) < 1e-12 * abs(lhs)


@pytest.mark.parametrize("pol", ["TM", "TE"])
@pytest.mark.parametrize("m", [1.5 + 0.1j, 0.3 - 2.5j])
def test_single_offset_cylinder_equals_series(pol, m):
    x = 2.3
    cl = CylinderCluster([[0.7, -0.4]], 1.0, m, x, pol, direction=0.3)
    ce, cs, ca = np.array(cl.cross_sections()) / 2
    ref = cylinder_efficiencies(m, x, pol)
    assert np.allclose([ce, cs, ca], ref, rtol=1e-10)
    assert np.isclose(cl.absorption_per_cylinder()[0] / 2, ref[2], rtol=1e-10)


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_optical_theorem_and_energy_balance(pol):
    # Lossless cylinders: C_ext = C_sca (the optical theorem ties the forward
    # amplitude to the integrated pattern). Lossy: sum of the per-cylinder
    # absorption (local power balance) = C_ext - C_sca.
    cl = CylinderCluster(m=2.0, polarization=pol, direction=0.9, **TRIO)
    ce, cs, ca = cl.cross_sections()
    assert abs(ca) < 1e-10 * ce
    cl = CylinderCluster(m=0.3 - 2.5j, polarization=pol, direction=0.9, **TRIO)
    ce, cs, ca = cl.cross_sections()
    assert ca > 0.05 * ce
    assert np.isclose(cl.absorption_per_cylinder().sum(), ca, rtol=1e-9)


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_reciprocity(pol):
    # f(theta_out; phi_in) = f(phi_in + pi; theta_out + pi)
    a = CylinderCluster(m=0.3 - 2.5j, polarization=pol, direction=0.4, **TRIO)
    b = CylinderCluster(m=0.3 - 2.5j, polarization=pol, direction=2.1 + np.pi, **TRIO)
    fa = a.far_amplitude(2.1)[0]
    fb = b.far_amplitude(0.4 + np.pi)[0]
    assert abs(fa - fb) < 1e-10 * abs(fa)


def test_mutation_wrong_translation_phase_breaks_energy_balance(monkeypatch):
    monkeypatch.setattr(CylinderCluster, "_graf_sign", -1)
    cl = CylinderCluster(m=2.0, polarization="TE", direction=0.9, **TRIO)
    ce, cs, ca = cl.cross_sections()
    assert abs(ca) > 1e-3 * ce
    # (reciprocity alone does NOT catch this mutation: the mirrored
    # translation operator is still symmetric, so the energy test is needed)
    cl = CylinderCluster(m=0.3 - 2.5j, polarization="TE", direction=0.9, **TRIO)
    assert not np.isclose(
        cl.absorption_per_cylinder().sum(), cl.cross_sections()[2], rtol=1e-3
    )


def test_large_separation_reduces_to_single_cylinders():
    # Coupling decays like |H_n(k d)| ~ sqrt(2 / (pi k d)): at k d = 2e4 the
    # coefficients of each cylinder equal the isolated ones (with the
    # plane-wave phase) to ~1e-2 relative; at k d = 20 they do not.
    m, k, a = 0.3 - 2.5j, 1.0, 1.0
    c1 = cylinder_coefficients(m, k * a, "TE", n_terms=8)
    c1 = np.concatenate([c1[:0:-1], c1])
    n = np.arange(-8, 9)
    for d, ok in ((2e4, True), (20.0, False)):
        cl = CylinderCluster([[-d / 2, 0], [d / 2, 0]], a, m, k, "TE", n_max=8,
                             direction=np.pi / 2)  # fmt: skip
        iso = -c1 * np.conj((1j**n) * np.exp(-1j * n * np.pi / 2))  # Wakis conv.
        err = np.abs(cl.b[0] - iso).max() / np.abs(iso).max()
        assert (err < 2e-2) == ok, (d, err)


def test_dimer_mirror_symmetry_and_convergence():
    # Au-like dimer (eps = -20 - 2j) in a host, gap 2 nm, a = 20 nm: the
    # gap enhancement converges in the multipole order (bipolar ratio
    # q ~ 0.53) and the pattern is symmetric under x -> -x.
    lam, nb, a, g = 700e-9, 2.0, 20e-9, 2e-9
    m = np.sqrt(-20 - 2j + 0j) / nb
    k = 2 * np.pi * nb / lam
    geo = dict(centers=[[-(a + g / 2), 0], [a + g / 2, 0]], radii=a, m=m, k=k,
               polarization="TE", direction=np.pi / 2)  # fmt: skip
    enh = []
    for N in (30, 45, 60):
        cl = CylinderCluster(n_max=N, **geo)
        enh.append(np.sum(np.abs(cl.in_plane_e([[0.0, 0.0]])[0]) ** 2))
    assert abs(enh[2] / enh[1] - 1) < 1e-5
    assert abs(enh[2] / enh[0] - 1) > 1e-5  # the test resolves the truncation
    th = np.linspace(0.1, 1.4, 7)
    f = cl.far_amplitude(th)
    assert np.allclose(f, cl.far_amplitude(np.pi - th), rtol=1e-10)
    # E at the gap centre is along the dimer axis
    E = cl.in_plane_e([[0.0, 0.0]])[0]
    assert abs(E[1]) < 1e-8 * abs(E[0])
