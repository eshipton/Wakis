"""
3D unstructured FIT on tetrahedra (E on edges, B on faces, Voronoi dual).

References: exact resonances of a spherical PEC cavity (zeros of j_n and
of (r j_n)'), of a dielectric sphere inside it (spherical Bessel matching),
of a cavity filled with a lossless Drude plasma (w^2 = w_c^2 + w_p^2), plus
the exact discrete identity D C = 0 (div B = 0 to round-off). The fast
tests use coarse meshes (h = a/4, ~2.5k tetrahedra, ~5 edges per vacuum
wavelength of the lowest mode); the convergence studies are marked slow and
were run on a 32-core VM (numbers in the physics guide).
"""

import numpy as np
import pytest
from scipy.constants import c

from wakis.dispersive import DispersiveMaterial, Drude
from wakis.resonances import harminv
from wakis.unstructured import analytic, ball_mesh
from wakis.unstructured.estimate import estimate_memory
from wakis.unstructured.solver3d import UnstructuredFIT3D


def gaussian_pulse(f0, bw):
    s = 1 / (2 * np.pi * bw)
    t0 = 5 * s
    return lambda t: np.exp(-(((t - t0) / s) ** 2) / 2) * np.sin(
        2 * np.pi * f0 * (t - t0)
    )


def k_of(f):
    return 2 * np.pi * np.asarray(f) / c


@pytest.fixture(scope="module")
def sphere():
    return ball_mesh([1.0], 0.25)


class TestEigenmodes:
    def test_pec_sphere(self, sphere):
        """First 10 modes (TM11 x3 at k a = 2.744, TM21 x5 at 3.870, TE11 at
        4.493) at h = a/4 (2.4k tetrahedra): TM11 within 0.5 %, all within
        2 % (TE11 has k h = 1.1, about 6 edges per wavelength)."""
        ref = analytic.sphere_in_cavity(1.0, 0.0, 1.0, "all", 10)
        s = UnstructuredFIT3D(sphere)
        f, e = s.eigenmodes(10, f_target=3.6 * c / (2 * np.pi))
        err = np.abs(k_of(f) / ref - 1)
        assert err.max() < 2e-2
        assert err[:3].max() < 5e-3
        assert np.allclose(e[s.fixed], 0)

    def test_whitney_fallback_agrees(self, sphere):
        """Consistent (non-diagonal) Whitney Hodges give the same modes to
        within the discretization error (they are the Nedelec FEM)."""
        ref = analytic.sphere_in_cavity(1.0, 0.0, 1.0, "all", 3)
        s = UnstructuredFIT3D(sphere, hodge="whitney")
        f, _ = s.eigenmodes(3, f_target=0.9 * ref[0] * c / (2 * np.pi))
        assert np.abs(k_of(f) / ref - 1).max() < 1.5e-2

    def test_dielectric_sphere(self):
        """eps = 4 sphere (radius a/2) in a PEC sphere: lowest TM (E crosses
        the interface) and TE triplets."""
        a, b, eps = 1.0, 0.5, 4.0
        ref_tm = analytic.sphere_in_cavity(a, b, eps, "TM", 3)
        ref_te = analytic.sphere_in_cavity(a, b, eps, "TE", 3)
        s = UnstructuredFIT3D(ball_mesh([b, a], 0.25), {0: [eps, 1.0]})
        f, _ = s.eigenmodes(6, f_target=2.63 * c / (2 * np.pi))
        k = np.sort(k_of(f))
        assert np.abs(k[:3] / ref_tm - 1).max() < 2.5e-2
        assert np.abs(k[3:] / ref_te - 1).max() < 2.5e-2

    @pytest.mark.slow
    @pytest.mark.parametrize("case", ["pec_te11", "dielectric_tm1"])
    def test_convergence(self, case):
        """Second-order convergence (VM, h = a/4 -> a/8, 2k -> 22k tets):
        TE11 of the PEC sphere (k a = 4.493) and the TM1 triplet of the
        eps = 4 sphere, whose E crosses the curved interface."""
        if case == "pec_te11":
            radii, mats = [1.0], {}
            ref = analytic.sphere_in_cavity(1.0, 0.0, 1.0, "TE", 1)[0]
        else:
            radii, mats = [0.5, 1.0], {0: [4.0, 1.0]}
            ref = analytic.sphere_in_cavity(1.0, 0.5, 4.0, "TM", 1)[0]
        errs = []
        hs = (0.25, 0.125)
        for h in hs:
            s = UnstructuredFIT3D(ball_mesh(radii, h), mats)
            f, _ = s.eigenmodes(3, f_target=ref * c / (2 * np.pi))
            errs.append(abs(np.mean(k_of(f)) / ref - 1))
        assert np.log(errs[0] / errs[1]) / np.log(hs[0] / hs[1]) > 1.7, errs


class TestTimeDomain:
    def test_div_b_energy_and_resonances(self, sphere):
        """PEC sphere excited by a Whitney point dipole: D b = 0 to
        round-off, bounded energy over 10^4 steps after the source, and
        harmonic inversion of a probe reproduces the eigen-solver (after the
        exact leapfrog correction sin(w_td dt/2) = w dt/2)."""
        s = UnstructuredFIT3D(sphere)
        f_eig, _ = s.eigenmodes(3, f_target=0.9 * 2.74 * c / (2 * np.pi))
        fc = f_eig.mean()
        s.add_point_dipole([0.2, 0.1, 0.3], [1, 1, 1], gaussian_pulse(fc, 0.5 * fc))
        probe = s.add_probe([[0.3, -0.2, 0.1]], 0)
        s.run(2500)
        e0 = s.energy()
        s.run(10000)
        assert np.abs(s.divergence_b()).max() < 1e-12 * np.abs(s.b).max()
        assert abs(s.energy() / e0 - 1) < 0.05
        sig = s.probe_signal(probe)[:, 0]
        res = harminv(sig[2000:], s.dt, 0.9 * f_eig[0], 1.1 * f_eig[-1], t0=2001 * s.dt)
        f_td = 2 * np.arcsin(np.pi * f_eig * s.dt) / (2 * np.pi * s.dt)
        # the strongest resonance matches one of the TM_11 triplet
        strongest = res.frequency[np.argmax(np.abs(res.amplitude))]
        assert np.min(np.abs(strongest / f_td - 1)) < 1e-4

    def test_absorbing_boundary_drains_energy(self, sphere):
        """Silver-Mueller boundary: after the pulse leaves, the energy left
        in the ball is < 1 % of the peak (PEC keeps all of it)."""
        s = UnstructuredFIT3D(sphere, bc="abc")
        f0 = 2.0 * c / (2 * np.pi)
        s.add_point_dipole([0.0, 0.0, 0.0], [0, 0, 1], gaussian_pulse(f0, 0.5 * f0))
        energies = []
        for _ in range(40):
            s.run(100)
            energies.append(s.energy())
        assert energies[-1] < 1e-2 * max(energies)

    def test_drude_filled_cavity(self, sphere):
        """Cavity filled with a lossless Drude plasma (k_p = 3/a): every mode
        moves to w^2 = w_c^2 + w_p^2; the edge ADE reproduces the shift of
        the TM_11 triplet to within the time discretization."""
        kp = 3.0
        f_c, _ = UnstructuredFIT3D(sphere).eigenmodes(3, f_target=2.5 * c / (2 * np.pi))
        plasma = DispersiveMaterial(1.0, [Drude(kp * c, 0.0)])
        s = UnstructuredFIT3D(sphere, {0: plasma})
        f_exp = np.sqrt(f_c**2 + (kp * c / (2 * np.pi)) ** 2)
        s.add_point_dipole(
            [0.2, 0.1, 0.3], [1, 1, 1], gaussian_pulse(f_exp.mean(), 0.3 * f_exp.mean())
        )
        probe = s.add_probe([[0.3, -0.2, 0.1]], 0)
        s.run(12000)
        sig = s.probe_signal(probe)[:, 0]
        res = harminv(
            sig[3000:], s.dt, 0.97 * f_exp.min(), 1.03 * f_exp.max(), t0=3001 * s.dt
        )
        assert res.frequency.size > 0
        # leapfrog + ADE time discretization error O((w dt)^2) ~ 1e-3 here
        assert np.min(np.abs(res.frequency[:, None] / f_exp[None] - 1)) < 2e-3


def test_estimate_memory():
    """Calibrated model (VM measurements): 2.1 kB per tetrahedron in time
    domain, super-linear sparse-LU fill-in for eigen-solves."""
    assert estimate_memory(1e5) < 1e9
    assert estimate_memory(1e5, eigen=True) > 2 * estimate_memory(1e5)
    assert estimate_memory(ball_mesh([1.0], 0.5)) < 0.3e9
