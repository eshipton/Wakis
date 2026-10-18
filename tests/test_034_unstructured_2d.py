"""
2D unstructured FIT (triangles, Voronoi dual): TM (Ez) and TE (Hz).

References are exact: Bessel matching for a dielectric rod and a lossless
Drude rod in a circular PEC cavity, and the 2D Green's function of a line
current for the absorbing boundary. Meshes from `disk_mesh` conform to the
interfaces (node rings), so the error is 2nd order in h in both
polarizations; the tolerances below are a factor ~2 above the observed
errors at the stated mesh sizes.
"""

import numpy as np
import pytest
from scipy.constants import c
from scipy.spatial import Delaunay

from wakis.dispersive import DispersiveMaterial, Drude
from wakis.resonances import harminv
from wakis.unstructured import SimplexMesh, analytic, disk_mesh
from wakis.unstructured.solver2d import UnstructuredFIT2D

A_CAV, B_ROD, EPS_ROD = 1.0, 0.5, 6.0


def gaussian_pulse(f0, bw):
    s = 1 / (2 * np.pi * bw)
    t0 = 5 * s
    return lambda t: np.exp(-(((t - t0) / s) ** 2) / 2) * np.sin(
        2 * np.pi * f0 * (t - t0)
    )


def k_of(f):
    return 2 * np.pi * np.asarray(f) / c


class TestEigenmodes:
    @pytest.mark.parametrize("pol", ["TM", "TE"])
    def test_rod_second_order(self, pol):
        """Lowest mode of a rod (eps 6, radius a/2) in a PEC cavity converges
        at 2nd order in both polarizations (TE: E crosses the interface)."""
        ref = analytic.rod_in_cavity(A_CAV, B_ROD, EPS_ROD, pol, 2)[0]
        errs = []
        for h in (0.1, 0.05, 0.025):
            s = UnstructuredFIT2D(
                disk_mesh([B_ROD, A_CAV], h), {0: [EPS_ROD, 1.0]}, polarization=pol
            )
            errs.append(abs(k_of(s.eigenmodes(1)[0][0]) / ref - 1))
        order = np.log2(errs[1] / errs[2])
        # h = 0.025: 11.5k triangles, ~20 nodes per wavelength in the rod
        assert errs[2] < 4e-5
        assert order > 1.5, errs

    def test_barycentric_fallback(self):
        """On a non-Delaunay mesh (random node jitter + diagonal flips) the
        Voronoi node areas can be negative; the barycentric dual keeps a
        positive mass and the cotangent (Whitney) stiffness, and converges."""
        ref = analytic.rod_in_cavity(A_CAV, 0.0001, 1.0, "TM", 1)[0]
        rng = np.random.default_rng(3)
        errs = []
        for h in (0.1, 0.05):
            m = disk_mesh([A_CAV], h)
            nodes = m.nodes.copy()
            inner = np.hypot(*nodes.T) < A_CAV - 0.6 * h
            nodes[inner] += 0.25 * h * rng.standard_normal((inner.sum(), 2))
            bad = SimplexMesh(nodes, _flip_some(nodes, Delaunay(nodes).simplices, rng))
            assert bad.diagnostics()["negative_dual_length"] > 0
            s = UnstructuredFIT2D(bad, polarization="TM", hodge="barycentric")
            errs.append(abs(k_of(s.eigenmodes(1)[0][0]) / ref - 1))
        assert errs[1] < 2e-3
        assert errs[1] < errs[0] / 2.5


def _flip_some(nodes, tri, rng, n_flips=20):
    """Flip interior edges of random triangle pairs (convex quads only)."""
    tri = tri.copy()
    edge_tris = {}
    for t, cell in enumerate(tri):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edge_tris.setdefault(tuple(sorted((cell[a], cell[b]))), []).append(t)
    pairs = [(e, ts) for e, ts in edge_tris.items() if len(ts) == 2]
    used = set()
    flips = 0
    for i in rng.permutation(len(pairs)):
        (p, q), (t1, t2) = pairs[i]
        if t1 in used or t2 in used:
            continue
        r = [v for v in tri[t1] if v not in (p, q)][0]
        s = [v for v in tri[t2] if v not in (p, q)][0]
        # the quad p r q s must be convex for the flip to be valid
        P = nodes

        def side(a, b, x):
            u, v = P[b] - P[a], P[x] - P[a]
            return np.sign(u[0] * v[1] - u[1] * v[0])

        if side(r, s, p) * side(r, s, q) >= 0:
            continue
        tri[t1], tri[t2] = [r, s, p], [r, s, q]
        used |= {t1, t2}
        flips += 1
        if flips == n_flips:
            break
    return tri


class TestTimeDomain:
    @pytest.mark.parametrize("pol", ["TM", "TE"])
    def test_resonances_and_stability(self, pol):
        """Leapfrog resonances (harmonic inversion of a probe) match the
        eigen-solver after the exact leapfrog correction
        sin(w_td dt / 2) = w dt / 2, and the energy stays bounded over
        50k steps (no late-time growth)."""
        m = disk_mesh([B_ROD, A_CAV], 0.1)
        s = UnstructuredFIT2D(m, {0: [EPS_ROD, 1.0]}, polarization=pol)
        f_eig = s.eigenmodes(4)[0]
        f_eig_td = 2 * np.arcsin(np.pi * f_eig * s.dt) / (2 * np.pi * s.dt)
        fc = f_eig[:3].mean()
        s.add_point_source([0.31, 0.17], gaussian_pulse(fc, fc))
        probe = s.add_probe([[0.23, -0.41]])
        s.run(4000)
        e0 = s.energy()
        sig = s.probe_signal(probe)[:, 0]
        res = harminv(sig[2000:], s.dt, 0.5 * f_eig[0], 1.02 * f_eig[2], t0=2001 * s.dt)
        for f in f_eig[:3]:
            f_td = 2 * np.arcsin(np.pi * f * s.dt) / (2 * np.pi * s.dt)
            assert np.min(np.abs(res.frequency / f_td - 1)) < 2e-5
        del f_eig_td
        energies = []
        for _ in range(10):
            s.run(5000)
            energies.append(s.energy())
        # staggered energy oscillates by O((w dt)^2) around a constant
        assert np.max(np.abs(np.array(energies) / e0 - 1)) < 0.1


class TestAbsorbingBoundary:
    def test_line_source_green_function(self):
        """TM line current in a disk with the first-order Silver-Mueller
        boundary: |Ez| at 4 radii vs the exact H0^(2) Green's function
        (16 nodes per wavelength, boundary at 3 wavelengths). PEC gives
        O(1) standing-wave errors, the ABC a few percent."""
        R, f0 = 1.0, 3 * c
        s_t = 1 / (np.pi * f0)
        t0 = 6 * s_t

        def amp(t):
            return np.exp(-(((t - t0) / s_t) ** 2) / 2)

        m = disk_mesh([R], 0.02)
        rs = np.array([0.2, 0.35, 0.5, 0.65])
        errs = {}
        for bc in ("abc", "pec"):
            s = UnstructuredFIT2D(m, polarization="TM", bc=bc)
            s.add_point_source([0.0, 0.0], amp)
            p = s.add_probe(np.stack([rs, 0 * rs], 1))
            n = int(12 * R / c / s.dt)
            s.run(n)
            tt = s.probe_times()
            ts = tt - 0.5 * s.dt
            ez = np.exp(-2j * np.pi * f0 * tt) @ s.probe_signal(p) * s.dt
            cur = np.exp(-2j * np.pi * f0 * ts) @ amp(ts) * s.dt
            exact = analytic.line_source_field(rs, k_of(f0), cur)
            errs[bc] = np.abs(np.abs(ez / exact) - 1)
        assert errs["abc"].max() < 0.08
        assert errs["pec"].max() > 0.5


class TestDrude:
    def test_surface_plasmon_rod(self):
        """Lossless Drude rod (k_p = 4/a) in a PEC cavity, TE: the n = 1
        surface-plasmon mode (eps(w) ~ -0.2 at resonance, fields peaked at
        the metal surface) from the dispersive eigen-solver converges at 2nd
        order to the exact Bessel/modified-Bessel matching, and the
        time-domain ADE run reproduces it."""
        kp = 4.0
        drude = DispersiveMaterial(1.0, [Drude(kp * c, 0.0)])
        k_sp = analytic.drude_rod_in_cavity_te(A_CAV, B_ROD, kp, n=1)[1]
        errs = []
        for h in (0.1, 0.05):
            s = UnstructuredFIT2D(
                disk_mesh([B_ROD, A_CAV], h), {0: drude}, polarization="TE"
            )
            f = s.dispersive_eigenmode(k_sp * c / (2 * np.pi))
            errs.append(abs(k_of(f) / k_sp - 1))
        assert errs[1] < 1e-3 and errs[0] / errs[1] > 3
        # Time domain on the h = 0.05 mesh with an in-plane dipole
        f_sp = k_sp * c / (2 * np.pi)
        s.add_point_source([0.6, 0.1], gaussian_pulse(f_sp, 0.3 * f_sp), [1, 0.3])
        probe = s.add_probe([[0.55, -0.2]])
        s.run(20000)
        sig = s.probe_signal(probe)[:, 0]
        res = harminv(sig[4000:], s.dt, 0.97 * f_sp, 1.02 * f_sp, t0=4001 * s.dt)
        assert np.min(np.abs(k_of(res.frequency) / k_sp - 1)) < 1e-3
        assert np.abs(sig[-2000:]).max() < 2 * np.abs(sig[4000:6000]).max()
