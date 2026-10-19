"""
Radial CFS-PML of `UnstructuredFIT2D` and the tabulated plane-wave source.

* Reflection: an off-centre line source (TM and TE, vacuum) radiates into a
  10-cell PML (h = 40 nm, layer 0.4 um = lambda_max / 8 at x = 2); the
  incoming/outgoing cylindrical-harmonic ratio on three circles stays below
  4e-3 over x = 2..10 (recorded 0.7-2.6e-3; the bare PEC wall gives ~1).
* Mutations must fail: negative sigma (unstable or reflection > 0.1), and
  the spoke/chord anisotropy swapped (reflection > 1e-2).
* Stability: 4000 fs (10x the usual window), no late-time growth.
* Mesh: a non-polar-aligned layer is rejected.
* Dispersive (Drude) rod inside the PML domain: stable, fields decay.
* LTS (p = 2) + PML: reflection as with the global step, stable; fine
  nodes inside the layer are refused.
* Plane-wave table: the separable-delay table reproduces the direct
  evaluation of s(t - tau) to < 5e-4 (recorded 1.3e-4 TM, 8e-5 TE; a one-entry
  delay offset would give ~6e-3).
"""

import numpy as np
import pytest
from scipy.constants import c as C0

from wakis.cylinder_harmonics import circle_points, fit_harmonics, project_harmonics
from wakis.sources import GaussianSource
from wakis.unstructured import UnstructuredFIT2D, graded_disk_mesh

A = 1e-6
R0 = 1.65e-6
X = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
F = X * C0 / (2 * np.pi * A)


def _solver(pol, h=40e-9, nl=10, mutate=None, materials=None, **cfg):
    m = graded_disk_mesh([A, R0], h, pml_layers=nl, pml_dr=h)
    s = UnstructuredFIT2D(m, materials or {}, polarization=pol, bc="pec",
                          pml=dict(r0=R0, r1=R0 + nl * h, **cfg))  # fmt: skip
    p = s._pml
    if mutate == "swap":  # spokes get s_r / s_t, chords s_t / s_r
        for f in (p.fs, p.fc):
            f.kn, f.kd, f.sn, f.sd = f.kd, f.kn, f.sd, f.sn
    s.add_point_source(
        [0.4e-6, 0.1e-6], GaussianSource(F.mean(), 0.25 * (F[-1] - F[0]))
    )
    return s


def _reflection(s, t_end=400e-15):
    rc = np.array([1.2, 1.35, 1.5]) * 1e-6
    pts = np.vstack([circle_points(r, 64)[0] for r in rc])
    P = s._interp(pts)
    acc = np.zeros((F.size, len(pts)), complex)
    for i in range(int(t_end / s.dt)):
        s.one_step()
        acc += np.outer(np.exp(-2j * np.pi * F * (i + 1) * s.dt), P @ s.u)
        if not np.all(np.isfinite(s.u)):
            return np.full(F.size, np.inf)
    Fh = project_harmonics(acc.reshape(F.size, 3, 64), 12)
    al, ga, _ = fit_harmonics(Fh, 2 * np.pi * F / C0, rc, 12)
    return np.sqrt((np.abs(ga) ** 2).sum(1) / (np.abs(al) ** 2).sum(1))


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_pml_reflection(pol):
    r = _reflection(_solver(pol))
    print(pol, "PML reflection", r)
    assert np.all(r < 4e-3)


@pytest.mark.parametrize("mutate", ["negative_sigma", "swap"])
def test_pml_mutations_fail(mutate):
    if mutate == "negative_sigma":
        s = _solver("TM", sigma_max=-2e15)
    else:
        s = _solver("TM", mutate="swap")
    r = _reflection(s)
    print(mutate, r)
    assert not np.all(r < (0.1 if mutate == "negative_sigma" else 1e-2))


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_pml_long_run_stable(pol):
    s = _solver(pol, h=60e-9, kappa_max=3.0, alpha=2 * np.pi * 20e12)
    n = int(4000e-15 / s.dt)
    peaks = []
    for i in range(n):
        s.one_step()
        if i % 100 == 0:
            peaks.append(np.abs(s.u).max())
    peaks = np.array(peaks)
    q = len(peaks) // 4
    print(
        pol,
        "peak",
        peaks.max(),
        "quarters",
        [peaks[k * q : (k + 1) * q].max() for k in range(4)],
    )
    assert np.all(np.isfinite(peaks))
    assert peaks[3 * q :].max() < peaks[q : 2 * q].max()
    assert peaks[-1] < 1e-3 * peaks.max()


def test_pml_rejects_unaligned_mesh():
    m = graded_disk_mesh([A, R0 + 0.4e-6], 40e-9)  # staggered rings, no polar layer
    with pytest.raises(ValueError, match="polar-aligned"):
        UnstructuredFIT2D(m, {}, polarization="TM", bc="pec",
                          pml=dict(r0=R0, r1=R0 + 0.4e-6))  # fmt: skip


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_plane_wave_table_matches_direct(pol):
    m = graded_disk_mesh([A, 2.0e-6], 50e-9)
    src = GaussianSource(F.mean(), 0.25 * (F[-1] - F[0]))
    out = []
    for direct in (True, False):
        s = UnstructuredFIT2D(m, {0: [4.0, 1.0]}, polarization=pol, bc="abc")
        s.add_plane_wave(src)
        if direct:
            pw = s.plane_wave

            def vals(n, s=s, pw=pw):
                e = s._incident_signal(pw["pos"], n * s.dt)
                if s.pol == "TE":
                    return np.bincount(
                        pw["e"], pw["coef"] * e, minlength=s.mesh.n_edges
                    )
                o = np.zeros(s.free.size)
                o[pw["idx"]] = pw["w"] * e / s.dt
                return o

            s._pw_values = vals
        s.run(int(120e-15 / s.dt))
        out.append(s.u.copy())
    d = np.abs(out[0] - out[1]).max() / np.abs(out[0]).max()
    print(pol, "table vs direct", d)
    assert d < 5e-4


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_pml_with_dispersive_scatterer(pol):
    """A lossy Drude rod inside the PML-terminated domain (dispersive ADEs
    outside the layer): stable over 2000 fs and the fields decay."""
    from wakis.dispersive import DispersiveMaterial, Drude

    drude = DispersiveMaterial(1.0, [Drude(omega_p=3e15, gamma=1e14)])
    s = _solver(pol, h=60e-9, materials={0: drude})
    peaks = []
    for i in range(int(2000e-15 / s.dt)):
        s.one_step()
        if i % 100 == 0:
            peaks.append(np.abs(s.u).max())
    peaks = np.array(peaks)
    print(pol, "Drude rod + PML: peak", peaks.max(), "end", peaks[-1])
    assert np.all(np.isfinite(peaks))
    assert peaks[-1] < 1e-2 * peaks.max()


def _graded_pml_solver(pol, lts=None, h=40e-9, nl=10):
    """Rod-graded mesh (h on and around an eps = 4 rod, 2h further out), so the
    LTS fine region is the vacuum band around the rod, away from the PML."""

    def size(r):
        d = r - 0.5e-6 - 2 * h
        return h if d <= 0 else min(2 * h, h * 1.2 ** (d / h))

    m = graded_disk_mesh([0.5e-6, R0], size, pml_layers=nl, pml_dr=2 * h)
    s = UnstructuredFIT2D(m, {0: [4.0, 1.0]}, polarization=pol, bc="pec", lts=lts,
                          pml=dict(r0=R0, r1=R0 + nl * 2 * h))  # fmt: skip
    s.add_point_source(
        [0.9e-6, 0.1e-6], GaussianSource(F.mean(), 0.25 * (F[-1] - F[0]))
    )
    return s


@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_pml_with_lts(pol):
    """
    LTS (p = 2) + PML: the fine region is the band around the rod (outside
    the layer), the outgoing-wave reflection matches the global-step run
    (within 2x of it; recorded LTS 0.2-0.8e-2 vs global 0.2-1.2e-2 on this
    coarse 80 nm vacuum mesh), and the run is stable and decays over
    2000 fs. Mutation: LTS fine nodes inside the layer are refused.
    """
    s_g = _graded_pml_solver(pol)
    s_l = _graded_pml_solver(pol, lts=dict(p=2))
    assert s_l.lts.F.size > 0
    assert s_l.dt > 1.5 * s_g.dt
    r_g, r_l = _reflection(s_g), _reflection(s_l)
    print(pol, "global", r_g, "LTS", r_l, "dt ratio", s_l.dt / s_g.dt)
    assert np.all(r_g < 2e-2)  # coarse vacuum (2h = 80 nm): discretization floor
    assert np.all(r_l < 2 * r_g + 1e-4)
    peaks = []
    for i in range(int(1600e-15 / s_l.dt)):
        s_l.one_step()
        if i % 50 == 0:
            peaks.append(np.abs(s_l.u).max())
    assert np.all(np.isfinite(peaks)) and peaks[-1] < 1e-2 * max(peaks)
    # mutation: force fine nodes into the layer
    s = _graded_pml_solver(pol)
    s.lts, s._G_F = s_l.lts, s_l._G_F
    s.lts.F = np.r_[s.lts.F, s._pml.nodes[:3]]
    with pytest.raises(ValueError, match="PML"):
        s._check_lts_pml()
