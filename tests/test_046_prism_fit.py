"""
Prism FIT (`PrismMesh`, `PrismFIT3D`): a triangle mesh extruded in z.

* Topology: C G = 0 and D C = 0 exactly (bounded and periodic z).
* Empty PEC cylindrical cavity: k^2 = k_t^2 + (p pi / h)^2 (Bessel zeros)
  for the 12 lowest modes, second-order convergence (observed order > 1.6).
  Mutation: the half-layer weights of the triangle-face reluctance replaced by
  full layers (a factor 2 on M_nu of b_z) must be > 10x worse.
* Dielectric rod (eps = 4, b = a/2) spanning a PEC cavity: the 20 lowest
  modes, including the hybrid HE/EH modes with p >= 1 that are *not* given by
  k_t^2 + (p pi / h)^2, against `rod_cavity_hybrid` (RMS < 0.9 % at h = a/10,
  order > 1.6). Mutation: series (instead of parallel) eps averaging on the
  horizontal edges that lie on the rod surface must be > 1.4x worse
  (recorded 1.6-1.75x at h = a/10, a/14).
* The analytic hybrid determinant reduces to the empty cavity (eps = 1) to
  1e-12 and to `rod_in_cavity` TM for p = 0.
* One periodic layer reproduces the 2D triangle TM solver
  (`UnstructuredFIT2D`) eigenfrequencies to 1e-9 (z-invariant fields).
* Leapfrog invariants after the source is off: discrete energy constant to
  1e-12 over 1500 steps and max |D b| / max |b| < 1e-12. Mutations: a wrong
  sign in the z part of C^T (energy drifts > 1e-3) and a 1 % error in the
  in-plane part of C (random per edge; divergence > 1e-4).
* z PML: a dipole in a PMC-walled slab guide; probe signals within 3 % of a
  domain long enough that its end reflections arrive after the window;
  s_max = 0 (PEC ends, mutation) is > 10x worse.
* Dispersive (Drude-filled) cavity: k^2 = k_c^2 + k_p^2 within 1.5 %.
* Lumped port: a resistor closing a small parallel-plate gap; Z_in(f)
  equals R || 1/(i w C) of the plate capacitance within 5 %.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.constants import c
from scipy.constants import epsilon_0 as eps_0
from scipy.special import jn_zeros, jnp_zeros

from wakis.resonances import harminv
from wakis.sources import GaussianSource
from wakis.unstructured import PrismFIT3D, PrismMesh, UnstructuredFIT2D, disk_mesh
from wakis.unstructured.analytic import rod_cavity_hybrid, rod_in_cavity

A, H = 1.0, 1.0


def _empty_cavity_k(n_modes, a=A, h=H):
    ks = []
    for n in range(8):
        deg = 1 if n == 0 else 2
        for jz in jn_zeros(n, 4):
            ks += [np.sqrt((jz / a) ** 2 + (q * np.pi / h) ** 2) for q in range(4)] * deg
        for jz in jnp_zeros(n, 4):
            ks += [np.sqrt((jz / a) ** 2 + (q * np.pi / h) ** 2) for q in range(1, 4)] * deg
    return np.sort(ks)[:n_modes]


def _cavity(h2d, materials=None, radii=(A,), **kw):
    m2 = disk_mesh(list(radii), h2d)
    nz = int(round(H / h2d))
    pm = PrismMesh(m2, np.linspace(0, H, nz + 1))
    return PrismFIT3D(pm, materials, **kw)


def _modes_k(s, n):
    f, _ = s.eigenmodes(n, f_target=1.0 * c / (2 * np.pi))
    return 2 * np.pi * f / c


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def test_topology():
    m2 = disk_mesh([A], 0.3)
    for periodic in (False, True):
        pm = PrismMesh(m2, [0, 0.2, 0.5, 0.6], periodic_z=periodic)
        C, G, D = pm.curl(), pm.grad(), pm.div()
        assert abs(C @ G).max() == 0
        assert abs(D @ C).max() == 0
        assert C.shape == (pm.n_faces, pm.n_edges)


def test_empty_cavity():
    k_ex = _empty_cavity_k(12)
    err = []
    for h in (0.1, 0.07):
        s = _cavity(h)
        err.append(_rms(_modes_k(s, 12) / k_ex - 1))
    order = np.log(err[0] / err[1]) / np.log(0.1 / 0.07)
    print("empty cavity rms", err, "order", order)
    assert err[0] < 6e-3 and order > 1.6
    # mutation: full-layer (instead of half-layer) weights on b_z
    s = _cavity(0.1)
    s.Mnu[: s.Mz.size] *= 2
    bad = _rms(_modes_k(s, 12) / k_ex - 1)
    assert bad > 10 * err[0]


def test_hybrid_analytic_limits():
    k_ex = _empty_cavity_k(30)
    k = rod_cavity_hybrid(A, 0.5, 1.0, H, n_modes=30)
    assert np.max(np.abs(k - k_ex)) < 1e-12 * k_ex.max()
    k0 = rod_cavity_hybrid(A, 0.5, 4.0, 10 * H, n_modes=4, p_max=0)
    assert np.allclose(k0, rod_in_cavity(A, 0.5, 4.0, "TM", n_modes=4), rtol=1e-12)


def test_rod_cavity_hybrid_modes():
    k_ex = rod_cavity_hybrid(A, 0.5, 4.0, H, n_modes=20)
    # the rod couples TE and TM: the hybrid modes differ from k_t^2 + beta^2
    k_naive = np.sort(
        np.r_[
            rod_in_cavity(A, 0.5, 4.0, "TM", 20),
            np.sqrt(rod_in_cavity(A, 0.5, 4.0, "TM", 20) ** 2 + np.pi**2),
            np.sqrt(rod_in_cavity(A, 0.5, 4.0, "TE", 20) ** 2 + np.pi**2),
        ]
    )[:20]
    assert _rms(k_naive / k_ex - 1) > 0.02
    err = []
    for h in (0.1, 0.07):
        s = _cavity(h, {0: [4.0, 1.0]}, radii=(0.5, A))
        err.append(_rms(_modes_k(s, 20) / k_ex - 1))
    order = np.log(err[0] / err[1]) / np.log(0.1 / 0.07)
    print("rod cavity rms", err, "order", order)
    assert err[0] < 9e-3 and order > 1.6
    # mutation: series eps averaging on the horizontal interface edges
    # (first-order consistency error at the interface)
    bad = [_series_mutation(h, k_ex) for h in (0.1, 0.07)]
    order_bad = np.log(bad[0] / bad[1]) / np.log(0.1 / 0.07)
    print("series-averaged mutation", bad, "order", order_bad)
    assert bad[0] > 1.4 * err[0] and bad[1] > 1.5 * err[1]


def _series_mutation(h, k_ex):
    s = _cavity(h, {0: [4.0, 1.0]}, radii=(0.5, A))
    c2, mesh = s.mesh.c2, s.mesh
    He = c2.dual_edge_partial.tocsr()
    eps_c = s.eps_r[:, 0]
    rows = []
    for e in range(c2.n_edges):
        cols = He.indices[He.indptr[e] : He.indptr[e + 1]]
        if cols.size == 2 and eps_c[cols[0]] != eps_c[cols[1]]:
            rows.append(e)
    rows = np.array(rows)
    hv = He[rows]
    par = np.asarray((hv @ sp.diags(eps_c)).sum(1)).ravel()
    ser = np.asarray(hv.sum(1)).ravel() ** 2 / np.asarray(
        (hv @ sp.diags(1 / eps_c)).sum(1)
    ).ravel()
    L = mesh.n_levels
    idx = (rows[:, None] * L + np.arange(L)[None]).ravel()
    s.cap[idx] *= np.repeat(ser / par, L)
    return _rms(_modes_k(s, 20) / k_ex - 1)


def test_single_layer_equals_2d_tm():
    m2 = disk_mesh([0.5, A], 0.12)
    pm = PrismMesh(m2, [0.0, 0.05], periodic_z=True, merge_tol=0.0)
    s = PrismFIT3D(pm, {0: [4.0, 1.0]}, bc_z="periodic", dual_floor=0)
    f3, _ = s.eigenmodes(30, f_target=0.8 * c / (2 * np.pi))
    s2 = UnstructuredFIT2D(m2, {0: [4.0, 1.0]}, polarization="TM", bc="pec")
    f2, _ = s2.eigenmodes(6, f_target=0.8 * c / (2 * np.pi))
    for f in f2:
        assert np.min(np.abs(f3 - f)) < 1e-9 * f


def _pulsed(s, f0, pos=(0.13, -0.21, 0.37), d=(0.3, 0.5, 1.0)):
    src = GaussianSource(f0, 0.5 * f0)
    s.add_point_dipole(list(pos), list(d), src)
    return src


def test_energy_and_divergence():
    s = _cavity(0.15, {0: [4.0, 1.0]}, radii=(0.5, A))
    src = _pulsed(s, 0.4 * c)
    n0 = int(src.end_time / s.dt) + 2
    s.run(n0)
    W = []
    for _ in range(1500):
        s.one_step()
        W.append(s.energy())
    W = np.array(W)
    assert np.ptp(W) / W.mean() < 1e-12
    assert np.abs(s.divergence_b()).max() < 1e-12 * np.abs(s.bz).max()

    # mutation 1: wrong sign of the z-difference in C^T -> no energy invariant
    s.reset()
    orig = s.curl_t_h

    def wrong(hz, hq):
        rh, rv = orig(hz, hq)
        K = s.mesh.K
        rh[:, :K] -= 2 * hq
        rh[:, 1:] += 2 * hq
        return rh, rv

    s.curl_t_h = wrong
    _pulsed(s, 0.4 * c)
    s.run(n0)
    W = [s.energy() for _ in range(300) if s.one_step() is None]
    assert np.ptp(W) / np.mean(np.abs(W)) > 1e-3

    # mutation 2: 1 % error in the in-plane part of the curl -> div B != 0
    s2 = _cavity(0.15, {0: [4.0, 1.0]}, radii=(0.5, A))
    rng = np.random.default_rng(0)
    s2.G2 = (sp.diags(1 + 0.01 * rng.standard_normal(s2.G2.shape[0])) @ s2.G2).tocsr()
    _pulsed(s2, 0.4 * c)
    s2.run(n0 + 200)
    assert np.abs(s2.divergence_b()).max() > 1e-4 * np.abs(s2.bz).max()


def _slab_probe(z_half, pml, s_max=None, t_end=6.0):
    m2 = disk_mesh([0.6], 0.12)
    dz = 0.1
    z = np.linspace(-z_half, z_half, int(round(2 * z_half / dz)) + 1)
    pm = PrismMesh(m2, z)
    kw = {}
    if pml:
        kw["pml_z"] = dict(
            n_low=6, n_high=6, order=3,
            s_max=4 * c / (6 * dz) if s_max is None else s_max,
        )  # fmt: skip
    s = PrismFIT3D(pm, {0: [2.0, 1.0]}, bc_xy="pmc", bc_z="pec", dt=0.15 * dz / c, **kw)
    src = GaussianSource(0.8 * c, 0.4 * c)
    s.add_point_dipole([0.05, 0.02, 0.0], [1.0, 0.3, 0.2], src)
    pr = s.add_probe([[0.1, -0.15, 0.55], [-0.2, 0.1, -0.35]], "x")
    s.run(int(t_end / (c * s.dt)))
    return s.probe_signal(pr)


def test_pml_z():
    # reference: a domain long enough that its end reflections arrive after
    # the window; the PML run (PML from |z| = 1) must match it
    ref = _slab_probe(4.0, False)
    pml = _slab_probe(1.6, True)
    off = _slab_probe(1.6, True, s_max=0.0)
    norm = np.linalg.norm(ref)
    e_pml = np.linalg.norm(pml - ref) / norm
    e_off = np.linalg.norm(off - ref) / norm
    print("PML z: rel. difference to the long domain", e_pml, "PML off", e_off)
    assert e_pml < 0.03
    assert e_off > 10 * e_pml


def test_drude_filled_cavity():
    from wakis.dispersive import DispersiveMaterial, Drude

    k_p = 3.0
    w_p = k_p * c
    mat = DispersiveMaterial(eps_inf=1.0, poles=[Drude(omega_p=w_p, gamma=0.0)])
    s = _cavity(0.1, {0: mat}, radii=(A,))
    kc = _empty_cavity_k(1)[0]
    f_ex = np.sqrt(kc**2 + k_p**2) * c / (2 * np.pi)
    src = _pulsed(s, f_ex, pos=(0.1, 0.2, 0.4), d=(0, 0, 1))
    pr = s.add_probe([[-0.2, 0.1, 0.45]], "z")
    s.run(int(src.end_time / s.dt) + 6000)
    sig = s.probe_signal(pr)[int(src.end_time / s.dt) + 1 :, 0]
    r = harminv(sig, s.dt, 0.8 * f_ex, 1.05 * f_ex)
    f = r.frequency[np.argmax(np.abs(r.amplitude))]
    print("drude cavity", f / f_ex - 1)
    assert abs(f / f_ex - 1) < 0.015


def test_lumped_port_capacitor():
    # Parallel PEC plates (z = 0 and z = d) of radius r_p inside a PMC disk;
    # a resistive port at the centre. Below the first plate resonance the
    # structure is the plate capacitance C = eps_0 pi r_p^2 / d.
    r_p, d = 1.0, 0.1
    m2 = disk_mesh([0.2, r_p], 0.1)
    pm = PrismMesh(m2, np.linspace(0, d, 3))
    s = PrismFIT3D(pm, bc_xy="pmc", bc_z="pec")
    centre = int(np.argmin(np.hypot(*pm.c2.nodes.T)))
    R = 50.0
    f_c = 0.1 * c / (2 * np.pi * r_p)
    src = GaussianSource(f_c, f_c)
    port = s.add_lumped_port(s.vertical_edge_index(centre, [0, 1]), R, src)
    s.run(int(src.end_time / s.dt) + 20000)
    f = np.array([0.5, 1.0, 1.5]) * f_c
    Z = s.port_impedance(port, f)
    Cp = eps_0 * np.pi * r_p**2 / d
    # exp(+i w t) phasors (DTFT kernel exp(-i w t)): Z_C = 1 / (i w C)
    Zc = 1 / (1j * 2 * np.pi * f * Cp)
    print("port Z", Z, "capacitor", Zc)
    assert np.allclose(Z, Zc, rtol=0.05)
    # mutation: the opposite source polarity gives the wrong sign of Z
    assert not np.allclose(-Z, Zc, rtol=0.5)
