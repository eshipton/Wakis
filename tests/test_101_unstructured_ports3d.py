"""
Lumped and waveguide ports of `UnstructuredFIT3D` (`wakis.unstructured.ports3d`).

Geometry: cross-sections (WR-90 rectangle a x b = 22.86 x 10.16 mm with a
jittered Delaunay triangulation; coax r = 2 / 4.6 mm) extruded along z into
right prisms split into tetrahedra (`extrude_section`), conductor walls as
PEC cells, PMC (natural) outer boundary so the end faces carry the ports.
On such meshes the leapfrog FIT equations of one TE/TEM section mode
separate exactly into a ladder network (`ExtrudedLine`, physics guide
section 3), which gives *discrete-exact* S-parameters; the continuum results
(TE10 / TEM transmission-line theory) differ from them by the derived O(h^2)
discretisation error.

* Mesh: the extruded dual is the tensor product of the 2D Voronoi dual and
  the 1D z dual (dual area h_e * D, half cells at the ends; diagonals 0).
* Port modes: TE10 cutoff k_c,h -> pi / a at second order (Voronoi and
  Whitney port Hodges); the TEM mode of the coax is a discrete gradient.
* Straight guide: |S11| (floor) and |S21 - exp(-i N theta)| at the DFT
  truncation level (< 1e-8; the residual-energy ratio is checked); the S21
  phase differs from the continuum beta L by the predicted discrete
  dispersion. Unitarity |S11|^2 + |S21|^2 = 1.
* Dielectric slab (eps 2.2) in the guide: two-run 2x2 S-matrix equals the
  discrete ladder to < 1e-7; reciprocity and unitarity.
* Coax line terminated by a modal (sheet) load R_L = 100 Ohm: reflection
  equals the discrete ladder (< 1e-8); vs continuum transmission-line
  theory the difference is bounded by the derived Z0 and beta errors.
* Lumped port: time-domain S11 equals an independent frequency-domain solve
  of the same discrete equations (< 1e-6 limited by the measured
  truncation); DC limit Z_in -> R_L with an inductive O(f) term;
  two lumped ports are reciprocal and lossless (unitary) to the same level.
* Mode mismatch (negative control): the Whitney-mass port mode on the
  Voronoi solver is not the discrete 3D mode; its S11 floor is O(k^2 h^2)
  (>> the matched floor), so the floor test can fail.
* Adjoint: the port step is linear with an exact transpose,
  <y, S x> = <S^T y, x> to ~1e-15 (a transpose without the massless term
  fails); the reverse-mode gradient of a port-voltage objective w.r.t. the
  source samples matches central finite differences.
"""

import numpy as np
import pytest
from scipy.constants import c, epsilon_0, mu_0

from wakis.unstructured.mesh import SimplexMesh
from wakis.unstructured.pml3d import _snap_degenerate_duals
from wakis.unstructured.ports3d import (
    ExtrudedLine,
    LumpedPort,
    ModalPort,
    PortPlane,
    attach_ports,
    coax_section,
    edge_path,
    extrude_section,
    power_waves,
    rectangle_section,
    s_matrix,
)
from wakis.unstructured.solver3d import UnstructuredFIT3D

ETA = np.sqrt(mu_0 / epsilon_0)
A_WG, B_WG = 22.86e-3, 10.16e-3
F_X = np.linspace(8e9, 12e9, 9)


def pulse(f0, bw):
    sig = 1 / (2 * np.pi * bw)
    t0 = 6 * sig
    return lambda t: np.exp(-(((t - t0) / sig) ** 2) / 2) * np.sin(
        2 * np.pi * f0 * (t - t0)
    )


def wr90(h_div=6, dz=2.5e-3, length=40e-3, slab=None, seed=1):
    h = A_WG / h_div
    nodes, tris, tags = rectangle_section(A_WG, B_WG, h, wall=h / 2, seed=seed)
    z = np.linspace(0, length, int(round(length / dz)) + 1)
    m = extrude_section(nodes, tris, tags, z)
    mats = {0: "pec"}
    eps_layers = np.ones(z.size - 1)
    if slab is not None:
        z0, z1, eps = slab
        zc = m.nodes[m.cells].mean(1)[:, 2]
        tg = m.cell_tags.copy()
        tg[(zc > z0) & (zc < z1) & (tg == 1)] = 2
        m = SimplexMesh(m.nodes, m.cells, tg)
        _snap_degenerate_duals(m)
        mats[2] = [eps, 1.0]
        mid = (z[:-1] + z[1:]) / 2
        eps_layers[(mid > z0) & (mid < z1)] = eps
    return m, mats, z, eps_layers


def modal_runs(m, mats, z, eps_layers, kind="te", n_runs=2, t_end=10e-9,
               port_hodge="voronoi", f0=10e9, bw=1.2e9):
    """Run n_runs excitations (port k excited in run k); return S, line, info."""
    V, I, info = [], [], {}
    for run in range(n_runs):
        s = UnstructuredFIT3D(m, mats, bc="pmc", massless_tol=1e-6)
        planes = [PortPlane(s, (0, 0, zz), (0, 0, 1), hodge=port_hodge)
                  for zz in (z[0], z[-1])]
        if kind == "te":
            kc, phi0 = planes[0].te_modes(1)
            phis = [phi0[:, 0], planes[1].te_modes(1)[1][:, 0]]
            kc = kc[0]
        else:
            phis = [p.tem_mode()[0] for p in planes]
            kc = 0.0
        line = ExtrudedLine(z, eps_layers, 1.0, kc, s.dt)

        def Y(f, line=line):
            return np.real(line.port_admittance(f) / line.cos_half(f))

        ports = [
            ModalPort(planes[k], phis[k], Y, f0=f0,
                      i_src=pulse(f0, bw) if k == run else None)
            for k in range(2)
        ]
        attach_ports(s, ports)
        n = int(t_end / s.dt)
        e_max = 0.0
        for _ in range(20):
            s.run(n // 20)
            e_max = max(e_max, s.energy())
        info.setdefault("energy_ratio", []).append(s.energy() / e_max)
        spec = [p.spectra(F_X) for p in ports]
        V.append([sp_[0] for sp_ in spec])
        I.append([sp_[1] for sp_ in spec])
    V = np.transpose(np.array(V), (2, 0, 1))
    I = np.transpose(np.array(I), (2, 0, 1))
    Yr = np.stack([Y(F_X), Y(F_X)], 1)
    info.update(V=V, I=I, Y=Yr, kc=kc, dt=s.dt)
    if n_runs == 2:
        return s_matrix(V, I, Yr), line, info
    return None, line, info


# ------------------------------------------------------------------ mesh
def test_extruded_dual_is_tensor_product():
    """Transverse edges: dual area = h_e (D_below + D_above) / 2; z edges:
    node dual area A_n; quad diagonals: 0. Non-uniform z levels."""
    h = A_WG / 4
    nodes, tris, tags = rectangle_section(A_WG, B_WG, h, seed=3)
    z = np.array([0.0, 2e-3, 5e-3, 6e-3])
    m = extrude_section(nodes, tris, tags, z)
    m2 = SimplexMesh(nodes, tris)
    n = len(nodes)
    area = m.dual_measure()
    E = m.edges
    lvl0, lvl1 = E[:, 0] // n, E[:, 1] // n
    i0, i1 = E[:, 0] % n, E[:, 1] % n
    half = np.r_[0, np.diff(z)] / 2 + np.r_[np.diff(z), 0] / 2  # dual height per level
    look = {tuple(e): k for k, e in enumerate(map(tuple, m2.edges))}
    he = m2.dual_length()
    trans = np.flatnonzero(lvl0 == lvl1)
    exp_t = np.array([he[look[(i0[k], i1[k])]] * half[lvl0[k]] for k in trans])
    assert np.allclose(area[trans], exp_t, rtol=1e-10, atol=1e-12 * h**2)
    vert = np.flatnonzero((lvl1 == lvl0 + 1) & (i0 == i1))
    An = np.asarray(m2.dual_area_partial.sum(1)).ravel()
    assert np.allclose(area[vert], An[i0[vert]], rtol=1e-10, atol=1e-12 * h**2)
    diag = np.flatnonzero((lvl1 == lvl0 + 1) & (i0 != i1))
    assert diag.size > 0 and np.all(area[diag] == 0.0)


# ------------------------------------------------------------------ port modes
@pytest.mark.parametrize("hodge", ["voronoi", "whitney"])
def test_te10_cutoff_second_order(hodge):
    """k_c,h of TE10 converges to pi / a at second order (h = a/4 .. a/16,
    jitter 0.1; observed order in [1.6, 2.6])."""
    errs = []
    hs = [A_WG / 4, A_WG / 8, A_WG / 16]
    for h in hs:
        nodes, tris, tags = rectangle_section(A_WG, B_WG, h, wall=h / 2, jitter=0.1, seed=0)
        m = extrude_section(nodes, tris, tags, [0.0, h])
        s = UnstructuredFIT3D(m, {0: "pec"}, bc="pmc", massless_tol=1e-6)
        kc, _ = PortPlane(s, (0, 0, 0), (0, 0, 1), hodge=hodge).te_modes(1)
        errs.append(abs(kc[0] / (np.pi / A_WG) - 1))
    p = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert 1.6 < p < 2.6, (errs, p)
    assert errs[-1] < 2e-3


def test_tem_mode_is_discrete_gradient():
    """TEM mode of the coax port: discrete curl 0 and divergence (G^T M phi)
    0 on interior nodes to round-off. The discrete line impedance
    Z0,h = eta volts^2 converges to eta ln(b/a) / 2 pi at second order
    (h = 1 -> 0.5 mm: error ratio in [3.5, 4.7], i.e. order 1.8-2.2)."""
    errs = []
    for h in (1.0e-3, 0.5e-3):
        nodes, tris, tags = coax_section(2e-3, 4.6e-3, h, wall=h)
        m = extrude_section(nodes, tris, tags, [0.0, h])
        s = UnstructuredFIT3D(m, {0: "pec", 2: "pec"}, bc="pmc", massless_tol=1e-6)
        P = PortPlane(s, (0, 0, 0), (0, 0, 1))
        phi, volts = P.tem_mode()
        assert np.abs(P.mesh2d.C @ phi).max() < 1e-12
        div = P.mesh2d.G.T @ (P.M @ phi)
        on = np.zeros(P.mesh2d.n_nodes, bool)
        on[P.mesh2d.edges[~P.dof].ravel()] = True
        assert np.abs(div[~on]).max() < 1e-12 * np.abs(P.M @ phi).max()
        errs.append(abs(volts**2 / (np.log(2.3) / (2 * np.pi)) - 1))
    assert 3.5 < errs[0] / errs[1] < 4.7, errs


# ------------------------------------------------------------------ waveguide
@pytest.fixture(scope="module")
def straight():
    m, mats, z, eps = wr90()
    return modal_runs(m, mats, z, eps, n_runs=1), z


def test_straight_guide_discrete_exact(straight):
    """Uniform guide, one excitation, port 2 terminated by g = Y_h(f0): with
    the symmetric 2-port relations, S11 = 0 and S21 = exp(-i N theta) to the
    DFT truncation level (energy left < 1e-16 of the peak)."""
    (_, line, info), z = straight
    V, I, Y = info["V"][:, 0], info["I"][:, 0], info["Y"]
    a, b = power_waves(V, I, Y)
    S11, S21 = [], []
    for k in range(F_X.size):
        M = np.array([[a[k, 0], a[k, 1]], [a[k, 1], a[k, 0]]])
        x = np.linalg.solve(M, b[k])
        S11.append(x[0])
        S21.append(x[1])
    S11, S21 = np.array(S11), np.array(S21)
    theta = line.theta(F_X)
    assert max(info["energy_ratio"]) < 1e-16
    assert np.abs(S11).max() < 1e-8
    assert np.abs(S21 - np.exp(-1j * theta * (z.size - 1))).max() < 1e-8
    assert np.abs(np.abs(S11) ** 2 + np.abs(S21) ** 2 - 1).max() < 1e-8
    # The termination is mismatched away from f0 (a2 != 0), so the test
    # really separates incident and reflected waves
    assert np.abs(a[:, 1] / a[:, 0]).max() > 1e-2


def test_straight_guide_phase_vs_continuum(straight):
    """Continuum S21 = exp(-i beta L): the phase error equals the discrete
    dispersion (k_c,h and the z/t stencil), which is O(h^2): predicted
    (beta_h - beta) L reproduces the measured phase error to 1e-8 rad and is
    < 0.15 rad at h = a/6, D = 2.5 mm."""
    (_, line, info), z = straight
    V, I, Y = info["V"][:, 0], info["I"][:, 0], info["Y"]
    a, b = power_waves(V, I, Y)
    S21 = np.array([np.linalg.solve([[a[k, 0], a[k, 1]], [a[k, 1], a[k, 0]]], b[k])[1]
                    for k in range(F_X.size)])
    L = z[-1]
    beta = np.sqrt((2 * np.pi * F_X / c) ** 2 - (np.pi / A_WG) ** 2)
    meas = np.angle(S21 * np.exp(1j * beta * L))
    pred = -(np.real(line.beta(F_X)) - beta) * L
    assert np.abs(meas - pred).max() < 1e-8
    assert np.abs(meas).max() < 0.15


def test_dielectric_slab_two_port():
    """eps = 2.2 slab (15-25 mm) in WR-90: 2x2 S from two runs equals the
    discrete ladder (< 1e-7, truncation-limited), reciprocal and unitary;
    |S11| spans 0.05-0.56 over 8-12 GHz."""
    m, mats, z, eps = wr90(slab=(15e-3, 25e-3, 2.2))
    S, line, info = modal_runs(m, mats, z, eps)
    Sd = line.s_matrix(F_X)
    assert np.abs(S - Sd).max() < 1e-7
    assert np.abs(S[:, 0, 1] - S[:, 1, 0]).max() < 1e-7
    U = np.einsum("fji,fjk->fik", S.conj(), S)
    assert np.abs(U - np.eye(2)).max() < 1e-7
    assert np.abs(S[:, 0, 0]).max() > 0.5


def test_mode_mismatch_raises_s11_floor():
    """Negative control: the Whitney-mass port mode is not the discrete mode
    of the Voronoi solver; the reflection floor of the straight guide rises to
    O((k h)^2) (1e-4 .. 1e-1), > 1e3 x the matched floor of 1e-8."""
    m, mats, z, eps = wr90(length=20e-3)
    _, line, info = modal_runs(m, mats, z, eps, n_runs=1, port_hodge="whitney")
    V, I, Y = info["V"][:, 0], info["I"][:, 0], info["Y"]
    a, b = power_waves(V, I, Y)
    S11 = np.array([np.linalg.solve([[a[k, 0], a[k, 1]], [a[k, 1], a[k, 0]]], b[k])[0]
                    for k in range(F_X.size)])
    assert 1e-4 < np.abs(S11).max() < 1e-1


# ------------------------------------------------------------------ TEM line
def coax(h=1.0e-3, length=10e-3, dz=1e-3):
    nodes, tris, tags = coax_section(2e-3, 4.6e-3, h, wall=0.8e-3)
    z = np.linspace(0, length, int(round(length / dz)) + 1)
    m = extrude_section(nodes, tris, tags, z)
    return m, {0: "pec", 2: "pec"}, z


def test_coax_resistive_termination():
    """Coax (Z0 = 50 Ohm continuum) of length 20 mm terminated by a sheet
    resistor R_L = 100 Ohm (modal conductance volts^2 / R_L): the input
    reflection equals the discrete ladder to < 1e-8, and the continuum
    Gamma_L exp(-2 i beta L) to within the bound derived from the discrete
    Z0 (2D Hodge) and beta (z/t stencil) errors."""
    m, mats, z = coax(length=20e-3)
    s = UnstructuredFIT3D(m, mats, bc="pmc", massless_tol=1e-6)
    P1 = PortPlane(s, (0, 0, 0), (0, 0, 1))
    P2 = PortPlane(s, (0, 0, z[-1]), (0, 0, 1))
    phi1, v1 = P1.tem_mode()
    phi2, v2 = P2.tem_mode()
    line = ExtrudedLine(z, 1.0, 1.0, 0.0, s.dt)

    def Y(f):
        return np.real(line.port_admittance(f) / line.cos_half(f))

    RL = 100.0
    g_load = v2**2 / RL
    p1 = ModalPort(P1, phi1, Y, f0=5e9, i_src=pulse(5e9, 1.5e9), voltage_scale=v1)
    p2 = ModalPort(P2, phi2, Y, g=g_load)
    attach_ports(s, [p1, p2])
    s.run(int(5e-9 / s.dt))
    f = np.linspace(2e9, 8e9, 7)
    V, I = p1.spectra(f)
    a, b = power_waves(V, I, Y(f))
    G = b / a
    assert np.abs(G - line.input_reflection(f, g_load)).max() < 1e-8
    # Continuum: Z0 = eta ln(b/a) / 2 pi, beta = w / c
    Z0c = ETA * np.log(2.3) / (2 * np.pi)
    Z0h = v1**2 / Y(f)  # discrete line impedance in volts (frequency dependent via the stencil)
    GLc = (RL - Z0c) / (RL + Z0c)
    Gc = GLc * np.exp(-2j * (2 * np.pi * f / c) * z[-1])
    # Triangle inequality: |G - Gc| <= |GL,h - GL,c| + |GL,c| 2 L |beta_h - beta|
    GLh = (RL - Z0h) / (RL + Z0h)
    dbeta = np.abs(np.real(line.beta(f)) - 2 * np.pi * f / c)
    bound = np.abs(GLh - GLc) + 2 * z[-1] * dbeta * abs(GLc)
    assert np.all(np.abs(G - Gc) <= bound * (1 + 1e-6) + 1e-8)
    assert np.abs(G - Gc).max() > 1e-4  # the continuum is not reproduced exactly


# ------------------------------------------------------------------ lumped
@pytest.fixture(scope="module")
def lumped_coax():
    m, mats, z = coax()
    return m, mats, z


def lumped_pair(m, mats, z, excite, R=(50.0, 50.0), f0=5e9, bw=1.5e9):
    s = UnstructuredFIT3D(m, mats, bc="pmc", massless_tol=1e-6)
    e1, s1 = edge_path(s, (2e-3, 0, 0), (4.6e-3, 0, 0))
    e2, s2 = edge_path(s, (0, 2e-3, z[-1]), (0, 4.6e-3, z[-1]))
    q = [LumpedPort(s, e1, s1, R=R[0], v_src=pulse(f0, bw) if excite == 0 else None),
         LumpedPort(s, e2, s2, R=R[1], v_src=pulse(f0, bw) if excite == 1 else None)]
    return s, q, attach_ports(s, q)


def test_lumped_port_td_equals_fd(lumped_coax):
    """Radial edge-path ports at both ends of a 10 mm coax (R = 50 Ohm):
    the time-domain S11 equals the direct frequency-domain solve of the same
    discrete equations (w~, cos(w dt/2) of the trapezoidal conductance) to
    < 1e-6; a V^n (instead of V^{n+1/2}) monitor would differ by O(w dt)."""
    m, mats, z = lumped_coax
    s, q, ps = lumped_pair(m, mats, z, 0)
    s.run(int(3e-9 / s.dt))
    f = np.linspace(1e9, 8e9, 8)
    V, I = q[0].spectra(f)
    a, b = power_waves(V, I, 1 / 50.0)
    Vfd = ps.frequency_response(f, [1.0, 0.0])[:, 0]
    afd, bfd = power_waves(Vfd, 1.0 - Vfd / 50.0, 1 / 50.0)
    assert np.abs(b / a - bfd / afd).max() < 1e-6
    assert np.abs(bfd / afd).max() > 0.5  # strongly reactive feed: a real test


def test_lumped_port_dc_limit(lumped_coax):
    """At f -> 0 the two radial paths connect the same two (equipotential)
    conductors: Z_in -> R_L exactly, plus an inductive term linear in f."""
    m, mats, z = lumped_coax
    _, _, ps = lumped_pair(m, mats, z, 0, R=(50.0, 75.0))
    f = np.array([1e6, 1e7])
    V = ps.frequency_response(f, [1.0, 0.0])[:, 0]
    Zin = V / (1.0 - V / 50.0)
    assert abs(Zin[0].real / 75.0 - 1) < 1e-6
    assert abs(Zin[1].imag / Zin[0].imag / 10 - 1) < 1e-3
    assert Zin[0].imag > 0


def test_lumped_two_port_reciprocal_lossless(lumped_coax):
    """Two runs, R = 50 / 75 Ohm references: S12 = S21 and S^H S = I to the
    truncation level of the lossless line (< 1e-6)."""
    m, mats, z = lumped_coax
    R = (50.0, 75.0)
    f = np.linspace(1e9, 8e9, 8)
    V, I = [], []
    for k in range(2):
        s, q, _ = lumped_pair(m, mats, z, k, R=R)
        s.run(int(3e-9 / s.dt))
        sp_ = [p.spectra(f) for p in q]
        V.append([x[0] for x in sp_])
        I.append([x[1] for x in sp_])
    V = np.transpose(np.array(V), (2, 0, 1))
    I = np.transpose(np.array(I), (2, 0, 1))
    S = s_matrix(V, I, np.tile(1 / np.array(R), (f.size, 1)))
    assert np.abs(S[:, 0, 1] - S[:, 1, 0]).max() < 1e-6
    U = np.einsum("fji,fjk->fik", S.conj(), S)
    assert np.abs(U - np.eye(2)).max() < 1e-6
    assert np.abs(S[:, 1, 0]).min() > 0.05


# ------------------------------------------------------------------ adjoint
@pytest.fixture(scope="module")
def small_ported():
    m, mats, z = coax(h=1.2e-3, length=4e-3)
    s = UnstructuredFIT3D(m, mats, bc="pmc", massless_tol=1e-6)
    e1, s1 = edge_path(s, (2e-3, 0, 0), (4.6e-3, 0, 0))
    P2 = PortPlane(s, (0, 0, z[-1]), (0, 0, 1))
    phi2, _ = P2.tem_mode()
    q = [LumpedPort(s, e1, s1, R=50.0),
         ModalPort(P2, phi2, lambda f: np.full(np.shape(f), 1e-3), g=2e-3)]
    ps = attach_ports(s, q)
    return s, ps


def test_linear_step_matches_solver(small_ported):
    """`PortSet.linear_step` reproduces `UnstructuredFIT3D.one_step` with ports
    (massless edges present) to round-off."""
    s, ps = small_ported
    assert s.massless.size > 0
    rng = np.random.default_rng(0)
    s.reset()
    s.u = rng.standard_normal(s.free.size)
    s.b = rng.standard_normal(s.mesh.n_faces) * 1e-12
    u0, b0 = s.u.copy(), s.b.copy()
    s.one_step()
    u1, b1 = ps.linear_step(u0, b0, np.zeros(2))
    assert np.abs(u1 - s.u).max() <= 1e-12 * np.abs(u1).max()
    assert np.abs(b1 - s.b).max() <= 1e-12 * np.abs(b1).max()


def test_dot_product(small_ported):
    """<(y_u, y_b), S(x_u, x_b, x_I)> = <S^T(y_u, y_b), (x_u, x_b, x_I)> to
    ~1e-15 relative; dropping the massless term of the transpose fails."""
    s, ps = small_ported
    rng = np.random.default_rng(1)
    # Physical scales: u ~ 1 V, b ~ dt V (Faraday), I ~ C u / dt, and the
    # adjoint states in the dual scaling, so that every term contributes
    nf, nb = s.free.size, s.mesh.n_faces
    bs = s.dt * 1.0
    xu, xb = rng.standard_normal(nf), bs * rng.standard_normal(nb)
    xi = s.cap.mean() / s.dt * rng.standard_normal(2)
    yu, yb = rng.standard_normal(nf), rng.standard_normal(nb) / bs * s.cap.mean()
    fu, fb = ps.linear_step(xu, xb, xi)
    lhs = yu @ fu + yb @ fb
    tu, tb, ti = ps.linear_step_T(yu, yb)
    rhs = tu @ xu + tb @ xb + ti @ xi
    scale = np.abs(yu * fu).sum() + np.abs(yb * fb).sum()
    assert abs(lhs - rhs) / scale < 1e-14
    # Negative control: transpose without the massless (Schur) term
    bad = tu - ps.dt * (s._Ksf.T @ s._Kss.solve(s.Cs.T @ tb, trans="T"))
    rhs_bad = bad @ xu + tb @ xb + ti @ xi
    assert abs(lhs - rhs_bad) / scale > 1e-6


def test_adjoint_gradient_vs_fd(small_ported):
    """Reverse-mode gradient of J = sum_n V_2^{n+1/2}^2 (load-port voltage)
    w.r.t. the impressed current samples I_1^{n+1/2}, through 200 ported
    steps, vs central differences (J is quadratic: FD exact up to round-off)."""
    s, ps = small_ported
    nt = 200
    rng = np.random.default_rng(2)
    Is = rng.standard_normal(nt)
    W = ps.W

    def forward(Is):
        u, b = np.zeros(s.free.size), np.zeros(s.mesh.n_faces)
        J, us = 0.0, [u]
        for n in range(nt):
            u1, b = ps.linear_step(u, b, np.array([Is[n], 0.0]))
            V2 = 0.5 * (W[:, 1].T @ (u + u1)).item()
            J += V2**2
            u = u1
            us.append(u)
        return J, us

    J0, us = forward(Is)
    ub, bb = np.zeros(s.free.size), np.zeros(s.mesh.n_faces)
    grad = np.zeros(nt)
    w2 = W[:, 1].toarray().ravel()
    for n in range(nt - 1, -1, -1):
        V2 = 0.5 * w2 @ (us[n] + us[n + 1])
        # dJ/du^{n+1} from V2^{n+1/2}; dJ/du^n is added after the transpose
        ub = ub + V2 * w2
        ub, bb, ib = ps.linear_step_T(ub, bb)
        grad[n] = ib[0]
        ub = ub + V2 * w2
    for n in (0, 57, 150):
        d = 1e-3
        Jp, _ = forward(Is + d * (np.arange(nt) == n))
        Jm, _ = forward(Is - d * (np.arange(nt) == n))
        fd = (Jp - Jm) / (2 * d)
        assert abs(grad[n] - fd) <= 1e-7 * (abs(fd) + 1e-30 + np.abs(grad).max())
    assert np.abs(grad).max() > 0
