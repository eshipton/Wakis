"""
Adjoint path for tetrahedral ports (`wakis.unstructured.ports3d` + `adjoint`).

What is checked, and how each check can fail:

* Forward replay and port DFT: `UnstructuredAdjointStepper.step` reproduces
  `UnstructuredFIT3D.one_step` bitwise (`assert_array_equal`) with modal and
  lumped ports and massless edges present; `PortDFTMonitor` in
  `UnstructuredAdjointSimulation.forward` reproduces `_Port.spectra(freqs)` and
  the power-wave scattering column `b / a_0` to round-off (~1e-15).
* Multi-step and blockwise dot-product test: for random states x, y and
  k in {1, 5, 10} steps of the port-augmented linear step L (`linear_step` /
  `linear_step_T`), |<y, L^k x> - <(L^T)^k y, x>| / (|<y, L^k x>| + |<(L^T)^k y, x>|)
  <= 1e-15 (and blockwise scale-invariant error <= 1e-15).
* Mutations of the transposed step: dropping the port term in the transpose or
  flipping its sign breaks the dot-product test (error > 1e-3 vs <= 1e-15) and
  breaks the finite-difference gradient check; omitting the half-step
  trapezoidal time-centering of the port monitor leaks an O(omega dt) phase
  error into the gradient and fails the FD check.
* Material-permittivity gradient vs central finite differences: for both a
  multi-frequency |S21|^2 / |S11|^2 objective (`PortDFTMonitor`) and an
  interior `CellDFTMonitor` objective on a ported WR-90 guide, the adjoint
  gradient agrees with central finite differences FD(h) within the derived
  Richardson O(h^2) truncation estimate

      tau_k(h) = |FD_k(2h) - FD_k(h)| / 3

  plus the floating-point cancellation bound r_k(h) = 1e3 * eps_mach * |F| / h
  (derivation in ``docs/physicsguide_sections/unstructured_ports3d_adjoint.md``).
* Binomial checkpointing (`memory="checkpoint"`) reproduces the stored-row
  gradient bitwise (`assert_array_equal`) with zero replay mismatches.
"""

import numpy as np
import pytest

from wakis.unstructured.adjoint import (
    CellDFTMonitor,
    CellMaterialGrid,
    UnstructuredAdjointSimulation,
    UnstructuredAdjointStepper,
)
from wakis.unstructured.mesh import SimplexMesh
from wakis.unstructured.pml3d import _snap_degenerate_duals
from wakis.unstructured.ports3d import (
    ExtrudedLine,
    LumpedPort,
    ModalPort,
    PortAdjointHelper,
    PortDFTMonitor,
    PortPlane,
    attach_ports,
    edge_path,
    extrude_section,
    power_waves,
    rectangle_section,
)
from wakis.unstructured.solver3d import UnstructuredFIT3D

A_WG, B_WG = 22.86e-3, 10.16e-3
F0 = 10e9
BW = 1.5e9
FREQS = np.array([9.0e9, 10.0e9, 11.0e9])
DOT_TOL = 1e-15
MUT_MIN = 1e-6  # as in test_100_unstructured_adjoint.py (>> DOT_TOL = 1e-15)


def _pulse(f0=F0, bw=BW):
    sig = 1 / (2 * np.pi * bw)
    t0 = 4 * sig
    return lambda t: (
        np.exp(-(((t - t0) / sig) ** 2) / 2) * np.sin(2 * np.pi * f0 * (t - t0))
    )


@pytest.fixture(scope="module")
def ported_wr90():
    """
    Small extruded WR-90 section (4 z-layers, h = a/4) with TE10 modal ports at
    both ends, an interior lumped port path, a dielectric slab in layer 2 and
    design tetrahedra in layer 1.
    """
    h = A_WG / 4
    nodes, tris, tags = rectangle_section(A_WG, B_WG, h, wall=h / 2, seed=1)
    z = np.linspace(0.0, 12e-3, 5)
    m = extrude_section(nodes, tris, tags, z)
    zc = m.nodes[m.cells].mean(1)[:, 2]
    tg = m.cell_tags.copy()
    tg[(zc > z[2]) & (zc < z[3]) & (tg == 1)] = 2
    m = SimplexMesh(m.nodes, m.cells, tg)
    _snap_degenerate_duals(m)
    mats = {0: "pec", 2: [2.2, 1.0]}
    eps_layers = np.array([1.0, 1.0, 2.2, 1.0])

    s0 = UnstructuredFIT3D(m, mats, bc="pmc", massless_tol=1e-6)
    dt = s0.dt
    planes0 = [PortPlane(s0, (0, 0, zz), (0, 0, 1)) for zz in (z[0], z[-1])]
    kc, phi0 = planes0[0].te_modes(1)
    phi1 = planes0[1].te_modes(1)[1][:, 0]
    phis = [phi0[:, 0], phi1]
    kc = kc[0]
    line = ExtrudedLine(z, eps_layers, 1.0, kc, dt)
    e_lp, s_lp = edge_path(s0, (A_WG / 2, 0.0, z[2]), (A_WG / 2, B_WG, z[2]))

    def Y(f):
        return np.real(line.port_admittance(f) / line.cos_half(f))

    def make_modal():
        s = UnstructuredFIT3D(m, mats, bc="pmc", massless_tol=1e-6, dt=dt)
        planes = [PortPlane(s, (0, 0, zz), (0, 0, 1)) for zz in (z[0], z[-1])]
        ports = [
            ModalPort(planes[0], phis[0], Y, f0=F0, i_src=_pulse(), name="p1"),
            ModalPort(planes[1], phis[1], Y, f0=F0, name="p2"),
        ]
        attach_ports(s, ports)
        return s

    def make_mixed():
        s = UnstructuredFIT3D(m, mats, bc="pmc", massless_tol=1e-6, dt=dt)
        planes = [PortPlane(s, (0, 0, zz), (0, 0, 1)) for zz in (z[0], z[-1])]
        ports = [
            ModalPort(planes[0], phis[0], Y, f0=F0, i_src=_pulse(), name="p1"),
            ModalPort(planes[1], phis[1], Y, f0=F0, name="p2"),
            LumpedPort(s, e_lp, s_lp, R=50.0, name="lp"),
        ]
        attach_ports(s, ports)
        return s

    # Layer 1 (z[1] < z < z[2]) interior cells do not touch outer boundary edges
    design_cells = np.flatnonzero((zc > z[1]) & (zc < z[2]) & (m.cell_tags == 1))
    return m, make_modal, make_mixed, design_cells, z


def _balanced_states(st, rng):
    """Random (x, y) in dual energy scaling so every block is O(1)."""
    s = st.solver
    cap = s.cap
    mnu = np.where(s.Mnu_diag > 0, s.Mnu_diag, 1.0)
    x = {
        "u": rng.standard_normal(cap.size) / np.sqrt(cap),
        "b": rng.standard_normal(mnu.size) / np.sqrt(mnu),
    }
    y = {
        "u": rng.standard_normal(cap.size) * np.sqrt(cap),
        "b": rng.standard_normal(mnu.size) * np.sqrt(mnu),
    }
    return x, y


def _multistep_dot_error(st, n_steps=1, seed=0):
    rng = np.random.default_rng(seed)
    x, y = _balanced_states(st, rng)
    Lx = {k: v.copy() for k, v in x.items()}
    for _ in range(n_steps):
        Lx = st.linear_step(Lx)
    LTy = {k: v.copy() for k, v in y.items()}
    for _ in range(n_steps):
        LTy = st.linear_step_T(LTy)
    lhs = st.inner(y, Lx)
    rhs = st.inner(LTy, x)
    return abs(lhs - rhs) / (abs(lhs) + abs(rhs))


def _blockwise_dot_error(st, seed=0):
    rng = np.random.default_rng(seed)
    keys = [k for k, _ in st.keys()]
    zero = st.zero_state()
    fwd, adj = {}, {}
    for k in keys:
        x = {kk: v.copy() for kk, v in zero.items()}
        x[k] = rng.standard_normal(x[k].size)
        fwd[k] = (x, st.linear_step(x))
        y = {kk: v.copy() for kk, v in zero.items()}
        y[k] = rng.standard_normal(y[k].size)
        adj[k] = (y, st.linear_step_T(y))
    worst = 0.0
    for i in keys:
        x, Ax = fwd[i]
        for j in keys:
            y, ATy = adj[j]
            a = float(Ax[j] @ y[j])
            b = float(x[i] @ ATy[i])
            scale = np.linalg.norm(Ax[j]) * np.linalg.norm(y[j]) + np.linalg.norm(
                x[i]
            ) * np.linalg.norm(ATy[i])
            if scale > 0:
                worst = max(worst, abs(a - b) / scale)
    return worst


def test_adjoint_forward_bitwise_and_dot_product(ported_wr90):
    """
    `UnstructuredAdjointStepper.step` equals `solver.one_step` bitwise with
    modal and lumped ports attached; `PortDFTMonitor` matches `_Port.spectra`
    to ~1e-15; dot-product test `<y, L^k x> - <(L^T)^k y, x>` relative <= 1e-15
    for k = 1, 5, 10 steps.
    """
    m, make_modal, make_mixed, cells, _ = ported_wr90
    s = make_mixed()
    assert s.massless.size > 0
    st = UnstructuredAdjointStepper(s, cells)
    assert st.ports is not None
    x = st.zero_state()
    nt = 60
    for n in range(nt):
        s.one_step()
        x, _ = st.step(x, n)
    assert np.abs(s.u).max() > 0
    np.testing.assert_array_equal(x["u"], s.u)
    np.testing.assert_array_equal(x["b"], s.b)

    # PortDFTMonitor vs _Port.spectra and power_waves
    design = CellMaterialGrid(m, cells, 1.0, 3.0)
    mon = PortDFTMonitor(FREQS)
    sim = UnstructuredAdjointSimulation(make_modal, design, {"p": mon}, nt)
    V_dft = sim.forward(np.zeros(cells.size), record=False)["p"]["dft"]
    s_ref = make_modal()
    s_ref.run(nt)
    V_ref = np.column_stack([p.spectra(FREQS)[0] for p in s_ref.ports.ports])
    I_ref = np.column_stack([p.spectra(FREQS)[1] for p in s_ref.ports.ports])
    Y_ref = np.column_stack([p.reference_admittance(FREQS) for p in s_ref.ports.ports])
    a_ref, b_ref = power_waves(V_ref, I_ref, Y_ref)
    S_ref = b_ref / a_ref[:, [0]]
    S_mon, _ = mon.s_column(V_dft, sim.solver, nt, excite=0)
    np.testing.assert_allclose(
        V_dft, V_ref, rtol=1e-14, atol=1e-14 * np.abs(V_ref).max()
    )
    np.testing.assert_allclose(
        S_mon, S_ref, rtol=1e-14, atol=1e-14 * np.abs(S_ref).max()
    )

    # Dot-product tests (blockwise and multi-step k = 1, 5, 10)
    blk_err = _blockwise_dot_error(st)
    print(f"blockwise dot error: {blk_err:.3e}")
    assert blk_err <= DOT_TOL
    for k_steps in (1, 5, 10):
        err = _multistep_dot_error(st, n_steps=k_steps, seed=k_steps)
        print(f"k={k_steps} multistep dot error: {err:.3e}")
        assert err <= DOT_TOL


def _drop_port_pre_T(self, u1b):
    return u1b


def _drop_port_post_T(self, rb):
    return np.zeros_like(rb)


def _flip_port_pre_T(self, u1b):
    z = self.W.T @ (self.dinv * u1b)
    return u1b + self.W @ (0.5 * self.g * (self._Sinv.T @ z))


def _flip_port_post_T(self, rb):
    return self.W @ (0.5 * self.g * (self.W.T @ rb))


@pytest.mark.parametrize(
    "name,pre_fn,post_fn",
    [
        ("drop_port_T", _drop_port_pre_T, _drop_port_post_T),
        ("flip_port_T_sign", _flip_port_pre_T, _flip_port_post_T),
    ],
)
def test_adjoint_dot_product_mutations_fail(
    ported_wr90, name, pre_fn, post_fn, monkeypatch
):
    """
    Dropping the port term in `linear_step_T` or flipping its sign must break
    the dot-product test by > 9 orders of magnitude (> MUT_MIN = 1e-6 vs
    DOT_TOL = 1e-15).
    """
    _, _, make_mixed, cells, _ = ported_wr90
    st = UnstructuredAdjointStepper(make_mixed(), cells)
    assert _multistep_dot_error(st, n_steps=3, seed=7) <= DOT_TOL
    monkeypatch.setattr(PortAdjointHelper, "pre_step_T", pre_fn)
    monkeypatch.setattr(PortAdjointHelper, "post_step_T", post_fn)
    err_1 = _multistep_dot_error(st, n_steps=1, seed=7)
    err_3 = _multistep_dot_error(st, n_steps=3, seed=7)
    blk = _blockwise_dot_error(st, seed=7)
    print(
        f"mutation {name}: k=1 err={err_1:.3e}, k=3 err={err_3:.3e}, blockwise={blk:.3e}"
    )
    assert err_1 > MUT_MIN and err_3 > MUT_MIN and blk > MUT_MIN


def _fd_check(sim, rho, g, cells, obj_fn, h=1e-3):
    """
    Compare adjoint gradient `g` against central finite differences FD(h) with
    the derived Richardson O(h^2) truncation estimate

        tau_k(h) = |FD_k(2h) - FD_k(h)| / 3

    and round-off bound r_k(h) = 1e3 * eps_mach * |F(rho)| / h.
    """
    F0_ = obj_fn(sim.forward(rho, record=False))[0]
    out = []
    for k in cells:
        fd = []
        for hh in (h, 2 * h):
            rp, rm = rho.copy(), rho.copy()
            rp[k] += hh
            rm[k] -= hh
            Fp = obj_fn(sim.forward(rp, record=False))[0]
            Fm = obj_fn(sim.forward(rm, record=False))[0]
            fd.append((Fp - Fm) / (2 * hh))
        trunc = abs(fd[1] - fd[0]) / 3.0
        roundoff = 1e3 * np.finfo(float).eps * abs(F0_) / h
        out.append((int(k), float(g[k]), float(fd[0]), float(trunc), float(roundoff)))
    return out


def test_adjoint_gradient_s21_s11_vs_finite_differences(ported_wr90, monkeypatch):
    """
    Adjoint gradient of F(rho) = sum_f (|S21(f)|^2 - 0.5 |S11(f)|^2) w.r.t.
    the per-cell permittivity design vector agrees with central finite
    differences within the Richardson truncation bound

        |g_k - FD_k(h)| <= 1.5 * |FD_k(2h) - FD_k(h)| / 3 + r_k(h).

    Dropping the port term in the transpose or omitting the half-step
    time-centering of the port monitor in the reverse sweep fails this check.
    """
    m, make_modal, _, cells, _ = ported_wr90
    nt = 100
    design = CellMaterialGrid(m, cells, 1.0, 3.0)
    mon = PortDFTMonitor(FREQS)
    sim = UnstructuredAdjointSimulation(make_modal, design, {"p": mon}, nt)

    def obj(out):
        V = out["p"]["dft"]
        S, vjp_s = mon.s_column(V, sim.solver, nt, excite=0)
        val = float(np.sum(np.abs(S[:, 1]) ** 2 - 0.5 * np.abs(S[:, 0]) ** 2))
        G_S = np.column_stack([-S[:, 0], 2.0 * S[:, 1]])
        return val, {"p": {"dft": vjp_s(G_S)}}

    rng = np.random.default_rng(2)
    rho = rng.uniform(0.2, 0.8, design.shape)
    F, g = sim.value_and_grad(rho, obj)
    assert np.all(np.isfinite(g)) and np.abs(g).max() > 0
    pick = rng.choice(rho.size, 4, replace=False)
    checks = _fd_check(sim, rho, g, pick, obj, h=1e-3)
    for k, adj, fd, trunc, roundoff in checks:
        rel = abs(adj - fd) / abs(fd)
        print(
            f"cell {k}: adj={adj:.6e}, fd={fd:.6e}, |adj-fd|={abs(adj - fd):.3e}, "
            f"trunc={trunc:.3e}, rel={rel:.3e}"
        )
        assert abs(adj - fd) <= 1.5 * trunc + roundoff
        assert rel < 1e-5

    # Mutation 1: drop port term in transpose -> gradient error >> truncation bound
    with monkeypatch.context() as mp:
        mp.setattr(PortAdjointHelper, "pre_step_T", _drop_port_pre_T)
        mp.setattr(PortAdjointHelper, "post_step_T", _drop_port_post_T)
        _, g_no_port = sim.value_and_grad(rho, obj)
        rel_bad = max(abs(g_no_port[k] - fd) / abs(fd) for k, _, fd, _, _ in checks)
        print(f"mutation drop_port_T on S-objective: max rel err = {rel_bad:.3e}")
        assert rel_bad > 1e-2

    # Mutation 2: integer-step monitor phase in reverse sweep (half-step error)
    with monkeypatch.context() as mp:
        mp.setattr(mon, "half_step", False)
        G_cot = obj(sim.forward(rho))[1]
        # Forward evaluated with half_step=True, reverse with half_step=False
        mon.half_step = True
        out_fwd = sim.forward(rho, record=True)
        _, G_cot = obj(out_fwd)
        mon.half_step = False
        g_half_err = sim.vjp(G_cot)
        rel_half = max(abs(g_half_err[k] - fd) / abs(fd) for k, _, fd, _, _ in checks)
        print(f"mutation half_step=False in vjp: max rel err = {rel_half:.3e}")
        assert rel_half > 1e-2


def test_adjoint_gradient_cell_dft_and_checkpoint(ported_wr90):
    """
    Adjoint gradient with an interior `CellDFTMonitor` and mixed modal + lumped
    ports matches central FD within the truncation bound, and binomial
    checkpointing matches stored-row mode bitwise (`assert_array_equal`).
    """
    m, _, make_mixed, cells, z = ported_wr90
    nt = 90
    design = CellMaterialGrid(m, cells, 1.0, 3.0, filter_radius=4e-3)
    pts = [[A_WG * 0.45, B_WG * 0.5, 0.5 * (z[2] + z[3])]]
    mon = CellDFTMonitor(FREQS, points=pts, component=1)
    sim = UnstructuredAdjointSimulation(make_mixed, design, {"m": mon}, nt)

    def obj(out):
        X = out["m"]["dft"]
        return float(np.sum(np.abs(X) ** 2)), {"m": {"dft": 2.0 * X}}

    rng = np.random.default_rng(4)
    rho = rng.uniform(0.2, 0.8, design.shape)
    F, g = sim.value_and_grad(rho, obj)
    assert F > 0 and np.abs(g).max() > 0
    pick = rng.choice(rho.size, 3, replace=False)
    for k, adj, fd, trunc, roundoff in _fd_check(sim, rho, g, pick, obj, h=1e-3):
        assert abs(adj - fd) <= 1.5 * trunc + roundoff
        assert abs(adj - fd) < 1e-5 * abs(fd)

    # Checkpointing bitwise equality
    ck = UnstructuredAdjointSimulation(
        make_mixed, design, {"m": mon}, nt, memory="checkpoint", checkpoints=3, block=8
    )
    ck.verify_replay = True
    F_ck, g_ck = ck.value_and_grad(rho, obj)
    assert F_ck == F
    np.testing.assert_array_equal(g_ck, g)
    assert ck.memory_stats["recompute_steps"] > 0
    assert ck._rung.mismatches == 0
