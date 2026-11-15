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

import json
import os
import sys

import numpy as np
import pytest

from wakis.unstructured import mpi as M
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
    dtft,
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


# ---------------------------------------------------------------------------
# General (non-extruded) tetrahedral meshes: accuracy, from W3-4 part D.
# Numbers are read from studies/unstructured-ports3d/out/general_mesh_accuracy.json
# (Batch job w3p-d-accuracy); the block below quotes them.
# ---------------------------------------------------------------------------

_STUDY_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "studies", "unstructured-ports3d")
)
if _STUDY_DIR not in sys.path:
    sys.path.insert(0, _STUDY_DIR)

import general_mesh_accuracy as gma


@pytest.fixture(scope="module")
def wr90_coarse2_results() -> dict:
    """Evaluate the 2 coarsest WR-90 resolutions (h_div = 4, 6) for general and extruded."""
    gen = [gma.evaluate_guide_case("wr90", "general", d) for d in gma.WR90_H_DIVS[:2]]
    ext = [gma.evaluate_guide_case("wr90", "extruded", d) for d in gma.WR90_H_DIVS[:2]]
    s11_fit = gma.fit_convergence_order(
        [r["h_m"] for r in gen], [r["s11_floor"] for r in gen]
    )
    phase_fit = gma.fit_convergence_order(
        [r["h_m"] for r in gen], [r["s21_phase_err_rad"] for r in gen]
    )
    return {
        "general": gen,
        "extruded": ext,
        "s11_order_p": s11_fit["coarse2_order_p"],
        "phase_order_p": phase_fit["coarse2_order_p"],
    }


@pytest.fixture(scope="module")
def coax_coarse2_results() -> dict:
    """Evaluate the 2 coarsest coaxial resolutions (nr = 2, 3) for general and extruded."""
    gen = [gma.evaluate_guide_case("coax", "general", nr) for nr in gma.COAX_NRS[:2]]
    ext = [gma.evaluate_guide_case("coax", "extruded", nr) for nr in gma.COAX_NRS[:2]]
    s11_fit = gma.fit_convergence_order(
        [r["h_m"] for r in gen], [r["s11_floor"] for r in gen]
    )
    phase_fit = gma.fit_convergence_order(
        [r["h_m"] for r in gen], [r["s21_phase_err_rad"] for r in gen]
    )
    return {
        "general": gen,
        "extruded": ext,
        "s11_order_p": s11_fit["coarse2_order_p"],
        "phase_order_p": phase_fit["coarse2_order_p"],
    }


def _assert_wr90_general_accuracy(gen_records: list[dict], s11_order_p: float) -> None:
    """Check WR-90 general-mesh reflection floor and convergence order window."""
    # From general_mesh_accuracy.json:
    # wr90_general_s11: order_p = 1.7695, log_rms_residual = 0.0257,
    # delta_p_2level = 0.1795, study_min_order_p = 1.5900, max floor = 2.3467e-02.
    assert gen_records[0]["s11_floor"] <= 2.3467e-02 * (1.0 + 0.0257)
    assert gen_records[1]["s11_floor"] < gen_records[0]["s11_floor"]
    assert s11_order_p >= 1.5900


def test_general_wr90_order_and_extruded_baseline(wr90_coarse2_results: dict) -> None:
    """WR-90 general mesh shows O((kh)^2) phase convergence and extruded baseline is exact."""
    gen = wr90_coarse2_results["general"]
    ext = wr90_coarse2_results["extruded"]

    # S21 phase error 2-level order (2.0685) meets theory order 2.0 minus
    # delta_p_2level = 0.1236 (from log_rms_residual = 0.0177): bound = 1.8764.
    assert wr90_coarse2_results["phase_order_p"] >= 1.8764

    # |S11| floor decreases with h and matches the 4-level study window (>= 1.5900).
    _assert_wr90_general_accuracy(gen, wr90_coarse2_results["s11_order_p"])

    # Extruded baseline (extrude_section) at equal cpw (5.25 and 7.87) has
    # machine-precision reflection floor <= 1.8851e-12.
    for r_gen, r_ext in zip(gen, ext):
        assert r_gen["cpw"] == pytest.approx(r_ext["cpw"], rel=1e-12)
        assert r_ext["s11_floor"] <= 1.8851e-12
        assert r_gen["s11_floor"] > r_ext["s11_floor"]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Pre-asymptotic WR-90 general-mesh |S11| 2-level order at cpw 5.25->7.87 "
        "is 1.7716 (4-level study fit 1.7695), below strict asymptotic theory "
        "bound 2.0 - 0.1795 = 1.8205"
    ),
)
def test_general_wr90_s11_strict_theory_order(wr90_coarse2_results: dict) -> None:
    """Strict asymptotic theory order check (p >= 2.0 - 0.1795 = 1.8205) for coarse WR-90 |S11|."""
    assert wr90_coarse2_results["s11_order_p"] >= 1.8205


def test_general_coax_order_and_extruded_baseline(coax_coarse2_results: dict) -> None:
    """Coaxial TEM general mesh meets convergence windows and extruded baseline is exact."""
    gen = coax_coarse2_results["general"]
    ext = coax_coarse2_results["extruded"]

    # S21 phase error 2-level order (1.7634) meets theory order 2.0 minus
    # delta_p_2level = 0.3219 (from log_rms_residual = 0.0461): bound = 1.6781.
    assert coax_coarse2_results["phase_order_p"] >= 1.6781

    # |S11| floor decreases from 5.9155e-04 to 2.8971e-04 (2-level order 1.7606),
    # inside the 4-level study window (order_p = 1.9224, delta_p_2level = 0.2067,
    # study_min_order_p = 1.7157; 4-level order 1.9224 >= theory_min_order_p = 1.7933).
    assert gen[0]["s11_floor"] <= 5.9155e-04 * (1.0 + 0.0296)
    assert gen[1]["s11_floor"] < gen[0]["s11_floor"]
    assert coax_coarse2_results["s11_order_p"] >= 1.7157

    # Extruded baseline (extrude_section) at equal cpw (23.06 and 34.59) has
    # machine-precision reflection floor <= 5.5024e-14.
    for r_gen, r_ext in zip(gen, ext):
        assert r_gen["cpw"] == pytest.approx(r_ext["cpw"], rel=1e-12)
        assert r_ext["s11_floor"] <= 5.5024e-14
        assert r_gen["s11_floor"] > r_ext["s11_floor"]


def test_general_port_face_area_mutation_fails() -> None:
    """Halving the port-face dual area weights (port_face_scale=0.5) breaks matching and order."""
    mut_gen = [
        gma.evaluate_guide_case("wr90", "general", d, port_face_scale=0.5)
        for d in gma.WR90_H_DIVS[:2]
    ]
    mut_fit = gma.fit_convergence_order(
        [r["h_m"] for r in mut_gen], [r["s11_floor"] for r in mut_gen]
    )
    # With port_face_scale = 0.5, |S11| jumps to 8.6769e-01 and 8.6313e-01
    # and the 2-level order collapses to 0.0130, failing the accuracy check.
    with pytest.raises(AssertionError):
        _assert_wr90_general_accuracy(mut_gen, mut_fit["coarse2_order_p"])


# ---------------------------------------------------------------------------
# Distributed tetrahedral ports under MPI (`wakis.unstructured.mpi`).
# Skipped unless mpi4py is present and MPI.COMM_WORLD.Get_size() >= 2.
# ---------------------------------------------------------------------------
try:
    from mpi4py import MPI

    _SIZE = MPI.COMM_WORLD.Get_size()
except ImportError:  # pragma: no cover
    MPI, _SIZE = None, 1

mpi = pytest.mark.skipif(_SIZE < 2, reason="needs mpiexec -n >= 2")
MPI_NT = 60


def _build_wr90_port_factories():
    """Standalone factory builder matching `ported_wr90` for CLI / pytest use."""
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

    design_cells = np.flatnonzero((zc > z[1]) & (zc < z[2]) & (m.cell_tags == 1))
    return m, make_modal, make_mixed, design_cells, z


def _make_mpi_port_case(ported_wr90, kind):
    """
    Build a port-bearing WR-90 solver (`kind` in {"modal", "lumped", "mixed"})
    with interior E/H probes and a running multi-frequency field DFT.

    In the `"lumped"` case, port `lp1` runs horizontally across `x in [0, A_WG]`
    at `y = B_WG / 2, z = z[1]`, so its 4 free edges are split `[2, 2]` across
    ranks `[0, 1]` on 2 RCB ranks and `[2, 0, 2, 0]` across ranks `[0, 2]` on
    4 RCB ranks.
    """
    m, make_modal, make_mixed, _, z = ported_wr90
    if kind == "modal":
        s = make_modal()
    elif kind == "mixed":
        s = make_mixed()
    elif kind == "lumped":
        mats = {0: "pec", 2: [2.2, 1.0]}
        s0 = make_modal()
        s = UnstructuredFIT3D(m, mats, bc="pmc", massless_tol=1e-6, dt=s0.dt)
        e1, s1 = edge_path(s, (0.0, B_WG / 2, z[1]), (A_WG, B_WG / 2, z[1]))
        e2, s2 = edge_path(s, (A_WG / 2, 0.0, z[3]), (A_WG / 2, B_WG, z[3]))
        attach_ports(
            s,
            [
                LumpedPort(s, e1, s1, R=50.0, v_src=_pulse(), name="lp1"),
                LumpedPort(s, e2, s2, R=50.0, name="lp2"),
            ],
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown port case kind {kind!r}")
    s.add_probe(
        [
            [A_WG * 0.45, B_WG * 0.5, 0.5 * (z[1] + z[2])],
            [A_WG * 0.55, B_WG * 0.5, 0.5 * (z[2] + z[3])],
        ],
        1,
    )
    s.add_probe([[A_WG * 0.35, B_WG * 0.4, 0.5 * (z[0] + z[1])]], 0)
    s.add_dft(FREQS)
    return s


def _probe_spectra(sig, dt):
    arr = np.atleast_2d(sig.T).T
    t = np.arange(arr.shape[0]) * dt
    return np.column_stack([dtft(arr[:, j], t, FREQS) for j in range(arr.shape[1])])


def _assert_port_run_bitwise(s_ref, g, probes, dft, psigs, pspecs):
    """Assert bitwise equality (`assert_array_equal`) on state, probes, DFTs, and ports."""
    np.testing.assert_array_equal(g["u"], s_ref.u)
    np.testing.assert_array_equal(g["f"], s_ref.b)
    for i, rec in enumerate(s_ref.recorder.probes):
        sig_ref = s_ref.probe_signal(rec)
        np.testing.assert_array_equal(probes[i], sig_ref)
        np.testing.assert_array_equal(
            _probe_spectra(probes[i], s_ref.dt),
            _probe_spectra(sig_ref, s_ref.dt),
        )
    np.testing.assert_array_equal(dft, s_ref.recorder.dfts[0]["data"])
    for k, p in enumerate(s_ref.ports.ports):
        t_ref, V_ref, Is_ref, I_ref = p.signals()
        t_d, V_d, Is_d, I_d = psigs[k]
        np.testing.assert_array_equal(t_d, t_ref)
        np.testing.assert_array_equal(V_d, V_ref)
        np.testing.assert_array_equal(Is_d, Is_ref)
        np.testing.assert_array_equal(I_d, I_ref)
        Vf_ref, If_ref = p.spectra(FREQS)
        Vf_d, If_d = pspecs[k]
        np.testing.assert_array_equal(Vf_d, Vf_ref)
        np.testing.assert_array_equal(If_d, If_ref)
        Y_ref = p.reference_admittance(FREQS)
        a_ref, b_ref = power_waves(Vf_ref, If_ref, Y_ref)
        a_d, b_d = power_waves(Vf_d, If_d, Y_ref)
        np.testing.assert_array_equal(a_d, a_ref)
        np.testing.assert_array_equal(b_d, b_ref)


@mpi
@pytest.mark.parametrize("kind", ["modal", "lumped", "mixed"])
@pytest.mark.parametrize("mode", ["scatter", "replicated"])
def test_mpi_ports_bitwise_serial(ported_wr90, kind, mode):
    """
    `DistributedFIT` with modal, lumped, and mixed ports reproduces `s.run`
    bitwise (`assert_array_equal`) on final state `(u, b)`, probe signals and
    probe DTFT spectra, running field DFTs, port time records `(t, V, Is, I)`,
    port DTFT spectra `(V(f), I(f))`, and power waves `(a(f), b(f))`.
    """
    comm = MPI.COMM_WORLD
    s = _make_mpi_port_case(ported_wr90, kind)
    d = M.DistributedFIT(
        s if (mode == "replicated" or comm.rank == 0) else None,
        comm,
        partitioner="rcb",
        mode=mode,
    )
    d.run(MPI_NT)
    g = d.gather_state()
    probes = [d.probe_signal(i) for i in range(len(s.recorder.probes))]
    dft = d.dft_data(0)
    psigs = [d.port_signals(k) for k in range(len(s.ports.ports))]
    pspecs = [d.port_spectra(FREQS, k) for k in range(len(s.ports.ports))]
    if comm.rank != 0:
        return
    s.run(MPI_NT)
    _assert_port_run_bitwise(s, g, probes, dft, psigs, pspecs)


@mpi
@pytest.mark.parametrize("kind", ["modal", "lumped", "mixed"])
@pytest.mark.parametrize("mutation", ["drop_port_rank", "swap_port_order"])
def test_mpi_ports_mutation_fails(ported_wr90, kind, mutation):
    """
    Both port-reduction mutations must fail the bitwise check (`assert_array_equal`):
    * `drop_port_rank`: omits the highest rank owning port edges from the
      port inner-product reduction, causing O(1) error in `u` and port spectra.
    * `swap_port_order`: reduces per-rank scalar partial sums `W_own^T u_own`
      in reverse rank order instead of serial free-edge order, shifting the
      IEEE-754 summation order by ~10^-16 to 10^-14 and failing bitwise equality.
    """
    comm = MPI.COMM_WORLD
    s = _make_mpi_port_case(ported_wr90, kind)
    d = M.DistributedFIT(
        s, comm, partitioner="rcb", mode="replicated", _mutation=mutation
    )
    d.run(MPI_NT)
    g = d.gather_state()
    probes = [d.probe_signal(i) for i in range(len(s.recorder.probes))]
    dft = d.dft_data(0)
    psigs = [d.port_signals(k) for k in range(len(s.ports.ports))]
    pspecs = [d.port_spectra(FREQS, k) for k in range(len(s.ports.ports))]
    if comm.rank != 0:
        return
    s.run(MPI_NT)
    with pytest.raises(AssertionError):
        _assert_port_run_bitwise(s, g, probes, dft, psigs, pspecs)
    du = float(np.abs(g["u"] - s.u).max())
    dV = float(
        max(
            np.abs(pspecs[k][0] - p.spectra(FREQS)[0]).max()
            for k, p in enumerate(s.ports.ports)
        )
    )
    assert du > 0.0 and dV > 0.0
    if mutation == "drop_port_rank":
        assert du > 1e-3 * float(np.abs(s.u).max())


@mpi
def test_mpi_ports_transpose_bitwise(ported_wr90):
    """
    `DistributedFIT.apply` and `DistributedFIT.apply_T` with ports reproduce
    `UnstructuredAdjointStepper.linear_step` and `linear_step_T` bitwise
    (`assert_array_equal`) and satisfy `<y, L x> == <L^T y, x>` to machine
    precision.
    """
    comm = MPI.COMM_WORLD
    s = _make_mpi_port_case(ported_wr90, "mixed")
    d = M.DistributedFIT(s, comm, partitioner="rib", mode="replicated")
    st = UnstructuredAdjointStepper(s)
    rng = np.random.default_rng(11)
    x, y = _balanced_states(st, rng)
    Fx = d.apply({"u": x["u"], "f": x["b"]})
    FTy = d.apply_T({"u": y["u"], "f": y["b"]})
    if comm.rank != 0:
        return
    fwd = st.linear_step(x)
    adj = st.linear_step_T(y)
    np.testing.assert_array_equal(Fx["u"], fwd["u"])
    np.testing.assert_array_equal(Fx["f"], fwd["b"])
    np.testing.assert_array_equal(FTy["u"], adj["u"])
    np.testing.assert_array_equal(FTy["f"], adj["b"])
    lhs = float(y["u"] @ Fx["u"] + y["b"] @ Fx["f"])
    rhs = float(x["u"] @ FTy["u"] + x["b"] @ FTy["f"])
    assert abs(lhs - rhs) / (abs(lhs) + abs(rhs)) <= DOT_TOL


def _eval_port_case_metrics(ported_wr90, kind, n_parts, comm=None):
    """Evaluate bitwise and mutation metrics for `(kind, n_parts)` via `DistributedFIT` or `LocalCluster`."""
    use_mpi = comm is not None and comm.Get_size() == n_parts
    rank = comm.Get_rank() if use_mpi else 0

    s_ok = _make_mpi_port_case(ported_wr90, kind)
    if use_mpi:
        d_ok = M.DistributedFIT(s_ok, comm, partitioner="rcb", mode="replicated")
        d_ok.run(MPI_NT)
        g_ok = d_ok.gather_state()
        probe0_ok = d_ok.probe_signal(0)
        Vspec_ok = [
            d_ok.port_spectra(FREQS, k)[0] for k in range(len(s_ok.ports.ports))
        ]
    else:
        cl_ok = M.LocalCluster(s_ok, n_parts=n_parts, partitioner="rcb")
        cl_ok.run(MPI_NT)
        g_ok = cl_ok.gather_state()
        probe0_ok = cl_ok.probe_signal(0)
        Vspec_ok = [
            cl_ok.port_spectra(FREQS, k)[0] for k in range(len(s_ok.ports.ports))
        ]

    mut_runs = {}
    for mut in ("drop_port_rank", "swap_port_order"):
        s_m = _make_mpi_port_case(ported_wr90, kind)
        if use_mpi:
            d_m = M.DistributedFIT(
                s_m, comm, partitioner="rcb", mode="replicated", _mutation=mut
            )
            d_m.run(MPI_NT)
            g_m = d_m.gather_state()
            probe0_m = d_m.probe_signal(0)
            Vspec_m = [
                d_m.port_spectra(FREQS, k)[0] for k in range(len(s_m.ports.ports))
            ]
        else:
            cl_m = M.LocalCluster(
                s_m, n_parts=n_parts, partitioner="rcb", _mutation=mut
            )
            cl_m.run(MPI_NT)
            g_m = cl_m.gather_state()
            probe0_m = cl_m.probe_signal(0)
            Vspec_m = [
                cl_m.port_spectra(FREQS, k)[0] for k in range(len(s_m.ports.ports))
            ]
        mut_runs[mut] = (g_m, probe0_m, Vspec_m)

    if rank != 0:
        return None

    s_ref = _make_mpi_port_case(ported_wr90, kind)
    s_ref.run(MPI_NT)
    u_norm = float(np.abs(s_ref.u).max())
    b_norm = float(np.abs(s_ref.b).max())
    probe0_ref = s_ref.probe_signal(s_ref.recorder.probes[0])
    pspec0_ref = _probe_spectra(probe0_ref, s_ref.dt)
    Vspec_ref = [p.spectra(FREQS)[0] for p in s_ref.ports.ports]
    pspec0_ok = _probe_spectra(probe0_ok, s_ref.dt)

    cl_part = M.LocalCluster(
        _make_mpi_port_case(ported_wr90, kind), n_parts=n_parts, partitioner="rcb"
    )
    port_edge_counts = [
        [int(r.ports["W_own"].getcol(k).nnz) for r in cl_part.ranks]
        for k in range(len(s_ref.ports.ports))
    ]

    rec = {
        "n_free_edges": int(s_ref.u.size),
        "n_massless_edges": int(s_ref.massless.size),
        "port_edge_counts_per_rank": port_edge_counts,
        "u_max": u_norm,
        "b_max": b_norm,
        "V0_spec_max": float(np.abs(Vspec_ref[0]).max()),
        "probe0_spec_max": float(np.abs(pspec0_ref).max()),
        "bitwise_max_du": float(np.abs(g_ok["u"] - s_ref.u).max()),
        "bitwise_max_db": float(np.abs(g_ok["f"] - s_ref.b).max()),
        "bitwise_max_dprobe_spec": float(np.abs(pspec0_ok - pspec0_ref).max()),
        "bitwise_max_dV_spec": float(
            max(np.abs(Vspec_ok[k] - Vspec_ref[k]).max() for k in range(len(Vspec_ref)))
        ),
        "mutations": {},
    }
    for mut, (g_m, probe0_m, Vspec_m) in mut_runs.items():
        pspec0_m = _probe_spectra(probe0_m, s_ref.dt)
        rec["mutations"][mut] = {
            "max_du": float(np.abs(g_m["u"] - s_ref.u).max()),
            "rel_du": float(np.abs(g_m["u"] - s_ref.u).max() / u_norm),
            "max_db": float(np.abs(g_m["f"] - s_ref.b).max()),
            "max_dprobe_spec": float(np.abs(pspec0_m - pspec0_ref).max()),
            "max_dV_spec": float(
                max(
                    np.abs(Vspec_m[k] - Vspec_ref[k]).max()
                    for k in range(len(Vspec_ref))
                )
            ),
        }
    return rec


def compute_mpi_ports_summary(
    ported_wr90=None, out_path="out/mpi_ports.json", comm=None
):
    """
    Compute and write `out/mpi_ports.json` with deterministic verification and
    mutation metrics for `n_parts in (2, 4)` across `modal`, `lumped`, and
    `mixed` port cases.
    """
    if ported_wr90 is None:
        ported_wr90 = _build_wr90_port_factories()
    rank = comm.Get_rank() if comm is not None else 0
    summary = {
        "nt": MPI_NT,
        "freqs_hz": [float(f) for f in FREQS],
        "cases": {},
    }
    for n_parts in (2, 4):
        p_key = f"ranks_{n_parts}"
        if rank == 0:
            summary["cases"][p_key] = {}
        for kind in ("modal", "lumped", "mixed"):
            if comm is not None and comm.Get_size() == n_parts:
                rec = _eval_port_case_metrics(ported_wr90, kind, n_parts, comm=comm)
            elif rank == 0:
                rec = _eval_port_case_metrics(ported_wr90, kind, n_parts, comm=None)
            else:
                rec = None
            if rank == 0:
                summary["cases"][p_key][kind] = rec

    if rank == 0 and out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
    return summary if rank == 0 else None


@mpi
def test_mpi_ports_write_json_summary(ported_wr90):
    """Write and print `out/mpi_ports.json` during the MPI verify step."""
    comm = MPI.COMM_WORLD
    summary = compute_mpi_ports_summary(
        ported_wr90, out_path="out/mpi_ports.json", comm=comm
    )
    if comm.rank == 0:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    print(json.dumps(compute_mpi_ports_summary(), indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# GPU port path (`wakis.unstructured.gpu.DeviceStepper` with `PortSet`).
# Numbers are printed and written to out/gpu_ports.json by running this file.
# ---------------------------------------------------------------------------

from wakis.unstructured import gpu as G

try:
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()
    HAVE_GPU = True
except Exception:
    cp, HAVE_GPU = None, False

N_GPU_STEPS = 120


def _make_ported_gpu_case(ported_wr90, kind="mixed"):
    _, make_modal, make_mixed, _, z = ported_wr90
    s = make_modal() if kind == "modal" else make_mixed()
    s.add_probe([[A_WG * 0.45, B_WG * 0.5, 0.5 * (z[2] + z[3])]], 1)
    s.add_dft(FREQS)
    return s


def _compare_ported_device(s, dev, n_steps, bound=None):
    bound = bound or G.agreement_bound(s, n_steps)
    dev.run(n_steps)
    s.run(n_steps)
    W = bound["weights"]
    st = dev.get_state()
    d = float(
        np.sqrt(
            np.sum(W["u"] * (st["u"] - s.u) ** 2)
            + np.sum(W["b"] * (st["b"] - s.b) ** 2)
        )
    )
    pr_ratio = float(
        np.max(
            np.abs(dev.probe_signal(0) - s.probe_signal(s.recorder.probes[0]))
            / np.maximum(bound["probe"][0], 1e-300)
        )
    )
    D = dev.dft_data(0) - s.recorder.dfts[0]["data"]
    dft_ratio = float(
        np.max(np.sqrt((W["u"] * np.abs(D) ** 2).sum(1)) / bound["dft"][0])
    )
    V_dev, Is_dev = dev.port_signals()
    V_cpu = np.column_stack([p.V for p in s.ports.ports])
    Is_cpu = np.column_stack([p.Is for p in s.ports.ports])
    port_d_max = float(np.max(np.abs(V_dev - V_cpu)))
    port_ratio = float(
        np.max(np.abs(V_dev - V_cpu) / np.maximum(bound["port"], 1e-300))
    )
    return dict(
        d=d,
        eps=float(bound["eps"][-1]),
        norm=float(bound["norm"].max()),
        probe_ratio=pr_ratio,
        dft_ratio=dft_ratio,
        port_d_max=port_d_max,
        port_ratio=port_ratio,
        Is_equal=bool(np.array_equal(Is_dev, Is_cpu)),
        bound=bound,
    )


@pytest.mark.skipif(not HAVE_GPU, reason="needs CuPy and a GPU")
@pytest.mark.parametrize("kind", ["modal", "mixed"])
def test_gpu_ports_match_cpu_within_roundoff_bound(ported_wr90, kind):
    """
    `DeviceStepper` on a ported 3D tetrahedral solver (TE10 modal ports and
    interior 50 Ohm lumped port, with massless edges, probe and DFT) agrees
    with CPU `UnstructuredFIT3D.run` within the derived round-off bound of
    `agreement_bound` for state, probe, DFT, and recorded half-step port
    voltages V^{n+1/2}, and `to_host()` reproduces `_Port.spectra` and
    Kurokawa power-wave S-parameters to round-off.
    """
    s = _make_ported_gpu_case(ported_wr90, kind)
    bound = G.agreement_bound(s, N_GPU_STEPS)
    dev = G.DeviceStepper(s, xp=cp)
    r = _compare_ported_device(s, dev, N_GPU_STEPS, bound=bound)
    print(
        f"gpu_ports {kind}: ||d||/||x||={r['d'] / r['norm']:.3e}, "
        f"eps/||x||={r['eps'] / r['norm']:.3e}, d/eps={r['d'] / r['eps']:.3e}, "
        f"port_ratio={r['port_ratio']:.3e}, K={bound['K']:.2f}, m={bound['depth']}"
    )
    assert r["norm"] > 0 and np.isfinite(r["d"])
    assert r["eps"] / r["norm"] < 1e-7, "bound is vacuous"
    assert r["d"] <= r["eps"], f"||d|| = {r['d']:.3e} > eps = {r['eps']:.3e}"
    assert r["probe_ratio"] <= 1.0
    assert r["dft_ratio"] <= 1.0
    assert r["port_ratio"] <= 1.0
    assert r["Is_equal"]

    # to_host() transfers port V and I_s so _Port.spectra and power_waves work directly
    s_gpu = _make_ported_gpu_case(ported_wr90, kind)
    dev_h = G.DeviceStepper(s_gpu, xp=cp)
    dev_h.run(N_GPU_STEPS)
    dev_h.to_host()
    V_gpu = np.column_stack([p.spectra(FREQS)[0] for p in s_gpu.ports.ports])
    I_gpu = np.column_stack([p.spectra(FREQS)[1] for p in s_gpu.ports.ports])
    Y_ref = np.column_stack([p.reference_admittance(FREQS) for p in s_gpu.ports.ports])
    a_gpu, b_gpu = power_waves(V_gpu, I_gpu, Y_ref)
    S_gpu = b_gpu / a_gpu[:, [0]]

    V_cpu = np.column_stack([p.spectra(FREQS)[0] for p in s.ports.ports])
    I_cpu = np.column_stack([p.spectra(FREQS)[1] for p in s.ports.ports])
    a_cpu, b_cpu = power_waves(V_cpu, I_cpu, Y_ref)
    S_cpu = b_cpu / a_cpu[:, [0]]
    np.testing.assert_allclose(
        S_gpu, S_cpu, rtol=1e-13, atol=1e-14 * np.abs(S_cpu).max()
    )


@pytest.mark.skipif(not HAVE_GPU, reason="needs CuPy and a GPU")
def test_gpu_ports_mutations_exceed_bound(ported_wr90):
    """
    Mutating the device port update (dropping the port step on device, dropping
    the Woodbury post-step, zeroing the explicit half-step conductance, scaling
    S^-1 by 1 + 1e-3, or flipping the conductance sign) must exceed the derived
    round-off bound eps_N by more than 100x.
    """
    s_ref = _make_ported_gpu_case(ported_wr90, "mixed")
    bound = G.agreement_bound(s_ref, N_GPU_STEPS)
    s_ref.run(N_GPU_STEPS)
    W = bound["weights"]
    eps_N = float(bound["eps"][-1])

    for mut in (
        "drop_port_step",
        "drop_port_post",
        "zero_port_pre_g",
        "scale_port_sinv_1e-3",
        "flip_port_pre_conductance",
    ):
        s_m = _make_ported_gpu_case(ported_wr90, "mixed")
        dev = G.DeviceStepper(s_m, xp=cp)
        if mut == "drop_port_step":
            dev.ports = None
        elif mut == "drop_port_post":
            dev._port_post = lambda u: None
        elif mut == "zero_port_pre_g":
            dev.ports["gh"] = dev.ports["gh"] * 0.0
        elif mut == "scale_port_sinv_1e-3":
            dev.ports["Sinv"] = dev.ports["Sinv"] * (1.0 + 1e-3)
        elif mut == "flip_port_pre_conductance":
            dev.ports["gh"] = -dev.ports["gh"]
        dev.run(N_GPU_STEPS)
        st = dev.get_state()
        d = float(
            np.sqrt(
                np.sum(W["u"] * (st["u"] - s_ref.u) ** 2)
                + np.sum(W["b"] * (st["b"] - s_ref.b) ** 2)
            )
        )
        print(f"gpu port mutation {mut}: d/eps = {d / eps_N:.3e}")
        assert not np.isfinite(d) or d > 100.0 * eps_N


@pytest.mark.skipif(not HAVE_GPU, reason="needs CuPy and a GPU")
def test_gpu_ports_step_does_not_synchronize(ported_wr90, monkeypatch):
    """
    `DeviceStepper.step` with tetrahedral modal + lumped ports enqueues kernels
    without synchronizing the CUDA stream (a 100 ms device spin kernel is still
    running when 3 ported steps return), whereas a device-to-host read inside
    `_port_post` is caught by the detector.
    """
    spin = cp.RawKernel(
        r"""
extern "C" __global__ void _spin_port(long long ns) {
    long long t0, t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0)); t = t0;
    while (t - t0 < ns) { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); }
}""",
        "_spin_port",
    )
    s = _make_ported_gpu_case(ported_wr90, "mixed")
    dev = G.DeviceStepper(s, xp=cp)
    dev.run(5)
    dev.prepare(50)
    for pr in dev.probes:
        dev._reserve(pr, pr["k"] + 50)
    cp.cuda.Stream.null.synchronize()

    def enqueue(n):
        ev = cp.cuda.Event()
        spin((1,), (1,), (np.int64(100_000_000),))
        ev.record()
        for _ in range(n):
            dev.step()
        done = ev.done
        cp.cuda.Stream.null.synchronize()
        return bool(done)

    assert not enqueue(3), "DeviceStepper.step with ports synchronized the stream"
    orig = G.DeviceStepper._port_post

    def bad_post(self, u):
        float(u[0])
        return orig(self, u)

    monkeypatch.setattr(G.DeviceStepper, "_port_post", bad_post)
    assert enqueue(1), "the sync detector missed a device-to-host read in _port_post"


def _export_gpu_ports_json():
    """Run the GPU port benchmarks and mutations, write out/gpu_ports.json and print it."""
    import json

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
    kc = float(kc[0])
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

    fx = (m, make_modal, make_mixed, np.array([], int), z)
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()

    cases_out = {}
    for kind in ("modal", "mixed"):
        s_info = _make_ported_gpu_case(fx, kind)
        ps = s_info.ports
        WT_nnz = G._max_row_nnz(ps.W.T)
        W_nnz = G._max_row_nnz(ps.W)
        np_ = len(ps.ports)
        port_depth_delta = 2 * WT_nnz + 2 * W_nnz + np_ + 6
        bound = G.agreement_bound(s_info, N_GPU_STEPS)

        # Subspace empirical amplification (b0 in ker(C_s^T M_nu))
        ke_sub = 1.0
        for seed in range(4):
            st_e = G.DeviceStepper(
                _make_ported_gpu_case(fx, kind), xp=np, sources=False
            )
            W_e = st_e.weights()
            rng = np.random.default_rng(seed)
            u0 = rng.standard_normal(W_e["u"].shape) / np.sqrt(W_e["u"])
            b0 = rng.standard_normal(W_e["b"].shape) / np.sqrt(
                np.where(W_e["b"] > 0, W_e["b"], np.inf)
            )
            b0 -= s_info.Cs @ s_info._Kss.solve(s_info.Cs.T @ (s_info.Mnu @ b0))
            st_e.set_state({"u": u0, "b": b0})
            n0 = G._wnorm(st_e.get_state(), W_e, ["u", "b"])
            for _ in range(N_GPU_STEPS):
                st_e.step()
                ke_sub = max(ke_sub, G._wnorm(st_e.get_state(), W_e, ["u", "b"]) / n0)

        backends_out = {}
        for bname, xp_mod in (("cupy", cp), ("numpy", np)):
            s_run = _make_ported_gpu_case(fx, kind)
            dev = G.DeviceStepper(s_run, xp=xp_mod)
            r = _compare_ported_device(s_run, dev, N_GPU_STEPS, bound=bound)

            s_th = _make_ported_gpu_case(fx, kind)
            dev_th = G.DeviceStepper(s_th, xp=xp_mod)
            dev_th.run(N_GPU_STEPS)
            dev_th.to_host()
            V_dev = np.column_stack([p.spectra(FREQS)[0] for p in s_th.ports.ports])
            I_dev = np.column_stack([p.spectra(FREQS)[1] for p in s_th.ports.ports])
            Y_ref = np.column_stack(
                [p.reference_admittance(FREQS) for p in s_th.ports.ports]
            )
            a_dev, b_dev = power_waves(V_dev, I_dev, Y_ref)
            S_dev = b_dev / a_dev[:, [0]]

            V_cpu = np.column_stack([p.spectra(FREQS)[0] for p in s_run.ports.ports])
            I_cpu = np.column_stack([p.spectra(FREQS)[1] for p in s_run.ports.ports])
            a_cpu, b_cpu = power_waves(V_cpu, I_cpu, Y_ref)
            S_cpu = b_cpu / a_cpu[:, [0]]
            s_rel_err = float(np.max(np.abs(S_dev - S_cpu)) / np.max(np.abs(S_cpu)))

            backends_out[bname] = {
                "d_W": r["d"],
                "eps_N": r["eps"],
                "norm_max": r["norm"],
                "d_over_norm": r["d"] / r["norm"],
                "eps_over_norm": r["eps"] / r["norm"],
                "d_over_eps": r["d"] / r["eps"],
                "probe_ratio_max": r["probe_ratio"],
                "dft_ratio_max": r["dft_ratio"],
                "port_V_diff_max": r["port_d_max"],
                "port_V_bound_max": float(bound["port"].max()),
                "port_ratio_max": r["port_ratio"],
                "s_param_rel_err": s_rel_err,
                "Is_equal": r["Is_equal"],
            }

        dev_meta = G.DeviceStepper(_make_ported_gpu_case(fx, kind), xp=cp)
        cases_out[kind] = {
            "n_ports": np_,
            "n_port_edges": int(dev_meta.ports["rows"].size),
            "max_row_nnz_WT": WT_nnz,
            "max_row_nnz_W": W_nnz,
            "port_depth_delta": port_depth_delta,
            "total_depth": int(bound["depth"]),
            "gamma_m": float(bound["gamma"]),
            "K_theory": float(bound["K"]),
            "K_empirical_subspace": float(ke_sub),
            "kappa_massless": float(dev_meta.kappa_massless),
            "massless_block": int(dev_meta.massless_block),
            "backends": backends_out,
        }

    # Mutations on mixed case
    s_ref = _make_ported_gpu_case(fx, "mixed")
    bound_mix = G.agreement_bound(s_ref, N_GPU_STEPS)
    s_ref.run(N_GPU_STEPS)
    W_mix = bound_mix["weights"]
    eps_mix = float(bound_mix["eps"][-1])
    mutations_out = {}
    for mut in (
        "drop_port_step",
        "drop_port_post",
        "zero_port_pre_g",
        "scale_port_sinv_1e-3",
        "flip_port_pre_conductance",
    ):
        s_m = _make_ported_gpu_case(fx, "mixed")
        dev = G.DeviceStepper(s_m, xp=cp)
        if mut == "drop_port_step":
            dev.ports = None
        elif mut == "drop_port_post":
            dev._port_post = lambda u: None
        elif mut == "zero_port_pre_g":
            dev.ports["gh"] = dev.ports["gh"] * 0.0
        elif mut == "scale_port_sinv_1e-3":
            dev.ports["Sinv"] = dev.ports["Sinv"] * (1.0 + 1e-3)
        elif mut == "flip_port_pre_conductance":
            dev.ports["gh"] = -dev.ports["gh"]
        dev.run(N_GPU_STEPS)
        st = dev.get_state()
        d_m = float(
            np.sqrt(
                np.sum(W_mix["u"] * (st["u"] - s_ref.u) ** 2)
                + np.sum(W_mix["b"] * (st["b"] - s_ref.b) ** 2)
            )
        )
        mutations_out[mut] = {
            "d_W": d_m,
            "d_over_eps": d_m / eps_mix,
        }

    out = {
        "gpu_name": gpu_name,
        "mesh": {
            "a_m": A_WG,
            "b_m": B_WG,
            "h_m": h,
            "length_m": float(z[-1]),
            "n_layers": int(z.size - 1),
            "n_cells": int(m.n_cells),
            "n_faces": int(m.n_faces),
            "n_free_edges": int(s0.free.size),
            "n_massless_edges": int(s0.massless.size),
            "dt_s": float(dt),
            "courant": float(s0.courant),
            "n_steps": N_GPU_STEPS,
            "freqs_hz": [float(f) for f in FREQS],
        },
        "cases": cases_out,
        "mutations": mutations_out,
    }
    out_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "out", "gpu_ports.json")
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _export_gpu_ports_json()
