"""
Exact discrete adjoint of `UnstructuredFIT3D` with the spherical CFS-PML.

The PML step is x' = S4 S3 S2 S1 x + B j on the extended state
x = [u, b, z] (z: PML dual fluxes d and the CFS filter memories). The
adjoint stepper rebuilds S3 and B (the only stages that depend on the
permittivity, through a1 / den of the non-PML edges) for every design and
applies S1^T S2^T S3^T S4^T in reverse.

What is checked, and how each check can fail:

* forward replay: `UnstructuredAdjointStepper.step` equals the solver's own
  `one_step` (`PML3D.apply`) bitwise in u, b and z (same matrices, same
  operation order); a stale S3 (design ignored) fails the gradient test.
* blockwise dot product over (u, b, z) x (u, b, z): both sides sum the same
  products in a different order, so the scale-invariant error is bounded by
  gamma_n ~ n u for n = largest block (~1e5 here, 1.1e-11) and is ~sqrt(n) u
  (~4e-14) for random rounding; threshold sqrt(n) u. Mutations: the
  transposed stages applied in forward order; the z adjoint dropped.
* gradient vs central finite differences: |g - FD(h)| <= 1.5 x the O(h^2)
  truncation estimate |FD(h) - FD(2h)| / 3 plus a round-off term (as
  test_100). Nt = 200 steps lets the adjoint wave cross the PML and return.
  Mutation: the PML adjoint state z zeroed each reverse step (the layer
  treated as a passive constant) gives O(1) gradient errors.
* checkpointed gradient bitwise equal to the stored-row gradient.
* design cells touching a PML edge are refused.
"""

import numpy as np
import pytest
from scipy.constants import c

from wakis.unstructured import UnstructuredFIT3D
from wakis.unstructured.adjoint import (
    CellDFTMonitor,
    CellMaterialGrid,
    UnstructuredAdjointSimulation,
    UnstructuredAdjointStepper,
)
from wakis.unstructured.pml3d import ball_with_pml_mesh

LAM0 = 3.0  # centre wavelength [m]; h = 0.5 m -> 6 cells per wavelength (fast test)
TAU = LAM0 / (np.pi * c)
T0 = 4 * TAU
NT = 200
U = np.finfo(float).eps / 2


def amp(t):
    x = (t - T0) / TAU
    return -x * np.exp(-(x**2))


@pytest.fixture(scope="module")
def setup():
    m, cfg = ball_with_pml_mesh(2.0, 0.5, 4)
    pml = dict(cfg, kappa_max=2.0, alpha=1e8)
    dt = UnstructuredFIT3D(m, bc="pec", pml=pml).dt

    def make():
        s = UnstructuredFIT3D(m, bc="pec", pml=pml, dt=dt)
        s.add_point_dipole([-1.1, 0.1, 0.05], [0, 0, 1], amp)
        return s

    cent = m.nodes[m.cells].mean(1)
    cells = np.flatnonzero(np.linalg.norm(cent - [0.3, 0, 0], axis=1) < 0.6)
    mon = CellDFTMonitor([c / LAM0], points=[[1.3, 0.05, 0.0]], component=2)
    return m, make, cells, mon


def objective(out):
    X = out["m"]["dft"]
    return float(np.sum(np.abs(X) ** 2)), {"m": {"dft": 2 * X}}


def _sim(setup, memory="store", **kw):
    m, make, cells, mon = setup
    design = CellMaterialGrid(m, cells, 1.0, 3.0)
    return UnstructuredAdjointSimulation(make, design, {"m": mon}, NT, memory=memory, **kw)


def blockwise_dot(st, seed=0):
    rng = np.random.default_rng(seed)
    keys = [k for k, _ in st.keys()]
    zero = st.zero_state()
    fwd, adj = {}, {}
    for k in keys:
        x = {kk: v.copy() for kk, v in zero.items()}
        x[k] = rng.standard_normal(x[k].size)
        fwd[k] = (x, st.step(x, 0, sources=False)[0])
        y = {kk: v.copy() for kk, v in zero.items()}
        y[k] = rng.standard_normal(y[k].size)
        adj[k] = (y, st.step_T(y)[0])
    worst = 0.0
    for i in keys:
        x, Ax = fwd[i]
        for j in keys:
            y, ATy = adj[j]
            a, b = float(Ax[j] @ y[j]), float(x[i] @ ATy[i])
            scale = np.linalg.norm(Ax[j]) * np.linalg.norm(y[j]) + np.linalg.norm(
                x[i]
            ) * np.linalg.norm(ATy[i])
            if scale > 0:
                worst = max(worst, abs(a - b) / scale)
    return worst


def _dot_tol(st):
    return np.sqrt(max(n for _, n in st.keys())) * U


def test_forward_replay_bitwise(setup):
    _, make, cells, _ = setup
    s = make()
    st = UnstructuredAdjointStepper(s, cells)
    assert st.pml3 is not None and s.massless.size > 0
    x = st.zero_state()
    for n in range(80):
        s.one_step()
        x, _ = st.step(x, n)
    p = s._pml
    nu, nf = s.free.size, s.mesh.n_faces
    assert np.abs(x["z"]).max() > 0  # the field has entered the layer
    np.testing.assert_array_equal(x["u"], s.u)
    np.testing.assert_array_equal(x["b"], s.b)
    np.testing.assert_array_equal(x["z"], p.x[nu + nf :])


def test_dot_product_including_pml_state(setup):
    _, make, cells, _ = setup
    st = UnstructuredAdjointStepper(make(), cells)
    err = blockwise_dot(st)
    print("blockwise dot error", err, "tol", _dot_tol(st))
    assert err < _dot_tol(st)


def _forward_order_T(self, xb1, xb):
    T1, T2, T3, T4 = self._pstagesT
    le = np.concatenate([xb1["u"], xb1["b"], xb1["z"]])
    z = T4 @ (T3 @ (T2 @ (T1 @ le)))
    nu, nf = xb1["u"].size, xb1["b"].size
    xb["u"], xb["b"], xb["z"] = z[:nu], z[nu : nu + nf], z[nu + nf :]
    return z[:nu].copy()


_orig_T = UnstructuredAdjointStepper._step3d_pml_T


def _drop_z(self, xb1, xb):
    out = _orig_T(self, xb1, xb)
    xb["z"] = 0 * xb["z"]
    return out


@pytest.mark.parametrize("func", [_forward_order_T, _drop_z])
def test_dot_product_mutations_fail(setup, func, monkeypatch):
    _, make, cells, _ = setup
    st = UnstructuredAdjointStepper(make(), cells)
    monkeypatch.setattr(UnstructuredAdjointStepper, "_step3d_pml_T", func)
    err = blockwise_dot(st)
    print(func.__name__, err)
    assert err > 1e-6


def fd_check(sim, rho, g, cells, h=1e-3):
    F0_ = objective(sim.forward(rho, record=False))[0]
    out = []
    for k in cells:
        fd = []
        for hh in (h, 2 * h):
            rp, rm = rho.copy(), rho.copy()
            rp[k] += hh
            rm[k] -= hh
            fd.append(
                (objective(sim.forward(rp, record=False))[0]
                 - objective(sim.forward(rm, record=False))[0]) / (2 * hh)
            )  # fmt: skip
        trunc = abs(fd[1] - fd[0]) / 3
        roundoff = 1e3 * np.finfo(float).eps * abs(F0_) / h
        out.append((g[k], fd[0], trunc, roundoff))
    return out


def test_gradient_vs_finite_differences(setup, monkeypatch):
    sim = _sim(setup)
    rng = np.random.default_rng(1)
    rho = rng.uniform(0.2, 0.8, sim.design.shape)
    _, g = sim.value_and_grad(rho, objective)
    ks = rng.choice(rho.size, 4, replace=False)
    good = fd_check(sim, rho, g, ks)
    for adj, fd, trunc, roundoff in good:
        print("adjoint", adj, "FD", fd, "rel", abs(adj - fd) / abs(fd), "trunc", trunc / abs(fd))
        assert abs(adj - fd) <= 1.5 * trunc + roundoff
        assert abs(adj - fd) < 1e-5 * abs(fd)
    # Mutation: the PML adjoint memories dropped (the layer frozen in reverse)
    monkeypatch.setattr(UnstructuredAdjointStepper, "_step3d_pml_T", _drop_z)
    _, g_bad = sim.value_and_grad(rho, objective)
    errs = [abs(g_bad[k] - fd) / abs(fd) for k, (_, fd, _, _) in zip(ks, good)]
    print("drop z: relative errors", errs)
    assert max(errs) > 1e-2


def test_stale_ampere_stage_fails(setup, monkeypatch):
    """Mutation: S3 not rebuilt for the design (frozen at the base eps)."""
    sim = _sim(setup)
    rho = np.full(sim.design.shape, 0.5)
    F_good = objective(sim.forward(rho, record=False))[0]
    st = sim.stepper
    orig = st.set_eps

    def stale(eps):
        orig(eps)
        p = st.pml3
        st._pstages = p.stages
        st._pstagesT = tuple(S.T.tocsr() for S in p.stages)

    monkeypatch.setattr(st, "set_eps", stale)
    F_bad = objective(sim.forward(rho, record=False))[0]
    print("objective good / stale", F_good, F_bad)
    assert abs(F_bad - F_good) > 1e-3 * F_good


def test_checkpoint_gradient_bitwise(setup):
    rho = np.random.default_rng(3).uniform(0.2, 0.8, _sim(setup).design.shape)
    sim = _sim(setup)
    _, g = sim.value_and_grad(rho, objective)
    for slots, block in ((2, None), (3, 7)):
        ck = _sim(setup, memory="checkpoint", checkpoints=slots, block=block)
        ck.verify_replay = True
        _, g_ck = ck.value_and_grad(rho, objective)
        np.testing.assert_array_equal(g_ck, g)
        assert ck.memory_stats["recompute_steps"] > 0
        assert ck._rung.mismatches == 0


def test_dft_matches_solver_recorder(setup):
    sim = _sim(setup)
    out = sim.forward(np.zeros(sim.design.shape), record=False)["m"]["dft"]
    s = sim.make_solver()
    rec = s.add_dft(sim.monitors["m"].freqs)
    s.run(sim.Nt)
    ref = (sim._W["m"] @ rec["data"].T).T
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


def test_design_in_pml_refused(setup):
    m, make, _, _ = setup
    s = make()
    shell = np.arange(m.n_cells)[np.linalg.norm(m.nodes[m.cells].mean(1), axis=1) > 2.2]
    with pytest.raises(ValueError, match="PML"):
        UnstructuredAdjointStepper(s, shell[:5])
