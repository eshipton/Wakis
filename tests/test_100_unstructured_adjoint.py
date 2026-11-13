"""
Exact discrete adjoint of the unstructured FIT solvers (wakis.unstructured.adjoint).

What is checked, and how each check can fail:

* forward equivalence: `UnstructuredAdjointStepper.step` reproduces
  `solver.one_step` bitwise (array_equal) for every covered feature, and the
  simulation DFT equals the solver's `Recorder` DFT to round-off.
* blockwise dot-product test: for every pair (input block i, output block j)
  of the state, <A_ji x_i, y_j> = <x_i, A_ji^T y_j> with the scale-invariant
  error |a - b| / (|A_ji x_i| |y_j| + |x_i| |A_ji^T y_j|). (A single global
  inner product is dominated by the largest coefficients and can hide a
  wrong O(1) block.) Round-off bound: both sides sum the same products in a
  different order, error <~ sqrt(n) eps_mach ~ 1e-14 for n <~ 8000 dofs;
  measured max 2.8e-17 over all 22 configurations; threshold 1e-14.
  Mutations (a transposed ADE/PML term dropped, the untransposed LTS
  polynomial, the Schur complement dropped, unsigned or replaced by its
  diagonal, the Whitney M_1 solve skipped) give 7e-6 ... 0.2 (> 1e-6).
* gradients vs central finite differences on several cells: the error must
  be below 1.5x the truncation-error estimate |FD(h) - FD(2h)| / 3 of the
  O(h^2) central difference (plus a round-off term). Wrong dual-area maps
  (barycentric instead of Voronoi; 1/eps instead of 1/eps^2 in TE) fail.
* checkpointing: gradients bitwise equal to the stored-row run, with a
  nonzero replay count and zero replayed-row mismatches.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.constants import c

from wakis.dispersive import Debye, DispersiveMaterial, Lorentz
from wakis.unstructured import (
    UnstructuredFIT2D,
    UnstructuredFIT3D,
    ball_mesh,
    disk_mesh,
    graded_disk_mesh,
)
from wakis.unstructured import adjoint as A
from wakis.unstructured.adjoint import (
    CellDFTMonitor,
    CellMaterialGrid,
    UnstructuredAdjointSimulation,
    UnstructuredAdjointStepper,
)

F0 = c / 1.0  # 1 m vacuum wavelength
DOT_TOL = 1e-14
MUT_MIN = 1e-6


def _mat():
    w0 = 2 * np.pi * F0
    return DispersiveMaterial(2.0, [Lorentz(1.0, w0, 0.1 * w0), Debye(0.5, 1 / w0)])


def pulse(t, t0=2.5e-9, w=0.7e-9):
    return np.exp(-(((t - t0) / w) ** 2))


# ---------------------------------------------------------------- meshes
@pytest.fixture(scope="module")
def disk():
    """
    Jittered disk mesh. The hexagonal lattice of `disk_mesh` makes nearly all
    triangles equilateral, where the Voronoi partial dual area of each vertex
    equals the barycentric A_T / 3: a wrong dual-area map would go unnoticed.
    """
    from scipy.spatial import Delaunay

    from wakis.unstructured import SimplexMesh

    h = 0.08
    base = disk_mesh([0.35, 1.0], h)
    X = base.nodes.copy()
    r = np.hypot(*X.T)
    lattice = (np.abs(r - 0.35) > 0.3 * h) & (r < 1.0 - 0.3 * h)
    X[lattice] += 0.2 * h * np.random.default_rng(7).uniform(-1, 1, (lattice.sum(), 2))
    tri = Delaunay(X).simplices
    rc = np.hypot(*X[tri].mean(1).T)
    return SimplexMesh(X, tri, (rc > 0.35).astype(int))


@pytest.fixture(scope="module")
def pml_mesh():
    return graded_disk_mesh([0.35, 1.0], 0.08, pml_layers=6, pml_dr=0.06)


@pytest.fixture(scope="module")
def lts_mesh():
    def size(r):
        d = r - 0.5 - 0.16
        return 0.08 if d <= 0 else min(0.16, 0.08 * 1.2 ** (d / 0.08))

    return graded_disk_mesh([0.5, 1.0], size)


@pytest.fixture(scope="module")
def ball():
    return ball_mesh([0.5, 1.0], 0.25)


def _annulus(mesh, r0, r1):
    cen = mesh.nodes[mesh.cells].mean(1)
    r = np.linalg.norm(cen, axis=1)
    sel = (r > r0) & (r < r1)
    if mesh.dim == 3:
        bed = np.zeros(mesh.n_edges, bool)
        bed[mesh.boundary_edges] = True
        sel &= ~bed[mesh.cell_edges].any(1)
    return np.flatnonzero(sel)


# ---------------------------------------------------------------- configurations
def _config(name, meshes):
    """(make_solver(dt=None), design cells, monitor) of a named configuration."""
    disk, pml_mesh, lts_mesh, ball = meshes
    mat = _mat()
    if name.startswith("3D"):
        hodge = "whitney" if "whitney" in name else "voronoi"
        bc = "pec" if ("pec" in name or hodge == "whitney") else "abc"
        mats = {0: mat if "disp" in name else [2.0, 1.0]}

        def make(dt=None):
            s = UnstructuredFIT3D(ball, mats, bc=bc, hodge=hodge, dt=dt)
            s.add_point_dipole([0.6, 0.1, 0.0], [0.3, 0.2, 1.0], pulse)
            return s

        mon = CellDFTMonitor([0.8 * F0, F0], points=[[-0.5, 0.2, 0.1], [0.3, -0.4, 0.2]],
                             component=2)  # fmt: skip
        return make, _annulus(ball, 0.5, 0.75), mon, ball
    pol = "TE" if name.startswith("TE") else "TM"
    kw, mesh, mats = {}, disk, {0: mat if "disp" in name else [2.0, 1.0]}
    bc = next((b for b in ("pec", "pmc") if b in name), "abc")
    src = [0.7, 0.1]
    design = (0.38, 0.6)
    if "absorber" in name:
        kw["absorber"] = dict(r0=0.75, r1=1.0, s_max=3e9)
    if "pml" in name:
        mesh, bc = pml_mesh, "pec"
        kw["pml"] = dict(r0=1.0, r1=1.36)
    pw = "pw" in name
    if "lts" in name:
        mesh, mats, src, design = lts_mesh, {0: [4.0, 1.0]}, [0.0, 0.85], (0.1, 0.32)
        kw["lts"] = dict(p=2)
        bc = "pec"

    def make(dt=None):
        k = dict(kw)
        if "lts" in k and dt is not None:  # dt is the coarse LTS step
            k["lts"], dt = dict(k["lts"], dt=dt), None
        s = UnstructuredFIT2D(mesh, mats, polarization=pol, bc=bc, dt=dt, **k)
        if pw:  # scattered-field plane wave (non-dispersive scatterer)
            s.add_plane_wave(pulse, direction=(1.0, 0.3))
        else:
            s.add_point_source(src, pulse)
        return s

    mon = CellDFTMonitor([0.8 * F0, F0], points=[[-0.5, 0.2], [0.3, -0.55]])
    return make, _annulus(mesh, *design), mon, mesh


CONFIGS_2D = [
    f"{p} {f}"
    for p in ("TM", "TE")
    for f in ("pec", "pmc", "abc", "disp", "absorber", "pml", "pml disp", "lts", "pw")
]
CONFIGS_3D = ["3D pec", "3D abc", "3D disp", "3D whitney"]
CONFIGS = CONFIGS_2D + CONFIGS_3D


@pytest.fixture(scope="module")
def meshes(disk, pml_mesh, lts_mesh, ball):
    return disk, pml_mesh, lts_mesh, ball


def _fixed_dt(make):
    """Solver factory with the dt of a first build (3D massless Lanczos is unseeded)."""
    dt = make().dt
    return (lambda: make(dt=dt)), dt


# ---------------------------------------------------------------- dot test
def blockwise_dot(st, mutate=None, seed=0):
    """Max scale-invariant dot-product error over all (input, output) blocks."""
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
            a = float(Ax[j] @ y[j])
            b = float(x[i] @ ATy[i])
            scale = np.linalg.norm(Ax[j]) * np.linalg.norm(y[j]) + np.linalg.norm(
                x[i]
            ) * np.linalg.norm(ATy[i])
            if scale > 0:
                worst = max(worst, abs(a - b) / scale)
    return worst


@pytest.mark.parametrize("name", CONFIGS)
def test_forward_bitwise_and_dot_product(name, meshes):
    make, *_ = _config(name, meshes)
    s = make()
    st = UnstructuredAdjointStepper(s)
    x = st.zero_state()
    for n in range(40):
        s.one_step()
        x, _ = st.step(x, n)
    assert np.abs(s.u).max() > 0
    np.testing.assert_array_equal(x["u"], s.u)
    if "pml" in name:
        assert st.pml.spokes.size > 0 and st.pml.nodes.size > 0
    if "lts" in name:
        assert st.lts.F.size > 0
    if "disp" in name:
        assert len(st.keys()) > 2
    if name in ("3D pec", "3D abc", "3D disp"):
        assert s.massless.size > 0  # the Schur complement path is exercised
    assert blockwise_dot(st) < DOT_TOL


# ---------------------------------------------------------------- mutations
def _drop_schur(self, esb):
    return np.zeros(self.solver.free.size)


def _unsigned_schur(self, esb):
    s = self.solver
    return s._Ksf.T @ s._Kss.solve(esb, trans="T")


def _diag_schur(self, esb):
    s = self.solver
    Kss_diag = (s.Cs.T @ s.Mnu @ s.Cs).diagonal()
    return -(s._Ksf.T @ (esb / Kss_diag))


_orig_cfs_T = A._CFS.transpose


def _cfs_drop_x0(self, yb_ext, xb1, xb):
    out = _orig_cfs_T(self, yb_ext, xb1, xb)
    xb[self.name + "x0"] = 0 * xb[self.name + "x0"]
    return out


_orig_cap_T = A._Capacitive.transpose


def _cap_drop_debye_u(self, ub_new, xb1, xb, pre):
    saved = [list(g["debye"]) for g in self.groups]
    for g in self.groups:  # transpose without the beta (u' + u) feedback
        g["debye"] = [(k, 0.0) for k, _ in g["debye"]]
    try:
        return _orig_cap_T(self, ub_new, xb1, xb, pre)
    finally:
        for g, d in zip(self.groups, saved):
            g["debye"] = d


def _cap_drop_lorentz_xi(self, ub_new, xb1, xb, pre):
    out = _orig_cap_T(self, ub_new, xb1, xb, pre)
    for gi, g in enumerate(self.groups):
        for k in range(len(g["lorentz"])):
            xb[f"{pre}{gi}L{k}o"] = 0 * xb[f"{pre}{gi}L{k}o"]
    return out


def _lts_untransposed(self, z):
    return self.lts.apply(self.solver.G.T @ z)


_orig_te_ade_T = UnstructuredAdjointStepper._te_ade_T


def _te_ade_no_D(self, xb1, xb, v1b):
    keep = v1b.copy()
    _orig_te_ade_T(self, xb1, xb, v1b)
    v1b[:] = keep  # drop the D-flux coupling of the series ADE


MUTATIONS = [
    ("3D abc", "_massless_T", _drop_schur),
    ("3D abc", "_massless_T", _unsigned_schur),
    ("3D abc", "_massless_T", _diag_schur),
    ("TM pml", "cfs", _cfs_drop_x0),
    ("TE pml", "cfs", _cfs_drop_x0),
    ("TM disp", "cap", _cap_drop_debye_u),
    ("TM disp", "cap", _cap_drop_lorentz_xi),
    ("3D disp", "cap", _cap_drop_lorentz_xi),
    ("TM lts", "_grad_T", _lts_untransposed),
    ("TE lts", "_grad_T", _lts_untransposed),
    ("TE disp", "_te_ade_T", _te_ade_no_D),
]


@pytest.mark.parametrize("name,target,func", MUTATIONS,
                         ids=[f"{n}-{f.__name__}" for n, _, f in MUTATIONS])  # fmt: skip
def test_dot_product_mutations_fail(name, target, func, meshes, monkeypatch):
    make, *_ = _config(name, meshes)
    st = UnstructuredAdjointStepper(make())
    assert blockwise_dot(st) < DOT_TOL
    if target == "cfs":
        monkeypatch.setattr(A._CFS, "transpose", func)
    elif target == "cap":
        monkeypatch.setattr(A._Capacitive, "transpose", func)
    else:
        monkeypatch.setattr(UnstructuredAdjointStepper, target, func)
    assert blockwise_dot(st) > MUT_MIN


def test_whitney_transpose_needs_M1_solve(meshes, monkeypatch):
    make, *_ = _config("3D whitney", meshes)
    st = UnstructuredAdjointStepper(make())
    assert blockwise_dot(st) < DOT_TOL
    lu = st._M1

    class Wrong:  # forward keeps the true solve; the transpose skips it
        def solve(self, r, trans="N"):
            return lu.solve(r) if trans == "N" else r * 1e-12

    st._M1 = Wrong()
    assert blockwise_dot(st) > MUT_MIN


# ---------------------------------------------------------------- gradients
def objective(out):
    X = out["m"]["dft"]
    return float(np.sum(np.abs(X) ** 2)), {"m": {"dft": 2 * X}}


def _sim(name, meshes, memory="store", Nt=None, **kw):
    make, cells, mon, mesh = _config(name, meshes)
    make, _ = _fixed_dt(make)
    # LTS: the design sits in the coarse eps = 4 rod (F is the vacuum shell);
    # eps >= 4 keeps the frozen dt stable (raising eps lowers lambda_max)
    lo, hi = (4.0, 6.0) if "lts" in name else (1.0, 3.0)
    design = CellMaterialGrid(mesh, cells, lo, hi)
    Nt = Nt or (80 if name.startswith("3D") else 120)
    return UnstructuredAdjointSimulation(make, design, {"m": mon}, Nt, memory=memory, **kw)


def fd_check(sim, rho, g, cells, h=1e-3):
    """Per cell: |g - FD(h)| and the truncation estimate |FD(h) - FD(2h)| / 3."""
    F0_ = objective(sim.forward(rho, record=False))[0]
    out = []
    for k in cells:
        fd = []
        for hh in (h, 2 * h):
            rp, rm = rho.copy(), rho.copy()
            rp[k] += hh
            rm[k] -= hh
            Fp = objective(sim.forward(rp, record=False))[0]
            Fm = objective(sim.forward(rm, record=False))[0]
            fd.append((Fp - Fm) / (2 * hh))
        trunc = abs(fd[1] - fd[0]) / 3
        roundoff = 1e3 * np.finfo(float).eps * abs(F0_) / h
        out.append((g[k], fd[0], trunc, roundoff))
    return out


GRAD_CONFIGS = ["TM abc", "TE abc", "TM disp", "TE disp", "TM pml", "TE lts",
                "3D abc", "3D disp", "3D whitney"]  # fmt: skip


@pytest.mark.parametrize("name", GRAD_CONFIGS)
def test_gradient_vs_finite_differences(name, meshes):
    sim = _sim(name, meshes)
    if name in ("3D abc", "3D disp"):
        assert sim.solver.massless.size > 0
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.2, 0.8, sim.design.shape)
    F, g = sim.value_and_grad(rho, objective)
    assert F > 0 and np.all(np.isfinite(g)) and np.abs(g).max() > 0
    cells = rng.choice(rho.size, 3, replace=False)
    for adj, fd, trunc, roundoff in fd_check(sim, rho, g, cells):
        assert abs(adj - fd) <= 1.5 * trunc + roundoff, (adj, fd, trunc)
        assert abs(adj - fd) < 1e-5 * abs(fd)


def test_gradient_wrong_dual_area_map_fails(meshes, monkeypatch):
    sim = _sim("TM abc", meshes)
    st, s = sim.stepper, sim.solver
    rng = np.random.default_rng(1)
    rho = rng.uniform(0.2, 0.8, sim.design.shape)
    F, g = sim.value_and_grad(rho, objective)
    cells = rng.choice(rho.size, 3, replace=False)
    good = fd_check(sim, rho, g, cells)
    # Barycentric (A_T/3) instead of the Voronoi partial dual areas
    G = {"m": {"dft": 2 * sim.forward(rho)["m"]["dft"]}}  # set_eps rebuilds dmap
    wrong = s.mesh.dual_area_barycentric[s.free[st.dofs]][:, st.design_cells]
    st.dmap = (A.eps_0 * wrong).tocsr()
    g_bad = sim.vjp(G)
    errs = [abs(gb - fd) / abs(fd) for gb, (_, fd, _, _) in zip(g_bad[cells], good)]
    assert max(errs) > 1e-3


def test_gradient_te_wrong_power_fails(meshes):
    sim = _sim("TE abc", meshes)
    st = sim.stepper
    rng = np.random.default_rng(2)
    rho = rng.uniform(0.2, 0.8, sim.design.shape)
    _, g = sim.value_and_grad(rho, objective)
    cells = rng.choice(rho.size, 3, replace=False)
    good = fd_check(sim, rho, g, cells)
    G = {"m": {"dft": 2 * sim.forward(rho)["m"]["dft"]}}
    # d(1/eps)/deps = -1/eps^2 replaced by -1/eps
    st.dmap = (st.dmap @ sp.diags(st.eps[st.design_cells])).tocsr()
    g_bad = sim.vjp(G)
    errs = [abs(gb - fd) / abs(fd) for gb, (_, fd, _, _) in zip(g_bad[cells], good)]
    assert max(errs) > 1e-3


# ---------------------------------------------------------------- memory
@pytest.mark.parametrize("name", ["TM disp", "TE disp", "TM pml", "3D abc", "3D whitney"])
def test_checkpoint_gradient_bitwise(name, meshes):
    rho = np.random.default_rng(3).uniform(0.2, 0.8, _sim(name, meshes).design.shape)
    sim = _sim(name, meshes)
    _, g = sim.value_and_grad(rho, objective)
    for slots, block in ((2, None), (3, 7)):
        ck = _sim(name, meshes, memory="checkpoint", checkpoints=slots, block=block)
        ck.solver = sim.solver  # same dt as the stored run
        ck.stepper = UnstructuredAdjointStepper(sim.solver, ck.design.cells)
        ck.verify_replay = True
        _, g_ck = ck.value_and_grad(rho, objective)
        np.testing.assert_array_equal(g_ck, g)
        assert ck.memory_stats["recompute_steps"] > 0
        assert ck._rung.mismatches == 0


@pytest.mark.parametrize("name", ["TM disp", "TE pml", "3D abc"])
def test_dft_matches_solver_recorder(name, meshes):
    """rho = 0 puts eps_min = 1 (the vacuum the design cells already hold)."""
    sim = _sim(name, meshes)
    assert np.all(sim.stepper.base_eps[sim.design.cells] == 1.0)
    out = sim.forward(np.zeros(sim.design.shape), record=False)["m"]["dft"]
    s = sim.make_solver()
    rec = s.add_dft(sim.monitors["m"].freqs)
    s.run(sim.Nt)
    ref = (sim._W["m"] @ rec["data"].T).T
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


# ---------------------------------------------------------------- design map
def test_design_checks(disk, ball):
    s = UnstructuredFIT2D(disk, {0: _mat()}, bc="abc")
    s.add_plane_wave(pulse)
    with pytest.raises(NotImplementedError, match="dispersive plane wave"):
        UnstructuredAdjointStepper(s)
    s = UnstructuredFIT2D(disk, {0: [2.0, 1.0]}, bc="abc")
    s.add_plane_wave(pulse)
    with pytest.raises(NotImplementedError, match="plane-wave"):
        UnstructuredAdjointStepper(s, np.flatnonzero(disk.cell_tags == 1)[:3])
    cen = disk.nodes[disk.cells].mean(1)
    outer = np.flatnonzero(np.hypot(*cen.T) > 0.97)
    s = UnstructuredFIT2D(disk, {0: [2.0, 1.0]}, bc="abc")
    with pytest.raises(ValueError, match="boundary"):
        UnstructuredAdjointStepper(s, outer)
    s = UnstructuredFIT2D(disk, {0: _mat()}, bc="abc")
    with pytest.raises(ValueError, match="dispersive"):
        UnstructuredAdjointStepper(s, np.flatnonzero(disk.cell_tags == 0)[:3])
    s = UnstructuredFIT2D(disk, {0: "pec"}, bc="pec")
    with pytest.raises(ValueError, match="PEC"):
        UnstructuredAdjointStepper(s, np.flatnonzero(disk.cell_tags == 0)[:3])


def test_whitney_local_mass_assembles_M1(ball):
    from wakis.unstructured.solver3d import whitney_mass_matrices

    eps = np.random.default_rng(0).uniform(1, 3, ball.n_cells)
    M1, _ = whitney_mass_matrices(ball, eps, np.ones(ball.n_cells), np.ones(ball.n_cells))
    loc = A.whitney_local_mass(ball)
    ce = ball.cell_edges
    rows = np.repeat(ce, 6, axis=1).ravel()
    cols = np.tile(ce, (1, 6)).ravel()
    M = sp.csr_matrix(((eps[:, None, None] * loc).ravel(), (rows, cols)), shape=M1.shape)
    assert abs(M - M1).max() <= 1e-14 * abs(M1).max()


@pytest.mark.parametrize("filt", ["cone", "helmholtz"])
def test_filter_projection_vjp(disk, filt):
    cells = _annulus(disk, 0.38, 0.8)
    d = CellMaterialGrid(disk, cells, 1.0, 4.0, filter_radius=0.2, filter=filt, beta=8.0)
    rng = np.random.default_rng(0)
    rho = rng.uniform(0, 1, d.shape)
    if filt == "cone":
        np.testing.assert_allclose(np.asarray(d.F.sum(1)).ravel(), 1.0, rtol=1e-14)
        assert d.F.nnz > 5 * rho.size  # the filter actually spreads
    # constant density is preserved by both filters
    np.testing.assert_allclose(d.filtered(np.full(rho.size, 0.3)), 0.3, rtol=1e-12)
    # vjp vs central differences of a random linear functional of eps
    w = rng.standard_normal(rho.size)
    g = d.vjp(rho, w)
    for k in rng.choice(rho.size, 4, replace=False):
        h = 1e-6
        rp, rm = rho.copy(), rho.copy()
        rp[k] += h
        rm[k] -= h
        fd = (w @ d.eps_r(rp) - w @ d.eps_r(rm)) / (2 * h)
        assert abs(g[k] - fd) <= 1e-6 * np.abs(g).max()
    # linear part: exact transpose
    a, b = rng.standard_normal(rho.size), rng.standard_normal(rho.size)
    lin = CellMaterialGrid(disk, cells, 0.0, 1.0, filter_radius=0.2, filter=filt)
    assert abs(lin.filtered(a) @ b - a @ lin.vjp(a, b)) < 1e-13 * np.linalg.norm(a) * np.linalg.norm(b)


def test_full_chain_gradient_with_filter_and_projection(meshes):
    make, cells, mon, mesh = _config("TM abc", meshes)
    make, _ = _fixed_dt(make)
    d = CellMaterialGrid(mesh, cells, 1.0, 3.0, filter_radius=0.12, beta=4.0)
    sim = UnstructuredAdjointSimulation(make, d, {"m": mon}, 120)
    rng = np.random.default_rng(4)
    rho = rng.uniform(0.2, 0.8, d.shape)
    _, g = sim.value_and_grad(rho, objective)
    for adj, fd, trunc, roundoff in fd_check(sim, rho, g, rng.choice(rho.size, 3, replace=False)):
        assert abs(adj - fd) <= 1.5 * trunc + roundoff


def test_jax_custom_vjp(meshes):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from wakis.inverse import jax_simulation

    sim = _sim("TE abc", meshes)
    f = jax_simulation(sim)

    def J(rho):
        X = f(rho)["m"]["dft"]
        return jnp.sum(jnp.abs(X) ** 2)

    rho = np.random.default_rng(5).uniform(0.2, 0.8, sim.design.shape)
    val, g_jax = jax.value_and_grad(J)(jnp.asarray(rho))
    F, g = sim.value_and_grad(rho, objective)
    np.testing.assert_allclose(float(val), F, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(g_jax), g, rtol=1e-12, atol=1e-12 * np.abs(g).max())
