"""
Distributed (MPI) unstructured FIT: `wakis.unstructured.mpi`.

What is checked (physics guide section ``unstructured_mpi.md``):

* Bitwise equality with the serial step. Every owned row of the distributed
  operators is the serial CSR row with the same entry order, and all other
  updates are elementwise, so the N-rank run must equal `one_step` bit for
  bit (fields, probes, DFTs) for 2, 3 and 4 parts and every partitioner;
  with massless edges it must equal the fused-elimination
  `DeviceStepper(xp=numpy)` bit for bit and the serial LU elimination
  within `gpu.agreement_bound`. The 3D PML (4 sparse stages) must equal
  `PML3D.apply` bit for bit.
* Global energy: the Allreduce of the per-rank partial energies differs
  from the serial dot product only by summation order,
  |dE| <= 2 gamma_n E (Higham 2002, eq. 3.5; all terms >= 0).
* Mutation: dropping one halo entry (never refreshed) must break the
  agreement (difference >> the round-off bound).
* Transpose: <F x, y> = <x, F^T y> of the distributed source-free step
  within (2 gamma_m + 2 gamma_N) |y|^T |F| |x|, and F^T bitwise equal to
  the serial adjoint `UnstructuredAdjointStepper.step_T` / `PML3D.step_T`.

Run locally with the in-process `LocalCluster` (all ranks in one process,
the same per-rank arithmetic and halo copies); under
``mpiexec -n 2`` / ``-n 4`` the ``mpi`` tests run the real `DistributedFIT`
(skipped in a serial pytest).
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.constants import c

from wakis.dispersive import Debye, DispersiveMaterial, Drude, Lorentz
from wakis.unstructured import UnstructuredFIT2D, UnstructuredFIT3D, ball_mesh, disk_mesh
from wakis.unstructured import gpu as G
from wakis.unstructured import mpi as M
from wakis.unstructured.adjoint import UnstructuredAdjointStepper
from wakis.unstructured.pml3d import ball_with_pml_mesh

F0, BW = 0.25 * c, 0.15 * c
FREQS = [0.15 * c, 0.25 * c, 0.35 * c]
DRUDE = DispersiveMaterial(1.5, [Drude(1.5 * c, 0.2 * c), Lorentz(0.8, 2.5 * c, 0.1 * c)])
DEBYE = DispersiveMaterial(2.0, [Debye(1.5, 0.5 / c), Lorentz(0.6, 2.0 * c, 0.3 * c)])
N_STEPS = 120


class Pulse:
    """Picklable Gaussian-modulated sine (scatter mode pickles the sources)."""

    def __init__(self, f0, bw):
        self.s = 1 / (2 * np.pi * bw)
        self.f0 = f0

    def __call__(self, t):
        t0 = 5 * self.s
        return np.exp(-(((t - t0) / self.s) ** 2) / 2) * np.sin(2 * np.pi * self.f0 * (t - t0))


def gamma(m):
    return m * M.UNIT_ROUNDOFF / (1 - m * M.UNIT_ROUNDOFF)


def case(name):
    """Configured serial solver of each feature combination (< 1 s set-up)."""
    p = Pulse(F0, BW)
    if name == "3d_pec":  # no massless edges: bitwise vs one_step
        s = UnstructuredFIT3D(ball_mesh([0.5, 1.0], 0.25), {0: [3.0, 1.0]}, bc="pec",
                              massless_tol=0.0)  # fmt: skip
        assert s.massless.size == 0
        s.add_point_dipole([0.1, 0.05, 0.2], [0.2, 0.1, 1.0], p)
    elif name == "3d_abc_massless_dispersive":
        s = UnstructuredFIT3D(ball_mesh([0.5, 1.0], 0.22), {0: DEBYE, 1: [1.0, 1.0, 1e-3]},
                              bc="abc")  # fmt: skip
        assert s.massless.size > 0
        s.add_point_dipole([0.6, 0.05, 0.2], [0.2, 0.1, 1.0], p)
    elif name == "3d_pec_pecball_massless":  # PEC object + massless, non-dispersive
        s = UnstructuredFIT3D(ball_mesh([0.4, 1.0], 0.22), {0: "pec", 1: [1.0, 1.0, 1e-3]},
                              bc="abc")  # fmt: skip
        assert s.massless.size > 0
        s.add_point_dipole([0.6, 0.05, 0.2], [0.2, 0.1, 1.0], p)
    elif name == "3d_pml":
        m, cfg = ball_with_pml_mesh(1.0, 0.3, 3)
        s = UnstructuredFIT3D(m, {0: [2.0, 1.0]}, bc="pec", pml=cfg)
        s.add_point_dipole([0.1, 0.05, 0.2], [0.2, 0.1, 1.0], p)
    elif name == "tm_abc_lossy_dispersive":
        s = UnstructuredFIT2D(disk_mesh([0.5, 1.0], 0.12), {0: DRUDE, 1: [1.0, 1.0, 2e-3]},
                              polarization="TM", bc="abc")  # fmt: skip
        s.add_point_source([0.71, 0.13], p)
    elif name == "tm_pec":
        s = UnstructuredFIT2D(disk_mesh([0.5, 1.0], 0.12), {0: [4.0, 1.0]},
                              polarization="TM", bc="pec")  # fmt: skip
        s.add_point_source([0.21, 0.13], p)
    elif name == "tm_absorber":
        s = UnstructuredFIT2D(disk_mesh([0.5, 1.0], 0.12), {0: DEBYE}, polarization="TM",
                              bc="pmc", absorber=dict(r0=0.7, r1=1.0, s_max=0.5 * c))  # fmt: skip
        s.add_point_source([0.61, -0.2], p)
    elif name == "te_abc_absorber":
        s = UnstructuredFIT2D(disk_mesh([0.5, 1.0], 0.12), {0: [3.0, 1.0]}, polarization="TE",
                              bc="abc", absorber=dict(r0=0.7, r1=1.0, s_max=0.5 * c))  # fmt: skip
        s.add_point_source([0.71, 0.1], p, direction=[1.0, 0.4])
        s.add_point_source([0.3, -0.2], p)
    else:
        raise KeyError(name)
    if s.mesh.dim == 2:
        s.add_probe([[0.35, 0.05], [-0.2, 0.3], [0.8, -0.1]])
    else:
        s.add_probe([[0.35, 0.05, 0.1], [-0.2, 0.3, 0.0], [0.5, -0.4, 0.2]], 2)
        s.add_probe([[0.1, 0.4, -0.3]], 0)
    s.add_dft(FREQS)
    return s


def serial_state(s):
    if s.mesh.dim == 3 and s._pml is not None:
        return {"x": s._pml.x}
    return {"u": s.u, "f": s.b if s.mesh.dim == 3 else s.v}


def assert_bitwise(a, b):
    for k in a:
        assert np.array_equal(a[k], b[k]), (k, np.abs(a[k] - b[k]).max())


BITWISE = ["3d_pec", "3d_pml", "tm_abc_lossy_dispersive", "tm_pec", "tm_absorber",
           "te_abc_absorber"]  # fmt: skip
MASSLESS = ["3d_abc_massless_dispersive", "3d_pec_pecball_massless"]


# ------------------------------------------------------------------ partitions
@pytest.mark.parametrize("method", ["rcb", "rib"])
def test_partition_balanced_and_complete(method):
    m = ball_mesh([0.5, 1.0], 0.25)
    for n in (2, 3, 4, 7):
        parts = M.partition_cells(m, n, method)
        st = M.partition_stats(m, parts)
        assert sorted(np.unique(parts)) == list(range(n))
        # balanced bisection: counts differ by the rounding of each split
        assert max(st["cells"]) - min(st["cells"]) <= int(np.ceil(np.log2(n)))
        assert 0 < st["cut_facets"] < M.cell_adjacency(m).nnz // 2


def test_metis_partition_if_available():
    pytest.importorskip("pymetis")
    m = ball_mesh([0.5, 1.0], 0.25)
    parts = M.partition_cells(m, 4, "metis")
    st = M.partition_stats(m, parts)
    rcb = M.partition_stats(m, M.partition_cells(m, 4, "rcb"))
    assert st["imbalance"] < 1.05
    assert st["cut_facets"] <= 1.2 * rcb["cut_facets"]


# ------------------------------------------------------------------ bitwise
@pytest.mark.parametrize("name", BITWISE)
@pytest.mark.parametrize("n_parts,method", [(2, "rcb"), (3, "rib"), (4, "rcb")])
def test_local_cluster_bitwise_serial(name, n_parts, method):
    s = case(name)
    cl = M.LocalCluster(s, n_parts, partitioner=method)
    # every rank must actually exchange something (no degenerate partition)
    assert all(sum(st.values()) > 0 for st in cl.comm_stats())
    cl.run(N_STEPS)
    s.run(N_STEPS)
    assert np.abs(s.u).max() > 0
    assert_bitwise(cl.gather_state(), serial_state(s))
    for i, rec in enumerate(s.recorder.probes):
        assert np.array_equal(cl.probe_signal(i), s.probe_signal(rec))
    assert np.array_equal(cl.dft_data(0), s.recorder.dfts[0]["data"])


@pytest.mark.parametrize("name", MASSLESS)
def test_massless_bitwise_fused_and_within_bound_of_serial(name):
    n = 80
    s = case(name)
    bound = G.agreement_bound(case(name), n, probes=False)
    dev = G.DeviceStepper(case(name), xp=np)
    cl = M.LocalCluster(s, 4, partitioner="rcb")
    cl.run(n)
    dev.run(n)
    g = cl.gather_state()
    st = dev.get_state()
    assert np.array_equal(g["u"], st["u"]) and np.array_equal(g["f"], st["b"])
    s.run(n)
    W = bound["weights"]
    d = np.sqrt(W["u"] @ (g["u"] - s.u) ** 2 + W["b"] @ (g["f"] - s.b) ** 2)
    assert 0 < d <= bound["eps"][-1], (d, bound["eps"][-1])


@pytest.mark.parametrize("name", ["3d_pec", "3d_pml", "tm_pec"])
def test_energy_allreduce_within_summation_bound(name):
    s = case(name)
    cl = M.LocalCluster(s, 4, partitioner="rib")
    cl.run(N_STEPS)
    s.run(N_STEPS)
    if name == "3d_pml":
        e_ser = sum(r.energy_partial() for r in M.LocalCluster(s, 1).ranks)
        n_terms = s._pml.layout["u"].size + s.mesh.n_faces
    else:
        e_ser = s.energy()
        n_terms = s.u.size + (s.b.size if s.mesh.dim == 3 else s.v.size)
    e = cl.energy()
    assert e_ser > 0
    assert abs(e - e_ser) <= 2 * gamma(n_terms + 2) * e_ser


@pytest.mark.parametrize("name", ["3d_pec", "3d_abc_massless_dispersive", "3d_pml",
                                  "tm_abc_lossy_dispersive", "te_abc_absorber"])  # fmt: skip
def test_mutation_drop_one_halo_entry_fails(name):
    s = case(name)
    good = M.LocalCluster(s, 4, partitioner="rcb")
    bad = M.LocalCluster(s, 4, partitioner="rcb", _mutation="drop_halo")
    good.run(N_STEPS)
    bad.run(N_STEPS)
    a, b = good.gather_state(), bad.gather_state()
    k = "x" if "x" in a else "u"
    rel = np.abs(a[k] - b[k]).max() / np.abs(a[k]).max()
    assert rel > 1e-6, rel  # round-off would be ~1e-15


# ------------------------------------------------------------------ transpose
def _abs_apply(s, x):
    """|F| |x| of the source-free step (absolute-value twin)."""
    if s.mesh.dim == 3 and s._pml is not None:
        y = np.abs(x["x"])
        S1, S2, S3, S4 = (abs(sp.csr_matrix(S)) for S in s._pml.stages)
        return {"x": S4 @ (S3 @ (S2 @ (S1 @ y)))}, sum(
            int(np.diff(sp.csr_matrix(S).indptr).max()) for S in s._pml.stages)
    tw = G.DeviceStepper(s, xp=np, absolute=True, sources=False)
    key = "b" if s.mesh.dim == 3 else "v"
    tw.set_state({**tw.get_state(), "u": np.abs(x["u"]), key: np.abs(x["f"])})
    tw._record = lambda t: None
    tw.step()
    st = tw.get_state()
    return {"u": st["u"], "f": st[key]}, tw.depth


def state_weights(s):
    """
    Diagonal energy-type weights W of the state entries.

    The dot-product test draws x = W^-1/2 z1, y = W^1/2 z2 so that every
    term of <F x, y> is O(1) in the energy norm; with raw N(0, 1) vectors
    the bound is dominated by a few huge (M_nu ~ 1/mu_0) terms and a wrong
    coefficient of an O(dt) term would be invisible.
    """
    pos = lambda w: np.where(np.asarray(w) > 0, w, 1.0)  # noqa: E731
    if s.mesh.dim == 3 and s._pml is not None:
        p, dt = s._pml, s.dt
        L = p.layout
        W = np.ones(L.n)
        cap_e = s.cap[p.ue]
        Mnu = pos(s.Mnu_diag)
        Mf = Mnu[p.f_pml]
        W[L["u"]] = s.cap
        W[L["b"]] = Mnu
        W[L["d"]] = 1 / cap_e
        for name, idx in L.blocks.items():
            base = cap_e if name.startswith("fe") else (1 / Mf if name.startswith("ff") else None)
            if base is None:
                continue
            W[idx] = base / dt**2 if name.endswith(("_Px", "_Py")) else base
        return {"x": W}
    if s.mesh.dim == 3:
        return {"u": s.cap, "f": pos(s.Mnu_diag)}
    return {"u": s.mass, "f": pos(s.stiff)}


def balanced_pair(s, seed):
    rng = np.random.default_rng(seed)
    W = state_weights(s)
    x = {k: rng.standard_normal(w.size) / np.sqrt(w) for k, w in W.items()}
    y = {k: rng.standard_normal(w.size) * np.sqrt(w) for k, w in W.items()}
    return x, y


@pytest.mark.parametrize("name", ["3d_pec", "3d_pec_pecball_massless", "3d_pml", "tm_pec"])
def test_distributed_transpose_dot_product(name):
    s = case(name)
    cl = M.LocalCluster(s, 4, partitioner="rib")
    shapes = {k: v.size for k, v in serial_state(s).items()}
    x, y = balanced_pair(s, 3)
    Fx = cl.apply(x)
    FTy = cl.apply_T(y)
    lhs = sum(Fx[k] @ y[k] for k in shapes)
    rhs = sum(x[k] @ FTy[k] for k in shapes)
    ax, depth = _abs_apply(s, x)
    scale = sum(ax[k] @ np.abs(y[k]) for k in shapes)
    N = sum(shapes.values())
    tol = (2 * gamma(depth) + 2 * gamma(N)) * scale
    assert abs(lhs - rhs) <= tol, (abs(lhs - rhs), tol)
    print(name, "dot", abs(lhs - rhs) / scale, "tol/scale", tol / scale)
    # mutation: a wrong coefficient in one rank's transpose must fail the test
    r = cl.ranks[1]
    key = "ST" if "ST" in r.pc else "AT"
    A = r.pc[key][2] if key == "ST" else r.pc[key]  # PML: Ampere stage S3^T
    A.data[np.argmax(np.abs(A.data))] *= 1.0 + 1e-3
    FTy_bad = cl.apply_T(y)
    rhs_bad = sum(x[k] @ FTy_bad[k] for k in shapes)
    print(name, "mutated", abs(lhs - rhs_bad) / tol)
    assert abs(lhs - rhs_bad) > 10 * tol


@pytest.mark.parametrize("name", ["3d_pec", "tm_pec", "3d_pml"])
def test_transpose_bitwise_serial_adjoint(name):
    s = case(name)
    cl = M.LocalCluster(s, 3, partitioner="rcb")
    rng = np.random.default_rng(5)
    st = serial_state(s)
    lam = {k: rng.standard_normal(v.size) for k, v in st.items()}
    got = cl.apply_T(lam)
    if name == "3d_pml":
        xb, gsrc = s._pml.step_T(lam["x"])
        assert np.array_equal(got["x"], xb) and np.array_equal(got["src"], gsrc)
        return
    adj = UnstructuredAdjointStepper(s)
    key = "b" if s.mesh.dim == 3 else "v"
    xb, _ = adj.step_T({"u": lam["u"], key: lam["f"]})
    assert np.array_equal(got["u"], xb["u"]) and np.array_equal(got["f"], xb[key])


def test_unsupported_raise():
    s = UnstructuredFIT3D(ball_mesh([1.0], 0.3), hodge="whitney")
    with pytest.raises(NotImplementedError):
        M.LocalCluster(s, 2)


# ------------------------------------------------------------------ real MPI
try:
    from mpi4py import MPI

    _SIZE = MPI.COMM_WORLD.Get_size()
except ImportError:  # pragma: no cover
    MPI, _SIZE = None, 1

mpi = pytest.mark.skipif(_SIZE < 2, reason="needs mpiexec -n >= 2")


@mpi
@pytest.mark.parametrize("name", BITWISE + MASSLESS)
@pytest.mark.parametrize("mode", ["scatter", "replicated"])
def test_mpi_bitwise_reference(name, mode):
    comm = MPI.COMM_WORLD
    s = case(name)
    d = M.DistributedFIT(s if (mode == "replicated" or comm.rank == 0) else None, comm,
                         partitioner="rcb", mode=mode)  # fmt: skip
    d.run(N_STEPS)
    g = d.gather_state()
    probes = [d.probe_signal(i) for i in range(len(s.recorder.probes))]
    dft = d.dft_data(0)
    e = d.energy()
    if comm.rank != 0:
        return
    if name in MASSLESS:  # fused elimination: bitwise vs the host DeviceStepper
        ref = G.DeviceStepper(case(name), xp=np)
        ref.run(N_STEPS)
        st = ref.get_state()
        assert_bitwise(g, {"u": st["u"], "f": st["b"]})
        return
    s.run(N_STEPS)
    assert_bitwise(g, serial_state(s))
    for i, rec in enumerate(s.recorder.probes):
        assert np.array_equal(probes[i], s.probe_signal(rec))
    assert np.array_equal(dft, s.recorder.dfts[0]["data"])
    if name not in MASSLESS and name != "3d_pml":
        n_terms = sum(v.size for v in serial_state(s).values())
        assert abs(e - s.energy()) <= 2 * gamma(n_terms + 2) * s.energy()


@mpi
@pytest.mark.parametrize("name", ["3d_pec", "3d_pml"])
def test_mpi_mutation_fails(name):
    comm = MPI.COMM_WORLD
    s = case(name)
    d = M.DistributedFIT(s, comm, partitioner="rcb", mode="replicated", _mutation="drop_halo")
    d.run(N_STEPS)
    g = d.gather_state()
    if comm.rank == 0:
        s.run(N_STEPS)
        ref = serial_state(s)
        k = "x" if "x" in ref else "u"
        assert np.abs(g[k] - ref[k]).max() > 1e-6 * np.abs(ref[k]).max()


@mpi
@pytest.mark.parametrize("name", ["3d_pec", "3d_pec_pecball_massless", "3d_pml", "tm_pec"])
def test_mpi_transpose_dot_product(name):
    comm = MPI.COMM_WORLD
    s = case(name)
    d = M.DistributedFIT(s, comm, partitioner="rib", mode="replicated")
    shapes = {k: v.size for k, v in serial_state(s).items()}
    x, y = balanced_pair(s, 3)
    Fx = d.apply(x)
    FTy = d.apply_T(y)
    if comm.rank != 0:
        return
    lhs = sum(Fx[k] @ y[k] for k in shapes)
    rhs = sum(x[k] @ FTy[k] for k in shapes)
    ax, depth = _abs_apply(s, x)
    scale = sum(ax[k] @ np.abs(y[k]) for k in shapes)
    tol = (2 * gamma(depth) + 2 * gamma(sum(shapes.values()))) * scale
    assert abs(lhs - rhs) <= tol
    if name in ("3d_pec", "tm_pec"):
        adj = UnstructuredAdjointStepper(s)
        key = "b" if s.mesh.dim == 3 else "v"
        xb, _ = adj.step_T({"u": y["u"], key: y["f"]})
        assert np.array_equal(FTy["u"], xb["u"]) and np.array_equal(FTy["f"], xb[key])
