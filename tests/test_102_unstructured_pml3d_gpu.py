"""
Device (CuPy) stepping of `UnstructuredFIT3D` with the spherical CFS-PML.

The device step is the four PML stages S1..S4 as SpMVs on one vector
x = [u, b, z] (z: PML fluxes and filter memories) with the sources folded
into B w. The host `PML3D.apply` evaluates the same recurrence with SciPy;
the two differ only by the order of the sums, bounded by `agreement_bound`:

    ||d_N||_W <= 2 K gamma_m sum_n || |A|(|x_n|) ||_W,

now with the filter states in the norm (weights: d 1/C, edge-filter
voltages C, face-filter currents 1/M_nu, filter integrals / dt^2) and
m = 12 + sum_k max_row_nnz(S_k) + 2 (the rounding of each stage propagates
through the absolute values of the later ones). K is the empirical
amplification over random states in that norm (the PML energy is not a
proven invariant; a lower estimate, as for the 2D PML in test_099).

Checks (xp=numpy always, xp=cupy with a GPU): state (u, b and z), probes and
DFTs within their bounds, a non-vacuous bound (eps / ||x|| < 1e-7);
mutations (face-filter stage scaled by 1 + 1e-3; the source not folded
through B) exceed the bound by > 100x; `to_host` hands the extended state
back so the host solver continues (to round-off); no host synchronisation in a step.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.constants import c

from wakis.unstructured import UnstructuredFIT3D
from wakis.unstructured import gpu as G
from wakis.unstructured.pml3d import ball_with_pml_mesh

try:
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()
    HAVE_GPU = True
except Exception:
    cp, HAVE_GPU = None, False

BACKENDS = [pytest.param(np, id="numpy")] + [
    pytest.param(cp, id="cupy", marks=pytest.mark.skipif(not HAVE_GPU, reason="needs a GPU"))
]
F0, BW = 0.25 * c, 0.15 * c
N_STEPS = 300


def pulse(t):
    s = 1 / (2 * np.pi * BW)
    t0 = 5 * s
    return np.exp(-(((t - t0) / s) ** 2) / 2) * np.sin(2 * np.pi * F0 * (t - t0))


_MESH = {}


def case(kappa_max=2.0, alpha=1e8):
    if not _MESH:
        _MESH["m"] = ball_with_pml_mesh(1.0, 0.25, 4)
    m, cfg = _MESH["m"]
    s = UnstructuredFIT3D(m, bc="pec", pml=dict(cfg, kappa_max=kappa_max, alpha=alpha))
    s.add_point_dipole([0.1, 0.05, 0.2], [0.2, 0.1, 1.0], pulse)
    s.add_point_dipole([-0.4, 0.3, 0.1], [1.0, 0.0, 0.3], pulse)
    s.add_probe([[0.35, 0.05, 0.1], [-0.2, 0.3, 0.0]], 2)
    s.add_dft([0.15 * c, F0])
    return s


def compare(s, dev, n_steps):
    bound = G.agreement_bound(s, n_steps)
    dev.run(n_steps)
    s.run(n_steps)
    nu, nf = s.free.size, s.mesh.n_faces
    host = {"u": s.u, "b": s.b, "z": s._pml.x[nu + nf :]}
    W = bound["weights"]
    st = dev.get_state()
    d = float(np.sqrt(sum(np.sum(W[k] * (st[k] - host[k]) ** 2) for k in W)))
    pr = max(
        np.max(np.abs(dev.probe_signal(i) - s.probe_signal(rec)) / np.maximum(b, 1e-300))
        for i, (rec, b) in enumerate(zip(s.recorder.probes, bound["probe"]))
    )
    dft = max(
        np.max(np.sqrt((W["u"] * np.abs(dev.dft_data(i) - rec["data"]) ** 2).sum(1)) / bound["dft"][i])
        for i, rec in enumerate(s.recorder.dfts)
    )
    return dict(d=d, eps=bound["eps"][-1], norm=bound["norm"].max(), probe=pr, dft=dft, bound=bound)


@pytest.mark.parametrize("xp", BACKENDS)
@pytest.mark.parametrize("opts", [dict(), dict(kappa_max=1.0, alpha=0.0)], ids=["cfs", "plain"])
def test_device_matches_cpu_within_bound(xp, opts):
    s = case(**opts)
    assert s.massless.size > 0  # stretched massless elimination inside S1
    dev = G.DeviceStepper(s, xp=xp)
    r = compare(s, dev, N_STEPS)
    print(f"[{xp.__name__} {opts}] ||d||/||x|| = {r['d'] / r['norm']:.2e}, eps/||x|| = "
          f"{r['eps'] / r['norm']:.2e}, d/eps = {r['d'] / r['eps']:.2e}, K = {r['bound']['K']:.2f}, "
          f"m = {r['bound']['depth']}, probe {r['probe']:.2e}, dft {r['dft']:.2e}")  # fmt: skip
    assert r["norm"] > 0 and np.isfinite(r["d"])
    assert np.abs(dev.get_state()["z"]).max() > 0  # the field entered the layer
    assert r["eps"] / r["norm"] < 1e-7, "bound is vacuous"
    assert r["d"] <= r["eps"]
    assert r["probe"] <= 1.0 and r["dft"] <= 1.0


def _scaled(dev, k, f):
    S = dev.pml3.stages[k] * f
    dev.P[k] = dev._mat(sp.csr_matrix(S))


@pytest.mark.parametrize("xp", BACKENDS)
@pytest.mark.parametrize("what", ["face_filter", "source_unfolded"])
def test_mutated_device_step_exceeds_bound(xp, what):
    s = case()
    if what == "source_unfolded":
        # the source added to u directly (as without the PML) instead of B w
        orig = s._pml.B
        nu = s.free.size
        s._pml.B = sp.vstack([sp.identity(nu), sp.csr_matrix((orig.shape[0] - nu, nu))]).tocsr()
        dev = G.DeviceStepper(s, xp=xp)
        s._pml.B = orig
    else:
        dev = G.DeviceStepper(s, xp=xp)
        _scaled(dev, 1, 1 + 1e-3)
    r = compare(s, dev, N_STEPS)
    print(f"mutation {what} [{xp.__name__}]: d/eps = {r['d'] / r['eps']:.2e}")
    assert not np.isfinite(r["d"]) or r["d"] > 100 * r["eps"]


def test_to_host_hands_back_extended_state():
    s, ref = case(), case()
    dev = G.DeviceStepper(s, xp=np)
    dev.run(30)
    dev.to_host()
    ref.run(30)
    assert s.n == 30 and len(s.probe_signal(s.recorder.probes[0])) == 30
    # numpy device path: same SciPy matrices (sorted indices), sums in the same
    # order except for the folded sources -> compare after one host step each
    nu = s.free.size
    np.testing.assert_array_equal(s.u, s._pml.x[:nu])
    s.one_step()
    ref.one_step()
    assert np.abs(s.u - ref.u).max() <= 1e-12 * np.abs(ref.u).max()


@pytest.mark.skipif(not HAVE_GPU, reason="needs CuPy and a GPU")
def test_device_step_does_not_synchronize():
    spin = cp.RawKernel(
        r"""
extern "C" __global__ void _spin(long long ns) {
    long long t0, t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0)); t = t0;
    while (t - t0 < ns) { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); }
}""",
        "_spin",
    )
    dev = G.DeviceStepper(case(), xp=cp)
    dev.run(5)
    dev.prepare(50)
    for pr in dev.probes:
        dev._reserve(pr, pr["k"] + 50)
    cp.cuda.Stream.null.synchronize()
    ev = cp.cuda.Event()
    spin((1,), (1,), (np.int64(100_000_000),))
    ev.record()
    for _ in range(3):
        dev.step()
    done = ev.done
    cp.cuda.Stream.null.synchronize()
    assert not done, "DeviceStepper.step synchronized the stream"
