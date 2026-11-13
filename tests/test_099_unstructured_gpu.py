"""
Device (CuPy) stepping of the unstructured FIT solvers (`wakis.unstructured.gpu`).

Every case runs the host SciPy solver (`one_step`) and the `DeviceStepper` for
N steps from the same initial state and checks, against the derived
round-off bound of `agreement_bound` (physics guide
``docs/physicsguide_sections/unstructured_gpu.md``):

* the state difference ||d_N||_W <= eps_N (energy norm, all fields),
* every probe sample within its componentwise bound,
* every DFT within its bound,

and that the bound is not vacuous (eps_N / max_n ||x_n||_W < 1e-7). The bound is
2 K gamma_m sum_n || |A|(|x_n|) ||_W; nothing in it is fitted.

The same checks run with ``xp=numpy`` (the device code path on the host:
always) and with ``xp=cupy`` (skipped without a GPU). Mutation tests perturb
the device step (massless-edge correction dropped, ADE coefficient, LTS
buffer coupling, PML filter swap) and must exceed the bound. The no-sync test
follows tests/test_080_particles_gpu.py (a 60 ms device spin kernel must still
be running when `step` returns).
"""

import numpy as np
import pytest
from scipy.constants import c

from wakis.dispersive import Debye, DispersiveMaterial, Drude, Lorentz
from wakis.unstructured import UnstructuredFIT2D, UnstructuredFIT3D, ball_mesh
from wakis.unstructured import gpu as G
from wakis.unstructured.meshers import disk_mesh, graded_disk_mesh

try:
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()
    HAVE_GPU = True
except Exception:
    cp, HAVE_GPU = None, False

BACKENDS = [pytest.param(np, id="numpy")] + [
    pytest.param(
        cp, id="cupy",
        marks=pytest.mark.skipif(not HAVE_GPU, reason="needs CuPy and a GPU"),
    )
]  # fmt: skip


def pulse(f0, bw):
    s = 1 / (2 * np.pi * bw)
    t0 = 5 * s
    return lambda t: np.exp(-(((t - t0) / s) ** 2) / 2) * np.sin(2 * np.pi * f0 * (t - t0))


F0, BW = 0.25 * c, 0.15 * c  # Hz, for meshes of size ~1 m
FREQS = [0.15 * c, 0.25 * c, 0.35 * c]
DRUDE = DispersiveMaterial(1.5, [Drude(1.5 * c, 0.2 * c), Lorentz(0.8, 2.5 * c, 0.1 * c)])
DEBYE = DispersiveMaterial(2.0, [Debye(1.5, 0.5 / c), Lorentz(0.6, 2.0 * c, 0.3 * c)])


# ------------------------------------------------------------------ cases
def _disk(h=0.12):
    return disk_mesh([0.5, 1.0], h)


def case(name):
    """Configured solver of each feature combination (small, < 1 s setup)."""
    if name == "tm_pec":
        s = UnstructuredFIT2D(_disk(), {0: [4.0, 1.0]}, polarization="TM", bc="pec")
        s.add_point_source([0.21, 0.13], pulse(F0, BW))
    elif name == "tm_abc_lossy_dispersive":
        s = UnstructuredFIT2D(_disk(), {0: DRUDE, 1: [1.0, 1.0, 2e-3]}, polarization="TM",
                              bc="abc")  # fmt: skip
        s.add_point_source([0.71, 0.13], pulse(F0, BW))
    elif name == "tm_pmc_debye_absorber":
        s = UnstructuredFIT2D(_disk(), {0: DEBYE}, polarization="TM", bc="pmc",
                              absorber=dict(r0=0.7, r1=1.0, s_max=0.5 * c))  # fmt: skip
        s.add_point_source([0.61, -0.2], pulse(F0, BW))
    elif name == "te_abc_dispersive":
        s = UnstructuredFIT2D(_disk(), {0: DEBYE}, polarization="TE", bc="abc")
        s.add_point_source([0.71, 0.1], pulse(F0, BW), direction=[1.0, 0.4])
        s.add_point_source([0.3, -0.2], pulse(F0, BW))
    elif name == "te_drude_absorber":
        s = UnstructuredFIT2D(_disk(), {0: DRUDE}, polarization="TE", bc="pec",
                              absorber=dict(r0=0.7, r1=1.0, s_max=0.5 * c))  # fmt: skip
        s.add_point_source([0.6, 0.1], pulse(F0, BW), direction=[0.3, 1.0])
    elif name in ("tm_pml", "te_pml"):
        h = 0.1
        m = graded_disk_mesh([0.4, 1.0], h, pml_layers=6, pml_dr=h)
        s = UnstructuredFIT2D(m, {0: DRUDE}, polarization=name[:2].upper(), bc="pec",
                              pml=dict(r0=1.0, r1=1.0 + 6 * h))  # fmt: skip
        s.add_point_source([0.6, 0.1], pulse(F0, BW))
    elif name in ("tm_lts", "te_lts"):
        h = 0.06

        def size(r):
            d = r - 0.4 - 2 * h
            return h if d <= 0 else min(2 * h, h * 1.2 ** (d / h))

        m = graded_disk_mesh([0.4, 1.0], size)
        s = UnstructuredFIT2D(m, {0: [4.0, 1.0]}, polarization=name[:2].upper(), bc="pec",
                              lts=dict(p=2))  # fmt: skip
        s.add_point_source([0.5, 0.1], pulse(F0, BW))
    elif name in ("tm_planewave", "te_planewave"):
        s = UnstructuredFIT2D(_disk(), {0: [3.0, 1.0]}, polarization=name[:2].upper(),
                              bc="abc")  # fmt: skip
        s.add_plane_wave(pulse(F0, BW), direction=(1.0, 0.3))
    elif name == "3d_pec":
        s = UnstructuredFIT3D(ball_mesh([0.5, 1.0], 0.25), {0: [3.0, 1.0]}, bc="pec")
        s.add_point_dipole([0.1, 0.05, 0.2], [0.2, 0.1, 1.0], pulse(F0, BW))
    elif name == "3d_abc_massless_dispersive":
        s = UnstructuredFIT3D(ball_mesh([0.5, 1.0], 0.22), {0: DEBYE, 1: [1.0, 1.0, 1e-3]},
                              bc="abc")  # fmt: skip
        assert s.massless.size > 0
        s.add_point_dipole([0.6, 0.05, 0.2], [0.2, 0.1, 1.0], pulse(F0, BW))
        s.add_current_density(
            lambda x: np.exp(-((x - [0.1, 0.0, 0.0]) ** 2).sum(1, keepdims=True) / 0.05)
            * np.array([[1.0, 0.0, 0.0]]), pulse(F0, BW),
        )  # fmt: skip
    elif name == "3d_pmc_drude":
        s = UnstructuredFIT3D(ball_mesh([0.5, 1.0], 0.25), {0: DRUDE}, bc="pmc")
        s.add_point_dipole([0.3, 0.05, 0.2], [1.0, 0.1, 0.2], pulse(F0, BW))
    else:
        raise KeyError(name)
    if s.mesh.dim == 2:
        s.add_probe([[0.35, 0.05], [-0.2, 0.3], [0.8, -0.1]])
    else:
        s.add_probe([[0.35, 0.05, 0.1], [-0.2, 0.3, 0.0]], 2)
        s.add_probe([[0.1, 0.4, -0.3]], 0)
    s.add_dft(FREQS)
    return s


CASES_2D = ["tm_pec", "tm_abc_lossy_dispersive", "tm_pmc_debye_absorber", "te_abc_dispersive",
            "te_drude_absorber", "tm_pml", "te_pml", "tm_lts", "te_lts", "tm_planewave",
            "te_planewave"]  # fmt: skip
CASES_3D = ["3d_pec", "3d_abc_massless_dispersive", "3d_pmc_drude"]
N_STEPS = 300


def _fields(state, kind):
    return {"u": state["u"], ("b" if kind == "3d" else "v"): state["b" if kind == "3d" else "v"]}


def _wdiff(a, b, W, keys):
    return float(np.sqrt(sum(np.sum(W[k] * (a[k] - b[k]) ** 2) for k in keys)))


def compare(s, dev, n_steps, bound=None):
    """Run CPU and device; return measured differences and the bound."""
    bound = bound or G.agreement_bound(s, n_steps)
    dev.run(n_steps)
    s.run(n_steps)
    kind = dev.kind
    keys = ["u", "b" if kind == "3d" else "v"]
    host = {"u": s.u, "b": getattr(s, "b", None), "v": getattr(s, "v", None)}
    W = bound["weights"]
    d = _wdiff(dev.get_state(), host, W, keys)
    pr = [
        np.max(np.abs(dev.probe_signal(i) - s.probe_signal(rec)) / np.maximum(b, 1e-300))
        for i, (rec, b) in enumerate(zip(s.recorder.probes, bound["probe"]))
    ]
    dft = []
    for i, rec in enumerate(s.recorder.dfts):
        D = dev.dft_data(i) - rec["data"]
        dft.append(np.max(np.sqrt((W["u"] * np.abs(D) ** 2).sum(1)) / bound["dft"][i]))
    return dict(d=d, eps=bound["eps"][-1], norm=bound["norm"].max(), probe_ratio=max(pr),
                dft_ratio=max(dft), bound=bound)  # fmt: skip


def assert_within(r):
    assert r["norm"] > 0 and np.isfinite(r["d"])
    # the bound must resolve relative changes of 1e-7 (the mutations below,
    # coefficient changes of 1e-3 .. 1e-2, give 1e-5 .. 1)
    assert r["eps"] / r["norm"] < 1e-7, "bound is vacuous"
    assert r["d"] <= r["eps"], f"||d|| = {r['d']:.3e} > eps = {r['eps']:.3e}"
    assert r["probe_ratio"] <= 1.0
    assert r["dft_ratio"] <= 1.0


# ------------------------------------------------------------------ agreement
@pytest.mark.parametrize("xp", BACKENDS)
@pytest.mark.parametrize("name", CASES_2D + CASES_3D)
def test_device_matches_cpu_within_roundoff_bound(name, xp):
    s = case(name)
    dev = G.DeviceStepper(s, xp=xp)
    r = compare(s, dev, N_STEPS)
    print(f"{name} [{xp.__name__}]: ||d||/||x|| = {r['d'] / r['norm']:.2e}, "
          f"eps/||x|| = {r['eps'] / r['norm']:.2e}, d/eps = {r['d'] / r['eps']:.2e}, "
          f"K = {r['bound']['K']:.2f}, m = {r['bound']['depth']}")  # fmt: skip
    assert_within(r)


@pytest.mark.parametrize("name", ["tm_pec", "te_pml", "3d_pec"])
def test_unfused_massless_and_to_host(name):
    """to_host() hands fields / probes / DFTs back; the solver can go on."""
    s = case(name)
    dev = G.DeviceStepper(s, xp=np, fuse_massless=False)
    dev.run(20)
    dev.to_host()
    assert s.n == 20 and len(s.probe_signal(s.recorder.probes[0])) == 20
    assert np.array_equal(s.u, dev.get_state()["u"])


def test_massless_unfused_equals_fused_within_bound():
    s = case("3d_abc_massless_dispersive")
    a = G.DeviceStepper(s, xp=np, fuse_massless=True)
    b = G.DeviceStepper(s, xp=np, fuse_massless=False)
    bnd = G.agreement_bound(s, 100, probes=False)
    a.run(100)
    b.run(100)
    W = bnd["weights"]
    d = _wdiff(a.get_state(), b.get_state(), W, ["u", "b"])
    assert d <= bnd["eps"][-1]


# ------------------------------------------------------------------ the bound
@pytest.mark.parametrize("name", ["tm_pec", "3d_pec"])
def test_theory_amplification_bounds_empirical(name):
    """Lossless leapfrog: random-state growth never exceeds sqrt((1+rho)/(1-rho))
    in the energy norm (fields only). Can fail if rho or the weights are wrong."""
    s = case(name)
    keys = ["u", "b" if s.mesh.dim == 3 else "v"]
    Kt = G.theory_amplification(s)
    Ke = G.empirical_amplification(s, 200, n_seeds=2, keys=keys)
    assert 1.0 <= Ke <= Kt


def test_bound_grows_linearly_and_measured_far_below():
    s = case("tm_pec")
    b = G.agreement_bound(s, 200, probes=False)
    e = b["eps"]
    assert np.all(np.diff(e) > 0)
    # eps_n / n roughly constant once the field is up (|A| |x| ~ ||x||)
    assert e[-1] / e[99] > 1.5


# ------------------------------------------------------------------ mutations
def _mutate(dev, s, what):
    if what == "drop_massless":
        dev.Ceff = dev._mat(s.Cf)
    elif what == "ade_zeta":
        L = dev.groups[0]["lorentz"][0]
        L["zeta"] *= 1.01
    elif what == "lts_buffer":
        dev.lts["kappa"] = [0.0] * len(dev.lts["kappa"])
    elif what == "pml_swap":
        f = dev.pml["fs"]
        f["kn"], f["sn"] = f["sn"] * 0 + 1.0, f["sn"] * 0
    elif what == "te_ade_den":
        dev.te_groups[0]["den"] *= 1.001
    elif what == "cap_a1":  # fused SpMV + capacitive kernel (no ADE groups)
        dev.a1 = dev.a1 * (1 + 1e-3)


MUTATIONS = [("3d_abc_massless_dispersive", "drop_massless"), ("tm_abc_lossy_dispersive", "ade_zeta"),
             ("3d_pmc_drude", "ade_zeta"), ("tm_lts", "lts_buffer"), ("tm_pml", "pml_swap"),
             ("te_abc_dispersive", "te_ade_den"), ("3d_pec", "cap_a1"), ("tm_pec", "cap_a1")]  # fmt: skip


@pytest.mark.parametrize("xp", BACKENDS)
@pytest.mark.parametrize("name,what", MUTATIONS)
def test_mutated_device_step_exceeds_bound(name, what, xp):
    s = case(name)
    dev = G.DeviceStepper(s, xp=xp)
    _mutate(dev, s, what)
    r = compare(s, dev, N_STEPS)
    print(f"mutation {what} on {name} [{xp.__name__}]: d/eps = {r['d'] / r['eps']:.2e}")
    assert not np.isfinite(r["d"]) or r["d"] > 100 * r["eps"]


# ------------------------------------------------------------------ no host sync
@pytest.mark.skipif(not HAVE_GPU, reason="needs CuPy and a GPU")
@pytest.mark.parametrize("name", ["tm_abc_lossy_dispersive", "te_pml", "tm_lts", "te_planewave",
                                  "3d_abc_massless_dispersive"])  # fmt: skip
def test_device_step_does_not_synchronize(name, monkeypatch):
    spin = cp.RawKernel(
        r"""
extern "C" __global__ void _spin(long long ns) {
    long long t0, t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0)); t = t0;
    while (t - t0 < ns) { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); }
}""",
        "_spin",
    )
    s = case(name)
    dev = G.DeviceStepper(s, xp=cp)
    dev.run(5)  # warm-up: kernel compilation, cuSPARSE buffers, tables
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
        return done

    assert not enqueue(3), "DeviceStepper.step synchronized the stream"
    # mutation: a device-to-host read inside the step must be detected
    orig = G.DeviceStepper._record

    def bad(self, t):
        float(self.S["u"][0])
        return orig(self, t)

    monkeypatch.setattr(G.DeviceStepper, "_record", bad)
    assert enqueue(1), "the sync detector missed a device-to-host copy"
