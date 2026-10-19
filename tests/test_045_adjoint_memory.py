"""
Memory ladder of the exact adjoint (`wakis.inverse.memory`).

- binomial checkpointing replays the forward run bitwise, so its gradient
  must equal the stored-increment gradient *exactly* for every physics case
  (loss, dispersion, CPML, lumped elements, Bloch, conformal/tensor
  averaging, dispersive tensor designs, TF/SF, mode sources);
- time reversal agrees with it to round-off on lossless problems, falls
  back to checkpointing (bitwise) otherwise, and breaks when one boundary
  record is corrupted;
- the schedule is the dynamic-programming optimum;
- the measured memory high-water mark matches the prediction.
"""

import sys
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pytest

import wakis
from wakis.inverse import MaterialGrid
from wakis.inverse import memory as mem
from wakis.sources import CurrentSource, GaussianSource

sys.path.insert(0, str(Path(__file__).parent))

# --------------------------------------------------------------------------
# Schedule


def _dp_cost(l, c, cache={}):
    """Minimal forward steps (recording included) to reverse l steps."""
    if l <= 1:
        return l
    if c == 0:
        return l * (l + 1) // 2
    if (l, c) not in cache:
        cache[l, c] = min(
            j + _dp_cost(l - j, c - 1) + _dp_cost(j, c) for j in range(1, l)
        )
    return cache[l, c]


def _rule_cost(l, c):
    if l <= 1:
        return l
    if c == 0:
        return l * (l + 1) // 2
    j = mem.revolve_split(l, c + 1)
    return j + _rule_cost(l - j, c - 1) + _rule_cost(j, c)


def test_revolve_schedule_is_optimal():
    """Split rule and closed-form cost equal the DP optimum (l < 120)."""
    for c in range(0, 6):
        for l in range(1, 120):
            opt = _dp_cost(l, c)
            assert _rule_cost(l, c) == opt, (l, c)
            assert mem.revolve_cost(l, c + 1) == opt, (l, c)


def test_revolve_schedule_mutation_detected():
    """A naive split (halving) is not optimal: the DP test can fail."""

    def halving(l, c):
        if l <= 1:
            return l
        if c == 0:
            return l * (l + 1) // 2
        j = l // 2
        return j + halving(l - j, c - 1) + halving(j, c)

    assert any(halving(l, 2) > _dp_cost(l, 2) for l in range(2, 60))


def test_plan_checkpoints_respects_budget():
    state, row, Nt = 1_000_000, 10_000, 20_000
    plan = mem.plan_checkpoints(Nt, state, row, budget_bytes=Nt * row)
    assert plan["mode"] == "store"
    for budget in (5e6, 2e7, 1e8):
        plan = mem.plan_checkpoints(Nt, state, row, budget)
        assert plan["mode"] == "checkpoint"
        assert plan["bytes"] <= budget
    with pytest.raises(MemoryError):
        mem.plan_checkpoints(Nt, state, row, 0.5 * state)


# --------------------------------------------------------------------------
# Physics cases (from the existing gradient tests)


def _jax_G(objective, out):
    """G = dF/dRe + 1j dF/dIm of a JAX objective of the raw outputs."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    ct = jax.grad(lambda o: objective(o))(out)
    return {n: {k: np.conj(np.asarray(v)) for k, v in c.items()} for n, c in ct.items()}


def _case(name, use_gpu):
    """(sim, function out -> G, rho, lossless) of one physics case."""
    import test_020_adjoint as t20

    if name in ("cpml", "pec-lossy"):
        if name == "cpml":
            sim = t20._scattering_sim(lambda: t20._make_2d_solver(use_gpu=use_gpu))
        else:
            sim = t20._scattering_sim(
                lambda: t20._make_2d_solver("pec", bg=(1, 1, 0.01), use_gpu=use_gpu)
            )
        rho = np.random.default_rng(1).uniform(0.2, 0.8, sim.design.shape)
        return sim, lambda out: t20._objective(out)[1], rho, name == "cpml"
    if name.startswith("tensor-"):
        import test_035_tensor_averaging as t35

        bc = name.split("-")[1]
        source = CurrentSource(
            GaussianSource(t35.DES_F0, 1e9), component="x", center=(-0.05, 0.013, 0)
        )
        monitors = {
            "probe": wakis.DFTMonitor(
                [t35.DES_F0], components=["Ex", "Ey"], center=(0.06, -0.01, 0.0)
            )
        }
        design = MaterialGrid(
            center=(0, 0, 0), size=(0.08, 0.08, 0.0), shape=(9, 9, 1),
            eps_min=1.0, eps_max=4.0, smoothing="tensor",
        )  # fmt: skip
        sim = wakis.inverse.AdjointSimulation(
            lambda: t35._design_solver(bc, use_gpu), [source], monitors, design, 250
        )
        rho = np.random.default_rng(0).uniform(0.1, 0.9, (9, 9, 1))
        return sim, lambda out: t35._design_objective(out)[1], rho, False
    if name == "network":
        import test_038_tensor_dispersive as t38
        from wakis.materials import Debye, DispersiveMaterial, Drude, Lorentz

        w = 2 * np.pi * t38.DES_F0
        mat = DispersiveMaterial(
            2.0, [Drude(1.5 * w, 0.1 * w), Lorentz(1.0, 1.2 * w, 0.05 * w),
                  Debye(0.5, 1 / w)], sigma=0.01,
        )  # fmt: skip
        source = CurrentSource(
            GaussianSource(t38.DES_F0, 1e9), component="x", center=(-0.05, 0.013, 0)
        )
        monitors = {
            "probe": wakis.DFTMonitor(
                [t38.DES_F0], components=["Ex", "Ey"], center=(0.06, -0.01, 0.0)
            )
        }
        design = MaterialGrid(
            center=(0, 0, 0), size=(0.08, 0.08, 0.0), shape=(9, 9, 1),
            eps_min=1.5, material=mat, smoothing="tensor",
        )  # fmt: skip
        sim = wakis.inverse.AdjointSimulation(
            lambda: t38._design_solver("cpml", use_gpu), [source], monitors,
            design, 250,
        )  # fmt: skip
        rho = np.random.default_rng(0).uniform(0.1, 0.9, (9, 9, 1))

        def G(out):
            Ex, Ey = out["probe"]["Ex"], out["probe"]["Ey"]
            return {"probe": {"Ex": 2e20 * Ex, "Ey": 1e20 * Ey}}

        return sim, G, rho, False
    if name == "tfsf":
        import jax
        import test_039_tfsf_planewave as t39
        from wakis.inverse import flux

        src = wakis.TFSFSource(
            GaussianSource(1.5e9, 0.6e9), size=(0.08, 0.08, 0.08), direction="+z"
        )
        monitors = {"scat": wakis.FluxMonitor([1.5e9], size=(0.11, 0.11, 0.11))}
        design = MaterialGrid(
            center=(0, 0, 0), size=(0.04, 0.04, 0.04), shape=(3, 3, 3)
        )
        sim = wakis.inverse.AdjointSimulation(
            partial(t39._solver, n=24, length=0.24, bc="cpml", n_pml=6,
                    use_gpu=use_gpu),
            [src], monitors, design, 200,
        )  # fmt: skip
        sim.output_shapes()
        rho = np.random.default_rng(0).uniform(0.3, 0.7, design.shape)

        def objective(out):
            return jax.numpy.sum(flux(out["scat"], sim.readouts["scat"])) * 1e20

        return sim, partial(_jax_G, objective), rho, True
    # test_026 cases (JAX objectives)
    import jax
    import test_026_adjoint_features as t26

    jax.config.update("jax_enable_x64", True)
    sim, objective, _ = t26._gradient_case(name, jax.numpy, use_gpu)
    if hasattr(sim, "output_shapes") and sim.readouts is None:
        sim.output_shapes()
    rho = np.random.default_rng(1).uniform(0.2, 0.8, sim.design.shape)
    lossless = name in ("flux", "near2far", "mode", "magnetic-current", "bloch")
    return sim, partial(_jax_G, objective), rho, lossless


CASES = [
    "cpml", "pec-lossy", "flux", "near2far", "mode", "magnetic-current",
    "lumped", "dispersive", "dispersive-design", "bloch", "tensor-pec",
    "tensor-cpml", "network", "tfsf",
]  # fmt: skip


def _grad(sim, rho, G_of, memory, **kw):
    sim.memory = memory
    sim.checkpoints = kw.get("checkpoints")
    sim.block = kw.get("block")
    sim.decimation = kw.get("decimation")
    sim.memory_budget_gb = kw.get("budget")
    out = sim.forward(rho)
    G = G_of(out)
    return sim.vjp(G), G


# Cases whose forward run itself is not bitwise reproducible from run to
# run (the mode solver of ModeSource/ModeMonitor starts ARPACK from a random
# vector): there, replay exactness is checked row by row within one run and
# the gradient against a separate store run to 1e-12.
NONDETERMINISTIC = {"mode"}


@pytest.mark.parametrize("case", CASES)
def test_checkpoint_gradient_is_bitwise(case, use_gpu):
    """
    Checkpoint replay restores the complete step state (fields, CPML psi,
    conduction and source currents, ADE polarizations, network state,
    lumped-element and TF/SF line state): every replayed row equals the
    forward row in every bit (`verify_replay`), the gradient equals the
    stored-increment gradient bitwise, for several slot/block settings, and
    a second `vjp` after one forward run gives the same result.
    """
    sim, G_of, rho, _ = _case(case, use_gpu)
    g_store, _ = _grad(sim, rho, G_of, "store")
    assert np.max(np.abs(g_store)) > 0
    sim.verify_replay = True
    for kw in (dict(checkpoints=3), dict(checkpoints=2, block=11),
               dict(checkpoints=5, block=1)):  # fmt: skip
        g, G = _grad(sim, rho, G_of, "checkpoint", **kw)
        assert sim._rung.mismatches == 0, (case, kw, sim._rung.mismatches)
        assert sim.memory_stats["recompute_steps"] > 0
        if case in NONDETERMINISTIC:
            err = np.max(np.abs(g - g_store)) / np.max(np.abs(g_store))
            assert err < 1e-12, (case, kw, err)
        else:
            assert np.array_equal(g, g_store), (case, kw, np.max(np.abs(g - g_store)))
        g2 = sim.vjp(G)
        assert sim._rung.mismatches == 0
        assert np.array_equal(g2, g), (case, kw, "second vjp")


@pytest.mark.parametrize("case", CASES)
def test_reverse_gradient(case, use_gpu):
    """
    Lossless cases: time reversal with recorded CPML interface layers and
    source terms matches the stored-increment gradient to round-off
    (< 1e-10 of max |grad|; observed ~1e-15 at Nt = 200-700). Other cases
    fall back to checkpointing with a warning and stay bitwise.
    """
    sim, G_of, rho, lossless = _case(case, use_gpu)
    g_store, _ = _grad(sim, rho, G_of, "store")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g, _ = _grad(sim, rho, G_of, "reverse")
    fell_back = any("checkpointing" in str(w.message) for w in caught)
    assert fell_back == (not lossless), (case, [str(w.message) for w in caught])
    if lossless:
        assert sim.memory_stats["mode"] == "reverse"
        err = np.max(np.abs(g - g_store)) / np.max(np.abs(g_store))
        print(f"{case}: reverse rel. error {err:.1e}")
        assert err < 1e-10
    elif case in NONDETERMINISTIC:
        assert np.max(np.abs(g - g_store)) / np.max(np.abs(g_store)) < 1e-12
    else:
        assert np.array_equal(g, g_store)


@pytest.mark.parametrize("key", ["H_layer", "E_layer", "J"])
def test_reverse_corrupted_record_fails(key, use_gpu):
    """
    Mutation check: scaling one recorded step of the CPML interface layer
    (H or E) or of the source current by 1.5 must break the gradient
    (relative error >> 1e-10), so the agreement above is evidence.
    """
    sim, G_of, rho, _ = _case("cpml", use_gpu)
    g_store, G = _grad(sim, rho, G_of, "store")
    sim.memory = "reverse"
    sim.forward(rho)
    rec = sim._rung.rec[key]
    nz = np.flatnonzero(np.any(np.asarray(rec.get() if hasattr(rec, "get") else rec)
                               != 0, axis=1))  # fmt: skip
    step = int(nz[len(nz) // 2])
    sim._rung.corrupt = (step, key)
    g = sim.vjp(G)
    err = np.max(np.abs(g - g_store)) / np.max(np.abs(g_store))
    print(f"corrupted {key} at step {step}: rel. error {err:.1e}")
    assert err > 1e-6


def test_checkpoint_missing_state_fails():
    """
    Mutation check of the snapshot: if CPML psi arrays are left out of the
    snapshot, replay is no longer exact and the gradient changes.
    """
    sim, G_of, rho, _ = _case("cpml", False)
    g_store, _ = _grad(sim, rho, G_of, "store")
    original = mem.SolverState.__init__

    def without_psi(self, solver, sources):
        original(self, solver, sources)
        self.psi = []

    mem.SolverState.__init__ = without_psi
    sim.verify_replay = True
    try:
        g, _ = _grad(sim, rho, G_of, "checkpoint", checkpoints=3)
    finally:
        mem.SolverState.__init__ = original
    assert sim._rung.mismatches > 0
    assert not np.allclose(g, g_store, rtol=1e-8, atol=0)


def test_checkpoint_missing_source_state_fails():
    """Mutation check: sources not snapshotted -> replayed rows differ."""
    sim, G_of, rho, _ = _case("tfsf", False)
    original = mem.SolverState.__init__

    def without_sources(self, solver, sources):
        original(self, solver, [])

    mem.SolverState.__init__ = without_sources
    sim.verify_replay = True
    try:
        _grad(sim, rho, G_of, "checkpoint", checkpoints=3)
    finally:
        mem.SolverState.__init__ = original
    assert sim._rung.mismatches > 0


def test_nyquist_is_approximate_and_not_default():
    """'nyquist' is opt-in, needs a decimation, and is only approximate."""
    sim, G_of, rho, _ = _case("cpml", False)
    assert sim.memory == "store"
    with pytest.raises(ValueError):
        wakis.inverse.AdjointSimulation(None, [], {}, sim.design, 10, memory="nyquist")
    g_store, _ = _grad(sim, rho, G_of, "store")
    g, _ = _grad(sim, rho, G_of, "nyquist", decimation=3)
    err = np.max(np.abs(g - g_store)) / np.max(np.abs(g_store))
    assert 0 < err < 1e-2
    g, _ = _grad(sim, rho, G_of, "nyquist", decimation=40)  # far below Nyquist
    assert np.max(np.abs(g - g_store)) / np.max(np.abs(g_store)) > 1e-2


def test_auto_mode_uses_budget():
    sim, G_of, rho, _ = _case("cpml", False)
    g_store, _ = _grad(sim, rho, G_of, "store")
    rows = sim.memory_stats["peak_bytes"]
    g, _ = _grad(sim, rho, G_of, "auto", budget=2 * rows / 1e9)
    assert sim.memory_stats["mode"] == "store"
    g, _ = _grad(sim, rho, G_of, "auto", budget=0.3 * rows / 1e9)
    assert sim.memory_stats["mode"] == "checkpoint"
    assert sim.memory_stats["peak_bytes"] <= 0.3 * rows
    assert np.array_equal(g, g_store)


# --------------------------------------------------------------------------
# Measured memory


def _big_design_sim(Nt):
    import test_020_adjoint as t20

    source = CurrentSource(
        GaussianSource(t20.F0, 1e9), component="z", center=(-0.08, 0.0, 0.0)
    )
    monitors = {"probe": wakis.DFTMonitor([t20.F0], components=["Ez"],
                                          center=(0.08, 0.01, 0.0))}  # fmt: skip
    design = MaterialGrid(center=(0, 0, 0), size=(0.12, 0.12, 0.0), shape=(7, 7, 1))
    return wakis.inverse.AdjointSimulation(
        lambda: t20._make_2d_solver(), [source], monitors, design, Nt
    )


def _peak(fn):
    import tracemalloc

    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    fn()
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    return peak


def test_memory_high_water_mark():
    """
    tracemalloc peak of forward + vjp: the saving of checkpointing over
    storing equals the predicted saving (stored rows minus snapshots and
    one block of rows) to within 15 %, and checkpointing uses less than a
    third of the store-mode row memory here.
    """
    Nt = 1200
    sim = _big_design_sim(Nt)
    rho = np.full(sim.design.shape, 0.5)

    def G_of(out):
        return {"probe": {"Ez": 1e20 * out["probe"]["Ez"]}}

    G = G_of(sim.forward(rho))

    def run(mode, **kw):
        sim.memory, sim.checkpoints, sim.block = mode, kw.get("c"), kw.get("b")
        sim.forward(rho)
        sim.vjp(G)

    run("store")  # warm-up (imports, caches)
    p_store = _peak(lambda: run("store"))
    stats_store = dict(sim.memory_stats)
    p_ck = _peak(lambda: run("checkpoint", c=4))
    stats_ck = dict(sim.memory_stats)
    predicted = stats_store["peak_bytes"] - stats_ck["peak_bytes"]
    measured = p_store - p_ck
    print(f"store rows {stats_store['peak_bytes'] / 1e6:.2f} MB, checkpoint "
          f"{stats_ck['peak_bytes'] / 1e6:.2f} MB; measured saving "
          f"{measured / 1e6:.2f} MB, predicted {predicted / 1e6:.2f} MB")  # fmt: skip
    assert stats_ck["peak_bytes"] < stats_store["peak_bytes"] / 3
    assert abs(measured - predicted) < 0.15 * predicted


def test_gpu_memory_high_water_mark(use_gpu):
    """Same on the GPU, with a CuPy memory hook tracking pool allocations."""
    if not use_gpu:
        pytest.skip("GPU only")
    import cupy as cp
    from cupy.cuda import memory_hook

    class Peak(memory_hook.MemoryHook):
        name = "peak"

        def __init__(self):
            self.cur = self.peak = 0
            self.sizes = {}

        def malloc_postprocess(self, device_id, size, mem_size, mem_ptr, pmem_id):
            self.sizes[mem_ptr] = mem_size
            self.cur += mem_size
            self.peak = max(self.peak, self.cur)

        def free_postprocess(self, device_id, mem_size, mem_ptr, pmem_id):
            self.cur -= self.sizes.pop(mem_ptr, mem_size)

    import test_020_adjoint as t20

    Nt = 1200
    source = CurrentSource(GaussianSource(t20.F0, 1e9), component="z",
                           center=(-0.08, 0.0, 0.0))  # fmt: skip
    monitors = {"probe": wakis.DFTMonitor([t20.F0], components=["Ez"],
                                          center=(0.08, 0.01, 0.0))}  # fmt: skip
    design = MaterialGrid(center=(0, 0, 0), size=(0.12, 0.12, 0.0), shape=(7, 7, 1))
    sim = wakis.inverse.AdjointSimulation(
        lambda: t20._make_2d_solver(use_gpu=True), [source], monitors, design, Nt,
        store_on_host=False,
    )  # fmt: skip
    rho = np.full(design.shape, 0.5)
    G = {"probe": {"Ez": 1e20 * sim.forward(rho)["probe"]["Ez"]}}
    peaks, stats = {}, {}
    for mode, c in (("store", None), ("checkpoint", 4)):
        sim.memory, sim.checkpoints = mode, c
        cp.get_default_memory_pool().free_all_blocks()
        hook = Peak()
        with hook:
            sim.forward(rho)
            sim.vjp(G)
        peaks[mode], stats[mode] = hook.peak, dict(sim.memory_stats)
    predicted = stats["store"]["peak_bytes"] - stats["checkpoint"]["peak_bytes"]
    measured = peaks["store"] - peaks["checkpoint"]
    print(f"GPU measured saving {measured / 1e6:.2f} MB, predicted "
          f"{predicted / 1e6:.2f} MB")  # fmt: skip
    assert abs(measured - predicted) < 0.25 * predicted


# --------------------------------------------------------------------------
# Fused GPU solvers (wakis.gpu_fused): 'store' (and 'auto' resolved to it)
# keeps the CUDA-graph loops; the other rungs run the generic loop, whose
# `one_step` is the fused step


@pytest.mark.parametrize("bc", ["cpml", "pec"])
def test_fused_solver_rungs(bc, use_gpu):
    """
    On fused solvers, checkpoint replay is bitwise (every replayed row equals
    the forward row) and checkpoint/reverse gradients agree with the fused
    CUDA-graph 'store' gradient to < 1e-12 (the graph loop records rows with
    its own kernels, so bit equality across the two loops is not required);
    'auto' with a large budget takes the fused store path bitwise.
    """
    if not use_gpu:
        pytest.skip("GPU only")
    import test_049_gpu_fused as t49

    sim = t49._adjoint_sim(fused=True, Nt=250)
    if bc == "pec":
        # Same problem in a closed PEC box with ohmic loss (standard step)
        def pec_solver():
            grid = wakis.GridFIT3D(-0.15, 0.15, -0.15, 0.15, -0.005, 0.005, 30, 30,
                                   1, verbose=0)  # fmt: skip
            return wakis.SolverFIT3D(
                grid, bc_low=["pec", "pec", "periodic"],
                bc_high=["pec", "pec", "periodic"], bg=[1.0, 1.0, 0.01],
                use_gpu=True, fused=True, verbose=0,
            )  # fmt: skip

        sim.make_solver = pec_solver
    rho = np.random.default_rng(3).uniform(0.2, 0.8, sim.design.shape)

    def G_of(out):
        return {"probe": {"Ez": 1e18 * out["probe"]["Ez"],
                          "Hy": 1e18 * out["probe"]["Hy"]}}  # fmt: skip

    g_store, _ = _grad(sim, rho, G_of, "store")
    assert sim._solver.fused and sim.memory_stats["mode"] == "store"
    g_auto, _ = _grad(sim, rho, G_of, "auto", budget=1e3)
    assert sim.memory_stats["mode"] == "store"
    assert np.array_equal(g_auto, g_store)
    sim.verify_replay = True
    for kw in (dict(checkpoints=3), dict(checkpoints=4, block=1)):
        g, G = _grad(sim, rho, G_of, "checkpoint", **kw)
        assert sim._rung.mismatches == 0, kw
        err = np.max(np.abs(g - g_store)) / np.max(np.abs(g_store))
        assert err < 1e-12, (kw, err)
        assert np.array_equal(sim.vjp(G), g)
    sim.verify_replay = False
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        g, _ = _grad(sim, rho, G_of, "reverse")
    assert sim.memory_stats["mode"] == ("reverse" if bc == "cpml" else "checkpoint")
    err = np.max(np.abs(g - g_store)) / np.max(np.abs(g_store))
    print(f"fused {bc}: reverse rel. error {err:.1e}")
    assert err < 1e-12
