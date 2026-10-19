"""
Fused GPU time stepping (``SolverFIT3D(..., fused=True)``, `wakis.gpu_fused`).

The fused kernels must reproduce the CuPy path:

* full runs (sources, step, monitors) agree to round-off on every boundary
  type and feature the fused step supports (relative 1e-11 after a few hundred
  steps: the unfused path differs only by the order of floating-point
  operations, ~1e-16 per step, growing at most linearly);
* the source and running-DFT kernels are bitwise identical to the CuPy
  expressions (compiled without FMA contraction);
* mutated kernel data (a psi coefficient, a dropped magnetic current) is
  detected by the same comparison, so it can fail.

GPU only (``pytest --gpu``).
"""

import numpy as np
import pytest
from scipy.constants import c

import wakis
from wakis.mode_solver import ModeMonitor, ModeSource
from wakis.sources import CurrentSource, GaussianSource, MagneticCurrentSource

F0 = 3e9
RTOL = 1e-11


@pytest.fixture
def gpu_only(use_gpu):
    if not use_gpu:
        pytest.skip("GPU only (pytest --gpu)")


def _solver(case, fused):
    grid = wakis.GridFIT3D(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 18, 16, 14, verbose=0)
    kw = dict(n_pml=4, use_gpu=True, fused=fused, verbose=0)
    if case == "pec":
        kw.update(bc_low=["pec"] * 3, bc_high=["pec"] * 3)
    elif case == "pec-lossy":
        kw.update(bc_low=["pec"] * 3, bc_high=["pec"] * 3, bg=[2.0, 1.0, 0.05])
    elif case == "periodic":
        kw.update(bc_low=["periodic"] * 3, bc_high=["periodic"] * 3)
    elif case == "cpml":
        kw.update(bc_low=["cpml"] * 3, bc_high=["cpml"] * 3)
    elif case == "cpml-lossy":
        kw.update(bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, bg=[2.0, 1.0, 0.05])
    elif case == "cpml-mixed":
        kw.update(
            bc_low=["cpml", "pec", "periodic"], bc_high=["pec", "cpml", "periodic"]
        )
    elif case == "bloch-cpml":
        kw.update(
            bc_low=["bloch", "bloch", "cpml"], bc_high=["bloch", "bloch", "cpml"],
            k_bloch=(17.0, -9.0, 0.0), dtype=np.complex128,
        )  # fmt: skip
    return wakis.SolverFIT3D(grid, **kw)


def _run(case, fused, Nt=300, mutate=None):
    solver = _solver(case, fused)
    pulse = GaussianSource(F0, 0.5 * F0)
    sources = [
        CurrentSource(pulse, component="z", center=(0.01, -0.02, 0.0)),
        MagneticCurrentSource(pulse, component="x", center=(-0.03, 0.02, 0.01)),
    ]
    f = np.linspace(2e9, 4e9, 7)
    monitors = [
        wakis.DFTMonitor(f, components=("Ex", "Hy", "Jz"), size=(0.1, 0.1, 0.0)),
        wakis.FluxMonitor(f, center=(0.0, 0.0, 0.03), size=(0.12, 0.12, 0.0)),
        wakis.FluxMonitor(f, size=(0.08, 0.08, 0.08)),
    ]
    if mutate is not None:
        sources[0].update(solver, 0.0)
        solver.one_step()
        mutate(solver)
        n0 = 1
    else:
        n0 = 0
    for n in range(n0, Nt):
        t = n * solver.dt
        for s in sources:
            s.update(solver, t)
        solver.one_step()
        for m in monitors:
            m.update(solver, t)
    out = {"E": solver.E.array.get(), "H": solver.H.array.get()}
    out["dft"] = monitors[0].get_field("Ex")[0]
    out["flux_plane"] = monitors[1].get_flux()
    out["flux_box"] = monitors[2].get_flux()
    return out, solver


def _close(a, b, rtol=RTOL):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b)) <= rtol * np.max(np.abs(b))


CASES = ["pec", "pec-lossy", "periodic", "cpml", "cpml-lossy", "cpml-mixed",
         "bloch-cpml"]  # fmt: skip


@pytest.mark.parametrize("case", CASES)
def test_fused_matches_cupy_path(case, gpu_only):
    """Fields, DFT and flux spectra of fused and CuPy runs agree to round-off."""
    ref, s0 = _run(case, False)
    out, s1 = _run(case, True)
    assert not s0.fused and s1.fused
    for key in ref:
        assert _close(out[key], ref[key]), key
    # Not trivially zero: the run carries fields everywhere it is compared
    assert np.max(np.abs(ref["E"])) > 0 and np.max(np.abs(ref["flux_box"])) > 0


@pytest.mark.parametrize(
    "mutation", ["psi_c", "drop_M", "sigma"], ids=["psi-c", "drop-M", "sigma"]
)
def test_mutations_are_detected(mutation, gpu_only):
    """The comparison above fails for slightly wrong fused kernel data."""
    case = "cpml-lossy"

    def mutate(solver):
        step = solver._fused_step
        if mutation == "psi_c":
            step.e.pc *= 1 + 1e-6
        elif mutation == "drop_M":
            solver.M = None
        else:
            solver.sigma.array *= 1 + 1e-6

    ref, _ = _run(case, False)
    out, _ = _run(case, True, mutate=mutate)
    assert not all(_close(out[k], ref[k]) for k in ref)


def test_mode_source_and_monitor(gpu_only):
    """ModeSource / ModeMonitor in a CPML-terminated PEC guide, fused vs CuPy."""
    DX = 1e-3
    res = {}
    for fused in (False, True):
        grid = wakis.GridFIT3D(
            0, 20 * DX, 0, 10 * DX, 0, 80 * DX, 20, 10, 80, verbose=0
        )
        bc = ["pec", "pec", "cpml"]
        solver = wakis.SolverFIT3D(
            grid, use_gpu=True, fused=fused, bc_low=bc, bc_high=bc, n_pml=10,
            verbose=0,
        )  # fmt: skip
        fc = c / (2 * 20 * DX)
        pulse = GaussianSource(frequency=1.5 * fc, fwidth=0.1 * 1.5 * fc)
        plane = dict(size=(np.inf, np.inf, 0.0))
        src = ModeSource(pulse, center=(0, 0, 0.03), **plane)
        f = np.linspace(1.3 * fc, 1.7 * fc, 5)
        mon = ModeMonitor(f, center=(0, 0, 0.06), **plane)
        solver.emsolve(400, source=src, monitors=[mon])
        res[fused] = np.array(mon.get_amplitudes())
    assert _close(res[True], res[False])
    assert np.max(np.abs(res[False])) > 0


def test_source_and_dft_kernels_bitwise(gpu_only):
    """One source update and one DFT accumulation on the same random state are
    bit-for-bit those of the CuPy expressions."""
    import cupy as cp

    states = {}
    for fused in (False, True):
        solver = _solver("cpml", fused)
        rng = np.random.default_rng(1)  # same state for both runs
        for F in (solver.E, solver.H, solver.J):
            F.array[:] = cp.asarray(rng.standard_normal(F.array.size))
        src = CurrentSource(
            GaussianSource(F0, 0.5 * F0), component="y", center=(0, 0, 0),
            size=(0.05, 0.05, 0.0),
        )  # fmt: skip
        f = np.linspace(1e9, 5e9, 9)
        mons = [
            wakis.DFTMonitor(f, components=("Ey", "Hz", "Jx"), size=(0.1, 0.1, 0.0)),
            wakis.FluxMonitor(f, size=(0.08, 0.08, 0.08)),
        ]
        t = 37 * solver.dt
        for k in range(3):
            src.update(solver, t + k * solver.dt)
            for m in mons:
                m.update(solver, t + k * solver.dt)
        states[fused] = (
            solver.J.array.get(),
            mons[0].get_field("Hz")[0],
            np.concatenate([np.ravel(term["H_acc"].get()) for term in mons[1]._terms]),
        )
    for a, b in zip(states[True], states[False]):
        assert np.array_equal(a, b)


def test_fused_falls_back_without_gpu_support():
    """fused=True on a CPU solver keeps the standard step."""
    grid = wakis.GridFIT3D(0, 1, 0, 1, 0, 1, 4, 4, 4, verbose=0)
    solver = wakis.SolverFIT3D(grid, fused=True, verbose=0)
    assert not solver.fused


# ---------------------------------------------------------------------------
# Adjoint: exact transpose of the fused step, unchanged gradients


def _dot_test(solver, stepper, rng, supports=None):
    """Relative error of <U x, y> = <x, U^H y> for random states."""
    xp = stepper.xp
    keys = list(stepper.zero_state().keys())
    psi = [k for k in keys if k.startswith("psi")]
    supports = supports or [(keys, keys)] + ([(psi, keys), (keys, psi)] if psi else [])
    errs = []
    for xs, ys in supports:

        def rand(support):
            out = {}
            for k, v in stepper.zero_state().items():
                a = rng.standard_normal(v.shape) * (k in support)
                if np.iscomplexobj(v):
                    a = a + 1j * rng.standard_normal(v.shape) * (k in support)
                out[k] = xp.asarray(a)
            return out

        x = rand(xs)
        stepper.set_state(x)
        solver.one_step()
        Ux = {k: v.copy() for k, v in stepper.get_state().items()}
        y = rand(ys)
        Uty = stepper.step(y)
        lhs, rhs = stepper.inner(Ux, y), stepper.inner(x, Uty)
        scale = stepper.inner({k: abs(v) for k, v in Ux.items()},
                              {k: abs(v) for k, v in y.items()}).real  # fmt: skip
        errs.append(abs(lhs - rhs) / scale)
    return max(errs)


@pytest.mark.parametrize("case", CASES)
def test_fused_transposed_step(case, gpu_only):
    """Dot-product test of the fused transposed step (GPU, float64): exact to
    rounding, also with x or y supported on the CPML psi only."""
    from wakis.inverse import AdjointStepper

    solver = _solver(case, True)
    solver.one_step()
    stepper = AdjointStepper(solver)
    assert stepper.fused
    err = _dot_test(solver, stepper, np.random.default_rng(0))
    assert err < 1e-12, err


@pytest.mark.parametrize("target", ["t_val", "pc", "loss"])
def test_fused_transpose_mutation_fails(target, gpu_only):
    """A wrong entry of the transposed data breaks the dot-product test."""
    from wakis.inverse import AdjointStepper

    solver = _solver("cpml-lossy", True)
    solver.one_step()
    stepper = AdjointStepper(solver)
    if target == "t_val":
        stepper._halves["E"].t_val[7] *= 1.001
    elif target == "pc":
        # A copy read by the transposed step only (the forward keeps its own)
        import copy

        half = copy.copy(stepper._halves["H"])
        half.pc = half.pc * (1 + 1e-3)
        stepper._halves["H"] = half
    else:
        stepper.loss *= 1 + 1e-6
    err = _dot_test(solver, stepper, np.random.default_rng(0))
    assert err > 100 * 1e-12, err  # 100x the tolerance of the exact transpose


def _adjoint_sim(fused, Nt=250):
    from wakis.inverse import AdjointSimulation, MaterialGrid

    def make_solver():
        grid = wakis.GridFIT3D(-0.15, 0.15, -0.15, 0.15, -0.005, 0.005, 30, 30, 1,
                               verbose=0)  # fmt: skip
        return wakis.SolverFIT3D(
            grid, bc_low=["cpml", "cpml", "periodic"],
            bc_high=["cpml", "cpml", "periodic"], n_pml=6, use_gpu=True,
            fused=fused, verbose=0,
        )  # fmt: skip

    src = CurrentSource(GaussianSource(F0, 0.5 * F0), component="z",
                        center=(-0.08, 0.0, 0.0))  # fmt: skip
    f = np.linspace(2e9, 4e9, 4)
    monitors = {
        "probe": wakis.DFTMonitor(
            f, components=("Ez", "Hy"), center=(0.08, 0, 0), size=(0.0, 0.1, 0.0)
        ),  # fmt: skip
        "flux": wakis.FluxMonitor(f, center=(0.1, 0, 0), size=(0.0, 0.2, np.inf)),
    }
    design = MaterialGrid((0.0, 0.0, 0.0), (0.1, 0.1, 0.0), (6, 6, 1), 1.0, 4.0)
    return AdjointSimulation(make_solver, [src], monitors, design, Nt)


def test_fused_adjoint_gradient_unchanged(gpu_only):
    """Outputs and dF/drho of AdjointSimulation agree between the fused and the
    CuPy path to round-off (1e-10 relative: ~250 steps of 1e-16 rounding
    differences in the order of operations, propagated through forward and
    adjoint runs), and the fused gradient matches a central finite
    difference of the fused forward run."""
    rng = np.random.default_rng(3)
    res = {}
    for fused in (False, True):
        sim = _adjoint_sim(fused)
        rho = np.random.default_rng(0).uniform(size=sim.design.shape)
        out = sim.forward(rho)
        G = {k: {c: rng.standard_normal(v.shape) + 1j * rng.standard_normal(v.shape)
                 for c, v in d.items()} for k, d in out.items()}  # fmt: skip
        rng = np.random.default_rng(3)
        res[fused] = (out, sim.vjp(G), G, sim, rho)
    (o0, g0, G, _, rho), (o1, g1, _, sim, _) = res[False], res[True]
    for k in o0:
        for comp in o0[k]:
            assert _close(o1[k][comp], o0[k][comp], 1e-10), (k, comp)
    assert _close(g1, g0, 1e-10)
    assert np.max(np.abs(g0)) > 0

    def F(r):
        o = sim.forward(r)
        return sum(np.sum(np.real(np.conj(G[k][c]) * o[k][c])) for k in o for c in o[k])

    d = np.zeros_like(rho)
    d.flat[7] = 1.0
    h = 1e-4
    fd = (F(rho + h * d) - F(rho - h * d)) / (2 * h)
    assert abs(fd - g1.flat[7]) < 1e-5 * np.max(np.abs(g1))


# ---------------------------------------------------------------------------
# CUDA graph loops (emsolve and AdjointSimulation replay captured steps)


def _emsolve_run(fused, graphs=True, stop=None, tfsf=False, Nt=260):
    if tfsf:
        dx, n = 50e-9, 48
        h, hz = n / 2 * dx, 2 * dx
        grid = wakis.GridFIT3D(-h, h, -h, h, -hz, hz, n, n, 4, verbose=0)
        solver = wakis.SolverFIT3D(
            grid, bc_low=["cpml", "cpml", "periodic"],
            bc_high=["cpml", "cpml", "periodic"], n_pml=8, use_gpu=True,
            fused=fused, verbose=0,
        )  # fmt: skip
        pulse = GaussianSource(frequency=250e12, fwidth=100e12)
        tf = 10 * dx
        sources = [wakis.TFSFSource(pulse, center=(0, 0, 0), size=(2 * tf, 2 * tf, None),
                                    direction="+x", polarization="z")]  # fmt: skip
        f = np.linspace(150e12, 350e12, 11)
        monitors = [
            wakis.FluxMonitor(f, size=(2 * tf + 6 * dx, 2 * tf + 6 * dx, 2 * dx))
        ]
    else:
        solver = _solver("cpml-lossy", fused)
        pulse = GaussianSource(F0, 0.5 * F0)
        sources = [
            CurrentSource(pulse, component="z", center=(0.01, -0.02, 0.0)),
            MagneticCurrentSource(pulse, component="x", center=(-0.03, 0.02, 0.01)),
        ]
        f = np.linspace(2e9, 4e9, 7)
        monitors = [
            wakis.DFTMonitor(f, components=("Ex", "Hy"), size=(0.1, 0.1, 0.0)),
            wakis.FluxMonitor(f, size=(0.08, 0.08, 0.08)),
        ]
    solver.use_graphs = graphs
    solver.emsolve(Nt, source=sources, monitors=monitors,
                   stop_when_energy_decayed=stop)  # fmt: skip
    out = {"E": solver.E.array.get(), "H": solver.H.array.get()}
    out["flux"] = monitors[-1].get_flux()
    if tfsf:
        out["series"] = sources[0]._series[:, : sources[0]._n_steps].get()
    return out, solver


@pytest.mark.parametrize("tfsf", [False, True], ids=["currents", "tfsf"])
def test_emsolve_graphs_match(tfsf, gpu_only):
    """emsolve with CUDA graphs (fused) = fused per-step loop = CuPy path."""
    ref, _ = _emsolve_run(False, tfsf=tfsf)
    per_step, _ = _emsolve_run(True, graphs=False, tfsf=tfsf)
    graph, _ = _emsolve_run(True, graphs=True, tfsf=tfsf)
    for key in ref:
        assert _close(per_step[key], ref[key]), key
        # Same kernels, same order: the graph replay is bitwise the per-step run
        assert np.array_equal(graph[key], per_step[key]), key


def test_emsolve_graphs_energy_stop(gpu_only):
    """The energy-decay stop happens at the same step with graphs."""
    _, s0 = _emsolve_run(False, stop=(1e-3, 50), Nt=4000)
    _, s1 = _emsolve_run(True, stop=(1e-3, 50), Nt=4000)
    assert s0.Nt == s1.Nt < 4000


def test_adjoint_graphs_match_per_step(gpu_only):
    """AdjointSimulation with CUDA graphs: forward outputs and gradient are
    bitwise those of the fused per-step loops."""
    rng = np.random.default_rng(5)
    res = []
    for graphs in (False, True):
        sim = _adjoint_sim(True)
        sim.use_graphs = graphs
        rho = np.random.default_rng(0).uniform(size=sim.design.shape)
        out = sim.forward(rho)
        rng = np.random.default_rng(5)
        G = {k: {c: rng.standard_normal(v.shape) + 0j for c, v in d.items()}
             for k, d in out.items()}  # fmt: skip
        res.append((out, sim.vjp(G)))
    (o0, g0), (o1, g1) = res
    for k in o0:
        for comp in o0[k]:
            assert np.array_equal(o1[k][comp], o0[k][comp])
    assert np.array_equal(g1, g0)


def _tensor_rod(fused, n=40):
    import pyvista as pv

    r = 0.1
    h = 2 * r / n
    rod = pv.Cylinder(center=(0.0031, -0.0017, 0), direction=(0, 0, 1),
                      radius=r / 2, height=20 * h, resolution=360).triangulate()  # fmt: skip
    grid = wakis.GridFIT3D(-r, r, -r, r, -h / 2, h / 2, n, n, 1,
                           stl_solids={"rod": rod},
                           stl_materials={"rod": [6.0, 1.0, 0.0]},
                           conformal="tensor", verbose=0)  # fmt: skip
    return wakis.SolverFIT3D(grid, bc_low=["cpml", "cpml", "periodic"],
                             bc_high=["cpml", "cpml", "periodic"], n_pml=6,
                             use_stl=True, use_gpu=True, fused=fused, verbose=0)  # fmt: skip


def test_tensor_averaging_fused(gpu_only):
    """Conformal 'tensor' solver: fused hooks (and CUDA graphs) = CuPy path,
    and the transposed step of the fused tensor solver is exact."""
    from wakis.inverse import AdjointStepper

    res = {}
    for fused in (False, True):
        solver = _tensor_rod(fused)
        assert solver.tensor_operator.rows.size > 0
        src = CurrentSource(GaussianSource(1e9, 0.5e9), component="z",
                            center=(0.06, 0.05, 0.0))  # fmt: skip
        mon = wakis.DFTMonitor([0.8e9, 1.2e9], components=("Ez",), size=(0.1, 0.1, 0))
        solver.emsolve(300, source=src, monitors=[mon])
        res[fused] = (solver.E.array.get(), mon.get_field("Ez")[0])
    for a, b in zip(res[True], res[False]):
        assert _close(a, b)
    solver = _tensor_rod(True)
    solver.one_step()
    stepper = AdjointStepper(solver)
    assert stepper.fused and stepper.tensor
    assert _dot_test(solver, stepper, np.random.default_rng(2)) < 1e-12


def test_ade_dispersive_fused(gpu_only):
    """Lorentz + Drude + Debye + sigma block: fused ADE hooks reproduce the
    CuPy ADE (fields and DFT to round-off), the transposed step of the fused
    dispersive solver is exact, and the graph loop runs with them."""
    from wakis.dispersive import Debye, DispersiveMaterial, Drude, Lorentz
    from wakis.inverse import AdjointStepper

    mat = DispersiveMaterial(
        2.0,
        [Lorentz(1.5, 2 * np.pi * 3.5e9, 2 * np.pi * 0.4e9),
         Drude(2 * np.pi * 2e9, 2 * np.pi * 0.5e9), Debye(0.8, 1e-10)],
        sigma=0.02,
    )  # fmt: skip

    def make(fused):
        solver = _solver("cpml", fused)
        solver.add_dispersive_material(mat, center=(0.03, 0, 0), size=(0.06, 0.1, 0.1))
        return solver

    res = {}
    for fused in (False, True):
        solver = make(fused)
        src = CurrentSource(GaussianSource(F0, 0.5 * F0), component="z",
                            center=(-0.04, 0.0, 0.0))  # fmt: skip
        mon = wakis.DFTMonitor([2e9, 3e9], components=("Ez",), size=(0.1, 0.1, 0))
        solver.emsolve(300, source=src, monitors=[mon])
        g = solver._dispersive_groups[0]
        res[fused] = (solver.E.array.get(), mon.get_field("Ez")[0],
                      g["lorentz"][0][3].get(), g["debye"][0][2].get())  # fmt: skip
        if fused:
            assert solver.one_step is solver._fused_step  # hooks, no wrapper
    for a, b in zip(res[True], res[False]):
        assert _close(a, b)
    solver = make(True)
    solver.one_step()
    stepper = AdjointStepper(solver)
    assert stepper.fused and stepper.dispersive
    assert _dot_test(solver, stepper, np.random.default_rng(4)) < 1e-12


def test_estimate_fused_step():
    """wakis.estimate models the measured fused GPU steps (studies/gpu-perf:
    0.45 ms plain and 0.68 ms CPML at 65^3 on an L4) and drops the CPML
    launch floor (7.1 ms) of the CuPy path."""
    from wakis.estimate import estimate

    def step_ms(cpml, fused):
        e = estimate((65, 65, 65), steps=1000, backend="gpu", cpml=cpml, fused=fused)
        return 1e3 * e.step_seconds

    assert abs(step_ms(False, True) - 0.45) < 0.05
    assert abs(step_ms(True, True) - 0.68) < 0.07
    assert step_ms(True, False) > 7.0
