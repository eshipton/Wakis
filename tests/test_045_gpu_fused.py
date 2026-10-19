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
        grid = wakis.GridFIT3D(0, 20 * DX, 0, 10 * DX, 0, 80 * DX, 20, 10, 80, verbose=0)
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
