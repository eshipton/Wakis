"""
Regression of the (accelerated) conformal setup against the implementation
before the performance work (develop c7a4c6d).

`tests/data/conformal_regression_n14.npz` holds, for four geometries on a
14^3 grid (PEC-background sphere cavity 'all', dielectric sphere 'tensor',
rotated PEC box 'pec', tilted dielectric cylinder + PEC ball + thin sheet
'all'), the open edge lengths and face areas inside PEC, the dual volume
fractions and interface normals, ieps, sigma, the merged-face operators,
the tensor couplings and the stable time step, as computed by the original
code (`benchmarks/conformal_regression.py --save`). The geometry is exact
arithmetic on the same triangles, so the new code must reproduce every
array to round-off (here: bitwise on the reference machine; 1e-12 relative
allowed for other platforms). The time step is an iterative Lanczos/ARPACK
estimate (relative tolerance 1e-3 on the residual, results scattered by
~1e-5 between ARPACK runs) and is compared to 1e-4.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

import wakis
from wakis import _lanczos, conformal, tensor_averaging

HERE = Path(__file__).parent
REF = HERE / "data" / "conformal_regression_n14.npz"


def _harness():
    path = HERE.parent / "benchmarks" / "conformal_regression.py"
    spec = importlib.util.spec_from_file_location("conformal_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conformal_setup_matches_reference():
    harness = _harness()
    ref = dict(np.load(REF))
    new = harness.arrays(14)
    report, worst, worst_dt = harness.compare(ref, new)
    assert worst <= 1e-12, report
    assert worst_dt <= 1e-4, report


def _tensor_solver(n=12):
    sphere = pv.Sphere(radius=0.06, center=(0.0071, -0.0043, 0.0029))
    grid = wakis.GridFIT3D(
        -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, n, n, n,
        stl_solids={"core": sphere}, stl_materials={"core": [4.0, 1.0, 0.0]},
        conformal="tensor", verbose=0,
    )  # fmt: skip
    return wakis.SolverFIT3D(
        grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, use_stl=True, verbose=0
    )


def test_lanczos_matches_arpack_tightly():
    """
    The symmetric Lanczos estimate of lambda_max(B A) (P K form, P = Ds N
    itDa) equals the Arnoldi estimate on B A itself when both are converged
    tightly; with the tensor couplings dropped from P (a wrong operator) it
    does not.
    """
    solver = _tensor_solver()
    op = solver.tensor_operator
    lanczos = tensor_averaging.max_stable_dt(solver, op, tol=1e-9)
    arpack = tensor_averaging.max_stable_dt(solver, op, tol=1e-9, method="arpack")
    assert abs(lanczos / arpack - 1) < 1e-9
    wrong = _lanczos.stable_dt(solver, None, tol=1e-9)
    assert abs(wrong / arpack - 1) > 1e-6
    diag = conformal.max_stable_dt(solver, tol=1e-9)
    diag_arpack = conformal.max_stable_dt(solver, tol=1e-9, method="arpack")
    assert abs(diag / diag_arpack - 1) < 1e-9


def test_parallel_products_bitwise():
    """Threaded row-block CSR products equal the serial ones bit for bit."""
    solver = _tensor_solver(16)
    C = solver.C.tocsr().astype(float)
    x = np.random.default_rng(1).standard_normal(C.shape[1])
    serial = C @ x
    for threads in (1, 3, 8):
        par = _lanczos.ParallelCSR(C, threads)
        par.cuts = [(int(a), int(b)) for a, b in zip(
            np.linspace(0, C.shape[0], threads + 1).astype(int)[:-1],
            np.linspace(0, C.shape[0], threads + 1).astype(int)[1:])]  # fmt: skip
        assert np.array_equal(par @ x, serial)


def test_gpu_lanczos_matches_cpu():
    pytest.importorskip("cupy")
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    solver = _tensor_solver()
    op = solver.tensor_operator
    cpu = _lanczos.stable_dt(solver, op, device="cpu")
    gpu = _lanczos.stable_dt(solver, op, device="gpu")
    assert abs(gpu / cpu - 1) < 1e-10
