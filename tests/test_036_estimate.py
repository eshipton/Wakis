"""
Calibrated resource model `wakis.estimate` (benchmarks/calibrate_memory.py,
GCP g2-standard-16 with an NVIDIA L4; fresh process per run, peak RSS
``ru_maxrss``, warmed-up step times). The coefficients were fitted on 32^3,
64^3, 96^3, 128^3 and 160^3 grids; the measurements below are held-out runs
(benchmarks/validate_estimate.py prints all of them).
"""

import pytest

from wakis.estimate import (
    MACHINES,
    estimate,
    estimate_unstructured,
    interface_cells_from_area,
    preflight,
)

# Held-out measurements: (estimate kwargs, peak host bytes, s per step,
# GPU bytes or None)
HELD_OUT = [
    (dict(cells=1442897), 1.981e9, 0.0970, None),  # plain CPU 112^3
    (dict(cells=5545233, backend="gpu"), 6.674e9, 0.01695, 5.221e9),  # GPU 176^3
    (dict(cells=531441, conformal="all", interface_cells=13806), 1.167e9, 0.0330, None),
    (dict(cells=3048625, conformal="tensor", interface_cells=44673), 4.787e9, 0.2397, None),
    (dict(cells=117649, conformal="pec", interface_cells=2903), 0.569e9, 0.00517, None),
    (
        dict(cells=912673, conformal="tensor", interface_cells=19841, backend="gpu"),
        2.235e9, 0.002257, 0.9917e9,
    ),
    (dict(cells=1442897, dispersive_edges=2091144, n_poles=3), 1.981e9, 0.300, None),
]  # fmt: skip


@pytest.mark.parametrize("kwargs, host, step, gpu", HELD_OUT)
def test_estimate_held_out(kwargs, host, step, gpu):
    """
    Peak host memory within 10 % and the time per step within 30 % of
    held-out measurements (plain, conformal 'pec'/'all'/'tensor' on CPU and
    GPU, dispersive edges). The conformal peak is the setup peak (geometry,
    merged faces, Lanczos time step), which the plain model missed by up to
    20 % on these grids.
    """
    est = estimate(steps=1000, safety=1.0, **kwargs)
    assert abs(est.host_bytes / host - 1) < 0.10
    assert abs(est.step_seconds / step - 1) < 0.30
    if gpu:
        assert abs(est.gpu_bytes / gpu - 1) < 0.10
    assert est.seconds == pytest.approx(est.setup_seconds + 1000 * est.step_seconds)


def test_conformal_setup_peak_dominates():
    """Small conformal grids peak during the setup, above the plain solver."""
    plain = estimate(145_000, safety=1.0)
    conf = estimate(145_000, conformal="tensor", interface_cells=5000, safety=1.0)
    assert "conformal setup peak (host)" in conf.breakdown
    assert conf.host_bytes > 1.05 * plain.host_bytes
    assert conf.setup_seconds > plain.setup_seconds
    # A sphere of radius 0.3 m on a 1 m grid of 112^3 cells: 27 062 measured
    cells = interface_cells_from_area(4 * 3.14159 * 0.3**2, 1 / 112, "tensor")
    assert abs(cells / 27062 - 1) < 0.05


def test_estimate_unstructured():
    """2.5 kB and 52 ns per tetrahedron (held-out 2.6e5 tets: 1.062 GB, 12.7 ms)."""
    est = estimate_unstructured(264215, steps=100, safety=1.0)
    assert abs(est.host_bytes / 1.062e9 - 1) < 0.10
    assert abs(est.step_seconds / 0.0127 - 1) < 0.30
    eig = estimate_unstructured(264215, eigen=True)
    assert eig.host_bytes > 2 * est.host_bytes


def test_estimate_scales_and_preflight():
    """
    Linear scaling (~1.1 kB per cell, CPU float64), GPU memory reported, and
    the preflight check rejects jobs that do not fit the target.
    """
    small = estimate((64, 64, 64), steps=1000, safety=1.0)
    large = estimate((128, 128, 128), steps=1000, safety=1.0)
    per_cell = (large.host_bytes - small.host_bytes) / (large.cells - small.cells)
    assert 1000 < per_cell < 1200

    gpu = estimate((400, 400, 400), steps=8000, backend="gpu", cpml=True)
    assert gpu.gpu_bytes > 24e9
    assert preflight(gpu, "g2-standard-32", raise_error=False)
    with pytest.raises(MemoryError):
        preflight(gpu, "g2-standard-32")
    assert not preflight(
        estimate((128,) * 3, backend="gpu", cpml=True),
        "g2-standard-16",
        raise_error=False,
    )

    adj = estimate(
        (100, 100, 1), steps=2000, adjoint={"design_edges": 5000, "store_bytes": 8}
    )
    assert adj.breakdown["adjoint stored increments"] == 2000 * 5000 * 8
    assert set(MACHINES) >= {"laptop", "n2-standard-8", "n2-standard-16", "n2-highmem-8"}
    assert MACHINES["n2-highmem-8"] == {"ram": 64e9, "gpu": 0}
    assert not preflight(estimate((96, 96, 200), steps=5000, conformal="pec"), "n2-highmem-8", raise_error=False)
