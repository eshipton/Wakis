import pytest

from wakis.estimate import MACHINES, estimate, preflight


def test_estimate_scales_and_preflight():
    """
    Calibrated peak-memory model (benchmarks/calibrate_memory.py): held-out
    160^3 runs on an L4 VM matched within ~4 % (4.90 GB measured vs 5.00 GB
    estimated for CPU float64). Here: linear scaling, GPU memory reported,
    and the preflight check rejects jobs that do not fit the target.
    """
    small = estimate((64, 64, 64), steps=1000, safety=1.0)
    large = estimate((128, 128, 128), steps=1000, safety=1.0)
    per_cell = (large.host_bytes - small.host_bytes) / (large.cells - small.cells)
    assert 1000 < per_cell < 1200  # ~1.1 kB per cell, CPU float64

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
    assert set(MACHINES) >= {"laptop", "n2-standard-8", "g2-standard-16"}
