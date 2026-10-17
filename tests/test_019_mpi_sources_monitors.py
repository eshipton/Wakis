import numpy as np
import pytest

import wakis
from wakis.sources import CurrentSource, GaussianSource

# Run with:
# mpiexec -n 2 python -m pytest --color=yes -v tests/test_019_mpi_sources_monitors.py

MPI = pytest.importorskip("mpi4py.MPI")


def _run(use_mpi):
    """Lossy PEC box with sources and monitors on the MPI subdomain interfaces."""
    kw = {"use_mpi": True} if use_mpi else {}
    grid = wakis.GridFIT3D(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 20, 20, 24, **kw)
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["pec"] * 3,
        bc_high=["pec"] * 3,
        bg=[1.0, 1.0, 0.02],
        verbose=0,
        **kw,
    )
    src_time = GaussianSource(frequency=1.5e9, fwidth=0.5e9)
    sources = [
        # z = 0 and z = 0.05 lie on subdomain interfaces for 2, 4 and 6 ranks
        CurrentSource(src_time, component="z", center=(0.01, 0.0, 0.0)),
        CurrentSource(src_time, component="x", center=(0.0, 0.02, 0.05)),
    ]
    f = np.linspace(1e9, 2e9, 5)
    box = wakis.FluxMonitor(f, center=(0.0, 0.0, 0.01), size=(0.1, 0.1, 0.1))
    plane = wakis.FluxMonitor(f, center=(0.0, 0.0, 0.05), size=(0.2, 0.2, 0.0))
    point = wakis.DFTMonitor(f, components=["Ez"], center=(0.01, 0.0, 0.0))
    solver.emsolve(300, source=sources, monitors=[box, plane, point])

    Ez = point.get_field("Ez")[0].ravel()
    if use_mpi:
        # Only the rank owning the point holds its samples
        Ez = MPI.COMM_WORLD.allreduce(Ez if Ez.size else np.zeros(f.size, complex))
    return box.get_flux(), plane.get_flux(), Ez


@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() < 2, reason="needs mpiexec -n >= 2")
def test_mpi_matches_serial():
    """
    Sources and monitors on MPI subdomain interfaces are applied by exactly one
    rank: flux spectra and point DFTs match a serial run to rounding error.
    """
    parallel = _run(use_mpi=True)
    if MPI.COMM_WORLD.Get_rank() != 0:
        return
    serial = _run(use_mpi=False)
    for p, s in zip(parallel, serial):
        np.testing.assert_allclose(p, s, rtol=1e-10, atol=1e-10 * np.abs(s).max())
