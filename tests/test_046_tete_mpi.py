import numpy as np
import pytest

import wakis

# Run with:
# mpiexec -n 2 python -m pytest -v tests/test_046_tete_mpi.py

MPI = pytest.importorskip("mpi4py.MPI")


def _pipe_wake(use_mpi, folder):
    """TE/TM wake of a rectangular cavity between pipes (CPML + TFSF beam)."""
    import pyvista as pv

    kw = {"use_mpi": True} if use_mpi else {}
    npml, dz, a = 8, 1e-3, 0.02
    nz = 80 + 2 * npml
    zmax = nz * dz
    solids = {
        "pipe": pv.Box(bounds=(-a / 4, a / 4, -a / 4, a / 4, -1, 1)),
        "cavity": pv.Box(bounds=(-0.4 * a, 0.4 * a, -0.4 * a, 0.4 * a,
                                 0.035, 0.06)),
    }  # fmt: skip
    grid = wakis.GridFIT3D(
        -a / 2, a / 2, -a / 2, a / 2, 0, zmax, 10, 10, nz,
        stl_solids=solids, stl_materials={"pipe": "vacuum", "cavity": "vacuum"},
        verbose=0, **kw,
    )  # fmt: skip
    wake = wakis.WakeSolver(
        q=1e-9, sigmaz=5e-3, beta=1.0, xsource=0, ysource=0, xtest=0, ytest=0,
        skip_cells=npml + 2, results_folder=folder, Ez_file=folder + "Ez.h5",
        save=False, verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid, wake, bc_low=["pec", "pec", "cpml"], bc_high=["pec", "pec", "cpml"],
        n_pml=npml, verbose=0, scheme="tete", bg="pec", use_stl=True, **kw,
    )  # fmt: skip
    # compute_plane="longitudinal" fails under MPI (pre-existing: mpi_gather
    # expects slices), so the 3x3 transverse block is saved
    solver.wakesolve(wakelength=0.04)
    return wake.WP if (not use_mpi or MPI.COMM_WORLD.Get_rank() == 0) else None


@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() < 2, reason="needs mpiexec -n >= 2")
def test_tete_mpi_matches_serial(tmp_path_factory):
    """Z-slab MPI TE/TM run (two exchanges per step) equals the serial run."""
    comm = MPI.COMM_WORLD
    folder = comm.bcast(str(tmp_path_factory.mktemp("mpi")) if comm.rank == 0 else None)
    wp_mpi = _pipe_wake(True, folder + "/par_")
    if comm.rank != 0:
        return
    wp_ser = _pipe_wake(False, folder + "/ser_")
    assert np.abs(wp_ser).max() > 1e-3
    np.testing.assert_allclose(wp_mpi, wp_ser, rtol=0, atol=1e-10 * np.abs(wp_ser).max())
