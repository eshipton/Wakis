"""
B4 Cartesian dipole arm geometry: the edge-wise staircase puts PEC exactly on
the wire (|z| in [g/2, L/2] on the axis), leaves the port gap free, and the
legacy STL cell masks lengthen the wire (documented failure mode).
"""

import numpy as np

from wakis.benchmarks.arms.cartesian_dipole import CartesianDipoleArm
from wakis.benchmarks.dipole import DipoleCase


def _build(conformal):
    case = DipoleCase(length=0.03, freqs=np.linspace(3e9, 6e9, 5), n_ref=40,
                      n_spread=(30, 40))  # fmt: skip
    grid, solver, *_ = CartesianDipoleArm(conformal=conformal, n2f=False).build(
        case, 0.5e-3
    )
    return case, grid, solver


def _axis_pec(conformal):
    case, grid, solver = _build(conformal)
    N = solver.Nx * solver.Ny * solver.Nz
    i = int(np.argmin(np.abs(grid.x[: solver.Nx])))
    j = int(np.argmin(np.abs(grid.y[: solver.Ny])))
    ie = np.asarray(solver.ieps.toarray())
    k = np.arange(solver.Nz)
    zmid = 0.5 * (grid.z[k] + grid.z[k + 1])
    pec = ie[2 * N + i + solver.Nx * j + solver.Nx * solver.Ny * k] == 0
    # the last z sample lies outside the domain (boundary), drop it
    return case, zmid[:-1], pec[:-1]


def _pec_zedges_in_plane(conformal, z0=7e-3):
    """Number of PEC z edges in the cross-section plane at z0 (mid-arm)."""
    case, grid, solver = _build(conformal)
    N = solver.Nx * solver.Ny * solver.Nz
    k = int(np.searchsorted(grid.z, z0)) - 1
    ie = np.asarray(solver.ieps.toarray())[2 * N : 3 * N]
    ie = ie.reshape((solver.Nx, solver.Ny, solver.Nz), order="F")[:, :, k]
    return int(np.sum(ie == 0))


def test_edgewise_staircase_wire_and_gap():
    case, zmid, pec = _axis_pec("edges")
    # an edge whose midpoint lies on the end face counts as PEC (weight 1/2)
    on_wire = (np.abs(zmid) > case.gap / 2) & (np.abs(zmid) <= case.length / 2 + 1e-12)
    assert np.all(pec[on_wire]) and not np.any(pec[~on_wire])


def test_legacy_stl_masks_thicken_the_wire():
    # a = 0.5 mm on 0.5 mm cells: edge-wise PEC = the 5 nodes with r <= a;
    # legacy cell masks (any node inside) zero the edges of every touched
    # cell, a 4x4 block
    n_edges = _pec_zedges_in_plane("edges")
    n_legacy = _pec_zedges_in_plane(False)
    assert n_edges == 5
    assert n_legacy > 2 * n_edges
    case, zmid, pec = _axis_pec(False)
    assert not np.any(pec[np.abs(zmid) < case.gap / 2])  # the gap stays free
