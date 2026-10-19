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


def test_sheet_port_edges_inside_wire():
    case = DipoleCase(length=0.03, freqs=np.linspace(3e9, 6e9, 5), n_ref=40,
                      n_spread=(30, 40))  # fmt: skip
    arm = CartesianDipoleArm(conformal="edges", port="sheet", n2f=False)
    assert arm.name == "cart-staircase-edges-graded-sheet"
    grid, solver, port, _, info = arm.build(case, case.radius / 2)
    s = float(np.max(grid.x[grid.x <= case.radius / np.sqrt(2) + 1e-12]))
    assert np.isclose(s, info["port_s"]) and s > 0
    nx = int(np.sum(np.abs(grid.x[:-1]) <= s + 1e-12))
    ny = int(np.sum(np.abs(grid.y[:-1]) <= s + 1e-12))
    assert port.flat_idx.size == nx * ny == 9
    (ix, iy, iz), (cx, cy, _) = grid.region_indices("E", "z", port.center, port.size)
    assert np.all(np.hypot(cx[:, None], cy[None, :]) <= case.radius + 1e-12)
    # mutation: the full circumscribed square |x|, |y| <= a includes corner
    # nodes at r = sqrt(2) a > a whose wire z edges are not PEC
    (ix_bad, iy_bad, _), (cx_bad, cy_bad, _) = grid.region_indices(
        "E", "z", (0, 0, 0), (2 * case.radius, 2 * case.radius, case.gap)
    )
    assert np.any(np.hypot(cx_bad[:, None], cy_bad[None, :]) > case.radius)
    ie = np.asarray(solver.ieps.toarray())[2 * solver.N : 3 * solver.N].reshape(
        (solver.Nx, solver.Ny, solver.Nz), order="F"
    )
    assert not np.all(ie[np.ix_(ix_bad, iy_bad, [iz[0] - 1, iz[-1] + 1])] == 0)


def _pec_z_profile(ends):
    case = DipoleCase(length=0.03, freqs=np.linspace(3e9, 6e9, 5), n_ref=40,
                      n_spread=(30, 40))  # fmt: skip
    arm = CartesianDipoleArm(
        conformal="edges", ends=ends, hz_ratio=0.45, pad_xy=0.07, pad_z=0.05,
        n_pml=4, n2f=False,
    )  # fmt: skip
    grid, solver, _, _, info = arm.build(case, 0.9 * case.radius)
    ie = np.asarray(solver.ieps.toarray())[2 * solver.N : 3 * solver.N].reshape(
        (solver.Nx, solver.Ny, solver.Nz), order="F"
    )
    i0 = int(np.argmin(np.abs(grid.x[: solver.Nx])))
    ir = int(np.argmin(np.abs(grid.x[: solver.Nx] - 0.9 * case.radius)))
    j0 = int(np.argmin(np.abs(grid.y[: solver.Ny])))
    assert np.isclose(grid.x[ir], 0.9 * case.radius)
    zmid = 0.5 * (grid.z[:-1] + grid.z[1:])[:-1]
    return case, info["hz"], zmid, ie[i0, j0, :-1] == 0, ie[ir, j0, :-1] == 0


def test_round_ends_hemisphere_profile():
    assert (
        CartesianDipoleArm(conformal="edges", ends="round").name
        == "cart-staircase-edges-graded-round"
    )
    assert (
        CartesianDipoleArm(conformal="edges", port="sheet", ends="round").name
        == "cart-staircase-edges-graded-sheet-round"
    )
    case, hz, zmid, pec_axis, pec_r = _pec_z_profile("round")
    a = case.radius
    z_cap = case.length / 2 - a + np.sqrt(a**2 - (0.9 * a) ** 2)
    # on the axis, PEC z edges reach |z| = L/2
    assert np.isclose(np.max(np.abs(zmid[pec_axis])), case.length / 2)
    # at r = 0.9 a, PEC extends only to z_cap (+ one h_z), not to L/2
    z_max_round = float(np.max(np.abs(zmid[pec_r])))
    assert z_cap - hz <= z_max_round <= z_cap + hz
    assert z_max_round < case.length / 2 - 1e-12
    # mutation: flat ends reach |z| = L/2 at r = 0.9 a and must fail this bound
    _, _, zmid_flat, _, pec_r_flat = _pec_z_profile("flat")
    z_max_flat = float(np.max(np.abs(zmid_flat[pec_r_flat])))
    assert not (z_max_flat <= z_cap + hz)


def test_b4_gap_variants_registry():
    from wakis.benchmarks import get_case
    from wakis.benchmarks.runner import Arm, run

    c1 = get_case("B4")
    c2 = get_case("B4-g2")
    c4 = get_case("B4-g4")
    assert c2.name == "B4-g2" and c2.gap == 2e-3
    assert c4.name == "B4-g4" and c4.gap == 4e-3
    assert CartesianDipoleArm(conformal="edges").supports(c4)
    assert abs(c4.reference["f_res"] - c1.reference["f_res"]) > 1e5

    class _DummyArm(Arm):
        name = "dummy"

        def run(self, case, h):
            return dict(z_in=case.reference["z_in"], cost=dict(unknowns=1, steps=1))

    rec = run(c4, _DummyArm(), 1e-3, catch=False)
    assert rec["case"] == "B4-g4"
