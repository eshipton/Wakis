"""
Conformal PEC STL solids in a non-PEC background: the conformal metrics, not
the legacy cell mask, decide which E edges carry field.

`SolverFIT3D._apply_stl_materials` zeroes ``ieps`` on the lower-corner edges
of every cell with any node inside a PEC solid (a staircase thickened by up
to one cell towards -x, -y, -z). Before the fix, `conformal.apply_conformal_
materials` kept those zeros when no dielectric solid was present and the
background was not PEC, so edges with a positive open length stayed dead and
the conformal PEC converged at first order (TM010 of a tube: +2.0 % at 24
cells per radius, local orders 0.65-1.5) instead of second order.
"""

import numpy as np
import pytest
import pyvista as pv
from scipy.constants import c, epsilon_0

import wakis
from wakis.conformal import rectangle_sheet
from wakis.sources import CurrentSource, GaussianSource

H = 0.5e-3


def _edge_arrays(solver):
    """ieps and open length L per E edge, shape (3, Nx, Ny, Nz)."""
    shape = (3, solver.Nx, solver.Ny, solver.Nz)
    ieps = np.asarray(solver.ieps.toarray())
    L = np.asarray(solver.L.toarray())
    split = lambda a: np.stack(  # noqa: E731
        [a[i * solver.N : (i + 1) * solver.N].reshape(shape[1:], order="F")
         for i in range(3)]
    )  # fmt: skip
    return split(ieps), split(L)


def _wire_solver(conformal, dielectric=False, sheet=False):
    """PEC STL cylinder (radius one cell) on the z axis, PEC box walls."""
    x = H * np.arange(-8, 9)
    solids = {
        "w": pv.Cylinder(
            center=(0, 0, 0), radius=H, height=6e-3, direction=(0, 0, 1),
            resolution=180,
        ).triangulate()
    }  # fmt: skip
    materials = {"w": "pec"}
    if dielectric:  # far from the wire: must not change the wire edges
        solids["d"] = pv.Cube(center=(2.6e-3, 2.6e-3, 0), x_length=1e-3,
                              y_length=1e-3, z_length=1e-3).triangulate()  # fmt: skip
        materials["d"] = [4.0, 1.0, 0.0]
    sheets = None
    if sheet:  # sheets alone also switch on the conformal PEC metrics
        sheets = {"s": rectangle_sheet((-3.0e-3, 3.0e-3, 0.0), (0.5e-3, 0.5e-3, 0.0))}
    grid = wakis.GridFIT3D(
        x=x, y=x.copy(), z=x.copy(), stl_solids=solids, stl_materials=materials,
        sheets=sheets, conformal=conformal, verbose=0,
    )  # fmt: skip
    return wakis.SolverFIT3D(
        grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, use_stl=True, verbose=0
    )


@pytest.mark.parametrize(
    "conformal, dielectric, sheet",
    [
        ("pec", False, False),  # the affected path before the fix
        ("all", False, False),
        ("tensor", False, False),
        (False, False, True),  # sheets only: conformal PEC metrics as well
        ("pec", True, False),  # dielectric present (was already correct)
        ("all", True, False),
    ],
)
def test_open_edges_carry_field(conformal, dielectric, sheet):
    """
    Every edge with open length L > 0 near the wire has the background
    1/eps_0, every closed edge (L = 0) has ieps = 0, and the closed z edges
    of a cross-section are the 5 nodes inside the cylinder, centred on the
    axis (the legacy mask zeroes 12, centred at (-h/2, -h/2)).
    """
    solver = _wire_solver(conformal, dielectric, sheet)
    ieps, L = _edge_arrays(solver)
    near = np.zeros(L.shape, bool)
    near[:, 4:12, 4:12, 2:14] = True  # around the wire, away from the extras
    open_ = (L > 0) & near
    assert np.count_nonzero(open_) > 0
    assert np.count_nonzero(ieps[open_] == 0) == 0
    np.testing.assert_allclose(ieps[open_], 1 / epsilon_0, rtol=1e-12)
    assert np.all(ieps[L == 0] == 0)

    x = np.asarray(solver.grid.x[:-1])
    closed = np.argwhere(L[2, :, :, 8] == 0)
    assert len(closed) == 5
    centroid = x[closed].mean(axis=0)
    np.testing.assert_allclose(centroid, 0.0, atol=1e-9)


def test_nonconformal_mask_unchanged():
    """
    Without conformal metrics the legacy cell mask is kept: 12 z edges of the
    cross-section are dead, biased towards -x and -y (documents the upstream
    behaviour; changing it would change all staircase results).
    """
    solver = _wire_solver(False)
    ieps, _ = _edge_arrays(solver)
    x = np.asarray(solver.grid.x[:-1])
    dead = np.argwhere(ieps[2, :, :, 8] == 0)
    assert len(dead) == 12
    np.testing.assert_allclose(x[dead].mean(axis=0), -H / 2, atol=1e-9)


# ------------------------------------------------- TM010 of a PEC tube (STL)
TUBE_A = 0.01
X01 = 2.404825557695773  # first zero of J0
TUBE_CENTER = np.array([0.00037, -0.00023])  # cuts the cells irregularly


class _Probe:
    def __init__(self, solver, point):
        coords = [solver.grid.x[:-1], solver.grid.y[:-1], solver.grid.z[:-1]]
        coords[2] = coords[2] + solver.grid.dz / 2
        self.index = (
            *(int(np.argmin(abs(q - p))) for q, p in zip(coords, point)),
            "z",
        )
        self.values = []

    def update(self, solver, t):
        self.values.append(float(solver.E[self.index]))


def _pencil(signal, dt, n_modes=16):
    """Frequencies and amplitudes of damped exponentials (matrix pencil)."""
    n = len(signal)
    Y = np.lib.stride_tricks.sliding_window_view(signal, n // 3 + 1)
    _, s, vh = np.linalg.svd(Y, full_matrices=False)
    m = min(n_modes, int(np.sum(s > 1e-9 * s[0])))
    v = vh[:m].T
    z = np.linalg.eigvals(np.linalg.pinv(v[:-1]) @ v[1:])
    amp = np.linalg.lstsq(np.vander(z, n, increasing=True).T, signal, rcond=None)[0]
    return np.angle(z) / (2 * np.pi * dt), np.abs(amp)


def _tube_tm010_error(n, use_gpu=False):
    """
    Relative TM010 error of the hollow (radius a, n cells per radius) of a
    PEC STL tube in vacuum, two cells high between PEC walls (no z
    variation). Leapfrog time dispersion removed, so the error is spatial.
    """
    h = TUBE_A / n
    ncell = 2 * n + 6
    half = ncell * h / 2
    surf = (
        pv.CylinderStructured(
            radius=[TUBE_A, TUBE_A + 2 * h], height=10 * h,
            center=(*TUBE_CENTER, 0), direction=(0, 0, 1),
            theta_resolution=360, z_resolution=2,
        ).extract_surface().triangulate()
    )  # fmt: skip
    cx, cy = TUBE_CENTER
    grid = wakis.GridFIT3D(
        cx - half, cx + half, cy - half, cy + half, -h, h, ncell, ncell, 2,
        stl_solids={"tube": surf}, stl_materials={"tube": "pec"},
        conformal="pec", verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, use_stl=True,
        use_gpu=use_gpu, verbose=0,
    )  # fmt: skip
    f0 = X01 * c / (2 * np.pi * TUBE_A)
    pulse = GaussianSource(frequency=f0, fwidth=0.25 * f0)
    src = CurrentSource(
        pulse, component="z", center=(*(TUBE_CENTER + np.array([0.13, 0.07]) * TUBE_A), 0)
    )
    probe = _Probe(solver, (*(TUBE_CENTER + np.array([-0.11, 0.17]) * TUBE_A), 0.0))
    Nt = int((pulse.end_time + 30 / f0) / solver.dt)
    solver.emsolve(Nt, source=src, monitors=[probe])
    start = int(np.ceil(pulse.end_time / solver.dt))
    step = max(1, int(1 / (6 * f0 * solver.dt)))
    f, amp = _pencil(np.array(probe.values[start::step]), step * solver.dt)
    keep = np.abs(f - f0) < 0.2 * f0
    f_res = f[keep][np.argmax(amp[keep])]
    f_space = 2 / solver.dt * np.sin(np.pi * f_res * solver.dt) / (2 * np.pi)
    return f_space / f0 - 1


def test_tube_tm010_second_order(use_gpu):
    """
    TM010 of a conformal PEC STL tube converges at second order: observed
    -2.32e-3 / -5.46e-4 at 8 / 16 cells per radius (order 2.09). With the
    legacy mask left in ieps the errors were +5.8e-2 / +2.8e-2 (order 1.05),
    worse than the plain staircase (+4.8e-2 / +2.2e-2).
    """
    e8, e16 = (_tube_tm010_error(n, use_gpu) for n in (8, 16))
    order = np.log(abs(e8) / abs(e16)) / np.log(2)
    assert order > 1.7, (e8, e16, order)
    assert abs(e16) < 1e-3, e16


# ------------------------------------------------------ exact adjoint (PEC only)
@pytest.mark.parametrize("bc", ["pec", "cpml"])
def test_pec_only_adjoint_dot_product(bc):
    """
    The PEC-only conformal path (background materials on the open edges,
    ieps = 0 on the closed ones) keeps the transposed step of
    `AdjointStepper` exact: <U x, y> = <x, U^T y> to float64 rounding.
    """
    from wakis.inverse import AdjointStepper

    grid = wakis.GridFIT3D(
        -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 14, 12, 12,
        stl_solids={"ball": pv.Sphere(radius=0.043, center=(0.021, -0.013, 0.008))},
        stl_materials={"ball": "pec"}, conformal="pec", verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid, bc_low=[bc] * 3, bc_high=[bc] * 3, use_stl=True, n_pml=4, verbose=0
    )
    # the fixed path: open edges near the ball carry the background material
    ieps, L = _edge_arrays(solver)
    ieps, L = ieps[:, :-1, :-1, :-1], L[:, :-1, :-1, :-1]  # no ghost edges
    assert np.all(ieps[L > 0] > 0) and np.all(ieps[L == 0] == 0)
    solver.one_step()
    stepper = AdjointStepper(solver)
    rng = np.random.default_rng(2)
    keys = stepper.zero_state().keys()

    def random_state():
        return {
            k: rng.standard_normal(v.shape) for k, v in stepper.zero_state().items()
        }

    x = random_state()
    stepper.set_state(x)
    solver.one_step()
    ux = stepper.get_state()
    y = random_state()
    uty = stepper.step(y)
    lhs = sum(np.dot(ux[k], y[k]) for k in keys)
    rhs = sum(np.dot(x[k], uty[k]) for k in keys)
    scale = sum(np.abs(ux[k]) @ np.abs(y[k]) for k in keys)
    assert abs(lhs - rhs) < 1e-13 * scale
