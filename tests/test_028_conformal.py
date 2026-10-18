"""
Conformal FIT: partially filled cells for PEC (Dey-Mittra metrics with a
stabilizing flux redistribution), anisotropic subpixel averaging of
dielectrics and conformal thin PEC sheets with the knife-edge correction.

Resonant frequencies are measured in the time domain (ring-down of a pulsed
current, matrix pencil fit) and compared with analytic values. The leapfrog
time dispersion is removed with sin(w dt / 2) = w_s dt / 2, so that the
errors are those of the spatial discretization.
"""

import numpy as np
import pytest
import pyvista as pv
from scipy.constants import c
from scipy.optimize import brentq
from scipy.special import jv, jvp, yv, yvp

import wakis
from wakis.conformal import rectangle_sheet
from wakis.inverse import AdjointStepper
from wakis.sources import CurrentSource, GaussianSource


class _Probe:
    """Time series of one E component at the sample nearest to `point`."""

    def __init__(self, solver, point, component):
        coords = [solver.grid.x[:-1], solver.grid.y[:-1], solver.grid.z[:-1]]
        a = "xyz".index(component)
        coords[a] = coords[a] + [solver.grid.dx, solver.grid.dy, solver.grid.dz][a] / 2
        self.index = (
            *(int(np.argmin(abs(q - p))) for q, p in zip(coords, point)),
            component,
        )
        self.values = []

    def update(self, solver, t):
        self.values.append(float(solver.E[self.index]))


def _pencil(signal, dt, n_modes=24):
    """Frequencies and amplitudes of damped exponentials (matrix pencil)."""
    n = len(signal)
    rows = n // 3
    Y = np.lib.stride_tricks.sliding_window_view(signal, rows + 1)
    _, s, vh = np.linalg.svd(Y, full_matrices=False)
    m = min(n_modes, int(np.sum(s > 1e-9 * s[0])))
    v = vh[:m].T
    z = np.linalg.eigvals(np.linalg.pinv(v[:-1]) @ v[1:])
    amp = np.linalg.lstsq(np.vander(z, n, increasing=True).T, signal, rcond=None)[0]
    return np.angle(z) / (2 * np.pi * dt), np.abs(amp)


def _ring_down(solver, f0, sources, probes, periods=25, band=0.12):
    """
    Spatial-discretization frequency of the strongest resonance within
    `band` of `f0` for each probe, after a Gaussian pulse at `f0`.
    """
    pulse = GaussianSource(frequency=f0, fwidth=0.25 * f0)
    sources = [
        CurrentSource(pulse, component=comp, center=pos) for comp, pos in sources
    ]
    monitors = [_Probe(solver, pos, comp) for comp, pos in probes]
    Nt = int((pulse.end_time + periods / f0) / solver.dt)
    solver.emsolve(Nt, source=sources, monitors=monitors)
    start = int(np.ceil(pulse.end_time / solver.dt))
    step = max(1, int(1 / (6 * f0 * solver.dt)))
    out = []
    for m in monitors:
        f, amp = _pencil(np.array(m.values[start::step]), step * solver.dt)
        keep = np.abs(f - f0) < band * f0
        f_res = f[keep][np.argmax(amp[keep])]
        out.append(2 / solver.dt * np.sin(np.pi * f_res * solver.dt) / (2 * np.pi))
    return out


# ------------------------------------------------------------- conformal PEC
SPHERE_A = 0.1
SPHERE_CENTER = np.array([0.0071, -0.0043, 0.0029])  # cuts the cells irregularly
X_TM11 = 2.743707269992265  # first zero of d/dx [x j1(x)]


def _sphere_cavity_frequency(n, conformal, use_gpu):
    """Lowest TM mode of a PEC sphere (n cells across the diameter)."""
    h = 2 * SPHERE_A / n
    half = (n + 4) * h / 2
    lo, hi = SPHERE_CENTER - half, SPHERE_CENTER + half
    sphere = pv.Sphere(
        radius=SPHERE_A, center=SPHERE_CENTER, theta_resolution=300, phi_resolution=300
    )
    grid = wakis.GridFIT3D(
        lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], n + 4, n + 4, n + 4,
        stl_solids={"cavity": sphere},
        stl_materials={"cavity": "vacuum"},
        conformal=conformal,
        verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["pec"] * 3,
        bc_high=["pec"] * 3,
        bg="pec",
        use_stl=True,
        use_gpu=use_gpu,
        verbose=0,
    )
    f0 = X_TM11 * c / (2 * np.pi * SPHERE_A)
    a = SPHERE_A
    (f,) = _ring_down(
        solver,
        f0,
        sources=[("z", tuple(SPHERE_CENTER + [0.2 * a, 0.1 * a, 0.0]))],
        probes=[("z", tuple(SPHERE_CENTER + [-0.15 * a, 0.25 * a, 0.1 * a]))],
        band=0.2,
    )
    return f / f0 - 1


def test_pec_sphere_cavity(use_gpu):
    """
    TM11 resonance of a PEC sphere (bg='pec' around a vacuum sphere STL,
    x = 2.7437 c / (2 pi a)) at 16, 24 and 32 cells across the diameter
    (17-33 cells per wavelength). The staircased cell masks converge at
    first order with errors of 5-12 %; the conformal metrics converge at
    second order (observed 3.1e-3, 9.2e-4, 4.8e-4; fitted order ~2.6).
    """
    n = np.array([16, 24, 32])
    stair = np.array([_sphere_cavity_frequency(k, False, use_gpu) for k in n])
    conf = np.array([_sphere_cavity_frequency(k, "pec", use_gpu) for k in n])
    order = -np.polyfit(np.log(n), np.log(np.abs(conf)), 1)[0]

    assert np.all(np.abs(stair) > 0.03)
    assert abs(conf[-1]) < 1e-3
    assert abs(conf[-1]) < abs(stair[-1]) / 30
    assert order > 1.6


# ------------------------------------------------------ conformal dielectric
CAV_B, ROD_A, ROD_EPS = 0.1, 0.05, 4.0
CAV_X0 = 0.0123  # off the grid symmetry


def _rod_cavity_tm0():
    """Lowest E_z (TM0) mode of a PEC circular cavity with a coaxial rod."""

    def det(k0):
        k1 = k0 * np.sqrt(ROD_EPS)
        # E_z = A J0(k1 r) inside, B J0(k0 r) + C Y0(k0 r) outside, E_z(b) = 0,
        # E_z and dE_z/dr continuous at r = a
        return np.linalg.det(
            [
                [0, jv(0, k0 * CAV_B), yv(0, k0 * CAV_B)],
                [jv(0, k1 * ROD_A), -jv(0, k0 * ROD_A), -yv(0, k0 * ROD_A)],
                [
                    k1 * jvp(0, k1 * ROD_A),
                    -k0 * jvp(0, k0 * ROD_A),
                    -k0 * yvp(0, k0 * ROD_A),
                ],
            ]
        )

    return brentq(det, 10.0, 16.0) * c / (2 * np.pi)


def _rod_cavity_frequency(n, conformal, f0, use_gpu):
    """2D (one periodic cell along z) rod-loaded cavity, n cells across 2.4 b."""
    h = 2.4 * CAV_B / n
    length = 20 * h

    def cylinder(radius):
        return pv.Cylinder(
            center=(CAV_X0, 0, 0), direction=(0, 0, 1), radius=radius,
            height=length, resolution=720,
        ).triangulate()  # fmt: skip

    grid = wakis.GridFIT3D(
        CAV_X0 - n * h / 2, CAV_X0 + n * h / 2, -n * h / 2, n * h / 2, -h / 2, h / 2,
        n, n, 1,
        stl_solids={"cavity": cylinder(CAV_B), "rod": cylinder(ROD_A)},
        stl_materials={"cavity": "vacuum", "rod": [ROD_EPS, 1.0, 0.0]},
        conformal=conformal,
        verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["pec", "pec", "periodic"],
        bc_high=["pec", "pec", "periodic"],
        bg="pec",
        use_stl=True,
        use_gpu=use_gpu,
        verbose=0,
    )
    (f,) = _ring_down(
        solver,
        f0,
        sources=[("z", (CAV_X0 + 0.013, 0.007, 0.0))],
        probes=[("z", (CAV_X0 - 0.021, 0.011, 0.0))],
        periods=30,
    )
    return f / f0 - 1


def test_dielectric_rod_cavity(use_gpu):
    """
    TM0 (E_z) mode of a PEC circular cavity (radius b) loaded with a coaxial
    rod (radius b/2, eps_r = 4); analytic value from the Bessel-function
    matching. Both runs use conformal PEC; E is parallel to the curved
    dielectric interface, where the subpixel average is the arithmetic mean.
    With the majority (staircase) dielectric ('pec') the error converges at
    first order (observed 2e-2, 5e-3, 1.6e-3 at 24, 48, 96 cells across 2.4 b),
    with the anisotropic average ('all') at second order or better (observed
    1.2e-4, 3.5e-5, 3e-6).
    """
    f0 = _rod_cavity_tm0()
    n = np.array([24, 48, 96])
    stair = np.array([_rod_cavity_frequency(k, "pec", f0, use_gpu) for k in n])
    kottke = np.array([_rod_cavity_frequency(k, "all", f0, use_gpu) for k in n])
    order = -np.polyfit(np.log(n), np.log(np.abs(kottke)), 1)[0]

    assert np.all(np.abs(kottke) < np.abs(stair) / 10)
    assert abs(kottke[1]) < 1e-4
    assert abs(stair[-1]) > 5e-4
    assert order > 1.8


# --------------------------------------------------------------- thin sheets
FIN_LX, FIN_LZ, FIN_Z0, FIN_XE = 0.1, 0.06, 0.03, 0.051
# Fine-mesh references of the two lowest fin-dependent modes (PEC box with a
# zero-thickness fin from x = 0 to x = 51 mm at mid-height, invariant along
# y): Richardson extrapolation of the node-aligned staircased fin at 0.5 and
# 0.25 mm (first order in the cell size), from the discrete eigenvalues.
FIN_F_EXZ = 1.1510739e9  # E in the (x, z) plane: charge at the fin edge
FIN_F_EY = 3.6353644e9  # E_y along the fin edge: current along the edge


def _fin_frequencies(h, correction, use_gpu):
    sheet = rectangle_sheet(
        (FIN_XE / 2 - FIN_LX, 0.0, FIN_Z0), (FIN_XE + 2 * FIN_LX, 1.0, 0.0)
    )
    grid = wakis.GridFIT3D(
        0, FIN_LX, -h / 2, h / 2, 0, FIN_LZ,
        int(round(FIN_LX / h)), 1, int(round(FIN_LZ / h)),
        sheets={"fin": sheet},
        sheet_edge_correction=correction,
        verbose=0,
    )  # fmt: skip
    out = []
    for f0, comp in ((FIN_F_EXZ, "x"), (FIN_F_EY, "y")):
        solver = wakis.SolverFIT3D(
            grid,
            bc_low=["pec", "periodic", "pec"],
            bc_high=["pec", "periodic", "pec"],
            use_gpu=use_gpu,
            verbose=0,
        )
        (f,) = _ring_down(
            solver,
            f0,
            sources=[(comp, (0.063, 0.0, 0.037))],
            probes=[(comp, (0.071, 0.0, 0.041))],
            periods=30,
            band=0.05,
        )
        out.append(f)
    return np.array(out) / [FIN_F_EXZ, FIN_F_EY] - 1


def test_thin_sheet_edge(use_gpu):
    """
    PEC box with a zero-thickness fin (sheet) ending at x = 51 mm, on a 2 mm
    mesh (edge half-way between nodes) and a 1 mm mesh (edge on a node). The
    plain conformal coverage acts as if the fin edge were 0.15-0.37 cells
    further out (first-order errors 2-6e-3). With the knife-edge correction
    the edge sits at its physical position for both alignments: errors below
    3e-4 (charge at the edge) and 8e-4 (current along the edge), i.e. the
    edge position within 0.004 and 0.04 cells.
    """
    plain = np.array([_fin_frequencies(h, False, use_gpu) for h in (2e-3, 1e-3)])
    corrected = np.array([_fin_frequencies(h, True, use_gpu) for h in (2e-3, 1e-3)])

    assert np.all(np.abs(plain) > 1.5e-3)
    assert np.all(np.abs(corrected[:, 0]) < 3e-4)
    assert np.all(np.abs(corrected[:, 1]) < 8e-4)
    assert np.all(np.abs(corrected) < np.abs(plain) / 3)


# ---------------------------------------------- stability and adjoint (exact)
def _mixed_solver(bc, use_gpu=False):
    """Conformal PEC ball, dielectric rod and a sheet in a 14 x 12 x 12 box."""
    grid = wakis.GridFIT3D(
        -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 14, 12, 12,
        stl_solids={
            "ball": pv.Sphere(radius=0.043, center=(0.021, -0.013, 0.008)),
            "rod": pv.Cylinder(
                center=(-0.05, 0.03, 0), direction=(1, 1, 0.3), radius=0.02,
                height=0.07,
            ),
        },
        stl_materials={"ball": "pec", "rod": [3.0, 1.0, 0.0]},
        sheets={"plate": rectangle_sheet((0.0, 0.0, -0.05), (0.083, 0.061, 0.0))},
        conformal=True,
        verbose=0,
    )  # fmt: skip
    return wakis.SolverFIT3D(
        grid, bc_low=[bc] * 3, bc_high=[bc] * 3, use_stl=True, n_pml=4,
        use_gpu=use_gpu, verbose=0,
    )  # fmt: skip


def test_conformal_long_run_stable(use_gpu):
    """
    A lossless PEC box with a conformal PEC ball, a dielectric rod and a
    sheet keeps its field energy for 20000 steps after a pulse at the
    automatic time step (cfln times the Lanczos stability limit of the
    conformal operators): the energy stays bounded without trend.
    """
    solver = _mixed_solver("pec", use_gpu)
    pulse = GaussianSource(frequency=2e9, fwidth=1e9)
    src = CurrentSource(pulse, component="z", center=(-0.03, 0.04, 0.05))
    n0 = int(pulse.end_time / solver.dt) + 1
    solver.emsolve(n0, source=src)
    energies = []
    for _ in range(40):
        solver.emsolve(500)
        energies.append(solver.field_energy())
    energies = np.array(energies)
    # field_energy pairs E and H of different time levels with the full
    # cell volumes, so it oscillates by ~10 %; an instability would grow
    # exponentially
    assert np.all(np.isfinite(energies))
    assert energies.max() < 1.4 * energies.min()
    assert 0.95 < energies[-10:].mean() / energies[:10].mean() < 1.05


@pytest.mark.parametrize("bc", ["pec", "cpml"])
def test_conformal_adjoint_dot_product(bc):
    """
    The conformal metrics (open lengths, stabilized non-diagonal inverse
    areas, sheet-edge factors) only change the operator coefficients, so the
    transposed step of `AdjointStepper` stays exact: <U x, y> = <x, U^T y>
    to float64 rounding, with PEC walls and with CPML.
    """
    solver = _mixed_solver(bc)
    solver.one_step()
    stepper = AdjointStepper(solver)
    rng = np.random.default_rng(1)
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


def test_box_metrics_exact():
    """
    A PEC box with faces between the nodes: the open edge lengths of the
    conformal solver are exact (the geometry is computed from planar
    sections of the triangulated surface), e.g. the total open length of
    the x edges equals the domain total minus the box chords.
    """
    grid = wakis.GridFIT3D(
        0, 1, 0, 1, 0, 1, 10, 12, 8,
        stl_solids={"box": pv.Box(bounds=(0.13, 0.71, 0.22, 0.9, 0.05, 0.63))},
        stl_materials={"box": "pec"},
        conformal="pec",
        verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, use_stl=True, verbose=0
    )
    y, z = grid.y[:-1], grid.z[:-1]
    inside = np.outer((y > 0.22) & (y < 0.9), (z > 0.05) & (z < 0.63))
    open_x = solver.L.to_matrix("x")[:10].sum(axis=0)  # per (y, z) node line
    np.testing.assert_allclose(open_x, 1.0 - 0.58 * inside, atol=1e-6)


@pytest.mark.slow
def test_patch_template_thin_sheet(use_gpu):
    """
    FR-4 patch template (2.45 GHz, 20 cells per wavelength) with the patch as
    a conformal sheet of its physical size and no empirical edge correction:
    the resonance lies 2.4 % below the transmission-line model, i.e. at the
    Delta -> 0 extrapolation of the staircased patch (-2.3 %, physics guide),
    while the uncorrected staircased patch is 7.3 % below.
    """
    from wakis.templates import patch_antenna, patch_results

    sim = patch_antenna(cells_per_wavelength=20, thin_sheet=True, use_gpu=use_gpu)
    sim.run(energy_decay=1e-4)
    result = patch_results(sim)
    assert abs(result["resonance_error"] + 0.023) < 0.005
    assert result["s11_min_db"] < -10
