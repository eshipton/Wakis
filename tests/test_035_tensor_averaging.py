"""
Full-tensor subpixel averaging of the permittivity (conformal='tensor' and
MaterialGrid(smoothing='tensor')).

The resonances are measured in the time domain (ring-down of a pulsed
current, matrix pencil fit) as in test_028_conformal, with the leapfrog time
dispersion removed by sin(w dt / 2) = w_s dt / 2.

Run the MPI test with:
mpiexec -n 2 python -m pytest -v tests/test_035_tensor_averaging.py -k mpi
(also -n 3, 4, 6, 8: 24 z-cells split evenly)
"""

import numpy as np
import pytest
import pyvista as pv
from scipy.constants import c
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn

import wakis
from wakis.inverse import AdjointSimulation, AdjointStepper, MaterialGrid
from wakis.sources import CurrentSource, GaussianSource

try:
    from mpi4py import MPI

    MPI_SIZE = MPI.COMM_WORLD.Get_size()
except ImportError:
    MPI, MPI_SIZE = None, 1


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


def _ring_down(solver, f0, sources, probes, periods=30, band=0.03):
    """Spatial-discretization frequency of the strongest resonance near f0."""
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
    return np.array(out), monitors


# ------------------------------------------------ 2D rod in a square PEC box
# Rod (eps_r = 6, radius R/2) off the grid centre in the square PEC box
# [-R, R]^2. Reference: Delaunay/Voronoi triangle FIT with mesh nodes on the
# interface (second order), Richardson-extrapolated from h = 0.01 R and
# 0.005 R (nfit-study/tensor2d.py, relative uncertainty ~1e-6). k R of the
# lowest TE (E in the plane) modes: 1.2415893902, 1.2420438035 (dipole pair),
# 1.9746767234 (n = 0, E tangential to the interface), 2.0603692919; lowest
# TM mode (E_z): 1.0916772683.
ROD_R, ROD_EPS = 0.1, 6.0
ROD_CENTER = (0.0031, -0.0017)
ROD_KR_TE0 = 1.9746767234
ROD_KR_TM1 = 1.0916772683


def _rod_solver(n, conformal, use_gpu=False):
    """2D box (one periodic cell along z) with n x n cells."""
    h = 2 * ROD_R / n
    rod = pv.Cylinder(
        center=(*ROD_CENTER, 0), direction=(0, 0, 1), radius=ROD_R / 2,
        height=20 * h, resolution=2880,
    ).triangulate()  # fmt: skip
    grid = wakis.GridFIT3D(
        -ROD_R, ROD_R, -ROD_R, ROD_R, -h / 2, h / 2, n, n, 1,
        stl_solids={"rod": rod},
        stl_materials={"rod": [ROD_EPS, 1.0, 0.0]},
        conformal=conformal,
        verbose=0,
    )  # fmt: skip
    return wakis.SolverFIT3D(
        grid,
        bc_low=["pec", "pec", "periodic"],
        bc_high=["pec", "pec", "periodic"],
        use_stl=True,
        use_gpu=use_gpu,
        verbose=0,
    )


def _rod_te0_error(n, conformal, use_gpu):
    f0 = ROD_KR_TE0 / ROD_R * c / (2 * np.pi)
    solver = _rod_solver(n, conformal, use_gpu)
    (f,), _ = _ring_down(
        solver,
        f0,
        sources=[("x", (0.013, 0.071, 0.0))],
        probes=[("y", (-0.067, 0.021, 0.0))],
    )
    return f / f0 - 1


def test_rod_te_mode(use_gpu):
    """
    n = 0 TE mode of the dielectric rod (E tangential to the interface but
    crossing the cell faces obliquely everywhere), at 32 and 64 cells across
    the box (21 and 42 cells per wavelength in the rod). The diagonal Kottke
    average ('all') converges at first order (observed +6.6e-3, +3.9e-3);
    the full tensor at second order between these meshes (observed -2.1e-3,
    -5.0e-4) and about 8x more accurate at 64 cells (discrete eigenvalues of
    the same operators agree to 1e-6).
    """
    n = np.array([32, 64])
    diag = np.array([_rod_te0_error(k, "all", use_gpu) for k in n])
    tensor = np.array([_rod_te0_error(k, "tensor", use_gpu) for k in n])
    order = -np.log(abs(tensor[1] / tensor[0])) / np.log(2)

    assert abs(tensor[1]) < abs(diag[1]) / 5
    assert abs(tensor[1]) < 1e-3
    assert order > 1.5
    assert abs(diag[1]) > 2e-3


def test_rod_tm_unchanged(use_gpu):
    """
    TM (E_z only): the interface normal of the rod has no z component, so
    the tensor couples no E_z edge and the run is identical to 'all' (the
    E_z time series agree exactly); the mode is within 1e-4 of the
    reference at 32 cells.
    """
    f0 = ROD_KR_TM1 / ROD_R * c / (2 * np.pi)
    out = []
    for conformal in ("all", "tensor"):
        solver = _rod_solver(32, conformal, use_gpu)
        f, (probe,) = _ring_down(
            solver,
            f0,
            sources=[("z", (0.013, 0.007, 0.0))],
            probes=[("z", (-0.021, 0.011, 0.0))],
            band=0.2,
        )
        out.append((f[0], np.array(probe.values)))
    assert np.array_equal(out[0][1], out[1][1])
    assert abs(out[1][0] / f0 - 1) < 1e-4


# ------------------------------------- 3D dielectric sphere in a PEC sphere
SPH_B, SPH_A, SPH_EPS = 0.1, 0.05, 6.0
SPH_CENTER = np.array([0.0071, -0.0043, 0.0029])


def _riccati(n, x, deriv=False):
    """Riccati-Bessel x j_n(x) and x y_n(x) (or their derivatives)."""
    if deriv:
        return (
            spherical_jn(n, x) + x * spherical_jn(n, x, derivative=True),
            spherical_yn(n, x) + x * spherical_yn(n, x, derivative=True),
        )
    return x * spherical_jn(n, x), x * spherical_yn(n, x)


def _sphere_tm1():
    """
    Lowest TM_1 mode of a dielectric sphere (radius a) concentric in a PEC
    sphere (radius b). With the Debye potential u(r) (H_t ~ u, E_t ~ u'/eps)
    u'(b) = 0 at the wall and u, u'/eps are continuous at r = a.
    """

    def det(k0):
        k1 = k0 * np.sqrt(SPH_EPS)
        dj, dy = _riccati(1, k0 * SPH_B, deriv=True)
        A, B = dy, -dj
        j0, y0 = _riccati(1, k0 * SPH_A)
        dj0, dy0 = _riccati(1, k0 * SPH_A, deriv=True)
        u0, du0 = A * j0 + B * y0, k0 * (A * dj0 + B * dy0)
        u1 = _riccati(1, k1 * SPH_A)[0]
        du1 = k1 * _riccati(1, k1 * SPH_A, deriv=True)[0]
        return SPH_EPS * du0 * u1 - du1 * u0

    return brentq(det, 15.0, 25.0) * c / (2 * np.pi)


def _sphere_error(n, conformal, use_gpu):
    """TM_1 resonance, n cells across the cavity diameter."""
    h = 2 * SPH_B / n
    half = (n + 4) * h / 2
    lo, hi = SPH_CENTER - half, SPH_CENTER + half

    def sphere(r):
        return pv.Sphere(
            radius=r, center=SPH_CENTER, theta_resolution=400, phi_resolution=400
        )

    grid = wakis.GridFIT3D(
        lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], n + 4, n + 4, n + 4,
        stl_solids={"cavity": sphere(SPH_B), "core": sphere(SPH_A)},
        stl_materials={"cavity": "vacuum", "core": [SPH_EPS, 1.0, 0.0]},
        conformal=conformal,
        verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, bg="pec", use_stl=True,
        use_gpu=use_gpu, verbose=0,
    )  # fmt: skip
    f0 = _sphere_tm1()
    a = SPH_A
    f, _ = _ring_down(
        solver,
        f0,
        sources=[("z", tuple(SPH_CENTER + [0.3 * a, 0.2 * a, 0.1 * a]))],
        probes=[("z", tuple(SPH_CENTER + [-0.25 * a, 0.35 * a, 0.15 * a]))],
    )
    return f[0] / f0 - 1


@pytest.mark.slow
def test_sphere_tm_mode(use_gpu):
    """
    TM_1 mode of a dielectric sphere (eps_r = 6, radius b/2) in a PEC sphere
    (radius b, conformal PEC in both runs): E is radial at the poles of the
    interface and tangential at its equator, so it crosses the curved
    interface at every angle. Observed at 24, 32 and 48 cells across 2b
    (6-12 cells per wavelength in the dielectric): diagonal ('all')
    -9.6e-3, -7.1e-3, -4.8e-3 (first order); tensor -8.2e-5, -1.2e-5,
    -3.9e-5, i.e. 100-600x more accurate. The remaining tensor error is at
    the level of the conformal PEC wall error and of the splitting of the
    threefold degenerate mode by the grid (~1e-4).
    """
    n = np.array([24, 32, 48])
    diag = np.array([_sphere_error(k, "all", use_gpu) for k in n])
    tensor = np.array([_sphere_error(k, "tensor", use_gpu) for k in n])
    assert np.all(np.abs(tensor) < np.abs(diag) / 20)
    assert np.all(np.abs(tensor) < 2e-4)
    assert np.all(np.abs(diag) > 3e-3)


# ---------------------------------------------- stability, adjoint, gradient
def _mixed_solver(bc, use_gpu=False):
    """Dielectric ball and oblique rod (tensor), a PEC ball and a sheet."""
    grid = wakis.GridFIT3D(
        -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 14, 12, 12,
        stl_solids={
            "ball": pv.Sphere(radius=0.043, center=(0.021, -0.013, 0.008)),
            "rod": pv.Cylinder(
                center=(-0.05, 0.03, 0), direction=(1, 1, 0.3), radius=0.02,
                height=0.07,
            ),
            "pec": pv.Sphere(radius=0.02, center=(-0.05, -0.06, 0.05)),
        },
        stl_materials={"ball": [8.0, 1.0, 0.0], "rod": [3.0, 1.0, 0.0],
                       "pec": "pec"},
        sheets={"plate": wakis.conformal.rectangle_sheet(
            (0.0, 0.0, -0.05), (0.083, 0.061, 0.0))},
        conformal="tensor",
        verbose=0,
    )  # fmt: skip
    return wakis.SolverFIT3D(
        grid, bc_low=[bc] * 3, bc_high=[bc] * 3, use_stl=True, n_pml=4,
        use_gpu=use_gpu, verbose=0,
    )  # fmt: skip


def test_tensor_long_run_stable(use_gpu):
    """
    A lossless PEC box with tensor-averaged dielectric ball and rod (eps_r 8
    and 3), a conformal PEC ball and a sheet keeps its field energy for
    20000 steps after a pulse at the automatic time step (cfln times the
    Arnoldi stability limit of the tensor operators): bounded without trend.
    """
    solver = _mixed_solver("pec", use_gpu)
    assert solver.tensor_operator.rows.size > 0
    pulse = GaussianSource(frequency=2e9, fwidth=1e9)
    src = CurrentSource(pulse, component="z", center=(-0.03, 0.04, 0.05))
    src2 = CurrentSource(pulse, component="x", center=(0.03, -0.02, 0.01))
    solver.emsolve(int(pulse.end_time / solver.dt) + 1, source=[src, src2])
    energies = []
    for _ in range(40):
        solver.emsolve(500)
        energies.append(solver.field_energy())
    energies = np.array(energies)
    # field_energy uses the diagonal permittivity and pairs E and H of
    # different time levels, so it oscillates; an instability would grow
    # exponentially
    assert np.all(np.isfinite(energies))
    assert energies.max() < 1.5 * energies.min()
    assert 0.9 < energies[-10:].mean() / energies[:10].mean() < 1.1


@pytest.mark.parametrize("bc", ["pec", "cpml"])
def test_tensor_adjoint_dot_product(bc):
    """
    The tensor correction E' = E* + K'(E* - E) on the coupled edges is
    linear and local, and `AdjointStepper` applies its exact transpose:
    <U x, y> = <x, U^T y> to float64 rounding, with PEC walls and CPML.
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


DES_L, DES_N, DES_F0 = 0.2, 30, 2e9


def _design_solver(bc, use_gpu=False):
    h = DES_L / DES_N
    grid = wakis.GridFIT3D(
        -DES_L / 2, DES_L / 2, -DES_L / 2, DES_L / 2, -h / 2, h / 2,
        DES_N, DES_N, 1, verbose=0,
    )  # fmt: skip
    return wakis.SolverFIT3D(
        grid, bc_low=[bc, bc, "periodic"], bc_high=[bc, bc, "periodic"],
        n_pml=6, use_gpu=use_gpu, verbose=0,
    )  # fmt: skip


def _design_objective(out):
    Ex, Ey = out["probe"]["Ex"], out["probe"]["Ey"]
    a = 1e20
    F = a * np.sum(np.abs(Ex) ** 2 + 0.5 * np.abs(Ey) ** 2)
    return F, {"probe": {"Ex": 2 * a * Ex, "Ey": a * Ey}}


@pytest.mark.parametrize("bc", ["pec", "cpml"])
def test_tensor_smoothing_gradient(bc, use_gpu):
    """
    MaterialGrid(smoothing='tensor') on a 9 x 9 node design (a smoothed disk
    plus noise, eps 1 to 4) in a 2D TE run (E in the plane, so the
    off-diagonal couplings matter): the adjoint gradient through the fill
    fractions and the density-gradient normals matches central finite
    differences with h = 1e-5 to 1e-6 relative (truncation error of the
    difference quotient; the adjoint itself is exact).
    """
    source = CurrentSource(
        GaussianSource(DES_F0, 1e9), component="x", center=(-0.05, 0.013, 0.0)
    )
    monitors = {
        "probe": wakis.DFTMonitor(
            [DES_F0, 1.2 * DES_F0], components=["Ex", "Ey"], center=(0.06, -0.01, 0.0)
        )
    }
    design = MaterialGrid(
        center=(0, 0, 0), size=(0.08, 0.08, 0.0), shape=(9, 9, 1),
        eps_min=1.0, eps_max=4.0, smoothing="tensor",
    )  # fmt: skip
    sim = AdjointSimulation(
        lambda: _design_solver(bc, use_gpu), [source], monitors, design, Nt=350
    )
    u = np.linspace(-1, 1, 9)
    X, Y = np.meshgrid(u, u, indexing="ij")
    rho = 0.5 + 0.5 * np.tanh(4 * (0.6 - np.hypot(X - 0.1, Y + 0.05)))
    rho = rho + 0.05 * np.random.default_rng(0).standard_normal(rho.shape)
    rho = np.clip(rho, 0.02, 0.98)[:, :, None]

    F, G = _design_objective(sim.forward(rho))
    assert sim._solver.tensor_operator.rows.size > 0
    grad = sim.vjp(G)
    h = 1e-5
    for k in [(4, 4, 0), (2, 5, 0), (6, 3, 0), (5, 7, 0)]:
        step = np.zeros_like(rho)
        step[k] = h
        Fp, _ = _design_objective(sim.forward(rho + step))
        Fm, _ = _design_objective(sim.forward(rho - step))
        fd = (Fp - Fm) / (2 * h)
        assert abs(grad[k] - fd) < 1e-6 * abs(fd), (k, grad[k], fd)


def test_tensor_smoothing_limits():
    """
    A uniform density has no interface normal: the smoothed tensor is the
    isotropic arithmetic mean, identical to the plain MaterialGrid; a
    dispersive design cannot be tensor-smoothed.
    """
    rho = np.full((5, 5, 1), 0.3)
    ieps = []
    for smoothing in (None, "tensor"):
        solver = _design_solver("pec")
        design = MaterialGrid(
            center=(0, 0, 0), size=(0.08, 0.08, 0.0), shape=(5, 5, 1),
            eps_min=1.0, eps_max=4.0, smoothing=smoothing,
        )  # fmt: skip
        design.apply(solver, rho)
        ieps.append(solver.ieps.toarray())
    np.testing.assert_allclose(ieps[0], ieps[1], rtol=1e-12)
    with pytest.raises(NotImplementedError):
        MaterialGrid((0, 0, 0), (0.1, 0.1, 0), (3, 3, 1), smoothing="tensor",
                     material=wakis.dispersive.DispersiveMaterial(eps_inf=2.0))  # fmt: skip


# ---------------------------------------------------------------------- MPI
@pytest.mark.skipif(MPI_SIZE < 2, reason="needs mpiexec -n >= 2")
def test_mpi_tensor_matches_serial():
    """
    With the z-slab decomposition the tensor couplings reach one z layer
    into the ghost layers; after the correction the E ghost layers are
    exchanged again. The gathered fields after 200 steps equal the serial
    run (to 1e-8 relative) at the same dt. The automatic dt is the minimum of the
    ranks' stability estimates: safe, and within a few percent of the
    serial one.
    """

    def run(use_mpi, dt=None):
        kw = {"use_mpi": True} if use_mpi else {}
        grid = wakis.GridFIT3D(
            -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 12, 12, 24,
            stl_solids={"ball": pv.Sphere(radius=0.05, center=(0.011, -0.013, 0.008))},
            stl_materials={"ball": [6.0, 1.0, 0.0]},
            conformal="tensor", verbose=0, **kw,
        )  # fmt: skip
        solver = wakis.SolverFIT3D(
            grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, use_stl=True,
            dt=dt, verbose=0, **kw,
        )  # fmt: skip
        if dt is None:
            return solver.dt
        pulse = GaussianSource(frequency=2e9, fwidth=1e9)
        src = CurrentSource(pulse, component="x", center=(0.02, 0.01, 0.006))
        solver.emsolve(200, source=src)
        if use_mpi:
            return solver.mpi_gather_asField("E")
        return solver.E

    dt_mpi = run(True)
    E_mpi = run(True, dt_mpi)
    if MPI.COMM_WORLD.Get_rank() != 0:
        return
    # The minimum of the per-slab Arnoldi estimates is at most a few percent
    # below the whole-grid limit
    assert 0.9 < dt_mpi / run(False) < 1 + 1e-3
    E_ser = run(False, dt_mpi)
    a, b = E_mpi.toarray(), E_ser.toarray()
    # Each rank computes the conformal geometry of its slab: with 2 ranks the
    # fill fractions differ in the last digits and the fields by ~1.5e-9
    # (the same with conformal='all'); 3e-15 with 3 to 8 ranks
    assert np.max(np.abs(a - b)) < 1e-8 * np.max(np.abs(b))
