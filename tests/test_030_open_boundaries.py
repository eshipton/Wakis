import numpy as np
from scipy.constants import c, mu_0
from scipy.special import hankel2

import wakis
from wakis.sources import CurrentSource, GaussianSource


def _solver_2d(half, n_pml, dx, use_gpu=False, dt=None):
    """
    TM (Ez) 2D simulation: one cell between PEC plates along z (the TEM mode
    of the plates is exactly the z-invariant 2D field), CPML along x and y
    outside the interior [-half, half]^2.
    """
    h = half + n_pml * dx
    N = round(2 * h / dx)
    grid = wakis.GridFIT3D(-h, h, -h, h, 0, 10 * dx, N, N, 1, verbose=0)
    return wakis.SolverFIT3D(
        grid,
        use_gpu=use_gpu,
        bc_low=["cpml", "cpml", "pec"],
        bc_high=["cpml", "cpml", "pec"],
        n_pml=n_pml,
        dt=dt,
        verbose=0,
    )


def _probe_signals(solver, probes, T):
    """Ez(t) at the probe points from a broadband line current."""
    src_time = GaussianSource(frequency=1e9, fwidth=0.5e9)
    line = CurrentSource(
        src_time, component="z", center=(-0.15, -0.15, 0.0), size=(0, 0, 1.0)
    )
    monitors = [
        wakis.resonances.ResonanceMonitor("Ez", center=p, t_start=0.0) for p in probes
    ]
    solver.emsolve(int(T / solver.dt), source=line, monitors=monitors)
    return np.array([m.signal()[1] for m in monitors]).T


def test_cpml_reflection_convergence(use_gpu):
    """
    Point (line) source near a corner of a 40 x 40 cell interior closed by
    CPML: probes along the far side of the same wall see the wall at grazing
    incidence (~77 deg from the normal), a probe near the opposite corner sees
    the corner region and one probe sits in the centre. Reflection error
    max|E - E_ref| / max|E_ref| over 6 ns against a domain large enough that
    its own boundary echoes arrive later; 10 cells per wavelength at 1.5 GHz.
    The error falls rapidly with the CPML thickness (6-8 %, 0.04-0.06 %,
    5e-6 at 5, 10, 15 cells) down to the 2-3e-6 floor of the reference.
    """
    dx, dt, T = 0.01, 2.3e-11, 6e-9
    probes = [(0.15, -0.18, 0), (0.18, 0.18, 0), (-0.18, 0.15, 0), (0.0, 0.0, 0)]
    ref = _probe_signals(_solver_2d(1.2, 10, dx, use_gpu, dt), probes, T)

    errors = []
    for n_pml in (5, 10, 15):
        E = _probe_signals(_solver_2d(0.2, n_pml, dx, use_gpu, dt), probes, T)
        errors.append(np.abs(E - ref).max(axis=0) / np.abs(ref).max(axis=0))
    errors = np.array(errors)

    assert np.all(errors[0] < 0.1)
    assert np.all(errors[1] < 1e-3)
    assert np.all(errors[2] < 2e-5)
    assert np.all(errors[1] < errors[0] / 50) and np.all(errors[2] < errors[1] / 20)


def test_line_source_hankel(use_gpu):
    """
    A z-directed line current I(f) radiates the 2D field
    Ez = -(omega mu0 I / 4) H0^(2)(k rho). Along a radial line from 2 to 38
    cells (25-37 cells per wavelength) the simulated |Ez/I| matches the
    analytic amplitude within 1 % beyond the first two cells (2.5 % at
    rho = 2 dx, where the source cell resolves the log singularity poorly),
    and |Ez| sqrt(rho) is constant to 0.5 % in the far field (k rho > 4).
    """
    dx = 0.01
    solver = _solver_2d(0.45, 15, dx, use_gpu)
    src_time = GaussianSource(frequency=1e9, fwidth=0.3e9)
    line = CurrentSource(src_time, component="z", center=(0, 0, 0), size=(0, 0, 1.0))
    f = np.array([0.8e9, 1.0e9, 1.2e9])
    radial = wakis.DFTMonitor(
        f, components=["Ez"], center=(0.2, 0.0, 0.0), size=(0.36, 0.0, 0.0)
    )
    solver.emsolve(
        int((src_time.end_time + 6e-9) / solver.dt), source=line, monitors=[radial]
    )

    E, coords = radial.get_field("Ez")
    E, rho = np.abs(E[:, :, 0, 0]), coords[0]
    k = 2 * np.pi * f / c
    analytic = (
        2 * np.pi * f[:, None] * mu_0 / 4 * np.abs(hankel2(0, np.outer(k, rho)))
    ) * np.abs(src_time.fourier_transform(f))[:, None]

    np.testing.assert_allclose(E[:, :2], analytic[:, :2], rtol=2.5e-2)
    np.testing.assert_allclose(E[:, 2:], analytic[:, 2:], rtol=1e-2)
    # |H0(x)| sqrt(x) -> sqrt(2/pi) (1 + O(1/x^2)): 0.4 % variation for x > 4
    far = np.outer(k, rho) > 4
    for i in range(f.size):
        scaled = E[i, far[i]] * np.sqrt(rho[far[i]])
        assert np.ptp(scaled) / scaled.mean() < 5e-3
