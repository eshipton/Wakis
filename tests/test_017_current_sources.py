import numpy as np
from scipy.constants import c, epsilon_0, mu_0

import wakis
from wakis.sources import CurrentSource, GaussianSource

Z0 = np.sqrt(mu_0 / epsilon_0)


def _dtft(signal, t, f):
    """Riemann-sum Fourier transform of uniformly sampled signals along axis 0."""
    kernel = np.exp(-2j * np.pi * np.outer(f, t))
    return np.tensordot(kernel, signal, axes=(1, 0)) * (t[1] - t[0])


def test_gaussian_spectrum():
    """
    The Gaussian time profile matches its analytic Fourier transform across the
    band and carries no DC component, so a pulse deposits no net charge.
    """
    src_time = GaussianSource(frequency=1e9, fwidth=0.3e9)
    t = np.arange(0.0, src_time.end_time, 1e-12)
    f = np.linspace(0.5e9, 1.5e9, 11)

    spectrum = _dtft(src_time(t), t, f)

    np.testing.assert_allclose(spectrum, src_time.fourier_transform(f), rtol=1e-4)
    peak = np.abs(src_time.fourier_transform(1e9))
    assert abs(_dtft(src_time(t), t, [0.0])[0]) < 1e-5 * peak


def _dipole_radiation_ratio(N, f):
    """Simulated over analytic radiation resistance of a point dipole in CPML."""
    L = 0.5
    grid = wakis.GridFIT3D(-L / 2, L / 2, -L / 2, L / 2, -L / 2, L / 2, N, N, N)
    solver = wakis.SolverFIT3D(
        grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=10, verbose=0
    )
    src_time = GaussianSource(frequency=1e9, fwidth=0.3e9)
    dipole = CurrentSource(src_time, component="z", center=(0.0, 0.0, 0.0))

    Nt = int((src_time.end_time + 6e-9) / solver.dt)
    Ez = np.zeros(Nt)
    for n in range(Nt):
        dipole.update(solver, n * solver.dt)
        solver.one_step()
        Ez[n] = solver.E.array[dipole.flat_idx[0]]

    # E is known at (n+1)*dt after each step. The source spectrum needs no
    # half-step correction: J samples s(t) itself at (n+1/2)*dt. A phase error
    # here would leak the large reactive near field into the radiated power.
    E_f = _dtft(Ez, (np.arange(Nt) + 1) * solver.dt, f)
    Il_f = src_time.fourier_transform(f)

    # P = -1/2 Re(E Il*) = eta k^2 |Il|^2 / (12 pi)  =>  -Re(E/Il) = eta k^2/(6 pi)
    k = 2 * np.pi * f / c
    return -np.real(E_f / Il_f) / (Z0 * k**2 / (6 * np.pi))


def test_dipole_radiated_power():
    """
    A point current moment Il radiates P = eta k^2 |Il|^2 / (12 pi) into free
    space (Hertzian dipole). The power delivered by the source, -1/2 Re(E Il*)
    at the dipole, converges to this value at second order in the mesh size,
    so the source normalization is mesh-independent.
    """
    f = np.linspace(0.8e9, 1.2e9, 5)
    ratio_coarse = _dipole_radiation_ratio(30, f)  # 15-22 cells per wavelength
    ratio_fine = _dipole_radiation_ratio(40, f)  # 20-30 cells per wavelength

    np.testing.assert_allclose(ratio_coarse, 1.0, atol=0.03)
    np.testing.assert_allclose(ratio_fine, 1.0, atol=0.015)
    # Grid dispersion error scales as dx^2: expect (40/30)^2 = 1.78
    order_gain = np.abs(ratio_coarse - 1) / np.abs(ratio_fine - 1)
    assert np.all(order_gain > 1.5)


def test_sheet_source_plane_wave():
    """
    A uniform sheet of surface current K launches plane waves of amplitude
    |E| = eta0 |K| / 2 in both directions, uniform over the transverse plane.
    """
    Nx = Ny = 4
    grid = wakis.GridFIT3D(-0.02, 0.02, -0.02, 0.02, -0.6, 0.6, Nx, Ny, 120)
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=10,
        verbose=0,
    )
    src_time = GaussianSource(frequency=1e9, fwidth=0.3e9)
    sheet = CurrentSource(
        src_time, component="x", center=(0.0, 0.0, 0.0), size=(0.04, 0.04, 0.0)
    )

    iz_below, iz_above = 30, 90
    Nt = int((src_time.end_time + 4e-9) / solver.dt)
    Ex = np.zeros((Nt, 2, Nx, Ny))
    for n in range(Nt):
        sheet.update(solver, n * solver.dt)
        solver.one_step()
        Ex[n, 0] = solver.E[:, :, iz_below, "x"]
        Ex[n, 1] = solver.E[:, :, iz_above, "x"]

    f = np.array([0.8e9, 1.0e9, 1.2e9])
    E_f = _dtft(Ex, (np.arange(Nt) + 1) * solver.dt, f)
    K_f = src_time.fourier_transform(f)
    ratio = np.abs(E_f) / (Z0 / 2 * np.abs(K_f))[:, None, None, None]

    # Discretization error grows as (f dz/c)^2, 0.55% at 30 cells per wavelength
    np.testing.assert_allclose(ratio, 1.0, rtol=1.5e-2)
    error = ratio.mean(axis=(1, 2, 3)) - 1
    np.testing.assert_allclose(error / error[1], (f / f[1]) ** 2, rtol=0.05)
    assert np.all(ratio.std(axis=(2, 3)) < 1e-10)


def test_superposition_lossy_medium():
    """
    Soft sources only add a forcing term to Ampere's law, so fields are linear
    in the sources: two sources together give the sum of their individual
    fields, also in a conducting medium where J carries the ohmic current.
    """

    def run(sources):
        grid = wakis.GridFIT3D(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 16, 16, 16)
        solver = wakis.SolverFIT3D(
            grid,
            bc_low=["pec"] * 3,
            bc_high=["pec"] * 3,
            bg=[2.0, 1.0, 0.05],
            verbose=0,
        )
        assert solver.use_conductivity
        solver.emsolve(300, source=sources)
        return solver.E.array.copy()

    src_time = GaussianSource(frequency=1.5e9, fwidth=0.5e9)

    def sources():
        return [
            CurrentSource(src_time, component="z", center=(0.0, 0.0, 0.0)),
            CurrentSource(
                src_time, component="x", center=(0.03, -0.02, 0.01), amplitude=-0.7
            ),
        ]

    E_a = run(sources()[0])
    E_b = run(sources()[1])
    E_ab = run(sources())
    E_aa = run([sources()[0], sources()[0]])

    scale = np.abs(E_ab).max()
    np.testing.assert_allclose(E_ab, E_a + E_b, atol=1e-10 * scale)
    np.testing.assert_allclose(E_aa, 2 * E_a, atol=1e-10 * scale)
