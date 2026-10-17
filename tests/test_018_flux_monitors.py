import numpy as np
from scipy.constants import c, epsilon_0, mu_0

import wakis
from wakis.sources import CurrentSource, GaussianSource

Z0 = np.sqrt(mu_0 / epsilon_0)


def test_dipole_flux_box(use_gpu):
    """
    Energy conservation for a point dipole radiating into CPML: the flux out of
    a closed box equals the work done by the source, -Re(E(f) Il(f)^*), is
    independent of the box size, and matches the Hertzian dipole radiated
    power eta k^2 |Il|^2 / (6 pi) per |S(f)|^2 (flux spectra carry a factor 2
    with respect to time-averaged power).
    """
    L, N = 0.5, 40
    dx = L / N
    grid = wakis.GridFIT3D(-L / 2, L / 2, -L / 2, L / 2, -L / 2, L / 2, N, N, N)
    solver = wakis.SolverFIT3D(
        grid,
        use_gpu=use_gpu,
        bc_low=["cpml"] * 3,
        bc_high=["cpml"] * 3,
        n_pml=10,
        verbose=0,
    )
    src_time = GaussianSource(frequency=1e9, fwidth=0.3e9)
    dipole = CurrentSource(src_time, component="z", center=(0.0, 0.0, 0.0))

    f = np.linspace(0.8e9, 1.2e9, 5)
    point = wakis.DFTMonitor(f, components=["Ez"], center=(0.0, 0.0, 0.0))
    small_box = wakis.FluxMonitor(f, size=(6 * dx,) * 3)
    large_box = wakis.FluxMonitor(f, size=(12 * dx,) * 3)
    Nt = int((src_time.end_time + 6e-9) / solver.dt)
    solver.emsolve(Nt, source=dipole, monitors=[point, small_box, large_box])

    S = src_time.fourier_transform(f)
    work = -np.real(point.get_field("Ez")[0].ravel() * np.conj(S))
    k = 2 * np.pi * f / c
    analytic = Z0 * k**2 / (6 * np.pi) * np.abs(S) ** 2

    np.testing.assert_allclose(small_box.get_flux(), work, rtol=2e-3)
    np.testing.assert_allclose(large_box.get_flux(), small_box.get_flux(), rtol=1e-4)
    # 20-30 cells per wavelength: grid dispersion error ~1%
    np.testing.assert_allclose(large_box.get_flux(), analytic, rtol=1.5e-2)


def _slab_fluxes(f, eps_r=None, thickness=0.1, dz=5e-3, use_gpu=False):
    """Flux spectra on planes below and above a sheet source, with an optional
    dielectric slab between z = 0 and z = thickness."""
    grid = wakis.GridFIT3D(-0.01, 0.01, -0.01, 0.01, -0.6, 0.6, 2, 2, int(1.2 / dz))
    solver = wakis.SolverFIT3D(
        grid,
        use_gpu=use_gpu,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=15,
        verbose=0,
    )
    if eps_r is not None:
        # Tangential E sits on the z nodes; nodes on an interface see the
        # arithmetic mean of the permittivities (tangential continuity of E)
        z = grid.z[:-1]
        tol = 1e-6 * dz
        inside = (z > tol) & (z < thickness - tol)
        interface = (np.abs(z) < tol) | (np.abs(z - thickness) < tol)
        for comp in "xy":
            solver.ieps[:, :, inside, comp] = 1 / (epsilon_0 * eps_r)
            solver.ieps[:, :, interface, comp] = 2 / (epsilon_0 * (1 + eps_r))

    src_time = GaussianSource(frequency=1e9, fwidth=0.4e9)
    sheet = CurrentSource(
        src_time, component="x", center=(0.0, 0.0, -0.3), size=(0.02, 0.02, 0.0)
    )
    planes = [
        wakis.FluxMonitor(f, center=(0.0, 0.0, z0), size=(0.02, 0.02, 0.0))
        for z0 in (-0.45, -0.15, 0.3)
    ]
    Nt = int((src_time.end_time + 12e-9) / solver.dt)
    solver.emsolve(Nt, source=sheet, monitors=planes)
    return [plane.get_flux() for plane in planes]


def test_slab_transmission(use_gpu):
    """
    Normal-incidence transmission through a lossless dielectric slab follows
    T = 1 / (1 + (n - 1/n)^2 sin^2(k0 n d) / 4). Transmission and reflection
    are normalized by a reference run without the slab, and R + T = 1. Flux
    planes count energy along +z, so a plane below the source sees the
    downward-going wave as negative flux.
    """
    f = np.linspace(0.5e9, 1.5e9, 11)
    eps_r, thickness = 4.0, 0.1
    back_ref, below_ref, above_ref = _slab_fluxes(f, use_gpu=use_gpu)
    _, below, above = _slab_fluxes(f, eps_r=eps_r, thickness=thickness, use_gpu=use_gpu)

    np.testing.assert_allclose(back_ref, -above_ref, rtol=1e-4)

    T = above / above_ref
    R = 1 - below / below_ref
    n = np.sqrt(eps_r)
    T_analytic = 1 / (
        1 + 0.25 * (n - 1 / n) ** 2 * np.sin(2 * np.pi * f / c * n * thickness) ** 2
    )
    # >= 20 cells per wavelength inside the slab: phase error shifts the
    # Fabry-Perot fringes by < 1%
    np.testing.assert_allclose(T, T_analytic, atol=1.5e-2)
    np.testing.assert_allclose(R + T, 1.0, atol=1e-4)
