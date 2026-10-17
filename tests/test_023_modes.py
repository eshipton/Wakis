import numpy as np
import pytest
from scipy.constants import c, epsilon_0, mu_0
from scipy.optimize import brentq

import wakis
from wakis.mode_solver import ModeMonitor, ModeSource, solve_modes
from wakis.sources import (
    CurrentSource,
    CustomSource,
    GaussianSource,
    MagneticCurrentSource,
)

Z0 = np.sqrt(mu_0 / epsilon_0)
DX = 1e-3


def _pec_guide(Na, Nb, length, axis="z", ends="cpml", use_gpu=False):
    """Rectangular PEC waveguide along `axis` with CPML ends. The last index
    along each non-periodic axis is a ghost layer, so the PEC walls sit on the
    nodes 0 and N-1: the broad wall is a = (Na - 1) * DX."""
    a = "xyz".index(axis)
    n = [0, 0, 0]
    n[a], n[(a + 1) % 3], n[(a + 2) % 3] = length, Na, Nb
    grid = wakis.GridFIT3D(0, n[0] * DX, 0, n[1] * DX, 0, n[2] * DX, *n, verbose=0)
    bc = ["pec"] * 3
    bc[a] = ends
    solver = wakis.SolverFIT3D(
        grid, use_gpu=use_gpu, bc_low=bc, bc_high=bc, n_pml=15, verbose=0
    )
    return grid, solver


def test_te10_dispersion():
    """
    TE10 mode of a PEC rectangular guide (a = 20, b = 10 cells): the mode
    solver reproduces the dispersion relation of the 3D FIT scheme exactly,
    (w_t/c)^2 = kx_t^2 + beta_t^2 with w_t = 2/dt sin(w dt/2),
    kx_t = 2/dx sin(pi dx/(2a)), beta_t = 2/dz sin(beta dz/2), and the
    continuum beta = sqrt(k^2 - (pi/a)^2) within grid dispersion
    (>= 20 cells per guide wavelength: < 0.5%). Ey follows sin(pi x/a) and the
    cutoff is c/(2a) to the same accuracy. TE20 is evanescent below 2 fc.
    """
    _, solver = _pec_guide(21, 11, 10, ends="pec")
    a = 20 * DX
    fc = c / (2 * a)
    plane = dict(center=(0, 0, 5 * DX), size=(np.inf, np.inf, 0))
    for f in (1.3 * fc, 1.6 * fc, 1.9 * fc):
        modes = solve_modes(solver, f, n_modes=3, **plane)
        assert len(modes) == 1
        mode = modes[0]

        k = 2 * np.pi * f / c
        np.testing.assert_allclose(
            mode.beta, np.sqrt(k**2 - (np.pi / a) ** 2), rtol=5e-3
        )
        w_t = 2 / solver.dt * np.sin(np.pi * f * solver.dt)
        kx_t = 2 / DX * np.sin(np.pi * DX / (2 * a))
        beta_t = np.sqrt((w_t / c) ** 2 - kx_t**2)
        np.testing.assert_allclose(
            mode.beta, 2 / DX * np.arcsin(beta_t * DX / 2), rtol=1e-9
        )
        # Discrete cutoff from the same relation
        fc_num = np.arcsin(
            c * np.sqrt(w_t**2 / c**2 - mode.beta_tilde**2) * solver.dt / 2
        )
        fc_num /= np.pi * solver.dt
        np.testing.assert_allclose(fc_num, fc, rtol=5e-3)

        x = mode.coords["Ec"][0]
        Ey = mode.e["c"][:, 5].real
        profile = np.sin(np.pi * x / a)
        overlap = abs(Ey @ profile) / (np.linalg.norm(Ey) * np.linalg.norm(profile))
        assert overlap > 0.999
        assert np.abs(mode.e["b"]).max() < 1e-9 * np.abs(Ey).max()


def test_slab_te0_neff():
    """
    TE0 mode of a symmetric dielectric slab (n1 = 2, n2 = 1, d = lambda/2),
    one periodic cell along the invariant direction and PEC walls 1.75
    wavelengths from the core. neff converges at second order to the root of
    tan(kappa d/2) = gamma/kappa, within 1% at 15 cells per wavelength in the
    core.
    """
    lam, n1, n2, d = 1.0, 2.0, 1.0, 0.5
    k0 = 2 * np.pi / lam

    def te_even(neff):
        kappa = k0 * np.sqrt(n1**2 - neff**2)
        gamma = k0 * np.sqrt(neff**2 - n2**2)
        return np.tan(kappa * d / 2) - gamma / kappa

    neff_lo = np.sqrt(n1**2 - (np.pi / (k0 * d)) ** 2)
    neff_exact = brentq(te_even, max(neff_lo, n2) + 1e-9, n1 - 1e-9)

    errors = []
    for cells in (15, 30):
        dx = lam / (n1 * cells)
        Nx = int(round(4 * lam / dx))
        grid = wakis.GridFIT3D(
            -Nx * dx / 2, Nx * dx / 2, 0, dx, 0, 4 * dx, Nx, 1, 4, verbose=0
        )
        solver = wakis.SolverFIT3D(
            grid,
            bc_low=["pec", "periodic", "pec"],
            bc_high=["pec", "periodic", "pec"],
            verbose=0,
        )
        # Ey and Ez sit on the x nodes, Ex on the midpoints; nodes on an
        # interface get the mean permittivity (tangential E continuous)
        nodes = grid.x[:Nx]
        mids = 0.5 * (grid.x[:-1] + grid.x[1:])
        for comp, x in (("x", mids), ("y", nodes), ("z", nodes)):
            dist = np.abs(x) - d / 2
            eps_r = np.where(dist < 0, n1**2, n2**2)
            eps_r = np.where(np.abs(dist) < 1e-6 * dx, (n1**2 + n2**2) / 2, eps_r)
            for i in range(Nx):
                solver.ieps[i, :, :, comp] = 1 / (epsilon_0 * eps_r[i])

        mode = solve_modes(
            solver, c / lam, center=(0, 0, 2 * dx), size=(np.inf, np.inf, 0)
        )[0]
        # TE: only Ey (transverse component c for a z-normal plane)
        assert np.abs(mode.e["b"]).max() < 1e-9 * np.abs(mode.e["c"]).max()
        errors.append(abs(mode.neff / neff_exact - 1))

    assert errors[0] < 1e-2
    assert errors[0] / errors[1] > 3.5


@pytest.mark.parametrize("boundary", ["cpml", "abc"])
def test_huygens_sheet(boundary, use_gpu):
    """
    A plane wave sheet made of an electric current K = n x H_inc and a
    magnetic current M_s = -n x E_inc (n = +z) radiates only along +z. On the
    staggered grid K sits on the E plane z0 at time t + dt/2 and M_s on the H
    plane z0 - dz/2 at time t, so the incident H is advanced by dz/(2c). The
    backward flux is < 1e-3 of the forward flux over the whole pulse band
    (measured ~1e-6) and the forward field equals the incident field E = S(f).
    Covers the magnetic current in the CPML and the standard (ABC) updates.
    """
    dz = 5e-3
    grid = wakis.GridFIT3D(
        -0.01, 0.01, -0.01, 0.01, -0.6, 0.6, 2, 2, int(1.2 / dz), verbose=0
    )
    solver = wakis.SolverFIT3D(
        grid,
        use_gpu=use_gpu,
        bc_low=["periodic", "periodic", boundary],
        bc_high=["periodic", "periodic", boundary],
        n_pml=15,
        verbose=0,
    )
    pulse = GaussianSource(frequency=1e9, fwidth=0.4e9)
    sheet = dict(size=(0.02, 0.02, 0.0))
    K = CurrentSource(
        CustomSource(lambda t: pulse(t + dz / (2 * c))),
        component="x",
        center=(0, 0, 0),
        amplitude=-1 / Z0,
        **sheet,
    )
    M = MagneticCurrentSource(
        pulse, component="y", center=(0, 0, -dz / 2), amplitude=-1.0, **sheet
    )
    # 0.3-1.7 GHz: 35 to 170 cells per wavelength
    f = np.linspace(0.3e9, 1.7e9, 8)
    back, forward = (
        wakis.FluxMonitor(f, center=(0, 0, z0), **sheet) for z0 in (-0.3, 0.3)
    )
    probe = wakis.DFTMonitor(f, components=["Ex"], center=(0, 0, 0.3))
    Nt = int((pulse.end_time + 6e-9) / solver.dt)
    solver.emsolve(Nt, source=[K, M], monitors=[back, forward, probe])

    ratio = -back.get_flux() / forward.get_flux()
    assert np.all(np.abs(ratio) < 1e-3), ratio
    S = pulse.fourier_transform(f)
    np.testing.assert_allclose(
        np.abs(probe.get_field("Ex")[0].ravel()), np.abs(S), rtol=2e-3
    )


@pytest.mark.parametrize("axis, direction", [("z", "+"), ("y", "-")])
def test_mode_source(axis, direction, use_gpu):
    """
    TE10 injected by a ModeSource into a PEC guide (a = 20 DX, 1.3-1.7 fc,
    24-31 cells per free-space wavelength) terminated by CPML. Downstream,
    the forward mode carries the injected power, |a+|^2 = |S|^2 within 1%,
    equal to the FluxMonitor flux/2 (single-mode guide); the backward
    amplitude there (CPML reflection) is < 1e-2. Upstream, the source leaks
    no backward wave at the design frequency (< 1e-3); away from it the TE10
    wave impedance changes and the leak grows (< 0.1 at +-0.2 fc).
    """
    a = "xyz".index(axis)
    sign = 1 if direction == "+" else -1
    grid, solver = _pec_guide(21, 11, 160, axis=axis, use_gpu=use_gpu)
    fc = c / (2 * 20 * DX)
    f0 = 1.5 * fc
    pulse = GaussianSource(frequency=f0, fwidth=0.1 * f0)

    def plane(s):
        center = np.full(3, 0.0)
        center[a] = 0.08 + sign * s
        size = np.full(3, np.inf)
        size[a] = 0
        return dict(center=center, size=size)

    source = ModeSource(pulse, direction=direction, **plane(-0.03))
    f = np.linspace(1.3 * fc, 1.7 * fc, 5)
    downstream = ModeMonitor(f, **plane(0.03))
    upstream = ModeMonitor(f, **plane(-0.05))
    extent = np.array([grid.x[-1], grid.y[-1], grid.z[-1]])
    flux_size, flux_center = extent.copy(), extent / 2
    flux_size[a], flux_center[a] = 0.0, plane(0.03)["center"][a]
    flux = wakis.FluxMonitor(f, center=flux_center, size=flux_size)
    Nt = int((pulse.end_time + 3e-9) / solver.dt)
    solver.emsolve(Nt, source=source, monitors=[downstream, upstream, flux])

    S = pulse.fourier_transform(f)
    a_plus, a_minus = downstream.get_amplitudes()
    b_plus, b_minus = upstream.get_amplitudes()
    if sign < 0:
        a_plus, a_minus = a_minus, a_plus
        b_plus, b_minus = b_minus, b_plus
        flux_sign = -1
    else:
        flux_sign = 1

    np.testing.assert_allclose(np.abs(a_plus) ** 2, np.abs(S) ** 2, rtol=1e-2)
    np.testing.assert_allclose(
        flux_sign * flux.get_flux(), 2 * np.abs(a_plus) ** 2, rtol=1e-4
    )
    assert np.all(np.abs(a_minus) < 1e-2 * np.abs(a_plus))
    leak = np.abs(b_minus) / np.abs(a_plus)
    assert leak[2] < 1e-3
    assert np.all(leak < 0.1), leak


def test_mode_orthogonality(use_gpu):
    """
    Injecting TE20 (second mode, 2.2-2.3 fc, below the TE01 cutoff of
    2.5 fc) excites no TE10: the ModeMonitor projection on TE10 is < 1e-6 of
    the TE20 amplitude, and TE20 carries the injected power within 1%.
    """
    _, solver = _pec_guide(21, 9, 160, use_gpu=use_gpu)
    fc = c / (2 * 20 * DX)
    f0 = 2.25 * fc
    pulse = GaussianSource(frequency=f0, fwidth=0.05 * f0)
    port = dict(size=(np.inf, np.inf, 0))
    source = ModeSource(pulse, center=(0, 0, 0.05), mode_index=1, **port)
    f = np.linspace(2.2 * fc, 2.3 * fc, 3)
    monitor = ModeMonitor(f, center=(0, 0, 0.11), n_modes=2, **port)
    Nt = int((pulse.end_time + 3e-9) / solver.dt)
    solver.emsolve(Nt, source=source, monitors=[monitor])

    te10, _ = monitor.get_amplitudes(0)
    te20, _ = monitor.get_amplitudes(1)
    S = pulse.fourier_transform(f)
    np.testing.assert_allclose(np.abs(te20) ** 2, np.abs(S) ** 2, rtol=1e-2)
    assert np.all(np.abs(te10) < 1e-6 * np.abs(te20))
