import numpy as np
import pytest
from scipy.constants import c

import wakis
from wakis.materials import (
    Debye,
    DispersiveMaterial,
    Drude,
    Lorentz,
    dispersive_lib,
    eps_from_nk,
    fit_poles,
)
from wakis.sources import CurrentSource, GaussianSource


def _slab_rt(eps, f, d):
    """Analytic normal-incidence power reflectance and transmittance of a slab
    of complex permittivity `eps` (exp(+i w t) convention) and thickness `d`
    in vacuum."""
    n = np.sqrt(eps.astype(complex))
    n = np.where(n.imag > 0, -n, n)  # decaying branch, Im n <= 0
    r12 = (1 - n) / (1 + n)
    phase = np.exp(-2j * 2 * np.pi * f / c * n * d)
    r = r12 * (1 - phase) / (1 - r12**2 * phase)
    t = (1 - r12**2) * np.sqrt(phase) / (1 - r12**2 * phase)
    return np.abs(r) ** 2, np.abs(t) ** 2


def _slab_fluxes(f, material=None, thickness=0.1, dz=5e-3, cfln=0.5, use_gpu=False):
    """Flux spectra below (reflected) and above (transmitted) a dispersive slab
    between z = 0 and z = thickness, illuminated at normal incidence by a sheet
    current in a quasi-1D periodic cell with CPML along z."""
    grid = wakis.GridFIT3D(
        -0.01, 0.01, -0.01, 0.01, -0.6, 0.6, 2, 2, int(1.2 / dz), verbose=0
    )
    solver = wakis.SolverFIT3D(
        grid,
        cfln=cfln,
        use_gpu=use_gpu,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=15,
        verbose=0,
    )
    if material is not None:
        solver.add_dispersive_material(
            material, center=(0.0, 0.0, thickness / 2), size=(1.0, 1.0, thickness)
        )
    src_time = GaussianSource(frequency=1e9, fwidth=0.4e9)
    sheet = CurrentSource(
        src_time, component="x", center=(0.0, 0.0, -0.3), size=(0.02, 0.02, 0.0)
    )
    planes = [
        wakis.FluxMonitor(f, center=(0.0, 0.0, z0), size=(0.02, 0.02, 0.0))
        for z0 in (-0.15, thickness + 0.1)
    ]
    Nt = int((src_time.end_time + 20e-9) / solver.dt)
    solver.emsolve(Nt, source=sheet, monitors=planes)
    return [plane.get_flux() for plane in planes], solver.dt


def _reflectance(material, f, thickness, dz=5e-3, use_gpu=False):
    (below_ref, above_ref), _ = _slab_fluxes(f, dz=dz, use_gpu=use_gpu)
    (below, above), dt = _slab_fluxes(f, material, thickness, dz, use_gpu=use_gpu)
    return 1 - below / below_ref, above / above_ref, dt


# ---------------------------------------------------------------- bulk media


def test_lorentz_slab(use_gpu):
    """
    A damped Lorentz slab (resonance at 1.3 GHz) reproduces the analytic
    reflectance and transmittance of the Fabry-Perot slab with eps(w),
    including the absorption band. dz = 2.5 mm gives >= 32 cells per
    wavelength inside the slab (|n| <= 2.9): R errors ~5e-3 from the grid
    dispersion and the staircased interface.
    """
    f = np.linspace(0.6e9, 1.4e9, 9)
    material = DispersiveMaterial(
        2.0,
        [Lorentz(delta_eps=2.0, omega_0=2 * np.pi * 1.3e9, gamma=2 * np.pi * 0.1e9)],
    )
    R, T, _ = _reflectance(material, f, 0.1, dz=2.5e-3, use_gpu=use_gpu)
    R_an, T_an = _slab_rt(material.epsilon(f), f, 0.1)
    np.testing.assert_allclose(R, R_an, atol=1e-2)
    np.testing.assert_allclose(T, T_an, atol=2e-3)


def test_drude_plasma_edge(use_gpu):
    """
    A Drude slab reflects almost totally below the plasma frequency (1 GHz)
    and becomes transparent above it. Reflectance and transmittance across the
    edge match the analytic slab with eps(w) = 1 - w_p^2 / (w^2 - i gamma w);
    dz = 5 mm gives >= 43 cells per vacuum wavelength.
    """
    f = np.linspace(0.6e9, 1.4e9, 9)
    material = DispersiveMaterial(
        1.0, [Drude(omega_p=2 * np.pi * 1e9, gamma=2 * np.pi * 0.02e9)]
    )
    R, T, _ = _reflectance(material, f, 0.2, use_gpu=use_gpu)
    R_an, T_an = _slab_rt(material.epsilon(f), f, 0.2)
    assert R_an[0] > 0.9 and R_an[-1] < 0.1
    np.testing.assert_allclose(R, R_an, atol=2e-3)
    np.testing.assert_allclose(T, T_an, atol=2e-3)


def test_debye_conductive_slab(use_gpu):
    """
    A lossy Debye dielectric (relaxation at 1 GHz) with static conductivity:
    reflectance and transmittance match the analytic slab. |n| <= 3 gives
    >= 20 cells per wavelength inside the slab.
    """
    f = np.linspace(0.6e9, 1.4e9, 9)
    material = DispersiveMaterial(
        3.0, [Debye(delta_eps=6.0, tau=1 / (2 * np.pi * 1e9))], sigma=0.05
    )
    R, T, _ = _reflectance(material, f, 0.1, use_gpu=use_gpu)
    R_an, T_an = _slab_rt(material.epsilon(f), f, 0.1)
    np.testing.assert_allclose(R, R_an, atol=1e-2)
    np.testing.assert_allclose(T, T_an, atol=2e-3)


def test_slab_convergence(use_gpu):
    """
    The reflectance error of the resonant Lorentz slab decreases at second
    order when the cell size (and with it the timestep) is halved.
    """
    f = np.linspace(0.9e9, 1.4e9, 6)
    material = DispersiveMaterial(
        2.0,
        [Lorentz(delta_eps=2.0, omega_0=2 * np.pi * 1.3e9, gamma=2 * np.pi * 0.1e9)],
    )
    R_an, _ = _slab_rt(material.epsilon(f), f, 0.1)
    errors = []
    for dz in (5e-3, 2.5e-3):
        R, _, _ = _reflectance(material, f, 0.1, dz=dz, use_gpu=use_gpu)
        errors.append(np.max(np.abs(R - R_an)))
    assert errors[0] / errors[1] > 3.0


class _UniformCurrent:
    """Spatially uniform impressed Jx(t) [A/m^2]; samples are recorded at the
    times n dt at which the solver applies them."""

    def __init__(self, src_time):
        self.src_time = src_time
        self.samples = []

    def update(self, solver, t):
        value = float(self.src_time(np.array([t]))[0])
        solver.J[:, :, :, "x"] = value
        self.samples.append(value)


def test_discrete_permittivity(use_gpu):
    """
    A spatially uniform field (k = 0, curl H = 0) isolates the time
    discretization: the permittivity extracted from E(f) and the impressed
    J(f), eps = -dt J / ((z - 1) eps_0 E) with z = exp(i w dt), equals the
    analytic discrete-model permittivity `epsilon_discrete` to round-off, for
    a mixed Lorentz + Drude + Debye + sigma material on a coarse timestep
    (about 25 steps per period), where it differs from eps(w) by ~1%.
    """
    from scipy.constants import epsilon_0

    grid = wakis.GridFIT3D(0, 1, 0, 1, 0, 1, 3, 3, 3, verbose=0)
    f0 = 1e9
    material = DispersiveMaterial(
        2.0,
        [
            Lorentz(1.0, 2 * np.pi * 1.2 * f0, 2 * np.pi * 0.2 * f0),
            Drude(2 * np.pi * 0.8 * f0, 2 * np.pi * 0.3 * f0),
            Debye(1.5, 1 / (2 * np.pi * f0)),
        ],
        sigma=0.01,
    )
    solver = wakis.SolverFIT3D(grid, dt=1 / (25 * f0), use_gpu=use_gpu, verbose=0)
    solver.add_dispersive_material(material, center=(0.5, 0.5, 0.5), size=(2, 2, 2))
    source = _UniformCurrent(GaussianSource(frequency=f0, fwidth=0.5 * f0))
    Nt = 3000
    E = np.empty(Nt)
    for n in range(Nt):
        source.update(solver, n * solver.dt)
        solver.one_step()
        E[n] = float(solver.E[1, 1, 1, "x"])

    f = np.linspace(0.6, 1.4, 9) * f0
    dt = solver.dt
    z = np.exp(2j * np.pi * f * dt)
    t = np.arange(Nt) * dt
    J_f = np.exp(-2j * np.pi * np.outer(f, t)) @ np.array(source.samples)
    E_f = np.exp(-2j * np.pi * np.outer(f, t + dt)) @ E  # E^(n+1) at (n+1) dt
    eps_sim = -dt * J_f / ((z - 1) * epsilon_0 * E_f)

    eps_d = material.epsilon_discrete(f, dt)
    np.testing.assert_allclose(eps_sim, eps_d, rtol=1e-8)
    assert np.max(np.abs(eps_d / material.epsilon(f) - 1)) > 5e-3


def test_epsilon_discrete_limit():
    """The discrete-model permittivity converges to eps(w) at O(dt^2)."""
    material = dispersive_lib["gold"]
    f = np.linspace(200e12, 700e12, 11)
    err = [
        np.max(np.abs(material.epsilon_discrete(f, dt) / material.epsilon(f) - 1))
        for dt in (2e-17, 1e-17)
    ]
    assert 3.8 < err[0] / err[1] < 4.2


# ------------------------------------------------------------------ stability


def test_stability_criterion():
    """
    The von Neumann check reproduces the closed-form limits of a lossless
    Drude pole, (w_p dt/2)^2 + C^2 <= eps_inf, and of a lossless Lorentz
    pole, (w_0 dt/2)^2 <= (eps_inf - C^2)/(eps_s - C^2), with C the vacuum
    Courant number.
    """
    C = 0.5
    drude = DispersiveMaterial(2.0, [Drude(omega_p=1.0, gamma=0.0)])
    dt_max = 2 * np.sqrt(2.0 - C**2)
    assert drude.is_stable(0.99 * dt_max, C)
    assert not drude.is_stable(1.01 * dt_max, C)

    lorentz = DispersiveMaterial(1.5, [Lorentz(delta_eps=3.0, omega_0=1.0, gamma=0.0)])
    dt_max = 2 * np.sqrt((1.5 - C**2) / (4.5 - C**2))
    assert lorentz.is_stable(0.99 * dt_max, C)
    assert not lorentz.is_stable(1.01 * dt_max, C)


def test_unstable_timestep_raises():
    """Assigning a material whose poles are not resolved by dt is refused."""
    grid = wakis.GridFIT3D(0, 1, 0, 1, 0, 1, 3, 3, 3, verbose=0)
    solver = wakis.SolverFIT3D(grid, verbose=0)
    material = DispersiveMaterial(1.0, [Drude(omega_p=10 / solver.dt, gamma=0.0)])
    with pytest.raises(ValueError, match="unstable"):
        solver.add_dispersive_material(material, size=(1, 1, 1))


def test_lossless_lorentz_long_run(use_gpu):
    """
    A lossless (gamma = 0) Lorentz medium filling a closed periodic cavity
    keeps a bounded field over 40000 steps (~ 270 periods of the resonance)
    at the default CFL number: no late-time growth.
    """
    grid = wakis.GridFIT3D(0, 0.02, 0, 0.02, 0, 0.4, 2, 2, 40, verbose=0)
    solver = wakis.SolverFIT3D(grid, use_gpu=use_gpu, verbose=0)
    material = DispersiveMaterial(
        1.5, [Lorentz(delta_eps=4.0, omega_0=2 * np.pi * 1e9, gamma=0.0)]
    )
    solver.add_dispersive_material(material, center=(0.01, 0.01, 0.1), size=(1, 1, 0.2))
    src_time = GaussianSource(frequency=1e9, fwidth=0.5e9)
    sheet = CurrentSource(
        src_time, component="x", center=(0.0, 0.0, 0.3), size=(0.02, 0.02, 0.0)
    )
    Nt = 40000
    amplitude = np.empty(Nt)
    for n in range(Nt):
        sheet.update(solver, n * solver.dt)
        solver.one_step()
        amplitude[n] = float(np.max(np.abs(solver.E[:, :, :, "x"])))
    early = amplitude[Nt // 10 : Nt // 5].max()
    late = amplitude[-Nt // 10 :].max()
    assert np.isfinite(late) and late < 1.5 * early


# -------------------------------------------------------------------- fitting


def test_fit_recovers_poles():
    """
    Fitting noiseless synthetic data recovers the generating Lorentz + Drude
    (optical) and two-pole Debye (RF) parameters, and the fitted models are
    passive: Im eps <= 0 (exp(+i w t) convention) at all frequencies.
    """
    from wakis.materials import EV

    true = DispersiveMaterial(
        2.0, [Lorentz(1.5, 3 * EV, 0.3 * EV), Drude(8 * EV, 0.1 * EV)]
    )
    f = np.linspace(0.5, 5, 60) * EV / (2 * np.pi)
    fit = fit_poles(f, true.epsilon(f), n_lorentz=1, n_drude=1)
    assert fit.fit_error < 1e-8
    np.testing.assert_allclose(fit.eps_inf, 2.0, rtol=1e-6)
    lorentz, drude = fit.poles
    np.testing.assert_allclose(
        [lorentz.delta_eps, lorentz.omega_0, lorentz.gamma],
        [1.5, 3 * EV, 0.3 * EV],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        [drude.omega_p, drude.gamma], [8 * EV, 0.1 * EV], rtol=1e-6
    )

    true = DispersiveMaterial(3.0, [Debye(4.0, 1e-10), Debye(2.0, 3e-12)])
    f = np.geomspace(1e8, 1e12, 60)
    fit = fit_poles(f, true.epsilon(f), n_lorentz=0, n_debye=2)
    taus = sorted(p.tau for p in fit.poles)
    np.testing.assert_allclose(taus, [3e-12, 1e-10], rtol=1e-6)

    # Imperfect model (gold with fewer poles): still passive everywhere
    f = np.linspace(1.0, 3.0, 40) * EV / (2 * np.pi)
    eps_gold = dispersive_lib["gold"].epsilon(f)
    n, k = np.sqrt(np.conj(eps_gold)).real, np.sqrt(np.conj(eps_gold)).imag
    np.testing.assert_allclose(eps_from_nk(n, k), eps_gold, rtol=1e-12)
    fit = fit_poles(f, eps_from_nk(n, k), n_lorentz=1, n_drude=1)
    assert fit.fit_error < 0.05
    assert np.all(fit.epsilon(np.geomspace(1e9, 1e17, 500)).imag <= 0)


def test_library_values():
    """
    Library models reproduce their source formulas: fused silica n = 1.4440
    at 1.55 um (Malitson), silicon n = 3.4777 at 1.55 um (Salzberg & Villa
    Sellmeier), FR-4 eps = 4.4 (1 - 0.02 i) at 1 GHz, and Drude-like metals
    with Re eps < 0 in the visible.
    """
    f = c / 1.55e-6
    n_sio2 = np.sqrt(dispersive_lib["fused silica"].epsilon(f)).real
    n_si = np.sqrt(dispersive_lib["silicon"].epsilon(f)).real
    np.testing.assert_allclose(n_sio2, 1.4440, atol=1e-4)
    np.testing.assert_allclose(n_si, 3.4777, atol=1e-4)
    np.testing.assert_allclose(
        dispersive_lib["fr4"].epsilon(1e9), 4.4 * (1 - 0.02j), rtol=3e-3
    )
    for metal in ("gold", "silver", "aluminium"):
        assert dispersive_lib[metal].epsilon(c / 600e-9).real < 0
        assert dispersive_lib[metal].epsilon(c / 600e-9).imag < 0


# ---------------------------------------------------------- STL assignment


def _sphere_run(material, use_gpu=False):
    """Fields after 150 steps of a dipole next to an STL sphere (r = 50 mm)
    on an anisotropic, off-centre grid (Nx != Ny != Nz)."""
    grid = wakis.GridFIT3D(
        -60e-3,
        70e-3,
        -65e-3,
        75e-3,
        -70e-3,
        62e-3,
        17,
        15,
        19,
        stl_solids={"Sphere": "tests/stl/003_sphere.stl"},
        stl_materials={"Sphere": material},
        stl_scale=1e-3,
        verbose=0,
    )
    solver = wakis.SolverFIT3D(
        grid,
        use_stl=True,
        use_gpu=use_gpu,
        bc_low=["pec"] * 3,
        bc_high=["pec"] * 3,
        verbose=0,
    )
    src = CurrentSource(
        GaussianSource(frequency=2e9, fwidth=1e9),
        component="z",
        center=(0.0, 55e-3, 0.0),
    )
    for n in range(150):
        src.update(solver, n * solver.dt)
        solver.one_step()
    E = solver.E.toarray()
    return solver, np.asarray(E.get() if use_gpu else E)


def test_stl_assignment(use_gpu):
    """
    A pole-free DispersiveMaterial assigned to an STL solid reproduces the
    static material [eps_r, 1, 0] exactly (same subpixel volume fractions and
    cell mapping); adding a Debye pole changes the fields and stays finite.
    """
    _, E_ref = _sphere_run([4.0, 1.0, 0.0], use_gpu)
    solver, E = _sphere_run(DispersiveMaterial(4.0), use_gpu)
    assert solver._dispersive_groups[0]["idx_host"].size > 0
    np.testing.assert_allclose(E, E_ref, rtol=0, atol=1e-10 * np.abs(E_ref).max())

    _, E_d = _sphere_run(DispersiveMaterial(4.0, [Debye(3.0, 1e-11)]), use_gpu)
    assert np.all(np.isfinite(E_d))
    assert np.abs(E_d - E).max() > 1e-3 * np.abs(E).max()


def test_region_override():
    """A later box assignment replaces the earlier material on shared edges."""
    grid = wakis.GridFIT3D(0, 1, 0, 1, 0, 1, 4, 4, 4, verbose=0)
    solver = wakis.SolverFIT3D(grid, verbose=0)
    first = DispersiveMaterial(2.0, [Debye(1.0, 1e-9)])
    second = DispersiveMaterial(3.0, [Debye(1.0, 1e-9)])
    solver.add_dispersive_material(first, center=(0.5, 0.5, 0.5), size=(2, 2, 2))
    solver.add_dispersive_material(second, center=(0.5, 0.5, 0.1), size=(2, 2, 0.2))
    a, b = solver._dispersive_groups
    assert np.intersect1d(a["idx_host"], b["idx_host"]).size == 0
    assert a["idx_host"].size + b["idx_host"].size == 3 * solver.N
