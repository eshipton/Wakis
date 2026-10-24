# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Tests for dispersive (frequency-dependent, lossy, diagonal-anisotropic)
permeability mu(w) via the magnetic ADE update (`wakis.dispersive`),
its time-stepping hooks in `SolverFIT3D`, and its exact transpose in
`wakis.inverse.AdjointStepper`.
"""

import numpy as np
import pytest
from scipy.constants import c, epsilon_0, mu_0

import wakis
from wakis.dispersive import (
    Debye,
    DispersiveMaterial,
    Drude,
    Lorentz,
)
from wakis.inverse import AdjointSimulation, AdjointStepper, MaterialGrid
from wakis.resonances import harminv
from wakis.sources import CurrentSource, GaussianSource


def _slab_rt_general(eps_r, mu_r, f, d):
    """
    Analytic normal-incidence complex reflection r, transmission t, and power
    reflectance R = |r|^2, transmittance T = |t|^2 of a homogeneous slab of
    relative permittivity `eps_r`, relative permeability `mu_r` (in the Wakis
    exp(+i w t) convention, Im <= 0 for passive media) and thickness `d` in
    vacuum.
    """
    eps_r = np.asarray(eps_r, dtype=complex)
    mu_r = np.asarray(mu_r, dtype=complex)
    n = np.sqrt(eps_r * mu_r)
    n = np.where(n.imag > 0, -n, n)  # passive branch: Im(n) <= 0
    eta = np.sqrt(mu_r / eps_r)
    eta = np.where(eta.real < 0, -eta, eta)  # Re(eta) >= 0
    r12 = (eta - 1.0) / (eta + 1.0)
    phase = np.exp(-2j * 2 * np.pi * f / c * n * d)
    r = r12 * (1.0 - phase) / (1.0 - r12**2 * phase)
    t = (1.0 - r12**2) * np.exp(-1j * 2 * np.pi * f / c * n * d) / (
        1.0 - r12**2 * phase
    )
    return np.abs(r) ** 2, np.abs(t) ** 2, r, t


def _slab_fluxes(
    f,
    material=None,
    thickness=0.1,
    dz=5e-3,
    cfln=0.5,
    pol="x",
    mutate=None,
    use_gpu=False,
):
    """
    Flux spectra below (z = -0.15 m) and above (z = thickness + 0.1 m) a
    dispersive slab between z = 0 and z = `thickness`, illuminated at normal
    incidence by a sheet current along `pol` ('x' -> Ex, Hy; 'y' -> Ey, Hx)
    in a quasi-1D periodic domain with CPML along z.
    """
    nz = int(round(1.2 / dz))
    grid = wakis.GridFIT3D(-0.01, 0.01, -0.01, 0.01, -0.6, 0.6, 2, 2, nz, verbose=0)
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
        if mutate == "flip_zeta":
            for g in getattr(solver, "_dispersive_h_groups", []):
                g["lorentz"] = [
                    (a, x, -z, m, m_old) for a, x, z, m, m_old in g["lorentz"]
                ]
        elif mutate == "drop_sigma_m":
            for g in getattr(solver, "_dispersive_h_groups", []):
                s_hat = (
                    g["weight"]
                    * material._sigma_m_xyz[0]
                    * solver.dt
                    / mu_0
                    / 2
                )
                g["a1"] = g["a1"] + s_hat
                g["den"] = g["den"] - s_hat

    src_time = GaussianSource(frequency=1e9, fwidth=0.4e9)
    sheet = CurrentSource(
        src_time, component=pol, center=(0.0, 0.0, -0.3), size=(0.02, 0.02, 0.0)
    )
    planes = [
        wakis.FluxMonitor(f, center=(0.0, 0.0, z0), size=(0.02, 0.02, 0.0))
        for z0 in (-0.15, thickness + 0.1)
    ]
    Nt = int((src_time.end_time + 20e-9) / solver.dt)
    solver.emsolve(Nt, source=sheet, monitors=planes)
    return [plane.get_flux() for plane in planes], solver.dt


def _reflectance(
    material, f, thickness=0.1, dz=5e-3, pol="x", mutate=None, use_gpu=False
):
    (below_ref, above_ref), _ = _slab_fluxes(f, dz=dz, pol=pol, use_gpu=use_gpu)
    (below, above), dt = _slab_fluxes(
        f, material, thickness, dz=dz, pol=pol, mutate=mutate, use_gpu=use_gpu
    )
    return 1.0 - below / below_ref, above / above_ref, dt


# ------------------------------------------------ 1. Coefficients & duality


class _UniformMagneticCurrent:
    """Spatially uniform impressed Mx(t) [V/m^2] sampled at n * dt."""

    def __init__(self, src_time):
        self.src_time = src_time
        self.samples = []

    def update(self, solver, t):
        value = float(self.src_time(np.array([t]))[0])
        if solver.M is None:
            from wakis.field import Field

            solver.M = Field(
                solver.Nx,
                solver.Ny,
                solver.Nz,
                use_gpu=solver.use_gpu,
                dtype=solver.dtype,
            )
        solver.M[:, :, :, "x"] = value
        self.samples.append(value)


def test_coefficients_and_duality(use_gpu):
    """
    1. For a material with only magnetic poles and magnetic conductivity,
       `mu(f)` and `mu_discrete(f, dt)` equal `epsilon(f)` and
       `epsilon_discrete(f, dt)` of the dual electric material (with
       sigma_e / eps_0 = sigma_m / mu_0) to 1e-14.
    2. `mu_discrete(f, dt)` converges to `mu(f)` at second order O(dt^2).
    3. `is_stable(dt, courant)` agrees with the dual electric condition for
       both Drude and Lorentz magnetic poles, and an unstable dt raises
       ValueError when registering the material on a solver.
    4. Uniform k = 0 magnetic-current drive recovers `mu_discrete(f, dt)` to
       < 1e-8 from the time-domain H response.
    """
    f0 = 1e9
    poles = [
        Lorentz(1.2, 2 * np.pi * 1.3 * f0, 2 * np.pi * 0.15 * f0),
        Drude(2 * np.pi * 0.7 * f0, 2 * np.pi * 0.2 * f0),
        Debye(1.5, 1 / (2 * np.pi * f0)),
    ]
    sigma_e = 0.015
    sigma_m = sigma_e * mu_0 / epsilon_0
    mat_e = DispersiveMaterial(eps_inf=2.2, poles=poles, sigma=sigma_e)
    mat_m = DispersiveMaterial(mu_inf=2.2, mu_poles=poles, sigma_m=sigma_m)

    f = np.linspace(0.5e9, 1.8e9, 15)
    dt = 1 / (25 * f0)
    np.testing.assert_allclose(mat_m.mu(f), mat_e.epsilon(f), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        mat_m.mu_discrete(f, dt), mat_e.epsilon_discrete(f, dt), rtol=1e-14, atol=1e-14
    )

    # O(dt^2) convergence of mu_discrete -> mu
    err = [
        np.max(np.abs(mat_m.mu_discrete(f, step) / mat_m.mu(f) - 1.0))
        for step in (2e-11, 1e-11)
    ]
    assert 3.8 < err[0] / err[1] < 4.2

    # Exact duality of the von Neumann stability boundary
    C = 0.5
    drude_m = DispersiveMaterial(mu_inf=2.0, mu_poles=[Drude(omega_p=1.0, gamma=0.0)])
    dt_max_d = 2 * np.sqrt(2.0 - C**2)
    assert drude_m.is_stable(0.99 * dt_max_d, C)
    assert not drude_m.is_stable(1.01 * dt_max_d, C)

    lor_m = DispersiveMaterial(
        mu_inf=1.5, mu_poles=[Lorentz(delta_eps=3.0, omega_0=1.0, gamma=0.0)]
    )
    dt_max_l = 2 * np.sqrt((1.5 - C**2) / (4.5 - C**2))
    assert lor_m.is_stable(0.99 * dt_max_l, C)
    assert not lor_m.is_stable(1.01 * dt_max_l, C)

    # Unstable dt raises ValueError on add_dispersive_material
    grid = wakis.GridFIT3D(0, 1, 0, 1, 0, 1, 3, 3, 3, verbose=0)
    solver = wakis.SolverFIT3D(grid, verbose=0)
    unstable_m = DispersiveMaterial(
        mu_inf=1.0, mu_poles=[Drude(omega_p=10 / solver.dt, gamma=0.0)]
    )
    with pytest.raises(ValueError, match="unstable"):
        solver.add_dispersive_material(unstable_m, size=(1, 1, 1))

    # Spatially uniform k = 0 drive: mu_0 dH/dt = -M_src => discrete mu_d
    grid_u = wakis.GridFIT3D(0, 1, 0, 1, 0, 1, 3, 3, 3, verbose=0)
    solver_u = wakis.SolverFIT3D(grid_u, dt=dt, use_gpu=use_gpu, verbose=0)
    solver_u.add_dispersive_material(mat_m, center=(0.5, 0.5, 0.5), size=(2, 2, 2))
    src = _UniformMagneticCurrent(GaussianSource(frequency=f0, fwidth=0.5 * f0))
    Nt = 3000
    H = np.empty(Nt)
    for n in range(Nt):
        src.update(solver_u, n * dt)
        solver_u.one_step()
        H[n] = float(solver_u.H[1, 1, 1, "x"])

    f_sub = np.linspace(0.6, 1.4, 9) * f0
    z = np.exp(2j * np.pi * f_sub * dt)
    t = np.arange(Nt) * dt
    M_f = np.exp(-2j * np.pi * np.outer(f_sub, t)) @ np.array(src.samples)
    H_f = np.exp(-2j * np.pi * np.outer(f_sub, t + dt)) @ H
    mu_sim = -dt * M_f / ((z - 1.0) * mu_0 * H_f)
    np.testing.assert_allclose(mu_sim, mat_m.mu_discrete(f_sub, dt), rtol=1e-8)


def test_unsupported_guards_and_conformal_warning():
    """
    Unsupported combinations (scheme='tete', fused GPU, SubgridFIT2D) raise
    NotImplementedError with a clear message; conformal STL with magnetic
    dispersive material emits a UserWarning and falls back to volume-fraction
    mapping on H.
    """
    import pyvista as pv
    from wakis.subgrid import Circle, Geometry, QuadMesh, SubgridFIT2D

    mat_m = DispersiveMaterial(
        mu_inf=1.5, mu_poles=[Lorentz(0.5, 2 * np.pi * 1e9, 2 * np.pi * 0.1e9)]
    )
    grid = wakis.GridFIT3D(0, 0.1, 0, 0.1, 0, 0.05, 4, 4, 4, verbose=0)
    solver_tete = wakis.SolverFIT3D(
        grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, scheme="tete", verbose=0
    )
    with pytest.raises(NotImplementedError, match="scheme='tete'"):
        solver_tete.add_dispersive_material(mat_m, center=(0.05, 0.05, 0.05), size=(1, 1, 1))

    grid_f = wakis.GridFIT3D(0, 0.1, 0, 0.1, 0, 0.1, 4, 4, 4, verbose=0)
    solver_f = wakis.SolverFIT3D(grid_f, verbose=0)
    solver_f.fused = True
    with pytest.raises(NotImplementedError, match="fused GPU"):
        solver_f.add_dispersive_material(mat_m, center=(0.05, 0.05, 0.05), size=(1, 1, 1))

    mesh = QuadMesh(np.linspace(-0.1, 0.1, 5), np.linspace(-0.1, 0.1, 5))
    geo = Geometry([Circle((0.0, 0.0), 0.04, mat_m)])
    with pytest.raises(NotImplementedError, match="SubgridFIT2D"):
        SubgridFIT2D(mesh, geo, polarization="TM")

    box = pv.Box(bounds=(-0.02, 0.02, -0.02, 0.02, -0.02, 0.02)).triangulate()
    grid_c = wakis.GridFIT3D(
        -0.05,
        0.05,
        -0.05,
        0.05,
        -0.05,
        0.05,
        6,
        6,
        6,
        stl_solids={"box": box},
        stl_materials={"box": mat_m},
        conformal="all",
        verbose=0,
    )
    with pytest.warns(UserWarning, match="Subpixel averaging of dispersive permeability"):
        solver_c = wakis.SolverFIT3D(grid_c, use_stl=True, verbose=0)
    assert len(solver_c._dispersive_h_groups) == 1


# -------------------------------------------- 2. Plane wave through a slab


def test_lorentz_and_debye_mu_slab(use_gpu):
    """
    Plane wave through a damped Lorentz-mu slab and a conductive Debye-mu
    slab: reflectance R(f) and transmittance T(f) match the analytic
    transfer matrix evaluated with `mu_discrete(f, dt)`, and the error
    converges at second order when the grid resolution is doubled.
    Mutating the Lorentz zeta sign or dropping sigma_m makes the comparison
    fail by > 0.1.
    """
    f = np.linspace(0.7e9, 1.4e9, 8)
    lor_mu = DispersiveMaterial(
        mu_inf=2.0,
        mu_poles=[
            Lorentz(delta_eps=2.0, omega_0=2 * np.pi * 1.3e9, gamma=2 * np.pi * 0.1e9)
        ],
    )
    errs_r, errs_t = [], []
    for dz in (5e-3, 2.5e-3):
        R, T, dt = _reflectance(lor_mu, f, thickness=0.1, dz=dz, use_gpu=use_gpu)
        R_an, T_an, _, _ = _slab_rt_general(1.0, lor_mu.mu_discrete(f, dt), f, 0.1)
        errs_r.append(np.max(np.abs(R - R_an)))
        errs_t.append(np.max(np.abs(T - T_an)))
    assert errs_r[1] < 6e-3
    assert errs_t[1] < 3e-3
    assert errs_r[0] / errs_r[1] > 3.0
    assert errs_t[0] / errs_t[1] > 3.0
    print(
        f"Lorentz slab: dR={errs_r[0]:.3e}->{errs_r[1]:.3e} "
        f"(ratio={errs_r[0]/errs_r[1]:.2f}, order={np.log2(errs_r[0]/errs_r[1]):.2f}), "
        f"dT={errs_t[0]:.3e}->{errs_t[1]:.3e} "
        f"(ratio={errs_t[0]/errs_t[1]:.2f}, order={np.log2(errs_t[0]/errs_t[1]):.2f})"
    )

    # Mutation 1: flip sign of zeta coupling -> must fail
    R_mut, T_mut, dt_m = _reflectance(
        lor_mu, f, thickness=0.1, dz=5e-3, mutate="flip_zeta", use_gpu=use_gpu
    )
    R_an_m, T_an_m, _, _ = _slab_rt_general(1.0, lor_mu.mu_discrete(f, dt_m), f, 0.1)
    assert np.max(np.abs(R_mut - R_an_m)) > 0.15

    # Debye-mu slab with magnetic conductivity sigma_m
    sigma_m = 0.05 * mu_0 / epsilon_0
    deb_mu = DispersiveMaterial(
        mu_inf=3.0,
        mu_poles=[Debye(delta_eps=6.0, tau=1 / (2 * np.pi * 1e9))],
        sigma_m=sigma_m,
    )
    R_d, T_d, dt_d = _reflectance(deb_mu, f, thickness=0.1, dz=5e-3, use_gpu=use_gpu)
    R_dan, T_dan, _, _ = _slab_rt_general(1.0, deb_mu.mu_discrete(f, dt_d), f, 0.1)
    np.testing.assert_allclose(R_d, R_dan, atol=1e-2)
    np.testing.assert_allclose(T_d, T_dan, atol=3e-3)
    print(
        f"Debye+sigma_m slab (dz=5mm): max|dR|={np.max(np.abs(R_d - R_dan)):.3e}, "
        f"max|dT|={np.max(np.abs(T_d - T_dan)):.3e}"
    )

    # Mutation 2: drop sigma_m from the H update -> must fail
    R_dmut, T_dmut, _ = _reflectance(
        deb_mu, f, thickness=0.1, dz=5e-3, mutate="drop_sigma_m", use_gpu=use_gpu
    )
    assert np.max(np.abs(T_dmut - T_dan)) > 0.05


# ----------------------------------------- 3. Impedance-matched dual slab


def test_impedance_matched_dual_slab(use_gpu):
    """
    A slab with identical dispersive poles in eps(w) and mu(w) has wave
    impedance eta(w) = sqrt(mu(w) / eps(w)) = 1 at all frequencies, so its
    continuum reflectance vanishes identically. On the staggered Yee grid
    |R| is orders of magnitude smaller than the eps-only control at every
    frequency and decreases as dz is halved.
    """
    f = np.linspace(0.6e9, 1.4e9, 9)
    poles = [Lorentz(delta_eps=2.5, omega_0=2 * np.pi * 1.2e9, gamma=2 * np.pi * 0.15e9)]
    eps_only = DispersiveMaterial(eps_inf=2.0, poles=poles)
    matched = DispersiveMaterial(
        eps_inf=2.0, poles=poles, mu_inf=2.0, mu_poles=poles
    )

    R_eps, _, _ = _reflectance(eps_only, f, thickness=0.1, dz=5e-3, use_gpu=use_gpu)
    R_5, T_5, dt_5 = _reflectance(matched, f, thickness=0.1, dz=5e-3, use_gpu=use_gpu)
    R_25, T_25, dt_25 = _reflectance(
        matched, f, thickness=0.1, dz=2.5e-3, use_gpu=use_gpu
    )

    assert np.all(np.abs(R_5) < 0.12 * np.abs(R_eps))
    assert np.all(np.abs(R_25) < 0.035 * np.abs(R_eps))
    assert np.all(np.abs(R_25) < 0.5 * np.abs(R_5))
    assert np.max(np.abs(R_25)) < 1.6e-2
    print(
        f"Impedance-matched slab: max|R_5|={np.max(np.abs(R_5)):.3e}, "
        f"max|R_25|={np.max(np.abs(R_25)):.3e} "
        f"(ratio={np.max(np.abs(R_5))/np.max(np.abs(R_25)):.2f}, "
        f"order={np.log2(np.max(np.abs(R_5))/np.max(np.abs(R_25))):.2f})"
    )

    # Transmittance matches the impedance-matched attenuation |exp(-i k d)|^2
    _, T_an, _, _ = _slab_rt_general(
        matched.epsilon_discrete(f, dt_25),
        matched.mu_discrete(f, dt_25),
        f,
        0.1,
    )
    np.testing.assert_allclose(T_25, T_an, atol=3e-3)


# ---------------------------------------------------------- 4. Anisotropy


def test_diagonal_anisotropy_polarization_selectivity(use_gpu):
    """
    A diagonal magnetic material with poles only on 'x' (mu_x dispersive,
    mu_y = mu_z = 1) leaves an Ex / Hy plane wave (H along y) completely
    unaffected (matches vacuum to round-off), while an Ey / Hx plane wave
    (H along x) experiences the full Lorentz-mu slab response.
    """
    f = np.linspace(0.7e9, 1.3e9, 7)
    pole_x = [Lorentz(delta_eps=2.0, omega_0=2 * np.pi * 1.1e9, gamma=2 * np.pi * 0.1e9)]
    aniso = DispersiveMaterial(
        mu_inf=(2.0, 1.0, 1.0),
        mu_poles={"x": pole_x},
        sigma_m=(20.0, 0.0, 0.0),
    )
    iso_x = DispersiveMaterial(mu_inf=2.0, mu_poles=pole_x, sigma_m=20.0)

    assert aniso.is_mu_anisotropic
    assert aniso.mu(f).shape == (3, f.size)
    np.testing.assert_allclose(aniso.mu(f, "y"), 1.0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(aniso.mu(f, "x"), iso_x.mu(f), rtol=1e-14)

    # Polarization 1: pol='x' excites Ex and Hy -> sees mu_y = 1 (vacuum)
    (below_vac, above_vac), _ = _slab_fluxes(f, dz=5e-3, pol="x", use_gpu=use_gpu)
    (below_hy, above_hy), _ = _slab_fluxes(
        f, aniso, thickness=0.1, dz=5e-3, pol="x", use_gpu=use_gpu
    )
    np.testing.assert_allclose(below_hy, below_vac, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(above_hy, above_vac, rtol=1e-14, atol=1e-14)

    # Polarization 2: pol='y' excites Ey and Hx -> sees mu_x = iso_x
    (below_hx, above_hx), _ = _slab_fluxes(
        f, aniso, thickness=0.1, dz=5e-3, pol="y", use_gpu=use_gpu
    )
    (below_iso, above_iso), _ = _slab_fluxes(
        f, iso_x, thickness=0.1, dz=5e-3, pol="y", use_gpu=use_gpu
    )
    np.testing.assert_allclose(below_hx, below_iso, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(above_hx, above_iso, rtol=1e-14, atol=1e-14)


# -------------------------------------------------- 5. Passivity / energy


def _cavity_discrete_energy(solver, E_n, H_prev, include_aux=True):
    """
    Exact discrete quadratic invariant of the leapfrog + Lorentz-mu ADE
    scheme in a closed PEC cavity filled with a homogeneous Lorentz-mu
    medium (weight = 1 on all active H edges):

        U^n = 1/2 <E^n, M_E E^n> + 1/2 <H^(n+1/2), M_H mu_inf H^(n-1/2)>
              + 1/(2 dmu w0^2) <(m^(n+1/2) - m^(n-1/2))/dt,
                                M_H (m^(n+1/2) - m^(n-1/2))/dt>
              + 1/(2 dmu) <m^(n+1/2), M_H m^(n-1/2)>

    where M_E = diag(L * tA / ieps) and M_H = diag(mu_0 * A * tL).
    """
    L = solver.grid.L.toarray()
    itA = solver.grid.itA.toarray()
    tA = np.where(itA > 0, 1.0 / np.where(itA > 0, itA, 1.0), 0.0)
    ieps = np.asarray(
        solver.ieps.array.get() if solver.use_gpu else solver.ieps.array
    )
    M_E = np.where(ieps > 0, L * tA / np.where(ieps > 0, ieps, 1.0), 0.0)

    iA = solver.grid.iA.toarray()
    A = np.where(iA > 0, 1.0 / np.where(iA > 0, iA, 1.0), 0.0)
    tL = solver.grid.tL.toarray()
    M_H = mu_0 * A * tL

    H_new = np.asarray(solver.H.array.get() if solver.use_gpu else solver.H.array)
    g = solver._dispersive_h_groups[0]
    idx = g["idx_host"]
    mat = g["material"]
    pole = mat.mu_lorentz_poles()[0]
    mu_inf = mat.mu_inf
    dt = solver.dt

    u_e = 0.5 * np.dot(E_n, M_E * E_n)
    u_h = 0.5 * mu_inf * np.dot(H_new[idx], M_H[idx] * H_prev[idx])
    if not include_aux:
        return u_e + u_h

    _, _, _, m_new, m_old = g["lorentz"][0]
    m_new = np.asarray(m_new.get() if hasattr(m_new, "get") else m_new)
    m_old = np.asarray(m_old.get() if hasattr(m_old, "get") else m_old)
    dm_dt = (m_new - m_old) / dt
    mh = M_H[idx]
    u_m = 0.5 / pole.strength * np.dot(dm_dt, mh * dm_dt) + 0.5 / pole.delta_eps * np.dot(
        m_new, mh * m_old
    )
    return u_e + u_h + u_m


def test_passivity_and_discrete_energy_conservation(use_gpu):
    """
    In a closed PEC cavity filled with a lossless Lorentz-mu medium
    (gamma = 0), the discrete total energy (electromagnetic + auxiliary
    magnetization energy) is conserved to round-off (~1e-14) over 10,000
    steps after the source turns off. Omitting the auxiliary energy fails
    conservation by > 15%. With gamma > 0, the discrete energy decays
    monotonically at every single step.
    """
    L = 0.06
    grid = wakis.GridFIT3D(0, L, 0, L, 0, L, 6, 6, 6, verbose=0)
    f0 = 2.5e9
    w0 = 2 * np.pi * f0

    def run_cavity(gamma, n_post):
        solver = wakis.SolverFIT3D(
            grid,
            bc_low=["pec"] * 3,
            bc_high=["pec"] * 3,
            cfln=0.5,
            use_gpu=use_gpu,
            verbose=0,
        )
        mat = DispersiveMaterial(
            mu_inf=1.5, mu_poles=[Lorentz(delta_eps=1.2, omega_0=w0, gamma=gamma)]
        )
        solver.add_dispersive_material(
            mat, center=(L / 2, L / 2, L / 2), size=(2 * L, 2 * L, 2 * L)
        )
        src_time = GaussianSource(frequency=f0, fwidth=0.5 * f0)
        src = CurrentSource(
            src_time, component="z", center=(0.02, 0.02, 0.03), size=(0.0, 0.0, 0.0)
        )
        n_exc = int(np.ceil(src_time.end_time / solver.dt)) + 2
        for n in range(n_exc):
            src.update(solver, n * solver.dt)
            solver.one_step()
        solver.J.fromarray(solver.J.xp.zeros_like(solver.J.array))

        energies = np.empty(n_post)
        em_only = np.empty(n_post)
        for k in range(n_post):
            E_n = np.asarray(
                solver.E.array.get() if solver.use_gpu else solver.E.array
            ).copy()
            H_prev = np.asarray(
                solver.H.array.get() if solver.use_gpu else solver.H.array
            ).copy()
            solver.one_step()
            energies[k] = _cavity_discrete_energy(solver, E_n, H_prev, include_aux=True)
            em_only[k] = _cavity_discrete_energy(solver, E_n, H_prev, include_aux=False)
        return energies, em_only

    # Lossless: 10,000 steps of free ring-down conserve total discrete energy
    u_lossless, u_em_only = run_cavity(gamma=0.0, n_post=10000)
    rel_drift = np.max(np.abs(u_lossless - u_lossless[0])) / u_lossless[0]
    assert rel_drift < 1e-12
    # Mutation check: without the auxiliary magnetization energy, energy oscillates
    assert np.ptp(u_em_only) / u_lossless[0] > 0.15

    # Lossy (gamma > 0): monotonic decay at every step
    u_lossy, _ = run_cavity(gamma=0.15 * w0, n_post=1500)
    diffs = np.diff(u_lossy)
    assert np.all(diffs <= 1e-15 * u_lossy[0])
    assert u_lossy[-1] < 0.05 * u_lossy[0]
    print(
        f"Cavity energy: lossless 10k-step rel_drift={rel_drift:.2e}, "
        f"lossy max(dE)/E0={np.max(diffs)/u_lossy[0]:.2e}, E_end/E0={u_lossy[-1]/u_lossy[0]:.3e}"
    )


# ---------------------------------------------- 6. Cavity eigenfrequencies


def test_cavity_eigenfrequencies_lorentz_mu(use_gpu):
    """
    A closed PEC cavity filled with a lossless Lorentz-mu medium splits each
    bare cavity mode k_mnp into two eigenfrequency branches (one below w_0,
    one above sqrt(mu_s / mu_inf) w_0) satisfying the dispersion relation

        k_mnp^2 = (w / c)^2 eps_r mu_discrete(w, dt).

    Harminv on the ring-down signal recovers both branches of the lowest
    modes to < 1e-4 relative frequency error (and < 1e-7 against the exact
    discrete grid wavenumber).
    """
    a, b, d = 0.10, 0.08, 0.06
    nx, ny, nz = 20, 16, 12
    dx = a / nx
    grid = wakis.GridFIT3D(0, a, 0, b, 0, d, nx, ny, nz, verbose=0)
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["pec"] * 3,
        bc_high=["pec"] * 3,
        cfln=0.5,
        use_gpu=use_gpu,
        verbose=0,
    )
    f_pole = 2.2e9
    w0 = 2 * np.pi * f_pole
    mu_inf, delta_mu = 1.2, 0.8
    mat = DispersiveMaterial(
        mu_inf=mu_inf, mu_poles=[Lorentz(delta_eps=delta_mu, omega_0=w0, gamma=0.0)]
    )
    solver.add_dispersive_material(
        mat, center=(a / 2, b / 2, d / 2), size=(2 * a, 2 * b, 2 * d)
    )

    # Excite TE110 (m=1, n=1, p=0) and TM/TE111 (m=1, n=1, p=1) modes with Ez
    src_time = GaussianSource(frequency=2.5e9, fwidth=1.8e9)
    src = CurrentSource(src_time, component="z", center=(0.035, 0.025, 0.02))
    n_exc = int(np.ceil(src_time.end_time / solver.dt)) + 2
    n_ring = 4000
    sig = np.empty(n_ring)
    for n in range(n_exc + n_ring):
        if n < n_exc:
            src.update(solver, n * solver.dt)
        elif n == n_exc:
            solver.J.fromarray(solver.J.xp.zeros_like(solver.J.array))
        solver.one_step()
        if n >= n_exc:
            sig[n - n_exc] = float(solver.E[13, 11, 7, "z"])

    res = harminv(sig, solver.dt, 1.0e9, 4.2e9)
    # Keep physical modes with significant amplitude
    amps = np.abs(res.amplitude)
    found = np.sort(res.frequency[amps > 1e-3 * amps.max()])

    dt = solver.dt
    # Predict the two branches of (1, 1, 0) and (1, 1, 1):
    # w_tilde^2 (mu_inf + delta_mu w0^2 / (w0^2 - w_tilde^2)) = c^2 k_grid^2
    expected_discrete = []
    for m, n_y, p in ((1, 1, 0), (1, 1, 1)):
        k2_grid = sum(
            (2 / dx * np.sin(q * np.pi * dx / (2 * ell))) ** 2
            for q, ell in ((m, a), (n_y, b), (p, d))
        )
        # Quadratic in x = w_tilde^2: mu_inf x^2 - (mu_s w0^2 + c^2 k2) x + c^2 k2 w0^2 = 0
        mu_s = mu_inf + delta_mu
        roots_x = np.roots([mu_inf, -(mu_s * w0**2 + c**2 * k2_grid), c**2 * k2_grid * w0**2])
        for x in np.sort(roots_x):
            w_tilde = np.sqrt(x)
            w_d = 2 / dt * np.arcsin(w_tilde * dt / 2)
            expected_discrete.append(w_d / (2 * np.pi))

    expected_discrete = np.sort(expected_discrete)
    rel_f_errs, rel_k2_errs = [], []
    for f_exp in expected_discrete:
        idx = np.argmin(np.abs(found - f_exp))
        f_harminv = found[idx]
        # Exact discrete dispersion relation match (~1e-7)
        rel_f = abs(f_harminv / f_exp - 1.0)
        rel_f_errs.append(rel_f)
        assert rel_f < 1e-4
        # Direct check of k_grid^2 = (w_tilde/c)^2 mu_discrete(f, dt)
        w_tilde = 2 / dt * np.sin(np.pi * f_harminv * dt)
        lhs = (w_tilde / c) ** 2 * mat.mu_discrete(f_harminv, dt).real
        # Match against one of the cavity k_grid^2 eigenvalues
        k2_list = [
            sum(
                (2 / dx * np.sin(q * np.pi * dx / (2 * ell))) ** 2
                for q, ell in ((1, a), (1, b), (p, d))
            )
            for p in (0, 1)
        ]
        rel_k2 = min(abs(lhs / k2 - 1.0) for k2 in k2_list)
        rel_k2_errs.append(rel_k2)
        assert rel_k2 < 1e-4
    print(
        f"Cavity Harminv 4 branches: max|df/f|={max(rel_f_errs):.2e}, "
        f"max|dk^2/k^2|={max(rel_k2_errs):.2e}"
    )


# -------------------------------------------------------------- 7. Adjoint


@pytest.mark.parametrize("bc", ["pec", "cpml", "bloch"])
def test_adjoint_dot_product_magnetic_groups(bc, use_gpu):
    """
    Dot-product identity <U x, y> = <x, U^H y> to ~1e-14 for a solver with
    anisotropic and isotropic magnetic dispersive groups (Lorentz + Drude +
    Debye + sigma_m) and electric dispersive groups, across PEC, CPML and
    complex Bloch boundary conditions. Perturbing a magnetic group's
    transposed coefficient breaks the identity.
    """
    if bc == "cpml":
        lo, hi = ["periodic", "periodic", "cpml"], ["periodic", "periodic", "cpml"]
    elif bc == "bloch":
        lo, hi = ["bloch"] * 3, ["bloch"] * 3
    else:
        lo, hi = ["pec"] * 3, ["pec"] * 3
    dtype = np.complex128 if bc == "bloch" else np.float64
    kw = {"k_bloch": (15.0, -25.0, 35.0)} if bc == "bloch" else {}

    grid = wakis.GridFIT3D(0, 0.05, 0, 0.04, 0, 0.08, 5, 4, 8, verbose=0)
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=lo,
        bc_high=hi,
        n_pml=2,
        cfln=0.5,
        dtype=dtype,
        use_gpu=use_gpu,
        verbose=0,
        **kw,
    )
    w0 = 2 * np.pi * 2e9
    mat_aniso = DispersiveMaterial(
        eps_inf=1.4,
        poles=[Debye(0.5, 1 / w0)],
        sigma=0.01,
        mu_inf=(1.6, 1.3, 1.1),
        mu_poles={
            "x": [Lorentz(0.7, w0, 0.1 * w0), Debye(0.4, 1 / w0)],
            "y": [Drude(0.8 * w0, 0.15 * w0)],
            "z": [Lorentz(0.5, 1.2 * w0, 0.05 * w0)],
        },
        sigma_m=(10.0, 5.0, 0.0),
    )
    solver.add_dispersive_material(
        mat_aniso, center=(0.025, 0.02, 0.04), size=(0.03, 0.02, 0.03)
    )
    solver.one_step()

    stepper = AdjointStepper(solver)
    xp = stepper.xp
    rng = np.random.default_rng(42)

    def rand_state():
        out = {}
        for k, v in stepper.zero_state().items():
            r = rng.standard_normal(v.shape)
            if np.iscomplexobj(v):
                r = r + 1j * rng.standard_normal(v.shape)
            out[k] = xp.asarray(r, dtype=v.dtype)
        return out

    x, y = rand_state(), rand_state()
    stepper.set_state(x)
    solver.one_step()
    Ux = stepper.get_state()
    Uhy = stepper.step(y)
    lhs = stepper.inner(Ux, y)
    rhs = stepper.inner(x, Uhy)
    scale = stepper.inner(
        {k: abs(v) for k, v in Ux.items()}, {k: abs(v) for k, v in y.items()}
    ).real
    rel_dp = abs(lhs - rhs) / scale
    print(f"Adjoint dot-product [{bc}]: |<Ux,y>-<x,U^H y>|/scale = {rel_dp:.2e}")
    assert rel_dp < 1e-13

    # Mutation check: perturbing a1 in a magnetic transpose group breaks <Ux, y> == <x, U^H y>
    stepper.dispersive_h[0].a1 = stepper.dispersive_h[0].a1 * 1.1
    Uhy_bad = stepper.step(y)
    assert abs(lhs - stepper.inner(x, Uhy_bad)) > 1e-4 * scale


def test_adjoint_gradient_with_dispersive_mu_region(use_gpu):
    """
    Finite-difference verification of the adjoint gradient dF/drho with
    respect to a permittivity design (`MaterialGrid`) in a domain containing
    a lossy Lorentz + Debye dispersive-mu region, for both 'store' and
    'checkpoint' memory modes (and verifying that 'reverse' falls back to
    'checkpoint' with a warning).
    """
    w0 = 2 * np.pi * 2.5e9
    mat_mu = DispersiveMaterial(
        mu_inf=1.5,
        mu_poles=[Lorentz(0.8, w0, 0.1 * w0), Debye(0.5, 1 / w0)],
        sigma_m=15.0,
    )

    def make_solver():
        grid = wakis.GridFIT3D(0, 0.04, 0, 0.04, 0, 0.06, 4, 4, 6, verbose=0)
        s = wakis.SolverFIT3D(
            grid,
            bc_low=["pec", "pec", "cpml"],
            bc_high=["pec", "pec", "cpml"],
            n_pml=2,
            cfln=0.5,
            use_gpu=use_gpu,
            verbose=0,
        )
        s.add_dispersive_material(
            mat_mu, center=(0.02, 0.02, 0.038), size=(0.03, 0.03, 0.015)
        )
        return s

    design = MaterialGrid(
        center=(0.02, 0.02, 0.023),
        size=(0.02, 0.02, 0.01),
        shape=(2, 2, 1),
        eps_min=1.0,
        eps_max=4.0,
    )
    f_mon = np.array([2.0e9, 2.8e9])
    src_time = GaussianSource(frequency=2.4e9, fwidth=1.0e9)
    src = CurrentSource(src_time, component="x", center=(0.02, 0.02, 0.015))
    mon = wakis.DFTMonitor(
        f_mon, components=["Ex"], center=(0.02, 0.02, 0.042), size=(0.02, 0.02, 0.0)
    )

    Nt = 60
    sim = AdjointSimulation(
        make_solver, [src], {"dft": mon}, design, Nt=Nt, memory="store"
    )
    rho0 = np.array([[[0.3], [0.7]], [[0.5], [0.2]]])

    def objective(rho, record=False):
        out = sim.forward(rho, record=record)["dft"]["Ex"]
        return float(np.sum(np.abs(out) ** 2).real), {"dft": {"Ex": 2.0 * out}}

    _, cot = objective(rho0, record=True)
    grad_store = sim.vjp(cot)

    # Check checkpoint mode (and 'reverse' fallback warning) gives identical gradient
    sim_rev = AdjointSimulation(
        make_solver, [src], {"dft": mon}, design, Nt=Nt, memory="reverse"
    )
    with pytest.warns(UserWarning, match="dispersive media"):
        sim_rev.forward(rho0, record=True)
    grad_chk = sim_rev.vjp(cot)
    np.testing.assert_allclose(grad_chk, grad_store, rtol=1e-13, atol=1e-15)

    # Central finite differences on all 4 design variables
    eps_fd = 1e-6
    grad_fd = np.zeros_like(rho0)
    for idx in np.ndindex(rho0.shape):
        dp = np.zeros_like(rho0)
        dp[idx] = eps_fd
        fp, _ = objective(rho0 + dp, record=False)
        fm, _ = objective(rho0 - dp, record=False)
        grad_fd[idx] = (fp - fm) / (2 * eps_fd)

    rel_grad = np.max(np.abs(grad_store - grad_fd) / np.abs(grad_fd))
    print(f"Adjoint vs FD gradient max rel err = {rel_grad:.2e}")
    np.testing.assert_allclose(grad_store, grad_fd, rtol=1e-5)


# -------------------------------------- 8. Mixed electric & magnetic media


def test_mixed_electric_and_magnetic_dispersion(use_gpu):
    """
    A region with distinct dispersive eps(w) and dispersive mu(w):
    1. Joint stability check: a timestep where the coupled plane-wave
       amplification matrix is unstable is caught by `is_stable`.
    2. Slab reflectance and transmittance match the general transfer-matrix
       prediction with n_d = sqrt(eps_d * mu_d) and eta_d = sqrt(mu_d / eps_d).
    3. `reset_dispersive_state()` zeroes both polarizations and magnetizations.
    """
    # Joint stability: with eps_inf = 1.2, mu_inf = 1.2 and strong Lorentz poles
    # in both eps and mu, the coupled mode at Courant C = 0.8 has a stricter
    # stability limit than at C = 0.2
    w0 = 1.0
    mixed_stab = DispersiveMaterial(
        eps_inf=1.2,
        poles=[Lorentz(2.0, w0, 0.0)],
        mu_inf=1.2,
        mu_poles=[Lorentz(2.0, w0, 0.0)],
    )
    assert mixed_stab.is_stable(dt=0.5, courant=0.5)
    assert not mixed_stab.is_stable(dt=1.3, courant=0.95)

    # Mixed slab R(f) and T(f) with eps(w) != mu(w)
    f = np.linspace(0.7e9, 1.3e9, 7)
    mixed = DispersiveMaterial(
        eps_inf=2.2,
        poles=[Debye(delta_eps=1.8, tau=1 / (2 * np.pi * 1.1e9))],
        sigma=0.01,
        mu_inf=1.6,
        mu_poles=[
            Lorentz(delta_eps=1.0, omega_0=2 * np.pi * 1.2e9, gamma=2 * np.pi * 0.15e9)
        ],
        sigma_m=10.0,
    )
    R, T, dt = _reflectance(mixed, f, thickness=0.1, dz=2.5e-3, use_gpu=use_gpu)
    R_an, T_an, _, _ = _slab_rt_general(
        mixed.epsilon_discrete(f, dt), mixed.mu_discrete(f, dt), f, 0.1
    )
    np.testing.assert_allclose(R, R_an, atol=6e-3)
    np.testing.assert_allclose(T, T_an, atol=3e-3)

    # Verify reset_dispersive_state zeroes both electric and magnetic auxiliaries
    grid = wakis.GridFIT3D(0, 0.04, 0, 0.04, 0, 0.04, 4, 4, 4, verbose=0)
    solver = wakis.SolverFIT3D(grid, use_gpu=use_gpu, verbose=0)
    solver.add_dispersive_material(mixed, center=(0.02, 0.02, 0.02), size=(1, 1, 1))
    solver.E[:, :, :, "x"] = 1.0
    solver.H[:, :, :, "y"] = 1.0
    solver.one_step()
    assert np.any(solver._dispersive_groups[0]["debye"][0][2] != 0)
    assert np.any(solver._dispersive_h_groups[0]["lorentz"][0][3] != 0)
    solver.reset_dispersive_state()
    assert np.all(solver._dispersive_groups[0]["debye"][0][2] == 0)
    assert np.all(solver._dispersive_h_groups[0]["lorentz"][0][3] == 0)
    assert np.all(solver._dispersive_h_groups[0]["lorentz"][0][4] == 0)
