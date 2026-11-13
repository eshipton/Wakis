# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Tests for the image-charge wall stencil ``ParticleSolver(wall_stencil="image")``.

Derivation: ``wakis/particles/wall_images.py`` and ``first_principles.md`` ("image wall stencil").

1. Gauss's law ``<= 1e-12`` on every interior vacuum plane from PEC birth through PEC absorption
   for ``p = 1, 2, 3`` (z walls), and at an x-z corner (double images).
2. A particle on the wall has no visible charge: the birth deposit (real + image) is exactly zero,
   no closure current, ``wall_kick_energy() == 0`` and ``q_bg`` untouched.
3. Gather/deposit consistency: for a discrete gradient field ``E = -grad_h phi`` the image gather
   equals ``-grad`` of the image-gathered potential (normal E even <=> phi odd), checked against a
   central difference near both z walls and an x wall; a sign mutation of the image rule fails.
4. Mechanism: in the first half cell the order-2 Galerkin gather of ``E_z`` is ``E_0`` (image) vs
   ``(1/2 + zeta) E_0`` (masked).
5. First-cell leapfrog energy error of a wall-born particle in a frozen gradient field: the
   single-step jump at the ``zeta = 1/2`` knot matches the closed-form trapezoid error with
   ``|E_1 - 2 E_0|`` (masked) and ``|E_1 - E_0|`` (image) to 2 %, and the image error is smaller
   by the predicted ratio.
6. CPU vs GPU agreement in image mode (skipped without CuPy).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import c as c_light
from scipy.constants import e as e_charge
from scipy.constants import m_e

from wakis.gridFIT3D import GridFIT3D
from wakis.particles import BeamInjector, ParticleSolver, Species, gauss_residual
from wakis.particles.shapes import _gather_component_3d, bspline_weights_1d, gather_fields
from wakis.solverFIT3D import SolverFIT3D


def _box(nx=6, ny=1, nz=40, bc_x="pec", bc_y="periodic", lx=0.06, lz=0.04, ly=0.01):
    grid = GridFIT3D(-0.5 * lx, 0.5 * lx, -0.5 * ly, 0.5 * ly, 0.0, lz, nx, ny, nz, verbose=0)
    solver = SolverFIT3D(
        grid,
        cfln=0.5,
        bc_low=[bc_x, bc_y, "pec"],
        bc_high=[bc_x, bc_y, "pec"],
        use_stl=False,
        verbose=0,
    )
    return grid, solver


def _transit(order, n_macro, wall_stencil, x0=0.0, radius=0.008, nx=6, track_energy=False):
    grid, solver = _box(nx=nx)
    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    gamma0 = 2.5
    vz0 = float(np.sqrt(1.0 - 1.0 / gamma0**2)) * c_light
    dt = float(solver.dt)

    def _pulse(t):
        return 0.2 if t <= 2.75 * dt else 0.0

    inj = BeamInjector(
        species=sp, z0=0.0, current=_pulse, gamma=gamma0, profile="sheet_lattice",
        radius=radius, x0=x0, n_macro_per_step=n_macro, seed=1234,
    )
    ps = ParticleSolver(
        solver, species=[sp], injectors=[inj], pusher="boris", shape=order, gather="galerkin",
        B_ext=(0.0, 0.0, 0.2), wall_stencil=wall_stencil, track_energy=track_energy,
    ).attach()
    n_steps = int(np.ceil(0.04 / (vz0 * dt))) + 14
    res = []
    for _ in range(n_steps):
        solver.one_step()
        res.append(float(gauss_residual(solver, sp, order=order, include_wall_planes=True)))
    return solver, ps, sp, inj, res


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("n_macro", [1, 16])
def test_image_gauss_birth_transit_absorption(order, n_macro):
    solver, ps, sp, inj, res = _transit(order, n_macro, "image")
    assert ps._wall_images is not None
    assert inj.emitter_kind == "pec_wall"
    assert sp.n_particles == 0 and ps.boundaries.absorbed_count["z_high"] == 3 * n_macro
    print(f"[image p={order} N={n_macro}] max gauss = {max(res):.3e}")
    assert max(res) <= 1e-12


def test_image_gauss_mutation_fails(monkeypatch):
    """Dropping the image current deposit (keeping the image charge in the Gauss check) breaks Gauss."""
    from wakis.particles import wall_images

    monkeypatch.setattr(wall_images.WallImages, "deposit_current", lambda self, *a, **k: None)
    _, _, _, _, res = _transit(2, 16, "image")
    print(f"[mutated] max gauss = {max(res):.3e}")
    assert max(res) > 1e-3


def test_image_gauss_at_corner_double_images():
    """Particles born half a cell from the x-low PEC wall on the z-low PEC wall (x-z corner)."""
    nx = 8
    dx = 0.06 / nx
    _, ps, sp, _, res = _transit(2, 4, "image", x0=-0.03 + 1.2 * dx, radius=0.9 * dx, nx=nx)
    print(f"[corner] max gauss = {max(res):.3e}")
    assert max(res) <= 1e-12


@pytest.mark.parametrize("order", [2, 3])
def test_image_birth_has_no_visible_charge_kick_or_closure(order):
    grid, solver = _box()
    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    inj = BeamInjector(
        species=sp, z0=0.0, current=0.2, gamma=2.5, profile="sheet_lattice", radius=0.008,
        n_macro_per_step=16,
    )
    ps = ParticleSolver(
        solver, species=[sp], injectors=[inj], shape=order, B_ext=(0.0, 0.0, 0.2),
        wall_stencil="image", track_energy=True,
    ).attach()
    # Visible birth charge of real + image at zeta = 0 is exactly zero.
    xi, eta, zeta = ps.grid_map.to_logical(
        np.linspace(-0.008, 0.008, 16), np.zeros(16), np.zeros(16)
    )
    from wakis.particles import deposit_charge

    q = np.zeros((solver.Nx, solver.Ny, solver.Nz), order="F")
    deposit_charge(xi, eta, zeta, -np.ones(16), solver.Nx, solver.Ny, solver.Nz, order=order,
                   periodic=ps.boundaries.periodic, q_nodes=q)
    assert np.max(np.abs(q)) > 0.0
    ps._image_charge(xi, eta, zeta, -np.ones(16), q)
    assert np.max(np.abs(q)) == 0.0
    for _ in range(5):
        solver.one_step()
    assert ps.wall_kick_energy() == 0.0
    assert ps.wall_closure_work() == 0.0
    assert np.max(np.abs(solver._particles_q_bg)) == 0.0
    # Masked mode (released default) does need both (control: this test can fail).
    grid2, solver2 = _box()
    sp2 = Species(name="electron", charge=-e_charge, mass=m_e)
    inj2 = BeamInjector(
        species=sp2, z0=0.0, current=0.2, gamma=2.5, profile="sheet_lattice", radius=0.008,
        n_macro_per_step=16,
    )
    ps2 = ParticleSolver(
        solver2, species=[sp2], injectors=[inj2], shape=order, B_ext=(0.0, 0.0, 0.2),
        track_energy=True,
    ).attach()
    for _ in range(5):
        solver2.one_step()
    assert ps2._wall_images is None
    assert abs(ps2.wall_kick_energy()) > 0.0
    assert np.max(np.abs(solver2._particles_q_bg)) > 0.0


def _random_gradient_field(solver, rng):
    """Node potential vanishing on the PEC walls and its discrete gradient ``E = -grad_h phi``."""
    Nx, Ny, Nz = solver.Nx, solver.Ny, solver.Nz
    nxp, nzp = Nx - 1, Nz - 1  # wall node indices (one ghost layer beyond)
    phi = np.zeros((Nx, Ny, Nz), order="F")
    phi[1:nxp, :, 1:nzp] = rng.standard_normal((nxp - 1, Ny, nzp - 1))
    dx = np.asarray(solver.grid.dx)
    dz = np.asarray(solver.grid.dz)
    ex = np.zeros_like(phi)
    ez = np.zeros_like(phi)
    ex[:nxp, :, :] = -(phi[1 : nxp + 1] - phi[:nxp]) / dx[:nxp, None, None]
    ez[:, :, :nzp] = -(phi[:, :, 1 : nzp + 1] - phi[:, :, :nzp]) / dz[None, None, :nzp]
    E = (ex.ravel(order="F"), np.zeros(phi.size), ez.ravel(order="F"))
    return phi, E


def _phi_gather(ps, phi, xi, eta, zeta, image_sign=-1.0):
    """Order-p nodal gather of phi with odd (``image_sign=-1``) wall images."""
    s = ps.solver
    p = ps.shape
    per = ps.boundaries.periodic

    def g(a, b, c):
        ix, wx = bspline_weights_1d(a, p, staggered=False)
        iy, wy = bspline_weights_1d(b, p, staggered=False)
        iz, wz = bspline_weights_1d(c, p, staggered=False)
        return _gather_component_3d(phi.ravel(order="F"), s.Nx, s.Ny, s.Nz, ix, wx, iy, wy, iz, wz, per, np)

    out = g(xi, eta, zeta)
    wi = ps._wall_images
    for S, idx, wall in wi.image_sets(([xi], [eta], [zeta])):
        m = wi._mirror((xi, eta, zeta), S, idx, wall)
        out[idx] += (image_sign ** len(S)) * g(*m)
    return out


@pytest.mark.parametrize("order", [2, 3])
def test_image_gather_is_gradient_of_image_potential(order):
    """``E_g = -grad Phi_g`` with images (z-low, z-high, x-low); a sign mutation fails."""
    rng = np.random.default_rng(7)
    grid, solver = _box(nx=8, ny=1, nz=12)
    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    ps = ParticleSolver(solver, species=[sp], shape=order, gather="galerkin", wall_stencil="image").attach()
    phi, E = _random_gradient_field(solver, rng)
    B0 = tuple(np.zeros(phi.size) for _ in range(3))
    n = 200
    zeta = np.concatenate([rng.uniform(0.0, 2.5, n), rng.uniform(9.5, 12.0, n)])
    xi = np.concatenate([rng.uniform(0.0, 8.0, n), rng.uniform(0.0, 2.5, n)])
    eta = np.full(2 * n, 0.5)
    dxl, dzl = float(solver.grid.dx[0]), float(solver.grid.dz[0])
    g = gather_fields(xi, eta, zeta, E, B0, solver.Nx, solver.Ny, solver.Nz, order=order,
                      gather="galerkin", periodic=ps.boundaries.periodic)
    corr = ps._image_field(xi, eta, zeta, E, B0)
    ez_img = g[2] + corr[2]
    ex_img = g[0] + corr[0]
    h = 1e-6
    # stay away from the wall itself for the central difference
    inside = (zeta > h) & (zeta < 12.0 - h) & (xi > h) & (xi < 8.0 - h)
    dphi_dz = (_phi_gather(ps, phi, xi, eta, zeta + h) - _phi_gather(ps, phi, xi, eta, zeta - h)) / (2 * h * dzl)
    dphi_dx = (_phi_gather(ps, phi, xi + h, eta, zeta) - _phi_gather(ps, phi, xi - h, eta, zeta)) / (2 * h * dxl)
    scale = np.max(np.abs(ez_img))
    err_z = np.max(np.abs(ez_img + dphi_dz)[inside]) / scale
    err_x = np.max(np.abs(ex_img + dphi_dx)[inside]) / scale
    print(f"[p={order}] |E_g + grad Phi_g| / max|E|: z {err_z:.2e}, x {err_x:.2e}")
    assert err_z < 1e-6 and err_x < 1e-6
    # Masked gather (no images) is NOT the gradient of the image potential near the wall.
    err_masked = np.max(np.abs(g[2] + dphi_dz)[inside]) / scale
    assert err_masked > 1e-2
    # Mutation: even potential images (wrong PEC parity) break the identity.
    dphi_dz_bad = (
        _phi_gather(ps, phi, xi, eta, zeta + h, image_sign=1.0)
        - _phi_gather(ps, phi, xi, eta, zeta - h, image_sign=1.0)
    ) / (2 * h * dzl)
    assert np.max(np.abs(ez_img + dphi_dz_bad)[inside]) / scale > 1e-2


def test_image_first_half_cell_gather_is_constant():
    grid, solver = _box(nx=6, nz=20)
    rng = np.random.default_rng(3)
    phi, E = _random_gradient_field(solver, rng)
    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    B0 = tuple(np.zeros(phi.size) for _ in range(3))
    zeta = np.linspace(0.0, 0.49, 50)
    xi = np.full(50, 3.0)  # on a node: transverse weights 3/4, 1/8, 1/8 around node 3
    eta = np.full(50, 0.5)
    for ws in ("masked", "image"):
        s2 = SolverFIT3D(grid, cfln=0.5, bc_low=["pec", "periodic", "pec"],
                         bc_high=["pec", "periodic", "pec"], use_stl=False, verbose=0)
        ps = ParticleSolver(s2, species=[sp], shape=2, gather="galerkin", wall_stencil=ws)
        g = gather_fields(xi, eta, zeta, E, B0, s2.Nx, s2.Ny, s2.Nz, order=2, gather="galerkin",
                          periodic=ps.boundaries.periodic)
        ez = g[2] if ws == "masked" else g[2] + ps._image_field(xi, eta, zeta, E, B0)[2]
        ez0 = g[2][0] / 0.5  # masked: E_g(0) = E0 / 2
        if ws == "masked":
            np.testing.assert_allclose(ez, (0.5 + zeta) * ez0, rtol=1e-13)
        else:
            np.testing.assert_allclose(ez, np.full_like(zeta, ez0), rtol=1e-13)


def _frozen_wall_birth(wall_stencil, dt_scale=1.0):
    """One electron born at z = 0 in a frozen gradient field; return excess history and fields."""
    nx, nz = 6, 60
    a, lz = 0.03, 0.06
    grid = GridFIT3D(-a, a, -0.005, 0.005, 0.0, lz, nx, 1, nz, verbose=0)
    solver = SolverFIT3D(grid, cfln=0.5, bc_low=["pec", "periodic", "pec"],
                         bc_high=["pec", "periodic", "pec"], use_stl=False, verbose=0)
    solver.dt = float(solver.dt) * dt_scale
    dt = float(solver.dt)
    Nx, Ny, Nz = solver.Nx, solver.Ny, solver.Nz
    x_n = np.asarray(grid.x[: nx + 1])
    z_n = np.asarray(grid.z[: nz + 1])
    L = 0.01
    phi = np.zeros((Nx, Ny, Nz), order="F")
    prof_z = 1.0 - np.exp(-z_n / L)
    prof_z[-1] = 0.0
    phi[: nx + 1, 0, : nz + 1] = -2.0e3 * (1.0 - (x_n / a) ** 2)[:, None] * prof_z[None, :]
    dz = lz / nz
    dxx = 2 * a / nx
    ex = np.zeros_like(phi)
    ez = np.zeros_like(phi)
    ex[:nx] = -(phi[1 : nx + 1] - phi[:nx]) / dxx
    ez[:, :, :nz] = -(phi[:, :, 1 : nz + 1] - phi[:, :, :nz]) / dz
    solver.E.from_matrix(ex, "x")
    solver.E.from_matrix(np.zeros_like(ex), "y")
    solver.E.from_matrix(ez, "z")
    solver.one_step = solver._one_step
    solver.step_0 = False
    solver._update_H = lambda: None
    solver._update_E = lambda: None
    gamma0 = 1.05
    sp = Species(name="e", charge=-e_charge, mass=m_e, capacity=4)
    inj = BeamInjector(species=sp, x0=0.0, y0=0.0, z0=0.0, radius=0.0, current=e_charge / dt,
                       gamma=gamma0, n_macro_per_step=1, profile="lattice")
    ps = ParticleSolver(solver, [sp], injectors=[inj], shape=2, gather="galerkin",
                        B_ext=(0.0, 0.0, 5.0), wall_stencil=wall_stencil).attach()
    q_mc2 = -e_charge / (m_e * c_light**2)
    B0 = tuple(np.zeros(phi.size) for _ in range(3))
    E = (ex.ravel(order="F"), np.zeros(phi.size), ez.ravel(order="F"))
    hist = []
    for k in range(80):
        solver.one_step()
        if k == 0:
            ps.injectors.clear()
        xi, eta, zeta = ps.grid_map.to_logical(sp.x[:1], sp.y[:1], sp.z[:1])
        g = gather_fields(xi, eta, zeta, E, B0, Nx, Ny, Nz, order=2, gather="galerkin",
                          periodic=ps.boundaries.periodic)
        corr = ps._image_field(xi, eta, zeta, E, B0)
        ez_g = g[2] + (corr[2] if corr is not None else 0.0)
        uz = sp.uz[0] + (-e_charge / m_e) * ez_g[0] * 0.5 * dt  # synchronise to t_n
        gam = np.sqrt(1.0 + (uz / c_light) ** 2)
        if wall_stencil == "image":
            phig = _phi_gather(ps, phi, xi, eta, zeta)[0]
        else:
            phig = _phi_gather_masked(ps, phi, xi, eta, zeta)[0]
        hist.append((float(zeta[0]), float(gam - gamma0 + q_mc2 * phig)))
        if zeta[0] > 4.0:
            break
    # line-gathered E0, E1 at x = 0 (node 3: transverse weights 1/8, 3/4, 1/8)
    w = np.array([0.125, 0.75, 0.125])
    e0 = float(w @ ez[2:5, 0, 0])
    e1 = float(w @ ez[2:5, 0, 1])
    return hist, e0, e1, dz


def _phi_gather_masked(ps, phi, xi, eta, zeta):
    s = ps.solver
    ix, wx = bspline_weights_1d(xi, 2, staggered=False)
    iy, wy = bspline_weights_1d(eta, 2, staggered=False)
    iz, wz = bspline_weights_1d(zeta, 2, staggered=False)
    return _gather_component_3d(phi.ravel(order="F"), s.Nx, s.Ny, s.Nz, ix, wx, iy, wy, iz, wz,
                                ps.boundaries.periodic, np)


def _knot_jump(hist):
    z = np.array([h[0] for h in hist])
    ex = np.array([h[1] for h in hist])
    m = int(np.argmax(z >= 0.5))
    dzeta = z[m] - z[m - 1]
    theta = (z[m] - 0.5) / dzeta
    return ex[m] - ex[m - 1], theta, dzeta, ex


def test_first_cell_knot_error_masked_vs_image():
    q_abs = e_charge / (m_e * c_light**2)
    out = {}
    for ws in ("masked", "image"):
        hist, e0, e1, dz = _frozen_wall_birth(ws)
        jump, theta, dzeta, ex = _knot_jump(hist)
        J = (e1 - 2 * e0) if ws == "masked" else (e1 - e0)
        pred = q_abs * dz * 0.5 * theta * (1 - theta) * dzeta**2 * abs(J)
        out[ws] = (jump, pred, ex)
        print(f"[{ws}] knot jump {jump:+.4e} pred {pred:.4e} theta {theta:.3f} dzeta {dzeta:.4f} "
              f"pre-knot |excess| max {np.max(np.abs(ex[: int(np.argmax(np.array([h[0] for h in hist]) >= 0.5))])):.2e}")
        assert abs(abs(jump) - pred) <= 0.02 * pred
    ratio_meas = abs(out["masked"][0]) / abs(out["image"][0])
    hist, e0, e1, _ = _frozen_wall_birth("masked")
    ratio_pred = abs(e1 - 2 * e0) / abs(e1 - e0)
    print(f"ratio masked/image: measured {ratio_meas:.3f}, predicted {ratio_pred:.3f}")
    assert abs(ratio_meas / ratio_pred - 1.0) < 0.03


def test_image_cpu_gpu_agreement():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no GPU")
    res = {}
    for use_gpu in (False, True):
        grid, _ = _box()
        solver = SolverFIT3D(grid, cfln=0.5, bc_low=["pec", "periodic", "pec"],
                             bc_high=["pec", "periodic", "pec"], use_stl=False, use_gpu=use_gpu, verbose=0)
        sp = Species(name="electron", charge=-e_charge, mass=m_e)
        inj = BeamInjector(species=sp, z0=0.0, current=0.2, gamma=2.5, profile="sheet_lattice",
                           radius=0.008, n_macro_per_step=16)
        ps = ParticleSolver(solver, species=[sp], injectors=[inj], shape=2, B_ext=(0.0, 0.0, 0.2),
                            wall_stencil="image").attach()
        for _ in range(40):
            solver.one_step()
        to_h = (lambda a: a.get()) if use_gpu else (lambda a: np.asarray(a))
        res[use_gpu] = (
            to_h(ps.J_particles).copy(),
            to_h(sp.uz).copy(),
            to_h(sp.z).copy(),
            float(gauss_residual(solver, sp, order=2, include_wall_planes=True)),
        )
    jc, jg = res[False][0], res[True][0]
    assert np.max(np.abs(jc - jg)) <= 1e-12 * np.max(np.abs(jc))
    assert np.max(np.abs(res[False][1] - res[True][1])) <= 1e-12 * np.max(np.abs(res[False][1]))
    assert np.max(np.abs(res[False][2] - res[True][2])) <= 1e-12 * np.max(np.abs(res[False][2]))
    assert res[True][3] <= 1e-12


def test_wall_stencil_validation():
    grid, solver = _box()
    sp = Species(name="electron", charge=-e_charge, mass=m_e)
    with pytest.raises(ValueError):
        ParticleSolver(solver, species=[sp], wall_stencil="mirror")
    with pytest.raises(ValueError):
        ParticleSolver(solver, species=[sp], wall_stencil="image", deposit_scheme="naive")
    ps = ParticleSolver(solver, species=[sp], shape=2)
    assert ps.wall_stencil == "masked" and ps._wall_images is None
