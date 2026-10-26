# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Validation tests for the photonic-integrated-circuit toolkit in
`wakis.templates.photonics` and subpixel Kottke averaging in
`wakis.templates.shapes`.
"""

import warnings

import numpy as np
import pytest
from scipy.constants import c as c_light

import wakis.templates as templates_mod
import wakis.templates.photonics as photonics_mod
from wakis.mode_solver import solve_modes
from wakis.subgrid import SubgridPhotonicCircuit
from wakis.templates import Box, Simulation
from wakis.templates.formulas import slab_waveguide_neff
from wakis.templates.photonics import (
    PHOTONIC_MATERIALS,
    PICSimulation,
    PhotonicCircuit,
    PolySlab,
    Port,
    euler_bend,
    n_to_eps,
    ring,
    sbend,
    spline_taper,
    straight,
)


def test_polyslab_fill_and_holes():
    """
    PolySlab.fill matches Box.weight in the interior of a square, integrates
    a right triangle to its exact area * thickness within 1e-3 at n_sub=8,
    and subtracts interior holes.
    """
    # 1. Square PolySlab vs Box in interior cells
    verts_sq = straight((-1.0e-6, 0.0), (1.0e-6, 0.0), width=2.0e-6)
    poly_sq = PolySlab(verts_sq, -0.1e-6, 0.1e-6, material=12.0)
    box_sq = Box((0.0, 0.0, 0.0), (2.0e-6, 2.0e-6, 0.2e-6), material=12.0)

    pts = np.linspace(-0.7e-6, 0.7e-6, 5)
    X, Y, Z = np.meshgrid(pts, pts, [0.0], indexing="ij")
    hx = np.full_like(X, 0.1e-6)
    hy = np.full_like(Y, 0.1e-6)
    hz = np.full_like(Z, 0.1e-6)
    fill_in = poly_sq.fill(X, Y, Z, hx, hy, hz, n_sub=6)
    w_box = box_sq.weight(X, Y, Z, tol=1e-12)
    assert np.allclose(fill_in, w_box, atol=1e-12)
    assert np.allclose(fill_in, 1.0, atol=1e-12)

    # Exterior points give zero fill
    X_out = np.array([2.0e-6, -2.0e-6])
    Y_out = np.array([0.0, 0.0])
    Z_out = np.array([0.0, 0.0])
    h_out = np.full(2, 0.1e-6)
    assert np.allclose(
        poly_sq.fill(X_out, Y_out, Z_out, h_out, h_out, h_out, n_sub=6), 0.0
    )

    # 2. Triangle integral: area = 0.5 * base * height = 0.5 * 2.0 * 1.5 = 1.5 um^2
    base, height, thickness = 2.0e-6, 1.5e-6, 0.22e-6
    verts_tri = np.array([[0.0, 0.0], [base, 0.0], [0.0, height]])
    poly_tri = PolySlab(verts_tri, 0.0, thickness, material=12.0)

    nx, ny, nz = 40, 30, 4
    xg = np.linspace(-0.2e-6, base + 0.2e-6, nx + 1)
    yg = np.linspace(-0.2e-6, height + 0.2e-6, ny + 1)
    zg = np.linspace(-0.05e-6, thickness + 0.05e-6, nz + 1)
    xc = 0.5 * (xg[:-1] + xg[1:])
    yc = 0.5 * (yg[:-1] + yg[1:])
    zc = 0.5 * (zg[:-1] + zg[1:])
    dx, dy, dz = xg[1] - xg[0], yg[1] - yg[0], zg[1] - zg[0]

    Xg, Yg, Zg = np.meshgrid(xc, yc, zc, indexing="ij")
    f_tri = poly_tri.fill(
        Xg,
        Yg,
        Zg,
        np.full_like(Xg, dx),
        np.full_like(Yg, dy),
        np.full_like(Zg, dz),
        n_sub=8,
    )
    vol_num = float(np.sum(f_tri) * dx * dy * dz)
    vol_exact = 0.5 * base * height * thickness
    rel_err = abs(vol_num - vol_exact) / vol_exact
    assert rel_err < 1e-3, f"Triangle volume relative error {rel_err:.4e} >= 1e-3"

    # 3. Hole subtraction: square with a central square hole
    hole = np.array(
        [[-0.4e-6, -0.4e-6], [0.4e-6, -0.4e-6], [0.4e-6, 0.4e-6], [-0.4e-6, 0.4e-6]]
    )
    poly_hole = PolySlab(verts_sq, 0.0, thickness, material=12.0, holes=[hole])
    f_no_hole = poly_sq.fill(
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.2e-6]),
        np.array([0.2e-6]),
        np.array([0.1e-6]),
        n_sub=6,
    )
    f_in_hole = poly_hole.fill(
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.5 * thickness]),
        np.array([0.2e-6]),
        np.array([0.2e-6]),
        np.array([0.1e-6]),
        n_sub=6,
    )
    assert float(f_no_hole[0]) == pytest.approx(1.0)
    assert float(f_in_hole[0]) == pytest.approx(0.0)

    # Also verify ring helper produces outer and inner hole with expected area
    outer_r, holes_r = ring((0.0, 0.0), radius=1.0e-6, width=0.4e-6, n=128)
    poly_ring = PolySlab(outer_r, 0.0, thickness, material=12.0, holes=holes_r)
    assert float(
        poly_ring.fill(
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.5 * thickness]),
            np.array([0.1e-6]),
            np.array([0.1e-6]),
            np.array([0.1e-6]),
        )[0]
    ) == pytest.approx(0.0)
    assert float(
        poly_ring.fill(
            np.array([1.0e-6]),
            np.array([0.0]),
            np.array([0.5 * thickness]),
            np.array([0.05e-6]),
            np.array([0.05e-6]),
            np.array([0.1e-6]),
        )[0]
    ) == pytest.approx(1.0)


def test_polyslab_normals():
    """
    PolySlab.normal returns the unit outward normal of the nearest sidewall
    for points near each side of a rotated square, and +z near the top face.
    """
    # Square of half-width L = 1 um rotated by theta = 35 deg
    theta = np.deg2rad(35.0)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    L = 1.0e-6
    base_sq = np.array([[-L, -L], [L, -L], [L, L], [-L, L]])
    verts = base_sq @ rot.T
    poly = PolySlab(verts, -0.5e-6, 0.5e-6, material=12.0)

    # Expected outward normals for the 4 sides in unrotated frame:
    # bottom (0, -1), right (1, 0), top (0, 1), left (-1, 0)
    local_normals = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    expected_xy = local_normals @ rot.T
    sample_xy = (L - 0.02e-6) * expected_xy

    n_side = poly.normal(sample_xy[:, 0], sample_xy[:, 1], np.zeros(4))
    assert n_side.shape == (4, 3)
    assert np.allclose(n_side[:, :2], expected_xy, atol=1e-6)
    assert np.allclose(n_side[:, 2], 0.0, atol=1e-6)

    # Point near the top face (z = 0.49 um) and bottom face (z = -0.49 um)
    n_z = poly.normal(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.49e-6, -0.49e-6]),
    )
    assert np.allclose(n_z[:, 0], [0.0, 0.0], atol=1e-6)
    assert np.allclose(n_z[:, 1], [0.0, 0.0], atol=1e-6)
    assert np.allclose(n_z[:, 2], [1.0, -1.0], atol=1e-6)


def test_euler_bend_curvature_and_angle():
    """
    euler_bend produces a clothoid with curvature linear in arc length on each
    half (R^2 > 0.999), ending at the requested angle and reaching radius_min
    at the midpoint.
    """
    r_min = 5.0e-6
    angle_deg = 90.0
    width = 0.5e-6
    poly, cl = euler_bend(r_min, angle_deg, width, p=1.0, n=401)
    assert poly.shape == (802, 2)
    assert cl.shape == (401, 2)

    # Compute tangent angle and curvature along the centreline from coordinates
    dxy = np.diff(cl, axis=0)
    ds = np.hypot(dxy[:, 0], dxy[:, 1])
    s_mid = np.cumsum(ds) - 0.5 * ds
    theta = np.unwrap(np.arctan2(dxy[:, 1], dxy[:, 0]))

    # End tangent angle matches requested angle_deg
    total_angle_deg = float(np.rad2deg(theta[-1] - theta[0]))
    assert total_angle_deg == pytest.approx(angle_deg, rel=1e-2)

    # Curvature kappa(s) = dtheta / ds
    kappa = np.gradient(theta, s_mid)
    half = len(s_mid) // 2
    s_first = s_mid[2 : half - 2]
    k_first = kappa[2 : half - 2]
    slope, intercept = np.polyfit(s_first, k_first, 1)
    k_fit = slope * s_first + intercept
    ss_res = float(np.sum((k_first - k_fit) ** 2))
    ss_tot = float(np.sum((k_first - np.mean(k_first)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot
    assert r_squared > 0.999, f"Curvature linearity R^2 = {r_squared:.6f} <= 0.999"

    # Minimum radius of curvature at the midpoint matches r_min
    r_measured = 1.0 / float(np.max(kappa))
    assert r_measured == pytest.approx(r_min, rel=5e-3)


def test_kottke_vs_staircase_slab_modes():
    """
    For a 220 nm Si slab in SiO2 positioned off-grid by 0.37 cell, Kottke
    subpixel averaging achieves strictly lower TE and TM effective-index error
    than staircasing at 3 resolutions and exhibits convergence order > 1.5.
    """
    lam0 = 1.55e-6
    f0 = c_light / lam0
    n_core = PHOTONIC_MATERIALS["Si"]
    n_clad = PHOTONIC_MATERIALS["SiO2"]
    eps_core = n_to_eps(n_core)
    eps_clad = n_to_eps(n_clad)
    thickness = 0.22e-6
    domain_y = 2.4e-6

    neff_te_exact = float(
        slab_waveguide_neff(n_core, n_clad, thickness, lam0, polarization="TE")
    )
    neff_tm_exact = float(
        slab_waveguide_neff(n_core, n_clad, thickness, lam0, polarization="TM")
    )

    # Uniform cell sizes in y (in nm: 40 nm, 25 nm, 16 nm)
    dys = [40.0e-9, 25.0e-9, 16.0e-9]
    err_kottke_te, err_stair_te = [], []
    err_kottke_tm, err_stair_tm = [], []

    for dy in dys:
        # Place the slab center off-grid by 0.37 * dy so both interfaces
        # lie off-grid inside cells
        y_center = 0.37 * dy
        # Background SiO2 box + Si PolySlab (so faces(y) is empty -> uniform y grid!)
        bg = Box((0.0, 0.0, 0.0), (20e-6, domain_y, 20e-6), material=eps_clad)
        verts_slab = straight((-10e-6, y_center), (10e-6, y_center), width=thickness)
        core = PolySlab(verts_slab, -10e-6, 10e-6, material=eps_core)

        # Setting cpw so background_cell / n_core == dy gives uniform cell size dy
        cpw = lam0 / (n_core * dy)
        for avg, te_list, tm_list in [
            ("kottke", err_kottke_te, err_kottke_tm),
            ("staircase", err_stair_te, err_stair_tm),
        ]:
            sim = Simulation(
                f_min=0.95 * f0,
                f_max=f0,
                structures=[bg, core],
                boundaries=(
                    ["periodic", "pec", "periodic"],
                    ["periodic", "pec", "periodic"],
                ),
                domain=(
                    -2 * dy,
                    2 * dy,
                    -0.5 * domain_y,
                    0.5 * domain_y,
                    -0.5 * dy,
                    0.5 * dy,
                ),
                cells_per_wavelength=cpw,
                averaging=avg,
            )
            solver = sim.build()
            modes = solve_modes(
                solver,
                f0,
                center=(0.0, 0.0, 0.0),
                size=(0.0, np.inf, np.inf),
                n_modes=2,
            )
            # modes[0] is TE (Ez polarized, higher neff), modes[1] is TM (Ey polarized)
            neff_te_num = float(np.real(modes[0].neff))
            neff_tm_num = float(np.real(modes[1].neff))
            te_list.append(abs(neff_te_num - neff_te_exact))
            tm_list.append(abs(neff_tm_num - neff_tm_exact))

    for i in range(len(dys)):
        assert err_kottke_te[i] < err_stair_te[i], (
            f"TE resolution {i}: Kottke {err_kottke_te[i]:.3e} >= Staircase {err_stair_te[i]:.3e}"
        )
        assert err_kottke_tm[i] < err_stair_tm[i], (
            f"TM resolution {i}: Kottke {err_kottke_tm[i]:.3e} >= Staircase {err_stair_tm[i]:.3e}"
        )

    log_h = np.log(dys)
    order_te = float(np.polyfit(log_h, np.log(err_kottke_te), 1)[0])
    order_tm = float(np.polyfit(log_h, np.log(err_kottke_tm), 1)[0])
    assert order_te > 1.5, f"Kottke TE convergence order {order_te:.2f} <= 1.5"
    assert order_tm > 1.5, f"Kottke TM convergence order {order_tm:.2f} <= 1.5"


def test_straight_waveguide_2d_eim():
    """
    A 2D EIM straight silicon strip waveguide gives |S21|^2 > 0.98 and
    |S11|^2 < 1e-3 across the 1.5-1.6 um band.
    """
    w_wg = 0.5e-6
    length = 3.0e-6
    wg = PolySlab(
        straight((-0.5 * length, 0.0), (0.5 * length, 0.0), width=w_wg),
        z_min=0.0,
        z_max=0.22e-6,
        material=n_to_eps(PHOTONIC_MATERIALS["Si"]),
    )
    ports = [
        Port(
            "in",
            center=(-1.0e-6, 0.0, 0.11e-6),
            size=(0.0, 2.0e-6, 0.8e-6),
            direction=+1,
        ),
        Port(
            "out",
            center=(1.0e-6, 0.0, 0.11e-6),
            size=(0.0, 2.0e-6, 0.8e-6),
            direction=+1,
        ),
    ]
    circuit = PhotonicCircuit(
        wavelength_min=1.50e-6,
        wavelength_max=1.60e-6,
        n_freqs=21,
        structures=[wg],
        ports=ports,
        two_d=True,
        cells_per_wavelength=14,
        domain=(-1.3e-6, 1.3e-6, -1.2e-6, 1.2e-6),
        n_pml=8,
        record_movie=True,
        movie_frames=10,
    )
    s_mat = circuit.run_s_parameters(
        drive=["in"], energy_decay=1e-4, collect_fields=True
    )
    t_power = np.abs(s_mat[("out", "in")]) ** 2
    r_power = np.abs(s_mat[("in", "in")]) ** 2
    # Check interior band points (avoiding Gaussian pulse edges)
    mid = slice(3, -3)
    assert float(np.min(t_power[mid])) > 0.98, (
        f"Min straight waveguide transmission {np.min(t_power[mid]):.4f} <= 0.98"
    )
    assert float(np.max(r_power[mid])) < 1e-3, (
        f"Max straight waveguide reflection {np.max(r_power[mid]):.4e} >= 1e-3"
    )
    assert 1550e-9 in circuit.dft_fields
    assert circuit.dft_fields[1550e-9]["E2"].max() > 0
    assert circuit.movie_data is not None and len(circuit.movie_data["frames"]) > 0
    assert circuit.port_time_trace is not None and circuit.port_time_trace["t"].size > 0
    assert circuit.flux_balance is not None
    np.testing.assert_allclose(circuit.flux_balance["total_box"][mid], 1.0, atol=0.03)


def test_y_junction_2d_eim_symmetry_and_power():
    """
    A symmetric 2D EIM Y-junction splits power identically between its two
    output arms (|S21|^2 == |S31|^2 within 1e-6) and conserves energy
    (|S11|^2 + |S21|^2 + |S31|^2 <= 1 + 1e-3).
    """
    widths = (
        np.array([0.5, 0.5, 0.6, 0.7, 0.9, 1.26, 1.4, 1.4, 1.4, 1.4, 1.31, 1.2, 1.2])
        * 1e-6
    )
    eps_si = n_to_eps(PHOTONIC_MATERIALS["Si"])
    in_wg = PolySlab(straight((-1.2e-6, 0.0), (0.0, 0.0), 0.5e-6), 0.0, 0.22e-6, eps_si)
    junc = PolySlab(
        spline_taper(2.0e-6, widths, x0=0.0, y0=0.0, n=100), 0.0, 0.22e-6, eps_si
    )
    arm_top = PolySlab(
        sbend((1.95e-6, 0.35e-6), dx=2.05e-6, dy=0.55e-6, width=0.5e-6, n=60),
        0.0,
        0.22e-6,
        eps_si,
    )
    arm_bot = PolySlab(
        sbend((1.95e-6, -0.35e-6), dx=2.05e-6, dy=-0.55e-6, width=0.5e-6, n=60),
        0.0,
        0.22e-6,
        eps_si,
    )
    out_top = PolySlab(
        straight((3.95e-6, 0.9e-6), (5.0e-6, 0.9e-6), 0.5e-6), 0.0, 0.22e-6, eps_si
    )
    out_bot = PolySlab(
        straight((3.95e-6, -0.9e-6), (5.0e-6, -0.9e-6), 0.5e-6), 0.0, 0.22e-6, eps_si
    )
    ports = [
        Port(
            "in",
            center=(-0.8e-6, 0.0, 0.11e-6),
            size=(0.0, 1.8e-6, 0.8e-6),
            direction=+1,
        ),
        Port(
            "out1",
            center=(4.5e-6, 0.9e-6, 0.11e-6),
            size=(0.0, 1.6e-6, 0.8e-6),
            direction=+1,
        ),
        Port(
            "out2",
            center=(4.5e-6, -0.9e-6, 0.11e-6),
            size=(0.0, 1.6e-6, 0.8e-6),
            direction=+1,
        ),
    ]
    circuit = PhotonicCircuit(
        wavelength_min=1.50e-6,
        wavelength_max=1.60e-6,
        n_freqs=21,
        structures=[in_wg, junc, arm_top, arm_bot, out_top, out_bot],
        ports=ports,
        two_d=True,
        cells_per_wavelength=12,
        domain=(-1.1e-6, 4.8e-6, -1.9e-6, 1.9e-6),
        n_pml=8,
    )
    s_mat = circuit.run_s_parameters(
        drive=["in"], energy_decay=1e-3, collect_fields=False
    )
    t1 = np.abs(s_mat[("out1", "in")]) ** 2
    t2 = np.abs(s_mat[("out2", "in")]) ** 2
    r = np.abs(s_mat[("in", "in")]) ** 2
    mid = slice(3, -3)

    sym_diff = float(np.max(np.abs(t1[mid] - t2[mid])))
    assert sym_diff < 1e-6, f"Symmetry violation |T1 - T2| = {sym_diff:.3e} >= 1e-6"

    total_p = t1[mid] + t2[mid] + r[mid]
    assert float(np.max(total_p)) <= 1.0 + 1e-3, (
        f"Passivity violation max(T1 + T2 + R) = {np.max(total_p):.5f} > 1.001"
    )
    assert float(np.mean(t1[mid])) > 0.40, (
        f"Unexpectedly low arm transmission mean(T1) = {np.mean(t1[mid]):.3f}"
    )


def test_off_center_polyslab_mesh_and_kottke_anisotropy():
    """
    Off-center PolySlabs trigger core mesh refinement in Simulation._axis, and
    Kottke subpixel averaging produces distinct harmonic (normal) vs arithmetic
    (tangential) permittivities on cut cells.
    """
    f0 = c_light / 1.55e-6
    eps_si = n_to_eps(PHOTONIC_MATERIALS["Si"])
    eps_sio2 = n_to_eps(PHOTONIC_MATERIALS["SiO2"])
    bg = Box((0.0, 0.0, 0.0), (4.0e-6, 6.0e-6, 2.0e-6), material=eps_sio2)
    # Off-center waveguide at y = +1.511 um (does not cross domain midpoint y = 0)
    wg = PolySlab(
        straight((-1.5e-6, 1.511e-6), (1.5e-6, 1.511e-6), width=0.5e-6),
        z_min=-0.5e-6,
        z_max=0.5e-6,
        material=eps_si,
    )
    kw = dict(
        f_min=0.95 * f0,
        f_max=f0,
        structures=[bg, wg],
        boundaries=(["periodic", "pec", "periodic"], ["periodic", "pec", "periodic"]),
        domain=(-1.0e-6, 1.0e-6, -2.5e-6, 2.5e-6, -0.1e-6, 0.1e-6),
        cells_per_wavelength=16,
    )
    sim_k = Simulation(**kw, averaging="kottke")
    sim_a = Simulation(**kw, averaging="arithmetic")
    solver_k = sim_k.build()
    solver_a = sim_a.build()

    dy = np.diff(sim_k.grid.y)
    expected_core_dy = 1.55e-6 / (PHOTONIC_MATERIALS["Si"] * 16)
    # Without interval-overlap check in Simulation._axis, dy would be ~67 nm (cladding)
    # instead of ~27.8 nm (Si core) when the PolySlab does not cross y = 0.
    assert float(np.max(dy)) <= expected_core_dy * 1.02

    # On the y-normal sidewall (n = (0, +-1, 0)):
    # - normal component Ey uses harmonic mean in Kottke (eps_k < eps_a => ieps_k > ieps_a)
    # - tangential component Ex uses arithmetic mean in both (ieps_k == ieps_a)
    iz = solver_k.Nz // 2
    ix = solver_k.Nx // 2
    ieps_yk = np.asarray(solver_k.ieps[ix, :, iz, "y"])
    ieps_ya = np.asarray(solver_a.ieps[ix, :, iz, "y"])
    ieps_xk = np.asarray(solver_k.ieps[ix, :, iz, "x"])
    ieps_xa = np.asarray(solver_a.ieps[ix, :, iz, "x"])
    cut_y = (1.0 / ieps_ya > eps_sio2 * 8.854e-12 * 1.05) & (
        1.0 / ieps_ya < eps_si * 8.854e-12 * 0.95
    )
    assert np.any(cut_y), "Expected cut Ey cells on off-grid sidewall"
    assert np.all(ieps_yk[cut_y] > ieps_ya[cut_y])
    assert np.allclose(ieps_xk, ieps_xa, rtol=1e-12)


def _circuit_straight(mesh_overrides=(), cpw=10):
    wg = PolySlab(
        straight((-1.5e-6, 0.0), (1.5e-6, 0.0), width=0.5e-6),
        z_min=0.0,
        z_max=0.22e-6,
        material=n_to_eps(PHOTONIC_MATERIALS["Si"]),
    )
    ports = [
        Port("in", center=(-1.0e-6, 0.0, 0.11e-6), size=(0.0, 2.0e-6, 0.8e-6)),
        Port("out", center=(1.0e-6, 0.0, 0.11e-6), size=(0.0, 2.0e-6, 0.8e-6)),
    ]
    return PhotonicCircuit(
        wavelength_min=1.50e-6,
        wavelength_max=1.60e-6,
        n_freqs=21,
        structures=[wg],
        ports=ports,
        cells_per_wavelength=cpw,
        domain=(-1.3e-6, 1.3e-6, -1.2e-6, 1.2e-6),
        n_pml=8,
        mesh_overrides=mesh_overrides,
    )


def test_mesh_override_refines_box_and_grades_out():
    """
    A mesh-override box caps the cell size inside it on x and y, puts nodes
    on its faces, and grades back to the background size at <= `grading`;
    without the box the same region is meshed coarsely.
    """
    h = 15e-9
    box = ((-0.2e-6, 0.25e-6, -np.inf), (0.2e-6, 0.35e-6, np.inf), h)
    plain = _circuit_straight()
    fine = _circuit_straight(mesh_overrides=[box])
    for axis, (a, b) in ((0, (-0.2e-6, 0.2e-6)), (1, (0.25e-6, 0.35e-6))):
        nodes = (fine.grid.x, fine.grid.y)[axis]
        d = np.diff(nodes)
        mid = 0.5 * (nodes[1:] + nodes[:-1])
        inside = (mid > a) & (mid < b)
        assert np.max(d[inside]) <= h * (1 + 1e-9)
        assert np.min(np.abs(nodes - a)) < 1e-12 and np.min(np.abs(nodes - b)) < 1e-12
        assert np.max(d[1:] / d[:-1]) < 1.6 and np.max(d[:-1] / d[1:]) < 1.6
        d0 = np.diff((plain.grid.x, plain.grid.y)[axis])
        assert np.max(d0) > 2.5 * h  # the background mesh would not resolve it
        assert np.max(d) == pytest.approx(np.max(d0), rel=0.05)  # far field unchanged


def test_mesh_override_graded_straight_waveguide_low_reflection():
    """
    Grading through a refinement box over the waveguide core adds no
    significant reflection: |S11|^2 < 1e-3, |S21|^2 > 0.98 (as uniform).
    """
    box = ((-0.3e-6, -0.3e-6, -np.inf), (0.3e-6, 0.3e-6, np.inf), 12e-9)
    circuit = _circuit_straight(mesh_overrides=[box], cpw=12)
    s_mat = circuit.run_s_parameters(drive=["in"], energy_decay=1e-4)
    mid = slice(3, -3)
    t_power = np.abs(s_mat[("out", "in")]) ** 2
    r_power = np.abs(s_mat[("in", "in")]) ** 2
    assert float(np.min(t_power[mid])) > 0.98
    assert float(np.max(r_power[mid])) < 1e-3


def test_mesh_override_per_axis_cap():
    """A per-axis cap (hx, hy, hz) refines only the axes that ask for it."""
    box = ((-0.2e-6, 0.25e-6, -np.inf), (0.2e-6, 0.35e-6, np.inf), (1.0, 15e-9, 1.0))
    plain, fine = _circuit_straight(), _circuit_straight(mesh_overrides=[box])
    np.testing.assert_allclose(
        np.diff(fine.grid.x).max(), np.diff(plain.grid.x).max(), rtol=0.05
    )
    mid = 0.5 * (fine.grid.y[1:] + fine.grid.y[:-1])
    assert np.diff(fine.grid.y)[(mid > 0.25e-6) & (mid < 0.35e-6)].max() <= 15e-9 * (
        1 + 1e-9
    )


def test_deprecated_pic_aliases_and_mutation():
    """
    Deprecated `PICSimulation` and `PIC_MATERIALS` still work and emit
    `DeprecationWarning`; `PhotonicCircuit` and `PHOTONIC_MATERIALS` emit no
    warning.
    """
    wg = PolySlab(
        straight((-1.5e-6, 0.0), (1.5e-6, 0.0), width=0.5e-6),
        z_min=0.0,
        z_max=0.22e-6,
        material=n_to_eps(PHOTONIC_MATERIALS["Si"]),
    )
    ports = [
        Port("in", center=(-1.0e-6, 0.0, 0.11e-6), size=(0.0, 2.0e-6, 0.8e-6)),
        Port("out", center=(1.0e-6, 0.0, 0.11e-6), size=(0.0, 2.0e-6, 0.8e-6)),
    ]
    kw = dict(
        wavelength_min=1.50e-6,
        wavelength_max=1.60e-6,
        n_freqs=11,
        structures=[wg],
        ports=ports,
        cells_per_wavelength=10,
        domain=(-1.3e-6, 1.3e-6, -1.2e-6, 1.2e-6),
        n_pml=6,
    )

    with warnings.catch_warnings(record=True) as rec_new:
        warnings.simplefilter("always", DeprecationWarning)
        circuit_new = PhotonicCircuit(**kw)
        _ = PHOTONIC_MATERIALS["Si"]
    dep_new = [w for w in rec_new if issubclass(w.category, DeprecationWarning)]
    assert len(dep_new) == 0

    with pytest.deprecated_call(match="PICSimulation is deprecated"):
        circuit_old = PICSimulation(**kw)
    assert isinstance(circuit_old, PhotonicCircuit)
    assert circuit_old.grid.Nx == circuit_new.grid.Nx

    with pytest.deprecated_call(match="PICSimulation is deprecated"):
        circuit_sg = PICSimulation(**kw, backend="subgrid")
    assert isinstance(circuit_sg, SubgridPhotonicCircuit)

    with pytest.deprecated_call(match="PIC_MATERIALS is deprecated"):
        mats_phot = photonics_mod.PIC_MATERIALS
    assert mats_phot == PHOTONIC_MATERIALS

    with pytest.deprecated_call(match="PIC_MATERIALS is deprecated"):
        mats_tmpl = templates_mod.PIC_MATERIALS
    assert mats_tmpl == PHOTONIC_MATERIALS
