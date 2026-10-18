"""
Linear plasma wakefield of a rigid, ultra-relativistic Gaussian bunch in a
perfectly conducting circular pipe filled with a cold collisionless plasma
(Drude medium, `wakis.plasma.cold_plasma`), compared with linear cold-fluid
theory (`wakis.plasma.linear_wake`, physics guide "Linear plasma
wakefield").

Setup shared by all runs: plasma wavelength lambda_p = 1 mm, pipe radius
a = 2 / k_p, a vacuum entrance of one lambda_p (the beam starts at the low
PEC wall, where an ultra-relativistic bunch in vacuum does not radiate),
then uniform plasma. The bunch is the line current of `wakis.sources.Beam`
on a grid node. E_z is read along z at the end of the run (one snapshot),
when the bunch centre is one lambda_p before the exit wall; the co-moving
coordinate is xi = z - z_bunch(t). The comparison window is
-(2 or 3) lambda_p < xi < -3 sigma_z behind the bunch, far (>= 15 lambda_p) from the
plasma entrance, whose transition radiation trails the bunch with group
velocity < c and decays with the distance from the entrance.

Error sources (observed on cem-gpu, see the physics guide):
* near the line source the lattice Green function of the transverse
  operator differs from K_0 by O(h^2 / r^2): compare at r ~ a / 2;
* the staircased wall shifts the effective radius by O(h): first-order
  error; conformal PEC removes it;
* time/space dispersion of the Drude ADE: O((k_p h)^2), below 1e-3 in k_p.
"""

import numpy as np
import pytest
import pyvista as pv
from scipy.constants import c, epsilon_0, mu_0

import wakis
from wakis.plasma import (
    cold_plasma,
    linear_wake,
    pipe_green_function,
    plasma_frequency,
    wake_amplitude,
)
from wakis.sources import Beam

LAM_P = 1e-3  # plasma wavelength [m]
K_P = 2 * np.pi / LAM_P
RADIUS = 2.0 / K_P  # pipe radius, k_p a = 2
Q = 1e-9  # bunch charge [C]


def _plasma_pipe_wake(n_lp, kps, length, conformal, x0_cells=0, use_gpu=False):
    """
    Run a bunch through the plasma-filled pipe.

    Parameters
    ----------
    n_lp : int
        Cells per plasma wavelength (cubic cells).
    kps : float
        k_p sigma_z.
    length : float
        Pipe length in plasma wavelengths (the first one is vacuum).
    conformal : False or 'pec'
        Staircased or conformal pipe wall.
    x0_cells : int
        Horizontal bunch offset [cells].

    Returns
    -------
    dict with the solver, the bunch and the co-moving coordinates of the
    E_z (z midpoints) and E_x (z nodes) samples.
    """
    h = LAM_P / n_lp
    n_half = int(np.ceil(RADIUS / h)) + 2
    L = length * LAM_P
    pipe = pv.Cylinder(
        center=(0, 0, L / 2), direction=(0, 0, 1), radius=RADIUS, height=3 * L,
        resolution=720,
    ).triangulate()  # fmt: skip
    grid = wakis.GridFIT3D(
        -n_half * h, n_half * h, -n_half * h, n_half * h, 0, L,
        2 * n_half, 2 * n_half, int(round(length * n_lp)),
        stl_solids={"pipe": pipe}, stl_materials={"pipe": "vacuum"},
        conformal=conformal, verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, bg="pec", use_stl=True,
        use_gpu=use_gpu, verbose=0,
    )  # fmt: skip
    # Plasma from z = lambda_p on; the box face lies between E samples, so
    # every edge is either vacuum or plasma
    z_start = LAM_P - h / 4
    solver.add_dispersive_material(
        cold_plasma(K_P * c), center=(0, 0, z_start + 2 * L), size=(4 * L,) * 3
    )
    sigmaz = kps / K_P
    beam = Beam(q=Q, sigmaz=sigmaz, xsource=x0_cells * h, ysource=0.0)
    t_end = (L - LAM_P - solver.z.min()) / c + beam.ti
    Nt = int(t_end / solver.dt)
    solver.emsolve(Nt, source=[beam])
    # bunch centre at the time of the last E update (see Beam.update)
    zc = solver.z.min() + c * (Nt * solver.dt - beam.ti)
    return dict(
        solver=solver,
        h=h,
        sigmaz=sigmaz,
        x0=x0_cells * h,
        xi_mid=solver.z - zc,
        xi_node=grid.z[:-1] - zc,
        xi_h=solver.z - (zc - c * solver.dt / 2),  # H is dt / 2 older
    )


def _host(v):
    return np.asarray(v.get() if hasattr(v, "get") else v)


def _fit(xi, y):
    """Amplitude, wavenumber/k_p and phase of y ~ A cos(k xi + phi)."""
    from scipy.optimize import least_squares

    basis = np.column_stack([np.cos(K_P * xi), np.sin(K_P * xi)])
    cc, ss = np.linalg.lstsq(basis, y, rcond=None)[0]
    p0 = [np.hypot(cc, ss), 1.0, np.arctan2(-ss, cc)]
    return least_squares(lambda p: p[0] * np.cos(p[1] * K_P * xi + p[2]) - y, p0).x


def _compare_ez(run, r_frac, window=3.0):
    """
    Errors of E_z on the x axis at r ~ r_frac * a against theory over
    -window lambda_p < xi < -3 sigma_z: relative RMS deviation, and relative
    errors of the fitted amplitude and wavenumber of A cos(k xi + phi).
    """
    solver, h = run["solver"], run["h"]
    grid = solver.grid
    i0 = int(np.argmin(abs(grid.x - run["x0"])))
    j0 = int(np.argmin(abs(grid.y)))
    i = i0 + int(round(r_frac * RADIUS / h))
    xi = run["xi_mid"]
    window = (xi < -3 * run["sigmaz"]) & (xi > -window * LAM_P)
    ez = _host(solver.E[i, j0, :, "z"])[window]
    th, _, _ = linear_wake(
        grid.x[i], 0.0, xi[window], K_P, run["sigmaz"], Q, RADIUS, x0=run["x0"]
    )
    fs, ft = _fit(xi[window], ez), _fit(xi[window], th)
    rms = np.sqrt(np.mean((ez - th) ** 2)) / np.max(abs(th))
    return dict(rms=rms, amp=fs[0] / ft[0] - 1, k=fs[1] / ft[1] - 1)


# ----------------------------------------------------------------- theory
def test_cold_plasma_model():
    """
    cold_plasma is the Drude medium eps = 1 - w_p^2 / w^2 (collisionless),
    w_p = 5.64e4 sqrt(n [cm^-3]) rad/s, and the discrete permittivity of the
    ADE vanishes at the discrete plasma frequency 2 asin(w_p dt / 2) / dt.
    """
    wp = plasma_frequency(1e24)  # 1e18 cm^-3
    assert wp == pytest.approx(5.641e4 * np.sqrt(1e18), rel=1e-3)
    plasma = cold_plasma(wp)
    f = np.array([0.5, 2.0]) * wp / (2 * np.pi)
    assert np.allclose(plasma.epsilon(f), 1 - (wp / (2 * np.pi * f)) ** 2)
    dt = 0.1 / wp
    f_d = 2 * np.arcsin(wp * dt / 2) / dt / (2 * np.pi)
    assert abs(plasma.epsilon_discrete(f_d, dt)) < 1e-12


def test_linear_wake_theory():
    """
    The closed-form wake satisfies its defining relations: g = 0 on the
    wall, reciprocity g(r; r0) = g(r0; r), (lap - k_p^2) g = 0 away from the
    source, Panofsky-Wenzel dF_x/dxi = dE_z/dx, a field-free region ahead of
    the bunch, the convolution integral, and the amplitude
    q k_p^2 g exp(-k_p^2 sigma^2 / 2) / (2 pi eps_0) far behind the bunch.
    """
    a, kp, s = RADIUS, K_P, 1.0 / K_P
    src = dict(k_p=kp, radius=a, x0=0.3 * a, y0=-0.2 * a)
    phi = np.linspace(0, 2 * np.pi, 7)
    assert (
        np.max(abs(pipe_green_function(a * np.cos(phi), a * np.sin(phi), **src)))
        < 1e-12
    )
    g1 = pipe_green_function(-0.4 * a, 0.5 * a, kp, a, x0=0.1 * a, y0=0.2 * a)
    g2 = pipe_green_function(0.1 * a, 0.2 * a, kp, a, x0=-0.4 * a, y0=0.5 * a)
    assert g1 == pytest.approx(g2, rel=1e-12)
    d = 1e-3 * a
    x, y = -0.5 * a, 0.4 * a
    g = pipe_green_function(x, y, **src)
    lap = (
        pipe_green_function(x + d, y, **src) + pipe_green_function(x - d, y, **src)
        + pipe_green_function(x, y + d, **src) + pipe_green_function(x, y - d, **src)
        - 4 * g
    ) / d**2  # fmt: skip
    assert abs(lap - kp**2 * g) < 1e-5 * kp**2 * abs(g)

    xi = np.linspace(-6, 0.5, 2601) * LAM_P
    kw = dict(k_p=kp, sigmaz=s, q=Q, radius=a, x0=0.3 * a, y0=-0.2 * a)
    Ez, Fx, _ = linear_wake(x, y, xi, **kw)
    dEz = (linear_wake(x + d, y, xi, **kw)[0] - linear_wake(x - d, y, xi, **kw)[0]) / (
        2 * d
    )
    dFx = np.gradient(Fx, xi)
    inner = slice(5, -5)
    assert np.max(abs(dFx - dEz)[inner]) < 1e-4 * np.max(abs(dEz))
    ahead = linear_wake(x, y, [8 * s], **kw)[0]
    assert abs(ahead[0]) < 1e-12 * np.max(abs(Ez))

    # convolution integral by quadrature at one point
    xi0 = -2.3 * LAM_P
    xp = np.linspace(xi0, xi0 + 20 * LAM_P, 400001)
    lam = Q * np.exp(-(xp**2) / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)
    conv = np.trapezoid(lam * np.cos(kp * (xi0 - xp)), xp)
    g = pipe_green_function(x, y, **src)
    ez0 = linear_wake(x, y, xi0, **kw)[0]
    assert ez0 == pytest.approx(-(kp**2) * g * conv / (2 * np.pi * epsilon_0), rel=1e-6)

    # far-behind amplitude, maximal at k_p sigma = sqrt(2) for fixed sigma
    r = 0.5 * a
    far = np.linspace(-12, -10, 801) * LAM_P
    ez_far = linear_wake(r, 0.0, far, kp, s, Q, a)[0]
    assert np.max(abs(ez_far)) == pytest.approx(
        wake_amplitude(r, kp, s, Q, a), rel=1e-4
    )
    # fixed sigma, density scan with k_p r and k_p a fixed: maximum at sqrt(2)
    kps = np.linspace(0.5, 3, 2501)
    amp = [wake_amplitude(r / k, k * kp, s, Q, a / k) for k in kps]
    assert kps[np.argmax(amp)] == pytest.approx(np.sqrt(2), abs=1e-3)


# ------------------------------------------------------------- simulation
def test_plasma_wake_longitudinal(use_gpu):
    """
    On-axis bunch, k_p sigma_z = 1, 20 cells per lambda_p (h = 0.05 mm,
    a = 6.4 cells, sigma_z = 3.2 cells), 20 lambda_p long pipe (window 15-18
    lambda_p from the plasma entrance). With the conformal wall E_z at
    r = a / 2 matches theory: amplitude 1.4 % (observed), wavenumber 3e-4,
    RMS deviation over the window 1.0 % of the amplitude. Near the wall
    (r = 3a/4) the staircased pipe overestimates the amplitude by 23 %
    (effective radius error ~ h / 2), the conformal one by 0.6 %.
    """
    run = _plasma_pipe_wake(20, 1.0, 20, "pec", use_gpu=use_gpu)
    err = _compare_ez(run, 0.5)
    assert abs(err["amp"]) < 0.025
    assert abs(err["k"]) < 1e-3
    assert err["rms"] < 0.02
    conformal = abs(_compare_ez(run, 0.75)["amp"])
    del run

    stair = _plasma_pipe_wake(20, 1.0, 20, False, use_gpu=use_gpu)
    staircase = abs(_compare_ez(stair, 0.75)["amp"])
    assert conformal < 0.015
    assert staircase > 0.1
    assert conformal < staircase / 5


@pytest.mark.slow
@pytest.mark.parametrize("kps", [0.5, 1.0, 2.0])
def test_plasma_wake_bunch_length(kps, use_gpu):
    """
    k_p sigma_z = 0.5, 1, 2 at 48 cells per lambda_p (a = 15.3 cells,
    sigma_z = 3.8-15 cells), conformal wall, 24 lambda_p pipe, window
    2 lambda_p behind the bunch (for k_p sigma_z = 0.5 the entrance transient
    reaches xi ~ -2 lambda_p after 22 lambda_p). E_z at r ~ a/4, a/2, 3a/4:
    observed amplitude errors 0.1-0.7 %, wavenumber 4e-5-1e-4, RMS
    deviation 0.1-0.6 %.
    """
    run = _plasma_pipe_wake(48, kps, 24, "pec", use_gpu=use_gpu)
    for r in (0.25, 0.5, 0.75):
        err = _compare_ez(run, r, window=2.0)
        assert abs(err["amp"]) < 0.01, (r, err)
        assert abs(err["k"]) < 5e-4, (r, err)
        assert err["rms"] < 0.01, (r, err)


@pytest.mark.slow
def test_plasma_wake_convergence(use_gpu):
    """
    k_p sigma_z = 1 at 24 and 48 cells per lambda_p. Near the wall
    (r = 0.79 a resp. 0.72 a, nearest node to 3a/4) the staircased pipe
    overestimates the amplitude by 16.8 % resp. 5.7 % (first order in h);
    the conformal pipe by 0.26 % resp. 0.11 %. At r ~ a/2 the conformal
    error converges at second order (0.73 % -> 0.15 %).
    """
    errs = {}
    for n in (24, 48):
        for conformal in (False, "pec"):
            run = _plasma_pipe_wake(n, 1.0, 24, conformal, use_gpu=use_gpu)
            errs[n, conformal] = abs(_compare_ez(run, 0.75)["amp"])
            if conformal:
                errs[n, "half"] = abs(_compare_ez(run, 0.5)["amp"])
            del run
    assert errs[24, False] > 0.1
    assert errs[48, False] < 0.5 * errs[24, False]
    assert errs[24, "pec"] < 0.005 and errs[48, "pec"] < 0.005
    assert errs[48, "half"] < errs[24, "half"] / 3


@pytest.mark.slow
def test_plasma_wake_transverse(use_gpu):
    """
    Bunch offset by lambda_p / 8 (0.39 a), 48 cells per lambda_p, conformal
    wall: the transverse force F_x = E_x - c B_y at 6.5 cells (0.14 lambda_p)
    on both sides of the bunch (one point next to the axis, one towards the
    wall) matches the Green-function theory including the wall images
    (observed amplitude errors 0.8 %; 3 % at 3.5 cells and 1.6 % at
    4.5 cells of 64 per lambda_p, i.e. second order at a fixed distance).
    E_z at 0.5 a from the bunch (0.92 a from the axis) is within 0.2 %.
    """
    n = 48
    run = _plasma_pipe_wake(n, 1.0, 24, "pec", x0_cells=n // 8, use_gpu=use_gpu)
    solver = run["solver"]
    i0 = int(np.argmin(abs(solver.grid.x - run["x0"])))
    j0 = int(np.argmin(abs(solver.grid.y)))
    xi = run["xi_node"]
    window = (xi < -3 * run["sigmaz"]) & (xi > -3 * LAM_P)
    for di in (-n // 8, n // 8):
        i = i0 + di if di > 0 else i0 + di - 1  # E_x lives on x midpoints
        ex = _host(solver.E[i, j0, :, "x"])
        hy = _host(solver.H[i, j0, :, "y"])
        fx = ex - c * mu_0 * np.interp(xi, run["xi_h"], hy)
        _, th, _ = linear_wake(
            solver.x[i], 0.0, xi, K_P, run["sigmaz"], Q, RADIUS, x0=run["x0"]
        )
        fs, ft = _fit(xi[window], fx[window]), _fit(xi[window], th[window])
        assert fs[0] / ft[0] == pytest.approx(1, abs=0.02)
        assert fs[1] / ft[1] == pytest.approx(1, abs=5e-4)
    assert abs(_compare_ez(run, 0.5)["amp"]) < 0.005
