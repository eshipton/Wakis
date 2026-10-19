"""
High-order schemes (`wakis.highorder`): nodal DGTD on curved triangles and
the Cartesian (2,4) scheme. Every test states the failure it would catch;
mutation checks show that the asserted property can fail.
"""

import numpy as np
import pytest
from scipy.constants import c as C0
from scipy.special import jn_zeros, jv

from wakis.highorder import DGTD2D, LSRK4, circles_mesh, lsrk_stability
from wakis.highorder.nodal2d import ReferenceTriangle, collapsed_cubature


def uniform(h):
    return lambda P: np.full(len(P), h)


# ---------------------------------------------------------------- reference
@pytest.mark.parametrize("p", [1, 3, 5])
def test_reference_operators_exact(p):
    T = ReferenceTriangle(p)
    f = T.r**p + T.r * T.s ** (p - 1) + 1
    dfr = p * T.r ** (p - 1) + T.s ** (p - 1)
    assert np.abs(T.Dr @ f - dfr).max() < 1e-11
    r, s, w = collapsed_cubature(p + 2)
    L = T.interp_matrix(r, s)
    assert np.abs(L.T @ (w[:, None] * L) - T.M).max() < 1e-12
    # face nodes are the Gauss-Lobatto points (warp-and-blend property)
    from wakis.highorder.nodal2d import gll_nodes

    assert np.allclose(T.tf, gll_nodes(p))


def test_lsrk4_is_fourth_order():
    z = 1j * np.array([0.2, 0.1])
    err = np.abs(lsrk_stability(z) - np.exp(z))
    order = np.log2(err[0] / err[1])
    assert 4.6 < order < 5.4  # local error O(dt^5)
    # mutation: perturb one coefficient -> order collapses
    bad = {k: v.copy() for k, v in LSRK4.items()}
    bad["b"][2] *= 1.001
    err = np.abs(lsrk_stability(z, bad) - np.exp(z))
    assert np.log2(err[0] / err[1]) < 2.5


# ---------------------------------------------------------------- DGTD
def _square_cavity_error(p, h, alpha=1.0):
    L = 1e-6
    m = circles_mesh([], uniform(h), box=(0, L, 0, L))
    s = DGTD2D(m, p, polarization="TM", bc="pec", alpha=alpha)
    w = C0 * np.pi * np.sqrt(2) / L
    s.u[:] = np.sin(np.pi * s.x / L) * np.sin(np.pi * s.y / L)
    n = int(np.ceil(2 * np.pi / w / s.dt))
    s.dt = 2 * np.pi / w / n
    e0 = s.energy()
    s.run(n)
    ex = np.sin(np.pi * s.x / L) * np.sin(np.pi * s.y / L) * np.cos(w * s.t)
    return np.abs(s.u - ex).max(), s.energy() / e0


@pytest.mark.parametrize("p", [2, 3])
def test_dg_convergence_order(p):
    e1, _ = _square_cavity_error(p, 0.4e-6)
    e2, _ = _square_cavity_error(p, 0.2e-6)
    assert np.log2(e1 / e2) > p + 0.0  # optimal is p + 1


def test_dg_central_flux_conserves_energy():
    _, r_central = _square_cavity_error(3, 0.4e-6, alpha=0.0)
    _, r_upwind = _square_cavity_error(1, 0.4e-6, alpha=1.0)
    assert abs(r_central - 1) < 3e-5  # only the RK4 dissipation O(dt^5)
    assert r_upwind < 1 - 1e-3  # upwind dissipates under-resolved modes
    assert 1 - r_upwind > 30 * abs(1 - r_central)


def _circle_cavity_error(p, h, curved):
    R = 1e-6
    k = jn_zeros(0, 1)[0] / R
    m = circles_mesh([], uniform(h), outer_circle=(0, 0, R))
    s = DGTD2D(m, p, polarization="TM", bc="pec", curved=curved)
    r = np.hypot(s.x, s.y)
    s.u[:] = jv(0, k * r)
    w = C0 * k
    n = int(np.ceil(2 * np.pi / w / s.dt))
    s.dt = 2 * np.pi / w / n
    s.run(n)
    return np.abs(s.u - jv(0, k * r) * np.cos(w * s.t)).max()


def test_curved_elements_restore_high_order():
    """p = 4 on a circular PEC cavity: curved elements converge at high
    order; straight-sided elements stall at the O(h^2) geometric error."""
    c1 = _circle_cavity_error(4, 0.5e-6, True)
    c2 = _circle_cavity_error(4, 0.25e-6, True)
    s2 = _circle_cavity_error(4, 0.25e-6, False)
    assert np.log2(c1 / c2) > 3.5
    assert s2 > 100 * c2


def test_operator_matrix_and_adjoint_dot_product():
    """The probed sparse A equals the matrix-free operator, and the adjoint
    of one LSRK step (same scheme with A^T) passes the dot-product test.
    Mutation: a wrong transpose (A instead of A^T) fails."""
    m = circles_mesh([(0, 0, 0.4e-6)], uniform(0.3e-6), box=(-1.6e-6, 1.6e-6, -1.6e-6, 1.6e-6))
    s = DGTD2D(m, 2, {1: 4.0}, polarization="TE", bc="abc",
               pml=dict(thickness=0.4e-6), dt=1e-17)  # fmt: skip
    A = s.operator_matrix()
    rng = np.random.default_rng(1)
    x = rng.standard_normal(s.n_state)
    y = rng.standard_normal(s.n_state)
    ax = s.apply_operator(x)
    assert np.linalg.norm(A @ x - ax) < 1e-12 * np.linalg.norm(ax)

    def step(M, v):
        k = np.zeros_like(v)
        v = v.copy()
        for a, b in zip(LSRK4["a"], LSRK4["b"]):
            k = a * k + s.dt * (M @ v)
            v = v + b * k
        return v

    s.X[:] = x
    s.one_step()  # matrix-free forward step (no sources)
    lhs = y @ s.X
    rhs = step(A.T.tocsr(), y) @ x
    assert abs(lhs - rhs) < 1e-12 * abs(lhs)
    wrong = step(A, y) @ x
    assert abs(lhs - wrong) > 1e-6 * abs(lhs)


def _pulse_run(d, R0, h, p, extra=0.0, bc="abc"):
    """Gaussian pulse at the centre of a box with physical half width
    0.9 um (+ PML of thickness d); field on a 15 x 15 grid of points inside
    at 6, 9 and 15 fs."""
    H = 0.9e-6 + d + extra
    m = circles_mesh([], uniform(h), box=(-H, H, -H, H))
    pml = dict(thickness=d, R0=R0) if d else None
    s = DGTD2D(m, p, polarization="TM", bc=bc, pml=pml)
    s.u[:] = np.exp(-(s.x**2 + s.y**2) / (0.25e-6) ** 2)
    g = np.linspace(-0.85e-6, 0.85e-6, 15)
    P = s.interp_matrix(np.array([(a, b) for a in g for b in g]))
    out = []
    for T in (6e-15, 9e-15, 15e-15):
        n = int(round((T - s.t) / s.dt))
        s.dt = (T - s.t) / n
        s.run(n)
        out.append(P @ s.u.ravel())
    return np.array(out)


def test_pml_absorbs():
    """Error against a 4.5x larger domain (reflections not yet back):
    the CFS-PML (0.6 um, 2 elements) is > 10x better than the bare
    first-order ABC. Mutation: sigma = 0 (R0 = 1) is much worse."""
    ref = _pulse_run(0.0, None, 0.2e-6, 4, extra=3.2e-6)
    err = {}
    for name, (d, R0) in dict(pml=(0.6e-6, 1e-6), abc=(0.0, None),
                              zero=(0.6e-6, 1.0)).items():  # fmt: skip
        err[name] = np.abs(_pulse_run(d, R0, 0.3e-6, 3) - ref).max(1)
    assert np.all(err["pml"] < 2e-3)
    assert np.all(err["pml"][:2] < 0.15 * err["abc"][:2])
    assert err["zero"][1] > 3 * err["pml"][1]


def test_drude_cavity_frequency():
    """A Drude-filled PEC cavity oscillates at w^2 = w_p^2 + (c k)^2.
    Mutation: dropping the polarization current gives w = c k."""
    from wakis.dispersive import DispersiveMaterial, Drude

    L = 1e-6
    m = circles_mesh([], uniform(0.25e-6), box=(0, L, 0, L))
    k = np.pi * np.sqrt(2) / L
    wp = 1.0 * C0 * k
    mat = DispersiveMaterial(1.0, [Drude(omega_p=wp, gamma=0.0)])
    freqs = {}
    for name, mats in (("drude", {0: mat}), ("vacuum", {})):
        s = DGTD2D(m, 3, mats, polarization="TM", bc="pec")
        s.u[:] = np.sin(np.pi * s.x / L) * np.sin(np.pi * s.y / L)
        probe = s.interp_matrix(np.array([[0.31e-6, 0.27e-6]]))
        sig, ts = [], []
        n = int(6 * 2 * np.pi / (C0 * k) / s.dt)
        for _ in range(n):
            s.one_step()
            sig.append((probe @ s.u.ravel())[0])
            ts.append(s.t)
        sig, ts = np.array(sig), np.array(ts)
        wgrid = np.linspace(0.5, 2.0, 3001) * C0 * k
        win = np.hanning(len(ts))
        spec = np.abs(np.exp(-1j * np.outer(wgrid, ts)) @ (win * sig))
        freqs[name] = wgrid[np.argmax(spec)]
    # ratio: the spatial discretization error cancels
    ratio = freqs["drude"] / freqs["vacuum"]
    assert abs(ratio / np.sqrt(wp**2 + (C0 * k) ** 2) * (C0 * k) - 1) < 4e-3


# ---------------------------------------------------------------- (2,4)
def _fd_cavity_phase_error(order, h, scheme="lsrk4"):
    from wakis.highorder.fdtd24 import FDTD24

    s = FDTD24(h=h, half_width=0.5e-6, n_pml=0, cylinders=[], polarization="TM",
               order=order, time_scheme=scheme, cfl=0.5)  # fmt: skip
    L = s.xn[-1] - s.xn[0]
    X, Y = np.meshgrid(s.xn - s.xn[0], s.xn - s.xn[0], indexing="ij")
    mode = np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
    s.Dz[:] = mode
    w = C0 * np.pi * np.sqrt(2) / L
    T = 2.25 * 2 * np.pi / w  # cos(w T) = 0: error linear in the phase error
    n = int(np.ceil(T / s.dt))
    s.dt = T / n
    for _ in range(n):
        s.one_step()
    return np.abs(s.scalar_field() - mode * np.cos(w * s.t)).max()


def test_fd24_spatial_order():
    """(2,4) converges at 4th order with the 4th-order time integrator; Yee
    (2,2) at 2nd order. (The PEC box is exactly representable: the ghost
    values of the sine mode are odd reflections of the interior.)"""
    e4 = [_fd_cavity_phase_error(4, h) for h in (0.1e-6, 0.05e-6)]
    e2 = [_fd_cavity_phase_error(2, h) for h in (0.1e-6, 0.05e-6)]
    assert np.log2(e4[0] / e4[1]) > 3.5
    assert 1.7 < np.log2(e2[0] / e2[1]) < 2.4
    assert e4[1] < 0.02 * e2[1]
