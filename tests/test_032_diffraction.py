from functools import lru_cache

import numpy as np
import pytest
from scipy.constants import c, epsilon_0

import wakis
from wakis.inverse import AdjointSimulation, MaterialGrid
from wakis.sources import CurrentSource, GaussianSource

# ---------------------------------------------------------------------------
# Lamellar grating benchmark (lengths in units of LAM, see
# benchmarks/diffraction_fmmax_reference.py): substrate eps 2.25 (z < 0),
# teeth eps 4 for 0 <= x < P/2 and 0 < z < H, air elsewhere and above.

LAM = 0.3  # m
F0 = c / LAM
P, H = 1.5 * LAM, 0.5 * LAM
EPS_SUB, EPS_TOOTH = 2.25, 4.0

# fmmax 1.7.1 (JONES_DIRECT formulation, 201 orders; change from 51 orders
# < 6e-5), efficiencies {order m: R or T} for incidence from the substrate
FMMAX = {
    (0.0, "TE"): (
        {-2: 0.02761, -1: 0.10495, 0: 0.21484, 1: 0.10495, 2: 0.02761},
        {-1: 0.19795, 0: 0.12414, 1: 0.19795},
    ),
    (0.0, "TM"): (
        {-2: 0.04949, -1: 0.04904, 0: 0.02369, 1: 0.04904, 2: 0.04949},
        {-1: 0.35040, 0: 0.07843, 1: 0.35040},
    ),
    (15.0, "TE"): (
        {-2: 0.02005, -1: 0.07602, 0: 0.24072, 1: 0.16239},
        {-2: 0.04229, -1: 0.44821, 0: 0.01033},
    ),
    (15.0, "TM"): (
        {-2: 0.00747, -1: 0.03555, 0: 0.26927, 1: 0.18981},
        {-2: 0.08941, -1: 0.32736, 0: 0.08115},
    ),
}


def _grating_eps(x, z):
    tooth = (np.mod(x, P) < P / 2) & (z > 0) & (z < H)
    return np.where(z < 0, EPS_SUB, np.where(tooth, EPS_TOOTH, 1.0))


def _set_grating(solver):
    """
    Permittivity of each E edge: mean of eps over its dual face, sampled just
    off the nodes it sits on (x and z), i.e. the arithmetic mean of the
    media touching the edge, which keeps tangential E continuous at the tooth
    walls and the layer interfaces.
    """
    grid, d = solver.grid, 1e-6 * LAM
    nodes = (grid.x[:-1], grid.y[:-1], grid.z[:-1])
    for ci, comp in enumerate("xyz"):
        pos = [(n + np.diff(g) / 2) if k == ci else n for k, (n, g) in
               enumerate(zip(nodes, (grid.x, grid.y, grid.z)))]  # fmt: skip
        X, _, Z = np.meshgrid(*pos, indexing="ij")
        offsets = [(ox, oz) for ox in ((0.0,) if ci == 0 else (-d, d))
                   for oz in ((0.0,) if ci == 2 else (-d, d))]  # fmt: skip
        eps = sum(_grating_eps(X + ox, Z + oz) for ox, oz in offsets) / len(offsets)
        solver.ieps[:, :, :, comp] = 1 / (epsilon_0 * eps)


@lru_cache(maxsize=None)
def _grating_run(n_per_period, pol, theta_deg=0.0, use_gpu=False):
    """
    2D cell (one cell along y), CPML along z, sheet source in the substrate
    launching the incident wave along +z, a reflection monitor between source
    and grating (incident = 'up' (0, 0), reflected = 'down') and a
    transmission monitor in the air. Oblique incidence uses complex Bloch
    fields with k_bloch = -k_par, i.e. tangential wavevector +k_par at f > 0.
    """
    dx, n_pml = P / n_per_period, 15
    zlo = -1.0 * LAM - n_pml * dx
    nz = int(round((2.5 * LAM + 2 * n_pml * dx) / dx))
    grid = wakis.GridFIT3D(
        0, P, 0, dx, zlo, zlo + nz * dx, n_per_period, 1, nz, verbose=0
    )
    k_par = 2 * np.pi / LAM * np.sqrt(EPS_SUB) * np.sin(np.radians(theta_deg))
    bloch = theta_deg != 0
    bc = ["bloch" if bloch else "periodic", "periodic", "cpml"]
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=bc,
        bc_high=bc,
        n_pml=n_pml,
        k_bloch=(-k_par, 0.0, 0.0) if bloch else None,
        dtype=np.complex128 if bloch else np.float64,
        use_gpu=use_gpu,
        verbose=0,
    )
    _set_grating(solver)
    freqs = [F0, -F0] if bloch else [F0]
    # Narrow band and long run at oblique incidence: the +1 order in air
    # grazes at 1.055 F0, where waves linger in the cell
    src = GaussianSource(frequency=F0, fwidth=(0.08 if bloch else 0.15) * F0)
    sheet = CurrentSource(
        src,
        component="y" if pol == "TE" else "x",
        center=(P / 2, dx / 2, -0.7 * LAM),
        size=(P, dx, 0.0),
        amp_func=(lambda x, y, z: np.exp(-1j * k_par * x)) if bloch else None,
    )
    refl = wakis.DiffractionMonitor(freqs, center=(0, 0, -0.4 * LAM), size=(P, dx, 0))
    trans = wakis.DiffractionMonitor(freqs, center=(0, 0, LAM), size=(P, dx, 0))
    extra = 120 if bloch else 40
    Nt = int((src.end_time + extra / F0) / solver.dt)
    solver.emsolve(Nt, source=sheet, monitors=[refl, trans])
    return refl, trans


def _efficiencies(refl, trans):
    """{m: R(f)}, {m: T(f)} normalized by the incident (0, 0) power."""
    pr, pt = refl.get_power(), trans.get_power()
    i0 = list(refl.orders[0]).index(0)
    incident = pr["up"][:, i0, 0]
    R = {int(m): pr["down"][:, i, 0] / incident for i, m in enumerate(refl.orders[0])}
    T = {int(m): pt["up"][:, i, 0] / incident for i, m in enumerate(trans.orders[0])}
    return R, T


def _max_error(R, T, theta, pol, index=0, mirror=False):
    R_ref, T_ref = FMMAX[(theta, pol)]
    sign = -1 if mirror else 1
    errors = [
        abs(sim[sign * m][index] - ref)
        for sim, refs in ((R, R_ref), (T, T_ref))
        for m, ref in refs.items()
    ]
    # Orders absent in the reference carry no power
    for sim, refs in ((R, R_ref), (T, T_ref)):
        errors += [sim[m][index] for m in sim if sign * m not in refs]
    return max(errors)


# ---------------------------------------------------------------------------
# Uniform slab: specular order only, Airy formula, Parseval


def test_slab_orders(use_gpu):
    """
    Plane wave (Ex sheet, normal incidence) on a dielectric slab (n = 1.5,
    d = 0.1 m) in a 3D periodic cell of period 0.45 m x 0.02 m, so that the
    (+-1, 0) orders propagate above 0.67 GHz. From one run: incident = 'up'
    (0, 0) power and R = 'down' (0, 0) power on a plane between source and
    slab, T = 'up' (0, 0) power above the slab. Checks:
    - R and T follow the Airy formula (>= 17 cells per wavelength in the slab:
      Fabry-Perot phase error < 0.3%) and R + T = 1 to 1e-5;
    - only the specular order carries power (< 1e-20 relative) and the wave
      is TM (E in the plane of incidence, Ex at normal incidence, as in
      fmmax/invrs-gym);
    - the order powers 'net' summed over all orders equal the FluxMonitor flux
      to round-off (Parseval) and up - down = net per propagating order;
    - the up/down split is exact for the discrete plane waves: the backward
      wave above the slab (CPML reflection only) is < 1e-8 of the incident.
    """
    period, width, d = 0.45, 0.02, 0.01
    eps_r, thickness = 2.25, 0.1
    grid = wakis.GridFIT3D(0, period, 0, width, -0.6, 0.6, 45, 2, 120, verbose=0)
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=15,
        use_gpu=use_gpu,
        verbose=0,
    )
    z, tol = grid.z[:-1], 1e-6 * d
    for comp in "xy":
        solver.ieps[:, :, (z > tol) & (z < thickness - tol), comp] = 1 / (
            epsilon_0 * eps_r
        )
        solver.ieps[:, :, (np.abs(z) < tol) | (np.abs(z - thickness) < tol), comp] = (
            2 / (epsilon_0 * (1 + eps_r))
        )
    solver.ieps[:, :, (solver.z > 0) & (solver.z < thickness), "z"] = 1 / (
        epsilon_0 * eps_r
    )

    f = np.array([0.6e9, 0.8e9, 1.0e9, 1.2e9])
    src = GaussianSource(frequency=0.9e9, fwidth=0.5e9)
    sheet = CurrentSource(
        src,
        component="x",
        center=(period / 2, width / 2, -0.3),
        size=(period, width, 0),
    )
    plane = dict(size=(period, width, 0.0))
    below = wakis.DiffractionMonitor(f, center=(0, 0, -0.15), **plane)
    above = wakis.DiffractionMonitor(f, center=(0, 0, 0.3), **plane)
    flux_below = wakis.FluxMonitor(f, center=(period / 2, width / 2, -0.15), **plane)
    flux_above = wakis.FluxMonitor(f, center=(period / 2, width / 2, 0.3), **plane)
    Nt = int((src.end_time + 15e-9) / solver.dt)
    solver.emsolve(Nt, source=sheet, monitors=[below, above, flux_below, flux_above])

    assert list(below.orders[0]) == [-1, 0, 1] and list(below.orders[1]) == [0]
    np.testing.assert_array_equal(below.propagating.sum(axis=(1, 2)), [1, 3, 3, 3])
    pb, pa = below.get_power(), above.get_power()
    incident = pb["up"][:, 1, 0]
    R, T = pb["down"][:, 1, 0] / incident, pa["up"][:, 1, 0] / incident
    n = np.sqrt(eps_r)
    T_airy = 1 / (
        1 + 0.25 * (n - 1 / n) ** 2 * np.sin(2 * np.pi * f / c * n * thickness) ** 2
    )
    print("T", T, "Airy", T_airy, "R + T - 1", R + T - 1)
    np.testing.assert_allclose(T, T_airy, atol=3e-3)
    np.testing.assert_allclose(R + T, 1.0, atol=1e-5)

    for power in (pb["up"], pb["down"], pa["up"]):
        others = power.copy()
        others[:, 1, 0] = 0
        assert np.all(others.max(axis=(1, 2)) < 1e-20 * incident)
    amps = below.get_amplitudes()
    assert np.all(np.abs(amps["up_s"][:, 1, 0]) ** 2 < 1e-20 * incident)

    for monitor, power, flux in ((below, pb, flux_below), (above, pa, flux_above)):
        np.testing.assert_allclose(
            power["net"].sum(axis=(1, 2)), flux.get_flux(), rtol=1e-10
        )
        prop = monitor.propagating
        np.testing.assert_allclose(
            (power["up"] - power["down"])[prop], power["net"][prop], rtol=1e-10
        )
    assert np.all(pa["down"][:, 1, 0] < 1e-8 * incident)


# ---------------------------------------------------------------------------
# Lamellar grating against fmmax (RCWA)


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_grating_normal_incidence(pol, use_gpu):
    """
    Order efficiencies of the lamellar grating at normal incidence from the
    substrate (5 reflected, 3 transmitted orders) against fmmax. At 40 cells
    per free-space wavelength (20 in the teeth) the largest absolute error is
    0.56% (TM T0); the error converges at second order (see the slow
    convergence test). R + T = 1 to 3e-4 and the efficiencies are symmetric
    in m.
    """
    R, T = _efficiencies(*_grating_run(60, pol, use_gpu=use_gpu))
    error = _max_error(R, T, 0.0, pol)
    total = sum(v[0] for v in R.values()) + sum(v[0] for v in T.values())
    print(f"{pol}: max |eta - eta_fmmax| = {error:.2e}, R + T - 1 = {total - 1:.1e}")
    assert error < 7.5e-3
    assert abs(total - 1) < 5e-4
    for m in (1, 2):
        np.testing.assert_allclose(R[m], R[-m], rtol=1e-10)


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_grating_convergence(pol):
    """
    Mesh convergence of the order efficiencies towards fmmax at 20, 40 and 80
    cells per free-space wavelength: max absolute errors TE 2.0e-2, 5.1e-3,
    1.3e-3 and TM 2.3e-2, 5.6e-3, 1.3e-3 (second order: the tooth walls and
    layer interfaces fall on grid nodes, where the edge permittivity is the
    arithmetic mean of the touching media).
    """
    errors = [
        _max_error(*_efficiencies(*_grating_run(n, pol)), 0.0, pol)
        for n in (30, 60, 120)
    ]
    print(pol, "errors", np.round(errors, 5))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 2e-3


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_grating_oblique_incidence(pol, use_gpu):
    """
    Oblique incidence at 15 degrees in the substrate (plane of incidence xz)
    with complex Bloch fields. At f > 0 the cell sees the tangential
    wavevector +k_par (k_bloch = -k_par): efficiencies of the 4 reflected and
    3 transmitted orders (specular, -1, -2 ...) match fmmax within 0.6% at 40
    cells per wavelength. At f < 0 the same run describes the mirror image
    (-k_par): the efficiencies of order m equal those of -m at f > 0 (the
    grating is mirror symmetric), exactly. The order angles follow the
    grating equation n sin(theta_m) = n_sub sin(theta) + m lambda / P, with
    the specular reflection at 15 degrees.
    """
    refl, trans = _grating_run(60, pol, 15.0, use_gpu=use_gpu)
    R, T = _efficiencies(refl, trans)
    error = _max_error(R, T, 15.0, pol)
    print(f"{pol}: max |eta - eta_fmmax| = {error:.2e}")
    assert error < 6e-3
    for sim in (R, T):
        for m in sim:
            np.testing.assert_allclose(sim[m][1], sim[-m][0], rtol=1e-9, atol=1e-14)

    sin_inc = np.sqrt(EPS_SUB) * np.sin(np.radians(15.0))
    for monitor, n in ((refl, np.sqrt(EPS_SUB)), (trans, 1.0)):
        theta, phi = monitor.order_angles()
        for i, m in enumerate(monitor.orders[0]):
            s = (sin_inc + m * LAM / P) / n
            if abs(s) < 1:
                expected = np.degrees(np.arcsin(abs(s)))
                np.testing.assert_allclose(np.degrees(theta[0, i, 0]), expected)
                assert np.cos(phi[0, i, 0]) == pytest.approx(np.sign(s))
            else:
                assert np.isnan(theta[0, i, 0])
    i0 = list(refl.orders[0]).index(0)
    assert np.degrees(refl.order_angles()[0][0, i0, 0]) == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Circular polarization


def test_circular_half_wave_plate(use_gpu):
    """
    A uniaxial half-wave layer (n_x = 1.5, n_y = 1.2, (n_x - n_y) k0 d = pi
    at 1 GHz) flips the handedness of circularly polarized light. Complex
    fields with a quarter-period-free source: Ex and Ey sheets with amplitudes
    1 and -1j launch, at f > 0, a purely left-handed (positive helicity)
    wave, and at f < 0 a purely right-handed one (< 1e-14 of the other
    handedness: the TE and TM admittances agree exactly at normal
    incidence). Transmitted: a_L = (t_x + t_y) / 2 and a_R = (t_x - t_y) / 2
    with the Airy amplitudes t_x, t_y of the two slabs, within 0.5% at >= 25
    cells per wavelength in the layer; 99.99% is converted at 1 GHz.
    """
    lam, dz = 0.3, 0.0075  # 40 cells per wavelength in vacuum
    n_x, n_y = 1.5, 1.2
    thickness = lam / (2 * (n_x - n_y))
    nz = int(round(1.8 / dz))
    grid = wakis.GridFIT3D(0, 2 * dz, 0, 2 * dz, -0.6, 1.2, 2, 2, nz, verbose=0)
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["bloch", "bloch", "cpml"],
        bc_high=["bloch", "bloch", "cpml"],
        k_bloch=(0.0, 0.0, 0.0),
        dtype=np.complex128,
        n_pml=20,
        use_gpu=use_gpu,
        verbose=0,
    )
    z, tol = grid.z[:-1], 1e-6 * dz
    for comp, n in (("x", n_x), ("y", n_y)):
        eps = n**2
        solver.ieps[:, :, (z > tol) & (z < thickness - tol), comp] = 1 / (
            epsilon_0 * eps
        )
        on_face = (np.abs(z) < tol) | (np.abs(z - thickness) < tol)
        solver.ieps[:, :, on_face, comp] = 2 / (epsilon_0 * (1 + eps))

    f = np.array([0.95e9, 1.0e9, 1.05e9])
    freqs = np.concatenate([f, -f])
    src = GaussianSource(frequency=1e9, fwidth=0.3e9)
    sheets = [
        CurrentSource(src, component=comp, center=(dz, dz, -0.3), size=(2 * dz, 2 * dz, 0),
                      amplitude=amp)
        for comp, amp in (("x", 1.0), ("y", -1j))
    ]  # fmt: skip
    below = wakis.DiffractionMonitor(freqs, center=(0, 0, -0.15), size=(1, 1, 0))
    above = wakis.DiffractionMonitor(freqs, center=(0, 0, 0.9), size=(1, 1, 0))
    Nt = int((src.end_time + 20e-9) / solver.dt)
    solver.emsolve(Nt, source=sheets, monitors=[below, above])

    inc = below.get_amplitudes("circular")
    out = above.get_amplitudes("circular")
    a_inc_L, a_inc_R = inc["up_L"][:, 0, 0], inc["up_R"][:, 0, 0]
    pos, neg = slice(0, 3), slice(3, 6)
    assert np.all(np.abs(a_inc_R[pos]) ** 2 < 1e-14 * np.abs(a_inc_L[pos]) ** 2)
    assert np.all(np.abs(a_inc_L[neg]) ** 2 < 1e-14 * np.abs(a_inc_R[neg]) ** 2)
    # Circular powers add up to the linear ones
    lin = above.get_power()["up"][:, 0, 0]
    circ = np.abs(out["up_L"][:, 0, 0]) ** 2 + np.abs(out["up_R"][:, 0, 0]) ** 2
    np.testing.assert_allclose(circ, lin, rtol=1e-12)

    def airy(n):
        k = 2 * np.pi * f / c
        r = (1 - n) / (1 + n)
        return (
            (1 - r**2)
            * np.exp(1j * n * k * thickness)
            / (1 - r**2 * np.exp(2j * n * k * thickness))
        )

    t_x, t_y = airy(n_x), airy(n_y)
    conserved, converted = np.abs(t_x + t_y) ** 2 / 4, np.abs(t_x - t_y) ** 2 / 4
    inc_power = np.abs(np.concatenate([a_inc_L[pos], a_inc_R[neg]])) ** 2
    L_out = np.abs(out["up_L"][:, 0, 0]) ** 2 / inc_power
    R_out = np.abs(out["up_R"][:, 0, 0]) ** 2 / inc_power
    print("converted", R_out[pos], L_out[neg], "error", np.abs(R_out[pos] - converted))
    np.testing.assert_allclose(R_out[pos], converted, atol=5e-3)
    np.testing.assert_allclose(L_out[pos], conserved, atol=5e-3)
    np.testing.assert_allclose(L_out[neg], converted, atol=5e-3)
    np.testing.assert_allclose(R_out[neg], conserved, atol=5e-3)
    assert R_out[1] > 0.999 and L_out[4] > 0.999


# ---------------------------------------------------------------------------
# Adjoint gradients of order efficiencies


def _adjoint_case(bloch):
    """
    Periodic cell (period 0.45 m, 16 cells per wavelength at F0, one cell
    along y), sheet source below a 6 x 2 node MaterialGrid, diffraction
    monitor above. Bloch: complex fields with k_bloch = 3 rad/m, monitored
    at +-F0.
    """
    dx, n_pml = P / 24, 10
    k_bloch = 3.0 if bloch else 0.0

    def make_solver():
        grid = wakis.GridFIT3D(0, P, 0, dx, -0.3, 0.3, 24, 1, 32, verbose=0)
        bc = ["bloch" if bloch else "periodic", "periodic", "cpml"]
        return wakis.SolverFIT3D(
            grid,
            bc_low=bc,
            bc_high=bc,
            n_pml=n_pml,
            k_bloch=(k_bloch, 0, 0) if bloch else None,
            dtype=np.complex128 if bloch else np.float64,
            verbose=0,
        )

    src = GaussianSource(F0, 0.5 * F0)
    sheet = CurrentSource(
        src,
        component="y",
        center=(P / 2, dx / 2, -0.12),
        size=(P, dx, 0),
        amp_func=(lambda x, y, z: np.exp(1j * k_bloch * x)) if bloch else None,
    )
    freqs = [F0, -F0] if bloch else [F0, 1.1 * F0]
    monitor = wakis.DiffractionMonitor(freqs, center=(0, 0, 0.12), size=(P, dx, 0))
    design = MaterialGrid(
        center=(P / 2, 0, 0), size=(P, 0, 0.06), shape=(6, 1, 2), eps_min=1, eps_max=4
    )
    sim = AdjointSimulation(make_solver, [sheet], {"orders": monitor}, design, 500)
    return sim


@pytest.mark.parametrize("bloch", [False, True], ids=["periodic", "bloch"])
def test_adjoint_order_efficiency(bloch):
    """
    jax.grad of an order-efficiency objective (TE power into the +1 and -1
    transmitted orders, the second weighted by 0.3, plus the right-handed
    power of order 0) through `jax_simulation` and `diffraction_orders`
    matches central finite differences of the forward run to 1e-6, for real
    periodic fields and complex Bloch fields at +-F0. The JAX helper
    reproduces `DiffractionMonitor.get_amplitudes` to rounding.
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    from wakis.inverse import diffraction_orders, jax_simulation

    sim = _adjoint_case(bloch)
    simulate = jax_simulation(sim)
    monitor = sim.readouts["orders"]
    m = list(monitor.orders[0])
    i_p, i_m, i_0 = m.index(1), m.index(-1), m.index(0)

    def objective(out):
        lin = diffraction_orders(out["orders"], monitor)
        circ = diffraction_orders(out["orders"], monitor, "circular")
        up = jnp.abs(lin["up_s"][:, :, 0]) ** 2
        value = up[:, i_p] + 0.3 * up[:, i_m] + jnp.abs(circ["up_R"][:, i_0, 0]) ** 2
        return 1e18 * jnp.sum(value)

    rho = np.random.default_rng(2).uniform(0.2, 0.8, sim.design.shape)
    value, grad = jax.value_and_grad(lambda r: objective(simulate(r)))(rho)
    assert float(value) > 0

    # Helper vs the post-processing of the monitor of the same forward run
    raw = sim.forward(rho)["orders"]
    ref = sim._monitors["orders"].get_amplitudes()
    for key, v in diffraction_orders(raw, monitor).items():
        np.testing.assert_allclose(v, ref[key], atol=1e-12 * np.abs(v).max())

    h = 1e-4
    for k in [(0, 0, 0), (2, 0, 1), (5, 0, 0)]:
        step = np.zeros_like(rho)
        step[k] = h
        fd = (
            float(objective(sim.forward(rho + step)))
            - float(objective(sim.forward(rho - step)))
        ) / (2 * h)
        error = abs(grad[k] - fd) / abs(fd)
        print(f"bloch={bloch} {k}: adjoint {grad[k]:.6e}, FD {fd:.6e}, rel {error:.1e}")
        assert error < 1e-6
