import numpy as np
import pytest

import wakis
from wakis.inverse import AdjointSimulation, AdjointStepper, MaterialGrid
from wakis.sources import CurrentSource, GaussianSource

F0 = 3e9  # Hz, 13 cells per wavelength on the 7.5 mm mesh below


def _small_solver(bc_low, bc_high, bg=(1.0, 1.0, 0.0)):
    grid = wakis.GridFIT3D(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 14, 12, 12, verbose=0)
    return wakis.SolverFIT3D(
        grid, bc_low=bc_low, bc_high=bc_high, bg=list(bg), n_pml=4, verbose=0
    )


@pytest.mark.parametrize(
    "bc_low, bc_high, bg",
    [
        (["pec"] * 3, ["pec"] * 3, (1.0, 1.0, 0.0)),
        (["pec"] * 3, ["pec"] * 3, (2.0, 1.0, 0.05)),
        (["periodic"] * 3, ["periodic"] * 3, (1.0, 1.0, 0.0)),
        (["cpml"] * 3, ["cpml"] * 3, (1.0, 1.0, 0.0)),
        (["cpml"] * 3, ["cpml"] * 3, (2.0, 1.0, 0.05)),
        (["cpml", "pec", "periodic"], ["pec", "cpml", "periodic"], (1.0, 1.0, 0.0)),
    ],
    ids=["pec", "pec-lossy", "periodic", "cpml", "cpml-lossy", "cpml-mixed"],
)
def test_transposed_step(bc_low, bc_high, bg):
    """
    Dot-product test of the adjoint step: <U x, y> = <x, U^T y> for random
    states x, y including the CPML auxiliary variables psi, where U is the
    solver's own source-free time step. Exact up to float64 rounding, which
    also holds when x or y is supported on psi only, so the small psi terms
    cannot hide behind the field terms.
    """
    solver = _small_solver(bc_low, bc_high, bg)
    solver.one_step()  # first-step initialization (ghosts, operator cleanup)
    stepper = AdjointStepper(solver)
    rng = np.random.default_rng(0)
    keys = stepper.zero_state().keys()
    psi_keys = [k for k in keys if k.startswith("psi")]
    assert bool(psi_keys) == ("cpml" in bc_low + bc_high)

    def random_state(support):
        return {
            k: rng.standard_normal(v.shape) * (k in support)
            for k, v in stepper.zero_state().items()
        }

    supports = [(keys, keys)]
    if psi_keys:
        supports += [(psi_keys, keys), (keys, psi_keys)]
    for x_support, y_support in supports:
        x = random_state(x_support)
        stepper.set_state(x)
        solver.J.fromarray(np.zeros(3 * solver.N))
        if solver.use_conductivity:
            solver.J_old = np.zeros(3 * solver.N)  # J = sigma E of this step
        solver.one_step()
        Ux = stepper.get_state()
        y = random_state(y_support)
        Uty = stepper.step(y)

        lhs = sum(np.dot(Ux[k], y[k]) for k in keys)
        rhs = sum(np.dot(x[k], Uty[k]) for k in keys)
        scale = sum(np.abs(Ux[k]) @ np.abs(y[k]) for k in keys)
        assert abs(lhs - rhs) < 1e-13 * scale


N_2D, L_2D = 40, 0.3
DX_2D = L_2D / N_2D


def _make_2d_solver(boundary="cpml", bg=(1.0, 1.0, 0.0)):
    """TM-like 2D domain: one periodic cell along z, 40 x 40 cells in x, y."""
    grid = wakis.GridFIT3D(
        -L_2D / 2,
        L_2D / 2,
        -L_2D / 2,
        L_2D / 2,
        -DX_2D / 2,
        DX_2D / 2,
        N_2D,
        N_2D,
        1,
        verbose=0,
    )
    return wakis.SolverFIT3D(
        grid,
        bc_low=[boundary, boundary, "periodic"],
        bc_high=[boundary, boundary, "periodic"],
        bg=list(bg),
        n_pml=8,
        verbose=0,
    )


def _scattering_sim(make_solver, Nt=400, store_dtype=np.float64):
    """Point current, 6 x 6 cm dielectric design region, probe behind it."""
    source = CurrentSource(
        GaussianSource(F0, 1e9), component="z", center=(-0.06, 0.0, 0.0)
    )
    monitors = {
        "probe": wakis.DFTMonitor(
            [F0, 1.1 * F0], components=["Ez", "Hy"], center=(0.07, 0.01, 0.0)
        )
    }
    design = MaterialGrid(
        center=(0.0, 0.0, 0.0),
        size=(0.06, 0.06, 0.0),
        shape=(5, 5, 1),
        eps_min=1.0,
        eps_max=4.0,
    )
    return AdjointSimulation(
        make_solver, [source], monitors, design, Nt, store_dtype=store_dtype
    )


def _objective(out):
    """
    F = |Ez(f0)|^2 + Re(Ez Hy*)(1.1 f0) (a flux-like term) at the probe,
    scaled to O(1), and its sensitivities G = dF/dRe + 1j dF/dIm.
    """
    Ez, Hy = out["probe"]["Ez"], out["probe"]["Hy"]
    a, b = 1e20, 1e22
    F = a * np.sum(np.abs(Ez[0]) ** 2) + b * np.sum(np.real(Ez[1] * np.conj(Hy[1])))
    G_Ez, G_Hy = np.zeros_like(Ez), np.zeros_like(Hy)
    G_Ez[0] = 2 * a * Ez[0]
    G_Ez[1] = b * Hy[1]
    G_Hy[1] = b * Ez[1]
    return F, {"probe": {"Ez": G_Ez, "Hy": G_Hy}}


@pytest.mark.parametrize("case", ["cpml", "pec-lossy", "cpml-float32"])
def test_adjoint_gradient(case):
    """
    The adjoint gradient of a DFT objective with respect to the design density
    matches central finite differences. The adjoint is exact for the discrete
    time stepping, so the only error is the O(h^2) truncation of the finite
    difference with h = 1e-4 (~1e-8 relative); float32 storage of the E
    increments adds ~1e-7.
    """
    if case == "pec-lossy":
        # Standard (non-CPML) step with ohmic loss in a closed PEC box
        sim = _scattering_sim(lambda: _make_2d_solver("pec", bg=(1.0, 1.0, 0.01)))
        tol = 1e-6
    else:
        store = np.float32 if case == "cpml-float32" else np.float64
        sim = _scattering_sim(_make_2d_solver, store_dtype=store)
        tol = 1e-5 if case == "cpml-float32" else 1e-6

    rho = np.random.default_rng(1).uniform(0.2, 0.8, sim.design.shape)
    F, G = _objective(sim.forward(rho))
    grad = sim.vjp(G)
    assert grad.shape == rho.shape

    h = 1e-4
    for k in [(0, 0, 0), (2, 2, 0), (4, 1, 0), (1, 3, 0)]:
        step = np.zeros_like(rho)
        step[k] = h
        Fp, _ = _objective(sim.forward(rho + step))
        Fm, _ = _objective(sim.forward(rho - step))
        fd = (Fp - Fm) / (2 * h)
        assert abs(grad[k] - fd) < tol * abs(fd), (k, grad[k], fd)


def test_adjoint_ascent():
    """
    Small normalized gradient-ascent steps on |Ez(f0)|^2 at the probe increase
    the objective monotonically, starting from a uniform density.
    """
    sim = _scattering_sim(_make_2d_solver)
    rho = np.full(sim.design.shape, 0.5)
    values = []
    for _ in range(4):
        F, G = _objective(sim.forward(rho))
        values.append(F)
        grad = sim.vjp(G)
        rho = np.clip(rho + 0.05 * grad / np.abs(grad).max(), 0.0, 1.0)
    assert np.all(np.diff(values) > 0), values


def test_jax_simulation():
    """
    jax.grad through filter + projection + simulation + a JAX objective equals
    the NumPy adjoint gradient pulled back through filter and projection, so
    the complex-cotangent convention of the JAX wrapper is consistent.
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    from wakis.inverse import conic_filter, jax_simulation, tanh_projection

    sim = _scattering_sim(_make_2d_solver, Nt=300)
    simulate = jax_simulation(sim)

    def parametrize(rho_raw):
        return tanh_projection(conic_filter(rho_raw, radius=1.5), beta=4.0)

    def phase_sensitive(out):
        Ez, Hy = out["probe"]["Ez"], out["probe"]["Hy"]
        return 1e22 * jnp.sum(jnp.real(Ez[0] * jnp.conj(Hy[0]))) + 1e10 * jnp.sum(
            jnp.imag(Ez[1])
        )

    rho_raw = jnp.asarray(np.random.default_rng(2).uniform(0, 1, sim.design.shape))
    value, grad = jax.value_and_grad(
        lambda r: phase_sensitive(simulate(parametrize(r)))
    )(rho_raw)

    rho, pullback = jax.vjp(parametrize, rho_raw)
    out = sim.forward(np.asarray(rho))
    Ez, Hy = out["probe"]["Ez"], out["probe"]["Hy"]
    G_Ez, G_Hy = np.zeros_like(Ez), np.zeros_like(Hy)
    G_Ez[0] = 1e22 * Hy[0]
    G_Hy[0] = 1e22 * Ez[0]
    G_Ez[1] = 1e10 * 1j
    expected = pullback(jnp.asarray(sim.vjp({"probe": {"Ez": G_Ez, "Hy": G_Hy}})))[0]

    np.testing.assert_allclose(float(value), phase_sensitive(out), rtol=1e-12)
    np.testing.assert_allclose(
        grad, expected, rtol=1e-10, atol=1e-12 * np.abs(expected).max()
    )


def test_unsupported_boundaries():
    """The adjoint refuses features whose transposed step is not implemented."""
    grid = wakis.GridFIT3D(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 8, 8, 8, verbose=0)
    solver = wakis.SolverFIT3D(
        grid, bc_low=["pec", "pec", "abc"], bc_high=["pec", "pec", "abc"], verbose=0
    )
    with pytest.raises(NotImplementedError):
        AdjointStepper(solver)
