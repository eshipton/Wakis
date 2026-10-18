import numpy as np
import pytest

import wakis
from wakis.inverse import (
    AdjointSimulation,
    AdjointStepper,
    LumpedReadout,
    MaterialGrid,
)
from wakis.lumped import LumpedElement, LumpedPort
from wakis.materials import Debye, DispersiveMaterial, Drude, Lorentz
from wakis.mode_solver import ModeMonitor, ModeSource
from wakis.sources import CurrentSource, GaussianSource, MagneticCurrentSource

F0 = 3e9  # Hz

# Poles chosen around F0 so that every ADE term is active and stable on the
# grids below (checked by DispersiveMaterial.is_stable on registration)
MATERIAL = DispersiveMaterial(
    2.0,
    [
        Lorentz(1.5, 2 * np.pi * 3.5e9, 2 * np.pi * 0.4e9),
        Drude(2 * np.pi * 2e9, 2 * np.pi * 0.5e9),
        Debye(0.8, 1e-10),
    ],
    sigma=0.02,
)


# ---------------------------------------------------------------------------
# Dot-product tests of the transposed step


def _small_solver(bc_low, bc_high, **kwargs):
    grid = wakis.GridFIT3D(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 14, 12, 12, verbose=0)
    return wakis.SolverFIT3D(
        grid, bc_low=bc_low, bc_high=bc_high, n_pml=4, verbose=0, **kwargs
    )


def _feature_case(case, use_gpu=False):
    """Solver and lumped elements of one dot-product test case."""
    elements = []
    if case.startswith("bloch"):
        z_bc = "cpml" if case == "bloch-cpml" else "pec"
        solver = _small_solver(
            ["bloch", "bloch", z_bc],
            ["bloch", "bloch", z_bc],
            k_bloch=(17.0, -9.0, 0.0),
            dtype=np.complex128,
            bg=[2.0, 1.0, 0.05],
            use_gpu=use_gpu,
        )
    elif case.startswith("lumped"):
        bc = ["cpml" if case == "lumped-cpml" else "pec"] * 3
        solver = _small_solver(bc, bc, use_gpu=use_gpu)
        gap = dict(component="z", size=(0.0, 0.0, 0.05))
        elements = [
            LumpedElement(center=(-0.03, 0.0, 0.0), R=50.0, L=2e-9, C=1e-12, **gap),
            LumpedElement(center=(0.0, 0.03, 0.0), L=5e-9, **gap),
            LumpedPort(None, center=(0.03, -0.02, 0.0), impedance=75.0, **gap),
        ]
    elif case.startswith("dispersive"):
        bc = ["cpml"] * 3 if case == "dispersive-cpml" else ["pec"] * 3
        solver = _small_solver(bc, bc, cfln=0.5, use_gpu=use_gpu)
        solver.add_dispersive_material(
            MATERIAL, center=(0.0, 0.0, 0.0), size=(0.07,) * 3
        )
        solver.add_dispersive_material(
            DispersiveMaterial(1.5, [Debye(1.0, 3e-11)]),
            center=(0.05, 0.0, 0.0),
            size=(0.02, 0.1, 0.1),
        )
    else:  # magnetic current allocated by a magnetic source
        bc = ["cpml"] * 3
        solver = _small_solver(bc, bc, use_gpu=use_gpu)
        src = MagneticCurrentSource(GaussianSource(F0, 1e9), component="z")
        src.update(solver, 0.0)
        assert solver.M is not None
    return solver, elements


def _forward_step(solver, elements, t=0.0):
    """One step of the time loop of `emsolve` (no impressed sources)."""
    for el in elements:
        el.update(solver, t)
    solver.one_step()
    for el in elements:
        el.post_step(solver, t)


@pytest.mark.parametrize(
    "case",
    [
        "magnetic-current",
        "lumped",
        "lumped-cpml",
        "dispersive",
        "dispersive-cpml",
        "bloch",
        "bloch-cpml",
    ],
)
def test_transposed_step(case, use_gpu):
    """
    Dot-product test <U x, y> = <x, U^H y> (complex inner product) of the
    transposed step against the real forward time loop (element `update`,
    `one_step`, element `post_step`) for random states x, y, including the
    auxiliary state of each feature: inductor currents, Lorentz/Drude/Debye
    polarizations and CPML psi. Exact up to float64 rounding (1e-13 relative
    to sum |Ux| |y|), also with x or y supported on the auxiliary variables
    only, so the small auxiliary terms cannot hide behind the field terms.
    The GPU runs in float64 too, so the same tolerance applies there.
    """
    solver, elements = _feature_case(case, use_gpu)
    _forward_step(solver, elements)  # first-step initialization
    stepper = AdjointStepper(solver, elements)
    xp = stepper.xp
    rng = np.random.default_rng(0)
    template = stepper.zero_state()
    keys = list(template)
    aux = [k for k in keys if k not in ("E", "H")]
    assert aux or case in ("bloch", "magnetic-current")

    def random_state(support):
        state = {}
        for k, v in template.items():
            x = rng.standard_normal(v.shape)
            if np.iscomplexobj(v):
                x = x + 1j * rng.standard_normal(v.shape)
            state[k] = xp.asarray(x * (k in support))
        return state

    supports = [(keys, keys)]
    if aux:
        supports += [(aux, keys), (keys, aux)]
    for x_support, y_support in supports:
        x = random_state(x_support)
        stepper.set_state(x)
        _forward_step(solver, elements)
        Ux = stepper.get_state()
        y = random_state(y_support)
        Uty = stepper.step(y)

        lhs = stepper.inner(Ux, y)
        rhs = stepper.inner(x, Uty)
        scale = stepper.inner({k: abs(v) for k, v in Ux.items()},
                              {k: abs(v) for k, v in y.items()}).real  # fmt: skip
        print(f"{case}: |<Ux,y> - <x,U^H y>| / scale = {abs(lhs - rhs) / scale:.1e}")
        assert abs(lhs - rhs) < 1e-13 * scale


# ---------------------------------------------------------------------------
# Finite-difference checks of the adjoint gradient through jax_simulation

N_2D, L_2D = 40, 0.3
DX_2D = L_2D / N_2D  # 7.5 mm: 13 cells per vacuum wavelength at F0
SRC = GaussianSource(F0, 1e9)


def _solver_2d(y_bc="cpml", use_gpu=False, **kwargs):
    """TM-like 2D domain: one periodic cell along z, 40 x 40 cells in x, y."""
    grid = wakis.GridFIT3D(
        -L_2D / 2, L_2D / 2, -L_2D / 2, L_2D / 2, -DX_2D / 2, DX_2D / 2,
        N_2D, N_2D, 1, verbose=0,
    )  # fmt: skip
    return wakis.SolverFIT3D(
        grid,
        bc_low=["cpml", y_bc, "periodic"],
        bc_high=["cpml", y_bc, "periodic"],
        n_pml=8,
        use_gpu=use_gpu,
        verbose=0,
        **kwargs,
    )


def _design(**kwargs):
    """6 x 6 cm design region in the centre, 5 x 5 density nodes."""
    kwargs = {"eps_min": 1.0, "eps_max": 4.0, **kwargs}
    return MaterialGrid(
        center=(0.0, 0.0, 0.0), size=(0.06, 0.06, 0.0), shape=(5, 5, 1), **kwargs
    )


def _line_source():
    return CurrentSource(SRC, component="z", center=(-0.06, 0.0, 0.0))


def _guide_solver(use_gpu=False):
    """2D slab waveguide along x (eps_r = 6, two cells wide)."""
    solver = _solver_2d(use_gpu=use_gpu)
    core = MaterialGrid(
        center=(0, 0, 0), size=(1.0, 2 * DX_2D, 0.0), shape=(1, 1, 1),
        eps_min=6.0, eps_max=6.0,
    )  # fmt: skip
    core.apply(solver, np.ones((1, 1, 1)))
    return solver


def _dispersive_solver(use_gpu=False):
    """2D domain with a dispersive block beside the design region."""
    solver = _solver_2d(cfln=0.5, use_gpu=use_gpu)
    solver.add_dispersive_material(
        MATERIAL, center=(0.0, 0.06, 0.0), size=(0.04, 0.03, 1.0)
    )
    return solver


# Parallel-plate TEM line along z (as in test_022): PEC plates in x, periodic
# in y, CPML at both z ends
DZ, GAP, WIDTH = 2e-3, 10e-3, 40e-3


def _line_solver(use_gpu=False):
    grid = wakis.GridFIT3D(
        0, GAP + DZ, -WIDTH / 2, WIDTH / 2, -0.06, 0.06, 6, 2, 60, verbose=0
    )
    return wakis.SolverFIT3D(
        grid,
        bc_low=["pec", "periodic", "cpml"],
        bc_high=["pec", "periodic", "cpml"],
        n_pml=10,
        use_gpu=use_gpu,
        verbose=0,
    )


def _gap(z, **kwargs):
    return dict(
        component="x", center=(GAP / 2, 0.0, z), size=(GAP, WIDTH, 0.0), **kwargs
    )


def _gradient_case(case, jnp, use_gpu=False):
    """AdjointSimulation, JAX objective and checked design nodes of a case."""
    from functools import partial

    from wakis.inverse import farfield, flux, mode_amplitudes, power_waves

    solver_2d = partial(_solver_2d, use_gpu=use_gpu)
    guide_solver = partial(_guide_solver, use_gpu=use_gpu)
    dispersive_solver = partial(_dispersive_solver, use_gpu=use_gpu)
    line_solver = partial(_line_solver, use_gpu=use_gpu)

    nodes = [(0, 0, 0), (2, 2, 0), (4, 1, 0)]
    if case == "flux":
        monitors = {
            "flux": wakis.FluxMonitor(
                [F0, 1.1 * F0], center=(0.07, 0, 0), size=(0, 0.1, DX_2D)
            )
        }
        sim = AdjointSimulation(solver_2d, [_line_source()], monitors, _design(), 400)

        def objective(out):
            return 1e20 * jnp.sum(flux(out["flux"], sim.readouts["flux"]))

    elif case == "near2far":
        monitors = {
            "n2f": wakis.Near2FarMonitor(
                [F0], center=(0.07, 0, 0), size=(0, 0.1, DX_2D)
            )
        }
        sim = AdjointSimulation(solver_2d, [_line_source()], monitors, _design(), 400)

        def objective(out):
            # Radiation intensity in two directions of the x-y plane
            n2f = sim.readouts["n2f"]
            ff = farfield(out["n2f"], n2f, np.pi / 2, np.array([0.0, 0.3]))
            U = (jnp.abs(ff["Etheta"]) ** 2 + jnp.abs(ff["Ephi"]) ** 2) / n2f.impedance
            return 1e20 * jnp.sum(U)

    elif case == "mode":
        # ModeSource (J and M sheets) in, mode amplitudes a+ and a- out
        window = (0, 0.16, np.inf)  # excludes the CPML layers along y
        sources = [ModeSource(SRC, center=(-0.08, 0, 0), size=window)]
        monitors = {
            "out": ModeMonitor([F0, 1.1 * F0], center=(0.08, 0, 0), size=window)
        }
        design = _design(eps_max=6.0)
        sim = AdjointSimulation(guide_solver, sources, monitors, design, 500)

        def objective(out):
            a_plus, a_minus = mode_amplitudes(out["out"], sim.readouts["out"])
            return 1e18 * jnp.sum(jnp.abs(a_plus) ** 2 + 0.3 * jnp.abs(a_minus) ** 2)

    elif case == "magnetic-current":
        sources = [MagneticCurrentSource(SRC, component="x", center=(-0.06, 0.01, 0.0))]
        monitors = {
            "probe": wakis.DFTMonitor([F0], components=["Ez"], center=(0.07, 0.01, 0))
        }
        sim = AdjointSimulation(solver_2d, sources, monitors, _design(), 400)

        def objective(out):
            return 1e18 * jnp.sum(jnp.abs(out["probe"]["Ez"]) ** 2)

    elif case == "lumped":
        # Driven port, design, passive port, RLC load: |S21|^2 + load current
        port1 = LumpedPort(GaussianSource(3e9, 1.5e9), **_gap(-0.03))
        port2 = LumpedPort(None, **_gap(0.03))
        load = LumpedElement(**_gap(0.04, R=200.0, L=3e-9, C=0.5e-12))
        freqs = [2e9, 3.5e9]  # 75 and 43 cells per wavelength
        monitors = {
            "p1": LumpedReadout(port1, freqs),
            "p2": LumpedReadout(port2, freqs),
            "load": LumpedReadout(load, freqs),
        }
        design = MaterialGrid(
            center=(GAP / 2, 0, 0), size=(GAP, WIDTH, 0.02), shape=(1, 1, 5)
        )
        sim = AdjointSimulation(
            line_solver, [port1, port2, load], monitors, design, 700
        )
        nodes = [(0, 0, 0), (0, 0, 2), (0, 0, 4)]

        def objective(out):
            a1, _ = power_waves(out["p1"]["V"], out["p1"]["I"], 50.0)
            _, b2 = power_waves(out["p2"]["V"], out["p2"]["I"], 50.0)
            load_current = 1e8 * jnp.sum(jnp.real(out["load"]["I"]))
            return jnp.sum(jnp.abs(b2 / a1) ** 2) + load_current

    elif case in ("dispersive", "dispersive-design", "bloch"):
        monitors = {
            "probe": wakis.DFTMonitor(
                [F0, 1.1 * F0], components=["Ez", "Hy"], center=(0.07, 0.01, 0)
            )
        }
        if case == "dispersive":
            make_solver, design = dispersive_solver, _design()
        elif case == "dispersive-design":
            # rho interpolates vacuum -> MATERIAL (fill fraction of the poles)
            design = _design(material=MATERIAL)
            make_solver = lambda: solver_2d(cfln=0.5)  # noqa: E731
        else:
            design = _design()
            make_solver = lambda: solver_2d(  # noqa: E731
                y_bc="bloch", k_bloch=(0.0, 12.0, 0.0), dtype=np.complex128
            )
            monitors["flux"] = wakis.FluxMonitor(
                [F0, -F0], center=(0.07, 0, 0), size=(0, 1.0, DX_2D)
            )
        sim = AdjointSimulation(make_solver, [_line_source()], monitors, design, 400)

        def objective(out):
            Ez, Hy = out["probe"]["Ez"], out["probe"]["Hy"]
            value = 1e18 * jnp.sum(jnp.abs(Ez) ** 2)
            value += 1e20 * jnp.sum(jnp.real(Ez * jnp.conj(Hy)))
            if "flux" in out:
                value += 1e20 * jnp.sum(flux(out["flux"], sim.readouts["flux"]))
            return value

    return sim, objective, nodes


GRADIENT_CASES = [
    "flux",
    "near2far",
    "mode",
    "magnetic-current",
    "lumped",
    "dispersive",
    "dispersive-design",
    "bloch",
]


@pytest.mark.parametrize("case", GRADIENT_CASES)
def test_adjoint_gradient(case, use_gpu):
    """
    jax.grad of an objective of the raw monitor outputs (flux, far field,
    mode amplitudes, |S21|^2 and element current, DFT fields) through
    `jax_simulation` matches central finite differences of the forward run.
    The adjoint is exact for the discrete time stepping, so the difference is
    the O(h^2) truncation of the finite difference with h = 1e-4 (~1e-8
    relative), well below the 1e-6 tolerance. Covers ModeSource and
    MagneticCurrentSource (M sources), lumped R/L/C elements and ports,
    dispersive media beside the design and as the design material, and
    complex Bloch fields (positive and negative frequencies).
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from wakis.inverse import jax_simulation

    sim, objective, nodes = _gradient_case(case, jax.numpy, use_gpu)
    simulate = jax_simulation(sim)
    rho = np.random.default_rng(1).uniform(0.2, 0.8, sim.design.shape)
    value, grad = jax.value_and_grad(lambda r: objective(simulate(r)))(rho)
    assert np.isfinite(float(value)) and abs(float(value)) > 0

    h = 1e-4
    for k in nodes:
        step = np.zeros_like(rho)
        step[k] = h
        fd = (
            float(objective(sim.forward(rho + step)))
            - float(objective(sim.forward(rho - step)))
        ) / (2 * h)
        error = abs(grad[k] - fd) / abs(fd)
        print(f"{case} {k}: adjoint {grad[k]:.6e}, FD {fd:.6e}, rel. error {error:.1e}")
        assert error < 1e-6


def test_readout_helpers(use_gpu):
    """
    The JAX helpers applied to the raw outputs reproduce the monitors' own
    post-processing (flux, far field, mode amplitudes, port power waves) to
    rounding, so objectives written with them optimize the quantities the
    monitors report.
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from wakis.inverse import farfield, flux, mode_amplitudes, power_waves

    for case in ("flux", "near2far", "mode", "lumped"):
        sim, _, _ = _gradient_case(case, jax.numpy, use_gpu)
        sim.output_shapes()
        out = sim.forward(np.full(sim.design.shape, 0.5))
        for name, monitor in sim._monitors.items():
            raw = out[name]
            if isinstance(monitor, wakis.Near2FarMonitor):
                theta, phi = np.pi / 2, np.array([0.0, 0.3])
                ff = farfield(raw, sim.readouts[name], theta, phi)
                ref = monitor.get_farfield(theta, phi)
                scale = np.abs(ref["Etheta"]).max()
                for key in ("Etheta", "Ephi"):
                    np.testing.assert_allclose(ff[key], ref[key], atol=1e-12 * scale)
            elif isinstance(monitor, wakis.FluxMonitor):
                np.testing.assert_allclose(
                    flux(raw, sim.readouts[name]), monitor.get_flux(), rtol=1e-12
                )
            elif isinstance(monitor, ModeMonitor):
                ours = mode_amplitudes(raw, sim.readouts[name])
                for a, ref in zip(ours, monitor.get_amplitudes()):
                    np.testing.assert_allclose(a, ref, rtol=1e-10)
            elif isinstance(monitor.element, LumpedPort):
                f = monitor.frequencies
                a, b = power_waves(raw["V"], raw["I"], monitor.element.impedance)
                for x, ref in zip((a, b), monitor.element.power_waves(f)):
                    np.testing.assert_allclose(x, ref, rtol=1e-12)


def test_design_overlap_raises():
    """A design region sharing edges with a lumped port is rejected."""
    port = LumpedPort(GaussianSource(3e9, 1.5e9), **_gap(0.0))
    design = MaterialGrid(
        center=(GAP / 2, 0, 0), size=(GAP, WIDTH, 0.02), shape=(1, 1, 3)
    )
    sim = AdjointSimulation(_line_solver, [port], {}, design, 5)
    with pytest.raises(NotImplementedError):
        sim.forward(np.full(design.shape, 0.5))


def test_mode_coupler_ascent():
    """
    User-guide example: a design region between two offset slab waveguides
    (eps_r = 6, 3 cells wide, 20 cells per vacuum wavelength) optimized for
    the power transmitted into the fundamental mode of the output guide,
    T = |a+|^2 / |S|^2 with the unit-power ModeSource. Normalized gradient
    ascent through filter + tanh projection increases T at every one of the
    first iterations, starting from T = 0.22 for the uniform density.
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    from wakis.inverse import (
        conic_filter,
        jax_simulation,
        mode_amplitudes,
        tanh_projection,
    )

    dx, eps_core, offset = 5e-3, 6.0, 0.02

    def make_solver():
        grid = wakis.GridFIT3D(
            -0.15, 0.15, -0.1, 0.1, -dx / 2, dx / 2, 60, 40, 1, verbose=0
        )
        solver = wakis.SolverFIT3D(
            grid,
            bc_low=["cpml", "cpml", "periodic"],
            bc_high=["cpml", "cpml", "periodic"],
            n_pml=10,
            verbose=0,
        )
        for x0, y0 in ((-0.1, -offset), (0.1, offset)):
            core = MaterialGrid(
                center=(x0, y0, 0), size=(0.12, 3 * dx, 0), shape=(1, 1, 1),
                eps_min=eps_core, eps_max=eps_core,
            )  # fmt: skip
            core.apply(solver, np.ones((1, 1, 1)))
        return solver

    window = (0, 0.12, np.inf)
    source = ModeSource(SRC, center=(-0.1, -offset, 0), size=window)
    monitors = {"out": ModeMonitor([F0], center=(0.1, offset, 0), size=window)}
    design = MaterialGrid(
        center=(0, 0, 0), size=(0.08, 0.08, 0), shape=(17, 17, 1),
        eps_min=1.0, eps_max=eps_core,
    )  # fmt: skip
    sim = AdjointSimulation(make_solver, [source], monitors, design, Nt=1200)
    simulate = jax_simulation(sim)
    S = SRC.fourier_transform(F0)

    def transmission(rho_raw):
        rho = tanh_projection(conic_filter(rho_raw, radius=1.5), 4.0)
        a_plus, _ = mode_amplitudes(simulate(rho)["out"], sim.readouts["out"])
        return jnp.sum(jnp.abs(a_plus) ** 2) / jnp.abs(S) ** 2

    rho = np.full(design.shape, 0.5)
    values = []
    for _ in range(4):
        T, grad = jax.value_and_grad(transmission)(rho)
        values.append(float(T))
        rho = np.clip(rho + 0.1 * grad / np.abs(grad).max(), 0.0, 1.0)
    print("transmission:", np.round(values, 3))
    assert np.all(np.diff(values) > 0), values
    assert 0.2 < values[0] < 0.25 and values[-1] > 0.8
