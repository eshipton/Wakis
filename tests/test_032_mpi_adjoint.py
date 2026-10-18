import numpy as np
import pytest

import wakis
from wakis.inverse import AdjointSimulation, AdjointStepper, MaterialGrid
from wakis.sources import CurrentSource, GaussianSource

# Run with:
# mpiexec -n 2 python -m pytest --color=yes -v tests/test_032_mpi_adjoint.py
# (also -n 4 and -n 8: 24 z-cells split evenly)

MPI = pytest.importorskip("mpi4py.MPI")
needs_mpi = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() < 2, reason="needs mpiexec -n >= 2"
)

F0 = 1.5e9  # Hz, 14 cells per wavelength on the ~14 mm mesh below


def _solver(use_mpi, bc="pec", bg=(1.0, 1.0, 0.02), use_gpu=False):
    """Box of 14 x 12 x 24 cells, z split into slabs with MPI."""
    kw = {"use_mpi": True} if use_mpi else {}
    grid = wakis.GridFIT3D(
        -0.1, 0.1, -0.1, 0.1, -0.15, 0.15, 14, 12, 24, verbose=0, **kw
    )
    return wakis.SolverFIT3D(
        grid,
        bc_low=[bc, bc, "pec"],
        bc_high=[bc, bc, "pec"],
        bg=list(bg),
        use_gpu=use_gpu,
        verbose=0,
        **kw,
    )


@needs_mpi
@pytest.mark.parametrize("bc", ["pec", "periodic"])
def test_transposed_step(bc, use_gpu):
    """
    Dot-product test <U x, y> = <x, U^T y> of the distributed transposed
    step, with the global inner product over the samples each rank owns.
    U is the forward MPI step (local updates and the H and E ghost-layer
    exchanges); U^T applies the local transposed updates and the reversed
    exchanges. Random x, y on every owned sample, so the subdomain interfaces
    are exercised; exact to float64 rounding (1e-13 of sum |Ux| |y|).
    """
    bg = (2.0, 1.0, 0.05) if bc == "pec" else (1.0, 1.0, 0.0)
    solver = _solver(True, bc, bg, use_gpu)
    solver.one_step()
    stepper = AdjointStepper(solver)
    xp = stepper.xp
    rng = np.random.default_rng(MPI.COMM_WORLD.Get_rank())

    def random_state():
        return {
            k: xp.asarray(rng.standard_normal(v.shape))
            for k, v in stepper.zero_state().items()
        }

    def local_dot(a, b):
        """Sum over ranks of the dot product of the full local arrays."""
        local = sum(complex(xp.vdot(a[k], b[k])) for k in a)
        return MPI.COMM_WORLD.allreduce(local)

    x = random_state()
    stepper.set_state(x)  # fills the ghost layers of x from the neighbours
    solver.one_step()
    Ux = stepper.get_state()
    y = random_state()
    Uty = stepper.step(y)

    # Ux holds copies of the owners' values on its ghost layers, so y acts on
    # the global state through all local entries, ghosts included; x enters
    # through its owned entries only (set_state overwrote its ghosts)
    lhs = local_dot(Ux, y)
    rhs = stepper.inner(x, Uty)
    scale = local_dot(
        {k: abs(v) for k, v in Ux.items()}, {k: abs(v) for k, v in y.items()}
    ).real
    assert abs(lhs - rhs) < 1e-13 * scale


def _simulation(use_mpi, use_gpu=False):
    """
    Lossy PEC box with a design region, a DFT line and two flux planes that
    cross the subdomain interfaces (z = 0 for 2 ranks, z = 0, +-0.075 for 4).
    """
    source = CurrentSource(
        GaussianSource(F0, 0.5e9), component="z", center=(-0.05, 0.01, 0.03)
    )
    f = [0.9 * F0, F0, 1.1 * F0]
    monitors = {
        "line": wakis.DFTMonitor(
            f, components=["Ez", "Hx"], center=(0.05, 0, 0), size=(0, 0, 0.2)
        ),
        "side": wakis.FluxMonitor(f, center=(0.06, 0, 0), size=(0, 0.15, 0.25)),
        "top": wakis.FluxMonitor(f, center=(0, 0, 0.0), size=(0.15, 0.15, 0)),
    }
    design = MaterialGrid(
        center=(0, 0, 0), size=(0.06, 0.06, 0.18), shape=(3, 3, 5), eps_max=3.0
    )
    return AdjointSimulation(
        lambda: _solver(use_mpi, use_gpu=use_gpu), [source], monitors, design, 150
    )


def _objective(sim, out):
    """
    F = a sum |Ez|^2 + b sum Re(Ez Hx*) + c (flux_side + flux_top) and its
    sensitivities G = dF/dRe + 1j dF/dIm, with the flux weights of
    `sim.readouts` (gathered in output order with MPI).
    """
    a, b, c = 1e18, 1e20, 1e20
    Ez, Hx = out["line"]["Ez"], out["line"]["Hx"]
    F = a * np.sum(np.abs(Ez) ** 2) + b * np.sum(np.real(Ez * np.conj(Hx)))
    G = {"line": {"Ez": 2 * a * Ez + b * Hx, "Hx": b * Ez}}
    for name in ("side", "top"):
        G[name] = {}
        for i, term in enumerate(sim.readouts[name]._terms):
            E, H, w = out[name][f"E{i}"], out[name][f"H{i}"], term["weight"]
            F += c * np.sum(np.real(E * np.conj(H)) @ w)
            G[name][f"E{i}"], G[name][f"H{i}"] = c * H * w, c * E * w
    return F, G


@needs_mpi
def test_gradient_matches_serial(use_gpu):
    """
    Forward outputs, objective and adjoint gradient of the distributed run
    equal those of a serial run to rounding (1e-12 relative): the design
    region, the DFT line and the flux planes span several ranks, and a flux
    plane lies on a subdomain interface (its H average reads ghost samples).
    One central finite difference (h = 1e-4, truncation ~1e-8) confirms the
    distributed gradient.
    """
    rng = np.random.default_rng(3)
    rho = rng.uniform(0.2, 0.8, (3, 3, 5))  # same on every rank (same seed)

    results = []
    for use_mpi in (True, False):
        sim = _simulation(use_mpi, use_gpu)
        shapes = sim.output_shapes()
        out = sim.forward(rho)
        for name, comps in shapes.items():
            for key, shape in comps.items():
                assert out[name][key].shape == shape
        F, G = _objective(sim, out)
        results.append((F, sim.vjp(G), sim))
    (F_mpi, grad_mpi, sim_mpi), (F_ser, grad_ser, _) = results

    assert abs(F_mpi - F_ser) < 1e-12 * abs(F_ser)
    np.testing.assert_allclose(
        grad_mpi, grad_ser, rtol=0, atol=1e-12 * np.abs(grad_ser).max()
    )

    k, h = (1, 1, 2), 1e-4
    step = np.zeros_like(rho)
    step[k] = h
    Fp, _ = _objective(sim_mpi, sim_mpi.forward(rho + step))
    Fm, _ = _objective(sim_mpi, sim_mpi.forward(rho - step))
    fd = (Fp - Fm) / (2 * h)
    assert abs(grad_mpi[k] - fd) < 1e-6 * abs(fd), (grad_mpi[k], fd)


@needs_mpi
def test_unsupported_monitor():
    """Monitors without a distributed adjoint are rejected before time-stepping."""
    monitors = {"n2f": wakis.Near2FarMonitor([F0], size=(0.1, 0.1, 0.1))}
    design = MaterialGrid(center=(0, 0, 0), size=(0.06, 0.06, 0.06), shape=(2, 2, 2))
    sim = AdjointSimulation(lambda: _solver(True), [], monitors, design, 5)
    with pytest.raises(NotImplementedError):
        sim.forward(np.full(design.shape, 0.5))
