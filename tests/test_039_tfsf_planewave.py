"""
Plane-wave total-field/scattered-field box source (`wakis.TFSFSource`).

The incident field comes from a 1D auxiliary FIT line with the grid's lengths
along the propagation axis, and the 3D update is corrected with the discrete
equivalence-principle (commutator) terms on the box surface. Outside the box
the incident field then cancels to round-off; the mutation tests show that a
half-step time shift or a one-cell E/H misregistration of the incident field
breaks the cancellation by many orders of magnitude.
"""

import numpy as np
import pytest
import pyvista as pv
from scipy.constants import c, epsilon_0, mu_0

import wakis
from wakis.sources import GaussianSource

ETA0 = np.sqrt(mu_0 / epsilon_0)


def _axis_nodes(n, length, graded):
    """Uniform nodes, or nodes graded smoothly (cell sizes 0.6x...1.6x)."""
    u = np.linspace(-1.0, 1.0, n + 1)
    if graded:
        u = u + 0.25 * np.sin(np.pi * (u + 1) / 2) * (1 - u**2)
    return 0.5 * length * u


def _solver(n=16, length=0.16, bc="pec", graded_axis=None, use_gpu=False, **kw):
    nodes = [_axis_nodes(n, length, d == graded_axis) for d in range(3)]
    grid_kw = {
        k: kw.pop(k) for k in list(kw) if k.startswith("stl") or k == "conformal"
    }
    grid = wakis.GridFIT3D(x=nodes[0], y=nodes[1], z=nodes[2], verbose=0, **grid_kw)
    return wakis.SolverFIT3D(
        grid,
        bc_low=[bc] * 3,
        bc_high=[bc] * 3,
        use_gpu=use_gpu,
        verbose=0,
        **kw,
    )


def _host(a):
    return a.get() if hasattr(a, "get") else a


def _leakage(solver, src, Nt):
    """max |field outside the box| / max |field inside the box| over the run,
    for E and for eta0 * H."""
    src.prepare(solver)
    masks = {f: src.total_field_mask(solver, f, flat=True) for f in "EH"}
    out = {"E": [0.0, 0.0], "H": [0.0, 0.0]}

    def record(s, t):
        for f in "EH":
            a = np.abs(_host(getattr(s, f).array))
            out[f][0] = max(out[f][0], a[~masks[f]].max())
            out[f][1] = max(out[f][1], a[masks[f]].max())

    solver.emsolve(Nt, source=[src], callback=record)
    return max(out[f][0] / out[f][1] for f in "EH"), out["E"][1]


def _pulse():
    return GaussianSource(3e9, 1.5e9)


def _steps(solver, t):
    return int(t / solver.dt)


DIRECTIONS = ["+x", "-x", "+y", "-y", "+z", "-z"]


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_vacuum_leakage(direction, graded, use_gpu):
    """
    Empty domain: the scattered-field region stays at round-off for every
    direction and both polarizations, on uniform grids and on grids graded
    along the propagation axis (cell sizes varying by ~2.6x). Recorded:
    ~1e-15 relative (float64).
    """
    a = "xyz".index(direction[1])
    for pol in [d for d in "xyz" if d != direction[1]]:
        solver = _solver(graded_axis=a if graded else None, use_gpu=use_gpu)
        src = wakis.TFSFSource(
            _pulse(), size=(0.08, 0.08, 0.08), direction=direction, polarization=pol
        )
        leak, peak = _leakage(solver, src, _steps(solver, 2.2e-9))
        print(f"{direction} pol {pol} graded={graded}: leakage {leak:.2e}")
        assert 0.9 < peak < 1.2  # the incident wave crossed the box
        assert leak < 1e-12


@pytest.mark.parametrize(
    "mutation, graded",
    [
        ("_mutation_time_shift", False),  # e_inc one step late in the H update
        ("_mutation_index_offset", False),  # E_inc one cell off w.r.t. H_inc
        ("_mutation_index_shift_both", True),  # both shifted, graded axis
    ],
)
def test_leakage_mutations_fail(mutation, graded):
    """
    Rule 1: the leakage test can fail. A half-step time-level error or a
    one-cell misregistration of the incident field leaks 1e-1 of the
    incident field (recorded: 0.098, 0.36, 0.14). Shifting both E_inc and
    H_inc by one cell on a uniform grid is only a translated incident wave
    (still a discrete solution, leakage ~1e-15), hence the graded grid there.
    """
    solver = _solver(graded_axis=2 if graded else None)
    src = wakis.TFSFSource(
        _pulse(), size=(0.08, 0.08, 0.08), direction="+z", polarization="x"
    )
    setattr(src, mutation, 1)
    leak, _ = _leakage(solver, src, _steps(solver, 2.2e-9))
    print(f"{mutation}: leakage {leak:.2e}")
    assert leak > 1e-3


def test_float32_and_cpml(use_gpu):
    """CPML outer boundary (float64 and float32): round-off leakage."""
    for dtype, tol in ((np.float64, 1e-12), (np.float32, 1e-5)):
        solver = _solver(
            n=28, length=0.28, bc="cpml", n_pml=8, use_gpu=use_gpu, dtype=dtype
        )
        src = wakis.TFSFSource(
            _pulse(), size=(0.08, 0.1, 0.08), direction="-y", polarization="z"
        )
        leak, _ = _leakage(solver, src, _steps(solver, 2.5e-9))
        print(f"CPML {np.dtype(dtype).name}: leakage {leak:.2e}")
        assert leak < tol


def test_box_inside_pml_raises():
    solver = _solver(n=20, length=0.2, bc="cpml", n_pml=8)
    src = wakis.TFSFSource(_pulse(), size=(0.12, 0.08, 0.08), direction="+z")
    with pytest.raises(ValueError, match="non-PML"):
        src.prepare(solver)


def test_box_surface_in_background_raises():
    sphere = pv.Sphere(radius=0.05, center=(0, 0, 0))
    solver = _solver(
        n=20,
        length=0.2,
        use_stl=True,
        stl_solids={"s": sphere},
        stl_materials={"s": [4.0, 1.0]},
    )
    src = wakis.TFSFSource(_pulse(), size=(0.08, 0.08, 0.08), direction="+z")
    with pytest.raises(ValueError, match="homogeneous"):
        src.prepare(solver)


@pytest.mark.parametrize("direction, pol", [("+z", "y"), ("-x", "z")])
def test_total_field_and_spectrum(direction, pol, use_gpu):
    """
    Inside the box the total field equals the auxiliary-line incident field
    (to round-off), the entry-plane spectrum equals amplitude * S(f) up to the
    grid dispersion over the cell between the injection point and the entry
    plane, ~(k h)^3/24 rad (5e-3 at 2 GHz, 10.6 cells per wavelength), and the incident flux density is
    |E|^2/eta cos(k h / 2) (H averaged from the two dual samples, as
    FluxMonitor does).
    """
    eps_r = 2.0
    solver = _solver(n=20, length=0.2, bg=[eps_r, 1.0], use_gpu=use_gpu)
    st = GaussianSource(1.5e9, 0.6e9, cutoff=8)
    src = wakis.TFSFSource(
        st,
        size=(0.1, 0.1, 0.1),
        direction=direction,
        polarization=pol,
        amplitude=2.5,
        background_eps_r=eps_r,
    )
    src.prepare(solver)
    a, p = "xyz".index(direction[1]), "xyz".index(pol)
    mask = src.total_field_mask(solver, "E")[p]
    idx_a = np.nonzero(mask)[a]  # index along the propagation axis of each TF edge
    err = [0.0, 0.0]

    def compare(s, t):
        E = _host(s.E.array).reshape(3, -1, order="C")[p]
        E = E.reshape(s.Nx, s.Ny, s.Nz, order="F")[mask]
        line = _host(src._e)[idx_a + src._offset]
        err[0] = max(err[0], np.abs(E - line).max())
        err[1] = max(err[1], np.abs(line).max())

    solver.emsolve(_steps(solver, 6e-9), source=[src], callback=compare)
    assert err[0] < 1e-12 * err[1]

    f = np.linspace(1.0e9, 2.0e9, 6)
    E, H = src.incident_fields(f)
    S = 2.5 * st.fourier_transform(f)
    k = 2 * np.pi * f * np.sqrt(eps_r) / c
    h = 0.01  # cell size at the entry plane
    print("entry-plane relative error", np.abs(E / S - 1), "bound", (k * h) ** 3 / 24)
    assert np.all(np.abs(E / S - 1) < 1.5 * (k * h) ** 3 / 24 + 1e-4)
    eta = ETA0 / np.sqrt(eps_r)
    np.testing.assert_allclose(
        src.incident_flux_density(f),
        np.abs(S) ** 2 / eta * np.cos(k * h / 2),
        rtol=4e-3,
    )
    assert np.all(src.incident_flux_density(f) > 0)


def _sphere_run(n, sigma=0.0, eps_r=4.0, conformal=False, use_gpu=False, disp=None):
    """Sphere (radius 3.5 cm) in a TF box (12 cm) in a 40 cm CPML domain.

    Returns the flux spectra through a closed box inside the TF region, a
    closed box in the SF region, the incident flux density, the Ohmic loss
    sum sigma |E(f)|^2 dV and the frequencies."""
    length = 0.4
    radius = 0.035
    sphere = pv.Sphere(radius=radius, theta_resolution=60, phi_resolution=60)
    material = [eps_r, 1.0, sigma] if disp is None else disp
    solver = _solver(
        n=n,
        length=length,
        bc="cpml",
        n_pml=8,
        use_gpu=use_gpu,
        use_stl=True,
        stl_solids={"s": sphere},
        stl_materials={"s": material},
        conformal=conformal,
    )
    st = GaussianSource(1.0e9, 0.4e9)
    src = wakis.TFSFSource(
        st, size=(0.12, 0.12, 0.12), direction="+x", polarization="z"
    )
    f = np.linspace(0.8e9, 1.2e9, 3)
    # Monitors placed from the box actually used on the grid: the TF flux box
    # one cell inside the TF/SF surface, the SF flux box three cells outside
    src.prepare(solver)
    h = length / n
    half = src.box_bounds()[0][1]
    tot = wakis.FluxMonitor(f, size=(2 * (half - h),) * 3)
    scat = wakis.FluxMonitor(f, size=(2 * (half + 3 * h),) * 3)
    region = wakis.DFTMonitor(
        f, components=("Ex", "Ey", "Ez"), size=(3.0 * radius,) * 3
    )
    solver.emsolve(_steps(solver, 14e-9), source=[src], monitors=[tot, scat, region])
    loss = np.zeros(f.size)
    if sigma > 0:
        grid = solver.grid
        sig = solver.sigma
        for comp in "xyz":
            F, _ = region.get_field("E" + comp)
            (I, J, K), _ = grid.region_indices(
                "E", comp, (0, 0, 0), (3.0 * radius,) * 3
            )
            I, J, K = np.meshgrid(I, J, K, indexing="ij")
            dV = grid.L[I, J, K, comp] / grid.itA[I, J, K, comp]
            s = _host(sig[I, J, K, comp]) if hasattr(sig, "__getitem__") else sig
            loss += np.sum(s * dV * np.abs(F) ** 2, axis=(1, 2, 3))
    return tot.get_flux(), scat.get_flux(), src.incident_flux_density(f), loss, f


def test_energy_lossless_and_lossy(use_gpu):
    """
    Lossless dielectric sphere (eps_r = 4): net flux out of a closed box in the
    TF region vanishes (< 1e-6 of the incident power on a box face, recorded
    ~3e-8; with the flux faces cutting the staircased sphere it was 2e-2)
    while the scattered power through a box in the SF region is positive. Lossy sphere
    (sigma = 0.05 S/m): absorbed power -Phi_tot equals the volume Ohmic loss
    sum sigma |E(f)|^2 dV to 2e-3 (recorded 3e-4).
    """
    tot, scat, inc, _, f = _sphere_run(40, use_gpu=use_gpu)
    face = 0.1**2  # TF flux box face (TF box faces at +-6 cm)
    print("lossless: net/inc face", tot / (inc * face), "sigma_sca [m^2]", scat / inc)
    assert np.all(np.abs(tot) < 1e-6 * inc * face)
    assert np.all(scat > 1e-2 * inc * face)

    tot, scat, inc, loss, f = _sphere_run(40, sigma=0.05, use_gpu=use_gpu)
    print("lossy: absorbed", -tot, "ohmic", loss, "sca", scat)
    assert np.all(-tot > 0)
    np.testing.assert_allclose(-tot, loss, rtol=2e-3)


def test_scattering_cross_section_mie():
    """
    Scattered cross-section sigma_sca = Phi_scat / Phi_inc of a conformal
    eps_r = 4 sphere (radius 3.5 cm, ka = 0.59-0.88) against Mie theory, at
    1 cm and 0.71 cm cells: within 10 % (recorded +6.5..7.5 % at 1 cm,
    +4.8..5.4 % at 0.71 cm) and closer at the finer grid. The legacy staircased sphere scatters 3.4x
    Mie at 1 cm because its marked volume is 1.7x the sphere volume; the
    study `studies/tfsf/` runs the full convergence.
    """
    f = np.linspace(0.8e9, 1.2e9, 3)
    k = 2 * np.pi * f / c
    mie = np.array([_mie_qsca(2.0, kk * 0.035) for kk in k]) * np.pi * 0.035**2
    err = []
    for n in (40, 56):
        _, scat, inc, _, _ = _sphere_run(n, conformal=True)
        err.append(np.abs(scat / inc / mie - 1))
        print(f"n={n}: sigma_sca/mie - 1 =", scat / inc / mie - 1)
    assert np.all(err[0] < 0.1) and np.all(err[1] < 0.1)
    assert np.all(err[1] < err[0])


def _mie_qsca(m, x, n_max=None):
    """Mie scattering efficiency of a homogeneous sphere (Bohren & Huffman)."""
    from scipy.special import spherical_jn, spherical_yn

    n_max = n_max or int(x + 4 * x ** (1 / 3) + 2)
    n = np.arange(1, n_max + 1)

    def psi(z):
        return z * spherical_jn(n, z)

    def dpsi(z):
        return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)

    def xi(z):
        return z * (spherical_jn(n, z) + 1j * spherical_yn(n, z))

    def dxi(z):
        h = spherical_jn(n, z) + 1j * spherical_yn(n, z)
        dh = spherical_jn(n, z, derivative=True) + 1j * spherical_yn(
            n, z, derivative=True
        )
        return h + z * dh

    mx = m * x
    an = (m * psi(mx) * dpsi(x) - psi(x) * dpsi(mx)) / (
        m * psi(mx) * dxi(x) - xi(x) * dpsi(mx)
    )
    bn = (psi(mx) * dpsi(x) - m * psi(x) * dpsi(mx)) / (
        psi(mx) * dxi(x) - m * xi(x) * dpsi(mx)
    )
    return 2 / x**2 * np.sum((2 * n + 1) * (np.abs(an) ** 2 + np.abs(bn) ** 2))


def test_conformal_and_dispersive_inside(use_gpu):
    """
    Conformal dielectric sphere and a lossy Drude sphere inside the box: the
    lossless conformal sphere conserves energy (net TF flux < 1e-6 of the
    face power), the Drude sphere absorbs (-Phi_tot > 0) and scatters.
    """
    tot, scat, inc, _, f = _sphere_run(32, conformal=True, use_gpu=use_gpu)
    face = 0.1**2
    print("conformal: net/inc face", tot / (inc * face))
    assert np.all(np.abs(tot) < 1e-6 * inc * face)
    assert np.all(scat > 0)

    from wakis.dispersive import DispersiveMaterial, Drude

    drude = DispersiveMaterial(
        eps_inf=1.0, poles=[Drude(omega_p=2 * np.pi * 3e9, gamma=2 * np.pi * 0.5e9)]
    )
    tot, scat, inc, _, f = _sphere_run(32, disp=drude, use_gpu=use_gpu)
    print("drude: absorbed", -tot / (inc * face), "sca", scat / (inc * face))
    assert np.all(-tot > 1e-4 * inc * face)
    assert np.all(scat > 0)


def test_adjoint_gradient_with_tfsf(use_gpu):
    """
    A TFSFSource is an additive forcing independent of the fields, so the
    exact discrete adjoint applies unchanged: the gradient of the scattered
    flux (SF box) with respect to a dielectric design block inside the TF box
    matches central finite differences to 1e-6.
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from wakis.inverse import (
        AdjointSimulation,
        MaterialGrid,
        flux,
        jax_simulation,
    )

    def make_solver():
        return _solver(n=24, length=0.24, bc="cpml", n_pml=6, use_gpu=use_gpu)

    src = wakis.TFSFSource(
        GaussianSource(1.5e9, 0.6e9), size=(0.08, 0.08, 0.08), direction="+z"
    )
    f = np.array([1.5e9])
    monitors = {"scat": wakis.FluxMonitor(f, size=(0.11, 0.11, 0.11))}
    design = MaterialGrid(
        center=(0, 0, 0),
        size=(0.04, 0.04, 0.04),
        shape=(3, 3, 3),
        eps_min=1.0,
        eps_max=4.0,
    )
    sim = AdjointSimulation(make_solver, [src], monitors, design, Nt=300)
    sim.output_shapes()
    simulate = jax_simulation(sim)

    def objective(out):
        return jax.numpy.sum(flux(out["scat"], sim.readouts["scat"])) * 1e20

    rho = np.random.default_rng(0).uniform(0.3, 0.7, design.shape)
    value, grad = jax.value_and_grad(lambda r: objective(simulate(r)))(rho)
    assert float(value) > 0
    h = 1e-4
    for k in [(1, 1, 1), (0, 2, 1)]:
        step = np.zeros_like(rho)
        step[k] = h
        fd = (
            float(objective(sim.forward(rho + step)))
            - float(objective(sim.forward(rho - step)))
        ) / (2 * h)
        err = abs(grad[k] - fd) / abs(fd)
        print(f"node {k}: adjoint {grad[k]:.6e} FD {fd:.6e} rel {err:.1e}")
        assert err < 1e-6


# --------------------------------------------------------------------------
# 2D: one-cell-thick domain, periodic along z, CPML in x and y
# --------------------------------------------------------------------------
def _solver_2d(n=40, length=0.4, n_pml=8, nz=1, use_gpu=False, dtype=np.float64, **kw):
    h = length / n
    x = np.linspace(-length / 2, length / 2, n + 1)
    z = h * (np.arange(nz + 1) - nz / 2)
    grid_kw = {
        k: kw.pop(k) for k in list(kw) if k.startswith("stl") or k == "conformal"
    }
    grid = wakis.GridFIT3D(x=x, y=x.copy(), z=z, verbose=0, **grid_kw)
    return wakis.SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=n_pml,
        use_gpu=use_gpu,
        dtype=dtype,
        verbose=0,
        **kw,
    )


@pytest.mark.parametrize("direction", ["+x", "-x", "+y", "-y"])
def test_2d_vacuum_leakage(direction, use_gpu):
    """
    2D (periodic z, one cell thick, size None along z: no faces on z): TM
    (E_z) and TE (E in-plane) incidence leak at round-off (recorded ~1e-15),
    and the leakage test can fail (half-step mutation).
    """
    in_plane = "y" if direction[1] == "x" else "x"
    for pol in ("z", in_plane):
        solver = _solver_2d(use_gpu=use_gpu)
        src = wakis.TFSFSource(
            GaussianSource(1.5e9, 0.6e9),
            size=(0.16, 0.16, None),
            direction=direction,
            polarization=pol,
        )
        leak, peak = _leakage(solver, src, _steps(solver, 3e-9))
        print(f"2D {direction} pol {pol}: leakage {leak:.2e}")
        assert src._idx[2] is None
        assert 0.9 < peak < 1.2
        assert leak < 1e-12
    solver = _solver_2d(use_gpu=use_gpu)
    src = wakis.TFSFSource(
        GaussianSource(1.5e9, 0.6e9),
        size=(0.16, 0.16, 1.0),  # >= the domain along z: also spans it
        direction=direction,
        polarization="z",
    )
    src._mutation_time_shift = 1
    leak, _ = _leakage(solver, src, _steps(solver, 3e-9))
    assert leak > 1e-3


def test_2d_none_size_on_open_axis_raises():
    solver = _solver_2d()
    src = wakis.TFSFSource(
        _pulse(), size=(0.16, None, None), direction="+x", polarization="z"
    )
    with pytest.raises(ValueError, match="periodic"):
        src.prepare(solver)


@pytest.mark.parametrize("pol", ["z", "y"])
def test_2d_total_field(pol, use_gpu):
    """2D: inside the box the total field equals the auxiliary-line field."""
    solver = _solver_2d(use_gpu=use_gpu)
    src = wakis.TFSFSource(
        GaussianSource(1.5e9, 0.6e9, cutoff=8),
        size=(0.16, 0.16, None),
        direction="+x",
        polarization=pol,
    )
    src.prepare(solver)
    p = "xyz".index(pol)
    mask = src.total_field_mask(solver, "E")[p]
    idx_a = np.nonzero(mask)[0]
    err = [0.0, 0.0]

    def compare(s, t):
        E = _host(s.E.array).reshape(3, -1)[p]
        E = E.reshape(s.Nx, s.Ny, s.Nz, order="F")[mask]
        line = _host(src._e)[idx_a + src._offset]
        err[0] = max(err[0], np.abs(E - line).max())
        err[1] = max(err[1], np.abs(line).max())

    solver.emsolve(_steps(solver, 4e-9), source=[src], callback=compare)
    assert err[1] > 0.9
    assert err[0] < 1e-12 * err[1]


def _cylinder_run(pol, sigma=0.0, eps_r=4.0, use_gpu=False):
    """Infinite cylinder (radius 3.5 cm) in 2D (3 cells along periodic z).
    Returns the TF-box net flux, the SF-box flux (both through a slab of
    thickness h along z), the incident flux density, the Ohmic loss in that
    slab and the area of a TF-box side face."""
    n, length = 40, 0.4
    h = length / n
    radius = 0.035
    cyl = pv.Cylinder(radius=radius, height=1.0, direction=(0, 0, 1), resolution=120)
    # Three cells along z: a closed FluxMonitor box needs its z faces on two
    # distinct nodes above node 0; their fluxes cancel by periodicity
    solver = _solver_2d(
        n=n,
        length=length,
        nz=3,
        use_gpu=use_gpu,
        use_stl=True,
        stl_solids={"c": cyl},
        stl_materials={"c": [eps_r, 1.0, sigma]},
    )
    st = GaussianSource(1.0e9, 0.4e9)
    src = wakis.TFSFSource(
        st, size=(0.12, 0.12, None), direction="+x", polarization=pol
    )
    src.prepare(solver)
    half = src.box_bounds()[0][1]
    f = np.linspace(0.8e9, 1.2e9, 3)
    # Closed boxes spanning z: the z faces see no net flux (periodic)
    tot = wakis.FluxMonitor(f, size=(2 * (half - h), 2 * (half - h), h))
    scat = wakis.FluxMonitor(f, size=(2 * (half + 3 * h), 2 * (half + 3 * h), h))
    region = wakis.DFTMonitor(
        f, components=("Ex", "Ey", "Ez"), size=(3 * radius, 3 * radius, h)
    )
    solver.emsolve(_steps(solver, 14e-9), source=[src], monitors=[tot, scat, region])
    loss = np.zeros(f.size)
    if sigma > 0:
        grid = solver.grid
        for comp in "xyz":
            F, _ = region.get_field("E" + comp)
            (I, J, K), _ = grid.region_indices(
                "E", comp, (0, 0, 0), (3 * radius, 3 * radius, h)
            )
            I, J, K = np.meshgrid(I, J, K, indexing="ij")
            dV = grid.L[I, J, K, comp] / grid.itA[I, J, K, comp]
            if comp != "z":
                # E_x, E_y sit on the z nodes +-h/2, the slab faces: half of
                # their control volume lies in the slab (trapezoid rule)
                dV = 0.5 * dV
            s = _host(solver.sigma[I, J, K, comp])
            loss += np.sum(s * dV * np.abs(F) ** 2, axis=(1, 2, 3))
    face = 2 * (half - h) * h
    return tot.get_flux(), scat.get_flux(), src.incident_flux_density(f), loss, face


@pytest.mark.parametrize("pol", ["z", "y"])
def test_2d_cylinder_energy(pol, use_gpu):
    """
    2D dielectric cylinder (eps_r = 4), TM and TE: lossless -> net flux out of
    a closed box in the TF region < 1e-6 of the incident power on a face,
    scattered power > 0; lossy (sigma = 0.05 S/m) -> absorbed power equals
    the volume Ohmic loss to 2e-3. Scattering width = P_sca / (L_z I_inc).
    """
    tot, scat, inc, _, face = _cylinder_run(pol, use_gpu=use_gpu)
    print(
        f"2D {pol} lossless: net/face {tot / (inc * face)}, width {scat / (inc * 0.01)} m"
    )
    assert np.all(np.abs(tot) < 1e-6 * inc * face)
    assert np.all(scat > 1e-3 * inc * face)
    tot, scat, inc, loss, face = _cylinder_run(pol, sigma=0.05, use_gpu=use_gpu)
    print(f"2D {pol} lossy: absorbed {-tot}, ohmic {loss}")
    assert np.all(-tot > 0)
    np.testing.assert_allclose(-tot, loss, rtol=2e-3)
