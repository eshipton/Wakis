"""
Grid conventions: where the domain walls sit and which lengths the dual grid
uses. A domain [min, max] meshed with N cells must have its non-periodic
boundaries (PEC, PMC, ABC, CPML) on the planes min and max, and the dual
lengths of a non-uniform mesh must be centred on the primary nodes.
"""

import numpy as np
from scipy.constants import c, epsilon_0, mu_0

import wakis
from wakis.sources import CurrentSource, GaussianSource


class _Probe:
    """Record one field sample per timestep (monitor protocol of emsolve)."""

    def __init__(self, field, component, center):
        self.field, self.component, self.center = field, component, center
        self.values = []
        self.idx = None

    def update(self, solver, t):
        if self.idx is None:
            indices, _ = solver.grid.region_indices(
                self.field, self.component, self.center, (0.0, 0.0, 0.0)
            )
            self.idx = tuple(int(i[0]) for i in indices)
        arr = getattr(solver, self.field)
        self.values.append(float(arr[(*self.idx, self.component)]))


def _resonance(solver, f_guess, src_center, probe_center, component="z"):
    """
    Frequency of the resonance closest to `f_guess`, from the free ring-down
    after a broadband pulse. The spectrum is windowed with a Gaussian (40
    periods rms), whose transform peaks exactly on the frequency of an
    isolated undamped mode; neighbouring modes are >= 25 window widths away.
    """
    src_time = GaussianSource(frequency=f_guess, fwidth=0.5 * f_guess)
    source = CurrentSource(src_time, component=component, center=src_center)
    probe = _Probe("E", component, probe_center)
    t_start = src_time.end_time
    sigma_t = 40 / f_guess
    Nt = int((t_start + 8 * sigma_t) / solver.dt)
    solver.emsolve(Nt, source=source, monitors=[probe])

    t = (np.arange(Nt) + 1) * solver.dt
    keep = t > t_start
    t, s = t[keep], np.asarray(probe.values)[keep]
    window = np.exp(-0.5 * ((t - t.mean()) / sigma_t) ** 2)
    f = f_guess * np.linspace(0.97, 1.03, 601)
    spectrum = np.abs(np.exp(-2j * np.pi * np.outer(f, t)) @ (s * window))
    i = np.argmax(spectrum)
    assert 0 < i < f.size - 1, "Resonance outside the search band"
    # Parabola through the log of the Gaussian peak (exact for one mode)
    y0, y1, y2 = np.log(spectrum[i - 1 : i + 2])
    return f[i] + 0.5 * (f[1] - f[0]) * (y0 - y2) / (y0 - 2 * y1 + y2)


def _discrete_frequency(k_tilde_sq, dt):
    """Leapfrog frequency for a discrete spatial eigenvalue k_tilde^2:
    (2/(c dt)) sin(w dt/2) = k_tilde."""
    return np.arcsin(0.5 * c * dt * np.sqrt(k_tilde_sq)) / (np.pi * dt)


def _k_tilde_sq(n_half_waves, length, d):
    """Discrete wavenumber^2 of sin(n pi x/length) on a uniform mesh d."""
    return (2 / d * np.sin(n_half_waves * np.pi * d / (2 * length))) ** 2


def _graded_axis(length, n):
    """Smoothly graded nodes: cell sizes shrink monotonically by a factor 3
    from the low to the high end of the axis."""
    u = np.linspace(0.0, 1.0, n + 1)
    return length * (u + 0.5 * u * (1 - u))


A, B, D = 1.0, 0.7, 0.2  # PEC box [m]
F110 = c / 2 * np.sqrt(1 / A**2 + 1 / B**2)
SRC = (0.31 * A, 0.27 * B, 0.5 * D)  # generic point: couples to all TMmn0
PROBE = (0.63 * A, 0.58 * B, 0.5 * D)


def test_pec_cavity_walls_at_domain_bounds(use_gpu):
    """
    TM110 resonance of an empty PEC box a x b x d whose walls are set by the
    domain bounds. On a uniform mesh Ez = sin(pi x/a) sin(pi y/b) is an exact
    discrete eigenmode, so the simulated frequency equals the leapfrog
    dispersion relation with a, b the domain lengths to the spectral
    estimator's precision (1e-5). A wall one cell inside the domain would
    shift it by ~dx/a = 10 % at 10 cells. The continuum error
    c/2 sqrt(1/a^2 + 1/b^2) then converges at second order (10 and 20 cells
    across a: 10-20 cells per wavelength).
    """
    errors = []
    for n in (10, 20):
        grid = wakis.GridFIT3D(0, A, 0, B, 0, D, n, int(0.7 * n), 2, verbose=0)
        solver = wakis.SolverFIT3D(
            grid, use_gpu=use_gpu, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0
        )
        f_sim = _resonance(solver, F110, SRC, PROBE)
        k2 = _k_tilde_sq(1, A, A / n) + _k_tilde_sq(1, B, B / int(0.7 * n))
        np.testing.assert_allclose(f_sim, _discrete_frequency(k2, solver.dt), rtol=1e-5)
        errors.append(abs(f_sim / F110 - 1))
    assert errors[0] < 0.01
    assert errors[0] / errors[1] > 3.5  # 2nd order: 4


def test_legacy_convention(use_gpu):
    """
    With boundary_at_domain_edge=False (legacy grids) the last index is a
    ghost layer and the high-side PEC walls sit one cell inside the domain:
    the TM110 frequency matches the discrete relation for a - dx, b - dy.
    The field arrays then hold one sample per cell.
    """
    n = 10
    grid = wakis.GridFIT3D(
        0, A, 0, B, 0, D, n, 7, 2, boundary_at_domain_edge=False, verbose=0
    )
    solver = wakis.SolverFIT3D(
        grid, use_gpu=use_gpu, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0
    )
    assert (solver.Nx, solver.Ny, solver.Nz) == (n, 7, 2)
    f_sim = _resonance(solver, 1.15 * F110, SRC, PROBE)
    dx, dy = A / n, B / 7
    k2 = _k_tilde_sq(1, A - dx, dx) + _k_tilde_sq(1, B - dy, dy)
    np.testing.assert_allclose(f_sim, _discrete_frequency(k2, solver.dt), rtol=1e-5)


def test_pec_cavity_graded_mesh(use_gpu):
    """
    The same TM110 resonance on smoothly graded meshes (cell sizes varying by
    a factor 3 along x and y). With dual lengths centred on the primary nodes
    the frequency error converges at second order (12, 24, 48 cells: -0.65 %,
    -0.16 %, -0.04 %); dual lengths centred on the next node, as in the
    legacy grid, make it first order (+4.9 %, +2.5 %, +1.2 %).
    """
    errors = []
    for n in (12, 24, 48):
        grid = wakis.GridFIT3D(
            x=_graded_axis(A, n),
            y=_graded_axis(B, int(0.7 * n)),
            z=np.linspace(0, D, 3),
            verbose=0,
        )
        solver = wakis.SolverFIT3D(
            grid, use_gpu=use_gpu, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0
        )
        errors.append(abs(_resonance(solver, F110, SRC, PROBE) / F110 - 1))
    order = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert errors[0] < 0.03
    assert np.all(order > 1.7), f"convergence order {order}, errors {errors}"


def test_energy_conservation_graded_mesh(use_gpu):
    """
    In a closed lossless PEC cavity the leapfrog scheme conserves the discrete
    energy W^n = 1/2 sum eps V_e (E^n)^2 + 1/2 sum mu V_h H^(n-1/2) H^(n+1/2),
    with V_e = L tA and V_h = A tL the cell volumes attached to each sample,
    for any (non-uniform) mesh: the H and E updates are mutual transposes in
    these metrics. Once the source is off W must stay constant to round-off.
    """
    n = 12
    grid = wakis.GridFIT3D(
        x=_graded_axis(A, n),
        y=_graded_axis(B, int(0.7 * n)),
        z=_graded_axis(D, 4),
        verbose=0,
    )
    solver = wakis.SolverFIT3D(
        grid, use_gpu=use_gpu, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0
    )
    L, tL = grid.L, grid.tL
    V_e = np.concatenate(
        [
            (L.field_x * tL.field_y * tL.field_z).ravel(order="F"),
            (tL.field_x * L.field_y * tL.field_z).ravel(order="F"),
            (tL.field_x * tL.field_y * L.field_z).ravel(order="F"),
        ]
    )
    V_h = np.concatenate(
        [
            (tL.field_x * L.field_y * L.field_z).ravel(order="F"),
            (L.field_x * tL.field_y * L.field_z).ravel(order="F"),
            (L.field_x * L.field_y * tL.field_z).ravel(order="F"),
        ]
    )

    class _Energy:
        def __init__(self):
            self.prev, self.W = None, []

        def update(self, solver, t):
            E = np.asarray(solver.E.toarray()).copy()
            H = np.asarray(solver.H.toarray()).copy()
            if self.prev is not None:
                E_n, H_old = self.prev  # E^n and H^(n-1/2); H is H^(n+1/2)
                self.W.append(
                    0.5 * epsilon_0 * np.sum(V_e * E_n**2)
                    + 0.5 * mu_0 * np.sum(V_h * H_old * H)
                )
            self.prev = (E, H)

    src_time = GaussianSource(frequency=F110, fwidth=0.5 * F110)
    source = CurrentSource(src_time, component="z", center=SRC)
    energy = _Energy()
    Nt = int(src_time.end_time / solver.dt) + 2000
    solver.emsolve(Nt, source=source, monitors=[energy])
    W = np.array(energy.W[-1500:])
    assert W[0] > 0
    np.testing.assert_allclose(W, W[0], rtol=1e-9)


def test_quarter_wave_pec_pmc(use_gpu):
    """
    TEM quarter-wave resonator between a PEC and a PMC plane a distance d
    apart (periodic x, y): Ex = sin(pi z/(2d)) with the PEC at z = 0, or
    cos(pi z/(2d)) with the PEC at z = d, both at f = c/(4d). Both walls sit
    on the domain bounds, so the simulation matches the 1D leapfrog
    dispersion relation with k = pi/(2d) (16 cells over d).
    """
    d, n = 1.0, 16
    f0 = c / (4 * d)
    for low, high in (("pec", "pmc"), ("pmc", "pec")):
        grid = wakis.GridFIT3D(0, 0.1, 0, 0.1, 0, d, 1, 1, n, verbose=0)
        solver = wakis.SolverFIT3D(
            grid,
            use_gpu=use_gpu,
            bc_low=["periodic", "periodic", low],
            bc_high=["periodic", "periodic", high],
            verbose=0,
        )
        f_sim = _resonance(
            solver, f0, (0.0, 0.0, 0.43 * d), (0.0, 0.0, 0.71 * d), component="x"
        )
        k2 = (2 / (d / n) * np.sin(np.pi / (2 * d) * (d / n) / 2)) ** 2
        np.testing.assert_allclose(
            f_sim, _discrete_frequency(k2, solver.dt), rtol=1e-5, err_msg=low
        )


def test_mixed_pec_pmc_box(use_gpu):
    """
    Box with PEC walls on x and z and PMC walls on y: Ez = sin(pi x/a) is a
    mode (Ez is tangential to the PEC x walls, normal to the PEC z walls, and
    Hy is normal to the PMC y walls) at f = c/(2a). Requires that the PMC
    faces leave the H samples next to the PEC faces untouched and that both
    kinds of walls sit on the domain bounds: the simulated frequency equals
    the discrete dispersion relation of sin(pi x/a) to 1e-5.
    """
    n = 10
    grid = wakis.GridFIT3D(0, A, 0, B, 0, D, n, 7, 2, verbose=0)
    solver = wakis.SolverFIT3D(
        grid,
        use_gpu=use_gpu,
        bc_low=["pec", "pmc", "pec"],
        bc_high=["pec", "pmc", "pec"],
        verbose=0,
    )
    f0 = c / (2 * A)
    f_sim = _resonance(solver, f0, SRC, PROBE)
    k2 = _k_tilde_sq(1, A, A / n)
    np.testing.assert_allclose(f_sim, _discrete_frequency(k2, solver.dt), rtol=1e-5)


def test_verbose_zero_is_quiet(capsys):
    """verbose=0 keeps grid and solver construction silent."""
    grid = wakis.GridFIT3D(0, 1, 0, 1, 0, 1, 4, 4, 4, verbose=0)
    wakis.SolverFIT3D(grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0)
    assert capsys.readouterr().out == ""
