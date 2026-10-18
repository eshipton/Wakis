import itertools

import numpy as np
import pytest
from scipy.constants import c, epsilon_0
from scipy.optimize import brentq, curve_fit
from scipy.special import h2vp, hankel2, jv, jvp

import wakis
from wakis.resonances import ResonanceMonitor, harminv
from wakis.sources import CurrentSource, GaussianSource

# Synthetic ring-down: sum of decaying modes x(t) = 2 Re sum a_k exp(2j pi F_k t)
DT = 1e-11
N_SAMPLES = 3000
T_RECORD = N_SAMPLES * DT
F_MODES = np.array([1.0e9, 1.0e9 + 0.3 / T_RECORD, 1.5e9])  # pair 0.3/T apart
Q_MODES = np.array([3000.0, 5000.0, 200.0])
A_MODES = np.array([1.0, 0.8j, 0.5])


def _synthetic(noise=0.0, real=True, seed=0):
    t = np.arange(N_SAMPLES) * DT
    F = F_MODES + 1j * F_MODES / (2 * Q_MODES)
    x = np.exp(2j * np.pi * np.outer(t, F)) @ A_MODES
    x = 2 * x.real if real else x
    return x + noise * np.random.default_rng(seed).standard_normal(N_SAMPLES)


@pytest.mark.parametrize("method", ["fdm", "pencil"])
@pytest.mark.parametrize("real", [True, False])
def test_harminv_exact_modes(method, real):
    """
    An exact sum of decaying exponentials is recovered to round-off by both
    filter diagonalization and the matrix pencil: frequencies, Q = f/(2 Im F)
    and complex amplitudes, including a pair 0.3/T apart (three times closer
    than the Fourier resolution 1/T of the record). For real signals the
    negative-frequency images lie outside the window.
    """
    res = harminv(_synthetic(real=real), DT, 0.9e9, 1.6e9, method=method)

    assert len(res) == 3
    np.testing.assert_allclose(res.frequency, F_MODES, rtol=1e-10)
    np.testing.assert_allclose(res.Q, Q_MODES, rtol=1e-6)
    np.testing.assert_allclose(res.amplitude, A_MODES, rtol=1e-8)
    np.testing.assert_allclose(res.decay_rate, np.pi * F_MODES / Q_MODES, rtol=1e-6)
    if method == "fdm":
        # Round-off level error estimate for an exact model
        assert np.all(res.error < 1e-12 * F_MODES)


@pytest.mark.parametrize("method", ["fdm", "pencil"])
def test_harminv_noise(method):
    """
    With white noise of 1e-5 times the peak signal, the well-resolved
    1.5 GHz mode (Q = 200) keeps its frequency to 1e-8 and Q to 1e-4; the close
    pair (0.3/T apart, sub-Fourier resolution) is still separated, with its
    frequencies to 1e-6. Frequency errors follow the Cramer-Rao scaling
    noise / (|a| T sqrt(N)) ~ 1e-9 f.
    """
    res = harminv(_synthetic(noise=1e-5), DT, 0.9e9, 1.6e9, method=method, rcond=1e-6)

    assert len(res) == 3
    np.testing.assert_allclose(res.frequency[2], F_MODES[2], rtol=1e-8)
    np.testing.assert_allclose(res.Q[2], Q_MODES[2], rtol=1e-4)
    np.testing.assert_allclose(res.frequency[:2], F_MODES[:2], rtol=1e-6)
    np.testing.assert_allclose(np.abs(res.amplitude), np.abs(A_MODES), rtol=1e-2)


def _box_complex_frequencies(a, b, d, dx, dt, sigma, fmin, fmax):
    """
    Exact complex frequencies of the leapfrog scheme in a PEC box filled with a
    medium of conductivity sigma (explicit loss term sigma E^n). A mode with
    discrete curl-curl eigenvalue Om^2 = c^2 sum (2/dx sin(k dx/2))^2 obeys
    z^2 + (Om^2 dt^2 - 2 + s) z + 1 - s = 0 with z = exp(2j pi F dt) and
    s = sigma dt / eps0, from eliminating H between the E and H updates.
    Only modes with Ez != 0 (m, n >= 1) are listed.
    """
    modes = {}
    s = sigma * dt / epsilon_0
    for m, n, p in itertools.product(range(1, 6), range(1, 6), range(0, 6)):
        k2 = sum(
            (2 / dx * np.sin(q * np.pi * dx / (2 * ell))) ** 2
            for q, ell in ((m, a), (n, b), (p, d))
        )
        z = np.roots([1.0, c**2 * k2 * dt**2 - 2 + s, 1 - s])
        F = np.log(z[np.argmax(np.angle(z))]) / (2j * np.pi * dt)
        if fmin < F.real < fmax:
            modes[(m, n, p)] = F
    return modes


@pytest.mark.parametrize("sigma", [0.0, 1e-3])
def test_cavity_modes(sigma, use_gpu):
    """
    Ring-down of a 100 x 80 x 60 mm PEC box (5 mm cells, 12-25 cells per
    wavelength) excited by a broadband dipole: harminv on Ez at a generic point
    finds every mode with Ez != 0 in 2-5 GHz. The walls lie exactly on the
    domain bounds, so the frequencies match the discrete dispersion relation
    (including the timestep) to 1e-9, and lossless modes have Q > 1e9.
    With a uniform conductivity the modes decay at the exact discrete rate
    (Q to 1e-6) and Q = omega eps0 / sigma holds to O(sigma dt/eps0) ~ 5e-4.
    """
    a, b, d, dx = 0.10, 0.08, 0.06, 5e-3
    grid = wakis.GridFIT3D(0, a, 0, b, 0, d, 20, 16, 12, verbose=0)
    solver = wakis.SolverFIT3D(
        grid,
        use_gpu=use_gpu,
        bc_low=["pec"] * 3,
        bc_high=["pec"] * 3,
        bg=[1.0, 1.0, sigma],
        verbose=0,
    )
    src_time = GaussianSource(frequency=3.5e9, fwidth=1.5e9)
    sources = [
        CurrentSource(src_time, component=comp, center=(0.031, 0.023, 0.017))
        for comp in "xyz"
    ]
    probe = ResonanceMonitor(
        "Ez", center=(0.067, 0.051, 0.039), t_start=src_time.end_time
    )
    Nt = int((src_time.end_time + 30e-9) / solver.dt)
    solver.emsolve(Nt, source=sources, monitors=[probe])

    res = probe.harminv(2e9, 5e9)
    expected = _box_complex_frequencies(a, b, d, dx, solver.dt, sigma, 2e9, 5e9)

    assert len(res) == len(expected) == 8
    F_expected = np.array(sorted(expected.values(), key=np.real))
    np.testing.assert_allclose(res.frequency, F_expected.real, rtol=1e-9)
    if sigma == 0:
        assert np.all(np.abs(res.Q) > 1e9)
    else:
        np.testing.assert_allclose(
            res.Q, F_expected.real / (2 * F_expected.imag), rtol=1e-6
        )
        omega = 2 * np.pi * res.frequency
        np.testing.assert_allclose(res.Q, omega * epsilon_0 / sigma, rtol=1e-3)


def _loaded_cavity_error(N, average, a=0.1, t_slab=0.033, eps_r=4.0):
    """
    Relative error of the TE_101-like mode of a square PEC cavity (side a, Ey
    only, 2D: one cell between PEC plates along y) loaded with a dielectric
    slab 0 < z < t_slab. Exact: with kx = pi/a, k1^2 = eps_r k0^2 - kx^2 and
    k2^2 = k0^2 - kx^2, continuity of Ey and dEy/dz at z = t_slab gives
    k1 cos(k1 t) sin(k2 (a - t)) + k2 sin(k1 t) cos(k2 (a - t)) = 0.
    """

    def g(k0):
        k1 = np.sqrt(complex(eps_r * k0**2 - (np.pi / a) ** 2))
        k2 = np.sqrt(complex(k0**2 - (np.pi / a) ** 2))
        return np.real(
            k1 * np.cos(k1 * t_slab) * np.sin(k2 * (a - t_slab))
            + k2 * np.sin(k1 * t_slab) * np.cos(k2 * (a - t_slab))
        )

    # Lowest root: first sign change above the slab cutoff kx / sqrt(eps_r)
    k0 = np.linspace(1.0001 / np.sqrt(eps_r), 3, 2000) * np.pi / a
    values = np.array([g(k) for k in k0])
    i = np.flatnonzero(values[:-1] * values[1:] < 0)[0]
    f_exact = brentq(g, k0[i], k0[i + 1]) * c / (2 * np.pi)

    dx = a / N
    grid = wakis.GridFIT3D(0, a, 0, dx, 0, a, N, 1, N, verbose=0)
    solver = wakis.SolverFIT3D(grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0)
    # Ey on the z nodes is tangential to the slab face: the arithmetic mean of
    # eps over the dual cell is second-order accurate; staircasing takes the
    # eps at the node
    z = grid.z[: solver.Nz]
    lo, hi = np.maximum(z - dx / 2, 0), np.minimum(z + dx / 2, a)
    fill = np.clip(np.minimum(hi, t_slab) - lo, 0, None) / (hi - lo)
    fill = fill if average else (z < t_slab).astype(float)
    for k in range(solver.Nz):
        solver.ieps[:, :, k, "y"] = 1 / (epsilon_0 * (1 + fill[k] * (eps_r - 1)))
    solver.update_tensors("ieps")
    solver.step_0 = True  # update_tensors skips the first-step initialization

    src_time = GaussianSource(frequency=1.5e9, fwidth=0.7e9)
    dipole = CurrentSource(src_time, component="y", center=(0.037, 0, 0.061))
    probe = ResonanceMonitor("Ey", center=(0.061, 0, 0.021), t_start=src_time.end_time)
    solver.emsolve(
        int((src_time.end_time + 8e-9) / solver.dt), source=dipole, monitors=[probe]
    )
    res = probe.harminv(0.8 * f_exact, 1.2 * f_exact)
    assert len(res) == 1
    return res.frequency[0] / f_exact - 1


def test_loaded_cavity_convergence():
    """
    Convergence order of a resonance with a dielectric interface between grid
    nodes (Meep's 2D convergence test idea): with subcell averaging the error
    of the slab-loaded cavity mode (f = 1.517 GHz) falls at second order,
    -0.23 %, -0.060 %, -0.017 % at 15, 30, 60 cells (13-50 cells per
    wavelength in the slab), whereas staircasing is first order and ten times
    less accurate (+4.2 %, +1.9 %, +0.71 %).
    """
    averaged = np.abs([_loaded_cavity_error(N, True) for N in (15, 30, 60)])
    staircase = np.abs([_loaded_cavity_error(N, False) for N in (15, 30, 60)])

    assert averaged[-1] < 3e-4
    assert np.all(averaged[:-1] / averaged[1:] > 3.0)  # 4 for second order
    assert np.all(staircase > 10 * averaged)


def _disk_resonance(m, n, x0):
    """
    Exact complex k R of a TM (Ez) whispering-gallery mode of a dielectric
    cylinder of index n in vacuum: continuity of Ez and H_phi gives
    n J_m'(n k R) / J_m(n k R) = H_m^(2)'(k R) / H_m^(2)(k R) (outgoing waves
    for phasors exp(+j omega t)). Newton iteration from x0.
    """

    def g(x):
        return n * jvp(m, n * x) / jv(m, n * x) - h2vp(m, x) / hankel2(m, x)

    x = complex(x0)
    for _ in range(50):
        h = 1e-7 * abs(x)
        step = g(x) * 2 * h / (g(x + h) - g(x - h))
        x -= step
        if abs(step) < 1e-13 * abs(x):
            return x
    raise RuntimeError("Newton iteration did not converge.")


def _disk_ringdown(dx, T, use_gpu=False, radius=0.1, n_disk=3.0, n_pml=10):
    """2D (one periodic cell along z) dielectric disk in CPML; Ez ring-down."""
    L = 0.36 + 2 * n_pml * dx
    N = round(L / dx)
    grid = wakis.GridFIT3D(-L / 2, L / 2, -L / 2, L / 2, 0, 10 * dx, N, N, 1, verbose=0)
    solver = wakis.SolverFIT3D(
        grid,
        use_gpu=use_gpu,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=n_pml,
        verbose=0,
    )
    # Ez sits on the nodes and is tangential to the disk surface: the
    # arithmetic mean of eps over each dual cell (8x8 subsamples) is the
    # second-order interface treatment for this polarization
    sub = dx * ((np.arange(8) + 0.5) / 8 - 0.5)
    xs = grid.x[: solver.Nx, None] + sub[None, :]
    ys = grid.y[: solver.Ny, None] + sub[None, :]
    inside = xs[:, None, :, None] ** 2 + ys[None, :, None, :] ** 2 < radius**2
    eps_r = 1 + inside.mean(axis=(2, 3)) * (n_disk**2 - 1)
    for k in range(solver.Nz):
        solver.ieps[:, :, k, "z"] = 1 / (epsilon_0 * eps_r)

    # Source and probe on the x axis excite only the cos(m phi) modes
    src_time = GaussianSource(frequency=1.0e9, fwidth=0.15e9)
    line = CurrentSource(
        src_time, component="z", center=(0.08, 0.0, 0.0), size=(0.0, 0.0, 10 * dx)
    )
    probe = ResonanceMonitor("Ez", center=(0.085, 0.0, 0.0), t_start=src_time.end_time)
    Nt = int((src_time.end_time + T) / solver.dt)
    solver.emsolve(Nt, source=line, monitors=[probe])
    return probe


def test_disk_radiative_q(use_gpu):
    """
    Radiation-limited TM_4 whispering-gallery mode of a dielectric disk
    (n = 3, R = 100 mm, Ez polarization, 2D): exact f = 0.9902 GHz, Q = 164.2
    from the Bessel/Hankel matching condition. With subcell eps averaging the
    errors of f and Q fall at second order with the mesh (16-33 cells per
    wavelength in the disk): f -0.34 %, -0.13 %, -0.08 % and Q +2.6 %, +0.85 %,
    +0.5 % at 6, 4, 3 mm. The record (60 ns) is about one 1/e decay time.
    """
    radius = 0.1
    kR = _disk_resonance(4, 3.0, 2.075 + 0.006j)
    F_exact = kR * c / (2 * np.pi * radius)
    Q_exact = F_exact.real / (2 * F_exact.imag)

    f_err, q_err = [], []
    for dx in (6e-3, 4e-3, 3e-3):
        res = _disk_ringdown(dx, 60e-9, use_gpu).harminv(0.95e9, 1.03e9)
        assert len(res) == 1
        f_err.append(res.frequency[0] / F_exact.real - 1)
        q_err.append(res.Q[0] / Q_exact - 1)
    f_err, q_err = np.abs(f_err), np.abs(q_err)

    assert f_err[-1] < 1.5e-3 and q_err[-1] < 0.01
    assert np.all(np.diff(f_err) < 0) and np.all(np.diff(q_err) < 0)
    # Second order: (6/3)^2 = 4
    assert f_err[0] / f_err[-1] > 3 and q_err[0] / q_err[-1] > 3


def test_disk_q_lorentzian(use_gpu):
    """
    Independent check of harminv on a physical signal: the power spectrum of a
    long ring-down (400 ns, ~8 decay times) of the disk mode is a Lorentzian of
    half width f/(2Q). Its fitted Q agrees with harminv on the first 60 ns to
    0.2 % (the fit is biased by the tail of a low-Q neighbour, Q = 14).
    """
    probe = _disk_ringdown(6e-3, 400e-9, use_gpu)
    t, x = probe.signal()

    res_short = harminv(x[t < t[0] + 60e-9], probe.dt, 0.95e9, 1.03e9, t0=t[0])
    assert len(res_short) == 1

    f = np.linspace(0.95e9, 1.03e9, 401)
    spectrum = np.abs(np.exp(-2j * np.pi * np.outer(f, t)) @ x) ** 2
    spectrum /= spectrum.max()

    def lorentzian(f_mhz, A, f0, hw, B):
        return A / ((f_mhz - f0) ** 2 + hw**2) + B

    f_mhz = (f - 0.95e9) / 1e6
    i0 = np.argmax(spectrum)
    p0 = [9.0, f_mhz[i0], 3.0, 0.0]
    (_, f0, hw, _), _ = curve_fit(lorentzian, f_mhz, spectrum, p0=p0)
    f0 = 0.95e9 + 1e6 * f0

    np.testing.assert_allclose(f0, res_short.frequency[0], rtol=2e-4)
    np.testing.assert_allclose(f0 / (2e6 * abs(hw)), res_short.Q[0], rtol=2e-3)
