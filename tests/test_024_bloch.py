from functools import lru_cache

import numpy as np
import pytest
from scipy.constants import c, epsilon_0
from scipy.optimize import brentq
from scipy.signal import find_peaks
from scipy.signal.windows import blackmanharris

import wakis
from wakis.bloch import incidence_angle, k_parallel
from wakis.sources import CurrentSource, GaussianSource

# ---------------------------------------------------------------------------
# Bloch phase


def _bloch_cell(n_cells, k_bloch, dtype, bc="bloch", use_gpu=False):
    """Run n_cells unit cells (period 0.1 m along x) of a homogeneous Bloch
    lattice, PEC in z, with an x/z point dipole pair in every unit cell whose
    amplitude follows the Bloch phase exp(1j kx n L)."""
    period = 0.1
    grid = wakis.GridFIT3D(
        0, n_cells * period, 0, 0.08, 0, 0.1, 8 * n_cells, 6, 7, verbose=0
    )
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=[bc, bc, "pec"],
        bc_high=[bc, bc, "pec"],
        k_bloch=k_bloch,
        dtype=dtype,
        use_gpu=use_gpu,
        verbose=0,
    )
    kx = 0.0 if k_bloch is None else k_bloch[0]
    src_time = GaussianSource(frequency=2e9, fwidth=1e9)
    sources = [
        CurrentSource(
            src_time,
            component=comp,
            center=(0.031 + n * period, 0.03, 0.05),
            amplitude=np.exp(1j * kx * n * period) if kx else 1.0,
        )
        for n in range(n_cells)
        for comp in "xz"
    ]
    solver.emsolve(60, source=sources)
    return solver


@pytest.mark.parametrize("dtype, rtol", [(np.complex128, 1e-12), (np.complex64, 1e-4)])
def test_bloch_phase(dtype, rtol, use_gpu):
    """
    A Bloch cell of length L and a two-cell supercell of length 2L (Bloch
    phase exp(2i kx L)) driven by Bloch-phased sources must give identical
    fields, with F(x + L) = F(x) exp(i kx L) between the two halves of the
    supercell. This is an exact identity of the discrete scheme: it checks the
    phase of the forward (E -> H) wrap-around coupling and the conjugate phase
    of the dual (H -> E) coupling, so it holds to round-off.
    """
    k_bloch = (17.0, -9.0, 0.0)
    cell = _bloch_cell(1, k_bloch, dtype, use_gpu=use_gpu)
    supercell = _bloch_cell(2, k_bloch, dtype, use_gpu=use_gpu)
    phase = np.exp(1j * k_bloch[0] * 0.1)
    for field in ("E", "H"):
        for comp in "xyz":
            F = getattr(cell, field).to_matrix(comp)
            F2 = getattr(supercell, field).to_matrix(comp)
            F, F2 = (F.get(), F2.get()) if use_gpu else (F, F2)
            scale = np.abs(F).max()
            np.testing.assert_allclose(F2[:8], F, rtol=0, atol=rtol * scale)
            np.testing.assert_allclose(F2[8:], F * phase, rtol=0, atol=rtol * scale)


def test_bloch_zero_k_is_periodic():
    """With k_bloch = 0, complex Bloch fields reproduce the real periodic run."""
    real = _bloch_cell(1, None, np.float64, bc="periodic")
    bloch = _bloch_cell(1, (0.0, 0.0, 0.0), np.complex128)
    E, Eb = real.E.toarray(), bloch.E.toarray()
    np.testing.assert_allclose(Eb, E, rtol=0, atol=1e-12 * np.abs(E).max())


def test_bloch_input_validation():
    """Bloch phases need complex fields and paired faces on the same axis."""
    grid = wakis.GridFIT3D(0, 0.1, 0, 0.1, 0, 0.1, 4, 4, 4, verbose=0)
    bloch = ["bloch", "periodic", "pec"]
    with pytest.raises(ValueError, match="complex"):
        wakis.SolverFIT3D(grid, bc_low=bloch, bc_high=bloch, k_bloch=(1, 0, 0))
    with pytest.raises(ValueError, match="paired"):
        wakis.SolverFIT3D(
            grid,
            bc_low=bloch,
            bc_high=["periodic"] * 2 + ["pec"],
            k_bloch=(1, 0, 0),
            dtype=np.complex128,
        )
    with pytest.raises(ValueError, match="not 'bloch'"):
        wakis.SolverFIT3D(
            grid, bc_low=bloch, bc_high=bloch, k_bloch=(1, 1, 0), dtype=np.complex128
        )


# ---------------------------------------------------------------------------
# Oblique incidence

DZ = 2.5e-3  # 120 cells per vacuum wavelength, >= 44 per wavelength at n = 2
F0 = 1e9
K_PAR = k_parallel(F0, np.radians(30.0))  # 30 degrees at 1 GHz
FREQS = np.array([0.85, 1.0, 1.2, 1.35]) * F0  # 36 to 22 degrees
N2 = 2.0


@lru_cache(maxsize=None)
def _oblique_run(pol, eps_r=None, use_gpu=False):
    """
    Bloch cell one cell wide in x and y (the layered problem is invariant
    along x), CPML in z, sheet current at z = 0.25 m with the Bloch phase
    profile exp(1j kx x), optional dielectric half-space z < 0. Returns the
    flux through z = -0.15 m and Ey (s) or Ex (p) on a line in z.
    """
    grid = wakis.GridFIT3D(
        0, DZ, 0, DZ, -0.6, 0.6, 1, 1, int(round(1.2 / DZ)), verbose=0
    )
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["bloch", "periodic", "cpml"],
        bc_high=["bloch", "periodic", "cpml"],
        k_bloch=K_PAR,
        dtype=np.complex128,
        n_pml=20,
        use_gpu=use_gpu,
        verbose=0,
    )
    if eps_r is not None:
        # Tangential E on z nodes, arithmetic mean on the interface node;
        # Ez on z midpoints lies entirely on one side.
        z_node, tol = grid.z[:-1], 1e-6 * DZ
        for comp in "xy":
            solver.ieps[:, :, z_node < -tol, comp] = 1 / (epsilon_0 * eps_r)
            solver.ieps[:, :, np.abs(z_node) < tol, comp] = 2 / (
                epsilon_0 * (1 + eps_r)
            )
        solver.ieps[:, :, solver.z < 0, "z"] = 1 / (epsilon_0 * eps_r)

    # Narrow band well above the cutoff c|k_par|/2pi = 0.5 GHz: components near
    # grazing incidence have vanishing group velocity along z and would not
    # leave the domain within the run.
    src_time = GaussianSource(frequency=1.1e9, fwidth=0.15e9)
    comp = "y" if pol == "s" else "x"
    sheet = CurrentSource(
        src_time,
        component=comp,
        center=(DZ / 2, DZ / 2, 0.25),
        size=(DZ, DZ, 0.0),
        amp_func=lambda x, y, z: np.exp(1j * (K_PAR[0] * x + K_PAR[1] * y)),
    )
    plane = wakis.FluxMonitor(FREQS, center=(DZ / 2, DZ / 2, -0.15), size=(DZ, DZ, 0))
    line = wakis.DFTMonitor(
        FREQS, components=["E" + comp], center=(0, 0, -0.1), size=(0, 0, 0.6)
    )
    Nt = int((src_time.end_time + 15e-9) / solver.dt)
    solver.emsolve(Nt, source=sheet, monitors=[plane, line])
    field, coords = line.get_field("E" + comp)
    return plane.get_flux(), field.reshape(FREQS.size, -1), coords[2], solver.dt


def test_oblique_plane_wave_angle(use_gpu):
    """
    A Bloch-phased sheet radiates a plane wave with tangential wavevector
    k_par. Its normal wavenumber, from the phase slope of the field along z,
    obeys the discrete dispersion relation of the scheme with the Bloch
    derivative (exp(i kx dx) - 1)/dx, and hence propagates at the angle
    theta(f) = arcsin(c k_par / 2 pi f). With the exp(+2i pi f t) phasor of
    the DFT monitors the wave travelling towards -z varies as exp(+i kz z).
    """
    _, field, z, dt = _oblique_run("s", use_gpu=use_gpu)
    for f, Ey in zip(FREQS, field):
        kz = np.polyfit(z, np.unwrap(np.angle(Ey)), 1)[0]
        # (sin(w dt/2)/c dt)^2 = sum_i (sin(k_i d_i/2)/d_i)^2 with dx = dz
        s = (np.sin(np.pi * f * dt) / (c * dt)) ** 2 - (
            np.sin(K_PAR[0] * DZ / 2) / DZ
        ) ** 2
        kz_discrete = 2 / DZ * np.arcsin(DZ * np.sqrt(s))
        np.testing.assert_allclose(kz, kz_discrete, rtol=1e-3)
        theta = np.arctan2(K_PAR[0], kz)
        assert abs(np.degrees(theta - incidence_angle(K_PAR, f))) < 0.1


@pytest.mark.parametrize("pol", ["s", "p"])
def test_oblique_fresnel(pol, use_gpu):
    """
    Power reflectance of a dielectric half-space (n = 2) at oblique incidence,
    R = 1 - T with T the transmitted z-flux normalized to a vacuum reference,
    against the Fresnel equations. At fixed k_par the angle of incidence
    changes with frequency (36 to 22 degrees). Interface and grid dispersion
    errors are second order in dz: < 0.6% at >= 44 cells per wavelength.
    """
    ref, *_ = _oblique_run(pol, use_gpu=use_gpu)
    transmitted, *_ = _oblique_run(pol, eps_r=N2**2, use_gpu=use_gpu)
    R = 1 - transmitted / ref

    cos_i = np.cos(incidence_angle(K_PAR, FREQS))
    cos_t = np.cos(incidence_angle(K_PAR, FREQS, n=N2))
    if pol == "s":
        r = (cos_i - N2 * cos_t) / (cos_i + N2 * cos_t)
    else:
        r = (N2 * cos_i - cos_t) / (N2 * cos_i + cos_t)
    np.testing.assert_allclose(R, r**2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Photonic crystal bands


@pytest.mark.parametrize("k_norm", [0.4, 0.9])
def test_photonic_crystal_bands(k_norm):
    """
    Bands of a 1D photonic crystal (period 0.1 m, half-period layers of
    n = 1.5 and 3) at Bloch wavevector k = k_norm * pi / period, propagating
    normal to the layers. Mode frequencies from the spectrum of a broadband
    pulse (low-sidelobe window, 100 ns) match the roots of the Yeh-Yariv-Hong
    dispersion relation cos(k L) = cos(k1 d1) cos(k2 d2)
    - (n1/n2 + n2/n1)/2 sin(k1 d1) sin(k2 d2). Bands below 2.5 GHz have >= 40
    cells per wavelength in the high-index layer: grid dispersion < 0.2%.
    """
    period, Nx, n1, n2 = 0.1, 80, 1.5, 3.0
    d1 = d2 = period / 2
    k = k_norm * np.pi / period
    grid = wakis.GridFIT3D(0, period, 0, 1.0, 0, 1.0, Nx, 1, 1, verbose=0)
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["bloch", "periodic", "periodic"],
        bc_high=["bloch", "periodic", "periodic"],
        k_bloch=(k, 0, 0),
        dtype=np.complex128,
        verbose=0,
    )
    # Ey sits on x nodes; interface nodes take the mean permittivity
    x, tol = grid.x[:-1], 1e-6 * period / Nx
    eps_r = np.where(x < d1 - tol, n1**2, n2**2)
    eps_r[(np.abs(x) < tol) | (np.abs(x - d1) < tol)] = (n1**2 + n2**2) / 2
    solver.ieps[:, 0, 0, "y"] = 1 / (epsilon_0 * eps_r)
    solver.update_tensors("ieps")
    solver.step_0 = True

    src_time = GaussianSource(frequency=1.5e9, fwidth=1e9)
    src = CurrentSource(src_time, component="y", center=(0.37 * period, 0.5, 0.5))
    probes = [int(p * Nx) for p in (0.13, 0.41, 0.77)]
    trace = []
    solver.emsolve(
        int(100e-9 / solver.dt),
        source=src,
        callback=lambda s, t: trace.append(s.E[probes, 0, 0, "y"].copy()),
    )
    signal = np.array(trace)[int(src_time.end_time / solver.dt) + 1 :]
    nfft = 16 * len(signal)
    window = blackmanharris(len(signal))[:, None]
    spectrum = np.abs(np.fft.fft(signal * window, n=nfft, axis=0)).sum(axis=1)
    freqs = np.fft.fftfreq(nfft, solver.dt)
    spectrum, freqs = spectrum[freqs > 0], freqs[freqs > 0]
    peaks, _ = find_peaks(spectrum, height=1e-3 * spectrum.max())
    measured = freqs[peaks][freqs[peaks] < 2.5e9]

    def dispersion(f):
        k1, k2 = 2 * np.pi * f * n1 / c, 2 * np.pi * f * n2 / c
        return (
            np.cos(k1 * d1) * np.cos(k2 * d2)
            - 0.5 * (n1 / n2 + n2 / n1) * np.sin(k1 * d1) * np.sin(k2 * d2)
            - np.cos(k * period)
        )

    f = np.linspace(1e6, 2.5e9, 2500)
    sign_change = np.flatnonzero(dispersion(f[:-1]) * dispersion(f[1:]) < 0)
    analytic = [brentq(dispersion, f[i], f[i + 1]) for i in sign_change]
    assert len(measured) == len(analytic) == 4
    np.testing.assert_allclose(measured, analytic, rtol=5e-3)
