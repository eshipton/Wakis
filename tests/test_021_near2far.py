import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import c, epsilon_0, mu_0

import wakis
from wakis.near2far import (
    axial_ratio,
    half_power_beamwidth,
    ludwig3,
    plot_pattern_3d,
    plot_pattern_cut,
)
from wakis.sources import CurrentSource, GaussianSource

Z0 = np.sqrt(mu_0 / epsilon_0)
DX = 0.0125  # 24 cells per wavelength at 1 GHz
F = np.linspace(0.8e9, 1.2e9, 3)
SRC_TIME = GaussianSource(frequency=1e9, fwidth=0.3e9)


def _run(grid, bc_low, component, center, monitors, extra_time=8e-9, use_gpu=False):
    """Point current moment in a CPML-truncated domain; returns the source work
    -Re(E(f) Il(f)^*) and the actual dipole position."""
    solver = wakis.SolverFIT3D(
        grid, use_gpu=use_gpu, bc_low=bc_low, bc_high=["cpml"] * 3, n_pml=10, verbose=0
    )
    dipole = CurrentSource(SRC_TIME, component=component, center=center)
    point = wakis.DFTMonitor(F, components=["E" + component], center=center)
    Nt = int((SRC_TIME.end_time + extra_time) / solver.dt)
    solver.emsolve(Nt, source=dipole, monitors=[point, *monitors])
    E, coords = point.get_field("E" + component)
    S = SRC_TIME.fourier_transform(F)
    return -np.real(E.ravel() * np.conj(S)), np.array([x[0] for x in coords])


def _free_space_power():
    """Hertzian dipole power eta k^2 |Il|^2 / (6 pi), flux normalization."""
    k = 2 * np.pi * F / c
    return Z0 * k**2 / (6 * np.pi) * np.abs(SRC_TIME.fourier_transform(F)) ** 2


def _ground_power_ratio(u, component):
    """
    P(h) / P_free of a dipole at height h over an infinite PEC plane, u = 2kh.
    The image (same moment for a vertical dipole, opposite for a horizontal
    one) adds its field at the source to the self field (mutual resistance).
    """
    if component == "z":
        return 1 + 3 * (np.sin(u) / u**3 - np.cos(u) / u**2)
    return 1 - 1.5 * (np.sin(u) / u + np.cos(u) / u**2 - np.sin(u) / u**3)


def test_dipole_far_field(use_gpu):
    """
    Hertzian dipole in free space: the far field is E_theta = j eta k Il
    sin(theta) exp(-jkR) / (4 pi R) with no E_phi, so D = 1.5 sin^2(theta),
    D_max = 1.5 (1.76 dBi) and the half-power beamwidth is 90 deg. The power
    integrated over the far-field sphere equals the Poynting flux through the
    equivalence box (energy conservation). 20-30 cells per wavelength.
    """
    grid = wakis.GridFIT3D(-0.25, 0.25, -0.25, 0.25, -0.25, 0.25, 40, 40, 40, verbose=0)
    n2f = wakis.Near2FarMonitor(F, size=(12 * DX,) * 3)
    flux = wakis.FluxMonitor(F, size=(12 * DX,) * 3)
    work, _ = _run(grid, ["cpml"] * 3, "z", (0.0, 0.0, 0.0), [n2f, flux], 6e-9, use_gpu)

    theta = np.linspace(0, np.pi, 91)[None, :]
    phi = np.array([0.0, 0.7, 2.0])[:, None]
    ff = n2f.get_farfield(theta, phi)
    pattern = np.abs(ff["Etheta"]) / np.abs(ff["Etheta"]).max(
        axis=(1, 2), keepdims=True
    )
    # Phase error of the discrete waves across the 0.5-wavelength box: < 1%
    assert np.abs(pattern - np.sin(theta)).max() < 0.02
    assert np.abs(ff["Ephi"]).max() < 1e-2 * np.abs(ff["Etheta"]).max()

    S = SRC_TIME.fourier_transform(F)
    k = 2 * np.pi * F / c
    E_far = n2f.get_farfield(np.pi / 2, 0.0, r=10.0)["Etheta"].ravel()
    np.testing.assert_allclose(
        np.abs(E_far), Z0 * k * np.abs(S) / (4 * np.pi * 10.0), rtol=0.02
    )

    P = n2f.radiated_power()
    np.testing.assert_allclose(P, flux.get_flux(), rtol=0.01)
    np.testing.assert_allclose(n2f.get_flux(), flux.get_flux(), rtol=1e-12)
    np.testing.assert_allclose(n2f.directivity(np.pi / 2, 0.0).ravel(), 1.5, rtol=0.02)
    D_max, direction = n2f.max_directivity()
    np.testing.assert_allclose(D_max, 1.5, rtol=0.02)
    assert np.all(np.abs(direction[:, 2]) < 0.1)
    # Lossless antenna: gain equals directivity, efficiency is 1
    np.testing.assert_allclose(n2f.radiation_efficiency(work), 1.0, rtol=0.01)
    np.testing.assert_allclose(
        n2f.gain(np.pi / 2, 0.0, work), n2f.directivity(np.pi / 2, 0.0), rtol=0.01
    )

    cut = n2f.directivity(theta[0], 0.0)[1]
    assert abs(np.degrees(half_power_beamwidth(theta[0], cut)) - 90.0) < 1.0

    fig, _ = plot_pattern_cut(theta[0], cut, label="1 GHz")
    plt.close(fig)
    th, ph = np.meshgrid(np.linspace(0, np.pi, 19), np.linspace(0, 2 * np.pi, 37))
    fig, _ = plot_pattern_3d(th, ph, n2f.directivity(th, ph)[1])
    plt.close(fig)


@pytest.mark.parametrize("component", ["z", "x"])
def test_dipole_above_ground_plane(component, use_gpu):
    """
    Dipole at height h above a PEC ground plane (the z-low domain boundary).
    By image theory the far field in the upper half-space is the free-space
    dipole pattern times the array factor 2 cos(k h cos(theta)) (vertical) or
    2 sin(k h cos(theta)) (horizontal), and the radiated power ratio follows
    the closed form of `_ground_power_ratio`. The power on the far-field
    hemisphere equals the work done by the source.
    """
    h = 4.5 * DX if component == "z" else 5 * DX  # Jz on edge midpoints
    grid = wakis.GridFIT3D(
        -0.25, 0.25, -0.25, 0.25, 0.0, 25 * DX, 40, 40, 25, verbose=0
    )
    n2f = wakis.Near2FarMonitor(
        F, center=(0, 0, 5 * DX), size=(12 * DX, 12 * DX, 10 * DX), ground_plane="-z"
    )
    bc_low = ["cpml", "cpml", "pec"]
    work, pos = _run(grid, bc_low, component, (0, 0, h), [n2f], use_gpu=use_gpu)
    assert np.isclose(pos[2], h)

    k = 2 * np.pi * F[:, None, None] / c
    theta = np.linspace(0, np.pi / 2, 46)[None, :]
    phi = np.array([0.0, 0.7, np.pi / 2])[:, None]
    ff = n2f.get_farfield(theta, phi)
    pattern = np.sqrt(np.abs(ff["Etheta"]) ** 2 + np.abs(ff["Ephi"]) ** 2)
    if component == "z":
        ref = np.sin(theta) * np.abs(2 * np.cos(k * h * np.cos(theta)))
    else:
        element = np.sqrt(1 - np.sin(theta) ** 2 * np.cos(phi) ** 2)
        array_factor = np.abs(2 * np.sin(k * h * np.cos(theta)))
        ref = element * array_factor
        array_factor /= ref.max(axis=(1, 2), keepdims=True)
    pattern /= pattern.max(axis=(1, 2), keepdims=True)
    ref /= ref.max(axis=(1, 2), keepdims=True)
    assert np.abs(pattern - ref).max() < 0.02

    P = n2f.radiated_power()
    np.testing.assert_allclose(P, work, rtol=0.01)
    u = 2 * k.ravel() * h
    # Free-space power from the closed form: grid dispersion ~1% at 20 cells/wl
    np.testing.assert_allclose(
        P / _free_space_power(), _ground_power_ratio(u, component), rtol=0.025
    )

    if component == "x":
        # Ludwig-3 components of an x dipole: co = cos(theta) cos^2(phi) +
        # sin^2(phi), cross = -(1 - cos(theta)) sin(phi) cos(phi), times AF
        co, cross = ludwig3(ff["Etheta"], ff["Ephi"], phi, reference="x")
        ct, cp, sp = np.cos(theta), np.cos(phi), np.sin(phi)
        scale = np.sqrt(np.abs(co) ** 2 + np.abs(cross) ** 2).max(axis=(1, 2))
        co_ref = (ct * cp**2 + sp**2) * array_factor
        cross_ref = (1 - ct) * sp * cp * array_factor
        assert np.abs(np.abs(co) / scale[:, None, None] - co_ref).max() < 0.02
        assert np.abs(np.abs(cross) / scale[:, None, None] - cross_ref).max() < 0.02
        ar, _, _ = axial_ratio(ff["Etheta"], ff["Ephi"])
        assert np.all(ar[pattern > 0.1] > 100)  # linear polarization


def test_ldos_ground_plane(use_gpu):
    """
    Local density of states near a PEC plane without near-to-far-field: the
    work -Re(E Il^*) done by a dipole at height h, relative to free space, is
    the image-theory ratio P(h)/P_free for both orientations and several
    heights (u = 2kh from 0.6 to 6). The free-space reference is simulated on
    the same mesh, cancelling the grid dispersion of the self term.
    """
    grid = wakis.GridFIT3D(-0.2, 0.2, -0.2, 0.2, -0.2, 0.2, 32, 32, 32, verbose=0)
    free, _ = _run(grid, ["cpml"] * 3, "z", (0, 0, 0), [], 6e-9, use_gpu)

    grid = wakis.GridFIT3D(-0.2, 0.2, -0.2, 0.2, 0.0, 22 * DX, 32, 32, 22, verbose=0)
    bc_low = ["cpml", "cpml", "pec"]
    k = 2 * np.pi * F / c
    for component, heights in (("z", (1.5, 4.5, 9.5)), ("x", (2, 5, 9))):
        for h in heights:
            work, pos = _run(grid, bc_low, component, (0, 0, h * DX), [], 8e-9, use_gpu)
            np.testing.assert_allclose(
                work / free,
                _ground_power_ratio(2 * k * pos[2], component),
                atol=0.02,
                err_msg=f"{component} dipole at h = {pos[2]} m",
            )


@pytest.mark.parametrize("component", ["z", "x"])
def test_purcell_parallel_plates(component, use_gpu):
    """
    Purcell factor of a dipole midway between two PEC plates a gap d apart.
    Summing the image series with the Fourier series of the Bernoulli
    polynomials gives P/P_free = 3 pi / (2 k d) for a perpendicular dipole
    (TEM mode only, 0 < kd < 2 pi) and 3 pi / (2 k d) (1 + (pi / k d)^2) for a
    parallel dipole (TE1 + TM1, pi < kd < 3 pi; zero below cutoff). Tested
    at kd / pi = 1.15 to 1.75, away from the cutoff steps, at >= 20 cells per
    wavelength. Note: the PEC on the high-z boundary lies on the last
    non-ghost node, one cell below zmax.
    """
    n_gap = 21 if component == "z" else 20  # dipole edge on the midplane
    d = n_gap * DX
    grid = wakis.GridFIT3D(
        -0.25, 0.25, -0.25, 0.25, 0.0, d + DX, 40, 40, n_gap + 1, verbose=0
    )
    solver = wakis.SolverFIT3D(
        grid,
        use_gpu=use_gpu,
        bc_low=["cpml", "cpml", "pec"],
        bc_high=["cpml", "cpml", "pec"],
        n_pml=10,
        verbose=0,
    )
    kd = np.array([1.15, 1.45, 1.75]) * np.pi
    f = kd * c / (2 * np.pi * d)
    src_time = GaussianSource(frequency=f.mean(), fwidth=0.3e9)
    dipole = CurrentSource(src_time, component=component, center=(0, 0, d / 2))
    point = wakis.DFTMonitor(f, components=["E" + component], center=(0, 0, d / 2))
    # Long ring-down: the guided modes near cutoff have a small group velocity
    Nt = int((src_time.end_time + 20e-9) / solver.dt)
    solver.emsolve(Nt, source=dipole, monitors=[point])
    assert np.isclose(point.get_field("E" + component)[1][2][0], d / 2)

    S = src_time.fourier_transform(f)
    work = -np.real(point.get_field("E" + component)[0].ravel() * np.conj(S))
    k = 2 * np.pi * f / c
    purcell = work / (Z0 * k**2 / (6 * np.pi) * np.abs(S) ** 2)
    expected = 3 * np.pi / (2 * kd)
    if component == "x":
        expected *= 1 + (np.pi / kd) ** 2
    # Dispersion shifts the TE1/TM1 cutoff, which matters most near kd = pi
    np.testing.assert_allclose(purcell, expected, atol=0.05)


def test_polarization_helpers():
    """
    Circular and linear polarization on analytic fields: E_theta = 1,
    E_phi = -j is right-hand circular (exp(+jwt), IEEE), with axial ratio 1;
    a linear field has infinite axial ratio; Ludwig-3 components of an x
    polarized boresight field are (1, 0) for any phi.
    """
    ar, E_R, E_L = axial_ratio(np.array([1.0]), np.array([-1j]))
    np.testing.assert_allclose(ar, 1.0)
    np.testing.assert_allclose(np.abs(E_L), 0.0, atol=1e-15)
    ar, _, _ = axial_ratio(np.array([1.0]), np.array([0.5]))
    assert np.isinf(ar[0])

    phi = np.linspace(0, 2 * np.pi, 7)
    co, cross = ludwig3(np.cos(phi), -np.sin(phi), phi, reference="x")
    np.testing.assert_allclose(co, 1.0)
    np.testing.assert_allclose(cross, 0.0, atol=1e-15)
    co, cross = ludwig3(np.cos(phi), -np.sin(phi), phi, reference="y")
    np.testing.assert_allclose(co, 0.0, atol=1e-15)
