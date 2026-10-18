import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.constants import c  # noqa: E402

from wakis.templates import (  # noqa: E402
    Box,
    Simulation,
    analysis,
    dielectric_slab_cell,
    dipole_antenna,
    dipole_results,
    formulas,
    meshing,
    optimize,
    parameter_sweep,
    patch_antenna,
    patch_array_cell,
    patch_results,
    rectangular_waveguide,
    reflection_transmission,
    slab_results,
    slab_waveguide,
    waveguide_s_parameters,
)
from wakis.templates.shapes import material_eps, resolve_material  # noqa: E402


# ----------------------------------------------------------------- setup rules
def test_automatic_setup():
    """
    Automatic setup of a dielectric block in vacuum: the mesh has nodes on all
    faces, at least `cells_per_wavelength` cells per wavelength in each
    material at f_max, adjacent cells grow by about the grading ratio, the
    structure is a quarter wavelength (f_min) from the CPML, and the CPML has
    the documented thickness.
    """
    f_min, f_max, eps_r = 1e9, 3e9, 4.0
    block = Box((0, 0, 0), (0.05, 0.04, 0.01), eps_r)
    sim = Simulation(f_min, f_max, [block], cells_per_wavelength=20)
    sim.build()
    lam_min = c / f_max
    for axis, nodes in enumerate((sim.grid.x, sim.grid.y, sim.grid.z)):
        cells = np.diff(nodes)
        for face in block.faces(axis):
            assert np.min(np.abs(nodes - face)) < 1e-12
        inside = np.abs(nodes[:-1] + cells / 2) < block.size[axis] / 2
        assert cells[inside].max() <= lam_min / np.sqrt(eps_r) / 20 * (1 + 1e-9)
        assert cells.max() <= lam_min / 20 * (1 + 1e-9)
        # Cell counts are rounded up per interval: ratio slightly above 1.4
        assert np.max(cells[1:] / cells[:-1]) < 1.5
        assert np.max(cells[:-1] / cells[1:]) < 1.5
        # CPML interface (n_pml cells in) to structure distance
        gap = (block.center[axis] - block.size[axis] / 2) - nodes[sim.n_pml]
        assert gap >= meshing.padding(f_min) * (1 - 1e-9)
    assert sim.n_pml == meshing.pml_cells(f_min, lam_min / 20) == 10
    assert meshing.pml_cells(f_min / 3, lam_min / 20) == 23


def test_design_formulas():
    """
    Closed-form references: the induced-EMF impedance of a half-wave dipole
    is 73.1 + 42.5j Ohm [Balanis]; the Airy slab formulas conserve energy and
    match T = 1/(1 + (n - 1/n)^2 sin^2(k n d)/4); a TLM patch designed for f0
    resonates at f0 with the cos^2 feed law giving Z0 at the feed offset and a
    two-slot directivity of 5-7 dBi; the TE slab-guide index lies between the
    cladding and core indices.
    """
    np.testing.assert_allclose(
        formulas.dipole_impedance(0.5, 1e-6, c), 73.08 + 42.51j, atol=0.02
    )

    f = np.linspace(1e9, 3e9, 7)
    R, T = formulas.slab_reflection_transmission(4.0, 0.05, f)
    np.testing.assert_allclose(R + T, 1.0, atol=1e-12)
    T_ref = 1 / (1 + 0.25 * 1.5**2 * np.sin(2 * np.pi * f / c * 2 * 0.05) ** 2)
    np.testing.assert_allclose(T, T_ref, rtol=1e-12)

    f0, eps_r, h = 2.45e9, 4.4, 1.6e-3
    design = formulas.patch_design(f0, eps_r, h)
    L, W = design["length"], design["width"]
    np.testing.assert_allclose(formulas.patch_resonance(L, W, eps_r, h), f0)
    R0 = formulas.patch_edge_resistance(L, W, f0)
    y0 = formulas.patch_feed_offset(L, R0, 50.0)
    np.testing.assert_allclose(R0 * np.cos(np.pi * y0 / L) ** 2, 50.0)
    D_dbi = 10 * np.log10(formulas.patch_directivity(L, W, f0))
    assert 5.0 < D_dbi < 7.0

    neff = formulas.slab_waveguide_neff(3.48, 1.444, 220e-9, 1550e-9)
    assert 1.444 < neff < 3.48


# -------------------------------------------------------------- periodic cells
def test_dielectric_slab_cell(use_gpu):
    """
    Normal-incidence reflectance and transmittance of a lossless dielectric
    slab (eps_r = 4, d = 10 mm) in a periodic unit cell, 2-12 GHz, against
    the Airy formulas. The z mesh is graded (21 cells per wavelength in the
    slab, 20 in air at 12 GHz); with the dual-length correction for graded
    meshes the error is the FIT phase error (< 2 %); without it, 6 %.
    """
    sim = dielectric_slab_cell(eps_r=4.0, thickness=10e-3, use_gpu=use_gpu)
    sim.run(reference=True)
    R, T = reflection_transmission(sim)
    R_ref, T_ref = formulas.slab_reflection_transmission(4.0, 10e-3, sim.frequencies)
    np.testing.assert_allclose(T, T_ref, atol=0.025)
    np.testing.assert_allclose(R, R_ref, atol=0.025)
    np.testing.assert_allclose(R + T, 1.0, atol=0.01)


def test_patch_array_cell(use_gpu):
    """
    Square PEC patch array (period 15 mm, patches 12 mm, zero thickness) at
    normal incidence: a capacitive surface, transparent at low frequency and
    increasingly reflecting up to total reflection just below the grating
    threshold c/p (qualitative check), lossless (R + T = 1).
    """
    sim = patch_array_cell(use_gpu=use_gpu)
    # Near the grating threshold the fields decay slowly: wait for -50 dB
    sim.run(reference=True, energy_decay=1e-5)
    R, T = reflection_transmission(sim)
    f = sim.frequencies
    assert R[0] < 0.3
    assert np.all(np.diff(R) > -1e-3)
    i = np.argmax(R)
    assert R[i] > 0.99
    assert 0.9 < f[i] / sim.design["grating_threshold"] < 1.0
    np.testing.assert_allclose(R + T, 1.0, atol=5e-3)


# ------------------------------------------------------------------ waveguides
def test_rectangular_waveguide(use_gpu):
    """
    WR-90 section with TE10 mode ports (20 cells across a), 1.25-1.9 fc: the
    empty guide transmits fully, |S21| = 1 within 1.5 % (largest near
    1.25 fc, where the source profile is furthest from its design frequency)
    and |S11| < 5e-3 (CPML and source leakage). The phase advance between
    the ports gives beta of the discrete TE10 dispersion relation
    (w_t/c)^2 = kx_t^2 + beta_t^2 to 1e-3 and the continuum
    sqrt(k^2 - (pi/a)^2) within grid dispersion (2 %, largest near cutoff).
    """
    sim = rectangular_waveguide(use_gpu=use_gpu)
    sim.run()
    S11, S21 = waveguide_s_parameters(sim)
    np.testing.assert_allclose(np.abs(S21), 1.0, atol=0.015)
    assert np.abs(S11).max() < 5e-3

    f, a = sim.frequencies, sim.design["a"]
    a1, _ = sim.monitor("port1").get_amplitudes()
    a2, _ = sim.monitor("port2").get_amplitudes()
    dz = sim.monitor("port2").plane.position - sim.monitor("port1").plane.position
    beta = -np.unwrap(np.angle(a2 / a1)) / dz
    beta_ref = formulas.te10_beta(a, f).real
    beta += 2 * np.pi / dz * np.round((beta_ref[0] - beta[0]) * dz / (2 * np.pi))
    np.testing.assert_allclose(beta, beta_ref, rtol=0.02)

    dt, dx, h = sim.solver.dt, sim.grid.dx[0], sim.grid.dz[0]
    w_t = 2 / dt * np.sin(np.pi * f * dt)
    kx_t = 2 / dx * np.sin(np.pi * dx / (2 * a))
    beta_t = np.sqrt((w_t / c) ** 2 - kx_t**2)
    np.testing.assert_allclose(beta, 2 / h * np.arcsin(beta_t * h / 2), rtol=1e-3)


def test_waveguide_iris(use_gpu):
    """
    Thin symmetric inductive iris (opening 12 mm in WR-90): the lossless
    junction conserves power, |S11|^2 + |S21|^2 = 1 within 2 %, and |S11|
    follows the shunt-susceptance model B/Y0 = -(lambda_g/a) cot^2(pi d/2a)
    within 0.06 (the model is a small-aperture approximation).
    """
    sim = rectangular_waveguide(iris_width=12e-3, use_gpu=use_gpu)
    sim.run()
    S11, S21 = waveguide_s_parameters(sim)
    np.testing.assert_allclose(np.abs(S11) ** 2 + np.abs(S21) ** 2, 1.0, atol=0.02)
    s11_model = formulas.inductive_iris_s11(sim.design["a"], 12e-3, sim.frequencies)
    np.testing.assert_allclose(np.abs(S11), np.abs(s11_model), atol=0.06)


def test_silicon_slab_waveguide(use_gpu):
    """
    220 nm silicon slab (Sellmeier silicon, ADE) in SiO2 (n = 1.444) at
    1395-1705 nm, TE0 mode port: the straight guide transmits fully
    (|a_out/a_in|^2 = 1 within 0.5 %) and the effective index from the phase
    advance matches the dispersive slab eigenvalue equation within 0.3 % at
    11 cells across the core; the mode solver index at the band centre
    matches within 0.5 %.
    """
    sim = slab_waveguide(use_gpu=use_gpu)
    sim.run()
    result = slab_results(sim)
    np.testing.assert_allclose(result["transmission"], 1.0, atol=5e-3)

    f = sim.frequencies
    silicon = resolve_material("silicon")
    neff_ref = np.array(
        [
            formulas.slab_waveguide_neff(
                np.sqrt(material_eps(silicon, fk)), 1.444, 220e-9, c / fk
            )
            for fk in f
        ]
    )
    np.testing.assert_allclose(result["neff"], neff_ref, rtol=3e-3)
    centre = f.size // 2
    np.testing.assert_allclose(
        result["neff_modes"][centre], neff_ref[centre], rtol=5e-3
    )


# -------------------------------------------------------------------- antennas
def _dipole_convergence(cells, use_gpu):
    """Resonant length L/lambda extrapolated to zero cell size (first order)."""
    results = []
    for n in cells:
        sim = dipole_antenna(frequency=1e9, cells_per_wavelength=n, use_gpu=use_gpu)
        sim.run()
        results.append((sim.grid.dz.min(), dipole_results(sim)))
    (h1, r1), (h2, r2) = results
    slope = (r1["length_wavelengths"] - r2["length_wavelengths"]) / (h1 - h2)
    return r2["length_wavelengths"] - slope * h2, [r1, r2]


def test_dipole_antenna(use_gpu):
    """
    Half-wave thin-wire dipole with a one-cell gap port (20 and 30 cells per
    wavelength at f_max = 1.3 GHz). The wire of zero radius has an effective
    radius and a gap proportional to the cell size, so the resonance length
    converges at first order; the Richardson extrapolation of the resonant
    length L/lambda lands in the textbook 0.46-0.485 range of thin dipoles.
    At resonance Zin is real and near 73 Ohm (60-80) and the directivity is
    1.64 (2.15 dBi) within 4 %.
    """
    extrapolated, results = _dipole_convergence((20, 30), use_gpu)
    assert 0.46 < extrapolated < 0.485
    assert results[1]["length_wavelengths"] < extrapolated  # from below
    for r in results:
        assert 60 < r["impedance"].real < 80
        np.testing.assert_allclose(r["directivity"], 1.64, rtol=0.04)


@pytest.mark.slow
def test_dipole_antenna_accuracy(use_gpu):
    """
    The same convergence study at 40 and 60 cells per wavelength: the
    extrapolated resonant length is 0.47 lambda within 0.01, the classical
    thin half-wave dipole value (0.47-0.48 lambda).
    """
    extrapolated, _ = _dipole_convergence((40, 60), use_gpu)
    np.testing.assert_allclose(extrapolated, 0.47, atol=0.01)


def test_patch_antenna_smoke(use_gpu):
    """
    Coarse probe-fed patch (eps_r = 2.2, h = 3.2 mm, 15 cells per wavelength,
    2 cells across the substrate) designed with the transmission-line model
    (patch length, width and cos^2 feed offset for 50 Ohm): the |S11|
    minimum lies within 5 % of the design frequency (the TLM is a few-percent
    approximation; observed -3 %) and is a matched dip (< -10 dB), and the
    maximum directivity is in the 6-9 dBi range of patches on a finite
    ground plane (TLM two-slot value 7.0 dBi; observed 8.0 dBi).
    """
    sim = patch_antenna(
        frequency=2.45e9,
        substrate=2.2,
        height=3.2e-3,
        cells_per_wavelength=15,
        substrate_cells=2,
        use_gpu=use_gpu,
    )
    sim.run(energy_decay=1e-3)
    result = patch_results(sim)
    assert abs(result["resonance_error"]) < 0.05
    assert result["s11_min_db"] < -10
    assert 6.0 < result["directivity_dbi"] < 9.0


@pytest.mark.slow
def test_patch_antenna_accuracy(use_gpu):
    """
    FR-4 patch (eps_r = 4.4, h = 1.6 mm) at 2.45 GHz on a 30-cell-per-
    wavelength mesh with 4 cells across the substrate: resonance within 5 %
    of the transmission-line model (the TLM itself is a few-percent
    approximation), |S11| < -10 dB at the design feed position and a
    directivity of 5-8 dBi.
    """
    sim = patch_antenna(cells_per_wavelength=30, substrate_cells=4, use_gpu=use_gpu)
    sim.run(energy_decay=1e-4)
    result = patch_results(sim)
    assert abs(result["resonance_error"]) < 0.05
    assert result["s11_min_db"] < -10
    assert 5.0 < result["directivity_dbi"] < 8.0


# ------------------------------------------------------- sweeps and optimizer
def _transmission_at_8ghz(sim):
    R, T = reflection_transmission(sim)
    return float(np.interp(8e9, sim.frequencies, T))


def _slab(thickness):
    return dielectric_slab_cell(eps_r=4.0, thickness=thickness, f_min=6e9, f_max=10e9)


def _reflection_at_8ghz(sim):
    return 1 - _transmission_at_8ghz(sim)


def test_sweep_and_optimize():
    """
    Sweep of the slab thickness (sequential and with 2 processes, identical
    results) and a bounded scalar optimization of the thickness for maximum
    transmission at 8 GHz: the optimum is the half-wave slab
    d = c / (2 n f) = 9.37 mm (n = 2), within 2 %.
    """
    thicknesses = {"thickness": [8e-3, 9.37e-3]}
    sweep = parameter_sweep(_slab, _transmission_at_8ghz, thicknesses, reference=True)
    parallel = parameter_sweep(
        _slab, _transmission_at_8ghz, thicknesses, processes=2, reference=True
    )
    assert [p["result"] for p in sweep] == pytest.approx(
        [p["result"] for p in parallel], abs=1e-12
    )
    assert sweep[1]["result"] > 0.99 > sweep[0]["result"]

    result = optimize(
        _slab,
        _reflection_at_8ghz,
        {"thickness": 8.5e-3},
        bounds={"thickness": (7e-3, 11e-3)},
        method="bounded",
        run_kwargs={"reference": True},
        xatol=2e-5,
    )
    np.testing.assert_allclose(result.params["thickness"], c / (2 * 2 * 8e9), rtol=0.02)
    assert len(result.history) == result.nfev


def test_plotting_helpers():
    """Plot helpers and geometry export run without errors (smoke test)."""
    f = np.linspace(1e9, 2e9, 11)
    s11 = 0.5 * np.exp(1j * f / 1e9)
    for fig, _ in (
        analysis.plot_s_parameters(f, {"S11": s11}),
        analysis.plot_impedance(f, 50 * (1 + s11) / (1 - s11)),
        analysis.plot_spectra(f, {"T": np.abs(s11) ** 2}),
    ):
        plt.close(fig)
    assert analysis.bandwidth(f, s11) == 0.0
    f_min, value = analysis.find_minimum(f, (f - 1.53e9) ** 2)
    np.testing.assert_allclose(f_min, 1.53e9)
    sim = Simulation(1e9, 2e9, [Box((0, 0, 0), (0.01, 0.01, 0.01), "pec")])
    assert sim.geometry().n_blocks == 1
