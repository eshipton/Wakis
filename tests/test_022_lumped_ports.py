import numpy as np
import pytest
from scipy.constants import c, epsilon_0, mu_0

import wakis
from wakis.lumped import LumpedElement, LumpedPort
from wakis.sources import GaussianSource
from wakis.sparameters import read_touchstone, s_matrix, write_touchstone

ETA0 = np.sqrt(mu_0 / epsilon_0)

# Parallel-plate TEM line along z: PEC plates at x = 0 and x = GAP (the last x
# node is the high PEC wall), periodic in y with width WIDTH, CPML at both z
# ends. Its characteristic impedance is Z = eta0 * GAP / WIDTH = 94.2 Ohm.
DZ = 2e-3
GAP, WIDTH = 10e-3, 40e-3
Z_LINE = ETA0 * GAP / WIDTH
FREQS = np.linspace(1e9, 5e9, 5)  # 150 to 30 cells per wavelength
SRC = GaussianSource(frequency=3e9, fwidth=1.5e9)
NT_MAX = 4000


def _line_solver(use_gpu=False):
    grid = wakis.GridFIT3D(
        0, GAP + DZ, -WIDTH / 2, WIDTH / 2, -0.06, 0.06, 6, 2, 60, verbose=0
    )
    return wakis.SolverFIT3D(
        grid,
        use_gpu=use_gpu,
        bc_low=["pec", "periodic", "cpml"],
        bc_high=["pec", "periodic", "cpml"],
        n_pml=10,
        verbose=0,
    )


def _gap(z, **kwargs):
    """Sheet of x edges spanning the plate gap over the full width at z."""
    return dict(
        component="x", center=(GAP / 2, 0.0, z), size=(GAP, WIDTH, 0.0), **kwargs
    )


def _discrete_line(f, dt):
    """
    Exact TEM propagation constant and node impedance of the Yee line.

    The fields are uniform transversally, so the grid is a 1-D Yee line with
    sin(beta dz/2)/dz = sin(w dt/2)/(c dt). Seen from a node carrying half of
    the node capacitance, a semi-infinite line has the real impedance
    Z / cos(beta dz/2), which converges to Z at second order.
    """
    beta = 2 / DZ * np.arcsin(DZ / (c * dt) * np.sin(np.pi * f * dt))
    return beta, Z_LINE / np.cos(beta * DZ / 2)


def _load_reflection(z_in, f, dt, n_cells):
    """
    Reflection coefficient at a shunt load n_cells to the right of a centre
    port, de-embedded from the port input impedance with the discrete line:
    the port sees the left half-line in parallel with the right side.
    """
    beta, z_d = _discrete_line(f, dt)
    z_right = 1 / (1 / z_in - 1 / z_d)
    gamma_port = (z_right - z_d) / (z_right + z_d)
    gamma = gamma_port * np.exp(2j * beta * n_cells * DZ)
    # The load is shunted by the matched line continuing behind it
    z_term = z_d * (1 + gamma) / (1 - gamma)
    z_load = 1 / (1 / z_term - 1 / z_d)
    return gamma, z_load


def _run_port_and_load(use_gpu, n_cells=5, **load):
    solver = _line_solver(use_gpu)
    port = LumpedPort(SRC, **_gap(0.0))
    element = LumpedElement(**_gap(n_cells * DZ, **load))
    solver.emsolve(NT_MAX, source=[port, element], stop_when_energy_decayed=1e-6)
    return _load_reflection(port.impedance_in(FREQS), FREQS, solver.dt, n_cells)


def test_pec_walls_with_periodic_axis():
    """
    Regression: PEC plates combined with a periodic transverse axis keep the
    tangential E on the plates at zero, so the line carries a uniform TEM mode
    (the PEC mask used to be reset on the edge lines shared with the periodic
    faces).
    """
    solver = _line_solver()
    port = LumpedPort(SRC, **_gap(0.0))
    solver.emsolve(300, source=port)
    Ex = solver.E[:-1, :, 35, "x"]
    assert np.abs(solver.E[[0, -1], :, :, "z"]).max() == 0
    assert np.abs(Ex).max() > 0
    np.testing.assert_allclose(Ex, Ex.mean(), rtol=1e-10)


@pytest.mark.parametrize("Z0", [50.0, 20.0])
def test_port_impedance(Z0, use_gpu):
    """
    A port in the middle of a matched parallel-plate line sees both half-lines
    in parallel, Zin = Z/2 with Z = eta0 d/w, and S11 = (Zin - Z0)/(Zin + Z0).
    The FIT error is the second-order node term 1/cos(beta dz/2), 0.55% at
    30 cells per wavelength; the CPML reflections are below 1e-3.
    """
    solver = _line_solver(use_gpu)
    port = LumpedPort(SRC, **_gap(0.0, impedance=Z0))
    solver.emsolve(NT_MAX, source=port, stop_when_energy_decayed=1e-6)

    z_in = port.impedance_in(FREQS)
    np.testing.assert_allclose(z_in, Z_LINE / 2, rtol=1e-2)
    _, z_d = _discrete_line(FREQS, solver.dt)
    np.testing.assert_allclose(z_in, z_d / 2, rtol=2e-3)

    s11_expected = (Z_LINE / 2 - Z0) / (Z_LINE / 2 + Z0)
    np.testing.assert_allclose(port.s11(FREQS), s11_expected, atol=5e-3)


@pytest.mark.parametrize("R_ratio", [1e-3, 0.1, 1.0, 10.0])
def test_resistor_load(R_ratio, use_gpu):
    """
    A resistor R across the line terminates the incident wave with R in
    parallel with the matched line behind it, Z_T = R || Z, so the reflection
    is Gamma = (Z_T - Z)/(Z_T + Z) = -Z/(2R + Z): from a near short
    (R = Z/1000, where an explicit resistor update would be unstable) to
    nearly transparent (R = 10 Z).
    """
    R = R_ratio * Z_LINE
    gamma, z_load = _run_port_and_load(use_gpu, R=R)
    np.testing.assert_allclose(gamma, -Z_LINE / (2 * R + Z_LINE), atol=3e-3)
    if 0.1 <= R_ratio <= 1.0:
        # De-embedding a nearly transparent shunt amplifies errors, skip 10 Z
        np.testing.assert_allclose(z_load, R, rtol=1e-2)


@pytest.mark.parametrize("load", [{"L": 5e-9}, {"C": 0.56e-12}])
def test_reactive_load(load, use_gpu):
    """
    The impedance of a lumped inductor or capacitor de-embedded from the port
    is jwL or 1/(jwC) in magnitude and phase (|Z| ~ 30-280 Ohm).
    """
    _, z_load = _run_port_and_load(use_gpu, **load)
    w = 2 * np.pi * FREQS
    expected = 1j * w * load["L"] if "L" in load else 1 / (1j * w * load["C"])
    np.testing.assert_allclose(z_load, expected, rtol=1.5e-2)


def test_lc_resonance(use_gpu):
    """
    A parallel LC load is an open circuit at f0 = 1/(2 pi sqrt(LC)), where the
    line becomes transparent and the reflection vanishes.
    """
    L, C = 5e-9, 0.56e-12
    f0 = 1 / (2 * np.pi * np.sqrt(L * C))
    solver = _line_solver(use_gpu)
    port = LumpedPort(SRC, **_gap(0.0))
    tank = LumpedElement(**_gap(5 * DZ, L=L, C=C))
    solver.emsolve(NT_MAX, source=[port, tank], stop_when_energy_decayed=1e-6)

    f = np.linspace(0.9 * f0, 1.1 * f0, 201)
    gamma, _ = _load_reflection(port.impedance_in(f), f, solver.dt, 5)
    assert abs(f[np.argmin(np.abs(gamma))] / f0 - 1) < 2e-3
    assert np.abs(gamma).min() < 5e-3


def test_two_port_s_matrix():
    """
    Two ports 10 cells apart on the matched line form a reciprocal two-port:
    each port node is shunted by the outer half-line, with the line section in
    between. The S-matrix agrees with the ABCD-matrix result for the discrete
    line and S21 = S12.
    """
    n_cells, Z0 = 10, 50.0
    ports = [
        LumpedPort(**_gap(-n_cells * DZ / 2, impedance=Z0)),
        LumpedPort(**_gap(n_cells * DZ / 2, impedance=Z0)),
    ]
    S = s_matrix(_line_solver, ports, SRC, NT_MAX, FREQS, stop_when_energy_decayed=1e-6)

    beta, z_d = _discrete_line(FREQS, _line_solver().dt)
    theta = beta * n_cells * DZ
    shunt = np.array([[np.ones_like(z_d), 0 * z_d], [1 / z_d, np.ones_like(z_d)]])
    line = np.array(
        [
            [np.cos(theta), 1j * z_d * np.sin(theta)],
            [1j * np.sin(theta) / z_d, np.cos(theta)],
        ]
    )
    (A, B), (C, D) = np.einsum("ijf,jkf,klf->ilf", shunt, line, shunt)
    delta = A + B / Z0 + C * Z0 + D
    S_expected = np.stack(
        [
            [(A + B / Z0 - C * Z0 - D) / delta, 2 / delta],
            [2 / delta, (-A + B / Z0 - C * Z0 + D) / delta],
        ]
    ).transpose(2, 0, 1)

    np.testing.assert_allclose(S, S_expected, atol=3e-3)
    np.testing.assert_allclose(S[:, 0, 1], S[:, 1, 0], atol=1e-3)


def test_energy_stop_criterion():
    """
    The -40 dB energy criterion stops the run well before Nt once the pulse has
    left through the CPML, and the port impedance equals that of a full run to
    within the residual-energy level.
    """
    results = []
    for stop in (1e-4, None):
        solver = _line_solver()
        port = LumpedPort(SRC, **_gap(0.0))
        solver.emsolve(NT_MAX, source=port, stop_when_energy_decayed=stop)
        results.append((solver.Nt, port.impedance_in(FREQS)))

    (nt_stop, z_stop), (nt_full, z_full) = results
    assert nt_full == NT_MAX
    assert nt_stop < NT_MAX / 4
    assert nt_stop * solver.dt > SRC.end_time
    np.testing.assert_allclose(z_stop, z_full, rtol=1e-2)


@pytest.mark.parametrize("n_ports", [1, 2, 3, 5])
def test_touchstone_roundtrip(n_ports, tmp_path):
    """
    Writing and reading a Touchstone file restores the S-matrix; scikit-rf, an
    independent reader, agrees on the port ordering.
    """
    rng = np.random.default_rng(n_ports)
    freqs = np.linspace(1e9, 2e9, 4)
    S = rng.normal(size=(4, n_ports, n_ports)) + 1j * rng.normal(
        size=(4, n_ports, n_ports)
    )
    filename = tmp_path / f"device.s{n_ports}p"
    write_touchstone(filename, freqs, S, z0=75.0, comment="wakis test")

    f_read, S_read, z0 = read_touchstone(filename)
    np.testing.assert_allclose(f_read, freqs)
    np.testing.assert_allclose(S_read, S, rtol=1e-11)
    assert z0 == 75.0

    skrf = pytest.importorskip("skrf")
    network = skrf.Network(str(filename))
    np.testing.assert_allclose(network.s, S, rtol=1e-11)
    np.testing.assert_allclose(network.z0, 75.0)
