import numpy as np
from scipy.constants import c, epsilon_0

import wakis
from wakis.sources import CurrentSource, GaussianSource


def _transfer_matrix_transmission(f, layers):
    """
    Normal-incidence power transmission of a lossless stack of (n, d) layers in
    vacuum, from the product of characteristic matrices
    [[cos d_j, i sin d_j / n_j], [i n_j sin d_j, cos d_j]], d_j = k0 n_j d_j
    (Born & Wolf, Principles of Optics, Sec. 1.6).
    """
    T = np.empty(f.size)
    for i, k0 in enumerate(2 * np.pi * f / c):
        M = np.eye(2, dtype=complex)
        for n, d in layers:
            delta = k0 * n * d
            M = M @ np.array(
                [
                    [np.cos(delta), 1j * np.sin(delta) / n],
                    [1j * n * np.sin(delta), np.cos(delta)],
                ]
            )
        T[i] = np.abs(2 / M.sum()) ** 2
    return T


def _stack_fluxes(f, layers=None, dz=2.5e-3, use_gpu=False):
    """Flux spectra below and above a quasi-1D stack starting at z = 0."""
    grid = wakis.GridFIT3D(
        -0.01, 0.01, -0.01, 0.01, -0.6, 0.9, 2, 2, round(1.5 / dz), verbose=0
    )
    solver = wakis.SolverFIT3D(
        grid,
        use_gpu=use_gpu,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=15,
        verbose=0,
    )
    if layers is not None:
        # Tangential E sits on the z nodes; each node sees the mean eps of the
        # two half cells around it (interfaces lie on nodes)
        bounds = np.cumsum([0.0] + [d for _, d in layers])
        eps = np.array([n**2 for n, _ in layers])

        def eps_at(z):
            j = np.searchsorted(bounds, z, side="right") - 1
            inside = (j >= 0) & (j < eps.size)
            return np.where(inside, eps[np.clip(j, 0, eps.size - 1)], 1.0)

        z = grid.z[: solver.Nz]
        eps_node = 0.5 * (eps_at(z - dz / 2) + eps_at(z + dz / 2))
        for comp in "xy":
            for k in range(solver.Nz):
                solver.ieps[:, :, k, comp] = 1 / (epsilon_0 * eps_node[k])

    src_time = GaussianSource(frequency=1e9, fwidth=0.4e9)
    sheet = CurrentSource(
        src_time, component="x", center=(0, 0, -0.3), size=(0.02, 0.02, 0)
    )
    planes = [
        wakis.FluxMonitor(f, center=(0, 0, z0), size=(0.02, 0.02, 0))
        for z0 in (-0.15, 0.6)
    ]
    Nt = int((src_time.end_time + 25e-9) / solver.dt)
    solver.emsolve(Nt, source=sheet, monitors=planes)
    return [plane.get_flux() for plane in planes]


def test_bragg_mirror(use_gpu):
    """
    Quarter-wave Bragg mirror (HL)^4 H at f0 = 1 GHz, n_H = 2, n_L = 1.5
    (37.5 mm and 50 mm layers, all interfaces on grid nodes, 2.5 mm cells:
    >= 40 cells per wavelength in the layers). T(f) from flux planes,
    normalized by a run without the stack, follows the transfer-matrix result
    across the stop band and the side lobes, 0.5-1.5 GHz: within 0.015
    absolute (largest at the band edge near 1.15 GHz, where dT/df is steepest
    and the 0.1 % phase error of the mesh shifts the edge), within 3 % in the
    stop band (|f - f0| < 0.1 f0, T = 0.1-0.15), and R + T = 1 to 1e-3.
    """
    f0, n_H, n_L = 1e9, 2.0, 1.5
    d_H, d_L = c / f0 / (4 * n_H), c / f0 / (4 * n_L)
    layers = [(n_H, round(d_H / 2.5e-3) * 2.5e-3), (n_L, round(d_L / 2.5e-3) * 2.5e-3)]
    layers = layers * 4 + layers[:1]
    f = np.linspace(0.5e9, 1.5e9, 41)

    below_ref, above_ref = _stack_fluxes(f, use_gpu=use_gpu)
    below, above = _stack_fluxes(f, layers, use_gpu=use_gpu)
    T = above / above_ref
    R = 1 - below / below_ref
    T_analytic = _transfer_matrix_transmission(f, layers)

    np.testing.assert_allclose(T, T_analytic, atol=1.5e-2)
    stop_band = np.abs(f - f0) < 0.1 * f0
    np.testing.assert_allclose(T[stop_band], T_analytic[stop_band], rtol=3e-2)
    assert T[stop_band].max() < 0.16  # > 84 % reflected across the stop band
    np.testing.assert_allclose(R + T, 1.0, atol=1e-3)
