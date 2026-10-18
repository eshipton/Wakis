import numpy as np
from scipy.constants import epsilon_0, mu_0

import wakis
from wakis.sources import CurrentSource, GaussianSource


def _box_weights(grid, solver, field, half):
    """
    Fraction of each sample's control volume inside the cube |x|,|y|,|z| <= half.

    The control interval of a sample is its primary edge along the direction
    where it sits on a midpoint and its dual edge where it sits on a node, so
    E samples on a face of the box (on the nodes) count 1/2, on an edge 1/4.
    """
    weights = []
    for comp in range(3):
        per_axis = []
        for d, nodes in enumerate((grid.x, grid.y, grid.z)):
            n = [solver.Nx, solver.Ny, solver.Nz][d]
            if (field == "E") == (d == comp):  # midpoint sample
                lo, hi = nodes[:n], nodes[1 : n + 1]
            else:  # node sample
                mid = (nodes[:-1] + nodes[1:]) / 2
                lo, hi = np.concatenate([[nodes[0]], mid[: n - 1]]), mid[:n]
            inside = np.clip(np.minimum(hi, half) - np.maximum(lo, -half), 0, None)
            per_axis.append(inside / (hi - lo))
        weights.append(np.einsum("i,j,k->ijk", *per_axis).ravel(order="F"))
    return np.concatenate(weights)


def _surface_flux_terms(solver, half):
    """
    Explicit discrete Poynting flux out of the cube, sum (E_b H_c - E_c H_b) dA
    over the six faces: tangential E on the face nodes, the H component it
    multiplies averaged from the two sides of the face, area elements = primary
    length of E x dual width of H, half weight on the cube edges (trapezoid
    rule). Returns flat indices and weights for (E, H below, H above).
    """
    grid = solver.grid
    tol = 1e-9
    idx_E, idx_Hlo, idx_Hhi, weight = [], [], [], []
    N = solver.Nx * solver.Ny * solver.Nz
    strides = (1, solver.Nx, solver.Nx * solver.Ny)
    for a in range(3):
        for sign in (-1.0, 1.0):
            center = np.zeros(3)
            center[a] = sign * half
            size = np.full(3, 2 * half)
            size[a] = 0.0
            b, c = (a + 1) % 3, (a + 2) % 3
            for e, h, s in ((b, c, 1.0), (c, b, -1.0)):
                (I, J, K), coords = grid.region_indices("E", "xyz"[e], center, size)
                I, J, K = np.meshgrid(I, J, K, indexing="ij")
                dA = grid.L[I, J, K, "xyz"[e]] * grid.tL[I, J, K, "xyz"[h]]
                for d in (b, c):
                    edge = np.abs(np.abs(coords[d]) - half) < tol
                    shape = [1, 1, 1]
                    shape[d] = -1
                    dA = dA * np.where(edge, 0.5, 1.0).reshape(shape)
                flat = I + strides[1] * J + strides[2] * K
                idx_E.append((e * N + flat).ravel())
                idx_Hhi.append((h * N + flat).ravel())
                idx_Hlo.append((h * N + flat - strides[a]).ravel())
                weight.append((sign * s * dA).ravel())
    return [np.concatenate(v) for v in (idx_E, idx_Hlo, idx_Hhi, weight)]


def test_poynting_theorem_time_domain():
    """
    Discrete Poynting theorem in a conducting medium (sigma = 3e-4 S/m), per
    timestep and over a cube around a dipole:

        (W^{n+1} - W^n)/dt + Phi^{n+1/2} + sum sigma E^n Ebar dV = -sum J Ebar dV,

    with Ebar = (E^n + E^{n+1})/2, the leapfrog energy
    W^n = 1/2 sum eps E^n E^n dV + 1/2 sum mu H^{n-1/2} H^{n+1/2} dV
    (control volumes cut by the cube faces), J the source current at
    t^{n+1/2} and Phi the explicit surface flux of Ebar x H^{n+1/2} with the
    FluxMonitor geometry. Since the curl matrices are mutual transposes, the
    balance is exact: it closes to round-off (~1e-14) at every step, while
    radiation through the surface (37 %) and ohmic loss (61 %, mostly in the
    dipole near field) each carry a large part of the work.
    The explicit loss term sigma E^n (not time-centred) appears as is.
    """
    N, L, sigma = 20, 0.2, 3e-4
    dx, half = L / N, 0.04
    grid = wakis.GridFIT3D(
        -L / 2, L / 2, -L / 2, L / 2, -L / 2, L / 2, N, N, N, verbose=0
    )
    solver = wakis.SolverFIT3D(
        grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, bg=[1.0, 1.0, sigma], verbose=0
    )
    src_time = GaussianSource(frequency=1e9, fwidth=0.5e9)
    dipole = CurrentSource(src_time, component="z", center=(dx, -dx, 0.0))

    tA = 1 / np.where(grid.itA.toarray() != 0, grid.itA.toarray(), np.inf)
    A = 1 / np.where(grid.iA.toarray() != 0, grid.iA.toarray(), np.inf)
    dV_E = _box_weights(grid, solver, "E", half) * grid.L.toarray() * tA
    dV_H = _box_weights(grid, solver, "H", half) * grid.tL.toarray() * A
    i_E, i_Hlo, i_Hhi, dA = _surface_flux_terms(solver, half)

    dt = solver.dt
    Nt = int((src_time.end_time + 2e-9) / dt)
    E_old, H_old = solver.E.toarray().copy(), solver.H.toarray().copy()
    W, flux, ohmic, work = [], [], [], []
    for n in range(Nt):
        dipole.update(solver, n * dt)
        solver.one_step()
        E, H = solver.E.toarray().copy(), solver.H.toarray().copy()
        E_bar = 0.5 * (E_old + E)
        W.append(
            0.5 * epsilon_0 * np.sum(dV_E * E_old**2)
            + 0.5 * mu_0 * np.sum(dV_H * H_old * H)
        )
        flux.append(np.sum(dA * E_bar[i_E] * 0.5 * (H[i_Hlo] + H[i_Hhi])))
        J_ohm = sigma * E_old
        J_src = solver.J.toarray() - J_ohm
        ohmic.append(np.sum(dV_E * J_ohm * E_bar))
        work.append(-np.sum(dV_E * J_src * E_bar))
        E_old, H_old = E, H
    W, flux, ohmic, work = map(np.array, (W, flux, ohmic, work))

    residual = np.diff(W) / dt + flux[:-1] + ohmic[:-1] - work[:-1]
    assert np.abs(residual).max() < 1e-12 * np.abs(work).max()

    # Energy budget of the whole run: both loss channels matter
    total = work.sum() * dt
    assert 0.2 < flux.sum() * dt / total < 0.8
    assert 0.2 < ohmic.sum() * dt / total < 0.8


def test_lossy_medium_attenuation(use_gpu):
    """
    A plane wave in a conducting medium decays as exp(-alpha z) with
    alpha - j beta = j k, k = omega sqrt(mu0 eps0 (1 - j sigma/(omega eps0)))
    (phasors exp(+j omega t)). Loss tangent sigma/(omega eps0) = 0.83-0.36
    over 0.6-1.4 GHz; alpha and beta fitted over 0.5 m (60 cells per
    wavelength) agree to 1 % (grid dispersion and the O(sigma dt/eps0) = 2 %
    explicit loss term give 0.4-0.8 %).
    """
    dz, sigma = 5e-3, 0.0278
    grid = wakis.GridFIT3D(
        -0.01, 0.01, -0.01, 0.01, -0.3, 0.9, 2, 2, int(1.2 / dz), verbose=0
    )
    solver = wakis.SolverFIT3D(
        grid,
        use_gpu=use_gpu,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=15,
        bg=[1.0, 1.0, sigma],
        verbose=0,
    )
    src_time = GaussianSource(frequency=1e9, fwidth=0.4e9)
    sheet = CurrentSource(
        src_time, component="x", center=(0, 0, -0.2), size=(0.02, 0.02, 0)
    )
    f = np.array([0.6e9, 0.8e9, 1.0e9, 1.2e9, 1.4e9])
    line = wakis.DFTMonitor(f, components=["Ex"], center=(0, 0, 0.25), size=(0, 0, 0.5))
    solver.emsolve(
        int((src_time.end_time + 8e-9) / solver.dt), source=sheet, monitors=[line]
    )

    E, coords = line.get_field("Ex")
    E, z = E[:, 0, 0, :], coords[2]
    omega = 2 * np.pi * f
    k = omega * np.sqrt(mu_0 * epsilon_0 * (1 - 1j * sigma / (omega * epsilon_0)))
    alpha = -np.polyfit(z, np.log(np.abs(E)).T, 1)[0]
    beta = -np.polyfit(z, np.unwrap(np.angle(E), axis=1).T, 1)[0]

    np.testing.assert_allclose(alpha, -k.imag, rtol=1e-2)
    np.testing.assert_allclose(beta, k.real, rtol=1e-2)
