# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Waveguide modes on a cross-section of the FIT grid, and mode ports.

`solve_modes` computes the propagating modes of a waveguide that is uniform
along the normal of a grid plane, using the same staggered (Yee/FIT) samples,
material tensors and boundary conditions as `SolverFIT3D`. With the discrete
propagation constant and the time-discrete frequency, the modes are exact
eigenmodes of the 3D update equations of a uniform guide.

`ModeSource` injects one mode in one direction through electric and magnetic
current sheets (equivalence principle), and `ModeMonitor` decomposes the
Fourier-transformed fields on a plane into forward and backward mode
amplitudes.
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import c as c_light
from scipy.sparse.linalg import eigs

from .field import Field

_PEC_BC = ("electric", "pec", "pml", "cpml")
_PMC_BC = ("magnetic", "pmc")


def _host(array):
    return array.get() if hasattr(array, "get") else array


class _Plane:
    """
    Geometry, boundary masks and sample indices of a grid plane normal to one
    axis. The transverse axes (b, c) follow the cyclic order of the normal a,
    so that b x c = a. Plane arrays have shape (n_b, n_c).
    """

    def __init__(self, solver, center, size):
        if solver.use_mpi:
            raise NotImplementedError("Mode ports do not support MPI yet.")
        center = np.asarray(center, dtype=float)
        size = np.asarray(size, dtype=float)
        zero_axes = np.flatnonzero(size == 0)
        if zero_axes.size != 1:
            raise ValueError("size must have exactly one zero entry (port plane).")
        grid = solver.grid
        self.solver = solver
        a = int(zero_axes[0])
        b, c = (a + 1) % 3, (a + 2) % 3
        self.a, self.b, self.c = a, b, c
        nodes = (grid.x, grid.y, grid.z)
        shape = (solver.Nx, solver.Ny, solver.Nz)

        # E-plane index: the transverse E components sit on the axis nodes
        positions = nodes[a][: shape[a]]
        k = int(np.argmin(np.abs(positions - center[a])))
        if k < 1 or k > shape[a] - 2:
            raise ValueError("The port plane must be at least one cell inside.")
        self.k = k
        self.n_b, self.n_c = shape[b], shape[c]

        self.L_b = self.take(grid.L, b, k)[:, 0]
        self.tL_b = self.take(grid.tL, b, k)[:, 0]
        self.L_c = self.take(grid.L, c, k)[0, :]
        self.tL_c = self.take(grid.tL, c, k)[0, :]
        # Primary cell length along the normal below and above the E plane: the
        # neighbouring H planes are half of these away
        self.L_a_lo = float(self.take(grid.L, a, k - 1)[0, 0])
        self.L_a_hi = float(self.take(grid.L, a, k)[0, 0])
        self.tL_a = self.take(grid.tL, a, k)

        # Sample coordinates [m]: nodes and edge midpoints along b and c
        node_b, node_c = nodes[b][: self.n_b], nodes[c][: self.n_c]
        mid_b = 0.5 * (nodes[b][:-1] + nodes[b][1:])[: self.n_b]
        mid_c = 0.5 * (nodes[c][:-1] + nodes[c][1:])[: self.n_c]
        self.coords = {
            "Eb": (mid_b, node_c),
            "Ec": (node_b, mid_c),
            "Hb": (node_b, mid_c),
            "Hc": (mid_b, node_c),
        }
        self.position = positions[k]
        self._set_masks(solver, center, size, nodes)

    def take(self, field, comp, k):
        """Component `comp` of a Field on the plane k along the normal."""
        m = np.take(_host(field.to_matrix(comp)), k, axis=self.a)
        return m.T if self.b > self.c else m

    def _set_masks(self, solver, center, size, nodes):
        b, c = self.b, self.c
        bc_lo = [s.lower() for s in solver.bc_low]
        bc_hi = [s.lower() for s in solver.bc_high]
        self.periodic_b = bc_lo[b] == "periodic"
        self.periodic_c = bc_lo[c] == "periodic"
        shape = (self.n_b, self.n_c)
        mEb, mEc, mEa = (np.ones(shape, dtype=bool) for _ in range(3))
        mHb, mHc, mHa = (np.ones(shape, dtype=bool) for _ in range(3))

        # Ghost layer: the edge midpoints of the last index along a
        # non-periodic axis
        if not self.periodic_b:
            mEb[-1, :] = mHc[-1, :] = mHa[-1, :] = False
        if not self.periodic_c:
            mEc[:, -1] = mHb[:, -1] = mHa[:, -1] = False
        # PMC is the natural FIT boundary when the walls sit on the domain
        # edge (see `_apply_bc_to_C`); legacy grids zero the tangential H
        pmc = () if solver.grid.boundary_at_domain_edge else _PMC_BC

        # PEC (and CPML/PML, terminated by PEC): tangential E on the walls is
        # zero; PMC: tangential H, as in `_apply_bc_to_C`
        for side, bcs in ((0, bc_lo), (-1, bc_hi)):
            if bcs[b] in _PEC_BC:
                mEc[side, :] = mEa[side, :] = False
            if bcs[c] in _PEC_BC:
                mEb[:, side] = mEa[:, side] = False
            if bcs[b] in pmc:
                mHc[side, :] = mHa[side, :] = False
            if bcs[c] in pmc:
                mHb[:, side] = mHa[:, side] = False

        # Port window smaller than the domain: electric walls on its edges
        lo, hi = center - size / 2, center + size / 2
        for d, axis_masks in ((b, 0), (c, 1)):
            x = nodes[d]
            tol = 1e-9 * (x[-1] - x[0])
            x_max = x[-1 - solver.grid.edge_ghosts[d]]  # domain bound
            if lo[d] <= x[0] + tol and hi[d] >= x_max - tol:
                continue
            n = self.n_b if axis_masks == 0 else self.n_c
            node = x[:n]
            inside_node = (node > lo[d] + tol) & (node < hi[d] - tol)
            inside_edge = (x[:n] >= lo[d] - tol) & (x[1 : n + 1] <= hi[d] + tol)
            if axis_masks == 0:
                mEb &= inside_edge[:, None]
                mEc &= inside_node[:, None]
                mEa &= inside_node[:, None]
            else:
                mEb &= inside_node[None, :]
                mEc &= inside_edge[None, :]
                mEa &= inside_node[None, :]

        # Perfect conductors inside the domain have ieps = 0
        k = self.k
        self.ieps_b = self.take(solver.ieps, b, k)
        self.ieps_c = self.take(solver.ieps, c, k)
        self.ieps_a = self.take(solver.ieps, self.a, k) * mEa
        self.imu_b = self.take(solver.imu, b, k)
        self.imu_c = self.take(solver.imu, c, k)
        self.imu_a = self.take(solver.imu, self.a, k) * mHa
        self.mask_E = np.concatenate(
            [
                (mEb & (self.ieps_b != 0)).ravel("F"),
                (mEc & (self.ieps_c != 0)).ravel("F"),
            ]
        )
        # H samples transversally colocated with an inactive E sample (H_b with
        # E_c, H_c with E_b) are normal to an electric wall: Ampere's law does
        # not hold there and Faraday's law gives zero
        n = self.n_b * self.n_c
        self.mask_H = np.concatenate([mHb.ravel("F"), mHc.ravel("F")])
        self.mask_H &= np.concatenate([self.mask_E[n:], self.mask_E[:n]])

        # Area elements of the (E_b, H_c) and (E_c, H_b) pairs, as in FluxMonitor
        self.dA_b = np.outer(self.L_b, self.tL_c)
        self.dA_c = np.outer(self.tL_b, self.L_c)

    def flat_index(self, comp, k):
        """Field.array indices of component `comp` on plane k, Fortran order."""
        s = self.solver
        idx = [None, None, None]
        idx[self.a] = k
        idx[self.b], idx[self.c] = np.meshgrid(
            np.arange(self.n_b), np.arange(self.n_c), indexing="ij"
        )
        flat = comp * s.N + idx[0] + s.Nx * idx[1] + s.Nx * s.Ny * idx[2]
        return np.broadcast_to(flat, (self.n_b, self.n_c)).ravel("F")


def _difference(lengths, periodic):
    """Forward difference node -> edge midpoint divided by the edge length,
    and the matching backward difference with the dual lengths."""
    n = lengths[0].size
    P = sp.lil_matrix(sp.diags([-1.0, 1.0], [0, 1], shape=(n, n)))
    if periodic:
        # Same wrap as the 3D curl; a single periodic cell has no derivative
        P[n - 1, 0] = 1.0 if n > 1 else 0.0
        if n == 1:
            P[0, 0] = 0.0
    P = P.tocsr()
    L, tL = lengths
    itL = np.divide(1.0, tL, out=np.zeros_like(tL), where=tL != 0)
    return sp.diags(1.0 / L) @ P, sp.diags(itL) @ (-P.T)


class Mode:
    """
    Waveguide mode on a grid plane, normalized to unit power.

    Attributes
    ----------
    frequency : float
        Frequency [Hz].
    beta : float
        Propagation constant [1/m] of the discrete 3D scheme.
    neff : float
        Effective index beta * c / (2 pi f).
    e : dict of ndarray
        Transverse E profile {'b': E_b, 'c': E_c} [V/m], shape (n_b, n_c), on
        the E samples of the port plane.
    h : dict of ndarray
        Transverse H profile {'b': H_b, 'c': H_c} [A/m] of the forward mode on
        the neighbouring H samples, with the phase referred to the H plane
        (H at distance d from the E plane is h * exp(-1j*beta*d)).
    h_mean : dict of ndarray
        H averaged from both sides onto the E plane, h * cos(beta*L_a/2).
    axes : str
        Names of the normal and transverse axes (a, b, c), e.g. 'zxy'.
    coords : dict
        Sample coordinates [m] of 'Eb', 'Ec', 'Hb', 'Hc' along (b, c).
    """

    def __init__(self, plane, frequency, beta_tilde, e, h):
        self.frequency = frequency
        self.beta_tilde = beta_tilde
        L_a = 0.5 * (plane.L_a_lo + plane.L_a_hi)
        self.beta = 2 / L_a * np.arcsin(beta_tilde * L_a / 2)
        self.neff = self.beta * c_light / (2 * np.pi * frequency)
        self.axes = "".join("xyz"[d] for d in (plane.a, plane.b, plane.c))
        self.coords = plane.coords
        n = plane.n_b * plane.n_c
        shape = (plane.n_b, plane.n_c)
        self.e = {
            "b": e[:n].reshape(shape, order="F"),
            "c": e[n:].reshape(shape, order="F"),
        }
        self.h = {
            "b": h[:n].reshape(shape, order="F"),
            "c": h[n:].reshape(shape, order="F"),
        }
        # FluxMonitor averages the H planes at -L_lo/2 and +L_hi/2
        cos_mean = 0.5 * (
            np.exp(1j * self.beta * plane.L_a_lo / 2)
            + np.exp(-1j * self.beta * plane.L_a_hi / 2)
        )
        self.h_mean = {k: v * cos_mean for k, v in self.h.items()}
        power = 0.5 * np.real(
            np.sum(
                self.e["b"] * np.conj(self.h_mean["c"]) * plane.dA_b
                - self.e["c"] * np.conj(self.h_mean["b"]) * plane.dA_c
            )
        )
        scale = 1 / np.sqrt(power)
        for d in (self.e, self.h, self.h_mean):
            for k in d:
                d[k] = d[k] * scale


def solve_modes(solver, frequency, center, size, n_modes=1):
    """
    Propagating modes of a guide that is uniform along the normal of a plane.

    The transverse fields are discretized on the FIT samples of the grid
    plane nearest to `center`. Eliminating the longitudinal components from
    the 3D curl equations with d/da -> -1j*beta_t and omega -> omega_t gives
    beta_t**2 e = P Q e for e = (E_b, E_c), where
    beta_t = 2/L_a sin(beta L_a/2) and omega_t = 2/dt sin(omega dt/2) are the
    spatial and temporal discrete-dispersion substitutes. The eigenvalues
    nearest to the plane-wave bound n_max**2 omega_t**2 / c**2 are found by
    shift-invert Arnoldi iteration.

    Parameters
    ----------
    solver : SolverFIT3D
        Solver providing grid, material tensors (`ieps`, `imu`), boundary
        conditions and timestep. Conductors with ieps = 0 act as PEC.
    frequency : float
        Frequency [Hz].
    center : tuple of float
        A point of the port plane (x, y, z) [m].
    size : tuple of float
        Extent of the port (sx, sy, sz) [m]; exactly one zero entry, which
        sets the plane normal. Transverse extents covering the domain (or
        np.inf) use the domain boundaries; smaller extents put electric walls
        on the port edges (e.g. to exclude CPML layers).
    n_modes : int, optional
        Number of modes to return, sorted by decreasing effective index.

    Returns
    -------
    list of Mode
        Modes normalized to unit time-averaged power,
        1/2 Re sum (e x h_mean^*) . a dA = 1 W.

    Notes
    -----
    Assumes lossless, isotropic materials that are uniform along the normal
    within one cell of the plane, and a locally uniform mesh along the normal.
    Evanescent solutions (beta_t**2 <= 0) are discarded.
    """
    plane = _Plane(solver, center, size)
    w = 2 / solver.dt * np.sin(np.pi * frequency * solver.dt)
    U_b, V_b = _difference((plane.L_b, plane.tL_b), plane.periodic_b)
    U_c, V_c = _difference((plane.L_c, plane.tL_c), plane.periodic_c)
    I_b, I_c = sp.identity(plane.n_b), sp.identity(plane.n_c)
    U_b, V_b = sp.kron(I_c, U_b), sp.kron(I_c, V_b)
    U_c, V_c = sp.kron(U_c, I_b), sp.kron(V_c, I_b)

    def D(values):
        return sp.diags(np.asarray(values).ravel("F"))

    def inv(values):
        values = np.asarray(values)
        return np.divide(1.0, values, out=np.zeros_like(values), where=values != 0)

    eps_b, eps_c = D(inv(plane.ieps_b)), D(inv(plane.ieps_c))
    mu_b, mu_c = D(inv(plane.imu_b)), D(inv(plane.imu_c))
    imu_a, ieps_a = D(plane.imu_a), D(plane.ieps_a)

    # beta_t h = Q e (Ampere's law with H_a eliminated) and beta_t e = P h
    # (Faraday's law with E_a eliminated); see docs/physicsguide.md
    Q = (
        sp.bmat(
            [
                [V_b @ imu_a @ U_c, -(w**2) * eps_c - V_b @ imu_a @ U_b],
                [w**2 * eps_b + V_c @ imu_a @ U_c, -V_c @ imu_a @ U_b],
            ]
        )
        / w
    )
    P = (
        sp.bmat(
            [
                [-U_b @ ieps_a @ V_c, w**2 * mu_c + U_b @ ieps_a @ V_b],
                [-(w**2) * mu_b - U_c @ ieps_a @ V_c, U_c @ ieps_a @ V_b],
            ]
        )
        / w
    )
    Q = sp.diags(plane.mask_H.astype(float)) @ Q
    active = np.flatnonzero(plane.mask_E)
    A = (P @ Q).tocsr()[active][:, active].tocsc()

    eps_mu = inv(np.concatenate([plane.ieps_b.ravel("F"), plane.ieps_c.ravel("F")]))
    eps_mu = eps_mu * inv(
        np.concatenate([plane.imu_c.ravel("F"), plane.imu_b.ravel("F")])
    )
    sigma = w**2 * eps_mu[active].max()
    k = min(n_modes, active.size - 2)
    if k < 1:
        raise ValueError("The port plane has too few active samples.")
    values, vectors = eigs(A, k=k, sigma=sigma, which="LM")
    order = np.argsort(-values.real)
    modes = []
    for i in order:
        if values[i].real <= 0:
            continue
        beta_tilde = np.sqrt(values[i].real)
        e = np.zeros(2 * plane.n_b * plane.n_c, dtype=complex)
        e[active] = vectors[:, i]
        # Remove the arbitrary eigenvector phase: largest E sample is real > 0
        peak = e[np.argmax(np.abs(e))]
        e *= np.abs(peak) / peak
        h = (Q @ e) / beta_tilde
        modes.append(Mode(plane, frequency, beta_tilde, e, h))
    return modes


class ModeSource:
    def __init__(
        self,
        src_time,
        center,
        size,
        direction="+",
        mode_index=0,
        amplitude=1.0,
        mode_frequency=None,
    ):
        """
        Unidirectional injection of a waveguide mode through a port plane.

        By the equivalence principle, the electric and magnetic surface
        currents K = n x H_inc and M_s = -n x E_inc on a plane, with n the
        propagation direction, radiate the incident mode on the n side and
        nothing on the other side. On the staggered grid they are placed as in
        a total-field/scattered-field boundary: K on the E plane, sampled at
        the J time level t + dt/2, and M_s on the H plane half a cell behind,
        sampled at the H time level t. The H profile is advanced in time by
        the phase delay beta*L_a/(2*omega) of the half-cell offset.

        The profile is the eigenmode at `mode_frequency`, so the source is
        unidirectional and modally pure at that frequency; away from it, the
        change of profile, wave impedance and phase velocity with frequency
        leaks a small backward wave.

        Parameters
        ----------
        src_time : GaussianSource, ContinuousSource or CustomSource
            Time profile s(t) [dimensionless].
        center, size : tuple of float
            Port plane, as in `solve_modes` [m].
        direction : {'+', '-'}, optional
            Propagation direction along the plane normal. Default '+'.
        mode_index : int, optional
            Mode number, sorted by decreasing effective index. Default 0.
        amplitude : float, optional
            Mode amplitude [sqrt(W)]. Default 1.
        mode_frequency : float, optional
            Frequency at which the mode is computed [Hz]. Default is
            `src_time.frequency`.

        Notes
        -----
        Normalization: the transverse field launched at the port is
        E_t(t) = amplitude * s(t) * e(x, y) with e the unit-power mode
        profile, so the forward mode amplitude measured by `ModeMonitor` is
        a+(f) = amplitude * S(f), with S(f) the Fourier transform of s(t)
        (`GaussianSource.fourier_transform`). For a time-harmonic source the
        mode carries amplitude**2 W, and a `FluxMonitor` downstream measures
        Phi(f) = 2 |a+(f)|**2. Lossless guides only; MPI is not supported.
        """
        if direction not in ("+", "-"):
            raise ValueError("direction must be '+' or '-'.")
        self.src_time = src_time
        self.center = np.asarray(center, dtype=float)
        self.size = np.asarray(size, dtype=float)
        self.direction = 1 if direction == "+" else -1
        self.mode_index = mode_index
        self.amplitude = amplitude
        self.mode_frequency = (
            src_time.frequency if mode_frequency is None else mode_frequency
        )
        self.is_first_update = True

    def _initialize(self, solver):
        modes = solve_modes(
            solver,
            self.mode_frequency,
            self.center,
            self.size,
            n_modes=self.mode_index + 1,
        )
        if len(modes) <= self.mode_index:
            raise ValueError(
                f"Mode {self.mode_index} does not propagate at "
                f"{self.mode_frequency:.4g} Hz."
            )
        plane = _Plane(solver, self.center, self.size)
        self.mode = mode = modes[self.mode_index]
        n, a, b, c = self.direction, plane.a, plane.b, plane.c
        e, h = mode.e, mode.h

        # K = a x h on the E plane, divided by the dual cell width along a as
        # for a CurrentSource sheet. The same for both directions: n x H_inc
        # with H_inc = n h.
        J_profile = {b: -h["c"].real, c: h["b"].real}
        # M_s = -n a x e on the H plane behind the E plane, divided by the
        # primary cell width along a
        k_H = plane.k - 1 if n > 0 else plane.k
        L_a_H = plane.take(solver.grid.L, a, k_H)
        M_profile = {b: n * e["c"].real, c: -n * e["b"].real}

        xp = solver.J.xp
        dtype = solver.J.dtype
        self.terms = []
        for field, profile, k, width in (
            ("J", J_profile, plane.k, plane.tL_a),
            ("M", M_profile, k_H, L_a_H),
        ):
            idx, weights = [], []
            for comp, values in profile.items():
                values = (self.amplitude * values / width).ravel("F")
                keep = values != 0
                idx.append(plane.flat_index(comp, k)[keep])
                weights.append(values[keep])
            weights = np.concatenate(weights)
            self.terms.append(
                {
                    "field": field,
                    "idx": xp.asarray(np.concatenate(idx)),
                    "weights": xp.asarray(weights, dtype=dtype),
                    "old": xp.zeros(weights.size, dtype=dtype),
                }
            )
        # Phase delay of the incident H half a cell upstream of the E plane
        omega = 2 * np.pi * self.mode_frequency
        L_half = plane.L_a_lo if n > 0 else plane.L_a_hi
        self.h_advance = mode.beta * L_half / (2 * omega)

        if solver.M is None:
            solver.M = Field(
                solver.Nx,
                solver.Ny,
                solver.Nz,
                use_gpu=solver.use_gpu,
                dtype=solver.dtype,
            )

    def update(self, solver, t):
        """
        Add the port currents at time t to the solver's J and M.

        Parameters
        ----------
        solver : SolverFIT3D
            Solver object.
        t : float
            Current simulation time [s].
        """
        if self.is_first_update:
            self._initialize(solver)
            self.is_first_update = False

        times = {"J": t + solver.dt / 2 + self.h_advance, "M": t}
        for term in self.terms:
            new = term["weights"] * float(self.src_time(times[term["field"]]))
            getattr(solver, term["field"]).array[term["idx"]] += new - term["old"]
            term["old"] = new


class ModeMonitor:
    def __init__(self, frequencies, center, size, n_modes=1, mode_frequency=None):
        """
        Forward and backward mode amplitudes on a port plane.

        The Fourier transforms of the transverse E on the plane and of H
        averaged from the two neighbouring H planes (as in `FluxMonitor`) are
        projected on the unit-power modes with the orthogonality relation of
        lossless guides, 1/2 int e_m x h_n^* . a dA = delta_mn:

            a+ = 1/4 int (E x h^* + e^* x H) . a dA
            a- = 1/4 int (E x h^* - e^* x H) . a dA

        so that E_t = sum (a+ + a-) e and H_t = sum (a+ - a-) h.

        Parameters
        ----------
        frequencies : array_like
            Frequencies to monitor [Hz].
        center, size : tuple of float
            Port plane, as in `solve_modes` [m].
        n_modes : int, optional
            Number of modes to decompose into. Default 1.
        mode_frequency : float, optional
            If given, use the modes at this frequency [Hz] for all monitored
            frequencies (faster, approximate for dispersive profiles). By
            default the modes are solved at every monitored frequency.

        Notes
        -----
        The amplitudes have units of sqrt(W) s. For a source with spectrum
        S(f), |a(f)|**2 / |S(f)|**2 is the time-harmonic power [W] carried by
        the mode, and a single forward mode gives a `FluxMonitor` flux
        Phi(f) = 2 |a+(f)|**2. MPI is not supported.
        """
        self.frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        self.center = np.asarray(center, dtype=float)
        self.size = np.asarray(size, dtype=float)
        self.n_modes = n_modes
        self.mode_frequency = mode_frequency
        self.modes = None
        self.is_first_update = True

    def _initialize(self, solver):
        self.solver = solver
        self.plane = plane = _Plane(solver, self.center, self.size)
        xp = solver.E.xp
        self.xp = xp
        n_f, n = self.frequencies.size, plane.n_b * plane.n_c
        self._idx = {"E": [], "H": []}
        for d in (plane.b, plane.c):
            self._idx["E"].append(xp.asarray(plane.flat_index(d, plane.k)))
            below_above = [
                plane.flat_index(d, plane.k - 1),
                plane.flat_index(d, plane.k),
            ]
            self._idx["H"].append(xp.asarray(np.concatenate(below_above)))
        self._acc = {
            "E": xp.zeros((2, n_f, n), dtype=xp.complex128),
            "H": xp.zeros((2, n_f, n), dtype=xp.complex128),
        }

    def update(self, solver, t):
        """
        Accumulate the fields after the timestep that started at time t.

        Parameters
        ----------
        solver : SolverFIT3D
            Solver object.
        t : float
            Time at the start of the step just taken [s].
        """
        if self.is_first_update:
            self._initialize(solver)
            self.is_first_update = False

        n = self.plane.n_b * self.plane.n_c
        for f, tf in (("E", t + solver.dt), ("H", t + solver.dt / 2)):
            kernel = self.xp.asarray(
                solver.dt * np.exp(-2j * np.pi * self.frequencies * tf)
            )[:, None]
            for d in range(2):
                values = getattr(solver, f).array[self._idx[f][d]]
                if f == "H":
                    values = 0.5 * (values[:n] + values[n:])
                self._acc[f][d] += kernel * values[None, :]

    def _solve_modes(self):
        if self.mode_frequency is not None:
            modes = solve_modes(
                self.solver, self.mode_frequency, self.center, self.size, self.n_modes
            )
            self.modes = [modes] * self.frequencies.size
        else:
            self.modes = [
                solve_modes(self.solver, f, self.center, self.size, self.n_modes)
                for f in self.frequencies
            ]

    def get_amplitudes(self, mode_index=0):
        """
        Forward and backward amplitudes of one mode.

        Parameters
        ----------
        mode_index : int, optional
            Mode number, sorted by decreasing effective index. Default 0.

        Returns
        -------
        a_plus, a_minus : ndarray of complex
            Amplitudes along +normal and -normal at each frequency [sqrt(W) s].
            Frequencies where the mode does not propagate give NaN.
        """
        if self.modes is None:
            self._solve_modes()
        plane = self.plane
        shape = (plane.n_b, plane.n_c)
        E = [_host(self._acc["E"][d]) for d in range(2)]
        H = [_host(self._acc["H"][d]) for d in range(2)]
        a_plus = np.full(self.frequencies.size, np.nan, dtype=complex)
        a_minus = np.full(self.frequencies.size, np.nan, dtype=complex)
        for i, modes in enumerate(self.modes):
            if len(modes) <= mode_index:
                continue
            m = modes[mode_index]
            Eb, Ec = (E[d][i].reshape(shape, order="F") for d in range(2))
            Hb, Hc = (H[d][i].reshape(shape, order="F") for d in range(2))
            E_x_h = np.sum(
                Eb * np.conj(m.h_mean["c"]) * plane.dA_b
                - Ec * np.conj(m.h_mean["b"]) * plane.dA_c
            )
            e_x_H = np.sum(
                np.conj(m.e["b"]) * Hc * plane.dA_b
                - np.conj(m.e["c"]) * Hb * plane.dA_c
            )
            a_plus[i] = 0.25 * (E_x_h + e_x_H)
            a_minus[i] = 0.25 * (E_x_h - e_x_H)
        return a_plus, a_minus
