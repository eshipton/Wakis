# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Lumped circuit elements and discrete (lumped) ports for SolverFIT3D.

A lumped element occupies a box of E edges along one axis spanning the gap
between two conductors. Edges along the axis are in series, edges side by side
are in parallel, and the element is distributed over them as a uniform block
of conductivity (R), extra permittivity (C) and an inductive current (L):

    sigma = l / (R A),   eps_C = C l / A,   dJ_L/dt = l / (L A) E,

with l the gap length and A the cross-section of the box [1, 2]. The resistor
uses the time-averaged (semi-implicit) update, which is stable for any R [1].
A `LumpedPort` is a resistor Z0 in parallel with a soft current source (Norton
form of a voltage source with internal impedance Z0) that records the port
voltage and current for impedances and S-parameters.

Elements are passed to `SolverFIT3D.emsolve` in the `source` list. Their
`update(solver, t)` runs before every time step and `post_step(solver, t)`
right after it.

References
----------
[1] A. Taflove and S. C. Hagness, Computational Electrodynamics: The
    Finite-Difference Time-Domain Method, 3rd ed., Artech House, 2005.
[2] M. Piket-May, A. Taflove and J. Baron, "FD-TD modeling of digital signal
    propagation in 3-D circuits with passive and active loads," IEEE Trans.
    Microw. Theory Techn. 42(8), 1514-1523 (1994), doi:10.1109/22.297814.
[3] K. Kurokawa, "Power waves and the scattering matrix," IEEE Trans. Microw.
    Theory Techn. 13(2), 194-202 (1965).
"""

import copy

import numpy as np


class LumpedElement:
    def __init__(
        self,
        component="z",
        center=(0.0, 0.0, 0.0),
        size=(0.0, 0.0, 0.0),
        R=None,
        L=None,
        C=None,
    ):
        """
        Lumped resistor, inductor and/or capacitor (in parallel) across a gap.

        Parameters
        ----------
        component : str, optional
            Axis of the gap, 'x', 'y' or 'z'. Default is 'z'.
        center, size : tuple of float, optional
            Centre and extent (x, y, z) [m] of the box of E edges forming the
            element. Along `component` the box should span the gap exactly from
            one conductor to the other; transversally a zero size selects one
            line of edges and a finite size a sheet or block of parallel lines.
        R : float, optional
            Resistance [Ohm].
        L : float, optional
            Inductance [H].
        C : float, optional
            Capacitance [F].

        Notes
        -----
        The voltage across the element is V = -sum(E . dl) along `component`,
        i.e. the potential of the high end of the gap minus the low end, and the
        element current I flows through the element from the high to the low
        end, so that the element impedance is V / I. Elements must not overlap
        each other or a conductor. The update is linear in the fields; MPI and
        GPU paths are implemented but not covered by tests.
        """
        component = component[-1].lower()
        if component not in ("x", "y", "z"):
            raise ValueError(f"component must be 'x', 'y' or 'z', got {component!r}.")
        if any(s < 0 for s in size):
            raise ValueError("size must be non-negative.")
        for name, value in (("R", R), ("L", L), ("C", C)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0.")
        self.component = component
        self.center = np.asarray(center, dtype=float)
        self.size = np.asarray(size, dtype=float)
        self.R, self.L, self.C = R, L, C
        self.end_time = 0.0  # passive: no excitation to wait for
        self.is_first_update = True

    def _source_current(self, t):
        """Impressed (Norton) current at time t [A]; zero for passive elements."""
        return 0.0

    def _set_edges(self, solver):
        """Select the element edges and precompute the per-edge coefficients."""
        grid = solver.grid
        c = "xyz".index(self.component)
        indices, _ = grid.region_indices("E", self.component, self.center, self.size)
        # The last E sample along a non-periodic axis lies outside the domain
        if solver.bc_high[c].lower() != "periodic":
            n_c = (solver.Nx, solver.Ny, solver.Nz)[c]
            indices[c] = indices[c][indices[c] < n_c - 1]
        I, J, K = np.meshgrid(*indices, indexing="ij")
        N = solver.Nx * solver.Ny * solver.Nz
        flat = (c * N + I + solver.Nx * J + solver.Nx * solver.Ny * K).ravel()

        dl = grid.L.toarray()[flat]
        itA = grid.itA.toarray()[flat]
        dA = np.divide(1.0, itA, out=np.zeros_like(itA), where=itA != 0)
        if hasattr(solver, "_reopen_thin_wire_edges"):
            solver._reopen_thin_wire_edges(flat)
        ieps = solver.ieps.toarray()[flat]
        ieps = ieps.get() if hasattr(ieps, "get") else np.asarray(ieps)
        if np.any(ieps == 0) or np.any(dA == 0):
            raise ValueError("Lumped element overlaps a conductor or boundary cell.")

        # Gap length l (series) and cross-section A = volume / l (parallel)
        line_c = indices[c]
        dl_axis = np.diff((grid.x, grid.y, grid.z)[c])[line_c]
        length = float(np.sum(dl_axis))
        volume = float(np.sum(dl * dA))
        if solver.use_mpi:
            comm = grid.comm
            volume = comm.allreduce(volume)
            length = (
                comm.allreduce(length)
                if c == 2
                else comm.allreduce(length, op=_mpi_max())
            )
        if volume <= 0 or length <= 0:
            raise ValueError("Lumped element region contains no E edges.")
        area = volume / length

        eps = 1.0 / ieps
        eps_tot = eps + (self.C * length / area if self.C is not None else 0.0)
        sigma = length / (self.R * area) if self.R is not None else 0.0
        alpha = sigma * solver.dt / (2 * eps_tot)
        # Semi-implicit update E^{n+1} = a E^n + b (E* - E^n), see `post_step`
        coef_a = (1 - alpha) / (1 + alpha)
        coef_b = eps / (eps_tot * (1 + alpha))

        xp = solver.E.xp
        dtype = solver.E.dtype
        self.length, self.area = length, area
        self.flat_idx = xp.asarray(flat)
        self.coef_a = xp.asarray(coef_a, dtype=dtype)
        self.coef_b = xp.asarray(coef_b, dtype=dtype)
        # V = -sum(g E), weights g = dl dA / A average the parallel lines
        self.g = xp.asarray(dl * dA / area, dtype=dtype)
        self.gamma = (
            length / (self.L * area) if self.L is not None else None
        )  # [1/(H m)]
        self.J_L = xp.zeros(flat.size, dtype=dtype)
        self.J_old = xp.zeros(flat.size, dtype=dtype)
        self.E_n = xp.zeros(flat.size, dtype=dtype)
        self.t_rec, self.V_rec, self.I_L_rec, self.I_s_rec = [], [], [], []
        self._comm = grid.comm if solver.use_mpi else None

    def update(self, solver, t):
        """
        Pre-step update at time t = n dt: store E^n on the element edges and add
        the inductor current J_L^{n+1/2} and the source current to solver.J.

        Parameters
        ----------
        solver : SolverFIT3D
            Solver object.
        t : float
            Current simulation time [s].
        """
        if self.is_first_update:
            self._set_edges(solver)
            self.is_first_update = False

        self.E_n = solver.E.array[self.flat_idx].copy()
        if self.gamma is not None:
            # J_L^{n+1/2} = J_L^{n-1/2} + dt l/(L A) E^n  (L dI/dt = V)
            self.J_L = self.J_L + solver.dt * self.gamma * self.E_n
        # The source current is uniform over the cross-section A
        self.I_s = float(self._source_current(t + solver.dt / 2))
        J_new = self.J_L + self.I_s / self.area
        solver.J.array[self.flat_idx] += J_new - self.J_old
        self.J_old = J_new

    def post_step(self, solver, t):
        """
        Post-step update: apply the semi-implicit R/C correction to the element
        edges and record the voltage and currents at t + dt/2.

        The solver step has computed E* = E^n + dt/eps (curl H - J). With the
        element, eps_tot (E^{n+1}-E^n)/dt = curl H - J - sigma (E^{n+1}+E^n)/2,
        whose solution is E^{n+1} = a E^n + b (E* - E^n) with
        alpha = sigma dt / (2 eps_tot), a = (1-alpha)/(1+alpha) and
        b = eps / (eps_tot (1+alpha)).

        Parameters
        ----------
        solver : SolverFIT3D
            Solver object.
        t : float
            Simulation time at the start of the step [s].
        """
        E_star = solver.E.array[self.flat_idx]
        E_new = self.coef_a * self.E_n + self.coef_b * (E_star - self.E_n)
        solver.E.array[self.flat_idx] = E_new
        if solver.use_mpi:
            solver._mpi_communicate(solver.E)

        # Time-centred voltage (V^n + V^{n+1})/2 at t + dt/2 (local partial sum)
        self.t_rec.append(t + solver.dt / 2)
        self.V_rec.append(-float(solver.E.xp.sum(self.g * (self.E_n + E_new))) / 2)
        self.I_s_rec.append(self.I_s)
        if self.gamma is not None:
            # J_L^{n+1/2} along +component; element current flows the other way
            self.I_L_rec.append(
                -float(solver.E.xp.sum(self.g * self.J_L)) * self.area / self.length
            )

    def _reduce(self, values):
        values = np.asarray(values, dtype=float)
        if self._comm is not None:
            values = self._comm.allreduce(values)
        return values

    def voltage(self, freqs):
        """
        Fourier transform of the element voltage.

        V(f) = sum_n Vbar^{n+1/2} exp(-2j pi f t_{n+1/2}) dt, with the
        time-centred voltage Vbar^{n+1/2} = -sum_e g_e (E_e^n + E_e^{n+1}) / 2.

        Parameters
        ----------
        freqs : array_like
            Frequencies [Hz].

        Returns
        -------
        ndarray of complex
            Voltage spectrum [V s].
        """
        return _dtft(self._reduce(self.V_rec), self.t_rec, freqs)

    def current(self, freqs):
        """
        Fourier transform of the current flowing through the element from the
        high to the low end of the gap: I = V/R + I_L + j 2 pi f C V.

        The capacitive current uses the continuous j 2 pi f C; the discrete
        one differs by O((2 pi f dt)^2).

        Parameters
        ----------
        freqs : array_like
            Frequencies [Hz].

        Returns
        -------
        ndarray of complex
            Current spectrum [A s].
        """
        freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
        V = self.voltage(freqs)
        I = np.zeros_like(V)
        if self.R is not None:
            I += V / self.R
        if self.L is not None:
            I += _dtft(self._reduce(self.I_L_rec), self.t_rec, freqs)
        if self.C is not None:
            I += 2j * np.pi * freqs * self.C * V
        return I

    def clone(self):
        """Return an unused copy of this element (for repeated simulations)."""
        if not self.is_first_update:
            raise RuntimeError(
                "Only an element that has not been simulated can be cloned."
            )
        return copy.deepcopy(self)


class LumpedPort(LumpedElement):
    def __init__(
        self,
        src_time=None,
        component="z",
        center=(0.0, 0.0, 0.0),
        size=(0.0, 0.0, 0.0),
        impedance=50.0,
        amplitude=1.0,
    ):
        """
        Discrete (lumped) port: voltage source with internal impedance across a
        gap, for input impedances and S-parameters.

        The source is modelled in Norton form, a soft current source
        I_s(t) = amplitude * s(t) / impedance in parallel with a resistor
        R = impedance, equivalent to an ideal voltage source of open-circuit
        voltage amplitude * s(t) in series with `impedance`. The resistor
        absorbs the waves returning to the port, so a passive port
        (`src_time=None`) is a matched termination.

        Parameters
        ----------
        src_time : GaussianSource, ContinuousSource, CustomSource or None
            Time profile s(t). None makes a passive (terminating) port.
        component : str, optional
            Axis of the gap, 'x', 'y' or 'z'. The positive terminal is at the
            high-coordinate end of the gap. Default is 'z'.
        center, size : tuple of float, optional
            Centre and extent (x, y, z) [m] of the gap, see `LumpedElement`.
        impedance : float, optional
            Port reference impedance Z0 [Ohm]. Default is 50.
        amplitude : float, optional
            Open-circuit source voltage amplitude [V]. Default is 1.

        Notes
        -----
        The port voltage V and the current I delivered by the port into the
        structure (out of the positive terminal) are recorded at the half time
        steps, I = I_s - V / Z0. From their Fourier transforms:
        Zin = V / I, a = (V + Z0 I) / (2 sqrt(Z0)), b = (V - Z0 I) / (2 sqrt(Z0)),
        S11 = b / a (power waves [3]). The displacement current of the gap
        cell belongs to the structure, which makes Zin second-order accurate.
        """
        if impedance <= 0:
            raise ValueError("impedance must be > 0.")
        super().__init__(component=component, center=center, size=size, R=impedance)
        self.src_time = src_time
        self.impedance = impedance
        self.amplitude = amplitude
        self.end_time = (
            0.0 if src_time is None else getattr(src_time, "end_time", np.inf)
        )

    def _source_current(self, t):
        if self.src_time is None:
            return 0.0
        return self.amplitude * self.src_time(t) / self.impedance

    def port_current(self, freqs):
        """
        Fourier transform of the current into the structure,
        I(f) = I_s(f) - V(f) / Z0, using the same discrete samples as the
        update so that charge is conserved exactly.

        Parameters
        ----------
        freqs : array_like
            Frequencies [Hz].

        Returns
        -------
        ndarray of complex
            Current spectrum [A s].
        """
        I_s = _dtft(np.asarray(self.I_s_rec, dtype=float), self.t_rec, freqs)
        return I_s - self.voltage(freqs) / self.impedance

    def impedance_in(self, freqs):
        """
        Input impedance Zin(f) = V(f) / I(f) [Ohm] seen by the port.

        Parameters
        ----------
        freqs : array_like
            Frequencies [Hz].

        Returns
        -------
        ndarray of complex
        """
        return self.voltage(freqs) / self.port_current(freqs)

    def power_waves(self, freqs):
        """
        Incident and reflected power waves (Kurokawa [3]) for the real
        reference impedance Z0: a = (V + Z0 I)/(2 sqrt(Z0)),
        b = (V - Z0 I)/(2 sqrt(Z0)).

        Parameters
        ----------
        freqs : array_like
            Frequencies [Hz].

        Returns
        -------
        a, b : ndarray of complex
            Power waves [sqrt(W) s].
        """
        V = self.voltage(freqs)
        I = self.port_current(freqs)
        root = 2 * np.sqrt(self.impedance)
        return (V + self.impedance * I) / root, (V - self.impedance * I) / root

    def s11(self, freqs):
        """
        Reflection coefficient S11(f) = b/a = (Zin - Z0)/(Zin + Z0).

        Parameters
        ----------
        freqs : array_like
            Frequencies [Hz].

        Returns
        -------
        ndarray of complex
        """
        a, b = self.power_waves(freqs)
        return b / a


def _dtft(signal, t, freqs):
    """Riemann-sum Fourier transform X(f) = sum_n x_n exp(-2j pi f t_n) dt."""
    t = np.asarray(t, dtype=float)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    if t.size < 2:
        raise RuntimeError("The element has not been simulated yet.")
    kernel = np.exp(-2j * np.pi * np.outer(freqs, t))
    return kernel @ np.asarray(signal, dtype=float) * (t[1] - t[0])


def _mpi_max():
    """MPI.MAX reduction operator (imported lazily, mpi4py is optional)."""
    from mpi4py import MPI

    return MPI.MAX
