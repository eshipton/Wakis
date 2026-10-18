# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #


from typing import Sequence

import numpy as np

from wakis.field import Field


class FieldMonitor:
    """Accumulate time-domain fields into frequency-domain components.

    The monitor sums snapshots of the three electric-field components with a
    complex exponential kernel for each monitored frequency, producing
    frequency-domain accumulators compatible with either NumPy or CuPy as the
    array backend. Use :meth:`update` to feed time-domain ``Field`` snapshots
    and :meth:`get_components` to retrieve the accumulated spectra.

    Attributes
    ----------
    frequencies : numpy.ndarray
        Array of monitored frequencies in Hz.
    time_index : int
        Current time-step counter (increments on every :meth:`update` call).
    dt : float or None
        Time-step size (set on first :meth:`update` call).
    Ex_acc, Ey_acc, Ez_acc : array or None
        Complex accumulators with shape ``(n_freq, Nx, Ny, Nz)`` storing the
        frequency-domain sums for each field component.
    shape : tuple or None
        Spatial shape of the monitored fields as ``(Nx, Ny, Nz)``.
    xp : module or None
        Array library used for storage and computation (``numpy`` or ``cupy``).
    """

    def __init__(self, frequencies: Sequence[float]):
        """Create a new FieldMonitor.

        Parameters
        ----------
        frequencies : sequence of float
            Frequencies (in Hz) where the field will be monitored. Provide a
            sequence (list or array) of values; they are stored as a
            one-dimensional NumPy array.
        """
        self.frequencies = np.array(frequencies)
        self.time_index = 0
        self.dt = None
        self.Ex_acc = None
        self.Ey_acc = None
        self.Ez_acc = None
        self.shape = None
        self.xp = None

    def update(self, E: Field, dt: float):
        """Accumulate a time-domain ``Field`` snapshot into the monitors.

        On the first call this method initializes internal arrays and records
        the time-step ``dt``. For each monitored frequency f it multiplies the
        spatial field snapshot by exp(-2j*pi*f*t) and adds the result to the
        corresponding complex accumulator. Time ``t`` is computed as
        ``time_index * dt`` before incrementing ``time_index``.

        Parameters
        ----------
        E : Field
            Electric-field container providing ``Nx``, ``Ny``, ``Nz``, the
            backend ``xp`` (numpy or cupy) and a ``to_matrix(component)``
            method that returns a Cartesian component as a dense array.
        dt : float
            Time-step size (s). If ``dt`` was not previously set it is stored
            internally and used for subsequent updates.
        """
        if self.dt is None:
            self.dt = dt
            self.shape = (E.Nx, E.Ny, E.Nz)
            self.xp = E.xp

            n_freqs = len(self.frequencies)
            shape = (n_freqs, *self.shape)
            self.Ex_acc = self.xp.zeros(shape, dtype=self.xp.complex128)
            self.Ey_acc = self.xp.zeros(shape, dtype=self.xp.complex128)
            self.Ez_acc = self.xp.zeros(shape, dtype=self.xp.complex128)
            t = self.time_index * self.dt

        t = self.time_index * self.dt

        Ex = E.to_matrix("x")  # E.array[0:E.N]
        Ey = E.to_matrix("y")
        Ez = E.to_matrix("z")

        for i, f in enumerate(self.frequencies):
            phase = self.xp.exp(-2j * self.xp.pi * f * t)
            self.Ex_acc[i] += Ex * phase
            self.Ey_acc[i] += Ey * phase
            self.Ez_acc[i] += Ez * phase

        self.time_index += 1

    def get_components(self):
        """Return the accumulated frequency-domain components.

        Returns
        -------
        dict
            Dictionary with keys ``'Ex'``, ``'Ey'``, ``'Ez'`` mapping to the
            respective complex accumulator arrays. Each array has shape
            ``(n_freq, Nx, Ny, Nz)`` and uses the configured backend dtype
            (complex128).
        """

        return {"Ex": self.Ex_acc, "Ey": self.Ey_acc, "Ez": self.Ez_acc}


def _flat_index(solver, component, I, J, K):
    """Lexicographic index into Field.array (Fortran order, components stacked)."""
    N = solver.Nx * solver.Ny * solver.Nz
    c = "xyz".index(component)
    return c * N + I + solver.Nx * J + solver.Nx * solver.Ny * K


class DFTMonitor:
    def __init__(
        self,
        frequencies,
        components=("Ex", "Ey", "Ez"),
        center=(0.0, 0.0, 0.0),
        size=(0.0, 0.0, 0.0),
    ):
        """
        Running discrete Fourier transform of field components in a region.

        Each component is sampled at its native location on the staggered grid
        and at its own time level: after the step started at time t, E (and J)
        are known at t + dt and H at t + dt/2. The transform approximates the
        continuous Fourier transform F(f) = int F(t) exp(-2j*pi*f*t) dt, with
        the same convention as `GaussianSource.fourier_transform`, so field
        spectra can be normalized directly by the source spectrum.

        Parameters
        ----------
        frequencies : array_like
            Frequencies to monitor [Hz].
        components : sequence of str, optional
            Field components, e.g. ('Ex', 'Hy', 'Jz'). Default is E.
        center, size : tuple of float, optional
            Centre and extent of the monitored region [m]. A zero size selects
            the nearest sample along that direction. Default is a point at the
            origin.

        Notes
        -----
        Call `update(solver, t)` after every `solver.one_step()` with the time
        t at which that step started, or pass the monitor to `solver.emsolve`.
        """
        for comp in components:
            if len(comp) != 2 or comp[0] not in "EHJ" or comp[1] not in "xyz":
                raise ValueError(f"Invalid component {comp!r}, expected e.g. 'Ex'.")
        self.frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        self.components = tuple(components)
        self.center = np.asarray(center, dtype=float)
        self.size = np.asarray(size, dtype=float)
        self.is_first_update = True

    def _initialize(self, solver):
        xp = solver.E.xp
        self.xp = xp
        # On the field device: the DFT kernels are then computed there,
        # without a host-to-device copy (and GPU synchronization) per step
        self._freqs = xp.asarray(self.frequencies)
        self.coords, self.shape, self._flat, self._acc = {}, {}, {}, {}
        for comp in self.components:
            indices, coords = solver.grid.region_indices(
                comp[0], comp[1], self.center, self.size
            )
            I, J, K = np.meshgrid(*indices, indexing="ij")
            self.coords[comp] = coords
            self.shape[comp] = I.shape
            self._flat[comp] = xp.asarray(_flat_index(solver, comp[1], I, J, K).ravel())
            self._acc[comp] = xp.zeros(
                (self.frequencies.size, I.size), dtype=xp.complex128
            )

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

        times = {"E": t + solver.dt, "J": t + solver.dt, "H": t + solver.dt / 2}
        kernels = {
            f: solver.dt * self.xp.exp(-2j * np.pi * self._freqs * times[f])[:, None]
            for f in {comp[0] for comp in self.components}
        }
        for comp in self.components:
            values = getattr(solver, comp[0]).array[self._flat[comp]]
            self._acc[comp] += kernels[comp[0]] * values[None, :]

    def get_field(self, component):
        """
        Fourier-transformed field component in the monitored region.

        Parameters
        ----------
        component : str
            One of the monitored components, e.g. 'Ez'.

        Returns
        -------
        field : ndarray of complex
            Array of shape (n_freq, nx, ny, nz) [field units x s].
        coords : list of 3 ndarray
            Sample coordinates [m] along x, y and z.

        Notes
        -----
        With MPI, each rank returns the samples inside its own subdomain.
        """
        acc = self._acc[component]
        acc = acc.get() if hasattr(acc, "get") else acc
        return (
            acc.reshape(self.frequencies.size, *self.shape[component]),
            self.coords[component],
        )


class FluxMonitor:
    def __init__(self, frequencies, center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0)):
        """
        Spectrum of the Poynting flux through a plane or out of a closed box.

        The flux spectrum is Phi(f) = Re sum (E(f) x H(f)^*) . n dA, where E(f)
        and H(f) are the Fourier transforms accumulated as in `DFTMonitor`. By
        Parseval's theorem the energy crossing the surface is
        2 * int_0^inf Phi(f) df [J]. For a time-harmonic excitation of complex
        amplitude S(f), the time-averaged power is P(f) = Phi(f) / (2 |S(f)|^2).
        With complex fields (Bloch boundaries) Phi(f) and Phi(-f) are
        independent, the energy is int_-inf^inf Phi(f) df, and each sign of f
        is a separate physical wave (see the physics guide).

        The tangential E components are taken on the grid plane nearest to the
        requested position, and H, staggered half a cell away, is averaged from
        both sides onto that plane. The area elements are the primary edge
        length of each E component times the dual width in the other direction.

        Parameters
        ----------
        frequencies : array_like
            Frequencies to monitor [Hz].
        center : tuple of float, optional
            Centre of the plane or box [m].
        size : tuple of float, optional
            Extent [m]. Exactly one zero entry defines a plane with normal along
            that axis and flux counted positive in the +axis direction. No zero
            entries define a box with flux counted positive outwards.
        """
        self.frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        self.center = np.asarray(center, dtype=float)
        self.size = np.asarray(size, dtype=float)
        zero_axes = np.flatnonzero(self.size == 0)
        if zero_axes.size > 1:
            raise ValueError("size must have at most one zero entry (plane or box).")

        if zero_axes.size == 1:
            self.faces = [(zero_axes[0], self.center, 1.0)]
        else:
            self.faces = []
            for a in range(3):
                for sign in (-1.0, 1.0):
                    face_center = self.center.copy()
                    face_center[a] += sign * self.size[a] / 2
                    self.faces.append((a, face_center, sign))
        self.is_first_update = True

    def _initialize(self, solver):
        xp = solver.E.xp
        n_freq = self.frequencies.size
        self._terms = []
        for a, face_center, sign in self.faces:
            face_size = self.size.copy()
            face_size[a] = 0.0
            b, c = (a + 1) % 3, (a + 2) % 3
            # S_a = E_b H_c - E_c H_b; each pair is colocated within the plane
            for e_comp, h_comp, term_sign in ((b, c, 1.0), (c, b, -1.0)):
                e_name, h_name = "xyz"[e_comp], "xyz"[h_comp]
                indices, coords = solver.grid.region_indices(
                    "E", e_name, face_center, face_size
                )
                if indices[a].size and indices[a][0] == 0:
                    raise ValueError("Flux plane cannot lie on the lower boundary.")
                I, J, K = np.meshgrid(*indices, indexing="ij")
                below = [I, J, K]
                below[a] = below[a] - 1
                dA = solver.grid.L[I, J, K, e_name] * solver.grid.tL[I, J, K, h_name]
                if len(self.faces) > 1:
                    # Samples on the box edges are shared by two faces: trapezoid
                    # rule, half of their dual cell lies outside each face
                    lo = face_center - face_size / 2
                    hi = face_center + face_size / 2
                    grid = solver.grid
                    tol = 1e-6 * min(np.min(grid.dx), np.min(grid.dy), np.min(grid.dz))
                    for d in (b, c):
                        on_edge = (np.abs(coords[d] - lo[d]) < tol) | (
                            np.abs(coords[d] - hi[d]) < tol
                        )
                        shape = [1, 1, 1]
                        shape[d] = -1
                        dA = dA * np.where(on_edge, 0.5, 1.0).reshape(shape)
                dA = dA.ravel()
                self._terms.append(
                    {
                        "E": xp.asarray(_flat_index(solver, e_name, I, J, K).ravel()),
                        "H_above": xp.asarray(
                            _flat_index(solver, h_name, I, J, K).ravel()
                        ),
                        "H_below": xp.asarray(
                            _flat_index(solver, h_name, *below).ravel()
                        ),
                        "weight": sign * term_sign * dA,
                        "coords": coords,
                        "axes": (e_comp, h_comp),
                        "E_acc": xp.zeros((n_freq, dA.size), dtype=xp.complex128),
                        "H_acc": xp.zeros((n_freq, dA.size), dtype=xp.complex128),
                    }
                )
        self.xp = xp
        self._freqs = xp.asarray(self.frequencies)  # see DFTMonitor._initialize
        self.comm = solver.grid.comm if solver.use_mpi else None

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

        phase = -2j * np.pi * self._freqs
        kernel_E = solver.dt * self.xp.exp(phase * (t + solver.dt))[:, None]
        kernel_H = solver.dt * self.xp.exp(phase * (t + solver.dt / 2))[:, None]
        for term in self._terms:
            E = solver.E.array[term["E"]]
            H = 0.5 * (
                solver.H.array[term["H_above"]] + solver.H.array[term["H_below"]]
            )
            term["E_acc"] += kernel_E * E[None, :]
            term["H_acc"] += kernel_H * H[None, :]

    def get_flux(self):
        """
        Flux spectrum through the monitored surface.

        Returns
        -------
        ndarray
            Phi(f) for each monitored frequency [J/Hz^2 = W s^2].
        """
        flux = np.zeros(self.frequencies.size)
        for term in self._terms:
            density = self.xp.real(term["E_acc"] * self.xp.conj(term["H_acc"]))
            density = density.get() if hasattr(density, "get") else density
            flux += density @ term["weight"]
        if self.comm is not None:
            # Each MPI rank holds the part of the surface in its subdomain
            flux = self.comm.allreduce(flux)
        return flux
