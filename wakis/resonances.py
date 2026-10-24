# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Resonance extraction by harmonic inversion of time signals.

A field probe recorded after the excitation has ended is a sum of damped
complex exponentials (the resonances of the structure in the probed band),

    x(t) = sum_k a_k exp(2j*pi*F_k*t),   F_k = f_k + 1j*g_k,

with the convention of the rest of Wakis (phasors exp(+j*omega*t)). A resonance
decaying in time has Im F_k = g_k > 0: its field amplitude decays as
exp(-2*pi*g_k*t), its energy as exp(-omega_k t / Q_k), so that

    Q_k = f_k / (2 g_k).

`harminv` fits this model to a short, uniformly sampled signal in a frequency
window with the filter diagonalization method (FDM) of Wall & Neuhauser [1]
in the form of Mandelshtam & Taylor [2], or with the matrix pencil method of
Hua & Sarkar [3]. Both resolve modes closer than the Fourier limit 1/T of the
record length T when the signal-to-noise ratio is high. `ResonanceMonitor`
records a field component during `SolverFIT3D.emsolve` and calls `harminv`.

References
----------
[1] M. R. Wall and D. Neuhauser, "Extraction, through filter-diagonalization,
    of general quantum eigenvalues or classical normal mode frequencies from a
    small number of residues or a short-time segment of a signal. I. Theory and
    application to a quantum-dynamics model," J. Chem. Phys. 102, 8011-8022
    (1995), doi:10.1063/1.468999.
[2] V. A. Mandelshtam and H. S. Taylor, "Harmonic inversion of time signals and
    its applications," J. Chem. Phys. 107, 6756-6769 (1997),
    doi:10.1063/1.475324.
[3] Y. Hua and T. K. Sarkar, "Matrix pencil method for estimating parameters of
    exponentially damped/undamped sinusoids in noise," IEEE Trans. Acoust.,
    Speech, Signal Process. 38(5), 814-824 (1990), doi:10.1109/29.56027.
[4] D. C. Rife and R. R. Boorstyn, "Single-tone parameter estimation from
    discrete-time observations," IEEE Trans. Inf. Theory 20(5), 591-598
    (1974), doi:10.1109/TIT.1974.1055282.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Resonances:
    """
    Resonances found by `harminv`, sorted by frequency.

    Attributes
    ----------
    complex_frequency : ndarray of complex
        F_k = f_k + 1j*g_k [Hz] in x(t) = sum_k a_k exp(2j*pi*F_k*t). Decaying
        modes have g_k > 0.
    amplitude : ndarray of complex
        a_k, complex amplitude at t = 0 [signal units].
    error : ndarray of float
        Estimated absolute error of F_k [Hz] (see `harminv`).
    """

    complex_frequency: np.ndarray
    amplitude: np.ndarray
    error: np.ndarray

    @property
    def frequency(self):
        """Resonance frequencies f_k = Re F_k [Hz]."""
        return self.complex_frequency.real

    @property
    def decay_rate(self):
        """Amplitude decay rates 2*pi*Im F_k [1/s]."""
        return 2 * np.pi * self.complex_frequency.imag

    @property
    def Q(self):
        """Quality factors Q_k = f_k / (2 Im F_k) (inf for undamped modes)."""
        with np.errstate(divide="ignore"):
            return self.frequency / (2 * self.complex_frequency.imag)

    def __len__(self):
        return self.complex_frequency.size

    def __getitem__(self, index):
        return Resonances(
            np.atleast_1d(self.complex_frequency[index]),
            np.atleast_1d(self.amplitude[index]),
            np.atleast_1d(self.error[index]),
        )


def _fdm(c, dt, fmin, fmax, density, n_basis, rcond):
    """
    Filter diagonalization [1, 2]: eigenvalues u_k = exp(2j*pi*F_k*dt) of the
    signal propagator in a basis of K windowed Fourier vectors.

    With c_n = sum_k d_k u_k^n (n = 0..2M+2) and basis z_j = exp(2j*pi*phi_j*dt)
    on the unit circle, the matrices U^(p)_jj' = sum_{n,m=0}^M z_j^-n z_j'^-m
    c_{n+m+p} (p = 0, 1, 2) have closed forms in O(K N) operations [2, Eqs.
    (24)-(25)]. The generalized eigenproblem U^(1) B = u U^(0) B gives u_k, the
    amplitudes are d_k = (B_k^T C)^2 with C_j = sum_n c_n z_j^-n and B_k
    normalized by B_k^T U^(0) B_k = 1, and the error estimate compares u_k^2
    with B_k^T U^(2) B_k [2].
    """
    M = (c.size - 3) // 2
    if M < 2:
        raise ValueError("The signal needs at least 7 samples.")
    if n_basis is None:
        # Nyquist spacing of the window is 1/((M+1) dt): basis vectors closer
        # than that are linearly dependent
        n_basis = int(np.ceil(density * (fmax - fmin) * (M + 1) * dt)) + 1
    n_basis = max(int(n_basis), 2)
    phi = np.linspace(fmin, fmax, n_basis)
    z = np.exp(2j * np.pi * phi * dt)

    k = np.arange(2 * M + 1)
    weights = M - np.abs(M - k) + 1
    A = np.empty((3, n_basis), complex)  # sum_{k=0}^{M} c_{k+p} z^-k
    B = np.empty((3, n_basis), complex)  # sum_{k=M+1}^{2M} c_{k+p} z^{M+1-k}
    D = np.empty((3, n_basis), complex)  # sum_{k=0}^{2M} w_k c_{k+p} z^-k
    for j in range(n_basis):
        zk = np.exp(-2j * np.pi * phi[j] * dt * k)  # z_j^-k without power drift
        for p in range(3):
            cp = c[p : p + 2 * M + 1]
            A[p, j] = cp[: M + 1] @ zk[: M + 1]
            B[p, j] = (cp[M + 1 :] @ zk[M + 1 :]) * z[j] ** (M + 1)
            D[p, j] = (weights * cp) @ zk

    zM = z ** (-M)
    U = np.empty((3, n_basis, n_basis), complex)
    zi, zj = z[:, None], z[None, :]
    off = ~np.eye(n_basis, dtype=bool)
    for p in range(3):
        num = (
            zi * A[p][None, :]
            - zj * A[p][:, None]
            + zM[None, :] * B[p][:, None]
            - zM[:, None] * B[p][None, :]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            U[p] = num / (zi - zj)
        U[p][~off] = D[p]

    # Regularize the (often singular) overlap matrix: solve the eigenproblem in
    # the subspace of its significant singular vectors
    W, s, Vh = np.linalg.svd(U[0])
    r = int(np.sum(s > rcond * s[0]))
    W, s, V = W[:, :r], s[:r], Vh[:r].conj().T
    u, y = np.linalg.eig((W.conj().T @ U[1] @ V) / s[:, None])
    vec = V @ y
    norm = np.einsum("jk,jl,lk->k", vec, U[0], vec)
    vec = vec / np.sqrt(norm)
    amplitude = (vec.T @ A[0]) ** 2
    u2 = np.einsum("jk,jl,lk->k", vec, U[2], vec)
    # u^2 = exp(4j*pi*F*dt): a deviation du^2 means dF = du^2 / (4*pi*dt*u^2)
    error = np.abs(u2 / u**2 - 1) / (4 * np.pi * dt)
    return u, amplitude, error


def _matrix_pencil(c, dt, rcond, pencil):
    """
    Matrix pencil method [3]: with the Hankel matrix Y_jl = c_{j+l} (L+1
    columns), the shifted blocks Y1 (first L columns) and Y2 (last L columns)
    satisfy Y2 = Y1 diag(u) in the signal subspace. The rank is set by the
    singular values above rcond times the largest, the u_k are the eigenvalues
    of the reduced pencil and the amplitudes come from a least-squares fit.
    """
    N = c.size
    L = int(pencil * N) if pencil < 1 else int(pencil)
    L = min(max(L, 2), N - 2)
    rows = N - L
    Y = np.lib.stride_tricks.sliding_window_view(c, L + 1)[:rows]
    _, s, Vh = np.linalg.svd(Y, full_matrices=False)
    r = int(np.sum(s > rcond * s[0]))
    # Rows of Y are combinations of the vectors (u_k^l)_l, and so are the rows
    # of Vh (not conjugated): V = Z T, hence V2 = V1 T^-1 diag(u) T
    V = Vh[:r].T
    u = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:])

    n = np.arange(N)
    Z = u[None, :] ** n[:, None]
    amplitude, *_ = np.linalg.lstsq(Z, c, rcond=None)
    # Error: Cramer-Rao bound of a single undamped tone in white noise [4] with
    # the fit residual as the noise, sqrt(6) sigma / (2 pi |a| N dt sqrt(N))
    sigma = np.linalg.norm(c - Z @ amplitude) / np.sqrt(N)
    tiny = np.finfo(float).tiny
    error = np.sqrt(6) * sigma / (2 * np.pi * np.maximum(np.abs(amplitude), tiny))
    error = error / (N * dt * np.sqrt(N))
    return u, amplitude, error


def harminv(
    signal,
    dt,
    fmin,
    fmax,
    t0=0.0,
    method="fdm",
    density=1.1,
    n_basis=None,
    rcond=1e-10,
    pencil=0.4,
    max_rel_error=1e-3,
    min_rel_amplitude=1e-6,
):
    """
    Extract complex resonance frequencies from a time signal in a band.

    Fits x(t_n) = sum_k a_k exp(2j*pi*F_k*t_n), t_n = t0 + n*dt, with
    F_k = f_k + 1j*g_k (g_k > 0 for decaying modes, Q_k = f_k / (2 g_k)), and
    returns the modes with fmin <= f_k <= fmax.

    Parameters
    ----------
    signal : array_like of float or complex
        Uniformly sampled signal x(t_n). Real signals contain each resonance
        at +f_k and -f_k; only the positive frequency in the band is returned
        and its amplitude is a_k (the real signal is 2 Re[a_k exp(...)]).
    dt : float
        Sampling interval [s]. The band must lie below the Nyquist frequency
        1/(2 dt).
    fmin, fmax : float
        Frequency window [Hz].
    t0 : float, optional
        Time of the first sample [s]; amplitudes are referred to t = 0.
    method : {'fdm', 'pencil'}, optional
        Filter diagonalization (default; cost O(N K) for N samples and K basis
        functions, suited to long records) or matrix pencil (cost O(N^3); use
        for short records, e.g. after decimation).
    density : float, optional
        FDM basis functions per Fourier bin 1/(T/2) in the window. Default 1.1.
    n_basis : int, optional
        FDM number of basis functions (overrides `density`).
    rcond : float, optional
        Relative singular-value cutoff of the overlap (FDM) or Hankel (pencil)
        matrix, of the order of the relative noise level. Default 1e-10.
    pencil : float, optional
        Pencil parameter L as a fraction of the number of samples (N/3 to N/2
        is optimal for noisy data [3]). Default 0.4.
    max_rel_error : float or None, optional
        Discard modes whose estimated error exceeds this fraction of the window
        width (spurious fits of noise). Default 1e-3.
    min_rel_amplitude : float or None, optional
        Discard modes with |a_k| below this fraction of the largest amplitude in
        the window. Default 1e-6.

    Returns
    -------
    Resonances
        Complex frequencies [Hz], amplitudes and error estimates [Hz].

    Notes
    -----
    FDM [1, 2]: the error estimate is |u_k^(2) / u_k^2 - 1| / (4 pi dt), the
    inconsistency between the propagator eigenvalue and its square obtained
    from the twice-shifted signal; it is close to round-off for an exact sum of
    exponentials and grows with noise and with modes just outside the window.
    Pencil [3]: the error is the Cramer-Rao bound of a single undamped tone [4]
    with the fit residual as noise, sqrt(6) rms / (2 pi |a_k| N dt sqrt(N)), a
    rough guide for well-separated modes. Only the FDM error is validated in
    the tests.
    """
    x = np.asarray(signal)
    x = x.astype(complex)
    fnyq = 1 / (2 * dt)
    if not (-fnyq < fmin < fmax < fnyq):
        raise ValueError("Need -1/(2 dt) < fmin < fmax < 1/(2 dt).")
    if method == "fdm":
        # Basis slightly wider than the window keeps edge modes accurate
        margin = 0.1 * (fmax - fmin)
        lo, hi = max(fmin - margin, -fnyq), min(fmax + margin, fnyq)
        u, amp, err = _fdm(x, dt, lo, hi, density, n_basis, rcond)
    elif method == "pencil":
        u, amp, err = _matrix_pencil(x, dt, rcond, pencil)
    else:
        raise ValueError(f"Unknown method {method!r}, expected 'fdm' or 'pencil'.")

    F = np.log(u.astype(complex)) / (2j * np.pi * dt)
    amp = amp * np.exp(-2j * np.pi * F * t0)
    keep = (F.real >= fmin) & (F.real <= fmax) & np.isfinite(F)
    if max_rel_error is not None:
        keep &= err <= max_rel_error * (fmax - fmin)
    if min_rel_amplitude is not None and np.any(keep):
        keep &= np.abs(amp) >= min_rel_amplitude * np.abs(amp[keep]).max()
    order = np.argsort(F.real[keep])
    return Resonances(F[keep][order], amp[keep][order], err[keep][order])


class ResonanceMonitor:
    def __init__(
        self, component="Ez", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0), t_start=0.0
    ):
        """
        Record a field component during a simulation for resonance extraction.

        Pass the monitor to `SolverFIT3D.emsolve(monitors=[...])` and call
        `harminv` afterwards. Start recording after the source has ended
        (`t_start = src_time.end_time`) so that the signal is a free ring-down,
        a pure sum of decaying resonances.

        Parameters
        ----------
        component : str, optional
            Field component, e.g. 'Ez' or 'Hy'. Default 'Ez'.
        center, size : tuple of float, optional
            Probe position and extent [m]. A nonzero size records the sum of
            the samples in the region (fewer modes are missed at nodal points).
            Default is a point at the origin.
        t_start : float, optional
            Samples at times before t_start are not recorded [s]. Default 0.

        Notes
        -----
        Samples are taken at their native time level: after the step that
        started at t, E is at t + dt and H at t + dt/2 (as in `DFTMonitor`).
        With MPI, ranks that do not own the probe record zeros and `signal`
        sums over ranks.
        """
        if (
            len(component) != 2
            or component[0] not in "EHJ"
            or component[1] not in "xyz"
        ):
            raise ValueError(f"Invalid component {component!r}, expected e.g. 'Ez'.")
        self.component = component
        self.center = np.asarray(center, dtype=float)
        self.size = np.asarray(size, dtype=float)
        self.t_start = t_start
        self._values, self._t_first = [], None
        self._flat = None

    def update(self, solver, t=None):
        """
        Record the probe after the timestep that started at time t.

        Parameters
        ----------
        solver : SolverFIT3D or SubgridFIT2D
            Solver object.
        t : float, optional
            Time at the start of the step just taken [s].
        """
        if t is None:
            t = (solver.n - 1) * float(solver.dt)
        if hasattr(solver, "interpolation") and hasattr(solver, "edge_field_matrices"):
            if self._flat is None:
                c = self.center[:2]
                s = self.size[:2]
                if np.all(s == 0):
                    pts = c[None, :]
                else:
                    nx = 5 if s[0] > 0 else 1
                    ny = 5 if s[1] > 0 else 1
                    xs = np.linspace(c[0] - 0.5 * s[0], c[0] + 0.5 * s[0], nx)
                    ys = np.linspace(c[1] - 0.5 * s[1], c[1] + 0.5 * s[1], ny)
                    X, Y = np.meshgrid(xs, ys, indexing="ij")
                    pts = np.column_stack([X.ravel(), Y.ravel()])
                comp = self.component
                if comp in ("Ez", "Hz"):
                    S = solver.interpolation(pts)
                    self._subgrid_w = np.asarray(S.sum(axis=0)).ravel()
                    self._subgrid_kind = "u"
                else:
                    Mx, My = solver.edge_field_matrices(pts)
                    if comp == "Ex":
                        M = Mx
                    elif comp == "Ey":
                        M = My
                    elif comp == "Hx":
                        M = -Mx
                    elif comp == "Hy":
                        M = -My
                    else:
                        raise ValueError(f"Unsupported 2D component {comp!r}.")
                    self._subgrid_w = np.asarray(M.sum(axis=0)).ravel()
                    self._subgrid_kind = "g"
                self._flat = True
                self.dt = float(solver.dt)
                self._comm = None
            t_sample = t + (
                solver.dt if self._subgrid_kind == "u" else 0.5 * solver.dt
            )
            if t_sample < self.t_start:
                return
            if self._t_first is None:
                self._t_first = float(t_sample)
            vec = solver.u if self._subgrid_kind == "u" else solver.g
            self._values.append(self._subgrid_w @ vec)
            return

        if self._flat is None:
            from wakis.field_monitors import _flat_index

            indices, _ = solver.grid.region_indices(
                self.component[0], self.component[1], self.center, self.size
            )
            I, J, K = np.meshgrid(*indices, indexing="ij")
            flat = _flat_index(solver, self.component[1], I, J, K).ravel()
            self._flat = solver.E.xp.asarray(flat)
            self.dt = float(solver.dt)
            self._comm = solver.grid.comm if solver.use_mpi else None
        t_sample = t + (solver.dt / 2 if self.component[0] == "H" else solver.dt)
        if t_sample < self.t_start:
            return
        if self._t_first is None:
            self._t_first = float(t_sample)
        field = getattr(solver, self.component[0]).array
        self._values.append(field[self._flat].sum())

    def signal(self):
        """
        Recorded time signal.

        Returns
        -------
        t : ndarray
            Sample times [s].
        x : ndarray
            Recorded values [field units].
        """
        x = np.array([v.get() if hasattr(v, "get") else v for v in self._values])
        if self._comm is not None:
            x = self._comm.allreduce(x)
        t = self._t_first + self.dt * np.arange(x.size)
        return t, x

    def harminv(self, fmin, fmax, **kwargs):
        """
        Resonances of the recorded signal in [fmin, fmax] [Hz].

        Keyword arguments are passed to `wakis.resonances.harminv`.

        Returns
        -------
        Resonances
            Complex frequencies, amplitudes (referred to t = 0) and errors.
        """
        t, x = self.signal()
        return harminv(x, self.dt, fmin, fmax, t0=t[0], **kwargs)
