# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Raw, differentiable outputs of monitors and lumped elements.

Every frequency-domain output of Wakis is a running DFT of a fixed linear
functional of the time-stepped state,

    X_m(f) = sum_n dt exp(-2j pi f (n dt + tau)) (W x^{n+1})_m,

with x^{n+1} the state (E, H or a lumped inductor current) after the step
that started at t = n dt, W a sparse sampling/averaging matrix and tau the
time offset of the sampled quantity within the step. `AdjointSimulation`
returns these accumulators X ("raw outputs") and, in the adjoint run, injects
W^H (conj(w_n) G) as adjoint sources for the cotangent G of X. Flux, mode
amplitudes, far fields and port waves are simple functions of the raw outputs
(see `wakis.inverse.jax_tools`).
"""

import numpy as np
from scipy.sparse import csr_matrix

from ..field_monitors import DFTMonitor, FluxMonitor
from ..lumped import LumpedElement, LumpedPort
from ..mode_solver import ModeMonitor


class LumpedReadout:
    def __init__(self, element, frequencies):
        """
        Differentiable voltage and current spectra of a lumped element or port.

        Pass the readout as a monitor of an `AdjointSimulation` whose sources
        contain `element` (the same object).

        Parameters
        ----------
        element : LumpedElement or LumpedPort
            Element in the simulation sources.
        frequencies : array_like
            Frequencies [Hz].

        Notes
        -----
        The raw outputs are 'V' = `element.voltage(f)` [V s] and 'I' [A s],
        `LumpedPort.port_current` (into the structure) for a port and
        `LumpedElement.current` (through the element) otherwise. Power waves
        and S-parameters follow from them (`jax_tools.power_waves`).
        """
        if not isinstance(element, LumpedElement):
            raise TypeError("element must be a LumpedElement or LumpedPort.")
        self.element = element
        self.frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))

    def update(self, solver, t):
        """No-op: the element records its own time signals."""

    def outputs(self):
        el, f = self.element, self.frequencies
        current = el.port_current(f) if isinstance(el, LumpedPort) else el.current(f)
        return {"V": el.voltage(f), "I": current}


def _selection(flat, size, weight=None, total=False):
    """
    Sparse matrix W (len(flat) x size) picking (and weighting) samples, or the
    1 x size row summing them if `total`.
    """
    flat = np.asarray(flat.get() if hasattr(flat, "get") else flat).ravel()
    weight = np.ones(flat.size) if weight is None else weight
    rows = np.zeros(flat.size, dtype=int) if total else np.arange(flat.size)
    return csr_matrix((weight, (rows, flat)), shape=(rows.max(initial=0) + 1, size))


def _host(array):
    return np.asarray(array.get() if hasattr(array, "get") else array)


def initialize(monitor, solver):
    """Initialize a monitor's sample geometry on `solver` (no time-stepping)."""
    if isinstance(monitor, (DFTMonitor, FluxMonitor, ModeMonitor)):
        monitor._initialize(solver)
        monitor.is_first_update = False


def check(name, monitor):
    """Raise for monitors whose raw outputs are not differentiable."""
    if isinstance(monitor, DFTMonitor):
        if any(comp[0] == "J" for comp in monitor.components):
            raise NotImplementedError("J monitors are not differentiable yet.")
    elif not isinstance(monitor, (FluxMonitor, ModeMonitor, LumpedReadout)):
        raise NotImplementedError(
            f"Monitor {name!r} of type {type(monitor).__name__} is not "
            "differentiable; use DFTMonitor, FluxMonitor, Near2FarMonitor, "
            "ModeMonitor or LumpedReadout."
        )


def outputs(monitor):
    """
    Raw complex outputs of an updated monitor.

    Returns
    -------
    dict of ndarray
        DFTMonitor: {component: (n_freq, nx, ny, nz)} (`get_field`).
        FluxMonitor and Near2FarMonitor: {'E<i>', 'H<i>': (n_freq, m_i)},
        the E and averaged-H accumulators of the i-th surface term.
        ModeMonitor: {'E', 'H': (2, n_freq, n_b * n_c)}, transverse (b, c)
        E and averaged H on the port plane.
        LumpedReadout: {'V', 'I': (n_freq,)}.
    """
    if isinstance(monitor, DFTMonitor):
        return {comp: monitor.get_field(comp)[0] for comp in monitor.components}
    if isinstance(monitor, FluxMonitor):
        raw = {}
        for i, term in enumerate(monitor._terms):
            raw[f"E{i}"] = _host(term["E_acc"]).copy()
            raw[f"H{i}"] = _host(term["H_acc"]).copy()
        return raw
    if isinstance(monitor, ModeMonitor):
        return {k: _host(v).copy() for k, v in monitor._acc.items()}
    return monitor.outputs()


def shapes(monitor):
    """Shapes of `outputs(monitor)` for an initialized monitor."""
    n_f = monitor.frequencies.size
    if isinstance(monitor, DFTMonitor):
        return {comp: (n_f, *monitor.shape[comp]) for comp in monitor.components}
    if isinstance(monitor, FluxMonitor):
        out = {}
        for i, term in enumerate(monitor._terms):
            out[f"E{i}"] = out[f"H{i}"] = (n_f, np.asarray(term["weight"]).size)
        return out
    if isinstance(monitor, ModeMonitor):
        n = monitor.plane.n_b * monitor.plane.n_c
        return {"E": (2, n_f, n), "H": (2, n_f, n)}
    return {"V": (n_f,), "I": (n_f,)}


class Tap:
    """
    One linear functional of the state sampled by a running DFT.

    Attributes
    ----------
    field : str
        State key ('E', 'H' or a lumped inductor key 'JL<i>').
    WH : csr_matrix
        Hermitian transpose of the sampling matrix W (state size x m).
    offset : float
        Sample time within the step started at n dt is n dt + offset [s].
    skip_last : bool
        True if the functional is not sampled after the last step.
    freqs : ndarray
        Frequencies [Hz].
    G : ndarray
        Cotangent of the DFT samples, shape (n_freq, m).
    """

    def __init__(self, field, W, offset, freqs, G, skip_last=False):
        self.field = field
        self.WH = W.conj().T.tocsr()
        self.offset = offset
        self.freqs = freqs
        self.G = np.asarray(G, dtype=complex).reshape(freqs.size, -1)
        self.skip_last = skip_last


def taps(monitor, solver, cotangents, element_keys):
    """
    Adjoint sources of a monitor for the cotangents of its raw outputs.

    Parameters
    ----------
    monitor : monitor or LumpedReadout
        Monitor after the forward run.
    solver : SolverFIT3D
        Forward solver.
    cotangents : dict
        G = dF/dRe(X) + 1j dF/dIm(X) for the raw outputs X (missing keys are
        zero).
    element_keys : dict
        {id(element): state key of its inductor current, or None}.

    Returns
    -------
    list of Tap
    """
    dt, size = solver.dt, 3 * solver.N
    freqs = monitor.frequencies
    out = []
    if isinstance(monitor, DFTMonitor):
        for comp, G in cotangents.items():
            offset = dt if comp[0] == "E" else dt / 2
            W = _selection(monitor._flat[comp], size)
            out.append(Tap(comp[0], W, offset, freqs, G))
        return out

    if isinstance(monitor, FluxMonitor):
        for i, term in enumerate(monitor._terms):
            if f"E{i}" in cotangents:
                W = _selection(term["E"], size)
                out.append(Tap("E", W, dt, freqs, cotangents[f"E{i}"]))
            if f"H{i}" in cotangents:
                m = np.asarray(term["weight"]).size
                W = _selection(term["H_above"], size, np.full(m, 0.5))
                W = W + _selection(term["H_below"], size, np.full(m, 0.5))
                out.append(Tap("H", W, dt / 2, freqs, cotangents[f"H{i}"]))
        return out

    if isinstance(monitor, ModeMonitor):
        n = monitor.plane.n_b * monitor.plane.n_c
        for d in range(2):
            if "E" in cotangents:
                W = _selection(monitor._idx["E"][d], size)
                out.append(Tap("E", W, dt, freqs, np.asarray(cotangents["E"])[d]))
            if "H" in cotangents:
                idx = _host(monitor._idx["H"][d])
                W = _selection(idx[:n], size, np.full(n, 0.5))
                W = W + _selection(idx[n:], size, np.full(n, 0.5))
                out.append(Tap("H", W, dt / 2, freqs, np.asarray(cotangents["H"])[d]))
        return out

    # Lumped element: V = -sum g (E^n + E^{n+1}) / 2 sampled at n dt + dt/2
    el = monitor.element
    G_V = np.asarray(cotangents.get("V", 0.0), dtype=complex) + np.zeros(freqs.size)
    G_I = cotangents.get("I")
    if G_I is not None:
        G_I = np.asarray(G_I, dtype=complex)
        # I is affine in V (and the inductor current): G_V += conj(dI/dV) G_I
        if isinstance(el, LumpedPort):
            G_V = G_V - G_I / el.impedance
        else:
            if el.R is not None:
                G_V = G_V + G_I / el.R
            if el.C is not None:
                G_V = G_V + np.conj(2j * np.pi * freqs * el.C) * G_I
    g = _host(el.g)
    W = _selection(el.flat_idx, size, -g / 2, total=True)
    # E^{n+1} at n dt + dt/2, and E^n, i.e. the state after step n - 1,
    # at (n - 1) dt + 3 dt/2 (E^0 = 0 does not depend on the design)
    out.append(Tap("E", W, dt / 2, freqs, G_V[:, None]))
    out.append(Tap("E", W, 3 * dt / 2, freqs, G_V[:, None], skip_last=True))
    if G_I is not None and getattr(el, "gamma", None) is not None:
        if isinstance(el, LumpedPort):
            raise NotImplementedError("Ports with inductance are not supported.")
        # I_L = -(A / l) sum g J_L^{n+1/2}, J_L of the state after step n
        weight = -g * el.area / el.length
        W = _selection(np.arange(g.size), g.size, weight, total=True)
        out.append(Tap(element_keys[id(el)], W, dt / 2, freqs, G_I[:, None]))
    return out
