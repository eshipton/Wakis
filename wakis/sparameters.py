# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Multi-port S-matrices from lumped-port simulations and Touchstone files.

S-parameters use power waves (Kurokawa) with the real reference impedance of
each port: a = (V + Z0 I)/(2 sqrt(Z0)), b = (V - Z0 I)/(2 sqrt(Z0)) and
S_ij = b_i / a_j with only port j excited.
"""

import numpy as np


def s_matrix(solver_factory, ports, src_time, Nt, freqs, elements=(), **emsolve_kwargs):
    """
    S-matrix of an N-port structure, exciting the ports one at a time.

    Runs N simulations. In run j port j is driven with `src_time` and all
    other ports are passive matched terminations (their resistor Z0).

    Parameters
    ----------
    solver_factory : callable
        Function with no arguments returning a new SolverFIT3D of the structure.
    ports : list of LumpedPort
        Port templates (not yet simulated). Their `src_time` is ignored.
    src_time : GaussianSource or other source time profile
        Excitation of the driven port; its spectrum must cover `freqs`.
    Nt : int
        Maximum number of timesteps per run.
    freqs : array_like
        Frequencies [Hz].
    elements : list of LumpedElement, optional
        Further lumped elements of the structure (templates, cloned per run).
    **emsolve_kwargs
        Passed to `SolverFIT3D.emsolve`, e.g. `stop_when_energy_decayed=1e-4`.

    Returns
    -------
    S : ndarray of complex, shape (len(freqs), N, N)
        S[k, i, j] = b_i / a_j at freqs[k].
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    n_ports = len(ports)
    S = np.zeros((freqs.size, n_ports, n_ports), dtype=complex)
    for j in range(n_ports):
        run_ports = []
        for i, template in enumerate(ports):
            port = template.clone()
            port.src_time = src_time if i == j else None
            port.end_time = src_time.end_time if i == j else 0.0
            run_ports.append(port)
        run_elements = [element.clone() for element in elements]
        solver = solver_factory()
        solver.emsolve(Nt, source=run_ports + run_elements, **emsolve_kwargs)
        a_j, _ = run_ports[j].power_waves(freqs)
        for i, port in enumerate(run_ports):
            S[:, i, j] = port.power_waves(freqs)[1] / a_j
    return S


def write_touchstone(filename, freqs, S, z0=50.0, comment=None):
    """
    Write S-parameters to a Touchstone v1 file (.sNp) in real/imaginary format.

    Parameters
    ----------
    filename : str
        Output path; by convention it ends in '.sNp' with N the port count.
    freqs : array_like
        Frequencies [Hz].
    S : array_like of complex, shape (len(freqs), N, N) or (len(freqs),)
        S-matrix; a 1-D array is taken as S11 of a 1-port.
    z0 : float, optional
        Reference impedance of all ports [Ohm]. Default is 50.
    comment : str, optional
        Text written as '!' comment lines at the top of the file.

    Notes
    -----
    The data order follows the Touchstone convention: a 2-port line is
    S11 S21 S12 S22; for N > 2 every matrix row S_i1 ... S_iN starts a new
    line with at most four complex pairs per line.
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    S = np.asarray(S, dtype=complex)
    if S.ndim == 1:
        S = S[:, None, None]
    n_ports = S.shape[1]
    lines = [f"! {text}" for text in (comment or "").splitlines()]
    lines.append(f"# HZ S RI R {z0:g}")
    for f, Sk in zip(freqs, S):
        if n_ports <= 2:
            values = Sk.T.ravel()  # 2-port order S11 S21 S12 S22
            rows = [values]
        else:
            rows = [
                Sk[i, k : k + 4] for i in range(n_ports) for k in range(0, n_ports, 4)
            ]
        for r, values in enumerate(rows):
            pairs = " ".join(f"{v.real:.12e} {v.imag:.12e}" for v in values)
            lines.append(f"{f:.12e} {pairs}" if r == 0 else f"  {pairs}")
    with open(filename, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def read_touchstone(filename):
    """
    Read a Touchstone v1 file (.sNp) with S-parameters.

    Supports the RI, MA and DB formats and Hz/kHz/MHz/GHz units. The number of
    ports is taken from the file extension.

    Parameters
    ----------
    filename : str
        Path to a '.sNp' file.

    Returns
    -------
    freqs : ndarray
        Frequencies [Hz].
    S : ndarray of complex, shape (len(freqs), N, N)
    z0 : float
        Reference impedance [Ohm].
    """
    n_ports = int(str(filename).lower().rsplit(".s", 1)[1].rstrip("p"))
    unit, fmt, z0 = 1e9, "MA", 50.0  # Touchstone defaults: GHz S MA R 50
    numbers = []
    with open(filename) as fh:
        for line in fh:
            line = line.split("!", 1)[0].strip()
            if not line:
                continue
            if line.startswith("#"):
                tokens = line[1:].upper().split()
                units = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
                for k, token in enumerate(tokens):
                    if token in units:
                        unit = units[token]
                    elif token in ("RI", "MA", "DB"):
                        fmt = token
                    elif token == "R":
                        z0 = float(tokens[k + 1])
                    elif token in ("Y", "Z", "H", "G"):
                        raise ValueError("Only S-parameter files are supported.")
                continue
            numbers.extend(float(v) for v in line.split())

    data = np.asarray(numbers).reshape(-1, 1 + 2 * n_ports**2)
    freqs = data[:, 0] * unit
    x, y = data[:, 1::2], data[:, 2::2]
    if fmt == "RI":
        values = x + 1j * y
    elif fmt == "MA":
        values = x * np.exp(1j * np.deg2rad(y))
    else:
        values = 10 ** (x / 20) * np.exp(1j * np.deg2rad(y))
    S = values.reshape(-1, n_ports, n_ports)
    if n_ports == 2:
        S = S.transpose(0, 2, 1)  # 2-port order S11 S21 S12 S22
    return freqs, S, z0


def to_skrf_network(freqs, S, z0=50.0, name=None):
    """
    Convert S-parameters to a scikit-rf Network (requires scikit-rf).

    Parameters
    ----------
    freqs : array_like
        Frequencies [Hz].
    S : array_like of complex, shape (len(freqs), N, N) or (len(freqs),)
    z0 : float, optional
        Reference impedance [Ohm]. Default is 50.
    name : str, optional
        Network name.

    Returns
    -------
    skrf.Network
    """
    import skrf

    S = np.asarray(S, dtype=complex)
    if S.ndim == 1:
        S = S[:, None, None]
    frequency = skrf.Frequency.from_f(np.asarray(freqs, dtype=float), unit="Hz")
    return skrf.Network(frequency=frequency, s=S, z0=z0, name=name)
