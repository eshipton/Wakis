# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Small helpers to read simulated spectra (resonances, dB, bandwidth) and plot
them.
"""

import matplotlib.pyplot as plt
import numpy as np


def to_db(values, power=False):
    """20 log10|x| (amplitudes) or 10 log10|x| (powers, `power=True`)."""
    return (10 if power else 20) * np.log10(np.abs(values))


def find_minimum(freqs, values):
    """
    Frequency [Hz] and value of the minimum of a sampled curve, refined by a
    parabola through the three samples around the discrete minimum.
    """
    freqs, values = np.asarray(freqs), np.asarray(values, dtype=float)
    i = int(np.clip(np.argmin(values), 1, values.size - 2))
    y0, y1, y2 = values[i - 1 : i + 2]
    curvature = y0 - 2 * y1 + y2
    shift = 0.5 * (y0 - y2) / curvature if curvature > 0 else 0.0
    shift = float(np.clip(shift, -1, 1))
    df = freqs[i + 1] - freqs[i]
    return freqs[i] + shift * df, y1 - 0.25 * (y0 - y2) * shift


def find_zero_crossings(freqs, values):
    """Frequencies [Hz] where a sampled real curve changes sign (linear interp.)."""
    freqs, values = np.asarray(freqs), np.asarray(values, dtype=float)
    idx = np.flatnonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)
    return freqs[idx] - values[idx] * (freqs[idx + 1] - freqs[idx]) / (
        values[idx + 1] - values[idx]
    )


def bandwidth(freqs, s11, level_db=-10.0):
    """
    Impedance bandwidth [Hz]: width of the frequency range around the |S11|
    minimum where |S11| < `level_db`. Zero if the minimum is above the level.
    """
    s11_db = to_db(s11)
    i = int(np.argmin(s11_db))
    if s11_db[i] >= level_db:
        return 0.0
    lo = i
    while lo > 0 and s11_db[lo - 1] < level_db:
        lo -= 1
    hi = i
    while hi < s11_db.size - 1 and s11_db[hi + 1] < level_db:
        hi += 1
    return freqs[hi] - freqs[lo]


def plot_s_parameters(freqs, curves, ax=None, level_db=-10.0):
    """
    Plot |S| in dB versus frequency.

    Parameters
    ----------
    freqs : array_like
        Frequencies [Hz].
    curves : dict of str -> array_like
        Complex S-parameters to plot, e.g. {'S11': s11}.
    ax : matplotlib Axes, optional
    level_db : float or None, optional
        Draw a reference line (default -10 dB, the usual matching criterion).

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    for label, values in curves.items():
        ax.plot(np.asarray(freqs) / 1e9, to_db(values), label=label)
    if level_db is not None:
        ax.axhline(level_db, color="gray", ls="--", lw=1)
    ax.set(xlabel="Frequency [GHz]", ylabel="|S| [dB]")
    ax.grid(True)
    ax.legend()
    return fig, ax


def plot_impedance(freqs, impedance, z0=50.0, ax=None):
    """Plot the real and imaginary parts of an impedance [Ohm] versus frequency."""
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    f = np.asarray(freqs) / 1e9
    ax.plot(f, np.real(impedance), label="R")
    ax.plot(f, np.imag(impedance), label="X")
    ax.axhline(z0, color="gray", ls="--", lw=1, label=f"Z0 = {z0:g}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="Frequency [GHz]", ylabel="Impedance [Ohm]")
    ax.grid(True)
    ax.legend()
    return fig, ax


def plot_spectra(freqs, curves, ax=None, ylabel="Power fraction"):
    """Plot real spectra (e.g. {'R': R, 'T': T}) versus frequency."""
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    for label, values in curves.items():
        ax.plot(np.asarray(freqs) / 1e9, values, label=label)
    ax.set(xlabel="Frequency [GHz]", ylabel=ylabel)
    ax.grid(True)
    ax.legend()
    return fig, ax
