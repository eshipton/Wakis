# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
B4: centre-fed thin-wire dipole (3D antenna benchmark).

A PEC wire of length L = 150 mm and radius a = 0.5 mm (L / a = 300,
Omega = 2 ln(L / a) = 11.4) along z, fed across a gap g = 1 mm at its
centre by a lumped port, in free space; band 0.7-1.2 GHz around the first
(series) resonance near 0.94 GHz (L ~ 0.47 lambda: thick-wire shortening).

Reference: the Hallen moment method of `wakis.analytic.thin_wire` with the
same finite gap (uniform generator field over |z| < g / 2), N = 200
segments (Delta = 1.5 a). The reduced kernel limits N (Delta >~ a), and the
input reactance still drifts slowly with N: the spread over N = 150...300
(``reference_uncertainty``) is part of the reference.

Metrics: resonance frequency (X_in = 0) relative error, R_in at resonance
relative error, NRMS of Z_in over the band (normalised by max |Z_ref|),
broadside directivity at the reference resonance (relative error).
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.optimize import brentq

from ..analytic.thin_wire import dipole_far_field, hallen_dipole

__all__ = ["DipoleCase"]


@dataclass
class DipoleCase:
    name: str = "B4"
    description: str = (
        "Centre-fed PEC dipole L = 150 mm, a = 0.5 mm, gap 1 mm, free space, "
        "0.7-1.2 GHz"
    )
    length: float = 0.150
    radius: float = 0.5e-3
    gap: float = 1.0e-3
    freqs: np.ndarray = field(default_factory=lambda: np.linspace(0.7e9, 1.2e9, 51))
    z0: float = 50.0
    n_ref: int = 200
    n_spread: tuple = (150, 300)
    t_run: float = 20e-9
    targets: dict = field(
        default_factory=lambda: dict(
            f_res=(0.01, 0.005), r_res=(0.05, 0.02), z_nrms=(0.05, 0.02),
            directivity=(0.02, 0.01),
        )
    )  # fmt: skip
    primary: str = "f_res"

    def source(self):
        from ..sources import GaussianSource

        f = self.freqs
        return GaussianSource(0.5 * (f.min() + f.max()), 0.25 * (f.max() - f.min()))

    def _z(self, f, n=None):
        return hallen_dipole(self.length, self.radius, f, n or self.n_ref, gap=self.gap)

    def _resonance(self, n=None):
        fr = brentq(
            lambda f: self._z(f, n)[0][0].imag,
            self.freqs.min(),
            self.freqs.max(),
            xtol=10.0,
        )
        return fr, float(self._z(fr, n)[0][0].real)

    @cached_property
    def reference(self):
        """
        ``z_in`` (n_freq,), ``f_res`` [Hz], ``r_res`` [Ohm], ``directivity``
        (broadside, at f_res) and the N = 150..300 spread.
        """
        z, zn, I = self._z(self.freqs)
        fr, rr = self._resonance()
        _, zn, Ir = self._z(fr)
        U, P = dipole_far_field(zn, Ir[0], fr, [np.pi / 2])
        out = dict(z_in=z, f_res=fr, r_res=rr, directivity=float(4 * np.pi * U[0] / P))
        spread = [self._resonance(n) for n in self.n_spread]
        out["reference_uncertainty"] = dict(
            f_res=max(abs(s[0] / fr - 1) for s in spread),
            r_res=max(abs(s[1] / rr - 1) for s in spread),
        )
        return out

    def evaluate(self, result):
        """
        Metrics of an arm result (``z_in`` over `freqs`; optional
        ``directivity`` (n_freq,) broadside).
        """
        ref = self.reference
        z = np.asarray(result["z_in"])
        if z.ndim == 2:  # [re, im] pairs
            z = z[:, 0] + 1j * z[:, 1]
        f = self.freqs
        X = z.imag
        out = dict(z_nrms=float(np.sqrt(np.mean(np.abs(z - ref["z_in"]) ** 2))
                                / np.abs(ref["z_in"]).max()))  # fmt: skip
        s = np.flatnonzero(np.sign(X[:-1]) != np.sign(X[1:]))
        if s.size:
            i = s[np.argmin(np.abs(f[s] - ref["f_res"]))]
            t = X[i] / (X[i] - X[i + 1])
            fr = f[i] + t * (f[i + 1] - f[i])
            rr = z.real[i] + t * (z.real[i + 1] - z.real[i])
            out["f_res"] = abs(fr / ref["f_res"] - 1)
            out["r_res"] = abs(rr / ref["r_res"] - 1)
            out["f_res_hz"] = float(fr)
            out["r_res_ohm"] = float(rr)
        else:
            out["f_res"] = out["r_res"] = np.nan
        if result.get("directivity") is not None:
            d = np.interp(ref["f_res"], result.get("d_freqs", f), result["directivity"])
            out["directivity"] = abs(d / ref["directivity"] - 1)
        out["spectra"] = dict(z_re=z.real.tolist(), z_im=z.imag.tolist())
        return out
