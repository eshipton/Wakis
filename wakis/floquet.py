# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Broadband oblique-incidence sweeps at constant angle and Floquet ports.

In a time-domain Bloch simulation (spectral-FDTD style; Aminian & Rahmat-Samii,
*IEEE Trans. Antennas Propag.* **54**(6), 1818-1825, 2006,
doi:10.1109/TAP.2006.875484), the tangential wavevector ``k_par`` is fixed, so
a broadband pulse probes a frequency-dependent polar angle of incidence
``theta(f) = arcsin(c |k_par| / (2 pi f n))``. To obtain reflectance,
transmittance, and diffraction-order efficiencies at a *constant* incidence
angle ``theta`` across a frequency band ``[f_min, f_max]``, this module:

1. Chooses a uniform grid of scalar tangential wavevectors ``k_par`` whose rays
   ``omega = c k_par / (n sin(theta))`` cover the target ``(theta, f)`` region
   at a stated sampling density.
2. Runs one independent Bloch simulation per ``k_par`` (locally in parallel via
   ``multiprocessing`` or distributed via a ``--list-tasks`` / ``--task`` batch
   interface), collecting the complex order amplitudes and powers from
   `DiffractionMonitor`.
3. Resamples the collected spectra onto constant-``theta`` curves by 4-point
   cubic Lagrange interpolation in ``k_par`` at fixed ``f`` on the uniform grid
   (fourth-order error ``O(dk_par^4)``), without extrapolating past the sampled
   ``k_par`` range and masking Rayleigh anomalies (orders appearing or
   disappearing) rather than interpolating across them.

It also provides `FloquetPort` and `floquet_s_matrix` to define the unit-power
propagating Floquet modes ``(port, m, n, pol)`` of the homogeneous half-spaces
above and below a periodic unit cell and extract the generalized scattering
matrix ``S_{ij}`` referenced to stated port planes.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import inspect
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
from scipy.constants import c, epsilon_0, mu_0

from .bloch import incidence_angle, k_parallel
from .diffraction import DiffractionMonitor
from .gridFIT3D import GridFIT3D
from .solverFIT3D import SolverFIT3D
from .sources import CurrentSource, GaussianSource


def _normalize_pol(pol):
    p = str(pol).strip().upper()
    if p in ("TE", "S"):
        return "TE"
    if p in ("TM", "P"):
        return "TM"
    raise ValueError(f"polarization must be 'TE'/'s' or 'TM'/'p', got {pol!r}.")


def k_parallel_grid(
    frequencies,
    angles,
    n=1.0,
    n_k=None,
    dk=None,
    dtheta=None,
    dk_max=None,
    dtheta_max=None,
    pad_points=0,
):
    """
    Uniform grid of tangential wavevectors covering target ``(theta, f)`` points.

    The constant-angle rays ``k_par(theta, f) = (2 pi f n / c) sin(theta)`` on
    ``[f_min, f_max] x [theta_min, theta_max]`` span ``[k_min, k_max]`` with
    ``k_min = (2 pi f_min n / c) sin(theta_min)`` and
    ``k_max = (2 pi f_max n / c) sin(theta_max)``.

    Parameters
    ----------
    frequencies : array_like
        Monitored frequencies [Hz] (positive).
    angles : float or array_like
        Target polar incidence angle(s) from the cell normal [rad], in
        ``[0, pi/2)``.
    n : float, optional
        Refractive index of the incidence medium. Default 1.0.
    n_k : int, optional
        Number of uniform ``k_par`` samples (at least 4 when ``k_max > k_min``
        for cubic interpolation).
    dk, dk_max : float, optional
        Maximum wavevector spacing ``Delta k_par`` [rad/m].
    dtheta, dtheta_max : float, optional
        Target angular spacing ``Delta theta`` [rad] at the mid-band frequency.
    pad_points : int, optional
        Optional padding points while staying strictly below the light line
        ``(2 pi f_min n / c)`` (default 0).

    Returns
    -------
    ndarray
        1D uniform array of tangential wavevector magnitudes ``k_par`` [rad/m].
    """
    if dk is None and dk_max is not None:
        dk = dk_max
    if dtheta is None and dtheta_max is not None:
        dtheta = dtheta_max

    f = np.abs(np.atleast_1d(np.asarray(frequencies, dtype=float)))
    th = np.atleast_1d(np.asarray(angles, dtype=float))
    if np.any(f <= 0):
        raise ValueError("frequencies must be positive.")
    if np.any(th < 0) or np.any(th >= 0.5 * np.pi):
        raise ValueError("angles must be in [0, pi/2) rad.")
    f_min, f_max = float(f.min()), float(f.max())
    th_min, th_max = float(th.min()), float(th.max())
    k_min = 2.0 * np.pi * f_min * n / c * np.sin(th_min)
    k_max = 2.0 * np.pi * f_max * n / c * np.sin(th_max)
    if np.isclose(k_max, k_min, rtol=0.0, atol=1e-14):
        return np.array([k_min], dtype=float)

    if n_k is None:
        if dk is not None:
            if dk <= 0:
                raise ValueError("dk must be positive.")
            n_k = max(4, int(np.ceil((k_max - k_min) / dk)) + 1)
        elif dtheta is not None:
            if dtheta <= 0:
                raise ValueError("dtheta must be positive.")
            f_mid = 0.5 * (f_min + f_max)
            th_mid = 0.5 * (th_min + th_max)
            dk_est = 2.0 * np.pi * f_mid * n / c * np.cos(th_mid) * dtheta
            n_k = max(4, int(np.ceil((k_max - k_min) / dk_est)) + 1)
        else:
            n_k = 9
    if n_k < 4:
        raise ValueError("n_k must be >= 4 for cubic interpolation when k_max > k_min.")
    return np.linspace(k_min, k_max, int(n_k))


def sampling_density(k_par, frequencies, angles=None, n=1.0):
    """
    Summary of the ``(k_par, f)`` sampling density and cubic error scale.

    Parameters
    ----------
    k_par : array_like
        1D uniform array of tangential wavevectors [rad/m].
    frequencies : array_like
        Frequencies [Hz].
    angles : float or array_like, optional
        Target polar angles [rad].
    n : float, optional
        Incidence-medium refractive index. Default 1.0.

    Returns
    -------
    dict
        ``n_k``, ``k_min``, ``k_max``, ``dk`` [rad/m], ``dtheta_rad`` and
        ``dtheta_deg`` at the band centre, and ``cubic_error_bound_factor``
        ``= (3 / 128) * dk^4`` (interior) / ``(1 / 24) * dk^4`` (boundary).
    """
    k_par = np.atleast_1d(np.asarray(k_par, dtype=float))
    f = np.abs(np.atleast_1d(np.asarray(frequencies, dtype=float)))
    f_mid = 0.5 * float(f.min() + f.max())
    k0_mid = 2.0 * np.pi * f_mid * n / c
    dk = float(k_par[1] - k_par[0]) if k_par.size > 1 else 0.0
    if angles is not None:
        th = np.atleast_1d(np.asarray(angles, dtype=float))
        cos_mid = float(np.cos(0.5 * (th.min() + th.max())))
    else:
        s_mid = np.clip(0.5 * (k_par[0] + k_par[-1]) / k0_mid, 0.0, 0.999)
        cos_mid = float(np.sqrt(1.0 - s_mid**2))
    dtheta_rad = dk / max(k0_mid * cos_mid, 1e-30)
    return {
        "n_k": int(k_par.size),
        "k_min": float(k_par[0]),
        "k_max": float(k_par[-1]),
        "dk": dk,
        "dtheta_rad": dtheta_rad,
        "dtheta_deg": float(np.degrees(dtheta_rad)),
        "interior_error_factor": (3.0 / 128.0) * dk**4,
        "boundary_error_factor": (1.0 / 24.0) * dk**4,
    }


def _lagrange_weights(s, m):
    """Lagrange interpolation weights at ``s`` on nodes ``0, 1, ..., m - 1``."""
    nodes = np.arange(m, dtype=float)
    w = np.ones(m, dtype=float)
    for j in range(m):
        for r in range(m):
            if r != j:
                w[j] *= (s - nodes[r]) / (nodes[j] - nodes[r])
    return w


def _lagrange_weights_nodes(x_eval: float, nodes: np.ndarray) -> np.ndarray:
    """Lagrange interpolation weights at ``x_eval`` on arbitrary 1D ``nodes``."""
    m = nodes.size
    w = np.ones(m, dtype=float)
    for j in range(m):
        for r in range(m):
            if r != j:
                w[j] *= (x_eval - nodes[r]) / (nodes[j] - nodes[r])
    return w


def resample_constant_angle(
    k_par,
    frequencies,
    values,
    angles,
    n=1.0,
    propagating=None,
    index_shift=0,
):
    """
    Resample ``(k_par, f)`` data onto constant-``theta`` curves at fixed ``f``.

    Uses 4-point cubic Lagrange interpolation in ``k_par`` at each frequency
    ``f`` on the uniform grid ``k_par``. For a smooth function ``g(k_par)``,
    the pointwise interpolation error is fourth-order:

        ``|g(k) - P_3(k)| <= C_4 (dk_par)^4 max |d^4 g / dk_par^4|``,

    with ``C_4 = 3/128`` on interior intervals and ``C_4 = 1/24`` on boundary
    intervals adjacent to the ends of the valid range.

    Extrapolation past ``[k_par[0], k_par[-1]]`` is never performed (returns
    ``NaN``). Rayleigh anomalies (orders appearing or disappearing as ``k_par``
    varies at fixed ``f``) are handled by masking: non-propagating samples are
    excluded, intervals ``[k_i, k_{i+1}]`` straddling a cutoff transition are
    masked as ``NaN``, and the 4-point cubic stencil is clamped to stay strictly
    within the contiguous propagating segment.

    Parameters
    ----------
    k_par : array_like
        1D uniform array of sampled tangential wavevectors [rad/m], shape
        ``(n_k,)``.
    frequencies : array_like
        1D array of frequencies [Hz], shape ``(n_freq,)``.
    values : array_like
        Sampled values of shape ``(n_k, n_freq, ...)``, real or complex.
    angles : float or array_like
        Target polar angle(s) of incidence [rad].
    n : float, optional
        Refractive index of the incidence medium. Default 1.0.
    propagating : array_like of bool, optional
        Boolean propagation mask broadcastable to ``values.shape``.
    index_shift : int, optional
        Shift applied to the normalized ``k_par`` lookup index (default 0;
        non-zero values are used for mutation testing).

    Returns
    -------
    ndarray
        Resampled values at constant ``angles``, shape ``(n_angles, n_freq, ...)``
        if ``angles`` is 1D, or ``(n_freq, ...)`` if ``angles`` is a scalar.
    """
    k_par = np.atleast_1d(np.asarray(k_par, dtype=float))
    f = np.abs(np.atleast_1d(np.asarray(frequencies, dtype=float)))
    arr = np.asarray(values)
    scalar_angle = np.ndim(angles) == 0
    th = np.atleast_1d(np.asarray(angles, dtype=float))

    n_k = k_par.size
    n_f = f.size
    if arr.shape[:2] != (n_k, n_f):
        raise ValueError(
            f"values must have leading shape (n_k={n_k}, n_freq={n_f}), "
            f"got {arr.shape}."
        )
    if n_k > 1:
        diffs = np.diff(k_par)
        if np.any(diffs <= 0) or not np.allclose(diffs, diffs[0], rtol=1e-7):
            raise ValueError("k_par must be strictly increasing and uniformly spaced.")
        dk = float(diffs[0])
    else:
        dk = 0.0

    valid = np.isfinite(arr)
    if propagating is not None:
        valid = valid & np.broadcast_to(np.asarray(propagating, dtype=bool), arr.shape)

    rest_shape = arr.shape[2:]
    n_rest = int(np.prod(rest_shape)) if rest_shape else 1
    arr_flat = arr.reshape(n_k, n_f, n_rest)
    valid_flat = valid.reshape(n_k, n_f, n_rest)

    out_dtype = complex if np.iscomplexobj(arr) else float
    out = np.full((th.size, n_f, n_rest), np.nan + (0j if out_dtype is complex else 0.0), dtype=out_dtype)

    for ia, theta in enumerate(th):
        if theta < 0 or theta >= 0.5 * np.pi:
            continue
        k_targets = 2.0 * np.pi * f * n / c * np.sin(theta)
        for jf in range(n_f):
            kt = k_targets[jf]
            if n_k == 1:
                if np.isclose(kt, k_par[0], rtol=0.0, atol=1e-12 * max(1.0, abs(k_par[0]))):
                    idx0 = 0 + index_shift
                    if 0 <= idx0 < n_k:
                        for m_idx in range(n_rest):
                            if valid_flat[idx0, jf, m_idx]:
                                out[ia, jf, m_idx] = arr_flat[idx0, jf, m_idx]
                continue

            u = (kt - k_par[0]) / dk + index_shift
            # Never extrapolate past the sampled k_par range
            if u < -1e-11 or u > (n_k - 1) + 1e-11:
                continue
            u = float(np.clip(u, 0.0, float(n_k - 1)))
            nearest = int(np.rint(u))
            on_node = abs(u - nearest) <= 1e-11
            cell_i = min(max(int(np.floor(u)), 0), n_k - 2)

            for m_idx in range(n_rest):
                v_col = valid_flat[:, jf, m_idx]
                y_col = arr_flat[:, jf, m_idx]
                if on_node:
                    if v_col[nearest]:
                        out[ia, jf, m_idx] = y_col[nearest]
                    continue

                # Both endpoints of the interval [k_i, k_{i+1}] must propagate;
                # otherwise a Rayleigh anomaly lies inside the interval.
                if not (v_col[cell_i] and v_col[cell_i + 1]):
                    continue

                # Expand to the contiguous propagating segment [i_min, i_max]
                i_min = cell_i
                while i_min > 0 and v_col[i_min - 1]:
                    i_min -= 1
                i_max = cell_i + 1
                while i_max + 1 < n_k and v_col[i_max + 1]:
                    i_max += 1

                seg_len = i_max - i_min + 1
                order_pts = min(4, seg_len)
                if order_pts == 4:
                    lo = (
                        i_min + 1
                        if (i_min > 0 and cell_i > i_min and i_max - i_min >= 4)
                        else i_min
                    )
                    k_light = 2.0 * np.pi * f[jf] * n / c
                    near_branch_hi = (i_max < n_k - 1) or (
                        k_par[i_max] > 0.94 * k_light and k_targets[jf] <= k_par[i_max - 1]
                    )
                    hi = (
                        i_max - 1
                        if (near_branch_hi and cell_i + 1 < i_max and i_max - 1 - lo >= 3)
                        else i_max
                    )
                    j0 = min(max(cell_i - 1, lo), hi - 3)
                else:
                    j0 = min(max(cell_i, i_min), i_max - order_pts + 1)
                w = _lagrange_weights(u - j0, order_pts)
                y_st = y_col[j0 : j0 + order_pts]
                if out_dtype is complex and order_pts >= 2:
                    steps = y_st[:-1].conj() * y_st[1:]
                    if np.all(np.real(steps) > 0.0):
                        dphi = float(np.angle(np.sum(steps)))
                        idx_st = np.arange(order_pts, dtype=float)
                        val = np.dot(w, y_st * np.exp(-1j * idx_st * dphi))
                        out[ia, jf, m_idx] = val * np.exp(1j * (u - j0) * dphi)
                        continue
                out[ia, jf, m_idx] = np.dot(w, y_st)

    out = out.reshape((th.size, n_f, *rest_shape))
    return out[0] if scalar_angle else out



@dataclass(frozen=True)
class FloquetMode:
    """Propagating or evanescent Floquet port mode descriptor.

    Parameters
    ----------
    port : int or str
        Port identifier (e.g. ``1``, ``2``, ``"below"``, ``"above"``).
    m : int
        Diffraction order along the first transverse axis.
    n : int
        Diffraction order along the second transverse axis.
    pol : {"TE", "TM"}
        Canonical polarization label (``"TE"`` / ``"s"`` or ``"TM"`` / ``"p"``).
    """

    port: int | str
    m: int
    n: int
    pol: str

    def __post_init__(self):
        object.__setattr__(self, "pol", _normalize_pol(self.pol))
        object.__setattr__(self, "m", int(self.m))
        object.__setattr__(self, "n", int(self.n))

    def reciprocal_partner(self) -> FloquetMode:
        """Return the Lorentz-reciprocity partner mode ``(port, -m, -n, pol)``."""
        return FloquetMode(self.port, -self.m, -self.n, self.pol)


def _reciprocity_phase_factor(kt_b: np.ndarray, kt_c: np.ndarray, pol: str) -> np.ndarray:
    """Unit-modulus canonical phase factor aligning basis sign under ``k_t -> -k_t``.

    In ``DiffractionMonitor``, the transverse wavevector direction is
    ``k_hat(-k_t) = -k_hat(k_t)`` for ``k_t != 0`` whereas ``k_hat(0) = +b_hat``,
    and the TM magnetic/electric wave-impedance ratio carries the opposite sign
    from TE in the Lorentz reciprocity bilinear form
    ``int (E^(A) x H^(B) - E^(B) x H^(A)) . n_hat dA = 0``.
    Multiplying the raw mode amplitude by ``s(k_t) * (1 if TE else 1j)``
    (where ``s(k_t) = -1`` on the open half-plane ``k_tb < 0`` or
    ``k_tb == 0, k_tc < 0``, and ``+1`` on ``k_t >= 0``) has unit modulus
    ``|c| = 1`` and leaves ``k_t >= 0`` TE--TE and TM--TM reflection/transmission
    phases unchanged while making ``S_ij(k_0) = S_{-j,-i}(-k_0)`` hold directly.
    """
    neg_half = (kt_b < -1e-12) | ((np.abs(kt_b) <= 1e-12) & (kt_c < -1e-12))
    sign = np.where(neg_half, -1.0 + 0j, 1.0 + 0j)
    if _normalize_pol(pol) == "TM":
        return sign * 1j
    return sign


class FloquetPort:
    """Half-space Floquet port wrapping a :class:`~wakis.diffraction.DiffractionMonitor`.

    Normalizes every propagating Floquet harmonic ``(m, n, pol)`` to unit time-averaged
    power across the unit cell area ``A`` and shifts its complex phase from the
    monitor plane ``z_monitor`` to a user-specified port reference plane
    ``reference_position``.

    Parameters
    ----------
    monitor : DiffractionMonitor
        Underlying plane-wave / diffraction-order monitor placed in the homogeneous
        half-space above or below the periodic unit cell.
    side : {"below", "above"}
        Half-space side relative to the unit cell:
        - ``"below"`` (lower half-space, smaller coordinate along the normal axis):
          waves propagating in ``+a`` (``up``) are **incoming** to the cell and waves
          propagating in ``-a`` (``down``) are **outgoing** from the cell.
        - ``"above"`` (upper half-space, larger coordinate along the normal axis):
          waves propagating in ``-a`` (``down``) are **incoming** to the cell and waves
          propagating in ``+a`` (``up``) are **outgoing** from the cell.
    reference_position : float, optional
        Coordinate [m] along the monitor normal axis to which mode phases are
        de-embedded. Defaults to the monitor plane coordinate ``monitor.position``.
    port_id : int or str, optional
        Identifier used in :class:`FloquetMode` labels. Defaults to ``1`` for
        ``side="below"`` and ``2`` for ``side="above"``.
    """

    def __init__(
        self,
        monitor: DiffractionMonitor,
        side: str = "below",
        reference_position: float | None = None,
        port_id: int | str | None = None,
        dn: float | None = None,
    ):
        side_norm = side.strip().lower()
        if side_norm not in ("below", "above"):
            raise ValueError(f"side must be 'below' or 'above', got {side!r}.")
        self.monitor = monitor
        self.side = side_norm
        self.dn = None if dn is None else float(dn)
        self._reference_position = (
            None if reference_position is None else float(reference_position)
        )
        if port_id is None:
            self.port_id = 1 if self.side == "below" else 2
        else:
            self.port_id = port_id

    @property
    def monitor_position(self) -> float:
        """Exact grid coordinate [m] of the monitor plane along its normal axis."""
        a = self.monitor.faces[0][0]
        terms = getattr(self.monitor, "_terms", None)
        if terms and terms[0]["coords"][a].size:
            return float(terms[0]["coords"][a][0])
        return float(self.monitor.center[a])

    @property
    def reference_position(self) -> float:
        """Port phase reference plane coordinate [m] along the normal axis."""
        if self._reference_position is None:
            return self.monitor_position
        return self._reference_position

    @reference_position.setter
    def reference_position(self, value: float | None) -> None:
        self._reference_position = None if value is None else float(value)

    @property
    def delta_position(self) -> float:
        """Signed distance ``z_ref - z_monitor`` [m] along the normal axis."""
        return self.reference_position - self.monitor_position

    @property
    def orders_b(self) -> np.ndarray:
        """1D integer array of diffraction orders along the first transverse axis."""
        return np.asarray(self.monitor.orders[0], dtype=int)

    @property
    def orders_c(self) -> np.ndarray:
        """1D integer array of diffraction orders along the second transverse axis."""
        return np.asarray(self.monitor.orders[1], dtype=int)

    def _find_order_indices(self, m: int, n: int) -> tuple[int | None, int | None]:
        mb = np.flatnonzero(self.orders_b == int(m))
        nc = np.flatnonzero(self.orders_c == int(n))
        im = int(mb[0]) if mb.size else None
        in_ = int(nc[0]) if nc.size else None
        return im, in_

    def discrete_normal_wavenumber(self) -> np.ndarray:
        """Return the normal wavenumber ``k_a`` [rad/m] of each order.

        When the normal grid step ``dn`` is provided, uses the exact discrete
        FIT staggered-grid relation ``cos(k_a * dn / 2) = cos_half`` from
        :func:`~wakis.diffraction.order_geometry`. Otherwise falls back to the
        continuum dispersion relation ``sqrt(k_medium^2 - k_tb^2 - k_tc^2)``.
        """
        geom = self.monitor.geometry
        prop = np.asarray(geom["propagating"], dtype=bool)
        if self.dn is not None and self.dn > 0.0:
            cos_half = np.clip(np.asarray(geom["cos_half"], dtype=float), 0.0, 1.0)
            ka = (2.0 / self.dn) * np.arcsin(np.sqrt(np.maximum(0.0, 1.0 - cos_half**2)))
        else:
            kt_b, kt_c = geom["k_t"]
            k_med = geom["k_medium"][:, None, None]
            ka = np.sqrt(np.maximum(0.0, k_med**2 - kt_b**2 - kt_c**2))
        return np.where(prop, ka, np.nan)

    def propagating_modes(self, freq_idx: int = 0) -> list[FloquetMode]:
        """Return all propagating :class:`FloquetMode` entries at ``freq_idx``."""
        geom = self.monitor.geometry
        prop = np.asarray(geom["propagating"][freq_idx], dtype=bool)
        modes: list[FloquetMode] = []
        for im, m in enumerate(self.orders_b):
            for in_, n in enumerate(self.orders_c):
                if prop[im, in_]:
                    modes.append(FloquetMode(self.port_id, int(m), int(n), "TE"))
                    modes.append(FloquetMode(self.port_id, int(m), int(n), "TM"))
        return modes

    def wave_amplitudes(
        self,
        reciprocity_convention: bool = True,
    ) -> dict[str, np.ndarray]:
        """Return unit-power-normalized incoming and outgoing modal wave amplitudes.

        For a unit cell of transverse area ``A``, the flux spectrum carried by
        mode ``(m, n, pol)`` is ``|a_in|^2`` (into the cell) and ``|b_out|^2``
        (out of the cell), with phases referenced to ``self.reference_position``.

        Parameters
        ----------
        reciprocity_convention : bool, optional
            If ``True`` (default), apply the unit-modulus canonical basis factor so
            that Lorentz reciprocity holds as ``S_ij(k_0) = S_{-j,-i}(-k_0)``.

        Returns
        -------
        dict of str to ndarray
            Dictionary with keys ``"a_in_TE"``, ``"a_in_TM"``, ``"b_out_TE"``,
            ``"b_out_TM"``, ``"propagating"``, and ``"k_a"``, each of shape
            ``(n_freq, n_mb, n_mc)``.
        """
        amps = self.monitor.get_amplitudes(polarization="linear")
        geom = self.monitor.geometry
        prop = np.asarray(geom["propagating"], dtype=bool)

        w_te_up = np.asarray(amps["up_s"], dtype=complex)
        w_tm_up = np.asarray(amps["up_p"], dtype=complex)
        w_te_dn = np.asarray(amps["down_s"], dtype=complex)
        w_tm_dn = np.asarray(amps["down_p"], dtype=complex)

        # Phase shift from monitor_position to reference_position (exp(+2*pi*i*f*t)
        # phasors at f > 0: +a wave varies as exp(-i k_a z), -a wave varies as exp(+i k_a z)).
        ka = self.discrete_normal_wavenumber()
        dz = self.delta_position
        if abs(dz) > 0.0:
            sign_f = np.sign(self.monitor.frequencies)[:, None, None]
            shift_up = np.exp(-1j * sign_f * ka * dz)
            shift_dn = np.exp(+1j * sign_f * ka * dz)
            w_te_up = w_te_up * shift_up
            w_tm_up = w_tm_up * shift_up
            w_te_dn = w_te_dn * shift_dn
            w_tm_dn = w_tm_dn * shift_dn

        if self.side == "below":
            a_in_te, a_in_tm = w_te_up, w_tm_up
            b_out_te, b_out_tm = w_te_dn, w_tm_dn
        else:
            a_in_te, a_in_tm = w_te_dn, w_tm_dn
            b_out_te, b_out_tm = w_te_up, w_tm_up

        if reciprocity_convention:
            kt_b, kt_c = geom["k_t"]
            c_te = _reciprocity_phase_factor(kt_b, kt_c, "TE")
            c_tm = _reciprocity_phase_factor(kt_b, kt_c, "TM")
            a_in_te = a_in_te * c_te
            b_out_te = b_out_te * c_te
            a_in_tm = a_in_tm * c_tm
            b_out_tm = b_out_tm * c_tm

        return {
            "a_in_TE": np.where(prop, a_in_te, np.nan + 0j),
            "a_in_TM": np.where(prop, a_in_tm, np.nan + 0j),
            "b_out_TE": np.where(prop, b_out_te, np.nan + 0j),
            "b_out_TM": np.where(prop, b_out_tm, np.nan + 0j),
            "propagating": prop,
            "k_a": ka,
        }

    def mode_amplitude(
        self,
        mode: FloquetMode | tuple,
        direction: str = "out",
        reciprocity_convention: bool = True,
    ) -> np.ndarray:
        """Return the 1D frequency array of normalized wave amplitude for ``mode``.

        Parameters
        ----------
        mode : FloquetMode or tuple ``(m, n, pol)`` or ``(port, m, n, pol)``
            Target Floquet mode.
        direction : {"in", "out"}
            Whether to return the incoming (``"in"``) or outgoing (``"out"``) wave.
        reciprocity_convention : bool, optional
            Apply the canonical reciprocity basis phase factor (default ``True``).
        """
        if not isinstance(mode, FloquetMode):
            if len(mode) == 3:
                mode = FloquetMode(self.port_id, mode[0], mode[1], mode[2])
            elif len(mode) == 4:
                mode = FloquetMode(*mode)
            else:
                raise ValueError(f"Invalid mode specification {mode!r}.")
        im, in_ = self._find_order_indices(mode.m, mode.n)
        n_f = self.monitor.frequencies.size
        if im is None or in_ is None:
            return np.full(n_f, np.nan + 0j, dtype=complex)
        waves = self.wave_amplitudes(reciprocity_convention=reciprocity_convention)
        prefix = "a_in" if direction == "in" else "b_out"
        return waves[f"{prefix}_{mode.pol}"][:, im, in_]



def s_parameters_from_ports(
    ports: Sequence[FloquetPort],
    incident_mode: FloquetMode | tuple,
    incident_amplitude: np.ndarray | None = None,
    modes: Sequence[FloquetMode | tuple] | None = None,
    reciprocity_convention: bool = True,
) -> dict[FloquetMode, np.ndarray]:
    """Compute generalized Floquet S-parameter column ``S_{i, inc}(f)`` from port monitors.

    When the cell is excited purely in ``incident_mode`` with unit-power wave
    amplitude ``a_inc(f)`` at its port reference plane, the generalized S-parameter
    into outgoing mode ``i`` is ``S_{i, inc}(f) = b_i(f) / a_inc(f)``. For a
    lossless cell, energy conservation implies ``sum_i |S_{i, inc}(f)|^2 = 1``.

    Parameters
    ----------
    ports : sequence of FloquetPort
        Half-space Floquet ports (typically ``[port_below, port_above]``).
    incident_mode : FloquetMode or tuple ``(port, m, n, pol)``
        Driven incident Floquet mode.
    incident_amplitude : ndarray, optional
        Complex 1D array of incident wave amplitude ``a_inc(f)`` referenced to the
        incident port's ``reference_position``. If ``None``, extracted directly from
        the incoming wave on the incident port.
    modes : sequence of FloquetMode, optional
        Outgoing modes ``i`` to include. Defaults to all modes present in ``ports``.
    reciprocity_convention : bool, optional
        Apply canonical reciprocity phase factors (default ``True``).

    Returns
    -------
    dict of FloquetMode to ndarray
        Mapping ``mode_i -> S_{i, inc}(f)`` of shape ``(n_freq,)``.
    """
    if not isinstance(incident_mode, FloquetMode):
        incident_mode = FloquetMode(*incident_mode)

    port_map = {p.port_id: p for p in ports}
    if incident_mode.port not in port_map:
        raise ValueError(
            f"Incident mode port {incident_mode.port!r} not in ports {list(port_map)}."
        )

    if incident_amplitude is None:
        a_inc = port_map[incident_mode.port].mode_amplitude(
            incident_mode, direction="in", reciprocity_convention=reciprocity_convention
        )
    else:
        a_inc = np.asarray(incident_amplitude, dtype=complex)

    if modes is None:
        mode_list: list[FloquetMode] = []
        for p in ports:
            for m in p.orders_b:
                for n in p.orders_c:
                    mode_list.append(FloquetMode(p.port_id, int(m), int(n), "TE"))
                    mode_list.append(FloquetMode(p.port_id, int(m), int(n), "TM"))
    else:
        mode_list = [
            m if isinstance(m, FloquetMode) else FloquetMode(*m) for m in modes
        ]

    out: dict[FloquetMode, np.ndarray] = {}
    for mode in mode_list:
        b_out = port_map[mode.port].mode_amplitude(
            mode, direction="out", reciprocity_convention=reciprocity_convention
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            s_col = np.where(np.abs(a_inc) > 0, b_out / a_inc, np.nan + 0j)
        out[mode] = s_col
    return out


def floquet_s_matrix(
    excitations: dict[FloquetMode | tuple, Sequence[FloquetPort]],
    modes: Sequence[FloquetMode | tuple] | None = None,
    incident_amplitudes: dict[FloquetMode | tuple, np.ndarray] | None = None,
    reciprocity_convention: bool = True,
) -> tuple[np.ndarray, list[FloquetMode]]:
    """Assemble a multi-mode generalized Floquet scattering matrix ``S(f)``.

    Given one simulation per incident Floquet mode ``j`` in ``modes`` (or a general
    set of excitations), assembles the complex matrix ``S[f, i, j]`` where ``i``
    indexes outgoing modes and ``j`` indexes incident modes.

    Parameters
    ----------
    excitations : dict
        Mapping ``incident_mode -> [port_below, port_above]`` for each excitation run.
    modes : sequence of FloquetMode, optional
        Ordered list of Floquet modes defining the rows and columns of ``S``. If
        ``None``, uses the keys of ``excitations`` in insertion order.
    incident_amplitudes : dict, optional
        Optional mapping ``incident_mode -> a_inc(f)`` when reference incident waves
        are supplied externally.
    reciprocity_convention : bool, optional
        Use the canonical reciprocity phase convention (default ``True``).

    Returns
    -------
    S : ndarray
        Complex scattering matrix of shape ``(n_freq, n_modes, n_modes)``.
    mode_list : list of FloquetMode
        Ordered list of :class:`FloquetMode` corresponding to axes 1 and 2 of ``S``.
    """
    norm_exc: dict[FloquetMode, Sequence[FloquetPort]] = {}
    for k, v in excitations.items():
        mk = k if isinstance(k, FloquetMode) else FloquetMode(*k)
        norm_exc[mk] = v

    norm_inc: dict[FloquetMode, np.ndarray] = {}
    if incident_amplitudes is not None:
        for k, v in incident_amplitudes.items():
            mk = k if isinstance(k, FloquetMode) else FloquetMode(*k)
            norm_inc[mk] = np.asarray(v, dtype=complex)

    if modes is None:
        mode_list = list(norm_exc.keys())
    else:
        mode_list = [
            m if isinstance(m, FloquetMode) else FloquetMode(*m) for m in modes
        ]

    first_ports = next(iter(norm_exc.values()))
    n_freq = first_ports[0].monitor.frequencies.size
    n_modes = len(mode_list)
    s_mat = np.full((n_freq, n_modes, n_modes), np.nan + 0j, dtype=complex)

    for j, inc_mode in enumerate(mode_list):
        if inc_mode not in norm_exc:
            continue
        col = s_parameters_from_ports(
            norm_exc[inc_mode],
            incident_mode=inc_mode,
            incident_amplitude=norm_inc.get(inc_mode),
            modes=mode_list,
            reciprocity_convention=reciprocity_convention,
        )
        for i, out_mode in enumerate(mode_list):
            s_mat[:, i, j] = col[out_mode]

    return s_mat, mode_list


def reciprocal_s_matrix(
    s_minus_k: np.ndarray,
    modes: Sequence[FloquetMode],
) -> np.ndarray:
    """Return the Lorentz-reciprocal transpose ``S^rec(k_0)`` from ``S(-k_0)``.

    By Lorentz reciprocity in a reciprocal medium, the generalized Floquet S-matrix
    at Bloch wavevector ``+k_0`` is related to the S-matrix at ``-k_0`` by
    ``S_{i, j}(+k_0) = S_{-j, -i}(-k_0)``, where ``-m = (port_m, -m_m, -n_m, pol_m)``.
    """
    mode_list = [m if isinstance(m, FloquetMode) else FloquetMode(*m) for m in modes]
    idx_map = {m: idx for idx, m in enumerate(mode_list)}
    n_m = len(mode_list)
    perm = np.zeros(n_m, dtype=int)
    for idx, m in enumerate(mode_list):
        partner = m.reciprocal_partner()
        if partner not in idx_map:
            raise ValueError(
                f"Reciprocal partner {partner} of {m} is not in modes list."
            )
        perm[idx] = idx_map[partner]
    # S_rec[:, i, j] = s_minus_k[:, perm[j], perm[i]]
    return np.swapaxes(s_minus_k[:, perm, :][:, :, perm], 1, 2)


@dataclass
class FloquetCell:
    """Specification of a periodic unit cell for an oblique-incidence Bloch sweep.

    Parameters
    ----------
    Lx, Ly : float
        Transverse unit-cell periods [m] along ``x`` and ``y``.
    zmin, zmax : float
        Computational domain bounds [m] along ``z`` (including PML regions).
    nx, ny, nz : int
        Number of grid cells along ``x``, ``y``, and ``z``.
    z_source : float
        Source plane coordinate [m] along ``z``.
    z_refl : float
        Reflection monitor coordinate [m] along ``z``.
    z_tran : float
        Transmission monitor coordinate [m] along ``z``.
    eps_below : float, optional
        Relative permittivity of the lower half-space (default 1.0).
    mu_below : float, optional
        Relative permeability of the lower half-space (default 1.0).
    eps_above : float, optional
        Relative permittivity of the upper half-space (default 1.0).
    mu_above : float, optional
        Relative permeability of the upper half-space (default 1.0).
    z_ref_below : float, optional
        Port 1 phase reference plane [m] (defaults to ``z_refl``).
    z_ref_above : float, optional
        Port 2 phase reference plane [m] (defaults to ``z_tran``).
    orders_x : sequence of int, optional
        Diffraction orders ``m`` along ``x`` to record (default ``(-1, 0, 1)``).
    orders_y : sequence of int, optional
        Diffraction orders ``n`` along ``y`` to record (default ``(0,)``).
    n_pml : int, optional
        Number of PML cells at ``zmin`` and ``zmax`` (default 10).
    pml_ profile / material_callback : callable, optional
        ``material_callback(solver)`` populates the structured region in ``solver.ieps``
        / ``solver.imu`` / ``solver.isigma`` after grid and PML initialization.
    run_time_cycles : float, optional
        Number of optical cycles at the center frequency to evolve when ``n_steps``
        is not explicitly set (default 25.0).
    n_steps : int, optional
        Explicit number of time steps per Bloch run.
    calibrate_incident : bool, optional
        If ``True``, run a homogeneous reference simulation in the incidence medium
        at each ``k_par`` to calibrate the exact discrete incident wave amplitude
        and phase (default ``True``).
    source_direction : {"up", "down"}, optional
        Propagation direction of the incident wave: ``"up"`` (``+z``, from the lower
        half-space) or ``"down"`` (``-z``, from the upper half-space). Default ``"up"``.
    cfl : float, optional
        Courant factor multiplier relative to the 3D FIT stability limit (default 0.95).
    """

    Lx: float
    Ly: float
    zmin: float
    zmax: float
    nx: int
    ny: int
    nz: int
    z_source: float
    z_refl: float
    z_tran: float
    eps_below: float = 1.0
    mu_below: float = 1.0
    eps_above: float = 1.0
    mu_above: float = 1.0
    z_ref_below: float | None = 0.0
    z_ref_above: float | None = 0.0
    orders_x: Sequence[int] = (-1, 0, 1)
    orders_y: Sequence[int] = (0,)
    n_pml: int = 10
    material_callback: Callable | None = None
    run_time_cycles: float = 25.0
    n_steps: int | None = None
    calibrate_incident: bool = False
    source_direction: str = "up"
    cfl: float = 0.95

    @property
    def n_inc(self) -> float:
        """Refractive index of the incidence half-space."""
        if self.source_direction == "up":
            return float(np.sqrt(self.eps_below * self.mu_below))
        return float(np.sqrt(self.eps_above * self.mu_above))

    @property
    def n_trn(self) -> float:
        """Refractive index of the transmission half-space."""
        if self.source_direction == "up":
            return float(np.sqrt(self.eps_above * self.mu_above))
        return float(np.sqrt(self.eps_below * self.mu_below))



class AngleSweepResult:
    """Container for broadband fixed-``k_par`` Bloch sweep data and constant-angle resampling.

    Stores complex Floquet port wave amplitudes, S-parameters, and diffraction
    efficiencies on the ``(k_par, frequency)`` grid and resamples them onto
    constant-angle curves ``theta`` via :func:`resample_constant_angle`.
    """

    def __init__(
        self,
        k_par: np.ndarray,
        frequencies: np.ndarray,
        angles: np.ndarray,
        polarization: str,
        phi: float,
        n_inc: float,
        n_trn: float,
        orders_x: Sequence[int],
        orders_y: Sequence[int],
        r_te: np.ndarray,
        r_tm: np.ndarray,
        t_te: np.ndarray,
        t_tm: np.ndarray,
        s_refl_te: np.ndarray,
        s_refl_tm: np.ndarray,
        s_tran_te: np.ndarray,
        s_tran_tm: np.ndarray,
        prop_refl: np.ndarray,
        prop_tran: np.ndarray,
        inc_power: np.ndarray,
        Lx: float = 1.0,
        Ly: float = 1.0,
    ):
        self.k_par = np.asarray(k_par, dtype=float)
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.angles = np.atleast_1d(np.asarray(angles, dtype=float))
        self.polarization = _normalize_pol(polarization)
        self.phi = float(phi)
        self.n_inc = float(n_inc)
        self.n_trn = float(n_trn)
        self.orders_x = tuple(int(m) for m in orders_x)
        self.orders_y = tuple(int(n) for n in orders_y)
        self.r_te = np.asarray(r_te, dtype=float)
        self.r_tm = np.asarray(r_tm, dtype=float)
        self.t_te = np.asarray(t_te, dtype=float)
        self.t_tm = np.asarray(t_tm, dtype=float)
        self.s_refl_te = np.asarray(s_refl_te, dtype=complex)
        self.s_refl_tm = np.asarray(s_refl_tm, dtype=complex)
        self.s_tran_te = np.asarray(s_tran_te, dtype=complex)
        self.s_tran_tm = np.asarray(s_tran_tm, dtype=complex)
        self.prop_refl = np.asarray(prop_refl, dtype=bool)
        self.prop_tran = np.asarray(prop_tran, dtype=bool)
        self.inc_power = np.asarray(inc_power, dtype=float)
        self.Lx = float(Lx)
        self.Ly = float(Ly)

    def _order_indices(self, m: int = 0, n: int = 0) -> tuple[int, int]:
        if m not in self.orders_x:
            raise KeyError(f"Order m={m} not in orders_x={self.orders_x}.")
        if n not in self.orders_y:
            raise KeyError(f"Order n={n} not in orders_y={self.orders_y}.")
        return self.orders_x.index(m), self.orders_y.index(n)

    @property
    def r_order_k(self) -> np.ndarray:
        """Total (TE + TM) reflection efficiency per order on the ``(k_par, f)`` grid."""
        return np.where(self.prop_refl, self.r_te + self.r_tm, np.nan)

    @property
    def t_order_k(self) -> np.ndarray:
        """Total (TE + TM) transmission efficiency per order on the ``(k_par, f)`` grid."""
        return np.where(self.prop_tran, self.t_te + self.t_tm, np.nan)

    def resample(
        self,
        angles: float | np.ndarray | None = None,
        index_shift: int = 0,
    ) -> dict[str, np.ndarray]:
        """Resample all efficiencies and S-parameters onto constant-angle curves.

        Parameters
        ----------
        angles : float or array_like, optional
            Target incidence angle(s) [rad]. Defaults to ``self.angles``.
        index_shift : int, optional
            Index offset passed to :func:`resample_constant_angle` (default 0;
            non-zero values are used for mutation testing).

        Returns
        -------
        dict of str to ndarray
            Dictionary with constant-angle arrays:
            - ``"R_TE"``, ``"R_TM"``, ``"R_order"``: reflection efficiencies per order
            - ``"T_TE"``, ``"T_TM"``, ``"T_order"``: transmission efficiencies per order
            - ``"S_refl_TE"``, ``"S_refl_TM"``: complex reflection Floquet S-parameters
            - ``"S_tran_TE"``, ``"S_tran_TM"``: complex transmission Floquet S-parameters
            - ``"R_total"``, ``"T_total"``: total reflected and transmitted power fractions
            - ``"R_00"``, ``"T_00"``: specular ``(0, 0)`` efficiencies
        """
        th = self.angles if angles is None else angles
        kw = dict(
            k_par=self.k_par,
            frequencies=self.frequencies,
            angles=th,
            n=self.n_inc,
            index_shift=index_shift,
        )
        s_r_te = resample_constant_angle(values=self.s_refl_te, propagating=self.prop_refl, **kw)
        s_r_tm = resample_constant_angle(values=self.s_refl_tm, propagating=self.prop_refl, **kw)
        s_t_te = resample_constant_angle(values=self.s_tran_te, propagating=self.prop_tran, **kw)
        s_t_tm = resample_constant_angle(values=self.s_tran_tm, propagating=self.prop_tran, **kw)

        r_te_lin = resample_constant_angle(
            values=self.r_te, propagating=self.prop_refl, **kw
        )
        r_tm_lin = resample_constant_angle(
            values=self.r_tm, propagating=self.prop_refl, **kw
        )
        t_te_lin = resample_constant_angle(
            values=self.t_te, propagating=self.prop_tran, **kw
        )
        t_tm_lin = resample_constant_angle(
            values=self.t_tm, propagating=self.prop_tran, **kw
        )

        r_te = np.abs(s_r_te) ** 2
        r_tm = np.abs(s_r_tm) ** 2
        t_te = np.maximum(0.0, t_te_lin)
        t_tm = np.maximum(0.0, t_tm_lin)

        if self.k_par.size > 1 and (len(self.orders_x) > 1 or len(self.orders_y) > 1):
            self._refine_multiorder_cutoff_resampling(
                th,
                index_shift,
                s_r_te,
                s_r_tm,
                s_t_te,
                s_t_tm,
                r_te_lin,
                r_tm_lin,
                t_te_lin,
                t_tm_lin,
            )
            r_te = np.abs(s_r_te) ** 2
            r_tm = np.abs(s_r_tm) ** 2
            t_te = np.maximum(0.0, t_te_lin)
            t_tm = np.maximum(0.0, t_tm_lin)

        im0, in0 = self._order_indices(0, 0)
        # Compensate the specular transmission for the quadratic reflection remainder
        # sum_m (L_w(|s_r|^2) - |L_w(s_r)|^2) across reflection zeros
        delta_r_te = np.nansum(
            np.where(np.isfinite(r_te), r_te_lin - r_te, 0.0), axis=(-2, -1)
        )
        delta_r_tm = np.nansum(
            np.where(np.isfinite(r_tm), r_tm_lin - r_tm, 0.0), axis=(-2, -1)
        )
        t_te[..., im0, in0] = np.maximum(0.0, t_te[..., im0, in0] + delta_r_te)
        t_tm[..., im0, in0] = np.maximum(0.0, t_tm[..., im0, in0] + delta_r_tm)

        r_ord = r_te + r_tm
        t_ord = t_te + t_tm

        # Sum over propagating orders at each (angle, freq)
        r_tot = np.nansum(r_ord, axis=(-2, -1))
        t_tot = np.nansum(t_ord, axis=(-2, -1))
        all_nan_r = np.all(np.isnan(r_ord), axis=(-2, -1))
        all_nan_t = np.all(np.isnan(t_ord), axis=(-2, -1))
        r_tot = np.where(all_nan_r, np.nan, r_tot)
        t_tot = np.where(all_nan_t, np.nan, t_tot)

        im0, in0 = self._order_indices(0, 0)
        return {
            "R_TE": r_te,
            "R_TM": r_tm,
            "R_order": r_ord,
            "T_TE": t_te,
            "T_TM": t_tm,
            "T_order": t_ord,
            "S_refl_TE": s_r_te,
            "S_refl_TM": s_r_tm,
            "S_tran_TE": s_t_te,
            "S_tran_TM": s_t_tm,
            "R_total": r_tot,
            "T_total": t_tot,
            "R_00": r_ord[..., im0, in0],
            "T_00": t_ord[..., im0, in0],
        }

    def _refine_multiorder_cutoff_resampling(
        self,
        angles: float | np.ndarray,
        index_shift: int,
        s_r_te: np.ndarray,
        s_r_tm: np.ndarray,
        s_t_te: np.ndarray,
        s_t_tm: np.ndarray,
        r_te_lin: np.ndarray,
        r_tm_lin: np.ndarray,
        t_te_lin: np.ndarray,
        t_tm_lin: np.ndarray,
    ) -> None:
        """Synchronize multi-order stencils and interpolate cutoff-straddling orders in ``k_z``."""
        scalar_angle = np.ndim(angles) == 0
        th = np.atleast_1d(np.asarray(angles, dtype=float))
        if scalar_angle:
            s_r_te = s_r_te[None, ...]
            s_r_tm = s_r_tm[None, ...]
            s_t_te = s_t_te[None, ...]
            s_t_tm = s_t_tm[None, ...]
            r_te_lin = r_te_lin[None, ...]
            r_tm_lin = r_tm_lin[None, ...]
            t_te_lin = t_te_lin[None, ...]
            t_tm_lin = t_tm_lin[None, ...]

        n_k = self.k_par.size
        n_f = self.frequencies.size
        dk = float(self.k_par[1] - self.k_par[0])
        gx = 2.0 * np.pi * np.asarray(self.orders_x, dtype=float)[:, None] / self.Lx
        gy = 2.0 * np.pi * np.asarray(self.orders_y, dtype=float)[None, :] / self.Ly
        cos_p = float(np.cos(self.phi))
        sin_p = float(np.sin(self.phi))

        def _kt_sq(kp: float) -> np.ndarray:
            return (kp * cos_p + gx) ** 2 + (kp * sin_p + gy) ** 2

        kt_sq_grid = np.stack([_kt_sq(float(kp)) for kp in self.k_par], axis=0)

        for ia, theta in enumerate(th):
            if theta < 0.0 or theta >= 0.5 * np.pi:
                continue
            for jf in range(n_f):
                fj = float(self.frequencies[jf])
                k0 = 2.0 * np.pi * fj / c
                kt = k0 * self.n_inc * float(np.sin(theta))
                u = (kt - self.k_par[0]) / dk + index_shift
                if u < -1e-11 or u > (n_k - 1) + 1e-11:
                    continue
                u = float(np.clip(u, 0.0, float(n_k - 1)))
                nearest = int(np.rint(u))
                on_node = abs(u - nearest) <= 1e-11
                k_eval = float(self.k_par[0] + u * dk)

                kz_r_eval = (k0 * self.n_inc) ** 2 - _kt_sq(k_eval)
                kz_t_eval = (k0 * self.n_trn) ** 2 - _kt_sq(k_eval)

                # Mask orders that are evanescent in the continuum at k_eval
                ev_r = kz_r_eval <= 0.0
                ev_t = kz_t_eval <= 0.0
                if np.any(ev_r):
                    s_r_te[ia, jf, ev_r] = np.nan + 0j
                    s_r_tm[ia, jf, ev_r] = np.nan + 0j
                    r_te_lin[ia, jf, ev_r] = np.nan
                    r_tm_lin[ia, jf, ev_r] = np.nan
                if np.any(ev_t):
                    s_t_te[ia, jf, ev_t] = np.nan + 0j
                    s_t_tm[ia, jf, ev_t] = np.nan + 0j
                    t_te_lin[ia, jf, ev_t] = np.nan
                    t_tm_lin[ia, jf, ev_t] = np.nan

                if on_node:
                    continue

                cell_i = min(max(int(np.floor(u)), 0), n_k - 2)
                kz_r_grid = (k0 * self.n_inc) ** 2 - kt_sq_grid
                kz_t_grid = (k0 * self.n_trn) ** 2 - kt_sq_grid
                v_r = (
                    self.prop_refl[:, jf]
                    & (kz_r_grid > 0.0)
                    & np.isfinite(self.r_te[:, jf])
                )
                v_t = (
                    self.prop_tran[:, jf]
                    & (kz_t_grid > 0.0)
                    & np.isfinite(self.t_te[:, jf])
                )

                # 1. Interpolate cutoff-straddling orders that propagate at k_eval
                #    in the normal wavenumber variable q = k_{z,mn}(k_par) with q=0 at cutoff.
                p_cut = 0.0
                for (
                    v_s,
                    kz_eval_s,
                    kz_grid_s,
                    eta_te_grid,
                    eta_tm_grid,
                    s_te_grid,
                    s_tm_grid,
                    eta_te_out,
                    eta_tm_out,
                    s_te_out,
                    s_tm_out,
                ) in (
                    (
                        v_r,
                        kz_r_eval,
                        kz_r_grid,
                        self.r_te[:, jf],
                        self.r_tm[:, jf],
                        self.s_refl_te[:, jf],
                        self.s_refl_tm[:, jf],
                        r_te_lin[ia, jf],
                        r_tm_lin[ia, jf],
                        s_r_te[ia, jf],
                        s_r_tm[ia, jf],
                    ),
                    (
                        v_t,
                        kz_t_eval,
                        kz_t_grid,
                        self.t_te[:, jf],
                        self.t_tm[:, jf],
                        self.s_tran_te[:, jf],
                        self.s_tran_tm[:, jf],
                        t_te_lin[ia, jf],
                        t_tm_lin[ia, jf],
                        s_t_te[ia, jf],
                        s_t_tm[ia, jf],
                    ),
                ):
                    straddle = (kz_eval_s > 0.0) & (v_s[cell_i] != v_s[cell_i + 1])
                    for im, in_ in np.argwhere(straddle):
                        idx_prop: list[int] = []
                        if v_s[cell_i, im, in_]:
                            r_idx = cell_i
                            while r_idx >= 0 and v_s[r_idx, im, in_] and len(idx_prop) < 2:
                                idx_prop.append(r_idx)
                                r_idx -= 1
                        else:
                            r_idx = cell_i + 1
                            while r_idx < n_k and v_s[r_idx, im, in_] and len(idx_prop) < 2:
                                idx_prop.append(r_idx)
                                r_idx += 1
                        if not idx_prop:
                            continue
                        q_nodes = np.array(
                            [0.0] + [float(np.sqrt(kz_grid_s[r, im, in_])) for r in idx_prop]
                        )
                        q_eval = float(np.sqrt(kz_eval_s[im, in_]))
                        w_q = _lagrange_weights_nodes(q_eval, q_nodes)[1:]

                        val_te = max(0.0, float(np.dot(w_q, eta_te_grid[idx_prop, im, in_])))
                        val_tm = max(0.0, float(np.dot(w_q, eta_tm_grid[idx_prop, im, in_])))
                        eta_te_out[im, in_] = val_te
                        eta_tm_out[im, in_] = val_tm
                        p_cut += val_te + val_tm

                        for val_p, s_grid_p, s_out_p in (
                            (val_te, s_te_grid, s_te_out),
                            (val_tm, s_tm_grid, s_tm_out),
                        ):
                            s_raw = complex(np.dot(w_q, s_grid_p[idx_prop, im, in_]))
                            phase = s_raw / abs(s_raw) if abs(s_raw) > 1e-30 else 1.0 + 0j
                            s_out_p[im, in_] = np.sqrt(val_p) * phase

                # 2. When a Rayleigh cutoff is straddled or crossed by the local stencil,
                #    reconcile the interior orders' sum with the interpolated raw FDTD
                #    total energy E_raw(k_i) so opening/closing orders do not create an
                #    artificial energy jump.
                int_r = (kz_r_eval > 0.0) & v_r[cell_i] & v_r[cell_i + 1]
                int_t = (kz_t_eval > 0.0) & v_t[cell_i] & v_t[cell_i + 1]
                if not (np.any(int_r) or np.any(int_t)):
                    continue

                order_pts = min(4, n_k)
                j0 = min(max(cell_i - 1, 0), n_k - order_pts)
                s_rel = u - j0
                w = _lagrange_weights(s_rel, order_pts)

                e_raw = np.zeros(order_pts, dtype=float)
                e_int = np.zeros(order_pts, dtype=float)
                for r_loc, k_idx in enumerate(range(j0, j0 + order_pts)):
                    pr_k = self.prop_refl[k_idx, jf]
                    pt_k = self.prop_tran[k_idx, jf]
                    e_raw[r_loc] = float(
                        np.nansum(
                            np.where(pr_k, self.r_te[k_idx, jf] + self.r_tm[k_idx, jf], 0.0)
                        )
                        + np.nansum(
                            np.where(pt_k, self.t_te[k_idx, jf] + self.t_tm[k_idx, jf], 0.0)
                        )
                    )
                    e_int[r_loc] = float(
                        np.nansum(
                            np.where(
                                v_r[k_idx] & int_r,
                                self.r_te[k_idx, jf] + self.r_tm[k_idx, jf],
                                0.0,
                            )
                        )
                        + np.nansum(
                            np.where(
                                v_t[k_idx] & int_t,
                                self.t_te[k_idx, jf] + self.t_tm[k_idx, jf],
                                0.0,
                            )
                        )
                    )

                e_target = float(np.dot(w, e_raw))
                e_int_interp = float(
                    np.nansum(r_te_lin[ia, jf, int_r] + r_tm_lin[ia, jf, int_r])
                    + np.nansum(t_te_lin[ia, jf, int_t] + t_tm_lin[ia, jf, int_t])
                )
                if e_int_interp > 1e-15 and (
                    p_cut > 0.0 or np.any(np.abs(e_raw - e_int) > 1e-12)
                ):
                    beta = max(0.0, e_target - p_cut) / e_int_interp
                    sqrt_beta = float(np.sqrt(beta))
                    r_te_lin[ia, jf, int_r] *= beta
                    r_tm_lin[ia, jf, int_r] *= beta
                    s_r_te[ia, jf, int_r] *= sqrt_beta
                    s_r_tm[ia, jf, int_r] *= sqrt_beta
                    t_te_lin[ia, jf, int_t] *= beta
                    t_tm_lin[ia, jf, int_t] *= beta
                    s_t_te[ia, jf, int_t] *= sqrt_beta
                    s_t_tm[ia, jf, int_t] *= sqrt_beta

    def order_efficiency(
        self,
        m: int = 0,
        n: int = 0,
        side: str = "tran",
        pol: str = "total",
        angles: float | np.ndarray | None = None,
        index_shift: int = 0,
    ) -> np.ndarray:
        """Return constant-angle diffraction efficiency for order ``(m, n)``."""
        im, in_ = self._order_indices(m, n)
        res = self.resample(angles=angles, index_shift=index_shift)
        prefix = "R" if side.lower().startswith("r") else "T"
        if pol.lower() == "total":
            return res[f"{prefix}_order"][..., im, in_]
        p_norm = _normalize_pol(pol)
        return res[f"{prefix}_{p_norm}"][..., im, in_]

    def save(self, path: str | Path) -> None:
        """Save the sweep result to a compressed ``.npz`` archive."""
        np.savez_compressed(
            path,
            k_par=self.k_par,
            frequencies=self.frequencies,
            angles=self.angles,
            polarization=np.array(self.polarization),
            phi=np.array(self.phi),
            n_inc=np.array(self.n_inc),
            n_trn=np.array(self.n_trn),
            orders_x=np.asarray(self.orders_x, dtype=int),
            orders_y=np.asarray(self.orders_y, dtype=int),
            r_te=self.r_te,
            r_tm=self.r_tm,
            t_te=self.t_te,
            t_tm=self.t_tm,
            s_refl_te=self.s_refl_te,
            s_refl_tm=self.s_refl_tm,
            s_tran_te=self.s_tran_te,
            s_tran_tm=self.s_tran_tm,
            prop_refl=self.prop_refl,
            prop_tran=self.prop_tran,
            inc_power=self.inc_power,
            Lx=np.array(self.Lx),
            Ly=np.array(self.Ly),
        )

    @classmethod
    def load(cls, path: str | Path) -> AngleSweepResult:
        """Load an :class:`AngleSweepResult` from a ``.npz`` archive."""
        with np.load(path, allow_pickle=False) as data:
            return cls(
                k_par=data["k_par"],
                frequencies=data["frequencies"],
                angles=data["angles"],
                polarization=str(data["polarization"]),
                phi=float(data["phi"]),
                n_inc=float(data["n_inc"]),
                n_trn=float(data["n_trn"]),
                orders_x=tuple(int(x) for x in data["orders_x"]),
                orders_y=tuple(int(y) for y in data["orders_y"]),
                r_te=data["r_te"],
                r_tm=data["r_tm"],
                t_te=data["t_te"],
                t_tm=data["t_tm"],
                s_refl_te=data["s_refl_te"],
                s_refl_tm=data["s_refl_tm"],
                s_tran_te=data["s_tran_te"],
                s_tran_tm=data["s_tran_tm"],
                prop_refl=data["prop_refl"],
                prop_tran=data["prop_tran"],
                inc_power=data["inc_power"],
                Lx=float(data["Lx"]),
                Ly=float(data["Ly"]),
            )



def _build_plane_wave_source(
    cell: FloquetCell,
    k_par: float,
    phi: float,
    polarization: str,
    f0: float,
    fwidth: float,
) -> tuple[list[CurrentSource], GaussianSource, tuple[float, float, float]]:
    """Construct the Bloch-phased sheet :class:`CurrentSource` list and ``k_bloch`` tuple.

    In Wakis's DFT convention (``exp(-2*pi*i*f*t)`` kernel, i.e., ``exp(+2*pi*i*f*t)``
    phasors at ``f > 0``), a physical plane wave with transverse wavevector
    ``+k_t = (k_x, k_y, 0)`` has spatial dependence ``exp(-i*(k_x*x + k_y*y))``,
    corresponding to ``solver.k_bloch = (-k_x, -k_y, 0.0)``.
    """
    kx = float(k_par) * float(np.cos(phi))
    ky = float(k_par) * float(np.sin(phi))
    k_bloch = (-kx, -ky, 0.0)

    pol_norm = _normalize_pol(polarization)
    # Transverse polarization unit vector in the xy plane matching DiffractionMonitor:
    #   s_hat (TE) = (-sin phi, cos phi, 0) -> (0, 1, 0) at phi == 0
    #   k_hat (TM) = ( cos phi, sin phi, 0) -> (1, 0, 0) at phi == 0
    if pol_norm == "TE":
        jx = float(-np.sin(phi))
        jy = float(np.cos(phi))
    else:
        jx = float(np.cos(phi))
        jy = float(np.sin(phi))

    def _amp(x, y, z):
        return np.exp(-1j * (kx * x + ky * y))

    profile = GaussianSource(frequency=f0, fwidth=fwidth)
    sources: list[CurrentSource] = []
    for comp, weight in (("x", jx), ("y", jy)):
        if abs(weight) > 1e-14:
            sources.append(
                CurrentSource(
                    profile,
                    component=comp,
                    center=(0.0, 0.0, float(cell.z_source)),
                    size=(float(cell.Lx), float(cell.Ly), 0.0),
                    amplitude=float(weight),
                    amp_func=_amp,
                )
            )
    return sources, profile, k_bloch


def _run_single_Solver(
    cell: FloquetCell,
    k_par: float,
    phi: float,
    polarization: str,
    frequencies: np.ndarray,
    homogeneous: bool = False,
) -> tuple[FloquetPort | None, FloquetPort | None]:
    """Run one 3D FIT Bloch simulation for a :class:`FloquetCell` at ``k_par``."""
    grid = GridFIT3D(
        -0.5 * cell.Lx,
        0.5 * cell.Lx,
        -0.5 * cell.Ly,
        0.5 * cell.Ly,
        cell.zmin,
        cell.zmax,
        cell.nx,
        cell.ny,
        cell.nz,
        verbose=0,
    )

    f_arr = np.asarray(frequencies, dtype=float)
    f_min = float(np.min(np.abs(f_arr)))
    f_max = float(np.max(np.abs(f_arr)))
    f0 = 0.5 * (f_min + f_max)
    default_bw = max(0.30 * f0, 0.70 * (f_max - f_min))

    # Avoid exciting grazing frequencies at or below the light-line cutoff
    # f_c = c * |k_par| / (2*pi*n_inc), where the normal group velocity vanishes.
    f_cut = float(c * abs(k_par) / (2.0 * np.pi * cell.n_inc))
    if f_cut > 0.0:
        f0 = max(f0, 0.5 * (max(f_min, f_cut) + f_max) + 0.06 * f_max)
        fwidth = min(default_bw, 0.28 * max(f0 - f_cut, 0.08 * f0))
    else:
        fwidth = default_bw

    sources, profile, k_bloch = _build_plane_wave_source(
        cell, k_par=k_par, phi=phi, polarization=polarization, f0=f0, fwidth=fwidth
    )

    solver = SolverFIT3D(
        grid,
        bc_low=["bloch", "bloch", "cpml"],
        bc_high=["bloch", "bloch", "cpml"],
        n_pml=cell.n_pml,
        k_bloch=k_bloch,
        dtype=np.complex128,
        cfln=cell.cfl,
        verbose=0,
    )

    # Set background half-space permittivities and permeabilities
    if homogeneous:
        eps_bg = cell.eps_below if cell.source_direction == "up" else cell.eps_above
        mu_bg = cell.mu_below if cell.source_direction == "up" else cell.mu_above
        for comp in ("x", "y", "z"):
            solver.ieps[:, :, :, comp] = 1.0 / (epsilon_0 * eps_bg)
            solver.imu[:, :, :, comp] = 1.0 / (mu_0 * mu_bg)
    else:
        z_split = 0.5 * (cell.z_refl + cell.z_tran)
        iz_split = int(np.searchsorted(grid.z[:-1], z_split))
        for comp in ("x", "y", "z"):
            solver.ieps[:, :, :iz_split, comp] = 1.0 / (epsilon_0 * cell.eps_below)
            solver.imu[:, :, :iz_split, comp] = 1.0 / (mu_0 * cell.mu_below)
            solver.ieps[:, :, iz_split:, comp] = 1.0 / (epsilon_0 * cell.eps_above)
            solver.imu[:, :, iz_split:, comp] = 1.0 / (mu_0 * cell.mu_above)
        if cell.material_callback is not None:
            cell.material_callback(solver)

    if cell.n_steps is not None:
        n_steps = int(cell.n_steps)
    else:
        t_total = float(profile.end_time) + float(cell.run_time_cycles) / f0
        n_steps = int(np.ceil(t_total / solver.dt))

    plane_size = (float(cell.Lx), float(cell.Ly), 0.0)
    mon_below = DiffractionMonitor(
        f_arr,
        center=(0.0, 0.0, float(cell.z_refl)),
        size=plane_size,
    )
    mon_above = DiffractionMonitor(
        f_arr,
        center=(0.0, 0.0, float(cell.z_tran)),
        size=plane_size,
    )

    active_monitors = []
    has_below = True
    has_above = True
    try:
        mon_below._initialize(solver)
        mon_below.is_first_update = False
        active_monitors.append(mon_below)
    except ValueError:
        has_below = False

    try:
        mon_above._initialize(solver)
        mon_above.is_first_update = False
        active_monitors.append(mon_above)
    except ValueError:
        has_above = False

    src_arg = sources[0] if len(sources) == 1 else sources
    solver.emsolve(n_steps, source=src_arg, monitors=active_monitors)

    dz = float(grid.dz[0])
    port_below = (
        FloquetPort(
            mon_below,
            side="below",
            reference_position=cell.z_ref_below,
            port_id=1,
            dn=dz,
        )
        if has_below
        else None
    )
    port_above = (
        FloquetPort(
            mon_above,
            side="above",
            reference_position=cell.z_ref_above,
            port_id=2,
            dn=dz,
        )
        if has_above
        else None
    )
    return port_below, port_above


def _extract_port_orders(
    port: FloquetPort | None,
    n_freq: int,
    orders_x: Sequence[int],
    orders_y: Sequence[int],
) -> dict[str, np.ndarray]:
    """Extract requested ``(orders_x, orders_y)`` slice from a :class:`FloquetPort`."""
    nx_ord = len(orders_x)
    ny_ord = len(orders_y)
    shape = (n_freq, nx_ord, ny_ord)
    out = {
        "a_in_TE": np.full(shape, np.nan + 0j, dtype=complex),
        "a_in_TM": np.full(shape, np.nan + 0j, dtype=complex),
        "b_out_TE": np.full(shape, np.nan + 0j, dtype=complex),
        "b_out_TM": np.full(shape, np.nan + 0j, dtype=complex),
        "propagating": np.zeros(shape, dtype=bool),
    }
    if port is None:
        return out
    waves = port.wave_amplitudes()
    for im_req, m in enumerate(orders_x):
        for in_req, n in enumerate(orders_y):
            im, in_ = port._find_order_indices(m, n)
            if im is not None and in_ is not None:
                for key in ("a_in_TE", "a_in_TM", "b_out_TE", "b_out_TM"):
                    out[key][:, im_req, in_req] = waves[key][:, im, in_]
                out["propagating"][:, im_req, in_req] = waves["propagating"][:, im, in_]
    return out


def _execute_k_task(args: tuple) -> dict[str, Any]:
    """Worker entrypoint executing a single ``k_par`` task for :class:`AngleSweep`."""
    idx, k_val, runner, frequencies, polarization, phi, orders_x, orders_y = args
    f_arr = np.asarray(frequencies, dtype=float)
    out = runner(float(k_val), f_arr, polarization, float(phi))

    if isinstance(out, dict):
        res = dict(out)
        res["index"] = int(idx)
        res["k_par"] = float(k_val)
        return res

    if len(out) == 2:
        port_below, port_above = out
        inc_port = None
    elif len(out) == 3:
        port_below, port_above, inc_port = out
    else:
        raise ValueError(f"Runner returned unexpected tuple of length {len(out)}.")

    pol_norm = _normalize_pol(polarization)
    inc_side_port = port_below if inc_port is None else inc_port
    if inc_side_port is None:
        raise RuntimeError(f"Incident port has no propagating orders at k_par={k_val}.")
    inc_mode = FloquetMode(inc_side_port.port_id, 0, 0, pol_norm)

    a_inc = inc_side_port.mode_amplitude(inc_mode, direction="in")
    p_inc = np.abs(a_inc) ** 2

    w_below = _extract_port_orders(port_below, f_arr.size, orders_x, orders_y)
    w_above = _extract_port_orders(port_above, f_arr.size, orders_x, orders_y)

    b_refl_te = w_below["b_out_TE"]
    b_refl_tm = w_below["b_out_TM"]
    b_tran_te = w_above["b_out_TE"]
    b_tran_tm = w_above["b_out_TM"]

    a_inc_3d = a_inc[:, None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        s_refl_te = np.where(np.abs(a_inc_3d) > 0, b_refl_te / a_inc_3d, np.nan + 0j)
        s_refl_tm = np.where(np.abs(a_inc_3d) > 0, b_refl_tm / a_inc_3d, np.nan + 0j)
        s_tran_te = np.where(np.abs(a_inc_3d) > 0, b_tran_te / a_inc_3d, np.nan + 0j)
        s_tran_tm = np.where(np.abs(a_inc_3d) > 0, b_tran_tm / a_inc_3d, np.nan + 0j)

    r_te = np.abs(s_refl_te) ** 2
    r_tm = np.abs(s_refl_tm) ** 2
    t_te = np.abs(s_tran_te) ** 2
    t_tm = np.abs(s_tran_tm) ** 2

    return {
        "index": int(idx),
        "k_par": float(k_val),
        "r_te": r_te,
        "r_tm": r_tm,
        "t_te": t_te,
        "t_tm": t_tm,
        "s_refl_te": s_refl_te,
        "s_refl_tm": s_refl_tm,
        "s_tran_te": s_tran_te,
        "s_tran_tm": s_tran_tm,
        "prop_refl": np.asarray(w_below["propagating"], dtype=bool),
        "prop_tran": np.asarray(w_above["propagating"], dtype=bool),
        "inc_power": np.asarray(p_inc, dtype=float),
    }



class _CellRunner:
    """Picklable runner wrapping a :class:`FloquetCell` or builder callable."""

    def __init__(self, cell_or_builder: FloquetCell | Callable):
        self.cell_or_builder = cell_or_builder

    def _resolve_cell(self, k_par: float, frequencies: np.ndarray, polarization: str, phi: float) -> FloquetCell | Any:
        if isinstance(self.cell_or_builder, FloquetCell):
            return self.cell_or_builder
        sig = inspect.signature(self.cell_or_builder)
        n_params = len(sig.parameters)
        if n_params == 0:
            return self.cell_or_builder()
        if n_params == 1:
            return self.cell_or_builder(k_par)
        return self.cell_or_builder(k_par, frequencies, polarization, phi)

    def __call__(self, k_par: float, frequencies: np.ndarray, polarization: str, phi: float):
        resolved = self._resolve_cell(k_par, frequencies, polarization, phi)
        if not isinstance(resolved, FloquetCell):
            # Builder already returned (port_below, port_above[, inc_port]) or dict
            return resolved
        cell = resolved
        port_below, port_above = _run_single_Solver(
            cell,
            k_par=k_par,
            phi=phi,
            polarization=polarization,
            frequencies=frequencies,
            homogeneous=False,
        )
        if cell.source_direction == "down":
            # Swap incident/transmission ports so port_below in _execute_k_task is the incident port
            port_inc_side, port_trn_side = port_above, port_below
        else:
            port_inc_side, port_trn_side = port_below, port_above

        if cell.calibrate_incident:
            ref_below, ref_above = _run_single_Solver(
                cell,
                k_par=k_par,
                phi=phi,
                polarization=polarization,
                frequencies=frequencies,
                homogeneous=True,
            )
            ref_inc = ref_above if cell.source_direction == "down" else ref_below
            return port_inc_side, port_trn_side, ref_inc
        return port_inc_side, port_trn_side


class AngleSweep:
    """Driver for broadband oblique-incidence Bloch sweeps at constant angle.

    Given a :class:`FloquetCell` (or unit-cell builder callable), polarization,
    frequency band ``[f_min, f_max]``, and target angles ``angles``, selects a
    uniform grid of tangential wavevectors ``k_par`` whose lines
    ``omega = c * k_par / (n * sin(theta))`` cover the requested ``(theta, f)``
    points at the stated sampling density, runs one independent Bloch simulation
    per ``k_par`` (locally in parallel via :mod:`multiprocessing` or via a
    ``--list-tasks`` / ``--task`` batch CLI), and resamples the complex Floquet
    amplitudes and diffraction efficiencies onto constant-angle curves.
    """

    def __init__(
        self,
        cell_builder: FloquetCell | Callable,
        frequencies: Sequence[float] | np.ndarray,
        angles: float | Sequence[float] | np.ndarray,
        polarization: str = "TE",
        phi: float = 0.0,
        n_k: int | None = None,
        dk_max: float | None = None,
        dtheta_max: float | None = None,
        n_inc: float | None = None,
        n_trn: float | None = None,
        orders_x: Sequence[int] | None = None,
        orders_y: Sequence[int] | None = None,
        Lx: float | None = None,
        Ly: float | None = None,
        pad_points: int = 1,
    ):
        self.cell_builder = cell_builder
        self.frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        self.angles = np.atleast_1d(np.asarray(angles, dtype=float))
        self.polarization = _normalize_pol(polarization)
        self.phi = float(phi)

        # Infer metadata from FloquetCell if available
        proto_cell: FloquetCell | None = None
        if isinstance(cell_builder, FloquetCell):
            proto_cell = cell_builder
        elif callable(cell_builder):
            sig = inspect.signature(cell_builder)
            if len(sig.parameters) == 0:
                maybe_cell = cell_builder()
                if isinstance(maybe_cell, FloquetCell):
                    proto_cell = maybe_cell

        self.n_inc = float(n_inc if n_inc is not None else (proto_cell.n_inc if proto_cell else 1.0))
        self.n_trn = float(n_trn if n_trn is not None else (proto_cell.n_trn if proto_cell else 1.0))
        self.orders_x = tuple(
            int(m)
            for m in (
                orders_x
                if orders_x is not None
                else (proto_cell.orders_x if proto_cell else (-1, 0, 1))
            )
        )
        self.orders_y = tuple(
            int(n)
            for n in (
                orders_y
                if orders_y is not None
                else (proto_cell.orders_y if proto_cell else (0,))
            )
        )
        self.Lx = float(Lx if Lx is not None else (proto_cell.Lx if proto_cell else 1.0))
        self.Ly = float(Ly if Ly is not None else (proto_cell.Ly if proto_cell else 1.0))

        self.k_par = k_parallel_grid(
            self.frequencies,
            self.angles,
            n=self.n_inc,
            n_k=n_k,
            dk_max=dk_max,
            dtheta_max=dtheta_max,
            pad_points=pad_points,
        )
        self.density = sampling_density(self.k_par, self.frequencies, n=self.n_inc)
        self._runner = _CellRunner(cell_builder)

    def list_tasks(self) -> list[dict[str, float | int]]:
        """Return metadata for each independent ``k_par`` simulation task."""
        return [
            {
                "index": int(i),
                "k_par": float(kv),
                "kx": float(kv * np.cos(self.phi)),
                "ky": float(kv * np.sin(self.phi)),
            }
            for i, kv in enumerate(self.k_par)
        ]

    def run_task(
        self,
        task_index: int,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Execute a single ``k_par`` task by index (and optionally save to ``.npz``)."""
        idx = int(task_index)
        if idx < 0 or idx >= self.k_par.size:
            raise IndexError(f"Task index {idx} out of range [0, {self.k_par.size - 1}].")
        args = (
            idx,
            float(self.k_par[idx]),
            self._runner,
            self.frequencies,
            self.polarization,
            self.phi,
            self.orders_x,
            self.orders_y,
        )
        res = _execute_k_task(args)
        if output_path is not None:
            np.savez_compressed(output_path, **res)
        return res

    def assemble(self, task_results: Sequence[dict[str, Any]]) -> AngleSweepResult:
        """Assemble per-``k_par`` task dictionaries into an :class:`AngleSweepResult`."""
        ordered = sorted(task_results, key=lambda d: int(d["index"]))
        if len(ordered) != self.k_par.size:
            raise ValueError(
                f"Expected {self.k_par.size} task results, got {len(ordered)}."
            )
        r_te = np.stack([d["r_te"] for d in ordered], axis=0)
        r_tm = np.stack([d["r_tm"] for d in ordered], axis=0)
        t_te = np.stack([d["t_te"] for d in ordered], axis=0)
        t_tm = np.stack([d["t_tm"] for d in ordered], axis=0)
        s_refl_te = np.stack([d["s_refl_te"] for d in ordered], axis=0)
        s_refl_tm = np.stack([d["s_refl_tm"] for d in ordered], axis=0)
        s_tran_te = np.stack([d["s_tran_te"] for d in ordered], axis=0)
        s_tran_tm = np.stack([d["s_tran_tm"] for d in ordered], axis=0)
        prop_refl = np.stack([d["prop_refl"] for d in ordered], axis=0)
        prop_tran = np.stack([d["prop_tran"] for d in ordered], axis=0)
        inc_power = np.stack([d["inc_power"] for d in ordered], axis=0)

        return AngleSweepResult(
            k_par=self.k_par,
            frequencies=self.frequencies,
            angles=self.angles,
            polarization=self.polarization,
            phi=self.phi,
            n_inc=self.n_inc,
            n_trn=self.n_trn,
            orders_x=self.orders_x,
            orders_y=self.orders_y,
            r_te=r_te,
            r_tm=r_tm,
            t_te=t_te,
            t_tm=t_tm,
            s_refl_te=s_refl_te,
            s_refl_tm=s_refl_tm,
            s_tran_te=s_tran_te,
            s_tran_tm=s_tran_tm,
            prop_refl=prop_refl,
            prop_tran=prop_tran,
            inc_power=inc_power,
            Lx=self.Lx,
            Ly=self.Ly,
        )

    def run(self, processes: int = 1) -> AngleSweepResult:
        """Execute all ``k_par`` tasks (serially or via multiprocessing) and assemble."""
        task_args = [
            (
                idx,
                float(kv),
                self._runner,
                self.frequencies,
                self.polarization,
                self.phi,
                self.orders_x,
                self.orders_y,
            )
            for idx, kv in enumerate(self.k_par)
        ]
        if processes is not None and int(processes) > 1 and len(task_args) > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=int(processes)) as pool:
                results = pool.map(_execute_k_task, task_args)
        else:
            results = [_execute_k_task(a) for a in task_args]
        return self.assemble(results)

    def cli(self, argv: Sequence[str] | None = None) -> Any:
        """Command-line interface supporting ``--list-tasks``, ``--task``, and ``--collect``.

        Parameters
        ----------
        argv : sequence of str, optional
            Command-line arguments (defaults to ``sys.argv[1:]``).

        Returns
        -------
        list or dict or AngleSweepResult
            - ``--list-tasks``: prints JSON task list and returns ``self.list_tasks()``.
            - ``--task I [--output PATH]``: runs task ``I`` and returns its result dict.
            - ``--collect FILE1 FILE2 ... [--output PATH]``: loads per-task ``.npz`` files,
              assembles :class:`AngleSweepResult`, optionally saves it, and returns it.
        """
        parser = argparse.ArgumentParser(
            description="Wakis oblique-incidence Bloch angle-sweep runner."
        )
        parser.add_argument(
            "--list-tasks",
            action="store_true",
            help="Print the list of independent k_par tasks as JSON and exit.",
        )
        parser.add_argument(
            "--task",
            type=int,
            default=None,
            help="Run only the specified 0-based k_par task index.",
        )
        parser.add_argument(
            "--collect",
            nargs="+",
            default=None,
            help="Collect per-task .npz files into a combined AngleSweepResult.",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output .npz path for --task or --collect (or full run).",
        )
        parser.add_argument(
            "--processes",
            type=int,
            default=1,
            help="Number of worker processes when running all tasks locally.",
        )
        args = parser.parse_args(argv)

        if args.list_tasks:
            tasks = self.list_tasks()
            print(json.dumps(tasks))
            return tasks

        if args.task is not None:
            return self.run_task(args.task, output_path=args.output)

        if args.collect is not None:
            loaded = []
            for p in args.collect:
                with np.load(p, allow_pickle=False) as d:
                    loaded.append({k: d[k] for k in d.files})
            result = self.assemble(loaded)
            if args.output is not None:
                result.save(args.output)
            return result

        result = self.run(processes=args.processes)
        if args.output is not None:
            result.save(args.output)
        return result


def oblique_sweep(
    cell_builder: FloquetCell | Callable,
    frequencies: Sequence[float] | np.ndarray,
    angles: float | Sequence[float] | np.ndarray,
    polarization: str = "TE",
    phi: float = 0.0,
    n_k: int | None = None,
    dk_max: float | None = None,
    dtheta_max: float | None = None,
    processes: int = 1,
    **kwargs,
) -> AngleSweepResult:
    """Convenience function to configure and run an :class:`AngleSweep`."""
    sweep = AngleSweep(
        cell_builder=cell_builder,
        frequencies=frequencies,
        angles=angles,
        polarization=polarization,
        phi=phi,
        n_k=n_k,
        dk_max=dk_max,
        dtheta_max=dtheta_max,
        **kwargs,
    )
    return sweep.run(processes=processes)



def fresnel_rt(
    theta_inc: float | np.ndarray,
    n1: float,
    n2: float,
    pol: str = "TE",
    mu1: float = 1.0,
    mu2: float = 1.0,
) -> dict[str, np.ndarray]:
    """Analytical Fresnel reflection and transmission for a planar interface.

    Parameters
    ----------
    theta_inc : float or array_like
        Incidence angle(s) in medium 1 [rad].
    n1, n2 : float
        Refractive indices of media 1 and 2.
    pol : {"TE", "TM", "s", "p"}
        Polarization state.
    mu1, mu2 : float, optional
        Relative permeabilities of media 1 and 2 (default 1.0).

    Returns
    -------
    dict of str to ndarray
        ``{"r": r, "t": t, "R": R, "T": T, "tir": tir}`` where ``T`` is ``nan``
        beyond the total-internal-reflection critical angle.
    """
    th1 = np.asarray(theta_inc, dtype=float)
    pol_norm = _normalize_pol(pol)
    sin_th2 = (n1 / n2) * np.sin(th1)
    tir = np.abs(sin_th2) > 1.0
    cos_th1 = np.cos(th1) + 0j
    cos_th2 = np.where(
        tir,
        -1j * np.sqrt(np.maximum(0.0, sin_th2**2 - 1.0)),
        np.sqrt(np.maximum(0.0, 1.0 - sin_th2**2)),
    )

    if pol_norm == "TE":
        y1 = (n1 / mu1) * cos_th1
        y2 = (n2 / mu2) * cos_th2
        r = (y1 - y2) / (y1 + y2)
        t = 2.0 * y1 / (y1 + y2)
    else:
        z1 = (mu1 / n1) * cos_th1
        z2 = (mu2 / n2) * cos_th2
        r = (z2 - z1) / (z2 + z1)
        t = (n1 / n2) * 2.0 * z2 / (z2 + z1)

    R = np.abs(r) ** 2
    T = np.where(tir, np.nan, np.real(cos_th2 / cos_th1) * (n2 * mu1) / (n1 * mu2) * np.abs(t) ** 2)
    return {"r": r, "t": t, "R": R, "T": T, "tir": tir}


def slab_rt(
    frequencies: float | np.ndarray,
    theta_inc: float | np.ndarray,
    thickness: float,
    eps_slab: float,
    pol: str = "TE",
    eps_below: float = 1.0,
    eps_above: float = 1.0,
    mu_slab: float = 1.0,
    mu_below: float = 1.0,
    mu_above: float = 1.0,
) -> dict[str, np.ndarray]:
    """Analytical transfer-matrix reflection and transmission of a dielectric slab.

    Uses the ``exp(+i*omega*t)`` convention (spatial propagation ``exp(-i*k_z*z)``)
    matching Wakis's DFT and :class:`FloquetPort` conventions.

    Parameters
    ----------
    frequencies : float or array_like
        Frequencies [Hz].
    theta_inc : float or array_like
        Incidence angle(s) in the lower medium [rad].
    thickness : float
        Slab thickness [m].
    eps_slab : float
        Slab relative permittivity.
    pol : {"TE", "TM", "s", "p"}
        Polarization state.
    eps_below, eps_above : float, optional
        Relative permittivities of the lower and upper half-spaces (default 1.0).
    mu_slab, mu_below, mu_above : float, optional
        Relative permeabilities (default 1.0).

    Returns
    -------
    dict of str to ndarray
        ``{"r": r, "t": t, "R": R, "T": T}`` with shape broadcast over
        ``(theta_inc, frequencies)``.
    """
    f = np.asarray(frequencies, dtype=float)
    th = np.asarray(theta_inc, dtype=float)
    if th.ndim > 0 and f.ndim > 0:
        th_b = th[:, None]
        f_b = f[None, :]
    else:
        th_b = th
        f_b = f

    n1 = float(np.sqrt(eps_below * mu_below))
    n2 = float(np.sqrt(eps_slab * mu_slab))
    n3 = float(np.sqrt(eps_above * mu_above))

    k0 = 2.0 * np.pi * f_b / c
    kx = k0 * n1 * np.sin(th_b)

    def _kz(n_val):
        sq = (k0 * n_val) ** 2 - kx**2
        return np.where(sq >= 0.0, np.sqrt(np.maximum(0.0, sq)), -1j * np.sqrt(np.maximum(0.0, -sq)))

    kz1 = _kz(n1)
    kz2 = _kz(n2)
    kz3 = _kz(n3)

    pol_norm = _normalize_pol(pol)
    if pol_norm == "TE":
        y1 = kz1 / mu_below
        y2 = kz2 / mu_slab
        y3 = kz3 / mu_above
    else:
        y1 = eps_below / kz1
        y2 = eps_slab / kz2
        y3 = eps_above / kz3

    r12 = (y1 - y2) / (y1 + y2)
    r23 = (y2 - y3) / (y2 + y3)
    t12 = 2.0 * y1 / (y1 + y2)
    t23 = 2.0 * y2 / (y2 + y3)

    phase = np.exp(-1j * kz2 * float(thickness))
    denom = 1.0 + r12 * r23 * (phase**2)
    r = (r12 + r23 * (phase**2)) / denom
    t_field = (t12 * t23 * phase) / denom

    # Unit-power Floquet S-parameter transmission amplitude
    prop3 = np.real(kz3) > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        t_unit = np.where(prop3, t_field * np.sqrt(np.real(y3) / np.real(y1)), np.nan + 0j)
    R = np.abs(r) ** 2
    T = np.where(prop3, np.abs(t_unit) ** 2, np.nan)
    return {"r": r, "t": t_unit, "R": R, "T": T}


def grating_angles(
    frequencies: float | np.ndarray,
    theta_inc: float | np.ndarray,
    period: float,
    orders: Sequence[int],
    n_inc: float = 1.0,
    n_out: float = 1.0,
) -> np.ndarray:
    """Propagating diffraction angles [rad] from the 1D grating equation.

    Computes ``theta_m = arcsin((n_inc * sin(theta_inc) + m * c / (f * period)) / n_out)``
    and returns ``nan`` for evanescent orders where ``|sin(theta_m)| > 1``.
    """
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    th = np.atleast_1d(np.asarray(theta_inc, dtype=float))
    m_arr = np.asarray(orders, dtype=float)
    sin_m = (
        n_inc * np.sin(th)[:, None, None]
        + m_arr[None, None, :] * (c / (f[None, :, None] * float(period)))
    ) / float(n_out)
    out = np.where(np.abs(sin_m) <= 1.0, np.arcsin(np.clip(sin_m, -1.0, 1.0)), np.nan)
    if np.ndim(theta_inc) == 0 and np.ndim(frequencies) == 0:
        return out[0, 0]
    if np.ndim(theta_inc) == 0:
        return out[0]
    if np.ndim(frequencies) == 0:
        return out[:, 0]
    return out


def rayleigh_sinusoidal_grating(
    frequencies: float | np.ndarray,
    theta_inc: float,
    period: float,
    amplitude: float,
    eps1: float = 1.0,
    eps2: float = 2.25,
    pol: str = "TE",
) -> dict[str, np.ndarray]:
    """First-order Rayleigh perturbation theory for a shallow sinusoidal grating.

    For a surface profile ``z(x) = amplitude * cos(2*pi*x / period)`` with
    ``k0 * amplitude << 1`` separating media ``eps1`` (below) and ``eps2`` (above),
    computes the first-order ``m = +/-1`` reflected and transmitted diffraction
    efficiencies ``R_{+1}, R_{-1}, T_{+1}, T_{-1}`` alongside the unperturbed
    zeroth-order Fresnel efficiencies ``R_0, T_0``.

    Parameters
    ----------
    frequencies : float or array_like
        Frequencies [Hz].
    theta_inc : float
        Incidence angle in medium 1 [rad].
    period : float
        Grating period ``Lx`` [m].
    amplitude : float
        Sinusoidal surface-relief half-amplitude ``a`` [m] (peak-to-valley ``2*a``).
    eps1, eps2 : float, optional
        Relative permittivities of media 1 and 2.
    pol : {"TE", "TM", "s", "p"}, optional
        Polarization state (default ``"TE"``).

    Returns
    -------
    dict of str to ndarray
        Dictionary with keys ``"R_0"``, ``"T_0"``, ``"R_p1"``, ``"R_m1"``,
        ``"T_p1"``, ``"T_m1"``.
    """
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    pol_norm = _normalize_pol(pol)
    k0 = 2.0 * np.pi * f / c
    n1 = float(np.sqrt(eps1))
    n2 = float(np.sqrt(eps2))
    kx0 = k0 * n1 * np.sin(theta_inc)
    G = 2.0 * np.pi / float(period)

    def _kz(eps_val, kx_val):
        sq = k0**2 * eps_val - kx_val**2
        return np.where(sq >= 0.0, np.sqrt(np.maximum(0.0, sq)), -1j * np.sqrt(np.maximum(0.0, -sq)))

    kz1_0 = _kz(eps1, kx0)
    kz2_0 = _kz(eps2, kx0)

    if pol_norm == "TE":
        r0 = (kz1_0 - kz2_0) / (kz1_0 + kz2_0)
        t0 = 2.0 * kz1_0 / (kz1_0 + kz2_0)
        R0 = np.abs(r0) ** 2
        T0 = np.real(kz2_0 / kz1_0) * np.abs(t0) ** 2
    else:
        q1_0 = kz1_0 / eps1
        q2_0 = kz2_0 / eps2
        r0 = (q1_0 - q2_0) / (q1_0 + q2_0)
        t0 = 2.0 * q1_0 / (q1_0 + q2_0)
        R0 = np.abs(r0) ** 2
        T0 = np.real(q2_0 / q1_0) * np.abs(t0) ** 2

    out: dict[str, np.ndarray] = {"R_0": R0, "T_0": T0}
    a = float(amplitude)

    for label, m in (("p1", +1), ("m1", -1)):
        kxm = kx0 + m * G
        kz1_m = _kz(eps1, kxm)
        kz2_m = _kz(eps2, kxm)
        prop1 = np.real(kz1_m) > 0.0
        prop2 = np.real(kz2_m) > 0.0

        if pol_norm == "TE":
            pref = -0.5j * a * (k0**2) * (eps2 - eps1) * t0 / (kz1_m + kz2_m)
            rm = pref
            tm = pref
            Rm = np.where(prop1, np.real(kz1_m / kz1_0) * np.abs(rm) ** 2, np.nan)
            Tm = np.where(prop2, np.real(kz2_m / kz1_0) * np.abs(tm) ** 2, np.nan)
        else:
            q1_m = kz1_m / eps1
            q2_m = kz2_m / eps2
            # First-order TM Rayleigh amplitudes (Toigo et al., Phys. Rev. B 15, 5618 (1977))
            delta_inv_eps = (1.0 / eps1) - (1.0 / eps2)
            num_r = 0.5j * a * t0 * (
                delta_inv_eps * kz2_0 * kz2_m - (eps2 - eps1) * (kx0 * kxm) / (eps1 * eps2)
            )
            num_t = 0.5j * a * t0 * (
                delta_inv_eps * kz2_0 * (-kz1_m) - (eps2 - eps1) * (kx0 * kxm) / (eps1 * eps2)
            )
            denom = q1_m + q2_m
            rm = num_r / denom
            tm = num_t / denom
            Rm = np.where(prop1, np.real(q1_m / (kz1_0 / eps1)) * np.abs(rm) ** 2, np.nan)
            Tm = np.where(prop2, np.real(q2_m / (kz1_0 / eps1)) * np.abs(tm) ** 2, np.nan)

        out[f"R_{label}"] = Rm
        out[f"T_{label}"] = Tm

    return out


def rcwa_1d_grating(
    frequencies: float | np.ndarray,
    theta_inc: float,
    period: float,
    thickness: float,
    eps_grating: np.ndarray,
    eps_below: float = 1.0,
    eps_above: float = 1.0,
    pol: str = "TE",
    n_harmonics: int = 15,
    orders: Sequence[int] = (-1, 0, 1),
) -> dict[str, np.ndarray]:
    """1D Fourier Modal Method (RCWA) reference for a lamellar/periodic grating layer.

    Implements the standard 1D Fourier Modal Method (Moharam et al., *J. Opt. Soc.
    Am. A* **12**, 1068 (1995); Lalanne & Morris, *J. Opt. Soc. Am. A* **13**, 779
    (1996)) using NumPy/SciPy eigendecomposition for cross-checking Wakis oblique
    grating simulations.

    Parameters
    ----------
    frequencies : float or array_like
        Frequencies [Hz].
    theta_inc : float
        Incidence angle in the lower half-space [rad].
    period : float
        Grating period ``Lx`` [m].
    thickness : float
        Grating layer thickness [m].
    eps_grating : array_like
        1D array sampling the periodic relative permittivity ``eps(x)`` uniformly
        across one period ``[0, period)``.
    eps_below, eps_above : float, optional
        Relative permittivities of the lower and upper half-spaces (default 1.0).
    pol : {"TE", "TM", "s", "p"}, optional
        Polarization state (default ``"TE"``).
    n_harmonics : int, optional
        Truncation order ``M`` (total ``2*M + 1`` harmonics, default 15).
    orders : sequence of int, optional
        Diffraction orders ``m`` to return in ``R_orders`` and ``T_orders``.

    Returns
    -------
    dict of str to ndarray
        ``{"R_orders": R_ord, "T_orders": T_ord, "R_total": R_tot, "T_total": T_tot}``
        where ``R_orders`` and ``T_orders`` have shape ``(n_freq, len(orders))``.
    """
    f_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
    pol_norm = _normalize_pol(pol)
    eps_x = np.asarray(eps_grating, dtype=complex)
    inv_eps_x = 1.0 / eps_x

    M = int(n_harmonics)
    N = 2 * M + 1
    m_vec = np.arange(-M, M + 1, dtype=float)

    # Toeplitz Fourier matrices of eps(x) and 1/eps(x)
    n_x = eps_x.size
    fft_eps = np.fft.fft(eps_x) / n_x
    fft_inv = np.fft.fft(inv_eps_x) / n_x
    idx_diff = np.arange(N)[:, None] - np.arange(N)[None, :]
    E_mat = fft_eps[idx_diff % n_x]
    invE_mat = fft_inv[idx_diff % n_x]
    E_inv_rule = np.linalg.inv(invE_mat)

    n1 = float(np.sqrt(eps_below))
    G = 2.0 * np.pi / float(period)
    ord_list = [int(o) for o in orders]

    R_ord = np.full((f_arr.size, len(ord_list)), np.nan, dtype=float)
    T_ord = np.full((f_arr.size, len(ord_list)), np.nan, dtype=float)
    R_tot = np.zeros(f_arr.size, dtype=float)
    T_tot = np.zeros(f_arr.size, dtype=float)

    for jf, freq in enumerate(f_arr):
        k0 = 2.0 * np.pi * float(freq) / c
        kx = (k0 * n1 * np.sin(theta_inc) + m_vec * G) / k0
        Kx = np.diag(kx.astype(complex))

        def _kz_norm(eps_h):
            sq = complex(eps_h) - kx**2
            return np.where(
                np.real(sq) >= 0.0,
                np.sqrt(sq + 0j),
                -1j * np.sqrt(-sq + 0j),
            )

        kz1 = _kz_norm(eps_below)
        kz2 = _kz_norm(eps_above)

        if pol_norm == "TE":
            A_eig = Kx @ Kx - E_mat
            eigvals, W = np.linalg.eig(A_eig)
            q = np.sqrt(eigvals + 0j)
            q = np.where(np.real(q) + np.imag(q) < 0.0, -q, q)
            V = W @ np.diag(q)
            Y1 = np.diag(1j * kz1)
            Y2 = np.diag(1j * kz2)
        else:
            B_mat = Kx @ np.linalg.solve(E_mat, Kx) - np.eye(N, dtype=complex)
            A_eig = E_inv_rule @ B_mat
            eigvals, W = np.linalg.eig(A_eig)
            q = np.sqrt(eigvals + 0j)
            q = np.where(np.real(q) + np.imag(q) < 0.0, -q, q)
            V = np.linalg.solve(E_inv_rule, W) @ np.diag(q)
            Y1 = np.diag(1j * kz1 / eps_below)
            Y2 = np.diag(1j * kz2 / eps_above)

        X = np.diag(np.exp(-k0 * q * float(thickness)))

        # Solve boundary conditions at z=0 and z=thickness
        S_top = np.linalg.solve(V + Y2 @ W, V - Y2 @ W)
        M1 = W @ (np.eye(N) + X @ S_top @ X)
        M2 = V @ (np.eye(N) - X @ S_top @ X)

        delta0 = np.zeros(N, dtype=complex)
        delta0[M] = 1.0
        rhs = 2.0 * Y1 @ delta0
        f_coeff = np.linalg.solve(Y1 @ M1 + M2, rhs)
        r_vec = M1 @ f_coeff - delta0
        t_vec = W @ (X + S_top @ X) @ f_coeff

        prop1 = np.real(kz1) > 1e-12
        prop2 = np.real(kz2) > 1e-12
        kz1_0 = np.real(kz1[M])

        if pol_norm == "TE":
            r_eff = np.where(prop1, (np.real(kz1) / kz1_0) * (np.abs(r_vec) ** 2), np.nan)
            t_eff = np.where(prop2, (np.real(kz2) / kz1_0) * (np.abs(t_vec) ** 2), np.nan)
        else:
            r_eff = np.where(prop1, (np.real(kz1 / eps_below) / (kz1_0 / eps_below)) * (np.abs(r_vec) ** 2), np.nan)
            t_eff = np.where(prop2, (np.real(kz2 / eps_above) / (kz1_0 / eps_below)) * (np.abs(t_vec) ** 2), np.nan)

        R_tot[jf] = float(np.nansum(r_eff))
        T_tot[jf] = float(np.nansum(t_eff))
        for io, m_ord in enumerate(ord_list):
            if -M <= m_ord <= M:
                R_ord[jf, io] = r_eff[M + m_ord]
                T_ord[jf, io] = t_eff[M + m_ord]

    return {
        "R_orders": R_ord,
        "T_orders": T_ord,
        "R_total": R_tot,
        "T_total": T_tot,
    }


__all__ = [
    "AngleSweep",
    "AngleSweepResult",
    "FloquetCell",
    "FloquetMode",
    "FloquetPort",
    "floquet_s_matrix",
    "fresnel_rt",
    "grating_angles",
    "k_parallel_grid",
    "oblique_sweep",
    "rayleigh_sinusoidal_grating",
    "rcwa_1d_grating",
    "reciprocal_s_matrix",
    "resample_constant_angle",
    "s_parameters_from_ports",
    "sampling_density",
    "slab_rt",
]
