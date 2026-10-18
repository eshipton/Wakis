# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Diffraction-order (Floquet) decomposition on a plane of a periodic cell.

The tangential fields sampled by a `FluxMonitor` plane that spans a periodic
or Bloch-periodic unit cell are expanded in the Floquet orders of the cell.
Each propagating order is split into waves travelling along +n ("up") and -n
("down") and into TE (s) and TM (p) polarizations, using the exact relations
between E and H of plane waves of the discrete FIT scheme, so that the order
powers add up to the monitor flux to round-off (see the physics guide,
'Diffraction orders').

The decomposition `decompose_orders` is written for any NumPy-like array
module, so that `wakis.inverse.diffraction_orders` evaluates the same formulas
with JAX on the raw monitor accumulators.
"""

import numpy as np

from .field_monitors import FluxMonitor


def _host(array):
    return np.asarray(array.get() if hasattr(array, "get") else array)


class DiffractionMonitor(FluxMonitor):
    def __init__(self, frequencies, center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0)):
        """
        Amplitudes and powers of the diffraction orders through a plane of a
        periodic unit cell.

        Order (m, n) is the plane wave with tangential wavevector

            k_t = k_inc + 2 pi (m / P_b, n / P_c),

        where (b, c) are the two transverse axes in cyclic order after the
        normal a (x, y for a z-normal plane), P_b, P_c the cell periods and
        k_inc the tangential wavevector of the excitation: 0 for real
        periodic fields and -k_bloch at f > 0 (+k_bloch at f < 0) for complex
        Bloch fields, following the convention of the physics guide. Order
        labels and wavevectors refer to physical waves exp(i(k.r - w t)), as
        in RCWA codes; complex amplitudes keep the exp(+2i pi f t) phasor
        convention of the other Wakis monitors (conjugate them to compare
        phases with an exp(-i w t) code).

        Parameters
        ----------
        frequencies : array_like
            Frequencies [Hz], nonzero. Negative frequencies are meaningful for
            complex (Bloch) fields only.
        center : tuple of float, optional
            A point of the plane [m]; the E samples are taken on the nearest
            grid plane, as in `FluxMonitor`.
        size : tuple of float, optional
            Exactly one zero entry, which defines the normal. The plane always
            spans the whole cell along the two transverse axes, which must
            both be 'periodic' or 'bloch' with uniform spacing.

        Notes
        -----
        The medium on and next to the plane must be homogeneous, isotropic and
        lossless, with uniform spacing along the normal across the plane. Use
        `get_amplitudes` and `get_power` after the run; `order_angles` gives
        the propagation directions. With MPI, only planes normal to z (owned
        by one rank) are supported.
        """
        super().__init__(frequencies, center, size)
        if len(self.faces) != 1:
            raise ValueError("size must have exactly one zero entry (a plane).")
        if np.any(self.frequencies == 0):
            raise ValueError("frequencies must be nonzero.")
        self.geometry = None

    def _initialize(self, solver):
        a = self.faces[0][0]
        for d in ((a + 1) % 3, (a + 2) % 3):
            if solver.bc_low[d].lower() not in ("periodic", "bloch"):
                raise ValueError(
                    f"Axis {'xyz'[d]} must be periodic or Bloch for a "
                    f"diffraction monitor normal to {'xyz'[a]}."
                )
            # The plane spans the whole cell along the periodic axes
            self.size[d] = np.inf
        super()._initialize(solver)
        self.geometry = order_geometry(solver, self)

    def _raw(self):
        raw = {}
        for i, term in enumerate(self._terms):
            raw[f"E{i}"] = _host(term["E_acc"])
            raw[f"H{i}"] = _host(term["H_acc"])
        return raw

    @property
    def orders(self):
        """Order numbers (m, n) along the two transverse axes (b, c)."""
        return self.geometry["orders"]

    @property
    def propagating(self):
        """Boolean mask (n_freq, n_m, n_n) of the propagating orders."""
        return self.geometry["propagating"]

    def get_amplitudes(self, polarization="linear"):
        """
        Complex amplitudes of the propagating diffraction orders.

        Parameters
        ----------
        polarization : {'linear', 'circular'}, optional
            'linear' returns TE (s) and TM (p) amplitudes, 'circular' right-
            and left-handed amplitudes (see the physics guide for the
            conventions). Default 'linear'.

        Returns
        -------
        dict of ndarray
            Keys 'up_s', 'up_p', 'down_s', 'down_p' (or 'up_R', 'up_L',
            'down_R', 'down_L'), each of shape (n_freq, n_m, n_n), indexed as
            `orders`, zero for evanescent orders. Units [sqrt(W) s]: |a|^2 is
            the flux spectrum carried by the wave, in the units of
            `FluxMonitor.get_flux`.
        """
        amps = decompose_orders(self._reduce(self._raw()), self.geometry, np)
        if polarization == "linear":
            return {k: v for k, v in amps.items() if k != "net"}
        if polarization == "circular":
            return circular_amplitudes(amps, self.geometry)
        raise ValueError("polarization must be 'linear' or 'circular'.")

    def get_power(self, incident=None):
        """
        Power carried by each propagating order in each direction.

        Parameters
        ----------
        incident : array_like, optional
            Incident flux spectrum (n_freq,) [W s^2] to normalize by, e.g. the
            'up' (0, 0) power of a monitor between the source and the
            structure, or the flux of a reference run. Default: no
            normalization.

        Returns
        -------
        dict of ndarray
            'up' and 'down' (n_freq, n_m, n_n): flux spectra [W s^2] (or
            efficiencies if `incident` is given) of the waves travelling along
            +n and -n, summed over polarizations. 'net' is the net flux along
            +n of every order, computed directly from the fields; summed over
            all orders it equals `get_flux()` to round-off, and for
            propagating orders net = up - down.
        """
        amps = decompose_orders(self._reduce(self._raw()), self.geometry, np)
        power = {
            "up": np.abs(amps["up_s"]) ** 2 + np.abs(amps["up_p"]) ** 2,
            "down": np.abs(amps["down_s"]) ** 2 + np.abs(amps["down_p"]) ** 2,
            "net": amps["net"],
        }
        if incident is not None:
            scale = 1 / np.asarray(incident, dtype=float)[:, None, None]
            power = {k: v * scale for k, v in power.items()}
        return power

    def order_angles(self):
        """
        Propagation directions of the orders in the medium at the plane.

        Returns
        -------
        theta, phi : ndarray
            Polar angle from the normal (for both directions) and azimuth of
            the tangential wavevector from the b axis [rad], shape
            (n_freq, n_m, n_n), NaN for evanescent orders. Computed from the
            continuum dispersion relation, |k_t| = n k0 sin(theta).
        """
        g = self.geometry
        kt = np.hypot(g["k_t"][0], g["k_t"][1])
        s = kt / g["k_medium"][:, None, None]
        theta = np.where(g["propagating"], np.arcsin(np.minimum(s, 1.0)), np.nan)
        phi = np.where(g["propagating"], np.arctan2(g["k_t"][1], g["k_t"][0]), np.nan)
        return theta, phi

    def _reduce(self, raw):
        """With MPI, the plane lives on one rank: broadcast its samples."""
        if self.comm is None:
            return raw
        owner = self.comm.allreduce(
            self.comm.rank if raw["E0"].shape[-1] else -1, op=_mpi_max()
        )
        return self.comm.bcast(raw, root=owner)


def _mpi_max():
    from mpi4py import MPI

    return MPI.MAX


def order_geometry(solver, monitor):
    """
    Static data of the order decomposition of an initialized monitor.

    Parameters
    ----------
    solver : SolverFIT3D
        Solver the monitor was initialized on.
    monitor : DiffractionMonitor
        Monitor whose `FluxMonitor` terms are set up.

    Returns
    -------
    dict
        DFT matrices, discrete wavenumbers and admittances of the orders
        (NumPy arrays), used by `decompose_orders`.
    """
    grid = solver.grid
    a = monitor.faces[0][0]
    b, c = (a + 1) % 3, (a + 2) % 3
    nodes = (grid.x, grid.y, grid.z)
    f = monitor.frequencies
    sign = np.sign(f)

    if solver.use_mpi and a != 2:
        raise NotImplementedError("With MPI, the plane must be normal to z.")
    n_samples, spacing, period = {}, {}, {}
    for d in (b, c):
        h = np.diff(nodes[d])
        if not np.allclose(h, h[0], rtol=1e-9):
            raise ValueError(f"The mesh along {'xyz'[d]} must be uniform.")
        n_samples[d], spacing[d], period[d] = h.size, h[0], nodes[d][-1] - nodes[d][0]

    term_E, term_H = monitor._terms
    k0 = term_E["coords"][a]
    if k0.size == 0:  # MPI rank without the plane: no samples to transform
        index = None
    else:
        index = int(np.argmin(np.abs(nodes[a][:-1] - k0[0])))
    if index is not None:
        h_lo = nodes[a][index] - nodes[a][index - 1]
        h_hi = nodes[a][index + 1] - nodes[a][index]
        if not np.isclose(h_lo, h_hi, rtol=1e-6):
            raise ValueError("The mesh must be uniform along the normal at the plane.")
        dn = 0.5 * (h_lo + h_hi)
        eps, mu = _medium(solver, monitor)
    else:
        dn, eps, mu = 1.0, 1.0, 1.0
    if solver.use_mpi:
        dn, eps, mu = _from_owner(solver.grid.comm, index is not None, (dn, eps, mu))

    # Discrete angular frequency of the leapfrog scheme, signed like f
    dt = solver.dt
    Omega = 2 / dt * np.sin(np.pi * f * dt)
    k_medium = 2 * np.pi * np.abs(f) * np.sqrt(eps * mu)

    # Physical tangential wavevector of order l: k_t = -sign(f) k_bloch + G_l;
    # the DFT phasor varies as exp(i q x) with q = -sign(f) k_t
    k_bloch = np.asarray(solver.k_bloch, dtype=float)
    orders, k_t, q, K = [], [], [], []
    for d in (b, c):
        n = n_samples[d]
        ell = np.arange(-(n // 2), n - n // 2)  # one DFT period of orders
        kt = -sign[:, None] * k_bloch[d] + 2 * np.pi * ell[None, :] / period[d]
        qd = -sign[:, None] * kt
        orders.append(ell)
        k_t.append(kt)
        q.append(qd)
        K.append(2 / spacing[d] * np.sin(qd * spacing[d] / 2))

    Kb, Kc = K[0][:, :, None], K[1][:, None, :]
    Kn2 = (Omega**2 * eps * mu)[:, None, None] - Kb**2 - Kc**2
    propagating = (Kn2 > 0) & (Kn2 * dn**2 / 4 < 1)

    # Keep the smallest box of orders containing every propagating one
    keep = []
    for axis in (1, 2):
        any_prop = np.any(propagating, axis=(0, 3 - axis))
        idx = np.flatnonzero(any_prop)
        keep.append(np.arange(idx.min(), idx.max() + 1) if idx.size else idx)
    if keep[0].size == 0 or keep[1].size == 0:
        raise ValueError("No propagating order at the monitored frequencies.")
    sel = np.ix_(np.arange(f.size), keep[0], keep[1])
    propagating = propagating[sel]
    Kn2 = Kn2[sel]
    Kb, Kc = K[0][:, keep[0]], K[1][:, keep[1]]
    Kb, Kc = np.broadcast_arrays(Kb[:, :, None], Kc[:, None, :])
    k_t_b, k_t_c = np.broadcast_arrays(
        k_t[0][:, keep[0], None], k_t[1][:, None, keep[1]]
    )

    # Discrete normal wavenumber, H averaging factor cos(k_n dn / 2), and
    # TE / TM wave admittances of the discrete plane waves (placeholders on
    # evanescent orders, whose amplitudes are masked to zero)
    Kn = np.sqrt(np.where(propagating, Kn2, 1.0))
    cos_half = np.sqrt(np.where(propagating, 1 - Kn**2 * dn**2 / 4, 1.0))
    abs_Omega = np.abs(Omega)[:, None, None]
    Y_te = np.where(propagating, Kn / (abs_Omega * mu), 1.0)
    Y_tm = np.where(propagating, abs_Omega * eps / Kn, 1.0)

    # Unit vector of the physical tangential wavevector (discrete), b axis at
    # normal incidence; s = n x k_t_hat
    Kt = np.hypot(Kb, Kc)
    normal = Kt <= 1e-12 * np.max(Kt, initial=1.0)
    k_hat_b = np.where(normal, 1.0, -sign[:, None, None] * Kb / np.where(normal, 1, Kt))
    k_hat_c = np.where(normal, 0.0, -sign[:, None, None] * Kc / np.where(normal, 1, Kt))

    # DFT matrices exp(-i q x_j) / N at the sample positions of each term
    dft = []
    shapes, transposed = [], []
    for term in monitor._terms:
        mats = []
        for d, sel_d in ((b, keep[0]), (c, keep[1])):
            x = term["coords"][d]
            qd = q[0 if d == b else 1][:, sel_d]
            mats.append(np.exp(-1j * qd[:, :, None] * x[None, None, :]) / n_samples[d])
        dft.append(mats)
        shape3 = [term["coords"][d].size for d in range(3)]
        shapes.append(tuple(shape3[d] for d in range(3) if d != a))
        transposed.append(b > c)

    return {
        "axes": (a, b, c),
        "orders": (orders[0][keep[0]], orders[1][keep[1]]),
        "frequencies": f,
        "propagating": propagating,
        "k_t": (k_t_b, k_t_c),
        "k_medium": k_medium,
        "k_hat": (k_hat_b, k_hat_c),
        "cos_half": cos_half,
        "Y_te": Y_te,
        "Y_tm": Y_tm,
        "area": period[b] * period[c],
        "dft": dft,
        "shapes": shapes,
        "transposed": transposed,
        "eps": eps,
        "mu": mu,
    }


def _medium(solver, monitor):
    """Absolute permittivity and permeability on the plane, checked uniform."""
    eps, mu = [], []
    ieps = _host(solver.ieps.toarray())
    imu = _host(solver.imu.toarray())
    for term in monitor._terms:
        eps.append(1 / ieps[_host(term["E"])])
        mu.append(1 / imu[_host(term["H_above"])])
        mu.append(1 / imu[_host(term["H_below"])])
    eps, mu = np.concatenate(eps), np.concatenate(mu)
    for name, v in (("permittivity", eps), ("permeability", mu)):
        if not np.allclose(v, v[0], rtol=1e-9):
            raise ValueError(f"The {name} must be uniform on the monitor plane.")
    return float(eps[0]), float(mu[0])


def _from_owner(comm, owner, values):
    rank = comm.allreduce(comm.rank if owner else -1, op=_mpi_max())
    return comm.bcast(values, root=rank)


def decompose_orders(raw, geometry, xp=np):
    """
    Up/down, TE/TM amplitudes of the orders from the raw plane fields.

    Parameters
    ----------
    raw : dict
        'E0', 'H0' (E_b and averaged H_c) and 'E1', 'H1' (E_c and averaged
        H_b) accumulators of the plane, shape (n_freq, n_samples), as stored
        by `FluxMonitor` (`wakis.inverse` raw outputs).
    geometry : dict
        `DiffractionMonitor.geometry`.
    xp : module, optional
        Array module (numpy or jax.numpy). Default numpy.

    Returns
    -------
    dict
        'up_s', 'up_p', 'down_s', 'down_p' [sqrt(W) s] and 'net' [W s^2],
        shape (n_freq, n_m, n_n).

    Notes
    -----
    With E_k, E_s (H_k, H_s) the tangential components along k_t_hat and
    s = n x k_t_hat, the discrete plane waves satisfy H_k = -/+ Y_te E_s
    (TE) and H_s = +/- Y_tm E_k (TM) for up/down waves, and the averaged H
    is cos(k_n dn/2) H. Hence, e.g. for TE,
    up = (E_s - H_k / (cos Y_te)) / 2, down = (E_s + H_k / (cos Y_te)) / 2,
    scaled by sqrt(A cos Y_te) so that |a|^2 is the flux of the wave.
    """
    g = geometry
    n_f = g["frequencies"].size
    hat = {}
    for i in (0, 1):
        Fb, Fc = g["dft"][i]
        for field in "EH":
            X = raw[f"{field}{i}"].reshape(n_f, *g["shapes"][i])
            if g["transposed"][i]:
                X = X.transpose(0, 2, 1)
            hat[field + str(i)] = xp.einsum("fmj,fjk,fnk->fmn", Fb, X, Fc)
    E_b, H_c, E_c, H_b = hat["E0"], hat["H0"], hat["E1"], hat["H1"]
    kb, kc = g["k_hat"]
    E_k, E_s = kb * E_b + kc * E_c, -kc * E_b + kb * E_c
    H_k, H_s = kb * H_b + kc * H_c, -kc * H_b + kb * H_c

    area, cos_half, mask = g["area"], g["cos_half"], g["propagating"]
    y_te, y_tm = cos_half * g["Y_te"], cos_half * g["Y_tm"]
    g_te, g_tm = H_k / y_te, H_s / y_tm
    w_te, w_tm = mask * np.sqrt(area * y_te) / 2, mask * np.sqrt(area * y_tm) / 2
    return {
        "up_s": w_te * (E_s - g_te),
        "down_s": w_te * (E_s + g_te),
        "up_p": w_tm * (E_k + g_tm),
        "down_p": w_tm * (E_k - g_tm),
        "net": area * xp.real(E_b * xp.conj(H_c) - E_c * xp.conj(H_b)),
    }


def circular_amplitudes(amplitudes, geometry):
    """
    Right- and left-handed circular amplitudes from TE/TM amplitudes.

    L is positive helicity (E rotating counter-clockwise about the direction
    of propagation), R negative helicity, the optics convention used by the
    invrs-gym meta-atom library: a_L = (a_p + i d sigma a_s) / sqrt(2),
    a_R = (a_p - i d sigma a_s) / sqrt(2), with d = +1 (up) or -1 (down) and
    sigma = sign(f) for the exp(+2i pi f t) phasors.

    Parameters
    ----------
    amplitudes : dict
        Output of `decompose_orders` (NumPy or JAX arrays).
    geometry : dict
        `DiffractionMonitor.geometry`.

    Returns
    -------
    dict
        'up_R', 'up_L', 'down_R', 'down_L', shape (n_freq, n_m, n_n).
    """
    sigma = np.sign(geometry["frequencies"])[:, None, None]
    out = {}
    for direction, d in (("up", 1.0), ("down", -1.0)):
        a_s, a_p = amplitudes[f"{direction}_s"], amplitudes[f"{direction}_p"]
        out[f"{direction}_R"] = (a_p - 1j * d * sigma * a_s) / np.sqrt(2)
        out[f"{direction}_L"] = (a_p + 1j * d * sigma * a_s) / np.sqrt(2)
    return out
