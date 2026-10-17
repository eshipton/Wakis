# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Near-to-far-field transformation and antenna metrics.

`Near2FarMonitor` records the tangential E and H spectra on a closed box (or
an open plane) around the radiator and radiates the equivalent surface currents
J = n x H and M = -n x E with the free-space Green's function. The far field is
a linear function of the recorded spectra, so every quantity below can be
re-expressed in an automatic-differentiation framework for inverse design.

The helper functions (`ludwig3`, `axial_ratio`, `half_power_beamwidth`,
`plot_pattern_cut`, `plot_pattern_3d`) act on plain arrays of far-field
components.
"""

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import epsilon_0, mu_0

from wakis.field_monitors import FluxMonitor

Z0 = np.sqrt(mu_0 / epsilon_0)  # Wave impedance in free space [Ohm]


class Near2FarMonitor(FluxMonitor):
    def __init__(
        self,
        frequencies,
        center=(0.0, 0.0, 0.0),
        size=(0.0, 0.0, 0.0),
        eps_r=None,
        mu_r=None,
        ground_plane=None,
        max_chunk=2**22,
    ):
        """
        Far-field radiation from the fields on a closed box or a plane.

        By the surface equivalence principle the field outside a closed surface
        S enclosing all sources equals the field radiated in free space by the
        surface currents J = n x H and M = -n x E on S (n outward). The radiation
        vectors are (Balanis, Antenna Theory, 4th ed., Sec. 12.3)

            N(r) = sum J dA exp(+j k r'.r),   L(r) = sum M dA exp(+j k r'.r),

        and the far field is E = F(r) exp(-j k R) / R with the pattern vector

            F = -j k / (4 pi) [eta (N - (N.r) r) + L x r],

        i.e. E_theta = -j k (L_phi + eta N_theta) exp(-j k R) / (4 pi R) and
        E_phi = j k (L_theta - eta N_phi) exp(-j k R) / (4 pi R), for the
        exp(+j w t) convention implied by the monitors' exp(-j 2 pi f t) kernel.

        The tangential E and H samples, H averaging and trapezoid weights are
        those of `FluxMonitor`, so `get_flux()` of this monitor returns the
        flux through the same surface and is consistent with `radiated_power`.

        Parameters
        ----------
        frequencies : array_like
            Frequencies to monitor [Hz].
        center : tuple of float, optional
            Centre of the box or plane [m].
        size : tuple of float, optional
            Extent [m]. No zero entry defines a closed box (outward normals);
            one zero entry defines an open plane radiating along +axis, which is
            only an approximation (fields beyond the plane are neglected).
        eps_r, mu_r : float, optional
            Relative permittivity and permeability of the homogeneous medium
            around the surface. Default: the solver background.
        ground_plane : str, optional
            Box face lying on an infinite PEC plane, e.g. '-z' for a box whose
            lower z face is on a PEC boundary at the domain bottom. That face is
            dropped and the remaining currents are radiated together with their
            images (electric: tangential reversed; magnetic: normal reversed).
            The far field is then only physical in the half-space on the box
            side of the plane.
        max_chunk : int, optional
            Upper bound on frequencies x points x directions complex entries
            held in memory at once. Default 2**22 (64 MB).

        Notes
        -----
        Call `update(solver, t)` after every step, or pass the monitor to
        `solver.emsolve`. All far-field outputs share the normalization of
        `FluxMonitor.get_flux`: for a time-harmonic source of complex amplitude
        S(f) divide powers by 2 |S(f)|^2 and fields by S(f). Ratios such as the
        directivity are normalization free.
        """
        super().__init__(frequencies, center=center, size=size)
        self.eps_r, self.mu_r = eps_r, mu_r
        self.max_chunk = int(max_chunk)
        self.ground_plane = ground_plane
        self._ground = None
        if ground_plane is None:
            return
        if len(self.faces) != 6:
            raise ValueError("ground_plane requires a closed box (no zero size).")
        if len(ground_plane) != 2 or ground_plane[0] not in "+-":
            raise ValueError(f"Invalid ground_plane {ground_plane!r}, e.g. '-z'.")
        axis = "xyz".index(ground_plane[1])
        side = 1.0 if ground_plane[0] == "+" else -1.0
        self.faces = [f for f in self.faces if not (f[0] == axis and f[2] == side)]
        self._ground = (axis, self.center[axis] + side * self.size[axis] / 2, -side)

    def _initialize(self, solver):
        super()._initialize(solver)
        if self.eps_r is None:
            self.eps_r = float(solver.eps_bg / epsilon_0)
        if self.mu_r is None:
            self.mu_r = float(solver.mu_bg / mu_0)
        if self._ground is not None:
            # Snap the image plane to the grid node plane carrying the PEC
            axis, position, side = self._ground
            nodes = (solver.grid.x, solver.grid.y, solver.grid.z)[axis]
            self._ground = (axis, nodes[np.argmin(np.abs(nodes - position))], side)

    @property
    def wavenumber(self):
        """Wavenumber k = 2 pi f sqrt(eps_r mu_r) / c in the medium [1/m]."""
        return 2 * np.pi * self.frequencies * np.sqrt(self.eps_r * self.mu_r) / c_light

    @property
    def impedance(self):
        """Wave impedance eta = eta0 sqrt(mu_r / eps_r) of the medium [Ohm]."""
        return Z0 * np.sqrt(self.mu_r / self.eps_r)

    def surface_currents(self):
        """
        Equivalent current elements on the surface (including images).

        Each term of the surface carries, at the same points, an electric
        current element J dA = -w H along `j_axis` and a magnetic current
        element M dA = -w E along `m_axis`, where w is the signed `FluxMonitor`
        area weight. This is the linear map from the recorded DFT samples to the
        sources of the far field.

        Returns
        -------
        list of dict
            Keys 'points' (n, 3) [m], 'j_axis', 'm_axis' (int), 'J', 'M'
            (n_freq, n) complex current elements [A m s] and [V m s].
        """
        if self.is_first_update:
            raise RuntimeError("Near2FarMonitor has not recorded any fields yet.")
        currents = []
        for term in self._terms:
            weight = np.asarray(term["weight"])
            E = term["E_acc"].get() if hasattr(term["E_acc"], "get") else term["E_acc"]
            H = term["H_acc"].get() if hasattr(term["H_acc"], "get") else term["H_acc"]
            X, Y, Z = np.meshgrid(*term["coords"], indexing="ij")
            elem = {
                "points": np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1),
                "j_axis": term["axes"][0],
                "m_axis": term["axes"][1],
                "J": -weight * H,
                "M": -weight * E,
            }
            currents.append(elem)
            if self._ground is None:
                continue
            axis, position, _ = self._ground
            image = dict(elem)
            image["points"] = elem["points"].copy()
            image["points"][:, axis] = 2 * position - elem["points"][:, axis]
            # Image in a PEC plane: tangential J and normal M are reversed
            image["J"] = elem["J"] * (1.0 if elem["j_axis"] == axis else -1.0)
            image["M"] = elem["M"] * (-1.0 if elem["m_axis"] == axis else 1.0)
            currents.append(image)
        return currents

    def radiation_vectors(self, directions, origin=None):
        """
        Radiation vectors N and L in Cartesian components.

        Parameters
        ----------
        directions : array_like, shape (n_dir, 3)
            Unit vectors of the observation directions.
        origin : tuple of float, optional
            Phase reference point [m]. Default is the monitor centre.

        Returns
        -------
        N, L : ndarray of complex, shape (n_freq, n_dir, 3)
            Electric [A m s] and magnetic [V m s] radiation vectors.
        """
        rhat = np.asarray(directions, dtype=float).reshape(-1, 3)
        origin = self.center if origin is None else np.asarray(origin, dtype=float)
        k = self.wavenumber
        n_f, n_dir = k.size, rhat.shape[0]
        N = np.zeros((n_f, n_dir, 3), dtype=complex)
        L = np.zeros((n_f, n_dir, 3), dtype=complex)
        chunk = max(1, self.max_chunk // (n_f * n_dir))
        for elem in self.surface_currents():
            points = elem["points"] - origin
            for s in range(0, points.shape[0], chunk):
                sl = slice(s, s + chunk)
                # exp(+j k r'.r): (n_freq, n_points, n_dir)
                phase = np.exp(1j * k[:, None, None] * (points[sl] @ rhat.T)[None])
                N[:, :, elem["j_axis"]] += np.einsum(
                    "fp,fpd->fd", elem["J"][:, sl], phase
                )
                L[:, :, elem["m_axis"]] += np.einsum(
                    "fp,fpd->fd", elem["M"][:, sl], phase
                )
        if self.comm is not None:
            N, L = self.comm.allreduce(N), self.comm.allreduce(L)
        return N, L

    def _pattern_vector(self, rhat, origin=None):
        """Far-field pattern vector F = R exp(jkR) E, shape (n_freq, n_dir, 3)."""
        N, L = self.radiation_vectors(rhat, origin)
        N_perp = N - np.einsum("fdi,di->fd", N, rhat)[..., None] * rhat
        L_cross_r = np.cross(L, rhat[None])
        k = self.wavenumber[:, None, None]
        return -1j * k / (4 * np.pi) * (self.impedance * N_perp + L_cross_r)

    def get_farfield(self, theta, phi, r=None, origin=None):
        """
        Far-field components in spherical coordinates.

        Parameters
        ----------
        theta, phi : array_like
            Polar and azimuthal angles [rad], broadcast against each other.
        r : float, optional
            Observation radius [m]. If given, the fields include the factor
            exp(-j k r) / r; otherwise the r-independent pattern r exp(j k r) E
            [V s] is returned.
        origin : tuple of float, optional
            Phase reference point [m]. Default is the monitor centre.

        Returns
        -------
        dict
            'Etheta', 'Ephi' [V/m s] and 'Htheta', 'Hphi' [A/m s], each of shape
            (n_freq, *broadcast(theta, phi).shape). H = r x E / eta.
        """
        theta, phi = np.broadcast_arrays(
            np.asarray(theta, float), np.asarray(phi, float)
        )
        shape = theta.shape
        theta, phi = theta.ravel(), phi.ravel()
        st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
        rhat = np.stack([st * cp, st * sp, ct], axis=1)
        theta_hat = np.stack([ct * cp, ct * sp, -st], axis=1)
        phi_hat = np.stack([-sp, cp, np.zeros_like(sp)], axis=1)

        F = self._pattern_vector(rhat, origin)
        if r is not None:
            F = F * (np.exp(-1j * self.wavenumber * r) / r)[:, None, None]
        Etheta = np.einsum("fdi,di->fd", F, theta_hat).reshape(-1, *shape)
        Ephi = np.einsum("fdi,di->fd", F, phi_hat).reshape(-1, *shape)
        return {
            "Etheta": Etheta,
            "Ephi": Ephi,
            "Htheta": -Ephi / self.impedance,
            "Hphi": Etheta / self.impedance,
        }

    def radiation_intensity(self, theta, phi):
        """
        Radiation intensity U = R^2 (|E_theta|^2 + |E_phi|^2) / eta.

        Parameters
        ----------
        theta, phi : array_like
            Polar and azimuthal angles [rad], broadcast against each other.

        Returns
        -------
        ndarray
            U of shape (n_freq, *broadcast shape) [W s^2 / sr], normalized like
            `FluxMonitor.get_flux` (divide by 2 |S(f)|^2 for time-harmonic power).
        """
        ff = self.get_farfield(theta, phi)
        return (np.abs(ff["Etheta"]) ** 2 + np.abs(ff["Ephi"]) ** 2) / self.impedance

    def radiated_power(self, n_theta=None, n_phi=None):
        """
        Total radiated power by integrating U over the far-field sphere.

        Gauss-Legendre quadrature in cos(theta) and the uniform rule in phi are
        exact for the band-limited patterns of a source of finite size; the
        default orders resolve the electrical size k R_max of the surface. With
        a ground plane only the physical half-space is integrated.

        Parameters
        ----------
        n_theta, n_phi : int, optional
            Quadrature orders. Default max(24, k R_max + 16) and twice that.

        Returns
        -------
        ndarray
            P(f) [W s^2], normalized like `FluxMonitor.get_flux`.
        """
        rhat, weights = self._sphere_quadrature(n_theta, n_phi)
        F = self._pattern_vector(rhat)
        U = np.sum(np.abs(F) ** 2, axis=-1) / self.impedance
        return U @ weights

    def _sphere_quadrature(self, n_theta=None, n_phi=None):
        if n_theta is None:
            r_max = max(
                np.max(np.linalg.norm(e["points"] - self.center, axis=1))
                for e in self.surface_currents()
            )
            n_theta = int(max(24, np.max(self.wavenumber) * r_max + 16))
        n_phi = 2 * n_theta if n_phi is None else n_phi
        x, w = np.polynomial.legendre.leggauss(n_theta)
        axis, side = 2, None
        if self._ground is not None:
            axis, _, side = self._ground
            # Map the nodes to cos(theta') in [0, 1] on the physical side
            x, w = side * (x + 1) / 2, w / 2
        phi = 2 * np.pi * np.arange(n_phi) / n_phi
        ct, ph = np.meshgrid(x, phi, indexing="ij")
        st = np.sqrt(1 - ct**2)
        local = [st * np.cos(ph), st * np.sin(ph), ct]
        # Polar axis of the local frame along `axis`
        rhat = np.zeros((ct.size, 3))
        rhat[:, axis] = local[2].ravel()
        rhat[:, (axis + 1) % 3] = local[0].ravel()
        rhat[:, (axis + 2) % 3] = local[1].ravel()
        weights = np.outer(w, np.full(n_phi, 2 * np.pi / n_phi)).ravel()
        return rhat, weights

    def directivity(self, theta, phi, n_theta=None, n_phi=None):
        """
        Directivity D = 4 pi U / P_rad.

        Parameters
        ----------
        theta, phi : array_like
            Polar and azimuthal angles [rad], broadcast against each other.
        n_theta, n_phi : int, optional
            Quadrature orders for P_rad, see `radiated_power`.

        Returns
        -------
        ndarray
            Linear directivity, shape (n_freq, *broadcast shape). Use
            10 log10(D) for dBi.
        """
        U = self.radiation_intensity(theta, phi)
        P = self.radiated_power(n_theta, n_phi)
        return 4 * np.pi * U / P.reshape(-1, *([1] * (U.ndim - 1)))

    def max_directivity(self, n_theta=None, n_phi=None):
        """
        Maximum directivity over the quadrature directions.

        Returns
        -------
        D_max : ndarray
            Maximum linear directivity per frequency.
        direction : ndarray, shape (n_freq, 3)
            Unit vector of the maximum. Refine with `directivity` if the
            quadrature grid is coarse compared with the beamwidth.
        """
        rhat, weights = self._sphere_quadrature(n_theta, n_phi)
        F = self._pattern_vector(rhat)
        U = np.sum(np.abs(F) ** 2, axis=-1) / self.impedance
        D = 4 * np.pi * U / (U @ weights)[:, None]
        i = np.argmax(D, axis=1)
        return D[np.arange(D.shape[0]), i], rhat[i]

    def gain(self, theta, phi, input_power):
        """
        Gain G = 4 pi U / P_in = e_rad D.

        Parameters
        ----------
        theta, phi : array_like
            Polar and azimuthal angles [rad], broadcast against each other.
        input_power : array_like
            Power accepted by the antenna per frequency, normalized like
            `FluxMonitor.get_flux`, e.g. the work done by a point source
            -Re(E(f) Il(f)^*) or the flux through a box tightly around the feed.

        Returns
        -------
        ndarray
            Linear gain, shape (n_freq, *broadcast shape).
        """
        U = self.radiation_intensity(theta, phi)
        P_in = np.asarray(input_power, dtype=float)
        return 4 * np.pi * U / P_in.reshape(-1, *([1] * (U.ndim - 1)))

    def radiation_efficiency(self, input_power, n_theta=None, n_phi=None):
        """
        Radiation efficiency e_rad = P_rad / P_in.

        Parameters
        ----------
        input_power : array_like
            Accepted power per frequency, see `gain`.

        Returns
        -------
        ndarray
            e_rad per frequency.
        """
        return self.radiated_power(n_theta, n_phi) / np.asarray(input_power, float)


def ludwig3(Etheta, Ephi, phi, reference="x"):
    """
    Co- and cross-polar components (Ludwig's third definition).

    Parameters
    ----------
    Etheta, Ephi : array_like
        Far-field components.
    phi : array_like
        Azimuthal angle [rad], broadcast against the components.
    reference : {'x', 'y'}, optional
        Reference polarization direction at boresight (theta = 0).

    Returns
    -------
    co, cross : ndarray
        Co- and cross-polar field components.

    References
    ----------
    A. C. Ludwig, "The definition of cross polarization", IEEE Trans.
    Antennas Propag. 21(1), 116-119 (1973).
    """
    cp, sp = np.cos(phi), np.sin(phi)
    u_x = Etheta * cp - Ephi * sp
    u_y = Etheta * sp + Ephi * cp
    if reference == "x":
        return u_x, u_y
    if reference == "y":
        return u_y, u_x
    raise ValueError(f"reference must be 'x' or 'y', got {reference!r}.")


def axial_ratio(Etheta, Ephi):
    """
    Polarization axial ratio and circular components.

    With the exp(+j w t) convention, E = E_R (theta - j phi)/sqrt(2) +
    E_L (theta + j phi)/sqrt(2), so E_R = (E_theta + j E_phi)/sqrt(2) is the
    right-hand (IEEE) circular component for an outgoing wave.

    Parameters
    ----------
    Etheta, Ephi : array_like
        Far-field components.

    Returns
    -------
    ar : ndarray
        Axial ratio (|E_R| + |E_L|) / ||E_R| - |E_L|| >= 1 (inf for linear
        polarization). Use 20 log10(ar) for dB.
    E_R, E_L : ndarray
        Right- and left-hand circular components.
    """
    E_R = (Etheta + 1j * Ephi) / np.sqrt(2)
    E_L = (Etheta - 1j * Ephi) / np.sqrt(2)
    a_R, a_L = np.abs(E_R), np.abs(E_L)
    with np.errstate(divide="ignore"):
        ar = (a_R + a_L) / np.abs(a_R - a_L)
    return ar, E_R, E_L


def half_power_beamwidth(angles, pattern):
    """
    Half-power (-3 dB) beamwidth of the main lobe of a pattern cut.

    Parameters
    ----------
    angles : array_like
        Monotonic angles of the cut [rad].
    pattern : array_like
        Power pattern (e.g. U or D, linear) along the cut.

    Returns
    -------
    float
        Angular width between the half-power points around the maximum
        [rad], found by linear interpolation. NaN if the pattern does not drop
        below half power on both sides within the cut.
    """
    angles = np.asarray(angles, dtype=float)
    p = np.asarray(pattern, dtype=float)
    i0 = int(np.argmax(p))
    half = 0.5 * p[i0]
    below = np.flatnonzero(p < half)
    left, right = below[below < i0], below[below > i0]
    if left.size == 0 or right.size == 0:
        return np.nan
    i, j = left[-1], right[0]
    a_left = np.interp(half, [p[i], p[i + 1]], [angles[i], angles[i + 1]])
    a_right = np.interp(half, [p[j], p[j - 1]], [angles[j], angles[j - 1]])
    return a_right - a_left


def plot_pattern_cut(angles, pattern, db=True, dynamic_range=40.0, ax=None, **kwargs):
    """
    Polar plot of a far-field pattern cut.

    Parameters
    ----------
    angles : array_like
        Angles of the cut [rad].
    pattern : array_like
        Power pattern (linear), e.g. directivity.
    db : bool, optional
        Plot 10 log10(pattern), clipped to `dynamic_range` below the maximum.
    dynamic_range : float, optional
        Plotted range in dB. Default 40.
    ax : matplotlib.axes.Axes, optional
        Polar axes to draw on. Created if None.
    **kwargs
        Passed to `ax.plot` (e.g. label).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    else:
        fig = ax.figure
    values = np.asarray(pattern, dtype=float)
    if db:
        values = 10 * np.log10(np.maximum(values, 1e-300))
        floor = values.max() - dynamic_range
        values = np.maximum(values, floor)
        ax.set_rmin(floor)
    ax.plot(angles, values, **kwargs)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    return fig, ax


def plot_pattern_3d(theta, phi, pattern, ax=None, cmap="jet"):
    """
    3D surface plot of a power pattern, radius proportional to the pattern.

    Parameters
    ----------
    theta, phi : ndarray
        2D grids of polar and azimuthal angles [rad].
    pattern : ndarray
        Linear power pattern on the grid, e.g. directivity.
    ax : matplotlib 3D Axes, optional
        Axes to draw on. Created if None.
    cmap : str, optional
        Colormap of the surface.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.figure
    p = np.asarray(pattern, dtype=float)
    r = p / p.max()
    X = r * np.sin(theta) * np.cos(phi)
    Y = r * np.sin(theta) * np.sin(phi)
    Z = r * np.cos(theta)
    colors = plt.get_cmap(cmap)(r)
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, linewidth=0)
    ax.set(xlabel="x", ylabel="y", zlabel="z", box_aspect=(1, 1, 1))
    return fig, ax
