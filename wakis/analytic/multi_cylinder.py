# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Exact 2D multiple scattering by parallel circular cylinders (T-matrix with
Graf's addition theorem), at normal incidence.

The scalar field u is E_z (``'TM'``, E along the axes) or H_z (``'TE'``, E in
the plane), in the exp(-i omega t) convention of Bohren & Huffman (as
`wakis.analytic.mie_cylinder`). Around cylinder j (centre r_j, radius a_j,
relative index m_j) the field is expanded in local polar coordinates
(rho_j, theta_j):

    exciting:   u_exc,j  = sum_n a_n^j J_n(k rho_j) e^{i n theta_j}
    scattered:  u_s,j    = sum_n b_n^j H_n^(1)(k rho_j) e^{i n theta_j}

The isolated-cylinder response is b_n^j = -c_n^j a_n^j, with c_n the Mie
coefficient (b_n for TM, a_n for TE) of `cylinder_coefficients`, consistent
with u_s = -U0 sum i^n c_n H_n e^{i n theta} for a plane wave. The wave
exciting cylinder j is the plane wave plus the waves scattered by all the
others, re-expanded about r_j with Graf's addition theorem (DLMF 10.23.7),
valid for rho_j < |r_j - r_l|:

    H_m(k rho_l) e^{i m theta_l}
        = sum_n H_{m-n}(k d_jl) e^{i (m-n) phi_jl} J_n(k rho_j) e^{i n theta_j},

where (d_jl, phi_jl) are the length and angle of r_j - r_l. The plane wave
u_inc = U0 exp(i k (x cos phi + y sin phi)) has a_n^{j,inc} =
U0 exp(i k.r_j) i^n e^{-i n phi}. The coupled system

    b^j + C^j sum_{l != j} T^{jl} b^l = -C^j a^{j,inc}

is solved directly for |n| <= N.

Observables (per unit length, widths in m, normalised by |U0|^2):

* far-field amplitude f(theta) = sum_j e^{-i k rhat.r_j} sum_n b_n^j (-i)^n
  e^{i n theta}, with u_s ~ f sqrt(2 / (pi k r)) e^{i (k r - pi/4)};
* C_sca = (2 / (pi k)) int |f|^2 dtheta, C_ext = -(4/k) Re f(phi) (optical
  theorem), C_abs = C_ext - C_sca;
* per-cylinder absorption C_abs,j = -(4/k) sum_n [Re(conj(a_n^j) b_n^j) +
  |b_n^j|^2] with a^j the total exciting field (an independent check of
  the energy balance: sum_j C_abs,j = C_ext - C_sca).

Complex m with Im m < 0 (the Wakis exp(+i w t) convention) is accepted and
the complex outputs are then conjugated, as in `wakis.analytic.mie`.

References
----------
C. F. Bohren and D. R. Huffman, Absorption and Scattering of Light by Small
Particles (Wiley, 1983), ch. 8 (single cylinder).
NIST Digital Library of Mathematical Functions, eq. 10.23.7 (Graf's
addition theorem).
"""

import numpy as np
from scipy.special import hankel1, jv

from .mie import _split_convention
from .mie_cylinder import _coefficients_bh, _n_stop

__all__ = ["CylinderCluster", "bipolar_ratio"]


def bipolar_ratio(d, a1, a2):
    """
    Geometric convergence ratio of the multipole series of two cylinders.

    For two cylinders of radii a1, a2 at centre distance d the image-charge
    (bipolar) construction converges like q^n with q = rho_1 rho_2 /
    (x_1 x_2) in terms of the foci; for equal radii q = (d/2 - x0) /
    (d/2 + x0), x0 = sqrt(d^2/4 - a^2). Used only to choose the truncation.
    """
    a = max(a1, a2)
    x0 = np.sqrt(max((d / 2) ** 2 - a**2, 1e-30))
    return (d / 2 - x0) / (d / 2 + x0)


class CylinderCluster:
    """
    Parallel circular cylinders illuminated by a plane wave (exact T-matrix).

    Parameters
    ----------
    centers : array_like (n_cyl, 2)
        Cylinder centres [m].
    radii : array_like (n_cyl,)
        Radii [m]; the cylinders must not overlap.
    m : complex or array_like (n_cyl,)
        Relative refractive index n_cyl / n_background of each cylinder.
        Im m < 0 is read as the exp(+i w t) convention (outputs conjugated).
    k : float
        Wavenumber in the (lossless) background [1/m].
    polarization : {'TM', 'TE'}
        'TM': u = E_z; 'TE': u = H_z (E in the plane).
    n_max : int, optional
        Highest multipole order |n| per cylinder. Default: the single-cylinder
        stop order plus the order needed for q^N < 1e-13 with the bipolar
        ratio q of the closest pair (capped at 60).
    direction : float
        Propagation angle phi of the incident plane wave [rad] (0: +x).

    Attributes
    ----------
    b : ndarray (n_cyl, 2 N + 1)
        Scattering coefficients b_n^j, n = -N..N (exp(-i w t)).
    a : ndarray (n_cyl, 2 N + 1)
        Total exciting coefficients a_n^j.
    """

    #: Sign of the translation phase e^{i (m - n) phi_jl}; only changed by the
    #: mutation tests (a wrong sign must break the optical theorem).
    _graf_sign = 1

    def __init__(
        self, centers, radii, m, k, polarization="TE", n_max=None, direction=0.0
    ):
        self.centers = np.atleast_2d(np.asarray(centers, dtype=float))
        nc = len(self.centers)
        self.radii = np.broadcast_to(np.asarray(radii, dtype=float), (nc,)).copy()
        m = np.broadcast_to(np.asarray(m, dtype=complex), (nc,))
        split = [_split_convention(mm) for mm in m]
        conj = {c for _, c in split}
        if len(conj) > 1 and np.any(np.abs(np.imag(m)) > 0):
            raise ValueError("mixed time conventions in m")
        self.m_bh = np.array([mm for mm, _ in split])
        self._conj = bool(split[0][1])
        self.k = float(k)
        self.pol = polarization.upper()
        if self.pol not in ("TM", "TE"):
            raise ValueError("polarization must be 'TM' or 'TE'")
        for j in range(nc):
            for l in range(j + 1, nc):
                d = np.hypot(*(self.centers[j] - self.centers[l]))
                if d <= self.radii[j] + self.radii[l]:
                    raise ValueError("cylinders overlap")
        if n_max is None:
            n_max = max(_n_stop(self.k * r) for r in self.radii)
            q = 0.0
            for j in range(nc):
                for l in range(j + 1, nc):
                    d = np.hypot(*(self.centers[j] - self.centers[l]))
                    q = max(q, bipolar_ratio(d, self.radii[j], self.radii[l]))
            if q > 0:
                n_max += int(np.ceil(np.log(1e-13) / np.log(q)))
            n_max = min(n_max, 60)
        self.N = int(n_max)
        self.direction = float(direction)
        self._solve()

    # ------------------------------------------------------------------ solve
    def _single(self):
        n = np.arange(self.N + 1)
        c = np.array(
            [
                _coefficients_bh(mm, self.k * r, self.pol, n_terms=self.N)
                for mm, r in zip(self.m_bh, self.radii)
            ]
        )
        # c_{-n} = c_n
        return np.concatenate([c[:, :0:-1], c], axis=1), n

    def _solve(self):
        """
        Solve the coupled system in surface-scaled amplitudes.

        Raw coefficients span hundreds of orders of magnitude for high n
        (a_n^j grows like (2 / k d)^n, b_n^j decays like (k a / 2)^n), and the
        raw system loses all accuracy for N >~ 20 in close dimers. With the
        surface amplitudes alpha_n = a_n J_n(k a_j) and beta_n = b_n
        H_n(k a_j) the system reads

            beta^j + D^j sum_l S^{jl} beta^l = -D^j alpha^{j,inc},
            D_n^j = c_n^j H_n(k a_j) / J_n(k a_j) = O(1),
            S_nm^{jl} = J_n(k a_j) T_nm^{jl} / H_m(k a_l) ~ (4 a_j a_l / d^2)^{|n|},

        which is well conditioned.
        """
        N, k, nc = self.N, self.k, len(self.centers)
        n = np.arange(-N, N + 1)
        with np.errstate(over="raise", invalid="raise"):
            c, _ = self._single()
            Jn = np.array([jv(n, k * r) for r in self.radii])
            Hn = np.array([hankel1(n, k * r) for r in self.radii])
        self.c = c
        phi = self.direction
        khat = np.array([np.cos(phi), np.sin(phi)])
        a_inc = (
            np.exp(1j * k * self.centers @ khat)[:, None]
            * ((1j**n) * np.exp(-1j * n * phi))[None, :]
        )
        D = c * Hn / Jn
        size = 2 * N + 1
        A = np.eye(nc * size, dtype=complex)
        S = {}
        for j in range(nc):
            for l in range(nc):
                if j == l:
                    continue
                R = self.centers[j] - self.centers[l]
                d, ang = np.hypot(*R), np.arctan2(R[1], R[0])
                diff = n[None, :] - n[:, None]  # m - n with rows n, cols m
                with np.errstate(over="raise", invalid="raise"):
                    Tjl = hankel1(diff, k * d) * np.exp(
                        1j * self._graf_sign * diff * ang
                    )
                    Sjl = Jn[j][:, None] * Tjl / Hn[l][None, :]
                if not np.all(np.isfinite(Sjl)):
                    raise FloatingPointError("translation overflow: reduce n_max")
                S[j, l] = Sjl
                A[j * size : (j + 1) * size, l * size : (l + 1) * size] = (
                    D[j][:, None] * Sjl
                )
        alpha_inc = a_inc * Jn
        beta = np.linalg.solve(A, -(D * alpha_inc).ravel()).reshape(nc, size)
        alpha = alpha_inc.copy()
        for (j, l), Sjl in S.items():
            alpha[j] += Sjl @ beta[l]
        self._alpha, self._beta, self._Jn, self._Hn = alpha, beta, Jn, Hn
        self._b = beta / Hn
        with np.errstate(over="ignore", invalid="ignore"):
            self._a = alpha / Jn
        self._a_inc, self._n = a_inc, n

    def _out(self, z):
        return np.conj(z) if self._conj else z

    @property
    def b(self):
        return self._out(self._b)

    @property
    def a(self):
        return self._out(self._a)

    # ------------------------------------------------------------- far field
    def _far(self, theta):
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        rhat = np.stack([np.cos(theta), np.sin(theta)], 1)
        ph = np.exp(-1j * self.k * rhat @ self.centers.T)  # (n_theta, n_cyl)
        modes = ((-1j) ** self._n)[None, :] * np.exp(1j * np.outer(theta, self._n))
        return np.einsum("tj,tn,jn->t", ph, modes, self._b)

    def far_amplitude(self, theta):
        """
        Far-field amplitude f(theta) per unit incident amplitude.

        u_s ~ f(theta) sqrt(2 / (pi k r)) exp(i (k r - pi / 4)) (exp(-i w t);
        conjugated for the Wakis convention).
        """
        return self._out(self._far(theta))

    def differential_width(self, theta):
        """Differential scattering width dC/dtheta = 2 |f|^2 / (pi k) [m/rad]."""
        return 2 * np.abs(self._far(theta)) ** 2 / (np.pi * self.k)

    def cross_sections(self, n_theta=None):
        """
        Extinction, scattering and absorption widths [m] per unit length.

        C_sca integrates |f|^2 with the trapezoidal rule on `n_theta` angles
        (exact for the band-limited f: default 4 (k R_max + N) + 64 angles,
        R_max the largest centre distance from the origin).
        """
        if n_theta is None:
            rmax = np.hypot(*self.centers.T).max()
            n_theta = int(4 * (self.k * rmax + self.N)) + 64
        th = 2 * np.pi * np.arange(n_theta) / n_theta
        f = self._far(th)
        c_sca = 2 / (np.pi * self.k) * np.mean(np.abs(f) ** 2) * 2 * np.pi
        c_ext = -4 / self.k * np.real(self._far(self.direction)[0])
        return c_ext, c_sca, c_ext - c_sca

    def absorption_per_cylinder(self):
        """C_abs,j [m] from the local incoming/outgoing power balance."""
        al, be, J, H = self._alpha, self._beta, self._Jn, self._Hn
        t = np.real(np.conj(al) * be / (J * H)) + np.abs(be / H) ** 2
        return -4 / self.k * np.sum(t, axis=1)

    # ------------------------------------------------------------ near field
    def _local(self, points, j):
        d = points - self.centers[j]
        rho = np.hypot(*d.T)
        th = np.arctan2(d[:, 1], d[:, 0])
        return rho, th

    def _scattered_and_grad(self, points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        k, n = self.k, self._n
        u = np.zeros(len(points), complex)
        dpx = np.zeros(len(points), complex)  # (dx + i dy) u
        dmx = np.zeros(len(points), complex)  # (dx - i dy) u
        inside = np.zeros(len(points), bool)
        for j in range(len(self.centers)):
            rho, th = self._local(points, j)
            inside |= rho < self.radii[j] * (1 - 1e-12)
            kr = k * np.maximum(rho, 1e-300)
            H = hankel1(n[None, :], kr[:, None])
            Hp = hankel1(n[None, :] + 1, kr[:, None])
            Hm = hankel1(n[None, :] - 1, kr[:, None])
            e = np.exp(1j * np.outer(th, n))
            # beta_n H(k rho) / H_n(k a): the ratios stay O((a / rho)^|n|)
            b = self._beta[j][None, :] / self._Hn[j][None, :]
            u += np.sum(b * H * e, 1)
            dpx += np.sum(b * (-k) * Hp * e * np.exp(1j * th)[:, None], 1)
            dmx += np.sum(b * k * Hm * e * np.exp(-1j * th)[:, None], 1)
        if np.any(inside):
            raise ValueError("field points must lie outside the cylinders")
        return u, 0.5 * (dpx + dmx), (dpx - dmx) / 2j

    def field(self, points):
        """
        Total field u (E_z for TM, H_z for TE) at points outside the cylinders,
        per unit incident amplitude.

        Returns
        -------
        u_total, u_scattered : ndarray of complex
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        us, _, _ = self._scattered_and_grad(points)
        khat = np.array([np.cos(self.direction), np.sin(self.direction)])
        ui = np.exp(1j * self.k * points @ khat)
        return self._out(ui + us), self._out(us)

    def in_plane_e(self, points):
        """
        TE only: in-plane electric field E = (i/k) (d_y u, -d_x u) normalised by
        the incident |E0| (E_inc = (-sin phi, cos phi) e^{i k.r}).

        Returns
        -------
        E : ndarray (n_points, 2) of complex
            Total field E / E0.
        """
        if self.pol != "TE":
            raise ValueError("in_plane_e needs polarization='TE'")
        points = np.atleast_2d(np.asarray(points, dtype=float))
        _, ux, uy = self._scattered_and_grad(points)
        khat = np.array([np.cos(self.direction), np.sin(self.direction)])
        ui = np.exp(1j * self.k * points @ khat)
        ux = ux + 1j * self.k * khat[0] * ui
        uy = uy + 1j * self.k * khat[1] * ui
        E = (1j / self.k) * np.stack([uy, -ux], 1)
        return self._out(E)
