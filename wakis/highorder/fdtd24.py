# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
2D Cartesian staggered finite differences of order (2, 2) (Yee) or (2, 4)
(fourth-order staggered differences with coefficients 9/8 and -1/24, see
e.g. Hwang & Cangellaris, Electromagnetics 23, 89 (2003); Yefet &
Petropoulos, J. Comput. Phys. 168, 286 (2001)), with

* leapfrog (2nd order in time) or Yoshida's 4th-order symplectic composition
  of three leapfrog steps (Phys. Lett. A 150, 262 (1990));
* sub-pixel averaging of circular inclusions: TM (E_z tangential) uses the
  area-averaged eps; TE uses the diagonal of the averaged tensor
  (eps^-1)_ii = n_i^2 <1/eps> + (1 - n_i^2) / <eps> with the local
  interface normal n (second-order "subpixel smoothing");
* scattered-field plane wave in the constitutive form D_s = D - eps0 E_inc,
  E_s = kappa D_s / eps0 + (kappa - 1) E_inc with kappa = eps_eff^-1
  (exact at every time level, no time derivative of the incident field);
* convolutional PML (Roden & Gedney, MOTL 27, 334 (2000)) on all sides.

Uniform square grid; the scalar field (Ez for TM on nodes, Hz for TE on cell
centres) is the monitored quantity. Linear in all fields with an explicitly
transposable step (sums of difference operators and diagonal scalings).
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import c as C0
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import mu_0 as MU0

__all__ = ["FDTD24"]



def _d_half(f, axis, c1, c2, h):
    """Derivative at half points (length N-1) from integer points (N).
    PEC image ghosts: integer-point fields are odd about the wall node."""
    f = np.moveaxis(f, axis, 0)
    fp = np.concatenate([-f[1:2], f, -f[-2:-1]])
    d = c1 * (fp[2:-1] - fp[1:-2])
    if c2:
        d = d + c2 * (fp[3:] - fp[:-3])
    return np.moveaxis(d / h, 0, axis)


def _d_int(g, axis, c1, c2, h):
    """Derivative at integer points 1..N-2 (returned with length N, ends 0)
    from half points (N-1 values). PEC image ghosts: half-point fields are
    even about the wall."""
    g = np.moveaxis(g, axis, 0)
    gp = np.concatenate([g[:1], g, g[-1:]])
    # node i (1..N-2): g[i-1/2] = gp[i], g[i+1/2] = gp[i+1]
    d = c1 * (gp[2:-1] - gp[1:-2])
    if c2:
        d = d + c2 * (gp[3:] - gp[:-3])
    out = np.zeros((g.shape[0] + 1,) + g.shape[1:])
    out[1:-1] = d
    return np.moveaxis(out / h, 0, axis)


def _cubic_weights(t):
    """4-point Lagrange weights for nodes -1, 0, 1, 2 at fraction t."""
    return np.array([
        -t * (t - 1) * (t - 2) / 6, (t + 1) * (t - 1) * (t - 2) / 2,
        -(t + 1) * t * (t - 2) / 2, (t + 1) * t * (t - 1) / 6,
    ])  # fmt: skip


class FDTD24:
    """
    Parameters
    ----------
    h : float
        Cell size [m].
    half_width : float
        Half width of the physical (non-PML) square [m].
    n_pml : int
        CPML cells on each side.
    cylinders : list of (cx, cy, R, eps_r) or (cx, cy, R, eps_r, sigma)
        Circular inclusions; sigma [S/m] is an optional static conductivity
        (area-averaged, Crank-Nicolson in leapfrog).
    polarization : {'TM', 'TE'}
    order : {2, 4}
        Spatial order.
    time_scheme : {'leapfrog', 'lsrk4'}
        Leapfrog (2nd order, staggered in time) or the 5-stage 4th-order
        low-storage Runge-Kutta of `wakis.highorder.dgtd2d` (all fields at the
        same time level; CPML auxiliaries integrated as ADEs). Yoshida's
        symplectic composition was tried and is unstable with the CPML: its
        negative sub-step turns the absorber into an amplifier.
    cfl : float
        Fraction of the stability limit.
    supersample : int
        Sub-samples per cell side for the averaging.
    """

    def __init__(self, h, half_width, n_pml, cylinders, polarization="TM",
                 order=4, time_scheme="leapfrog", cfl=0.95, supersample=16, pml_R0=1e-8):  # fmt: skip
        self.h, self.pol, self.order = h, polarization.upper(), order
        self.rk = time_scheme == "lsrk4"
        if time_scheme not in ("leapfrog", "lsrk4"):
            raise ValueError("time_scheme must be 'leapfrog' or 'lsrk4'")
        self.c1, self.c2 = (9 / 8, -1 / 24) if order == 4 else (1.0, 0.0)
        M = int(np.ceil(half_width / h)) + n_pml
        self.N = 2 * M + 1
        N = self.N
        self.xn = h * (np.arange(N) - M)  # integer nodes
        self.xh = 0.5 * (self.xn[:-1] + self.xn[1:])  # half nodes
        self.n_cells = (N - 1) ** 2
        self.n_dofs = 3 * self.n_cells
        # (cx, cy, R, eps_r[, sigma [S/m]])
        self.cylinders = [tuple(c) + (0.0,) * (5 - len(c)) for c in cylinders]
        # stability: max |symbol| = c1 + |c2| (7/6 for (2,4)) per direction
        lim = h / (C0 * np.sqrt(2) * (abs(self.c1) + abs(self.c2)))
        if self.rk:
            from .dgtd2d import _imag_axis_limit

            lim = lim * _imag_axis_limit() / 2  # leapfrog limit is |lambda| dt <= 2
        self.dt = cfl * lim
        # TE samples Hz, which lives at half steps in plain leapfrog
        self.sample_shift = -0.5 * self.dt if (self.pol == "TE" and not self.rk) else 0.0
        self._setup_materials(supersample)
        self._setup_pml(n_pml, pml_R0)
        self.reset()
        self.plane_wave = None

    # ------------------------------------------------------------ geometry
    def _fractions(self, xc, yc, ss):
        """Area fraction of cylinders and averaged eps, 1/eps and normals
        over the h x h cell centred at (xc, yc) (2D arrays)."""
        h = self.h
        o = (np.arange(ss) + 0.5) / ss - 0.5
        eps_m = np.zeros(xc.shape)
        inv_m = np.zeros(xc.shape)
        sig_m = np.zeros(xc.shape)
        nxs = np.zeros(xc.shape)
        nys = np.zeros(xc.shape)
        for ox in o:
            for oy in o:
                X, Y = xc + ox * h, yc + oy * h
                e = np.ones(xc.shape)
                sg = np.zeros(xc.shape)
                for cx, cy, R, er, sgm in self.cylinders:
                    inside = np.hypot(X - cx, Y - cy) < R
                    e = np.where(inside, er, e)
                    sg = np.where(inside, sgm, sg)
                eps_m += e
                inv_m += 1 / e
                sig_m += sg
        eps_m /= ss * ss
        inv_m /= ss * ss
        self._sig_tmp = sig_m / (ss * ss)
        for cx, cy, R, er, _ in self.cylinders:
            r = np.hypot(xc - cx, yc - cy)
            near = np.abs(r - R) < h
            nxs = np.where(near, (xc - cx) / np.maximum(r, 1e-30), nxs)
            nys = np.where(near, (yc - cy) / np.maximum(r, 1e-30), nys)
        return eps_m, inv_m, nxs, nys

    def _setup_materials(self, ss):
        xn, xh = self.xn, self.xh
        if self.pol == "TM":
            X, Y = np.meshgrid(xn, xn, indexing="ij")  # Ez nodes
            eps_m, _, _, _ = self._fractions(X, Y, ss)
            self.kz = 1 / eps_m  # kappa = eps_eff^-1
            self.sig_z = self._sig_tmp  # area-averaged conductivity
        else:
            # Ex at (xh_i, xn_j), Ey at (xn_i, xh_j)
            for comp, (gx, gy) in (("x", (xh, xn)), ("y", (xn, xh))):
                X, Y = np.meshgrid(gx, gy, indexing="ij")
                eps_m, inv_m, nx, ny = self._fractions(X, Y, ss)
                n2 = nx**2 if comp == "x" else ny**2
                kap = n2 * inv_m + (1 - n2) / eps_m
                setattr(self, "k" + comp, kap)
                setattr(self, "sig_" + comp, self._sig_tmp)

    def _setup_pml(self, n_pml, R0):
        h, dt = self.h, self.dt
        d = max(n_pml, 1e-9) * h
        m = 3
        smax = -(m + 1) * np.log(R0) / (2 * np.sqrt(MU0 / EPS0) * d)
        xmax = self.xn[-1]
        x0 = xmax - d

        def coeffs(x):
            depth = np.clip((np.abs(x) - x0) / d, 0, 1)
            sig = smax * depth**m
            alpha = 0.05 * (1 - depth) * 2 * np.pi * 100e12 * EPS0  # CFS
            b = np.exp(-(sig + alpha) * dt / EPS0)
            a = np.where(sig > 0, sig / (sig + alpha + 1e-300) * (b - 1), 0.0)
            return a, b

        def ade(x):
            depth = np.clip((np.abs(x) - x0) / d, 0, 1)
            sig = smax * depth**m
            alpha = 0.05 * (1 - depth) * 2 * np.pi * 100e12 * EPS0
            return None, (sig + alpha) / EPS0 * (sig > 0), sig / EPS0

        self._ade_n = ade(self.xn)
        self._ade_h = ade(self.xh)
        self.pml_n = coeffs(self.xn)
        self.pml_h = coeffs(self.xh)
        self.inpml_n = np.abs(self.xn) > x0
        self.inpml_h = np.abs(self.xh) > x0

    # ------------------------------------------------------------ state
    def reset(self):
        N = self.N
        if self.pol == "TM":
            self.Dz = np.zeros((N, N))  # D_s / eps0 on nodes
            self.Hx = np.zeros((N, N - 1))
            self.Hy = np.zeros((N - 1, N))
            self.psi = dict(ezx=np.zeros((N, N)), ezy=np.zeros((N, N)),
                            hx=np.zeros((N, N - 1)), hy=np.zeros((N - 1, N)))  # fmt: skip
        else:
            self.Hz = np.zeros((N - 1, N - 1))
            self.Dx = np.zeros((N - 1, N))
            self.Dy = np.zeros((N, N - 1))
            self.psi = dict(hzx=np.zeros((N - 1, N - 1)), hzy=np.zeros((N - 1, N - 1)),
                            ex=np.zeros((N - 1, N)), ey=np.zeros((N, N - 1)))  # fmt: skip
        self.t = 0.0
        self._K = None

    def add_plane_wave(self, src_time, amplitude=1.0):
        """+x plane wave, scattered-field formulation (vacuum background)."""
        s0 = min(c[0] - c[2] for c in self.cylinders)
        self.plane_wave = dict(s=src_time, amp=amplitude, s0=s0)

    def _einc(self, x, t):
        pw = self.plane_wave
        if pw is None:
            return 0.0
        return pw["amp"] * np.asarray(pw["s"](t - (x - pw["s0"]) / C0))

    def incident_field(self, points, t):
        """Incident Ez (TM) or Hz (TE) [A/m]."""
        pw = self.plane_wave
        x = np.atleast_2d(points)[:, 0]
        t = np.atleast_1d(t)
        e = pw["amp"] * np.asarray(pw["s"](t[:, None] - (x - pw["s0"]) / C0))
        return e if self.pol == "TM" else e / np.sqrt(MU0 / EPS0)

    # ------------------------------------------------------------ updates
    def _E_TM(self, t):
        Ei = self._einc(self.xn[:, None], t) * np.ones((1, self.N))
        Ez = self.kz * self.Dz + (self.kz - 1) * Ei
        Ez[0, :] = Ez[-1, :] = Ez[:, 0] = Ez[:, -1] = 0.0
        return Ez

    def _E_TE(self, t):
        # Ey on (xn_i, xh_j): incident Ey = s(t - x/c) (E along +y for +x
        # propagation, H_inc = +z |E| / eta)
        Eyi = self._einc(self.xn[:, None], t) * np.ones((1, self.N - 1))
        Ex = self.kx * self.Dx  # incident Ex = 0
        Ey = self.ky * self.Dy + (self.ky - 1) * Eyi
        return Ex, Ey

    def _kick_H(self, tau, t):
        """H += tau * (...) using E at time t."""
        c1, c2, h = self.c1, self.c2, self.h
        if self.pol == "TM":
            Ez = self._E_TM(t)
            dyE = _d_half(Ez, 1, c1, c2, h)  # at Hx (N, N-1)
            dxE = _d_half(Ez, 0, c1, c2, h)  # at Hy (N-1, N)
            a, b = self.pml_h
            self._cpml("hx", dyE, a[None, :], b[None, :], tau)
            self._cpml("hy", dxE, a[:, None], b[:, None], tau)
            self.Hx -= tau / MU0 * (dyE + self.psi["hx"])
            self.Hy += tau / MU0 * (dxE + self.psi["hy"])
        else:
            Ex, Ey = self._E_TE(t)
            dyEx = _d_half(Ex, 1, c1, c2, h)  # (N-1, N-1)
            dxEy = _d_half(Ey, 0, c1, c2, h)
            a, b = self.pml_h
            self._cpml("hzy", dyEx, a[None, :], b[None, :], tau)
            self._cpml("hzx", dxEy, a[:, None], b[:, None], tau)
            self.Hz += tau / MU0 * (dyEx + self.psi["hzy"] - dxEy - self.psi["hzx"])

    def _lossy(self, D, inc, sig, kap, Ei0, Ei1, tau):
        """Semi-implicit (Crank-Nicolson) conductive update of D' = D_s / eps0:
        dD'/dt = curl / eps0 - sigma kappa (D' + E_inc) / eps0."""
        if not np.any(sig):
            D += inc
            return
        beta = tau * sig * kap / (2 * EPS0)
        D[...] = (D * (1 - beta) + inc - beta * (Ei0 + Ei1)) / (1 + beta)

    def _kick_D(self, tau, t=0.0):
        c1, c2, h = self.c1, self.c2, self.h
        a, b = self.pml_n
        if self.pol == "TM":
            dxH = _d_int(self.Hy, 0, c1, c2, h)  # (N, N)
            dyH = _d_int(self.Hx, 1, c1, c2, h)
            self._cpml("ezx", dxH, a[:, None], b[:, None], tau)
            self._cpml("ezy", dyH, a[None, :], b[None, :], tau)
            inc = tau / EPS0 * (dxH + self.psi["ezx"] - dyH - self.psi["ezy"])
            ei0 = self._einc(self.xn[:, None], t) * np.ones((1, self.N))
            ei1 = self._einc(self.xn[:, None], t + tau) * np.ones((1, self.N))
            self._lossy(self.Dz, inc, self.sig_z, self.kz, ei0, ei1, tau)
        else:
            dyH = _d_int(self.Hz, 1, c1, c2, h)  # Ex (N-1, N)
            dxH = _d_int(self.Hz, 0, c1, c2, h)  # Ey (N, N-1)
            self._cpml("ex", dyH, a[None, :], b[None, :], tau)
            self._cpml("ey", dxH, a[:, None], b[:, None], tau)
            self._lossy(self.Dx, tau / EPS0 * (dyH + self.psi["ex"]), self.sig_x,
                        self.kx, 0.0, 0.0, tau)  # fmt: skip
            ei0 = self._einc(self.xn[:, None], t) * np.ones((1, self.N - 1))
            ei1 = self._einc(self.xn[:, None], t + tau) * np.ones((1, self.N - 1))
            self._lossy(self.Dy, -tau / EPS0 * (dxH + self.psi["ey"]), self.sig_y,
                        self.ky, ei0, ei1, tau)  # fmt: skip

    def _cpml(self, key, deriv, a, b, tau):
        """psi = b psi + a deriv (CPML recursion for the leapfrog step)."""
        self.psi[key] = b * self.psi[key] + a * deriv

    def one_step(self):
        dt = self.dt
        if self.rk:
            self._rk_step()
            return
        self._kick_H(dt, self.t)
        self._kick_D(dt, self.t)
        self.t += dt

    # ------------------------------------------------------------ Runge-Kutta
    def _fields(self):
        if self.pol == "TM":
            return ["Dz", "Hx", "Hy"]
        return ["Hz", "Dx", "Dy"]

    def _derivs(self, t):
        """Time derivatives of fields and (ADE) CPML auxiliaries at time t."""
        c1, c2, h = self.c1, self.c2, self.h
        an, gn, sn = self._ade_n
        ah, gh, sh = self._ade_h
        P = self.psi
        d = {}
        if self.pol == "TM":
            Ez = self._E_TM(t)
            dyE = _d_half(Ez, 1, c1, c2, h)
            dxE = _d_half(Ez, 0, c1, c2, h)
            d["Hx"] = -(dyE + P["hx"]) / MU0
            d["Hy"] = (dxE + P["hy"]) / MU0
            d["hx"] = -gh[None, :] * P["hx"] - sh[None, :] * dyE
            d["hy"] = -gh[:, None] * P["hy"] - sh[:, None] * dxE
            dxH = _d_int(self.Hy, 0, c1, c2, h)
            dyH = _d_int(self.Hx, 1, c1, c2, h)
            d["Dz"] = (dxH + P["ezx"] - dyH - P["ezy"]) / EPS0
            if np.any(self.sig_z):
                ei = self._einc(self.xn[:, None], t)
                d["Dz"] -= self.sig_z * self.kz * (self.Dz + ei) / EPS0
            d["ezx"] = -gn[:, None] * P["ezx"] - sn[:, None] * dxH
            d["ezy"] = -gn[None, :] * P["ezy"] - sn[None, :] * dyH
        else:
            Ex, Ey = self._E_TE(t)
            dyEx = _d_half(Ex, 1, c1, c2, h)
            dxEy = _d_half(Ey, 0, c1, c2, h)
            d["Hz"] = (dyEx + P["hzy"] - dxEy - P["hzx"]) / MU0
            d["hzy"] = -gh[None, :] * P["hzy"] - sh[None, :] * dyEx
            d["hzx"] = -gh[:, None] * P["hzx"] - sh[:, None] * dxEy
            dyH = _d_int(self.Hz, 1, c1, c2, h)
            dxH = _d_int(self.Hz, 0, c1, c2, h)
            d["Dx"] = (dyH + P["ex"]) / EPS0
            d["Dy"] = -(dxH + P["ey"]) / EPS0
            if np.any(self.sig_x) or np.any(self.sig_y):
                ei = self._einc(self.xn[:, None], t)
                d["Dx"] -= self.sig_x * self.kx * self.Dx / EPS0
                d["Dy"] -= self.sig_y * self.ky * (self.Dy + ei) / EPS0
            d["ex"] = -gn[None, :] * P["ex"] - sn[None, :] * dyH
            d["ey"] = -gn[:, None] * P["ey"] - sn[:, None] * dxH
        return d

    def _get(self, k):
        return getattr(self, k) if k in self._fields() else self.psi[k]

    def _rk_step(self):
        from .dgtd2d import LSRK4

        dt = self.dt
        keys = self._fields() + list(self.psi)
        if not hasattr(self, "_K") or self._K is None:
            self._K = {k: np.zeros_like(self._get(k)) for k in keys}
        Kr = self._K
        for k in keys:
            Kr[k][:] = 0.0
        for a, b, c in zip(LSRK4["a"], LSRK4["b"], LSRK4["c"]):
            d = self._derivs(self.t + c * dt)
            for k in keys:
                Kr[k] *= a
                Kr[k] += dt * d[k]
                self._get(k)[...] += b * Kr[k]
        self.t += dt

    def scalar_field(self):
        if self.pol == "TM":
            return self._E_TM(self.t)
        return self.Hz

    # ------------------------------------------------------------ monitors
    def interp_matrix(self, points):
        """Bicubic (4 x 4 Lagrange) interpolation of the scalar field."""
        g = self.xn if self.pol == "TM" else self.xh
        n = len(g)
        rows, cols, vals = [], [], []
        for p, (px, py) in enumerate(np.atleast_2d(points)):
            i = int(np.floor((px - g[0]) / self.h))
            j = int(np.floor((py - g[0]) / self.h))
            wx = _cubic_weights((px - g[i]) / self.h)
            wy = _cubic_weights((py - g[j]) / self.h)
            for a_ in range(4):
                for b_ in range(4):
                    rows.append(p)
                    cols.append((i - 1 + a_) * n + (j - 1 + b_))
                    vals.append(wx[a_] * wy[b_])
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(points), n * n))
