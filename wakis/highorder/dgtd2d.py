# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
High-order nodal discontinuous Galerkin time-domain (DGTD) solver for 2D
Maxwell equations (TM: Ez, Hx, Hy; TE: Hz, Ex, Ey) on triangles.

Formulation (derived in the physics guide, "High-order DGTD"). Both
polarizations are one scalar-wave system for (u, vx, vy) with coefficients
(a, b)::

    a du/dt  = d_x vy - d_y vx        TM: u = Ez, v = (Hx, Hy), a = eps, b = mu
    b dvx/dt = -d_y u                 TE: u = Hz, v = -(Ex, Ey), a = mu, b = eps
    b dvy/dt =  d_x u

* Nodal basis of degree p (warp-and-blend nodes), strong DG form, exact
  upwind (Riemann) flux for piecewise-constant media with impedance
  Z = sqrt(b / a); ``alpha`` in [0, 1] blends to the impedance-weighted
  central flux (alpha = 0, energy-conserving).
* Curved (isoparametric) elements on every circle of the mesh: element nodes
  are displaced by a linear blending of the exact arc (Gordon-Hall type),
  with per-element J-weighted mass and face mass matrices by cubature.
* Time stepping: 5-stage 4th-order low-storage Runge-Kutta (Carpenter &
  Kennedy 1994, NASA TM-109112).
* Boundaries: first-order Silver-Mueller absorbing ('abc', zero exterior
  state), 'pec', 'pmc'; optional stretched-coordinate PML in a rectangular
  frame (auxiliary-differential-equation form, see `_rhs`).
* Plane-wave incidence in the scattered-field formulation (volume source
  inside non-background elements; exact for any element geometry because
  the flux couples only the continuous tangential components).
* Dispersive media (`wakis.dispersive.DispersiveMaterial`: Drude, Lorentz,
  Debye poles and conductivity) by ADEs integrated with the same RK scheme;
  they act on u (TM) or on v (TE), i.e. always on the electric field.

The semi-discrete system dX/dt = A X + f(t) is linear; `operator_matrix`
assembles A as a sparse matrix (by graph-coloured probing of the
matrix-free right-hand side), so the exact discrete adjoint of a step is the
same Runge-Kutta scheme with A^T.

All quantities SI.
"""

import time

import numpy as np
import scipy.sparse as sp
from scipy.constants import c as C0
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import mu_0 as MU0

from .nodal2d import collapsed_cubature, reference_triangle

__all__ = ["DGTD2D", "LSRK4"]

# Carpenter & Kennedy (1994) five-stage fourth-order 2N-storage RK
LSRK4 = dict(
    a=np.array([0.0, -567301805773 / 1357537059087, -2404267990393 / 2016746695238,
                -3550918686646 / 2091501179385, -1275806237668 / 842570457699]),
    b=np.array([1432997174477 / 9575080441755, 5161836677717 / 13612068292357,
                1720146321549 / 2090206949498, 3134564353537 / 4481467310338,
                2277821191437 / 14882151754819]),
    c=np.array([0.0, 1432997174477 / 9575080441755, 2526269341429 / 6820363183699,
                2006345519317 / 3224310063776, 2802321613138 / 2924317926251]),
)  # fmt: skip


def lsrk_stability(z, scheme=LSRK4):
    """Amplification factor R(z) of the low-storage RK for y' = lambda y,
    z = lambda dt."""
    z = np.asarray(z, complex)
    y = np.ones_like(z)
    k = np.zeros_like(z)
    for a, b in zip(scheme["a"], scheme["b"]):
        k = a * k + z * y
        y = y + b * k
    return y


def _imag_axis_limit(scheme=LSRK4):
    beta = np.linspace(0, 6, 60001)
    ok = np.abs(lsrk_stability(1j * beta, scheme)) <= 1 + 1e-12
    return beta[np.argmin(ok) - 1] if not ok.all() else beta[-1]


def _ddt(f, t, dt):
    """4th-order central difference of a callable."""
    return (8 * (f(t + dt) - f(t - dt)) - (f(t + 2 * dt) - f(t - 2 * dt))) / (12 * dt)


def _source_derivative(src, t, dt_fd):
    """ds/dt: analytic for `wakis.sources.GaussianSource`, else a 4th-order
    central difference."""
    from wakis.sources import GaussianSource

    if type(src) is GaussianSource:
        tau = t - src.t0
        w = 2 * np.pi * src.frequency
        s2 = src.sigma_t**2
        g = np.exp(-(tau**2) / (2 * s2))
        d = g * (-2 * tau / s2 * np.cos(w * tau)
                 + (tau**2 / (w * s2**2) - w - 1 / (w * s2)) * np.sin(w * tau))  # fmt: skip
        return np.where((t >= src.start_time) & (t <= src.end_time), d, 0.0)
    return _ddt(lambda q: np.asarray(src(q)), t, dt_fd)


class DGTD2D:
    """
    Nodal DGTD solver on a `wakis.highorder.mesh2d.TriMesh`.

    Parameters
    ----------
    mesh : TriMesh
    p : int
        Polynomial degree (>= 1).
    materials : dict
        tag -> relative permittivity (float), [eps_r, mu_r] or a
        `DispersiveMaterial`. Missing tags are vacuum.
    polarization : {'TM', 'TE'}
    bc : {'abc', 'pec', 'pmc'}
        Outer boundary condition.
    pml : dict, optional
        Rectangular PML inside a box mesh: ``thickness`` [m] (from every box
        side inwards), ``R0`` (normal-incidence reflection, default 1e-8),
        ``order`` (default 3), ``alpha`` (CFS shift [1/s], default 5e13).
        Elements must not straddle the PML start
        (the mesh should be fine enough there; the profile is polynomial
        inside elements anyway).
    alpha : float
        Upwind parameter (1 = upwind, 0 = central).
    curved : bool
        Curve the elements along the mesh circles (False: straight-sided
        polygonal interfaces, for studies of the geometric error).
    cfl : float
        Fraction of the RK stability limit (from the spectral radius).
    background_eps_r : float
        Relative permittivity of the (lossless) background: tags missing from
        `materials`, the medium of the incident wave and of the PML.
    """

    def __init__(
        self,
        mesh,
        p,
        materials=None,
        polarization="TM",
        bc="abc",
        pml=None,
        alpha=1.0,
        curved=True,
        cfl=0.8,
        dt=None,
        verbose=False,
        background_eps_r=1.0,
    ):
        t0 = time.time()
        self.mesh, self.p = mesh, int(p)
        self.pol = polarization.upper()
        if self.pol not in ("TM", "TE"):
            raise ValueError("polarization must be 'TM' or 'TE'")
        self.bc, self.alpha, self.curved_on = bc, float(alpha), curved
        T = reference_triangle(self.p)
        self.ref = T
        Np, Nfp, K = T.Np, T.Nfp, mesh.n_cells
        self.Np, self.Nfp, self.K = Np, Nfp, K
        self.eps_b = float(background_eps_r)
        self._setup_materials(materials or {})
        self._setup_geometry()
        self._setup_connectivity()
        self._setup_pml(pml)
        self._setup_dispersive()
        self._alloc_state()
        self.plane_wave = None
        self.t = 0.0
        self.time_rhs = 0.0
        self.dt = float(dt) if dt is not None else None
        if self.dt is None:
            self.rho = self.spectral_radius()
            self.dt = cfl * _imag_axis_limit() / self.rho
        self.setup_s = time.time() - t0
        if verbose:
            print(
                f"DGTD2D p={p} K={K} dofs={self.n_dofs} curved={self.n_curved} "
                f"dt={self.dt:.3e} s setup {self.setup_s:.1f} s"
            )

    # ------------------------------------------------------------ setup
    def _setup_materials(self, materials):
        from wakis.dispersive import DispersiveMaterial

        K = self.K
        eps = np.full(K, self.eps_b)
        mu = np.ones(K)
        self.disp_mats = {}
        for tag, m in materials.items():
            sel = self.mesh.tags == tag
            if isinstance(m, DispersiveMaterial):
                eps[sel] = m.eps_inf
                if m.poles or m.sigma:
                    self.disp_mats[tag] = m
            elif np.ndim(m) == 0:
                eps[sel] = float(m)
            else:
                eps[sel], mu[sel] = float(m[0]), float(m[1])
        self.eps_r, self.mu_r = eps, mu
        if self.pol == "TM":
            self.a, self.b = EPS0 * eps, MU0 * mu
            self.a0, self.b0 = EPS0 * self.eps_b, MU0
        else:
            self.a, self.b = MU0 * mu, EPS0 * eps
            self.a0, self.b0 = MU0, EPS0 * self.eps_b
        self.Z = np.sqrt(self.b / self.a)
        self.Y = 1 / self.Z

    def _setup_geometry(self):
        T, mesh = self.ref, self.mesh
        Np, K = self.Np, self.K
        V = mesh.nodes[mesh.cells]  # (K, 3, 2)
        r, s = T.r[:, None], T.s[:, None]
        L1, L2, L3 = -(r + s) / 2, (1 + r) / 2, (1 + s) / 2
        lam = [L1, L2, L3]
        x = L1 * V[:, 0, 0] + L2 * V[:, 1, 0] + L3 * V[:, 2, 0]
        y = L1 * V[:, 0, 1] + L2 * V[:, 1, 1] + L3 * V[:, 2, 1]
        self.x_affine, self.y_affine = x.copy(), y.copy()
        self.curved = np.zeros(K, bool)
        self.curved_face = -np.ones(K, int)
        self.curved_circle = -np.ones(K, int)
        if self.curved_on:
            circ = list(mesh.circles)
            if getattr(mesh, "outer_circle", None) is not None:
                circ.append(tuple(mesh.outer_circle))
            for ci, (cx, cy, R) in enumerate(circ):
                rv = np.hypot(V[..., 0] - cx, V[..., 1] - cy) - R
                on = np.abs(rv) < 1e-9 * R
                for f in range(3):
                    i, j = f, (f + 1) % 3
                    ks = np.flatnonzero(on[:, i] & on[:, j])
                    if np.any(self.curved[ks]):
                        raise RuntimeError("element with two curved faces")
                    for k in ks:
                        thi = np.arctan2(V[k, i, 1] - cy, V[k, i, 0] - cx)
                        thj = np.arctan2(V[k, j, 1] - cy, V[k, j, 0] - cx)
                        dth = (thj - thi + np.pi) % (2 * np.pi) - np.pi
                        # Szabo-Babuska blending: d = 4 li lj delta(xi) / (1 - xi^2)
                        # with xi = lj - li and delta the arc-minus-chord
                        # displacement of the edge (smooth, polynomial-friendly)
                        li, lj = lam[i][:, 0], lam[j][:, 0]
                        xi = lj - li
                        tt = 0.5 * (1 + xi)
                        th = thi + tt * dth
                        Cx, Cy = cx + R * np.cos(th), cy + R * np.sin(th)
                        Xx = (1 - tt) * V[k, i, 0] + tt * V[k, j, 0]
                        Xy = (1 - tt) * V[k, i, 1] + tt * V[k, j, 1]
                        den = 1 - xi**2
                        fac = np.where(den > 1e-12, 4 * li * lj / np.maximum(den, 1e-300), 0.0)
                        x[:, k] += fac * (Cx - Xx)
                        y[:, k] += fac * (Cy - Xy)
                        self.curved[k] = True
                        self.curved_face[k] = f
                        self.curved_circle[k] = ci
        self.x, self.y = x, y
        self.n_curved = int(self.curved.sum())
        xr, xs, yr, ys = T.Dr @ x, T.Ds @ x, T.Dr @ y, T.Ds @ y
        J = xr * ys - xs * yr
        if np.any(J <= 0):
            raise RuntimeError("non-positive Jacobian (inverted element)")
        self.J = J
        self.rx, self.sx = ys / J, -yr / J
        self.ry, self.sy = -xs / J, xr / J
        # face normals and line Jacobians at face nodes
        Fm = T.Fmask
        Nfp = self.Nfp
        nx = np.zeros((3 * Nfp, K))
        ny = np.zeros_like(nx)
        sJ = np.zeros_like(nx)
        for f in range(3):
            sl = slice(f * Nfp, (f + 1) * Nfp)
            fr, fs_ = xr[Fm[f]], xs[Fm[f]]
            gr, gs = yr[Fm[f]], ys[Fm[f]]
            if f == 0:
                tx, ty = fr, gr
            elif f == 1:
                tx, ty = -fr + fs_, -gr + gs
            else:
                tx, ty = -fs_, -gs
            L = np.hypot(tx, ty)
            nx[sl], ny[sl], sJ[sl] = ty / L, -tx / L, L
        self.nx, self.ny, self.sJ = nx, ny, sJ
        self.Fscale = sJ / J[Fm.ravel()]
        # curved elements: J-weighted mass and face mass by cubature
        self.curved_idx = np.flatnonzero(self.curved)
        if self.n_curved:
            q = 2 * self.p + 3
            rc, sc, wc = collapsed_cubature(q)
            Lc = T.interp_matrix(rc, sc)
            Lr, Ls = T.grad_interp_matrices(rc, sc)
            tg, wg = np.polynomial.legendre.leggauss(self.p + 4)
            Lf = T.face_interp_matrix(tg)
            lifts = []
            for k in self.curved_idx:
                Jc = (Lr @ x[:, k]) * (Ls @ y[:, k]) - (Ls @ x[:, k]) * (Lr @ y[:, k])
                MJ = Lc.T @ ((wc * Jc)[:, None] * Lc)
                E = np.zeros((Np, 3 * Nfp))
                for f in range(3):
                    rf, sf = T.face_points(f, tg)
                    Gr, Gs = T.grad_interp_matrices(rf, sf)
                    dxr, dxs = Gr @ x[:, k], Gs @ x[:, k]
                    dyr, dys = Gr @ y[:, k], Gs @ y[:, k]
                    if f == 0:
                        tx, ty = dxr, dyr
                    elif f == 1:
                        tx, ty = -dxr + dxs, -dyr + dys
                    else:
                        tx, ty = -dxs, -dys
                    sJg = np.hypot(tx, ty)
                    Mf = Lf.T @ ((wg * sJg)[:, None] * Lf)
                    E[np.ix_(Fm[f], f * Nfp + np.arange(Nfp))] = Mf
                lifts.append(np.linalg.solve(MJ, E))
            self.LIFT_curved = np.array(lifts)  # (Kc, Np, 3 Nfp)
            self.Fscale[:, self.curved_idx] = 0.0  # lift handled separately
        else:
            self.LIFT_curved = np.zeros((0, Np, 3 * Nfp))
        # integration weights for norms: w_n J (straight elements: exact mass
        # diagonal is not available; use the mass-matrix quadratic form)
        self.area = (T.M.sum(0)[:, None] * J).sum(0)

    def _setup_connectivity(self):
        T, mesh = self.ref, self.mesh
        K, Np, Nfp = self.K, self.Np, self.Nfp
        c = mesh.cells
        fv = np.stack([c[:, [0, 1]], c[:, [1, 2]], c[:, [2, 0]]], 1)  # (K, 3, 2)
        key = np.sort(fv, 2).reshape(-1, 2)
        order = np.lexsort((key[:, 1], key[:, 0]))
        ks = key[order]
        same = np.all(ks[1:] == ks[:-1], 1)
        EToE = np.repeat(np.arange(K), 3).reshape(K, 3).copy()
        EToF = np.tile(np.arange(3), (K, 1))
        i1, i2 = order[:-1][same], order[1:][same]
        k1, f1 = i1 // 3, i1 % 3
        k2, f2 = i2 // 3, i2 % 3
        EToE[k1, f1], EToF[k1, f1] = k2, f2
        EToE[k2, f2], EToF[k2, f2] = k1, f1
        self.EToE, self.EToF = EToE, EToF
        self.boundary = EToE == np.arange(K)[:, None]  # (K, 3)
        # node maps: global index n * K + k (C-order of (Np, K) arrays)
        Fm = T.Fmask
        idxM = np.zeros((3 * Nfp, K), np.int64)
        idxP = np.zeros_like(idxM)
        xa, ya = self.x_affine, self.y_affine
        for f in range(3):
            sl = slice(f * Nfp, (f + 1) * Nfp)
            idxM[sl] = Fm[f][:, None] * K + np.arange(K)
        for f in range(3):
            sl = slice(f * Nfp, (f + 1) * Nfp)
            kn, fn = EToE[:, f], EToF[:, f]
            # neighbour face nodes are traversed in the opposite direction
            nodes_n = Fm[fn][:, ::-1].T  # (Nfp, K)
            idxP[sl] = nodes_n * K + kn
            # verify coordinates
            xm, ym = xa.ravel()[idxM[sl]], ya.ravel()[idxM[sl]]
            xp, yp = xa.ravel()[idxP[sl]], ya.ravel()[idxP[sl]]
            dist = np.hypot(xm - xp, ym - yp)
            h = np.sqrt(self.area).max()
            interior = ~self.boundary[:, f]
            if np.any(dist[:, interior] > 1e-8 * h):
                raise RuntimeError("face node matching failed")
            idxP[sl][:, ~interior] = idxM[sl][:, ~interior]
        self.idxM, self.idxP = idxM, idxP
        self.bmask = np.repeat(self.boundary.T, Nfp, 0)  # (3 Nfp, K)
        kP = idxP % K
        self.ZM = np.broadcast_to(self.Z, (3 * Nfp, K)).copy()
        self.ZP = np.where(self.bmask, self.ZM, self.Z[kP])
        self.YM, self.YP = 1 / self.ZM, 1 / self.ZP

    def _setup_pml(self, pml):
        self.pml = None
        if not pml:
            return
        box = self.mesh.box
        if box is None:
            raise ValueError("PML needs a box mesh")
        d = float(pml["thickness"])
        R0 = pml.get("R0", 1e-8)
        m = pml.get("order", 3)
        c_max = 1 / np.sqrt(self.a0 * self.b0)
        smax = -(m + 1) * c_max * np.log(R0) / (2 * d)
        x0, x1, y0, y1 = box

        def prof(q, lo, hi):
            dd = np.maximum(np.maximum(lo + d - q, q - (hi - d)), 0) / d
            return smax * dd**m

        sx = prof(self.x, x0, x1)
        sy = prof(self.y, y0, y1)
        kp = np.flatnonzero((sx.max(0) > 0) | (sy.max(0) > 0))
        if np.any(self.curved[kp]):
            raise ValueError("curved elements inside the PML")
        if np.any(self.eps_r[kp] != self.eps_b) or np.any(self.mu_r[kp] != 1):
            raise ValueError("PML must be in the background medium")
        alpha = pml.get("alpha", 5e13)  # CFS shift [1/s]
        self.pml = dict(k=kp, sx=sx[:, kp], sy=sy[:, kp], d=d, smax=smax, alpha=alpha)

    def _setup_dispersive(self):
        """ADE state layout for dispersive elements."""
        self.disp = []
        for tag, m in self.disp_mats.items():
            ks = np.flatnonzero(self.mesh.tags == tag)
            coef = self.a[ks] if self.pol == "TM" else self.b[ks]
            entry = dict(k=ks, coef=coef, sigma=m.sigma, lor=[], deb=[])
            for pole in m.lorentz_poles:
                entry["lor"].append((pole.strength, pole.omega_0**2, pole.gamma))
            for pole in m.debye_poles:
                entry["deb"].append((pole.delta_eps, pole.tau))
            self.disp.append(entry)

    def _alloc_state(self):
        Np, K = self.Np, self.K
        sizes = [("u", Np * K), ("vx", Np * K), ("vy", Np * K)]
        if self.pml is not None:
            nk = len(self.pml["k"])
            sizes += [(q, Np * nk) for q in ("Qx", "Qy", "Rx", "Ry")]
        nw = 1 if self.pol == "TM" else 2
        for i, d in enumerate(self.disp):
            nk = len(d["k"])
            for j, (S, w02, g) in enumerate(d["lor"]):
                sizes.append((f"J{i}_{j}", nw * Np * nk))
                if w02 > 0:
                    sizes.append((f"P{i}_{j}", nw * Np * nk))
            for j in range(len(d["deb"])):
                sizes.append((f"D{i}_{j}", nw * Np * nk))
        self.layout = {}
        off = 0
        for name, n in sizes:
            self.layout[name] = slice(off, off + n)
            off += n
        self.n_state = off
        self.n_dofs = 3 * Np * K
        self.X = np.zeros(off)
        self._res = np.zeros(off)

    def _view(self, X, name, shape=None):
        v = X[self.layout[name]]
        return v.reshape(shape) if shape else v.reshape(self.Np, -1)

    # ------------------------------------------------------------ sources
    def add_plane_wave(self, src_time, direction=(1.0, 0.0), amplitude=1.0):
        """
        Plane wave in the scattered-field formulation (background medium).

        u_inc = A s(t - (k.r - s0) / c) with s0 the smallest k.r over the
        nodes of non-background elements; A = amplitude (TM, Ez [V/m]) or
        amplitude / eta_b (TE, Hz [A/m]). After this call the state holds the
        scattered field.
        """
        k = np.asarray(direction, float)
        k = k / np.linalg.norm(k)
        scat = (self.eps_r != self.eps_b) | (self.mu_r != 1)
        for d in self.disp:
            scat[d["k"]] = True
        ks = np.flatnonzero(scat)
        pos = (self.x * k[0] + self.y * k[1])[:, ks]
        s0 = float(pos.min()) if ks.size else 0.0
        eta_b = np.sqrt(MU0 / (EPS0 * self.eps_b))
        amp = amplitude if self.pol == "TM" else amplitude / eta_b
        Zg0 = np.sqrt(self.b0 / self.a0)
        ka = self.a[ks] - self.a0
        kb = self.b[ks] - self.b0
        self.plane_wave = dict(
            s=src_time, k=k, s0=s0, amp=amp, ks=ks, pos=pos, c=C0 / np.sqrt(self.eps_b),
            ca=-(ka / self.a[ks]), cb=-(kb / self.b[ks]),
            vdir=np.array([k[1], -k[0]]) / Zg0,
            dt_fd=1e-3 / max(getattr(src_time, "frequency", C0 / 1e-6), 1.0),
        )  # fmt: skip
        # index of each dispersive element in the scatterer list
        loc = -np.ones(self.K, int)
        loc[ks] = np.arange(ks.size)
        for d in self.disp:
            d["kloc"] = loc[d["k"]]

    def _incident(self, t, deriv=False):
        """u_inc (or d/dt) at the scatterer nodes, (Np, Ks)."""
        pw = self.plane_wave
        tau = t - (pw["pos"] - pw["s0"]) / pw["c"]
        if deriv:
            return pw["amp"] * _source_derivative(pw["s"], tau, pw["dt_fd"])
        return pw["amp"] * np.asarray(pw["s"](tau))

    def incident_field(self, points, t):
        """Incident u (Ez for TM [V/m], Hz for TE [A/m]) at points (n, 2)
        and times t (array)."""
        pw = self.plane_wave
        pos = np.atleast_2d(points) @ pw["k"]
        t = np.atleast_1d(t)
        return pw["amp"] * np.asarray(pw["s"](t[:, None] - (pos - pw["s0"]) / pw["c"]))

    # ------------------------------------------------------------ operator
    def _lift(self, F):
        """Lift face values F (3 Nfp, K) (already multiplied by nothing):
        straight elements LIFT_ref (Fscale F), curved per-element LIFT."""
        nb = F.shape[1] // self.K
        out = self.ref.LIFT @ (np.tile(self.Fscale, nb) * F)
        if self.n_curved:
            for q in range(nb):
                ci = self.curved_idx + q * self.K
                out[:, ci] = np.einsum("kij,jk->ik", self.LIFT_curved, F[:, ci])
        return out

    def _rhs(self, X, t, with_source=True):
        Np, K = self.Np, self.K
        T = self.ref
        u = self._view(X, "u")
        vx = self._view(X, "vx")
        vy = self._view(X, "vy")
        res = np.zeros_like(X)
        du = self._view(res, "u")
        dvx = self._view(res, "vx")
        dvy = self._view(res, "vy")
        # volume derivatives
        ur, us = T.Dr @ u, T.Ds @ u
        vxr, vxs = T.Dr @ vx, T.Ds @ vx
        vyr, vys = T.Dr @ vy, T.Ds @ vy
        dxvy = self.rx * vyr + self.sx * vys
        dyvx = self.ry * vxr + self.sy * vxs
        dxu = self.rx * ur + self.sx * us
        dyu = self.ry * ur + self.sy * us
        # faces
        uf, vxf, vyf = u.ravel(), vx.ravel(), vy.ravel()
        uM, uP = uf[self.idxM], uf[self.idxP]
        nx, ny = self.nx, self.ny
        HtM = nx * vyf[self.idxM] - ny * vxf[self.idxM]
        HtP = nx * vyf[self.idxP] - ny * vxf[self.idxP]
        bm = self.bmask
        if self.bc == "abc":
            uP = np.where(bm, 0.0, uP)
            HtP = np.where(bm, 0.0, HtP)
        else:
            pec_like = (self.bc == "pec") == (self.pol == "TM")
            if pec_like:  # u = 0 on the wall
                uP = np.where(bm, -uM, uP)
                HtP = np.where(bm, HtM, HtP)
            else:
                uP = np.where(bm, uM, uP)
                HtP = np.where(bm, -HtM, HtP)
        dU, dH = uP - uM, HtP - HtM
        al = self.alpha
        d = (self.ZP * dH + al * dU) / (self.ZP + self.ZM)
        e = (self.YP * dU + al * dH) / (self.YP + self.YM)
        L = self._lift(np.concatenate([d, -ny * e, nx * e], 1))
        Ld, Lvx, Lvy = L[:, :K], L[:, K : 2 * K], L[:, 2 * K :]
        du[:] = (dxvy - dyvx + Ld) / self.a
        dvx[:] = (-dyu + Lvx) / self.b
        dvy[:] = (dxu + Lvy) / self.b
        # PML: complex-frequency-shifted stretched coordinates in ADE form,
        # 1/s_q = 1 - sigma_q / (s + alpha + sigma_q) applied to every weak
        # derivative d_q f  ->  d_q f - Psi, dPsi/dt = sigma_q d_q f
        # - (sigma_q + alpha) Psi (q = x, y)
        if self.pml is not None:
            pm = self.pml
            kp = pm["k"]
            Lx = self._lift_subset(nx[:, kp] ** 2 * d[:, kp], kp)
            Ly = self._lift_subset(ny[:, kp] ** 2 * d[:, kp], kp)
            F = dict(
                Qx=dxvy[:, kp] + Lx,  # d_x vy
                Qy=dyvx[:, kp] - Ly,  # d_y vx
                Rx=dxu[:, kp] + Lvy[:, kp],  # d_x u
                Ry=dyu[:, kp] - Lvx[:, kp],  # d_y u
            )
            Psi = {k: self._view(X, k) for k in F}
            du[:, kp] += (-Psi["Qx"] + Psi["Qy"]) / self.a[kp]
            dvx[:, kp] += Psi["Ry"] / self.b[kp]
            dvy[:, kp] -= Psi["Rx"] / self.b[kp]
            al = pm["alpha"]
            for k, f in F.items():
                sg = pm["sx"] if k[1] == "x" else pm["sy"]
                self._view(res, k)[:] = sg * f - (sg + al) * Psi[k]
        # sources: scattered-field plane wave (non-dispersive part)
        pw = self.plane_wave if with_source else None
        if pw is not None:
            ks = pw["ks"]
            uinc_t = self._incident(t, deriv=True)
            du[:, ks] += pw["ca"] * uinc_t
            dvx[:, ks] += pw["cb"] * pw["vdir"][0] * uinc_t
            dvy[:, ks] += pw["cb"] * pw["vdir"][1] * uinc_t
            uinc = self._incident(t) if self.disp else None
        # dispersive ADEs (act on the electric field: u for TM, v for TE)
        for i, dsp in enumerate(self.disp):
            kk = dsp["k"]
            if self.pol == "TM":
                W = u[None, :, kk]
                dW_views = [du]
            else:
                W = np.stack([vx[:, kk], vy[:, kk]])
                dW_views = [dvx, dvy]
            Wt = W.copy()
            if pw is not None:
                ui = uinc[:, dsp["kloc"]]
                if self.pol == "TM":
                    Wt = Wt + ui[None]
                else:
                    Wt = Wt + np.stack([pw["vdir"][0] * ui, pw["vdir"][1] * ui])
            nw = W.shape[0]
            shape = (nw, Np, len(kk))
            acc = -dsp["sigma"] * Wt
            for j, (S, w02, g) in enumerate(dsp["lor"]):
                Jp = self._view(X, f"J{i}_{j}", shape)
                dJ = self._view(res, f"J{i}_{j}", shape)
                dJ[:] = -g * Jp + EPS0 * S * Wt
                if w02 > 0:
                    Pp = self._view(X, f"P{i}_{j}", shape)
                    dJ[:] -= w02 * Pp
                    self._view(res, f"P{i}_{j}", shape)[:] = Jp
                acc = acc - Jp
            for j, (de, tau) in enumerate(dsp["deb"]):
                Pp = self._view(X, f"D{i}_{j}", shape)
                dP = (EPS0 * de * Wt - Pp) / tau
                self._view(res, f"D{i}_{j}", shape)[:] = dP
                acc = acc - dP
            for q, dv in enumerate(dW_views):
                dv[:, kk] += acc[q] / dsp["coef"]
        return res

    def _lift_subset(self, F, ks):
        return self.ref.LIFT @ (self.Fscale[:, ks] * F)

    # ------------------------------------------------------------ stepping
    def one_step(self):
        """Advance the state by dt (LSRK(5,4))."""
        t1 = time.perf_counter()
        X, k = self.X, self._res
        k[:] = 0.0
        dt = self.dt
        for a, b, c in zip(LSRK4["a"], LSRK4["b"], LSRK4["c"]):
            r = self._rhs(X, self.t + c * dt)
            k *= a
            k += dt * r
            X += b * k
        self.t += dt
        self.time_rhs += time.perf_counter() - t1

    def run(self, n_steps, callback=None):
        for n in range(n_steps):
            self.one_step()
            if callback is not None:
                callback(self, n)

    def reset(self):
        self.X[:] = 0.0
        self.t = 0.0

    # ------------------------------------------------------------ linear algebra
    def apply_operator(self, X):
        """A X (homogeneous right-hand side, no sources)."""
        return self._rhs(X, 0.0, with_source=False)

    def spectral_radius(self, n_iter=60, seed=0):
        """Largest |eigenvalue| of A by ARPACK (fallback: power iteration)."""
        from scipy.sparse.linalg import LinearOperator, eigs

        n = self.n_state
        op = LinearOperator((n, n), matvec=self.apply_operator, dtype=float)
        try:
            vals = eigs(op, k=1, which="LM", return_eigenvectors=False, tol=1e-3,
                        maxiter=2000, v0=np.random.default_rng(seed).standard_normal(n))  # fmt: skip
            return float(np.abs(vals).max()) * 1.02
        except Exception:
            x = np.random.default_rng(seed).standard_normal(n)
            lam = 0.0
            for _ in range(n_iter):
                y = self.apply_operator(x)
                lam = np.linalg.norm(y) / np.linalg.norm(x)
                x = y / np.linalg.norm(y)
            return 1.1 * lam

    def operator_matrix(self):
        """
        Sparse matrix of A (dX/dt = A X), assembled by probing the matrix-free
        operator with distance-2 colourings of the elements: two elements of
        one colour share no neighbour, so the columns of one probe never
        overlap in their row support.
        """
        K, Np = self.K, self.Np
        # element adjacency
        nb = self.EToE
        adj = [set(nb[k]) | {k} for k in range(K)]
        adj2 = []
        for k in range(K):
            s = set()
            for q in adj[k]:
                s |= adj[q]
            adj2.append(s)
        color = -np.ones(K, int)
        for k in range(K):
            used = {color[q] for q in adj2[k] if color[q] >= 0}
            cc = 0
            while cc in used:
                cc += 1
            color[k] = cc
        # element of every state entry
        owner = np.zeros(self.n_state, int)
        local = np.zeros(self.n_state, int)
        for name, sl in self.layout.items():
            n = sl.stop - sl.start
            if name in ("u", "vx", "vy"):
                ks = np.arange(K)
            elif name in ("Qx", "Qy", "Rx", "Ry"):
                ks = self.pml["k"]
            else:
                i = int(name[1:].split("_")[0])
                ks = self.disp[i]["k"]
            nk = len(ks)
            idx = np.arange(n)
            owner[sl] = ks[idx % nk]
            local[sl] = sl.start + (idx // nk)  # unique per (field, node row)
        rows, cols, vals = [], [], []
        group = {}
        for j in range(self.n_state):
            key = (local[j], color[owner[j]])
            group.setdefault(key, []).append(j)
        support = [np.array(sorted(adj[k])) for k in range(K)]
        state_elem = owner
        for key, js in group.items():
            x = np.zeros(self.n_state)
            x[js] = 1.0
            y = self.apply_operator(x)
            nzr = np.flatnonzero(y)
            if nzr.size == 0:
                continue
            # attribute each nonzero row to the probe column whose element
            # neighbourhood contains the row's element
            col_of_elem = {}
            for j in js:
                for q in support[state_elem[j]]:
                    col_of_elem[q] = j
            re = state_elem[nzr]
            cj = np.array([col_of_elem[q] for q in re])
            rows.append(nzr)
            cols.append(cj)
            vals.append(y[nzr])
        A = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_state, self.n_state),
        )
        return A

    # ------------------------------------------------------------ monitors
    def locate(self, points):
        """Element and reference coordinates (r, s) of points; curved
        elements are inverted by Newton iteration on the isoparametric map."""
        mesh = self.mesh
        T = self.ref
        P = np.atleast_2d(points)
        V = mesh.nodes[mesh.cells]
        out_k = np.full(len(P), -1)
        out_rs = np.zeros((len(P), 2))
        v1, v2, v3 = V[:, 0], V[:, 1], V[:, 2]
        A = np.stack([v2 - v1, v3 - v1], 2)  # (K, 2, 2)
        Ainv = np.linalg.inv(A)
        cen = np.stack([self.x.mean(0), self.y.mean(0)], 1)
        hmax = np.hypot(self.x - cen[:, 0], self.y - cen[:, 1]).max(0) * 1.5
        for i, p in enumerate(P):
            cand = np.flatnonzero(np.linalg.norm(cen - p, axis=1) <= hmax + 1e-12)
            best = (-np.inf, -1, None)
            for k in cand:
                lam = Ainv[k] @ (p - v1[k])
                rs = 2 * lam - 1
                if self.curved[k]:
                    for _ in range(30):
                        L = T.interp_matrix(rs[:1], rs[1:])[0]
                        Gr, Gs = T.grad_interp_matrices(rs[:1], rs[1:])
                        f = np.array([L @ self.x[:, k], L @ self.y[:, k]]) - p
                        Jm = np.array([[Gr[0] @ self.x[:, k], Gs[0] @ self.x[:, k]],
                                       [Gr[0] @ self.y[:, k], Gs[0] @ self.y[:, k]]])  # fmt: skip
                        step = np.linalg.solve(Jm, f)
                        rs = rs - step
                        if np.abs(step).max() < 1e-14:
                            break
                m = min(rs[0] + 1, rs[1] + 1, -(rs[0] + rs[1]))
                if m > best[0]:
                    best = (m, k, rs)
            if best[0] < -1e-8:
                raise ValueError(f"point {p} outside the mesh")
            out_k[i], out_rs[i] = best[1], best[2]
        return out_k, out_rs

    def interp_matrix(self, points):
        """Sparse (n_points, Np K) map from a field array (u.ravel()) to the
        values at points (polynomial evaluation)."""
        k, rs = self.locate(points)
        L = self.ref.interp_matrix(rs[:, 0], rs[:, 1])  # (n, Np)
        rows = np.repeat(np.arange(len(k)), self.Np)
        cols = (np.arange(self.Np)[None, :] * self.K + k[:, None]).ravel()
        return sp.csr_matrix(
            (L.ravel(), (rows, cols)), shape=(len(k), self.Np * self.K)
        )

    @property
    def u(self):
        return self._view(self.X, "u")

    def energy(self):
        """Discrete electromagnetic energy per unit length (scattered field,
        mass-matrix weighted; PML auxiliaries excluded) [J/m]."""
        T = self.ref
        u = self._view(self.X, "u")
        vx = self._view(self.X, "vx")
        vy = self._view(self.X, "vy")
        E = 0.0
        for w, f in ((self.a, u), (self.b, vx), (self.b, vy)):
            # straight elements: J constant; curved: approximate with nodal J
            E += 0.5 * np.sum(w * np.einsum("ik,ij,jk->k", f * self.J, T.M, f))
        return E
