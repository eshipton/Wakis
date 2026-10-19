# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
2D FIT leapfrog on a nested Cartesian mesh (`QuadMesh`) with hanging nodes.

Both polarizations are written for one scalar node field u and one edge flux
f (per unit length in z):

* TM: u = E_z, f_e = L mu (H x t)_z (magnetic flux through the edge),
  node mass from eps, edge Hodge from mu;
* TE: u = H_z, f_e = L eps (t x E)_z (electric flux through the edge),
  node mass from mu, edge Hodge from eps.

With the incidence matrix G of the composite mesh and the prolongation P of
the hanging nodes (u_all = P u), K = G P and

    f^{n+1/2} = f^{n-1/2} - dt K u^n,
    D (u^{n+1} - u^n) = dt K^T W f^{n+1/2},

with D = P^T m (row-sum lumping of P^T diag(m) P; m the dual-cell masses of
all nodes) and W the diagonal edge Hodge. On a uniform grid this is exactly
the Yee/FIT scheme; on the composite mesh it is the conforming Q1 finite
element scheme with trapezoidal (mass- and stiffness-) lumping and hanging
nodes, i.e. a Galerkin interface operator: the coarse-to-fine map P and the
fine-to-coarse map P^T are exact transposes. Hence

* the semi-discrete operator D^-1 K^T W K is self-adjoint and >= 0 in the
  D inner product, the leapfrog conserves
  E^n = 1/2 u^n.D u^n + 1/2 f^{n+1/2}.W f^{n-1/2} and is stable iff
  dt < 2 / sqrt(lambda_max(D^-1 K^T W K)) (computed at setup);
* every update is linear with an exact transpose (discrete adjoint).

Materials are averaged sub-pixel inside each leaf cell (sampling cut cells
with n_sub x n_sub points): arithmetic for the node masses (E_z tangential
to every interface in TM), and for the TE edge Hodge the exact rule of a
strip with D_n continuous across it and E_t continuous along it (series
across the dual edge, parallel along the primal edge).
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import c as C0
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import mu_0 as MU0
from scipy.sparse.linalg import LinearOperator, eigsh

from .shapes import Geometry

__all__ = ["SubgridFIT2D"]

ETA0 = np.sqrt(MU0 / EPS0)


class SubgridFIT2D:
    """
    TM/TE time-domain FIT on a `QuadMesh` with nested refinement.

    Parameters
    ----------
    mesh : QuadMesh
    geometry : Geometry, optional
        Shapes and materials (vacuum if None).
    polarization : {'TM', 'TE'}
    pml : dict, optional
        ``box=(xa, xb, ya, yb)`` interior (non-absorbing) region [m]; the
        Berenger split-field PML fills the rest of the mesh with a graded
        loss rate sigma(d) = sigma_max (d / T)^order, sigma_max = (order + 1)
        c ln(1 / R0) / (2 T). Keys ``R0`` (1e-8), ``order`` (3).
    cfl : float
        dt = cfl * 2 / sqrt(lambda_max).
    n_sub : int
        Sub-pixel samples per direction in cells cut by an interface.
    dt : float, optional
        Override the timestep [s] (checked against the stability limit).
    lts : int, optional
        Local time stepping ratio p (stabilized leapfrog-Chebyshev LTS of
        `wakis.unstructured.lts`): the coarse step is ~p dt_global and the
        nodes that need it take p sub-steps.
    lts_kw : dict, optional
        Extra keywords of `wakis.unstructured.lts.build_lts`.

    Notes
    -----
    Outer boundary: PEC (TM: E_z = 0 nodes removed; TE: natural boundary).
    """

    def __init__(self, mesh, geometry=None, polarization="TM", pml=None, cfl=0.95,
                 n_sub=8, dt=None, lts=None, lts_kw=None, bloch_phase=None):  # fmt: skip
        if polarization not in ("TM", "TE"):
            raise ValueError("polarization must be 'TM' or 'TE'")
        self.mesh = mesh
        self.geo = geometry if geometry is not None else Geometry([])
        self.pol = polarization
        self.n_sub = int(n_sub)
        self._sample_materials()
        self._assemble()
        self.lam_max = self._lambda_max()
        self.dt_max = 2 / np.sqrt(self.lam_max)
        self.cfl = cfl
        self.dt = cfl * self.dt_max if dt is None else float(dt)
        if self.dt >= self.dt_max:
            raise ValueError(f"dt = {self.dt:.3e} exceeds the stability limit "
                             f"{self.dt_max:.3e}")  # fmt: skip
        self.lts, self.lts_info = None, None
        if lts:
            from wakis.unstructured.lts import build_lts

            K2 = (self.K.T @ sp.diags(self.W) @ self.K).tocsr()
            kw = dict(p=int(lts))
            kw.update(lts_kw or {})
            self.lts, self.lts_info = build_lts(K2, self.D, lts * self.dt,
                                                lam=self.lam_max, **kw)  # fmt: skip
            self.dt = self.lts_info["dt"]
        self._setup_pml(pml)
        self._setup_dispersive()
        self.src = None
        self.reset()

    # ============================================================ materials
    def _sample_materials(self):
        """Per-cell quarter (node) and half-strip (edge) material integrals."""
        m, geo, ns = self.mesh, self.geo, self.n_sub
        c = m.cells
        x0, x1, y0, y1 = c["x0"], c["x1"], c["y0"], c["y1"]
        dx, dy = x1 - x0, y1 - y0
        xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        cut = geo.cut(xm, ym, 0.5 * np.hypot(dx, dy))
        mid_c = geo.material_id(xm, ym)
        eps_tab = geo.eps
        n_mat = len(eps_tab)
        nc = m.n_cells
        t = (np.arange(ns) + 0.5) / ns

        # ---- nodes: quarter cells (corner order (x0,y0),(x1,y0),(x0,y1),(x1,y1))
        qa = 0.25 * dx * dy
        q_eps = np.repeat((eps_tab[mid_c] * qa)[:, None], 4, 1)  # int eps dA
        q_frac = np.zeros((nc, 4, n_mat))  # int 1_mat dA
        q_frac[np.arange(nc), :, mid_c] = qa[:, None]
        # ---- edges: half strips (bottom, top, left, right), ns columns along
        hw = np.stack([dy / 2, dy / 2, dx / 2, dx / 2], 1)  # half widths
        Lside = np.stack([dx, dx, dy, dy], 1)
        inv_e = 1 / eps_tab[mid_c]
        R = np.repeat((hw * inv_e[:, None])[:, :, None], ns, 2)  # int dn/eps
        Q = np.repeat((hw * (1 - inv_e[:, None]))[:, :, None], ns, 2)
        s_eps = (hw * Lside) * eps_tab[mid_c][:, None]  # int eps dA
        s_frac = np.zeros((nc, 4, n_mat))
        s_frac[np.arange(nc), :, mid_c] = hw * Lside

        k = np.flatnonzero(cut)
        for chunk in np.array_split(k, max(1, k.size // 20000 + 1)):
            if chunk.size == 0:
                continue
            X0, X1, Y0, Y1 = x0[chunk], x1[chunk], y0[chunk], y1[chunk]
            DX, DY = X1 - X0, Y1 - Y0
            for q, (ox, oy) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
                px = X0[:, None, None] + DX[:, None, None] * (ox + t[:, None]) / 2
                py = Y0[:, None, None] + DY[:, None, None] * (oy + t[None, :]) / 2
                mid = geo.material_id(*np.broadcast_arrays(px, py))
                q_eps[chunk, q] = qa[chunk] * eps_tab[mid].mean((1, 2))
                for j in range(n_mat):
                    q_frac[chunk, q, j] = qa[chunk] * (mid == j).mean((1, 2))
            for s in range(4):
                if s < 2:  # x-directed side: columns along x, rows across y
                    ya = Y0 if s == 0 else Y1 - DY / 2
                    px = X0[:, None, None] + DX[:, None, None] * t[:, None]
                    py = ya[:, None, None] + (DY / 2)[:, None, None] * t[None, :]
                else:  # y-directed: columns along y, rows across x
                    xa = X0 if s == 2 else X1 - DX / 2
                    py = Y0[:, None, None] + DY[:, None, None] * t[:, None]
                    px = xa[:, None, None] + (DX / 2)[:, None, None] * t[None, :]
                mid = geo.material_id(*np.broadcast_arrays(px, py))
                e = eps_tab[mid]  # (n, col, row)
                h = hw[chunk, s][:, None]
                R[chunk, s] = h * (1 / e).mean(2)
                Q[chunk, s] = h * (1 - 1 / e).mean(2)
                s_eps[chunk, s] = hw[chunk, s] * Lside[chunk, s] * e.mean((1, 2))
                for j in range(n_mat):
                    s_frac[chunk, s, j] = (
                        hw[chunk, s] * Lside[chunk, s] * (mid == j).mean((1, 2))
                    )
        self._q = dict(area=qa, eps=q_eps, frac=q_frac)
        self._s = dict(hw=hw, R=R, Q=Q, eps=s_eps, frac=s_frac)
        self.cut_cells = int(cut.sum())

    # ============================================================= assembly
    def _assemble(self):
        m = self.mesh
        nn, ne = m.n_nodes, m.n_edges
        cn, ce = m.cell_nodes.ravel(), m.cell_edges.ravel()
        n_mat = len(self.geo.eps)
        q, s = self._q, self._s
        # node (dual-cell) integrals on all nodes
        area = np.bincount(cn, np.repeat(q["area"], 4), nn)
        eps_a = np.bincount(cn, q["eps"].ravel(), nn)
        frac_a = np.stack([np.bincount(cn, q["frac"][:, :, j].ravel(), nn)
                           for j in range(n_mat)], 1)  # fmt: skip
        # edge integrals
        Lst = np.bincount(ce, s["hw"].ravel(), ne)  # dual length L*
        L = m.edge_length
        ns = self.n_sub
        Rk = np.zeros((ne, ns))
        Qk = np.zeros((ne, ns))
        np.add.at(Rk, ce, s["R"].reshape(-1, ns))
        np.add.at(Qk, ce, s["Q"].reshape(-1, ns))
        eps_s = np.bincount(ce, s["eps"].ravel(), ne)
        frac_s = np.stack([np.bincount(ce, s["frac"][:, :, j].ravel(), ne)
                           for j in range(n_mat)], 1)  # fmt: skip

        G = m.gradient()
        P = m.P
        disp_ids = [k for k, _ in self.geo.dispersive]
        if self.pol == "TM":
            m_all = EPS0 * eps_a
            m0_all = EPS0 * area
            mp_all = {k: EPS0 * frac_a[:, k] for k in disp_ids}
            W = Lst / (MU0 * L)
            boundary = self._boundary_nodes()
            keep = ~boundary[m.free]
        else:
            m_all = MU0 * area
            m0_all = m_all
            mp_all = {}
            inv_mean = (1 / Rk).mean(1)
            W = 1 / (EPS0 * L * inv_mean)
            c_e = (Qk / Rk).sum(1) / (1 / Rk).sum(1)
            # dispersive edges: arithmetic fill-fraction mixing (as SolverFIT3D)
            fd = frac_s[:, disp_ids].sum(1) if disp_ids else np.zeros(ne)
            de = fd > 0
            minf = EPS0 * (L / Lst) * eps_s / (L * Lst)
            m0 = EPS0 * L / Lst
            W[de] = 1 / minf[de]
            c_e[de] = Lst[de] * (minf[de] - m0[de]) / minf[de]
            self._edge_disp = {k: EPS0 * (L / Lst) * frac_s[:, k] / (L * Lst)
                               for k in disp_ids}  # fmt: skip
            self.c_e = c_e
            keep = np.ones(m.free.size, bool)
        self.dof = m.free[keep]  # mesh node index of each unknown
        Pk = P[:, keep]
        K = (G @ Pk).tocsr()
        live = np.diff(K.indptr) > 0  # drop edges with no unknown (PEC walls)
        self.edge_idx = np.flatnonzero(live)
        K = K[live]
        self.K = K
        self.W = W[live]
        self.Lst = Lst[live]
        self.L = L[live]
        self.D = Pk.T @ m_all
        self.D0 = Pk.T @ m0_all
        self.Pk = Pk
        self.m_all, self.m0_all, self.mp_all = m_all, m0_all, mp_all
        if self.pol == "TE":
            self.c_e = self.c_e[live]
            self._edge_disp = {k: v[live] for k, v in self._edge_disp.items()}
        self.n_dof = int(K.shape[1])
        self.n_edges = int(K.shape[0])
        self.n_unknowns = self.n_dof + self.n_edges

    def _boundary_nodes(self):
        m = self.mesh
        x, y = m.nodes[:, 0], m.nodes[:, 1]
        b = np.isclose(x, m.xc[0]) | np.isclose(x, m.xc[-1])
        if not m.periodic_y:
            b |= np.isclose(y, m.yc[0]) | np.isclose(y, m.yc[-1])
        return b

    def _lambda_max(self, tol=1e-6):
        s = 1 / np.sqrt(self.D)
        K, W = self.K, self.W

        def mv(z):
            z = np.asarray(z).ravel()
            return s * (K.T @ (W * (K @ (s * z))))

        n = self.n_dof
        op = LinearOperator((n, n), matvec=mv, dtype=float)
        lam = eigsh(op, k=1, which="LA", tol=tol, return_eigenvectors=False)[0]
        return float(lam) * (1 + 10 * tol)

    def stiffness(self, bloch_phase=None):
        """(K^T W K, D) on the unknowns (optionally with a Bloch phase)."""
        if bloch_phase is None:
            K = self.K
        else:
            G = self.mesh.gradient(bloch_phase)
            K = (G @ self.Pk).tocsr()[self.edge_idx]
        return (K.conj().T @ sp.diags(self.W) @ K).tocsr(), self.D

    # ================================================================== PML
    def _setup_pml(self, pml):
        m = self.mesh
        xn, yn = m.nodes[self.dof, 0], m.nodes[self.dof, 1]
        em = m.edge_mid[self.edge_idx]
        edir = m.edge_dir[self.edge_idx]
        self.pml = pml
        if not pml:
            self.pml_nodes = np.array([], int)
            self.pml_edges = np.array([], int)
            return
        xa, xb, ya, yb = pml["box"]
        R0, order = pml.get("R0", 1e-8), pml.get("order", 3)

        def rate(v, lo, hi, vmin, vmax):
            T = min(lo - vmin, vmax - hi)
            smax = (order + 1) * C0 * np.log(1 / R0) / (2 * T)
            d = np.maximum(lo - v, 0) + np.maximum(v - hi, 0)
            return smax * (d / T) ** order

        sx_n = rate(xn, xa, xb, m.xc[0], m.xc[-1])
        sy_n = rate(yn, ya, yb, m.yc[0], m.yc[-1])
        sx_e = rate(em[:, 0], xa, xb, m.xc[0], m.xc[-1])
        sy_e = rate(em[:, 1], ya, yb, m.yc[0], m.yc[-1])
        dt = self.dt
        s_e = np.where(edir == 0, sx_e, sy_e)
        pe = np.flatnonzero(s_e > 0)
        pn = np.flatnonzero((sx_n > 0) | (sy_n > 0))
        self.pml_edges, self.pml_nodes = pe, pn

        def ab(sig):
            a = np.exp(-sig * dt)
            b = np.where(sig > 0, (1 - a) / np.where(sig > 0, sig, 1), dt)
            return a, b

        self._ae, self._be = ab(s_e[pe])
        self._axn, self._bxn = ab(sx_n[pn])
        self._ayn, self._byn = ab(sy_n[pn])
        if self.lts is not None:
            near = np.zeros(self.n_dof, bool)
            near[self.lts.F] = near[self.lts.C] = True
            if near[pn].any():
                raise ValueError("LTS fine nodes reach the PML")
        # x-edge part of the node update on PML nodes
        Kx = self.K[np.flatnonzero(edir == 0)]
        self._x_edges = np.flatnonzero(edir == 0)
        self._BWx_p = (sp.diags(1 / self.D[pn]) @ Kx.T.tocsr()[pn]
                       @ sp.diags(self.W[self._x_edges])).tocsr()  # fmt: skip

    # =========================================================== dispersive
    def _setup_dispersive(self):
        from wakis.dispersive import _ade_coefficients

        self.groups = []
        for k, mat in self.geo.dispersive:
            if mat.debye_poles or mat.sigma:
                raise NotImplementedError("only Lorentz/Drude poles")
            if self.pol == "TM":
                mp = self.Pk.T @ self.mp_all[k]
                idx = np.flatnonzero(mp > 0)
                ratio = mp[idx] / self.D[idx]
            else:
                mp = self._edge_disp[k]
                idx = np.flatnonzero(mp > 0)
                ratio = mp[idx]  # multiplied by W in the step
            if not mat.is_stable(self.dt, courant=self.cfl):
                raise ValueError(f"dt = {self.dt:.3e} unstable for {mat!r}")
            coef = _ade_coefficients(mat, self.dt)[0]
            self.groups.append(dict(idx=idx, ratio=ratio, coef=coef, mat=mat))
        self._disp_rows = None

    # =============================================================== source
    def add_plane_wave(self, src_time, direction=(1.0, 0.0), amplitude=1.0):
        """
        Scattered-field plane wave in vacuum.

        u_inc(r, t) = amplitude * s(t - (k.r - s_0) / c) with s_0 the smallest
        k.r over the source locations (the scatterer). TM: E_z,inc = u_inc;
        TE: H_z,inc = u_inc / eta_0... expressed so that `u` is H_z and
        E_inc = eta_0 H_z,inc (z x k).

        TM: the node equation gets -P^T[(m - m_0)(u_inc^{n+1} - u_inc^n)]
        (the discrete time derivative of the leapfrog). TE: the constitutive
        relation E_s = D_s / eps - (1 - 1/eps) E_inc integrated across each
        edge strip gives g = W f - c_e E_inc,n with c_e from the sub-pixel
        columns. Dispersive poles are driven by the total field.
        """
        k = np.asarray(direction, float)
        k = k / np.linalg.norm(k)
        m = self.mesh
        self.src = dict(s=src_time, k=k, amp=float(amplitude))
        if self.pol == "TM":
            coef = self.m_all - self.m0_all
            nodes = np.flatnonzero(np.abs(coef) > 1e-12 * np.abs(self.m0_all).max())
            disp_nodes = np.unique(np.concatenate(
                [np.flatnonzero(v > 0) for v in self.mp_all.values()] + [nodes]))  # fmt: skip
            pos_all = m.nodes[disp_nodes]
            s0 = (pos_all @ k).min() if pos_all.size else 0.0
            self.src.update(s0=s0)
            self._src_pos = m.nodes[nodes]
            S = sp.diags(1 / self.D) @ self.Pk.T[:, nodes] @ sp.diags(coef[nodes])
            self._src_S = S.tocsr()
            # incident at the unknowns driving the ADE
            self._dof_pos = m.nodes[self.dof]
        else:
            em = m.edge_mid[self.edge_idx]
            edir = m.edge_dir[self.edge_idx]
            disp_e = (np.concatenate([g["idx"] for g in self.groups])
                      if self.groups else np.array([], int))  # fmt: skip
            e = np.flatnonzero(np.abs(self.c_e) > 1e-12 * self.L.max())
            e = np.union1d(e, disp_e)
            s0 = (em[e] @ k).min() if e.size else 0.0
            self.src.update(s0=s0)
            self._src_e = e
            self._src_pos = em[e]
            self._src_kn = np.where(edir[e] == 0, k[0], k[1]) * ETA0
            self._src_c = self.c_e[e]
            self._src_B = (sp.diags(1 / self.D) @ self.K.T.tocsr()[:, e]).tocsr()
            # incident E_n * L* on dispersive edges (drives the ADE)
            self._edge_pos = em
            self._edge_kn = np.where(edir == 0, k[0], k[1]) * ETA0
        return self

    def incident(self, points, t):
        """u_inc at points (n, 2) and times t (scalar or (nt,)) -> (nt, n)."""
        s = self.src
        tau = (
            np.atleast_1d(t)[:, None] - (np.atleast_2d(points) @ s["k"] - s["s0"]) / C0
        )
        return s["amp"] * s["s"](tau)

    # =============================================================== stepping
    def reset(self):
        self.u = np.zeros(self.n_dof)
        self.u_prev = np.zeros(self.n_dof)
        self.f = np.zeros(self.n_edges)
        self.f_prev = np.zeros(self.n_edges)
        self.n = 0
        self._ux = np.zeros(self.pml_nodes.size)
        self._uy = np.zeros(self.pml_nodes.size)
        for g in self.groups:
            g["p"] = [np.zeros(g["idx"].size) for _ in g["coef"]]
            g["p_old"] = [np.zeros(g["idx"].size) for _ in g["coef"]]
            g["x_prev"] = np.zeros(g["idx"].size)  # TE: g_tot^{n-1/2}

    def _r(self, u):
        return u if self.lts is None else self.lts.apply(u)

    def one_step(self):
        """Advance u^n -> u^{n+1}, f^{n-1/2} -> f^{n+1/2}."""
        dt, n = self.dt, self.n
        u, f = self.u, self.f
        Ku = self.K @ self._r(u)
        f_new = f - dt * Ku
        pe = self.pml_edges
        if pe.size:
            f_new[pe] = self._ae * f[pe] - self._be * Ku[pe]
        g = self.W * f_new
        th = (n + 0.5) * dt
        if self.pol == "TE":
            if self.src is not None:
                e = self._src_e
                einc = self._src_kn * self.incident(self._src_pos, th)[0]
                g[e] -= self._src_c * einc
            for grp in self.groups:  # ADE on edges (half-integer levels)
                idx = grp["idx"]
                gt_old = grp["x_prev"]
                dp = np.zeros(idx.size)
                for kk, (alpha, xi, zeta) in enumerate(grp["coef"]):
                    p, po = grp["p"][kk], grp["p_old"][kk]
                    p_new = alpha * p + xi * po + zeta * gt_old
                    dp += p_new
                    grp["p_old"][kk], grp["p"][kk] = p, p_new
                g[idx] -= self.W[idx] * grp["ratio"] * dp
                gt = g[idx].copy()
                if self.src is not None:
                    gt += (self.Lst[idx] * self._edge_kn[idx]
                           * self.incident(self._edge_pos[idx], th)[0])  # fmt: skip
                grp["x_prev"] = gt
        r = self.K.T @ g
        r /= self.D
        u_new = u + dt * r
        pn = self.pml_nodes
        if pn.size:
            rx = self._BWx_p @ f_new[self._x_edges]
            ry = r[pn] - rx
            self._ux = self._axn * self._ux + self._bxn * rx
            self._uy = self._ayn * self._uy + self._byn * ry
            u_new[pn] = self._ux + self._uy
        if self.pol == "TM":
            if self.src is not None:
                ui = self.incident(self._src_pos, [n * dt, (n + 1) * dt])
                u_new -= self._src_S @ (ui[1] - ui[0])
            for grp in self.groups:  # ADE on nodes (integer levels)
                idx = grp["idx"]
                ut = u[idx]
                if self.src is not None:
                    ut = ut + self.incident(self._dof_pos[idx], n * dt)[0]
                dp = np.zeros(idx.size)
                for kk, (alpha, xi, zeta) in enumerate(grp["coef"]):
                    p, po = grp["p"][kk], grp["p_old"][kk]
                    p_new = alpha * p + xi * po + zeta * ut
                    dp += p_new - p
                    grp["p_old"][kk], grp["p"][kk] = p, p_new
                u_new[idx] -= grp["ratio"] * dp
        self.f_prev = f
        self.f = f_new
        self.u_prev = u
        self.u = u_new
        self.n = n + 1

    def energy(self):
        """
        Discrete energy 1/2 f^{n+1/2}.W f^{n+1/2} + 1/2 u^n.D u^{n+1} [J/m].

        Exactly conserved by the lossless, source-free leapfrog (any mesh,
        any dt below the limit); positive definite iff dt < `dt_max`.
        """
        return 0.5 * self.f @ (self.W * self.f) + 0.5 * self.u_prev @ (self.D * self.u)

    # ======================================================= linear (adjoint)
    def linear_step(self, u, f):
        """
        Source-free, lossless step (u^n, f^{n-1/2}) -> (u^{n+1}, f^{n+1/2})
        as a pure function (PML, ADE and sources excluded).
        """
        f1 = f - self.dt * (self.K @ self._r(u))
        u1 = u + self.dt * (self.K.T @ (self.W * f1)) / self.D
        return u1, f1

    def linear_step_T(self, ub1, fb1):
        """Exact transpose of `linear_step` (reverse-mode adjoint of a step)."""
        fb = fb1 + self.dt * self.W * (self.K @ (ub1 / self.D))
        z = self.K.T @ fb
        if self.lts is not None:
            z = self.lts.apply_T(z)
        ub = ub1 - self.dt * z
        return ub, fb

    # ============================================================== analysis
    def interpolation(self, points):
        """Sparse map from the unknowns to the field at `points`."""
        S = self.mesh.interpolation_matrix(points)
        keep = np.zeros(self.mesh.n_nodes, bool)
        keep[self.dof] = True
        col = keep[self.mesh.free]
        return S[:, col].tocsr()
