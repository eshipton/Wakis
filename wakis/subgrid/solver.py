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
                 n_sub=8, dt=None, lts=None, lts_kw=None, bloch_phase=None,
                 averaging="strip"):  # fmt: skip
        if polarization not in ("TM", "TE"):
            raise ValueError("polarization must be 'TM' or 'TE'")
        if averaging not in ("strip", "kottke"):
            raise ValueError("averaging must be 'strip' or 'kottke'")
        self.mesh = mesh
        self.geo = geometry if geometry is not None else Geometry([])
        # Check every material: a purely magnetic dispersive material has no electric poles and is not in
        # geo.dispersive, so it would otherwise be ignored silently.
        for mat in getattr(self.geo, "materials", [m for _, m in self.geo.dispersive]):
            if getattr(mat, "has_magnetic", False):
                raise NotImplementedError(
                    "SubgridFIT2D does not support dispersive permeability (mu) yet."
                )
            if getattr(mat, "is_gain", False):
                raise NotImplementedError(
                    "SubgridFIT2D does not support gain media yet."
                )
        self.pol = polarization
        self.averaging = averaging
        self.n_sub = int(n_sub)
        self.bloch_phase = (
            1.0
            if bloch_phase is None
            else (
                complex(bloch_phase)
                if np.iscomplexobj(bloch_phase)
                else float(bloch_phase)
            )
        )
        self._is_complex = bool(np.iscomplexobj(bloch_phase))
        self._dtype = complex if self._is_complex else float
        self.v_b = C0 / np.sqrt(self.geo.background_eps)  # background speed
        self.eta_b = ETA0 / np.sqrt(self.geo.background_eps)
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

            W_op = sp.diags(self.W) if self.W_off is None else (sp.diags(self.W) + self.W_off)
            K2 = (self.KH @ W_op @ self.K).tocsr()
            kw = dict(p=int(lts))
            kw.update(lts_kw or {})
            self.lts, self.lts_info = build_lts(K2, self.D, lts * self.dt,
                                                lam=self.lam_max, **kw)  # fmt: skip
            self.dt = self.lts_info["dt"]
        self._setup_pml(pml)
        self._setup_dispersive()
        self.src = None
        self.currents = []
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
        pec_ids = getattr(geo, "pec_ids", [])
        is_pec_tab = np.zeros(len(eps_tab), bool)
        if pec_ids:
            is_pec_tab[pec_ids] = True
        eb = geo.background_eps
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
        inv_e = np.where(is_pec_tab[mid_c], 0.0, 1.0 / eps_tab[mid_c])
        R = np.repeat((hw * inv_e[:, None])[:, :, None], ns, 2)  # int dn/eps
        Q = np.repeat((hw * (1 - eb * inv_e[:, None]))[:, :, None], ns, 2)
        s_eps = (hw * Lside) * eps_tab[mid_c][:, None]  # int eps dA
        s_frac = np.zeros((nc, 4, n_mat))
        s_frac[np.arange(nc), :, mid_c] = hw * Lside
        disp_ids = [j for j, _ in geo.dispersive]
        Bd = {}  # per dispersive material: int 1_mat dn per column
        for j in disp_ids:
            Bd[j] = np.repeat((hw * (mid_c == j)[:, None])[:, :, None], ns, 2)

        # ---- Kottke quadrant inverse-permittivity tensor (for TE averaging='kottke')
        use_kottke = self.pol == "TE" and self.averaging == "kottke"
        if use_kottke:
            q_eta_xx = np.repeat((inv_e / EPS0)[:, None], 4, 1)
            q_eta_yy = np.repeat((inv_e / EPS0)[:, None], 4, 1)
            q_eta_xy = np.zeros((nc, 4))

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
                eq = eps_tab[mid]
                q_eps[chunk, q] = qa[chunk] * eq.mean((1, 2))
                for j in range(n_mat):
                    q_frac[chunk, q, j] = qa[chunk] * (mid == j).mean((1, 2))
                if use_kottke:
                    inv_eq = np.where(is_pec_tab[mid], 0.0, 1.0 / eq)
                    eta_perp = inv_eq.mean((1, 2)) / EPS0
                    eta_par = np.where(
                        is_pec_tab[mid].all((1, 2)), 0.0, 1.0 / (EPS0 * eq.mean((1, 2)))
                    )
                    if ns > 1:
                        gx = np.diff(eq, axis=1).mean((1, 2)) / (DX / (2 * ns))
                        gy = np.diff(eq, axis=2).mean((1, 2)) / (DY / (2 * ns))
                        gn = np.hypot(gx, gy)
                        nx = np.where(gn > 1e-12, gx / np.where(gn > 1e-12, gn, 1.0), 1.0)
                        ny = np.where(gn > 1e-12, gy / np.where(gn > 1e-12, gn, 1.0), 0.0)
                    else:
                        nx = np.ones(chunk.size)
                        ny = np.zeros(chunk.size)
                    q_eta_xx[chunk, q] = eta_par * ny**2 + eta_perp * nx**2
                    q_eta_yy[chunk, q] = eta_par * nx**2 + eta_perp * ny**2
                    q_eta_xy[chunk, q] = (eta_perp - eta_par) * nx * ny
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
                inv_e_s = np.where(is_pec_tab[mid], 0.0, 1.0 / e)
                h = hw[chunk, s][:, None]
                R[chunk, s] = h * inv_e_s.mean(2)
                Q[chunk, s] = h * (1 - eb * inv_e_s).mean(2)
                s_eps[chunk, s] = hw[chunk, s] * Lside[chunk, s] * e.mean((1, 2))
                for j in range(n_mat):
                    s_frac[chunk, s, j] = (
                        hw[chunk, s] * Lside[chunk, s] * (mid == j).mean((1, 2))
                    )
                for j in disp_ids:
                    Bd[j][chunk, s] = h * (mid == j).mean(2)
        self._q = dict(area=qa, eps=q_eps, frac=q_frac)
        if use_kottke:
            self._q.update(eta_xx=q_eta_xx, eta_yy=q_eta_yy, eta_xy=q_eta_xy)
        self._s = dict(hw=hw, R=R, Q=Q, eps=s_eps, frac=s_frac, B=Bd)
        self.cut_cells = int(cut.sum())

    # ============================================================= assembly
    def _assemble(self):
        m = self.mesh
        nn, ne = m.n_nodes, m.n_edges
        cn, ce = m.cell_nodes.ravel(), m.cell_edges.ravel()
        n_mat = len(self.geo.eps)
        pec_ids = getattr(self.geo, "pec_ids", [])
        sigma_tab = getattr(self.geo, "sigma", np.zeros(n_mat))
        q, s = self._q, self._s
        # node (dual-cell) integrals on all nodes
        area = np.bincount(cn, np.repeat(q["area"], 4), nn)
        eps_a = np.bincount(cn, q["eps"].ravel(), nn)
        frac_a = np.stack([np.bincount(cn, q["frac"][:, :, j].ravel(), nn)
                           for j in range(n_mat)], 1)  # fmt: skip
        pec_area = frac_a[:, pec_ids].sum(1) if pec_ids else np.zeros(nn)
        # edge integrals
        Lst = np.bincount(ce, s["hw"].ravel(), ne)  # dual length L*
        L = m.edge_length
        ns = self.n_sub
        Rk = np.zeros((ne, ns))
        Qk = np.zeros((ne, ns))
        np.add.at(Rk, ce, s["R"].reshape(-1, ns))
        np.add.at(Qk, ce, s["Q"].reshape(-1, ns))

        bp = self.bloch_phase
        G = m.gradient(bp)
        P = m.P if bp == 1.0 else m.prolongation(bp)
        disp_ids = [k for k, _ in self.geo.dispersive]
        self.W_off = None
        if self.pol == "TM":
            m_all = EPS0 * eps_a
            m0_all = EPS0 * self.geo.background_eps * area
            mp_all = {k: EPS0 * frac_a[:, k] for k in disp_ids}
            W = Lst / (MU0 * L)
            boundary = self._boundary_nodes()
            if pec_ids:
                pec_pt = (
                    self.geo.pec_at_points(m.nodes[:, 0], m.nodes[:, 1])
                    if hasattr(self.geo, "pec_at_points")
                    else np.zeros(nn, bool)
                )
                boundary = boundary | pec_pt | (pec_area >= 0.5 * area)
            keep = ~boundary[m.free]
        else:
            open_area = np.maximum(area - pec_area, 0.0)
            if pec_ids:
                open_area = np.where(open_area < 0.05 * area, 0.0, open_area)
            m_all = MU0 * np.where(open_area > 0, open_area, area)
            m0_all = m_all
            mp_all = {}
            # parallel (along the edge) of series (across it) columns; the
            # dispersive metal enters with eps_inf, its poles per column
            r_min = 0.05 * (Lst[:, None] / np.max(self.geo.eps))
            inv_Rk = np.where(Rk > r_min, 1.0 / np.where(Rk > r_min, Rk, 1.0), 0.0)
            inv_mean = inv_Rk.mean(1)
            W = np.where(
                inv_mean > 0,
                1.0 / (EPS0 * L * np.where(inv_mean > 0, inv_mean, 1.0)),
                0.0,
            )
            if pec_ids and hasattr(self.geo, "pec_at_points"):
                # Thin PEC sheets/boundaries where the dual edge is inside PEC
                em = m.edge_mid
                edir = m.edge_dir
                # Check midpoint and two dual-edge offsets
                dn_x = np.where(edir == 1, 0.25 * Lst, 0.0)
                dn_y = np.where(edir == 0, 0.25 * Lst, 0.0)
                pec_dual = (
                    self.geo.pec_at_points(em[:, 0], em[:, 1])
                    & self.geo.pec_at_points(em[:, 0] + dn_x, em[:, 1] + dn_y)
                    & self.geo.pec_at_points(em[:, 0] - dn_x, em[:, 1] - dn_y)
                )
                W = np.where(pec_dual, 0.0, W)
            c_e = Lst - self.geo.background_eps * EPS0 * L * W
            self._edge_disp = {}
            for k in disp_ids:
                Bk = np.zeros((ne, ns))
                np.add.at(Bk, ce, s["B"][k].reshape(-1, ns))
                self._edge_disp[k] = (Bk, Rk)
            self.c_e = c_e
            keep = open_area[m.free] > 0
        self._keep = keep
        self.dof = m.free[keep]  # mesh node index of each unknown
        Pk = P[:, keep]
        Pk_abs = np.abs(Pk)
        K = (G @ Pk).tocsr()
        live = (np.diff(K.indptr) > 0) & (W > 0)  # drop edges with no unknown or PEC
        self.edge_idx = np.flatnonzero(live)
        K = K[live]
        self.K = K
        self.KH = K.conj().T.tocsr()
        self.W = W[live]
        self.Lst = Lst[live]
        self.L = L[live]
        self.D = np.asarray(Pk_abs.T @ m_all, dtype=float)
        self.D0 = np.asarray(Pk_abs.T @ m0_all, dtype=float)
        self.Pk = Pk
        self._Pk_abs = Pk_abs
        self.m_all, self.m0_all, self.mp_all = m_all, m0_all, mp_all

        # Assemble off-diagonal Kottke tensor Hodge on TE edges if requested
        if self.pol == "TE" and self.averaging == "kottke":
            ce_grid = m.cell_edges  # (n_cells, 4): bottom(0), top(1), left(2), right(3)
            off_r, off_c, off_v = [], [], []
            for q_idx, (ox, oy) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
                e_bt = ce_grid[:, 0 if oy == 0 else 1]
                e_lr = ce_grid[:, 2 if ox == 0 else 3]
                w_xy = -0.25 * q["eta_xy"][:, q_idx]
                nz = np.flatnonzero(np.abs(w_xy) > 1e-18)
                if nz.size:
                    off_r.append(np.concatenate([e_lr[nz], e_bt[nz]]))
                    off_c.append(np.concatenate([e_bt[nz], e_lr[nz]]))
                    off_v.append(np.concatenate([w_xy[nz], w_xy[nz]]))
            if off_r:
                W_off_all = sp.csr_matrix(
                    (
                        np.concatenate(off_v),
                        (np.concatenate(off_r), np.concatenate(off_c)),
                    ),
                    shape=(ne, ne),
                )
                self.W_off = W_off_all[live][:, live].tocsr()

        # Static conductivity damping rates
        if np.any(sigma_tab > 0):
            if self.pol == "TM":
                sig_n = np.bincount(cn, (q["frac"] @ sigma_tab).ravel(), nn)
                G_sig = np.asarray(Pk_abs.T @ sig_n, dtype=float)
                self._sigma_node_rate = G_sig / self.D
                self._sigma_edge_rate = np.zeros(int(K.shape[0]))
            else:
                sig_e = np.bincount(ce, (s["frac"] @ sigma_tab).ravel(), ne)[live]
                eps_e = np.bincount(ce, s["eps"].ravel(), ne)[live]
                self._sigma_edge_rate = sig_e / np.maximum(EPS0 * eps_e, 1e-30)
                self._sigma_node_rate = np.zeros(int(K.shape[1]))
        else:
            self._sigma_node_rate = np.zeros(int(K.shape[1]))
            self._sigma_edge_rate = np.zeros(int(K.shape[0]))

        if self.pol == "TE":
            self.c_e = self.c_e[live]
            self._edge_disp = {
                k: (B[live], R[live]) for k, (B, R) in self._edge_disp.items()
            }
        self.n_dof = int(K.shape[1])
        self.n_edges = int(K.shape[0])
        self.n_unknowns = self.n_dof + self.n_edges

    def _apply_W(self, f):
        if self.W_off is None:
            return self.W * f
        return self.W * f + self.W_off @ f

    def _boundary_nodes(self):
        m = self.mesh
        x, y = m.nodes[:, 0], m.nodes[:, 1]
        b = np.isclose(x, m.xc[0]) | np.isclose(x, m.xc[-1])
        if not m.periodic_y:
            b |= np.isclose(y, m.yc[0]) | np.isclose(y, m.yc[-1])
        return b

    def _lambda_max(self, tol=1e-6):
        s = 1 / np.sqrt(self.D)
        K, KH = self.K, self.KH

        def mv(z):
            z = np.asarray(z).ravel()
            return s * (KH @ self._apply_W(K @ (s * z)))

        n = self.n_dof
        op = LinearOperator((n, n), matvec=mv, dtype=self._dtype)
        lam = eigsh(op, k=1, which="LA", tol=tol, return_eigenvectors=False)[0]
        return float(np.real(lam)) * (1 + 10 * tol)

    def stiffness(self, bloch_phase=None):
        """(K^T W K, D) on the unknowns (optionally with a Bloch phase)."""
        if bloch_phase is None:
            K = self.K
        else:
            G = self.mesh.gradient(bloch_phase)
            P = self.mesh.prolongation(bloch_phase)[:, self._keep]
            K = (G @ P).tocsr()[self.edge_idx]
        W_op = sp.diags(self.W) if self.W_off is None else (sp.diags(self.W) + self.W_off)
        return (K.conj().T @ W_op @ K).tocsr(), self.D

    # ================================================================== PML
    def _setup_pml(self, pml):
        m = self.mesh
        xn, yn = m.nodes[self.dof, 0], m.nodes[self.dof, 1]
        em = m.edge_mid[self.edge_idx]
        edir = m.edge_dir[self.edge_idx]
        self.pml = pml
        dt = self.dt

        def ab(sig):
            x = sig * dt
            a = np.exp(-x)
            b = dt * np.where(
                x > 1e-15, -np.expm1(-x) / np.where(x > 1e-15, x, 1.0), 1.0
            )
            return a, b

        sx_n = np.zeros(self.n_dof)
        sy_n = np.zeros(self.n_dof)
        s_e = self._sigma_edge_rate.copy()
        if pml:
            xa, xb, ya, yb = pml["box"]
            R0, order = pml.get("R0", 1e-8), pml.get("order", 3)

            def rate(v, lo, hi, vmin, vmax):
                out = np.zeros_like(v, dtype=float)
                T_lo = lo - vmin
                if T_lo > 1e-15:
                    smax_lo = (order + 1) * self.v_b * np.log(1 / R0) / (2 * T_lo)
                    d_lo = np.maximum(lo - v, 0.0)
                    out += smax_lo * (d_lo / T_lo) ** order
                T_hi = vmax - hi
                if T_hi > 1e-15:
                    smax_hi = (order + 1) * self.v_b * np.log(1 / R0) / (2 * T_hi)
                    d_hi = np.maximum(v - hi, 0.0)
                    out += smax_hi * (d_hi / T_hi) ** order
                return out

            sx_n = rate(xn, xa, xb, m.xc[0], m.xc[-1])
            sy_n = rate(yn, ya, yb, m.yc[0], m.yc[-1])
            sx_e = rate(em[:, 0], xa, xb, m.xc[0], m.xc[-1])
            sy_e = rate(em[:, 1], ya, yb, m.yc[0], m.yc[-1])
            s_e = s_e + np.where(edir == 0, sx_e, sy_e)

        pe = np.flatnonzero(s_e > 0)
        pn = np.flatnonzero((sx_n > 0) | (sy_n > 0))
        self.pml_edges, self.pml_nodes = pe, pn

        # Non-PML conductive nodes in TM mode
        sig_n = self._sigma_node_rate.copy()
        if pn.size:
            sx_n[pn] += sig_n[pn]
            sy_n[pn] += sig_n[pn]
            sig_n[pn] = 0.0
        self._sig_nodes = np.flatnonzero(sig_n > 0)
        self._asn, self._bsn = ab(sig_n[self._sig_nodes])

        self._ae, self._be = ab(s_e[pe])
        self._axn, self._bxn = ab(sx_n[pn])
        self._ayn, self._byn = ab(sy_n[pn])
        self._ae_full = np.ones(self.n_edges)
        self._be_full = np.full(self.n_edges, dt)
        if pe.size:
            self._ae_full[pe] = self._ae
            self._be_full[pe] = self._be
        if self.lts is not None and pn.size:
            near = np.zeros(self.n_dof, bool)
            near[self.lts.F] = near[self.lts.C] = True
            if near[pn].any():
                raise ValueError("LTS fine nodes reach the PML")
        # x-edge part of the node update on PML nodes
        self._x_edges = np.flatnonzero(edir == 0)
        if pn.size:
            Kx = self.K[self._x_edges]
            self._BWx_p = (
                sp.diags(1 / self.D[pn])
                @ Kx.conj().T.tocsr()[pn]
                @ sp.diags(self.W[self._x_edges])
            ).tocsr()
        else:
            self._BWx_p = sp.csr_matrix((0, self._x_edges.size), dtype=self._dtype)

    # =========================================================== dispersive
    def _setup_dispersive(self):
        from wakis.dispersive import _ade_coefficients

        self.groups = []
        for k, mat in self.geo.dispersive:
            if mat.debye_poles:
                raise NotImplementedError("only Lorentz/Drude poles")
            if self.pol == "TM":
                mp = np.asarray(self._Pk_abs.T @ self.mp_all[k], dtype=float)
                idx = np.flatnonzero(mp > 0)
                ratio = mp[idx] / self.D[idx]
            else:
                Bk, Rk = self._edge_disp[k]
                idx = np.flatnonzero(Bk.max(1) > 0)
                Bk, Rk = Bk[idx], Rk[idx]
                ns = Bk.shape[1]
                einf = mat.eps_inf
                # passivity: the metal part of a column cannot exceed it
                if np.any(Bk / einf > Rk * (1 + 1e-9)):
                    raise RuntimeError("inconsistent dispersive column data")
                # g -= sum_k beta_k p_k ; E_m,k = ((g_tot + b p / einf) / R - p) / einf
                beta = (
                    (self.W[idx] * EPS0 * self.L[idx] / ns)[:, None] * Bk / (einf * Rk)
                )
                ratio = dict(beta=beta, b=Bk, R=Rk, einf=einf)
            if not mat.is_stable(self.dt, courant=self.cfl):
                raise ValueError(f"dt = {self.dt:.3e} unstable for {mat!r}")
            coef = _ade_coefficients(mat, self.dt)[0]
            self.groups.append(dict(idx=idx, ratio=ratio, coef=coef, mat=mat))
        self._disp_rows = None

    # =============================================================== source
    def add_plane_wave(self, src_time, direction=(1.0, 0.0), amplitude=1.0):
        """
        Scattered-field plane wave in the lossless background eps_b.

        u_inc(r, t) = amplitude * s(t - (k.r - s_0) / v_b), v_b = c /
        sqrt(eps_b), with s_0 the smallest k.r over the source locations (the
        scatterer). TM: E_z,inc = u_inc. TE: H_z,inc = u_inc and E_inc =
        eta_b u_inc (z x k), eta_b = eta_0 / sqrt(eps_b). After this call `u`
        is the scattered field.

        TM: the node equation gets -P^T[(m - m_b)(u_inc^{n+1} - u_inc^n)]
        (the discrete time derivative of the leapfrog; m_b the background
        mass). TE: the constitutive relation E_s = D_s / eps - (1 - eps_b /
        eps) E_inc integrated across each edge strip gives g = W f - c_e
        E_inc,n with c_e from the sub-pixel columns. Dispersive poles are
        driven by the total field.
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
            S = sp.diags(1 / self.D) @ self.Pk.conj().T[:, nodes] @ sp.diags(coef[nodes])
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
            self._src_kn = np.where(edir[e] == 0, k[0], k[1]) * self.eta_b
            self._src_c = self.c_e[e]
            self._src_B = (sp.diags(1 / self.D) @ self.KH[:, e]).tocsr()
            # incident E_n * L* on dispersive edges (drives the ADE)
            self._edge_pos = em
            self._edge_kn = np.where(edir == 0, k[0], k[1]) * self.eta_b
        self._inc_tab, self._inc_cache = {}, {}
        self._register("src", self._src_pos)
        for gi, grp in enumerate(self.groups):
            pos = self._dof_pos if self.pol == "TM" else self._edge_pos
            self._register(f"disp{gi}", pos[grp["idx"]])
        return self

    def _register(self, name, points):
        """Unique propagation delays of a point set (one pulse evaluation
        per distinct k.r: on the lattice many points share it)."""
        proj = np.atleast_2d(points) @ self.src["k"]
        u, inv = np.unique(np.round(proj, 15), return_inverse=True)
        self._inc_tab[name] = ((u - self.src["s0"]) / self.v_b, inv.ravel())
        self._inc_cache[name] = {}

    def _inc(self, name, t):
        """u_inc on a registered point set at time t (cached, 2 levels)."""
        cache = self._inc_cache[name]
        if t in cache:
            return cache[t]
        tau, inv = self._inc_tab[name]
        v = (self.src["amp"] * self.src["s"](t - tau))[inv]
        if len(cache) >= 2:
            cache.pop(next(iter(cache)))
        cache[t] = v
        return v

    def incident(self, points, t):
        """u_inc at points (n, 2) and times t (scalar or (nt,)) -> (nt, n)."""
        s = self.src
        tau = (
            np.atleast_1d(t)[:, None]
            - (np.atleast_2d(points) @ s["k"] - s["s0"]) / self.v_b
        )
        return s["amp"] * s["s"](tau)

    def add_current(self, points_or_edges, weights, waveform=None, kind="auto",
                    component=None, time_offset=0.0):  # fmt: skip
        """
        Register a soft current source entering `one_step`, `linear_step` and `linear_step_T`.

        Parameters
        ----------
        points_or_edges : array_like
            * `(n_pts, 2)` coordinates [m] when `kind='auto'` or `'points'`,
              interpolated onto `u` via `interpolation` or onto `f` via
              `edge_field_matrices`;
            * 1D integer array of unknown node indices `0 .. n_dof - 1` when
              `kind='node'` or `'u'`;
            * 1D integer array of active edge indices `0 .. n_edges - 1` when
              `kind='edge'` or `'f'`.
        weights : array_like
            Source weights matching `points_or_edges`. Nodal weights `b_u` add
            `dt * (b_u / D) * q(t + dt/2)` to `u^{n+1}`; edge weights `b_f` add
            `dt * b_f * q(t)` to `f^{n+1/2}`.
        waveform : callable, optional
            `waveform(t) -> float` time dependence [dimensionless].
        kind : {'auto', 'points', 'node', 'u', 'edge', 'f'}, optional
            Target unknown space. Default `'auto'`.
        component : {'u', 'Hz', 'Ez', 'Ex', 'Ey', 'Hx', 'Hy'}, optional
            Field component when `points_or_edges` are 2D point coordinates.
        time_offset : float, optional
            Time shift [s] added to the evaluation time of `waveform`.

        Returns
        -------
        int
            Index of the registered source in `self.currents`.
        """
        arr = np.asarray(points_or_edges)
        if kind == "auto":
            if arr.ndim == 2 and arr.shape[1] == 2 and np.issubdtype(arr.dtype, np.floating):
                kind = "points"
            elif component in ("edge", "f", "Ex", "Ey", "Hx", "Hy"):
                kind = "edge"
            else:
                kind = "node"

        w_raw = np.asarray(weights)
        w_dtype = complex if (self._is_complex or np.iscomplexobj(w_raw)) else float
        b_u, b_f = None, None
        if kind in ("node", "u"):
            idx = np.atleast_1d(np.asarray(points_or_edges, dtype=int)).ravel()
            w = np.broadcast_to(np.asarray(weights, dtype=w_dtype), idx.shape)
            b_u = np.zeros(self.n_dof, dtype=w_dtype)
            np.add.at(b_u, idx, w)
        elif kind in ("edge", "f"):
            idx = np.atleast_1d(np.asarray(points_or_edges, dtype=int)).ravel()
            w = np.broadcast_to(np.asarray(weights, dtype=w_dtype), idx.shape)
            b_f = np.zeros(self.n_edges, dtype=w_dtype)
            np.add.at(b_f, idx, w)
        elif kind == "points":
            pts = np.atleast_2d(np.asarray(points_or_edges, dtype=float))
            w_arr = np.asarray(weights, dtype=w_dtype)
            if component in ("Ex", "Ey", "Hx", "Hy") or (
                w_arr.ndim == 2 and w_arr.shape == (len(pts), 2)
            ):
                Mx, My = self.edge_field_matrices(pts)
                if component in ("Ex", "Hx"):
                    w = np.broadcast_to(w_arr, (len(pts),))
                    b_f = np.asarray(Mx.conj().T @ w, dtype=w_dtype).ravel()
                elif component in ("Ey", "Hy"):
                    w = np.broadcast_to(w_arr, (len(pts),))
                    b_f = np.asarray(My.conj().T @ w, dtype=w_dtype).ravel()
                else:
                    b_f = np.asarray(
                        Mx.conj().T @ w_arr[:, 0] + My.conj().T @ w_arr[:, 1],
                        dtype=w_dtype,
                    ).ravel()
            else:
                S = self.interpolation(pts)
                w = np.broadcast_to(w_arr, (len(pts),))
                b_u = np.asarray(S.conj().T @ w, dtype=w_dtype).ravel()
        else:
            raise ValueError(f"Unknown current kind {kind!r}.")

        self.currents.append(
            dict(
                b_u=b_u,
                b_f=b_f,
                waveform=waveform,
                time_offset=float(time_offset),
            )
        )
        return len(self.currents) - 1

    def clear_currents(self):
        """Remove all soft current sources registered with `add_current`."""
        self.currents.clear()

    # =============================================================== stepping
    def reset(self):
        dt_type = self._dtype
        self.u = np.zeros(self.n_dof, dtype=dt_type)
        self.u_prev = np.zeros(self.n_dof, dtype=dt_type)
        self.f = np.zeros(self.n_edges, dtype=dt_type)
        self.f_prev = np.zeros(self.n_edges, dtype=dt_type)
        self.n = 0
        self._ux = np.zeros(self.pml_nodes.size, dtype=dt_type)
        self._uy = np.zeros(self.pml_nodes.size, dtype=dt_type)
        for g in self.groups:
            shape = g["ratio"]["b"].shape if self.pol == "TE" else g["idx"].shape
            g["p"] = [np.zeros(shape, dtype=dt_type) for _ in g["coef"]]
            g["p_old"] = [np.zeros(shape, dtype=dt_type) for _ in g["coef"]]
            g["ptot"] = np.zeros(shape, dtype=dt_type)
            g["x_prev"] = np.zeros(g["idx"].size, dtype=dt_type)  # TE: g_tot^{n-1/2}

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
        t_n = n * dt
        for cur in self.currents:
            if cur["b_f"] is not None and cur["waveform"] is not None:
                qf = cur["waveform"](t_n + cur["time_offset"])
                if qf != 0.0:
                    f_new = f_new + (dt * qf) * cur["b_f"]
        g = self._apply_W(f_new)
        th = (n + 0.5) * dt
        if self.pol == "TE":
            if self.src is not None:
                e = self._src_e
                einc = self._src_kn * self._inc("src", th)
                g[e] -= self._src_c * einc
            for gi, grp in enumerate(self.groups):  # ADE on edges (half levels)
                idx, cf = grp["idx"], grp["ratio"]
                # metal-segment field of each column at t^{n-1/2} (series)
                pt = grp["ptot"]
                em = ((grp["x_prev"][:, None] + cf["b"] * pt / cf["einf"]) / cf["R"]
                      - pt) / cf["einf"]  # fmt: skip
                pt_new = np.zeros_like(pt)
                for kk, (alpha, xi, zeta) in enumerate(grp["coef"]):
                    p, po = grp["p"][kk], grp["p_old"][kk]
                    p_new = alpha * p + xi * po + zeta * em
                    pt_new += p_new
                    grp["p_old"][kk], grp["p"][kk] = p, p_new
                grp["ptot"] = pt_new
                g[idx] -= (cf["beta"] * pt_new).sum(1)
                gt = g[idx].copy()
                if self.src is not None:
                    gt += (self.Lst[idx] * self._edge_kn[idx]
                           * self._inc(f"disp{gi}", th))  # fmt: skip
                grp["x_prev"] = gt
        self.g = g  # dual-edge voltages at t^{n+1/2} (scattered)
        r = self.KH @ g
        for cur in self.currents:
            if cur["b_u"] is not None and cur["waveform"] is not None:
                qu = cur["waveform"](th + cur["time_offset"])
                if qu != 0.0:
                    r = r + qu * cur["b_u"]
        r /= self.D
        u_new = u + dt * r
        sn = self._sig_nodes
        if sn.size:
            u_new[sn] = self._asn * u[sn] + self._bsn * r[sn]
        pn = self.pml_nodes
        if pn.size:
            rx = self._BWx_p @ f_new[self._x_edges]
            ry = r[pn] - rx
            self._ux = self._axn * self._ux + self._bxn * rx
            self._uy = self._ayn * self._uy + self._byn * ry
            u_new[pn] = self._ux + self._uy
        if self.pol == "TM":
            if self.src is not None:
                du = self._inc("src", (n + 1) * dt) - self._inc("src", n * dt)
                u_new -= self._src_S @ du
            for gi, grp in enumerate(self.groups):  # ADE on nodes (integer levels)
                idx = grp["idx"]
                ut = u[idx]
                if self.src is not None:
                    ut = ut + self._inc(f"disp{gi}", n * dt)
                dp = np.zeros(idx.size, dtype=self._dtype)
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
        return 0.5 * float(
            np.real(
                np.vdot(self.f, self._apply_W(self.f))
                + np.vdot(self.u_prev, self.D * self.u)
            )
        )

    # ======================================================= linear (adjoint)
    def linear_step(self, u, f, q=None):
        """
        Lossless step (u^n, f^{n-1/2}[, q]) -> (u^{n+1}, f^{n+1/2})
        as a pure function (PML and ADE excluded; optional soft-source drives
        `q` for sources registered with `add_current`).
        """
        f1 = f - self.dt * (self.K @ self._r(u))
        if q is not None and self.currents:
            q_arr = np.broadcast_to(
                np.atleast_1d(np.asarray(q, dtype=self._dtype)), (len(self.currents),)
            )
            for cur, qk in zip(self.currents, q_arr):
                if cur["b_f"] is not None:
                    f1 = f1 + (self.dt * qk) * cur["b_f"]
        r = self.KH @ self._apply_W(f1)
        if q is not None and self.currents:
            for cur, qk in zip(self.currents, q_arr):
                if cur["b_u"] is not None:
                    r = r + qk * cur["b_u"]
        u1 = u + self.dt * r / self.D
        return u1, f1

    def linear_step_T(self, ub1, fb1, with_sources=False):
        """
        Exact transpose of `linear_step` (reverse-mode adjoint of a step).

        When `with_sources=True`, also returns the adjoint sensitivity `qb`
        with respect to the source drives `q` of `self.currents`.
        """
        u_scaled = ub1 / self.D
        fb = fb1 + self.dt * self._apply_W(self.K @ u_scaled)
        z = self.KH @ fb
        if self.lts is not None:
            z = self.lts.apply_T(z)
        ub = ub1 - self.dt * z
        if with_sources:
            qb = np.array(
                [
                    self.dt
                    * (
                        (np.vdot(cur["b_f"], fb) if cur["b_f"] is not None else 0.0)
                        + (np.vdot(cur["b_u"], u_scaled) if cur["b_u"] is not None else 0.0)
                    )
                    for cur in self.currents
                ],
                dtype=self._dtype,
            )
            return ub, fb, qb
        return ub, fb

    def step_pml(self, u, f, ux=None, uy=None, q=None):
        """
        Pure-function step `(u, f, ux, uy[, q]) -> (u1, f1, ux1, uy1)` including
        Berenger split-field PML, static conductivity, Kottke tensor Hodge, and
        registered soft sources `q`.
        """
        pn = self.pml_nodes
        if ux is None:
            ux = np.zeros(pn.size, dtype=self._dtype)
        if uy is None:
            uy = np.zeros(pn.size, dtype=self._dtype)
        Ku = self.K @ self._r(u)
        f1 = self._ae_full * f - self._be_full * Ku
        if q is not None and self.currents:
            q_arr = np.broadcast_to(
                np.atleast_1d(np.asarray(q, dtype=self._dtype)), (len(self.currents),)
            )
            for cur, qk in zip(self.currents, q_arr):
                if cur["b_f"] is not None:
                    f1 = f1 + (self.dt * qk) * cur["b_f"]
        g = self._apply_W(f1)
        r0 = self.KH @ g
        if q is not None and self.currents:
            for cur, qk in zip(self.currents, q_arr):
                if cur["b_u"] is not None:
                    r0 = r0 + qk * cur["b_u"]
        r = r0 / self.D
        u1 = u + self.dt * r
        sn = self._sig_nodes
        if sn.size:
            u1 = u1.copy()
            u1[sn] = self._asn * u[sn] + self._bsn * r[sn]
        if pn.size:
            rx = self._BWx_p @ f1[self._x_edges]
            ry = r[pn] - rx
            ux1 = self._axn * ux + self._bxn * rx
            uy1 = self._ayn * uy + self._byn * ry
            u1 = u1.copy()
            u1[pn] = ux1 + uy1
        else:
            ux1, uy1 = ux.copy(), uy.copy()
        return u1, f1, ux1, uy1

    def step_pml_T(self, ub1, fb1, uxb1=None, uyb1=None, with_sources=False):
        """
        Exact reverse-mode transpose of `step_pml`.

        Parameters
        ----------
        ub1, fb1, uxb1, uyb1 : ndarray
            Adjoint cotangents of `(u1, f1, ux1, uy1)`.
        with_sources : bool, optional
            If True, also return `qb` (cotangent w.r.t. `q`).
        """
        pn = self.pml_nodes
        if uxb1 is None:
            uxb1 = np.zeros(pn.size, dtype=self._dtype)
        if uyb1 is None:
            uyb1 = np.zeros(pn.size, dtype=self._dtype)
        ub_p = np.array(ub1, dtype=self._dtype, copy=True)
        fb_p = np.array(fb1, dtype=self._dtype, copy=True)
        if pn.size:
            uxb_tot = uxb1 + ub_p[pn]
            uyb_tot = uyb1 + ub_p[pn]
            ub_p[pn] = 0.0
            uxb = self._axn * uxb_tot
            uyb = self._ayn * uyb_tot
            rxb = self._bxn * uxb_tot
            ryb = self._byn * uyb_tot
            rxb = rxb - ryb
        else:
            uxb, uyb = uxb1.copy(), uyb1.copy()
        sn = self._sig_nodes
        rb = self.dt * ub_p
        ub = ub_p.copy()
        if sn.size:
            rb[sn] = self._bsn * ub_p[sn]
            ub[sn] = self._asn * ub_p[sn]
        if pn.size:
            rb[pn] = ryb
            fb_p[self._x_edges] = fb_p[self._x_edges] + (self._BWx_p.conj().T @ rxb)
        r0b = rb / self.D
        gb = self.K @ r0b
        fb_p = fb_p + self._apply_W(gb)
        fb = self._ae_full * fb_p
        z = self.KH @ (self._be_full * fb_p)
        if self.lts is not None:
            z = self.lts.apply_T(z)
        ub = ub - z
        if with_sources:
            qb = np.array(
                [
                    (self.dt * np.vdot(cur["b_f"], fb_p) if cur["b_f"] is not None else 0.0)
                    + (np.vdot(cur["b_u"], r0b) if cur["b_u"] is not None else 0.0)
                    for cur in self.currents
                ],
                dtype=self._dtype,
            )
            return ub, fb, uxb, uyb, qb
        return ub, fb, uxb, uyb

    # ============================================================== analysis
    def edge_field_matrices(self, points):
        """
        Sparse maps (Mx, My) from the dual-edge voltages `g` to the in-plane
        field at `points`: (E_x, E_y) in TE, (H_y, -H_x)... in TM, i.e. the
        field normal to each edge, E_n = g / L* with n = t x z.

        In a leaf cell the x-component comes from its left/right sides
        (y-directed edges, (t x E)_z = -E_x) and the y-component from its
        bottom/top sides, each linear across the cell (lowest-order
        edge-element reconstruction). Valid away from material interfaces.
        """
        m = self.mesh
        points = np.atleast_2d(points)
        k = m.locate(points)
        c = m.cells
        tx = (points[:, 0] - c["x0"][k]) / (c["x1"][k] - c["x0"][k])
        ty = (points[:, 1] - c["y0"][k]) / (c["y1"][k] - c["y0"][k])
        pos = -np.ones(m.n_edges, int)
        pos[self.edge_idx] = np.arange(self.n_edges)
        ce = pos[m.cell_edges[k]]  # bottom, top, left, right
        n = len(points)
        rows = np.repeat(np.arange(n), 2)
        inv = 1 / self.Lst

        def mat(cols, w):
            ok = cols >= 0
            dtype = complex if np.iscomplexobj(w) else float
            return sp.csr_matrix(
                (w[ok].astype(dtype), (rows[ok], cols[ok])), shape=(n, self.n_edges)
            )

        lr = np.stack([ce[:, 2], ce[:, 3]], 1).ravel()
        wx = -np.stack([1 - tx, tx], 1).ravel() * inv[np.maximum(lr, 0)]
        top_wrap = m.periodic_y & ((c["j"][k] + c["n"][k]) == m.NJ)
        bp = np.where(top_wrap, self.bloch_phase, 1.0)
        bt = np.stack([ce[:, 0], ce[:, 1]], 1).ravel()
        wy = np.stack([1 - ty, ty * bp], 1).ravel() * inv[np.maximum(bt, 0)]
        return mat(lr, wx), mat(bt, wy)

    def interpolation(self, points):
        """Sparse map from the unknowns to the field at `points`."""
        S = self.mesh.interpolation_matrix(points, self.bloch_phase)
        keep = np.zeros(self.mesh.n_nodes, bool)
        keep[self.dof] = True
        col = keep[self.mesh.free]
        return S[:, col].tocsr()
