# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Time-domain nonorthogonal FIT (NFIT) on body-fitted quadrilateral meshes.

Unknowns per unit length in z (as in `wakis.unstructured.UnstructuredFIT2D`):

TM (Ez on nodes, magnetic flux b on primal edges)::

    b'  = -G e
    M e' = G^T W b              M = eps_0 sum_c eps_c A_{n,c} (diagonal)

TE (Hz on nodes, electric flux d on primal edges)::

    d'  = G h
    M h' = -G^T W d             M = mu_0 sum_c mu_c A_{n,c}

G is the topological edge-node incidence (unchanged by the metric). All metric
information is in the edge Hodge W (flux -> dual-edge voltage), assembled from
*field components*: every cell supplies components q with a reconstruction
row R_q (field component from the fluxes of the cell's edges), a weight w_q
(area) and a material,

    W = R^T diag(w / p) R,       p = mu (TM) or eps (TE),

so W is symmetric positive semi-definite by construction (the discrete
magnetic/electric energy is (1/2) b^T W b). Two kinds of cells:

* general quadrilateral (straight sides): at each corner the in-plane vector
  B (or D) is reconstructed from the fluxes of the two sides meeting there
  (exact for a uniform field), weight A/4. W couples the edges of each
  corner: the banded symmetric NFIT material matrix (Schuhmann & Weiland
  2000); on rectangles it is diagonal and equal to standard FIT.
* polar annular sector (exact arcs): orthogonal, so each side contributes
  one normal component (flux / side length) on its half dual edge: diagonal,
  standard FIT in cylindrical coordinates with exact lengths and areas.

Because W (flux -> voltage) is assembled directly, the explicit leapfrog
needs no inversion of a non-diagonal mass matrix: the node mass is diagonal
(lumped dual areas) and W is sparse. Stability: dt < 2 / sqrt(lambda_max)
of M^-1 G^T W G (Lanczos).

Perfectly matched layer: complex coordinate stretching in polar (s_r, s_phi
= r~/r) or Cartesian (s_x, s_y) coordinates. On orthogonal (diagonal-Hodge)
cells the stretching only rescales lengths: a node mass gets s_a s_b, an edge
Hodge entry s_perp / s_par (s_par: stretching along the edge), giving local
auxiliary-differential updates. PML cells must be polar (radial PML) or
rectangles (Cartesian PML).
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0
from scipy.sparse.linalg import eigsh

from ..dispersive import _ade_coefficients
from ..unstructured._core import (
    CapacitiveUpdate,
    cell_materials,
    check_dispersive_stability,
    max_eigenvalue,
)

__all__ = ["CurvilinearFIT2D", "hodge_components"]


def hodge_components(mesh):
    """
    Field components of the edge Hodge.

    Returns
    -------
    dict with
        R : csr (n_comp, n_edges) -- component = R @ flux (per unit length:
            B or D [flux / m]);
        w : (n_comp,) weights [m^2];
        cell : (n_comp,) owning cell;
        point : (n_comp, 2) quadrature point [m];
        direction : (n_comp, 2) unit vector of the component;
        node_share : csr (n_nodes, n_cells) -- area of cell c attributed to
            node n [m^2] (lumped dual areas).
    """
    nodes, cells, ce = mesh.nodes, mesh.cells, mesh.cell_edges
    t = nodes[mesh.edges[:, 1]] - nodes[mesh.edges[:, 0]]
    N = np.stack([t[:, 1], -t[:, 0]], 1)  # length-weighted normal, flux sign
    Lc = np.linalg.norm(t, axis=1)
    Nhat = N / Lc[:, None]
    rows, cols, vals = [], [], []
    w, cell, point, direc = [], [], [], []
    nrow = 0
    # ---------------- general quads: corner reconstruction
    gq = np.flatnonzero(~mesh.is_polar)
    if gq.size:
        P = nodes[cells[gq]]
        A = mesh.cell_area[gq]
        cen = P.mean(1)
        for k in range(4):
            e1, e2 = ce[gq, (k - 1) % 4], ce[gq, k]
            Minv = np.linalg.inv(np.stack([N[e1], N[e2]], 1))  # B = Minv [f1, f2]
            m_prev = 0.5 * (P[:, k] + P[:, (k - 1) % 4])
            m_next = 0.5 * (P[:, k] + P[:, (k + 1) % 4])
            q = 0.25 * (P[:, k] + m_prev + m_next + cen)
            for comp in range(2):
                r = nrow + np.arange(gq.size)
                rows += [r, r]
                cols += [e1, e2]
                vals += [Minv[:, comp, 0], Minv[:, comp, 1]]
                w.append(A / 4)
                cell.append(gq)
                point.append(q)
                d = np.zeros((gq.size, 2))
                d[:, comp] = 1
                direc.append(d)
                nrow += gq.size
    # ---------------- polar sectors: diagonal (one component per side)
    pc = np.flatnonzero(mesh.is_polar)
    if pc.size:
        r0, r1, t0, t1 = mesh.polar[pc].T
        rm, dth = 0.5 * (r0 + r1), t1 - t0
        tm = 0.5 * (t0 + t1)
        # side 0: radial at t0; 1: arc at r1; 2: radial at t1; 3: arc at r0
        L = [r1 - r0, r1 * dth, r1 - r0, r0 * dth]
        h = [rm * dth / 2, (r1 - r0) / 2, rm * dth / 2, (r1 - r0) / 2]
        pts = [
            (rm, t0 + dth / 4), (r1 - (r1 - r0) / 4, tm),
            (rm, t1 - dth / 4), (r0 + (r1 - r0) / 4, tm),
        ]  # fmt: skip
        for k in range(4):
            e = ce[pc, k]
            r = nrow + np.arange(pc.size)
            rows.append(r)
            cols.append(e)
            vals.append(1.0 / L[k])
            w.append(L[k] * h[k])
            cell.append(pc)
            rr, tt = pts[k]
            point.append(np.stack([rr * np.cos(tt), rr * np.sin(tt)], 1)
                         + mesh.polar_center[pc])
            direc.append(Nhat[e])
            nrow += pc.size
    R = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nrow, mesh.n_edges),
    )
    # ---------------- lumped node areas
    nr, nc_, nv = [], [], []
    if gq.size:
        for k in range(4):
            nr.append(cells[gq, k])
            nc_.append(gq)
            nv.append(mesh.cell_area[gq] / 4)
    if pc.size:
        quarter = [
            0.5 * (rm**2 - r0**2) * dth / 2,  # (r0, t0)
            0.5 * (r1**2 - rm**2) * dth / 2,  # (r1, t0)
            0.5 * (r1**2 - rm**2) * dth / 2,  # (r1, t1)
            0.5 * (rm**2 - r0**2) * dth / 2,  # (r0, t1)
        ]
        for k in range(4):
            nr.append(cells[pc, k])
            nc_.append(pc)
            nv.append(quarter[k])
    share = sp.csr_matrix(
        (np.concatenate(nv), (np.concatenate(nr), np.concatenate(nc_))),
        shape=(mesh.n_nodes, mesh.n_cells),
    )
    return dict(
        R=R, w=np.concatenate(w), cell=np.concatenate(cell),
        point=np.concatenate(point), direction=np.concatenate(direc),
        node_share=share,
    )  # fmt: skip


def _pml_profile(pml):
    """sigma(xi) [1/s] and its integral over the layer coordinate."""
    d = pml["r1"] - pml["r0"]
    m = pml.get("order", 3)
    smax = pml.get("sigma_max")
    if smax is None:
        R = pml.get("reflection", 1e-6)
        v = pml.get("v", 1 / np.sqrt(eps_0 * mu_0))
        smax = (m + 1) * v * np.log(1 / R) / (2 * d)

    def sig(x):
        xi = np.clip((x - pml["r0"]) / d, 0, 1)
        return smax * xi**m

    def integral(x):  # int_{r0}^x sigma
        xi = np.clip((x - pml["r0"]) / d, 0, None)
        xi1 = np.minimum(xi, 1)
        return smax * d * xi1 ** (m + 1) / (m + 1) + smax * d * (xi - xi1)

    return sig, integral, smax


class CurvilinearFIT2D:
    """
    Leapfrog FIT on a body-fitted quadrilateral `QuadMesh`.

    Parameters
    ----------
    mesh : QuadMesh
    materials : dict, optional
        ``{tag: material}``, as in `UnstructuredFIT2D`: ``[eps_r, mu_r]``,
        ``[eps_r, mu_r, sigma]`` (TM), `DispersiveMaterial` or ``"pec"``.
    polarization : {"TM", "TE"}
    bc : {"pec", "pmc"}
        Outer boundary (behind the PML if any).
    pml : dict, optional
        ``dict(kind="radial", r0=..., r1=...)`` (complex stretching of r,
        with the curvature factor s_phi = r~/r) or ``dict(kind="cartesian",
        r0=..., r1=...)`` (|x| and |y| beyond r0). Optional ``order``
        (default 3) and ``sigma_max`` [1/s] or ``reflection`` (normal-incidence
        target, default 1e-6); ``curvature=False`` drops the s_phi = r~/r
        factor of the radial PML (a mismatched layer, for tests).
    dt : float, optional
    cfl : float
        Fraction of the leapfrog stability limit.
    W_override : sparse matrix, optional
        Replace the assembled edge Hodge (for mutation tests).
    """

    def __init__(self, mesh, materials=None, polarization="TM", bc="pec", pml=None,
                 dt=None, cfl=0.95, W_override=None):  # fmt: skip
        if polarization not in ("TM", "TE"):
            raise ValueError("polarization must be 'TM' or 'TE'")
        self.mesh, self.pol, self.bc = mesh, polarization, bc
        eps_r, mu_r, sigma, pec, dispersive = cell_materials(mesh, materials)
        self.eps_r, self.mu_r, self.sigma, self.pec_cells = eps_r, mu_r, sigma, pec
        self.dispersive = dispersive
        if polarization == "TE" and np.any(sigma > 0):
            raise NotImplementedError("conductivity in TE: use a Drude pole")
        hc = hodge_components(mesh)
        self.hc = hc
        live = ~pec
        disp_cell = np.zeros(mesh.n_cells, bool)
        for _, sel in dispersive:
            disp_cell |= sel
        share = hc["node_share"]

        # ---------------- fixed nodes
        fixed = np.zeros(mesh.n_nodes, bool)
        used = np.zeros(mesh.n_nodes, bool)
        used[mesh.cells[live].ravel()] = True
        if polarization == "TM":
            fixed[mesh.cells[pec].ravel()] = True
            if bc == "pec":
                fixed[mesh.boundary_nodes] = True
        elif bc == "pmc":
            fixed[mesh.boundary_nodes] = True
        fixed |= ~used
        self.free = np.flatnonzero(~fixed)
        self.G = mesh.G[:, self.free].tocsr()

        # ---------------- Hodge and masses
        cc = hc["cell"]
        if polarization == "TM":
            pcoef = 1.0 / (mu_0 * mu_r[cc])
            keep = live[cc]
            mass = eps_0 * (share @ (eps_r * live))
            loss = share @ (sigma * live)
        else:
            pcoef = 1.0 / (eps_0 * eps_r[cc])
            keep = live[cc] & ~disp_cell[cc]
            mass = mu_0 * (share @ (mu_r * live))
            loss = np.zeros(mesh.n_nodes)
        R = hc["R"]
        wq = hc["w"] * pcoef * keep
        self.W = (R.T @ sp.diags(wq) @ R).tocsr()
        if W_override is not None:
            self.W = sp.csr_matrix(W_override)
        # instantaneous (eps_inf) Hodge incl. dispersive TE cells: stability
        W_inst = self.W
        if polarization == "TE" and disp_cell.any():
            W_inst = (R.T @ sp.diags(hc["w"] * pcoef * live[cc]) @ R).tocsr()
        self.mass = mass[self.free]
        if np.any(self.mass <= 0):
            raise ValueError("non-positive node mass")

        # ---------------- PML
        self._pml = None
        sa = np.zeros(mesh.n_nodes)
        sb = np.zeros(mesh.n_nodes)
        if pml is not None:
            sa, sb = self._setup_pml(pml)
        self._psi_coef = (self.mass * sa[self.free] * sb[self.free])
        node_cond = loss[self.free] + self.mass * (sa + sb)[self.free]
        self._psi_idx = np.flatnonzero(self._psi_coef)

        # ---------------- stability
        self.K = (self.G.T @ W_inst @ self.G).tocsr()
        lam = max_eigenvalue(self.K, M_inv_sqrt=1 / np.sqrt(self.mass))
        self.lam_max = lam
        self.dt_max = 2 / np.sqrt(lam)
        self.dt = cfl * self.dt_max if dt is None else float(dt)
        self.courant = self.dt * np.sqrt(lam) / 2
        if dispersive:
            check_dispersive_stability([m for m, _ in dispersive], self.dt, self.courant)

        # ---------------- node update (with dispersion for TM)
        groups = []
        if polarization == "TM":
            for mat, sel in dispersive:
                c = eps_0 * (share @ (sel & live).astype(float))
                idx = np.flatnonzero(c[self.free] > 0)
                groups.append((mat, idx, c[self.free][idx]))
        self._update = CapacitiveUpdate(self.mass, node_cond, groups, self.dt)

        # ---------------- TE dispersive components (series ADE per component)
        self._te_groups = []
        if polarization == "TE":
            for mat, sel in dispersive:
                q = np.flatnonzero(sel[cc] & live[cc])
                lorentz, debye = _ade_coefficients(mat, self.dt)
                z = np.zeros(q.size)
                self._te_groups.append(dict(
                    q=q, R=R[q], RT=(R[q].T @ sp.diags(hc["w"][q])).tocsr(),
                    eps_inf=mat.eps_inf, beta=sum(b for _, b in debye), E=z.copy(),
                    lorentz=[(*cf, z.copy(), z.copy()) for cf in lorentz],
                    debye=[(*cf, z.copy()) for cf in debye],
                ))  # fmt: skip
        self.plane_wave = None
        self.sources = []
        self.reset()

    # ------------------------------------------------------------------ PML
    def _setup_pml(self, pml):
        mesh = self.mesh
        sig, integ, smax = _pml_profile(pml)
        self.pml_sigma_max = smax
        kind = pml.get("kind", "radial")
        X = mesh.nodes
        t = X[mesh.edges[:, 1]] - X[mesh.edges[:, 0]]
        mid = 0.5 * (X[mesh.edges[:, 1]] + X[mesh.edges[:, 0]])
        if kind == "radial":
            # polar edges: arc midpoint for azimuthal edges
            r_n = np.hypot(*X.T)
            curv = 1.0 if pml.get("curvature", True) else 0.0

            def sbar(r):  # s_phi = r~/r = 1 + sbar/(i w)
                return curv * integ(r) / np.maximum(r, 1e-300)

            sa, sb = sig(r_n), sbar(r_n)
            rmid = np.hypot(*mid.T)
            radial = np.abs(np.einsum("ij,ij->i", t, mid)) > 0.9 * np.linalg.norm(
                t, axis=1
            ) * rmid
            r_e = np.where(radial, rmid, np.hypot(*X[mesh.edges[:, 0]].T))
            s_par = np.where(radial, sig(r_e), sbar(r_e))
            s_perp = np.where(radial, sbar(r_e), sig(r_e))
        elif kind == "cartesian":
            sa, sb = sig(np.abs(X[:, 0])), sig(np.abs(X[:, 1]))
            xdir = np.abs(t[:, 0]) > np.abs(t[:, 1])
            sx, sy = sig(np.abs(mid[:, 0])), sig(np.abs(mid[:, 1]))
            s_par = np.where(xdir, sx, sy)
            s_perp = np.where(xdir, sy, sx)
        else:
            raise ValueError("pml kind must be 'radial' or 'cartesian'")
        tiny = 1e-9 * smax  # round-off at the layer's inner surface
        s_par = np.where(s_par > tiny, s_par, 0.0)
        s_perp = np.where(s_perp > tiny, s_perp, 0.0)
        sa = np.where(sa > tiny, sa, 0.0)
        sb = np.where(sb > tiny, sb, 0.0)
        e = np.flatnonzero((s_par > 0) | (s_perp > 0))
        Wc = self.W[e].tocoo()
        if np.any(Wc.col != e[Wc.row]):
            raise ValueError("PML edges must have a diagonal Hodge (polar or rectangular cells)")
        c = np.asarray(self.W[e, e]).ravel()
        self._pml = dict(e=e, c=c, s_par=s_par[e], s_perp=s_perp[e])
        return sa, sb

    # ------------------------------------------------------------------ state
    def reset(self):
        self.u = np.zeros(self.free.size)
        self.u_old = np.zeros(self.free.size)
        self.v = np.zeros(self.mesh.n_edges)
        self._d_prev = self.v
        self.psi = np.zeros(self.free.size)
        self.n = 0
        self._update.reset()
        if self._pml is not None:
            self._pml["h"] = np.zeros(self._pml["e"].size)
        for g in self._te_groups:
            g["E"][:] = 0
            g["lorentz"] = [(a, x, z, 0 * p, 0 * q) for a, x, z, p, q in g["lorentz"]]
            g["debye"] = [(k, b, 0 * p) for k, b, p in g["debye"]]
        self.volt = np.zeros(self.mesh.n_edges)

    @property
    def node_field(self):
        """Ez (TM) or Hz (TE) on all nodes (0 on fixed nodes)."""
        out = np.zeros(self.mesh.n_nodes)
        out[self.free] = self.u
        return out

    @property
    def n_unknowns(self):
        """Free node values + edge fluxes."""
        return int(self.free.size + self.mesh.n_edges)

    def energy(self):
        """
        Discrete energy per unit length [J/m] that the leapfrog conserves
        exactly in a lossless, non-dispersive, source-free domain:

        TM: (1/2) b^{n+1/2} W b^{n+1/2} + (1/2) e^{n+1} M e^n;
        TE: (1/2) d^n W d^n + (1/2) h^{n+1/2} M h^{n-1/2}

        (latest step n -> n+1). Positive for dt below the stability limit.
        """
        if self.pol == "TM":
            return 0.5 * self.v @ (self.W @ self.v) + 0.5 * self.mass @ (self.u * self.u_old)
        d = self._d_prev
        return 0.5 * d @ (self.W @ d) + 0.5 * self.mass @ (self.u * self.u_old)

    # ------------------------------------------------------------------ sources
    def add_point_source(self, position, amp_func):
        """Soft node source: current amp_func(t) [A] (TM) at the nearest free node."""
        d = np.hypot(*(self.mesh.nodes[self.free] - np.asarray(position)).T)
        w = np.zeros(self.free.size)
        w[np.argmin(d)] = 1.0
        self.sources.append((w, amp_func))

    def add_plane_wave(self, src_time, direction=(1.0, 0.0), amplitude=1.0,
                       background_eps_r=1.0):  # fmt: skip
        """
        Plane wave in the scattered-field formulation (see
        `UnstructuredFIT2D.add_plane_wave`). The solver then advances the
        scattered field; total = scattered + `incident_field`.

        The incident field enters through the constitutive relation of the
        non-background cells: with D_s = D - eps_0 eps_b E_inc,

        * TM: the node update runs on the total field e_s + E_inc at nodes
          touching the scatterer (D_s' = curl H_s with the full, possibly
          dispersive, capacitance), e_s = e_tot - E_inc;
        * TE: per Hodge component q in the scatterer, E_tot,q follows from
          D_tot,q = D_s,q + eps_0 eps_b E_inc . n_q (directly, or through
          the ADE for dispersive cells) and the dual-edge voltage uses
          E_s,q = E_tot,q - E_inc . n_q.

        Lossless background, non-magnetic scatterers.
        """
        mesh = self.mesh
        eps_b = float(background_eps_r)
        if np.any(self.pec_cells):
            raise NotImplementedError("plane wave with PEC cells")
        if np.any(self.mu_r != 1):
            raise NotImplementedError("plane wave: non-magnetic media only")
        k = np.asarray(direction, float)
        k = k / np.linalg.norm(k)
        disp_cell = np.zeros(mesh.n_cells, bool)
        for _, sel in self.dispersive:
            disp_cell |= sel
        scat = (~np.isclose(self.eps_r, eps_b)) | disp_cell | (self.sigma > 0)
        on_b = np.zeros(mesh.n_nodes, bool)
        on_b[mesh.boundary_nodes] = True
        if np.any(on_b[mesh.cells[scat].ravel()]):
            raise ValueError("the scatterer must not touch the outer boundary")
        s0 = float((mesh.nodes[mesh.cells[scat].ravel()] @ k).min()) if scat.any() else 0.0
        pw = dict(s=src_time, k=k, amp=float(amplitude), eps_b=eps_b, s0=s0,
                  v=1 / np.sqrt(mu_0 * eps_0 * eps_b), eta=np.sqrt(mu_0 / (eps_0 * eps_b)))  # fmt: skip
        if self.pol == "TM":
            on = np.zeros(mesh.n_nodes, bool)
            on[mesh.cells[scat].ravel()] = True
            idx = np.flatnonzero(on[self.free])
            area = np.asarray(self.hc["node_share"].sum(1)).ravel()[self.free[idx]]
            pw.update(idx=idx, cb=eps_0 * eps_b * area,
                      pos=mesh.nodes[self.free[idx]] @ k)  # fmt: skip
        else:
            hc = self.hc
            p = np.array([-k[1], k[0]])  # E_inc direction z x k
            q = np.flatnonzero(scat[hc["cell"]] & ~disp_cell[hc["cell"]])
            coef = -hc["w"][q] * (1 - eps_b / self.eps_r[hc["cell"][q]]) * (hc["direction"][q] @ p)
            pw.update(RT=hc["R"][q].T.tocsr(), coef=coef, pos=hc["point"][q] @ k)
            for g in self._te_groups:
                g["pw_proj"] = hc["direction"][g["q"]] @ p
                g["pw_pos"] = hc["point"][g["q"]] @ k
        self.plane_wave = pw

    def _sig(self, pos, t):
        pw = self.plane_wave
        return pw["amp"] * np.asarray(pw["s"](t - (pos - pw["s0"]) / pw["v"]))

    def incident_field(self, points, t):
        """Incident Ez (TM) [V/m] or Hz (TE) [A/m] at `points` and times `t`."""
        pw = self.plane_wave
        e = self._sig(np.atleast_2d(points) @ pw["k"], t)
        return e if self.pol == "TM" else e / pw["eta"]

    # ------------------------------------------------------------------ voltages
    def _voltage(self, d, t):
        """Dual-edge voltages W(d) incl. dispersive and incident parts (TE),
        or the magnetic voltages (TM, no extras)."""
        vol = self.W @ d
        pw = self.plane_wave
        if self.pol == "TE":
            if pw is not None:
                vol += pw["RT"] @ (pw["coef"] * self._sig(pw["pos"], t))
            for g in self._te_groups:
                Dn = (g["R"] @ d) / eps_0
                einc = 0.0
                if pw is not None:
                    einc = g["pw_proj"] * self._sig(g["pw_pos"], t)
                    Dn = Dn + pw["eps_b"] * einc
                En = g["E"]
                psum = np.zeros(En.size)
                for j, (alpha, xi, zeta, p, p_old) in enumerate(g["lorentz"]):
                    p_new = alpha * p + xi * p_old + zeta * En
                    psum += p_new
                    g["lorentz"][j] = (alpha, xi, zeta, p_new, p)
                for kappa, beta, p in g["debye"]:
                    psum += kappa * p + beta * En
                E_new = (Dn - psum) / (g["eps_inf"] + g["beta"])
                for j, (kappa, beta, p) in enumerate(g["debye"]):
                    g["debye"][j] = (kappa, beta, kappa * p + beta * (E_new + En))
                g["E"] = E_new
                vol += g["RT"] @ (E_new - einc)
        if self._pml is not None:
            P = self._pml
            e = P["e"]
            dt = self.dt
            a, b = 0.5 * dt * P["s_par"], 0.5 * dt * P["s_perp"]
            P["h"] = ((1 - a) * P["h"] + P["c"] * ((1 + b) * d[e] - (1 - b) * self._d_prev[e])) / (1 + a)
            vol[e] = P["h"]
        return vol

    # ------------------------------------------------------------------ stepping
    def one_step(self):
        """Advance one leapfrog step."""
        dt, n = self.dt, self.n
        src = np.zeros(self.free.size)
        for w, f in self.sources:
            src += w * f((n + 0.5) * dt)
        pw = self.plane_wave
        if self.pol == "TM":
            self.u_old = self.u
            self._d_prev = self.v
            self.v = self.v - dt * (self.G @ self.u)
            vol = self._voltage(self.v, (n + 0.5) * dt)
            r = self.G.T @ vol - src
            if self._psi_idx.size:
                self.psi[self._psi_idx] += dt * self.u[self._psi_idx]
                r -= self._psi_coef * self.psi
            if pw is None:
                self.u = self._update.step(self.u, r)
            else:
                i = pw["idx"]
                e0 = self._sig(pw["pos"], n * dt)
                e1 = self._sig(pw["pos"], (n + 1) * dt)
                r[i] += pw["cb"] * (e1 - e0) / dt
                u = self.u.copy()
                u[i] += e0
                u = self._update.step(u, r)
                u[i] -= e1
                self.u = u
        else:
            if n == 0:
                self._d_prev = self.v
                self.volt = self._voltage(self.v, 0.0)
            r = -(self.G.T @ self.volt) - src
            if self._psi_idx.size:
                self.psi[self._psi_idx] += dt * self.u[self._psi_idx]
                r -= self._psi_coef * self.psi
            self.u_old = self.u
            self.u = self._update.step(self.u, r)
            self._d_prev = self.v
            self.v = self.v + dt * (self.G @ self.u)
            self.volt = self._voltage(self.v, (n + 1) * dt)
        self.n += 1

    def run(self, n_steps):
        for _ in range(int(n_steps)):
            self.one_step()

    # ------------------------------------------------------------------ modes
    def eigenmodes(self, n_modes=6, k_target=0.0):
        """
        Lowest resonant wavenumbers k = omega / c [1/m] (lossless, no PML)
        from G^T W G x = omega^2 M x; the zero-frequency TE mode is dropped.
        """
        M = sp.diags(self.mass)
        shift = (k_target / np.sqrt(eps_0 * mu_0)) ** 2 if k_target else -1e-3 * (
            self.K.diagonal().mean() / self.mass.mean())  # fmt: skip
        w2 = eigsh(self.K, k=n_modes + 1, M=M, sigma=shift, which="LM",
                   return_eigenvectors=False)  # fmt: skip
        w2 = np.sort(w2[w2 > 1e-8 * np.abs(w2).max()])
        return np.sqrt(w2[:n_modes] * eps_0 * mu_0)
