# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
2D FIT on general polygonal meshes (`PolyMesh`): TM (Ez on nodes) and TE
(Hz on nodes), with a diagonal (orthogonal-dual) or mimetic Hodge, a
scattered-field plane-wave source, dispersive media and a split-field PML on
axis-aligned (rectangular) regions.

Per unit length in z, with G the edge-node incidence:

TM (Ez, Hx, Hy)::

    b' = -G ez                     b_e: magnetic flux through primal edge e
    M_eps ez' = G^T S_nu b - i     S_nu: edge Hodge with 1 / (mu_0 mu_C)

TE (Hz, Ex, Ey)::

    d' = G hz - i_e                d_e: electric flux through primal edge e
    M_mu hz' = -G^T S_eps d - k    S_eps: edge Hodge with 1 / (eps_0 eps_C)

M_eps / M_mu are diagonal node masses (eps or mu integrated over the dual
cell of each node). Only the node mass is inverted, so the edge Hodge S may
be any symmetric positive (semi)definite matrix:

* ``hodge='orthogonal'``: S = diag(sum_C w_C h_{e,C} / L_e) for orthogonal
  primal/dual pairs (Delaunay-Voronoi, rectangles, hybrids of the two,
  Voronoi polygons with their generators);
* ``hodge='mfd'``: S = sum_C w_C M_C with the mimetic local matrices of
  `PolyMesh.mfd_local_matrices` (any convex polygons, non-diagonal).

The leapfrog is stable for dt < 2 / sqrt(lambda_max(M^-1 G^T S G)) and
conserves the discrete energy (u^T M u + v^T S v) / 2 (lossless case).
See the physics guide, "Polygonal FIT".
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import c as c_0
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0
from scipy.sparse.linalg import eigsh

from ..dispersive import _ade_coefficients
from ._core import (
    ETA_0,
    CapacitiveUpdate,
    Recorder,
    cell_materials,
    check_dispersive_stability,
    max_eigenvalue,
)

__all__ = ["PolygonFIT2D", "pml_rate"]


def pml_rate(inner, thickness, s_max=None, order=3, R0=1e-6, eps_r=1.0):
    """
    Graded loss rate s(q) [1/s] of a planar PML along one axis.

    s(q) = s_max (dist / thickness)^order with dist the distance outside the
    interval `inner` = (q_min, q_max). The default s_max gives a normal-
    incidence round-trip reflection R0 of the continuous layer:
    s_max = (order + 1) v ln(1 / R0) / (2 thickness), v = c / sqrt(eps_r).
    """
    v = c_0 / np.sqrt(eps_r)
    if s_max is None:
        s_max = (order + 1) * v * np.log(1 / R0) / (2 * thickness)
    lo, hi = inner

    def rate(q):
        d = np.maximum(np.maximum(lo - q, q - hi), 0.0)
        return s_max * np.clip(d / thickness, 0, 1) ** order

    return rate


class PolygonFIT2D:
    """
    Time-domain FIT solver on a polygonal 2D mesh.

    Parameters
    ----------
    mesh : PolyMesh
    materials : dict, optional
        ``{tag: material}``: ``[eps_r, mu_r]``, ``[eps_r, mu_r, sigma]`` (TM
        only for sigma), a `wakis.dispersive.DispersiveMaterial`, or ``"pec"``.
    polarization : {"TM", "TE"}
    bc : {"pec", "pmc", "abc"}
        Outer boundary; 'abc' is the first-order Silver-Mueller condition.
    hodge : {"auto", "orthogonal", "mfd"}
        Edge Hodge (see the module docstring); 'auto' uses 'orthogonal' when
        the mesh is orthogonal (max defect < `ortho_tol`), else 'mfd'.
    pml : dict, optional
        Split-field (Berenger) PML on axis-aligned edges,
        ``dict(x=(x_min, x_max), y=(y_min, y_max), thickness=d, order=3,
        R0=1e-6, eps_r=1)`` (eps_r: background permittivity for s_max): inside the box the field is unchanged; outside it the
        rates s_x(x), s_y(y) of `pml_rate` act on the x- and y-parts of the
        node field and on the fluxes of x- and y-directed edges. All edges
        touching a lossy node must be axis-aligned (true on rectangles).
        Combine with ``bc='pec'``.
    dt, cfl : float, optional
        Timestep [s] (default ``cfl`` times the leapfrog limit).
    ortho_tol : float
        Orthogonality tolerance of ``hodge='orthogonal'``.
    prolongation : sparse matrix (n_nodes, n_nodes), optional
        Hanging-node prolongation ``P`` (identity on ordinary nodes, zero
        column and master interpolation weights on hanging rows).
    lts : int or dict, optional
        Two-level stabilized leapfrog-Chebyshev local time stepping (`build_lts`).
    """

    def __init__(
        self,
        mesh,
        materials=None,
        polarization="TM",
        bc="pec",
        hodge="auto",
        pml=None,
        dt=None,
        cfl=0.95,
        ortho_tol=1e-6,
        prolongation=None,
        lts=None,
    ):
        if polarization not in ("TM", "TE"):
            raise ValueError("polarization must be 'TM' or 'TE'")
        if bc not in ("pec", "pmc", "abc"):
            raise ValueError("bc must be 'pec', 'pmc' or 'abc'")
        if prolongation is not None:
            P = sp.csr_matrix(prolongation)
            if P.shape != (mesh.n_nodes, mesh.n_nodes):
                raise ValueError("prolongation must have shape (n_nodes, n_nodes)")
            hanging = np.asarray(abs(P).sum(axis=0)).ravel() == 0
            ortho_e = ~hanging[mesh.edges].any(1)
            defect = float(mesh.orthogonality_defect[ortho_e].max(initial=0.0))
        else:
            P = None
            hanging = np.zeros(mesh.n_nodes, bool)
            defect = mesh.orthogonality_defect.max()
        if hodge == "auto":
            hodge = "orthogonal" if defect < ortho_tol else "mfd"
        if hodge == "orthogonal" and defect >= ortho_tol:
            raise ValueError(
                f"mesh is not orthogonal (defect {defect:.2e}); use hodge='mfd'"
            )
        if hodge not in ("orthogonal", "mfd"):
            raise ValueError("hodge must be 'auto', 'orthogonal' or 'mfd'")
        self.mesh, self.pol, self.bc, self.hodge = mesh, polarization, bc, hodge
        eps_r, mu_r, sigma, pec, dispersive = cell_materials(mesh, materials)
        self.eps_r, self.mu_r, self.sigma, self.pec_cells = eps_r, mu_r, sigma, pec
        self.dispersive = dispersive
        if polarization == "TE" and np.any(sigma > 0):
            raise NotImplementedError("conductivity in TE: use a Drude pole")
        live = ~pec
        kind = "orthogonal" if hodge == "orthogonal" else "median"
        Pn = mesh.dual_area_partial(kind)
        self._Pn = Pn
        L = mesh.edge_length

        fixed = np.zeros(mesh.n_nodes, bool)
        used = np.zeros(mesh.n_nodes, bool)
        cn, cc = mesh.cell_nodes, mesh.corner_cell
        used[cn[~pec[cc]]] = True
        if polarization == "TM":
            fixed[cn[pec[cc]]] = True
            if bc == "pec":
                fixed[mesh.boundary_nodes] = True
        fixed |= ~used
        fixed |= hanging
        self._hanging = hanging
        self.free = np.flatnonzero(~fixed)
        if P is None:
            self.Pf = None
            G = mesh.G[:, self.free].tocsr()
        else:
            self.Pf = P[:, self.free].tocsr()
            G = (mesh.G @ self.Pf).tocsr()
        self.G, self.GT = G, G.T.tocsr()

        # Edge Hodge weights per cell
        if polarization == "TM":
            w_cell = live / (mu_0 * mu_r)
            mass = eps_0 * (Pn @ (eps_r * live))
            loss = Pn @ (sigma * live)
        else:
            w_cell = live / (eps_0 * eps_r)
            mass = mu_0 * (Pn @ (mu_r * live))
            loss = np.zeros(mesh.n_nodes)
        disp_cells = np.zeros(mesh.n_cells, bool)
        for _, sel in dispersive:
            disp_cells |= sel
        if polarization == "TE" and dispersive:
            if hodge != "orthogonal":
                raise NotImplementedError("dispersive TE needs hodge='orthogonal'")
            # the stiffness of K (time step, eigenmodes) keeps the eps_inf of
            # the dispersive cells; the stepping Hodge drops them (their
            # segments carry their own field, see _te_voltage)
            self.S, self.stiff = self._edge_hodge(w_cell)
            K_S = self._apply_S_matrix()
            w_cell = w_cell * ~disp_cells
        else:
            K_S = None
        self._w_cell = w_cell
        self.S, self.stiff = self._edge_hodge(w_cell)
        if K_S is None:
            K_S = self._apply_S_matrix()
        if bc == "abc":
            bl = mesh.boundary_node_length(kind)
            ce = mesh.edge_cells[mesh.boundary_edges, 0]
            eta_b = ETA_0 * np.sqrt(mu_r[ce] / eps_r[ce])
            eta_node = np.full(mesh.n_nodes, ETA_0)
            eta_node[mesh.edges[mesh.boundary_edges, 0]] = eta_b
            eta_node[mesh.edges[mesh.boundary_edges, 1]] = eta_b
            loss = loss + (bl / eta_node if polarization == "TM" else bl * eta_node)
        self.mass = self._lump(mass)
        if np.any(self.mass <= 0):
            raise ValueError(f"{np.sum(self.mass <= 0)} non-positive node masses")
        self.loss = self._lump(loss)

        # PML
        self.edge_rate = np.zeros(mesh.n_edges)
        self._pml = None
        if pml is not None:
            self._init_pml(pml)

        K = (self.GT @ K_S @ G).tocsr()
        self.K = K
        lam = max_eigenvalue(K, M_inv_sqrt=1 / np.sqrt(self.mass))
        self.dt_max = 2 / np.sqrt(lam)
        self.dt = cfl * self.dt_max if dt is None else float(dt)
        self.courant = self.dt * np.sqrt(lam) / 2
        self.lts = None
        self.lts_info = None
        self._G_F = None
        if lts is not None:
            self._setup_lts(lts, lam)
        if dispersive:
            check_dispersive_stability(
                [m for m, _ in dispersive], self.dt, self.courant
            )

        groups = []
        if polarization == "TM":
            for mat, sel in dispersive:
                c = self._lump(eps_0 * (Pn @ (sel & live).astype(float)))
                idx = np.flatnonzero(c > 0)
                groups.append((mat, idx, c[idx]))
            self._update = CapacitiveUpdate(self.mass, self.loss, groups, self.dt)
            self._te_groups = []
        else:
            self._init_te_dispersive(live)
        if self._pml is not None:
            p = self._pml
            m = self.mass[p["idx"]]
            for ax in ("x", "y"):
                s = p["s" + ax]
                p["a" + ax] = m * (1 - 0.5 * self.dt * s)
                p["b" + ax] = m * (1 + 0.5 * self.dt * s)
        self._ce = 0.5 * self.dt * self.edge_rate
        self._lossy_edges = np.flatnonzero(self.edge_rate > 0)
        self.sources = []
        self.plane_wave = None
        self.recorder = Recorder()
        self.reset()

    def _lump(self, x):
        return x[self.free] if self.Pf is None else np.asarray(self.Pf.T @ x).ravel()

    # ------------------------------------------------------------ Hodge
    def _edge_hodge(self, w_cell):
        """Return (S matrix or None, diagonal stiffness or None)."""
        mesh = self.mesh
        if self.hodge == "orthogonal":
            stiff = (
                np.bincount(
                    mesh.corner_edge,
                    mesh.corner_h * w_cell[mesh.corner_cell],
                    minlength=mesh.n_edges,
                )
                / mesh.edge_length
            )
            return None, stiff
        rows, cols, vals = [], [], []
        for i, M in enumerate(mesh.mfd_local_matrices()):
            if w_cell[i] == 0:
                continue
            k = slice(mesh.cell_ptr[i], mesh.cell_ptr[i + 1])
            e, sg = mesh.corner_edge[k], mesh.corner_sign[k]
            Mg = (sg[:, None] * M * sg[None, :]) * w_cell[i]
            rows.append(np.repeat(e, len(e)))
            cols.append(np.tile(e, len(e)))
            vals.append(Mg.ravel())
        ne = mesh.n_edges
        S = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(ne, ne),
        )
        return S, None

    def _apply_S_matrix(self):
        return sp.diags(self.stiff) if self.S is None else self.S

    def _apply_S(self, v):
        return self.stiff * v if self.S is None else self.S @ v

    # ------------------------------------------------------------ PML
    def _init_pml(self, pml):
        mesh = self.mesh
        d = pml["thickness"]
        order = pml.get("order", 3)
        R0 = pml.get("R0", 1e-6)
        er = pml.get("eps_r", 1.0)
        rx = pml_rate(pml["x"], d, pml.get("s_max"), order, R0, er)
        ry = pml_rate(pml["y"], d, pml.get("s_max"), order, R0, er)
        t = mesh.edge_tangent
        is_x = np.abs(t[:, 1]) < 1e-9
        is_y = np.abs(t[:, 0]) < 1e-9
        mid = mesh.edge_midpoint
        rate = np.where(is_x, rx(mid[:, 0]), 0.0) + np.where(is_y, ry(mid[:, 1]), 0.0)
        node_x = rx(mesh.nodes[self.free, 0])
        node_y = ry(mesh.nodes[self.free, 1])
        lossy = np.flatnonzero((node_x > 0) | (node_y > 0))
        # nodes adjacent to lossy edges also need the split form
        lossy_e = np.flatnonzero(rate > 0)
        if self.Pf is not None:
            h_nodes = np.flatnonzero(self._hanging)
            if h_nodes.size and (
                np.any(rx(mesh.nodes[h_nodes, 0]) > 0)
                or np.any(ry(mesh.nodes[h_nodes, 1]) > 0)
                or np.any(self._hanging[mesh.edges[lossy_e].ravel()])
            ):
                raise ValueError(
                    "PML: hanging nodes cannot lie in or touch the PML layer"
                )
        adj = np.zeros(self.free.size, bool)
        adj[lossy] = True
        adj[np.unique(self.G[lossy_e].indices)] = True
        idx = np.flatnonzero(adj)
        sub = self.GT[idx]  # rows: PML nodes, cols: edges
        if np.any(~(is_x | is_y)[np.unique(sub.indices)]):
            raise ValueError("PML nodes must only touch axis-aligned edges")
        self.edge_rate = rate
        Dx = sp.diags(is_x.astype(float))
        Dy = sp.diags(is_y.astype(float))
        self._pml = dict(
            idx=idx,
            GxT=(sub @ Dx).tocsr(),
            GyT=(sub @ Dy).tocsr(),
            sx=node_x[idx],
            sy=node_y[idx],
        )

    # ------------------------------------------------------------ LTS
    def _setup_lts(self, opts, lam):
        """Partition the nodes and build the stabilized two-level LTS."""
        from .lts import build_lts, gershgorin_rate

        opts = {"p": opts} if np.isscalar(opts) else dict(opts)
        if any(m.debye_poles for m, _ in self.dispersive):
            raise NotImplementedError("LTS with Debye poles")
        p = int(opts.pop("p", 2))
        dt_c = opts.pop("dt", None) or p * self.dt
        if opts.get("fine") is None and (
            self._pml is not None or np.any(self.loss != 0)
        ):
            rate = gershgorin_rate(self.K, self.mass)
            rate = rate * (lam / rate.max())
            theta = opts.get("theta", 1.0)
            if self._pml is not None and np.any(
                rate[self._pml["idx"]] * dt_c**2 > 4 * theta
            ):
                raise ValueError("LTS: fine nodes inside or next to the PML layer")
            if np.any(rate[self.loss != 0] * dt_c**2 > 4 * theta):
                raise ValueError(
                    "LTS: fine nodes or buffer cannot have loss or touch lossy edges"
                )
        lts, info = build_lts(self.K, self.mass, dt_c, p=p, lam=lam, **opts)
        near = np.r_[lts.F, lts.C]
        G_F = self.G[:, lts.F].tocsr()
        edges_near = np.unique(self.G[:, near].nonzero()[0])
        if np.any(self.loss[near] != 0) or np.any(self.edge_rate[edges_near] != 0):
            raise ValueError(
                "LTS: fine nodes or buffer cannot have loss or touch lossy edges"
            )
        if self._pml is not None and np.isin(near, self._pml["idx"]).any():
            raise ValueError("LTS: fine nodes inside or next to the PML layer")
        self.lts, self.lts_info, self._G_F = lts, info, G_F
        self.dt = info["dt"]
        self.dt_fine = self.dt / p
        self.courant = info["rho"]

    def _grad(self, u):
        """``G r(X) u``: the flux-update gradient (``G u`` without LTS)."""
        g = self.G @ u
        if self.lts is not None:
            g = g + self._G_F @ self.lts.correction(u)
        return g

    def lts_work_per_step(self):
        """Unknown-weighted cost of one coarse step (equals `n_unknowns` without LTS)."""
        if self.lts is None:
            return self.n_unknowns
        n_ef = int(np.unique(self._G_F.nonzero()[0]).size)
        return int(self.n_unknowns + (self.lts.p - 1) * (self.lts.F.size + n_ef))

    # ------------------------------------------------------------ TE dispersive
    def _init_te_dispersive(self, live):
        mesh = self.mesh
        self._te_groups = []
        e_all, C_all, h_all = mesh.corner_edge, mesh.corner_cell, mesh.corner_h
        for mat, sel in self.dispersive:
            keep = sel[C_all] & live[C_all]
            e, h = e_all[keep], h_all[keep]
            lorentz, debye = _ade_coefficients(mat, self.dt)
            beta = sum(b for _, b in debye)
            z = np.zeros(e.size)
            self._te_groups.append(
                {
                    "e": e,
                    "h": h,
                    "L": mesh.edge_length[e],
                    "eps_inf": mat.eps_inf,
                    "beta": beta,
                    "E": z.copy(),
                    "lorentz": [(*cf, z.copy(), z.copy()) for cf in lorentz],
                    "debye": [(*cf, z.copy()) for cf in debye],
                }
            )

    def _te_voltage(self, d, t=None):
        v = self._apply_S(d)
        for g in self._te_groups:
            E = g["E"]
            if "inc" in g and self.plane_wave is not None:
                E = E - g["inc"] * self._incident_signal(g["pos"], t)
            v += np.bincount(g["e"], g["h"] * E, minlength=v.size)
        return v

    def _te_ade_step(self, d, t=None):
        for g in self._te_groups:
            En = g["E"]
            Dn = d[g["e"]] / (g["L"] * eps_0)
            if "inc" in g and self.plane_wave is not None:
                Dn = Dn + g["eps_b"] * g["inc"] * self._incident_signal(g["pos"], t)
            psum = np.zeros(En.size)
            for k, (alpha, xi, zeta, p, p_old) in enumerate(g["lorentz"]):
                p_new = alpha * p + xi * p_old + zeta * En
                psum += p_new
                g["lorentz"][k] = (alpha, xi, zeta, p_new, p)
            for kappa, beta, p in g["debye"]:
                psum += kappa * p + beta * En
            E_new = (Dn - psum) / (g["eps_inf"] + g["beta"])
            for k, (kappa, beta, p) in enumerate(g["debye"]):
                g["debye"][k] = (kappa, beta, kappa * p + beta * (E_new + En))
            g["E"] = E_new

    # ------------------------------------------------------------ state
    def reset(self):
        """Zero all fields, auxiliary variables, probes and DFTs."""
        self.u = np.zeros(self.free.size)
        self.v = np.zeros(self.mesh.n_edges)
        self.n = 0
        if getattr(self, "plane_wave", None) is not None:
            self.plane_wave["cache_n"] = None
            self._pw_cache = None
        if self.pol == "TM":
            self._update.reset()
        for g in self._te_groups:
            g["E"][:] = 0
            g["lorentz"] = [(a, x, z, 0 * p, 0 * q) for a, x, z, p, q in g["lorentz"]]
            g["debye"] = [(k, b, 0 * p) for k, b, p in g["debye"]]
        if self._pml is not None:
            n = self._pml["idx"].size
            self._pml["ux"] = np.zeros(n)
            self._pml["uy"] = np.zeros(n)
        for rec in self.recorder.probes:
            rec["samples"] = []
        for rec in self.recorder.dfts:
            rec["data"][:] = 0

    @property
    def node_field(self):
        """Ez (TM) or Hz (TE) on all mesh nodes (0 on fixed nodes)."""
        if self.Pf is not None:
            return np.asarray(self.Pf @ self.u).ravel()
        out = np.zeros(self.mesh.n_nodes)
        out[self.free] = self.u
        return out

    @property
    def n_unknowns(self):
        """Field unknowns: free nodes + edges."""
        return int(self.free.size + self.mesh.n_edges)

    def energy(self):
        """Discrete energy per unit length [J/m] (non-dispersive, staggered)."""
        return 0.5 * self.mass @ self.u**2 + 0.5 * self.v @ self._apply_S(self.v)

    # ------------------------------------------------------------ sources
    def _interp(self, points):
        M = self.mesh.interpolation_matrix(points)
        if self.Pf is not None:
            return (M @ self.Pf).tocsr()
        return M[:, self.free].tocsr()

    def add_point_source(self, position, amp_func):
        """
        Soft line source at a point: electric line current I_z(t) [A] (TM) or
        magnetic line current K_z(t) [V] (TE), distributed with mean-value
        weights.
        """
        w = np.asarray(self._interp([position]).todense()).ravel()
        self.sources.append(("node", w, amp_func))

    def add_plane_wave(
        self, src_time, direction=(1.0, 0.0), amplitude=1.0, background_eps_r=1.0,
        table_sub=8,
    ):  # fmt: skip
        """
        Plane wave in the scattered-field formulation (as
        `UnstructuredFIT2D.add_plane_wave`).

        The solver then advances the scattered field. The incident wave
        E_inc = amplitude s(t - (k.r - s_0)/v) p (p = z for TM, z x k for TE)
        enters only where eps differs from the background:

        * TM: node current i_n = (M_eps - eps_0 eps_b A_n) dE_inc/dt;
        * TE: E_s = D_s / (eps_0 eps) - (1 - eps_b / eps) E_inc on every
          dual-edge segment of a scatterer cell C, i.e. a voltage
          -(1 - eps_b / eps_C) g_{e,C} . E_inc(midpoint), with g_{e,C} the
          segment vector oriented along n_e: c_C -> foot point (orthogonal
          Hodge) or centroid -> edge midpoint (mimetic Hodge, the vector that
          makes the local matrix exact for uniform fields, M_C N = R).

        Dispersive (and, in TM, conducting) scatterers: the auxiliary
        equations must see the *total* field. In TM the nodes touching such
        cells are advanced in total-field form,
        u_tot = u_s + E_inc, with the background current
        eps_0 eps_b A_n dE_inc/dt (then u_s = u_tot - E_inc); in TE each
        dispersive dual-edge segment carries the total normal field, driven
        by D_n = d_e / L_e + eps_0 eps_b E_inc . n_e, and contributes
        h_{e,C} (E_{e,C} - E_inc . n_e) to the scattered voltage.

        Non-magnetic, non-PEC scatterers in a lossless background that
        fills the outer boundary cells.

        ``table_sub``: the non-dispersive source terms use the separable-delay
        table of `UnstructuredFIT2D._build_pw_table` (s(t) sampled at
        dt / table_sub, linear interpolation in the delay, error
        ~ (omega dt / table_sub)^2 / 8), one sparse mat-vec per step instead
        of evaluating s(t) on every target; None: direct evaluation.
        """
        mesh = self.mesh
        eps_b = float(background_eps_r)
        if np.any(self.pec_cells):
            raise NotImplementedError("plane wave: PEC scatterers are not supported")
        if np.any(self.mu_r != 1):
            raise NotImplementedError("plane wave: non-magnetic media only")
        if not np.allclose(self.eps_r[mesh.edge_cells[mesh.boundary_edges, 0]], eps_b):
            raise ValueError("the outer boundary cells must hold the background")
        k = np.asarray(direction, float)
        k = k / np.linalg.norm(k)
        disp = np.zeros(mesh.n_cells, bool)
        for _, sel_d in self.dispersive:
            disp |= sel_d
        scat = ~np.isclose(self.eps_r, eps_b) | disp | (self.sigma > 0)
        sel = scat[mesh.corner_cell]
        s0 = (
            float((mesh.nodes[mesh.corner_node[sel]] @ k).min()) if np.any(sel) else 0.0
        )
        pw = dict(
            s=src_time, k=k, amp=float(amplitude), eps_b=eps_b, s0=s0,
            v=c_0 / np.sqrt(eps_b), eta=ETA_0 / np.sqrt(eps_b),
        )  # fmt: skip
        if self.pol == "TM":
            area_f = self._lump(self._Pn @ (~self.pec_cells).astype(float))
            if self.dispersive or np.any(self.sigma > 0):
                touch = self._lump(self._Pn @ scat.astype(float)) > 0
                idx = np.flatnonzero(touch)
                pw.update(
                    total=True, idx=idx, w=eps_0 * eps_b * area_f[idx],
                    pos=mesh.nodes[self.free[idx]] @ k,
                )  # fmt: skip
            else:
                w = self.mass - eps_0 * eps_b * area_f
                idx = np.flatnonzero(np.abs(w) > 1e-12 * np.abs(self.mass).max())
                pw.update(
                    total=False, idx=idx, w=w[idx], pos=mesh.nodes[self.free[idx]] @ k
                )
        else:
            p_vec = np.array([-k[1], k[0]])
            for g_, (mat, sel_d) in zip(self._te_groups, self.dispersive):
                n_e = mesh.edge_normal[g_["e"]]
                g_["inc"] = n_e @ p_vec
                # segment midpoints: (dual point + foot) / 2 of each corner
                keep = sel_d[mesh.corner_cell] & ~self.pec_cells[mesh.corner_cell]
                C = mesh.corner_cell[keep]
                ee = mesh.corner_edge[keep]
                X0 = mesh.nodes[mesh.edges[ee, 0]]
                foot = X0 + mesh.corner_s[keep][:, None] * mesh.edge_tangent[ee]
                g_["pos"] = (0.5 * (foot + mesh.dual_point[C])) @ k
                g_["eps_b"] = eps_b
            sel = sel & ~disp[mesh.corner_cell]
            e, C, sg = (
                mesh.corner_edge[sel],
                mesh.corner_cell[sel],
                mesh.corner_sign[sel],
            )
            if self.hodge == "orthogonal":
                c = mesh.dual_point[C]
                X0 = mesh.nodes[mesh.edges[e, 0]]
                foot = X0 + mesh.corner_s[sel][:, None] * mesh.edge_tangent[e]
            else:
                c = mesh.cell_centroid[C]
                foot = mesh.edge_midpoint[e]
            g = sg[:, None] * (foot - c)
            p = np.array([-k[1], k[0]])
            coef = (1 - eps_b / self.eps_r[C]) * (g @ p)
            keep = coef != 0
            mid = 0.5 * (foot + c)
            pw.update(e=e[keep], coef=coef[keep], pos=(mid @ k)[keep])
        self.plane_wave = pw
        self._pw_W = None
        if table_sub and not pw.get("total", False):
            self._build_pw_table(int(table_sub))

    def _build_pw_table(self, sub):
        """Separable-delay table (as `UnstructuredFIT2D._build_pw_table`)."""
        pw = self.plane_wave
        tau = np.maximum((pw["pos"] - pw["s0"]) / pw["v"] / self.dt * sub, 0.0)
        j = np.floor(tau).astype(int)
        f = tau - j
        if self.pol == "TM":
            rows, wgt, n_rows = pw["idx"], pw["w"] / self.dt, self.free.size
        else:
            rows, wgt, n_rows = pw["e"], pw["coef"], self.mesh.n_edges
        r2 = np.concatenate([rows, rows])
        c2 = np.concatenate([j, j + 1])
        v2 = pw["amp"] * np.concatenate([wgt * (1 - f), wgt * f])
        self._pw_W = sp.csr_matrix(
            (v2, (r2, c2)), shape=(n_rows, int(c2.max(initial=0)) + 1)
        )
        self._pw_sub = sub
        self._pw_S = np.zeros(0)
        self._pw_cache = None

    def _pw_values(self, n):
        """W @ [S[sub n - j]]_j: the weighted source term at step level n."""
        pw, sub = self.plane_wave, self._pw_sub
        nj = self._pw_W.shape[1]
        top = sub * n
        if self._pw_S.size <= top:
            m = np.arange(max(2 * self._pw_S.size, top + 1 + 4096))
            self._pw_S = np.asarray(pw["s"](m * self.dt / sub), dtype=float)
        idx = top - np.arange(nj)
        g = np.where(idx >= 0, self._pw_S[np.maximum(idx, 0)], 0.0)
        return self._pw_W @ g

    def _incident_signal(self, pos, t):
        pw = self.plane_wave
        return pw["amp"] * np.asarray(pw["s"](t - (pos - pw["s0"]) / pw["v"]))

    def incident_field(self, points, t):
        """Incident Ez (TM) [V/m] or Hz (TE) [A/m] at `points` and time(s) t."""
        pw = self.plane_wave
        e = self._incident_signal(np.atleast_2d(points) @ pw["k"], t)
        return e if self.pol == "TM" else e / pw["eta"]

    # ------------------------------------------------------------ monitors
    def add_probe(self, points):
        """Record Ez (TM) / Hz (TE) at `points` every step."""
        return self.recorder.add_probe(self._interp(points))

    def add_dft(self, freqs):
        """Running DFT of the node field at `freqs` [Hz]."""
        return self.recorder.add_dft(freqs, self.free.size)

    def probe_signal(self, probe):
        return np.asarray(probe["samples"])

    # ------------------------------------------------------------ stepping
    def _edge_update(self, rhs):
        """v += dt * rhs with the (semi-implicit) PML edge loss."""
        if self._lossy_edges.size == 0:
            self.v += self.dt * rhs
            return
        ce = self._ce
        self.v = ((1 - ce) * self.v + self.dt * rhs) / (1 + ce)

    def _pml_nodes(self, r, sign):
        """Split-field node update of PML nodes; r = G^T(...) (curl term)."""
        p = self._pml
        rx = p["GxT"] @ r
        ry = p["GyT"] @ r
        dt = self.dt
        p["ux"] = (p["ax"] * p["ux"] + sign * dt * rx) / p["bx"]
        p["uy"] = (p["ay"] * p["uy"] + sign * dt * ry) / p["by"]
        self.u[p["idx"]] = p["ux"] + p["uy"]

    def one_step(self):
        """Advance one leapfrog step (sources at the half step)."""
        dt = self.dt
        node_src = None
        if self.sources:
            node_src = np.zeros(self.free.size)
            t_half = (self.n + 0.5) * dt
            for _, w, f in self.sources:
                node_src += w * f(t_half)
        pw = self.plane_wave
        if self.pol == "TM":
            self._edge_update(-self._grad(self.u))
            sv = self._apply_S(self.v)
            r = self.GT @ sv
            if pw is not None and self._pw_W is not None:
                v_new = self._pw_values(self.n + 1)
                c = self._pw_cache
                v_old = (
                    c[1]
                    if c is not None and c[0] == self.n
                    else self._pw_values(self.n)
                )
                self._pw_cache = (self.n + 1, v_new)
                r -= v_new - v_old
            elif pw is not None:
                # cache E_inc^{n+1} for the next step (one evaluation per step)
                e_new = self._incident_signal(pw["pos"], (self.n + 1) * dt)
                if pw.get("cache_n") != self.n:
                    pw["cache"] = self._incident_signal(pw["pos"], self.n * dt)
                e_old = pw["cache"]
                pw["cache"], pw["cache_n"] = e_new, self.n + 1
                if pw["total"]:
                    r[pw["idx"]] += pw["w"] * (e_new - e_old) / dt
                else:
                    r[pw["idx"]] -= pw["w"] * (e_new - e_old) / dt
            if node_src is not None:
                r -= node_src
            if pw is not None and pw["total"] and self._pw_W is None:
                u = self.u.copy()
                u[pw["idx"]] += e_old
                self.u = self._update.step(u, r)
                self.u[pw["idx"]] -= e_new
            else:
                self.u = self._update.step(self.u, r)
            if self._pml is not None:
                self._pml_nodes(sv, +1.0)
        else:
            vol = self._te_voltage(self.v, self.n * dt)
            if pw is not None and self._pw_W is not None:
                vol -= self._pw_values(self.n)
            elif pw is not None:
                einc = self._incident_signal(pw["pos"], self.n * dt)
                vol -= np.bincount(pw["e"], pw["coef"] * einc, minlength=vol.size)
            r = self.GT @ vol
            if node_src is not None:
                r += node_src
            a = self.mass - self.loss * dt / 2
            b = self.mass + self.loss * dt / 2
            self.u = (a * self.u - dt * r) / b
            if self._pml is not None:
                self._pml_nodes(vol, -1.0)
            self._edge_update(self._grad(self.u))
            self._te_ade_step(self.v, (self.n + 1) * dt)
        self.n += 1
        self.recorder.record(self.u, self.n * dt, dt)

    def run(self, n_steps):
        for _ in range(int(n_steps)):
            self.one_step()

    # ------------------------------------------------------------ eigenmodes
    def eigenmodes(self, n_modes=6, f_target=0.0):
        """
        Resonant frequencies [Hz] of the closed lossless structure
        (K u = omega^2 M u, shift-invert Lanczos around `f_target`).
        """
        M = sp.diags(self.mass)
        sigma = (2 * np.pi * f_target) ** 2
        if sigma == 0:
            sigma = -1e-3 * (self.K.diagonal().mean() / self.mass.mean())
        k = n_modes + (1 if self.pol == "TE" else 0)
        w2, vec = eigsh(self.K, k=k, M=M, sigma=sigma, which="LM")
        o = np.argsort(w2)
        w2, vec = w2[o], vec[:, o]
        keep = np.abs(w2) > 1e-8 * np.abs(w2).max()
        w2, vec = w2[keep][:n_modes], vec[:, keep][:, :n_modes]
        return np.sqrt(np.abs(w2)) / (2 * np.pi), vec
